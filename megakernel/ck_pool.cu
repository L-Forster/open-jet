// MKEngine::Impl: the checkpoint pool (DeltaNet state + conv / h rings at a position, saved and restored around
// prefills; pins) and the staging of the checkpoints a prefill saves on its way.

#include "engine_impl.h"

void MKEngine::Impl::ck_alloc() {
    const char * cd = getenv("MK_CK_DEVICE");
    ck_dev = cd && cd[0] == '1';
    int nslots = 8;
    if (const char * e = getenv("MK_CK_SLOTS")) nslots = std::max(1, std::min(64, atoi(e)));
    const size_t nf = CK_NS + CK_NC + CK_NH;
    for (int i = 0; i < (ck_dev ? 1 : nslots); i++) {
        CkSlot s;
        if (!ck_dev && cudaHostAlloc(&s.mem, nf * 4, cudaHostAllocMapped) != cudaSuccess) {
            cudaGetLastError();
            s.mem = nullptr;
            if (i > 0) {
                fprintf(stderr, "[mk] warning: no pinned host memory for DeltaNet checkpoint %d: keeping %d\n", i + 1, i);
                break;
            }
            fprintf(stderr, "[mk] warning: no pinned host memory for DeltaNet checkpoints: one in device memory instead (%.0f MiB)\n", nf * 4 / 1048576.0);
            ck_dev = true;
        }
        if (ck_dev) s.mem = dalloc<float>(nf, false);
        s.dstate = static_cast<float *>(s.mem);
        s.conv = s.dstate + CK_NS;
        s.hhist = s.conv + CK_NC;
        if (ck_dev) s.hdev = s.hhist;
        else CK(cudaHostGetDevicePointer(reinterpret_cast<void **>(&s.hdev), s.hhist, 0));
        CK(cudaEventCreateWithFlags(&s.ev, cudaEventDisableTiming));
        ck.push_back(s);
    }
    pin_max = std::min(2, (int) ck.size() / 4);
    CK(cudaStreamCreateWithFlags(&ck_st, cudaStreamNonBlocking));
    CK(cudaStreamCreateWithFlags(&ck_sv, cudaStreamNonBlocking));
    for (cudaEvent_t * e : {&ck_ev_src, &ck_ev_sv, &ck_ev_scr}) CK(cudaEventCreateWithFlags(e, cudaEventDisableTiming));
    for (auto & e : ck_ev_grp) CK(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
    for (auto & e : stg_ev) CK(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
    // MK_EXACT (see `exact`): with MK_KV_WIN rings, each checkpoint also holds the rings as of its position
    if (const char * e = getenv("MK_EXACT")) exact = atoi(e) != 0;
    if (exact && hb.kvwin > 0) {
        if (ck_dev) throw std::runtime_error("MK_EXACT with MK_KV_WIN needs host checkpoints (MK_CK_DEVICE=0)");
        for (int s = 0; s < N_ATT_SLOTS; s++)
            for (int w = 0; w < 2; w++)
                if (hb.kvrp[w][s]) {
                    rsides[s].push_back({hb.kvrp[w][s], hb.kvrs[w][s], ring_bytes_ck});
                    ring_bytes_ck += ring_vb() + ring_sb();
                }
        for (auto & s : ck)
            if (cudaHostAlloc(reinterpret_cast<void **>(&s.ring), ring_bytes_ck, cudaHostAllocDefault) != cudaSuccess) {
                cudaGetLastError();
                s.ring = nullptr;
                fprintf(stderr, "[mk] warning: no pinned host memory for a checkpoint's KV window rings: its restores refill them (not exact)\n");
            }
        for (int L = 0, d = 0; L < NL; L++) { // slot L / 4 (pf_run): restore group of the DeltaNet layer before it
            if (!hw.L[L].attn) {
                d++;
                continue;
            }
            int g = 0;
            while (g + 1 < CK_NG && ck_gb[g + 1] <= d - 1) g++;
            ring_grp[L / 4] = g;
        }
        ring_grp[N_ATT_SLOTS - 1] = CK_NG - 1; // MTP block: after the trunk
        for (auto & e : rs_ev) CK(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
        for (cudaEvent_t * e : {&rs_ev_pf, &rs_ev_done}) CK(cudaEventCreateWithFlags(e, cudaEventDisableTiming));
        fprintf(stderr, "[mk] MK_EXACT: checkpoints keep the KV window rings (%.1f MiB each, pinned host)\n", ring_bytes_ck / 1048576.0);
    }
    if (exact) fprintf(stderr, "[mk] MK_EXACT=1: prefill rows independent of the chunking (attention unsplit, per-tile window, grid checkpoints)\n");
    mem_mark(ck_dev ? "DeltaNet checkpoint" : "DeltaNet checkpoints (host)");
}

void MKEngine::Impl::ck_free() {
    if (ck_st) cudaStreamSynchronize(ck_st);
    if (ck_sv) cudaStreamSynchronize(ck_sv);
    for (auto & s : ck) {
        if (s.ev) cudaEventDestroy(s.ev);
        if (s.mem && !ck_dev) cudaFreeHost(s.mem);
        if (s.ring) cudaFreeHost(s.ring);
    }
    for (cudaEvent_t e : rs_ev)
        if (e) cudaEventDestroy(e);
    for (cudaEvent_t e : {rs_ev_pf, rs_ev_done})
        if (e) cudaEventDestroy(e);
    for (cudaEvent_t e : {ck_ev_src, ck_ev_sv, ck_ev_scr})
        if (e) cudaEventDestroy(e);
    for (cudaEvent_t e : ck_ev_grp)
        if (e) cudaEventDestroy(e);
    for (cudaEvent_t e : stg_ev)
        if (e) cudaEventDestroy(e);
    if (ck_st) cudaStreamDestroy(ck_st);
    if (ck_sv) cudaStreamDestroy(ck_sv);
}

void MKEngine::Impl::ck_make_plan() {
    ck_plan.clear();
    stg_n = (int) std::min<size_t>(STG_MAX, (size_t) hp.cap * D / STG_F);
    if (ck_dev) return; // (device checkpoint: the end save copies on stream 0)
    const size_t cap = hp.cap;
    const std::pair<void *, size_t> scr[] = {
        {hp.y0, y0_floats * 4}, {hp.y1, y1_floats * 4}, {hp.xh2, cap * FF * 2}, {hp.y2, cap * VD * 4}, {hp.x, cap * D * 4},
        {hp.xm, cap * D * 4},  {hp.dq, cap * QKV * 2}, {hp.xh, cap * 2 * D * 2}, {hp.qh, cap * NH * HD * 2}};
    const std::tuple<void *, size_t, size_t> reg[] = { // state, its byte offset in a slot, bytes
        {hb.dstate, 0, CK_NS * 4}, {hb.conv, CK_NS * 4, CK_NC * 4}, {hb.h_hist, (CK_NS + CK_NC) * 4, CK_NH * 4}};
    size_t si = 0, so = 0; // scratch buffer, offset in it
    for (const auto & r : reg) {
        size_t done = 0;
        while (done < std::get<2>(r)) {
            while (si < std::size(scr) && scr[si].second - so < 256) si++, so = 0;
            if (si == std::size(scr)) {
                ck_plan.clear();
                return;
            }
            const size_t n = std::min(std::get<2>(r) - done, (scr[si].second - so) & ~(size_t) 255);
            ck_plan.push_back({static_cast<char *>(std::get<0>(r)) + done, static_cast<char *>(scr[si].first) + so,
                               std::get<1>(r) + done, n});
            done += n;
            so += n;
        }
    }
}

void MKEngine::Impl::ck_scr_guard() {
    if (!ck_scr_busy) return;
    CK(cudaStreamWaitEvent(0, ck_ev_scr, 0));
    ck_scr_busy = false;
}

void MKEngine::Impl::ck_guard_state() {
    if (ck_direct < 0) return;
    CK(cudaStreamWaitEvent(0, ck[ck_direct].ev, 0));
    ck_direct = -1;
}

void MKEngine::Impl::ring_snap_begin(int sn, const int * ss) {
    CK(cudaEventRecord(rs_ev_pf, 0));
    CK(cudaStreamWaitEvent(ck_sv, rs_ev_pf, 0));
    for (int j = 0; j < sn; j++) {
        if (ck[ss[j]].pending) CK(cudaStreamWaitEvent(ck_sv, ck[ss[j]].ev, 0)); // (ck_sv is in order anyway)
        ck[ss[j]].ring_ok = ck[ss[j]].ring != nullptr;
    }
    const bool rs_in = ck_rs_next < CK_NG; // a restore's copies may still be writing the rings
    for (int s = 0; s < N_ATT_SLOTS; s++) {
        if (rs_in && !rsides[s].empty()) CK(cudaStreamWaitEvent(ck_sv, ck_ev_grp[ring_grp[s]], 0));
        for (int j = 0; j < sn; j++) {
            char * dst = ck[ss[j]].ring;
            if (!dst) continue;
            for (const RingSide & r : rsides[s]) {
                CK(cudaMemcpyAsync(dst + r.off, r.q, ring_vb(), cudaMemcpyDeviceToHost, ck_sv));
                CK(cudaMemcpyAsync(dst + r.off + ring_vb(), r.s, ring_sb(), cudaMemcpyDeviceToHost, ck_sv));
            }
        }
        CK(cudaEventRecord(rs_ev[s], ck_sv));
    }
    CK(cudaEventRecord(rs_ev_done, ck_sv));
    rs_busy = true;
}

void MKEngine::Impl::ring_snap_rows(int s, int n, int sn, const int * sp, const int * ss) {
    if (rsides[s].empty()) return;
    CK(cudaEventRecord(rs_ev_pf, 0));
    CK(cudaStreamWaitEvent(ck_sv, rs_ev_pf, 0));
    const size_t rc = hb.kvrc, vrow = HD, srow = (HD / 32) * sizeof(__half);
    for (int j = 0; j < sn; j++) {
        char * dst = ck[ss[j]].ring;
        if (!dst) continue;
        for (int p = n; p < sp[j];) {
            const int r0 = p % (int) rc, c = std::min(sp[j] - p, (int) rc - r0);
            for (const RingSide & r : rsides[s]) {
                CK(cudaMemcpy2DAsync(dst + r.off + r0 * vrow, rc * vrow, r.q + r0 * vrow, rc * vrow, c * vrow, NKV,
                                     cudaMemcpyDeviceToHost, ck_sv));
                CK(cudaMemcpy2DAsync(dst + r.off + ring_vb() + r0 * srow, rc * srow, reinterpret_cast<const char *>(r.s) + r0 * srow,
                                     rc * srow, c * srow, NKV, cudaMemcpyDeviceToHost, ck_sv));
            }
            p += c;
        }
    }
    CK(cudaEventRecord(rs_ev_done, ck_sv));
}

void MKEngine::Impl::ring_snap_guard() {
    if (!rs_busy) return;
    CK(cudaStreamWaitEvent(0, rs_ev_done, 0));
    rs_busy = false;
}

void MKEngine::Impl::ck_issue_restore() {
    if (ck_rs_slot < 0) return;
    const CkSlot & s = ck[ck_rs_slot];
    ck_rs_src = ck_rs_slot;
    ck_rs_slot = -1;
    // stream 0's earlier work on the state (and on the slot's h ring) is complete before ck_st overwrites the
    // state, and so are the slot's save copies and a direct save still reading the state (all normally long done)
    ring_snap_guard(); // (ring copies of an earlier save still reading the rings)
    CK(cudaEventRecord(ck_ev_src, 0));
    CK(cudaStreamWaitEvent(ck_st, ck_ev_src, 0));
    if (s.pending) CK(cudaStreamWaitEvent(ck_st, s.ev, 0));
    if (ck_direct >= 0) {
        CK(cudaStreamWaitEvent(ck_st, ck[ck_direct].ev, 0));
        ck_direct = -1;
    }
    const cudaMemcpyKind k = ck_dev ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice;
    CK(cudaMemcpyAsync(hb.h_hist, s.hhist, CK_NH * 4, k, ck_st)); // group 0
    for (int g = 0; g < CK_NG; g++) {
        const size_t l0 = ck_gb[g], nl = ck_gb[g + 1] - ck_gb[g];
        CK(cudaMemcpyAsync(hb.conv + l0 * CK_CL, s.conv + l0 * CK_CL, nl * CK_CL * 4, k, ck_st));
        CK(cudaMemcpyAsync(hb.dstate + l0 * CK_SL, s.dstate + l0 * CK_SL, nl * CK_SL * 4, k, ck_st));
        if (s.ring_ok) // MK_EXACT: the KV window rings of the attention slots stream 0 reaches after this group
            for (int a = 0; a < N_ATT_SLOTS; a++)
                if (ring_grp[a] == g)
                    for (const RingSide & r : rsides[a]) {
                        CK(cudaMemcpyAsync(r.q, s.ring + r.off, ring_vb(), cudaMemcpyHostToDevice, ck_st));
                        CK(cudaMemcpyAsync(r.s, s.ring + r.off + ring_vb(), ring_sb(), cudaMemcpyHostToDevice, ck_st));
                    }
        CK(cudaEventRecord(ck_ev_grp[g], ck_st));
    }
    ck_rs_next = 0;
}

void MKEngine::Impl::ck_wait_restore(int dl) {
    ck_issue_restore();
    if (ck_rs_next >= CK_NG || (dl >= 0 && dl < ck_gb[ck_rs_next])) return;
    int g = CK_NG - 1;
    if (dl >= 0)
        while (ck_gb[g] > dl) g--;
    CK(cudaStreamWaitEvent(0, ck_ev_grp[g], 0));
    ck_rs_next = g + 1;
    if (ck_rs_next >= CK_NG) ck_rs_src = -1; // (stream 0 waited for all of it)
}

void MKEngine::Impl::ck_save_end(int si) {
    ck_wait_restore();
    CkSlot & s = ck[si];
    const auto t0 = std::chrono::steady_clock::now();
    if (ck_dev) { // device checkpoint: stream-ordered device copies
        CK(cudaMemcpyAsync(s.dstate, hb.dstate, CK_NS * 4, cudaMemcpyDeviceToDevice, 0));
        CK(cudaMemcpyAsync(s.conv, hb.conv, CK_NC * 4, cudaMemcpyDeviceToDevice, 0));
        CK(cudaMemcpyAsync(s.hhist, hb.h_hist, CK_NH * 4, cudaMemcpyDeviceToDevice, 0));
    } else {
        ck_scr_guard(); // a previous end save may still read the scratch (normally long done)
        for (const auto & c : ck_plan) CK(cudaMemcpyAsync(c.scr, c.dev, c.n, cudaMemcpyDeviceToDevice, 0));
        CK(cudaEventRecord(ck_ev_sv, 0));
        CK(cudaStreamWaitEvent(ck_sv, ck_ev_sv, 0));
        char * base = static_cast<char *>(s.mem);
        if (!ck_plan.empty()) {
            for (const auto & c : ck_plan) CK(cudaMemcpyAsync(base + c.off, c.scr, c.n, cudaMemcpyDeviceToHost, ck_sv));
            CK(cudaEventRecord(ck_ev_scr, ck_sv));
            ck_scr_busy = true;
        } else {
            CK(cudaMemcpyAsync(s.dstate, hb.dstate, CK_NS * 4, cudaMemcpyDeviceToHost, ck_sv));
            CK(cudaMemcpyAsync(s.conv, hb.conv, CK_NC * 4, cudaMemcpyDeviceToHost, ck_sv));
            CK(cudaMemcpyAsync(s.hhist, hb.h_hist, CK_NH * 4, cudaMemcpyDeviceToHost, ck_sv));
            ck_direct = si;
        }
        CK(cudaEventRecord(s.ev, ck_sv));
        s.pending = true;
    }
    s.ctl = hctl;
    s.pos = hctl.dn_c;
    s.pin = false;
    s.ring_ok = false; // (no KV window rings: a restore refills them)
    ck_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    if (memlog)
        fprintf(stderr, "[mk-mem] checkpoint save at %d enqueued in %.2f ms (%s)\n", s.pos, ck_ms,
                ck_dev ? "device" : ck_plan.empty() ? "host, direct" : "host, via the prefill scratch");
}

void MKEngine::Impl::ck_restore(int si) {
    ck_wait_restore(); // (a previous restore not yet waited for: normally none)
    const auto t0 = std::chrono::steady_clock::now();
    ck_rs_slot = si; // issued by the next pf_run after its token upload, or by whatever touches the state first
    const CkSlot & s = ck[si];
    hctl = s.ctl;
    hctl.mtp_valid = std::min(hctl.mtp_valid, s.pos);
    CK(cudaMemcpy(hb.ctl, &hctl, sizeof(Ctl), cudaMemcpyHostToDevice));
    seq.resize(s.pos);
    for (auto & o : ck) // later checkpoints belong to the abandoned continuation
        if (o.pos > s.pos) o.pos = -1;
    if (s.ring_ok) kv_top = std::min(kv_top, s.pos); // MK_EXACT: the restore copies the rings as of s.pos
    else ring_rewind(s.pos);
    ck_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    if (memlog) fprintf(stderr, "[mk-mem] checkpoint restore to %d enqueued in %.2f ms\n", s.pos, ck_ms);
}

void MKEngine::Impl::ring_rewind(int P) {
    if (hb.kvwin > 0) {
        const int p0 = std::max(0, P - hb.kvrc), p1 = std::min(P, kv_top - hb.kvrc);
        if (p1 > p0) {
            ring_snap_guard(); // (MK_EXACT: ring copies of a checkpoint save still reading those rows)
            mkl::k_ring_refill(p0, p1);
            CK(cudaGetLastError());
        }
        if (memlog) fprintf(stderr, "[mk-mem] KV window ring: rewind to %d (rows written to %d), refilled %d rows\n", P, kv_top, std::max(0, p1 - p0));
    }
    kv_top = std::min(kv_top, P);
}

int MKEngine::Impl::ck_find(int pos) const {
    for (int i = 0; i < (int) ck.size(); i++)
        if (ck[i].pos == pos && pos >= 0) return i;
    return -1;
}

int MKEngine::Impl::pin_count() const {
    int n = 0;
    for (const auto & k : ck) n += k.pos >= 0 && k.pin;
    return n;
}

int MKEngine::Impl::pin_slot(int P) const {
    int best = -1;
    for (int i = 0; i < (int) ck.size(); i++)
        if (ck[i].pos > P - 32 && ck[i].pos <= P && (best < 0 || ck[i].pos > ck[best].pos)) best = i;
    return best;
}

void MKEngine::Impl::pin_set(int si, unsigned t) {
    if (pin_max <= 0) return;
    ck[si].pin = true;
    ck[si].pin_t = t;
    while (pin_count() > pin_max) {
        int old = -1;
        for (int i = 0; i < (int) ck.size(); i++)
            if (i != si && ck[i].pos >= 0 && ck[i].pin &&
                (old < 0 || ck[i].pin_t < ck[old].pin_t || (ck[i].pin_t == ck[old].pin_t && ck[i].pos < ck[old].pos)))
                old = i;
        if (old < 0) break;
        ck[old].pin = false;
    }
}

std::vector<int> MKEngine::Impl::ck_assign(const std::vector<int> & want, int keep, int E, const std::vector<char> & wpin) {
    struct C { int pos, slot, wi; }; // slot >= 0: existing checkpoint; wi >= 0: new (index into want)
    std::vector<C> c;
    for (int i = 0; i < (int) ck.size(); i++)
        if (ck[i].pos >= 0) c.push_back({ck[i].pos, i, -1});
    for (int i = 0; i < (int) want.size(); i++) c.push_back({want[i], -1, i});
    std::stable_sort(c.begin(), c.end(), [](const C & a, const C & b) { return a.pos < b.pos; });
    while (c.size() > ck.size()) {
        int best = -1;
        double bs = 1e300;
        for (int pass = 0; pass < 2 && best < 0; pass++)
            for (int i = 0; i < (int) c.size(); i++) {
                if (c[i].wi >= 0 ? c[i].pos == keep || (c[i].wi < (int) wpin.size() && wpin[c[i].wi]) : ck[c[i].slot].pin) continue;
                if (!pass && c[i].slot >= 0 && (c[i].slot == ck_rs_src || c[i].slot == ck_rs_slot)) continue;
                const double prev = i ? c[i - 1].pos : 0, next = i + 1 < (int) c.size() ? c[i + 1].pos : E;
                const double sc = (next - prev) / std::max(1.0, E - prev);
                if (sc < bs) bs = sc, best = i;
            }
        if (best < 0) break;
        c.erase(c.begin() + best);
    }
    std::vector<char> used(ck.size(), 0);
    for (const auto & x : c)
        if (x.slot >= 0) used[x.slot] = 1;
    for (int i = 0; i < (int) ck.size(); i++)
        if (!used[i]) ck[i].pos = -1;
    std::vector<int> ord(ck.size()); // free slots in this order
    std::iota(ord.begin(), ord.end(), 0);
    if (sw_on) {
        auto rank = [&](int i) { return !sw_ck_held(i) ? 0 : sw_ck[i].pin ? 2 : 1; };
        std::stable_sort(ord.begin(), ord.end(), [&](int a, int b) { return rank(a) < rank(b); });
    }
    std::vector<int> out(want.size(), -1);
    int f = 0;
    for (const auto & x : c)
        if (x.wi >= 0) {
            while (f < (int) ck.size() && used[ord[f]]) f++;
            if (f == (int) ck.size()) break;
            used[ord[f]] = 1;
            out[x.wi] = ord[f];
        }
    if (sw_on)
        for (int & s : out)
            if (s >= 0 && !sw_ck_take(s)) s = -1;
    return out;
}

int MKEngine::Impl::stg_take() {
    const int s = stg_next;
    stg_next = (stg_next + 1) % stg_n;
    if (stg_busy[s]) {
        CK(cudaStreamWaitEvent(0, stg_ev[s], 0));
        stg_busy[s] = false;
    }
    return s;
}

void MKEngine::Impl::stg_flush(int dl, int sn, const int * sg, const int * ss) {
    CK(cudaEventRecord(ck_ev_sv, 0));
    CK(cudaStreamWaitEvent(ck_sv, ck_ev_sv, 0));
    const cudaMemcpyKind k = ck_dev ? cudaMemcpyDeviceToDevice : cudaMemcpyDeviceToHost;
    for (int j = 0; j < sn; j++) {
        CkSlot & s = ck[ss[j]];
        if (ss[j] == ck_rs_src) CK(cudaStreamWaitEvent(ck_sv, ck_ev_grp[CK_NG - 1], 0)); // (still being restored from)
        CK(cudaMemcpyAsync(s.dstate + (size_t) dl * CK_SL, stg_state(sg[j]), CK_SL * 4, k, ck_sv));
        CK(cudaMemcpyAsync(s.conv + (size_t) dl * CK_CL, stg_conv(sg[j]), CK_CL * 4, k, ck_sv));
        CK(cudaEventRecord(stg_ev[sg[j]], ck_sv));
        stg_busy[sg[j]] = true;
        stg_last = sg[j];
    }
}

void MKEngine::Impl::stg_drain() {
    if (stg_last < 0) return;
    CK(cudaStreamWaitEvent(0, stg_ev[stg_last], 0));
    for (bool & b : stg_busy) b = false;
    stg_last = -1;
}
