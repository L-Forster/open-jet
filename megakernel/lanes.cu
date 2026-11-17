// MKEngine::Impl: lanes, sequences of their own decoded together by batched launches, and the slot API over them.

#include "engine_impl.h"

void MKEngine::Impl::lane_swap(Lane & l) {
    std::swap(ctx, l.ctx);
    std::swap(hb, l.hb);
    std::swap(seq, l.seq);
    std::swap(hctl, l.hctl);
    std::swap(kv_top, l.kv_top);
    std::swap(nd_avail, l.nd_avail);
    std::swap(ndraft, l.ndraft);
    std::swap(last_acc, l.last_acc);
    std::swap(demoted, l.demoted);
    std::swap(dv_n, l.dv_n);
    std::swap(all_acc, l.all_acc);
    std::swap(stale_replay, l.stale_replay);
    std::swap(stale_pcnt, l.stale_pcnt);
    std::swap(smp_temp, l.smp_temp);
    std::swap(smp_top_p, l.smp_top_p);
    std::swap(smp_min_p, l.smp_min_p);
    std::swap(smp_top_k, l.smp_top_k);
    std::swap(smp_seed, l.smp_seed);
    std::swap(pen_pres, l.pen_pres); // (the own sequence's penalties and their counts never reach a lane)
    std::swap(pen_freq, l.pen_freq);
    std::swap(pcnt_dirty, l.pcnt_dirty);
    std::swap(spans, l.spans);
    std::swap(rope_spans, l.rope_spans);
    std::swap(ctl_eg_host, l.ctl_eg_host);
    std::swap(ck, l.ck);
    std::swap(ck_plan, l.ck_plan);
    std::swap(rsides, l.rsides);
}

void MKEngine::Impl::set_cview(int v) {
    if (v == live_view) return;
    CK(mkl::set_cView(&v, sizeof(int)));
    live_view = v;
}

void MKEngine::Impl::lane_in(int k) {
    if (k < 0 || k >= (int) lanes.size()) throw std::runtime_error("no slot " + std::to_string(k));
    if (live == k) return;
    lane_out();
    hs_quiesce();
    lane_swap(lanes[k]);
    live = k;
    set_cview(view_full(k));
}

void MKEngine::Impl::lane_out() {
    if (live >= 0) {
        hs_quiesce();
        lane_swap(lanes[live]);
        live = -1;
    }
    set_cview(0); // (also after a failed copy left the device on a lane's view)
}

void MKEngine::Impl::lane_out_nothrow() noexcept {
    const int k = live;
    for (int i = 0;; i++) {
        std::string why = "unknown exception";
        try {
            lane_out();
            return;
        } catch (const std::exception & e) {
            why = e.what();
        } catch (...) {
        }
        lane_dead(k, why);
        if (i > 0) {
            fprintf(stderr, "[mk] the CUDA error repeats while a slot is put away (sticky): exiting with code 4\n");
            fflush(stdout);
            fflush(stderr);
            _exit(4);
        }
    }
}

void MKEngine::Impl::lane_dead(int k, const std::string & why) {
    if (k < 0 || k >= (int) lanes.size()) return;
    Lane & l = lanes[k];
    l.dead = true;
    l.dead_why = why;
    l.pend.clear();
    l.first = -1;
}

auto MKEngine::Impl::lane_ok(int k) -> Lane & {
    if (k < 0 || k >= (int) lanes.size()) throw std::runtime_error("no slot " + std::to_string(k));
    if (lanes[k].dead)
        throw std::runtime_error("slot " + std::to_string(k) + " failed (" + lanes[k].dead_why +
                                 "): begin a new prompt");
    return lanes[k];
}

void MKEngine::Impl::slots_ok(const std::vector<int> & slots) {
    std::vector<char> seen(lanes.size(), 0);
    for (int s : slots) {
        lane_ok(s);
        if (seen[s]) throw std::runtime_error("slot " + std::to_string(s) + " is listed twice");
        seen[s] = 1;
    }
}

void MKEngine::Impl::ck_slots_free(std::vector<CkSlot> & v) noexcept {
    for (CkSlot & s : v) {
        if (s.ev) cudaEventDestroy(s.ev);
        if (s.mem) cudaFreeHost(s.mem);
        if (s.ring) cudaFreeHost(s.ring);
    }
    v.clear();
}

void MKEngine::Impl::slots_undo(size_t na, size_t ab) noexcept {
    lane_out_nothrow();
    cudaDeviceSynchronize();
    for (Lane & l : lanes) {
        ck_slots_free(l.ck);
        if (l.ctl_eg_host) cudaFreeHost(const_cast<Ctl *>(l.ctl_eg_host));
    }
    lanes.clear();
    for (size_t i = na; i < allocs.size(); i++) use_pool ? cudaFreeAsync(allocs[i], 0) : cudaFree(allocs[i]);
    allocs.resize(na);
    alloc_bytes = ab;
    // trimmed: what the lanes held, and what a failed allocation left reserved (several GB after a large one failed
    // part way, outside the VRAM), go back to the device: an open reusing that remainder at once can fail with out of
    // memory, which the context then keeps reporting.
    pool_trim();
    cudaGetLastError();
}

auto MKEngine::Impl::ck_new_slots(int n) -> std::vector<CkSlot> {
    std::vector<CkSlot> v;
    const size_t nf = CK_NS + CK_NC + CK_NH;
    for (int i = 0; i < n; i++) {
        CkSlot s;
        if (cudaHostAlloc(&s.mem, nf * 4, cudaHostAllocMapped) != cudaSuccess) {
            cudaGetLastError();
            fprintf(stderr, "[mk] warning: no pinned host memory for slot checkpoint %d: keeping %d\n", i + 1, i);
            break;
        }
        s.dstate = static_cast<float *>(s.mem);
        s.conv = s.dstate + CK_NS;
        s.hhist = s.conv + CK_NC;
        try {
            CK(cudaHostGetDevicePointer(reinterpret_cast<void **>(&s.hdev), s.hhist, 0));
            CK(cudaEventCreateWithFlags(&s.ev, cudaEventDisableTiming));
        } catch (...) { // (the slots made so far go with it)
            v.push_back(s);
            ck_slots_free(v);
            throw;
        }
        if (ring_bytes_ck > 0 && cudaHostAlloc(reinterpret_cast<void **>(&s.ring), ring_bytes_ck, cudaHostAllocDefault) != cudaSuccess) {
            cudaGetLastError();
            s.ring = nullptr; // (as ck_alloc's: restores of this checkpoint refill the rings, not exact)
            fprintf(stderr, "[mk] warning: no pinned host memory for a slot checkpoint's KV window rings: its restores refill them (not exact)\n");
        }
        v.push_back(s);
    }
    return v;
}

size_t MKEngine::Impl::lane_kv_bytes(int lctx) const { // (+ dalloc's 16 B per buffer: kc, vc, ksc, vsc, 2 per ring)
    const size_t srows = (size_t) NKV * (lctx + KV_PAD);
    size_t b = 4 * 16;
    for (int w = 0; w < 2; w++)
        for (int sl = 0; sl < kv_slots; sl++) {
            b += srows * ((size_t) HD * hb.kvb[w][sl] / 8 + (HD / 32) * sizeof(__half));
            if (hb.kvrp[w][sl]) b += (size_t) NKV * hb.kvrc * (HD + (HD / 32) * sizeof(__half)) + 2 * 16;
        }
    return b;
}

size_t MKEngine::Impl::lane_bytes(int lctx) const { // (slots_open's device buffers of a lane, + 16 B each)
    return lane_kv_bytes(lctx) + sizeof(float) * ((size_t) 48 * CONV_RING * QKV + (size_t) 48 * KV_RING * KVGB +
           (size_t) 48 * NVH * 128 * 128 + (size_t) NKV * MAX_CH * AR * 256 + (size_t) NKV * MAX_CH * AR * 2 +
           (size_t) H_RING * D + D) + sizeof(float2) * (size_t) MAXD * (2 * TOPK_MAX + 1) + sizeof(Ctl) + 10 * 16;
}

int MKEngine::Impl::slots_open(int n, int lctx) {
    if (!lanes.empty()) throw std::runtime_error("slots are open already");
    if (n < 1 || n > NSEQ) throw std::runtime_error("slots: 1 .. " + std::to_string(NSEQ));
    // (2+ lanes run batched passes of MAXT rows, two lanes always: embed_prologue skips the rows past a smaller grid)
    if (n >= 2 && nsm < mk_min_grid(MAXT))
        throw std::runtime_error("parallel slots: " + std::to_string(n) + " slots need a GPU with at least " +
                                 std::to_string(mk_min_grid(MAXT)) + " SMs (batched passes of " + std::to_string(MAXT) +
                                 " rows); this one has " + std::to_string(nsm) + " (1 slot runs)");
    // a slot's positions: at most the model's context length, also for "the most that fit"; checked before anything is
    // allocated (100,000,000 positions would otherwise build a 26 GB host RoPE table first)
    const int lmax = ctx_limit();
    if (lctx > lmax)
        throw std::runtime_error("parallel slots: " + std::to_string(lctx) + " positions per slot exceed the model's context length (" +
                                 std::to_string(lmax) + ", qwen35.context_length)");
    lane_out();
    // (any MK_KV / MK_KV_WIN preset: every lane gets the engine's per-slot formats and window rings of its own; every
    // device path reads them through the lane's views, the host's per-sequence ring state is in the Lane)
    if (!hb.kvq8) throw std::runtime_error("parallel slots need the q8 KV cache (--kv q8; MK_KV / MK_KV_WIN presets are fine)");
#if MK_KV_EMU
    for (int w = 0; w < 2; w++)
        for (int sl = 0; sl < N_ATT_SLOTS; sl++)
            if (hb.kvemu[w][sl]) throw std::runtime_error("parallel slots: no MK_KV_EMU_SPEC (emulated formats)");
#endif
    for (int L = 0; L < NL; L++)
        if (!hw.L[L].attn && !(hw.L[L].qkv.fm && hw.L[L].gate.fm))
            throw std::runtime_error("parallel slots: this model's delta-net projections are not all in the wv layout (types the batched launch does not serve)");
    if (!spans.empty()) throw std::runtime_error("parallel slots: the engine's own sequence holds images (reset it first)");
    // a failure from here on: slots_undo frees what this call allocated, lanes stay empty (a call while slots are
    // open throws above, before any change, so lanes it did not build are never undone)
    struct Undo {
        Impl * p;
        size_t na, ab;
        bool done;
        ~Undo() {
            if (!done) p->slots_undo(na, ab);
        }
    } undo{this, allocs.size(), alloc_bytes, false};
    par_wide = false;
#if MK_PAR_WIDE
    {
        // the RMAX-row pass (wv.cuh "batched passes wider than MAXT rows") runs every main matrix but ffn_gate / ffn_up in
        // the wv layout, and one item per warp in every weight phase
        std::string why = wide_why();
        if (getenv("MK_PAR_RMAX") && atoi(getenv("MK_PAR_RMAX")) <= MAXT) why = "MK_PAR_RMAX";
        par_wide = why.empty();
        fprintf(stderr, "[mk] parallel slots: batched passes of %s\n", par_wide ? (std::to_string(RMAX) + " rows").c_str()
                                                                           : (std::to_string(MAXT) + " rows (" + why + ")").c_str());
    }
#endif
    CK(cudaDeviceSynchronize());
    // the slots' device bytes at c positions each: the lanes, + the RoPE rows when the engine's table is too short
    auto slots_need = [&](int c) {
        const size_t rope = c + KV_PAD > hb.ctx ? (size_t) (c + KV_PAD) * 32 * sizeof(float2) + 16 : 0;
        return (size_t) n * lane_bytes(c) + rope;
    };
    if (lctx <= 0) {
        size_t free_b = 0, total_b = 0;
        CK(cudaMemGetInfo(&free_b, &total_b));
        const size_t reserve = (size_t) (getenv("MK_SLOT_RESERVE_MIB") ? atol(getenv("MK_SLOT_RESERVE_MIB")) : 1536) << 20;
        const size_t avail = free_b > reserve ? free_b - reserve : 0;
        int lo = 0, hi = lmax;
        while (lo < hi) {
            const int mid = (lo + hi + 1) / 2;
            if (slots_need(mid) <= avail) lo = mid;
            else hi = mid - 1;
        }
        lctx = lo / ATT_BUCKET * ATT_BUCKET;
        if (lctx < 2 * ATT_BUCKET) throw std::runtime_error("parallel slots: not enough free device memory");
    }
    // the slots' device memory against the device's free memory (+ what the allocation pool holds unused), before any of
    // it is allocated: past it a driver with the CUDA sysmem fallback (Windows and WSL2, on by default) does not fail but
    // puts the rest in system memory, charged to Windows commit, and the KV cache is then read over PCIe (4 x 200000
    // positions beside a ctx 8192 engine on a 24 GB GPU: ~26 GB of it, Windows commit down to 0 free, 2026-11-01).
    // MK_SLOT_SPILL=1: no check (the driver decides).
    if (!(getenv("MK_SLOT_SPILL") && atoi(getenv("MK_SLOT_SPILL")) != 0)) {
        const size_t need = slots_need(lctx);
        size_t free_b = 0, total_b = 0;
        CK(cudaMemGetInfo(&free_b, &total_b));
        if (use_pool) {
            cudaMemPool_t pool;
            unsigned long long res = 0, used = 0;
            if (cudaDeviceGetMemPool(&pool, cuda_dev) == cudaSuccess &&
                cudaMemPoolGetAttribute(pool, cudaMemPoolAttrReservedMemCurrent, &res) == cudaSuccess &&
                cudaMemPoolGetAttribute(pool, cudaMemPoolAttrUsedMemCurrent, &used) == cudaSuccess && res > used)
                free_b += res - used;
            cudaGetLastError();
        }
        if (need > free_b) {
            char msg[384];
            snprintf(msg, sizeof msg,
                     "parallel slots: %d x %d positions need %.2f GB of device memory, %.2f GB is free (fewer or shorter slots, or "
                     "0 positions: the most that fit; MK_SLOT_SPILL=1 lets a driver's system-memory fallback take the rest)",
                     n, lctx, need / 1e9, free_b / 1e9);
            throw std::runtime_error(msg);
        }
    }
    // RoPE rows for the lanes (plain positions; the engine's own table when it is long enough)
    float2 * rope = const_cast<float2 *>(hb.rope);
    if (lctx + KV_PAD > hb.ctx) {
        std::vector<float2> rt((size_t) (lctx + KV_PAD) * 32);
        for (int p = 0; p < lctx + KV_PAD; p++)
            for (int i = 0; i < 32; i++) {
                const double th = (double) p * std::pow((double) ROPE_BASE, -2.0 * i / 64.0);
                rt[(size_t) p * 32 + i] = make_float2((float) std::cos(th), (float) std::sin(th));
            }
        rope = dalloc<float2>(rt.size(), false);
        CK(cudaMemcpy(rope, rt.data(), rt.size() * sizeof(float2), cudaMemcpyHostToDevice));
    }
    const int nck = getenv("MK_SLOT_CK") ? std::max(1, atoi(getenv("MK_SLOT_CK"))) : 2;
    lanes.resize(n);
    for (int k = 0; k < n; k++) {
        Lane & l = lanes[k];
        l.ctx = lctx;
        l.hb = hb;
        Bufs & b = l.hb;
        b.ctx = lctx + KV_PAD;
        // the KV cache in the engine's layout (the constructor's): per slot and side its MK_KV bits (kvb, kvr / kvrv rotation
        // as hb's), slots packed in order (all-q8: [slot][kv head][ctx][HD]), fp16 scales [slot][kv head][ctx][HD / 32];
        // a q8 window ring of its own for every side the engine keeps one for (MK_KV_WIN, kvwin / kvrc as hb's)
        const size_t srows = (size_t) NKV * b.ctx;
        size_t vbytes[2] = {0, 0};
        for (int w = 0; w < 2; w++)
            for (int sl = 0; sl < kv_slots; sl++) vbytes[w] += srows * HD * hb.kvb[w][sl] / 8;
        b.kc = dalloc<int8_t>(vbytes[0]);
        b.vc = dalloc<int8_t>(vbytes[1]);
        b.ksc = dalloc<__half>((size_t) kv_slots * srows * (HD / 32));
        b.vsc = dalloc<__half>((size_t) kv_slots * srows * (HD / 32));
        size_t off[2] = {0, 0};
        for (int sl = 0; sl < N_ATT_SLOTS; sl++)
            for (int w = 0; w < 2; w++) {
                b.kvrp[w][sl] = nullptr;
                b.kvrs[w][sl] = nullptr;
                if (sl >= kv_slots) {
                    b.kvp[w][sl] = nullptr;
                    b.kvs[w][sl] = nullptr;
                    continue;
                }
                b.kvp[w][sl] = static_cast<int8_t *>(w ? b.vc : b.kc) + off[w];
                b.kvs[w][sl] = (w ? b.vsc : b.ksc) + (size_t) sl * srows * (HD / 32);
                off[w] += srows * HD * hb.kvb[w][sl] / 8;
                if (hb.kvrp[w][sl]) {
                    b.kvrp[w][sl] = dalloc<int8_t>((size_t) NKV * hb.kvrc * HD);
                    b.kvrs[w][sl] = dalloc<__half>((size_t) NKV * hb.kvrc * (HD / 32));
                }
            }
        for (int sl = 0; sl < N_ATT_SLOTS; sl++) { // (ck_alloc's ring sides, offsets and order: the lane's rings)
            l.rsides[sl].clear();
            for (const RingSide & r : rsides[sl])
                for (int w = 0; w < 2; w++)
                    if (r.q == hb.kvrp[w][sl]) l.rsides[sl].push_back({b.kvrp[w][sl], b.kvrs[w][sl], r.off});
        }
        b.conv = dalloc<float>((size_t) 48 * CONV_RING * QKV);
        b.kvgb = dalloc<float>((size_t) 48 * KV_RING * KVGB);
        b.dstate = dalloc<float>((size_t) 48 * NVH * 128 * 128);
        b.apart = dalloc<float>((size_t) NKV * MAX_CH * AR * 256);
        b.aml = dalloc<float2>((size_t) NKV * MAX_CH * AR);
        b.h_hist = dalloc<float>((size_t) H_RING * D);
        b.hm = dalloc<float>(D);
        b.dlog = dalloc<float2>((size_t) MAXD * (TOPK_MAX + 1));
        b.dq = dalloc<float2>((size_t) MAXD * TOPK_MAX);
        b.ctl = dalloc<Ctl>(1);
        b.pcnt = nullptr;
        b.logits = nullptr;
        b.rope = rope;
        {
            Ctl * ce = nullptr;
            CK(cudaHostAlloc(reinterpret_cast<void **>(&ce), sizeof(Ctl), cudaHostAllocMapped));
            memset(ce, 0, sizeof(Ctl));
            l.ctl_eg_host = ce;
            CK(cudaHostGetDevicePointer(reinterpret_cast<void **>(&b.ctl_eg), ce, 0));
        }
        l.ck = ck_new_slots(nck);
        CK(mkl::set_cBs(&b, sizeof(Bufs), view_full(k) * sizeof(Bufs)));
        CK(mkl::set_cBs(&b, sizeof(Bufs), view_main(k) * sizeof(Bufs)));
        {
            const Bufs vm = lane_main_view(l, k); // (batched MTP: lane k's rows at row k)
            CK(mkl::set_cBs(&vm, sizeof(Bufs), view_mtp(k) * sizeof(Bufs)));
        }
        lane_in(k);
        ck_make_plan();
        reset();
        seq.clear();
        lane_out();
#if MK_TEST_SLOTS_FAIL
        if (getenv("MK_TEST_SLOTS_FAIL_AT") && atoi(getenv("MK_TEST_SLOTS_FAIL_AT")) == k) {
            fprintf(stderr, "[mk] MK_TEST_SLOTS_FAIL: slots_open fails after lane %d\n", k);
            throw MKCudaError("MK_TEST_SLOTS_FAIL: slots_open failed after lane " + std::to_string(k));
        }
#endif
    }
    dv_reset();
    dv_frozen = true;
    undo.done = true;
    bool lowbit = false;
    for (int w = 0; w < 2; w++)
        for (int sl = 0; sl < kv_slots; sl++) lowbit |= hb.kvb[w][sl] != 8 || (w ? hb.kvrv[sl] : hb.kvr[sl]);
    int ck_rings = 0;
    for (const CkSlot & c : lanes[0].ck) ck_rings += c.ring != nullptr;
    fprintf(stderr, "[mk] %d parallel slots of %d positions (%s KV%s: %.2f GB, %.2f GB each in all, %zu checkpoints each",
            n, lctx, lowbit ? "MK_KV" : "q8", hb.kvwin > 0 ? " + MK_KV_WIN rings" : "", lane_kv_bytes(lctx) / 1e9,
            lane_bytes(lctx) / 1e9, lanes[0].ck.size());
    if (ring_bytes_ck > 0) fprintf(stderr, ", %d with the window rings, %.1f MiB pinned host each", ck_rings, ring_bytes_ck / 1048576.0);
    fprintf(stderr, ")\n");
    return lctx;
}

#if MK_PAR_WIDE
std::string MKEngine::Impl::wide_why() const {
    std::string why;
    for (int L = 0; L < NL && why.empty(); L++) {
        const LayerW & W = hw.L[L];
        if (W.attn && !(W.q.fm && W.k.fm && W.v.fm && W.o.fm)) why = "an attention q / k / v / out matrix of a type outside the wv layout";
        if (!W.attn && !W.sout.fm) why = "a delta-net ssm_out matrix of a type outside the wv layout";
        if (!W.fd.fm) why = "an ffn_down matrix outside the wv layout";
    }
    const int G = nsm / WV_S;
    if (why.empty() && ((D / 16 + G - 1) / std::max(G, 1) > NW || (FF / 16 / WV_S + G - 1) / std::max(G, 1) > NW ||
                        (QKV / 16 + VD / 16 + nsm - 1) / nsm > NW || QG / 16 + 4 * (KVD / 16) > nsm * NW))
        why = "a GPU with fewer SMs than one item per warp needs";
    return why;
}
#endif

#if MK_MAXD > 8
std::string MKEngine::Impl::deep_why() const {
    if (!hb.kvq8) return "the q8 KV cache (--kv q8)";
    for (int w = 0; w < 2; w++)
        for (int sl = 0; sl < std::min(kv_slots, N_ATT_SLOTS - 1); sl++)
            if (hb.kvb[w][sl] != 8) return "8-bit K / V in every attention layer (no low-bit MK_KV preset)";
    if (hb.kvwin > 0) return "no MK_KV_WIN rings";
    for (int L = 0; L < NL; L++)
        if (!hw.L[L].attn && !(hw.L[L].qkv.fm && hw.L[L].gate.fm)) return "delta-net projections in the wv layout";
    if (nsm < mk_min_grid(MAXT)) return "a GPU with at least " + std::to_string(mk_min_grid(MAXT)) + " SMs";
    const std::string why = wide_why();
    return why.empty() ? why : "the RMAX-row weight phases: " + why;
}

void MKEngine::Impl::set_view_hi() {
    Bufs v = rows_view(hb, MAXT);
    v.apart = apart_hi;
    v.aml = aml_hi;
    CK(mkl::set_cBs_async(&v, sizeof(Bufs), view_hi() * sizeof(Bufs), cudaMemcpyHostToDevice, 0));
}
#endif

Bufs MKEngine::Impl::lane_main_view(const Lane & l, int r0) const { return rows_view(l.hb, r0); }

Bufs MKEngine::Impl::rows_view(const Bufs & b, int r0) {
    Bufs v = b;
    v.x += (size_t) r0 * D;
    v.xm += (size_t) r0 * D;
    auto qoff = [&](QBuf & q) {
        q.xa += (size_t) r0 * q.U * 16;
        q.xb += (size_t) r0 * q.U * 16;
        q.meta += (size_t) r0 * q.U;
        q.ssq += (size_t) r0 * q.U;
    };
    qoff(v.qn);
    qoff(v.qo);
    qoff(v.qa);
    qoff(v.qe);
    v.z += (size_t) r0 * VD;
    v.ab += (size_t) r0 * 2 * NVH;
    v.qbuf += (size_t) r0 * QG;
    v.kbuf += (size_t) r0 * KVD;
    v.vbuf += (size_t) r0 * KVD;
    v.cand += (size_t) r0 * TOPK_MAX;
    return v;
}

std::vector<std::vector<int>> MKEngine::Impl::lanes_step(const std::vector<int> & ks) {
    lane_out();
    const int nl = (int) ks.size();
    std::vector<std::vector<int>> out(nl);
    // rows per lane: the lanes' slots (lane_slot: T <= 5, or 8 rows) share RMAX rows (par_wide) or MAXT
    const int rmax = par_wide ? RMAX : MAXT;
    int tcap = rmax / nl >= MAXT ? MAXT : std::min(5, rmax / nl);
    // two lanes: both in one MAXT-row pass (3 drafts each) are faster than a wide pass with deeper drafts (the
    // wide passes are faster at 3 and 4 lanes)
    if (nl == 2) tcap = MAXT / 2;
    Launch L{};
    L.mode = MODE_DECODE;
    L.nl = 0;
    int rows = 0, top = 0;
    std::vector<int> li(nl, -1), nd(nl, 0), Ti(nl, 0);
    for (int i = 0; i < nl; i++) {
        Lane & l = lanes[ks[i]];
        const Ctl & c = l.hctl;
        if (c.n + MAXT + 2 >= l.ctx) continue; // (context full: no tokens)
        int T = att_cap_T(c.n, std::min(tcap, 1 + l.nd_avail));
        if (T > 5 && att_cap_T(c.n, MAXT) < MAXT) T = 5; // (an 8-row slot lies in one attention partition, as step_live's)
        int d = std::min(depth_top(c.n), tcap - 1);
        if (pol.depth == 0) d = (!l.all_acc && d > 3) ? d - 1 : d;
        else if (pol.depth == 2) d = std::min(d, std::max(std::min(pol.dmin, d), 1 + l.last_acc));
        Launch::Lane & q = L.ln[L.nl];
        q.lane = ks[i];
        q.T = T;
        q.r0 = rows;
        q.nd = d;
        q.top_k = l.smp_top_k;
        q.spec_smp = l.smp_temp > 0.f && pol.h1;
        q.temp = l.smp_temp;
        q.top_p = l.smp_top_p;
        q.min_p = l.smp_min_p;
        q.seed = l.smp_seed;
        li[i] = L.nl++;
        nd[i] = d;
        Ti[i] = T;
        rows += lane_slot(T);
        top = std::max(top, d);
    }
    if (L.nl == 0) return out;
    if (L.nl == 1) { // (the others are full: a solo step)
        int k = -1;
        for (int i = 0; i < nl; i++)
            if (li[i] >= 0) k = i;
        lane_in(ks[k]);
        std::vector<int> o;
        step_live(o);
        out[k] = o;
        lane_out();
        return out;
    }
    L.T = rows <= 5 ? rows : rows <= MAXT ? MAXT : RMAX;
    L.mtpb = hb.mtp_win <= 0; // (MK_MTP_WIN: the window start follows the pass's first row)
    L.ndraft = top;
    for (int k = 0; k < MAXD - 1; k++) {
        L.gate_drop[k] = pol.gate_drop[k];
        L.gate_stop[k] = pol.gate_stop[k];
    }
    if (top >= 1) L.gate_drop[top - 1] = std::max(L.gate_drop[top - 1], pol.drop_last);
    L.gate_chain = pol.gate_chain;
    L.spec_smp = 0;
    for (int i = 0; i < L.nl; i++) L.spec_smp |= L.ln[i].spec_smp;
    if (L.gate_chain == 2 && L.spec_smp)
        for (int k = 0; k < MAXD - 1; k++) L.gate_drop[k] = 0.f;
    for (int i = 0; i < nl; i++) {
        if (li[i] < 0) continue;
        Lane & l = lanes[ks[i]];
        l.kv_top = std::max(l.kv_top, l.hctl.n + Ti[i] + nd[i]);
        const Bufs v = lane_main_view(l, L.ln[li[i]].r0);
        CK(mkl::set_cBs_async(&v, sizeof(Bufs), view_main(ks[i]) * sizeof(Bufs), cudaMemcpyHostToDevice, 0));
        if (L.mtpb) { // (the lane's MTP view: its buffers as they are now, rows at row ks[i])
            const Bufs vm = lane_main_view(l, ks[i]);
            CK(mkl::set_cBs_async(&vm, sizeof(Bufs), view_mtp(ks[i]) * sizeof(Bufs), cudaMemcpyHostToDevice, 0));
        }
    }
    void * args[] = {&L};
    ck_wait_restore();
    ck_guard_state();
    ring_snap_guard();
    if (!bar_live) {
        CK(cudaMemsetAsync(hb.bar, 0, C1_WORDS * 4));
        bar_live = true;
    }
    if (profiling) CK(cudaMemsetAsync(hb.prof, 0, (4096 + 1024 * 128) * 8));
    CK(cudaLaunchCooperativeKernel(mkl::k_mk_cycle, dim3(nsm), dim3(NT), args, SM_TOTAL, 0));
    wd_T = L.T;
    wd_mode = MODE_DECODE;
    wait_device("a batched decode launch", wd_limit);
    if (err_host[0] | err_host[1]) launch_failed();
    launches_seen++;
    if (profiling) prof_collect(0);
    for (int i = 0; i < nl; i++) {
        if (li[i] < 0) continue;
        Lane & l = lanes[ks[i]];
        Ctl & c = l.hctl;
        memcpy(&c, const_cast<const Ctl *>(l.ctl_eg_host), sizeof(Ctl));
        l.nd_avail = nd[i] > 0 ? std::min(nd[i], c.nd_prod) : 0;
        l.ndraft = nd[i];
        out[i].assign(c.out, c.out + c.n_out);
        l.seq.insert(l.seq.end(), c.out, c.out + c.n_out);
        l.dv_n = -1;
        l.all_acc = c.n_acc == Ti[i] - 1;
        l.last_acc = c.n_acc;
    }
    return out;
}

// ---- parallel slots (open-jet's slot API over lanes, see Impl::Lane) ----
// Every slot call leaves the engine's own sequence live again, so the single-sequence API is unaffected between them.
// An MKCudaError in one marks the lanes it ran on dead (Impl::lanes_call): refused until slot_begin starts them afresh.
#define MK_LANE_OUT()                                       \
    struct LaneOut {                                        \
        Impl * p;                                           \
        ~LaneOut() { p->lane_out_nothrow(); }               \
    } lane_out_guard{p_}

int MKEngine::slots_open(int n, int ctx, int /*seg_tokens*/) { return p_->slots_open(n, ctx); }

int MKEngine::slots_count() const { return (int) p_->lanes.size(); }

int MKEngine::slot_begin(int slot, const std::vector<int> & toks, float temp, float top_p, int top_k, float min_p,
                         unsigned long long seed, int /*share_len*/, bool /*pin*/) {
    Impl & I = *p_;
    MK_LANE_OUT();
    if (toks.empty()) throw std::runtime_error("empty prompt");
    check_ids(toks, 0);
    int reused = 0;
    I.lanes_call({slot}, [&] {
        I.lane_in(slot);
        set_sampling(temp, top_p, top_k, min_p, seed);
        Impl::Lane & L = I.lanes[slot];
        L.first = -1;
        L.pend.clear();
        if ((int) toks.size() + 8 > I.ctx) throw std::runtime_error("prompt exceeds the slot's context");
        if (L.dead) { // (a failed call left its state undefined: from scratch, nothing reused)
            reset();
            L.dead = false;
            L.dead_why.clear();
        }
        size_t c = 0;
        while (c < I.seq.size() && c < toks.size() && I.seq[c] == toks[c]) c++;
        // MK_EXACT: the last prompt's last row and every row after it came from the megakernel (they depend on the draft
        // schedule): a prompt reaching them goes back into the last prompt, as slot_begin(slot, that prompt) first would
        if (I.exact && !L.prompt.empty() && c >= L.prompt.size()) c = L.prompt.size() - 1;
        if (c < I.seq.size() || c == toks.size()) {
            // back to a position <= c from which at least the prompt's last token is fed: a truncate (emitted tokens past
            // the committed position), else the newest checkpoint there, else from scratch
            const int target = (int) std::min(c, toks.size() - 1);
            int best = -1;
            for (const auto & k : I.ck)
                if (k.pos >= 0 && k.pos <= target && k.pos > best) best = k.pos;
            if (target >= I.hctl.dn_c) truncate(target);
            else if (best >= 0 && rewind_to(best)) {
            } else reset();
        }
        reused = (int) I.seq.size();
        L.prompt = toks;
        L.pend.assign(toks.begin() + reused, toks.end());
    });
    return reused;
}

int MKEngine::slot_prefill(int slot, int budget) {
    Impl & I = *p_;
    MK_LANE_OUT();
    Impl::Lane & L = I.lane_ok(slot);
    int left = 0;
    I.lanes_call({slot}, [&] {
        I.lane_in(slot);
        if (budget <= 0) { // (abandoned: the sequence keeps what was prefilled)
            L.pend.clear();
            return;
        }
        if (L.pend.empty()) return;
        size_t n = std::min<size_t>(L.pend.size(), (size_t) budget);
        // MK_EXACT: a chunk before the last ends on the 32-token scan grid (the next grid point if the budget reaches none)
        if (I.exact && n < L.pend.size()) {
            const size_t s = I.seq.size(), e = (s + n) / 32 * 32;
            n = e > s ? e - s : std::min(L.pend.size(), (s / 32 + 1) * 32 - s);
            // a last chunk of <= 32 tokens from a grid point would save no checkpoint (ck_positions snaps it down onto the
            // chunk's start): this one takes the rest, up to 32 over the budget (fed from the grid, batched: the same rows)
            const char * mode = getenv("MK_PREFILL");
            if (L.pend.size() - n <= 32 && I.hctl.dn_c % 32 == 0 && !(mode && std::string(mode) == "mk")) n = L.pend.size();
        }
        const std::vector<int> chunk(L.pend.begin(), L.pend.begin() + n);
        L.pend.erase(L.pend.begin(), L.pend.begin() + n);
        try {
            if (L.pend.empty()) { // the prompt's last chunk: its first token, a checkpoint before the prompt's last token
                const int end = (int) I.seq.size() + (int) chunk.size();
                L.first = prefill(chunk, end - 1);
            } else { // a chunk before the last: batched rows only (with MK_PREFILL=mk its last row samples a token: dropped)
                struct Mid {
                    Impl * p;
                    ~Mid() { p->pf_mid = false; }
                } mid{p_};
                I.pf_mid = true;
                if (prefill(chunk) >= 0) truncate((int) I.seq.size() - 1);
            }
        } catch (const MKCancelled &) {
            // the sequence went back to a checkpoint before this call (or was reset): a prefix of the prompt, whose
            // rest is pending again (the next slot_prefill continues there)
            const size_t s = I.seq.size();
            if (s <= L.prompt.size() && std::equal(I.seq.begin(), I.seq.end(), L.prompt.begin()))
                L.pend.assign(L.prompt.begin() + s, L.prompt.end());
            else
                L.pend.clear(); // (not a prefix: nothing queued, the next slot_begin matches again)
            throw;
        }
        left = (int) L.pend.size();
    });
    return left;
}

std::vector<std::vector<int>> MKEngine::slots_step(const std::vector<int> & slots) {
    Impl & I = *p_;
    MK_LANE_OUT();
    I.slots_ok(slots); // (every slot before any of them changes)
    std::vector<std::vector<int>> out(slots.size());
    std::vector<int> ks, at, firsts;
    for (size_t i = 0; i < slots.size(); i++) {
        Impl::Lane & L = I.lanes[slots[i]];
        if (L.first >= 0) { // (its prefill's token first: this call only emits it)
            out[i].push_back(L.first);
            firsts.push_back(slots[i]);
            continue;
        }
        ks.push_back(slots[i]);
        at.push_back((int) i);
    }
    I.lanes_call(ks, [&] {
        if (ks.size() == 1) {
            I.lane_in(ks[0]);
            std::vector<int> o;
            const Ctl & c = I.hctl;
            if (c.n + MAXD + 2 < I.ctx) step(o); // (else full: no tokens)
            out[at[0]] = o;
        } else if (ks.size() >= 2) {
            const auto r = I.lanes_step(ks); // (a failure there: every lane in ks, the batched launch wrote them all)
            for (size_t j = 0; j < ks.size(); j++) out[at[j]] = r[j];
        }
    });
    for (int s : firsts) I.lanes[s].first = -1; // (taken only now: a throw above leaves the token for the next call)
    return out;
}

// the next chunk of slot pslot's prompt, then the step (separate launches; the chunk first: a failure in it leaves the
// decoding slots unstepped, so no emitted token is lost with the call's error)
std::vector<std::vector<int>> MKEngine::slots_step_mixed(int pslot, int budget, const std::vector<int> & slots, int & remain) {
    p_->lane_ok(pslot); // (a dead slot, or one listed twice, fails the call before anything changes)
    p_->slots_ok(slots);
    remain = slot_prefill(pslot, budget);
    return slots_step(slots);
}

std::vector<int> MKEngine::slot_sequence(int slot) {
    MK_LANE_OUT();
    p_->lane_in(slot);
    return p_->seq;
}

std::vector<float> MKEngine::slot_h(int slot) {
    MK_LANE_OUT();
    p_->lane_in(slot);
    return last_hidden();
}
