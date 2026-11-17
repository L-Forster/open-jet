// The engine's host side: MKEngine's API and MKEngine::Impl's core (launches, decode steps, policy, buffers). Impl's
// other member groups (and their API): ck_pool.cu, loader.cu, prefill_host.cu, state_files.cu, host_store.cu, lanes.cu.

#include "engine_impl.h"

#if MK_SPEC_LOG
void MKEngine::Impl::slog_drafts() {
    if (!slog) return;
    slog_q.resize((size_t) MAXD * (TOPK_MAX + 1));
    CK(cudaMemcpy(slog_q.data(), hb.dlog, slog_q.size() * sizeof(float2), cudaMemcpyDeviceToHost));
}
#endif
void MKEngine::Impl::ctl_fetch(Ctl & c) {
    if (ctl_eg_host) memcpy(&c, const_cast<const Ctl *>(ctl_eg_host), sizeof(Ctl));
    else CK(cudaMemcpy(&c, hb.ctl, sizeof(Ctl), cudaMemcpyDeviceToHost));
}

void MKEngine::Impl::launch_failed() {
    const int bad = err_host[0], mtp = err_host[1];
    err_host[0] = err_host[1] = 0;
    if (mtp && !err_mtp_warned) {
        err_mtp_warned = true;
        fprintf(stderr, "[mk] warning: MTP catch-up out of range near position %d (drafts only, output unaffected)\n", hctl.n);
    }
    if (bad)
        throw MKCudaError("no finite logit in the output head (NaN / inf model state); the sequence is reset (a parallel "
                          "slot's when it begins its next prompt)");
}

void MKEngine::Impl::rope_sync() {
    if (spans == rope_spans) return;
    size_t k = 0;
    while (k < spans.size() && k < rope_spans.size() && spans[k] == rope_spans[k]) k++;
    const int nr = hb.ctx; // the table's rows (KV_PAD past ctx)
    int from = nr, shift = 0;
    if (k < spans.size()) from = std::min(from, spans[k].pos);
    if (k < rope_spans.size()) from = std::min(from, rope_spans[k].pos);
    for (size_t j = 0; j < k; j++) shift += spans[j].n() - std::max(spans[j].nx, spans[j].ny);
    float2 * dst = const_cast<float2 *>(hb.rope);
    std::vector<float2> rows;
    size_t j = k;
    for (int p = from; p < nr;) {
        const int next = j < spans.size() ? spans[j].pos : nr;
        if (p < next) { // text rows p .. next - 1
            CK(cudaMemcpy(dst + (size_t) p * 32, rope_base.data() + (size_t) (p - shift) * 32, (size_t) (next - p) * 32 * sizeof(float2),
                          cudaMemcpyHostToDevice));
            p = next;
            continue;
        }
        const Span & s = spans[j++];
        const int r = p - shift, n = std::min(s.n(), nr - p);
        rows.resize((size_t) n * 32);
        for (int q = 0; q < n; q++) {
            const int pos3[3] = {r, r + q / s.nx, r + q % s.nx};
            for (int i = 0; i < 32; i++) rows[(size_t) q * 32 + i] = rope_base[(size_t) pos3[i % 3] * 32 + i];
        }
        CK(cudaMemcpy(dst + (size_t) p * 32, rows.data(), rows.size() * sizeof(float2), cudaMemcpyHostToDevice));
        shift += s.n() - std::max(s.nx, s.ny);
        p += s.n();
    }
    rope_spans = spans;
}

void MKEngine::Impl::spans_cut(int len) {
    while (!spans.empty() && spans.back().pos >= len) spans.pop_back();
    rope_sync();
}

void MKEngine::Impl::dv_add(const int * toks, size_t n) {
    if (dv_frozen) return; // (parallel slots: the base list for every sequence)
    if (dv_in.empty()) return; // full-vocabulary draft head
    const int n0 = (int) dv_ids.size();
    for (size_t i = 0; i < n && (int) dv_ids.size() < dv_base + DV_EXTRA - 1; i++)
        if (!dv_in[toks[i]]) {
            dv_in[toks[i]] = 1;
            dv_ids.push_back(toks[i]);
        }
    if ((int) dv_ids.size() == n0) return;
    if (dv_ids.size() & 1) dv_ids.push_back(dv_ids.back()); // the draft lm_head takes rows in pairs
    const int k = (int) dv_ids.size() - n0;
    Mat & m = hw.head_d;
    if (dv_pin) {
        // asynchronous on the launch stream (ordered before the next launch). The staged ids [n0, n0 + k) are not
        // rewritten before a later synchronisation: appends go past them, and dv_truncate's synchronous row-count copy
        // waits for every copy queued before it (the next dv_add after it may restage the same ids)
        m.N = (int) dv_ids.size();
        memcpy(dv_pin + n0, dv_ids.data() + n0, k * 4);
        dv_pin[dv_base + DV_EXTRA] = m.N;
        CK(cudaMemcpyAsync(const_cast<int *>(hw.draft_ids) + n0, dv_pin + n0, k * 4, cudaMemcpyHostToDevice, 0));
        CK(mkl::set_cW_async(dv_pin + dv_base + DV_EXTRA, sizeof(int), offsetof(ModelW, head_d) + offsetof(Mat, N),
                             cudaMemcpyHostToDevice, 0));
        return;
    }
    CK(cudaMemcpy(const_cast<int *>(hw.draft_ids) + n0, dv_ids.data() + n0, k * 4, cudaMemcpyHostToDevice));
    m.N = (int) dv_ids.size();
    CK(mkl::set_cW(&m.N, sizeof(int), offsetof(ModelW, head_d) + offsetof(Mat, N)));
}

void MKEngine::Impl::dv_reset() {
    dv_truncate(dv_base);
    dv_n = 0;
}

void MKEngine::Impl::dv_rebuild(int n, const std::vector<Span> & sp) {
    if (dv_frozen) return; // (parallel slots: the base list for every sequence)
    dv_reset();
    dv_n = n;
    if (dv_in.empty()) return;
    std::vector<int> text;
    text.reserve((size_t) n);
    size_t k = 0;
    for (int q = 0; q < n; q++) {
        while (k < sp.size() && sp[k].pos + sp[k].n() <= q) k++;
        if (k == sp.size() || q < sp[k].pos) text.push_back(seq[q]);
    }
    dv_add(text.data(), text.size());
}

void MKEngine::Impl::dv_truncate(size_t n) {
    if (dv_frozen) return; // (parallel slots: the base list for every sequence)
    if (dv_in.empty() || dv_ids.size() <= n) return;
    for (size_t i = n; i < dv_ids.size(); i++) dv_in[dv_ids[i]] = 0;
    for (size_t i = 0; i < n; i++) dv_in[dv_ids[i]] = 1; // a padding duplicate may have been cleared
    dv_ids.resize(n);
    hw.head_d.N = (int) n;
    CK(mkl::set_cW(&hw.head_d.N, sizeof(int), offsetof(ModelW, head_d) + offsetof(Mat, N)));
}

void MKEngine::Impl::wait_device(const char * what, double limit) {
    if (limit <= 0.0 || !wd_ev) {
        CK(cudaStreamSynchronize(0));
        return;
    }
    CK(cudaEventRecord(wd_ev, 0));
    wait_event(wd_ev, what, limit);
}

void MKEngine::Impl::wait_event(cudaEvent_t ev, const char * what, double limit) {
    if (limit <= 0.0) {
        CK(cudaEventSynchronize(ev));
        return;
    }
    const auto t0 = std::chrono::steady_clock::now();
    for (;;) {
        const cudaError_t e = cudaEventQuery(ev);
        if (e == cudaSuccess) return;
        if (e != cudaErrorNotReady) mk_cuda_fail(e, (std::string("(the GPU work of ") + what + ")").c_str(), __FILE__, __LINE__);
        const double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        if (el > limit) {
            const cudaError_t le = cudaPeekAtLastError();
            fprintf(stderr,
                    "[mk] watchdog: the GPU did not finish %s within %g s (mode %d, T %d, position %d, committed %d, "
                    "launches %ld, last CUDA error: %s). The device or its driver is stuck: exiting with code 3.\n",
                    what, limit, wd_mode, wd_T, hctl.n, hctl.dn_c, launches_seen, cudaGetErrorString(le));
            fflush(stderr);
            _exit(3);
        }
#ifdef _WIN32
        // (sleep_for(100 us) is Sleep(1) there: until the next timer tick, 15.6 ms unless the timer resolution was raised)
        if (el > wd_spin_s) mk_win_sleep_us(el > 2.0 ? 2000 : 100);
#else
        if (el > wd_spin_s) std::this_thread::sleep_for(std::chrono::microseconds(el > 2.0 ? 2000 : 100));
#endif
    }
}

void MKEngine::Impl::mem_mark(const char * tag) {
    if (!memlog) return;
    CK(cudaDeviceSynchronize());
    size_t fr = 0, tot = 0;
    CK(cudaMemGetInfo(&fr, &tot));
    if (mark_free)
        fprintf(stderr, "[mk-mem] %-26s alloc %9.2f MiB   free drop %9.2f MiB   (free %9.1f MiB)\n", tag,
                (alloc_bytes - mark_alloc) / 1048576.0, ((double) mark_free - (double) fr) / 1048576.0, fr / 1048576.0);
    else
        fprintf(stderr, "[mk-mem] %-26s free %.1f MiB of %.1f MiB\n", tag, fr / 1048576.0, tot / 1048576.0);
    mark_free = fr;
    mark_alloc = alloc_bytes;
}

// MK_RESERVE_EARLY: a full device (cudaMemGetInfo's free ~0) means the CUDA sysmem fallback (Windows, WSL2) may have put part of
// this process's memory in system memory; a stack limit above the one reserved at load means a kernel's frame was reserved late
void MKEngine::Impl::vram_check(const char * where) {
    size_t fr = 0, tot = 0, stk = 0;
    CK(cudaMemGetInfo(&fr, &tot));
    CK(cudaDeviceGetLimit(&stk, cudaLimitStackSize));
    if (stk > stack_res)
        fprintf(stderr, "[mk] warning: %s: the stack limit grew to %zu B per thread (%zu reserved at load): a kernel's local memory was "
                        "reserved late\n", where, stk, stack_res);
    if (fr < ((size_t) 1 << 20))
        fprintf(stderr, "[mk] warning: %s: the device memory is full (%.2f MiB free): with the CUDA sysmem fallback (Windows, WSL2) part of "
                        "this process's memory may now sit in system memory (slower decode, varying per process); a smaller context "
                        "(-c) or a larger MK_VRAM_MARGIN avoids it\n", where, fr / 1048576.0);
}
#if MK_SPEC_LOG
void MKEngine::Impl::slog_cycle(int n0, int T, const int * drafts, const Ctl & c) {
    if (!slog) return;
    const bool smp = smp_temp > 0.f;
    std::vector<float2> cand;
    if (smp) {
        cand.resize((size_t) T * TOPK_MAX);
        CK(cudaMemcpy(cand.data(), hb.cand, cand.size() * sizeof(float2), cudaMemcpyDeviceToHost));
    }
    auto id_of = [](float y) { int v; memcpy(&v, &y, 4); return v; };
    fprintf(slog, "{\"n\":%d,\"T\":%d,\"a\":%d,\"nd\":%d,\"d\":[", n0, T, c.n_acc, nd_avail);
    for (int i = 1; i < T; i++) fprintf(slog, "%s%d", i > 1 ? "," : "", drafts[i]);
    fprintf(slog, "],\"tg\":[");
    for (int i = 0; i < T; i++) fprintf(slog, "%s%d", i ? "," : "", c.tgt[i]);
    fprintf(slog, "],\"o\":[");
    for (int i = 0; i < c.n_out; i++) fprintf(slog, "%s%d", i ? "," : "", c.out[i]);
    fprintf(slog, "],\"q\":[");
    for (int i = 0; i + 1 < T; i++) {
        const float2 * q = slog_q.data() + (size_t) i * (TOPK_MAX + 1);
        fprintf(slog, "%s[%.7g,%.7g", i ? "," : "", q[TOPK_MAX].x, q[TOPK_MAX].y);
        for (int j = 0; j < TOPK_MAX; j++) fprintf(slog, ",%.7g,%d", q[j].x, id_of(q[j].y));
        fprintf(slog, "]");
    }
    fprintf(slog, "]");
    if (smp) {
        fprintf(slog, ",\"p\":[");
        for (int i = 0; i < T; i++) {
            fprintf(slog, "%s[", i ? "," : "");
            for (int j = 0; j < TOPK_MAX; j++)
                fprintf(slog, "%s%.7g,%d", j ? "," : "", cand[(size_t) i * TOPK_MAX + j].x, id_of(cand[(size_t) i * TOPK_MAX + j].y));
            fprintf(slog, "]");
        }
        fprintf(slog, "]");
    }
    fprintf(slog, "}\n");
    fflush(slog);
    slog_drafts(); // the drafts this launch produced, verified by the next cycle
}

void MKEngine::Impl::slog_prompt(int n0) {
    if (!slog) return;
    fprintf(slog, "{\"prompt\":%zu,\"n0\":%d,\"ids\":[", seq.size(), n0);
    for (size_t i = 0; i < seq.size(); i++) fprintf(slog, "%s%d", i ? "," : "", seq[i]);
    fprintf(slog, "]}\n");
    fflush(slog);
}
#endif
MKEngine::Impl::~Impl() {
#if MK_SPEC_LOG
    if (slog) fclose(slog);
#endif
    try { // (CK throws; a destructor must not)
        print_profile();
        mem_mark("at exit");
        ck_wait_restore();
    } catch (const std::exception & e) {
        fprintf(stderr, "[mk] teardown: %s\n", e.what());
    }
    release();
}

void MKEngine::Impl::release() noexcept {
    if (!allocs.empty()) cudaDeviceSynchronize(); // (a non-blocking stream's work may still use them)
    ck_free(); // (checkpoint copies may still read or write device buffers)
    for (auto & l : lanes) { // (a live lane's resources are in Impl's members, the engine's own ones in its Lane)
        ck_slots_free(l.ck);
        if (l.ctl_eg_host) cudaFreeHost(const_cast<Ctl *>(l.ctl_eg_host));
    }
    for (auto & p : parked)
        for (char * c : p.second.blob.chunks) cudaFreeHost(c);
    for (char * c : sw_blob.chunks) cudaFreeHost(c);
    for (char * c : hs_free) cudaFreeHost(c);
    for (cudaStream_t s : {pf_s2, pf_s3}) // (MK_K8 bit 1, pf_fork)
        if (s) {
            cudaStreamSynchronize(s);
            cudaStreamDestroy(s);
        }
    for (cudaEvent_t e : {pf_e0, pf_e1, pf_e2, pf_cev[0], pf_cev[1], wd_ev})
        if (e) cudaEventDestroy(e);
    for (cudaEvent_t e : pf_gev) cudaEventDestroy(e);
    for (void * p : allocs) use_pool ? cudaFreeAsync(p, 0) : cudaFree(p);
    if (!allocs.empty()) pool_trim(); // (a caller's retry finds the VRAM free)
    // a constructor that threw while the weights loaded: wv_dev_done did not give back the staging buffer (192 MiB) and
    // the gather tables (cudaMalloc, outside allocs)
    if (stg_d) cudaFree(stg_d);
    for (void * t : wv_tabs) cudaFree(t);
    stg_d = nullptr;
    wv_tabs.clear();
    if (embd_host) cudaFreeHost(embd_host);
    if (emb_host) cudaFreeHost(emb_host);
    if (err_host) cudaFreeHost(const_cast<int *>(err_host));
    if (ctl_eg_host) cudaFreeHost(const_cast<Ctl *>(ctl_eg_host));
    if (dv_pin) cudaFreeHost(dv_pin);
}

void MKEngine::Impl::pool_trim() noexcept {
    if (!use_pool) return;
    cudaStreamSynchronize(0);
    cudaMemPool_t pool;
    if (cudaDeviceGetDefaultMemPool(&pool, cuda_dev) == cudaSuccess) cudaMemPoolTrimTo(pool, 0);
}

int MKEngine::Impl::ctx_limit() const {
    const auto cl = file.ints.find("qwen35.context_length");
    return (int) std::min<uint64_t>(cl != file.ints.end() && cl->second > 0 ? cl->second : 1u << 20, 1u << 20);
}

void MKEngine::Impl::emb_reserve(size_t floats) {
    if (floats <= emb_cap) return;
    pf_init();
    CK(cudaDeviceSynchronize()); // (no kernel still reads the old rows)
    if (emb_host) CK(cudaFreeHost(emb_host));
    emb_host = nullptr;
    emb_cap = 0;
    CK(cudaHostAlloc(reinterpret_cast<void **>(&emb_host), floats * sizeof(float), cudaHostAllocMapped));
    emb_cap = floats;
    float * dev = nullptr;
    CK(cudaHostGetDevicePointer(reinterpret_cast<void **>(&dev), emb_host, 0));
    hp.emb = dev;
    CK(mkl::set_cP(&hp, sizeof(PBufs)));
}

void MKEngine::Impl::launch(int mode, int T, int last, int score) {
    sw_need(hctl.n + T + ndraft); // (side request: the kept sequence's rows there go to the host store first)
    kv_top = std::max(kv_top, hctl.n + T + ndraft); // rows hctl.n .. + T - 1, the MTP's draft rows after them (ring_rewind)
    Launch L{mode, T, last, ndraft, score, smp_temp, smp_top_p, smp_min_p, smp_top_k, smp_seed};
    for (int k = 0; k < MAXD - 1; k++) {
        L.gate_drop[k] = pol.gate_drop[k];
        L.gate_stop[k] = pol.gate_stop[k];
    }
    const int top = depth_top(hctl.n); // drop_last: the deepest draft of this position's policy depth
    if (pol.g7 && top > 5) {
        for (int k = 1; k <= 3; k++) L.gate_stop[k] = 0.25f;
        L.gate_stop[4] = L.gate_stop[5] = 0.3f;
        L.gate_drop[4] = 0.3f;
    } else if (top >= 1) L.gate_drop[top - 1] = std::max(L.gate_drop[top - 1], pol.drop_last);
#if MK_MAXD > 8
    if (pol.g7 && top > MAXT - 1)
        for (int k = MAXT - 1; k < MAXD - 1; k++) L.gate_stop[k] = pol.dstop;
#endif
    L.gate_chain = pol.gate_chain;
    L.spec_smp = smp_temp > 0.f && pol.h1;
    // chain=2 with sampled drafts: the chain holds the drafted token's own q' probability, so a drop gate on it would
    // pick which drafts get verified by their value (biased output: tests/spec_rng_sim.py); stop gates only
    if (L.gate_chain == 2 && L.spec_smp)
        for (int k = 0; k < MAXD - 1; k++) L.gate_drop[k] = 0.f;
    if (!score && (pen_pres != 0.f || pen_freq != 0.f)) {
        L.pres = pen_pres;
        L.freq = pen_freq;
        pcnt_dirty = true;
    }
#if MK_MAXD > 8
    if (mode == MODE_DECODE && T > MAXT) set_view_hi();
#endif
    void * args[] = {&L};
    ck_wait_restore(); // (cheap no-ops unless a checkpoint copy is pending on ck_st)
    ck_guard_state();
    ring_snap_guard();
#if MK_TEST_NAN
    if (mode == MODE_DECODE && test_nan_at > 0 && ++test_nan_n == test_nan_at) {
        __half * ks = hb.kvs[0][0] ? hb.kvs[0][0] : hb.ksc;
        if (ks) CK(cudaMemset(ks, 0xff, (size_t) NKV * hb.ctx * (HD / 32) * sizeof(__half)));
        fprintf(stderr, "[mk] MK_TEST_NAN: NaN K scales before decode launch %d\n", test_nan_n);
    }
#endif
    if (!bar_live) { // (the counts run on from the previous launch's BAR_BASE / C1_BASE: zeroed once, and after bench launches)
        CK(cudaMemsetAsync(hb.bar, 0, C1_WORDS * 4));
        bar_live = true;
    }
    if (profiling) CK(cudaMemsetAsync(hb.prof, 0, (4096 + 1024 * 128) * 8));
    CK(cudaLaunchCooperativeKernel(mkl::k_mk_cycle, dim3(nsm), dim3(NT), args, SM_TOTAL, 0));
    wd_T = T;
    wd_mode = mode;
    wait_device(mode == MODE_DECODE ? "a decode launch" : "a prefill launch", wd_limit);
    if (err_host[0] | err_host[1]) launch_failed();
    launches_seen++;
    if (memlog && launches_seen == 1) mem_mark("first mk_cycle launch");
    if (reserve_early && launches_seen == 1) vram_check("first mk_cycle launch");
    if (profiling && mode == MODE_DECODE) prof_collect(T);

}

static const char * const kPhName[PH_COUNT] = {
    "embed", "attn_proj", "attn_core", "attn_comb", "resid_o", "dn_proj", "dn_core", "resid_sout", "ffn_gu", "ffn_down",
    "lm_head", "mtp_m0", "mtp_eh", "mtp_attn_proj", "mtp_attn_core", "mtp_attn_comb", "mtp_resid_o", "mtp_ffn_gu",
    "mtp_ffn_down", "mtp_lm_head", "mtp_kv_m0", "mtp_kv_eh", "mtp_kv_proj"};

// MK_PROF_DUMP: one launch's raw stamps, times in ns (56-bit globaltimer, wrap-safe like prof_collect): per barrier its
// phase, end (after the launch start), duration, fill mark and every block's arrival (after the phase start; -1: none,
// barriers >= 1024 keep no arrivals). MK_PROF_DUMP_N = k > 1: launches MK_PROF_DUMP .. + k - 1, each to MK_PROF_DUMP_FILE.<n>
// (a phase's minimum over several launches filters the occasional whole-GPU pause). %p in MK_PROF_DUMP_FILE becomes the
// process id: Nsight Compute's application replay runs the program once per pass, and each pass keeps its own stamps
static void prof_dump(const std::vector<unsigned long long> & pr, long n, int T, int nsm, bool multi) {
    const unsigned long long M = (1ull << 56) - 1;
    const char * env = getenv("MK_PROF_DUMP_FILE");
    std::string ps = env ? env : "";
    const std::string pid = std::to_string((long) getpid());
    for (size_t i = ps.find("%p"); i != std::string::npos; i = ps.find("%p", i + pid.size())) ps.replace(i, 2, pid);
    if (env && multi) ps += "." + std::to_string(n);
    const char * path = env ? ps.c_str() : nullptr;
    FILE * f = path ? fopen(path, "w") : stderr;
    if (!f) {
        fprintf(stderr, "[mk] MK_PROF_DUMP: cannot write %s\n", path);
        return;
    }
    const unsigned long long t0 = pr[0] >> 8;
    int K = 1;
    while (K < 2048 && pr[K]) K++;
    fprintf(f, "# mk_prof_dump launch %ld T %d nsm %d t0_ns %llu barriers %d\n", n, T, nsm, t0, K - 1);
    fprintf(f, "# k phase t_end_ns dt_ns fill_ns arrival_ns[block 0 .. nsm - 1]\n");
    unsigned long long prev = t0;
    for (int k = 1; k < K; k++) {
        const unsigned long long t = pr[k] >> 8, mark = pr[2048 + k] & M;
        const int tag = (int) (pr[k] & 0xff);
        fprintf(f, "%d %s %llu %llu %llu", k, tag < PH_COUNT ? kPhName[tag] : "?", (t - t0) & M, (t - prev) & M,
                mark ? (mark - prev) & M : 0ull);
        for (int b = 0; b < nsm; b++) {
            const unsigned long long a = k < 1024 ? pr[4096 + k * 128 + b] & M : 0;
            if (a) fprintf(f, " %llu", (a - prev) & M);
            else fputs(" -1", f);
        }
        fputc('\n', f);
#if MK_PROFILE
        if (k < 1024 && nsm <= 120 && pr[4096 + k * 128 + 120]) { // the last block's sub-stamps 0..3 (sub_stamp), after the phase start
            fprintf(f, "# sub %d", k);
            for (int i = 0; i < 4; i++) {
                const unsigned long long a = pr[4096 + k * 128 + 120 + i] & M;
                if (a) fprintf(f, " %llu", (a - prev) & M);
                else fputs(" -1", f);
            }
            fputc('\n', f);
        }
#endif
        prev = t;
    }
    if (path) fclose(f);
    fprintf(stderr, "[mk] MK_PROF_DUMP: launch %ld (T=%d, %d barriers) -> %s\n", n, T, K - 1, path ? path : "stderr");
}

#if MK_PROFILE
// -DMK_PROFILE=1: the last block's serial section per (T, phase) from its sub-stamps (sub_stamp): sums of ticket -> merged,
// -> sampled, -> control done, -> the barrier's end (ns) and their count
static double g_sub[MAXD + 1][PH_COUNT][4];
static long g_sub_n[MAXD + 1][PH_COUNT];
#endif

void MKEngine::Impl::prof_collect(int T) {
    std::vector<unsigned long long> pr(4096 + 1024 * 128);
    CK(cudaMemcpy(pr.data(), hb.prof, pr.size() * 8, cudaMemcpyDeviceToHost));
    static const long dump_at = getenv("MK_PROF_DUMP") ? atol(getenv("MK_PROF_DUMP")) : 0;
    static const long dump_n = getenv("MK_PROF_DUMP_N") ? std::max(1L, atol(getenv("MK_PROF_DUMP_N"))) : 1;
    if (dump_at > 0 && launches_seen >= dump_at && launches_seen < dump_at + dump_n)
        prof_dump(pr, launches_seen, T, nsm, dump_n > 1);
    unsigned long long prev = pr[0] >> 8;
    auto & bn = best_ns[T];
    auto & bt = best_tag[T];
    auto & bf = best_fill[T];
    for (int k = 1; k < 4096 && pr[k]; k++) {
        const unsigned long long t = pr[k] >> 8;
        const int tag = (int) (pr[k] & 0xff);
        const double dt = (double) (t - prev);
        ph_ns[tag] += dt;
        ph_n[tag]++;
        launch_ns += dt;
        prev = t;
        const unsigned long long mark = pr[2048 + k] & ((1ull << 56) - 1);
        const double fill = mark ? (double) ((mark - (t - (unsigned long long) dt)) & ((1ull << 56) - 1)) : 0.0;
        if ((int) bn.size() < k) { bn.resize(k, 1e30); bt.resize(k, -1); bf.resize(k, 0.0); }
        if (k < 1024) {
            std::vector<double> arr;
            const unsigned long long t0 = t - (unsigned long long) dt;
            for (int b = 0; b < nsm; b++) {
                const unsigned long long a = pr[4096 + k * 128 + b];
                if (a) arr.push_back((double) (((a & ((1ull << 56) - 1)) - t0) & ((1ull << 56) - 1)));
            }
            if (arr.size() == (size_t) nsm) {
                std::sort(arr.begin(), arr.end());
                arr_first[tag] += arr.front();
                arr_med[tag] += arr[arr.size() / 2];
                arr_last[tag] += arr.back();
                arr_n[tag]++;
            }
        }
#if MK_PROFILE
        if (k < 1024 && nsm <= 120 && T >= 0 && T <= MAXD) {
            const unsigned long long M = (1ull << 56) - 1, * sb = &pr[4096 + k * 128 + 120];
            if (sb[0] && sb[1] && sb[2] && sb[3]) {
                g_sub[T][tag][0] += (double) (((sb[1] & M) - (sb[0] & M)) & M);
                g_sub[T][tag][1] += (double) (((sb[2] & M) - (sb[1] & M)) & M);
                g_sub[T][tag][2] += (double) (((sb[3] & M) - (sb[2] & M)) & M);
                g_sub[T][tag][3] += (double) ((t - (sb[3] & M)) & M);
                g_sub_n[T][tag]++;
            }
        }
#endif
        if (bt[k - 1] != tag) { bt[k - 1] = tag; bn[k - 1] = dt; bf[k - 1] = fill; } // sequence changed
        else if (dt < bn[k - 1]) { bn[k - 1] = dt; bf[k - 1] = fill; }
    }
    launches++;
}

void MKEngine::Impl::print_profile() {
    if (!profiling || !launches) return;
    const char * const * names = kPhName;
    fprintf(stderr, "[mk-profile] %ld decode launches, mean %.3f ms in barriers-to-barrier phases\n", launches,
            launch_ns / launches / 1e6);
    for (int i = 0; i < PH_COUNT; i++) {
        if (!ph_n[i]) continue;
        fprintf(stderr, "  %-16s total/launch %8.3f ms   per call %8.2f us   calls/launch %6.1f\n", names[i],
                ph_ns[i] / launches / 1e6, ph_ns[i] / ph_n[i] / 1e3, (double) ph_n[i] / launches);
    }
#if MK_PROFILE
    fprintf(stderr, "[mk-profile] last block's serial section (us, mean): ticket -> merged | -> sampled | -> control done | -> barrier end\n");
    for (int T = 0; T <= MAXD; T++)
        for (int i = 0; i < PH_COUNT; i++)
            if (const long n = g_sub_n[T][i])
                fprintf(stderr, "  T=%d %-16s %7.2f %7.2f %7.2f %7.2f  (x%ld)\n", T, names[i], g_sub[T][i][0] / n / 1e3,
                        g_sub[T][i][1] / n / 1e3, g_sub[T][i][2] / n / 1e3, g_sub[T][i][3] / n / 1e3, n);
#endif
    fprintf(stderr, "[mk-profile] block arrival at barrier (us after phase start; mean over launches): first / median / last\n");
    for (int i = 0; i < PH_COUNT; i++)
        if (arr_n[i])
            fprintf(stderr, "  %-16s %8.2f %8.2f %8.2f\n", names[i], arr_first[i] / arr_n[i] / 1e3, arr_med[i] / arr_n[i] / 1e3,
                    arr_last[i] / arr_n[i] / 1e3);
    for (int T = 0; T <= MAXD; T++) { // (T = 0: batched launches)
        if (best_ns[T].empty()) continue;
        std::vector<double> tot(PH_COUNT, 0.0), fil(PH_COUNT, 0.0);
        std::vector<int> cnt(PH_COUNT, 0);
        double all = 0;
        for (size_t k = 0; k < best_ns[T].size(); k++) {
            if (best_tag[T][k] < 0) continue;
            tot[best_tag[T][k]] += best_ns[T][k];
            fil[best_tag[T][k]] += best_fill[T][k];
            cnt[best_tag[T][k]]++;
            all += best_ns[T][k];
        }
        fprintf(stderr, "[mk-profile] T=%d min-composed launch %.3f ms (interference-free estimate)\n", T, all / 1e6);
        for (int i = 0; i < PH_COUNT; i++)
            if (cnt[i])
                fprintf(stderr, "  %-16s %8.3f ms  %8.2f us/call (fill %5.2f us)  x%d\n", names[i], tot[i] / 1e6,
                        tot[i] / cnt[i] / 1e3, fil[i] / cnt[i] / 1e3, cnt[i]);
        if (getenv("MK_PROFILE_SEQ")) // every barrier position: its phase and min time (per-layer phase costs)
            for (size_t k = 0; k < best_ns[T].size(); k++)
                if (best_tag[T][k] >= 0)
                    fprintf(stderr, "[mk-seq] T=%d k=%zu %s %.2f fill %.2f\n", T, k, names[best_tag[T][k]],
                            best_ns[T][k] / 1e3, best_fill[T][k] / 1e3);
    }
}

void MKEngine::Impl::gsync_bench_host(int N) {
    bar_live = false; // (these launches zero the barrier words themselves and leave no BAR_BASE)
    const int reps = getenv("MK_GSB_REPS") ? std::max(3, atoi(getenv("MK_GSB_REPS"))) : 15;
    mkl::st_warm(nsm * 4, (unsigned long long) 1500 * 1000000ull);
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());
    const int nbs[2] = {nsm, nsm / 2};
    std::vector<float> t[2][4][2]; // [blocks][form][N or 0] ms
    cudaEvent_t a, b;
    CK(cudaEventCreate(&a));
    CK(cudaEventCreate(&b));
    for (int r = 0; r < reps; r++)
        for (int bi = 0; bi < 2; bi++)
            for (int f = 0; f < 4; f++)
                for (int z = 0; z < 2; z++) {
                    Launch L{MODE_BENCH, 1, 3, z ? 0 : N, f};
                    void * args[] = {&L};
                    CK(cudaMemsetAsync(hb.bar, 0, 2 * GS_GO * 4));
                    CK(cudaEventRecord(a));
                    CK(cudaLaunchCooperativeKernel(mkl::k_mk_cycle, dim3(nbs[bi]), dim3(NT), args, SM_TOTAL, 0));
                    CK(cudaEventRecord(b));
                    CK(cudaEventSynchronize(b));
                    float ms = 0.f;
                    CK(cudaEventElapsedTime(&ms, a, b));
                    t[bi][f][z].push_back(ms);
                }
    CK(cudaEventDestroy(a));
    CK(cudaEventDestroy(b));
    auto med = [](std::vector<float> v) {
        std::sort(v.begin(), v.end());
        return (double) v[v.size() / 2];
    };
    static const char * fn[4] = {"0 upstream (red.release, acquire polls)", "1 flag broadcast, relaxed polls",
                                 "2 flag broadcast, acquire polls", "3 upstream arrival, relaxed polls"};
    for (int bi = 0; bi < 2; bi++)
        for (int f = 0; f < 4; f++) {
            const double mn = med(t[bi][f][0]), m0 = med(t[bi][f][1]);
            const double best = *std::min_element(t[bi][f][0].begin(), t[bi][f][0].end());
            const double b0 = *std::min_element(t[bi][f][1].begin(), t[bi][f][1].end());
            fprintf(stderr, "[mk-gsb] blocks %2d form %s: %.3f us per barrier (median; min %.3f), N %d, launch %.1f us, %d reps\n",
                    nbs[bi], fn[f], (mn - m0) * 1e3 / N, (best - b0) * 1e3 / N, N, m0 * 1e3, reps);
        }
    fprintf(stderr, "[mk-gsb] the engine's gsync is form 0\n");
}

void MKEngine::Impl::bench() {
    bar_live = false; // (these launches zero the barrier words themselves and leave no BAR_BASE)
    if (const char * g = getenv("MK_GSB")) {
        gsync_bench_host(std::max(1, atoi(g)));
        return;
    }
    const char * names[3] = {"ffn_gu", "ffn_down", "dn_proj"};
    if (profiling) {
        // i-cache test: 4 back-to-back iterations of one phase inside one launch
        for (int which = 0; which < 3; which++) {
            std::vector<double> best(5, 1e30);
            for (int rep = 0; rep < 60; rep++) {
                Launch L{MODE_BENCH, 4, which, which == 2 ? 0 : 1, 0};
                void * args[] = {&L};
                CK(cudaMemsetAsync(hb.bar, 0, 2 * GS_GO * 4));
                CK(cudaMemsetAsync(hb.prof, 0, 4096 * 8));
                CK(cudaLaunchCooperativeKernel(mkl::k_mk_cycle, dim3(nsm), dim3(NT), args, SM_TOTAL, 0));
                std::vector<unsigned long long> pr(8);
                CK(cudaMemcpy(pr.data(), hb.prof, 8 * 8, cudaMemcpyDeviceToHost));
                for (int k = 1; k <= 4; k++) best[k] = std::min(best[k], (double) ((pr[k] >> 8) - (pr[k - 1] >> 8)));
            }
            fprintf(stderr, "[mk-bench] %-8s T=4 back-to-back iterations: %.1f / %.1f / %.1f / %.1f us\n", names[which],
                    best[1] / 1e3, best[2] / 1e3, best[3] / 1e3, best[4] / 1e3);
        }
        return;
    }
    for (int which = 0; which < 3; which++)
        for (int T : {1, 4}) {
            float best = 1e30f;
            cudaEvent_t a, b;
            cudaEventCreate(&a);
            cudaEventCreate(&b);
            for (int rep = 0; rep < 400; rep++) {
                cudaEventRecord(a);
                Launch L{MODE_BENCH, T, which, which == 2 ? 0 : (rep % 16), 0};
                void * args[] = {&L};
                CK(cudaMemsetAsync(hb.bar, 0, 2 * GS_GO * 4));
                CK(cudaLaunchCooperativeKernel(mkl::k_mk_cycle, dim3(nsm), dim3(NT), args, SM_TOTAL, 0));
                cudaEventRecord(b);
                CK(cudaEventSynchronize(b));
                float ms;
                cudaEventElapsedTime(&ms, a, b);
                best = std::min(best, ms);
            }
            fprintf(stderr, "[mk-bench] %-8s T=%d: %.1f us/launch (1 layer + barrier + launch overhead)\n", names[which], T, best * 1000);
        }
    err_host[0] = err_host[1] = 0; // (bench launches run on synthetic state)
}

void MKEngine::Impl::logits_alloc() {
    if (hb.logits) return;
    hb.logits = dalloc<float>((size_t) MAXT * VOCAB);
    CK(mkl::set_cBs(&hb.logits, sizeof(hb.logits), live_view * sizeof(Bufs) + offsetof(Bufs, logits)));
    mem_mark("scoring logits");
}

void MKEngine::Impl::kv_emu_parse(const std::string & s) {
#if MK_KV_EMU
    for (int w = 0; w < 2; w++)
        for (int q = 0; q < N_ATT_SLOTS; q++) hb.kvemu[w][q] = 0;
    hb.kvemu_sink = 0;
    hb.kvemu_win = 0;
    demoted = 0;
    size_t i = 0;
    while (i < s.size()) {
        size_t j = s.find(',', i);
        if (j == std::string::npos) j = s.size();
        const std::string tok = s.substr(i, j - i);
        i = j + 1;
        if (tok.empty()) continue;
        if (tok.compare(0, 5, "sink=") == 0) {
            hb.kvemu_sink = std::stoi(tok.substr(5));
            continue;
        }
        if (tok.compare(0, 4, "win=") == 0) {
            hb.kvemu_win = std::stoi(tok.substr(4));
            continue;
        }
        const size_t e = tok.find('=');
        if (e == std::string::npos || e == 0 || !strchr("KV*", tok[0])) throw std::runtime_error("MK_KV_EMU_SPEC: bad item " + tok);
        const std::string sl = tok.substr(1, e - 1), md = tok.substr(e + 1);
        int bits = 0, mode = 0;
        size_t k = 0;
        while (k < md.size() && isdigit((unsigned char) md[k])) bits = 10 * bits + (md[k++] - '0');
        for (; k < md.size(); k++) {
            if (md[k] == 'r') mode |= 16;
            else if (md[k] == 'a') mode |= 32;
            else if (md[k] == 'l') mode |= 64;
            else if (md[k] == 'n') mode |= 128;
            else if (md[k] == 's') mode |= 256 | 16;
            else if (md[k] == 't') mode |= 2048 | 256 | 16;
            else if (md[k] == 'T') mode |= 4096 | 256 | 16;
            else throw std::runtime_error("MK_KV_EMU_SPEC: bad mode " + md);
        }
        if (bits < 2 || bits > 8) throw std::runtime_error("MK_KV_EMU_SPEC: bits must be 2..8 in " + tok);
        if ((mode & 64) && (bits > 6 || (mode & 32))) throw std::runtime_error("MK_KV_EMU_SPEC: l needs 2..6 bits and no a: " + tok);
        if ((mode & (2048 | 4096)) && (mode & (32 | 64 | 128))) throw std::runtime_error("MK_KV_EMU_SPEC: t / T take no a, l, n: " + tok);
        mode |= bits;
        const bool all = sl.empty() || sl == "*";
        const int slot = all ? -1 : std::stoi(sl);
        if (!all && (slot < 0 || slot >= N_ATT_SLOTS)) throw std::runtime_error("MK_KV_EMU_SPEC: bad slot in " + tok);
        for (int w = 0; w < 2; w++)
            if (tok[0] == '*' || tok[0] == "KV"[w])
                for (int q = 0; q < kv_slots; q++) // (no MTP block: its slot stays 0, not allocated)
                    if (all || q == slot) hb.kvemu[w][q] = mode == 8 ? 0 : mode;
    }
    fprintf(stderr, "[mk] MK_KV_EMU_SPEC %s: K", s.c_str());
    for (int q = 0; q < N_ATT_SLOTS; q++) fprintf(stderr, " %d", hb.kvemu[0][q]);
    fprintf(stderr, " | V");
    for (int q = 0; q < N_ATT_SLOTS; q++) fprintf(stderr, " %d", hb.kvemu[1][q]);
    fprintf(stderr, " | sink %d win %d\n", hb.kvemu_sink, hb.kvemu_win);
    if (!hb.kvq8) throw std::runtime_error("MK_KV_EMU_SPEC needs the q8 KV cache (--kv q8)");
    for (int w = 0; w < 2; w++)
        for (int q = 0; q < N_ATT_SLOTS; q++)
            if (hb.kvb[w][q] != 8 || hb.kvr[q] || hb.kvrv[q]) throw std::runtime_error("MK_KV_EMU_SPEC needs an all-q8 cache (no MK_KV)");
#else
    (void) s;
    throw std::runtime_error("MK_KV_EMU_SPEC needs a build with -DMK_KV_EMU=1");
#endif
}

void MKEngine::Impl::set_kv_emu(const std::string & spec) { // between sequences (no kernel in flight): new spec, cB re-uploaded
    kv_emu_parse(spec);
#if MK_KV_EMU
    CK(cudaDeviceSynchronize());
    CK(mkl::set_cBs(&hb, sizeof(Bufs), live_view * sizeof(Bufs)));
#endif
}

void MKEngine::Impl::kv_demote_to(int upto) { // demote everything older than `upto` (positions demoted .. upto - 1)
#if MK_KV_EMU
    if (hb.kvemu_win <= 0 || upto <= demoted) return;
    CK(cudaDeviceSynchronize());
    mkl::k_kv_demote(demoted, upto);
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());
    demoted = upto;
#else
    (void) upto;
#endif
}

void MKEngine::Impl::set_penalty(float pres, float freq) {
    if (!std::isfinite(pres) || !std::isfinite(freq)) throw std::runtime_error("penalties must be finite");
    if ((pres != 0.f || freq != 0.f) && !hb.pcnt) {
        // (hb.pcnt only after the kernels' copy: a failed copy leaves both unset, and the next call retries)
        decltype(hb.pcnt) pc = dalloc<unsigned short>(VOCAB, DA_INIT0); // (the first response's counts: these zeros)
        CK(mkl::set_cBs(&pc, sizeof(pc), live_view * sizeof(Bufs) + offsetof(Bufs, pcnt)));
        hb.pcnt = pc;
        mem_mark("penalty counts");
    }
    pen_pres = pres;
    pen_freq = freq;
}

void MKEngine::Impl::pcnt_clear() {
    if (!pcnt_dirty) return;
    CK(cudaMemsetAsync(hb.pcnt, 0, (size_t) VOCAB * sizeof(unsigned short)));
    pcnt_dirty = false;
}

void MKEngine::Impl::reset() {
    ck_rs_slot = -1;   // (a restore not issued yet is moot)
    ck_wait_restore(); // (an issued one still writes the state)
    ck_guard_state();
    kv_top = 0; // (MK_KV_WIN: every position is rewritten before it is read; the ring needs nothing)
    CK(cudaMemset(hb.dstate, 0, sizeof(float) * (size_t) 48 * NVH * 128 * 128));
    CK(cudaMemset(hb.h_hist, 0, sizeof(float) * (size_t) H_RING * D));
    memset(&hctl, 0, sizeof(hctl));
    CK(cudaMemcpy(hb.ctl, &hctl, sizeof(Ctl), cudaMemcpyHostToDevice));
    CK(cudaMemset(hb.cnt, 0, 16));
    stale_replay = stale_pcnt = false;
}


MKEngine::MKEngine(const std::string & path, int ctx, int n_draft, const std::string & draft_vocab, int kv_mode,
                   const std::string & mtp)
    : p_(nullptr), n_draft_(n_draft) {
    if (n_draft < 0 || n_draft > MAXD - 1) throw std::runtime_error("n_draft must be in [0, " + std::to_string(MAXD - 1) + "]");
    if (ctx < 1) throw std::runtime_error("context of " + std::to_string(ctx) + " positions (-c): expected at least 1");
    if (mkl::abi_sig() != mk_abi_sig())
        throw std::runtime_error("engine build: the device object (qwen38_mk.cu) was compiled with other settings than the host files: rebuild both");
    p_ = new Impl(path, ctx, n_draft, draft_vocab, kv_mode, mtp);
    struct DeleteOnThrow { // (~MKEngine does not run for a constructor that throws: a bad MK_POLICY, a CUDA error)
        Impl *& p;
        int n0;
        ~DeleteOnThrow() {
            if (std::uncaught_exceptions() <= n0) return;
            delete p;
            p = nullptr;
        }
    } delete_on_throw{p_, std::uncaught_exceptions()};
    n_draft = n_draft_ = p_->n_draft_cap; // (0 without an MTP block)
#if MK_MAXD > 8
    if (n_draft > MAXT - 1) {
        const std::string why = p_->deep_why();
        if (!why.empty()) throw std::runtime_error("n_draft > " + std::to_string(MAXT - 1) + " (the deep pass) needs " + why);
    }
#endif
    // the wide pass (T >= 6) runs the pipelined q8-cache decode attention only (attn_core<T > 5>: attn_core_w48q, or
    // attn_core_w48m for the low-bit MK_KV sides)
    if (n_draft > 4 && !p_->hb.kvq8) throw std::runtime_error("n_draft > 4 needs the q8 KV cache (--kv q8)");
    reset();
    if (const char * pol = getenv("MK_POLICY")) set_policy(pol);
#if MK_SPEC_LOG
    if (const char * f = getenv("MK_SPEC_LOG_FILE")) p_->slog = fopen(f, "a");
#endif
}

// "key=value,...": depth=0|1|2 (see Impl::Policy), dmin=N; on-device early stop (Launch::gate_*): drop=F,
// stop=F (every draft), dropK=F / stopK=F (draft K = 1 .. MAXD - 1), chain=0|1; h1=0|1 (sampled drafts when
// sampling); dq4=0|1 (MK_DRAFT_Q4=1 only: the draft head reads lm_head / its Q4_K copy; kept until changed);
// droplast=F (the deepest draft), dctx=N (from position N on at most 4 drafts),
// chain=2 (sampled drafts: the chain multiplies each drafted token's probability under q' instead of the top-1's);
// g7=0|1 (the depth > 5 gates, Impl::Policy); blk=0 only (blk=1, block verification of sampled drafts, is refused);
// "" = defaults
void MKEngine::set_policy(const std::string & spec) {
    Impl::Policy pol;
    float dk[MAXD - 1], sk[MAXD - 1]; // per-draft overrides
    for (int j = 0; j < MAXD - 1; j++) dk[j] = sk[j] = -1.f;
    auto draft_key = [](const std::string & k) { // dropK / stopK: K (1 .. MAXD - 1), else 0
        if (k.size() < 5 || k.size() > 6 || (k.compare(0, 4, "drop") != 0 && k.compare(0, 4, "stop") != 0)) return 0;
        int K = 0;
        for (size_t i = 4; i < k.size(); i++) {
            if (k[i] < '0' || k[i] > '9') return 0;
            K = 10 * K + (k[i] - '0');
        }
        return K >= 1 && K <= MAXD - 1 ? K : 0;
    };
    size_t i = 0;
    while (i < spec.size()) {
        size_t j = spec.find(',', i);
        if (j == std::string::npos) j = spec.size();
        const std::string kv = spec.substr(i, j - i);
        i = j + 1;
        if (kv.empty()) continue;
        const size_t e = kv.find('=');
        if (e == std::string::npos) throw std::runtime_error("policy: expected key=value, got " + kv);
        const std::string k = kv.substr(0, e), v = kv.substr(e + 1);
        const int dkey = draft_key(k);
        if (k == "drop" || k == "stop" || dkey)
            pol.g7 = 0; // explicit gates replace the depth > 5 defaults
        if (k == "depth") pol.depth = std::stoi(v);
        else if (k == "dmin") pol.dmin = std::stoi(v);
        else if (k == "drop") for (float & x : pol.gate_drop) x = std::stof(v);
        else if (k == "stop") for (float & x : pol.gate_stop) x = std::stof(v);
        else if (dkey) // dropK / stopK: draft K alone (e.g. drop4: the T = 5 verify row)
            (k[0] == 'd' ? dk : sk)[dkey - 1] = std::stof(v);
        else if (k == "droplast") pol.drop_last = std::stof(v), pol.g7 = 0; // the deepest draft (k = n_draft - 1)
        else if (k == "g7") pol.g7 = std::stoi(v);
        else if (k == "dstop") pol.dstop = std::stof(v); // (deep chain: g7's stop past draft 7)
        else if (k == "tier") pol.tier = std::stof(v);   // (deep chain: Policy::tier, 0 = always verify every draft)
        else if (k == "dctx") pol.dctx = std::stoi(v); // from position dctx on: at most 4 drafts
        else if (k == "chain") pol.gate_chain = std::stoi(v);
        else if (k == "h1") pol.h1 = std::stoi(v);
        else if (k == "blk") {
            if (std::stoi(v)) throw std::runtime_error("policy blk=" + v + " was removed (block verification of sampled drafts; blk=0 is the only value)");
        }
        else if (k == "mtpq") throw std::runtime_error("policy mtpq was removed: the MTP block is always re-quantised to Q4_K");
        else if (k == "dq4") p_->set_draft_q4(std::stoi(v)); // MK_DRAFT_Q4=1 only; from the next launch
        else throw std::runtime_error("policy: unknown key " + k);
    }
    for (int j = 0; j < MAXD - 1; j++) {
        if (dk[j] >= 0.f) pol.gate_drop[j] = dk[j];
        if (sk[j] >= 0.f) pol.gate_stop[j] = sk[j];
    }
    p_->pol = pol;
}

MKEngine::~MKEngine() { delete p_; }

void MKEngine::bench_phases() { p_->bench(); }

void MKEngine::set_penalty(float presence, float frequency) { p_->set_penalty(presence, frequency); }

void MKEngine::set_kv_emu(const std::string & spec) { p_->set_kv_emu(spec); }

void MKEngine::reset() {
    p_->pcnt_clear();
    p_->reset();
    p_->demoted = 0;
    p_->seq.clear();
    p_->spans_cut(0);
    for (auto & k : p_->ck) k.pos = -1;
    p_->dv_reset();
    stats = MKStats();
}

void MKEngine::cancel(bool on) { p_->pf_cancel.store(on ? 1 : 0); }

void MKEngine::prefill_progress(int * done, int * total) const {
    const int a = p_->pf_from.load(), b = p_->pf_to.load(), q = p_->pf_pos.load();
    *total = std::max(b - a, 0);
    *done = std::min(std::max(q - a, 0), *total);
}

int MKEngine::position() const { return p_->hctl.n; }

void MKEngine::set_n_draft(int n) {
    if (!p_->has_mtp && n > 0) { // (no MTP block: depth 0, a notice once)
        static bool told = false;
        if (!told) fprintf(stderr, "[mk] n_draft %d -> 0 (no MTP layer)\n", n);
        told = true;
        n = 0;
    }
    if (n < 0 || n > p_->n_draft_cap)
        throw std::runtime_error("set_n_draft: 0 .. " + std::to_string(p_->n_draft_cap) + " (the engine's construction depth)");
    p_->n_draft_max = p_->ndraft = n;
    p_->nd_avail = std::min(p_->nd_avail, n); // (the drafts the last launch made, not more: tok[] / dq past them are stale)
    n_draft_ = n;
}

int MKEngine::prefill_state(const std::vector<int> & tokens, const std::string & dir, int * how) {
    if (dir.empty()) throw std::runtime_error("prefill_state: no directory");
    struct Clear {
        Impl * p;
        ~Clear() { p->st_dir.clear(); }
    } clear{p_};
    p_->st_dir = dir;
    p_->st_how = 0;
    const int r = prefill(tokens);
    if (how) *how = p_->st_how;
    return r;
}

int MKEngine::prefill(const std::vector<int> & tokens, int ck_at) {
    return prefill(tokens, ck_at >= 0 ? std::vector<int>{ck_at} : std::vector<int>{}, ck_at);
}

int MKEngine::prefill(const std::vector<int> & tokens, const std::vector<int> & cks_in, int ck_exact) {
    return prefill(tokens, cks_in, ck_exact, {});
}

int MKEngine::prefill(const std::vector<int> & tokens, const std::vector<int> & cks_in, int ck_exact,
                      const std::vector<MKImage> & images_in) {
    struct PinClear { // (ck_pin's requests are for this call only, whatever happens to it)
        Impl * p;
        ~PinClear() { p->pin_req.clear(); }
    } pin_clear{p_};
    if (tokens.empty()) throw std::runtime_error("empty prompt");
    check_ids(tokens, p_->seq.size());
    std::vector<int> cks = cks_in;
    if (p_->exact && ck_exact >= 0) { // MK_EXACT: on the scan grid like the others (a side scan's state is not on it)
        cks.push_back(ck_exact);
        ck_exact = -1;
    }
    Ctl & c = p_->hctl;
    CK(cudaMemcpy(&c, p_->hb.ctl, sizeof(Ctl), cudaMemcpyDeviceToHost));
    std::vector<int> & seq = p_->seq;
    if ((int) (seq.size() + tokens.size()) + 8 > p_->ctx) throw std::runtime_error("prompt exceeds context");
    // Everything from the committed delta-net position on is fed: on a continued sequence that is
    // the last emitted token (not yet processed) and any accepted drafts not yet folded into the
    // recurrent state, followed by the new tokens.  Their KV / MTP rows are simply rewritten.
    const int n0 = c.dn_c;
    const int s0 = (int) seq.size(), end = s0 + (int) tokens.size(); // this prompt's positions: s0 .. end - 1
    const char * mode = getenv("MK_PREFILL");
    const bool batched = !(mode && std::string(mode) == "mk");
    const bool rows_only = p_->pf_mid && batched; // (returns -1: no output token)
    // Images, checked before anything changes. The spans after this prompt: the earlier ones before n0, then these.
    // An earlier span cut by n0 (a rewind to a checkpoint inside it) must come again (same position and grid): its rows
    // from n0 on are fed from it.
    using Span = Impl::Span;
    std::vector<Span> nsp;
    const Span * cut = nullptr;
    for (const Span & s : p_->spans)
        if (s.pos + s.n() <= n0) nsp.push_back(s);
        else if (s.pos < n0) cut = &s;
    std::vector<MKImage> images = images_in;
    std::sort(images.begin(), images.end(), [](const MKImage & a, const MKImage & b) { return a.pos < b.pos; });
    std::vector<int> erow; // the batched part's rows in emb_host (positions n0 .. end - 2), -1: token_embd
    if (!images.empty() || cut) {
        if (!batched || end - n0 < 2 || !p_->st_dir.empty())
            throw std::runtime_error("images need the batched prefill (not MK_PREFILL=mk, not prefill_state)");
        bool cut_ok = false;
        int prev = nsp.empty() ? 0 : nsp.back().pos + nsp.back().n();
        size_t rows = 0;
        for (const MKImage & m : images) {
            if (m.nx <= 0 || m.ny <= 0 || !m.embd) throw std::runtime_error("image: empty grid or no embedding rows");
            if (m.pos < prev) throw std::runtime_error("images overlap, or one starts inside an earlier one");
            if (m.pos + (long long) m.nx * m.ny > end - 1)
                throw std::runtime_error("an image must end before the prompt's last token");
            const int n = m.nx * m.ny; // (no overflow: checked against the prompt's end)
            if (m.pos < s0) { // continues the span cut by n0
                if (!cut || cut->pos != m.pos || cut->nx != m.nx || cut->ny != m.ny)
                    throw std::runtime_error("image at " + std::to_string(m.pos) + " starts before this prompt (" + std::to_string(s0) +
                                             ") and is not the image the sequence was rewound into");
                cut_ok = true;
            }
            nsp.push_back({m.pos, m.nx, m.ny});
            rows += (size_t) (m.pos + n - std::max(m.pos, n0));
            prev = m.pos + n;
        }
        if (cut && !cut_ok)
            throw std::runtime_error("the sequence was rewound to " + std::to_string(n0) + ", inside the image at " +
                                     std::to_string(cut->pos) + ": pass that image again");
        p_->emb_reserve(rows * D);
        erow.assign((size_t) (end - 1 - n0), -1);
        size_t r = 0;
        for (const MKImage & m : images) {
            const int a = std::max(m.pos, n0), n = m.pos + m.nx * m.ny - a;
            memcpy(p_->emb_host + r * D, m.embd + (size_t) (a - m.pos) * D, (size_t) n * D * sizeof(float));
            for (int q = 0; q < n; q++) erow[(size_t) (a - n0 + q)] = (int) (r + q);
            r += n;
        }
    }
    // MKEngine::cancel: a batched part stopped at a chunk boundary leaves a partial prefill in the KV rows, the recurrent
    // state and its checkpoint slots, so nothing of this call may count as processed. The checkpoints this call saved or
    // evicted are dropped, the last earlier one is restored (KV rows before it were never rewritten; without one: a
    // reset), the draft vocabulary and the image spans go back to their state before the call.
    struct ClearCancel {
        Impl * p;
        ~ClearCancel() { p->pf_cancel.store(0); }
    } clear_cancel{p_};
    const std::vector<Span> spans0 = p_->spans;
    const size_t dv0 = p_->dv_ids.size();
    const int dvn0 = p_->dv_n;
    std::vector<int> ck0;
    for (const auto & k : p_->ck) ck0.push_back(k.pos);
    auto stop_if_cancelled = [&] {
        if (!p_->pf_cancelled) return;
        p_->pf_cancelled = false;
        int best = -1;
        for (size_t k = 0; k < p_->ck.size(); k++) {
            Impl::CkSlot & s = p_->ck[k];
            if (s.pos != ck0[k]) s.pos = -1;
            else if (s.pos >= 0 && (best < 0 || s.pos > p_->ck[best].pos)) best = (int) k;
        }
        p_->spans = spans0;
        const int back = best >= 0 ? p_->ck[best].pos : 0;
        if (best >= 0) {
            p_->ck_restore(best);
            p_->spans_cut(back);
            if (back < s0 || dvn0 == s0) { // (as a fresh prefill of seq[0, back) has it)
                p_->dv_rebuild(back, p_->spans);
            } else {
                p_->dv_truncate(dv0);
                p_->dv_n = dvn0;
            }
        } else {
            reset();
        }
        throw MKCancelled("prefill of positions " + std::to_string(n0) + ".." + std::to_string(end - 1) + " cancelled at " +
                          std::to_string(p_->pf_stop) + "; the sequence is back at position " + std::to_string(back));
    };
    p_->pcnt_clear(); // a new response: the penalties count its generated tokens only
    if (p_->fault_at && ++p_->fault_seen == p_->fault_at) p_->fault_now = true;
    seq.insert(seq.end(), tokens.begin(), tokens.end());
    const bool dv_fresh = p_->dv_n == s0; // (after a reset, rewind or truncate: as a fresh prefill of seq[0, end) builds it)
    if (dv_fresh) {
        p_->dv_rebuild(end, nsp);
    } else if (images.empty()) {
        p_->dv_add(tokens.data(), tokens.size());
    } else { // (the draft vocabulary takes the prompt's text tokens, not the image rows' placeholder ids)
        std::vector<int> text;
        size_t k = 0;
        for (int q = s0; q < end; q++) {
            while (k < images.size() && images[k].pos + images[k].nx * images[k].ny <= q) k++;
            if (k == images.size() || q < images[k].pos) text.push_back(seq[q]);
        }
        p_->dv_add(text.data(), text.size());
    }
    if (!dv_fresh) p_->dv_n = -1;
    p_->spans = nsp;
    p_->rope_sync();
    const int * er = erow.empty() ? nullptr : erow.data();
    const std::vector<int> feed(seq.begin() + n0, seq.end());
    c.mtp_valid = std::min(c.mtp_valid, n0);
    size_t i = 0;
    bool end_save = !p_->exact; // a checkpoint at the end of the prompt, unless ck_exact is inside the batched part
    { // prefill_progress: this call's batched part (an empty range while it is set: total 0)
        const int to = rows_only ? n0 + (int) feed.size() : batched && feed.size() >= 2 ? n0 + (int) feed.size() - 1 : n0;
        p_->pf_to.store(n0);
        p_->pf_from.store(n0);
        p_->pf_pos.store(n0);
        p_->pf_to.store(to);
    }
    if (batched && (feed.size() >= 2 || rows_only)) {
        if (p_->exact && n0 % 32 != 0 && !p_->exact_off_grid) {
            p_->exact_off_grid = true;
            fprintf(stderr, "[mk] warning: MK_EXACT: a prefill from position %d, off the 32-token scan grid: its rows depend on "
                            "where it starts (not exact)\n", n0);
        }
        // all but the last token in batched kernels; the last one goes through the megakernel,
        // which produces the first output token and the MTP drafts (rows_only: every token in batched kernels)
        const std::vector<int> head(feed.begin(), feed.end() - (rows_only ? 0 : 1));
        const int hend = n0 + (int) head.size(); // the batched part: positions n0 .. hend - 1
        end_save = !p_->exact && !(ck_exact > n0 && ck_exact < hend); // (MK_EXACT: no megakernel state in checkpoints)
        // post-prefill state file (prefill_state): the batched part's result restored from the file, or computed and saved
        const bool st = !p_->st_dir.empty() && n0 == 0 && cks.empty() && ck_exact < 0;
        std::string st_path, st_key;
        if (st) p_->st_name(head, st_path, st_key);
        if (st && p_->st_load(st_path, st_key, hend)) {
            p_->st_how = 1;
        } else {
        // too few staging slots (MK_PF_CAP below 384): two pf_runs around ck_exact, other positions ignored (MK_EXACT:
        // its ck_exact joined cks, so no checkpoint is saved inside the prompt)
        if (p_->stg_n < 2) {
            if (ck_exact > n0 && ck_exact < hend) {
                p_->pf_run(std::vector<int>(head.begin(), head.begin() + (ck_exact - n0)), n0, nullptr, nullptr, 0, nullptr, er);
                stop_if_cancelled();
                c.dn_c = ck_exact;
                c.mtp_valid = c.dn_c;
                CK(cudaMemcpy(p_->hb.ctl, &c, sizeof(Ctl), cudaMemcpyHostToDevice));
                const int si = p_->ck_assign({ck_exact}, ck_exact, (int) seq.size())[0];
                if (si >= 0) p_->ck_save_end(si);
                p_->pf_run(std::vector<int>(head.begin() + (ck_exact - n0), head.end()), ck_exact, nullptr, nullptr, 0, nullptr,
                           er ? er + (ck_exact - n0) : nullptr);
                stop_if_cancelled();
            } else {
                p_->pf_run(head, n0, nullptr, nullptr, 0, nullptr, er);
                stop_if_cancelled();
            }
        } else {
            std::vector<int> pins; // (pin requests no checkpoint covers yet: this prefill saves them)
            for (int P : p_->pin_req)
                if (p_->pin_slot(P) < 0) pins.push_back(P);
            std::vector<char> wpin;
            const std::vector<int> want = p_->ck_positions(cks, ck_exact, n0, hend, pins, &wpin);
            const std::vector<int> slot = p_->ck_assign(want, ck_exact, (int) seq.size(), wpin);
            std::vector<Impl::Snap> snaps;
            for (size_t k = 0; k < want.size(); k++)
                if (slot[k] >= 0) snaps.push_back({want[k], slot[k]});
            const auto t0 = std::chrono::steady_clock::now();
            p_->pf_run(head, n0, nullptr, nullptr, 0, &snaps, er);
            stop_if_cancelled();
            for (size_t k = 0; k < want.size(); k++) { // (the state after position want[k] - 1)
                if (slot[k] < 0) continue;
                Impl::CkSlot & s = p_->ck[slot[k]];
                s.ctl = c;
                s.ctl.dn_c = s.ctl.mtp_valid = want[k];
                s.pos = want[k];
                s.pin = false;
                if (wpin[k]) p_->pin_set(slot[k], p_->pin_stamp);
            }
            if (p_->memlog && !snaps.empty()) {
                fprintf(stderr, "[mk-mem] prefill %d..%d with %zu checkpoints (", n0, hend, snaps.size());
                for (const auto & q : snaps) fprintf(stderr, " %d%s", q.pos, p_->ck[q.slot].pin ? " (pinned)" : "");
                fprintf(stderr, " ) in %.1f ms\n", std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
            }
        }
        if (st) {
            p_->st_save(st_path, st_key, hend);
            p_->st_how = 2;
        }
        }
        i = head.size();
        c.dn_c = hend;
        c.mtp_valid = c.dn_c;
    }
    if (rows_only) { // (the next chunk continues from here: its prefill reads the control block back)
        c.n = c.dn_c;
        CK(cudaMemcpy(p_->hb.ctl, &c, sizeof(Ctl), cudaMemcpyHostToDevice));
        CK(cudaGetLastError());
        return -1;
    }
    while (i < feed.size()) {
        const int T = att_cap_T(c.dn_c, (int) std::min<size_t>(MAXT_PF, feed.size() - i));
        const bool last = i + T == feed.size();
        c.n = c.dn_c;
        for (int k = 0; k < T; k++) c.tok[k] = feed[i + k];
        CK(cudaMemcpy(p_->hb.ctl, &c, sizeof(Ctl), cudaMemcpyHostToDevice));
        p_->ndraft = p_->depth_top(c.n);
        p_->launch(MODE_PREFILL, T, last ? 1 : 0);
        CK(cudaMemcpy(&c, p_->hb.ctl, sizeof(Ctl), cudaMemcpyDeviceToHost));
        i += T;
    }
    p_->nd_avail = p_->ndraft > 0 ? std::min(p_->ndraft, c.nd_prod) : 0;
    p_->all_acc = true;
    p_->last_acc = p_->nd_avail;
    p_->stale_replay = p_->stale_pcnt = false;
    CK(cudaGetLastError());
#if MK_SPEC_LOG
    p_->slog_drafts();
#endif
    if (end_save && !p_->pf_mid && p_->ck_find(c.dn_c) < 0) {
        const int si = p_->ck_assign({c.dn_c}, c.dn_c, c.dn_c)[0];
        if (si >= 0) p_->ck_save_end(si);
    }
    seq.push_back(c.out[0]);
#if MK_SPEC_LOG
    p_->slog_prompt(n0);
#endif
    return c.out[0];
}

void MKEngine::set_sampling(float temp, float top_p, int top_k, float min_p, unsigned long long seed) {
    if (temp > 0.f && (top_k <= 0 || top_k > TOPK_MAX))
        throw std::runtime_error("sampling needs 1 <= top_k <= " + std::to_string(TOPK_MAX));
    p_->smp_temp = temp;
    p_->smp_top_p = top_p;
    p_->smp_top_k = top_k;
    p_->smp_min_p = min_p;
    p_->smp_seed = seed;
}

const std::vector<int> & MKEngine::sequence() const { return p_->seq; }
int MKEngine::committed() const { return p_->hctl.dn_c; }
std::vector<int> MKEngine::checkpoints() const {
    std::vector<int> v;
    for (const auto & k : p_->ck)
        if (k.pos >= 0) v.push_back(k.pos);
    std::sort(v.begin(), v.end());
    return v;
}
int MKEngine::checkpoint_pos() const {
    const std::vector<int> v = checkpoints();
    return v.empty() ? -1 : v.back();
}
bool MKEngine::rewind_to(int pos) {
    const int si = p_->ck_find(pos);
    if (si < 0 || pos > (int) p_->seq.size()) return false;
    p_->ck_restore(si);
    p_->spans_cut(pos);
    p_->dv_rebuild(pos, p_->spans); // (the abandoned continuation's tokens leave the draft vocabulary)
    return true;
}
bool MKEngine::rewind_to_checkpoint() { return rewind_to(checkpoint_pos()); }

int MKEngine::ck_pin(const std::vector<int> & positions) {
    Impl & I = *p_;
    I.pin_req.clear();
    if (I.pin_max <= 0) return 0;
    const unsigned t = I.pin_stamp = ++I.pin_clock;
    int n = 0;
    for (size_t k = 0; k < positions.size() && (int) k < I.pin_max; k++) {
        const int si = I.pin_slot(positions[k]);
        if (si >= 0) {
            I.pin_set(si, t);
            n++;
        } else if (positions[k] > 0) {
            I.pin_req.push_back(positions[k]);
        }
    }
    return n;
}

std::vector<int> MKEngine::ck_pinned() const {
    std::vector<int> v;
    for (const auto & k : p_->ck)
        if (k.pos >= 0 && k.pin) v.push_back(k.pos);
    std::sort(v.begin(), v.end());
    return v;
}

void MKEngine::truncate(int len) {
    std::vector<int> & seq = p_->seq;
    if (len < p_->hctl.dn_c || len > (int) seq.size())
        throw std::runtime_error("truncate: can only drop tokens emitted after the committed position");
    p_->ring_rewind(len); // MK_KV_WIN: the dropped positions' ring slots held window rows
    seq.resize(len);
    p_->spans_cut(len);
    p_->dv_rebuild(len, p_->spans);
}

std::vector<int> MKEngine::score(const std::vector<int> & tokens) {
    check_ids(tokens, 0);
    reset();
    std::vector<int> pred;
    const char * mode = getenv("MK_PREFILL");
    if (!(mode && std::string(mode) == "mk")) {
        p_->pf_run(tokens, 0, &pred);
        reset();
        return pred;
    }
    Ctl & c = p_->hctl;
    CK(cudaMemcpy(&c, p_->hb.ctl, sizeof(Ctl), cudaMemcpyDeviceToHost));
    size_t i = 0;
    while (i < tokens.size()) {
        const int T = att_cap_T((int) i, (int) std::min<size_t>(MAXT_PF, tokens.size() - i));
        c.n = (int) i;
        for (int k = 0; k < T; k++) c.tok[k] = tokens[i + k];
        CK(cudaMemcpy(p_->hb.ctl, &c, sizeof(Ctl), cudaMemcpyHostToDevice));
        p_->launch(MODE_PREFILL, T, 0, 1);
        CK(cudaMemcpy(&c, p_->hb.ctl, sizeof(Ctl), cudaMemcpyDeviceToHost));
        for (int k = 0; k < T; k++) pred.push_back(c.tgt[k]);
        i += T;
    }
    return pred;
}

int MKEngine::n_vocab() { return VOCAB; }
int MKEngine::n_embd() { return D; }
int MKEngine::warm_launch(void * stream, const unsigned * stop, double ms, int blocks, int threads, int spin) {
    if (blocks < 1 || threads < 1 || threads > 1024 || !(ms > 0.0)) return (int) cudaErrorInvalidValue;
    mkl::kw_warm(blocks, threads, stop, (unsigned long long) (ms * 1e6), spin, (cudaStream_t) stream);
    return (int) cudaGetLastError();
}

void MKEngine::score_logits(const std::vector<int> & tokens, int mk_from, int first, const LogitSink & sink) {
    score_logits(tokens, mk_from, first, sink, {});
}

void MKEngine::score_logits(const std::vector<int> & tokens, int mk_from, int first, const LogitSink & sink,
                            const std::vector<MKImage> & images) {
    const int N = (int) tokens.size();
    if (N + 8 > p_->ctx) throw std::runtime_error("sequence exceeds context");
    check_ids(tokens, 0);
    reset();
    p_->logits_alloc();
    mk_from = std::min(std::max(mk_from, 0), N);
    std::vector<int> erow;
    if (!images.empty()) { // (the batched part's rows, as prefill stages them)
        size_t rows = 0;
        int prev = 0;
        std::vector<MKImage> im = images;
        std::sort(im.begin(), im.end(), [](const MKImage & a, const MKImage & b) { return a.pos < b.pos; });
        for (const MKImage & m : im) {
            if (m.nx <= 0 || m.ny <= 0 || !m.embd || m.pos < prev || m.pos + m.nx * m.ny > mk_from)
                throw std::runtime_error("score_logits: images must be disjoint, inside the batched part (mk_from)");
            prev = m.pos + m.nx * m.ny;
            rows += (size_t) m.nx * m.ny;
            p_->spans.push_back({m.pos, m.nx, m.ny});
        }
        p_->rope_sync();
        p_->emb_reserve(rows * D);
        erow.assign((size_t) mk_from, -1);
        size_t r = 0;
        for (const MKImage & m : im) {
            const int n = m.nx * m.ny;
            memcpy(p_->emb_host + r * D, m.embd, (size_t) n * D * sizeof(float));
            for (int q = 0; q < n; q++) erow[(size_t) m.pos + q] = (int) (r + q);
            r += n;
        }
    }
    if (mk_from > 0)
        p_->pf_run(std::vector<int>(tokens.begin(), tokens.begin() + mk_from), 0, nullptr, first < mk_from ? &sink : nullptr, first,
                   nullptr, erow.empty() ? nullptr : erow.data());
    Ctl & c = p_->hctl;
    CK(cudaMemcpy(&c, p_->hb.ctl, sizeof(Ctl), cudaMemcpyDeviceToHost));
    c.dn_c = mk_from;
    std::vector<float> rows((size_t) MAXT * VOCAB);
    for (int i = mk_from; i < N;) {
        const int T = att_cap_T(i, std::min(MAXT_PF, N - i));
#if MK_KV_EMU
        p_->kv_demote_to(i - p_->hb.kvemu_win); // recent-window emulation
#endif
        c.n = i;
        for (int k = 0; k < T; k++) c.tok[k] = tokens[i + k];
        CK(cudaMemcpy(p_->hb.ctl, &c, sizeof(Ctl), cudaMemcpyHostToDevice));
        p_->launch(MODE_PREFILL, T, 0, 2);
        CK(cudaMemcpy(&c, p_->hb.ctl, sizeof(Ctl), cudaMemcpyDeviceToHost));
        const int s = std::max(0, first - i);
        if (s < T) {
            CK(cudaMemcpy(rows.data(), p_->hb.logits + (size_t) s * VOCAB, (size_t) (T - s) * VOCAB * sizeof(float), cudaMemcpyDeviceToHost));
            sink(i + s, T - s, rows.data());
        }
        i += T;
    }
    reset();
}

int MKEngine::step(std::vector<int> & out) {
    const int T = p_->step_live(out);
    const Ctl & c = p_->hctl;
    stats.cycles++;
    stats.drafted += T - 1;
    stats.accepted += c.n_acc;
    stats.last_T = T;
    stats.last_nd = p_->nd_avail;
    stats.last_acc = c.n_acc;
    return c.n_out;
}

// MKEngine::step's body on the live sequence; returns the launch's T
int MKEngine::Impl::step_live(std::vector<int> & out) {
    Ctl & c = hctl;
    if (c.n + MAXD + 2 >= ctx) throw std::runtime_error("context full"); // (rows past ctx: KV_PAD)
    // (after book_set: kvgb is read for accepted drafts not folded in, dq to verify sampled drafts, pcnt for penalties)
    if ((stale_replay && (c.dn_c < c.n || (nd_avail > 0 && c.dsmp && smp_temp > 0.f))) ||
        (stale_pcnt && (pen_pres != 0.f || pen_freq != 0.f)))
        throw std::runtime_error("step: the sequence was brought back (unpark / side_end) without the state this "
                                 "step reads (DeltaNet replay rows, the drafts' q', penalty counts): prefill first");
    // verify the previous launch's drafts, minus any that would put this launch's rows across an
    // attention-partition boundary (once per ATT_BUCKET positions; keeps greedy output depth-independent)
    // T >= 6 runs as one MAXT-row pass (padding rows): the whole n + 1 .. n + MAXT must lie in one partition, else at most 5
    int T = att_cap_T(c.n, 1 + nd_avail);
#if MK_MAXD > 8
    if (T > MAXT && pol.tier > 0.f) { // the deep pass only when drafts MAXT.. are worth it (Policy::tier)
        float cp = 1.f, e = 0.f;
        if (c.dsmp && smp_temp > 0.f) {
            // sampled drafts, sampled verify: dconf[k >= MAXT] (pass k is fed draft k) and the draft count follow
            // draft MAXT's own value, and verifying drafts MAXT.. by them biases the output (as chain=2's drop gates:
            // launch). Only what was fixed before draft MAXT was drawn: dconf[0 .. MAXT - 1], the last repeated up to
            // the drafting launch's depth
            for (int k = 0; k < MAXT; k++) cp *= c.dconf[k];
            for (int k = MAXT - 1; k < ndraft; k++, cp *= c.dconf[MAXT - 1]) e += cp;
        } else {
            for (int k = 0; k < T - 1; k++) {
                cp *= c.dconf[k];
                if (k >= MAXT - 1) e += cp;
            }
        }
        if (e < pol.tier) T = MAXT;
    }
#endif
    if (T > MAXT && att_cap_T(c.n, MAXD) < MAXD) T = MAXT; // (the deep pass: n + 1 .. n + MAXD in one partition)
    if (T > 5 && att_cap_T(c.n, MAXT) < MAXT) T = 5;
    // Draft depth (an upper bound: the on-device gate may stop drafting earlier), by Policy::depth (default 1: the
    // full depth). Depth 0: after a cycle that rejected a draft, draft one token less (but at least 3), since a deeper
    // draft is then unlikely to be accepted; the full depth again after a fully accepted cycle.
    const int top = depth_top(c.n); // n_draft (at most 4 from Policy::dctx on)
    int nd = top;
    if (pol.depth == 0) nd = (!all_acc && top > 3) ? top - 1 : top;
    else if (pol.depth == 2) nd = std::min(top, std::max(std::min(pol.dmin, top), 1 + last_acc));
    ndraft = nd;
#if MK_SPEC_LOG
    const int n0 = c.n;
    int drafts[MAXD];
    for (int i = 1; i < T; i++) drafts[i] = c.tok[i];
#endif
    launch(MODE_DECODE, T, 0);
    ctl_fetch(c);
    stale_replay = false; // (this launch wrote the replay rows and q' the next one reads)
    nd_avail = nd > 0 ? std::min(nd, c.nd_prod) : 0;
#if MK_SPEC_LOG
    slog_cycle(n0, T, drafts, c);
#endif
    for (int i = 0; i < c.n_out; i++) out.push_back(c.out[i]);
    seq.insert(seq.end(), c.out, c.out + c.n_out);
    dv_add(c.out, c.n_out);
    dv_n = -1;
    all_acc = c.n_acc == T - 1;
    last_acc = c.n_acc;
    return T;
}

std::vector<float> MKEngine::last_hidden() {
    std::vector<float> h(D);
    const int pos = p_->hctl.dn_c - 1;
    p_->ck_wait_restore();
    CK(cudaMemcpy(h.data(), p_->hb.h_hist + (size_t) (pos % H_RING) * D, D * 4, cudaMemcpyDeviceToHost));
    return h;
}

bool MKEngine::exact() const { return p_->exact; }

std::vector<int> MKEngine::draft_vocab() {
    const Impl & I = *p_;
    if (I.dv_in.empty()) return {};
    int n = 0;
    CK(mkl::get_cW(&n, sizeof(int), offsetof(ModelW, head_d) + offsetof(Mat, N)));
    std::vector<int> v((size_t) std::max(n, 0));
    if (n > 0) CK(cudaMemcpy(v.data(), I.hw.draft_ids, (size_t) n * sizeof(int), cudaMemcpyDeviceToHost));
    if (v != I.dv_ids || I.hw.head_d.N != n) throw std::runtime_error("draft vocabulary: the device's rows differ from the host's list");
    return v;
}

bool MKEngine::has_mtp() const { return p_->has_mtp; }

// state_digest (test helper for the exact-reuse probes; slow, host-side hashing): 64-bit words
//   [0] 0x4d4b444701, [1] n = committed position (dn_c), [2] mtp_valid, [3] kvrc (0: no MK_KV_WIN ring),
//   [4] ring_lo = max(0, n - kvrc), [5] N_ATT_SLOTS, [6] NKV, [7] H_RING; then
//   per (slot, w = K / V, kv head g, position p < n): its row in the main buffer (values + scales);
//   per (slot, w, g, position ring_lo <= p < n): its q8 ring row (0: the side has no ring);
//   48 DeltaNet states, 48 conv rings, H_RING h ring rows (ring index order).
static unsigned long long dg_hash(const void * p, size_t nb, unsigned long long h = 0xcbf29ce484222325ULL) {
    const unsigned char * b = static_cast<const unsigned char *>(p);
    size_t i = 0;
    for (; i + 8 <= nb; i += 8) {
        unsigned long long w;
        memcpy(&w, b + i, 8);
        h = (h ^ w) * 0x100000001b3ULL;
        h ^= h >> 31;
    }
    for (; i < nb; i++) h = (h ^ b[i]) * 0x100000001b3ULL;
    return h;
}
std::vector<unsigned long long> MKEngine::state_digest() {
    Impl & I = *p_;
    I.ck_wait_restore();
    I.ring_snap_guard();
    CK(cudaDeviceSynchronize());
    Ctl c;
    CK(cudaMemcpy(&c, I.hb.ctl, sizeof(Ctl), cudaMemcpyDeviceToHost));
    typedef unsigned long long u64;
    const int n = c.dn_c, rc = I.hb.kvwin > 0 ? I.hb.kvrc : 0, rlo = rc ? std::max(0, n - rc) : n;
    std::vector<u64> out = {0x4d4b444701ULL, (u64) n, (u64) c.mtp_valid, (u64) rc, (u64) rlo, (u64) I.kv_slots, (u64) NKV, (u64) H_RING};
    const size_t ctx = (size_t) I.hb.ctx, sbq = (HD / 32) * sizeof(__half); // (row stride per kv head)
    std::vector<unsigned char> vb, sb;
    auto d2h = [&](std::vector<unsigned char> & dst, const void * src, size_t nb) {
        dst.resize(std::max<size_t>(1, nb));
        if (nb) CK(cudaMemcpy(dst.data(), src, nb, cudaMemcpyDeviceToHost));
    };
    for (int s = 0; s < I.kv_slots; s++)
        for (int w = 0; w < 2; w++)
            for (int g = 0; g < NKV; g++) {
                size_t rb, srb = 0;
                if (I.hb.kvq8) {
                    rb = (size_t) HD * I.hb.kvb[w][s] / 8;
                    d2h(vb, I.hb.kvp[w][s] + (size_t) g * ctx * rb, n * rb);
                    srb = sbq;
                    d2h(sb, I.hb.kvs[w][s] + (size_t) g * ctx * (HD / 32), n * srb);
                } else {
                    rb = (size_t) HD * sizeof(__half);
                    d2h(vb, static_cast<const __half *>(w ? I.hb.vc : I.hb.kc) + ((size_t) s * NKV + g) * ctx * HD, n * rb);
                }
                for (int p = 0; p < n; p++) out.push_back(dg_hash(sb.data() + p * srb, srb, dg_hash(vb.data() + p * rb, rb)));
            }
    for (int s = 0; s < I.kv_slots; s++)
        for (int w = 0; w < 2; w++)
            for (int g = 0; g < NKV; g++) {
                if (!rc || !I.hb.kvrp[w][s]) {
                    out.insert(out.end(), n - rlo, 0ULL);
                    continue;
                }
                d2h(vb, I.hb.kvrp[w][s] + (size_t) g * rc * HD, (size_t) rc * HD);
                d2h(sb, I.hb.kvrs[w][s] + (size_t) g * rc * (HD / 32), (size_t) rc * sbq);
                for (int p = rlo; p < n; p++) {
                    const size_t r = p % rc;
                    out.push_back(dg_hash(sb.data() + r * sbq, sbq, dg_hash(vb.data() + r * HD, HD)));
                }
            }
    for (int l = 0; l < 48; l++) {
        d2h(vb, I.hb.dstate + l * Impl::CK_SL, Impl::CK_SL * 4);
        out.push_back(dg_hash(vb.data(), Impl::CK_SL * 4));
    }
    for (int l = 0; l < 48; l++) {
        d2h(vb, I.hb.conv + l * Impl::CK_CL, Impl::CK_CL * 4);
        out.push_back(dg_hash(vb.data(), Impl::CK_CL * 4));
    }
    for (int r = 0; r < H_RING; r++) {
        d2h(vb, I.hb.h_hist + (size_t) r * D, (size_t) D * 4);
        out.push_back(dg_hash(vb.data(), (size_t) D * 4));
    }
    return out;
}
