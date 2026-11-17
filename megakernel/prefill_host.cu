// MKEngine::Impl: the batched prefill's host side (chunks, the GEMM / attention launches, checkpoints on the way).

#include "engine_impl.h"

namespace {

// Host: work split for a chunk of T queries at n0 on `slots` resident CTAs.  The items of the last, partial
// wave (the highest q tiles of the last kv heads) are split into S tile ranges when that shortens the tail;
// S = 1 / nfull = all items: no split.  part_floats: capacity of the partials buffer.
inline void a2_plan(int T, int n0, int slots, size_t part_floats, int & nfull, int & S) {
    const int nqt = (T + A2_Q - 1) / A2_Q, N = nqt * NKV, rem = slots > 0 ? N % slots : 0;
    nfull = N;
    S = 1;
    if (rem == 0) return;
    const int qt = (N - rem) % nqt; // smallest tail item
    const int ntile_min = (n0 + qt * A2_Q + std::min(A2_Q, T - qt * A2_Q) - 1) / A2_K + 1;
    const int smax = std::min(std::min(32, ntile_min / 2), (int) std::min<size_t>(1024, part_floats / ((size_t) A2_PART * rem)));
    double best = 1.0; // tail length in item times
    for (int s = 2; s <= smax; s++) {
        const double t = (double) ((rem * s + slots - 1) / slots) / s;
        if (t < best - 0.02) {
            best = t;
            S = s;
        }
    }
    if (S > 1) nfull = N - rem;
}

} // namespace

// MK_PF_CAP: tokens per prefill chunk (multiple of 128); scratch ~0.25 MB per token (MK_EXACT, every gate / up fused)
int MKEngine::Impl::pf_cap_req() const {
    const char * e = getenv("MK_PF_CAP");
    return e ? std::max(128, std::min(2048, atoi(e) / 128 * 128)) : 2048;
}

// the chunk pf_init allocates for: pf_cap_req, but no longer than the KV window rings (kvrc rows) when MK_EXACT checkpoints
// keep them (ring_ck), whose ring_snap_rows runs after a whole chunk, which must not wrap a ring
int MKEngine::Impl::pf_cap_eff(bool ring_ck, int kvrc) const {
    const int cap = pf_cap_req();
    return ring_ck && cap > kvrc ? kvrc / 128 * 128 : cap;
}

// pf_init's scratch in bytes at `cap` tokens (ex: MK_EXACT), dalloc's 16-byte pads included: keep in step with pf_init (the
// loader compares the two after pf_init when MK_RESERVE_EARLY is on)
size_t MKEngine::Impl::pf_scratch_bytes(int cap, bool ex) {
    bool all_fused = true;
    for (int L = 0; L < NL; L++) all_fused &= glu_fused(hw.L[L].fg, hw.L[L].fu) || (glu_mix_on && glu_mix(hw.L[L].fg, hw.L[L].fu, 0));
    const bool y1full = getenv("MK_PF_Y1") && atoi(getenv("MK_PF_Y1")) == 1;
    const size_t c = cap;
    return c * D * 4 * 2 + c * 4 + c * 2 * D * 2 + c * FF * 2 + c * (all_fused ? QG : FF) * 4 +
           c * (ex && all_fused && !y1full ? VD : FF) * 4 + c * VD * 4 + c * QKV * 2 + c * 2 * NVH * 4 + (size_t) 8 * c * 2 * NVH * 4 +
           (c + 1) * D * 4 + c * NH * HD * 2 + 3 * c * 4 + 16 * 16;
}

void MKEngine::Impl::pf_kernel_attrs() {
    mmh_attr<pf::TQ4K, false>();
    mmh_attr<pf::TQ5K, false>();
    mmh_attr<pf::TQ6K, false>();
    mmh_attr<pf::TQ80, false>();
    mmh_attr<pf::TIQ4, false>();
    mmh_attr<pf::TIQ4NL, false>();
    mmh_attr<pf::TQ3K, false>();
    mmh_attr<pf::TIQ3S, false>();
    mmh_attr<pf::TQ4K, true>();
    mmh_attr<pf::TQ5K, true>();
    mmh_attr<pf::TIQ4, true>();
    mmh_attr<pf::TQ4K, false, 1>();
    mmh_attr<pf::TQ5K, false, 1>();
    mmh_attr<pf::TQ80, false, 1>();
    mmh_attr<pf::TIQ4, false, 1>();
    mmh_attr<pf::TIQ4NL, false, 1>();
    mmh_attr<pf::TQ4K, true, 1>();
    mmh_attr<pf::TQ5K, true, 1>();
    mmh_attr<pf::TIQ4, true, 1>();
    mmh_attr<pf::TQ6K, false, 1>();
    mmh_attr<pf::TQ3K, false, 1>();
    mmh_attr<pf::TIQ3S, false, 1>();
    glu_mix_attr();
}

void MKEngine::Impl::pf_init() {
    if (pf_ready) return;
    const int req = pf_cap_req(), cap = pf_cap_eff(ring_bytes_ck > 0, hb.kvrc);
    if (const char * e = getenv("MK_PF_CAP"); e && atoi(e) != req)
        fprintf(stderr, "[mk] MK_PF_CAP %s -> %d (a multiple of 128, 128 .. 2048)\n", e, req);
    if (cap != req)
        fprintf(stderr, "[mk] MK_PF_CAP %d -> %d: MK_EXACT checkpoints keep the KV window rings, %d rows (MK_KV_WIN=%d), and a "
                        "prefill chunk must not wrap them\n", req, cap, hb.kvrc, hb.kvwin);
    hp.cap = cap;
    mem_mark("before prefill scratch");
    hp.x = dalloc<float>((size_t) cap * D);
    hp.xm = dalloc<float>((size_t) cap * D);
    hp.rs = dalloc<float>(cap);
    hp.xh = dalloc<__half>((size_t) cap * 2 * D); // GEMM inputs: K = D, VD (attention / delta-net out) or 2 D (MTP eh)
    hp.xh2 = dalloc<__half>((size_t) cap * FF);
    bool all_fused = true;
    for (int L = 0; L < NL; L++) all_fused &= glu_fused(hw.L[L].fg, hw.L[L].fu) || (glu_mix_on && glu_mix(hw.L[L].fg, hw.L[L].fu, 0));
    // y0: [cap][FF] for the two-GEMM gate / up path (gate's output); every pair fused: QG, the most its other users need
    y0_floats = (size_t) cap * (all_fused ? QG : FF);
    hp.y0 = dalloc<float>(y0_floats);
    // y1: [cap][FF] for the two-GEMM gate / up path (up's fp32 output) and as a2_plan's tail-split partials (non-exact
    // mode: its capacity bounds the split count, i.e. the numerics, so it stays FF there). In MK_EXACT mode with every
    // gate / up pair fused (same-type or V2 mixed), y1's other users need VD floats per token (DeltaNet z; attention
    // k / v, 2 KVD; EXA partials, whose split is numerically free): 45 KB less per token. MK_PF_Y1=1: always FF.
    const bool y1full = getenv("MK_PF_Y1") && atoi(getenv("MK_PF_Y1")) == 1;
    y1_floats = (size_t) cap * (exact && all_fused && !y1full ? VD : FF);
    hp.y1 = dalloc<float>(y1_floats);
    hp.y2 = dalloc<float>((size_t) cap * VD);
    hp.dq = dalloc<__half>((size_t) cap * QKV);
    hp.gb = dalloc<float>((size_t) cap * 2 * NVH);
    hp.ab = dalloc<float>((size_t) 8 * cap * 2 * NVH);
    hp.h = dalloc<float>((size_t) (cap + 1) * D);
    hp.qh = dalloc<__half>((size_t) cap * NH * HD);
    hp.tok = dalloc<int>(cap);
    hp.erow = dalloc<int>(cap);
    hp.emb = nullptr; // (emb_reserve)
    hp.amax = dalloc<int>(cap);
    hp.wsrow = exact ? 1 : 0;
    mem_mark("prefill scratch");
    CK(mkl::set_cP(&hp, sizeof(PBufs)));
    pf_kernel_attrs();
    if (const char * e = getenv("MK_PF_BN")) { // force one GEMM token tile (tuning / tests); 0 = by prompt size
        pf_bn_force = atoi(e);
        if (pf_bn_force != 0 && pf_bn_force != 32 && pf_bn_force != 64 && pf_bn_force != 128)
            throw std::runtime_error("MK_PF_BN: 0, 32, 64 or 128");
    }
    mkl::pk_attn_attr();
    int dev = 0, nsm = 0, occ = 0;
    CK(cudaGetDevice(&dev));
    CK(cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev));
    mkl::pk_attn2_occupancy(&occ);
    a2_slots = nsm * occ;
    // MK_ATTN=1: pk_attn; MK_ATTN_SPLIT=0: no tail split; MK_ATTN_PV16=0: fp32-accumulated P V
    // (pk_attn2 with both off is bit-identical to pk_attn)
    if (const char * e = getenv("MK_ATTN")) attn_old = atoi(e) == 1;
    if (const char * e = getenv("MK_ATTN_SPLIT")) attn_split = atoi(e) != 0;
    if (const char * e = getenv("MK_ATTN_PV16")) attn_pv16 = atoi(e) != 0;
    for (int w = 0; w < 2; w++)
        for (int s = 0; s < N_ATT_SLOTS; s++)
            if (attn_old && hb.kvb[w][s] != 8) throw std::runtime_error("MK_ATTN=1 (the original prefill attention) reads q8_0 only: no MK_KV");
    // MK_ATTN_S8 (q8 cache, with PV16): 1 = int8 Q K^T, Q scales per 32-dim block (pk_attn8); 2 = per row; 0 = off
    if (const char * e = getenv("MK_ATTN_S8")) attn_s8 = atoi(e);
    if (attn_s8) {
        int occ8 = 0;
        mkl::pk_attn8_occupancy(&occ8);
        if (occ8 != occ) attn_s8 = 0; // the split plan assumes pk_attn2's residency
    }
    // MK_EXACT: pk_attn8's EXA form (absolute key segments of MK_EXACT_SEG positions, default 4096); other prefill
    // attention kernels run unsplit (one online softmax per row over absolute tiles)
    if (exact) {
        attn_split = false;
        if (const char * e = getenv("MK_EXACT_SEG")) ex_segt = std::max(1, atoi(e) / A2_K);
        if (const char * e = getenv("MK_EXACT_SPLIT")) ex_split = atoi(e) != 0;
        mkl::pk_attn8_exa_attr();
    }
    // MK_PFX (bits; a set bit keeps the old path): 1 pf_fork off, 2 pk_dn_scan (no pk_dn_tri / pk_dn_scan4), 4 pk_ab
    // after the DN in-projection join (not before it), 8 attention k on pf_s2 and v on pf_s3 next to q (not k then q on
    // stream 0, v on pf_s2), 16 pk_ab (not pk_ab2), 64 pk_attn8 (not pk_attn9), 128 pk_conv_ring's float loop (not
    // float4), 256 pk_attn2_comb with 12 rows per warp (not one)
    if (const char * e = getenv("MK_PFX")) pfx_bits = atoi(e);
    mkl::pk_attn_old = pfx_bits & 64 ? 1 : 0;
    pf_fork = !(pfx_bits & 1);
#if MK_PF_INFCHK
    pf_fork = false; // (chk_set's site / layer are stream-ordered on stream 0)
#endif
    if ((k8 & 1) || pf_fork) { // MK_K8 bit 1: pk_ab's stream; pf_fork: the side GEMMs' (non-blocking: ordered only by events)
        CK(cudaStreamCreateWithFlags(&pf_s2, cudaStreamNonBlocking));
        CK(cudaEventCreateWithFlags(&pf_e0, cudaEventDisableTiming));
        CK(cudaEventCreateWithFlags(&pf_e1, cudaEventDisableTiming));
    }
    if (pf_fork) {
        CK(cudaStreamCreateWithFlags(&pf_s3, cudaStreamNonBlocking));
        CK(cudaEventCreateWithFlags(&pf_e2, cudaEventDisableTiming));
    }
    if (const char * e = getenv("MK_PF_SYNC")) pf_sync = atoi(e) != 0;
    if (const char * e = getenv("MK_PF_GAPS")) pf_gaps = atoi(e) != 0;
    for (cudaEvent_t & e : pf_cev) CK(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
    mem_mark("prefill kernel attributes");
    pf_ready = true;
}

auto MKEngine::Impl::ex_plan(int T, int n0, std::pair<size_t, size_t> cap) -> ExPlan {
    const int nqt = (T + A2_Q - 1) / A2_Q, N = nqt * NKV, P = std::max(1, a2_slots), segt = ex_segt;
    if (!ex_split) return {N, 1, 0, 0};
    const auto key = std::make_tuple(T, n0, cap.first, cap.second);
    const auto f = ex_memo.find(key);
    if (f != ex_memo.end()) return f->second;
    if (ex_memo.size() > 4096) ex_memo.clear();
    // costs in key tiles (~3.2 us of one CTA): c0 per CTA (Q fragments, pipeline fill); cf per fold of a segment into R
    // (R.o read + write, 192 KB at a resident CTA's share of DRAM bandwidth); cw per partial slot written; cc per slot
    // read by pk_attn8_combx (the whole GPU's bandwidth, after the attention kernel)
    const double c0 = 2.0, cf = 4.0, cw = 3.0, cc = 0.04;
    std::vector<int> nt(N); // key tiles of each item
    for (int i = 0; i < N; i++) {
        const int t0 = i % nqt * A2_Q;
        nt[i] = (n0 + t0 + std::min(A2_Q, T - t0) - 1) / A2_K + 1;
    }
    std::vector<double> fr; // the slots' busy-until times (min-heap)
    auto sim = [&](int nfull, int S, int h) {
        fr.assign(P, 0.0);
        auto run = [&](double d) {
            std::pop_heap(fr.begin(), fr.end(), std::greater<double>());
            fr.back() += d;
            std::push_heap(fr.begin(), fr.end(), std::greater<double>());
        };
        for (int i = 0; i < nfull; i++) run(nt[i] + c0 + cf * ((nt[i] - 1) / segt));
        const int ns = N - nfull;
        for (int u = 0; u < (ns > 0 ? S : 0); u++)
            for (int j = 0; j < ns; j++) {
                const int t = nt[nfull + j], k0 = u == 0 ? 0 : h + u - 1, k1 = u == 0 ? h : k0 + 1;
                if (k0 * segt >= t) {
                    run(0.1); // (idle unit)
                    continue;
                }
                const int e = std::min(t, k1 * segt);
                run(e - k0 * segt + c0 + cw + (u == 0 ? cf * ((e - 1) / segt) : 0.0));
            }
        // + pk_attn8_combx: one more launch, reads S partial slots per split item
        return *std::max_element(fr.begin(), fr.end()) + (ns > 0 ? c0 + cc * ns * S : 0.0);
    };
    ExPlan best{N, 1, 0, 0};
    double bt = sim(N, 1, 0);
    const int rem = N % P;
    for (const int ns : {rem, rem + P}) {
        if (ns <= 0 || ns > N) continue;
        int mx = 0; // the most segments of a split item
        for (int j = N - ns; j < N; j++) mx = std::max(mx, (nt[j] + segt - 1) / segt);
        for (int h = 1; h < mx; h++) {
            const int S = 1 + mx - h;
            const size_t na = cap.first / S; // (split items whose slots fit the first buffer)
            if ((size_t) ns > na + cap.second / S) continue;
            const double t = sim(N - ns, S, h);
            if (t < bt) bt = t, best = {N - ns, S, h, (int) std::min<size_t>(na, ns)};
        }
    }
    ex_memo[key] = best;
    return best;
}

void MKEngine::Impl::attn(int T, int n, int slot, int r0) {
    if (attn_old) {
        if (r0) throw std::runtime_error("prefill attention: MK_ATTN=1 (pk_attn) has no row offset (MK_EXACT with MK_KV_WIN)");
        if (hb.kvq8) mkl::pk_attn<true>(T, n, slot);
        else mkl::pk_attn<false>(T, n, slot);
        return;
    }
    if (exact && hb.kvq8 && attn_pv16 && attn_s8 == 1) { // MK_EXACT: pk_attn8's EXA form
        const int N = (T + A2_Q - 1) / A2_Q * NKV;
        const bool pcs = hb.kvrp[0][slot] || hb.kvrp[1][slot]; // (the chunk runs in pieces: pf_run)
        const ExPlan p = ex_plan(T, n, ex_cap(pcs));
        const auto pt = ex_part(pcs);
        const int grid = p.nfull + (N - p.nfull) * p.S;
        if (kv_k6v5(slot))
            mkl::pk_attn8<false, true, true, 6, 5>(grid, T, n, slot, p.nfull, p.S, pt.first, r0, ex_segt, p.h, pt.second, p.nA);
        else if (kv_k5v5(slot))
            mkl::pk_attn8<false, true, true, 5, 5>(grid, T, n, slot, p.nfull, p.S, pt.first, r0, ex_segt, p.h, pt.second, p.nA);
        else if (hb.kvb[0][slot] != 8)
            mkl::pk_attn8<false, true, true>(grid, T, n, slot, p.nfull, p.S, pt.first, r0, ex_segt, p.h, pt.second, p.nA);
        else if (hb.kvb[1][slot] == 8) mkl::pk_attn8<false, false, true, 0, 8>(grid, T, n, slot, p.nfull, p.S, pt.first, r0, ex_segt, p.h, pt.second, p.nA);
        else mkl::pk_attn8<false, false, true>(grid, T, n, slot, p.nfull, p.S, pt.first, r0, ex_segt, p.h, pt.second, p.nA);
        if (N > p.nfull) mkl::pk_attn8_combx(N - p.nfull, T, n, p.nfull, p.S, ex_segt, p.h, r0, pt.first, pt.second, p.nA);
        return;
    }
    // y1 (k, v of the layer) is free between pk_attn_prep and the FFN: partials of split items
    const int N = (T + A2_Q - 1) / A2_Q * NKV;
    int nfull = N, S = 1;
    if (attn_split) a2_plan(T, n, a2_slots, y1_floats, nfull, S);
    if (r0 && N > nfull) // (r0 != 0 only with MK_EXACT, which runs unsplit)
        throw std::runtime_error("prefill attention: the split merge (pk_attn2_comb) has no row offset");
    const int grid = nfull + (N - nfull) * S;
    if (hb.kvq8) {
        // pk_attn8 reads K in any MK_KV format (low-bit tiles are expanded to int8 in its staging) and any V format
        // (a8_cv_v converts). Rotated K too: pk_attn_prep rotates Q (fp32 fwht256) before its fp16 store to qh, so the
        // int8 Q fragments are rotated as well; pk_attn_out un-rotates a rotated V's output for every kernel. MK_ATTN_S8=0
        // keeps pk_attn2 (fp16 Q) for comparison.
        const bool s8 = attn_pv16;
        const bool kl = hb.kvb[0][slot] != 8; // low-bit K: the expanding instantiation (q8 K: the plain kernel)
        if (s8 && attn_s8 == 1) {
            if (kv_k6v5(slot)) mkl::pk_attn8<false, true, false, 6, 5>(grid, T, n, slot, nfull, S, hp.y1, r0);
            else if (kv_k5v5(slot)) mkl::pk_attn8<false, true, false, 5, 5>(grid, T, n, slot, nfull, S, hp.y1, r0);
            else if (kl) mkl::pk_attn8<false, true>(grid, T, n, slot, nfull, S, hp.y1, r0);
            else if (hb.kvb[1][slot] == 8) mkl::pk_attn8<false, false, false, 0, 8>(grid, T, n, slot, nfull, S, hp.y1, r0);
            else mkl::pk_attn8<false, false>(grid, T, n, slot, nfull, S, hp.y1, r0);
        } else if (s8 && attn_s8 == 2) {
            if (kl) mkl::pk_attn8<true, true>(grid, T, n, slot, nfull, S, hp.y1, r0);
            else mkl::pk_attn8<true, false>(grid, T, n, slot, nfull, S, hp.y1, r0);
        }
        else if (attn_pv16) mkl::pk_attn2<true, true>(grid, T, n, slot, nfull, S, hp.y1, r0);
        else mkl::pk_attn2<true, false>(grid, T, n, slot, nfull, S, hp.y1, r0);
    } else {
        if (attn_pv16) mkl::pk_attn2<false, true>(grid, T, n, slot, nfull, S, hp.y1, r0);
        else mkl::pk_attn2<false, false>(grid, T, n, slot, nfull, S, hp.y1, r0);
    }
    if (N > nfull) mkl::pk_attn2_comb(dim3(N - nfull, pfx_bits & 256 ? 1 : 6 * A2_Q / 8), T, nfull, S, hp.y1); // (a row per warp)
}

int MKEngine::Impl::pf_nb(int ntok, int nrt) const {
    if (pf_bn_force) return pf_bn_force;
    if (ntok > 384) return 128;
    const int n32 = nrt * ((ntok + 31) / 32), n64 = nrt * ((ntok + 63) / 64), n128 = nrt * ((ntok + 127) / 128);
    const double t32 = 0.80 * (n32 <= nsm ? 1.0 : 1.77 * ((n32 + 2 * nsm - 1) / (2 * nsm)));
    const double t64 = 0.95 * ((n64 + nsm - 1) / nsm), t128 = 1.44 * ((n128 + nsm - 1) / nsm);
    return t32 <= t64 && t32 <= t128 ? 32 : t64 <= t128 ? 64 : 128;
}

void MKEngine::Impl::pf_shape_check(const Mat & m, int rows) {
    if (m.K % (2 * pf::BK) != 0 || m.N % rows != 0)
        throw std::runtime_error("prefill GEMM: unsupported matrix shape K " + std::to_string(m.K) + " x N " + std::to_string(m.N) +
                                 " (needs K % " + std::to_string(2 * pf::BK) + " == 0 and N % " + std::to_string(rows) + " == 0)");
}
#if MK_PF_INFCHK
void MKEngine::Impl::chk_set(int layer, int site) {
    if (layer >= 0) CK(mkl::set_chk_layer_async(&layer, sizeof(int), 0, cudaMemcpyHostToDevice, 0));
    if (site >= 0) CK(mkl::set_chk_site_async(&site, sizeof(int), 0, cudaMemcpyHostToDevice, 0));
}

void MKEngine::Impl::chk_mat(const Mat & m) { // a GEMM's site and layer, from its matrix
    for (int l = 0; l < NLA; l++) {
        const LayerW & W = hw.L[l];
        const Mat * ms[10] = {&W.q, &W.k, &W.v, &W.o, &W.qkv, &W.gate, &W.sout, &W.fg, &W.fu, &W.fd}; // (pf::CHK_Q ..)
        for (int i = 0; i < 10; i++)
            if (m.w && ms[i]->w == m.w) return chk_set(l, i);
    }
    if (m.w == hw.eh.w) chk_set(NL, pf::CHK_EH);
    else chk_set(pf::CHK_LAYERS - 1, pf::CHK_HEAD);
}

void MKEngine::Impl::chk_report(int npos) {
    static const char * const nm[pf::CHK_SITES] = {"attn_q", "attn_k", "attn_v", "attn_o", "dn_qkv", "dn_gate", "dn_out", "ffn_gate",
                                                  "ffn_up", "ffn_down", "mtp_eh", "head", "in norm_quant", "in dn_out", "in mtp_in",
                                                  "in attn_gate", "in silu*up", "dn_scan S", "dn_scan K S", "dn_scan U", "dn_scan U d"};
    constexpr int NL_ = pf::CHK_LAYERS, NS = pf::CHK_SITES;
    std::vector<unsigned long long> c((size_t) NL_ * NS * 2);
    CK(mkl::get_chk_cnt(c.data(), c.size() * sizeof(unsigned long long)));
    unsigned long long t[3][2] = {{0, 0}, {0, 0}, {0, 0}}; // [GEMM | input | DeltaNet scan operand][kind]
    for (int l = 0; l < NL_; l++)
        for (int s = 0; s < NS; s++)
            for (int k = 0; k < 2; k++) t[(s >= pf::CHK_IN_NORM) + (s >= pf::CHK_DN_S)][k] += c[((size_t) l * NS + s) * 2 + k];
    fprintf(stderr, "[mk] MK_PF_INFCHK prefill of %d positions: GEMM fp16 partial sums inf/NaN %llu, non-finite GEMM outputs %llu;"
                    " fp16 GEMM inputs clamped (|v| > 65504) %llu, NaN (stored as -65504) %llu; DeltaNet scan fp16 operands |v| > 65504"
                    " (not clamped: inf from 65520) %llu, NaN %llu\n", npos, t[0][0], t[0][1], t[1][0], t[1][1], t[2][0], t[2][1]);
    for (int s = 0; s < NS; s++) {
        unsigned long long a[2] = {0, 0};
        std::string where;
        for (int l = 0; l < NL_; l++) {
            const unsigned long long x = c[((size_t) l * NS + s) * 2], y = c[((size_t) l * NS + s) * 2 + 1];
            a[0] += x, a[1] += y;
            if ((x || y) && where.size() < 120) where += " L" + std::to_string(l) + ":" + std::to_string(x) + "/" + std::to_string(y);
        }
        if (a[0] || a[1])
            fprintf(stderr, "[mk]   %-14s %s %llu, %s %llu  (layer:counts%s)\n", nm[s], s < pf::CHK_IN_NORM ? "partials" : "> 65504", a[0],
                    s < pf::CHK_IN_NORM ? "outputs" : "NaN", a[1], where.c_str());
    }
    std::fill(c.begin(), c.end(), 0ull);
    CK(mkl::set_chk_cnt(c.data(), c.size() * sizeof(unsigned long long)));
}
#endif
void MKEngine::Impl::gemm(const Mat & m, int ntok, float * y, int ldy, int mode, int t0, const __half * xin, cudaStream_t st) {
    pf_shape_check(m, pf::BM);
#if MK_PF_INFCHK
    chk_mat(m);
#endif
    const pf::WMat W{m.w, m.d, m.type, m.K, m.N, m.rb, m.nd};
    const __half * X = (xin ? xin : hp.xh) + (size_t) t0 * m.K;
    const pf::Out O{y, ldy, mode, nullptr};
    const int nrt = m.N / pf::BM;
    if (m.fm) {
        switch (m.type) {
            case T_Q4K: mmh_launch<pf::TQ4K, false, 1>(nrt, W, X, ntok, O, nullptr, st); return;
            case T_Q5K: mmh_launch<pf::TQ5K, false, 1>(nrt, W, X, ntok, O, nullptr, st); return;
            case T_Q8_0: mmh_launch<pf::TQ80, false, 1>(nrt, W, X, ntok, O, nullptr, st); return;
            case T_IQ4X: mmh_launch<pf::TIQ4, false, 1>(nrt, W, X, ntok, O, nullptr, st); return;
            case T_IQ4NL: mmh_launch<pf::TIQ4NL, false, 1>(nrt, W, X, ntok, O, nullptr, st); return;
            case T_Q6K: mmh_launch<pf::TQ6K, false, 1>(nrt, W, X, ntok, O, nullptr, st); return;
            case T_Q3K: mmh_launch<pf::TQ3K, false, 1>(nrt, W, X, ntok, O, nullptr, st); return;
            case T_IQ3S: mmh_launch<pf::TIQ3S, false, 1>(nrt, W, X, ntok, O, nullptr, st); return;
            default: throw std::runtime_error("gemm: unsupported weight type in the wv layout");
        }
    }
    switch (m.type) {
        case T_Q4K: mmh_launch<pf::TQ4K, false>(nrt, W, X, ntok, O, nullptr, st); break;
        case T_Q5K: mmh_launch<pf::TQ5K, false>(nrt, W, X, ntok, O, nullptr, st); break;
        case T_Q6K: mmh_launch<pf::TQ6K, false>(nrt, W, X, ntok, O, nullptr, st); break;
        case T_Q8_0: mmh_launch<pf::TQ80, false>(nrt, W, X, ntok, O, nullptr, st); break;
        case T_IQ4X: mmh_launch<pf::TIQ4, false>(nrt, W, X, ntok, O, nullptr, st); break;
        case T_IQ4NL: mmh_launch<pf::TIQ4NL, false>(nrt, W, X, ntok, O, nullptr, st); break;
        case T_Q3K: mmh_launch<pf::TQ3K, false>(nrt, W, X, ntok, O, nullptr, st); break;
        case T_IQ3S: mmh_launch<pf::TIQ3S, false>(nrt, W, X, ntok, O, nullptr, st); break;
        default: throw std::runtime_error("gemm: unsupported weight type");
    }
}

bool MKEngine::Impl::glu_fused(const Mat & g, const Mat & u) {
    if (g.fm != u.fm) return false;
    return g.type == u.type && g.rb == u.rb && (g.type == T_Q4K || g.type == T_Q5K || g.type == T_IQ4X);
}

void MKEngine::Impl::gemm_glu(const Mat & g, const Mat & u, int ntok) {
    pf_shape_check(g, pf::BM / 2);
    pf_shape_check(u, pf::BM / 2);
#if MK_PF_INFCHK
    chk_mat(g); // (the fused launches; two GEMMs + pk_glu set their own)
#endif
    const pf::WMat W{g.w, g.d, g.type, g.K, g.N, g.rb, g.nd};
    const pf::Out O{nullptr, g.N, pf::OUT_STORE, hp.xh2};
    const int nrt = g.N / (pf::BM / 2);
    if (glu_fused(g, u) && g.fm) {
        switch (g.type) {
            case T_Q4K: mmh_launch<pf::TQ4K, true, 1>(nrt, W, hp.xh, ntok, O, u.w); return;
            case T_Q5K: mmh_launch<pf::TQ5K, true, 1>(nrt, W, hp.xh, ntok, O, u.w); return;
            case T_IQ4X: mmh_launch<pf::TIQ4, true, 1>(nrt, W, hp.xh, ntok, O, u.w); return;
            default: break;
        }
    }
    if (glu_fused(g, u)) {
        switch (g.type) {
            case T_Q4K: mmh_launch<pf::TQ4K, true>(nrt, W, hp.xh, ntok, O, u.w); return;
            case T_Q5K: mmh_launch<pf::TQ5K, true>(nrt, W, hp.xh, ntok, O, u.w); return;
            case T_IQ4X: mmh_launch<pf::TIQ4, true>(nrt, W, hp.xh, ntok, O, u.w); return;
            default: break;
        }
    }
    if (glu_mix_on && glu_mix(g, u, ntok)) return;
    gemm(g, ntok, hp.y0, g.N, pf::OUT_STORE);
    gemm(u, ntok, hp.y1, u.N, pf::OUT_STORE);
    mkl::pk_glu(ntok, hp.y0, hp.y1, hp.xh2, g.N);
}

void MKEngine::Impl::glu_mix_attr() {
#define MK_GLU_ATTR(TG, LG, TU, LU) mmh_attr_mix<TG, LG, TU, LU>();
    MK_GLU_MIX_PAIRS(MK_GLU_ATTR)
#undef MK_GLU_ATTR
}

bool MKEngine::Impl::glu_mix(const Mat & g, const Mat & u, int ntok) {
    if (glu_fused(g, u) || g.K != u.K || g.N != u.N) return false;
    const pf::WMat Wg{g.w, g.d, g.type, g.K, g.N, g.rb, g.nd}, Wu{u.w, u.d, u.type, u.K, u.N, u.rb, u.nd};
    const pf::Out O{nullptr, g.N, pf::OUT_STORE, hp.xh2};
    const int nrt = g.N / (pf::BM / 2), gl = g.fm, ul = u.fm;
#define MK_GLU_GO(TG, LG, TU, LU)                                                                                          \
if (g.type == TG && gl == LG && u.type == TU && ul == LU) {                                                            \
    if (ntok > 0) mmh_launch_mix<TG, LG, TU, LU>(nrt, Wg, Wu, ntok, O);                                                \
    return true;                                                                                                       \
}
    MK_GLU_MIX_PAIRS(MK_GLU_GO)
#undef MK_GLU_GO
    return false;
}

void MKEngine::Impl::pf_norm_quant(const float * src, const float * w, int T) {
    if (k8 & 2) mkl::pk_norm_quant1(T, src, w);
    else mkl::pk_norm_quant(T, src, w);
}

void MKEngine::Impl::pf_dn_prep(int T, int n, int dl, const LayerW & W) {
    if (k8 & 4) {
        if (k8_nt == 8) mkl::pk_dn_prep_t<8>(T, n, dl, W.conv, W.dt, W.A);
        else mkl::pk_dn_prep_t<4>(T, n, dl, W.conv, W.dt, W.A);
    } else {
        mkl::pk_dn_prep(T, n, dl, W.conv, W.dt, W.A);
    }
}

void MKEngine::Impl::pf_report() {
    if (!pf_prof || pf_ev.empty()) return;
    CK(cudaDeviceSynchronize());
    // per category: sum, and median-per-launch x launches (robust to time-slicing by other GPU users; it assumes one population
    // of launches per label, so launches of different sizes get labels of their own). Rows by sum, its share last.
    std::vector<std::string> cats;
    std::vector<std::vector<float>> ts;
    for (auto & e : pf_ev) {
        float ms;
        CK(cudaEventElapsedTime(&ms, e.a, e.b));
        auto it = std::find(cats.begin(), cats.end(), std::string(e.cat));
        if (it == cats.end()) { cats.push_back(e.cat); ts.push_back({}); it = cats.end() - 1; }
        ts[it - cats.begin()].push_back(ms);
        cudaEventDestroy(e.a);
        cudaEventDestroy(e.b);
    }
    pf_ev.clear();
    double all = 0, all_med = 0;
    std::vector<std::tuple<double, double, std::string, size_t>> rows;
    for (size_t i = 0; i < cats.size(); i++) {
        auto v = ts[i];
        double sum = 0;
        for (float x : v) sum += x;
        std::sort(v.begin(), v.end());
        const double med = v[v.size() / 2] * v.size();
        rows.emplace_back(sum, med, cats[i], v.size());
        all += sum;
        all_med += med;
    }
    for (auto & kv : pf_iso) fprintf(stderr, "  [isolated] %-12s %8.1f ms\n", kv.first.c_str(), kv.second);
    pf_iso.clear();
    std::sort(rows.rbegin(), rows.rend());
    fprintf(stderr, "[mk-prefill-profile] %.1f ms in kernels (median-based %.1f ms)\n", all, all_med);
    for (auto & r : rows)
        fprintf(stderr, "  %-14s median-based %8.1f ms %5.1f%%   sum %8.1f ms   x%zu   %5.1f%% of the sum\n", std::get<2>(r).c_str(),
                std::get<1>(r), 100 * std::get<1>(r) / all_med, std::get<0>(r), std::get<3>(r), 100 * std::get<0>(r) / all);
}

void MKEngine::Impl::pf_run(const std::vector<int> & toks, int n0, std::vector<int> * pred, const MKEngine::LogitSink * sink,
            int first, const std::vector<Snap> * snaps, const int * erow) {
    pf_init();
    const int kv_top0 = kv_top;
    sw_need(n0 + (int) toks.size()); // (side request)
    kv_top = std::max(kv_top, n0 + (int) toks.size()); // (ring_rewind)
    pf_cancelled = false;
    ck_scr_guard(); // the scratch may still be the source of an end save's copy
    ck_guard_state();
    const int N = (int) toks.size(), cap = hp.cap;
    // chunk c's GPU work is enqueued before the host waits for chunk c - 1 (its pf_cev event), so the GPU does not
    // idle while the host wakes up and launches; the scoring paths (pred / sink) read results back per chunk
    const bool ahead = !pf_sync && !pred && !sink;
    const int nch = (N + cap - 1) / cap;
    if (pf_gaps) // 4 timing events per chunk: start, trunk end, MTP part start, end
        while (pf_gev.size() < (size_t) 4 * nch) {
            cudaEvent_t e;
            CK(cudaEventCreate(&e));
            pf_gev.push_back(e);
        }
    auto gap_mark = [&](int c, int k) {
        if (pf_gaps) CK(cudaEventRecord(pf_gev[4 * c + k], 0));
    };
    for (int c0 = 0; c0 < N; c0 += cap) {
        // MKEngine::cancel: stop at this chunk boundary (a host flag, no device sync; the chunks already enqueued finish
        // in the tail wait below). The caller (MKEngine::prefill) discards what they computed.
        if (c0 > 0 && !pred && !sink && pf_cancel.load(std::memory_order_relaxed)) {
            pf_cancelled = true;
            pf_stop = n0 + c0;
            kv_top = std::max(kv_top0, n0 + c0); // (rows of the chunks never enqueued were not written)
            break;
        }
#if MK_KV_EMU
        kv_demote_to(n0 + c0 - hb.kvemu_win); // recent-window emulation: rows older than the window before this chunk
#endif
        const int T = std::min(cap, N - c0), n = n0 + c0, ci = c0 / cap;
        int sn = 0, sp[STG_MAX], ss[STG_MAX]; // this chunk's checkpoints (n < P <= n + T) and their slots
        if (snaps && !pred && !sink)
            for (const Snap & q : *snaps)
                if (q.pos > n && q.pos <= n + T && sn < stg_n) sp[sn] = q.pos, ss[sn++] = q.slot;
        const bool img = erow && std::any_of(erow + c0, erow + c0 + T, [](int r) { return r >= 0; });
        if (ahead) { // stream-ordered behind the previous chunk (pageable source: staged at the call, no device wait)
            CK(cudaMemcpyAsync(hp.tok, toks.data() + c0, T * sizeof(int), cudaMemcpyHostToDevice, 0));
            if (img) CK(cudaMemcpyAsync(hp.erow, erow + c0, T * sizeof(int), cudaMemcpyHostToDevice, 0));
        } else {
            CK(cudaMemcpy(hp.tok, toks.data() + c0, T * sizeof(int), cudaMemcpyHostToDevice));
            if (img) CK(cudaMemcpy(hp.erow, erow + c0, T * sizeof(int), cudaMemcpyHostToDevice));
        }
        ck_issue_restore(); // a due checkpoint restore streams in from here on (waited for per layer group below)
        ring_snap_guard();  // (MK_EXACT: a previous chunk's ring row copies, before this chunk's preps)
        const bool rsnap = sn > 0 && ring_bytes_ck > 0; // MK_EXACT + MK_KV_WIN: the checkpoints keep the rings
        for (int j = 0; j < sn; j++) ck[ss[j]].ring_ok = false;
        if (rsnap) ring_snap_begin(sn, ss);
        gap_mark(ci, 0); // (after the token upload: MK_PF_SYNC's idle is before it)
        pf_time("embed", [&] { // (token_embd's type)
            if (hw.embd_t == T_Q6K) {
                if (img) mkl::pk_embed_img<T_Q6K>(T);
                else mkl::pk_embed<T_Q6K>(T);
            } else if (hw.embd_t == T_Q8_0) {
                if (img) mkl::pk_embed_img<T_Q8_0>(T);
                else mkl::pk_embed<T_Q8_0>(T);
            } else {
                if (img) mkl::pk_embed_img<T_Q4K>(T);
                else mkl::pk_embed<T_Q4K>(T);
            }
        });
        int dl = 0;
        for (int L = 0; L < NL; L++) {
            const LayerW & W = hw.L[L];
#if MK_PF_INFCHK
            chk_set(L, -1); // (this layer's input stores; its GEMMs set their own site)
#endif
            if (fault_now && L == NL / 2) fault_inject();
            pf_time("norm_quant", [&] { pf_norm_quant(hp.x, W.attn_norm, T); });
            const bool fk = pf_fork && !pf_prof; // (MK_PF_PROFILE times every launch in order on stream 0)
            if (W.attn) {
                // k (y1 columns 0 .. KVD - 1), v (y1 columns KVD ..), q (y0) all read xh, which is final at pf_e0. pf_fork:
                // v on pf_s2 next to k then q on stream 0, so q's blocks finish last and on stream 0 and the join waits for
                // nothing (k / v on the side streams were dispatched behind q's blocks and finished last: their join cost
                // ~45 us of idle GPU per layer). MK_PFX bit 8: k on pf_s2, v on pf_s3, q on stream 0.
                const bool kq = fk && !(pfx_bits & 8);
                if (fk) {
                    CK(cudaEventRecord(pf_e0, 0));
                    CK(cudaStreamWaitEvent(pf_s2, pf_e0, 0));
                    if (!kq) {
                        CK(cudaStreamWaitEvent(pf_s3, pf_e0, 0));
                        gemm(W.k, T, hp.y1, 2 * KVD, pf::OUT_STORE, 0, nullptr, pf_s2);
                    }
                    gemm(W.v, T, hp.y1 + KVD, 2 * KVD, pf::OUT_STORE, 0, nullptr, kq ? pf_s2 : pf_s3);
                    CK(cudaEventRecord(pf_e1, pf_s2));
                    if (!kq) CK(cudaEventRecord(pf_e2, pf_s3));
                }
                if (kq) gemm(W.k, T, hp.y1, 2 * KVD, pf::OUT_STORE);
                pf_time("gemm_attn_q", [&] { gemm(W.q, T, hp.y0, QG, pf::OUT_STORE); });
                if (fk) {
                    CK(cudaStreamWaitEvent(0, pf_e1, 0));
                    if (!kq) CK(cudaStreamWaitEvent(0, pf_e2, 0));
                } else {
                    pf_time("gemm_attn_kv", [&] { gemm(W.k, T, hp.y1, 2 * KVD, pf::OUT_STORE); gemm(W.v, T, hp.y1 + KVD, 2 * KVD, pf::OUT_STORE); });
                }
                const int sl = L / 4;
                if (rsnap) CK(cudaStreamWaitEvent(0, rs_ev[sl], 0)); // (the rings' whole copies are done)
                if (exact && (hb.kvrp[0][sl] || hb.kvrp[1][sl])) {
                    // MK_EXACT with a KV window ring: the chunk in pieces ending at multiples of ATT_BUCKET (as decode
                    // launches end): a piece's rows read the window of their bucket (PBufs::wsrow), which the ring
                    // (kvrc rows) still holds after the piece's own rows are stored; a whole chunk's rows would have
                    // overwritten the oldest window rows of its first queries
                    for (int a = 0; a < T;) {
                        const int b = std::min(T, ((n + a) / ATT_BUCKET + 1) * ATT_BUCKET - n);
                        pf_time("attn_prep", [&] { mkl::pk_attn_prep(b - a, n, sl, W.qn, W.kn, 1, a); });
                        pf_time("attn", [&] { attn(b - a, n + a, sl, a); });
                        a = b;
                    }
                } else {
                    pf_time("attn_prep", [&] { mkl::pk_attn_prep(T, n, sl, W.qn, W.kn, 1); });
                    pf_time("attn", [&] { attn(T, n, sl); });
                }
                if (rsnap) ring_snap_rows(sl, n, sn, sp, ss);
                pf_time("attn_out", [&] { mkl::pk_attn_out(T, L / 4); });
                pf_time("gemm_attn_o", [&] { gemm(W.o, T, hp.x, D, pf::OUT_ADD); });
            } else {
                // MK_K8 bit 1: pk_ab (reads x and rs, writes ab: only pk_dn_prep reads it) on pf_s2 next to the GEMMs, which
                // read xh; pk_dn_prep waits for it. (MK_PF_PROFILE: in order on stream 0, so it is timed alone.)
                // pf_fork: the gate GEMM (y1, read by pk_dn_out) on pf_s2 too, after pk_ab if that is there, next to qkv (y0)
                // pk_ab (unless MK_K8 bit 1) on stream 0 before the join (MK_PFX bit 4: after it). The gate GEMM's blocks
                // are dispatched behind qkv's and finish last; waiting for them cost ~45 us of idle GPU per layer (the
                // cross-stream event wait), and pk_ab's blocks now fill the gate's last wave instead
                const bool k8a = (k8 & 1) && pf_s2 && !pf_prof;
                const bool abj = !k8a && !(pfx_bits & 4);
                auto ab = [&](cudaStream_t st) { // (MK_PFX bit 16: pk_ab, else pk_ab2: the same values)
                    if (pfx_bits & 16) mkl::pk_ab(W.ab, W.abq, W.abd, W.attn_norm, T, st);
                    else mkl::pk_ab2(W.ab, W.abq, W.abd, W.attn_norm, T, st);
                };
                if (k8a || fk) {
                    CK(cudaEventRecord(pf_e0, 0));
                    CK(cudaStreamWaitEvent(pf_s2, pf_e0, 0));
                    if (k8a) ab(pf_s2);
                    if (fk) gemm(W.gate, T, hp.y1, VD, pf::OUT_STORE, 0, nullptr, pf_s2);
                    CK(cudaEventRecord(pf_e1, pf_s2));
                }
                pf_time("gemm_dn_in", [&] {
                    gemm(W.qkv, T, hp.y0, QKV, pf::OUT_STORE);
                    if (!fk) gemm(W.gate, T, hp.y1, VD, pf::OUT_STORE);
                });
                if (abj) pf_time("dn_ab", [&] { ab(0); });
                if (k8a || fk) CK(cudaStreamWaitEvent(0, pf_e1, 0));
                if (!k8a && !abj) pf_time("dn_ab", [&] { ab(0); });
                ck_wait_restore(dl); // this layer's conv ring / state may still be arriving from the checkpoint
                int sg[STG_MAX], ta = -1; // staging slots of this layer's checkpoint pieces (+ a side scan's input)
                for (int j = 0; j < sn; j++) sg[j] = stg_take();
                for (int j = 0; j < sn; j++)
                    if ((sp[j] - n) % 32 && sp[j] - n > 32) ta = stg_take();
                pf_time("dn_prep", [&] {
                    pf_dn_prep(T, n, dl, W);
                    for (int j = 0; j < sn; j++) mkl::pk_ck_conv(n, sp[j], dl, stg_conv(sg[j]));
                    mkl::pk_conv_ring(T, n, dl, pfx_bits & 128);
                });
                pf_time("dn_scan", [&] {
                    // The scan is split only at points of its 32-token grid, which changes no arithmetic: the state
                    // at an on-grid checkpoint goes to its staging slot on the way. An off-grid one (the exact
                    // ck_at) is computed on the side: a scan of the rows from the grid point before it, from the
                    // state there, into its staging slot (= what a prefill ending at it computes); the main scan
                    // then rewrites those rows' y2.
                    // MK_PFX bit 2 off: the per-chunk matrices that do not depend on the state come from pk_dn_tri (all
                    // chunks at once, into y0: free from pk_conv_ring to the next GEMM) and pk_dn_scan4 runs the state
                    // recurrence. Pieces on the 32-token grid ending at T only: the off-grid side scan pads its last chunk
                    // at te, so it keeps pk_dn_scan (same arithmetic either way).
                    float * live = hb.dstate + (size_t) dl * CK_SL;
                    const float * in = live;
                    __half * tri = pfx_bits & 2 ? nullptr : reinterpret_cast<__half *>(hp.y0);
                    if (tri) mkl::pk_dn_tri(T, tri);
                    auto scan = [&](int nt, int ts, const float * a, float * b) {
                        if (tri) mkl::pk_dn_scan4(nt, ts, a, b, tri);
                        else mkl::pk_dn_scan(nt, ts, a, b);
                    };
                    int t = 0;
                    for (int j = 0; j < sn; j++) {
                        const int te = sp[j] - n, tg = te / 32 * 32;
                        if (te == tg) {
                            scan(tg - t, t, in, stg_state(sg[j]));
                            in = stg_state(sg[j]);
                            t = tg;
                            continue;
                        }
                        if (tg > t) {
                            scan(tg - t, t, in, stg_state(ta));
                            in = stg_state(ta);
                            t = tg;
                        }
                        mkl::pk_dn_scan(te - tg, tg, in, stg_state(sg[j]));
                    }
                    scan(T - t, t, in, live);
                });
                if (sn) stg_flush(dl, sn, sg, ss);
                pf_time("dn_out", [&] { mkl::pk_dn_out(T, W.snorm); });
                pf_time("gemm_dn_out", [&] { gemm(W.sout, T, hp.x, D, pf::OUT_ADD); });
                dl++;
            }
            pf_time("norm_quant", [&] { pf_norm_quant(hp.x, W.post_norm, T); });
            // (profile: mixed-type fused gate / up pairs are reported as gemm_ffn_gum, pairs that run as two GEMMs +
            // pk_glu as gemm_ffn_gu2)
            pf_time(glu_fused(W.fg, W.fu) ? "gemm_ffn_gu" : glu_mix_on && glu_mix(W.fg, W.fu, 0) ? "gemm_ffn_gum" : "gemm_ffn_gu2",
                    [&] { gemm_glu(W.fg, W.fu, T); });
            pf_time("gemm_ffn_down", [&] { gemm(W.fd, T, hp.x, D, pf::OUT_ADD, 0, hp.xh2); });
        }
        wd_T = T;
        wd_mode = MODE_PREFILL;
        gap_mark(ci, 1);
        if (!ahead) wait_device("a prefill chunk", std::max(wd_limit, 120.0));
        gap_mark(ci, 2);
        if (pred || sink) {
#if MK_PF_INFCHK
            chk_set(pf::CHK_LAYERS - 1, -1);
#endif
            pf_norm_quant(hp.x, hw.onorm, T);
            std::vector<float> rows;
            const int tile = (int) std::min<size_t>(128, y0_floats / VOCAB); // logits rows that fit in y0
            for (int s = 0; s < T; s += tile) {
                const int nt = std::min(tile, T - s);
                const int lo = std::max(s, first - n); // rows from lo on are at positions >= first
                if (!pred && lo >= s + nt) continue;
                gemm(hw.head, nt, hp.y0, VOCAB, pf::OUT_STORE, s);
                if (pred) mkl::pk_argmax(nt, s);
                if (sink && lo < s + nt) {
                    rows.resize((size_t) (s + nt - lo) * VOCAB);
                    CK(cudaMemcpy(rows.data(), hp.y0 + (size_t) (lo - s) * VOCAB, rows.size() * sizeof(float), cudaMemcpyDeviceToHost));
                    (*sink)(n + lo, s + nt - lo, rows.data());
                }
            }
            if (pred) {
                std::vector<int> am(T);
                CK(cudaMemcpy(am.data(), hp.amax, T * sizeof(int), cudaMemcpyDeviceToHost));
                pred->insert(pred->end(), am.begin(), am.end());
            }
            continue;
        }
        if (c0 == 0) { // hp.h row 0: the hidden state before the chunk, read only by pk_mtp_in (after every restore wait)
            if (n0 == 0) CK(cudaMemsetAsync(hp.h, 0, D * sizeof(float), 0));
            else CK(cudaMemcpyAsync(hp.h, hb.h_hist + (size_t) ((n0 - 1) % H_RING) * D, D * sizeof(float), cudaMemcpyDeviceToDevice, 0));
        }
        stg_drain(); // (the MTP part writes hp.xm)
        for (int j = 0; j < sn; j++) { // h ring at each checkpoint: positions before the chunk, from the ring now
            if (ck[ss[j]].pending) CK(cudaStreamWaitEvent(0, ck[ss[j]].ev, 0)); // (an earlier save into the slot)
            if (sp[j] - H_RING < n) mkl::pk_ck_hist(n - (sp[j] - H_RING), sp[j] - H_RING, n, 1, ck[ss[j]].hdev);
        }
        pf_time("mtp", [&] {
            mkl::pk_final_h(T, n);
            for (int j = 0; j < sn; j++) { // ... and positions of the chunk, from its h rows
                const int lo = std::max(n, sp[j] - H_RING);
                mkl::pk_ck_hist(sp[j] - lo, lo, n, 0, ck[ss[j]].hdev);
            }
            if (!has_mtp) return; // (no MTP block: no MTP K / V rows)
#if MK_PF_INFCHK
            chk_set(NL, -1);
#endif
            if (hw.embd_t == T_Q6K) {
                if (img) mkl::pk_mtp_in<true, T_Q6K>(T);
                else mkl::pk_mtp_in<false, T_Q6K>(T);
            } else if (hw.embd_t == T_Q8_0) {
                if (img) mkl::pk_mtp_in<true, T_Q8_0>(T);
                else mkl::pk_mtp_in<false, T_Q8_0>(T);
            } else {
                if (img) mkl::pk_mtp_in<true, T_Q4K>(T);
                else mkl::pk_mtp_in<false, T_Q4K>(T);
            }
            gemm(hw.eh, T, hp.xm, D, pf::OUT_STORE);
            const LayerW & M = hw.L[NL];
            pf_norm_quant(hp.xm, M.attn_norm, T);
            gemm(M.k, T, hp.y1, 2 * KVD, pf::OUT_STORE);
            gemm(M.v, T, hp.y1 + KVD, 2 * KVD, pf::OUT_STORE);
            if (rsnap) CK(cudaStreamWaitEvent(0, rs_ev[N_ATT_SLOTS - 1], 0));
            mkl::pk_attn_prep(T, n, 16, M.qn, M.kn, 0);
            if (rsnap) ring_snap_rows(N_ATT_SLOTS - 1, n, sn, sp, ss);
        });
        CK(cudaMemcpyAsync(hp.h, hp.h + (size_t) T * D, D * sizeof(float), cudaMemcpyDeviceToDevice));
        gap_mark(ci, 3);
        if (ahead) { // (the watchdog still covers every chunk, one behind)
            CK(cudaEventRecord(pf_cev[ci & 1], 0));
            if (ci > 0) wait_event(pf_cev[(ci - 1) & 1], "a prefill chunk", std::max(wd_limit, 120.0));
        }
        pf_pos.store(n, std::memory_order_relaxed); // (the chunks before this one are done: waited for above)
    }
    if (snaps && !pred && !sink)
        for (const Snap & q : *snaps) { // (its last piece was copied before the last chunk's MTP part)
            CK(cudaEventRecord(ck[q.slot].ev, ck_sv));
            ck[q.slot].pending = true;
        }
    CK(cudaGetLastError());
    wait_device("the prefill tail", std::max(wd_limit, 120.0));
    if (!pf_cancelled) pf_pos.store(n0 + N, std::memory_order_relaxed);
    if (memlog && pf_runs_seen++ == 0) mem_mark("first prefill run");
    if (pf_gaps && !pred && !sink && !pf_cancelled) { // GPU idle at the host's points: before a chunk's first kernel, before its MTP part
        double g0 = 0, g1 = 0, wall = 0;
        float ms;
        for (int c = 0; c < nch; c++) {
            if (c > 0) {
                CK(cudaEventElapsedTime(&ms, pf_gev[4 * (c - 1) + 3], pf_gev[4 * c]));
                g0 += ms;
            }
            CK(cudaEventElapsedTime(&ms, pf_gev[4 * c + 1], pf_gev[4 * c + 2]));
            g1 += ms;
        }
        CK(cudaEventElapsedTime(&ms, pf_gev[0], pf_gev[4 * (nch - 1) + 3]));
        wall = ms;
        fprintf(stderr, "[mk] prefill gaps (%s): %d chunks, %.1f ms on the GPU; idle before chunks %.2f ms, before MTP parts %.2f ms\n",
                ahead ? "ahead" : "MK_PF_SYNC", nch, wall, g0, g1);
    }
#if MK_PF_INFCHK
    chk_report(N);
#endif
    pf_report();
}

std::vector<int> MKEngine::Impl::ck_positions(const std::vector<int> & cks, int exact, int n0, int hend, const std::vector<int> & pins,
                              std::vector<char> * vpin) {
    const int cap = hp.cap;
    auto chunk = [&](int P) { return (P - n0 - 1) / cap; }; // the chunk P belongs to (n < P <= n + T)
    auto at = [&](int P) { // where P is saved (-1: outside this part or on an existing checkpoint)
        if (P <= n0 || P > hend) return -1;
        if (P != exact) {
            const int cs = n0 + chunk(P) * cap;
            P = cs + (P - cs) / 32 * 32;
            if (P <= n0) return -1;
        }
        return ck_find(P) < 0 ? P : -1;
    };
    std::vector<int> v, pv;
    for (int P : cks)
        if ((P = at(P)) >= 0) v.push_back(P);
    for (int P : pins)
        if ((P = at(P)) >= 0) v.push_back(P), pv.push_back(P);
    std::sort(v.begin(), v.end());
    v.erase(std::unique(v.begin(), v.end()), v.end());
    auto kept = [&](int P) { return P == exact || std::find(pv.begin(), pv.end(), P) != pv.end(); };
    for (size_t i = 0; i < v.size();) { // per chunk: at most stg_n staging slots
        size_t j = i;
        while (j < v.size() && chunk(v[j]) == chunk(v[i])) j++;
        const bool ex = std::find(v.begin() + i, v.begin() + j, exact) != v.begin() + j;
        int extra = (int) (j - i) + (ex ? 1 : 0) - stg_n;
        for (int pass = 0; pass < 2; pass++) // (a pin only when nothing else is left)
            for (size_t k = i; k < j && extra > 0;)
                if (v[k] != exact && (pass || !kept(v[k]))) v.erase(v.begin() + k), j--, extra--;
                else k++;
        i = j;
    }
    if (vpin) {
        vpin->assign(v.size(), 0);
        for (size_t k = 0; k < v.size(); k++) (*vpin)[k] = std::find(pv.begin(), pv.end(), v[k]) != pv.end();
    }
    return v;
}
