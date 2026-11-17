// Prefill attention kernels.  Included from prefill.cuh inside qwen38_mk.cu's anonymous namespace (and by the
// standalone microbenchmark bench/attn_bench.cu with a minimal environment).  Uses NH, NKV, HD, VD,
// cB (kc, vc, ksc, vsc, ctx), cP (qh, y2), cp_async16, q8_to_f16, mma16816 and pf:: cpa_commit, cpa_wait,
// ldsm_x4, hmma16, hmma16z, u2h.

// Causal flash attention over the cache for the chunk's queries.  CTA = (16 query positions, kv
// head); warp = one of the 6 query heads of the group (16 rows).  K / V tiles of 32 positions in
// shared memory, S = Q K^T and O += P V on tensor cores (fp16 operands, fp32 accumulation).
constexpr int AT_Q = 16, AT_K = 32, AT_LD = HD + 8;
constexpr int AT_SMEM = (6 * AT_Q + 2 * AT_K) * AT_LD * 2; // Q, K/V tile
template <bool Q8>
__global__ void __launch_bounds__(192) pk_attn(int T, int n0, int slot) { MK_VIEW_INIT();
    extern __shared__ __align__(16) uint8_t asm_[];
    __half * sQ = reinterpret_cast<__half *>(asm_); // [96][AT_LD]
    __half * sK = sQ + 6 * AT_Q * AT_LD;            // [32][AT_LD]
    __half * sV = sK + AT_K * AT_LD;                // [32][AT_LD]
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g8 = lane >> 2, t4 = lane & 3;
    const int t0 = blockIdx.x * AT_Q, g = blockIdx.y;
    const size_t ctx = cB.ctx;
    const size_t row0 = ((size_t) slot * NKV + g) * ctx;
    const __half * Kc = static_cast<const __half *>(cB.kc) + row0 * HD;
    const __half * Vc = static_cast<const __half *>(cB.vc) + row0 * HD;
    const int nq = min(AT_Q, T - t0);
    for (int i = tid; i < 6 * AT_Q * 32; i += 192) { // 96 rows x 32 chunks of 16 B
        const int r = i >> 5, c = i & 31, hq = r / AT_Q, tt = r % AT_Q;
        __half * d = sQ + r * AT_LD + 8 * c;
        if (tt < nq) cp_async16(d, cP.qh + ((size_t) (t0 + tt) * NH + g * 6 + hq) * HD + 8 * c);
        else *reinterpret_cast<int4 *>(d) = make_int4(0, 0, 0, 0);
    }
    pf::cpa_commit();
    const int qpos0 = n0 + t0, pmax = qpos0 + nq - 1;
    float o[32][4];
#pragma unroll
    for (int i = 0; i < 32; i++) o[i][0] = o[i][1] = o[i][2] = o[i][3] = 0.f;
    float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.f, l1 = 0.f; // rows g8, g8 + 8
    const int qr0 = qpos0 + g8, qr1 = qpos0 + g8 + 8;
    const __half * qa = sQ + (warp * AT_Q + g8) * AT_LD + 2 * t4;
    for (int kb = 0; kb <= pmax; kb += AT_K) {
        const int np = min(AT_K, pmax + 1 - kb);
        if constexpr (Q8) { // q8 K/V tile -> fp16 in shared memory (all loads issued before the first conversion)
            constexpr int PER = (2 * AT_K * 32 + 191) / 192;
            uint2 q[PER];
            float d[PER];
#pragma unroll
            for (int k = 0; k < PER; k++) {
                const int i = tid + 192 * k, which = i / (AT_K * 32), r = (i >> 5) % AT_K, c = i & 31;
                q[k] = make_uint2(0, 0);
                d[k] = 0.f;
                if (i < 2 * AT_K * 32 && r < np) {
                    const size_t row = row0 + kb + r;
                    q[k] = __ldg(reinterpret_cast<const uint2 *>(static_cast<const int8_t *>(which ? cB.vc : cB.kc) + row * HD) + c);
                    d[k] = __half2float((which ? cB.vsc : cB.ksc)[row * (HD / 32) + (c >> 2)]);
                }
            }
            pf::cpa_wait<0>(); // Q rows (first tile)
#pragma unroll
            for (int k = 0; k < PER; k++) {
                const int i = tid + 192 * k, which = i / (AT_K * 32), r = (i >> 5) % AT_K, c = i & 31;
                if (i >= 2 * AT_K * 32) break;
                __half2 h[4];
                q8_to_f16(q[k], d[k], h);
                *reinterpret_cast<int4 *>((which ? sV : sK) + r * AT_LD + 8 * c) = *reinterpret_cast<int4 *>(h);
            }
        } else {
            for (int i = tid; i < AT_K * 32; i += 192) {
                const int p = i >> 5, c = i & 31;
                if (p < np) {
                    cp_async16(sK + p * AT_LD + 8 * c, Kc + (size_t) (kb + p) * HD + 8 * c);
                    cp_async16(sV + p * AT_LD + 8 * c, Vc + (size_t) (kb + p) * HD + 8 * c);
                } else {
                    *reinterpret_cast<int4 *>(sK + p * AT_LD + 8 * c) = make_int4(0, 0, 0, 0);
                    *reinterpret_cast<int4 *>(sV + p * AT_LD + 8 * c) = make_int4(0, 0, 0, 0);
                }
            }
            pf::cpa_commit();
            pf::cpa_wait<0>();
        }
        __syncthreads();
        float s[4][4];
#pragma unroll
        for (int ni = 0; ni < 4; ni++) s[ni][0] = s[ni][1] = s[ni][2] = s[ni][3] = 0.f;
#pragma unroll
        for (int ks = 0; ks < HD / 16; ks++) {
            const unsigned a0 = *reinterpret_cast<const unsigned *>(qa + 16 * ks);
            const unsigned a1 = *reinterpret_cast<const unsigned *>(qa + 8 * AT_LD + 16 * ks);
            const unsigned a2 = *reinterpret_cast<const unsigned *>(qa + 16 * ks + 8);
            const unsigned a3 = *reinterpret_cast<const unsigned *>(qa + 8 * AT_LD + 16 * ks + 8);
#pragma unroll
            for (int ni = 0; ni < 4; ni++) {
                const __half * kp = sK + (8 * ni + g8) * AT_LD + 2 * t4 + 16 * ks;
                mma16816(s[ni], a0, a1, a2, a3, *reinterpret_cast<const unsigned *>(kp),
                         *reinterpret_cast<const unsigned *>(kp + 8));
            }
        }
        // causal mask + online softmax (rows g8 and g8 + 8; a row's 4 lanes share it)
        float mx0 = -INFINITY, mx1 = -INFINITY;
#pragma unroll
        for (int ni = 0; ni < 4; ni++) {
            const int p = kb + 8 * ni + 2 * t4;
            if (p > qr0) s[ni][0] = -INFINITY;
            if (p + 1 > qr0) s[ni][1] = -INFINITY;
            if (p > qr1) s[ni][2] = -INFINITY;
            if (p + 1 > qr1) s[ni][3] = -INFINITY;
            mx0 = fmaxf(mx0, fmaxf(s[ni][0], s[ni][1]));
            mx1 = fmaxf(mx1, fmaxf(s[ni][2], s[ni][3]));
        }
        mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 1));
        mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 2));
        mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 1));
        mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 2));
        const float mn0 = fmaxf(m0, mx0), mn1 = fmaxf(m1, mx1);
        const float al0 = mn0 == -INFINITY ? 1.f : expf(m0 - mn0), al1 = mn1 == -INFINITY ? 1.f : expf(m1 - mn1);
        float ps0 = 0.f, ps1 = 0.f;
        unsigned pa[4][2]; // P as fp16 pairs: [n8 tile][row g8 / g8 + 8]
#pragma unroll
        for (int ni = 0; ni < 4; ni++) {
            const float p0 = mn0 == -INFINITY ? 0.f : expf(s[ni][0] - mn0);
            const float p1 = mn0 == -INFINITY ? 0.f : expf(s[ni][1] - mn0);
            const float p2 = mn1 == -INFINITY ? 0.f : expf(s[ni][2] - mn1);
            const float p3 = mn1 == -INFINITY ? 0.f : expf(s[ni][3] - mn1);
            ps0 += p0 + p1;
            ps1 += p2 + p3;
            __half2 h01 = __floats2half2_rn(p0, p1), h23 = __floats2half2_rn(p2, p3);
            pa[ni][0] = *reinterpret_cast<unsigned *>(&h01);
            pa[ni][1] = *reinterpret_cast<unsigned *>(&h23);
        }
        ps0 += __shfl_xor_sync(0xffffffffu, ps0, 1);
        ps0 += __shfl_xor_sync(0xffffffffu, ps0, 2);
        ps1 += __shfl_xor_sync(0xffffffffu, ps1, 1);
        ps1 += __shfl_xor_sync(0xffffffffu, ps1, 2);
        l0 = l0 * al0 + ps0;
        l1 = l1 * al1 + ps1;
        m0 = mn0;
        m1 = mn1;
#pragma unroll
        for (int i = 0; i < 32; i++) {
            o[i][0] *= al0;
            o[i][1] *= al0;
            o[i][2] *= al1;
            o[i][3] *= al1;
        }
        // O += P V: k = positions (2 steps of 16), n = 32 tiles of 8 dims
#pragma unroll
        for (int ks = 0; ks < AT_K / 16; ks++) {
            const unsigned a0 = pa[2 * ks][0], a1 = pa[2 * ks][1], a2 = pa[2 * ks + 1][0], a3 = pa[2 * ks + 1][1];
#pragma unroll
            for (int dt2 = 0; dt2 < 16; dt2++) { // two 8-dim tiles per ldmatrix.x4
                const __half * vb = sV + (16 * ks + (lane & 15)) * AT_LD + 16 * dt2 + (lane >> 4) * 8;
                unsigned b0, b1, b2, b3;
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
                             : "=r"(b0), "=r"(b1), "=r"(b2), "=r"(b3)
                             : "r"((unsigned) __cvta_generic_to_shared(vb)));
                mma16816(o[2 * dt2], a0, a1, a2, a3, b0, b1);
                mma16816(o[2 * dt2 + 1], a0, a1, a2, a3, b2, b3);
            }
        }
        __syncthreads();
    }
    const float il0 = l0 > 0.f ? 1.f / l0 : 0.f, il1 = l1 > 0.f ? 1.f / l1 : 0.f;
    const int hq = g * 6 + warp;
#pragma unroll
    for (int i = 0; i < 32; i++) {
        const int d = 8 * i + 2 * t4;
        if (g8 < nq) {
            float * dst = cP.y2 + (size_t) (t0 + g8) * VD + hq * HD + d;
            dst[0] = o[i][0] * il0;
            dst[1] = o[i][1] * il0;
        }
        if (g8 + 8 < nq) {
            float * dst = cP.y2 + (size_t) (t0 + g8 + 8) * VD + hq * HD + d;
            dst[0] = o[i][2] * il1;
            dst[1] = o[i][3] * il1;
        }
    }
}

// ------------------------------------------------------------------------------------------
// pk_attn2: the same causal flash attention, restructured for sm_86.
// CTA = 16 query positions x one kv head (6 q heads x 16 rows), as pk_attn, but 12 warps: warps 2 h and
// 2 h + 1 serve q head h.  For each 32-position K / V tile, warp half j computes S for positions
// 16 j .. 16 j + 15 (each element the same k = 256 MMA chain as pk_attn) and O += P V for head dims
// 128 j .. 128 j + 127, so o takes 64 registers instead of 128 and the 4 SMSPs get 3 warps each.  The pair
// exchanges row maxima, P and partial row sums through shared memory in an order that reproduces
// pk_attn's arithmetic exactly (bit-identical output without PV16).  Loads overlap compute: V of tile i
// arrives during S of tile i, K of tile i + 1 during P V of tile i (q8: register prefetch, converted to
// fp16 in shared memory between the MMAs of the phase; fp16 cache: cp.async).
// PV16: P V of each tile accumulated in fp16 (full-rate tensor cores on GeForce), then added to the fp32
// accumulators (not exact; no measurable KLD change).  The fp16 sum of a tile is at most 32 max|v|: overflow
// would need |v| > 2047 (the largest |v| in the V cache over a 101k-token prompt is 111).
// Work units: blockIdx.x < nfull: item blockIdx.x (q tile fastest, then kv head), all tiles, final output.
// Units beyond nfull split each remaining item into S tile ranges whose partial (o, m, l) go to `part`
// and are merged by pk_attn2_comb: the last wave of equal-sized items is otherwise mostly idle.
// Microbenchmark: bench/attn_bench.cu.
constexpr int A2_XCH = 6 * 32 * (2 * 8 + 2 * 16 + 8); // per head pair: maxima, P fragments, partial sums
constexpr int A2_SMEM = (6 * A2_Q + 2 * A2_K) * A2_LD * 2 + A2_XCH;
static_assert(A2_SMEM <= 99 * 1024, "pk_attn2 shared memory");

__device__ __forceinline__ void a2_pair_sync(int hq) { asm volatile("bar.sync %0, 64;" ::"r"(1 + hq) : "memory"); }

__device__ __forceinline__ void a2_ldsm_t(unsigned (&r)[4], unsigned addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(addr));
}

// q8 K or V tile in registers: 32 rows x 32 chunks of 8 values, chunk i = tid + A2_NT k
struct A2Q8 {
    uint2 q[3];
    __half d[3];
};
// q4 (MK_KV): rows of HD / 2 bytes, chunk c = one word (kv_store_q4's nibble order) in q[k].x
// bits: 8 (q8: 8 bytes per chunk) or 4..7 (kv_store_q4 / kv_store_qn: the lane word in .x, its bits - 4 high bytes in .y)
__device__ __forceinline__ void a2_ld_q8(A2Q8 & r, const int8_t * rows, const __half * sc, int np, int bits = 8) {
#pragma unroll
    for (int k = 0; k < 3; k++) {
        const int i = threadIdx.x + A2_NT * k, p = i >> 5, c = i & 31;
        r.q[k] = make_uint2(0, 0);
        r.d[k] = __ushort_as_half(0);
        if (i < A2_K * 32 && p < np) {
            if (bits == 8) {
                r.q[k] = __ldg(reinterpret_cast<const uint2 *>(rows + (size_t) p * HD) + c);
            } else {
                const int8_t * rp = rows + (size_t) p * (HD * bits / 8);
                r.q[k].x = __ldg(reinterpret_cast<const unsigned *>(rp) + c);
                const int hb = bits - 4;
                const uint8_t * hp = reinterpret_cast<const uint8_t *>(rp) + 128 + c * hb;
                unsigned hi = 0;
                for (int b = 0; b < hb; b++) hi |= (unsigned) __ldg(hp + b) << (8 * b);
                r.q[k].y = hi;
            }
            r.d[k] = sc[(size_t) p * (HD / 32) + (c >> 2)];
        }
    }
}
// 8 q4 values (nibble n = q + 8) + fp16 scale -> 4 fp16 pairs: 0x6400 | n is 1024 + n, minus 1032 is q exactly; d * q
// rounded once to fp16, bit-identical to q4_to_f16 (decode)
__device__ __forceinline__ void a2_q4h(unsigned w, __half d, __half2 (&h)[4]) {
    const unsigned lo = w & 0x0F0F0F0Fu, hi = (w >> 4) & 0x0F0F0F0Fu;
    const __half2 off = __half2half2(__ushort_as_half(0x6408)), dd = __half2half2(d);
    h[0] = __hmul2(__hsub2(pf::u2h(__byte_perm(lo, 0x64646464u, 0x4140)), off), dd);
    h[1] = __hmul2(__hsub2(pf::u2h(__byte_perm(lo, 0x64646464u, 0x4342)), off), dd);
    h[2] = __hmul2(__hsub2(pf::u2h(__byte_perm(hi, 0x64646464u, 0x4140)), off), dd);
    h[3] = __hmul2(__hsub2(pf::u2h(__byte_perm(hi, 0x64646464u, 0x4342)), off), dd);
}
// 8 q8 values + fp16 scale -> 4 fp16 pairs, bit-identical to q8_to_f16: 0x6400 | (q ^ 0x80) is 1152 + q exactly,
// and d * q is rounded once to fp16 either way (exact in fp32 there).  No I2F (quarter rate on sm_86).
__device__ __forceinline__ void a2_q8h(uint2 q, __half d, __half2 (&h)[4]) {
    const unsigned x = q.x ^ 0x80808080u, y = q.y ^ 0x80808080u;
    const __half2 off = __half2half2(__ushort_as_half(0x6480)), dd = __half2half2(d);
    h[0] = __hmul2(__hsub2(pf::u2h(__byte_perm(x, 0x64646464u, 0x4140)), off), dd);
    h[1] = __hmul2(__hsub2(pf::u2h(__byte_perm(x, 0x64646464u, 0x4342)), off), dd);
    h[2] = __hmul2(__hsub2(pf::u2h(__byte_perm(y, 0x64646464u, 0x4140)), off), dd);
    h[3] = __hmul2(__hsub2(pf::u2h(__byte_perm(y, 0x64646464u, 0x4342)), off), dd);
}
// 8 q4..q7 values (lane word w, high bits hi: kv_store_qn's layout) + fp16 scale -> 4 fp16 pairs; bits = 4 is a2_q4h, and
// the value d * (n - 2^(bits-1)) is rounded once to fp16 like qn_to_f16 (decode)
__device__ __forceinline__ void a2_qnh(unsigned w, unsigned hi, __half d, int bits, __half2 (&h)[4]) {
    unsigned lo = w & 0x0F0F0F0Fu, hv = (w >> 4) & 0x0F0F0F0Fu;
    if (bits > 4) {
        lo |= kv_hi4(hi, 0, bits - 4);
        hv |= kv_hi4(hi, 4, bits - 4);
    }
    const __half2 off = __half2half2(__ushort_as_half((unsigned short) (0x6400 + (1 << (bits - 1))))), dd = __half2half2(d);
    h[0] = __hmul2(__hsub2(pf::u2h(__byte_perm(lo, 0x64646464u, 0x4140)), off), dd);
    h[1] = __hmul2(__hsub2(pf::u2h(__byte_perm(lo, 0x64646464u, 0x4342)), off), dd);
    h[2] = __hmul2(__hsub2(pf::u2h(__byte_perm(hv, 0x64646464u, 0x4140)), off), dd);
    h[3] = __hmul2(__hsub2(pf::u2h(__byte_perm(hv, 0x64646464u, 0x4342)), off), dd);
}
// chunk k of the tile -> fp16 rows in shared memory (bits has no default: a2_st_q8(r, dst, bits) is the whole-tile overload)
__device__ __forceinline__ void a2_st_q8(const A2Q8 & r, __half * dst, int k, int bits) {
    const int i = threadIdx.x + A2_NT * k, p = i >> 5, c = i & 31;
    if (i < A2_K * 32) {
        __half2 h[4];
        if (bits == 4) a2_q4h(r.q[k].x, r.d[k], h);
        else if (bits != 8) a2_qnh(r.q[k].x, r.q[k].y, r.d[k], bits, h);
        else a2_q8h(r.q[k], r.d[k], h);
        *reinterpret_cast<int4 *>(dst + p * A2_LD + 8 * c) = *reinterpret_cast<int4 *>(h);
    }
}
__device__ __forceinline__ void a2_st_q8(const A2Q8 & r, __half * dst, int bits) {
#pragma unroll
    for (int k = 0; k < 3; k++) a2_st_q8(r, dst, k, bits);
}
// fp16 K or V tile -> shared memory (cp.async, one commit group); rows >= np zeroed
__device__ __forceinline__ void a2_cp_f16(__half * dst, const __half * src, int np) {
    for (int i = threadIdx.x; i < A2_K * 32; i += A2_NT) {
        const int p = i >> 5, c = i & 31;
        if (p < np) cp_async16(dst + p * A2_LD + 8 * c, src + (size_t) p * HD + 8 * c);
        else *reinterpret_cast<int4 *>(dst + p * A2_LD + 8 * c) = make_int4(0, 0, 0, 0);
    }
    pf::cpa_commit();
}

template <bool Q8, bool PV16>
__global__ void __launch_bounds__(A2_NT, 1) pk_attn2(int T, int n0, int slot, int nfull, int S, float * __restrict__ part, int r0 = 0) { MK_VIEW_INIT();
    extern __shared__ __align__(16) uint8_t asm_[];
    __half * sQ = reinterpret_cast<__half *>(asm_); // [96][A2_LD]
    __half * sK = sQ + 6 * A2_Q * A2_LD;            // [32][A2_LD]
    __half * sV = sK + A2_K * A2_LD;                // [32][A2_LD]
    float2 * xmax = reinterpret_cast<float2 *>(sV + A2_K * A2_LD); // [6][2][32] row maxima (finally: l)
    uint4 * xp = reinterpret_cast<uint4 *>(xmax + 6 * 2 * 32);     // [6][2][32] P fragments
    float2 * xsum = reinterpret_cast<float2 *>(xp + 6 * 2 * 32);   // [6][32] half 0's partial row sums
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g8 = lane >> 2, t4 = lane & 3;
    const int hq = warp >> 1, hf = warp & 1;
    // work unit -> (q tile, kv head, tile range)
    const int nqt = (T + A2_Q - 1) / A2_Q;
    int item = blockIdx.x, seg = 0, nseg = 1;
    if (item >= nfull) {
        const int v = item - nfull;
        item = nfull + v / S;
        seg = v % S;
        nseg = S;
    }
    const int qt = item % nqt, g = item / nqt;
    const int t0 = qt * A2_Q, nq = min(A2_Q, T - t0);
    const int qpos0 = n0 + t0, pmax = qpos0 + nq - 1;
    const int ntile = pmax / A2_K + 1;
    const int it0 = ntile * seg / nseg, it1 = ntile * (seg + 1) / nseg;
    const size_t row0 = ((size_t) slot * NKV + g) * cB.ctx;
    // q8 cache: the slot's K / V tiles in their MK_KV formats (rows of HD bits / 8 bytes), or the MK_KV_WIN q8 ring for
    // tiles at positions >= wsr (kv_src per tile); kbi / vbi = the bits of the K / V tile in flight (rq)
    const int wsr = Q8 ? kv_ws(cP.wsrow ? pmax + 1 : n0 + T) : 0, wsm = Q8 ? kv_wsm(wsr) : 0; // (MK_EXACT: per query tile)
    int kbi = 8, vbi = 8;
    const int8_t * tq = nullptr;
    const __half * ts = nullptr;
    const __half * Kh = static_cast<const __half *>(cB.kc) + row0 * HD;
    const __half * Vh = static_cast<const __half *>(cB.vc) + row0 * HD;

    for (int i = tid; i < 6 * A2_Q * 32; i += A2_NT) { // 96 rows x 32 chunks of 16 B
        const int r = i >> 5, c = i & 31, h = r / A2_Q, tt = r % A2_Q;
        __half * d = sQ + r * A2_LD + 8 * c;
        if (tt < nq) cp_async16(d, cP.qh + ((size_t) (r0 + t0 + tt) * NH + g * 6 + h) * HD + 8 * c);
        else *reinterpret_cast<int4 *>(d) = make_int4(0, 0, 0, 0);
    }
    pf::cpa_commit();
    A2Q8 rq; // q8: V of the current tile, or K of the next one, in flight
    if (it0 < it1) {
        const int kb = it0 * A2_K, np = min(A2_K, pmax + 1 - kb);
        if constexpr (Q8) {
            kbi = kv_src(0, slot, g, kb, wsr, wsm, tq, ts);
            a2_ld_q8(rq, tq, ts, np, kbi);
            a2_st_q8(rq, sK, kbi);
            vbi = kv_src(1, slot, g, kb, wsr, wsm, tq, ts);
            a2_ld_q8(rq, tq, ts, np, vbi);
        } else {
            a2_cp_f16(sK, Kh + (size_t) kb * HD, np);
        }
    }

    float o[16][4];
#pragma unroll
    for (int j = 0; j < 16; j++) o[j][0] = o[j][1] = o[j][2] = o[j][3] = 0.f;
    float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.f, l1 = 0.f; // rows g8, g8 + 8 (l: kept by half 1)
    const int qr0 = qpos0 + g8, qr1 = qpos0 + g8 + 8;
    const unsigned qa = (unsigned) __cvta_generic_to_shared(sQ + (hq * A2_Q + (lane & 15)) * A2_LD + (lane >> 4) * 8);
    const unsigned ka = (unsigned) __cvta_generic_to_shared(sK + (16 * hf + ((lane >> 4) << 3) + (lane & 7)) * A2_LD + ((lane >> 3) & 1) * 8);
    const unsigned va = (unsigned) __cvta_generic_to_shared(sV + (lane & 15) * A2_LD + 128 * hf + (lane >> 4) * 8);
    float2 * mx_own = xmax + (2 * hq + hf) * 32 + lane;
    const float2 * mx_oth = xmax + (2 * hq + (hf ^ 1)) * 32 + lane;
    uint4 * p_own = xp + (2 * hq + hf) * 32 + lane;
    const uint4 * p_oth = xp + (2 * hq + (hf ^ 1)) * 32 + lane;
    float2 * xs = xsum + hq * 32 + lane;

    for (int it = it0; it < it1; it++) {
        const int kb = it * A2_K;
        const int kn = kb + A2_K, npn = min(A2_K, pmax + 1 - kn); // next tile
        pf::cpa_wait<0>(); // Q (first tile); fp16: K of this tile
        __syncthreads();   // sK = K(it); every warp is done with P V of it - 1, so sV is free
        if constexpr (!Q8) a2_cp_f16(sV, Vh + (size_t) kb * HD, min(A2_K, pmax + 1 - kb));
        // S = Q K^T for this warp's 16 positions (n8 tiles ni = 2 hf, 2 hf + 1 of pk_attn)
        float s[2][4];
#pragma unroll
        for (int ni = 0; ni < 2; ni++) s[ni][0] = s[ni][1] = s[ni][2] = s[ni][3] = 0.f;
#pragma unroll
        for (int ks = 0; ks < HD / 16; ks++) {
            unsigned a[4], b[4];
            pf::ldsm_x4(a, qa + 32 * ks);
            pf::ldsm_x4(b, ka + 32 * ks);
            mma16816(s[0], a[0], a[1], a[2], a[3], b[0], b[1]);
            mma16816(s[1], a[0], a[1], a[2], a[3], b[2], b[3]);
            // V(it) -> sV (free since the barrier above) between the MMAs, late enough for the loads to have landed
            if constexpr (Q8)
                if (ks >= 6 && (ks & 3) == 2) a2_st_q8(rq, sV, (ks - 6) >> 2, vbi);
        }
        // causal mask, row maxima over the pair
        float mx0 = -INFINITY, mx1 = -INFINITY;
#pragma unroll
        for (int ni = 0; ni < 2; ni++) {
            const int p = kb + 16 * hf + 8 * ni + 2 * t4;
            if (p > qr0) s[ni][0] = -INFINITY;
            if (p + 1 > qr0) s[ni][1] = -INFINITY;
            if (p > qr1) s[ni][2] = -INFINITY;
            if (p + 1 > qr1) s[ni][3] = -INFINITY;
            mx0 = fmaxf(mx0, fmaxf(s[ni][0], s[ni][1]));
            mx1 = fmaxf(mx1, fmaxf(s[ni][2], s[ni][3]));
        }
        mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 1));
        mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 2));
        mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 1));
        mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 2));
        *mx_own = make_float2(mx0, mx1);
        a2_pair_sync(hq);
        {
            const float2 t = *mx_oth;
            mx0 = fmaxf(mx0, t.x);
            mx1 = fmaxf(mx1, t.y);
        }
        const float mn0 = fmaxf(m0, mx0), mn1 = fmaxf(m1, mx1);
        const float al0 = mn0 == -INFINITY ? 1.f : expf(m0 - mn0), al1 = mn1 == -INFINITY ? 1.f : expf(m1 - mn1);
        float x0[2], x1[2]; // per n8 tile: p0 + p1 (row g8), p2 + p3 (row g8 + 8)
        unsigned pa[2][2];
#pragma unroll
        for (int ni = 0; ni < 2; ni++) {
            const float p0 = mn0 == -INFINITY ? 0.f : expf(s[ni][0] - mn0);
            const float p1 = mn0 == -INFINITY ? 0.f : expf(s[ni][1] - mn0);
            const float p2 = mn1 == -INFINITY ? 0.f : expf(s[ni][2] - mn1);
            const float p3 = mn1 == -INFINITY ? 0.f : expf(s[ni][3] - mn1);
            x0[ni] = p0 + p1;
            x1[ni] = p2 + p3;
            __half2 h01 = __floats2half2_rn(p0, p1), h23 = __floats2half2_rn(p2, p3);
            pa[ni][0] = *reinterpret_cast<unsigned *>(&h01);
            pa[ni][1] = *reinterpret_cast<unsigned *>(&h23);
        }
        const uint4 pown = make_uint4(pa[0][0], pa[0][1], pa[1][0], pa[1][1]);
        *p_own = pown;
        if (hf == 0) { // pk_attn's per-thread row sums run over n8 tiles 0, 1, 2, 3 in order: half 0 starts them
            float ps0 = 0.f, ps1 = 0.f;
            ps0 += x0[0];
            ps0 += x0[1];
            ps1 += x1[0];
            ps1 += x1[1];
            *xs = make_float2(ps0, ps1);
        }
        a2_pair_sync(hq);
        const uint4 poth = *p_oth;
        if (hf == 1) { // ... and half 1 finishes them
            const float2 t = *xs;
            float ps0 = t.x, ps1 = t.y;
            ps0 += x0[0];
            ps0 += x0[1];
            ps1 += x1[0];
            ps1 += x1[1];
            ps0 += __shfl_xor_sync(0xffffffffu, ps0, 1);
            ps0 += __shfl_xor_sync(0xffffffffu, ps0, 2);
            ps1 += __shfl_xor_sync(0xffffffffu, ps1, 1);
            ps1 += __shfl_xor_sync(0xffffffffu, ps1, 2);
            l0 = l0 * al0 + ps0;
            l1 = l1 * al1 + ps1;
        }
        m0 = mn0;
        m1 = mn1;
        const uint4 P0 = hf ? poth : pown, P1 = hf ? pown : poth; // A fragments of positions 0-15 / 16-31
        if constexpr (Q8) {
            if (it + 1 < it1) {
                kbi = kv_src(0, slot, g, kn, wsr, wsm, tq, ts);
                a2_ld_q8(rq, tq, ts, npn, kbi);
            }
        } else {
            pf::cpa_wait<0>(); // V(it)
        }
        __syncthreads(); // sV = V(it); every warp is done with S, so sK is free
        if constexpr (!Q8)
            if (it + 1 < it1) a2_cp_f16(sK, Kh + (size_t) kn * HD, npn);
        // O += P V for this warp's 128 dims: n8 tiles 16 hf + 0 .. 15
        const bool nxt = it + 1 < it1;
        if constexpr (!PV16) {
            if (!__all_sync(0xffffffffu, al0 == 1.f && al1 == 1.f)) { // o * 1 == o: skipping is exact
#pragma unroll
                for (int j = 0; j < 16; j++) {
                    o[j][0] *= al0;
                    o[j][1] *= al0;
                    o[j][2] *= al1;
                    o[j][3] *= al1;
                }
            }
#pragma unroll
            for (int ks = 0; ks < 2; ks++) {
                const uint4 A = ks ? P1 : P0;
#pragma unroll
                for (int j = 0; j < 8; j++) {
                    unsigned b[4];
                    a2_ldsm_t(b, va + ks * 16 * A2_LD * 2 + 32 * j);
                    mma16816(o[2 * j], A.x, A.y, A.z, A.w, b[0], b[1]);
                    mma16816(o[2 * j + 1], A.x, A.y, A.z, A.w, b[2], b[3]);
                    // K(it + 1) -> sK (free since the barrier above) late in the loop
                    if constexpr (Q8)
                        if (ks == 1 && (j & 1) && j < 7 && nxt) a2_st_q8(rq, sK, j >> 1, kbi);
                }
            }
        } else {
            const unsigned A0[4] = {P0.x, P0.y, P0.z, P0.w}, A1[4] = {P1.x, P1.y, P1.z, P1.w};
#pragma unroll
            for (int j = 0; j < 8; j++) {
                unsigned b[4], h0[2], h1[2];
                a2_ldsm_t(b, va + 32 * j);
                pf::hmma16z(h0, A0, b[0], b[1]);
                pf::hmma16z(h1, A0, b[2], b[3]);
                a2_ldsm_t(b, va + 16 * A2_LD * 2 + 32 * j);
                pf::hmma16(h0, A1, b[0], b[1]);
                pf::hmma16(h1, A1, b[2], b[3]);
                const float2 u0 = __half22float2(pf::u2h(h0[0])), u1 = __half22float2(pf::u2h(h0[1]));
                const float2 w0 = __half22float2(pf::u2h(h1[0])), w1 = __half22float2(pf::u2h(h1[1]));
                o[2 * j][0] = o[2 * j][0] * al0 + u0.x;
                o[2 * j][1] = o[2 * j][1] * al0 + u0.y;
                o[2 * j][2] = o[2 * j][2] * al1 + u1.x;
                o[2 * j][3] = o[2 * j][3] * al1 + u1.y;
                o[2 * j + 1][0] = o[2 * j + 1][0] * al0 + w0.x;
                o[2 * j + 1][1] = o[2 * j + 1][1] * al0 + w0.y;
                o[2 * j + 1][2] = o[2 * j + 1][2] * al1 + w1.x;
                o[2 * j + 1][3] = o[2 * j + 1][3] * al1 + w1.y;
                if constexpr (Q8)
                    if (j >= 4 && j < 7 && nxt) a2_st_q8(rq, sK, j - 4, kbi); // K(it + 1) -> sK late in the loop
            }
        }
        if constexpr (Q8) {
            if (nxt) {
                vbi = kv_src(1, slot, g, kn, wsr, wsm, tq, ts);
                a2_ld_q8(rq, tq, ts, npn, vbi);
            }
        }
    }
    pf::cpa_wait<0>();
    // row sums: half 1 -> half 0 (the maxima slot is no longer read)
    if (hf) *mx_own = make_float2(l0, l1);
    a2_pair_sync(hq);
    if (!hf) {
        const float2 t = *mx_oth;
        l0 = t.x;
        l1 = t.y;
    }
    const int head = g * 6 + hq;
    if (nseg == 1) {
        const float il0 = l0 > 0.f ? 1.f / l0 : 0.f, il1 = l1 > 0.f ? 1.f / l1 : 0.f;
#pragma unroll
        for (int j = 0; j < 16; j++) {
            const int d = 128 * hf + 8 * j + 2 * t4;
            if (g8 < nq) *reinterpret_cast<float2 *>(cP.y2 + (size_t) (r0 + t0 + g8) * VD + head * HD + d) = make_float2(o[j][0] * il0, o[j][1] * il0);
            if (g8 + 8 < nq) *reinterpret_cast<float2 *>(cP.y2 + (size_t) (r0 + t0 + g8 + 8) * VD + head * HD + d) = make_float2(o[j][2] * il1, o[j][3] * il1);
        }
    } else {
        float * pp = part + (size_t) (blockIdx.x - nfull) * A2_PART;
        const int r0 = hq * A2_Q + g8;
#pragma unroll
        for (int j = 0; j < 16; j++) {
            const int d = 128 * hf + 8 * j + 2 * t4;
            *reinterpret_cast<float2 *>(pp + (size_t) r0 * HD + d) = make_float2(o[j][0], o[j][1]);
            *reinterpret_cast<float2 *>(pp + (size_t) (r0 + 8) * HD + d) = make_float2(o[j][2], o[j][3]);
        }
        if (hf && t4 == 0) {
            float2 * ml = reinterpret_cast<float2 *>(pp + 6 * A2_Q * HD);
            ml[r0] = make_float2(m0, l0);
            ml[r0 + 8] = make_float2(m1, l1);
        }
    }
}

// merge the S partials of each split item (block = item nfull + blockIdx.x) into y2: warp w of block row y merges rows
// w + 8 y, w + 8 (y + gridDim.y), .. (gridDim.y 12: one row per warp; 1: twelve, MK_PFX bit 256), each with the same code
__global__ void __launch_bounds__(256) pk_attn2_comb(int T, int nfull, int S, const float * __restrict__ part) { MK_VIEW_INIT();
    const int item = nfull + blockIdx.x, nqt = (T + A2_Q - 1) / A2_Q, qt = item % nqt, g = item / nqt;
    const int t0 = qt * A2_Q, nq = min(A2_Q, T - t0);
    const float * P = part + (size_t) blockIdx.x * S * A2_PART;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    for (int r = warp + 8 * blockIdx.y; r < 6 * A2_Q; r += 8 * gridDim.y) {
        const int h = r / A2_Q, tt = r % A2_Q;
        if (tt >= nq) continue;
        float m = -INFINITY;
        for (int s = 0; s < S; s++) m = fmaxf(m, P[(size_t) s * A2_PART + 6 * A2_Q * HD + 2 * r]);
        float L = 0.f, acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
        for (int s = 0; s < S; s++) {
            const float * ps = P + (size_t) s * A2_PART;
            const float ms = ps[6 * A2_Q * HD + 2 * r], ls = ps[6 * A2_Q * HD + 2 * r + 1];
            const float w = ms == -INFINITY ? 0.f : expf(ms - m);
            L += w * ls;
            const float4 a = *reinterpret_cast<const float4 *>(ps + (size_t) r * HD + 8 * lane);
            const float4 b = *reinterpret_cast<const float4 *>(ps + (size_t) r * HD + 8 * lane + 4);
            acc[0] += w * a.x; acc[1] += w * a.y; acc[2] += w * a.z; acc[3] += w * a.w;
            acc[4] += w * b.x; acc[5] += w * b.y; acc[6] += w * b.z; acc[7] += w * b.w;
        }
        const float il = L > 0.f ? 1.f / L : 0.f;
        float * dst = cP.y2 + (size_t) (t0 + tt) * VD + (g * 6 + h) * HD + 8 * lane;
        *reinterpret_cast<float4 *>(dst) = make_float4(acc[0] * il, acc[1] * il, acc[2] * il, acc[3] * il);
        *reinterpret_cast<float4 *>(dst + 4) = make_float4(acc[4] * il, acc[5] * il, acc[6] * il, acc[7] * il);
    }
}

// ------------------------------------------------------------------------------------------
// pk_attn8: pk_attn2's PV16 path on the q8 cache, with S = Q K^T on int8 tensor cores (mma m16n8k32 s8, int32
// accumulation) straight from the cache's int8 K: no K conversion, half the S operand bytes, and int8 MMAs at 4x
// the rate of pk_attn2's fp32-accumulated fp16 MMAs.  Q is quantized in the prologue per (row, 32-dim block) like
// q8_0 (QROW: one scale per row) and stays in registers as int8 A fragments.  Each block's int32 partial is
// converted (I2FP) and scaled by q scale x k scale in fp32 (8 partials per score at HD 256; log2 e is folded into
// the q scales and P = exp2(s - m)).
// With Q out of shared memory, the raw q8 K / V tiles (and V scales) stream through a 3-stage cp.async ring; K scales
// go through a register into an fp32 [block][position] table.  V(it + 1) is converted to fp16 during tile it into
// the other half of a double-buffered fp16 V tile, so there is one block barrier per tile (pk_attn2: two).
// Not exact (Q quantization; no measurable KLD change).  Same work units, split and partials as pk_attn2
// (pk_attn2_comb merges; partials carry m in natural-log units).  Microbenchmark: bench/attn_bench.cu.
constexpr int A8_NS = 3;             // ring stages (tile it + 1 must have landed at tile it's barrier)
constexpr int A8_KLD = HD + 16;      // int8 K row stride (bytes): the 8 rows of an ldmatrix hit distinct banks
constexpr int A8_KB = A2_K * A8_KLD; // K tile
constexpr int A8_KSB = 8 * A2_K * 4; // K scales, fp32 [8 blocks][32 positions]
constexpr int A8_VB = A2_K * HD;     // V tile (int8)
constexpr int A8_VSB = A2_K * 8 * 2; // V scales, fp16 [32 positions][8 blocks] (cache layout)
constexpr int A8_STG = A8_KB + A8_KSB + A8_VB + A8_VSB;
constexpr int A8_SMEM = A8_NS * A8_STG + 2 * A2_K * A2_LD * 2 + A2_XCH; // + 2 fp16 V tiles + exchange
static_assert(A8_SMEM <= 99 * 1024, "pk_attn8 shared memory");
// int32 accumulation from zero; the int32 -> fp32 conversion compiles to I2FP.F32.S32 (full rate on sm_86, unlike I2F)
__device__ __forceinline__ void a8_imma(int (&d)[4], const unsigned (&a)[4], unsigned b0, unsigned b1) {
    asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%10,%10,%10};"
        : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(0));
}
__device__ __forceinline__ float a8_i2f(int d) { return __int2float_rn(d); }

// chunk k (of 3) of this thread's cp.async share of tile rows kb .. kb + np - 1 (K, V and V scales) into ring stage
// st, rows >= np zeroed.  Chunk kinds are warp-uniform (i = tid + 384 k: K < 512 <= V < 1024 <= V scales < 1056).
// kbits / vbits: the tile's K / V format (8, or 4..7: rows of HD bits / 8 bytes = 2 bits chunks, staged at the same row
// offsets; a low-bit K tile is expanded to int8 in place before its S MMAs, kv_tile_expand)
__device__ __forceinline__ void a8_issue_k(uint8_t * st, const int8_t * Kq, const int8_t * Vq, const __half * Vs, int kb, int np, int k, int kbits, int vbits) {
    const int i = threadIdx.x + A2_NT * k;
    if (i < 33 * A2_K) {
        int p, off;
        const void * src;
        if (i < 16 * A2_K) {
            if ((i & 15) >= 2 * kbits) return; // beyond a low-bit row's bytes (overwritten by the expansion)
            p = i >> 4;
            off = p * A8_KLD + 16 * (i & 15);
            src = Kq + (size_t) (kb + p) * (HD * kbits / 8) + 16 * (i & 15);
        } else if (i < 32 * A2_K) {
            const int j = i - 16 * A2_K;
            if ((j & 15) >= 2 * vbits) return; // beyond a low-bit row's bytes (never read)
            p = j >> 4;
            off = A8_KB + A8_KSB + p * HD + 16 * (j & 15);
            src = Vq + (size_t) (kb + p) * (HD * vbits / 8) + 16 * (j & 15);
        } else {
            p = i - 32 * A2_K;
            off = A8_KB + A8_KSB + A8_VB + 16 * p;
            src = Vs + (size_t) (kb + p) * (HD / 32);
        }
        if (p < np) cp_async16(st + off, src);
        else *reinterpret_cast<int4 *>(st + off) = make_int4(0, 0, 0, 0);
    }
}
__device__ __forceinline__ void a8_issue(uint8_t * st, const int8_t * Kq, const int8_t * Vq, const __half * Vs, int kb, int np, int kbits, int vbits) {
#pragma unroll
    for (int k = 0; k < 3; k++) a8_issue_k(st, Kq, Vq, Vs, kb, np, k, kbits, vbits);
    pf::cpa_commit();
}
// K scales of tile rows kb ..: thread t = tid - 256 (warps 8 - 11, which have no third a8_cv_v / V chunk) loads the fp16
// pair of position t >> 2, blocks 2 (t & 3) and 2 (t & 3) + 1 as raw bits (rows of 16 B: 4-B aligned); a8_st_ks converts
// them and stores them transposed as fp32. The conversion sits at the store, a tile later: converting right after the
// load stalled warps 0 - 7 on the load's latency before the S MMAs of every tile (and warps 8 - 11 at the next barrier).
constexpr int A8_KST = A2_NT - 4 * A2_K; // first thread of the K-scale loads
__device__ __forceinline__ unsigned a8_ld_ks(const __half * Ks, int kb, int np) {
    const int t = (int) threadIdx.x - A8_KST, p = t >> 2;
    if (t < 0 || p >= np) return 0u;
    return __ldg(reinterpret_cast<const unsigned *>(Ks + (size_t) (kb + p) * (HD / 32)) + (t & 3));
}
__device__ __forceinline__ void a8_st_ks(uint8_t * st, unsigned v) {
    const int t = (int) threadIdx.x - A8_KST;
    if (t >= 0) {
        const float2 f = __half22float2(pf::u2h(v));
        float * ks = reinterpret_cast<float *>(st + A8_KB) + 2 * (t & 3) * A2_K + (t >> 2);
        ks[0] = f.x;
        ks[A2_K] = f.y;
    }
}
// chunk k (of 3) of this thread: 8 V values of ring stage st -> fp16 in sV (a2_q8h / a2_q4h / a2_qnh, as pk_attn2).
// VB (4 .. 7): vbits is 8 or VB (the slot's low-bit V, known at compile time); 0: any
template <int VB = 0>
__device__ __forceinline__ void a8_cv_v(const uint8_t * st, __half * sV, int k, int vbits) {
    const int i = threadIdx.x + A2_NT * k, p = i >> 5, c = i & 31;
    if (i < A2_K * 32) {
        const uint8_t * row = st + A8_KB + A8_KSB + p * HD;
        const __half d = reinterpret_cast<const __half *>(st + A8_KB + A8_KSB + A8_VB)[p * (HD / 32) + (c >> 2)];
        __half2 h[4];
        const int vb = VB ? VB : vbits; // (when not 8)
        if (vbits == 8) a2_q8h(*reinterpret_cast<const uint2 *>(row + 8 * c), d, h);
        else if (vb == 4) a2_q4h(*reinterpret_cast<const unsigned *>(row + 4 * c), d, h);
        else a2_qnh(*reinterpret_cast<const unsigned *>(row + 4 * c), kv_hi(row + 128 + c * (vb - 4), vb - 4), d, vb, h);
        *reinterpret_cast<int4 *>(sV + p * A2_LD + 8 * c) = *reinterpret_cast<int4 *>(h);
    }
}

// scores in log2 units: log2 e is folded into the Q scales, P = exp2(s - m)
constexpr float A8_L2E = 1.4426950408889634f;
// exp2f(x), or v where the row maximum mn is still -inf (x is NaN there): computed unconditionally, then selected. The
// ternary `mn == -inf ? v : exp2f(x)` compiled to a branch (BSSY / BRA / BSYNC) around every pair of exp2f; same values
__device__ __forceinline__ float a8_ex2(float x, float mn, float v) {
    const float e = exp2f(x);
    return mn == -INFINITY ? v : e;
}
// KLOW: the slot's K is low-bit (q4..q7): its tiles are expanded to int8 in the staging. The KLOW = false instantiation is the
// plain q8 kernel (no per-tile format checks: the host picks the instantiation per slot). KB / VB (4 .. 7, 0 = read per slot):
// the slot's main-buffer K / V bits at compile time (a tile is then KB / VB or 8, the q8 window ring): folds the expansion's
// and the V conversion's format switches and a8_issue_k's chunk predicates (same arithmetic: the same bits as KB = VB = 0).
// EXA (MK_EXACT): a row's result is a function of its position only, whatever the chunk / launch / split: its key range
// is cut into absolute segments of segt tiles (segment k = tiles k segt ..), each computed from a fresh online softmax,
// and folded in order: R <- ex_fold(R, s_k) (ex_ab / ex_f; max in log2 units). A unit computes either a whole item
// (folding in registers + R in y2), the head of a split item (segments 0 .. nsg - Q - 1, R -> its partial slot) or
// one tail segment (-> its slot); pk_attn8_combx folds the tails into the head's R in segment order.
__device__ __forceinline__ void ex_ab(float M, float m, float & Mn, float & a, float & b) {
    Mn = fmaxf(M, m);
    a = M == -INFINITY ? 0.f : exp2f(M - Mn);
    b = m == -INFINITY ? 0.f : exp2f(m - Mn);
}
__device__ __forceinline__ float ex_f(float a, float R, float b, float s) { return __fadd_rn(__fmul_rn(a, R), __fmul_rn(b, s)); }

template <bool QROW, bool KLOW, bool EXA = false, int KB = 0, int VB = 0>
__global__ void __launch_bounds__(A2_NT, 1) pk_attn8(int T, int n0, int slot, int nfull, int S, float * __restrict__ part, int r0 = 0,
                                                     int segt = 0, int exh = 0, float * __restrict__ part2 = nullptr, int exa = 0) { MK_VIEW_INIT();
    static_assert(KB == 0 || (KLOW && KB >= 4 && KB <= 7), "pk_attn8 KB");
    static_assert(VB == 0 || (VB >= 4 && VB <= 7), "pk_attn8 VB");
    extern __shared__ __align__(16) uint8_t asm_[];
    __half * sV0 = reinterpret_cast<__half *>(asm_ + A8_NS * A8_STG); // [2][32][A2_LD] fp16 V(it) / V(it + 1)
    float2 * xmax = reinterpret_cast<float2 *>(sV0 + 2 * A2_K * A2_LD); // [6][2][32] row maxima (finally: l)
    uint4 * xp = reinterpret_cast<uint4 *>(xmax + 6 * 2 * 32);       // [6][2][32] P fragments
    float2 * xsum = reinterpret_cast<float2 *>(xp + 6 * 2 * 32);     // [6][32] half 0's partial row sums
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g8 = lane >> 2, t4 = lane & 3;
    const int hq = warp >> 1, hf = warp & 1;
    const int nqt = (T + A2_Q - 1) / A2_Q;
    int item = blockIdx.x, seg = 0, nseg = 1;
    if (item >= nfull) {
        const int v = item - nfull;
        if (EXA) { // units interleaved over the split items: every head, then every item's first single segment, ..
            const int nsplit = nqt * NKV - nfull;
            item = nfull + v % nsplit;
            seg = v / nsplit;
        } else {
            item = nfull + v / S;
            seg = v % S;
        }
        nseg = S;
    }
    const int qt = item % nqt, g = item / nqt;
    const int t0 = qt * A2_Q, nq = min(A2_Q, T - t0);
    const int qpos0 = n0 + t0, pmax = qpos0 + nq - 1;
    const int ntile = pmax / A2_K + 1;
    int it0 = ntile * seg / nseg, it1 = ntile * (seg + 1) / nseg;
    // EXA: 0 whole item (final output); split item (S units, S partial slots): unit 0 = the head, segments 0 .. exh - 1
    // (folds, R -> slot 0), role 1; unit u >= 1 = segment exh + u - 1 alone (-> slot u), role 2. (Role 2 code also
    // handles a unit of several segments, each to slot 1 + k - ex_h.) Split item j's S slots: in part for j < exa, else
    // in part2 (two idle scratch buffers, see MKEngine's ex_part).
    int ex_role = 0, ex_h = 0;
    float * ex_base = nullptr;
    if (EXA) {
        const int nsg = (ntile + segt - 1) / segt;
        it0 = 0;
        it1 = ntile;
        if (nseg > 1) {
            const int k0 = seg == 0 ? 0 : exh + seg - 1, k1 = seg == 0 ? exh : k0 + 1;
            if (k0 >= nsg) return; // (idle unit: this item has fewer segments than the plan's longest)
            it0 = k0 * segt;
            it1 = min(ntile, k1 * segt);
            ex_h = exh;
            ex_role = seg == 0 ? 1 : 2;
            const int j = item - nfull;
            ex_base = j < exa ? part + (size_t) j * S * A2_PART : part2 + (size_t) (j - exa) * S * A2_PART;
        }
    }
    // this slot's q8_0 K / V through the MK_KV slot pointers (as pk_attn2): slots are packed in kc / vc, so with a q4
    // slot before this one the uniform [slot][head][ctx][HD] offset lands in another slot's rows; for an all-q8
    // cache kvp / kvs give exactly the [slot][head][ctx][HD] addresses
    // Per tile (kv_src): the slot's main buffer (K q8 or q4..q7, expanded to int8 in its stage by kv_tile_expand during the
    // previous tile; V q8 or q4..q7, converted to fp16 by a8_cv_v) or, with MK_KV_WIN, the q8 ring for tiles at positions
    // >= wsr. kbt(it) / vbt(it) = the K / V format of tile it.
    const int wsr = kv_ws(cP.wsrow ? pmax + 1 : n0 + T), wsm = kv_wsm(wsr); // (MK_EXACT: per query tile, see PBufs)
    const KvFmt kf = kv_fmt(slot); // per-slot formats once; per tile one compare of the tile index (kv_ring_tile)
    const int itr_k = kv_ring_tile(kf.kr, wsr, 0, A2_K), itr_v = kv_ring_tile(kf.vr, wsr, 0, A2_K);
    const int kbs = KB ? KB : kf.kb, vbs = VB ? VB : kf.vb; // the main buffer's K / V bits
    auto kbt = [&](int it) { return it >= itr_k ? 8 : kbs; };
    auto vbt = [&](int it) { return it >= itr_v ? 8 : vbs; };
    auto issue_tile = [&](uint8_t * st, int it, int np) { // K, V, V scales of tile it -> stage st; returns this thread's K scales (raw)
        const int8_t *tk, *tv;
        const __half *tks, *tvs;
        kv_tile_src(0, slot, g, it * A2_K, wsr, wsm, kbs, it >= itr_k, tk, tks);
        kv_tile_src(1, slot, g, it * A2_K, wsr, wsm, vbs, it >= itr_v, tv, tvs);
        a8_issue(st, tk, tv, tvs, 0, np, KLOW ? kbt(it) : 8, vbt(it));
        return a8_ld_ks(tks, 0, np);
    };
    auto expand_k = [&](int it, int r0, int rs) { // low-bit K rows r0, r0 + rs, .. of tile it (landed) -> int8 in place
        if constexpr (KLOW) {
            // all A2_K rows: rows >= np were zeroed by a8_issue_k (finite values, masked by the causal test)
            const int kb_ = kbt(it);
            if constexpr (KB != 0) {
                if (kb_ != 8)
                    for (int r = r0; r < A2_K; r += 3 * rs) kv_rows_expand<KB, true>(asm_ + (it % A8_NS) * A8_STG, A2_K, r, rs, A8_KLD);
            } else if (kb_ != 8) kv_tile_expand<true>(asm_ + (it % A8_NS) * A8_STG, kb_, A2_K, r0, rs, A8_KLD);
        }
    };
    (void) kbt;

    // ring prologue: tiles it0 .. it0 + A8_NS - 2
#pragma unroll
    for (int j = 0; j < A8_NS - 1; j++) {
        const int it = it0 + j;
        uint8_t * st = asm_ + (it % A8_NS) * A8_STG;
        if (it < it1) {
            const int kb = it * A2_K, np = min(A2_K, pmax + 1 - kb);
            a8_st_ks(st, issue_tile(st, it, np));
        } else {
            pf::cpa_commit();
        }
    }

    // Q of head g * 6 + hq, rows g8 and g8 + 8 -> int8 A fragments qf[block] (m16n8k32: a0 / a1 = dims 4 t4 ..
    // of rows g8 / g8 + 8, a2 / a3 = dims 16 + 4 t4 ..), scales qs (QROW: per row)
    unsigned qf[8][4];
    float qs[QROW ? 1 : 8][2];
    {
        const __half * qg = cP.qh + ((size_t) (r0 + t0) * NH + g * 6 + hq) * HD + 4 * t4;
        auto ldq = [&](int b, int rr, float (&v)[8]) {
            const int t = g8 + 8 * rr;
            uint2 lo = make_uint2(0, 0), hi = make_uint2(0, 0);
            if (t < nq) {
                const __half * src = qg + (size_t) t * NH * HD + 32 * b;
                lo = __ldg(reinterpret_cast<const uint2 *>(src));
                hi = __ldg(reinterpret_cast<const uint2 *>(src + 16));
            }
            float2 f = __half22float2(pf::u2h(lo.x));
            v[0] = f.x; v[1] = f.y;
            f = __half22float2(pf::u2h(lo.y));
            v[2] = f.x; v[3] = f.y;
            f = __half22float2(pf::u2h(hi.x));
            v[4] = f.x; v[5] = f.y;
            f = __half22float2(pf::u2h(hi.y));
            v[6] = f.x; v[7] = f.y;
        };
        auto qmax = [&](const float (&v)[8], float a) {
#pragma unroll
            for (int k = 0; k < 8; k++) a = fmaxf(a, fabsf(v[k]));
            a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, 1));
            return fmaxf(a, __shfl_xor_sync(0xffffffffu, a, 2));
        };
        float am[2] = {0.f, 0.f};
        if constexpr (QROW) {
#pragma unroll
            for (int b = 0; b < 8; b++)
#pragma unroll
                for (int rr = 0; rr < 2; rr++) {
                    float v[8];
                    ldq(b, rr, v);
                    am[rr] = qmax(v, am[rr]);
                }
            qs[0][0] = am[0] * (A8_L2E / 127.f);
            qs[0][1] = am[1] * (A8_L2E / 127.f);
        }
#pragma unroll
        for (int b = 0; b < 8; b++)
#pragma unroll
            for (int rr = 0; rr < 2; rr++) {
                float v[8];
                ldq(b, rr, v);
                float a = am[rr];
                if constexpr (!QROW) {
                    a = qmax(v, 0.f);
                    qs[b][rr] = a * (A8_L2E / 127.f);
                }
                const float inv = a > 0.f ? 127.f / a : 0.f;
                int w[8]; // low byte of 1.5 * 2^23 + round(v / scale) = the int8 value
#pragma unroll
                for (int k = 0; k < 8; k++) w[k] = __float_as_int(fmaf(v[k], inv, 12582912.f));
                qf[b][rr] = __byte_perm(__byte_perm(w[0], w[1], 0x0040), __byte_perm(w[2], w[3], 0x0040), 0x5410);
                qf[b][2 + rr] = __byte_perm(__byte_perm(w[4], w[5], 0x0040), __byte_perm(w[6], w[7], 0x0040), 0x5410);
            }
    }

    if (it0 < it1) { // K(it0) -> int8, V(it0) -> fp16 (later tiles: K(it + 1), V(it + 1) during tile it)
        pf::cpa_wait<A8_NS - 2>();
        __syncthreads();
        expand_k(it0, warp, A2_NT / 32);
#pragma unroll
        for (int k = 0; k < 3; k++) a8_cv_v<VB>(asm_ + (it0 % A8_NS) * A8_STG, sV0 + (it0 & 1) * A2_K * A2_LD, k, vbt(it0));
    }
    float o[16][4];
#pragma unroll
    for (int j = 0; j < 16; j++) o[j][0] = o[j][1] = o[j][2] = o[j][3] = 0.f;
    float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.f, l1 = 0.f; // rows g8, g8 + 8 (l: kept by half 1)
    const int qr0 = qpos0 + g8, qr1 = qpos0 + g8 + 8;
    const unsigned smem0 = (unsigned) __cvta_generic_to_shared(asm_);
    const unsigned kofs = (16 * hf + 8 * (lane >> 4) + (lane & 7)) * A8_KLD + ((lane >> 3) & 1) * 16;
    const unsigned va0 = (unsigned) __cvta_generic_to_shared(sV0 + (lane & 15) * A2_LD + 128 * hf + (lane >> 4) * 8);
    float2 * mx_own = xmax + (2 * hq + hf) * 32 + lane;
    const float2 * mx_oth = xmax + (2 * hq + (hf ^ 1)) * 32 + lane;
    uint4 * p_own = xp + (2 * hq + hf) * 32 + lane;
    const uint4 * p_oth = xp + (2 * hq + (hf ^ 1)) * 32 + lane;
    float2 * xs = xsum + hq * 32 + lane;
    // EXA: the fold R of the finished segments: M / L (L: half 1) in registers, R.o in global memory at this thread's own
    // elements (whole item: its y2 rows; head of a split item: its partial slot)
    float M0 = -INFINITY, M1 = -INFINITY, L0 = 0.f, L1 = 0.f;
    bool have_r = false, rv0 = true, rv1 = true;
    float *rb0 = nullptr, *rb1 = nullptr;
    if (EXA) {
        if (ex_role == 0) {
            rb0 = cP.y2 + (size_t) (r0 + t0 + g8) * VD + (g * 6 + hq) * HD + 128 * hf + 2 * t4;
            rb1 = rb0 + (size_t) 8 * VD;
            rv0 = g8 < nq;
            rv1 = g8 + 8 < nq;
        } else {
            rb0 = ex_base + (size_t) (hq * A2_Q + g8) * HD + 128 * hf + 2 * t4; // (slot 0)
            rb1 = rb0 + (size_t) 8 * HD;
        }
    }
    // R <- fold(R, this segment); reset: store R.o and start the next segment, else (the last one) R stays in o / m / l
    auto ex_fold = [&](bool reset) {
        float Mn0, a0, b0, Mn1, a1, b1;
        ex_ab(M0, m0, Mn0, a0, b0);
        ex_ab(M1, m1, Mn1, a1, b1);
#pragma unroll
        for (int j = 0; j < 16; j++) {
            float2 *p0 = reinterpret_cast<float2 *>(rb0 + 8 * j), *p1 = reinterpret_cast<float2 *>(rb1 + 8 * j);
            float2 v0 = make_float2(o[j][0], o[j][1]), v1 = make_float2(o[j][2], o[j][3]);
            if (have_r && rv0) {
                const float2 r = *p0;
                v0 = make_float2(ex_f(a0, r.x, b0, v0.x), ex_f(a0, r.y, b0, v0.y));
            }
            if (have_r && rv1) {
                const float2 r = *p1;
                v1 = make_float2(ex_f(a1, r.x, b1, v1.x), ex_f(a1, r.y, b1, v1.y));
            }
            if (reset) {
                if (rv0) *p0 = v0;
                if (rv1) *p1 = v1;
                o[j][0] = o[j][1] = o[j][2] = o[j][3] = 0.f;
            } else {
                o[j][0] = v0.x, o[j][1] = v0.y, o[j][2] = v1.x, o[j][3] = v1.y;
            }
        }
        if (have_r) {
            L0 = ex_f(a0, L0, b0, l0), L1 = ex_f(a1, L1, b1, l1), M0 = Mn0, M1 = Mn1;
        } else {
            L0 = l0, L1 = l1, M0 = m0, M1 = m1;
        }
        have_r = true;
        if (reset) m0 = m1 = -INFINITY, l0 = l1 = 0.f;
        else m0 = M0, m1 = M1, l0 = L0, l1 = L1;
    };

    // EXA: tiles in segments (it0 is a boundary: 0 or k0 segt), the fold between two segments' tile loops: it touches only
    // this thread's registers and global memory, so it is the same as folding after tile sb's barrier
    for (int sb = it0, se; sb < it1; sb = se) {
        se = EXA ? min(it1, sb + segt) : it1;
        if (EXA && sb > it0) { // a segment boundary: tiles < sb are done
            if (ex_role != 2) {
                ex_fold(true);
            } else { // segment sb / segt - 1 -> its slot, then a fresh online softmax
                float * q = ex_base + (size_t) (1 + sb / segt - 1 - ex_h) * A2_PART;
#pragma unroll
                for (int j = 0; j < 16; j++) {
                    float * e = q + (size_t) (hq * A2_Q + g8) * HD + 128 * hf + 8 * j + 2 * t4;
                    *reinterpret_cast<float2 *>(e) = make_float2(o[j][0], o[j][1]);
                    *reinterpret_cast<float2 *>(e + 8 * HD) = make_float2(o[j][2], o[j][3]);
                    o[j][0] = o[j][1] = o[j][2] = o[j][3] = 0.f;
                }
                if (hf && t4 == 0) {
                    float2 * ml = reinterpret_cast<float2 *>(q + 6 * A2_Q * HD);
                    ml[hq * A2_Q + g8] = make_float2(m0, l0);
                    ml[hq * A2_Q + g8 + 8] = make_float2(m1, l1);
                }
                m0 = m1 = -INFINITY, l0 = l1 = 0.f;
            }
        }
    for (int it = sb; it < se; it++) { // (the tile loop: one pass over it0 .. it1 - 1 unless EXA)
        const int kb = it * A2_K;
        pf::cpa_wait<0>(); // own copies of tiles it and it + 1
        __syncthreads();   // everyone's copies landed; every warp is done with tile it - 1 (P V read sV[(it - 1) & 1])
        __half * sVc = sV0 + ((it + 1) & 1) * A2_K * A2_LD; // V(it + 1) is converted into it during this tile
        const unsigned va = va0 + (it & 1) * A2_K * A2_LD * 2;
        const uint8_t * stv = asm_ + ((it + 1) % A8_NS) * A8_STG;
        const bool cvn = it + 1 < it1;
        const int vbn = vbt(it + 1); // V(it + 1)'s format
        const int ia = it + A8_NS - 1; // refill the stage of tile it - 1 with tile ia
        uint8_t * sta = asm_ + (ia % A8_NS) * A8_STG;
        unsigned ksn = 0u;
        if (ia < it1) {
            const int kba = ia * A2_K, npa = min(A2_K, pmax + 1 - kba);
            ksn = issue_tile(sta, ia, npa);
        } else {
            pf::cpa_commit();
        }
        const uint8_t * st = asm_ + (it % A8_NS) * A8_STG;
        const unsigned ka = smem0 + (it % A8_NS) * A8_STG + kofs;
        const float * ksp = reinterpret_cast<const float *>(st + A8_KB) + 16 * hf + 2 * t4;
        // S for this warp's 16 positions: per 32-dim block two IMMAs (n8 tiles of positions 16 hf + 0..7, 8..15)
        float s[2][4];
#pragma unroll
        for (int ni = 0; ni < 2; ni++) s[ni][0] = s[ni][1] = s[ni][2] = s[ni][3] = 0.f;
#pragma unroll
        for (int b = 0; b < 8; b++) {
            // K(it + 1) -> int8: rows warp, warp + 12, warp + 24 in one call at b = 0 (loads of all rows first; its stage
            // landed before this tile's barrier and is first read by tile it + 1's S MMAs)
            if (b == 0 && cvn) expand_k(it + 1, warp, A2_NT / 32);
            unsigned kf[4];
            pf::ldsm_x4(kf, ka + 32 * b);
            int d0[4], d1[4];
            a8_imma(d0, qf[b], kf[0], kf[1]);
            a8_imma(d1, qf[b], kf[2], kf[3]);
            const float2 k0 = *reinterpret_cast<const float2 *>(ksp + A2_K * b);
            const float2 k1 = *reinterpret_cast<const float2 *>(ksp + A2_K * b + 8);
            if constexpr (QROW) {
                s[0][0] = fmaf(a8_i2f(d0[0]), k0.x, s[0][0]);
                s[0][1] = fmaf(a8_i2f(d0[1]), k0.y, s[0][1]);
                s[0][2] = fmaf(a8_i2f(d0[2]), k0.x, s[0][2]);
                s[0][3] = fmaf(a8_i2f(d0[3]), k0.y, s[0][3]);
                s[1][0] = fmaf(a8_i2f(d1[0]), k1.x, s[1][0]);
                s[1][1] = fmaf(a8_i2f(d1[1]), k1.y, s[1][1]);
                s[1][2] = fmaf(a8_i2f(d1[2]), k1.x, s[1][2]);
                s[1][3] = fmaf(a8_i2f(d1[3]), k1.y, s[1][3]);
            } else {
                const float q0 = qs[b][0], q1 = qs[b][1];
                s[0][0] = fmaf(a8_i2f(d0[0]), q0 * k0.x, s[0][0]);
                s[0][1] = fmaf(a8_i2f(d0[1]), q0 * k0.y, s[0][1]);
                s[0][2] = fmaf(a8_i2f(d0[2]), q1 * k0.x, s[0][2]);
                s[0][3] = fmaf(a8_i2f(d0[3]), q1 * k0.y, s[0][3]);
                s[1][0] = fmaf(a8_i2f(d1[0]), q0 * k1.x, s[1][0]);
                s[1][1] = fmaf(a8_i2f(d1[1]), q0 * k1.y, s[1][1]);
                s[1][2] = fmaf(a8_i2f(d1[2]), q1 * k1.x, s[1][2]);
                s[1][3] = fmaf(a8_i2f(d1[3]), q1 * k1.y, s[1][3]);
            }
            // V(it + 1) -> fp16 (its buffer was last read by P V of tile it - 1) between the MMAs
            if ((b == 1 || b == 3 || b == 5) && cvn) a8_cv_v<VB>(stv, sVc, b >> 1, vbn);
        }
        if constexpr (QROW) {
#pragma unroll
            for (int ni = 0; ni < 2; ni++) {
                s[ni][0] *= qs[0][0];
                s[ni][1] *= qs[0][0];
                s[ni][2] *= qs[0][1];
                s[ni][3] *= qs[0][1];
            }
        }
        // causal mask, row maxima over the pair, P, row sums: as pk_attn2
        float mx0 = -INFINITY, mx1 = -INFINITY;
#pragma unroll
        for (int ni = 0; ni < 2; ni++) {
            const int p = kb + 16 * hf + 8 * ni + 2 * t4;
            if (p > qr0) s[ni][0] = -INFINITY;
            if (p + 1 > qr0) s[ni][1] = -INFINITY;
            if (p > qr1) s[ni][2] = -INFINITY;
            if (p + 1 > qr1) s[ni][3] = -INFINITY;
            mx0 = fmaxf(mx0, fmaxf(s[ni][0], s[ni][1]));
            mx1 = fmaxf(mx1, fmaxf(s[ni][2], s[ni][3]));
        }
        mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 1));
        mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 2));
        mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 1));
        mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 2));
        *mx_own = make_float2(mx0, mx1);
        a2_pair_sync(hq);
        {
            const float2 t = *mx_oth;
            mx0 = fmaxf(mx0, t.x);
            mx1 = fmaxf(mx1, t.y);
        }
        const float mn0 = fmaxf(m0, mx0), mn1 = fmaxf(m1, mx1);
        const float al0 = a8_ex2(m0 - mn0, mn0, 1.f), al1 = a8_ex2(m1 - mn1, mn1, 1.f);
        float x0[2], x1[2];
        unsigned pa[2][2];
#pragma unroll
        for (int ni = 0; ni < 2; ni++) {
            const float p0 = a8_ex2(s[ni][0] - mn0, mn0, 0.f);
            const float p1 = a8_ex2(s[ni][1] - mn0, mn0, 0.f);
            const float p2 = a8_ex2(s[ni][2] - mn1, mn1, 0.f);
            const float p3 = a8_ex2(s[ni][3] - mn1, mn1, 0.f);
            x0[ni] = p0 + p1;
            x1[ni] = p2 + p3;
            __half2 h01 = __floats2half2_rn(p0, p1), h23 = __floats2half2_rn(p2, p3);
            pa[ni][0] = *reinterpret_cast<unsigned *>(&h01);
            pa[ni][1] = *reinterpret_cast<unsigned *>(&h23);
        }
        const uint4 pown = make_uint4(pa[0][0], pa[0][1], pa[1][0], pa[1][1]);
        *p_own = pown;
        if (hf == 0) {
            float ps0 = 0.f, ps1 = 0.f;
            ps0 += x0[0];
            ps0 += x0[1];
            ps1 += x1[0];
            ps1 += x1[1];
            *xs = make_float2(ps0, ps1);
        }
        a2_pair_sync(hq);
        const uint4 poth = *p_oth;
        if (hf == 1) {
            const float2 t = *xs;
            float ps0 = t.x, ps1 = t.y;
            ps0 += x0[0];
            ps0 += x0[1];
            ps1 += x1[0];
            ps1 += x1[1];
            ps0 += __shfl_xor_sync(0xffffffffu, ps0, 1);
            ps0 += __shfl_xor_sync(0xffffffffu, ps0, 2);
            ps1 += __shfl_xor_sync(0xffffffffu, ps1, 1);
            ps1 += __shfl_xor_sync(0xffffffffu, ps1, 2);
            l0 = l0 * al0 + ps0;
            l1 = l1 * al1 + ps1;
        }
        m0 = mn0;
        m1 = mn1;
        const uint4 P0 = hf ? poth : pown, P1 = hf ? pown : poth;
        // O += P V for this warp's 128 dims (fp16-accumulated per tile, as pk_attn2's PV16)
        {
            const unsigned A0[4] = {P0.x, P0.y, P0.z, P0.w}, A1[4] = {P1.x, P1.y, P1.z, P1.w};
#pragma unroll
            for (int j = 0; j < 8; j++) {
                unsigned b[4], h0[2], h1[2];
                a2_ldsm_t(b, va + 32 * j);
                pf::hmma16z(h0, A0, b[0], b[1]);
                pf::hmma16z(h1, A0, b[2], b[3]);
                a2_ldsm_t(b, va + 16 * A2_LD * 2 + 32 * j);
                pf::hmma16(h0, A1, b[0], b[1]);
                pf::hmma16(h1, A1, b[2], b[3]);
                const float2 u0 = __half22float2(pf::u2h(h0[0])), u1 = __half22float2(pf::u2h(h0[1]));
                const float2 w0 = __half22float2(pf::u2h(h1[0])), w1 = __half22float2(pf::u2h(h1[1]));
                o[2 * j][0] = o[2 * j][0] * al0 + u0.x;
                o[2 * j][1] = o[2 * j][1] * al0 + u0.y;
                o[2 * j][2] = o[2 * j][2] * al1 + u1.x;
                o[2 * j][3] = o[2 * j][3] * al1 + u1.y;
                o[2 * j + 1][0] = o[2 * j + 1][0] * al0 + w0.x;
                o[2 * j + 1][1] = o[2 * j + 1][1] * al0 + w0.y;
                o[2 * j + 1][2] = o[2 * j + 1][2] * al1 + w1.x;
                o[2 * j + 1][3] = o[2 * j + 1][3] * al1 + w1.y;
            }
        }
        if (ia < it1) a8_st_ks(sta, ksn); // K scales of tile ia (its stage was freed at this tile's first barrier)
    }
    }
    pf::cpa_wait<0>();
    if (EXA && ex_role != 2) ex_fold(false); // R of all segments of this unit -> o / m / l
    // row sums: half 1 -> half 0 (the maxima slot is no longer read)
    if (hf) *mx_own = make_float2(l0, l1);
    a2_pair_sync(hq);
    if (!hf) {
        const float2 t = *mx_oth;
        l0 = t.x;
        l1 = t.y;
    }
    const int head = g * 6 + hq;
    if (nseg == 1) {
        const float il0 = l0 > 0.f ? 1.f / l0 : 0.f, il1 = l1 > 0.f ? 1.f / l1 : 0.f;
#pragma unroll
        for (int j = 0; j < 16; j++) {
            const int d = 128 * hf + 8 * j + 2 * t4;
            if (g8 < nq) *reinterpret_cast<float2 *>(cP.y2 + (size_t) (r0 + t0 + g8) * VD + head * HD + d) = make_float2(o[j][0] * il0, o[j][1] * il0);
            if (g8 + 8 < nq) *reinterpret_cast<float2 *>(cP.y2 + (size_t) (r0 + t0 + g8 + 8) * VD + head * HD + d) = make_float2(o[j][2] * il1, o[j][3] * il1);
        }
    } else {
        float * pp = !EXA ? part + (size_t) (blockIdx.x - nfull) * A2_PART
                          : ex_base + (size_t) (ex_role == 1 ? 0 : 1 + (it1 - 1) / segt - ex_h) * A2_PART;
        const int r0 = hq * A2_Q + g8;
#pragma unroll
        for (int j = 0; j < 16; j++) {
            const int d = 128 * hf + 8 * j + 2 * t4;
            *reinterpret_cast<float2 *>(pp + (size_t) r0 * HD + d) = make_float2(o[j][0], o[j][1]);
            *reinterpret_cast<float2 *>(pp + (size_t) (r0 + 8) * HD + d) = make_float2(o[j][2], o[j][3]);
        }
        if (hf && t4 == 0) {
            float2 * ml = reinterpret_cast<float2 *>(pp + 6 * A2_Q * HD);
            ml[r0] = make_float2(EXA ? m0 : m0 * (1.f / A8_L2E), l0); // natural-log units for pk_attn2_comb (EXA: log2)
            ml[r0 + 8] = make_float2(EXA ? m1 : m1 * (1.f / A8_L2E), l1);
        }
    }
}


// EXA: the split items' results: the head unit's R (slot 0, segments 0 .. sh - 1) folded with the single segments
// (slot q = segment sh + q - 1) in segment order, with pk_attn8's own ex_ab / ex_f arithmetic, then normalised -> y2 (rows
// r0 + ..). Block = split item; S slots per item (in part for the first exa items, then in part2); exh >= 1 = the head
// unit's segments (an item may have fewer).
__global__ void __launch_bounds__(256) pk_attn8_combx(int T, int n0, int nfull, int S, int segt, int exh, int r0, const float * __restrict__ part,
                                                      const float * __restrict__ part2, int exa) { MK_VIEW_INIT();
    const int item = nfull + blockIdx.x, nqt = (T + A2_Q - 1) / A2_Q, qt = item % nqt, g = item / nqt;
    const int t0 = qt * A2_Q, nq = min(A2_Q, T - t0);
    const int ntile = (n0 + t0 + nq - 1) / A2_K + 1, nsg = (ntile + segt - 1) / segt, sh = min(exh, nsg);
    const int j = blockIdx.x;
    const float * P = j < exa ? part + (size_t) j * S * A2_PART : part2 + (size_t) (j - exa) * S * A2_PART;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    for (int r = warp; r < 6 * A2_Q; r += 8) {
        const int h = r / A2_Q, tt = r % A2_Q;
        if (tt >= nq) continue;
        const float * P0 = P;
        float M = P0[6 * A2_Q * HD + 2 * r], L = P0[6 * A2_Q * HD + 2 * r + 1], acc[8];
#pragma unroll
        for (int k = 0; k < 8; k++) acc[k] = P0[(size_t) r * HD + 8 * lane + k];
        for (int q = 1; q <= nsg - sh; q++) {
            const float * ps = P + (size_t) q * A2_PART;
            float Mn, a, b;
            ex_ab(M, ps[6 * A2_Q * HD + 2 * r], Mn, a, b);
            L = ex_f(a, L, b, ps[6 * A2_Q * HD + 2 * r + 1]);
#pragma unroll
            for (int k = 0; k < 8; k++) acc[k] = ex_f(a, acc[k], b, ps[(size_t) r * HD + 8 * lane + k]);
            M = Mn;
        }
        const float il = L > 0.f ? 1.f / L : 0.f;
        float * dst = cP.y2 + (size_t) (r0 + t0 + tt) * VD + (g * 6 + h) * HD + 8 * lane;
#pragma unroll
        for (int k = 0; k < 8; k++) dst[k] = acc[k] * il;
    }
}

// ------------------------------------------------------------------------------------------
// pk_attn9: pk_attn8 (same units, same stages, every output element the same floating-point chain: bit-identical) with the
// per-tile data movement rewritten for instruction count, the kernel being compute-limited. Each thread's copies are a fixed
// plan set up once (running source pointers, stage offsets; warp-uniform kinds; idle chunks predicated, not branched
// around), for the main-format tiles with all rows; the rest (ring tiles, the last partial tile) keep pk_attn8's a8_issue.
// Order: tile t's copies are issued in tile t - 2, every thread waits for its own before barrier(t - 1) (wait_group 0, then __syncthreads), the
// conversions run in tile t - 1 only and S / P V read the results after barrier(t). What a tile writes it does not read. Warps are in no fixed
// order inside a tile (one can reach its tail before another starts P V), so a conversion store goes only to this thread's own slots.
__device__ __forceinline__ void cp_async16_to(unsigned d, const void * gsrc) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(d), "l"(gsrc) : "memory");
}
__device__ __forceinline__ void a9_cp16p(unsigned p, unsigned d, const void * gsrc) { // p != 0: copy (no branch: no divergence)
    asm volatile("{\n.reg .pred p;\nsetp.ne.u32 p, %0, 0;\n@p cp.async.cg.shared.global [%1], [%2], 16;\n}" ::"r"(p), "r"(d), "l"(gsrc) : "memory");
}
// raw K scales of the tile (row p's in its padding) -> the stage's fp32 table [8 blocks][32 positions] (lane = position: warp 9)
__device__ __forceinline__ void a9_ks(uint8_t * st) {
    const int p = threadIdx.x & 31;
    const uint4 r = *reinterpret_cast<const uint4 *>(st + p * A8_KLD + HD);
    const float2 f0 = __half22float2(pf::u2h(r.x)), f1 = __half22float2(pf::u2h(r.y));
    const float2 f2 = __half22float2(pf::u2h(r.z)), f3 = __half22float2(pf::u2h(r.w));
    float * ks = reinterpret_cast<float *>(st + A8_KB) + p;
    ks[0] = f0.x, ks[A2_K] = f0.y, ks[2 * A2_K] = f1.x, ks[3 * A2_K] = f1.y;
    ks[4 * A2_K] = f2.x, ks[5 * A2_K] = f2.y, ks[6 * A2_K] = f3.x, ks[7 * A2_K] = f3.y;
}
// chunk k (of 3; warps 8 - 11 have two) of this thread: 8 V values of ring stage st -> fp16 in sV: a8_cv_v without its per-thread
// test; compile-time VB: kv_spread4 for kv_hi4 (the same bits)
template <int VB = 0>
__device__ __forceinline__ void a9_cv(const uint8_t * st, __half * sV, int k, int vbits) {
    const int w = threadIdx.x >> 5, p = w + 12 * k, c = threadIdx.x & 31;
    const uint8_t * row = st + (A8_KB + A8_KSB + 12 * k * HD) + w * HD;
    const __half d = reinterpret_cast<const __half *>(st + (A8_KB + A8_KSB + A8_VB + 12 * k * (HD / 32) * 2))[w * (HD / 32) + (c >> 2)];
    __half2 h[4];
    if (vbits == 8) {
        a2_q8h(*reinterpret_cast<const uint2 *>(row + 8 * c), d, h);
    } else if constexpr (VB > 4 && VB < 8) {
        constexpr int HB = VB - 4;
        const unsigned w = *reinterpret_cast<const unsigned *>(row + 4 * c);
        unsigned hi = row[128 + c * HB];
        if constexpr (HB > 1) hi |= (unsigned) row[129 + c * HB] << 8;
        if constexpr (HB > 2) hi |= (unsigned) row[130 + c * HB] << 16;
        const unsigned lo = (w & 0x0F0F0F0Fu) | kv_spread4<HB>(hi), hv = ((w >> 4) & 0x0F0F0F0Fu) | kv_spread4<HB>(hi >> (4 * HB));
        const __half2 off = __half2half2(__ushort_as_half((unsigned short) (0x6400 + (1 << (VB - 1))))), dd = __half2half2(d);
        h[0] = __hmul2(__hsub2(pf::u2h(__byte_perm(lo, 0x64646464u, 0x4140)), off), dd);
        h[1] = __hmul2(__hsub2(pf::u2h(__byte_perm(lo, 0x64646464u, 0x4342)), off), dd);
        h[2] = __hmul2(__hsub2(pf::u2h(__byte_perm(hv, 0x64646464u, 0x4140)), off), dd);
        h[3] = __hmul2(__hsub2(pf::u2h(__byte_perm(hv, 0x64646464u, 0x4342)), off), dd);
    } else if (vbits == 4) {
        a2_q4h(*reinterpret_cast<const unsigned *>(row + 4 * c), d, h);
    } else {
        a2_qnh(*reinterpret_cast<const unsigned *>(row + 4 * c), kv_hi(row + 128 + c * (vbits - 4), vbits - 4), d, vbits, h);
    }
    *reinterpret_cast<int4 *>(sV + p * A2_LD + 8 * c) = *reinterpret_cast<int4 *>(h);
}
// on != 0: stores 8 B (no branch)
__device__ __forceinline__ void a9_st64p(unsigned on, unsigned addr, unsigned a, unsigned b) {
    asm volatile("{\n.reg .pred p;\nsetp.ne.u32 p, %0, 0;\n@p st.shared.v2.b32 [%1], {%2, %3};\n}" ::"r"(on), "r"(addr), "r"(a), "r"(b) : "memory");
}
// NR rows r0, r0 + 12, .. of a staged low-bit K tile -> int8 in place (kv_rows_expand<BITS> with every row present); on = 0: no store
template <int BITS, int NR>
__device__ __forceinline__ void a9_k_rows(uint8_t * st, int r0, unsigned on) {
    constexpr int HB = BITS - 4;
    const int lane = threadIdx.x & 31;
    unsigned wd[NR], hi[NR];
#pragma unroll
    for (int k = 0; k < NR; k++) {
        const uint8_t * row = st + (r0 + 12 * k) * A8_KLD;
        wd[k] = *reinterpret_cast<const unsigned *>(row + 4 * lane);
        hi[k] = 0u;
        if constexpr (HB == 1) hi[k] = row[128 + lane];
        if constexpr (HB == 2) hi[k] = *reinterpret_cast<const unsigned short *>(row + 128 + 2 * lane);
        if constexpr (HB == 3) hi[k] = (unsigned) row[128 + 3 * lane] | ((unsigned) row[129 + 3 * lane] << 8) | ((unsigned) row[130 + 3 * lane] << 16);
    }
    __syncwarp();
#pragma unroll
    for (int k = 0; k < NR; k++) {
        unsigned lo = wd[k] & 0x0f0f0f0fu, hv = (wd[k] >> 4) & 0x0f0f0f0fu;
        if constexpr (HB > 0) {
            lo |= kv_spread4<HB>(hi[k]);
            hv |= kv_spread4<HB>(hi[k] >> (4 * HB));
        }
        constexpr unsigned bias = (1u << (BITS - 1)) * 0x01010101u;
        lo = ((lo | 0x80808080u) - bias) ^ 0x80808080u;
        hv = ((hv | 0x80808080u) - bias) ^ 0x80808080u;
        a9_st64p(on, (unsigned) __cvta_generic_to_shared(st + (r0 + 12 * k) * A8_KLD + 8 * lane), lo, hv);
    }
}

// pk_attn9's exact critical-path variants (wave 6, notes r1108prof/wave6/pfattn6.md section 5, code/pf6.md): compile-time; the
// default A9_RS 1, A9_CP 1, A9_EX 2 (A9F 67: prefill attention -4.5% at 101k q8_0 / D, real prompt); -DA9_RS=0 -DA9_CP=0 -DA9_EX=0
// (A9F 0) = 1ecb507's kernel, SASS for SASS. The engine instantiates A9F = A9_FLAGS; bench/attn_bench also builds A9F = 0 (and A9_TV's
// values) to time them in one process. Every variant keeps each output's floating-point operations, operands and order (exactness
// notes at each site).
// A9_RS 1 (A9F 1): half 1's row-sum chain (LDS, adds, two shuffle rounds, l update) in every warp, without a branch, inside the
// P V block; the result is kept by half 1 only. A9_CP bit 1 (A9F 2): the fast-path copies spread over the S loop instead of all
// at the barrier; bit 2 (A9F 4): a copy's source = its slot's base + tile offset x stride (one IMAD.WIDE into a fresh pair), no
// running pointer. A9_MX 1 (A9F 16): row maxima without the shuffle rounds (each lane reads its row's 8 lane maxima of the pair
// after the pair sync; half 0's partial row sums move to the exchange's free tail). A9_EX 1 (A9F 32): the plain ex2.approx.ftz
// for the tile's exponentials when no argument of the warp is below -126 (exp2f's subnormal path), else exp2f again; A9_EX 2
// (A9F 64): the same choice made before the exponentials, from each lane's row minima (no exponential computed twice).
#ifndef A9_RS
#define A9_RS 1
#endif
#ifndef A9_CP
#define A9_CP 1
#endif
#ifndef A9_MX
#define A9_MX 0
#endif
#ifndef A9_EX
#define A9_EX 2
#endif
constexpr int A9_FLAGS = (A9_RS ? 1 : 0) | ((A9_CP & 3) << 1) | (A9_MX ? 16 : 0) | (A9_EX == 1 ? 32 : A9_EX == 2 ? 64 : 0);
// = exp2f(x) for x >= -126 and NaN (exp2f is this MUFU.EX2 there; it scales only below -126)
__device__ __forceinline__ float a9_ex2(float x) {
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

template <bool QROW, bool KLOW, bool EXA = false, int KB = 0, int VB = 0, int A9F = A9_FLAGS>
__global__ void __launch_bounds__(A2_NT, 1) pk_attn9(int T, int n0, int slot, int nfull, int S, float * __restrict__ part, int r0 = 0,
                                                     int segt = 0, int exh = 0, float * __restrict__ part2 = nullptr, int exa = 0) { MK_VIEW_INIT();
    static_assert(KB == 0 || (KLOW && KB >= 4 && KB <= 7), "pk_attn9 KB");
    static_assert(VB == 0 || VB == 8 || (VB >= 4 && VB <= 7), "pk_attn9 VB");
    constexpr bool A9RS = A9F & 1, A9CS = A9F & 2, A9CW = A9F & 4, A9MX = A9F & 16, A9EX = A9F & 32, A9EE = (A9F & 64) && !A9EX;
    extern __shared__ __align__(16) uint8_t asm_[];
    __half * sV0 = reinterpret_cast<__half *>(asm_ + A8_NS * A8_STG); // [2][32][A2_LD] fp16 V(it) / V(it + 1)
    float2 * xmax = reinterpret_cast<float2 *>(sV0 + 2 * A2_K * A2_LD); // [6][2][32] row maxima (finally: l)
    uint4 * xp = reinterpret_cast<uint4 *>(xmax + 6 * 2 * 32);       // [6][2][32] P fragments
    float2 * xps = reinterpret_cast<float2 *>(xp + 6 * 2 * 32);     // [6][32] A9_MX: half 0's partial row sums (A2_XCH's tail)
    (void) xps;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g8 = lane >> 2, t4 = lane & 3;
    const int hq = warp >> 1, hf = warp & 1;
    const int nqt = (T + A2_Q - 1) / A2_Q;
    int item = blockIdx.x, seg = 0, nseg = 1;
    if (item >= nfull) {
        const int v = item - nfull;
        if (EXA) { // units interleaved over the split items: every head, then every item's first single segment, ..
            const int nsplit = nqt * NKV - nfull;
            item = nfull + v % nsplit;
            seg = v / nsplit;
        } else {
            item = nfull + v / S;
            seg = v % S;
        }
        nseg = S;
    }
    const int qt = item % nqt, g = item / nqt;
    const int t0 = qt * A2_Q, nq = min(A2_Q, T - t0);
    const int qpos0 = n0 + t0, pmax = qpos0 + nq - 1;
    const int ntile = pmax / A2_K + 1;
    int it0 = ntile * seg / nseg, it1 = ntile * (seg + 1) / nseg;
    // EXA: 0 whole item (final output); split item (S units, S partial slots): unit 0 = the head, segments 0 .. exh - 1
    // (folds, R -> slot 0), role 1; unit u >= 1 = segment exh + u - 1 alone (-> slot u), role 2. (Role 2 code also
    // handles a unit of several segments, each to slot 1 + k - ex_h.) Split item j's S slots: in part for j < exa, else
    // in part2 (two idle scratch buffers, see MKEngine's ex_part).
    int ex_role = 0, ex_h = 0;
    float * ex_base = nullptr;
    if (EXA) {
        const int nsg = (ntile + segt - 1) / segt;
        it0 = 0;
        it1 = ntile;
        if (nseg > 1) {
            const int k0 = seg == 0 ? 0 : exh + seg - 1, k1 = seg == 0 ? exh : k0 + 1;
            if (k0 >= nsg) return; // (idle unit: this item has fewer segments than the plan's longest)
            it0 = k0 * segt;
            it1 = min(ntile, k1 * segt);
            ex_h = exh;
            ex_role = seg == 0 ? 1 : 2;
            const int j = item - nfull;
            ex_base = j < exa ? part + (size_t) j * S * A2_PART : part2 + (size_t) (j - exa) * S * A2_PART;
        }
    }
    // this slot's q8_0 K / V through the MK_KV slot pointers (as pk_attn2): slots are packed in kc / vc, so with a q4
    // slot before this one the uniform [slot][head][ctx][HD] offset lands in another slot's rows; for an all-q8
    // cache kvp / kvs give exactly the [slot][head][ctx][HD] addresses
    // Per tile (kv_src): the slot's main buffer (K q8 or q4..q7, expanded to int8 in its stage by kv_tile_expand during the
    // previous tile; V q8 or q4..q7, converted to fp16 by a8_cv_v) or, with MK_KV_WIN, the q8 ring for tiles at positions
    // >= wsr. kbt(it) / vbt(it) = the K / V format of tile it.
    const int wsr = kv_ws(cP.wsrow ? pmax + 1 : n0 + T), wsm = kv_wsm(wsr); // (MK_EXACT: per query tile, see PBufs)
    const KvFmt kf = kv_fmt(slot); // per-slot formats once; per tile one compare of the tile index (kv_ring_tile)
    const int itr_k = kv_ring_tile(kf.kr, wsr, 0, A2_K), itr_v = kv_ring_tile(kf.vr, wsr, 0, A2_K);
    const int kbs = KB ? KB : kf.kb, vbs = VB ? VB : kf.vb; // the main buffer's K / V bits
    auto kbt = [&](int it) { return it >= itr_k ? 8 : kbs; };
    auto vbt = [&](int it) { return it >= itr_v ? 8 : vbs; };
    auto issue_tile = [&](uint8_t * st, int it, int np) { // K, V, V scales of tile it -> stage st; returns this thread's K scales (raw)
        const int8_t *tk, *tv;
        const __half *tks, *tvs;
        kv_tile_src(0, slot, g, it * A2_K, wsr, wsm, kbs, it >= itr_k, tk, tks);
        kv_tile_src(1, slot, g, it * A2_K, wsr, wsm, vbs, it >= itr_v, tv, tvs);
        a8_issue(st, tk, tv, tvs, 0, np, KLOW ? kbt(it) : 8, vbt(it));
        return a8_ld_ks(tks, 0, np);
    };
    auto expand_k = [&](int it, int r0, int rs) { // low-bit K rows r0, r0 + rs, .. of tile it (landed) -> int8 in place
        if constexpr (KLOW) {
            // all A2_K rows: rows >= np were zeroed by a8_issue_k (finite values, masked by the causal test)
            const int kb_ = kbt(it);
            if constexpr (KB != 0) {
                if (kb_ != 8)
                    for (int r = r0; r < A2_K; r += 3 * rs) kv_rows_expand<KB, true>(asm_ + (it % A8_NS) * A8_STG, A2_K, r, rs, A8_KLD);
            } else if (kb_ != 8) kv_tile_expand<true>(asm_ + (it % A8_NS) * A8_STG, kb_, A2_K, r0, rs, A8_KLD);
        }
    };
    (void) kbt;

    // Tiles it < itf (main format, all rows): this thread's fixed share of the copies. Chunk j = tid + A2_NT k of the tile:
    // K rows, V rows, V scales, K scales (raw, in the K rows' padding), then none: warp-uniform kinds, no branch.
    const unsigned smem0 = (unsigned) __cvta_generic_to_shared(asm_);
    const int itm = min(itr_k, itr_v), itf = min(itm, (pmax + 1) / A2_K);
    constexpr bool A9_HK = VB != 0 && (KB != 0 || !KLOW); // compile-time main formats
    constexpr int A9_NSL = (KB && VB) ? 2 : 3;
    const char * cps[A9_NSL];
    int cpi[A9_NSL];
    unsigned cpd[A9_NSL];
    const int kbm = KLOW ? kbs : 8, nkc = 2 * kbm, nvc = 2 * vbs;
    const int cpn = A2_K * (nkc + nvc + 2); // chunks of a tile
    {
        const int8_t *tk, *tv;
        const __half *tks, *tvs;
        kv_tile_src(0, slot, g, it0 * A2_K, wsr, wsm, kbs, false, tk, tks);
        kv_tile_src(1, slot, g, it0 * A2_K, wsr, wsm, vbs, false, tv, tvs);
#pragma unroll
        for (int k = 0; k < A9_NSL; k++) {
            int j = tid + A2_NT * k, p;
            const int nk = A2_K * nkc, nv = A2_K * nvc;
            if (j < nk) {
                p = j / nkc;
                j -= p * nkc;
                cps[k] = reinterpret_cast<const char *>(tk) + p * (HD * kbm / 8) + 16 * j;
                cpi[k] = A2_K * (HD * kbm / 8);
                cpd[k] = p * A8_KLD + 16 * j;
            } else if (j < nk + nv) {
                j -= nk;
                p = j / nvc;
                j -= p * nvc;
                cps[k] = reinterpret_cast<const char *>(tv) + p * (HD * vbs / 8) + 16 * j;
                cpi[k] = A2_K * (HD * vbs / 8);
                cpd[k] = A8_KB + A8_KSB + p * HD + 16 * j;
            } else if (j < nk + nv + A2_K) {
                p = j - nk - nv;
                cps[k] = reinterpret_cast<const char *>(tvs + p * (HD / 32));
                cpi[k] = A2_K * (HD / 32) * 2;
                cpd[k] = A8_KB + A8_KSB + A8_VB + 16 * p;
            } else { // K scales; none: never issued
                p = (j - nk - nv - A2_K) & 31;
                cps[k] = reinterpret_cast<const char *>(tks + p * (HD / 32));
                cpi[k] = A2_K * (HD / 32) * 2;
                cpd[k] = p * A8_KLD + HD;
            }
        }
    }
    // A9_CP bit 2: slot k's source for tile it0 + n is cps[k] + n cpi[k] (cps stays the tile-it0 base): the address the running
    // pointer has wherever a copy is issued (a fast tile's predecessors from it0 on are all fast tiles). (No lambda for it: a dead
    // closure capturing these by reference changes the A9F = 0 kernel's code.)
    auto issue_fast = [&](unsigned sto, int n) { // (n = tile - it0)
        if constexpr (A9CW) {
            cp_async16_to(smem0 + sto + cpd[0], cps[0] + (size_t) (unsigned) n * (unsigned) cpi[0]);
#pragma unroll
            for (int k = 1; k < A9_NSL; k++)
                a9_cp16p(tid + A2_NT * k < cpn, smem0 + sto + cpd[k], cps[k] + (size_t) (unsigned) n * (unsigned) cpi[k]);
        } else {
            cp_async16_to(smem0 + sto + cpd[0], cps[0]);
            cps[0] += cpi[0];
#pragma unroll
            for (int k = 1; k < A9_NSL; k++) {
                a9_cp16p(tid + A2_NT * k < cpn, smem0 + sto + cpd[k], cps[k]);
                cps[k] += cpi[k];
            }
        }
        pf::cpa_commit();
    };

    // ring prologue: tiles it0 .. it0 + A8_NS - 2
#pragma unroll
    for (int j = 0; j < A8_NS - 1; j++) {
        const int it = it0 + j;
        uint8_t * st = asm_ + (it % A8_NS) * A8_STG;
        if (it < it1) {
            if (it < itf) {
                issue_fast((it % A8_NS) * A8_STG, j);
            } else {
                const int kb = it * A2_K, np = min(A2_K, pmax + 1 - kb);
                a8_st_ks(st, issue_tile(st, it, np));
            }
        } else {
            pf::cpa_commit();
        }
    }

    // Q of head g * 6 + hq, rows g8 and g8 + 8 -> int8 A fragments qf[block] (m16n8k32: a0 / a1 = dims 4 t4 ..
    // of rows g8 / g8 + 8, a2 / a3 = dims 16 + 4 t4 ..), scales qs (QROW: per row)
    unsigned qf[8][4];
    float qs[QROW ? 1 : 8][2];
    {
        const __half * qg = cP.qh + ((size_t) (r0 + t0) * NH + g * 6 + hq) * HD + 4 * t4;
        auto ldq = [&](int b, int rr, float (&v)[8]) {
            const int t = g8 + 8 * rr;
            uint2 lo = make_uint2(0, 0), hi = make_uint2(0, 0);
            if (t < nq) {
                const __half * src = qg + (size_t) t * NH * HD + 32 * b;
                lo = __ldg(reinterpret_cast<const uint2 *>(src));
                hi = __ldg(reinterpret_cast<const uint2 *>(src + 16));
            }
            float2 f = __half22float2(pf::u2h(lo.x));
            v[0] = f.x; v[1] = f.y;
            f = __half22float2(pf::u2h(lo.y));
            v[2] = f.x; v[3] = f.y;
            f = __half22float2(pf::u2h(hi.x));
            v[4] = f.x; v[5] = f.y;
            f = __half22float2(pf::u2h(hi.y));
            v[6] = f.x; v[7] = f.y;
        };
        auto qmax = [&](const float (&v)[8], float a) {
#pragma unroll
            for (int k = 0; k < 8; k++) a = fmaxf(a, fabsf(v[k]));
            a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, 1));
            return fmaxf(a, __shfl_xor_sync(0xffffffffu, a, 2));
        };
        float am[2] = {0.f, 0.f};
        if constexpr (QROW) {
#pragma unroll
            for (int b = 0; b < 8; b++)
#pragma unroll
                for (int rr = 0; rr < 2; rr++) {
                    float v[8];
                    ldq(b, rr, v);
                    am[rr] = qmax(v, am[rr]);
                }
            qs[0][0] = am[0] * (A8_L2E / 127.f);
            qs[0][1] = am[1] * (A8_L2E / 127.f);
        }
#pragma unroll
        for (int b = 0; b < 8; b++)
#pragma unroll
            for (int rr = 0; rr < 2; rr++) {
                float v[8];
                ldq(b, rr, v);
                float a = am[rr];
                if constexpr (!QROW) {
                    a = qmax(v, 0.f);
                    qs[b][rr] = a * (A8_L2E / 127.f);
                }
                const float inv = a > 0.f ? 127.f / a : 0.f;
                int w[8]; // low byte of 1.5 * 2^23 + round(v / scale) = the int8 value
#pragma unroll
                for (int k = 0; k < 8; k++) w[k] = __float_as_int(fmaf(v[k], inv, 12582912.f));
                qf[b][rr] = __byte_perm(__byte_perm(w[0], w[1], 0x0040), __byte_perm(w[2], w[3], 0x0040), 0x5410);
                qf[b][2 + rr] = __byte_perm(__byte_perm(w[4], w[5], 0x0040), __byte_perm(w[6], w[7], 0x0040), 0x5410);
            }
    }

    if (it0 < it1) { // K(it0) -> int8, V(it0) -> fp16 (later tiles: K(it + 1), V(it + 1) during tile it)
        pf::cpa_wait<A8_NS - 2>();
        __syncthreads();
        expand_k(it0, warp, A2_NT / 32);
        a9_cv<VB>(asm_ + (it0 % A8_NS) * A8_STG, sV0 + (it0 & 1) * A2_K * A2_LD, 0, vbt(it0));
        a9_cv<VB>(asm_ + (it0 % A8_NS) * A8_STG, sV0 + (it0 & 1) * A2_K * A2_LD, 1, vbt(it0));
        if (warp < 8) a9_cv<VB>(asm_ + (it0 % A8_NS) * A8_STG, sV0 + (it0 & 1) * A2_K * A2_LD, 2, vbt(it0));
        else if (warp == 9 && it0 < itf) a9_ks(asm_ + (it0 % A8_NS) * A8_STG);
    }
    float o[16][4];
#pragma unroll
    for (int j = 0; j < 16; j++) o[j][0] = o[j][1] = o[j][2] = o[j][3] = 0.f;
    float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.f, l1 = 0.f; // rows g8, g8 + 8 (l: kept by half 1)
    int mnf = 0; // every lane of the warp has finite row maxima (m only grows): reset with the online softmax of an EXA segment
    const int qr0 = qpos0 + g8, qr1 = qpos0 + g8 + 8;
    const unsigned kofs = (16 * hf + 8 * (lane >> 4) + (lane & 7)) * A8_KLD + ((lane >> 3) & 1) * 16;
    const unsigned va0 = (unsigned) __cvta_generic_to_shared(sV0 + (lane & 15) * A2_LD + 128 * hf + (lane >> 4) * 8);
    // (slots by tid: warp = 2 hq + hf. A warp's partial row sums go to its partner's maxima slot, which it has already read and
    // the partner does not read)
    float2 * mx_own = xmax + tid;
    const float2 * mx_oth = xmax + (tid ^ 32);
    // EXA: the fold R of the finished segments: M / L (L: half 1) in registers, R.o in global memory at this thread's own
    // elements (whole item: its y2 rows; head of a split item: its partial slot)
    float M0 = -INFINITY, M1 = -INFINITY, L0 = 0.f, L1 = 0.f;
    bool have_r = false, rv0 = true, rv1 = true;
    float *rb0 = nullptr, *rb1 = nullptr;
    if (EXA) {
        if (ex_role == 0) {
            rb0 = cP.y2 + (size_t) (r0 + t0 + g8) * VD + (g * 6 + hq) * HD + 128 * hf + 2 * t4;
            rb1 = rb0 + (size_t) 8 * VD;
            rv0 = g8 < nq;
            rv1 = g8 + 8 < nq;
        } else {
            rb0 = ex_base + (size_t) (hq * A2_Q + g8) * HD + 128 * hf + 2 * t4; // (slot 0)
            rb1 = rb0 + (size_t) 8 * HD;
        }
    }
    // R <- fold(R, this segment); reset: store R.o and start the next segment, else (the last one) R stays in o / m / l
    auto ex_fold = [&](bool reset) {
        float Mn0, a0, b0, Mn1, a1, b1;
        ex_ab(M0, m0, Mn0, a0, b0);
        ex_ab(M1, m1, Mn1, a1, b1);
#pragma unroll
        for (int j = 0; j < 16; j++) {
            float2 *p0 = reinterpret_cast<float2 *>(rb0 + 8 * j), *p1 = reinterpret_cast<float2 *>(rb1 + 8 * j);
            float2 v0 = make_float2(o[j][0], o[j][1]), v1 = make_float2(o[j][2], o[j][3]);
            if (have_r && rv0) {
                const float2 r = *p0;
                v0 = make_float2(ex_f(a0, r.x, b0, v0.x), ex_f(a0, r.y, b0, v0.y));
            }
            if (have_r && rv1) {
                const float2 r = *p1;
                v1 = make_float2(ex_f(a1, r.x, b1, v1.x), ex_f(a1, r.y, b1, v1.y));
            }
            if (reset) {
                if (rv0) *p0 = v0;
                if (rv1) *p1 = v1;
                o[j][0] = o[j][1] = o[j][2] = o[j][3] = 0.f;
            } else {
                o[j][0] = v0.x, o[j][1] = v0.y, o[j][2] = v1.x, o[j][3] = v1.y;
            }
        }
        if (have_r) {
            L0 = ex_f(a0, L0, b0, l0), L1 = ex_f(a1, L1, b1, l1), M0 = Mn0, M1 = Mn1;
        } else {
            L0 = l0, L1 = l1, M0 = m0, M1 = m1;
        }
        have_r = true;
        if (reset) m0 = m1 = -INFINITY, l0 = l1 = 0.f, mnf = 0;
        else m0 = M0, m1 = M1, l0 = L0, l1 = L1;
    };

    unsigned st_a = (it0 % A8_NS) * A8_STG, st_b = ((it0 + 1) % A8_NS) * A8_STG, st_c = ((it0 + 2) % A8_NS) * A8_STG; // tiles it, it + 1, it + 2
    // EXA: tiles in segments (it0 is a boundary: 0 or k0 segt), the fold between two segments' tile loops: it touches only
    // this thread's registers and global memory, so it is the same as folding after tile sb's barrier
    for (int sb = it0, se; sb < it1; sb = se) {
        se = EXA ? min(it1, sb + segt) : it1;
        if (EXA && sb > it0) { // a segment boundary: tiles < sb are done
            if (ex_role != 2) {
                ex_fold(true);
            } else { // segment sb / segt - 1 -> its slot, then a fresh online softmax
                float * q = ex_base + (size_t) (1 + sb / segt - 1 - ex_h) * A2_PART;
#pragma unroll
                for (int j = 0; j < 16; j++) {
                    float * e = q + (size_t) (hq * A2_Q + g8) * HD + 128 * hf + 8 * j + 2 * t4;
                    *reinterpret_cast<float2 *>(e) = make_float2(o[j][0], o[j][1]);
                    *reinterpret_cast<float2 *>(e + 8 * HD) = make_float2(o[j][2], o[j][3]);
                    o[j][0] = o[j][1] = o[j][2] = o[j][3] = 0.f;
                }
                if (hf && t4 == 0) {
                    float2 * ml = reinterpret_cast<float2 *>(q + 6 * A2_Q * HD);
                    ml[hq * A2_Q + g8] = make_float2(m0, l0);
                    ml[hq * A2_Q + g8 + 8] = make_float2(m1, l1);
                }
                m0 = m1 = -INFINITY, l0 = l1 = 0.f, mnf = 0;
            }
        }
    for (int it = sb; it < se; it++) { // (the tile loop: one pass over it0 .. it1 - 1 unless EXA)
        const int kb = it * A2_K;
        pf::cpa_wait<0>(); // own copies of tiles it and it + 1
        __syncthreads();   // everyone's copies landed; every warp is done with tile it - 1 (P V read sV[(it - 1) & 1])
        __half * sVc = sV0 + ((it + 1) & 1) * A2_K * A2_LD; // V(it + 1) is converted into it during this tile
        const unsigned va = va0 + (it & 1) * A2_K * A2_LD * 2;
        uint8_t * stv = asm_ + st_b;
        const int vbn = vbt(it + 1); // V(it + 1)'s format
        const int ia = it + A8_NS - 1; // refill the stage of tile it - 1 with tile ia
        uint8_t * sta = asm_ + st_c;
        const unsigned cpok = ia < itf && ia < it1;
        if constexpr (A9CW && !A9CS) { // A9_CP bit 2 alone: the copies here, sources from the base
#pragma unroll
            for (int k = 0; k < A9_NSL; k++)
                a9_cp16p(k ? cpok & (tid + A2_NT * k < cpn) : cpok, smem0 + st_c + cpd[k], cps[k] + (size_t) (unsigned) (ia - it0) * (unsigned) cpi[k]);
            pf::cpa_commit();
        } else if constexpr (!A9CS) {
            a9_cp16p(cpok, smem0 + st_c + cpd[0], cps[0]);
            cps[0] += cpi[0];
#pragma unroll
            for (int k = 1; k < A9_NSL; k++) {
                a9_cp16p(cpok & (tid + A2_NT * k < cpn), smem0 + st_c + cpd[k], cps[k]);
                cps[k] += cpi[k];
            }
            pf::cpa_commit();
        }
        const uint8_t * st = asm_ + st_a;
        const unsigned ka = smem0 + st_a + kofs;
        // K(it + 1), V(it + 1) in the main format (their stage landed before this tile's barrier; a tile past it1 converts a stale stage
        // nobody reads) run during P V(it): the common part inside the P V loop's basic block, where ptxas overlaps it with the MMAs
        // (no format test: a branch inside it costs ~9%), the warp-specific part just before it. On a tile of another format the K
        // stores are predicated off and the V stores (each thread's own slots) are overwritten by the same thread in the tail, with
        // the real conversion.
        const unsigned kon = it + 1 < itm;
        const float * ksp = reinterpret_cast<const float *>(st + A8_KB) + 16 * hf + 2 * t4;
        // S for this warp's 16 positions: per 32-dim block two IMMAs (n8 tiles of positions 16 hf + 0..7, 8..15)
        float s[2][4];
#pragma unroll
        for (int ni = 0; ni < 2; ni++) s[ni][0] = s[ni][1] = s[ni][2] = s[ni][3] = 0.f;
#pragma unroll
        for (int b = 0; b < 8; b++) {
            unsigned kf[4];
            pf::ldsm_x4(kf, ka + 32 * b);
            int d0[4], d1[4];
            a8_imma(d0, qf[b], kf[0], kf[1]);
            a8_imma(d1, qf[b], kf[2], kf[3]);
            const float2 k0 = *reinterpret_cast<const float2 *>(ksp + A2_K * b);
            const float2 k1 = *reinterpret_cast<const float2 *>(ksp + A2_K * b + 8);
            if constexpr (QROW) {
                s[0][0] = fmaf(a8_i2f(d0[0]), k0.x, s[0][0]);
                s[0][1] = fmaf(a8_i2f(d0[1]), k0.y, s[0][1]);
                s[0][2] = fmaf(a8_i2f(d0[2]), k0.x, s[0][2]);
                s[0][3] = fmaf(a8_i2f(d0[3]), k0.y, s[0][3]);
                s[1][0] = fmaf(a8_i2f(d1[0]), k1.x, s[1][0]);
                s[1][1] = fmaf(a8_i2f(d1[1]), k1.y, s[1][1]);
                s[1][2] = fmaf(a8_i2f(d1[2]), k1.x, s[1][2]);
                s[1][3] = fmaf(a8_i2f(d1[3]), k1.y, s[1][3]);
            } else {
                const float q0 = qs[b][0], q1 = qs[b][1];
                s[0][0] = fmaf(a8_i2f(d0[0]), q0 * k0.x, s[0][0]);
                s[0][1] = fmaf(a8_i2f(d0[1]), q0 * k0.y, s[0][1]);
                s[0][2] = fmaf(a8_i2f(d0[2]), q1 * k0.x, s[0][2]);
                s[0][3] = fmaf(a8_i2f(d0[3]), q1 * k0.y, s[0][3]);
                s[1][0] = fmaf(a8_i2f(d1[0]), q0 * k1.x, s[1][0]);
                s[1][1] = fmaf(a8_i2f(d1[1]), q0 * k1.y, s[1][1]);
                s[1][2] = fmaf(a8_i2f(d1[2]), q1 * k1.x, s[1][2]);
                s[1][3] = fmaf(a8_i2f(d1[3]), q1 * k1.y, s[1][3]);
            }
            // A9_CP bit 1: slot k after block 2 k + 1 (its fragment and scale loads are issued), off the barrier's load burst;
            // the stage is free from barrier(it) on and the group must land by barrier(it + 1): it does, it is committed here
            if constexpr (A9CS) {
                if (b % 2 == 1 && b / 2 < A9_NSL) {
                    const int k = b / 2;
                    const unsigned p = k ? cpok & (tid + A2_NT * k < cpn) : cpok;
                    if constexpr (A9CW) {
                        a9_cp16p(p, smem0 + st_c + cpd[k], cps[k] + (size_t) (unsigned) (ia - it0) * (unsigned) cpi[k]);
                    } else {
                        a9_cp16p(p, smem0 + st_c + cpd[k], cps[k]);
                        cps[k] += cpi[k];
                    }
                }
                if (b == 2 * A9_NSL - 1) pf::cpa_commit();
            }
        }
        if constexpr (QROW) {
#pragma unroll
            for (int ni = 0; ni < 2; ni++) {
                s[ni][0] *= qs[0][0];
                s[ni][1] *= qs[0][0];
                s[ni][2] *= qs[0][1];
                s[ni][3] *= qs[0][1];
            }
        }
        // causal mask, row maxima over the pair, P, row sums: as pk_attn2
        if (kb + A2_K - 1 > qpos0) { // (else every position of the tile is <= qpos0 <= the rows: nothing masked)
#pragma unroll
            for (int ni = 0; ni < 2; ni++) {
                const int p = kb + 16 * hf + 8 * ni + 2 * t4;
                if (p > qr0) s[ni][0] = -INFINITY;
                if (p + 1 > qr0) s[ni][1] = -INFINITY;
                if (p > qr1) s[ni][2] = -INFINITY;
                if (p + 1 > qr1) s[ni][3] = -INFINITY;
            }
        }
        float mx0 = fmaxf(fmaxf(s[0][0], s[0][1]), fmaxf(s[1][0], s[1][1])), mx1 = fmaxf(fmaxf(s[0][2], s[0][3]), fmaxf(s[1][2], s[1][3]));
        float sn0 = 0.f, sn1 = 0.f; // A9_EX 2: this lane's row minima
        (void) sn0, (void) sn1;
        if constexpr (A9EE) {
            sn0 = fminf(fminf(s[0][0], s[0][1]), fminf(s[1][0], s[1][1]));
            sn1 = fminf(fminf(s[0][2], s[0][3]), fminf(s[1][2], s[1][3]));
        }
        if constexpr (!A9MX) {
            mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 1));
            mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 2));
            mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 1));
            mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 2));
        }
        xmax[tid] = make_float2(mx0, mx1);
        a2_pair_sync(hq);
        if constexpr (A9MX) {
            // the lane maxima of the row's quad in this warp and in the partner: the max of the same 32 scores (fmaxf is
            // order-free here: NaN is dropped in any order, and no score is -0, so there is no signed-zero tie). Read by both
            // halves until pair sync 2: half 0's partial sums go to xps, not into these slots.
            const float4 * qo = reinterpret_cast<const float4 *>(xmax + (tid & ~3)), * qp = reinterpret_cast<const float4 *>(xmax + ((tid ^ 32) & ~3));
            const float4 a = qo[0], b = qo[1], c = qp[0], d = qp[1];
            mx0 = fmaxf(fmaxf(fmaxf(a.x, a.z), fmaxf(b.x, b.z)), fmaxf(fmaxf(c.x, c.z), fmaxf(d.x, d.z)));
            mx1 = fmaxf(fmaxf(fmaxf(a.y, a.w), fmaxf(b.y, b.w)), fmaxf(fmaxf(c.y, c.w), fmaxf(d.y, d.w)));
        } else {
            const float2 t = xmax[tid ^ 32];
            mx0 = fmaxf(mx0, t.x);
            mx1 = fmaxf(mx1, t.y);
        }
        const float mn0 = fmaxf(m0, mx0), mn1 = fmaxf(m1, mx1);
        float pv[2][4];
        float al0, al1;
        if constexpr (A9EX) {
            // A9_EX: exp2f(x) is the bare MUFU.EX2 for x >= -126 and NaN, so with every argument of the warp there (min ignores
            // NaN; any -inf or all-NaN argument sends the warp to exp2f) the plain ex2.approx.ftz gives exp2f's bits; else
            // exp2f again. Same arguments (the same FADDs) either way.
            float xa[2][4];
#pragma unroll
            for (int ni = 0; ni < 2; ni++)
#pragma unroll
                for (int j = 0; j < 4; j++) xa[ni][j] = s[ni][j] - (j < 2 ? mn0 : mn1);
            const float xb0 = m0 - mn0, xb1 = m1 - mn1;
#pragma unroll
            for (int ni = 0; ni < 2; ni++)
#pragma unroll
                for (int j = 0; j < 4; j++) pv[ni][j] = a9_ex2(xa[ni][j]);
            al0 = a9_ex2(xb0);
            al1 = a9_ex2(xb1);
            const float lo = fminf(fminf(fminf(fminf(xa[0][0], xa[0][1]), fminf(xa[1][0], xa[1][1])), fminf(fminf(xa[0][2], xa[0][3]), fminf(xa[1][2], xa[1][3]))),
                                   fminf(xb0, xb1));
            if (!__all_sync(0xffffffffu, lo >= -126.f)) {
#pragma unroll
                for (int ni = 0; ni < 2; ni++)
#pragma unroll
                    for (int j = 0; j < 4; j++) pv[ni][j] = exp2f(xa[ni][j]);
                al0 = exp2f(xb0);
                al1 = exp2f(xb1);
            }
        } else if constexpr (A9EE) {
            // A9_EX 2: the same choice before the exponentials. fl(s - mn) >= fl(min s - mn) for every non-NaN s of the lane's row (IEEE
            // rounding is monotonic; fminf skips NaN), so lo >= -126 means no argument of this lane is below -126 (a -inf one sends the
            // warp to exp2f). A NaN min s - mn (fminf drops it from lo) means every argument of that row is NaN: the same MUFU in both forms.
            const float xb0 = m0 - mn0, xb1 = m1 - mn1;
            const float lo = fminf(fminf(sn0 - mn0, sn1 - mn1), fminf(xb0, xb1));
            if (__all_sync(0xffffffffu, lo >= -126.f)) {
#pragma unroll
                for (int ni = 0; ni < 2; ni++)
#pragma unroll
                    for (int j = 0; j < 4; j++) pv[ni][j] = a9_ex2(s[ni][j] - (j < 2 ? mn0 : mn1));
                al0 = a9_ex2(xb0);
                al1 = a9_ex2(xb1);
            } else {
#pragma unroll
                for (int ni = 0; ni < 2; ni++)
#pragma unroll
                    for (int j = 0; j < 4; j++) pv[ni][j] = exp2f(s[ni][j] - (j < 2 ? mn0 : mn1));
                al0 = exp2f(xb0);
                al1 = exp2f(xb1);
            }
        } else {
#pragma unroll
            for (int ni = 0; ni < 2; ni++)
#pragma unroll
                for (int j = 0; j < 4; j++) pv[ni][j] = exp2f(s[ni][j] - (j < 2 ? mn0 : mn1));
            // mn < -inf can only happen in a warp with a row not yet finite: exp2f(NaN) -> 0 there (a8_ex2's selection); P above, al below
            al0 = exp2f(m0 - mn0), al1 = exp2f(m1 - mn1);
        }
        if (mnf == 0) {
            al0 = mn0 == -INFINITY ? 1.f : al0;
            al1 = mn1 == -INFINITY ? 1.f : al1;
#pragma unroll
            for (int ni = 0; ni < 2; ni++)
#pragma unroll
                for (int j = 0; j < 4; j++) pv[ni][j] = (j < 2 ? mn0 : mn1) == -INFINITY ? 0.f : pv[ni][j];
            mnf = __all_sync(0xffffffffu, mn0 != -INFINITY && mn1 != -INFINITY);
        }
        float x0[2], x1[2];
        unsigned pa[2][2];
#pragma unroll
        for (int ni = 0; ni < 2; ni++) {
            const float p0 = pv[ni][0], p1 = pv[ni][1], p2 = pv[ni][2], p3 = pv[ni][3];
            x0[ni] = p0 + p1;
            x1[ni] = p2 + p3;
            __half2 h01 = __floats2half2_rn(p0, p1), h23 = __floats2half2_rn(p2, p3);
            pa[ni][0] = *reinterpret_cast<unsigned *>(&h01);
            pa[ni][1] = *reinterpret_cast<unsigned *>(&h23);
        }
        xp[tid] = make_uint4(pa[0][0], pa[0][1], pa[1][0], pa[1][1]);
        if (hf == 0) {
            float ps0 = 0.f, ps1 = 0.f;
            ps0 += x0[0];
            ps0 += x0[1];
            ps1 += x1[0];
            ps1 += x1[1];
            if constexpr (A9MX) xps[hq * 32 + lane] = make_float2(ps0, ps1);
            else xmax[tid | 32] = make_float2(ps0, ps1);
        }
        a2_pair_sync(hq);
        const uint4 P0 = xp[tid & ~32], P1 = xp[tid | 32];
        // A9_RS: half 1's row-sum chain in every warp, finished in the P V block. Half 0 reads its own slot (xmax[tid]: its maxima,
        // read-only until its next maxima store; xps: its own partial sums, rewritten only by itself after the next pair sync 1)
        // and drops the result. Half 1: the same adds and shuffles in the same order as below.
        float rs0 = 0.f, rs1 = 0.f;
        (void) rs0, (void) rs1;
        if constexpr (A9RS) {
            const float2 t = A9MX ? xps[hq * 32 + lane] : xmax[tid];
            rs0 = t.x;
            rs1 = t.y;
            rs0 += x0[0];
            rs0 += x0[1];
            rs1 += x1[0];
            rs1 += x1[1];
        } else if (hf == 1) {
            const float2 t = A9MX ? xps[hq * 32 + lane] : xmax[tid];
            float ps0 = t.x, ps1 = t.y;
            ps0 += x0[0];
            ps0 += x0[1];
            ps1 += x1[0];
            ps1 += x1[1];
            ps0 += __shfl_xor_sync(0xffffffffu, ps0, 1);
            ps0 += __shfl_xor_sync(0xffffffffu, ps0, 2);
            ps1 += __shfl_xor_sync(0xffffffffu, ps1, 1);
            ps1 += __shfl_xor_sync(0xffffffffu, ps1, 2);
            l0 = l0 * al0 + ps0;
            l1 = l1 * al1 + ps1;
        }
        m0 = mn0;
        m1 = mn1;
        if constexpr (A9_HK) {
            if (warp < 8) {
                a9_cv<VB>(stv, sVc, 2, vbs);
                if constexpr (KB != 0) a9_k_rows<KB, 1>(stv, warp + 24, kon);
            } else if (warp == 9 && it + 1 < itf) {
                a9_ks(stv);
            }
        }
        // O += P V for this warp's 128 dims (fp16-accumulated per tile, as pk_attn2's PV16)
        {
            const unsigned A0[4] = {P0.x, P0.y, P0.z, P0.w}, A1[4] = {P1.x, P1.y, P1.z, P1.w};
#pragma unroll
            for (int j = 0; j < 8; j++) {
                unsigned b[4], h0[2], h1[2];
                if constexpr (A9_HK) {
                    if constexpr (KB != 0) {
                        if (j == 0) a9_k_rows<KB, 2>(stv, warp, kon);
                    }
                    if (j == 3) a9_cv<VB>(stv, sVc, 0, vbs);
                    if (j == 6) a9_cv<VB>(stv, sVc, 1, vbs);
                }
                if constexpr (A9RS) { // the two shuffle rounds between the first HMMA groups
                    if (j == 1) {
                        rs0 += __shfl_xor_sync(0xffffffffu, rs0, 1);
                        rs1 += __shfl_xor_sync(0xffffffffu, rs1, 1);
                    }
                    if (j == 2) {
                        rs0 += __shfl_xor_sync(0xffffffffu, rs0, 2);
                        rs1 += __shfl_xor_sync(0xffffffffu, rs1, 2);
                    }
                }
                a2_ldsm_t(b, va + 32 * j);
                pf::hmma16z(h0, A0, b[0], b[1]);
                pf::hmma16z(h1, A0, b[2], b[3]);
                a2_ldsm_t(b, va + 16 * A2_LD * 2 + 32 * j);
                pf::hmma16(h0, A1, b[0], b[1]);
                pf::hmma16(h1, A1, b[2], b[3]);
                const float2 u0 = __half22float2(pf::u2h(h0[0])), u1 = __half22float2(pf::u2h(h0[1]));
                const float2 w0 = __half22float2(pf::u2h(h1[0])), w1 = __half22float2(pf::u2h(h1[1]));
                o[2 * j][0] = o[2 * j][0] * al0 + u0.x;
                o[2 * j][1] = o[2 * j][1] * al0 + u0.y;
                o[2 * j][2] = o[2 * j][2] * al1 + u1.x;
                o[2 * j][3] = o[2 * j][3] * al1 + u1.y;
                o[2 * j + 1][0] = o[2 * j + 1][0] * al0 + w0.x;
                o[2 * j + 1][1] = o[2 * j + 1][1] * al0 + w0.y;
                o[2 * j + 1][2] = o[2 * j + 1][2] * al1 + w1.x;
                o[2 * j + 1][3] = o[2 * j + 1][3] * al1 + w1.y;
            }
        }
        if constexpr (A9RS) { // the hf == 1 block's l * al + ps compiles to this FFMA (contracted); half 0 keeps its l
            l0 = hf ? __fmaf_rn(l0, al0, rs0) : l0;
            l1 = hf ? __fmaf_rn(l1, al1, rs1) : l1;
        }
        if (ia < it1 && ia >= itf) { // not on the copy plan
            const int kba = ia * A2_K, npa = min(A2_K, pmax + 1 - kba);
            a8_st_ks(sta, issue_tile(sta, ia, npa));
        }
        if (!A9_HK || it + 1 >= itm) { // the next tile's real conversion
            expand_k(it + 1, warp, A2_NT / 32);
            a9_cv<VB>(stv, sVc, 0, vbn);
            a9_cv<VB>(stv, sVc, 1, vbn);
            if (warp < 8) a9_cv<VB>(stv, sVc, 2, vbn);
            if constexpr (!A9_HK) {
                if (warp == 9 && it + 1 < itf) a9_ks(stv);
            }
        }
        const unsigned st_t = st_a;
        st_a = st_b;
        st_b = st_c;
        st_c = st_t;
    }
    }
    pf::cpa_wait<0>();
    if (EXA && ex_role != 2) ex_fold(false); // R of all segments of this unit -> o / m / l
    // row sums: half 1 -> half 0 (the maxima slot is no longer read)
    if (hf) *mx_own = make_float2(l0, l1);
    a2_pair_sync(hq);
    if (!hf) {
        const float2 t = *mx_oth;
        l0 = t.x;
        l1 = t.y;
    }
    const int head = g * 6 + hq;
    if (nseg == 1) {
        const float il0 = l0 > 0.f ? 1.f / l0 : 0.f, il1 = l1 > 0.f ? 1.f / l1 : 0.f;
#pragma unroll
        for (int j = 0; j < 16; j++) {
            const int d = 128 * hf + 8 * j + 2 * t4;
            if (g8 < nq) *reinterpret_cast<float2 *>(cP.y2 + (size_t) (r0 + t0 + g8) * VD + head * HD + d) = make_float2(o[j][0] * il0, o[j][1] * il0);
            if (g8 + 8 < nq) *reinterpret_cast<float2 *>(cP.y2 + (size_t) (r0 + t0 + g8 + 8) * VD + head * HD + d) = make_float2(o[j][2] * il1, o[j][3] * il1);
        }
    } else {
        float * pp = !EXA ? part + (size_t) (blockIdx.x - nfull) * A2_PART
                          : ex_base + (size_t) (ex_role == 1 ? 0 : 1 + (it1 - 1) / segt - ex_h) * A2_PART;
        const int r0 = hq * A2_Q + g8;
#pragma unroll
        for (int j = 0; j < 16; j++) {
            const int d = 128 * hf + 8 * j + 2 * t4;
            *reinterpret_cast<float2 *>(pp + (size_t) r0 * HD + d) = make_float2(o[j][0], o[j][1]);
            *reinterpret_cast<float2 *>(pp + (size_t) (r0 + 8) * HD + d) = make_float2(o[j][2], o[j][3]);
        }
        if (hf && t4 == 0) {
            float2 * ml = reinterpret_cast<float2 *>(pp + 6 * A2_Q * HD);
            ml[r0] = make_float2(EXA ? m0 : m0 * (1.f / A8_L2E), l0); // natural-log units for pk_attn2_comb (EXA: log2)
            ml[r0 + 8] = make_float2(EXA ? m1 : m1 * (1.f / A8_L2E), l1);
        }
    }
}
