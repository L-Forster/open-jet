// Batched (T = 8 .. 16 rows) weight passes on the int8 tensor cores, ported from bench/mma_bench.cu for in-engine
// timing (MODE_BENCH, which 4 / 5) on the real weights. Activations are quantised per 32 values (q8_1-like);
// here they are synthetic (timing only). Each phase: 4-warp groups, 16-row items (mma rows g, g + 8), a quad loads
// a row's 64-byte window (2 chunks) and one shfl_xor(2) gives each lane 8 bytes of both chunks; header / meta words
// are shared in the quad; double-buffered steps of 2 pairs; per-warp partial sums combined per item.
#pragma once

namespace bmm {

constexpr int TM = 16;                              // rows a step may have (activation buffers' row stride)
constexpr int TS = 12;                              // rows staged in shared memory; rows TS .. TM - 1 are read from
                                                    // global memory (L1 / L2: every warp reads the same rows)
constexpr int SMEM = 101376;                        // dynamic shared memory of the bmm kernels (sm_86 opt-in maximum)
constexpr int NJ5 = D / 32, XS5 = NJ5 * 8 + 8;      // K = D: sub-blocks, activation row stride (words)
constexpr int DCB = 17, DCP = 2 * DCB, NCH = FF / 256 / DCB, DXS = DCB * 64 + 8; // K = FF: 4 chunks of 17 blocks
constexpr int Q6B = 192, Q6M = 18; // Q6_K per block: values (Mat::w), meta (Mat::d, in pairs of blocks)

__device__ const unsigned * g_bx;  // [TM][FF / 4] per-32 quantised activations (ffn_down input)
__device__ const float2 * g_bd;    // [FF / 32][TM] (scale, scale * sum)
__device__ float * g_bout;         // [TM][2 FF] outputs (timing sink)

#ifndef WEIGHT_EF
#define WEIGHT_EF ".L1::evict_first" // weights: streamed once
#endif
__device__ __forceinline__ int4 ld4(const void * p) {
    int4 r;
    asm volatile("ld.global.nc" WEIGHT_EF ".v4.s32 {%0,%1,%2,%3}, [%4];" : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
    return r;
}
__device__ __forceinline__ uint2 ld2(const void * p) {
    uint2 r;
    asm volatile("ld.global.nc.v2.u32 {%0,%1}, [%2];" : "=r"(r.x), "=r"(r.y) : "l"(p));
    return r;
}
// activation rows read from global (rows TS ..): kept in L1 ahead of the streaming weights
__device__ __forceinline__ uint2 ld2x(const void * p) {
    uint2 r;
    asm volatile("ld.global.nc.L1::evict_last.v2.u32 {%0,%1}, [%2];" : "=r"(r.x), "=r"(r.y) : "l"(p));
    return r;
}
__device__ __forceinline__ unsigned ld1(const void * p) {
    unsigned r;
    asm volatile("ld.global.nc.u32 %0, [%1];" : "=r"(r) : "l"(p));
    return r;
}
// Q6_K meta (Mat::d): per pair of blocks sc[16] even | sc[16] odd | d even | d odd (2 Q6M bytes). Word tg of a
// quad for 64-groups pc, pc + 1 of block b (absolute in the row): sc(pc) | d pair | sc(pc + 1) | d pair.
__device__ __forceinline__ unsigned q6_meta(const Mat & m, int row, int b, int pc, int tg) {
    const uint8_t * mp = reinterpret_cast<const uint8_t *>(m.d) + (size_t) row * m.nd * 2 + (b >> 1) * (2 * Q6M);
    return ld1(tg & 1 ? mp + 2 * Q6M - 4 : mp + 16 * (b & 1) + 4 * (pc + (tg >> 1)));
}
// block b's d from a d-pair word
__device__ __forceinline__ float q6_d(unsigned dw, int b) {
    return __half2float(__ushort_as_half((unsigned short) (dw >> (16 * (b & 1)))));
}
// activation B fragment (8 bytes at word w of row n): staged rows from shared memory (row stride xs words), the rest
// from global (row stride gs words); per-32 scale pairs (cols col, col + 1) likewise ([j][TS] staged, [j][TM] global)
__device__ __forceinline__ uint2 xfrag(const unsigned * xw, int xs, int trows, const unsigned * gxr, int gs, int n, int w) {
    return n < trows ? *reinterpret_cast<const uint2 *>(xw + n * xs + w) : ld2x(gxr + (size_t) n * gs + w);
}
// scales staged [j][xsc] for cols < trows, global [j][TM] past them
__device__ __forceinline__ float4 sfragr(const float2 * xd, int xsc, int trows, const float2 * gdr, int j, int col, int T) {
    if (col >= T) return make_float4(0.f, 0.f, 0.f, 0.f);
    return col < trows ? *reinterpret_cast<const float4 *>(xd + j * xsc + col) : __ldg(reinterpret_cast<const float4 *>(gdr + (size_t) j * TM + col));
}
template <int XSC> // scale columns staged per sub-block (TM: all, TS: rows TS .. from global)
__device__ __forceinline__ float4 sfrag(const float2 * xd, const float2 * gdr, int j, int col, int T) {
    if (col >= T) return make_float4(0.f, 0.f, 0.f, 0.f);
    return XSC == TM || col < TS ? *reinterpret_cast<const float4 *>(xd + j * XSC + col) : __ldg(reinterpret_cast<const float4 *>(gdr + (size_t) j * TM + col));
}
__device__ __forceinline__ void mma32(int (&d)[4], unsigned a0, unsigned a1, unsigned a2, unsigned a3, uint2 b) {
#if __CUDA_ARCH__ >= 800
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%10,%10,%10};"
                 : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b.x), "r"(b.y), "r"(0));
#else // (Turing: four m8n8k16, k 0..15 then 16..31 of rows g / g + 8)
    d[0] = d[1] = d[2] = d[3] = 0;
    mma_s8_8816(d[0], d[1], a0, b.x);
    mma_s8_8816(d[0], d[1], a2, b.y);
    mma_s8_8816(d[2], d[3], a1, b.x);
    mma_s8_8816(d[2], d[3], a3, b.y);
#endif
}

// K = D activations in shared memory: all T rows (xw [TM][XS5] words), then the per-32 scales compact ([NJ5][TM]
// words: fp16 scale | int16 sum of the int8 values) so 16 rows fit; then the epilogue's reduction slots
__device__ __forceinline__ unsigned pack_scale(float2 d) {
    const float sm = d.x > 0.f ? rintf(d.y / d.x) : 0.f;
    return (unsigned) __half_as_ushort(__float2half_rn(d.x)) | ((unsigned) (unsigned short) (short) sm << 16);
}
__device__ __forceinline__ float2 unpack_scale(unsigned u) {
    const float dx = __half2float(__ushort_as_half((unsigned short) (u & 0xffff)));
    return make_float2(dx, dx * (float) (short) (u >> 16));
}
// scales of cols col, col + 1 (zero past T) as (scale, scale * sum) x 2
__device__ __forceinline__ float4 sfragc(const unsigned * xdc, int j, int col, int T) {
    if (col >= T) return make_float4(0.f, 0.f, 0.f, 0.f);
    const uint2 u = *reinterpret_cast<const uint2 *>(xdc + j * TM + col);
    const float2 a = unpack_scale(u.x), b = unpack_scale(u.y);
    return make_float4(a.x, a.y, b.x, b.y);
}
// scale pair of the K = D kernels: compact (16 rows staged) or float2 [NJ5][TM] (up to 12 rows)
__device__ __forceinline__ float4 ld4fx(const void * p) { // (L1::evict_last: kept ahead of the streaming weights)
    float4 r;
    asm volatile("ld.global.nc.L1::evict_last.v4.f32 {%0,%1,%2,%3}, [%4];" : "=f"(r.x), "=f"(r.y), "=f"(r.z), "=f"(r.w) : "l"(p));
    return r;
}
// SG (16 rows): the scales read from global memory (L1) instead of decoding the compact ones (ffn gate / up 146 -> 139 us;
// the smaller K = D launches got ~1.5 us slower)
template <bool CS, bool SG = false>
__device__ __forceinline__ float4 sread(const unsigned * xdc, const float2 * gdp, int j, int col, int T) {
    if constexpr (CS) {
        if (SG) return col < T ? ld4fx(gdp + (size_t) j * TM + col) : make_float4(0.f, 0.f, 0.f, 0.f);
        return sfragc(xdc, j, col, T);
    }
    if (col >= T) return make_float4(0.f, 0.f, 0.f, 0.f);
    return *reinterpret_cast<const float4 *>(reinterpret_cast<const float2 *>(xdc) + j * TM + col);
}
// a group's 4 warps' partial sums (same fragment layout) -> warp 0, as (w0 + w2) + (w1 + w3); red: 2 slots per group
template <int NTL>
__device__ __forceinline__ void group_reduce(float (&acc)[NTL][4], float * red, int grp, int gw, int lane) {
    float4 * sl = reinterpret_cast<float4 *>(red) + (size_t) grp * 2 * 32 * NTL;
    if (gw >= 2)
#pragma unroll
        for (int nt = 0; nt < NTL; nt++) sl[((gw - 2) * 32 + lane) * NTL + nt] = make_float4(acc[nt][0], acc[nt][1], acc[nt][2], acc[nt][3]);
    asm volatile("bar.sync %0, 128;" ::"r"(1 + grp) : "memory");
    if (gw < 2)
#pragma unroll
        for (int nt = 0; nt < NTL; nt++) {
            const float4 v = sl[(gw * 32 + lane) * NTL + nt];
            acc[nt][0] += v.x; acc[nt][1] += v.y; acc[nt][2] += v.z; acc[nt][3] += v.w;
        }
    asm volatile("bar.sync %0, 128;" ::"r"(1 + grp) : "memory");
    if (gw == 1)
#pragma unroll
        for (int nt = 0; nt < NTL; nt++) sl[lane * NTL + nt] = make_float4(acc[nt][0], acc[nt][1], acc[nt][2], acc[nt][3]);
    asm volatile("bar.sync %0, 128;" ::"r"(1 + grp) : "memory");
    if (gw == 0)
#pragma unroll
        for (int nt = 0; nt < NTL; nt++) {
            const float4 v = sl[lane * NTL + nt];
            acc[nt][0] += v.x; acc[nt][1] += v.y; acc[nt][2] += v.z; acc[nt][3] += v.w;
        }
    asm volatile("bar.sync %0, 128;" ::"r"(1 + grp) : "memory");
}
template <bool CS = true, bool SG = false> __device__ void stage_x5(const unsigned * gx, const float2 * gd, int T);
__device__ void stage_x(const unsigned * gx, const float2 * gd, int T, int K, int TR, int gs, int xsc);

// synthetic per-32 activations for K = D in shared memory (xw [TM][XS5] words, xd [NJ5][TM])
__device__ void setup_x5(int T) {
    unsigned * xw = reinterpret_cast<unsigned *>(smem);
    unsigned * xdc = xw + TM * XS5;
    for (int i = threadIdx.x; i < T * NJ5 * 8; i += NT) {
        const int t = i / (NJ5 * 8), q = i % (NJ5 * 8);
        xw[t * XS5 + q] = 0x01020304u * (unsigned) (t + 1) ^ (unsigned) q * 2654435761u;
    }
    for (int i = threadIdx.x; i < NJ5 * TM; i += NT) xdc[i] = pack_scale(make_float2(0.004f, 0.1f));
    __syncthreads();
}

// Q4_K GEMV (K = D) of matrix m for T rows -> out [T][m.N]
template <int NTL_>
__device__ __forceinline__ void q4(const Mat m, float * out, int ld, int add, int T, const unsigned * sgx = nullptr, const float2 * sgd = nullptr) {
    // NTL_ 1: up to 8 rows, 2: up to 12, 3: up to 16 (2 tiles, all rows staged with compact scales)
    constexpr int SP = 2, NTL = NTL_ == 3 ? 2 : NTL_, SPC = (10 + SP - 1) / SP, NH = 1; // (pa, SP even: a stage stays in one block)
    constexpr bool CS = NTL_ == 3;
    const int NI = m.N / 16, rb = m.rb;
    const unsigned * xw = reinterpret_cast<const unsigned *>(smem);
    const unsigned * xdc = xw + (CS ? TM : TS) * XS5;                     // scales [NJ5][TM]: compact words or float2
    float * red = reinterpret_cast<float *>(const_cast<unsigned *>(xdc + NJ5 * TM * (CS ? 1 : 2))); // [4 groups][2][32][NTL] float4
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, g = lane >> 2, tg = lane & 3, qb = lane & ~3;
    // a group's 4 warps take alternate blocks (warp gw: blocks 4 ks + gw): the group reads adjacent memory at the same
    // time (contiguous quarters per warp streamed 5-10% slower)
    const int grp = warp >> 2, gw = warp & 3, pa = SP * gw, pb = 40, PST = 4 * SP;
    const int offA = (tg & 1) * 16 + (tg >> 1) * 8, offB = offA ^ 8;
    const int it0 = 4 * blockIdx.x + grp, istep = 4 * gridDim.x;
#define B4_LOAD(W, H, IT_, KS_)                                                                        \
    {                                                                                                  \
        const int p0_ = pa + PST * (KS_), hb_ = p0_ / 2;                                                \
        _Pragma("unroll") for (int k = 0; k < 2; k++) {                                                \
            const uint8_t * w = m.w + (size_t) (16 * (IT_) + g + 8 * k) * rb;                          \
            H[k] = ld4(w + hb_ * 144); /* the block header, whole in every lane (one broadcast load per quad) */ \
            _Pragma("unroll") for (int i = 0; i < SP; i++) {                                           \
                const int P = p0_ + i;                                                                 \
                if (P < pb) { const int4 v = ld4(w + (P / 2) * 144 + 16 + 64 * (P % 2) + 16 * tg); W[k][i] = make_uint4(v.x, v.y, v.z, v.w); } \
                else W[k][i] = make_uint4(0, 0, 0, 0);                                                 \
            }                                                                                          \
        }                                                                                              \
    }
#define B4_STEP(W, H, IT_, KS_)                                                                        \
    {                                                                                                  \
        const int p0_ = pa + PST * (KS_), hb_ = p0_ / 2;                                                \
        _Pragma("unroll") for (int i = 0; i < SP; i++) {                                               \
            const int P = p0_ + i;                                                                     \
            if (P < pb) {                                                                              \
            const int bk = P / 2, bi = bk - hb_;                                                       \
            uint2 cA[2], cB[2];                                                                        \
            int4 hcur[2];                                                                              \
            _Pragma("unroll") for (int k = 0; k < 2; k++) {                                            \
                const unsigned ox = __shfl_xor_sync(~0u, W[k][i].z, 2), oy = __shfl_xor_sync(~0u, W[k][i].w, 2); \
                const uint2 own = make_uint2(W[k][i].x, W[k][i].y), got = make_uint2(ox, oy);         \
                cA[k] = tg < 2 ? own : got;                                                            \
                cB[k] = tg < 2 ? got : own;                                                            \
                hcur[k] = H[k];                                                                        \
            }                                                                                          \
            _Pragma("unroll") for (int hc = 0; hc < 2; hc++) {                                         \
                const int p = 2 * (P % 2) + hc, off = hc ? offB : offA;                                \
                float fA[2], gA[2], fB[2], gB[2];                                                      \
                _Pragma("unroll") for (int k = 0; k < 2; k++) q4k_scales(hcur[k], p, fA[k], gA[k], fB[k], gB[k]); \
                const uint2 r0 = hc ? cB[0] : cA[0], r1 = hc ? cB[1] : cA[1];                          \
                _Pragma("unroll") for (int nib = 0; nib < 2; nib++) {                                  \
                    const int sj = 8 * bk + 2 * p + nib, sh = 4 * nib;                                 \
                    const float f0 = nib ? fB[0] : fA[0], f1 = nib ? fB[1] : fA[1];                    \
                    const float g0 = nib ? gB[0] : gA[0], g1 = nib ? gB[1] : gA[1];                    \
                    const unsigned a0 = (r0.x >> sh) & M4, a1 = (r1.x >> sh) & M4, a2 = (r0.y >> sh) & M4, a3 = (r1.y >> sh) & M4; \
                    _Pragma("unroll") for (int nt = 0; nt < NTL; nt++) {                               \
                        const int n = 8 * nt + g;                                                      \
                        const uint2 bf = n < T ? *reinterpret_cast<const uint2 *>(xw + n * XS5 + 8 * sj + off / 4) : make_uint2(0, 0); \
                        const float4 sc = sread<CS>(xdc, sgd, sj, 8 * nt + 2 * tg, T); \
                        int d[4];                                                                      \
                        mma32(d, a0, a1, a2, a3, bf);                                                  \
                        acc[nt][0] = fmaf(f0 * sc.x, (float) d[0], acc[nt][0]);                        \
                        acc[nt][1] = fmaf(f0 * sc.z, (float) d[1], acc[nt][1]);                        \
                        acc[nt][2] = fmaf(f1 * sc.x, (float) d[2], acc[nt][2]);                        \
                        acc[nt][3] = fmaf(f1 * sc.z, (float) d[3], acc[nt][3]);                        \
                        acc[nt][0] = fmaf(-g0, sc.y, acc[nt][0]);                                      \
                        acc[nt][1] = fmaf(-g0, sc.w, acc[nt][1]);                                      \
                        acc[nt][2] = fmaf(-g1, sc.y, acc[nt][2]);                                      \
                        acc[nt][3] = fmaf(-g1, sc.w, acc[nt][3]);                                      \
                    }                                                                                  \
                }                                                                                      \
            }                                                                                          \
            }                                                                                          \
        }                                                                                              \
        if ((KS_) == SPC - 1) {                                                                        \
            group_reduce<NTL>(acc, red, grp, gw, lane);                                               \
            if (gw == 0)                                                                               \
            _Pragma("unroll") for (int nt = 0; nt < NTL; nt++)                                         \
                _Pragma("unroll") for (int c = 0; c < 4; c++) {                                        \
                    const int t = 8 * nt + 2 * tg + (c & 1), row = g + 8 * (c >> 1);                   \
                    if (t < T) { float * o_ = out + (size_t) t * ld + 16 * (IT_) + row; *o_ = add ? *o_ + acc[nt][c] : acc[nt][c]; }                                                     \
                }                                                                                      \
            _Pragma("unroll") for (int nt = 0; nt < NTL; nt++)                                         \
                _Pragma("unroll") for (int c = 0; c < 4; c++) acc[nt][c] = 0.f;                        \
        }                                                                                              \
    }
    float acc[NTL][4] = {};
    uint4 WA[2][SP], WB[2][SP];
    int4 HA[2], HB[2];
    int I0 = it0, K0 = 0;
    if (!sgx && I0 >= NI) return;
    if (I0 < NI) B4_LOAD(WA, HA, I0, K0) // (in flight while the activations are staged)
    if (sgx) { stage_x5<CS>(sgx, sgd, T); if (I0 >= NI) return; }
    while (true) {
        int I1 = I0, K1 = K0 + 1;
        if (K1 == SPC) { K1 = 0; I1 += istep; }
        const bool v1 = I1 < NI;
        if (v1) B4_LOAD(WB, HB, I1, K1)
        B4_STEP(WA, HA, I0, K0)
        if (!v1) break;
        int I2 = I1, K2 = K1 + 1;
        if (K2 == SPC) { K2 = 0; I2 += istep; }
        const bool v2 = I2 < NI;
        if (v2) B4_LOAD(WA, HA, I2, K2)
        B4_STEP(WB, HB, I1, K1)
        if (!v2) break;
        I0 = I2; K0 = K2;
    }
#undef B4_LOAD
#undef B4_STEP
}

// q4 over up to 3 Q4_K matrices with the same input (K = D) in one launch: items of the matrices back to back, so
// the groups stream more items and the launch has one tail instead of three (gate + up; qkv + gate; q + k + v)
struct MV {
    const uint8_t * w[3];
    float * out[3];
    int ld[3], iend[3]; // cumulative 16-row items
    int nm, rb, add;
};
template <int NTL_, bool SG = false>
__device__ __forceinline__ void q4m(const MV mv, int T, const unsigned * sgx = nullptr, const float2 * sgd = nullptr) {
    // NTL_ 1: up to 8 rows, 2: up to 12, 3: up to 16 (2 tiles, all rows staged with compact scales)
    constexpr int SP = 2, NTL = NTL_ == 3 ? 2 : NTL_, SPC = (10 + SP - 1) / SP, NH = 1; // (pa, SP even: a stage stays in one block)
    constexpr bool CS = NTL_ == 3;
    const int NI = mv.iend[2], rb = mv.rb; // (unused iend entries repeat the total)
    // (the parameter's fields in registers, selected by constant index: a runtime index into its arrays, or taking
    // its address, puts it on the stack)
    const int ie0 = mv.iend[0], ie1 = mv.iend[1], ld0 = mv.ld[0], ld1_ = mv.ld[1], ld2 = mv.ld[2], madd = mv.add;
    const uint8_t *w0 = mv.w[0], *w1 = mv.w[1], *w2 = mv.w[2];
    float *o0 = mv.out[0], *o1 = mv.out[1], *o2 = mv.out[2];
    auto msel = [&](int it) { return it < ie0 ? 0 : it < ie1 ? 1 : 2; };
    auto mbase = [&](int mi) { return mi == 0 ? 0 : mi == 1 ? ie0 : ie1; };
    auto mw = [&](int mi) { return mi == 0 ? w0 : mi == 1 ? w1 : w2; };
    auto mout = [&](int mi) { return mi == 0 ? o0 : mi == 1 ? o1 : o2; };
    auto mld = [&](int mi) { return mi == 0 ? ld0 : mi == 1 ? ld1_ : ld2; };
    const unsigned * xw = reinterpret_cast<const unsigned *>(smem);
    const unsigned * xdc = xw + (CS ? TM : TS) * XS5;                     // scales [NJ5][TM]: compact words or float2
    float * red = reinterpret_cast<float *>(const_cast<unsigned *>(xdc + NJ5 * TM * (CS ? 1 : 2))); // [4 groups][2][32][NTL] float4
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, g = lane >> 2, tg = lane & 3, qb = lane & ~3;
    // a group's 4 warps take alternate blocks (warp gw: blocks 4 ks + gw): the group reads adjacent memory at the same
    // time (contiguous quarters per warp streamed 5-10% slower)
    const int grp = warp >> 2, gw = warp & 3, pa = SP * gw, pb = 40, PST = 4 * SP;
    const int offA = (tg & 1) * 16 + (tg >> 1) * 8, offB = offA ^ 8;
    const int it0 = 4 * blockIdx.x + grp, istep = 4 * gridDim.x;
#define B4M_LOAD(W, H, IT_, KS_)                                                                        \
    {                                                                                                  \
        const int p0_ = pa + PST * (KS_), hb_ = p0_ / 2;                                                \
        _Pragma("unroll") for (int k = 0; k < 2; k++) {                                                \
            const int mi_ = msel(IT_), li_ = (IT_) - mbase(mi_);                                       \
            const uint8_t * w = mw(mi_) + (size_t) (16 * li_ + g + 8 * k) * rb;                        \
            H[k] = ld4(w + hb_ * 144); /* the block header, whole in every lane (one broadcast load per quad) */ \
            _Pragma("unroll") for (int i = 0; i < SP; i++) {                                           \
                const int P = p0_ + i;                                                                 \
                if (P < pb) { const int4 v = ld4(w + (P / 2) * 144 + 16 + 64 * (P % 2) + 16 * tg); W[k][i] = make_uint4(v.x, v.y, v.z, v.w); } \
                else W[k][i] = make_uint4(0, 0, 0, 0);                                                 \
            }                                                                                          \
        }                                                                                              \
    }
#define B4M_STEP(W, H, IT_, KS_)                                                                        \
    {                                                                                                  \
        const int p0_ = pa + PST * (KS_), hb_ = p0_ / 2;                                                \
        _Pragma("unroll") for (int i = 0; i < SP; i++) {                                               \
            const int P = p0_ + i;                                                                     \
            if (P < pb) {                                                                              \
            const int bk = P / 2, bi = bk - hb_;                                                       \
            uint2 cA[2], cB[2];                                                                        \
            int4 hcur[2];                                                                              \
            _Pragma("unroll") for (int k = 0; k < 2; k++) {                                            \
                const unsigned ox = __shfl_xor_sync(~0u, W[k][i].z, 2), oy = __shfl_xor_sync(~0u, W[k][i].w, 2); \
                const uint2 own = make_uint2(W[k][i].x, W[k][i].y), got = make_uint2(ox, oy);         \
                cA[k] = tg < 2 ? own : got;                                                            \
                cB[k] = tg < 2 ? got : own;                                                            \
                hcur[k] = H[k];                                                                        \
            }                                                                                          \
            _Pragma("unroll") for (int hc = 0; hc < 2; hc++) {                                         \
                const int p = 2 * (P % 2) + hc, off = hc ? offB : offA;                                \
                float fA[2], gA[2], fB[2], gB[2];                                                      \
                _Pragma("unroll") for (int k = 0; k < 2; k++) q4k_scales(hcur[k], p, fA[k], gA[k], fB[k], gB[k]); \
                const uint2 r0 = hc ? cB[0] : cA[0], r1 = hc ? cB[1] : cA[1];                          \
                _Pragma("unroll") for (int nib = 0; nib < 2; nib++) {                                  \
                    const int sj = 8 * bk + 2 * p + nib, sh = 4 * nib;                                 \
                    const float f0 = nib ? fB[0] : fA[0], f1 = nib ? fB[1] : fA[1];                    \
                    const float g0 = nib ? gB[0] : gA[0], g1 = nib ? gB[1] : gA[1];                    \
                    const unsigned a0 = (r0.x >> sh) & M4, a1 = (r1.x >> sh) & M4, a2 = (r0.y >> sh) & M4, a3 = (r1.y >> sh) & M4; \
                    _Pragma("unroll") for (int nt = 0; nt < NTL; nt++) {                               \
                        const int n = 8 * nt + g;                                                      \
                        const uint2 bf = n < T ? *reinterpret_cast<const uint2 *>(xw + n * XS5 + 8 * sj + off / 4) : make_uint2(0, 0); \
                        const float4 sc = sread<CS, SG>(xdc, sgd, sj, 8 * nt + 2 * tg, T); \
                        int d[4];                                                                      \
                        mma32(d, a0, a1, a2, a3, bf);                                                  \
                        acc[nt][0] = fmaf(f0 * sc.x, (float) d[0], acc[nt][0]);                        \
                        acc[nt][1] = fmaf(f0 * sc.z, (float) d[1], acc[nt][1]);                        \
                        acc[nt][2] = fmaf(f1 * sc.x, (float) d[2], acc[nt][2]);                        \
                        acc[nt][3] = fmaf(f1 * sc.z, (float) d[3], acc[nt][3]);                        \
                        acc[nt][0] = fmaf(-g0, sc.y, acc[nt][0]);                                      \
                        acc[nt][1] = fmaf(-g0, sc.w, acc[nt][1]);                                      \
                        acc[nt][2] = fmaf(-g1, sc.y, acc[nt][2]);                                      \
                        acc[nt][3] = fmaf(-g1, sc.w, acc[nt][3]);                                      \
                    }                                                                                  \
                }                                                                                      \
            }                                                                                          \
            }                                                                                          \
        }                                                                                              \
        if ((KS_) == SPC - 1) {                                                                        \
            group_reduce<NTL>(acc, red, grp, gw, lane);                                               \
            if (gw == 0)                                                                               \
            _Pragma("unroll") for (int nt = 0; nt < NTL; nt++)                                         \
                _Pragma("unroll") for (int c = 0; c < 4; c++) {                                        \
                    const int t = 8 * nt + 2 * tg + (c & 1), row = g + 8 * (c >> 1);                   \
                    if (t < T) { const int mi_ = msel(IT_), li_ = (IT_) - mbase(mi_); float * o_ = mout(mi_) + (size_t) t * mld(mi_) + 16 * li_ + row; *o_ = madd ? *o_ + acc[nt][c] : acc[nt][c]; }                                                     \
                }                                                                                      \
            _Pragma("unroll") for (int nt = 0; nt < NTL; nt++)                                         \
                _Pragma("unroll") for (int c = 0; c < 4; c++) acc[nt][c] = 0.f;                        \
        }                                                                                              \
    }
    float acc[NTL][4] = {};
    uint4 WA[2][SP], WB[2][SP];
    int4 HA[2], HB[2];
    int I0 = it0, K0 = 0;
    if (!sgx && I0 >= NI) return;
    if (I0 < NI) B4M_LOAD(WA, HA, I0, K0) // (in flight while the activations are staged)
    if (sgx) { stage_x5<CS, SG>(sgx, sgd, T); if (I0 >= NI) return; }
    while (true) {
        int I1 = I0, K1 = K0 + 1;
        if (K1 == SPC) { K1 = 0; I1 += istep; }
        const bool v1 = I1 < NI;
        if (v1) B4M_LOAD(WB, HB, I1, K1)
        B4M_STEP(WA, HA, I0, K0)
        if (!v1) break;
        int I2 = I1, K2 = K1 + 1;
        if (K2 == SPC) { K2 = 0; I2 += istep; }
        const bool v2 = I2 < NI;
        if (v2) B4M_LOAD(WA, HA, I2, K2)
        B4M_STEP(WB, HB, I1, K1)
        if (!v2) break;
        I0 = I2; K0 = K2;
    }
#undef B4M_LOAD
#undef B4M_STEP
}

// Q4_K / Q5_K GEMV with runtime K (a multiple of 512) for T rows: as q4, activations staged by stage_x (row stride
// K / 4 + 8 words, scales [K / 32][TM]). Q5_K (native 176-byte blocks: header | qh[32] | qs[128]): lane tg keeps qh
// bytes 8 tg .. 8 tg + 7 of each of its blocks; a chunk's fifth bits come from quad lane off / 8.
template <int NTL_, int TYPE>
__device__ __forceinline__ void qk(const Mat m, float * out, int ld, int add, int T, const unsigned * sgx = nullptr, const float2 * sgd = nullptr,
                                   int gs = 0) {
    constexpr int SP = 2, NTL = NTL_, NH = 1, BS = TYPE == T_Q5K ? 176 : 144, WO = TYPE == T_Q5K ? 48 : 16;
    const int K = m.K, PW = K / 512, SPC = (PW + SP - 1) / SP, KB = K / 256, XS = K / 4 + 8, NJ = K / 32;
    const int NI = m.N / 16, rb = m.rb;
    const unsigned * xw = reinterpret_cast<const unsigned *>(smem);
    // staged rows: all 16 when K <= 3072 (the K = 6144 matrices run as two halves past 12 rows), else 12 (rows
    // past them from global memory); gs: the activations' row stride in words (K / 4 unless a K half)
    const int XSC = K <= 3072 ? TM : TS, TRr = min(T, XSC);
    if (!gs) gs = K / 4;
    const float2 * xd = reinterpret_cast<const float2 *>(xw + TRr * XS);
    float * red = reinterpret_cast<float *>(smem + (TRr * XS * 4 + NJ * XSC * 8)); // [4 groups][2 slots][16][T]
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, g = lane >> 2, tg = lane & 3, qb = lane & ~3;
    const int grp = warp >> 2, gw = warp & 3, pa = SP * gw, pb = 4 * PW, PST = 4 * SP; // (alternate blocks, as q4)
    const int offA = (tg & 1) * 16 + (tg >> 1) * 8, offB = offA ^ 8;
    const int it0 = 4 * blockIdx.x + grp, istep = 4 * gridDim.x;
#define BK_LOAD(W, H, QH, IT_, KS_)                                                                        \
    {                                                                                                  \
        const int p0_ = pa + PST * (KS_), hb_ = p0_ / 2;                                                \
        _Pragma("unroll") for (int k = 0; k < 2; k++) {                                                \
            const uint8_t * w = m.w + (size_t) (16 * (IT_) + g + 8 * k) * rb;                          \
            _Pragma("unroll") for (int b = 0; b < NH; b++) {                                           \
                H[k] = ld4(w + hb_ * BS); /* the block header, whole in every lane */                 \
                if (TYPE == T_Q5K) QH[k][b] = (hb_ + b) < KB ? ld2(w + (hb_ + b) * BS + 16 + 8 * tg) : make_uint2(0, 0); \
            }                                                                                          \
            _Pragma("unroll") for (int i = 0; i < SP; i++) {                                           \
                const int P = p0_ + i;                                                                 \
                if (P < pb) { const int4 v = ld4(w + (P / 2) * BS + WO + 64 * (P % 2) + 16 * tg); W[k][i] = make_uint4(v.x, v.y, v.z, v.w); } \
                else W[k][i] = make_uint4(0, 0, 0, 0);                                                 \
            }                                                                                          \
        }                                                                                              \
    }
#define BK_STEP(W, H, QH, IT_, KS_)                                                                        \
    {                                                                                                  \
        const int p0_ = pa + PST * (KS_), hb_ = p0_ / 2;                                                \
        _Pragma("unroll") for (int i = 0; i < SP; i++) {                                               \
            const int P = p0_ + i;                                                                     \
            if (P < pb) {                                                                              \
            const int bk = P / 2, bi = bk - hb_;                                                       \
            uint2 cA[2], cB[2];                                                                        \
            int4 hcur[2];                                                                              \
            _Pragma("unroll") for (int k = 0; k < 2; k++) {                                            \
                const unsigned ox = __shfl_xor_sync(~0u, W[k][i].z, 2), oy = __shfl_xor_sync(~0u, W[k][i].w, 2); \
                const uint2 own = make_uint2(W[k][i].x, W[k][i].y), got = make_uint2(ox, oy);         \
                cA[k] = tg < 2 ? own : got;                                                            \
                cB[k] = tg < 2 ? got : own;                                                            \
                hcur[k] = H[k];                                                                        \
            }                                                                                          \
            _Pragma("unroll") for (int hc = 0; hc < 2; hc++) {                                         \
                const int p = 2 * (P % 2) + hc, off = hc ? offB : offA;                                \
                float fA[2], gA[2], fB[2], gB[2];                                                      \
                _Pragma("unroll") for (int k = 0; k < 2; k++) q4k_scales(hcur[k], p, fA[k], gA[k], fB[k], gB[k]); \
                const uint2 r0 = hc ? cB[0] : cA[0], r1 = hc ? cB[1] : cA[1];                          \
                uint2 f5[2]; /* Q5_K: this lane's qh bytes off .. off + 7 of rows g, g + 8 */          \
                if (TYPE == T_Q5K) {                                                                   \
                    _Pragma("unroll") for (int k = 0; k < 2; k++) {                                    \
                        const uint2 qv = QH[k][0];                                                     \
                        f5[k] = make_uint2(__shfl_sync(~0u, qv.x, qb + off / 8), __shfl_sync(~0u, qv.y, qb + off / 8)); \
                    }                                                                                  \
                }                                                                                      \
                _Pragma("unroll") for (int nib = 0; nib < 2; nib++) {                                  \
                    const int sj = 8 * bk + 2 * p + nib, sh = 4 * nib;                                 \
                    const float f0 = nib ? fB[0] : fA[0], f1 = nib ? fB[1] : fA[1];                    \
                    const float g0 = nib ? gB[0] : gA[0], g1 = nib ? gB[1] : gA[1];                    \
                    unsigned a0 = (r0.x >> sh) & M4, a1 = (r1.x >> sh) & M4, a2 = (r0.y >> sh) & M4, a3 = (r1.y >> sh) & M4; \
                    if (TYPE == T_Q5K) {                                                               \
                        const int fb = 2 * p + nib;                                                    \
                        a0 |= ((f5[0].x >> fb) & 0x01010101u) << 4; a1 |= ((f5[1].x >> fb) & 0x01010101u) << 4; \
                        a2 |= ((f5[0].y >> fb) & 0x01010101u) << 4; a3 |= ((f5[1].y >> fb) & 0x01010101u) << 4; \
                    }                                                                                  \
                    _Pragma("unroll") for (int nt = 0; nt < NTL; nt++) {                               \
                        const int n = 8 * nt + g;                                                      \
                        const uint2 bf = n < T ? xfrag(xw, XS, TRr, sgx, gs, n, 8 * sj + off / 4) : make_uint2(0, 0); \
                        const float4 sc = sfragr(xd, XSC, TRr, sgd, sj, 8 * nt + 2 * tg, T); \
                        int d[4];                                                                      \
                        mma32(d, a0, a1, a2, a3, bf);                                                  \
                        acc[nt][0] = fmaf(f0 * sc.x, (float) d[0], acc[nt][0]);                        \
                        acc[nt][1] = fmaf(f0 * sc.z, (float) d[1], acc[nt][1]);                        \
                        acc[nt][2] = fmaf(f1 * sc.x, (float) d[2], acc[nt][2]);                        \
                        acc[nt][3] = fmaf(f1 * sc.z, (float) d[3], acc[nt][3]);                        \
                        acc[nt][0] = fmaf(-g0, sc.y, acc[nt][0]);                                      \
                        acc[nt][1] = fmaf(-g0, sc.w, acc[nt][1]);                                      \
                        acc[nt][2] = fmaf(-g1, sc.y, acc[nt][2]);                                      \
                        acc[nt][3] = fmaf(-g1, sc.w, acc[nt][3]);                                      \
                    }                                                                                  \
                }                                                                                      \
            }                                                                                          \
            }                                                                                          \
        }                                                                                              \
        if ((KS_) == SPC - 1) {                                                                        \
            /* two passes over 2 slots per group (warps 0, 1 store, 2, 3 add): half the shared memory */ \
            float * slot = red + (grp * 2 + (gw & 1)) * 16 * T;                                        \
            if (gw < 2)                                                                                \
            _Pragma("unroll") for (int nt = 0; nt < NTL; nt++)                                         \
                _Pragma("unroll") for (int c = 0; c < 4; c++) {                                        \
                    const int t = 8 * nt + 2 * tg + (c & 1);                                           \
                    if (t < T) slot[(g + 8 * (c >> 1)) * T + t] = acc[nt][c];                          \
                }                                                                                      \
            asm volatile("bar.sync %0, 128;" ::"r"(1 + grp) : "memory");                               \
            if (gw >= 2)                                                                               \
            _Pragma("unroll") for (int nt = 0; nt < NTL; nt++)                                         \
                _Pragma("unroll") for (int c = 0; c < 4; c++) {                                        \
                    const int t = 8 * nt + 2 * tg + (c & 1);                                           \
                    if (t < T) slot[(g + 8 * (c >> 1)) * T + t] += acc[nt][c];                         \
                }                                                                                      \
            _Pragma("unroll") for (int nt = 0; nt < NTL; nt++)                                         \
                _Pragma("unroll") for (int c = 0; c < 4; c++) acc[nt][c] = 0.f;                        \
            asm volatile("bar.sync %0, 128;" ::"r"(1 + grp) : "memory");                               \
            for (int o = threadIdx.x & 127; o < 16 * T; o += 128) {                                    \
                const int row = o / T, t = o % T;                                                      \
                float a = 0.f;                                                                         \
                _Pragma("unroll") for (int w2 = 0; w2 < 2; w2++) a += red[((grp * 2 + w2) * 16 + row) * T + t]; \
                { float * o_ = out + (size_t) t * ld + 16 * (IT_) + row; *o_ = add ? *o_ + a : a; }     \
            }                                                                                          \
            asm volatile("bar.sync %0, 128;" ::"r"(1 + grp) : "memory");                               \
        }                                                                                              \
    }
    float acc[NTL][4] = {};
    uint4 WA[2][SP], WB[2][SP];
    int4 HA[2], HB[2];
    uint2 QA[2][NH], QB[2][NH];
    int I0 = it0, K0 = 0;
    if (!sgx && I0 >= NI) return;
    if (I0 < NI) BK_LOAD(WA, HA, QA, I0, K0) // (in flight while the activations are staged)
    if (sgx) { stage_x(sgx, sgd, T, K, TRr, gs, XSC); if (I0 >= NI) return; }
    while (true) {
        int I1 = I0, K1 = K0 + 1;
        if (K1 == SPC) { K1 = 0; I1 += istep; }
        const bool v1 = I1 < NI;
        if (v1) BK_LOAD(WB, HB, QB, I1, K1)
        BK_STEP(WA, HA, QA, I0, K0)
        if (!v1) break;
        int I2 = I1, K2 = K1 + 1;
        if (K2 == SPC) { K2 = 0; I2 += istep; }
        const bool v2 = I2 < NI;
        if (v2) BK_LOAD(WA, HA, QA, I2, K2)
        BK_STEP(WB, HB, QB, I1, K1)
        if (!v2) break;
        I0 = I2; K0 = K2;
    }
#undef BK_LOAD
#undef BK_STEP
}

// Q6_K GEMV (K = D, repacked 192-byte blocks + the meta stream) of matrix m for T rows -> out [T][m.N]: masked-half mma pairs
template <int NTL_>
__device__ __forceinline__ void q6(const Mat m, float * out, int ld, int add, int T, const unsigned * sgx = nullptr, const float2 * sgd = nullptr) {
    constexpr int SP = 2;
    constexpr int NTL = NTL_ == 3 ? 2 : NTL_, SPC = (10 + SP - 1) / SP; // (NTL_ as q4)
    constexpr bool CS = NTL_ == 3;
    const int NI = m.N / 16;
    const unsigned * xw = reinterpret_cast<const unsigned *>(smem);
    const unsigned * xdc = xw + (CS ? TM : TS) * XS5;                     // scales [NJ5][TM]: compact words or float2
    float * red = reinterpret_cast<float *>(const_cast<unsigned *>(xdc + NJ5 * TM * (CS ? 1 : 2))); // [4 groups][2][32][NTL] float4
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, g = lane >> 2, tg = lane & 3, hh = tg & 1, qb = lane & ~3;
    // a group's 4 warps take alternate blocks (warp gw: blocks 4 ks + gw): the group reads adjacent memory at the same
    // time (contiguous quarters per warp streamed 5-10% slower)
    const int grp = warp >> 2, gw = warp & 3, pa = SP * gw, pb = 40, PST = 4 * SP;
    const int offA = (tg & 1) * 16 + (tg >> 1) * 8, offB = offA ^ 8;
    const int it0 = 4 * blockIdx.x + grp, istep = 4 * gridDim.x;

    // per pair P (chunks 2(P%2), +1 of block P/2), per row k: window (16 bytes), the lane's qhi words for both chunks
    // (unit (p, hh)), both chunks' meta (scales + d)
#define Q62_LOAD(W, QH, MT, L_, IT_, KS_)                                                               \
    {                                                                                                  \
        const int p0_ = pa + PST * (KS_);                                                               \
        _Pragma("unroll") for (int k = 0; k < 2; k++) {                                                \
            const uint8_t * w = m.w + (size_t) (16 * (IT_) + g + 8 * k) * m.rb;             \
            _Pragma("unroll") for (int i = 0; i < SP; i++) {                                           \
                const int P = p0_ + i;                                                                 \
                if (P < pb) {                                                                          \
                    const uint8_t * blk = w + (P / 2) * Q6B;                                           \
                    const int pc = 2 * (P % 2);                                                        \
                    const int4 v = ld4(blk + 64 * (P % 2) + 16 * tg);                                \
                    W[k][i] = make_uint4(v.x, v.y, v.z, v.w);                                          \
                    QH[k][i] = ld2(blk + 128 + 8 * (2 * pc + tg)); /* unit (pc + tg / 2, tg & 1) */            \
                    MT[k][i] = q6_meta(m, 16 * (IT_) + g + 8 * k, P / 2, pc, tg); \
                } else { W[k][i] = make_uint4(0, 0, 0, 0); QH[k][i] = make_uint2(0, 0); MT[k][i] = 0u; } \
            }                                                                                          \
        }                                                                                              \
    }
    // QH[k][i]: unit (pc + tg / 2, tg & 1) [x = hA, y = hB]; a lane's unit (p, hh) is in quad lane 2 (p - pc) + hh
#define Q62_STEP(W, QH, QH2, MT, IT_, KS_)                                                              \
    {                                                                                                  \
        const int p0_ = pa + PST * (KS_);                                                               \
        _Pragma("unroll") for (int i = 0; i < SP; i++) {                                               \
            const int P = p0_ + i;                                                                     \
            if (P < pb) {                                                                              \
            const int bk = P / 2;                                                                      \
            uint2 cA[2], cB[2];                                                                        \
            _Pragma("unroll") for (int k = 0; k < 2; k++) {                                            \
                const unsigned ox = __shfl_xor_sync(~0u, W[k][i].z, 2), oy = __shfl_xor_sync(~0u, W[k][i].w, 2); \
                const uint2 own = make_uint2(W[k][i].x, W[k][i].y), got = make_uint2(ox, oy);         \
                cA[k] = tg < 2 ? own : got;                                                            \
                cB[k] = tg < 2 ? got : own;                                                            \
            }                                                                                          \
            _Pragma("unroll") for (int hc = 0; hc < 2; hc++) {                                         \
                const int p = 2 * (P % 2) + hc, off = hc ? offB : offA, i0 = off % 16;                 \
                const uint2 r[2] = {hc ? cB[0] : cA[0], hc ? cB[1] : cA[1]};                           \
                int sc[2][2][2]; float dd[2]; /* [row][nib][half] */                                   \
                unsigned hw[2][2]; /* [row][nib]: this lane's unit (p, hh) high-bit words */           \
                _Pragma("unroll") for (int k = 0; k < 2; k++) {                                        \
                    const unsigned qb_ = lane & ~3;                                                    \
                    const uint4 m4 = make_uint4(__shfl_sync(~0u, MT[k][i], qb_), __shfl_sync(~0u, MT[k][i], qb_ + 1), __shfl_sync(~0u, MT[k][i], qb_ + 2), __shfl_sync(~0u, MT[k][i], qb_ + 3)); \
                    const unsigned scw = hc ? m4.z : m4.x, dw = hc ? m4.w : m4.y;                      \
                    dd[k] = q6_d(dw, bk);            \
                    sc[k][0][0] = (int) (int8_t) (scw); sc[k][0][1] = (int) (int8_t) (scw >> 8);       \
                    sc[k][1][0] = (int) (int8_t) (scw >> 16); sc[k][1][1] = (int) (int8_t) (scw >> 24); \
                    hw[k][0] = __shfl_sync(~0u, QH[k][i].x, qb + 2 * hc + hh);                         \
                    hw[k][1] = __shfl_sync(~0u, QH[k][i].y, qb + 2 * hc + hh);                         \
                }                                                                                      \
                _Pragma("unroll") for (int nib = 0; nib < 2; nib++) {                                  \
                    const int sj = 8 * bk + 2 * p + nib, sh = 4 * nib;                                 \
                    unsigned a[4]; /* rows g (a0, a2), g + 8 (a1, a3): bytes off .. off + 3 / off + 4 .. off + 7 */ \
                    _Pragma("unroll") for (int k = 0; k < 2; k++) {                                    \
                        const unsigned lo0 = (r[k].x >> sh) & M4, lo1 = (r[k].y >> sh) & M4;           \
                        const unsigned h0 = (hw[k][nib] >> (2 * (i0 / 4))) & 0x03030303u, h1 = (hw[k][nib] >> (2 * (i0 / 4 + 1))) & 0x03030303u; \
                        a[k] = __vsub4(lo0 | (h0 << 4), 0x20202020u);                                  \
                        a[2 + k] = __vsub4(lo1 | (h1 << 4), 0x20202020u);                              \
                    }                                                                                  \
                    const unsigned z0 = hh ? 0u : ~0u, z1 = ~z0;                                       \
                    _Pragma("unroll") for (int nt = 0; nt < NTL; nt++) {                               \
                        const int n = 8 * nt + g;                                                      \
                        const uint2 bf = n < T ? *reinterpret_cast<const uint2 *>(xw + n * XS5 + 8 * sj + off / 4) : make_uint2(0, 0); \
                        const float4 xs = sread<CS>(xdc, sgd, sj, 8 * nt + 2 * tg, T); \
                        int d0[4], d1[4];                                                              \
                        mma32(d0, a[0] & z0, a[1] & z0, a[2] & z0, a[3] & z0, bf);                \
                        mma32(d1, a[0] & z1, a[1] & z1, a[2] & z1, a[3] & z1, bf);                \
                        _Pragma("unroll") for (int c = 0; c < 4; c++) {                                \
                            const int rr = c >> 1;                                                     \
                            const int di = sc[rr][nib][0] * d0[c] + sc[rr][nib][1] * d1[c];            \
                            acc[nt][c] = fmaf(dd[rr] * ((c & 1) ? xs.z : xs.x), (float) di, acc[nt][c]); \
                        }                                                                              \
                    }                                                                                  \
                }                                                                                      \
            }                                                                                          \
            }                                                                                          \
        }                                                                                              \
        if ((KS_) == SPC - 1) {                                                                        \
            group_reduce<NTL>(acc, red, grp, gw, lane);                                               \
            if (gw == 0)                                                                               \
            _Pragma("unroll") for (int nt = 0; nt < NTL; nt++)                                         \
                _Pragma("unroll") for (int c = 0; c < 4; c++) {                                        \
                    const int t = 8 * nt + 2 * tg + (c & 1), row = g + 8 * (c >> 1);                   \
                    if (t < T) { float * o_ = out + (size_t) t * ld + 16 * (IT_) + row; *o_ = add ? *o_ + acc[nt][c] : acc[nt][c]; }                                                     \
                }                                                                                      \
            _Pragma("unroll") for (int nt = 0; nt < NTL; nt++)                                         \
                _Pragma("unroll") for (int c = 0; c < 4; c++) acc[nt][c] = 0.f;                        \
        }                                                                                              \
    }
    float acc[NTL][4] = {};
    uint4 WA[2][SP], WB[2][SP];
    uint2 QA[2][SP], QB[2][SP];
    unsigned MA[2][SP], MB[2][SP];
    int I0 = it0, K0 = 0;
    if (!sgx && I0 >= NI) return;
    if (I0 < NI) Q62_LOAD(WA, QA, MA, 0, I0, K0) // (in flight while the activations are staged)
    if (sgx) { stage_x5<CS>(sgx, sgd, T); if (I0 >= NI) return; }
    while (true) {
        int I1 = I0, K1 = K0 + 1;
        if (K1 == SPC) { K1 = 0; I1 += istep; }
        const bool v1 = I1 < NI;
        if (v1) Q62_LOAD(WB, QB, MB, 0, I1, K1)
        Q62_STEP(WA, QA, QA, MA, I0, K0)
        if (!v1) break;
        int I2 = I1, K2 = K1 + 1;
        if (K2 == SPC) { K2 = 0; I2 += istep; }
        const bool v2 = I2 < NI;
        if (v2) Q62_LOAD(WA, QA, MA, 0, I2, K2)
        Q62_STEP(WB, QB, QB, MB, I1, K1)
        if (!v2) break;
        I0 = I2; K0 = K2;
    }
#undef Q62_LOAD
#undef Q62_STEP
}


// ffn_down (N = D rows, K = FF), Q4_K or Q6_K (repacked 192-byte blocks + the meta stream): K in NCH chunks staged in shared memory
// from g_bx / g_bd; block = 4 groups x 1 item (16 rows) each; out [T][D]
template <int NTL_, int TYPE>
__device__ __forceinline__ void down(const Mat m, const unsigned * gx, const float2 * gd, float * out, int ld, int add, int T) {
    constexpr int SP = 2, NTL = NTL_, NIT = D / 16, SPC = (DCP / 4 + 1 + SP - 1) / SP, NS = NCH * SPC;
    constexpr int BS = TYPE == T_Q6K ? Q6B : 144;
    unsigned * xw = reinterpret_cast<unsigned *>(smem);
    float2 * xd = reinterpret_cast<float2 *>(xw + TS * DXS);
    float * red = reinterpret_cast<float *>(xd + DCB * 8 * TM);
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, g = lane >> 2, tg = lane & 3, hh = tg & 1, qb = lane & ~3;
    const int grp = warp >> 2, gw = warp & 3;
    const int offA = (tg & 1) * 16 + (tg >> 1) * 8, offB = offA ^ 8;
    const int pa = DCP * gw / 4, pb = DCP * (gw + 1) / 4;
    for (int item = blockIdx.x + gridDim.x * grp, round = 0; round * 4 * gridDim.x < NIT; round++, item += 4 * gridDim.x) {
        const bool active = item < NIT;
        const uint8_t * wrow = m.w + (size_t) (16 * (active ? item : 0) + g) * m.rb;
        float acc[NTL][4] = {};
        uint4 WA[2][SP], WB[2][SP];
    uint2 QA[2][SP], QB[2][SP];
        unsigned MA[2][SP], MB[2][SP];
#define BD_LOAD(W, QH, MT, S_)                                                                         \
        if (active && (S_) < NS) {                                                                     \
            const int ch_ = (S_) / SPC, p0_ = pa + SP * ((S_) % SPC);                                  \
            _Pragma("unroll") for (int k = 0; k < 2; k++) {                                            \
                const uint8_t * w = wrow + (size_t) 8 * k * m.rb + ch_ * DCB * BS;                     \
                _Pragma("unroll") for (int i = 0; i < SP; i++) {                                       \
                    const int P = p0_ + i;                                                             \
                    if (P < pb) {                                                                      \
                        const uint8_t * blk = w + (P / 2) * BS;                                        \
                        const int pc = 2 * (P % 2);                                                    \
                        if (TYPE == T_Q6K) {                                                           \
                            const int4 v = ld4(blk + 64 * (P % 2) + 16 * tg);                          \
                            W[k][i] = make_uint4(v.x, v.y, v.z, v.w);                                  \
                            QH[k][i] = ld2(blk + 128 + 8 * (2 * pc + tg));                             \
                            MT[k][i] = q6_meta(m, 16 * (active ? item : 0) + g + 8 * k, ch_ * DCB + P / 2, pc, tg);                               \
                        } else {                                                                       \
                            const int4 v = ld4(blk + 16 + 64 * (P % 2) + 16 * tg);                     \
                            W[k][i] = make_uint4(v.x, v.y, v.z, v.w);                                  \
                            MT[k][i] = ld1(blk + 4 * tg); /* header word tg */                        \
                        }                                                                              \
                    } else { W[k][i] = make_uint4(0, 0, 0, 0); QH[k][i] = make_uint2(0, 0); MT[k][i] = 0u; } \
                }                                                                                      \
            }                                                                                          \
        }
#define BD_STEP(W, QH, MT, S_)                                                                         \
        {                                                                                              \
            const int ch_ = (S_) / SPC, ks_ = (S_) % SPC, p0_ = pa + SP * ks_;                         \
            if (ks_ == 0) {                                                                            \
                __syncthreads();                                                                       \
                for (int i = threadIdx.x; i < min(T, TS) * DCB * 16; i += NT) {                                 \
                    const int t = i / (DCB * 16), q = i % (DCB * 16);                                  \
                    *reinterpret_cast<uint4 *>(xw + t * DXS + 4 * q) = __ldcg(reinterpret_cast<const uint4 *>(gx + (size_t) t * (FF / 4) + ch_ * DCB * 64) + q); \
                }                                                                                      \
                for (int i = threadIdx.x; i < DCB * 8 * TM; i += NT) xd[i] = __ldcg(gd + ch_ * DCB * 8 * TM + i); \
                __syncthreads();                                                                       \
            }                                                                                          \
            if (active)                                                                                \
            _Pragma("unroll") for (int i = 0; i < SP; i++) {                                           \
                const int P = p0_ + i;                                                                 \
                if (P < pb) {                                                                          \
                const int bk = P / 2;                                                                  \
                uint2 cA[2], cB[2];                                                                    \
                uint4 m4[2];                                                                           \
                _Pragma("unroll") for (int k = 0; k < 2; k++) {                                        \
                    const unsigned ox = __shfl_xor_sync(~0u, W[k][i].z, 2), oy = __shfl_xor_sync(~0u, W[k][i].w, 2); \
                    const uint2 own = make_uint2(W[k][i].x, W[k][i].y), got = make_uint2(ox, oy);     \
                    cA[k] = tg < 2 ? own : got;                                                        \
                    cB[k] = tg < 2 ? got : own;                                                        \
                    m4[k] = make_uint4(__shfl_sync(~0u, MT[k][i], qb), __shfl_sync(~0u, MT[k][i], qb + 1), __shfl_sync(~0u, MT[k][i], qb + 2), __shfl_sync(~0u, MT[k][i], qb + 3)); \
                }                                                                                      \
                _Pragma("unroll") for (int hc = 0; hc < 2; hc++) {                                     \
                    const int p = 2 * (P % 2) + hc, off = hc ? offB : offA, i0 = off % 16;             \
                    const uint2 r[2] = {hc ? cB[0] : cA[0], hc ? cB[1] : cA[1]};                       \
                    if (TYPE == T_Q6K) {                                                               \
                        int sc[2][2][2]; float dd[2]; unsigned hw[2][2];                               \
                        _Pragma("unroll") for (int k = 0; k < 2; k++) {                                \
                            const unsigned scw = hc ? m4[k].z : m4[k].x, dw = hc ? m4[k].w : m4[k].y;  \
                            dd[k] = q6_d(dw, ch_ * DCB + bk);    \
                            sc[k][0][0] = (int) (int8_t) (scw); sc[k][0][1] = (int) (int8_t) (scw >> 8); \
                            sc[k][1][0] = (int) (int8_t) (scw >> 16); sc[k][1][1] = (int) (int8_t) (scw >> 24); \
                            hw[k][0] = __shfl_sync(~0u, QH[k][i].x, qb + 2 * hc + hh);                 \
                            hw[k][1] = __shfl_sync(~0u, QH[k][i].y, qb + 2 * hc + hh);                 \
                        }                                                                              \
                        _Pragma("unroll") for (int nib = 0; nib < 2; nib++) {                          \
                            const int sj = 8 * bk + 2 * p + nib, sh = 4 * nib;                         \
                            unsigned a[4];                                                             \
                            _Pragma("unroll") for (int k = 0; k < 2; k++) {                            \
                                const unsigned lo0 = (r[k].x >> sh) & M4, lo1 = (r[k].y >> sh) & M4;   \
                                const unsigned h0 = (hw[k][nib] >> (2 * (i0 / 4))) & 0x03030303u, h1 = (hw[k][nib] >> (2 * (i0 / 4 + 1))) & 0x03030303u; \
                                a[k] = __vsub4(lo0 | (h0 << 4), 0x20202020u);                          \
                                a[2 + k] = __vsub4(lo1 | (h1 << 4), 0x20202020u);                      \
                            }                                                                          \
                            const unsigned z0 = hh ? 0u : ~0u, z1 = ~z0;                               \
                            _Pragma("unroll") for (int nt = 0; nt < NTL; nt++) {                       \
                                const int n = 8 * nt + g;                                              \
                                const uint2 bf = n < T ? xfrag(xw, DXS, TS, gx + ch_ * DCB * 64, FF / 4, n, 8 * sj + off / 4) : make_uint2(0, 0); \
                                const float4 xs = sfrag<TM>(xd, gd + ch_ * DCB * 8 * TM, sj, 8 * nt + 2 * tg, T); \
                                int d0[4], d1[4];                                                      \
                                mma32(d0, a[0] & z0, a[1] & z0, a[2] & z0, a[3] & z0, bf);             \
                                mma32(d1, a[0] & z1, a[1] & z1, a[2] & z1, a[3] & z1, bf);             \
                                _Pragma("unroll") for (int c = 0; c < 4; c++) {                        \
                                    const int rr = c >> 1;                                             \
                                    const int di = sc[rr][nib][0] * d0[c] + sc[rr][nib][1] * d1[c];    \
                                    acc[nt][c] = fmaf(dd[rr] * ((c & 1) ? xs.z : xs.x), (float) di, acc[nt][c]); \
                                }                                                                      \
                            }                                                                          \
                        }                                                                              \
                    } else {                                                                           \
                        float fA[2], gA[2], fB[2], gB[2];                                              \
                        _Pragma("unroll") for (int k = 0; k < 2; k++) q4k_scales(make_int4(m4[k].x, m4[k].y, m4[k].z, m4[k].w), p, fA[k], gA[k], fB[k], gB[k]); \
                        _Pragma("unroll") for (int nib = 0; nib < 2; nib++) {                          \
                            const int sj = 8 * bk + 2 * p + nib, sh = 4 * nib;                         \
                            const float f0 = nib ? fB[0] : fA[0], f1 = nib ? fB[1] : fA[1];            \
                            const float g0 = nib ? gB[0] : gA[0], g1 = nib ? gB[1] : gA[1];            \
                            const unsigned a0 = (r[0].x >> sh) & M4, a1 = (r[1].x >> sh) & M4, a2 = (r[0].y >> sh) & M4, a3 = (r[1].y >> sh) & M4; \
                            _Pragma("unroll") for (int nt = 0; nt < NTL; nt++) {                       \
                                const int n = 8 * nt + g;                                              \
                                const uint2 bf = n < T ? xfrag(xw, DXS, TS, gx + ch_ * DCB * 64, FF / 4, n, 8 * sj + off / 4) : make_uint2(0, 0); \
                                const float4 sc = sfrag<TM>(xd, gd + ch_ * DCB * 8 * TM, sj, 8 * nt + 2 * tg, T); \
                                int d[4];                                                              \
                                mma32(d, a0, a1, a2, a3, bf);                                          \
                                acc[nt][0] = fmaf(f0 * sc.x, (float) d[0], acc[nt][0]);                \
                                acc[nt][1] = fmaf(f0 * sc.z, (float) d[1], acc[nt][1]);                \
                                acc[nt][2] = fmaf(f1 * sc.x, (float) d[2], acc[nt][2]);                \
                                acc[nt][3] = fmaf(f1 * sc.z, (float) d[3], acc[nt][3]);                \
                                acc[nt][0] = fmaf(-g0, sc.y, acc[nt][0]);                              \
                                acc[nt][1] = fmaf(-g0, sc.w, acc[nt][1]);                              \
                                acc[nt][2] = fmaf(-g1, sc.y, acc[nt][2]);                              \
                                acc[nt][3] = fmaf(-g1, sc.w, acc[nt][3]);                              \
                            }                                                                          \
                        }                                                                              \
                    }                                                                                  \
                }                                                                                      \
                }                                                                                      \
            }                                                                                          \
        }
        BD_LOAD(WA, QA, MA, 0)
        for (int s0 = 0; s0 < NS; s0 += 2) {
            BD_LOAD(WB, QB, MB, s0 + 1)
            BD_STEP(WA, QA, MA, s0)
            BD_LOAD(WA, QA, MA, s0 + 2)
            BD_STEP(WB, QB, MB, s0 + 1)
        }
#undef BD_LOAD
#undef BD_STEP
        float * slot = red + warp * 16 * T;
#pragma unroll
        for (int nt = 0; nt < NTL; nt++)
#pragma unroll
            for (int c = 0; c < 4; c++) {
                const int t = 8 * nt + 2 * tg + (c & 1);
                if (t < T) slot[(g + 8 * (c >> 1)) * T + t] = acc[nt][c];
            }
        __syncthreads();
        if (active)
            for (int o = threadIdx.x & 127; o < 16 * T; o += 128) {
                const int row = o / T, t = o % T;
                float a = 0.f;
#pragma unroll
                for (int w4 = 0; w4 < 4; w4++) a += red[((grp * 4 + w4) * 16 + row) * T + t];
                { float * o_ = out + (size_t) t * ld + 16 * item + row; *o_ = add ? *o_ + a : a; }
            }
    }
}

template <int NTL_, int TYPE>
__device__ __forceinline__ void down2(const Mat m, const unsigned * gx, const float2 * gd, float * part, int T) {
    // grid (row blocks, NCH k-ranges); block = 4 groups of 4 warps; group grp: items rb IPB + grp + 4 j (16 rows each);
    // warp gw of a group takes the range's blocks gw, gw + 4, .. (the group reads 4 adjacent blocks of each row at the
    // same time: 8% faster than 2-warp groups on contiguous halves), steps of SP pairs double-buffered across items.
    // part [NCH][T][D]: this range's partial sums (k_down_reduce adds them, in order, to the residual)
    constexpr int SP = 2, NTL = NTL_, SPCG = (DCB + 3) / 4; // (SPCG: steps per item)
    constexpr int BS = TYPE == T_Q6K ? Q6B : 144;
    const int kr = blockIdx.y, IPB = (D / 16) / gridDim.x, NS = (IPB / 4) * SPCG;
    unsigned * xw = reinterpret_cast<unsigned *>(smem);
    float2 * xd = reinterpret_cast<float2 *>(xw + TM * DXS);
    float * red = reinterpret_cast<float *>(xd + DCB * 8 * TM); // group_reduce: [4 groups][2][32][NTL] float4
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, g = lane >> 2, tg = lane & 3, hh = tg & 1, qb = lane & ~3;
    const int grp = warp >> 2, gw = warp & 3;
    const int offA = (tg & 1) * 16 + (tg >> 1) * 8, offB = offA ^ 8;
    // a stage (SP = 2 pairs) is one block, whose Q4_K header is loaded once
#define D2_P0(ks) (SP * (4 * (ks) + gw))
#define D2_PB DCP
    {
        float acc[NTL][4] = {};
        uint4 WA[2][SP], WB[2][SP];
        uint2 QA[2][SP], QB[2][SP];
        unsigned MA[2][SP], MB[2][SP];
        uint4 HA4[2], HB4[2]; // Q4_K: the stage block's header, whole in every lane
        auto item_of = [&](int S_) { return blockIdx.x * IPB + grp + 4 * ((S_) / SPCG); };
#define BD2_LOAD(W, QH, MT, H4, S_)                                                                         \
        if ((S_) < NS) {                                                                               \
            const int p0_ = D2_P0((S_) % SPCG);                                                    \
            const uint8_t * wrow = m.w + (size_t) (16 * item_of(S_) + g) * m.rb + kr * DCB * BS;      \
            _Pragma("unroll") for (int k = 0; k < 2; k++) {                                            \
                const uint8_t * w = wrow + (size_t) 8 * k * m.rb;                                      \
                if (TYPE != T_Q6K) { const int4 h_ = p0_ < D2_PB ? ld4(w + (p0_ / 2) * BS) : make_int4(0, 0, 0, 0); H4[k] = make_uint4(h_.x, h_.y, h_.z, h_.w); } \
                _Pragma("unroll") for (int i = 0; i < SP; i++) {                                       \
                    const int P = p0_ + i;                                                             \
                    if (P < D2_PB) {                                                                      \
                        const uint8_t * blk = w + (P / 2) * BS;                                        \
                        const int pc = 2 * (P % 2);                                                    \
                        if (TYPE == T_Q6K) {                                                           \
                            const int4 v = ld4(blk + 64 * (P % 2) + 16 * tg);                          \
                            W[k][i] = make_uint4(v.x, v.y, v.z, v.w);                                  \
                            QH[k][i] = ld2(blk + 128 + 8 * (2 * pc + tg));                             \
                            MT[k][i] = q6_meta(m, 16 * item_of(S_) + g + 8 * k, kr * DCB + P / 2, pc, tg);                               \
                        } else {                                                                       \
                            const int4 v = ld4(blk + 16 + 64 * (P % 2) + 16 * tg);                     \
                            W[k][i] = make_uint4(v.x, v.y, v.z, v.w);                                  \
                            MT[k][i] = 0u;                                                             \
                        }                                                                              \
                    } else { W[k][i] = make_uint4(0, 0, 0, 0); QH[k][i] = make_uint2(0, 0); MT[k][i] = 0u; } \
                }                                                                                      \
            }                                                                                          \
        }
#define BD2_STEP(W, QH, MT, H4, S_)                                                                         \
        {                                                                                              \
            const int ks_ = (S_) % SPCG, p0_ = D2_P0(ks_);                                           \
            _Pragma("unroll") for (int i = 0; i < SP; i++) {                                           \
                const int P = p0_ + i;                                                                 \
                if (P < D2_PB) {                                                                          \
                const int bk = P / 2;                                                                  \
                uint2 cA[2], cB[2];                                                                    \
                uint4 m4[2];                                                                           \
                _Pragma("unroll") for (int k = 0; k < 2; k++) {                                        \
                    const unsigned ox = __shfl_xor_sync(~0u, W[k][i].z, 2), oy = __shfl_xor_sync(~0u, W[k][i].w, 2); \
                    const uint2 own = make_uint2(W[k][i].x, W[k][i].y), got = make_uint2(ox, oy);     \
                    cA[k] = tg < 2 ? own : got;                                                        \
                    cB[k] = tg < 2 ? got : own;                                                        \
                    if (TYPE == T_Q6K) m4[k] = make_uint4(__shfl_sync(~0u, MT[k][i], qb), __shfl_sync(~0u, MT[k][i], qb + 1), __shfl_sync(~0u, MT[k][i], qb + 2), __shfl_sync(~0u, MT[k][i], qb + 3)); \
                    else m4[k] = H4[k];                                                                \
                }                                                                                      \
                _Pragma("unroll") for (int hc = 0; hc < 2; hc++) {                                     \
                    const int p = 2 * (P % 2) + hc, off = hc ? offB : offA, i0 = off % 16;             \
                    const uint2 r[2] = {hc ? cB[0] : cA[0], hc ? cB[1] : cA[1]};                       \
                    if (TYPE == T_Q6K) {                                                               \
                        int sc[2][2][2]; float dd[2]; unsigned hw[2][2];                               \
                        _Pragma("unroll") for (int k = 0; k < 2; k++) {                                \
                            const unsigned scw = hc ? m4[k].z : m4[k].x, dw = hc ? m4[k].w : m4[k].y;  \
                            dd[k] = q6_d(dw, kr * DCB + bk);    \
                            sc[k][0][0] = (int) (int8_t) (scw); sc[k][0][1] = (int) (int8_t) (scw >> 8); \
                            sc[k][1][0] = (int) (int8_t) (scw >> 16); sc[k][1][1] = (int) (int8_t) (scw >> 24); \
                            hw[k][0] = __shfl_sync(~0u, QH[k][i].x, qb + 2 * hc + hh);                 \
                            hw[k][1] = __shfl_sync(~0u, QH[k][i].y, qb + 2 * hc + hh);                 \
                        }                                                                              \
                        _Pragma("unroll") for (int nib = 0; nib < 2; nib++) {                          \
                            const int sj = 8 * bk + 2 * p + nib, sh = 4 * nib;                         \
                            unsigned a[4];                                                             \
                            _Pragma("unroll") for (int k = 0; k < 2; k++) {                            \
                                const unsigned lo0 = (r[k].x >> sh) & M4, lo1 = (r[k].y >> sh) & M4;   \
                                const unsigned h0 = (hw[k][nib] >> (2 * (i0 / 4))) & 0x03030303u, h1 = (hw[k][nib] >> (2 * (i0 / 4 + 1))) & 0x03030303u; \
                                a[k] = __vsub4(lo0 | (h0 << 4), 0x20202020u);                          \
                                a[2 + k] = __vsub4(lo1 | (h1 << 4), 0x20202020u);                      \
                            }                                                                          \
                            const unsigned z0 = hh ? 0u : ~0u, z1 = ~z0;                               \
                            _Pragma("unroll") for (int nt = 0; nt < NTL; nt++) {                       \
                                const int n = 8 * nt + g;                                              \
                                const uint2 bf = n < T ? *reinterpret_cast<const uint2 *>(xw + n * DXS + 8 * sj + off / 4) : make_uint2(0, 0); \
                                const float4 xs = sfrag<TM>(xd, gd + kr * DCB * 8 * TM, sj, 8 * nt + 2 * tg, T); \
                                int d0[4], d1[4];                                                      \
                                mma32(d0, a[0] & z0, a[1] & z0, a[2] & z0, a[3] & z0, bf);             \
                                mma32(d1, a[0] & z1, a[1] & z1, a[2] & z1, a[3] & z1, bf);             \
                                _Pragma("unroll") for (int c = 0; c < 4; c++) {                        \
                                    const int rr = c >> 1;                                             \
                                    const int di = sc[rr][nib][0] * d0[c] + sc[rr][nib][1] * d1[c];    \
                                    acc[nt][c] = fmaf(dd[rr] * ((c & 1) ? xs.z : xs.x), (float) di, acc[nt][c]); \
                                }                                                                      \
                            }                                                                          \
                        }                                                                              \
                    } else {                                                                           \
                        float fA[2], gA[2], fB[2], gB[2];                                              \
                        _Pragma("unroll") for (int k = 0; k < 2; k++) q4k_scales(make_int4(m4[k].x, m4[k].y, m4[k].z, m4[k].w), p, fA[k], gA[k], fB[k], gB[k]); \
                        _Pragma("unroll") for (int nib = 0; nib < 2; nib++) {                          \
                            const int sj = 8 * bk + 2 * p + nib, sh = 4 * nib;                         \
                            const float f0 = nib ? fB[0] : fA[0], f1 = nib ? fB[1] : fA[1];            \
                            const float g0 = nib ? gB[0] : gA[0], g1 = nib ? gB[1] : gA[1];            \
                            const unsigned a0 = (r[0].x >> sh) & M4, a1 = (r[1].x >> sh) & M4, a2 = (r[0].y >> sh) & M4, a3 = (r[1].y >> sh) & M4; \
                            _Pragma("unroll") for (int nt = 0; nt < NTL; nt++) {                       \
                                const int n = 8 * nt + g;                                              \
                                const uint2 bf = n < T ? *reinterpret_cast<const uint2 *>(xw + n * DXS + 8 * sj + off / 4) : make_uint2(0, 0); \
                                const float4 sc = sfrag<TM>(xd, gd + kr * DCB * 8 * TM, sj, 8 * nt + 2 * tg, T); \
                                int d[4];                                                              \
                                mma32(d, a0, a1, a2, a3, bf);                                          \
                                acc[nt][0] = fmaf(f0 * sc.x, (float) d[0], acc[nt][0]);                \
                                acc[nt][1] = fmaf(f0 * sc.z, (float) d[1], acc[nt][1]);                \
                                acc[nt][2] = fmaf(f1 * sc.x, (float) d[2], acc[nt][2]);                \
                                acc[nt][3] = fmaf(f1 * sc.z, (float) d[3], acc[nt][3]);                \
                                acc[nt][0] = fmaf(-g0, sc.y, acc[nt][0]);                              \
                                acc[nt][1] = fmaf(-g0, sc.w, acc[nt][1]);                              \
                                acc[nt][2] = fmaf(-g1, sc.y, acc[nt][2]);                              \
                                acc[nt][3] = fmaf(-g1, sc.w, acc[nt][3]);                              \
                            }                                                                          \
                        }                                                                              \
                    }                                                                                  \
                }                                                                                      \
                }                                                                                      \
            }                                                                                          \
            if (ks_ == SPCG - 1) { /* item done: the group's sums to warp 0, which stores the partial */ \
                const int it_ = item_of(S_);                                                           \
                group_reduce<NTL>(acc, red, grp, gw, lane);                                           \
                if (gw == 0)                                                                           \
                _Pragma("unroll") for (int nt = 0; nt < NTL; nt++)                                     \
                    _Pragma("unroll") for (int c = 0; c < 4; c++) {                                    \
                        const int t = 8 * nt + 2 * tg + (c & 1), row = g + 8 * (c >> 1);              \
                        if (t < T) part[((size_t) kr * T + t) * D + 16 * it_ + row] = acc[nt][c];      \
                    }                                                                                  \
                _Pragma("unroll") for (int nt = 0; nt < NTL; nt++)                                     \
                    _Pragma("unroll") for (int c = 0; c < 4; c++) acc[nt][c] = 0.f;                    \
            }                                                                                          \
        }
        BD2_LOAD(WA, QA, MA, HA4, 0) // (in flight while the block's activation range is staged)
        for (int i = threadIdx.x; i < T * DCB * 16; i += NT) { // (all T rows: the k range fits)
            const int t = i / (DCB * 16), q = i % (DCB * 16);
            *reinterpret_cast<uint4 *>(xw + t * DXS + 4 * q) = __ldcg(reinterpret_cast<const uint4 *>(gx + (size_t) t * (FF / 4) + kr * DCB * 64) + q);
        }
        for (int i = threadIdx.x; i < DCB * 8 * TM; i += NT) xd[i] = __ldcg(gd + kr * DCB * 8 * TM + i);
        __syncthreads();
        for (int s0 = 0; s0 < NS; s0 += 2) {
            BD2_LOAD(WB, QB, MB, HB4, s0 + 1)
            BD2_STEP(WA, QA, MA, HA4, s0)
            BD2_LOAD(WA, QA, MA, HA4, s0 + 2)
            BD2_STEP(WB, QB, MB, HB4, s0 + 1)
        }
#undef BD2_LOAD
#undef BD2_STEP
#undef D2_P0
#undef D2_PB
    }
}

// x[t][n] += sum over the k-ranges of part[kr][t][n] (fixed order: deterministic)
__global__ void k_down_reduce(const float * part, float * x, int T) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= T * D) return;
    float a = 0.f;
#pragma unroll
    for (int kr = 0; kr < NCH; kr++) a += __ldcg(part + (size_t) kr * T * D + i);
    x[i] += a;
}

// activations from global memory (per-32 int8, K = D: gx [TM][D / 4] words, gd [D / 32][TM]) into shared memory
template <bool CS, bool SG> __device__ void stage_x5(const unsigned * gx, const float2 * gd, int T) {
    unsigned * xw = reinterpret_cast<unsigned *>(smem);
    unsigned * xdc = xw + (CS ? TM : TS) * XS5;
    for (int i = threadIdx.x; i < T * NJ5 * 2; i += NT) { // 16-byte pieces
        const int t = i / (NJ5 * 2), q = i % (NJ5 * 2);
        *reinterpret_cast<uint4 *>(xw + t * XS5 + 4 * q) = __ldcg(reinterpret_cast<const uint4 *>(gx + (size_t) t * (D / 4)) + q);
    }
    if constexpr (CS) { if (!SG) for (int i = threadIdx.x; i < NJ5 * TM; i += NT) xdc[i] = pack_scale(__ldcg(gd + i)); }
    else for (int i = threadIdx.x; i < NJ5 * TM; i += NT) reinterpret_cast<float2 *>(xdc)[i] = __ldcg(gd + i);
    __syncthreads();
}

// per-32 int8 quantisation of rows t < T: v = src[t][k] * (w ? w[k] : 1) * (rs ? rs[t] : 1), or silu(src) * src2
// (GLU, src2 != null) -> gx [TM][K / 4] (natural order), gd [K / 32][TM] = (scale, scale * sum of the int8 values).
// grid (K / 256, T), 256 threads: warp = one 32-value sub-block, lane = value
__global__ void __launch_bounds__(256) k_quant32(const float * src, const float * src2, int ld, const float * w, const float * rs, int K,
                                                 unsigned * gx, float2 * gd) {
    const int t = blockIdx.y, lane = threadIdx.x & 31, j = blockIdx.x * 8 + (threadIdx.x >> 5), k = 32 * j + lane;
    float v = __ldcg(src + (size_t) t * ld + k);
    if (src2) v = v / (1.f + __expf(-v)) * __ldcg(src2 + (size_t) t * ld + k);
    if (w) v *= __ldg(w + k);
    if (rs) v *= __ldcg(rs + t);
    float amax = fabsf(v);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(~0u, amax, o));
    const float dx = amax / 127.f;
    const int q = amax == 0.f ? 0 : __float2int_rn(v / dx);
    int sum = q;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) sum += __shfl_xor_sync(~0u, sum, o);
    // bytes of lanes 4i .. 4i + 3 -> word i
    const unsigned b = (unsigned) (q & 0xff);
    const unsigned b1 = __shfl_down_sync(~0u, b, 1), b2 = __shfl_down_sync(~0u, b, 2), b3 = __shfl_down_sync(~0u, b, 3);
    if ((lane & 3) == 0) gx[(size_t) t * (K / 4) + 8 * j + lane / 4] = b | (b1 << 8) | (b2 << 16) | (b3 << 24);
    if (lane == 0) gd[(size_t) j * TM + t] = make_float2(dx, dx * (float) sum);
}

// RMS norm (weight w) of the rows of x [T][D] quantised per 32 (as k_quant32 with K = D) in one pass; rs[t] = the
// row's 1 / rms (the delta-net alpha / beta projection uses it)
__global__ void __launch_bounds__(256) k_norm_quant32(const float * x, const float * w, float * rs, unsigned * gx, float2 * gd) {
    // grid (D / 256, T): each block sums its row's squares (the row is L2-resident) and quantises its 256 values
    __shared__ float red[8];
    const int t = blockIdx.y, lane = threadIdx.x & 31, warp = threadIdx.x >> 5, j = blockIdx.x * 8 + warp, k = 32 * j + lane;
    const float * xr = x + (size_t) t * D;
    float ss = 0.f;
#pragma unroll
    for (int i = 0; i < D / 1024; i++) {
        const float4 v = __ldcg(reinterpret_cast<const float4 *>(xr) + threadIdx.x + 256 * i);
        ss += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
    }
    const float v = __ldcg(xr + k);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(~0u, ss, o);
    if (lane == 0) red[warp] = ss;
    __syncthreads();
    float tot = 0.f;
#pragma unroll
    for (int i = 0; i < 8; i++) tot += red[i];
    const float r = rsqrtf(tot / (float) D + EPS);
    if (threadIdx.x == 0 && blockIdx.x == 0) rs[t] = r;
    const float y = v * r * __ldg(w + k);
    float amax = fabsf(y);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(~0u, amax, o));
    const float dx = amax / 127.f;
    const int q = amax == 0.f ? 0 : __float2int_rn(y / dx);
    int sum = q;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) sum += __shfl_xor_sync(~0u, sum, o);
    const unsigned b = (unsigned) (q & 0xff);
    const unsigned b1 = __shfl_down_sync(~0u, b, 1), b2 = __shfl_down_sync(~0u, b, 2), b3 = __shfl_down_sync(~0u, b, 3);
    if ((lane & 3) == 0) gx[(size_t) t * (D / 4) + 8 * j + lane / 4] = b | (b1 << 8) | (b2 << 16) | (b3 << 24);
    if (lane == 0) gd[(size_t) j * TM + t] = make_float2(dx, dx * (float) sum);
}

template <int NTL, bool SG = false>
__global__ void __launch_bounds__(NT, 1) k_q4mx(const MV mv, const unsigned * gx, const float2 * gd, int T) {
    q4m<NTL, NTL == 3 && SG>(mv, T, gx, gd);
}
// one Q4_K / Q6_K matrix (K = D) on activations gx / gd: out[t * ld + n] (+)= (W x_t)_n
template <int NTL>
__global__ void __launch_bounds__(NT, 1) k_q4x(const Mat m, const unsigned * gx, const float2 * gd, float * out, int ld, int add, int T) {
    q4<NTL>(m, out, ld, add, T, gx, gd);
}
template <int NTL>
__global__ void __launch_bounds__(NT, 1) k_q6x(const Mat m, const unsigned * gx, const float2 * gd, float * out, int ld, int add, int T) {
    q6<NTL>(m, out, ld, add, T, gx, gd);
}
// ffn_down split over k-ranges (grid (D / 16 / IPB, NCH)), partials then k_down_reduce
template <int NTL, int TYPE>
__global__ void __launch_bounds__(NT, 1) k_down2(const Mat m, const unsigned * gx, const float2 * gd, float * part, int T) {
    down2<NTL, TYPE>(m, gx, gd, part, T);
}
// ffn_down (K = FF) on gx / gd (K = FF layout)
template <int NTL, int TYPE>
__global__ void __launch_bounds__(NT, 1) k_downx(const Mat m, const unsigned * gx, const float2 * gd, float * out, int ld, int add, int T) {
    down<NTL, TYPE>(m, gx, gd, out, ld, add, T);
}

// activations of row length K (per-32 int8: gx [TM][K / 4] words, gd [K / 32][TM]) into shared memory (stride K / 4 + 8)
__device__ void stage_x(const unsigned * gx, const float2 * gd, int T, int K, int TR, int gs, int xsc) {
    const int XS = K / 4 + 8, NJ = K / 32;
    unsigned * xw = reinterpret_cast<unsigned *>(smem);
    float2 * xd = reinterpret_cast<float2 *>(xw + TR * XS);
    for (int i = threadIdx.x; i < min(T, TR) * K / 16; i += NT) {
        const int t = i / (K / 16), q = i % (K / 16);
        *reinterpret_cast<uint4 *>(xw + t * XS + 4 * q) = __ldcg(reinterpret_cast<const uint4 *>(gx + (size_t) t * gs) + q);
    }
    for (int i = threadIdx.x; i < NJ * xsc; i += NT) xd[i] = __ldcg(gd + i / xsc * TM + i % xsc);
    __syncthreads();
}
template <int NTL, int TYPE>
__global__ void __launch_bounds__(NT, 1) k_qkx(const Mat m, const unsigned * gx, const float2 * gd, float * out, int ld, int add, int T,
                                               int gs = 0) {
    qk<NTL, TYPE>(m, out, ld, add, T, gx, gd, gs);
}
// K = VD projections past TS rows: both K halves in one launch (blockIdx.y = half; every row staged) -> part [2][T][N],
// then k_kh_reduce (rows TS .. read from global memory instead cost 74 us at 16 rows)
template <int TYPE>
__global__ void __launch_bounds__(NT, 1) k_qkx_kh(const Mat m, const unsigned * gx, const float2 * gd, float * part, int T) {
    const int h = blockIdx.y, KH = m.K / 2;
    Mat mh = m;
    mh.K = KH;
    mh.w = m.w + (size_t) h * (KH / 256) * (TYPE == T_Q5K ? 176 : 144);
    qk<2, TYPE>(mh, part + (size_t) h * T * m.N, m.N, 0, T, gx + h * KH / 4, gd + (size_t) h * (KH / 32) * TM, m.K / 4);
}
// x[t][n] += part[0][t][n] + part[1][t][n] (N = D)
__global__ void k_kh_reduce(const float * part, float * x, int T) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < T * D) x[i] += __ldcg(part + i) + __ldcg(part + (size_t) T * D + i);
}
// fp32 inputs of the K = VD projections. Delta net: gated rms-norm per 128-wide head, o * rs * snorm * silu(z);
// attention: y * sigmoid(gate)
__global__ void __launch_bounds__(256) k_gate_dn(const float * y2, const float * z, const float * snorm, float * out) {
    const int t = blockIdx.x, lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    for (int hh = warp; hh < NVH; hh += 8) {
        const float * o = y2 + (size_t) t * VD + hh * 128;
        float v[4], ss = 0.f;
#pragma unroll
        for (int i = 0; i < 4; i++) { v[i] = __ldcg(o + lane + 32 * i); ss += v[i] * v[i]; }
        const float rs = rsqrtf(wsum(ss) / 128.0f + EPS);
#pragma unroll
        for (int i = 0; i < 4; i++) {
            const int c = lane + 32 * i;
            out[(size_t) t * VD + hh * 128 + c] = v[i] * rs * __ldg(snorm + c) * silu(__ldcg(z + (size_t) t * VD + hh * 128 + c));
        }
    }
}
__global__ void __launch_bounds__(256) k_gate_attn(const float * y0, const float * y2, float * out) {
    const int t = blockIdx.x;
    for (int e = threadIdx.x; e < VD; e += 256) {
        const int h = e / HD, d = e % HD;
        const float gate = __ldcg(y0 + (size_t) t * QG + h * 512 + 256 + d);
        out[(size_t) t * VD + e] = __ldcg(y2 + (size_t) t * VD + e) / (1.0f + expf(-gate));
    }
}
// per-32 quantisation of a warp's 32 values v (lane = value) as sub-block j of row t (K = VD layout, as k_quant32)
__device__ __forceinline__ void quant32_vd(float v, int t, int j, int lane, unsigned * gx, float2 * gd) {
    float amax = fabsf(v);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(~0u, amax, o));
    const float dx = amax / 127.f;
    const int q = amax == 0.f ? 0 : __float2int_rn(v / dx);
    int sum = q;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) sum += __shfl_xor_sync(~0u, sum, o);
    const unsigned b = (unsigned) (q & 0xff);
    const unsigned b1 = __shfl_down_sync(~0u, b, 1), b2 = __shfl_down_sync(~0u, b, 2), b3 = __shfl_down_sync(~0u, b, 3);
    if ((lane & 3) == 0) gx[(size_t) t * (VD / 4) + 8 * j + lane / 4] = b | (b1 << 8) | (b2 << 16) | (b3 << 24);
    if (lane == 0) gd[(size_t) j * TM + t] = make_float2(dx, dx * (float) sum);
}
// k_gate_dn then the per-32 quantisation in one pass: grid (NVH / 8, T), warp = head
__global__ void __launch_bounds__(256) k_gate_dn_q(const float * y2, const float * z, const float * snorm, unsigned * gx, float2 * gd) {
    const int t = blockIdx.y, lane = threadIdx.x & 31, hh = blockIdx.x * 8 + (threadIdx.x >> 5);
    const float * o = y2 + (size_t) t * VD + hh * 128;
    float v[4], ss = 0.f;
#pragma unroll
    for (int i = 0; i < 4; i++) { v[i] = __ldcg(o + lane + 32 * i); ss += v[i] * v[i]; }
    const float rs = rsqrtf(wsum(ss) / 128.0f + EPS);
#pragma unroll
    for (int i = 0; i < 4; i++) {
        const int c = lane + 32 * i;
        quant32_vd(v[i] * rs * __ldg(snorm + c) * silu(__ldcg(z + (size_t) t * VD + hh * 128 + c)), t, hh * 4 + i, lane, gx, gd);
    }
}
// k_gate_attn then the per-32 quantisation: grid (VD / 256, T), warp = sub-block
__global__ void __launch_bounds__(256) k_gate_attn_q(const float * y0, const float * y2, unsigned * gx, float2 * gd) {
    const int t = blockIdx.y, lane = threadIdx.x & 31, j = blockIdx.x * 8 + (threadIdx.x >> 5), e = 32 * j + lane;
    const int h = e / HD, d = e % HD;
    const float gate = __ldcg(y0 + (size_t) t * QG + h * 512 + 256 + d);
    quant32_vd(__ldcg(y2 + (size_t) t * VD + e) / (1.0f + expf(-gate)), t, j, lane, gx, gd);
}
// ---- MTP eh_proj on int8 dp4a: the MTP input concat(enorm(embed(tok)), hnorm(h_prev)) quantised per 32 as two K = D
// halves (ex / ed: the embedding half, hx / hd: the hidden half), then eh = [E | H] as two K = D GEMVs over the Q8_0
// weights as loaded (int8 rows w8 [N][2 D], permuted per 256 for the megakernel, + per-32 fp16 scales ws [N][2 D / 32]).
__device__ __forceinline__ void quant32_d(float v, int t, int j, int lane, unsigned * gx, float2 * gd) {
    float amax = fabsf(v);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(~0u, amax, o));
    const float dx = amax / 127.f;
    const int q = amax == 0.f ? 0 : __float2int_rn(v / dx);
    int sum = q;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) sum += __shfl_xor_sync(~0u, sum, o);
    const unsigned b = (unsigned) (q & 0xff);
    const unsigned b1 = __shfl_down_sync(~0u, b, 1), b2 = __shfl_down_sync(~0u, b, 2), b3 = __shfl_down_sync(~0u, b, 3);
    if ((lane & 3) == 0) gx[(size_t) t * (D / 4) + 8 * j + lane / 4] = b | (b1 << 8) | (b2 << 16) | (b3 << 24);
    if (lane == 0) gd[(size_t) j * TM + t] = make_float2(dx, dx * (float) sum);
}
__global__ void __launch_bounds__(256) k_mtp_in_q(const int * tok, const float * h, const float * enw, const float * hnw,
                                                   unsigned * ex, float2 * ed, unsigned * hx, float2 * hd) {
    __shared__ float se[D];
    __shared__ float red[8];
    const int t = blockIdx.x, lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const uint8_t * row = cW.embd + (size_t) tok[t] * (D / 256 * 144);
    const float * hp = h + (size_t) t * D;
    float ae = 0.f, ah = 0.f;
    for (int e = threadIdx.x; e < D; e += 256) {
        const float v = dequant_q4k_ldg(row, e), hv = __ldcg(hp + e);
        se[e] = v;
        ae += v * v;
        ah += hv * hv;
    }
    const float rse = rsqrtf(block_sum256(ae, red) / (float) D + EPS);
    const float rsh = rsqrtf(block_sum256(ah, red) / (float) D + EPS);
    for (int j = warp; j < 2 * D / 32; j += 8) {
        const int jj = j % (D / 32), e = 32 * jj + lane;
        if (j < D / 32) quant32_d(se[e] * rse * __ldg(enw + e), t, jj, lane, ex, ed);
        else quant32_d(__ldcg(hp + e) * rsh * __ldg(hnw + e), t, jj, lane, hx, hd);
    }
}
// out[t][n] (+)= sum_k w8[n][koff + k] ws x_t[k] (K = D, x per 32 in gx / gd as stage_x5): warp = row, lane = 16 bytes
__global__ void __launch_bounds__(NT, 1) k_q8dp(const int8_t * w8, const __half * ws, int koff, const unsigned * gx, const float2 * gd,
                                               float * out, int ld, int add, int N, int T) {
    stage_x5(gx, gd, T);
    const unsigned * xw = reinterpret_cast<const unsigned *>(smem);
    const unsigned * xdc = xw + TM * XS5; // (stage_x5: all rows, compact scales)
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nw = NT / 32;
    for (int r = blockIdx.x * nw + warp; r < N; r += gridDim.x * nw) {
        const int8_t * w = w8 + (size_t) r * 2 * D + koff;
        const __half * s = ws + (size_t) r * (2 * D / 32) + koff / 32;
        int4 wv[D / 512];
#pragma unroll
        for (int it = 0; it < D / 512; it++) wv[it] = ld4(w + 512 * it + 16 * lane);
        float acc[TM];
#pragma unroll
        for (int t = 0; t < TM; t++) acc[t] = 0.f;
#pragma unroll
        for (int it = 0; it < D / 512; it++) {
            // stored bytes o .. o + 15 are original values orig .. orig + 15 (the loader's per-256 permutation: stored
            // 32-byte chunk u = 2 p + hh holds values 64 p + 16 hh .. + 15, then 64 p + 32 + 16 hh .. + 15)
            const int o = 512 * it + 16 * lane, sb = o & 255, u = sb >> 5, half = (sb >> 4) & 1;
            const int orig = (o & ~255) + 64 * (u >> 1) + 16 * (u & 1) + 32 * half, j = orig / 32;
            const float sc = __half2float(s[j]);
#pragma unroll
            for (int t = 0; t < TM; t++)
                if (t < T) {
                    const uint4 xv = *reinterpret_cast<const uint4 *>(xw + t * XS5 + orig / 4);
                    int d = __dp4a(wv[it].x, (int) xv.x, 0);
                    d = __dp4a(wv[it].y, (int) xv.y, d);
                    d = __dp4a(wv[it].z, (int) xv.z, d);
                    d = __dp4a(wv[it].w, (int) xv.w, d);
                    acc[t] = fmaf(sc * unpack_scale(xdc[j * TM + t]).x, (float) d, acc[t]);
                }
        }
#pragma unroll
        for (int t = 0; t < TM; t++)
            if (t < T) {
                const float v = wsum(acc[t]);
                if (lane == 0) { float * o_ = out + (size_t) t * ld + r; *o_ = add ? *o_ + v : v; }
            }
    }
}

// alpha / beta for S rows: out[s][j] = rs[s] * (Wab_j . x_s) (the norm weight is folded into Wab); block = row j
__global__ void __launch_bounds__(256) k_ab_ms(const float * Wab, const float * x, const float * rs, int S, float * out) {
    __shared__ float red[8][TM];
    const int j = blockIdx.x, lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    float acc[TM];
#pragma unroll
    for (int s = 0; s < TM; s++) acc[s] = 0.f;
    for (int k = threadIdx.x * 4; k < D; k += 1024) {
        const float4 w = __ldg(reinterpret_cast<const float4 *>(Wab + (size_t) j * D + k));
#pragma unroll
        for (int s = 0; s < TM; s++)
            if (s < S) {
                const float4 v = __ldcg(reinterpret_cast<const float4 *>(x + (size_t) s * D + k));
                acc[s] += w.x * v.x + w.y * v.y + w.z * v.z + w.w * v.w;
            }
    }
#pragma unroll
    for (int s = 0; s < TM; s++) {
        const float v = wsum(acc[s]);
        if (lane == 0) red[warp][s] = v;
    }
    __syncthreads();
    if (threadIdx.x < S) {
        float v = 0.f;
#pragma unroll
        for (int w = 0; w < 8; w++) v += red[w][threadIdx.x];
        out[threadIdx.x * 96 + j] = v * __ldcg(rs + threadIdx.x);
    }
}

// ---- multi-sequence single-token delta net (decode): grid (NVH value heads, S sequences), 512 threads.
// Per (head, sequence): conv1d (4 taps, from that sequence's conv ring) + silu of the head's q / k / v channels,
// l2-norm of q (x 1 / sqrt(128)) and k, g = exp(softplus(alpha + dt) A), beta = sigmoid(beta pre-activation),
// then the gated delta rule on the sequence's state (fp32, thread = state column `col`, rows [32 r, 32 r + 32),
// as the megakernel's dn_core): kv = S k, S <- g S + k (beta (v - g kv)), o = S q -> y2 row s. The raw qkv
// values of this position go to the conv ring (q / k channels by the head-group's first block).
struct MsDn {
    float * conv[TM];
    float * dstate[TM];
    int pos[TM];
};
__global__ void __launch_bounds__(512) k_dn_ms(const MsDn P, int dl, const float * convw, const float * dt, const float * A,
                                               const float * y0, const float * ab, int cap, float * y2, const float * ab_flat) {
    constexpr int QS = 4 * 36;
    __shared__ __align__(16) float s_q[QS], s_k[QS], s_v[128], s_raw[384], s_gb[2];
    const int h = blockIdx.x, s = blockIdx.y, kh = h % NKH;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, col = tid >> 2, r = tid & 3;
    const int pos = P.pos[s];
    float * Sg = P.dstate[s] + ((size_t) dl * NVH + h) * 128 * 128 + col * 128 + 32 * r;
    float4 S[8];
#pragma unroll
    for (int i = 0; i < 8; i++) S[i] = __ldcg(reinterpret_cast<const float4 *>(Sg) + i);
    float * ring = P.conv[s] + (size_t) dl * CONV_RING * QKV;
    if (tid < 384) {
        const int ch = tid < 128 ? kh * 128 + tid : (tid < 256 ? 2048 + kh * 128 + (tid - 128) : 4096 + h * 128 + (tid - 256));
        const float4 w = __ldg(reinterpret_cast<const float4 *>(convw + ch * 4));
        float raw[4];
#pragma unroll
        for (int j = 0; j < 3; j++) {
            const int p = pos - 3 + j;
            raw[j] = p >= 0 ? __ldcg(ring + (size_t) (p % CONV_RING) * QKV + ch) : 0.f;
        }
        raw[3] = __ldcg(y0 + (size_t) s * QKV + ch);
        s_raw[tid] = silu(w.x * raw[0] + w.y * raw[1] + w.z * raw[2] + w.w * raw[3]);
        if (tid >= 256 || h < NKH) ring[(size_t) (pos % CONV_RING) * QKV + ch] = raw[3];
    } else if (tid == 384) {
        float al = 0.f, be = 0.f;
        if (ab_flat) {
            al = __ldcg(ab_flat + s * 96 + h);
            be = __ldcg(ab_flat + s * 96 + NVH + h);
        } else
            for (int c = 0; c < AB_SPLITS; c++) {
                al += __ldcg(ab + ((size_t) c * cap + s) * 96 + h);
                be += __ldcg(ab + ((size_t) c * cap + s) * 96 + NVH + h);
            }
        float sp = al + __ldg(dt + h);
        sp = sp > 20.0f ? sp : logf(1.0f + expf(sp));
        s_gb[0] = expf(sp * __ldg(A + h));
        s_gb[1] = 1.0f / (1.0f + expf(-be));
    }
    __syncthreads();
    if (warp < 2) { // warp 0: q, warp 1: k
        const float4 v = *reinterpret_cast<const float4 *>(s_raw + warp * 128 + lane * 4);
        const float ss = wsum(v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w);
        float sc = rsqrtf(fmaxf(ss, EPS * EPS));
        if (!warp) sc *= 0.08838834764831845f; // 1/sqrt(128)
        float * dst = (warp ? s_k : s_q) + (lane >> 3) * 36 + (lane & 7) * 4;
        *reinterpret_cast<float4 *>(dst) = make_float4(v.x * sc, v.y * sc, v.z * sc, v.w * sc);
    } else if (tid >= 128 && tid < 256) s_v[tid - 128] = s_raw[256 + tid - 128];
    __syncthreads();
    const float g = s_gb[0], beta = s_gb[1];
    const float4 * kq = reinterpret_cast<const float4 *>(s_k + r * 36);
    float kv = 0.f;
#pragma unroll
    for (int i = 0; i < 8; i++) {
        const float4 k4 = kq[i];
        kv += S[i].x * k4.x + S[i].y * k4.y + S[i].z * k4.z + S[i].w * k4.w;
    }
    kv += __shfl_xor_sync(0xffffffffu, kv, 1);
    kv += __shfl_xor_sync(0xffffffffu, kv, 2);
    const float dlt = (s_v[col] - g * kv) * beta;
    const float4 * qq = reinterpret_cast<const float4 *>(s_q + r * 36);
    float o = 0.f;
#pragma unroll
    for (int i = 0; i < 8; i++) {
        const float4 k4 = kq[i], q4 = qq[i];
        S[i].x = g * S[i].x + k4.x * dlt;
        S[i].y = g * S[i].y + k4.y * dlt;
        S[i].z = g * S[i].z + k4.z * dlt;
        S[i].w = g * S[i].w + k4.w * dlt;
        o += S[i].x * q4.x + S[i].y * q4.y + S[i].z * q4.z + S[i].w * q4.w;
    }
    o += __shfl_xor_sync(0xffffffffu, o, 1);
    o += __shfl_xor_sync(0xffffffffu, o, 2);
    if (r == 0) y2[(size_t) s * VD + h * 128 + col] = o;
#pragma unroll
    for (int i = 0; i < 8; i++) reinterpret_cast<float4 *>(Sg)[i] = S[i];
}

// ---- delta net for speculative steps: per sequence, rows r0 .. r0 + nt - 1 (positions pos ..), nt <= 3, preceded by a
// replay of the previous step's rep_n accepted drafts (their k / v / g / beta, saved in `rep` [layer][2][RPL]). The state is
// committed after the first row (the accepted token); a second row (the draft) is computed from the uncommitted
// state, and its update inputs are saved for the next step's replay. Raw qkv of every row goes to the conv ring
// (slots by position: a rejected draft's slot is rewritten when its position comes again).
constexpr int RPL = NKH * 128 + NVH * 128 + 2 * NVH; // replay entry per layer: k (per k head) | v | g | beta
constexpr int DMAX = 4; // most drafts a slot verifies per step (replay entries per layer)
struct MsDn2 {
    float * conv[TM];
    float * dstate[TM];
    float * rep[TM];  // replay entry read this step
    float * repw[TM]; // ... written this step (the other buffer: a k head is read by 3 value-head blocks)
    int pos[TM], r0[TM], nt[TM], rep_n[TM];
};
__global__ void __launch_bounds__(512) k_dn_ms2(const MsDn2 P, int dl, const float * convw, const float * dt, const float * A,
                                                const float * y0, const float * abf, float * y2) {
    constexpr int QS = 4 * 36;
    __shared__ __align__(16) float s_q[QS], s_k[QS], s_v[128], s_raw[384], s_gb[2];
    const int h = blockIdx.x, s = blockIdx.y, kh = h % NKH;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, col = tid >> 2, r = tid & 3;
    const int pos = P.pos[s], r0 = P.r0[s], nt = P.nt[s];
    float * Sg = P.dstate[s] + ((size_t) dl * NVH + h) * 128 * 128 + col * 128 + 32 * r;
    const float * rp0 = P.rep[s] + (size_t) dl * DMAX * RPL;   // [2 entries][RPL] per layer
    float * rpw0 = P.repw[s] + (size_t) dl * DMAX * RPL;
    float4 S[8];
#pragma unroll
    for (int i = 0; i < 8; i++) S[i] = __ldcg(reinterpret_cast<const float4 *>(Sg) + i);
    float * ring = P.conv[s] + (size_t) dl * CONV_RING * QKV;
    const int ch = tid < 128 ? kh * 128 + tid : (tid < 256 ? 2048 + kh * 128 + (tid - 128) : 4096 + h * 128 + (tid - 256));
    for (int e = 0; e < P.rep_n[s]; e++) { // the previous step's accepted drafts, in order
        const float * rp = rp0 + e * RPL;
        for (int i = tid; i < 128; i += 512) {
            s_k[(i >> 5) * 36 + (i & 31)] = __ldcg(rp + kh * 128 + i);
            s_v[i] = __ldcg(rp + NKH * 128 + h * 128 + i);
        }
        if (tid == 0) { s_gb[0] = __ldcg(rp + NKH * 128 + NVH * 128 + h); s_gb[1] = __ldcg(rp + NKH * 128 + NVH * 128 + NVH + h); }
        __syncthreads();
        const float g = s_gb[0], beta = s_gb[1];
        const float4 * kq = reinterpret_cast<const float4 *>(s_k + r * 36);
        float kv = 0.f;
#pragma unroll
        for (int i = 0; i < 8; i++) { const float4 k4 = kq[i]; kv += S[i].x * k4.x + S[i].y * k4.y + S[i].z * k4.z + S[i].w * k4.w; }
        kv += __shfl_xor_sync(0xffffffffu, kv, 1);
        kv += __shfl_xor_sync(0xffffffffu, kv, 2);
        const float dlt = (s_v[col] - g * kv) * beta;
#pragma unroll
        for (int i = 0; i < 8; i++) {
            const float4 k4 = kq[i];
            S[i].x = g * S[i].x + k4.x * dlt; S[i].y = g * S[i].y + k4.y * dlt;
            S[i].z = g * S[i].z + k4.z * dlt; S[i].w = g * S[i].w + k4.w * dlt;
        }
        __syncthreads();
    }
    if (nt == 0) { // a flush: the replayed drafts become committed state
#pragma unroll
        for (int i = 0; i < 8; i++) reinterpret_cast<float4 *>(Sg)[i] = S[i];
    }
    for (int t = 0; t < nt; t++) {
        const int p = pos + t, row = r0 + t;
        if (tid < 384) {
            const float4 w = __ldg(reinterpret_cast<const float4 *>(convw + ch * 4));
            float raw[4];
#pragma unroll
            for (int j = 0; j < 4; j++) {
                const int q = p - 3 + j;
                raw[j] = q >= pos ? __ldcg(y0 + (size_t) (r0 + q - pos) * QKV + ch) : q >= 0 ? __ldcg(ring + (size_t) (q % CONV_RING) * QKV + ch) : 0.f;
            }
            s_raw[tid] = silu(w.x * raw[0] + w.y * raw[1] + w.z * raw[2] + w.w * raw[3]);
        } else if (tid == 384) {
            const float al = __ldcg(abf + row * 96 + h), be = __ldcg(abf + row * 96 + NVH + h);
            float sp = al + __ldg(dt + h);
            sp = sp > 20.0f ? sp : logf(1.0f + expf(sp));
            s_gb[0] = expf(sp * __ldg(A + h));
            s_gb[1] = 1.0f / (1.0f + expf(-be));
        }
        __syncthreads();
        if (warp < 2) {
            const float4 v = *reinterpret_cast<const float4 *>(s_raw + warp * 128 + lane * 4);
            const float ss = wsum(v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w);
            float sc = rsqrtf(fmaxf(ss, EPS * EPS));
            if (!warp) sc *= 0.08838834764831845f;
            float * dst = (warp ? s_k : s_q) + (lane >> 3) * 36 + (lane & 7) * 4;
            *reinterpret_cast<float4 *>(dst) = make_float4(v.x * sc, v.y * sc, v.z * sc, v.w * sc);
        } else if (tid >= 128 && tid < 256) s_v[tid - 128] = s_raw[256 + tid - 128];
        __syncthreads();
        if (t >= 1) { // a draft row: save its update inputs for a replay
            float * rpw = rpw0 + (t - 1) * RPL;
            if (h < NKH && tid < 128) rpw[kh * 128 + tid] = s_k[(tid >> 5) * 36 + (tid & 31)];
            if (tid < 128) rpw[NKH * 128 + h * 128 + tid] = s_v[tid];
            if (tid == 0) { rpw[NKH * 128 + NVH * 128 + h] = s_gb[0]; rpw[NKH * 128 + NVH * 128 + NVH + h] = s_gb[1]; }
        }
        const float g = s_gb[0], beta = s_gb[1];
        const float4 * kq = reinterpret_cast<const float4 *>(s_k + r * 36);
        float kv = 0.f;
#pragma unroll
        for (int i = 0; i < 8; i++) { const float4 k4 = kq[i]; kv += S[i].x * k4.x + S[i].y * k4.y + S[i].z * k4.z + S[i].w * k4.w; }
        kv += __shfl_xor_sync(0xffffffffu, kv, 1);
        kv += __shfl_xor_sync(0xffffffffu, kv, 2);
        const float dlt = (s_v[col] - g * kv) * beta;
        const float4 * qq = reinterpret_cast<const float4 *>(s_q + r * 36);
        float o = 0.f;
#pragma unroll
        for (int i = 0; i < 8; i++) {
            const float4 k4 = kq[i], q4 = qq[i];
            S[i].x = g * S[i].x + k4.x * dlt; S[i].y = g * S[i].y + k4.y * dlt;
            S[i].z = g * S[i].z + k4.z * dlt; S[i].w = g * S[i].w + k4.w * dlt;
            o += S[i].x * q4.x + S[i].y * q4.y + S[i].z * q4.z + S[i].w * q4.w;
        }
        o += __shfl_xor_sync(0xffffffffu, o, 1);
        o += __shfl_xor_sync(0xffffffffu, o, 2);
        if (r == 0) y2[(size_t) row * VD + h * 128 + col] = o;
        if (t == 0)
#pragma unroll
            for (int i = 0; i < 8; i++) reinterpret_cast<float4 *>(Sg)[i] = S[i];
        __syncthreads(); // s_q / s_k / s_v / s_raw reuse
    }
    // raw qkv of the rows -> conv ring (after every block read the ring for this step's rows: slots differ)
    if (tid < 384 && (tid >= 256 || h < NKH))
        for (int t = 0; t < nt; t++) ring[(size_t) ((pos + t) % CONV_RING) * QKV + ch] = __ldcg(y0 + (size_t) (r0 + t) * QKV + ch);
}
__global__ void __launch_bounds__(256, 3) k_dn_ms3(const MsDn2 P, int dl, const float * convw, const float * dt, const float * A,
                                                const float * y0, const float * abf, float * y2) {
    constexpr int QS = 4 * 36;
    __shared__ __align__(16) float s_q[QS], s_k[QS], s_v[128], s_raw[384], s_gb[2];
    // two blocks per (value head, sequence): block half `hf` owns state columns [64 hf, 64 hf + 64) (256 threads)
    const int h = blockIdx.x >> 1, hf = blockIdx.x & 1, s = blockIdx.y, kh = h % NKH;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, col = 64 * hf + (tid >> 2), r = tid & 3;
    const int pos = P.pos[s], r0 = P.r0[s], nt = P.nt[s];
    float * Sg = P.dstate[s] + ((size_t) dl * NVH + h) * 128 * 128 + col * 128 + 32 * r;
    const float * rp0 = P.rep[s] + (size_t) dl * DMAX * RPL;   // [2 entries][RPL] per layer
    float * rpw0 = P.repw[s] + (size_t) dl * DMAX * RPL;
    float4 S[8];
#pragma unroll
    for (int i = 0; i < 8; i++) S[i] = __ldcg(reinterpret_cast<const float4 *>(Sg) + i);
    float * ring = P.conv[s] + (size_t) dl * CONV_RING * QKV;
    // channels: q (128) | k (128) | this half's 64 v; thread tid covers c0 = tid and (tid < 64) c1 = 256 + tid
    auto chan = [&](int c) { return c < 128 ? kh * 128 + c : (c < 256 ? 2048 + kh * 128 + (c - 128) : 4096 + h * 128 + 64 * hf + (c - 256)); };
    for (int e = 0; e < P.rep_n[s]; e++) { // the previous step's accepted drafts, in order
        const float * rp = rp0 + e * RPL;
        for (int i = tid; i < 128; i += 256) {
            s_k[(i >> 5) * 36 + (i & 31)] = __ldcg(rp + kh * 128 + i);
            s_v[i] = __ldcg(rp + NKH * 128 + h * 128 + i);
        }
        if (tid == 0) { s_gb[0] = __ldcg(rp + NKH * 128 + NVH * 128 + h); s_gb[1] = __ldcg(rp + NKH * 128 + NVH * 128 + NVH + h); }
        __syncthreads();
        const float g = s_gb[0], beta = s_gb[1];
        const float4 * kq = reinterpret_cast<const float4 *>(s_k + r * 36);
        float kv = 0.f;
#pragma unroll
        for (int i = 0; i < 8; i++) { const float4 k4 = kq[i]; kv += S[i].x * k4.x + S[i].y * k4.y + S[i].z * k4.z + S[i].w * k4.w; }
        kv += __shfl_xor_sync(0xffffffffu, kv, 1);
        kv += __shfl_xor_sync(0xffffffffu, kv, 2);
        const float dlt = (s_v[col] - g * kv) * beta;
#pragma unroll
        for (int i = 0; i < 8; i++) {
            const float4 k4 = kq[i];
            S[i].x = g * S[i].x + k4.x * dlt; S[i].y = g * S[i].y + k4.y * dlt;
            S[i].z = g * S[i].z + k4.z * dlt; S[i].w = g * S[i].w + k4.w * dlt;
        }
        __syncthreads();
    }
    if (nt == 0) { // a flush: the replayed drafts become committed state
#pragma unroll
        for (int i = 0; i < 8; i++) reinterpret_cast<float4 *>(Sg)[i] = S[i];
    }
    for (int t = 0; t < nt; t++) {
        const int p = pos + t, row = r0 + t;
        for (int c = tid; c < 320; c += 256) {
            const int ch = chan(c);
            const float4 w = __ldg(reinterpret_cast<const float4 *>(convw + ch * 4));
            float raw[4];
#pragma unroll
            for (int j = 0; j < 4; j++) {
                const int q = p - 3 + j;
                raw[j] = q >= pos ? __ldcg(y0 + (size_t) (r0 + q - pos) * QKV + ch) : q >= 0 ? __ldcg(ring + (size_t) (q % CONV_RING) * QKV + ch) : 0.f;
            }
            s_raw[c] = silu(w.x * raw[0] + w.y * raw[1] + w.z * raw[2] + w.w * raw[3]);
        }
        if (tid == 255) {
            const float al = __ldcg(abf + row * 96 + h), be = __ldcg(abf + row * 96 + NVH + h);
            float sp = al + __ldg(dt + h);
            sp = sp > 20.0f ? sp : logf(1.0f + expf(sp));
            s_gb[0] = expf(sp * __ldg(A + h));
            s_gb[1] = 1.0f / (1.0f + expf(-be));
        }
        __syncthreads();
        if (warp < 2) {
            const float4 v = *reinterpret_cast<const float4 *>(s_raw + warp * 128 + lane * 4);
            const float ss = wsum(v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w);
            float sc = rsqrtf(fmaxf(ss, EPS * EPS));
            if (!warp) sc *= 0.08838834764831845f;
            float * dst = (warp ? s_k : s_q) + (lane >> 3) * 36 + (lane & 7) * 4;
            *reinterpret_cast<float4 *>(dst) = make_float4(v.x * sc, v.y * sc, v.z * sc, v.w * sc);
        } else if (tid >= 128 && tid < 192) s_v[64 * hf + tid - 128] = s_raw[256 + tid - 128];
        __syncthreads();
        if (t >= 1) { // a draft row: save its update inputs for a replay
            float * rpw = rpw0 + (t - 1) * RPL;
            if (h < NKH && hf == 0 && tid < 128) rpw[kh * 128 + tid] = s_k[(tid >> 5) * 36 + (tid & 31)];
            if (tid < 64) rpw[NKH * 128 + h * 128 + 64 * hf + tid] = s_v[64 * hf + tid];
            if (tid == 0 && hf == 0) { rpw[NKH * 128 + NVH * 128 + h] = s_gb[0]; rpw[NKH * 128 + NVH * 128 + NVH + h] = s_gb[1]; }
        }
        const float g = s_gb[0], beta = s_gb[1];
        const float4 * kq = reinterpret_cast<const float4 *>(s_k + r * 36);
        float kv = 0.f;
#pragma unroll
        for (int i = 0; i < 8; i++) { const float4 k4 = kq[i]; kv += S[i].x * k4.x + S[i].y * k4.y + S[i].z * k4.z + S[i].w * k4.w; }
        kv += __shfl_xor_sync(0xffffffffu, kv, 1);
        kv += __shfl_xor_sync(0xffffffffu, kv, 2);
        const float dlt = (s_v[col] - g * kv) * beta;
        const float4 * qq = reinterpret_cast<const float4 *>(s_q + r * 36);
        float o = 0.f;
#pragma unroll
        for (int i = 0; i < 8; i++) {
            const float4 k4 = kq[i], q4 = qq[i];
            S[i].x = g * S[i].x + k4.x * dlt; S[i].y = g * S[i].y + k4.y * dlt;
            S[i].z = g * S[i].z + k4.z * dlt; S[i].w = g * S[i].w + k4.w * dlt;
            o += S[i].x * q4.x + S[i].y * q4.y + S[i].z * q4.z + S[i].w * q4.w;
        }
        o += __shfl_xor_sync(0xffffffffu, o, 1);
        o += __shfl_xor_sync(0xffffffffu, o, 2);
        if (r == 0) y2[(size_t) row * VD + h * 128 + col] = o;
        if (t == 0)
#pragma unroll
            for (int i = 0; i < 8; i++) reinterpret_cast<float4 *>(Sg)[i] = S[i];
        __syncthreads(); // s_q / s_k / s_v / s_raw reuse
    }
    // raw qkv of the rows -> conv ring (after every block read the ring for this step's rows: slots differ)
    for (int c = tid; c < 320; c += 256)
        if (c >= 256 || (h < NKH && hf == 0)) {
            const int ch = chan(c);
            for (int t = 0; t < nt; t++) ring[(size_t) ((pos + t) % CONV_RING) * QKV + ch] = __ldcg(y0 + (size_t) (r0 + t) * QKV + ch);
        }
}
// k_dn_ms3 with every input staged up front: the replay entries, all rows' conv windows and gates are loaded while
// the state loads are in flight, then the updates (replays, rows) run from shared memory without barriers.
// Rows per sequence <= DN4_R.
constexpr int DN4_R = 1 + DMAX;
__global__ void __launch_bounds__(256, 3) k_dn_ms4(const MsDn2 P, int dl, const float * convw, const float * dt, const float * A,
                                                   const float * y0, const float * Wab, const float * x, const float * rs, float * y2) {
    // thread (col, r) holds state rows 16 i + 4 r .. + 3 (i < 8) of column col: a quad's float4 accesses are 64
    // contiguous bytes (whole sectors), q / k are plain 128-float vectors
    __shared__ __align__(16) float s_q[DN4_R][128], s_k[DN4_R + DMAX][128], s_raw[DN4_R][384], s_rv[DMAX][64], s_gb[DN4_R + DMAX][2];
    __shared__ float s_ab[8][2 * DN4_R]; // per warp: the rows' alpha / beta pre-activation partial sums
    const int h = blockIdx.x >> 1, hf = blockIdx.x & 1, s = blockIdx.y, kh = h % NKH;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, col = 64 * hf + (tid >> 2), r = tid & 3;
    const int pos = P.pos[s], r0 = P.r0[s], nt = P.nt[s], ne = P.rep_n[s];
    float4 * Sg = reinterpret_cast<float4 *>(P.dstate[s] + ((size_t) dl * NVH + h) * 128 * 128 + col * 128) + r; // [4 i]
    const float * rp0 = P.rep[s] + (size_t) dl * DMAX * RPL;
    float * rpw0 = P.repw[s] + (size_t) dl * DMAX * RPL;
    float4 S[8];
#pragma unroll
    for (int i = 0; i < 8; i++) S[i] = __ldcg(Sg + 4 * i);
    float * ring = P.conv[s] + (size_t) dl * CONV_RING * QKV;
    auto chan = [&](int c) { return c < 128 ? kh * 128 + c : (c < 256 ? 2048 + kh * 128 + (c - 128) : 4096 + h * 128 + 64 * hf + (c - 256)); };
    // replay entries: k -> s_k[DN4_R + e], this half's v -> s_rv[e], (g, beta) -> s_gb[DN4_R + e]
    for (int e = 0; e < ne; e++) {
        const float * rp = rp0 + e * RPL;
        if (tid < 128) s_k[DN4_R + e][tid] = __ldcg(rp + kh * 128 + tid);
        else if (tid < 192) s_rv[e][tid - 128] = __ldcg(rp + NKH * 128 + h * 128 + 64 * hf + tid - 128);
        else if (tid == 192) { s_gb[DN4_R + e][0] = __ldcg(rp + NKH * 128 + NVH * 128 + h); s_gb[DN4_R + e][1] = __ldcg(rp + NKH * 128 + NVH * 128 + NVH + h); }
    }
    // the rows' conv outputs (q | k | this half's v) and gates
    for (int i = tid; i < nt * 320; i += 256) {
        const int t = i / 320, c = i % 320, p = pos + t, ch = chan(c);
        const float4 w = __ldg(reinterpret_cast<const float4 *>(convw + ch * 4));
        float raw[4];
#pragma unroll
        for (int j = 0; j < 4; j++) {
            const int q = p - 3 + j;
            raw[j] = q >= pos ? __ldcg(y0 + (size_t) (r0 + q - pos) * QKV + ch) : q >= 0 ? __ldcg(ring + (size_t) (q % CONV_RING) * QKV + ch) : 0.f;
        }
        s_raw[t][c] = silu(w.x * raw[0] + w.y * raw[1] + w.z * raw[2] + w.w * raw[3]);
    }
    { // this head's alpha / beta pre-activations of the rows: rs[row] * (Wab x[row]) (attn_norm folded into Wab)
        float pa[DN4_R] = {}, pb[DN4_R] = {};
#pragma unroll
        for (int k = tid * 4; k < D; k += 1024) {
            const float4 wa = __ldg(reinterpret_cast<const float4 *>(Wab + (size_t) h * D + k));
            const float4 wb = __ldg(reinterpret_cast<const float4 *>(Wab + (size_t) (NVH + h) * D + k));
#pragma unroll
            for (int t = 0; t < DN4_R; t++)
                if (t < nt) {
                    const float4 v = __ldcg(reinterpret_cast<const float4 *>(x + (size_t) (r0 + t) * D + k));
                    pa[t] += wa.x * v.x + wa.y * v.y + wa.z * v.z + wa.w * v.w;
                    pb[t] += wb.x * v.x + wb.y * v.y + wb.z * v.z + wb.w * v.w;
                }
        }
#pragma unroll
        for (int t = 0; t < DN4_R; t++) {
            const float va = wsum(pa[t]), vb = wsum(pb[t]);
            if (lane == 0) { s_ab[warp][2 * t] = va; s_ab[warp][2 * t + 1] = vb; }
        }
    }
    __syncthreads();
    if (tid >= 224 && tid - 224 < nt) {
        const int t = tid - 224, row = r0 + t;
        float al = 0.f, be = 0.f;
#pragma unroll
        for (int w8 = 0; w8 < 8; w8++) { al += s_ab[w8][2 * t]; be += s_ab[w8][2 * t + 1]; }
        al *= __ldcg(rs + row);
        be *= __ldcg(rs + row);
        float sp = al + __ldg(dt + h);
        sp = sp > 20.0f ? sp : logf(1.0f + expf(sp));
        s_gb[t][0] = expf(sp * __ldg(A + h));
        s_gb[t][1] = 1.0f / (1.0f + expf(-be));
    }
    __syncthreads();
    for (int task = warp; task < 2 * nt; task += 8) { // l2 norms of q (scaled) and k, per row
        const int t = task >> 1, isk = task & 1;
        const float4 v = *reinterpret_cast<const float4 *>(&s_raw[t][isk * 128 + lane * 4]);
        const float ss = wsum(v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w);
        float sc = rsqrtf(fmaxf(ss, EPS * EPS));
        if (!isk) sc *= 0.08838834764831845f;
        float * dst = (isk ? s_k[t] : s_q[t]) + lane * 4;
        *reinterpret_cast<float4 *>(dst) = make_float4(v.x * sc, v.y * sc, v.z * sc, v.w * sc);
    }
    __syncthreads();
    // draft rows: their update inputs for a replay next step
    for (int t = 1; t < nt; t++) {
        float * rpw = rpw0 + (t - 1) * RPL;
        if (h < NKH && hf == 0 && tid < 128) rpw[kh * 128 + tid] = s_k[t][tid];
        if (tid < 64) rpw[NKH * 128 + h * 128 + 64 * hf + tid] = s_raw[t][256 + tid];
        if (tid == 0 && hf == 0) { rpw[NKH * 128 + NVH * 128 + h] = s_gb[t][0]; rpw[NKH * 128 + NVH * 128 + NVH + h] = s_gb[t][1]; }
    }
    auto update = [&](const float * kk, float v, float g, float beta, const float * qv, float & o) {
        const float4 * kq = reinterpret_cast<const float4 *>(kk) + r; // [4 i]
        float kv = 0.f;
#pragma unroll
        for (int i = 0; i < 8; i++) { const float4 k4 = kq[4 * i]; kv += S[i].x * k4.x + S[i].y * k4.y + S[i].z * k4.z + S[i].w * k4.w; }
        kv += __shfl_xor_sync(0xffffffffu, kv, 1);
        kv += __shfl_xor_sync(0xffffffffu, kv, 2);
        const float dlt = (v - g * kv) * beta;
        const float4 * qq = reinterpret_cast<const float4 *>(qv ? qv : kk) + r;
        o = 0.f;
#pragma unroll
        for (int i = 0; i < 8; i++) {
            const float4 k4 = kq[4 * i];
            S[i].x = g * S[i].x + k4.x * dlt; S[i].y = g * S[i].y + k4.y * dlt;
            S[i].z = g * S[i].z + k4.z * dlt; S[i].w = g * S[i].w + k4.w * dlt;
            if (qv) { const float4 q4 = qq[4 * i]; o += S[i].x * q4.x + S[i].y * q4.y + S[i].z * q4.z + S[i].w * q4.w; }
        }
    };
    float o;
    for (int e = 0; e < ne; e++) update(s_k[DN4_R + e], s_rv[e][col - 64 * hf], s_gb[DN4_R + e][0], s_gb[DN4_R + e][1], nullptr, o);
    if (nt == 0) { // a flush: the replayed drafts become committed state
#pragma unroll
        for (int i = 0; i < 8; i++) Sg[4 * i] = S[i];
    }
    for (int t = 0; t < nt; t++) {
        update(s_k[t], s_raw[t][256 + col - 64 * hf], s_gb[t][0], s_gb[t][1], s_q[t], o);
        o += __shfl_xor_sync(0xffffffffu, o, 1);
        o += __shfl_xor_sync(0xffffffffu, o, 2);
        if (r == 0) y2[(size_t) (r0 + t) * VD + h * 128 + col] = o;
        if (t == 0)
#pragma unroll
            for (int i = 0; i < 8; i++) Sg[4 * i] = S[i];
    }
    // raw qkv of the rows -> conv ring (positions pos .. pos + nt - 1; this step read only pos - 3 .. pos - 1)
    for (int c = tid; c < 320; c += 256)
        if (c >= 256 || (h < NKH && hf == 0)) {
            const int ch = chan(c);
            for (int t = 0; t < nt; t++) ring[(size_t) ((pos + t) % CONV_RING) * QKV + ch] = __ldcg(y0 + (size_t) (r0 + t) * QKV + ch);
        }
}
// rms-norm of rows: out[r] = x[r] * rsqrt(mean(x[r]^2) + eps) * w (the MTP layer's own hidden for a chained draft)
__global__ void __launch_bounds__(256) k_rmsnorm(const float * x, const float * w, float * out) {
    __shared__ float red[8];
    const int r = blockIdx.x, lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    float ss = 0.f;
    for (int e = threadIdx.x; e < D; e += 256) { const float v = __ldcg(x + (size_t) r * D + e); ss += v * v; }
    ss = wsum(ss);
    if (lane == 0) red[warp] = ss;
    __syncthreads();
    float t = lane < 8 ? red[lane] : 0.f;
    t = wsum(t);
    const float rs = rsqrtf(t / (float) D + EPS);
    for (int e = threadIdx.x; e < D; e += 256) out[(size_t) r * D + e] = __ldcg(x + (size_t) r * D + e) * rs * __ldg(w + e);
}
// top 32 (logit, id) of each row, descending: thread keeps its strided top 8, the block sorts the 2048 survivors
__global__ void __launch_bounds__(256) k_topk32(const float * y, int ld, int N, const int * ids, float2 * out) {
    __shared__ float sv[2048];
    __shared__ int si[2048];
    const int r = blockIdx.x, tid = threadIdx.x;
    float v[8];
    int ix[8];
#pragma unroll
    for (int j = 0; j < 8; j++) { v[j] = -INFINITY; ix[j] = -1; }
    for (int i = tid; i < N; i += 256) {
        const float x = __ldcg(y + (size_t) r * ld + i);
        if (x > v[7]) {
            int j = 7;
            while (j > 0 && v[j - 1] < x) { v[j] = v[j - 1]; ix[j] = ix[j - 1]; j--; }
            v[j] = x; ix[j] = i;
        }
    }
#pragma unroll
    for (int j = 0; j < 8; j++) { sv[tid * 8 + j] = v[j]; si[tid * 8 + j] = ix[j]; }
    __syncthreads();
    // bitonic sort, descending
    for (int k = 2; k <= 2048; k <<= 1)
        for (int jj = k >> 1; jj > 0; jj >>= 1) {
            for (int i = tid; i < 2048; i += 256) {
                const int l = i ^ jj;
                if (l > i) {
                    const bool desc = (i & k) == 0;
                    const float a = sv[i], b = sv[l];
                    if (desc ? a < b : a > b) {
                        sv[i] = b; sv[l] = a;
                        const int t = si[i]; si[i] = si[l]; si[l] = t;
                    }
                }
            }
            __syncthreads();
        }
    if (tid < 32) {
        const int id = si[tid];
        out[r * 32 + tid] = make_float2(sv[tid], __int_as_float(id < 0 ? -1 : (ids ? ids[id] : id)));
    }
}

// argmax of rows: y [R][ld], first N entries; ids: row index -> token id (null: identity)
__global__ void __launch_bounds__(256) k_argmax(const float * y, int ld, int N, const int * ids, int * out) {
    __shared__ float bv[8];
    __shared__ int bi[8];
    const int t = blockIdx.x, lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const float * l = y + (size_t) t * ld;
    float b = -INFINITY;
    int ix = 0;
    for (int i = threadIdx.x; i < N; i += 256) { const float v = __ldcg(l + i); if (v > b) { b = v; ix = i; } }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        const float ob = __shfl_xor_sync(0xffffffffu, b, o);
        const int oi = __shfl_xor_sync(0xffffffffu, ix, o);
        if (ob > b || (ob == b && oi < ix)) { b = ob; ix = oi; }
    }
    if (lane == 0) { bv[warp] = b; bi[warp] = ix; }
    __syncthreads();
    if (threadIdx.x == 0) {
        for (int w = 1; w < 8; w++) if (bv[w] > b || (bv[w] == b && bi[w] < ix)) { b = bv[w]; ix = bi[w]; }
        out[t] = ids ? ids[ix] : ix;
    }
}

// k_argmax / k_topk32 split over HB chunks of each row (one block per row left 70 SMs idle for a 248k-vocab row):
// pass 1 per (chunk, row), pass 2 merges the row's chunk results (same results: lowest index among equal maxima)
constexpr int HB = 32;
__global__ void __launch_bounds__(256) k_argmax_p1(const float * y, int ld, int N, float2 * part) {
    __shared__ float bv[8];
    __shared__ int bi[8];
    const int t = blockIdx.y, c = blockIdx.x, lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int len = (N + HB - 1) / HB, i0 = c * len, i1 = min(N, i0 + len);
    const float * l = y + (size_t) t * ld;
    float b = -INFINITY;
    int ix = 0x7fffffff;
    for (int i = i0 + threadIdx.x; i < i1; i += 256) { const float v = __ldcg(l + i); if (v > b) { b = v; ix = i; } }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        const float ob = __shfl_xor_sync(0xffffffffu, b, o);
        const int oi = __shfl_xor_sync(0xffffffffu, ix, o);
        if (ob > b || (ob == b && oi < ix)) { b = ob; ix = oi; }
    }
    if (lane == 0) { bv[warp] = b; bi[warp] = ix; }
    __syncthreads();
    if (threadIdx.x == 0) {
        for (int w = 1; w < 8; w++) if (bv[w] > b || (bv[w] == b && bi[w] < ix)) { b = bv[w]; ix = bi[w]; }
        part[t * HB + c] = make_float2(b, __int_as_float(ix));
    }
}
__global__ void k_argmax_p2(const float2 * part, const int * ids, int * out) {
    const int t = blockIdx.x, lane = threadIdx.x;
    const float2 p = part[t * HB + lane];
    float b = p.x;
    int ix = __float_as_int(p.y);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        const float ob = __shfl_xor_sync(0xffffffffu, b, o);
        const int oi = __shfl_xor_sync(0xffffffffu, ix, o);
        if (ob > b || (ob == b && oi < ix)) { b = ob; ix = oi; }
    }
    if (ix == 0x7fffffff) ix = 0; // (no finite logit: NaN rows must not yield an out-of-range token)
    if (lane == 0) out[t] = ids ? ids[ix] : ix;
}
// bitonic sort (descending) of n values / ids in shared memory, 256 threads
__device__ void bitonic_desc(float * sv, int * si, int n) {
    const int tid = threadIdx.x;
    for (int k = 2; k <= n; k <<= 1)
        for (int jj = k >> 1; jj > 0; jj >>= 1) {
            for (int i = tid; i < n; i += 256) {
                const int l2 = i ^ jj;
                if (l2 > i) {
                    const bool desc = (i & k) == 0;
                    const float a = sv[i], b = sv[l2];
                    if (desc ? a < b : a > b) {
                        sv[i] = b; sv[l2] = a;
                        const int x = si[i]; si[i] = si[l2]; si[l2] = x;
                    }
                }
            }
            __syncthreads();
        }
}
// pass 1: top 32 of a chunk (per-thread top 4 of its ~30 values, then a sort of the block's 1024: more than 4 of
// the chunk's top 32 on one thread's stride is vanishingly rare)
__global__ void __launch_bounds__(256) k_topk_p1(const float * y, int ld, int N, float2 * part) {
    __shared__ float sv[1024];
    __shared__ int si[1024];
    const int t = blockIdx.y, c = blockIdx.x, len = (N + HB - 1) / HB, i0 = c * len, i1 = min(N, i0 + len), tid = threadIdx.x;
    const float * l = y + (size_t) t * ld;
    float v[4];
    int ix[4];
#pragma unroll
    for (int j = 0; j < 4; j++) { v[j] = -INFINITY; ix[j] = -1; }
    for (int i = i0 + tid; i < i1; i += 256) {
        const float x = __ldcg(l + i);
        if (x > v[3]) {
            int j = 3;
            while (j > 0 && v[j - 1] < x) { v[j] = v[j - 1]; ix[j] = ix[j - 1]; j--; }
            v[j] = x; ix[j] = i;
        }
    }
#pragma unroll
    for (int j = 0; j < 4; j++) { sv[tid * 4 + j] = v[j]; si[tid * 4 + j] = ix[j]; }
    __syncthreads();
    bitonic_desc(sv, si, 1024);
    if (tid < 32) part[((size_t) t * HB + c) * 32 + tid] = make_float2(sv[tid], __int_as_float(si[tid]));
}
// pass 2: the HB x 32 candidates of row t -> its top 32
__global__ void __launch_bounds__(256) k_topk_p2(const float2 * part, const int * ids, float2 * out) {
    __shared__ float sv[HB * 32];
    __shared__ int si[HB * 32];
    const int t = blockIdx.x, tid = threadIdx.x;
    for (int i = tid; i < HB * 32; i += 256) {
        const float2 p = part[(size_t) t * HB * 32 + i];
        sv[i] = p.x;
        si[i] = __float_as_int(p.y);
    }
    __syncthreads();
    bitonic_desc(sv, si, HB * 32);
    if (tid < 32) {
        const int id = si[tid];
        out[t * 32 + tid] = make_float2(sv[tid], __int_as_float(id < 0 ? -1 : (ids ? ids[id] : id)));
    }
}

// standalone timing kernels (synthetic activations), one launch per layer phase
template <int T>
__global__ void __launch_bounds__(NT, 1) k_gu(const Mat g, const Mat u) {
    setup_x5(T);
    q4<(T + 7) / 8>(g, g_bout, g.N, 0, T);
    q4<(T + 7) / 8>(u, g_bout + T * FF, u.N, 0, T);
}
template <int T, int TYPE>
__global__ void __launch_bounds__(NT, 1) k_down(const Mat m) { down<(T + 7) / 8, TYPE>(m, g_bx, g_bd, g_bout, D, 0, T); }
template <int T>
__global__ void __launch_bounds__(NT, 1) k_dn(const Mat qkv, const Mat gate) {
    setup_x5(T);
    if (qkv.type == T_Q6K) q6<(T + 7) / 8>(qkv, g_bout, qkv.N, 0, T);
    else q4<(T + 7) / 8>(qkv, g_bout, qkv.N, 0, T);
    q4<(T + 7) / 8>(gate, g_bout + T * QKV, gate.N, 0, T);
}

// ---- decode attention over several sequences (rows r0[s] .. r0[s] + nt[s] at positions pos[s] ..), each with its own
// K/V cache: prep (norms, RoPE, K / V -> the sequence's cache) for all rows in one launch; flash-decoding over context
// splits of `chunk` positions (grid split x kv head x sequence: long contexts fill the GPU); a combine of the splits.
constexpr int ATT_SPL = 32; // most splits per sequence (a segment's, then the own context's)
constexpr int ATT_SMEM = (6 * 8 + 4 * AT_K) * AT_LD * 2; // k_attn_ms: Q rows, double-buffered K / V tiles
struct AttnMs {
    void * kc[TM];
    void * vc[TM];
    __half * ksc[TM];
    __half * vsc[TM];
    int pos[TM], nt[TM], r0[TM], rseq[TM];
    // the sequence's shared segment (positions < segn, stride segctx; the own cache holds positions segn ..)
    void * kseg[TM];
    void * vseg[TM];
    __half * ksseg[TM];
    __half * vsseg[TM];
    int segn[TM], segctx[TM];
    const int * segptab[TM]; // the segment's pages (nullptr: contiguous, stride segctx)
    const int * ptab[TM]; // the own cache's pages (nullptr: contiguous, stride ctx)
    // k_attn_ms2: positions lo .., as splits soff .. of the rows; ns: all splits of the rows (k_attn_comb)
    int lo[TM], soff[TM], ns[TM];
    int ring[TM]; // MK_KV_WIN: the sequence's block of q8 rings in cKP.ring (its slot)
};
// K/V row of sequence s (attention slot, kv head g, position p) and its cache: segment below segn, own (stride ctx) above
__device__ __forceinline__ size_t ms_kv_row(const AttnMs & A, int s, int slot, int g, int p, int ctx, bool & seg) {
    seg = p < A.segn[s];
    return seg ? kv_own_row(A.segptab[s], A.segctx[s], slot, g, p) : kv_own_row(A.ptab[s], ctx, slot, g, p - A.segn[s]);
}
// MK_KV: K (which 0) / V (1) row of sequence s at position p: its q8 ring at p >= ws if the side has one (8 bits), else
// the segment (p < segn) or its own pages in the slot's format; returns the bits, q / sc <- values / scales
__device__ __forceinline__ int ms_kv_src(const AttnMs & A, int s, int which, int slot, int g, int p, int ws, const int8_t *& q, const __half *& sc) {
    const int ro = cKP.ro[which][slot];
    if (ro >= 0 && p >= ws) {
        const size_t rr = (size_t) g * cB.kvrc + p % cB.kvrc;
        const int8_t * blk = cKP.ring + A.ring[s] * cKP.rstride;
        q = blk + ro + rr * HD;
        sc = reinterpret_cast<const __half *>(blk + cKP.rso[which][slot]) + rr * (HD / 32);
        return 8;
    }
    const bool seg = p < A.segn[s];
    const int * pt = seg ? A.segptab[s] : A.ptab[s];
    const int i = seg ? p : p - A.segn[s];
    q = static_cast<const int8_t *>(seg ? (which ? A.vseg[s] : A.kseg[s]) : (which ? A.vc[s] : A.kc[s])) + kv_pg_voff(which, slot, g, i, pt);
    sc = (seg ? (which ? A.vsseg[s] : A.ksseg[s]) : (which ? A.vsc[s] : A.ksc[s])) + kv_own_row(pt, 0, slot, g, i) * (HD / 32);
    return cB.kvb[which][slot];
}
// a staged row (values | scales, ld ATQ_ROW) of `bits` -> the q8 layout in place (kv_rows_expand); bits is warp-uniform
__device__ __forceinline__ void kv_row_expand_any(uint8_t * row, int bits) {
    switch (bits) {
    case 4: kv_rows_expand<4, true>(row, 1, 0, 1, 0); break;
    case 5: kv_rows_expand<5, true>(row, 1, 0, 1, 0); break;
    case 6: kv_rows_expand<6, true>(row, 1, 0, 1, 0); break;
    case 7: kv_rows_expand<7, true>(row, 1, 0, 1, 0); break;
    default: break;
    }
}
// MK_KV_WIN, parallel slots: the ring rows of positions p0 .. p1 - 1 of the slot in cB (ms_bufs) after it went back
// (rewind; their ring slots held the abandoned continuation's rows): positions below cB.segn from its segment's ring
// block sring (the exact q8 rows the segment's builder wrote), the others re-stored as q8 from the own cache (the main
// format's precision until they age out, as k_ring_refill). One warp per row.
__global__ void __launch_bounds__(256) k_ring_fill_ms(int p0, int p1, const int8_t * sring) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int n = p1 - p0;
    for (int i = blockIdx.x * 8 + warp; i < n * NKV * N_ATT_SLOTS * 2; i += gridDim.x * 8) {
        const int w = i & 1, slot = (i >> 1) % N_ATT_SLOTS, g = ((i >> 1) / N_ATT_SLOTS) % NKV;
        const int p = p0 + (i >> 1) / (N_ATT_SLOTS * NKV);
        if (!cB.kvrp[w][slot]) continue;
        const size_t rr = (size_t) g * cB.kvrc + p % cB.kvrc;
        int8_t * q = cB.kvrp[w][slot] + rr * HD;
        __half * sc = cB.kvrs[w][slot] + rr * (HD / 32);
        if (p < cB.segn) {
            *reinterpret_cast<uint2 *>(q + 8 * lane) = *reinterpret_cast<const uint2 *>(sring + cKP.ro[w][slot] + rr * HD + 8 * lane);
            if (lane < HD / 32) sc[lane] = reinterpret_cast<const __half *>(sring + cKP.rso[w][slot])[rr * (HD / 32) + lane];
            continue;
        }
        const __half * ms;
        const int bits = cB.kvb[w][slot];
        const uint8_t * rp = reinterpret_cast<const uint8_t *>(kv_vptr(w, slot, g, p, ms));
        float v[8];
        kv_row_load(rp, bits, __half2float(ms[lane >> 2]), v);
        __syncwarp();
        kv_store_q8(q, sc, v);
    }
}
// LB: MK_KV (the slot's packed format in the pages, Hadamard rotation, MK_KV_WIN ring rows)
template <bool LB = false>
__global__ void __launch_bounds__(256) k_attn_prep_ms(const AttnMs A, int slot, int ctx, const float * qn, const float * kn,
                                                       const float * y0, const float * y1, __half * qh) {
    const int t = blockIdx.x, s = A.rseq[t], pos = A.pos[s] + t - A.r0[s], lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    for (int it = warp; it < 32; it += 8) {
        float v[8];
        if (it < NH) load8(y0 + (size_t) t * QG + it * 512 + lane * 8, v);
        else load8(y1 + (size_t) t * 2 * KVD + (it >= NH + NKV ? KVD : 0) + ((it - NH) % NKV) * HD + lane * 8, v);
        const bool isv = it >= NH + NKV;
        if (!isv) {
            const float * nw = it < NH ? qn : kn;
            float ss = 0.f;
#pragma unroll
            for (int k = 0; k < 8; k++) ss += v[k] * v[k];
            const float rs = rsqrtf(wsum(ss) / 256.0f + EPS);
#pragma unroll
            for (int k = 0; k < 8; k++) v[k] *= rs * __ldg(nw + lane * 8 + k);
            rope8(v, lane, pos);
            if constexpr (LB) if (it < NH && cB.kvr[slot]) fwht256(v); // (the slot's K is stored rotated)
        }
        const float sc = it < NH ? 0.0625f : 1.0f;
        __half2 hv[4];
#pragma unroll
        for (int k = 0; k < 4; k++) hv[k] = __floats2half2_rn(v[2 * k] * sc, v[2 * k + 1] * sc);
        if (it < NH) {
            *reinterpret_cast<int4 *>(qh + ((size_t) t * NH + it) * HD + lane * 8) = *reinterpret_cast<int4 *>(hv);
        } else if constexpr (LB) {
            const int w = isv, g = (it - NH) % NKV, i = pos - A.segn[s];
            if (w ? cB.kvrv[slot] : cB.kvr[slot]) fwht256(v);
            kv_store_fmt(static_cast<int8_t *>(w ? A.vc[s] : A.kc[s]) + kv_pg_voff(w, slot, g, i, A.ptab[s]),
                         (w ? A.vsc[s] : A.ksc[s]) + kv_own_row(A.ptab[s], ctx, slot, g, i) * (HD / 32), v, cB.kvb[w][slot]);
            if (cKP.ro[w][slot] >= 0) {
                const size_t rr = (size_t) g * cB.kvrc + pos % cB.kvrc;
                int8_t * blk = cKP.ring + A.ring[s] * cKP.rstride;
                kv_store_q8(blk + cKP.ro[w][slot] + rr * HD, reinterpret_cast<__half *>(blk + cKP.rso[w][slot]) + rr * (HD / 32), v);
            }
        } else {
            const size_t row = kv_own_row(A.ptab[s], ctx, slot, (it - NH) % NKV, pos - A.segn[s]); // (new positions: own cache)
            if (cB.kvq8) kv_store_q8(static_cast<int8_t *>(isv ? A.vc[s] : A.kc[s]) + row * HD, (isv ? A.vsc[s] : A.ksc[s]) + row * (HD / 32), v);
            else *reinterpret_cast<int4 *>(static_cast<__half *>(isv ? A.vc[s] : A.kc[s]) + row * HD + lane * 8) = *reinterpret_cast<int4 *>(hv);
        }
    }
}
// as pk_attn for one sequence's rows (<= 16) and the context split [chunk c, chunk (c + 1)): unnormalised output and
// (max, sum) per row and head -> po [row][NH][ATT_SPL][HD], pml [row][NH][ATT_SPL]
template <bool Q8>
__global__ void __launch_bounds__(192) k_attn_ms(const AttnMs A, int slot, int ctx, int chunk, const __half * qh, float * po, float2 * pml) {
    extern __shared__ __align__(16) uint8_t asm_[];
    __half * sQ = reinterpret_cast<__half *>(asm_); // [6 heads][8 rows][AT_LD] (A rows 8 .. 15 are zero registers)
    __half * sKV = sQ + 6 * 8 * AT_LD;              // [2 buffers][K, V][AT_K][AT_LD] (Q8: buffer 0 only)
    const int c = blockIdx.x, g = blockIdx.y, s = blockIdx.z;
    const int nq = A.nt[s];
    if (!nq) return;
    const int qpos0 = A.pos[s], pmax = qpos0 + nq - 1, k_lo = c * chunk;
    if (k_lo > pmax) return;
    const int k_hi = min(k_lo + chunk, pmax + 1);
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g8 = lane >> 2, t4 = lane & 3;
    const size_t row0 = ((size_t) slot * NKV + g) * ctx;
    const __half * Kc = static_cast<const __half *>(A.kc[s]) + row0 * HD;
    const __half * Vc = static_cast<const __half *>(A.vc[s]) + row0 * HD;
    const int rq = A.r0[s];
    for (int i = tid; i < 6 * 8 * 32; i += 192) {
        const int r = i >> 5, cc = i & 31, hq = r / 8, tt = r % 8;
        __half * d = sQ + r * AT_LD + 8 * cc;
        if (tt < nq) cp_async16(d, qh + ((size_t) (rq + tt) * NH + g * 6 + hq) * HD + 8 * cc);
        else *reinterpret_cast<int4 *>(d) = make_int4(0, 0, 0, 0);
    }
    pf::cpa_commit();
    float o[32][4];
#pragma unroll
    for (int i = 0; i < 32; i++) o[i][0] = o[i][1] = o[i][2] = o[i][3] = 0.f;
    float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.f, l1 = 0.f;
    const int qr0 = qpos0 + g8, qr1 = qpos0 + g8 + 8;
    const __half * qa = sQ + (warp * 8 + g8) * AT_LD + 2 * t4;
    auto issue = [&](int kb, int buf) { // fp16: tile kb -> buffer buf (cp.async)
        __half * sK = sKV + buf * 2 * AT_K * AT_LD, * sV = sK + AT_K * AT_LD;
        const int np = min(AT_K, k_hi - kb);
        for (int i = tid; i < AT_K * 32; i += 192) {
            const int p = i >> 5, cc = i & 31;
            if (p < np) {
                bool seg;
                const size_t row = ms_kv_row(A, s, slot, g, kb + p, ctx, seg);
                cp_async16(sK + p * AT_LD + 8 * cc, static_cast<const __half *>(seg ? A.kseg[s] : A.kc[s]) + row * HD + 8 * cc);
                cp_async16(sV + p * AT_LD + 8 * cc, static_cast<const __half *>(seg ? A.vseg[s] : A.vc[s]) + row * HD + 8 * cc);
            } else {
                *reinterpret_cast<int4 *>(sK + p * AT_LD + 8 * cc) = make_int4(0, 0, 0, 0);
                *reinterpret_cast<int4 *>(sV + p * AT_LD + 8 * cc) = make_int4(0, 0, 0, 0);
            }
        }
        pf::cpa_commit();
    };
    if constexpr (!Q8) issue(k_lo, 0);
    int buf = 0;
    for (int kb = k_lo; kb < k_hi; kb += AT_K, buf ^= 1) {
        const int np = min(AT_K, k_hi - kb);
        __half * sK = sKV + (Q8 ? 0 : buf) * 2 * AT_K * AT_LD, * sV = sK + AT_K * AT_LD;
        if constexpr (Q8) {
            constexpr int PER = (2 * AT_K * 32 + 191) / 192;
            uint2 q[PER];
            float d[PER];
#pragma unroll
            for (int k = 0; k < PER; k++) {
                const int i = tid + 192 * k, which = i / (AT_K * 32), r = (i >> 5) % AT_K, cc = i & 31;
                q[k] = make_uint2(0, 0);
                d[k] = 0.f;
                if (i < 2 * AT_K * 32 && r < np) {
                    bool seg;
                    const size_t row = ms_kv_row(A, s, slot, g, kb + r, ctx, seg);
                    q[k] = __ldg(reinterpret_cast<const uint2 *>(static_cast<const int8_t *>(seg ? (which ? A.vseg[s] : A.kseg[s]) : (which ? A.vc[s] : A.kc[s])) + row * HD) + cc);
                    d[k] = __half2float((seg ? (which ? A.vsseg[s] : A.ksseg[s]) : (which ? A.vsc[s] : A.ksc[s]))[row * (HD / 32) + (cc >> 2)]);
                }
            }
            pf::cpa_wait<0>();
#pragma unroll
            for (int k = 0; k < PER; k++) {
                const int i = tid + 192 * k, which = i / (AT_K * 32), r = (i >> 5) % AT_K, cc = i & 31;
                if (i >= 2 * AT_K * 32) break;
                __half2 h[4];
                q8_to_f16(q[k], d[k], h);
                *reinterpret_cast<int4 *>((which ? sV : sK) + r * AT_LD + 8 * cc) = *reinterpret_cast<int4 *>(h);
            }
        } else { // the next tile streams in while this one is used
            if (kb + AT_K < k_hi) { issue(kb + AT_K, buf ^ 1); pf::cpa_wait<1>(); }
            else pf::cpa_wait<0>();
        }
        __syncthreads();
        float sc[AT_K / 8][4];
#pragma unroll
        for (int ni = 0; ni < AT_K / 8; ni++) sc[ni][0] = sc[ni][1] = sc[ni][2] = sc[ni][3] = 0.f;
#pragma unroll
        for (int ks = 0; ks < HD / 16; ks++) {
            const unsigned a0 = *reinterpret_cast<const unsigned *>(qa + 16 * ks), a1 = 0u;
            const unsigned a2 = *reinterpret_cast<const unsigned *>(qa + 16 * ks + 8), a3 = 0u;
#pragma unroll
            for (int ni = 0; ni < AT_K / 8; ni++) {
                const __half * kp = sK + (8 * ni + g8) * AT_LD + 2 * t4 + 16 * ks;
                mma16816(sc[ni], a0, a1, a2, a3, *reinterpret_cast<const unsigned *>(kp), *reinterpret_cast<const unsigned *>(kp + 8));
            }
        }
        float mx0 = -INFINITY, mx1 = -INFINITY;
#pragma unroll
        for (int ni = 0; ni < AT_K / 8; ni++) {
            const int p = kb + 8 * ni + 2 * t4;
            if (p > qr0 || p >= k_hi) sc[ni][0] = -INFINITY;
            if (p + 1 > qr0 || p + 1 >= k_hi) sc[ni][1] = -INFINITY;
            if (p > qr1 || p >= k_hi) sc[ni][2] = -INFINITY;
            if (p + 1 > qr1 || p + 1 >= k_hi) sc[ni][3] = -INFINITY;
            mx0 = fmaxf(mx0, fmaxf(sc[ni][0], sc[ni][1]));
            mx1 = fmaxf(mx1, fmaxf(sc[ni][2], sc[ni][3]));
        }
        mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 1));
        mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 2));
        mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 1));
        mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 2));
        const float mn0 = fmaxf(m0, mx0), mn1 = fmaxf(m1, mx1);
        const float al0 = mn0 == -INFINITY ? 1.f : expf(m0 - mn0), al1 = mn1 == -INFINITY ? 1.f : expf(m1 - mn1);
        float ps0 = 0.f, ps1 = 0.f;
        unsigned pa[AT_K / 8][2];
#pragma unroll
        for (int ni = 0; ni < AT_K / 8; ni++) {
            const float p0 = mn0 == -INFINITY ? 0.f : expf(sc[ni][0] - mn0);
            const float p1 = mn0 == -INFINITY ? 0.f : expf(sc[ni][1] - mn0);
            const float p2 = mn1 == -INFINITY ? 0.f : expf(sc[ni][2] - mn1);
            const float p3 = mn1 == -INFINITY ? 0.f : expf(sc[ni][3] - mn1);
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
#pragma unroll
        for (int ks = 0; ks < AT_K / 16; ks++) {
            const unsigned a0 = pa[2 * ks][0], a1 = pa[2 * ks][1], a2 = pa[2 * ks + 1][0], a3 = pa[2 * ks + 1][1];
#pragma unroll
            for (int dt2 = 0; dt2 < 16; dt2++) {
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
    const int hq = g * 6 + warp;
#pragma unroll
    for (int half_ = 0; half_ < 1; half_++) { // (rows 8 .. 15 are padding)
        const int tt = g8 + 8 * half_;
        if (tt >= nq) continue;
        const size_t idx = ((size_t) (rq + tt) * NH + hq) * ATT_SPL + c;
#pragma unroll
        for (int i = 0; i < 32; i++) {
            float * dst = po + idx * HD + 8 * i + 2 * t4;
            dst[0] = o[i][2 * half_];
            dst[1] = o[i][2 * half_ + 1];
        }
        if (t4 == 0) pml[idx] = half_ ? make_float2(m1, l1) : make_float2(m0, l0);
    }
}
// k_attn_ms with the GQA group packed into the MMA rows: row r = (q head r / nq, query r % nq) of the kv head (6 nq
// <= 16 rows, so nq <= 2) instead of 16 rows per head with nq used. 4 warps share each 32-position K / V tile:
// warp w scores positions 8 w .. 8 w + 7 for all rows, the tile's softmax goes through shared memory, and warp w
// accumulates dims 64 w .. 64 w + 63 of P V. Same outputs as k_attn_ms.
constexpr int ATP_LS = AT_K + 2; // (even: float2 reads)
constexpr int ATQ_ROW = HD + HD / 32 * 2, ATQ_STAGE = 2 * AT_K * ATQ_ROW; // q8: a staged K / V row (values | scales), a tile
// Q, K/V tiles (fp16: 2 buffers; q8: 1 fp16 buffer + 2 raw q8 stages), scores, row maxima
template <int MT, bool Q8 = false> constexpr int att_smem2() {
    return (16 * MT + (Q8 ? 2 : 4) * AT_K) * AT_LD * 2 + (Q8 ? 2 * ATQ_STAGE : 0) + (16 * MT * ATP_LS + 4 * 16 * MT) * 4;
}
template <bool Q8, int MT>
// 256 threads: all stage and convert the K / V tiles, warps 0 .. 3 compute (twice the loads in flight and conversion
// throughput of 4 warps; the scores / softmax / P V split stays 4-way)
__global__ void __launch_bounds__(256) k_attn_ms2(const AttnMs A, int slot, int ctx, int chunk, const __half * qh, float * po, float2 * pml) {
    extern __shared__ __align__(16) uint8_t asm_[];
    constexpr int NR = 16 * MT;                     // packed rows: 6 nq <= NR
    __half * sQ = reinterpret_cast<__half *>(asm_); // [NR][AT_LD]
    __half * sKV = sQ + NR * AT_LD;                 // [2 buffers (Q8: 1)][K, V][AT_K][AT_LD]
    uint8_t * sRaw = reinterpret_cast<uint8_t *>(sKV + (Q8 ? 2 : 4) * AT_K * AT_LD); // Q8: [2 stages][K, V][AT_K][ATQ_ROW]
    float * sS = reinterpret_cast<float *>(sRaw + (Q8 ? 2 * ATQ_STAGE : 0)); // [NR][ATP_LS] tile scores
    float * sM = sS + NR * ATP_LS;                                  // [4 warps][NR] their row maxima
    // grid (sequence, kv head, split): the sequences' blocks of a context split run together, so a shared segment's
    // K / V tiles come from L2 for all but the first
    const int s = blockIdx.x, g = blockIdx.y, c = blockIdx.z;
    const int nq = A.nt[s];
    if (!nq) return;
    const int qpos0 = A.pos[s], pmax = qpos0 + nq - 1, k_lo = A.lo[s] + c * chunk;
    if (k_lo > pmax) return;
    const int k_hi = min(k_lo + chunk, pmax + 1), nr = 6 * nq;
    const int tid = threadIdx.x, lane = tid & 31, w = tid >> 5, g8 = lane >> 2, t4 = lane & 3;
    const size_t row0 = ((size_t) slot * NKV + g) * ctx;
    const __half * Kc = static_cast<const __half *>(A.kc[s]) + row0 * HD;
    const __half * Vc = static_cast<const __half *>(A.vc[s]) + row0 * HD;
    const int rq = A.r0[s];
    for (int i = tid; i < NR * 32; i += 256) {
        const int r = i >> 5, cc = i & 31;
        __half * d = sQ + r * AT_LD + 8 * cc;
        if (r < nr) cp_async16(d, qh + ((size_t) (rq + r % nq) * NH + g * 6 + r / nq) * HD + 8 * cc);
        else *reinterpret_cast<int4 *>(d) = make_int4(0, 0, 0, 0);
    }
    pf::cpa_commit();
    auto issue = [&](int kb, int buf) { // fp16: tile kb -> buffer buf (cp.async)
        __half * sK = sKV + buf * 2 * AT_K * AT_LD, * sV = sK + AT_K * AT_LD;
        const int np = min(AT_K, k_hi - kb);
        for (int i = tid; i < AT_K * 32; i += 256) {
            const int p = i >> 5, cc = i & 31;
            if (p < np) {
                bool seg;
                const size_t row = ms_kv_row(A, s, slot, g, kb + p, ctx, seg);
                cp_async16(sK + p * AT_LD + 8 * cc, static_cast<const __half *>(seg ? A.kseg[s] : A.kc[s]) + row * HD + 8 * cc);
                cp_async16(sV + p * AT_LD + 8 * cc, static_cast<const __half *>(seg ? A.vseg[s] : A.vc[s]) + row * HD + 8 * cc);
            } else {
                *reinterpret_cast<int4 *>(sK + p * AT_LD + 8 * cc) = make_int4(0, 0, 0, 0);
                *reinterpret_cast<int4 *>(sV + p * AT_LD + 8 * cc) = make_int4(0, 0, 0, 0);
            }
        }
        pf::cpa_commit();
    };
    auto issue_q8 = [&](int kb, int st) { // q8: raw rows of tile kb (values, scales) -> stage st (cp.async)
        uint8_t * base = sRaw + st * ATQ_STAGE;
        const int np = min(AT_K, k_hi - kb);
        for (int i = tid; i < 2 * AT_K * 17; i += 256) {
            const int which = i / (AT_K * 17), r = (i / 17) % AT_K, cc = i % 17;
            uint8_t * d = base + (which * AT_K + r) * ATQ_ROW + 16 * cc;
            if (r < np) {
                bool seg;
                const size_t row = ms_kv_row(A, s, slot, g, kb + r, ctx, seg);
                if (cc < 16) cp_async16(d, static_cast<const int8_t *>(seg ? (which ? A.vseg[s] : A.kseg[s]) : (which ? A.vc[s] : A.kc[s])) + row * HD + 16 * cc);
                else cp_async16(d, (seg ? (which ? A.vsseg[s] : A.ksseg[s]) : (which ? A.vsc[s] : A.ksc[s])) + row * (HD / 32));
            } else if (cc == 16) {
                *reinterpret_cast<int4 *>(d) = make_int4(0, 0, 0, 0); // (zero scales: the row contributes nothing)
            }
        }
        pf::cpa_commit();
    };
    if constexpr (Q8) issue_q8(k_lo, 0);
    else issue(k_lo, 0);
    float o[MT][8][4];
    float m[MT][2], l[MT][2];
    int qr[MT][2];
#pragma unroll
    for (int mt = 0; mt < MT; mt++) {
#pragma unroll
        for (int i = 0; i < 8; i++) o[mt][i][0] = o[mt][i][1] = o[mt][i][2] = o[mt][i][3] = 0.f;
#pragma unroll
        for (int hr = 0; hr < 2; hr++) {
            const int r = 16 * mt + g8 + 8 * hr;
            m[mt][hr] = -INFINITY;
            l[mt][hr] = 0.f;
            qr[mt][hr] = r < nr ? qpos0 + r % nq : -1;
        }
    }
    int buf = 0;
    for (int kb = k_lo; kb < k_hi; kb += AT_K, buf ^= 1) {
        __half * sK = sKV + (Q8 ? 0 : buf) * 2 * AT_K * AT_LD, * sV = sK + AT_K * AT_LD;
        if constexpr (Q8) { // the next tile's raw rows stream in while this one converts (fp16) and is used
            if (kb + AT_K < k_hi) { issue_q8(kb + AT_K, buf ^ 1); pf::cpa_wait<1>(); }
            else pf::cpa_wait<0>();
            __syncthreads();
            const uint8_t * base = sRaw + buf * ATQ_STAGE;
            for (int i = tid; i < 2 * AT_K * 32; i += 256) {
                const int which = i / (AT_K * 32), r = (i >> 5) % AT_K, cc = i & 31;
                const uint8_t * src = base + (which * AT_K + r) * ATQ_ROW;
                __half2 h[4];
                q8_to_f16(*reinterpret_cast<const uint2 *>(src + 8 * cc), __half2float(reinterpret_cast<const __half *>(src + HD)[cc >> 2]), h);
                *reinterpret_cast<int4 *>((which ? sV : sK) + r * AT_LD + 8 * cc) = *reinterpret_cast<int4 *>(h);
            }
        } else {
            if (kb + AT_K < k_hi) { issue(kb + AT_K, buf ^ 1); pf::cpa_wait<1>(); }
            else pf::cpa_wait<0>();
        }
        __syncthreads();
        if (w < 4) {
        // scores of positions 8 w .. 8 w + 7
        float sc[MT][4];
#pragma unroll
        for (int mt = 0; mt < MT; mt++) sc[mt][0] = sc[mt][1] = sc[mt][2] = sc[mt][3] = 0.f;
#pragma unroll
        for (int ks = 0; ks < HD / 16; ks++) {
            const __half * kp = sK + (8 * w + g8) * AT_LD + 2 * t4 + 16 * ks;
            const unsigned b0 = *reinterpret_cast<const unsigned *>(kp), b1 = *reinterpret_cast<const unsigned *>(kp + 8);
#pragma unroll
            for (int mt = 0; mt < MT; mt++) {
                const __half * qa = sQ + (16 * mt + g8) * AT_LD + 2 * t4 + 16 * ks;
                mma16816(sc[mt], *reinterpret_cast<const unsigned *>(qa), *reinterpret_cast<const unsigned *>(qa + 8 * AT_LD),
                         *reinterpret_cast<const unsigned *>(qa + 8), *reinterpret_cast<const unsigned *>(qa + 8 * AT_LD + 8), b0, b1);
            }
        }
        const int p = kb + 8 * w + 2 * t4;
#pragma unroll
        for (int mt = 0; mt < MT; mt++) {
#pragma unroll
            for (int hr = 0; hr < 2; hr++) {
                if (p > qr[mt][hr] || p >= k_hi) sc[mt][2 * hr] = -INFINITY;
                if (p + 1 > qr[mt][hr] || p + 1 >= k_hi) sc[mt][2 * hr + 1] = -INFINITY;
                float mx = fmaxf(sc[mt][2 * hr], sc[mt][2 * hr + 1]);
                mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 1));
                mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 2));
                const int r = 16 * mt + g8 + 8 * hr;
                sS[r * ATP_LS + 8 * w + 2 * t4] = sc[mt][2 * hr];
                sS[r * ATP_LS + 8 * w + 2 * t4 + 1] = sc[mt][2 * hr + 1];
                if (t4 == 0) sM[w * NR + r] = mx;
            }
        }
        }
        __syncthreads();
        if (w < 4) {
#pragma unroll
        for (int mt = 0; mt < MT; mt++) {
            float al[2], mn[2], ps[2] = {0.f, 0.f};
#pragma unroll
            for (int hr = 0; hr < 2; hr++) {
                const int r = 16 * mt + g8 + 8 * hr;
                const float mx = fmaxf(fmaxf(sM[r], sM[NR + r]), fmaxf(sM[2 * NR + r], sM[3 * NR + r]));
                mn[hr] = fmaxf(m[mt][hr], mx);
                al[hr] = mn[hr] == -INFINITY ? 1.f : expf(m[mt][hr] - mn[hr]);
            }
            unsigned pa[AT_K / 16][4]; // P as the A operand: [k step][a0 .. a3]
#pragma unroll
            for (int ks = 0; ks < AT_K / 16; ks++)
#pragma unroll
                for (int hk = 0; hk < 2; hk++) {
                    const int col = 16 * ks + 8 * hk + 2 * t4;
#pragma unroll
                    for (int hr = 0; hr < 2; hr++) {
                        const float2 sv = *reinterpret_cast<const float2 *>(sS + (16 * mt + g8 + 8 * hr) * ATP_LS + col);
                        const float p0 = mn[hr] == -INFINITY ? 0.f : expf(sv.x - mn[hr]), p1 = mn[hr] == -INFINITY ? 0.f : expf(sv.y - mn[hr]);
                        ps[hr] += p0 + p1;
                        __half2 hh = __floats2half2_rn(p0, p1);
                        pa[ks][2 * hk + hr] = *reinterpret_cast<unsigned *>(&hh);
                    }
                }
#pragma unroll
            for (int hr = 0; hr < 2; hr++) {
                ps[hr] += __shfl_xor_sync(0xffffffffu, ps[hr], 1);
                ps[hr] += __shfl_xor_sync(0xffffffffu, ps[hr], 2);
                l[mt][hr] = l[mt][hr] * al[hr] + ps[hr];
                m[mt][hr] = mn[hr];
            }
#pragma unroll
            for (int i = 0; i < 8; i++) {
                o[mt][i][0] *= al[0];
                o[mt][i][1] *= al[0];
                o[mt][i][2] *= al[1];
                o[mt][i][3] *= al[1];
            }
#pragma unroll
            for (int ks = 0; ks < AT_K / 16; ks++)
#pragma unroll
                for (int dt2 = 0; dt2 < 4; dt2++) { // two 8-dim tiles of this warp's 64 per ldmatrix.x4
                    const __half * vb = sV + (16 * ks + (lane & 15)) * AT_LD + 64 * w + 16 * dt2 + (lane >> 4) * 8;
                    unsigned b0, b1, b2, b3;
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
                                 : "=r"(b0), "=r"(b1), "=r"(b2), "=r"(b3)
                                 : "r"((unsigned) __cvta_generic_to_shared(vb)));
                    mma16816(o[mt][2 * dt2], pa[ks][0], pa[ks][1], pa[ks][2], pa[ks][3], b0, b1);
                    mma16816(o[mt][2 * dt2 + 1], pa[ks][0], pa[ks][1], pa[ks][2], pa[ks][3], b2, b3);
                }
        }
        }
        __syncthreads();
    }
    if (w >= 4) return;
#pragma unroll
    for (int mt = 0; mt < MT; mt++)
#pragma unroll
        for (int hr = 0; hr < 2; hr++) {
            const int r = 16 * mt + g8 + 8 * hr;
            if (r >= nr) continue;
            const int tt = r % nq, hq = g * 6 + r / nq;
            const size_t idx = ((size_t) (rq + tt) * NH + hq) * ATT_SPL + A.soff[s] + c;
#pragma unroll
            for (int i = 0; i < 8; i++) {
                float * dst = po + idx * HD + 64 * w + 8 * i + 2 * t4;
                dst[0] = o[mt][i][2 * hr];
                dst[1] = o[mt][i][2 * hr + 1];
            }
            if (w == 0 && t4 == 0) pml[idx] = make_float2(m[mt][hr], l[mt][hr]);
        }
}
// ---- q8 decode attention as jobs: a job = positions lo .. hi - 1 of one K / V cache, read by one or more sequences ----
// A shared segment is one job for all the sequences attached to it (each tile staged and converted once for all their
// rows), each sequence's own cache another. One block per (job, kv head, split) scores the job's packed rows (6 nq
// per sequence: q head r / nq of the kv head, query r % nq, sequence after sequence; <= 96 rows), causal per row.
// 8 warps: warp w scores positions 8 (w & 3) .. + 7 (K fragments straight from the raw q8 rows) and accumulates dims
// 64 (w & 3) .. + 63 of P V, for m-tiles w / 4, w / 4 + 2, ... Split c of a job is split soff + c of its rows.
struct AttnJobs {
    const void * k[TM];
    const void * v[TM];
    const __half * ks[TM];
    const __half * vs[TM];
    int stride[TM];  // positions per (attention slot, kv head) of the cache (contiguous) ...
    const int * ptab[TM]; // ... or its pages (own caches of slots)
    int lo[TM], hi[TM], off[TM], soff[TM]; // positions lo .. hi - 1 at cache rows position - off; first split
    int n[TM];       // sequences in job i
    int seq[TM][TM]; // their indices in AttnMs
};
constexpr int ATT_SEG_SPL = 20; // most splits of a segment (one wave of blocks for a group on 82 SMs)
static_assert(KV_PG % AT_K == 0, "a K / V tile lies in one page");
static_assert(sizeof(AttnMs) + sizeof(AttnJobs) + 32 <= 4096, "k_attn_jobs' parameters (4 KB before CUDA 12.1)");
// Q, the V tile (fp16), the raw K / V tile (q8), scores, row maxima
template <int MT> constexpr int att_smem_jobs() {
    return (16 * MT + AT_K) * AT_LD * 2 + ATQ_STAGE + (16 * MT * ATP_LS + 4 * 16 * MT) * 4;
}
// LB (MK_KV): rows are staged one by one from where ms_kv_src puts them (an own job: q8 ring at positions >= the
// sequence's ws, else its pages in the slot's format; a segment job: the segment's pages), the low-bit ones expanded to
// the q8 layout in place, then the q8 path as is. A segment job's sequences may have different windows: each row masks
// the keys from its sequence's ws on (the host starts that sequence's own job at min(segn, ws) instead).
template <int MT, bool LB = false>
__global__ void __launch_bounds__(256, 1) k_attn_jobs(const AttnMs A, const AttnJobs G, int slot, int chunk, const __half * qh, float * po, float2 * pml) {
    extern __shared__ __align__(16) uint8_t asm_[];
    constexpr int NR = 16 * MT, MH = (MT + 1) / 2; // (odd MT: warps 4 .. 7 have one m-tile fewer)
    __half * sQ = reinterpret_cast<__half *>(asm_);                   // [NR][AT_LD]
    __half * sV = sQ + NR * AT_LD;                                    // [AT_K][AT_LD]
    uint8_t * sRaw = reinterpret_cast<uint8_t *>(sV + AT_K * AT_LD); // [K, V][AT_K][ATQ_ROW]
    float * sS = reinterpret_cast<float *>(sRaw + ATQ_STAGE);        // [NR][ATP_LS]
    float * sM = sS + NR * ATP_LS;                                    // [4 position groups][NR]
    const int gi = blockIdx.x, g = blockIdx.y, c = blockIdx.z, ng = G.n[gi];
    const int k_lo = G.lo[gi] + c * chunk;
    if (k_lo >= G.hi[gi]) return;
    const int k_hi = min(k_lo + chunk, G.hi[gi]);
    const int tid = threadIdx.x, lane = tid & 31, w = tid >> 5, g8 = lane >> 2, t4 = lane & 3, wq = w & 3, wm = w >> 2;
    const int8_t * Kq = static_cast<const int8_t *>(G.k[gi]);
    const int8_t * Vq = static_cast<const int8_t *>(G.v[gi]);
    const __half * Ks = G.ks[gi], * Vs = G.vs[gi];
    const int * ptab = G.ptab[gi];
    const int stride = G.stride[gi], off = G.off[gi];
    const int s0 = G.seq[gi][0];
    const bool own = LB && off == A.segn[s0];           // (a segment job: off 0, its sequences' segn > 0)
    const int ws0 = LB ? kv_ws(A.pos[s0] + A.nt[s0]) : 0; // (own job: its sequence's window start)
    // packed row r -> its row in qh, its q head and query position (false: padding)
    auto locate = [&](int r, int & qrow, int & hq, int & qp) {
        for (int j = 0; j < ng; j++) {
            const int sj = G.seq[gi][j], nq = A.nt[sj];
            if (r < 6 * nq) {
                qrow = A.r0[sj] + r % nq;
                hq = g * 6 + r / nq;
                qp = A.pos[sj] + r % nq;
                return true;
            }
            r -= 6 * nq;
        }
        return false;
    };
    for (int i = tid; i < NR * 32; i += 256) {
        const int r = i >> 5, cc = i & 31;
        __half * d = sQ + r * AT_LD + 8 * cc;
        int qrow, hq, qp;
        if (locate(r, qrow, hq, qp)) cp_async16(d, qh + ((size_t) qrow * NH + hq) * HD + 8 * cc);
        else *reinterpret_cast<int4 *>(d) = make_int4(0, 0, 0, 0);
    }
    auto issue = [&](int kb, int which) { // raw rows of tile kb (values, scales) of K (0) or V (1) (cp.async)
        uint8_t * base = sRaw + which * AT_K * ATQ_ROW;
        const int np = min(AT_K, k_hi - kb);
        if constexpr (LB) {
            for (int i = tid; i < AT_K * 17; i += 256) {
                const int r = i / 17, cc = i % 17;
                uint8_t * d = base + r * ATQ_ROW + 16 * cc;
                if (r < np) {
                    const int8_t * q;
                    const __half * sc;
                    int bits;
                    if (own) bits = ms_kv_src(A, s0, which, slot, g, kb + r, ws0, q, sc);
                    else {
                        bits = cB.kvb[which][slot];
                        q = (which ? Vq : Kq) + kv_pg_voff(which, slot, g, kb + r, ptab);
                        sc = (which ? Vs : Ks) + kv_own_row(ptab, stride, slot, g, kb + r) * (HD / 32);
                    }
                    if (cc < 16) {
                        if (cc < 2 * bits) cp_async16(d, q + 16 * cc);
                    } else cp_async16(d, sc);
                } else if (cc == 16) {
                    *reinterpret_cast<int4 *>(d) = make_int4(0, 0, 0, 0);
                }
            }
            pf::cpa_commit();
            return;
        }
        const size_t rb = kv_own_row(ptab, stride, slot, g, kb - off); // (tiles start at multiples of AT_K past off: one page)
        for (int i = tid; i < AT_K * 17; i += 256) {
            const int r = i / 17, cc = i % 17;
            uint8_t * d = base + r * ATQ_ROW + 16 * cc;
            if (r < np) {
                const size_t row = rb + r;
                if (cc < 16) cp_async16(d, (which ? Vq : Kq) + row * HD + 16 * cc);
                else cp_async16(d, (which ? Vs : Ks) + row * (HD / 32));
            } else if (cc == 16) {
                *reinterpret_cast<int4 *>(d) = make_int4(0, 0, 0, 0); // (zero scales: the row contributes nothing)
            }
        }
        pf::cpa_commit();
    };
    issue(k_lo, 0);
    issue(k_lo, 1);
    float o[MH][8][4], m[MH][2], l[MH][2];
    int qr[MH][2]; // query positions of this lane's rows (-1: padding)
#pragma unroll
    for (int j = 0; j < MH; j++) {
#pragma unroll
        for (int i = 0; i < 8; i++) o[j][i][0] = o[j][i][1] = o[j][i][2] = o[j][i][3] = 0.f;
        m[j][0] = m[j][1] = -INFINITY;
        l[j][0] = l[j][1] = 0.f;
#pragma unroll
        for (int hr = 0; hr < 2; hr++) {
            int qrow, hq;
            if (!locate(16 * (wm + 2 * j) + g8 + 8 * hr, qrow, hq, qr[j][hr])) qr[j][hr] = -1;
            else if constexpr (LB) {
                if (!own && (cKP.ro[0][slot] >= 0 || cKP.ro[1][slot] >= 0)) { // (keys >= the row's ws: its own job)
                    int r = 16 * (wm + 2 * j) + g8 + 8 * hr;
                    for (int k = 0; k < ng; k++) {
                        const int sj = G.seq[gi][k];
                        if (r < 6 * A.nt[sj]) { qr[j][hr] = min(qr[j][hr], kv_ws(A.pos[sj] + A.nt[sj]) - 1); break; }
                        r -= 6 * A.nt[sj];
                    }
                }
            }
        }
    }
    const uint8_t * kr = sRaw + (8 * wq + g8) * ATQ_ROW; // this lane's K row (scores)
    for (int kb = k_lo; kb < k_hi; kb += AT_K) {
        const bool more = kb + AT_K < k_hi;
        pf::cpa_wait<0>();
        __syncthreads();
        if constexpr (LB) { // low-bit rows -> the q8 layout (one warp per row)
            for (int rr = w; rr < 2 * AT_K; rr += 8) {
                const int which = rr / AT_K, p = kb + rr % AT_K;
                if (p >= k_hi) continue;
                kv_row_expand_any(sRaw + rr * ATQ_ROW, own && cKP.ro[which][slot] >= 0 && p >= ws0 ? 8 : cB.kvb[which][slot]);
            }
            __syncthreads();
        }
        for (int i = tid; i < AT_K * 32; i += 256) { // V -> fp16
            const int r = i >> 5, cc = i & 31;
            const uint8_t * src = sRaw + (AT_K + r) * ATQ_ROW;
            __half2 h[4];
            q8_to_f16(*reinterpret_cast<const uint2 *>(src + 8 * cc), __half2float(reinterpret_cast<const __half *>(src + HD)[cc >> 2]), h);
            *reinterpret_cast<int4 *>(sV + r * AT_LD + 8 * cc) = *reinterpret_cast<int4 *>(h);
        }
        __syncthreads();
        if (more) issue(kb + AT_K, 1); // (the next V streams in during the scores and P V)
        float sc[MH][4];
#pragma unroll
        for (int j = 0; j < MH; j++) sc[j][0] = sc[j][1] = sc[j][2] = sc[j][3] = 0.f;
#pragma unroll
        for (int ks = 0; ks < HD / 16; ks++) {
            const float d = __half2float(reinterpret_cast<const __half *>(kr + HD)[ks >> 1]);
            const unsigned x0 = *reinterpret_cast<const unsigned short *>(kr + 16 * ks + 2 * t4);
            const unsigned x1 = *reinterpret_cast<const unsigned short *>(kr + 16 * ks + 2 * t4 + 8);
            __half2 h0 = __floats2half2_rn(d * (float) (int8_t) x0, d * (float) (int8_t) (x0 >> 8));
            __half2 h1 = __floats2half2_rn(d * (float) (int8_t) x1, d * (float) (int8_t) (x1 >> 8));
            const unsigned b0 = *reinterpret_cast<unsigned *>(&h0), b1 = *reinterpret_cast<unsigned *>(&h1);
#pragma unroll
            for (int j = 0; j < MH; j++) {
                if (wm + 2 * j >= MT) continue;
                const __half * qa = sQ + (16 * (wm + 2 * j) + g8) * AT_LD + 2 * t4 + 16 * ks;
                mma16816(sc[j], *reinterpret_cast<const unsigned *>(qa), *reinterpret_cast<const unsigned *>(qa + 8 * AT_LD),
                         *reinterpret_cast<const unsigned *>(qa + 8), *reinterpret_cast<const unsigned *>(qa + 8 * AT_LD + 8), b0, b1);
            }
        }
        const int p = kb + 8 * wq + 2 * t4;
#pragma unroll
        for (int j = 0; j < MH; j++)
#pragma unroll
            for (int hr = 0; hr < 2; hr++) {
                if (wm + 2 * j >= MT) continue;
                if (p > qr[j][hr] || p >= k_hi) sc[j][2 * hr] = -INFINITY;
                if (p + 1 > qr[j][hr] || p + 1 >= k_hi) sc[j][2 * hr + 1] = -INFINITY;
                float mx = fmaxf(sc[j][2 * hr], sc[j][2 * hr + 1]);
                mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 1));
                mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 2));
                const int r = 16 * (wm + 2 * j) + g8 + 8 * hr;
                *reinterpret_cast<float2 *>(sS + r * ATP_LS + 8 * wq + 2 * t4) = make_float2(sc[j][2 * hr], sc[j][2 * hr + 1]);
                if (t4 == 0) sM[wq * NR + r] = mx;
            }
        __syncthreads();
        if (more) issue(kb + AT_K, 0); // (raw K free: the next K streams in during P V)
#pragma unroll
        for (int j = 0; j < MH; j++) {
            const int mt = wm + 2 * j;
            if (mt >= MT) continue;
            float al[2], mn[2], ps[2] = {0.f, 0.f};
#pragma unroll
            for (int hr = 0; hr < 2; hr++) {
                const int r = 16 * mt + g8 + 8 * hr;
                const float mx = fmaxf(fmaxf(sM[r], sM[NR + r]), fmaxf(sM[2 * NR + r], sM[3 * NR + r]));
                mn[hr] = fmaxf(m[j][hr], mx);
                al[hr] = mn[hr] == -INFINITY ? 1.f : expf(m[j][hr] - mn[hr]);
            }
            unsigned pa[AT_K / 16][4];
#pragma unroll
            for (int ks = 0; ks < AT_K / 16; ks++)
#pragma unroll
                for (int hk = 0; hk < 2; hk++) {
                    const int col = 16 * ks + 8 * hk + 2 * t4;
#pragma unroll
                    for (int hr = 0; hr < 2; hr++) {
                        const float2 sv = *reinterpret_cast<const float2 *>(sS + (16 * mt + g8 + 8 * hr) * ATP_LS + col);
                        const float p0 = mn[hr] == -INFINITY ? 0.f : expf(sv.x - mn[hr]), p1 = mn[hr] == -INFINITY ? 0.f : expf(sv.y - mn[hr]);
                        ps[hr] += p0 + p1;
                        __half2 hh = __floats2half2_rn(p0, p1);
                        pa[ks][2 * hk + hr] = *reinterpret_cast<unsigned *>(&hh);
                    }
                }
#pragma unroll
            for (int hr = 0; hr < 2; hr++) {
                ps[hr] += __shfl_xor_sync(0xffffffffu, ps[hr], 1);
                ps[hr] += __shfl_xor_sync(0xffffffffu, ps[hr], 2);
                l[j][hr] = l[j][hr] * al[hr] + ps[hr];
                m[j][hr] = mn[hr];
            }
#pragma unroll
            for (int i = 0; i < 8; i++) {
                o[j][i][0] *= al[0];
                o[j][i][1] *= al[0];
                o[j][i][2] *= al[1];
                o[j][i][3] *= al[1];
            }
#pragma unroll
            for (int ks = 0; ks < AT_K / 16; ks++)
#pragma unroll
                for (int dt2 = 0; dt2 < 4; dt2++) {
                    const __half * vb = sV + (16 * ks + (lane & 15)) * AT_LD + 64 * wq + 16 * dt2 + (lane >> 4) * 8;
                    unsigned b0, b1, b2, b3;
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
                                 : "=r"(b0), "=r"(b1), "=r"(b2), "=r"(b3)
                                 : "r"((unsigned) __cvta_generic_to_shared(vb)));
                    mma16816(o[j][2 * dt2], pa[ks][0], pa[ks][1], pa[ks][2], pa[ks][3], b0, b1);
                    mma16816(o[j][2 * dt2 + 1], pa[ks][0], pa[ks][1], pa[ks][2], pa[ks][3], b2, b3);
                }
        }
    }
#pragma unroll
    for (int j = 0; j < MH; j++)
#pragma unroll
        for (int hr = 0; hr < 2; hr++) {
            int qrow, hq, qp;
            if (wm + 2 * j >= MT || !locate(16 * (wm + 2 * j) + g8 + 8 * hr, qrow, hq, qp)) continue;
            const size_t idx = ((size_t) qrow * NH + hq) * ATT_SPL + G.soff[gi] + c;
#pragma unroll
            for (int i = 0; i < 8; i++)
                *reinterpret_cast<float2 *>(po + idx * HD + 64 * wq + 8 * i + 2 * t4) = make_float2(o[j][i][2 * hr], o[j][i][2 * hr + 1]);
            if (wq == 0 && t4 == 0) pml[idx] = make_float2(m[j][hr], l[j][hr]);
        }
}
// y2 [row][hq HD + d] = the splits of row, head hq merged
__global__ void __launch_bounds__(256) k_attn_comb(const AttnMs A, const float * po, const float2 * pml, float * y2) {
    const int t = blockIdx.x, hq = blockIdx.y, d = threadIdx.x, s = A.rseq[t];
    const int ns = A.ns[s];
    const size_t base = ((size_t) t * NH + hq) * ATT_SPL;
    float M = -INFINITY;
    for (int c = 0; c < ns; c++) M = fmaxf(M, pml[base + c].x);
    float L = 0.f, a = 0.f;
    for (int c = 0; c < ns; c++) {
        const float2 ml = pml[base + c];
        if (ml.x == -INFINITY) continue;
        const float w = expf(ml.x - M);
        L += ml.y * w;
        a += po[(base + c) * HD + d] * w;
    }
    y2[(size_t) t * VD + hq * HD + d] = L > 0.f ? a / L : 0.f;
}

} // namespace bmm
