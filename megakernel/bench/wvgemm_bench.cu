// wvgemm_bench: a flat int8 tensor-core verify GEMM ("wv": mma.m16n8k32, tokens on N) for the main-pass roles, against
// the engine's dp4a GEMV (its own rows_* / accR code, copied verbatim into wv_ref.cuh by wv_extract.py).
//
//   python3 wv_extract.py && nvcc -O3 -std=c++17 -arch=sm_86 -lineinfo -Xptxas -v wvgemm_bench.cu -o wvgemm_bench
//   ./wvgemm_bench check            exactness: T-invariance (T = 1..16 bit-identical per row) + wv / dp4a vs fp64
//                                   (exit status 1 on a failure; `make check` runs it)
//   ./wvgemm_bench time [roles] [types] [reps]
// The engine's layout / loop: -DWV_TILEMAJ=2 -DWV_PF=0; -DWV_PF=2: a 2-deep loop (q32 roles only: X16 kernels keep the
// plain loop; measured slower in the engine, which runs the plain one); -DWV_L2PF=d: an L2 prefetch d super-blocks ahead (with WV_TILEMAJ 2: whole FM units)
//
// Weights are random bytes in the engine's on-device layouts (native Q4_K / Q5_K, the load-time repacks of IQ4_XS, IQ4_NL,
// Q8_0, Q6_K, Q3_K, IQ3_S), so both kernels read exactly the bytes they would read in mk_cycle. Activations are random
// q8 rows in the engine's unit layout (QBuf: per-32 scales like q32_store, or per-16 like q16_store for ffn_down).
// Both kernels run like a mk_cycle phase: 82 blocks x 512 threads (1 block / SM, 128 registers), the block fills the
// activations into shared memory, computes, and a grid barrier ends the phase; NLAY layer copies defeat L2.
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#define CK(x)                                                                                                    \
    do {                                                                                                         \
        cudaError_t e_ = (x);                                                                                    \
        if (e_ != cudaSuccess) { fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); exit(1); } \
    } while (0)

constexpr int NT = 512, NW = 16;
#include "wv_ref.cuh"

// ------------------------------------------------------------------------------------------------------------------
// grid barrier (one cooperative launch per measurement; the counter is cleared before each launch)
__device__ unsigned g_bar;
__device__ __forceinline__ void gsync(unsigned & epoch) {
    __syncthreads();
    if (threadIdx.x == 0) {
        epoch += gridDim.x;
        asm volatile("red.release.gpu.global.add.u32 [%0], 1;" ::"l"(&g_bar) : "memory");
        unsigned v;
        do { asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(&g_bar) : "memory"); } while (v < epoch);
    }
    __syncthreads();
}
__device__ __forceinline__ void cp16(void * dst, const void * src) {
    const unsigned d = (unsigned) __cvta_generic_to_shared(dst);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(d), "l"(src) : "memory");
}
__device__ __forceinline__ void cp_wait() { asm volatile("cp.async.commit_group;\ncp.async.wait_all;" ::: "memory"); }

// Activations in the engine's unit layout (QBuf): unit u = (super-block b, 64-group p, half h) holds A = elements
// 256b + 64p + 16h + [0, 16) and B = 256b + 64p + 32 + 16h + [0, 16); meta per unit: half2 (scale of A / B), short2
// (int sums of A / B).
struct QB {
    const int8_t * xa;
    const int8_t * xb;
    const uint2 * meta;
    int U; // units per token row
};

// ==================================================================================================================
// the engine's GEMV: gemv_pairs (2-row warp items over all warps) with the engine's mrows<T, IT, 2, TYPE>
// ==================================================================================================================
struct DpArgs {
    const Mat * mats;
    int nlay;
    QB q;
    float * out;
    int M;
    float * park; // k_dp4a_k2: per-lane partial sums between the two K slices [M / 2][2][T][32]
};
template <int T, int IT, int TYPE>
__global__ void __launch_bounds__(NT, 1) k_dp4a(DpArgs A) {
    extern __shared__ __align__(16) uint8_t sm[];
    constexpr int U = 32 * IT;
    SX sx;
    sx.xa = reinterpret_cast<int4 *>(sm);
    sx.xb = sx.xa + T * U;
    sx.xm = reinterpret_cast<uint2 *>(sx.xb + T * U);
    sx.U = U;
    const int lane = threadIdx.x & 31;
    unsigned epoch = 0;
    for (int L = 0; L < A.nlay; L++) {
        for (int i = threadIdx.x; i < T * U; i += NT) {
            cp16(sx.xa + i, A.q.xa + (size_t) i * 16);
            cp16(sx.xb + i, A.q.xb + (size_t) i * 16);
        }
        for (int i = threadIdx.x; i < T * U / 2; i += NT) cp16(sx.xm + 2 * i, A.q.meta + 2 * i);
        cp_wait();
        __syncthreads();
        const Mat m = A.mats[L];
        for (int it = gwarp(); it < A.M / 2; it += nwarps()) {
            float a[2][T];
            zero_acc<T, 2>(a);
            mrows<T, IT, 2, TYPE>(m, 2 * it, sx, a);
            wsum_acc<T, 2>(a);
#pragma unroll
            for (int t = 0; t < T; t++)
                if ((lane >> 1) == t) A.out[(size_t) t * A.M + 2 * it + (lane & 1)] = (lane & 1) ? a[1][t] : a[0][t];
        }
        gsync(epoch);
    }
}

// The activations of all K do not fit at this T (ffn_down, T >= 5): two K slices of ITS0 = ceil(IT / 2) and IT - ITS0
// lane units, each filled in turn; the per-lane partial sums of a warp's items are parked in global memory (L2) between
// the slices (the engine's T = 5 resid32 parks them in shared memory). Same arithmetic order per lane as k_dp4a.
template <int T, int ITS, int TYPE>
__device__ __forceinline__ void dp_slice(const DpArgs & A, const Mat & m, int u0, bool first, bool last, uint8_t * sm) {
    constexpr int US = 32 * ITS;
    SX sx;
    sx.xa = reinterpret_cast<int4 *>(sm);
    sx.xb = sx.xa + T * US;
    sx.xm = reinterpret_cast<uint2 *>(sx.xb + T * US);
    sx.U = US;
    __syncthreads(); // the previous slice's readers are done
    for (int i = threadIdx.x; i < T * US; i += NT) {
        const int t = i / US, u = u0 + (i - t * US);
        cp16(sx.xa + i, A.q.xa + ((size_t) t * A.q.U + u) * 16);
        cp16(sx.xb + i, A.q.xb + ((size_t) t * A.q.U + u) * 16);
    }
    for (int i = threadIdx.x; i < T * US / 2; i += NT) {
        const int t = (2 * i) / US, u = u0 + (2 * i - t * US);
        cp16(sx.xm + 2 * i, A.q.meta + (size_t) t * A.q.U + u);
    }
    cp_wait();
    __syncthreads();
    const int lane = threadIdx.x & 31;
    for (int it = gwarp(); it < A.M / 2; it += nwarps()) {
        float a[2][T];
        float * pk = A.park + (size_t) it * 2 * T * 32 + lane;
        if (first) zero_acc<T, 2>(a);
        else {
#pragma unroll
            for (int k = 0; k < 2; k++)
#pragma unroll
                for (int t = 0; t < T; t++) a[k][t] = __ldcg(pk + (k * T + t) * 32);
        }
        mrows<T, ITS, 2, TYPE>(m, 2 * it, sx, a, u0);
        if (last) {
            wsum_acc<T, 2>(a);
#pragma unroll
            for (int t = 0; t < T; t++)
                if ((lane >> 1) == t) A.out[(size_t) t * A.M + 2 * it + (lane & 1)] = (lane & 1) ? a[1][t] : a[0][t];
        } else {
#pragma unroll
            for (int k = 0; k < 2; k++)
#pragma unroll
                for (int t = 0; t < T; t++) __stcg(pk + (k * T + t) * 32, a[k][t]);
        }
    }
}
template <int T, int IT, int TYPE>
__global__ void __launch_bounds__(NT, 1) k_dp4a_k2(DpArgs A) {
    extern __shared__ __align__(16) uint8_t sm[];
    constexpr int I0 = (IT + 1) / 2, I1 = IT - I0;
    unsigned epoch = 0;
    for (int L = 0; L < A.nlay; L++) {
        const Mat m = A.mats[L];
        dp_slice<T, I0, TYPE>(A, m, 0, true, false, sm);
        dp_slice<T, I1, TYPE>(A, m, 32 * I0, false, true, sm);
        gsync(epoch);
    }
}

// ==================================================================================================================
// wv: int8 mma.m16n8k32, a warp computes a 16-row tile (rows g, g + 8 per thread) x 8 tokens per MMA over a K range.
//
// K permutation (the same for A and B, so the integer dot is unchanged): MMA k-slot 4 tq + i <-> element 8 tq + i and
// k-slot 16 + 4 tq + i <-> element 8 tq + 4 + i of the 32-element block, i.e. each thread supplies 8 consecutive
// elements: one 8-byte load from the weight row and one 8-byte LDS of the activations.
// Integer -> float without I2F: the accumulator input is MAGIC = 0x4B400000 (12582912.0f), so the MMA returns
// MAGIC + S, whose float value is exactly 12582912 + S (|S| < 2^22); fmaf(dx, that, -12582912 dx) = dx * S rounded once
// (the product -12582912 dx is exact).
// Per-16 scales (Q6_K / Q3_K weights, or the q16 activations of ffn_down): "split" MMAs. Threads tq 0-1 hold chunk 0
// (elements 0-15), tq 2-3 chunk 1. B column g carries token (g & 3) of the group, and only the k-slots of chunk g >> 2
// (the other slots are zero), so C column n = the 16-element dot of token n & 3 over chunk n >> 2: thread tq owns
// chunk tq >> 1 of tokens 2 tq & 3, + 1. The two chunk sums of a token meet in one shuffle after the K loop.
// Offset types (Q6_K, Q3_K, IQ3_S feed q + 32 like the dp4a path): the accumulator input is MAGIC - 32 * qsum(chunk),
// so the MMA returns MAGIC + sum (q + 32 - 32) x = the signed dot, with no float min term.
// ==================================================================================================================
#ifndef WV_L2PF
#define WV_L2PF 0 // L2 prefetch distance in super-blocks (0: off)
#endif
constexpr int MAGIC = 0x4B400000;
constexpr float FMAGIC = 12582912.0f;

__device__ __forceinline__ void mma16832(int (&d)[4], const int (&a)[4], int b0, int b1, const int (&c)[4]) {
    asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};"
        : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(c[0]), "r"(c[1]), "r"(c[2]), "r"(c[3]));
}
__device__ __forceinline__ void mma16816(int (&d)[4], int a0, int a1, int b0, const int (&c)[4]) {
    asm("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%8,%9,%10};"
        : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
        : "r"(a0), "r"(a1), "r"(b0), "r"(c[0]), "r"(c[1]), "r"(c[2]), "r"(c[3]));
}
// float(byte k of w) + 2^23 (k compile-time after unrolling)
__device__ __forceinline__ float mag23(unsigned w, int k) { return __int_as_float(__byte_perm(w, 0x4b000000u, 0x7540u | k)); }

// Per weight type: Raw = one super-block of one row as this thread loads it; Sc = its decoded per-super-block scale
// state; frag(s) = the two int8x4 words of elements 8 tq .. 8 tq + 7 of sub-block s; wsc(s, h) = the float scale of
// sub-block s (chunk h for per-16 types) and, for MIN == 1, the min coefficient (weight = fA q - gA).
// MIN: 0 none, 1 per-row min (Q4_K / Q5_K: one FFMA per output), 2 constant offset 32 (via the accumulator input).
template <int TYPE> struct WV;

// weight loads of the wv kernel: the engine's evict-first non-coherent loads, optionally with the L2::256B fetch-size
// hint (WV_LD256=1: an L2 miss fetches the aligned 256 bytes around the access, a prefetch of the row's next block)
#ifndef WV_LD256
#define WV_LD256 0
#endif
#ifndef WV_NOPOL
#define WV_NOPOL 0 // 1: weight loads without the evict-first L2 policy
#endif
#if WV_NOPOL
__device__ __forceinline__ int4 wld4(const void * p) {
    int4 r;
    asm("ld.global.nc.L1::no_allocate.v4.s32 {%0,%1,%2,%3}, [%4];" : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
    return r;
}
__device__ __forceinline__ int2 wld2(const void * p) {
    int2 r;
    asm("ld.global.nc.L1::no_allocate.v2.s32 {%0,%1}, [%2];" : "=r"(r.x), "=r"(r.y) : "l"(p));
    return r;
}
__device__ __forceinline__ int wld1(const void * p) {
    int r;
    asm("ld.global.nc.L1::no_allocate.s32 %0, [%1];" : "=r"(r) : "l"(p));
    return r;
}
#elif WV_LD256
__device__ __forceinline__ int4 wld4(const void * p) {
    int4 r;
    asm("ld.global.nc.L1::no_allocate.L2::cache_hint.L2::256B.v4.s32 {%0,%1,%2,%3}, [%4], %5;"
        : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p), "l"(l2_evict_first()));
    return r;
}
__device__ __forceinline__ int2 wld2(const void * p) {
    int2 r;
    asm("ld.global.nc.L1::no_allocate.L2::cache_hint.L2::256B.v2.s32 {%0,%1}, [%2], %3;" : "=r"(r.x), "=r"(r.y) : "l"(p), "l"(l2_evict_first()));
    return r;
}
__device__ __forceinline__ int wld1(const void * p) {
    int r;
    asm("ld.global.nc.L1::no_allocate.L2::cache_hint.L2::256B.s32 %0, [%1], %2;" : "=r"(r) : "l"(p), "l"(l2_evict_first()));
    return r;
}
#else
__device__ __forceinline__ int4 wld4(const void * p) { return ldw4(p); }
__device__ __forceinline__ int2 wld2(const void * p) { return ldw2(p); }
__device__ __forceinline__ int wld1(const void * p) { return ldw1(p); }
#endif
__device__ __forceinline__ void pf_l2n(const void * p) { asm volatile("prefetch.global.L2 [%0];" ::"l"(p)); }

// block idx (of BS bytes) of row `row`: the engine's row-major layout, or (WV_TILEMAJ, a timing experiment: the same
// bytes at other addresses) tile-major: the 16 rows of a 16-row tile stored block-interleaved, unit (tile, idx) = 16 x BS
// contiguous bytes
#ifndef WV_TILEMAJ
#define WV_TILEMAJ 0
#endif
__device__ __forceinline__ const uint8_t * wblk(const Mat & m, int row, int idx, int BS) {
#if WV_TILEMAJ
    return m.w + (((size_t) (row >> 4) * (m.rb / BS) + idx) * 16 + (row & 15)) * BS;
#else
    return m.w + (size_t) row * m.rb + (size_t) idx * BS;
#endif
}
// the meta stream (Mat::d) likewise, in MB-byte entries idx
__device__ __forceinline__ const uint8_t * wmeta(const Mat & m, int row, int idx, int MB) {
    const uint8_t * d = reinterpret_cast<const uint8_t *>(m.d);
#if WV_TILEMAJ
    return d + (((size_t) (row >> 4) * (2 * m.nd / MB) + idx) * 16 + (row & 15)) * MB;
#else
    return d + (size_t) row * 2 * m.nd + (size_t) idx * MB;
#endif
}
template <> struct WV<T_Q4K> { // native 144 B: d, dmin | scales[12] | qs[128]
    static constexpr bool S16 = false;
    static constexpr int MIN = 1;
    struct Raw { int4 h; int2 q[4]; };
    struct Sc { unsigned s0, s1, m0, m1; float d, nd, dm, ndm; };
    static __device__ __forceinline__ void load(Raw & r, const Mat & m, int row, int sb, int tq) {
        const uint8_t * blk = wblk(m, row, sb, 144);
        r.h = wld4(blk);
#pragma unroll
        for (int p = 0; p < 4; p++) r.q[p] = wld2(blk + 16 + 32 * p + 8 * tq);
    }
    static __device__ __forceinline__ void scales(const Raw & r, Sc & s) {
        const float2 dm = __half22float2(*reinterpret_cast<const __half2 *>(&r.h.x));
        const unsigned y = r.h.y, z = r.h.z, w = r.h.w;
        s.s0 = y & 0x3f3f3f3fu;
        s.m0 = z & 0x3f3f3f3fu;
        s.s1 = (w & 0x0f0f0f0fu) | ((y >> 2) & 0x30303030u);
        s.m1 = ((w >> 4) & 0x0f0f0f0fu) | ((z >> 2) & 0x30303030u);
        s.d = dm.x; s.nd = -8388608.0f * dm.x; s.dm = dm.y; s.ndm = -8388608.0f * dm.y;
    }
    static __device__ __forceinline__ void wsc(const Sc & s, int k, int, float & fA, float & gA) {
        fA = fmaf(s.d, mag23(k < 4 ? s.s0 : s.s1, k & 3), s.nd);
        gA = fmaf(s.dm, mag23(k < 4 ? s.m0 : s.m1, k & 3), s.ndm);
    }
    static __device__ __forceinline__ void frag(const Raw & r, int k, int, int & w0, int & w1) {
        const int sh = 4 * (k & 1);
        w0 = (r.q[k >> 1].x >> sh) & M4;
        w1 = (r.q[k >> 1].y >> sh) & M4;
    }
};

template <> struct WV<T_Q5K> { // native 176 B: d, dmin | scales[12] | qh[32] | qs[128]
    static constexpr bool S16 = false;
    static constexpr int MIN = 1;
    struct Raw { int4 h; int2 qh; int2 q[4]; };
    using Sc = WV<T_Q4K>::Sc;
    static __device__ __forceinline__ void load(Raw & r, const Mat & m, int row, int sb, int tq) {
        const uint8_t * blk = wblk(m, row, sb, 176);
        r.h = wld4(blk);
        r.qh = wld2(blk + 16 + 8 * tq);
#pragma unroll
        for (int p = 0; p < 4; p++) r.q[p] = wld2(blk + 48 + 32 * p + 8 * tq);
    }
    static __device__ __forceinline__ void scales(const Raw & r, Sc & s) {
        WV<T_Q4K>::Raw t;
        t.h = r.h;
        WV<T_Q4K>::scales(t, s);
    }
    static __device__ __forceinline__ void wsc(const Sc & s, int k, int h, float & fA, float & gA) { WV<T_Q4K>::wsc(s, k, h, fA, gA); }
    static __device__ __forceinline__ void frag(const Raw & r, int k, int, int & w0, int & w1) {
        const int sh = 4 * (k & 1);
        w0 = ((r.q[k >> 1].x >> sh) & M4) | ((((unsigned) r.qh.x >> k) & 0x01010101u) << 4);
        w1 = ((r.q[k >> 1].y >> sh) & M4) | ((((unsigned) r.qh.y >> k) & 0x01010101u) << 4);
    }
};

template <> struct WV<T_IQ4X> { // repacked pairs of blocks, 272 B: hdr0 | hdr1 (d | scales_h | scales_l[4]) | qs0 | qs1
    static constexpr bool S16 = false;
    static constexpr int MIN = 0;
    struct Raw { int2 h; int2 q[4]; };
    struct Sc { unsigned sl, sh; float d, nd; };
    static __device__ __forceinline__ void load(Raw & r, const Mat & m, int row, int sb, int tq) {
        const uint8_t * pr = wblk(m, row, sb >> 1, 272);
        r.h = wld2(pr + 8 * (sb & 1));
#pragma unroll
        for (int p = 0; p < 4; p++) r.q[p] = wld2(pr + 16 + 128 * (sb & 1) + 32 * p + 8 * tq);
    }
    static __device__ __forceinline__ void scales(const Raw & r, Sc & s) {
        s.d = __half2float(__ushort_as_half((unsigned short) (r.h.x & 0xffff)));
        s.nd = -FMAGIC * s.d;
        s.sh = (unsigned) r.h.x >> 16;
        s.sl = (unsigned) r.h.y;
    }
    static __device__ __forceinline__ void wsc(const Sc & s, int k, int, float & fA, float & gA) {
        const int ls = (int) (((s.sl >> (4 * k)) & 0xfu) | (((s.sh >> (2 * k)) & 3u) << 4)); // k < 8: nibble k of sl
        fA = fmaf(s.d, __int_as_float(MAGIC - 32 + ls), s.nd);
        gA = 0.f;
    }
    static __device__ __forceinline__ void frag(const Raw & r, int k, int, int & w0, int & w1) {
        int lo, hi;
        iq4_lut((unsigned) r.q[k >> 1].x, IQ4_T0, IQ4_T1, IQ4_T2, IQ4_T3, lo, hi);
        w0 = (k & 1) ? hi : lo;
        iq4_lut((unsigned) r.q[k >> 1].y, IQ4_T0, IQ4_T1, IQ4_T2, IQ4_T3, lo, hi);
        w1 = (k & 1) ? hi : lo;
    }
};

template <> struct WV<T_IQ4NL> { // repacked 144 B: d[8] (fp16, one per 32) | qs[128] in Q4_K order
    static constexpr bool S16 = false;
    static constexpr int MIN = 0;
    struct Raw { int4 h; int2 q[4]; };
    struct Sc { int4 h; };
    static __device__ __forceinline__ void load(Raw & r, const Mat & m, int row, int sb, int tq) {
        const uint8_t * blk = wblk(m, row, sb, 144);
        r.h = wld4(blk);
#pragma unroll
        for (int p = 0; p < 4; p++) r.q[p] = wld2(blk + 16 + 32 * p + 8 * tq);
    }
    static __device__ __forceinline__ void scales(const Raw & r, Sc & s) { s.h = r.h; }
    static __device__ __forceinline__ void wsc(const Sc & s, int k, int, float & fA, float & gA) {
        const int w = k == 0 || k == 1 ? s.h.x : k == 2 || k == 3 ? s.h.y : k == 4 || k == 5 ? s.h.z : s.h.w;
        fA = __half2float(__ushort_as_half((unsigned short) ((unsigned) w >> (16 * (k & 1)))));
        gA = 0.f;
    }
    static __device__ __forceinline__ void frag(const Raw & r, int k, int, int & w0, int & w1) {
        int lo, hi;
        iq4_lut((unsigned) r.q[k >> 1].x, IQ4_T0, IQ4_T1, IQ4_T2, IQ4_T3, lo, hi);
        w0 = (k & 1) ? hi : lo;
        iq4_lut((unsigned) r.q[k >> 1].y, IQ4_T0, IQ4_T1, IQ4_T2, IQ4_T3, lo, hi);
        w1 = (k & 1) ? hi : lo;
    }
};

template <> struct WV<T_Q8_0> { // repacked: per unit 32 B (A 16 | B 16); fp16 d per natural 32-block in Mat::d
    static constexpr bool S16 = false;
    static constexpr int MIN = 0;
    struct Raw { int4 d; int2 q[8]; };
    struct Sc { int4 d; };
    static __device__ __forceinline__ void load(Raw & r, const Mat & m, int row, int sb, int tq) {
        const uint8_t * rp = wblk(m, row, sb, 256);
        r.d = wld4(wmeta(m, row, sb, 16));
#pragma unroll
        for (int k = 0; k < 8; k++) r.q[k] = wld2(rp + 32 * (2 * (k >> 1) + (tq >> 1)) + 16 * (k & 1) + 8 * (tq & 1));
    }
    static __device__ __forceinline__ void scales(const Raw & r, Sc & s) { s.d = r.d; }
    static __device__ __forceinline__ void wsc(const Sc & s, int k, int, float & fA, float & gA) {
        WV<T_IQ4NL>::Sc t;
        t.h = s.d;
        WV<T_IQ4NL>::wsc(t, k, 0, fA, gA);
    }
    static __device__ __forceinline__ void frag(const Raw & r, int k, int, int & w0, int & w1) { w0 = r.q[k].x; w1 = r.q[k].y; }
};

// Q6_K repack: values 192 B per block: qlo[128] (Q4_K-order nibbles) | qhi[64] (8 B per unit u' = 2p + h: A's highs | B's,
// byte m bits 2j = element 4j + m of the 16-chunk); meta (Mat::d) 18 B per block, in pairs: sc[16] even | sc[16] odd |
// d even | d odd. Values are q + 32 in [0, 63] (offset 32).
template <> struct WV<T_Q6K> {
    static constexpr bool S16 = true;
    static constexpr int MIN = 2;
    struct Raw { int sc[4]; int dd; int2 q[4]; int2 hb[4]; };
    struct Sc { int sc[4]; float d, nd; };
    static __device__ __forceinline__ void load(Raw & r, const Mat & m, int row, int sb, int tq) {
        const uint8_t * blk = wblk(m, row, sb, Q6_BLK);
        const uint8_t * mp = wmeta(m, row, sb >> 1, 2 * Q6_META);
#pragma unroll
        for (int k = 0; k < 4; k++) r.sc[k] = wld1(mp + 16 * (sb & 1) + 4 * k); // (meta is only 4-byte aligned)
        r.dd = wld1(mp + 32);
        r.dd = (int) ((unsigned) r.dd >> (16 * (sb & 1)));
#pragma unroll
        for (int p = 0; p < 4; p++) {
            r.q[p] = wld2(blk + 32 * p + 8 * tq);
            r.hb[p] = wld2(blk + 128 + 8 * (2 * p + (tq >> 1)));
        }
    }
    static __device__ __forceinline__ void scales(const Raw & r, Sc & s) {
#pragma unroll
        for (int k = 0; k < 4; k++) s.sc[k] = r.sc[k];
        s.d = __half2float(__ushort_as_half((unsigned short) (r.dd & 0xffff)));
        s.nd = -FMAGIC * s.d;
    }
    // chunk c = 4p + 2 part + h of the block: byte c of sc[16] (h = tq >> 1 is not compile-time: select the word pair)
    static __device__ __forceinline__ void wsc(const Sc & s, int k, int h, float & fA, float & gA) {
        const int c0 = 4 * (k >> 1) + 2 * (k & 1); // even: bytes c0, c0 + 1 lie in word c0 >> 2
        const unsigned w = (unsigned) s.sc[c0 >> 2] >> (8 * h);
        const int v = (int) (signed char) (w >> (8 * (c0 & 3)));
        fA = fmaf(s.d, __int_as_float(MAGIC + v), s.nd);
        gA = 0.f;
    }
    static __device__ __forceinline__ void frag(const Raw & r, int k, int tq, int & w0, int & w1) {
        const int sh = 4 * (k & 1);
        const unsigned hs = (k & 1) ? (unsigned) r.hb[k >> 1].y : (unsigned) r.hb[k >> 1].x;
        const int j0 = 2 * (tq & 1);
        w0 = ((r.q[k >> 1].x >> sh) & M4) | (int) (((hs >> (2 * j0)) & 0x03030303u) << 4);
        w1 = ((r.q[k >> 1].y >> sh) & M4) | (int) (((hs >> (2 * j0 + 2)) & 0x03030303u) << 4);
    }
};

// Q3_K repack: values 96 B per block: unit u' = 2p + h: 2-bit lows at 8u' (A word | B word; byte m bits 2j = element 4j + m),
// highs at 64 + 4u' (byte m: bit j = A element 4j + m, bit 4 + j = B); meta as Q6_K (sc - 32 as int8, d). Value fed:
// u3 + 28 = the Q6_K value (offset 32).
template <> struct WV<T_Q3K> {
    static constexpr bool S16 = true;
    static constexpr int MIN = 2;
    struct Raw { int sc[4]; int dd; int2 lo[4]; int hi[4]; };
    using Sc = WV<T_Q6K>::Sc;
    static __device__ __forceinline__ void load(Raw & r, const Mat & m, int row, int sb, int tq) {
        const uint8_t * blk = wblk(m, row, sb, Q3_BLK);
        const uint8_t * mp = wmeta(m, row, sb >> 1, 2 * Q6_META);
#pragma unroll
        for (int k = 0; k < 4; k++) r.sc[k] = wld1(mp + 16 * (sb & 1) + 4 * k);
        r.dd = (int) ((unsigned) wld1(mp + 32) >> (16 * (sb & 1)));
#pragma unroll
        for (int p = 0; p < 4; p++) {
            const int u = 2 * p + (tq >> 1);
            r.lo[p] = wld2(blk + 8 * u);
            r.hi[p] = wld1(blk + 64 + 4 * u);
        }
    }
    static __device__ __forceinline__ void scales(const Raw & r, Sc & s) {
#pragma unroll
        for (int k = 0; k < 4; k++) s.sc[k] = r.sc[k];
        s.d = __half2float(__ushort_as_half((unsigned short) (r.dd & 0xffff)));
        s.nd = -FMAGIC * s.d;
    }
    static __device__ __forceinline__ void wsc(const Sc & s, int k, int h, float & fA, float & gA) { WV<T_Q6K>::wsc(s, k, h, fA, gA); }
    static __device__ __forceinline__ void frag(const Raw & r, int k, int tq, int & w0, int & w1) {
        const unsigned x = (k & 1) ? (unsigned) r.lo[k >> 1].y : (unsigned) r.lo[k >> 1].x, hh = (unsigned) r.hi[k >> 1];
        const int j0 = 2 * (tq & 1), hs = 4 * (k & 1);
        w0 = (int) ((((x >> (2 * j0)) & 0x03030303u) | (((hh >> (j0 + hs)) & 0x01010101u) << 2)) + 0x1c1c1c1cu);
        w1 = (int) ((((x >> (2 * j0 + 2)) & 0x03030303u) | (((hh >> (j0 + 1 + hs)) & 0x01010101u) << 2)) + 0x1c1c1c1cu);
    }
};

// IQ3_S repack: values 96 B per block: unit u' = 2p + h: the low bytes of A's 4 grid indices then B's at 8u', signs at
// 64 + 4u' (byte m: bit c = A element 4c + m, bit 4 + c = B); meta 16 B per block: qh[8] (byte u': index bit 8, A in
// bits 0-3, B in 4-7) | scales[4] (byte p: sub-block 2p | 2p + 1 << 4) | fp16 d | pad. Value fed: 32 +- grid (offset 32).
template <> struct WV<T_IQ3S> {
    static constexpr bool S16 = false;
    static constexpr int MIN = 2;
    struct Raw { int2 qh; int2 sd; int2 ix[4]; int sg[4]; };
    struct Sc { unsigned sc; float d, nd; };
    static __device__ __forceinline__ void load(Raw & r, const Mat & m, int row, int sb, int tq) {
        const uint8_t * blk = wblk(m, row, sb, Q3_BLK);
        const uint8_t * mp = wmeta(m, row, sb, IQ3S_META);
        r.qh = wld2(mp);
        r.sd = wld2(mp + 8);
#pragma unroll
        for (int p = 0; p < 4; p++) {
            const int u = 2 * p + (tq >> 1);
            r.ix[p] = wld2(blk + 8 * u);
            r.sg[p] = wld1(blk + 64 + 4 * u);
        }
    }
    static __device__ __forceinline__ void scales(const Raw & r, Sc & s) {
        s.sc = (unsigned) r.sd.x;
        s.d = __half2float(__ushort_as_half((unsigned short) (r.sd.y & 0xffff)));
        s.nd = -FMAGIC * s.d;
    }
    static __device__ __forceinline__ void wsc(const Sc & s, int k, int, float & fA, float & gA) {
        const int v = 1 + 2 * (int) ((s.sc >> (4 * k)) & 0xfu); // nibble k = sub-block k
        fA = fmaf(s.d, __int_as_float(MAGIC + v), s.nd);
        gA = 0.f;
    }
    // (the qh byte of unit u' is per thread: u' = 2p + (tq >> 1))
    static __device__ __forceinline__ void frag(const Raw & r, int k, int tq, int & w0, int & w1) {
        const int p = k >> 1, part = k & 1, u = 2 * p + (tq >> 1);
        const unsigned qhw = (unsigned) (u < 4 ? r.qh.x : r.qh.y) >> (8 * (u & 3));
        const unsigned x = part ? (unsigned) r.ix[p].y : (unsigned) r.ix[p].x, sv = (unsigned) r.sg[p];
        const int c0 = 2 * (tq & 1), hs = 4 * part;
        int w[2];
#pragma unroll
        for (int i = 0; i < 2; i++) {
            const int c = c0 + i;
            const unsigned g = iq3s_grid(((x >> (8 * c)) & 0xffu) | (((qhw >> (c + hs)) & 1u) << 8));
            const unsigned mk = ((sv >> (c + hs)) & 0x01010101u) * 0xffu;
            w[i] = (int) (0x20202020u + (g ^ mk) - mk);
        }
        w0 = w[0];
        w1 = w[1];
    }
};

// shared-memory activations of the block's K chunk: xq[token][ld] int8 in natural element order (ld = Kc + 32, so the
// 8-byte B loads of a half-warp hit distinct banks); md[(j * NG + n) * 4 + tq] = {dx_a, -12582912 dx_a, dx_b, -..dx_b}
// for the two tokens (a, b) of the thread's C columns in group n and 32-block j (split mode: of its chunk); mx = the
// matching (sq_a, sq_b) = dx * qsum for MIN 1, or (MAGIC - 32 qsum) bits for MIN 2.
struct XSm {
    const int8_t * xq;
    int ld;
    const float4 * md;
    const float2 * mx;
    const float4 * mx4; // K16: per (32-block j, group n, tq): MAGIC - 32 qsum of (token a, chunk 0), (b, 0), (a, 1), (b, 1)
};

// WV_Q6FM: Q6_K in a fragment-ordered unit with the meta inline (wv_tile_q6f); 0 = off (TM / row-major),
// 1 = split m16n8k32 at every T, 2 = m16n8k16 per 16-chunk at every T, 3 = split at T <= 4, k16 at T > 4.
// All forms are bit-identical to each other and to the split path on the engine layouts (same integer chunk sums, same
// float sequence per (row, token, chunk), the two chunk accumulators added last).
#ifndef WV_Q6FM
#define WV_Q6FM 0
#endif
template <int TYPE, int T, bool X16>
struct Cfg {
    static constexpr bool K16 = TYPE == T_Q6K && (WV_Q6FM == 2 || (WV_Q6FM == 3 && T > 4));
    static constexpr bool SPLIT = !K16 && (WV<TYPE>::S16 || X16);
    static constexpr int TPG = SPLIT ? 4 : 8;
    static constexpr int NG = (T + TPG - 1) / TPG;
    static constexpr int NTOKP = NG * TPG;
};

// one super-block (8 sub-blocks) of the tile from the thread's raw rows r0 (row g) and r1 (row g + 8); j0 = 8 * the
// super-block's index within the shared-memory chunk
template <int TYPE, int T, bool X16>
__device__ __forceinline__ void wv_sb(const typename WV<TYPE>::Raw & r0, const typename WV<TYPE>::Raw & r1, int j0,
                                      const XSm & xs, const int8_t * xrow, bool bon, float (&acc)[Cfg<TYPE, T, X16>::NG][4]) {
    using W = WV<TYPE>;
    using C = Cfg<TYPE, T, X16>;
    const int tq = threadIdx.x & 3, h = tq >> 1;
    typename W::Sc s0, s1;
    W::scales(r0, s0);
    W::scales(r1, s1);
#pragma unroll
    for (int k = 0; k < 8; k++) {
        int a[4];
        W::frag(r0, k, tq, a[0], a[2]);
        W::frag(r1, k, tq, a[1], a[3]);
        float fA0, gA0, fA1, gA1;
        W::wsc(s0, k, h, fA0, gA0);
        W::wsc(s1, k, h, fA1, gA1);
        const int j = j0 + k;
#pragma unroll
        for (int n = 0; n < C::NG; n++) {
            int2 bb = make_int2(0, 0);
            if (bon) bb = *reinterpret_cast<const int2 *>(xrow + C::TPG * n * xs.ld + 32 * j);
            const float4 md = xs.md[(j * C::NG + n) * 4 + tq];
            int ci[4] = {MAGIC, MAGIC, MAGIC, MAGIC};
            float2 mx = make_float2(0.f, 0.f);
            if (W::MIN) mx = xs.mx[(j * C::NG + n) * 4 + tq];
            if (W::MIN == 2) {
                ci[0] = ci[2] = __float_as_int(mx.x);
                ci[1] = ci[3] = __float_as_int(mx.y);
            }
            int cc[4];
            mma16832(cc, a, bb.x, bb.y, ci);
            acc[n][0] = fmaf(fA0, fmaf(md.x, __int_as_float(cc[0]), md.y), acc[n][0]);
            acc[n][1] = fmaf(fA0, fmaf(md.z, __int_as_float(cc[1]), md.w), acc[n][1]);
            acc[n][2] = fmaf(fA1, fmaf(md.x, __int_as_float(cc[2]), md.y), acc[n][2]);
            acc[n][3] = fmaf(fA1, fmaf(md.z, __int_as_float(cc[3]), md.w), acc[n][3]);
            if (W::MIN == 1) {
                acc[n][0] = fmaf(-gA0, mx.x, acc[n][0]);
                acc[n][1] = fmaf(-gA0, mx.y, acc[n][1]);
                acc[n][2] = fmaf(-gA1, mx.x, acc[n][2]);
                acc[n][3] = fmaf(-gA1, mx.y, acc[n][3]);
            }
        }
    }
}

#ifndef WV_SBU
#define WV_SBU 1 // super-blocks loaded per batch (PF = 0 only)
#endif
#ifndef WV_TILEMAJ
#define WV_TILEMAJ 0
#endif

// WV_TILEMAJ == 2 (timing experiment, Q4_K / Q5_K / IQ4_XS): tile-major and fragment-ordered. Unit (tile, sb) = 16 x BS
// contiguous bytes: headers paired per g (row g | row g + 8) at 32 g (Q4_K / Q5_K, 16 B each) or 16 g (IQ4_XS, 8 B each);
// Q5_K qh, then the 4 qs groups, each 512 B with lane l = 4 g + tq at 16 l = {row g's 8 B | row g + 8's 8 B}. Every
// lane load is 16 B and every warp load instruction reads contiguous bytes.
template <int TYPE>
__device__ __forceinline__ void wv_load2(typename WV<TYPE>::Raw & r0, typename WV<TYPE>::Raw & r1, const Mat & m, int row0, int sb,
                                         int g, int tq) {
#if WV_TILEMAJ == 2
    if constexpr (TYPE == T_Q4K || TYPE == T_Q5K || TYPE == T_IQ4X) {
        constexpr int BS = TYPE == T_Q4K ? 144 : TYPE == T_Q5K ? 176 : 136;
        const uint8_t * u = m.w + ((size_t) (row0 >> 4) * (m.K / 256) + sb) * 16 * BS;
        const int l = 4 * g + tq;
        int qo;
        if constexpr (TYPE == T_IQ4X) {
            const int4 h = wld4(u + 16 * g);
            r0.h = make_int2(h.x, h.y);
            r1.h = make_int2(h.z, h.w);
            qo = 128;
        } else {
            r0.h = wld4(u + 32 * g);
            r1.h = wld4(u + 32 * g + 16);
            qo = 256;
        }
        if constexpr (TYPE == T_Q5K) {
            const int4 v = wld4(u + 256 + 16 * l);
            r0.qh = make_int2(v.x, v.y);
            r1.qh = make_int2(v.z, v.w);
            qo = 768;
        }
#pragma unroll
        for (int p = 0; p < 4; p++) {
            const int4 v = wld4(u + qo + 512 * p + 16 * l);
            r0.q[p] = make_int2(v.x, v.y);
            r1.q[p] = make_int2(v.z, v.w);
        }
        return;
    }
#endif
    WV<TYPE>::load(r0, m, row0 + g, sb, tq);
    WV<TYPE>::load(r1, m, row0 + g + 8, sb, tq);
}
#ifndef WV_SPS
#define WV_SPS 1
#endif
#ifndef WV_RH
#define WV_RH 16
#endif
#ifndef WV_MODE
#define WV_MODE 0 // diagnostics: 0 normal, 1 loads only (XOR of the raw words), 2 compute only (raw loaded once)
#endif
template <class R> __device__ __forceinline__ int raw_xor(const R & r) {
    const int * p = reinterpret_cast<const int *>(&r);
    int x = 0;
#pragma unroll
    for (int i = 0; i < (int) (sizeof(R) / 4); i++) x ^= p[i];
    return x;
}
template <class R> __device__ __forceinline__ void raw_opaque(R & r) {
    int * p = reinterpret_cast<int *>(&r);
#pragma unroll
    for (int i = 0; i < (int) (sizeof(R) / 4); i++) asm volatile("" : "+r"(p[i]));
}
// ---- WV_Q6FM: fragment-ordered Q6_K ---------------------------------------------------------------------
// Unit (16-row tile, super-block) = 3360 B = 16 x (192 + 18), meta inline:
//   [0, 256)     sc: per g {row g sc[16] | row g + 8 sc[16]} (int8, natural 16-chunk order)
//   [256, 288)   d: per g {row g fp16 | row g + 8 fp16}
//   [288, 2336)  qlo: 4 groups p x 512 B, lane l = 4 g + tq: {row g | row g + 8} x {word A | word B}; A = the Q4_K-order
//                nibble bytes 32 p + 4 tq .. + 3 (chunk 0 elements 4 tq .. + 3 of sub-block 2 p in the low nibbles, of
//                2 p + 1 in the high nibbles), B = bytes 32 p + 16 + 4 tq .. + 3 (chunk 1)
//   [2336, 3360) qhi: 2 x 512 B (row g, then row g + 8), lane l: H_p, p = 0..3; byte m of H_p = the high 2 bits of element
//                4 tq + m of (sub-block, chunk) (2p, 0) at bits 4-5, (2p + 1, 0) at 6-7, (2p, 1) at 0-1, (2p + 1, 1) at 2-3.
// So thread tq supplies elements 4 tq .. + 3 of both 16-chunks of every sub-block: 9 LDG (7 x 16 B per lane, all warp
// loads contiguous) per unit instead of the TM path's 26 x 4-8 B.
#ifndef WV_Q6HP
#define WV_Q6HP 0
#endif
constexpr int Q6F_UNIT = 3360, Q6F_D = 256, Q6F_LO = 288, Q6F_HI = 2336;
struct Q6FRaw { int4 sc[2]; int dd; int2 q[2][4]; int hi[2][4]; };
__device__ __forceinline__ void q6f_load(Q6FRaw & r, const Mat & m, int row0, int sb, int g, int tq) {
    const uint8_t * u = m.w + ((size_t) (row0 >> 4) * (m.K >> 8) + sb) * Q6F_UNIT;
    const int l = 4 * g + tq;
    r.sc[0] = wld4(u + 32 * g);
    r.sc[1] = wld4(u + 32 * g + 16);
    r.dd = wld1(u + Q6F_D + 4 * g);
#pragma unroll
    for (int p = 0; p < 4; p++) {
        const int4 v = wld4(u + Q6F_LO + 512 * p + 16 * l);
        r.q[0][p] = make_int2(v.x, v.y);
        r.q[1][p] = make_int2(v.z, v.w);
    }
#if WV_Q6HP // qhi p-major: per p 256 B, lane l: {row g H_p | row g + 8 H_p} (a prefill stage reads one contiguous block)
#pragma unroll
    for (int p = 0; p < 4; p++) {
        const int2 v = wld2(u + Q6F_HI + 256 * p + 8 * l);
        r.hi[0][p] = v.x;
        r.hi[1][p] = v.y;
    }
#else
#pragma unroll
    for (int rr = 0; rr < 2; rr++) {
        const int4 v = wld4(u + Q6F_HI + 512 * rr + 16 * l);
        r.hi[rr][0] = v.x;
        r.hi[rr][1] = v.y;
        r.hi[rr][2] = v.z;
        r.hi[rr][3] = v.w;
    }
#endif
}
// the int8x4 operand (q + 32) of row rr (0: g, 1: g + 8), sub-block k, chunk c: elements 16 c + 4 tq .. + 3
__device__ __forceinline__ int q6f_frag(const Q6FRaw & r, int rr, int k, int c) {
    const int p = k >> 1, part = k & 1;
    const unsigned base = c ? (unsigned) r.q[rr][p].y : (unsigned) r.q[rr][p].x;
    const unsigned lo = (part ? base >> 4 : base) & 0x0f0f0f0fu;
    const unsigned H = (unsigned) r.hi[rr][p];
    const unsigned hp = c == 0 ? (part ? H >> 2 : H) : (part ? H << 2 : H << 4);
    return (int) (lo | (hp & 0x30303030u));
}
// d sc[cn] of row rr, 16-chunk cn = 2 k + c, as the TM path's wsc2 computes it (fmaf form: exact)
__device__ __forceinline__ float q6f_fa(const Q6FRaw & r, float d, float nd, int rr, int cn) {
    const int4 s = r.sc[rr];
    const int w = (cn >> 2) == 0 ? s.x : (cn >> 2) == 1 ? s.y : (cn >> 2) == 2 ? s.z : s.w;
    const int v = (int) (signed char) ((unsigned) w >> (8 * (cn & 3)));
    return fmaf(d, __int_as_float(MAGIC + v), nd);
}
template <int T, bool X16>
__device__ __forceinline__ void wv_tile_q6f(const Mat & m, int row0, int sb0, int nsb, const XSm & xs,
                                            float (&acc)[Cfg<T_Q6K, T, X16>::NG][4]) {
    using C = Cfg<T_Q6K, T, X16>;
    const int lane = threadIdx.x & 31, g = lane >> 2, tq = lane & 3, h = tq >> 1;
    float acc1[C::K16 ? C::NG : 1][4]; // K16: chunk 1's accumulators (acc: chunk 0's)
#pragma unroll
    for (int n = 0; n < C::NG; n++)
#pragma unroll
        for (int c = 0; c < 4; c++) acc[n][c] = 0.f;
#pragma unroll
    for (int n = 0; n < (C::K16 ? C::NG : 1); n++)
#pragma unroll
        for (int c = 0; c < 4; c++) acc1[n][c] = 0.f;
    // B: K16 token 8 n + g, elements 16 c + 4 tq; split token 4 n + (g & 3), chunk g >> 2 only (k-slots 16 (g >> 2) + ..)
    const int8_t * xrow = C::K16 ? xs.xq + g * xs.ld + 4 * tq : xs.xq + (g & 3) * xs.ld + 16 * (g >> 2) + 4 * tq;
#pragma unroll 1
    for (int sb = 0; sb < nsb; sb++) {
        Q6FRaw r;
        q6f_load(r, m, row0, sb0 + sb, g, tq);
        const float d0 = __half2float(__ushort_as_half((unsigned short) (r.dd & 0xffff)));
        const float d1 = __half2float(__ushort_as_half((unsigned short) ((unsigned) r.dd >> 16)));
        const float nd0 = -FMAGIC * d0, nd1 = -FMAGIC * d1;
#pragma unroll
        for (int k = 0; k < 8; k++) {
            const int j = 8 * sb + k;
            const int w00 = q6f_frag(r, 0, k, 0), w10 = q6f_frag(r, 1, k, 0);
            const int w01 = q6f_frag(r, 0, k, 1), w11 = q6f_frag(r, 1, k, 1);
            if constexpr (C::K16) {
                const float f00 = q6f_fa(r, d0, nd0, 0, 2 * k), f10 = q6f_fa(r, d1, nd1, 1, 2 * k);
                const float f01 = q6f_fa(r, d0, nd0, 0, 2 * k + 1), f11 = q6f_fa(r, d1, nd1, 1, 2 * k + 1);
#pragma unroll
                for (int n = 0; n < C::NG; n++) {
                    const int8_t * xb = xrow + 8 * n * xs.ld + 32 * j;
                    const int b0 = *reinterpret_cast<const int *>(xb), b1 = *reinterpret_cast<const int *>(xb + 16);
                    const float4 mx = xs.mx4[(j * C::NG + n) * 4 + tq];
                    float4 m0, m1;
                    if (X16) {
                        m0 = xs.md[((2 * j) * C::NG + n) * 4 + tq];
                        m1 = xs.md[((2 * j + 1) * C::NG + n) * 4 + tq];
                    } else {
                        m0 = m1 = xs.md[(j * C::NG + n) * 4 + tq];
                    }
                    const int ci0[4] = {__float_as_int(mx.x), __float_as_int(mx.y), __float_as_int(mx.x), __float_as_int(mx.y)};
                    const int ci1[4] = {__float_as_int(mx.z), __float_as_int(mx.w), __float_as_int(mx.z), __float_as_int(mx.w)};
                    int c0[4], c1[4];
                    mma16816(c0, w00, w10, b0, ci0);
                    mma16816(c1, w01, w11, b1, ci1);
                    acc[n][0] = fmaf(f00, fmaf(m0.x, __int_as_float(c0[0]), m0.y), acc[n][0]);
                    acc[n][1] = fmaf(f00, fmaf(m0.z, __int_as_float(c0[1]), m0.w), acc[n][1]);
                    acc[n][2] = fmaf(f10, fmaf(m0.x, __int_as_float(c0[2]), m0.y), acc[n][2]);
                    acc[n][3] = fmaf(f10, fmaf(m0.z, __int_as_float(c0[3]), m0.w), acc[n][3]);
                    acc1[n][0] = fmaf(f01, fmaf(m1.x, __int_as_float(c1[0]), m1.y), acc1[n][0]);
                    acc1[n][1] = fmaf(f01, fmaf(m1.z, __int_as_float(c1[1]), m1.w), acc1[n][1]);
                    acc1[n][2] = fmaf(f11, fmaf(m1.x, __int_as_float(c1[2]), m1.y), acc1[n][2]);
                    acc1[n][3] = fmaf(f11, fmaf(m1.z, __int_as_float(c1[3]), m1.w), acc1[n][3]);
                }
            } else {
                const float fA0 = q6f_fa(r, d0, nd0, 0, 2 * k + h), fA1 = q6f_fa(r, d1, nd1, 1, 2 * k + h);
                const int a[4] = {w00, w10, w01, w11};
#pragma unroll
                for (int n = 0; n < C::NG; n++) {
                    const int x = *reinterpret_cast<const int *>(xrow + 4 * n * xs.ld + 32 * j);
                    const int b0 = (g >> 2) ? 0 : x, b1 = (g >> 2) ? x : 0;
                    const float4 md = xs.md[(j * C::NG + n) * 4 + tq];
                    const float2 mx = xs.mx[(j * C::NG + n) * 4 + tq];
                    const int ci[4] = {__float_as_int(mx.x), __float_as_int(mx.y), __float_as_int(mx.x), __float_as_int(mx.y)};
                    int cc[4];
                    mma16832(cc, a, b0, b1, ci);
                    acc[n][0] = fmaf(fA0, fmaf(md.x, __int_as_float(cc[0]), md.y), acc[n][0]);
                    acc[n][1] = fmaf(fA0, fmaf(md.z, __int_as_float(cc[1]), md.w), acc[n][1]);
                    acc[n][2] = fmaf(fA1, fmaf(md.x, __int_as_float(cc[2]), md.y), acc[n][2]);
                    acc[n][3] = fmaf(fA1, fmaf(md.z, __int_as_float(cc[3]), md.w), acc[n][3]);
                }
            }
        }
    }
    if constexpr (C::K16) { // chunk 0 + chunk 1 (the split path's tq < 2 sum: own chunk-0 sum + the partner's chunk 1)
#pragma unroll
        for (int n = 0; n < C::NG; n++)
#pragma unroll
            for (int c = 0; c < 4; c++) acc[n][c] += acc1[n][c];
    } else {
#pragma unroll
        for (int n = 0; n < C::NG; n++)
#pragma unroll
            for (int c = 0; c < 4; c++) acc[n][c] += __shfl_xor_sync(0xffffffffu, acc[n][c], 2);
    }
}

template <int TYPE, int T, bool X16, int PF>
__device__ __forceinline__ void wv_tile(const Mat & m, int row0, int sb0, int nsb, const XSm & xs,
                                        float (&acc)[Cfg<TYPE, T, X16>::NG][4]) {
    if constexpr (TYPE == T_Q6K && WV_Q6FM) {
        wv_tile_q6f<T, X16>(m, row0, sb0, nsb, xs, acc);
        return;
    }
    using W = WV<TYPE>;
    using C = Cfg<TYPE, T, X16>;
    const int lane = threadIdx.x & 31, g = lane >> 2, tq = lane & 3, h = tq >> 1;
#pragma unroll
    for (int n = 0; n < C::NG; n++)
#pragma unroll
        for (int c = 0; c < 4; c++) acc[n][c] = 0.f;
    const int8_t * xrow = C::SPLIT ? xs.xq + (g & 3) * xs.ld + 8 * tq : xs.xq + g * xs.ld + 8 * tq;
    const bool bon = !C::SPLIT || (g >> 2) == h;
    if (PF == 2 && !X16 && !WV_MODE) { // 2-deep: sb + 2's loads right after sb's MMAs (2 units in flight)
        typename W::Raw a0, a1, b0, b1;
        if (nsb > 0) wv_load2<TYPE>(a0, a1, m, row0, sb0, g, tq);
        if (nsb > 1) wv_load2<TYPE>(b0, b1, m, row0, sb0 + 1, g, tq);
#pragma unroll 1
        for (int sb = 0; sb < nsb; sb += 2) {
            wv_sb<TYPE, T, X16>(a0, a1, 8 * sb, xs, xrow, bon, acc);
            if (sb + 2 < nsb) wv_load2<TYPE>(a0, a1, m, row0, sb0 + sb + 2, g, tq);
            if (sb + 1 < nsb) {
                wv_sb<TYPE, T, X16>(b0, b1, 8 * sb + 8, xs, xrow, bon, acc);
                if (sb + 3 < nsb) wv_load2<TYPE>(b0, b1, m, row0, sb0 + sb + 3, g, tq);
            }
        }
    } else if (!PF && WV_SBU == 2) {
        typename W::Raw a0, a1, b0, b1;
        int sb = 0;
#pragma unroll 1
        for (; sb + 1 < nsb; sb += 2) {
            wv_load2<TYPE>(a0, a1, m, row0, sb0 + sb, g, tq);
            wv_load2<TYPE>(b0, b1, m, row0, sb0 + sb + 1, g, tq);
            wv_sb<TYPE, T, X16>(a0, a1, 8 * sb, xs, xrow, bon, acc);
            wv_sb<TYPE, T, X16>(b0, b1, 8 * sb + 8, xs, xrow, bon, acc);
        }
        if (sb < nsb) {
            wv_load2<TYPE>(a0, a1, m, row0, sb0 + sb, g, tq);
            wv_sb<TYPE, T, X16>(a0, a1, 8 * sb, xs, xrow, bon, acc);
        }
    } else {
    typename W::Raw r0, r1, n0, n1;
    if (PF == 1 || WV_MODE == 2) {
        wv_load2<TYPE>(r0, r1, m, row0, sb0, g, tq);
    }
    if (WV_MODE == 4 || WV_MODE == 6) { // loads only, coalesced: lane -> 16-byte chunk c = lane + 32 i of the tile's step
        const int CH = TYPE == T_IQ4X ? 17 : m.rb / (m.K / 256) / 16, bst = 16 * CH;
        const int nst = TYPE == T_IQ4X ? (nsb + 1) / 2 : nsb, st0 = TYPE == T_IQ4X ? sb0 / 2 : sb0;
        auto ld = [&](int s, int4 (&v)[9]) {
#pragma unroll
            for (int i = 0; i < 9; i++) {
                const int c = lane + 32 * i, r = c / CH, pc = c - r * CH;
                v[i] = make_int4(0, 0, 0, 0);
                if (c < 16 * CH) v[i] = wld4(m.w + (size_t) (row0 + r) * m.rb + (size_t) (st0 + s) * bst + 16 * pc);
            }
        };
        int x = 0;
        int4 va[9], vb[9];
        if (WV_MODE == 6) ld(0, va);
#pragma unroll 1
        for (int s = 0; s < nst; s++) {
            if (WV_MODE == 4) ld(s, va);
            else if (s + 1 < nst) ld(s + 1, vb);
#pragma unroll
            for (int i = 0; i < 9; i++) x ^= va[i].x ^ va[i].y ^ va[i].z ^ va[i].w;
            if (WV_MODE == 6)
#pragma unroll
                for (int i = 0; i < 9; i++) va[i] = vb[i];
        }
        acc[0][0] += __int_as_float(x & 0x007fffff);
        return;
    }
    if (WV_MODE == 8) { // loads only, coalesced 16-B chunks: WV_SPS super-blocks x WV_RH rows per step (Q4_K / Q5_K)
        constexpr int SPS = WV_SPS, RH = WV_RH, NV = (RH * 11 * SPS + 31) / 32;
        const int bps = m.rb / (m.K / 256), CH = bps / 16 * SPS;
        int x = 0;
#pragma unroll 1
        for (int hf = 0; hf < 16 / RH; hf++)
#pragma unroll 1
            for (int st = 0; st < nsb / SPS; st++) {
                int4 v[NV];
#pragma unroll
                for (int i = 0; i < NV; i++) {
                    const int c = lane + 32 * i, r = c / CH, pc = c - r * CH;
                    v[i] = make_int4(0, 0, 0, 0);
#if WV_TILEMAJ // as if the matrix were stored tile-major: each (16-row tile, super-block) unit 16 x bps contiguous bytes
                    if (c < RH * CH) v[i] = wld4(m.w + ((size_t) (row0 / 16) * (m.K / 256) + sb0 + SPS * st) * 16 * bps + (size_t) RH * hf * CH * 16 + 16 * c);
#else
                    if (c < RH * CH) v[i] = wld4(m.w + (size_t) (row0 + RH * hf + r) * m.rb + (size_t) (sb0 + SPS * st) * bps + 16 * pc);
#endif
                }
#pragma unroll
                for (int i = 0; i < NV; i++) x ^= v[i].x ^ v[i].y ^ v[i].z ^ v[i].w;
            }
        acc[0][0] += __int_as_float(x & 0x007fffff);
        return;
    }
    if (WV_MODE == 3) { // loads only, the wv pattern, 2 units in flight
        int x = 0;
        wv_load2<TYPE>(r0, r1, m, row0, sb0, g, tq);
#pragma unroll 1
        for (int sb = 0; sb < nsb; sb++) {
            if (sb + 1 < nsb) {
                wv_load2<TYPE>(n0, n1, m, row0, sb0 + sb + 1, g, tq);
            }
            x ^= raw_xor(r0) ^ raw_xor(r1);
            r0 = n0;
            r1 = n1;
        }
        acc[0][0] += __int_as_float(x & 0x007fffff);
        return;
    }
    if (WV_MODE) {
        int x = 0;
#pragma unroll 1
        for (int sb = 0; sb < nsb; sb++) {
            if (WV_MODE == 1) {
                wv_load2<TYPE>(r0, r1, m, row0, sb0 + sb, g, tq);
                x ^= raw_xor(r0) ^ raw_xor(r1);
            } else {
                raw_opaque(r0);
                raw_opaque(r1);
                wv_sb<TYPE, T, X16>(r0, r1, 8 * sb, xs, xrow, bon, acc);
            }
        }
        acc[0][0] += __int_as_float(x & 0x007fffff);
        return;
    }
#if WV_L2PF
    // L2 prefetch of the main weight stream WV_L2PF super-blocks ahead: lane l -> row l & 15, 128-byte line l >> 4;
    // FM units (WV_TILEMAJ 2, Q4_K / Q5_K / IQ4_XS: 16 bps contiguous bytes per (tile, super-block)): lane l -> line l
    const int bps = m.rb / (m.K / 256);
#if WV_TILEMAJ == 2
    constexpr bool FMU = TYPE == T_Q4K || TYPE == T_Q5K || TYPE == T_IQ4X;
#else
    constexpr bool FMU = false;
#endif
    const uint8_t * pfrow = FMU ? m.w + (size_t) (row0 >> 4) * (m.K / 256) * 16 * bps + 128 * lane
                                : m.w + (size_t) (row0 + (lane & 15)) * m.rb + 128 * (lane >> 4);
    const int pfst = FMU ? 16 * bps : bps;
    const bool pfon = FMU ? 128 * lane < 16 * bps : 128 * (lane >> 4) < bps;
#pragma unroll
    for (int d = 0; d < WV_L2PF; d++)
        if (d < nsb && pfon) pf_l2n(pfrow + (size_t) (sb0 + d) * pfst);
#endif
#pragma unroll 1
    for (int sb = 0; sb < nsb; sb++) {
#if WV_L2PF
        if (sb + WV_L2PF < nsb && pfon) pf_l2n(pfrow + (size_t) (sb0 + sb + WV_L2PF) * pfst);
#endif
        if (PF == 1) {
            if (sb + 1 < nsb) {
                wv_load2<TYPE>(n0, n1, m, row0, sb0 + sb + 1, g, tq);
            }
        } else {
            wv_load2<TYPE>(r0, r1, m, row0, sb0 + sb, g, tq);
        }
        wv_sb<TYPE, T, X16>(r0, r1, 8 * sb, xs, xrow, bon, acc);
        if (PF == 1) {
            r0 = n0;
            r1 = n1;
        }
    }
    }
    if (C::SPLIT) { // chunk 0 (tq 0-1) + chunk 1 (tq 2-3) of the same tokens
#pragma unroll
        for (int n = 0; n < C::NG; n++)
#pragma unroll
            for (int c = 0; c < 4; c++) acc[n][c] += __shfl_xor_sync(0xffffffffu, acc[n][c], 2);
    }
}

// Fill the block's K chunk [k0, k0 + Kc) of T activation rows: values to xq, scales to md / mx.
template <int TYPE, int T, bool X16>
__device__ void wv_fill(const QB & q, int k0, int Kc, uint8_t * sm, XSm & xs) {
    using C = Cfg<TYPE, T, X16>;
    // (K16: the B loads are LDS.32 of 8 token rows x 4 tq: a row stride of 16 mod 128 B keeps them conflict-free)
    const int ld = Kc + (C::K16 ? 16 : 32), NJ = Kc / 32;
    int8_t * xq = reinterpret_cast<int8_t *>(sm);
    float4 * md = reinterpret_cast<float4 *>(sm + C::NTOKP * ld);
    float2 * mx = reinterpret_cast<float2 *>(md + NJ * C::NG * 4 * (C::K16 && X16 ? 2 : 1));
    const int u0 = k0 / 32, nu = Kc / 32;
    for (int i = threadIdx.x; i < T * nu * 2; i += NT) {
        const int t = i / (2 * nu), r = i - t * 2 * nu, u = u0 + (r >> 1), part = r & 1;
        const int b = u >> 3, p = (u >> 1) & 3, hh = u & 1;
        const int e = 256 * b + 64 * p + 32 * part + 16 * hh - k0;
        cp16(xq + t * ld + e, (part ? q.xb : q.xa) + ((size_t) t * q.U + u) * 16);
    }
    if constexpr (C::K16) { // tokens 8 n + 2 tq + {0, 1}; md per 32-block (X16: per 16-chunk h at 2 j + h), mx4 per chunk
        float4 * mx4 = reinterpret_cast<float4 *>(mx);
        for (int i = threadIdx.x; i < NJ * C::NG * 4; i += NT) {
            const int j = i / (C::NG * 4), n = (i / 4) % C::NG, tq = i & 3;
            const int jg = u0 + j, b = jg >> 3, s = jg & 7, p = s >> 1, part = s & 1;
            float dx[2][2];
            int qs[2][2];
#pragma unroll
            for (int k = 0; k < 2; k++) {
                const int t = 8 * n + 2 * tq + k;
#pragma unroll
                for (int hh = 0; hh < 2; hh++) {
                    dx[k][hh] = 0.f;
                    qs[k][hh] = 0;
                    if (t < T) {
                        const uint2 mm = q.meta[(size_t) t * q.U + 8 * b + 2 * p + hh];
                        dx[k][hh] = __half2float(reinterpret_cast<const __half *>(&mm.x)[part]);
                        qs[k][hh] = reinterpret_cast<const short *>(&mm.y)[part];
                    }
                }
            }
            if (X16) {
#pragma unroll
                for (int hh = 0; hh < 2; hh++)
                    md[((2 * j + hh) * C::NG + n) * 4 + tq] =
                        make_float4(dx[0][hh], -FMAGIC * dx[0][hh], dx[1][hh], -FMAGIC * dx[1][hh]);
            } else {
                md[i] = make_float4(dx[0][0], -FMAGIC * dx[0][0], dx[1][0], -FMAGIC * dx[1][0]);
            }
            mx4[i] = make_float4(__int_as_float(MAGIC - 32 * qs[0][0]), __int_as_float(MAGIC - 32 * qs[1][0]),
                                 __int_as_float(MAGIC - 32 * qs[0][1]), __int_as_float(MAGIC - 32 * qs[1][1]));
        }
        cp_wait();
        __syncthreads();
        xs.xq = xq;
        xs.ld = ld;
        xs.md = md;
        xs.mx = nullptr;
        xs.mx4 = mx4;
        return;
    }
    for (int i = threadIdx.x; i < NJ * C::NG * 4; i += NT) {
        const int j = i / (C::NG * 4), n = (i / 4) % C::NG, tq = i & 3;
        const int jg = u0 + j, b = jg >> 3, s = jg & 7, p = s >> 1, part = s & 1;
        float v[4];
        float2 x2;
        int qs2[2];
        for (int k = 0; k < 2; k++) {
            int t, hh = -1;
            if (C::SPLIT) {
                t = 4 * n + ((2 * tq) & 3) + k;
                hh = tq >> 1;
            } else {
                t = 8 * n + 2 * tq + k;
            }
            float dx = 0.f;
            int qs = 0;
            if (t < T) {
                const uint2 m0 = q.meta[(size_t) t * q.U + 8 * b + 2 * p + (hh < 0 ? 0 : hh)];
                const uint2 m1 = q.meta[(size_t) t * q.U + 8 * b + 2 * p + 1];
                dx = __half2float(reinterpret_cast<const __half *>(&m0.x)[part]);
                const int q0 = reinterpret_cast<const short *>(&m0.y)[part], q1 = reinterpret_cast<const short *>(&m1.y)[part];
                qs = hh < 0 ? q0 + q1 : q0;
            }
            v[2 * k] = dx;
            v[2 * k + 1] = -FMAGIC * dx;
            qs2[k] = qs;
            (k ? x2.y : x2.x) = WV<TYPE>::MIN == 2 ? __int_as_float(MAGIC - 32 * qs) : dx * (float) qs;
        }
        md[i] = make_float4(v[0], v[1], v[2], v[3]);
        mx[i] = x2;
    }
    cp_wait();
    __syncthreads();
    xs.xq = xq;
    xs.ld = ld;
    xs.md = md;
    xs.mx = mx;
}
template <int TYPE, int T, bool X16>
constexpr int wv_smem(int Kc) {
    if (Cfg<TYPE, T, X16>::K16) return Cfg<TYPE, T, X16>::NTOKP * (Kc + 16) + (Kc / 32) * Cfg<TYPE, T, X16>::NG * 4 * (X16 ? 48 : 32);
    return Cfg<TYPE, T, X16>::NTOKP * (Kc + 32) + (Kc / 32) * Cfg<TYPE, T, X16>::NG * 4 * 24;
}

struct WvArgs {
    const Mat * mats;
    int nlay;
    QB q;
    float * out;
    float * part; // [S][T][M] split-K partial sums
    unsigned * cnt; // [nlay][M / 16] arrival counters (count mod S)
    int M, K, S;
    float * scr; // v2: [gridDim][NW][2][16][16] partial sums of tiles split between warps of a block
};
// Block b < S * G (G = gridDim / S) takes K chunk b / G and the tiles [ntile * bi / G, ntile * (bi + 1) / G) of its
// group index bi; its warps take those tiles round-robin. S > 1: each chunk writes its partial sums, the last of the S
// to arrive at a tile adds them in chunk order 0 .. S - 1 (fixed, so the result does not depend on who arrives last).
template <int TYPE, int T, bool X16, int PF>
__global__ void __launch_bounds__(NT, 1) k_wv(WvArgs A) {
    extern __shared__ __align__(16) uint8_t sm[];
    using C = Cfg<TYPE, T, X16>;
    const int S = A.S, G = gridDim.x / S, b = blockIdx.x;
    const bool active = b < S * G;
    const int kc = b / G, bi = b % G, NSB = A.K / 256;
    const int sb0 = NSB * kc / S, sb1 = NSB * (kc + 1) / S;
    const int ntile = A.M / 16, t0 = ntile * bi / G, t1 = ntile * (bi + 1) / G;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, g = lane >> 2, tq = lane & 3;
    unsigned epoch = 0;
    for (int L = 0; L < A.nlay; L++) {
        XSm xs;
        if (active) wv_fill<TYPE, T, X16>(A.q, 256 * sb0, 256 * (sb1 - sb0), sm, xs);
        const Mat m = A.mats[L];
        if (active)
            for (int tile = t0 + warp; tile < t1; tile += NW) {
                float acc[C::NG][4];
                wv_tile<TYPE, T, X16, PF>(m, 16 * tile, sb0, sb1 - sb0, xs, acc);
                const bool st = !C::SPLIT || tq < 2;
                float * dst = S == 1 ? A.out : A.part + (size_t) kc * T * A.M;
#pragma unroll
                for (int n = 0; n < C::NG; n++)
#pragma unroll
                    for (int c = 0; c < 4; c++) {
                        const int t = C::SPLIT ? 4 * n + 2 * tq + (c & 1) : 8 * n + 2 * tq + (c & 1);
                        if (st && t < T) dst[(size_t) t * A.M + 16 * tile + g + 8 * (c >> 1)] = acc[n][c];
                    }
                if (S > 1) {
                    __threadfence();
                    __syncwarp();
                    unsigned old = 0;
                    if (lane == 0) old = atomicAdd(A.cnt + (size_t) L * ntile + tile, 1u);
                    old = __shfl_sync(0xffffffffu, old, 0);
                    if (old % S == S - 1) {
                        __threadfence();
                        for (int i = lane; i < 16 * T; i += 32) {
                            const int t = i / 16, r = 16 * tile + (i & 15);
                            float s = 0.f;
                            for (int k = 0; k < S; k++) s += __ldcg(A.part + ((size_t) k * T + t) * A.M + r);
                            A.out[(size_t) t * A.M + r] = s;
                        }
                    }
                }
            }
        gsync(epoch);
    }
}

// ==================================================================================================================
// wv v2 (wvgemm2): the same per-MMA arithmetic as k_wv, restructured per warp.
//  - Activations interleaved per 64-group G (32-blocks j = 2G, 2G + 1): byte 64 G + 16 tq + 8 (j & 1) + i holds element
//    32 j + 8 tq + i, so one LDS.128 gives the B fragments of the MMAs of both 32-blocks; ld = Kc + 64 (LDS.128 of two
//    tokens per 8-lane phase hit disjoint banks).
//  - Token scales: MIN 0 one float4 per (64-group, n, tq) {dx_a, dx_b of j = 2G, then of 2G + 1}; MIN 1 / 2 one float4
//    per (32-block, n, tq) {dx_a, dx_b, e_a, e_b}, e = dx * qsum (MIN 1) or the accumulator input MAGIC - 32 qsum (MIN 2).
//    -12582912 dx by FMUL (the product is exact, so fmaf(dx, C, -12582912 dx) is the same number as in k_wv).
//  - Stream-K inside the block: the block's (tile, super-block) units are split evenly over its 16 warps (k_wv gives
//    whole tiles to warps: ffn_gu 26.5 tiles per block = 1 or 2 per warp, dn_proj 12.5 = 0 or 1). A tile split between
//    warps is summed after the block's last unit in warp order: fixed by M, K and the grid, so the same for every T.
//  - Rolling register prefetch: unit u + 1's header / group words are loaded into the registers of unit u as soon as
//    those are consumed, so one unit of compute covers the load latency without a second register buffer.
// ==================================================================================================================
struct WPtr {
    const uint8_t * w; // the row's value block (or block pair) of this super-block
    const uint8_t * d; // the row's meta of this super-block (Q6_K / Q3_K pair, IQ3_S block, Q8_0 scales)
    int odd;           // super-block parity (block pairs)
};
template <int TYPE> struct WVL;
template <> struct WVL<T_Q4K> {
    using R = WV<T_Q4K>::Raw;
    static __device__ __forceinline__ WPtr ptr(const Mat & m, int row, int sb) { return {m.w + (size_t) row * m.rb + sb * 144, nullptr, 0}; }
    static __device__ __forceinline__ void hdr(R & r, const WPtr & p, int) { r.h = wld4(p.w); }
    static __device__ __forceinline__ void grp(R & r, const WPtr & p, int tq, int g) { r.q[g] = wld2(p.w + 16 + 32 * g + 8 * tq); }
    static __device__ __forceinline__ void late(R &, const WPtr &, int) {}
};
template <> struct WVL<T_Q5K> {
    using R = WV<T_Q5K>::Raw;
    static __device__ __forceinline__ WPtr ptr(const Mat & m, int row, int sb) { return {m.w + (size_t) row * m.rb + sb * 176, nullptr, 0}; }
    static __device__ __forceinline__ void hdr(R & r, const WPtr & p, int) { r.h = wld4(p.w); }
    static __device__ __forceinline__ void grp(R & r, const WPtr & p, int tq, int g) { r.q[g] = wld2(p.w + 48 + 32 * g + 8 * tq); }
    static __device__ __forceinline__ void late(R & r, const WPtr & p, int tq) { r.qh = wld2(p.w + 16 + 8 * tq); }
};
template <> struct WVL<T_IQ4X> {
    using R = WV<T_IQ4X>::Raw;
    static __device__ __forceinline__ WPtr ptr(const Mat & m, int row, int sb) { return {m.w + (size_t) row * m.rb + (sb >> 1) * 272, nullptr, sb & 1}; }
    static __device__ __forceinline__ void hdr(R & r, const WPtr & p, int) { r.h = wld2(p.w + 8 * p.odd); }
    static __device__ __forceinline__ void grp(R & r, const WPtr & p, int tq, int g) { r.q[g] = wld2(p.w + 16 + 128 * p.odd + 32 * g + 8 * tq); }
    static __device__ __forceinline__ void late(R &, const WPtr &, int) {}
};
template <> struct WVL<T_IQ4NL> {
    using R = WV<T_IQ4NL>::Raw;
    static __device__ __forceinline__ WPtr ptr(const Mat & m, int row, int sb) { return {m.w + (size_t) row * m.rb + sb * 144, nullptr, 0}; }
    static __device__ __forceinline__ void hdr(R & r, const WPtr & p, int) { r.h = wld4(p.w); }
    static __device__ __forceinline__ void grp(R & r, const WPtr & p, int tq, int g) { r.q[g] = wld2(p.w + 16 + 32 * g + 8 * tq); }
    static __device__ __forceinline__ void late(R &, const WPtr &, int) {}
};
template <> struct WVL<T_Q8_0> {
    using R = WV<T_Q8_0>::Raw;
    static __device__ __forceinline__ WPtr ptr(const Mat & m, int row, int sb) {
        return {m.w + (size_t) row * m.rb + sb * 256, reinterpret_cast<const uint8_t *>(m.d + (size_t) row * m.nd + 8 * sb), 0};
    }
    static __device__ __forceinline__ void hdr(R & r, const WPtr & p, int) { r.d = wld4(p.d); }
    static __device__ __forceinline__ void grp(R & r, const WPtr & p, int tq, int g) {
#pragma unroll
        for (int k = 2 * g; k < 2 * g + 2; k++) r.q[k] = wld2(p.w + 32 * (2 * (k >> 1) + (tq >> 1)) + 16 * (k & 1) + 8 * (tq & 1));
    }
    static __device__ __forceinline__ void late(R &, const WPtr &, int) {}
};
template <> struct WVL<T_Q6K> {
    using R = WV<T_Q6K>::Raw;
    static __device__ __forceinline__ WPtr ptr(const Mat & m, int row, int sb) {
        return {m.w + (size_t) row * m.rb + sb * Q6_BLK,
                reinterpret_cast<const uint8_t *>(m.d + (size_t) row * m.nd) + (sb >> 1) * (2 * Q6_META), sb & 1};
    }
    static __device__ __forceinline__ void hdr(R & r, const WPtr & p, int) {
#pragma unroll
        for (int k = 0; k < 4; k++) r.sc[k] = wld1(p.d + 16 * p.odd + 4 * k);
        r.dd = (int) ((unsigned) wld1(p.d + 32) >> (16 * p.odd));
    }
    static __device__ __forceinline__ void grp(R & r, const WPtr & p, int tq, int g) {
        r.q[g] = wld2(p.w + 32 * g + 8 * tq);
        r.hb[g] = wld2(p.w + 128 + 8 * (2 * g + (tq >> 1)));
    }
    static __device__ __forceinline__ void late(R &, const WPtr &, int) {}
};
template <> struct WVL<T_Q3K> {
    using R = WV<T_Q3K>::Raw;
    static __device__ __forceinline__ WPtr ptr(const Mat & m, int row, int sb) {
        return {m.w + (size_t) row * m.rb + sb * Q3_BLK,
                reinterpret_cast<const uint8_t *>(m.d + (size_t) row * m.nd) + (sb >> 1) * (2 * Q6_META), sb & 1};
    }
    static __device__ __forceinline__ void hdr(R & r, const WPtr & p, int) {
#pragma unroll
        for (int k = 0; k < 4; k++) r.sc[k] = wld1(p.d + 16 * p.odd + 4 * k);
        r.dd = (int) ((unsigned) wld1(p.d + 32) >> (16 * p.odd));
    }
    static __device__ __forceinline__ void grp(R & r, const WPtr & p, int tq, int g) {
        const int u = 2 * g + (tq >> 1);
        r.lo[g] = wld2(p.w + 8 * u);
        r.hi[g] = wld1(p.w + 64 + 4 * u);
    }
    static __device__ __forceinline__ void late(R &, const WPtr &, int) {}
};
template <> struct WVL<T_IQ3S> {
    using R = WV<T_IQ3S>::Raw;
    static __device__ __forceinline__ WPtr ptr(const Mat & m, int row, int sb) {
        return {m.w + (size_t) row * m.rb + sb * Q3_BLK, reinterpret_cast<const uint8_t *>(m.d + (size_t) row * m.nd) + sb * IQ3S_META, 0};
    }
    static __device__ __forceinline__ void hdr(R & r, const WPtr & p, int) { r.sd = wld2(p.d + 8); }
    static __device__ __forceinline__ void grp(R & r, const WPtr & p, int tq, int g) {
        const int u = 2 * g + (tq >> 1);
        r.ix[g] = wld2(p.w + 8 * u);
        r.sg[g] = wld1(p.w + 64 + 4 * u);
    }
    static __device__ __forceinline__ void late(R & r, const WPtr & p, int) { r.qh = wld2(p.d); }
};
template <int TYPE> __device__ __forceinline__ void wvl_load(typename WV<TYPE>::Raw & r, const WPtr & p, int tq) {
    WVL<TYPE>::hdr(r, p, tq);
#pragma unroll
    for (int g = 0; g < 4; g++) WVL<TYPE>::grp(r, p, tq, g);
    WVL<TYPE>::late(r, p, tq);
}

#ifndef WV2_ROLL
#define WV2_ROLL 1 // 1: rolling prefetch (reload words as consumed), 0: whole next unit loaded at the top
#endif
struct XS2 {
    const int8_t * xq; // [NTOKP][ld], interleaved per 64-group
    int ld;
    const float4 * sc; // token scales (see above)
};

// one super-block of the tile from the raw rows r0 (row g) / r1 (row g + 8); G0 = its first 64-group within the
// shared-memory chunk. more: reload r0 / r1 from n0 / n1 (the warp's next unit) as their words are consumed.
template <int TYPE, int T, bool X16>
__device__ __forceinline__ void wv2_unit(typename WV<TYPE>::Raw & r0, typename WV<TYPE>::Raw & r1, const WPtr & n0, const WPtr & n1,
                                         const Mat & m, int nrow0, int nsba, bool more, int G0, const XS2 & xs, const int8_t * xrow, bool bon,
                                         float (&acc)[Cfg<TYPE, T, X16>::NG][4]) {
    using W = WV<TYPE>;
    using L = WVL<TYPE>;
    using C = Cfg<TYPE, T, X16>;
    constexpr bool PAIR = W::MIN == 0;
    const int tq = threadIdx.x & 3, h = tq >> 1;
    typename W::Raw x0, x1;
    if (!WV2_ROLL && more) wv_load2<TYPE>(x0, x1, m, nrow0, nsba, (threadIdx.x & 31) >> 2, tq);
    typename W::Sc s0, s1;
    W::scales(r0, s0);
    W::scales(r1, s1);
    if (WV2_ROLL && more) {
        L::hdr(r0, n0, tq);
        L::hdr(r1, n1, tq);
    }
#pragma unroll
    for (int p = 0; p < 4; p++) {
        int4 bq[C::NG];
        float4 sp[C::NG];
#pragma unroll
        for (int part = 0; part < 2; part++) {
            const int k = 2 * p + part;
            int a[4];
            W::frag(r0, k, tq, a[0], a[2]);
            W::frag(r1, k, tq, a[1], a[3]);
            float fA0, gA0, fA1, gA1;
            W::wsc(s0, k, h, fA0, gA0);
            W::wsc(s1, k, h, fA1, gA1);
#pragma unroll
            for (int n = 0; n < C::NG; n++) {
                if (part == 0) {
                    bq[n] = make_int4(0, 0, 0, 0);
                    if (bon) bq[n] = *reinterpret_cast<const int4 *>(xrow + C::TPG * n * xs.ld + 64 * (G0 + p));
                    if (PAIR) sp[n] = xs.sc[((G0 + p) * C::NG + n) * 4 + tq];
                }
                const float4 e4 = PAIR ? sp[n] : xs.sc[((2 * (G0 + p) + part) * C::NG + n) * 4 + tq];
                const float dxa = PAIR ? (part ? e4.z : e4.x) : e4.x;
                const float dxb = PAIR ? (part ? e4.w : e4.y) : e4.y;
                int ci[4] = {MAGIC, MAGIC, MAGIC, MAGIC};
                if (W::MIN == 2) {
                    ci[0] = ci[2] = __float_as_int(e4.z);
                    ci[1] = ci[3] = __float_as_int(e4.w);
                }
                int cc[4];
                mma16832(cc, a, part ? bq[n].z : bq[n].x, part ? bq[n].w : bq[n].y, ci);
                const float nma = -FMAGIC * dxa, nmb = -FMAGIC * dxb;
                acc[n][0] = fmaf(fA0, fmaf(dxa, __int_as_float(cc[0]), nma), acc[n][0]);
                acc[n][1] = fmaf(fA0, fmaf(dxb, __int_as_float(cc[1]), nmb), acc[n][1]);
                acc[n][2] = fmaf(fA1, fmaf(dxa, __int_as_float(cc[2]), nma), acc[n][2]);
                acc[n][3] = fmaf(fA1, fmaf(dxb, __int_as_float(cc[3]), nmb), acc[n][3]);
                if (W::MIN == 1) {
                    acc[n][0] = fmaf(-gA0, e4.z, acc[n][0]);
                    acc[n][1] = fmaf(-gA0, e4.w, acc[n][1]);
                    acc[n][2] = fmaf(-gA1, e4.z, acc[n][2]);
                    acc[n][3] = fmaf(-gA1, e4.w, acc[n][3]);
                }
            }
        }
        if (WV2_ROLL && more) {
            L::grp(r0, n0, tq, p);
            L::grp(r1, n1, tq, p);
        }
    }
    if (WV2_ROLL && more) {
        L::late(r0, n0, tq);
        L::late(r1, n1, tq);
    }
    if (!WV2_ROLL && more) {
        r0 = x0;
        r1 = x1;
    }
}

// Fill the block's K chunk [k0, k0 + Kc) of T activation rows: values interleaved to xq, scales to sc.
template <int TYPE, int T, bool X16>
__device__ void wv2_fill(const QB & q, int k0, int Kc, uint8_t * sm, XS2 & xs) {
    using C = Cfg<TYPE, T, X16>;
    constexpr bool PAIR = WV<TYPE>::MIN == 0;
    const int ld = Kc + 64, NJ = Kc / 32, G00 = k0 / 64;
    int8_t * xq = reinterpret_cast<int8_t *>(sm);
    float4 * sc = reinterpret_cast<float4 *>(sm + C::NTOKP * ld);
    const int u0 = k0 / 32, nu = Kc / 32;
    // unit u = (b, p, hh): A = elements 16 hh + [0, 16) of 32-block 8 b + 2 p, B the same of the next 32-block; its lane
    // pieces tq = 2 hh, 2 hh + 1 of 64-group 4 b + p are [A 8 B | B 8 B] at 64 G + 16 tq
    for (int i = threadIdx.x; i < T * nu; i += NT) {
        const int t = i / nu, u = u0 + (i - t * nu);
        const int b = u >> 3, p = (u >> 1) & 3, hh = u & 1;
        const int4 va = *reinterpret_cast<const int4 *>(q.xa + ((size_t) t * q.U + u) * 16);
        const int4 vb = *reinterpret_cast<const int4 *>(q.xb + ((size_t) t * q.U + u) * 16);
        int8_t * d = xq + t * ld + 64 * (4 * b + p - G00) + 32 * hh;
        *reinterpret_cast<int4 *>(d) = make_int4(va.x, va.y, vb.x, vb.y);
        *reinterpret_cast<int4 *>(d + 16) = make_int4(va.z, va.w, vb.z, vb.w);
    }
    const int NE = PAIR ? NJ / 2 : NJ;
    for (int i = threadIdx.x; i < NE * C::NG * 4; i += NT) {
        const int e = i / (C::NG * 4), n = (i / 4) % C::NG, tq = i & 3;
        float v[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
        for (int k = 0; k < 2; k++) {
            int t, hh = -1;
            if (C::SPLIT) {
                t = 4 * n + ((2 * tq) & 3) + k;
                hh = tq >> 1;
            } else {
                t = 8 * n + 2 * tq + k;
            }
#pragma unroll
            for (int jj = 0; jj < (PAIR ? 2 : 1); jj++) {
                const int jg = u0 + (PAIR ? 2 * e + jj : e), b = jg >> 3, p = (jg >> 1) & 3, part = jg & 1;
                float dx = 0.f;
                int qs = 0;
                if (t < T) {
                    const uint2 m0 = q.meta[(size_t) t * q.U + 8 * b + 2 * p + (hh < 0 ? 0 : hh)];
                    const uint2 m1 = q.meta[(size_t) t * q.U + 8 * b + 2 * p + 1];
                    dx = __half2float(reinterpret_cast<const __half *>(&m0.x)[part]);
                    const int q0 = reinterpret_cast<const short *>(&m0.y)[part], q1 = reinterpret_cast<const short *>(&m1.y)[part];
                    qs = hh < 0 ? q0 + q1 : q0;
                }
                if (PAIR) {
                    v[2 * jj + k] = dx;
                } else {
                    v[k] = dx;
                    v[2 + k] = WV<TYPE>::MIN == 2 ? __int_as_float(MAGIC - 32 * qs) : dx * (float) qs;
                }
            }
        }
        sc[i] = make_float4(v[0], v[1], v[2], v[3]);
    }
    __syncthreads();
    xs.xq = xq;
    xs.ld = ld;
    xs.sc = sc;
}
template <int TYPE, int T, bool X16>
constexpr int wv2_smem(int Kc) {
    return Cfg<TYPE, T, X16>::NTOKP * (Kc + 64) + (WV<TYPE>::MIN == 0 ? Kc / 64 : Kc / 32) * Cfg<TYPE, T, X16>::NG * 4 * 16;
}

// S > 1: this warp has written chunk kc's sums of `tile` to A.part; the last of the S chunks to arrive adds them in
// chunk order 0 .. S - 1
template <int T>
__device__ __forceinline__ void wv2_merge(const WvArgs & A, int L, int tile) {
    const int lane = threadIdx.x & 31, S = A.S;
    __threadfence();
    __syncwarp();
    unsigned old = 0;
    if (lane == 0) old = atomicAdd(A.cnt + (size_t) L * (A.M / 16) + tile, 1u);
    old = __shfl_sync(0xffffffffu, old, 0);
    if (old % S == S - 1) {
        __threadfence();
        for (int i = lane; i < 16 * T; i += 32) {
            const int t = i / 16, r = 16 * tile + (i & 15);
            float s = 0.f;
            for (int k = 0; k < S; k++) s += __ldcg(A.part + ((size_t) k * T + t) * A.M + r);
            A.out[(size_t) t * A.M + r] = s;
        }
    }
}

template <int TYPE, int T, bool X16>
__global__ void __launch_bounds__(NT, 1) k_wv2(WvArgs A) {
    extern __shared__ __align__(16) uint8_t sm[];
    using W = WV<TYPE>;
    using L = WVL<TYPE>;
    using C = Cfg<TYPE, T, X16>;
    const int S = A.S, G = gridDim.x / S, b = blockIdx.x;
    const bool active = b < S * G;
    const int kc = b / G, bi = b % G, NSB = A.K / 256;
    const int sb0 = NSB * kc / S, nsb = NSB * (kc + 1) / S - sb0;
    const int ntile = A.M / 16, t0 = ntile * bi / G, t1 = ntile * (bi + 1) / G;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, g = lane >> 2, tq = lane & 3, h = tq >> 1;
    const int nu = (t1 - t0) * nsb, ub = nu * warp / NW, ue = nu * (warp + 1) / NW;
    float * scr = A.scr + (size_t) b * NW * 2 * 16 * T; // [NW][2][16][T]: slot 0 = a warp's first segment, 1 = its last
    float * dst = S == 1 ? A.out : A.part + (size_t) kc * T * A.M;
    unsigned epoch = 0;
    for (int Ly = 0; Ly < A.nlay; Ly++) {
        XS2 xs;
        if (active) wv2_fill<TYPE, T, X16>(A.q, 256 * sb0, 256 * nsb, sm, xs);
        const Mat m = A.mats[Ly];
        if (active && ub < ue) {
            const int8_t * xrow = xs.xq + (C::SPLIT ? (g & 3) : g) * xs.ld + 16 * tq;
            const bool bon = !C::SPLIT || (g >> 2) == h;
            int tile = ub / nsb, sb = ub - tile * nsb, seg0 = sb;
            bool first = true;
            typename W::Raw r0, r1;
            if (WV2_ROLL) {
                wvl_load<TYPE>(r0, L::ptr(m, 16 * (t0 + tile) + g, sb0 + sb), tq);
                wvl_load<TYPE>(r1, L::ptr(m, 16 * (t0 + tile) + g + 8, sb0 + sb), tq);
            } else {
                wv_load2<TYPE>(r0, r1, m, 16 * (t0 + tile), sb0 + sb, g, tq);
            }
            float acc[C::NG][4];
#pragma unroll
            for (int n = 0; n < C::NG; n++)
#pragma unroll
                for (int c = 0; c < 4; c++) acc[n][c] = 0.f;
#pragma unroll 1
            for (int u = ub; u < ue; u++) {
                int ntl = tile, nsbi = sb + 1;
                if (nsbi == nsb) {
                    nsbi = 0;
                    ntl++;
                }
                const bool more = u + 1 < ue;
                const WPtr n0 = L::ptr(m, 16 * (t0 + ntl) + g, sb0 + nsbi), n1 = L::ptr(m, 16 * (t0 + ntl) + g + 8, sb0 + nsbi);
                wv2_unit<TYPE, T, X16>(r0, r1, n0, n1, m, 16 * (t0 + ntl), sb0 + nsbi, more, 4 * sb, xs, xrow, bon, acc);
                if (nsbi == 0 || !more) { // the warp's segment [seg0, sb] of `tile` ends here
                    if (C::SPLIT) {
#pragma unroll
                        for (int n = 0; n < C::NG; n++)
#pragma unroll
                            for (int c = 0; c < 4; c++) acc[n][c] += __shfl_xor_sync(0xffffffffu, acc[n][c], 2);
                    }
                    const bool full = seg0 == 0 && sb == nsb - 1;
                    float * sp = scr + (size_t) (2 * warp + (first ? 0 : 1)) * 16 * T;
                    if (!C::SPLIT || tq < 2) {
#pragma unroll
                        for (int n = 0; n < C::NG; n++)
#pragma unroll
                            for (int c = 0; c < 4; c++) {
                                const int t = C::SPLIT ? 4 * n + 2 * tq + (c & 1) : 8 * n + 2 * tq + (c & 1);
                                const int r = g + 8 * (c >> 1);
                                if (t < T) {
                                    if (full) dst[(size_t) t * A.M + 16 * (t0 + tile) + r] = acc[n][c];
                                    else sp[r * T + t] = acc[n][c];
                                }
                            }
                    }
                    if (full && S > 1) wv2_merge<T>(A, Ly, t0 + tile);
#pragma unroll
                    for (int n = 0; n < C::NG; n++)
#pragma unroll
                        for (int c = 0; c < 4; c++) acc[n][c] = 0.f;
                    seg0 = 0;
                    first = false;
                }
                tile = ntl;
                sb = nsbi;
            }
        }
        if (active) {
            __syncthreads();
            // a warp whose first unit is inside a tile that started in an earlier warp, and which ends that tile, adds the
            // tile's pieces in warp order
            const int tr = ub / nsb;
            if (ub < ue && ub != tr * nsb && (tr + 1) * nsb <= ue) {
                const int fu = tr * nsb;
                int wf = warp - 1;
                while (wf > 0 && nu * wf / NW > fu) wf--;
                for (int i = lane; i < 16 * T; i += 32) {
                    const int r = i / T, t = i - r * T;
                    float s = 0.f;
                    for (int w = wf; w <= warp; w++) {
                        const int wb = nu * w / NW, we = nu * (w + 1) / NW;
                        if (wb == we) continue;
                        const int slot = (w == wf && wb != fu) ? 1 : 0;
                        s += scr[((size_t) (2 * w + slot) * 16 + r) * T + t];
                    }
                    dst[(size_t) t * A.M + 16 * (t0 + tr) + r] = s;
                }
                if (S > 1) wv2_merge<T>(A, Ly, t0 + tr);
            }
        }
        gsync(epoch);
    }
}

// ==================================================================================================================
// scalar reference decode of one weight in the engine layout: w = fA * v - gA, v = the integer operand both kernels feed
// ==================================================================================================================
__constant__ signed char c_kv[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};
__device__ float hf(const uint8_t * p) { return __half2float(*reinterpret_cast<const __half *>(p)); }
__device__ void ref_w(const Mat & m, int row, int e, int & v, double & fA, double & gA) {
    const int b = e >> 8, w = e & 255, p = w >> 6, part = (w >> 5) & 1, l = w & 31, hh = l >> 4, i = l & 15;
    const int sbk = 2 * p + part, c = 4 * p + 2 * part + hh; // sub-block (32), chunk (16)
    const uint8_t * r = m.w + (size_t) row * m.rb;
    if (m.type == T_Q4K || m.type == T_Q5K) {
        const uint8_t * blk = r + b * (m.type == T_Q4K ? 144 : 176);
        const uint8_t * sc = blk + 4;
        int s, mm;
        if (sbk < 4) { s = sc[sbk] & 63; mm = sc[sbk + 4] & 63; }
        else { s = (sc[sbk + 4] & 0xF) | ((sc[sbk - 4] >> 6) << 4); mm = (sc[sbk + 4] >> 4) | ((sc[sbk] >> 6) << 4); }
        const uint8_t qb = blk[(m.type == T_Q4K ? 16 : 48) + 32 * p + l];
        v = part ? qb >> 4 : qb & 15;
        if (m.type == T_Q5K) v |= ((blk[16 + l] >> sbk) & 1) << 4;
        fA = (double) hf(blk) * s;
        gA = (double) hf(blk + 2) * mm;
    } else if (m.type == T_IQ4X) {
        const uint8_t * pr = r + (b >> 1) * 272, * hd = pr + 8 * (b & 1);
        const int sh = hd[2] | (hd[3] << 8), sl = hd[4 + p];
        const int ls = ((sl >> (4 * part)) & 15) | (((sh >> (2 * sbk)) & 3) << 4);
        const uint8_t qb = pr[16 + 128 * (b & 1) + 32 * p + l];
        v = c_kv[part ? qb >> 4 : qb & 15];
        fA = (double) hf(hd) * (ls - 32);
        gA = 0;
    } else if (m.type == T_IQ4NL) {
        const uint8_t * blk = r + b * 144;
        const uint8_t qb = blk[16 + 32 * p + l];
        v = c_kv[part ? qb >> 4 : qb & 15];
        fA = hf(blk + 2 * sbk);
        gA = 0;
    } else if (m.type == T_Q8_0) {
        const int u = 8 * b + 2 * p + hh;
        v = (int8_t) r[32 * u + 16 * part + i];
        fA = __half2float(m.d[(size_t) row * m.nd + (e >> 5)]);
        gA = 0;
    } else if (m.type == T_Q6K || m.type == T_Q3K) {
        const uint8_t * mp = reinterpret_cast<const uint8_t *>(m.d + (size_t) row * m.nd) + (b >> 1) * 36;
        const int sc = (int8_t) mp[16 * (b & 1) + c];
        const double d = hf(mp + 32 + 2 * (b & 1));
        const int u = 2 * p + hh, jj = i >> 2, mb = i & 3;
        if (m.type == T_Q6K) {
            const uint8_t * blk = r + b * 192;
            const uint8_t qb = blk[32 * p + 16 * hh + i];
            const int lo = part ? qb >> 4 : qb & 15, hi = (blk[128 + 8 * u + 4 * part + mb] >> (2 * jj)) & 3;
            v = lo | (hi << 4);
        } else {
            const uint8_t * blk = r + b * 96;
            const int lo = (blk[8 * u + 4 * part + mb] >> (2 * jj)) & 3, hi = (blk[64 + 4 * u + mb] >> (jj + 4 * part)) & 1;
            v = (lo | (hi << 2)) + 28;
        }
        fA = d * sc;
        gA = 32 * fA;
    } else { // IQ3_S
        const uint8_t * blk = r + b * 96;
        const uint8_t * mp = reinterpret_cast<const uint8_t *>(m.d + (size_t) row * m.nd) + b * 16;
        const int u = 2 * p + hh, cc = i >> 2, mb = i & 3;
        const int idx = blk[8 * u + 4 * part + cc] | (((mp[u] >> (cc + 4 * part)) & 1) << 8);
        const int mag = (g_iq3s_grid[idx] >> (8 * mb)) & 0xff;
        const int sgn = (blk[64 + 4 * u + mb] >> (cc + 4 * part)) & 1;
        v = 32 + (sgn ? -mag : mag);
        const int s4 = (mp[8 + p] >> (4 * part)) & 15;
        fA = (double) hf(mp + 12) * (1 + 2 * s4);
        gA = 32 * fA;
    }
}
// activation element e of token t: integer value, its scale, and the scale-group index (per 16 or per 32)
__device__ void ref_x(const QB & q, int t, int e, int & x, double & dx) {
    const int b = e >> 8, w = e & 255, p = w >> 6, part = (w >> 5) & 1, hh = (w >> 4) & 1, i = w & 15;
    const int u = 8 * b + 2 * p + hh;
    x = (part ? q.xb : q.xa)[((size_t) t * q.U + u) * 16 + i];
    dx = __half2float(reinterpret_cast<const __half *>(&q.meta[(size_t) t * q.U + u].x)[part]);
}
// fp64 reference: out[t][row] = sum_e w_e x_e dx_e; also the integer dot per (row, t) of the operands (sum v x) and
// of the weights' offsets removed (for offset types sum (v - 32) x)
__global__ void k_ref(Mat m, QB q, int T, int K, double * out, long long * idot) {
    const int row = blockIdx.x, t = threadIdx.x / 32, lane = threadIdx.x & 31;
    if (t >= T) return;
    double s = 0;
    long long id = 0;
    for (int e = lane; e < K; e += 32) {
        int v, x;
        double fA, gA, dx;
        ref_w(m, row, e, v, fA, gA);
        ref_x(q, t, e, x, dx);
        s += (fA * v - gA) * (dx * x);
        id += (long long) v * x;
    }
    for (int o = 16; o > 0; o >>= 1) {
        s += __shfl_xor_sync(0xffffffffu, s, o);
        id += __shfl_xor_sync(0xffffffffu, id, o);
    }
    if (lane == 0) {
        out[(size_t) t * m.N + row] = s;
        idot[(size_t) t * m.N + row] = id;
    }
}

// ==================================================================================================================
// host
// ==================================================================================================================
static const char * tname(int ty) {
    switch (ty) {
        case T_Q4K: return "Q4_K";
        case T_Q5K: return "Q5_K";
        case T_Q6K: return "Q6_K";
        case T_Q8_0: return "Q8_0";
        case T_IQ4X: return "IQ4_XS";
        case T_IQ4NL: return "IQ4_NL";
        case T_Q3K: return "Q3_K";
        case T_IQ3S: return "IQ3_S";
    }
    return "?";
}
static int row_bytes(int ty, int K) {
    const int nb = K / 256;
    switch (ty) {
        case T_Q4K: return nb * 144;
        case T_Q5K: return nb * 176;
        case T_Q6K: return nb * Q6_BLK;
        case T_Q8_0: return K;
        case T_IQ4X: return nb * 136;
        case T_IQ4NL: return nb * 144;
        default: return nb * Q3_BLK; // Q3_K, IQ3_S
    }
}
static int row_meta_halves(int ty, int K) {
    const int nb = K / 256;
    switch (ty) {
        case T_Q6K: case T_Q3K: return nb * (Q6_META / 2);
        case T_IQ3S: return nb * (IQ3S_META / 2);
        case T_Q8_0: return K / 32;
        default: return 0;
    }
}
static uint16_t h16(float f) {
    __half h = __float2half(f);
    uint16_t u;
    memcpy(&u, &h, 2);
    return u;
}
// random weights in the engine layout; fp16 scale fields set to finite positive values
static void make_weights(int ty, int M, int K, std::vector<uint8_t> & w, std::vector<uint16_t> & d, std::mt19937 & rng) {
    const int rb = row_bytes(ty, K), nd = row_meta_halves(ty, K), nb = K / 256;
    w.resize((size_t) M * rb);
    d.assign((size_t) M * nd, 0);
    for (auto & x : w) x = (uint8_t) rng();
    std::uniform_real_distribution<float> U(0.5f, 1.0f);
    auto dv = [&]() { return h16(U(rng) * (1.0f / 1024)); };
    for (int r = 0; r < M; r++) {
        uint8_t * row = w.data() + (size_t) r * rb;
        uint16_t * dr = d.data() + (size_t) r * nd;
        for (int b = 0; b < nb; b++) {
            if (ty == T_Q4K || ty == T_Q5K) {
                uint16_t a = dv(), c = dv();
                memcpy(row + b * (ty == T_Q4K ? 144 : 176), &a, 2);
                memcpy(row + b * (ty == T_Q4K ? 144 : 176) + 2, &c, 2);
            } else if (ty == T_IQ4X) {
                uint16_t a = dv();
                memcpy(row + (b >> 1) * 272 + 8 * (b & 1), &a, 2);
            } else if (ty == T_IQ4NL) {
                for (int k = 0; k < 8; k++) { uint16_t a = dv(); memcpy(row + b * 144 + 2 * k, &a, 2); }
            } else if (ty == T_Q6K || ty == T_Q3K) {
                uint8_t * mp = reinterpret_cast<uint8_t *>(dr) + (b >> 1) * 36;
                for (int k = 0; k < 16; k++) mp[16 * (b & 1) + k] = (uint8_t) rng();
                uint16_t a = dv();
                memcpy(mp + 32 + 2 * (b & 1), &a, 2);
            } else if (ty == T_IQ3S) {
                uint8_t * mp = reinterpret_cast<uint8_t *>(dr) + b * 16;
                for (int k = 0; k < 12; k++) mp[k] = (uint8_t) rng();
                uint16_t a = dv();
                memcpy(mp + 12, &a, 2);
                mp[14] = mp[15] = 0;
            }
        }
        if (ty == T_Q8_0)
            for (int k = 0; k < nd; k++) dr[k] = dv();
    }
}
// random q8 activations (NTOK rows) in the unit layout; per-32 (q32) or per-16 (q16) scales
static void make_acts(int K, int ntok, bool q16, std::vector<int8_t> & xa, std::vector<int8_t> & xb,
                      std::vector<uint2> & meta, std::mt19937 & rng) {
    const int U = K / 32;
    xa.resize((size_t) ntok * U * 16);
    xb.resize(xa.size());
    meta.resize((size_t) ntok * U);
    std::uniform_int_distribution<int> Q(-127, 127);
    std::uniform_real_distribution<float> S(0.5f, 1.0f);
    for (int t = 0; t < ntok; t++)
        for (int j = 0; j < K / 32; j++) { // natural block j: units (b, p, 0 / 1), part j & 1
            const int b = j >> 3, p = (j >> 1) & 3, part = j & 1;
            const float d32 = S(rng) / 64;
            for (int hh = 0; hh < 2; hh++) {
                const int u = 8 * b + 2 * p + hh;
                int8_t * dst = (part ? xb.data() : xa.data()) + ((size_t) t * U + u) * 16;
                int s = 0;
                for (int i = 0; i < 16; i++) { dst[i] = (int8_t) Q(rng); s += dst[i]; }
                uint2 & m = meta[(size_t) t * U + u];
                const uint16_t dh = h16(q16 ? S(rng) / 64 : d32);
                reinterpret_cast<uint16_t *>(&m.x)[part] = dh;
                reinterpret_cast<short *>(&m.y)[part] = (short) s;
            }
        }
}

template <class F> static float timeit(F f, int reps) {
    cudaEvent_t a, b;
    cudaEventCreate(&a);
    cudaEventCreate(&b);
    f();
    CK(cudaDeviceSynchronize());
    float best = 1e30f;
    for (int i = 0; i < reps; i++) {
        cudaEventRecord(a);
        f();
        cudaEventRecord(b);
        CK(cudaEventSynchronize(b));
        float ms;
        cudaEventElapsedTime(&ms, a, b);
        best = ms < best ? ms : best;
    }
    return best;
}

static int g_nsm = 82;
static void * g_bar_ptr = nullptr;
static float * g_park = nullptr; // k_dp4a_k2 [5120 / 2][2][8][32]
static float * g_scr = nullptr; // k_wv2 split-tile partials [nsm][NW][2][16][16]
template <class K> static void prep(K k) { CK(cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, 99 * 1024)); }
template <class K, class Args> static void coop(K k, int smem, Args & a) {
    CK(cudaMemsetAsync(g_bar_ptr, 0, 4));
    void * args[] = {&a};
    CK(cudaLaunchCooperativeKernel((void *) k, dim3(g_nsm), dim3(NT), args, smem, 0));
}

// roles: rows M, K, dp4a IT (K / 1024), q16 activations, split-K S (fixed per role, for every T)
struct Role {
    const char * name;
    int M, K, IT;
    bool q16;
    int S;
};
static Role ROLES[] = {
    {"ffn_gu", 2 * 17408, 5120, 5, false, 1},   // gate + up rows
    {"dn_proj", 10240 + 6144, 5120, 5, false, 1}, // qkv + z
    {"attn_proj", 12288 + 2048, 5120, 5, false, 1}, // q (+gate) + k + v
    {"out", 5120, 6144, 6, false, 4},            // attn_output / ssm_out
    {"ffn_down", 5120, 17408, 17, true, 4},
    {"gu1", 17408, 5120, 5, false, 1},  // one ffn_gate / ffn_up matrix (UD-Q4_K_XL: 4 + 2 of them are Q6_K)
    {"qkv", 10240, 5120, 5, false, 1},  // attn_qkv (1 Q6_K)
    {"kv", 1024, 5120, 5, false, 1},    // attn_k / attn_v (12 + 3 Q6_K)
};

// ---- dispatch tables ------------------------------------------------------------------------------------------------
template <int T, int TYPE>
static float run_dp4a(int IT, DpArgs & a, int reps) {
    auto go = [&](auto kfun, int U) {
        const int smem = T * U * 40;
        if (smem > 99 * 1024) return -1.f;
        prep(kfun);
        return timeit([&] { coop(kfun, smem, a); }, reps);
    };
    if (IT == 5) return go(k_dp4a<T, 5, TYPE>, 160);
    if (IT == 6) return go(k_dp4a<T, 6, TYPE>, 192);
    if constexpr (T * 544 * 40 > 99 * 1024) return go(k_dp4a_k2<T, 17, TYPE>, 32 * 9); // two K slices (9 + 8 lane units)
    else
    return go(k_dp4a<T, 17, TYPE>, 544);
}
template <int TYPE>
static float dp4a_T(int T, int IT, DpArgs & a, int reps) {
    switch (T) {
        case 1: return run_dp4a<1, TYPE>(IT, a, reps);
#ifndef WV_QUICK
        case 2: return run_dp4a<2, TYPE>(IT, a, reps);
        case 3: return run_dp4a<3, TYPE>(IT, a, reps);
        case 4: return run_dp4a<4, TYPE>(IT, a, reps);
        case 5: return run_dp4a<5, TYPE>(IT, a, reps);
#ifndef WV_DPMAX5
        case 6: return run_dp4a<6, TYPE>(IT, a, reps);
        case 7: return run_dp4a<7, TYPE>(IT, a, reps);
        case 8: return run_dp4a<8, TYPE>(IT, a, reps);
#endif
#endif
    }
    return -1.f;
}
#ifndef WV_V2
#define WV_V2 0 // 1: time / check k_wv2 instead of k_wv
#endif
template <int TYPE, int T, int PF>
static float run_wv(bool q16, WvArgs & a, int reps) {
    const int NSB = a.K / 256, Kc = 256 * ((NSB + a.S - 1) / a.S);
    auto go = [&](auto kfun, int smem) {
        if (smem > 99 * 1024) return -1.f;
        prep(kfun);
        return timeit([&] { coop(kfun, smem, a); }, reps);
    };
#if WV_V2
    if (q16) return go(k_wv2<TYPE, T, true>, wv2_smem<TYPE, T, true>(Kc));
    return go(k_wv2<TYPE, T, false>, wv2_smem<TYPE, T, false>(Kc));
#else
    if (q16) return go(k_wv<TYPE, T, true, PF>, wv_smem<TYPE, T, true>(Kc));
    return go(k_wv<TYPE, T, false, PF>, wv_smem<TYPE, T, false>(Kc));
#endif
}
template <int TYPE, int PF>
static float wv_T(int T, bool q16, WvArgs & a, int reps) {
#ifdef WV_QUICK // experiment builds: T = 1, 4, 8 only
    switch (T) {
#ifndef WV_T8ONLY
        case 1: return run_wv<TYPE, 1, PF>(q16, a, reps);
        case 4: return run_wv<TYPE, 4, PF>(q16, a, reps);
#endif
        case 8: return run_wv<TYPE, 8, PF>(q16, a, reps);
    }
    return -1.f;
#else
    switch (T) {
        case 1: return run_wv<TYPE, 1, PF>(q16, a, reps);
        case 2: return run_wv<TYPE, 2, PF>(q16, a, reps);
        case 3: return run_wv<TYPE, 3, PF>(q16, a, reps);
        case 4: return run_wv<TYPE, 4, PF>(q16, a, reps);
        case 5: return run_wv<TYPE, 5, PF>(q16, a, reps);
        case 6: return run_wv<TYPE, 6, PF>(q16, a, reps);
        case 7: return run_wv<TYPE, 7, PF>(q16, a, reps);
        case 8: return run_wv<TYPE, 8, PF>(q16, a, reps);
        case 16: return run_wv<TYPE, 16, PF>(q16, a, reps);
    }
    return -1.f;
#endif
}
#ifndef WV_PF
#define WV_PF 0
#endif
static float dp4a_any(int ty, int T, int IT, DpArgs & a, int reps) {
#if defined(WV_Q6ONLY) // Q6_K-only builds
    return ty == T_Q6K ? dp4a_T<T_Q6K>(T, IT, a, reps) : -1.f;
#elif defined(WV_QUICK)
    if (T != 1) return -1.f;
    switch (ty) {
        case T_Q4K: return dp4a_T<T_Q4K>(T, IT, a, reps);
        case T_Q5K: return dp4a_T<T_Q5K>(T, IT, a, reps);
        case T_IQ4X: return dp4a_T<T_IQ4X>(T, IT, a, reps);
    }
    return -1.f;
#else
    switch (ty) {
        case T_Q4K: return dp4a_T<T_Q4K>(T, IT, a, reps);
        case T_Q5K: return dp4a_T<T_Q5K>(T, IT, a, reps);
        case T_Q6K: return dp4a_T<T_Q6K>(T, IT, a, reps);
        case T_Q8_0: return dp4a_T<T_Q8_0>(T, IT, a, reps);
        case T_IQ4X: return dp4a_T<T_IQ4X>(T, IT, a, reps);
        case T_IQ4NL: return dp4a_T<T_IQ4NL>(T, IT, a, reps);
        case T_Q3K: return dp4a_T<T_Q3K>(T, IT, a, reps);
        default: return dp4a_T<T_IQ3S>(T, IT, a, reps);
    }
#endif
}
static float wv_any(int ty, int T, bool q16, WvArgs & a, int reps) {
#if defined(WV_DPONLY)
    return -1.f;
#elif defined(WV_Q6ONLY)
    return ty == T_Q6K ? wv_T<T_Q6K, WV_PF>(T, q16, a, reps) : -1.f;
#elif defined(WV_QUICK)
    switch (ty) {
        case T_Q4K: return wv_T<T_Q4K, WV_PF>(T, q16, a, reps);
        case T_Q5K: return wv_T<T_Q5K, WV_PF>(T, q16, a, reps);
        case T_IQ4X: return wv_T<T_IQ4X, WV_PF>(T, q16, a, reps);
    }
    return -1.f;
#else
    switch (ty) {
        case T_Q4K: return wv_T<T_Q4K, WV_PF>(T, q16, a, reps);
        case T_Q5K: return wv_T<T_Q5K, WV_PF>(T, q16, a, reps);
        case T_Q6K: return wv_T<T_Q6K, WV_PF>(T, q16, a, reps);
        case T_Q8_0: return wv_T<T_Q8_0, WV_PF>(T, q16, a, reps);
        case T_IQ4X: return wv_T<T_IQ4X, WV_PF>(T, q16, a, reps);
        case T_IQ4NL: return wv_T<T_IQ4NL, WV_PF>(T, q16, a, reps);
        case T_Q3K: return wv_T<T_Q3K, WV_PF>(T, q16, a, reps);
        default: return wv_T<T_IQ3S, WV_PF>(T, q16, a, reps);
    }
#endif
}

struct Dev {
    std::vector<Mat> mats;
    Mat * dmats = nullptr;
    uint8_t * w = nullptr;
    __half * d = nullptr;
    size_t bytes = 0;
    int nlay = 0;
};
// Host re-layout of a row-major engine matrix for WV_TILEMAJ (the load-time repack the wv kernel would need):
// 1: tile-major units (16-row tile, block idx) of 16 whole blocks, value and meta streams alike;
// 2: Q4_K / Q5_K / IQ4_XS fragment-ordered inside each unit (see wv_load2), the other types as 1.
// WV_Q6FM: Q6_K into the fragment-ordered units of wv_tile_q6f (values + meta in one stream; d is emptied)
static void relayout_q6f(int M, int K, std::vector<uint8_t> & w, std::vector<uint16_t> & d) {
    const int nb = K / 256, rb = nb * Q6_BLK, ndb = nb * Q6_META;
    std::vector<uint8_t> o((size_t) M * nb * 3360 / 16);
    const uint8_t * meta = reinterpret_cast<const uint8_t *>(d.data());
    for (int r = 0; r < M; r++)
        for (int sb = 0; sb < nb; sb++) {
            uint8_t * u = o.data() + ((size_t) (r >> 4) * nb + sb) * 3360;
            const int g = r & 7, half = (r >> 3) & 1;
            const uint8_t * blk = w.data() + (size_t) r * rb + (size_t) sb * Q6_BLK;
            const uint8_t * mp = meta + (size_t) r * ndb + (sb >> 1) * 2 * Q6_META;
            memcpy(u + 32 * g + 16 * half, mp + 16 * (sb & 1), 16);
            memcpy(u + 256 + 4 * g + 2 * half, mp + 32 + 2 * (sb & 1), 2);
            for (int tq = 0; tq < 4; tq++) {
                const int l = 4 * g + tq;
                for (int p = 0; p < 4; p++) {
                    memcpy(u + 288 + 512 * p + 16 * l + 8 * half, blk + 32 * p + 4 * tq, 4);
                    memcpy(u + 288 + 512 * p + 16 * l + 8 * half + 4, blk + 32 * p + 16 + 4 * tq, 4);
                    for (int mm = 0; mm < 4; mm++) {
                        auto hb = [&](int part, int c) { return (blk[128 + 8 * (2 * p + c) + 4 * part + mm] >> (2 * tq)) & 3; };
                        u[(WV_Q6HP ? 2336 + 256 * p + 8 * l + 4 * half : 2336 + 512 * half + 16 * l + 4 * p) + mm] =
                            (uint8_t) ((hb(0, 0) << 4) | (hb(1, 0) << 6) | hb(0, 1) | (hb(1, 1) << 2));
                    }
                }
            }
        }
    w.swap(o);
    d.clear();
}
static void relayout(int ty, int M, int K, std::vector<uint8_t> & w, std::vector<uint16_t> & d, int mode) {
    if (WV_Q6FM && ty == T_Q6K) {
        relayout_q6f(M, K, w, d);
        return;
    }
    if (!mode) return;
    const int rb = row_bytes(ty, K), nb = K / 256, ndb = 2 * row_meta_halves(ty, K);
    std::vector<uint8_t> o(w.size());
    auto tm = [&](const uint8_t * src, uint8_t * dst, int rowb, int BS) { // whole blocks of BS bytes, tile-major
        const int nbk = rowb / BS;
        for (int r = 0; r < M; r++)
            for (int i = 0; i < nbk; i++)
                memcpy(dst + (((size_t) (r >> 4) * nbk + i) * 16 + (r & 15)) * BS, src + (size_t) r * rowb + (size_t) i * BS, BS);
    };
    const bool fm = mode == 2 && (ty == T_Q4K || ty == T_Q5K || ty == T_IQ4X);
    if (fm) {
        const int BS = ty == T_Q4K ? 144 : ty == T_Q5K ? 176 : 136;
        for (int r = 0; r < M; r++)
            for (int sb = 0; sb < nb; sb++) {
                uint8_t * u = o.data() + ((size_t) (r >> 4) * nb + sb) * 16 * BS;
                const int g = r & 7, half = (r >> 3) & 1;
                const uint8_t * row = w.data() + (size_t) r * rb;
                for (int tq = 0; tq < 4; tq++) {
                    const int l = 4 * g + tq;
                    if (ty == T_IQ4X) {
                        const uint8_t * pr = row + (sb >> 1) * 272;
                        const int odd = sb & 1;
                        memcpy(u + 16 * g + 8 * half, pr + 8 * odd, 8);
                        for (int p = 0; p < 4; p++) memcpy(u + 128 + 512 * p + 16 * l + 8 * half, pr + 16 + 128 * odd + 32 * p + 8 * tq, 8);
                    } else {
                        const uint8_t * blk = row + (size_t) sb * BS;
                        memcpy(u + 32 * g + 16 * half, blk, 16);
                        const int qs = ty == T_Q4K ? 16 : 48, qo = ty == T_Q4K ? 256 : 768;
                        if (ty == T_Q5K) memcpy(u + 256 + 16 * l + 8 * half, blk + 16 + 8 * tq, 8);
                        for (int p = 0; p < 4; p++) memcpy(u + qo + 512 * p + 16 * l + 8 * half, blk + qs + 32 * p + 8 * tq, 8);
                    }
                }
            }
    } else {
        const int BS = ty == T_Q4K || ty == T_IQ4NL ? 144 : ty == T_Q5K ? 176 : ty == T_IQ4X ? 272 : ty == T_Q6K ? Q6_BLK
                     : ty == T_Q8_0 ? 256 : Q3_BLK;
        tm(w.data(), o.data(), rb, BS);
    }
    w.swap(o);
    if (ndb) {
        const int MB = ty == T_Q6K || ty == T_Q3K ? 2 * Q6_META : 16;
        std::vector<uint16_t> od(d.size());
        tm(reinterpret_cast<const uint8_t *>(d.data()), reinterpret_cast<uint8_t *>(od.data()), ndb, MB);
        d.swap(od);
    }
}
static Dev upload_wd(int ty, int M, int K, int nlay, const std::vector<uint8_t> & w, const std::vector<uint16_t> & d);
static Dev upload(int ty, int M, int K, int nlay, std::mt19937 & rng) {
    std::vector<uint8_t> w;
    std::vector<uint16_t> d;
    make_weights(ty, M, K, w, d, rng);
    relayout(ty, M, K, w, d, WV_TILEMAJ); // timing: the bytes the wv kernel reads (dp4a reads them as row-major: same cost)
    return upload_wd(ty, M, K, nlay, w, d);
}
static Dev upload_wd(int ty, int M, int K, int nlay, const std::vector<uint8_t> & w, const std::vector<uint16_t> & d) {
    Dev D;
    const size_t mw = w.size(), md = d.size();
    D.nlay = nlay;
    CK(cudaMalloc(&D.w, mw * nlay));
    if (md) CK(cudaMalloc(&D.d, md * 2 * nlay));
    for (int L = 0; L < nlay; L++) { // the same random matrix in every layer copy (distinct addresses defeat L2)
        CK(cudaMemcpy(D.w + mw * L, w.data(), mw, cudaMemcpyHostToDevice));
        if (md) CK(cudaMemcpy(D.d + md * L, d.data(), md * 2, cudaMemcpyHostToDevice));
        Mat m{D.w + mw * L, md ? D.d + md * L : nullptr, ty, K, M, row_bytes(ty, K), row_meta_halves(ty, K)};
        D.mats.push_back(m);
    }
    D.bytes = (mw + md * 2);
    CK(cudaMalloc(&D.dmats, sizeof(Mat) * nlay));
    CK(cudaMemcpy(D.dmats, D.mats.data(), sizeof(Mat) * nlay, cudaMemcpyHostToDevice));
    return D;
}
static void release(Dev & D) {
    cudaFree(D.w);
    if (D.d) cudaFree(D.d);
    cudaFree(D.dmats);
}

struct Acts {
    int8_t * xa, * xb;
    uint2 * meta;
    QB q;
};
static Acts upload_acts(int K, int ntok, bool q16, std::mt19937 & rng) {
    std::vector<int8_t> xa, xb;
    std::vector<uint2> meta;
    make_acts(K, ntok, q16, xa, xb, meta, rng);
    Acts A;
    CK(cudaMalloc(&A.xa, xa.size()));
    CK(cudaMalloc(&A.xb, xb.size()));
    CK(cudaMalloc(&A.meta, meta.size() * 8));
    CK(cudaMemcpy(A.xa, xa.data(), xa.size(), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(A.xb, xb.data(), xb.size(), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(A.meta, meta.data(), meta.size() * 8, cudaMemcpyHostToDevice));
    A.q = QB{A.xa, A.xb, A.meta, K / 32};
    return A;
}

static const int ALLTYPES[] = {T_Q4K, T_Q5K, T_IQ4X, T_Q6K, T_IQ4NL, T_Q8_0, T_Q3K, T_IQ3S};

static void init_grid(std::mt19937 & rng) {
    std::vector<uint32_t> grid(512);
    for (auto & gv : grid) {
        uint32_t v = 0;
        for (int k = 0; k < 4; k++) v |= (uint32_t) (1 + 2 * (rng() % 8)) << (8 * k);
        gv = v;
    }
    CK(cudaMemcpyToSymbol(g_iq3s_grid, grid.data(), 512 * 4));
}

// exactness: per type x role: wv at T = 1..8, 16 bit-identical per (row, token) to the largest T that ran; wv and the
// engine's dp4a (T = 4) vs fp64 within CHK_RMS / CHK_MAX of the rows' rms, |wv - dp4a| within CHK_MAX (NaN fails)
constexpr double CHK_RMS = 1e-5, CHK_MAX = 1e-4; // (correct kernels: rms errors <= 6e-7, max <= 4.5e-6)
static int check(const std::vector<int> & roles, const std::vector<int> & types) {
    std::mt19937 rng(1234);
    init_grid(rng);
    int bad = 0;
    const int TS[] = {1, 2, 3, 4, 5, 6, 7, 8, 16};
    for (int ri : roles) {
        const Role & R = ROLES[ri];
        Acts X = upload_acts(R.K, 16, R.q16, rng);
        for (int ty : types) {
            std::vector<uint8_t> hw;
            std::vector<uint16_t> hd16;
            make_weights(ty, R.M, R.K, hw, hd16, rng);
            Dev D = upload_wd(ty, R.M, R.K, 1, hw, hd16); // row-major: the fp64 reference and dp4a
            relayout(ty, R.M, R.K, hw, hd16, WV_TILEMAJ);
            Dev D2 = upload_wd(ty, R.M, R.K, 1, hw, hd16); // the wv kernel's layout
            float * out, * part;
            unsigned * cnt;
            double * ref;
            long long * idot;
            CK(cudaMalloc(&out, sizeof(float) * 16 * R.M));
            CK(cudaMalloc(&part, sizeof(float) * 16 * R.M * R.S));
            CK(cudaMalloc(&cnt, 4 * (R.M / 16)));
            CK(cudaMemset(cnt, 0, 4 * (R.M / 16)));
            CK(cudaMalloc(&ref, sizeof(double) * 16 * R.M));
            CK(cudaMalloc(&idot, sizeof(long long) * 16 * R.M));
            k_ref<<<R.M, 16 * 32>>>(D.mats[0], X.q, 16, R.K, ref, idot);
            CK(cudaGetLastError());
            std::vector<double> href((size_t) 16 * R.M);
            CK(cudaMemcpy(href.data(), ref, href.size() * 8, cudaMemcpyDeviceToHost));
            std::vector<std::vector<float>> res;
            std::vector<char> ran; // the T's whose kernel launched (the others need more shared memory than there is)
            for (int T : TS) {
                CK(cudaMemset(out, 0xff, sizeof(float) * 16 * R.M));
                WvArgs a{D2.dmats, 1, X.q, out, part, cnt, R.M, R.K, R.S, g_scr};
                const float ms = wv_any(ty, T, R.q16, a, 0);
                CK(cudaDeviceSynchronize());
                std::vector<float> h((size_t) 16 * R.M);
                CK(cudaMemcpy(h.data(), out, h.size() * 4, cudaMemcpyDeviceToHost));
                if (ms < 0) h.assign(h.size(), NAN);
                ran.push_back(ms >= 0);
                res.push_back(h);
            }
            // T-invariance against the largest T that ran (rows t < T); none ran: a failure (tref 0, its rows NaN)
            int tinv_bad = 0, tref = 0, ntested = 0;
            for (int ti = (int) res.size() - 1; ti > 0; ti--)
                if (ran[ti]) {
                    tref = ti;
                    break;
                }
            for (size_t ti = 0; ti < res.size(); ti++) {
                const int T = TS[ti];
                if (!ran[ti]) continue;
                ntested++;
                for (int t = 0; t < T; t++)
                    if (memcmp(res[ti].data() + (size_t) t * R.M, res[tref].data() + (size_t) t * R.M, 4 * R.M)) tinv_bad++;
            }
            const int Tmax = TS[tref];
            const std::vector<float> & best = res[tref];
            const int Td = 4; // dp4a reference rows
            CK(cudaMemset(out, 0, sizeof(float) * 16 * R.M));
            DpArgs da{D.dmats, 1, X.q, out, R.M, g_park};
            const bool dp_ran = dp4a_any(ty, Td, R.IT, da, 0) >= 0;
            CK(cudaDeviceSynchronize());
            std::vector<float> hd((size_t) 16 * R.M);
            CK(cudaMemcpy(hd.data(), out, hd.size() * 4, cudaMemcpyDeviceToHost));
            double ew = 0, ed = 0, mw = 0, md = 0, rms = 0, dwd = 0;
            size_t n = 0, nd = 0;
            for (int t = 0; t < Tmax; t++)
                for (int r = 0; r < R.M; r++) {
                    const double rv = href[(size_t) t * R.M + r];
                    const double vw = best[(size_t) t * R.M + r];
                    rms += rv * rv;
                    ew += (vw - rv) * (vw - rv);
                    mw = fmax(mw, fabs(vw - rv));
                    n++;
                    if (t < Td) {
                        const double vd = hd[(size_t) t * R.M + r];
                        ed += (vd - rv) * (vd - rv);
                        md = fmax(md, fabs(vd - rv));
                        dwd = fmax(dwd, fabs(vd - (double) best[(size_t) t * R.M + r]));
                        nd++;
                    }
                }
            rms = sqrt(rms / n);
            unsigned long long hsh = 1469598103934665603ull; // FNV-1a of rows t < 8 (compare across layouts / builds)
            {
                const unsigned char * hb = reinterpret_cast<const unsigned char *>(best.data());
                const size_t nbh = (size_t) 4 * R.M * (Tmax < 8 ? Tmax : 8);
                for (size_t i = 0; i < nbh; i++) hsh = (hsh ^ hb[i]) * 1099511628211ull;
            }
            const double e_w = sqrt(ew / n) / rms, m_w = mw / rms, e_d = sqrt(ed / nd) / rms, m_d = md / rms;
            const double m_wd = dwd / rms;
            const bool in_bounds =
                e_w <= CHK_RMS && m_w <= CHK_MAX && e_d <= CHK_RMS && m_d <= CHK_MAX && m_wd <= CHK_MAX;
            printf("%-9s %-6s T-inv(%d T's, ref T=%d) %s (%d rows differ) | rel err vs fp64: wv rms %.2e max %.2e, dp4a rms "
                   "%.2e max %.2e | max |wv-dp4a| %.2e | hash %016llx%s\n",
                   R.name, tname(ty), ntested, Tmax, tinv_bad || !ntested ? "FAIL" : "ok", tinv_bad, e_w, m_w, e_d,
                   m_d, m_wd, hsh, !dp_ran ? " | FAIL: dp4a did not run" : !in_bounds ? " | FAIL: error bound" : "");
            if (tinv_bad || !ntested || !dp_ran || !in_bounds) bad++;
            cudaFree(out); cudaFree(part); cudaFree(cnt); cudaFree(ref); cudaFree(idot);
            release(D);
            release(D2);
        }
        cudaFree(X.xa); cudaFree(X.xb); cudaFree(X.meta);
    }
    printf(bad ? "CHECK FAILED (%d)\n" : "CHECK OK\n", bad);
    return bad;
}

static void timing(const std::vector<int> & roles, const std::vector<int> & types, int reps, const std::vector<int> & wvT,
                   const std::vector<int> & dpT) {
    std::mt19937 rng(77);
    init_grid(rng);
    for (int ri : roles) {
        const Role & R = ROLES[ri];
        Acts X = upload_acts(R.K, 16, R.q16, rng);
        for (int ty : types) {
            const size_t mb = (size_t) R.M * row_bytes(ty, R.K) + (size_t) R.M * row_meta_halves(ty, R.K) * 2;
            int nlay = (int) ((320u << 20) / mb);
            nlay = nlay < 2 ? 2 : nlay > 24 ? 24 : nlay;
            // dp4a reads the row-major copy, wv its own layout (the same bytes per matrix; separate copies since FM Q6_K
            // also moves the meta stream into the value units)
            std::vector<uint8_t> hw;
            std::vector<uint16_t> hd16;
            make_weights(ty, R.M, R.K, hw, hd16, rng);
            Dev D = upload_wd(ty, R.M, R.K, nlay, hw, hd16);
            relayout(ty, R.M, R.K, hw, hd16, WV_TILEMAJ);
            Dev D2 = upload_wd(ty, R.M, R.K, nlay, hw, hd16);
            float * out, * part;
            unsigned * cnt;
            CK(cudaMalloc(&out, sizeof(float) * 16 * R.M));
            CK(cudaMalloc(&part, sizeof(float) * 16 * R.M * R.S));
            CK(cudaMalloc(&cnt, 4 * (R.M / 16) * nlay));
            CK(cudaMemset(cnt, 0, 4 * (R.M / 16) * nlay));
            const double gb = (double) D.bytes * nlay / 1e9;
            printf("%-9s %-6s %5.1f MB/layer x %2d |", R.name, tname(ty), D.bytes / 1e6, nlay);
            DpArgs da{D.dmats, nlay, X.q, out, R.M, g_park};
            for (int T : dpT) {
                const float ms = dp4a_any(ty, T, R.IT, da, reps);
                if (ms < 0) printf(" dp4a%d    -    ", T);
                else printf(" dp4a%d %6.1fus %4.0f", T, 1e3 * ms / nlay, gb / ms * 1e3);
            }
            printf(" |");
            WvArgs a{D2.dmats, nlay, X.q, out, part, cnt, R.M, R.K, R.S, g_scr};
            for (int T : wvT) {
                const float ms = wv_any(ty, T, R.q16, a, reps);
                if (ms < 0) printf(" wv%d    -    ", T);
                else printf(" wv%d %6.1fus %4.0f", T, 1e3 * ms / nlay, gb / ms * 1e3);
            }
            printf("\n");
            fflush(stdout);
            cudaFree(out); cudaFree(part); cudaFree(cnt);
            release(D);
            release(D2);
        }
        cudaFree(X.xa); cudaFree(X.xb); cudaFree(X.meta);
    }
}

static std::vector<int> parse_list(const char * s, bool roles) {
    std::vector<int> v;
    std::string str(s);
    size_t pos = 0;
    while (pos <= str.size()) {
        size_t e = str.find(',', pos);
        if (e == std::string::npos) e = str.size();
        std::string tok = str.substr(pos, e - pos);
        if (roles) {
            for (int i = 0; i < (int) (sizeof(ROLES) / sizeof(ROLES[0])); i++)
                if (tok == ROLES[i].name || tok == "all") v.push_back(i);
        } else if (tok == "all") {
            v.assign(std::begin(ALLTYPES), std::end(ALLTYPES));
        } else {
            for (int ty : ALLTYPES)
                if (tok == tname(ty)) v.push_back(ty);
        }
        pos = e + 1;
    }
    return v;
}

int main(int argc, char ** argv) {
    cudaDeviceGetAttribute(&g_nsm, cudaDevAttrMultiProcessorCount, 0);
    CK(cudaGetSymbolAddress(&g_bar_ptr, g_bar));
    CK(cudaMalloc(&g_scr, sizeof(float) * g_nsm * NW * 2 * 16 * 16));
    CK(cudaMalloc(&g_park, sizeof(float) * 5120 * 8 * 32));
    const std::string mode = argc > 1 ? argv[1] : "check";
    const std::vector<int> roles = parse_list(argc > 2 ? argv[2] : "all", true);
    const std::vector<int> types = parse_list(argc > 3 ? argv[3] : "all", false);
    if (argc > 6) { // split-K override: one value for every role or a list in ROLES order; 0 keeps the role's default
        const char * q = argv[6];
        for (int i = 0; i < (int) (sizeof(ROLES) / sizeof(ROLES[0])); i++) {
            const int v = atoi(q);
            if (v > 0) ROLES[i].S = v;
            const char * c = strchr(q, ',');
            if (c) q = c + 1;
        }
    }
    if (mode == "check") return check(roles, types) ? 1 : 0;
    const int reps = argc > 4 ? atoi(argv[4]) : 5;
    std::vector<int> wvT = {1, 2, 4, 8, 16}, dpT = {1, 2, 4, 5};
    auto ints = [](const char * s) {
        std::vector<int> v;
        for (const char * p = s; *p;) {
            v.push_back(atoi(p));
            while (*p && *p != ',') p++;
            if (*p == ',') p++;
        }
        return v;
    };
    if (argc > 5 && strcmp(argv[5], "-")) wvT = ints(argv[5]);
    if (argc > 7) dpT = ints(argv[7]);
    timing(roles, types, reps, wvT, dpT);
    return 0;
}
