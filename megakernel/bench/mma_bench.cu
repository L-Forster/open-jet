// ffn gate/up at T = 8 .. 16 rows: the dp4a GEMV (ffn_gu's rows_q4k) against int8 tensor-core mma (m16n8k16)
// variants, 17408 rows x 2 matrices (Q4_K, K = 5120), several layers to defeat L2. Standalone (ncu works on it).
//   nvcc -O3 -arch=sm_86 -std=c++17 -lineinfo mma_bench.cu -o mma_bench && ./mma_bench
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1);} } while (0)
constexpr int K = 5120, U = K / 32, UP = U + 1, RB = K / 256 * 144, FF = 17408, NT = 512, NW = 16, M4 = 0x0F0F0F0F, NLAY = 8;
constexpr int TMAX = 12;

__device__ __forceinline__ int4 ldw4(const void * p) {
    int4 r;
    asm("ld.global.nc.L1::no_allocate.v4.s32 {%0,%1,%2,%3}, [%4];" : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
    return r;
}
__device__ __forceinline__ unsigned ldw1(const void * p) {
    unsigned r;
    asm("ld.global.nc.L1::no_allocate.u32 %0, [%1];" : "=r"(r) : "l"(p));
    return r;
}
__device__ __forceinline__ unsigned ldw1c(const void * p) { // L1-allocating: the other half of the 32-byte sector hits L1
    unsigned r;
    asm("ld.global.nc.u32 %0, [%1];" : "=r"(r) : "l"(p));
    return r;
}
__device__ __forceinline__ int4 ldw4c(const void * p) {
    int4 r;
    asm("ld.global.nc.v4.s32 {%0,%1,%2,%3}, [%4];" : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
    return r;
}
__device__ __forceinline__ uint2 ldw2(const void * p) {
    uint2 r;
    asm("ld.global.nc.L1::no_allocate.v2.u32 {%0,%1}, [%2];" : "=r"(r.x), "=r"(r.y) : "l"(p));
    return r;
}
__device__ __forceinline__ uint2 ldw2c(const void * p) {
    uint2 r;
    asm("ld.global.nc.v2.u32 {%0,%1}, [%2];" : "=r"(r.x), "=r"(r.y) : "l"(p));
    return r;
}
__device__ __forceinline__ int4 sel4(bool c, const int4 & a, const int4 & b) { // c ? a : b, componentwise (no dynamic index)
    return make_int4(c ? a.x : b.x, c ? a.y : b.y, c ? a.z : b.z, c ? a.w : b.w);
}
__device__ __forceinline__ float wsum(float v) { for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(~0u, v, o); return v; }
__device__ __forceinline__ void q4k_scales(const int4 & h, int p, float & fA, float & gA, float & fB, float & gB) {
    const float2 dm = __half22float2(*reinterpret_cast<const __half2 *>(&h.x));
    const int sh = (p & 1) * 16; const unsigned y = (unsigned) h.y >> sh, z = (unsigned) h.z >> sh, w = (unsigned) h.w >> sh;
    unsigned sc0, sc1, m0, m1;
    if (p < 2) { sc0 = y & 63; sc1 = (y >> 8) & 63; m0 = z & 63; m1 = (z >> 8) & 63; }
    else { sc0 = (w & 0xF) | ((y >> 2) & 0x30); sc1 = ((w >> 8) & 0xF) | ((y >> 10) & 0x30); m0 = ((w >> 4) & 0xF) | ((z >> 2) & 0x30); m1 = ((w >> 12) & 0xF) | ((z >> 10) & 0x30); }
    fA = dm.x * sc0; fB = dm.x * sc1; gA = dm.y * m0; gB = dm.y * m1;
}
__device__ __forceinline__ void mma_s8(int (&d)[4], unsigned a0, unsigned a1, unsigned b0) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%7,%7,%7};"
                 : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3]) : "r"(a0), "r"(a1), "r"(b0), "r"(0));
}

// activations in shared memory: T rows of U units (xa / xb: 16 int8 each, xm: half2 scales + short2 sums), row stride S
struct SX { int4 * xa; int4 * xb; uint2 * xm; };
__device__ void setup(SX & s, uint8_t * sm, int T, int S) {
    s.xa = (int4 *) sm; s.xb = s.xa + TMAX * UP; s.xm = (uint2 *) (s.xb + TMAX * UP);
    for (int i = threadIdx.x; i < T * U; i += NT) {
        const int t = i / U, u = i % U, d = t * S + u;
        s.xa[d] = make_int4(0x01020304 + t, 0x05060708, 0x01010101 * (u & 3), 0x7f7f7f7f);
        s.xb[d] = make_int4(0x03020104, 0x05060708 - t, 0x01010101, 0x7f7f7f7f);
        const __half2 dx = __floats2half2_rn(0.01f, 0.02f);
        s.xm[d] = make_uint2(*reinterpret_cast<const unsigned *>(&dx), 0x00050003u);
    }
    __syncthreads();
}

// A: the megakernel's dp4a path (ffn_gu: 16-row warp items, RG rows of each matrix per step, lanes split the units)
template <int T, int R>
__device__ __forceinline__ void rows_q4k(const uint8_t * const (&rp)[R], const SX & s, float (&acc)[R][T]) {
    const int lane = threadIdx.x & 31;
#pragma unroll(T >= 5 ? 1 : 5)
    for (int i = 0; i < 5; i++) {
        const int u = lane + 32 * i, b = u >> 3, p = (u >> 1) & 3, hh = u & 1;
        int4 h[R], q[R];
#pragma unroll
        for (int k = 0; k < R; k++) { const uint8_t * blk = rp[k] + b * 144; h[k] = ldw4(blk); q[k] = ldw4(blk + 16 + 32 * p + 16 * hh); }
        float fA[R], gA[R], fB[R], gB[R];
#pragma unroll
        for (int k = 0; k < R; k++) q4k_scales(h[k], p, fA[k], gA[k], fB[k], gB[k]);
#pragma unroll
        for (int t = 0; t < T; t++) {
            const int ix = t * U + u; const int4 va = s.xa[ix], vb = s.xb[ix]; const uint2 mm = s.xm[ix];
            const float2 dx = __half22float2(*reinterpret_cast<const __half2 *>(&mm.x)); const int qq = (int) mm.y;
            const float sqA = dx.x * (float) (short) (qq & 0xffff), sqB = dx.y * (float) (short) (qq >> 16);
#pragma unroll
            for (int k = 0; k < R; k++) {
                int sa = __dp4a(q[k].x & M4, va.x, 0); sa = __dp4a(q[k].y & M4, va.y, sa); sa = __dp4a(q[k].z & M4, va.z, sa); sa = __dp4a(q[k].w & M4, va.w, sa);
                int sb = __dp4a((q[k].x >> 4) & M4, vb.x, 0); sb = __dp4a((q[k].y >> 4) & M4, vb.y, sb); sb = __dp4a((q[k].z >> 4) & M4, vb.z, sb); sb = __dp4a((q[k].w >> 4) & M4, vb.w, sb);
                acc[k][t] += fA[k] * (dx.x * (float) sa) + fB[k] * (dx.y * (float) sb) - gA[k] * sqA - gB[k] * sqB;
            }
        }
    }
}
template <int T>
__global__ void __launch_bounds__(NT, 1) ffn_dp4a(const uint8_t * Wg, const uint8_t * Wu, float * out, size_t ls) {
    extern __shared__ __align__(16) uint8_t sm[]; SX s; setup(s, sm, T, U);
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, gw = warp * gridDim.x + blockIdx.x, NWt = NW * gridDim.x;
    constexpr int RG = T >= 3 ? 1 : 2;
    for (int L = 0; L < NLAY; L++) {
        const uint8_t * gw_ = Wg + L * ls, * uw_ = Wu + L * ls;
        for (int j = gw; j < FF / 16; j += NWt) {
            float act[T] = {};
#pragma unroll 1
            for (int r = 0; r < 16; r += RG) {
                const uint8_t * rp[2 * RG];
#pragma unroll
                for (int k = 0; k < RG; k++) { rp[k] = gw_ + (size_t) (16 * j + r + k) * RB; rp[RG + k] = uw_ + (size_t) (16 * j + r + k) * RB; }
                float a[2 * RG][T] = {};
                rows_q4k<T, 2 * RG>(rp, s, a);
#pragma unroll
                for (int k = 0; k < 2 * RG; k++)
#pragma unroll
                    for (int t = 0; t < T; t++) a[k][t] = wsum(a[k][t]);
#pragma unroll
                for (int k = 0; k < RG; k++) if (lane == r + k)
#pragma unroll
                    for (int t = 0; t < T; t++) act[t] = a[k][t] / (1.f + __expf(-a[k][t])) * a[RG + k][t];
            }
            if (lane < 16) for (int t = 0; t < T; t++) out[(size_t) t * FF + 16 * j + lane] = act[t];
        }
        __syncthreads();
    }
}

// M: int8 mma. Warp item = 16 rows of both matrices; lane (g, tg) holds rows g, g + 8 for tokens 2 tg, 2 tg + 1 of
// each n-tile. Per 16-value chunk: one mma per matrix and n-tile, then the chunk scales in fp32.
// PB: blocks of weights loaded per step (registers: 4 rows x (4 header + 8 qs words) per block)
template <int T, int PB>
__global__ void __launch_bounds__(NT, 1) ffn_mma(const uint8_t * Wg, const uint8_t * Wu, float * out, size_t ls) {
    constexpr int NTL = (T + 7) / 8;
    extern __shared__ __align__(16) uint8_t sm[]; SX s; setup(s, sm, T, UP);
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, gw = warp * gridDim.x + blockIdx.x, NWt = NW * gridDim.x;
    const int g = lane >> 2, tg = lane & 3;
    const unsigned * xa = reinterpret_cast<const unsigned *>(s.xa);
    const unsigned * xb = reinterpret_cast<const unsigned *>(s.xb);
    for (int L = 0; L < NLAY; L++) {
        for (int j = gw; j < FF / 16; j += NWt) {
            const uint8_t * w[4] = {Wg + L * ls + (size_t) (16 * j + g) * RB, Wg + L * ls + (size_t) (16 * j + g + 8) * RB,
                                    Wu + L * ls + (size_t) (16 * j + g) * RB, Wu + L * ls + (size_t) (16 * j + g + 8) * RB};
            float acc[2][NTL][4] = {};
#pragma unroll 1
            for (int b0 = 0; b0 < K / 256; b0 += PB) {
                int4 h[PB][4];
                unsigned qw[PB][4][8];
#pragma unroll
                for (int bb = 0; bb < PB; bb++)
#pragma unroll
                    for (int k = 0; k < 4; k++) {
                        h[bb][k] = ldw4(w[k] + (b0 + bb) * 144);
#pragma unroll
                        for (int i = 0; i < 8; i++) qw[bb][k][i] = ldw1(w[k] + (b0 + bb) * 144 + 16 + 16 * i + 4 * tg);
                    }
#pragma unroll
                for (int bb = 0; bb < PB; bb++)
#pragma unroll
                for (int p = 0; p < 4; p++) {
                    float fA[4], gA[4], fB[4], gB[4];
#pragma unroll
                    for (int k = 0; k < 4; k++) q4k_scales(h[bb][k], p, fA[k], gA[k], fB[k], gB[k]);
#pragma unroll
                    for (int hh = 0; hh < 2; hh++) {
                        const int u = 8 * (b0 + bb) + 2 * p + hh;
#pragma unroll
                        for (int nt = 0; nt < NTL; nt++) {
                            const int n = 8 * nt + g;
                            const unsigned bA = n < T ? xa[(n * UP + u) * 4 + tg] : 0u;
                            const unsigned bB = n < T ? xb[(n * UP + u) * 4 + tg] : 0u;
                            float dxA[2], dxB[2], sqA[2], sqB[2];
#pragma unroll
                            for (int e = 0; e < 2; e++) {
                                const int t = 8 * nt + 2 * tg + e;
                                dxA[e] = dxB[e] = sqA[e] = sqB[e] = 0.f;
                                if (t < T) {
                                    const uint2 mm = s.xm[t * UP + u];
                                    const float2 dx = __half22float2(*reinterpret_cast<const __half2 *>(&mm.x));
                                    const int qq = (int) mm.y;
                                    dxA[e] = dx.x; dxB[e] = dx.y;
                                    sqA[e] = dx.x * (float) (short) (qq & 0xffff); sqB[e] = dx.y * (float) (short) (qq >> 16);
                                }
                            }
#pragma unroll
                            for (int m = 0; m < 2; m++) {
                                const unsigned q0 = qw[bb][2 * m][2 * p + hh], q1 = qw[bb][2 * m + 1][2 * p + hh];
                                int dA[4], dB[4];
                                mma_s8(dA, q0 & M4, q1 & M4, bA);
                                mma_s8(dB, (q0 >> 4) & M4, (q1 >> 4) & M4, bB);
#pragma unroll
                                for (int c = 0; c < 4; c++) {
                                    const int r = 2 * m + (c >> 1), e = c & 1;
                                    acc[m][nt][c] += fA[r] * (dxA[e] * (float) dA[c]) + fB[r] * (dxB[e] * (float) dB[c]) - gA[r] * sqA[e] - gB[r] * sqB[e];
                                }
                            }
                        }
                    }
                }
            }
#pragma unroll
            for (int nt = 0; nt < NTL; nt++)
#pragma unroll
                for (int c = 0; c < 4; c++) {
                    const int t = 8 * nt + 2 * tg + (c & 1), row = 16 * j + g + 8 * (c >> 1);
                    const float a = acc[0][nt][c];
                    if (t < T) out[(size_t) t * FF + row] = a / (1.f + __expf(-a)) * acc[1][nt][c];
                }
        }
        __syncthreads();
    }
}


// M32: activations quantised per 32 values (one Q4_K sub-block: llama.cpp's q8_1 granularity), so a sub-block is one
// m16n8k32 mma per matrix and n-tile and one scale step. Shared layout per token row (stride XS words): sub-block j's
// 8 words permuted [w0 w4 w1 w5 w2 w6 w3 w7], so lane tg's b fragment (words tg, tg + 4) is one 64-bit load;
// scales xd[j][t] = (dx, dx * sum) so a lane's two tokens (2 tg, 2 tg + 1) are one 128-bit load.
constexpr int NJ = K / 32, XS = NJ * 8 + 8;
__device__ __forceinline__ void mma_s8_k32(int (&d)[4], unsigned a0, unsigned a1, unsigned a2, unsigned a3, uint2 b) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%10,%10,%10};"
                 : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b.x), "r"(b.y), "r"(0));
}
__host__ __device__ inline int8_t xval(int t, int k) { return (int8_t) (((t * 7 + k * 13) % 251) - 125); }
__host__ __device__ inline float xdx(int t, int j) { return 0.004f + 0.0005f * ((t + 3 * j) % 7); }
__device__ void setup32(uint8_t * sm, int T, bool natural = false) {
    unsigned * xw = (unsigned *) sm;
    float2 * xd = (float2 *) (xw + TMAX * XS);
    for (int i = threadIdx.x; i < T * NJ; i += NT) {
        const int t = i / NJ, j = i % NJ;
        int sum = 0;
        unsigned w[8];
        for (int q = 0; q < 8; q++) {
            unsigned v = 0;
            for (int b = 0; b < 4; b++) { const int8_t x = xval(t, 32 * j + 4 * q + b); sum += x; v |= (unsigned) (uint8_t) x << (8 * b); }
            w[q] = v;
        }
        const int perm[8] = {0, 4, 1, 5, 2, 6, 3, 7};
        for (int q = 0; q < 8; q++) xw[t * XS + 8 * j + q] = w[natural ? q : perm[q]];
        xd[j * TMAX + t] = make_float2(xdx(t, j), xdx(t, j) * (float) sum);
    }
    __syncthreads();
}
__device__ void setup_q32(uint8_t * sm, int T);
template <int T>
__global__ void __launch_bounds__(NT, 1) ffn_mma32(const uint8_t * Wg, const uint8_t * Wu, float * out, size_t ls) {
    constexpr int NTL = (T + 7) / 8;
    extern __shared__ __align__(16) uint8_t sm[];
    setup32(sm, T);
    const unsigned * xw = (const unsigned *) sm;
    const float2 * xd = (const float2 *) (xw + TMAX * XS);
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, gw = warp * gridDim.x + blockIdx.x, NWt = NW * gridDim.x;
    const int g = lane >> 2, tg = lane & 3;
    for (int L = 0; L < NLAY; L++) {
        for (int j = gw; j < FF / 16; j += NWt) {
            const uint8_t * w[4] = {Wg + L * ls + (size_t) (16 * j + g) * RB, Wg + L * ls + (size_t) (16 * j + g + 8) * RB,
                                    Wu + L * ls + (size_t) (16 * j + g) * RB, Wu + L * ls + (size_t) (16 * j + g + 8) * RB};
            float acc[2][NTL][4] = {};
#pragma unroll 1
            for (int b = 0; b < K / 256; b++) {
                int4 h[4];
                unsigned qw[4][8];
#pragma unroll
                for (int k = 0; k < 4; k++) {
                    h[k] = ldw4(w[k] + b * 144);
#pragma unroll
                    for (int i = 0; i < 8; i++) qw[k][i] = ldw1(w[k] + b * 144 + 16 + 16 * i + 4 * tg);
                }
#pragma unroll
                for (int p = 0; p < 4; p++) {
                    float fA[4], gA[4], fB[4], gB[4];
#pragma unroll
                    for (int k = 0; k < 4; k++) q4k_scales(h[k], p, fA[k], gA[k], fB[k], gB[k]);
#pragma unroll
                    for (int half = 0; half < 2; half++) { // sub-block 2p (low nibbles), 2p + 1 (high nibbles)
                        const int sj = 8 * b + 2 * p + half, sh = 4 * half;
#pragma unroll
                        for (int nt = 0; nt < NTL; nt++) {
                            const int n = 8 * nt + g;
                            const uint2 bf = n < T ? *reinterpret_cast<const uint2 *>(xw + n * XS + 8 * sj + 2 * tg) : make_uint2(0, 0);
                            const float4 sc = *reinterpret_cast<const float4 *>(xd + sj * TMAX + 8 * nt + 2 * tg); // (dx, sq) x 2 tokens
#pragma unroll
                            for (int m = 0; m < 2; m++) {
                                const unsigned * r0 = qw[2 * m], * r1 = qw[2 * m + 1];
                                int d[4];
                                mma_s8_k32(d, (r0[2 * p] >> sh) & M4, (r1[2 * p] >> sh) & M4, (r0[2 * p + 1] >> sh) & M4, (r1[2 * p + 1] >> sh) & M4, bf);
#pragma unroll
                                for (int c = 0; c < 4; c++) {
                                    const int r = 2 * m + (c >> 1);
                                    const float f = half ? fB[r] : fA[r], gm = half ? gB[r] : gA[r];
                                    const float dx = (c & 1) ? sc.z : sc.x, sq = (c & 1) ? sc.w : sc.y;
                                    acc[m][nt][c] += f * (dx * (float) d[c]) - gm * sq;
                                }
                            }
                        }
                    }
                }
            }
#pragma unroll
            for (int nt = 0; nt < NTL; nt++)
#pragma unroll
                for (int c = 0; c < 4; c++) {
                    const int t = 8 * nt + 2 * tg + (c & 1), row = 16 * j + g + 8 * (c >> 1);
                    const float a = acc[0][nt][c];
                    if (t < T) out[(size_t) t * FF + row] = a / (1.f + __expf(-a)) * acc[1][nt][c];
                }
        }
        __syncthreads();
    }
}

// M32K: M32 split over K: the block's 16 warps share one 16-row item (gate + up), warp w takes p-chunks [5w, 5w + 5)
// (sub-blocks [10w, 10w + 10), within 2 Q4_K blocks), all of its weight words loaded up front (72 registers);
// partial sums meet in shared memory (atomicAdd), then silu(g) * u per (row, token)
template <int T, bool C = false, int PF = 0>
__global__ void __launch_bounds__(NT, 1) ffn_mma32k(const uint8_t * Wg, const uint8_t * Wu, float * out, size_t ls) {
    constexpr int NTL = (T + 7) / 8;
    extern __shared__ __align__(16) uint8_t sm[];
    setup32(sm, T);
    const unsigned * xw = (const unsigned *) sm;
    const float2 * xd = (const float2 *) (xw + TMAX * XS);
    float * red = (float *) (xd + NJ * TMAX); // [2][16][T]
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, g = lane >> 2, tg = lane & 3;
    for (int i = threadIdx.x; i < 2 * 16 * T; i += NT) red[i] = 0.f;
    __syncthreads();
    const int pc0 = 5 * warp, b0 = pc0 / 4;
    for (int L = 0; L < NLAY; L++) {
        for (int it = blockIdx.x; it < FF / 16; it += gridDim.x) {
            const uint8_t * w[4] = {Wg + L * ls + (size_t) (16 * it + g) * RB, Wg + L * ls + (size_t) (16 * it + g + 8) * RB,
                                    Wu + L * ls + (size_t) (16 * it + g) * RB, Wu + L * ls + (size_t) (16 * it + g + 8) * RB};
            int4 h[4][2];
            unsigned qw[4][5][2];
#pragma unroll
            for (int k = 0; k < 4; k++) {
                h[k][0] = C ? ldw4c(w[k] + b0 * 144) : ldw4(w[k] + b0 * 144);
                h[k][1] = C ? ldw4c(w[k] + (b0 + 1) * 144) : ldw4(w[k] + (b0 + 1) * 144);
#pragma unroll
                for (int i = 0; i < 5; i++) {
                    const int pc = pc0 + i;
                    const uint8_t * c = w[k] + (pc / 4) * 144 + 16 + 32 * (pc % 4) + 4 * tg;
                    qw[k][i][0] = C ? ldw1c(c) : ldw1(c);
                    qw[k][i][1] = C ? ldw1c(c + 16) : ldw1(c + 16);
                }
            }
            if (PF && it + gridDim.x < FF / 16) { // the next item's slice of these 4 rows into L2 (lane tg: one 32 B sector each)
                const size_t nx = (size_t) 16 * gridDim.x * RB;
#pragma unroll
                for (int k = 0; k < 4; k++)
#pragma unroll
                    for (int i = 0; i < 2; i++) {
                        const uint8_t * a = w[k] + nx + b0 * 144 + 32 * (2 * i + (tg >> 1)) + 16 * (tg & 1) * 0;
                        if (i * 4 + tg < 7) asm volatile("prefetch.global.L2 [%0];" ::"l"(w[k] + nx + b0 * 144 + 32 * (4 * i + tg)));
                        (void) a;
                    }
            }
            float acc[2][NTL][4] = {};
#pragma unroll
            for (int i = 0; i < 5; i++) {
                const int pc = pc0 + i, p = pc % 4;
                const bool second = pc / 4 != b0;
                float fA[4], gA[4], fB[4], gB[4];
#pragma unroll
                for (int k = 0; k < 4; k++) {
                    const int4 hh = second ? h[k][1] : h[k][0];
                    q4k_scales(hh, p, fA[k], gA[k], fB[k], gB[k]);
                }
#pragma unroll
                for (int half = 0; half < 2; half++) {
                    const int sj = 2 * pc + half, sh = 4 * half;
#pragma unroll
                    for (int nt = 0; nt < NTL; nt++) {
                        const int n = 8 * nt + g;
                        const uint2 bf = n < T ? *reinterpret_cast<const uint2 *>(xw + n * XS + 8 * sj + 2 * tg) : make_uint2(0, 0);
                        const float4 sc = *reinterpret_cast<const float4 *>(xd + sj * TMAX + 8 * nt + 2 * tg);
#pragma unroll
                        for (int m = 0; m < 2; m++) {
                            int d[4];
                            mma_s8_k32(d, (qw[2 * m][i][0] >> sh) & M4, (qw[2 * m + 1][i][0] >> sh) & M4, (qw[2 * m][i][1] >> sh) & M4,
                                       (qw[2 * m + 1][i][1] >> sh) & M4, bf);
#pragma unroll
                            for (int c = 0; c < 4; c++) {
                                const int r = 2 * m + (c >> 1);
                                const float f = half ? fB[r] : fA[r], gm = half ? gB[r] : gA[r];
                                const float dx = (c & 1) ? sc.z : sc.x, sq = (c & 1) ? sc.w : sc.y;
                                acc[m][nt][c] += f * (dx * (float) d[c]) - gm * sq;
                            }
                        }
                    }
                }
            }
#pragma unroll
            for (int m = 0; m < 2; m++)
#pragma unroll
                for (int nt = 0; nt < NTL; nt++)
#pragma unroll
                    for (int c = 0; c < 4; c++) {
                        const int t = 8 * nt + 2 * tg + (c & 1), row = g + 8 * (c >> 1);
                        if (t < T) atomicAdd(red + (m * 16 + row) * T + t, acc[m][nt][c]);
                    }
            __syncthreads();
            if (threadIdx.x < 16 * T) {
                const int row = threadIdx.x / T, t = threadIdx.x % T;
                const float a = red[row * T + t], u = red[(16 + row) * T + t];
                out[(size_t) t * FF + 16 * it + row] = a / (1.f + __expf(-a)) * u;
                red[row * T + t] = 0.f;
                red[(16 + row) * T + t] = 0.f;
            }
            __syncthreads();
        }
    }
}

// access-pattern probes: M32K's loads only (LDG.32 per lane per row chunk), and the same bytes as 16-byte loads
// (lane = one row's 16-byte qs piece: 4 lanes per 64 bytes of a row)
template <bool C>
__global__ void __launch_bounds__(NT, 1) load32k(const uint8_t * Wg, const uint8_t * Wu, float * out, size_t ls) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, g = lane >> 2, tg = lane & 3;
    const int pc0 = 5 * warp, b0 = pc0 / 4;
    unsigned x = 0;
    for (int L = 0; L < NLAY; L++)
        for (int it = blockIdx.x; it < FF / 16; it += gridDim.x) {
            const uint8_t * w[4] = {Wg + L * ls + (size_t) (16 * it + g) * RB, Wg + L * ls + (size_t) (16 * it + g + 8) * RB,
                                    Wu + L * ls + (size_t) (16 * it + g) * RB, Wu + L * ls + (size_t) (16 * it + g + 8) * RB};
#pragma unroll
            for (int k = 0; k < 4; k++) {
                const int4 h0 = C ? ldw4c(w[k] + b0 * 144) : ldw4(w[k] + b0 * 144), h1 = C ? ldw4c(w[k] + (b0 + 1) * 144) : ldw4(w[k] + (b0 + 1) * 144);
                x ^= h0.x ^ h1.y;
#pragma unroll
                for (int i = 0; i < 5; i++) {
                    const int pc = pc0 + i;
                    const uint8_t * c = w[k] + (pc / 4) * 144 + 16 + 32 * (pc % 4) + 4 * tg;
                    x ^= C ? ldw1c(c) ^ ldw1c(c + 16) : ldw1(c) ^ ldw1(c + 16);
                }
            }
            __syncthreads();
        }
    if (x == 0x12345678) out[0] = (float) x;
}
template <bool C>
__global__ void __launch_bounds__(NT, 1) load32k_64(const uint8_t * Wg, const uint8_t * Wu, float * out, size_t ls) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, g = lane >> 2, tg = lane & 3;
    const int pc0 = 5 * warp, b0 = pc0 / 4;
    unsigned x = 0;
    for (int L = 0; L < NLAY; L++)
        for (int it = blockIdx.x; it < FF / 16; it += gridDim.x) {
            const uint8_t * w[4] = {Wg + L * ls + (size_t) (16 * it + g) * RB, Wg + L * ls + (size_t) (16 * it + g + 8) * RB,
                                    Wu + L * ls + (size_t) (16 * it + g) * RB, Wu + L * ls + (size_t) (16 * it + g + 8) * RB};
#pragma unroll
            for (int k = 0; k < 4; k++) {
                const int4 h0 = C ? ldw4c(w[k] + b0 * 144) : ldw4(w[k] + b0 * 144), h1 = C ? ldw4c(w[k] + (b0 + 1) * 144) : ldw4(w[k] + (b0 + 1) * 144);
                x ^= h0.x ^ h1.y;
#pragma unroll
                for (int i = 0; i < 5; i++) {
                    const int pc = pc0 + i;
                    const uint8_t * a = w[k] + (pc / 4) * 144 + 16 + 32 * (pc % 4) + 8 * tg;
                    const uint2 v = C ? ldw2c(a) : ldw2(a);
                    x ^= v.x ^ v.y;
                }
            }
            __syncthreads();
        }
    if (x == 0x12345678) out[0] = (float) x;
}
// quad-contiguous 16-byte loads: lanes (g, tg) read 16 tg of each 64-byte span of row g (the warp's ~192-byte slice)
template <bool C>
__global__ void __launch_bounds__(NT, 1) load32k_128(const uint8_t * Wg, const uint8_t * Wu, float * out, size_t ls) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, g = lane >> 2, tg = lane & 3;
    unsigned x = 0;
    for (int L = 0; L < NLAY; L++)
        for (int it = blockIdx.x; it < FF / 16; it += gridDim.x) {
            const uint8_t * w[4] = {Wg + L * ls + (size_t) (16 * it + g) * RB, Wg + L * ls + (size_t) (16 * it + g + 8) * RB,
                                    Wu + L * ls + (size_t) (16 * it + g) * RB, Wu + L * ls + (size_t) (16 * it + g + 8) * RB};
#pragma unroll
            for (int k = 0; k < 4; k++)
#pragma unroll
                for (int i = 0; i < 3; i++) {
                    const int off = 180 * warp + 64 * i + 16 * tg; // the warp's 180-byte share of the row
                    if (off < RB) { const int4 v = C ? ldw4c(w[k] + off) : ldw4(w[k] + off); x ^= v.x ^ v.w; }
                }
            __syncthreads();
        }
    if (x == 0x12345678) out[0] = (float) x;
}
__global__ void __launch_bounds__(NT, 1) load32k_v4(const uint8_t * Wg, const uint8_t * Wu, float * out, size_t ls) {
    // warp w: rows 2w, 2w+1 of the item's 32 (gate 16 + up 16), whole row: 2880 B = 180 x 16 B, lanes stride 16 B
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    unsigned x = 0;
    for (int L = 0; L < NLAY; L++)
        for (int it = blockIdx.x; it < FF / 16; it += gridDim.x) {
#pragma unroll
            for (int rr = 0; rr < 2; rr++) {
                const int r = 2 * warp + rr;
                const uint8_t * row = (r < 16 ? Wg + L * ls + (size_t) (16 * it + r) * RB : Wu + L * ls + (size_t) (16 * it + r - 16) * RB);
#pragma unroll
                for (int i = 0; i < 6; i++) if (lane + 32 * i < RB / 16) { const int4 v = ldw4(row + 16 * (lane + 32 * i)); x ^= v.x ^ v.w; }
            }
            __syncthreads();
        }
    if (x == 0x12345678) out[0] = (float) x;
}

// M32K2: M32K with 8-byte weight loads: lane tg takes bytes 8 tg .. 8 tg + 7 of each 32-byte chunk (a quad reads the
// whole chunk: full sectors), which is the mma's k positions {4 tg .. 4 tg + 3} u {16 + 4 tg ..} under the same
// permutation on both sides: the activations stay in natural order (b fragment = words 2 tg, 2 tg + 1). L1-allocating
// loads; the next item's loads are issued right after this item's mma work, overlapping the reduction and barriers.
template <int T, bool NOLOAD = false>
__global__ void __launch_bounds__(NT, 1) ffn_mma32k2(const uint8_t * Wg, const uint8_t * Wu, float * out, size_t ls) {
    constexpr int NTL = (T + 7) / 8;
    extern __shared__ __align__(16) uint8_t sm[];
    setup32(sm, T, true);
    const unsigned * xw = (const unsigned *) sm;
    const float2 * xd = (const float2 *) (xw + TMAX * XS);
    float * red = (float *) (xd + NJ * TMAX); // [2][16][T]
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, g = lane >> 2, tg = lane & 3;
    for (int i = threadIdx.x; i < 2 * 16 * T; i += NT) red[i] = 0.f;
    __syncthreads();
    const int pc0 = 5 * warp, b0 = pc0 / 4;
    int4 h[4][2];
    uint2 qw[4][5];
    auto load = [&](int L, int it) {
#pragma unroll
        for (int k = 0; k < 4; k++) {
            if (NOLOAD) { // compute-only probe: synthesised weights
                const unsigned v = 0x9e3779b9u * (unsigned) (it * 4 + k + L) ^ lane;
                h[k][0] = make_int4(0x14001400, (int) v, (int) (v >> 3), (int) (v * 7));
                h[k][1] = h[k][0];
#pragma unroll
                for (int i = 0; i < 5; i++) qw[k][i] = make_uint2(v * (i + 1), v ^ (i * 77));
                continue;
            }
            const uint8_t * w = (k < 2 ? Wg : Wu) + L * ls + (size_t) (16 * it + g + 8 * (k & 1)) * RB;
            h[k][0] = ldw4c(w + b0 * 144);
            h[k][1] = ldw4c(w + (b0 + 1) * 144);
#pragma unroll
            for (int i = 0; i < 5; i++) {
                const int pc = pc0 + i;
                qw[k][i] = ldw2c(w + (pc / 4) * 144 + 16 + 32 * (pc % 4) + 8 * tg);
            }
        }
    };
    int L = 0, it = blockIdx.x;
    load(L, it);
    while (L < NLAY) {
        float acc[2][NTL][4] = {};
#pragma unroll
        for (int i = 0; i < 5; i++) {
            const int pc = pc0 + i, p = pc % 4;
            const bool second = pc / 4 != b0;
            float fA[4], gA[4], fB[4], gB[4];
#pragma unroll
            for (int k = 0; k < 4; k++) {
                const int4 hh = second ? h[k][1] : h[k][0];
                q4k_scales(hh, p, fA[k], gA[k], fB[k], gB[k]);
            }
#pragma unroll
            for (int half = 0; half < 2; half++) {
                const int sj = 2 * pc + half, sh = 4 * half;
#pragma unroll
                for (int nt = 0; nt < NTL; nt++) {
                    const int n = 8 * nt + g;
                    const uint2 bf = n < T ? *reinterpret_cast<const uint2 *>(xw + n * XS + 8 * sj + 2 * tg) : make_uint2(0, 0);
                    const float4 sc = *reinterpret_cast<const float4 *>(xd + sj * TMAX + 8 * nt + 2 * tg);
#pragma unroll
                    for (int m = 0; m < 2; m++) {
                        int d[4];
                        mma_s8_k32(d, (qw[2 * m][i].x >> sh) & M4, (qw[2 * m + 1][i].x >> sh) & M4, (qw[2 * m][i].y >> sh) & M4,
                                   (qw[2 * m + 1][i].y >> sh) & M4, bf);
#pragma unroll
                        for (int c = 0; c < 4; c++) {
                            const int r = 2 * m + (c >> 1);
                            const float f = half ? fB[r] : fA[r], gm = half ? gB[r] : gA[r];
                            const float dx = (c & 1) ? sc.z : sc.x, sq = (c & 1) ? sc.w : sc.y;
                            acc[m][nt][c] += f * (dx * (float) d[c]) - gm * sq;
                        }
                    }
                }
            }
        }
        const int cur_it = it, cur_L = L;
        it += gridDim.x;
        if (it >= FF / 16) { it = blockIdx.x; L++; }
        if (L < NLAY) load(L, it);
#pragma unroll
        for (int m = 0; m < 2; m++)
#pragma unroll
            for (int nt = 0; nt < NTL; nt++)
#pragma unroll
                for (int c = 0; c < 4; c++) {
                    const int t = 8 * nt + 2 * tg + (c & 1), row = g + 8 * (c >> 1);
                    if (t < T) atomicAdd(red + (m * 16 + row) * T + t, acc[m][nt][c]);
                }
        __syncthreads();
        if (threadIdx.x < 16 * T) {
            const int row = threadIdx.x / T, t = threadIdx.x % T;
            const float a = red[row * T + t], u = red[(16 + row) * T + t];
            out[(size_t) t * FF + 16 * cur_it + row] = a / (1.f + __expf(-a)) * u;
            red[row * T + t] = 0.f;
            red[(16 + row) * T + t] = 0.f;
        }
        __syncthreads();
        (void) cur_L;
    }
}
// M32K5: M32K2 with the atomic-free two-pass combine of M32K4.
// (base) M32K2: M32K with 8-byte weight loads: lane tg takes bytes 8 tg .. 8 tg + 7 of each 32-byte chunk (a quad reads the
// whole chunk: full sectors), which is the mma's k positions {4 tg .. 4 tg + 3} u {16 + 4 tg ..} under the same
// permutation on both sides: the activations stay in natural order (b fragment = words 2 tg, 2 tg + 1). L1-allocating
// loads; the next item's loads are issued right after this item's mma work, overlapping the reduction and barriers.
template <int T, bool NOLOAD = false>
__global__ void __launch_bounds__(NT, 1) ffn_mma32k5(const uint8_t * Wg, const uint8_t * Wu, float * out, size_t ls) {
    constexpr int NTL = (T + 7) / 8;
    extern __shared__ __align__(16) uint8_t sm[];
    setup32(sm, T, true);
    const unsigned * xw = (const unsigned *) sm;
    const float2 * xd = (const float2 *) (xw + TMAX * XS);
    float * red = (float *) (xd + NJ * TMAX); // [2][16][T]
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, g = lane >> 2, tg = lane & 3;
    const int pc0 = 5 * warp, b0 = pc0 / 4;
    int4 h[4][2];
    uint2 qw[4][5];
    auto load = [&](int L, int it) {
#pragma unroll
        for (int k = 0; k < 4; k++) {
            if (NOLOAD) { // compute-only probe: synthesised weights
                const unsigned v = 0x9e3779b9u * (unsigned) (it * 4 + k + L) ^ lane;
                h[k][0] = make_int4(0x14001400, (int) v, (int) (v >> 3), (int) (v * 7));
                h[k][1] = h[k][0];
#pragma unroll
                for (int i = 0; i < 5; i++) qw[k][i] = make_uint2(v * (i + 1), v ^ (i * 77));
                continue;
            }
            const uint8_t * w = (k < 2 ? Wg : Wu) + L * ls + (size_t) (16 * it + g + 8 * (k & 1)) * RB;
            h[k][0] = ldw4c(w + b0 * 144);
            h[k][1] = ldw4c(w + (b0 + 1) * 144);
#pragma unroll
            for (int i = 0; i < 5; i++) {
                const int pc = pc0 + i;
                qw[k][i] = ldw2c(w + (pc / 4) * 144 + 16 + 32 * (pc % 4) + 8 * tg);
            }
        }
    };
    int L = 0, it = blockIdx.x;
    load(L, it);
    while (L < NLAY) {
        float acc[2][NTL][4] = {};
#pragma unroll
        for (int i = 0; i < 5; i++) {
            const int pc = pc0 + i, p = pc % 4;
            const bool second = pc / 4 != b0;
            float fA[4], gA[4], fB[4], gB[4];
#pragma unroll
            for (int k = 0; k < 4; k++) {
                const int4 hh = second ? h[k][1] : h[k][0];
                q4k_scales(hh, p, fA[k], gA[k], fB[k], gB[k]);
            }
#pragma unroll
            for (int half = 0; half < 2; half++) {
                const int sj = 2 * pc + half, sh = 4 * half;
#pragma unroll
                for (int nt = 0; nt < NTL; nt++) {
                    const int n = 8 * nt + g;
                    const uint2 bf = n < T ? *reinterpret_cast<const uint2 *>(xw + n * XS + 8 * sj + 2 * tg) : make_uint2(0, 0);
                    const float4 sc = *reinterpret_cast<const float4 *>(xd + sj * TMAX + 8 * nt + 2 * tg);
#pragma unroll
                    for (int m = 0; m < 2; m++) {
                        int d[4];
                        mma_s8_k32(d, (qw[2 * m][i].x >> sh) & M4, (qw[2 * m + 1][i].x >> sh) & M4, (qw[2 * m][i].y >> sh) & M4,
                                   (qw[2 * m + 1][i].y >> sh) & M4, bf);
#pragma unroll
                        for (int c = 0; c < 4; c++) {
                            const int r = 2 * m + (c >> 1);
                            const float f = half ? fB[r] : fA[r], gm = half ? gB[r] : gA[r];
                            const float dx = (c & 1) ? sc.z : sc.x, sq = (c & 1) ? sc.w : sc.y;
                            acc[m][nt][c] += f * (dx * (float) d[c]) - gm * sq;
                        }
                    }
                }
            }
        }
        const int cur_it = it, cur_L = L;
        it += gridDim.x;
        if (it >= FF / 16) { it = blockIdx.x; L++; }
        if (L < NLAY) load(L, it);
        float * slot = red + (warp & 7) * 2 * 16 * T;
        if (warp < 8)
#pragma unroll
            for (int m = 0; m < 2; m++)
#pragma unroll
                for (int nt = 0; nt < NTL; nt++)
#pragma unroll
                    for (int c = 0; c < 4; c++) {
                        const int t = 8 * nt + 2 * tg + (c & 1), row = g + 8 * (c >> 1);
                        if (t < T) slot[(m * 16 + row) * T + t] = acc[m][nt][c];
                    }
        __syncthreads();
        if (warp >= 8)
#pragma unroll
            for (int m = 0; m < 2; m++)
#pragma unroll
                for (int nt = 0; nt < NTL; nt++)
#pragma unroll
                    for (int c = 0; c < 4; c++) {
                        const int t = 8 * nt + 2 * tg + (c & 1), row = g + 8 * (c >> 1);
                        if (t < T) slot[(m * 16 + row) * T + t] += acc[m][nt][c];
                    }
        __syncthreads();
        if (threadIdx.x < 16 * T) {
            const int row = threadIdx.x / T, t = threadIdx.x % T;
            float a = 0.f, u = 0.f;
#pragma unroll
            for (int w8 = 0; w8 < 8; w8++) {
                a += red[(w8 * 2 * 16 + row) * T + t];
                u += red[(w8 * 2 * 16 + 16 + row) * T + t];
            }
            out[(size_t) t * FF + 16 * cur_it + row] = a / (1.f + __expf(-a)) * u;
        }
        __syncthreads();
        (void) cur_L;
    }
}
// M32K3: M32K2 with progressive prefetch: as soon as chunk i is consumed, its registers load the next item's chunk i
// (the headers after the last chunk), so the next item streams in under this item's mma work.
// M32K2 (base): M32K with 8-byte weight loads: lane tg takes bytes 8 tg .. 8 tg + 7 of each 32-byte chunk (a quad reads the
// whole chunk: full sectors), which is the mma's k positions {4 tg .. 4 tg + 3} u {16 + 4 tg ..} under the same
// permutation on both sides: the activations stay in natural order (b fragment = words 2 tg, 2 tg + 1). L1-allocating
// loads; the next item's loads are issued right after this item's mma work, overlapping the reduction and barriers.
template <int T, bool NOLOAD = false>
__global__ void __launch_bounds__(NT, 1) ffn_mma32k3(const uint8_t * Wg, const uint8_t * Wu, float * out, size_t ls) {
    constexpr int NTL = (T + 7) / 8;
    extern __shared__ __align__(16) uint8_t sm[];
    setup32(sm, T, true);
    const unsigned * xw = (const unsigned *) sm;
    const float2 * xd = (const float2 *) (xw + TMAX * XS);
    float * red = (float *) (xd + NJ * TMAX); // [2][16][T]
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, g = lane >> 2, tg = lane & 3;
    for (int i = threadIdx.x; i < 2 * 16 * T; i += NT) red[i] = 0.f;
    __syncthreads();
    const int pc0 = 5 * warp, b0 = pc0 / 4;
    int4 h[4][2];
    uint2 qw[4][5];
    auto load = [&](int L, int it) {
#pragma unroll
        for (int k = 0; k < 4; k++) {
            if (NOLOAD) { // compute-only probe: synthesised weights
                const unsigned v = 0x9e3779b9u * (unsigned) (it * 4 + k + L) ^ lane;
                h[k][0] = make_int4(0x14001400, (int) v, (int) (v >> 3), (int) (v * 7));
                h[k][1] = h[k][0];
#pragma unroll
                for (int i = 0; i < 5; i++) qw[k][i] = make_uint2(v * (i + 1), v ^ (i * 77));
                continue;
            }
            const uint8_t * w = (k < 2 ? Wg : Wu) + L * ls + (size_t) (16 * it + g + 8 * (k & 1)) * RB;
            h[k][0] = ldw4c(w + b0 * 144);
            h[k][1] = ldw4c(w + (b0 + 1) * 144);
#pragma unroll
            for (int i = 0; i < 5; i++) {
                const int pc = pc0 + i;
                qw[k][i] = ldw2c(w + (pc / 4) * 144 + 16 + 32 * (pc % 4) + 8 * tg);
            }
        }
    };
    int L = 0, it = blockIdx.x;
    load(L, it);
    while (L < NLAY) {
        const int cur_it = it;
        it += gridDim.x;
        if (it >= FF / 16) { it = blockIdx.x; L++; }
        const bool more = L < NLAY;
        const uint8_t * nw[4];
#pragma unroll
        for (int k = 0; k < 4; k++) nw[k] = (k < 2 ? Wg : Wu) + (more ? L : 0) * ls + (size_t) (16 * it + g + 8 * (k & 1)) * RB;
        float acc[2][NTL][4] = {};
#pragma unroll
        for (int i = 0; i < 5; i++) {
            const int pc = pc0 + i, p = pc % 4;
            const bool second = pc / 4 != b0;
            float fA[4], gA[4], fB[4], gB[4];
#pragma unroll
            for (int k = 0; k < 4; k++) {
                const int4 hh = second ? h[k][1] : h[k][0];
                q4k_scales(hh, p, fA[k], gA[k], fB[k], gB[k]);
            }
#pragma unroll
            for (int half = 0; half < 2; half++) {
                const int sj = 2 * pc + half, sh = 4 * half;
#pragma unroll
                for (int nt = 0; nt < NTL; nt++) {
                    const int n = 8 * nt + g;
                    const uint2 bf = n < T ? *reinterpret_cast<const uint2 *>(xw + n * XS + 8 * sj + 2 * tg) : make_uint2(0, 0);
                    const float4 sc = *reinterpret_cast<const float4 *>(xd + sj * TMAX + 8 * nt + 2 * tg);
#pragma unroll
                    for (int m = 0; m < 2; m++) {
                        int d[4];
                        mma_s8_k32(d, (qw[2 * m][i].x >> sh) & M4, (qw[2 * m + 1][i].x >> sh) & M4, (qw[2 * m][i].y >> sh) & M4,
                                   (qw[2 * m + 1][i].y >> sh) & M4, bf);
#pragma unroll
                        for (int c = 0; c < 4; c++) {
                            const int r = 2 * m + (c >> 1);
                            const float f = half ? fB[r] : fA[r], gm = half ? gB[r] : gA[r];
                            const float dx = (c & 1) ? sc.z : sc.x, sq = (c & 1) ? sc.w : sc.y;
                            acc[m][nt][c] += f * (dx * (float) d[c]) - gm * sq;
                        }
                    }
                }
            }
            if (more && !NOLOAD)
#pragma unroll
                for (int k = 0; k < 4; k++) qw[k][i] = ldw2c(nw[k] + (pc / 4) * 144 + 16 + 32 * (pc % 4) + 8 * tg);
        }
        if (more && !NOLOAD)
#pragma unroll
            for (int k = 0; k < 4; k++) {
                h[k][0] = ldw4c(nw[k] + b0 * 144);
                h[k][1] = ldw4c(nw[k] + (b0 + 1) * 144);
            }
        if (more && NOLOAD) load(L, it);
        const int cur_L = L;
#pragma unroll
        for (int m = 0; m < 2; m++)
#pragma unroll
            for (int nt = 0; nt < NTL; nt++)
#pragma unroll
                for (int c = 0; c < 4; c++) {
                    const int t = 8 * nt + 2 * tg + (c & 1), row = g + 8 * (c >> 1);
                    if (t < T) atomicAdd(red + (m * 16 + row) * T + t, acc[m][nt][c]);
                }
        __syncthreads();
        if (threadIdx.x < 16 * T) {
            const int row = threadIdx.x / T, t = threadIdx.x % T;
            const float a = red[row * T + t], u = red[(16 + row) * T + t];
            out[(size_t) t * FF + 16 * cur_it + row] = a / (1.f + __expf(-a)) * u;
            red[row * T + t] = 0.f;
            red[(16 + row) * T + t] = 0.f;
        }
        __syncthreads();
        (void) cur_L;
    }
}

// M32K4: split-K over chunk pairs: a quad loads a row's 64-byte window (chunks 2q, 2q + 1 of a block: 16 bytes per
// lane, full sectors), then one shfl_xor(2) of the second halves gives every lane 8 bytes of each chunk:
//   chunk 2q:     lanes 0..3 hold bytes {0, 16, 8, 24} + 0..7      chunk 2q + 1: bytes {8, 24, 0, 16} + 0..7
// (the k order is free as long as the b fragment reads the same activation bytes). Warp w: pairs [P0, P0 + NP)
// (3 pairs for w < 8, else 2: 40 pairs per row). Partial sums: per-warp slots (warps w and w + 8 share one, in two
// passes), no atomics.
template <int T>
__global__ void __launch_bounds__(NT, 1) ffn_mma32k4(const uint8_t * Wg, const uint8_t * Wu, float * out, size_t ls) {
    constexpr int NTL = (T + 7) / 8;
    extern __shared__ __align__(16) uint8_t sm[];
    setup32(sm, T, true);
    const unsigned * xw = (const unsigned *) sm;
    const float2 * xd = (const float2 *) (xw + TMAX * XS);
    float * red = (float *) (xd + NJ * TMAX); // [8][2][16][T]
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, g = lane >> 2, tg = lane & 3;
    const int NP = warp < 8 ? 3 : 2, P0 = warp < 8 ? 3 * warp : 24 + 2 * (warp - 8), bb0 = P0 / 2;
    const int offA = (tg & 1) * 16 + (tg >> 1) * 8, offB = offA ^ 8; // byte offset of this lane's 8 bytes in chunk 2q / 2q + 1
    for (int L = 0; L < NLAY; L++) {
        for (int it = blockIdx.x; it < FF / 16; it += gridDim.x) {
            int4 h[4][2];
            uint4 win[4][3];
#pragma unroll
            for (int k = 0; k < 4; k++) {
                const uint8_t * w = (k < 2 ? Wg : Wu) + L * ls + (size_t) (16 * it + g + 8 * (k & 1)) * RB;
                h[k][0] = ldw4c(w + bb0 * 144);
                h[k][1] = ldw4c(w + (bb0 + 1) * 144);
#pragma unroll
                for (int i = 0; i < 3; i++)
                    if (i < NP) {
                        const int P = P0 + i;
                        const int4 v = ldw4c(w + (P / 2) * 144 + 16 + 64 * (P % 2) + 16 * tg);
                        win[k][i] = make_uint4(v.x, v.y, v.z, v.w);
                    }
            }
            float acc[2][NTL][4] = {};
#pragma unroll
            for (int i = 0; i < 3; i++) {
                if (i >= NP) break;
                const int P = P0 + i, bk = P / 2;
                const bool second = bk != bb0;
                // own first half stays, partner's (tg ^ 2) second half arrives: lanes 0, 1 -> (chunk 2q, chunk 2q + 1),
                // lanes 2, 3 -> (chunk 2q + 1 [own], chunk 2q [from partner])
                uint2 cA[4], cB[4];
#pragma unroll
                for (int k = 0; k < 4; k++) {
                    const unsigned ox = __shfl_xor_sync(0xffffffffu, win[k][i].z, 2), oy = __shfl_xor_sync(0xffffffffu, win[k][i].w, 2);
                    const uint2 own = make_uint2(win[k][i].x, win[k][i].y), got = make_uint2(ox, oy);
                    cA[k] = tg < 2 ? own : got;
                    cB[k] = tg < 2 ? got : own;
                }
#pragma unroll
                for (int half_chunk = 0; half_chunk < 2; half_chunk++) { // chunk 2q + half_chunk of block bk
                    const int p = 2 * (P % 2) + half_chunk;
                    float fA[4], gA[4], fB[4], gB[4];
#pragma unroll
                    for (int k = 0; k < 4; k++) q4k_scales(second ? h[k][1] : h[k][0], p, fA[k], gA[k], fB[k], gB[k]);
                    const int off = half_chunk ? offB : offA;
#pragma unroll
                    for (int nib = 0; nib < 2; nib++) { // sub-block 2p (low nibbles), 2p + 1 (high)
                        const int sj = 8 * bk + 2 * p + nib, sh = 4 * nib;
#pragma unroll
                        for (int nt = 0; nt < NTL; nt++) {
                            const int n = 8 * nt + g;
                            const uint2 bf = n < T ? *reinterpret_cast<const uint2 *>(xw + n * XS + 8 * sj + off / 4) : make_uint2(0, 0);
                            const float4 sc = *reinterpret_cast<const float4 *>(xd + sj * TMAX + 8 * nt + 2 * tg);
#pragma unroll
                            for (int m = 0; m < 2; m++) {
                                const uint2 r0 = half_chunk ? cB[2 * m] : cA[2 * m], r1 = half_chunk ? cB[2 * m + 1] : cA[2 * m + 1];
                                int d[4];
                                mma_s8_k32(d, (r0.x >> sh) & M4, (r1.x >> sh) & M4, (r0.y >> sh) & M4, (r1.y >> sh) & M4, bf);
#pragma unroll
                                for (int c = 0; c < 4; c++) {
                                    const int r = 2 * m + (c >> 1);
                                    const float f = nib ? fB[r] : fA[r], gm = nib ? gB[r] : gA[r];
                                    const float dx = (c & 1) ? sc.z : sc.x, sq = (c & 1) ? sc.w : sc.y;
                                    acc[m][nt][c] += f * (dx * (float) d[c]) - gm * sq;
                                }
                            }
                        }
                    }
                }
            }
            // two-pass combine: warps 0..7 store, warps 8..15 add, then 16 T threads sum the 8 slots
            float * slot = red + (warp & 7) * 2 * 16 * T;
            if (warp < 8)
#pragma unroll
                for (int m = 0; m < 2; m++)
#pragma unroll
                    for (int nt = 0; nt < NTL; nt++)
#pragma unroll
                        for (int c = 0; c < 4; c++) {
                            const int t = 8 * nt + 2 * tg + (c & 1), row = g + 8 * (c >> 1);
                            if (t < T) slot[(m * 16 + row) * T + t] = acc[m][nt][c];
                        }
            __syncthreads();
            if (warp >= 8)
#pragma unroll
                for (int m = 0; m < 2; m++)
#pragma unroll
                    for (int nt = 0; nt < NTL; nt++)
#pragma unroll
                        for (int c = 0; c < 4; c++) {
                            const int t = 8 * nt + 2 * tg + (c & 1), row = g + 8 * (c >> 1);
                            if (t < T) slot[(m * 16 + row) * T + t] += acc[m][nt][c];
                        }
            __syncthreads();
            if (threadIdx.x < 16 * T) {
                const int row = threadIdx.x / T, t = threadIdx.x % T;
                float a = 0.f, u = 0.f;
#pragma unroll
                for (int w8 = 0; w8 < 8; w8++) {
                    a += red[(w8 * 2 * 16 + row) * T + t];
                    u += red[(w8 * 2 * 16 + 16 + row) * T + t];
                }
                out[(size_t) t * FF + 16 * it + row] = a / (1.f + __expf(-a)) * u;
            }
            __syncthreads();
        }
    }
}

// M32K6: 8-row items: the mma's rows g / g + 8 are gate row g / up row g of the item, so a lane holds both of its
// rows' results (silu(g) * u needs no exchange) and an item's weight words per lane halve (36 registers), which
// leaves room to double-buffer: the next item's loads are issued before this item's mma work. Split-K over the 16
// warps as M32K2 (p-chunks [5w, 5w + 5), 8-byte L1-allocating loads); atomic-free two-pass combine.
template <int T>
__global__ void __launch_bounds__(NT, 1) ffn_mma32k6(const uint8_t * Wg, const uint8_t * Wu, float * out, size_t ls) {
    constexpr int NTL = (T + 7) / 8, NI = FF / 8;
    extern __shared__ __align__(16) uint8_t sm[];
    setup32(sm, T, true);
    const unsigned * xw = (const unsigned *) sm;
    const float2 * xd = (const float2 *) (xw + TMAX * XS);
    float * red = (float *) (xd + NJ * TMAX); // [8 slots][16 (gate 8, up 8)][T]
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, g = lane >> 2, tg = lane & 3;
    const int pc0 = 5 * warp, b0 = pc0 / 4;
#define K6_LOAD(H, Q, L_, IT_)                                                                        \
    _Pragma("unroll") for (int k = 0; k < 2; k++) {                                                   \
        const uint8_t * w = (k ? Wu : Wg) + (L_) * ls + (size_t) (8 * (IT_) + g) * RB;                \
        H[k][0] = ldw4c(w + b0 * 144);                                                                \
        H[k][1] = ldw4c(w + (b0 + 1) * 144);                                                          \
        _Pragma("unroll") for (int i = 0; i < 5; i++) {                                               \
            const int pc = pc0 + i;                                                                   \
            Q[k][i] = ldw2c(w + (pc / 4) * 144 + 16 + 32 * (pc % 4) + 8 * tg);                        \
        }                                                                                             \
    }
#define K6_ITEM(BH, BQ, ITEM_IT) \
        float acc[NTL][4] = {}; \
_Pragma("unroll") \
        for (int i = 0; i < 5; i++) { \
            const int pc = pc0 + i, p = pc % 4; \
            const bool second = pc / 4 != b0; \
            float fA[2], gA[2], fB[2], gB[2]; \
_Pragma("unroll") \
            for (int k = 0; k < 2; k++) q4k_scales(sel4(second, BH[k][1], BH[k][0]), p, fA[k], gA[k], fB[k], gB[k]); \
_Pragma("unroll") \
            for (int half = 0; half < 2; half++) { \
                const int sj = 2 * pc + half, sh = 4 * half; \
_Pragma("unroll") \
                for (int nt = 0; nt < NTL; nt++) { \
                    const int n = 8 * nt + g; \
                    const uint2 bf = n < T ? *reinterpret_cast<const uint2 *>(xw + n * XS + 8 * sj + 2 * tg) : make_uint2(0, 0); \
                    const float4 sc = *reinterpret_cast<const float4 *>(xd + sj * TMAX + 8 * nt + 2 * tg); \
                    int d[4]; \
                    mma_s8_k32(d, (BQ[0][i].x >> sh) & M4, (BQ[1][i].x >> sh) & M4, (BQ[0][i].y >> sh) & M4, (BQ[1][i].y >> sh) & M4, bf); \
_Pragma("unroll") \
                    for (int c = 0; c < 4; c++) { \
                        const int r = c >> 1; \
                        const float f = half ? fB[r] : fA[r], gm = half ? gB[r] : gA[r]; \
                        const float dx = (c & 1) ? sc.z : sc.x, sq = (c & 1) ? sc.w : sc.y; \
                        acc[nt][c] += f * (dx * (float) d[c]) - gm * sq; \
                    } \
                } \
            } \
        } \
        float * slot = red + (warp & 7) * 16 * T; \
        if (warp < 8) \
_Pragma("unroll") \
            for (int nt = 0; nt < NTL; nt++) \
_Pragma("unroll") \
                for (int c = 0; c < 4; c++) { \
                    const int t = 8 * nt + 2 * tg + (c & 1); \
                    if (t < T) slot[(g + 8 * (c >> 1)) * T + t] = acc[nt][c]; \
                } \
        __syncthreads(); \
        if (warp >= 8) \
_Pragma("unroll") \
            for (int nt = 0; nt < NTL; nt++) \
_Pragma("unroll") \
                for (int c = 0; c < 4; c++) { \
                    const int t = 8 * nt + 2 * tg + (c & 1); \
                    if (t < T) slot[(g + 8 * (c >> 1)) * T + t] += acc[nt][c]; \
                } \
        __syncthreads(); \
        if (threadIdx.x < 8 * T) { \
            const int row = threadIdx.x / T, t = threadIdx.x % T; \
            float a = 0.f, u = 0.f; \
_Pragma("unroll") \
            for (int w8 = 0; w8 < 8; w8++) { \
                a += red[(w8 * 16 + row) * T + t]; \
                u += red[(w8 * 16 + 8 + row) * T + t]; \
            } \
            out[(size_t) t * FF + 8 * (ITEM_IT) + row] = a / (1.f + __expf(-a)) * u; \
        } \
        __syncthreads(); \
    {}
    int4 hA[2][2], hB[2][2];
    uint2 qA[2][5], qB[2][5];
    int L = 0, it = blockIdx.x;
    K6_LOAD(hA, qA, L, it)
    while (true) {
        int L1 = L, it1 = it + gridDim.x;
        if (it1 >= NI) { it1 = blockIdx.x; L1++; }
        if (L1 < NLAY) { K6_LOAD(hB, qB, L1, it1) }
        { K6_ITEM(hA, qA, it) }
        if (L1 >= NLAY) break;
        int L2 = L1, it2 = it1 + gridDim.x;
        if (it2 >= NI) { it2 = blockIdx.x; L2++; }
        if (L2 < NLAY) { K6_LOAD(hA, qA, L2, it2) }
        { K6_ITEM(hB, qB, it1) }
        if (L2 >= NLAY) break;
        L = L2; it = it2;
    }
#undef K6_LOAD
#undef K6_ITEM
}

// M32K7: full-bandwidth loads + double buffering. Two items per block at a time (warp groups 0..7 / 8..15, named
// barriers); in a group, warp w takes chunk pairs [5w, 5w + 5) of the item's 8 gate + 8 up rows (lane: gate row g,
// up row g = mma rows g, g + 8). A quad loads each row's 64-byte window (16 bytes per lane: full sectors), one
// shfl_xor(2) of the second halves gives each lane 8 bytes of both chunks (offsets as M32K4). Headers: lane tg
// keeps word tg of each (row, block) header (6 registers), the quad's 4 words are shuffled together when a pair
// needs them. 46 registers per item buffer, two buffers.
#define K7_BAR(id) asm volatile("bar.sync %0, 256;" ::"r"(id) : "memory")
template <int T, bool DB = true>
__global__ void __launch_bounds__(NT, 1) ffn_mma32k7(const uint8_t * Wg, const uint8_t * Wu, float * out, size_t ls) {
    constexpr int NTL = (T + 7) / 8, NI = FF / 8;
    extern __shared__ __align__(16) uint8_t sm[];
    setup32(sm, T, true);
    const unsigned * xw = (const unsigned *) sm;
    const float2 * xd = (const float2 *) (xw + TMAX * XS);
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, g = lane >> 2, tg = lane & 3;
    const int grp = warp >> 3, gw = warp & 7, qb = lane & ~3;
    float * red = (float *) (xd + NJ * TMAX) + grp * 8 * 16 * T; // this group's [8 warps][16][T]
    const int P0 = 5 * gw, hb0 = P0 / 2;
    const int offA = (tg & 1) * 16 + (tg >> 1) * 8, offB = offA ^ 8;
#define K7_LOAD(W, HW, L_, IT_)                                                                        \
    _Pragma("unroll") for (int k = 0; k < 2; k++) {                                                    \
        const uint8_t * w = (k ? Wu : Wg) + (L_) * ls + (size_t) (8 * (IT_) + g) * RB;                 \
        _Pragma("unroll") for (int b = 0; b < 3; b++) HW[k][b] = (hb0 + b) < K / 256 ? ldw1c(w + (hb0 + b) * 144 + 4 * tg) : 0u; \
        _Pragma("unroll") for (int i = 0; i < 5; i++) {                                                \
            const int P = P0 + i;                                                                      \
            const int4 v = ldw4c(w + (P / 2) * 144 + 16 + 64 * (P % 2) + 16 * tg);                     \
            W[k][i] = make_uint4(v.x, v.y, v.z, v.w);                                                  \
        }                                                                                              \
    }
#define K7_ITEM(W, HW, IT_)                                                                            \
    {                                                                                                  \
        float acc[NTL][4] = {};                                                                        \
        _Pragma("unroll") for (int i = 0; i < 5; i++) {                                                \
            const int P = P0 + i, bk = P / 2, bi = bk - hb0;                                           \
            int4 hh[2];                                                                                \
            uint2 cA[2], cB[2];                                                                        \
            _Pragma("unroll") for (int k = 0; k < 2; k++) {                                            \
                const unsigned hw = bi == 0 ? HW[k][0] : bi == 1 ? HW[k][1] : HW[k][2];                \
                hh[k] = make_int4(__shfl_sync(~0u, hw, qb), __shfl_sync(~0u, hw, qb + 1), __shfl_sync(~0u, hw, qb + 2), __shfl_sync(~0u, hw, qb + 3)); \
                const unsigned ox = __shfl_xor_sync(~0u, W[k][i].z, 2), oy = __shfl_xor_sync(~0u, W[k][i].w, 2); \
                const uint2 own = make_uint2(W[k][i].x, W[k][i].y), got = make_uint2(ox, oy);         \
                cA[k] = tg < 2 ? own : got;                                                            \
                cB[k] = tg < 2 ? got : own;                                                            \
            }                                                                                          \
            _Pragma("unroll") for (int hc = 0; hc < 2; hc++) {                                         \
                const int p = 2 * (P % 2) + hc, off = hc ? offB : offA;                                \
                float fA[2], gA[2], fB[2], gB[2];                                                      \
                _Pragma("unroll") for (int k = 0; k < 2; k++) q4k_scales(hh[k], p, fA[k], gA[k], fB[k], gB[k]); \
                const uint2 r0 = hc ? cB[0] : cA[0], r1 = hc ? cB[1] : cA[1];                          \
                _Pragma("unroll") for (int nib = 0; nib < 2; nib++) {                                  \
                    const int sj = 8 * bk + 2 * p + nib, sh = 4 * nib;                                 \
                    _Pragma("unroll") for (int nt = 0; nt < NTL; nt++) {                               \
                        const int n = 8 * nt + g;                                                      \
                        const uint2 bf = n < T ? *reinterpret_cast<const uint2 *>(xw + n * XS + 8 * sj + off / 4) : make_uint2(0, 0); \
                        const float4 sc = *reinterpret_cast<const float4 *>(xd + sj * TMAX + 8 * nt + 2 * tg); \
                        int d[4];                                                                      \
                        mma_s8_k32(d, (r0.x >> sh) & M4, (r1.x >> sh) & M4, (r0.y >> sh) & M4, (r1.y >> sh) & M4, bf); \
                        _Pragma("unroll") for (int c = 0; c < 4; c++) {                                \
                            const int r = c >> 1;                                                      \
                            const float f = nib ? fB[r] : fA[r], gm = nib ? gB[r] : gA[r];             \
                            const float dx = (c & 1) ? sc.z : sc.x, sq = (c & 1) ? sc.w : sc.y;        \
                            acc[nt][c] += f * (dx * (float) d[c]) - gm * sq;                           \
                        }                                                                              \
                    }                                                                                  \
                }                                                                                      \
            }                                                                                          \
        }                                                                                              \
        float * slot = red + gw * 16 * T;                                                              \
        _Pragma("unroll") for (int nt = 0; nt < NTL; nt++)                                             \
            _Pragma("unroll") for (int c = 0; c < 4; c++) {                                            \
                const int t = 8 * nt + 2 * tg + (c & 1);                                               \
                if (t < T) slot[(g + 8 * (c >> 1)) * T + t] = acc[nt][c];                              \
            }                                                                                          \
        K7_BAR(1 + grp);                                                                               \
        const int tid = threadIdx.x & 255;                                                             \
        if (tid < 8 * T) {                                                                             \
            const int row = tid / T, t = tid % T;                                                      \
            float a = 0.f, u = 0.f;                                                                    \
            _Pragma("unroll") for (int w8 = 0; w8 < 8; w8++) {                                         \
                a += red[(w8 * 16 + row) * T + t];                                                     \
                u += red[(w8 * 16 + 8 + row) * T + t];                                                 \
            }                                                                                          \
            out[(size_t) t * FF + 8 * (IT_) + row] = a / (1.f + __expf(-a)) * u;                      \
        }                                                                                              \
        K7_BAR(1 + grp);                                                                               \
    }
    const int it0 = 2 * blockIdx.x + grp, step = 2 * gridDim.x;
    if (!DB) {
        uint4 wA[2][5];
        unsigned hA[2][3];
        for (int L = 0; L < NLAY; L++)
            for (int it = it0; it < NI; it += step) {
                K7_LOAD(wA, hA, L, it)
                K7_ITEM(wA, hA, it)
            }
        return;
    }
    uint4 wA[2][5], wB[2][5];
    unsigned hA[2][3], hB[2][3];
    int L = 0, it = it0;
    if (it < NI) { K7_LOAD(wA, hA, L, it) }
    while (it < NI) {
        int L1 = L, it1 = it + step;
        if (it1 >= NI) { it1 = it0; L1++; }
        const bool m1 = L1 < NLAY && it1 < NI;
        if (m1) { K7_LOAD(wB, hB, L1, it1) }
        K7_ITEM(wA, hA, it)
        if (!m1) break;
        int L2 = L1, it2 = it1 + step;
        if (it2 >= NI) { it2 = it0; L2++; }
        const bool m2 = L2 < NLAY && it2 < NI;
        if (m2) { K7_LOAD(wA, hA, L2, it2) }
        K7_ITEM(wB, hB, it1)
        if (!m2) break;
        L = L2; it = it2;
    }
#undef K7_LOAD
#undef K7_ITEM
}

// M32K8: M32K7 single-buffered, full headers in registers (no header shuffles / select chains), and an explicit
// 4-op epilogue per output: fx = f * dx once per (row, token), then acc = fma(fx, d, acc), accm = fma(gm, sq, accm).
template <int T, bool NOLOAD = false, bool XF = false, int PFL2 = 0, bool PROG = false>
__global__ void __launch_bounds__(NT, 1) ffn_mma32k8(const uint8_t * Wg, const uint8_t * Wu, float * out, size_t ls) {
    constexpr int NTL = (T + 7) / 8, NI = FF / 8;
    extern __shared__ __align__(16) uint8_t sm[];
    if (XF) setup_q32(sm, T); else setup32(sm, T, true);
    const unsigned * xw = (const unsigned *) sm;
    const float2 * xd = (const float2 *) (xw + TMAX * XS);
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, g = lane >> 2, tg = lane & 3;
    const int grp = warp >> 3, gw = warp & 7;
    float * red = (float *) (xd + NJ * TMAX) + grp * 8 * 16 * T;
    const int P0 = 5 * gw, hb0 = P0 / 2;
    const int offA = (tg & 1) * 16 + (tg >> 1) * 8, offB = offA ^ 8;
    const int it0 = 2 * blockIdx.x + grp, step = 2 * gridDim.x;
    uint4 W[2][5];
    int4 H[2][3];
    for (int L = 0; L < NLAY; L++)
        for (int it = it0; it < NI; it += step) {
            int nL = L, nit = it + step;
            if (nit >= NI) { nit = it0; nL++; }
            const bool pnext = PROG && nL < NLAY;
#pragma unroll
            for (int k = 0; k < 2; k++) {
                if (PROG && !(L == 0 && it == it0)) break; // (progressive: loaded during the previous item)
                if (NOLOAD) {
                    const unsigned v = 0x9e3779b9u * (unsigned) (it * 2 + k + L) ^ lane;
#pragma unroll
                    for (int b = 0; b < 3; b++) H[k][b] = make_int4(0x14001400, (int) (v + b), (int) (v >> 3), (int) (v * 7));
#pragma unroll
                    for (int i = 0; i < 5; i++) W[k][i] = make_uint4(v * (i + 1), v ^ (i * 77), v + i, v - i);
                    continue;
                }
                const uint8_t * w = (k ? Wu : Wg) + L * ls + (size_t) (8 * it + g) * RB;
#pragma unroll
                for (int b = 0; b < 3; b++) H[k][b] = (hb0 + b) < K / 256 ? ldw4c(w + (hb0 + b) * 144) : make_int4(0, 0, 0, 0);
#pragma unroll
                for (int i = 0; i < 5; i++) {
                    const int P = P0 + i;
                    const int4 v = ldw4c(w + (P / 2) * 144 + 16 + 64 * (P % 2) + 16 * tg);
                    W[k][i] = make_uint4(v.x, v.y, v.z, v.w);
                }
            }
            if (PFL2) { // the next item's windows + headers of these rows into L2 (one lane per 64-byte window)
                int nL = L, nit = it + step;
                if (nit >= NI) { nit = it0; nL++; }
                if (nL < NLAY && tg == 0) {
#pragma unroll
                    for (int k = 0; k < 2; k++) {
                        const uint8_t * w = (k ? Wu : Wg) + nL * ls + (size_t) (8 * nit + g) * RB;
#pragma unroll
                        for (int b = 0; b < 3; b++) if ((hb0 + b) < K / 256) asm volatile("prefetch.global.L2 [%0];" ::"l"(w + (hb0 + b) * 144));
#pragma unroll
                        for (int i = 0; i < 5; i++) {
                            const int P = P0 + i;
                            asm volatile("prefetch.global.L2 [%0];" ::"l"(w + (P / 2) * 144 + 16 + 64 * (P % 2)));
                            asm volatile("prefetch.global.L2 [%0];" ::"l"(w + (P / 2) * 144 + 16 + 64 * (P % 2) + 63));
                        }
                    }
                }
            }
            float acc[NTL][4] = {}, accm[NTL][4] = {};
#pragma unroll
            for (int i = 0; i < 5; i++) {
                const int P = P0 + i, bk = P / 2, bi = bk - hb0; // bi in {0, 1, 2}: P0 + i spans 3 blocks at most
                uint2 cA[2], cB[2];
#pragma unroll
                for (int k = 0; k < 2; k++) {
                    const unsigned ox = __shfl_xor_sync(~0u, W[k][i].z, 2), oy = __shfl_xor_sync(~0u, W[k][i].w, 2);
                    const uint2 own = make_uint2(W[k][i].x, W[k][i].y), got = make_uint2(ox, oy);
                    cA[k] = tg < 2 ? own : got;
                    cB[k] = tg < 2 ? got : own;
                }
#pragma unroll
                for (int hc = 0; hc < 2; hc++) {
                    const int p = 2 * (P % 2) + hc, off = hc ? offB : offA;
                    float fA[2], gA[2], fB[2], gB[2];
#pragma unroll
                    for (int k = 0; k < 2; k++) {
                        const int4 h = bi == 0 ? H[k][0] : (bi == 1 ? H[k][1] : H[k][2]);
                        q4k_scales(h, p, fA[k], gA[k], fB[k], gB[k]);
                    }
                    const uint2 r0 = hc ? cB[0] : cA[0], r1 = hc ? cB[1] : cA[1];
#pragma unroll
                    for (int nib = 0; nib < 2; nib++) {
                        const int sj = 8 * bk + 2 * p + nib, sh = 4 * nib;
                        const float f0 = nib ? fB[0] : fA[0], f1 = nib ? fB[1] : fA[1];
                        const float g0 = nib ? gB[0] : gA[0], g1 = nib ? gB[1] : gA[1];
                        const unsigned a0 = (r0.x >> sh) & M4, a1 = (r1.x >> sh) & M4, a2 = (r0.y >> sh) & M4, a3 = (r1.y >> sh) & M4;
#pragma unroll
                        for (int nt = 0; nt < NTL; nt++) {
                            const int n = 8 * nt + g;
                            const uint2 bf = n < T ? *reinterpret_cast<const uint2 *>(xw + n * XS + 8 * sj + off / 4) : make_uint2(0, 0);
                            const float4 sc = *reinterpret_cast<const float4 *>(xd + sj * TMAX + 8 * nt + 2 * tg);
                            int d[4];
                            mma_s8_k32(d, a0, a1, a2, a3, bf);
                            acc[nt][0] = fmaf(f0 * sc.x, (float) d[0], acc[nt][0]);
                            acc[nt][1] = fmaf(f0 * sc.z, (float) d[1], acc[nt][1]);
                            acc[nt][2] = fmaf(f1 * sc.x, (float) d[2], acc[nt][2]);
                            acc[nt][3] = fmaf(f1 * sc.z, (float) d[3], acc[nt][3]);
                            accm[nt][0] = fmaf(g0, sc.y, accm[nt][0]);
                            accm[nt][1] = fmaf(g0, sc.w, accm[nt][1]);
                            accm[nt][2] = fmaf(g1, sc.y, accm[nt][2]);
                            accm[nt][3] = fmaf(g1, sc.w, accm[nt][3]);
                        }
                    }
                }
                if (pnext) // pair i is consumed: its registers load the next item's pair i
#pragma unroll
                    for (int k = 0; k < 2; k++) {
                        const int4 v = ldw4c((k ? Wu : Wg) + nL * ls + (size_t) (8 * nit + g) * RB + (P / 2) * 144 + 16 + 64 * (P % 2) + 16 * tg);
                        W[k][i] = make_uint4(v.x, v.y, v.z, v.w);
                    }
            }
            if (pnext)
#pragma unroll
                for (int k = 0; k < 2; k++)
#pragma unroll
                    for (int b = 0; b < 3; b++)
                        H[k][b] = (hb0 + b) < K / 256 ? ldw4c((k ? Wu : Wg) + nL * ls + (size_t) (8 * nit + g) * RB + (hb0 + b) * 144) : make_int4(0, 0, 0, 0);
            float * slot = red + gw * 16 * T;
#pragma unroll
            for (int nt = 0; nt < NTL; nt++)
#pragma unroll
                for (int c = 0; c < 4; c++) {
                    const int t = 8 * nt + 2 * tg + (c & 1);
                    if (t < T) slot[(g + 8 * (c >> 1)) * T + t] = acc[nt][c] - accm[nt][c];
                }
            K7_BAR(1 + grp);
            const int tid = threadIdx.x & 255;
            if (tid < 8 * T) {
                const int row = tid / T, t = tid % T;
                float a = 0.f, u = 0.f;
#pragma unroll
                for (int w8 = 0; w8 < 8; w8++) {
                    a += red[(w8 * 16 + row) * T + t];
                    u += red[(w8 * 16 + 8 + row) * T + t];
                }
                out[(size_t) t * FF + 8 * it + row] = a / (1.f + __expf(-a)) * u;
            }
            K7_BAR(1 + grp);
        }
}
// M32K9: M32K8 with GS-warp groups (NT / 32 / GS items per block at a time, out of phase): warp w of a group takes
// HALVES x 5 chunk pairs, loaded and computed 5 pairs at a time. (base) M32K8: M32K7 single-buffered, full headers in registers (no header shuffles / select chains), and an explicit
// 4-op epilogue per output: fx = f * dx once per (row, token), then acc = fma(fx, d, acc), accm = fma(gm, sq, accm).
template <int T, int GS, bool NOLOAD = false>
__global__ void __launch_bounds__(NT, 1) ffn_mma32k9(const uint8_t * Wg, const uint8_t * Wu, float * out, size_t ls) {
    constexpr int NTL = (T + 7) / 8, NI = FF / 8;
    extern __shared__ __align__(16) uint8_t sm[];
    setup32(sm, T, true);
    const unsigned * xw = (const unsigned *) sm;
    const float2 * xd = (const float2 *) (xw + TMAX * XS);
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, g = lane >> 2, tg = lane & 3;
    constexpr int NG = 16 / GS, HALVES = 8 / GS;
    const int grp = warp / GS, gw = warp % GS;
    float * red = (float *) (xd + NJ * TMAX) + grp * GS * 16 * T;
    const int offA = (tg & 1) * 16 + (tg >> 1) * 8, offB = offA ^ 8;
    const int it0 = NG * blockIdx.x + grp, step = NG * gridDim.x;
    for (int L = 0; L < NLAY; L++)
        for (int it = it0; it < NI; it += step) {
          float acc[NTL][4] = {}, accm[NTL][4] = {};
#pragma unroll 1
          for (int hv = 0; hv < HALVES; hv++) {
            const int P0 = 5 * (gw * HALVES + hv), hb0 = P0 / 2;
            uint4 W[2][5];
            int4 H[2][3];
#pragma unroll
            for (int k = 0; k < 2; k++) {
                if (NOLOAD) {
                    const unsigned v = 0x9e3779b9u * (unsigned) (it * 2 + k + L) ^ lane;
#pragma unroll
                    for (int b = 0; b < 3; b++) H[k][b] = make_int4(0x14001400, (int) (v + b), (int) (v >> 3), (int) (v * 7));
#pragma unroll
                    for (int i = 0; i < 5; i++) W[k][i] = make_uint4(v * (i + 1), v ^ (i * 77), v + i, v - i);
                    continue;
                }
                const uint8_t * w = (k ? Wu : Wg) + L * ls + (size_t) (8 * it + g) * RB;
#pragma unroll
                for (int b = 0; b < 3; b++) H[k][b] = (hb0 + b) < K / 256 ? ldw4c(w + (hb0 + b) * 144) : make_int4(0, 0, 0, 0);
#pragma unroll
                for (int i = 0; i < 5; i++) {
                    const int P = P0 + i;
                    const int4 v = ldw4c(w + (P / 2) * 144 + 16 + 64 * (P % 2) + 16 * tg);
                    W[k][i] = make_uint4(v.x, v.y, v.z, v.w);
                }
            }
#pragma unroll
            for (int i = 0; i < 5; i++) {
                const int P = P0 + i, bk = P / 2, bi = bk - hb0; // bi in {0, 1, 2}: P0 + i spans 3 blocks at most
                uint2 cA[2], cB[2];
#pragma unroll
                for (int k = 0; k < 2; k++) {
                    const unsigned ox = __shfl_xor_sync(~0u, W[k][i].z, 2), oy = __shfl_xor_sync(~0u, W[k][i].w, 2);
                    const uint2 own = make_uint2(W[k][i].x, W[k][i].y), got = make_uint2(ox, oy);
                    cA[k] = tg < 2 ? own : got;
                    cB[k] = tg < 2 ? got : own;
                }
#pragma unroll
                for (int hc = 0; hc < 2; hc++) {
                    const int p = 2 * (P % 2) + hc, off = hc ? offB : offA;
                    float fA[2], gA[2], fB[2], gB[2];
#pragma unroll
                    for (int k = 0; k < 2; k++) {
                        const int4 h = bi == 0 ? H[k][0] : (bi == 1 ? H[k][1] : H[k][2]);
                        q4k_scales(h, p, fA[k], gA[k], fB[k], gB[k]);
                    }
                    const uint2 r0 = hc ? cB[0] : cA[0], r1 = hc ? cB[1] : cA[1];
#pragma unroll
                    for (int nib = 0; nib < 2; nib++) {
                        const int sj = 8 * bk + 2 * p + nib, sh = 4 * nib;
                        const float f0 = nib ? fB[0] : fA[0], f1 = nib ? fB[1] : fA[1];
                        const float g0 = nib ? gB[0] : gA[0], g1 = nib ? gB[1] : gA[1];
                        const unsigned a0 = (r0.x >> sh) & M4, a1 = (r1.x >> sh) & M4, a2 = (r0.y >> sh) & M4, a3 = (r1.y >> sh) & M4;
#pragma unroll
                        for (int nt = 0; nt < NTL; nt++) {
                            const int n = 8 * nt + g;
                            const uint2 bf = n < T ? *reinterpret_cast<const uint2 *>(xw + n * XS + 8 * sj + off / 4) : make_uint2(0, 0);
                            const float4 sc = *reinterpret_cast<const float4 *>(xd + sj * TMAX + 8 * nt + 2 * tg);
                            int d[4];
                            mma_s8_k32(d, a0, a1, a2, a3, bf);
                            acc[nt][0] = fmaf(f0 * sc.x, (float) d[0], acc[nt][0]);
                            acc[nt][1] = fmaf(f0 * sc.z, (float) d[1], acc[nt][1]);
                            acc[nt][2] = fmaf(f1 * sc.x, (float) d[2], acc[nt][2]);
                            acc[nt][3] = fmaf(f1 * sc.z, (float) d[3], acc[nt][3]);
                            accm[nt][0] = fmaf(g0, sc.y, accm[nt][0]);
                            accm[nt][1] = fmaf(g0, sc.w, accm[nt][1]);
                            accm[nt][2] = fmaf(g1, sc.y, accm[nt][2]);
                            accm[nt][3] = fmaf(g1, sc.w, accm[nt][3]);
                        }
                    }
                }
            }
          }
            float * slot = red + gw * 16 * T;
#pragma unroll
            for (int nt = 0; nt < NTL; nt++)
#pragma unroll
                for (int c = 0; c < 4; c++) {
                    const int t = 8 * nt + 2 * tg + (c & 1);
                    if (t < T) slot[(g + 8 * (c >> 1)) * T + t] = acc[nt][c] - accm[nt][c];
                }
            asm volatile("bar.sync %0, %1;" ::"r"(1 + grp), "r"(32 * GS) : "memory");
            const int tid = threadIdx.x % (32 * GS);
            for (int o = tid; o < 8 * T; o += 32 * GS) {
                const int tid = o;
                const int row = tid / T, t = tid % T;
                float a = 0.f, u = 0.f;
#pragma unroll
                for (int w8 = 0; w8 < GS; w8++) {
                    a += red[(w8 * 16 + row) * T + t];
                    u += red[(w8 * 16 + 8 + row) * T + t];
                }
                out[(size_t) t * FF + 8 * it + row] = a / (1.f + __expf(-a)) * u;
            }
            asm volatile("bar.sync %0, %1;" ::"r"(1 + grp), "r"(32 * GS) : "memory");
        }
}

// ---- activation-granularity study: float activations xf (smooth + noise + rare outliers), quantised per 32
// (q8_1-like, M32K8) or per 256 (Q8_K-like, M256), both against the exact float result
__host__ __device__ inline float xf(int t, int k) {
    unsigned h = (unsigned) (t * 1000003 + k * 7919) * 2654435761u;
    float v = ((h >> 8) & 0xffff) / 32768.f - 1.f;
    if ((h & 0x3ff) == 7) v *= 12.f; // outliers (~0.1%)
    return v * (1.f + 0.5f * sinf(0.01f * k + t));
}
__device__ void setup_q32(uint8_t * sm, int T) { // M32K8's layout (natural order), from xf
    unsigned * xw = (unsigned *) sm;
    float2 * xd = (float2 *) (xw + TMAX * XS);
    for (int i = threadIdx.x; i < T * NJ; i += NT) {
        const int t = i / NJ, j = i % NJ;
        float amax = 0.f;
        for (int k = 0; k < 32; k++) amax = fmaxf(amax, fabsf(xf(t, 32 * j + k)));
        const float dx = amax / 127.f;
        int sum = 0;
        for (int q = 0; q < 8; q++) {
            unsigned v = 0;
            for (int b = 0; b < 4; b++) { const int x = amax == 0.f ? 0 : __float2int_rn(xf(t, 32 * j + 4 * q + b) / dx); sum += x; v |= (unsigned) (uint8_t) (int8_t) x << (8 * b); }
            xw[t * XS + 8 * j + q] = v;
        }
        xd[j * TMAX + t] = make_float2(dx, dx * (float) sum);
    }
    __syncthreads();
}
// M256 layout: xw as above (natural order, 8-bit values quantised per 256), scale xb[blk][t], sub-block sums xs[j][t]
__device__ void setup_q256(uint8_t * sm, int T) {
    unsigned * xw = (unsigned *) sm;
    float * xb = (float *) (xw + TMAX * XS);
    int * xs = (int *) (xb + (K / 256) * TMAX);
    for (int i = threadIdx.x; i < T * (K / 256); i += NT) {
        const int t = i / (K / 256), b = i % (K / 256);
        float amax = 0.f;
        for (int k = 0; k < 256; k++) amax = fmaxf(amax, fabsf(xf(t, 256 * b + k)));
        const float dx = amax / 127.f;
        xb[b * TMAX + t] = dx;
        for (int jj = 0; jj < 8; jj++) {
            const int j = 8 * b + jj;
            int sum = 0;
            for (int q = 0; q < 8; q++) {
                unsigned v = 0;
                for (int bb = 0; bb < 4; bb++) { const int x = amax == 0.f ? 0 : __float2int_rn(xf(t, 32 * j + 4 * q + bb) / dx); sum += x; v |= (unsigned) (uint8_t) (int8_t) x << (8 * bb); }
                xw[t * XS + 8 * j + q] = v;
            }
            xs[j * TMAX + t] = sum;
        }
    }
    __syncthreads();
}
__device__ __forceinline__ void q4k_ints(const int4 & h, int p, int & sc0, int & m0, int & sc1, int & m1) { // 6-bit scales / mins of sub-blocks 2p, 2p + 1
    const int sh = (p & 1) * 16; const unsigned y = (unsigned) h.y >> sh, z = (unsigned) h.z >> sh, w = (unsigned) h.w >> sh;
    if (p < 2) { sc0 = y & 63; sc1 = (y >> 8) & 63; m0 = z & 63; m1 = (z >> 8) & 63; }
    else { sc0 = (w & 0xF) | ((y >> 2) & 0x30); sc1 = ((w >> 8) & 0xF) | ((y >> 10) & 0x30); m0 = ((w >> 4) & 0xF) | ((z >> 2) & 0x30); m1 = ((w >> 12) & 0xF) | ((z >> 10) & 0x30); }
}
// M256: M32K8 with per-256 activation scales: per sub-block only integer work (acc_i += sc * D, accm_i += m * S),
// per (block, row, token) one float step: acc += dx_b * (d * acc_i - dmin * accm_i)
template <int T>
__global__ void __launch_bounds__(NT, 1) ffn_mma256(const uint8_t * Wg, const uint8_t * Wu, float * out, size_t ls) {
    constexpr int NTL = (T + 7) / 8, NI = FF / 8;
    extern __shared__ __align__(16) uint8_t sm[];
    setup_q256(sm, T);
    const unsigned * xw = (const unsigned *) sm;
    const float * xb = (const float *) (xw + TMAX * XS);
    const int * xs = (const int *) (xb + (K / 256) * TMAX);
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, g = lane >> 2, tg = lane & 3;
    const int grp = warp >> 3, gw = warp & 7;
    float * red = (float *) (xs + NJ * TMAX) + grp * 8 * 16 * T;
    const int P0 = 5 * gw, hb0 = P0 / 2;
    const int offA = (tg & 1) * 16 + (tg >> 1) * 8, offB = offA ^ 8;
    const int it0 = 2 * blockIdx.x + grp, step = 2 * gridDim.x;
    for (int L = 0; L < NLAY; L++)
        for (int it = it0; it < NI; it += step) {
            uint4 W[2][5];
            int4 H[2][3];
#pragma unroll
            for (int k = 0; k < 2; k++) {
                const uint8_t * w = (k ? Wu : Wg) + L * ls + (size_t) (8 * it + g) * RB;
#pragma unroll
                for (int b = 0; b < 3; b++) H[k][b] = (hb0 + b) < K / 256 ? ldw4c(w + (hb0 + b) * 144) : make_int4(0, 0, 0, 0);
#pragma unroll
                for (int i = 0; i < 5; i++) {
                    const int P = P0 + i;
                    const int4 v = ldw4c(w + (P / 2) * 144 + 16 + 64 * (P % 2) + 16 * tg);
                    W[k][i] = make_uint4(v.x, v.y, v.z, v.w);
                }
            }
            float acc[NTL][4] = {};
            int ai[NTL][4] = {}, am[NTL][4] = {};
#pragma unroll
            for (int i = 0; i < 5; i++) {
                const int P = P0 + i, bk = P / 2, bi = bk - hb0;
                uint2 cA[2], cB[2];
#pragma unroll
                for (int k = 0; k < 2; k++) {
                    const unsigned ox = __shfl_xor_sync(~0u, W[k][i].z, 2), oy = __shfl_xor_sync(~0u, W[k][i].w, 2);
                    const uint2 own = make_uint2(W[k][i].x, W[k][i].y), got = make_uint2(ox, oy);
                    cA[k] = tg < 2 ? own : got;
                    cB[k] = tg < 2 ? got : own;
                }
                int4 hcur[2];
#pragma unroll
                for (int k = 0; k < 2; k++) hcur[k] = bi == 0 ? H[k][0] : (bi == 1 ? H[k][1] : H[k][2]);
#pragma unroll
                for (int hc = 0; hc < 2; hc++) {
                    const int p = 2 * (P % 2) + hc, off = hc ? offB : offA;
                    int sc[2][2], mn[2][2]; // [row][nib]
#pragma unroll
                    for (int k = 0; k < 2; k++) q4k_ints(hcur[k], p, sc[k][0], mn[k][0], sc[k][1], mn[k][1]);
                    const uint2 r0 = hc ? cB[0] : cA[0], r1 = hc ? cB[1] : cA[1];
#pragma unroll
                    for (int nib = 0; nib < 2; nib++) {
                        const int sj = 8 * bk + 2 * p + nib, sh = 4 * nib;
                        const unsigned a0 = (r0.x >> sh) & M4, a1 = (r1.x >> sh) & M4, a2 = (r0.y >> sh) & M4, a3 = (r1.y >> sh) & M4;
#pragma unroll
                        for (int nt = 0; nt < NTL; nt++) {
                            const int n = 8 * nt + g;
                            const uint2 bf = n < T ? *reinterpret_cast<const uint2 *>(xw + n * XS + 8 * sj + off / 4) : make_uint2(0, 0);
                            const int2 S = *reinterpret_cast<const int2 *>(xs + sj * TMAX + 8 * nt + 2 * tg);
                            int d[4];
                            mma_s8_k32(d, a0, a1, a2, a3, bf);
                            ai[nt][0] += sc[0][nib] * d[0]; ai[nt][1] += sc[0][nib] * d[1];
                            ai[nt][2] += sc[1][nib] * d[2]; ai[nt][3] += sc[1][nib] * d[3];
                            am[nt][0] += mn[0][nib] * S.x; am[nt][1] += mn[0][nib] * S.y;
                            am[nt][2] += mn[1][nib] * S.x; am[nt][3] += mn[1][nib] * S.y;
                        }
                    }
                }
                // block done (the next pair is in another block, or the slice ends): one float step per output
                if (i == 4 || (P + 1) / 2 != bk) {
                    float2 dm[2];
#pragma unroll
                    for (int k = 0; k < 2; k++) dm[k] = __half22float2(*reinterpret_cast<const __half2 *>(&hcur[k].x));
#pragma unroll
                    for (int nt = 0; nt < NTL; nt++) {
                        const float2 dx = *reinterpret_cast<const float2 *>(xb + bk * TMAX + 8 * nt + 2 * tg);
#pragma unroll
                        for (int c = 0; c < 4; c++) {
                            const int r = c >> 1;
                            acc[nt][c] = fmaf((c & 1) ? dx.y : dx.x, dm[r].x * (float) ai[nt][c] - dm[r].y * (float) am[nt][c], acc[nt][c]);
                            ai[nt][c] = 0; am[nt][c] = 0;
                        }
                    }
                }
            }
            float * slot = red + gw * 16 * T;
#pragma unroll
            for (int nt = 0; nt < NTL; nt++)
#pragma unroll
                for (int c = 0; c < 4; c++) {
                    const int t = 8 * nt + 2 * tg + (c & 1);
                    if (t < T) slot[(g + 8 * (c >> 1)) * T + t] = acc[nt][c];
                }
            K7_BAR(1 + grp);
            const int tid = threadIdx.x & 255;
            if (tid < 8 * T) {
                const int row = tid / T, t = tid % T;
                float a = 0.f, u = 0.f;
#pragma unroll
                for (int w8 = 0; w8 < 8; w8++) {
                    a += red[(w8 * 16 + row) * T + t];
                    u += red[(w8 * 16 + 8 + row) * T + t];
                }
                out[(size_t) t * FF + 8 * it + row] = a / (1.f + __expf(-a)) * u;
            }
            K7_BAR(1 + grp);
        }
}
// exact reference: dequantised weights . float activations xf (last layer)
__global__ void ref_exact(const uint8_t * Wg, const uint8_t * Wu, float * out, size_t ls, int T) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= T * FF) return;
    const int t = i / FF, row = i % FF;
    float a[2];
    for (int m = 0; m < 2; m++) {
        const uint8_t * w = (m ? Wu : Wg) + (NLAY - 1) * ls + (size_t) row * RB;
        float acc = 0.f;
        for (int b = 0; b < K / 256; b++) {
            const int4 h = *reinterpret_cast<const int4 *>(w + b * 144);
            for (int p = 0; p < 4; p++) {
                float fA, gA, fB, gB;
                q4k_scales(h, p, fA, gA, fB, gB);
                for (int half = 0; half < 2; half++) {
                    const int sj = 8 * b + 2 * p + half;
                    for (int k = 0; k < 32; k++) {
                        const int q = (w[b * 144 + 16 + 32 * p + k] >> (4 * half)) & 15;
                        acc += ((half ? fB : fA) * (float) q - (half ? gB : gA)) * xf(t, 32 * sj + k);
                    }
                }
            }
        }
        a[m] = acc;
    }
    out[(size_t) t * FF + row] = a[0] / (1.f + __expf(-a[0])) * a[1];
}

// ---- ffn_down shape: N = 5120 rows, K = 17408 (Q4_K), activations T x K quantised per 32 in global memory
// (L2-resident: the previous phase's output). MD: block = 4 groups of 4 warps; group q owns the 16-row item
// blockIdx.x + gridDim.x q; K in chunks of 17 blocks (4352 values) staged in shared memory; in a chunk, the group's
// warp gw takes chunk pairs [34 gw / 4, 34 (gw + 1) / 4) (64-byte windows, loaded 5 pairs at a time); partial
// sums stay in registers across chunks, then the group's 4 warps combine.
constexpr int DN = 5120, DK = 17408, DRB = DK / 256 * 144, DNJ = DK / 32, DCB = 17, DCP = 2 * DCB, DXS = DCB * 64 + 8;
__host__ __device__ inline float xfd(int t, int k) { return xf(t + 31, k); }
__global__ void prep_down_act(unsigned * gxw, float2 * gxd, int T) { // per-32 quantised, [t][K/4 words], [j][TMAX]
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= T * DNJ) return;
    const int t = i / DNJ, j = i % DNJ;
    float amax = 0.f;
    for (int k = 0; k < 32; k++) amax = fmaxf(amax, fabsf(xfd(t, 32 * j + k)));
    const float dx = amax / 127.f;
    int sum = 0;
    for (int q = 0; q < 8; q++) {
        unsigned v = 0;
        for (int b = 0; b < 4; b++) { const int x = amax == 0.f ? 0 : __float2int_rn(xfd(t, 32 * j + 4 * q + b) / dx); sum += x; v |= (unsigned) (uint8_t) (int8_t) x << (8 * b); }
        gxw[(size_t) t * (DK / 4) + 8 * j + q] = v;
    }
    gxd[j * TMAX + t] = make_float2(dx, dx * (float) sum);
}
template <int T>
__global__ void __launch_bounds__(NT, 1) down_mma(const uint8_t * Wd, const unsigned * gxw, const float2 * gxd, float * out, size_t ls) {
    constexpr int NTL = (T + 7) / 8, NIT = DN / 16;
    extern __shared__ __align__(16) uint8_t sm[];
    unsigned * xw = (unsigned *) sm;                // [T][DXS] words: the chunk's activations
    float2 * xd = (float2 *) (xw + TMAX * DXS);     // [DCB * 8 sub-blocks][TMAX]
    float * red = (float *) (xd + DCB * 8 * TMAX);  // [16 warps][16 rows][T]
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, g = lane >> 2, tg = lane & 3;
    const int grp = warp >> 2, gw = warp & 3;
    const int item = blockIdx.x + gridDim.x * grp;
    const bool active = item < NIT;
    const int pa = DCP * gw / 4, pb = DCP * (gw + 1) / 4; // this warp's pairs within a chunk
    const int offA = (tg & 1) * 16 + (tg >> 1) * 8, offB = offA ^ 8;
    for (int L = 0; L < NLAY; L++) {
        float acc[NTL][4] = {}, accm[NTL][4] = {};
        for (int ch = 0; ch < DK / 256 / DCB; ch++) {
            __syncthreads(); // the previous chunk's readers are done
            for (int i = threadIdx.x; i < T * DCB * 64 / 4; i += NT) { // 16-byte pieces
                const int t = i / (DCB * 16), q = i % (DCB * 16);
                *reinterpret_cast<uint4 *>(xw + t * DXS + 4 * q) = __ldcg(reinterpret_cast<const uint4 *>(gxw + (size_t) t * (DK / 4) + ch * DCB * 64) + q);
            }
            for (int i = threadIdx.x; i < DCB * 8 * TMAX; i += NT) xd[i] = __ldcg(gxd + ch * DCB * 8 * TMAX + i);
            __syncthreads();
            if (!active) continue;
            const uint8_t * w0 = Wd + L * ls + (size_t) (16 * item + g) * DRB + ch * DCB * 144;
            const uint8_t * w1 = w0 + (size_t) 8 * DRB;
            for (int pp = pa; pp < pb; pp += 5) {
                const int hb0 = pp / 2;
                uint4 W[2][5];
                int4 H[2][3];
#pragma unroll
                for (int k = 0; k < 2; k++) {
                    const uint8_t * w = k ? w1 : w0;
#pragma unroll
                    for (int b = 0; b < 3; b++) H[k][b] = (hb0 + b) < DCB ? ldw4c(w + (hb0 + b) * 144) : make_int4(0, 0, 0, 0);
#pragma unroll
                    for (int i = 0; i < 5; i++) {
                        const int P = pp + i;
                        if (P < pb) { const int4 v = ldw4c(w + (P / 2) * 144 + 16 + 64 * (P % 2) + 16 * tg); W[k][i] = make_uint4(v.x, v.y, v.z, v.w); }
                        else W[k][i] = make_uint4(0, 0, 0, 0);
                    }
                }
#pragma unroll
                for (int i = 0; i < 5; i++) {
                    const int P = pp + i;
                    if (P >= pb) break;
                    const int bk = P / 2, bi = bk - hb0;
                    uint2 cA[2], cB[2];
#pragma unroll
                    for (int k = 0; k < 2; k++) {
                        const unsigned ox = __shfl_xor_sync(~0u, W[k][i].z, 2), oy = __shfl_xor_sync(~0u, W[k][i].w, 2);
                        const uint2 own = make_uint2(W[k][i].x, W[k][i].y), got = make_uint2(ox, oy);
                        cA[k] = tg < 2 ? own : got;
                        cB[k] = tg < 2 ? got : own;
                    }
                    int4 hcur[2];
#pragma unroll
                    for (int k = 0; k < 2; k++) hcur[k] = sel4(bi == 0, H[k][0], sel4(bi == 1, H[k][1], H[k][2]));
#pragma unroll
                    for (int hc = 0; hc < 2; hc++) {
                        const int p = 2 * (P % 2) + hc, off = hc ? offB : offA;
                        float fA[2], gA[2], fB[2], gB[2];
#pragma unroll
                        for (int k = 0; k < 2; k++) q4k_scales(hcur[k], p, fA[k], gA[k], fB[k], gB[k]);
                        const uint2 r0 = hc ? cB[0] : cA[0], r1 = hc ? cB[1] : cA[1];
#pragma unroll
                        for (int nib = 0; nib < 2; nib++) {
                            const int sj = 8 * bk + 2 * p + nib, sh = 4 * nib; // chunk-local sub-block
                            const float f0 = nib ? fB[0] : fA[0], f1 = nib ? fB[1] : fA[1];
                            const float g0 = nib ? gB[0] : gA[0], g1 = nib ? gB[1] : gA[1];
                            const unsigned a0 = (r0.x >> sh) & M4, a1 = (r1.x >> sh) & M4, a2 = (r0.y >> sh) & M4, a3 = (r1.y >> sh) & M4;
#pragma unroll
                            for (int nt = 0; nt < NTL; nt++) {
                                const int n = 8 * nt + g;
                                const uint2 bf = n < T ? *reinterpret_cast<const uint2 *>(xw + n * DXS + 8 * sj + off / 4) : make_uint2(0, 0);
                                const float4 sc = *reinterpret_cast<const float4 *>(xd + sj * TMAX + 8 * nt + 2 * tg);
                                int d[4];
                                mma_s8_k32(d, a0, a1, a2, a3, bf);
                                acc[nt][0] = fmaf(f0 * sc.x, (float) d[0], acc[nt][0]);
                                acc[nt][1] = fmaf(f0 * sc.z, (float) d[1], acc[nt][1]);
                                acc[nt][2] = fmaf(f1 * sc.x, (float) d[2], acc[nt][2]);
                                acc[nt][3] = fmaf(f1 * sc.z, (float) d[3], acc[nt][3]);
                                accm[nt][0] = fmaf(g0, sc.y, accm[nt][0]);
                                accm[nt][1] = fmaf(g0, sc.w, accm[nt][1]);
                                accm[nt][2] = fmaf(g1, sc.y, accm[nt][2]);
                                accm[nt][3] = fmaf(g1, sc.w, accm[nt][3]);
                            }
                        }
                    }
                }
            }
        }
        // combine the group's 4 warps
        float * slot = red + warp * 16 * T;
#pragma unroll
        for (int nt = 0; nt < NTL; nt++)
#pragma unroll
            for (int c = 0; c < 4; c++) {
                const int t = 8 * nt + 2 * tg + (c & 1);
                if (t < T) slot[(g + 8 * (c >> 1)) * T + t] = acc[nt][c] - accm[nt][c];
            }
        __syncthreads();
        const int tid = threadIdx.x & 127;
        if (active)
            for (int o = tid; o < 16 * T; o += 128) {
                const int row = o / T, t = o % T;
                float a = 0.f;
#pragma unroll
                for (int w4 = 0; w4 < 4; w4++) a += red[((grp * 4 + w4) * 16 + row) * T + t];
                out[(size_t) t * DN + 16 * item + row] = a;
            }
    }
}

// MD2: down_mma with double-buffered steps of 3 pairs (40 registers per buffer): step s + 1's weights are loaded
// before step s is computed, across chunk and layer boundaries (weights do not depend on the staged activations).
// Every warp takes 3 steps per chunk (its 8 or 9 pairs), so the block meets at chunk boundaries in lockstep.
template <int T>
__global__ void __launch_bounds__(NT, 1) down_mma2(const uint8_t * Wd, const unsigned * gxw, const float2 * gxd, float * out, size_t ls) {
    constexpr int NTL = (T + 7) / 8, NIT = DN / 16, NCH = DK / 256 / DCB, SPC = 3, NS = NLAY * NCH * SPC;
    extern __shared__ __align__(16) uint8_t sm[];
    unsigned * xw = (unsigned *) sm;
    float2 * xd = (float2 *) (xw + TMAX * DXS);
    float * red = (float *) (xd + DCB * 8 * TMAX);
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, g = lane >> 2, tg = lane & 3;
    const int grp = warp >> 2, gw = warp & 3;
    const int item = blockIdx.x + gridDim.x * grp;
    const bool active = item < NIT;
    const int pa = DCP * gw / 4, pb = DCP * (gw + 1) / 4;
    const int offA = (tg & 1) * 16 + (tg >> 1) * 8, offB = offA ^ 8;
    const uint8_t * wrow = Wd + (size_t) (16 * (active ? item : 0) + g) * DRB;
#define MD_LOAD(W, H, S_)                                                                              \
    if (active && (S_) < NS) {                                                                         \
        const int L_ = (S_) / (NCH * SPC), ch_ = ((S_) / SPC) % NCH, p0_ = pa + 3 * ((S_) % SPC), hb_ = p0_ / 2; \
        _Pragma("unroll") for (int k = 0; k < 2; k++) {                                                \
            const uint8_t * w = wrow + L_ * ls + (size_t) 8 * k * DRB + ch_ * DCB * 144;               \
            _Pragma("unroll") for (int b = 0; b < 2; b++) H[k][b] = (hb_ + b) < DCB ? ldw4c(w + (hb_ + b) * 144) : make_int4(0, 0, 0, 0); \
            _Pragma("unroll") for (int i = 0; i < 3; i++) {                                            \
                const int P = p0_ + i;                                                                 \
                if (P < pb) { const int4 v = ldw4c(w + (P / 2) * 144 + 16 + 64 * (P % 2) + 16 * tg); W[k][i] = make_uint4(v.x, v.y, v.z, v.w); } \
                else W[k][i] = make_uint4(0, 0, 0, 0);                                                 \
            }                                                                                          \
        }                                                                                              \
    }
#define MD_STEP(W, H, S_)                                                                              \
    {                                                                                                  \
        const int L_ = (S_) / (NCH * SPC), ch_ = ((S_) / SPC) % NCH, ks_ = (S_) % SPC, p0_ = pa + 3 * ks_, hb_ = p0_ / 2; \
        if (ks_ == 0) {                                                                                \
            __syncthreads();                                                                           \
            for (int i = threadIdx.x; i < T * DCB * 16; i += NT) {                                     \
                const int t = i / (DCB * 16), q = i % (DCB * 16);                                      \
                *reinterpret_cast<uint4 *>(xw + t * DXS + 4 * q) = __ldcg(reinterpret_cast<const uint4 *>(gxw + (size_t) t * (DK / 4) + ch_ * DCB * 64) + q); \
            }                                                                                          \
            for (int i = threadIdx.x; i < DCB * 8 * TMAX; i += NT) xd[i] = __ldcg(gxd + ch_ * DCB * 8 * TMAX + i); \
            __syncthreads();                                                                           \
        }                                                                                              \
        if (active)                                                                                    \
        _Pragma("unroll") for (int i = 0; i < 3; i++) {                                                \
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
            int4 hcur[2];                                                                              \
            _Pragma("unroll") for (int k = 0; k < 2; k++) hcur[k] = sel4(bk == hb_, H[k][0], H[k][1]); \
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
                        const uint2 bf = n < T ? *reinterpret_cast<const uint2 *>(xw + n * DXS + 8 * sj + off / 4) : make_uint2(0, 0); \
                        const float4 sc = *reinterpret_cast<const float4 *>(xd + sj * TMAX + 8 * nt + 2 * tg); \
                        int d[4];                                                                      \
                        mma_s8_k32(d, a0, a1, a2, a3, bf);                                             \
                        acc[nt][0] = fmaf(f0 * sc.x, (float) d[0], acc[nt][0]);                        \
                        acc[nt][1] = fmaf(f0 * sc.z, (float) d[1], acc[nt][1]);                        \
                        acc[nt][2] = fmaf(f1 * sc.x, (float) d[2], acc[nt][2]);                        \
                        acc[nt][3] = fmaf(f1 * sc.z, (float) d[3], acc[nt][3]);                        \
                        accm[nt][0] = fmaf(g0, sc.y, accm[nt][0]);                                     \
                        accm[nt][1] = fmaf(g0, sc.w, accm[nt][1]);                                     \
                        accm[nt][2] = fmaf(g1, sc.y, accm[nt][2]);                                     \
                        accm[nt][3] = fmaf(g1, sc.w, accm[nt][3]);                                     \
                    }                                                                                  \
                }                                                                                      \
            }                                                                                          \
            }                                                                                          \
        }                                                                                              \
        if ((S_) % (NCH * SPC) == NCH * SPC - 1) { /* layer done: combine the group's 4 warps */      \
            float * slot = red + warp * 16 * T;                                                        \
            _Pragma("unroll") for (int nt = 0; nt < NTL; nt++)                                         \
                _Pragma("unroll") for (int c = 0; c < 4; c++) {                                        \
                    const int t = 8 * nt + 2 * tg + (c & 1);                                           \
                    if (t < T) slot[(g + 8 * (c >> 1)) * T + t] = acc[nt][c] - accm[nt][c];            \
                    acc[nt][c] = 0.f; accm[nt][c] = 0.f;                                               \
                }                                                                                      \
            __syncthreads();                                                                           \
            if (active)                                                                                \
                for (int o = threadIdx.x & 127; o < 16 * T; o += 128) {                                \
                    const int row = o / T, t = o % T;                                                  \
                    float a = 0.f;                                                                     \
                    _Pragma("unroll") for (int w4 = 0; w4 < 4; w4++) a += red[((grp * 4 + w4) * 16 + row) * T + t]; \
                    out[(size_t) t * DN + 16 * item + row] = a;                                        \
                }                                                                                      \
            (void) L_;                                                                                 \
        }                                                                                              \
    }
    float acc[NTL][4] = {}, accm[NTL][4] = {};
    uint4 WA[2][3], WB[2][3];
    int4 HA[2][2], HB[2][2];
    MD_LOAD(WA, HA, 0)
    for (int s0 = 0; s0 < NS; s0 += 2) {
        MD_LOAD(WB, HB, s0 + 1)
        MD_STEP(WA, HA, s0)
        MD_LOAD(WA, HA, s0 + 2)
        MD_STEP(WB, HB, s0 + 1)
    }
#undef MD_LOAD
#undef MD_STEP
}
// MD3: MD2 with header words (lane tg keeps word tg of each 16-byte header; the quad shuffles them together when
// a pair needs them): 28 registers per buffer.
// (base) MD2: down_mma with double-buffered steps of 3 pairs (40 registers per buffer): step s + 1's weights are loaded
// before step s is computed, across chunk and layer boundaries (weights do not depend on the staged activations).
// Every warp takes 3 steps per chunk (its 8 or 9 pairs), so the block meets at chunk boundaries in lockstep.
template <int T, int SP = 3>
__global__ void __launch_bounds__(NT, 1) down_mma3(const uint8_t * Wd, const unsigned * gxw, const float2 * gxd, float * out, size_t ls) {
    constexpr int NTL = (T + 7) / 8, NIT = DN / 16, NCH = DK / 256 / DCB, SPC = (DCP / 4 + 1 + SP - 1) / SP, NS = NLAY * NCH * SPC;
    extern __shared__ __align__(16) uint8_t sm[];
    unsigned * xw = (unsigned *) sm;
    float2 * xd = (float2 *) (xw + TMAX * DXS);
    float * red = (float *) (xd + DCB * 8 * TMAX);
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, g = lane >> 2, tg = lane & 3;
    const int grp = warp >> 2, gw = warp & 3;
    const int item = blockIdx.x + gridDim.x * grp;
    const bool active = item < NIT;
    const int pa = DCP * gw / 4, pb = DCP * (gw + 1) / 4;
    const int offA = (tg & 1) * 16 + (tg >> 1) * 8, offB = offA ^ 8;
    const uint8_t * wrow = Wd + (size_t) (16 * (active ? item : 0) + g) * DRB;
#define MD3_LOAD(W, H, S_)                                                                              \
    if (active && (S_) < NS) {                                                                         \
        const int L_ = (S_) / (NCH * SPC), ch_ = ((S_) / SPC) % NCH, p0_ = pa + SP * ((S_) % SPC), hb_ = p0_ / 2; \
        _Pragma("unroll") for (int k = 0; k < 2; k++) {                                                \
            const uint8_t * w = wrow + L_ * ls + (size_t) 8 * k * DRB + ch_ * DCB * 144;               \
            _Pragma("unroll") for (int b = 0; b < 2; b++) H[k][b] = (hb_ + b) < DCB ? ldw1c(w + (hb_ + b) * 144 + 4 * tg) : 0u; \
            _Pragma("unroll") for (int i = 0; i < SP; i++) {                                           \
                const int P = p0_ + i;                                                                 \
                if (P < pb) { const int4 v = ldw4c(w + (P / 2) * 144 + 16 + 64 * (P % 2) + 16 * tg); W[k][i] = make_uint4(v.x, v.y, v.z, v.w); } \
                else W[k][i] = make_uint4(0, 0, 0, 0);                                                 \
            }                                                                                          \
        }                                                                                              \
    }
#define MD3_STEP(W, H, S_)                                                                              \
    {                                                                                                  \
        const int L_ = (S_) / (NCH * SPC), ch_ = ((S_) / SPC) % NCH, ks_ = (S_) % SPC, p0_ = pa + SP * ks_, hb_ = p0_ / 2; \
        if (ks_ == 0) {                                                                                \
            __syncthreads();                                                                           \
            for (int i = threadIdx.x; i < T * DCB * 16; i += NT) {                                     \
                const int t = i / (DCB * 16), q = i % (DCB * 16);                                      \
                *reinterpret_cast<uint4 *>(xw + t * DXS + 4 * q) = __ldcg(reinterpret_cast<const uint4 *>(gxw + (size_t) t * (DK / 4) + ch_ * DCB * 64) + q); \
            }                                                                                          \
            for (int i = threadIdx.x; i < DCB * 8 * TMAX; i += NT) xd[i] = __ldcg(gxd + ch_ * DCB * 8 * TMAX + i); \
            __syncthreads();                                                                           \
        }                                                                                              \
        if (active)                                                                                    \
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
            int4 hcur[2];                                                                              \
            _Pragma("unroll") for (int k = 0; k < 2; k++) {                                            \
                const unsigned hw = bk == hb_ ? H[k][0] : H[k][1];                                     \
                hcur[k] = make_int4(__shfl_sync(~0u, hw, lane & ~3), __shfl_sync(~0u, hw, (lane & ~3) + 1), __shfl_sync(~0u, hw, (lane & ~3) + 2), __shfl_sync(~0u, hw, (lane & ~3) + 3)); \
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
                        const uint2 bf = n < T ? *reinterpret_cast<const uint2 *>(xw + n * DXS + 8 * sj + off / 4) : make_uint2(0, 0); \
                        const float4 sc = *reinterpret_cast<const float4 *>(xd + sj * TMAX + 8 * nt + 2 * tg); \
                        int d[4];                                                                      \
                        mma_s8_k32(d, a0, a1, a2, a3, bf);                                             \
                        acc[nt][0] = fmaf(f0 * sc.x, (float) d[0], acc[nt][0]);                        \
                        acc[nt][1] = fmaf(f0 * sc.z, (float) d[1], acc[nt][1]);                        \
                        acc[nt][2] = fmaf(f1 * sc.x, (float) d[2], acc[nt][2]);                        \
                        acc[nt][3] = fmaf(f1 * sc.z, (float) d[3], acc[nt][3]);                        \
                        acc[nt][0] = fmaf(-g0, sc.y, acc[nt][0]);                                     \
                        acc[nt][1] = fmaf(-g0, sc.w, acc[nt][1]);                                     \
                        acc[nt][2] = fmaf(-g1, sc.y, acc[nt][2]);                                     \
                        acc[nt][3] = fmaf(-g1, sc.w, acc[nt][3]);                                     \
                    }                                                                                  \
                }                                                                                      \
            }                                                                                          \
            }                                                                                          \
        }                                                                                              \
        if ((S_) % (NCH * SPC) == NCH * SPC - 1) { /* layer done: combine the group's 4 warps */      \
            float * slot = red + warp * 16 * T;                                                        \
            _Pragma("unroll") for (int nt = 0; nt < NTL; nt++)                                         \
                _Pragma("unroll") for (int c = 0; c < 4; c++) {                                        \
                    const int t = 8 * nt + 2 * tg + (c & 1);                                           \
                    if (t < T) slot[(g + 8 * (c >> 1)) * T + t] = acc[nt][c];            \
                    acc[nt][c] = 0.f;                                               \
                }                                                                                      \
            __syncthreads();                                                                           \
            if (active)                                                                                \
                for (int o = threadIdx.x & 127; o < 16 * T; o += 128) {                                \
                    const int row = o / T, t = o % T;                                                  \
                    float a = 0.f;                                                                     \
                    _Pragma("unroll") for (int w4 = 0; w4 < 4; w4++) a += red[((grp * 4 + w4) * 16 + row) * T + t]; \
                    out[(size_t) t * DN + 16 * item + row] = a;                                        \
                }                                                                                      \
            (void) L_;                                                                                 \
        }                                                                                              \
    }
    float acc[NTL][4] = {};
    uint4 WA[2][SP], WB[2][SP];
    unsigned HA[2][2], HB[2][2];
    MD3_LOAD(WA, HA, 0)
    for (int s0 = 0; s0 < NS; s0 += 2) {
        MD3_LOAD(WB, HB, s0 + 1)
        MD3_STEP(WA, HA, s0)
        MD3_LOAD(WA, HA, s0 + 2)
        MD3_STEP(WB, HB, s0 + 1)
    }
#undef MD3_LOAD
#undef MD3_STEP
}

// MDQ6: ffn_down with Q6_K rows (repacked 224-byte blocks): down_mma3's K-chunked pipeline + q6_mma2's masked-half math
constexpr int Q6BD = 224, DRB6 = DK / 256 * Q6BD;
template <int T, int SP = 2>
__global__ void __launch_bounds__(NT, 1) down_q6(const uint8_t * Wd, const unsigned * gxw, const float2 * gxd, float * out, size_t ls) {
    constexpr int NTL = (T + 7) / 8, NIT = DN / 16, NCH = DK / 256 / DCB, SPC = (DCP / 4 + 1 + SP - 1) / SP, NS = NLAY * NCH * SPC;
    extern __shared__ __align__(16) uint8_t sm[];
    unsigned * xw = (unsigned *) sm;
    float2 * xd = (float2 *) (xw + TMAX * DXS);
    float * red = (float *) (xd + DCB * 8 * TMAX);
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, g = lane >> 2, tg = lane & 3, hh = tg & 1, qb = lane & ~3;
    const int grp = warp >> 2, gw = warp & 3;
    const int item = blockIdx.x + gridDim.x * grp;
    const bool active = item < NIT;
    const int pa = DCP * gw / 4, pb = DCP * (gw + 1) / 4;
    const int offA = (tg & 1) * 16 + (tg >> 1) * 8, offB = offA ^ 8;
    const uint8_t * wrow = Wd + (size_t) (16 * (active ? item : 0) + g) * DRB6;
#define DQ_LOAD(W, QH, MT, S_)                                                                         \
    if (active && (S_) < NS) {                                                                         \
        const int L_ = (S_) / (NCH * SPC), ch_ = ((S_) / SPC) % NCH, p0_ = pa + SP * ((S_) % SPC);     \
        _Pragma("unroll") for (int k = 0; k < 2; k++) {                                                \
            const uint8_t * w = wrow + L_ * ls + (size_t) 8 * k * DRB6 + ch_ * DCB * Q6BD;              \
            _Pragma("unroll") for (int i = 0; i < SP; i++) {                                           \
                const int P = p0_ + i;                                                                 \
                if (P < pb) {                                                                          \
                    const uint8_t * blk = w + (P / 2) * Q6BD;                                           \
                    const int pc = 2 * (P % 2);                                                        \
                    const int4 v = ldw4c(blk + 64 * (P % 2) + 16 * tg);                                \
                    W[k][i] = make_uint4(v.x, v.y, v.z, v.w);                                          \
                    QH[k][i] = ldw2c(blk + 128 + 8 * (2 * pc + tg));                                      \
                    MT[k][i] = ldw1c(blk + 192 + 8 * pc + 4 * tg);                                     \
                } else { W[k][i] = make_uint4(0, 0, 0, 0); QH[k][i] = make_uint2(0, 0); MT[k][i] = 0u; }        \
            }                                                                                          \
        }                                                                                              \
    }
#define DQ_STEP(W, QH, MT, S_)                                                                         \
    {                                                                                                  \
        const int ch_ = ((S_) / SPC) % NCH, ks_ = (S_) % SPC, p0_ = pa + SP * ks_;                    \
        if (ks_ == 0) {                                                                                \
            __syncthreads();                                                                           \
            for (int i = threadIdx.x; i < T * DCB * 16; i += NT) {                                     \
                const int t = i / (DCB * 16), q = i % (DCB * 16);                                      \
                *reinterpret_cast<uint4 *>(xw + t * DXS + 4 * q) = __ldcg(reinterpret_cast<const uint4 *>(gxw + (size_t) t * (DK / 4) + ch_ * DCB * 64) + q); \
            }                                                                                          \
            for (int i = threadIdx.x; i < DCB * 8 * TMAX; i += NT) xd[i] = __ldcg(gxd + ch_ * DCB * 8 * TMAX + i); \
            __syncthreads();                                                                           \
        }                                                                                              \
        if (active)                                                                                    \
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
                int sc[2][2][2]; float dd[2]; unsigned hw[2][2];                                       \
                _Pragma("unroll") for (int k = 0; k < 2; k++) {                                        \
                    const uint4 m4 = make_uint4(__shfl_sync(~0u, MT[k][i], qb), __shfl_sync(~0u, MT[k][i], qb + 1), __shfl_sync(~0u, MT[k][i], qb + 2), __shfl_sync(~0u, MT[k][i], qb + 3)); \
                    const unsigned scw = hc ? m4.z : m4.x, dw = hc ? m4.w : m4.y;                      \
                    dd[k] = __half2float(__ushort_as_half((unsigned short) (dw & 0xffff)));            \
                    sc[k][0][0] = (int) (int8_t) (scw); sc[k][0][1] = (int) (int8_t) (scw >> 8);       \
                    sc[k][1][0] = (int) (int8_t) (scw >> 16); sc[k][1][1] = (int) (int8_t) (scw >> 24); \
                    hw[k][0] = __shfl_sync(~0u, QH[k][i].x, (lane & ~3) + 2 * hc + hh);                \
                    hw[k][1] = __shfl_sync(~0u, QH[k][i].y, (lane & ~3) + 2 * hc + hh);                \
                }                                                                                      \
                _Pragma("unroll") for (int nib = 0; nib < 2; nib++) {                                  \
                    const int sj = 8 * bk + 2 * p + nib, sh = 4 * nib;                                 \
                    unsigned a[4];                                                                     \
                    _Pragma("unroll") for (int k = 0; k < 2; k++) {                                    \
                        const unsigned lo0 = (r[k].x >> sh) & M4, lo1 = (r[k].y >> sh) & M4;           \
                        const unsigned h0 = (hw[k][nib] >> (2 * (i0 / 4))) & 0x03030303u, h1 = (hw[k][nib] >> (2 * (i0 / 4 + 1))) & 0x03030303u; \
                        a[k] = __vsub4(lo0 | (h0 << 4), 0x20202020u);                                  \
                        a[2 + k] = __vsub4(lo1 | (h1 << 4), 0x20202020u);                              \
                    }                                                                                  \
                    const unsigned z0 = hh ? 0u : ~0u, z1 = ~z0;                                       \
                    _Pragma("unroll") for (int nt = 0; nt < NTL; nt++) {                               \
                        const int n = 8 * nt + g;                                                      \
                        const uint2 bf = n < T ? *reinterpret_cast<const uint2 *>(xw + n * DXS + 8 * sj + off / 4) : make_uint2(0, 0); \
                        const float4 xs = *reinterpret_cast<const float4 *>(xd + sj * TMAX + 8 * nt + 2 * tg); \
                        int d0[4], d1[4];                                                              \
                        mma_s8_k32(d0, a[0] & z0, a[1] & z0, a[2] & z0, a[3] & z0, bf);                \
                        mma_s8_k32(d1, a[0] & z1, a[1] & z1, a[2] & z1, a[3] & z1, bf);                \
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
        if ((S_) % (NCH * SPC) == NCH * SPC - 1) {                                                     \
            float * slot = red + warp * 16 * T;                                                        \
            _Pragma("unroll") for (int nt = 0; nt < NTL; nt++)                                         \
                _Pragma("unroll") for (int c = 0; c < 4; c++) {                                        \
                    const int t = 8 * nt + 2 * tg + (c & 1);                                           \
                    if (t < T) slot[(g + 8 * (c >> 1)) * T + t] = acc[nt][c];                          \
                    acc[nt][c] = 0.f;                                                                  \
                }                                                                                      \
            __syncthreads();                                                                           \
            if (active)                                                                                \
                for (int o = threadIdx.x & 127; o < 16 * T; o += 128) {                                \
                    const int row = o / T, t = o % T;                                                  \
                    float a = 0.f;                                                                     \
                    _Pragma("unroll") for (int w4 = 0; w4 < 4; w4++) a += red[((grp * 4 + w4) * 16 + row) * T + t]; \
                    out[(size_t) t * DN + 16 * item + row] = a;                                        \
                }                                                                                      \
        }                                                                                              \
    }
    float acc[NTL][4] = {};
    uint4 WA[2][SP], WB[2][SP];
    uint2 QA[2][SP], QB[2][SP];
    unsigned MA[2][SP], MB[2][SP];
    DQ_LOAD(WA, QA, MA, 0)
    for (int s0 = 0; s0 < NS; s0 += 2) {
        DQ_LOAD(WB, QB, MB, s0 + 1)
        DQ_STEP(WA, QA, MA, s0)
        DQ_LOAD(WA, QA, MA, s0 + 2)
        DQ_STEP(WB, QB, MB, s0 + 1)
    }
#undef DQ_LOAD
#undef DQ_STEP
}
// exact reference for MDQ6 (last layer): integer dots on the per-32 quantised activations (gxw / gxd)
__global__ void ref_down_q6(const uint8_t * Wd, const unsigned * gxw, const float2 * gxd, float * out, size_t ls, int T) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= T * DN) return;
    const int t = i / DN, row = i % DN;
    const uint8_t * w = Wd + (NLAY - 1) * ls + (size_t) row * DRB6;
    const int8_t * xq = (const int8_t *) (gxw + (size_t) t * (DK / 4));
    double acc = 0;
    for (int b = 0; b < DK / 256; b++) {
        const uint8_t * blk = w + b * Q6BD;
        for (int p = 0; p < 4; p++) {
            unsigned short dbits; memcpy(&dbits, blk + 192 + 8 * p + 4, 2);
            const float d = __half2float(__ushort_as_half(dbits));
            for (int nib = 0; nib < 2; nib++) {
                const int sj = 8 * b + 2 * p + nib;
                const float dx = gxd[sj * TMAX + t].x;
                for (int v = 0; v < 32; v++) {
                    const int h = v / 16, iv = v % 16;
                    const int lo = (blk[32 * p + v] >> (4 * nib)) & 15;
                    unsigned hwd; memcpy(&hwd, blk + 128 + 8 * (2 * p + h) + 4 * nib, 4);
                    const int hi = (hwd >> (8 * (iv % 4) + 2 * (iv / 4))) & 3;
                    const int q = (lo | (hi << 4)) - 32;
                    const int sc = (int8_t) blk[192 + 8 * p + 2 * nib + h];
                    acc += (double) d * sc * q * dx * xq[32 * sj + v];
                }
            }
        }
    }
    out[(size_t) t * DN + row] = (float) acc;
}
// float reference (last layer)
__global__ void ref_down(const uint8_t * Wd, float * out, size_t ls, int T) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= T * DN) return;
    const int t = i / DN, row = i % DN;
    const uint8_t * w = Wd + (NLAY - 1) * ls + (size_t) row * DRB;
    float acc = 0.f;
    for (int b = 0; b < DK / 256; b++) {
        const int4 h = *reinterpret_cast<const int4 *>(w + b * 144);
        for (int p = 0; p < 4; p++) {
            float fA, gA, fB, gB;
            q4k_scales(h, p, fA, gA, fB, gB);
            for (int half = 0; half < 2; half++)
                for (int k = 0; k < 32; k++) {
                    const int q = (w[b * 144 + 16 + 32 * p + k] >> (4 * half)) & 15;
                    acc += ((half ? fB : fA) * (float) q - (half ? gB : gA)) * xfd(t, 256 * b + 64 * p + 32 * half + k);
                }
        }
    }
    out[(size_t) t * DN + row] = acc;
}
// dp4a reference point at T = 1 (lane-per-unit GEMV over each row, as the megakernel's T = 1 path): rows split over warps
template <int T>
__global__ void __launch_bounds__(NT, 1) down_dp4a(const uint8_t * Wd, const unsigned * gxw, const float2 * gxd, float * out, size_t ls) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, gwp = warp * gridDim.x + blockIdx.x, NWt = NW * gridDim.x;
    for (int L = 0; L < NLAY; L++)
        for (int row = gwp; row < DN; row += NWt) {
            const uint8_t * w = Wd + L * ls + (size_t) row * DRB;
            float a[T] = {};
            for (int u = lane; u < DNJ / 2; u += 32) { // unit = 64 values: chunk p of block b
                const int b = u / 4, p = u % 4;
                const int4 h = ldw4(w + b * 144);
                float fA, gA, fB, gB;
                q4k_scales(h, p, fA, gA, fB, gB);
                const int4 q0 = ldw4(w + b * 144 + 16 + 32 * p), q1 = ldw4(w + b * 144 + 16 + 32 * p + 16);
                const int qq[8] = {q0.x, q0.y, q0.z, q0.w, q1.x, q1.y, q1.z, q1.w};
#pragma unroll
                for (int t = 0; t < T; t++) {
                    const uint4 xa0 = __ldg(reinterpret_cast<const uint4 *>(gxw + (size_t) t * (DK / 4) + 16 * u));
                    const uint4 xa1 = __ldg(reinterpret_cast<const uint4 *>(gxw + (size_t) t * (DK / 4) + 16 * u) + 1);
                    const uint4 xb0 = __ldg(reinterpret_cast<const uint4 *>(gxw + (size_t) t * (DK / 4) + 16 * u) + 2);
                    const uint4 xb1 = __ldg(reinterpret_cast<const uint4 *>(gxw + (size_t) t * (DK / 4) + 16 * u) + 3);
                    const unsigned xa[8] = {xa0.x, xa0.y, xa0.z, xa0.w, xa1.x, xa1.y, xa1.z, xa1.w};
                    const unsigned xb[8] = {xb0.x, xb0.y, xb0.z, xb0.w, xb1.x, xb1.y, xb1.z, xb1.w};
                    int sa = 0, sb = 0;
#pragma unroll
                    for (int e = 0; e < 8; e++) { sa = __dp4a((int) (qq[e] & M4), (int) xa[e], sa); sb = __dp4a((int) ((qq[e] >> 4) & M4), (int) xb[e], sb); }
                    const float2 dA = __ldg(gxd + (2 * u) * TMAX + t), dB = __ldg(gxd + (2 * u + 1) * TMAX + t);
                    a[t] += fA * dA.x * (float) sa - gA * dA.y + fB * dB.x * (float) sb - gB * dB.y;
                }
            }
#pragma unroll
            for (int t = 0; t < T; t++) { const float v = wsum(a[t]); if (lane == 0) out[(size_t) t * DN + row] = v; }
        }
}

// GU3: gate/up with MD3's pipeline: 4-warp groups, 4 items (8 gate + 8 up rows) per block at a time; in an item
// warp gw takes pairs [10 gw, 10 gw + 10) in steps of SP pairs, double-buffered across steps and items (header
// words, one accumulator); per item the group's 4 warps combine behind a named barrier.
template <int T, int SP = 3>
__global__ void __launch_bounds__(NT, 1) gu_mma3(const uint8_t * Wg, const uint8_t * Wu, float * out, size_t ls) {
    constexpr int NTL = (T + 7) / 8, NI = FF / 8, SPC = (10 + SP - 1) / SP;
    extern __shared__ __align__(16) uint8_t sm[];
    setup32(sm, T, true);
    const unsigned * xw = (const unsigned *) sm;
    const float2 * xd = (const float2 *) (xw + TMAX * XS);
    float * red = (float *) (xd + NJ * TMAX); // [16 warps][16][T]
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, g = lane >> 2, tg = lane & 3;
    const int grp = warp >> 2, gw = warp & 3, pa = 10 * gw, pb = pa + 10;
    const int offA = (tg & 1) * 16 + (tg >> 1) * 8, offB = offA ^ 8;
    const int it0 = 4 * blockIdx.x + grp, istep = 4 * gridDim.x;
    // step descriptor: (L, it, ks); next(): ks + 1, then the next item, then the next layer
#define GU_NEXT(L_, IT_, KS_) { if (++KS_ == SPC) { KS_ = 0; IT_ += istep; if (IT_ >= NI) { IT_ = it0; L_++; } } }
#define GU_LOAD(W, H, L_, IT_, KS_)                                                                    \
    {                                                                                                  \
        const int p0_ = pa + SP * (KS_), hb_ = p0_ / 2;                                                \
        _Pragma("unroll") for (int k = 0; k < 2; k++) {                                                \
            const uint8_t * w = (k ? Wu : Wg) + (L_) * ls + (size_t) (8 * (IT_) + g) * RB;             \
            _Pragma("unroll") for (int b = 0; b < 2; b++) H[k][b] = (hb_ + b) < K / 256 ? ldw1c(w + (hb_ + b) * 144 + 4 * tg) : 0u; \
            _Pragma("unroll") for (int i = 0; i < SP; i++) {                                           \
                const int P = p0_ + i;                                                                 \
                if (P < pb) { const int4 v = ldw4c(w + (P / 2) * 144 + 16 + 64 * (P % 2) + 16 * tg); W[k][i] = make_uint4(v.x, v.y, v.z, v.w); } \
                else W[k][i] = make_uint4(0, 0, 0, 0);                                                 \
            }                                                                                          \
        }                                                                                              \
    }
#define GU_STEP(W, H, IT_, KS_)                                                                        \
    {                                                                                                  \
        const int p0_ = pa + SP * (KS_), hb_ = p0_ / 2;                                                \
        _Pragma("unroll") for (int i = 0; i < SP; i++) {                                               \
            const int P = p0_ + i;                                                                     \
            if (P < pb) {                                                                              \
            const int bk = P / 2;                                                                      \
            uint2 cA[2], cB[2];                                                                        \
            int4 hcur[2];                                                                              \
            _Pragma("unroll") for (int k = 0; k < 2; k++) {                                            \
                const unsigned ox = __shfl_xor_sync(~0u, W[k][i].z, 2), oy = __shfl_xor_sync(~0u, W[k][i].w, 2); \
                const uint2 own = make_uint2(W[k][i].x, W[k][i].y), got = make_uint2(ox, oy);         \
                cA[k] = tg < 2 ? own : got;                                                            \
                cB[k] = tg < 2 ? got : own;                                                            \
                const unsigned hw = bk == hb_ ? H[k][0] : H[k][1];                                     \
                hcur[k] = make_int4(__shfl_sync(~0u, hw, lane & ~3), __shfl_sync(~0u, hw, (lane & ~3) + 1), __shfl_sync(~0u, hw, (lane & ~3) + 2), __shfl_sync(~0u, hw, (lane & ~3) + 3)); \
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
                        const uint2 bf = n < T ? *reinterpret_cast<const uint2 *>(xw + n * XS + 8 * sj + off / 4) : make_uint2(0, 0); \
                        const float4 sc = *reinterpret_cast<const float4 *>(xd + sj * TMAX + 8 * nt + 2 * tg); \
                        int d[4];                                                                      \
                        mma_s8_k32(d, a0, a1, a2, a3, bf);                                             \
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
        if ((KS_) == SPC - 1) { /* item done */                                                        \
            float * slot = red + warp * 16 * T;                                                        \
            _Pragma("unroll") for (int nt = 0; nt < NTL; nt++)                                         \
                _Pragma("unroll") for (int c = 0; c < 4; c++) {                                        \
                    const int t = 8 * nt + 2 * tg + (c & 1);                                           \
                    if (t < T) slot[(g + 8 * (c >> 1)) * T + t] = acc[nt][c];                          \
                    acc[nt][c] = 0.f;                                                                  \
                }                                                                                      \
            asm volatile("bar.sync %0, 128;" ::"r"(1 + grp) : "memory");                               \
            for (int o = threadIdx.x & 127; o < 8 * T; o += 128) {                                     \
                const int row = o / T, t = o % T;                                                      \
                float a = 0.f, u = 0.f;                                                                \
                _Pragma("unroll") for (int w4 = 0; w4 < 4; w4++) {                                     \
                    a += red[((grp * 4 + w4) * 16 + row) * T + t];                                     \
                    u += red[((grp * 4 + w4) * 16 + 8 + row) * T + t];                                 \
                }                                                                                      \
                out[(size_t) t * FF + 8 * (IT_) + row] = a / (1.f + __expf(-a)) * u;                  \
            }                                                                                          \
            asm volatile("bar.sync %0, 128;" ::"r"(1 + grp) : "memory");                               \
        }                                                                                              \
    }
    float acc[NTL][4] = {};
    uint4 WA[2][SP], WB[2][SP];
    unsigned HA[2][2], HB[2][2];
    int L0 = 0, I0 = it0, K0 = 0;
    if (I0 >= NI) return;
    GU_LOAD(WA, HA, L0, I0, K0)
    while (true) {
        int L1 = L0, I1 = I0, K1 = K0;
        GU_NEXT(L1, I1, K1)
        const bool v1 = L1 < NLAY;
        if (v1) GU_LOAD(WB, HB, L1, I1, K1)
        GU_STEP(WA, HA, I0, K0)
        if (!v1) break;
        int L2 = L1, I2 = I1, K2 = K1;
        GU_NEXT(L2, I2, K2)
        const bool v2 = L2 < NLAY;
        if (v2) GU_LOAD(WA, HA, L2, I2, K2)
        GU_STEP(WB, HB, I1, K1)
        if (!v2) break;
        L0 = L2; I0 = I2; K0 = K2;
    }
#undef GU_NEXT
#undef GU_LOAD
#undef GU_STEP
}

// ---- Q6_K (the megakernel's repacked 224-byte block: qlo[128] q4k-style nibbles | qhi[64]: per unit (p, hh) 8 bytes
// (hA: sub-block 2p's 16 values' high 2 bits, hB: 2p + 1's; value i at byte i % 4, bit 2 (i / 4)) | meta[32]: per p
// sc[4p .. 4p + 3] (int8: (2p, hh0), (2p, hh1), (2p+1, hh0), (2p+1, hh1)), d fp16, pad). N x 5120 (attn_qkv shape).
// Q6M: 4-warp groups, 16-row items (rows g, g + 8), warp gw takes pairs [10 gw, 10 gw + 10) in steps of 3, double
// buffered. A lane's 8 bytes of a chunk lie in one 16-value half (hh = tg & 1), so each sub-block is two mma with the
// other half's lanes zeroed; values become signed (q - 32) before the mma, so there is no min term:
//   acc += d * dx * (sc_h0 * D0 + sc_h1 * D1)
constexpr int Q6B = 224, Q6N = 10240, Q6RB = K / 256 * Q6B;
template <int T, int SP = 2>
__global__ void __launch_bounds__(NT, 1) q6_mma(const uint8_t * Wq, float * out, size_t ls) {
    constexpr int NTL = (T + 7) / 8, NI = Q6N / 16, SPC = (10 + SP - 1) / SP;
    extern __shared__ __align__(16) uint8_t sm[];
    setup32(sm, T, true);
    const unsigned * xw = (const unsigned *) sm;
    const float2 * xd = (const float2 *) (xw + TMAX * XS);
    float * red = (float *) (xd + NJ * TMAX);
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, g = lane >> 2, tg = lane & 3, hh = tg & 1;
    const int grp = warp >> 2, gw = warp & 3, pa = 10 * gw, pb = pa + 10;
    const int offA = (tg & 1) * 16 + (tg >> 1) * 8, offB = offA ^ 8;
    const int it0 = 4 * blockIdx.x + grp, istep = 4 * gridDim.x;
#define Q6_NEXT(L_, IT_, KS_) { if (++KS_ == SPC) { KS_ = 0; IT_ += istep; if (IT_ >= NI) { IT_ = it0; L_++; } } }
    // per pair P (chunks 2(P%2), +1 of block P/2), per row k: window (16 bytes), the lane's qhi words for both chunks
    // (unit (p, hh)), both chunks' meta (scales + d)
#define Q6_LOAD(W, QH, MT, L_, IT_, KS_)                                                               \
    {                                                                                                  \
        const int p0_ = pa + SP * (KS_);                                                               \
        _Pragma("unroll") for (int k = 0; k < 2; k++) {                                                \
            const uint8_t * w = Wq + (L_) * ls + (size_t) (16 * (IT_) + g + 8 * k) * Q6RB;             \
            _Pragma("unroll") for (int i = 0; i < SP; i++) {                                           \
                const int P = p0_ + i;                                                                 \
                if (P < pb) {                                                                          \
                    const uint8_t * blk = w + (P / 2) * Q6B;                                           \
                    const int pc = 2 * (P % 2);                                                        \
                    const int4 v = ldw4c(blk + 64 * (P % 2) + 16 * tg);                                \
                    W[k][i] = make_uint4(v.x, v.y, v.z, v.w);                                          \
                    const uint2 h0 = ldw2c(blk + 128 + 8 * (2 * pc + hh)), h1 = ldw2c(blk + 128 + 8 * (2 * pc + 2 + hh)); \
                    QH[k][i] = make_uint4(h0.x, h0.y, h1.x, h1.y);                                     \
                    const int4 mt = ldw4c(blk + 192 + 8 * pc);                                         \
                    MT[k][i] = make_uint4(mt.x, mt.y, mt.z, mt.w);                                     \
                } else { W[k][i] = make_uint4(0, 0, 0, 0); QH[k][i] = W[k][i]; MT[k][i] = W[k][i]; }  \
            }                                                                                          \
        }                                                                                              \
    }
    // QH[k][i]: this lane's unit (pc, hh) [x = hA, y = hB] and (pc + 1, hh) [z, w]
#define Q6_STEP(W, QH, QH2, MT, IT_, KS_)                                                              \
    {                                                                                                  \
        const int p0_ = pa + SP * (KS_);                                                               \
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
                    const uint4 m4 = MT[k][i];                                                         \
                    const unsigned scw = hc ? m4.z : m4.x, dw = hc ? m4.w : m4.y;                      \
                    dd[k] = __half2float(__ushort_as_half((unsigned short) (dw & 0xffff)));            \
                    sc[k][0][0] = (int) (int8_t) (scw); sc[k][0][1] = (int) (int8_t) (scw >> 8);       \
                    sc[k][1][0] = (int) (int8_t) (scw >> 16); sc[k][1][1] = (int) (int8_t) (scw >> 24); \
                    const uint4 q4 = QH[k][i];                                                         \
                    hw[k][0] = hc ? q4.z : q4.x; hw[k][1] = hc ? q4.w : q4.y;                          \
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
                        const uint2 bf = n < T ? *reinterpret_cast<const uint2 *>(xw + n * XS + 8 * sj + off / 4) : make_uint2(0, 0); \
                        const float4 xs = *reinterpret_cast<const float4 *>(xd + sj * TMAX + 8 * nt + 2 * tg); \
                        int d0[4], d1[4];                                                              \
                        mma_s8_k32(d0, a[0] & z0, a[1] & z0, a[2] & z0, a[3] & z0, bf);                \
                        mma_s8_k32(d1, a[0] & z1, a[1] & z1, a[2] & z1, a[3] & z1, bf);                \
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
            float * slot = red + warp * 16 * T;                                                        \
            _Pragma("unroll") for (int nt = 0; nt < NTL; nt++)                                         \
                _Pragma("unroll") for (int c = 0; c < 4; c++) {                                        \
                    const int t = 8 * nt + 2 * tg + (c & 1);                                           \
                    if (t < T) slot[(g + 8 * (c >> 1)) * T + t] = acc[nt][c];                          \
                    acc[nt][c] = 0.f;                                                                  \
                }                                                                                      \
            asm volatile("bar.sync %0, 128;" ::"r"(1 + grp) : "memory");                               \
            for (int o = threadIdx.x & 127; o < 16 * T; o += 128) {                                    \
                const int row = o / T, t = o % T;                                                      \
                float a = 0.f;                                                                         \
                _Pragma("unroll") for (int w4 = 0; w4 < 4; w4++) a += red[((grp * 4 + w4) * 16 + row) * T + t]; \
                out[(size_t) t * Q6N + 16 * (IT_) + row] = a;                                          \
            }                                                                                          \
            asm volatile("bar.sync %0, 128;" ::"r"(1 + grp) : "memory");                               \
        }                                                                                              \
    }
    float acc[NTL][4] = {};
    uint4 WA[2][SP], WB[2][SP], QA[2][SP], QB[2][SP], MA[2][SP], MB[2][SP];
    int L0 = 0, I0 = it0, K0 = 0;
    if (I0 >= NI) return;
    Q6_LOAD(WA, QA, MA, L0, I0, K0)
    while (true) {
        int L1 = L0, I1 = I0, K1 = K0;
        Q6_NEXT(L1, I1, K1)
        const bool v1 = L1 < NLAY;
        if (v1) Q6_LOAD(WB, QB, MB, L1, I1, K1)
        Q6_STEP(WA, QA, QA, MA, I0, K0)
        if (!v1) break;
        int L2 = L1, I2 = I1, K2 = K1;
        Q6_NEXT(L2, I2, K2)
        const bool v2 = L2 < NLAY;
        if (v2) Q6_LOAD(WA, QA, MA, L2, I2, K2)
        Q6_STEP(WB, QB, QB, MB, I1, K1)
        if (!v2) break;
        L0 = L2; I0 = I2; K0 = K2;
    }
#undef Q6_NEXT
#undef Q6_LOAD
#undef Q6_STEP
}

template <int T, int SP = 2>
__global__ void __launch_bounds__(NT, 1) q6_mma2(const uint8_t * Wq, float * out, size_t ls) {
    constexpr int NTL = (T + 7) / 8, NI = Q6N / 16, SPC = (10 + SP - 1) / SP;
    extern __shared__ __align__(16) uint8_t sm[];
    setup32(sm, T, true);
    const unsigned * xw = (const unsigned *) sm;
    const float2 * xd = (const float2 *) (xw + TMAX * XS);
    float * red = (float *) (xd + NJ * TMAX);
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, g = lane >> 2, tg = lane & 3, hh = tg & 1;
    const int grp = warp >> 2, gw = warp & 3, pa = 10 * gw, pb = pa + 10;
    const int offA = (tg & 1) * 16 + (tg >> 1) * 8, offB = offA ^ 8;
    const int it0 = 4 * blockIdx.x + grp, istep = 4 * gridDim.x;
#define Q62_NEXT(L_, IT_, KS_) { if (++KS_ == SPC) { KS_ = 0; IT_ += istep; if (IT_ >= NI) { IT_ = it0; L_++; } } }
    // per pair P (chunks 2(P%2), +1 of block P/2), per row k: window (16 bytes), the lane's qhi words for both chunks
    // (unit (p, hh)), both chunks' meta (scales + d)
#define Q62_LOAD(W, QH, MT, L_, IT_, KS_)                                                               \
    {                                                                                                  \
        const int p0_ = pa + SP * (KS_);                                                               \
        _Pragma("unroll") for (int k = 0; k < 2; k++) {                                                \
            const uint8_t * w = Wq + (L_) * ls + (size_t) (16 * (IT_) + g + 8 * k) * Q6RB;             \
            _Pragma("unroll") for (int i = 0; i < SP; i++) {                                           \
                const int P = p0_ + i;                                                                 \
                if (P < pb) {                                                                          \
                    const uint8_t * blk = w + (P / 2) * Q6B;                                           \
                    const int pc = 2 * (P % 2);                                                        \
                    const int4 v = ldw4c(blk + 64 * (P % 2) + 16 * tg);                                \
                    W[k][i] = make_uint4(v.x, v.y, v.z, v.w);                                          \
                    QH[k][i] = ldw2c(blk + 128 + 8 * (2 * pc + tg));                                      \
                    MT[k][i] = ldw1c(blk + 192 + 8 * pc + 4 * tg); /* word tg of the pair's 16-byte meta */ \
                } else { W[k][i] = make_uint4(0, 0, 0, 0); QH[k][i] = make_uint2(0, 0); MT[k][i] = 0u; }        \
            }                                                                                          \
        }                                                                                              \
    }
    // QH[k][i]: this lane's unit (pc, hh) [x = hA, y = hB] and (pc + 1, hh) [z, w]
#define Q62_STEP(W, QH, QH2, MT, IT_, KS_)                                                              \
    {                                                                                                  \
        const int p0_ = pa + SP * (KS_);                                                               \
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
                    dd[k] = __half2float(__ushort_as_half((unsigned short) (dw & 0xffff)));            \
                    sc[k][0][0] = (int) (int8_t) (scw); sc[k][0][1] = (int) (int8_t) (scw >> 8);       \
                    sc[k][1][0] = (int) (int8_t) (scw >> 16); sc[k][1][1] = (int) (int8_t) (scw >> 24); \
                    hw[k][0] = __shfl_sync(~0u, QH[k][i].x, (lane & ~3) + 2 * hc + hh);                \
                    hw[k][1] = __shfl_sync(~0u, QH[k][i].y, (lane & ~3) + 2 * hc + hh);                \
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
                        const uint2 bf = n < T ? *reinterpret_cast<const uint2 *>(xw + n * XS + 8 * sj + off / 4) : make_uint2(0, 0); \
                        const float4 xs = *reinterpret_cast<const float4 *>(xd + sj * TMAX + 8 * nt + 2 * tg); \
                        int d0[4], d1[4];                                                              \
                        mma_s8_k32(d0, a[0] & z0, a[1] & z0, a[2] & z0, a[3] & z0, bf);                \
                        mma_s8_k32(d1, a[0] & z1, a[1] & z1, a[2] & z1, a[3] & z1, bf);                \
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
            float * slot = red + warp * 16 * T;                                                        \
            _Pragma("unroll") for (int nt = 0; nt < NTL; nt++)                                         \
                _Pragma("unroll") for (int c = 0; c < 4; c++) {                                        \
                    const int t = 8 * nt + 2 * tg + (c & 1);                                           \
                    if (t < T) slot[(g + 8 * (c >> 1)) * T + t] = acc[nt][c];                          \
                    acc[nt][c] = 0.f;                                                                  \
                }                                                                                      \
            asm volatile("bar.sync %0, 128;" ::"r"(1 + grp) : "memory");                               \
            for (int o = threadIdx.x & 127; o < 16 * T; o += 128) {                                    \
                const int row = o / T, t = o % T;                                                      \
                float a = 0.f;                                                                         \
                _Pragma("unroll") for (int w4 = 0; w4 < 4; w4++) a += red[((grp * 4 + w4) * 16 + row) * T + t]; \
                out[(size_t) t * Q6N + 16 * (IT_) + row] = a;                                          \
            }                                                                                          \
            asm volatile("bar.sync %0, 128;" ::"r"(1 + grp) : "memory");                               \
        }                                                                                              \
    }
    float acc[NTL][4] = {};
    uint4 WA[2][SP], WB[2][SP];
    uint2 QA[2][SP], QB[2][SP];
    unsigned MA[2][SP], MB[2][SP];
    int L0 = 0, I0 = it0, K0 = 0;
    if (I0 >= NI) return;
    Q62_LOAD(WA, QA, MA, L0, I0, K0)
    while (true) {
        int L1 = L0, I1 = I0, K1 = K0;
        Q62_NEXT(L1, I1, K1)
        const bool v1 = L1 < NLAY;
        if (v1) Q62_LOAD(WB, QB, MB, L1, I1, K1)
        Q62_STEP(WA, QA, QA, MA, I0, K0)
        if (!v1) break;
        int L2 = L1, I2 = I1, K2 = K1;
        Q62_NEXT(L2, I2, K2)
        const bool v2 = L2 < NLAY;
        if (v2) Q62_LOAD(WA, QA, MA, L2, I2, K2)
        Q62_STEP(WB, QB, QB, MB, I1, K1)
        if (!v2) break;
        L0 = L2; I0 = I2; K0 = K2;
    }
#undef Q62_NEXT
#undef Q62_LOAD
#undef Q62_STEP
}

// Q6 float reference (last layer) on the per-32-quantised activations of setup32 (natural): exact integer math
__global__ void ref_q6(const uint8_t * Wq, float * out, size_t ls, int T) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= T * Q6N) return;
    const int t = i / Q6N, row = i % Q6N;
    const uint8_t * w = Wq + (NLAY - 1) * ls + (size_t) row * Q6RB;
    double acc = 0;
    for (int b = 0; b < K / 256; b++) {
        const uint8_t * blk = w + b * Q6B;
        for (int p = 0; p < 4; p++) {
            unsigned short dbits; memcpy(&dbits, blk + 192 + 8 * p + 4, 2);
            const float d = __half2float(__ushort_as_half(dbits));
            for (int nib = 0; nib < 2; nib++) {
                const int sj = 8 * b + 2 * p + nib;
                const float dx = xdx(t, sj);
                for (int v = 0; v < 32; v++) {
                    const int h = v / 16, iv = v % 16;
                    const int lo = (blk[32 * p + v] >> (4 * nib)) & 15;
                    unsigned hwd; memcpy(&hwd, blk + 128 + 8 * (2 * p + h) + 4 * nib, 4);
                    const int hi = (hwd >> (8 * (iv % 4) + 2 * (iv / 4))) & 3;
                    const int q = (lo | (hi << 4)) - 32;
                    const int sc = (int8_t) blk[192 + 8 * p + 2 * nib + h];
                    acc += (double) d * sc * q * dx * xval(t, 32 * sj + v);
                }
            }
        }
    }
    out[(size_t) t * Q6N + row] = (float) acc;
}
// GU4: GU3 with two reduction slot sets (alternating per item): one named barrier per item.
// (base) GU3: gate/up with MD3's pipeline: 4-warp groups, 4 items (8 gate + 8 up rows) per block at a time; in an item
// warp gw takes pairs [10 gw, 10 gw + 10) in steps of SP pairs, double-buffered across steps and items (header
// words, one accumulator); per item the group's 4 warps combine behind a named barrier.
template <int T, int SP = 3>
__global__ void __launch_bounds__(NT, 1) gu_mma4(const uint8_t * Wg, const uint8_t * Wu, float * out, size_t ls) {
    constexpr int NTL = (T + 7) / 8, NI = FF / 8, SPC = (10 + SP - 1) / SP;
    extern __shared__ __align__(16) uint8_t sm[];
    setup32(sm, T, true);
    const unsigned * xw = (const unsigned *) sm;
    const float2 * xd = (const float2 *) (xw + TMAX * XS);
    float * red0 = (float *) (xd + NJ * TMAX); // [2 sets][16 warps][16][T]
    int rs = 0;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, g = lane >> 2, tg = lane & 3;
    const int grp = warp >> 2, gw = warp & 3, pa = 10 * gw, pb = pa + 10;
    const int offA = (tg & 1) * 16 + (tg >> 1) * 8, offB = offA ^ 8;
    const int it0 = 4 * blockIdx.x + grp, istep = 4 * gridDim.x;
    // step descriptor: (L, it, ks); next(): ks + 1, then the next item, then the next layer
#define GU4_NEXT(L_, IT_, KS_) { if (++KS_ == SPC) { KS_ = 0; IT_ += istep; if (IT_ >= NI) { IT_ = it0; L_++; } } }
#define GU4_LOAD(W, H, L_, IT_, KS_)                                                                    \
    {                                                                                                  \
        const int p0_ = pa + SP * (KS_), hb_ = p0_ / 2;                                                \
        _Pragma("unroll") for (int k = 0; k < 2; k++) {                                                \
            const uint8_t * w = (k ? Wu : Wg) + (L_) * ls + (size_t) (8 * (IT_) + g) * RB;             \
            _Pragma("unroll") for (int b = 0; b < 2; b++) H[k][b] = (hb_ + b) < K / 256 ? ldw1c(w + (hb_ + b) * 144 + 4 * tg) : 0u; \
            _Pragma("unroll") for (int i = 0; i < SP; i++) {                                           \
                const int P = p0_ + i;                                                                 \
                if (P < pb) { const int4 v = ldw4c(w + (P / 2) * 144 + 16 + 64 * (P % 2) + 16 * tg); W[k][i] = make_uint4(v.x, v.y, v.z, v.w); } \
                else W[k][i] = make_uint4(0, 0, 0, 0);                                                 \
            }                                                                                          \
        }                                                                                              \
    }
#define GU4_STEP(W, H, IT_, KS_)                                                                        \
    {                                                                                                  \
        const int p0_ = pa + SP * (KS_), hb_ = p0_ / 2;                                                \
        _Pragma("unroll") for (int i = 0; i < SP; i++) {                                               \
            const int P = p0_ + i;                                                                     \
            if (P < pb) {                                                                              \
            const int bk = P / 2;                                                                      \
            uint2 cA[2], cB[2];                                                                        \
            int4 hcur[2];                                                                              \
            _Pragma("unroll") for (int k = 0; k < 2; k++) {                                            \
                const unsigned ox = __shfl_xor_sync(~0u, W[k][i].z, 2), oy = __shfl_xor_sync(~0u, W[k][i].w, 2); \
                const uint2 own = make_uint2(W[k][i].x, W[k][i].y), got = make_uint2(ox, oy);         \
                cA[k] = tg < 2 ? own : got;                                                            \
                cB[k] = tg < 2 ? got : own;                                                            \
                const unsigned hw = bk == hb_ ? H[k][0] : H[k][1];                                     \
                hcur[k] = make_int4(__shfl_sync(~0u, hw, lane & ~3), __shfl_sync(~0u, hw, (lane & ~3) + 1), __shfl_sync(~0u, hw, (lane & ~3) + 2), __shfl_sync(~0u, hw, (lane & ~3) + 3)); \
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
                        const uint2 bf = n < T ? *reinterpret_cast<const uint2 *>(xw + n * XS + 8 * sj + off / 4) : make_uint2(0, 0); \
                        const float4 sc = *reinterpret_cast<const float4 *>(xd + sj * TMAX + 8 * nt + 2 * tg); \
                        int d[4];                                                                      \
                        mma_s8_k32(d, a0, a1, a2, a3, bf);                                             \
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
        if ((KS_) == SPC - 1) { /* item done */                                                        \
            float * red = red0 + rs * 16 * 16 * T;                                                     \
            rs ^= 1;                                                                                   \
            float * slot = red + warp * 16 * T;                                                        \
            _Pragma("unroll") for (int nt = 0; nt < NTL; nt++)                                         \
                _Pragma("unroll") for (int c = 0; c < 4; c++) {                                        \
                    const int t = 8 * nt + 2 * tg + (c & 1);                                           \
                    if (t < T) slot[(g + 8 * (c >> 1)) * T + t] = acc[nt][c];                          \
                    acc[nt][c] = 0.f;                                                                  \
                }                                                                                      \
            asm volatile("bar.sync %0, 128;" ::"r"(1 + grp) : "memory");                               \
            for (int o = threadIdx.x & 127; o < 8 * T; o += 128) {                                     \
                const int row = o / T, t = o % T;                                                      \
                float a = 0.f, u = 0.f;                                                                \
                _Pragma("unroll") for (int w4 = 0; w4 < 4; w4++) {                                     \
                    a += red[((grp * 4 + w4) * 16 + row) * T + t];                                     \
                    u += red[((grp * 4 + w4) * 16 + 8 + row) * T + t];                                 \
                }                                                                                      \
                out[(size_t) t * FF + 8 * (IT_) + row] = a / (1.f + __expf(-a)) * u;                  \
            }                                                                                          \
        }                                                                                              \
    }
    float acc[NTL][4] = {};
    uint4 WA[2][SP], WB[2][SP];
    unsigned HA[2][2], HB[2][2];
    int L0 = 0, I0 = it0, K0 = 0;
    if (I0 >= NI) return;
    GU4_LOAD(WA, HA, L0, I0, K0)
    while (true) {
        int L1 = L0, I1 = I0, K1 = K0;
        GU4_NEXT(L1, I1, K1)
        const bool v1 = L1 < NLAY;
        if (v1) GU4_LOAD(WB, HB, L1, I1, K1)
        GU4_STEP(WA, HA, I0, K0)
        if (!v1) break;
        int L2 = L1, I2 = I1, K2 = K1;
        GU4_NEXT(L2, I2, K2)
        const bool v2 = L2 < NLAY;
        if (v2) GU4_LOAD(WA, HA, L2, I2, K2)
        GU4_STEP(WB, HB, I1, K1)
        if (!v2) break;
        L0 = L2; I0 = I2; K0 = K2;
    }
#undef GU4_NEXT
#undef GU4_LOAD
#undef GU4_STEP
}


// Q4M: generic Q4_K GEMV for T rows on the tensor cores, structured as Q6M2: 4-warp groups, 16-row items (rows g,
// g + 8 of one matrix), warp gw takes pairs [10 gw, 10 gw + 10) in steps of SP, double-buffered; header words shared
// in the quad (lane tg loads word tg, shuffled together per pair). Gate/up as one 2 FF-row matrix (silu(g) * u then
// belongs to the consumer).
template <int T, int SP = 2>
__global__ void __launch_bounds__(NT, 1) q4_mma(const uint8_t * Wm, int nrows, float * out, size_t ls) {
    constexpr int NTL = (T + 7) / 8, SPC = (10 + SP - 1) / SP;
    const int NI = nrows / 16;
    extern __shared__ __align__(16) uint8_t sm[];
    setup32(sm, T, true);
    const unsigned * xw = (const unsigned *) sm;
    const float2 * xd = (const float2 *) (xw + TMAX * XS);
    float * red = (float *) (xd + NJ * TMAX);
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, g = lane >> 2, tg = lane & 3, qb = lane & ~3;
    const int grp = warp >> 2, gw = warp & 3, pa = 10 * gw, pb = pa + 10;
    const int offA = (tg & 1) * 16 + (tg >> 1) * 8, offB = offA ^ 8;
    const int it0 = 4 * blockIdx.x + grp, istep = 4 * gridDim.x;
#define Q4_NEXT(L_, IT_, KS_) { if (++KS_ == SPC) { KS_ = 0; IT_ += istep; if (IT_ >= NI) { IT_ = it0; L_++; } } }
#define Q4_LOAD(W, H, L_, IT_, KS_)                                                                    \
    {                                                                                                  \
        const int p0_ = pa + SP * (KS_), hb_ = p0_ / 2;                                                \
        _Pragma("unroll") for (int k = 0; k < 2; k++) {                                                \
            const uint8_t * w = Wm + (L_) * ls + (size_t) (16 * (IT_) + g + 8 * k) * RB;               \
            _Pragma("unroll") for (int b = 0; b < (SP + 3) / 2; b++) H[k][b] = (hb_ + b) < K / 256 ? ldw1c(w + (hb_ + b) * 144 + 4 * tg) : 0u; \
            _Pragma("unroll") for (int i = 0; i < SP; i++) {                                           \
                const int P = p0_ + i;                                                                 \
                if (P < pb) { const int4 v = ldw4c(w + (P / 2) * 144 + 16 + 64 * (P % 2) + 16 * tg); W[k][i] = make_uint4(v.x, v.y, v.z, v.w); } \
                else W[k][i] = make_uint4(0, 0, 0, 0);                                                 \
            }                                                                                          \
        }                                                                                              \
    }
#define Q4_STEP(W, H, IT_, KS_)                                                                        \
    {                                                                                                  \
        const int p0_ = pa + SP * (KS_), hb_ = p0_ / 2;                                                \
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
                unsigned hw = H[k][0];                                                                 \
                _Pragma("unroll") for (int b = 1; b < (SP + 3) / 2; b++) hw = bi == b ? H[k][b] : hw;  \
                hcur[k] = make_int4(__shfl_sync(~0u, hw, qb), __shfl_sync(~0u, hw, qb + 1), __shfl_sync(~0u, hw, qb + 2), __shfl_sync(~0u, hw, qb + 3)); \
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
                        const uint2 bf = n < T ? *reinterpret_cast<const uint2 *>(xw + n * XS + 8 * sj + off / 4) : make_uint2(0, 0); \
                        const float4 sc = *reinterpret_cast<const float4 *>(xd + sj * TMAX + 8 * nt + 2 * tg); \
                        int d[4];                                                                      \
                        mma_s8_k32(d, a0, a1, a2, a3, bf);                                             \
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
            float * slot = red + warp * 16 * T;                                                        \
            _Pragma("unroll") for (int nt = 0; nt < NTL; nt++)                                         \
                _Pragma("unroll") for (int c = 0; c < 4; c++) {                                        \
                    const int t = 8 * nt + 2 * tg + (c & 1);                                           \
                    if (t < T) slot[(g + 8 * (c >> 1)) * T + t] = acc[nt][c];                          \
                    acc[nt][c] = 0.f;                                                                  \
                }                                                                                      \
            asm volatile("bar.sync %0, 128;" ::"r"(1 + grp) : "memory");                               \
            for (int o = threadIdx.x & 127; o < 16 * T; o += 128) {                                    \
                const int row = o / T, t = o % T;                                                      \
                float a = 0.f;                                                                         \
                _Pragma("unroll") for (int w4 = 0; w4 < 4; w4++) a += red[((grp * 4 + w4) * 16 + row) * T + t]; \
                out[(size_t) t * nrows + 16 * (IT_) + row] = a;                                        \
            }                                                                                          \
            asm volatile("bar.sync %0, 128;" ::"r"(1 + grp) : "memory");                               \
        }                                                                                              \
    }
    float acc[NTL][4] = {};
    uint4 WA[2][SP], WB[2][SP];
    unsigned HA[2][(SP + 3) / 2], HB[2][(SP + 3) / 2];
    int L0 = 0, I0 = it0, K0 = 0;
    if (I0 >= NI) return;
    Q4_LOAD(WA, HA, L0, I0, K0)
    while (true) {
        int L1 = L0, I1 = I0, K1 = K0;
        Q4_NEXT(L1, I1, K1)
        const bool v1 = L1 < NLAY;
        if (v1) Q4_LOAD(WB, HB, L1, I1, K1)
        Q4_STEP(WA, HA, I0, K0)
        if (!v1) break;
        int L2 = L1, I2 = I1, K2 = K1;
        Q4_NEXT(L2, I2, K2)
        const bool v2 = L2 < NLAY;
        if (v2) Q4_LOAD(WA, HA, L2, I2, K2)
        Q4_STEP(WB, HB, I1, K1)
        if (!v2) break;
        L0 = L2; I0 = I2; K0 = K2;
    }
#undef Q4_NEXT
#undef Q4_LOAD
#undef Q4_STEP
}
// float reference of M32 (last layer): dequantised weights . dequantised per-32 activations
__global__ void ref32(const uint8_t * Wg, const uint8_t * Wu, float * out, size_t ls, int T) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= T * FF) return;
    const int t = i / FF, row = i % FF;
    float a[2];
    for (int m = 0; m < 2; m++) {
        const uint8_t * w = (m ? Wu : Wg) + (NLAY - 1) * ls + (size_t) row * RB;
        float acc = 0.f;
        for (int b = 0; b < K / 256; b++) {
            const int4 h = *reinterpret_cast<const int4 *>(w + b * 144);
            for (int p = 0; p < 4; p++) {
                float fA, gA, fB, gB;
                q4k_scales(h, p, fA, gA, fB, gB);
                for (int half = 0; half < 2; half++) {
                    const int sj = 8 * b + 2 * p + half;
                    float s = 0.f, sx = 0.f;
                    for (int k = 0; k < 32; k++) {
                        const int q = (w[b * 144 + 16 + 32 * p + k] >> (4 * half)) & 15;
                        const float x = xdx(t, sj) * (float) xval(t, 32 * sj + k);
                        s += (float) q * x;
                        sx += x;
                    }
                    acc += (half ? fB : fA) * s - (half ? gB : gA) * sx;
                }
            }
        }
        a[m] = acc;
    }
    out[(size_t) t * FF + row] = a[0] / (1.f + __expf(-a[0])) * a[1];
}
template <class F> float timeit(F f, int reps = 20) {
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b); f(); CK(cudaDeviceSynchronize());
    float best = 1e30f;
    for (int i = 0; i < reps; i++) {
        cudaEventRecord(a); f(); cudaEventRecord(b); CK(cudaEventSynchronize(b));
        float ms; cudaEventElapsedTime(&ms, a, b); best = ms < best ? ms : best;
    }
    return best;
}
__global__ void stream_read(const int4 * p, size_t n, int * out) {
    int acc = 0;
    for (size_t i = blockIdx.x * (size_t) blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x * blockDim.x) { int4 v = ldw4(p + i); acc ^= v.x ^ v.y ^ v.z ^ v.w; }
    if (acc == 0x12345678) out[0] = acc;
}
int main(int argc, char ** argv) {
    const char * only = argc > 1 ? argv[1] : "";
    const size_t mat = (size_t) FF * RB, bytes = 2 * NLAY * mat;
    std::vector<uint8_t> h(bytes);
    for (size_t i = 0; i < bytes; i++) h[i] = (uint8_t) (i * 2654435761u >> 13);
    for (size_t r = 0; r < bytes / 144; r++) { __half d = __float2half(0.001f); memcpy(&h[r * 144], &d, 2); memcpy(&h[r * 144 + 2], &d, 2); }
    uint8_t * dW; float * dout; int * dummy;
    CK(cudaMalloc(&dW, bytes)); CK(cudaMalloc(&dout, (size_t) TMAX * FF * 4)); CK(cudaMalloc(&dummy, 4));
    CK(cudaMemcpy(dW, h.data(), bytes, cudaMemcpyHostToDevice));
    int nsm; cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, 0);
    const double gb = bytes / 1e9;
    const int smem = (int) (TMAX * XS * 4 + NJ * TMAX * 8 + 16 * 16 * TMAX * 4);
    const int smem8 = (int) (TMAX * XS * 4 + NJ * TMAX * 8 + 2 * 16 * 16 * 8 * 4);
    std::vector<float> ref;
    auto run = [&](auto k, const char * name, bool keep = false, int sm_ = 0) {
        if (*only && !strstr(name, only)) return;
        const int sz = sm_ ? sm_ : smem;
        CK(cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, sz));
        const float ms = timeit([&] { k<<<nsm, NT, sz>>>(dW, dW + NLAY * mat, dout, mat); });
        CK(cudaGetLastError());
        std::vector<float> o((size_t) TMAX * FF);
        CK(cudaMemcpy(o.data(), dout, o.size() * 4, cudaMemcpyDeviceToHost));
        double md = 0, mx = 0;
        if (!keep && !ref.empty()) for (size_t i = 0; i < o.size(); i++) { md = fmax(md, fabs(o[i] - ref[i])); mx = fmax(mx, fabs(ref[i])); }
        if (keep) ref = o;
        printf("%-22s %7.1f us/layer  %6.1f GB/s%s", name, ms / NLAY * 1e3, gb / ms * 1e3, keep || ref.empty() ? "\n" : "");
        if (!keep && !ref.empty()) printf("   max |diff| %.3g of %.3g\n", md, mx);
        CK(cudaMemset(dout, 0, (size_t) TMAX * FF * 4));
    };
    if (!strcmp(only, "q6")) {
        const size_t qmat = (size_t) Q6N * Q6RB, qbytes = NLAY * qmat;
        std::vector<uint8_t> hq(qbytes);
        for (size_t i = 0; i < qbytes; i++) hq[i] = (uint8_t) (i * 3266489917u >> 13);
        for (size_t b = 0; b < qbytes / Q6B; b++)
            for (int p = 0; p < 4; p++) { __half d = __float2half(0.002f); memcpy(&hq[b * Q6B + 192 + 8 * p + 4], &d, 2); }
        uint8_t * dWq; float * dref;
        CK(cudaMalloc(&dWq, qbytes)); CK(cudaMemcpy(dWq, hq.data(), qbytes, cudaMemcpyHostToDevice));
        CK(cudaMalloc(&dref, (size_t) TMAX * Q6N * 4));
        float * dq; CK(cudaMalloc(&dq, (size_t) TMAX * Q6N * 4));
        const double qgb = qbytes / 1e9;
        const float ms0 = timeit([&] { stream_read<<<nsm * 4, 512>>>((const int4 *) dWq, qbytes / 16, dummy); });
        printf("q6 stream read         %7.1f us/layer  %6.1f GB/s\n", ms0 / NLAY * 1e3, qgb / ms0 * 1e3);
        auto go = [&](auto k, int T, const char * name) {
            CK(cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
            const float ms = timeit([&] { k<<<nsm, NT, smem>>>(dWq, dq, qmat); });
            CK(cudaGetLastError());
            ref_q6<<<(T * Q6N + 255) / 256, 256>>>(dWq, dref, qmat, T);
            CK(cudaDeviceSynchronize());
            std::vector<float> a((size_t) T * Q6N), r(a.size());
            CK(cudaMemcpy(a.data(), dq, a.size() * 4, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(r.data(), dref, r.size() * 4, cudaMemcpyDeviceToHost));
            double se = 0, sr = 0; for (size_t i = 0; i < a.size(); i++) { se += (a[i] - r[i]) * (a[i] - r[i]); sr += r[i] * r[i]; }
            printf("%-22s %7.1f us/layer  %6.1f GB/s   rel RMS err vs exact-int %.2e\n", name, ms / NLAY * 1e3, qgb / ms * 1e3, sqrt(se / sr));
        };
        go(q6_mma<8>, 8, "q6 mma SP2 T=8"); go(q6_mma<12>, 12, "q6 mma SP2 T=12");
        go(q6_mma<8, 1>, 8, "q6 mma SP1 T=8"); go(q6_mma<12, 1>, 12, "q6 mma SP1 T=12");
        go(q6_mma2<8>, 8, "q6 mma2 SP2 T=8"); go(q6_mma2<12>, 12, "q6 mma2 SP2 T=12"); go(q6_mma2<8, 3>, 8, "q6 mma2 SP3 T=8");
        return 0;
    }
    if (!strcmp(only, "down")) {
        const size_t dmat = (size_t) DN * DRB, dbytes = NLAY * dmat;
        std::vector<uint8_t> hd(dbytes);
        for (size_t i = 0; i < dbytes; i++) hd[i] = (uint8_t) (i * 2246822519u >> 13);
        for (size_t r = 0; r < dbytes / 144; r++) { __half d = __float2half(0.001f); memcpy(&hd[r * 144], &d, 2); memcpy(&hd[r * 144 + 2], &d, 2); }
        uint8_t * dWd; unsigned * gxw; float2 * gxd; float * dref;
        CK(cudaMalloc(&dWd, dbytes)); CK(cudaMemcpy(dWd, hd.data(), dbytes, cudaMemcpyHostToDevice));
        CK(cudaMalloc(&gxw, (size_t) TMAX * DK)); CK(cudaMalloc(&gxd, (size_t) DNJ * TMAX * 8)); CK(cudaMalloc(&dref, (size_t) TMAX * DN * 4));
        const double dgb = dbytes / 1e9;
        const int dsm = TMAX * DXS * 4 + DCB * 8 * TMAX * 8 + 16 * 16 * TMAX * 4;
        auto go = [&](auto k, int T, const char * name, bool check) {
            prep_down_act<<<(T * DNJ + 255) / 256, 256>>>(gxw, gxd, T);
            CK(cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, dsm));
            const float ms = timeit([&] { k<<<nsm, NT, dsm>>>(dWd, gxw, gxd, dout, dmat); });
            CK(cudaGetLastError());
            printf("%-22s %7.1f us/layer  %6.1f GB/s", name, ms / NLAY * 1e3, dgb / ms * 1e3);
            if (check) {
                ref_down<<<(T * DN + 255) / 256, 256>>>(dWd, dref, dmat, T);
                CK(cudaDeviceSynchronize());
                std::vector<float> a((size_t) T * DN), r(a.size());
                CK(cudaMemcpy(a.data(), dout, a.size() * 4, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(r.data(), dref, r.size() * 4, cudaMemcpyDeviceToHost));
                double se = 0, sr = 0; for (size_t i = 0; i < a.size(); i++) { se += (a[i] - r[i]) * (a[i] - r[i]); sr += r[i] * r[i]; }
                printf("   rel RMS err vs exact %.3e", sqrt(se / sr));
            }
            printf("\n");
        };
        const float ms = timeit([&] { stream_read<<<nsm * 4, 512>>>((const int4 *) dWd, dbytes / 16, dummy); });
        printf("down stream read       %7.1f us/layer  %6.1f GB/s\n", ms / NLAY * 1e3, dgb / ms * 1e3);
        go(down_dp4a<1>, 1, "down dp4a T=1", true); go(down_dp4a<8>, 8, "down dp4a T=8", true);
        go(down_mma<8>, 8, "down mma T=8", true); go(down_mma<12>, 12, "down mma T=12", true);
        go(down_mma3<8>, 8, "down mma3 T=8", true); go(down_mma3<8, 2>, 8, "down mma3 SP2 T=8", true);
        go(down_mma3<12, 2>, 12, "down mma3 SP2 T=12", true); go(down_mma3<12, 3>, 12, "down mma3 T=12", true);
        { // Q6_K down
            const size_t qmat = (size_t) DN * DRB6, qbytes = NLAY * qmat;
            std::vector<uint8_t> hq(qbytes);
            for (size_t i = 0; i < qbytes; i++) hq[i] = (uint8_t) (i * 3266489917u >> 13);
            for (size_t b = 0; b < qbytes / Q6BD; b++)
                for (int p = 0; p < 4; p++) { __half d = __float2half(0.002f); memcpy(&hq[b * Q6BD + 192 + 8 * p + 4], &d, 2); }
            uint8_t * dWq; CK(cudaMalloc(&dWq, qbytes)); CK(cudaMemcpy(dWq, hq.data(), qbytes, cudaMemcpyHostToDevice));
            const double qgb = qbytes / 1e9;
            const float ms0 = timeit([&] { stream_read<<<nsm * 4, 512>>>((const int4 *) dWq, qbytes / 16, dummy); });
            printf("down q6 stream read    %7.1f us/layer  %6.1f GB/s\n", ms0 / NLAY * 1e3, qgb / ms0 * 1e3);
            auto goq = [&](auto k, int T, const char * name) {
                prep_down_act<<<(T * DNJ + 255) / 256, 256>>>(gxw, gxd, T);
                CK(cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, dsm));
                const float ms = timeit([&] { k<<<nsm, NT, dsm>>>(dWq, gxw, gxd, dout, qmat); });
                CK(cudaGetLastError());
                ref_down_q6<<<(T * DN + 255) / 256, 256>>>(dWq, gxw, gxd, dref, qmat, T);
                CK(cudaDeviceSynchronize());
                std::vector<float> a((size_t) T * DN), r(a.size());
                CK(cudaMemcpy(a.data(), dout, a.size() * 4, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(r.data(), dref, r.size() * 4, cudaMemcpyDeviceToHost));
                double se = 0, sr = 0; for (size_t i = 0; i < a.size(); i++) { se += (a[i] - r[i]) * (a[i] - r[i]); sr += r[i] * r[i]; }
                printf("%-22s %7.1f us/layer  %6.1f GB/s   rel RMS err vs exact-int %.2e\n", name, ms / NLAY * 1e3, qgb / ms * 1e3, sqrt(se / sr));
            };
            goq(down_q6<8>, 8, "down q6 T=8"); goq(down_q6<12>, 12, "down q6 T=12"); goq(down_q6<12, 1>, 12, "down q6 SP1 T=12"); goq(down_q6<8, 1>, 8, "down q6 SP1 T=8");
        }
        return 0;
    }
    const float ms = timeit([&] { stream_read<<<nsm * 4, 512>>>((const int4 *) dW, bytes / 16, dummy); });
    printf("stream read            %7.1f us/layer  %6.1f GB/s\n", ms / NLAY * 1e3, gb / ms * 1e3);
    for (int T : {8, 12}) { // M32 against its float reference
        auto kk = T == 8 ? gu_mma4<8> : gu_mma3<12>;
        const int csm = T == 8 ? smem8 : smem;
        CK(cudaFuncSetAttribute(kk, cudaFuncAttributeMaxDynamicSharedMemorySize, csm));
        kk<<<nsm, NT, csm>>>(dW, dW + NLAY * mat, dout, mat);
        CK(cudaDeviceSynchronize());
        float * dref; CK(cudaMalloc(&dref, (size_t) T * FF * 4));
        ref32<<<(T * FF + 255) / 256, 256>>>(dW, dW + NLAY * mat, dref, mat, T);
        CK(cudaDeviceSynchronize());
        std::vector<float> a((size_t) T * FF), r(a.size());
        CK(cudaMemcpy(a.data(), dout, a.size() * 4, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(r.data(), dref, r.size() * 4, cudaMemcpyDeviceToHost));
        double md = 0, mx = 0; for (size_t i = 0; i < a.size(); i++) { md = fmax(md, fabs(a[i] - r[i]) / (fabs(r[i]) + 1e-3 * 0)); mx = fmax(mx, fabs(r[i])); }
        double mabs = 0; for (size_t i = 0; i < a.size(); i++) mabs = fmax(mabs, fabs(a[i] - r[i]));
        printf("gu_mma4 T=%d vs float reference: max |diff| %.3g (max |ref| %.3g)\n", T, mabs, mx);
        CK(cudaFree(dref));
    }
    for (int T : {8, 12}) { // per-32 vs per-256 activations against the exact float result
        float * dref; CK(cudaMalloc(&dref, (size_t) T * FF * 4));
        ref_exact<<<(T * FF + 255) / 256, 256>>>(dW, dW + NLAY * mat, dref, mat, T);
        CK(cudaDeviceSynchronize());
        std::vector<float> r((size_t) T * FF), a(r.size());
        CK(cudaMemcpy(r.data(), dref, r.size() * 4, cudaMemcpyDeviceToHost));
        auto err = [&](auto kk, const char * name) {
            CK(cudaFuncSetAttribute(kk, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
            kk<<<nsm, NT, smem>>>(dW, dW + NLAY * mat, dout, mat);
            CK(cudaDeviceSynchronize());
            CK(cudaMemcpy(a.data(), dout, a.size() * 4, cudaMemcpyDeviceToHost));
            double se = 0, sr = 0, mx = 0;
            for (size_t i = 0; i < a.size(); i++) { se += (a[i] - r[i]) * (a[i] - r[i]); sr += r[i] * r[i]; mx = fmax(mx, fabs(a[i] - r[i])); }
            printf("%-14s T=%2d vs exact: rel RMS err %.3e, max |err| %.3g (rms ref %.3g)\n", name, T, sqrt(se / sr), mx, sqrt(sr / a.size()));
        };
        if (T == 8) { err(ffn_mma32k8<8, false, true>, "per-32 (M32K8)"); err(ffn_mma256<8>, "per-256 (M256)"); }
        else { err(ffn_mma32k8<12, false, true>, "per-32 (M32K8)"); err(ffn_mma256<12>, "per-256 (M256)"); }
        CK(cudaFree(dref));
    }
    run(load32k<false>, "load32k (no math)"); run(load32k<true>, "load32k L1 (no math)"); run(load32k_64<false>, "load32k 64-bit (no math)"); run(load32k_64<true>, "load32k 64-bit L1 (no math)"); run(load32k_128<false>, "quad 16B (no math)"); run(load32k_128<true>, "quad 16B L1 (no math)"); run(load32k_v4, "load rows v4 (no math)");
    run(ffn_dp4a<1>, "dp4a T=1");
    run(ffn_dp4a<8>, "dp4a T=8", true); run(ffn_mma<8, 1>, "mma T=8 PB=1"); run(ffn_mma<8, 2>, "mma T=8 PB=2");
    run(ffn_mma32<8>, "mma32 T=8"); run(ffn_mma32k<8>, "mma32k T=8"); run(ffn_mma32k<8, true>, "mma32k L1 T=8"); run(ffn_mma32k2<8>, "mma32k2 T=8"); run(ffn_mma32k2<8, true>, "mma32k2 compute-only T=8"); run(ffn_mma32k5<8>, "mma32k5 T=8"); run(ffn_mma32k6<8>, "mma32k6 T=8"); run(ffn_mma32k7<8>, "mma32k7 T=8"); run(ffn_mma32k7<8, false>, "mma32k7 1buf T=8"); run(ffn_mma32k8<8>, "mma32k8 T=8"); run(gu_mma3<8>, "gu3 T=8"); run(gu_mma4<8>, "gu4 T=8", false, smem8);
    if (!*only || strstr("q4m", only)) {
        auto qrun = [&](auto k, const char * name) {
            CK(cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
            const float ms = timeit([&] { k<<<nsm, NT, smem>>>(dW, 2 * FF, dout, 2 * mat); });
            CK(cudaGetLastError());
            printf("%-22s %7.1f us/layer  %6.1f GB/s\n", name, ms / NLAY * 1e3, gb / ms * 1e3);
        };
        qrun(q4_mma<8>, "q4m (gate+up) T=8"); qrun(q4_mma<12>, "q4m (gate+up) T=12"); qrun(q4_mma<8, 3>, "q4m SP3 T=8");
    }
    run(ffn_dp4a<12>, "dp4a T=12", true); run(ffn_mma<12, 1>, "mma T=12 PB=1");
    run(ffn_mma32<12>, "mma32 T=12"); run(ffn_mma32k<12>, "mma32k T=12"); run(ffn_mma32k<12, true>, "mma32k L1 T=12"); run(ffn_mma32k2<12>, "mma32k2 T=12"); run(ffn_mma32k2<12, true>, "mma32k2 compute-only T=12"); run(ffn_mma32k5<12>, "mma32k5 T=12"); run(ffn_mma32k6<12>, "mma32k6 T=12"); run(ffn_mma32k7<12>, "mma32k7 T=12"); run(ffn_mma32k7<12, false>, "mma32k7 1buf T=12"); run(ffn_mma32k8<12>, "mma32k8 T=12"); run(gu_mma3<12>, "gu3 T=12");
}
