// Microbenchmark for Q4_K GEMV variants (T tokens, int8 activations) on a ~1 GB matrix.
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cstdint>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1);} } while (0)

constexpr int K = 5120, U = K / 32, NB = K / 256, RB = NB * 144;
constexpr int NT = 512, NW = 16;
constexpr int M4 = 0x0F0F0F0F;

__device__ __forceinline__ int4 ldw4(const void * p) {
    int4 r;
    asm("ld.global.nc.L1::no_allocate.v4.s32 {%0,%1,%2,%3}, [%4];" : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
    return r;
}
__device__ __forceinline__ float wsum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

__device__ __forceinline__ void q4k_scales(const int4 & h, int p, float & fA, float & gA, float & fB, float & gB) {
    const float2 dm = __half22float2(*reinterpret_cast<const __half2 *>(&h.x));
    const int sh = (p & 1) * 16;
    const unsigned y = (unsigned) h.y >> sh, z = (unsigned) h.z >> sh, w = (unsigned) h.w >> sh;
    unsigned sc0, sc1, m0, m1;
    if (p < 2) { sc0 = y & 63; sc1 = (y >> 8) & 63; m0 = z & 63; m1 = (z >> 8) & 63; }
    else {
        sc0 = (w & 0xF) | ((y >> 2) & 0x30); sc1 = ((w >> 8) & 0xF) | ((y >> 10) & 0x30);
        m0 = ((w >> 4) & 0xF) | ((z >> 2) & 0x30); m1 = ((w >> 12) & 0xF) | ((z >> 10) & 0x30);
    }
    fA = dm.x * (float) sc0; fB = dm.x * (float) sc1; gA = dm.y * (float) m0; gB = dm.y * (float) m1;
}

// ---------------- V0: current (uint2 xf, 1 row per row_dot, rows in pairs) ----------------
template <int T>
__device__ __forceinline__ void acc_unit0(const int4 * xa, const int4 * xb, const uint2 * xf, int u, const int (&a)[4], const int (&b)[4],
                                          float fA, float gA, float fB, float gB, float (&acc)[T]) {
#pragma unroll
    for (int t = 0; t < T; t++) {
        const int i = t * U + u;
        const int4 va = xa[i], vb = xb[i];
        const uint2 f = xf[i];
        int sa = __dp4a(a[0], va.x, 0); sa = __dp4a(a[1], va.y, sa); sa = __dp4a(a[2], va.z, sa); sa = __dp4a(a[3], va.w, sa);
        int sb = __dp4a(b[0], vb.x, 0); sb = __dp4a(b[1], vb.y, sb); sb = __dp4a(b[2], vb.z, sb); sb = __dp4a(b[3], vb.w, sb);
        const float2 dx = __half22float2(*reinterpret_cast<const __half2 *>(&f.x));
        const float qa = (float) (short) (f.y & 0xffffu), qb = (float) (short) (f.y >> 16);
        acc[t] += dx.x * (fA * (float) sa - gA * qa) + dx.y * (fB * (float) sb - gB * qb);
    }
}

template <int T>
__global__ void __launch_bounds__(NT, 1) gemv_v0(const uint8_t * W, int N, float * out) {
    extern __shared__ __align__(16) uint8_t sm[];
    int4 * xa = (int4 *) sm; int4 * xb = xa + T * U; uint2 * xf = (uint2 *) (xb + T * U);
    for (int i = threadIdx.x; i < T * U; i += NT) {
        xa[i] = make_int4(0x01020304 * (i & 7), 0x05060708, 0x01010101, 0x7f7f7f7f);
        xb[i] = make_int4(0x02020202, 0x03030303, 0x04040404, 0x05050505);
        __half2 h = __floats2half2_rn(0.01f, 0.02f);
        xf[i] = make_uint2(*(unsigned *) &h, 0x00050003);
    }
    __syncthreads();
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int gw = warp * gridDim.x + blockIdx.x, W_ = NW * gridDim.x;
    for (int r = gw; r < N; r += 2 * W_) {
        const int r2 = r + W_;
        float a0[T], a1[T];
#pragma unroll
        for (int t = 0; t < T; t++) a0[t] = a1[t] = 0.f;
#pragma unroll
        for (int rr = 0; rr < 2; rr++) {
            const int row = rr ? r2 : r;
            if (row >= N) break;
            const uint8_t * rp = W + (size_t) row * RB;
#pragma unroll
            for (int i = 0; i < 5; i++) {
                const int u = lane + 32 * i, b = u >> 3, p = (u >> 1) & 3, hh = u & 1;
                const uint8_t * blk = rp + b * 144;
                const int4 h = ldw4(blk);
                const int4 q = ldw4(blk + 16 + 32 * p + 16 * hh);
                float fA, gA, fB, gB;
                q4k_scales(h, p, fA, gA, fB, gB);
                const int a[4] = {q.x & M4, q.y & M4, q.z & M4, q.w & M4};
                const int bb[4] = {(q.x >> 4) & M4, (q.y >> 4) & M4, (q.z >> 4) & M4, (q.w >> 4) & M4};
                if (rr == 0) acc_unit0<T>(xa, xb, xf, u, a, bb, fA, gA, fB, gB, a0);
                else acc_unit0<T>(xa, xb, xf, u, a, bb, fA, gA, fB, gB, a1);
            }
        }
#pragma unroll
        for (int t = 0; t < T; t++) {
            a0[t] = wsum(a0[t]);
            a1[t] = wsum(a1[t]);
        }
        if (lane < T) {
            out[(size_t) lane * N + r] = a0[lane < T ? lane : 0];
            if (r2 < N) out[(size_t) lane * N + r2] = a1[lane < T ? lane : 0];
        }
    }
}

// ---------------- V1: R rows per warp share activation loads; float4 x scales ----------------
template <int T, int R>
__global__ void __launch_bounds__(NT, 1) gemv_v1(const uint8_t * W, int N, float * out) {
    extern __shared__ __align__(16) uint8_t sm[];
    int4 * xa = (int4 *) sm; int4 * xb = xa + T * U; float4 * xf = (float4 *) (xb + T * U);
    for (int i = threadIdx.x; i < T * U; i += NT) {
        xa[i] = make_int4(0x01020304 * (i & 7), 0x05060708, 0x01010101, 0x7f7f7f7f);
        xb[i] = make_int4(0x02020202, 0x03030303, 0x04040404, 0x05050505);
        xf[i] = make_float4(0.01f, 0.02f, 0.03f, 0.05f);
    }
    __syncthreads();
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int gw = warp * gridDim.x + blockIdx.x, W_ = NW * gridDim.x;
    for (int r0 = gw * R; r0 < N; r0 += W_ * R) {
        float acc[R][T];
#pragma unroll
        for (int k = 0; k < R; k++)
#pragma unroll
            for (int t = 0; t < T; t++) acc[k][t] = 0.f;
#pragma unroll
        for (int i = 0; i < 5; i++) {
            const int u = lane + 32 * i, b = u >> 3, p = (u >> 1) & 3, hh = u & 1;
            int4 h[R], q[R];
#pragma unroll
            for (int k = 0; k < R; k++) {
                const uint8_t * blk = W + (size_t) (r0 + k) * RB + b * 144;
                h[k] = ldw4(blk);
                q[k] = ldw4(blk + 16 + 32 * p + 16 * hh);
            }
            float fA[R], gA[R], fB[R], gB[R];
#pragma unroll
            for (int k = 0; k < R; k++) q4k_scales(h[k], p, fA[k], gA[k], fB[k], gB[k]);
#pragma unroll
            for (int t = 0; t < T; t++) {
                const int ix = t * U + u;
                const int4 va = xa[ix], vb = xb[ix];
                const float4 f = xf[ix];
#pragma unroll
                for (int k = 0; k < R; k++) {
                    int sa = __dp4a(q[k].x & M4, va.x, 0); sa = __dp4a(q[k].y & M4, va.y, sa);
                    sa = __dp4a(q[k].z & M4, va.z, sa); sa = __dp4a(q[k].w & M4, va.w, sa);
                    int sb = __dp4a((q[k].x >> 4) & M4, vb.x, 0); sb = __dp4a((q[k].y >> 4) & M4, vb.y, sb);
                    sb = __dp4a((q[k].z >> 4) & M4, vb.z, sb); sb = __dp4a((q[k].w >> 4) & M4, vb.w, sb);
                    acc[k][t] += fA[k] * (f.x * (float) sa) + fB[k] * (f.y * (float) sb) - gA[k] * f.z - gB[k] * f.w;
                }
            }
        }
#pragma unroll
        for (int k = 0; k < R; k++)
#pragma unroll
            for (int t = 0; t < T; t++) acc[k][t] = wsum(acc[k][t]);
        if (lane == 0)
#pragma unroll
            for (int k = 0; k < R; k++)
#pragma unroll
                for (int t = 0; t < T; t++) out[(size_t) t * N + r0 + k] = acc[k][t];
    }
}


// ---------------- V2: R rows/warp, compact scales (float2 dx + short2 qsum), K and type templated -------
template <int KK, int TYPE, int T, int R, int UF = KK / 1024>
__global__ void __launch_bounds__(NT, 1) gemv_v2(const uint8_t * W, const __half * Dq, int N, float * out) {
    constexpr int UU = KK / 32, IT = UU / 32, NBB = KK / 256;
    constexpr int RBB = TYPE == 12 ? NBB * 144 : (TYPE == 18 ? NBB * 224 : NBB * 208);
    extern __shared__ __align__(16) uint8_t sm[];
    int4 * xa = (int4 *) sm; int4 * xb = xa + T * UU; float2 * xd = (float2 *) (xb + T * UU); int * xq = (int *) (xd + T * UU);
    for (int i = threadIdx.x; i < T * UU; i += NT) {
        xa[i] = make_int4(0x01020304 * (i & 7), 0x05060708, 0x01010101, 0x7f7f7f7f);
        xb[i] = make_int4(0x02020202, 0x03030303, 0x04040404, 0x05050505);
        xd[i] = make_float2(0.01f, 0.02f);
        xq[i] = 0x00050003;
    }
    __syncthreads();
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int gw = warp * gridDim.x + blockIdx.x, W_ = NW * gridDim.x;
    for (int r0 = gw * R; r0 < N; r0 += W_ * R) {
        float acc[R][T];
#pragma unroll
        for (int k = 0; k < R; k++)
#pragma unroll
            for (int t = 0; t < T; t++) acc[k][t] = 0.f;
#pragma unroll UF
        for (int i = 0; i < IT; i++) {
            const int u = lane + 32 * i, b = u >> 3, p = (u >> 1) & 3, hh = u & 1;
            int A[R][4], B[R][4];
            float fA[R], gA[R], fB[R], gB[R];
            if (TYPE == 12) {
                int4 h[R], q[R];
#pragma unroll
                for (int k = 0; k < R; k++) {
                    const uint8_t * blk = W + (size_t) (r0 + k) * RBB + b * 144;
                    h[k] = ldw4(blk);
                    q[k] = ldw4(blk + 16 + 32 * p + 16 * hh);
                }
#pragma unroll
                for (int k = 0; k < R; k++) {
                    q4k_scales(h[k], p, fA[k], gA[k], fB[k], gB[k]);
                    A[k][0] = q[k].x & M4; A[k][1] = q[k].y & M4; A[k][2] = q[k].z & M4; A[k][3] = q[k].w & M4;
                    B[k][0] = (q[k].x >> 4) & M4; B[k][1] = (q[k].y >> 4) & M4; B[k][2] = (q[k].z >> 4) & M4; B[k][3] = (q[k].w >> 4) & M4;
                }
            } else if (TYPE == 18) {
                int4 q[R]; int2 hb[R]; int2 meta[R];
#pragma unroll
                for (int k = 0; k < R; k++) {
                    const uint8_t * blk = W + (size_t) (r0 + k) * RBB + b * 224;
                    q[k] = ldw4(blk + 32 * p + 16 * hh);
                    int2 t2; asm("ld.global.nc.L1::no_allocate.v2.s32 {%0,%1}, [%2];" : "=r"(t2.x), "=r"(t2.y) : "l"(blk + 128 + 8 * (u & 7))); hb[k] = t2;
                    asm("ld.global.nc.L1::no_allocate.v2.s32 {%0,%1}, [%2];" : "=r"(t2.x), "=r"(t2.y) : "l"(blk + 192 + 8 * p)); meta[k] = t2;
                }
#pragma unroll
                for (int k = 0; k < R; k++) {
                    const float d = __half2float(__ushort_as_half((unsigned short) (meta[k].y & 0xffff)));
                    fA[k] = d * (float) (int8_t) (meta[k].x >> (8 * hh)); fB[k] = d * (float) (int8_t) (meta[k].x >> (16 + 8 * hh));
                    gA[k] = 32.f * fA[k]; gB[k] = 32.f * fB[k];
                    const unsigned hx = hb[k].x, hy = hb[k].y;
                    A[k][0] = (q[k].x & M4) | (int) (((hx >> 0) & 0x03030303u) << 4); A[k][1] = (q[k].y & M4) | (int) (((hx >> 2) & 0x03030303u) << 4);
                    A[k][2] = (q[k].z & M4) | (int) (((hx >> 4) & 0x03030303u) << 4); A[k][3] = (q[k].w & M4) | (int) (((hx >> 6) & 0x03030303u) << 4);
                    B[k][0] = ((q[k].x >> 4) & M4) | (int) (((hy >> 0) & 0x03030303u) << 4); B[k][1] = ((q[k].y >> 4) & M4) | (int) (((hy >> 2) & 0x03030303u) << 4);
                    B[k][2] = ((q[k].z >> 4) & M4) | (int) (((hy >> 4) & 0x03030303u) << 4); B[k][3] = ((q[k].w >> 4) & M4) | (int) (((hy >> 6) & 0x03030303u) << 4);
                }
            } else {
                int4 q[R]; int2 hb[R]; int sc4[R]; unsigned short dd[R];
#pragma unroll
                for (int k = 0; k < R; k++) {
                    const uint8_t * blk = W + (size_t) (r0 + k) * RBB + b * 208;
                    q[k] = ldw4(blk + 32 * p + 16 * hh);
                    int2 t2; asm("ld.global.nc.L1::no_allocate.v2.s32 {%0,%1}, [%2];" : "=r"(t2.x), "=r"(t2.y) : "l"(blk + 128 + 8 * (u & 7))); hb[k] = t2;
                    int t1 = 0x01020304; if (TYPE == 14 || TYPE == 15) asm("ld.global.nc.L1::no_allocate.s32 %0, [%1];" : "=r"(t1) : "l"(blk + 192 + 4 * p)); sc4[k] = t1;
                    unsigned short s = 0x1400; if (TYPE == 14 || TYPE == 16) asm("ld.global.nc.L1::no_allocate.u16 %0, [%1];" : "=h"(s) : "l"(Dq + (size_t) (r0 + k) * NBB + b)); dd[k] = s;
                }
#pragma unroll
                for (int k = 0; k < R; k++) {
                    const float d = __half2float(__ushort_as_half(dd[k]));
                    fA[k] = d * (float) (int8_t) (sc4[k] >> (8 * hh)); fB[k] = d * (float) (int8_t) (sc4[k] >> (16 + 8 * hh));
                    gA[k] = 32.f * fA[k]; gB[k] = 32.f * fB[k];
                    const unsigned hx = hb[k].x, hy = hb[k].y;
                    A[k][0] = (q[k].x & M4) | (int) (((hx >> 0) & 0x03030303u) << 4); A[k][1] = (q[k].y & M4) | (int) (((hx >> 2) & 0x03030303u) << 4);
                    A[k][2] = (q[k].z & M4) | (int) (((hx >> 4) & 0x03030303u) << 4); A[k][3] = (q[k].w & M4) | (int) (((hx >> 6) & 0x03030303u) << 4);
                    B[k][0] = ((q[k].x >> 4) & M4) | (int) (((hy >> 0) & 0x03030303u) << 4); B[k][1] = ((q[k].y >> 4) & M4) | (int) (((hy >> 2) & 0x03030303u) << 4);
                    B[k][2] = ((q[k].z >> 4) & M4) | (int) (((hy >> 4) & 0x03030303u) << 4); B[k][3] = ((q[k].w >> 4) & M4) | (int) (((hy >> 6) & 0x03030303u) << 4);
                }
            }
#pragma unroll
            for (int t = 0; t < T; t++) {
                const int ix = t * UU + u;
                const int4 va = xa[ix], vb = xb[ix];
                const float2 dx = xd[ix];
                const int qq = xq[ix];
                const float sqA = dx.x * (float) (short) (qq & 0xffff), sqB = dx.y * (float) (short) (qq >> 16);
#pragma unroll
                for (int k = 0; k < R; k++) {
                    int sa = __dp4a(A[k][0], va.x, 0); sa = __dp4a(A[k][1], va.y, sa); sa = __dp4a(A[k][2], va.z, sa); sa = __dp4a(A[k][3], va.w, sa);
                    int sb = __dp4a(B[k][0], vb.x, 0); sb = __dp4a(B[k][1], vb.y, sb); sb = __dp4a(B[k][2], vb.z, sb); sb = __dp4a(B[k][3], vb.w, sb);
                    acc[k][t] += fA[k] * (dx.x * (float) sa) + fB[k] * (dx.y * (float) sb) - gA[k] * sqA - gB[k] * sqB;
                }
            }
        }
#pragma unroll
        for (int k = 0; k < R; k++)
#pragma unroll
            for (int t = 0; t < T; t++) acc[k][t] = wsum(acc[k][t]);
        if (lane == 0)
#pragma unroll
            for (int k = 0; k < R; k++)
#pragma unroll
                for (int t = 0; t < T; t++) out[(size_t) t * N + r0 + k] = acc[k][t];
    }
}

__global__ void stream_read(const int4 * p, size_t n, int * out) {
    int acc = 0;
    for (size_t i = blockIdx.x * (size_t) blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x * blockDim.x) {
        int4 v = ldw4(p + i);
        acc ^= v.x ^ v.y ^ v.z ^ v.w;
    }
    if (acc == 0x12345678) out[0] = acc;
}

template <class F> float timeit(F f, int reps = 25) {
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b); f(); CK(cudaDeviceSynchronize());
    float best = 1e30f;
    for (int i = 0; i < reps; i++) {
        cudaEventRecord(a); f(); cudaEventRecord(b); CK(cudaEventSynchronize(b));
        float ms; cudaEventElapsedTime(&ms, a, b); best = ms < best ? ms : best;
    }
    return best;
}

int main() {
    const int N = 248320 / 2 * 2 + 2 * 105472; // ~ 1.3 GB of Q4_K rows
    const size_t bytes = (size_t) N * RB;
    std::vector<uint8_t> h(bytes);
    for (size_t i = 0; i < bytes; i++) h[i] = (uint8_t) (i * 2654435761u >> 13);
    for (size_t r = 0; r < (size_t) N * NB; r++) { // sane d / dmin
        __half d = __float2half(0.001f), m = __float2half(0.002f);
        memcpy(&h[r * 144], &d, 2); memcpy(&h[r * 144 + 2], &m, 2);
    }
    uint8_t * dW; float * dout; int * dummy;
    CK(cudaMalloc(&dW, bytes)); CK(cudaMalloc(&dout, (size_t) 4 * N * 4)); CK(cudaMalloc(&dummy, 4));
    CK(cudaMemcpy(dW, h.data(), bytes, cudaMemcpyHostToDevice));
    int nsm; cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, 0);
    const double gb = bytes / 1e9;
    printf("matrix %.2f GB, %d rows\n", gb, N);
    for (int bs : {256, 512, 1024}) {
        float ms = timeit([&] { stream_read<<<nsm * (2048 / bs), bs>>>((const int4 *) dW, bytes / 16, dummy); });
        printf("stream read (block %d): %.1f GB/s\n", bs, gb / ms * 1e3);
    }
    auto run0 = [&](auto kern, int T, const char * name) {
        const int smem = 48 * 1024 + 16 * 1024;
        CK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
        float ms = timeit([&] { kern<<<nsm, NT, smem>>>(dW, N, dout); });
        CK(cudaGetLastError());
        printf("%-12s T=%d: %.3f ms  %.1f GB/s\n", name, T, ms, gb / ms * 1e3);
    };
    
    run0(gemv_v1<1, 1>, 1, "v1 R1"); run0(gemv_v1<4, 1>, 4, "v1 R1");
    run0(gemv_v1<1, 2>, 1, "v1 R2"); run0(gemv_v1<2, 2>, 2, "v1 R2"); run0(gemv_v1<4, 2>, 4, "v1 R2");
    run0(gemv_v1<1, 4>, 1, "v1 R4"); run0(gemv_v1<4, 4>, 4, "v1 R4");
    __half * dD; CK(cudaMalloc(&dD, bytes / 100)); CK(cudaMemset(dD, 0x11, bytes / 100));
    auto run2 = [&](auto kern, int KK, int TYPE, int T, int R) {
        const int rb = TYPE == 12 ? KK / 256 * 144 : (TYPE == 18 ? KK / 256 * 224 : KK / 256 * 208);
        const int n = (int) (bytes / rb) / 64 * 64;
        const int smem = 100 * 1024 - 1024;
        CK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
        float ms = timeit([&] { kern<<<nsm, NT, smem>>>(dW, (const __half *) dD, n, dout); });
        CK(cudaGetLastError());
        const double eff = TYPE == 18 ? (double) n * (KK / 256 * 210) / 1e9 / ms * 1e3 : (double) n * rb / 1e9 / ms * 1e3;
        printf("v2 K=%5d %s T=%d R=%d: %.3f ms  %.1f GB/s raw, %.1f GB/s gguf-equivalent\n", KK, TYPE == 12 ? "q4k" : (TYPE == 18 ? "q6k-pad" : "q6k"), T, R, ms, (double) n * rb / 1e9 / ms * 1e3, eff);
    };
    for (int rep = 0; rep < 2; rep++) {
    run2(gemv_v2<17408, 18, 4, 2>, 17408, 18, 4, 2); run2(gemv_v2<17408, 18, 4, 2, 4>, 17408, 18, 4, 2); run2(gemv_v2<17408, 18, 4, 2, 2>, 17408, 18, 4, 2);
    run2(gemv_v2<17408, 12, 4, 2>, 17408, 12, 4, 2); run2(gemv_v2<17408, 12, 4, 2, 4>, 17408, 12, 4, 2); run2(gemv_v2<17408, 12, 4, 2, 2>, 17408, 12, 4, 2);
    run2(gemv_v2<17408, 18, 1, 2>, 17408, 18, 1, 2); run2(gemv_v2<17408, 12, 1, 2>, 17408, 12, 1, 2);
    }
    return 0;
}
