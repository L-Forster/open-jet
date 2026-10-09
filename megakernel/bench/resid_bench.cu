// ffn_down-shaped phase (N=5120 rows, K=17408, Q4_K) : block-level 32-row items vs warp-level 16-row items
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <vector>
#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1);} } while (0)
constexpr int K = 17408, U = K / 32, IT = U / 32, RB = K / 256 * 144, N = 5120, NT = 512, NW = 16, M4 = 0x0F0F0F0F, NLAY = 12;
__device__ __forceinline__ int4 ldw4(const void * p) { int4 r; asm("ld.global.nc.L1::no_allocate.v4.s32 {%0,%1,%2,%3}, [%4];" : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p)); return r; }
__device__ __forceinline__ float wsum(float v) { for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(~0u, v, o); return v; }
__device__ __forceinline__ void q4k_scales(const int4 & h, int p, float & fA, float & gA, float & fB, float & gB) {
    const float2 dm = __half22float2(*reinterpret_cast<const __half2 *>(&h.x));
    const int sh = (p & 1) * 16; const unsigned y = (unsigned) h.y >> sh, z = (unsigned) h.z >> sh, w = (unsigned) h.w >> sh;
    unsigned sc0, sc1, m0, m1;
    if (p < 2) { sc0 = y & 63; sc1 = (y >> 8) & 63; m0 = z & 63; m1 = (z >> 8) & 63; }
    else { sc0 = (w & 0xF) | ((y >> 2) & 0x30); sc1 = ((w >> 8) & 0xF) | ((y >> 10) & 0x30); m0 = ((w >> 4) & 0xF) | ((z >> 2) & 0x30); m1 = ((w >> 12) & 0xF) | ((z >> 10) & 0x30); }
    fA = dm.x * sc0; fB = dm.x * sc1; gA = dm.y * m0; gB = dm.y * m1;
}
struct SX { int4 * xa; int4 * xb; float2 * xd; int * xq; };
template <int T, int R>
__device__ __forceinline__ void rows_q4k(const uint8_t * const (&rp)[R], const SX & s, float (&acc)[R][T]) {
    const int lane = threadIdx.x & 31;
#pragma unroll
    for (int i = 0; i < IT; i++) {
        const int u = lane + 32 * i, b = u >> 3, p = (u >> 1) & 3, hh = u & 1;
        int4 h[R], q[R];
#pragma unroll
        for (int k = 0; k < R; k++) { const uint8_t * blk = rp[k] + b * 144; h[k] = ldw4(blk); q[k] = ldw4(blk + 16 + 32 * p + 16 * hh); }
        float fA[R], gA[R], fB[R], gB[R];
#pragma unroll
        for (int k = 0; k < R; k++) q4k_scales(h[k], p, fA[k], gA[k], fB[k], gB[k]);
#pragma unroll
        for (int t = 0; t < T; t++) {
            const int ix = t * U + u; const int4 va = s.xa[ix], vb = s.xb[ix]; const float2 dx = s.xd[ix]; const int qq = s.xq[ix];
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
__device__ unsigned g_bar;
__device__ __forceinline__ void gsync(unsigned & epoch) {
    __syncthreads();
    if (threadIdx.x == 0) {
        epoch += gridDim.x;
        asm volatile("red.release.gpu.global.add.u32 [%0], 1;" ::"l"(&g_bar) : "memory");
        unsigned v; do { asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(&g_bar) : "memory"); } while (v < epoch);
    }
    __syncthreads();
}
__device__ void setup(SX & s, uint8_t * sm, int T) {
    s.xa = (int4 *) sm; s.xb = s.xa + 4 * U; s.xd = (float2 *) (s.xb + 4 * U); s.xq = (int *) (s.xd + 4 * U);
    for (int i = threadIdx.x; i < T * U; i += NT) { s.xa[i] = make_int4(0x01020304, 0x05060708, 0x01010101, 0x7f7f7f7f); s.xb[i] = s.xa[i]; s.xd[i] = make_float2(0.01f, 0.02f); s.xq[i] = 0x00050003; }
    __syncthreads();
}
// MODE 0: block items of 32 rows (16 warps x 2 rows) + smem exchange + quantise by T warps
// MODE 1: warp items of 16 rows (8 steps of 2 rows), quantise in-warp
template <int T, int MODE>
__global__ void __launch_bounds__(NT, 1) resid(const uint8_t * W, float * x, int8_t * qout) {
    extern __shared__ __align__(16) uint8_t sm[]; SX s; setup(s, sm, T);
    __shared__ float s_res[4 * 32];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    unsigned epoch = 0;
    if (threadIdx.x == 0 && blockIdx.x == 0) g_bar = 0;
    for (int L = 0; L < NLAY; L++) {
        const uint8_t * Wl = W + (size_t) L * N * RB;
        if (MODE == 0) {
            for (int j = blockIdx.x; j < N / 32; j += gridDim.x) {
                const uint8_t * rp[2] = {Wl + (size_t) (32 * j + 2 * warp) * RB, Wl + (size_t) (32 * j + 2 * warp + 1) * RB};
                float a[2][T] = {};
                rows_q4k<T, 2>(rp, s, a);
                for (int k = 0; k < 2; k++) for (int t = 0; t < T; t++) a[k][t] = wsum(a[k][t]);
                if (lane == 0) for (int t = 0; t < T; t++) { s_res[t * 32 + 2 * warp] = a[0][t]; s_res[t * 32 + 2 * warp + 1] = a[1][t]; }
                __syncthreads();
                if (warp < T) {
                    float xv = s_res[warp * 32 + lane] + __ldcg(x + warp * N + 32 * j + lane);
                    x[warp * N + 32 * j + lane] = xv;
                    float am = fabsf(xv); for (int o = 16; o; o >>= 1) am = fmaxf(am, __shfl_xor_sync(~0u, am, o));
                    qout[warp * N + 32 * j + lane] = (int8_t) roundf(xv * 127.f / (am + 1e-9f));
                }
                __syncthreads();
            }
        } else {
            const int gw = warp * gridDim.x + blockIdx.x, NWt = NW * gridDim.x;
            for (int j = gw; j < N / 16; j += NWt) {
                float keep[T] = {};
#pragma unroll 1
                for (int r = 0; r < 16; r += 2) {
                    const uint8_t * rp[2] = {Wl + (size_t) (16 * j + r) * RB, Wl + (size_t) (16 * j + r + 1) * RB};
                    float a[2][T] = {};
                    rows_q4k<T, 2>(rp, s, a);
#pragma unroll
                    for (int k = 0; k < 2; k++)
#pragma unroll
                        for (int t = 0; t < T; t++) a[k][t] = wsum(a[k][t]);
                    if (lane == r) for (int t = 0; t < T; t++) keep[t] = a[0][t];
                    if (lane == r + 1) for (int t = 0; t < T; t++) keep[t] = a[1][t];
                }
                if (lane < 16)
                    for (int t = 0; t < T; t++) {
                        float xv = keep[t] + __ldcg(x + t * N + 16 * j + lane);
                        x[t * N + 16 * j + lane] = xv;
                        qout[t * N + 16 * j + lane] = (int8_t) roundf(xv);
                    }
            }
        }
        gsync(epoch);
    }
}
template <class F> float timeit(F f, int reps = 25) {
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b); f(); CK(cudaDeviceSynchronize());
    float best = 1e30f;
    for (int i = 0; i < reps; i++) { cudaEventRecord(a); f(); cudaEventRecord(b); CK(cudaEventSynchronize(b)); float ms; cudaEventElapsedTime(&ms, a, b); best = ms < best ? ms : best; }
    return best;
}
int main() {
    const size_t bytes = (size_t) NLAY * N * RB;
    std::vector<uint8_t> h(bytes);
    for (size_t i = 0; i < bytes; i++) h[i] = (uint8_t) (i * 2654435761u >> 13);
    for (size_t r = 0; r < bytes / 144; r++) { __half d = __float2half(0.001f); memcpy(&h[r * 144], &d, 2); memcpy(&h[r * 144 + 2], &d, 2); }
    uint8_t * dW; float * dx; int8_t * dq;
    CK(cudaMalloc(&dW, bytes)); CK(cudaMalloc(&dx, 4 * N * 4)); CK(cudaMalloc(&dq, 4 * N)); CK(cudaMemset(dx, 0, 4 * N * 4));
    CK(cudaMemcpy(dW, h.data(), bytes, cudaMemcpyHostToDevice));
    int nsm; cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, 0);
    auto run = [&](auto k, const char * name) {
        const int smem = 97792;
        CK(cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
        void * args[] = {&dW, &dx, &dq};
        float ms = timeit([&] { CK(cudaLaunchCooperativeKernel((void *) k, dim3(nsm), dim3(NT), args, smem, 0)); });
        printf("%-34s %.1f us/layer  %.1f GB/s\n", name, ms * 1000 / NLAY, bytes / 1e9 / ms * 1e3);
    };
    for (int rep = 0; rep < 2; rep++) {
        run(resid<1, 0>, "block 32-row items T=1"); run(resid<4, 0>, "block 32-row items T=4");
        run(resid<1, 1>, "warp 16-row items T=1"); run(resid<4, 1>, "warp 16-row items T=4");
    }
}
