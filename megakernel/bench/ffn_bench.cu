// ffn gate/up phase replica: 17408 rows x 2 matrices (Q4_K, K=5120), several layers to defeat L2
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <vector>
#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1);} } while (0)
constexpr int K = 5120, U = K / 32, RB = K / 256 * 144, FF = 17408, NT = 512, NW = 16, M4 = 0x0F0F0F0F, NLAY = 8;
#ifndef HINT
#define HINT 0
#endif
__device__ __forceinline__ int4 ldw4(const void * p) {
    int4 r;
#if HINT == 0
    asm("ld.global.nc.L1::no_allocate.v4.s32 {%0,%1,%2,%3}, [%4];" : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
#elif HINT == 1
    unsigned long long pol; asm("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(pol));
    asm("ld.global.nc.L1::no_allocate.L2::cache_hint.v4.s32 {%0,%1,%2,%3}, [%4], %5;" : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p), "l"(pol));
#elif HINT == 2
    asm("ld.global.nc.L1::no_allocate.L2::256B.v4.s32 {%0,%1,%2,%3}, [%4];" : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
#else
    unsigned long long pol; asm("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(pol));
    asm("ld.global.nc.L1::no_allocate.L2::cache_hint.L2::256B.v4.s32 {%0,%1,%2,%3}, [%4], %5;" : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p), "l"(pol));
#endif
    return r;
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
struct SX { int4 * xa; int4 * xb; float2 * xd; int * xq; };
template <int T, int R>
__device__ __forceinline__ void rows_q4k(const uint8_t * const (&rp)[R], const SX & s, float (&acc)[R][T]) {
    const int lane = threadIdx.x & 31;
#pragma unroll
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
__device__ void setup(SX & s, uint8_t * sm, int T) {
    s.xa = (int4 *) sm; s.xb = s.xa + 4 * U; s.xd = (float2 *) (s.xb + 4 * U); s.xq = (int *) (s.xd + 4 * U);
    for (int i = threadIdx.x; i < T * U; i += NT) { s.xa[i] = make_int4(0x01020304, 0x05060708, 0x01010101, 0x7f7f7f7f); s.xb[i] = s.xa[i]; s.xd[i] = make_float2(0.01f, 0.02f); s.xq[i] = 0x00050003; }
    __syncthreads();
}
// variant A: current megakernel ffn_gu (16-row warp items, 2 gate + 2 up rows per step)
template <int T, int RG>
__global__ void __launch_bounds__(NT, 1) ffn_a(const uint8_t * Wg, const uint8_t * Wu, float * out, int layer_stride) {
    extern __shared__ __align__(16) uint8_t sm[]; SX s; setup(s, sm, T);
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, gw = warp * gridDim.x + blockIdx.x, NWt = NW * gridDim.x;
    for (int L = 0; L < NLAY; L++) {
        const uint8_t * gw_ = Wg + (size_t) L * layer_stride, * uw_ = Wu + (size_t) L * layer_stride;
        for (int j = gw; j < FF / 16; j += NWt) {
            float act[T] = {};
#pragma unroll 1
            for (int r = 0; r < 16; r += RG) {
                const int row = 16 * j + r;
                const uint8_t * rp[2 * RG];
#pragma unroll
                for (int k = 0; k < RG; k++) { rp[k] = gw_ + (size_t) (row + k) * RB; rp[RG + k] = uw_ + (size_t) (row + k) * RB; }
                float a[2 * RG][T] = {};
                rows_q4k<T, 2 * RG>(rp, s, a);
#pragma unroll
                for (int k = 0; k < 2 * RG; k++)
#pragma unroll
                    for (int t = 0; t < T; t++) a[k][t] = wsum(a[k][t]);
                if (lane >= r && lane < r + RG) {
#pragma unroll
                    for (int k = 0; k < RG; k++) if (lane == r + k)
#pragma unroll
                        for (int t = 0; t < T; t++) act[t] = a[k][t] / (1.f + __expf(-a[k][t])) * a[RG + k][t];
                }
            }
            if (lane < 16) for (int t = 0; t < T; t++) out[(size_t) t * FF + 16 * j + lane] = act[t];
        }
        __syncthreads();
    }
}

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
template <int T, int RG, int PF>
__global__ void __launch_bounds__(NT, 1) ffn_b(const uint8_t * Wg, const uint8_t * Wu, float * out, int layer_stride) {
    extern __shared__ __align__(16) uint8_t sm[]; SX s; setup(s, sm, T);
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, gw = warp * gridDim.x + blockIdx.x, NWt = NW * gridDim.x;
    unsigned epoch = 0;
    if (threadIdx.x == 0 && blockIdx.x == 0) g_bar = 0;
    for (int L = 0; L < NLAY; L++) {
        const uint8_t * gw_ = Wg + (size_t) L * layer_stride, * uw_ = Wu + (size_t) L * layer_stride;
        // emulate fill: reload activations through L2
        for (int i = threadIdx.x; i < T * U; i += NT) { s.xd[i].x += __ldcg(out + i) * 0.f; }
        __syncthreads();
        for (int j = gw; j < FF / 16; j += NWt) {
            float act[T] = {};
#pragma unroll 1
            for (int r = 0; r < 16; r += RG) {
                const int row = 16 * j + r;
                const uint8_t * rp[2 * RG];
#pragma unroll
                for (int k = 0; k < RG; k++) { rp[k] = gw_ + (size_t) (row + k) * RB; rp[RG + k] = uw_ + (size_t) (row + k) * RB; }
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
        if (PF && L + 1 < NLAY && gw < FF / 16) {
            const uint8_t * ng = Wg + (size_t) (L + 1) * layer_stride + (size_t) 16 * gw * RB;
            const uint8_t * nu = Wu + (size_t) (L + 1) * layer_stride + (size_t) 16 * gw * RB;
            for (int k = 0; k < PF; k++) {
                const int off = lane * 128 + k * 4096;
                if (off < RB * 2) {
                    asm volatile("prefetch.global.L2::evict_last [%0];" ::"l"(ng + off));
                    asm volatile("prefetch.global.L2::evict_last [%0];" ::"l"(nu + off));
                }
            }
        }
        gsync(epoch);
    }
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
__global__ void stream_read(const int4 * p, size_t n, int * out) {
    int acc = 0;
    for (size_t i = blockIdx.x * (size_t) blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x * blockDim.x) { int4 v = ldw4(p + i); acc ^= v.x ^ v.y ^ v.z ^ v.w; }
    if (acc == 0x12345678) out[0] = acc;
}
int main() {
    const size_t mat = (size_t) FF * RB, bytes = 2 * NLAY * mat;
    std::vector<uint8_t> h(bytes);
    for (size_t i = 0; i < bytes; i++) h[i] = (uint8_t) (i * 2654435761u >> 13);
    for (size_t r = 0; r < bytes / 144; r++) { __half d = __float2half(0.001f); memcpy(&h[r * 144], &d, 2); memcpy(&h[r * 144 + 2], &d, 2); }
    uint8_t * dW; float * dout; int * dummy;
    CK(cudaMalloc(&dW, bytes)); CK(cudaMalloc(&dout, 4 * FF * 4)); CK(cudaMalloc(&dummy, 4));
    CK(cudaMemcpy(dW, h.data(), bytes, cudaMemcpyHostToDevice));
    int nsm; cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, 0);
    const double gb = bytes / 1e9;
    // gate matrices at [0, NLAY*mat), up at [NLAY*mat, ...) ; layer stride = mat
    for (int rep = 0; rep < 2; rep++) {
        float ms = timeit([&] { stream_read<<<nsm * 4, 512>>>((const int4 *) dW, bytes / 16, dummy); });
        printf("stream: %.1f GB/s\n", gb / ms * 1e3);
        auto run = [&](auto k, const char * name) {
            CK(cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, 64 * 1024));
            float ms = timeit([&] { k<<<nsm, NT, 64 * 1024>>>(dW, dW + NLAY * mat, dout, (int) mat); });
            CK(cudaGetLastError());
            printf("%-24s %.3f ms/layer  %.1f GB/s\n", name, ms / NLAY, gb / ms * 1e3);
        };
        run(ffn_a<1, 2>, "A T=1 RG=2 (no barrier)"); run(ffn_a<4, 2>, "A T=4 RG=2 (no barrier)");
        auto runc = [&](auto k, const char * name, int smem = 64 * 1024) {
            CK(cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
            const uint8_t * a0 = dW; const uint8_t * a1 = dW + NLAY * mat; int ls = (int) mat;
            void * args[] = {&a0, &a1, &dout, &ls};
            float ms = timeit([&] { CK(cudaLaunchCooperativeKernel((void *) k, dim3(nsm), dim3(NT), args, smem, 0)); });
            printf("%-28s %.3f ms/layer  %.1f GB/s\n", name, ms / NLAY, gb / ms * 1e3);
        };
        runc(ffn_b<1, 2, 0>, "B T=1 barrier"); runc(ffn_b<4, 2, 0>, "B T=4 barrier");
        runc(ffn_b<1, 2, 0>, "B T=1 barrier smem97K", 97792); runc(ffn_b<4, 2, 0>, "B T=4 barrier smem97K", 97792);
    }
}
