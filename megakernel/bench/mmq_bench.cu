// Correctness + throughput of the prefill fp16 tensor-core GEMM (prefill_gemm.cuh) on random data.
//   nvcc -O3 -arch=sm_86 -std=c++17 mmq_bench.cu -o mmq_bench && ./mmq_bench [ntok]
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include "../prefill_gemm.cuh"

#define CK(x)                                                                                        \
    do {                                                                                             \
        cudaError_t e_ = (x);                                                                        \
        if (e_ != cudaSuccess) { fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); exit(1); } \
    } while (0)

using namespace pf;

// reference dequantisation of one weight element
__device__ float deq(const WMat & W, int row, int e) {
    const uint8_t * r = W.w + (size_t) row * W.rb;
    const int b = e >> 8, w = e & 255;
    if (W.type == TQ4K || W.type == TQ5K) {
        const uint8_t * blk = r + b * (W.type == TQ4K ? 144 : 176);
        const float d = __half2float(*reinterpret_cast<const __half *>(blk));
        const float dmin = __half2float(*reinterpret_cast<const __half *>(blk + 2));
        const uint8_t * sc = blk + 4;
        const int sb = w >> 5, l = w & 31;
        int s, m;
        if (sb < 4) { s = sc[sb] & 63; m = sc[sb + 4] & 63; }
        else { s = (sc[sb + 4] & 0xF) | ((sc[sb - 4] >> 6) << 4); m = (sc[sb + 4] >> 4) | ((sc[sb] >> 6) << 4); }
        int q;
        if (W.type == TQ4K) {
            const uint8_t qb = blk[16 + 32 * (sb >> 1) + l];
            q = (sb & 1) ? (qb >> 4) : (qb & 0xF);
        } else {
            const uint8_t qb = blk[48 + 32 * (sb >> 1) + l];
            q = ((sb & 1) ? (qb >> 4) : (qb & 0xF)) | (((blk[16 + l] >> sb) & 1) << 4);
        }
        return d * s * q - dmin * m;
    } else if (W.type == TQ6K) {
        const uint8_t * blk = r + b * 224;
        const int p = w >> 6, ww = w & 63, sub = ww >> 5, i = ww & 31, hh = i >> 4, ii = i & 15;
        const int lo = sub ? (blk[32 * p + i] >> 4) : (blk[32 * p + i] & 0xF);
        const unsigned hw = *reinterpret_cast<const unsigned *>(blk + 128 + 8 * (2 * p + hh) + 4 * sub);
        const int hi = (hw >> (8 * (ii % 4) + 2 * (ii / 4))) & 3;
        const int8_t sc = (int8_t) blk[192 + 8 * p + (ww >> 4)];
        const float d = __half2float(*reinterpret_cast<const __half *>(blk + 192 + 8 * p + 4));
        return d * sc * ((lo | (hi << 4)) - 32);
    } else {
        const int p = w >> 6, ww = w & 63, sub = ww >> 5, i = ww & 31, hh = i >> 4, ii = i & 15;
        const int u = 8 * b + 2 * p + hh;
        const int8_t q = (int8_t) r[32 * u + 16 * sub + ii];
        return __half2float(W.d[(size_t) row * W.nd + (e >> 5)]) * q;
    }
}

__global__ void ref_gemm(WMat W, const __half * X, int ntok, float * y) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x, t = blockIdx.y;
    if (row >= W.N) return;
    float acc = 0.f;
    for (int e = 0; e < W.K; e++) acc += deq(W, row, e) * __half2float(X[(size_t) t * W.K + hperm(e)]);
    y[(size_t) t * W.N + row] = acc;
}

static uint16_t h(float f) {
    __half x = __float2half(f);
    return *reinterpret_cast<uint16_t *>(&x);
}

int main(int argc, char ** argv) {
    const int ntok = argc > 1 ? atoi(argv[1]) : 1024;
    std::mt19937 rng(1);
    struct Case { int type, K, N; const char * name; };
    const Case cases[] = {{TQ4K, 5120, 17408, "q4k ffn_gate"}, {TQ4K, 17408, 5120, "q4k ffn_down"},
                          {TQ6K, 17408, 5120, "q6k ffn_down"}, {TQ5K, 6144, 5120, "q5k ssm_out"},
                          {TQ6K, 5120, 10240, "q6k qkv"}, {TQ80, 10240, 5120, "q8_0 eh_proj"}};
    for (const Case & cs : cases) {
        const int K = cs.K, N = cs.N, nb = K / 256;
        const int bsz = cs.type == TQ4K ? 144 : cs.type == TQ5K ? 176 : cs.type == TQ6K ? 224 : 256;
        std::vector<uint8_t> w((size_t) N * nb * bsz);
        for (auto & b : w) b = rng() & 0xff;
        std::vector<uint16_t> dsc;
        for (int r = 0; r < N; r++)
            for (int b = 0; b < nb; b++) {
                uint8_t * blk = w.data() + ((size_t) r * nb + b) * bsz;
                std::uniform_real_distribution<float> u(0.0005f, 0.004f);
                if (cs.type == TQ4K || cs.type == TQ5K) {
                    uint16_t d = h(u(rng)), dm = h(u(rng));
                    memcpy(blk, &d, 2);
                    memcpy(blk + 2, &dm, 2);
                } else if (cs.type == TQ6K) {
                    uint16_t d = h(u(rng));
                    for (int p = 0; p < 4; p++) memcpy(blk + 192 + 8 * p + 4, &d, 2); // d per super-block
                }
            }
        if (cs.type == TQ80) {
            dsc.resize((size_t) N * (K / 32));
            std::uniform_real_distribution<float> u(0.0005f, 0.004f);
            for (auto & d : dsc) d = h(u(rng));
        }
        std::vector<__half> xh((size_t) ntok * K);
        std::normal_distribution<float> nd(0.f, 1.f);
        for (auto & v : xh) v = __float2half(nd(rng));
        uint8_t * dw; __half *dx, *dd = nullptr; float *dy, *dr;
        CK(cudaMalloc(&dw, w.size()));
        CK(cudaMemcpy(dw, w.data(), w.size(), cudaMemcpyHostToDevice));
        if (!dsc.empty()) {
            CK(cudaMalloc(&dd, dsc.size() * 2));
            CK(cudaMemcpy(dd, dsc.data(), dsc.size() * 2, cudaMemcpyHostToDevice));
        }
        CK(cudaMalloc(&dx, xh.size() * 2));
        CK(cudaMemcpy(dx, xh.data(), xh.size() * 2, cudaMemcpyHostToDevice));
        CK(cudaMalloc(&dy, (size_t) ntok * N * 4));
        CK(cudaMalloc(&dr, (size_t) ntok * N * 4));
        const WMat W{dw, dd, cs.type, K, N, nb * bsz, K / 32};
        const Out O{dy, N, OUT_STORE, nullptr};
        const dim3 grid(N / BM, (ntok + BN - 1) / BN);
        void (*kern)(WMat, const __half *, int, Out, const uint8_t *) = nullptr;
        int smem = 0;
        switch (cs.type) {
            case TQ4K: kern = mmh<TQ4K>; smem = hm_smem<TQ4K>(); break;
            case TQ5K: kern = mmh<TQ5K>; smem = hm_smem<TQ5K>(); break;
            case TQ6K: kern = mmh<TQ6K>; smem = hm_smem<TQ6K>(); break;
            default: kern = mmh<TQ80>; smem = hm_smem<TQ80>(); break;
        }
        CK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
        kern<<<grid, 256, smem>>>(W, dx, ntok, O, nullptr);
        CK(cudaGetLastError());
        const int rt = 64; // reference on the first tokens only
        ref_gemm<<<dim3((N + 127) / 128, rt), 128>>>(W, dx, rt, dr);
        CK(cudaDeviceSynchronize());
        std::vector<float> y((size_t) rt * N), r((size_t) rt * N);
        CK(cudaMemcpy(y.data(), dy, y.size() * 4, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(r.data(), dr, r.size() * 4, cudaMemcpyDeviceToHost));
        double num = 0, den = 0, mx = 0;
        for (size_t i = 0; i < y.size(); i++) {
            num += (y[i] - r[i]) * (y[i] - r[i]);
            den += r[i] * r[i];
            mx = std::max(mx, (double) fabs(y[i] - r[i]));
        }
        cudaEvent_t a, b;
        cudaEventCreate(&a);
        cudaEventCreate(&b);
        float best = 1e30f;
        for (int it = 0; it < 20; it++) {
            cudaEventRecord(a);
            kern<<<grid, 256, smem>>>(W, dx, ntok, O, nullptr);
            cudaEventRecord(b);
            cudaEventSynchronize(b);
            float ms;
            cudaEventElapsedTime(&ms, a, b);
            best = std::min(best, ms);
        }
        const double tops = 2.0 * N * K * ntok / (best * 1e-3) / 1e12;
        printf("%-14s N=%5d K=%5d tok=%d: rel err %.2e (max abs %.2e)  %.3f ms  %.1f TFLOPS\n", cs.name, N, K, ntok,
               sqrt(num / den), mx, best, tops);
        cudaFree(dw); cudaFree(dx); cudaFree(dy); cudaFree(dr);
        if (dd) cudaFree(dd);
    }
    return 0;
}
