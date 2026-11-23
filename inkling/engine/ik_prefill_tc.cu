// Tensor-core (fp16 m16n8k16) batched prefill kernels for Inkling 2 Mini.
//
// Reuses megakernel/prefill_gemm.cuh directly (no copy of the GEMM pipeline, megakernel/ unchanged) for dense Q8_0
// (and GGUF Q4_K / Q5_K) projections, and applies its same hperm + ldsm_x4 + cp.async + m16n8k16 pattern to grouped
// NVFP4 and Q8_0 MoE experts (fused gate/up -> hperm fp16 H -> down).

#include "ik_prefill.cuh"

#include <cuda_fp16.h>
#include <cstdint>

// Definitions expected by megakernel/prefill_gemm.cuh (matching megakernel/mk_shared.cuh lines 564-578)
namespace {

__device__ __forceinline__ void iq4_lut(unsigned q, unsigned t0, unsigned t1, unsigned t2, unsigned t3, int & lo, int & hi) {
    const unsigned sel = 0x32103210u | ((q & 0x88888888u) >> 1);
    const unsigned a = __byte_perm(__byte_perm(t0, t1, q), __byte_perm(t2, t3, q), sel);
    const unsigned b = __byte_perm(__byte_perm(t0, t1, q >> 16), __byte_perm(t2, t3, q >> 16), sel >> 16);
    lo = (int) __byte_perm(a, b, 0x6420);
    hi = (int) __byte_perm(a, b, 0x7531);
}

__device__ __forceinline__ unsigned iq3s_grid(unsigned) { return 0u; }  // unused: Inkling has no IQ3_S tensors

}  // namespace

namespace pf {
constexpr int BM = 128, BN = 128, BK = 64;
constexpr int TQ4K = 12, TQ5K = 13, TQ6K = 14, TQ80 = 8, TIQ4 = 23, TIQ4NL = 20, TQ3K = 11, TIQ3S = 21;
struct WMat {
    const uint8_t * w;
    const __half * d;
    int type, K, N, rb, nd;
};
enum { OUT_STORE = 0, OUT_ADD = 1 };
struct Out {
    float * y;
    int ldy;
    int mode;
    __half * yh;
};
}  // namespace pf

#include "../../megakernel/prefill_gemm.cuh"

namespace {

// fp32 [T][K] -> fp16 [T_pad][K] with every 16-group permuted by pf::hperm (zero-padded for t >= T)
__global__ void f32_to_hperm_kernel(const float * __restrict__ X, __half * __restrict__ Xh, int T, int T_pad, int K) {
    const int t = blockIdx.y;
    const int g16 = blockIdx.x * (blockDim.x / 16) + threadIdx.x / 16;
    const int e = threadIdx.x & 15;
    if (t >= T_pad || g16 * 16 >= K) return;
    __half v = __float2half_rn(0.f);
    if (t < T) {
        const float f = __ldg(X + size_t(t) * K + g16 * 16 + e);
        v = __float2half_rn(fminf(fmaxf(f, -65504.f), 65504.f));
    }
    Xh[size_t(t) * K + g16 * 16 + pf::hperm(e)] = v;
}

// gather X[list[ge] / k] -> Xg[ge] as hperm fp16, for grouped expert GEMMs (ge = 0 .. n_entries - 1)
__global__ void gather_hperm_kernel(const float * __restrict__ X, const int * __restrict__ list, int k, int n_entries, int K,
                                    __half * __restrict__ Xg) {
    const int ge = blockIdx.y;
    const int g16 = blockIdx.x * (blockDim.x / 16) + threadIdx.x / 16;
    const int e = threadIdx.x & 15;
    if (ge >= n_entries || g16 * 16 >= K) return;
    const int tok = __ldg(list + ge) / k;
    const float f = __ldg(X + size_t(tok) * K + g16 * 16 + e);
    Xg[size_t(ge) * K + g16 * 16 + pf::hperm(e)] = __float2half_rn(fminf(fmaxf(f, -65504.f), 65504.f));
}

template <int TYPE, int NB>
cudaError_t launch_mmh(pf::WMat W, const __half * Xh, float * Y, int rows, int T, cudaStream_t st) {
    constexpr int SM = pf::hm_smem<TYPE, NB>();
    static bool configured = false;
    if (!configured && SM > 48 * 1024) {
        cudaError_t e = cudaFuncSetAttribute(pf::mmh<TYPE, false, NB, 0>, cudaFuncAttributeMaxDynamicSharedMemorySize, SM);
        if (e != cudaSuccess) return e;
        configured = true;
    }
    const int nr = rows / pf::BM;
    const int nt = (T + NB - 1) / NB;
    pf::Out O{Y, rows, pf::OUT_STORE, nullptr};
    pf::mmh<TYPE, false, NB, 0><<<dim3(nr, nt), 256, SM, st>>>(W, Xh, T, O);
    return cudaGetLastError();
}

// ---------------------------------------------------------------- grouped tensor-core MoE experts (NVFP4 & Q8_0)

__device__ __forceinline__ void hmma16_f32(float (&d)[4], const unsigned (&a)[4], unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ float ue4m3_tc(uint32_t x) {
    x &= 0x7F;
    const float normal = __int_as_float(int((x << 20) + (119u << 23)));
    return x == 0x7F ? 0.f : (x < 8 ? float(x) * 0.0009765625f : normal);
}

__device__ __forceinline__ float situ_tc(float g, float u, float beta, float lb) {
    const float act = beta * tanhf(g / beta) / (1.f + expf(-g));
    if (lb > 0.f) u = lb * tanhf(u / lb);
    return act * u;
}

__device__ const float k_fp4_tc[16] = {0.f, 1.f, 2.f, 3.f, 4.f, 6.f, 8.f, 12.f, -0.f, -1.f, -2.f, -3.f, -4.f, -6.f, -8.f, -12.f};

// Build A fragment for 16-group G (0..3 in BK=64 stage), rows ra (g) and rb (g+8), lane t4 (elements 4*t4 .. +3)
template <bool NVFP4>
__device__ __forceinline__ void expert_frag(const uint8_t * ra, const uint8_t * rb, int G, int t4, const __half * lut,
                                            unsigned (&a)[4]) {
    if constexpr (NVFP4) {
        const __half2 sa = __float2half2_rn(ue4m3_tc(ra[32 + G]));
        const __half2 sb = __float2half2_rn(ue4m3_tc(rb[32 + G]));
        const uint16_t wa = *reinterpret_cast<const uint16_t *>(ra + 8 * G + 2 * t4);
        const uint16_t wb = *reinterpret_cast<const uint16_t *>(rb + 8 * G + 2 * t4);
        a[0] = pf::h2u(__hmul2(make_half2(lut[wa & 0xF], lut[(wa >> 4) & 0xF]), sa));
        a[1] = pf::h2u(__hmul2(make_half2(lut[wb & 0xF], lut[(wb >> 4) & 0xF]), sb));
        a[2] = pf::h2u(__hmul2(make_half2(lut[(wa >> 8) & 0xF], lut[(wa >> 12) & 0xF]), sa));
        a[3] = pf::h2u(__hmul2(make_half2(lut[(wb >> 8) & 0xF], lut[(wb >> 12) & 0xF]), sb));
    } else {
        const __half2 off2 = __float2half2_rn(-1152.f);
        const __half2 sa = __half2half2(*reinterpret_cast<const __half *>(ra + 64 + 2 * (G >> 1)));
        const __half2 sb = __half2half2(*reinterpret_cast<const __half *>(rb + 64 + 2 * (G >> 1)));
        const unsigned wa = *reinterpret_cast<const unsigned *>(ra + 16 * G + 4 * t4) ^ 0x80808080u;
        const unsigned wb = *reinterpret_cast<const unsigned *>(rb + 16 * G + 4 * t4) ^ 0x80808080u;
        a[0] = pf::h2u(__hmul2(__hadd2(pf::u2h(__byte_perm(wa, 0x64646464u, 0x4140)), off2), sa));
        a[1] = pf::h2u(__hmul2(__hadd2(pf::u2h(__byte_perm(wb, 0x64646464u, 0x4140)), off2), sb));
        a[2] = pf::h2u(__hmul2(__hadd2(pf::u2h(__byte_perm(wa, 0x64646464u, 0x4342)), off2), sa));
        a[3] = pf::h2u(__hmul2(__hadd2(pf::u2h(__byte_perm(wb, 0x64646464u, 0x4342)), off2), sb));
    }
}

// Grouped tensor-core expert kernel:
//   GLU = true:  warps 0-3 compute 64 gate rows (row0..+63), warps 4-7 compute 64 up rows (row0+n_ff..+63),
//                epilogue writes hperm fp16 Hg[ge][row0..+63] = situ(s * gate, s * up)
//   GLU = false: warps 0-7 compute 128 down rows (row0..+127),
//                epilogue writes fp32 partial[list[ge]][row0..+127] = wts[list[ge]] * s * out
// Grid: (rows_out / (GLU ? 64 : 128), n_expert), 256 threads. Each CTA loops over the expert's tokens in 32-token tiles.
template <bool NVFP4, bool GLU>
__global__ void __launch_bounds__(256)
    experts_tc_kernel(const uint8_t * const * __restrict__ experts, const float * __restrict__ scale,
                      const int * __restrict__ offsets, const int * __restrict__ list, const float * __restrict__ wts,
                      const __half * __restrict__ Xg, __half * __restrict__ Hg, float * __restrict__ partial,
                      int K, int n_out, float beta, float lb) {
    constexpr int BM = 128, BN = 32, BK = 64;
    constexpr int SR = NVFP4 ? 48 : 80;
    constexpr int WPART = BM * SR;
    constexpr int XPART = BN * BK * 2;  // 4096 B
    constexpr int STG = WPART + XPART;
    constexpr int ELD = BM + 4;
    constexpr int SM_BYTES = (2 * STG > BN * ELD * 4) ? 2 * STG : BN * ELD * 4;

    __shared__ __align__(16) uint8_t sm[SM_BYTES];
    __shared__ __half lut[16];

    const int e = blockIdx.y;
    const uint8_t * m = experts[e];
    const int b0 = offsets[e], cnt = offsets[e + 1] - b0;
    if (!m || cnt == 0) return;

    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, t4 = lane & 3;
    if constexpr (NVFP4) {
        if (tid < 16) lut[tid] = __float2half_rn(k_fp4_tc[tid]);
    }

    const int rows_total = GLU ? 2 * n_out : n_out;
    const int row0 = blockIdx.x * (GLU ? 64 : 128);
    const float sc_exp = scale ? scale[e] : 1.f;
    const unsigned sbase = (unsigned) __cvta_generic_to_shared(sm);
    const int nst = K / BK;

    // Weight copy addresses for stage st:
    //   NVFP4: 128 rows * 2 chunks (32 B q) = 256 threads x 1 cpa16, plus tid < 128 copies 4 B of UE4M3 d
    //   Q8_0:  128 rows * 4 chunks (64 B q) = 256 threads x 2 cpa16, plus tid < 128 copies 4 B of fp16 d
    constexpr int NCH = NVFP4 ? 2 : 4;
    constexpr int WIT = (BM * NCH) / 256;
    const uint8_t * wsrc[WIT];
    unsigned wdst[WIT];
    const size_t q_row_bytes = NVFP4 ? size_t(K / 2) : size_t(K);
    const size_t d_off_base = NVFP4 ? size_t(rows_total) * K / 2 : size_t(rows_total) * K;
    const size_t d_row_bytes = NVFP4 ? size_t(K / 16) : size_t(K / 32) * 2;
#pragma unroll
    for (int k = 0; k < WIT; ++k) {
        const int i = tid + k * 256;
        const int lr = i / NCH, c = i % NCH;
        const int mr = GLU ? (lr < 64 ? row0 + lr : row0 + n_out + (lr - 64)) : (row0 + lr);
        wsrc[k] = m + size_t(mr) * q_row_bytes + 16 * c;
        wdst[k] = lr * SR + 16 * c;
    }
    const int mr_d = GLU ? (tid < 64 ? row0 + tid : row0 + n_out + (tid - 64)) : (row0 + tid);
    const uint8_t * dsrc = (tid < BM) ? (m + d_off_base + size_t(mr_d) * d_row_bytes) : nullptr;
    const unsigned ddst = (tid < BM) ? (tid * SR + (NVFP4 ? 32 : 64)) : 0u;

    // Activation copy layout: 256 threads -> 32 tokens x 8 chunks of 16 B (64 halves)
    const int xt = tid >> 3, xc = tid & 7;
    const unsigned xdst = WPART + xt * 128 + ((xc ^ (xt & 7)) << 4);

    // ldmatrix addresses for B fragments (2 pairs of n8 tiles = 32 tokens)
    const unsigned xl = WPART + (8 * (lane >> 4) + (lane & 7)) * 128;
    unsigned xo[4];
#pragma unroll
    for (int G = 0; G < 4; ++G) xo[G] = xl + (((2 * G + ((lane >> 3) & 1)) ^ (lane & 7)) << 4);

    const uint8_t * ra_off = sm + (warp * 16 + g) * SR;
    const uint8_t * rb_off = ra_off + 8 * SR;

    for (int j0 = 0; j0 < cnt; j0 += BN) {
        const bool x_valid = (j0 + xt) < cnt;
        const __half * xbase = x_valid ? (Xg + size_t(b0 + j0 + xt) * K + 8 * xc) : nullptr;

        auto load_stage = [&](int st) {
            const unsigned s = sbase + (st & 1) * STG;
#pragma unroll
            for (int k = 0; k < WIT; ++k) pf::cpa16(s + wdst[k], wsrc[k] + st * (NVFP4 ? 32 : 64));
            if (tid < BM) pf::cpa4(s + ddst, dsrc + st * 4);
            if (x_valid) {
                pf::cpa16(s + xdst, xbase + BK * st);
            } else {
                *reinterpret_cast<uint4 *>(sm + (st & 1) * STG + xdst) = make_uint4(0u, 0u, 0u, 0u);
            }
            pf::cpa_commit();
        };

        float acc[4][4];
#pragma unroll
        for (int ni = 0; ni < 4; ++ni) acc[ni][0] = acc[ni][1] = acc[ni][2] = acc[ni][3] = 0.f;

        load_stage(0);
        for (int st = 0; st < nst; ++st) {
            if (st + 1 < nst) load_stage(st + 1);
            if (st + 1 < nst) pf::cpa_wait<1>();
            else pf::cpa_wait<0>();
            __syncthreads();

            const int buf = st & 1;
            const unsigned sx = sbase + buf * STG;
            const uint8_t * ra = ra_off + buf * STG;
            const uint8_t * rb = rb_off + buf * STG;
#pragma unroll
            for (int G = 0; G < 4; ++G) {
                unsigned a[4];
                expert_frag<NVFP4>(ra, rb, G, t4, lut, a);
#pragma unroll
                for (int np = 0; np < 2; ++np) {
                    unsigned r[4];
                    pf::ldsm_x4(r, sx + xo[G] + np * 16 * 128);
                    hmma16_f32(acc[2 * np], a, r[0], r[1]);
                    hmma16_f32(acc[2 * np + 1], a, r[2], r[3]);
                }
            }
            __syncthreads();
        }

        // Epilogue through shared memory se[32][132]
        float * se = reinterpret_cast<float *>(sm);
        const int rA = warp * 16 + g, rB = rA + 8;
#pragma unroll
        for (int ni = 0; ni < 4; ++ni) {
            const int tk = 8 * ni + 2 * t4;
            se[tk * ELD + rA] = acc[ni][0];
            se[(tk + 1) * ELD + rA] = acc[ni][1];
            se[tk * ELD + rB] = acc[ni][2];
            se[(tk + 1) * ELD + rB] = acc[ni][3];
        }
        __syncthreads();

        if constexpr (GLU) {
            // 32 tokens x 4 groups of 16 outputs = 128 groups
            if (tid < BN * 4) {
                const int tk = tid >> 2, q = tid & 3;
                if (j0 + tk < cnt) {
                    const float * gp = se + tk * ELD + 16 * q;
                    __align__(16) __half hv[16];
#pragma unroll
                    for (int i = 0; i < 16; ++i) {
                        const float v = situ_tc(gp[i] * sc_exp, gp[64 + i] * sc_exp, beta, lb);
                        hv[pf::hperm(i)] = __float2half_rn(fminf(fmaxf(v, -65504.f), 65504.f));
                    }
                    uint4 * dst = reinterpret_cast<uint4 *>(Hg + size_t(b0 + j0 + tk) * n_out + row0 + 16 * q);
                    dst[0] = reinterpret_cast<const uint4 *>(hv)[0];
                    dst[1] = reinterpret_cast<const uint4 *>(hv)[1];
                }
            }
        } else {
            // 32 tokens x 32 float4s = 1024 float4s -> 4 iterations per thread
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                const int i = tid + 256 * k;
                const int tk = i >> 5, c = (i & 31) * 4;
                if (j0 + tk < cnt) {
                    const int en = list[b0 + j0 + tk];
                    const float coeff = wts[en] * sc_exp;
                    const float4 v = *reinterpret_cast<const float4 *>(se + tk * ELD + c);
                    *reinterpret_cast<float4 *>(partial + size_t(en) * n_out + row0 + c) =
                        make_float4(v.x * coeff, v.y * coeff, v.z * coeff, v.w * coeff);
                }
            }
        }
        __syncthreads();
    }
}

// Dense linear-repacked Q8_0 tensor-core GEMM: Y[t][r] = W[r] . X[t]
// Uses the same cp.async + hperm + ldsm_x4 + m16n8k16 pattern as megakernel/prefill_gemm.cuh (with fp32 MMA
// accumulation and linear Q8_0 block order matching expert_kernels.cu's repack_q8_0_kernel).
template <int NB>
__global__ void __launch_bounds__(256)
    dense_q8r_tc_kernel(const uint8_t * __restrict__ w, const __half * __restrict__ Xh, float * __restrict__ Y,
                        int rows, int K, int T) {
    constexpr int BM = 128, BK = 64, SR = 80;
    constexpr int WPART = BM * SR;
    constexpr int XPART = NB * BK * 2;
    constexpr int STG = WPART + XPART;
    constexpr int ELD = BM + 4;
    constexpr int SM_BYTES = (2 * STG > NB * ELD * 4) ? 2 * STG : NB * ELD * 4;
    constexpr int HNI = NB / 8;
    constexpr int XIT = NB * 8 / 256;

    __shared__ __align__(16) uint8_t sm[SM_BYTES];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, t4 = lane & 3;
    const int row0 = blockIdx.x * BM, tok0 = blockIdx.y * NB;
    const unsigned sbase = (unsigned) __cvta_generic_to_shared(sm);
    const int nst = K / BK;

    const uint8_t * wsrc[2];
    unsigned wdst[2];
#pragma unroll
    for (int k = 0; k < 2; ++k) {
        const int i = tid + k * 256;
        const int lr = i >> 2, c = i & 3;
        wsrc[k] = w + size_t(row0 + lr) * K + 16 * c;
        wdst[k] = lr * SR + 16 * c;
    }
    const uint8_t * dsrc = (tid < BM) ? (w + size_t(rows) * K + size_t(row0 + tid) * (K / 32) * 2) : nullptr;
    const unsigned ddst = (tid < BM) ? (tid * SR + 64) : 0u;

    const __half * xsrc[XIT];
    unsigned xdst[XIT];
#pragma unroll
    for (int k = 0; k < XIT; ++k) {
        const int i = tid + k * 256, xt = i >> 3, xc = i & 7;
        xsrc[k] = Xh + size_t(tok0 + xt) * K + 8 * xc;
        xdst[k] = WPART + xt * 128 + ((xc ^ (xt & 7)) << 4);
    }

    const unsigned xl = WPART + (8 * (lane >> 4) + (lane & 7)) * 128;
    unsigned xo[4];
#pragma unroll
    for (int G = 0; G < 4; ++G) xo[G] = xl + (((2 * G + ((lane >> 3) & 1)) ^ (lane & 7)) << 4);

    const uint8_t * ra_off = sm + (warp * 16 + g) * SR;
    const uint8_t * rb_off = ra_off + 8 * SR;

    auto load_stage = [&](int st) {
        const unsigned s = sbase + (st & 1) * STG;
#pragma unroll
        for (int k = 0; k < 2; ++k) pf::cpa16(s + wdst[k], wsrc[k] + st * 64);
        if (tid < BM) pf::cpa4(s + ddst, dsrc + st * 4);
#pragma unroll
        for (int k = 0; k < XIT; ++k) pf::cpa16(s + xdst[k], xsrc[k] + BK * st);
        pf::cpa_commit();
    };

    float acc[HNI][4];
#pragma unroll
    for (int ni = 0; ni < HNI; ++ni) acc[ni][0] = acc[ni][1] = acc[ni][2] = acc[ni][3] = 0.f;

    load_stage(0);
    for (int st = 0; st < nst; ++st) {
        if (st + 1 < nst) load_stage(st + 1);
        if (st + 1 < nst) pf::cpa_wait<1>();
        else pf::cpa_wait<0>();
        __syncthreads();

        const int buf = st & 1;
        const unsigned sx = sbase + buf * STG;
        const uint8_t * ra = ra_off + buf * STG;
        const uint8_t * rb = rb_off + buf * STG;
#pragma unroll
        for (int G = 0; G < 4; ++G) {
            unsigned a[4];
            expert_frag<false>(ra, rb, G, t4, nullptr, a);
#pragma unroll
            for (int np = 0; np < HNI / 2; ++np) {
                unsigned r[4];
                pf::ldsm_x4(r, sx + xo[G] + np * 16 * 128);
                hmma16_f32(acc[2 * np], a, r[0], r[1]);
                hmma16_f32(acc[2 * np + 1], a, r[2], r[3]);
            }
        }
        __syncthreads();
    }

    float * se = reinterpret_cast<float *>(sm);
    const int rA = warp * 16 + g, rB = rA + 8;
#pragma unroll
    for (int ni = 0; ni < HNI; ++ni) {
        const int tk = 8 * ni + 2 * t4;
        se[tk * ELD + rA] = acc[ni][0];
        se[(tk + 1) * ELD + rA] = acc[ni][1];
        se[tk * ELD + rB] = acc[ni][2];
        se[(tk + 1) * ELD + rB] = acc[ni][3];
    }
    __syncthreads();

#pragma unroll
    for (int k = 0; k < NB / 8; ++k) {
        const int i = tid + 256 * k;
        const int tk = i >> 5, c = (i & 31) * 4;
        if (tok0 + tk < T) {
            *reinterpret_cast<float4 *>(Y + size_t(tok0 + tk) * rows + row0 + c) =
                *reinterpret_cast<const float4 *>(se + tk * ELD + c);
        }
    }
}

}  // namespace

extern "C" {

void ik_pf_f32_to_hperm(const float * X, void * Xh, int T, int T_pad, int K, cudaStream_t st) {
    const int n16 = K / 16;
    const dim3 grid((n16 + 15) / 16, T_pad);
    f32_to_hperm_kernel<<<grid, 256, 0, st>>>(X, reinterpret_cast<__half *>(Xh), T, T_pad, K);
}

void ik_pf_gather_hperm(const float * X, const int * list, int k, int n_entries, int K, void * Xg, cudaStream_t st) {
    const int n16 = K / 16;
    const dim3 grid((n16 + 15) / 16, n_entries);
    gather_hperm_kernel<<<grid, 256, 0, st>>>(X, list, k, n_entries, K, reinterpret_cast<__half *>(Xg));
}

int ik_pf_gemm_q8r(const void * w, const void * Xh, float * Y, int rows, int K, int T, cudaStream_t st) {
    if (rows % 128 != 0 || K % 128 != 0) return (int) cudaErrorInvalidValue;
    const auto * b = reinterpret_cast<const uint8_t *>(w);
    const auto * X = reinterpret_cast<const __half *>(Xh);
    if (T <= 32) {
        dense_q8r_tc_kernel<32><<<dim3(rows / 128, (T + 31) / 32), 256, 0, st>>>(b, X, Y, rows, K, T);
    } else {
        dense_q8r_tc_kernel<64><<<dim3(rows / 128, (T + 63) / 64), 256, 0, st>>>(b, X, Y, rows, K, T);
    }
    return (int) cudaGetLastError();
}

int ik_pf_gemm_gguf(int gtype, const void * w, const void * Xh, float * Y, int rows, int K, int T, cudaStream_t st) {
    if (rows % 128 != 0 || K % 256 != 0) return (int) cudaErrorInvalidValue;
    const auto * b = reinterpret_cast<const uint8_t *>(w);
    const auto * X = reinterpret_cast<const __half *>(Xh);
    if (gtype == pf::TQ4K) {
        pf::WMat W{b, nullptr, pf::TQ4K, K, rows, K / 256 * 144, 0};
        if (T <= 32) return (int) launch_mmh<pf::TQ4K, 32>(W, X, Y, rows, T, st);
        if (T <= 64) return (int) launch_mmh<pf::TQ4K, 64>(W, X, Y, rows, T, st);
        return (int) launch_mmh<pf::TQ4K, 128>(W, X, Y, rows, T, st);
    }
    if (gtype == pf::TQ5K) {
        pf::WMat W{b, nullptr, pf::TQ5K, K, rows, K / 256 * 176, 0};
        if (T <= 32) return (int) launch_mmh<pf::TQ5K, 32>(W, X, Y, rows, T, st);
        if (T <= 64) return (int) launch_mmh<pf::TQ5K, 64>(W, X, Y, rows, T, st);
        return (int) launch_mmh<pf::TQ5K, 128>(W, X, Y, rows, T, st);
    }
    return (int) cudaErrorNotSupported;
}

int ik_pf_experts_tc(int type, const void * const * gu, const void * const * dn, const float * gu_scale, const float * dn_scale,
                     const int * offsets, const int * list, const float * wts, const void * Xg, void * Hg, float * partial,
                     int n_embd, int n_ff, int n_expert, float beta, float lb, cudaStream_t st) {
    if (n_embd % 128 != 0 || n_ff % 64 != 0 || n_ff % 128 != 0) return (int) cudaErrorInvalidValue;
    const auto * g = reinterpret_cast<const uint8_t * const *>(gu);
    const auto * d = reinterpret_cast<const uint8_t * const *>(dn);
    const auto * X = reinterpret_cast<const __half *>(Xg);
    auto * H = reinterpret_cast<__half *>(Hg);
    const dim3 g_gu(n_ff / 64, n_expert);
    const dim3 g_dn(n_embd / 128, n_expert);
    if (type == 0) {
        experts_tc_kernel<true, true><<<g_gu, 256, 0, st>>>(g, gu_scale, offsets, list, wts, X, H, partial, n_embd, n_ff, beta, lb);
        experts_tc_kernel<true, false><<<g_dn, 256, 0, st>>>(d, dn_scale, offsets, list, wts, H, nullptr, partial, n_ff, n_embd, beta, lb);
    } else {
        experts_tc_kernel<false, true><<<g_gu, 256, 0, st>>>(g, gu_scale, offsets, list, wts, X, H, partial, n_embd, n_ff, beta, lb);
        experts_tc_kernel<false, false><<<g_dn, 256, 0, st>>>(d, dn_scale, offsets, list, wts, H, nullptr, partial, n_ff, n_embd, beta, lb);
    }
    return (int) cudaGetLastError();
}

}  // extern "C"
