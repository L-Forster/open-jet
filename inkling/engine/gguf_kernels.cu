// Matrix-vector and expert kernels for the ggml types Inkling GGUFs use besides NVFP4 and Q8_0 (which have repacked
// fast paths in expert_kernels.cu): Q4_K, Q5_K, Q6_K, IQ4_XS, IQ4_NL, Q5_0, and Q8_0 / NVFP4 in GGUF layout. Weights
// stay in their GGUF blocks. Each lane of a warp decodes 32-value sub-blocks exactly as ggml's dequantize_row_*
// (ggml-quants.c) and multiplies them with the fp32 input; accumulation is fp32.

#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include "iq_tables.h"

namespace {

constexpr int WARP = 32;

enum : int {
    T_Q5_0 = 6, T_Q8_0 = 8, T_Q4_K = 12, T_Q5_K = 13, T_Q6_K = 14,
    T_IQ2_XXS = 16, T_IQ2_XS = 17, T_IQ3_XXS = 18, T_IQ4_NL = 20, T_IQ2_S = 22, T_IQ4_XS = 23,
    T_NVFP4 = 40
};

__device__ __forceinline__ float h2f(const uint8_t * p) { return __half2float(*reinterpret_cast<const __half *>(p)); }

__device__ __forceinline__ uint32_t ldg_u32_unaligned(const uint8_t * p) {
    return uint32_t(__ldg(p)) | (uint32_t(__ldg(p + 1)) << 8)
         | (uint32_t(__ldg(p + 2)) << 16) | (uint32_t(__ldg(p + 3)) << 24);
}

__device__ __forceinline__ float iq4_half(uint32_t g, uint32_t s4, float4 xv) {
    const float w0 = float(g & 0xFFu), w1 = float((g >> 8) & 0xFFu);
    const float w2 = float((g >> 16) & 0xFFu), w3 = float(g >> 24);
    return ((s4 & 1u) ? -w0 : w0) * xv.x
         + ((s4 & 2u) ? -w1 : w1) * xv.y
         + ((s4 & 4u) ? -w2 : w2) * xv.z
         + ((s4 & 8u) ? -w3 : w3) * xv.w;
}

__device__ __forceinline__ float iq8_dot_u32(uint32_t g_lo, uint32_t g_hi, uint32_t signs, float db, float4 x0, float4 x1) {
    return db * (iq4_half(g_lo, signs, x0) + iq4_half(g_hi, signs >> 4, x1));
}

__device__ __forceinline__ float iq8_dot_v4(uint64_t grid, uint8_t signs, float db, float4 x0, float4 x1) {
    return iq8_dot_u32(uint32_t(grid), uint32_t(grid >> 32), signs, db, x0, x1);
}

__device__ __forceinline__ uint32_t ldg_u32_2align(const uint8_t * p) {
    const uint16_t * h = reinterpret_cast<const uint16_t *>(p);
    return uint32_t(__ldg(h)) | (uint32_t(__ldg(h + 1)) << 16);
}

__device__ __forceinline__ float iq8_dot(uint64_t grid, uint8_t signs, float db, const float * x) {
    const float4 x0 = __ldg(reinterpret_cast<const float4 *>(x));
    const float4 x1 = __ldg(reinterpret_cast<const float4 *>(x + 4));
    return iq8_dot_v4(grid, signs, db, x0, x1);
}

template <int NT>
__device__ __forceinline__ void iq8_dot_multi(uint64_t grid, uint8_t signs, float db, const float * const * xs,
                                              int cnt, int off, float acc[NT]) {
    float w[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const float mag = float((grid >> (8 * j)) & 0xFFu);
        w[j] = db * ((signs & (1u << j)) ? -mag : mag);
    }
#pragma unroll
    for (int t = 0; t < NT; ++t) {
        if (t < cnt) {
            const float * xp = xs[t] + off;
            const float4 x0 = __ldg(reinterpret_cast<const float4 *>(xp));
            const float4 x1 = __ldg(reinterpret_cast<const float4 *>(xp + 4));
            acc[t] += w[0] * x0.x + w[1] * x0.y + w[2] * x0.z + w[3] * x0.w
                    + w[4] * x1.x + w[5] * x1.y + w[6] * x1.z + w[7] * x1.w;
        }
    }
}

__device__ const int8_t k_iq4nl[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};
__device__ const int8_t k_fp4x2[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};

__device__ __forceinline__ float ue4m3_half(uint32_t x) {
    x &= 0x7F;
    if (x == 0x7F) return 0.f;
    if (x < 8) return float(x) * 0.0009765625f;
    return __int_as_float(int((x << 20) + (119u << 23)));
}

// bytes per block and values per block
__host__ __device__ inline void block_shape(int type, int & bytes, int & values) {
    switch (type) {
        case T_Q5_0: bytes = 22; values = 32; break;
        case T_Q8_0: bytes = 34; values = 32; break;
        case T_Q4_K: bytes = 144; values = 256; break;
        case T_Q5_K: bytes = 176; values = 256; break;
        case T_Q6_K: bytes = 210; values = 256; break;
        case T_IQ2_XXS: bytes = 66; values = 256; break;
        case T_IQ2_XS: bytes = 74; values = 256; break;
        case T_IQ3_XXS: bytes = 98; values = 256; break;
        case T_IQ4_NL: bytes = 18; values = 32; break;
        case T_IQ2_S: bytes = 82; values = 256; break;
        case T_IQ4_XS: bytes = 136; values = 256; break;
        case T_NVFP4: bytes = 36; values = 64; break;
        default: bytes = 0; values = 1;
    }
}

__device__ __forceinline__ void k_scale_min(int j, const uint8_t * q, int & sc, int & m) {
    if (j < 4) { sc = q[j] & 63; m = q[j + 4] & 63; }
    else { sc = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4); m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4); }
}

// the 32 values of sub-block sb (0-based, in values / 32) of a row
template <int TYPE>
__device__ __forceinline__ void dequant32(const uint8_t * row, int sb, float * y) {
    if constexpr (TYPE == T_Q8_0) {
        const uint8_t * b = row + sb * 34;
        const float d = h2f(b);
        for (int j = 0; j < 32; ++j) y[j] = d * float(int8_t(b[2 + j]));
    } else if constexpr (TYPE == T_Q5_0) {
        const uint8_t * b = row + sb * 22;
        const float d = h2f(b);
        uint32_t qh;
        memcpy(&qh, b + 2, 4);
        const uint8_t * qs = b + 6;
        for (int j = 0; j < 16; ++j) {
            const int x0 = ((qs[j] & 0x0F) | (((qh >> j) << 4) & 0x10)) - 16;
            const int x1 = ((qs[j] >> 4) | ((qh >> (j + 12)) & 0x10)) - 16;
            y[j] = d * x0;
            y[j + 16] = d * x1;
        }
    } else if constexpr (TYPE == T_IQ4_NL) {
        const uint8_t * b = row + sb * 18;
        const float d = h2f(b);
        for (int j = 0; j < 16; ++j) {
            y[j] = d * k_iq4nl[b[2 + j] & 0xF];
            y[j + 16] = d * k_iq4nl[b[2 + j] >> 4];
        }
    } else if constexpr (TYPE == T_IQ4_XS) {
        const uint8_t * b = row + (sb / 8) * 136;
        const int ib = sb % 8;
        const float d = h2f(b);
        const uint16_t sh = uint16_t(b[2]) | (uint16_t(b[3]) << 8);
        const int ls = ((b[4 + ib / 2] >> (4 * (ib % 2))) & 0xF) | (((sh >> (2 * ib)) & 3) << 4);
        const float dl = d * (ls - 32);
        const uint8_t * qs = b + 8 + ib * 16;
        for (int j = 0; j < 16; ++j) {
            y[j] = dl * k_iq4nl[qs[j] & 0xF];
            y[j + 16] = dl * k_iq4nl[qs[j] >> 4];
        }
    } else if constexpr (TYPE == T_Q4_K) {
        const uint8_t * b = row + (sb / 8) * 144;
        const int is = sb % 8, p = is / 2;
        const float d = h2f(b), dmin = h2f(b + 2);
        int sc, m;
        k_scale_min(is, b + 4, sc, m);
        const float d1 = d * sc, m1 = dmin * m;
        const uint8_t * q = b + 16 + 32 * p;
        if (is & 1) for (int l = 0; l < 32; ++l) y[l] = d1 * (q[l] >> 4) - m1;
        else        for (int l = 0; l < 32; ++l) y[l] = d1 * (q[l] & 0xF) - m1;
    } else if constexpr (TYPE == T_Q5_K) {
        const uint8_t * b = row + (sb / 8) * 176;
        const int is = sb % 8, p = is / 2;
        const float d = h2f(b), dmin = h2f(b + 2);
        int sc, m;
        k_scale_min(is, b + 4, sc, m);
        const float d1 = d * sc, m1 = dmin * m;
        const uint8_t * qh = b + 16;
        const uint8_t * ql = b + 48 + 32 * p;
        for (int l = 0; l < 32; ++l) {
            const int lo = (is & 1) ? (ql[l] >> 4) : (ql[l] & 0xF);
            y[l] = d1 * (lo + (((qh[l] >> is) & 1) << 4)) - m1;
        }
    } else if constexpr (TYPE == T_Q6_K) {
        const uint8_t * b = row + (sb / 8) * 210;
        const int k8 = sb % 8, half = k8 / 4, k = k8 % 4;
        const uint8_t * ql = b + 64 * half;
        const uint8_t * qh = b + 128 + 32 * half;
        const int8_t * sc = reinterpret_cast<const int8_t *>(b + 192) + 8 * half;
        const float d = h2f(b + 208);
        for (int l = 0; l < 32; ++l) {
            const int lo = (k & 1) ? ql[l + 32] : ql[l];
            const int q = (((k & 2) ? (lo >> 4) : (lo & 0xF)) | (((qh[l] >> (2 * k)) & 3) << 4)) - 32;
            y[l] = d * sc[l / 16 + 2 * k] * q;
        }
    } else if constexpr (TYPE == T_NVFP4) {
        const uint8_t * b = row + (sb / 2) * 36;
        const int h = sb % 2;  // values 32h .. 32h + 31: sub-blocks 2h and 2h + 1
        for (int s2 = 0; s2 < 2; ++s2) {
            const int s = 2 * h + s2;
            const float d = ue4m3_half(b[s]);
            const uint8_t * qs = b + 4 + s * 8;
            for (int j = 0; j < 8; ++j) {
                y[s2 * 16 + j] = d * k_fp4x2[qs[j] & 0xF];
                y[s2 * 16 + 8 + j] = d * k_fp4x2[qs[j] >> 4];
            }
        }
    }
}

// partial dot of a GGUF row (n values) with x; coalesced warp-wide loops for K-quants & IQ4_XS (no local-memory spills)
template <int TYPE>
__device__ __forceinline__ float dot_row(const uint8_t * row, const float * __restrict__ x, int n, int lane) {
    float acc = 0.f;
    if constexpr (TYPE == T_Q6_K) {
        const int is = lane >> 4;
        for (int sb = 0; sb < n / 256; ++sb) {
            const uint8_t * b = row + size_t(sb) * 210;
            const float d = h2f(b + 208);
            const float * xb = x + sb * 256;
            float sum = 0.f;
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const uint8_t ql0 = __ldg(b + 64 * h + lane);
                const uint8_t ql1 = __ldg(b + 64 * h + 32 + lane);
                const uint8_t qhv = __ldg(b + 128 + 32 * h + lane);
                const int8_t * sc = reinterpret_cast<const int8_t *>(b + 192) + 8 * h;
                const int q0 = int((ql0 & 0xF) | ((qhv & 3) << 4)) - 32;
                const int q1 = int((ql1 & 0xF) | (((qhv >> 2) & 3) << 4)) - 32;
                const int q2 = int((ql0 >> 4)  | (((qhv >> 4) & 3) << 4)) - 32;
                const int q3 = int((ql1 >> 4)  | (((qhv >> 6) & 3) << 4)) - 32;
                const float * xh = xb + 128 * h + lane;
                sum += float(__ldg(sc + is + 0)) * float(q0) * __ldg(xh)
                     + float(__ldg(sc + is + 2)) * float(q1) * __ldg(xh + 32)
                     + float(__ldg(sc + is + 4)) * float(q2) * __ldg(xh + 64)
                     + float(__ldg(sc + is + 6)) * float(q3) * __ldg(xh + 96);
            }
            acc += d * sum;
        }
        return acc;
    } else if constexpr (TYPE == T_Q4_K) {
        for (int sb = 0; sb < n / 256; ++sb) {
            const uint8_t * b = row + size_t(sb) * 144;
            const float d = h2f(b), dmin = h2f(b + 2);
            const float * xb = x + sb * 256;
#pragma unroll
            for (int p = 0; p < 4; ++p) {
                int sc0, m0, sc1, m1;
                k_scale_min(2 * p, b + 4, sc0, m0);
                k_scale_min(2 * p + 1, b + 4, sc1, m1);
                const uint8_t q = __ldg(b + 16 + 32 * p + lane);
                const float x0 = __ldg(xb + 64 * p + lane);
                const float x1 = __ldg(xb + 64 * p + 32 + lane);
                acc += (d * float(sc0) * float(q & 0xF) - dmin * float(m0)) * x0
                     + (d * float(sc1) * float(q >> 4)  - dmin * float(m1)) * x1;
            }
        }
        return acc;
    } else if constexpr (TYPE == T_Q5_K) {
        for (int sb = 0; sb < n / 256; ++sb) {
            const uint8_t * b = row + size_t(sb) * 176;
            const float d = h2f(b), dmin = h2f(b + 2);
            const uint8_t qh = __ldg(b + 16 + lane);
            const float * xb = x + sb * 256;
#pragma unroll
            for (int p = 0; p < 4; ++p) {
                int sc0, m0, sc1, m1;
                k_scale_min(2 * p, b + 4, sc0, m0);
                k_scale_min(2 * p + 1, b + 4, sc1, m1);
                const uint8_t ql = __ldg(b + 48 + 32 * p + lane);
                const int v0 = (ql & 0xF) | (((qh >> (2 * p)) & 1) << 4);
                const int v1 = (ql >> 4)  | (((qh >> (2 * p + 1)) & 1) << 4);
                const float x0 = __ldg(xb + 64 * p + lane);
                const float x1 = __ldg(xb + 64 * p + 32 + lane);
                acc += (d * float(sc0) * float(v0) - dmin * float(m0)) * x0
                     + (d * float(sc1) * float(v1) - dmin * float(m1)) * x1;
            }
        }
        return acc;
    } else if constexpr (TYPE == T_IQ4_XS) {
        const int half = lane >> 4, j = lane & 15;
        for (int sb = 0; sb < n / 256; ++sb) {
            const uint8_t * b = row + size_t(sb) * 136;
            const float d = h2f(b);
            const uint16_t sh = uint16_t(__ldg(b + 2)) | (uint16_t(__ldg(b + 3)) << 8);
            const float * xb = x + sb * 256;
#pragma unroll
            for (int p = 0; p < 4; ++p) {
                const int ib = 2 * p + half;
                const int ls = ((__ldg(b + 4 + ib / 2) >> (4 * (ib & 1))) & 0xF) | (((sh >> (2 * ib)) & 3) << 4);
                const float dl = d * float(ls - 32);
                const uint8_t q = __ldg(b + 8 + ib * 16 + j);
                acc += dl * (float(k_iq4nl[q & 0xF]) * __ldg(xb + ib * 32 + j)
                           + float(k_iq4nl[q >> 4])  * __ldg(xb + ib * 32 + 16 + j));
            }
        }
        return acc;
    } else if constexpr (TYPE == T_Q5_0) {
        const int half = lane >> 4, j = lane & 15;
        for (int sb = 0; sb < n / 32; sb += 2) {
            const uint8_t * b = row + size_t(sb + half) * 22;
            const float d = h2f(b);
            const uint32_t qh = uint32_t(__ldg(b + 2)) | (uint32_t(__ldg(b + 3)) << 8)
                              | (uint32_t(__ldg(b + 4)) << 16) | (uint32_t(__ldg(b + 5)) << 24);
            const uint8_t q = __ldg(b + 6 + j);
            const int x0 = int((q & 0x0F) | (((qh >> j) << 4) & 0x10)) - 16;
            const int x1 = int((q >> 4)   | ((qh >> (j + 12)) & 0x10)) - 16;
            const float * xb = x + (sb + half) * 32;
            acc += d * (float(x0) * __ldg(xb + j) + float(x1) * __ldg(xb + 16 + j));
        }
        return acc;
    } else if constexpr (TYPE == T_IQ4_NL) {
        const int half = lane >> 4, j = lane & 15;
        for (int sb = 0; sb < n / 32; sb += 2) {
            const uint8_t * b = row + size_t(sb + half) * 18;
            const float d = h2f(b);
            const uint8_t q = __ldg(b + 2 + j);
            const float * xb = x + (sb + half) * 32;
            acc += d * (float(k_iq4nl[q & 0xF]) * __ldg(xb + j) + float(k_iq4nl[q >> 4]) * __ldg(xb + 16 + j));
        }
        return acc;
    } else if constexpr (TYPE == T_IQ2_XXS) {
        const int ib32 = lane >> 2, l = lane & 3;
        for (int sb = 0; sb < n / 256; ++sb) {
            const uint8_t * b = row + size_t(sb) * 66;
            const float d = h2f(b);
            const uint8_t * g = b + 2 + 8 * ib32;
            const uint32_t aux1 = ldg_u32_unaligned(g + 4);
            const float db = d * (0.5f + float(aux1 >> 28)) * 0.25f;
            const uint64_t grid = __ldg(&k_iq2xxs_grid[__ldg(g + l)]);
            const uint8_t signs = __ldg(&k_ksigns_iq2xs[(aux1 >> (7 * l)) & 127u]);
            acc += iq8_dot(grid, signs, db, x + sb * 256 + lane * 8);
        }
        return acc;
    } else if constexpr (TYPE == T_IQ2_XS) {
        const int ib32 = lane >> 2, l = lane & 3;
        for (int sb = 0; sb < n / 256; ++sb) {
            const uint8_t * b = row + size_t(sb) * 74;
            const float d = h2f(b);
            const uint8_t sc_b = __ldg(b + 66 + ib32);
            const float db = d * (0.5f + float(l < 2 ? (sc_b & 0xF) : (sc_b >> 4))) * 0.25f;
            const uint16_t q16 = uint16_t(__ldg(b + 2 + 2 * lane)) | (uint16_t(__ldg(b + 3 + 2 * lane)) << 8);
            const uint64_t grid = __ldg(&k_iq2xs_grid[q16 & 511u]);
            const uint8_t signs = __ldg(&k_ksigns_iq2xs[q16 >> 9]);
            acc += iq8_dot(grid, signs, db, x + sb * 256 + lane * 8);
        }
        return acc;
    } else if constexpr (TYPE == T_IQ2_S) {
        const int ib32 = lane >> 2, l = lane & 3;
        for (int sb = 0; sb < n / 256; ++sb) {
            const uint8_t * b = row + size_t(sb) * 82;
            const float d = h2f(b);
            const uint8_t sc_b = __ldg(b + 74 + ib32);
            const float db = d * (0.5f + float(l < 2 ? (sc_b & 0xF) : (sc_b >> 4))) * 0.25f;
            const uint32_t qs_b = __ldg(b + 2 + lane);
            const uint8_t signs = __ldg(b + 34 + lane);
            const uint32_t qh_b = __ldg(b + 66 + ib32);
            const uint32_t idx = qs_b | (((qh_b >> (2 * l)) & 3u) << 8);
            const uint64_t grid = __ldg(&k_iq2s_grid[idx]);
            acc += iq8_dot(grid, signs, db, x + sb * 256 + lane * 8);
        }
        return acc;
    } else if constexpr (TYPE == T_IQ3_XXS) {
        const int ib32 = lane >> 2, l = lane & 3;
        for (int sb = 0; sb < n / 256; ++sb) {
            const uint8_t * b = row + size_t(sb) * 98;
            const float d = h2f(b);
            const uint32_t aux32 = ldg_u32_unaligned(b + 66 + 4 * ib32);
            const float db = d * (0.5f + float(aux32 >> 28)) * 0.5f;
            const uint8_t signs = __ldg(&k_ksigns_iq2xs[(aux32 >> (7 * l)) & 127u]);
            const uint64_t grid = uint64_t(__ldg(&k_iq3xxs_grid[__ldg(b + 2 + 2 * lane)]))
                                | (uint64_t(__ldg(&k_iq3xxs_grid[__ldg(b + 3 + 2 * lane)])) << 32);
            acc += iq8_dot(grid, signs, db, x + sb * 256 + lane * 8);
        }
        return acc;
    } else if constexpr (TYPE == T_NVFP4) {
        const int s = lane >> 3, j = lane & 7;
        for (int sb = 0; sb < n / 64; ++sb) {
            const uint8_t * b = row + size_t(sb) * 36;
            const float d = ue4m3_half(__ldg(b + s));
            const uint8_t q = __ldg(b + 4 + s * 8 + j);
            const float * xb = x + sb * 64 + s * 16;
            acc += d * (float(__ldg(&k_fp4x2[q & 0xF])) * __ldg(xb + j) + float(__ldg(&k_fp4x2[q >> 4])) * __ldg(xb + 8 + j));
        }
        return acc;
    } else if constexpr (TYPE == T_Q8_0) {
        for (int sb = 0; sb < n / 32; ++sb) {
            const uint8_t * b = row + size_t(sb) * 34;
            acc += h2f(b) * float(int8_t(__ldg(b + 2 + lane))) * __ldg(x + sb * 32 + lane);
        }
        return acc;
    } else {
        float y[32];
        for (int sb = lane; sb < n / 32; sb += WARP) {
            dequant32<TYPE>(row, sb, y);
            const float4 * xv = reinterpret_cast<const float4 *>(x + sb * 32);
            float a0 = 0.f, a1 = 0.f;
#pragma unroll
            for (int j = 0; j < 8; j += 2) {
                const float4 p = __ldg(xv + j), q = __ldg(xv + j + 1);
                a0 += y[4 * j] * p.x + y[4 * j + 1] * p.y + y[4 * j + 2] * p.z + y[4 * j + 3] * p.w;
                a1 += y[4 * j + 4] * q.x + y[4 * j + 5] * q.y + y[4 * j + 6] * q.z + y[4 * j + 7] * q.w;
            }
            acc += a0 + a1;
        }
        return acc;
    }
}

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = WARP / 2; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

__host__ __device__ inline size_t row_bytes(int type, int n) {
    int b, v;
    block_shape(type, b, v);
    return size_t(n) / v * b;
}

// register-cached x for K-quant matvec (Q6_K, Q4_K, Q5_K): each lane holds x[sb*256 + 32*k + lane] for k = 0..7
template <int TYPE, int NB>
__global__ void __launch_bounds__(256) matvec_k_reg_kernel(const uint8_t * __restrict__ w, const float * __restrict__ x,
                                                           float * __restrict__ y, int rows) {
    const int lane = threadIdx.x % WARP;
    float xr[NB][8];
#pragma unroll
    for (int sb = 0; sb < NB; ++sb) {
        const float * xb = x + sb * 256 + lane;
#pragma unroll
        for (int k = 0; k < 8; ++k) xr[sb][k] = __ldg(xb + k * 32);
    }
    const size_t rb = row_bytes(TYPE, NB * 256);
    const int stride = gridDim.x * (blockDim.x / WARP);
    const int is = lane >> 4;
    for (int r = blockIdx.x * (blockDim.x / WARP) + threadIdx.x / WARP; r < rows; r += stride) {
        const uint8_t * row = w + size_t(r) * rb;
        float acc = 0.f;
        if constexpr (TYPE == T_Q6_K) {
#pragma unroll
            for (int sb = 0; sb < NB; ++sb) {
                const uint8_t * b = row + size_t(sb) * 210;
                const float d = h2f(b + 208);
                float sum = 0.f;
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    const uint8_t ql0 = __ldg(b + 64 * h + lane);
                    const uint8_t ql1 = __ldg(b + 64 * h + 32 + lane);
                    const uint8_t qhv = __ldg(b + 128 + 32 * h + lane);
                    const int8_t * sc = reinterpret_cast<const int8_t *>(b + 192) + 8 * h;
                    const int q0 = int((ql0 & 0xF) | ((qhv & 3) << 4)) - 32;
                    const int q1 = int((ql1 & 0xF) | (((qhv >> 2) & 3) << 4)) - 32;
                    const int q2 = int((ql0 >> 4)  | (((qhv >> 4) & 3) << 4)) - 32;
                    const int q3 = int((ql1 >> 4)  | (((qhv >> 6) & 3) << 4)) - 32;
                    sum += float(__ldg(sc + is + 0)) * float(q0) * xr[sb][4 * h + 0]
                         + float(__ldg(sc + is + 2)) * float(q1) * xr[sb][4 * h + 1]
                         + float(__ldg(sc + is + 4)) * float(q2) * xr[sb][4 * h + 2]
                         + float(__ldg(sc + is + 6)) * float(q3) * xr[sb][4 * h + 3];
                }
                acc += d * sum;
            }
        } else if constexpr (TYPE == T_Q4_K) {
#pragma unroll
            for (int sb = 0; sb < NB; ++sb) {
                const uint8_t * b = row + size_t(sb) * 144;
                const float d = h2f(b), dmin = h2f(b + 2);
#pragma unroll
                for (int p = 0; p < 4; ++p) {
                    int sc0, m0, sc1, m1;
                    k_scale_min(2 * p, b + 4, sc0, m0);
                    k_scale_min(2 * p + 1, b + 4, sc1, m1);
                    const uint8_t q = __ldg(b + 16 + 32 * p + lane);
                    acc += (d * float(sc0) * float(q & 0xF) - dmin * float(m0)) * xr[sb][2 * p]
                         + (d * float(sc1) * float(q >> 4)  - dmin * float(m1)) * xr[sb][2 * p + 1];
                }
            }
        } else if constexpr (TYPE == T_Q5_K) {
#pragma unroll
            for (int sb = 0; sb < NB; ++sb) {
                const uint8_t * b = row + size_t(sb) * 176;
                const float d = h2f(b), dmin = h2f(b + 2);
                const uint8_t qh = __ldg(b + 16 + lane);
#pragma unroll
                for (int p = 0; p < 4; ++p) {
                    int sc0, m0, sc1, m1;
                    k_scale_min(2 * p, b + 4, sc0, m0);
                    k_scale_min(2 * p + 1, b + 4, sc1, m1);
                    const uint8_t ql = __ldg(b + 48 + 32 * p + lane);
                    const int v0 = (ql & 0xF) | (((qh >> (2 * p)) & 1) << 4);
                    const int v1 = (ql >> 4)  | (((qh >> (2 * p + 1)) & 1) << 4);
                    acc += (d * float(sc0) * float(v0) - dmin * float(m0)) * xr[sb][2 * p]
                         + (d * float(sc1) * float(v1) - dmin * float(m1)) * xr[sb][2 * p + 1];
                }
            }
        }
        const float v = warp_sum(acc);
        if (lane == 0) y[r] = v;
    }
}

template <int TYPE>
__global__ void matvec_kernel(const uint8_t * __restrict__ w, const float * __restrict__ x, float * __restrict__ y, int rows, int n) {
    const int lane = threadIdx.x % WARP;
    const size_t rb = row_bytes(TYPE, n);
    const int stride = gridDim.x * (blockDim.x / WARP);
    for (int r = blockIdx.x * (blockDim.x / WARP) + threadIdx.x / WARP; r < rows; r += stride) {
        const float v = warp_sum(dot_row<TYPE>(w + size_t(r) * rb, x, n, lane));
        if (lane == 0) y[r] = v;
    }
}

__device__ __forceinline__ float situ(float g, float u, float beta, float lb) {
    const float act = beta * tanhf(g / beta) * (1.f / (1.f + expf(-g)));
    if (lb > 0.f) u = lb * tanhf(u / lb);
    return act * u;
}

// Decode 2 rows (gate + up) sharing a single float4 load of shared-memory `xs` and __shared__ codebooks
template <int TYPE>
__global__ void __launch_bounds__(256, 2) gate_up_kernel(const uint8_t * const * __restrict__ experts, const float * __restrict__ scale,
                                                         const int32_t * __restrict__ ids, const float * __restrict__ x, float * __restrict__ h,
                                                         int n_embd, int n_ff, float beta, float lb) {
    const int k = blockIdx.y, e = ids[k];
    const uint8_t * m = experts[e];
    if (!m) return;
    const int lane = threadIdx.x % WARP;
    const size_t rb = row_bytes(TYPE, n_embd);
    const float s = scale ? scale[e] : 1.f;
    const int stride = gridDim.x * (blockDim.x / WARP);
    if constexpr (TYPE == T_IQ3_XXS) {
        if (n_embd <= 2304) {
            __shared__ __align__(16) float xs[2304];
            __shared__ uint32_t s_grid[256];
            __shared__ uint8_t s_signs[128];
            for (int i = threadIdx.x; i < n_embd; i += blockDim.x) xs[i] = __ldg(x + i);
            s_grid[threadIdx.x] = k_iq3xxs_grid[threadIdx.x];
            if (threadIdx.x < 128) s_signs[threadIdx.x] = k_ksigns_iq2xs[threadIdx.x];
            __syncthreads();
            const int ib32 = lane >> 2, l = lane & 3;
            for (int r = blockIdx.x * (blockDim.x / WARP) + threadIdx.x / WARP; r < n_ff; r += stride) {
                const uint8_t * r0 = m + size_t(r) * rb, * r1 = m + size_t(r + n_ff) * rb;
                float ag = 0.f, au = 0.f;
                for (int sb = 0; sb < n_embd / 256; ++sb) {
                    const float * xp = xs + sb * 256 + lane * 8;
                    const float4 x0 = *reinterpret_cast<const float4 *>(xp);
                    const float4 x1 = *reinterpret_cast<const float4 *>(xp + 4);
                    const uint8_t * b0 = r0 + size_t(sb) * 98, * b1 = r1 + size_t(sb) * 98;
                    const uint32_t aux0 = ldg_u32_2align(b0 + 66 + 4 * ib32);
                    const uint32_t aux1 = ldg_u32_2align(b1 + 66 + 4 * ib32);
                    const uint16_t p0 = __ldg(reinterpret_cast<const uint16_t *>(b0 + 2 + 2 * lane));
                    const uint16_t p1 = __ldg(reinterpret_cast<const uint16_t *>(b1 + 2 + 2 * lane));
                    ag += iq8_dot_u32(s_grid[p0 & 0xFFu], s_grid[p0 >> 8], s_signs[(aux0 >> (7 * l)) & 127u],
                                      h2f(b0) * (0.5f + float(aux0 >> 28)) * 0.5f, x0, x1);
                    au += iq8_dot_u32(s_grid[p1 & 0xFFu], s_grid[p1 >> 8], s_signs[(aux1 >> (7 * l)) & 127u],
                                      h2f(b1) * (0.5f + float(aux1 >> 28)) * 0.5f, x0, x1);
                }
                const float g = warp_sum(ag), u = warp_sum(au);
                if (lane == 0) h[size_t(k) * n_ff + r] = situ(g * s, u * s, beta, lb);
            }
            return;
        }
    } else if constexpr (TYPE == T_IQ2_S) {
        if (n_embd <= 2304) {
            __shared__ __align__(16) float xs[2304];
            __shared__ uint2 s_grid[1024];
            for (int i = threadIdx.x; i < n_embd; i += blockDim.x) xs[i] = __ldg(x + i);
            for (int i = threadIdx.x; i < 1024; i += blockDim.x) {
                const uint64_t g = k_iq2s_grid[i];
                s_grid[i] = make_uint2(uint32_t(g), uint32_t(g >> 32));
            }
            __syncthreads();
            const int ib32 = lane >> 2, l = lane & 3;
            for (int r = blockIdx.x * (blockDim.x / WARP) + threadIdx.x / WARP; r < n_ff; r += stride) {
                const uint8_t * r0 = m + size_t(r) * rb, * r1 = m + size_t(r + n_ff) * rb;
                float ag = 0.f, au = 0.f;
                for (int sb = 0; sb < n_embd / 256; ++sb) {
                    const float * xp = xs + sb * 256 + lane * 8;
                    const float4 x0 = *reinterpret_cast<const float4 *>(xp);
                    const float4 x1 = *reinterpret_cast<const float4 *>(xp + 4);
                    const uint8_t * b0 = r0 + size_t(sb) * 82, * b1 = r1 + size_t(sb) * 82;
                    const uint8_t sc0 = __ldg(b0 + 74 + ib32), sc1 = __ldg(b1 + 74 + ib32);
                    const uint32_t idx0 = uint32_t(__ldg(b0 + 2 + lane)) | (((uint32_t(__ldg(b0 + 66 + ib32)) >> (2 * l)) & 3u) << 8);
                    const uint32_t idx1 = uint32_t(__ldg(b1 + 2 + lane)) | (((uint32_t(__ldg(b1 + 66 + ib32)) >> (2 * l)) & 3u) << 8);
                    const uint2 g0 = s_grid[idx0], g1 = s_grid[idx1];
                    ag += iq8_dot_u32(g0.x, g0.y, __ldg(b0 + 34 + lane),
                                      h2f(b0) * (0.5f + float(l < 2 ? (sc0 & 0xF) : (sc0 >> 4))) * 0.25f, x0, x1);
                    au += iq8_dot_u32(g1.x, g1.y, __ldg(b1 + 34 + lane),
                                      h2f(b1) * (0.5f + float(l < 2 ? (sc1 & 0xF) : (sc1 >> 4))) * 0.25f, x0, x1);
                }
                const float g = warp_sum(ag), u = warp_sum(au);
                if (lane == 0) h[size_t(k) * n_ff + r] = situ(g * s, u * s, beta, lb);
            }
            return;
        }
    } else if constexpr (TYPE == T_IQ2_XS) {
        if (n_embd <= 2304) {
            __shared__ __align__(16) float xs[2304];
            __shared__ uint2 s_grid[512];
            __shared__ uint8_t s_signs[128];
            for (int i = threadIdx.x; i < n_embd; i += blockDim.x) xs[i] = __ldg(x + i);
            for (int i = threadIdx.x; i < 512; i += blockDim.x) {
                const uint64_t g = k_iq2xs_grid[i];
                s_grid[i] = make_uint2(uint32_t(g), uint32_t(g >> 32));
            }
            if (threadIdx.x < 128) s_signs[threadIdx.x] = k_ksigns_iq2xs[threadIdx.x];
            __syncthreads();
            const int ib32 = lane >> 2, l = lane & 3;
            for (int r = blockIdx.x * (blockDim.x / WARP) + threadIdx.x / WARP; r < n_ff; r += stride) {
                const uint8_t * r0 = m + size_t(r) * rb, * r1 = m + size_t(r + n_ff) * rb;
                float ag = 0.f, au = 0.f;
                for (int sb = 0; sb < n_embd / 256; ++sb) {
                    const float * xp = xs + sb * 256 + lane * 8;
                    const float4 x0 = *reinterpret_cast<const float4 *>(xp);
                    const float4 x1 = *reinterpret_cast<const float4 *>(xp + 4);
                    const uint8_t * b0 = r0 + size_t(sb) * 74, * b1 = r1 + size_t(sb) * 74;
                    const uint8_t sc0 = __ldg(b0 + 66 + ib32), sc1 = __ldg(b1 + 66 + ib32);
                    const uint16_t q0 = __ldg(reinterpret_cast<const uint16_t *>(b0 + 2 + 2 * lane));
                    const uint16_t q1 = __ldg(reinterpret_cast<const uint16_t *>(b1 + 2 + 2 * lane));
                    const uint2 g0 = s_grid[q0 & 511u], g1 = s_grid[q1 & 511u];
                    ag += iq8_dot_u32(g0.x, g0.y, s_signs[q0 >> 9],
                                      h2f(b0) * (0.5f + float(l < 2 ? (sc0 & 0xF) : (sc0 >> 4))) * 0.25f, x0, x1);
                    au += iq8_dot_u32(g1.x, g1.y, s_signs[q1 >> 9],
                                      h2f(b1) * (0.5f + float(l < 2 ? (sc1 & 0xF) : (sc1 >> 4))) * 0.25f, x0, x1);
                }
                const float g = warp_sum(ag), u = warp_sum(au);
                if (lane == 0) h[size_t(k) * n_ff + r] = situ(g * s, u * s, beta, lb);
            }
            return;
        }
    } else if constexpr (TYPE == T_IQ2_XXS) {
        if (n_embd <= 2304) {
            __shared__ __align__(16) float xs[2304];
            __shared__ uint2 s_grid[256];
            __shared__ uint8_t s_signs[128];
            for (int i = threadIdx.x; i < n_embd; i += blockDim.x) xs[i] = __ldg(x + i);
            {
                const uint64_t g = k_iq2xxs_grid[threadIdx.x];
                s_grid[threadIdx.x] = make_uint2(uint32_t(g), uint32_t(g >> 32));
            }
            if (threadIdx.x < 128) s_signs[threadIdx.x] = k_ksigns_iq2xs[threadIdx.x];
            __syncthreads();
            const int ib32 = lane >> 2, l = lane & 3;
            for (int r = blockIdx.x * (blockDim.x / WARP) + threadIdx.x / WARP; r < n_ff; r += stride) {
                const uint8_t * r0 = m + size_t(r) * rb, * r1 = m + size_t(r + n_ff) * rb;
                float ag = 0.f, au = 0.f;
                for (int sb = 0; sb < n_embd / 256; ++sb) {
                    const float * xp = xs + sb * 256 + lane * 8;
                    const float4 x0 = *reinterpret_cast<const float4 *>(xp);
                    const float4 x1 = *reinterpret_cast<const float4 *>(xp + 4);
                    const uint8_t * b0 = r0 + size_t(sb) * 66, * b1 = r1 + size_t(sb) * 66;
                    const uint8_t * g0p = b0 + 2 + 8 * ib32, * g1p = b1 + 2 + 8 * ib32;
                    const uint32_t aux0 = ldg_u32_2align(g0p + 4), aux1 = ldg_u32_2align(g1p + 4);
                    const uint2 g0 = s_grid[__ldg(g0p + l)], g1 = s_grid[__ldg(g1p + l)];
                    ag += iq8_dot_u32(g0.x, g0.y, s_signs[(aux0 >> (7 * l)) & 127u],
                                      h2f(b0) * (0.5f + float(aux0 >> 28)) * 0.25f, x0, x1);
                    au += iq8_dot_u32(g1.x, g1.y, s_signs[(aux1 >> (7 * l)) & 127u],
                                      h2f(b1) * (0.5f + float(aux1 >> 28)) * 0.25f, x0, x1);
                }
                const float g = warp_sum(ag), u = warp_sum(au);
                if (lane == 0) h[size_t(k) * n_ff + r] = situ(g * s, u * s, beta, lb);
            }
            return;
        }
    }
    for (int r = blockIdx.x * (blockDim.x / WARP) + threadIdx.x / WARP; r < n_ff; r += stride) {
        const float g = warp_sum(dot_row<TYPE>(m + size_t(r) * rb, x, n_embd, lane));
        const float u = warp_sum(dot_row<TYPE>(m + size_t(r + n_ff) * rb, x, n_embd, lane));
        if (lane == 0) h[size_t(k) * n_ff + r] = situ(g * s, u * s, beta, lb);
    }
}

template <int TYPE>
__global__ void __launch_bounds__(256, 2) down_kernel(const uint8_t * const * __restrict__ experts, const float * __restrict__ scale,
                                                      const int32_t * __restrict__ ids, const float * __restrict__ wts, const float * __restrict__ h,
                                                      float * __restrict__ partial, int n_embd, int n_ff) {
    const int lane = threadIdx.x % WARP;
    const int k = blockIdx.y, e = ids[k];
    const uint8_t * m = experts[e];
    const size_t rb = row_bytes(TYPE, n_ff);
    const int stride = gridDim.x * (blockDim.x / WARP);
    const float c = m ? wts[k] * (scale ? scale[e] : 1.f) : 0.f;
    if constexpr (TYPE == T_NVFP4) {
        if (m && n_ff <= 1152) {
            __shared__ __align__(16) float hs[1152];
            __shared__ float s_fp4[16];
            const float * hk = h + size_t(k) * n_ff;
            for (int i = threadIdx.x; i < n_ff; i += blockDim.x) hs[i] = __ldg(hk + i);
            if (threadIdx.x < 16) s_fp4[threadIdx.x] = float(k_fp4x2[threadIdx.x]);
            __syncthreads();
            const int s = lane >> 3, j = lane & 7;
            for (int r = blockIdx.x * (blockDim.x / WARP) + threadIdx.x / WARP; r < n_embd; r += stride) {
                const uint8_t * row = m + size_t(r) * rb;
                float acc = 0.f;
                for (int sb = 0; sb < n_ff / 64; ++sb) {
                    const uint8_t * b = row + size_t(sb) * 36;
                    const float d = ue4m3_half(__ldg(b + s));
                    const uint8_t q = __ldg(b + 4 + s * 8 + j);
                    const float * xb = hs + sb * 64 + s * 16;
                    acc += d * (s_fp4[q & 0xF] * xb[j] + s_fp4[q >> 4] * xb[8 + j]);
                }
                const float v = warp_sum(acc);
                if (lane == 0) partial[size_t(k) * n_embd + r] = c * v;
            }
            return;
        }
    }
    for (int r = blockIdx.x * (blockDim.x / WARP) + threadIdx.x / WARP; r < n_embd; r += stride) {
        const float v = m ? warp_sum(dot_row<TYPE>(m + size_t(r) * rb, h + size_t(k) * n_ff, n_ff, lane)) : 0.f;
        if (lane == 0) partial[size_t(k) * n_embd + r] = c * v;
    }
}

__global__ void sum_kernel(const float * __restrict__ partial, float * __restrict__ y, int n_embd, int n_sel) {
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= n_embd) return;
    float acc = 0.f;
    for (int k = 0; k < n_sel; ++k) acc += partial[size_t(k) * n_embd + r];
    y[r] = acc;
}

// ---------------------------------------------------------------- prefill: T tokens per launch

constexpr int PTG = 8;  // tokens per pass (accumulators in registers next to the 32 decoded weights)

// Coalesced multi-token dot of one GGUF row against `cnt` (<= NT) input pointers `xs[0..cnt-1]` (reads weights ONCE).
template <int TYPE, int NT>
__device__ __forceinline__ void dot_row_multi(const uint8_t * row, const float * const * xs, int cnt, int n, int lane, float acc[NT]) {
#pragma unroll
    for (int j = 0; j < NT; ++j) acc[j] = 0.f;
    if constexpr (TYPE == T_Q6_K) {
        const int is = lane >> 4;
        for (int sb = 0; sb < n / 256; ++sb) {
            const uint8_t * b = row + size_t(sb) * 210;
            const float d = h2f(b + 208);
            float s_tok[NT];
#pragma unroll
            for (int j = 0; j < NT; ++j) s_tok[j] = 0.f;
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const uint8_t ql0 = __ldg(b + 64 * h + lane);
                const uint8_t ql1 = __ldg(b + 64 * h + 32 + lane);
                const uint8_t qhv = __ldg(b + 128 + 32 * h + lane);
                const int8_t * sc = reinterpret_cast<const int8_t *>(b + 192) + 8 * h;
                const float w0 = float(__ldg(sc + is + 0)) * float(int((ql0 & 0xF) | ((qhv & 3) << 4)) - 32);
                const float w1 = float(__ldg(sc + is + 2)) * float(int((ql1 & 0xF) | (((qhv >> 2) & 3) << 4)) - 32);
                const float w2 = float(__ldg(sc + is + 4)) * float(int((ql0 >> 4)  | (((qhv >> 4) & 3) << 4)) - 32);
                const float w3 = float(__ldg(sc + is + 6)) * float(int((ql1 >> 4)  | (((qhv >> 6) & 3) << 4)) - 32);
                const int off = sb * 256 + 128 * h + lane;
#pragma unroll
                for (int j = 0; j < NT; ++j) {
                    if (j < cnt) {
                        const float * xh = xs[j] + off;
                        s_tok[j] += w0 * __ldg(xh) + w1 * __ldg(xh + 32) + w2 * __ldg(xh + 64) + w3 * __ldg(xh + 96);
                    }
                }
            }
#pragma unroll
            for (int j = 0; j < NT; ++j) if (j < cnt) acc[j] += d * s_tok[j];
        }
    } else if constexpr (TYPE == T_Q4_K) {
        for (int sb = 0; sb < n / 256; ++sb) {
            const uint8_t * b = row + size_t(sb) * 144;
            const float d = h2f(b), dmin = h2f(b + 2);
#pragma unroll
            for (int p = 0; p < 4; ++p) {
                int sc0, m0, sc1, m1;
                k_scale_min(2 * p, b + 4, sc0, m0);
                k_scale_min(2 * p + 1, b + 4, sc1, m1);
                const uint8_t q = __ldg(b + 16 + 32 * p + lane);
                const float w0 = d * float(sc0) * float(q & 0xF) - dmin * float(m0);
                const float w1 = d * float(sc1) * float(q >> 4)  - dmin * float(m1);
                const int off = sb * 256 + 64 * p + lane;
#pragma unroll
                for (int j = 0; j < NT; ++j)
                    if (j < cnt) acc[j] += w0 * __ldg(xs[j] + off) + w1 * __ldg(xs[j] + off + 32);
            }
        }
    } else if constexpr (TYPE == T_Q5_K) {
        for (int sb = 0; sb < n / 256; ++sb) {
            const uint8_t * b = row + size_t(sb) * 176;
            const float d = h2f(b), dmin = h2f(b + 2);
            const uint8_t qh = __ldg(b + 16 + lane);
#pragma unroll
            for (int p = 0; p < 4; ++p) {
                int sc0, m0, sc1, m1;
                k_scale_min(2 * p, b + 4, sc0, m0);
                k_scale_min(2 * p + 1, b + 4, sc1, m1);
                const uint8_t ql = __ldg(b + 48 + 32 * p + lane);
                const int v0 = (ql & 0xF) | (((qh >> (2 * p)) & 1) << 4);
                const int v1 = (ql >> 4)  | (((qh >> (2 * p + 1)) & 1) << 4);
                const float w0 = d * float(sc0) * float(v0) - dmin * float(m0);
                const float w1 = d * float(sc1) * float(v1) - dmin * float(m1);
                const int off = sb * 256 + 64 * p + lane;
#pragma unroll
                for (int j = 0; j < NT; ++j)
                    if (j < cnt) acc[j] += w0 * __ldg(xs[j] + off) + w1 * __ldg(xs[j] + off + 32);
            }
        }
    } else if constexpr (TYPE == T_IQ4_XS) {
        const int half = lane >> 4, j0 = lane & 15;
        for (int sb = 0; sb < n / 256; ++sb) {
            const uint8_t * b = row + size_t(sb) * 136;
            const float d = h2f(b);
            const uint16_t sh = uint16_t(__ldg(b + 2)) | (uint16_t(__ldg(b + 3)) << 8);
#pragma unroll
            for (int p = 0; p < 4; ++p) {
                const int ib = 2 * p + half;
                const int ls = ((__ldg(b + 4 + ib / 2) >> (4 * (ib & 1))) & 0xF) | (((sh >> (2 * ib)) & 3) << 4);
                const float dl = d * float(ls - 32);
                const uint8_t q = __ldg(b + 8 + ib * 16 + j0);
                const float w0 = dl * float(k_iq4nl[q & 0xF]), w1 = dl * float(k_iq4nl[q >> 4]);
                const int off = sb * 256 + ib * 32 + j0;
#pragma unroll
                for (int j = 0; j < NT; ++j)
                    if (j < cnt) acc[j] += w0 * __ldg(xs[j] + off) + w1 * __ldg(xs[j] + off + 16);
            }
        }
    } else if constexpr (TYPE == T_Q5_0) {
        const int half = lane >> 4, j0 = lane & 15;
        for (int sb = 0; sb < n / 32; sb += 2) {
            const uint8_t * b = row + size_t(sb + half) * 22;
            const float d = h2f(b);
            const uint32_t qh = uint32_t(__ldg(b + 2)) | (uint32_t(__ldg(b + 3)) << 8)
                              | (uint32_t(__ldg(b + 4)) << 16) | (uint32_t(__ldg(b + 5)) << 24);
            const uint8_t q = __ldg(b + 6 + j0);
            const float w0 = d * float(int((q & 0x0F) | (((qh >> j0) << 4) & 0x10)) - 16);
            const float w1 = d * float(int((q >> 4)   | ((qh >> (j0 + 12)) & 0x10)) - 16);
            const int off = (sb + half) * 32 + j0;
#pragma unroll
            for (int j = 0; j < NT; ++j)
                if (j < cnt) acc[j] += w0 * __ldg(xs[j] + off) + w1 * __ldg(xs[j] + off + 16);
        }
    } else if constexpr (TYPE == T_IQ4_NL) {
        const int half = lane >> 4, j0 = lane & 15;
        for (int sb = 0; sb < n / 32; sb += 2) {
            const uint8_t * b = row + size_t(sb + half) * 18;
            const float d = h2f(b);
            const uint8_t q = __ldg(b + 2 + j0);
            const float w0 = d * float(k_iq4nl[q & 0xF]), w1 = d * float(k_iq4nl[q >> 4]);
            const int off = (sb + half) * 32 + j0;
#pragma unroll
            for (int j = 0; j < NT; ++j)
                if (j < cnt) acc[j] += w0 * __ldg(xs[j] + off) + w1 * __ldg(xs[j] + off + 16);
        }
    } else if constexpr (TYPE == T_IQ2_XXS) {
        const int ib32 = lane >> 2, l = lane & 3;
        for (int sb = 0; sb < n / 256; ++sb) {
            const uint8_t * b = row + size_t(sb) * 66;
            const float d = h2f(b);
            const uint8_t * g = b + 2 + 8 * ib32;
            const uint32_t aux1 = ldg_u32_unaligned(g + 4);
            const float db = d * (0.5f + float(aux1 >> 28)) * 0.25f;
            const uint64_t grid = __ldg(&k_iq2xxs_grid[__ldg(g + l)]);
            const uint8_t signs = __ldg(&k_ksigns_iq2xs[(aux1 >> (7 * l)) & 127u]);
            iq8_dot_multi<NT>(grid, signs, db, xs, cnt, sb * 256 + lane * 8, acc);
        }
    } else if constexpr (TYPE == T_IQ2_XS) {
        const int ib32 = lane >> 2, l = lane & 3;
        for (int sb = 0; sb < n / 256; ++sb) {
            const uint8_t * b = row + size_t(sb) * 74;
            const float d = h2f(b);
            const uint8_t sc_b = __ldg(b + 66 + ib32);
            const float db = d * (0.5f + float(l < 2 ? (sc_b & 0xF) : (sc_b >> 4))) * 0.25f;
            const uint16_t q16 = uint16_t(__ldg(b + 2 + 2 * lane)) | (uint16_t(__ldg(b + 3 + 2 * lane)) << 8);
            const uint64_t grid = __ldg(&k_iq2xs_grid[q16 & 511u]);
            const uint8_t signs = __ldg(&k_ksigns_iq2xs[q16 >> 9]);
            iq8_dot_multi<NT>(grid, signs, db, xs, cnt, sb * 256 + lane * 8, acc);
        }
    } else if constexpr (TYPE == T_IQ2_S) {
        const int ib32 = lane >> 2, l = lane & 3;
        for (int sb = 0; sb < n / 256; ++sb) {
            const uint8_t * b = row + size_t(sb) * 82;
            const float d = h2f(b);
            const uint8_t sc_b = __ldg(b + 74 + ib32);
            const float db = d * (0.5f + float(l < 2 ? (sc_b & 0xF) : (sc_b >> 4))) * 0.25f;
            const uint32_t qs_b = __ldg(b + 2 + lane);
            const uint8_t signs = __ldg(b + 34 + lane);
            const uint32_t qh_b = __ldg(b + 66 + ib32);
            const uint32_t idx = qs_b | (((qh_b >> (2 * l)) & 3u) << 8);
            const uint64_t grid = __ldg(&k_iq2s_grid[idx]);
            iq8_dot_multi<NT>(grid, signs, db, xs, cnt, sb * 256 + lane * 8, acc);
        }
    } else if constexpr (TYPE == T_IQ3_XXS) {
        const int ib32 = lane >> 2, l = lane & 3;
        for (int sb = 0; sb < n / 256; ++sb) {
            const uint8_t * b = row + size_t(sb) * 98;
            const float d = h2f(b);
            const uint32_t aux32 = ldg_u32_unaligned(b + 66 + 4 * ib32);
            const float db = d * (0.5f + float(aux32 >> 28)) * 0.5f;
            const uint8_t signs = __ldg(&k_ksigns_iq2xs[(aux32 >> (7 * l)) & 127u]);
            const uint64_t grid = uint64_t(__ldg(&k_iq3xxs_grid[__ldg(b + 2 + 2 * lane)]))
                                | (uint64_t(__ldg(&k_iq3xxs_grid[__ldg(b + 3 + 2 * lane)])) << 32);
            iq8_dot_multi<NT>(grid, signs, db, xs, cnt, sb * 256 + lane * 8, acc);
        }
    } else if constexpr (TYPE == T_NVFP4) {
        const int s = lane >> 3, j0 = lane & 7;
        for (int sb = 0; sb < n / 64; ++sb) {
            const uint8_t * b = row + size_t(sb) * 36;
            const float d = ue4m3_half(__ldg(b + s));
            const uint8_t q = __ldg(b + 4 + s * 8 + j0);
            const float w0 = d * float(__ldg(&k_fp4x2[q & 0xF])), w1 = d * float(__ldg(&k_fp4x2[q >> 4]));
            const int off = sb * 64 + s * 16 + j0;
#pragma unroll
            for (int j = 0; j < NT; ++j)
                if (j < cnt) acc[j] += w0 * __ldg(xs[j] + off) + w1 * __ldg(xs[j] + off + 8);
        }
    } else if constexpr (TYPE == T_Q8_0) {
        for (int sb = 0; sb < n / 32; ++sb) {
            const uint8_t * b = row + size_t(sb) * 34;
            const float w = h2f(b) * float(int8_t(__ldg(b + 2 + lane)));
            const int off = sb * 32 + lane;
#pragma unroll
            for (int j = 0; j < NT; ++j)
                if (j < cnt) acc[j] += w * __ldg(xs[j] + off);
        }
    } else {
        float y[32];
        for (int sb = lane; sb < n / 32; sb += WARP) {
            dequant32<TYPE>(row, sb, y);
#pragma unroll
            for (int j = 0; j < NT; ++j) {
                if (j >= cnt) break;
                const float4 * xv = reinterpret_cast<const float4 *>(xs[j] + sb * 32);
                float a = 0.f;
#pragma unroll
                for (int q = 0; q < 8; ++q) { const float4 v = __ldg(xv + q); a += y[4 * q] * v.x + y[4 * q + 1] * v.y + y[4 * q + 2] * v.z + y[4 * q + 3] * v.w; }
                acc[j] += a;
            }
        }
    }
}

// Y[t][r] = W[r] . X[t]: coalesced warp-wide multi-token kernel (no local-memory spills on K-quants / IQ4 / Q5_0)
template <int TYPE, int NT = PTG>
__global__ void __launch_bounds__(256) matvec_t_kernel(const uint8_t * __restrict__ w, const float * __restrict__ X,
                                                       float * __restrict__ Y, int rows, int n, int T) {
    const int lane = threadIdx.x % WARP;
    const size_t rb = row_bytes(TYPE, n);
    const int stride = gridDim.x * (blockDim.x / WARP);
    for (int r = blockIdx.x * (blockDim.x / WARP) + threadIdx.x / WARP; r < rows; r += stride) {
        const uint8_t * row = w + size_t(r) * rb;
        for (int t0 = 0; t0 < T; t0 += NT) {
            const int cnt = min(NT, T - t0);
            const float * xs[NT];
#pragma unroll
            for (int j = 0; j < NT; ++j) xs[j] = X + size_t(t0 + (j < cnt ? j : 0)) * n;
            float acc[NT];
            dot_row_multi<TYPE, NT>(row, xs, cnt, n, lane, acc);
#pragma unroll
            for (int j = 0; j < NT; ++j) {
                if (j < cnt) {
                    const float v = warp_sum(acc[j]);
                    if (lane == 0) Y[size_t(t0 + j) * rows + r] = v;
                }
            }
        }
    }
}

// grouped experts (entries of expert e: offsets[e] .. offsets[e+1] in list, entry = t * k + slot)
template <int TYPE>
__global__ void __launch_bounds__(256) gate_up_grouped_kernel(const uint8_t * const * __restrict__ experts, const float * __restrict__ scale,
                                                              const int * __restrict__ offsets, const int * __restrict__ list, int k,
                                                              const float * __restrict__ X, float * __restrict__ H, int n_embd, int n_ff,
                                                              float beta, float lb) {
    const int e = blockIdx.y;
    const int b0 = offsets[e], cnt = offsets[e + 1] - b0;
    const uint8_t * m = experts[e];
    if (!m || cnt == 0) return;
    const int lane = threadIdx.x % WARP;
    const size_t rb = row_bytes(TYPE, n_embd);
    const float s = scale ? scale[e] : 1.f;
    const int stride = gridDim.x * (blockDim.x / WARP);
    if (cnt == 1) {
        const int en0 = list[b0];
        const float * x0 = X + size_t(en0 / k) * n_embd;
        for (int r = blockIdx.x * (blockDim.x / WARP) + threadIdx.x / WARP; r < n_ff; r += stride) {
            const float g = warp_sum(dot_row<TYPE>(m + size_t(r) * rb, x0, n_embd, lane));
            const float u = warp_sum(dot_row<TYPE>(m + size_t(r + n_ff) * rb, x0, n_embd, lane));
            if (lane == 0) H[size_t(en0) * n_ff + r] = situ(g * s, u * s, beta, lb);
        }
        return;
    }
    if (cnt <= 4) {
        const float * xs[4];
        int en[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            en[j] = j < cnt ? list[b0 + j] : 0;
            xs[j] = X + size_t(en[j] / k) * n_embd;
        }
        for (int r = blockIdx.x * (blockDim.x / WARP) + threadIdx.x / WARP; r < n_ff; r += stride) {
            float ag[4], au[4];
            dot_row_multi<TYPE, 4>(m + size_t(r) * rb, xs, cnt, n_embd, lane, ag);
            dot_row_multi<TYPE, 4>(m + size_t(r + n_ff) * rb, xs, cnt, n_embd, lane, au);
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                if (j < cnt) {
                    const float g = warp_sum(ag[j]), u = warp_sum(au[j]);
                    if (lane == 0) H[size_t(en[j]) * n_ff + r] = situ(g * s, u * s, beta, lb);
                }
            }
        }
        return;
    }
    for (int r = blockIdx.x * (blockDim.x / WARP) + threadIdx.x / WARP; r < n_ff; r += stride) {
        for (int j0 = 0; j0 < cnt; j0 += PTG) {
            const int ncur = min(PTG, cnt - j0);
            const float * xs[PTG];
            int en[PTG];
#pragma unroll
            for (int j = 0; j < PTG; ++j) {
                en[j] = j < ncur ? list[b0 + j0 + j] : 0;
                xs[j] = X + size_t(en[j] / k) * n_embd;
            }
            float ag[PTG], au[PTG];
            dot_row_multi<TYPE, PTG>(m + size_t(r) * rb, xs, ncur, n_embd, lane, ag);
            dot_row_multi<TYPE, PTG>(m + size_t(r + n_ff) * rb, xs, ncur, n_embd, lane, au);
#pragma unroll
            for (int j = 0; j < PTG; ++j) {
                if (j < ncur) {
                    const float g = warp_sum(ag[j]), u = warp_sum(au[j]);
                    if (lane == 0) H[size_t(en[j]) * n_ff + r] = situ(g * s, u * s, beta, lb);
                }
            }
        }
    }
}

template <int TYPE>
__global__ void __launch_bounds__(256) down_grouped_kernel(const uint8_t * const * __restrict__ experts, const float * __restrict__ scale,
                                                           const int * __restrict__ offsets, const int * __restrict__ list,
                                                           const float * __restrict__ wts, const float * __restrict__ H,
                                                           float * __restrict__ partial, int n_embd, int n_ff) {
    const int e = blockIdx.y;
    const int b0 = offsets[e], cnt = offsets[e + 1] - b0;
    const uint8_t * m = experts[e];
    if (!m || cnt == 0) return;
    const int lane = threadIdx.x % WARP;
    const size_t rb = row_bytes(TYPE, n_ff);
    const float s = scale ? scale[e] : 1.f;
    const int stride = gridDim.x * (blockDim.x / WARP);
    if (cnt == 1) {
        const int en0 = list[b0];
        const float c = wts[en0] * s;
        const float * h0 = H + size_t(en0) * n_ff;
        for (int r = blockIdx.x * (blockDim.x / WARP) + threadIdx.x / WARP; r < n_embd; r += stride) {
            const float v = warp_sum(dot_row<TYPE>(m + size_t(r) * rb, h0, n_ff, lane));
            if (lane == 0) partial[size_t(en0) * n_embd + r] = c * v;
        }
        return;
    }
    if (cnt <= 4) {
        const float * hs[4];
        int en[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            en[j] = j < cnt ? list[b0 + j] : 0;
            hs[j] = H + size_t(en[j]) * n_ff;
        }
        for (int r = blockIdx.x * (blockDim.x / WARP) + threadIdx.x / WARP; r < n_embd; r += stride) {
            float acc[4];
            dot_row_multi<TYPE, 4>(m + size_t(r) * rb, hs, cnt, n_ff, lane, acc);
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                if (j < cnt) {
                    const float v = warp_sum(acc[j]);
                    if (lane == 0) partial[size_t(en[j]) * n_embd + r] = wts[en[j]] * s * v;
                }
            }
        }
        return;
    }
    for (int r = blockIdx.x * (blockDim.x / WARP) + threadIdx.x / WARP; r < n_embd; r += stride) {
        for (int j0 = 0; j0 < cnt; j0 += PTG) {
            const int ncur = min(PTG, cnt - j0);
            const float * hs[PTG];
            int en[PTG];
#pragma unroll
            for (int j = 0; j < PTG; ++j) {
                en[j] = j < ncur ? list[b0 + j0 + j] : 0;
                hs[j] = H + size_t(en[j]) * n_ff;
            }
            float acc[PTG];
            dot_row_multi<TYPE, PTG>(m + size_t(r) * rb, hs, ncur, n_ff, lane, acc);
#pragma unroll
            for (int j = 0; j < PTG; ++j) {
                if (j < ncur) {
                    const float v = warp_sum(acc[j]);
                    if (lane == 0) partial[size_t(en[j]) * n_embd + r] = wts[en[j]] * s * v;
                }
            }
        }
    }
}

constexpr int TPB = 256;

#define IK_DISPATCH(type, LAUNCH)                  \
    switch (type) {                                \
        case T_Q5_0: LAUNCH(T_Q5_0); break;        \
        case T_Q8_0: LAUNCH(T_Q8_0); break;        \
        case T_Q4_K: LAUNCH(T_Q4_K); break;        \
        case T_Q5_K: LAUNCH(T_Q5_K); break;        \
        case T_Q6_K: LAUNCH(T_Q6_K); break;        \
        case T_IQ2_XXS: LAUNCH(T_IQ2_XXS); break;  \
        case T_IQ2_XS: LAUNCH(T_IQ2_XS); break;    \
        case T_IQ3_XXS: LAUNCH(T_IQ3_XXS); break;  \
        case T_IQ4_NL: LAUNCH(T_IQ4_NL); break;    \
        case T_IQ2_S: LAUNCH(T_IQ2_S); break;      \
        case T_IQ4_XS: LAUNCH(T_IQ4_XS); break;    \
        case T_NVFP4: LAUNCH(T_NVFP4); break;      \
        default: return (int) cudaErrorInvalidValue; \
    }

}  // namespace

extern "C" {

// 1 if the GGUF-layout kernels read ggml type `type`
int ik_gguf_type_supported(int type) {
    int b, v;
    block_shape(type, b, v);
    return b > 0;
}

size_t ik_gguf_row_bytes(int type, int n) { return row_bytes(type, n); }

// y = W x, W in GGUF blocks [rows][n] (n a multiple of 256 for K-quants / IQ4_XS, of 32 or 64 otherwise)
int ik_matvec_gguf(int type, const void * w, const float * x, float * y, int rows, int n, cudaStream_t st) {
    const int blocks = (rows + TPB / WARP - 1) / (TPB / WARP);
    const int grid = blocks < 1024 ? blocks : 1024;
    if (n == 2304) {
        if (type == T_Q6_K) { matvec_k_reg_kernel<T_Q6_K, 9><<<grid, TPB, 0, st>>>((const uint8_t *) w, x, y, rows); return (int) cudaGetLastError(); }
        if (type == T_Q5_K) { matvec_k_reg_kernel<T_Q5_K, 9><<<grid, TPB, 0, st>>>((const uint8_t *) w, x, y, rows); return (int) cudaGetLastError(); }
        if (type == T_Q4_K) { matvec_k_reg_kernel<T_Q4_K, 9><<<grid, TPB, 0, st>>>((const uint8_t *) w, x, y, rows); return (int) cudaGetLastError(); }
    } else if (n == 3072) {
        if (type == T_Q6_K) { matvec_k_reg_kernel<T_Q6_K, 12><<<grid, TPB, 0, st>>>((const uint8_t *) w, x, y, rows); return (int) cudaGetLastError(); }
        if (type == T_Q5_K) { matvec_k_reg_kernel<T_Q5_K, 12><<<grid, TPB, 0, st>>>((const uint8_t *) w, x, y, rows); return (int) cudaGetLastError(); }
        if (type == T_Q4_K) { matvec_k_reg_kernel<T_Q4_K, 12><<<grid, TPB, 0, st>>>((const uint8_t *) w, x, y, rows); return (int) cudaGetLastError(); }
    }
#define L(T) matvec_kernel<T><<<grid, TPB, 0, st>>>((const uint8_t *) w, x, y, rows, n)
    IK_DISPATCH(type, L)
#undef L
    return (int) cudaGetLastError();
}

int ik_experts_gate_up_gguf(int type, const void * const * experts, const float * scale, const int32_t * ids, int n_sel,
                            const float * x, float * h, int n_embd, int n_ff, float beta, float lb, cudaStream_t st) {
    const dim3 grid(36, n_sel);
#define L(T) gate_up_kernel<T><<<grid, TPB, 0, st>>>((const uint8_t * const *) experts, scale, ids, x, h, n_embd, n_ff, beta, lb)
    IK_DISPATCH(type, L)
#undef L
    return (int) cudaGetLastError();
}

int ik_experts_down_gguf(int type, const void * const * experts, const float * scale, const int32_t * ids, const float * wts,
                         int n_sel, const float * h, float * partial, float * y, int n_embd, int n_ff, cudaStream_t st) {
    const dim3 grid(36, n_sel);
#define L(T) down_kernel<T><<<grid, TPB, 0, st>>>((const uint8_t * const *) experts, scale, ids, wts, h, partial, n_embd, n_ff)
    IK_DISPATCH(type, L)
#undef L
    sum_kernel<<<(n_embd + TPB - 1) / TPB, TPB, 0, st>>>(partial, y, n_embd, n_sel);
    return (int) cudaGetLastError();
}

// prefill: Y[t][r] = W[r] . X[t] for T tokens, W in GGUF blocks
int ik_pf_matvec_gguf(int type, const void * w, const float * X, float * Y, int rows, int n, int T, cudaStream_t st) {
    if (T == 1) return ik_matvec_gguf(type, w, X, Y, rows, n, st);
    const int blocks = (rows + TPB / WARP - 1) / (TPB / WARP);
    const int grid = blocks < 2048 ? blocks : 2048;
    if (T <= 4) {
#define L4(TY) matvec_t_kernel<TY, 4><<<grid, TPB, 0, st>>>((const uint8_t *) w, X, Y, rows, n, T)
        IK_DISPATCH(type, L4)
#undef L4
        return (int) cudaGetLastError();
    }
#define L(TY) matvec_t_kernel<TY, PTG><<<grid, TPB, 0, st>>>((const uint8_t *) w, X, Y, rows, n, T)
    IK_DISPATCH(type, L)
#undef L
    return (int) cudaGetLastError();
}

// prefill: grouped experts in GGUF blocks (tables: null = not here); writes H and the partial rows of their entries
int ik_pf_experts_gguf(int gu_type, int dn_type, const void * const * gu, const void * const * dn, const float * gu_scale,
                       const float * dn_scale, const int * offsets, const int * list, int k, const float * wts, const float * X,
                       float * H, float * partial, int n_embd, int n_ff, int n_expert, float beta, float lb, cudaStream_t st) {
    const dim3 grid(36, n_expert);
    const auto * g = (const uint8_t * const *) gu;
    const auto * d = (const uint8_t * const *) dn;
#define L(TY) gate_up_grouped_kernel<TY><<<grid, TPB, 0, st>>>(g, gu_scale, offsets, list, k, X, H, n_embd, n_ff, beta, lb)
    IK_DISPATCH(gu_type, L)
#undef L
#define L(TY) down_grouped_kernel<TY><<<grid, TPB, 0, st>>>(d, dn_scale, offsets, list, wts, H, partial, n_embd, n_ff)
    IK_DISPATCH(dn_type, L)
#undef L
    return (int) cudaGetLastError();
}

}  // extern "C"
