// CPU expert kernels: the experts the VRAM cache misses are computed next to the RAM, straight from the GGUF blocks
// (mmap'd file, so the page cache is the RAM tier and the SSD sits behind it).
//
//   gate_up: h[r] = situ(s * (W[r] . x), s * (W[r + n_ff] . x))     W: [2*n_ff][n_embd]
//   down:    y[r] += c * s * (W[r] . h)                             W: [n_embd][n_ff]
//
// GGUF blocks: NVFP4 36 bytes / 64 values (4 UE4M3 scales, then 16 values per sub-block: 8 bytes, low nibbles are
// values 0..7, high nibbles 8..15); Q8_0 34 bytes / 32 values (fp16 scale, 32 int8). fp32 activations and accumulation,
// AVX2 + FMA (the build adds -mavx2 -mfma -mf16c), with a scalar path for other CPUs. Rows are split over threads by
// the caller.

#include <cmath>
#include <cstdint>
#include <cstring>
#include "iq_tables.h"

#if (defined(__AVX2__) && defined(__FMA__)) || (defined(_MSC_VER) && defined(__AVX2__))
#include <immintrin.h>
#define IK_AVX2 1
#endif

namespace {

// 128-entry UE4M3 -> float (halved, ggml_ue4m3_to_fp32) table: removes all branches from NVFP4 inner loops
struct Ue4m3Table {
    float v[128];
    Ue4m3Table() {
        for (int x = 0; x < 128; ++x) {
            if (x == 0x7F) v[x] = 0.f;
            else if (x < 8) v[x] = float(x) * 0.0009765625f;
            else {
                uint32_t b = (uint32_t(x) << 20) + (119u << 23);
                memcpy(&v[x], &b, 4);
            }
        }
    }
};
const Ue4m3Table k_ue4m3;

inline float ue4m3(uint8_t x) { return k_ue4m3.v[x & 0x7F]; }

inline float fp16(uint16_t h) {
#if defined(IK_AVX2)
    return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(h)));
#else
    const uint32_t s = uint32_t(h & 0x8000) << 16, e = (h >> 10) & 0x1F, m = h & 0x3FF;
    uint32_t b;
    if (e == 0) {
        if (m == 0) b = s;
        else { float f = std::ldexp(float(m), -24); memcpy(&b, &f, 4); b |= s; }
    } else if (e == 31) b = s | 0x7F800000u | (m << 13);
    else b = s | ((e + 112) << 23) | (m << 13);
    float f;
    memcpy(&f, &b, 4);
    return f;
#endif
}

const float k_fp4[16] = {0.f, 1.f, 2.f, 3.f, 4.f, 6.f, 8.f, 12.f, -0.f, -1.f, -2.f, -3.f, -4.f, -6.f, -8.f, -12.f};
const int8_t k_fp4_i8[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};

#if defined(IK_AVX2)
inline float hsum(__m256 v) {
    __m128 a = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    a = _mm_add_ps(a, _mm_movehl_ps(a, a));
    a = _mm_add_ss(a, _mm_movehdup_ps(a));
    return _mm_cvtss_f32(a);
}

inline void decode_fp4_16(const uint8_t * qs, __m128i lut, __m128i m4, __m256 & lo, __m256 & hi) {
    uint64_t q8;
    memcpy(&q8, qs, 8);
    const __m128i q = _mm_cvtsi64_si128((long long) q8);
    lo = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_shuffle_epi8(lut, _mm_and_si128(q, m4))));
    hi = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_shuffle_epi8(lut, _mm_and_si128(_mm_srli_epi16(q, 4), m4))));
}
#endif

float dot_nvfp4(const uint8_t * row, const float * x, int n) {
#if defined(IK_AVX2)
    const __m256i lut256 = _mm256_broadcastsi128_si256(_mm_loadu_si128(reinterpret_cast<const __m128i *>(k_fp4_i8)));
    const __m256i m4 = _mm256_set1_epi8(0x0F);
    __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();
    for (int b = 0; b < n / 64; ++b) {
        const uint8_t * blk = row + b * 36;
        // decode all 64 E2M1 weights of the block in 2 vpshufb instructions (low nibbles = 0..7, high = 8..15 per 16-subblock)
        const __m256i q = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(blk + 4));
        const __m256i lo_all = _mm256_shuffle_epi8(lut256, _mm256_and_si256(q, m4));
        const __m256i hi_all = _mm256_shuffle_epi8(lut256, _mm256_and_si256(_mm256_srli_epi16(q, 4), m4));
        const __m128i lo01 = _mm256_castsi256_si128(lo_all), lo23 = _mm256_extracti128_si256(lo_all, 1);
        const __m128i hi01 = _mm256_castsi256_si128(hi_all), hi23 = _mm256_extracti128_si256(hi_all, 1);
        const float * xb = x + b * 64;
        // sub-block 0
        __m256 t0 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(lo01)), _mm256_loadu_ps(xb));
        t0 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(hi01)), _mm256_loadu_ps(xb + 8), t0);
        acc0 = _mm256_fmadd_ps(t0, _mm256_set1_ps(ue4m3(blk[0])), acc0);
        // sub-block 1
        __m256 t1 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(lo01, 8))), _mm256_loadu_ps(xb + 16));
        t1 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(hi01, 8))), _mm256_loadu_ps(xb + 24), t1);
        acc1 = _mm256_fmadd_ps(t1, _mm256_set1_ps(ue4m3(blk[1])), acc1);
        // sub-block 2
        __m256 t2 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(lo23)), _mm256_loadu_ps(xb + 32));
        t2 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(hi23)), _mm256_loadu_ps(xb + 40), t2);
        acc0 = _mm256_fmadd_ps(t2, _mm256_set1_ps(ue4m3(blk[2])), acc0);
        // sub-block 3
        __m256 t3 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(lo23, 8))), _mm256_loadu_ps(xb + 48));
        t3 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(hi23, 8))), _mm256_loadu_ps(xb + 56), t3);
        acc1 = _mm256_fmadd_ps(t3, _mm256_set1_ps(ue4m3(blk[3])), acc1);
    }
    return hsum(_mm256_add_ps(acc0, acc1));
#else
    float acc = 0.f;
    for (int b = 0; b < n / 64; ++b) {
        const uint8_t * blk = row + b * 36;
        for (int s = 0; s < 4; ++s) {
            const float d = ue4m3(blk[s]);
            const uint8_t * qs = blk + 4 + s * 8;
            const float * xs = x + b * 64 + s * 16;
            float a = 0.f;
            for (int j = 0; j < 8; ++j) a += k_fp4[qs[j] & 0xF] * xs[j] + k_fp4[qs[j] >> 4] * xs[8 + j];
            acc += a * d;
        }
    }
    return acc;
#endif
}

#if defined(IK_AVX2)
// one NVFP4 row against n_tok (<= 8) input vectors: decodes each weight block from RAM ONCE for all tokens
inline void dot_nvfp4_batch(const uint8_t * row, const float * const * xs, int n_tok, int n, float * out) {
    const __m128i lut = _mm_loadu_si128(reinterpret_cast<const __m128i *>(k_fp4_i8));
    const __m128i m4 = _mm_set1_epi8(0x0F);
    __m256 acc[8];
    for (int t = 0; t < n_tok; ++t) acc[t] = _mm256_setzero_ps();
    for (int b = 0; b < n / 64; ++b) {
        const uint8_t * blk = row + b * 36;
        for (int s = 0; s < 4; ++s) {
            __m256 lo, hi;
            decode_fp4_16(blk + 4 + s * 8, lut, m4, lo, hi);
            const __m256 d = _mm256_set1_ps(ue4m3(blk[s]));
            const int off = b * 64 + s * 16;
            for (int t = 0; t < n_tok; ++t) {
                const float * xp = xs[t] + off;
                __m256 v = _mm256_fmadd_ps(hi, _mm256_loadu_ps(xp + 8), _mm256_mul_ps(lo, _mm256_loadu_ps(xp)));
                acc[t] = _mm256_fmadd_ps(v, d, acc[t]);
            }
        }
    }
    for (int t = 0; t < n_tok; ++t) out[t] = hsum(acc[t]);
}

inline void dot_q8_0_batch(const uint8_t * row, const float * const * xs, int n_tok, int n, float * out) {
    __m256 acc[8];
    for (int t = 0; t < n_tok; ++t) acc[t] = _mm256_setzero_ps();
    for (int b = 0; b < n / 32; ++b) {
        const uint8_t * blk = row + b * 34;
        uint16_t dh;
        memcpy(&dh, blk, 2);
        const __m256 d = _mm256_set1_ps(fp16(dh));
        const __m256i q = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(blk + 2));
        const __m128i ql = _mm256_castsi256_si128(q), qh = _mm256_extracti128_si256(q, 1);
        const __m256 w0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(ql));
        const __m256 w1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(ql, 8)));
        const __m256 w2 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(qh));
        const __m256 w3 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(qh, 8)));
        const int off = b * 32;
        for (int t = 0; t < n_tok; ++t) {
            const float * xp = xs[t] + off;
            __m256 v = _mm256_mul_ps(w0, _mm256_loadu_ps(xp));
            v = _mm256_fmadd_ps(w1, _mm256_loadu_ps(xp + 8), v);
            v = _mm256_fmadd_ps(w2, _mm256_loadu_ps(xp + 16), v);
            v = _mm256_fmadd_ps(w3, _mm256_loadu_ps(xp + 24), v);
            acc[t] = _mm256_fmadd_ps(v, d, acc[t]);
        }
    }
    for (int t = 0; t < n_tok; ++t) out[t] = hsum(acc[t]);
}
#endif

float dot_q8_0(const uint8_t * row, const float * x, int n) {
#if defined(IK_AVX2)
    __m256 acc = _mm256_setzero_ps();
    for (int b = 0; b < n / 32; ++b) {
        const uint8_t * blk = row + b * 34;
        uint16_t dh;
        memcpy(&dh, blk, 2);
        const __m256 d = _mm256_set1_ps(fp16(dh));
        const __m256i q = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(blk + 2));
        const float * xs = x + b * 32;
        __m256 t = _mm256_setzero_ps();
        for (int k = 0; k < 4; ++k) {
            const __m128i q8 = k < 2 ? _mm256_castsi256_si128(q) : _mm256_extracti128_si256(q, 1);
            const __m128i part = (k & 1) ? _mm_srli_si128(q8, 8) : q8;
            const __m256 v = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(part));
            t = _mm256_fmadd_ps(v, _mm256_loadu_ps(xs + k * 8), t);
        }
        acc = _mm256_fmadd_ps(t, d, acc);
    }
    return hsum(acc);
#else
    float acc = 0.f;
    for (int b = 0; b < n / 32; ++b) {
        const uint8_t * blk = row + b * 34;
        uint16_t dh;
        memcpy(&dh, blk, 2);
        const float d = fp16(dh);
        float a = 0.f;
        for (int j = 0; j < 32; ++j) a += float(int8_t(blk[2 + j])) * x[b * 32 + j];
        acc += a * d;
    }
    return acc;
#endif
}

// ---- the other ggml types (scalar; they appear in a few layers only): 32-value sub-blocks as ggml's dequantize_row_*

enum : int {
    T_Q5_0 = 6, T_Q8_0 = 8, T_Q4_K = 12, T_Q5_K = 13, T_Q6_K = 14,
    T_IQ2_XXS = 16, T_IQ2_XS = 17, T_IQ3_XXS = 18, T_IQ4_NL = 20, T_IQ2_S = 22, T_IQ4_XS = 23,
    T_NVFP4 = 40
};
const int8_t k_iq4nl[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};

inline float h2f(const uint8_t * p) { uint16_t h; memcpy(&h, p, 2); return fp16(h); }
inline uint32_t u32le(const uint8_t * p) { uint32_t v; memcpy(&v, p, 4); return v; }
inline uint16_t u16le(const uint8_t * p) { uint16_t v; memcpy(&v, p, 2); return v; }

inline void k_scale_min(int j, const uint8_t * q, int & sc, int & m) {
    if (j < 4) { sc = q[j] & 63; m = q[j + 4] & 63; }
    else { sc = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4); m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4); }
}

void dequant32(int type, const uint8_t * row, int sb, float * y) {
    switch (type) {
        case T_Q5_0: {
            const uint8_t * b = row + sb * 22;
            const float d = h2f(b);
            uint32_t qh;
            memcpy(&qh, b + 2, 4);
            for (int j = 0; j < 16; ++j) {
                // (int: qh is unsigned, and the - 16 must not wrap)
                y[j] = d * (int((b[6 + j] & 0x0F) | (((qh >> j) << 4) & 0x10)) - 16);
                y[j + 16] = d * (int((b[6 + j] >> 4) | ((qh >> (j + 12)) & 0x10)) - 16);
            }
            break;
        }
        case T_IQ4_NL: {
            const uint8_t * b = row + sb * 18;
            const float d = h2f(b);
            for (int j = 0; j < 16; ++j) { y[j] = d * k_iq4nl[b[2 + j] & 0xF]; y[j + 16] = d * k_iq4nl[b[2 + j] >> 4]; }
            break;
        }
        case T_IQ4_XS: {
            const uint8_t * b = row + (sb / 8) * 136;
            const int ib = sb % 8;
            const uint16_t sh = uint16_t(b[2]) | (uint16_t(b[3]) << 8);
            const int ls = ((b[4 + ib / 2] >> (4 * (ib % 2))) & 0xF) | (((sh >> (2 * ib)) & 3) << 4);
            const float dl = h2f(b) * (ls - 32);
            const uint8_t * qs = b + 8 + ib * 16;
            for (int j = 0; j < 16; ++j) { y[j] = dl * k_iq4nl[qs[j] & 0xF]; y[j + 16] = dl * k_iq4nl[qs[j] >> 4]; }
            break;
        }
        case T_Q4_K: {
            const uint8_t * b = row + (sb / 8) * 144;
            const int is = sb % 8;
            int sc, m;
            k_scale_min(is, b + 4, sc, m);
            const float d1 = h2f(b) * sc, m1 = h2f(b + 2) * m;
            const uint8_t * q = b + 16 + 32 * (is / 2);
            for (int l = 0; l < 32; ++l) y[l] = d1 * ((is & 1) ? (q[l] >> 4) : (q[l] & 0xF)) - m1;
            break;
        }
        case T_Q5_K: {
            const uint8_t * b = row + (sb / 8) * 176;
            const int is = sb % 8;
            int sc, m;
            k_scale_min(is, b + 4, sc, m);
            const float d1 = h2f(b) * sc, m1 = h2f(b + 2) * m;
            const uint8_t * qh = b + 16;
            const uint8_t * ql = b + 48 + 32 * (is / 2);
            for (int l = 0; l < 32; ++l) {
                const int lo = (is & 1) ? (ql[l] >> 4) : (ql[l] & 0xF);
                y[l] = d1 * (lo + (((qh[l] >> is) & 1) << 4)) - m1;
            }
            break;
        }
        case T_Q6_K: {
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
            break;
        }
        case T_IQ2_XXS: {
            const uint8_t * b = row + (sb / 8) * 66;
            const int ib32 = sb % 8;
            const float d = h2f(b);
            const uint8_t * g = b + 2 + 8 * ib32;
            const uint32_t aux1 = u32le(g + 4);
            const float db = d * (0.5f + float(aux1 >> 28)) * 0.25f;
            for (int l = 0; l < 4; ++l) {
                const uint64_t grid = k_iq2xxs_grid[g[l]];
                const uint8_t signs = k_ksigns_iq2xs[(aux1 >> (7 * l)) & 127u];
                for (int j = 0; j < 8; ++j)
                    y[8 * l + j] = db * float((grid >> (8 * j)) & 0xFFu) * ((signs & (1u << j)) ? -1.f : 1.f);
            }
            break;
        }
        case T_IQ2_XS: {
            const uint8_t * b = row + (sb / 8) * 74;
            const int ib32 = sb % 8;
            const float d = h2f(b);
            const uint8_t sc_b = b[66 + ib32];
            for (int l = 0; l < 4; ++l) {
                const float db = d * (0.5f + float(l < 2 ? (sc_b & 0xF) : (sc_b >> 4))) * 0.25f;
                const uint16_t q16 = u16le(b + 2 + 2 * (4 * ib32 + l));
                const uint64_t grid = k_iq2xs_grid[q16 & 511u];
                const uint8_t signs = k_ksigns_iq2xs[q16 >> 9];
                for (int j = 0; j < 8; ++j)
                    y[8 * l + j] = db * float((grid >> (8 * j)) & 0xFFu) * ((signs & (1u << j)) ? -1.f : 1.f);
            }
            break;
        }
        case T_IQ2_S: {
            const uint8_t * b = row + (sb / 8) * 82;
            const int ib32 = sb % 8;
            const float d = h2f(b);
            const uint8_t sc_b = b[74 + ib32];
            const uint32_t qh_b = b[66 + ib32];
            for (int l = 0; l < 4; ++l) {
                const float db = d * (0.5f + float(l < 2 ? (sc_b & 0xF) : (sc_b >> 4))) * 0.25f;
                const uint32_t idx = uint32_t(b[2 + 4 * ib32 + l]) | (((qh_b >> (2 * l)) & 3u) << 8);
                const uint64_t grid = k_iq2s_grid[idx];
                const uint8_t signs = b[34 + 4 * ib32 + l];
                for (int j = 0; j < 8; ++j)
                    y[8 * l + j] = db * float((grid >> (8 * j)) & 0xFFu) * ((signs & (1u << j)) ? -1.f : 1.f);
            }
            break;
        }
        case T_IQ3_XXS: {
            const uint8_t * b = row + (sb / 8) * 98;
            const int ib32 = sb % 8;
            const float d = h2f(b);
            const uint32_t aux32 = u32le(b + 66 + 4 * ib32);
            const float db = d * (0.5f + float(aux32 >> 28)) * 0.5f;
            for (int l = 0; l < 4; ++l) {
                const uint8_t signs = k_ksigns_iq2xs[(aux32 >> (7 * l)) & 127u];
                const uint64_t grid = uint64_t(k_iq3xxs_grid[b[2 + 2 * (4 * ib32 + l)]])
                                    | (uint64_t(k_iq3xxs_grid[b[3 + 2 * (4 * ib32 + l)]]) << 32);
                for (int j = 0; j < 8; ++j)
                    y[8 * l + j] = db * float((grid >> (8 * j)) & 0xFFu) * ((signs & (1u << j)) ? -1.f : 1.f);
            }
            break;
        }
        default: memset(y, 0, 32 * sizeof(float));
    }
}

float dot_generic(int type, const uint8_t * row, const float * x, int n) {
    float y[32], acc = 0.f;
    for (int sb = 0; sb < n / 32; ++sb) {
        dequant32(type, row, sb, y);
        float a = 0.f;
        for (int j = 0; j < 32; ++j) a += y[j] * x[sb * 32 + j];
        acc += a;
    }
    return acc;
}

inline size_t row_bytes(int type, int n) {
    switch (type) {
        case T_NVFP4: return size_t(n) / 64 * 36;
        case T_Q8_0: return size_t(n) / 32 * 34;
        case T_Q5_0: return size_t(n) / 32 * 22;
        case T_IQ4_NL: return size_t(n) / 32 * 18;
        case T_IQ2_XXS: return size_t(n) / 256 * 66;
        case T_IQ2_XS: return size_t(n) / 256 * 74;
        case T_IQ3_XXS: return size_t(n) / 256 * 98;
        case T_IQ2_S: return size_t(n) / 256 * 82;
        case T_IQ4_XS: return size_t(n) / 256 * 136;
        case T_Q4_K: return size_t(n) / 256 * 144;
        case T_Q5_K: return size_t(n) / 256 * 176;
        case T_Q6_K: return size_t(n) / 256 * 210;
        default: return 0;
    }
}

#if defined(IK_AVX2)
// dot 16 signed int8s in `v` with `xs[0..15]`, accumulating into `t`
inline __m256 fmadd_i8x16(__m128i v, const float * xs, __m256 t) {
    t = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(v)), _mm256_loadu_ps(xs), t);
    return _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(v, 8))), _mm256_loadu_ps(xs + 8), t);
}

inline __m256 fmadd_u8x16(__m128i v, const float * xs, __m256 t) {
    t = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(v)), _mm256_loadu_ps(xs), t);
    return _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_srli_si128(v, 8))), _mm256_loadu_ps(xs + 8), t);
}

float dot_iq4_nl_avx2(const uint8_t * row, const float * x, int n) {
    const __m128i lut = _mm_loadu_si128(reinterpret_cast<const __m128i *>(k_iq4nl));
    const __m128i m4 = _mm_set1_epi8(0x0F);
    __m256 acc = _mm256_setzero_ps();
    for (int sb = 0; sb < n / 32; ++sb) {
        const uint8_t * b = row + sb * 18;
        const __m128i q = _mm_loadu_si128(reinterpret_cast<const __m128i *>(b + 2));
        const __m128i lo = _mm_shuffle_epi8(lut, _mm_and_si128(q, m4));
        const __m128i hi = _mm_shuffle_epi8(lut, _mm_and_si128(_mm_srli_epi16(q, 4), m4));
        __m256 t = fmadd_i8x16(lo, x + sb * 32, _mm256_setzero_ps());
        t = fmadd_i8x16(hi, x + sb * 32 + 16, t);
        acc = _mm256_fmadd_ps(t, _mm256_set1_ps(h2f(b)), acc);
    }
    return hsum(acc);
}

float dot_iq4_xs_avx2(const uint8_t * row, const float * x, int n) {
    const __m128i lut = _mm_loadu_si128(reinterpret_cast<const __m128i *>(k_iq4nl));
    const __m128i m4 = _mm_set1_epi8(0x0F);
    __m256 acc = _mm256_setzero_ps();
    for (int sb = 0; sb < n / 256; ++sb) {
        const uint8_t * b = row + sb * 136;
        const float d = h2f(b);
        const uint16_t sh = uint16_t(b[2]) | (uint16_t(b[3]) << 8);
        for (int ib = 0; ib < 8; ++ib) {
            const int ls = ((b[4 + ib / 2] >> (4 * (ib & 1))) & 0xF) | (((sh >> (2 * ib)) & 3) << 4);
            const __m128i q = _mm_loadu_si128(reinterpret_cast<const __m128i *>(b + 8 + ib * 16));
            const __m128i lo = _mm_shuffle_epi8(lut, _mm_and_si128(q, m4));
            const __m128i hi = _mm_shuffle_epi8(lut, _mm_and_si128(_mm_srli_epi16(q, 4), m4));
            const float * xs = x + sb * 256 + ib * 32;
            __m256 t = fmadd_i8x16(lo, xs, _mm256_setzero_ps());
            t = fmadd_i8x16(hi, xs + 16, t);
            acc = _mm256_fmadd_ps(t, _mm256_set1_ps(d * float(ls - 32)), acc);
        }
    }
    return hsum(acc);
}

float dot_q5_0_avx2(const uint8_t * row, const float * x, int n) {
    const __m128i m4 = _mm_set1_epi8(0x0F);
    const __m128i c16 = _mm_set1_epi8(16);
    const __m128i bit_pos = _mm_setr_epi8(1, 2, 4, 8, 16, 32, 64, (char) 128, 1, 2, 4, 8, 16, 32, 64, (char) 128);
    __m256 acc = _mm256_setzero_ps();
    for (int sb = 0; sb < n / 32; ++sb) {
        const uint8_t * b = row + sb * 22;
        uint32_t qh;
        memcpy(&qh, b + 2, 4);
        const __m128i q = _mm_loadu_si128(reinterpret_cast<const __m128i *>(b + 6));
        const __m128i h_lo_b = _mm_shuffle_epi8(_mm_cvtsi32_si128((int) qh), _mm_setr_epi8(0,0,0,0,0,0,0,0, 1,1,1,1,1,1,1,1));
        const __m128i h_hi_b = _mm_shuffle_epi8(_mm_cvtsi32_si128((int) (qh >> 16)), _mm_setr_epi8(0,0,0,0,0,0,0,0, 1,1,1,1,1,1,1,1));
        const __m128i h_lo = _mm_and_si128(_mm_cmpeq_epi8(_mm_and_si128(h_lo_b, bit_pos), bit_pos), c16);
        const __m128i h_hi = _mm_and_si128(_mm_cmpeq_epi8(_mm_and_si128(h_hi_b, bit_pos), bit_pos), c16);
        const __m128i lo = _mm_sub_epi8(_mm_or_si128(_mm_and_si128(q, m4), h_lo), c16);
        const __m128i hi = _mm_sub_epi8(_mm_or_si128(_mm_and_si128(_mm_srli_epi16(q, 4), m4), h_hi), c16);
        __m256 t = fmadd_i8x16(lo, x + sb * 32, _mm256_setzero_ps());
        t = fmadd_i8x16(hi, x + sb * 32 + 16, t);
        acc = _mm256_fmadd_ps(t, _mm256_set1_ps(h2f(b)), acc);
    }
    return hsum(acc);
}

float dot_q4_k_avx2(const uint8_t * row, const float * x, int n) {
    const __m128i m4 = _mm_set1_epi8(0x0F);
    __m256 acc = _mm256_setzero_ps();
    for (int sb = 0; sb < n / 256; ++sb) {
        const uint8_t * b = row + sb * 144;
        const float d = h2f(b), dmin = h2f(b + 2);
        for (int p = 0; p < 4; ++p) {
            int sc0, m0, sc1, m1;
            k_scale_min(2 * p, b + 4, sc0, m0);
            k_scale_min(2 * p + 1, b + 4, sc1, m1);
            const uint8_t * q = b + 16 + 32 * p;
            const float * x0 = x + sb * 256 + 64 * p;
            const float * x1 = x0 + 32;
            const __m128i q0 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(q));
            const __m128i q1 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(q + 16));
            __m256 t0 = fmadd_u8x16(_mm_and_si128(q0, m4), x0, _mm256_setzero_ps());
            t0 = fmadd_u8x16(_mm_and_si128(q1, m4), x0 + 16, t0);
            __m256 s0 = _mm256_add_ps(_mm256_add_ps(_mm256_loadu_ps(x0), _mm256_loadu_ps(x0 + 8)),
                                      _mm256_add_ps(_mm256_loadu_ps(x0 + 16), _mm256_loadu_ps(x0 + 24)));
            acc = _mm256_fmadd_ps(t0, _mm256_set1_ps(d * float(sc0)), acc);
            acc = _mm256_fnmadd_ps(s0, _mm256_set1_ps(dmin * float(m0)), acc);

            __m256 t1 = fmadd_u8x16(_mm_and_si128(_mm_srli_epi16(q0, 4), m4), x1, _mm256_setzero_ps());
            t1 = fmadd_u8x16(_mm_and_si128(_mm_srli_epi16(q1, 4), m4), x1 + 16, t1);
            __m256 s1 = _mm256_add_ps(_mm256_add_ps(_mm256_loadu_ps(x1), _mm256_loadu_ps(x1 + 8)),
                                      _mm256_add_ps(_mm256_loadu_ps(x1 + 16), _mm256_loadu_ps(x1 + 24)));
            acc = _mm256_fmadd_ps(t1, _mm256_set1_ps(d * float(sc1)), acc);
            acc = _mm256_fnmadd_ps(s1, _mm256_set1_ps(dmin * float(m1)), acc);
        }
    }
    return hsum(acc);
}

float dot_q5_k_avx2(const uint8_t * row, const float * x, int n) {
    const __m128i m4 = _mm_set1_epi8(0x0F);
    const __m128i c16 = _mm_set1_epi8(16);
    __m256 acc = _mm256_setzero_ps();
    for (int sb = 0; sb < n / 256; ++sb) {
        const uint8_t * b = row + sb * 176;
        const float d = h2f(b), dmin = h2f(b + 2);
        const __m128i qh0 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(b + 16));
        const __m128i qh1 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(b + 32));
        for (int p = 0; p < 4; ++p) {
            int sc0, m0, sc1, m1;
            k_scale_min(2 * p, b + 4, sc0, m0);
            k_scale_min(2 * p + 1, b + 4, sc1, m1);
            const uint8_t * ql = b + 48 + 32 * p;
            const float * x0 = x + sb * 256 + 64 * p;
            const float * x1 = x0 + 32;
            const __m128i q0 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(ql));
            const __m128i q1 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(ql + 16));
            const __m128i b0 = _mm_set1_epi8(char(1 << (2 * p)));
            const __m128i b1 = _mm_set1_epi8(char(1 << (2 * p + 1)));
            const __m128i h0a = _mm_and_si128(_mm_cmpeq_epi8(_mm_and_si128(qh0, b0), b0), c16);
            const __m128i h0b = _mm_and_si128(_mm_cmpeq_epi8(_mm_and_si128(qh1, b0), b0), c16);
            const __m128i h1a = _mm_and_si128(_mm_cmpeq_epi8(_mm_and_si128(qh0, b1), b1), c16);
            const __m128i h1b = _mm_and_si128(_mm_cmpeq_epi8(_mm_and_si128(qh1, b1), b1), c16);

            __m256 t0 = fmadd_u8x16(_mm_or_si128(_mm_and_si128(q0, m4), h0a), x0, _mm256_setzero_ps());
            t0 = fmadd_u8x16(_mm_or_si128(_mm_and_si128(q1, m4), h0b), x0 + 16, t0);
            __m256 s0 = _mm256_add_ps(_mm256_add_ps(_mm256_loadu_ps(x0), _mm256_loadu_ps(x0 + 8)),
                                      _mm256_add_ps(_mm256_loadu_ps(x0 + 16), _mm256_loadu_ps(x0 + 24)));
            acc = _mm256_fmadd_ps(t0, _mm256_set1_ps(d * float(sc0)), acc);
            acc = _mm256_fnmadd_ps(s0, _mm256_set1_ps(dmin * float(m0)), acc);

            __m256 t1 = fmadd_u8x16(_mm_or_si128(_mm_and_si128(_mm_srli_epi16(q0, 4), m4), h1a), x1, _mm256_setzero_ps());
            t1 = fmadd_u8x16(_mm_or_si128(_mm_and_si128(_mm_srli_epi16(q1, 4), m4), h1b), x1 + 16, t1);
            __m256 s1 = _mm256_add_ps(_mm256_add_ps(_mm256_loadu_ps(x1), _mm256_loadu_ps(x1 + 8)),
                                      _mm256_add_ps(_mm256_loadu_ps(x1 + 16), _mm256_loadu_ps(x1 + 24)));
            acc = _mm256_fmadd_ps(t1, _mm256_set1_ps(d * float(sc1)), acc);
            acc = _mm256_fnmadd_ps(s1, _mm256_set1_ps(dmin * float(m1)), acc);
        }
    }
    return hsum(acc);
}

inline __m256 decode_iq8_avx2(uint64_t grid, uint8_t signs, __m256i smask_bits, __m256i sgn_neg) {
    const __m256 mag = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_cvtsi64_si128((long long) grid)));
    const __m256i sb = _mm256_cmpeq_epi32(_mm256_and_si256(_mm256_set1_epi32(signs), smask_bits), smask_bits);
    return _mm256_xor_ps(mag, _mm256_castsi256_ps(_mm256_and_si256(sb, sgn_neg)));
}

float dot_iq2_xxs_avx2(const uint8_t * row, const float * x, int n) {
    const __m256i smask_bits = _mm256_setr_epi32(1, 2, 4, 8, 16, 32, 64, 128);
    const __m256i sgn_neg = _mm256_set1_epi32(int(0x80000000u));
    __m256 acc = _mm256_setzero_ps();
    for (int sb = 0; sb < n / 256; ++sb) {
        const uint8_t * b = row + size_t(sb) * 66;
        const float d = h2f(b);
        for (int ib32 = 0; ib32 < 8; ++ib32) {
            const uint8_t * g = b + 2 + 8 * ib32;
            const uint32_t aux1 = u32le(g + 4);
            const float db = d * (0.5f + float(aux1 >> 28)) * 0.25f;
            const float * xs = x + sb * 256 + ib32 * 32;
            __m256 t = _mm256_setzero_ps();
            for (int l = 0; l < 4; ++l) {
                const __m256 w = decode_iq8_avx2(k_iq2xxs_grid[g[l]], k_ksigns_iq2xs[(aux1 >> (7 * l)) & 127u], smask_bits, sgn_neg);
                t = _mm256_fmadd_ps(w, _mm256_loadu_ps(xs + 8 * l), t);
            }
            acc = _mm256_fmadd_ps(t, _mm256_set1_ps(db), acc);
        }
    }
    return hsum(acc);
}

float dot_iq2_xs_avx2(const uint8_t * row, const float * x, int n) {
    const __m256i smask_bits = _mm256_setr_epi32(1, 2, 4, 8, 16, 32, 64, 128);
    const __m256i sgn_neg = _mm256_set1_epi32(int(0x80000000u));
    __m256 acc = _mm256_setzero_ps();
    for (int sb = 0; sb < n / 256; ++sb) {
        const uint8_t * b = row + size_t(sb) * 74;
        const float d = h2f(b);
        for (int ib32 = 0; ib32 < 8; ++ib32) {
            const uint8_t sc_b = b[66 + ib32];
            const float * xs = x + sb * 256 + ib32 * 32;
            const uint16_t q0 = u16le(b + 2 + 8 * ib32 + 0), q1 = u16le(b + 2 + 8 * ib32 + 2);
            const uint16_t q2 = u16le(b + 2 + 8 * ib32 + 4), q3 = u16le(b + 2 + 8 * ib32 + 6);
            __m256 t0 = _mm256_mul_ps(decode_iq8_avx2(k_iq2xs_grid[q0 & 511u], k_ksigns_iq2xs[q0 >> 9], smask_bits, sgn_neg),
                                      _mm256_loadu_ps(xs));
            t0 = _mm256_fmadd_ps(decode_iq8_avx2(k_iq2xs_grid[q1 & 511u], k_ksigns_iq2xs[q1 >> 9], smask_bits, sgn_neg),
                                 _mm256_loadu_ps(xs + 8), t0);
            acc = _mm256_fmadd_ps(t0, _mm256_set1_ps(d * (0.5f + float(sc_b & 0xF)) * 0.25f), acc);
            __m256 t1 = _mm256_mul_ps(decode_iq8_avx2(k_iq2xs_grid[q2 & 511u], k_ksigns_iq2xs[q2 >> 9], smask_bits, sgn_neg),
                                      _mm256_loadu_ps(xs + 16));
            t1 = _mm256_fmadd_ps(decode_iq8_avx2(k_iq2xs_grid[q3 & 511u], k_ksigns_iq2xs[q3 >> 9], smask_bits, sgn_neg),
                                 _mm256_loadu_ps(xs + 24), t1);
            acc = _mm256_fmadd_ps(t1, _mm256_set1_ps(d * (0.5f + float(sc_b >> 4)) * 0.25f), acc);
        }
    }
    return hsum(acc);
}

float dot_iq2_s_avx2(const uint8_t * row, const float * x, int n) {
    const __m256i smask_bits = _mm256_setr_epi32(1, 2, 4, 8, 16, 32, 64, 128);
    const __m256i sgn_neg = _mm256_set1_epi32(int(0x80000000u));
    __m256 acc = _mm256_setzero_ps();
    for (int sb = 0; sb < n / 256; ++sb) {
        const uint8_t * b = row + size_t(sb) * 82;
        const float d = h2f(b);
        for (int ib32 = 0; ib32 < 8; ++ib32) {
            const uint8_t sc_b = b[74 + ib32];
            const uint32_t qh_b = b[66 + ib32];
            const uint8_t * qs = b + 2 + 4 * ib32;
            const uint8_t * sg = b + 34 + 4 * ib32;
            const float * xs = x + sb * 256 + ib32 * 32;
            __m256 t0 = _mm256_mul_ps(decode_iq8_avx2(k_iq2s_grid[qs[0] | ((qh_b & 3u) << 8)], sg[0], smask_bits, sgn_neg),
                                      _mm256_loadu_ps(xs));
            t0 = _mm256_fmadd_ps(decode_iq8_avx2(k_iq2s_grid[qs[1] | (((qh_b >> 2) & 3u) << 8)], sg[1], smask_bits, sgn_neg),
                                 _mm256_loadu_ps(xs + 8), t0);
            acc = _mm256_fmadd_ps(t0, _mm256_set1_ps(d * (0.5f + float(sc_b & 0xF)) * 0.25f), acc);
            __m256 t1 = _mm256_mul_ps(decode_iq8_avx2(k_iq2s_grid[qs[2] | (((qh_b >> 4) & 3u) << 8)], sg[2], smask_bits, sgn_neg),
                                      _mm256_loadu_ps(xs + 16));
            t1 = _mm256_fmadd_ps(decode_iq8_avx2(k_iq2s_grid[qs[3] | (((qh_b >> 6) & 3u) << 8)], sg[3], smask_bits, sgn_neg),
                                 _mm256_loadu_ps(xs + 24), t1);
            acc = _mm256_fmadd_ps(t1, _mm256_set1_ps(d * (0.5f + float(sc_b >> 4)) * 0.25f), acc);
        }
    }
    return hsum(acc);
}

float dot_iq3_xxs_avx2(const uint8_t * row, const float * x, int n) {
    const __m256i smask_bits = _mm256_setr_epi32(1, 2, 4, 8, 16, 32, 64, 128);
    const __m256i sgn_neg = _mm256_set1_epi32(int(0x80000000u));
    __m256 acc = _mm256_setzero_ps();
    for (int sb = 0; sb < n / 256; ++sb) {
        const uint8_t * b = row + size_t(sb) * 98;
        const float d = h2f(b);
        for (int ib32 = 0; ib32 < 8; ++ib32) {
            const uint32_t aux32 = u32le(b + 66 + 4 * ib32);
            const float db = d * (0.5f + float(aux32 >> 28)) * 0.5f;
            const uint8_t * qs = b + 2 + 8 * ib32;
            const float * xs = x + sb * 256 + ib32 * 32;
            __m256 t = _mm256_setzero_ps();
            for (int l = 0; l < 4; ++l) {
                const uint64_t grid = uint64_t(k_iq3xxs_grid[qs[2 * l]]) | (uint64_t(k_iq3xxs_grid[qs[2 * l + 1]]) << 32);
                const __m256 w = decode_iq8_avx2(grid, k_ksigns_iq2xs[(aux32 >> (7 * l)) & 127u], smask_bits, sgn_neg);
                t = _mm256_fmadd_ps(w, _mm256_loadu_ps(xs + 8 * l), t);
            }
            acc = _mm256_fmadd_ps(t, _mm256_set1_ps(db), acc);
        }
    }
    return hsum(acc);
}
#endif

// type: the ggml type id (AVX2 for NVFP4, Q8_0, Q5_0, IQ2_XXS, IQ2_XS, IQ3_XXS, IQ2_S, IQ4_NL, IQ4_XS, Q4_K, Q5_K)
inline float dot(int type, const uint8_t * row, const float * x, int n) {
    if (type == T_NVFP4) return dot_nvfp4(row, x, n);
    if (type == T_Q8_0) return dot_q8_0(row, x, n);
#if defined(IK_AVX2)
    if (type == T_Q5_0) return dot_q5_0_avx2(row, x, n);
    if (type == T_IQ2_XXS) return dot_iq2_xxs_avx2(row, x, n);
    if (type == T_IQ2_XS) return dot_iq2_xs_avx2(row, x, n);
    if (type == T_IQ3_XXS) return dot_iq3_xxs_avx2(row, x, n);
    if (type == T_IQ2_S) return dot_iq2_s_avx2(row, x, n);
    if (type == T_IQ4_NL) return dot_iq4_nl_avx2(row, x, n);
    if (type == T_IQ4_XS) return dot_iq4_xs_avx2(row, x, n);
    if (type == T_Q4_K) return dot_q4_k_avx2(row, x, n);
    if (type == T_Q5_K) return dot_q5_k_avx2(row, x, n);
#endif
    return dot_generic(type, row, x, n);
}

}  // namespace

extern "C" {

#if defined(IK_AVX2)
// Exact fp32 2-row NVFP4 dot: shares the 8 x-vector loads in YMM registers across two rows (gate+up or 2 down rows).
inline void dot_nvfp4_2rows(const uint8_t * r0, const uint8_t * r1, const float * x, int n, float & o0, float & o1) {
    const __m256i lut256 = _mm256_broadcastsi128_si256(_mm_loadu_si128(reinterpret_cast<const __m128i *>(k_fp4_i8)));
    const __m256i m4 = _mm256_set1_epi8(0x0F);
    __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
    const int nb = n / 64;
    for (int b = 0; b < nb; ++b) {
        const size_t off = size_t(b) * 36;
        if (b + 2 < nb) {
            _mm_prefetch(reinterpret_cast<const char *>(r0 + off + 72), _MM_HINT_T0);
            _mm_prefetch(reinterpret_cast<const char *>(r1 + off + 72), _MM_HINT_T0);
        }
        const uint8_t * b0 = r0 + off, * b1 = r1 + off;
        const __m256i q0 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(b0 + 4));
        const __m256i q1 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(b1 + 4));
        const __m256i l0 = _mm256_shuffle_epi8(lut256, _mm256_and_si256(q0, m4));
        const __m256i h0 = _mm256_shuffle_epi8(lut256, _mm256_and_si256(_mm256_srli_epi16(q0, 4), m4));
        const __m256i l1 = _mm256_shuffle_epi8(lut256, _mm256_and_si256(q1, m4));
        const __m256i h1 = _mm256_shuffle_epi8(lut256, _mm256_and_si256(_mm256_srli_epi16(q1, 4), m4));
        const __m128i l0a = _mm256_castsi256_si128(l0), l0b = _mm256_extracti128_si256(l0, 1);
        const __m128i h0a = _mm256_castsi256_si128(h0), h0b = _mm256_extracti128_si256(h0, 1);
        const __m128i l1a = _mm256_castsi256_si128(l1), l1b = _mm256_extracti128_si256(l1, 1);
        const __m128i h1a = _mm256_castsi256_si128(h1), h1b = _mm256_extracti128_si256(h1, 1);
        const float * xb = x + b * 64;
        // sub-block 0
        __m256 xlo = _mm256_loadu_ps(xb), xhi = _mm256_loadu_ps(xb + 8);
        __m256 t0 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(h0a)), xhi,
                                    _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(l0a)), xlo));
        __m256 t1 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(h1a)), xhi,
                                    _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(l1a)), xlo));
        a0 = _mm256_fmadd_ps(t0, _mm256_set1_ps(ue4m3(b0[0])), a0);
        a1 = _mm256_fmadd_ps(t1, _mm256_set1_ps(ue4m3(b1[0])), a1);
        // sub-block 1
        xlo = _mm256_loadu_ps(xb + 16); xhi = _mm256_loadu_ps(xb + 24);
        t0 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(h0a, 8))), xhi,
                             _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(l0a, 8))), xlo));
        t1 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(h1a, 8))), xhi,
                             _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(l1a, 8))), xlo));
        a0 = _mm256_fmadd_ps(t0, _mm256_set1_ps(ue4m3(b0[1])), a0);
        a1 = _mm256_fmadd_ps(t1, _mm256_set1_ps(ue4m3(b1[1])), a1);
        // sub-block 2
        xlo = _mm256_loadu_ps(xb + 32); xhi = _mm256_loadu_ps(xb + 40);
        t0 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(h0b)), xhi,
                             _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(l0b)), xlo));
        t1 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(h1b)), xhi,
                             _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(l1b)), xlo));
        a0 = _mm256_fmadd_ps(t0, _mm256_set1_ps(ue4m3(b0[2])), a0);
        a1 = _mm256_fmadd_ps(t1, _mm256_set1_ps(ue4m3(b1[2])), a1);
        // sub-block 3
        xlo = _mm256_loadu_ps(xb + 48); xhi = _mm256_loadu_ps(xb + 56);
        t0 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(h0b, 8))), xhi,
                             _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(l0b, 8))), xlo));
        t1 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(h1b, 8))), xhi,
                             _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(l1b, 8))), xlo));
        a0 = _mm256_fmadd_ps(t0, _mm256_set1_ps(ue4m3(b0[3])), a0);
        a1 = _mm256_fmadd_ps(t1, _mm256_set1_ps(ue4m3(b1[3])), a1);
    }
    o0 = hsum(a0);
    o1 = hsum(a1);
}
#endif

// rows [r0, r1) of the gate/up product of one expert; w: GGUF blocks of the expert's [2*n_ff][n_embd] matrix;
// type: its ggml type id
void ik_cpu_gate_up(int type, const uint8_t * w, float scale, const float * x, float * h, int n_embd, int n_ff, float beta, float lb,
                    int r0, int r1) {
    const size_t rb = row_bytes(type, n_embd);
    for (int r = r0; r < r1; ++r) {
        float g, u;
#if defined(IK_AVX2)
        if (type == T_NVFP4) {
            dot_nvfp4_2rows(w + size_t(r) * rb, w + size_t(r + n_ff) * rb, x, n_embd, g, u);
            g *= scale;
            u *= scale;
        } else
#endif
        {
            g = dot(type, w + size_t(r) * rb, x, n_embd) * scale;
            u = dot(type, w + size_t(r + n_ff) * rb, x, n_embd) * scale;
        }
        const float act = beta * std::tanh(g / beta) / (1.f + std::exp(-g));
        if (lb > 0.f) u = lb * std::tanh(u / lb);
        h[r] = act * u;
    }
}

// rows [r0, r1) of the down product of one expert, added to y with coefficient c (router weight x expert scale)
void ik_cpu_down(int type, const uint8_t * w, float c, const float * h, float * y, int n_embd, int n_ff, int r0, int r1) {
    const size_t rb = row_bytes(type, n_ff);
    int r = r0;
#if defined(IK_AVX2)
    if (type == T_NVFP4) {
        for (; r + 1 < r1; r += 2) {
            float d0, d1;
            dot_nvfp4_2rows(w + size_t(r) * rb, w + size_t(r + 1) * rb, h, n_ff, d0, d1);
            y[r] += c * d0;
            y[r + 1] += c * d1;
        }
    }
#endif
    for (; r < r1; ++r) y[r] += c * dot(type, w + size_t(r) * rb, h, n_ff);
    (void) n_embd;
}

// batched multi-token CPU expert kernels (1 <= n_tok <= 8): reads each expert weight row from RAM ONCE across all tokens
void ik_cpu_gate_up_batch(int type, const uint8_t * w, float scale, const float * const * xs, float * const * hs, int n_tok,
                          int n_embd, int n_ff, float beta, float lb, int r0, int r1) {
    if (n_tok == 1) {
        ik_cpu_gate_up(type, w, scale, xs[0], hs[0], n_embd, n_ff, beta, lb, r0, r1);
        return;
    }
    const size_t rb = row_bytes(type, n_embd);
    float gv[8], uv[8];
    for (int r = r0; r < r1; ++r) {
#if defined(IK_AVX2)
        if (type == T_NVFP4) {
            dot_nvfp4_batch(w + size_t(r) * rb, xs, n_tok, n_embd, gv);
            dot_nvfp4_batch(w + size_t(r + n_ff) * rb, xs, n_tok, n_embd, uv);
        } else if (type == T_Q8_0) {
            dot_q8_0_batch(w + size_t(r) * rb, xs, n_tok, n_embd, gv);
            dot_q8_0_batch(w + size_t(r + n_ff) * rb, xs, n_tok, n_embd, uv);
        } else
#endif
        {
            for (int t = 0; t < n_tok; ++t) {
                gv[t] = dot(type, w + size_t(r) * rb, xs[t], n_embd);
                uv[t] = dot(type, w + size_t(r + n_ff) * rb, xs[t], n_embd);
            }
        }
        for (int t = 0; t < n_tok; ++t) {
            const float g = gv[t] * scale;
            float u = uv[t] * scale;
            const float act = beta * std::tanh(g / beta) / (1.f + std::exp(-g));
            if (lb > 0.f) u = lb * std::tanh(u / lb);
            hs[t][r] = act * u;
        }
    }
}

void ik_cpu_down_batch(int type, const uint8_t * w, const float * cs, const float * const * hs, float * const * ys, int n_tok,
                       int n_embd, int n_ff, int r0, int r1) {
    if (n_tok == 1) {
        ik_cpu_down(type, w, cs[0], hs[0], ys[0], n_embd, n_ff, r0, r1);
        return;
    }
    const size_t rb = row_bytes(type, n_ff);
    float out[8];
    for (int r = r0; r < r1; ++r) {
#if defined(IK_AVX2)
        if (type == T_NVFP4) {
            dot_nvfp4_batch(w + size_t(r) * rb, hs, n_tok, n_ff, out);
        } else if (type == T_Q8_0) {
            dot_q8_0_batch(w + size_t(r) * rb, hs, n_tok, n_ff, out);
        } else
#endif
        {
            for (int t = 0; t < n_tok; ++t) out[t] = dot(type, w + size_t(r) * rb, hs[t], n_ff);
        }
        for (int t = 0; t < n_tok; ++t) ys[t][r] += cs[t] * out[t];
    }
    (void) n_embd;
}

}  // extern "C"
