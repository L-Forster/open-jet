// Checks the i-quant re-layouts of qwen38_mixed.cu (iq_repack.h) against ggml's dequantization, on the tensors of a
// real GGUF: every weight is rebuilt from the repacked block twice, the way the decode GEMV (rows_iq3 / rows_iq2)
// reads it and the way the prefill GEMM (mmh) reads it, and compared with ggml's to_float of the source row.
//
//   tools/iq_repack_check model.gguf [rows per tensor, default 256]
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "ggml.h"
#define GGML_COMMON_DECL_C
#define GGML_COMMON_IMPL_C
#include "ggml-common.h" // llama.cpp ggml/src (-I in the Makefile)
#include "../iq_repack.h"

static float f16(const uint8_t * p) {
    uint16_t h;
    memcpy(&h, p, 2);
    return ggml_fp16_to_fp32(h);
}

// the 8 integer weights of an IQ2 / IQ1 group (repack_iq2's layout): grid index bits 0-7 lo8, bits 8-9 hi2, sign byte sg
static void group_values(int type, unsigned lo8, unsigned hi2, unsigned sg, int8_t * v) {
    if (type == gguf::IQ1_S || type == gguf::IQ1_M) {
        const int8_t * g = (const int8_t *) (iq1s_grid + (lo8 | ((sg & 7) << 8)));
        for (int e = 0; e < 8; e++) v[e] = (int8_t) (8 * g[e] + ((sg & 8) ? -1 : 1));
        return;
    }
    const uint64_t * grid = type == gguf::IQ2_XXS ? iq2xxs_grid : type == gguf::IQ2_XS ? iq2xs_grid : iq2s_grid;
    const uint8_t * g = (const uint8_t *) (grid + (lo8 | (hi2 << 8)));
    for (int e = 0; e < 8; e++) v[e] = (int8_t) (((sg >> e) & 1) ? -g[e] : g[e]);
}

// one 256-weight block, as rows_iq3 / rows_iq2 decode it (unit uu = 2 p + h: A = sub-block 2 p, B = 2 p + 1)
static void gemv_block(int type, const uint8_t * blk, float * y) {
    for (int uu = 0; uu < 8; uu++) {
        const int p = uu >> 1, hh = uu & 1;
        float * yA = y + 64 * p + 16 * hh, * yB = yA + 32;
        if (type == gguf::Q3_K) {
            const float d = f16(blk + 108);
            for (int part = 0; part < 2; part++) {
                const int i = 4 * p + 2 * part + hh;
                const int sc = (int) (((blk[96 + i / 2] >> (4 * (i & 1))) & 15) | (((blk[104 + i / 4] >> (2 * (i % 4))) & 3) << 4)) - 32;
                uint32_t w;
                uint16_t hm;
                memcpy(&w, blk + 8 * uu + 4 * part, 4);
                memcpy(&hm, blk + 64 + 4 * uu + 2 * part, 2);
                for (int m = 0; m < 4; m++)
                    for (int e = 0; e < 4; e++) {
                        const int q = (int) (((w >> (2 * m + 8 * e)) & 3) | (((hm >> (4 * m + e)) & 1) << 2)) - 4;
                        (part ? yB : yA)[4 * m + e] = d * (float) sc * (float) q;
                    }
            }
        } else if (type == gguf::Q2_K) {
            const float d = f16(blk + 80), dmin = f16(blk + 82);
            for (int part = 0; part < 2; part++) {
                const unsigned sc = blk[64 + 4 * p + 2 * part + hh];
                uint32_t w;
                memcpy(&w, blk + 8 * uu + 4 * part, 4);
                for (int m = 0; m < 4; m++)
                    for (int e = 0; e < 4; e++) {
                        const int q = (w >> (2 * m + 8 * e)) & 3;
                        (part ? yB : yA)[4 * m + e] = d * (float) (sc & 15) * (float) q - dmin * (float) (sc >> 4);
                    }
            }
        } else if (type == gguf::IQ3_S || type == gguf::IQ3_XXS) {
            const bool xxs = type == gguf::IQ3_XXS;
            const float d = f16(blk + 104) * (xxs ? 0.25f : 1.0f);
            const unsigned sc = blk[106 + p];
            const float fA = d * (float) (1 + 2 * (int) (sc & 15)), fB = d * (float) (1 + 2 * (int) (sc >> 4));
            const unsigned qh = xxs ? 0 : blk[96 + uu];
            uint32_t sv;
            memcpy(&sv, blk + 64 + 4 * uu, 4);
            for (int j = 0; j < 4; j++)
                for (int part = 0; part < 2; part++) {
                    const unsigned idx = blk[8 * uu + 4 * part + j] | (((qh >> (4 * part + j)) & 1) << 8);
                    const uint8_t * g = (const uint8_t *) ((xxs ? iq3xxs_grid : iq3s_grid) + idx);
                    for (int e = 0; e < 4; e++) {
                        const bool neg = (sv >> (16 * part + 4 * j + e)) & 1;
                        (part ? yB : yA)[4 * j + e] = (part ? fB : fA) * (float) (neg ? -g[e] : g[e]);
                    }
                }
        } else {
            const float d = f16(blk + 80) * 0.125f;
            auto scale = [&](int i) { return d * (float) (1 + 2 * (int) ((blk[72 + i / 2] >> (4 * (i & 1))) & 0xF)); };
            const unsigned hb = blk[64 + uu];
            for (int g = 0; g < 4; g++) {
                const unsigned sg = blk[8 * uu + 4 + g];
                const float f = g < 2 ? scale(4 * p + hh) : scale(4 * p + 2 + hh);
                float * o = (g < 2 ? yA : yB) + 8 * (g & 1);
                int8_t v[8];
                group_values(type, blk[8 * uu + g], (hb >> (2 * g)) & 3, sg, v);
                for (int e = 0; e < 8; e++) o[e] = f * (float) v[e];
            }
        }
    }
}

// the same block as the prefill GEMM reads it: stage p copies [16 B at 16 p | 16 B header | 8 B], and the thread
// (t4) of A-fragment word wd[j][kk] holds elements 16 kk + 4 t4 .. + 3 of sub-block 2 p + j
static void gemm_block(int type, const uint8_t * blk, float * y) {
    const bool iq3 = type == gguf::IQ3_S || type == gguf::IQ3_XXS, xxs = type == gguf::IQ3_XXS;
    if (type == gguf::Q3_K) {
        for (int p = 0; p < 4; p++) {
            uint8_t rr[48] = {};
            memcpy(rr, blk + 16 * p, 16);
            memcpy(rr + 16, blk + 96, 16);
            memcpy(rr + 32, blk + 64 + 8 * p, 8);
            for (int j = 0; j < 2; j++)
                for (int kk = 0; kk < 2; kk++) {
                    const int i = 2 * (2 * p + j) + kk;
                    const int sc = (int) (((rr[16 + i / 2] >> (4 * (i & 1))) & 15) | (((rr[24 + i / 4] >> (2 * (i % 4))) & 3) << 4)) - 32;
                    const float S = f16(rr + 28) * (float) sc;
                    uint32_t w, hw;
                    memcpy(&w, rr + 8 * kk + 4 * j, 4);
                    memcpy(&hw, rr + 32 + 4 * kk, 4);
                    for (int t4 = 0; t4 < 4; t4++)
                        for (int e = 0; e < 4; e++) {
                            const int q = (int) (((w >> (2 * t4 + 8 * e)) & 3) | (((hw >> (16 * j + 4 * t4 + e)) & 1) << 2)) - 4;
                            y[64 * p + 32 * j + 16 * kk + 4 * t4 + e] = S * (float) q;
                        }
                }
        }
        return;
    }
    if (type == gguf::Q2_K) {
        for (int p = 0; p < 4; p++) {
            uint8_t rr[48] = {};
            memcpy(rr, blk + 16 * p, 16);
            memcpy(rr + 16, blk + 64, 16);
            memcpy(rr + 32, blk + 80, 8);
            for (int j = 0; j < 2; j++)
                for (int kk = 0; kk < 2; kk++) {
                    const unsigned sc = rr[16 + 2 * (2 * p + j) + kk];
                    const float S = f16(rr + 32) * (float) (sc & 15), M = -f16(rr + 34) * (float) (sc >> 4);
                    for (int t4 = 0; t4 < 4; t4++) {
                        uint32_t w;
                        memcpy(&w, rr + 8 * kk + 4 * j, 4);
                        w = (w >> (2 * t4)) & 0x03030303u;
                        for (int e = 0; e < 4; e++) y[64 * p + 32 * j + 16 * kk + 4 * t4 + e] = (float) ((w >> (8 * e)) & 0xff) * S + M;
                    }
                }
        }
        return;
    }
    for (int p = 0; p < 4; p++) {
        uint8_t rr[48] = {};
        memcpy(rr, blk + 16 * p, 16);
        memcpy(rr + 16, blk + (iq3 ? 96 : 64), 16);
        memcpy(rr + 32, blk + (iq3 ? 64 + 8 * p : 80), 8);
        for (int j = 0; j < 2; j++)
            for (int kk = 0; kk < 2; kk++) {
                float S;
                if (iq3) {
                    const float d = f16(rr + 24) * (xxs ? 0.25f : 1.0f);
                    const unsigned sc = rr[26 + p];
                    S = d * (float) (1 + 2 * (int) (j ? sc >> 4 : sc & 15));
                } else {
                    const float d = f16(rr + 32) * 0.125f;
                    const int i = 2 * (2 * p + j) + kk;
                    S = d * (float) (1 + 2 * (int) ((rr[24 + i / 2] >> (4 * (i & 1))) & 0xF));
                }
                for (int t4 = 0; t4 < 4; t4++) {
                    int8_t v[4];
                    if (iq3) {
                        const unsigned qh = xxs ? 0u : rr[16 + 2 * p + kk];
                        uint32_t sw;
                        memcpy(&sw, rr + 32 + 4 * kk, 4);
                        const unsigned idx = rr[8 * kk + 4 * j + t4] | (((qh >> (4 * j + t4)) & 1) << 8);
                        const uint8_t * g = (const uint8_t *) ((xxs ? iq3xxs_grid : iq3s_grid) + idx);
                        for (int e = 0; e < 4; e++) v[e] = (int8_t) (((sw >> (16 * j + 4 * t4 + e)) & 1) ? -g[e] : g[e]);
                    } else {
                        const unsigned hb = rr[16 + 2 * p + kk];
                        const int gi = 2 * j + (t4 >> 1);
                        int8_t v8[8];
                        group_values(type, rr[8 * kk + gi], (hb >> (2 * gi)) & 3, rr[8 * kk + 4 + gi], v8);
                        for (int e = 0; e < 4; e++) v[e] = v8[4 * (t4 & 1) + e];
                    }
                    for (int e = 0; e < 4; e++) y[64 * p + 32 * j + 16 * kk + 4 * t4 + e] = S * (float) v[e];
                }
            }
    }
}

// repack + both decodes of `rows` rows of K weights of `type` at src, against ggml's dequantization; max |diff|
static double check_rows(int type, const uint8_t * src_rows, int64_t rows, int64_t K) {
    const int64_t nb = K / 256;
    const size_t srb = gguf::type_row_bytes(type, K), blk = repacked_block_bytes(type);
    const auto * tr = ggml_get_type_traits((ggml_type) type);
    std::vector<float> ref(K), y1(K), y2(K);
    std::vector<uint8_t> pk(nb * blk);
    double maxd = 0;
    for (int64_t r = 0; r < rows; r++) {
        const uint8_t * src = src_rows + (size_t) r * srb;
        tr->to_float(src, ref.data(), K);
        repack_blocks(type, src, pk.data(), nb);
        for (int64_t b = 0; b < nb; b++) {
            gemv_block(type, pk.data() + b * blk, y1.data() + 256 * b);
            gemm_block(type, pk.data() + b * blk, y2.data() + 256 * b);
        }
        for (int64_t e = 0; e < K; e++) maxd = std::max({maxd, (double) fabsf(y1[e] - ref[e]), (double) fabsf(y2[e] - ref[e])});
    }
    return maxd;
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s model.gguf [rows per tensor]\n       %s --synthetic\n", argv[0], argv[0]);
        return 2;
    }
    if (std::string(argv[1]) == "--synthetic") { // random rows quantized by ggml (an all-ones importance matrix)
        const int64_t K = 4096, rows = 64;
        std::vector<float> x((size_t) K * rows), imat(K, 1.0f);
        uint32_t st = 12345;
        for (auto & v : x) { st = st * 1664525u + 1013904223u; v = ((int) (st >> 9) % 2001 - 1000) / 1000.0f * (1 + (st & 3)); }
        int bad = 0;
        for (int type : {gguf::IQ3_S, gguf::IQ3_XXS, gguf::IQ2_XXS, gguf::IQ2_XS, gguf::IQ2_S, gguf::IQ1_S, gguf::IQ1_M, gguf::Q2_K, gguf::Q3_K}) {
            ggml_quantize_init((ggml_type) type);
            std::vector<uint8_t> q(gguf::type_row_bytes(type, K) * rows);
            ggml_quantize_chunk((ggml_type) type, x.data(), q.data(), 0, rows, K, imat.data());
            const double d = check_rows(type, q.data(), rows, K);
            printf("synthetic %-8s %lld rows x %lld: max |diff| %g %s\n", ggml_type_name((ggml_type) type), (long long) rows,
                   (long long) K, d, d == 0 ? "exact" : "MISMATCH");
            bad += d != 0;
        }
        return bad ? 1 : 0;
    }
    const int max_rows = argc > 2 ? atoi(argv[2]) : 256;
    gguf::File file(argv[1]);
    int checked[64] = {}, bad = 0;
    for (const auto & [name, t] : file.tensors) {
        const int type = (int) t.type;
        const bool iq3 = type == gguf::IQ3_S || type == gguf::IQ3_XXS;
        const bool iq2 = type == gguf::IQ2_XXS || type == gguf::IQ2_XS || type == gguf::IQ2_S || type == gguf::IQ1_S ||
                         type == gguf::IQ1_M, q2k = type == gguf::Q2_K || type == gguf::Q3_K;
        if (!(iq3 || iq2 || q2k) || checked[type] >= 3) continue;
        if (t.data + t.nbytes > file.base + file.size) continue; // (a partly downloaded file)
        const int64_t K = (int64_t) t.ne[0], rows = std::min<int64_t>((int64_t) t.ne[1], max_rows);
        const double maxd = check_rows(type, t.data, rows, K);
        const auto * tr = ggml_get_type_traits((ggml_type) type);
        printf("%-36s %-8s %lld rows x %lld: max |diff| %g %s\n", name.c_str(), tr->type_name, (long long) rows, (long long) K,
               maxd, maxd == 0 ? "exact" : "MISMATCH");
        bad += maxd != 0;
        checked[type]++;
    }
    return bad ? 1 : 0;
}
