// Load-time re-layout of ggml's i-quant blocks for qwen38_mixed.cu (decode GEMV rows_iq3 / rows_iq2 and the prefill
// GEMM). Host code; also compiled by tools/iq_repack_check.cpp, which checks the layouts against ggml's dequantization.
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>

#include "gguf.h"

// GGUF IQ3_S (110 B: fp16 d | qs[64] | qh[8] | signs[32] | scales[4]) or IQ3_XXS blocks (98 B: fp16 d | qs[64] |
// per sub-block a u32 of four 7-bit ksigns_iq2xs indices and the 4-bit scale) -> the 112-byte layout rows_iq3 and
// the prefill GEMM read (see rows_iq3). Lossless: IQ3_XXS's sign bytes get their parity bit, as ksigns_iq2xs does.
inline void repack_iq3(const uint8_t * src, uint8_t * dst, size_t nblocks, bool xxs) {
    const size_t sbytes = xxs ? 98 : 110;
    for (size_t i = 0; i < nblocks; i++) {
        const uint8_t *s = src + i * sbytes, *qs = s + 2;
        uint8_t * o = dst + i * 112;
        uint8_t sg[32], qh[8] = {}, sc[4] = {};
        if (xxs) {
            for (int ib = 0; ib < 8; ib++) {
                uint32_t aux;
                memcpy(&aux, s + 66 + 4 * ib, 4);
                for (int l = 0; l < 4; l++) {
                    const uint32_t v = (aux >> (7 * l)) & 127;
                    sg[4 * ib + l] = (uint8_t) (v | ((__builtin_popcount(v) & 1) << 7));
                }
                sc[ib / 2] |= (uint8_t) ((aux >> 28) << (4 * (ib & 1)));
            }
        } else {
            memcpy(qh, s + 66, 8);
            memcpy(sg, s + 74, 32);
            memcpy(sc, s + 106, 4);
        }
        memset(o, 0, 112);
        for (int uu = 0; uu < 8; uu++) {
            const int h = uu & 1, a = 2 * (uu >> 1), b = a + 1;
            memcpy(o + 8 * uu, qs + 8 * a + 4 * h, 4);
            memcpy(o + 8 * uu + 4, qs + 8 * b + 4 * h, 4);
            memcpy(o + 64 + 4 * uu, sg + 4 * a + 2 * h, 2);
            memcpy(o + 64 + 4 * uu + 2, sg + 4 * b + 2 * h, 2);
            o[96 + uu] = (uint8_t) (((qh[a] >> (4 * h)) & 0xF) | (((qh[b] >> (4 * h)) & 0xF) << 4));
        }
        memcpy(o + 104, s, 2);
        memcpy(o + 106, sc, 4);
    }
}

// GGUF IQ2_XXS (66 B: fp16 d | per 32-group two u32: four grid indices (8 bits), four 7-bit ksigns_iq2xs indices and
// the 4-bit scale), IQ2_XS (74 B: fp16 d | u16 qs[32]: 9-bit index, 7-bit ksigns index | u8 scales[8], two 4-bit per
// 32-group) or IQ2_S blocks (82 B: fp16 d | qs[32] index bits 0-7 | signs[32] | qh[8] index bits 8-9 | scales[8])
// -> the 96-byte layout rows_iq2 and the prefill GEMM read (see rows_iq2). Lossless.
// IQ1_S (50 B: fp16 d | qs[32] index bits 0-7 | u16 qh[8]: index bits 8-10 of the four groups, 3-bit scale, delta sign)
// and IQ1_M (56 B: qs[32] | qh[16]: per 8-group index bits 8-10 and delta sign | scales[8]: 3-bit scales per 16-group
// with the fp16 d spread over their top nibbles) take the same layout: the sign byte of a group holds its index bits
// 8-10 and, in bit 3, the sign of its delta; a weight is d (1 + 2 s) (8 g + / - 1) / 8, g in {-1, 0, 1}.
// GGUF Q2_K blocks (84 B: scales[16] (4-bit scale | 4-bit min per 16-group) | qs[64] | fp16 d | fp16 dmin) -> 96 B:
// unit uu = 2 p + h at 8 uu: A's 16 2-bit values then B's (A and B as for rows_iq3), value i of a chunk at bit
// 8 (i % 4) + 2 (i / 4), so that (word >> 2 m) & 0x03030303 is values 4 m .. 4 m + 3; scales[16] at 64; d, dmin at 80.
inline void repack_q2k(const uint8_t * src, uint8_t * dst, size_t nblocks) {
    for (size_t i = 0; i < nblocks; i++) {
        const uint8_t *s = src + i * 84, *qs = s + 16;
        uint8_t * o = dst + i * 96;
        memset(o, 0, 96);
        auto q = [&](int e) { return (qs[32 * (e >> 7) + (e & 31)] >> (2 * ((e >> 5) & 3))) & 3; };
        for (int uu = 0; uu < 8; uu++) {
            const int e0 = 64 * (uu >> 1) + 16 * (uu & 1);
            uint32_t a = 0, b = 0;
            for (int k = 0; k < 16; k++) {
                a |= (uint32_t) q(e0 + k) << (8 * (k % 4) + 2 * (k / 4));
                b |= (uint32_t) q(e0 + 32 + k) << (8 * (k % 4) + 2 * (k / 4));
            }
            memcpy(o + 8 * uu, &a, 4);
            memcpy(o + 8 * uu + 4, &b, 4);
        }
        memcpy(o + 64, s, 16);
        memcpy(o + 80, s + 80, 4);
    }
}

inline void repack_iq2(const uint8_t * src, uint8_t * dst, size_t nblocks, int type);

// GGUF Q3_K blocks (110 B: hmask[32] | qs[64] | scales[12], 6-bit | fp16 d) -> 112 B: unit uu's A then B low 2-bit
// values at 8 uu (packed as repack_q2k's), their third bits at 64 + 4 uu (A's u16 then B's, bit i = value i), the 16
// 6-bit scales at 96 (low nibbles, 16-group i in byte 96 + i / 2) and 104 (top 2 bits, byte 104 + i / 4), fp16 d at
// 108.  A weight is d (sc - 32) ((lo | hi << 2) - 4).
inline void repack_q3k(const uint8_t * src, uint8_t * dst, size_t nblocks) {
    for (size_t i = 0; i < nblocks; i++) {
        const uint8_t *s = src + i * 110, *hm = s, *qs = s + 32;
        uint8_t * o = dst + i * 112;
        memset(o, 0, 112);
        uint32_t aux[4];
        memcpy(aux, s + 96, 12);
        const uint32_t m1 = 0x03030303u, m2 = 0x0f0f0f0fu, tmp = aux[2];
        aux[2] = ((aux[0] >> 4) & m2) | (((tmp >> 4) & m1) << 4);
        aux[3] = ((aux[1] >> 4) & m2) | (((tmp >> 6) & m1) << 4);
        aux[0] = (aux[0] & m2) | (((tmp >> 0) & m1) << 4);
        aux[1] = (aux[1] & m2) | (((tmp >> 2) & m1) << 4);
        const uint8_t * sc = (const uint8_t *) aux; // the 16 6-bit scales, ggml order
        auto lo = [&](int e) { return (qs[32 * (e >> 7) + (e & 31)] >> (2 * ((e >> 5) & 3))) & 3; };
        auto hi = [&](int e) { return (hm[e & 31] >> (4 * (e >> 7) + ((e >> 5) & 3))) & 1; };
        for (int uu = 0; uu < 8; uu++) {
            const int e0 = 64 * (uu >> 1) + 16 * (uu & 1);
            uint32_t a = 0, b = 0;
            uint16_t ha = 0, hb = 0;
            for (int k = 0; k < 16; k++) {
                a |= (uint32_t) lo(e0 + k) << (8 * (k % 4) + 2 * (k / 4));
                b |= (uint32_t) lo(e0 + 32 + k) << (8 * (k % 4) + 2 * (k / 4));
                ha |= (uint16_t) (hi(e0 + k) << k);
                hb |= (uint16_t) (hi(e0 + 32 + k) << k);
            }
            memcpy(o + 8 * uu, &a, 4);
            memcpy(o + 8 * uu + 4, &b, 4);
            memcpy(o + 64 + 4 * uu, &ha, 2);
            memcpy(o + 64 + 4 * uu + 2, &hb, 2);
        }
        for (int g = 0; g < 16; g++) {
            o[96 + g / 2] |= (uint8_t) ((sc[g] & 15) << (4 * (g & 1)));
            o[104 + g / 4] |= (uint8_t) ((sc[g] >> 4) << (2 * (g % 4)));
        }
        memcpy(o + 108, s + 108, 2);
    }
}

// Bytes of a repacked 256-weight block of a ggml type, 0 for types the kernels read as stored (or not at all)
inline size_t repacked_block_bytes(int type) {
    switch (type) {
        case gguf::IQ3_S: case gguf::IQ3_XXS: case gguf::Q3_K: return 112;
        case gguf::IQ2_XXS: case gguf::IQ2_XS: case gguf::IQ2_S: case gguf::IQ1_S: case gguf::IQ1_M: case gguf::Q2_K: return 96;
        default: return 0;
    }
}
inline void repack_blocks(int type, const uint8_t * src, uint8_t * dst, size_t nblocks) {
    if (type == gguf::IQ3_S || type == gguf::IQ3_XXS) repack_iq3(src, dst, nblocks, type == gguf::IQ3_XXS);
    else if (type == gguf::Q2_K) repack_q2k(src, dst, nblocks);
    else if (type == gguf::Q3_K) repack_q3k(src, dst, nblocks);
    else repack_iq2(src, dst, nblocks, type);
}

inline void repack_iq2(const uint8_t * src, uint8_t * dst, size_t nblocks, int type) {
    const size_t sbytes = type == gguf::IQ2_XXS ? 66 : type == gguf::IQ2_XS ? 74 : type == gguf::IQ2_S ? 82
                        : type == gguf::IQ1_S ? 50 : 56;
    auto ksign = [](uint32_t v) { return (uint8_t) (v | ((__builtin_popcount(v) & 1) << 7)); };
    for (size_t i = 0; i < nblocks; i++) {
        const uint8_t * s = src + i * sbytes;
        uint8_t * o = dst + i * 96;
        memset(o, 0, 96);
        for (int sb = 0; sb < 8; sb++) {
            uint8_t sc[2]; // the 4-bit scales of the two 16-groups
            for (int l = 0; l < 4; l++) {
                uint32_t idx, sgn;
                if (type == gguf::IQ2_XXS) {
                    uint32_t aux[2];
                    memcpy(aux, s + 2 + 8 * sb, 8);
                    idx = (aux[0] >> (8 * l)) & 0xff;
                    sgn = ksign((aux[1] >> (7 * l)) & 127);
                    sc[0] = sc[1] = (uint8_t) (aux[1] >> 28);
                } else if (type == gguf::IQ2_XS) {
                    uint16_t q;
                    memcpy(&q, s + 2 + 2 * (4 * sb + l), 2);
                    idx = q & 511;
                    sgn = ksign(q >> 9);
                    sc[0] = s[66 + sb] & 0xf;
                    sc[1] = s[66 + sb] >> 4;
                } else if (type == gguf::IQ1_S) {
                    uint16_t qh;
                    memcpy(&qh, s + 34 + 2 * sb, 2);
                    idx = s[2 + 4 * sb + l] | (((qh >> (3 * l)) & 7) << 8);
                    sgn = (idx >> 8) | ((qh >> 15) << 3);
                    idx &= 0xff;
                    sc[0] = sc[1] = (uint8_t) ((qh >> 12) & 7);
                } else if (type == gguf::IQ1_M) {
                    const uint8_t qh = s[32 + 2 * sb + (l >> 1)] >> (4 * (l & 1));
                    idx = s[4 * sb + l];
                    sgn = (qh & 7) | (qh & 8);
                    uint16_t sc16;
                    memcpy(&sc16, s + 48 + 2 * (sb / 2), 2);
                    sc[0] = (uint8_t) ((sc16 >> (6 * (sb % 2))) & 7);
                    sc[1] = (uint8_t) ((sc16 >> (6 * (sb % 2) + 3)) & 7);
                } else {
                    idx = s[2 + 4 * sb + l] | ((s[66 + sb] << (8 - 2 * l)) & 0x300);
                    sgn = s[34 + 4 * sb + l];
                    sc[0] = s[74 + sb] & 0xf;
                    sc[1] = s[74 + sb] >> 4;
                }
                const int h = l >> 1, uu = 2 * (sb >> 1) + h, g = 2 * (sb & 1) + (l & 1);
                o[8 * uu + g] = (uint8_t) (idx & 0xff);
                o[8 * uu + 4 + g] = (uint8_t) sgn;
                o[64 + uu] |= (uint8_t) ((idx >> 8) << (2 * g));
            }
            for (int h = 0; h < 2; h++) o[72 + sb] |= (uint8_t) (sc[h] << (4 * h)); // 16-group 2 sb + h
        }
        if (type == gguf::IQ1_M) {
            uint16_t sc[4];
            memcpy(sc, s + 48, 8);
            const uint16_t d = (uint16_t) ((sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000));
            memcpy(o + 80, &d, 2);
        } else {
            memcpy(o + 80, s, 2);
        }
    }
}

