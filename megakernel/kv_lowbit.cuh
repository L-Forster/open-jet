// Low-bit K/V cache formats (MK_KV) and the q8 recent-window ring (MK_KV_WIN), shared by the engines (qwen38_mk.cu,
// qwen38_mixed.cu): included after the engine's kv_store_q8. Nothing here runs unless MK_KV is set (cB.kvlow).
//
// Per attention slot (16 layers + MTP) and K / V: bits 4..8 per value (8 = q8_0). A q4..q7 row is HD bits / 8 bytes:
// the lane words (low nibbles) in the first 128 bytes, then lane l's high bits at 128 + l (bits - 4). Scales stay one
// fp16 per 32 values in the q8 layout (ksc / vsc, same row index). Slots are packed one after the other in kc / vc, so an
// all-q8 cache has the old [slot][kv head][ctx][HD] addresses.
#pragma once

// q4 store of one KV row (row = HD / 2 bytes), q4_0 rounding: per 32-value block d = (the value of largest
// magnitude) / -8, nibble = min(15, (int) (x / d + 8.5)), value = (nibble - 8) d. Lane holds values 8 lane .. + 7 and
// writes one word: low nibbles = values 0..3, high nibbles = values 4..7. Ties of |x| go to the larger x (the same
// in all 4 lanes of a block).
__device__ __forceinline__ void kv_store_q4(int8_t * q, __half * sc, const float (&v)[8]) {
    const int lane = threadIdx.x & 31;
    float am = 0.f, sv = 0.f;
#pragma unroll
    for (int k = 0; k < 8; k++)
        if (fabsf(v[k]) > am || (fabsf(v[k]) == am && v[k] > sv)) { am = fabsf(v[k]); sv = v[k]; }
#pragma unroll
    for (int m = 1; m <= 2; m <<= 1) {
        const float oa = __shfl_xor_sync(0xffffffffu, am, m), os = __shfl_xor_sync(0xffffffffu, sv, m);
        if (oa > am || (oa == am && os > sv)) { am = oa; sv = os; }
    }
    const float d = sv / -8.0f, id = d != 0.f ? 1.0f / d : 0.f;
    unsigned w = 0;
#pragma unroll
    for (int k = 0; k < 8; k++) w |= (unsigned) min(15, (int) (v[k] * id + 8.5f)) << (k < 4 ? 8 * k : 8 * (k - 4) + 4);
    *reinterpret_cast<unsigned *>(q + lane * 4) = w;
    if ((lane & 3) == 0) sc[lane >> 2] = __float2half_rn(d);
}

// 256-point fast Walsh-Hadamard transform of one row, 8 values per lane (scaled by 1/16: orthonormal, its own inverse)
__device__ __forceinline__ void fwht256(float (&v)[8]) {
#pragma unroll
    for (int m = 1; m < 8; m <<= 1)
#pragma unroll
        for (int k = 0; k < 8; k++)
            if (!(k & m)) {
                const float a = v[k], b = v[k | m];
                v[k] = a + b;
                v[k | m] = a - b;
            }
    const int lane = threadIdx.x & 31;
#pragma unroll
    for (int m = 1; m < 32; m <<= 1)
#pragma unroll
        for (int k = 0; k < 8; k++) {
            const float o = __shfl_xor_sync(0xffffffffu, v[k], m);
            v[k] = (lane & m) ? o - v[k] : v[k] + o;
        }
#pragma unroll
    for (int k = 0; k < 8; k++) v[k] *= 0.0625f;
}

// q5 / q6 / q7 store of one KV row (HD bits / 8 bytes): the q4 word of each lane (low nibbles of the 2^bits-level index
// n = q + 2^(bits-1), kv_store_q4's nibble order) in the first 128 bytes, then the high bits, bits - 4 per value: lane l's
// 8 values at bytes 128 + l (bits - 4) .., value k at bit (bits - 4) k. q4_0-style scale d = (signed max) / -2^(bits-1),
// value = (n - 2^(bits-1)) d. bits = 4 would give kv_store_q4's bytes.
__device__ __forceinline__ void kv_store_qn(int8_t * q, __half * sc, const float (&v)[8], int bits) {
    const int lane = threadIdx.x & 31, h = 1 << (bits - 1), hb = bits - 4;
    float am = 0.f, sv = 0.f;
#pragma unroll
    for (int k = 0; k < 8; k++)
        if (fabsf(v[k]) > am || (fabsf(v[k]) == am && v[k] > sv)) { am = fabsf(v[k]); sv = v[k]; }
#pragma unroll
    for (int m = 1; m <= 2; m <<= 1) {
        const float oa = __shfl_xor_sync(0xffffffffu, am, m), os = __shfl_xor_sync(0xffffffffu, sv, m);
        if (oa > am || (oa == am && os > sv)) { am = oa; sv = os; }
    }
    const float d = sv / (float) -h, id = d != 0.f ? 1.0f / d : 0.f;
    unsigned lo = 0, hi = 0;
#pragma unroll
    for (int k = 0; k < 8; k++) {
        const unsigned n = (unsigned) min(2 * h - 1, (int) (v[k] * id + ((float) h + 0.5f)));
        lo |= (n & 15u) << (k < 4 ? 8 * k : 8 * (k - 4) + 4);
        hi |= (n >> 4) << (hb * k);
    }
    *reinterpret_cast<unsigned *>(q + lane * 4) = lo;
    for (int k = 0; k < hb; k++) q[128 + lane * hb + k] = (int8_t) ((hi >> (8 * k)) & 0xffu);
    if ((lane & 3) == 0) sc[lane >> 2] = __float2half_rn(d);
}

// one row in a format of `bits` (q: the row's value bytes, sc: its HD / 32 scales)
__device__ __forceinline__ void kv_store_fmt(int8_t * q, __half * sc, const float (&v)[8], int bits) {
    if (bits == 8) kv_store_q8(q, sc, v);
    else if (bits == 4) kv_store_q4(q, sc, v);
    else kv_store_qn(q, sc, v, bits);
}

// one K (w = 0) or V (w = 1) row (kv head g, position pos) of attention slot `slot` in its MK_KV format; with a
// recent-window ring (MK_KV_WIN) the row is also stored q8 in the ring (same rotation: every tile of a slot must share
// the basis). The engine's own contiguous cache (single stream).
__device__ __forceinline__ void kv_store_row(int w, int slot, int g, int pos, const float (&vin)[8]) {
    float v[8];
#pragma unroll
    for (int k = 0; k < 8; k++) v[k] = vin[k];
    if (w == 0 ? cB.kvr[slot] : cB.kvrv[slot]) fwht256(v);
    const int bits = cB.kvb[w][slot];
    const size_t row = (size_t) g * cB.ctx + pos;
    kv_store_fmt(cB.kvp[w][slot] + row * (HD * bits / 8), cB.kvs[w][slot] + row * (HD / 32), v, bits);
    if (cB.kvrp[w][slot]) {
        const size_t rr = (size_t) g * cB.kvrc + pos % cB.kvrc;
        kv_store_q8(cB.kvrp[w][slot] + rr * HD, cB.kvrs[w][slot] + rr * (HD / 32), v);
    }
}

// MK_KV_WIN: the window start for context length L_ctx (positions >= ws are read from the q8 ring). Stepped by ATT_BUCKET
// like the attention partition, so all rows of one launch and every draft depth see the same ring / main split of the
// tiles: greedy output stays independent of the draft depth. The window covers Wb ATT_BUCKET + 1 .. (Wb + 1) ATT_BUCKET
// positions (>= W), the ring holds (Wb + 1) ATT_BUCKET rows (+ ATP slack, see the allocation).
__host__ __device__ __forceinline__ int kv_ws_of(int L_ctx, int win) {
    if (win <= 0) return 0;
    const int ws = (L_ctx + ATT_BUCKET - 1) / ATT_BUCKET - 1 - (win + ATT_BUCKET - 1) / ATT_BUCKET;
    return ws > 0 ? ws * ATT_BUCKET : 0;
}
__device__ __forceinline__ int kv_ws(int L_ctx) { return kv_ws_of(L_ctx, cB.kvwin); }
// bits of the tile at position p0 (a multiple of ATP) of side w of a slot: 8 from the ring, else the main format
__device__ __forceinline__ int kv_tbits(int w, int slot, int p0, int ws) {
    return (cB.kvrp[w][slot] != nullptr && p0 >= ws) ? 8 : cB.kvb[w][slot];
}
// ws % kvrc, computed once per launch / item (a runtime integer modulo is ~25 dependent instructions)
__device__ __forceinline__ int kv_wsm(int ws) { return cB.kvrc > 0 ? ws % cB.kvrc : 0; }
// Per-slot format facts, loaded once per launch: the hot kernels then decide a tile's format and source with one
// integer compare of the tile index against kv_ring_tile (per-tile kv_tbits / kv_src lookups cost a uniform-datapath
// chain right after the tile barrier).
struct KvFmt { int kb, vb, kr, vr; }; // main-buffer bits of K / V; 1 if the side has a q8 ring (MK_KV_WIN)
__device__ __forceinline__ KvFmt kv_fmt(int slot) {
    return {cB.kvb[0][slot], cB.kvb[1][slot], cB.kvrp[0][slot] != nullptr, cB.kvrp[1][slot] != nullptr};
}
// first index of the tiles of tp positions starting at position base that comes from the ring (p0 >= ws); none: INT_MAX
__device__ __forceinline__ int kv_ring_tile(int ring, int ws, int base, int tp) {
    return ring ? max(0, (ws - base + tp - 1) / tp) : 0x7fffffff;
}
// the tile's rows: q / s = values / scales of its first row (ring: the slot's q8 ring, else the main buffer of `bits`).
// wsm = kv_wsm(ws): a ring tile's row is p0 % kvrc = wsm + (p0 - ws) folded once (positions ws .. L_ctx - 1 span at most
// kvrc rows)
__device__ __forceinline__ void kv_tile_src(int w, int slot, int g, int p0, int ws, int wsm, int bits, bool ring, const int8_t *& q, const __half *& s) {
    if (ring) {
        int r0 = wsm + (p0 - ws);
        if (r0 >= cB.kvrc) r0 -= cB.kvrc;
        const size_t rr = (size_t) g * cB.kvrc + r0;
        q = cB.kvrp[w][slot] + rr * HD;
        s = cB.kvrs[w][slot] + rr * (HD / 32);
    } else {
        const size_t row = (size_t) g * cB.ctx + p0;
        q = cB.kvp[w][slot] + row * (HD * bits / 8);
        s = cB.kvs[w][slot] + row * (HD / 32);
    }
}
// kv_tile_src with the decision made here, returns the tile's bits
__device__ __forceinline__ int kv_src(int w, int slot, int g, int p0, int ws, int wsm, const int8_t *& q, const __half *& s) {
    const bool ring = cB.kvrp[w][slot] != nullptr && p0 >= ws;
    const int bits = cB.kvb[w][slot];
    kv_tile_src(w, slot, g, p0, ws, wsm, bits, ring, q, s);
    return ring ? 8 : bits;
}

// 8 q4 values (one word in kv_store_q4's nibble order) + scale -> 4 fp16 pairs, the q8_to_f16 arithmetic
__device__ __forceinline__ void q4_to_f16(unsigned w, float d, __half2 (&h)[4]) {
#pragma unroll
    for (int k = 0; k < 4; k++) {
        const int j0 = 2 * k, j1 = 2 * k + 1;
        const int a = (int) ((w >> (j0 < 4 ? 8 * j0 : 8 * (j0 - 4) + 4)) & 0xfu) - 8;
        const int b = (int) ((w >> (j1 < 4 ? 8 * j1 : 8 * (j1 - 4) + 4)) & 0xfu) - 8;
        h[k] = __floats2half2_rn(d * (float) a, d * (float) b);
    }
}

// the high bits of one lane word's 8 values of a q5 / q6 / q7 row: hb = bits - 4 bytes at p, value k at bit hb k
__device__ __forceinline__ unsigned kv_hi(const uint8_t * p, int hb) {
    unsigned x = p[0];
    if (hb > 1) x |= (unsigned) p[1] << 8;
    if (hb > 2) x |= (unsigned) p[2] << 16;
    return x;
}
// the hb-bit fields k0 .. k0 + 3 of x as the high nibbles of 4 bytes (OR into a word of 4 low nibbles)
__device__ __forceinline__ unsigned kv_hi4(unsigned x, int k0, int hb) {
    const unsigned m = (1u << hb) - 1u;
    x >>= hb * k0;
    return ((x & m) | (((x >> hb) & m) << 8) | (((x >> (2 * hb)) & m) << 16) | (((x >> (3 * hb)) & m) << 24)) << 4;
}
// the four HB-bit fields k = 0 .. 3 of x (field k at bit HB k) -> bits 4 .. 4 + HB - 1 of byte k (the high nibbles of a
// word of four low nibbles, as kv_hi4 with a compile-time width)
template <int HB>
__device__ __forceinline__ unsigned kv_spread4(unsigned x) {
    if constexpr (HB == 1) return ((x & 0xfu) * 0x02040810u) & 0x10101010u; // bit k -> bit 8 k + 4 (no two products collide)
    if constexpr (HB == 2) {
        // two IMAD + two LOP3 (the shifted copies never overlap, so the multiply is the OR): x * 0x10010 puts fields
        // 0, 1 at bits 4 .. 7 and 2, 3 at 20 .. 23 (mask), then * 0x41 moves field 1 to 12 .. 13 and 3 to 28 .. 29
        x = ((x & 0xffu) * 0x10010u) & 0x00f000f0u;
        return (x * 0x41u) & 0x30303030u;
    }
    if constexpr (HB == 3) {
        x &= 0xfffu;
        x = (x | (x << 10)) & 0x003f003fu;
        x = (x | (x << 5)) & 0x07070707u;
        return x << 4;
    }
    return 0u;
}
// Staged q4 .. q7 K / V rows expanded IN PLACE to the q8 layout: 256 int8 values q = n - 2^(BITS - 1), the scales
// untouched. The fp16 product d q is the same number the low-bit fp16 conversion computes (qn_to_f16), so every
// consumer of the q8 layout sees exactly the low-bit values.
// One warp per row, up to 3 rows (r0, r0 + rs, r0 + 2 rs, all < np) per call, loads of all rows first, then every lane has
// read its words before any lane writes (the output overlaps the input).
// SIGNED = false leaves the biased bytes n (0 .. 2^BITS - 1) in place: the fp16 consumers subtract the bias for free in
// their magic-number conversion; the int8 IMMA consumer (pk_attn8) needs q.
template <int BITS, bool SIGNED = true>
__device__ __forceinline__ void kv_rows_expand(uint8_t * st, int np, int r0, int rs, int ld) {
    constexpr int HB = BITS - 4;
    const int lane = threadIdx.x & 31;
    unsigned wd[3], hi[3];
#pragma unroll
    for (int k = 0; k < 3; k++) {
        const int r = r0 + k * rs;
        wd[k] = 0u;
        hi[k] = 0u;
        if (r < np) {
            const uint8_t * row = st + r * ld;
            wd[k] = *reinterpret_cast<const unsigned *>(row + 4 * lane);
            if constexpr (HB == 1) hi[k] = row[128 + lane];
            if constexpr (HB == 2) hi[k] = *reinterpret_cast<const unsigned short *>(row + 128 + 2 * lane);
            if constexpr (HB == 3) hi[k] = (unsigned) row[128 + 3 * lane] | ((unsigned) row[129 + 3 * lane] << 8) | ((unsigned) row[130 + 3 * lane] << 16);
        }
    }
    __syncwarp();
#pragma unroll
    for (int k = 0; k < 3; k++) {
        const int r = r0 + k * rs;
        if (r < np) {
            unsigned lo = wd[k] & 0x0f0f0f0fu, hv = (wd[k] >> 4) & 0x0f0f0f0fu; // values 0 .. 3 / 4 .. 7 (biased, n = q + 2^(BITS-1))
            if constexpr (HB > 0) {
                lo |= kv_spread4<HB>(hi[k]);
                hv |= kv_spread4<HB>(hi[k] >> (4 * HB));
            }
            if constexpr (SIGNED) {
                // per byte n - 2^(BITS-1) mod 256, without borrows between bytes (n < 128: bit 7 is free to absorb them)
                constexpr unsigned bias = (1u << (BITS - 1)) * 0x01010101u;
                lo = ((lo | 0x80808080u) - bias) ^ 0x80808080u;
                hv = ((hv | 0x80808080u) - bias) ^ 0x80808080u;
            }
            *reinterpret_cast<uint2 *>(st + r * ld + 8 * lane) = make_uint2(lo, hv);
        }
    }
}
// rows r0, r0 + rs, .. < np of a staged low-bit tile (row stride ld bytes) -> q8 layout (SIGNED: int8 q; else the biased
// bytes n, see kv_rows_expand); bits is block-uniform
template <bool SIGNED = true>
__device__ __forceinline__ void kv_tile_expand(uint8_t * st, int bits, int np, int r0, int rs, int ld) {
    switch (bits) {
    case 4:
        for (int r = r0; r < np; r += 3 * rs) kv_rows_expand<4, SIGNED>(st, np, r, rs, ld);
        break;
    case 5:
        for (int r = r0; r < np; r += 3 * rs) kv_rows_expand<5, SIGNED>(st, np, r, rs, ld);
        break;
    case 6:
        for (int r = r0; r < np; r += 3 * rs) kv_rows_expand<6, SIGNED>(st, np, r, rs, ld);
        break;
    case 7:
        for (int r = r0; r < np; r += 3 * rs) kv_rows_expand<7, SIGNED>(st, np, r, rs, ld);
        break;
    default:
        break;
    }
}

// 8 q4 .. q7 values (lane word w in kv_store_q4's nibble order, high bits hi) + scale -> 4 fp16 pairs (q4: q4_to_f16)
__device__ __forceinline__ void qn_to_f16(unsigned w, unsigned hi, float d, int bits, __half2 (&h)[4]) {
    const int hb = bits - 4, mid = 1 << (bits - 1);
    const unsigned m = (1u << hb) - 1u;
#pragma unroll
    for (int k = 0; k < 4; k++) {
        const int j0 = 2 * k, j1 = 2 * k + 1;
        const int a = (int) (((w >> (j0 < 4 ? 8 * j0 : 8 * (j0 - 4) + 4)) & 0xfu) | (((hi >> (hb * j0)) & m) << 4)) - mid;
        const int b = (int) (((w >> (j1 < 4 ? 8 * j1 : 8 * (j1 - 4) + 4)) & 0xfu) | (((hi >> (hb * j1)) & m) << 4)) - mid;
        h[k] = __floats2half2_rn(d * (float) a, d * (float) b);
    }
}

// lane c's 8 values of a staged row (values, then scales at byte 256: KVQ_ROW) in a format of `bits` -> 4 fp16 pairs
__device__ __forceinline__ void kv_row_to_f16(const uint8_t * row, int bits, int c, __half2 (&h)[4]) {
    const float d = __half2float(reinterpret_cast<const __half *>(row + 256)[c >> 2]);
    if (bits == 8) q8_to_f16(*reinterpret_cast<const uint2 *>(row + 8 * c), d, h);
    else if (bits == 4) q4_to_f16(*reinterpret_cast<const unsigned *>(row + 4 * c), d, h);
    else qn_to_f16(*reinterpret_cast<const unsigned *>(row + 4 * c), kv_hi(row + 128 + c * (bits - 4), bits - 4), d, bits, h);
}

// staged K / V rows of per-tensor formats (KVQ_ROW bytes: values, then scales at 256) -> fp16 rows of stride ld; rows x 2
__device__ __forceinline__ void kv_tile_to_f16_fmt(const uint8_t * src, __half * dK, __half * dV, int rows, int ld, int kbits, int vbits) {
    for (int i = threadIdx.x; i < 2 * rows * 32; i += blockDim.x) {
        const int which = i / (rows * 32), r = (i >> 5) % rows, c = i & 31;
        __half2 h[4];
        kv_row_to_f16(src + (size_t) (which * rows + r) * KVQ_ROW, which ? vbits : kbits, c, h);
        *reinterpret_cast<int4 *>((which ? dV : dK) + (size_t) r * ld + 8 * c) = *reinterpret_cast<int4 *>(h);
    }
}

// async copy of K/V rows r < np of one tile in per-tensor formats (Kq / Ks, Vq / Vs: the tile's first row, kv_src) into
// the staging layout (a row of HD bits / 8 bytes = 2 bits chunks of 16; rows >= np: zero scales)
__device__ __forceinline__ void kv_stage_src(uint8_t * dst, const int8_t * Kq, const int8_t * Vq, const __half * Ks, const __half * Vs,
                                             int rows, int np, int kbits, int vbits) {
    for (int i = threadIdx.x; i < 2 * rows * 17; i += blockDim.x) {
        const int which = i / (rows * 17), r = (i / 17) % rows, c = i % 17;
        uint8_t * d = dst + (size_t) (which * rows + r) * KVQ_ROW + 16 * c;
        if (r < np) {
            const int bits = which ? vbits : kbits;
            if (c < 16) {
                if (c < 2 * bits) cp_async16(d, (which ? Vq : Kq) + (size_t) r * (HD * bits / 8) + 16 * c);
            } else {
                cp_async16(d, (which ? Vs : Ks) + (size_t) r * (HD / 32));
            }
        } else if (c == 16) {
            *reinterpret_cast<int4 *>(d) = make_int4(0, 0, 0, 0);
        }
    }
}

// one row in a format of `bits` (rp: its value bytes, d: the scale of this lane's 32-value block) -> this lane's 8 values
__device__ __forceinline__ void kv_row_load(const uint8_t * rp, int bits, float d, float (&v)[8]) {
    const int lane = threadIdx.x & 31;
    if (bits == 8) {
        const uint2 qq = *reinterpret_cast<const uint2 *>(rp + lane * 8);
#pragma unroll
        for (int k = 0; k < 8; k++) v[k] = d * (float) (int8_t) ((k < 4 ? qq.x : qq.y) >> (8 * (k & 3)));
    } else { // q4..q7: lane word (values 0..3 low nibbles, 4..7 high) + bits - 4 high bits per value (kv_store_qn)
        const unsigned wd = *reinterpret_cast<const unsigned *>(rp + lane * 4);
        const int hb = bits - 4;
        const unsigned hi = hb > 0 ? kv_hi(rp + 128 + lane * hb, hb) : 0u;
#pragma unroll
        for (int k = 0; k < 8; k++) {
            const int nb = (int) ((wd >> (k < 4 ? 8 * k : 8 * (k - 4) + 4)) & 0xfu) | (int) (((hi >> (hb * k)) & ((1u << hb) - 1u)) << 4);
            v[k] = d * (float) (nb - (1 << (bits - 1)));
        }
    }
}

// MK_KV_WIN after a rewind to position p1: the ring rows of positions p0 .. p1 - 1 may have been overwritten by the
// abandoned continuation, so they are refilled from the main buffer (its low-bit values, re-stored as q8: the window is
// at the main format's precision until those rows age out). One warp per row. The engine's own contiguous cache.
__global__ void __launch_bounds__(256) k_ring_refill(int p0, int p1) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int n = p1 - p0;
    for (int i = blockIdx.x * 8 + warp; i < n * NKV * N_ATT_SLOTS * 2; i += gridDim.x * 8) {
        const int w = i & 1, slot = (i >> 1) % N_ATT_SLOTS, g = ((i >> 1) / N_ATT_SLOTS) % NKV;
        const int p = p0 + (i >> 1) / (N_ATT_SLOTS * NKV);
        if (!cB.kvrp[w][slot]) continue;
        const int bits = cB.kvb[w][slot];
        const size_t row = (size_t) g * cB.ctx + p;
        const uint8_t * rp = reinterpret_cast<const uint8_t *>(cB.kvp[w][slot] + row * (HD * bits / 8));
        float v[8];
        kv_row_load(rp, bits, __half2float(cB.kvs[w][slot][row * (HD / 32) + (lane >> 2)]), v);
        __syncwarp();
        const size_t rr = (size_t) g * cB.kvrc + p % cB.kvrc;
        kv_store_q8(cB.kvrp[w][slot] + rr * HD, cB.kvrs[w][slot] + rr * (HD / 32), v);
    }
}
