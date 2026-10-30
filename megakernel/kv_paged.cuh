// K/V cache addressing shared by the engines (qwen38_mk.cu, qwen38_mixed.cu) and prefill.cuh: included after the
// engine's `__constant__ Bufs cB`. The mixed engine keeps the segment and page-table fields zero (one contiguous cache).
#pragma once

constexpr int KV_PG = 256;
#ifdef OPENJET_MIXED_QUANTS
constexpr bool KV_SLOTS = false; // (no parallel slots: the paged MK_KV branches compile out)
#else
constexpr bool KV_SLOTS = true;
#endif
// row of own-cache position i (attention slot, kv head g): paged (ptab) or contiguous of stride ctx. (The tables
// change only between launches: read-only loads, through L1 — plain loads skip it under -dlcm=cg.)
__device__ __forceinline__ size_t kv_own_row(const int * ptab, int ctx, int slot, int g, int i) {
    return ptab ? (((size_t) __ldg(ptab + i / KV_PG) * N_ATT_SLOTS + slot) * NKV + g) * KV_PG + i % KV_PG : ((size_t) slot * NKV + g) * ctx + i;
}

// K/V row of (attention slot, kv head, position) in the segment (pos < segn) or the slot's own cache
__device__ __forceinline__ size_t kv_row(int slot, int g, int pos, bool & seg) {
    seg = pos < cB.segn;
    return seg ? kv_own_row(cB.segptab, cB.segctx, slot, g, pos) : kv_own_row(cB.ptab, cB.ctx, slot, g, pos - cB.segn);
}
__device__ __forceinline__ const void * kv_base(bool seg, int which) { return seg ? (which ? cB.vseg : cB.kseg) : (which ? cB.vc : cB.kc); }
__device__ __forceinline__ const __half * kv_sbase(bool seg, int which) { return seg ? (which ? cB.vsseg : cB.ksseg) : (which ? cB.vsc : cB.ksc); }

// MK_KV (kv_lowbit.cuh) in the paged pool of the parallel slots: a page keeps the scales' q8 layout (kv_own_row) and
// packs the values, the slots one after the other, each [kv head][KV_PG][HD * bits / 8] (all-q8: kv_own_row * HD).
// MK_KV_WIN: every slot (sequence) has its own q8 rings, one block of rstride bytes per slot in `ring`, laid out per
// side as the single-stream rings (values [kv head][kvrc][HD], scales [kv head][kvrc][HD / 32]). Set by ms_alloc.
struct KvPaged {
    long long pgb[2];        // value bytes of one page (all slots), K / V
    int pgo[2][N_ATT_SLOTS]; // byte offset of a slot's values in a page
    int8_t * ring;
    long long rstride;
    int ro[2][N_ATT_SLOTS], rso[2][N_ATT_SLOTS]; // a side's values / scales in a ring block (-1: no ring)
};
__constant__ KvPaged cKP;

// value byte offset of own-cache position i (attention slot, kv head g) in a paged pool of packed rows
__device__ __forceinline__ size_t kv_pg_voff(int w, int slot, int g, int i, const int * ptab) {
    return (size_t) __ldg(ptab + i / KV_PG) * cKP.pgb[w] + cKP.pgo[w][slot] + ((size_t) g * KV_PG + i % KV_PG) * (HD * cB.kvb[w][slot] / 8);
}
// MK_KV, paged caches (parallel slots: cB.ptab set): the K (w 0) / V (w 1) row of position pos (attention slot, kv head
// g) in its slot's format, in the shared segment below cB.segn (read only) or the own cache (or the segment being
// built): its HD * bits / 8 value bytes; sc <- its HD / 32 scales
__device__ __forceinline__ int8_t * kv_vptr_paged(int w, int slot, int g, int pos, __half *& sc) {
    const bool seg = pos < cB.segn;
    const int * pt = seg ? cB.segptab : cB.ptab;
    const int i = seg ? pos : pos - cB.segn;
    sc = (seg ? (w ? cB.vsseg : cB.ksseg) : (w ? cB.vsc : cB.ksc)) + kv_own_row(pt, 0, slot, g, i) * (HD / 32);
    return static_cast<int8_t *>(seg ? (w ? cB.vseg : cB.kseg) : (w ? cB.vc : cB.kc)) + kv_pg_voff(w, slot, g, i, pt);
}
// MK_KV, read side: the row in the paged caches, or in the engine's own contiguous one (kvp / kvs)
__device__ __forceinline__ const int8_t * kv_vptr(int w, int slot, int g, int pos, const __half *& sc) {
    if (cB.ptab) {
        __half * s;
        const int8_t * q = kv_vptr_paged(w, slot, g, pos, s);
        sc = s;
        return q;
    }
    const size_t row = (size_t) g * cB.ctx + pos;
    sc = cB.kvs[w][slot] + row * (HD / 32);
    return cB.kvp[w][slot] + row * (HD * cB.kvb[w][slot] / 8);
}
// MK_KV, upstream-style readers (a row at a time): the K (w 0) / V (w 1) row of position pos for a launch whose window
// starts at ws: the q8 ring row (MK_KV_WIN) at pos >= ws, else kv_vptr; returns its bits (q / sc: values / scales)
__device__ __forceinline__ int kv_src_row(int w, int slot, int g, int pos, int ws, const int8_t *& q, const __half *& sc) {
    if (cB.kvrp[w][slot] && pos >= ws) {
        const size_t rr = (size_t) g * cB.kvrc + pos % cB.kvrc;
        q = cB.kvrp[w][slot] + rr * HD;
        sc = cB.kvrs[w][slot] + rr * (HD / 32);
        return 8;
    }
    q = kv_vptr(w, slot, g, pos, sc);
    return cB.kvb[w][slot];
}
// (the host's window start: kv_ws_of, kv_lowbit.cuh)
