// K/V cache addressing shared by the engines (qwen38_mk.cu, qwen38_mixed.cu) and prefill.cuh: included after the
// engine's `__constant__ Bufs cB`. The mixed engine keeps the segment and page-table fields zero (one contiguous cache).
#pragma once

constexpr int KV_PG = 256;
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
