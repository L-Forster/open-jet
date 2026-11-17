// mk_launch.h: the device TU's (qwen38_mk.cu) kernel launchers, kernel handles and __constant__ / __device__ symbol copies
// for the host files (engine.cu and the Makefile's other HOST_SRC files). Only types with external linkage cross it:
// engine structs go as const void *.
#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstddef>
#include <cstdint>

#define MK_GLU_MIX_PAIRS(X)                                                                                                \
    X(pf::TQ4K, 1, pf::TQ5K, 1) X(pf::TIQ4, 1, pf::TQ5K, 1) X(pf::TIQ4, 1, pf::TQ4K, 1) X(pf::TQ4K, 1, pf::TIQ4, 1)      \
    X(pf::TQ6K, 1, pf::TQ6K, 1) X(pf::TIQ4, 1, pf::TQ3K, 0) X(pf::TQ6K, 1, pf::TQ5K, 1) X(pf::TQ3K, 0, pf::TIQ4, 1)      \
    X(pf::TIQ4NL, 1, pf::TIQ4, 1) X(pf::TIQ4NL, 1, pf::TQ5K, 1) X(pf::TQ5K, 1, pf::TIQ4NL, 1)

namespace mkl {

// mk_abi_sig() as the device TU computes it (MKEngine refuses an object built with other settings)
unsigned long long abi_sig();
extern const void * const k_mk_cycle;

cudaError_t set_cW(const void * src, size_t count, size_t offset = 0);
cudaError_t set_cW_async(const void * src, size_t count, size_t offset, cudaMemcpyKind kind, cudaStream_t st);
cudaError_t get_cW(void * dst, size_t count, size_t offset = 0);
cudaError_t set_cBs(const void * src, size_t count, size_t offset = 0);
cudaError_t set_cBs_async(const void * src, size_t count, size_t offset, cudaMemcpyKind kind, cudaStream_t st);
cudaError_t set_cView(const void * src, size_t count, size_t offset = 0);
#if MK_L2_TUNE
cudaError_t set_cL2(const void * src, size_t count, size_t offset = 0);
#endif
cudaError_t wv_deal_init(int nsm); // (MK_FFNA builds: the wv phase tops' per-block dealing table, checked; else nothing)
cudaError_t set_cP(const void * src, size_t count, size_t offset = 0);
cudaError_t set_g_iq3s_grid(const void * src, size_t count, size_t offset = 0);
cudaError_t set_chk_layer_async(const void * src, size_t count, size_t offset, cudaMemcpyKind kind, cudaStream_t st);
cudaError_t set_chk_site_async(const void * src, size_t count, size_t offset, cudaMemcpyKind kind, cudaStream_t st);
cudaError_t set_chk_cnt(const void * src, size_t count, size_t offset = 0);
cudaError_t get_chk_cnt(void * dst, size_t count, size_t offset = 0);

void k_ring_refill(int p0, int p1);
void k_kv_demote(int p0, int p1);
void pk_fault();
void st_warm(dim3 grid, unsigned long long ns);
void kw_warm(dim3 grid, dim3 block, const unsigned * stop, unsigned long long ns, int spin, cudaStream_t st = 0);
// Enqueues on st the device re-layout of the staged matrix sw (values, 8-byte aligned) / sm (meta) into out / out_meta, as
// wv_relayout writes it (after wv_dev_ok said yes; FM types: their WvProbe's gather table dtab of W words, G, seg).
// Returns the launch error.
cudaError_t wv_dev_launch(int type, int N, int K, int rb, const uint8_t * sw, const uint8_t * sm, size_t meta_n, uint8_t * out,
                          uint8_t * out_meta, cudaStream_t st, const uint32_t * dtab, int W, int G, int seg);

template <int ET> void pk_embed(int T);
template <int ET> void pk_embed_img(int T);
void pk_norm_quant(dim3 grid, const float * src, const float * w);
void pk_norm_quant1(dim3 grid, const float * src, const float * w);
void pk_ab(const float * Wab, const int8_t * abq, const __half * abd, const float * fn, int T, cudaStream_t st = 0);
void pk_ab2(const float * Wab, const int8_t * abq, const __half * abd, const float * fn, int T, cudaStream_t st = 0);
void pk_dn_prep(int T, int n0, int dl, const float * convw, const float * dt, const float * A);
template <int NT> void pk_dn_prep_t(int T, int n0, int dl, const float * convw, const float * dt, const float * A);
void pk_conv_ring(int T, int n0, int dl, bool old = false);
void pk_ck_conv(int n0, int P, int dl, float * dst);
void pk_ck_hist(dim3 grid, int lo, int n0, int from_ring, float * dst);
void pk_dn_scan(int T, int t0, const float * st_in, float * st_out);
void pk_dn_tri(int T, __half * tri);
void pk_dn_scan4(int T, int t0, const float * st_in, float * st_out, const __half * tri);
void pk_dn_out(dim3 grid, const float * snorm);
void pk_attn_prep(dim3 grid, int n0, int slot, const float * qn, const float * kn, int do_q, int r0 = 0);
void pk_attn_out(dim3 grid, int slot);
void pk_final_h(int T, int n0);
template <bool IMG, int ET> void pk_mtp_in(dim3 grid);
void pk_argmax(dim3 grid, int t0);
void pk_glu(dim3 grid, const float * g, const float * u, __half * out, int ld);

template <bool Q8> void pk_attn(int T, int n0, int slot);
template <bool Q8, bool PV16> void pk_attn2(dim3 grid, int T, int n0, int slot, int nfull, int S, float * part, int r0 = 0);
void pk_attn2_comb(dim3 grid, int T, int nfull, int S, const float * part);
extern int pk_attn_old; // MK_PFX bit 64
template <bool QROW, bool KLOW, bool EXA = false, int KB = 0, int VB = 0>
void pk_attn8(dim3 grid, int T, int n0, int slot, int nfull, int S, float * part, int r0 = 0, int segt = 0, int exh = 0,
              float * part2 = nullptr, int exa = 0);
void pk_attn8_combx(dim3 grid, int T, int n0, int nfull, int S, int segt, int exh, int r0, const float * part, const float * part2,
                    int exa);
void pk_attn_attr();
void pk_attn8_exa_attr();
void pk_attn2_occupancy(int * nb);
void pk_attn8_occupancy(int * nb);

template <int TYPE, bool GLU, int LAY>
void pf_mmh(int nb, int nrt, const void * W, const __half * X, int ntok, const void * O, const uint8_t * W2, cudaStream_t st = 0);
template <int TYPE, bool GLU, int LAY> void pf_mmh_attr();
template <int TG, int LG, int TU, int LU>
void pf_mmh_mix(int nb, int nrt, const void * Wg, const void * Wu, const __half * X, int ntok, const void * O);
template <int TG, int LG, int TU, int LU> void pf_mmh_mix_attr();

} // namespace mkl
