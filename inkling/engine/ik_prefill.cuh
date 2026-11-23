// Batched (prefill) kernels: T tokens per launch, activations [T][dim]; see ik_prefill.cu.
#pragma once

#include "ik_kernels.cuh"

namespace ik {
namespace pf {

void embed(const uint8_t * rows, const float * w, float * out, int n, float eps, int T, cudaStream_t st, int emb_type = 8);
// states / out: [T][...]; MixStates holds the base pointers
void mix(const MixStates & s, const float * query, int n_q, float * out, int n, int T, cudaStream_t st);
// cur / msg / out: base pointers with their per-token strides; scratch [T][count][64]
void control(const ControlArgs & a, int count, size_t cur_stride, size_t msg_stride, size_t out_stride, int n, int rank, int T,
             float * scratch, cudaStream_t st);
void rmsnorm(const NormArgs & a, int count, size_t x_stride, size_t out_stride, int n, float eps, int T, cudaStream_t st);
int matvec_q8r(const void * w, const float * X, float * Y, int rows, int n, int T, cudaStream_t st);
int matvec_q8r_multi(int count, const void * const * ws, const float * X, float * const * ys, const int * rows, int n, int T,
                     cudaStream_t st);
void matvec_f32(const float * w, const float * X, float * Y, int rows, int n, int T, cudaStream_t st);
void conv(const float * X, float * hist, const float * w, const float * dyn, int heads, float * out, int C, int T, float norm, cudaStream_t st,
          float * hist_snaps = nullptr);
// a: as for decode, with q / r / q_out / r_out [T][...]; k_in / v_in: the convolved K and V [T][kv]
void attn_prep2(const AttnPrepArgs & a, const float * k_in, const float * v_in, int pos0, int T, cudaStream_t st,
                const int * d_pos = nullptr);
// a: q / r / sinks / gate / out [T][...]
void attention(const AttnArgs & a, int pos0, int T, cudaStream_t st, const int * d_pos = nullptr);
void conv_res(const float * X, float * hist, const float * w, const float * add, float * out, int n, int T, cudaStream_t st,
              float * hist_snaps = nullptr);
void router(const float * w, const float * bias, const float * X, int32_t * ids, float * wts, float * probs, int n_expert, int n, int k,
            float scale, int T, cudaStream_t st);
void group(const int32_t * ids, int T, int k, int n_expert, int * offsets, int * list, cudaStream_t st,
           int * active = nullptr, int * n_active = nullptr);
int experts_fast(int type, const void * const * gu, const void * const * dn, const float * gu_scale, const float * dn_scale,
                 const int * offsets, const int * list, int k, const float * wts, const float * X, float * H, float * partial,
                 int n_embd, int n_ff, int n_expert, float beta, float lb, cudaStream_t st,
                 const int * active = nullptr, const int * n_active = nullptr, int max_active = 0);
void sum_entries(const float * partial, float * Y, int n_embd, int k, int T, cudaStream_t st);

}  // namespace pf
}  // namespace ik

extern "C" {
int ik_pf_matvec_gguf(int type, const void * w, const float * X, float * Y, int rows, int n, int T, cudaStream_t st);
int ik_pf_experts_gguf(int gu_type, int dn_type, const void * const * gu, const void * const * dn, const float * gu_scale,
                       const float * dn_scale, const int * offsets, const int * list, int k, const float * wts, const float * X,
                       float * H, float * partial, int n_embd, int n_ff, int n_expert, float beta, float lb, cudaStream_t st);
// ik_prefill_tc.cu: fp16 tensor-core GEMMs reusing megakernel/prefill_gemm.cuh
void ik_pf_f32_to_hperm(const float * X, void * Xh, int T, int T_pad, int K, cudaStream_t st);
void ik_pf_gather_hperm(const float * X, const int * list, int k, int n_entries, int K, void * Xg, cudaStream_t st);
int ik_pf_gemm_q8r(const void * w, const void * Xh, float * Y, int rows, int K, int T, cudaStream_t st);
int ik_pf_gemm_gguf(int gtype, const void * w, const void * Xh, float * Y, int rows, int K, int T, cudaStream_t st);
int ik_pf_experts_tc(int type, const void * const * gu, const void * const * dn, const float * gu_scale, const float * dn_scale,
                     const int * offsets, const int * list, const float * wts, const void * Xg, void * Hg, float * partial,
                     int n_embd, int n_ff, int n_expert, float beta, float lb, cudaStream_t st);
}
