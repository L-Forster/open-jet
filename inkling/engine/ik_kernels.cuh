// Per-token kernels of the Inkling 2 Mini decode step; see ik_kernels.cu.
#pragma once

#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace ik {

#ifdef __CUDACC__
// q8_0 KV cache (same split int8 + fp16-per-32 layout and round-to-even quantization as megakernel/qwen38_mk.cu's
// kv_store_q8): for D = 128, 32 lanes hold 4 consecutive values each (lane*4 .. lane*4+3), so 8 consecutive lanes
// form one 32-element block (blk = lane >> 3).
__device__ __forceinline__ void kv_store_q8_128(int8_t * q, __half * sc, const float (&v)[4]) {
    const int lane = threadIdx.x & 31;
    float am = fmaxf(fmaxf(fabsf(v[0]), fabsf(v[1])), fmaxf(fabsf(v[2]), fabsf(v[3])));
    am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, 1));
    am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, 2));
    am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, 4));
    const float id = am > 0.f ? 127.0f / am : 0.f;
    unsigned x = 0;
#pragma unroll
    for (int k = 0; k < 4; ++k) x |= ((unsigned) __float2int_rn(v[k] * id) & 0xffu) << (8 * k);
    *reinterpret_cast<unsigned *>(q + lane * 4) = x;
    if ((lane & 7) == 0) sc[lane >> 3] = __float2half_rn(am / 127.0f);
}

__device__ __forceinline__ float4 kv_load_q8_raw(const int8_t * q, int lane) {
    const unsigned x = __ldg(reinterpret_cast<const unsigned *>(q + lane * 4));
    return make_float4(float(int8_t(x & 0xffu)), float(int8_t((x >> 8) & 0xffu)),
                       float(int8_t((x >> 16) & 0xffu)), float(int8_t((x >> 24) & 0xffu)));
}

__device__ __forceinline__ float kv_load_q8_scale(const __half * sc, int lane) {
    return __half2float(__ldg(sc + (lane >> 3)));
}
#endif

constexpr int MAX_STATES = 8;  // depth-mixing states: earlier block outputs plus the current stream

struct MixStates {
    const float * p[MAX_STATES];
    int n;
};

// up to 4 controllers (q, k, v, ffn) in one launch
struct ControlArgs {
    const float * cur[4];
    const float * msg[4];
    const float * down[4];    // [rank][2n]
    const float * gain[4];    // [n][rank]
    const float * forget[4];  // [n][rank]
    float * out[4];
    int count;  // set by control()
};

struct NormArgs {
    const float * x[3];
    const float * w[3];
    float * out[3];
};

struct AttnPrepArgs {
    const float * q;      // [n_head * d_head] raw projection
    const float * k;      // [n_head_kv * d_head] raw projection
    const float * v;
    const float * r;      // [n_head * n_rel]
    const float * kdyn;   // [n_head_kv * 4] input-dependent taps (full-attention layers) or null
    const float * vdyn;
    const float * k_conv; // [kv][4]
    const float * v_conv;
    float * hist_k;       // [3][kv]
    float * hist_v;
    const float * q_clip;   // [n_head][d_head]
    const float * rel_clip; // [n_head][n_rel]
    float * q_out;
    float * r_out;
    float * k_cache;      // [slots][kv] fp32 (when kv_q8 == 0)
    float * v_cache;
    int8_t * k_q;         // [slots][kv] int8 (when kv_q8 != 0)
    int8_t * v_q;
    __half * k_s;         // [slots][kv / 32] fp16 scales
    __half * v_s;
    const int * pos;
    int window;           // sliding-window layers: the visible positions; 0 = full cache
    int global;           // full-attention layer: tau scaling
    int n_head, n_head_kv, d_head, n_rel;
    int ring;             // sliding-window layers: rows of the ring cache (window + the largest prefill chunk); 0 = full
    int kv_q8;            // 1: q8_0 KV cache, 0: fp32 KV cache
    float qk_clip, rel_clip_thr, eps, log_alpha, log_floor;
};

struct AttnArgs {
    const float * q;        // [n_head][128], clipped and tau-scaled
    const float * r;        // [n_head][16]
    const float * sinks;    // [n_head]
    const float * gate;     // [n_head * 128] pre-sigmoid
    const float * rel_bias; // [extent][16]
    const float * k_cache;
    const float * v_cache;
    const int8_t * k_q;
    const int8_t * v_q;
    const __half * k_s;
    const __half * v_s;
    float * out;
    const int * pos;
    int window, extent, n_head, n_head_kv;
    int ring;  // rows of a sliding-window layer's ring cache, 0 = full cache
    int kv_q8; // 1: q8_0 KV cache, 0: fp32 KV cache
    float * scratch;  // decode: [n_head][16][130] floats for the split-K partials (ik::attention_scratch_floats)
};

constexpr int attention_scratch_floats(int n_head) { return n_head * 16 * 130; }

// one layer application's exchange with the CPU, in mapped host memory (device-visible pointers)
struct HostSlot {
    volatile int * ready;  // = pos + 1 once ids, wts, x are written
    volatile int * done;   // = pos + 1 once the CPU has written y
    int32_t * ids;
    float * wts;
    float * x;
    volatile float * y;
};

// the copy path: which of a layer application's selected experts (indices into ids) were copied to the scratch slots
struct CopySel {
    int n;
    int k[8];
};

void embed(const uint8_t * row, const float * w, float * out, int n, float eps, cudaStream_t st, int emb_type = 8);
void mix(const MixStates & s, const float * query, int n_q, float * out, int n, cudaStream_t st);
void mix_norm(const MixStates & s, const float * query, int n_q, const float * const * norm_w, float eps, float * out, int n, cudaStream_t st);
void control(const ControlArgs & a, int count, int n, int rank, float * scratch /* [count][64] */, cudaStream_t st);
void rmsnorm(const NormArgs & a, int count, int n, float eps, cudaStream_t st);
void matvec_f32(const float * w, const float * x, float * y, int rows, int n, cudaStream_t st);
void matvec_bf16(const __nv_bfloat16 * w, const float * x, float * y, int rows, int n, cudaStream_t st);
void attn_prep(const AttnPrepArgs & a, cudaStream_t st);
void attention(const AttnArgs & a, cudaStream_t st);
void conv_residual(const float * x, float * hist, const float * w, const float * add, float * out, int n, cudaStream_t st);
void router(const float * w, const float * bias, const float * x, int32_t * ids, float * wts, int n_expert, int n, int k, float scale,
            float * scratch /* [n_expert] */, cudaStream_t st);
void softcap(const float * x, float * y, int n, float cap, cudaStream_t st);
// fused softcap + optional penalties + 2-stage exact argmax (writes winning id + max logit to mapped host pointers)
void softcap_pen_argmax(const float * raw, float * logits, const int32_t * pen_counts, const float * pen_params,
                        int n, float cap, float * bval /* [64] */, int32_t * bidx /* [64] */,
                        int32_t * out_idx, float * out_val, cudaStream_t st);
// collect every (id, logit) with logit >= (*max_val - margin) up to cap; *cnt_out holds total matches (> cap => overflow)
void filter_logits(const float * logits, int n, const float * max_val, float margin,
                   int32_t * out_ids, float * out_vals, int * cnt_out, int cap, cudaStream_t st);
void pen_inc(int32_t * pen_counts, int tok, cudaStream_t st);
void add_pos(int * pos, int d, cudaStream_t st);
void publish(const int32_t * ids, const float * wts, const float * x, const HostSlot & h, const int * pos, int k, int n, cudaStream_t st);
void zc_select(const void * const * table, const void * const * zc_gu, const void * const * zc_dn, const int32_t * ids, int k,
               float share, const void ** out_gu, const void ** out_dn, cudaStream_t st);
void add(float * y, const float * x, int n, cudaStream_t st);
// copy path, in the decode graph: when any selected expert is missing from the VRAM cache, wait for *flag == *pos + 1
// (mapped host memory, set by the copy thread once the copies have landed),
// then out_gu / out_dn [ids[k]] = the scratch slot of each copied expert (gate/up at base + i * slot, down at
// + gu_area), null for the others
void copy_wait(const void * const * table, const int32_t * ids, int k, const int * pos, const volatile int * flag,
               const CopySel * sel, uint8_t * base, size_t slot, size_t gu_area, const void ** out_gu, const void ** out_dn,
               cudaStream_t st);
void wait_add(const HostSlot & h, const void * const * table, const int32_t * ids, int k, const int * pos, float * y, int n, cudaStream_t st);

}  // namespace ik

// expert_kernels.cu (C API)
extern "C" {
size_t ik_matrix_bytes(int type, int rows, int n);
int ik_repack(int type, const void * src, void * dst, int rows, int n, cudaStream_t st);
int ik_matvec_q8_0(const void * w, const float * x, float * y, int rows, int n, cudaStream_t st);
// fused shared-input repacked Q8_0 projections (up to 5 outputs sharing one x of length n = 2304)
int ik_matvec_q8_0_multi(int count, const void * const * ws, const float * x, float * const * ys, const int * rows, int n, cudaStream_t st);
int ik_experts_gate_up(int type, const void * const * experts, const float * scale, const int32_t * ids, int n_sel,
                       const float * x, float * h, int n_embd, int n_ff, float beta, float lb, cudaStream_t st);
int ik_experts_down(int type, const void * const * experts, const float * scale, const int32_t * ids, const float * wts, int n_sel,
                    const float * h, float * partial, float * y, int n_embd, int n_ff, cudaStream_t st);
// gguf_kernels.cu: other ggml types (ggml type ids), weights in their GGUF blocks
int ik_gguf_type_supported(int type);
size_t ik_gguf_row_bytes(int type, int n);
int ik_matvec_gguf(int type, const void * w, const float * x, float * y, int rows, int n, cudaStream_t st);
int ik_experts_gate_up_gguf(int type, const void * const * experts, const float * scale, const int32_t * ids, int n_sel,
                            const float * x, float * h, int n_embd, int n_ff, float beta, float lb, cudaStream_t st);
int ik_experts_down_gguf(int type, const void * const * experts, const float * scale, const int32_t * ids, const float * wts,
                         int n_sel, const float * h, float * partial, float * y, int n_embd, int n_ff, cudaStream_t st);
}
