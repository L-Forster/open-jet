// Persistent cooperative "megakernel" for Qwen3.8-27B (llama.cpp arch "qwen35", shared with Qwen3.5/3.6;
// 64 layers: 48 Gated-DeltaNet + 16 gated full-attention, plus one MTP block), tuned for
// RTX 3090 (sm_86, 82 SMs).
//
// One kernel launch executes an entire speculative decoding cycle:
//   main forward over T = 1 + n_draft tokens  ->  lm_head + greedy verify on device
//   -> MTP catch-up rows  ->  n_draft MTP draft steps.
// Phases inside the launch are separated by a software grid barrier.  Weights stay in their
// GGUF k-quant encodings (Q6_K / Q8_0 are re-laid-out for aligned loads, values unchanged)
// and are streamed once per cycle with int8 dp4a dot products against q8 activations.
//
// Gated-DeltaNet rollback after rejected drafts is done by lazy replay: the committed state
// only ever includes surely-accepted tokens; accepted drafts are replayed from a small ring
// of (k, v, g, beta) at the start of the next cycle, so no state snapshots are written.

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <map>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "ggml.h"
#include "gguf.h"
#include "qwen38_mk.h"

#define CK(x)                                                                                   \
    do {                                                                                        \
        cudaError_t e_ = (x);                                                                   \
        if (e_ != cudaSuccess) {                                                                \
            fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #x, __FILE__, __LINE__,             \
                    cudaGetErrorString(e_));                                                    \
            exit(1);                                                                            \
        }                                                                                       \
    } while (0)

namespace {

constexpr int D = 5120, FF = 17408, VOCAB = 248320, NL = 64, NLA = 65;
constexpr int NH = 24, NKV = 4, HD = 256;
constexpr int NKH = 16, NVH = 48, HK = 128;
constexpr int QKV = 10240, VD = 6144, QG = 12288, KVD = 1024;
constexpr int MAXT = 5;
constexpr int NT = 512, NW = NT / 32;
constexpr float EPS = 1e-6f;
constexpr float ROPE_BASE = 10000000.0f;
constexpr int CONV_RING = 8, KV_RING = 16, H_RING = 16;
constexpr int KVGB = NKH * HK + NVH * HK + 2 * NVH; // k | v | g | beta
constexpr int KVGB_V = NKH * HK, KVGB_G = KVGB_V + NVH * HK, KVGB_B = KVGB_G + NVH;
constexpr int N_ATT_SLOTS = 17; // 16 trunk attention layers + MTP block
constexpr int MAX_CH = 32;      // attention split-K chunks per kv head
constexpr int DV_EXTRA = 4096;  // draft-vocabulary rows reserved for tokens of the current sequence
constexpr int AB_SPLIT = 4;     // K-chunks per delta-net alpha/beta row
constexpr int U_D = D / 32, U_FF = FF / 32, U_VD = VD / 32, U_EH = 2 * D / 32;
constexpr int UMAX = U_FF;

constexpr int AR = 6 * MAXT;   // attention query rows per kv head (6 q heads x tokens)
constexpr int SM_XU = 4 * UMAX; // activation units (x tokens) shared memory holds; T = 5 ffn_down fills in two halves
constexpr int SM_XA = 0;
constexpr int SM_XB = SM_XA + SM_XU * 16;
constexpr int SM_XM = SM_XB + SM_XU * 16;
constexpr int SM_MISC = SM_XM + SM_XU * 8;
constexpr int ATP = 32; // attention: positions per K/V tile (two tiles in flight)
constexpr int KVQ_ROW = 256 + 16; // staged q8 K or V row: 256 int8 values + 8 fp16 scales
// K/V tile region: fp16 cache: 2 stages of fp16 K/V tiles; q8 cache: one fp16 K/V tile + 2 stages of q8 tiles
constexpr int SM_ATTN_KV = (4 * ATP * (256 + 8) * 2 > 2 * ATP * (256 + 8) * 2 + 4 * ATP * KVQ_ROW)
                               ? 4 * ATP * (256 + 8) * 2 : 2 * ATP * (256 + 8) * 2 + 4 * ATP * KVQ_ROW;
constexpr int SM_ATTN = 32 * (256 + 8) * 2 + SM_ATTN_KV + 32 * ATP * 4 + 32 * (ATP + 8) * 2 + 96 * 4; // Q, K/V, S, P, stats
constexpr int SM_TOTAL = (SM_MISC + 2048 > SM_ATTN ? SM_MISC + 2048 : SM_ATTN);
static_assert(SM_TOTAL <= 99 * 1024, "shared memory budget");

enum { T_F32 = 0, T_Q8_0 = 8, T_Q4K = 12, T_Q5K = 13, T_Q6K = 14 };
enum { MODE_PREFILL = 0, MODE_DECODE = 1, MODE_BENCH = 2 };

struct Mat {
    const uint8_t * w;
    const __half * d; // per-row block scales for repacked Q6_K / Q8_0
    int type, K, N, rb, nd;
};

struct LayerW {
    int attn;
    const float *attn_norm, *post_norm;
    // gated delta net
    Mat qkv, gate, sout;
    const float *ab, *conv, *dt, *A, *snorm;
    // gated attention
    Mat q, k, v, o;
    const float *qn, *kn;
    // ffn
    Mat fg, fu, fd;
};

struct ModelW {
    LayerW L[NLA];
    Mat eh, head;
    Mat head_d;            // MTP draft head: subset of lm_head rows (== head when no subset is used)
    const int * draft_ids; // row -> token id for head_d (nullptr: identity)
    const float *enorm, *hnorm, *shnorm, *onorm;
    const uint8_t * embd;
};

// Quantised activation vector(s) in the "unit" layout consumed by the GEMV:
// unit u = (super-block b, 64-group p, half h) covers elements A=[256b+64p+16h,+16) and
// B=[256b+64p+32+16h,+16).  xa/xb hold the int8 values, d the per-32 scales of the natural
// blocks the A/B halves live in, qs the int sums of each 16-chunk, ssq per natural block x^2.
struct QBuf {
    int8_t *xa, *xb;
    uint2 * meta; // per unit: half2 (scale of A / B chunk), short2 (int sums of A / B chunk)
    float * ssq;
    int U;
};

struct Ctl {
    int n;         // position of tok[0]
    int tok[8];    // tokens fed to the main model this cycle
    int dn_c;      // delta-net committed position (state covers positions < dn_c)
    int mtp_valid; // MTP kv valid for positions < mtp_valid
    int n_out, out[8];
    int n_acc;
    int mtp_R, mtp_logits, mtp_P;
    int mtp_pos[8], mtp_tok[8];
    int tgt[8];
    int pad[8];
};

struct Bufs {
    float *x, *xm;
    QBuf qn, qo, qa, qe;
    float *z, *ab, *qbuf, *kbuf, *vbuf;
    float *conv, *kvgb, *dstate;
    void *kc, *vc;     // KV cache [slot][kv head][ctx][HD]: fp16, or int8 (q8_0) when kvq8 ...
    __half *ksc, *vsc; // ... with one fp16 scale per 32 values: [slot][kv head][ctx][HD / 32]
    int kvq8;
    float * apart;
    float2 * aml;
    float *h_hist, *hm;
    const float2 * rope; // [ctx][32] (cos, sin) of pos * base^(-2i/64)
    float2 * blk_best; // per block: best (logit, id) per row, or the TOPK_MAX best when sampling
    float2 * cand;     // [MAXT][TOPK_MAX] best (logit, id) of the main lm_head rows, descending (sampling)
    float * logits;    // [MAXT][VOCAB] main lm_head logits of a scoring launch (Launch::score == 2)
    unsigned * bar;
    unsigned * cnt;
    Ctl * ctl;
    unsigned long long * prof; // optional per-barrier timestamps (block 0)
    int ctx;
};

__constant__ ModelW cW;
__constant__ Bufs cB;

struct Launch {
    int mode, T, last, ndraft;
    int score; // prefill only: greedy argmax for every row (teacher-forced evaluation), no MTP; 2: also their logits
    // sampling (temp > 0; greedy otherwise), applied in llama.cpp's order: top_k, top_p, min_p, temperature
    float temp, top_p, min_p;
    int top_k; // <= TOPK_MAX
    unsigned long long seed;
};
constexpr int TOPK_MAX = 32; // candidates kept per main lm_head row when sampling

// ------------------------------------------------------------------------------------------
// small device helpers
// ------------------------------------------------------------------------------------------

// Weights are streamed once per cycle: bypass L1 and mark them evict-first in L2 so that data
// prefetched with evict-last (recurrent state, KV cache) survives the stream.
__device__ __forceinline__ unsigned long long l2_evict_first() {
    unsigned long long pol;
    asm("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(pol));
    return pol;
}
__device__ __forceinline__ int4 ldw4(const void * p) {
    int4 r;
    asm("ld.global.nc.L1::no_allocate.L2::cache_hint.v4.s32 {%0,%1,%2,%3}, [%4], %5;"
        : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w)
        : "l"(p), "l"(l2_evict_first()));
    return r;
}
__device__ __forceinline__ int2 ldw2(const void * p) {
    int2 r;
    asm("ld.global.nc.L1::no_allocate.L2::cache_hint.v2.s32 {%0,%1}, [%2], %3;" : "=r"(r.x), "=r"(r.y) : "l"(p), "l"(l2_evict_first()));
    return r;
}
__device__ __forceinline__ int ldw1(const void * p) {
    int r;
    asm("ld.global.nc.L1::no_allocate.L2::cache_hint.s32 %0, [%1], %2;" : "=r"(r) : "l"(p), "l"(l2_evict_first()));
    return r;
}
__device__ __forceinline__ void pf_l2(const void * p) { asm volatile("prefetch.global.L2::evict_last [%0];" ::"l"(p)); }
// prefetch `bytes` starting at p (128-byte lines) with the threads of the calling block
__device__ __forceinline__ void pf_block(const void * p, int bytes) {
    for (int off = threadIdx.x * 128; off < bytes; off += NT * 128) pf_l2(reinterpret_cast<const uint8_t *>(p) + off);
}
__device__ __forceinline__ float4 ldw4f(const float * p) {
    int4 r = ldw4(p);
    return make_float4(__int_as_float(r.x), __int_as_float(r.y), __int_as_float(r.z), __int_as_float(r.w));
}
__device__ __forceinline__ unsigned short ldwu16(const void * p) {
    unsigned short r;
    asm("ld.global.nc.L1::no_allocate.u16 %0, [%1];" : "=h"(r) : "l"(p));
    return r;
}

__device__ __forceinline__ float wsum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
__device__ __forceinline__ float wmax(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}
__device__ __forceinline__ float silu(float x) { return x / (1.0f + expf(-x)); }

enum {
    PH_EMBED, PH_ATTN_PROJ, PH_ATTN_CORE, PH_ATTN_COMB, PH_RESID_O, PH_DN_PROJ, PH_DN_CORE, PH_RESID_SOUT, PH_FFN_GU,
    PH_FFN_DOWN, PH_LMHEAD, PH_MTP_M0, PH_MTP_EH, PH_MTP_ATTN_PROJ, PH_MTP_ATTN_CORE, PH_MTP_ATTN_COMB, PH_MTP_RESID_O,
    PH_MTP_FFN_GU, PH_MTP_FFN_DOWN, PH_MTP_LMHEAD, PH_COUNT
};

__device__ __forceinline__ unsigned long long gtimer() {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

__shared__ unsigned s_bar_k; // index of the last grid barrier passed (profiling only)
extern __shared__ __align__(16) uint8_t smem[]; // dynamic shared memory of the persistent block

// Grid barrier: release-add on arrival, acquire-poll until every block of this generation arrived.
__device__ __forceinline__ void gsync(unsigned & epoch, int tag) {
    __syncthreads();
    if (threadIdx.x == 0) {
        if (cB.prof) {
            const unsigned k = epoch / gridDim.x + 1;
            if (k < 1024) cB.prof[4096 + k * 128 + blockIdx.x] = gtimer();
        }
        epoch += gridDim.x;
        asm volatile("red.release.gpu.global.add.u32 [%0], 1;" ::"l"(cB.bar) : "memory");
        unsigned v;
        do {
            asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(cB.bar) : "memory");
        } while (v < epoch);
        if (cB.prof && blockIdx.x == 0) {
            const unsigned k = epoch / gridDim.x;
            s_bar_k = k;
            if (k < 2048) cB.prof[k] = (gtimer() << 8) | (unsigned long long) tag;
        }
    }
    __syncthreads();
}

// profiling: timestamp the end of a phase's activation fill (block 0)
__device__ __forceinline__ void mark_fill() {
    if (cB.prof && blockIdx.x == 0 && threadIdx.x == 0 && s_bar_k + 1 < 2048) cB.prof[2048 + s_bar_k + 1] = gtimer();
}

// block-wide sum, result broadcast. red must hold >= NW floats.
__device__ __forceinline__ float bsum(float v, float * red) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    v = wsum(v);
    __syncthreads();
    if (lane == 0) red[warp] = v;
    __syncthreads();
    float r = lane < NW ? red[lane] : 0.f;
    r = wsum(r);
    return r;
}

// lane i holds element 32*j+i of row t; quantise the natural 32-block j into the unit layout.
__device__ __forceinline__ void q32_store(const QBuf & q, int t, int j, float v, float ss, bool wss) {
    const int lane = threadIdx.x & 31;
    const float amax = wmax(fabsf(v));
    const float d = amax / 127.0f;
    const int qi = amax == 0.0f ? 0 : (int) roundf(v / d);
    const int b = j >> 3, p = (j >> 1) & 3, part = j & 1, hh = lane >> 4, byte = lane & 15;
    const int gi = t * q.U + 8 * b + 2 * p + hh;
    (part ? q.xb : q.xa)[gi * 16 + byte] = (int8_t) qi;
    int qs = qi;
    qs += __shfl_xor_sync(0xffffffffu, qs, 8);
    qs += __shfl_xor_sync(0xffffffffu, qs, 4);
    qs += __shfl_xor_sync(0xffffffffu, qs, 2);
    qs += __shfl_xor_sync(0xffffffffu, qs, 1);
    if (byte == 0) {
        reinterpret_cast<__half *>(q.meta + gi)[part] = __float2half_rn(d);
        reinterpret_cast<short *>(q.meta + gi)[2 + part] = (short) qs;
    }
    if (wss) {
        const float s = wsum(ss);
        if (lane == 0) q.ssq[t * q.U + j] = s;
    }
}

struct SX {
    int4 * xa;
    int4 * xb;
    uint2 * xm; // per unit: half2 scales of the A / B 16-chunks, short2 int sums of the chunks
    int U;
};

__device__ __forceinline__ SX make_sx(uint8_t * /*sm*/, int U) {
    uint8_t * const sm = smem;
    SX s;
    s.xa = reinterpret_cast<int4 *>(sm + SM_XA);
    s.xb = reinterpret_cast<int4 *>(sm + SM_XB);
    s.xm = reinterpret_cast<uint2 *>(sm + SM_XM);
    s.U = U;
    return s;
}

__device__ __forceinline__ float * misc(uint8_t * /*sm*/) {
    uint8_t * const sm = smem; return reinterpret_cast<float *>(sm + SM_MISC); }

__device__ __forceinline__ void cp_async16(void * smem_dst, const void * gsrc) {
    const unsigned d = (unsigned) __cvta_generic_to_shared(smem_dst);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(d), "l"(gsrc) : "memory");
}

// Load T rows (starting at row t0) of a quantised activation buffer into shared memory with
// async copies.  norm: s_rs[t] receives the RMS-norm scale of row t (from the producer's
// per-block sums of squares); consumers apply it to their dot products, otherwise s_rs = 1.
template <int T>
__device__ void fill_xs(const QBuf & q, int t0, bool norm, const SX & s, float * s_rs) {
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int U = q.U;
    const int8_t * ga = q.xa + (size_t) t0 * U * 16;
    const int8_t * gb = q.xb + (size_t) t0 * U * 16;
    const uint2 * gm = q.meta + (size_t) t0 * U;
    for (int i = tid; i < T * U; i += NT) {
        cp_async16(s.xa + i, ga + (size_t) i * 16);
        cp_async16(s.xb + i, gb + (size_t) i * 16);
    }
    for (int i = tid; i < T * U / 2; i += NT) cp_async16(s.xm + 2 * i, gm + 2 * i);
    asm volatile("cp.async.commit_group;" ::: "memory");
    if (norm) { // normed consumers always read the K = D residual stream (U_D = 160 partial sums)
        if (warp < T) {
            float p[U_D / 32];
#pragma unroll
            for (int k = 0; k < U_D / 32; k++) p[k] = __ldcg(q.ssq + (t0 + warp) * U_D + lane + 32 * k);
            float acc = 0.f;
#pragma unroll
            for (int k = 0; k < U_D / 32; k++) acc += p[k];
            acc = wsum(acc);
            if (lane == 0) s_rs[warp] = rsqrtf(acc / (float) D + EPS);
        }
    } else if (tid < T) {
        s_rs[tid] = 1.0f;
    }
    asm volatile("cp.async.wait_all;" ::: "memory");
    __syncthreads();
    mark_fill();
}

// Async copy of units [u0, u0 + US) of T activation rows (one K-slice) into shared memory.
template <int T, int US>
__device__ void fill_xs_part(const QBuf & q, int u0, const SX & s) {
    const int tid = threadIdx.x, U = q.U;
    __syncthreads(); // the previous slice may still be in use
    for (int i = tid; i < T * US; i += NT) {
        const int t = i / US, u = i - t * US;
        const size_t gi = (size_t) t * U + u0 + u;
        cp_async16(s.xa + i, q.xa + gi * 16);
        cp_async16(s.xb + i, q.xb + gi * 16);
    }
    for (int i = tid; i < T * US / 2; i += NT) {
        const int t = 2 * i / US, u = 2 * i - t * US;
        cp_async16(s.xm + 2 * i, q.meta + (size_t) t * U + u0 + u);
    }
    asm volatile("cp.async.commit_group;\ncp.async.wait_all;" ::: "memory");
    __syncthreads();
}

// ------------------------------------------------------------------------------------------
// quantised GEMV: R rows per warp share every activation load (per-lane partial sums;
// the caller reduces across the warp).  Lane handles units u = lane + 32 i.
// ------------------------------------------------------------------------------------------

constexpr int M4 = 0x0F0F0F0F;

__device__ __forceinline__ int gwarp() { return (threadIdx.x >> 5) * gridDim.x + blockIdx.x; }
__device__ __forceinline__ int nwarps() { return NW * gridDim.x; }

template <int T, int R>
__device__ __forceinline__ void zero_acc(float (&a)[R][T]) {
#pragma unroll
    for (int k = 0; k < R; k++)
#pragma unroll
        for (int t = 0; t < T; t++) a[k][t] = 0.f;
}

template <int T, int R>
__device__ __forceinline__ void wsum_acc(float (&a)[R][T]) {
#pragma unroll
    for (int k = 0; k < R; k++)
#pragma unroll
        for (int t = 0; t < T; t++) a[k][t] = wsum(a[k][t]);
}

template <int T, int R, int UU>
__device__ __forceinline__ void accR(const SX & s, int u, const int (&A)[R][4], const int (&B)[R][4],
                                     const float (&fA)[R], const float (&gA)[R], const float (&fB)[R],
                                     const float (&gB)[R], float (&acc)[R][T]) {
#pragma unroll
    for (int t = 0; t < T; t++) {
        const int ix = t * UU + u;
        const int4 va = s.xa[ix], vb = s.xb[ix];
        const uint2 mm = s.xm[ix];
        const float2 dx = __half22float2(*reinterpret_cast<const __half2 *>(&mm.x));
        const int qq = (int) mm.y;
        const float sqA = dx.x * (float) (short) (qq & 0xffff), sqB = dx.y * (float) (short) (qq >> 16);
#pragma unroll
        for (int k = 0; k < R; k++) {
            int sa = __dp4a(A[k][0], va.x, 0);
            sa = __dp4a(A[k][1], va.y, sa);
            sa = __dp4a(A[k][2], va.z, sa);
            sa = __dp4a(A[k][3], va.w, sa);
            int sb = __dp4a(B[k][0], vb.x, 0);
            sb = __dp4a(B[k][1], vb.y, sb);
            sb = __dp4a(B[k][2], vb.z, sb);
            sb = __dp4a(B[k][3], vb.w, sb);
            acc[k][t] += fA[k] * (dx.x * (float) sa) + fB[k] * (dx.y * (float) sb) - gA[k] * sqA - gB[k] * sqB;
        }
    }
}

__device__ __forceinline__ void q4k_scales(const int4 & h, int p, float & fA, float & gA, float & fB, float & gB) {
    const float2 dm = __half22float2(*reinterpret_cast<const __half2 *>(&h.x));
    const int sh = (p & 1) * 16;
    const unsigned y = (unsigned) h.y >> sh, z = (unsigned) h.z >> sh, w = (unsigned) h.w >> sh;
    unsigned sc0, sc1, m0, m1;
    if (p < 2) {
        sc0 = y & 63;
        sc1 = (y >> 8) & 63;
        m0 = z & 63;
        m1 = (z >> 8) & 63;
    } else {
        sc0 = (w & 0xF) | ((y >> 2) & 0x30);
        sc1 = ((w >> 8) & 0xF) | ((y >> 10) & 0x30);
        m0 = ((w >> 4) & 0xF) | ((z >> 2) & 0x30);
        m1 = ((w >> 12) & 0xF) | ((z >> 10) & 0x30);
    }
    fA = dm.x * (float) sc0;
    fB = dm.x * (float) sc1;
    gA = dm.y * (float) m0;
    gB = dm.y * (float) m1;
}

template <int T, int IT, int R>
__device__ __forceinline__ void rows_q4k(const uint8_t * const (&rp)[R], const SX & s, float (&acc)[R][T]) {
    const int lane = threadIdx.x & 31;
#pragma unroll(IT > 8 ? (T >= 5 ? 1 : 4) : IT)
    for (int i = 0; i < IT; i++) {
        const int u = lane + 32 * i, b = u >> 3, p = (u >> 1) & 3, hh = u & 1;
        int4 h[R], q[R];
#pragma unroll
        for (int k = 0; k < R; k++) {
            const uint8_t * blk = rp[k] + b * 144;
            h[k] = ldw4(blk);
            q[k] = ldw4(blk + 16 + 32 * p + 16 * hh);
        }
        int A[R][4], B[R][4];
        float fA[R], gA[R], fB[R], gB[R];
#pragma unroll
        for (int k = 0; k < R; k++) {
            q4k_scales(h[k], p, fA[k], gA[k], fB[k], gB[k]);
            A[k][0] = q[k].x & M4; A[k][1] = q[k].y & M4; A[k][2] = q[k].z & M4; A[k][3] = q[k].w & M4;
            B[k][0] = (q[k].x >> 4) & M4; B[k][1] = (q[k].y >> 4) & M4;
            B[k][2] = (q[k].z >> 4) & M4; B[k][3] = (q[k].w >> 4) & M4;
        }
        accR<T, R, 32 * IT>(s, u, A, B, fA, gA, fB, gB, acc);
    }
}

template <int T, int IT, int R>
__device__ __forceinline__ void rows_q5k(const uint8_t * const (&rp)[R], const SX & s, float (&acc)[R][T]) {
    const int lane = threadIdx.x & 31;
#pragma unroll(IT > 8 ? (T >= 5 ? 1 : 4) : IT)
    for (int i = 0; i < IT; i++) {
        const int u = lane + 32 * i, b = u >> 3, p = (u >> 1) & 3, hh = u & 1;
        int4 h[R], qh[R], q[R];
#pragma unroll
        for (int k = 0; k < R; k++) {
            const uint8_t * blk = rp[k] + b * 176;
            h[k] = ldw4(blk);
            qh[k] = ldw4(blk + 16 + 16 * hh);
            q[k] = ldw4(blk + 48 + 32 * p + 16 * hh);
        }
        int A[R][4], B[R][4];
        float fA[R], gA[R], fB[R], gB[R];
        const int s0 = 2 * p, s1 = 2 * p + 1;
#pragma unroll
        for (int k = 0; k < R; k++) {
            q4k_scales(h[k], p, fA[k], gA[k], fB[k], gB[k]);
            A[k][0] = (q[k].x & M4) | (((qh[k].x >> s0) & 0x01010101) << 4);
            A[k][1] = (q[k].y & M4) | (((qh[k].y >> s0) & 0x01010101) << 4);
            A[k][2] = (q[k].z & M4) | (((qh[k].z >> s0) & 0x01010101) << 4);
            A[k][3] = (q[k].w & M4) | (((qh[k].w >> s0) & 0x01010101) << 4);
            B[k][0] = ((q[k].x >> 4) & M4) | (((qh[k].x >> s1) & 0x01010101) << 4);
            B[k][1] = ((q[k].y >> 4) & M4) | (((qh[k].y >> s1) & 0x01010101) << 4);
            B[k][2] = ((q[k].z >> 4) & M4) | (((qh[k].z >> s1) & 0x01010101) << 4);
            B[k][3] = ((q[k].w >> 4) & M4) | (((qh[k].w >> s1) & 0x01010101) << 4);
        }
        accR<T, R, 32 * IT>(s, u, A, B, fA, gA, fB, gB, acc);
    }
}

// repacked Q6_K block (224 B): qlo[128] (q4k-style nibbles) | qhi[64] (8 B per unit) |
// meta[32]: for each 64-group p: sc[4p..4p+3], d (fp16), 2 pad bytes  -> one 8-byte load per unit
constexpr int Q6_BLK = 224;
template <int T, int IT, int R>
__device__ __forceinline__ void rows_q6k(const uint8_t * const (&rp)[R], const SX & s, float (&acc)[R][T]) {
    const int lane = threadIdx.x & 31;
#pragma unroll(IT > 8 ? (T >= 5 ? 1 : 4) : IT)
    for (int i = 0; i < IT; i++) {
        const int u = lane + 32 * i, b = u >> 3, p = (u >> 1) & 3, hh = u & 1;
        int4 q[R];
        int2 hb[R], meta[R];
#pragma unroll
        for (int k = 0; k < R; k++) {
            const uint8_t * blk = rp[k] + b * Q6_BLK;
            q[k] = ldw4(blk + 32 * p + 16 * hh);
            hb[k] = ldw2(blk + 128 + 8 * (u & 7));
            meta[k] = ldw2(blk + 192 + 8 * p);
        }
        int A[R][4], B[R][4];
        float fA[R], gA[R], fB[R], gB[R];
#pragma unroll
        for (int k = 0; k < R; k++) {
            const float d = __half2float(__ushort_as_half((unsigned short) (meta[k].y & 0xffff)));
            fA[k] = d * (float) (int8_t) (meta[k].x >> (8 * hh));
            fB[k] = d * (float) (int8_t) (meta[k].x >> (16 + 8 * hh));
            gA[k] = 32.0f * fA[k];
            gB[k] = 32.0f * fB[k];
            const unsigned hx = (unsigned) hb[k].x, hy = (unsigned) hb[k].y;
            A[k][0] = (q[k].x & M4) | (int) (((hx >> 0) & 0x03030303u) << 4);
            A[k][1] = (q[k].y & M4) | (int) (((hx >> 2) & 0x03030303u) << 4);
            A[k][2] = (q[k].z & M4) | (int) (((hx >> 4) & 0x03030303u) << 4);
            A[k][3] = (q[k].w & M4) | (int) (((hx >> 6) & 0x03030303u) << 4);
            B[k][0] = ((q[k].x >> 4) & M4) | (int) (((hy >> 0) & 0x03030303u) << 4);
            B[k][1] = ((q[k].y >> 4) & M4) | (int) (((hy >> 2) & 0x03030303u) << 4);
            B[k][2] = ((q[k].z >> 4) & M4) | (int) (((hy >> 4) & 0x03030303u) << 4);
            B[k][3] = ((q[k].w >> 4) & M4) | (int) (((hy >> 6) & 0x03030303u) << 4);
        }
        accR<T, R, 32 * IT>(s, u, A, B, fA, gA, fB, gB, acc);
    }
}

// repacked Q8_0: per unit 32 B (A 16 int8 | B 16 int8), scales per natural 32-block
template <int T, int IT, int R>
__device__ __forceinline__ void rows_q80(const uint8_t * const (&rp)[R], const __half * const (&dp)[R], const SX & s,
                                         float (&acc)[R][T]) {
    const int lane = threadIdx.x & 31;
#pragma unroll(IT > 8 ? (T >= 5 ? 1 : 4) : IT)
    for (int i = 0; i < IT; i++) {
        const int u = lane + 32 * i;
        int A[R][4], B[R][4];
        float fA[R], gA[R], fB[R], gB[R];
#pragma unroll
        for (int k = 0; k < R; k++) {
            const int4 a = ldw4(rp[k] + 32 * u), bq = ldw4(rp[k] + 32 * u + 16);
            const int dd = ldw1(dp[k] + 2 * (u >> 1));
            const float2 df = __half22float2(*reinterpret_cast<const __half2 *>(&dd));
            A[k][0] = a.x; A[k][1] = a.y; A[k][2] = a.z; A[k][3] = a.w;
            B[k][0] = bq.x; B[k][1] = bq.y; B[k][2] = bq.z; B[k][3] = bq.w;
            fA[k] = df.x; fB[k] = df.y; gA[k] = 0.f; gB[k] = 0.f;
        }
        accR<T, R, 32 * IT>(s, u, A, B, fA, gA, fB, gB, acc);
    }
}

// rows row0 .. row0+R-1 of m, weight type fixed at compile time
// u0: first unit of a K-slice (multiple of 32) when the activations in s hold only that slice
template <int TYPE>
__device__ __forceinline__ int unit_bytes(int u0) {
    return TYPE == T_Q4K ? (u0 >> 3) * 144 : TYPE == T_Q5K ? (u0 >> 3) * 176 : TYPE == T_Q6K ? (u0 >> 3) * Q6_BLK : 32 * u0;
}
template <int T, int IT, int R, int TYPE>
__device__ __forceinline__ void mrows(const Mat & m, int row0, const SX & s, float (&acc)[R][T], int u0 = 0) {
    const uint8_t * rp[R];
    const __half * dp[R];
#pragma unroll
    for (int k = 0; k < R; k++) {
        rp[k] = m.w + (size_t) (row0 + k) * m.rb + unit_bytes<TYPE>(u0);
        dp[k] = m.d + (size_t) (row0 + k) * m.nd + u0;
    }
    if constexpr (TYPE == T_Q4K) rows_q4k<T, IT, R>(rp, s, acc);
    else if constexpr (TYPE == T_Q5K) rows_q5k<T, IT, R>(rp, s, acc);
    else if constexpr (TYPE == T_Q6K) rows_q6k<T, IT, R>(rp, s, acc);
    else rows_q80<T, IT, R>(rp, dp, s, acc);
}

// Row pairs [lo, hi) of a single-matrix item range, distributed over all warps of the grid.
// out(row0, acc) receives the reduced sums for rows row0, row0 + 1.
template <int T, int IT, int TYPE, class Out>
__device__ __noinline__ void gemv_pairs(const Mat m, int lo, int hi, Out out) {
    const SX sx = make_sx(nullptr, 32 * IT); // activations already in shared memory
    const int W = nwarps(), gw = gwarp();
    const int first = lo + (((gw - lo) % W) + W) % W;
    for (int it = first; it < hi; it += W) {
        const int r = 2 * (it - lo);
        float a[2][T];
        zero_acc<T, 2>(a);
        mrows<T, IT, 2, TYPE>(m, r, sx, a);
        wsum_acc<T, 2>(a);
        out(r, a);
    }
}

// Q4_K or Q6_K matrix (the two types that occur for qkv / v projections)
template <int T, int IT, class Out>
__device__ __forceinline__ void gemv_pairs_46(const Mat & m, int lo, int hi, Out out) {
    if (m.type == T_Q6K) gemv_pairs<T, IT, T_Q6K>(m, lo, hi, out);
    else gemv_pairs<T, IT, T_Q4K>(m, lo, hi, out);
}


// Prefetch the first `bytes` of rows row0..row0+n-1 into L2 so the DRAM stream of a phase starts
// while its activations are still being loaded into shared memory.
__device__ __forceinline__ void pf_rows(const Mat & m, int row0, int n, int bytes) {
    const int lane = threadIdx.x & 31;
    for (int k = 0; k < n; k++) {
        const uint8_t * p = m.w + (size_t) (row0 + k) * m.rb;
        for (int off = lane * 128; off < bytes; off += 32 * 128) asm volatile("prefetch.global.L2 [%0];" ::"l"(p + off));
    }
}

// lanes 0-15 quantise 16-chunk c of row t, lanes 16-31 the same chunk of row t + 1 (if valid)
__device__ __forceinline__ void q16_store(const QBuf & q, int t, int c, float v, bool valid) {
    const int lane = threadIdx.x & 31;
    float amax = fabsf(v);
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 8));
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 4));
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 2));
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1));
    const float d = amax / 127.0f;
    const int qi = amax == 0.0f ? 0 : (int) roundf(v / d);
    int qs = qi;
    qs += __shfl_xor_sync(0xffffffffu, qs, 8);
    qs += __shfl_xor_sync(0xffffffffu, qs, 4);
    qs += __shfl_xor_sync(0xffffffffu, qs, 2);
    qs += __shfl_xor_sync(0xffffffffu, qs, 1);
    if (!valid) return;
    const int b = c >> 4, p = (c >> 2) & 3, part = (c >> 1) & 1, hh = c & 1, byte = lane & 15;
    const int gi = t * q.U + 8 * b + 2 * p + hh;
    (part ? q.xb : q.xa)[gi * 16 + byte] = (int8_t) qi;
    if (byte == 0) {
        reinterpret_cast<__half *>(q.meta + gi)[part] = __float2half_rn(d);
        reinterpret_cast<short *>(q.meta + gi)[2 + part] = (short) qs;
    }
}

template <int T>
__device__ __forceinline__ void wsum_all(float (&a)[T]) {
#pragma unroll
    for (int t = 0; t < T; t++) a[t] = wsum(a[t]);
}

// ------------------------------------------------------------------------------------------
// phases
// ------------------------------------------------------------------------------------------

__device__ __forceinline__ float dequant_q4k(const uint8_t * row, int e) {
    const int b = e >> 8, w = e & 255, sb = w >> 5, l = w & 31;
    const uint8_t * blk = row + b * 144;
    const float d = __half2float(*reinterpret_cast<const __half *>(blk));
    const float dmin = __half2float(*reinterpret_cast<const __half *>(blk + 2));
    const uint8_t * sc = blk + 4;
    int s, m;
    if (sb < 4) {
        s = sc[sb] & 63;
        m = sc[sb + 4] & 63;
    } else {
        s = (sc[sb + 4] & 0xF) | ((sc[sb - 4] >> 6) << 4);
        m = (sc[sb + 4] >> 4) | ((sc[sb] >> 6) << 4);
    }
    const uint8_t qb = blk[16 + 32 * (sb >> 1) + l];
    const int qv = (sb & 1) ? (qb >> 4) : (qb & 0xF);
    return d * (float) s * (float) qv - dmin * (float) m;
}

// token embedding rows -> x, plus quantised attn_norm input for layer 0
template <int T>
__device__ __noinline__ void embed_prologue() {
    const int t = blockIdx.x;
    if (t >= T) return;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int tok = ((volatile Ctl *) cB.ctl)->tok[t];
    const uint8_t * row = cW.embd + (size_t) tok * (D / 256 * 144);
    const float * w0 = cW.L[0].attn_norm;
    for (int j = warp; j < U_D; j += NW) {
        const int e = 32 * j + lane;
        const float v = dequant_q4k(row, e);
        cB.x[t * D + e] = v;
        q32_store(cB.qn, t, j, v * w0[e], v * v, true);
    }
}

// Output rows (32 consecutive rows per block item) with optional residual add; the new
// residual value is re-quantised for the next RMS-normed consumer (weight wnext).
template <int T, int IT, int TYPE>
__device__ __noinline__ void resid32(const Mat W, const QBuf in, float * x, bool add, const float * wnext, uint8_t * /*sm*/) {
    uint8_t * const sm = smem;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    float * s_rs = misc(sm);
    float * s_res = misc(sm) + 8;
    if (blockIdx.x < D / 32) pf_rows(W, 32 * blockIdx.x + 2 * warp, 2, W.rb / IT);
    if (threadIdx.x < 4 && blockIdx.x + threadIdx.x * gridDim.x < D / 32) pf_l2(wnext + 32 * (blockIdx.x + threadIdx.x * gridDim.x));
    auto epilogue = [&](int j, float (&a)[2][T]) {
        wsum_acc<T, 2>(a);
        if (lane == 0) {
#pragma unroll
            for (int t = 0; t < T; t++) {
                s_res[t * 32 + 2 * warp] = a[0][t];
                s_res[t * 32 + 2 * warp + 1] = a[1][t];
            }
        }
        __syncthreads();
        if (warp < T) {
            const int t = warp, row = 32 * j + lane;
            float xv = s_res[t * 32 + lane];
            if (add) xv += __ldcg(x + t * D + row);
            x[t * D + row] = xv;
            q32_store(cB.qn, t, j, xv * wnext[row], xv * xv, true);
        }
        __syncthreads();
    };
    if constexpr (T * 32 * IT > SM_XU) {
        // The activations do not fit (T = 5 ffn_down): fill one K-half at a time.  A block's two row
        // groups are processed one after the other; the first group's per-lane partial sums of the
        // first half are parked in shared memory while the second group is computed.
        constexpr int U0 = 32 * ((IT + 1) / 2), U1 = 32 * IT - U0;
        static_assert(T * U0 * 40 + 2 * T * NT * 4 <= SM_MISC, "T = 5 ffn_down shared memory");
        auto sxh = [&](int US) {
            SX r;
            r.xa = reinterpret_cast<int4 *>(sm);
            r.xb = reinterpret_cast<int4 *>(sm + T * US * 16);
            r.xm = reinterpret_cast<uint2 *>(sm + 2 * T * US * 16);
            r.U = US;
            return r;
        };
        float * park = reinterpret_cast<float *>(sm + T * U0 * 40); // [2 * T][NT]
        const SX s0 = sxh(U0), s1 = sxh(U1);
        for (int j0 = blockIdx.x; j0 < D / 32; j0 += 2 * gridDim.x) {
            const int j1 = j0 + gridDim.x;
            fill_xs_part<T, U0>(in, 0, s0);
            if (j0 == blockIdx.x) mark_fill();
            float a[2][T];
            zero_acc<T, 2>(a);
            mrows<T, U0 / 32, 2, TYPE>(W, 32 * j0 + 2 * warp, s0, a);
#pragma unroll
            for (int t = 0; t < T; t++) {
                park[t * NT + threadIdx.x] = a[0][t];
                park[(T + t) * NT + threadIdx.x] = a[1][t];
            }
            if (j1 < D / 32) {
                zero_acc<T, 2>(a);
                mrows<T, U0 / 32, 2, TYPE>(W, 32 * j1 + 2 * warp, s0, a);
            }
            fill_xs_part<T, U1>(in, U0, s1);
            if (j1 < D / 32) {
                mrows<T, U1 / 32, 2, TYPE>(W, 32 * j1 + 2 * warp, s1, a, U0);
                epilogue(j1, a);
            }
#pragma unroll
            for (int t = 0; t < T; t++) {
                a[0][t] = park[t * NT + threadIdx.x];
                a[1][t] = park[(T + t) * NT + threadIdx.x];
            }
            mrows<T, U1 / 32, 2, TYPE>(W, 32 * j0 + 2 * warp, s1, a, U0);
            epilogue(j0, a);
        }
    } else {
        const SX sx = make_sx(sm, in.U);
        fill_xs<T>(in, 0, false, sx, s_rs);
        for (int j = blockIdx.x; j < D / 32; j += gridDim.x) {
            float a[2][T];
            zero_acc<T, 2>(a);
            mrows<T, IT, 2, TYPE>(W, 32 * j + 2 * warp, sx, a);
            epilogue(j, a);
        }
    }
}

// FFN gate/up: warp item = 16 consecutive rows of both matrices (4 rows in flight sharing the
// activations) -> silu(g)*u quantised per 16-chunk for the down projection.
template <int T>
__device__ __noinline__ void ffn_gu(const LayerW & W, uint8_t * /*sm*/) {
    uint8_t * const sm = smem;
    const int lane = threadIdx.x & 31;
    const SX sx = make_sx(sm, U_D);
    const float * s_rs = misc(sm);
    if (gwarp() < FF / 16) {
        pf_rows(W.fg, 16 * gwarp(), 2, 576);
        pf_rows(W.fu, 16 * gwarp(), 2, 576);
    }
    fill_xs<T>(cB.qn, 0, true, sx, misc(sm));
    const uint8_t * gw = W.fg.w;
    const uint8_t * uw = W.fu.w;
    const int rb = W.fg.rb;
    for (int j = gwarp(); j < FF / 16; j += nwarps()) {
        float act[T];
#pragma unroll
        for (int t = 0; t < T; t++) act[t] = 0.f;
        constexpr int RG = T >= 3 ? 1 : 2; // rows per matrix per step (fewer at large T: register budget)
#pragma unroll 1
        for (int r = 0; r < 16; r += RG) {
            const int row = 16 * j + r;
            const uint8_t * rp[2 * RG];
#pragma unroll
            for (int k = 0; k < RG; k++) {
                rp[k] = gw + (size_t) (row + k) * rb;
                rp[RG + k] = uw + (size_t) (row + k) * rb;
            }
            float a[2 * RG][T];
            zero_acc<T, 2 * RG>(a);
            rows_q4k<T, 5, 2 * RG>(rp, sx, a);
            wsum_acc<T, 2 * RG>(a);
#pragma unroll
            for (int k = 0; k < RG; k++)
                if (lane == r + k) {
#pragma unroll
                    for (int t = 0; t < T; t++) act[t] = silu(a[k][t] * s_rs[t]) * (a[RG + k][t] * s_rs[t]);
                }
        }
#pragma unroll
        for (int t = 0; t < T; t += 2) {
            const float other = __shfl_sync(0xffffffffu, act[t + 1 < T ? t + 1 : t], lane & 15);
            const float v = lane < 16 ? act[t] : other;
            q16_store(cB.qa, lane < 16 ? t : t + 1, j, v, lane < 16 || t + 1 < T);
        }
    }
}

template <int T>
__device__ __forceinline__ void store_lane_t(float * base, int stride, int idx, const float (&a)[T]) {
    const int lane = threadIdx.x & 31;
#pragma unroll
    for (int t = 0; t < T; t++)
        if (lane == t) base[t * stride + idx] = a[t];
}

// a[k][t] -> base[t * stride + idx + k]
template <int T>
__device__ __forceinline__ void store_pair_t(float * base, int stride, int idx, const float (&a)[2][T]) {
    const int lane = threadIdx.x & 31;
    const float * s_rs = misc(nullptr);
#pragma unroll
    for (int t = 0; t < T; t++)
        if ((lane >> 1) == t) base[t * stride + idx + (lane & 1)] = ((lane & 1) ? a[1][t] : a[0][t]) * s_rs[t];
}

// delta-net input projections: alpha/beta (fp32), qkv -> conv ring, z
template <int T>
__device__ __noinline__ void dn_proj(const LayerW & W, int dl, int n, int dn_c, uint8_t * /*sm*/) {
    uint8_t * const sm = smem;
    const int lane = threadIdx.x & 31;
    const SX sx = make_sx(sm, U_D);
    float * s_rs = misc(sm);
    {
        const int gw = gwarp(), W_ = nwarps(), lo = AB_SPLIT * 2 * NVH;
        const int first = lo + (((gw - lo) % W_) + W_) % W_;
        if (first < lo + QKV / 2) pf_rows(W.qkv, 2 * (first - lo), 2, 1024);
    }
    fill_xs<T>(cB.qn, 0, true, sx, s_rs);
    float * conv = cB.conv + (size_t) dl * CONV_RING * QKV;
    const int n_ab = AB_SPLIT * 2 * NVH, n_qkv = QKV / 2, n_z = VD / 2;
    // alpha / beta: fp32 rows with the norm weight folded in (dot with the raw residual), each
    // row split into AB_SPLIT K-chunks so these heavier items do not form a tail
    for (int it = gwarp(); it < n_ab; it += nwarps()) {
        const int row = it % (2 * NVH), ch = it / (2 * NVH);
        const float * wr = W.ab + (size_t) row * D;
        float acc[T];
#pragma unroll
        for (int t = 0; t < T; t++) acc[t] = 0.f;
        for (int k = ch * (D / AB_SPLIT) + lane * 4; k < (ch + 1) * (D / AB_SPLIT); k += 128) {
            const float4 w = ldw4f(wr + k);
#pragma unroll
            for (int t = 0; t < T; t++) {
                const float4 xv = __ldcg(reinterpret_cast<const float4 *>(cB.x + t * D + k));
                acc[t] += w.x * xv.x + w.y * xv.y + w.z * xv.z + w.w * xv.w;
            }
        }
#pragma unroll
        for (int t = 0; t < T; t++) acc[t] = wsum(acc[t]) * s_rs[t];
        store_lane_t<T>(cB.ab + ch * MAXT * 2 * NVH, 2 * NVH, row, acc);
    }
    gemv_pairs_46<T, 5>(W.qkv, n_ab, n_ab + n_qkv, [=](int r, const float (&a)[2][T]) {
        const int lane = threadIdx.x & 31;
#pragma unroll
        for (int t = 0; t < T; t++)
            if ((lane >> 1) == t)
                conv[((n + t) % CONV_RING) * QKV + r + (lane & 1)] = ((lane & 1) ? a[1][t] : a[0][t]) * misc(nullptr)[t];
    });
    gemv_pairs<T, 5, T_Q4K>(W.gate, n_ab + n_qkv, n_ab + n_qkv + n_z,
                            [=](int r, const float (&a)[2][T]) { store_pair_t<T>(cB.z, VD, r, a); });
    // Warm L2 with what dn_core will read, while the other blocks are still projecting:
    // the head's recurrent state (64 KB), its conv weights, older conv rows and replay entries.
    if (blockIdx.x < NVH) {
        const int h = blockIdx.x, kh = h % NKH, tid = threadIdx.x;
        pf_l2(cB.dstate + ((size_t) dl * NVH + h) * 128 * 128 + tid * 32);
        if (tid == 0) {
            pf_l2(W.dt + (h & ~31));
            pf_l2(W.A + (h & ~31));
            pf_l2(W.snorm);
            pf_l2(W.snorm + 32);
            pf_l2(W.snorm + 64);
            pf_l2(W.snorm + 96);
        }
        if (tid < 384) {
            const int ch = tid < 128 ? kh * 128 + tid : (tid < 256 ? 2048 + kh * 128 + (tid - 128) : 4096 + h * 128 + (tid - 256));
            if ((tid & 7) == 0) pf_l2(W.conv + ch * 4);
            if ((tid & 31) == 0)
                for (int j = 0; j < 3; j++)
                    if (n - 3 + j >= 0) pf_l2(conv + ((n - 3 + j) % CONV_RING) * QKV + ch);
        }
        const float * ring = cB.kvgb + (size_t) dl * KV_RING * KVGB;
        for (int i = tid; i < (n - dn_c) * 8; i += NT) {
            const float * slot = ring + (size_t) ((dn_c + (i >> 3)) % KV_RING) * KVGB;
            const int c = (i & 7);
            pf_l2(c < 4 ? slot + kh * 128 + c * 32 : slot + KVGB_V + h * 128 + (c - 4) * 32);
        }
    }
}

// delta-net core: one block per value head (48 blocks).  conv1d+silu, l2-norm, gated delta
// rule (with replay of pending accepted tokens), gated rms-norm, quantise for ssm_out.
template <int T>
__device__ __noinline__ void dn_core(const LayerW & W, int dl, int n, int dn_c, int n_commit, uint8_t * /*sm*/) {
    uint8_t * const sm = smem;
    const int h = blockIdx.x;
    if (h >= NVH) return;
    const int kh = h % NKH;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    // thread owns state column `col`, rows [32 r, 32 r + 32)  (S stored as [col][row])
    const int col = tid >> 2, r = tid & 3;
    constexpr int NE = T + MAXT - 1;     // max events: up to MAXT - 1 replayed accepted drafts + T new tokens
    constexpr int QS = 4 * 36;           // q / k with a 4-float pad per 32-row quarter (bank spread)
    float * s_q = reinterpret_cast<float *>(sm); // [NE][QS]
    float * s_k = s_q + NE * QS;                 // [NE][QS]
    float * s_v = s_k + NE * QS;                 // [NE][128]
    float * s_o = s_v + NE * 128;                // [T][128]
    float * s_z = s_o + T * 128;                 // [T][128]
    float * s_gb = s_z + T * 128;                // [NE][2]
    float * s_raw = s_gb + 32;                   // [T][384] conv outputs (silu), 16-byte aligned

    float * Sg = cB.dstate + ((size_t) dl * NVH + h) * 128 * 128 + col * 128 + 32 * r;
    float4 S[8];
#pragma unroll
    for (int i = 0; i < 8; i++) S[i] = __ldcg(reinterpret_cast<const float4 *>(Sg) + i);

    const int np = n - dn_c; // replayed (accepted, not yet committed) tokens
    float * ring = cB.kvgb + (size_t) dl * KV_RING * KVGB;
    const float * conv = cB.conv + (size_t) dl * CONV_RING * QKV;

    // ---- prologue: every input of every event, one round of loads ----
    if (tid < 384) {
        const int ch = tid < 128 ? kh * 128 + tid : (tid < 256 ? 2048 + kh * 128 + (tid - 128) : 4096 + h * 128 + (tid - 256));
        const float4 w = *reinterpret_cast<const float4 *>(W.conv + ch * 4);
        float raw[T + 3];
#pragma unroll
        for (int j = 0; j < T + 3; j++) {
            const int p = n - 3 + j;
            raw[j] = p >= 0 ? __ldcg(conv + (p % CONV_RING) * QKV + ch) : 0.f;
        }
#pragma unroll
        for (int t = 0; t < T; t++)
            s_raw[t * 384 + tid] = silu(w.x * raw[t] + w.y * raw[t + 1] + w.z * raw[t + 2] + w.w * raw[t + 3]);
    } else if (tid < 384 + T) {
        const int t = tid - 384;
        float al = 0.f, be = 0.f;
#pragma unroll
        for (int c = 0; c < AB_SPLIT; c++) {
            al += __ldcg(cB.ab + (c * MAXT + t) * 2 * NVH + h);
            be += __ldcg(cB.ab + (c * MAXT + t) * 2 * NVH + NVH + h);
        }
        float sp = al + W.dt[h];
        sp = sp > 20.0f ? sp : logf(1.0f + expf(sp));
        s_gb[2 * (np + t)] = expf(sp * W.A[h]);
        s_gb[2 * (np + t) + 1] = 1.0f / (1.0f + expf(-be));
    } else if (tid >= 448 && tid < 448 + np) {
        const int e = tid - 448;
        const float * slot = ring + (size_t) ((dn_c + e) % KV_RING) * KVGB;
        s_gb[2 * e] = __ldcg(slot + KVGB_G + h);
        s_gb[2 * e + 1] = __ldcg(slot + KVGB_B + h);
    }
    for (int i = tid; i < np * 128; i += NT) { // replayed k / v
        const int e = i >> 7, c = i & 127;
        const float * slot = ring + (size_t) ((dn_c + e) % KV_RING) * KVGB;
        s_k[e * QS + (c >> 5) * 36 + (c & 31)] = __ldcg(slot + kh * 128 + c);
        s_v[e * 128 + c] = __ldcg(slot + KVGB_V + h * 128 + c);
    }
    for (int i = tid; i < T * 128; i += NT) s_z[i] = __ldcg(cB.z + (i >> 7) * VD + h * 128 + (i & 127));
    __syncthreads();

    // l2-normalise q and k of the new tokens (warp 2t: q of token t, warp 2t+1: k), copy v
    if (warp < 2 * T) {
        const int t = warp >> 1, isk = warp & 1;
        const float4 v = *reinterpret_cast<const float4 *>(s_raw + t * 384 + isk * 128 + lane * 4);
        const float ss = wsum(v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w);
        float sc = rsqrtf(fmaxf(ss, EPS * EPS));
        if (!isk) sc *= 0.08838834764831845f; // 1/sqrt(128)
        float * dst = (isk ? s_k : s_q) + (np + t) * QS + (lane >> 3) * 36 + (lane & 7) * 4;
        *reinterpret_cast<float4 *>(dst) = make_float4(v.x * sc, v.y * sc, v.z * sc, v.w * sc);
    }
    for (int i = tid; i < T * 128; i += NT) s_v[(np + (i >> 7)) * 128 + (i & 127)] = s_raw[(i >> 7) * 384 + 256 + (i & 127)];
    __syncthreads();

    // save the new tokens' (k, v, g, beta) for a possible replay next cycle
    for (int i = tid; i < T * 128; i += NT) {
        const int t = i >> 7, c = i & 127;
        float * slot = ring + (size_t) ((n + t) % KV_RING) * KVGB;
        if (h < NKH) slot[h * 128 + c] = s_k[(np + t) * QS + (c >> 5) * 36 + (c & 31)];
        slot[KVGB_V + h * 128 + c] = s_v[(np + t) * 128 + c];
        if (c == 0) {
            slot[KVGB_G + h] = s_gb[2 * (np + t)];
            slot[KVGB_B + h] = s_gb[2 * (np + t) + 1];
        }
    }

    // ---- gated delta rule over all events; inputs are in shared memory, no barriers needed ----
    for (int e = 0; e < np + T; e++) {
        const float g = s_gb[2 * e], beta = s_gb[2 * e + 1];
        const float4 * kq = reinterpret_cast<const float4 *>(s_k + e * QS + r * 36);
        float kv = 0.f;
#pragma unroll
        for (int i = 0; i < 8; i++) {
            const float4 k4 = kq[i];
            kv += S[i].x * k4.x + S[i].y * k4.y + S[i].z * k4.z + S[i].w * k4.w;
        }
        kv += __shfl_xor_sync(0xffffffffu, kv, 1);
        kv += __shfl_xor_sync(0xffffffffu, kv, 2);
        const float dlt = (s_v[e * 128 + col] - g * kv) * beta;
#pragma unroll
        for (int i = 0; i < 8; i++) {
            const float4 k4 = kq[i];
            S[i].x = g * S[i].x + k4.x * dlt;
            S[i].y = g * S[i].y + k4.y * dlt;
            S[i].z = g * S[i].z + k4.z * dlt;
            S[i].w = g * S[i].w + k4.w * dlt;
        }
        if (e >= np) {
            const float4 * qq = reinterpret_cast<const float4 *>(s_q + e * QS + r * 36);
            float o = 0.f;
#pragma unroll
            for (int i = 0; i < 8; i++) {
                const float4 q4 = qq[i];
                o += S[i].x * q4.x + S[i].y * q4.y + S[i].z * q4.z + S[i].w * q4.w;
            }
            o += __shfl_xor_sync(0xffffffffu, o, 1);
            o += __shfl_xor_sync(0xffffffffu, o, 2);
            if (r == 0) s_o[(e - np) * 128 + col] = o;
        }
        if (e == np + n_commit - 1) {
#pragma unroll
            for (int i = 0; i < 8; i++) reinterpret_cast<float4 *>(Sg)[i] = S[i];
        }
    }
    __syncthreads();

    // gated rms-norm + quantise: warp w -> token w / 4, 32-element block w % 4 of this head
    for (int w = warp; w < 4 * T; w += NW) {
        const int t = w >> 2, jb = w & 3;
        const float4 v = *reinterpret_cast<const float4 *>(s_o + t * 128 + lane * 4);
        const float rs = rsqrtf(wsum(v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w) / 128.0f + EPS);
        const int c = jb * 32 + lane;
        const float y = s_o[t * 128 + c] * rs * W.snorm[c] * silu(s_z[t * 128 + c]);
        q32_store(cB.qo, t, 4 * h + jb, y, 0.f, false);
    }
}

// q8_0 store of one KV row: lane holds values 8 lane .. 8 lane + 7 (4 lanes per 32-value block)
__device__ __forceinline__ void kv_store_q8(int8_t * q, __half * sc, const float (&v)[8]) {
    const int lane = threadIdx.x & 31;
    float am = 0.f;
#pragma unroll
    for (int k = 0; k < 8; k++) am = fmaxf(am, fabsf(v[k]));
    am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, 1));
    am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, 2));
    const float id = am > 0.f ? 127.0f / am : 0.f;
    unsigned w[2];
#pragma unroll
    for (int h = 0; h < 2; h++) {
        unsigned x = 0;
#pragma unroll
        for (int k = 0; k < 4; k++) x |= ((unsigned) __float2int_rn(v[4 * h + k] * id) & 0xffu) << (8 * k);
        w[h] = x;
    }
    *reinterpret_cast<uint2 *>(q + lane * 8) = make_uint2(w[0], w[1]);
    if ((lane & 3) == 0) sc[lane >> 2] = __float2half_rn(am / 127.0f);
}

// fp16 store of one KV row (same lane layout)
__device__ __forceinline__ void kv_store_f16(__half * dst, const float (&v)[8]) {
    __half2 h[4];
#pragma unroll
    for (int k = 0; k < 4; k++) h[k] = __floats2half2_rn(v[2 * k], v[2 * k + 1]);
    *reinterpret_cast<int4 *>(dst + (threadIdx.x & 31) * 8) = *reinterpret_cast<int4 *>(h);
}

// 8 q8 values + scale -> 4 fp16 pairs
__device__ __forceinline__ void q8_to_f16(uint2 q, float d, __half2 (&h)[4]) {
#pragma unroll
    for (int k = 0; k < 4; k++) {
        const unsigned x = k < 2 ? q.x : q.y;
        const int sh = 16 * (k & 1);
        h[k] = __floats2half2_rn(d * (float) (int8_t) (x >> sh), d * (float) (int8_t) (x >> (sh + 8)));
    }
}

// staged q8 K/V rows (KVQ_ROW bytes: values, then scales) -> fp16 rows of stride ld; rows x 2 tensors
__device__ __forceinline__ void kv_tile_to_f16(const uint8_t * src, __half * dK, __half * dV, int rows, int ld) {
    for (int i = threadIdx.x; i < 2 * rows * 32; i += blockDim.x) {
        const int which = i / (rows * 32), r = (i >> 5) % rows, c = i & 31;
        const uint8_t * row = src + (size_t) (which * rows + r) * KVQ_ROW;
        __half2 h[4];
        q8_to_f16(*reinterpret_cast<const uint2 *>(row + 8 * c), __half2float(reinterpret_cast<const __half *>(row + 256)[c >> 2]), h);
        *reinterpret_cast<int4 *>((which ? dV : dK) + (size_t) r * ld + 8 * c) = *reinterpret_cast<int4 *>(h);
    }
}

// async copy of q8 K/V rows r < np (positions row0 + r) into the staging layout (rows >= np: zero scales)
__device__ __forceinline__ void kv_stage(uint8_t * dst, size_t row0, int rows, int np) {
    const int8_t * Kq = static_cast<const int8_t *>(cB.kc), * Vq = static_cast<const int8_t *>(cB.vc);
    for (int i = threadIdx.x; i < 2 * rows * 17; i += blockDim.x) {
        const int which = i / (rows * 17), r = (i / 17) % rows, c = i % 17;
        uint8_t * d = dst + (size_t) (which * rows + r) * KVQ_ROW + 16 * c;
        if (r < np) {
            const size_t row = row0 + r;
            if (c < 16) cp_async16(d, (which ? Vq : Kq) + row * HD + 16 * c);
            else cp_async16(d, (which ? cB.vsc : cB.ksc) + row * (HD / 32));
        } else if (c == 16) {
            *reinterpret_cast<int4 *>(d) = make_int4(0, 0, 0, 0);
        }
    }
}

// split-K over the context: at most 20 chunks per kv head so all 4 x nch items run in one wave
__device__ __forceinline__ void attn_dims(int L_ctx, int & nch, int & cs) {
    nch = max(1, min(20, (L_ctx + 63) / 64));
    cs = (L_ctx + nch - 1) / nch;
}

// attention q/k/v projections
template <int T>
__device__ __noinline__ void attn_proj(const LayerW & W, int slot, int n, uint8_t * /*sm*/) {
    uint8_t * const sm = smem;
    const int lane = threadIdx.x & 31;
    (void) lane;
    const SX sx = make_sx(sm, U_D);
    if (gwarp() < QG / 2) pf_rows(W.q, 2 * gwarp(), 2, 576);
    fill_xs<T>(cB.qn, 0, true, sx, misc(sm));
    const int nq = QG / 2, nk = KVD / 2;
    gemv_pairs<T, 5, T_Q4K>(W.q, 0, nq, [=](int r, const float (&a)[2][T]) { store_pair_t<T>(cB.qbuf, QG, r, a); });
    gemv_pairs<T, 5, T_Q4K>(W.k, nq, nq + nk, [=](int r, const float (&a)[2][T]) { store_pair_t<T>(cB.kbuf, KVD, r, a); });
    gemv_pairs_46<T, 5>(W.v, nq + nk, nq + 2 * nk, [=](int r, const float (&a)[2][T]) { store_pair_t<T>(cB.vbuf, KVD, r, a); });
    // Warm L2 with what attn_core reads first: q/k norm weights, RoPE rows of the new positions,
    // and the first K / V tile of this block's attention item (positions < n).
    pf_block(W.qn, HD * 4);
    pf_block(W.kn, HD * 4);
    pf_block(cB.rope + (size_t) n * 32, T * 32 * 8);
    int nch, CS;
    attn_dims(n + T, nch, CS);
    const size_t ctx = cB.ctx;
    const int rowb = cB.kvq8 ? HD : HD * 2, lines = rowb / 128; // bytes, 128-byte lines per K / V row
    const uint8_t * Kc = static_cast<const uint8_t *>(cB.kc) + (size_t) slot * NKV * ctx * rowb;
    const uint8_t * Vc = static_cast<const uint8_t *>(cB.vc) + (size_t) slot * NKV * ctx * rowb;
    for (int item = blockIdx.x; item < NKV * nch; item += gridDim.x) {
        const int g = item / nch, c0 = (item % nch) * CS, c1 = min(n, min(c0 + CS, c0 + 64));
        for (int i = threadIdx.x; i < (c1 - c0) * lines; i += NT) {
            const size_t off = ((size_t) g * ctx + c0 + i / lines) * rowb + (i % lines) * 128;
            pf_l2(Kc + off);
            pf_l2(Vc + off);
        }
    }
}

__device__ __forceinline__ void rope8(float (&v)[8], int lane, int pos) {
    float o[8];
#pragma unroll
    for (int k = 0; k < 8; k++) o[k] = __shfl_xor_sync(0xffffffffu, v[k], 4);
    if (lane < 8) {
        const bool first = lane < 4;
        const float2 * tab = cB.rope + (size_t) pos * 32 + (lane & 3) * 8;
#pragma unroll
        for (int k = 0; k < 8; k++) {
            const float2 cs = __ldg(tab + k);
            v[k] = first ? v[k] * cs.x - o[k] * cs.y : o[k] * cs.y + v[k] * cs.x;
        }
    }
}

__device__ __forceinline__ void load8(const float * p, float (&v)[8]) {
    const float4 a = __ldcg(reinterpret_cast<const float4 *>(p));
    const float4 b = __ldcg(reinterpret_cast<const float4 *>(p + 4));
    v[0] = a.x; v[1] = a.y; v[2] = a.z; v[3] = a.w;
    v[4] = b.x; v[5] = b.y; v[6] = b.z; v[7] = b.w;
}


// split-K flash decoding over (kv head, chunk) items; also writes this cycle's K/V.
__device__ __forceinline__ void mma16816(float (&c)[4], unsigned a0, unsigned a1, unsigned a2, unsigned a3, unsigned b0,
                                         unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// Split-K flash decoding over (kv head, chunk) items.  The 6*T query rows of a kv head form the
// M dimension of tensor-core MMAs: S = Q K^T and O += P V per 64-position tile of K / V staged
// in shared memory (fp16 operands, fp32 accumulation, online softmax).  Also writes this cycle's K/V.
template <int T, bool Q8>
__device__ __forceinline__ void attn_core_t(const LayerW & W, int slot, int n) {
    uint8_t * const sm = smem;
    constexpr int R = 6 * T;
    constexpr int MT = (R + 15) / 16; // 16-row MMA tiles of query rows
    constexpr int TP = ATP;           // positions per K/V tile; tiles are double-buffered
    constexpr int LDH = HD + 8;       // smem row stride in halves (conflict-free fragment loads)
    constexpr int LDP = TP + 8;
    const int L_ctx = n + T;
    int nch, CS;
    attn_dims(L_ctx, nch, CS);
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g8 = lane >> 2, tig = lane & 3;
    __half * sQ = reinterpret_cast<__half *>(sm); // [32][LDH]
    // fp16 cache: 2 stages x (K [TP][LDH], V [TP][LDH]); q8 cache: one fp16 K/V tile + 2 q8 stages
    __half * sKV = sQ + 32 * LDH;
    uint8_t * sKVq = reinterpret_cast<uint8_t *>(sKV + 2 * TP * LDH);
    float * sS = reinterpret_cast<float *>(reinterpret_cast<uint8_t *>(sKV) + SM_ATTN_KV); // [32][TP]
    __half * sP = reinterpret_cast<__half *>(sS + 32 * TP); // [32][LDP]
    float * sM = reinterpret_cast<float *>(sP + 32 * LDP);
    float * sL = sM + 32;
    float * sAl = sL + 32;
    const size_t ctx = cB.ctx;
    const size_t slot0 = (size_t) slot * NKV * ctx; // first row of this slot
    __half * Kc = static_cast<__half *>(cB.kc) + slot0 * HD;
    __half * Vc = static_cast<__half *>(cB.vc) + slot0 * HD;
    const int mt = warp >> 3, wn = warp & 7; // MMA work split: warp -> (row tile, column group)

    auto issue_tile = [&](int g, int tb, int np_, int st) {
        if constexpr (Q8) {
            kv_stage(sKVq + st * 2 * TP * KVQ_ROW, slot0 + (size_t) g * ctx + tb, TP, np_);
            asm volatile("cp.async.commit_group;" ::: "memory");
            return;
        }
        __half * sK = sKV + st * 2 * TP * LDH, * sV = sK + TP * LDH;
        const __half * gK = Kc + ((size_t) g * ctx + tb) * HD;
        const __half * gV = Vc + ((size_t) g * ctx + tb) * HD;
        for (int i = tid; i < TP * (HD / 8); i += NT) {
            const int p = i >> 5, cc = (i & 31) * 8;
            if (p < np_) {
                cp_async16(sK + p * LDH + cc, gK + (size_t) p * HD + cc);
                cp_async16(sV + p * LDH + cc, gV + (size_t) p * HD + cc);
            } else {
                *reinterpret_cast<int4 *>(sK + p * LDH + cc) = make_int4(0, 0, 0, 0);
                *reinterpret_cast<int4 *>(sV + p * LDH + cc) = make_int4(0, 0, 0, 0);
            }
        }
        asm volatile("cp.async.commit_group;" ::: "memory");
    };
    for (int item = blockIdx.x; item < NKV * nch; item += gridDim.x) {
        const int g = item / nch, c = item % nch;
        const int c0 = c * CS, c1 = min(L_ctx, c0 + CS);
        // one round of small loads (q rows, norm weights, RoPE rows) before any bulk K/V traffic:
        // per-SM requests are served roughly in order
        float qv[2][8], qw[8];
        float2 rc[2][8];
        load8(W.qn + lane * 8, qw);
#pragma unroll
        for (int k2 = 0; k2 < 2; k2++) {
            const int r = warp + 16 * k2;
            if (r < R) {
                const int hq = r / T, t = r % T, head = g * 6 + hq;
                load8(cB.qbuf + t * QG + head * 512 + lane * 8, qv[k2]);
                if (lane < 8) {
                    const float2 * tab = cB.rope + (size_t) (n + t) * 32 + (lane & 3) * 8;
#pragma unroll
                    for (int k = 0; k < 8; k++) rc[k2][k] = __ldg(tab + k);
                }
            }
        }
#pragma unroll
        for (int k2 = 0; k2 < 2; k2++) {
            const int r = warp + 16 * k2;
            if (r < R) {
                float * v = qv[k2];
                float ss = 0.f;
#pragma unroll
                for (int k = 0; k < 8; k++) ss += v[k] * v[k];
                const float rs = rsqrtf(wsum(ss) / 256.0f + EPS);
#pragma unroll
                for (int k = 0; k < 8; k++) v[k] *= rs * qw[k];
                float o[8];
#pragma unroll
                for (int k = 0; k < 8; k++) o[k] = __shfl_xor_sync(0xffffffffu, v[k], 4);
                if (lane < 8) {
#pragma unroll
                    for (int k = 0; k < 8; k++)
                        v[k] = lane < 4 ? v[k] * rc[k2][k].x - o[k] * rc[k2][k].y : o[k] * rc[k2][k].y + v[k] * rc[k2][k].x;
                }
                __half2 hq2[4];
#pragma unroll
                for (int k = 0; k < 4; k++) hq2[k] = __floats2half2_rn(v[2 * k] * 0.0625f, v[2 * k + 1] * 0.0625f);
                *reinterpret_cast<int4 *>(sQ + r * LDH + lane * 8) = *reinterpret_cast<int4 *>(hq2);
            } else {
                *reinterpret_cast<int4 *>(sQ + r * LDH + lane * 8) = make_int4(0, 0, 0, 0);
                if (2 * lane < TP) *reinterpret_cast<unsigned *>(sP + r * LDP + 2 * lane) = 0u; // padded rows of P stay 0
            }
        }
        // this cycle's K / V rows that fall into the chunk (positions >= n)
        const int pn0 = max(c0, n), pn1 = min(c1, n + T);
        for (int pp = pn0 + warp; pp < pn1; pp += NW) {
            const int t = pp - n;
            float v[8];
            load8(cB.kbuf + t * KVD + g * 256 + lane * 8, v);
            float ss = 0.f;
#pragma unroll
            for (int k = 0; k < 8; k++) ss += v[k] * v[k];
            const float rs = rsqrtf(wsum(ss) / 256.0f + EPS);
#pragma unroll
            for (int k = 0; k < 8; k++) v[k] *= rs * W.kn[lane * 8 + k];
            rope8(v, lane, pp);
            const size_t row = slot0 + (size_t) g * ctx + pp;
            if constexpr (Q8) kv_store_q8(static_cast<int8_t *>(cB.kc) + row * HD, cB.ksc + row * (HD / 32), v);
            else kv_store_f16(static_cast<__half *>(cB.kc) + row * HD, v);
            load8(cB.vbuf + t * KVD + g * 256 + lane * 8, v);
            if constexpr (Q8) kv_store_q8(static_cast<int8_t *>(cB.vc) + row * HD, cB.vsc + row * (HD / 32), v);
            else kv_store_f16(static_cast<__half *>(cB.vc) + row * HD, v);
        }
        if (tid < 32) {
            sM[tid] = -INFINITY;
            sL[tid] = 0.f;
        }
        __threadfence_block();
        __syncthreads();

        float acc[4][4]; // O tiles: rows 16*mt + {g8, g8+8}, dims 8*(wn + 8j) + {2tig, 2tig+1}
#pragma unroll
        for (int j = 0; j < 4; j++)
#pragma unroll
            for (int k = 0; k < 4; k++) acc[j][k] = 0.f;

        issue_tile(g, c0, min(TP, c1 - c0), 0);
        int st = 0;
        for (int tb = c0; tb < c1; tb += TP, st ^= 1) {
            const int np_ = min(TP, c1 - tb);
            const __half * sK = Q8 ? sKV : sKV + st * 2 * TP * LDH, * sV = sK + TP * LDH;
            // the next tile streams into the other stage while this one is processed
            if (tb + TP < c1) {
                issue_tile(g, tb + TP, min(TP, c1 - tb - TP), st ^ 1);
                asm volatile("cp.async.wait_group 1;" ::: "memory");
            } else {
                asm volatile("cp.async.wait_group 0;" ::: "memory");
            }
            __syncthreads();
            if constexpr (Q8) { // q8 stage -> the fp16 tile
                kv_tile_to_f16(sKVq + st * 2 * TP * KVQ_ROW, sKV, sKV + TP * LDH, TP, LDH);
                __syncthreads();
            }
            // S = Q K^T : warp -> row tile mt, positions 8*wn .. 8*wn+7
            if (mt < MT && wn < TP / 8) {
                float s[4] = {0.f, 0.f, 0.f, 0.f};
                const __half * qa = sQ + (16 * mt + g8) * LDH + 2 * tig;
                const __half * kb = sK + (8 * wn + g8) * LDH + 2 * tig;
#pragma unroll
                for (int ks = 0; ks < HD / 16; ks++) {
                    const unsigned a0 = *reinterpret_cast<const unsigned *>(qa + 16 * ks);
                    const unsigned a1 = *reinterpret_cast<const unsigned *>(qa + 8 * LDH + 16 * ks);
                    const unsigned a2 = *reinterpret_cast<const unsigned *>(qa + 16 * ks + 8);
                    const unsigned a3 = *reinterpret_cast<const unsigned *>(qa + 8 * LDH + 16 * ks + 8);
                    const unsigned b0 = *reinterpret_cast<const unsigned *>(kb + 16 * ks);
                    const unsigned b1 = *reinterpret_cast<const unsigned *>(kb + 16 * ks + 8);
                    mma16816(s, a0, a1, a2, a3, b0, b1);
                }
                float * so = sS + (16 * mt + g8) * TP + 8 * wn + 2 * tig;
                so[0] = s[0];
                so[1] = s[1];
                so[8 * TP] = s[2];
                so[8 * TP + 1] = s[3];
            }
            __syncthreads();
            // online softmax per query row (causal mask: row token t sees positions <= n + t)
            for (int r = warp; r < R; r += NW) {
                const int lim = n + (r % T);
                float s0 = sS[r * TP + lane], s1 = TP > 32 ? sS[r * TP + (lane + 32) % TP] : -INFINITY;
                if (lane >= np_ || tb + lane > lim) s0 = -INFINITY;
                if (lane + 32 >= np_ || tb + lane + 32 > lim) s1 = -INFINITY;
                const float mt_ = wmax(fmaxf(s0, s1));
                const float mold = sM[r];
                const float mnew = fmaxf(mold, mt_);
                float p0 = 0.f, p1 = 0.f, al = 1.f;
                if (mnew != -INFINITY) {
                    p0 = expf(s0 - mnew);
                    p1 = expf(s1 - mnew);
                    al = expf(mold - mnew);
                }
                const float ps = wsum(p0 + p1);
                sP[r * LDP + lane] = __float2half_rn(p0);
                if (TP > 32) sP[r * LDP + (lane + 32) % TP] = __float2half_rn(p1);
                if (lane == 0) {
                    sL[r] = sL[r] * al + ps;
                    sM[r] = mnew;
                    sAl[r] = al;
                }
            }
            __syncthreads();
            // O = O * alpha + P V : warp -> row tile mt, dims 8 * (wn + 8 j)
            if (mt < MT) {
                const float al0 = sAl[16 * mt + g8], al1 = sAl[16 * mt + g8 + 8];
#pragma unroll
                for (int j = 0; j < 4; j++) {
                    acc[j][0] *= al0;
                    acc[j][1] *= al0;
                    acc[j][2] *= al1;
                    acc[j][3] *= al1;
                }
                const __half * pa = sP + (16 * mt + g8) * LDP + 2 * tig;
#pragma unroll
                for (int ks = 0; ks < TP / 16; ks++) {
                    const unsigned a0 = *reinterpret_cast<const unsigned *>(pa + 16 * ks);
                    const unsigned a1 = *reinterpret_cast<const unsigned *>(pa + 8 * LDP + 16 * ks);
                    const unsigned a2 = *reinterpret_cast<const unsigned *>(pa + 16 * ks + 8);
                    const unsigned a3 = *reinterpret_cast<const unsigned *>(pa + 8 * LDP + 16 * ks + 8);
#pragma unroll
                    for (int j = 0; j < 4; j++) {
                        const __half * vb = sV + (16 * ks + (lane & 15)) * LDH + 8 * (wn + 8 * j);
                        unsigned b0, b1;
                        asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];"
                                     : "=r"(b0), "=r"(b1)
                                     : "r"((unsigned) __cvta_generic_to_shared(vb)));
                        mma16816(acc[j], a0, a1, a2, a3, b0, b1);
                    }
                }
            }
            __syncthreads();
        }
        if (mt < MT) {
#pragma unroll
            for (int j = 0; j < 4; j++) {
                const int r0 = 16 * mt + g8, d0 = 8 * (wn + 8 * j) + 2 * tig;
                if (r0 < R) {
                    float * o = cB.apart + ((size_t) item * AR + r0) * 256 + d0;
                    o[0] = acc[j][0];
                    o[1] = acc[j][1];
                }
                if (r0 + 8 < R) {
                    float * o = cB.apart + ((size_t) item * AR + r0 + 8) * 256 + d0;
                    o[0] = acc[j][2];
                    o[1] = acc[j][3];
                }
            }
        }
        if (tid < R) cB.aml[item * AR + tid] = make_float2(sM[tid], sL[tid]);
        __syncthreads();
    }
}

template <int T>
__device__ __noinline__ void attn_core(const LayerW & W, int slot, int n, uint8_t * /*sm*/) {
    if (cB.kvq8) attn_core_t<T, true>(W, slot, n);
    else attn_core_t<T, false>(W, slot, n);
}

// combine split-K partials, apply sigmoid output gate, quantise for o_proj
template <int T>
__device__ __noinline__ void attn_combine(int n) {
    const int L_ctx = n + T;
    int nch, CS;
    attn_dims(L_ctx, nch, CS);
    const int tid = threadIdx.x, d = tid & 255, wh = (tid & 255) >> 5;
    for (int it = blockIdx.x * 2 + (tid >> 8); it < NH * T; it += 2 * gridDim.x) {
        const int h = it / T, t = it % T, g = h / 6, r = (h % 6) * T + t;
        float M = -INFINITY;
        for (int c = 0; c < nch; c++) M = fmaxf(M, __ldcg(&cB.aml[(g * nch + c) * AR + r].x));
        float num = 0.f, den = 0.f;
        for (int c = 0; c < nch; c++) {
            const float2 ml = __ldcg(&cB.aml[(g * nch + c) * AR + r]);
            const float w = ml.x == -INFINITY ? 0.f : expf(ml.x - M);
            num += w * __ldcg(cB.apart + ((size_t) (g * nch + c) * AR + r) * 256 + d);
            den += w * ml.y;
        }
        const float gate = __ldcg(cB.qbuf + t * QG + h * 512 + 256 + d);
        const float o = num / den / (1.0f + expf(-gate));
        q32_store(cB.qo, t, 8 * h + wh, o, 0.f, false);
    }
}

// final RMS norm rows -> h (fp32) for the MTP head.
template <int T>
__device__ __noinline__ void write_h_rows(const float * x, const QBuf & q, const float * w, float * dst_base, int pos0, bool ring,
                             uint8_t * /*sm*/) {
    uint8_t * const sm = smem;
    if (blockIdx.x != 0) return;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    float * s_rs = misc(sm);
    if (warp < T) {
        float acc = 0.f;
        for (int j = lane; j < U_D; j += 32) acc += __ldcg(q.ssq + warp * U_D + j);
        acc = wsum(acc);
        if (lane == 0) s_rs[warp] = rsqrtf(acc / (float) D + EPS);
    }
    __syncthreads();
    for (int i = threadIdx.x; i < T * D; i += NT) {
        const int t = i / D, e = i - t * D;
        float * dst = ring ? dst_base + (size_t) ((pos0 + t) % H_RING) * D : dst_base;
        if (!ring && t != T - 1) continue;
        dst[e] = __ldcg(x + i) * s_rs[t] * w[e];
    }
    __syncthreads();
}

// warp-wide sorted (descending) list of the 32 best (value, id), lane i holding rank i; v is warp-uniform.
// Returns false (and inserts nothing) when v does not beat the current 32nd value.
__device__ __forceinline__ bool topk_insert(float & tv, int & ti, float v, int id) {
    const float mn = __shfl_sync(0xffffffffu, tv, 31);
    if (!(v > mn)) return false;
    const int lane = threadIdx.x & 31;
    const int pos = __popc(__ballot_sync(0xffffffffu, tv >= v));
    const float uv = __shfl_up_sync(0xffffffffu, tv, 1);
    const int ui = __shfl_up_sync(0xffffffffu, ti, 1);
    if (lane == pos) { tv = v; ti = id; }
    else if (lane > pos) { tv = uv; ti = ui; }
    return true;
}

// lm_head over TL rows (qn rows t0..t0+TL-1) with greedy argmax; returns true in the single
// block that finished last, which holds the reduced argmax in s_tok.  TOPK: additionally keeps the
// TOPK_MAX best logits of each row in cB.cand (for sampling).  FULL: additionally stores every
// logit of each row in cB.logits (scoring).
template <int TL, bool TOPK = false, bool FULL = false>
__device__ __noinline__ bool lm_head(int t0, bool draft, uint8_t * /*sm*/, int * s_tok) {
    uint8_t * const sm = smem;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const SX sx = make_sx(sm, U_D);
    float * m = misc(sm);
    pf_rows(draft ? cW.head_d : cW.head, 2 * gwarp(), 2, 896);
    fill_xs<TL>(cB.qn, t0, true, sx, m);
    float bv[TL];
    int bi[TL];
#pragma unroll
    for (int t = 0; t < TL; t++) {
        bv[t] = -INFINITY;
        bi[t] = TOPK ? -1 : 0;
    }
    const Mat head = draft ? cW.head_d : cW.head;
    for (int r = 2 * gwarp(); r < head.N; r += 2 * nwarps()) {
        float a[2][TL];
        zero_acc<TL, 2>(a);
        mrows<TL, 5, 2, T_Q6K>(head, r, sx, a);
        wsum_acc<TL, 2>(a);
        if constexpr (FULL) {
            if (lane == 0) {
#pragma unroll
                for (int t = 0; t < TL; t++) {
                    cB.logits[(size_t) (t0 + t) * VOCAB + r] = a[0][t] * m[t];
                    cB.logits[(size_t) (t0 + t) * VOCAB + r + 1] = a[1][t] * m[t];
                }
            }
        }
#pragma unroll
        for (int t = 0; t < TL; t++) {
            if constexpr (TOPK) { // true logits: apply the final RMS norm scale (irrelevant to the argmax)
                topk_insert(bv[t], bi[t], a[0][t] * m[t], r);
                topk_insert(bv[t], bi[t], a[1][t] * m[t], r + 1);
            } else {
                if (a[0][t] > bv[t]) { bv[t] = a[0][t]; bi[t] = r; }
                if (a[1][t] > bv[t]) { bv[t] = a[1][t]; bi[t] = r + 1; }
            }
        }
    }
    if constexpr (TOPK) {
        // merge the warps' lists (shared memory, the activations are no longer needed), then the blocks'
        __syncthreads();
        float2 * wl = reinterpret_cast<float2 *>(sm); // [NW][TL][32]
#pragma unroll
        for (int t = 0; t < TL; t++) wl[(warp * TL + t) * 32 + lane] = make_float2(bv[t], __int_as_float(bi[t]));
        __syncthreads();
        if (warp < TL) {
            float v = -INFINITY;
            int id = -1;
            for (int w = 0; w < NW; w++)
                for (int j = 0; j < 32; j++) {
                    const float2 e = wl[(w * TL + warp) * 32 + j];
                    if (!topk_insert(v, id, e.x, __float_as_int(e.y))) break;
                }
            cB.blk_best[((size_t) blockIdx.x * MAXT + warp) * TOPK_MAX + lane] = make_float2(v, __int_as_float(id));
        }
    } else {
        float * wv = m + 16;                                    // [NW][MAXT]
        int * wi = reinterpret_cast<int *>(m + 16 + NW * MAXT); // [NW][MAXT]
        if (lane == 0) {
#pragma unroll
            for (int t = 0; t < TL; t++) {
                wv[warp * MAXT + t] = bv[t];
                wi[warp * MAXT + t] = bi[t];
            }
        }
        __syncthreads();
        if (tid < TL) {
            float b = -INFINITY;
            int ix = 0;
            for (int w = 0; w < NW; w++) {
                const float v = wv[w * MAXT + tid];
                const int id = wi[w * MAXT + tid];
                if (v > b || (v == b && id < ix)) { b = v; ix = id; }
            }
            cB.blk_best[blockIdx.x * MAXT + tid] = make_float2(b, __int_as_float(ix));
        }
    }
    __threadfence();
    __syncthreads();
    int * s_last = reinterpret_cast<int *>(m + 200);
    if (tid == 0) s_last[0] = (atomicAdd(cB.cnt, 1u) == gridDim.x - 1);
    __syncthreads();
    if (!s_last[0]) return false;
    __threadfence();
    if constexpr (TOPK) {
        if (warp < TL) {
            float v = -INFINITY;
            int id = -1;
            for (int k = 0; k < (int) gridDim.x; k++)
                for (int j = 0; j < 32; j++) {
                    const float2 e = __ldcg(&cB.blk_best[((size_t) k * MAXT + warp) * TOPK_MAX + j]);
                    if (!topk_insert(v, id, e.x, __float_as_int(e.y))) break;
                }
            cB.cand[warp * TOPK_MAX + lane] = make_float2(v, __int_as_float(id));
            const int best = __shfl_sync(0xffffffffu, id, 0);
            if (lane == 0) s_tok[warp] = best;
        }
    } else if (tid < TL) {
        float b = -INFINITY;
        int ix = 0;
        for (int k = 0; k < (int) gridDim.x; k++) {
            const float2 e = __ldcg(&cB.blk_best[k * MAXT + tid]);
            const int id = __float_as_int(e.y);
            if (e.x > b || (e.x == b && id < ix)) { b = e.x; ix = id; }
        }
        s_tok[tid] = (draft && cW.draft_ids) ? cW.draft_ids[ix] : ix;
    }
    if (tid == 0) *cB.cnt = 0;
    __syncthreads();
    return true;
}

// ---- sampling (single thread) ----
// counter-based uniform random number in [0, 1)
__device__ __forceinline__ float urand(unsigned long long seed, unsigned long long ctr) {
    unsigned long long z = seed + (ctr + 1) * 0x9E3779B97F4A7C15ull;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    z ^= z >> 31;
    return (float) (z >> 40) * (1.0f / 16777216.0f);
}

// sampling distribution over one row's candidates (descending), llama.cpp order: top_k, top_p, min_p,
// temperature; returns the number kept, p[0..K) normalised
__device__ int cand_probs(const float2 * c, const Launch & P, float (&p)[TOPK_MAX]) {
    int K = P.top_k > 0 ? min(P.top_k, TOPK_MAX) : TOPK_MAX;
    while (K > 1 && !(c[K - 1].x > -INFINITY)) K--;
    const float mx = c[0].x;
    float s = 0.f;
    for (int j = 0; j < K; j++) s += (p[j] = expf(c[j].x - mx));
    if (P.top_p < 1.0f) {
        float cum = 0.f;
        for (int j = 0; j < K; j++) {
            cum += p[j] / s;
            if (cum >= P.top_p) { K = j + 1; break; }
        }
    }
    if (P.min_p > 0.0f) {
        int k = 1;
        while (k < K && p[k] >= P.min_p * p[0]) k++;
        K = k;
    }
    s = 0.f;
    for (int j = 0; j < K; j++) s += (p[j] = expf((c[j].x - mx) / P.temp));
    for (int j = 0; j < K; j++) p[j] /= s;
    return K;
}

// draw from p (optionally without token excl, renormalised)
__device__ int cand_draw(const float2 * c, const float (&p)[TOPK_MAX], int K, int excl, float u) {
    float s = 0.f;
    for (int j = 0; j < K; j++)
        if (__float_as_int(c[j].y) != excl) s += p[j];
    float x = u * s;
    int last = __float_as_int(c[0].y);
    for (int j = 0; j < K; j++) {
        const int id = __float_as_int(c[j].y);
        if (id == excl) continue;
        last = id;
        x -= p[j];
        if (x < 0.f) return id;
    }
    return last;
}

// Sampled targets for the verified rows (speculative sampling with point-mass drafts): draft token d
// at row i is accepted with probability p_i(d); on rejection the row's token is drawn from p_i
// without d (the residual distribution), and the rows after it are dropped.  Rows with no draft to
// verify (the bonus row, the last prompt row) are drawn from p_i.  tg then drives the greedy
// acceptance logic of main_control unchanged.
__device__ void sample_targets(const Launch & P, int T, int tl, int * tg) {
    const Ctl * c = cB.ctl;
    const unsigned long long n = (unsigned long long) c->n;
    float p[TOPK_MAX];
    for (int i = 0; i < tl; i++) {
        const float2 * cd = cB.cand + i * TOPK_MAX;
        const int K = cand_probs(cd, P, p);
        if (P.mode == MODE_DECODE && i < T - 1) {
            const int d = c->tok[i + 1];
            float pd = 0.f;
            for (int j = 0; j < K; j++)
                if (__float_as_int(cd[j].y) == d) pd += p[j];
            if (urand(P.seed, 2 * (n + i)) < pd) {
                tg[i] = d;
                continue;
            }
            tg[i] = cand_draw(cd, p, K, d, urand(P.seed, 2 * (n + i) + 1));
            return;
        }
        tg[i] = cand_draw(cd, p, K, -1, urand(P.seed, 2 * (n + i) + 1));
        return;
    }
}

// ---- main-model control (single thread) ----
__device__ void main_control(int mode, int last, int T, const int * tgt) {
    Ctl * c = cB.ctl;
    const int n = c->n;
    int a, bonus = -1;
    if (mode == MODE_DECODE) {
        a = 0;
        while (a < T - 1 && tgt[a] == c->tok[a + 1]) a++;
        for (int i = 0; i < a; i++) c->out[i] = c->tok[i + 1];
        c->out[a] = tgt[a];
        c->n_out = a + 1;
        c->n_acc = a;
        bonus = tgt[a];
        c->dn_c = n + 1;
    } else {
        a = T - 1;
        c->dn_c = n + T;
        c->n_acc = 0;
        if (last) {
            bonus = tgt[0];
            c->out[0] = bonus;
            c->n_out = 1;
        } else {
            c->n_out = 0;
        }
    }
    const int m = c->mtp_valid;
    const int P = bonus >= 0 ? n + a + 1 : n + T - 1;
    int R = 0;
    for (int p = m; p <= P && R < 8; p++, R++) {
        c->mtp_pos[R] = p;
        c->mtp_tok[R] = (p >= n && p <= n + a) ? c->tok[p - n] : bonus;
    }
    c->mtp_R = R;
    c->mtp_P = P;
    c->mtp_logits = bonus >= 0;
    c->mtp_valid = P + 1;
    if (bonus >= 0) {
        c->n = P;
        c->tok[0] = bonus;
    }
}

template <int T>
__device__ void main_forward(const Launch & P, uint8_t * /*sm*/, unsigned & epoch) {
    uint8_t * const sm = smem;
    volatile Ctl * vc = cB.ctl;
    const int n = vc->n;
    const int dn_c = vc->dn_c;
    const int n_commit = P.mode == MODE_DECODE ? 1 : T;

    embed_prologue<T>();
    gsync(epoch, PH_EMBED);
    int dl = 0;
    for (int L = 0; L < NL; L++) {
        const LayerW & W = cW.L[L];
        const float * wnext = (L + 1 < NL) ? cW.L[L + 1].attn_norm : cW.onorm;
        if (W.attn) {
            attn_proj<T>(W, L / 4, n, sm);
            gsync(epoch, PH_ATTN_PROJ);
            attn_core<T>(W, L / 4, n, sm);
            gsync(epoch, PH_ATTN_CORE);
            attn_combine<T>(n);
            gsync(epoch, PH_ATTN_COMB);
            resid32<T, 6, T_Q4K>(W.o, cB.qo, cB.x, true, W.post_norm, sm);
            gsync(epoch, PH_RESID_O);
        } else {
            dn_proj<T>(W, dl, n, dn_c, sm);
            gsync(epoch, PH_DN_PROJ);
            dn_core<T>(W, dl, n, dn_c, n_commit, sm);
            gsync(epoch, PH_DN_CORE);
            resid32<T, 6, T_Q5K>(W.sout, cB.qo, cB.x, true, W.post_norm, sm);
            gsync(epoch, PH_RESID_SOUT);
            dl++;
        }
        ffn_gu<T>(W, sm);
        gsync(epoch, PH_FFN_GU);
        if (W.fd.type == T_Q6K) resid32<T, 17, T_Q6K>(W.fd, cB.qa, cB.x, true, wnext, sm);
        else resid32<T, 17, T_Q4K>(W.fd, cB.qa, cB.x, true, wnext, sm);
        gsync(epoch, PH_FFN_DOWN);
    }
    write_h_rows<T>(cB.x, cB.qn, cW.onorm, cB.h_hist, n, true, sm);

    int * s_tok = reinterpret_cast<int *>(misc(sm) + 400);
    bool am_last;
    int tl = 0;
    const bool sampling = P.temp > 0.0f && !P.score;
    if (P.mode == MODE_DECODE || P.score) {
        if (sampling) am_last = lm_head<T, true>(0, false, sm, s_tok);
        else if (P.score == 2) am_last = lm_head<T, false, true>(0, false, sm, s_tok);
        else am_last = lm_head<T>(0, false, sm, s_tok);
        tl = T;
    } else if (P.last) {
        am_last = sampling ? lm_head<1, true>(T - 1, false, sm, s_tok) : lm_head<1>(T - 1, false, sm, s_tok);
        tl = 1;
    } else {
        am_last = blockIdx.x == 0;
    }
    if (am_last && threadIdx.x == 0) {
        int tg[MAXT];
        for (int i = 0; i < tl; i++) tg[i] = s_tok[i];
        if (sampling) sample_targets(P, T, tl, tg);
        for (int i = 0; i < tl; i++) cB.ctl->tgt[i] = tg[i];
        main_control(P.mode, P.last, T, P.score ? s_tok + T - 1 : tg);
        __threadfence();
    }
    gsync(epoch, PH_LMHEAD);
}

// one MTP pass over R consecutive positions; rows read from ctl (catch-up) or a draft row.
template <int R>
__device__ __noinline__ void mtp_pass(int row0, bool draft_row, int draft_idx, bool logits, uint8_t * /*sm*/, unsigned & epoch) {
    uint8_t * const sm = smem;
    volatile Ctl * vc = cB.ctl;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    int pos0;
    if (draft_row) pos0 = vc->mtp_P + draft_idx;
    else pos0 = vc->mtp_pos[row0];

    // M0: concat(enorm(embed(tok)), hnorm(h_prev)) -> qe
    if (blockIdx.x < R) {
        const int r = blockIdx.x;
        const int pos = pos0 + r;
        const int tok = draft_row ? vc->tok[draft_idx] : vc->mtp_tok[row0 + r];
        const float * hsrc = draft_row ? cB.hm : (pos >= 1 ? cB.h_hist + (size_t) ((pos - 1) % H_RING) * D : nullptr);
        float * se = reinterpret_cast<float *>(sm);
        float * sh = se + D;
        float * red = sh + D;
        const uint8_t * erow = cW.embd + (size_t) tok * (D / 256 * 144);
        float ae = 0.f, ah = 0.f;
        for (int e = tid; e < D; e += NT) {
            const float v = dequant_q4k(erow, e);
            const float hv = hsrc ? __ldcg(hsrc + e) : 0.f;
            se[e] = v;
            sh[e] = hv;
            ae += v * v;
            ah += hv * hv;
        }
        ae = bsum(ae, red);
        ah = bsum(ah, red + 16);
        const float rse = rsqrtf(ae / (float) D + EPS), rsh = rsqrtf(ah / (float) D + EPS);
        for (int j = warp; j < 2 * U_D; j += NW) {
            const int e = 32 * (j % U_D) + lane;
            const float v = j < U_D ? se[e] * rse * cW.enorm[e] : sh[e] * rsh * cW.hnorm[e];
            q32_store(cB.qe, r, j, v, 0.f, false);
        }
    }
    gsync(epoch, PH_MTP_M0);
    const LayerW & W = cW.L[NL];
    resid32<R, 10, T_Q8_0>(cW.eh, cB.qe, cB.xm, false, W.attn_norm, sm);
    gsync(epoch, PH_MTP_EH);
    attn_proj<R>(W, 16, pos0, sm);
    gsync(epoch, PH_MTP_ATTN_PROJ);
    attn_core<R>(W, 16, pos0, sm);
    gsync(epoch, PH_MTP_ATTN_CORE);
    attn_combine<R>(pos0);
    gsync(epoch, PH_MTP_ATTN_COMB);
    resid32<R, 6, T_Q4K>(W.o, cB.qo, cB.xm, true, W.post_norm, sm);
    gsync(epoch, PH_MTP_RESID_O);
    ffn_gu<R>(W, sm);
    gsync(epoch, PH_MTP_FFN_GU);
    if (W.fd.type == T_Q6K) resid32<R, 17, T_Q6K>(W.fd, cB.qa, cB.xm, true, cW.shnorm, sm);
    else resid32<R, 17, T_Q4K>(W.fd, cB.qa, cB.xm, true, cW.shnorm, sm);
    gsync(epoch, PH_MTP_FFN_DOWN);
    if (logits) {
        write_h_rows<R>(cB.xm, cB.qn, cW.shnorm, cB.hm, 0, false, sm);
        int * s_tok = reinterpret_cast<int *>(misc(sm) + 400);
        if (lm_head<1>(R - 1, true, sm, s_tok) && threadIdx.x == 0) {
            cB.ctl->tok[1 + (draft_row ? draft_idx : 0)] = s_tok[0];
            __threadfence();
        }
        gsync(epoch, PH_MTP_LMHEAD);
    }
}

template <int T>
__device__ void run_main(const Launch & P, uint8_t * /*sm*/, unsigned & epoch) {
    uint8_t * const sm = smem;
    main_forward<T>(P, sm, epoch);
}

template <int T>
__device__ void bench_phase(int which, int layer, uint8_t * /*sm*/, unsigned & epoch) {
    uint8_t * const sm = smem;
    for (int it = 0; it < 4; it++) {
        const int L = layer + 4 * it; // different layers: weights are not L2-resident, code is
        const LayerW & W = cW.L[L];
        if (which == 0) ffn_gu<T>(W, sm);
        else if (which == 1) {
            if (W.fd.type == T_Q6K) resid32<T, 17, T_Q6K>(W.fd, cB.qa, cB.x, true, W.attn_norm, sm);
            else resid32<T, 17, T_Q4K>(W.fd, cB.qa, cB.x, true, W.attn_norm, sm);
        } else if (which == 2) {
            if (!W.attn) dn_proj<T>(W, 0, 0, 0, sm);
        }
        gsync(epoch, PH_FFN_GU);
    }
}

__global__ void __launch_bounds__(NT, 1) mk_cycle(Launch P) {
    uint8_t * const sm = smem;
    unsigned epoch = 0;
    if (P.mode == MODE_BENCH) {
        switch (P.T) {
            case 1: bench_phase<1>(P.last, P.ndraft, sm, epoch); break;
            case 4: bench_phase<4>(P.last, P.ndraft, sm, epoch); break;
        }
        return;
    }
    if (cB.prof && blockIdx.x == 0 && threadIdx.x == 0) {
        cB.prof[0] = gtimer() << 8;
        s_bar_k = 0;
    }
    switch (P.T) {
        case 1: run_main<1>(P, sm, epoch); break;
        case 2: run_main<2>(P, sm, epoch); break;
        case 3: run_main<3>(P, sm, epoch); break;
        case 4: run_main<4>(P, sm, epoch); break;
        case 5: run_main<5>(P, sm, epoch); break;
    }
    if (P.ndraft == 0 || P.score) return;
    volatile Ctl * vc = cB.ctl;
    const int Rtot = vc->mtp_R;
    const bool lg = vc->mtp_logits;
    int done = 0;
    while (done < Rtot) {
        const int R = min(MAXT, Rtot - done);
        const bool last_pass = done + R == Rtot;
        const bool want = lg && last_pass && P.ndraft > 0;
        switch (R) {
            case 1: mtp_pass<1>(done, false, 0, want, sm, epoch); break;
            case 2: mtp_pass<2>(done, false, 0, want, sm, epoch); break;
            case 3: mtp_pass<3>(done, false, 0, want, sm, epoch); break;
            case 4: mtp_pass<4>(done, false, 0, want, sm, epoch); break;
            case 5: mtp_pass<5>(done, false, 0, want, sm, epoch); break;
        }
        done += R;
    }
    if (lg) {
        for (int i = 1; i < P.ndraft; i++) mtp_pass<1>(0, true, i, true, sm, epoch);
    }
}

__global__ void gather_rows(const uint8_t * src, uint8_t * dst, const __half * dsrc, __half * ddst, const int * ids,
                            int rb, int nd) {
    const int r = blockIdx.x, id = ids[r];
    for (int i = threadIdx.x; i < rb / 16; i += blockDim.x)
        reinterpret_cast<int4 *>(dst + (size_t) r * rb)[i] = reinterpret_cast<const int4 *>(src + (size_t) id * rb)[i];
    for (int i = threadIdx.x; i < nd; i += blockDim.x) ddst[(size_t) r * nd + i] = dsrc[(size_t) id * nd + i];
}

#include "prefill.cuh"

// ------------------------------------------------------------------------------------------
// host side: loading / repacking
// ------------------------------------------------------------------------------------------

uint16_t f2h_bits(float f) {
    __half h = __float2half(f);
    return *reinterpret_cast<uint16_t *>(&h);
}

template <class F>
void parallel_rows(int64_t n, F f) {
    int nt = std::max(1u, std::thread::hardware_concurrency());
    std::vector<std::thread> th;
    for (int i = 0; i < nt; i++) {
        th.emplace_back([&, i] {
            for (int64_t r = i; r < n; r += nt) f(r);
        });
    }
    for (auto & t : th) t.join();
}

} // namespace

struct MKEngine::Impl {
    gguf::File file;
    ModelW hw{};
    Bufs hb{};
    std::vector<void *> allocs;
    int ctx;
    int nsm = 0;
    float smp_temp = 0.f, smp_top_p = 1.f, smp_min_p = 0.f; // sampling (temp 0: greedy)
    int smp_top_k = TOPK_MAX;
    unsigned long long smp_seed = 0;
    int ndraft;          // MTP drafts the next launch produces
    int nd_avail;        // drafts produced by the previous launch (verified by the next decode cycle)
    bool all_acc = true; // the last decode cycle accepted all of its drafts
    std::vector<int> seq; // the sequence so far: prompt(s) and emitted tokens (position = index)
    // checkpoint of the state right after the last prefill (recurrent state, conv ring, h ring, ctl):
    // lets a caller rewind to the end of the last prompt (KV rows before it are never rewritten)
    float *ck_dstate = nullptr, *ck_conv = nullptr, *ck_hhist = nullptr;
    Ctl ck_ctl;
    int ck_pos = -1;
    void ck_save() {
        const size_t ns = (size_t) 48 * NVH * 128 * 128, nc = (size_t) 48 * CONV_RING * QKV, nh = (size_t) H_RING * D;
        if (!ck_dstate) {
            ck_dstate = dalloc<float>(ns, false);
            ck_conv = dalloc<float>(nc, false);
            ck_hhist = dalloc<float>(nh, false);
        }
        CK(cudaMemcpy(ck_dstate, hb.dstate, ns * 4, cudaMemcpyDeviceToDevice));
        CK(cudaMemcpy(ck_conv, hb.conv, nc * 4, cudaMemcpyDeviceToDevice));
        CK(cudaMemcpy(ck_hhist, hb.h_hist, nh * 4, cudaMemcpyDeviceToDevice));
        CK(cudaMemcpy(&ck_ctl, hb.ctl, sizeof(Ctl), cudaMemcpyDeviceToHost));
        ck_pos = ck_ctl.dn_c;
    }
    void ck_restore() {
        const size_t ns = (size_t) 48 * NVH * 128 * 128, nc = (size_t) 48 * CONV_RING * QKV, nh = (size_t) H_RING * D;
        CK(cudaMemcpy(hb.dstate, ck_dstate, ns * 4, cudaMemcpyDeviceToDevice));
        CK(cudaMemcpy(hb.conv, ck_conv, nc * 4, cudaMemcpyDeviceToDevice));
        CK(cudaMemcpy(hb.h_hist, ck_hhist, nh * 4, cudaMemcpyDeviceToDevice));
        hctl = ck_ctl;
        hctl.mtp_valid = std::min(hctl.mtp_valid, ck_pos);
        CK(cudaMemcpy(hb.ctl, &hctl, sizeof(Ctl), cudaMemcpyHostToDevice));
        seq.resize(ck_pos);
    }
    // draft vocabulary: ids of head_d rows (base list + tokens added from the sequence)
    std::vector<int> dv_ids;
    std::vector<uint8_t> dv_in;
    int dv_base = 0;

    // Adds the tokens not yet in the draft head's vocabulary (copies their lm_head rows).
    void dv_add(const int * toks, size_t n) {
        if (dv_in.empty()) return; // full-vocabulary draft head
        const int n0 = (int) dv_ids.size();
        for (size_t i = 0; i < n && (int) dv_ids.size() < dv_base + DV_EXTRA - 1; i++)
            if (!dv_in[toks[i]]) {
                dv_in[toks[i]] = 1;
                dv_ids.push_back(toks[i]);
            }
        if ((int) dv_ids.size() == n0) return;
        if (dv_ids.size() & 1) dv_ids.push_back(dv_ids.back()); // the draft lm_head takes rows in pairs
        const int k = (int) dv_ids.size() - n0;
        Mat & m = hw.head_d;
        CK(cudaMemcpy(const_cast<int *>(hw.draft_ids) + n0, dv_ids.data() + n0, k * 4, cudaMemcpyHostToDevice));
        gather_rows<<<k, 256>>>(hw.head.w, const_cast<uint8_t *>(m.w) + (size_t) n0 * m.rb, hw.head.d,
                                m.d ? const_cast<__half *>(m.d) + (size_t) n0 * m.nd : nullptr, hw.draft_ids + n0, m.rb, m.nd);
        m.N = (int) dv_ids.size();
        CK(cudaMemcpyToSymbol(cW, &m.N, sizeof(int), offsetof(ModelW, head_d) + offsetof(Mat, N)));
    }
    void dv_reset() {
        if (dv_in.empty() || (int) dv_ids.size() == dv_base) return;
        for (size_t i = dv_base; i < dv_ids.size(); i++) dv_in[dv_ids[i]] = 0;
        for (int i = 0; i < dv_base; i++) dv_in[dv_ids[i]] = 1; // a padding duplicate may have been cleared
        dv_ids.resize(dv_base);
        hw.head_d.N = dv_base;
        CK(cudaMemcpyToSymbol(cW, &hw.head_d.N, sizeof(int), offsetof(ModelW, head_d) + offsetof(Mat, N)));
    }
    Ctl hctl{};
    size_t wbytes = 0;
    bool profiling = false;
    std::vector<double> ph_ns = std::vector<double>(PH_COUNT, 0.0);
    std::vector<long> ph_n = std::vector<long>(PH_COUNT, 0);
    double launch_ns = 0;
    long launches = 0;
    // min over launches of each barrier-to-barrier interval, keyed by (T, barrier index)
    std::vector<std::vector<double>> best_ns = std::vector<std::vector<double>>(MAXT + 1);
    std::vector<std::vector<int>> best_tag = std::vector<std::vector<int>>(MAXT + 1);
    std::vector<std::vector<double>> best_fill = std::vector<std::vector<double>>(MAXT + 1);
    std::vector<double> arr_first = std::vector<double>(PH_COUNT, 0.0), arr_med = std::vector<double>(PH_COUNT, 0.0),
                        arr_last = std::vector<double>(PH_COUNT, 0.0);
    std::vector<long> arr_n = std::vector<long>(PH_COUNT, 0);

    template <class T> T * dalloc(size_t n, bool zero = true) {
        void * p;
        CK(cudaMalloc(&p, n * sizeof(T) + 16));
        if (zero) CK(cudaMemset(p, 0, n * sizeof(T) + 16));
        allocs.push_back(p);
        return (T *) p;
    }

    // A tensor of the file in the quant type `to` its kernel is built for. A plain
    // `llama-quantize ... Q4_K_M` stores four roles as Q4_K that the kernels read as something
    // else (ssm_out Q5_K, ssm_alpha / ssm_beta F32, nextn.eh_proj Q8_0): those are re-encoded here,
    // dequantized and quantized again by ggml. 145 small tensors, a few seconds at load.
    struct Src {
        uint32_t type = 0;
        std::vector<uint64_t> ne;
        const uint8_t * data = nullptr;
        uint64_t nbytes = 0;
        std::vector<uint8_t> own; // the re-encoded copy, when there is one
    };
    int n_conv = 0;
    double conv_s = 0;

    Src want(const std::string & name, int to) {
        const auto & t = file.get(name);
        Src s;
        s.type = t.type;
        s.ne = t.ne;
        s.data = t.data;
        s.nbytes = t.nbytes;
        if ((int) t.type == to || t.type != gguf::Q4_K) return s; // any other mix is refused by need() below
        const int64_t K = (int64_t) t.ne[0];
        int64_t rows = 1;
        for (size_t d = 1; d < t.ne.size(); d++) rows *= (int64_t) t.ne[d];
        const size_t rs_from = ggml_row_size(GGML_TYPE_Q4_K, K), rs_to = ggml_row_size((ggml_type) to, K);
        const auto t0 = std::chrono::steady_clock::now();
        s.own.resize(rs_to * (size_t) rows);
        if (to != T_F32) ggml_quantize_init((ggml_type) to);
        const auto * tr = ggml_get_type_traits(GGML_TYPE_Q4_K);
        parallel_rows(rows, [&](int64_t r) {
            thread_local std::vector<float> f;
            f.resize((size_t) K);
            tr->to_float(t.data + (size_t) r * rs_from, f.data(), K);
            if (to == T_F32) memcpy(s.own.data() + (size_t) r * rs_to, f.data(), (size_t) K * 4);
            else ggml_quantize_chunk((ggml_type) to, f.data(), s.own.data() + (size_t) r * rs_to, 0, 1, K, nullptr);
        });
        s.type = (uint32_t) to;
        s.data = s.own.data();
        s.nbytes = s.own.size();
        n_conv++;
        conv_s += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        return s;
    }

    const float * up_f32(const std::string & name, size_t expect) {
        const auto & t = file.get(name);
        if (t.type != gguf::F32) throw std::runtime_error(name + " not f32");
        size_t n = t.nbytes / 4;
        if (expect && n != expect) throw std::runtime_error(name + " bad size");
        float * d = dalloc<float>(n, false);
        CK(cudaMemcpy(d, t.data, t.nbytes, cudaMemcpyHostToDevice));
        wbytes += t.nbytes;
        return d;
    }

    Mat up_mat(const std::string & name, int K, int N, int to = -1) {
        const Src t = want(name, to < 0 ? (int) file.get(name).type : to);
        if ((int) t.ne[0] != K || (int) t.ne[1] != N) throw std::runtime_error(name + " bad shape");
        Mat m{};
        m.K = K;
        m.N = N;
        m.type = (int) t.type;
        const int nb = K / 256;
        if (t.type == gguf::Q4_K || t.type == gguf::Q5_K) {
            m.rb = (int) gguf::type_row_bytes(t.type, K);
            uint8_t * d = dalloc<uint8_t>(t.nbytes, false);
            CK(cudaMemcpy(d, t.data, t.nbytes, cudaMemcpyHostToDevice));
            m.w = d;
            wbytes += t.nbytes;
        } else if (t.type == gguf::Q6_K) {
            m.rb = nb * 224;
            m.nd = 0;
            std::vector<uint8_t> q((size_t) N * m.rb, 0);
            const uint8_t * src = t.data;
            parallel_rows(N, [&](int64_t r) {
                for (int b = 0; b < nb; b++) {
                    const uint8_t * blk = src + ((size_t) r * nb + b) * 210;
                    const uint8_t * ql = blk;
                    const uint8_t * qh = blk + 128;
                    const int8_t * sc = (const int8_t *) (blk + 192);
                    uint16_t dbits;
                    memcpy(&dbits, blk + 208, 2);
                    uint8_t vals[256];
                    for (int nn = 0; nn < 2; nn++)
                        for (int l = 0; l < 32; l++) {
                            const uint8_t * L = ql + 64 * nn;
                            const uint8_t * H = qh + 32 * nn;
                            vals[128 * nn + l + 0] = (L[l] & 0xF) | (((H[l] >> 0) & 3) << 4);
                            vals[128 * nn + l + 32] = (L[l + 32] & 0xF) | (((H[l] >> 2) & 3) << 4);
                            vals[128 * nn + l + 64] = (L[l] >> 4) | (((H[l] >> 4) & 3) << 4);
                            vals[128 * nn + l + 96] = (L[l + 32] >> 4) | (((H[l] >> 6) & 3) << 4);
                        }
                    uint8_t * dst = q.data() + (size_t) r * m.rb + b * 224;
                    for (int p = 0; p < 4; p++)
                        for (int i = 0; i < 32; i++)
                            dst[32 * p + i] = (vals[64 * p + i] & 0xF) | ((vals[64 * p + 32 + i] & 0xF) << 4);
                    for (int p = 0; p < 4; p++)
                        for (int hh = 0; hh < 2; hh++) {
                            uint32_t hA = 0, hB = 0;
                            for (int i = 0; i < 16; i++) {
                                const int sh = 8 * (i % 4) + 2 * (i / 4);
                                hA |= (uint32_t) (vals[64 * p + 16 * hh + i] >> 4) << sh;
                                hB |= (uint32_t) (vals[64 * p + 32 + 16 * hh + i] >> 4) << sh;
                            }
                            memcpy(dst + 128 + 8 * (2 * p + hh), &hA, 4);
                            memcpy(dst + 128 + 8 * (2 * p + hh) + 4, &hB, 4);
                        }
                    for (int p = 0; p < 4; p++) {
                        memcpy(dst + 192 + 8 * p, sc + 4 * p, 4);
                        memcpy(dst + 192 + 8 * p + 4, &dbits, 2);
                    }
                }
            });
            uint8_t * dq = dalloc<uint8_t>(q.size(), false);
            CK(cudaMemcpy(dq, q.data(), q.size(), cudaMemcpyHostToDevice));
            m.w = dq;
            m.d = nullptr;
            wbytes += q.size();
        } else if (t.type == gguf::Q8_0) {
            const int nb32 = K / 32;
            m.rb = K;
            m.nd = nb32;
            std::vector<uint8_t> q((size_t) N * K);
            std::vector<uint16_t> ds((size_t) N * nb32);
            const uint8_t * src = t.data;
            parallel_rows(N, [&](int64_t r) {
                const uint8_t * row = src + (size_t) r * nb32 * 34;
                std::vector<int8_t> vals(K);
                for (int j = 0; j < nb32; j++) {
                    memcpy(&ds[(size_t) r * nb32 + j], row + 34 * j, 2);
                    memcpy(&vals[32 * j], row + 34 * j + 2, 32);
                }
                uint8_t * dst = q.data() + (size_t) r * K;
                for (int u = 0; u < K / 32; u++) {
                    const int b = u >> 3, p = (u >> 1) & 3, hh = u & 1;
                    memcpy(dst + 32 * u, &vals[256 * b + 64 * p + 16 * hh], 16);
                    memcpy(dst + 32 * u + 16, &vals[256 * b + 64 * p + 32 + 16 * hh], 16);
                }
            });
            uint8_t * dq = dalloc<uint8_t>(q.size(), false);
            CK(cudaMemcpy(dq, q.data(), q.size(), cudaMemcpyHostToDevice));
            uint16_t * dd = dalloc<uint16_t>(ds.size(), false);
            CK(cudaMemcpy(dd, ds.data(), ds.size() * 2, cudaMemcpyHostToDevice));
            m.w = dq;
            m.d = (const __half *) dd;
            wbytes += q.size() + ds.size() * 2;
        } else {
            throw std::runtime_error(name + ": unsupported quant type " + std::to_string(t.type));
        }
        return m;
    }

    QBuf qbuf(int U) {
        QBuf q;
        q.U = U;
        q.xa = dalloc<int8_t>((size_t) MAXT * U * 16);
        q.xb = dalloc<int8_t>((size_t) MAXT * U * 16);
        q.meta = dalloc<uint2>((size_t) MAXT * U);
        q.ssq = dalloc<float>((size_t) MAXT * U);
        return q;
    }

    int n_draft_max;
    int kv_mode; // KV cache format: 0 fp16, 1 q8_0, -1 auto (q8_0 when fp16 does not fit)
    Impl(const std::string & path, int ctx_, int ndraft_, const std::string & draft_vocab, int kv_mode_)
        : file(path), ctx(ctx_), ndraft(ndraft_), nd_avail(ndraft_), n_draft_max(ndraft_), kv_mode(kv_mode_) {
        if (file.strs["general.architecture"] != "qwen35") throw std::runtime_error("expected qwen35 architecture");
        if (file.ints["qwen35.block_count"] != NLA || file.ints["qwen35.embedding_length"] != D)
            throw std::runtime_error("unexpected model dimensions (this kernel is specialised for the 27B)");
        cudaDeviceProp prop;
        CK(cudaGetDeviceProperties(&prop, 0));
        nsm = prop.multiProcessorCount;

        auto blk = [](int i, const char * s) { return "blk." + std::to_string(i) + "." + s; };
        // The decode kernels are compiled for one quant type per tensor role (resid32 / gemv_pairs
        // take it as a template argument), so a GGUF quantized with a different per-tensor recipe
        // loads without complaint and then reads past its weights on the first token. Check every
        // role against the type its kernel was built for.
        auto tname = [](int t) -> std::string {
            switch (t) {
                case gguf::F32: return "F32";
                case gguf::F16: return "F16";
                case gguf::Q8_0: return "Q8_0";
                case gguf::Q4_K: return "Q4_K";
                case gguf::Q5_K: return "Q5_K";
                case gguf::Q6_K: return "Q6_K";
                default: return "type " + std::to_string(t);
            }
        };
        auto need = [&](const std::string & name, int got, std::initializer_list<int> ok) {
            for (int t : ok)
                if (got == t) return;
            std::string want;
            for (int t : ok) want += (want.empty() ? "" : " or ") + tname(t);
            throw std::runtime_error(name + " is " + tname(got) + ", the kernel needs " + want +
                                     " (see README: the GGUF)");
        };
        for (int i = 0; i < NLA; i++) {
            LayerW & L = hw.L[i];
            const bool mtp = i == NL;
            L.attn = mtp || ((i + 1) % 4 == 0);
            L.attn_norm = up_f32(blk(i, "attn_norm.weight"), D);
            L.post_norm = up_f32(blk(i, "post_attention_norm.weight"), D);
            if (L.attn) {
                L.q = up_mat(blk(i, "attn_q.weight"), D, QG);
                L.k = up_mat(blk(i, "attn_k.weight"), D, KVD);
                L.v = up_mat(blk(i, "attn_v.weight"), D, KVD);
                L.o = up_mat(blk(i, "attn_output.weight"), VD, D);
                need(blk(i, "attn_q.weight"), L.q.type, {T_Q4K});
                need(blk(i, "attn_k.weight"), L.k.type, {T_Q4K});
                need(blk(i, "attn_v.weight"), L.v.type, {T_Q4K, T_Q6K});
                need(blk(i, "attn_output.weight"), L.o.type, {T_Q4K});
                L.qn = up_f32(blk(i, "attn_q_norm.weight"), HD);
                L.kn = up_f32(blk(i, "attn_k_norm.weight"), HD);
            } else {
                L.qkv = up_mat(blk(i, "attn_qkv.weight"), D, QKV);
                L.gate = up_mat(blk(i, "attn_gate.weight"), D, VD);
                L.sout = up_mat(blk(i, "ssm_out.weight"), VD, D, T_Q5K);
                need(blk(i, "attn_qkv.weight"), L.qkv.type, {T_Q4K, T_Q6K});
                need(blk(i, "attn_gate.weight"), L.gate.type, {T_Q4K});
                need(blk(i, "ssm_out.weight"), L.sout.type, {T_Q5K});
                L.conv = up_f32(blk(i, "ssm_conv1d.weight"), 4 * QKV);
                L.dt = up_f32(blk(i, "ssm_dt.bias"), NVH);
                L.A = up_f32(blk(i, "ssm_a"), NVH);
                L.snorm = up_f32(blk(i, "ssm_norm.weight"), HK);
                // alpha/beta projections in fp32 with the attn_norm weight folded in
                const Src ta = want(blk(i, "ssm_alpha.weight"), T_F32);
                const Src tb = want(blk(i, "ssm_beta.weight"), T_F32);
                const auto & tn = file.get(blk(i, "attn_norm.weight"));
                // read below as raw floats, so anything but F32 would be reinterpreted silently
                need(blk(i, "ssm_alpha.weight"), (int) ta.type, {T_F32});
                need(blk(i, "ssm_beta.weight"), (int) tb.type, {T_F32});
                if (ta.ne[0] != D || ta.ne[1] != NVH || tb.ne[0] != D || tb.ne[1] != NVH)
                    throw std::runtime_error(blk(i, "ssm_alpha/beta.weight") + " bad shape");
                std::vector<float> ab((size_t) 2 * NVH * D);
                const float * fa = (const float *) ta.data;
                const float * fb = (const float *) tb.data;
                const float * fn = (const float *) tn.data;
                for (int h = 0; h < NVH; h++)
                    for (int e = 0; e < D; e++) {
                        ab[(size_t) h * D + e] = fa[(size_t) h * D + e] * fn[e];
                        ab[(size_t) (NVH + h) * D + e] = fb[(size_t) h * D + e] * fn[e];
                    }
                float * d = dalloc<float>(ab.size(), false);
                CK(cudaMemcpy(d, ab.data(), ab.size() * 4, cudaMemcpyHostToDevice));
                L.ab = d;
                wbytes += ab.size() * 4;
            }
            L.fg = up_mat(blk(i, "ffn_gate.weight"), D, FF);
            L.fu = up_mat(blk(i, "ffn_up.weight"), D, FF);
            L.fd = up_mat(blk(i, "ffn_down.weight"), FF, D);
            if (L.fg.type != T_Q4K || L.fu.type != T_Q4K) throw std::runtime_error("ffn gate/up must be Q4_K");
            need(blk(i, "ffn_down.weight"), L.fd.type, {T_Q4K, T_Q6K});
        }
        hw.eh = up_mat(blk(NL, "nextn.eh_proj.weight"), 2 * D, D, T_Q8_0);
        need(blk(NL, "nextn.eh_proj.weight"), hw.eh.type, {T_Q8_0});
        hw.enorm = up_f32(blk(NL, "nextn.enorm.weight"), D);
        hw.hnorm = up_f32(blk(NL, "nextn.hnorm.weight"), D);
        hw.shnorm = up_f32(blk(NL, "nextn.shared_head_norm.weight"), D);
        hw.onorm = up_f32("output_norm.weight", D);
        hw.head = up_mat("output.weight", D, VOCAB);
        if (hw.head.type != T_Q6K) throw std::runtime_error("output.weight must be Q6_K");
        hw.head_d = hw.head;
        hw.draft_ids = nullptr;
        if (!draft_vocab.empty()) {
            FILE * f = fopen(draft_vocab.c_str(), "rb");
            if (!f) throw std::runtime_error("cannot open draft vocab " + draft_vocab);
            std::vector<int> ids;
            int id;
            while (fread(&id, 4, 1, f) == 1) {
                if (id < 0 || id >= VOCAB) throw std::runtime_error("bad id in draft vocab");
                ids.push_back(id);
            }
            fclose(f);
            if (ids.size() & 1) ids.push_back(ids.back());
            // room for DV_EXTRA more rows: tokens of the current sequence that are not in the list
            // are added on the fly (identifiers of pasted code, see dv_add)
            const int cap = (int) ids.size() + DV_EXTRA;
            int * did = dalloc<int>(cap, false);
            CK(cudaMemcpy(did, ids.data(), ids.size() * 4, cudaMemcpyHostToDevice));
            Mat m = hw.head;
            m.N = (int) ids.size();
            uint8_t * w = dalloc<uint8_t>((size_t) cap * m.rb, false);
            __half * d = m.nd ? dalloc<__half>((size_t) cap * m.nd, false) : nullptr;
            gather_rows<<<m.N, 256>>>(hw.head.w, w, hw.head.d, d, did, m.rb, m.nd);
            CK(cudaDeviceSynchronize());
            m.w = w;
            m.d = d;
            hw.head_d = m;
            hw.draft_ids = did;
            dv_ids = ids;
            dv_base = (int) ids.size();
            dv_in.assign(VOCAB, 0);
            for (int t : ids) dv_in[t] = 1;
            fprintf(stderr, "[mk] MTP draft head restricted to %d tokens (%s)\n", m.N, draft_vocab.c_str());
        }
        {
            const auto & t = file.get("token_embd.weight");
            if (t.type != gguf::Q4_K) throw std::runtime_error("token_embd must be Q4_K");
            uint8_t * d = dalloc<uint8_t>(t.nbytes, false);
            CK(cudaMemcpy(d, t.data, t.nbytes, cudaMemcpyHostToDevice));
            hw.embd = d;
        }
        CK(cudaMemcpyToSymbol(cW, &hw, sizeof(ModelW)));
        if (n_conv)
            fprintf(stderr, "[mk] %d tensors re-encoded in %.0f s: a plain Q4_K_M quant mix, not the one the kernels are built for "
                            "(README: the GGUF)\n", n_conv, conv_s);

        hb.x = dalloc<float>(MAXT * D);
        hb.xm = dalloc<float>(MAXT * D);
        hb.qn = qbuf(U_D);
        hb.qo = qbuf(U_VD);
        hb.qa = qbuf(U_FF);
        hb.qe = qbuf(U_EH);
        hb.z = dalloc<float>(MAXT * VD);
        hb.ab = dalloc<float>(AB_SPLIT * MAXT * 2 * NVH);
        hb.qbuf = dalloc<float>(MAXT * QG);
        hb.kbuf = dalloc<float>(MAXT * KVD);
        hb.vbuf = dalloc<float>(MAXT * KVD);
        hb.conv = dalloc<float>((size_t) 48 * CONV_RING * QKV);
        hb.kvgb = dalloc<float>((size_t) 48 * KV_RING * KVGB);
        hb.dstate = dalloc<float>((size_t) 48 * NVH * 128 * 128);
        {
            const size_t rows = (size_t) N_ATT_SLOTS * NKV * ctx;
            size_t free_b = 0, total_b = 0;
            CK(cudaMemGetInfo(&free_b, &total_b));
            // fp16 needs 2 x rows x HD x 2 bytes; leave ~1.5 GB for the prefill buffers, checkpoint and runtime
            const bool q8 = kv_mode == 1 || (kv_mode < 0 && rows * HD * 4 + (size_t) 1536 * 1024 * 1024 > free_b);
            hb.kvq8 = q8;
            hb.kc = q8 ? (void *) dalloc<int8_t>(rows * HD) : (void *) dalloc<__half>(rows * HD);
            hb.vc = q8 ? (void *) dalloc<int8_t>(rows * HD) : (void *) dalloc<__half>(rows * HD);
            hb.ksc = q8 ? dalloc<__half>(rows * (HD / 32)) : nullptr;
            hb.vsc = q8 ? dalloc<__half>(rows * (HD / 32)) : nullptr;
            fprintf(stderr, "[mk] KV cache %s, %.2f GB for %d positions\n", q8 ? "q8_0" : "fp16",
                    rows * HD * (q8 ? 2.0 + 2.0 * 2 / 32 : 4.0) / 1e9, ctx);
        }
        hb.apart = dalloc<float>((size_t) NKV * MAX_CH * AR * 256);
        hb.aml = dalloc<float2>((size_t) NKV * MAX_CH * AR);
        hb.h_hist = dalloc<float>((size_t) H_RING * D);
        {
            std::vector<float2> rt((size_t) ctx * 32);
            for (int p = 0; p < ctx; p++)
                for (int i = 0; i < 32; i++) {
                    const double th = (double) p * std::pow((double) ROPE_BASE, -2.0 * i / 64.0);
                    rt[(size_t) p * 32 + i] = make_float2((float) std::cos(th), (float) std::sin(th));
                }
            float2 * d = dalloc<float2>(rt.size(), false);
            CK(cudaMemcpy(d, rt.data(), rt.size() * sizeof(float2), cudaMemcpyHostToDevice));
            hb.rope = d;
        }
        hb.hm = dalloc<float>(D);
        hb.blk_best = dalloc<float2>((size_t) nsm * MAXT * TOPK_MAX);
        hb.cand = dalloc<float2>((size_t) MAXT * TOPK_MAX);
        hb.logits = dalloc<float>((size_t) MAXT * VOCAB);
        hb.bar = dalloc<unsigned>(4);
        hb.cnt = dalloc<unsigned>(4);
        hb.ctl = dalloc<Ctl>(1);
        hb.ctx = ctx;
        profiling = getenv("MK_PROFILE") != nullptr;
        hb.prof = profiling ? dalloc<unsigned long long>(4096 + 1024 * 128) : nullptr;
        CK(cudaMemcpyToSymbol(cB, &hb, sizeof(Bufs)));

        CK(cudaFuncSetAttribute(mk_cycle, cudaFuncAttributeMaxDynamicSharedMemorySize, SM_TOTAL));
        int nb = 0;
        CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, mk_cycle, NT, SM_TOTAL));
        if (nb < 1) throw std::runtime_error("megakernel does not fit on an SM");
        fprintf(stderr, "[mk] weights %.2f GB on device, %d SMs, ctx %d, n_draft %d\n", wbytes / 1e9, nsm, ctx, ndraft);
    }

    ~Impl() {
        print_profile();
        for (void * p : allocs) cudaFree(p);
    }

    void launch(int mode, int T, int last, int score = 0) {
        Launch L{mode, T, last, ndraft, score, smp_temp, smp_top_p, smp_min_p, smp_top_k, smp_seed};
        void * args[] = {&L};
        CK(cudaMemsetAsync(hb.bar, 0, 16));
        if (profiling) CK(cudaMemsetAsync(hb.prof, 0, (4096 + 1024 * 128) * 8));
        CK(cudaLaunchCooperativeKernel((void *) mk_cycle, dim3(nsm), dim3(NT), args, SM_TOTAL, 0));
        if (profiling && mode == MODE_DECODE) {
            std::vector<unsigned long long> pr(4096 + 1024 * 128);
            CK(cudaMemcpy(pr.data(), hb.prof, pr.size() * 8, cudaMemcpyDeviceToHost));
            unsigned long long prev = pr[0] >> 8;
            auto & bn = best_ns[T];
            auto & bt = best_tag[T];
            auto & bf = best_fill[T];
            for (int k = 1; k < 4096 && pr[k]; k++) {
                const unsigned long long t = pr[k] >> 8;
                const int tag = (int) (pr[k] & 0xff);
                const double dt = (double) (t - prev);
                ph_ns[tag] += dt;
                ph_n[tag]++;
                launch_ns += dt;
                prev = t;
                const unsigned long long mark = pr[2048 + k] & ((1ull << 56) - 1);
                const double fill = mark ? (double) ((mark - (t - (unsigned long long) dt)) & ((1ull << 56) - 1)) : 0.0;
                if ((int) bn.size() < k) { bn.resize(k, 1e30); bt.resize(k, -1); bf.resize(k, 0.0); }
                if (k < 1024) {
                    std::vector<double> arr;
                    const unsigned long long t0 = t - (unsigned long long) dt;
                    for (int b = 0; b < nsm; b++) {
                        const unsigned long long a = pr[4096 + k * 128 + b];
                        if (a) arr.push_back((double) (((a & ((1ull << 56) - 1)) - t0) & ((1ull << 56) - 1)));
                    }
                    if (arr.size() == (size_t) nsm) {
                        std::sort(arr.begin(), arr.end());
                        arr_first[tag] += arr.front();
                        arr_med[tag] += arr[arr.size() / 2];
                        arr_last[tag] += arr.back();
                        arr_n[tag]++;
                    }
                }
                if (bt[k - 1] != tag) { bt[k - 1] = tag; bn[k - 1] = dt; bf[k - 1] = fill; } // sequence changed
                else if (dt < bn[k - 1]) { bn[k - 1] = dt; bf[k - 1] = fill; }
            }
            launches++;
        }
    }

    void print_profile() {
        if (!profiling || !launches) return;
        static const char * names[PH_COUNT] = {"embed", "attn_proj", "attn_core", "attn_comb", "resid_o", "dn_proj",
                                               "dn_core", "resid_sout", "ffn_gu", "ffn_down", "lm_head", "mtp_m0",
                                               "mtp_eh", "mtp_attn_proj", "mtp_attn_core", "mtp_attn_comb",
                                               "mtp_resid_o", "mtp_ffn_gu", "mtp_ffn_down", "mtp_lm_head"};
        fprintf(stderr, "[mk-profile] %ld decode launches, mean %.3f ms in barriers-to-barrier phases\n", launches,
                launch_ns / launches / 1e6);
        for (int i = 0; i < PH_COUNT; i++) {
            if (!ph_n[i]) continue;
            fprintf(stderr, "  %-16s total/launch %8.3f ms   per call %8.2f us   calls/launch %6.1f\n", names[i],
                    ph_ns[i] / launches / 1e6, ph_ns[i] / ph_n[i] / 1e3, (double) ph_n[i] / launches);
        }
        fprintf(stderr, "[mk-profile] block arrival at barrier (us after phase start; mean over launches): first / median / last\n");
        for (int i = 0; i < PH_COUNT; i++)
            if (arr_n[i])
                fprintf(stderr, "  %-16s %8.2f %8.2f %8.2f\n", names[i], arr_first[i] / arr_n[i] / 1e3, arr_med[i] / arr_n[i] / 1e3,
                        arr_last[i] / arr_n[i] / 1e3);
        for (int T = 1; T <= MAXT; T++) {
            if (best_ns[T].empty()) continue;
            std::vector<double> tot(PH_COUNT, 0.0), fil(PH_COUNT, 0.0);
            std::vector<int> cnt(PH_COUNT, 0);
            double all = 0;
            for (size_t k = 0; k < best_ns[T].size(); k++) {
                if (best_tag[T][k] < 0) continue;
                tot[best_tag[T][k]] += best_ns[T][k];
                fil[best_tag[T][k]] += best_fill[T][k];
                cnt[best_tag[T][k]]++;
                all += best_ns[T][k];
            }
            fprintf(stderr, "[mk-profile] T=%d min-composed launch %.3f ms (interference-free estimate)\n", T, all / 1e6);
            for (int i = 0; i < PH_COUNT; i++)
                if (cnt[i])
                    fprintf(stderr, "  %-16s %8.3f ms  %8.2f us/call (fill %5.2f us)  x%d\n", names[i], tot[i] / 1e6,
                            tot[i] / cnt[i] / 1e3, fil[i] / cnt[i] / 1e3, cnt[i]);
        }
    }

    void bench() {
        const char * names[3] = {"ffn_gu", "ffn_down", "dn_proj"};
        if (profiling) {
            // i-cache test: 4 back-to-back iterations of one phase inside one launch
            for (int which = 0; which < 3; which++) {
                std::vector<double> best(5, 1e30);
                for (int rep = 0; rep < 60; rep++) {
                    Launch L{MODE_BENCH, 4, which, which == 2 ? 0 : 1, 0};
                    void * args[] = {&L};
                    CK(cudaMemsetAsync(hb.bar, 0, 16));
                    CK(cudaMemsetAsync(hb.prof, 0, 4096 * 8));
                    CK(cudaLaunchCooperativeKernel((void *) mk_cycle, dim3(nsm), dim3(NT), args, SM_TOTAL, 0));
                    std::vector<unsigned long long> pr(8);
                    CK(cudaMemcpy(pr.data(), hb.prof, 8 * 8, cudaMemcpyDeviceToHost));
                    for (int k = 1; k <= 4; k++) best[k] = std::min(best[k], (double) ((pr[k] >> 8) - (pr[k - 1] >> 8)));
                }
                fprintf(stderr, "[mk-bench] %-8s T=4 back-to-back iterations: %.1f / %.1f / %.1f / %.1f us\n", names[which],
                        best[1] / 1e3, best[2] / 1e3, best[3] / 1e3, best[4] / 1e3);
            }
            return;
        }
        for (int which = 0; which < 3; which++)
            for (int T : {1, 4}) {
                float best = 1e30f;
                cudaEvent_t a, b;
                cudaEventCreate(&a);
                cudaEventCreate(&b);
                for (int rep = 0; rep < 400; rep++) {
                    cudaEventRecord(a);
                    Launch L{MODE_BENCH, T, which, which == 2 ? 0 : (rep % 16), 0};
                    void * args[] = {&L};
                    CK(cudaMemsetAsync(hb.bar, 0, 16));
                    CK(cudaLaunchCooperativeKernel((void *) mk_cycle, dim3(nsm), dim3(NT), args, SM_TOTAL, 0));
                    cudaEventRecord(b);
                    CK(cudaEventSynchronize(b));
                    float ms;
                    cudaEventElapsedTime(&ms, a, b);
                    best = std::min(best, ms);
                }
                fprintf(stderr, "[mk-bench] %-8s T=%d: %.1f us/launch (1 layer + barrier + launch overhead)\n", names[which], T, best * 1000);
            }
    }

    // ---------------- batched prefill (prefill.cuh) ----------------
    PBufs hp{};
    bool pf_ready = false;

    void pf_init() {
        if (pf_ready) return;
        const int cap = 2048;
        hp.cap = cap;
        hp.x = dalloc<float>((size_t) cap * D);
        hp.xm = dalloc<float>((size_t) cap * D);
        hp.rs = dalloc<float>(cap);
        hp.xh = dalloc<__half>((size_t) cap * FF);
        hp.xh2 = dalloc<__half>((size_t) cap * FF);
        hp.y0 = dalloc<float>((size_t) cap * FF);
        hp.y1 = dalloc<float>((size_t) cap * FF);
        hp.y2 = dalloc<float>((size_t) cap * VD);
        hp.dq = dalloc<__half>((size_t) cap * QKV);
        hp.gb = dalloc<float>((size_t) cap * 2 * NVH);
        hp.ab = dalloc<float>((size_t) 8 * cap * 2 * NVH);
        hp.h = dalloc<float>((size_t) (cap + 1) * D);
        hp.qh = dalloc<__half>((size_t) cap * NH * HD);
        hp.tok = dalloc<int>(cap);
        hp.amax = dalloc<int>(cap);
        CK(cudaMemcpyToSymbol(cP, &hp, sizeof(PBufs)));
        CK(cudaFuncSetAttribute(pf::mmh<pf::TQ4K>, cudaFuncAttributeMaxDynamicSharedMemorySize, pf::hm_smem<pf::TQ4K>()));
        CK(cudaFuncSetAttribute(pf::mmh<pf::TQ5K>, cudaFuncAttributeMaxDynamicSharedMemorySize, pf::hm_smem<pf::TQ5K>()));
        CK(cudaFuncSetAttribute(pf::mmh<pf::TQ6K>, cudaFuncAttributeMaxDynamicSharedMemorySize, pf::hm_smem<pf::TQ6K>()));
        CK(cudaFuncSetAttribute(pf::mmh<pf::TQ80>, cudaFuncAttributeMaxDynamicSharedMemorySize, pf::hm_smem<pf::TQ80>()));
        CK(cudaFuncSetAttribute(pf::mmh<pf::TQ4K, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, pf::hm_smem<pf::TQ4K>()));
        CK(cudaFuncSetAttribute(pk_attn<false>, cudaFuncAttributeMaxDynamicSharedMemorySize, AT_SMEM));
        CK(cudaFuncSetAttribute(pk_attn<true>, cudaFuncAttributeMaxDynamicSharedMemorySize, AT_SMEM));
        pf_ready = true;
    }

    // y[t][row] (+)= W x_t for the first ntok fp16 rows of xh (token offset t0)
    void gemm(const Mat & m, int ntok, float * y, int ldy, int mode, int t0 = 0, const __half * xin = nullptr) {
        const pf::WMat W{m.w, m.d, m.type, m.K, m.N, m.rb, m.nd};
        const __half * X = (xin ? xin : hp.xh) + (size_t) t0 * m.K;
        const pf::Out O{y, ldy, mode, nullptr};
        const dim3 grid(m.N / pf::BM, (ntok + pf::BN - 1) / pf::BN);
        switch (m.type) {
            case T_Q4K: pf::mmh<pf::TQ4K><<<grid, 256, pf::hm_smem<pf::TQ4K>()>>>(W, X, ntok, O); break;
            case T_Q5K: pf::mmh<pf::TQ5K><<<grid, 256, pf::hm_smem<pf::TQ5K>()>>>(W, X, ntok, O); break;
            case T_Q6K: pf::mmh<pf::TQ6K><<<grid, 256, pf::hm_smem<pf::TQ6K>()>>>(W, X, ntok, O); break;
            case T_Q8_0: pf::mmh<pf::TQ80><<<grid, 256, pf::hm_smem<pf::TQ80>()>>>(W, X, ntok, O); break;
            default: throw std::runtime_error("gemm: unsupported weight type");
        }
    }

    // xh2[t] = silu(Wg x_t) * (Wu x_t) for the first ntok rows of xh (both Q4_K)
    void gemm_glu(const Mat & g, const Mat & u, int ntok) {
        if (g.type != T_Q4K || u.type != T_Q4K || u.rb != g.rb) throw std::runtime_error("gemm_glu: expected Q4_K gate / up");
        const pf::WMat W{g.w, g.d, g.type, g.K, g.N, g.rb, g.nd};
        const pf::Out O{nullptr, g.N, pf::OUT_STORE, hp.xh2};
        const dim3 grid(g.N / (pf::BM / 2), (ntok + pf::BN - 1) / pf::BN);
        pf::mmh<pf::TQ4K, true><<<grid, 256, pf::hm_smem<pf::TQ4K>()>>>(W, hp.xh, ntok, O, u.w);
    }

    // MK_PF_PROFILE=1: per-category GPU time of the batched prefill (events around each launch)
    struct PfEv { const char * cat; cudaEvent_t a, b; };
    std::vector<PfEv> pf_ev;
    bool pf_prof = getenv("MK_PF_PROFILE") != nullptr;
    template <class F> void pf_time(const char * cat, F f) {
        if (!pf_prof) { f(); return; }
        // MK_PF_PROFILE=2: idempotent kernels are timed alone (5 repeats, min)
        static const char * idem[] = {"attn_prep", "attn_out", "dn_out", "norm_quant", "dn_ab", "attn", "embed", "gemm_ffn_gu", "gemm_dn_in", "gemm_attn_qkv"};
        if (getenv("MK_PF_PROFILE")[0] == '2' && std::any_of(std::begin(idem), std::end(idem), [&](const char * c) { return !strcmp(c, cat); })) {
            float best = 1e30f;
            for (int r = 0; r < 5; r++) {
                cudaEvent_t a, b;
                CK(cudaEventCreate(&a));
                CK(cudaEventCreate(&b));
                CK(cudaDeviceSynchronize());
                CK(cudaEventRecord(a));
                f();
                CK(cudaEventRecord(b));
                CK(cudaEventSynchronize(b));
                float ms;
                CK(cudaEventElapsedTime(&ms, a, b));
                best = std::min(best, ms);
                cudaEventDestroy(a);
                cudaEventDestroy(b);
            }
            pf_iso[cat] += best;
            return;
        }
        PfEv e{cat, nullptr, nullptr};
        CK(cudaEventCreate(&e.a));
        CK(cudaEventCreate(&e.b));
        CK(cudaEventRecord(e.a));
        f();
        CK(cudaEventRecord(e.b));
        pf_ev.push_back(e);
    }
    std::map<std::string, double> pf_iso;
    void pf_report() {
        if (!pf_prof || pf_ev.empty()) return;
        CK(cudaDeviceSynchronize());
        // per category: sum, and median-per-launch x launches (robust to time-slicing by other GPU users)
        std::vector<std::string> cats;
        std::vector<std::vector<float>> ts;
        for (auto & e : pf_ev) {
            float ms;
            CK(cudaEventElapsedTime(&ms, e.a, e.b));
            auto it = std::find(cats.begin(), cats.end(), std::string(e.cat));
            if (it == cats.end()) { cats.push_back(e.cat); ts.push_back({}); it = cats.end() - 1; }
            ts[it - cats.begin()].push_back(ms);
            cudaEventDestroy(e.a);
            cudaEventDestroy(e.b);
        }
        pf_ev.clear();
        double all = 0, all_med = 0;
        std::vector<std::tuple<double, double, std::string, size_t>> rows;
        for (size_t i = 0; i < cats.size(); i++) {
            auto v = ts[i];
            double sum = 0;
            for (float x : v) sum += x;
            std::sort(v.begin(), v.end());
            const double med = v[v.size() / 2] * v.size();
            rows.emplace_back(med, sum, cats[i], v.size());
            all += sum;
            all_med += med;
        }
        for (auto & kv : pf_iso) fprintf(stderr, "  [isolated] %-12s %8.1f ms\n", kv.first.c_str(), kv.second);
        pf_iso.clear();
        std::sort(rows.rbegin(), rows.rend());
        fprintf(stderr, "[mk-prefill-profile] %.1f ms in kernels (median-based %.1f ms)\n", all, all_med);
        for (auto & r : rows)
            fprintf(stderr, "  %-14s median-based %8.1f ms %5.1f%%   sum %8.1f ms   x%zu\n", std::get<2>(r).c_str(), std::get<0>(r),
                    100 * std::get<0>(r) / all_med, std::get<1>(r), std::get<3>(r));
    }

    // Runs positions n0 .. n0 + toks.size() - 1 through the trunk (and the MTP block's K/V).
    // score: instead of MTP, greedy argmax of every position -> pred, and / or the logits of the
    // positions >= first -> sink.
    void pf_run(const std::vector<int> & toks, int n0, std::vector<int> * pred, const MKEngine::LogitSink * sink = nullptr,
                int first = 0) {
        pf_init();
        const int N = (int) toks.size(), cap = hp.cap;
        if (n0 == 0) CK(cudaMemset(hp.h, 0, D * sizeof(float)));
        else CK(cudaMemcpy(hp.h, hb.h_hist + (size_t) ((n0 - 1) % H_RING) * D, D * sizeof(float), cudaMemcpyDeviceToDevice));
        for (int c0 = 0; c0 < N; c0 += cap) {
            const int T = std::min(cap, N - c0), n = n0 + c0;
            CK(cudaMemcpy(hp.tok, toks.data() + c0, T * sizeof(int), cudaMemcpyHostToDevice));
            pf_time("embed", [&] { pk_embed<<<T, 256>>>(T); });
            int dl = 0;
            for (int L = 0; L < NL; L++) {
                const LayerW & W = hw.L[L];
                pf_time("norm_quant", [&] { pk_norm_quant<<<T, 256>>>(hp.x, W.attn_norm); });
                if (W.attn) {
                    pf_time("gemm_attn_qkv", [&] { gemm(W.q, T, hp.y0, QG, pf::OUT_STORE); });
                    pf_time("gemm_attn_qkv", [&] { gemm(W.k, T, hp.y1, 2 * KVD, pf::OUT_STORE); gemm(W.v, T, hp.y1 + KVD, 2 * KVD, pf::OUT_STORE); });
                    pf_time("attn_prep", [&] { pk_attn_prep<<<T, 256>>>(n, L / 4, W.qn, W.kn, 1); });
                    pf_time("attn", [&] {
                        if (hb.kvq8) pk_attn<true><<<dim3((T + AT_Q - 1) / AT_Q, NKV), 192, AT_SMEM>>>(T, n, L / 4);
                        else pk_attn<false><<<dim3((T + AT_Q - 1) / AT_Q, NKV), 192, AT_SMEM>>>(T, n, L / 4);
                    });
                    pf_time("attn_out", [&] { pk_attn_out<<<T, 256>>>(); });
                    pf_time("gemm_attn_o", [&] { gemm(W.o, T, hp.x, D, pf::OUT_ADD); });
                } else {
                    pf_time("gemm_dn_in", [&] { gemm(W.qkv, T, hp.y0, QKV, pf::OUT_STORE); gemm(W.gate, T, hp.y1, VD, pf::OUT_STORE); });
                    pf_time("dn_ab", [&] { pk_ab<<<dim3((T + 63) / 64, AB_SPLITS), 256>>>(W.ab, T); });
                    pf_time("dn_prep", [&] { pk_dn_prep<<<T, 256>>>(T, n, dl, W.conv, W.dt, W.A); pk_conv_ring<<<std::min(T, CONV_RING), 256>>>(T, n, dl); });
                    pf_time("dn_scan", [&] { pk_dn_scan<<<dim3(NVH, 2), 256>>>(T, dl); });
                    pf_time("dn_out", [&] { pk_dn_out<<<T, 256>>>(W.snorm); });
                    pf_time("gemm_dn_out", [&] { gemm(W.sout, T, hp.x, D, pf::OUT_ADD); });
                    dl++;
                }
                pf_time("norm_quant", [&] { pk_norm_quant<<<T, 256>>>(hp.x, W.post_norm); });
                pf_time("gemm_ffn_gu", [&] { gemm_glu(W.fg, W.fu, T); });
                pf_time("gemm_ffn_down", [&] { gemm(W.fd, T, hp.x, D, pf::OUT_ADD, 0, hp.xh2); });
            }
            if (pred || sink) {
                pk_norm_quant<<<T, 256>>>(hp.x, hw.onorm);
                std::vector<float> rows;
                for (int s = 0; s < T; s += 128) {
                    const int nt = std::min(128, T - s);
                    const int lo = std::max(s, first - n); // rows from lo on are at positions >= first
                    if (!pred && lo >= s + nt) continue;
                    gemm(hw.head, nt, hp.y0, VOCAB, pf::OUT_STORE, s);
                    if (pred) pk_argmax<<<nt, 256>>>(s);
                    if (sink && lo < s + nt) {
                        rows.resize((size_t) (s + nt - lo) * VOCAB);
                        CK(cudaMemcpy(rows.data(), hp.y0 + (size_t) (lo - s) * VOCAB, rows.size() * sizeof(float), cudaMemcpyDeviceToHost));
                        (*sink)(n + lo, s + nt - lo, rows.data());
                    }
                }
                if (pred) {
                    std::vector<int> am(T);
                    CK(cudaMemcpy(am.data(), hp.amax, T * sizeof(int), cudaMemcpyDeviceToHost));
                    pred->insert(pred->end(), am.begin(), am.end());
                }
                continue;
            }
            pf_time("mtp", [&] {
                pk_final_h<<<T, 256>>>(T, n);
                pk_mtp_in<<<T, 256>>>();
                gemm(hw.eh, T, hp.xm, D, pf::OUT_STORE);
                const LayerW & M = hw.L[NL];
                pk_norm_quant<<<T, 256>>>(hp.xm, M.attn_norm);
                gemm(M.k, T, hp.y1, 2 * KVD, pf::OUT_STORE);
                gemm(M.v, T, hp.y1 + KVD, 2 * KVD, pf::OUT_STORE);
                pk_attn_prep<<<T, 256>>>(n, 16, M.qn, M.kn, 0);
            });
            CK(cudaMemcpyAsync(hp.h, hp.h + (size_t) T * D, D * sizeof(float), cudaMemcpyDeviceToDevice));
        }
        CK(cudaGetLastError());
        CK(cudaDeviceSynchronize());
        pf_report();
    }

    void reset() {
        CK(cudaMemset(hb.dstate, 0, sizeof(float) * (size_t) 48 * NVH * 128 * 128));
        CK(cudaMemset(hb.h_hist, 0, sizeof(float) * (size_t) H_RING * D));
        memset(&hctl, 0, sizeof(hctl));
        CK(cudaMemcpy(hb.ctl, &hctl, sizeof(Ctl), cudaMemcpyHostToDevice));
        CK(cudaMemset(hb.cnt, 0, 16));
    }
};

MKEngine::MKEngine(const std::string & path, int ctx, int n_draft, const std::string & draft_vocab, int kv_mode)
    : p_(nullptr), n_draft_(n_draft) {
    if (n_draft < 0 || n_draft > MAXT - 1) throw std::runtime_error("n_draft must be in [0, " + std::to_string(MAXT - 1) + "]");
    p_ = new Impl(path, ctx, n_draft, draft_vocab, kv_mode);
    reset();
}

MKEngine::~MKEngine() { delete p_; }

void MKEngine::bench_phases() { p_->bench(); }

void MKEngine::reset() {
    p_->reset();
    p_->seq.clear();
    p_->ck_pos = -1;
    p_->dv_reset();
    stats = MKStats();
}

int MKEngine::position() const { return p_->hctl.n; }

int MKEngine::prefill(const std::vector<int> & tokens) {
    if (tokens.empty()) throw std::runtime_error("empty prompt");
    Ctl & c = p_->hctl;
    CK(cudaMemcpy(&c, p_->hb.ctl, sizeof(Ctl), cudaMemcpyDeviceToHost));
    std::vector<int> & seq = p_->seq;
    if ((int) (seq.size() + tokens.size()) + 8 > p_->ctx) throw std::runtime_error("prompt exceeds context");
    seq.insert(seq.end(), tokens.begin(), tokens.end());
    p_->dv_add(tokens.data(), tokens.size());
    // Everything from the committed delta-net position on is fed: on a continued sequence that is
    // the last emitted token (not yet processed) and any accepted drafts not yet folded into the
    // recurrent state, followed by the new tokens.  Their KV / MTP rows are simply rewritten.
    const int n0 = c.dn_c;
    const std::vector<int> feed(seq.begin() + n0, seq.end());
    c.mtp_valid = std::min(c.mtp_valid, n0);
    size_t i = 0;
    const char * mode = getenv("MK_PREFILL");
    const bool batched = !(mode && std::string(mode) == "mk");
    if (batched && feed.size() >= 2) {
        // all but the last token in batched kernels; the last one goes through the megakernel,
        // which produces the first output token and the MTP drafts
        const std::vector<int> head(feed.begin(), feed.end() - 1);
        p_->pf_run(head, n0, nullptr);
        i = head.size();
        c.dn_c = n0 + (int) i;
        c.mtp_valid = c.dn_c;
    }
    while (i < feed.size()) {
        const int T = (int) std::min<size_t>(MAXT, feed.size() - i);
        const bool last = i + T == feed.size();
        c.n = c.dn_c;
        for (int k = 0; k < T; k++) c.tok[k] = feed[i + k];
        CK(cudaMemcpy(p_->hb.ctl, &c, sizeof(Ctl), cudaMemcpyHostToDevice));
        p_->ndraft = p_->n_draft_max;
        p_->launch(MODE_PREFILL, T, last ? 1 : 0);
        CK(cudaMemcpy(&c, p_->hb.ctl, sizeof(Ctl), cudaMemcpyDeviceToHost));
        i += T;
    }
    p_->nd_avail = p_->ndraft;
    p_->all_acc = true;
    CK(cudaGetLastError());
    p_->ck_save();
    seq.push_back(c.out[0]);
    return c.out[0];
}

void MKEngine::set_sampling(float temp, float top_p, int top_k, float min_p, unsigned long long seed) {
    if (temp > 0.f && (top_k <= 0 || top_k > TOPK_MAX))
        throw std::runtime_error("sampling needs 1 <= top_k <= " + std::to_string(TOPK_MAX));
    p_->smp_temp = temp;
    p_->smp_top_p = top_p;
    p_->smp_top_k = top_k;
    p_->smp_min_p = min_p;
    p_->smp_seed = seed;
}

const std::vector<int> & MKEngine::sequence() const { return p_->seq; }
int MKEngine::committed() const { return p_->hctl.dn_c; }
int MKEngine::checkpoint_pos() const { return p_->ck_pos; }
bool MKEngine::rewind_to_checkpoint() {
    if (p_->ck_pos < 0 || p_->ck_pos > (int) p_->seq.size()) return false;
    p_->ck_restore();
    return true;
}

void MKEngine::truncate(int len) {
    std::vector<int> & seq = p_->seq;
    if (len < p_->hctl.dn_c || len > (int) seq.size())
        throw std::runtime_error("truncate: can only drop tokens emitted after the committed position");
    seq.resize(len);
}

std::vector<int> MKEngine::score(const std::vector<int> & tokens) {
    reset();
    std::vector<int> pred;
    const char * mode = getenv("MK_PREFILL");
    if (!(mode && std::string(mode) == "mk")) {
        p_->pf_run(tokens, 0, &pred);
        reset();
        return pred;
    }
    Ctl & c = p_->hctl;
    CK(cudaMemcpy(&c, p_->hb.ctl, sizeof(Ctl), cudaMemcpyDeviceToHost));
    size_t i = 0;
    while (i < tokens.size()) {
        const int T = (int) std::min<size_t>(MAXT, tokens.size() - i);
        c.n = (int) i;
        for (int k = 0; k < T; k++) c.tok[k] = tokens[i + k];
        CK(cudaMemcpy(p_->hb.ctl, &c, sizeof(Ctl), cudaMemcpyHostToDevice));
        p_->launch(MODE_PREFILL, T, 0, 1);
        CK(cudaMemcpy(&c, p_->hb.ctl, sizeof(Ctl), cudaMemcpyDeviceToHost));
        for (int k = 0; k < T; k++) pred.push_back(c.tgt[k]);
        i += T;
    }
    return pred;
}

int MKEngine::n_vocab() { return VOCAB; }

void MKEngine::score_logits(const std::vector<int> & tokens, int mk_from, int first, const LogitSink & sink) {
    const int N = (int) tokens.size();
    if (N + 8 > p_->ctx) throw std::runtime_error("sequence exceeds context");
    reset();
    mk_from = std::min(std::max(mk_from, 0), N);
    if (mk_from > 0)
        p_->pf_run(std::vector<int>(tokens.begin(), tokens.begin() + mk_from), 0, nullptr, first < mk_from ? &sink : nullptr, first);
    Ctl & c = p_->hctl;
    CK(cudaMemcpy(&c, p_->hb.ctl, sizeof(Ctl), cudaMemcpyDeviceToHost));
    c.dn_c = mk_from;
    std::vector<float> rows((size_t) MAXT * VOCAB);
    for (int i = mk_from; i < N;) {
        const int T = std::min(MAXT, N - i);
        c.n = i;
        for (int k = 0; k < T; k++) c.tok[k] = tokens[i + k];
        CK(cudaMemcpy(p_->hb.ctl, &c, sizeof(Ctl), cudaMemcpyHostToDevice));
        p_->launch(MODE_PREFILL, T, 0, 2);
        CK(cudaMemcpy(&c, p_->hb.ctl, sizeof(Ctl), cudaMemcpyDeviceToHost));
        const int s = std::max(0, first - i);
        if (s < T) {
            CK(cudaMemcpy(rows.data(), p_->hb.logits + (size_t) s * VOCAB, (size_t) (T - s) * VOCAB * sizeof(float), cudaMemcpyDeviceToHost));
            sink(i + s, T - s, rows.data());
        }
        i += T;
    }
    reset();
}

int MKEngine::step(std::vector<int> & out) {
    Ctl & c = p_->hctl;
    if (c.n + MAXT + 2 >= p_->ctx) throw std::runtime_error("context full");
    const int T = 1 + p_->nd_avail;
    // Draft depth: after a cycle that rejected a draft, draft one token less (but at least 3), since a
    // deeper draft is then unlikely to be accepted; the full depth again after a fully accepted cycle.
    p_->ndraft = (!p_->all_acc && n_draft_ > 3) ? n_draft_ - 1 : n_draft_;
    p_->launch(MODE_DECODE, T, 0);
    p_->nd_avail = p_->ndraft;
    CK(cudaMemcpy(&c, p_->hb.ctl, sizeof(Ctl), cudaMemcpyDeviceToHost));
    for (int i = 0; i < c.n_out; i++) out.push_back(c.out[i]);
    p_->seq.insert(p_->seq.end(), c.out, c.out + c.n_out);
    p_->dv_add(c.out, c.n_out);
    p_->all_acc = c.n_acc == T - 1;
    stats.cycles++;
    stats.drafted += T - 1;
    stats.accepted += c.n_acc;
    return c.n_out;
}

std::vector<float> MKEngine::last_hidden() {
    std::vector<float> h(D);
    const int pos = p_->hctl.dn_c - 1;
    CK(cudaMemcpy(h.data(), p_->hb.h_hist + (size_t) (pos % H_RING) * D, D * 4, cudaMemcpyDeviceToHost));
    return h;
}
