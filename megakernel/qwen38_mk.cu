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

#define MK_SHARED_NS // (anonymous)
#include "mk_shared.cuh"
#include "mk_launch.h"

namespace {
__constant__ ModelW cW;
__constant__ Bufs cBs[NVIEW];
__constant__ int cView;
#if MK_L2_TUNE
__constant__ L2Tune cL2; // (its own symbol: cB's fields are an LDS of s_view plus an indexed LDC)
#define MK_ATT_PF_CTX cL2.attn_ctx
#else
#define MK_ATT_PF_CTX (2 * ATT_BUCKET)
#endif
__shared__ int s_view;
#define cB cBs[s_view]
#define MK_VIEW_INIT()                          \
    do {                                        \
        if (threadIdx.x == 0) s_view = cView;   \
        __syncthreads();                        \
    } while (0)

// the rows of a batched launch's lanes (Launch::ln without the sampling): row t belongs to lane i when
// r0[i] <= t < r0[i] + T[i]; rows past the lanes (padding of the pass) belong to none
struct RowTab {
    int nl;
    signed char lane[NSEQ], T[NSEQ], r0[NSEQ];
};
// the lane of row t of a batched pass (-1: a row past the lanes) and the lane's first row; unrolled over NSEQ, so that rt is
// read at constant indices (a runtime index copies it to local memory)
__device__ __forceinline__ int rt_lane(const RowTab & rt, int t, int & r0) {
    int ln = -1;
    r0 = 0;
#pragma unroll
    for (int i = 0; i < NSEQ; i++)
        if (i < rt.nl && t >= rt.r0[i] && t < rt.r0[i] + rt.T[i]) {
            ln = rt.lane[i];
            r0 = rt.r0[i];
        }
    return ln;
}
// a batched launch's row table, built at each call that takes it (held across the layer loop, its fields would stay live)
__device__ __forceinline__ RowTab row_tab(const Launch & P) {
    RowTab rt;
    rt.nl = P.nl;
#pragma unroll
    for (int i = 0; i < NSEQ; i++) {
        rt.lane[i] = (signed char) (i < P.nl ? P.ln[i].lane : 0);
        rt.T[i] = (signed char) (i < P.nl ? P.ln[i].T : 0);
        rt.r0[i] = (signed char) (i < P.nl ? P.ln[i].r0 : 0);
    }
    return rt;
}

// lane i of a batched launch as a solo decode launch of its own (its rows, drafts and sampling)
__device__ __forceinline__ Launch lane_launch(const Launch & P, int i) {
    Launch Q = P;
    const Launch::Lane & l = P.ln[i];
    Q.T = l.T;
    Q.ndraft = l.nd;
    Q.top_k = l.top_k;
    Q.spec_smp = l.spec_smp;
    Q.temp = l.temp;
    Q.top_p = l.top_p;
    Q.min_p = l.min_p;
    Q.seed = l.seed;
    Q.nl = 0;
    return Q;
}

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
// normal-priority L2 prefetch of a streamed weight line (lm_head<16>; wv.cuh's pf_l2n is the same for the _ck phases)
__device__ __forceinline__ void pf_l2s(const void * p) { asm volatile("prefetch.global.L2 [%0];" ::"l"(p)); }
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

__device__ __forceinline__ unsigned long long gtimer() {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

__shared__ unsigned s_bar_k; // thread 0: grid barriers this block has arrived at in this launch (gsync, k5_arrive)
extern __shared__ __align__(16) uint8_t smem[]; // dynamic shared memory of the persistent block

// Grid barrier, thread 0 of each block (epoch: the arrival count that completes this generation, already advanced).
// Form V (gsync runs form 0; the gsync bench, MK_GSB, runs every form): 0 = release-add on arrival, acquire-poll the same word until every block of this generation arrived;
// 1 / 2 = flag broadcast: acq_rel-add the count word, the last arriver release-writes the generation to the go word (its
// own 128-byte line, cB.bar + GS_GO) and the others poll only go (1: relaxed polls, then one acquire re-read; 2: acquire
// polls); 3 = form 0's arrival with relaxed polls and one acquire re-read. Bench launches start both words at 0.
// Other launches carry the counts on. Word BAR_BASE holds the barrier count at the end of the previous launch and
// C1_BASE the C1 chunk count (every chunk's counter advances by C1_NJC tiles per split edge); block 0 writes both after the launch's last
// barrier (mk_egress), thread 0 of every block reads them at the start (s_bar0, s_c1b). A count wraps after ~10^5
// launches: the waits compare wrap-safe (MK_BAR_LT); MK_EPOCH turns a launch-relative count into the absolute one.
constexpr int BAR_BASE = GS_GO + 8, C1_BASE = GS_GO + 9;
__shared__ unsigned s_bar0, s_c1b;
#define MK_BAR_LT(v, e) ((int) ((v) - (e)) < 0)
#define MK_EPOCH(e) ((e) + s_bar0)
// blockIdx.x read where it is used (asm volatile: not hoisted ahead of the branch that guards it; a copy held in a register
// across the phase calls is spilled, and reloading it put a local-memory load on the barrier's release path)
__device__ __forceinline__ unsigned ctaid_x() {
    unsigned b;
    asm volatile("mov.u32 %0, %%ctaid.x;" : "=r"(b));
    return b;
}
template <int V>
__device__ __forceinline__ void gs_arrive_wait(unsigned epoch) {
    unsigned * const bar = cB.bar; // (read once: the asm's memory clobbers made every poll re-read s_view and the view's pointer)
    unsigned v;
    if constexpr (V == 0 || V == 3) {
        asm volatile("red.release.gpu.global.add.u32 [%0], 1;" ::"l"(bar) : "memory");
        if constexpr (V == 0) {
            do {
                asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(bar) : "memory");
            } while (MK_BAR_LT(v, epoch));
        } else {
            do {
                asm volatile("ld.relaxed.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(bar) : "memory");
            } while (MK_BAR_LT(v, epoch));
            asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(bar) : "memory");
        }
    } else {
        unsigned * const go = bar + GS_GO;
        asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], 1;" : "=r"(v) : "l"(bar) : "memory");
        if (v == epoch - 1) { // last arrival of this generation: every other block's writes are visible to this thread
            asm volatile("st.release.gpu.global.u32 [%0], %1;" ::"l"(go), "r"(epoch) : "memory");
        } else if constexpr (V == 2) {
            do {
                asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(go) : "memory");
            } while (MK_BAR_LT(v, epoch));
        } else {
            do {
                asm volatile("ld.relaxed.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(go) : "memory");
            } while (MK_BAR_LT(v, epoch));
            asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(go) : "memory");
        }
    }
}

// Grid barrier: every block's writes before it are visible to every block's reads after it. Barrier k of the launch
// completes at the count MK_EPOCH(k gridDim.x); thread 0 keeps k in shared memory (s_bar_k). (It was the callers' epoch:
// in the out-of-line passes a reference into mk_cycle's frame, so every arrival first did two dependent local-memory
// loads, the pointer's spill slot and then epoch, plus a store, and the profiling stamps a division.) epoch: unused (the
// callers' signatures stay).
__device__ __forceinline__ void gsync(unsigned & /*epoch*/, int tag) {
    __syncthreads();
    if (threadIdx.x == 0) {
        const unsigned k = s_bar_k + 1;
        s_bar_k = k;
        unsigned long long * const prof = cB.prof;
        if (prof && k < 1024) prof[4096 + k * 128 + ctaid_x()] = gtimer();
        gs_arrive_wait<0>(MK_EPOCH(k * gridDim.x));
        if (prof && k < 2048 && ctaid_x() == 0) prof[k] = (gtimer() << 8) | (unsigned long long) tag;
    }
    __syncthreads();
}

// K5: the dn_proj -> dn_core grid barrier is a split barrier (k5_arrive after dn_proj, k5_wait inside dn_core) and the
// 48 heads go to the first 48 blocks to finish dn_proj, which load the head's state and replay its accepted drafts
// (inputs of earlier launches) before the wait. Bitwise exact. Form 0's counter and epoch accounting, gsync cut in two.
// k5_arrive: every block, right after its producer items; release-adds one arrival (atom: the returned count is the
// block's arrival rank in this generation) and advances s_bar_k as gsync does. The first NVH blocks to arrive take the
// delta-net heads 0 .. NVH - 1 (s_k5h, -1: none), s_k5e = the count that completes the generation. Every block then calls
// k5_wait before its next arrival anywhere: the count is cumulative, so a block one generation ahead could otherwise
// complete this generation's count while a producer has not arrived. MK_PROFILE: the arrival is stamped as in gsync and
// the LAST arriver stamps the release word (the phase boundary) right after its atom returns.
__shared__ int s_k5h;
__shared__ unsigned s_k5e;
__device__ __forceinline__ void k5_arrive(unsigned & /*epoch*/, int tag) {
    __syncthreads();
    if (threadIdx.x == 0) {
        const unsigned k = s_bar_k + 1; // this barrier's index (as gsync's)
        s_bar_k = k;
        unsigned long long * const prof = cB.prof;
        if (prof && k < 1024) prof[4096 + k * 128 + ctaid_x()] = gtimer();
        const unsigned e0 = MK_EPOCH((k - 1) * gridDim.x); // the count before this generation's first arrival
        unsigned old;
        asm volatile("atom.release.gpu.global.add.u32 %0, [%1], 1;" : "=r"(old) : "l"(cB.bar) : "memory");
        const unsigned rank = old - e0; // arrivals of this generation before this one
        s_k5h = rank < (unsigned) NVH ? (int) rank : -1;
        s_k5e = e0 + gridDim.x;
        if (prof && rank == gridDim.x - 1 && k < 2048) prof[k] = (gtimer() << 8) | (unsigned long long) tag;
    }
    __syncthreads();
}
// k5_wait: thread 0 acquire-polls the count until the generation is complete, then the block (as gsync's wait half)
__device__ __forceinline__ void k5_wait() {
    if (threadIdx.x == 0) {
        const unsigned e = s_k5e;
        const unsigned * const bar = cB.bar;
        unsigned v;
        do {
            asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(bar) : "memory");
        } while (MK_BAR_LT(v, e));
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
// roundf(v / (amax / 127)), half away from zero (llama.cpp's quantize_q8_1); kv_store_q8 / aq_i8_q round v * (127 / amax) to
// even instead: two fixed conventions, don't harmonise them (it would change the stored activations / KV bytes)
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

#ifndef MK_Q32N
#define MK_Q32N 1 // 1: resid_wv / resid_ck_last's K2 combine runs a batch's q32_store chains interleaved (q32_store_n)
#endif
#if MK_Q32N
// q32_store(q, t0 + i, j, v[i], ss[i], true) for the N tokens i, one warp: the tokens' shuffle chains and divisions
// interleaved, each token's ops and stores those of q32_store (dn_q32_store's form plus the ssq sum; keep the three in step)
template <int N>
__device__ __forceinline__ void q32_store_n(const QBuf & q, int t0, int j, const float (&v)[N], const float (&ss)[N]) {
    const int lane = threadIdx.x & 31;
    const int b = j >> 3, p = (j >> 1) & 3, part = j & 1, hh = lane >> 4, byte = lane & 15;
    float am[N], s[N];
    int qs[N];
#pragma unroll
    for (int i = 0; i < N; i++) am[i] = fabsf(v[i]);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) // wmax
#pragma unroll
        for (int i = 0; i < N; i++) am[i] = fmaxf(am[i], __shfl_xor_sync(0xffffffffu, am[i], o));
#pragma unroll
    for (int i = 0; i < N; i++) {
        const float d = am[i] / 127.0f;
        qs[i] = am[i] == 0.0f ? 0 : (int) roundf(v[i] / d);
        am[i] = d;
        (part ? q.xb : q.xa)[((t0 + i) * q.U + 8 * b + 2 * p + hh) * 16 + byte] = (int8_t) qs[i];
    }
#pragma unroll
    for (int o = 8; o > 0; o >>= 1)
#pragma unroll
        for (int i = 0; i < N; i++) qs[i] += __shfl_xor_sync(0xffffffffu, qs[i], o);
    if (byte == 0) {
#pragma unroll
        for (int i = 0; i < N; i++) {
            const int gi = (t0 + i) * q.U + 8 * b + 2 * p + hh;
            reinterpret_cast<__half *>(q.meta + gi)[part] = __float2half_rn(am[i]);
            reinterpret_cast<short *>(q.meta + gi)[2 + part] = (short) qs[i];
        }
    }
#pragma unroll
    for (int i = 0; i < N; i++) s[i] = ss[i];
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) // wsum
#pragma unroll
        for (int i = 0; i < N; i++) s[i] += __shfl_xor_sync(0xffffffffu, s[i], o);
    if (lane == 0) {
#pragma unroll
        for (int i = 0; i < N; i++) q.ssq[(t0 + i) * q.U + j] = s[i];
    }
}
#endif

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

// lm_head<16>'s activations: 16 rows of U_D units at 40 B do not fit fill_xs's layout (102 KB). xa | xb of all rows from
// 0; the meta split in two: the half2 scales once per unit pair below misc() (units 2m and 2m + 1 hold the two halves of
// the same natural 32-blocks, so q32_store, the only writer of qn, gives them the same d) and the short2 chunk sums of
// every unit above it (15 KB in all; the uint2 meta of all 16 rows, 20 KB, does not fit, and re-summing rows 8..15's
// int8 values instead cost 64 dp4a per unit)
struct SXH {
    int4 * xa;
    int4 * xb;
    unsigned * xd; // [16][U_D / 2]: the half2 (dA, dB) of unit pair u >> 1
    unsigned * xq; // [16][U_D]: the short2 (sum A, sum B) of unit u
};
constexpr int LH_XB = 16 * U_D * 16, LH_XD = 2 * LH_XB, LH_XQ = SM_MISC + 2048;
static_assert(LH_XD + 16 * (U_D / 2) * 4 <= SM_MISC && LH_XQ + 16 * U_D * 4 <= SM_TOTAL, "lm_head<16> activations");
__device__ __forceinline__ SXH make_sxh() {
    uint8_t * const sm = smem;
    return SXH{reinterpret_cast<int4 *>(sm), reinterpret_cast<int4 *>(sm + LH_XB), reinterpret_cast<unsigned *>(sm + LH_XD),
               reinterpret_cast<unsigned *>(sm + LH_XQ)};
}

// wv activation layout (wv.cuh, wv_fill): tokens per C-column group (8, or 4 for the per-16 split of X16), groups, rows
template <int T, bool X16> struct WvCfg {
    static constexpr int TPG = X16 ? 4 : 8;
    static constexpr int NG = (T + TPG - 1) / TPG;
    static constexpr int NTOKP = NG * TPG;
};
// the dp4a view (rows_* with XP = 2) of wv_fill<T, false, US>'s activation units: token stride US + 2
template <int T, int US>
__device__ __forceinline__ SX wv_sx() {
    constexpr int LDU = US + 2, NTOKP = WvCfg<T, false>::NTOKP;
    SX s;
    s.xa = reinterpret_cast<int4 *>(smem);
    s.xb = s.xa + NTOKP * LDU;
    s.xm = reinterpret_cast<uint2 *>(s.xb + NTOKP * LDU);
    s.U = LDU;
    return s;
}

__device__ __forceinline__ float * misc(uint8_t * /*sm*/) {
    uint8_t * const sm = smem; return reinterpret_cast<float *>(sm + SM_MISC); }

// threadIdx.x read afresh (an S2R the compiler can neither hoist nor CSE): for values derived from the thread index that
// would otherwise stay live across a long loop. At the 128-register cap ptxas parks such values in spill slots, and a
// reload after a long gap misses L1 (e.g. the W48 item prologue's reload)
__device__ __forceinline__ int tid_x() {
    int t;
    asm volatile("mov.u32 %0, %%tid.x;" : "=r"(t));
    return t;
}

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

// fill_xs<16> into the SXH layout (normed rows of the K = D residual stream: q.U == U_D)
__device__ void fill_xs16(const QBuf & q, int t0, const SXH & s, float * s_rs) {
    static_assert(NW >= 16, "one warp per row for the norm scales");
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int8_t * ga = q.xa + (size_t) t0 * U_D * 16;
    const int8_t * gb = q.xb + (size_t) t0 * U_D * 16;
    const uint2 * gm = q.meta + (size_t) t0 * U_D;
    for (int i = tid; i < 16 * U_D; i += NT) {
        cp_async16(s.xa + i, ga + (size_t) i * 16);
        cp_async16(s.xb + i, gb + (size_t) i * 16);
    }
    asm volatile("cp.async.commit_group;" ::: "memory");
    for (int i = tid; i < 16 * U_D / 2; i += NT) { // unit pair i: meta[2i], meta[2i + 1] (the same half2 d; their sums)
        const int4 v = __ldcg(reinterpret_cast<const int4 *>(gm + 2 * i));
        s.xd[i] = (unsigned) v.x;
        s.xq[2 * i] = (unsigned) v.y;
        s.xq[2 * i + 1] = (unsigned) v.w;
    }
    if (warp < 16) {
        float p[U_D / 32];
#pragma unroll
        for (int k = 0; k < U_D / 32; k++) p[k] = __ldcg(q.ssq + (t0 + warp) * U_D + lane + 32 * k);
        float acc = 0.f;
#pragma unroll
        for (int k = 0; k < U_D / 32; k++) acc += p[k];
        acc = wsum(acc);
        if (lane == 0) s_rs[warp] = rsqrtf(acc / (float) D + EPS);
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

// accR on lm_head<16>'s activations (SXH): the unit pair's half2 scales and the unit's short2 sums (the values accR(SX)
// reads from the uint2 meta), then the same expression per row
template <int T, int R, int UU>
__device__ __forceinline__ void accR(const SXH & s, int u, const int (&A)[R][4], const int (&B)[R][4],
                                     const float (&fA)[R], const float (&gA)[R], const float (&fB)[R],
                                     const float (&gB)[R], float (&acc)[R][T]) {
    static_assert(T == 16 && UU == U_D, "SXH: the 16-row lm_head (token stride U_D, as make_sxh / fill_xs16)");
#pragma unroll
    for (int t = 0; t < T; t++) {
        const int ix = t * UU + u;
        const int4 va = s.xa[ix], vb = s.xb[ix];
        const unsigned dw = s.xd[t * (UU / 2) + (u >> 1)], qq = s.xq[ix];
        const int qa = (short) (qq & 0xffffu), qb = (short) (qq >> 16);
        const float2 dx = __half22float2(*reinterpret_cast<const __half2 *>(&dw));
        const float sqA = dx.x * (float) qa, sqB = dx.y * (float) qb;
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

template <int T, int IT, int R, int XP = 0>
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
        accR<T, R, 32 * IT + XP>(s, u, A, B, fA, gA, fB, gB, acc);
    }
}

template <int T, int IT, int R, int XP = 0>
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
        accR<T, R, 32 * IT + XP>(s, u, A, B, fA, gA, fB, gB, acc);
    }
}

// repacked Q6_K, 210 B per 256 weights like the GGUF block, in two streams per row:
//   rows (Mat::w, Q6_BLK = 192 B per block): qlo[128] (q4k-style nibbles) | qhi[64] (8 B per unit);
//   meta (Mat::d, 18 B per block, in pairs of blocks: sc[16] even | sc[16] odd | d even | d odd, 36 B).
// A unit reads its 4 scales sc[4p..4p+3] and the pair's two d (one 4-byte load each).
template <int T, int IT, int R, int XP = 0, class S = SX>
__device__ __forceinline__ void rows_q6k(const uint8_t * const (&rp)[R], const __half * const (&dp)[R], const S & s,
                                         float (&acc)[R][T]) {
    const int lane = threadIdx.x & 31;
    // (T > 8, lm_head<16>: one unit per iteration, the registers of 16 rows' sums)
#pragma unroll(IT > 8 ? (T >= 5 ? 1 : 4) : (T > 8 ? 1 : IT))
    for (int i = 0; i < IT; i++) {
        const int u = lane + 32 * i, b = u >> 3, p = (u >> 1) & 3, hh = u & 1;
        int4 q[R];
        int2 hb[R];
        int sc4[R], dd[R];
#pragma unroll
        for (int k = 0; k < R; k++) {
            const uint8_t * blk = rp[k] + b * Q6_BLK;
            const uint8_t * mp = reinterpret_cast<const uint8_t *>(dp[k]) + (b >> 1) * (2 * Q6_META);
            q[k] = ldw4(blk + 32 * p + 16 * hh);
            hb[k] = ldw2(blk + 128 + 8 * (u & 7));
            sc4[k] = ldw1(mp + 16 * (b & 1) + 4 * p);
            dd[k] = ldw1(mp + 32);
        }
        int A[R][4], B[R][4];
        float fA[R], gA[R], fB[R], gB[R];
#pragma unroll
        for (int k = 0; k < R; k++) {
            const float d = __half2float(__ushort_as_half((unsigned short) ((unsigned) dd[k] >> (16 * (b & 1)))));
            fA[k] = d * (float) (int8_t) (sc4[k] >> (8 * hh));
            fB[k] = d * (float) (int8_t) (sc4[k] >> (16 + 8 * hh));
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
        accR<T, R, 32 * IT + XP>(s, u, A, B, fA, gA, fB, gB, acc);
    }
}

// repacked Q8_0: per unit 32 B (A 16 int8 | B 16 int8), scales per natural 32-block
template <int T, int IT, int R, int XP = 0, class S = SX>
__device__ __forceinline__ void rows_q80(const uint8_t * const (&rp)[R], const __half * const (&dp)[R], const S & s,
                                         float (&acc)[R][T]) {
    const int lane = threadIdx.x & 31;
    // (T > 8, lm_head<16>: one unit per iteration, the registers of 16 rows' sums)
#pragma unroll(IT > 8 ? (T >= 5 ? 1 : 4) : (T > 8 ? 1 : IT))
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
        accR<T, R, 32 * IT + XP>(s, u, A, B, fA, gA, fB, gB, acc);
    }
}

// kvalues_iq4nl as int8, four per word
constexpr unsigned IQ4_T0 = 0xBFAD9881u, IQ4_T1 = 0xF6EADDCFu, IQ4_T2 = 0x26190D01u, IQ4_T3 = 0x71594535u;
// The 8 4-bit indices of word q -> table bytes of its low nibbles (lo) and high nibbles (hi), four
// per int (llama.cpp's get_int_from_table_16). __byte_perm reads only bits 0-2 of each selector nibble.
__device__ __forceinline__ void iq4_lut(unsigned q, unsigned t0, unsigned t1, unsigned t2, unsigned t3, int & lo, int & hi) {
    const unsigned sel = 0x32103210u | ((q & 0x88888888u) >> 1);
    const unsigned a = __byte_perm(__byte_perm(t0, t1, q), __byte_perm(t2, t3, q), sel);
    const unsigned b = __byte_perm(__byte_perm(t0, t1, q >> 16), __byte_perm(t2, t3, q >> 16), sel >> 16);
    lo = (int) __byte_perm(a, b, 0x6420);
    hi = (int) __byte_perm(a, b, 0x7531);
}

// accR for signed weights (no min term)
template <int T, int R, int UU>
__device__ __forceinline__ void accR0(const SX & s, int u, const int (&A)[R][4], const int (&B)[R][4], const float (&fA)[R],
                                      const float (&fB)[R], float (&acc)[R][T]) {
#pragma unroll
    for (int t = 0; t < T; t++) {
        const int ix = t * UU + u;
        const int4 va = s.xa[ix], vb = s.xb[ix];
        const uint2 mm = s.xm[ix];
        const float2 dx = __half22float2(*reinterpret_cast<const __half2 *>(&mm.x));
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
            acc[k][t] += fA[k] * (dx.x * (float) sa) + fB[k] * (dx.y * (float) sb);
        }
    }
}

// IQ4_XS re-laid-out at load in pairs of 256-weight blocks, 272 B per pair (repack_iq4xs): the two GGUF block
// headers (fp16 d | u16 scales_h | u8 scales_l[4]), then each block's 128 index bytes in Q4_K order.
// A weight is d * (ls - 32) * kvalues_iq4nl[index], ls the 6-bit scale of its 32-weight sub-block.
template <int T, int IT, int R, int XP = 0>
__device__ __forceinline__ void rows_iq4x(const uint8_t * const (&rp)[R], const SX & s, float (&acc)[R][T]) {
    const int lane = threadIdx.x & 31;
#pragma unroll(IT > 8 ? (T >= 5 ? 1 : 4) : IT)
    for (int i = 0; i < IT; i++) {
        const int u = lane + 32 * i, b = u >> 3, p = (u >> 1) & 3, hh = u & 1;
        int2 h[R];
        int4 q[R];
#pragma unroll
        for (int k = 0; k < R; k++) {
            const uint8_t * pr = rp[k] + (b >> 1) * 272;
            h[k] = ldw2(pr + 8 * (b & 1));
            q[k] = ldw4(pr + 16 + 128 * (b & 1) + 32 * p + 16 * hh);
        }
        int A[R][4], B[R][4];
        float fA[R], fB[R];
#pragma unroll
        for (int k = 0; k < R; k++) {
            const float d = __half2float(__ushort_as_half((unsigned short) (h[k].x & 0xffff)));
            const unsigned sl = ((unsigned) h[k].y >> (8 * p)) & 0xffu, sh = ((unsigned) h[k].x >> (16 + 4 * p)) & 0xfu;
            fA[k] = d * (float) ((int) ((sl & 0xfu) | ((sh & 3u) << 4)) - 32);
            fB[k] = d * (float) ((int) ((sl >> 4) | ((sh >> 2) << 4)) - 32);
            iq4_lut(q[k].x, IQ4_T0, IQ4_T1, IQ4_T2, IQ4_T3, A[k][0], B[k][0]);
            iq4_lut(q[k].y, IQ4_T0, IQ4_T1, IQ4_T2, IQ4_T3, A[k][1], B[k][1]);
            iq4_lut(q[k].z, IQ4_T0, IQ4_T1, IQ4_T2, IQ4_T3, A[k][2], B[k][2]);
            iq4_lut(q[k].w, IQ4_T0, IQ4_T1, IQ4_T2, IQ4_T3, A[k][3], B[k][3]);
        }
        accR0<T, R, 32 * IT + XP>(s, u, A, B, fA, fB, acc);
    }
}

// IQ4_NL re-laid-out at load into 144-byte blocks of 256 weights (repack_iq4nl): the fp16 scales d of the eight
// 32-weight blocks, then the 128 index bytes in Q4_K order (as IQ4_XS).  A weight is d * kvalues_iq4nl[index].
// Same integer products and the same float arithmetic (accR with zero mins) as a Q8_0 re-encode of these tensors,
// so the results are bit-identical to it at 0.53x the bytes.
template <int T, int IT, int R, int XP = 0>
__device__ __forceinline__ void rows_iq4nl(const uint8_t * const (&rp)[R], const SX & s, float (&acc)[R][T]) {
    const int lane = threadIdx.x & 31;
#pragma unroll(IT > 8 ? (T >= 5 ? 1 : 4) : IT)
    for (int i = 0; i < IT; i++) {
        const int u = lane + 32 * i, b = u >> 3, p = (u >> 1) & 3, hh = u & 1;
        int4 q[R];
        int dd[R];
#pragma unroll
        for (int k = 0; k < R; k++) {
            const uint8_t * blk = rp[k] + b * 144;
            dd[k] = ldw1(blk + 4 * p);
            q[k] = ldw4(blk + 16 + 32 * p + 16 * hh);
        }
        int A[R][4], B[R][4];
        float fA[R], gA[R], fB[R], gB[R];
#pragma unroll
        for (int k = 0; k < R; k++) {
            const float2 df = __half22float2(*reinterpret_cast<const __half2 *>(&dd[k]));
            fA[k] = df.x; fB[k] = df.y; gA[k] = 0.f; gB[k] = 0.f;
            iq4_lut(q[k].x, IQ4_T0, IQ4_T1, IQ4_T2, IQ4_T3, A[k][0], B[k][0]);
            iq4_lut(q[k].y, IQ4_T0, IQ4_T1, IQ4_T2, IQ4_T3, A[k][1], B[k][1]);
            iq4_lut(q[k].z, IQ4_T0, IQ4_T1, IQ4_T2, IQ4_T3, A[k][2], B[k][2]);
            iq4_lut(q[k].w, IQ4_T0, IQ4_T1, IQ4_T2, IQ4_T3, A[k][3], B[k][3]);
        }
        accR<T, R, 32 * IT + XP>(s, u, A, B, fA, gA, fB, gB, acc);
    }
}

// Native Q3_K (repack_q3k): per row a value stream of Q3_BLK = 96 B per 256-block, unit u' = 2p + hh at 8 u' (the 2-bit
// lows: A's 16 in the first word, B's in the second, element 4j + m at bits 2j of byte m) and its high bits at 64 + 4 u'
// (byte m: bit j = A element 4j + m, bit 4 + j = B), plus the Q6_K meta stream (sc - 32 as int8, d; Mat::d). A weight is
// d (sc - 32) (u3 - 4), u3 = lows + 4 high. The GEMV feeds u3 + 28 = the Q6_K value (q + 32) of a Q6_K re-encode,
// with the same scales and the same float arithmetic as rows_q6k: bit-identical to it at 114 / 210 the bytes.
template <int T, int IT, int R, int XP = 0>
__device__ __forceinline__ void rows_q3k(const uint8_t * const (&rp)[R], const __half * const (&dp)[R], const SX & s,
                                         float (&acc)[R][T]) {
    const int lane = threadIdx.x & 31;
#pragma unroll(IT > 8 ? (T >= 5 ? 1 : 2) : IT) // (4 at T < 5 spills 84-152 B in resid32<1, 17 / 10, Q3_K>)
    for (int i = 0; i < IT; i++) {
        const int u = lane + 32 * i, b = u >> 3, p = (u >> 1) & 3, hh = u & 1, w = u & 7;
        int2 lo[R];
        int hi[R], sc4[R], dd[R];
#pragma unroll
        for (int k = 0; k < R; k++) {
            const uint8_t * blk = rp[k] + b * Q3_BLK;
            const uint8_t * mp = reinterpret_cast<const uint8_t *>(dp[k]) + (b >> 1) * (2 * Q6_META);
            lo[k] = ldw2(blk + 8 * w);
            hi[k] = ldw1(blk + 64 + 4 * w);
            sc4[k] = ldw1(mp + 16 * (b & 1) + 4 * p);
            dd[k] = ldw1(mp + 32);
        }
        int A[R][4], B[R][4];
        float fA[R], gA[R], fB[R], gB[R];
#pragma unroll
        for (int k = 0; k < R; k++) {
            const float d = __half2float(__ushort_as_half((unsigned short) ((unsigned) dd[k] >> (16 * (b & 1)))));
            fA[k] = d * (float) (int8_t) (sc4[k] >> (8 * hh));
            fB[k] = d * (float) (int8_t) (sc4[k] >> (16 + 8 * hh));
            gA[k] = 32.0f * fA[k];
            gB[k] = 32.0f * fB[k];
            const unsigned x = (unsigned) lo[k].x, y = (unsigned) lo[k].y, h = (unsigned) hi[k];
#pragma unroll
            for (int j = 0; j < 4; j++) {
                A[k][j] = (int) ((((x >> (2 * j)) & 0x03030303u) | (((h >> j) & 0x01010101u) << 2)) + 0x1c1c1c1cu);
                B[k][j] = (int) ((((y >> (2 * j)) & 0x03030303u) | (((h >> (4 + j)) & 0x01010101u) << 2)) + 0x1c1c1c1cu);
            }
        }
        accR<T, R, 32 * IT + XP>(s, u, A, B, fA, gA, fB, gB, acc);
    }
}

// iq3s_grid (ggml): 512 entries of four magnitudes {1, 3, .., 15}; filled at load from ggml's own dequantization
__device__ uint32_t g_iq3s_grid[512];
// (through L1, kept there: the weight streams bypass L1, so the 2 KB table stays resident)
__device__ __forceinline__ unsigned ldg_l1(const uint32_t * p) {
    unsigned r;
    asm("ld.global.nc.L1::evict_last.u32 %0, [%1];" : "=r"(r) : "l"(p));
    return r;
}
__device__ __forceinline__ unsigned iq3s_grid(unsigned idx) { return ldg_l1(g_iq3s_grid + idx); }
// Native IQ3_S (repack_iq3s): per row a value stream of 96 B per 256-block, unit u' = 2p + hh at 8 u' (the low bytes of
// the four grid indices of A (sub-block 2p, indices 4 hh ..) then of B (sub-block 2p + 1)) and its signs at 64 + 4 u'
// (byte m: bit c = A element 4c + m, bit 4 + c = B), and a meta stream of 16 B per block (Mat::d): qh[8] (byte u': the
// index high bits, A's in bits 0-3, B's in 4-7) | scales[4] (as in the file: byte p = sub-block 2p | 2p + 1 << 4) | fp16 d
// | pad. A weight is d (1 + 2 s) (+-grid). As rows_q3k, the GEMV feeds 32 +- grid = the Q6_K value of a Q6_K
// re-encode, with the same scales and arithmetic as rows_q6k: bit-identical to it at 112 / 210 the bytes.
template <int T, int IT, int R, int XP = 0>
__device__ __forceinline__ void rows_iq3s(const uint8_t * const (&rp)[R], const __half * const (&dp)[R], const SX & s,
                                          float (&acc)[R][T]) {
    const int lane = threadIdx.x & 31;
#pragma unroll(IT > 8 ? (T >= 5 ? 1 : 2) : IT) // (4 at T < 5 spills 200-280 B in resid32<*, 17 / 10, IQ3_S>)
    for (int i = 0; i < IT; i++) {
        const int u = lane + 32 * i, b = u >> 3, p = (u >> 1) & 3, w = u & 7;
        int2 qs[R], sd[R];
        int sg[R], qh[R];
#pragma unroll
        for (int k = 0; k < R; k++) {
            const uint8_t * blk = rp[k] + b * Q3_BLK;
            const uint8_t * mp = reinterpret_cast<const uint8_t *>(dp[k]) + b * IQ3S_META;
            qs[k] = ldw2(blk + 8 * w);
            sg[k] = ldw1(blk + 64 + 4 * w);
            qh[k] = ldw1(mp + 4 * (w >> 2));
            sd[k] = ldw2(mp + 8);
        }
        int A[R][4], B[R][4];
        float fA[R], gA[R], fB[R], gB[R];
#pragma unroll
        for (int k = 0; k < R; k++) {
            const float d = __half2float(__ushort_as_half((unsigned short) (sd[k].y & 0xffff)));
            const unsigned sc = (unsigned) sd[k].x >> (8 * p);
            fA[k] = d * (float) (int) (1 + 2 * (sc & 0xfu));
            fB[k] = d * (float) (int) (1 + 2 * ((sc >> 4) & 0xfu));
            gA[k] = 32.0f * fA[k];
            gB[k] = 32.0f * fB[k];
            const unsigned h = (unsigned) qh[k] >> (8 * (w & 3)), x = (unsigned) qs[k].x, y = (unsigned) qs[k].y,
                           sv = (unsigned) sg[k];
#pragma unroll
            for (int c = 0; c < 4; c++) {
                const unsigned ga = iq3s_grid(((x >> (8 * c)) & 0xffu) | (((h >> c) & 1u) << 8));
                const unsigned gb = iq3s_grid(((y >> (8 * c)) & 0xffu) | (((h >> (4 + c)) & 1u) << 8));
                const unsigned ma = ((sv >> c) & 0x01010101u) * 0xffu, mb = ((sv >> (4 + c)) & 0x01010101u) * 0xffu;
                // 32 +- g per byte: (g ^ m) - m is -g in the bytes where m = 0xff (the borrows cancel in the sum)
                A[k][c] = (int) (0x20202020u + (ga ^ ma) - ma);
                B[k][c] = (int) (0x20202020u + (gb ^ mb) - mb);
            }
        }
        accR<T, R, 32 * IT + XP>(s, u, A, B, fA, gA, fB, gB, acc);
    }
}

// rows row0 .. row0+R-1 of m, weight type fixed at compile time
// u0: first unit of a K-slice (multiple of 32) when the activations in s hold only that slice
template <int TYPE>
__device__ __forceinline__ int unit_bytes(int u0) {
    return TYPE == T_Q4K || TYPE == T_IQ4NL ? (u0 >> 3) * 144
         : TYPE == T_IQ4X                 ? (u0 >> 3) * 136 // u0 is a multiple of 32 units: whole block pairs
         : TYPE == T_Q5K                  ? (u0 >> 3) * 176
         : TYPE == T_Q6K                  ? (u0 >> 3) * Q6_BLK
         : TYPE == T_Q3K || TYPE == T_IQ3S ? (u0 >> 3) * Q3_BLK
                                          : 32 * u0;
}
// R rows of one weight type given their row pointers (and, for Q8_0, their scale rows)
template <int T, int IT, int R, int TYPE, int XP = 0, class S = SX>
__device__ __forceinline__ void rows_t(const uint8_t * const (&rp)[R], const __half * const (&dp)[R], const S & s,
                                       float (&acc)[R][T]) {
    if constexpr (TYPE == T_Q4K) rows_q4k<T, IT, R, XP>(rp, s, acc);
    else if constexpr (TYPE == T_Q5K) rows_q5k<T, IT, R, XP>(rp, s, acc);
    else if constexpr (TYPE == T_Q6K) rows_q6k<T, IT, R, XP>(rp, dp, s, acc);
    else if constexpr (TYPE == T_IQ4X) rows_iq4x<T, IT, R, XP>(rp, s, acc);
    else if constexpr (TYPE == T_IQ4NL) rows_iq4nl<T, IT, R, XP>(rp, s, acc);
    else if constexpr (TYPE == T_Q3K) rows_q3k<T, IT, R, XP>(rp, dp, s, acc);
    else if constexpr (TYPE == T_IQ3S) rows_iq3s<T, IT, R, XP>(rp, dp, s, acc);
    else rows_q80<T, IT, R, XP>(rp, dp, s, acc);
}
template <int T, int IT, int R, int TYPE, int XP = 0>
__device__ __forceinline__ void mrows(const Mat & m, int row0, const SX & s, float (&acc)[R][T], int u0 = 0) {
    const uint8_t * rp[R];
    const __half * dp[R];
#pragma unroll
    for (int k = 0; k < R; k++) {
        rp[k] = m.w + (size_t) (row0 + k) * m.rb + unit_bytes<TYPE>(u0);
        // Q6_K / Q3_K: meta (9 halves per block), IQ3_S: meta (8 halves per block), Q8_0: d
        dp[k] = m.d + (size_t) (row0 + k) * m.nd +
                (TYPE == T_Q6K || TYPE == T_Q3K ? (u0 >> 3) * (Q6_META / 2) : TYPE == T_IQ3S ? (u0 >> 3) * (IQ3S_META / 2) : u0);
    }
    rows_t<T, IT, R, TYPE, XP>(rp, dp, s, acc);
}

// Row pairs [lo, hi) of a single-matrix item range, distributed over all warps of the grid.
// out(row0, acc) receives the reduced sums for rows row0, row0 + 1.
template <int T, int IT, int TYPE, int XP = 0, class Out>
__device__ __noinline__ void gemv_pairs(const Mat m, int lo, int hi, Out out) {
    const SX sx = XP ? wv_sx<T, 32 * IT>() : make_sx(nullptr, 32 * IT); // activations already in shared memory
    const int W = nwarps(), gw = gwarp();
    const int first = lo + (((gw - lo) % W) + W) % W;
    for (int it = first; it < hi; it += W) {
        const int r = 2 * (it - lo);
        float a[2][T];
        zero_acc<T, 2>(a);
        mrows<T, IT, 2, TYPE, XP>(m, r, sx, a);
        wsum_acc<T, 2>(a);
        out(r, a);
    }
}

// Runtime dispatch on a matrix's type (uniform across the grid). Every main-layer matrix of Q4_K, Q5_K, Q8_0, IQ4_NL,
// IQ4_XS and Q6_K is in the wv layout and the MTP block is re-quantised to Q4_K, so these read the
// hybrid types and the MTP block only; the loader refuses any matrix they would not read. Plain switches: a capturing
// lambda would take the address of the caller's locals and push them to local memory.
// (mk_cycle's stack frame is the deepest call chain incl. the callees' spills: a spilling GEMV instantiation of any
// type raises the local memory of the whole kernel, ~0.12 MiB per byte. A second out-of-line switch for the newer types
// raises it, 816 -> 1232 B.)
template <int T, int IT, class Out>
__device__ __forceinline__ void gemv_pairs_any(const Mat & m, int lo, int hi, Out out) {
    switch (m.type) {
        case T_Q4K: gemv_pairs<T, IT, T_Q4K>(m, lo, hi, out); break;
        case T_Q3K: gemv_pairs<T, IT, T_Q3K>(m, lo, hi, out); break;
        case T_IQ3S: gemv_pairs<T, IT, T_IQ3S>(m, lo, hi, out); break;
        default: gemv_pairs<T, IT, T_IQ4X>(m, lo, hi, out); break;
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

constexpr int EMBD_RB = D / 256 * 144; // bytes of one Q4_K token_embd row

// token_embd of another type (ModelW::embd_t; files whose token_embd is Q6_K or Q8_0): ggml's rows as stored, element e
// dequantised as ggml does (every product here is exact in fp32: d * sc has at most 18 significant bits, times a 6-bit q)
template <int ET>
__host__ __device__ constexpr int embd_rb() {
    return ET == T_Q4K ? D / 256 * 144 : ET == T_Q6K ? D / 256 * 210 : D / 32 * 34;
}
__device__ __forceinline__ float dequant_q6k_e(const uint8_t * row, int e) { // ggml block_q6_K: ql[128] qh[64] sc[16] d
    const int b = e >> 8, w = e & 255, n = w >> 7, r = w & 127, qq = r >> 5, l = r & 31;
    const uint8_t * blk = row + b * 210;
    const uint8_t ql = blk[64 * n + l + (qq & 1) * 32], qh = blk[128 + 32 * n + l];
    const int q = (int) (((qq & 2) ? (ql >> 4) : (ql & 0xF)) | (((qh >> (2 * qq)) & 3) << 4)) - 32;
    const int sc = (int) reinterpret_cast<const int8_t *>(blk + 192)[8 * n + (l >> 4) + 2 * qq];
    const float d = __half2float(*reinterpret_cast<const __half *>(blk + 208));
    return d * (float) sc * (float) q;
}
__device__ __forceinline__ float dequant_q80_e(const uint8_t * row, int e) { // ggml block_q8_0: d, qs[32]
    const uint8_t * blk = row + (e >> 5) * 34;
    return (float) reinterpret_cast<const int8_t *>(blk + 2)[e & 31] * __half2float(*reinterpret_cast<const __half *>(blk));
}
template <int ET>
__device__ __forceinline__ float dequant_embd(const uint8_t * row, int e) {
    if constexpr (ET == T_Q4K) return dequant_q4k(row, e);
    else if constexpr (ET == T_Q6K) return dequant_q6k_e(row, e);
    else return dequant_q80_e(row, e);
}
// nbytes (a multiple of 4) of the table at byte offset off -> shared memory, 4-byte reads (Q6_K rows are only 8-byte
// aligned); every thread of the block calls it
__device__ __forceinline__ void stage_embd_bytes(size_t off, int nbytes, uint8_t * dst) {
    for (int i = threadIdx.x; i < nbytes / 4; i += blockDim.x)
        reinterpret_cast<unsigned *>(dst)[i] = reinterpret_cast<const unsigned *>(cW.embd + off)[i];
    __syncthreads();
}
// embed_prologue's element e (block s: elements [512 s, 512 s + 512)) for a Q6_K / Q8_0 table, out of line
__device__ __noinline__ float embed_elem_other(int tok, int s, int e) {
    if (cW.embd_t == T_Q6K) {
        stage_embd_bytes((size_t) tok * embd_rb<T_Q6K>() + 2 * s * 210, 2 * 210, smem);
        return dequant_q6k_e(smem, e - 512 * s);
    }
    stage_embd_bytes((size_t) tok * embd_rb<T_Q8_0>() + 16 * s * 34, 16 * 34, smem);
    return dequant_q80_e(smem, e - 512 * s);
}
// a whole Q6_K / Q8_0 token_embd row dequantised into se[0 .. D) (the MTP M0 phases; staged at stage, out of line);
// each thread writes the elements e = threadIdx.x + k NT it reads back
static_assert(2 * D * 4 + 32 * 4 + embd_rb<T_Q8_0>() <= SM_TOTAL, "MTP M0: se, sh, red and a staged token_embd row");
__device__ __noinline__ void embed_row_other(int tok, float * se, uint8_t * stage) {
    if (cW.embd_t == T_Q6K) {
        stage_embd_bytes((size_t) tok * embd_rb<T_Q6K>(), embd_rb<T_Q6K>(), stage);
        for (int e = threadIdx.x; e < D; e += NT) se[e] = dequant_q6k_e(stage, e);
    } else {
        stage_embd_bytes((size_t) tok * embd_rb<T_Q8_0>(), embd_rb<T_Q8_0>(), stage);
        for (int e = threadIdx.x; e < D; e += NT) se[e] = dequant_q80_e(stage, e);
    }
}

// token_embd row -> shared memory in one coalesced read: the table is in mapped host memory, where
// every dependent access pays a PCIe round trip
__device__ __forceinline__ const uint8_t * stage_embd_row(int tok, uint8_t * dst) {
    if (threadIdx.x < EMBD_RB / 16)
        reinterpret_cast<uint4 *>(dst)[threadIdx.x] = reinterpret_cast<const uint4 *>(cW.embd + (size_t) tok * EMBD_RB)[threadIdx.x];
    __syncthreads();
    return dst;
}

// token embedding rows -> x, plus quantised attn_norm input for layer 0.  Block s of row t handles the 16 units of
// super-blocks 2s, 2s+1 (one per warp; 10 blocks per row, U_D / 16): with a single block per row the grid would wait
// on its serial per-unit quantise chain (the row fetch itself is ~free)
static_assert(U_D % EMB_UB == 0 && EMB_UB == NW, "embed_prologue: one unit per warp");
__device__ __forceinline__ void stage_embd_part(int tok, int b0, int nbytes, uint8_t * dst) {
    if (threadIdx.x < nbytes / 16)
        reinterpret_cast<uint4 *>(dst)[threadIdx.x] = reinterpret_cast<const uint4 *>(cW.embd + (size_t) tok * EMBD_RB + b0)[threadIdx.x];
    __syncthreads();
}
// BAT (batched launches): row t's token is its lane's ctl token t - r0 (rows past the lanes: token 0)
template <int T, bool BAT = false>
__device__ __noinline__ void embed_prologue(RowTab rt = RowTab{}) {
    constexpr int NB = U_D / EMB_UB; // blocks per row
#if MK_PAR_WIDE
    if constexpr (T > MAXT) { // (more items than blocks: each block takes items b, b + G, ...; !BAT: the deep pass)
        const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
        const float * w0 = cW.L[0].attn_norm;
#pragma unroll 1
        for (int it = blockIdx.x; it < T * NB; it += gridDim.x) {
            const int t = it / NB, s = it - t * NB;
            if (it != (int) blockIdx.x) __syncthreads(); // (the previous item's staged embedding is read)
            int tok;
            if constexpr (BAT) {
                int r0;
                const int ln = rt_lane(rt, t, r0);
                tok = ln < 0 ? 0 : ((volatile Ctl *) cBs[view_main(ln)].ctl)->tok[t - r0];
            } else {
                tok = ((volatile Ctl *) cB.ctl)->tok[t];
            }
            const int j = EMB_UB * s + warp, e = 32 * j + lane;
            float v;
            if (cW.embd_t == T_Q4K) {
                stage_embd_part(tok, 2 * s * 144, 2 * 144, smem);
                v = dequant_q4k(smem, e - 512 * s);
            } else {
                v = embed_elem_other(tok, s, e);
            }
            cB.x[t * D + e] = v;
            q32_store(cB.qn, t, j, v * w0[e], v * v, true);
        }
        return;
    }
#endif
    const int t = blockIdx.x / NB, s = blockIdx.x - t * NB;
    if (t >= T) return;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    int tok;
    if constexpr (BAT) {
        int r0;
        const int ln = rt_lane(rt, t, r0);
        tok = ln < 0 ? 0 : ((volatile Ctl *) cBs[view_main(ln)].ctl)->tok[t - r0];
    } else {
        tok = ((volatile Ctl *) cB.ctl)->tok[t];
    }
    const float * w0 = cW.L[0].attn_norm;
    const int j = EMB_UB * s + warp, e = 32 * j + lane;
    float v;
    if (cW.embd_t == T_Q4K) {
        // units 16 s .. 16 s + 15 = elements [512 s, 512 s + 512) = Q4_K super-blocks 2 s, 2 s + 1 (288 bytes)
        stage_embd_part(tok, 2 * s * 144, 2 * 144, smem);
        v = dequant_q4k(smem, e - 512 * s);
    } else {
        v = embed_elem_other(tok, s, e);
    }
    cB.x[t * D + e] = v;
    q32_store(cB.qn, t, j, v * w0[e], v * v, true);
}

// ffn_down (K = FF) whole-K rows out of line (half_rows) at the T of mask bit T: inlined into resid32's item loop,
// resid32<2..4,17,*> spills 20-88 B. T = 5
// splits K anyway; the K = 5120 / 6144 roles stay inline (out of line they are slower).
#ifndef MK_FD_NOINL
#define MK_FD_NOINL 0x1c
#endif
// T = 5 ffn_down K-half, out of line: inlined, resid32<5,17,*> spills 136-188 B into this loop. The sums go in and out
// by value (registers): by reference, resid32's a lives in local memory.
template <int T> struct PairT { float a[2][T]; };
template <int T, int IT, int TYPE>
__device__ __noinline__ PairT<T> half_rows(const Mat W, int row0, const SX s, PairT<T> b, int u0) {
    mrows<T, IT, 2, TYPE>(W, row0, s, b.a, u0);
    return b;
}

// Output rows (32 consecutive rows per block item) with optional residual add; the new
// residual value is re-quantised for the next RMS-normed consumer (weight wnext).
template <int T, int IT, int TYPE>
__device__ __noinline__ void resid32(const Mat W, const QBuf in, float * x, bool add, const float * wnext, uint8_t * /*sm*/) {
    uint8_t * const sm = smem;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    float * s_rs = misc(sm);
    float * s_res = misc(sm) + 8;
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
            PairT<T> p;
            float (&a)[2][T] = p.a;
            zero_acc<T, 2>(a);
            p = half_rows<T, U0 / 32, TYPE>(W, 32 * j0 + 2 * warp, s0, p, 0);
#pragma unroll
            for (int t = 0; t < T; t++) {
                park[t * NT + threadIdx.x] = a[0][t];
                park[(T + t) * NT + threadIdx.x] = a[1][t];
            }
            if (j1 < D / 32) {
                zero_acc<T, 2>(a);
                p = half_rows<T, U0 / 32, TYPE>(W, 32 * j1 + 2 * warp, s0, p, 0);
            }
            fill_xs_part<T, U1>(in, U0, s1);
            if (j1 < D / 32) {
                p = half_rows<T, U1 / 32, TYPE>(W, 32 * j1 + 2 * warp, s1, p, U0);
                epilogue(j1, a);
            }
#pragma unroll
            for (int t = 0; t < T; t++) {
                a[0][t] = park[t * NT + threadIdx.x];
                a[1][t] = park[(T + t) * NT + threadIdx.x];
            }
            p = half_rows<T, U1 / 32, TYPE>(W, 32 * j0 + 2 * warp, s1, p, U0);
            epilogue(j0, a);
        }
    } else {
        const SX sx = make_sx(sm, in.U);
        fill_xs<T>(in, 0, false, sx, s_rs);
        for (int j = blockIdx.x; j < D / 32; j += gridDim.x) {
            PairT<T> p;
            float (&a)[2][T] = p.a;
            zero_acc<T, 2>(a);
            if constexpr (IT == 17 && ((MK_FD_NOINL >> T) & 1)) p = half_rows<T, IT, TYPE>(W, 32 * j + 2 * warp, sx, p, 0);
            else mrows<T, IT, 2, TYPE>(W, 32 * j + 2 * warp, sx, a);
            epilogue(j, a);
        }
    }
}

template <int T, int IT>
__device__ __forceinline__ void resid_any(const Mat & W, const QBuf & in, float * x, bool add, const float * wnext, uint8_t * sm) {
    switch (W.type) {
        case T_Q4K: resid32<T, IT, T_Q4K>(W, in, x, add, wnext, sm); break;
        case T_Q3K: resid32<T, IT, T_Q3K>(W, in, x, add, wnext, sm); break;
        case T_IQ3S: resid32<T, IT, T_IQ3S>(W, in, x, add, wnext, sm); break;
        default: resid32<T, IT, T_IQ4X>(W, in, x, add, wnext, sm); break;
    }
}

// FFN gate/up: warp item = 16 consecutive rows of both matrices (4 rows in flight sharing the
// activations) -> silu(g)*u quantised per 16-chunk for the down projection.
template <int T>
__device__ __noinline__ void ffn_gu_q4k(const LayerW & W, uint8_t * /*sm*/) {
    uint8_t * const sm = smem;
    const int lane = threadIdx.x & 31;
    const SX sx = make_sx(sm, U_D);
    const float * s_rs = misc(sm);
    fill_xs<T>(cB.qn, 0, true, sx, misc(sm));
    const uint8_t * gw = W.fg.w;
    const uint8_t * uw = W.fu.w;
    const int rb = W.fg.rb;
    for (int j = gwarp(); j < FF / 16; j += nwarps()) {
        float act[T];
#pragma unroll
        for (int t = 0; t < T; t++) act[t] = 0.f;
        // one row per matrix per step at every T (a 2 + 2-row step at T <= 2 spills and is slower)
        constexpr int RG = 1;
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

template <int T> struct LaneT { float v[T]; };

// Rows 16 j .. 16 j + 15 of m against the normed activations in shared memory: lane r gets row
// 16 j + r (lanes 16-31: 0). As many rows per step as ffn_gu_q4k shares activations across.
template <int T, int TYPE, int XP = 0>
__device__ __noinline__ LaneT<T> rows16(const Mat m, int j) {
    const int lane = threadIdx.x & 31;
    const SX sx = XP ? wv_sx<T, U_D>() : make_sx(nullptr, U_D);
    const float * s_rs = misc(nullptr);
    constexpr int R = 2; // at every T (4 rows at T <= 2 spill 44-160 B)
    LaneT<T> res;
#pragma unroll
    for (int t = 0; t < T; t++) res.v[t] = 0.f;
#pragma unroll 1
    for (int r = 0; r < 16; r += R) {
        float a[R][T];
        zero_acc<T, R>(a);
        mrows<T, 5, R, TYPE, XP>(m, 16 * j + r, sx, a);
        wsum_acc<T, R>(a);
#pragma unroll
        for (int k = 0; k < R; k++)
            if (lane == r + k) {
#pragma unroll
                for (int t = 0; t < T; t++) res.v[t] = a[k][t] * s_rs[t];
            }
    }
    return res;
}

template <int T>
__device__ __forceinline__ LaneT<T> rows16_any(const Mat & m, int j) {
    switch (m.type) {
        case T_Q4K: return rows16<T, T_Q4K>(m, j);
        case T_Q3K: return rows16<T, T_Q3K>(m, j);
        case T_IQ3S: return rows16<T, T_IQ3S>(m, j);
        default: return rows16<T, T_IQ4X>(m, j);
    }
}

// ffn_gu_q4k for any other gate / up types (each may be any type): gate and up rows of a warp item
// are computed by one call per matrix, so each weight type gets its own register allocation.
template <int T>
__device__ __noinline__ void ffn_gu_any(const LayerW & W, uint8_t * /*sm*/) {
    uint8_t * const sm = smem;
    const int lane = threadIdx.x & 31;
    const SX sx = make_sx(sm, U_D);
    fill_xs<T>(cB.qn, 0, true, sx, misc(sm));
    for (int j = gwarp(); j < FF / 16; j += nwarps()) {
        const LaneT<T> g = rows16_any<T>(W.fg, j);
        const LaneT<T> u = rows16_any<T>(W.fu, j);
        float act[T];
#pragma unroll
        for (int t = 0; t < T; t++) act[t] = lane < 16 ? silu(g.v[t]) * u.v[t] : 0.f;
#pragma unroll
        for (int t = 0; t < T; t += 2) {
            const float other = __shfl_sync(0xffffffffu, act[t + 1 < T ? t + 1 : t], lane & 15);
            const float v = lane < 16 ? act[t] : other;
            q16_store(cB.qa, lane < 16 ? t : t + 1, j, v, lane < 16 || t + 1 < T);
        }
    }
}

template <int T>
__device__ __forceinline__ void ffn_gu(const LayerW & W, uint8_t * sm) {
    if (W.fg.type == T_Q4K && W.fu.type == T_Q4K) ffn_gu_q4k<T>(W, sm);
    else ffn_gu_any<T>(W, sm);
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

// One store functor type for every gemv_pairs call site. A lambda per site would make one gemv_pairs instance per
// (T, type, site): ~47% of the module's PTX, and ptxas's memory grows with it (> 25 GB).
// Rows t of base (or, ring >= 0, the conv ring rows (ring + t) % CONV_RING), scaled by
// the RMS-norm scale s_rs[t] (dn_proj's and attn_proj's stores).
struct PairOut {
    float * base;
    int stride, ring;
    template <int T> __device__ __forceinline__ void operator()(int r, const float (&a)[2][T]) const {
        const int lane = threadIdx.x & 31, t = lane >> 1;
        const float * s_rs = misc(nullptr);
        // selects, not a store per t (merged into a runtime index into a, gemv_pairs' a would go to local memory)
        float v = 0.f;
#pragma unroll
        for (int k = 0; k < T; k++) v = t == k ? ((lane & 1) ? a[1][k] : a[0][k]) : v;
        if (t < T) {
            const int row = ring >= 0 ? (ring + t) % CONV_RING : t;
            base[row * stride + r + (lane & 1)] = v * s_rs[t];
        }
    }
};

// delta-net input projections: alpha/beta (fp32), qkv -> conv ring, z
template <int T>
__device__ __noinline__ void dn_proj(const LayerW & W, int dl, int n, int dn_c, uint8_t * /*sm*/) {
    uint8_t * const sm = smem;
    const int lane = threadIdx.x & 31;
    const SX sx = make_sx(sm, U_D);
    float * s_rs = misc(sm);
    fill_xs<T>(cB.qn, 0, true, sx, s_rs);
    float * conv = cB.conv + (size_t) dl * CONV_RING * QKV;
    const int n_ab = AB_SPLIT * 2 * NVH, n_qkv = QKV / 2, n_z = VD / 2;
    // alpha / beta: fp32 rows with the norm weight folded in (dot with the raw residual), each
    // row split into AB_SPLIT K-chunks so these heavier items do not form a tail
    for (int it = gwarp(); it < n_ab; it += nwarps()) {
        const int row = it % (2 * NVH), ch = it / (2 * NVH);
        float acc[T];
#pragma unroll
        for (int t = 0; t < T; t++) acc[t] = 0.f;
        if (W.abq) { // Q8_0 rows: the folded fp32 weights (q * d) * attn_norm, formed on the fly (same values)
            const int8_t * qr = W.abq + (size_t) row * D;
            const __half * dr = W.abd + (size_t) row * (D / 32);
            for (int k = ch * (D / AB_SPLIT) + lane * 4; k < (ch + 1) * (D / AB_SPLIT); k += 128) {
                const int qw = ldw1(qr + k);
                const float d = __half2float(dr[k >> 5]);
                const float4 fn = __ldg(reinterpret_cast<const float4 *>(W.attn_norm + k));
                float4 w;
                w.x = ((float) (int8_t) (qw) * d) * fn.x;
                w.y = ((float) (int8_t) (qw >> 8) * d) * fn.y;
                w.z = ((float) (int8_t) (qw >> 16) * d) * fn.z;
                w.w = ((float) (int8_t) (qw >> 24) * d) * fn.w;
#pragma unroll
                for (int t = 0; t < T; t++) {
                    const float4 xv = __ldcg(reinterpret_cast<const float4 *>(cB.x + t * D + k));
                    acc[t] += w.x * xv.x + w.y * xv.y + w.z * xv.z + w.w * xv.w;
                }
            }
        } else {
            const float * wr = W.ab + (size_t) row * D;
            for (int k = ch * (D / AB_SPLIT) + lane * 4; k < (ch + 1) * (D / AB_SPLIT); k += 128) {
                const float4 w = ldw4f(wr + k);
#pragma unroll
                for (int t = 0; t < T; t++) {
                    const float4 xv = __ldcg(reinterpret_cast<const float4 *>(cB.x + t * D + k));
                    acc[t] += w.x * xv.x + w.y * xv.y + w.z * xv.z + w.w * xv.w;
                }
            }
        }
#pragma unroll
        for (int t = 0; t < T; t++) acc[t] = wsum(acc[t]) * s_rs[t];
        store_lane_t<T>(cB.ab + ch * RMAX * 2 * NVH, 2 * NVH, row, acc);
    }
    gemv_pairs_any<T, 5>(W.qkv, n_ab, n_ab + n_qkv, PairOut{conv, QKV, n});
    gemv_pairs_any<T, 5>(W.gate, n_ab + n_qkv, n_ab + n_qkv + n_z, PairOut{cB.z, VD, -1});
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

// float index of state element (col, row) within a head's 128 x 128 block. The DeltaNet state of each head (64 KB,
// logical S[col][row], col = value, row = key dim) is permuted so that a warp's float4 loads are runs of whole sectors:
// float4 (8 w + i) 32 + 4 (col & 7) + r holds rows 32 r + 4 i .. + 3 of col (w = col >> 3; dn_core's lanes read it in
// 64-byte runs); upstream's [col][row] gave a warp's float4 load / store 32 lines 512 B apart (50% sector efficiency).
// dn_core and pk_dn_scan (prefill) address it through dn_sidx; every other user copies whole layers.
__host__ __device__ __forceinline__ int dn_sidx(int col, int row) {
    return (((col >> 3) * 256 + ((row >> 2) & 7) * 32 + (col & 7) * 4 + (row >> 5)) << 2) + (row & 3);
}

// dn_core's recurrence ops in the exact forms cicc / ptxas gave the plain expressions in every instantiation (d16w1 PTX
// and SASS): s.x k.x + s.y k.y + s.z k.z + s.w k.w and g s + k dlt; _rn intrinsics pin them where the code is reordered
__device__ __forceinline__ float dn_dot4(const float4 & s, const float4 & k) {
    return __fmaf_rn(k.w, s.w, __fmaf_rn(k.z, s.z, __fmaf_rn(k.x, s.x, __fmul_rn(k.y, s.y))));
}
__device__ __forceinline__ float4 dn_upd4(const float4 & s, const float4 & k, float g, float dlt) {
    return make_float4(__fmaf_rn(g, s.x, __fmul_rn(k.x, dlt)), __fmaf_rn(dlt, k.y, __fmul_rn(g, s.y)),
                       __fmaf_rn(dlt, k.z, __fmul_rn(g, s.z)), __fmaf_rn(dlt, k.w, __fmul_rn(g, s.w)));
}
// dn_core's sums over a column's 128 rows (lane pair hf 0 / 1 = rows 32 r + 16 hf + [0, 16) of row quarter r): per vector c
// the one-thread 32-row chain (((0 + p0) + p1) ... + p7) split over the pair (hf 0: p0 .. p3; hf 1 continues hf 0's sum
// with p4 .. p7; d[c][i] = the half's 4 dn_dot4 terms), then the quarter tree (s0 + s1) + (s2 + s3) by lane xor 2, 4.
// Valid on hf 1 lanes
template <int NC>
__device__ __forceinline__ void dn_sum2(const float (&d)[NC][4], float (&y)[NC]) {
    float x[NC];
#pragma unroll
    for (int c = 0; c < NC; c++) x[c] = __fadd_rn(__fadd_rn(__fadd_rn(__fadd_rn(0.f, d[c][0]), d[c][1]), d[c][2]), d[c][3]);
#pragma unroll
    for (int c = 0; c < NC; c++) x[c] = __shfl_xor_sync(0xffffffffu, x[c], 1);
#pragma unroll
    for (int c = 0; c < NC; c++) y[c] = __fadd_rn(__fadd_rn(__fadd_rn(__fadd_rn(x[c], d[c][0]), d[c][1]), d[c][2]), d[c][3]);
#pragma unroll
    for (int c = 0; c < NC; c++) y[c] = __fadd_rn(y[c], __shfl_xor_sync(0xffffffffu, y[c], 2));
#pragma unroll
    for (int c = 0; c < NC; c++) y[c] = __fadd_rn(y[c], __shfl_xor_sync(0xffffffffu, y[c], 4));
}
// both columns' kv = S k (S[4 c + i] . k[i]) on every lane (hf 0 takes hf 1's sums)
__device__ __forceinline__ void dn_kv2(const float4 (&S)[8], const float4 (&k)[4], float & kv0, float & kv1) {
    float d[2][4], y[2];
#pragma unroll
    for (int i = 0; i < 4; i++) {
        d[0][i] = dn_dot4(S[i], k[i]);
        d[1][i] = dn_dot4(S[4 + i], k[i]);
    }
    dn_sum2<2>(d, y);
    const float t0 = __shfl_xor_sync(0xffffffffu, y[0], 1), t1 = __shfl_xor_sync(0xffffffffu, y[1], 1);
    const bool h1 = threadIdx.x & 1;
    kv0 = h1 ? y[0] : t0;
    kv1 = h1 ? y[1] : t1;
}
// one event's update of both columns: dlt = (v - g kv) beta per column, S = g S + k dlt (dn_upd4)
__device__ __forceinline__ void dn_upd2(float4 (&S)[8], const float4 (&k)[4], float g, float beta, float2 v, float kv0, float kv1) {
    const float d0 = __fmul_rn(__fmaf_rn(-g, kv0, v.x), beta), d1 = __fmul_rn(__fmaf_rn(-g, kv1, v.y), beta);
#pragma unroll
    for (int i = 0; i < 4; i++) {
        S[i] = dn_upd4(S[i], k[i], g, d0);
        S[4 + i] = dn_upd4(S[4 + i], k[i], g, d1);
    }
}
// q32_store(q, t[i], j, v[i], 0.f, false) for rows i < n of R, one warp: the rows' shuffle chains and divisions
// interleaved, each row's ops and stores those of q32_store (keep the two in step)
template <int R>
__device__ __forceinline__ void dn_q32_store(const QBuf & q, const int (&t)[R], int j, const float (&v)[R], int n) {
    const int lane = threadIdx.x & 31;
    const int b = j >> 3, p = (j >> 1) & 3, part = j & 1, hh = lane >> 4, byte = lane & 15;
    float am[R];
    int qs[R];
#pragma unroll
    for (int i = 0; i < R; i++) am[i] = fabsf(v[i]);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) // wmax
#pragma unroll
        for (int i = 0; i < R; i++) am[i] = fmaxf(am[i], __shfl_xor_sync(0xffffffffu, am[i], o));
#pragma unroll
    for (int i = 0; i < R; i++) {
        const float d = am[i] / 127.0f;
        qs[i] = am[i] == 0.0f ? 0 : (int) roundf(v[i] / d);
        am[i] = d;
        if (i < n) (part ? q.xb : q.xa)[(t[i] * q.U + 8 * b + 2 * p + hh) * 16 + byte] = (int8_t) qs[i];
    }
#pragma unroll
    for (int o = 8; o > 0; o >>= 1)
#pragma unroll
        for (int i = 0; i < R; i++) qs[i] += __shfl_xor_sync(0xffffffffu, qs[i], o);
    if (byte == 0) {
#pragma unroll
        for (int i = 0; i < R; i++) {
            if (i < n) {
                const int gi = t[i] * q.U + 8 * b + 2 * p + hh;
                reinterpret_cast<__half *>(q.meta + gi)[part] = __float2half_rn(am[i]);
                reinterpret_cast<short *>(q.meta + gi)[2 + part] = (short) qs[i];
            }
        }
    }
}

// the gate z of head h (output norm only) into s_z: this thread's loads, then its stores. FULL (T 128 a multiple of NT:
// every i < T 128, as tid < NT): no guards; else zr defined on both paths. A guarded load into an otherwise undefined zr
// (ptxas: a predicated def, the register's old value live into it) kept two K5 replay-loop shuffle results live across
// k5_wait: 2 spill stores per replayed event (dn_core<8,true>: 66 MB STL per T8 launch)
template <int T>
__device__ __forceinline__ void dn_z(float * s_z, int h) {
    constexpr int NZ = (T * 128 + NT - 1) / NT;
    constexpr bool FULL = NZ * NT == T * 128;
    const int tid = threadIdx.x;
    float zr[NZ];
#pragma unroll
    for (int k = 0; k < NZ; k++) {
        const int i = tid + k * NT;
        zr[k] = 0.f;
        if (FULL || i < T * 128) zr[k] = __ldcg(cB.z + (i >> 7) * VD + h * 128 + (i & 127));
    }
#pragma unroll
    for (int k = 0; k < NZ; k++) {
        const int i = tid + k * NT;
        if (FULL || i < T * 128) s_z[i] = zr[k];
    }
}
// dn_core's new-token inputs: conv1d + silu -> s_raw (tid < 384), alpha / beta -> g / beta in s_gb (the next T), z ->
// s_z (all); RP (K5 false) also the replayed events' g / beta (tid 448 ..). Each thread's branch issues its loads,
// stores its z, then computes: nothing crosses from one branch to another, since every value that did in earlier
// versions (z, then all of the branches' loads) went to local memory around the silu / softplus divisions' slow-path
// calls
template <int T, bool RP>
__device__ __forceinline__ void dn_inputs(const LayerW & W, int h, int kh, int n, int np, int dn_c, const float * conv,
                                          const float * ring, float * s_raw, float * s_gb, float * s_z) {
    const int tid = threadIdx.x;
    if (tid < 384) {
        const int ch = tid < 128 ? kh * 128 + tid : (tid < 256 ? 2048 + kh * 128 + (tid - 128) : 4096 + h * 128 + (tid - 256));
        const float4 w = *reinterpret_cast<const float4 *>(W.conv + ch * 4);
        float raw[T + 3];
#pragma unroll
        for (int j = 0; j < T + 3; j++) {
            const int p = n - 3 + j;
            raw[j] = p >= 0 ? __ldcg(conv + (p % CONV_RING) * QKV + ch) : 0.f;
        }
        dn_z<T>(s_z, h);
#pragma unroll
        for (int t = 0; t < T; t++)
            s_raw[t * 384 + tid] = silu(w.x * raw[t] + w.y * raw[t + 1] + w.z * raw[t + 2] + w.w * raw[t + 3]);
    } else if (tid < 384 + T) {
        const int t = tid - 384;
        float pa[AB_SPLIT], pb[AB_SPLIT];
#pragma unroll
        for (int c = 0; c < AB_SPLIT; c++) {
            pa[c] = __ldcg(cB.ab + (c * RMAX + t) * 2 * NVH + h);
            pb[c] = __ldcg(cB.ab + (c * RMAX + t) * 2 * NVH + NVH + h);
        }
        const float dtv = W.dt[h], av = W.A[h];
        dn_z<T>(s_z, h);
        float al = 0.f, be = 0.f;
#pragma unroll
        for (int c = 0; c < AB_SPLIT; c++) {
            al += pa[c];
            be += pb[c];
        }
        float sp = al + dtv;
        sp = sp > 20.0f ? sp : logf(1.0f + expf(sp));
        s_gb[2 * (np + t)] = expf(sp * av);
        s_gb[2 * (np + t) + 1] = 1.0f / (1.0f + expf(-be));
    } else {
        const int e = tid - 448;
        const bool rp = RP && e >= 0 && e < np;
        float rg = 0.f, rb = 0.f;
        if (rp) {
            const float * slot = ring + (size_t) ((dn_c + e) % KV_RING) * KVGB;
            rg = __ldcg(slot + KVGB_G + h);
            rb = __ldcg(slot + KVGB_B + h);
        }
        dn_z<T>(s_z, h);
        if (rp) {
            s_gb[2 * e] = rg;
            s_gb[2 * e + 1] = rb;
        }
    }
}

// K5 commit point of head h. (1) The state goes out L2 evict_first: dead for the rest of the launch, its lines
// (dn_proj's evict_last warm-up, or the newest normal lines) would otherwise outlive the resid_sout weights prefetched
// below and at its phase top; as the first victims they are written back by the prefetch's allocations while DRAM idles
// through dn_core (~13.5 us per layer at T8, ~22.7 at T16), not in resid_sout's stream. (2) The block's 1/NVH share of
// a resid_sout prefetch: each resid_sout warp (resid_wv / resid_wv_ck: item = tile, split-K chunk kc, super-blocks ch
// kc .. + ch - 1) fetches its first unit at the phase top; this adds the first L lines of its second, both within
// DN_PF_ITEM bytes per item (1280 items, ~5 MB of the 6 MB L2): Q4_K 14 of 18 lines, Q5_K 10 of 22, Q6_K 5 of 27,
// IQ4_XS 15 of 17, Q8_0 none. Normal priority. Issued here, after the commit, rather than by the 34 headless blocks at
// the K5 release, so that its allocations displace the state lines. Hints only: no value changes
__device__ __forceinline__ void dn_commit(float4 * Sg, const float4 (&S)[8], const Mat & sout, int h) {
    const unsigned long long pol = l2_evict_first();
#pragma unroll
    for (int i = 0; i < 8; i++)
        asm volatile("st.global.L2::cache_hint.v4.f32 [%0], {%1, %2, %3, %4}, %5;" ::"l"(Sg + 32 * (i & 3) + 4 * (i >> 2)),
                     "f"(S[i].x), "f"(S[i].y), "f"(S[i].z), "f"(S[i].w), "l"(pol));
    const int ty = sout.type, nsb = sout.K >> 8, ch = nsb / WV_S;
    // (wv.cuh's wv_ubytes, defined after this point: keep the two in step)
    const int ub = ty == T_Q6K ? Q6F_UNIT
                               : 16 * (ty == T_Q5K ? 176 : ty == T_IQ4X ? 136 : ty == T_Q8_0 ? 256 : ty == T_Q3K || ty == T_IQ3S ? Q3_BLK : 144);
#if MK_L2_TUNE
    const int L = min(((ch - 1) * ub + 127) >> 7, (cL2.dn - ub) >> 7); // (up to the end of the item's ch units)
#else
    const int L = min((ub + 127) >> 7, (DN_PF_ITEM - ub) >> 7);
#endif
    if (!sout.fm || ch < 2 || L <= 0) return;
    constexpr int NI = D / 16 * WV_S;
    for (int j = (int) threadIdx.x * NVH + h; j < NI * L; j += NT * NVH) { // (blocks interleaved: ~NI L / NVH lines each)
        const int it = j % NI, l = j / NI;
        pf_l2s(sout.w + ((size_t) (it / WV_S) * nsb + ch * (it % WV_S) + 1) * ub + 128 * l);
    }
}

// delta-net core: one block per value head (48 blocks).  conv1d+silu, l2-norm, gated delta
// rule (with replay of pending accepted tokens), gated rms-norm, quantise for ssm_out.
// K5 (solo launches): the head this block took at its dn_proj arrival (k5_arrive), the replay before the
// barrier completes. K5 false: head hh (batched launches: one (lane, head) item after a grid barrier; hh < 0: blockIdx.x),
// the same events in the same order (bit-identical).
template <int T, bool K5 = true>
__device__ __noinline__ void dn_core(const LayerW & W, int dl, int n, int dn_c, int n_commit, uint8_t * /*sm*/, int hh = -1) {
    uint8_t * const sm = smem;
    int h;
    if constexpr (K5) {
        h = s_k5h;
        if (h < 0) {
            k5_wait();
            return;
        }
    } else
    {
        h = hh >= 0 ? hh : (int) blockIdx.x;
        if (h >= NVH) return;
    }
    const int kh = h % NKH;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    // lane 8 cp + 2 r + hf owns state columns c0 = 8 warp + 2 cp and c0 + 1, rows [32 r + 16 hf, + 16) (S stored as
    // [col][row]): S[4 c + i] = column c0 + c, rows 32 r + 16 hf + 4 i .. + 3. One k / q float4 serves both columns (half the
    // shared-memory wavefronts of one column x 32 rows); a column's 32-row sums are split over the lane pair (dn_sum2)
    const int hf = lane & 1, r = (lane >> 1) & 3, c0 = 8 * warp + 2 * (lane >> 3);
    constexpr int NE = T + MAXD - 1;     // max events: up to MAXD - 1 replayed accepted drafts + T new tokens
    constexpr int QS = 4 * 36;           // q / k with a 4-float pad per 32-row quarter (bank spread)
    float * s_q = reinterpret_cast<float *>(sm); // [NE][QS]
    float * s_k = s_q + NE * QS;                 // [NE][QS]
    float * s_v = s_k + NE * QS;                 // [NE][128]
    float * s_o = s_v + NE * 128;                // [T][128]
    float * s_z = s_o + T * 128;                 // [T][128]
    float * s_gb = s_z + T * 128;                // [NE][2]
    constexpr int GBF = 2 * NE > 32 ? (2 * NE + 3) & ~3 : 32;
    float * s_raw = s_gb + GBF;                  // [T][384] conv outputs (silu), 16-byte aligned
    static_assert((2 * NE * QS + NE * 128 + 2 * T * 128 + GBF + T * 384) * 4 <= SM_TOTAL, "dn_core shared memory");

    // S[i] at Sg[32 (i & 3) + 4 (i >> 2)] (dn_sidx)
    float4 * const Sg = reinterpret_cast<float4 *>(cB.dstate + ((size_t) dl * NVH + h) * 128 * 128) + warp * 256 + 128 * hf +
                        8 * (lane >> 3) + r;
    float4 S[8];
#pragma unroll
    for (int i = 0; i < 8; i++) S[i] = __ldcg(Sg + 32 * (i & 3) + 4 * (i >> 2));

    const int np = n - dn_c; // replayed (accepted, not yet committed) tokens
    float * ring = cB.kvgb + (size_t) dl * KV_RING * KVGB;
    const float * conv = cB.conv + (size_t) dl * CONV_RING * QKV;

    if constexpr (K5) {
        // ---- K5 Pre (before the dn_proj barrier completes): the replayed events' inputs come from earlier launches (kvgb
        // ring), so load them and run the replay now; the events' arithmetic and order are those of the single loop below ----
        if (tid >= 448 && tid < 448 + np) {
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
        __syncthreads();
        for (int e = 0; e < np; e++) { // (the commit event np + n_commit - 1 is never a replayed one)
            const float g = s_gb[2 * e], beta = s_gb[2 * e + 1];
            const float4 * kq = reinterpret_cast<const float4 *>(s_k + e * QS + r * 36 + 16 * hf);
            float4 k4[4];
    #pragma unroll
            for (int i = 0; i < 4; i++) k4[i] = kq[i];
            float kv0, kv1;
            dn_kv2(S, k4, kv0, kv1);
            dn_upd2(S, k4, g, beta, *reinterpret_cast<const float2 *>(s_v + e * 128 + c0), kv0, kv1);
        }
        k5_wait(); // every block's dn_proj outputs (conv rows n .. n + T - 1, z, alpha / beta partials) are visible from here

        // ---- K5 Post: the new tokens' inputs ----
        dn_inputs<T, false>(W, h, kh, n, np, dn_c, conv, ring, s_raw, s_gb, s_z);
        __syncthreads();
    } else
    {
        // ---- prologue: every input of every event, one round of loads ----
        dn_inputs<T, true>(W, h, kh, n, np, dn_c, conv, ring, s_raw, s_gb, s_z);
        for (int i = tid; i < np * 128; i += NT) { // replayed k / v
            const int e = i >> 7, c = i & 127;
            const float * slot = ring + (size_t) ((dn_c + e) % KV_RING) * KVGB;
            s_k[e * QS + (c >> 5) * 36 + (c & 31)] = __ldcg(slot + kh * 128 + c);
            s_v[e * 128 + c] = __ldcg(slot + KVGB_V + h * 128 + c);
        }
        __syncthreads();
    }

    // l2-normalise q and k of the new tokens (warp 2t: q of token t, warp 2t+1: k), copy v
    for (int w = warp; w < 2 * T; w += NW) {
        const int t = w >> 1, isk = w & 1;
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

    const float sn = W.snorm[(warp & 3) * 32 + lane]; // the gated rms-norm weight of this warp's output tasks (below)
    if constexpr (K5) {
        // ---- gated delta rule over the new tokens' events (the replayed ones ran before the wait), pipelined:
        // step e applies event e, then forms o_e and the next event's kv from the new state together (independent dot
        // products and shuffle chains) while the next event's k / v / g / beta load, so one event's chain no longer
        // waits for the previous event's o reduction and loads. The commit event's state is stored between the loop's
        // two parts (no store in its body). Per value the plain loop's ops in its order (dn_dot4 / dn_upd4 / dn_sum2).
        const int e1 = np + T, ec = np + n_commit - 1; // (n_commit >= 1: ec in [np, e1))
        float4 kn[4]; // k (the lane's 16 rows) of the event the next step applies
        float kv0, kv1, g, beta;
        float2 v;
        {
            const float4 * kq = reinterpret_cast<const float4 *>(s_k + np * QS + r * 36 + 16 * hf);
    #pragma unroll
            for (int i = 0; i < 4; i++) kn[i] = kq[i];
            dn_kv2(S, kn, kv0, kv1);
            g = s_gb[2 * np];
            beta = s_gb[2 * np + 1];
            v = *reinterpret_cast<const float2 *>(s_v + np * 128 + c0);
        }
        int e = np;
    #pragma unroll 1
        for (int part = 0; part < 2; part++) { // events np .. ec, the commit store, events ec + 1 .. e1 - 1
            for (const int ee = part ? e1 : ec + 1; e < ee; e++) {
                const float4 * qq = reinterpret_cast<const float4 *>(s_q + e * QS + r * 36 + 16 * hf);
                float4 q4[4]; // (issued ahead of the update, which does not need them)
    #pragma unroll
                for (int i = 0; i < 4; i++) q4[i] = qq[i];
                dn_upd2(S, kn, g, beta, v, kv0, kv1);
                const int en = e + 1 < e1 ? e + 1 : e; // (the last event's next kv is not used)
                const float4 * kq = reinterpret_cast<const float4 *>(s_k + en * QS + r * 36 + 16 * hf);
                g = s_gb[2 * en];
                beta = s_gb[2 * en + 1];
                v = *reinterpret_cast<const float2 *>(s_v + en * 128 + c0);
                float d[4][4], y[4]; // o_e of both columns, then the next kv of both
    #pragma unroll
                for (int i = 0; i < 4; i++) {
                    kn[i] = kq[i];
                    d[0][i] = dn_dot4(S[i], q4[i]);
                    d[1][i] = dn_dot4(S[4 + i], q4[i]);
                    d[2][i] = dn_dot4(S[i], kn[i]);
                    d[3][i] = dn_dot4(S[4 + i], kn[i]);
                }
                dn_sum2<4>(d, y);
                const float t0 = __shfl_xor_sync(0xffffffffu, y[2], 1), t1 = __shfl_xor_sync(0xffffffffu, y[3], 1);
                kv0 = hf ? y[2] : t0;
                kv1 = hf ? y[3] : t1;
                if (hf && r == 0) *reinterpret_cast<float2 *>(s_o + (e - np) * 128 + c0) = make_float2(y[0], y[1]);
            }
            if (part == 0) dn_commit(Sg, S, W.sout, h);
        }
    } else
    {
        // ---- gated delta rule over all events; inputs are in shared memory, no barriers needed ----
        for (int e = 0; e < np + T; e++) {
            const float g = s_gb[2 * e], beta = s_gb[2 * e + 1];
            const float4 * kq = reinterpret_cast<const float4 *>(s_k + e * QS + r * 36 + 16 * hf);
            float4 k4[4];
    #pragma unroll
            for (int i = 0; i < 4; i++) k4[i] = kq[i];
            float kv0, kv1;
            dn_kv2(S, k4, kv0, kv1);
            dn_upd2(S, k4, g, beta, *reinterpret_cast<const float2 *>(s_v + e * 128 + c0), kv0, kv1);
            if (e >= np) {
                const float4 * qq = reinterpret_cast<const float4 *>(s_q + e * QS + r * 36 + 16 * hf);
                float d[2][4], y[2];
    #pragma unroll
                for (int i = 0; i < 4; i++) {
                    const float4 q4 = qq[i];
                    d[0][i] = dn_dot4(S[i], q4);
                    d[1][i] = dn_dot4(S[4 + i], q4);
                }
                dn_sum2<2>(d, y);
                if (hf && r == 0) *reinterpret_cast<float2 *>(s_o + (e - np) * 128 + c0) = make_float2(y[0], y[1]);
            }
            if (e == np + n_commit - 1) {
    #pragma unroll
                for (int i = 0; i < 8; i++) Sg[32 * (i & 3) + 4 * (i >> 2)] = S[i];
            }
        }
    }
    __syncthreads();

    // gated rms-norm + quantise: warp w -> token w / 4, 32-element block w % 4 of this head. A warp's tasks (w = warp +
    // NW i: one block jb, the one snorm value loaded before the events) run together, their shuffle chains and
    // divisions interleaved; per value the ops of the one-task-at-a-time loop (wsum, / 128 + EPS, q32_store)
    if (warp < 4 * T) {
        constexpr int R = (4 * T + NW - 1) / NW;
        const int jb = warp & 3, c = jb * 32 + lane;
        int tt[R];
        float ss[R], y[R];
    #pragma unroll
        for (int i = 0; i < R; i++) {
            tt[i] = min((warp + NW * i) >> 2, T - 1); // (only the last can be absent: reads token T - 1, no store)
            const float4 v = *reinterpret_cast<const float4 *>(s_o + tt[i] * 128 + lane * 4);
            ss[i] = __fmaf_rn(v.w, v.w, __fmaf_rn(v.z, v.z, __fmaf_rn(v.x, v.x, __fmul_rn(v.y, v.y))));
        }
    #pragma unroll
        for (int o = 16; o > 0; o >>= 1) // wsum
    #pragma unroll
            for (int i = 0; i < R; i++) ss[i] = __fadd_rn(ss[i], __shfl_xor_sync(0xffffffffu, ss[i], o));
    #pragma unroll
        for (int i = 0; i < R; i++) {
            const float rs = rsqrtf(__fmaf_rn(ss[i], 0.0078125f, EPS)); // ss / 128 + EPS
            y[i] = s_o[tt[i] * 128 + c] * rs * sn * silu(s_z[tt[i] * 128 + c]);
        }
        dn_q32_store<R>(cB.qo, tt, 4 * h + jb, y, warp + NW * (R - 1) < 4 * T ? R : R - 1);
    }
}

// q8_0 store of one KV row: lane holds values 8 lane .. 8 lane + 7 (4 lanes per 32-value block). Rounds v * (127 / am) to
// even, not q32_store's roundf(v / d): the committed KV bytes depend on it
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

// q4 store of one KV row (row = HD / 2 bytes), q4_0 rounding: per 32-value block d = (the value of largest
// magnitude) / -8, nibble = min(15, (int) (x / d + 8.5)), value = (nibble - 8) d. Lane holds values 8 lane .. + 7 and
// writes one word: low nibbles = values 0..3, high nibbles = values 4..7. Ties of |x| go to the larger x (the same
// in all 4 lanes of a block). Bit-identical to MK_KV_EMU mode "4a".
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

// one K (w = 0) or V (w = 1) row (kv head g, position pos) of attention slot `slot` in its MK_KV format; with a recent-window
// ring (MK_KV_WIN) the row is also stored q8 in the ring (same rotation: every tile of a slot must share the basis)
__device__ __forceinline__ void kv_store(int w, int slot, int g, int pos, const float (&vin)[8]) {
    float v[8];
#pragma unroll
    for (int k = 0; k < 8; k++) v[k] = vin[k];
    if (w == 0 ? cB.kvr[slot] : cB.kvrv[slot]) fwht256(v);
    const int bits = cB.kvb[w][slot];
    const size_t row = (size_t) g * cB.ctx + pos;
    __half * sc = cB.kvs[w][slot] + row * (HD / 32);
    if (bits == 8) kv_store_q8(cB.kvp[w][slot] + row * HD, sc, v);
    else if (bits == 4) kv_store_q4(cB.kvp[w][slot] + row * (HD / 2), sc, v);
    else kv_store_qn(cB.kvp[w][slot] + row * (HD * bits / 8), sc, v, bits);
    if (cB.kvrp[w][slot]) {
        const size_t rr = (size_t) g * cB.kvrc + pos % cB.kvrc;
        kv_store_q8(cB.kvrp[w][slot] + rr * HD, cB.kvrs[w][slot] + rr * (HD / 32), v);
    }
}

// MK_KV_WIN: the window start for context length L_ctx (positions >= ws are read from the q8 ring). Stepped by ATT_BUCKET
// like the attention partition, so all rows of one launch (att_cap_T) and both draft depths see the same ring / main
// split of the tiles: greedy output stays independent of the draft depth. The window covers Wb ATT_BUCKET + 1 ..
// (Wb + 1) ATT_BUCKET positions (>= W), the ring holds (Wb + 1) ATT_BUCKET rows.
__device__ __forceinline__ int kv_ws(int L_ctx) {
    if (cB.kvwin <= 0) return 0;
    const int ws = (L_ctx + ATT_BUCKET - 1) / ATT_BUCKET - 1 - (cB.kvwin + ATT_BUCKET - 1) / ATT_BUCKET;
    return ws > 0 ? ws * ATT_BUCKET : 0;
}
// bits of the tile at position p0 (a multiple of ATP) of side w of a slot: 8 from the ring, else the main format
__device__ __forceinline__ int kv_tbits(int w, int slot, int p0, int ws) {
    return (cB.kvrp[w][slot] != nullptr && p0 >= ws) ? 8 : cB.kvb[w][slot];
}
// ws % kvrc for kv_src (computed once per launch / item: a runtime integer modulo is ~25 dependent instructions)
__device__ __forceinline__ int kv_wsm(int ws) { return cB.kvrc > 0 ? ws % cB.kvrc : 0; }
// Per-slot KV format facts, loaded once per launch: the hot kernels then decide a tile's format and source with one integer
// compare of the tile index against kv_ring_tile (kv_tbits / kv_src per tile cost a uniform-datapath chain with R2UR
// conversions of the ring pointer right after the tile barrier).
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
// kv_tile_src with the decision made here (attn_core_t, pk_attn2: per-tile cost acceptable), returns the tile's bits
__device__ __forceinline__ int kv_src(int w, int slot, int g, int p0, int ws, int wsm, const int8_t *& q, const __half *& s) {
    const bool ring = cB.kvrp[w][slot] != nullptr && p0 >= ws;
    const int bits = cB.kvb[w][slot];
    kv_tile_src(w, slot, g, p0, ws, wsm, bits, ring, q, s);
    return ring ? 8 : bits;
}

#if MK_KV_EMU

// Lloyd-Max (MSE-optimal) codebooks for N(0,1), positive half (symmetric), 2..6 bits.
// MSE / variance: 0.1175 (2 bits), 0.03455 (3), 0.009501 (4), 0.002505 (5), 0.000644 (6).
__constant__ float LM_LEVELS[62] = {
    0.4527800f, 1.5104176f,                                                                                     // 2 bits
    0.2450942f, 0.7560053f, 1.3439093f, 2.1519457f,                                                             // 3
    0.1283950f, 0.3880483f, 0.6567591f, 0.9423405f, 1.2562312f, 1.6180464f, 2.0690172f, 2.7325896f,             // 4
    0.0658897f, 0.1980518f, 0.3313783f, 0.4666995f, 0.6049336f, 0.7471357f, 0.8945651f, 1.0487833f,             // 5
    1.2118044f, 1.3863403f, 1.5762281f, 1.7872332f, 2.0287284f, 2.3177394f, 2.6911196f, 3.2607325f,
    0.0335182f, 0.1006038f, 0.1678377f, 0.2353205f, 0.3031553f, 0.3714493f, 0.4403149f, 0.5098710f,             // 6
    0.5802446f, 0.6515730f, 0.7240059f, 0.7977081f, 0.8728625f, 0.9496748f, 1.0283779f, 1.1092386f,
    1.1925660f, 1.2787219f, 1.3681356f, 1.4613233f, 1.5589154f, 1.6616948f, 1.7706536f, 1.8870784f,
    2.0126849f, 2.1498391f, 2.3019465f, 2.4741863f, 2.6750484f, 2.9200457f, 3.2428991f, 3.7463072f,
};
__constant__ int LM_OFF[5] = {0, 2, 6, 14, 30}; // bits 2..6
// fixed random signs for mode 's' (randomised Hadamard): bit 8 lane + k flips value k of the lane
__constant__ unsigned EMU_SIGNS[8] = {0x6818f096u, 0x0205cd15u, 0xca000ab0u, 0x0ce98e8du, 0x005d3defu, 0xf897127cu, 0x0b0b0cf8u, 0x8fd7fa69u};

// A K / V row rounded to what a low-bit cache format would reconstruct, written in the q8 layout (accuracy experiments,
// MK_KV_EMU_SPEC). mode: bits (2..8) | 16 Hadamard rotation (WHT) first | 32 q4_0-style scale d = (signed max) / -2^(b-1)
// (else symmetric d = max |x| / (2^(b-1) - 1)) | 64 Lloyd-Max codebook for the (near-Gaussian, rotated) coordinates with
// scale = rms of the group | 128 scale group = the whole 256-value row (else 32 values) | 256 random signs before the
// WHT (randomised Hadamard; undone after the inverse). The scale is fp16. A plain integer grid without rotation is
// stored exactly (integers + scale, as a real kernel would: "4a" is bit-identical to kv_store_q4); everything else is
// dequantised, transformed back and stored as ordinary q8 (adds q8 rounding). Positions < cB.kvemu_sink stay plain q8.
__device__ __noinline__ void kv_store_emu(int8_t * q, __half * sc, const float (&vin)[8], int mode, int pos) {
    const int lane = threadIdx.x & 31, bits = mode & 15;
    const bool rot = mode & (16 | 256), q40 = mode & 32, lm = mode & 64, rowg = mode & 128, sgn = mode & 256;
    const bool ta = mode & (2048 | 4096); // TurboAngle (t: 4-bit log norms, T: 8-bit linear norms); 1024: demotion pass
    if (pos < cB.kvemu_sink || (cB.kvemu_win > 0 && !(mode & 1024))) {
        kv_store_q8(q, sc, vin);
        return;
    }
    float v[8];
#pragma unroll
    for (int k = 0; k < 8; k++) v[k] = vin[k];
    if (sgn) {
#pragma unroll
        for (int k = 0; k < 8; k++)
            if ((EMU_SIGNS[lane >> 2] >> ((lane & 3) * 8 + k)) & 1u) v[k] = -v[k];
    }
    if (rot) fwht256(v);
    float r[8]; // the reconstruction
    if (ta) {
        // TurboAngle (2603.27467): after the randomised Hadamard, consecutive pairs -> (norm, angle); angles are uniform
        // on the circle and quantised to 2^bits bins; the 128 pair norms of the row are min-max quantised (t: 4 bits in
        // log space, T: 8 bits linear) with the row's min / max kept in fp32
        const int nb = 1 << bits;
        const bool lg = mode & 2048;
        const float ql = (float) ((1 << (lg ? 4 : 8)) - 1);
        float rn[4], th[4], mn = INFINITY, mx = -INFINITY;
#pragma unroll
        for (int j = 0; j < 4; j++) {
            const float x = v[2 * j], y = v[2 * j + 1];
            th[j] = atan2f(y, x);
            const float nrm = sqrtf(x * x + y * y);
            rn[j] = lg ? logf(fmaxf(nrm, 1e-30f)) : nrm;
            mn = fminf(mn, rn[j]);
            mx = fmaxf(mx, rn[j]);
        }
#pragma unroll
        for (int m = 1; m <= 16; m <<= 1) {
            mn = fminf(mn, __shfl_xor_sync(0xffffffffu, mn, m));
            mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, m));
        }
        const float span = mx - mn, qs = span > 0.f ? ql / span : 0.f, iqs = span > 0.f ? span / ql : 0.f;
#pragma unroll
        for (int j = 0; j < 4; j++) {
            const float u = mn + (float) __float2int_rn((rn[j] - mn) * qs) * iqs;
            const float nrm = lg ? expf(u) : u;
            int ka = __float2int_rn(th[j] * ((float) nb / 6.283185307f));
            ka = ((ka % nb) + nb) % nb;
            const float a = (float) ka * (6.283185307f / (float) nb);
            r[2 * j] = nrm * cosf(a);
            r[2 * j + 1] = nrm * sinf(a);
        }
    } else {
    // group statistics: largest |x| and that x (ties: the larger x, the same in every lane of the group), sum of squares
    float am = 0.f, sv = 0.f, ss = 0.f;
#pragma unroll
    for (int k = 0; k < 8; k++) {
        ss += v[k] * v[k];
        if (fabsf(v[k]) > am || (fabsf(v[k]) == am && v[k] > sv)) { am = fabsf(v[k]); sv = v[k]; }
    }
    const int gm = rowg ? 16 : 2;
    for (int m = 1; m <= gm; m <<= 1) {
        const float oa = __shfl_xor_sync(0xffffffffu, am, m), os = __shfl_xor_sync(0xffffffffu, sv, m);
        if (oa > am || (oa == am && os > sv)) { am = oa; sv = os; }
        ss += __shfl_xor_sync(0xffffffffu, ss, m);
    }
    const int h = 1 << (bits - 1);
    const float d = lm ? sqrtf(ss / (rowg ? 256.f : 32.f)) : q40 ? sv / (float) -h : am / (float) (h - 1);
    const __half dh = __float2half_rn(d);
    const float df = __half2float(dh);
    if (lm) {
        const float * lv = LM_LEVELS + LM_OFF[bits - 2];
        const float id = df > 0.f ? 1.0f / df : 0.f;
#pragma unroll
        for (int k = 0; k < 8; k++) {
            const float a = fabsf(v[k]) * id;
            int i = 0; // nearest level: thresholds are the midpoints
            for (int j = 1; j < h; j++) i = a >= 0.5f * (lv[j - 1] + lv[j]) ? j : i;
            r[k] = copysignf(df * lv[i], v[k]);
        }
    } else {
        const float id = d != 0.f ? 1.0f / d : 0.f;
        int qi[8];
#pragma unroll
        for (int k = 0; k < 8; k++) {
            qi[k] = q40 ? min(2 * h - 1, (int) (v[k] * id + ((float) h + 0.5f))) - h : max(1 - h, min(h - 1, __float2int_rn(v[k] * id)));
            r[k] = df * (float) qi[k];
        }
        if (!rot && !rowg) { // exact store of the integers and the fp16 scale
            unsigned w[2] = {0u, 0u};
#pragma unroll
            for (int k = 0; k < 8; k++) w[k >> 2] |= ((unsigned) qi[k] & 0xffu) << (8 * (k & 3));
            *reinterpret_cast<uint2 *>(q + lane * 8) = make_uint2(w[0], w[1]);
            if ((lane & 3) == 0) sc[lane >> 2] = dh;
            return;
        }
    }
    } // grid modes
    if (rot) {
        fwht256(r);
        if (sgn) {
#pragma unroll
            for (int k = 0; k < 8; k++)
                if ((EMU_SIGNS[lane >> 2] >> ((lane & 3) * 8 + k)) & 1u) r[k] = -r[k];
        }
    }
    kv_store_q8(q, sc, r);
}

// Recent-window emulation ("win=N"): rows are stored as plain q8 and demoted to their slot's mode once they are older
// than N positions. Re-quantises rows of positions p0 .. p1 - 1 of every emulated K / V slot (one warp per row; q8
// in, kv_store_emu with the force bit out; sink rows are re-stored as q8, which is idempotent).
__global__ void __launch_bounds__(256) k_kv_demote(int p0, int p1) { MK_VIEW_INIT();
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int n = p1 - p0;
    for (int i = blockIdx.x * 8 + warp; i < n * NKV * N_ATT_SLOTS * 2; i += gridDim.x * 8) {
        const int w = i & 1, slot = (i >> 1) % N_ATT_SLOTS, g = ((i >> 1) / N_ATT_SLOTS) % NKV;
        const int p = p0 + (i >> 1) / (N_ATT_SLOTS * NKV);
        const int mode = cB.kvemu[w][slot];
        if (!mode) continue;
        const size_t row = ((size_t) slot * NKV + g) * cB.ctx + p;
        int8_t * q = static_cast<int8_t *>(w ? cB.vc : cB.kc) + row * HD;
        __half * sc = (w ? cB.vsc : cB.ksc) + row * (HD / 32);
        const uint2 qq = *reinterpret_cast<const uint2 *>(q + lane * 8);
        const float d = __half2float(sc[lane >> 2]);
        float v[8];
#pragma unroll
        for (int k = 0; k < 8; k++) v[k] = d * (float) (int8_t) ((k < 4 ? qq.x : qq.y) >> (8 * (k & 3)));
        __syncwarp();
        kv_store_emu(q, sc, v, mode | 1024, p);
    }
}
#endif

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
// Staged q4 .. q7 K / V rows (kv_store_q4 / kv_store_qn's layout: lane word w = the low nibbles of values 8 w .. 8 w + 7, then
// at 128 + w (BITS - 4) their high bits) expanded IN PLACE to the q8 layout: 256 int8 values q = n - 2^(BITS - 1), the scales
// untouched. The fp16 product d q is the same number the low-bit decode computes (qn_to_f16 / a2_qnh), so every consumer of
// the q8 layout (aq_q8x2, a8_imma with the scale applied after the int8 dot) sees exactly the low-bit values.
// One warp per row, up to 3 rows (r0, r0 + rs, r0 + 2 rs, all < np) per call, loads of all rows first (one shared-memory
// latency instead of a load -> ALU -> store chain per row), then every lane has read its words before any lane writes (the
// output overlaps the input).
// SIGNED = false leaves the biased bytes n (0 .. 2^BITS - 1) in place: the fp16 consumers subtract the bias for free in
// their magic-number conversion (aq_q8x2 / aq_q8x2p with the tile's xm / sub); the int8 IMMA consumer (pk_attn8) needs q.
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

// staged q8 / q4..q7 K/V rows (KVQ_ROW bytes: values, then scales at 256) -> fp16 rows of stride ld; rows x 2 tensors
__device__ __forceinline__ void kv_tile_to_f16(const uint8_t * src, __half * dK, __half * dV, int rows, int ld, int kbits, int vbits) {
    for (int i = threadIdx.x; i < 2 * rows * 32; i += blockDim.x) {
        const int which = i / (rows * 32), r = (i >> 5) % rows, c = i & 31;
        const uint8_t * row = src + (size_t) (which * rows + r) * KVQ_ROW;
        const float d = __half2float(reinterpret_cast<const __half *>(row + 256)[c >> 2]);
        const int bits = which ? vbits : kbits;
        __half2 h[4];
        if (bits == 4) q4_to_f16(*reinterpret_cast<const unsigned *>(row + 4 * c), d, h);
        else if (bits != 8) qn_to_f16(*reinterpret_cast<const unsigned *>(row + 4 * c), kv_hi(row + 128 + c * (bits - 4), bits - 4), d, bits, h);
        else q8_to_f16(*reinterpret_cast<const uint2 *>(row + 8 * c), d, h);
        *reinterpret_cast<int4 *>((which ? dV : dK) + (size_t) r * ld + 8 * c) = *reinterpret_cast<int4 *>(h);
    }
}

// MK_KV_WIN after a checkpoint restore to position p1: the ring rows of positions p0 .. p1 - 1 may have been overwritten by
// the abandoned continuation, so they are refilled from the main buffer (its low-bit values, re-stored as q8: the window
// is at the main format's precision until those rows age out). One warp per row.
__global__ void __launch_bounds__(256) k_ring_refill(int p0, int p1) { MK_VIEW_INIT();
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int n = p1 - p0;
    for (int i = blockIdx.x * 8 + warp; i < n * NKV * N_ATT_SLOTS * 2; i += gridDim.x * 8) {
        const int w = i & 1, slot = (i >> 1) % N_ATT_SLOTS, g = ((i >> 1) / N_ATT_SLOTS) % NKV;
        const int p = p0 + (i >> 1) / (N_ATT_SLOTS * NKV);
        if (!cB.kvrp[w][slot]) continue;
        const int bits = cB.kvb[w][slot];
        const size_t row = (size_t) g * cB.ctx + p;
        const uint8_t * rp = reinterpret_cast<const uint8_t *>(cB.kvp[w][slot] + row * (HD * bits / 8));
        const float d = __half2float(cB.kvs[w][slot][row * (HD / 32) + (lane >> 2)]);
        float v[8];
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
        __syncwarp();
        const size_t rr = (size_t) g * cB.kvrc + p % cB.kvrc;
        kv_store_q8(cB.kvrp[w][slot] + rr * HD, cB.kvrs[w][slot] + rr * (HD / 32), v);
    }
}

// async copy of q8 / q4..q7 K/V rows r < np of one tile (Kq / Ks, Vq / Vs: the tile's first row, kv_src) into the staging
// layout (rows >= np: zero scales). Chunk order: the value chunks of K then V, then the rows' scale chunks (aq_issue's)
__device__ __forceinline__ void kv_stage(uint8_t * dst, const int8_t * Kq, const int8_t * Vq, const __half * Ks, const __half * Vs,
                                         int rows, int np, int kbits, int vbits) {
    for (int i = threadIdx.x; i < 2 * rows * 17; i += blockDim.x) {
        const bool sc = i >= 2 * rows * 16;
        const int s = sc ? i - 2 * rows * 16 : i;
        const int which = sc ? s / rows : s / (rows * 16), r = sc ? s % rows : (s / 16) % rows, c = sc ? 16 : s % 16;
        uint8_t * d = dst + (size_t) (which * rows + r) * KVQ_ROW + 16 * c;
        if (r < np) {
            const int bits = which ? vbits : kbits; // a row of HD bits / 8 bytes = 2 bits chunks of 16
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

// split-K over the context: at most ATT_NCH chunks per kv head so all 4 x nch items run in one wave.
// The chunk size is a multiple of the K/V tile that only changes when L_ctx crosses a multiple of
// ATT_BUCKET, and the host never lets one launch's rows straddle such a boundary (att_cap_T). So the
// chunk boundaries below any query position p (and the float summation order of its attention) depend
// on p alone, not on how many tokens the cycle verifies: greedy output is independent of the draft depth.
// Same tiles per item as a ceil(L / 20) split for L > 1280; one tile per item up to 640.
__device__ __forceinline__ void attn_dims(int L_ctx, int & nch, int & cs) {
    cs = ATP * max(1, (L_ctx + ATT_BUCKET - 1) / ATT_BUCKET);
    nch = max(1, (L_ctx + cs - 1) / cs);
}
// MK_MTP_WIN=N (measurement knob, default off): the MTP slot's decode attention covers the first tile of positions
// (sinks, StreamingLLM) and the last N ones; ws = the window start (0: everything, also when the window would start
// inside the first two tiles). The chunk partition is attn_dims over [ws, L_ctx) plus one sink chunk [0, ATP).
// ws is rounded down to a multiple of ATP: a window ring's tiles must start on one (kv_tile_src: a tile never wraps)
__device__ __forceinline__ int mtp_ws(int slot, int n) {
    if (slot != N_ATT_SLOTS - 1 || cB.mtp_win <= 0) return 0;
    const int ws = n + 1 - cB.mtp_win;
    return ws > 2 * ATP ? ws & ~(ATP - 1) : 0;
}
__device__ __forceinline__ void attn_dims_w(int L_ctx, int ws, int & nch, int & cs) {
    attn_dims(L_ctx - ws, nch, cs);
    if (ws > 0) nch++;
}
__device__ __forceinline__ void attn_chunk(int c, int ws, int cs, int L_ctx, int & c0, int & c1) {
    if (ws > 0 && c == 0) {
        c0 = 0;
        c1 = ATP;
        return;
    }
    c0 = ws + (c - (ws > 0 ? 1 : 0)) * cs;
    c1 = min(L_ctx, c0 + cs);
}
// this block's rank in an attention / combine call whose items start at block b0 (b0 >= 0): a rotation of the grid, so
// every b0 runs every item exactly once; independent calls back to back (no barrier between) pass the item count of the
// calls before them, so at short contexts their few items run concurrently on disjoint blocks. Partition, merge order
// and per-item arithmetic do not depend on the block.
__device__ __forceinline__ int att_rank(int b0) {
    const int r = ((int) blockIdx.x - b0) % (int) gridDim.x;
    return r < 0 ? r + (int) gridDim.x : r;
}
// the idle blocks of a short-context attention call: L2 prefetch of resid_o's weights (defined after wv.cuh)
__device__ __forceinline__ void attn_idle_pf(const Mat & m, int ib, int nib);

// attention q/k/v projections
template <int T>
__device__ __noinline__ void attn_proj(const LayerW & W, int slot, int n, uint8_t * /*sm*/) {
    uint8_t * const sm = smem;
    const int lane = threadIdx.x & 31;
    (void) lane;
    const SX sx = make_sx(sm, U_D);
    fill_xs<T>(cB.qn, 0, true, sx, misc(sm));
    const int nq = QG / 2, nk = KVD / 2;
    gemv_pairs_any<T, 5>(W.q, 0, nq, PairOut{cB.qbuf, QG, -1});
    gemv_pairs_any<T, 5>(W.k, nq, nq + nk, PairOut{cB.kbuf, KVD, -1});
    gemv_pairs_any<T, 5>(W.v, nq + nk, nq + 2 * nk, PairOut{cB.vbuf, KVD, -1});
    // Warm L2 with what attn_core reads first: q/k norm weights, RoPE rows of the new positions,
    // and the first K / V tile of this block's attention item (positions < n).
    pf_block(W.qn, HD * 4);
    pf_block(W.kn, HD * 4);
    pf_block(cB.rope + (size_t) n * 32, T * 32 * 8);
    int nch, CS;
    attn_dims(n + T, nch, CS);
    const size_t ctx = cB.ctx;
    // bytes and 128-byte lines per K / V row (fp16 256 x 2, q8 256, q4 128)
    const int rk = cB.kvq8 ? HD * cB.kvb[0][slot] / 8 : HD * 2, rv = cB.kvq8 ? HD * cB.kvb[1][slot] / 8 : HD * 2;
    const int lk = rk / 128, lv = rv / 128, lines = max(lk, lv);
    const uint8_t * Kc = cB.kvq8 ? reinterpret_cast<const uint8_t *>(cB.kvp[0][slot])
                                 : static_cast<const uint8_t *>(cB.kc) + (size_t) slot * NKV * ctx * rk;
    const uint8_t * Vc = cB.kvq8 ? reinterpret_cast<const uint8_t *>(cB.kvp[1][slot])
                                 : static_cast<const uint8_t *>(cB.vc) + (size_t) slot * NKV * ctx * rv;
    for (int item = blockIdx.x; item < NKV * nch; item += gridDim.x) {
        const int g = item / nch, c0 = (item % nch) * CS, c1 = min(n, min(c0 + CS, c0 + 64));
        for (int i = threadIdx.x; i < (c1 - c0) * lines; i += NT) {
            const size_t pos = (size_t) g * ctx + c0 + i / lines;
            const int l = i % lines;
            if (l < lk) pf_l2(Kc + pos * rk + l * 128);
            if (l < lv) pf_l2(Vc + pos * rv + l * 128);
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

// the RoPE pairs (lane & 3) * 8 .. + 7 of position pos (a q row's rc[]) as 4 16-byte loads (rows are 256-byte aligned):
// the decode attention prologues' 8-byte loads per pair filled the LSU queue (Nsight Compute at 40k context: lg
// throttle 68% there)
__device__ __forceinline__ void rope_row(float2 (&rc)[8], int pos, int lane) {
    const float4 * tab = reinterpret_cast<const float4 *>(cB.rope + (size_t) pos * 32 + (lane & 3) * 8);
#pragma unroll
    for (int k = 0; k < 4; k++) {
        const float4 x = __ldg(tab + k);
        rc[2 * k] = make_float2(x.x, x.y);
        rc[2 * k + 1] = make_float2(x.z, x.w);
    }
}

__device__ __forceinline__ void load8(const float * p, float (&v)[8]) {
    const float4 a = __ldcg(reinterpret_cast<const float4 *>(p));
    const float4 b = __ldcg(reinterpret_cast<const float4 *>(p + 4));
    v[0] = a.x; v[1] = a.y; v[2] = a.z; v[3] = a.w;
    v[4] = b.x; v[5] = b.y; v[6] = b.z; v[7] = b.w;
}
// load8 through the read-only (L1-cached) path, for data no phase writes (the q-norm weights: every warp of every working
// block reads the same 1 KB at its item start, which __ldcg sent to L2 1,280 times at once at 40k)
__device__ __forceinline__ void load8_ro(const float * p, float (&v)[8]) {
    const float4 a = __ldg(reinterpret_cast<const float4 *>(p));
    const float4 b = __ldg(reinterpret_cast<const float4 *>(p + 4));
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

// ---- MK_ATT_I8 (-DMK_ATT_I8=1 builds; Bufs::att_i8): int8 Q.K^T in the decode attention of all-q8 slots, in every kernel
// that can serve it (attn_core_ti, attn_core_q8i, attn_core_w48i; the dispatch in attn_core and main_forward_d's deep pass). A query row
// (after the RMS norm, RoPE and the slot's K rotation, as the fp16 path) is quantized per 32-dim block with q8_0 rounding
// (aq_i8_q: kv_store_q8's absmax / round-to-nearest; the fp32 block scale absmax / 127 carries the softmax scale 1/16);
// a score is the sum over the 8 blocks, in order 0 .. 7 from 0, of the block's int32 dot product (exact) x (q scale x
// k scale), every step an explicitly rounded intrinsic (aq_i8_acc: no FMA contraction left to nvcc or ptxas) in ONE
// helper (aq_i8_s) that all three kernels call. So a row's S, hence its partials, are bit-identical in every kernel, T,
// draft depth and lane: the engine's exactness holds with int8 Q too. Not bit-identical to the fp16 Q path (KLD-gated).
constexpr float AQ_I8_QS = 0.0625f / 127.0f; // q scale = block absmax x 1/16 / 127
constexpr int AQ_I8_LDQ = HD + 16;           // int8 q row stride (bytes): the 8 rows of an ldmatrix hit distinct banks
// one query row: lane l holds values 8 l .. 8 l + 7 (block l >> 2) -> 256 int8 values at q, block scales at qs[b * ld]
// (block-major: the 8 rows g8 of an MMA fragment read 8 consecutive words). Whole warp (the shuffles).
__device__ __forceinline__ void aq_i8_q(const float (&v)[8], int8_t * q, float * qs, int ld) {
    const int lane = threadIdx.x & 31;
    float am = 0.f;
#pragma unroll
    for (int k = 0; k < 8; k++) am = fmaxf(am, fabsf(v[k]));
    am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, 1));
    am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, 2));
    const float id = am > 0.f ? __fdiv_rn(127.0f, am) : 0.f;
    unsigned w[2] = {0u, 0u};
#pragma unroll
    for (int k = 0; k < 8; k++) w[k >> 2] |= ((unsigned) __float2int_rn(__fmul_rn(v[k], id)) & 0xffu) << (8 * (k & 3));
    *reinterpret_cast<uint2 *>(q + lane * 8) = make_uint2(w[0], w[1]);
    if ((lane & 3) == 0) qs[(lane >> 2) * ld] = __fmul_rn(am, AQ_I8_QS);
}
// a padding row (rows >= R): zero values and scales
__device__ __forceinline__ void aq_i8_q0(int8_t * q, float * qs, int ld) {
    const int lane = threadIdx.x & 31;
    *reinterpret_cast<uint2 *>(q + lane * 8) = make_uint2(0u, 0u);
    if ((lane & 3) == 0) qs[(lane >> 2) * ld] = 0.f;
}
// s + (float) c x (q scale x k scale): the only form any I8 kernel accumulates a score block with ((float) c is exact:
// |c| <= 32 x 127 x 127 < 2^24)
__device__ __forceinline__ float aq_i8_acc(float s, int c, float qs, float ks) {
    return __fmaf_rn(__int2float_rn(c), __fmul_rn(qs, ks), s);
}
// int8 m16n8k32 MMA from a zero accumulator: one 32-dim block's int32 dot products
__device__ __forceinline__ void mma_s8z(int (&d)[4], unsigned a0, unsigned a1, unsigned a2, unsigned a3, unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%10,%10,%10};"
                 : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(0));
}
// S of one warp: B = K rows (positions) 8 wn + g8 of the staged tile kst (KVQ_ROW rows: int8 values, fp16 scales at 256;
// all-q8 slots only: signed q8 bytes), A = int8 query rows 16 i + (lane & 15) of q8 (stride AQ_I8_LDQ) for the MT row tiles,
// q scales qs[b * QLD + row]. s[i] = rows 16 i + g8 (0, 1) / + 8 (2, 3), positions 8 wn + 2 tig (0, 2) / + 1 (1, 3).
// Whole warp (ldmatrix / mma.sync); q8 / qs may point at a row tile (attn_core_ti: one tile per warp).
template <int MT, int QLD>
__device__ __forceinline__ void aq_i8_s(float (&s)[MT][4], const int8_t * q8, const float * qs, const uint8_t * kst, int wn) {
    const int lane = threadIdx.x & 31, g8 = lane >> 2, tig = lane & 3;
    const uint8_t * krow = kst + (8 * wn + g8) * KVQ_ROW;
    const __half * ksc0 = reinterpret_cast<const __half *>(kst + (8 * wn + 2 * tig) * KVQ_ROW + 256);
    const __half * ksc1 = reinterpret_cast<const __half *>(kst + (8 * wn + 2 * tig + 1) * KVQ_ROW + 256);
    const unsigned qaddr = (unsigned) __cvta_generic_to_shared(q8 + (lane & 15) * AQ_I8_LDQ + (lane >> 4) * 16);
    const float * qsr = qs + g8;
#pragma unroll
    for (int i = 0; i < MT; i++)
#pragma unroll
        for (int k = 0; k < 4; k++) s[i][k] = 0.f;
#pragma unroll
    for (int kb = 0; kb < HD / 32; kb++) {
        const unsigned b0 = *reinterpret_cast<const unsigned *>(krow + 32 * kb + 4 * tig);
        const unsigned b1 = *reinterpret_cast<const unsigned *>(krow + 32 * kb + 16 + 4 * tig);
        const float k0 = __half2float(ksc0[kb]), k1 = __half2float(ksc1[kb]);
#pragma unroll
        for (int i = 0; i < MT; i++) {
            unsigned a[4];
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                         : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3])
                         : "r"(qaddr + 16 * i * AQ_I8_LDQ + 32 * kb));
            int c[4];
            mma_s8z(c, a[0], a[1], a[2], a[3], b0, b1);
            const float q0 = qsr[kb * QLD + 16 * i], q1 = qsr[kb * QLD + 16 * i + 8];
            s[i][0] = aq_i8_acc(s[i][0], c[0], q0, k0);
            s[i][1] = aq_i8_acc(s[i][1], c[1], q0, k1);
            s[i][2] = aq_i8_acc(s[i][2], c[2], q1, k0);
            s[i][3] = aq_i8_acc(s[i][3], c[3], q1, k1);
        }
    }
}

// Split-K flash decoding over (kv head, chunk) items.  The 6*T query rows of a kv head form the
// M dimension of tensor-core MMAs: S = Q K^T and O += P V per 64-position tile of K / V staged
// in shared memory (fp16 operands, fp32 accumulation, online softmax).  Also writes this cycle's K/V.
// I8 (attn_core_ti, all-q8 slots with MK_ATT_I8): int8 q rows and S = aq_i8_s from the tile's q8 stage (the fp16 K tile
// is converted as before but not read).
template <int T, bool Q8, bool I8 = false>
__device__ __forceinline__ void attn_core_t(const LayerW & W, int slot, int n, int b0 = 0) { // b0: att_rank
    static_assert(!I8 || Q8, "int8 Q.K^T reads the q8 stage");
    uint8_t * const sm = smem;
    constexpr int R = 6 * T;
    constexpr int MT = (R + 15) / 16; // 16-row MMA tiles of query rows
    constexpr int TP = ATP;           // positions per K/V tile; tiles are double-buffered
    constexpr int LDH = HD + 8;       // smem row stride in halves (conflict-free fragment loads)
    constexpr int LDP = TP + 8;
    const int L_ctx = n + T;
    const int ws = mtp_ws(slot, n); // MK_MTP_WIN (ws <= n: this cycle's rows are always inside the window)
    int nch, CS;
    attn_dims_w(L_ctx, ws, nch, CS);
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g8 = lane >> 2, tig = lane & 3;
    __half * sQ = reinterpret_cast<__half *>(sm); // [32][LDH]
    int8_t * const sQ8 = reinterpret_cast<int8_t *>(sm);              // I8: int8 q rows [32][AQ_I8_LDQ] in sQ's place,
    float * const sQs = reinterpret_cast<float *>(sm + 32 * AQ_I8_LDQ); // then their block scales [8][32]
    static_assert(32 * AQ_I8_LDQ + 8 * 32 * 4 <= 32 * LDH * 2, "I8 q rows inside sQ");
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
    const int wsr = Q8 ? kv_ws(L_ctx) : 0; // MK_KV_WIN: tiles at positions >= wsr come from the q8 ring (kv_src)
    const int wsm = Q8 ? kv_wsm(wsr) : 0;
    const int mt = warp >> 3, wn = warp & 7; // MMA work split: warp -> (row tile, column group)

    auto issue_tile = [&](int g, int tb, int np_, int st) {
        if constexpr (Q8) {
            const int8_t *tk, *tv;
            const __half *tks, *tvs;
            const int kb_ = kv_src(0, slot, g, tb, wsr, wsm, tk, tks), vb_ = kv_src(1, slot, g, tb, wsr, wsm, tv, tvs);
            kv_stage(sKVq + st * 2 * TP * KVQ_ROW, tk, tv, tks, tvs, TP, np_, kb_, vb_);
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
    for (int item = att_rank(b0); item < NKV * nch; item += gridDim.x) {
        const int g = item / nch, c = item % nch;
        int c0, c1;
        attn_chunk(c, ws, CS, L_ctx, c0, c1);
        // one round of small loads (q rows, norm weights, RoPE rows) before any bulk K/V traffic:
        // per-SM requests are served roughly in order
        float qv[2][8], qw[8];
        float2 rc[2][8];
        // Unconditional loads (rows >= R and lanes >= 8 load valid, unused data): conditionally assigned arrays
        // would be carried across the item loop in local memory (~1.5 KB of mk_cycle's stack frame, which the driver
        // reserves for all 1,536 thread slots of every SM).
        constexpr bool RT1 = 16 % T == 0; // the warp's rows warp, warp + 16 are one token (t = warp % T): one RoPE row
        // (k2 >= MT, i.e. T <= 2: rows 16 .. 31 are in no MMA row tile, nothing reads them: no loads, no stores)
        // The q rows' loads first (their sums come first), then the q-norm and RoPE ones (attn_core_q8_t's order)
        const float * const qb = cB.qbuf;
#pragma unroll
        for (int k2 = 0; k2 < 2; k2++) {
            if (k2 >= MT) continue;
            const int r = min(warp + 16 * k2, R - 1);
            load8(qb + (r % T) * QG + (g * 6 + r / T) * 512 + lane * 8, qv[k2]);
        }
        load8_ro(W.qn + lane * 8, qw);
#pragma unroll
        for (int k2 = 0; k2 < 2; k2++) {
            if (k2 >= MT) continue;
            const int r = min(warp + 16 * k2, R - 1);
            if (!RT1 || k2 == 0) {
                rope_row(rc[k2], n + (RT1 ? warp % T : r % T), lane);
            } else {
#pragma unroll
                for (int k = 0; k < 8; k++) rc[k2][k] = rc[0][k];
            }
        }
#pragma unroll
        for (int k2 = 0; k2 < 2; k2++) {
            if (k2 >= MT) continue;
            const int r = warp + 16 * k2;
            if (r < R) {
                // a copy, not a pointer into qv: a pointer makes ptxas keep qv / qw / rc in local memory (~2 KB of
                // mk_cycle's 2.8 KB stack frame over the 10 attn_core_t instances, reserved for 1,536 threads per SM)
                float v[8];
#pragma unroll
                for (int k = 0; k < 8; k++) v[k] = qv[k2][k];
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
                if (Q8 && cB.kvr[slot]) fwht256(v); // K of this slot is stored rotated (MK_KV "4r")
                if constexpr (I8) {
                    aq_i8_q(v, sQ8 + r * AQ_I8_LDQ, sQs + r, 32);
                } else {
                __half2 hq2[4];
#pragma unroll
                for (int k = 0; k < 4; k++) hq2[k] = __floats2half2_rn(v[2 * k] * 0.0625f, v[2 * k + 1] * 0.0625f);
                *reinterpret_cast<int4 *>(sQ + r * LDH + lane * 8) = *reinterpret_cast<int4 *>(hq2);
                }
            } else {
                if constexpr (I8) aq_i8_q0(sQ8 + r * AQ_I8_LDQ, sQs + r, 32);
                else
                *reinterpret_cast<int4 *>(sQ + r * LDH + lane * 8) = make_int4(0, 0, 0, 0);
                if (2 * lane < TP) *reinterpret_cast<unsigned *>(sP + r * LDP + 2 * lane) = 0u; // padded rows of P stay 0
            }
        }
        // this cycle's K / V rows that fall into the chunk (positions >= n) as jobs: j < np the K row of position pn0 + j,
        // j >= np the V row of position pn0 + j - np, from the last warp down: a position's K and V rows run on two warps
        // at once, on warps without q rows where there are such (the first warps hold them)
        const int pn0 = max(c0, n), np = min(c1, n + T) - pn0;
        for (int j = NW - 1 - warp; j < 2 * np; j += NW) {
            const bool vr = j >= np;
            const int pp = pn0 + (vr ? j - np : j), t = pp - n;
            const size_t row = slot0 + (size_t) g * ctx + pp;
            (void) row;
            float v[8];
            if (!vr) {
                load8(cB.kbuf + t * KVD + g * 256 + lane * 8, v);
                float ss = 0.f;
#pragma unroll
                for (int k = 0; k < 8; k++) ss += v[k] * v[k];
                const float rs = rsqrtf(wsum(ss) / 256.0f + EPS);
#pragma unroll
                for (int k = 0; k < 8; k++) v[k] *= rs * W.kn[lane * 8 + k];
                rope8(v, lane, pp);
#if MK_KV_EMU
                if (Q8 && cB.kvemu[0][slot]) kv_store_emu(static_cast<int8_t *>(cB.kc) + row * HD, cB.ksc + row * (HD / 32), v, cB.kvemu[0][slot], pp);
                else
#endif
                if constexpr (Q8) kv_store(0, slot, g, pp, v);
                else kv_store_f16(static_cast<__half *>(cB.kc) + row * HD, v);
            } else {
                load8(cB.vbuf + t * KVD + g * 256 + lane * 8, v);
#if MK_KV_EMU
                if (Q8 && cB.kvemu[1][slot]) kv_store_emu(static_cast<int8_t *>(cB.vc) + row * HD, cB.vsc + row * (HD / 32), v, cB.kvemu[1][slot], pp);
                else
#endif
                if constexpr (Q8) kv_store(1, slot, g, pp, v);
                else kv_store_f16(static_cast<__half *>(cB.vc) + row * HD, v);
            }
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
            if constexpr (Q8) { // q8 stage -> the fp16 tile (its formats: ring q8 or the slot's, per tile)
                kv_tile_to_f16(sKVq + st * 2 * TP * KVQ_ROW, sKV, sKV + TP * LDH, TP, LDH, kv_tbits(0, slot, tb, wsr), kv_tbits(1, slot, tb, wsr));
                __syncthreads();
            }
            // S = Q K^T : warp -> row tile mt, positions 8*wn .. 8*wn+7
            if constexpr (I8) { // int8 Q.K^T: the K rows of the q8 stage (signed q8 bytes: all-q8 slot, ring or main)
                if (mt < MT && wn < TP / 8) {
                    float s[1][4];
                    aq_i8_s<1, 32>(s, sQ8 + 16 * mt * AQ_I8_LDQ, sQs + 16 * mt, sKVq + st * 2 * TP * KVQ_ROW, wn);
                    float * so = sS + (16 * mt + g8) * TP + 8 * wn + 2 * tig;
                    so[0] = s[0][0];
                    so[1] = s[0][1];
                    so[8 * TP] = s[0][2];
                    so[8 * TP + 1] = s[0][3];
                }
            } else
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
#if MK_FORMAL_SYNC
                __syncwarp(); // the warp's sM[r] reads before lane 0 rewrites it (shfl.sync orders no memory)
#endif
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

// ---------------------------------------------------------------------------------------------------------------------
// Decode attention on the all-q8 cache: attn_core_t's per-row arithmetic (bit-identical partials, so also T-invariant),
// restructured for long contexts. Microbenchmark: bench/dattn_bench.cu.
// - Three-deep software pipeline with ONE block barrier per tile: interval j computes S(j) = Q K(j)^T, the online softmax
//   of tile j - 1 and O = O * alpha + P(j - 2) V(j - 2) on specialised warps, and issues the next copies:
//     warps 0-3   S for positions 8 w .. 8 w + 7, all row tiles (one K fragment conversion feeds the MT chains)
//     warps 4-11  P V for dims 32 (w - 4) .. + 31, all row tiles (one V fragment conversion feeds the MT tiles)
//     warps 12-15 softmax, a quad of lanes per query row (8 positions per lane)
// - q8 K / V tiles stream through separate cp.async rings (K(t) issued in interval t - 2 and read in t: 3 stages; V(t)
//   issued in t - 1 and read in t + 2: 4 stages). No fp16 K / V tiles: both MMA B operands are converted from the q8
//   rows in registers with q8_to_f16's rounding (d * q, exact in fp32, rounded once to fp16; here one fp16 multiply).
// - P V B fragments: lane (g8, tig) of column group wn holds dims 32 wn + 4 g8 + jj of n8 tile jj, so one 32-bit q8 load
//   per position feeds 4 tiles. Each output element is the same MMA dot product (same operand values, same k order),
//   only its column slot in the MMA differs.
// - The softmax keeps attn_core_t's float operations and its butterfly sum tree (position bit 4 first, then 3 .. 0).
// q4 MK_KV slots run the MIX instantiation (attn_core_q8m: q4 K / V fragments with q4_to_f16's values); chunks of a
// single tile (L_ctx <= ATT_BUCKET) and MK_DATTN=0 run attn_core_t (same partials).

// d * q rounded once to fp16 for the two int8 values in bytes 0 and 1 of u: 0x6400 | (q ^ 0x80) is the fp16 1152 + q,
// minus 1152 is q exactly, and one fp16 multiply rounds the exact product (= q8_to_f16's value).
// xm / sub generalise it to a tile of biased bytes n = q + 2^(b-1) (MIX, kv_rows_expand<_, false>): xm = 0, sub = 1024 +
// 2^(b-1) (aq_sub) gives n - 2^(b-1) = q, the very same fp16 q; the all-q8 path's constants fold at compile time.
__device__ __forceinline__ unsigned aq_q8x2(unsigned u, __half2 d, unsigned xm = 0x8080u, unsigned sub = 0x64806480u) {
    const unsigned h = __byte_perm(u ^ xm, 0x64646464u, 0x4140);
    const __half2 q = __hsub2(*reinterpret_cast<const __half2 *>(&h), *reinterpret_cast<const __half2 *>(&sub));
    const __half2 r = __hmul2(q, d);
    return *reinterpret_cast<const unsigned *>(&r);
}
// the same for bytes 0 / 2 or 1 / 3 of y (selector 0x4140 / 0x4342 after the caller's interleave; the caller applied xm)
__device__ __forceinline__ unsigned aq_q8x2p(unsigned y, unsigned sel, __half2 d, unsigned sub = 0x64806480u) {
    const unsigned h = __byte_perm(y, 0x64646464u, sel);
    const __half2 q = __hsub2(*reinterpret_cast<const __half2 *>(&h), *reinterpret_cast<const __half2 *>(&sub));
    const __half2 r = __hmul2(q, d);
    return *reinterpret_cast<const unsigned *>(&r);
}
// the magic-number subtrahend (as a half2 bit pattern) for a tile of bits b: q8 tiles hold q ^ 0x80 after xm (1152),
// expanded low-bit tiles hold n = q + 2^(b-1) (1024 + 2^(b-1): 0x6400 | 2^(b-1), exact below 2048)
__device__ __forceinline__ unsigned aq_sub(int bits) {
    const unsigned s = bits == 8 ? 0x6480u : 0x6400u | (1u << (bits - 1));
    return s * 0x10001u;
}
// one commit group's copies, issued by NTH threads (this one is thread t of them): K rows of a tile (kn valid rows,
// kn < 0: none) -> K stage dk, V rows of a tile -> dv; rows >= the valid count get zero scales (their values are
// never used: masked S, P = 0). MIX: q4..q7 rows (kbits / vbits) copy 2 bits value chunks (kv_stage's staging layout).
// Kq / Ks and Vq / Vs: the tiles' first rows (kv_src: the slot's main buffer, or the MK_KV_WIN q8 ring).
template <int NTH, bool MIX = false>
__device__ __forceinline__ void aq_issue(int t, uint8_t * dk, const int8_t * Kq, const __half * Ks, int kn, uint8_t * dv, const int8_t * Vq,
                                         const __half * Vs, int vn, int kbits = 8, int vbits = 8) {
#pragma unroll
    for (int k = 0; k < (2 * ATP * 17 + NTH - 1) / NTH; k++) {
        const int i = t + NTH * k;
        if (i < 2 * ATP * 17) {
            // the value chunks of K then V (chunk c of row r), then the 2 ATP rows' scale chunks (c = 16): a warp's 32 chunks
            // are 512 contiguous bytes of one array (17 chunks per row mixed values and scales in every warp request: ncu
            // L2 theoretical sectors 1.53 x ideal on these copies). Same destinations: the staged bytes do not change.
            const bool sc = i >= 2 * ATP * 16;
            const int s = sc ? i - 2 * ATP * 16 : i;
            const int which = sc ? s >= ATP : s >= ATP * 16, r = sc ? s % ATP : (s / 16) % ATP, c = sc ? 16 : s % 16;
            const int np = which ? vn : kn;
            if (np < 0) continue;
            uint8_t * d = (which ? dv : dk) + r * KVQ_ROW + 16 * c;
            if (r < np) {
                const int bits = MIX ? (which ? vbits : kbits) : 8;
                if (c < 16) {
                    if (c < 2 * bits) cp_async16(d, (which ? Vq : Kq) + (size_t) r * (HD * bits / 8) + 16 * c);
                } else cp_async16(d, (which ? Vs : Ks) + (size_t) r * (HD / 32));
            } else if (c == 16) {
                *reinterpret_cast<int4 *>(d) = make_int4(0, 0, 0, 0);
            }
        }
    }
}

#if MK_W48X
// one async copy when c < lim: a predicate, no branch (divergent lanes through BSSY / BRA cost more than the copy)
__device__ __forceinline__ void aqx_cp16(unsigned d, const void * s, unsigned c, unsigned lim) {
    asm volatile("{ .reg .pred p; setp.lt.u32 p, %2, %3; @p cp.async.cg.shared.global [%0], [%1], 16; }" ::"r"(d), "l"(s), "r"(c), "r"(lim) : "memory");
}
// aq_issue for one side of one full tile (MIX; the caller decides, aq_issue serves the partial last tile of a chunk),
// issued by WG warps (wi: this one): a warp's lanes copy a pair of rows per step, chunk c = lane & 15 of row rp = lane >> 4,
// and warp 0 of them the scales. q / s: the side's buffers (main or ring), row: the tile's first row in them (rows of at most
// 256 bytes: the 32-bit byte offset needs row + 16 <= 2^24; rows < NKV (ctx + KV_PAD) < 2^23 with ctx <= 2^20, ctx_limit).
// Same chunks and destinations as aq_issue for such a tile.
template <int WG>
__device__ __forceinline__ void aqx_side(int wi, int lane, const int8_t * q, const __half * s, unsigned row, int bits, uint8_t * st) {
    const unsigned rb = 32u * bits, c = lane & 15, rp = lane >> 4, lim = 2u * bits, st0 = (unsigned) __cvta_generic_to_shared(st);
    const int8_t * const p = q + (size_t) ((row + 2 * wi + rp) * rb + 16u * c);
    const unsigned d = st0 + (2 * wi + rp) * KVQ_ROW + 16u * c;
#pragma unroll
    for (int k = 0; k < 16 / WG; k++) aqx_cp16(d + 2 * WG * k * KVQ_ROW, p + 2 * WG * k * rb, c, lim);
    if (wi == 0) {
        const unsigned ds = st0 + lane * KVQ_ROW + 256;
        const __half * const ps = s + (size_t) (row + lane) * (HD / 32);
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(ds), "l"(ps) : "memory");
    }
}
// the S warps' B fragments of one K tile row (attn_core_q8_t's conversion, the same words and operations), kept in acc's
// registers across the interval barrier: a warp of the S / softmax groups converts in the interval before it multiplies
template <int MTA>
__device__ __forceinline__ void aqx_conv(float (&acc)[MTA][4][4], const uint8_t * krow, int tig, unsigned kxm, unsigned ksub) {
    const uint4 scv = *reinterpret_cast<const uint4 *>(krow + 256);
    const unsigned scw[4] = {scv.x, scv.y, scv.z, scv.w};
#pragma unroll
    for (int ks = 0; ks < HD / 16; ks++) {
        const unsigned sw = __byte_perm(scw[ks >> 2], 0, (ks & 2) ? 0x3232 : 0x1010);
        const __half2 dd = *reinterpret_cast<const __half2 *>(&sw);
        float (&a)[4] = acc[ks >> 3][(ks >> 1) & 3];
        a[2 * (ks & 1)] = __uint_as_float(aq_q8x2(*reinterpret_cast<const unsigned short *>(krow + 16 * ks + 2 * tig), dd, kxm, ksub));
        a[2 * (ks & 1) + 1] = __uint_as_float(aq_q8x2(*reinterpret_cast<const unsigned short *>(krow + 16 * ks + 2 * tig + 8), dd, kxm, ksub));
    }
}
// S = Q K^T from the fragments above (the same MMAs in the same order: k-step ks, then row tile i). The Q loads go out AQX_LA MMAs
// ahead in the program text: a small module keeps that, in the engine ptxas moves each load back in front of its MMA (no effect
// measured there; the stall is ~30% of the S warps' samples at 101k). KS (MK_W48Z): k-step ks + 1's Q loads before k-step ks's MMAs,
// two k-steps of fragments; with attn_core_w48's acc[2] relief ptxas then keeps w48m<8>'s loads a k-step ahead (wave 6, r5).
constexpr int AQX_LA = 4;
template <int MT, int MTA, int LDH, bool KS = false>
__device__ __forceinline__ void aqx_mma(float (&s)[MT][4], const float (&acc)[MTA][4][4], unsigned qaddr) {
#pragma unroll
    for (int i = 0; i < MT; i++)
#pragma unroll
        for (int k = 0; k < 4; k++) s[i][k] = 0.f;
    auto ldq = [&](int q, unsigned (&a)[4]) { // the Q fragment of MMA q = ks MT + i
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                     : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3])
                     : "r"(qaddr + 2 * (16 * (q % MT) * LDH + 16 * (q / MT))));
    };
    if constexpr (KS) {
        unsigned a[2][MT][4];
#pragma unroll
        for (int i = 0; i < MT; i++) ldq(i, a[0][i]);
#pragma unroll
        for (int ks = 0; ks < HD / 16; ks++) {
            if (ks + 1 < HD / 16) {
#pragma unroll
                for (int i = 0; i < MT; i++) ldq((ks + 1) * MT + i, a[(ks + 1) & 1][i]);
            }
            const float (&b)[4] = acc[ks >> 3][(ks >> 1) & 3];
#pragma unroll
            for (int i = 0; i < MT; i++) {
                const unsigned (&x)[4] = a[ks & 1][i];
                mma16816(s[i], x[0], x[1], x[2], x[3], __float_as_uint(b[2 * (ks & 1)]), __float_as_uint(b[2 * (ks & 1) + 1]));
            }
        }
    } else {
        unsigned a[AQX_LA][4];
#pragma unroll
        for (int q = 0; q < AQX_LA; q++) ldq(q, a[q]);
#pragma unroll
        for (int q = 0; q < (HD / 16) * MT; q++) {
            const int ks = q / MT, w = q % AQX_LA;
            const float (&b)[4] = acc[ks >> 3][(ks >> 1) & 3];
            mma16816(s[q % MT], a[w][0], a[w][1], a[w][2], a[w][3], __float_as_uint(b[2 * (ks & 1)]), __float_as_uint(b[2 * (ks & 1) + 1]));
            if (q + AQX_LA < (HD / 16) * MT) ldq(q + AQX_LA, a[w]);
        }
    }
}
// MK_W48Z: a row tile of acc that the S / softmax-group warps never read (attn_core_w48 MIX: acc[2]) re-zeroed at the end of their
// branches: its value (0 since the item start for those warps) is unchanged, its registers are dead inside the branches (with aqx_mma's
// KS order: the S warp's Q-fragment loads a k-step ahead of their MMAs; 258k F T8 attn_core -0.8 ms in wave 6)
__device__ __forceinline__ void aqx_zero(float (&a)[4][4]) {
#pragma unroll
    for (int j = 0; j < 4; j++)
#pragma unroll
        for (int k = 0; k < 4; k++) a[j][k] = 0.f;
}
// rows r0 .. r0 + N - 1 of a staged tile -> biased bytes: kv_rows_expand<BITS, false>'s bytes without a branch per row; all loads
// first, then every lane has read before any lane writes (the output overlaps the input)
template <int BITS, int N>
__device__ __forceinline__ void kvx_rows(uint8_t * st, int r0) {
    constexpr int HB = BITS - 4;
    const int lane = threadIdx.x & 31;
    unsigned wd[N], hi[N];
#pragma unroll
    for (int k = 0; k < N; k++) {
        const uint8_t * row = st + (r0 + k) * KVQ_ROW;
        wd[k] = *reinterpret_cast<const unsigned *>(row + 4 * lane);
        hi[k] = 0u;
        if constexpr (HB == 1) hi[k] = row[128 + lane];
        if constexpr (HB == 2) hi[k] = *reinterpret_cast<const unsigned short *>(row + 128 + 2 * lane);
        if constexpr (HB == 3) hi[k] = (unsigned) row[128 + 3 * lane] | ((unsigned) row[129 + 3 * lane] << 8) | ((unsigned) row[130 + 3 * lane] << 16);
    }
    __syncwarp();
#pragma unroll
    for (int k = 0; k < N; k++) {
        unsigned lo = wd[k] & 0x0f0f0f0fu, hv = (wd[k] >> 4) & 0x0f0f0f0fu;
        if constexpr (HB > 0) {
            lo |= kv_spread4<HB>(hi[k]);
            hv |= kv_spread4<HB>(hi[k] >> (4 * HB));
        }
        *reinterpret_cast<uint2 *>(st + (r0 + k) * KVQ_ROW + 8 * lane) = make_uint2(lo, hv);
    }
}
// the most common formats first: compare and branch, no jump table (LDC + BRX per tile and warp); bits != 8
template <int N>
__device__ __forceinline__ void kvx_tile(uint8_t * st, int bits, int r0) {
    if (bits == 5) kvx_rows<5, N>(st, r0);
    else if (bits == 6) kvx_rows<6, N>(st, r0);
    else if (bits == 4) kvx_rows<4, N>(st, r0);
    else kvx_rows<7, N>(st, r0);
}
// attn_core_q8_t's split of the work around the MMAs by T, from the role shares in the engine (Nsight, 101k positions): up to T
// AQX_CP the K and V copies run on the P V warps (the converting group's warp 0 holds all of the softmax there), above on the
// converting group; up to T AQX_VH the multiplying group expands V; the P V warps expand the rest after their MMAs
constexpr int AQX_CP = 2, AQX_VH = 2;
#endif

// MIX = false: all-q8 slots (attn_core_q8). MIX = true (attn_core_q8m): K and V of the slot each q8 or q4 .. q7 (MK_KV), the
// low-bit tiles staged in their packed layout and expanded to int8 once in shared memory (kv_row_expand: q = n - 2^(bits-1)
// with the scale unchanged, by warps 4 - 15 one interval before the tile is read; the K pipeline is one stage deeper for
// it), so the MMA path is the q8 one for every tile and the partials equal attn_core_t's for every MK_KV format.
// T >= 6 (both forms): 6 T > 32 query rows do not fit shared memory, so the items run twice, over query
// heads 0-2 and 3-5 of each kv head (HG = 3, partial rows ho T ..); the second pass re-reads K / V (MIX: and re-expands
// the low-bit tiles) and skips the K / V store (the first pass of the same item, in the same block, wrote those rows, the
// MK_KV_WIN ring's q8 copies included). Per-row arithmetic is unchanged.
// I8 (attn_core_q8i, all-q8 slots with MK_ATT_I8): int8 q rows (sQ's place) and S = aq_i8_s; the fp16 form's copy
// schedule (copies issued early by the S warps measured slower).
template <int T, bool MIX = false, bool I8 = false>
__device__ __forceinline__ void attn_core_q8_t(const LayerW & W, int slot, int n, int b0 = 0) { // b0: att_rank
    static_assert(!(MIX && I8), "int8 Q.K^T: all-q8 slots only");
    uint8_t * const sm = smem;
    constexpr int HG = T > 5 ? 3 : 6; // query heads per pass
    constexpr int R = HG * T;
    constexpr int MT = (R + 15) / 16; // 16-row MMA tiles of query rows
    constexpr int TP = ATP;
    constexpr int LDH = HD + 8;
    constexpr int LDP = TP + 8;
    constexpr int LDS = AQ_LDS;
    // 4 K stages in both forms (SM_ATTN_Q8 reserves them): MIX expands K(t) in interval t - 1; all-q8 slots issue K(t)
    // in t - 3 and wait for it at the end of t - 1, so every copy has two intervals to land (Nsight Compute, 40k
    // context, 5-row int8 attention: the copies issued one interval ahead were late, a WARPSYNC long_sb wait before the
    // interval barrier)
    constexpr int NSK = AQ_NSKM, NSV = AQ_NSV;
    static_assert(R <= 32 && TP == 32 && MT <= 2, "8 rows per softmax warp, 32-position tiles");
    const int L_ctx = n + T;
    const int ws = mtp_ws(slot, n); // MK_MTP_WIN
    int nch, CS;
    attn_dims_w(L_ctx, ws, nch, CS);
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g8 = lane >> 2, tig = lane & 3;
    __half * sQ = reinterpret_cast<__half *>(sm);                 // [32][LDH]
    int8_t * const sQ8 = reinterpret_cast<int8_t *>(sm);              // I8: int8 q rows [32][AQ_I8_LDQ] in sQ's place,
    float * const sQs = reinterpret_cast<float *>(sm + 32 * AQ_I8_LDQ); // then their block scales [8][32]
    static_assert(32 * AQ_I8_LDQ + 8 * 32 * 4 <= 32 * LDH * 2, "I8 q rows inside sQ");
    uint8_t * kring = reinterpret_cast<uint8_t *>(sQ + 32 * LDH); // NSK stages
    uint8_t * vring = kring + NSK * AQ_STG;                       // NSV stages
    float * sS = reinterpret_cast<float *>(vring + NSV * AQ_STG); // [2][32][LDS]; pad columns: m (32), l (33), alpha (34)
    __half * sP = reinterpret_cast<__half *>(sS + 2 * 32 * LDS);  // [2][32][LDP]
    auto sM = [&](int r) -> float & { return sS[r * LDS + 32]; };
    auto sL = [&](int r) -> float & { return sS[r * LDS + 33]; };
    auto sAl = [&](int b, int r) -> float & { return sS[(b * 32 + r) * LDS + 34]; };
    const size_t ctx = cB.ctx;
    const size_t slot0 = (size_t) slot * NKV * ctx; // first row of this slot
    (void) slot0;
    // MK_KV per-slot formats (MIX only): tiles are copied in the slot's format (per tile: kv_tbits, since with MK_KV_WIN the
    // tiles at positions >= wsr are the ring's q8 copies) and the low-bit ones expanded to int8 in their stage before use
    const int wsr = MIX ? kv_ws(L_ctx) : 0, wsm = MIX ? kv_wsm(wsr) : 0;
    const KvFmt kf = MIX ? kv_fmt(slot) : KvFmt{8, 8, 0, 0}; // per-slot formats, once per launch (kv_ring_tile per item)

    for (int ho = 0; ho < 6; ho += HG) // (T <= 5: one pass, ho = 0)
    for (int item = att_rank(b0); item < NKV * nch; item += gridDim.x) {
        const int g = item / nch, c = item % nch;
        int c0, c1;
        attn_chunk(c, ws, CS, L_ctx, c0, c1);
        // K / V tile copies (the ring's)
        const int ntl = (c1 - c0 + TP - 1) / TP;
        auto tnp = [&](int t) { return t < ntl ? min(TP, c1 - c0 - t * TP) : -1; }; // valid rows of tile t, -1: none
        // MIX: tiles jtr_k / jtr_v .. of this chunk come from the q8 rings (MK_KV_WIN, positions >= wsr), the others from
        // the slot's main buffer in its format: one compare per tile (kv_ring_tile)
        const int jtr_k = MIX ? kv_ring_tile(kf.kr, wsr, c0, TP) : 0x7fffffff, jtr_v = MIX ? kv_ring_tile(kf.vr, wsr, c0, TP) : 0x7fffffff;
        auto kbt = [&](int t) { return t >= jtr_k ? 8 : kf.kb; }; // format of K tile t (MIX; 8 otherwise)
        auto vbt = [&](int t) { return t >= jtr_v ? 8 : kf.vb; };
        auto issue = [&](int t_, uint8_t * dk, int tk, uint8_t * dv, int tv, int nth_tid) {
            const int8_t *kq, *vq;
            const __half *ks, *vs;
            int kb_ = 8, vb_ = 8;
            if constexpr (MIX) {
                kb_ = kbt(tk);
                vb_ = vbt(max(tv, 0));
                kv_tile_src(0, slot, g, c0 + tk * TP, wsr, wsm, kf.kb, tk >= jtr_k, kq, ks);
                kv_tile_src(1, slot, g, c0 + max(tv, 0) * TP, wsr, wsm, kf.vb, max(tv, 0) >= jtr_v, vq, vs);
            } else { // all-q8 slot: the plain layout
                const size_t kr = (size_t) g * ctx + c0 + tk * TP, vr = (size_t) g * ctx + c0 + max(tv, 0) * TP;
                kq = cB.kvp[0][slot] + kr * HD;
                ks = cB.kvs[0][slot] + kr * (HD / 32);
                vq = cB.kvp[1][slot] + vr * HD;
                vs = cB.kvs[1][slot] + vr * (HD / 32);
            }
            if (nth_tid == NT) aq_issue<NT, MIX>(t_, dk, kq, ks, tnp(tk), dv, vq, vs, tv < 0 ? -1 : tnp(tv), kb_, vb_);
            else aq_issue<NT - 128, MIX>(t_, dk, kq, ks, tnp(tk), dv, vq, vs, tv < 0 ? -1 : tnp(tv), kb_, vb_);
        };
#if MK_W48X
        // MIX: tiles below jff are full (aqx_side; the partial last tile of a chunk: aq_issue); the first row of this (kv head, chunk) in 32 bits
        const int jff = MIX ? (c1 - c0) / TP : 0;
        const unsigned rowb = (unsigned) g * (unsigned) ctx + (unsigned) c0;
        auto rrow = [&](int t) { // (kv_tile_src's ring row)
            int r0 = wsm + (c0 + t * TP - wsr);
            if (r0 >= cB.kvrc) r0 -= cB.kvrc;
            return (unsigned) g * cB.kvrc + r0;
        };
        auto issue_kx = [&](int tk, uint8_t * dk) { // by the 4 warps of a group (warp & 3)
            if (tk < jff) {
                if (tk < jtr_k) aqx_side<4>(warp & 3, lane, cB.kvp[0][slot], cB.kvs[0][slot], rowb + tk * TP, kf.kb, dk);
                else aqx_side<4>(warp & 3, lane, cB.kvrp[0][slot], cB.kvrs[0][slot], rrow(tk), 8, dk);
            } else {
                const int8_t * kq;
                const __half * ks;
                kv_tile_src(0, slot, g, c0 + tk * TP, wsr, wsm, kf.kb, tk >= jtr_k, kq, ks);
                aq_issue<128, true>((warp & 3) * 32 + lane, dk, kq, ks, tnp(tk), vring, kq, ks, -1, kbt(tk), 8);
            }
        };
        auto issue_vx = [&](int tv, uint8_t * dv) { // likewise
            if (tv < jff) {
                if (tv < jtr_v) aqx_side<4>(warp & 3, lane, cB.kvp[1][slot], cB.kvs[1][slot], rowb + tv * TP, kf.vb, dv);
                else aqx_side<4>(warp & 3, lane, cB.kvrp[1][slot], cB.kvrs[1][slot], rrow(tv), 8, dv);
            } else {
                const int8_t * vq;
                const __half * vs;
                kv_tile_src(1, slot, g, c0 + tv * TP, wsr, wsm, kf.vb, tv >= jtr_v, vq, vs);
                aq_issue<128, true>((warp & 3) * 32 + lane, kring, vq, vs, -1, dv, vq, vs, tnp(tv), 8, vbt(tv));
            }
        };
        auto issue_kp = [&](int tk, uint8_t * dk) { // by the 8 P V warps
            if (tk < jff) {
                if (tk < jtr_k) aqx_side<8>(warp - 4, lane, cB.kvp[0][slot], cB.kvs[0][slot], rowb + tk * TP, kf.kb, dk);
                else aqx_side<8>(warp - 4, lane, cB.kvrp[0][slot], cB.kvrs[0][slot], rrow(tk), 8, dk);
            } else {
                const int8_t * kq;
                const __half * ks;
                kv_tile_src(0, slot, g, c0 + tk * TP, wsr, wsm, kf.kb, tk >= jtr_k, kq, ks);
                aq_issue<256, true>((warp - 4) * 32 + lane, dk, kq, ks, tnp(tk), vring, kq, ks, -1, kbt(tk), 8);
            }
        };
        auto issue_vp = [&](int tv, uint8_t * dv) { // likewise
            if (tv < jff) {
                if (tv < jtr_v) aqx_side<8>(warp - 4, lane, cB.kvp[1][slot], cB.kvs[1][slot], rowb + tv * TP, kf.vb, dv);
                else aqx_side<8>(warp - 4, lane, cB.kvrp[1][slot], cB.kvrs[1][slot], rrow(tv), 8, dv);
            } else {
                const int8_t * vq;
                const __half * vs;
                kv_tile_src(1, slot, g, c0 + tv * TP, wsr, wsm, kf.vb, tv >= jtr_v, vq, vs);
                aq_issue<256, true>((warp - 4) * 32 + lane, kring, vq, vs, -1, dv, vq, vs, tnp(tv), 8, vbt(tv));
            }
        };
#endif
        auto issue012 = [&]() { // all-q8: commit groups {K(0)}, {K(1)}, {K(2), V(0)}
            issue(tid, kring, 0, vring, -1, NT);
            asm volatile("cp.async.commit_group;" ::: "memory");
            issue(tid, kring + AQ_STG, 1, vring, -1, NT);
            asm volatile("cp.async.commit_group;" ::: "memory");
            issue(tid, kring + 2 * AQ_STG, 2, vring, 0, NT);
            asm volatile("cp.async.commit_group;" ::: "memory");
        };
        // a chunk wholly below this launch's first new row n (no K / V row job reaches it; the MTP slot's catch-up rows are
        // stored phases before) holds the same bytes before and after the prologue: its first copies go out right after
        // the q loads and land under the prologue (MIX: after the prologue)
        const bool kv0 = !MIX && c1 <= n;
        // ---- item prologue, attn_core_t's (q rows -> sQ, this cycle's K / V rows -> cache, m / l init)
        // the q rows' loads first, then the q-norm and RoPE ones (Nsight Compute, 40k context, T5: the q loads went out
        // last, behind those at lg throttle, and the first row's math waited on them: ~4% of attn_core_q8<5>)
        const float * const qb = cB.qbuf;
        float qv[2][8], qw[8];
        float2 rc[2][8];
#pragma unroll
        for (int k2 = 0; k2 < 2; k2++) {
            if (k2 >= MT) continue;
            const int r = min(warp + 16 * k2, R - 1);
            load8(qb + (r % T) * QG + (g * 6 + ho + r / T) * 512 + lane * 8, qv[k2]);
        }
        load8_ro(W.qn + lane * 8, qw);
        constexpr bool RT1 = 16 % T == 0; // (attn_core_t's: one RoPE row per warp; k2 >= MT: rows no MMA tile reads)
#pragma unroll
        for (int k2 = 0; k2 < 2; k2++) {
            if (k2 >= MT) continue;
            const int r = min(warp + 16 * k2, R - 1);
            if (!RT1 || k2 == 0) {
                rope_row(rc[k2], n + (RT1 ? warp % T : r % T), lane);
            } else {
#pragma unroll
                for (int k = 0; k < 8; k++) rc[k2][k] = rc[0][k];
            }
        }
        if (kv0) issue012();
#pragma unroll
        for (int k2 = 0; k2 < 2; k2++) {
            if (k2 >= MT) continue;
            const int r = warp + 16 * k2;
            if (r < R) {
                float v[8];
#pragma unroll
                for (int k = 0; k < 8; k++) v[k] = qv[k2][k];
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
                if (cB.kvr[slot]) fwht256(v); // K of this slot is stored rotated (MK_KV "r")
                if constexpr (I8) {
                    aq_i8_q(v, sQ8 + r * AQ_I8_LDQ, sQs + r, 32);
                } else {
                __half2 hq2[4];
#pragma unroll
                for (int k = 0; k < 4; k++) hq2[k] = __floats2half2_rn(v[2 * k] * 0.0625f, v[2 * k + 1] * 0.0625f);
                *reinterpret_cast<int4 *>(sQ + r * LDH + lane * 8) = *reinterpret_cast<int4 *>(hq2);
                }
            } else {
                if constexpr (I8) aq_i8_q0(sQ8 + r * AQ_I8_LDQ, sQs + r, 32);
                else
                *reinterpret_cast<int4 *>(sQ + r * LDH + lane * 8) = make_int4(0, 0, 0, 0);
                if (2 * lane < TP) { // padded rows of both P buffers stay 0
                    *reinterpret_cast<unsigned *>(sP + r * LDP + 2 * lane) = 0u;
                    *reinterpret_cast<unsigned *>(sP + 32 * LDP + r * LDP + 2 * lane) = 0u;
                }
            }
        }
        const int pn0 = max(c0, n), np = ho ? 0 : min(c1, n + T) - pn0; // (the first head-group pass stores)
        for (int j = NW - 1 - warp; j < 2 * np; j += NW) { // attn_core_t's K / V row jobs
            const bool vr = j >= np;
            const int pp = pn0 + (vr ? j - np : j), t = pp - n;
            const size_t row = slot0 + (size_t) g * ctx + pp;
            (void) row;
            float v[8];
            if (!vr) {
                load8(cB.kbuf + t * KVD + g * 256 + lane * 8, v);
                float ss = 0.f;
#pragma unroll
                for (int k = 0; k < 8; k++) ss += v[k] * v[k];
                const float rs = rsqrtf(wsum(ss) / 256.0f + EPS);
#pragma unroll
                for (int k = 0; k < 8; k++) v[k] *= rs * W.kn[lane * 8 + k];
                rope8(v, lane, pp);
#if MK_KV_EMU
                if (cB.kvemu[0][slot]) kv_store_emu(static_cast<int8_t *>(cB.kc) + row * HD, cB.ksc + row * (HD / 32), v, cB.kvemu[0][slot], pp);
                else
#endif
                kv_store(0, slot, g, pp, v);
            } else {
                load8(cB.vbuf + t * KVD + g * 256 + lane * 8, v);
#if MK_KV_EMU
                if (cB.kvemu[1][slot]) kv_store_emu(static_cast<int8_t *>(cB.vc) + row * HD, cB.vsc + row * (HD / 32), v, cB.kvemu[1][slot], pp);
                else
#endif
                kv_store(1, slot, g, pp, v);
            }
        }
        if (tid < 32) {
            sM(tid) = -INFINITY;
            sL(tid) = 0.f;
        }
        if (kv0) asm volatile("cp.async.wait_group 2;" ::: "memory"); // (one barrier for the prologue and K(0))
        __threadfence_block();
        __syncthreads();

        // ---- ring prologue: commit groups {K(0)}, {K(1)}, {K(2), V(0)} (kv0: already out); then per interval jt {K(jt + 3),
        // V(jt + 1)}, at its end waiting for {K(jt + 1), V(jt - 1)} (cp.async.wait_group 2)
        // (MIX: {K(0), K(1)}, {K(2), V(0)}; per interval {K(jt + 3), V(jt + 1)}, waiting for {K(jt + 2), V(jt)}, and the
        // low-bit tiles that landed at the end of interval jt - 1, K(jt + 1) and V(jt - 1), are expanded to int8 in interval jt)
        if constexpr (MIX) {
#if MK_W48X
            // Warp groups 0 - 3, 12 - 15: S(t) on the group of t's parity; in t + 1 the same group runs softmax(t) and converts K(t + 2)
            // (AQX_CP, AQX_VH: who else does what). K(t): issued in interval t - 4, landed by the end of t - 3, to bytes in t - 2,
            // converted to fp16 fragments in t - 1, its stage free from then on (S(t) reads the fragments in registers). V(t): issued
            // in t - 1, landed by the end of t, to bytes in t + 1, read in t + 2. So S's conversion runs under the other group's
            // MMAs and each copy has at least one interval to land. Prologue: K(0) .. K(3) and V(0) landed, K(0) and K(1) -> bytes
            // (all warps), K(0) converted.
            issue(tid, kring, 0, vring, -1, NT);
            issue(tid, kring + AQ_STG, 1, vring, -1, NT);
            issue(tid, kring + 2 * AQ_STG, 2, vring, -1, NT);
            issue(tid, kring + 3 * AQ_STG, 3, vring, 0, NT);
            asm volatile("cp.async.commit_group;" ::: "memory");
            asm volatile("cp.async.wait_group 0;" ::: "memory");
#else
            issue(tid, kring, 0, vring, -1, NT);
            issue(tid, kring + AQ_STG, 1, vring, -1, NT);
            asm volatile("cp.async.commit_group;" ::: "memory");
            issue(tid, kring + 2 * AQ_STG, 2, vring, 0, NT);
            asm volatile("cp.async.commit_group;" ::: "memory");
            asm volatile("cp.async.wait_group 1;" ::: "memory");
#endif
            __syncthreads();
            const int kb0 = kbt(0); // K(0) -> bytes now (all warps); K(1) in interval 0
#if MK_W48X
            const int kb1 = ntl > 1 ? kbt(1) : 8;
            if (kb1 != 8) kv_tile_expand<false>(kring + AQ_STG, kb1, TP, warp, NW, KVQ_ROW);
#endif
            if (kb0 != 8) {
                kv_tile_expand<false>(kring, kb0, TP, warp, NW, KVQ_ROW);
#if MK_W48X
            }
            if (kb0 != 8 || kb1 != 8) {
#endif
                __syncthreads();
            }
        } else if (!kv0) {
            issue012();
            asm volatile("cp.async.wait_group 2;" ::: "memory");
            __syncthreads();
        }

#if MK_W48X
        constexpr int MTA = MIX ? (MT > 2 ? MT : 2) : MT; // MIX: the S / softmax groups keep their 32 B fragments here
#else
        constexpr int MTA = MT;
#endif
        float acc[MTA][4][4]; // P V warps: O rows 16 i + {g8, g8 + 8}, dims 32 wn + 8 tig + jj and + 4
#pragma unroll
        for (int i = 0; i < MTA; i++)
#pragma unroll
            for (int j = 0; j < 4; j++)
#pragma unroll
                for (int k = 0; k < 4; k++) acc[i][j][k] = 0.f;

#if MK_W48X
        if constexpr (MIX) { // (a barrier: interval 0's first copy goes into K(0)'s stage)
            if (warp < 4) {
                const int kbj = kbt(0);
                aqx_conv(acc, kring + (8 * warp + g8) * KVQ_ROW, tig, kbj == 8 ? 0x8080u : 0u, aq_sub(kbj));
            }
            __syncthreads();
        }
        for (int jt = 0; jt < ntl + 2; jt++) {
            // MIX: the two warp groups (one warp of each per SMSP), rC: this interval's converting group
            const int sg = warp < 4 ? 0 : warp >= 12 ? 1 : -1;
            const bool rC = sg >= 0 && sg != (jt & 1);
            if constexpr (MIX) {
                if (rC) {
                    if constexpr (T > AQX_CP) {
                        if (jt + 4 < ntl) issue_kx(jt + 4, kring + ((jt + 4) % NSK) * AQ_STG);
                        if (jt + 1 < ntl) issue_vx(jt + 1, vring + ((jt + 1) % NSV) * AQ_STG);
                    }
                    if (jt + 1 < ntl) {
                        const int kbj = kbt(jt + 1);
                        aqx_conv(acc, kring + ((jt + 1) % NSK) * AQ_STG + (8 * (warp & 3) + g8) * KVQ_ROW, tig, kbj == 8 ? 0x8080u : 0u, aq_sub(kbj));
                    }
                } else if (sg >= 0) {
                    if (jt < ntl) {
                        float s[MT][4];
                        aqx_mma<MT, MTA, LDH>(s, acc, (unsigned) __cvta_generic_to_shared(sQ + (lane & 15) * LDH + (lane >> 4) * 8));
#pragma unroll
                        for (int i = 0; i < MT; i++) {
                            float * so = sS + (jt & 1) * 32 * LDS + (16 * i + g8) * LDS + 8 * (warp & 3) + 2 * tig;
                            so[0] = s[i][0];
                            so[1] = s[i][1];
                            so[8 * LDS] = s[i][2];
                            so[8 * LDS + 1] = s[i][3];
                        }
                    }
                    if constexpr (T <= AQX_VH) {
                        if (jt >= 1 && jt - 1 < ntl) {
                            const int vbx = vbt(jt - 1);
                            if (vbx != 8) kvx_tile<8>(vring + ((jt - 1) % NSV) * AQ_STG, vbx, 8 * (warp & 3));
                        }
                    }
                }
            }
            if (!MIX && warp < 4) {
#else
        for (int jt = 0; jt < ntl + 2; jt++) {
            if (warp < 4) {
#endif
                // ---- S(jt) = Q K(jt)^T: B = positions 8 wn + g8 (all 16 k-steps converted first), all row tiles
                if (jt < ntl) {
                    const int wn = warp;
                    if constexpr (I8) { // int8 Q.K^T (aq_i8_s: the S of every I8 kernel)
                        float s[MT][4];
                        aq_i8_s<MT, 32>(s, sQ8, sQs, kring + (jt % NSK) * AQ_STG, wn);
#pragma unroll
                        for (int i = 0; i < MT; i++) {
                            float * so = sS + (jt & 1) * 32 * LDS + (16 * i + g8) * LDS + 8 * wn + 2 * tig;
                            so[0] = s[i][0];
                            so[1] = s[i][1];
                            so[8 * LDS] = s[i][2];
                            so[8 * LDS + 1] = s[i][3];
                        }
                    } else {
                    const uint8_t * krow = kring + (jt % NSK) * AQ_STG + (8 * wn + g8) * KVQ_ROW;
                    const uint4 scv = *reinterpret_cast<const uint4 *>(krow + 256);
                    const unsigned scw[4] = {scv.x, scv.y, scv.z, scv.w};
                    // MIX: a low-bit tile was expanded to biased bytes n = q + 2^(b-1) in interval jt - 1; its xm / sub make
                    // the conversion below yield the same fp16 q (aq_q8x2). All-q8: the constants fold.
                    const int kbj = kbt(jt);
                    const unsigned kxm = kbj == 8 ? 0x8080u : 0u, ksub = aq_sub(kbj);
                    unsigned kb[HD / 16][2];
#pragma unroll
                    for (int ks = 0; ks < HD / 16; ks++) {
                        const unsigned sw = __byte_perm(scw[ks >> 2], 0, (ks & 2) ? 0x3232 : 0x1010);
                        const __half2 dd = *reinterpret_cast<const __half2 *>(&sw);
                        kb[ks][0] = aq_q8x2(*reinterpret_cast<const unsigned short *>(krow + 16 * ks + 2 * tig), dd, kxm, ksub);
                        kb[ks][1] = aq_q8x2(*reinterpret_cast<const unsigned short *>(krow + 16 * ks + 2 * tig + 8), dd, kxm, ksub);
                    }
                    float s[MT][4];
#pragma unroll
                    for (int i = 0; i < MT; i++)
#pragma unroll
                        for (int k = 0; k < 4; k++) s[i][k] = 0.f;
                    const unsigned qaddr = (unsigned) __cvta_generic_to_shared(sQ + (lane & 15) * LDH + (lane >> 4) * 8);
#pragma unroll
                    for (int ks = 0; ks < HD / 16; ks++) {
#pragma unroll
                        for (int i = 0; i < MT; i++) {
                            unsigned a[4];
                            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                                         : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3])
                                         : "r"(qaddr + 2 * (16 * i * LDH + 16 * ks)));
                            mma16816(s[i], a[0], a[1], a[2], a[3], kb[ks][0], kb[ks][1]);
                        }
                    }
#pragma unroll
                    for (int i = 0; i < MT; i++) {
                        float * so = sS + (jt & 1) * 32 * LDS + (16 * i + g8) * LDS + 8 * wn + 2 * tig;
                        so[0] = s[i][0];
                        so[1] = s[i][1];
                        so[8 * LDS] = s[i][2];
                        so[8 * LDS + 1] = s[i][3];
                    }
                    }
                }
#if MK_W48X
            } else if (warp >= 4 && warp < 12) {
#else
            } else if (warp < 12) {
#endif
                // ---- O = O * alpha + P(jt - 2) V(jt - 2): B = dims 32 wn + 4 g8 + jj of n8 tile jj, all row tiles
                if (jt >= 2) {
                    const int wn = warp - 4, jp = jt - 2, b = jp & 1;
#pragma unroll
                    for (int i = 0; i < MT; i++) {
                        const float al0 = sAl(b, 16 * i + g8), al1 = sAl(b, 16 * i + g8 + 8);
#pragma unroll
                        for (int j = 0; j < 4; j++) {
                            acc[i][j][0] *= al0;
                            acc[i][j][1] *= al0;
                            acc[i][j][2] *= al1;
                            acc[i][j][3] *= al1;
                        }
                    }
                    const unsigned paddr = (unsigned) __cvta_generic_to_shared(sP + b * 32 * LDP + (lane & 15) * LDP + (lane >> 4) * 8);
                    // (MIX: a low-bit V tile was expanded to biased bytes in interval jt - 1: xm / sub as for K)
                    const uint8_t * vst = vring + (jp % NSV) * AQ_STG + 32 * wn + 4 * g8;
                    const uint8_t * vsc = vring + (jp % NSV) * AQ_STG + 256 + 2 * wn;
                    const int vbj = vbt(jp);
                    const unsigned vxm = vbj == 8 ? 0x80808080u : 0u, vsub = aq_sub(vbj);
#pragma unroll
                    for (int ks = 0; ks < TP / 16; ks++) {
                        unsigned bf[2][4]; // [b0 / b1][tile jj]
#pragma unroll
                        for (int h = 0; h < 2; h++) {
                            const int p = 16 * ks + 2 * tig + 8 * h; // k rows p, p + 1 of the fragment
                            const unsigned xa = *reinterpret_cast<const unsigned *>(vst + p * KVQ_ROW) ^ vxm;
                            const unsigned xb = *reinterpret_cast<const unsigned *>(vst + (p + 1) * KVQ_ROW) ^ vxm;
                            const unsigned sw = (unsigned) *reinterpret_cast<const unsigned short *>(vsc + p * KVQ_ROW) |
                                                ((unsigned) *reinterpret_cast<const unsigned short *>(vsc + (p + 1) * KVQ_ROW) << 16);
                            const __half2 dd = *reinterpret_cast<const __half2 *>(&sw);
                            const unsigned y0 = __byte_perm(xa, xb, 0x5140), y1 = __byte_perm(xa, xb, 0x7362);
                            bf[h][0] = aq_q8x2p(y0, 0x4140, dd, vsub);
                            bf[h][1] = aq_q8x2p(y0, 0x4342, dd, vsub);
                            bf[h][2] = aq_q8x2p(y1, 0x4140, dd, vsub);
                            bf[h][3] = aq_q8x2p(y1, 0x4342, dd, vsub);
                        }
#pragma unroll
                        for (int i = 0; i < MT; i++) {
                            unsigned a[4];
                            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                                         : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3])
                                         : "r"(paddr + 2 * (16 * i * LDP + 16 * ks)));
#pragma unroll
                            for (int j = 0; j < 4; j++) mma16816(acc[i][j], a[0], a[1], a[2], a[3], bf[0][j], bf[1][j]);
                        }
                    }
                }
#if MK_W48X
            } else if (!MIX || rC) {
                const int sn_ = MIX ? warp & 3 : warp - 12;
#else
            } else {
                const int sn_ = warp - 12;
#endif
                // ---- online softmax of tile jt - 1 (causal mask: row token t sees positions <= n + t); lane l4 of the
                // row's quad holds positions l4 + 4 k
                if (jt >= 1 && jt <= ntl && 8 * sn_ < R) {
                    const int r = 8 * sn_ + (lane >> 2), l4 = lane & 3; // rows >= R (zero S) are not stored
                    const int jp = jt - 1, b = jp & 1;
                    const int tb = c0 + jp * TP, np_ = min(TP, c1 - tb);
                    const int lim = n + (r % T);
                    const float * srow = sS + b * 32 * LDS + r * LDS;
                    float sv[8];
#pragma unroll
                    for (int k = 0; k < 8; k++) {
                        const int q = l4 + 4 * k;
                        sv[k] = srow[q];
                        if (q >= np_ || tb + q > lim) sv[k] = -INFINITY;
                    }
                    float m = sv[0];
#pragma unroll
                    for (int k = 1; k < 8; k++) m = fmaxf(m, sv[k]);
                    m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 2));
                    m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 1));
                    const float mold = sM(r);
                    const float mnew = fmaxf(mold, m);
                    float p[8], al = 1.f;
#pragma unroll
                    for (int k = 0; k < 8; k++) p[k] = 0.f;
                    if (mnew != -INFINITY) {
#pragma unroll
                        for (int k = 0; k < 8; k++) p[k] = expf(sv[k] - mnew);
                        al = expf(mold - mnew);
                    }
                    // attn_core_t's wsum tree: position bit 4 (k, k + 4), bit 3 (k, k + 2), bit 2 (k, k + 1), then lanes
                    const float t0 = p[0] + p[4], t1 = p[1] + p[5], t2 = p[2] + p[6], t3 = p[3] + p[7];
                    const float u0 = t0 + t2, u1 = t1 + t3;
                    float ps = u0 + u1;
                    ps += __shfl_xor_sync(0xffffffffu, ps, 2);
                    ps += __shfl_xor_sync(0xffffffffu, ps, 1);
#if MK_FORMAL_SYNC
                    __syncwarp(); // the quad's sM(r) reads before lane l4 == 0 rewrites it (shfl.sync orders no memory)
#endif
                    if (r < R) {
                        __half * prow = sP + b * 32 * LDP + r * LDP;
#pragma unroll
                        for (int k = 0; k < 8; k++) prow[l4 + 4 * k] = __float2half_rn(p[k]);
                        if (l4 == 0) {
                            sL(r) = sL(r) * al + ps;
                            sM(r) = mnew;
                            sAl(b, r) = al;
                        }
                    }
                }
            }
            // ---- copies: K(jt + 3) into the stage of K(jt - 1), V(jt + 1) into the stage of V(jt - 3) (both read in
            // interval jt - 1), issued by warps 4-15 (the S warps carry the longest MMA chains);
            // then wait for K(jt + 1) and V(jt - 1) (MIX: K(jt + 2) and V(jt), expanded in interval jt + 1)
            if (warp >= 4) {
                if constexpr (MIX) {
#if MK_W48X
                    if (warp < 12) {
                        // K(jt + 2) and V(jt - 1) -> bytes (both landed by the end of jt - 1; all TP rows: stale ones past np stay finite
                        // under zero scales)
                        if (jt + 2 < ntl) {
                            const int kbx = kbt(jt + 2);
                            if (kbx != 8) kvx_tile<4>(kring + ((jt + 2) % NSK) * AQ_STG, kbx, 4 * (warp - 4));
                        }
                        if constexpr (T > AQX_VH) {
                            if (jt >= 1 && jt - 1 < ntl) {
                                const int vbx = vbt(jt - 1);
                                if (vbx != 8) kvx_tile<4>(vring + ((jt - 1) % NSV) * AQ_STG, vbx, 4 * (warp - 4));
                            }
                        }
                        if constexpr (T <= AQX_CP) { // the copies of interval jt - 1 landed; the next ones
                            asm volatile("cp.async.wait_group 0;" ::: "memory");
                            if (jt + 4 < ntl) issue_kp(jt + 4, kring + ((jt + 4) % NSK) * AQ_STG);
                            if (jt + 1 < ntl) issue_vp(jt + 1, vring + ((jt + 1) % NSV) * AQ_STG);
                        }
                    }
#else
                    // K(jt + 3) into the stage of K(jt - 1); then the
                    // low-bit tiles that landed at the end of interval jt - 1 are expanded to bytes for interval jt + 1: K(jt + 1)
                    // (S warps) and V(jt - 1) (P V warps), up to 3 rows per warp per call; all TP rows (rows >= np hold stale
                    // but finite bytes under zero scales: masked S, P = 0). Their stages are not touched by any copy in flight.
                    if (jt + 1 < ntl) issue(tid - 128, kring + ((jt + 3) % NSK) * AQ_STG, jt + 3, vring + ((jt + 1) % NSV) * AQ_STG, jt + 1, NT - 128);
                    // row split of each tile: rows round-robin over the 12 warps 4-15
                    int r0 = warp - 4, rs = NW - 4, r1 = TP;
                    if (jt + 1 < ntl) {
                        const int kb_ = kbt(jt + 1);
                        if (kb_ != 8) kv_tile_expand<false>(kring + ((jt + 1) % NSK) * AQ_STG, kb_, r1, r0, rs, KVQ_ROW);
                    }
                    if (jt >= 1 && jt - 1 < ntl) {
                        const int vb_ = vbt(jt - 1);
                        if (vb_ != 8) kv_tile_expand<false>(vring + ((jt - 1) % NSV) * AQ_STG, vb_, r1, r0, rs, KVQ_ROW);
                    }
#endif
                } else {
                    if (jt + 1 < ntl) issue(tid - 128, kring + ((jt + 3) % NSK) * AQ_STG, jt + 3, vring + ((jt + 1) % NSV) * AQ_STG, jt + 1, NT - 128);
                }
            }
            asm volatile("cp.async.commit_group;" ::: "memory");
#if MK_W48X
            if constexpr (MIX) { // (the copies this group issued in jt - 1: expanded in jt + 1)
                if constexpr (T > AQX_CP) {
                    if (sg >= 0 && !rC) asm volatile("cp.async.wait_group 0;" ::: "memory");
                }
            } else
#endif
            if constexpr (MIX) asm volatile("cp.async.wait_group 1;" ::: "memory");
            else asm volatile("cp.async.wait_group 2;" ::: "memory");
            __syncthreads();
        }
        if (warp >= 4 && warp < 12) {
            const int wn = warp - 4;
            // tile j's columns 2 tig, 2 tig + 1 are dims d0 + j, d0 + j + 4: the lane's 8 dims of a row are one 32-byte
            // sector, stored as two 16-byte words (8 scalar stores per row had the LSU queue full: lg throttle)
#pragma unroll
            for (int i = 0; i < MT; i++) {
                const int r0 = 16 * i + g8, d0 = 32 * wn + 8 * tig;
                if (r0 < R) {
                    float4 * o = reinterpret_cast<float4 *>(cB.apart + ((size_t) item * AR + ho * T + r0) * 256 + d0);
                    o[0] = make_float4(acc[i][0][0], acc[i][1][0], acc[i][2][0], acc[i][3][0]);
                    o[1] = make_float4(acc[i][0][1], acc[i][1][1], acc[i][2][1], acc[i][3][1]);
                }
                if (r0 + 8 < R) {
                    float4 * o = reinterpret_cast<float4 *>(cB.apart + ((size_t) item * AR + ho * T + r0 + 8) * 256 + d0);
                    o[0] = make_float4(acc[i][0][2], acc[i][1][2], acc[i][2][2], acc[i][3][2]);
                    o[1] = make_float4(acc[i][0][3], acc[i][1][3], acc[i][2][3], acc[i][3][3]);
                }
            }
        }
        if (tid < R) cB.aml[item * AR + ho * T + tid] = make_float2(sM(tid), sL(tid));
        __syncthreads();
    }
}

template <int T>
__device__ __noinline__ void attn_core_q8(const LayerW & W, int slot, int n, int b0) {
    attn_core_q8_t<T>(W, slot, n, b0);
}

template <int T>
__device__ __noinline__ void attn_core_q8m(const LayerW & W, int slot, int n, int b0) {
    attn_core_q8_t<T, true>(W, slot, n, b0);
}

#if MK_ATT_I8
// MK_ATT_I8 instances (all-q8 slots), out of line: attn_core's fp16 forms keep their code
template <int T>
__device__ __noinline__ void attn_core_q8i(const LayerW & W, int slot, int n, int b0) {
    attn_core_q8_t<T, false, true>(W, slot, n, b0);
}
template <int T>
__device__ __noinline__ void attn_core_ti(const LayerW & W, int slot, int n, int b0) {
    attn_core_t<T, true, true>(W, slot, n, b0);
}
#endif

// attn_core_w48 (MIX): P V warp wn expands dims 32 wn .. 32 wn + 31 of all 32 staged rows of a low-bit V tile in place to the
// biased bytes n (kv_rows_expand<BITS, false>'s bytes): lane = row, its lane words 4 wn .. 4 wn + 3 and their high bits are
// loaded first, then a named barrier of the 8 P V warps (a slice's output overlaps other slices' input), then the stores.
template <int BITS>
__device__ __forceinline__ void kv_slice_expand(uint8_t * st, int wn) {
    constexpr int HB = BITS - 4;
    const int lane = threadIdx.x & 31;
    uint8_t * row = st + lane * KVQ_ROW;
    const uint4 w = *reinterpret_cast<const uint4 *>(row + 16 * wn);
    unsigned h[4] = {0u, 0u, 0u, 0u};
    if constexpr (HB == 1) {
        const unsigned x = *reinterpret_cast<const unsigned *>(row + 128 + 4 * wn);
#pragma unroll
        for (int k = 0; k < 4; k++) h[k] = (x >> (8 * k)) & 0xffu;
    }
    if constexpr (HB == 2) {
        const uint2 x = *reinterpret_cast<const uint2 *>(row + 128 + 8 * wn);
        h[0] = x.x & 0xffffu;
        h[1] = x.x >> 16;
        h[2] = x.y & 0xffffu;
        h[3] = x.y >> 16;
    }
    if constexpr (HB == 3) {
        const unsigned * p = reinterpret_cast<const unsigned *>(row + 128 + 12 * wn);
        const unsigned a = p[0], b = p[1], c = p[2];
        h[0] = a & 0xffffffu;
        h[1] = (a >> 24) | ((b & 0xffffu) << 8);
        h[2] = (b >> 16) | ((c & 0xffu) << 16);
        h[3] = c >> 8;
    }
    asm volatile("bar.sync 1, 256;" ::: "memory");
    const unsigned wd[4] = {w.x, w.y, w.z, w.w};
    unsigned o[8];
#pragma unroll
    for (int k = 0; k < 4; k++) {
        unsigned lo = wd[k] & 0x0f0f0f0fu, hv = (wd[k] >> 4) & 0x0f0f0f0fu;
        if constexpr (HB > 0) {
            lo |= kv_spread4<HB>(h[k]);
            hv |= kv_spread4<HB>(h[k] >> (4 * HB));
        }
        o[2 * k] = lo;
        o[2 * k + 1] = hv;
    }
    *reinterpret_cast<uint4 *>(row + 32 * wn) = make_uint4(o[0], o[1], o[2], o[3]);
    *reinterpret_cast<uint4 *>(row + 32 * wn + 16) = make_uint4(o[4], o[5], o[6], o[7]);
}

// T >= 6: decode attention over all R = 6 T <= 48 query rows of a kv head in ONE pass, so K / V are read
// once (attn_core_q8_t's two head-group passes read them twice). Shared memory: 48-row sQ, sS,
// sP, 3 K and 3 V stages (99,072 B): V(t) is issued in interval t (not t - 1), lands by the end of t + 1, read in t + 2. MIX:
// no expansion stage; the consumers expand in place right before use: S warp wn its K rows 8 wn .. 8 wn + 7 of K(jt)
// (kv_rows_expand, row-local), P V warp wn its dims of V(jt - 2) (kv_slice_expand), both to the biased bytes the MIX
// conversions take (xm / sub). The softmax warps take 12 rows each (two sub-passes). Per-row arithmetic is attn_core_t's
// (the same S / softmax / P V code per row): the partials equal attn_core_q8_t's and attn_core_t's for every format.
constexpr int SM_ATTN_W48 = 48 * (256 + 8) * 2 + 6 * AQ_STG + 2 * 48 * AQ_LDS * 4 + 2 * 48 * (ATP + 8) * 2;
static_assert(SM_ATTN_W48 + NW * 16 * 4 <= SM_TOTAL, "attn_core_w48 shared memory (+ the per-warp call scalars)");
template <int T, bool MIX, bool I8 = false>
__device__ __forceinline__ void attn_core_w48(const LayerW & W, int slot, int n, int nw = -1, int b0 = 0) { // nw: first K / V row written; b0: att_rank
    uint8_t * const sm = smem;
    constexpr int R = 6 * T, RP = 48;
    constexpr int MT = 3; // 16-row MMA tiles of query rows
    constexpr int TP = ATP;
    constexpr int LDH = HD + 8;
    constexpr int LDP = TP + 8;
    constexpr int LDS = AQ_LDS;
    constexpr int NSK = 3, NSV = 3;
    static_assert(R > 32 && R <= RP && TP == 32, "6 T in 33 .. 48 rows, 32-position tiles");
    const int L_ctx = n + T;
    const int ws = mtp_ws(slot, n); // MK_MTP_WIN
    int nch, CS;
    attn_dims_w(L_ctx, ws, nch, CS);
    // the call's scalars in shared memory, one copy per warp (16 words: the 7 ints, then cB.qbuf at words 8, 9), read
    // with volatile loads where used: kept in registers, ptxas spills them and their reload at each item's start waits on
    // local memory (L2) behind the K / V stream
    if ((threadIdx.x & 31) < 7) {
        const int l = threadIdx.x & 31;
        reinterpret_cast<int *>(sm + SM_ATTN_W48)[16 * (threadIdx.x >> 5) + l] =
            l == 0 ? n : l == 1 ? ws : l == 2 ? CS : l == 3 ? L_ctx : l == 4 ? nch : l == 5 ? (nw < 0 ? n : nw) : att_rank(b0);
    }
    if ((threadIdx.x & 31) == 8) reinterpret_cast<const float **>(sm + SM_ATTN_W48)[8 * (threadIdx.x >> 5) + 4] = cB.qbuf;
    __syncwarp();
    auto w48s = [&](int i) -> int { return reinterpret_cast<volatile const int *>(sm + SM_ATTN_W48)[16 * (tid_x() >> 5) + i]; };
    // the q buffer from the scalar block: cB is cBs[s_view], so cB.qbuf at the item start was an LDS (queued behind the
    // q-norm / RoPE loads) -> LDC -> the q rows' address chain
    auto w48qb = [&]() -> const float * {
        return reinterpret_cast<const float * const volatile *>(sm + SM_ATTN_W48)[8 * (tid_x() >> 5) + 4];
    };
    if (nw < 0 && slot < N_ATT_SLOTS - 1 && L_ctx <= MK_ATT_PF_CTX) { // a block without items: resid_o's prefetch (deep
                                                                         // pass: its first call only)
        const int rk = att_rank(b0), ni = NKV * nch;
        if (rk >= ni) {
            attn_idle_pf(W.o, rk - ni, (int) gridDim.x - ni);
            return;
        }
    }
#define W48_N w48s(0)
#define W48_WS w48s(1)
#define W48_CS w48s(2)
#define W48_LC w48s(3)
#define W48_NCH w48s(4)
#define W48_NW w48s(5)
#define W48_RANK w48s(6)
    __half * sQ = reinterpret_cast<__half *>(sm);                 // [RP][LDH]
    int8_t * const sQ8 = reinterpret_cast<int8_t *>(sm);              // I8: int8 q rows [RP][AQ_I8_LDQ] in sQ's place,
    float * const sQs = reinterpret_cast<float *>(sm + RP * AQ_I8_LDQ); // then their block scales [8][RP]
    static_assert(RP * AQ_I8_LDQ + 8 * RP * 4 <= RP * LDH * 2, "I8 q rows inside sQ");
    uint8_t * kring = reinterpret_cast<uint8_t *>(sQ + RP * LDH); // NSK stages
    uint8_t * vring = kring + NSK * AQ_STG;                       // NSV stages
    float * sS = reinterpret_cast<float *>(vring + NSV * AQ_STG); // [2][RP][LDS]; pad columns: m (32), l (33), alpha (34)
    __half * sP = reinterpret_cast<__half *>(sS + 2 * RP * LDS);  // [2][RP][LDP]
    auto sM = [&](int r) -> float & { return sS[r * LDS + 32]; };
    auto sL = [&](int r) -> float & { return sS[r * LDS + 33]; };
    auto sAl = [&](int b, int r) -> float & { return sS[(b * RP + r) * LDS + 34]; };
    const size_t ctx = cB.ctx;
    const size_t slot0 = (size_t) slot * NKV * ctx; // first row of this slot
    (void) slot0;
    const int wsr = MIX ? kv_ws(L_ctx) : 0, wsm = MIX ? kv_wsm(wsr) : 0;
    const KvFmt kf = MIX ? kv_fmt(slot) : KvFmt{8, 8, 0, 0};

    for (int item = W48_RANK; item < NKV * W48_NCH; item += gridDim.x) {
        // thread indices per item (tid_x): hoisted out of the item loop, their derived offsets are spilled and the
        // prologue's reload misses L1
        const int tid = tid_x(), lane = tid & 31, warp = tid >> 5, g8 = lane >> 2, tig = lane & 3;
        const int nch_ = W48_NCH, g = item / nch_, c = item % nch_;
        int c0, c1;
        attn_chunk(c, W48_WS, W48_CS, W48_LC, c0, c1);
        // ---- item prologue, attn_core_t's (q rows -> sQ, this cycle's K / V rows -> cache, m / l init)
        // The three q rows' loads first, then the q-norm and RoPE ones (Nsight Compute, 40k context, T8: the q rows
        // went out last, behind those and the cB.qbuf chain; the first row's sum waited ~3.5% of the kernel at
        // long_sb). Every row is loaded unconditionally (r clamped; for T = 8 every r < R): a row array assigned under
        // `r < R` was carried across the item loop in local memory (an earlier version).
        float qv[3][8];
        {
            const float * const qb = w48qb();
#pragma unroll
            for (int k2 = 0; k2 < 3; k2++) {
                const int r = min(warp + 16 * k2, R - 1);
                load8(qb + (r % T) * QG + (g * 6 + r / T) * 512 + lane * 8, qv[k2]);
            }
        }
        float qw[8];
        load8_ro(W.qn + lane * 8, qw);
        // T = 8: the warp's q rows warp + 16 k2 are all token warp % 8, one RoPE row for the three (rope_row)
        constexpr bool RT1 = 16 % T == 0;
        float2 rcw[8];
        if constexpr (RT1) rope_row(rcw, W48_N + warp % T, lane);
        (void) rcw;
#pragma unroll
        for (int k2 = 0; k2 < 3; k2++) {
            const int r = warp + 16 * k2;
            if (r < R) {
                const int t = r % T;
                float v[8]; // (a copy, not a pointer into qv: attn_core_t's note)
#pragma unroll
                for (int k = 0; k < 8; k++) v[k] = qv[k2][k];
                float2 rc[8];
                if constexpr (RT1) {
#pragma unroll
                    for (int k = 0; k < 8; k++) rc[k] = rcw[k];
                } else {
                    rope_row(rc, W48_N + t, lane);
                }
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
                        v[k] = lane < 4 ? v[k] * rc[k].x - o[k] * rc[k].y : o[k] * rc[k].y + v[k] * rc[k].x;
                }
                if (cB.kvr[slot]) fwht256(v); // K of this slot is stored rotated (MK_KV "r")
                if constexpr (I8) {
                    aq_i8_q(v, sQ8 + r * AQ_I8_LDQ, sQs + r, RP);
                } else {
                __half2 hq2[4];
#pragma unroll
                for (int k = 0; k < 4; k++) hq2[k] = __floats2half2_rn(v[2 * k] * 0.0625f, v[2 * k + 1] * 0.0625f);
                *reinterpret_cast<int4 *>(sQ + r * LDH + lane * 8) = *reinterpret_cast<int4 *>(hq2);
                }
            } else {
                if constexpr (I8) aq_i8_q0(sQ8 + r * AQ_I8_LDQ, sQs + r, RP);
                else
                *reinterpret_cast<int4 *>(sQ + r * LDH + lane * 8) = make_int4(0, 0, 0, 0);
                if (2 * lane < TP) { // padded rows of both P buffers stay 0
                    *reinterpret_cast<unsigned *>(sP + r * LDP + 2 * lane) = 0u;
                    *reinterpret_cast<unsigned *>(sP + RP * LDP + r * LDP + 2 * lane) = 0u;
                }
            }
        }
        const int nn = W48_N, pn0 = max(c0, MAXD > MAXT ? W48_NW : nn), np = min(c1, nn + T) - pn0; // (rows nw .. n - 1: view rows < 0)
        // K / V rows as jobs (attn_core_t's): j < np the K row of position pn0 + j, j >= np the V row of pn0 + j - np
        for (int j = NW - 1 - warp; j < 2 * np; j += NW) {
            const bool vr = j >= np;
            const int pp = pn0 + (vr ? j - np : j), t = pp - nn;
            float v[8];
#if MK_KV_EMU
            const size_t row = slot0 + (size_t) g * ctx + pp;
#endif
            if (!vr) {
                load8(cB.kbuf + t * KVD + g * 256 + lane * 8, v);
                float ss = 0.f;
#pragma unroll
                for (int k = 0; k < 8; k++) ss += v[k] * v[k];
                const float rs = rsqrtf(wsum(ss) / 256.0f + EPS);
#pragma unroll
                for (int k = 0; k < 8; k++) v[k] *= rs * W.kn[lane * 8 + k];
                rope8(v, lane, pp);
#if MK_KV_EMU
                if (cB.kvemu[0][slot]) kv_store_emu(static_cast<int8_t *>(cB.kc) + row * HD, cB.ksc + row * (HD / 32), v, cB.kvemu[0][slot], pp);
                else
#endif
                kv_store(0, slot, g, pp, v);
            } else {
                load8(cB.vbuf + t * KVD + g * 256 + lane * 8, v);
#if MK_KV_EMU
                if (cB.kvemu[1][slot]) kv_store_emu(static_cast<int8_t *>(cB.vc) + row * HD, cB.vsc + row * (HD / 32), v, cB.kvemu[1][slot], pp);
                else
#endif
                kv_store(1, slot, g, pp, v);
            }
        }
        if (tid < RP) {
            sM(tid) = -INFINITY;
            sL(tid) = 0.f;
        }
        __threadfence_block();
        __syncthreads();

        // ---- ring prologue: commit groups {K(0)}, {K(1)}; then per interval jt {K(jt + 2), V(jt)}
        const int ntl = (c1 - c0 + TP - 1) / TP;
        auto tnp = [&](int t) { return t < ntl ? min(TP, c1 - c0 - t * TP) : -1; }; // valid rows of tile t, -1: none
        const int jtr_k = MIX ? kv_ring_tile(kf.kr, wsr, c0, TP) : 0x7fffffff, jtr_v = MIX ? kv_ring_tile(kf.vr, wsr, c0, TP) : 0x7fffffff;
        auto kbt = [&](int t) { return t >= jtr_k ? 8 : kf.kb; }; // format of K tile t (MIX; 8 otherwise)
        auto vbt = [&](int t) { return t >= jtr_v ? 8 : kf.vb; };
        auto issue = [&](int t_, uint8_t * dk, int tk, uint8_t * dv, int tv, int nth_tid) {
            const int8_t *kq, *vq;
            const __half *ks, *vs;
            int kb_ = 8, vb_ = 8;
            if constexpr (MIX) {
                kb_ = kbt(tk);
                vb_ = vbt(max(tv, 0));
                kv_tile_src(0, slot, g, c0 + tk * TP, wsr, wsm, kf.kb, tk >= jtr_k, kq, ks);
                kv_tile_src(1, slot, g, c0 + max(tv, 0) * TP, wsr, wsm, kf.vb, max(tv, 0) >= jtr_v, vq, vs);
            } else { // all-q8 slot: the plain layout
                const size_t kr = (size_t) g * ctx + c0 + tk * TP, vr = (size_t) g * ctx + c0 + max(tv, 0) * TP;
                kq = cB.kvp[0][slot] + kr * HD;
                ks = cB.kvs[0][slot] + kr * (HD / 32);
                vq = cB.kvp[1][slot] + vr * HD;
                vs = cB.kvs[1][slot] + vr * (HD / 32);
            }
            if (nth_tid == NT) aq_issue<NT, MIX>(t_, dk, kq, ks, tnp(tk), dv, vq, vs, tv < 0 ? -1 : tnp(tv), kb_, vb_);
            else aq_issue<NT - 128, MIX>(t_, dk, kq, ks, tnp(tk), dv, vq, vs, tv < 0 ? -1 : tnp(tv), kb_, vb_);
        };
        issue(tid, kring, 0, vring, -1, NT);
        asm volatile("cp.async.commit_group;" ::: "memory");
        issue(tid, kring + AQ_STG, 1, vring, -1, NT);
        asm volatile("cp.async.commit_group;" ::: "memory");
#if MK_W48X
        // MIX: attn_core_q8_t's warp groups, in the same three K and three V stages, so one interval for a copy to land: K(t + 3)
        // goes into K(t)'s stage at the start of interval t (K(t) was converted in t - 1), V(t) into V(t - 3)'s (read in t - 1).
        // Prologue: K(0) .. K(2) landed, K(0) and K(1) -> bytes, K(0) converted.
        if constexpr (MIX) {
            issue(tid, kring + 2 * AQ_STG, 2, vring, -1, NT);
            asm volatile("cp.async.commit_group;" ::: "memory");
            asm volatile("cp.async.wait_group 0;" ::: "memory");
            __syncthreads();
            const int kb0 = kbt(0), kb1 = ntl > 1 ? kbt(1) : 8;
            if (kb0 != 8) kv_tile_expand<false>(kring, kb0, TP, warp, NW, KVQ_ROW);
            if (kb1 != 8) kv_tile_expand<false>(kring + AQ_STG, kb1, TP, warp, NW, KVQ_ROW);
            if (kb0 != 8 || kb1 != 8) __syncthreads();
        } else {
            asm volatile("cp.async.wait_group 1;" ::: "memory");
            __syncthreads();
        }
        const int jff = MIX ? (c1 - c0) / TP : 0;
        const unsigned rowb = (unsigned) g * (unsigned) ctx + (unsigned) c0;
        auto rrow = [&](int t) { // (kv_tile_src's ring row)
            int r0 = wsm + (c0 + t * TP - wsr);
            if (r0 >= cB.kvrc) r0 -= cB.kvrc;
            return (unsigned) g * cB.kvrc + r0;
        };
        auto issue_kx = [&](int tk, uint8_t * dk) { // by the 4 warps of a group (warp & 3)
            if (tk < jff) {
                if (tk < jtr_k) aqx_side<4>(warp & 3, lane, cB.kvp[0][slot], cB.kvs[0][slot], rowb + tk * TP, kf.kb, dk);
                else aqx_side<4>(warp & 3, lane, cB.kvrp[0][slot], cB.kvrs[0][slot], rrow(tk), 8, dk);
            } else {
                const int8_t * kq;
                const __half * ks;
                kv_tile_src(0, slot, g, c0 + tk * TP, wsr, wsm, kf.kb, tk >= jtr_k, kq, ks);
                aq_issue<128, true>((warp & 3) * 32 + lane, dk, kq, ks, tnp(tk), vring, kq, ks, -1, kbt(tk), 8);
            }
        };
        auto issue_vx = [&](int tv, uint8_t * dv) { // likewise
            if (tv < jff) {
                if (tv < jtr_v) aqx_side<4>(warp & 3, lane, cB.kvp[1][slot], cB.kvs[1][slot], rowb + tv * TP, kf.vb, dv);
                else aqx_side<4>(warp & 3, lane, cB.kvrp[1][slot], cB.kvrs[1][slot], rrow(tv), 8, dv);
            } else {
                const int8_t * vq;
                const __half * vs;
                kv_tile_src(1, slot, g, c0 + tv * TP, wsr, wsm, kf.vb, tv >= jtr_v, vq, vs);
                aq_issue<128, true>((warp & 3) * 32 + lane, kring, vq, vs, -1, dv, vq, vs, tnp(tv), 8, vbt(tv));
            }
        };
#else
        asm volatile("cp.async.wait_group 1;" ::: "memory");
        __syncthreads();
#endif

        float acc[MT][4][4]; // P V warps: O rows 16 i + {g8, g8 + 8}, dims 32 wn + 8 tig + jj and + 4
#pragma unroll
        for (int i = 0; i < MT; i++)
#pragma unroll
            for (int j = 0; j < 4; j++)
#pragma unroll
                for (int k = 0; k < 4; k++) acc[i][j][k] = 0.f;

#if MK_W48X
        if constexpr (MIX) { // (a barrier: interval 0's first copy goes into K(0)'s stage)
            if (warp < 4) {
                const int kbj = kbt(0);
                aqx_conv(acc, kring + (8 * warp + g8) * KVQ_ROW, tig, kbj == 8 ? 0x8080u : 0u, aq_sub(kbj));
            }
            __syncthreads();
        }
        for (int jt = 0; jt < ntl + 2; jt++) {
            // MIX: the two warp groups of attn_core_q8_t, rC: this interval's converting group
            const int sg = warp < 4 ? 0 : warp >= 12 ? 1 : -1;
            const bool rC = sg >= 0 && sg != (jt & 1);
            if constexpr (MIX) {
                if (rC) {
                    if (jt + 3 < ntl) issue_kx(jt + 3, kring + ((jt + 3) % NSK) * AQ_STG);
                    if (jt < ntl) issue_vx(jt, vring + (jt % NSV) * AQ_STG);
                    if (jt + 1 < ntl) {
                        const int kbj = kbt(jt + 1);
                        aqx_conv(acc, kring + ((jt + 1) % NSK) * AQ_STG + (8 * (warp & 3) + g8) * KVQ_ROW, tig, kbj == 8 ? 0x8080u : 0u, aq_sub(kbj));
                    }
                    if (jt + 2 < ntl) {
                        const int kbx = kbt(jt + 2);
#if MK_K6X
                        if (kbx != 8 && kbx != 6) kvx_tile<8>(kring + ((jt + 2) % NSK) * AQ_STG, kbx, 8 * (warp & 3)); // (6: the P V warps)
#else
                        if (kbx != 8) kvx_tile<8>(kring + ((jt + 2) % NSK) * AQ_STG, kbx, 8 * (warp & 3));
#endif
                    }
#if MK_W48Z
                    aqx_zero(acc[2]);
#endif
                } else if (sg < 0) { // V -> bytes before the P V MMAs: the S warps get the tensor pipe first, two warps share its tail
                    if (jt >= 1 && jt - 1 < ntl) {
                        const int vbx = vbt(jt - 1);
                        if (vbx != 8) kvx_tile<4>(vring + ((jt - 1) % NSV) * AQ_STG, vbx, 4 * (warp - 4));
                    }
                } else if (jt < ntl) {
                    {
                        float s[MT][4];
#if MK_W48Z
                        aqx_mma<MT, MT, LDH, true>(s, acc, (unsigned) __cvta_generic_to_shared(sQ + (lane & 15) * LDH + (lane >> 4) * 8));
#else
                        aqx_mma<MT, MT, LDH>(s, acc, (unsigned) __cvta_generic_to_shared(sQ + (lane & 15) * LDH + (lane >> 4) * 8));
#endif
#pragma unroll
                        for (int i = 0; i < MT; i++) {
                            float * so = sS + (jt & 1) * RP * LDS + (16 * i + g8) * LDS + 8 * (warp & 3) + 2 * tig;
                            so[0] = s[i][0];
                            so[1] = s[i][1];
                            so[8 * LDS] = s[i][2];
                            so[8 * LDS + 1] = s[i][3];
                        }
                    }
#if MK_W48Z
                    aqx_zero(acc[2]);
#endif
                }
            }
            if (!MIX && warp < 4) {
#else
        for (int jt = 0; jt < ntl + 2; jt++) {
            if (warp < 4) {
#endif
                // ---- S(jt) = Q K(jt)^T: B = positions 8 wn + g8 (all 16 k-steps converted first), all row tiles
                if (jt < ntl) {
                    const int wn = warp;
                    uint8_t * kst = kring + (jt % NSK) * AQ_STG;
                    const int kbj = kbt(jt);
                    if constexpr (MIX) { // this warp's rows of a low-bit tile -> biased bytes (row-local: no other warp reads them)
                        if (kbj != 8) {
                            kv_tile_expand<false>(kst, kbj, 8 * wn + 8, 8 * wn, 1, KVQ_ROW);
                            __syncwarp();
                        }
                    }
                    if constexpr (I8) { // int8 Q.K^T (aq_i8_s: the S of every I8 kernel)
                        float s[MT][4];
                        aq_i8_s<MT, RP>(s, sQ8, sQs, kst, wn);
#pragma unroll
                        for (int i = 0; i < MT; i++) {
                            float * so = sS + (jt & 1) * RP * LDS + (16 * i + g8) * LDS + 8 * wn + 2 * tig;
                            so[0] = s[i][0];
                            so[1] = s[i][1];
                            so[8 * LDS] = s[i][2];
                            so[8 * LDS + 1] = s[i][3];
                        }
                    } else {
                    const uint8_t * krow = kst + (8 * wn + g8) * KVQ_ROW;
                    const uint4 scv = *reinterpret_cast<const uint4 *>(krow + 256);
                    const unsigned scw[4] = {scv.x, scv.y, scv.z, scv.w};
                    const unsigned kxm = kbj == 8 ? 0x8080u : 0u, ksub = aq_sub(kbj);
                    unsigned kb[HD / 16][2];
#pragma unroll
                    for (int ks = 0; ks < HD / 16; ks++) {
                        const unsigned sw = __byte_perm(scw[ks >> 2], 0, (ks & 2) ? 0x3232 : 0x1010);
                        const __half2 dd = *reinterpret_cast<const __half2 *>(&sw);
                        kb[ks][0] = aq_q8x2(*reinterpret_cast<const unsigned short *>(krow + 16 * ks + 2 * tig), dd, kxm, ksub);
                        kb[ks][1] = aq_q8x2(*reinterpret_cast<const unsigned short *>(krow + 16 * ks + 2 * tig + 8), dd, kxm, ksub);
                    }
                    float s[MT][4];
#pragma unroll
                    for (int i = 0; i < MT; i++)
#pragma unroll
                        for (int k = 0; k < 4; k++) s[i][k] = 0.f;
                    const unsigned qaddr = (unsigned) __cvta_generic_to_shared(sQ + (lane & 15) * LDH + (lane >> 4) * 8);
#pragma unroll
                    for (int ks = 0; ks < HD / 16; ks++) {
#pragma unroll
                        for (int i = 0; i < MT; i++) {
                            unsigned a[4];
                            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                                         : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3])
                                         : "r"(qaddr + 2 * (16 * i * LDH + 16 * ks)));
                            mma16816(s[i], a[0], a[1], a[2], a[3], kb[ks][0], kb[ks][1]);
                        }
                    }
#pragma unroll
                    for (int i = 0; i < MT; i++) {
                        float * so = sS + (jt & 1) * RP * LDS + (16 * i + g8) * LDS + 8 * wn + 2 * tig;
                        so[0] = s[i][0];
                        so[1] = s[i][1];
                        so[8 * LDS] = s[i][2];
                        so[8 * LDS + 1] = s[i][3];
                    }
                    }
                }
#if MK_W48X
            } else if (warp >= 4 && warp < 12) {
#else
            } else if (warp < 12) {
#endif
                // ---- O = O * alpha + P(jt - 2) V(jt - 2): B = dims 32 wn + 4 g8 + jj of n8 tile jj, all row tiles
                if (jt >= 2) {
                    const int wn = warp - 4, jp = jt - 2, b = jp & 1;
                    uint8_t * vstg = vring + (jp % NSV) * AQ_STG;
                    const int vbj = vbt(jp);
#if !MK_W48X
                    if constexpr (MIX) { // this warp's dims of a low-bit tile -> biased bytes (block-uniform branch: named barrier)
                        switch (vbj) {
                        case 4: kv_slice_expand<4>(vstg, wn); break;
                        case 5: kv_slice_expand<5>(vstg, wn); break;
                        case 6: kv_slice_expand<6>(vstg, wn); break;
                        case 7: kv_slice_expand<7>(vstg, wn); break;
                        default: break;
                        }
                        __syncwarp();
                    }
#endif
#pragma unroll
                    for (int i = 0; i < MT; i++) {
                        const float al0 = sAl(b, 16 * i + g8), al1 = sAl(b, 16 * i + g8 + 8);
#pragma unroll
                        for (int j = 0; j < 4; j++) {
                            acc[i][j][0] *= al0;
                            acc[i][j][1] *= al0;
                            acc[i][j][2] *= al1;
                            acc[i][j][3] *= al1;
                        }
                    }
                    const unsigned paddr = (unsigned) __cvta_generic_to_shared(sP + b * RP * LDP + (lane & 15) * LDP + (lane >> 4) * 8);
                    const uint8_t * vst = vstg + 32 * wn + 4 * g8;
                    const uint8_t * vsc = vstg + 256 + 2 * wn;
                    const unsigned vxm = vbj == 8 ? 0x80808080u : 0u, vsub = aq_sub(vbj);
#pragma unroll
                    for (int ks = 0; ks < TP / 16; ks++) {
                        unsigned bf[2][4]; // [b0 / b1][tile jj]
#pragma unroll
                        for (int h = 0; h < 2; h++) {
                            const int p = 16 * ks + 2 * tig + 8 * h; // k rows p, p + 1 of the fragment
                            const unsigned xa = *reinterpret_cast<const unsigned *>(vst + p * KVQ_ROW) ^ vxm;
                            const unsigned xb = *reinterpret_cast<const unsigned *>(vst + (p + 1) * KVQ_ROW) ^ vxm;
                            const unsigned sw = (unsigned) *reinterpret_cast<const unsigned short *>(vsc + p * KVQ_ROW) |
                                                ((unsigned) *reinterpret_cast<const unsigned short *>(vsc + (p + 1) * KVQ_ROW) << 16);
                            const __half2 dd = *reinterpret_cast<const __half2 *>(&sw);
                            const unsigned y0 = __byte_perm(xa, xb, 0x5140), y1 = __byte_perm(xa, xb, 0x7362);
                            bf[h][0] = aq_q8x2p(y0, 0x4140, dd, vsub);
                            bf[h][1] = aq_q8x2p(y0, 0x4342, dd, vsub);
                            bf[h][2] = aq_q8x2p(y1, 0x4140, dd, vsub);
                            bf[h][3] = aq_q8x2p(y1, 0x4342, dd, vsub);
                        }
#pragma unroll
                        for (int i = 0; i < MT; i++) {
                            unsigned a[4];
                            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                                         : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3])
                                         : "r"(paddr + 2 * (16 * i * LDP + 16 * ks)));
#pragma unroll
                            for (int j = 0; j < 4; j++) mma16816(acc[i][j], a[0], a[1], a[2], a[3], bf[0][j], bf[1][j]);
                        }
                    }
                }
#if MK_K6X && MK_W48X
                if constexpr (MIX) { // MK_K6X: a 6-bit K(jt + 2) -> bytes here, not on the converting group (after the MMAs, as T <= 5)
                    if (jt + 2 < ntl && kbt(jt + 2) == 6) kvx_rows<6, 4>(kring + ((jt + 2) % NSK) * AQ_STG, 4 * (warp - 4));
                }
#endif
#if MK_W48X
            } else if (!MIX || rC) {
                const int sn_ = MIX ? warp & 3 : warp - 12;
#else
            } else {
                const int sn_ = warp - 12;
#endif
                // ---- online softmax of tile jt - 1 (causal mask: row token t sees positions <= n + t); lane l4 of the
                // row's quad holds positions l4 + 4 k. Rows 8 (warp - 12) + q, then 32 + 8 (warp - 12) + q.
                if (jt >= 1 && jt <= ntl) {
                    const int jp = jt - 1, b = jp & 1;
                    const int tb = c0 + jp * TP, np_ = min(TP, c1 - tb);
#pragma unroll
                    for (int sub = 0; sub < 2; sub++) {
                        if (32 * sub + 8 * sn_ >= R) continue; // (warp-uniform)
                        const int r = 32 * sub + 8 * sn_ + (lane >> 2), l4 = lane & 3; // rows >= R (zero S) are not stored
                        const int lim = W48_N + (r % T);
                        const float * srow = sS + b * RP * LDS + r * LDS;
                        float sv[8];
#pragma unroll
                        for (int k = 0; k < 8; k++) {
                            const int q = l4 + 4 * k;
                            sv[k] = srow[q];
                            if (q >= np_ || tb + q > lim) sv[k] = -INFINITY;
                        }
                        float m = sv[0];
#pragma unroll
                        for (int k = 1; k < 8; k++) m = fmaxf(m, sv[k]);
                        m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 2));
                        m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 1));
                        const float mold = sM(r);
                        const float mnew = fmaxf(mold, m);
                        float p[8], al = 1.f;
#pragma unroll
                        for (int k = 0; k < 8; k++) p[k] = 0.f;
                        if (mnew != -INFINITY) {
#pragma unroll
                            for (int k = 0; k < 8; k++) p[k] = expf(sv[k] - mnew);
                            al = expf(mold - mnew);
                        }
                        // attn_core_t's wsum tree: position bit 4 (k, k + 4), bit 3 (k, k + 2), bit 2 (k, k + 1), then lanes
                        const float t0 = p[0] + p[4], t1 = p[1] + p[5], t2 = p[2] + p[6], t3 = p[3] + p[7];
                        const float u0 = t0 + t2, u1 = t1 + t3;
                        float ps = u0 + u1;
                        ps += __shfl_xor_sync(0xffffffffu, ps, 2);
                        ps += __shfl_xor_sync(0xffffffffu, ps, 1);
#if MK_FORMAL_SYNC
                        __syncwarp(); // the quad's sM(r) reads before lane l4 == 0 rewrites it (shfl.sync orders no memory)
#endif
                        if (r < R) {
                            __half * prow = sP + b * RP * LDP + r * LDP;
#pragma unroll
                            for (int k = 0; k < 8; k++) prow[l4 + 4 * k] = __float2half_rn(p[k]);
                            if (l4 == 0) {
                                sL(r) = sL(r) * al + ps;
                                sM(r) = mnew;
                                sAl(b, r) = al;
                            }
                        }
                    }
                }
#if MK_W48Z && MK_W48X
                if constexpr (MIX) aqx_zero(acc[2]);
#endif
            }
            // ---- copies: K(jt + 2) into the stage of K(jt - 1), V(jt) into the stage of V(jt - 3) (both read in interval
            // jt - 1), issued by warps 4-15; then wait for K(jt + 1) and V(jt - 1)
#if MK_W48X
            if constexpr (MIX) { // attn_core_q8_t's
                asm volatile("cp.async.commit_group;" ::: "memory");
                if (sg >= 0) asm volatile("cp.async.wait_group 0;" ::: "memory");
                __syncthreads();
                continue;
            }
#endif
            if (warp >= 4 && jt < ntl) issue(tid - 128, kring + ((jt + 2) % NSK) * AQ_STG, jt + 2, vring + (jt % NSV) * AQ_STG, jt, NT - 128);
            asm volatile("cp.async.commit_group;" ::: "memory");
            asm volatile("cp.async.wait_group 1;" ::: "memory");
            __syncthreads();
        }
        { // item epilogue: thread indices read afresh (not carried across the interval loop)
        const int tid = tid_x(), lane = tid & 31, warp = tid >> 5, g8 = lane >> 2, tig = lane & 3;
        (void) lane;
        if (warp >= 4 && warp < 12) {
            const int wn = warp - 4;
            // tile j's columns 2 tig, 2 tig + 1 are dims d0 + j, d0 + j + 4: one 32-byte sector per lane and row, two
            // 16-byte stores (attn_core_q8_t's epilogue)
#pragma unroll
            for (int i = 0; i < MT; i++) {
                const int r0 = 16 * i + g8, d0 = 32 * wn + 8 * tig;
                if (r0 < R) {
                    float4 * o = reinterpret_cast<float4 *>(cB.apart + ((size_t) item * AR + r0) * 256 + d0);
                    o[0] = make_float4(acc[i][0][0], acc[i][1][0], acc[i][2][0], acc[i][3][0]);
                    o[1] = make_float4(acc[i][0][1], acc[i][1][1], acc[i][2][1], acc[i][3][1]);
                }
                if (r0 + 8 < R) {
                    float4 * o = reinterpret_cast<float4 *>(cB.apart + ((size_t) item * AR + r0 + 8) * 256 + d0);
                    o[0] = make_float4(acc[i][0][2], acc[i][1][2], acc[i][2][2], acc[i][3][2]);
                    o[1] = make_float4(acc[i][0][3], acc[i][1][3], acc[i][2][3], acc[i][3][3]);
                }
            }
        }
        if (tid < R) cB.aml[item * AR + tid] = make_float2(sM(tid), sL(tid));
        }
        __syncthreads();
    }
#undef W48_N
#undef W48_WS
#undef W48_CS
#undef W48_LC
#undef W48_NCH
#undef W48_NW
#undef W48_RANK
}
template <int T>
__device__ __noinline__ void attn_core_w48q(const LayerW & W, int slot, int n, int nw = -1, int b0 = 0) {
    attn_core_w48<T, false>(W, slot, n, nw, b0);
}
template <int T>
__device__ __noinline__ void attn_core_w48m(const LayerW & W, int slot, int n, int b0 = 0) {
    attn_core_w48<T, true>(W, slot, n, -1, b0);
}
#if MK_ATT_I8
template <int T>
__device__ __noinline__ void attn_core_w48i(const LayerW & W, int slot, int n, int nw = -1, int b0 = 0) {
    attn_core_w48<T, false, true>(W, slot, n, nw, b0);
}
#endif

// T <= 5: a block without items in this call (attn_core_t / attn_core_q8_t deal NKV nch items from rank 0) does
// resid_o's L2 prefetch (attn_idle_pf, as attn_core_w48's) and returns true (the MTP slot's out is not in the wv layout)
template <int T>
__device__ __forceinline__ bool attn_idle(const LayerW & W, int slot, int n, int b0) {
    int nch, cs;
    attn_dims_w(n + T, mtp_ws(slot, n), nch, cs);
    const int rk = att_rank(b0), ni = NKV * nch;
    if (rk < ni) return false;
    if (slot < N_ATT_SLOTS - 1 && n + T <= MK_ATT_PF_CTX) attn_idle_pf(W.o, rk - ni, (int) gridDim.x - ni);
    return true;
}

// q8 cache: the pipelined kernel when a chunk has more than one tile (L_ctx > ATT_BUCKET), attn_core_t for
// one-tile chunks (the pipeline fill does not pay there) and with MK_DATTN=0. Every choice gives the same partials.
// MK_ATT_I8 (cB.att_i8, q8 cache): all-q8 slots take the same choice in the int8 Q.K^T instances (attn_core_q8i,
// attn_core_ti, attn_core_w48i; aq_i8_s: again the same partials in every one); other slots keep fp16 Q.
template <int T>
__device__ __noinline__ void attn_core(const LayerW & W, int slot, int n, uint8_t * /*sm*/, int b0 = 0) { // b0: att_rank
    if constexpr (T > 5) { // q8 cache (checked by the host): the pipelined kernel at every context, one pass over all
                           // 6 T rows (attn_core_w48); slots with a low-bit MK_KV side run the MIX form (the same partials
                           // per row)
        if (cB.kvb[0][slot] != 8 || cB.kvb[1][slot] != 8) attn_core_w48m<T>(W, slot, n, b0);
#if MK_ATT_I8
        else if (cB.att_i8) attn_core_w48i<T>(W, slot, n, -1, b0);
#endif
        else
#if MK_W48X_Q8 // (all-q8 slots through the MIX form: the same partials)
        attn_core_w48m<T>(W, slot, n, b0);
#else
        attn_core_w48q<T>(W, slot, n, -1, b0);
#endif
        return;
    } else if (attn_idle<T>(W, slot, n, b0)) {
        return;
    } else
    if (cB.kvq8) {
        // one call site per kernel: attn_core_t is force-inlined, a second site would duplicate it
        const bool q8 = cB.kvb[0][slot] == 8 && cB.kvb[1][slot] == 8;
#if MK_ATT_I8
        if (q8 && cB.att_i8) { // (the I8 instances are out of line: no second inlined attn_core_t here)
            if (cB.dattn && n + T > ATT_BUCKET) attn_core_q8i<T>(W, slot, n, b0);
            else attn_core_ti<T>(W, slot, n, b0);
            return;
        }
#endif
        if (cB.dattn && n + T > ATT_BUCKET) {
            if (!q8) attn_core_q8m<T>(W, slot, n, b0);
            else
#if MK_W48X_Q8
            attn_core_q8m<T>(W, slot, n, b0);
#else
            attn_core_q8<T>(W, slot, n, b0);
#endif
        } else {
            attn_core_t<T, true>(W, slot, n, b0);
        }
    } else {
        attn_core_t<T, false>(W, slot, n, b0);
    }
}

// combine split-K partials, apply sigmoid output gate, quantise for o_proj
template <int T>
__device__ __noinline__ void attn_combine(int n, int ws, int slot, int b0 = 0) { // ws: the slot's window start (attn_core's partition); b0: att_rank
    const int L_ctx = n + T;
    int nch, CS;
    attn_dims_w(L_ctx, ws, nch, CS);
    // Four items per block and round: group q = tid / 128 (warps 4 q .. 4 q + 3) runs items 4 rank + q, + 4 gridDim.x, ..,
    // its warp wq = warp % 4 the dims d0 = 32 wq + lane and d1 = d0 + 128 (units 8 h + wq, 8 h + wq + 4). NH T = 192 items
    // at T 8 fit one round of 82 blocks (two per block took a second round on 14 blocks: 40k T8 attn_comb 9.2 us vs 6.1
    // on the one-round blocks). Each dim's sums, the rotation, the gate and each unit's q32_store are the old ones.
    const int tid = threadIdx.x, lane = tid & 31, wq = (tid >> 5) & 3, grp = tid >> 7, d0 = 32 * wq + lane, d1 = d0 + 128;
    // V stored rotated (MK_KV "r" on V): the head's output is rotated back (WHT, its own inverse) before the gate; the
    // group's 4 warps meet at their own named barrier 1 + grp (the groups' item counts differ)
    // nch <= ATT_NCH (attn_dims), + 1 with the MTP window's sink chunk: one chunk's (m, l) per lane
    constexpr int NC = ATT_NCH + 1;
    static_assert(NC <= 32, "attn_combine: one chunk's (m, l) per lane");
    static_assert(NT == 512, "attn_combine: four groups of 128 threads");
    const bool rotv = cB.kvq8 && cB.kvrv[slot];
    float * so = reinterpret_cast<float *>(smem) + grp * 256; // scratch (the attention phase's shared memory is free)
    float * const sp = reinterpret_cast<float *>(smem) + 4 * 256 + grp * NC * 256; // the group's item partials [chunk][256]
    // (inside what every attention core overwrites, SM_ATTN / SM_ATTN_W48 / SM_ATTN_Q8: nothing lives there across)
    static_assert((4 * 256 + 4 * NC * 256) * 4 <= SM_ATTN, "attn_combine partials in shared memory");
    for (int it = att_rank(b0) * 4 + grp; it < NH * T; it += 4 * gridDim.x) {
        const int h = it / T, t = it % T, g = h / 6, r = (h % 6) * T + t;
        const float * const pg = cB.qbuf + t * QG + h * 512 + 256; // (the gate: independent of the partials, issued first)
        const float gate0 = __ldcg(pg + d0), gate1 = __ldcg(pg + d1);
        // all of an item's loads in one round trip: lane c holds chunk c's (m, l), and the nch partials go to shared memory
        // by cp.async, not to registers: this out-of-line function gets only the registers its call sites leave free, which
        // moves between builds (earlier versions' (m, l) batches spilled at R77, their 21 register partials at R33).
        // Warp wq copies and reads only its dims d0, d1 of every chunk (16 x 16 B per chunk): a __syncwarp orders its
        // copies and reads.
        // Same floats: M is the max of the same values (exact, order-free; its zero sign cannot change expf(x - M)), lane c
        // evaluates chunk c's old weight expression, and the sums run over c in the old order with the old expressions.
        const float2 * const pml = cB.aml + (size_t) g * nch * AR + r;                    // chunk c: pml[c AR]
        const float * const ppa = cB.apart + ((size_t) g * nch * AR + r) * 256 + 32 * wq; // chunk c: ppa[c AR 256]
        const float2 ml = lane < nch ? __ldcg(pml + (size_t) lane * AR) : make_float2(-INFINITY, 0.f);
        __syncwarp(); // (this warp's reads of the previous item's partials are done)
        for (int j = lane; j < 16 * nch; j += 32) { // chunk j / 16, dims 32 wq + 128 (j / 8 % 2) + 4 (j % 8) .. + 3
            const int e = 128 * ((j >> 3) & 1) + 4 * (j & 7);
            cp_async16(sp + (j >> 4) * 256 + 32 * wq + e, ppa + (size_t) (j >> 4) * AR * 256 + e);
        }
        asm volatile("cp.async.commit_group;" ::: "memory");
        const float M = wmax(ml.x);
        const float wl = ml.x == -INFINITY ? 0.f : expf(ml.x - M);
        asm volatile("cp.async.wait_group 0;" ::: "memory");
        __syncwarp();
        float num0 = 0.f, num1 = 0.f, den = 0.f;
        for (int c = 0; c < nch; c++) {
            const float w = __shfl_sync(0xffffffffu, wl, c);
            num0 += w * sp[c * 256 + d0];
            num1 += w * sp[c * 256 + d1];
            den += w * __shfl_sync(0xffffffffu, ml.y, c);
        }
        float y0 = num0 / den, y1 = num1 / den;
        if (rotv) {
            so[d0] = y0;
            so[d1] = y1;
            asm volatile("bar.sync %0, 128;" ::"r"(1 + grp) : "memory");
            if (wq == 0) {
                float v[8];
#pragma unroll
                for (int k = 0; k < 8; k++) v[k] = so[8 * lane + k];
                fwht256(v);
#pragma unroll
                for (int k = 0; k < 8; k++) so[8 * lane + k] = v[k];
            }
            asm volatile("bar.sync %0, 128;" ::"r"(1 + grp) : "memory");
            y0 = so[d0];
            y1 = so[d1];
        }
        const float o0 = y0 / (1.0f + expf(-gate0)), o1 = y1 / (1.0f + expf(-gate1));
        q32_store(cB.qo, t, 8 * h + wq, o0, 0.f, false);
        q32_store(cB.qo, t, 8 * h + wq + 4, o1, 0.f, false);
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
// write_h_rows (ring) over the whole grid: every block computes the T row scales (block 0's sums, the same order) and
// writes a slice of the T x D values (block 0 alone delayed its lm_head<16> rows by ~90 us)
template <int T>
__device__ __noinline__ void write_h_rows_g(const float * x, const QBuf & q, const float * w, float * dst_base, int pos0) {
    static_assert(T <= NW, "one warp per row");
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    float * s_rs = misc(smem);
    if (warp < T) {
        float acc = 0.f;
        for (int j = lane; j < U_D; j += 32) acc += __ldcg(q.ssq + warp * U_D + j);
        acc = wsum(acc);
        if (lane == 0) s_rs[warp] = rsqrtf(acc / (float) D + EPS);
    }
    __syncthreads();
    for (int i = blockIdx.x * NT + threadIdx.x; i < T * D; i += gridDim.x * NT) {
        const int t = i / D, e = i - t * D;
        dst_base[(size_t) ((pos0 + t) % H_RING) * D + e] = __ldcg(x + i) * s_rs[t] * w[e];
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

// Merging sorted top-32 lists (the sampled lm_head's per-warp and per-block candidate lists).  Inserting lists one
// after the other with topk_insert keeps, of all entries, the best 32 under the order (value descending, then
// insertion order): an entry goes behind equal values already there and does not displace an equal 32nd.  So the
// insertion order can be replaced by an explicit sequence key s (list index * 32 + rank in the list) and the lists
// merged in any order with a comparison on (value, s): the result is identical, entry for entry, to the sequential
// merge.  Unfilled slots are (-inf, -1) and rank below every real entry (whose logits are > -inf: topk_insert never
// takes -inf or NaN); they are interchangeable, so their keys do not matter.
__device__ __forceinline__ bool tk_before(float av, int as, float bv, int bs) { return av > bv || (av == bv && as < bs); }

#if MK_PROFILE
// -DMK_PROFILE=1 builds (with env MK_PROFILE): sub-stamps i = 0..3 of the last block's serial section before barrier
// s_bar_k + 1 (lm_head / mtp_lm_head: 0 ticket passed, 1 merge done, 2 sampling done, 3 control done), kept in the barrier's
// arrival slots 120 + i (unused: blocks < 120); MK_PROF_DUMP prints them as "# sub" lines, the profile their means
__device__ __forceinline__ void sub_stamp(int i) {
    const unsigned k = s_bar_k + 1;
    if (cB.prof && threadIdx.x == 0 && k < 1024 && gridDim.x <= 120) cB.prof[4096 + k * 128 + 120 + i] = gtimer();
}
#endif

// (v, id, s): this warp's sorted list (lane = rank); merged with the sorted list (bv, bid, bs) into the best 32 of
// both, sorted: reversing b and keeping the better of each pair gives a bitonic sequence holding the best 32, which
// a 5-step bitonic merge sorts.
__device__ __forceinline__ void tk_merge(float & v, int & id, int & s, float bv, int bid, int bs) {
    const int lane = threadIdx.x & 31;
    {
        const float rv = __shfl_sync(0xffffffffu, bv, 31 - lane);
        const int ri = __shfl_sync(0xffffffffu, bid, 31 - lane);
        const int rs = __shfl_sync(0xffffffffu, bs, 31 - lane);
        if (tk_before(rv, rs, v, s)) { v = rv; id = ri; s = rs; }
    }
#pragma unroll
    for (int k = 16; k >= 1; k >>= 1) {
        const float ov = __shfl_xor_sync(0xffffffffu, v, k);
        const int oi = __shfl_xor_sync(0xffffffffu, id, k);
        const int os = __shfl_xor_sync(0xffffffffu, s, k);
        if (((lane & k) == 0) == tk_before(ov, os, v, s)) { v = ov; id = oi; s = os; } // low lane keeps the better
    }
}

// shared-memory lists [TL][16][32] (values tv, ids ti, sequence keys ts) -> list 0 of each row holds the best 32 of
// its 16 lists (pairwise tree, the merges of a level spread over all warps). Ends with a barrier.
template <int TL>
__device__ __noinline__ void tk_tree16(float * tv, int * ti, int * ts) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
#pragma unroll 1
    for (int d = 1; d < 16; d *= 2) {
        const int per = 8 / d; // merges per row at this level
#pragma unroll 1
        for (int q = warp; q < TL * per; q += NW) {
            const int t = q / per, l = (q % per) * 2 * d;
            const int a = (t * 16 + l) * 32 + lane, b = a + d * 32;
            float v = tv[a];
            int id = ti[a], s = ts[a];
            tk_merge(v, id, s, tv[b], ti[b], ts[b]);
            tv[a] = v;
            ti[a] = id;
            ts[a] = s;
        }
        __syncthreads();
    }
}
// warp w of the last block: block lists w, w + NW, ... of row t (cB.blk_best, BS rows per block) merged into
// shared-memory list (slot, w) (6 loads in flight per batch; out of line: own registers)
template <int BS>
__device__ __noinline__ void tk_grid_chain(int t, int slot, float * tv, int * ti, int * ts) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, G = gridDim.x;
    float v = -INFINITY;
    int id = -1, s = 0;
#pragma unroll 1
    for (int k0 = warp; k0 < G; k0 += 6 * NW) {
        float2 e[6];
#pragma unroll
        for (int i = 0; i < 6; i++) {
            const int k = k0 + i * NW;
            e[i] = k < G ? __ldcg(&cB.blk_best[((size_t) k * BS + t) * TOPK_MAX + lane]) : make_float2(-INFINITY, __int_as_float(-1));
        }
#pragma unroll
        for (int i = 0; i < 6; i++) {
            const int k = k0 + i * NW;
            if (k < G) tk_merge(v, id, s, e[i].x, __float_as_int(e[i].y), k * 32 + lane);
        }
    }
    const int a = (slot * 16 + warp) * 32 + lane;
    tv[a] = v;
    ti[a] = id;
    ts[a] = s;
}
// sampled lm_head: the final candidate rows [MAXT][32] (value, id) in shared memory, above the top-k merge lists,
constexpr int SM_CAND = MAXT * 16 * 32 * 12 > 32768 ? MAXT * 16 * 32 * 12 : 32768;
constexpr int SM_DQ = SM_CAND + MAXT * 32 * 8; // and the drafts' proposals q' staged for sample_targets
constexpr int SM_LSE = SM_DQ + MAXT * 32 * 8;   // the draft lm_head's per-block (max, sum of exp), [gridDim.x]
static_assert(MAXT * 16 * 32 * 12 <= SM_CAND && SM_LSE + MK_MAX_GRID * 8 <= SM_MISC, "top-k merge shared memory");

// lm_head over TL rows (qn rows t0..t0+TL-1) with greedy argmax; returns true in the single
// block that finished last, which holds the reduced argmax in s_tok.  TOPK: additionally keeps the
// TOPK_MAX best logits of each row in cand (default cB.cand; for sampling).  FULL: additionally stores
// every logit of each row in cB.logits (scoring).  LSE (TL == 1): also the log-sum-exp of the row's
// true logits; the last block leaves misc[LM_LSE] = lse and misc[LM_BEST] = the best true logit.
// Draft rows: s_tok and the TOPK ids are token ids (mapped through cW.draft_ids).
// pen: presence / frequency penalty (Launch::pres / freq) on the true logits, the occurrences of a token = its count in
// cB.pcnt plus its occurrences in ctl->tok[1 .. nx0 + t] (the drafts in row t's context; nx0: the drafts before row 0).
// Every value is then the true logit minus the penalty (correctly rounded, no contraction: the same in any T).
// misc() slots ([LM_WLSE, +2 * NW): per-warp lse), above the argmax lists [16, 16 + 2 NW MAXT) and s_last (LM_LAST)
constexpr int LM_WLSE = 16 + 2 * NW * MAXT, LM_LSE = LM_WLSE + 32, LM_BEST = LM_LSE + 1;
constexpr int LM_LAST = LM_BEST + 1;
constexpr int LM_XT = 448; // lm_head<16>: the pen drafts tok[1 .. MAXD - 1] (misc slots above s_tok's 400 .. 415)
static_assert(LM_XT >= 416 && LM_XT + MAXD <= 2048 / 4, "lm_head<16> pen drafts in misc()");
static_assert(LM_LAST < 400 && (MAXT == 5 || LM_WLSE >= 16 + 2 * NW * MAXT), "lm_head misc slots (s_tok at 400)");
__device__ __forceinline__ float pen_logit(float a, float sc, int c, float pres, float freq) {
    return __fsub_rn(__fsub_rn(__fmul_rn(a, sc), __fmul_rn(freq, (float) c)), c > 0 ? pres : 0.f);
}
// lm_head<16>'s (greedy) warp sums of rows [0, H) as a reduce-scatter: wsum's butterfly (offsets 16, 8, 4, 2, 1), but the
// step at offset H keeps only the half of the rows the lane's H bit picks (packed to the front), so lanes 2j and 2j + 1
// end with row j's sums in a[k][0]. Each kept value is wsum's (own + the partner's value of the same row: the partner
// kept that row in every earlier step). Selects on lane bits, fixed indices: a stays in registers (lane-indexed reads of
// a were rewritten by cicc into a dynamic index, putting a in local memory)
template <int H, int R>
__device__ __forceinline__ void wsum_rows16(float (&a)[R][16]) {
    if constexpr (H == 1) {
#pragma unroll
        for (int k = 0; k < R; k++) a[k][0] += __shfl_xor_sync(0xffffffffu, a[k][0], 1);
    } else {
        constexpr int h = H / 2;
        const bool hi = threadIdx.x & H; // keeps rows [h, H) of the current H
#pragma unroll
        for (int k = 0; k < R; k++)
#pragma unroll
            for (int i = 0; i < h; i++) {
                const float lo_v = a[k][i], hi_v = a[k][i + h];
                const float keep = hi ? hi_v : lo_v, send = hi ? lo_v : hi_v;
                a[k][i] = keep + __shfl_xor_sync(0xffffffffu, send, H);
            }
        wsum_rows16<h, R>(a);
    }
}
// output.weight is Q6_K, or Q8_0 in some fine-tunes: one function per variant reads either, the type switch around the
// row reader only (the two readers are exclusive branches: the loop needs the registers of one, not of both).
template <int TL, bool TOPK = false, bool FULL = false, bool LSE = false>
__device__ __noinline__ bool lm_head(int t0, bool draft, uint8_t * /*sm*/, int * s_tok, float2 * cand = nullptr,
                                     bool pen = false, float pres = 0.f, float freq = 0.f, int nx0 = 0) {
    static_assert(!LSE || TL == 1, "LSE: one row");
    // H16 (TL = 16 > MAXT: the deep and the 16-row batched passes): the activations in the SXH layout, the block / grid
    // merges in two halves of TLL = 8 rows, cB.blk_best at BS = LM_RB rows per block, candidates only in cand (these
    // callers read cB.cand and cB.dq: no SM_CAND / SM_DQ copies)
    constexpr bool H16 = TL == LM_RB && TL > MAXT;
    constexpr int BS = H16 ? LM_RB : MAXT, TLL = H16 ? 8 : TL;
    static_assert(!H16 || (!FULL && !LSE), "lm_head<16>: argmax / top-k only");
    if constexpr (FULL) pen = false; // no penalty when scoring (the FULL instances keep no penalty code)
    uint8_t * const sm = smem;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const SX sx = make_sx(sm, U_D);
    float * m = misc(sm);
    // the draft head reads lm_head rows through cW.draft_ids (row r of head_d is lm_head row draft_ids[r])
    const int * ids = draft ? cW.draft_ids : nullptr;
    const SXH sxh = make_sxh();
    if constexpr (H16) fill_xs16(cB.qn, t0, sxh, m);
    else fill_xs<TL>(cB.qn, t0, true, sx, m);
    float bv[TL];
    int bi[TL];
    if constexpr (!(H16 && TOPK)) { // (H16 TOPK: set up before its second pass; the lists are not live across the stream)
#pragma unroll
        for (int t = 0; t < TL; t++) {
            bv[t] = -INFINITY;
            bi[t] = TOPK ? -1 : 0;
        }
    }
    // H16 TOPK: the stream writes each row pair's 16 true logits (lane 2j + p: row j, vocab row r + p) as one 128 B line of
    // cB.lmscr, line k * nwarps + gwarp for the warp's k-th pair; the top-32 lists are built from those lines afterwards
    // (kept in registers across the stream, the 16 lists spilled: ~300 MB of local loads per launch)
    float * scr = nullptr;
    if constexpr (H16 && TOPK) scr = cB.lmscr + (size_t) gwarp() * 32 + lane;
    float lm_m = -INFINITY, lm_s = 0.f; // LSE: running max / sum of exp (warp-uniform)
    const Mat head = draft ? cW.head_d : cW.head;
    const bool q8 = cW.head.type == T_Q8_0; // (head_d: a row subset of head, the same type)
    // MK_DRAFT_Q4 (draft head only: its calls are the LSE ones): rows [0, nq4) from the Q4_K copy cW.dq4
    int nq4 = 0;
    if constexpr (LSE) nq4 = draft ? cW.dq4_n : 0;
    // pen: the drafts tok[1..MAXD - 1] (only the first nx0 + t count for row t); H16 and the draft head (LSE): in shared
    // memory (volatile: not hoisted into 15 registers across the row loop). In registers, the conditionally assigned array
    // went to local memory in the draft head: every call (pen or not) loaded its 15 old words after the fill, the pen
    // loads waited on them (Nsight Compute, 8-row code prompt: ~1.3 us per draft step before the row loop, 2.5 MB of
    // local traffic)
    constexpr bool XS = H16 || LSE;
#ifndef MK_LMH8
#define MK_LMH8 1 // (default: mk_shared.cuh)
#endif
    // MK_LMH8 (TL = MAXT): the drafts in shared memory too, loaded into registers inside the pen block of each row pair: the
    // MAXD - 1 registers are no longer live across the row loop (lm_head<8> spills 5 address offsets, reloaded per pair)
    constexpr bool XR = MK_LMH8 && !XS && TL == MAXT;
    int xt[XS || XR ? 1 : MAXD - 1];
    volatile int * const xts = reinterpret_cast<volatile int *>(m + LM_XT);
    if (pen) {
        if constexpr (XS || XR) {
            if (tid < MAXD - 1) xts[tid] = ((volatile Ctl *) cB.ctl)->tok[1 + tid];
            __syncthreads();
        } else {
#pragma unroll
            for (int i = 0; i < MAXD - 1; i++) xt[i] = ((volatile Ctl *) cB.ctl)->tok[1 + i];
        }
    }
    for (int r = 2 * gwarp(); r < head.N; r += 2 * nwarps()) {
        float a[2][TL];
        zero_acc<TL, 2>(a);
        int p0, p1; // token ids of the rows
        {
            const uint8_t * rp[2];
            const __half * dp[2]; // the rows' scale streams (Q6_K: meta, Q8_0: d)
            p0 = ids ? __ldg(ids + r) : r;
            p1 = ids ? __ldg(ids + r + 1) : r + 1;
            if (LSE && r < nq4) { // (rows come in pairs and nq4 is even)
                rp[0] = cW.dq4 + (size_t) r * DQ4_RB;
                rp[1] = rp[0] + DQ4_RB;
                rows_t<TL, 5, 2, T_Q4K>(rp, dp, sx, a);
            } else {
                rp[0] = head.w + (size_t) p0 * head.rb;
                rp[1] = head.w + (size_t) p1 * head.rb;
                dp[0] = head.d + (size_t) p0 * head.nd;
                dp[1] = head.d + (size_t) p1 * head.nd;
                if constexpr (H16) {
                    if (q8) rows_t<TL, 5, 2, T_Q8_0>(rp, dp, sxh, a);
                    else { // one unit per call (the 5-unit loop's loads and sums, in its order)
                        const uint8_t * ru[2] = {rp[0], rp[1]};
                        const __half * du[2] = {dp[0], dp[1]};
                        SXH su = sxh;
#pragma unroll 1
                        for (int i = 0; i < 5; i++) {
                            rows_t<TL, 1, 2, T_Q6K, U_D - 32>(ru, du, su, a); // (XP: the token stride stays U_D)
                            ru[0] += 4 * Q6_BLK;
                            ru[1] += 4 * Q6_BLK;
                            du[0] += 4 * Q6_META / 2;
                            du[1] += 4 * Q6_META / 2;
                            su.xa += 32;
                            su.xb += 32;
                            su.xd += 16;
                            su.xq += 32;
                        }
                    }
                } else {
                    if (q8) rows_t<TL, 5, 2, T_Q8_0>(rp, dp, sx, a);
                    else rows_t<TL, 5, 2, T_Q6K>(rp, dp, sx, a);
                }
            }
        }
        if constexpr (H16) wsum_rows16<16, 2>(a); // lanes 2j, 2j + 1: row j's sums in a[0][0], a[1][0]
        else wsum_acc<TL, 2>(a);
        if (pen) { // a[k][t] <- the penalised true logit
            const int c0 = __ldcg(cB.pcnt + p0), c1 = __ldcg(cB.pcnt + p1);
            if constexpr (H16) { // row t = lane >> 1 only: the counts of the drafts [0, nx0 + t) (as below)
                const int t = lane >> 1;
                int e0 = c0, e1 = c1;
                for (int i = 0; i < nx0 + t && i < MAXD - 1; i++) {
                    const int x = xts[i];
                    e0 += x == p0;
                    e1 += x == p1;
                }
                a[0][0] = pen_logit(a[0][0], m[t], e0, pres, freq);
                a[1][0] = pen_logit(a[1][0], m[t], e1, pres, freq);
            } else {
#if MK_LMH8
                int xr[XR ? MAXD - 1 : 1];
                if constexpr (XR) {
#pragma unroll
                    for (int i = 0; i < MAXD - 1; i++) xr[i] = xts[i];
                }
#endif
#pragma unroll
                for (int t = 0; t < TL; t++) {
                    int e0 = c0, e1 = c1;
#pragma unroll
                    for (int i = 0; i < MAXD - 1; i++)
                        if (i < nx0 + t) {
                            int x;
                            if constexpr (XS) x = xts[i];
#if MK_LMH8
                            else if constexpr (XR) x = xr[i];
#endif
                            else x = xt[i];
                            e0 += x == p0;
                            e1 += x == p1;
                        }
                    a[0][t] = pen_logit(a[0][t], m[t], e0, pres, freq);
                    a[1][t] = pen_logit(a[1][t], m[t], e1, pres, freq);
                }
            }
        }
        // (below, pen ? a : a * m = the true logits: pen's a already are. LSE: __fmul_rn keeps v - mx a separate
        // multiply and subtract, as without the select (no contraction))
        if constexpr (LSE) {
            const float v0 = pen ? a[0][0] : __fmul_rn(a[0][0], m[0]), v1 = pen ? a[1][0] : __fmul_rn(a[1][0], m[0]);
            const float mx = fmaxf(lm_m, fmaxf(v0, v1));
            lm_s = lm_s * __expf(lm_m - mx) + __expf(v0 - mx) + __expf(v1 - mx);
            lm_m = mx;
        }
        if constexpr (FULL) {
            if (lane == 0) {
#pragma unroll
                for (int t = 0; t < TL; t++) {
                    cB.logits[(size_t) (t0 + t) * VOCAB + r] = a[0][t] * m[t];
                    cB.logits[(size_t) (t0 + t) * VOCAB + r + 1] = a[1][t] * m[t];
                }
            }
        }
        if constexpr (H16 && TOPK) { // the lane's true logit (topk_insert's argument in the second pass) to the scratch line
            const float s = (lane & 1) ? a[1][0] : a[0][0];
            __stcg(scr, pen ? s : s * m[lane >> 1]);
            scr += (size_t) nwarps() * 32;
        } else if constexpr (H16) { // lanes 2j, 2j + 1 track row j only (2 registers, not 32; the same sequence per row)
            if (a[0][0] > bv[0]) { bv[0] = a[0][0]; bi[0] = r; }
            if (a[1][0] > bv[0]) { bv[0] = a[1][0]; bi[0] = r + 1; }
        } else {
#pragma unroll
            for (int t = 0; t < TL; t++) {
                if constexpr (TOPK) { // true logits: apply the final RMS norm scale (irrelevant to the argmax)
                    topk_insert(bv[t], bi[t], pen ? a[0][t] : a[0][t] * m[t], r);
                    topk_insert(bv[t], bi[t], pen ? a[1][t] : a[1][t] * m[t], r + 1);
                } else {
                    if (a[0][t] > bv[t]) { bv[t] = a[0][t]; bi[t] = r; }
                    if (a[1][t] > bv[t]) { bv[t] = a[1][t]; bi[t] = r + 1; }
                }
            }
        }
    }
    if constexpr (H16 && TOPK) { // second pass over the warp's lines: the same topk_insert sequence per row as the one the
        // stream made in the base (ascending r, vocab row r before r + 1; the rows' lists are independent), so the lists,
        // the merges and the candidates are the same bit for bit. Only the lists are live here: no spills.
        static_assert(TL == 16, "one 128 B line = 16 rows x 2 vocab rows");
        __syncwarp(); // (the lines were written by all lanes)
#pragma unroll
        for (int t = 0; t < TL; t++) {
            bv[t] = -INFINITY;
            bi[t] = -1;
        }
        const float4 * sl = reinterpret_cast<const float4 *>(cB.lmscr) + (size_t) gwarp() * 8;
        for (int r = 2 * gwarp(); r < head.N; r += 2 * nwarps(), sl += (size_t) nwarps() * 8) {
#pragma unroll
            for (int i = 0; i < 8; i++) { // floats 4i .. 4i + 3: rows 2i (r, r + 1), 2i + 1 (r, r + 1); warp-uniform loads
                const float4 q = __ldcg(sl + i);
                topk_insert(bv[2 * i], bi[2 * i], q.x, r);
                topk_insert(bv[2 * i], bi[2 * i], q.y, r + 1);
                topk_insert(bv[2 * i + 1], bi[2 * i + 1], q.z, r);
                topk_insert(bv[2 * i + 1], bi[2 * i + 1], q.w, r + 1);
            }
        }
    }
    if constexpr (LSE) {
        if (lane == 0) {
            m[LM_WLSE + 2 * warp] = lm_m;
            m[LM_WLSE + 2 * warp + 1] = lm_s;
        }
    }
    // TOPK: lists [TLL][16][32] in shared memory for the merges (the activations are no longer needed)
    float * const tkv = reinterpret_cast<float *>(sm);
    int * const tki = reinterpret_cast<int *>(sm) + TLL * 512;
    int * const tks = reinterpret_cast<int *>(sm) + 2 * TLL * 512;
    if constexpr (TOPK) {
        // merge the warps' lists into the block's (the order of a sequential merge over warps 0..15: tk_merge); H16 per
        // half (each row's merge is the same)
        __syncthreads();
#pragma unroll
        for (int h = 0; h < TL / TLL; h++) {
#pragma unroll
            for (int t = 0; t < TLL; t++) {
                const int a = (t * 16 + warp) * 32 + lane;
                tkv[a] = bv[TLL * h + t];
                tki[a] = bi[TLL * h + t];
                tks[a] = warp * 32 + lane;
            }
            __syncthreads();
            tk_tree16<TLL>(tkv, tki, tks);
            for (int q = tid; q < TLL * 32; q += NT) {
                const int t = q >> 5, a = t * 512 + (q & 31);
                cB.blk_best[((size_t) blockIdx.x * BS + TLL * h + t) * TOPK_MAX + (q & 31)] = make_float2(tkv[a], __int_as_float(tki[a]));
            }
            if (h + 1 < TL / TLL) __syncthreads(); // (the next half's lists overwrite these)
        }
    } else if constexpr (H16) {
        float * const wv = reinterpret_cast<float *>(sm); // [NW][16] (the activations are no longer needed)
        int * const wi = reinterpret_cast<int *>(sm) + NW * 16;
        __syncthreads();
        if (!(lane & 1)) { // (lanes 2j, 2j + 1 hold row j)
            wv[warp * 16 + (lane >> 1)] = bv[0];
            wi[warp * 16 + (lane >> 1)] = bi[0];
        }
        __syncthreads();
        if (tid < TL) {
            float b = -INFINITY;
            int ix = 0;
            for (int w = 0; w < NW; w++) {
                const float v = wv[w * 16 + tid];
                const int id = wi[w * 16 + tid];
                if (v > b || (v == b && id < ix)) { b = v; ix = id; }
            }
            cB.blk_best[blockIdx.x * BS + tid] = make_float2(b, __int_as_float(ix));
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
    if constexpr (LSE) { // block's (max, sum of exp); the per-warp values were published before a barrier above
        if (tid == NT - 1) {
            float M = -INFINITY, S = 0.f;
            for (int w = 0; w < NW; w++) M = fmaxf(M, m[LM_WLSE + 2 * w]);
            for (int w = 0; w < NW; w++) S += m[LM_WLSE + 2 * w + 1] * __expf(m[LM_WLSE + 2 * w] - M);
            cB.blk_lse[blockIdx.x] = make_float2(M, S);
        }
    }
#ifndef MK_FORMAL_SYNC
#define MK_FORMAL_SYNC 1 // (default: mk_shared.cuh)
#endif
#if MK_FORMAL_SYNC
    // the ticket in the PTX model's terms (correct6 F4): the block's stores -> bar.sync -> thread 0's fence + ticket is a release
    // pattern for every thread's stores (causality); thread 0's ticket + fence -> bar.sync an acquire for every reading lane
    __syncthreads();
    int * s_last = reinterpret_cast<int *>(m + LM_LAST);
    if (tid == 0) {
        __threadfence();
        const bool last = atomicAdd(cB.cnt, 1u) == gridDim.x - 1;
        if (last) __threadfence();
        s_last[0] = last;
    }
    __syncthreads();
    if (!s_last[0]) return false;
#else
    __threadfence();
    __syncthreads();
    int * s_last = reinterpret_cast<int *>(m + LM_LAST);
    if (tid == 0) s_last[0] = (atomicAdd(cB.cnt, 1u) == gridDim.x - 1);
    __syncthreads();
    if (!s_last[0]) return false;
    __threadfence();
#endif
#if MK_PROFILE
    sub_stamp(0);
#endif
    if constexpr (LSE) { // the blocks' (max, sum of exp) -> shared memory in one parallel load; the sum stays in block order
        // On the last warp alone (warp syncs only): warp 0's argmax below runs alongside. The max and the sum stay chains in
        // block order (every lane chains the max, its loads batched ahead; lane l takes the exps of blocks l, l + 32, ..:
        // the same operations on the same values, the sum's step the FFMA it compiled to). Was ~2.6 us serial on one thread.
        if (warp == NW - 1) {
            float2 * const sl = reinterpret_cast<float2 *>(sm + SM_LSE);
            const int G = gridDim.x;
            for (int k = lane; k < G; k += 32) sl[k] = __ldcg(&cB.blk_lse[k]);
            __syncwarp();
            float M = -INFINITY;
            int k = 0;
            for (; k + 8 <= G; k += 8) {
                float v[8];
#pragma unroll
                for (int i = 0; i < 8; i++) v[i] = sl[k + i].x;
#pragma unroll
                for (int i = 0; i < 8; i++) M = fmaxf(M, v[i]);
            }
            for (; k < G; k++) M = fmaxf(M, sl[k].x);
            __syncwarp();
            for (int q = lane; q < G; q += 32) sl[q].x = __expf(sl[q].x - M); // (block q's exp, in place)
            __syncwarp();
            if (lane == 31) {
                float S = 0.f;
                for (int q = 0; q < G; q++) S = __fmaf_rn(sl[q].y, sl[q].x, S); // (S += e.y * __expf(e.x - M))
                m[LM_LSE] = M + __logf(S);
            }
        }
    }
    if constexpr (TOPK) {
        // merge the blocks' lists (the order of a sequential merge over blocks 0..G-1): warp w merges lists
        // w, w + 16, ... of each row, then a tree over the 16 warps' lists (H16: per half)
#pragma unroll 1
        for (int h = 0; h < TL / TLL; h++) {
#pragma unroll 1
            for (int t = 0; t < TLL; t++) tk_grid_chain<BS>(TLL * h + t, t, tkv, tki, tks);
            // main head: the drafts' q' (previous launch) for sample_targets, staged while the merge runs
            if (!H16 && !draft)
                for (int q = tid; q < (TL - 1) * TOPK_MAX; q += NT) reinterpret_cast<float2 *>(sm + SM_DQ)[q] = __ldcg(cB.dq + q);
            __syncthreads();
            tk_tree16<TLL>(tkv, tki, tks);
            if (warp < TLL) {
                const int row = TLL * h + warp;
                const float v = tkv[warp * 512 + lane];
                int id = tki[warp * 512 + lane];
                if (draft && cW.draft_ids && id >= 0) id = cW.draft_ids[id];
                (cand ? cand : cB.cand)[row * TOPK_MAX + lane] = make_float2(v, __int_as_float(id));
                if (!H16) reinterpret_cast<float2 *>(sm + SM_CAND)[warp * TOPK_MAX + lane] = make_float2(v, __int_as_float(id));
                const int best = __shfl_sync(0xffffffffu, id, 0);
                if (lane == 0) {
                    s_tok[row] = best;
                    if (LSE) m[LM_BEST] = v;
                    if (!draft && !(v > -INFINITY)) cB.err[0] = 1;
                }
            }
            if (h + 1 < TL / TLL) __syncthreads(); // (the next half's chains overwrite the lists)
        }
    } else if (warp < TL) {
        // the blocks' best of row `warp`: "larger value, then smaller id" is a total order (NaN never wins), so a warp
        // reduction finds the same entry as the sequential scan over blocks
        float b = -INFINITY;
        int ix = 0;
        for (int k = lane; k < (int) gridDim.x; k += 32) {
            const float2 e = __ldcg(&cB.blk_best[k * BS + warp]);
            const int id = __float_as_int(e.y);
            if (e.x > b || (e.x == b && id < ix)) { b = e.x; ix = id; }
        }
#pragma unroll
        for (int o = 16; o >= 1; o >>= 1) {
            const float ob = __shfl_xor_sync(0xffffffffu, b, o);
            const int oi = __shfl_xor_sync(0xffffffffu, ix, o);
            if (ob > b || (ob == b && oi < ix)) { b = ob; ix = oi; }
        }
        if (lane == 0) {
            s_tok[warp] = (draft && cW.draft_ids) ? cW.draft_ids[ix] : ix;
            if (LSE) m[LM_BEST] = pen ? b : b * m[0];
            if (!draft && !(b > -INFINITY)) cB.err[0] = 1;
        }
    }
    if (tid == 0) *cB.cnt = 0;
    __syncthreads();
#if MK_PROFILE
    sub_stamp(1);
#endif
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
// Keys: a row's acceptance / residual uniforms by its position, a draft's by its absolute position. The per-token rule
// uses each once (a position is decided once), and a draft drawn again at a position (after a rejection before it) is
// independent of everything that decided its prefix: exact as is.

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

// q'(id) of a sampled draft's proposal list (prob, id) ... terminated by id -1. Sums every entry of the
// token: the draft head can hold a token twice (dv_add pads the draft vocabulary to an even row count with
// a duplicate), and the proposal probability of the token is then the sum.
__device__ __forceinline__ float dq_prob(const float2 * q, int id) {
    float s = 0.f;
    for (int j = 0; j < TOPK_MAX; j++) {
        const int qi = __float_as_int(q[j].y);
        if (qi < 0) break;
        if (qi == id) s += q[j].x;
    }
    return s;
}

// Sampled draft (single thread): q' = the sampling distribution (top_k, top_p, min_p, temp) of the draft
// head's top logits src; writes q' to dst as (prob, token id) terminated by (0, -1) (dst may alias src) and
// draws the draft from it with its own random stream, keyed by the draft's absolute position.
__device__ __forceinline__ int draft_sample(const Launch & P, const float2 * src, float2 * dst, unsigned long long pos) {
    float p[TOPK_MAX];
    const int K = cand_probs(src, P, p);
    float x = urand(P.seed ^ 0xD1B54A32D192ED03ull, pos);
    int pick = -1, last = __float_as_int(src[0].y);
    for (int j = 0; j < K; j++) {
        const float2 e = src[j];
        last = __float_as_int(e.y);
        if (pick < 0) {
            x -= p[j];
            if (x < 0.f) pick = last;
        }
        dst[j] = make_float2(p[j], e.y);
    }
    if (K < TOPK_MAX) dst[K] = make_float2(0.f, __int_as_float(-1));
    return pick >= 0 ? pick : last;
}

// draft_sample out of line, the launch's sampling fields as values (as sample_targets_ol). Inlined into the five
// mtp_pass<R>, cicc unrolls its candidate loops and keeps the 32 p[j] in registers, ptxas spills them, and the five
// spill blocks take disjoint slots of the stack frame (5 x 128 B above the MTP chain)
__device__ __noinline__ int draft_sample_ol(int top_k, float temp, float top_p, float min_p, unsigned long long seed,
                                            const float2 * src, float2 * dst, unsigned long long pos) {
    Launch S{}; // (the fields draft_sample and cand_probs read)
    S.top_k = top_k;
    S.temp = temp;
    S.top_p = top_p;
    S.min_p = min_p;
    S.seed = seed;
    return draft_sample(S, src, dst, pos);
}

// Sampled targets for the verified rows.  Point-mass drafts (argmax of the MTP): draft token d at row
// i is accepted with probability p_i(d); on rejection the row's token is drawn from p_i without d.
// Sampled drafts (Ctl::dsmp, d ~ q'_i): accepted with probability min(1, p_i(d) / q'_i(d)); on
// rejection drawn from the residual norm(max(0, p_i - q'_i)).  Either way the output is distributed
// exactly as p, and the rows after a rejection are dropped.  Rows with no draft to verify (the bonus
// row, the last prompt row) are drawn from p_i.  tg then drives the greedy acceptance logic of
// main_control unchanged.  cand: the rows' candidates (cB.cand, here the shared-memory copy lm_head leaves at
// SM_CAND), dq: the drafts' proposals (cB.dq, staged at SM_DQ): single-thread code, so every load is latency.
__device__ __forceinline__ void sample_targets(const Launch & P, int T, int tl, int * tg, const float2 * cand, const float2 * dq) {
    const Ctl * c = cB.ctl;
    const unsigned long long n = (unsigned long long) c->n;
    float p[TOPK_MAX];
    for (int i = 0; i < tl; i++) {
        const float2 * cd = cand + i * TOPK_MAX;
        const int K = cand_probs(cd, P, p);
        if (P.mode == MODE_DECODE && i < T - 1 && c->dsmp) {
            const int d = c->tok[i + 1];
            const float2 * q = dq + i * TOPK_MAX;
            float pd = 0.f;
            for (int j = 0; j < K; j++)
                if (__float_as_int(cd[j].y) == d) pd += p[j];
            if (urand(P.seed, 2 * (n + i)) * dq_prob(q, d) < pd) {
                tg[i] = d;
                continue;
            }
            float r[TOPK_MAX], rs = 0.f;
            for (int j = 0; j < K; j++) rs += (r[j] = fmaxf(0.f, p[j] - dq_prob(q, __float_as_int(cd[j].y))));
            const float u = urand(P.seed, 2 * (n + i) + 1);
            if (!(rs > 0.f)) { // rounding only; the row must not return the rejected d
                tg[i] = cand_draw(cd, p, K, d, u);
                return;
            }
            float x = u * rs;
            int pick = -1;
            for (int j = 0; j < K; j++) {
                if (r[j] <= 0.f) continue;
                pick = __float_as_int(cd[j].y);
                x -= r[j];
                if (x < 0.f) break;
            }
            tg[i] = pick;
            return;
        }
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

// sample_targets out of line, the launch's sampling fields as values: its p / r arrays (dynamically indexed, 256 B of
// local memory) get a frame of their own beside the main pass's call chain, and a caller's Launch copy (a lane's) none
__device__ __noinline__ void sample_targets_ol(int mode, int top_k, float temp, float top_p, float min_p,
                                               unsigned long long seed, int T, int tl, int * tg, const float2 * cand,
                                               const float2 * dq) {
    Launch S{}; // (the fields sample_targets and cand_probs read)
    S.mode = mode;
    S.top_k = top_k;
    S.temp = temp;
    S.top_p = top_p;
    S.min_p = min_p;
    S.seed = seed;
    sample_targets(S, T, tl, tg, cand, dq);
}

// ---- main-model control (single thread) ----
#include "wv.cuh"

// The blocks without items in a short-context attention call (NKV nch < gridDim.x, L_ctx <= 2 ATT_BUCKET: the call's K
// / V, <= 2.8 MB, leaves the lines in L2; at 66 positions 70 of 82 blocks wait ~15 us at the attn_core barrier, DRAM
// idle through attn_core and attn_combine) prefetch part of resid_o's weights to L2: of each of out's (W.o, wv layout)
// D / 16 x WV_S (tile, split-K chunk) pairs dealt by resid_wv / resid_wv_ck, the first ATT_PFO_L lines of the pair's
// SECOND super-block unit (CH kc + 1; the first is their own phase-top prefetch, wv_pf_unit): 1280 x 1 KB, normal
// priority (the weight loads' evict_first lines go first). ib / nib: the block's index among the call's idle blocks /
// their count. Hints only.
__device__ __forceinline__ void attn_idle_pf(const Mat & m, int ib, int nib) {
    if (!m.fm) return;
    constexpr int NP = D / 16 * WV_S;
#if MK_L2_TUNE
    const int nsb = m.K >> 8, ch = nsb / WV_S, ub = wv_ubytes(m.type), lpu = min(((ch - 1) * ub + 127) / 128, cL2.attn);
#else
    const int nsb = m.K >> 8, ch = nsb / WV_S, ub = wv_ubytes(m.type), lpu = min((ub + 127) / 128, ATT_PFO_L);
#endif
    for (int i = ib * NT + (int) threadIdx.x; i < NP * lpu; i += nib * NT) {
        const int p = i / lpu, l = i - p * lpu, tile = p / WV_S, kc = p - tile * WV_S;
        pf_l2n(m.w + ((size_t) tile * nsb + ch * kc + 1) * ub + 128 * l);
    }
}

__device__ void main_control(int mode, int last, int T, const int * tgt, bool pen, bool mtp) {
    Ctl * c = cB.ctl;
    const int n = c->n;
    int a, bonus = -1;
    // pen: count the emitted tokens (the response so far) for the presence / frequency penalty
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
    for (int i = 0; i < c->n_out; i++)
        if (c->out[i] < 0) cB.err[0] = 1;
    const int m = c->mtp_valid;
    const int P = bonus >= 0 ? n + a + 1 : n + T - 1;
    if (mtp && (m < n || P - m >= MAXD + 3)) cB.err[1] = 1; // the loop below only knows tokens >= n and holds MAXD + 3 rows
    int R = 0;
    for (int p = m; p <= P && R < MAXD + 3; p++, R++) {
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
    if (pen)
        for (int i = 0; i < c->n_out; i++) {
            const int v = c->out[i];
            if (cB.pcnt[v] < 0xffff) cB.pcnt[v]++;
        }
}

// The main loops' layer state lives in shared memory and is re-read after every phase call (ml_*): a value held in a
// register across the out-of-line phase calls is spilled, and its reload after the next grid barrier went to L2 at the
// start of every phase (Nsight Compute, an earlier version: ~140 us of long_sb per T5 launch in mk_cycle's inlined
// loops, ~115 us per T16 launch in main_forward_d). Per warp, written by all its lanes (no cross-warp race): layer L,
// delta-net layer dl.
// Per pass, written by thread 0 before the pass's first grid barrier, read after it: n, dn_c, n_commit, the pass's view.
__shared__ volatile int s_mlw[NW][2];
__shared__ volatile int s_mlb[4];
static_assert(SM_TOTAL + 192 <= 99 * 1024, "mk_cycle's dynamic + static shared memory (s_bar0 .., s_c1n, s_mlw, s_mlb)");
__device__ __forceinline__ void ml_begin(int n_commit) {
    if (threadIdx.x == 0) {
        volatile Ctl * vc = cB.ctl;
        s_mlb[0] = vc->n;
        s_mlb[1] = vc->dn_c;
        s_mlb[2] = n_commit;
        s_mlb[3] = s_view;
    }
    s_mlw[threadIdx.x >> 5][0] = 0;
    s_mlw[threadIdx.x >> 5][1] = 0;
}
__device__ __forceinline__ int ml_L() { return s_mlw[threadIdx.x >> 5][0]; }
__device__ __forceinline__ int ml_dl() { return s_mlw[threadIdx.x >> 5][1]; }
__device__ __forceinline__ int ml_n() { return s_mlb[0]; }
__device__ __forceinline__ int ml_dnc() { return s_mlb[1]; }
__device__ __forceinline__ int ml_ncommit() { return s_mlb[2]; }
__device__ __forceinline__ int ml_view() { return s_mlb[3]; }
__device__ __forceinline__ const LayerW & ml_W() { return cW.L[ml_L()]; }
__device__ __forceinline__ const float * ml_wnext() {
    const int L = ml_L();
    return (L + 1 < NL) ? cW.L[L + 1].attn_norm : cW.onorm;
}
// next layer / delta-net layer: every lane of the warp reads the old value before any lane stores the new one
__device__ __forceinline__ void ml_next_L() {
    const int L = ml_L();
    __syncwarp();
    s_mlw[threadIdx.x >> 5][0] = L + 1;
}
__device__ __forceinline__ void ml_next_dl() {
    const int d = ml_dl();
    __syncwarp();
    s_mlw[threadIdx.x >> 5][1] = d + 1;
}

template <int T>
__device__ void main_forward(const Launch & P, uint8_t * /*sm*/, unsigned & epoch) {
    uint8_t * const sm = smem;
    ml_begin(P.mode == MODE_DECODE ? 1 : T); // (n, dn_c, n_commit: read after the embed barrier)
    embed_prologue<T>();
    gsync(epoch, PH_EMBED);
    for (; ml_L() < NL; ml_next_L()) {
        // wv phases where at least one matrix of the phase is in the wv layout (the others: dp4a, per tensor at every T).
        // T >= 6: always the wv phase functions (their dp4a part runs the hybrid types, the only ones not in
        // the wv layout there), and every main-layer ffn_down is in the wv layout (the loader checks it).
        // (W = ml_W() is bound anew after every call: nothing of the layer is held across one.)
        constexpr bool WIDE = T > 5;
        if (ml_W().attn) {
            {
                const LayerW & W = ml_W();
                if constexpr (WIDE) attn_proj_wv<T>(WV_LAYER_ARG(W, ml_L()), ml_L() / 4, ml_n());
                else if (W.q.fm || W.k.fm || W.v.fm) attn_proj_wv<T>(WV_LAYER_ARG(W, ml_L()), ml_L() / 4, ml_n());
                else attn_proj<T>(W, ml_L() / 4, ml_n(), sm);
            }
            gsync(epoch, PH_ATTN_PROJ);
            attn_core<T>(ml_W(), ml_L() / 4, ml_n(), sm);
            gsync(epoch, PH_ATTN_CORE);
            attn_combine<T>(ml_n(), 0, ml_L() / 4);
            gsync(epoch, PH_ATTN_COMB);
            {
                const LayerW & W = ml_W();
                if (W.o.fm) resid_out_wv<T>(W.o, W.post_norm);
                else resid_dp<T, 6>(W.o, cB.qo, cB.x, true, W.post_norm, sm);
            }
            gsync(epoch, PH_RESID_O);
        } else {
            {
                const LayerW & W = ml_W();
                if constexpr (WIDE) dn_proj_wv<T>(WV_LAYER_ARG(W, ml_L()), ml_dl(), ml_n(), ml_dnc());
                else if (W.qkv.fm || W.gate.fm) dn_proj_wv<T>(WV_LAYER_ARG(W, ml_L()), ml_dl(), ml_n(), ml_dnc());
                else dn_proj<T>(W, ml_dl(), ml_n(), ml_dnc(), sm);
            }
            k5_arrive(epoch, PH_DN_PROJ); // split barrier: dn_core waits after its replay (k5_wait)
            dn_core<T>(ml_W(), ml_dl(), ml_n(), ml_dnc(), ml_ncommit(), sm);
            gsync(epoch, PH_DN_CORE);
            {
                const LayerW & W = ml_W();
                if (W.sout.fm) resid_out_wv<T>(W.sout, W.post_norm);
                else resid_dp<T, 6>(W.sout, cB.qo, cB.x, true, W.post_norm, sm);
            }
            gsync(epoch, PH_RESID_SOUT);
            ml_next_dl();
        }
        // C1: no grid barrier between gate / up and ffn_down: each ffn_down chunk waits for its own gate / up tiles
        // (resid_wv role bit 1). Not with a dp4a ffn_gu (T <= 5, gate and up both off the wv layout).
        bool c1;
        {
            const LayerW & W = ml_W();
            c1 = WIDE || ((W.fg.fm || W.fu.fm) && W.fd.fm);
        }
        if (c1) {
            ffn_gu_wv<T>(ml_W(), true);
            resid_wv<T, true, FF>(ml_W().fd, cB.qa, ml_wnext(), 3);
            gsync(epoch, PH_FFN_DOWN);
            continue;
        }
        {
            const LayerW & W = ml_W();
            if constexpr (WIDE) ffn_gu_wv<T>(W);
            else if (W.fg.fm || W.fu.fm) ffn_gu_wv<T>(W);
            else ffn_gu<T>(W, sm);
        }
        gsync(epoch, PH_FFN_GU);
        {
            const LayerW & W = ml_W();
            if constexpr (WIDE) resid_wv<T, true, FF>(W.fd, cB.qa, ml_wnext(), 1);
            else if (W.fd.fm) resid_wv<T, true, FF>(W.fd, cB.qa, ml_wnext(), 1);
            else resid_any<T, 17>(W.fd, cB.qa, cB.x, true, ml_wnext(), sm);
        }
        gsync(epoch, PH_FFN_DOWN);
    }
    write_h_rows<T>(cB.x, cB.qn, cW.onorm, cB.h_hist, ml_n(), true, sm);

    int * s_tok = reinterpret_cast<int *>(misc(sm) + 400);
    bool am_last;
    int tl = 0;
    const bool sampling = P.temp > 0.0f && !P.score;
    const bool pen = (P.pres != 0.f || P.freq != 0.f) && !P.score; // (the first token of a response has nothing to penalise)
    // The wide pass (T = MAXT): decode launches of P.T = 6 .. MAXT rows run as one MAXT-row pass; the rows
    // P.T .. MAXT - 1 are padding (stale ctl tokens: their K / V, conv, delta-net ring and h rows land at future
    // positions, rewritten before any real row reads them; real rows never attend to them, and the host keeps
    // positions n + 1 .. n + MAXT in one attention partition). Only rows < P.T reach the verify / control.
    // (Prefill and scoring launches stay at T <= 5: no FULL lm_head here.)
    const int Tr = T == MAXT ? P.T : T;
    if (P.mode == MODE_DECODE || P.score) {
        if (sampling) am_last = lm_head<T, true>(0, false, sm, s_tok, nullptr, pen, P.pres, P.freq, 0);
        else if (P.score == 2) {
            if constexpr (T <= 5) am_last = lm_head<T, false, true>(0, false, sm, s_tok);
            else am_last = false;
        } else am_last = lm_head<T>(0, false, sm, s_tok, nullptr, pen, P.pres, P.freq, 0);
        tl = Tr;
    } else if (P.last) {
        am_last = sampling ? lm_head<1, true>(T - 1, false, sm, s_tok) : lm_head<1>(T - 1, false, sm, s_tok);
        tl = 1;
    } else {
        am_last = blockIdx.x == 0;
    }
    if (am_last && threadIdx.x == 0) {
        int tg[MAXT];
        for (int i = 0; i < tl; i++) tg[i] = s_tok[i];
        if (sampling) // lm_head left the candidates at SM_CAND and the drafts' q' at SM_DQ
            sample_targets_ol(P.mode, P.top_k, P.temp, P.top_p, P.min_p, P.seed, Tr, tl, tg,
                              reinterpret_cast<const float2 *>(sm + SM_CAND), reinterpret_cast<const float2 *>(sm + SM_DQ));
#if MK_PROFILE
        sub_stamp(2);
#endif
        for (int i = 0; i < tl; i++) cB.ctl->tgt[i] = tg[i];
        main_control(P.mode, P.last, Tr, P.score ? s_tok + T - 1 : tg, pen, !P.score);
        __threadfence();
#if MK_PROFILE
        sub_stamp(3);
#endif
    }
    gsync(epoch, PH_LMHEAD);
}

// MTP L2 prefetch. Each MTP phase streams its weights cold behind a grid barrier (fill, barrier and tail: DRAM idle),
// and per draft step M0, the attention and its combine leave DRAM idle on most blocks. A block prefetches the first
// lines of the rows it reads first in the next weight phase, from inside the idle phase or right before the barrier:
// ~3-4 MB per phase, normal priority (the weight loads' evict_first lines go first), addresses inside the rows. Warps
// 1.. only: thread 0 runs the barrier's fence right after. Hints: no value changes. (thread / block index read afresh,
// tid_x / ctaid_x: a copy kept across the phase calls was spilled, and its reload after the barrier's L1 invalidation
// sat on the way to the next barrier)

#if MK_L2_TUNE
// the same lines as the constant build's below at the default budgets; more lines than threads 32..: a thread takes several
__device__ __forceinline__ void mtp_pf_l2(const void * p) {
    if (cL2.keep) pf_l2(p);
    else pf_l2s(p);
}
__device__ __forceinline__ void mtp_pf_resid(const Mat & m, int) {
    const int tid = tid_x(), lines = cL2.mtpr, b = (int) ctaid_x();
    if (tid < 32 || 32 * (b + 1) > m.N) return;
    for (int i = tid - 32; i < 32 * lines; i += NT - 32) {
        const int r = i / lines, l = i - r * lines;
        if (128 * l < m.rb) mtp_pf_l2(m.w + (size_t) (32 * b + r) * m.rb + 128 * l);
    }
}
__device__ __forceinline__ void mtp_pf_m0eh() { mtp_pf_resid(cW.eh, MTP_PFR); }
// f(global warp of the block whose first pair this is, line of the pair, lines per row)
template <class F>
__device__ __forceinline__ void mtp_pf_pairs(F && f) {
    const int tid = tid_x(), lines = cL2.mtp;
    if (tid < 32) return;
    for (int i = tid - 32; i < 2 * lines * NW; i += NT - 32) {
        const int w = i / (2 * lines), l = i - w * 2 * lines;
        f(w * (int) gridDim.x + (int) ctaid_x(), l, lines);
    }
}
__device__ __forceinline__ void mtp_pf_line(const uint8_t * a, const uint8_t * b, int rb, int l, int lines) {
    const int ll = l < lines ? l : l - lines;
    if (128 * ll < rb) mtp_pf_l2((l < lines ? a : b) + 128 * ll);
}
__device__ __forceinline__ void mtp_pf_q(const LayerW & W) {
    mtp_pf_pairs([&](int gw, int l, int lines) {
        if (2 * gw + 1 >= W.q.N) return;
        const uint8_t * a = W.q.w + (size_t) 2 * gw * W.q.rb;
        mtp_pf_line(a, a + W.q.rb, W.q.rb, l, lines);
    });
}
__device__ __forceinline__ void mtp_pf_kv(const LayerW & W) {
    mtp_pf_pairs([&](int gw, int l, int lines) {
        const int nk = KVD / 2;
        if (gw >= 2 * nk) return;
        const Mat & m = gw < nk ? W.k : W.v;
        const uint8_t * a = m.w + (size_t) 2 * (gw < nk ? gw : gw - nk) * m.rb;
        mtp_pf_line(a, a + m.rb, m.rb, l, lines);
    });
}
__device__ __forceinline__ void mtp_pf_gu(const LayerW & W) {
    mtp_pf_pairs([&](int j, int l, int lines) {
        if (j >= FF / 16) return;
        mtp_pf_line(W.fg.w + (size_t) 16 * j * W.fg.rb, W.fu.w + (size_t) 16 * j * W.fu.rb, W.fg.rb, l, lines);
    });
}
__device__ __forceinline__ void mtp_pf_head() {
    mtp_pf_pairs([&](int gw, int l, int lines) {
        if (2 * gw >= cW.dq4_n) return;
        const uint8_t * a = cW.dq4 + (size_t) 2 * gw * DQ4_RB;
        mtp_pf_line(a, a + DQ4_RB, DQ4_RB, l, lines);
    });
}
#else
// resid32's first item of this block (rows 32 b .. 32 b + 31): the first `lines` (<= 15) 128-B lines of each row
__device__ __forceinline__ void mtp_pf_resid(const Mat & m, int lines) {
    const int i = tid_x() - 32, r = i / lines, l = i - r * lines, b = (int) ctaid_x();
    if (i < 0 || r >= 32 || 128 * l >= m.rb || 32 * (b + 1) > m.N) return;
    pf_l2s(m.w + (size_t) (32 * b + r) * m.rb + 128 * l);
}
// eh's first rows, from the M0 phase. The 4 MB go out while the M0 blocks' latency chain runs (h, token_embd, norm
// weights, instruction fetches: ~10 serial misses behind the flood): mtp_m0 +2.2 .. 2.6 us per draft step, mtp_eh
// -0.5 .. -0.7 us. Folding the draft passes' M0 into the previous lm_head phase (no M0 phase, so no prefetch from it)
// measured +0.06 .. 0.09 ms per launch
__device__ __forceinline__ void mtp_pf_m0eh() { mtp_pf_resid(cW.eh, MTP_PFR); }
// row-pair phases (gemv_pairs, ffn_gu_q4k, the draft head): warp w's first pair (rows a, b), its first MTP_PFL lines of
// each row (2880 B Q4_K rows: the first half), prefetched by thread 32 + 2 MTP_PFL w + l of the block
static_assert(32 + 2 * MTP_PFL * NW <= NT, "mtp_pf_gw: one line per thread");
// this thread's (global warp of the block whose pair it prefetches, line l of the pair); -1: none
__device__ __forceinline__ int mtp_pf_gw(int & l) {
    const int i = tid_x() - 32, w = i / (2 * MTP_PFL);
    l = i - w * 2 * MTP_PFL;
    return i < 0 || w >= NW ? -1 : w * (int) gridDim.x + (int) ctaid_x();
}
__device__ __forceinline__ void mtp_pf_line(const uint8_t * a, const uint8_t * b, int l) {
    pf_l2s(l < MTP_PFL ? a + 128 * l : b + 128 * (l - MTP_PFL));
}
// attn_proj's q pairs (gemv_pairs from item 0: warp gw's first pair is rows 2 gw, 2 gw + 1 of W.q)
__device__ __forceinline__ void mtp_pf_q(const LayerW & W) {
    int l;
    const int gw = mtp_pf_gw(l);
    if (gw < 0 || 2 * gw + 1 >= W.q.N) return;
    const uint8_t * a = W.q.w + (size_t) 2 * gw * W.q.rb;
    mtp_pf_line(a, a + W.q.rb, l);
}
// the K / V-only pass's k / v pairs (warps [0, KVD / 2): W.k, [KVD / 2, KVD): W.v)
__device__ __forceinline__ void mtp_pf_kv(const LayerW & W) {
    int l;
    const int gw = mtp_pf_gw(l), nk = KVD / 2;
    if (gw < 0 || gw >= 2 * nk) return;
    const Mat & m = gw < nk ? W.k : W.v;
    const uint8_t * a = m.w + (size_t) 2 * (gw < nk ? gw : gw - nk) * m.rb;
    mtp_pf_line(a, a + m.rb, l);
}
// ffn_gu_q4k's (warp item j = gwarp(): gate row 16 j and up row 16 j first)
__device__ __forceinline__ void mtp_pf_gu(const LayerW & W) {
    int l;
    const int j = mtp_pf_gw(l);
    if (j < 0 || j >= FF / 16) return;
    mtp_pf_line(W.fg.w + (size_t) 16 * j * W.fg.rb, W.fu.w + (size_t) 16 * j * W.fu.rb, l);
}
// the draft head's (rows 2 gwarp(), + 1; the MK_DRAFT_Q4 copy only)
__device__ __forceinline__ void mtp_pf_head() {
    int l;
    const int gw = mtp_pf_gw(l);
    if (gw < 0 || 2 * gw >= cW.dq4_n) return;
    const uint8_t * a = cW.dq4 + (size_t) 2 * gw * DQ4_RB;
    mtp_pf_line(a, a + DQ4_RB, l);
}
#endif
// the next draft's token_embd row (its M0's PCIe read of the mapped table; Q4_K rows), one line per lane of a warp
__device__ __forceinline__ void mtp_pf_embd(int tok) {
    const int lane = threadIdx.x & 31;
    if (cW.embd_t == T_Q4K && 128 * lane < EMBD_RB) pf_l2s(cW.embd + (size_t) tok * EMBD_RB + 128 * lane);
}

// M0 of the MTP block for row r, block s of its MTP_M0_NB: concat(enorm(embed(tok)), hnorm(h)) -> qe row r (mtp_pass,
// mtp_kv_pass, mtp_m0_rows). Each block computes the row's two norms (the same sums in the same order in every block) and
// quantises its two units per warp (one block doing all 320 units of a row would keep the grid waiting). The h row, the
// token_embd row (PCIe) and the units' norm weights load together; the h half (sum, norm, its unit) runs before the
// token_embd row is staged: same values, same arithmetic in the same order.
// Out of line: inlined, its registers (ten h loads in flight) pushed the callers' long-lived values to local memory
// (mtp_pass<1> +32 / +28 B, mtp_pass_b<4> +48 / +36 B spill st / ld: pos0 reloaded before attn_proj / core / combine).
// Only ints and a pointer cross the call (no float value): the same arithmetic.
static_assert(D % NT == 0 && 2 * U_D == 2 * MTP_M0_NB * NW, "mtp_m0_block: D / NT elements per thread, two units per warp");
__device__ __noinline__ void mtp_m0_block(int r, int s, int tok, const float * hsrc) {
    uint8_t * const sm = smem;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    float * se = reinterpret_cast<float *>(sm);
    float * sh = se + D;
    float * red = sh + D;
    const bool eq4 = cW.embd_t == T_Q4K; // (Q6_K / Q8_0 tables: embed_row_other leaves the row's values in se)
    uint8_t * const erow = reinterpret_cast<uint8_t *>(red + 32);
    float hp[D / NT];
#pragma unroll
    for (int k = 0; k < D / NT; k++) hp[k] = hsrc ? __ldcg(hsrc + tid + k * NT) : 0.f;
    uint4 erw = make_uint4(0u, 0u, 0u, 0u); // (stage_embd_row's load)
    if (eq4 && tid < EMBD_RB / 16) erw = reinterpret_cast<const uint4 *>(cW.embd + (size_t) tok * EMBD_RB)[tid];
    // unit j = s NW + warp + m M0_NB NW, m = 0, 1 (the former j loop's two iterations): m = 0 an embedding unit (j < U_D,
    // s < M0_NB), m = 1 an h unit (M0_NB NW == U_D). The norm weights load with the rows (were issued after the element
    // loop, their latency ahead of the quantisation)
    float nwt[2];
#pragma unroll
    for (int m = 0; m < 2; m++) {
        const int j = s * NW + warp + m * MTP_M0_NB * NW, e = 32 * (j % U_D) + lane;
        nwt[m] = j < U_D ? cW.enorm[e] : cW.hnorm[e];
    }
    // the h half first: its sum, norm and unit need only the h row (L2), so they run while the token_embd row (mapped host
    // memory, PCIe) is in flight. The same values in the same order: ah chains hp[k] = sh[tid + k NT] over k, as the
    // former element loop did (ah += hv * hv), bsum over the same partials, the same expressions for rsh and v
    float ah = 0.f;
#pragma unroll
    for (int k = 0; k < D / NT; k++) {
        sh[tid + k * NT] = hp[k];
        ah += hp[k] * hp[k];
    }
    ah = bsum(ah, red + 16); // (its barriers publish sh)
    {
        const int j = s * NW + warp + MTP_M0_NB * NW, e = 32 * (j % U_D) + lane;
        const float rsh = rsqrtf(ah / (float) D + EPS);
        q32_store(cB.qe, r, j, sh[e] * rsh * nwt[1], 0.f, false);
    }
    if (eq4) { // stage_embd_row's store and barrier
        if (tid < EMBD_RB / 16) reinterpret_cast<uint4 *>(erow)[tid] = erw;
        __syncthreads();
    } else {
        embed_row_other(tok, se, erow);
    }
    float ae = 0.f;
#pragma unroll
    for (int k = 0; k < D / NT; k++) { // (element e = tid + k NT: the order of the former e loop)
        const int e = tid + k * NT;
        const float v = eq4 ? dequant_q4k(erow, e) : se[e];
        se[e] = v;
        ae += v * v;
    }
    ae = bsum(ae, red);
    {
        const int j = s * NW + warp, e = 32 * (j % U_D) + lane;
        const float rse = rsqrtf(ae / (float) D + EPS);
        q32_store(cB.qe, r, j, se[e] * rse * nwt[0], 0.f, false);
    }
}

// The MTP passes' state in shared memory, re-read after every phase call (main_forward's ml_* pattern). Values held in
// registers across the out-of-line phase calls were spilled, and the grid barrier's acquire poll invalidates L1 (CCTL.IVALL),
// so each reload after a barrier went to L2 / DRAM on every block's path (Nsight Compute, 8-row code prompt:
// mtp_pass<1> after PH_MTP_FFN_DOWN li, the launch pointer, its sampling / penalty fields, the draft index: ~2.9 us per
// draft step before the draft head streams; logits and the prefetch's warp index before that barrier ~0.6 us;
// mtp_tail's draft loop i, ndraft, li, the ctl pointer: ~2.2 us per draft step before M0). Thread 0 writes, a later
// __syncthreads / grid barrier publishes.
// mtp_tail, per call: MS_LI .. MS_I (MS_ND: the draft passes' end, 0 without logits) and s_mtp_pl; mtp_pass, per pass
// (written before its first barrier, the previous pass's last reads are before its last one): MS_POS0, MS_K, MS_LG.
enum { MS_LI, MS_ND, MS_SMP, MS_PRES, MS_FREQ, MS_P, MS_I, MS_POS0, MS_K, MS_LG, MS_N };
__shared__ volatile int s_mtp[MS_N];
__shared__ const Launch * volatile s_mtp_pl;
static_assert(SM_TOTAL + 192 + 4 * MS_N + 8 <= 99 * 1024, "mk_cycle's dynamic + static shared memory (+ s_mtp, s_mtp_pl)");

// one MTP pass over R consecutive positions; rows read from ctl (catch-up: row0) or a draft row (draft_idx; dtok: its
// token, read by mtp_tail, on the M0 blocks). The launch the drafts are for: s_mtp (mtp_tail).
template <int R>
__device__ __noinline__ void mtp_pass(int row0, int dtok, bool draft_row, int draft_idx, bool logits, unsigned & epoch) {
    uint8_t * const sm = smem;
    volatile Ctl * vc = cB.ctl;
    constexpr int M0_NB = MTP_M0_NB;
    const int pos0 = draft_row ? s_mtp[MS_P] + draft_idx : vc->mtp_pos[row0];
    if (threadIdx.x == 0) {
        s_mtp[MS_POS0] = pos0;
        s_mtp[MS_K] = draft_row ? draft_idx : 0; // this pass produces draft k + 1 (tok[k + 1])
        s_mtp[MS_LG] = logits;
    }

    // M0: concat(enorm(embed(tok)), hnorm(h_prev)) -> qe (mtp_m0_block, M0_NB blocks per row)
    mtp_pf_m0eh();
    if (blockIdx.x < R * M0_NB) {
        const int r = blockIdx.x / M0_NB, s = blockIdx.x - r * M0_NB;
        const int pos = pos0 + r;
        const int tok = draft_row ? dtok : vc->mtp_tok[row0 + r];
        const float * hsrc = draft_row ? cB.hm : (pos >= 1 ? cB.h_hist + (size_t) ((pos - 1) % H_RING) * D : nullptr);
        mtp_m0_block(r, s, tok, hsrc);
    }
    gsync(epoch, PH_MTP_M0);
    const LayerW & W = cW.L[NL];
    resid_any<R, 10>(cW.eh, cB.qe, cB.xm, false, W.attn_norm, sm);
    mtp_pf_q(W);
    gsync(epoch, PH_MTP_EH);
    attn_proj<R>(W, 16, s_mtp[MS_POS0], sm);
    gsync(epoch, PH_MTP_ATTN_PROJ);
    attn_core<R>(W, 16, s_mtp[MS_POS0], sm);
    mtp_pf_resid(W.o, 12); // (the blocks without attention items at once: through the attention and its combine)
    gsync(epoch, PH_MTP_ATTN_CORE);
    {
        const int p0 = s_mtp[MS_POS0];
        attn_combine<R>(p0, mtp_ws(N_ATT_SLOTS - 1, p0), N_ATT_SLOTS - 1);
    }
    gsync(epoch, PH_MTP_ATTN_COMB);
    resid_any<R, 6>(W.o, cB.qo, cB.xm, true, W.post_norm, sm);
    mtp_pf_gu(W);
    gsync(epoch, PH_MTP_RESID_O);
    ffn_gu<R>(W, sm);
    mtp_pf_resid(W.fd, 12);
    gsync(epoch, PH_MTP_FFN_GU);
    resid_any<R, 17>(W.fd, cB.qa, cB.xm, true, cW.shnorm, sm);
    // (read before the barrier: without logits this barrier ends the pass, and the next pass's thread 0 rewrites MS_LG)
    const bool lgt = s_mtp[MS_LG];
    if (lgt) mtp_pf_head();
    gsync(epoch, PH_MTP_FFN_DOWN);
    if (lgt) {
        write_h_rows<R>(cB.xm, cB.qn, cW.shnorm, cB.hm, 0, false, sm);
        int * s_tok = reinterpret_cast<int *>(misc(sm) + 400);
        bool lastb;
        { // (the call's arguments from shared memory; the tail below re-reads what it needs: nothing held across the call)
            const int k = s_mtp[MS_K]; // this pass produces draft k + 1 (tok[k + 1])
            // sampled drafts need the draft head's top logits (TOPK path); argmax drafts only the reduction
#if MK_SPEC_LOG
            float2 * const tl = cB.dlog + k * (TOPK_MAX + 1);
            lastb = lm_head<1, true, false, true>(R - 1, true, sm, s_tok, tl);
#else
            float2 * const tl = cB.dq + k * TOPK_MAX;
            // presence / frequency penalty: draft k + 1 has the committed tokens and drafts 1..k in its context (the
            // launch's pres / freq: lane_launch keeps them)
            const float pres = __int_as_float(s_mtp[MS_PRES]), freq = __int_as_float(s_mtp[MS_FREQ]);
            const bool pen = pres != 0.f || freq != 0.f;
            lastb = s_mtp[MS_SMP] ? lm_head<1, true, false, true>(R - 1, true, sm, s_tok, tl, pen, pres, freq, k)
                                  : lm_head<1, false, false, true>(R - 1, true, sm, s_tok, nullptr, pen, pres, freq, k);
#endif
        }
        if (lastb) {
            // thread 0 decides the drafted token and the gates, the block barrier hands the token over (s_tok[1]), then
            // thread 0 writes the control words while warp 1 prefetches the next draft's token_embd row
            const int k = s_mtp[MS_K];
            int dt = 0, prod = 0, stop = 0, smp = 0;
            float conf = 0.f, chain = 0.f;
            if (threadIdx.x == 0) {
                // the launch the drafts are for (a lane's: lane_launch; fixed fields only, so the copy lives in registers)
                const Launch & PL = *s_mtp_pl;
                const int li = s_mtp[MS_LI];
                const Launch P = li < 0 ? PL : lane_launch(PL, li);
                Ctl * c = cB.ctl;
                const float * mm = misc(sm);
                conf = __expf(mm[LM_BEST] - mm[LM_LSE]);
                // (draft_sample reads lm_head's shared-memory copy of tl: the same values, without the L2 round trips)
                dt = P.spec_smp ? draft_sample_ol(P.top_k, P.temp, P.top_p, P.min_p, P.seed,
                                                  reinterpret_cast<const float2 *>(sm + SM_CAND),
                                                  cB.dq + k * TOPK_MAX, (unsigned long long) c->mtp_P + 1 + k)
                                : s_tok[0];
#if MK_PROFILE
                sub_stamp(2);
#endif
                // gate_chain 2 (policy chain=2) with sampled drafts: the chain runs over the drafted token's own probability
                // under q' (the distribution it was drawn from) instead of the draft head's top-1 probability
                chain = (k == 0 ? 1.f : c->dchain) * (P.gate_chain == 2 && P.spec_smp ? dq_prob(cB.dq + k * TOPK_MAX, dt) : conf);
                const float f = P.gate_chain ? chain : conf;
                smp = P.spec_smp;
                // (the gates are the launch's, lane_launch keeps them: read from PL, as a dynamic index into P would keep
                // the whole copy in local memory)
                prod = f < PL.gate_drop[k] ? k : k + 1;
                stop = f < PL.gate_stop[k] || f < PL.gate_drop[k];
                s_tok[1] = dt;
            }
            __syncthreads();
            if (threadIdx.x == 0) {
                Ctl * c = cB.ctl;
#if MK_SPEC_LOG
                const float * mm = misc(sm);
                cB.dlog[k * (TOPK_MAX + 1) + TOPK_MAX] = make_float2(mm[LM_LSE], mm[LM_BEST]);
#endif
                c->tok[1 + k] = dt;
                c->dsmp = smp;
                c->dconf[k] = conf;
                c->dchain = chain;
                c->nd_prod = prod;
                c->nd_stop = stop;
                __threadfence();
#if MK_PROFILE
                sub_stamp(3);
#endif
            }
            if (threadIdx.x >> 5 == 1) mtp_pf_embd(s_tok[1]); // (the next draft's token_embd row to L2)
        }
        gsync(epoch, PH_MTP_LMHEAD);
    }
}

#define MTP_KVC (!MK_KV_EMU)
#if MTP_KVC
// K / V-only MTP catch-up rows (MK_MTP_KVC). Of a cycle's catch-up rows (the accepted drafts + the bonus, R = a + 1) only
// the last row's MTP output is ever read (write_h_rows keeps the last row: it feeds the draft head and the chain); the
// other rows are there for their K / V rows in the MTP slot. Rows 0 .. R - 6 run this pass (M0, eh, the k / v projections,
// K norm + RoPE + store: the same code and per-row arithmetic as the full pass, hence the same K / V bits) and the last 5
// rows the full pass (the dp4a ffn_down holds <= 5 rows).
// The K / V store runs in the next pass's M0 phase (nothing there reads kbuf / vbuf or the MTP slot): no barrier of its own.
template <int R>
__device__ __noinline__ void mtp_kv_pass(int row0, unsigned & epoch) {
    uint8_t * const sm = smem;
    volatile Ctl * vc = cB.ctl;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    (void) tid, (void) warp;
    const int pos0 = vc->mtp_pos[row0];
    constexpr int M0_NB = MTP_M0_NB; // (M0 of mtp_pass, catch-up rows)
    mtp_pf_m0eh();
    if (blockIdx.x < R * M0_NB) {
        const int r = blockIdx.x / M0_NB, s = blockIdx.x - r * M0_NB;
        const int pos = pos0 + r;
        const int tok = vc->mtp_tok[row0 + r];
        const float * hsrc = pos >= 1 ? cB.h_hist + (size_t) ((pos - 1) % H_RING) * D : nullptr;
        mtp_m0_block(r, s, tok, hsrc);
    }
    gsync(epoch, PH_MTP_KV_M0);
    const LayerW & W = cW.L[NL];
    resid_any<R, 10>(cW.eh, cB.qe, cB.xm, false, W.attn_norm, sm);
    mtp_pf_kv(W);
    gsync(epoch, PH_MTP_KV_EH);
    {
        const SX sx = make_sx(sm, U_D);
        fill_xs<R>(cB.qn, 0, true, sx, misc(sm));
        const int nk = KVD / 2;
        gemv_pairs_any<R, 5>(W.k, 0, nk, PairOut{cB.kbuf, KVD, -1});
        gemv_pairs_any<R, 5>(W.v, nk, 2 * nk, PairOut{cB.vbuf, KVD, -1});
    }
    gsync(epoch, PH_MTP_KV_PROJ);
    constexpr int slot = N_ATT_SLOTS - 1;
    // (as attn_core's K / V store of the cycle's rows) on warp 0 of the LAST blocks: this runs inside the next pass's M0
    // phase, whose blocks are the first R' MTP_M0_NB <= 50 (gwarp() put these items on M0 blocks 0 .. R NKV - 1)
    for (int it = (threadIdx.x >> 5) * gridDim.x + (gridDim.x - 1 - blockIdx.x); it < R * NKV; it += nwarps()) {
        const int t = it / NKV, g = it - t * NKV, pp = pos0 + t;
        float v[8];
        load8(cB.kbuf + t * KVD + g * 256 + lane * 8, v);
        float ss = 0.f;
#pragma unroll
        for (int k = 0; k < 8; k++) ss += v[k] * v[k];
        const float rs = rsqrtf(wsum(ss) / 256.0f + EPS);
#pragma unroll
        for (int k = 0; k < 8; k++) v[k] *= rs * W.kn[lane * 8 + k];
        rope8(v, lane, pp);
        const size_t row = ((size_t) slot * NKV + g) * cB.ctx + pp; // (fp16 cache: attn_core_t's row)
        if (cB.kvq8) kv_store(0, slot, g, pp, v);
        else kv_store_f16(static_cast<__half *>(cB.kc) + row * HD, v);
        load8(cB.vbuf + t * KVD + g * 256 + lane * 8, v);
        if (cB.kvq8) kv_store(1, slot, g, pp, v);
        else kv_store_f16(static_cast<__half *>(cB.vc) + row * HD, v);
    }
}
#endif

template <int T>
__device__ void run_main(const Launch & P, uint8_t * /*sm*/, unsigned & epoch) {
    uint8_t * const sm = smem;
    main_forward<T>(P, sm, epoch);
}
// T = 6 .. MAXT: one MAXT-row pass (padding rows, see main_forward), out of line: a separate function keeps mk_cycle (and
// ptxas's memory for it) at the size of the T <= 5 passes
__device__ __noinline__ void run_wide(const Launch & P, unsigned & epoch) { main_forward<MAXT>(P, smem, epoch); }

// the launch's MTP passes (catch-up rows, then the drafts the next launch verifies) of the sequence of view s_view; li:
// its lane in the batched launch PL (-1: PL is its own launch). (A lane's Launch copy passed by reference would take
// 320 B of local memory, under every MTP pass in the stack frame.)
__device__ __noinline__ void mtp_tail(const Launch & PL, int li, unsigned & epoch) {
    volatile Ctl * vc = cB.ctl;
    const int ndraft = li < 0 ? PL.ndraft : PL.ln[li].nd; // (lane_launch's ndraft)
    const int Rtot = vc->mtp_R;
    const bool lg = vc->mtp_logits;
    if (threadIdx.x == 0) { // the passes' launch state (s_mtp; the callers' barrier / set_view ended the previous lane's reads)
        s_mtp[MS_LI] = li;
        s_mtp[MS_ND] = lg ? ndraft : 0;
        s_mtp[MS_SMP] = li < 0 ? PL.spec_smp : PL.ln[li].spec_smp; // (lane_launch's spec_smp)
        s_mtp[MS_PRES] = __float_as_int(PL.pres);
        s_mtp[MS_FREQ] = __float_as_int(PL.freq);
        s_mtp[MS_P] = vc->mtp_P;
        s_mtp[MS_I] = 1;
        s_mtp_pl = &PL;
    }
    __syncthreads();
    int done = 0;
#if MTP_KVC
    while (cB.mtp_kvc && Rtot - done > 5) { // all but the last 5 rows: K / V only (at most 3 rows per pass)
        const int R = min(3, Rtot - done - 5);
        switch (R) {
            case 1: mtp_kv_pass<1>(done, epoch); break;
            case 2: mtp_kv_pass<2>(done, epoch); break;
            default: mtp_kv_pass<3>(done, epoch); break;
        }
        done += R;
    }
#endif
    while (done < Rtot) {
        const int R = min(5, Rtot - done); // (the MTP block's dp4a ffn_down holds at most 5 rows)
        const bool last_pass = done + R == Rtot;
        const bool want = lg && last_pass && ndraft > 0;
        switch (R) {
            case 1: mtp_pass<1>(done, 0, false, 0, want, epoch); break;
            case 2: mtp_pass<2>(done, 0, false, 0, want, epoch); break;
            case 3: mtp_pass<3>(done, 0, false, 0, want, epoch); break;
            case 4: mtp_pass<4>(done, 0, false, 0, want, epoch); break;
            case 5: mtp_pass<5>(done, 0, false, 0, want, epoch); break;
        }
        done += R;
    }
    // the draft passes i = 1 .. ndraft - 1 (none without logits), the loop's state in s_mtp (nothing held across the pass).
    // nd_stop is written before the PH_MTP_LMHEAD barrier, so every block takes the same exit. The M0 blocks read the
    // draft's token with nd_stop: one L2 round trip, not two
    for (;;) {
        const int i = s_mtp[MS_I];
        if (i >= s_mtp[MS_ND]) break;
        volatile Ctl * c = cB.ctl;
        const int stop = c->nd_stop;
        const int tok = ctaid_x() < (unsigned) MTP_M0_NB ? c->tok[i] : 0;
        if (stop) break;
        mtp_pass<1>(0, tok, true, i, true, epoch);
        if (threadIdx.x == 0) s_mtp[MS_I] = s_mtp[MS_I] + 1;
        __syncthreads();
    }
}

// ---- batched launches (MKEngine::slots_step): the rows of several lanes in one pass ----
// The weight phases run once over all rows (view 0's row buffers). The per-sequence phases run per lane in its main view
// (its rows from row 0): attention, its combine and the delta-net core as the lanes' items back to back (no grid barrier
// between lanes: a block done with lane i's items takes lane i + 1's), the h rows, the verify / control. Every phase is
// T-invariant per row, so each lane gets the arithmetic of a solo launch of its own T: bit-identical rows.
__device__ __forceinline__ void set_view(int v) {
    __syncthreads();
    if (threadIdx.x == 0) s_view = v;
    __syncthreads();
}
__device__ __forceinline__ void attn_core_n(int T, const LayerW & W, int slot, int n, int b0) {
    switch (T) {
        case 1: attn_core<1>(W, slot, n, smem, b0); break;
        case 2: attn_core<2>(W, slot, n, smem, b0); break;
        case 3: attn_core<3>(W, slot, n, smem, b0); break;
        case 4: attn_core<4>(W, slot, n, smem, b0); break;
        case 5: attn_core<5>(W, slot, n, smem, b0); break;
        default: attn_core<MAXT>(W, slot, n, smem, b0); break; // (an 8-row slot: rows past the lane's T are padding)
    }
}
__device__ __forceinline__ void attn_combine_n(int T, int n, int slot, int b0) {
    switch (T) {
        case 1: attn_combine<1>(n, 0, slot, b0); break;
        case 2: attn_combine<2>(n, 0, slot, b0); break;
        case 3: attn_combine<3>(n, 0, slot, b0); break;
        case 4: attn_combine<4>(n, 0, slot, b0); break;
        case 5: attn_combine<5>(n, 0, slot, b0); break;
        default: attn_combine<MAXT>(n, 0, slot, b0); break;
    }
}
__device__ __forceinline__ void dn_core_n(int T, const LayerW & W, int dl, int n, int dn_c, int h) {
    switch (T) {
        case 1: dn_core<1, false>(W, dl, n, dn_c, 1, smem, h); break;
        case 2: dn_core<2, false>(W, dl, n, dn_c, 1, smem, h); break;
        case 3: dn_core<3, false>(W, dl, n, dn_c, 1, smem, h); break;
        case 4: dn_core<4, false>(W, dl, n, dn_c, 1, smem, h); break;
        case 5: dn_core<5, false>(W, dl, n, dn_c, 1, smem, h); break;
        default: dn_core<MAXT, false>(W, dl, n, dn_c, 1, smem, h); break;
    }
}
__device__ __forceinline__ void write_h_rows_n(int T, int n) {
    switch (T) {
        case 1: write_h_rows<1>(cB.x, cB.qn, cW.onorm, cB.h_hist, n, true, smem); break;
        case 2: write_h_rows<2>(cB.x, cB.qn, cW.onorm, cB.h_hist, n, true, smem); break;
        case 3: write_h_rows<3>(cB.x, cB.qn, cW.onorm, cB.h_hist, n, true, smem); break;
        case 4: write_h_rows<4>(cB.x, cB.qn, cW.onorm, cB.h_hist, n, true, smem); break;
        case 5: write_h_rows<5>(cB.x, cB.qn, cW.onorm, cB.h_hist, n, true, smem); break;
        default: write_h_rows<MAXT>(cB.x, cB.qn, cW.onorm, cB.h_hist, n, true, smem); break;
    }
}

// a greedy row's token from its sampling candidate list (descending, insertion order among equals): the smallest id of
// the leading equal values, as the argmax's "larger value, then smaller id"
__device__ __forceinline__ int cand_argmax(const float2 * c) {
    const float v0 = c[0].x;
    int id = __float_as_int(c[0].y);
    for (int k = 1; k < TOPK_MAX && c[k].x == v0; k++) id = min(id, __float_as_int(c[k].y));
    return id;
}

// the batched passes' lm_head in LT-row passes over the head (each pass's last block leaves its rows in cB.cand: sampling,
// the candidates; greedy, the argmax id at .y of the row's first entry), a grid barrier after each
template <int LT, int RT>
__device__ __forceinline__ void lm_head_passes(bool smp, uint8_t * sm, int * s_tok, unsigned & epoch) {
    static_assert(RT % LT == 0, "whole lm_head passes");
#pragma unroll 1
    for (int g = 0; g < RT / LT; g++) {
        const bool last = smp ? lm_head<LT, true>(LT * g, false, sm, s_tok, cB.cand + LT * g * TOPK_MAX) : lm_head<LT>(LT * g, false, sm, s_tok);
        if (last && !smp && threadIdx.x < LT) cB.cand[(LT * g + threadIdx.x) * TOPK_MAX] = make_float2(0.f, __int_as_float(s_tok[threadIdx.x]));
        __threadfence();
        gsync(epoch, PH_LMHEAD);
    }
}
// the main pass of a batched launch: RT rows (the lanes' slots, then padding rows up to RT). RT > MAXT (MK_PAR_WIDE): the
// weight phases in K sub-chunk fills (the _ck phases), lm_head in 16-row passes (8 when RT is not a multiple of 16) with
// the candidates / argmax in cB.cand
template <int RT>
__device__ __noinline__ void main_forward_b(const Launch & P, unsigned & epoch) {
    uint8_t * const sm = smem;
    constexpr bool CK = RT > MAXT;
    embed_prologue<RT, true>(row_tab(P));
    gsync(epoch, PH_EMBED);
    int dl = 0;
    for (int L = 0; L < NL; L++) {
        const LayerW & W = cW.L[L];
        const float * wnext = (L + 1 < NL) ? cW.L[L + 1].attn_norm : cW.onorm;
        constexpr bool WIDE = RT > 5;
        if (W.attn) {
#if MK_PAR_WIDE
            if constexpr (CK) attn_proj_ck<RT>(W);
            else
#endif
                attn_proj_wv<RT>(WV_LAYER_ARG(W, L), L / 4, 0, false);
            gsync(epoch, PH_ATTN_PROJ);
            int ab0 = 0; // lane i's items start on the block after lane i - 1's last one (att_rank)
            for (int i = 0; i < P.nl; i++) {
                set_view(view_main(P.ln[i].lane));
                const int Ti = lane_slot(P.ln[i].T), ni = ((volatile Ctl *) cB.ctl)->n;
                attn_core_n(Ti, W, L / 4, ni, ab0);
                int nch, cs;
                attn_dims(ni + Ti, nch, cs); // (attn_core<Ti>'s partition: trunk slots have ws = 0)
                ab0 += NKV * nch;
            }
            set_view(0);
            gsync(epoch, PH_ATTN_CORE);
            ab0 = 0;
            for (int i = 0; i < P.nl; i++) {
                set_view(view_main(P.ln[i].lane));
                const int Ti = lane_slot(P.ln[i].T);
                attn_combine_n(Ti, ((volatile Ctl *) cB.ctl)->n, L / 4, ab0);
                ab0 += NH * Ti / 4; // (attn_combine: four items per block)
            }
            set_view(0);
            gsync(epoch, PH_ATTN_COMB);
#if MK_PAR_WIDE
            if constexpr (CK) resid_wv_ck<RT, false, VD>(W.o, cB.qo, W.post_norm, 0); // (out in the wv layout: host check)
            else
#endif
            if (W.o.fm) resid_out_wv<RT>(W.o, W.post_norm);
            else resid_dp<RT, 6>(W.o, cB.qo, cB.x, true, W.post_norm, sm);
            gsync(epoch, PH_RESID_O);
        } else {
#if MK_PAR_WIDE
            if constexpr (CK) dn_proj_ck<RT>(W, dl, row_tab(P));
            else
#endif
                dn_proj_wv<RT, true>(WV_LAYER_ARG(W, L), dl, 0, 0, row_tab(P));
            gsync(epoch, PH_DN_PROJ);
            for (int it = blockIdx.x; it < P.nl * NVH; it += gridDim.x) {
                const int i = it / NVH;
                set_view(view_main(P.ln[i].lane));
                volatile Ctl * c = cB.ctl;
                dn_core_n(lane_slot(P.ln[i].T), W, dl, c->n, c->dn_c, it - i * NVH);
            }
            set_view(0);
            gsync(epoch, PH_DN_CORE);
#if MK_PAR_WIDE
            if constexpr (CK) resid_wv_ck<RT, false, VD>(W.sout, cB.qo, W.post_norm, 0);
            else
#endif
            if (W.sout.fm) resid_out_wv<RT>(W.sout, W.post_norm);
            else resid_dp<RT, 6>(W.sout, cB.qo, cB.x, true, W.post_norm, sm);
            gsync(epoch, PH_RESID_SOUT);
            dl++;
        }
#if MK_PAR_WIDE
        if constexpr (CK) {
            ffn_gu_ck<RT>(W);
            resid_wv_ck<RT, true, FF>(W.fd, cB.qa, wnext, 3);
            gsync(epoch, PH_FFN_DOWN);
        } else
#endif
        {
            if (WIDE || ((W.fg.fm || W.fu.fm) && W.fd.fm)) {
                ffn_gu_wv<RT>(W, true);
                resid_wv<RT, true, FF>(W.fd, cB.qa, wnext, 3);
                gsync(epoch, PH_FFN_DOWN);
                continue;
            }
            if constexpr (WIDE) ffn_gu_wv<RT>(W);
            else if (W.fg.fm || W.fu.fm) ffn_gu_wv<RT>(W);
            else ffn_gu<RT>(W, sm);
            gsync(epoch, PH_FFN_GU);
            if constexpr (WIDE) resid_wv<RT, true, FF>(W.fd, cB.qa, wnext, 1);
            else if (W.fd.fm) resid_wv<RT, true, FF>(W.fd, cB.qa, wnext, 1);
            else resid_any<RT, 17>(W.fd, cB.qa, cB.x, true, wnext, sm);
            gsync(epoch, PH_FFN_DOWN);
        }
    }
    for (int i = 0; i < P.nl; i++) { // (block 0 writes them)
        set_view(view_main(P.ln[i].lane));
        write_h_rows_n(lane_slot(P.ln[i].T), ((volatile Ctl *) cB.ctl)->n);
    }
    set_view(0);

    int * s_tok = reinterpret_cast<int *>(misc(sm) + 400);
    bool smp = false;
    for (int i = 0; i < P.nl; i++) smp |= P.ln[i].temp > 0.f;
    if constexpr (CK) {
        if constexpr (RT % LM_RB == 0) lm_head_passes<LM_RB, RT>(smp, sm, s_tok, epoch); // (each pass over the whole head)
        else lm_head_passes<8, RT>(smp, sm, s_tok, epoch);
        if (blockIdx.x == 0 && threadIdx.x == 0) {
            const float2 * cand = cB.cand; // (view 0's rows)
            for (int i = 0; i < P.nl; i++) {
                const Launch Q = lane_launch(P, i);
                const int T = Q.T, r0 = P.ln[i].r0;
                s_view = view_main(P.ln[i].lane);
                int tg[MAXT];
                if (Q.temp > 0.f)
                    sample_targets_ol(Q.mode, Q.top_k, Q.temp, Q.top_p, Q.min_p, Q.seed, T, T, tg, cand + r0 * TOPK_MAX, cB.dq);
                else if (smp)
                    for (int t = 0; t < T; t++) tg[t] = cand_argmax(cand + (r0 + t) * TOPK_MAX);
                else
                    for (int t = 0; t < T; t++) tg[t] = __float_as_int(cand[(r0 + t) * TOPK_MAX].y);
                for (int t = 0; t < T; t++) cB.ctl->tgt[t] = tg[t];
                main_control(MODE_DECODE, 0, T, tg, false, true);
            }
            s_view = 0;
            __threadfence();
        }
        gsync(epoch, PH_LMHEAD);
    } else {
    // (sampling lanes: the candidates at SM_CAND; a greedy lane beside them takes each row's best candidate)
    const bool am_last = smp ? lm_head<RT, true>(0, false, sm, s_tok) : lm_head<RT>(0, false, sm, s_tok);
    if (am_last && threadIdx.x == 0) {
        const float2 * cand = reinterpret_cast<const float2 *>(sm + SM_CAND);
        for (int i = 0; i < P.nl; i++) {
            const Launch Q = lane_launch(P, i);
            const int T = Q.T, r0 = P.ln[i].r0;
            s_view = view_main(P.ln[i].lane);
            int tg[MAXT];
            if (Q.temp > 0.f)
                sample_targets_ol(Q.mode, Q.top_k, Q.temp, Q.top_p, Q.min_p, Q.seed, T, T, tg, cand + r0 * TOPK_MAX, cB.dq);
            else if (smp)
                for (int t = 0; t < T; t++) tg[t] = cand_argmax(cand + (r0 + t) * TOPK_MAX);
            else
                for (int t = 0; t < T; t++) tg[t] = s_tok[r0 + t];
            for (int t = 0; t < T; t++) cB.ctl->tgt[t] = tg[t];
            main_control(MODE_DECODE, 0, T, tg, false, true);
        }
        s_view = 0;
        __threadfence();
    }
    gsync(epoch, PH_LMHEAD);
    }
}

#if MK_MAXD > 8
// deep decode pass, T = MAXT + 1 .. MAXD as one MAXD-row pass (padding rows as main_forward's wide pass; bit-identical rows)
__device__ __noinline__ void main_forward_d(const Launch & P, unsigned & epoch) {
    constexpr int T = MAXD;
    uint8_t * const sm = smem;
    ml_begin(1); // (layer state in shared memory as main_forward's; ml_view(): this pass's view, v0)
    embed_prologue<T>();
    gsync(epoch, PH_EMBED);
    for (; ml_L() < NL; ml_next_L()) {
        if (ml_W().attn) {
            attn_proj_ck<T>(ml_W());
            gsync(epoch, PH_ATTN_PROJ);
#if MK_ATT_I8
            // (every trunk slot is all-q8 here, deep_why: the int8 form is attn_core's choice for them)
            if (cB.att_i8) attn_core_w48i<MAXT>(ml_W(), ml_L() / 4, ml_n());
            else
#endif
            attn_core_w48q<MAXT>(ml_W(), ml_L() / 4, ml_n());
            set_view(view_hi()); // rows MAXT .. at n + MAXT; they store K / V from n (identical bytes): no barrier between
            {
                // their items start on the block after the first call's last item (att_rank): at short contexts both
                // calls' few items run at once, on disjoint blocks
                const int n = ml_n();
                int nch1, cs1;
                attn_dims(n + MAXT, nch1, cs1);
#if MK_ATT_I8
                if (cB.att_i8) attn_core_w48i<MAXT>(ml_W(), ml_L() / 4, n + MAXT, n, NKV * nch1);
                else
#endif
                attn_core_w48q<MAXT>(ml_W(), ml_L() / 4, n + MAXT, n, NKV * nch1);
            }
            set_view(ml_view());
            gsync(epoch, PH_ATTN_CORE);
            attn_combine<MAXT>(ml_n(), 0, ml_L() / 4);
            set_view(view_hi());
            attn_combine<MAXT>(ml_n() + MAXT, 0, ml_L() / 4, NH * MAXT / 4); // (after the first call's NH MAXT / 4 block slots)
            set_view(ml_view());
            gsync(epoch, PH_ATTN_COMB);
            {
                const LayerW & W = ml_W();
                resid_wv_ck<T, false, VD>(W.o, cB.qo, W.post_norm, 0);
            }
            gsync(epoch, PH_RESID_O);
        } else {
            dn_proj_ck<T>(ml_W(), ml_dl(), RowTab{});
            k5_arrive(epoch, PH_DN_PROJ);
            dn_core<T>(ml_W(), ml_dl(), ml_n(), ml_dnc(), 1, sm);
            gsync(epoch, PH_DN_CORE);
            {
                const LayerW & W = ml_W();
                resid_wv_ck<T, false, VD>(W.sout, cB.qo, W.post_norm, 0);
            }
            gsync(epoch, PH_RESID_SOUT);
            ml_next_dl();
        }
        ffn_gu_ck<T>(ml_W());
        resid_wv_ck<T, true, FF>(ml_W().fd, cB.qa, ml_wnext(), 3);
        gsync(epoch, PH_FFN_DOWN);
    }
    write_h_rows_g<T>(cB.x, cB.qn, cW.onorm, cB.h_hist, ml_n());
    int * s_tok = reinterpret_cast<int *>(misc(sm) + 400);
    const bool smp = P.temp > 0.0f, pen = P.pres != 0.f || P.freq != 0.f;
    // one T-row lm_head pass (lm_head<16>), the rows' candidates / argmax in cB.cand
    const bool last = smp ? lm_head<T, true>(0, false, sm, s_tok, cB.cand, pen, P.pres, P.freq, 0)
                          : lm_head<T>(0, false, sm, s_tok, nullptr, pen, P.pres, P.freq, 0);
    if (last && !smp && threadIdx.x < T) cB.cand[threadIdx.x * TOPK_MAX] = make_float2(0.f, __int_as_float(s_tok[threadIdx.x]));
    __threadfence();
    gsync(epoch, PH_LMHEAD);
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        const int Tr = P.T;
        int tg[MAXD];
        if (smp) sample_targets_ol(P.mode, P.top_k, P.temp, P.top_p, P.min_p, P.seed, Tr, Tr, tg, cB.cand, cB.dq);
        else
            for (int t = 0; t < Tr; t++) tg[t] = __float_as_int(cB.cand[t * TOPK_MAX].y);
        for (int t = 0; t < Tr; t++) cB.ctl->tgt[t] = tg[t];
        main_control(MODE_DECODE, 0, Tr, tg, pen, true);
        __threadfence();
    }
    gsync(epoch, PH_LMHEAD);
}
#endif

template <int T>
__device__ void bench_phase(int which, int layer, uint8_t * /*sm*/, unsigned & epoch) {
    uint8_t * const sm = smem;
    for (int it = 0; it < 4; it++) {
        const int L = layer + 4 * it; // different layers: weights are not L2-resident, code is
        const LayerW & W = cW.L[L];
        if (which == 0) {
            if (W.fg.fm || W.fu.fm) ffn_gu_wv<T>(W);
            else ffn_gu<T>(W, sm);
        } else if (which == 1) {
            if (W.fd.fm) resid_wv<T, true, FF>(W.fd, cB.qa, W.attn_norm, 1);
            else resid_any<T, 17>(W.fd, cB.qa, cB.x, true, W.attn_norm, sm);
        } else if (which == 2) {
            if (!W.attn) {
                if (W.qkv.fm || W.gate.fm) dn_proj_wv<T>(WV_LAYER_ARG(W, L), 0, 0, 0);
                else dn_proj<T>(W, 0, 0, 0, sm);
            }
        }
        gsync(epoch, PH_FFN_GU);
    }
}

// n grid barriers of form V (gs_arrive_wait) with no work in between (bench mode, MK_GSB)
template <int V>
__device__ __noinline__ void gsync_bench(int n) {
    unsigned epoch = 0;
    for (int i = 0; i < n; i++) {
        __syncthreads();
        if (threadIdx.x == 0) {
            epoch += gridDim.x;
            gs_arrive_wait<V>(epoch);
        }
        __syncthreads();
    }
}

#if MTP_KVC
// ---- batched MTP (Launch::mtpb): the lanes' MTP passes shared, lane i at row i of view 0's row buffers ----
// view_mtp(i) = lane i's full view with the row buffers at row i: the per-lane parts (attention at the lane's position,
// its K / V rows, h, the draft head and its control) run there unchanged on one row; the GEMV phases run once over all
// rows in view 0. A lane's catch-up rows: all but the last K / V only (mtp_kv_pass's arithmetic, one round per row), the
// last one a full pass producing draft 1 (a solo full pass's last row: the same partition, the same K / V before it), then
// draft k for the lanes still drafting. Rows of lanes not in a pass are computed and dropped.
struct MtpRows {
    int nl;
    bool live[NSEQ];
    int lane[NSEQ]; // (K / V-only passes: the row's lane slot; full passes: row s = lane s)
    int pos[NSEQ], tok[NSEQ];
    const float * hsrc[NSEQ];
};
static_assert(NSEQ <= 5, "batched MTP: one row per lane in the MTP block's dp4a passes (<= 5 rows); constant memory (views)");
// M0 of mtp_pass for rows from several lanes (each row's token / position / h source from MtpRows)
template <int R>
__device__ __forceinline__ void mtp_m0_rows(const MtpRows & mr) {
    constexpr int M0_NB = MTP_M0_NB;
    if (blockIdx.x < R * M0_NB) {
        const int r = blockIdx.x / M0_NB, s = blockIdx.x - r * M0_NB;
        const int tok = mr.live[r] ? mr.tok[r] : 0;
        const float * hsrc = mr.live[r] ? mr.hsrc[r] : nullptr;
        mtp_m0_block(r, s, tok, hsrc);
    }
}

// round j of the K / V-only catch-up rows: row i = lane i's catch-up row j (when j < its mtp_R - 1)
template <int R>
__device__ __noinline__ void mtp_kv_pass_b(const MtpRows & mr, unsigned & epoch) {
    uint8_t * const sm = smem;
    const int lane = threadIdx.x & 31;
    mtp_m0_rows<R>(mr);
    gsync(epoch, PH_MTP_KV_M0);
    const LayerW & W = cW.L[NL];
    resid_any<R, 10>(cW.eh, cB.qe, cB.xm, false, W.attn_norm, sm);
    gsync(epoch, PH_MTP_KV_EH);
    {
        const SX sx = make_sx(sm, U_D);
        fill_xs<R>(cB.qn, 0, true, sx, misc(sm));
        const int nk = KVD / 2;
        gemv_pairs_any<R, 5>(W.k, 0, nk, PairOut{cB.kbuf, KVD, -1});
        gemv_pairs_any<R, 5>(W.v, nk, 2 * nk, PairOut{cB.vbuf, KVD, -1});
    }
    gsync(epoch, PH_MTP_KV_PROJ);
    constexpr int slot = N_ATT_SLOTS - 1;
    const float * const kb0 = cB.kbuf, * const vb0 = cB.vbuf; // (view 0's rows)
    for (int i = 0; i < R; i++) { // (row i into its lane's MTP slot, as mtp_kv_pass's store; no barrier after it)
        if (!mr.live[i]) continue;
        set_view(view_full(mr.lane[i]));
        const int pp = mr.pos[i];
        // (on the last blocks, as mtp_kv_pass's store: not on the next M0 phase's blocks)
        for (int g = (threadIdx.x >> 5) * gridDim.x + (gridDim.x - 1 - blockIdx.x); g < NKV; g += nwarps()) {
            float v[8];
            load8(kb0 + i * KVD + g * 256 + lane * 8, v);
            float ss = 0.f;
#pragma unroll
            for (int k = 0; k < 8; k++) ss += v[k] * v[k];
            const float rs = rsqrtf(wsum(ss) / 256.0f + EPS);
#pragma unroll
            for (int k = 0; k < 8; k++) v[k] *= rs * W.kn[lane * 8 + k];
            rope8(v, lane, pp);
            kv_store(0, slot, g, pp, v);
            load8(vb0 + i * KVD + g * 256 + lane * 8, v);
            kv_store(1, slot, g, pp, v);
        }
    }
    set_view(0);
}

// a full MTP pass over one row per lane (k = 0: the lanes' last catch-up rows, k >= 1: their draft-k rows) -> draft k + 1
template <int R>
__device__ __noinline__ void mtp_pass_b(const Launch & P, const MtpRows & mr, const int (&li)[NSEQ], int k, unsigned & epoch) {
    uint8_t * const sm = smem;
    mtp_pf_m0eh();
    mtp_m0_rows<R>(mr);
    gsync(epoch, PH_MTP_M0);
    const LayerW & W = cW.L[NL];
    resid_any<R, 10>(cW.eh, cB.qe, cB.xm, false, W.attn_norm, sm);
    mtp_pf_q(W);
    gsync(epoch, PH_MTP_EH);
    attn_proj<R>(W, 16, 0, sm); // (position: L2 warm-up only)
    gsync(epoch, PH_MTP_ATTN_PROJ);
    int ab0 = 0; // lane i's items start on the block after lane i - 1's last one (att_rank)
    for (int i = 0; i < R; i++) {
        if (!mr.live[i]) continue;
        set_view(view_mtp(i));
        attn_core<1>(W, 16, mr.pos[i], sm, ab0);
        int nch, cs;
        attn_dims_w(mr.pos[i] + 1, mtp_ws(N_ATT_SLOTS - 1, mr.pos[i]), nch, cs); // (attn_core<1>'s partition)
        ab0 += NKV * nch;
    }
    set_view(0);
    mtp_pf_resid(W.o, 12);
    gsync(epoch, PH_MTP_ATTN_CORE);
    ab0 = 0;
    for (int i = 0; i < R; i++) {
        if (!mr.live[i]) continue;
        set_view(view_mtp(i));
        attn_combine<1>(mr.pos[i], mtp_ws(N_ATT_SLOTS - 1, mr.pos[i]), N_ATT_SLOTS - 1, ab0);
        ab0 += NH / 4;
    }
    set_view(0);
    gsync(epoch, PH_MTP_ATTN_COMB);
    resid_any<R, 6>(W.o, cB.qo, cB.xm, true, W.post_norm, sm);
    mtp_pf_gu(W);
    gsync(epoch, PH_MTP_RESID_O);
    ffn_gu<R>(W, sm);
    mtp_pf_resid(W.fd, 12);
    gsync(epoch, PH_MTP_FFN_GU);
    resid_any<R, 17>(W.fd, cB.qa, cB.xm, true, cW.shnorm, sm);
    mtp_pf_head();
    gsync(epoch, PH_MTP_FFN_DOWN);
    for (int i = 0; i < R; i++) { // (block 0)
        if (!mr.live[i]) continue;
        set_view(view_mtp(i));
        write_h_rows<1>(cB.xm, cB.qn, cW.shnorm, cB.hm, 0, false, sm);
    }
    // the draft head of each lane on its row (mtp_pass's, with the lane's launch parameters), one grid pass each
    for (int i = 0; i < R; i++) {
        if (!mr.live[i]) continue;
        set_view(view_mtp(i));
        const Launch Q = lane_launch(P, li[i]);
        {
            int * s_tok = reinterpret_cast<int *>(misc(sm) + 400);
            // sampled drafts need the draft head's top logits (TOPK path); argmax drafts only the reduction
    #if MK_SPEC_LOG
            float2 * const tl = cB.dlog + k * (TOPK_MAX + 1);
            const bool lastb = lm_head<1, true, false, true>(0, true, sm, s_tok, tl);
    #else
            float2 * const tl = cB.dq + k * TOPK_MAX;
            // presence / frequency penalty: draft k + 1 has the committed tokens and drafts 1..k in its context
            const bool pen = Q.pres != 0.f || Q.freq != 0.f;
            const bool lastb = Q.spec_smp ? lm_head<1, true, false, true>(0, true, sm, s_tok, tl, pen, Q.pres, Q.freq, k)
                                          : lm_head<1, false, false, true>(0, true, sm, s_tok, nullptr, pen, Q.pres, Q.freq, k);
    #endif
            if (lastb && threadIdx.x == 0) {
                Ctl * c = cB.ctl;
                const float * mm = misc(sm);
                const float conf = __expf(mm[LM_BEST] - mm[LM_LSE]);
    #if MK_SPEC_LOG
                cB.dlog[k * (TOPK_MAX + 1) + TOPK_MAX] = make_float2(mm[LM_LSE], mm[LM_BEST]);
    #endif
                // (draft_sample reads lm_head's shared-memory copy of tl: the same values, without the L2 round trips)
                const int dt = Q.spec_smp ? draft_sample_ol(Q.top_k, Q.temp, Q.top_p, Q.min_p, Q.seed,
                                                            reinterpret_cast<const float2 *>(sm + SM_CAND),
                                                            cB.dq + k * TOPK_MAX, (unsigned long long) c->mtp_P + 1 + k)
                                          : s_tok[0];
                c->tok[1 + k] = dt;
                s_tok[1] = dt; // (for mtp_pf_embd)
                // gate_chain 2 (policy chain=2) with sampled drafts: the chain runs over the drafted token's own probability
                // under q' (the distribution it was drawn from) instead of the draft head's top-1 probability
                const float chain = (k == 0 ? 1.f : c->dchain) * (Q.gate_chain == 2 && Q.spec_smp ? dq_prob(cB.dq + k * TOPK_MAX, dt) : conf);
                const float f = Q.gate_chain ? chain : conf;
                c->dsmp = Q.spec_smp;
                c->dconf[k] = conf;
                c->dchain = chain;
                // (the gates are the launch's, lane_launch keeps them: read from P, as a dynamic index into Q would keep
                // the whole copy in local memory)
                c->nd_prod = f < P.gate_drop[k] ? k : k + 1;
                c->nd_stop = f < P.gate_stop[k] || f < P.gate_drop[k];
                __threadfence();
            }
            if (lastb) { // (as mtp_pass's)
                __syncthreads();
                if (threadIdx.x >> 5 == 1) mtp_pf_embd(s_tok[1]);
            }
        }
        bool more = false; // (another lane's draft head pass follows: its first rows)
        for (int q = i + 1; q < R; q++) more = more || mr.live[q];
        if (more) mtp_pf_head();
        set_view(0);
        gsync(epoch, PH_MTP_LMHEAD);
    }
    set_view(0);
}

__device__ __noinline__ void mtp_batch(const Launch & P, unsigned & epoch) {
    // rows by lane slot (view_mtp(s): row s); every block reads the same lane ctls after a barrier: the same rows everywhere
    int li[NSEQ];
#pragma unroll
    for (int s = 0; s < NSEQ; s++) li[s] = -1;
    for (int i = 0; i < P.nl; i++) li[P.ln[i].lane] = i;
    int nkv[NSEQ], maxnd = 0, tot = 0;
    for (int s = 0; s < NSEQ; s++) {
        nkv[s] = 0;
        if (li[s] < 0) continue;
        const volatile Ctl * c = cBs[view_full(s)].ctl;
        if (P.ln[li[s]].nd <= 0 || !c->mtp_logits) continue;
        nkv[s] = max(0, c->mtp_R - 1);
        tot += nkv[s];
        maxnd = max(maxnd, P.ln[li[s]].nd);
    }
    MtpRows mr;
    mr.nl = NSEQ;
    // the K / V-only catch-up rows of all lanes (each lane's rows 0 .. mtp_R - 2), NSEQ rows per pass in lane order
    int ks = 0, kj = 0; // next (lane slot, row)
    for (int done = 0; done < tot; done += NSEQ) {
        for (int r = 0; r < NSEQ; r++) {
            while (ks < NSEQ && kj >= nkv[ks]) {
                ks++;
                kj = 0;
            }
            mr.live[r] = ks < NSEQ;
            if (!mr.live[r]) {
                mr.lane[r] = 0;
                mr.pos[r] = 0;
                mr.tok[r] = 0;
                mr.hsrc[r] = nullptr;
                continue;
            }
            const Bufs & V = cBs[view_full(ks)];
            const volatile Ctl * c = V.ctl;
            mr.lane[r] = ks;
            mr.pos[r] = c->mtp_pos[kj];
            mr.tok[r] = c->mtp_tok[kj];
            mr.hsrc[r] = mr.pos[r] >= 1 ? V.h_hist + (size_t) ((mr.pos[r] - 1) % H_RING) * D : nullptr;
            kj++;
        }
        mtp_kv_pass_b<NSEQ>(mr, epoch);
    }
    for (int k = 0; k < maxnd; k++) {
        bool any = false;
        for (int s = 0; s < NSEQ; s++) {
            const Bufs & V = cBs[view_full(s)];
            const volatile Ctl * c = V.ctl;
            bool live = li[s] >= 0 && P.ln[li[s]].nd > k && c->mtp_logits;
            if (k > 0) live = live && !c->nd_stop;
            mr.live[s] = live;
            mr.lane[s] = s;
            any |= live;
            if (!live) {
                mr.pos[s] = 0;
                mr.tok[s] = 0;
                mr.hsrc[s] = nullptr;
            } else if (k == 0) {
                const int last = c->mtp_R - 1;
                mr.pos[s] = c->mtp_pos[last];
                mr.tok[s] = c->mtp_tok[last];
                mr.hsrc[s] = mr.pos[s] >= 1 ? V.h_hist + (size_t) ((mr.pos[s] - 1) % H_RING) * D : nullptr;
            } else {
                mr.pos[s] = c->mtp_P + k;
                mr.tok[s] = c->tok[k];
                mr.hsrc[s] = V.hm;
            }
        }
        if (!any) break;
        mtp_pass_b<NSEQ>(P, mr, li, k, epoch);
    }
}
#endif

// Egress: block 0 copies the launch's Ctl to a mapped host mirror after the launch's last grid barrier (step() reads the
// mirror instead of a device-to-host copy), the barrier counts run on across launches (each launch starts from the counts
// the previous one left, kept in cB.bar words; wrap-safe compares) so launch() zeroes them only once, and dv_add uploads
// asynchronously from pinned staging on the launch stream (fewer copy-engine switches per launch, which the native
// Windows build pays most for).
// mk_egress, after the launch's last grid barrier (every block has passed it, and with it every write to ctl): block 0
// publishes the counts the next launch starts from (BAR_BASE, C1_BASE) and copies ctl to its mapped host mirror (the host
// reads that after the launch instead of a device-to-host copy)
__device__ __forceinline__ void mk_egress() {
    if (blockIdx.x != 0) return;
    if (threadIdx.x == 0) {
        cB.bar[BAR_BASE] = s_bar0 + s_bar_k * gridDim.x; // (s_bar_k: this launch's grid barriers)
        cB.bar[C1_BASE] = s_c1b + s_c1n * C1_NJC; // (block 0 is in chunk 0 and passes every c1_wait: s_c1n = this launch's split edges)
    }
    if (cB.ctl_eg && threadIdx.x < sizeof(Ctl) / 4) {
        reinterpret_cast<volatile int *>(cB.ctl_eg)[threadIdx.x] = reinterpret_cast<volatile const int *>(cB.ctl)[threadIdx.x];
        __threadfence_system();
    }
}

// a batched launch: the main pass, each lane's MTP passes (its full view: rows from 0), then mk_egress per lane
__device__ __noinline__ void run_batch(const Launch & P, unsigned & epoch) {
    switch (P.T) {
        case 2: main_forward_b<2>(P, epoch); break;
        case 3: main_forward_b<3>(P, epoch); break;
        case 4: main_forward_b<4>(P, epoch); break;
        case 5: main_forward_b<5>(P, epoch); break;
#if MK_PAR_WIDE
        case RMAX: main_forward_b<RMAX>(P, epoch); break;
#endif
        default: main_forward_b<MAXT>(P, epoch); break;
    }
#if MTP_KVC
    if (P.mtpb) {
        mtp_batch(P, epoch);
    } else
#endif
    {
        for (int i = 0; i < P.nl; i++) {
            if (P.ln[i].nd == 0) continue;
            set_view(view_full(P.ln[i].lane));
            mtp_tail(P, i, epoch);
        }
        set_view(0);
    }
    mk_egress(); // (view 0: the barrier counts; its ctl mirror is not the lanes')
    if (blockIdx.x != 0) return;
    for (int i = 0; i < P.nl; i++) {
        const Bufs & V = cBs[view_main(P.ln[i].lane)];
        if (V.ctl_eg && threadIdx.x < sizeof(Ctl) / 4) {
            reinterpret_cast<volatile int *>(V.ctl_eg)[threadIdx.x] = reinterpret_cast<volatile const int *>(V.ctl)[threadIdx.x];
            __threadfence_system();
        }
    }
}

// P is __grid_constant__: the out-of-line callees take it by reference, and a plain parameter whose address is taken is
// copied to local memory (320 B of the stack frame, which the driver reserves for every thread slot of the GPU)
__global__ void __launch_bounds__(NT, 1) mk_cycle(const __grid_constant__ Launch P) { MK_VIEW_INIT();
    uint8_t * const sm = smem;
    unsigned epoch = 0; // (unused: the barrier count is s_bar_k)
    if (threadIdx.x == 0) { // (0 after the host's memset: the first launch, bench launches)
        s_bar0 = __ldcg(cB.bar + BAR_BASE);
        s_c1b = __ldcg(cB.bar + C1_BASE);
        s_bar_k = 0;
    }
    if (P.mode == MODE_BENCH) {
        if (P.last == 3) { // gsync bench: P.ndraft barriers of form P.score
            switch (P.score) {
                case 1: gsync_bench<1>(P.ndraft); break;
                case 2: gsync_bench<2>(P.ndraft); break;
                case 3: gsync_bench<3>(P.ndraft); break;
                default: gsync_bench<0>(P.ndraft); break;
            }
            return;
        }
        switch (P.T) {
            case 1: bench_phase<1>(P.last, P.ndraft, sm, epoch); break;
            case 4: bench_phase<4>(P.last, P.ndraft, sm, epoch); break;
        }
        return;
    }
    if (cB.prof && blockIdx.x == 0 && threadIdx.x == 0) cB.prof[0] = gtimer() << 8;
    if (threadIdx.x == 0) s_c1n = 0; // (C1 split edges passed: c1_wait's target)
    if (P.nl > 0) {
        run_batch(P, epoch);
        return;
    }
    switch (P.T) {
        case 1: run_main<1>(P, sm, epoch); break;
        case 2: run_main<2>(P, sm, epoch); break;
        case 3: run_main<3>(P, sm, epoch); break;
        case 4: run_main<4>(P, sm, epoch); break;
        case 5: run_main<5>(P, sm, epoch); break;
        default:
#if MK_MAXD > 8
            if (P.T > MAXT) {
                main_forward_d(P, epoch); // T = MAXT + 1 .. MAXD
                break;
            }
#endif
            run_wide(P, epoch); // T = 6 .. MAXT
            break;
    }
    if (P.ndraft == 0 || P.score) {
        mk_egress();
        return;
    }
    mtp_tail(P, -1, epoch);
    mk_egress();
}

#include "prefill.cuh"

// MK_FAULT (tests of the error path only, see Impl::fault_*): a kernel that traps, i.e. a sticky error
__global__ void pk_fault() { __trap(); }

// silu(gate) * up from the fp32 outputs of two GEMMs, written like mmh<.., true>'s epilogue (fp16,
// hperm order): the gate / up pairs it cannot fuse (different types, or Q6_K / Q8_0).
__global__ void __launch_bounds__(256) pk_glu(const float * __restrict__ g, const float * __restrict__ u, __half * __restrict__ out, int ld) { MK_VIEW_INIT();
    // thread = 4 consecutive values (one float4 of g and u: a warp reads 512 contiguous bytes of each); within a
    // 16-group, values 4 j .. 4 j + 3 go to hperm slots 2 j, 2 j + 1 and 8 + 2 j, 9 + 2 j (two 4-byte stores)
    const size_t t = blockIdx.x;
    const float4 * g4 = reinterpret_cast<const float4 *>(g + t * ld);
    const float4 * u4 = reinterpret_cast<const float4 *>(u + t * ld);
    __half * o = out + t * ld;
    for (int k = threadIdx.x; k < ld / 4; k += blockDim.x) {
        const float4 a = g4[k], b = u4[k];
        const float gv[4] = {a.x, a.y, a.z, a.w}, uv[4] = {b.x, b.y, b.z, b.w};
        __half hv[4];
#pragma unroll
        for (int e = 0; e < 4; e++) {
            const float v = gv[e] / (1.0f + expf(-gv[e])) * uv[e];
#if MK_PF_INFCHK
            pf::chk_clamp(v, pf::CHK_IN_GLU);
#endif
            hv[e] = __float2half_rn(fminf(fmaxf(v, -65504.f), 65504.f));
        }
        __half * og = o + 16 * (k >> 2) + 2 * (k & 3);
        *reinterpret_cast<__half2 *>(og) = __halves2half2(hv[0], hv[1]);
        *reinterpret_cast<__half2 *>(og + 8) = __halves2half2(hv[2], hv[3]);
    }
}

// state files: after a restore the GPU has been idle through the load (clocks down); this keeps every SM busy for ns
// nanoseconds so the decode that follows starts at the clocks a decode after a fresh prefill sees (touches no memory)
__global__ void st_warm(unsigned long long ns) { MK_VIEW_INIT();
    unsigned long long t0, t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
    do {
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    } while (t - t0 < ns);
}

// keep-warm load (MKEngine::warm_launch; the server's idle windows, capi.cpp): keeps the GPU's clocks up while touching
// no engine memory. Ends once the host-mapped word *stop is non-zero (thread 0 of each block polls it) or after ns
// nanoseconds. spin 1: every thread busy-loops on the timer; 0: thread 0 alone waits, __nanosleep between polls.
__global__ void kw_warm(const unsigned * stop, unsigned long long ns, int spin) { MK_VIEW_INIT();
    __shared__ volatile int s_stop;
    if (threadIdx.x == 0) s_stop = 0;
    __syncthreads();
    if (!spin && threadIdx.x) return;
    unsigned long long t0, t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
    for (unsigned i = 0;; i++) {
        if (threadIdx.x == 0 && (!spin || (i & 31) == 0)) {
            unsigned v;
            asm volatile("ld.relaxed.sys.global.u32 %0, [%1];" : "=r"(v) : "l"(stop) : "memory");
            if (v) s_stop = 1;
        }
        if (s_stop) break;
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
        if (t - t0 >= ns) break;
        if (!spin) __nanosleep(1000);
    }
}

#include "wv_dev.cuh"

} // namespace

// mk_launch.h's definitions (the kernels and symbols above are in the anonymous namespace: the host TU reaches them here)
namespace mkl {

unsigned long long abi_sig() { return mk_abi_sig(); }
extern const void * const k_mk_cycle = (const void *) ::mk_cycle;

cudaError_t set_cW(const void * src, size_t count, size_t offset) { return cudaMemcpyToSymbol(::cW, src, count, offset); }
cudaError_t set_cW_async(const void * src, size_t count, size_t offset, cudaMemcpyKind kind, cudaStream_t st) {
    return cudaMemcpyToSymbolAsync(::cW, src, count, offset, kind, st);
}
cudaError_t get_cW(void * dst, size_t count, size_t offset) { return cudaMemcpyFromSymbol(dst, ::cW, count, offset); }
cudaError_t set_cBs(const void * src, size_t count, size_t offset) { return cudaMemcpyToSymbol(::cBs, src, count, offset); }
cudaError_t set_cBs_async(const void * src, size_t count, size_t offset, cudaMemcpyKind kind, cudaStream_t st) {
    return cudaMemcpyToSymbolAsync(::cBs, src, count, offset, kind, st);
}
cudaError_t set_cView(const void * src, size_t count, size_t offset) { return cudaMemcpyToSymbol(::cView, src, count, offset); }
#if MK_L2_TUNE
cudaError_t set_cL2(const void * src, size_t count, size_t offset) { return cudaMemcpyToSymbol(::cL2, src, count, offset); }
#endif
// MK_FFNA (wv.cuh WvDeal): the wv phase tops' per-block dealing for nsm blocks with the old prologues' integer formulas,
// then k_deal_check compares every entry with the device formulas (a difference: a message and cudaErrorInvalidValue)
cudaError_t wv_deal_init(int nsm) {
#if MK_FFNA & 9
    if (nsm < 1 || nsm > MK_MAX_GRID) return cudaErrorInvalidValue;
    WvDeal h[MK_MAX_GRID] = {};
    const unsigned nb = (unsigned) nsm, NJ = FF / 16, NTILE = D / 16, NTL = QKV / 16 + VD / 16, NQ = QKV / 16, NAB = AB_SPLIT * 2 * NVH,
                   NJC = C1_NJC;
    const auto pk = [](unsigned t0, unsigned t1) { return t0 | t1 << 16; };
    for (unsigned b = 0; b < nb; b++) {
        const unsigned kc = (WV_S * b + WV_S - 1) / nb, c0 = nb * kc / WV_S, G = nb * (kc + 1) / WV_S - c0, bi = b - c0;
        h[b].c1 = pk(NJC * kc + NJC * bi / G, NJC * kc + NJC * (bi + 1) / G);
        h[b].ff = pk(NJ * b / nb, NJ * (b + 1) / nb);
        h[b].rt = pk(NTILE * bi / G, NTILE * (bi + 1) / G);
        h[b].dn = pk(NTL * b / nb, NTL * (b + 1) / nb);
        h[b].qk = pk(NQ * b / nb, NQ * (b + 1) / nb);
        h[b].ab = pk(NAB * b / nb, NAB * (b + 1) / nb);
        h[b].kc = (int) kc;
    }
    const int zero = 0;
    int bad = 0;
    cudaError_t e = cudaMemcpyToSymbol(::cDeal, h, sizeof h);
    if (e == cudaSuccess) e = cudaMemcpyToSymbol(::g_deal_bad, &zero, sizeof zero);
    if (e == cudaSuccess) {
        ::k_deal_check<<<nsm, 32>>>();
        e = cudaGetLastError();
    }
    if (e == cudaSuccess) e = cudaMemcpyFromSymbol(&bad, ::g_deal_bad, sizeof bad);
    if (e != cudaSuccess) return e;
    if (bad) {
        fprintf(stderr, "[mk] MK_FFNA: %d of %d blocks' dealing table entries differ from the device formulas\n", bad, nsm);
        return cudaErrorInvalidValue;
    }
    fprintf(stderr, "[mk] MK_FFNA: wv phase-top dealing table, %d blocks (checked)\n", nsm);
    return cudaSuccess;
#else
    (void) nsm;
    return cudaSuccess;
#endif
}
cudaError_t set_cP(const void * src, size_t count, size_t offset) { return cudaMemcpyToSymbol(::cP, src, count, offset); }
cudaError_t set_g_iq3s_grid(const void * src, size_t count, size_t offset) {
    return cudaMemcpyToSymbol(::g_iq3s_grid, src, count, offset);
}
#if MK_PF_INFCHK
cudaError_t set_chk_layer_async(const void * src, size_t count, size_t offset, cudaMemcpyKind kind, cudaStream_t st) {
    return cudaMemcpyToSymbolAsync(::pf::chk_layer, src, count, offset, kind, st);
}
cudaError_t set_chk_site_async(const void * src, size_t count, size_t offset, cudaMemcpyKind kind, cudaStream_t st) {
    return cudaMemcpyToSymbolAsync(::pf::chk_site, src, count, offset, kind, st);
}
cudaError_t set_chk_cnt(const void * src, size_t count, size_t offset) {
    return cudaMemcpyToSymbol(::pf::chk_cnt, src, count, offset);
}
cudaError_t get_chk_cnt(void * dst, size_t count, size_t offset) {
    return cudaMemcpyFromSymbol(dst, ::pf::chk_cnt, count, offset);
}
#endif

void k_ring_refill(int p0, int p1) {
    ::k_ring_refill<<<std::min(1024, (p1 - p0) * NKV * N_ATT_SLOTS * 2 / 8 + 1), 256>>>(p0, p1);
}
#if MK_KV_EMU
void k_kv_demote(int p0, int p1) { ::k_kv_demote<<<256, 256>>>(p0, p1); }
#endif
void pk_fault() { ::pk_fault<<<1, 1>>>(); }
void st_warm(dim3 grid, unsigned long long ns) { ::st_warm<<<grid, 256>>>(ns); }
void kw_warm(dim3 grid, dim3 block, const unsigned * stop, unsigned long long ns, int spin, cudaStream_t st) {
    ::kw_warm<<<grid, block, 0, st>>>(stop, ns, spin);
}
cudaError_t wv_dev_launch(int type, int N, int K, int rb, const uint8_t * sw, const uint8_t * sm, size_t meta_n, uint8_t * out,
                          uint8_t * out_meta, cudaStream_t st, const uint32_t * dtab, int W, int G, int seg) {
    const int nb = K / 256, TPB = 256;
    auto grid = [](int64_t n) { return (unsigned) std::max<int64_t>(1, std::min<int64_t>(4096, (n + 255) / 256)); };
    if (type == T_Q6K) {
        ::wv_q6f_dev<<<grid((int64_t) (N / 16) * nb * 16), TPB, 0, st>>>(sw, sm, out, N, nb, rb);
        return cudaGetLastError();
    }
    if (wv_fm(type)) {
        const int ngrp = nb / G;
        const int64_t total = (int64_t) (N / 16) * ngrp * W;
        ::wv_fm_dev<<<grid(total), TPB, 0, st>>>(reinterpret_cast<const uint64_t *>(sw), reinterpret_cast<uint64_t *>(out), dtab, W, ngrp,
                                                 rb / 8, seg / 8, total);
        return cudaGetLastError();
    }
    const int BS = type == T_Q8_0 ? 256 : type == T_Q5K ? 176 : type == T_Q6K ? Q6_BLK : type == T_Q3K || type == T_IQ3S ? Q3_BLK : 144;
    const int MB = type == T_Q6K || type == T_Q3K ? 2 * Q6_META : 16;
    const int64_t tw = (int64_t) N * rb / 4;
    ::wv_tm_dev<<<grid(tw), TPB, 0, st>>>(reinterpret_cast<const uint32_t *>(sw), reinterpret_cast<uint32_t *>(out), rb / 4, BS / 4, rb / BS, tw);
    if (meta_n) {
        const size_t mrow = meta_n / N;
        const int64_t tm = (int64_t) meta_n / 4;
        ::wv_tm_dev<<<grid(tm), TPB, 0, st>>>(reinterpret_cast<const uint32_t *>(sm), reinterpret_cast<uint32_t *>(out_meta), (int64_t) mrow / 4, MB / 4,
                                              (int) (mrow / MB), tm);
    }
    return cudaGetLastError();
}

template <int ET> void pk_embed(int T) { ::pk_embed<ET><<<T, 256>>>(T); }
template <int ET> void pk_embed_img(int T) { ::pk_embed_img<ET><<<T, 256>>>(T); }
template void pk_embed<T_Q6K>(int);
template void pk_embed<T_Q8_0>(int);
template void pk_embed<T_Q4K>(int);
template void pk_embed_img<T_Q6K>(int);
template void pk_embed_img<T_Q8_0>(int);
template void pk_embed_img<T_Q4K>(int);
void pk_norm_quant(dim3 grid, const float * src, const float * w) { ::pk_norm_quant<<<grid, 256>>>(src, w); }
void pk_norm_quant1(dim3 grid, const float * src, const float * w) { ::pk_norm_quant1<<<grid, 256>>>(src, w); }
void pk_ab(const float * Wab, const int8_t * abq, const __half * abd, const float * fn, int T, cudaStream_t st) {
    ::pk_ab<<<dim3((T + 63) / 64, AB_SPLITS), 256, 0, st>>>(Wab, abq, abd, fn, T);
}
void pk_ab2(const float * Wab, const int8_t * abq, const __half * abd, const float * fn, int T, cudaStream_t st) {
    const dim3 grid((T + 63) / 64, AB_SPLITS);
    if (abq) ::pk_ab2<true><<<grid, 128, 0, st>>>(Wab, abq, abd, fn, T);
    else ::pk_ab2<false><<<grid, 128, 0, st>>>(Wab, abq, abd, fn, T);
}
void pk_dn_prep(int T, int n0, int dl, const float * convw, const float * dt, const float * A) {
    ::pk_dn_prep<<<T, 256>>>(T, n0, dl, convw, dt, A);
}
template <int NT> void pk_dn_prep_t(int T, int n0, int dl, const float * convw, const float * dt, const float * A) {
    ::pk_dn_prep_t<NT><<<(T + NT - 1) / NT, 256>>>(T, n0, dl, convw, dt, A);
}
template void pk_dn_prep_t<4>(int, int, int, const float *, const float *, const float *);
template void pk_dn_prep_t<8>(int, int, int, const float *, const float *, const float *);
void pk_conv_ring(int T, int n0, int dl, bool old) {
    if (old) ::pk_conv_ring<false><<<std::min(T, CONV_RING), 256>>>(T, n0, dl); // MK_PFX bit 128
    else ::pk_conv_ring<true><<<std::min(T, CONV_RING), 256>>>(T, n0, dl);
}
void pk_ck_conv(int n0, int P, int dl, float * dst) { ::pk_ck_conv<<<CONV_RING, 256>>>(n0, P, dl, dst); }
void pk_ck_hist(dim3 grid, int lo, int n0, int from_ring, float * dst) { ::pk_ck_hist<<<grid, 256>>>(lo, n0, from_ring, dst); }
void pk_dn_scan(int T, int t0, const float * st_in, float * st_out) { ::pk_dn_scan<<<dim3(NVH, 2), 256>>>(T, t0, st_in, st_out); }
void pk_dn_tri(int T, __half * tri) { ::pk_dn_tri<<<dim3(NKH, (T + DC - 1) / DC), 256>>>(T, tri); }
void pk_dn_scan4(int T, int t0, const float * st_in, float * st_out, const __half * tri) {
    static const bool carve = [] { // three blocks of ~27 KB static shared memory per SM
        CK(cudaFuncSetAttribute(::pk_dn_scan4, cudaFuncAttributePreferredSharedMemoryCarveout, (int) cudaSharedmemCarveoutMaxShared));
        return true;
    }();
    (void) carve;
    ::pk_dn_scan4<<<dim3(NVH, 4), 128>>>(T, t0, st_in, st_out, tri);
}
void pk_dn_out(dim3 grid, const float * snorm) { ::pk_dn_out<<<grid, 256>>>(snorm); }
void pk_attn_prep(dim3 grid, int n0, int slot, const float * qn, const float * kn, int do_q, int r0) {
    ::pk_attn_prep<<<grid, 256>>>(n0, slot, qn, kn, do_q, r0);
}
void pk_attn_out(dim3 grid, int slot) { ::pk_attn_out<<<grid, 256>>>(slot); }
void pk_final_h(int T, int n0) { ::pk_final_h<<<T, 256>>>(T, n0); }
template <bool IMG, int ET> void pk_mtp_in(dim3 grid) { ::pk_mtp_in<IMG, ET><<<grid, 256>>>(); }
template void pk_mtp_in<false, T_Q6K>(dim3);
template void pk_mtp_in<false, T_Q8_0>(dim3);
template void pk_mtp_in<false, T_Q4K>(dim3);
template void pk_mtp_in<true, T_Q6K>(dim3);
template void pk_mtp_in<true, T_Q8_0>(dim3);
template void pk_mtp_in<true, T_Q4K>(dim3);
void pk_argmax(dim3 grid, int t0) { ::pk_argmax<<<grid, 256>>>(t0); }
void pk_glu(dim3 grid, const float * g, const float * u, __half * out, int ld) { ::pk_glu<<<grid, 256>>>(g, u, out, ld); }

template <bool Q8> void pk_attn(int T, int n0, int slot) {
    ::pk_attn<Q8><<<dim3((T + AT_Q - 1) / AT_Q, NKV), 192, AT_SMEM>>>(T, n0, slot);
}
template void pk_attn<false>(int, int, int);
template void pk_attn<true>(int, int, int);
template <bool Q8, bool PV16> void pk_attn2(dim3 grid, int T, int n0, int slot, int nfull, int S, float * part, int r0) {
    ::pk_attn2<Q8, PV16><<<grid, A2_NT, A2_SMEM>>>(T, n0, slot, nfull, S, part, r0);
}
template void pk_attn2<false, false>(dim3, int, int, int, int, int, float *, int);
template void pk_attn2<false, true>(dim3, int, int, int, int, int, float *, int);
template void pk_attn2<true, false>(dim3, int, int, int, int, int, float *, int);
template void pk_attn2<true, true>(dim3, int, int, int, int, int, float *, int);
void pk_attn2_comb(dim3 grid, int T, int nfull, int S, const float * part) { ::pk_attn2_comb<<<grid, 256>>>(T, nfull, S, part); }
// MK_PFX bit 64: pk_attn8, else pk_attn9 (the same outputs). VB = 8: pk_attn9's all-q8 form (pk_attn8 reads the slot's bits at run time).
int pk_attn_old = 0;
template <bool QROW, bool KLOW, bool EXA, int KB, int VB>
void pk_attn8(dim3 grid, int T, int n0, int slot, int nfull, int S, float * part, int r0, int segt, int exh, float * part2,
              int exa) {
    if (pk_attn_old) ::pk_attn8<QROW, KLOW, EXA, KB, (VB == 8 ? 0 : VB)><<<grid, A2_NT, A8_SMEM>>>(T, n0, slot, nfull, S, part, r0, segt, exh, part2, exa);
    else ::pk_attn9<QROW, KLOW, EXA, KB, VB><<<grid, A2_NT, A8_SMEM>>>(T, n0, slot, nfull, S, part, r0, segt, exh, part2, exa);
}
template void pk_attn8<false, false>(dim3, int, int, int, int, int, float *, int, int, int, float *, int);
template void pk_attn8<true, false>(dim3, int, int, int, int, int, float *, int, int, int, float *, int);
template void pk_attn8<false, true>(dim3, int, int, int, int, int, float *, int, int, int, float *, int);
template void pk_attn8<true, true>(dim3, int, int, int, int, int, float *, int, int, int, float *, int);
template void pk_attn8<false, true, false, 6, 5>(dim3, int, int, int, int, int, float *, int, int, int, float *, int);
template void pk_attn8<false, true, false, 5, 5>(dim3, int, int, int, int, int, float *, int, int, int, float *, int);
template void pk_attn8<false, false, true>(dim3, int, int, int, int, int, float *, int, int, int, float *, int);
template void pk_attn8<false, true, true>(dim3, int, int, int, int, int, float *, int, int, int, float *, int);
template void pk_attn8<false, true, true, 6, 5>(dim3, int, int, int, int, int, float *, int, int, int, float *, int);
template void pk_attn8<false, true, true, 5, 5>(dim3, int, int, int, int, int, float *, int, int, int, float *, int);
template void pk_attn8<false, false, false, 0, 8>(dim3, int, int, int, int, int, float *, int, int, int, float *, int);
template void pk_attn8<false, false, true, 0, 8>(dim3, int, int, int, int, int, float *, int, int, int, float *, int);
void pk_attn8_combx(dim3 grid, int T, int n0, int nfull, int S, int segt, int exh, int r0, const float * part, const float * part2,
                    int exa) {
    ::pk_attn8_combx<<<grid, 256>>>(T, n0, nfull, S, segt, exh, r0, part, part2, exa);
}
void pk_attn_attr() {
    CK(cudaFuncSetAttribute(::pk_attn<false>, cudaFuncAttributeMaxDynamicSharedMemorySize, AT_SMEM));
    CK(cudaFuncSetAttribute(::pk_attn<true>, cudaFuncAttributeMaxDynamicSharedMemorySize, AT_SMEM));
    CK(cudaFuncSetAttribute(::pk_attn2<false, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, A2_SMEM));
    CK(cudaFuncSetAttribute(::pk_attn2<false, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, A2_SMEM));
    CK(cudaFuncSetAttribute(::pk_attn2<true, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, A2_SMEM));
    CK(cudaFuncSetAttribute(::pk_attn2<true, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, A2_SMEM));
    CK(cudaFuncSetAttribute(::pk_attn8<false, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, A8_SMEM));
    CK(cudaFuncSetAttribute(::pk_attn8<true, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, A8_SMEM));
    CK(cudaFuncSetAttribute(::pk_attn8<false, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, A8_SMEM));
    CK(cudaFuncSetAttribute(::pk_attn8<true, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, A8_SMEM));
    CK(cudaFuncSetAttribute(::pk_attn8<false, true, false, 6, 5>, cudaFuncAttributeMaxDynamicSharedMemorySize, A8_SMEM));
    CK(cudaFuncSetAttribute(::pk_attn8<false, true, false, 5, 5>, cudaFuncAttributeMaxDynamicSharedMemorySize, A8_SMEM));
    CK(cudaFuncSetAttribute(::pk_attn9<false, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, A8_SMEM));
    CK(cudaFuncSetAttribute(::pk_attn9<true, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, A8_SMEM));
    CK(cudaFuncSetAttribute(::pk_attn9<false, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, A8_SMEM));
    CK(cudaFuncSetAttribute(::pk_attn9<true, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, A8_SMEM));
    CK(cudaFuncSetAttribute(::pk_attn9<false, true, false, 6, 5>, cudaFuncAttributeMaxDynamicSharedMemorySize, A8_SMEM));
    CK(cudaFuncSetAttribute(::pk_attn9<false, true, false, 5, 5>, cudaFuncAttributeMaxDynamicSharedMemorySize, A8_SMEM));
    CK(cudaFuncSetAttribute(::pk_attn9<false, false, false, 0, 8>, cudaFuncAttributeMaxDynamicSharedMemorySize, A8_SMEM));
}
void pk_attn8_exa_attr() {
    CK(cudaFuncSetAttribute(::pk_attn8<false, false, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, A8_SMEM));
    CK(cudaFuncSetAttribute(::pk_attn8<false, true, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, A8_SMEM));
    CK(cudaFuncSetAttribute(::pk_attn8<false, true, true, 6, 5>, cudaFuncAttributeMaxDynamicSharedMemorySize, A8_SMEM));
    CK(cudaFuncSetAttribute(::pk_attn8<false, true, true, 5, 5>, cudaFuncAttributeMaxDynamicSharedMemorySize, A8_SMEM));
    CK(cudaFuncSetAttribute(::pk_attn9<false, false, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, A8_SMEM));
    CK(cudaFuncSetAttribute(::pk_attn9<false, true, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, A8_SMEM));
    CK(cudaFuncSetAttribute(::pk_attn9<false, true, true, 6, 5>, cudaFuncAttributeMaxDynamicSharedMemorySize, A8_SMEM));
    CK(cudaFuncSetAttribute(::pk_attn9<false, true, true, 5, 5>, cudaFuncAttributeMaxDynamicSharedMemorySize, A8_SMEM));
    CK(cudaFuncSetAttribute(::pk_attn9<false, false, true, 0, 8>, cudaFuncAttributeMaxDynamicSharedMemorySize, A8_SMEM));
}
void pk_attn2_occupancy(int * nb) {
    CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(nb, ::pk_attn2<true, false>, A2_NT, A2_SMEM));
}
void pk_attn8_occupancy(int * nb) {
    CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(nb, ::pk_attn8<false, false>, A2_NT, A8_SMEM));
}

template <int TYPE, bool GLU, int LAY> void pf_mmh_attr() {
    CK(cudaFuncSetAttribute(::pf::mmh<TYPE, GLU, 32, LAY>, cudaFuncAttributeMaxDynamicSharedMemorySize, ::pf::hm_smem<TYPE, 32>()));
    CK(cudaFuncSetAttribute(::pf::mmh<TYPE, GLU, 64, LAY>, cudaFuncAttributeMaxDynamicSharedMemorySize, ::pf::hm_smem<TYPE, 64>()));
    CK(cudaFuncSetAttribute(::pf::mmh<TYPE, GLU, 128, LAY>, cudaFuncAttributeMaxDynamicSharedMemorySize, ::pf::hm_smem<TYPE, 128>()));
}
template <int TYPE, bool GLU, int LAY>
void pf_mmh(int nb, int nrt, const void * W, const __half * X, int ntok, const void * O, const uint8_t * W2, cudaStream_t st) {
    ::pf::WMat w;
    ::pf::Out o;
    memcpy(&w, W, sizeof w);
    memcpy(&o, O, sizeof o);
    switch (nb) {
        case 32: ::pf::mmh<TYPE, GLU, 32, LAY><<<dim3(nrt, (ntok + 31) / 32), 256, ::pf::hm_smem<TYPE, 32>(), st>>>(w, X, ntok, o, W2); break;
        case 64: ::pf::mmh<TYPE, GLU, 64, LAY><<<dim3(nrt, (ntok + 63) / 64), 256, ::pf::hm_smem<TYPE, 64>(), st>>>(w, X, ntok, o, W2); break;
        default: ::pf::mmh<TYPE, GLU, 128, LAY><<<dim3(nrt, (ntok + 127) / 128), 256, ::pf::hm_smem<TYPE, 128>(), st>>>(w, X, ntok, o, W2); break;
    }
}
template <int TG, int LG, int TU, int LU> void pf_mmh_mix_attr() {
    CK(cudaFuncSetAttribute(::pf::mmh<TG, true, 32, LG, TU, LU>, cudaFuncAttributeMaxDynamicSharedMemorySize, ::pf::hm_smem2<TG, TU, 32>()));
    CK(cudaFuncSetAttribute(::pf::mmh<TG, true, 64, LG, TU, LU>, cudaFuncAttributeMaxDynamicSharedMemorySize, ::pf::hm_smem2<TG, TU, 64>()));
    CK(cudaFuncSetAttribute(::pf::mmh<TG, true, 128, LG, TU, LU>, cudaFuncAttributeMaxDynamicSharedMemorySize, ::pf::hm_smem2<TG, TU, 128>()));
}
template <int TG, int LG, int TU, int LU>
void pf_mmh_mix(int nb, int nrt, const void * Wg, const void * Wu, const __half * X, int ntok, const void * O) {
    ::pf::WMat wg, wu;
    ::pf::Out o;
    memcpy(&wg, Wg, sizeof wg);
    memcpy(&wu, Wu, sizeof wu);
    memcpy(&o, O, sizeof o);
    switch (nb) {
        case 32: ::pf::mmh<TG, true, 32, LG, TU, LU><<<dim3(nrt, (ntok + 31) / 32), 256, ::pf::hm_smem2<TG, TU, 32>()>>>(wg, X, ntok, o, nullptr, wu); break;
        case 64: ::pf::mmh<TG, true, 64, LG, TU, LU><<<dim3(nrt, (ntok + 63) / 64), 256, ::pf::hm_smem2<TG, TU, 64>()>>>(wg, X, ntok, o, nullptr, wu); break;
        default: ::pf::mmh<TG, true, 128, LG, TU, LU><<<dim3(nrt, (ntok + 127) / 128), 256, ::pf::hm_smem2<TG, TU, 128>()>>>(wg, X, ntok, o, nullptr, wu); break;
    }
}
#define MKL_MMH(TYPE, GLU, LAY)                                                                                        \
    template void pf_mmh<TYPE, GLU, LAY>(int, int, const void *, const __half *, int, const void *, const uint8_t *, cudaStream_t); \
    template void pf_mmh_attr<TYPE, GLU, LAY>();
MKL_MMH(pf::TQ4K, false, 0) MKL_MMH(pf::TQ5K, false, 0) MKL_MMH(pf::TQ6K, false, 0) MKL_MMH(pf::TQ80, false, 0)
MKL_MMH(pf::TIQ4, false, 0) MKL_MMH(pf::TIQ4NL, false, 0) MKL_MMH(pf::TQ3K, false, 0) MKL_MMH(pf::TIQ3S, false, 0)
MKL_MMH(pf::TQ4K, true, 0) MKL_MMH(pf::TQ5K, true, 0) MKL_MMH(pf::TIQ4, true, 0)
MKL_MMH(pf::TQ4K, false, 1) MKL_MMH(pf::TQ5K, false, 1) MKL_MMH(pf::TQ80, false, 1) MKL_MMH(pf::TIQ4, false, 1)
MKL_MMH(pf::TIQ4NL, false, 1) MKL_MMH(pf::TQ6K, false, 1) MKL_MMH(pf::TQ3K, false, 1) MKL_MMH(pf::TIQ3S, false, 1)
MKL_MMH(pf::TQ4K, true, 1) MKL_MMH(pf::TQ5K, true, 1) MKL_MMH(pf::TIQ4, true, 1)
#undef MKL_MMH
#define MKL_MIX(TG, LG, TU, LU)                                                                                        \
    template void pf_mmh_mix<TG, LG, TU, LU>(int, int, const void *, const void *, const __half *, int, const void *); \
    template void pf_mmh_mix_attr<TG, LG, TU, LU>();
MK_GLU_MIX_PAIRS(MKL_MIX)
#undef MKL_MIX

} // namespace mkl
