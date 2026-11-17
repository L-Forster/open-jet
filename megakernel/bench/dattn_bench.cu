// Decode attention microbenchmark: the megakernel's attn_core_t<T, true> (variant 0, the simple q8 / q4 decode attention)
// against attn_core_q8_t<T> (variant 1, all-q8 slots) and attn_core_q8_t<T, true> (variant 2, attn_core_q8m: q8 or q4 K / V)
// on a random KV cache: bitwise comparison of the split-K partials (apart, aml), a T-invariance check and the time per call
// at a given depth. The kernels are the bench's own copies of the decode attention kernels, taken from an older engine
// (q8 / q4 K / V, T up to 5) and not regenerated automatically: they do not follow changes to the engine's kernels.
//   nvcc -O3 -arch=sm_86 -std=c++17 -Xptxas -dlcm=cg -Xptxas -v dattn_bench.cu -o dattn_bench
//   ./dattn_bench <n> <T> <reps> 0 1 2    (positions 0 .. n-1 cached, T new rows at n .. n+T-1; TINV=1 checks T-invariance)
//   exit status 1 when a variant's partials differ from the first variant's or a T-invariance check fails (`make check`)
//   e.g. ./dattn_bench 101600 4 20 0 1 2;  MK_KV-like formats: KQ4=1 / VQ4=1 (q4 K / V; then compare 0 2), KROT=1 (rotated K)
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CK(x)                                                                                                       \
    do {                                                                                                            \
        cudaError_t e_ = (x);                                                                                       \
        if (e_ != cudaSuccess) {                                                                                    \
            fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #x, __FILE__, __LINE__, cudaGetErrorString(e_));        \
            exit(1);                                                                                                \
        }                                                                                                           \
    } while (0)

#ifndef MK_KV_EMU
#define MK_KV_EMU 0
#endif

namespace {

constexpr int D = 5120, NH = 24, NKV = 4, HD = 256;
constexpr int QG = 12288, KVD = 1024;
constexpr int MAXT = 5;
constexpr int NT = 512, NW = NT / 32;
constexpr float EPS = 1e-6f;
constexpr int N_ATT_SLOTS = 2;
constexpr int MAX_CH = 32;
constexpr int AR = 6 * MAXT;
constexpr int ATP = 32;
constexpr int ATT_NCH = 20;
constexpr int ATT_BUCKET = ATT_NCH * ATP;
inline int att_cap_T(int n, int T_req) {
    const int lim = ATT_BUCKET * ((n + ATT_BUCKET) / ATT_BUCKET) - n;
    return T_req < lim ? T_req : lim;
}
constexpr int KVQ_ROW = 256 + 16;
constexpr int SM_ATTN_KV = (4 * ATP * (256 + 8) * 2 > 2 * ATP * (256 + 8) * 2 + 4 * ATP * KVQ_ROW)
                               ? 4 * ATP * (256 + 8) * 2 : 2 * ATP * (256 + 8) * 2 + 4 * ATP * KVQ_ROW;
constexpr int SM_ATTN = 32 * (256 + 8) * 2 + SM_ATTN_KV + 32 * ATP * 4 + 32 * (ATP + 8) * 2 + 96 * 4;
constexpr int SM_MISC_2K = 89088; // the megakernel's SM_MISC + 2048 (activation fills)
constexpr int SM_TOTAL = (SM_MISC_2K > SM_ATTN ? SM_MISC_2K : SM_ATTN);
constexpr int AQ_NSK = 3, AQ_NSV = 4; // K(t) is issued in interval t - 2 and read in t; V(t) issued in t - 1, read in t + 2
constexpr int AQ_LDS = ATP + 4;       // S row stride (floats): the softmax quads of 8 rows hit distinct banks
constexpr int AQ_STG = ATP * KVQ_ROW; // one K or V ring stage
constexpr int SM_ATTN_Q8 = 32 * (256 + 8) * 2 + (AQ_NSK + AQ_NSV) * AQ_STG + 2 * 32 * AQ_LDS * 4 + 2 * 32 * (ATP + 8) * 2 + 4 * 32 * 4;
constexpr int SM_LAUNCH = SM_ATTN_Q8 > SM_TOTAL ? SM_ATTN_Q8 : SM_TOTAL;
static_assert(SM_LAUNCH <= 99 * 1024, "shared memory budget");

struct LayerW {
    const float *qn, *kn;
};

struct Bufs {
    float *qbuf, *kbuf, *vbuf;
    void *kc, *vc;
    __half *ksc, *vsc;
    int kvq8;
    int8_t * kvp[2][N_ATT_SLOTS];
    __half * kvs[2][N_ATT_SLOTS];
    int kvb[2][N_ATT_SLOTS];
    int kvr[N_ATT_SLOTS];
    float * apart;
    float2 * aml;
    const float2 * rope;
    unsigned long long * prof; // unused
    int ctx;
};

__constant__ LayerW cWL;
__constant__ Bufs cB;
extern __shared__ __align__(16) uint8_t smem[];

// ===== the bench's copies of the decode attention kernels (from an older qwen38_mk.cu, not kept in step) =====
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
__device__ __forceinline__ void cp_async16(void * smem_dst, const void * gsrc) {
    const unsigned d = (unsigned) __cvta_generic_to_shared(smem_dst);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(d), "l"(gsrc) : "memory");
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

// one K (w = 0) or V (w = 1) row of attention slot `slot` in its MK_KV format; row = kv head * ctx + position
__device__ __forceinline__ void kv_store(int w, int slot, size_t row, const float (&vin)[8]) {
    float v[8];
#pragma unroll
    for (int k = 0; k < 8; k++) v[k] = vin[k];
    if (w == 0 && cB.kvr[slot]) fwht256(v);
    if (cB.kvb[w][slot] == 4) kv_store_q4(cB.kvp[w][slot] + row * (HD / 2), cB.kvs[w][slot] + row * (HD / 32), v);
    else kv_store_q8(cB.kvp[w][slot] + row * HD, cB.kvs[w][slot] + row * (HD / 32), v);
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

// staged q8 / q4 K/V rows (KVQ_ROW bytes: values, then scales at 256) -> fp16 rows of stride ld; rows x 2 tensors
__device__ __forceinline__ void kv_tile_to_f16(const uint8_t * src, __half * dK, __half * dV, int rows, int ld, bool kq4, bool vq4) {
    for (int i = threadIdx.x; i < 2 * rows * 32; i += blockDim.x) {
        const int which = i / (rows * 32), r = (i >> 5) % rows, c = i & 31;
        const uint8_t * row = src + (size_t) (which * rows + r) * KVQ_ROW;
        const float d = __half2float(reinterpret_cast<const __half *>(row + 256)[c >> 2]);
        __half2 h[4];
        if (which ? vq4 : kq4) q4_to_f16(*reinterpret_cast<const unsigned *>(row + 4 * c), d, h);
        else q8_to_f16(*reinterpret_cast<const uint2 *>(row + 8 * c), d, h);
        *reinterpret_cast<int4 *>((which ? dV : dK) + (size_t) r * ld + 8 * c) = *reinterpret_cast<int4 *>(h);
    }
}

// async copy of q8 / q4 K/V rows r < np (slot rows row0 + r) into the staging layout (rows >= np: zero scales)
__device__ __forceinline__ void kv_stage(uint8_t * dst, const int8_t * Kq, const int8_t * Vq, const __half * Ks, const __half * Vs,
                                         size_t row0, int rows, int np, bool kq4, bool vq4) {
    for (int i = threadIdx.x; i < 2 * rows * 17; i += blockDim.x) {
        const int which = i / (rows * 17), r = (i / 17) % rows, c = i % 17;
        uint8_t * d = dst + (size_t) (which * rows + r) * KVQ_ROW + 16 * c;
        if (r < np) {
            const size_t row = row0 + r;
            const bool q4 = which ? vq4 : kq4;
            if (c < 16) {
                if (!q4 || c < 8) cp_async16(d, (which ? Vq : Kq) + row * (q4 ? HD / 2 : HD) + 16 * c);
            } else {
                cp_async16(d, (which ? Vs : Ks) + row * (HD / 32));
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
    const bool kq4 = Q8 && cB.kvb[0][slot] == 4, vq4 = Q8 && cB.kvb[1][slot] == 4; // MK_KV per-slot formats
    const int mt = warp >> 3, wn = warp & 7; // MMA work split: warp -> (row tile, column group)

    auto issue_tile = [&](int g, int tb, int np_, int st) {
        if constexpr (Q8) {
            kv_stage(sKVq + st * 2 * TP * KVQ_ROW, cB.kvp[0][slot], cB.kvp[1][slot], cB.kvs[0][slot], cB.kvs[1][slot],
                     (size_t) g * ctx + tb, TP, np_, kq4, vq4);
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
        // Unconditional loads (rows >= R and lanes >= 8 load valid, unused data): conditionally assigned arrays
        // would be carried across the item loop in local memory (~1.5 KB of mk_cycle's stack frame, which the driver
        // reserves for all 1,536 thread slots of every SM).
#pragma unroll
        for (int k2 = 0; k2 < 2; k2++) {
            const int r = min(warp + 16 * k2, R - 1);
            const int hq = r / T, t = r % T, head = g * 6 + hq;
            load8(cB.qbuf + t * QG + head * 512 + lane * 8, qv[k2]);
            const float2 * tab = cB.rope + (size_t) (n + t) * 32 + (lane & 3) * 8;
#pragma unroll
            for (int k = 0; k < 8; k++) rc[k2][k] = __ldg(tab + k);
        }
#pragma unroll
        for (int k2 = 0; k2 < 2; k2++) {
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
            const size_t row = slot0 + (size_t) g * ctx + pp, srow = (size_t) g * ctx + pp;
#if MK_KV_EMU
            if (Q8 && cB.kvemu[0][slot]) kv_store_emu(static_cast<int8_t *>(cB.kc) + row * HD, cB.ksc + row * (HD / 32), v, cB.kvemu[0][slot]);
            else
#endif
            if constexpr (Q8) kv_store(0, slot, srow, v);
            else kv_store_f16(static_cast<__half *>(cB.kc) + row * HD, v);
            load8(cB.vbuf + t * KVD + g * 256 + lane * 8, v);
#if MK_KV_EMU
            if (Q8 && cB.kvemu[1][slot]) kv_store_emu(static_cast<int8_t *>(cB.vc) + row * HD, cB.vsc + row * (HD / 32), v, cB.kvemu[1][slot]);
            else
#endif
            if constexpr (Q8) kv_store(1, slot, srow, v);
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
                kv_tile_to_f16(sKVq + st * 2 * TP * KVQ_ROW, sKV, sKV + TP * LDH, TP, LDH, kq4, vq4);
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
// minus 1152 is q exactly, and one fp16 multiply rounds the exact product (= q8_to_f16's value)
__device__ __forceinline__ unsigned aq_q8x2(unsigned u, __half2 d) {
    const unsigned h = __byte_perm(u ^ 0x8080u, 0x64646464u, 0x4140);
    const __half2 q = __hsub2(*reinterpret_cast<const __half2 *>(&h), __half2half2(__ushort_as_half(0x6480)));
    const __half2 r = __hmul2(q, d);
    return *reinterpret_cast<const unsigned *>(&r);
}
// the same for bytes 0 / 2 or 1 / 3 of y (selector 0x4140 / 0x4342 after the caller's interleave)
__device__ __forceinline__ unsigned aq_q8x2p(unsigned y, unsigned sel, __half2 d) {
    const unsigned h = __byte_perm(y, 0x64646464u, sel);
    const __half2 q = __hsub2(*reinterpret_cast<const __half2 *>(&h), __half2half2(__ushort_as_half(0x6480)));
    const __half2 r = __hmul2(q, d);
    return *reinterpret_cast<const unsigned *>(&r);
}
// attn_core_q8m (q8 and q4 slots): bytes sel of y hold biased values b (q8: q + 128 = q ^ 0x80, q4: the nibble = q + 8);
// 0x6400 | b is the fp16 1024 + b, minus bias (0x6480 = 1152 for q8, 0x6408 = 1032 for q4) is q exactly, and one fp16
// multiply rounds the exact product d * q (= q8_to_f16's / q4_to_f16's value)
__device__ __forceinline__ unsigned aq_x2p(unsigned y, unsigned sel, __half2 d, __half2 bias) {
    const unsigned h = __byte_perm(y, 0x64646464u, sel);
    const __half2 q = __hsub2(*reinterpret_cast<const __half2 *>(&h), bias);
    const __half2 r = __hmul2(q, d);
    return *reinterpret_cast<const unsigned *>(&r);
}

// one commit group's copies, issued by NTH threads (this one is thread t of them): K rows of a tile (kn valid rows,
// kn < 0: none) -> K stage dk, V rows of a tile -> dv; rows >= the valid count get zero scales (their values are
// never used: masked S, P = 0). MIX: q4 rows (kq4 / vq4) copy 8 value chunks (128 B, kv_stage's staging layout).
template <int NTH, bool MIX = false>
__device__ __forceinline__ void aq_issue(int t, uint8_t * dk, size_t krow0, int kn, uint8_t * dv, size_t vrow0, int vn, int slot,
                                         bool kq4 = false, bool vq4 = false) {
#pragma unroll
    for (int k = 0; k < (2 * ATP * 17 + NTH - 1) / NTH; k++) {
        const int i = t + NTH * k;
        if (i < 2 * ATP * 17) {
            const int which = i >= ATP * 17, j = which ? i - ATP * 17 : i, r = j / 17, c = j - 17 * r;
            const int np = which ? vn : kn;
            if (np < 0) continue;
            uint8_t * d = (which ? dv : dk) + r * KVQ_ROW + 16 * c;
            if (r < np) {
                const size_t row = (which ? vrow0 : krow0) + r;
                if (MIX && (which ? vq4 : kq4)) {
                    if (c < 8) cp_async16(d, cB.kvp[which][slot] + row * (HD / 2) + 16 * c);
                    else if (c == 16) cp_async16(d, cB.kvs[which][slot] + row * (HD / 32));
                } else if (c < 16) cp_async16(d, cB.kvp[which][slot] + row * HD + 16 * c);
                else cp_async16(d, cB.kvs[which][slot] + row * (HD / 32));
            } else if (c == 16) {
                *reinterpret_cast<int4 *>(d) = make_int4(0, 0, 0, 0);
            }
        }
    }
}

// MIX = false: all-q8 slots (attn_core_q8). MIX = true (attn_core_q8m): K and V of the slot each q8 or q4 (MK_KV), the q4
// fragments converted with q4_to_f16's values (aq_x2p), so the partials equal attn_core_t's for every MK_KV format.
template <int T, bool MIX = false>
__device__ __forceinline__ void attn_core_q8_t(const LayerW & W, int slot, int n) {
    uint8_t * const sm = smem;
    constexpr int R = 6 * T;
    constexpr int MT = (R + 15) / 16; // 16-row MMA tiles of query rows
    constexpr int TP = ATP;
    constexpr int LDH = HD + 8;
    constexpr int LDP = TP + 8;
    constexpr int LDS = AQ_LDS;
    constexpr int NSK = AQ_NSK, NSV = AQ_NSV;
    static_assert(R <= 32 && TP == 32 && MT <= 2, "8 rows per softmax warp, 32-position tiles");
    const int L_ctx = n + T;
    int nch, CS;
    attn_dims(L_ctx, nch, CS);
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g8 = lane >> 2, tig = lane & 3;
    __half * sQ = reinterpret_cast<__half *>(sm);                 // [32][LDH]
    uint8_t * kring = reinterpret_cast<uint8_t *>(sQ + 32 * LDH); // NSK stages
    uint8_t * vring = kring + NSK * AQ_STG;                       // NSV stages
    float * sS = reinterpret_cast<float *>(vring + NSV * AQ_STG); // [2][32][LDS]
    __half * sP = reinterpret_cast<__half *>(sS + 2 * 32 * LDS);  // [2][32][LDP]
    float * sM = reinterpret_cast<float *>(sP + 2 * 32 * LDP);
    float * sL = sM + 32;
    float * sAl = sL + 32; // [2][32]
    const size_t ctx = cB.ctx;
    const size_t slot0 = (size_t) slot * NKV * ctx; // first row of this slot
    (void) slot0;
    const bool kq4 = MIX && cB.kvb[0][slot] == 4, vq4 = MIX && cB.kvb[1][slot] == 4; // MK_KV per-slot formats

    for (int item = blockIdx.x; item < NKV * nch; item += gridDim.x) {
        const int g = item / nch, c = item % nch;
        const int c0 = c * CS, c1 = min(L_ctx, c0 + CS);
        // ---- item prologue, attn_core_t's (q rows -> sQ, this cycle's K / V rows -> cache, m / l init)
        float qv[2][8], qw[8];
        float2 rc[2][8];
        load8(W.qn + lane * 8, qw);
#pragma unroll
        for (int k2 = 0; k2 < 2; k2++) {
            const int r = min(warp + 16 * k2, R - 1);
            const int hq = r / T, t = r % T, head = g * 6 + hq;
            load8(cB.qbuf + t * QG + head * 512 + lane * 8, qv[k2]);
            const float2 * tab = cB.rope + (size_t) (n + t) * 32 + (lane & 3) * 8;
#pragma unroll
            for (int k = 0; k < 8; k++) rc[k2][k] = __ldg(tab + k);
        }
#pragma unroll
        for (int k2 = 0; k2 < 2; k2++) {
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
                __half2 hq2[4];
#pragma unroll
                for (int k = 0; k < 4; k++) hq2[k] = __floats2half2_rn(v[2 * k] * 0.0625f, v[2 * k + 1] * 0.0625f);
                *reinterpret_cast<int4 *>(sQ + r * LDH + lane * 8) = *reinterpret_cast<int4 *>(hq2);
            } else {
                *reinterpret_cast<int4 *>(sQ + r * LDH + lane * 8) = make_int4(0, 0, 0, 0);
                if (2 * lane < TP) { // padded rows of both P buffers stay 0
                    *reinterpret_cast<unsigned *>(sP + r * LDP + 2 * lane) = 0u;
                    *reinterpret_cast<unsigned *>(sP + 32 * LDP + r * LDP + 2 * lane) = 0u;
                }
            }
        }
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
            const size_t row = slot0 + (size_t) g * ctx + pp, srow = (size_t) g * ctx + pp;
            (void) row;
#if MK_KV_EMU
            if (cB.kvemu[0][slot]) kv_store_emu(static_cast<int8_t *>(cB.kc) + row * HD, cB.ksc + row * (HD / 32), v, cB.kvemu[0][slot]);
            else
#endif
            kv_store(0, slot, srow, v);
            load8(cB.vbuf + t * KVD + g * 256 + lane * 8, v);
#if MK_KV_EMU
            if (cB.kvemu[1][slot]) kv_store_emu(static_cast<int8_t *>(cB.vc) + row * HD, cB.vsc + row * (HD / 32), v, cB.kvemu[1][slot]);
            else
#endif
            kv_store(1, slot, srow, v);
        }
        if (tid < 32) {
            sM[tid] = -INFINITY;
            sL[tid] = 0.f;
        }
        __threadfence_block();
        __syncthreads();

        // ---- ring prologue: commit groups {K(0)}, {K(1), V(0)}; then per interval jt {K(jt + 2), V(jt + 1)}
        const int ntl = (c1 - c0 + TP - 1) / TP;
        const size_t grow = (size_t) g * ctx + c0;
        auto tnp = [&](int t) { return t < ntl ? min(TP, c1 - c0 - t * TP) : -1; }; // valid rows of tile t, -1: none
        aq_issue<NT, MIX>(tid, kring, grow, tnp(0), vring, grow, -1, slot, kq4, vq4);
        asm volatile("cp.async.commit_group;" ::: "memory");
        aq_issue<NT, MIX>(tid, kring + AQ_STG, grow + TP, tnp(1), vring, grow, tnp(0), slot, kq4, vq4);
        asm volatile("cp.async.commit_group;" ::: "memory");
        asm volatile("cp.async.wait_group 1;" ::: "memory");
        __syncthreads();

        float acc[MT][4][4]; // P V warps: O rows 16 i + {g8, g8 + 8}, dims 32 wn + 8 tig + jj and + 4
#pragma unroll
        for (int i = 0; i < MT; i++)
#pragma unroll
            for (int j = 0; j < 4; j++)
#pragma unroll
                for (int k = 0; k < 4; k++) acc[i][j][k] = 0.f;

        for (int jt = 0; jt < ntl + 2; jt++) {
            if (warp < 4) {
                // ---- S(jt) = Q K(jt)^T: B = positions 8 wn + g8 (all 16 k-steps converted first), all row tiles
                if (jt < ntl) {
                    const int wn = warp;
                    const uint8_t * krow = kring + (jt % NSK) * AQ_STG + (8 * wn + g8) * KVQ_ROW;
                    const uint4 scv = *reinterpret_cast<const uint4 *>(krow + 256);
                    const unsigned scw[4] = {scv.x, scv.y, scv.z, scv.w};
                    unsigned kb[HD / 16][2];
                    if (!MIX || !kq4) {
#pragma unroll
                        for (int ks = 0; ks < HD / 16; ks++) {
                            const unsigned sw = __byte_perm(scw[ks >> 2], 0, (ks & 2) ? 0x3232 : 0x1010);
                            const __half2 dd = *reinterpret_cast<const __half2 *>(&sw);
                            kb[ks][0] = aq_q8x2(*reinterpret_cast<const unsigned short *>(krow + 16 * ks + 2 * tig), dd);
                            kb[ks][1] = aq_q8x2(*reinterpret_cast<const unsigned short *>(krow + 16 * ks + 2 * tig + 8), dd);
                        }
                    } else {
                        // q4 K: dims 16 ks + 2 tig (+1) are nibbles (tig >> 1: high) of bytes 2 (tig & 1) (+1) of word 2 ks,
                        // dims + 8 the same of word 2 ks + 1
                        const uint8_t * kq = krow + 2 * (tig & 1);
                        const int ksh = 4 * (tig >> 1);
                        const __half2 kbias = __half2half2(__ushort_as_half(0x6408));
#pragma unroll
                        for (int ks = 0; ks < HD / 16; ks++) {
                            const unsigned sw = __byte_perm(scw[ks >> 2], 0, (ks & 2) ? 0x3232 : 0x1010);
                            const __half2 dd = *reinterpret_cast<const __half2 *>(&sw);
                            const unsigned w0 = (unsigned) *reinterpret_cast<const unsigned short *>(kq + 8 * ks) >> ksh;
                            const unsigned w1 = (unsigned) *reinterpret_cast<const unsigned short *>(kq + 8 * ks + 4) >> ksh;
                            kb[ks][0] = aq_x2p(w0 & 0x0f0fu, 0x4140, dd, kbias);
                            kb[ks][1] = aq_x2p(w1 & 0x0f0fu, 0x4140, dd, kbias);
                        }
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
            } else if (warp < 12) {
                // ---- O = O * alpha + P(jt - 2) V(jt - 2): B = dims 32 wn + 4 g8 + jj of n8 tile jj, all row tiles
                if (jt >= 2) {
                    const int wn = warp - 4, jp = jt - 2, b = jp & 1;
#pragma unroll
                    for (int i = 0; i < MT; i++) {
                        const float al0 = sAl[b * 32 + 16 * i + g8], al1 = sAl[b * 32 + 16 * i + g8 + 8];
#pragma unroll
                        for (int j = 0; j < 4; j++) {
                            acc[i][j][0] *= al0;
                            acc[i][j][1] *= al0;
                            acc[i][j][2] *= al1;
                            acc[i][j][3] *= al1;
                        }
                    }
                    const unsigned paddr = (unsigned) __cvta_generic_to_shared(sP + b * 32 * LDP + (lane & 15) * LDP + (lane >> 4) * 8);
                    // q4 V (MIX): dims 32 wn + 4 g8 + jj are nibbles (g8 & 1: high) of bytes jj of word 4 wn + (g8 >> 1)
                    const uint8_t * vst = vring + (jp % NSV) * AQ_STG + (MIX && vq4 ? 16 * wn + 4 * (g8 >> 1) : 32 * wn + 4 * g8);
                    const uint8_t * vsc = vring + (jp % NSV) * AQ_STG + 256 + 2 * wn;
                    const int vsh = MIX && vq4 ? 4 * (g8 & 1) : 0;
                    const unsigned vmsk = MIX && vq4 ? 0x0f0f0f0fu : 0xffffffffu, vxor = MIX && vq4 ? 0u : 0x80808080u;
                    const __half2 vbias = __half2half2(__ushort_as_half(MIX && vq4 ? 0x6408 : 0x6480));
#pragma unroll
                    for (int ks = 0; ks < TP / 16; ks++) {
                        unsigned bf[2][4]; // [b0 / b1][tile jj]
#pragma unroll
                        for (int h = 0; h < 2; h++) {
                            const int p = 16 * ks + 2 * tig + 8 * h; // k rows p, p + 1 of the fragment
                            unsigned xa, xb;
                            if constexpr (MIX) {
                                xa = ((*reinterpret_cast<const unsigned *>(vst + p * KVQ_ROW) >> vsh) & vmsk) ^ vxor;
                                xb = ((*reinterpret_cast<const unsigned *>(vst + (p + 1) * KVQ_ROW) >> vsh) & vmsk) ^ vxor;
                            } else {
                                xa = *reinterpret_cast<const unsigned *>(vst + p * KVQ_ROW) ^ 0x80808080u;
                                xb = *reinterpret_cast<const unsigned *>(vst + (p + 1) * KVQ_ROW) ^ 0x80808080u;
                            }
                            const unsigned sw = (unsigned) *reinterpret_cast<const unsigned short *>(vsc + p * KVQ_ROW) |
                                                ((unsigned) *reinterpret_cast<const unsigned short *>(vsc + (p + 1) * KVQ_ROW) << 16);
                            const __half2 dd = *reinterpret_cast<const __half2 *>(&sw);
                            const unsigned y0 = __byte_perm(xa, xb, 0x5140), y1 = __byte_perm(xa, xb, 0x7362);
                            if constexpr (MIX) {
                                bf[h][0] = aq_x2p(y0, 0x4140, dd, vbias);
                                bf[h][1] = aq_x2p(y0, 0x4342, dd, vbias);
                                bf[h][2] = aq_x2p(y1, 0x4140, dd, vbias);
                                bf[h][3] = aq_x2p(y1, 0x4342, dd, vbias);
                            } else {
                                bf[h][0] = aq_q8x2p(y0, 0x4140, dd);
                                bf[h][1] = aq_q8x2p(y0, 0x4342, dd);
                                bf[h][2] = aq_q8x2p(y1, 0x4140, dd);
                                bf[h][3] = aq_q8x2p(y1, 0x4342, dd);
                            }
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
            } else {
                // ---- online softmax of tile jt - 1 (causal mask: row token t sees positions <= n + t); lane l4 of the
                // row's quad holds positions l4 + 4 k
                if (jt >= 1 && jt <= ntl && 8 * (warp - 12) < R) {
                    const int r = 8 * (warp - 12) + (lane >> 2), l4 = lane & 3; // rows >= R (zero S) are not stored
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
                    const float mold = sM[r];
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
                    if (r < R) {
                        __half * prow = sP + b * 32 * LDP + r * LDP;
#pragma unroll
                        for (int k = 0; k < 8; k++) prow[l4 + 4 * k] = __float2half_rn(p[k]);
                        if (l4 == 0) {
                            sL[r] = sL[r] * al + ps;
                            sM[r] = mnew;
                            sAl[b * 32 + r] = al;
                        }
                    }
                }
            }
            // ---- copies: K(jt + 2) into the stage of K(jt - 1), V(jt + 1) into the stage of V(jt - 3) (both read in
            // interval jt - 1), issued by warps 4-15 (the S warps carry the longest MMA chains);
            // then wait for K(jt + 1) and V(jt)
            if (jt + 1 < ntl && warp >= 4)
                aq_issue<NT - 128, MIX>(tid - 128, kring + ((jt + 2) % NSK) * AQ_STG, grow + (jt + 2) * TP, tnp(jt + 2),
                                        vring + ((jt + 1) % NSV) * AQ_STG, grow + (jt + 1) * TP, tnp(jt + 1), slot, kq4, vq4);
            asm volatile("cp.async.commit_group;" ::: "memory");
            asm volatile("cp.async.wait_group 1;" ::: "memory");
            __syncthreads();
        }
        if (warp >= 4 && warp < 12) {
            const int wn = warp - 4;
#pragma unroll
            for (int i = 0; i < MT; i++)
#pragma unroll
                for (int j = 0; j < 4; j++) {
                    const int r0 = 16 * i + g8, d0 = 32 * wn + 8 * tig + j; // columns 2 tig, 2 tig + 1: dims d0, d0 + 4
                    if (r0 < R) {
                        float * o = cB.apart + ((size_t) item * AR + r0) * 256 + d0;
                        o[0] = acc[i][j][0];
                        o[4] = acc[i][j][1];
                    }
                    if (r0 + 8 < R) {
                        float * o = cB.apart + ((size_t) item * AR + r0 + 8) * 256 + d0;
                        o[0] = acc[i][j][2];
                        o[4] = acc[i][j][3];
                    }
                }
        }
        if (tid < R) cB.aml[item * AR + tid] = make_float2(sM[tid], sL[tid]);
        __syncthreads();
    }
}

// ===== end of the copied kernels =====

// ===== driver =====

template <int T> __device__ __noinline__ void v_ref(int slot, int n) { attn_core_t<T, true>(cWL, slot, n); }
template <int T> __device__ __noinline__ void v_new(int slot, int n) { attn_core_q8_t<T>(cWL, slot, n); }
template <int T> __device__ __noinline__ void v_mix(int slot, int n) { attn_core_q8_t<T, true>(cWL, slot, n); }

#define DISP(V, F)                                                                                                  \
    if (v == V) {                                                                                                   \
        switch (T) {                                                                                                \
        case 1: F<1>(slot, n); break;                                                                               \
        case 2: F<2>(slot, n); break;                                                                               \
        case 3: F<3>(slot, n); break;                                                                               \
        case 4: F<4>(slot, n); break;                                                                               \
        case 5: F<5>(slot, n); break;                                                                               \
        }                                                                                                           \
        return;                                                                                                     \
    }

__global__ void __launch_bounds__(NT, 1) k_attn(int v, int T, int slot, int n) {
    DISP(0, v_ref)
    DISP(1, v_new)
    DISP(2, v_mix)
}

__global__ void k_init8(int8_t * p, size_t n, unsigned seed) {
    for (size_t i = blockIdx.x * (size_t) blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x * blockDim.x) {
        unsigned h = (unsigned) i * 2654435761u ^ seed;
        h ^= h >> 15; h *= 2246822519u; h ^= h >> 13; h *= 3266489917u; h ^= h >> 16;
        int v = (int) (h & 0xff) - 128;
        p[i] = (int8_t) (v < -127 ? -127 : v);
    }
}
__global__ void k_initsc(__half * p, size_t n, unsigned seed) {
    for (size_t i = blockIdx.x * (size_t) blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x * blockDim.x) {
        unsigned h = (unsigned) i * 2654435761u ^ seed;
        h ^= h >> 15; h *= 2246822519u; h ^= h >> 13;
        p[i] = __float2half(0.008f + (float) (h % 2000) * 1.2e-5f);
    }
}

struct Lcg {
    unsigned long long s;
    float u() { s = s * 6364136223846793005ull + 1442695040888963407ull; return (float) ((s >> 40) & 0xffffff) / 16777216.0f; }
};

} // namespace

static void attn_dims_h(int L_ctx, int & nch, int & cs) {
    cs = ATP * std::max(1, (L_ctx + ATT_BUCKET - 1) / ATT_BUCKET);
    nch = std::max(1, (L_ctx + cs - 1) / cs);
}

int main(int argc, char ** argv) {
    if (argc < 5) {
        fprintf(stderr, "usage: %s <n> <T> <reps> <variant> ...   env: TINV=1 (T-invariance check), CTX=<ctx>\n", argv[0]);
        return 1;
    }
    const int n = atoi(argv[1]), Treq = atoi(argv[2]), reps = atoi(argv[3]);
    if (Treq < 1 || Treq > MAXT) { // (the copies predate the engine's 6-8 token verify passes: other T launch nothing)
        fprintf(stderr, "T must be 1..%d\n", MAXT);
        return 1;
    }
    std::vector<int> vars;
    for (int i = 4; i < argc; i++) vars.push_back(atoi(argv[i]));
    const int T = att_cap_T(n, Treq);
    if (T != Treq) printf("T capped %d -> %d (bucket boundary)\n", Treq, T);
    const int ctx = getenv("CTX") ? atoi(getenv("CTX")) : std::max(120000, n + 64);
    const int L_ctx = n + T, R = 6 * T;
    int nch, cs;
    attn_dims_h(L_ctx, nch, cs);
    int nsm;
    CK(cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, 0));

    Bufs hb{};
    for (int s = 0; s < N_ATT_SLOTS; s++)
        for (int w = 0; w < 2; w++) {
            CK(cudaMalloc(&hb.kvp[w][s], (size_t) NKV * ctx * HD));
            CK(cudaMalloc(&hb.kvs[w][s], (size_t) NKV * ctx * (HD / 32) * 2));
            k_init8<<<1024, 256>>>(hb.kvp[w][s], (size_t) NKV * ctx * HD, 0x1234u + 77u * w); // same data in both slots
            k_initsc<<<1024, 256>>>(hb.kvs[w][s], (size_t) NKV * ctx * (HD / 32), 0x777u + 31u * w);
            hb.kvb[w][s] = 8;
        }
    hb.kvq8 = 1;
    // MK_KV-like formats for both slots: KQ4=1 (K q4), VQ4=1 (V q4), KROT=1 (K Hadamard-rotated)
    for (int s = 0; s < N_ATT_SLOTS; s++) {
        if (getenv("KQ4") && atoi(getenv("KQ4"))) hb.kvb[0][s] = 4;
        if (getenv("VQ4") && atoi(getenv("VQ4"))) hb.kvb[1][s] = 4;
        hb.kvr[s] = getenv("KROT") && atoi(getenv("KROT"));
    }
    printf("formats: K q%d%s, V q%d\n", hb.kvb[0][0], hb.kvr[0] ? " rotated" : "", hb.kvb[1][0]);
    hb.ctx = ctx;
    Lcg rg{12345};
    std::vector<float> q((size_t) MAXT * QG), kb((size_t) MAXT * KVD), vb((size_t) MAXT * KVD), qn(HD), kn(HD);
    for (auto & x : q) x = 4.f * rg.u() - 2.f;
    for (auto & x : kb) x = 4.f * rg.u() - 2.f;
    for (auto & x : vb) x = 4.f * rg.u() - 2.f;
    for (auto & x : qn) x = 0.8f + 0.4f * rg.u();
    for (auto & x : kn) x = 0.8f + 0.4f * rg.u();
    std::vector<float2> rope((size_t) ctx * 32);
    for (int p = 0; p < ctx; p++)
        for (int i = 0; i < 32; i++) {
            const double th = p * pow(10000000.0, -2.0 * i / 64.0);
            rope[(size_t) p * 32 + i] = make_float2((float) cos(th), (float) sin(th));
        }
    auto up = [](const void * h, size_t bytes) { void * d; CK(cudaMalloc(&d, bytes)); CK(cudaMemcpy(d, h, bytes, cudaMemcpyHostToDevice)); return d; };
    hb.qbuf = (float *) up(q.data(), q.size() * 4);
    hb.kbuf = (float *) up(kb.data(), kb.size() * 4);
    hb.vbuf = (float *) up(vb.data(), vb.size() * 4);
    hb.rope = (const float2 *) up(rope.data(), rope.size() * 8);
    LayerW hw{(const float *) up(qn.data(), HD * 4), (const float *) up(kn.data(), HD * 4)};
    const size_t napart = (size_t) NKV * MAX_CH * AR * 256, naml = (size_t) NKV * MAX_CH * AR;
    float * apart2; float2 * aml2;
    CK(cudaMalloc(&hb.apart, napart * 4));
    CK(cudaMalloc(&hb.aml, naml * 8));
    CK(cudaMalloc(&apart2, napart * 4));
    CK(cudaMalloc(&aml2, naml * 8));
    CK(cudaMalloc(&hb.prof, 256 * 16 * 8));
    CK(cudaMemcpyToSymbol(cB, &hb, sizeof hb));
    CK(cudaMemcpyToSymbol(cWL, &hw, sizeof hw));
    CK(cudaFuncSetAttribute(k_attn, cudaFuncAttributeMaxDynamicSharedMemorySize, SM_LAUNCH));
    CK(cudaDeviceSynchronize());
    printf("n %d T %d L_ctx %d ctx %d: nch %d, chunk %d positions (%d tiles), %d items on %d SMs, smem %d B\n", n, T, L_ctx,
           ctx, nch, cs, cs / ATP, NKV * nch, nsm, SM_LAUNCH);

    // correctness: every variant's partials vs variant 0, bitwise
    const int nitems = NKV * nch;
    auto run = [&](int v, int TT, int slot, int nn) {
        k_attn<<<nsm, NT, SM_LAUNCH>>>(v, TT, slot, nn);
        CK(cudaGetLastError());
    };
    auto fetch = [&](std::vector<float> & ap, std::vector<float2> & ml, const float * dap, const float2 * dml) {
        ap.resize(napart);
        ml.resize(naml);
        CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(ap.data(), dap, napart * 4, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(ml.data(), dml, naml * 8, cudaMemcpyDeviceToHost));
    };
    std::vector<float> ap0, ap;
    std::vector<float2> ml0, ml;
    long fails = 0;
    for (int v : vars) {
        CK(cudaMemset(hb.apart, 0xff, napart * 4));
        CK(cudaMemset(hb.aml, 0xff, naml * 8));
        run(v, T, 0, n);
        fetch(ap, ml, hb.apart, hb.aml);
        if (v == vars[0]) { ap0 = ap; ml0 = ml; }
        long ndiff = 0, nml = 0, nnan = 0;
        double maxd = 0, ss = 0, sr = 0;
        for (int it = 0; it < nitems; it++)
            for (int r = 0; r < R; r++) {
                const size_t o = (size_t) it * AR + r;
                if (memcmp(&ml[o], &ml0[o], 8)) nml++;
                for (int d = 0; d < 256; d++) {
                    const float a = ap[o * 256 + d], b = ap0[o * 256 + d];
                    if (std::isnan(a)) nnan++;
                    if (memcmp(&a, &b, 4)) {
                        ndiff++;
                        maxd = std::max(maxd, (double) fabsf(a - b));
                    }
                    ss += (double) (a - b) * (a - b);
                    sr += (double) b * b;
                }
            }
        printf("variant %d vs %d: %ld / %ld partials differ (max |d| %.3g, rel rms %.3g), %ld / %d (m, l) differ, %ld NaN\n", v,
               vars[0], ndiff, (long) nitems * R * 256, maxd, sqrt(ss / std::max(sr, 1e-30)), nml, nitems * R, nnan);
        fails += ndiff || nml || nnan;
    }

    // T-invariance: row (hq, t) of the T-row launch at n == row hq of a 1-row launch at n + t (same q / k / v data)
    if (getenv("TINV") && T > 1) {
        for (int v : vars) {
            run(v, T, 0, n);
            fetch(ap0, ml0, hb.apart, hb.aml);
            long bad = 0, chk = 0;
            for (int t = 0; t < T; t++) {
                Bufs h1 = hb;
                h1.qbuf += (size_t) t * QG;
                h1.kbuf += (size_t) t * KVD;
                h1.vbuf += (size_t) t * KVD;
                h1.apart = apart2;
                h1.aml = aml2;
                CK(cudaMemcpyToSymbol(cB, &h1, sizeof h1));
                run(v, 1, 0, n + t);
                fetch(ap, ml, apart2, aml2);
                CK(cudaMemcpyToSymbol(cB, &hb, sizeof hb));
                int nch1, cs1;
                attn_dims_h(n + t + 1, nch1, cs1);
                if (cs1 != cs) { printf("chunk size differs (%d vs %d)\n", cs1, cs); bad++; }
                for (int g = 0; g < NKV; g++)
                    for (int c = 0; c < nch; c++)
                        for (int hq = 0; hq < 6; hq++) {
                            const size_t oT = (size_t) (g * nch + c) * AR + hq * T + t;
                            chk++;
                            if (c < nch1) {
                                const size_t o1 = (size_t) (g * nch1 + c) * AR + hq;
                                if (memcmp(&ml0[oT], &ml[o1], 8) || memcmp(&ap0[oT * 256], &ap[o1 * 256], 1024)) bad++;
                            } else {
                                bool z = ml0[oT].x == -INFINITY && ml0[oT].y == 0.f;
                                for (int d = 0; d < 256; d++) z = z && ap0[oT * 256 + d] == 0.f;
                                if (!z) bad++;
                            }
                        }
            }
            printf("T-invariance variant %d: %ld / %ld (item, row) partials differ from the 1-row launches\n", v, bad, chk);
            fails += bad > 0;
        }
    }

    printf(fails ? "CHECK FAILED (%ld)\n" : "CHECK OK\n", fails);

    // timing: variants interleaved per rep (drift-free ratios), slots alternate
    std::vector<std::vector<float>> tm(vars.size());
    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0));
    CK(cudaEventCreate(&e1));
    for (int v : vars) { run(v, T, 0, n); run(v, T, 1, n); }
    for (int r = 0; r < reps; r++)
        for (size_t i = 0; i < vars.size(); i++) {
            CK(cudaEventRecord(e0));
            run(vars[i], T, r & 1, n);
            CK(cudaEventRecord(e1));
            CK(cudaEventSynchronize(e1));
            float ms;
            CK(cudaEventElapsedTime(&ms, e0, e1));
            tm[i].push_back(ms);
        }
    const double bytes = (double) L_ctx * NKV * 2 * KVQ_ROW;
    for (size_t i = 0; i < vars.size(); i++) {
        std::vector<float> t = tm[i];
        std::sort(t.begin(), t.end());
        const double med = t[t.size() / 2];
        printf("variant %d: min %.1f us  median %.1f us  (%.0f GB/s at the median, %.2f us per tile)\n", vars[i], t[0] * 1e3,
               med * 1e3, bytes / (med * 1e-3) / 1e9, med * 1e3 / (cs / ATP));
    }
    return fails ? 1 : 0;
}
