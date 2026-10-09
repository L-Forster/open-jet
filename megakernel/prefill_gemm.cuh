// Batched (prefill) GEMM on fp16 tensor cores against the k-quant weights in their stored layout.
//
//   Y[tok][row] (+)= sum_k W[row][k] * X[tok][k]
//
// X is fp16 [tok][K] with the elements of every 16-group permuted (hperm) so that the 4-byte weight
// words give m16n8k16 A fragments directly.  W stays Q4_K / Q5_K (GGUF layout), Q6_K (224-byte
// repack, see qwen38_mk.cu) or Q8_0 (unit repack) and is dequantised to fp16 in registers.
// Tile: 128 weight rows x 128 tokens per CTA, 8 warps of 16 rows x 128 tokens.  K is staged 64 at a
// time (one 64-group of a super-block: two 32-element sub-blocks) through a cp.async pipeline with
// one block barrier per stage; per-row sub-block scales are decoded one stage ahead.  Products
// accumulate in fp16 over two stages (k = 128) and are then added to fp32 accumulators.
#pragma once

#include <cuda_fp16.h>
#include <cstdint>
#include <type_traits>

namespace pf {

constexpr int BM = 128, BN = 128, BK = 64;
constexpr int TQ4K = 12, TQ5K = 13, TQ6K = 14, TQ80 = 8;

struct WMat { // device view of one weight matrix (mirrors Mat in qwen38_mk.cu)
    const uint8_t * w;
    const __half * d;
    int type, K, N, rb, nd;
};

// Raw bytes of one stage (64-group p of super-block sb) per weight row, copied as 16 B chunks:
//   Q4_K: [hdr 16 | qs 32]                    src: blk + {0, 16 + 32p}
//   Q5_K: [hdr 16 | qh 32 | qs 32]            src: blk + {0, 16, 32, 48 + 32p, 64 + 32p}
//   Q6_K: [qlo 32 | qhi 16 | meta 8]          src: blk + {32p, 16 + 32p, 128 + 16p}, 8 B at 192 + 8p
//   Q8_0: [units 64 | d 4]                    src: blk + {64p + 16c}, 4 B of d at 2 * (2 * st)
template <int TYPE> struct WT;
template <> struct WT<TQ4K> { static constexpr int BLK = 144, NCH = 3; };
template <> struct WT<TQ5K> { static constexpr int BLK = 176, NCH = 5; };
template <> struct WT<TQ6K> { static constexpr int BLK = 224, NCH = 3; };
template <> struct WT<TQ80> { static constexpr int BLK = 256, NCH = 4; };

template <int TYPE> __device__ __forceinline__ void chunk_src(int c, int & off, int & inc) {
    if constexpr (TYPE == TQ4K) { off = c == 0 ? 0 : 16 * c; inc = c == 0 ? 0 : 32; }
    else if constexpr (TYPE == TQ5K) { off = c < 3 ? 16 * c : 48 + 16 * (c - 3); inc = c < 3 ? 0 : 32; }
    else if constexpr (TYPE == TQ6K) { off = c < 2 ? 16 * c : 128; inc = c < 2 ? 32 : 16; }
    else { off = 16 * c; inc = 64; }
}

__device__ __forceinline__ void cpa16(unsigned s, const void * g) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(s), "l"(g) : "memory");
}
__device__ __forceinline__ void cpa8(unsigned s, const void * g) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 8;" ::"r"(s), "l"(g) : "memory");
}
__device__ __forceinline__ void cpa4(unsigned s, const void * g) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;" ::"r"(s), "l"(g) : "memory");
}
__device__ __forceinline__ void cpa_commit() { asm volatile("cp.async.commit_group;" ::: "memory"); }
template <int N> __device__ __forceinline__ void cpa_wait() { asm volatile("cp.async.wait_group %0;" ::"n"(N) : "memory"); }

__device__ __forceinline__ void ldsm_x4(unsigned (&r)[4], unsigned addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(addr));
}

constexpr unsigned M4 = 0x0F0F0F0Fu;

enum { OUT_STORE = 0, OUT_ADD = 1 };

struct Out {
    float * y; // y[tok * ldy + row]
    int ldy;
    int mode;
    __half * yh; // fused gate / up (mmh<.., true>): yh[tok * ldy + row] = silu(gate) * up, fp16 in hperm order
};

// Element e of a 16-group sits at slot hperm(e): 4 t + {0, 1} -> 2 t + {0, 1}, 4 t + {2, 3} ->
// 8 + 2 t + {0, 1}.  With that order a weight word (elements 4 t4 .. 4 t4 + 3) gives the A fragment
// registers a0 / a1 = (e0, e1) and a2 / a3 = (e2, e3) of an m16n8k16 MMA.
__host__ __device__ constexpr int hperm(int e) {
    return (e & ~15) | ((((e >> 1) & 1) << 2 | ((e >> 2) & 3)) << 1) | (e & 1);
}

template <int TYPE> struct HT;
// SR: smem bytes per weight row; SEP: fp16 scales in a separate double buffer (else in the row:
// Q6_K rows are 64 B, chunks XOR-swizzled by (row / 2) % 4, scales in the meta chunk's spare 8 B)
template <> struct HT<TQ4K> { static constexpr int SR = 48, NS = 4; static constexpr bool SEP = true; };
template <> struct HT<TQ5K> { static constexpr int SR = 80, NS = 3; static constexpr bool SEP = true; };
template <> struct HT<TQ6K> { static constexpr int SR = 64, NS = 4; static constexpr bool SEP = false; };
template <> struct HT<TQ80> { static constexpr int SR = 80, NS = 3; static constexpr bool SEP = true; };
constexpr int HCH = 2; // stages accumulated in fp16 before adding to the fp32 accumulators
constexpr int HMI = 1, HWR = 16 * HMI, HNWM = BM / HWR, HWT = BN / (8 / HNWM), HNI = HWT / 8; // warp tiles
constexpr int HM_X = BN * 128; // activation tile per stage: 128 tokens x 64 halves, 16 B chunks XOR-swizzled
template <int TYPE> __host__ __device__ constexpr int hm_stage() { return HT<TYPE>::SR * BM + HM_X; }
template <int TYPE> __host__ __device__ constexpr int hm_smem() { return HT<TYPE>::NS * hm_stage<TYPE>() + (HT<TYPE>::SEP ? 2 * 2 * BM * 8 : 0); }
static_assert(BN * (BM + 4) * 4 <= hm_smem<TQ5K>() && BN * (BM + 4) * 4 <= hm_smem<TQ4K>(), "epilogue tile");

__device__ __forceinline__ void hmma16(unsigned (&d)[2], const unsigned (&a)[4], unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};"
                 : "+r"(d[0]), "+r"(d[1])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void hmma16z(unsigned (&d)[2], const unsigned (&a)[4], unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%8,%8};"
                 : "=r"(d[0]), "=r"(d[1])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(0));
}
__device__ __forceinline__ __half2 u2h(unsigned u) { return *reinterpret_cast<__half2 *>(&u); }
__device__ __forceinline__ unsigned h2u(__half2 h) { return *reinterpret_cast<unsigned *>(&h); }

// GLU: W2 holds the up matrix (same type and shape as W = gate); the CTA computes 64 gate rows
// (warps 0-3) and the same 64 up rows (warps 4-7) and writes silu(gate) * up as fp16.
template <int TYPE, bool GLU = false>
__global__ void __launch_bounds__(256, 1) mmh(WMat W, const __half * X, int ntok, Out O, const uint8_t * W2 = nullptr) {
    static_assert(!GLU || TYPE == TQ4K, "fused gate / up is only built for Q4_K");
    extern __shared__ __align__(16) uint8_t sm[];
    constexpr int BLK = WT<TYPE>::BLK, NCH = WT<TYPE>::NCH, SR = HT<TYPE>::SR, NSH = HT<TYPE>::NS;
    constexpr int STG = hm_stage<TYPE>();
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, t4 = lane & 3;
    const int wm = warp % HNWM, wn = warp / HNWM; // warp tile: rows HWR wm .., tokens HWT wn ..
    const int ntt = gridDim.y;
    const int lin = blockIdx.y * gridDim.x + blockIdx.x;
    const int row0 = (lin / ntt) * (GLU ? BM / 2 : BM), tok0 = (lin % ntt) * BN;
    const unsigned sbase = (unsigned) __cvta_generic_to_shared(sm);
    uint2 * sws = reinterpret_cast<uint2 *>(sm + NSH * STG); // [2][2 j][BM] fp16 scale pairs

    constexpr int WCH = BM * NCH, WIT = (WCH + 255) / 256;
    const uint8_t * wsrc[WIT];
    int winc[WIT];
    unsigned wdst[WIT];
#pragma unroll
    for (int k = 0; k < WIT; k++) {
        const int i = tid + k * 256, r = i / NCH, c = i % NCH;
        int off, inc;
        chunk_src<TYPE>(c, off, inc);
        const int rr = i < WCH ? r : 0;
        wsrc[k] = GLU ? (rr < BM / 2 ? W.w : W2) + (size_t) (row0 + (rr & (BM / 2 - 1))) * W.rb + off
                      : W.w + (size_t) (row0 + rr) * W.rb + off;
        winc[k] = inc;
        wdst[k] = r * SR + 16 * (TYPE == TQ6K ? c ^ ((r >> 1) & 3) : c);
    }
    const uint8_t * esrc = W.w + (size_t) (row0 + (tid & (BM - 1))) * W.rb + 192;
    const __half * dsrc = W.d + (size_t) (row0 + (tid & (BM - 1))) * W.nd;
    const unsigned edst = (tid & (BM - 1)) * SR + (TYPE == TQ6K ? 16 * (3 ^ (((tid & (BM - 1)) >> 1) & 3)) : 64);
    const __half * xsrc[4];
    unsigned xdst[4];
#pragma unroll
    for (int k = 0; k < 4; k++) {
        const int i = tid + k * 256, t = i >> 3, c = i & 7;
        xsrc[k] = X + (size_t) (tok0 + t) * W.K + 8 * c;
        xdst[k] = SR * BM + t * 128 + ((c ^ (t & 7)) << 4);
    }

    auto load = [&](int st) {
        const unsigned s = sbase + (st % NSH) * STG;
        const int sb = st >> 2, p = st & 3;
#pragma unroll
        for (int k = 0; k < WIT; k++)
            if (WCH % 256 == 0 || tid + k * 256 < WCH) cpa16(s + wdst[k], wsrc[k] + sb * BLK + p * winc[k]);
        if constexpr (TYPE == TQ6K) {
            if (tid < BM) cpa8(s + edst, esrc + sb * BLK + 8 * p);
        } else if constexpr (TYPE == TQ80) {
            if (tid < BM) cpa4(s + edst, dsrc + 2 * st);
        }
#pragma unroll
        for (int k = 0; k < 4; k++) cpa16(s + xdst[k], xsrc[k] + BK * st);
    };

    // fp16 scales of stage st -> sws[st & 1][j][row] (thread = (row, j)):
    //   Q4_K / Q5_K: (S, S, -M, -M);  Q6_K: (S_lo, S_lo, S_hi, S_hi);  Q8_0: (d, d, d, d)
    auto scales = [&](int st) {
        const uint8_t * row = sm + (st % NSH) * STG + (tid >> 1) * SR;
        const int j = tid & 1, p = st & 3;
        __half2 a, b;
        if constexpr (TYPE == TQ4K || TYPE == TQ5K) {
            const int4 hd = *reinterpret_cast<const int4 *>(row);
            const float2 dm = __half22float2(*reinterpret_cast<const __half2 *>(&hd.x));
            const int sh = (p & 1) * 16 + 8 * j;
            const unsigned y = (unsigned) hd.y >> sh, z = (unsigned) hd.z >> sh, w = (unsigned) hd.w >> sh;
            unsigned sc, m;
            if (p < 2) {
                sc = y & 63;
                m = z & 63;
            } else {
                sc = (w & 0xF) | ((y >> 2) & 0x30);
                m = ((w >> 4) & 0xF) | ((z >> 2) & 0x30);
            }
            a = __float2half2_rn(dm.x * (float) sc);
            b = __float2half2_rn(-dm.y * (float) m);
        } else if constexpr (TYPE == TQ6K) {
            uint8_t * mrow = const_cast<uint8_t *>(row) + 16 * (3 ^ (((tid >> 1) >> 1) & 3));
            const int2 meta = *reinterpret_cast<const int2 *>(mrow);
            const float d = __half2float(__ushort_as_half((unsigned short) (meta.y & 0xffff)));
            const unsigned sp = (unsigned) meta.x >> (16 * j);
            *reinterpret_cast<__half2 *>(mrow + 8 + 4 * j) =
                __floats2half2_rn(d * (float) (int8_t) (sp & 0xff), d * (float) (int8_t) ((sp >> 8) & 0xff));
            return;
        } else {
            const __half dd = *reinterpret_cast<const __half *>(row + 64 + 2 * j);
            a = b = __half2half2(dd);
        }
        if constexpr (HT<TYPE>::SEP) sws[(st & 1) * 2 * BM + j * BM + (tid >> 1)] = make_uint2(h2u(a), h2u(b));
    };

    float acc[HMI][HNI][4];
#pragma unroll
    for (int m = 0; m < HMI; m++)
#pragma unroll
        for (int i = 0; i < HNI; i++) acc[m][i][0] = acc[m][i][1] = acc[m][i][2] = acc[m][i][3] = 0.f;

    const int nst = W.K / BK;
#pragma unroll
    for (int s = 0; s < NSH - 1; s++) {
        if (s < nst) load(s);
        cpa_commit();
    }
    cpa_wait<NSH - 2>();
    __syncthreads();
    scales(0);

    // ldmatrix rows: lanes 0-7 / 8-15 / 16-23 / 24-31 -> tokens +0..7 (slots 0-7) / +0..7 (slots 8-15)
    // / +8..15 (slots 0-7) / +8..15 (slots 8-15) of a 16-token pair of n8 tiles
    const unsigned xl = SR * BM + (HWT * wn + 8 * (lane >> 4) + (lane & 7)) * 128;
    unsigned xo[4];
#pragma unroll
    for (int G = 0; G < 4; G++) xo[G] = xl + (((2 * G + ((lane >> 3) & 1)) ^ (lane & 7)) << 4);

    // integer-valued fp16 weights (offset removed) of a stage: vq[mi][j][kk][0..3], A-fragment order
    unsigned vq[HMI][2][2][4];
    auto rawq = [&](int st) {
        const uint8_t * sw = sm + (st % NSH) * STG;
        const int p = st & 3;
        constexpr float OFF = TYPE == TQ6K ? -1056.f : TYPE == TQ80 ? -1152.f : -1024.f;
        const __half2 off2 = __float2half2_rn(OFF);
#pragma unroll
        for (int mi = 0; mi < HMI; mi++) {
        const int rw = HWR * wm + 16 * mi + g;
        const uint8_t * ra = sw + rw * SR;
        const uint8_t * rb = ra + 8 * SR;
        // weight words: wd[j][kk][r] = 4 values of elements 16 kk + 4 t4 .. of sub-block j, row g + 8 r
        unsigned wd[2][2][2];
        if constexpr (TYPE == TQ4K || TYPE == TQ5K) {
            constexpr int QO = TYPE == TQ4K ? 16 : 48;
            const unsigned w0 = *reinterpret_cast<const unsigned *>(ra + QO + 4 * t4);
            const unsigned w1 = *reinterpret_cast<const unsigned *>(ra + QO + 16 + 4 * t4);
            const unsigned w2 = *reinterpret_cast<const unsigned *>(rb + QO + 4 * t4);
            const unsigned w3 = *reinterpret_cast<const unsigned *>(rb + QO + 16 + 4 * t4);
            wd[0][0][0] = w0 & M4; wd[0][0][1] = w2 & M4; wd[0][1][0] = w1 & M4; wd[0][1][1] = w3 & M4;
            wd[1][0][0] = (w0 >> 4) & M4; wd[1][0][1] = (w2 >> 4) & M4;
            wd[1][1][0] = (w1 >> 4) & M4; wd[1][1][1] = (w3 >> 4) & M4;
            if constexpr (TYPE == TQ5K) {
                const unsigned h0 = *reinterpret_cast<const unsigned *>(ra + 16 + 4 * t4) >> (2 * p);
                const unsigned h1 = *reinterpret_cast<const unsigned *>(ra + 32 + 4 * t4) >> (2 * p);
                const unsigned h2 = *reinterpret_cast<const unsigned *>(rb + 16 + 4 * t4) >> (2 * p);
                const unsigned h3 = *reinterpret_cast<const unsigned *>(rb + 32 + 4 * t4) >> (2 * p);
                wd[0][0][0] |= (h0 & 0x01010101u) << 4; wd[0][0][1] |= (h2 & 0x01010101u) << 4;
                wd[0][1][0] |= (h1 & 0x01010101u) << 4; wd[0][1][1] |= (h3 & 0x01010101u) << 4;
                wd[1][0][0] |= (h0 & 0x02020202u) << 3; wd[1][0][1] |= (h2 & 0x02020202u) << 3;
                wd[1][1][0] |= (h1 & 0x02020202u) << 3; wd[1][1][1] |= (h3 & 0x02020202u) << 3;
            }
        } else if constexpr (TYPE == TQ6K) {
            const int z = (g >> 1) & 3; // rows g and g + 8 share the swizzle
            const unsigned w0 = *reinterpret_cast<const unsigned *>(ra + 16 * z + 4 * t4);
            const unsigned w1 = *reinterpret_cast<const unsigned *>(ra + 16 * (1 ^ z) + 4 * t4);
            const unsigned w2 = *reinterpret_cast<const unsigned *>(rb + 16 * z + 4 * t4);
            const unsigned w3 = *reinterpret_cast<const unsigned *>(rb + 16 * (1 ^ z) + 4 * t4);
            const uint4 ha = *reinterpret_cast<const uint4 *>(ra + 16 * (2 ^ z));
            const uint4 hb = *reinterpret_cast<const uint4 *>(rb + 16 * (2 ^ z));
            const int s = 2 * t4;
            auto mk = [](unsigned lo4, unsigned hi2) { return lo4 | ((hi2 & 0x03030303u) << 4); }; // 0..63
            wd[0][0][0] = mk(w0 & M4, ha.x >> s);
            wd[0][0][1] = mk(w2 & M4, hb.x >> s);
            wd[0][1][0] = mk(w1 & M4, ha.z >> s);
            wd[0][1][1] = mk(w3 & M4, hb.z >> s);
            wd[1][0][0] = mk((w0 >> 4) & M4, ha.y >> s);
            wd[1][0][1] = mk((w2 >> 4) & M4, hb.y >> s);
            wd[1][1][0] = mk((w1 >> 4) & M4, ha.w >> s);
            wd[1][1][1] = mk((w3 >> 4) & M4, hb.w >> s);
        } else {
#pragma unroll
            for (int j = 0; j < 2; j++)
#pragma unroll
                for (int kk = 0; kk < 2; kk++) {
                    wd[j][kk][0] = *reinterpret_cast<const unsigned *>(ra + 16 * j + 32 * kk + 4 * t4) ^ 0x80808080u;
                    wd[j][kk][1] = *reinterpret_cast<const unsigned *>(rb + 16 * j + 32 * kk + 4 * t4) ^ 0x80808080u;
                }
        }
#pragma unroll
        for (int j = 0; j < 2; j++)
#pragma unroll
            for (int kk = 0; kk < 2; kk++)
#pragma unroll
                for (int r = 0; r < 2; r++)
#pragma unroll
                    for (int hh = 0; hh < 2; hh++)
                        vq[mi][j][kk][2 * hh + r] =
                            h2u(__hadd2(u2h(__byte_perm(wd[j][kk][r], 0x64646464u, hh ? 0x4342 : 0x4140)), off2));
        }
    };
    unsigned ch[HMI][HNI][2];
    auto step = [&](int st, auto ph) {
        constexpr int PH = decltype(ph)::value;
        cpa_wait<NSH - 3>();
        __syncthreads();

        const uint8_t * sw = sm + (st % NSH) * STG;
        const unsigned sx = sbase + (st % NSH) * STG;
        const uint2 * ws = sws + (st & 1) * 2 * BM;
        const int p = st & 3;
        rawq(st);
        unsigned a[HMI][2][2][4]; // fp16 A fragments [mi][j][kk][0..3]
#pragma unroll
        for (int mi = 0; mi < HMI; mi++) {
        const int rw = HWR * wm + 16 * mi + g;
        const uint8_t * ra = sw + rw * SR;
        const uint8_t * rb = ra + 8 * SR;
#pragma unroll
        for (int j = 0; j < 2; j++) {
            uint2 sA, sB;
            if constexpr (HT<TYPE>::SEP) {
                sA = ws[j * BM + rw];
                sB = ws[j * BM + rw + 8];
            } else { // Q6_K: (S_j0, S_j1) in the row -> broadcast pairs
                const int z = (g >> 1) & 3;
                const __half2 qa = *reinterpret_cast<const __half2 *>(ra + 16 * (3 ^ z) + 8 + 4 * j);
                const __half2 qb = *reinterpret_cast<const __half2 *>(rb + 16 * (3 ^ z) + 8 + 4 * j);
                sA = make_uint2(h2u(__low2half2(qa)), h2u(__high2half2(qa)));
                sB = make_uint2(h2u(__low2half2(qb)), h2u(__high2half2(qb)));
            }
#pragma unroll
            for (int kk = 0; kk < 2; kk++)
#pragma unroll
                for (int r = 0; r < 2; r++) {
                    const uint2 sc = r ? sB : sA;
#pragma unroll
                    for (int hh = 0; hh < 2; hh++) {
                        const __half2 v = u2h(vq[mi][j][kk][2 * hh + r]);
                        __half2 o;
                        if constexpr (TYPE == TQ4K || TYPE == TQ5K) o = __hfma2(v, u2h(sc.x), u2h(sc.y));
                        else if constexpr (TYPE == TQ6K) o = __hmul2(v, u2h(kk ? sc.y : sc.x));
                        else o = __hmul2(v, u2h(sc.x));
                        a[mi][j][kk][2 * hh + r] = h2u(o);
                    }
                }
        }
        }
#pragma unroll
        for (int G = 0; G < 4; G++) {
            const int j = G >> 1, kk = G & 1;
#pragma unroll
            for (int np = 0; np < HNI / 2; np++) {
                unsigned r[4];
                ldsm_x4(r, sx + xo[G] + np * 16 * 128);
#pragma unroll
                for (int mi = 0; mi < HMI; mi++) {
                    if (G == 0 && (PH == 0 || HCH == 1)) {
                        hmma16z(ch[mi][2 * np], a[mi][j][kk], r[0], r[1]);
                        hmma16z(ch[mi][2 * np + 1], a[mi][j][kk], r[2], r[3]);
                    } else {
                        hmma16(ch[mi][2 * np], a[mi][j][kk], r[0], r[1]);
                        hmma16(ch[mi][2 * np + 1], a[mi][j][kk], r[2], r[3]);
                    }
                }
            }
            // next stages' copies and scales are issued once the tensor pipe has work queued
            if (G == 0) {
                if (st + NSH - 1 < nst) load(st + NSH - 1);
                cpa_commit();
            }
            if (G == 1 && st + 1 < nst) scales(st + 1);
        }
        if (PH == HCH - 1 || HCH == 1)
#pragma unroll
        for (int mi = 0; mi < HMI; mi++)
#pragma unroll
        for (int ni = 0; ni < HNI; ni++) {
            const float2 lo = __half22float2(u2h(ch[mi][ni][0])), hi = __half22float2(u2h(ch[mi][ni][1]));
            acc[mi][ni][0] += lo.x; acc[mi][ni][1] += lo.y; acc[mi][ni][2] += hi.x; acc[mi][ni][3] += hi.y;
        }
    };
    for (int st = 0; st < nst; st += 2) { // nst is even for every K of the model
        step(st, std::integral_constant<int, 0>());
        step(st + 1, std::integral_constant<int, 1>());
    }
    cpa_wait<0>();
    __syncthreads();
    constexpr int ELD = BM + 4;
    float * se = reinterpret_cast<float *>(sm);
#pragma unroll
    for (int mi = 0; mi < HMI; mi++)
#pragma unroll
    for (int ni = 0; ni < HNI; ni++) {
        const int r = HWR * wm + 16 * mi + g, tk = HWT * wn + 8 * ni + 2 * t4;
        se[tk * ELD + r] = acc[mi][ni][0];
        se[(tk + 1) * ELD + r] = acc[mi][ni][1];
        se[tk * ELD + r + 8] = acc[mi][ni][2];
        se[(tk + 1) * ELD + r + 8] = acc[mi][ni][3];
    }
    __syncthreads();
    if constexpr (GLU) {
        for (int i = tid; i < BN * 4; i += 256) { // (token, 16-group of the 64 outputs)
            const int tk = i >> 2, q = i & 3;
            if (tok0 + tk >= ntok) continue;
            const float * gp = se + tk * ELD + 16 * q;
            __align__(16) __half hv[16];
#pragma unroll
            for (int e = 0; e < 16; e++) {
                const float gv = gp[e], v = gv / (1.0f + expf(-gv)) * gp[e + BM / 2];
                hv[hperm(e)] = __float2half_rn(fminf(fmaxf(v, -65504.f), 65504.f));
            }
            uint4 * dst = reinterpret_cast<uint4 *>(O.yh + (size_t) (tok0 + tk) * O.ldy + row0 + 16 * q);
            dst[0] = reinterpret_cast<const uint4 *>(hv)[0];
            dst[1] = reinterpret_cast<const uint4 *>(hv)[1];
        }
        return;
    }
    for (int i = tid; i < BN * BM / 4; i += 256) {
        const int tk = i / (BM / 4), c = (i % (BM / 4)) * 4;
        if (tok0 + tk >= ntok) continue;
        float4 v = *reinterpret_cast<const float4 *>(se + tk * ELD + c);
        float4 * dst = reinterpret_cast<float4 *>(O.y + (size_t) (tok0 + tk) * O.ldy + row0 + c);
        if (O.mode == OUT_ADD) {
            const float4 o = __ldcg(dst);
            v.x += o.x; v.y += o.y; v.z += o.z; v.w += o.w;
        }
        *dst = v;
    }
}

} // namespace pf
