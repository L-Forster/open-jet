// Batched (prefill) GEMM on fp16 tensor cores against the k-quant weights in their stored layout.
//
//   Y[tok][row] (+)= sum_k W[row][k] * X[tok][k]
//
// X is fp16 [tok][K] with the elements of every 16-group permuted (hperm) so that the 4-byte weight
// words give m16n8k16 A fragments directly.  W stays Q4_K / Q5_K (GGUF layout), Q6_K (210-byte
// two-stream repack: layout in qwen38_mk.cu, made by loader.cu), Q8_0 (unit repack), IQ4_XS (272-byte block pairs,
// indices in Q4_K order) or IQ4_NL (Q4_K-shaped 144-byte repack) and is dequantised to fp16 in registers.
// Tile: 128 weight rows x 128 tokens per CTA, 8 warps of 16 rows x 128 tokens.  K is staged 64 at a
// time (one 64-group of a super-block: two 32-element sub-blocks) through a cp.async pipeline with
// one block barrier per stage (Q5_K / IQ4_NL: per two stages, see pf_wown); per-row sub-block scales
// are decoded one stage ahead.  Products accumulate in fp16 over two stages (k = 128) and are then
// added to fp32 accumulators.
#pragma once

#include <cuda_fp16.h>
#include <cstdint>
#include <type_traits>

#ifndef MK_PF_FRAG_G
#define MK_PF_FRAG_G -1 // (tuning) the MMA group of a stage after which the next stage's scales + fragments are made; -1: frag_g
#endif

namespace pf {

#if MK_PF_INFCHK
// MK_PF_INFCHK=1: event counters per (layer, site, kind), printed and cleared after each batched prefill. GEMM sites
// (the matrix, set by the host before each launch): kind 0 = fp16 partial sums (two 64-k stages accumulated in fp16 by
// the MMAs) that are inf / NaN when added to the fp32 accumulators, kind 1 = non-finite fp32 outputs. Input sites (the
// fp16 stores of GEMM inputs, saturated to +-65504): kind 0 = |v| > 65504 (inf included), kind 1 = NaN (stored as
// -65504 by the clamp).
__device__ unsigned long long chk_cnt[CHK_LAYERS][CHK_SITES][2];
__device__ int chk_layer, chk_site;
__device__ __forceinline__ void chk_add(int site, int kind, unsigned n) {
    if (n) atomicAdd(&chk_cnt[min(max(chk_layer, 0), CHK_LAYERS - 1)][site][kind], (unsigned long long) n);
}
__device__ __forceinline__ void chk_clamp(float v, int site) { // v: about to be stored as fp16 with the +-65504 clamp
    if (v != v) chk_add(site, 1, 1);
    else if (fabsf(v) > 65504.f) chk_add(site, 0, 1);
}
#endif

// kvalues_iq4nl + 128 as bytes (the Q8_0 path's offset)
constexpr unsigned IQ4U_T0 = 0x3F2D1801u, IQ4U_T1 = 0x766A5D4Fu, IQ4U_T2 = 0xA6998D81u, IQ4U_T3 = 0xF1D9C5B5u;

// Raw bytes of one stage (64-group p of super-block sb) per weight row, copied as 16 B chunks:
//   Q4_K: [hdr 16 | qs 32]                    src: blk + {0, 16 + 32p}
//   Q5_K: [hdr 16 | qh 32 | qs 32]            src: blk + {0, 16, 32, 48 + 32p, 64 + 32p}
//   Q6_K: [qlo 32 | qhi 16 | meta 8]          src: blk + {32p, 16 + 32p, 128 + 16p}, 8 B at 192 + 8p
//   Q8_0: [units 64 | d 4]                    src: blk + {64p + 16c}, 4 B of d at 2 * (2 * st)
//   IQ4_XS, IQ4_NL: as Q4_K
template <int TYPE> struct WT;
template <> struct WT<TQ4K> { static constexpr int BLK = 144, NCH = 3; };
template <> struct WT<TIQ4> { static constexpr int BLK = 136, NCH = 3; }; // in 272 B pairs, see mmh's load
template <> struct WT<TIQ4NL> { static constexpr int BLK = 144, NCH = 3; };
template <> struct WT<TQ5K> { static constexpr int BLK = 176, NCH = 2; }; // qs; header + qh: SBT
template <> struct WT<TQ6K> { static constexpr int BLK = 192, NCH = 3; }; // + the meta stream (W.d), see load
template <> struct WT<TQ80> { static constexpr int BLK = 256, NCH = 4; };
// native Q3_K / IQ3_S (rows_q3k / rows_iq3s): the 16 B of index / low bits of the stage's two units; the rest (Q3_K: high
// bits 8 B + Q6_K-style meta 4 + 4 B; IQ3_S: signs 8 B + the block's 16 B meta) is copied by the first BM threads, see load
template <> struct WT<TQ3K> { static constexpr int BLK = 96, NCH = 1; };
template <> struct WT<TIQ3S> { static constexpr int BLK = 96, NCH = 1; };

template <int TYPE> __device__ __forceinline__ void chunk_src(int c, int & off, int & inc) {
    if constexpr (TYPE == TQ3K || TYPE == TIQ3S) { off = 0; inc = 16; }
    else if constexpr (TYPE == TQ4K || TYPE == TIQ4 || TYPE == TIQ4NL) { off = c == 0 ? 0 : 16 * c; inc = c == 0 ? 0 : 32; }
    else if constexpr (TYPE == TQ5K) { off = 48 + 16 * c; inc = 32; } // (the stage holds qs only; header + qh: SBT)
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
// mbarriers (sm_80): init, arrive, the asynchronous arrive of a thread's prior cp.async copies, wait for a phase (test_wait)
__device__ __forceinline__ void mb_init(unsigned a, unsigned n) { asm volatile("mbarrier.init.shared.b64 [%0], %1;" ::"r"(a), "r"(n) : "memory"); }
__device__ __forceinline__ void mb_arrive(unsigned a) {
    asm volatile("{\n.reg .b64 st;\nmbarrier.arrive.shared.b64 st, [%0];\n}" ::"r"(a) : "memory");
}
__device__ __forceinline__ void cpa_mb_arrive(unsigned a) { asm volatile("cp.async.mbarrier.arrive.noinc.shared.b64 [%0];" ::"r"(a) : "memory"); }
__device__ __forceinline__ void mb_wait(unsigned a, unsigned parity) {
    asm volatile("{\n.reg .pred p;\nWAIT:\nmbarrier.test_wait.parity.shared.b64 p, [%0], %1;\n@!p bra WAIT;\n}" ::"r"(a), "r"(parity) : "memory");
}
template <int N> __device__ __forceinline__ void cpa_wait() { asm volatile("cp.async.wait_group %0;" ::"n"(N) : "memory"); }

__device__ __forceinline__ void ldsm_x4(unsigned (&r)[4], unsigned addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(addr));
}

constexpr unsigned M4 = 0x0F0F0F0Fu;

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
#ifndef MK_PF_NS_IQ4
#define MK_PF_NS_IQ4 4 // (experiment) cp.async stages of the IQ4_XS kernels
#endif
template <> struct HT<TIQ4> { static constexpr int SR = 48, NS = MK_PF_NS_IQ4; static constexpr bool SEP = true; };
template <> struct HT<TIQ4NL> { static constexpr int SR = 48, NS = 4; static constexpr bool SEP = true; };
template <> struct HT<TQ5K> { static constexpr int SR = 32, NS = 4; static constexpr bool SEP = true; }; // + SBT
template <> struct HT<TQ6K> { static constexpr int SR = 64, NS = 4; static constexpr bool SEP = false; };
template <> struct HT<TQ80> { static constexpr int SR = 80, NS = 3; static constexpr bool SEP = true; };
// Q3_K row: [lows 16 | highs 8 | sc 4 | d pair 4 | pad 16]; IQ3_S row: [index lows 16 | meta 16 | signs 8 | pad 8]
// (48 B rows: the 8 row groups g of a fragment load hit distinct banks, as Q4_K's)
template <> struct HT<TQ3K> { static constexpr int SR = 48, NS = 4; static constexpr bool SEP = true; };
template <> struct HT<TIQ3S> { static constexpr int SR = 48, NS = 4; static constexpr bool SEP = true; };
// Per-super-block bytes that the 4 stages of a super-block share (Q5_K: header 16 + qh 32 B per row) go to a double buffer
// (SBB, by super-block parity) once per super-block instead of into every stage: Q5_K's stage is then 32 B per row, so it
// gets 4 stages like the 48-B types. R: bytes per row, NC: 16-B chunks per row.
template <int TYPE> struct SBT { static constexpr int R = 0, NC = 0; };
template <> struct SBT<TQ5K> { static constexpr int R = 48, NC = 3; };
constexpr int HCH = 2; // stages accumulated in fp16 before adding to the fp32 accumulators
constexpr int HMI = 1, HWR = 16 * HMI, HNWM = BM / HWR; // warp tiles: 8 warps of 16 rows, each over all NB tokens
// NB: tokens per CTA (BN = 128 or, for short prompts, 64 / 32; picked per GEMM by the host, see Impl::pf_nb).
// Each output element is the same sequence of fp16 products and sums in any NB (the K order per row and token does
// not depend on which other tokens share the CTA), so every NB gives bit-identical results.
template <int NB> __host__ __device__ constexpr int hm_x() { return NB * 128; } // activation tile per stage: NB tokens x 64 halves, 16 B chunks XOR-swizzled
template <int TYPE, int NB = BN> __host__ __device__ constexpr int hm_stage() { return HT<TYPE>::SR * BM + hm_x<NB>(); }
// (A deeper cp.async pipeline for the narrow tiles, up to 8 stages, is no faster.)
// At NB = 32 every type fits two CTAs per SM (<= 49,216 B smem, Q6_K's; <= 96 registers), at NB = 64 / 128 one.
// Per-stage mbarriers (MBL) instead of the per-stage CTA barrier: full[b] completes when the 256 threads' copies of the
// stage in buffer b have landed (cp.async.mbarrier.arrive), empty[b] when the 8 warps are done reading it; the copies run
// two stages ahead (not three), so a warp waits for the others only to reuse the buffer of two stages back. Per
// kernel: faster for Q6_K, Q3_K, IQ3_S and these mixed-type gate / up pairs, slower for Q4_K, IQ4_XS and Q8_0
// (Q5_K / IQ4_NL: pf_wown). Needs 4 stages; MBB bytes at the end of the dynamic shared memory.
__host__ __device__ constexpr bool pf_mbl(int tg, int tu, bool mix) {
    if (!mix) return tg == TQ6K || tg == TQ3K || tg == TIQ3S;
    return (tg == TQ4K && (tu == TQ5K || tu == TIQ4)) || (tg == TIQ4 && (tu == TQ4K || tu == TQ3K)) ||
           (tg == TQ6K && (tu == TQ6K || tu == TQ5K)) || (tg == TQ3K && tu == TIQ4);
}
constexpr int MBB = 64; // MBL: full[4] + empty[4]
template <int TYPE, int NB = BN> __host__ __device__ constexpr int hm_smem() {
    return HT<TYPE>::NS * hm_stage<TYPE, NB>() + 2 * SBT<TYPE>::R * BM + (HT<TYPE>::SEP ? 2 * 2 * BM * 8 : 0) +
           (pf_mbl(TYPE, TYPE, false) ? MBB : 0);
}

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
// (warps 0-3) and the same 64 up rows (warps 4-7) and writes silu(gate) * up as fp16. Only for
// types whose rows carry their own scales (Q6_K / Q8_0 read them through W alone).
// Mixed-type GLU (V2: TYPE2 >= 0, the 27 UD layers whose gate / up types or layouts differ, incl. FM Q6_K / FM Q6_K):
// warps 0-3 compute the 64 gate rows of W (TYPE, LAY), warps 4-7 the 64 up rows of Wu (TYPE2, LAY2), each half with its
// own stage layout (gate's weight bytes, then up's), copies, scales and fragments; the products, sums and the epilogue
// are those of mmh<TYPE> + mmh<TYPE2> into fp32 + pk_glu: bit-identical.
// LAY 1: W (and W2) are stored in the wv layout of wv.cuh: units (16-row tile, super-block) of 16 x UB bytes.
// Q4_K / Q5_K / IQ4_XS are fragment-ordered there (FMT: a 16-B chunk of qh / qs holds 8 B of rows g and g + 8), IQ4_NL /
// Q8_0 (and Q8_0's scales) are tile-major whole blocks (shared memory receives the same bytes as from the row-major layout).
// FMT stage layout: the FM chunks land whole (one cp.async 16 each): per 16-row tile a 512-B region of qs (Q5_K: and qh)
// chunks, chunk (g, tq) at fmt_off = 128 (2 (g & 1) + (tq >> 1)) + 16 (2 (g >> 1) + (tq & 1)), then the headers (16 B per
// row; IQ4_XS: 16 B per row pair g / g + 8). rawq's 32-bit reads are bank-conflict-free because odd-g lanes read their
// chunks' second half (row g + 8) as their "row A": the lane's MMA rows are g + 8 (g & 1) and the other one, which the
// scales and the epilogue follow (the same products and sums per row: exact).
// FM6: Q6_K in wv.cuh's FM Q6_K units (3360 B per tile and super-block, meta inline). Stage layout (the
// weight part, 64 B x ROWS): qlo of tile t at F6L + 512 t (the unit's group-p block, whole), qhi at F6H + 256 t (likewise),
// per row 16 B at F6M + 16 r: sc[4p .. 4p + 3] | d of rows g / g + 8 | the fp16 scale pairs (scales). The unit's qlo / qhi
// order is the thread order of rawq (lane l = 4 g + t4: rows g / g + 8, elements 4 t4 .. + 3 of each 16-chunk), so rawq
// reads one 16-B and one 8-B word and builds the same wd words as from the row-major stage: exact.

// The MMA group (0-3) of a stage after which the next stage's scales + fragments are made (pipelined step): tuned per
// kernel on the microbenchmark (launcher layouts, 1024 tokens).
__host__ __device__ constexpr int frag_g(int tg, int tu, bool glu, bool mix) {
    if (MK_PF_FRAG_G >= 0) return MK_PF_FRAG_G;
    if (!mix) return tg == TQ5K || tg == TQ80 || (tg == TIQ4 && !glu) || (tg == TQ4K && glu) ? 1 : 2;
    return (tg == TIQ4 && (tu == TQ5K || tu == TQ4K)) || tg == TIQ4NL || tu == TIQ4NL || (tg == TQ6K && tu == TQ6K) ? 2 : 1;
}
// Warp-owned weight copies (HalfW OWN): each warp copies the weight bytes of its own 16 rows (the only ones it reads), so a
// warp-level wait covers them and the CTA barrier is needed for the activation tiles only, which are then copied in pairs:
// one barrier per two stages instead of one per stage. Faster for the Q5_K and IQ4_NL GEMMs (plain and
// the Q5_K fused gate / up); slower for Q4_K, IQ4_XS, Q6_K, Q3_K, IQ3_S and the mixed-type kernels, which keep
// the per-stage barrier. Needs the 4 cp.async stages of the same-type kernels.
__host__ __device__ constexpr bool pf_wown(int t, bool mix) { return !mix && (t == TQ5K || t == TIQ4NL); }

// Mixed-type GLU tiles: both halves' weight bytes + the activations per stage; 4 stages if they fit (NB 32: in 48 KB, two
// CTAs per SM like the same-type kernels; else in 99 KB), else 3
template <int TG, int TU, int NB> __host__ __device__ constexpr int hm_stage2() { return BM / 2 * (HT<TG>::SR + HT<TU>::SR) + hm_x<NB>(); }
template <int TG, int TU> __host__ __device__ constexpr int hm_sep2() { // + the halves' super-block buffers
    return (HT<TG>::SEP || HT<TU>::SEP ? 2 * 2 * BM * 8 : 0) + 2 * BM / 2 * (SBT<TG>::R + SBT<TU>::R);
}
template <int TG, int TU, int NB> __host__ __device__ constexpr int hm_ns2() {
    return 4 * hm_stage2<TG, TU, NB>() + hm_sep2<TG, TU>() + (pf_mbl(TG, TU, true) ? MBB : 0) <= (NB <= 32 ? 48 : 99) * 1024 ? 4 : 3;
}
template <int TG, int TU, int NB> __host__ __device__ constexpr int hm_smem2() {
    return hm_ns2<TG, TU, NB>() * hm_stage2<TG, TU, NB>() + hm_sep2<TG, TU>() + (pf_mbl(TG, TU, true) ? MBB : 0);
}

// The weight rows of a CTA that share a type: all BM rows (TW2: the same-type fused gate / up, rows 0 .. BM/2 - 1 from W,
// the rest from W2), or one half (BM/2 rows) of a mixed-type gate / up. HB: the rows' offset in a stage; TB: the first
// thread of the per-row copies (scale streams, Q6_K meta, Q3_K / IQ3_S extra bytes): threads TB .. TB + ROWS - 1.
template <int TYPE, int LAY, int ROWS, bool TW2, int HB, int TB, bool OWN = false> struct HalfW {
    static constexpr bool FMT = LAY && (TYPE == TIQ4 || TYPE == TQ4K || TYPE == TQ5K); // fragment-ordered
    static constexpr bool FM6 = LAY && TYPE == TQ6K;
    static constexpr int BLK = WT<TYPE>::BLK, NCH = WT<TYPE>::NCH, SR = HT<TYPE>::SR;
    static constexpr bool SEP = HT<TYPE>::SEP;
    static constexpr int F6L = 0, F6H = 32 * ROWS, F6M = 48 * ROWS, F6U = 3360;
    // FMT stage layout (see above): Q5_K qh regions at FQH, qs regions at FQS, headers at FHO; FCPT chunks per tile and stage
    // (Q5_K: the stage holds qs only; its qh region (FQH) and headers (FHO) are in the super-block buffer, SBT)
    static constexpr int FQH = 0, FQS = 0, FHO = ROWS * 32;
    static constexpr int FNQ = 32, FCPT = FNQ + (TYPE == TQ5K ? 0 : TYPE == TIQ4 ? 8 : 16);
    static constexpr int SBR = SBT<TYPE>::R, SNC = SBT<TYPE>::NC; // super-block buffer bytes per row, its chunks per row
    static constexpr int WCHS = SNC ? (FMT ? ROWS / 16 * 48 : ROWS * SNC) : 0; // super-block buffer chunks (per super-block)
    static constexpr int WCH = FMT ? ROWS / 16 * FCPT : FM6 ? ROWS / 16 * 48 : ROWS * NCH;        // stage chunks
    // Copies: thread i copies chunk i (i = tid + 256 k), or (OWN, see pf_wown) warp WB + t copies the CPT stage chunks, the
    // CPTS super-block buffer chunks and the per-row bytes of the half's 16-row tile t (the chunk index order of the tiles is
    // contiguous in every layout: FMT / FM6 by tile, row-major by row).
    static constexpr int WB = TB / 16, CPT = WCH / (ROWS / 16), CPTS = WCHS / (ROWS / 16);
    static constexpr int WIT = OWN ? (CPT + 31) / 32 : (WCH + 255) / 256;
    static constexpr int WITS = !WCHS ? 1 : OWN ? (CPTS + 31) / 32 : (WCHS + 255) / 256;
    static constexpr int UB = FM6 ? F6U : 16 * (TYPE == TIQ4 ? 136 : BLK); // LAY: bytes per (tile, super-block) unit
    // LAY 1 offset types (ffn_down): value blocks and meta entries are tile-major, so the row's stream advances by
    // 16 entries per block (VST: value bytes per super-block, EST: meta bytes per Q6_K / Q3_K block pair or IQ3_S block)
    static constexpr int VST = LAY ? 16 * BLK : BLK, EST = LAY ? 16 * (TYPE == TIQ3S ? 16 : 36) : (TYPE == TIQ3S ? 16 : 36);
    static_assert(!FMT || SBR || FHO + ROWS * 16 <= SR * ROWS, "FMT stage layout fits the stage's weight part");
    static_assert(!FM6 || F6M + 16 * ROWS == SR * ROWS, "FM6: 48 chunks per tile and stage");
    static_assert(!LAY || TYPE == TQ4K || TYPE == TQ5K || TYPE == TIQ4 || TYPE == TIQ4NL || TYPE == TQ80 || TYPE == TQ6K ||
                      TYPE == TQ3K || TYPE == TIQ3S, "wv layout types");

    const uint8_t * wsrc[WIT];
    int winc[WIT];
    unsigned wdst[WIT];
    bool wq[WIT]; // IQ4_XS: an index chunk (not the pair's header), at + 128 in the odd block of a pair (LAY: not a header)
    const uint8_t * ssrc[WITS]; // super-block buffer chunks (SBT): source of super-block 0, offset in a buffer
    unsigned sdst[WITS];
    const uint8_t * sbbp; // this half's two super-block buffers (generic / shared address)
    unsigned sbba;
    const uint8_t * f6src, * f6d, * dsrc8, * esrc, * vsrc;
    const __half * dsrc;
    unsigned edst, rdst;
    int rt; // this thread's row for the per-row copies (valid if 0 <= rt < ROWS)
    unsigned wmask, smask; // OWN: the stage / super-block buffer copies k this thread issues

    __device__ __forceinline__ void init(int tid, const WMat & W, const uint8_t * W2, int row0, uint8_t * sbb = nullptr) {
        const int tw = (tid >> 5) - WB, wl = tid & 31; // OWN: the warp's tile in this half, its lane
        const bool town = OWN && tw >= 0 && tw < ROWS / 16;
        wmask = smask = 0u;
#pragma unroll
        for (int k = 0; k < WIT; k++) {
            int i = tid + k * 256;
            if constexpr (OWN) {
                const bool own = town && wl + 32 * k < CPT;
                i = own ? tw * CPT + wl + 32 * k : WCH;
                wmask |= (unsigned) own << k;
            }
            const int r = i / NCH, c = i % NCH;
            int off, inc;
            chunk_src<TYPE>(c, off, inc);
            const int rr = i < WCH ? r : 0;
            if constexpr (LAY) {
                const uint8_t * mb = TW2 ? (rr < ROWS / 2 ? W.w : W2) : W.w;
                const int rowm = row0 + (TW2 ? (rr & (ROWS / 2 - 1)) : rr), q16 = rowm & 15;
                const uint8_t * ub = mb + (size_t) (rowm >> 4) * (W.K >> 8) * UB;
                if constexpr (FMT) { // whole FM chunks: thread i -> (tile t, chunk cc); 8 consecutive threads -> 8 bank quads
                    const int ii = i < WCH ? i : 0, t = ii / FCPT, cc = ii % FCPT;
                    const uint8_t * mbt = TW2 ? (t < ROWS / 32 ? W.w : W2) : W.w;
                    const int trow = row0 + 16 * (TW2 ? (t & (ROWS / 32 - 1)) : t);
                    const uint8_t * ubt = mbt + (size_t) (trow >> 4) * (W.K >> 8) * UB;
                    int o2, i2 = 0, d2;
                    if (cc < FNQ) { // qs chunk (Q5_K: cc < 32 qh, then qs) of lane (gq, tq), stored at fmt_off(gq, tq)
                        const int qq = cc & 31, q = qq & 7, rs = qq >> 3, gq = 2 * (q >> 1) + (rs & 1), tq = 2 * (rs >> 1) + (q & 1);
                        o2 = (TYPE == TQ4K ? 256 : TYPE == TQ5K ? 768 : 128) + 16 * (4 * gq + tq);
                        i2 = 512; // qs of 64-group p: + 512 p
                        d2 = FQS + 512 * t + 128 * (2 * (rs & 1) + (rs >> 1)) + 16 * q;
                    } else { // header: Q4_K / Q5_K row h of the tile (FM chunk 2 (h & 7) + (h >> 3)); IQ4_XS rows h, h + 8 (chunk h)
                        const int h = cc - FNQ;
                        o2 = TYPE == TIQ4 ? 16 * h : 16 * (2 * (h & 7) + (h >> 3));
                        d2 = FHO + 16 * ((TYPE == TIQ4 ? 8 : 16) * t + h);
                    }
                    wsrc[k] = ubt + o2;
                    winc[k] = i2;
                    wdst[k] = HB + d2;
                } else if constexpr (FM6) { // thread i -> (tile t, chunk cc): qlo chunks 0..31, qhi chunks 32..47 of group p
                    const int ii = i < WCH ? i : 0, t = ii / 48, cc = ii % 48;
                    const uint8_t * ubt = W.w + (size_t) (row0 / 16 + t) * (W.K >> 8) * UB;
                    wsrc[k] = ubt + (cc < 32 ? 288 + 16 * cc : 2336 + 16 * (cc - 32));
                    winc[k] = cc < 32 ? 512 : 256;
                    wdst[k] = HB + (cc < 32 ? F6L + 512 * t + 16 * cc : F6H + 256 * t + 16 * (cc - 32));
                } else {
                    wsrc[k] = ub + q16 * BLK + off;
                    winc[k] = inc;
                }
            } else {
                wsrc[k] = TW2 ? (rr < ROWS / 2 ? W.w : W2) + (size_t) (row0 + (rr & (ROWS / 2 - 1))) * W.rb + off
                              : W.w + (size_t) (row0 + rr) * W.rb + off;
                winc[k] = inc;
            }
            wq[k] = c != 0;
            if constexpr (!FMT && !FM6) wdst[k] = HB + r * SR + 16 * (TYPE == TQ6K ? c ^ ((r >> 1) & 3) : c);
        }
        rt = !OWN ? tid - TB : town && wl < 16 ? 16 * tw + wl : -1;
        const int r = rt & (ROWS - 1);
        // FM6: the thread's row in its unit: sc at 32 g + 16 half (+ 4 p per stage), the d pair at 256 + 4 g
        f6src = f6d = dsrc8 = nullptr;
        if constexpr (FM6) {
            const int q16 = r & 15;
            const uint8_t * ur = W.w + (size_t) ((row0 + r) >> 4) * (W.K >> 8) * UB;
            f6src = ur + 32 * (q16 & 7) + 16 * (q16 >> 3);
            f6d = ur + 256 + 4 * (q16 & 7);
        }
        // LAY Q8_0: the thread's row of the tile-major scale stream (16 B per row and super-block)
        if constexpr (LAY && TYPE == TQ80) {
            const int rowm = row0 + r;
            dsrc8 = reinterpret_cast<const uint8_t *>(W.d) + ((size_t) (rowm >> 4) * (W.K >> 8) * 16 + (rowm & 15)) * 16;
        }
        // Q6_K: the row's meta stream (pairs of blocks: sc even | sc odd | d even | d odd, 36 B)
        esrc = reinterpret_cast<const uint8_t *>(W.d + (size_t) (row0 + r) * W.nd);
        dsrc = W.d + (size_t) (row0 + r) * W.nd;
        edst = HB + r * SR + (TYPE == TQ6K ? 16 * (3 ^ ((r >> 1) & 3)) : 64);
        // Q3_K / IQ3_S: the row's value stream (high bits / signs at 64 + 8 p of each block) and its smem row
        vsrc = W.w + (size_t) (row0 + r) * W.rb;
        if constexpr (LAY && !FM6 && (TYPE == TQ6K || TYPE == TQ3K || TYPE == TIQ3S)) {
            const int rowm = row0 + r, nb = W.K >> 8;
            esrc = reinterpret_cast<const uint8_t *>(W.d) + ((size_t) (rowm >> 4) * (TYPE == TIQ3S ? nb : nb / 2) * 16 + (rowm & 15)) *
                                                                (TYPE == TIQ3S ? 16 : 36);
            vsrc = W.w + ((size_t) (rowm >> 4) * nb * 16 + (rowm & 15)) * BLK;
        }
        rdst = HB + r * SR;
        // super-block buffer chunks (Q5_K: header + qh of each row; FMT: per tile 32 qh chunks to the qh region, then 16
        // header chunks), copied once per super-block (load)
        sbbp = sbb;
        sbba = sbb ? (unsigned) __cvta_generic_to_shared(sbb) : 0u;
        if constexpr (SNC > 0) {
#pragma unroll
            for (int k = 0; k < WITS; k++) {
                int i = tid + k * 256;
                if constexpr (OWN) {
                    const bool own = town && wl + 32 * k < CPTS;
                    i = own ? tw * CPTS + wl + 32 * k : WCHS;
                    smask |= (unsigned) own << k;
                }
                const int ii = i < WCHS ? i : 0;
                if constexpr (FMT) {
                    const int t = ii / 48, cc = ii % 48;
                    const uint8_t * mbt = TW2 ? (t < ROWS / 32 ? W.w : W2) : W.w;
                    const int trow = row0 + 16 * (TW2 ? (t & (ROWS / 32 - 1)) : t);
                    const uint8_t * ubt = mbt + (size_t) (trow >> 4) * (W.K >> 8) * UB;
                    if (cc < 32) { // qh chunk of lane (gq, tq), as the stage's qs chunks
                        const int q = cc & 7, rs = cc >> 3, gq = 2 * (q >> 1) + (rs & 1), tq = 2 * (rs >> 1) + (q & 1);
                        ssrc[k] = ubt + 256 + 16 * (4 * gq + tq);
                        sdst[k] = FQH + 512 * t + 128 * (2 * (rs & 1) + (rs >> 1)) + 16 * q;
                    } else { // header of row h of the tile (FM chunk 2 (h & 7) + (h >> 3))
                        const int h = cc - 32;
                        ssrc[k] = ubt + 16 * (2 * (h & 7) + (h >> 3));
                        sdst[k] = FHO + 16 * (16 * t + h);
                    }
                } else {
                    const int rr = ii / SNC, c = ii % SNC;
                    const uint8_t * mb = TW2 ? (rr < ROWS / 2 ? W.w : W2) : W.w;
                    const int rowm = row0 + (TW2 ? (rr & (ROWS / 2 - 1)) : rr);
                    ssrc[k] = LAY ? mb + (size_t) (rowm >> 4) * (W.K >> 8) * UB + (rowm & 15) * BLK + 16 * c
                                  : mb + (size_t) rowm * W.rb + 16 * c;
                    sdst[k] = rr * SBR + 16 * c;
                }
            }
        }
    }

    // stage st's weight bytes into the stage at shared address s
    __device__ __forceinline__ void load(unsigned s, int st, int tid) const {
        const int sb = st >> 2, p = st & 3;
        if constexpr (SNC > 0) // the super-block's header + qh, with its first stage (read from stage 4 sb - 1's step on)
            if (p == 0) {
                const unsigned d = sbba + (sb & 1) * SBR * ROWS;
#pragma unroll
                for (int k = 0; k < WITS; k++)
                    if (OWN ? (smask >> k) & 1u : WCHS % 256 == 0 || tid + k * 256 < WCHS)
                        cpa16(d + sdst[k], ssrc[k] + (size_t) sb * (LAY ? UB : BLK));
            }
#pragma unroll
        for (int k = 0; k < WIT; k++)
            if (OWN ? (wmask >> k) & 1u : WCH % 256 == 0 || tid + k * 256 < WCH) {
                if constexpr (LAY) { // (FMT: whole FM chunks to the FMT stage layout)
                    cpa16(s + wdst[k], wsrc[k] + (size_t) sb * UB + p * winc[k]);
                } else if constexpr (TYPE == TIQ4) // block pairs of 272 B: headers, then the even and the odd block's indices
                    cpa16(s + wdst[k], wsrc[k] + (size_t) (sb >> 1) * 272 + (wq[k] && (sb & 1) ? 128 : 0) + p * winc[k]);
                else
                    cpa16(s + wdst[k], wsrc[k] + sb * BLK + p * winc[k]);
            }
        if constexpr (FM6) { // the row's meta slot: sc[4p..4p+3] | d of the tile's rows g / g + 8 (scales picks one)
            if (rt >= 0 && rt < ROWS) {
                cpa4(s + HB + F6M + 16 * rt, f6src + (size_t) sb * UB + 4 * p);
                cpa4(s + HB + F6M + 16 * rt + 4, f6d + (size_t) sb * UB);
            }
        } else if constexpr (TYPE == TQ6K) { // meta slot: sc[4p..4p+3] | the pair's two d (scales picks one)
            if (rt >= 0 && rt < ROWS) {
                cpa4(s + edst, esrc + (sb >> 1) * EST + 16 * (sb & 1) + 4 * p);
                cpa4(s + edst + 4, esrc + (sb >> 1) * EST + 32);
            }
        } else if constexpr (TYPE == TQ80) {
            if (rt >= 0 && rt < ROWS) {
                if constexpr (LAY) cpa4(s + edst, dsrc8 + (size_t) sb * 256 + 4 * p);
                else cpa4(s + edst, dsrc + 2 * st);
            }
        } else if constexpr (TYPE == TQ3K) { // high bits of the stage's two units | sc[4p..4p+3] | the pair's two d
            if (rt >= 0 && rt < ROWS) {
                cpa8(s + rdst + 16, vsrc + sb * VST + 64 + 8 * p);
                cpa4(s + rdst + 24, esrc + (sb >> 1) * EST + 16 * (sb & 1) + 4 * p);
                cpa4(s + rdst + 28, esrc + (sb >> 1) * EST + 32);
            }
        } else if constexpr (TYPE == TIQ3S) { // the block's meta (qh[8] | scales[4] | d) | signs of the stage's two units
            if (rt >= 0 && rt < ROWS) {
                cpa16(s + rdst + 16, esrc + sb * EST);
                cpa8(s + rdst + 32, vsrc + sb * VST + 64 + 8 * p);
            }
        }
    }

    // fp16 scales of stage st (stage bytes at stg) -> sws[st & 1][j][hg] for local row hl (thread = (row, j)):
    //   Q4_K / Q5_K: (S, S, -M, -M);  Q6_K: (S_lo, S_lo, S_hi, S_hi) into the row's meta slot;  Q8_0: (d, d, d, d)
    __device__ __forceinline__ void scales(const uint8_t * stg, int st, int hl, int j, uint2 * sws, int hg) const {
        // FMT: the headers at FHO (IQ4_XS: row pair chunk (tile, hl & 7), half hl >> 3 & 1)
        const uint8_t * row = SBR ? sbbp + ((st >> 2) & 1) * SBR * ROWS + (FMT ? FHO + 16 * hl : hl * SBR)
                              : stg + HB + (!FMT ? hl * SR : TYPE == TIQ4 ? FHO + 16 * (8 * (hl >> 4) + (hl & 7)) + 8 * ((hl >> 3) & 1) : FHO + 16 * hl);
        const int p = st & 3;
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
        } else if constexpr (TYPE == TIQ4) { // the pair's two GGUF headers: fp16 d | u16 scales_h | u8 scales_l[4]
            const int2 hd = *reinterpret_cast<const int2 *>(row + (FMT ? 0 : 8 * ((st >> 2) & 1)));
            const float d = __half2float(__ushort_as_half((unsigned short) (hd.x & 0xffff)));
            const unsigned sl = ((unsigned) hd.y >> (8 * p + 4 * j)) & 0xfu, sh = ((unsigned) hd.x >> (16 + 4 * p + 2 * j)) & 3u;
            a = b = __float2half2_rn(d * (float) ((int) (sl | (sh << 4)) - 32));
        } else if constexpr (TYPE == TIQ4NL) { // header: the fp16 d of the 8 sub-blocks
            const __half dd = *reinterpret_cast<const __half *>(row + 4 * p + 2 * j);
            a = b = __half2half2(dd);
        } else if constexpr (TYPE == TQ3K) { // as Q6_K: (S_lo, S_lo), (S_hi, S_hi) of sub-block j, the same roundings
            const unsigned sc4 = *reinterpret_cast<const unsigned *>(row + 24), dd = *reinterpret_cast<const unsigned *>(row + 28);
            const float d = __half2float(__ushort_as_half((unsigned short) (dd >> (16 * ((st >> 2) & 1)))));
            const unsigned sp = sc4 >> (16 * j);
            a = __float2half2_rn(d * (float) (int8_t) (sp & 0xff));
            b = __float2half2_rn(d * (float) (int8_t) ((sp >> 8) & 0xff));
        } else if constexpr (TYPE == TIQ3S) { // d (1 + 2 s) of sub-block 2p + j, for both 16-halves
            const unsigned sc = *reinterpret_cast<const unsigned *>(row + 24);
            const float d = __half2float(*reinterpret_cast<const __half *>(row + 28));
            a = b = __float2half2_rn(d * (float) (int) (1 + 2 * ((sc >> (8 * p + 4 * j)) & 0xfu)));
        } else if constexpr (TYPE == TQ6K) {
            uint8_t * mrow = FM6 ? const_cast<uint8_t *>(stg + HB + F6M + 16 * hl) : const_cast<uint8_t *>(row) + 16 * (3 ^ ((hl >> 1) & 3));
            const int2 meta = *reinterpret_cast<const int2 *>(mrow);
            const int dh = FM6 ? (hl >> 3) & 1 : (st >> 2) & 1; // FM6: rows g / g + 8 of the tile, else the pair's block
            const float d = __half2float(__ushort_as_half((unsigned short) ((unsigned) meta.y >> (16 * dh))));
            const unsigned sp = (unsigned) meta.x >> (16 * j);
            *reinterpret_cast<__half2 *>(mrow + 8 + 4 * j) =
                __floats2half2_rn(d * (float) (int8_t) (sp & 0xff), d * (float) (int8_t) ((sp >> 8) & 0xff));
            return;
        } else {
            const __half dd = *reinterpret_cast<const __half *>(row + 64 + 2 * j);
            a = b = __half2half2(dd);
        }
        if constexpr (SEP) sws[(st & 1) * 2 * BM + j * BM + hg] = make_uint2(h2u(a), h2u(b));
    }

    // fp16 A fragments a[j][kk][0..3] of stage st for the lane's rows (local rw = 16-row tile base + g, global rwg)
    __device__ __forceinline__ void frag(const uint8_t * stg, int st, int rw, int rwg, int g, int t4, const uint2 * sws,
                                         unsigned (&a)[2][2][4]) const {
        const uint8_t * sw = stg + HB;
        const uint2 * ws = sws + (st & 1) * 2 * BM;
        const int p = st & 3;
        constexpr float OFF = TYPE == TQ6K || TYPE == TIQ3S ? -1056.f
                            : TYPE == TQ3K                  ? -1028.f
                            : (TYPE == TQ80 || TYPE == TIQ4 || TYPE == TIQ4NL) ? -1152.f
                                                                               : -1024.f;
        const __half2 off2 = __float2half2_rn(OFF);
        const uint8_t * ra = sw + rw * SR;
        const uint8_t * rb = ra + 8 * SR;
        // FMT: word (half, sec) = bytes 16 half + 4 t4 .. of the 32-B group of the lane's row A (sec 0: g + 8 (g & 1)) / B
        const int od = g & 1;
        auto fw = [&](int fr, int half, int sec) {
            return *reinterpret_cast<const unsigned *>(sw + fr + 512 * (rw >> 4) + 128 * (2 * od + half) +
                                                       16 * (2 * (g >> 1) + (t4 >> 1)) + 8 * (od ^ sec) + 4 * (t4 & 1));
        };
        // weight words: wd[j][kk][r] = 4 values of elements 16 kk + 4 t4 .. of sub-block j, row g + 8 r
        unsigned wd[2][2][2];
        if constexpr (TYPE == TQ4K || TYPE == TQ5K) {
            constexpr int QO = TYPE == TQ4K ? 16 : 0; // (Q5_K stage rows: qs only)
            const unsigned w0 = FMT ? fw(FQS, 0, 0) : *reinterpret_cast<const unsigned *>(ra + QO + 4 * t4);
            const unsigned w1 = FMT ? fw(FQS, 1, 0) : *reinterpret_cast<const unsigned *>(ra + QO + 16 + 4 * t4);
            const unsigned w2 = FMT ? fw(FQS, 0, 1) : *reinterpret_cast<const unsigned *>(rb + QO + 4 * t4);
            const unsigned w3 = FMT ? fw(FQS, 1, 1) : *reinterpret_cast<const unsigned *>(rb + QO + 16 + 4 * t4);
            wd[0][0][0] = w0 & M4; wd[0][0][1] = w2 & M4; wd[0][1][0] = w1 & M4; wd[0][1][1] = w3 & M4;
            wd[1][0][0] = (w0 >> 4) & M4; wd[1][0][1] = (w2 >> 4) & M4;
            wd[1][1][0] = (w1 >> 4) & M4; wd[1][1][1] = (w3 >> 4) & M4;
            if constexpr (TYPE == TQ5K) {
                // qh from the super-block buffer (FMT: the same word positions as the stage's qs)
                const uint8_t * sq = sbbp + ((st >> 2) & 1) * SBR * ROWS;
                auto fs = [&](int half, int sec) {
                    return *reinterpret_cast<const unsigned *>(sq + FQH + 512 * (rw >> 4) + 128 * (2 * od + half) +
                                                               16 * (2 * (g >> 1) + (t4 >> 1)) + 8 * (od ^ sec) + 4 * (t4 & 1));
                };
                const uint8_t * qa = sq + rw * SBR, * qb = qa + 8 * SBR;
                const unsigned h0 = (FMT ? fs(0, 0) : *reinterpret_cast<const unsigned *>(qa + 16 + 4 * t4)) >> (2 * p);
                const unsigned h1 = (FMT ? fs(1, 0) : *reinterpret_cast<const unsigned *>(qa + 32 + 4 * t4)) >> (2 * p);
                const unsigned h2 = (FMT ? fs(0, 1) : *reinterpret_cast<const unsigned *>(qb + 16 + 4 * t4)) >> (2 * p);
                const unsigned h3 = (FMT ? fs(1, 1) : *reinterpret_cast<const unsigned *>(qb + 32 + 4 * t4)) >> (2 * p);
                wd[0][0][0] |= (h0 & 0x01010101u) << 4; wd[0][0][1] |= (h2 & 0x01010101u) << 4;
                wd[0][1][0] |= (h1 & 0x01010101u) << 4; wd[0][1][1] |= (h3 & 0x01010101u) << 4;
                wd[1][0][0] |= (h0 & 0x02020202u) << 3; wd[1][0][1] |= (h2 & 0x02020202u) << 3;
                wd[1][1][0] |= (h1 & 0x02020202u) << 3; wd[1][1][1] |= (h3 & 0x02020202u) << 3;
            }
        } else if constexpr (TYPE == TIQ4 || TYPE == TIQ4NL) { // indices -> kvalues + 128 (low nibbles: sub-block 0, high: 1)
            const unsigned w0 = FMT ? fw(FQS, 0, 0) : *reinterpret_cast<const unsigned *>(ra + 16 + 4 * t4);
            const unsigned w1 = FMT ? fw(FQS, 1, 0) : *reinterpret_cast<const unsigned *>(ra + 32 + 4 * t4);
            const unsigned w2 = FMT ? fw(FQS, 0, 1) : *reinterpret_cast<const unsigned *>(rb + 16 + 4 * t4);
            const unsigned w3 = FMT ? fw(FQS, 1, 1) : *reinterpret_cast<const unsigned *>(rb + 32 + 4 * t4);
            int lo, hi;
            iq4_lut(w0, IQ4U_T0, IQ4U_T1, IQ4U_T2, IQ4U_T3, lo, hi); wd[0][0][0] = lo; wd[1][0][0] = hi;
            iq4_lut(w2, IQ4U_T0, IQ4U_T1, IQ4U_T2, IQ4U_T3, lo, hi); wd[0][0][1] = lo; wd[1][0][1] = hi;
            iq4_lut(w1, IQ4U_T0, IQ4U_T1, IQ4U_T2, IQ4U_T3, lo, hi); wd[0][1][0] = lo; wd[1][1][0] = hi;
            iq4_lut(w3, IQ4U_T0, IQ4U_T1, IQ4U_T2, IQ4U_T3, lo, hi); wd[0][1][1] = lo; wd[1][1][1] = hi;
        } else if constexpr (TYPE == TQ3K) { // u3 = lows | high << 2 (the value is u3 - 4: OFF)
#pragma unroll
            for (int kk = 0; kk < 2; kk++)
#pragma unroll
                for (int r = 0; r < 2; r++) {
                    const uint8_t * rr = r ? rb : ra;
                    const unsigned x = *reinterpret_cast<const unsigned *>(rr + 8 * kk);
                    const unsigned y = *reinterpret_cast<const unsigned *>(rr + 8 * kk + 4);
                    const unsigned h = *reinterpret_cast<const unsigned *>(rr + 16 + 4 * kk);
                    wd[0][kk][r] = ((x >> (2 * t4)) & 0x03030303u) | (((h >> t4) & 0x01010101u) << 2);
                    wd[1][kk][r] = ((y >> (2 * t4)) & 0x03030303u) | (((h >> (4 + t4)) & 0x01010101u) << 2);
                }
        } else if constexpr (TYPE == TIQ3S) { // 32 +- grid (the value: OFF)
#pragma unroll
            for (int kk = 0; kk < 2; kk++)
#pragma unroll
                for (int r = 0; r < 2; r++) {
                    const uint8_t * rr = r ? rb : ra;
                    const unsigned qh = rr[16 + 2 * p + kk];
                    const unsigned sg = *reinterpret_cast<const unsigned *>(rr + 32 + 4 * kk);
#pragma unroll
                    for (int j = 0; j < 2; j++) {
                        const unsigned qs = rr[8 * kk + 4 * j + t4];
                        const unsigned gv = iq3s_grid(qs | (((qh >> (4 * j + t4)) & 1u) << 8));
                        const unsigned m = ((sg >> (4 * j + t4)) & 0x01010101u) * 0xffu;
                        wd[j][kk][r] = 0x20202020u + (gv ^ m) - m;
                    }
                }
        } else if constexpr (FM6) { // the lane's 16 B of qlo (rows g, g + 8: words A, B) and 8 B of qhi (rows g, g + 8)
            const uint4 q = *reinterpret_cast<const uint4 *>(sw + F6L + 512 * (rw >> 4) + 16 * (4 * g + t4));
            const uint2 hq = *reinterpret_cast<const uint2 *>(sw + F6H + 256 * (rw >> 4) + 8 * (4 * g + t4));
            constexpr unsigned M30 = 0x30303030u;
            wd[0][0][0] = (q.x & M4) | (hq.x & M30);
            wd[0][0][1] = (q.z & M4) | (hq.y & M30);
            wd[0][1][0] = (q.y & M4) | ((hq.x << 4) & M30);
            wd[0][1][1] = (q.w & M4) | ((hq.y << 4) & M30);
            wd[1][0][0] = ((q.x >> 4) & M4) | ((hq.x >> 2) & M30);
            wd[1][0][1] = ((q.z >> 4) & M4) | ((hq.y >> 2) & M30);
            wd[1][1][0] = ((q.y >> 4) & M4) | ((hq.x << 2) & M30);
            wd[1][1][1] = ((q.w >> 4) & M4) | ((hq.y << 2) & M30);
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
        // integer-valued fp16 weights (offset removed) times the scales, A-fragment order
#pragma unroll
        for (int j = 0; j < 2; j++) {
            uint2 sA, sB;
            if constexpr (SEP) {
                sA = ws[j * BM + rwg + (FMT ? 8 * (g & 1) : 0)]; // FMT: the lane's rows A / B (above)
                sB = ws[j * BM + rwg + (FMT ? 8 - 8 * (g & 1) : 8)];
            } else { // Q6_K: (S_j0, S_j1) in the row -> broadcast pairs (FM6: in the row's meta slot)
                const int z = (g >> 1) & 3;
                const __half2 qa = *reinterpret_cast<const __half2 *>(FM6 ? sw + F6M + 16 * rw + 8 + 4 * j : ra + 16 * (3 ^ z) + 8 + 4 * j);
                const __half2 qb = *reinterpret_cast<const __half2 *>(FM6 ? sw + F6M + 16 * (rw + 8) + 8 + 4 * j : rb + 16 * (3 ^ z) + 8 + 4 * j);
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
                        const __half2 v = __hadd2(u2h(__byte_perm(wd[j][kk][r], 0x64646464u, hh ? 0x4342 : 0x4140)), off2);
                        __half2 o;
                        if constexpr (TYPE == TQ4K || TYPE == TQ5K) o = __hfma2(v, u2h(sc.x), u2h(sc.y));
                        else if constexpr (TYPE == TQ6K || TYPE == TQ3K || TYPE == TIQ3S) o = __hmul2(v, u2h(kk ? sc.y : sc.x));
                        else o = __hmul2(v, u2h(sc.x));
                        a[j][kk][2 * hh + r] = h2u(o);
                    }
                }
        }
    }
};

template <int TYPE, bool GLU = false, int NB = BN, int LAY = 0, int TYPE2 = -1, int LAY2 = LAY>
__global__ void __launch_bounds__(256, 1)
    mmh(WMat W, const __half * X, int ntok, Out O, const uint8_t * W2 = nullptr, WMat Wu = WMat{}) {
    constexpr bool MIX = TYPE2 >= 0;
    static_assert(MIX || !GLU || TYPE == TQ4K || TYPE == TQ5K || TYPE == TIQ4, "same-type fused gate / up needs in-row scales");
    static_assert(!MIX || GLU, "mixed types: fused gate / up only");
    static_assert(NB % 32 == 0 && NB <= BN, "token tile: a multiple of 32 (X-tile copy loop, n8-tile pairs)");
    static_assert(HMI == 1, "one 16-row MMA tile per warp");
    constexpr int TU = MIX ? TYPE2 : TYPE, LU = MIX ? LAY2 : LAY;
    constexpr bool WOWN = pf_wown(TYPE, MIX) && HT<TYPE>::NS == 4; // warp-owned weight copies, one barrier per two stages
    using HA = HalfW<TYPE, LAY, MIX ? BM / 2 : BM, GLU && !MIX, 0, 0, WOWN>;
    using HU = HalfW<TU, LU, BM / 2, false, BM / 2 * HT<TYPE>::SR, BM / 2>;
    constexpr int WPART = MIX ? BM / 2 * (HT<TYPE>::SR + HT<TU>::SR) : BM * HT<TYPE>::SR; // weight bytes per stage
    constexpr int STG = MIX ? hm_stage2<TYPE, TU, NB>() : hm_stage<TYPE, NB>();
    constexpr int NSH = MIX ? hm_ns2<TYPE, TU, NB>() : HT<TYPE>::NS;
    constexpr bool MBL = pf_mbl(TYPE, TU, MIX) && NSH == 4 && !WOWN; // per-stage mbarriers, copies two stages ahead
    constexpr int SMEM = MIX ? hm_smem2<TYPE, TU, NB>() : hm_smem<TYPE, NB>();
    static_assert(STG == WPART + hm_x<NB>(), "stage layout");
    static_assert(NB * (BM + 4) * 4 <= (MIX ? hm_smem2<TYPE, TU, NB>() : hm_smem<TYPE, NB>()), "epilogue tile");
    constexpr int HWT = NB / (8 / HNWM), HNI = HWT / 8, XIT = NB * 8 / 256, FG = frag_g(TYPE, TU, GLU, MIX);
    static_assert(FG >= 0 && FG < 4, "MK_PF_FRAG_G: an MMA group of the stage, 0 .. 3 (else the next stage's fragments are never made)");
    extern __shared__ __align__(16) uint8_t sm[];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, t4 = lane & 3;
    const int wm = warp % HNWM, wn = warp / HNWM; // warp tile: rows HWR wm .., tokens HWT wn ..
    const int ntt = gridDim.y;
    const int lin = blockIdx.y * gridDim.x + blockIdx.x;
    const int row0 = (lin / ntt) * (GLU ? BM / 2 : BM), tok0 = (lin % ntt) * NB;
    const unsigned sbase = (unsigned) __cvta_generic_to_shared(sm);
    // after the stages: the halves' super-block buffers (SBT), then the scale pairs sws [2][2 j][BM]
    uint8_t * sbbs = sm + NSH * STG;
    constexpr int SBBA = 2 * HA::SBR * (MIX ? BM / 2 : BM), SBBU = MIX ? 2 * HU::SBR * BM / 2 : 0;
    uint2 * sws = reinterpret_cast<uint2 *>(sbbs + SBBA + SBBU);

    HA ha;
    HU hu;
    ha.init(tid, W, W2, row0, sbbs);
    if constexpr (MIX) hu.init(tid, Wu, nullptr, row0, sbbs + SBBA);
    const __half * xsrc[XIT];
    unsigned xdst[XIT];
#pragma unroll
    for (int k = 0; k < XIT; k++) {
        const int i = tid + k * 256, t = i >> 3, c = i & 7;
        xsrc[k] = X + (size_t) (tok0 + t) * W.K + 8 * c;
        xdst[k] = WPART + t * 128 + ((c ^ (t & 7)) << 4);
    }

    auto loadw = [&](int st) { // stage st's weight bytes
        const unsigned s = sbase + (st % NSH) * STG;
        ha.load(s, st, tid);
        if constexpr (MIX) hu.load(s, st, tid);
    };
    auto loadx = [&](int st) { // stage st's activation tile
        const unsigned s = sbase + (st % NSH) * STG;
#pragma unroll
        for (int k = 0; k < XIT; k++) cpa16(s + xdst[k], xsrc[k] + BK * st);
    };
    // the scales of stage st: thread = (row tid / 2, j); every warp makes those of its own 16 rows
    auto scales = [&](int st) {
        const uint8_t * stg = sm + (st % NSH) * STG;
        const int hr = tid >> 1, j = tid & 1;
        if constexpr (!MIX) ha.scales(stg, st, hr, j, sws, hr);
        else if (warp < 4) ha.scales(stg, st, hr, j, sws, hr);
        else hu.scales(stg, st, hr - BM / 2, j, sws, hr);
    };
    // fp16 A fragments of stage st: the stage's weights times their scales (sws / the Q6_K meta slot)
    auto frag = [&](int st, unsigned (&a)[HMI][2][2][4]) {
        const uint8_t * stg = sm + (st % NSH) * STG;
        const int rwg = HWR * wm + g;
        if constexpr (!MIX) ha.frag(stg, st, rwg, rwg, g, t4, sws, a[0]);
        else if (warp < 4) ha.frag(stg, st, rwg, rwg, g, t4, sws, a[0]);
        else hu.frag(stg, st, rwg - BM / 2, rwg, g, t4, sws, a[0]);
    };

    float acc[HMI][HNI][4];
#pragma unroll
    for (int m = 0; m < HMI; m++)
#pragma unroll
        for (int i = 0; i < HNI; i++) acc[m][i][0] = acc[m][i][1] = acc[m][i][2] = acc[m][i][3] = 0.f;

    const int nst = W.K / BK;
    const unsigned mbf = sbase + SMEM - MBB, mbe = mbf + 32; // MBL: full[4], empty[4] (8 B each); stage s: buffer s % 4, phase s / 4
    if constexpr (MBL) {
        if (tid == 0)
#pragma unroll
            for (int b = 0; b < 4; b++) {
                mb_init(mbf + 8 * b, 256);
                mb_init(mbe + 8 * b, 8);
            }
        __syncthreads();
        for (int s = 0; s < 2; s++) { // (nst >= 2)
            loadw(s);
            loadx(s);
            cpa_commit();
            cpa_mb_arrive(mbf + 8 * s);
        }
        mb_wait(mbf, 0);
    } else if constexpr (WOWN) { // commit groups {X 0, X 1, W 0}, {W 1}, {W 2}, then one per step st: {X st+2, X st+3, W st+3} (st even), {W st+3}
        loadx(0);
        loadx(1);
        loadw(0);
        cpa_commit();
        loadw(1); // (nst >= 2: K is a multiple of 2 BK)
        cpa_commit();
        if (2 < nst) loadw(2);
        cpa_commit();
        cpa_wait<2>();
        __syncwarp(); // the warp's own weights of stage 0 (activations: step 0's barrier)
    } else {
#pragma unroll
        for (int s = 0; s < NSH - 1; s++) {
            if (s < nst) {
                loadw(s);
                loadx(s);
            }
            cpa_commit();
        }
        cpa_wait<NSH - 2>();
        __syncthreads();
    }
    scales(0);

    // ldmatrix rows: lanes 0-7 / 8-15 / 16-23 / 24-31 -> tokens +0..7 (slots 0-7) / +0..7 (slots 8-15)
    // / +8..15 (slots 0-7) / +8..15 (slots 8-15) of a 16-token pair of n8 tiles
    const unsigned xl = WPART + (HWT * wn + 8 * (lane >> 4) + (lane & 7)) * 128;
    unsigned xo[4];
#pragma unroll
    for (int G = 0; G < 4; G++) xo[G] = xl + (((2 * G + ((lane >> 3) & 1)) ^ (lane & 7)) << 4);

    // Software pipeline: a stage's MMAs use fragments made during the previous stage, and the next stage's
    // scales + fragments are made between this stage's MMAs (the warp's own rows: scales -> fragments needs only
    // __syncwarp). ch: the fp16 sums of the current 128-k group; they go into acc at the start of the next group
    // (interleaved with its first MMAs) and after the loop. ch starts at 0, so the first add is 0 + 0 (exactly 0):
    // every acc element gets the same sequence of fp32 adds as without the pipeline.
    static_assert(HCH == 2, "pipelined step: fp16 groups of two stages");
    unsigned ch[HMI][HNI][2];
#pragma unroll
    for (int mi = 0; mi < HMI; mi++)
#pragma unroll
        for (int ni = 0; ni < HNI; ni++) ch[mi][ni][0] = ch[mi][ni][1] = 0u;
#if MK_PF_INFCHK
    unsigned chk_p = 0, chk_o = 0; // non-finite fp16 partial sums / fp32 outputs of this thread (token columns < ntok)
#endif
    auto conv = [&](int mi, int ni) {
        const float2 lo = __half22float2(u2h(ch[mi][ni][0])), hi = __half22float2(u2h(ch[mi][ni][1]));
#if MK_PF_INFCHK
        const int tc = tok0 + HWT * wn + 8 * ni + 2 * t4; // the columns of lo / hi: tc, tc + 1
        if (tc < ntok) chk_p += !isfinite(lo.x) + !isfinite(hi.x);
        if (tc + 1 < ntok) chk_p += !isfinite(lo.y) + !isfinite(hi.y);
#endif
        acc[mi][ni][0] += lo.x; acc[mi][ni][1] += lo.y; acc[mi][ni][2] += hi.x; acc[mi][ni][3] += hi.y;
    };
    auto step = [&](int st, auto ph, const unsigned (&a)[HMI][2][2][4], unsigned (&an)[HMI][2][2][4]) {
        constexpr int PH = decltype(ph)::value;
        if constexpr (MBL) {
            mb_wait(mbf + 8 * ((st + 1) & 3), ((st + 1) >> 2) & 1); // stage st + 1 landed (st: waited a step earlier)
        } else if constexpr (WOWN) { // this thread's copies of X st, X st + 1, W st + 1 (st even) / W st + 1 (odd) complete
            cpa_wait<1>();
            if constexpr (PH == 0) __syncthreads(); // (also: every warp is done with the activation buffers of st - 2, st - 1)
            else __syncwarp();
        } else {
            cpa_wait<NSH - 3>(); // stages <= st + 1 complete: st's activations, st + 1's weights for the next fragments
            __syncthreads();
        }
        const unsigned sx = sbase + (st % NSH) * STG;
#pragma unroll
        for (int G = 0; G < 4; G++) {
            const int j = G >> 1, kk = G & 1;
#pragma unroll
            for (int np = 0; np < HNI / 2; np++) {
                unsigned r[4];
                ldsm_x4(r, sx + xo[G] + np * 16 * 128);
#pragma unroll
                for (int mi = 0; mi < HMI; mi++) {
                    if (G == 0 && PH == 0) {
                        conv(mi, 2 * np);
                        conv(mi, 2 * np + 1);
                        hmma16z(ch[mi][2 * np], a[mi][j][kk], r[0], r[1]);
                        hmma16z(ch[mi][2 * np + 1], a[mi][j][kk], r[2], r[3]);
                    } else {
                        hmma16(ch[mi][2 * np], a[mi][j][kk], r[0], r[1]);
                        hmma16(ch[mi][2 * np + 1], a[mi][j][kk], r[2], r[3]);
                    }
                }
            }
            // next stages' copies, then the next stage's scales and fragments, once the tensor pipe has work queued
            // (at the last stage they read a stale buffer: harmless, unused)
            if (G == 0) {
                if constexpr (MBL) { // stage st + 2 into the buffer of st - 2, once every warp is done with that
                    if (st >= 2) mb_wait(mbe + 8 * ((st - 2) & 3), ((st - 2) >> 2) & 1);
                    if (st + 2 < nst) {
                        loadw(st + 2);
                        loadx(st + 2);
                    }
                } else if constexpr (WOWN) { // (the barrier of an even step precedes the activation copies into st - 2's, st - 1's buffers)
                    if constexpr (PH == 0) {
                        if (st + 2 < nst) loadx(st + 2);
                        if (st + 3 < nst) loadx(st + 3);
                    }
                    if (st + 3 < nst) loadw(st + 3);
                } else if (st + NSH - 1 < nst) {
                    loadw(st + NSH - 1);
                    loadx(st + NSH - 1);
                }
                cpa_commit();
                if constexpr (MBL) cpa_mb_arrive(mbf + 8 * ((st + 2) & 3));
            }
            if (G == FG) {
                scales(st + 1);
                __syncwarp();
                frag(st + 1, an);
            }
        }
        if constexpr (MBL) { // the warp is done with stage st (activations now, weights in the previous step)
            __syncwarp();
            if (lane == 0) mb_arrive(mbe + 8 * (st & 3));
        }
    };
    unsigned fa[HMI][2][2][4], fb[HMI][2][2][4];
    __syncwarp(); // (scales(0) above: the warp's own rows)
    frag(0, fa);
    for (int st = 0; st < nst; st += 2) { // nst is even for every K of the model
        step(st, std::integral_constant<int, 0>(), fa, fb);
        step(st + 1, std::integral_constant<int, 1>(), fb, fa);
    }
    // OUT_ADD: the y elements this thread adds to in the store loop below, loaded here (they do not depend on the tile), so
    // their latency overlaps the last conversions and the staging through se. The loop loaded each one right before its
    // add, behind the previous iteration's store: one exposed load latency per iteration (16 at NB = 128)
    constexpr int EIT = GLU ? 1 : NB * BM / 4 / 256;
    static_assert(GLU || NB * BM / 4 % 256 == 0, "epilogue: whole float4 iterations per thread");
    float4 yo[EIT];
    if constexpr (!GLU) {
        if (O.mode == OUT_ADD)
#pragma unroll
            for (int k = 0; k < EIT; k++) {
                const int i = tid + 256 * k, tk = i / (BM / 4), c = (i % (BM / 4)) * 4;
                yo[k] = tok0 + tk < ntok ? __ldcg(reinterpret_cast<const float4 *>(O.y + (size_t) (tok0 + tk) * O.ldy + row0 + c))
                                         : make_float4(0.f, 0.f, 0.f, 0.f);
            }
    }
#pragma unroll
    for (int mi = 0; mi < HMI; mi++)
#pragma unroll
        for (int ni = 0; ni < HNI; ni++) conv(mi, ni);
    cpa_wait<0>();
    __syncthreads();
    constexpr int ELD = BM + 4;
    float * se = reinterpret_cast<float *>(sm);
    const bool fmt = MIX && warp >= 4 ? HU::FMT : HA::FMT; // FMT: rows A / B (rawq)
#pragma unroll
    for (int mi = 0; mi < HMI; mi++)
#pragma unroll
    for (int ni = 0; ni < HNI; ni++) {
        const int r = HWR * wm + 16 * mi + g, tk = HWT * wn + 8 * ni + 2 * t4;
        const int rA = r + (fmt ? 8 * (g & 1) : 0), rB = r + (fmt ? 8 - 8 * (g & 1) : 8);
        se[tk * ELD + rA] = acc[mi][ni][0];
        se[(tk + 1) * ELD + rA] = acc[mi][ni][1];
        se[tk * ELD + rB] = acc[mi][ni][2];
        se[(tk + 1) * ELD + rB] = acc[mi][ni][3];
    }
    __syncthreads();
    if constexpr (GLU) {
        // (token, 16-group of the 64 outputs): lanes 8 q' .. 8 q' + 7 take 8 consecutive tokens of one group, so their float4
        // reads of se (ELD = 132: rows of 33 16-B units) hit 8 distinct bank quads per quarter warp
        for (int i = tid; i < NB * 4; i += 256) {
            const int tk = 8 * (i >> 5) + (i & 7), q = (i >> 3) & 3;
            if (tok0 + tk >= ntok) continue;
            const float * gp = se + tk * ELD + 16 * q;
            float gf[16], uf[16];
#pragma unroll
            for (int m = 0; m < 4; m++) {
                *reinterpret_cast<float4 *>(gf + 4 * m) = *reinterpret_cast<const float4 *>(gp + 4 * m);
                *reinterpret_cast<float4 *>(uf + 4 * m) = *reinterpret_cast<const float4 *>(gp + BM / 2 + 4 * m);
            }
            __align__(16) __half hv[16];
            // the 16 denominators first: each division below branches to its slow path (FCHK), which kept the next
            // element's expf behind it (one latency chain per element); the same expressions per element
            float den[16];
#pragma unroll
            for (int e = 0; e < 16; e++) den[e] = 1.0f + expf(-gf[e]);
#pragma unroll
            for (int e = 0; e < 16; e++) {
                const float gv = gf[e], v = gv / den[e] * uf[e];
#if MK_PF_INFCHK
                chk_o += !isfinite(gv) + !isfinite(uf[e]);
                chk_clamp(v, CHK_IN_GLU);
#endif
                hv[hperm(e)] = __float2half_rn(fminf(fmaxf(v, -65504.f), 65504.f));
            }
            uint4 * dst = reinterpret_cast<uint4 *>(O.yh + (size_t) (tok0 + tk) * O.ldy + row0 + 16 * q);
            dst[0] = reinterpret_cast<const uint4 *>(hv)[0];
            dst[1] = reinterpret_cast<const uint4 *>(hv)[1];
        }
#if MK_PF_INFCHK
        chk_add(chk_site, 0, chk_p);
        chk_add(chk_site, 1, chk_o);
#endif
        return;
    }
#pragma unroll
    for (int k = 0; k < EIT; k++) { // (i = tid, tid + 256, .. < NB BM / 4)
        const int i = tid + 256 * k, tk = i / (BM / 4), c = (i % (BM / 4)) * 4;
        if (tok0 + tk >= ntok) continue;
        float4 v = *reinterpret_cast<const float4 *>(se + tk * ELD + c);
#if MK_PF_INFCHK
        chk_o += !isfinite(v.x) + !isfinite(v.y) + !isfinite(v.z) + !isfinite(v.w);
#endif
        float4 * dst = reinterpret_cast<float4 *>(O.y + (size_t) (tok0 + tk) * O.ldy + row0 + c);
        if (O.mode == OUT_ADD) {
            const float4 o = yo[k];
            v.x += o.x; v.y += o.y; v.z += o.z; v.w += o.w;
        }
        *dst = v;
    }
#if MK_PF_INFCHK
    chk_add(chk_site, 0, chk_p);
    chk_add(chk_site, 1, chk_o);
#endif
}

} // namespace pf
