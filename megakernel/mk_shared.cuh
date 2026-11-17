// Definitions both sides of the engine need (qwen38_mk.cu: device code + launchers; engine.cu and the Makefile's other
// HOST_SRC files: host side): build settings, model dimensions, device buffer layouts, launch parameters. Namespace mk
// on the host side; the device TU defines MK_SHARED_NS empty, keeping them in its anonymous namespace with its kernels
// (the two sides exchange no C++ types).
#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <climits>
#include <condition_variable>
#include <map>
#include <mutex>
#include <memory>
#include <numeric>
#include <sstream>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <functional>
#include <iterator>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

#ifdef _WIN32
#include "compat_win32.h"
#else
#include <unistd.h>
#endif

#include "ggml.h"
#include "gguf.h"
#include "qwen38_mk.h"

#ifndef _WIN32
extern char ** environ; // (state files: the MK_* settings in the key)
#endif

// MK_SPEC_LOG=1: the draft head keeps its top logits, and with MK_SPEC_LOG_FILE=<path> every decode
// cycle appends a JSON line (drafts + their MTP top logits, verify rows, outcome). Analysis only.
#ifndef MK_SPEC_LOG
#define MK_SPEC_LOG 0
#endif
// MK_TEST_NAN=1 (tests only): env MK_TEST_NAN_AT=N writes NaN K scales into the first attention layer before decode launch N
#ifndef MK_TEST_NAN
#define MK_TEST_NAN 0
#endif
// MK_TEST_SLOTS_FAIL=1 (tests only): env MK_TEST_SLOTS_FAIL_AT=k makes slots_open throw an MKCudaError once lane k is built
#ifndef MK_TEST_SLOTS_FAIL
#define MK_TEST_SLOTS_FAIL 0
#endif
// MK_KV_EMU=1 (accuracy experiments): MK_KV_EMU_SPEC rounds chosen K / V cache slots to a lower-bit grid (kv_store_emu)
// while keeping the q8 layout, so lower-bit KV formats can be scored per layer before the kernels support them.
#ifndef MK_KV_EMU
#define MK_KV_EMU 0
#endif
// wv: the main-pass GEMV phases run the int8 tensor-core verify GEMM (wv.cuh) on a load-time fragment-ordered
// tile-major weight layout, at every T (T-invariant, not bit-identical to the dp4a GEMVs).
// MK_K2_TB: tokens per batch of resid_wv's last-arrival combine (their partial loads issued together)
#ifndef MK_K2_TB
#define MK_K2_TB 8
#endif
#ifndef MK_K6_DEF // MK_K6's default (env overrides)
#define MK_K6_DEF 6
#endif
// MK_K8 (prefill small kernels, exact; env bits, default MK_K8_DEF = 6: bit 1 is unmeasured, opt-in): 1 pk_ab on a
// second stream next to
// gemm_dn_in, 2 pk_norm_quant keeps x in registers (one read), 4 pk_dn_prep_t: MK_K8_NT tokens per block, conv taps read once
#ifndef MK_K8_DEF
#define MK_K8_DEF 6
#endif
#ifndef MK_K8_NT
#define MK_K8_NT 4
#endif
// MK_ATT_I8 1: build the int8 Q.K^T decode attention (experimental, in development: the env MK_ATT_I8=1 then selects it at
// run time; not bit-identical to the default fp16 Q). 0: not compiled (the default; it adds ~5 GiB and ~5 minutes to ptxas)
#ifndef MK_ATT_I8
#define MK_ATT_I8 0
#endif
// MK_W48X 1 (default): decode attention on low-bit MK_KV slots with cheap K / V copies and expansion (attn_core_q8m, attn_core_w48m:
// same staged bytes, same partials). 0: the old path. Compile-time: a run-time bit shifts ptxas's register allocation.
#ifndef MK_W48X
#define MK_W48X 1
#endif
// MK_W48X_Q8 1 (with MK_W48X 1): all-q8 slots run those MIX kernels too (no q8 instances in the build). 0: their own kernels
#ifndef MK_W48X_Q8
#define MK_W48X_Q8 0
#endif
// MK_L2_TUNE 1: the L2 prefetch budgets (L2Tune below) are read from the environment (MK_L2_*, docs/EXPERIMENTAL.md), to
// tune them on a GPU with another L2 size. 0: the constants, none of this code (the default)
#ifndef MK_L2_TUNE
#define MK_L2_TUNE 0
#endif
// Wave 6 (exact, measured as one build against 1ecb507; -D...=0 gives back 1ecb507's code for each). Defined here, before any use:
// MK_FORMAL_SYNC is tested in qwen38_mk.cu ahead of its local #ifndef, so a flag-less build needs these to be set first.
#ifndef MK_Q32N
#define MK_Q32N 1 // resid K2 combine: a batch's q32_store chains interleaved
#endif
#ifndef MK_FORMAL_SYNC
#define MK_FORMAL_SYNC 1 // correct6's PTX-model fences / __syncwarp (lm_head ticket, softmax, resid last arrival)
#endif
#ifndef MK_FFNA
#define MK_FFNA 7 // wv phase-top dealing table, ssq loads first, meta-first fill (bit 8, the proj wrappers' dealing: off)
#endif
#ifndef MK_WVREMAT
#define MK_WVREMAT 1 // dn_proj_wv / attn_proj_wv rebuild their loop state after each tile call (needs MK_FFNA's table)
#endif
#ifndef MK_W48Z
#define MK_W48Z 1 // attn_core_w48m: dead acc[2] tile re-zeroed, next k-step's Q loads first
#endif
#ifndef MK_K6X
#define MK_K6X 1 // preset D T8: K6 expansion on the P V warps
#endif
#ifndef MK_LMH8
#define MK_LMH8 1 // lm_head<8>: pen drafts staged in shared memory, loaded per row pair
#endif
#ifndef MK_SILUPAD
#define MK_SILUPAD 1 // ffn_gu_wv at T < 8: padding columns feed silu 1 (fast division path), results unchanged
#endif
// Not compile-time switches (the served behaviour is built in): a -D for one of these names stops the build.
#if defined(MK_WV)
#error "MK_WV is built in (its served value): drop -DMK_WV"
#elif defined(MK_MAXT)
#error "MK_MAXT is built in (its served value): drop -DMK_MAXT"
#elif defined(MK_C1)
#error "MK_C1 is built in (its served value): drop -DMK_C1"
#elif defined(MK_EGRESS)
#error "MK_EGRESS is built in (its served value): drop -DMK_EGRESS"
#elif defined(MK_WV_Q6F)
#error "MK_WV_Q6F is built in (its served value): drop -DMK_WV_Q6F"
#elif defined(MK_K5)
#error "MK_K5 is built in (its served value): drop -DMK_K5"
#elif defined(MK_GSYNC)
#error "MK_GSYNC is built in (its served value): drop -DMK_GSYNC"
#elif defined(MK_K7)
#error "MK_K7 is built in (its served value): drop -DMK_K7"
#elif defined(MK_W48)
#error "MK_W48 is built in (its served value): drop -DMK_W48"
#elif defined(MK_DA_ROUTE)
#error "MK_DA_ROUTE is built in (its served value): drop -DMK_DA_ROUTE"
#elif defined(MK_DA_Q4)
#error "MK_DA_Q4 is built in (its served value): drop -DMK_DA_Q4"
#elif defined(MK_AQ_EARLY)
#error "MK_AQ_EARLY is built in (its served value): drop -DMK_AQ_EARLY"
#elif defined(MK_XP_SM)
#error "MK_XP_SM is built in (its served value): drop -DMK_XP_SM"
#elif defined(MK_K6_LD)
#error "MK_K6_LD is built in (its served value): drop -DMK_K6_LD"
#elif defined(MK_W48S)
#error "MK_W48S is built in (its served value): drop -DMK_W48S"
#elif defined(MK_MTP_KVC)
#error "MK_MTP_KVC is built in: drop -DMK_MTP_KVC (env MK_MTP_KVC=0 turns the K / V-only catch-up rows off)"
#elif defined(MK_WV_FMK)
#error "MK_WV_FMK is built in (its served value): drop -DMK_WV_FMK"
#elif defined(MK_WV_OUT)
#error "MK_WV_OUT is built in (its served value): drop -DMK_WV_OUT"
#endif

// A failed CUDA call prints its message and throws MKCudaError (the C API returns -1 with the message and resets
// the engine, a slot call's slots at their next prompt; the server answers with an HTTP error and keeps serving). A
// sticky error (the context is unusable: a faulting kernel, illegal address, trap, ECC, ...) cannot be recovered
// in-process: every later CUDA call fails too, so the process exits with code 4 after the message (the launcher / user
// restarts it).
[[noreturn]] inline void mk_cuda_fail(cudaError_t e, const char * expr, const char * file, int line) {
    char msg[768];
    snprintf(msg, sizeof msg, "CUDA error %s at %s:%d: %s", expr, file, line, cudaGetErrorString(e));
    fprintf(stderr, "%s\n", msg);
    bool sticky = false;
    switch (e) {
        case cudaErrorIllegalAddress: case cudaErrorLaunchFailure: case cudaErrorHardwareStackError:
        case cudaErrorIllegalInstruction: case cudaErrorMisalignedAddress: case cudaErrorInvalidAddressSpace:
        case cudaErrorInvalidPc: case cudaErrorLaunchTimeout: case cudaErrorECCUncorrectable: case cudaErrorAssert:
        case cudaErrorContextIsDestroyed:
            sticky = true;
            break;
        default: // anything else is sticky if it survives clearing the last error
            cudaGetLastError();
            sticky = cudaGetLastError() != cudaSuccess;
    }
    if (sticky) {
        fprintf(stderr, "[mk] the CUDA error is sticky (the CUDA context is unusable): exiting with code 4\n");
        fflush(stdout);
        fflush(stderr);
        _exit(4);
    }
    throw MKCudaError(msg);
}
#define CK(x)                                                                                   \
    do {                                                                                        \
        const cudaError_t e_ = (x);                                                             \
        if (e_ != cudaSuccess) mk_cuda_fail(e_, #x, __FILE__, __LINE__);                        \
    } while (0)

#ifndef MK_PF_INFCHK
#define MK_PF_INFCHK 0 // debug builds: 1 = count the prefill GEMMs' fp16 overflows (chk_*; the served build has none of it)
#endif

#ifndef MK_SHARED_NS
#define MK_SHARED_NS mk
#endif
namespace MK_SHARED_NS {

constexpr int D = 5120, FF = 17408, VOCAB = 248320, NL = 64, NLA = 65;
constexpr int NH = 24, NKV = 4, HD = 256;
constexpr int NKH = 16, NVH = 48, HK = 128;
constexpr int QKV = 10240, VD = 6144, QG = 12288, KVD = 1024;
// MAXT: the most tokens of one verify pass (1 + up to 7 drafts). The dp4a ffn_down cannot hold T >= 6 rows, so every
// main-layer ffn_down runs on wv (its Q6_K / Q3_K / IQ3_S too, tile-major), the conv ring has 16 slots, and the
// attention cores run two head-group passes at T >= 6 (48 query rows do not fit).
constexpr int MAXT = 8;
#ifndef MK_MAXD
#define MK_MAXD 8 // 16: the deep build (T 9 .. 16 run as one 16-row pass, main_forward_d)
#endif
constexpr int MAXD = MK_MAXD; // the most tokens of one decode verify pass
static_assert(MAXD == MAXT || MAXD == 2 * MAXT, "MK_MAXD: 8 or 16 (the deep pass: two 8-row attention calls)");
// batched launches (parallel slots) wider than a solo pass: RMAX rows over the lanes, the wv phases in K sub-chunk fills
// (wv.cuh "batched passes wider than MAXT rows"); the row buffers hold RMAX rows
#ifndef MK_PAR_RMAX
// (16: 16-row passes are faster than 24-row ones at 3-4 lanes)
#define MK_PAR_RMAX 16
#endif
#define MK_PAR_WIDE (MK_PAR_RMAX > 8) // (8: MAXT, which the preprocessor cannot see)
constexpr int RMAX = MK_PAR_WIDE ? MK_PAR_RMAX : MAXT;
static_assert(RMAX % 8 == 0 && RMAX <= 32, "RMAX: whole 8-row lm_head passes, store_lane_t's 32 lanes");
static_assert(MAXD <= RMAX, "MK_MAXD=16 runs the RMAX-row weight phases (MK_PAR_RMAX >= 16)");
constexpr int MAXT_PF = 5; // rows of prefill-mode and scoring launches (T >= 6 runs only as a decode verify pass)
constexpr int NT = 512, NW = NT / 32;
constexpr float EPS = 1e-6f;
constexpr float ROPE_BASE = 10000000.0f;
// conv ring: dn_proj writes positions n .. n + T - 1 while dn_core reads n - 3 .. n + T - 1 (T + 3 slots)
constexpr int CONV_RING = 2 * MAXD, KV_RING = 2 * MAXD, H_RING = 2 * MAXD;
static_assert(MAXD + 3 <= CONV_RING && 2 * MAXD - 1 <= KV_RING && MAXD + 1 <= H_RING, "rings");
// KV cache rows per kv head and RoPE table rows past the usable context (Bufs::ctx = ctx + KV_PAD): a decode launch at
// position n writes K / V rows up to n + 2 MAXD - 2 (the MTP block's catch-up rows up to P <= n + MAXD, then its draft
// rows up to P + MAXD - 2), and step() admits n while n + MAXD + 2 < ctx, so rows up to ctx + MAXD - 5 are written
constexpr int KV_PAD = 2 * MAXD;
static_assert((2 * MAXD - 2) - (MAXD + 3) < KV_PAD, "KV_PAD: the deepest MTP draft row at the end of the context");
constexpr int KVGB = NKH * HK + NVH * HK + 2 * NVH; // k | v | g | beta
constexpr int KVGB_V = NKH * HK, KVGB_G = KVGB_V + NVH * HK, KVGB_B = KVGB_G + NVH;
constexpr int N_ATT_SLOTS = 17; // 16 trunk attention layers + MTP block
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
constexpr int ATT_NCH = 20;                // max split-K chunks per kv head (4 x 20 items fit one wave)
constexpr int ATT_BUCKET = ATT_NCH * ATP;  // the chunk size steps by one tile every ATT_BUCKET positions
constexpr int MAX_CH = ATT_NCH + 1;        // apart / aml chunks per kv head: attn_dims' <= ATT_NCH + the MTP window's sink chunk
static_assert(MAX_CH <= 32, "attn_combine: one chunk's (m, l) per lane");
// The L2 prefetch budgets, sized for the 6 MB of an RTX 3090 (hints only: no value depends on them)
constexpr int DN_PF_ITEM = 4096; // dn_commit: bytes of resid_sout per (tile, chunk) item, the phase-top unit included
constexpr int ATT_PFO_L = 8;     // attn_idle_pf: 128-B lines of resid_o per item, past the phase-top unit
constexpr int MTP_PFL = 11;      // mtp_pf_gw: lines per row of a warp's first row pair in the MTP's row-pair phases
constexpr int MTP_PFR = 12;      // mtp_pf_resid: lines per row of a block's first resid32 item in the MTP's phases
#ifndef MK_CK_PFL
#define MK_CK_PFL 12 // wv_pf_next: lines per warp and matrix before a sub-chunk fill of the _ck phases
#endif
// -DMK_L2_TUNE=1 builds: the budgets above as run-time values (MK_L2_DN, _ATTN, _ATTN_CTX, _TOP, _CK, _MTP, _MTP_R, _MTP_KEEP)
struct L2Tune {
    int dn;       // DN_PF_ITEM
    int attn;     // ATT_PFO_L
    int attn_ctx; // attn_idle_pf: the longest context it runs at (positions; constant build: 2 ATT_BUCKET)
    int top;      // wv_pf_unit: units per warp at a phase top (constant build: 1)
    int ck;       // MK_CK_PFL
    int mtp;      // MTP_PFL
    int mtpr;     // MTP_PFR
    int keep;     // 1: the MTP phases' prefetches go out evict_last, so the lines stay in L2 for the next draft step
};
// largest T <= T_req whose rows n .. n+T-1 all see the same attention partition (see attn_dims)
inline int att_cap_T(int n, int T_req) {
    const int lim = ATT_BUCKET * ((n + ATT_BUCKET) / ATT_BUCKET) - n; // rows with n + 1 + t <= next multiple
    return T_req < lim ? T_req : lim;
}
constexpr int KVQ_ROW = 256 + 16; // staged q8 K or V row: 256 int8 values + 8 fp16 scales
// K/V tile region: fp16 cache: 2 stages of fp16 K/V tiles; q8 cache: one fp16 K/V tile + 2 stages of q8 tiles
constexpr int SM_ATTN_KV = (4 * ATP * (256 + 8) * 2 > 2 * ATP * (256 + 8) * 2 + 4 * ATP * KVQ_ROW)
                               ? 4 * ATP * (256 + 8) * 2 : 2 * ATP * (256 + 8) * 2 + 4 * ATP * KVQ_ROW;
constexpr int SM_ATTN = 32 * (256 + 8) * 2 + SM_ATTN_KV + 32 * ATP * 4 + 32 * (ATP + 8) * 2 + 96 * 4; // Q, K/V, S, P, stats
// attn_core_q8 (decode attention, all-q8 slots): q8 K / V ring stages, double-buffered S and P
constexpr int AQ_NSV = 4;             // V(t) is issued in interval t - 1 and read in t + 2
constexpr int AQ_NSKM = 4;            // K stages (all-q8 and MIX forms): K(t) is issued in interval t - 3 (MK_W48X MIX: t - 4), read in t
constexpr int AQ_LDS = ATP + 4;       // S row stride (floats): the softmax quads of 8 rows hit distinct banks; the pad
                                      // columns 32 .. 34 hold the rows' m, l and alpha (the 99 KB budget with AQ_NSKM)
constexpr int AQ_STG = ATP * KVQ_ROW; // one K or V ring stage
constexpr int SM_ATTN_Q8 = 32 * (256 + 8) * 2 + (AQ_NSKM + AQ_NSV) * AQ_STG + 2 * 32 * AQ_LDS * 4 + 2 * 32 * (ATP + 8) * 2;
static_assert(SM_ATTN_Q8 + 16 <= 99 * 1024, "attention ring budget (mk_cycle has a few bytes of static shared memory)");
constexpr int SM_ATTN_MAX = SM_ATTN > SM_ATTN_Q8 ? SM_ATTN : SM_ATTN_Q8;
constexpr int SM_TOTAL = (SM_MISC + 2048 > SM_ATTN_MAX ? SM_MISC + 2048 : SM_ATTN_MAX);
static_assert(SM_TOTAL <= 99 * 1024, "shared memory budget");

enum { T_F32 = 0, T_Q8_0 = 8, T_Q3K = 11, T_Q4K = 12, T_Q5K = 13, T_Q6K = 14, T_IQ4NL = 20, T_IQ3S = 21, T_IQ4X = 23 };
enum { MODE_PREFILL = 0, MODE_DECODE = 1, MODE_BENCH = 2 };

struct Mat {
    const uint8_t * w;
    const __half * d; // per-row block scales for repacked Q6_K / Q8_0
    int type, K, N, rb, nd;
    int fm; // 1: stored in the wv layout (wv.cuh; main-pass matrices of the wv types), read by the wv phases and mmh<.., 1>
};

struct LayerW {
    int attn;
    const float *attn_norm, *post_norm;
    // gated delta net
    Mat qkv, gate, sout;
    const float *ab, *conv, *dt, *A, *snorm; // ab: alpha | beta rows, fp32 with attn_norm folded in, or (UD, Q8_0):
    const int8_t * abq;                      // int8 rows [2 NVH][D] + fp16 scales per 32 (abd): w = (q * d) * attn_norm
    const __half * abd;
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
    // MK_DRAFT_Q4: a Q4_K copy of head_d's first dq4_n rows (row r = lm_head row draft_ids[r], DQ4_RB bytes) that the
    // draft head reads instead; 0 rows: off. Rows past it (tokens dv_add appended) come from lm_head.
    const uint8_t * dq4;
    int dq4_n;
    int embd_t; // token_embd's type: T_Q4K, T_Q6K or T_Q8_0 (ggml rows, embd_rb<type>() bytes)
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
    int tok[MAXD]; // tokens fed to the main model this cycle
    int dn_c;      // delta-net committed position (state covers positions < dn_c)
    int mtp_valid; // MTP kv valid for positions < mtp_valid
    int n_out, out[MAXD];
    int n_acc;
    int mtp_R, mtp_logits, mtp_P;
    int mtp_pos[MAXD + 3], mtp_tok[MAXD + 3]; // MTP catch-up rows (<= T + 1)
    int tgt[MAXD];
    // draft confidence: dconf[k] = the MTP's own softmax probability of draft k + 1 (tok[k + 1]) over the
    // draft vocabulary, dchain = their running product; nd_prod = drafts this launch produced (the
    // early-stop gate may end drafting before Launch::ndraft), nd_stop = the gate fired
    float dconf[MAXD];
    float dchain;
    int nd_prod, nd_stop;
    int dsmp; // the drafts in tok[1..] were sampled from q' (cB.dq; speculative sampling with sampled drafts)
    int pad[4];
};
static_assert(sizeof(Ctl) % 4 == 0 && sizeof(Ctl) / 4 <= NT, "the ctl egress copies Ctl one 4-byte word per thread of a block");

struct Bufs {
    float *x, *xm;
    QBuf qn, qo, qa, qe;
    float *z, *ab, *qbuf, *kbuf, *vbuf;
    float *conv, *kvgb, *dstate;
    void *kc, *vc;     // KV cache [slot][kv head][ctx][HD]: fp16, or int8 (q8_0) when kvq8 ...
    __half *ksc, *vsc; // ... with one fp16 scale per 32 values: [slot][kv head][ctx][HD / 32]
    int kvq8;
    // kvq8: per attention slot (16 layers + MTP) and K / V (MK_KV): kvb bits per value, 8 (q8_0) or 4 (q4: q4_0
    // rounding, 16 levels, low nibbles of a lane's word = values 0..3), kvp its values [kv head][ctx][HD * bits / 8]
    // (slots packed one after the other in kc / vc), kvs its scales [kv head][ctx][HD / 32] (= ksc / vsc + slot offset)
    int8_t * kvp[2][N_ATT_SLOTS];
    __half * kvs[2][N_ATT_SLOTS];
    int kvb[2][N_ATT_SLOTS];
    int kvr[N_ATT_SLOTS]; // K stored Hadamard-rotated (MK_KV "4r"): the queries get the same orthonormal 256-point rotation
    int kvrv[N_ATT_SLOTS]; // V stored Hadamard-rotated (MK_KV "6r" on V): the head's attention output is rotated back
    // MK_KV_WIN=W (recent-window ring): a side (w, slot) whose main buffer is low-bit also keeps its most recent rows as
    // q8 in a ring of kvrc rows per kv head: kvrp [kv head][kvrc][HD] values, kvrs [kv head][kvrc][HD / 32] scales
    // (nullptr: no ring). A tile at position p is read from the ring when p >= kv_ws(L_ctx); kvrc = (Wb + 1) ATT_BUCKET + ATP
    // with Wb = ceil(W / ATT_BUCKET), so the ring always holds every position the window can name (+ ATP: draft rows
    // written past the partition boundary do not overwrite the window's oldest rows).
    int8_t * kvrp[2][N_ATT_SLOTS];
    __half * kvrs[2][N_ATT_SLOTS];
    int kvwin, kvrc;
    float * apart;
    float2 * aml;
    float *h_hist, *hm;
    const float2 * rope; // [ctx][32] (cos, sin) of pos * base^(-2i/64)
    float2 * blk_best; // per block: best (logit, id) per row, or the TOPK_MAX best when sampling
    float * lmscr;     // lm_head<16, true>: the rows' true logits, one 128 B line per (row pair, warp) [VOCAB / 2 + nwarps][32]
    float2 * blk_lse;  // per block: (max, sum of exp) of the draft logits (draft confidence)
    float2 * dlog;     // MK_SPEC_LOG: per draft [MAXD][TOPK_MAX + 1]: its top logits (logit, token id), then (lse, best)
    float2 * dq;       // sampled drafts: per draft [MAXD][TOPK_MAX] its proposal q' as (prob, token id), then (0, -1)
    float2 * cand;     // [MAXT][TOPK_MAX] best (logit, id) of the main lm_head rows, descending (sampling)
    float * logits;    // [MAXT][VOCAB] main lm_head logits of a scoring launch (Launch::score == 2)
    unsigned short * pcnt; // [VOCAB] presence / frequency penalty: occurrences of each token in the response (saturating)
    unsigned * bar;
    unsigned * cnt;
    Ctl * ctl;
    unsigned long long * prof; // optional per-barrier timestamps (block 0)
    int ctx; // KV rows per kv head and RoPE rows (the [ctx] above): the engine's context + KV_PAD
#if MK_KV_EMU
    int kvemu[2][N_ATT_SLOTS]; // per slot, K / V: 0 plain q8, else the kv_store_emu mode (MK_KV_EMU_SPEC)
    int kvemu_sink;            // positions < kvemu_sink stay plain q8 (attention sinks; MK_KV_EMU_SPEC "sink=N")
    int kvemu_win;             // > 0: rows are stored q8 and demoted to their mode once older than kvemu_win ("win=N")
#endif
    int dattn;   // decode attention of all-q8 slots: 1 attn_core_q8 (default), 0 attn_core_t (MK_DATTN=0)
    int att_i8;  // MK_ATT_I8 (q8 cache): 1 int8 Q.K^T for all-q8 slots in every decode attention kernel (aq_i8_s;
                 // exact across depths / lanes, KLD-gated), 0 off (default; always 0 unless built with -DMK_ATT_I8=1)
    int mtp_win; // MK_MTP_WIN > 0: the MTP block's decode attention sees the last mtp_win positions only (drafts only)
    int mtp_kvc; // K / V-only MTP catch-up rows (mtp_kv_pass) on (default), 0 with env MK_MTP_KVC=0
    float * wvpart;  // wv split-K partial sums [WV_S][MAXT][D] (resid_wv)
    unsigned * wvcnt; // [2][D / 32] arrivals per 32-row group (out, ffn_down); the last arrival resets it to 0
    // runtime switches (env, read at init; uniform per launch, outside the inner loops; all bit-exact):
    int k1;          // K1, MK_K1 (default 1): dn_proj_wv deals the alpha / beta items to the warps without a qkv / z tile
    int k2;          // K2, MK_K2 (default 1): resid_wv's last-arrival combine issues MK_K2_TB tokens' partial loads together
    // K6 (MK_K6, bits; attn_proj_wv's dealing, uniform per launch): 2 deal the phase's items across the blocks (item
    // i -> block i % G) instead of in contiguous runs, 4 split each Q8_0 k / v tile into two 8-row items (attn_kv_rh: EXACT).
    // (The host refuses bit 1.)
    int k6;
    int * err; // mapped host words, set only on failure: [0] no finite main logit / invalid token, [1] MTP catch-up out of range
    Ctl * ctl_eg; // mapped host copy of ctl, written by block 0 at the end of every launch (mk_egress)
};

// Buffer views: device code reads cB = view s_view of its block (MK_VIEW_INIT at every kernel's entry copies cView).
// View 0: the engine's own sequence (solo launches, prefill), also the shared row buffers of a batched launch. Lanes
// (MKEngine::slots_open) k = 0 .. MK_NSEQ - 1: view 1 + k = lane k's MAIN view in a batched launch (the row buffers
// start at the lane's first row of the pass: uploaded per launch), view 1 + MK_NSEQ + k = its FULL view (row buffers from
// row 0: prefill and solo launches of the lane, its MTP passes and control inside a batched launch).
#ifndef MK_NSEQ
#define MK_NSEQ 4
#endif
constexpr int NSEQ = MK_NSEQ;
constexpr int NVIEW = 1 + 3 * NSEQ + (MAXD > MAXT);
__host__ __device__ constexpr int view_hi() { return 1 + 3 * NSEQ; } // (MAXD > MAXT: the deep pass's rows MAXT ..)
__host__ __device__ constexpr int view_main(int k) { return 1 + k; }
__host__ __device__ constexpr int view_full(int k) { return 1 + NSEQ + k; }
__host__ __device__ constexpr int view_mtp(int k) { return 1 + 2 * NSEQ + k; } // (lane k's full view, row buffers at row k)

struct Launch {
    int mode, T, last, ndraft;
    int score; // prefill only: greedy argmax for every row (teacher-forced evaluation), no MTP; 2: also their logits
    // sampling (temp > 0; greedy otherwise), applied in llama.cpp's order: top_k, top_p, min_p, temperature
    float temp, top_p, min_p;
    int top_k; // <= TOPK_MAX
    unsigned long long seed;
    // draft early stop on the MTP's confidence f (its softmax top-1 probability, or with gate_chain the
    // running product over this launch's drafts), per draft index k: f < gate_drop[k] drops the draft and
    // stops drafting, f < gate_stop[k] keeps it and stops. All 0: off (always Launch::ndraft drafts).
    float gate_drop[MAXD - 1], gate_stop[MAXD - 1];
    int gate_chain;
    // sampling: draft from the MTP's own sampling distribution q' and verify with min(1, p / q') plus the
    // residual max(0, p - q') (distribution-exact; higher acceptance than argmax drafts at temp 1)
    int spec_smp;
    // presence / frequency penalty (vLLM's semantics, OpenAI's parameters): the logit of token v is lowered by
    // freq * c(v) + pres * (c(v) > 0), c(v) = how often v occurs in this response's generated tokens before the row's
    // position (Bufs::pcnt: the committed ones; a verify row adds the drafts in its own context). Applied to the main
    // decode rows and to the MTP draft head before the argmax / top-k; 0 / 0: off (lm_head skips the penalty code).
    float pres, freq;
    // batched launch (MKEngine::slots_step; 0: a solo launch): nl lanes, lane i = ln[i].lane (views view_main / view_full)
    // with its rows r0 .. r0 + T - 1 of the pass (Launch::T = the pass's rows, at least their sum), the drafts its MTP
    // passes produce for the next launch, and its sampling
    int nl;
    int mtpb; // batched launch: the lanes' MTP passes shared (mtp_batch; 0: one lane after the other, mtp_tail)
    struct Lane {
        int lane, T, r0, nd, top_k, spec_smp;
        float temp, top_p, min_p;
        unsigned long long seed;
    } ln[NSEQ];
};
constexpr int TOPK_MAX = 32; // candidates kept per main lm_head row when sampling
constexpr int LM_RB = 16;    // rows per block in Bufs::blk_best: lm_head<16> (the deep and the 16-row batched passes)
static_assert(MAXD <= LM_RB && MAXT <= LM_RB, "blk_best rows");

enum {
    PH_EMBED, PH_ATTN_PROJ, PH_ATTN_CORE, PH_ATTN_COMB, PH_RESID_O, PH_DN_PROJ, PH_DN_CORE, PH_RESID_SOUT, PH_FFN_GU,
    PH_FFN_DOWN, PH_LMHEAD, PH_MTP_M0, PH_MTP_EH, PH_MTP_ATTN_PROJ, PH_MTP_ATTN_CORE, PH_MTP_ATTN_COMB, PH_MTP_RESID_O,
    PH_MTP_FFN_GU, PH_MTP_FFN_DOWN, PH_MTP_LMHEAD, PH_MTP_KV_M0, PH_MTP_KV_EH, PH_MTP_KV_PROJ, PH_COUNT
};
// the grid barrier's go word, cB.bar + GS_GO (gs_arrive_wait)
constexpr int GS_GO = 32;
// repacked Q6_K / native Q3_K / native IQ3_S bytes per 256-block (rows_q6k, rows_q3k, rows_iq3s)
constexpr int Q6_BLK = 192, Q6_META = 18;
constexpr int Q3_BLK = 96;
constexpr int IQ3S_META = 16;
constexpr int EMB_UB = 16; // units per embed block
// Grid size (one block per SM): phases that hand one block to each item (embed_prologue U_D / EMB_UB blocks per row,
// dn_core one per DeltaNet head, the MTP M0 MTP_M0_NB per catch-up row, up to 5 rows) skip items silently on a smaller
// grid, so the engine refuses a GPU with fewer SMs than mk_min_grid(rows per launch) at load (slots_open: 2+ slots,
// mk_min_grid(MAXT)). lm_head's per-block (max, sum) table (SM_LSE) and the MK_PROFILE stamps hold MK_MAX_GRID blocks:
// a larger GPU runs that many.
constexpr int MTP_M0_NB = 10;
constexpr int MK_MAX_GRID = 128;
constexpr int mk_min_grid(int rows) {
    return rows * (U_D / EMB_UB) > NVH && rows * (U_D / EMB_UB) > 5 * MTP_M0_NB ? rows * (U_D / EMB_UB) : NVH > 5 * MTP_M0_NB ? NVH : 5 * MTP_M0_NB;
}
static_assert(mk_min_grid(MAXT) <= MK_MAX_GRID, "grid size");
constexpr int DQ4_RB = D / 256 * 144; // MK_DRAFT_Q4: bytes of one Q4_K lm_head row (ModelW::dq4)
// the slot rows of a lane verifying T tokens in a batched pass: T <= 5 as they are, else an 8-row slot (padding rows, as a
// solo MAXT-row pass: the host keeps the slot's rows in one attention partition)
__host__ __device__ constexpr int lane_slot(int T) { return T <= 5 ? T : MAXT; }

// the wv layout (wv.cuh)
// fragment-ordered (FM) types; the other wv types are tile-major whole blocks
__host__ __device__ constexpr bool wv_fm(int t) { return t == T_IQ4X || t == T_Q4K || t == T_Q5K; }
// FM Q6_K units (wv.cuh "FM Q6_K")
constexpr int Q6F_UNIT = 3360, Q6F_D = 256, Q6F_LO = 288, Q6F_HI = 2336;
constexpr int WV_S = 4; // split-K chunks of out / ffn_down (fixed for every T)
constexpr int C1_W0 = 2 * GS_GO, C1_WORDS = C1_W0 + WV_S * GS_GO; // (cB.bar words incl. gsync's count / go lines)

// batched prefill (prefill.cuh, prefill_attn.cuh, prefill_gemm.cuh)
struct PBufs {
    int cap;      // token capacity of a chunk (multiple of 128)
    float * x;    // [cap][D] residual stream
    float * xm;   // [cap][D] MTP stream
    float * rs;   // [cap] RMS scale of the last normed input
    __half * xh;  // [cap][FF] fp16 GEMM input, elements of each 16-group in pf::hperm order
    __half * xh2; // [cap][FF] fp16 silu(gate) * up (ffn_down input), same order
    float * y0;   // [cap][FF]  qkv | q+gate | ffn gate | logits ([cap][QG] when every gate / up pair is fused)
    float * y1;   // [cap][FF]  z | k,v | ffn up
    float * y2;   // [cap][VD]  delta-net / attention output
    __half * dq;  // [cap][QKV] conv'd + normalised q | k | v (fp16)
    float * gb;   // [cap][2 * NVH] log g | beta
    float * ab;   // [AB_SPLITS][cap][2 * NVH] alpha | beta pre-activations (split-K partials)
    float * h;    // [cap + 1][D] final normed hidden of positions n0 - 1 .. n0 + T - 1
    __half * qh;  // [cap][NH][HD] attention queries (normed, rotated, pre-scaled)
    int * tok;    // [cap]
    int * erow;   // [cap] input-embedding row of each token in emb (MKImage rows), -1: its token_embd row
    const float * emb; // [rows][D] the prefill's image rows (mapped pinned host memory; null until the first image)
    int * amax;   // [cap] argmax output (scoring)
    int wsrow;    // MK_EXACT: the MK_KV_WIN ring / main split per 16-row query tile (kv_ws of its last row, as decode
                  // takes it per row), not per chunk (kv_ws(n0 + T)): a row's keys then do not depend on its chunk
};
// pk_attn2: query rows / key rows per tile, row stride, threads; floats per split-K partial
constexpr int A2_Q = 16, A2_K = 32, A2_LD = HD + 8, A2_NT = 384;
constexpr int A2_PART = 6 * A2_Q * HD + 2 * 6 * A2_Q; // floats per partial: o [96][HD], then (m, l) [96]

namespace pf {
#if MK_PF_INFCHK
enum { CHK_Q, CHK_K, CHK_V, CHK_O, CHK_QKV, CHK_GATE, CHK_SOUT, CHK_GU, CHK_UP, CHK_DOWN, CHK_EH, CHK_HEAD, // GEMMs
       CHK_IN_NORM, CHK_IN_DN, CHK_IN_MTP, CHK_IN_ATTN, CHK_IN_GLU,                                         // inputs
       CHK_DN_S, CHK_DN_KS, CHK_DN_U, CHK_DN_UD, CHK_SITES }; // the DeltaNet scan's fp16 MMA operands (S, K S, U, U d)
constexpr int CHK_LAYERS = 66; // 64 trunk layers, 64 = the MTP block, 65 = the output head
#endif
constexpr int BM = 128, BN = 128, BK = 64;
constexpr int TQ4K = 12, TQ5K = 13, TQ6K = 14, TQ80 = 8, TIQ4 = 23, TIQ4NL = 20, TQ3K = 11, TIQ3S = 21;
struct WMat { // device view of one weight matrix (mirrors Mat)
    const uint8_t * w;
    const __half * d;
    int type, K, N, rb, nd;
};
enum { OUT_STORE = 0, OUT_ADD = 1 };
struct Out {
    float * y; // y[tok * ldy + row]
    int ldy;
    int mode;
    __half * yh; // fused gate / up (mmh<.., true>): yh[tok * ldy + row] = silu(gate) * up, fp16 in hperm order
};
} // namespace pf

// What the two sides exchange as bytes or must agree on (struct sizes, build settings, GPU architectures): the host
// refuses a device object built with other settings (mkl::abi_sig) instead of reading garbage.
constexpr unsigned long long mk_abi_sig() {
    const unsigned long long v[] = {sizeof(ModelW), sizeof(Bufs), sizeof(PBufs), sizeof(Launch), sizeof(Ctl), sizeof(pf::WMat),
                                    sizeof(pf::Out), NVIEW, MAXT, MAXD, RMAX, NSEQ, MK_SPEC_LOG, MK_TEST_NAN, MK_TEST_SLOTS_FAIL, MK_KV_EMU,
                                    MK_PF_INFCHK, MK_ATT_I8, MK_L2_TUNE,
#ifdef __CUDA_ARCH_LIST__
                                    __CUDA_ARCH_LIST__, // (nvcc: the -arch list, e.g. 860)
#endif
    };
    unsigned long long h = 0xcbf29ce484222325ull;
    for (unsigned long long x : v) h = (h ^ x) * 0x100000001b3ull;
    return h;
}

} // namespace MK_SHARED_NS
