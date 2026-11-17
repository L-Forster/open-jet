// MKEngine::Impl: the engine's host-side state and its members' declarations (definitions: engine.cu and the group
// files it lists). Host files only: the device TU (qwen38_mk.cu) is reached through mk_launch.h.
#pragma once

#include <array>
#include "mk_shared.cuh"
#include "mk_launch.h"

using namespace mk;

// the loader's host helpers (types of Impl's members): one definition for every host file
namespace mk {
#include "loader.cuh"
#include "wv_host.cuh"
} // namespace mk

struct MKEngine::Impl {
    gguf::File file;
    LoadClock lc;                         // load phases (MK_LOADLOG=1 prints them)
    lclk::time_point t_load = lclk::now(); // (after the GGUF parse)
    ModelW hw{};
    Bufs hb{};
    std::vector<void *> allocs;
    void * embd_host = nullptr;
    int ctx;
    int nsm = 0;
    float smp_temp = 0.f, smp_top_p = 1.f, smp_min_p = 0.f; // sampling (temp 0: greedy)
    int smp_top_k = TOPK_MAX;
    unsigned long long smp_seed = 0;
    int ndraft;          // MTP drafts the next launch produces (at most; the gate may stop earlier)
    int nd_avail;        // drafts produced by the previous launch (verified by the next decode cycle)
    bool all_acc = true; // the last decode cycle accepted all of its drafts
    int last_acc = 0;    // drafts the last decode cycle accepted
    // After book_set (unpark, side_unpark, side_end) the device holds another sequence's decode-only state. stale_replay:
    // the DeltaNet replay rows of the drafts accepted since dn_c (kvgb) and the drafts' q' (dq), until the sequence's next
    // launch rewrites them; stale_pcnt: the response's penalty counts (pcnt), until its next prefill restarts them.
    // step_live refuses a step that would read them; a prefill rebuilds all of it (it refeeds from dn_c).
    bool stale_replay = false, stale_pcnt = false;
    // draft-depth policy (MKEngine::set_policy). Default: up to n_draft drafts, the on-device gate on the running
    // product c_k of the MTP's own top-1 probabilities stops drafting after draft k >= 2 when c_k < 0.4 and drops
    // the 4th draft (the T = 5 verify row) when c_4 < 0.4; sampled drafts when sampling. Tuned on replayed logs
    // + live A/B. "depth=0,drop=0,stop=0,chain=0,h1=0": depth rule 0, no stop gates, top-1 drafts when sampling; the
    // one gate left drops the deepest draft when its own probability (chain=0, not the product) is below droplast 0.4.
    struct Policy {
        int depth = 1;       // 0: n_draft - 1 after a cycle with a rejection (at least 3), 1: always n_draft,
                             // 2: 1 + the accepted count of the last cycle (clamped to [dmin, n_draft])
        int dmin = 1;
        // generalised gate (any depth): stop after draft k >= 1 below 0.4, drop the deepest draft (k = n_draft - 1, the
        // T = n_draft + 1 verify row) below drop_last (at n_draft 4: the default gates above)
        static std::array<float, MAXD - 1> stop_init() {
            std::array<float, MAXD - 1> a{};
            for (int k = 1; k < MAXD - 1; k++) a[k] = 0.4f;
            return a;
        }
        std::array<float, MAXD - 1> gate_drop{}, gate_stop = stop_init();
        float drop_last = 0.4f;
        // context cap: from position dctx on, at most 4 drafts (T <= 5: at long context the wide pass's second
        // head-group pass over K / V costs more than drafts 5..7 gain)
        int dctx = INT_MAX;
        // g7 (default on): at depth > 5 (the wide tier T 6-8 costs the same per cycle, a draft there only its ~0.5 ms MTP
        // pass) the gates are stop after drafts 2-4 below 0.25, drop draft 5 below 0.3 (stay at T <= 5), stop after drafts
        // 5-6 below 0.3, never drop the deepest draft (from an offline policy search on open-policy logs; greedy output
        // is unchanged). Any explicit stop / drop key turns it off.
        int g7 = 1;
        float dstop = 0.5f, tier = 3.0f; // deep chain: g7's stop past draft 7; drafts 8.. verified if their chain sum >= tier
        int gate_chain = 1;
        int h1 = 1; // sampling: sampled drafts (Launch::spec_smp)
    } pol;
    // the policy's draft depth at position pos: n_draft, or at most 4 from Policy::dctx on
    int depth_top(int pos) const { return pos >= pol.dctx ? std::min(n_draft_max, 4) : n_draft_max; }
#if MK_SPEC_LOG
    FILE * slog = nullptr;               // MK_SPEC_LOG_FILE: one JSON line per decode cycle
    std::vector<float2> slog_q;          // dlog of the drafts the next cycle verifies
    void slog_drafts();
#endif
    std::vector<int> seq; // the sequence so far: prompt(s) and emitted tokens (position = index)
    // Images (MKEngine::MKImage): spans of positions whose input is an embedding row, with M-RoPE positions. llama.cpp's
    // layout for Qwen-VL (mtmd, IMROPE): the span's rope base r is the position after the text before it; raster row q
    // has (t, h, w) = (r, r + q / nx, r + q % nx), RoPE frequency i takes t, h, w for i % 3 = 0, 1, 2 (sections 11 / 11 /
    // 10 of the 32 pairs); the text after the span continues at r + max(nx, ny), so each span shifts the rope position of
    // everything after it down by nx * ny - max(nx, ny). Attention stays causal in sequence order (llama.cpp's 2D mask
    // over (t, h, w) orders a span's rows by raster position, which is their sequence order). Only hb.rope changes: its
    // row n holds the (cos, sin) of position n's rope angles, so the decode and prefill kernels index it by sequence
    // position (as for text). rope_base keeps the plain table (row p = position p): every rewritten row is a gather from it.
    struct Span {
        int pos, nx, ny;
        bool operator==(const Span & o) const { return pos == o.pos && nx == o.nx && ny == o.ny; }
        int n() const { return nx * ny; }
    };
    std::vector<Span> spans;      // the images in seq (ascending, disjoint)
    std::vector<Span> rope_spans; // the spans hb.rope's rows reflect
    std::vector<float2> rope_base;
    float * emb_host = nullptr;   // the prefill's image rows: pinned, mapped (cP.emb = its device alias)
    size_t emb_cap = 0;           // (floats)
    volatile int * err_host = nullptr; // cB.err's host side
    volatile Ctl * ctl_eg_host = nullptr; // cB.ctl_eg's host side
    bool bar_live = false;                // the cB.bar counts run on from the previous launch (zeroed once, and after bench launches)
    // ctl after a decode launch: the kernel's mapped mirror (a device-to-host copy without one)
    void ctl_fetch(Ctl & c);
    bool err_mtp_warned = false;
#if MK_TEST_NAN
    int test_nan_at = getenv("MK_TEST_NAN_AT") ? atoi(getenv("MK_TEST_NAN_AT")) : 0, test_nan_n = 0;
#endif
    void launch_failed();
    // Rewrites hb.rope's rows from the first position where `spans` and `rope_spans` differ (none: no copy). Text rows are
    // rope_base rows shifted by the images before them, image rows per-frequency gathers; pageable host-to-device copies
    // on the legacy stream, ordered after every kernel that read the old rows.
    void rope_sync();
    // the sequence goes back to `len` positions (reset, rewind, truncate): the images from there on are dropped; one cut
    // by `len` stays (its rows before len are processed; a prefill continuing inside it must supply it again)
    void spans_cut(int len);
    // DeltaNet checkpoints: the state after position pos - 1 (recurrent state, conv ring, h ring, ctl), which lets a
    // caller rewind to pos (KV rows before it are never rewritten) and prefill from there. A pool of MK_CK_SLOTS
    // (default 8) checkpoints of 0.17 GB each in pinned host memory (MK_CK_DEVICE=1: one, in device memory); when it
    // is full, ck_assign evicts by coverage. Saving never waits on the host and adds no VRAM:
    // - Inside a prefill (pf_run, Snap): at position P of a chunk, each DeltaNet layer's scan is split at P and its
    //   first part writes the state at P into a staging slot in hp.xm (idle until the chunk's MTP part), next to the
    //   conv ring at P (pk_ck_conv); a device-to-host copy on ck_sv moves the staging slot into the checkpoint while
    //   the next layers run. The h ring at P goes straight into the mapped host buffer (pk_ck_hist). Stream 0 waits
    //   (on the device) only before reusing a staging slot and before the MTP part. The scan is split only at points
    //   of its 32-token grid, which changes no arithmetic: saving a checkpoint never changes what the prefill computes.
    //   A position off the grid (the exact ck_at of a chat turn) is saved by a side scan from the grid point before
    //   it (the state a prefill ending there computes).
    // - At the end of a prompt (ck_save_end): a device-to-device snapshot into the idle batched-prefill scratch
    //   on stream 0 (~0.4 ms), then the device-to-host copy on ck_sv, which runs under the first decode launch.
    // - Restore: the host-to-device copy runs on ck_st in CK_NG groups of DeltaNet layers, issued once the next
    //   prefill has uploaded its tokens (ck_issue_restore: nothing on stream 0 then queues behind it on the copy
    //   engine); stream 0 waits for a group's event only before the first kernel that reads that group's state
    //   (pf_run: dn_prep of the group's first layer). Same bytes, same order of use: bit-identical.
    // If the scratch is too small for the end snapshot (MK_PF_CAP below ~640), the end save copies straight from the
    // state and stream 0 waits for it before the state is next modified; below 384 there are too few staging slots
    // too: only ck_exact is saved inside a prompt, by two pf_runs around it (MK_EXACT: no checkpoint inside it).
    static constexpr size_t CK_SL = (size_t) NVH * 128 * 128, CK_CL = (size_t) CONV_RING * QKV; // floats per DeltaNet layer
    static constexpr size_t CK_NS = 48 * CK_SL, CK_NC = 48 * CK_CL, CK_NH = (size_t) H_RING * D;
    static constexpr int CK_NG = 5;
    static constexpr int ck_gb[CK_NG + 1] = {0, 1, 4, 12, 24, 48}; // restore group g: DeltaNet layers [ck_gb[g], ck_gb[g + 1])
    struct CkSlot {
        void * mem = nullptr;                                       // pinned host block (device memory: MK_CK_DEVICE)
        float *dstate = nullptr, *conv = nullptr, *hhist = nullptr; // in it
        float * hdev = nullptr;                                     // hhist as kernels address it
        Ctl ctl{};
        int pos = -1;                                               // -1: free
        cudaEvent_t ev = nullptr;                                   // the copies into this slot (ck_sv) are done
        bool pending = false;                                       // ... may still be running
        char * ring = nullptr;                                      // MK_EXACT + MK_KV_WIN: the q8 window rings at pos
        bool ring_ok = false;                                       // ring holds this checkpoint's rings
        bool pin = false;                                           // pinned (pin_set; meaningful while pos >= 0)
        unsigned pin_t = 0;                                         // ... by the request with this stamp
    };
    struct Snap { int pos, slot; };
    std::vector<CkSlot> ck;
    bool ck_dev = false;
    // MK_EXACT=1 (exact reuse): every result of a prefill row is independent of how the sequence was cut into prefill
    // calls and chunks, so a prefill continued from a checkpoint computes bit for bit what a prefill from 0 computes:
    // - the prefill attention runs unsplit (no a2_plan tail split: one online softmax per row over absolute tiles),
    // - the MK_KV_WIN ring / main split is taken per query tile (PBufs::wsrow), as decode takes it per row,
    // - checkpoints sit on the absolute 32-token scan grid (ck_exact is snapped like the others; callers restore only
    //   checkpoints of a prefill that started on the grid, which every restore and a prefill from 0 do) and are only
    //   saved inside the batched prefill (no end-of-prompt save: the prompt's last row runs through the megakernel),
    // - with MK_KV_WIN, a checkpoint also keeps the q8 window rings as of its position (a restore then needs no
    //   lossy refill from the low-bit main buffer); pinned host, ~NKV * kvrc * 272 B per ring side and slot.
    // The caller (mk_server --exact-reuse) never appends to decoded rows: it restores the last checkpoint at or
    // before the divergence and prefills from there (the reply's rows were written by the decode kernels).
    bool exact = false;
    bool exact_off_grid = false; // a batched prefill started off the grid (warned once)
    struct RingSide { int8_t * q; __half * s; size_t off; }; // device ring values / scales, offset in CkSlot::ring
    std::vector<RingSide> rsides[N_ATT_SLOTS];
    size_t ring_bytes_ck = 0;          // per checkpoint
    cudaEvent_t rs_ev[N_ATT_SLOTS] = {}; // pf_run: a chunk's whole-ring copies of slot s were issued (ck_sv)
    cudaEvent_t rs_ev_pf = nullptr, rs_ev_done = nullptr;
    bool rs_busy = false;              // ring copies on ck_sv may still read rings stream 0 will overwrite
    size_t ring_vb() const { return (size_t) NKV * hb.kvrc * HD; }
    size_t ring_sb() const { return (size_t) NKV * hb.kvrc * (HD / 32) * sizeof(__half); }
    double ck_ms = 0;             // last save / restore enqueue time (MK_MEMLOG)
    cudaStream_t ck_st = nullptr; // restores
    cudaStream_t ck_sv = nullptr; // saves
    cudaEvent_t ck_ev_src = nullptr, ck_ev_sv = nullptr, ck_ev_scr = nullptr, ck_ev_grp[CK_NG] = {};
    bool ck_scr_busy = false; // a save copy on ck_sv may still read the prefill scratch (ck_ev_scr)
    int ck_direct = -1;       // slot of an end save that reads the live state directly: stream 0 waits before modifying it
    int ck_rs_slot = -1;      // slot of a restore whose copies are not issued yet (ck_issue_restore)
    int ck_rs_src = -1;       // slot of the last restore (its copies may be running until its prefill waited for them)
    int ck_rs_next = CK_NG;   // first restore group stream 0 has not waited for (CK_NG: no restore pending)
    struct CkChunk { char * dev; char * scr; size_t off, n; };
    std::vector<CkChunk> ck_plan; // end save: state -> prefill scratch (stream 0) -> slot byte offset off (ck_sv)
    // staging slots of in-prefill saves in hp.xm: one DeltaNet layer's state + conv ring each
    static constexpr size_t STG_F = (CK_SL + CK_CL + 63) & ~(size_t) 63;
    static constexpr int STG_MAX = 16;
    int stg_n = 0, stg_next = 0, stg_last = -1;
    cudaEvent_t stg_ev[STG_MAX] = {};
    bool stg_busy[STG_MAX] = {};

    // Allocated at construction: cudaHostAlloc of 0.17 GB takes 0.1-0.5 s under WSL2, which would otherwise land
    // in the first prompt's time. A pinned slot that fails to allocate ends the pool there (the first: one device slot).
    void ck_alloc();
    void ck_free();
    // end-save snapshot plan over the prefill scratch buffers (after pf_init; empty when they are too small) and
    // the staging slots in hp.xm
    void ck_make_plan();
    // stream 0: before overwriting the prefill scratch, wait for an end save's copy still reading it
    void ck_scr_guard();
    // stream 0: before modifying the live state, wait for an end save that reads it directly (no snapshot)
    void ck_guard_state();
    // MK_EXACT + MK_KV_WIN, pf_run chunk n .. n + T - 1 with checkpoints sp[0 .. sn) in slots ss: a checkpoint's rings as
    // of its position P = the rings before the chunk stores its K / V rows (positions P - kvrc .. n - 1: whole-ring
    // copies, issued at the chunk start) with the rows of positions n .. P - 1 laid over them after the slot's
    // pk_attn_prep (ring_snap_rows). All copies on ck_sv; stream 0 waits (rs_ev[s]) before slot s's prep overwrites
    // its rings, and (ring_snap_guard) before a later chunk or launch could overwrite rows a copy still reads.
    int ring_grp[N_ATT_SLOTS] = {}; // restore group after whose wait stream 0 first touches slot s's rings
    void ring_snap_begin(int sn, const int * ss);
    void ring_snap_rows(int s, int n, int sn, const int * sp, const int * ss);
    void ring_snap_guard();
    // Issues a due restore's host-to-device copies on ck_st (5 groups, an event after each). Deferred from ck_restore
    // to just after the next prefill's token upload: a host-to-device copy on stream 0 (ctl, token ids, draft
    // vocabulary rows) queued behind the 0.17 GB on the copy engine would stall the host for the whole restore.
    void ck_issue_restore();
    // stream 0: wait for the restore groups up to the one holding DeltaNet layer dl (-1: all of them)
    void ck_wait_restore(int dl = -1);
    // checkpoint at the end of a prompt into slot si (prefill read ctl back after its last launch: hctl is the device's)
    void ck_save_end(int si);
    void ck_restore(int si);
    // MK_KV_WIN: the sequence goes back to position P (checkpoint restore, truncate). The abandoned positions P .. kv_top - 1
    // (kv_top: every launch's rows, draft rows included) occupied the ring slots of the window rows P - kvrc .. kv_top - kvrc - 1,
    // so exactly those rows are refilled from the main buffer (they stay at its precision until they age out of the window;
    // a fresh prefill of the same text has them at q8). Positions >= P are rewritten before they are read.
    int kv_top = 0;
    void ring_rewind(int P);
    int ck_find(int pos) const;
    // Pinned checkpoints (MKEngine::ck_pin): the caller's checkpoint at the end of a prompt's shared prefix (a client's
    // system prompt and tools), kept against ck_assign's thinning and carried by every park of the sequence, so a new
    // session of the same client resumes there however long the conversation grew. A request for position P pins the
    // checkpoint at the last position in (P - 32, P] (where ck_positions moves P, whatever the chunk start); at most
    // pin_max (min(2, slots / 4)) are pinned, more release the least recently requested ones. The flag goes with its
    // checkpoint (side requests hide it with the kept sequence, parks carry it, a rewind before it or a reset drops it
    // with the checkpoint), so a pin always covers a prefix of the live sequence.
    int pin_max = 0;
    unsigned pin_clock = 0, pin_stamp = 0; // the requests' stamps; the last ck_pin's
    std::vector<int> pin_req;              // pins for the next prefill to save (ck_pin; that prefill clears them)
    int pin_count() const;
    // the slot of the checkpoint a pin request for P covers (-1: none)
    int pin_slot(int P) const;
    // pins slot si for the request stamped t; beyond pin_max the least recently requested other pin is released
    void pin_set(int si, unsigned t);
    // Slots for checkpoints at the new positions `want` (ascending): free ones, else evicting by coverage. While
    // there are more checkpoints (existing + new) than slots, the one dropped is the one (not `keep`; the slot of a
    // restore whose copies may still run only if nothing else is left) whose removal leaves the smallest gap relative
    // to its distance from the end E: (next - prev) / (E - prev). Old checkpoints thin out geometrically, recent ones
    // stay dense. Returns the slot of each new position (-1: dropped); evicted slots become free.
    // A side request (sw_on) takes free slots that hold none of the kept sequence's checkpoints first (one holding a
    // kept pinned checkpoint last); a slot that does gets its block saved to the host store before it is handed out
    // (sw_ck_take; no room: no checkpoint there). Pinned checkpoints and the new positions wpin marks (pin requests) are
    // never dropped.
    std::vector<int> ck_assign(const std::vector<int> & want, int keep, int E, const std::vector<char> & wpin = {});
    // staging slot for one DeltaNet layer's piece of an in-prefill checkpoint (stream 0 waits for its previous copy)
    float * stg_state(int s) { return hp.xm + (size_t) s * STG_F; }
    float * stg_conv(int s) { return stg_state(s) + CK_SL; }
    int stg_take();
    // copies DeltaNet layer dl's staged pieces (staging slots sg[j] -> checkpoint slots ss[j]) on ck_sv
    void stg_flush(int dl, int sn, const int * sg, const int * ss);
    // stream 0, before hp.xm is used again (a chunk's MTP part): every staged copy has finished (ck_sv is in order)
    void stg_drain();
    // draft vocabulary: ids of head_d rows (base list + tokens added from the sequence)
    std::vector<int> dv_ids;
    std::vector<uint8_t> dv_in;
    int dv_base = 0;
    // dv_ids is what a fresh prefill of seq[0, dv_n) builds (the base list, then ONE dv_add of its text tokens), or -1:
    // not (step's additions). A rewind / truncate rebuilds it that way, and a prefill continuing at dv_n rebuilds it over
    // the whole new sequence, so a reused prefix drafts (and samples) exactly as the same prompt sent fresh.
    int dv_n = 0;
    int * dv_pin = nullptr; // dv_add's pinned staging: the ids (dv_base + DV_EXTRA), then [dv_base + DV_EXTRA] the row count

    // Adds the tokens not yet in the draft head's vocabulary (appends their ids).
    void dv_add(const int * toks, size_t n);
    void dv_reset();
    // dv as a fresh prefill of seq[0, n) with image spans sp builds it (positions inside an image hold placeholder ids:
    // skipped, as prefill skips them)
    void dv_rebuild(int n, const std::vector<Span> & sp);
    // back to the first n ids (n >= dv_base, even: the size after a dv_add; a cancelled prefill's additions)
    void dv_truncate(size_t n);
    Ctl hctl{};

    // Watchdog. Every wait for the device in the run-time paths goes through wait_device(), which records an
    // event and polls it with a deadline (MK_WATCHDOG: seconds per decode launch, default 60; prefill chunks and
    // checkpoint copies get at least 120 s; 0 = the runtime's plain blocking waits). The CUDA runtime's own waits
    // never time out: after a GPU / driver fault the WSL2 guest is not told about (nvlddmkm event 153 on the Windows
    // side, no error in the guest) the host thread would spin indefinitely with an idle GPU.
    // The watchdog turns that into a message and exit code 3. The first wd_spin_s of a wait spin (a decode launch
    // finishes in 20-35 ms, so no latency is added); after that it sleeps between polls.
    cudaEvent_t wd_ev = nullptr;
    double wd_limit = 60.0, wd_spin_s = 0.25;
    int wd_T = 0, wd_mode = -1;
    void wait_device(const char * what, double limit);
    // the same deadline for an event recorded elsewhere (the checkpoint copy stream)
    void wait_event(cudaEvent_t ev, const char * what, double limit);

    // MK_MEMLOG=1: device memory accounting. Each mark prints the bytes dalloc'd since the previous mark
    // and the drop of cudaMemGetInfo's free memory over the same span (the difference is allocation
    // granularity, module code, local memory or another process).
    const bool memlog = getenv("MK_MEMLOG") != nullptr;
    size_t alloc_bytes = 0, mark_alloc = 0, mark_free = 0;
    long launches_seen = 0, pf_runs_seen = 0;
    void mem_mark(const char * tag);
    // MK_RESERVE_EARLY=1 (loader.cu): the local-memory reservation (stack limit stack_res, mk_cycle's frame) at the context and
    // the kernels' lazy loads before the KV cache is sized, which is then checked against what is really left (it refuses
    // with the -c that fits); vram_check warns at load end and after the first launch if the device is full
    bool reserve_early = false;
    size_t stack_res = 0;
    void vram_check(const char * where);

    size_t wbytes = 0;
    bool profiling = false;
    std::vector<double> ph_ns = std::vector<double>(PH_COUNT, 0.0);
    std::vector<long> ph_n = std::vector<long>(PH_COUNT, 0);
    double launch_ns = 0;
    long launches = 0;
    // min over launches of each barrier-to-barrier interval, keyed by (T, barrier index)
    std::vector<std::vector<double>> best_ns = std::vector<std::vector<double>>(MAXD + 1);
    std::vector<std::vector<int>> best_tag = std::vector<std::vector<int>>(MAXD + 1);
    std::vector<std::vector<double>> best_fill = std::vector<std::vector<double>>(MAXD + 1);
    std::vector<double> arr_first = std::vector<double>(PH_COUNT, 0.0), arr_med = std::vector<double>(PH_COUNT, 0.0),
                        arr_last = std::vector<double>(PH_COUNT, 0.0);
    std::vector<long> arr_n = std::vector<long>(PH_COUNT, 0);

    // Device buffers come from the default memory pool (stream-ordered allocator, never trimmed): it packs the
    // allocations, where cudaMalloc rounds each one up to whole 2 MiB pages (~0.5 GB over the ~900 weight tensors).
    // MK_CUDAMALLOC=1: plain cudaMalloc.
    bool pool_init = false, use_pool = true;
    int cuda_dev = 0;    // the device current at construction (a caller's cudaSetDevice)
    double dalloc_s = 0; // time in dalloc (load report)
    // MK_POISON=1 (test runs, read-before-write check): every device allocation is filled with 0xFF bytes (NaN in fp32 /
    // fp16, -1 in integers) instead of zeros, the weight buffers too (their padding), except DA_INIT0 buffers, whose zeros
    // are their initial value (arrival counters). An output that differs from a normal run's read memory nothing wrote.
    // MK_POISON_SKIP=i,j,..: allocations number i, j zero-filled instead (bisecting); MK_POISON_LOG=1 lists every one.
    static constexpr int DA_INIT0 = 2; // dalloc fill: zeros that are the buffer's initial value (kept under MK_POISON)
    bool poison = false;               // (read on the first dalloc)
    std::vector<long> poison_skip;
    long n_dalloc = 0;
    template <class T> T * dalloc(size_t n, int zero = 1) {
        void * p;
        if (!pool_init) {
            pool_init = true;
            poison = getenv("MK_POISON") && atoi(getenv("MK_POISON")) != 0;
            if (poison) {
                for (const char * s = getenv("MK_POISON_SKIP"); s && *s; s = strchr(s, ',') ? strchr(s, ',') + 1 : "")
                    poison_skip.push_back(atol(s));
                fprintf(stderr, "[mk] MK_POISON=1: device allocations filled with 0xFF bytes (test mode; %zu skipped)\n", poison_skip.size());
            }
            const char * cm = getenv("MK_CUDAMALLOC");
            int ok = 0;
            cudaDeviceGetAttribute(&ok, cudaDevAttrMemoryPoolsSupported, cuda_dev);
            use_pool = ok && !(cm && cm[0] == '1');
            if (use_pool) {
                cudaMemPool_t pool;
                CK(cudaDeviceGetDefaultMemPool(&pool, cuda_dev));
                unsigned long long thr = ~0ull; // keep freed memory in the pool (release() hands it back to the driver)
                CK(cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &thr));
            }
        }
        const auto ta = lclk::now();
        if (use_pool) {
            CK(cudaMallocAsync(&p, n * sizeof(T) + 16, 0));
            CK(cudaStreamSynchronize(0));
        } else {
            CK(cudaMalloc(&p, n * sizeof(T) + 16));
        }
        allocs.push_back(p); // (before the fill: a buffer whose fill fails is freed with the others)
        alloc_bytes += n * sizeof(T) + 16;
        if (poison) {
            const bool skip = zero == DA_INIT0 || std::find(poison_skip.begin(), poison_skip.end(), n_dalloc) != poison_skip.end();
            CK(cudaMemset(p, skip ? 0 : 0xff, n * sizeof(T) + 16));
            CK(cudaStreamSynchronize(0)); // (before an upload on another stream writes the buffer)
            if (getenv("MK_POISON_LOG")) fprintf(stderr, "[mk-poison] #%ld %zu bytes%s\n", n_dalloc, n * sizeof(T), skip ? " zeroed" : "");
        } else if (zero) {
            CK(cudaMemset(p, 0, n * sizeof(T) + 16));
        }
        n_dalloc++;
        dalloc_s += lsec(ta);
        if (wrec_on) wrec.push_back({p, n * sizeof(T)});
        return (T *) p;
    }

    // A tensor of the file in a type the kernels read. Q3_K and IQ3_S are re-encoded as Q6_K and
    // IQ4_NL as Q8_0, exactly: every weight is a small integer times an fp16 scale in both (checked
    // bit for bit against ggml's dequantization of the source row). `to == T_F32` dequantizes
    // (ssm_alpha / ssm_beta, which are folded into fp32 rows).
    struct Src {
        uint32_t type = 0;
        std::vector<uint64_t> ne;
        const uint8_t * data = nullptr;
        uint64_t nbytes = 0;
        std::vector<uint8_t> own; // the re-encoded copy, when there is one
    };
    // a tensor's shape is [k, n] (any further dimensions 1): its bytes are exactly n rows of k (gguf::shape_is)
    static bool shape2(const std::vector<uint64_t> & ne, uint64_t k, uint64_t n) { return gguf::shape_is(ne, {k, n}); }
    int n_conv = 0;
    double conv_s = 0;
    // The MTP draft block's (blk.64 + nextn.eh_proj) matrices stored wider than Q4_K (Q5_K, Q6_K, Q8_0, Q4_0, IQ4_NL;
    // UD-Q4_K_XL: all of them, Q4_K_M: eh, v, ffn_down) are re-quantised to Q4_K at load (ggml_quantize_chunk, no
    // imatrix): the dp4a kernels read none of those types. Lossy, but only the drafts can change: they are verified by
    // the exact main model, so the output (greedy ids, the sampled distribution) is unchanged and only acceptance moves.
    // Each MTP pass streams ~0.1 GB less.
    bool req_q4 = false; // want(): re-quantise the current tensor to Q4_K (set while the MTP block loads)
    int n_req = 0;
    double req_s = 0;

    // MK_DRAFT_Q4 (default 1): the draft head's base rows (the draft vocabulary, --draft-vocab) re-quantised from lm_head's
    // Q6_K to a Q4_K copy at load (ggml_quantize_chunk, no imatrix; +94 MB of VRAM): each draft streams 0.69x the head's
    // bytes. Lossy, but only the drafts can change (they are verified by the exact main head), so the output (greedy ids,
    // the sampled distribution) is unchanged and only acceptance moves. MK_DRAFT_Q4=0: off. Policy key dq4=0|1 switches it.
    bool draft_q4 = true;
    bool draft_q4_env = false; // MK_DRAFT_Q4 given explicitly (only then is a missing draft vocabulary worth a note)
    int dq4_rows = 0; // rows of the copy (0: none)
    void set_draft_q4(int on);

    // MK_REENC3=1: Q3_K and IQ3_S re-encoded to Q6_K at load (instead of their native layouts; A/B tests)
    const bool reenc3 = getenv("MK_REENC3") && atoi(getenv("MK_REENC3")) != 0;
    int n_nat = 0; // Q3_K / IQ3_S tensors repacked to their native layouts
    double nat_s = 0;
    bool iq3s_grid_up = false;

    // The MTP block's source: the model file; a second GGUF (MKEngine's mtp argument: --mtp FILE, MK_MTP=FILE),
    // whose blk.<NL>.* tensors replace the model file's (its token_embd / output copies are not used: the MTP block shares
    // the model's embedding and lm_head); or none (a file without the MTP layer, or --no-mtp / MK_NO_MTP=1): plain
    // decoding at depth 0, no MTP weights, no MTP KV slot, no draft head.
    std::unique_ptr<gguf::File> mtpf;
    bool has_mtp = true;
    std::string mtp_desc = "model"; // (state-file key) "model", "none" or the --mtp file's identity
    int kv_slots = N_ATT_SLOTS;     // KV slots allocated (N_ATT_SLOTS - 1 without the MTP block: its slot is the last)
    static bool is_mtp_name(const std::string & n);
    const gguf::File & src_of(const std::string & name) const { return mtpf && is_mtp_name(name) ? *mtpf : file; }

    // The GGUF tensor `name`, its bytes read in (Prefetch, MK_LOAD_IO)
    std::unique_ptr<Prefetch> pfx, pfx_mtp;
    const gguf::Tensor & tget(const std::string & name);
    // MK_MTP_Q4 re-quantisation of one tensor's rows to Q4_K (rs_to bytes per row into out)
    static void requant_q4k(const gguf::Tensor & t, uint8_t * out);
    static bool requant_type(uint32_t t) {
        return t == gguf::Q5_K || t == gguf::Q6_K || t == gguf::Q8_0 || t == gguf::Q4_0 || t == gguf::IQ4_NL;
    }
    // The MTP block's re-quantisation (~2.5 s of CPU on UD-Q4_K_XL) runs on a background thread from early in the load
    // (rq_start; MK_LOAD_RQBG=0: inline); want() takes its results. Same rows, same calls, same bytes.
    std::map<std::string, std::vector<uint8_t>> rq_res;
    double rq_s = 0; // the background thread's own time
    struct Joiner {
        std::thread t;
        ~Joiner() {
            if (t.joinable()) t.join();
        }
    } rq_th;
    bool rq_bg = false;
    void rq_start();

    Src want(const std::string & name, int to = -1);

    // ---- weight upload ----
    // Every weight goes up through `upl` (loader.cuh Uploader: a worker thread, MK_LOAD_ASYNC=0 synchronous). Tensors the
    // kernels read as stored are copied straight from the GGUF mapping; repacked ones are built in one of two reused host
    // buffer sets (hset), so the next tensor is prepared while the previous one is copied. Reusing a set changes
    // no uploaded byte (the repacks write every byte they upload).
    struct HSet {
        HBuf q, m, o, om;
        uint64_t seq = 0; // last upload reading this set
    };
    HSet hs[2];
    int hs_i = 0;
    Uploader upl;
    HSet & hset();
    void put(void * d, const void * s, size_t n, HSet * S = nullptr);
    // device buffer of n bytes holding s (the weight allocations are recorded for MK_WDIGEST)
    template <class T> T * dput(const void * s, size_t n_elem, HSet * S = nullptr) {
        T * d = dalloc<T>(n_elem, false);
        put(d, s, n_elem * sizeof(T), S);
        return d;
    }

    const float * up_f32(const std::string & name, size_t expect);

    Mat up_mat(const std::string & name, int K, int N, int to = -1);
    std::vector<std::vector<uint8_t>> own_keep; // want()'s copies, until the uploads are done (freed after the load)

    // the matrices of the main layers (set while they load) of the wv types are re-laid-out on the host before the upload
    // (wv_fd while a main layer's ffn_down loads: its Q6_K / Q3_K / IQ3_S too)
    bool wv_lay = false, wv_fd = false;
    int wv_n = 0;
    double wv_s = 0;
    size_t wv_relay(Mat & m, const uint8_t * w, const uint8_t * meta, size_t meta_n, uint8_t * out, uint8_t * out_meta);
    // MK_LOAD_GPURL (default 1): the wv layout made on the device (mkl::wv_dev_launch, on the upload worker: the
    // pre-wv bytes go to a staging buffer, a kernel writes them re-laid into the matrix's buffer); 0, or a layout / size the
    // device path does not take: wv_relayout on the host. Same bytes either way.
    const bool gpurl = !(getenv("MK_LOAD_GPURL") && atoi(getenv("MK_LOAD_GPURL")) == 0);
    std::map<int, WvProbe> wv_probes;
    uint8_t * stg_d = nullptr;   // device staging (cudaMalloc, freed after the weights)
    size_t stg_cap = 0;
    uint64_t relay_seq = 0;      // the last device re-layout queued
    std::vector<void *> wv_tabs; // the probes' gather tables (cudaMalloc)
    int wv_dev_n = 0;
    void wv_place(Mat & m, const uint8_t * w, size_t wn, const uint8_t * meta, size_t meta_n, HSet & S);
    // after the weights: the staging buffer and the gather tables go back (before the KV cache is allocated)
    void wv_dev_done();

    QBuf qbuf(int U);

    // MK_FAULT=cuda:<n> | sticky:<n> (tests of the error path only): the n-th prefill (1-based) fails half-way
    // through the batched layers, with a rejected CUDA call (non-sticky: an invalid memcpy direction; the state is
    // left half-updated) or a trapping kernel (sticky: the context is lost)
    int fault_kind = 0, fault_at = 0, fault_seen = 0;
    bool fault_now = false;
    void fault_inject();

    int n_draft_max;
    int n_draft_cap; // the depth given at construction (set_n_draft: at most this)
    int kv_mode; // KV cache format: 0 fp16, 1 q8_0, -1 auto (q8_0 when fp16 does not fit)
    // A --mtp file must hold the MTP layer of this model: same architecture and shapes, its MTP block at layer index NL
    // (block_count NLA), the same vocabulary; every tensor of the block present (shapes: up_mat / up_f32 as they load)
    void check_mtp_file(const gguf::File & m, const std::string & mp) const;
    // the state-file identity of a file: real path, size, mtime, hash of its first and last 4 MiB
    static std::string file_ident(const std::string & p, const gguf::File & f);

    Impl(const std::string & path, int ctx_, int ndraft_, const std::string & draft_vocab, int kv_mode_, const std::string & mtp_);

    // one line of load phases (MK_LOADLOG=1: every phase on its own line)
    void load_report(double pf_wait, size_t pf_bytes);

    // MK_WDIGEST=1 (load gates): a 64-bit hash over every buffer allocated while the weights load (dalloc, in allocation
    // order: size + device bytes) and the mapped token_embd table. Slow (reads ~17 GB back); test runs only.
    bool wrec_on = false;
    std::vector<std::pair<const void *, size_t>> wrec;
    void weights_digest();

#if MK_SPEC_LOG
    // one JSON line per decode cycle: start position n, T, accepted a, the verified drafts d, the verify
    // rows' targets tg, per draft its MTP [lse, best logit, (logit, id) x TOPK_MAX], and when sampling the
    // main rows' candidates p [(logit, id) x TOPK_MAX]
    void slog_cycle(int n0, int T, const int * drafts, const Ctl & c);
    // after a prefill: the whole sequence (its last id is the first output token) and where the fed part began
    void slog_prompt(int n0);
#endif

    ~Impl();
    // the device and pinned resources: ~Impl, and a constructor that throws part way (~Impl does not run then). Every
    // handle starts null or empty, so a half-built engine frees what it got; no CUDA call when nothing was made.
    void release() noexcept;

    // The default pool keeps what is freed into it, and what a failed allocation grew it by before failing: back to the
    // device (after the stream-ordered frees). No CUDA call without the pool.
    void pool_trim() noexcept;

    // the most positions a sequence may have: the model's context length (2^20 without the key)
    int ctx_limit() const;

    // emb_host holding at least `floats` (grown: pinned, mapped; cP.emb follows it)
    void emb_reserve(size_t floats);

    void launch(int mode, int T, int last, int score = 0);

    // MK_PROFILE: one decode launch's barrier stamps (T: its rows; 0 = a batched launch)
    void prof_collect(int T);

    void print_profile();

    // grid barrier cost. MK_GSB=N: launches of N barriers with no work (and of 0 barriers: the launch cost) for
    // every gsync form (gs_arrive_wait) at nsm and nsm / 2 blocks, round-robin over the configurations so clock drift hits
    // them alike; per barrier = (median t(N) - median t(0)) / N. MK_GSB_REPS (default 15) rounds; 1.5 s GPU warm-up first.
    void gsync_bench_host(int N);

    void bench();

    // ---------------- batched prefill (prefill.cuh) ----------------
    PBufs hp{};
    bool pf_ready = false;
    // the host enqueues prefill chunk k + 1 before it waits for chunk k (pf_cev); MK_PF_SYNC=1: the token upload of every
    // chunk is synchronous and the host waits for each chunk's trunk before its MTP part (the GPU idles for the host's wake-up
    // and launches at both points). MK_PF_GAPS=1: report the GPU's idle time at those points per prefill run.
    bool pf_sync = false, pf_gaps = false;
    // MKEngine::cancel: pf_run reads pf_cancel (set from any thread) at each chunk boundary and stops there (pf_cancelled)
    std::atomic<int> pf_cancel{0};
    bool pf_cancelled = false;
    int pf_stop = 0; // (the position it stopped at)
    // MKEngine::prefill_progress (read from any thread): the batched part of the running prefill() covers positions
    // pf_from .. pf_to - 1; pf_run moves pf_pos past each chunk once the GPU finished it. Host counters only.
    std::atomic<int> pf_from{0}, pf_to{0}, pf_pos{0};
    cudaEvent_t pf_cev[2] = {};
    std::vector<cudaEvent_t> pf_gev; // MK_PF_GAPS: timing events, 4 per chunk (start, trunk end, MTP part start, end)

    void pf_init();
    int pf_cap_req() const;                   // tokens per prefill chunk as requested (MK_PF_CAP), before pf_init's ring clamp
    int pf_cap_eff(bool ring_ck, int kvrc) const; // after it: what pf_init allocates for (and the fit check counts)
    void pf_kernel_attrs();                   // the prefill GEMMs' attributes (pf_init; MK_RESERVE_EARLY: before the KV cache too)
    size_t pf_scratch_bytes(int cap, bool ex); // the device bytes pf_init's scratch takes (MK_RESERVE_EARLY's fit check)

    // causal attention of the chunk's T queries (positions n ..) over the cache of attention slot `slot` -> y2
    int a2_slots = 0;
    size_t y1_floats = 0; // hp.y1 in floats (pf_init: cap x FF, or cap x VD in MK_EXACT mode with every gate / up fused)
    size_t y0_floats = 0; // hp.y0 in floats (pf_init: cap x FF, or cap x QG with every gate / up fused)
    // the slot's main buffer holds K q6 / V q5 (the D / D-prime presets) or K q5 / V q5 (preset F, K rotated or not):
    // pk_attn8's compile-time-format instantiations
    bool kv_k6v5(int slot) const { return hb.kvb[0][slot] == 6 && hb.kvb[1][slot] == 5; }
    bool kv_k5v5(int slot) const { return hb.kvb[0][slot] == 5 && hb.kvb[1][slot] == 5; }
    bool attn_old = false, attn_split = true, attn_pv16 = true;
    int attn_s8 = 1;
    // MK_EXACT: key tiles per absolute attention segment (pk_attn8 EXA). 4096 positions (long-context prefills, fresh or
    // after a restore, are faster than with 8192 or 16384)
    int ex_segt = 128;
    // EXA partials, two buffers: y1 (k, v of the layer, dead after pk_attn_prep) as for a2_plan, then xh (the attention
    // input, dead after the q / k / v GEMMs until pk_attn_out). A slot with a KV window ring runs its chunk in pieces
    // (pf_run), and a piece's partials in y1 would overwrite the k / v of the next pieces' rows: there xh2 (the ffn_down
    // input, idle until the FFN, half y1's size), then xh.
    std::pair<float *, float *> ex_part(bool pieces) {
        return {pieces ? reinterpret_cast<float *>(hp.xh2) : hp.y1, reinterpret_cast<float *>(hp.xh)};
    }
    std::pair<size_t, size_t> ex_cap(bool pieces) const { // (partial slots)
        return {(pieces ? (size_t) hp.cap * FF / 2 : y1_floats) / A2_PART, (size_t) hp.cap * D / A2_PART};
    }
    // MK_EXACT work split (pk_attn8 EXA): numerically free (every split gives the same rows), so this only balances.
    // The last N - nfull items (blockIdx order) are split into a head unit (segments 0 .. h - 1, folded in registers ->
    // partial slot 0) and single-segment units (segment h + u - 1 -> slot u), launched interleaved (every head, then
    // every item's segment h, ..) so that the CTA scheduler fills the tail evenly; S = 1 + (most segments of a split item)
    // - h units and partial slots per split item (ex_part). The plan minimises a list-scheduling estimate (a unit
    // takes its key tiles + a per-CTA overhead; the CTAs start in blockIdx order on a2_slots resident slots) over
    // nsplit in {the last partial wave, + one full wave} x h; memoised per (T, n0, capacity). The first nA split items'
    // slots go to the first partial buffer, the others to the second. MK_EXACT_SPLIT=0: whole items only.
    struct ExPlan { int nfull, S, h, nA; };
    std::map<std::tuple<int, int, size_t, size_t>, ExPlan> ex_memo;
    bool ex_split = true;
    ExPlan ex_plan(int T, int n0, std::pair<size_t, size_t> cap);
    // (r0: the queries are chunk rows r0 .. r0 + T - 1 at positions n ..: MK_EXACT with KV window rings, see pf_run)
    void attn(int T, int n, int slot, int r0 = 0);

    // GEMM token tile (pf::mmh NB): every NB gives bit-identical results, so this is a pure speed choice.
    // At 128 a short prompt pads to 128 columns and the CTA is tensor-pipe bound on the padding (42 tokens: 2.1 us
    // per 64-K stage, 86 of 128 columns wasted); 32 / 64 cut the MMA work and add CTAs.
    // The choice is a small cost model over waves of CTAs (nrt row tiles x token tiles) on the nsm SMs, with the
    // per-stage time of a lone CTA at NB = 32 / 64 / 128 (0.80 / 0.95 / 1.44, relative; the model's Q4_K .. Q8_0
    // shapes, microbenchmark). NB 64 / 128 run one CTA per SM (shared memory), NB 32 two (a
    // pair takes 1.77x as long as one CTA). It picks 32 for <= 32 tokens; up to 64 tokens 64, or 32 when the GEMM is
    // narrow (40 row tiles x 2 token tiles still fit one wave); up to 128 tokens 128, or 64 when narrow; above that
    // mostly 128. Chunks of more than 384 tokens always use 128.
    int pf_bn_force = 0; // MK_PF_BN
    int pf_nb(int ntok, int nrt) const;
    template <int TYPE, bool GLU, int LAY = 0> void mmh_attr() { mkl::pf_mmh_attr<TYPE, GLU, LAY>(); }
    // nrt row tiles (CTAs along N) x token tiles of pf_nb tokens; LAY 1: the matrix is in the wv layout
    template <int TYPE, bool GLU, int LAY = 0>
    void mmh_launch(int nrt, const pf::WMat & W, const __half * X, int ntok, const pf::Out & O, const uint8_t * W2 = nullptr,
                    cudaStream_t st = 0) {
        mkl::pf_mmh<TYPE, GLU, LAY>(pf_nb(ntok, nrt), nrt, &W, X, ntok, &O, W2, st);
    }

    // pf::mmh's shapes: K in pairs of BK-wide stages (the pipelined loop steps two stages; fp16 sums span two), N in whole
    // tiles of `rows` (BM; the fused gate / up: BM / 2 of each). Another shape would skip rows or columns silently.
    static void pf_shape_check(const Mat & m, int rows);
    static_assert(D % (2 * pf::BK) == 0 && 2 * D % (2 * pf::BK) == 0 && FF % (2 * pf::BK) == 0 && VD % (2 * pf::BK) == 0, "prefill GEMM: K");
    static_assert(D % pf::BM == 0 && FF % pf::BM == 0 && VD % pf::BM == 0 && QG % pf::BM == 0 && KVD % pf::BM == 0 && QKV % pf::BM == 0 &&
                      VOCAB % pf::BM == 0,
                  "prefill GEMM: N");
#if MK_PF_INFCHK
    // MK_PF_INFCHK builds: the (layer, site) the counters charge from the next launch on (stream-ordered; < 0: unchanged)
    void chk_set(int layer, int site);
    void chk_mat(const Mat & m);
    // prints the counts of the prefill that just finished (device synchronised) and clears them
    void chk_report(int npos);
#endif
    // y[t][row] (+)= W x_t for the first ntok fp16 rows of xh (token offset t0), on stream st
    void gemm(const Mat & m, int ntok, float * y, int ldy, int mode, int t0 = 0, const __half * xin = nullptr, cudaStream_t st = 0);
    // pf_fork (MK_PFX bit 1 clear): a layer's independent GEMMs run side by side, so their CTAs share one partial last wave
    // instead of one each (DN: gate on pf_s2 next to qkv; attention: k on pf_s2 and v on pf_s3 next to q; joined by
    // pf_e1 / pf_e2 before their outputs are read). Launch placement only: the same kernels on the same data.
    int pfx_bits = 0;
    bool pf_fork = false;
    cudaStream_t pf_s3 = nullptr;
    cudaEvent_t pf_e2 = nullptr;

    // xh2[t] = silu(Wg x_t) * (Wu x_t) for the first ntok rows of xh: one fused GEMM when gate and
    // up share a type with in-row scales, else two GEMMs into y0 / y1 and pk_glu
    static bool glu_fused(const Mat & g, const Mat & u);
    void gemm_glu(const Mat & g, const Mat & u, int ntok);
    // V2: a gate / up pair of different types (or layouts) as one mixed-type fused launch, pf::mmh<TG, true, NB, LG, TU, LU>
    // (warps 0-3 gate rows, 4-7 up rows, each half its own type): the same products, sums and silu * up as the two GEMMs +
    // pk_glu, so bit-identical, without the fp32 y0 / y1 round trip (y1 can then shrink, pf_init). MK_PF_GLU_MIX=0: off.
    // The pairs of UD-Q4_K_XL (gate / up; main layers are in the wv layout except Q3_K); any other pair takes the two-GEMM
    // path.
    bool glu_mix_on = !(getenv("MK_PF_GLU_MIX") && atoi(getenv("MK_PF_GLU_MIX")) == 0);
    template <int TG, int LG, int TU, int LU> void mmh_attr_mix() { mkl::pf_mmh_mix_attr<TG, LG, TU, LU>(); }
    template <int TG, int LG, int TU, int LU> void mmh_launch_mix(int nrt, const pf::WMat & Wg, const pf::WMat & Wu, int ntok, const pf::Out & O) {
        mkl::pf_mmh_mix<TG, LG, TU, LU>(pf_nb(ntok, nrt), nrt, &Wg, &Wu, hp.xh, ntok, &O);
    }
    void glu_mix_attr();
    // ntok > 0: launch it if the pair has an instantiation; ntok = 0: only whether it has one
    bool glu_mix(const Mat & g, const Mat & u, int ntok);

    // MK_PF_PROFILE=1: per-category GPU time of the batched prefill (events around each launch)
    struct PfEv { const char * cat; cudaEvent_t a, b; };
    std::vector<PfEv> pf_ev;
    bool pf_prof = getenv("MK_PF_PROFILE") != nullptr;
    // K8 (MK_K8 bits, read at init): 1 pk_ab on pf_s2 (after pf_e0 = the layer's pk_norm_quant; pk_dn_prep waits for
    // pf_e1 = pk_ab), 2 pk_norm_quant1, 4 pk_dn_prep_t
    int k8 = MK_K8_DEF;
    cudaStream_t pf_s2 = nullptr;
    cudaEvent_t pf_e0 = nullptr, pf_e1 = nullptr;
    void pf_norm_quant(const float * src, const float * w, int T);
    void pf_dn_prep(int T, int n, int dl, const LayerW & W);
    int k8_nt = MK_K8_NT; // (MK_K8_NT env: 4 or 8)
    template <class F> void pf_time(const char * cat, F f) {
        if (!pf_prof) { f(); return; }
        // MK_PF_PROFILE=2: idempotent kernels are timed alone (5 repeats, min)
        static const char * idem[] = {"attn_prep", "attn_out", "dn_out", "norm_quant", "dn_ab", "attn", "embed", "gemm_ffn_gu", "gemm_ffn_gum", "gemm_ffn_gu2", "gemm_dn_in", "gemm_attn_q", "gemm_attn_kv"};
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
    void pf_report();

    // Runs positions n0 .. n0 + toks.size() - 1 through the trunk (and the MTP block's K/V).
    // score: instead of MTP, greedy argmax of every position -> pred, and / or the logits of the
    // positions >= first -> sink.
    // snaps: checkpoints to save on the way (positions n0 < P <= n0 + toks.size(), ascending, slots from ck_assign;
    // at most stg_n per chunk: ck_positions); complete when pf_run returns.
    // erow (images, aligned with toks): each token's row in emb_host, -1 for a token_embd row; chunks with an image row
    // run pk_embed_img / pk_mtp_in<true>, the others the text kernels.
    void pf_run(const std::vector<int> & toks, int n0, std::vector<int> * pred, const MKEngine::LogitSink * sink = nullptr,
                int first = 0, const std::vector<Snap> * snaps = nullptr, const int * erow = nullptr);

    // Checkpoint positions for a batched part n0 .. hend - 1 (pf_run chunks of hp.cap from n0): the requested P with
    // n0 < P <= hend, `exact` as it is (saved by a side scan), every other one moved down onto the DeltaNet scan's
    // 32-token grid from its chunk's start; ascending, without duplicates or existing checkpoints, at most stg_n per
    // chunk (one less with `exact`, whose side scan may need a staging slot; dropping the earliest).
    // pins: pin requests (pin_req) to save too, never dropped; *vpin marks their positions in the result.
    std::vector<int> ck_positions(const std::vector<int> & cks, int exact, int n0, int hend, const std::vector<int> & pins = {},
                                  std::vector<char> * vpin = nullptr);

    // cB.logits ([MAXT][VOCAB] fp32, 5 MB): only scoring launches (Launch::score == 2) write it
    void logits_alloc();

    // presence / frequency penalty (Launch::pres / freq): the response's token counts (cB.pcnt), zeroed by the next
    // prefill once a penalised launch has counted into them
    float pen_pres = 0.f, pen_freq = 0.f;
    bool pcnt_dirty = false;
    // MK_KV_EMU_SPEC / set_kv_emu (MK_KV_EMU builds): comma-separated <K|V|*><slot>=<bits>[flags] and sink=N. slot
    // 0..15 (attention layer 4 slot + 3), 16 (MTP) or * / empty (all). Flags (kv_store_emu): r Hadamard rotation,
    // s random signs + rotation, a q4_0-style scale, l Lloyd-Max codebook with rms scale (bits 2..6), n one scale per
    // 256-value row (else per 32). "8" without flags = plain q8. sink=N: positions < N stay q8.
    void kv_emu_parse(const std::string & s);
    void set_kv_emu(const std::string & spec);
    int demoted = 0; // recent-window emulation: positions < demoted have been demoted to their slot's mode
    void kv_demote_to(int upto);
    void set_penalty(float pres, float freq);
    void pcnt_clear();

    // ---- post-prefill state files (MKEngine::prefill_state) ----
    // A file holds what a batched prefill of positions 0 .. hend - 1 leaves for the megakernel launch of the prompt's last
    // token: every attention slot's K / V rows < hend (values + scales, or fp16), the MK_KV_WIN rings, the DeltaNet states,
    // conv rings and the h ring; nothing else of the sequence state is read before it is rewritten (the restore runs no
    // checkpoint, and prefill() then sets ctl / seq / the draft vocabulary as after the batched part). Named by a hash of
    // the key text (stored in the file and compared on load): the binary's bytes, the model file, ctx, the KV format, the
    // prefill chunk cap and every MK_* variable not known to be decode-only or diagnostic.
    std::string st_dir, model_path;
    int st_how = 0;
    double st_s = 0;
    static unsigned long long file_hash(const std::string & path);
    // the object file holding this code (the CLI binary, or libqwen38mk.so under the server), from /proc/self/maps
    static unsigned long long self_hash();
    static std::string hex64(unsigned long long v);
    void st_name(const std::vector<int> & head, std::string & path, std::string & key);
    // the device regions of the state after a batched prefill of 0 .. hend - 1 (pointer, bytes), in file order
    std::vector<std::pair<char *, size_t>> st_regions(int hend);
    // file: magic "MKSTATE1" | u64 key bytes | key | u64 hend | u64 regions | u64 bytes of each region | the regions' bytes
    void st_save(const std::string & path, const std::string & key, int hend);
    bool st_load(const std::string & path, const std::string & key, int hend);
    // ---- conversation state in host memory (MKEngine::side_begin / park) ----
    // Host store: pinned host chunks of hs_chunk bytes (MK_STORE_CHUNK_MIB, default 32), allocated on first use and kept in
    // a free list (cudaHostAlloc runs at ~1 GB/s under WSL2, a first touch of pageable memory about as slow: a chunk is
    // paid for once), at most hs_budget bytes (MKEngine::store_budget; 0: no store). A Blob lays items (bytes of device or
    // host memory) end to end over its chunks; blob_get copies each item back to where it came from.
    size_t hs_chunk = (size_t) (getenv("MK_STORE_CHUNK_MIB") ? std::max(1, atoi(getenv("MK_STORE_CHUNK_MIB"))) : 32) << 20;
    size_t hs_budget = 0, hs_alloc = 0;
    std::vector<char *> hs_free;
    double hs_ms = 0;      // the last side / park operation: wall time
    size_t hs_bytes = 0;   // ... bytes it copied
    struct Blob {
        struct Item {
            char * at; // device memory (async copies on stream 0), or host memory (memcpy)
            size_t n;
            bool host;
        };
        std::vector<char *> chunks;
        std::vector<Item> items;
        size_t used = 0;
    };
    char * hs_take();
    char * hs_new();
    void hs_release(Blob & b);
    void hs_trim();
    // appends n bytes at src (device memory: a device-to-host copy on stream 0, which the caller waits for; host memory:
    // memcpy now) to b; false: the store is full (b then holds a partial item: the caller releases it)
    bool blob_put(Blob & b, const void * src, size_t n, bool host);
    // copies every item of b back (device items first, as host-to-device copies on stream 0 that the caller waits for, so
    // that the host items' memcpy runs meanwhile)
    void blob_get(const Blob & b);
    // the sequence's KV rows a .. b - 1 of every slot (values + scales, or fp16), K and V, each kv head
    bool kv_put(Blob & bl, int a, int b);
    // ring slots a .. b - 1 (of kvrc) of every MK_KV_WIN ring side, each kv head
    bool ring_put(Blob & bl, int a, int b);
    // the recurrent state: DeltaNet states, conv rings, h ring (what a checkpoint holds; not kvgb, dq or pcnt: the next
    // prefill refeeds every position from dn_c on, stale_replay)
    bool live_put(Blob & bl);
    size_t kv_token_bytes() const;
    size_t ring_bytes() const;
    // the host side of a sequence: everything a continuation reads besides the device state above
    struct Book {
        std::vector<int> seq, dv;
        Ctl ctl{};
        std::vector<Span> spans;
        int kv_top = 0, nd_avail = 0, ndraft = 0, last_acc = 0, demoted = 0, dv_n = 0;
        bool all_acc = true;
    };
    Book book_get() const;
    void book_set(const Book & k);
    // the draft vocabulary as `ids` (a dv_ids saved by book_get: the base list, then the sequence's additions in order)
    void dv_set(const std::vector<int> & ids);
    // no copy of a checkpoint save or restore in flight, nothing on the device still reading or writing the state
    void hs_quiesce();
    // Side request (MKEngine::side_begin): the kept sequence's rows main_top = sw_top and its window ring; sw_hi rows and
    // sw_rhi ring slots are in sw_blob. Every launch / prefill that writes rows < X calls sw_need(X) first (rings_all:
    // every ring slot, for side_unpark). The checkpoint pool serves the side request: the kept sequence's checkpoints are
    // hidden (sw_ck: position, ctl, rings flag), and a slot holding one has its block (and rings) copied into sw_blob
    // before the side request writes it (sw_ck_take); side_end copies them back with the rows.
    static constexpr int SW_BLK = 1024; // rows per copy-out step
    bool sw_on = false, sw_broken = false;
    int sw_top = 0, sw_hi = 0, sw_rhi = 0;
    Blob sw_blob;
    Book sw_book;
    struct SwCk {
        int pos;
        Ctl ctl;
        bool ring_ok, saved;
        bool pin;
        unsigned pin_t;
    };
    std::vector<SwCk> sw_ck;
    double sw_ms_out = 0;
    void sw_need(int X, bool rings_all = false);
    // slot i holds a checkpoint of the kept sequence that is not in sw_blob yet
    bool sw_ck_held(int i) const { return sw_on && !sw_broken && i < (int) sw_ck.size() && sw_ck[i].pos >= 0 && !sw_ck[i].saved; }
    // before the side request writes slot i: the kept checkpoint there goes to sw_blob (false: no room, nothing changed)
    bool sw_ck_take(int i);
    // bytes the store can still hand out (free chunks, then new ones within the budget)
    size_t hs_room() const { return hs_free.size() * hs_chunk + (hs_budget > hs_alloc ? (hs_budget - hs_alloc) / hs_chunk * hs_chunk : 0); }
    // Parked conversations (MKEngine::park): a snapshot of everything above, plus up to max_cks of the newest checkpoints
    // and the pinned ones
    struct Parked {
        Blob blob;
        Book book;
        struct Ck {
            int slot, pos;
            Ctl ctl;
            bool ring_ok, pin;
            unsigned pin_t;
        };
        std::vector<Ck> cks;
    };
    std::map<int, Parked> parked;
    int park_next = 1;
    size_t hs_used() const;
    int seq_top() const { return std::max({kv_top, (int) seq.size(), hctl.dn_c}); }
    std::vector<int> park_cks(int max_cks) const;
    size_t ck_block_bytes() const { return (CK_NS + CK_NC + CK_NH) * 4; }
    double hs_park_ms = 0;    // the last side_end_park: its park of the side request's conversation
    size_t hs_park_bytes = 0;
    // park(): a snapshot of the sequence (device copies quiesced by the caller) as a new entry: its id, -1 no room
    int park_entry(int max_cks);
    // side_end: the kept sequence comes back (rows, rings, recurrent state, its checkpoints' blocks, the bookkeeping)
    void side_restore();

    // ---- lanes (MKEngine::slots_*): sequences of their own, decoded together by batched launches ----
    // A lane holds a whole conversation: device state of its own (KV cache, recurrent state, rings, ctl), its host
    // bookkeeping, a small checkpoint pool. While a lane is live (lane_in) its state sits in the Impl's own members and
    // cView selects its full view, so every single-sequence path (prefill, rewind, truncate, a solo decode step) runs on it
    // unchanged; lane_out swaps it back. Batched launches (lanes_step) read every lane through its views at once. The
    // row buffers (x, qn, ..), the barrier words and the lm_head scratch are the engine's own, shared by all lanes.
    struct Lane {
        int ctx = 0;
        Bufs hb{};
        std::vector<int> seq;
        Ctl hctl{};
        int kv_top = 0, nd_avail = 0, ndraft = 0, last_acc = 0, demoted = 0, dv_n = 0;
        bool all_acc = true;
        bool stale_replay = false, stale_pcnt = false; // (never set for a lane: swapped so the own sequence keeps its)
        float smp_temp = 0.f, smp_top_p = 1.f, smp_min_p = 0.f;
        int smp_top_k = TOPK_MAX;
        unsigned long long smp_seed = 0;
        float pen_pres = 0.f, pen_freq = 0.f; // (always 0: a lane has no penalty counts, hb.pcnt null)
        bool pcnt_dirty = false;
        std::vector<Span> spans, rope_spans;
        volatile Ctl * ctl_eg_host = nullptr;
        std::vector<CkSlot> ck;
        std::vector<CkChunk> ck_plan;
        std::vector<RingSide> rsides[N_ATT_SLOTS]; // MK_EXACT + MK_KV_WIN: its window rings as its checkpoints copy them
        std::vector<int> prompt; // slot_begin's prompt (a cancelled chunk queues the rest again from it)
        std::vector<int> pend;   // prompt tokens slot_begin queued, not prefilled yet
        int first = -1;          // the first token of the finished prefill: emitted by the next slots_step
        bool dead = false;       // a slot call on it failed (MKCudaError): state undefined until slot_begin resets it
        std::string dead_why;    // (that error, in the refusals)
    };
    std::vector<Lane> lanes;
    int live = -1;          // the lane whose state the Impl's members hold (-1: the engine's own sequence)
    int live_view = 0;      // cView: view_full(live), or 0
    bool dv_frozen = false; // lanes open: the draft vocabulary stays the base list (shared by every sequence)
    bool par_wide = false;  // batched passes up to RMAX rows (MK_PAR_WIDE build, the model's matrices and the GPU allow them)
    // slot_prefill's chunks before the prompt's last: every row in the batched prefill (no megakernel row, no output
    // token, no checkpoint at the chunk's end), so in MK_EXACT mode the rows are those of the whole prompt's prefill
    bool pf_mid = false;
    void lane_swap(Lane & l);
    void set_cview(int v);
    void lane_in(int k);
    void lane_out();
    // ~LaneOut's lane_out: never throws. A failed device call marks the lane dead and is retried once; failing again it
    // is taken as sticky (as in mk_cuda_fail: exit code 4), since unsynchronised or on a stale view the next sequence's
    // work would run on the wrong state
    void lane_out_nothrow() noexcept;
    // a slot call failed on lane k (MKCudaError, also the NaN check's): nothing runs on it until slot_begin resets it
    void lane_dead(int k, const std::string & why);
    // lane k for a call that continues its prompt or reply: refused when it does not exist or is dead
    Lane & lane_ok(int k);
    // the lanes of one step: each lane_ok and listed once (two rows of a batched launch would update one lane's KV cache);
    // so at most lanes.size() <= NSEQ of them, what Launch::ln holds
    void slots_ok(const std::vector<int> & slots);
    // a slot call's body f on lanes ks, then the engine's own sequence back in; an MKCudaError marks every lane in ks
    // dead on its way to the caller (MKCancelled does not: the prefill put the state back itself)
    template <class F> void lanes_call(const std::vector<int> & ks, F f) {
        try {
            f();
            lane_out(); // (here rather than in ~LaneOut: its failure is this call's error)
        } catch (const MKCudaError & e) {
            for (int k : ks) lane_dead(k, e.what());
            throw;
        }
    }
    // frees ck_new_slots' slots (events, pinned blocks, ring copies; nothing may still use them)
    void ck_slots_free(std::vector<CkSlot> & v) noexcept;
    // slots_open failing part way (allocs had na entries, alloc_bytes ab): its lanes and every device / pinned
    // allocation it made go, lanes stay empty, the engine's own sequence is live (never throws: results ignored, the last
    // error cleared)
    void slots_undo(size_t na, size_t ab) noexcept;
    // n pinned checkpoint slots (as ck_alloc's: the streams and events are the engine's)
    std::vector<CkSlot> ck_new_slots(int n);
    // device bytes of one lane's KV cache of lctx positions in the engine's format (MK_KV bits per slot and side, fp16 scales,
    // the MK_KV_WIN q8 rings of the sides that have one)
    size_t lane_kv_bytes(int lctx) const;
    // device bytes of one lane of ctx positions (KV cache; state, rings, partials, ctl)
    size_t lane_bytes(int lctx) const;
    // MKEngine::slots_open: n lanes of lctx positions each (0: the most that fit, leaving MK_SLOT_RESERVE_MIB, default
    // 1536, for the prefill and the runtime); returns lctx
    int slots_open(int n, int lctx);
    // the main view of lane k for a batched launch: its rows start at row r0 of the pass
    Bufs lane_main_view(const Lane & l, int r0) const;
    static Bufs rows_view(const Bufs & b, int r0); // b with its row buffers from row r0
    std::string wide_why() const; // why the RMAX-row weight phases cannot run here ("": they can)
    std::string deep_why() const; // why the deep pass (n_draft > MAXT - 1) cannot run here ("": it can)
    float * apart_hi = nullptr;   // the deep pass's attention partials of rows MAXT .. (view_hi)
    float2 * aml_hi = nullptr;
    void set_view_hi();
    // one batched decode launch of lanes ks (>= 2, none live, every one past its prefill): returns each lane's tokens
    // (empty: its context is full). Rows: lane i verifies T_i <= Tcap = MAXT / nl tokens (its drafts beyond are dropped,
    // as att_cap_T drops them) and drafts up to Tcap - 1 for the next launch.
    std::vector<std::vector<int>> lanes_step(const std::vector<int> & ks);
    // one solo decode step of the live sequence (MKEngine::step's body): appends the emitted tokens to out
    int step_live(std::vector<int> & out);

    void reset();
};

// token ids outside the vocabulary would index host arrays and embedding rows: refused before anything changes
inline void check_ids(const std::vector<int> & toks, size_t pos0) {
    for (size_t i = 0; i < toks.size(); i++)
        if (toks[i] < 0 || toks[i] >= VOCAB)
            throw std::runtime_error("token id " + std::to_string(toks[i]) + " at position " + std::to_string(pos0 + i) +
                                     " out of range (vocabulary " + std::to_string(VOCAB) + ")");
}
