// OpenJet's Inkling 2 Mini engine: decode with every weight in VRAM (24 GB cards).
//
// Loads a mainline-layout (inkling2_loop) GGUF, repacks the quantized matrices into the kernels' layout, and records
// one decode step (48 layer applications: 6 prelude, 12 core layers x 3 passes, 6 tail) as a CUDA graph that is
// replayed for every token; the position lives in device memory. The op order is reference/inkling_ref.py's, and
// tests/check_engine.py compares the logits with it.
//
// Offload mode (ik_new2 with a VRAM budget for experts): only part of the experts live in VRAM, in cache slots; a CPU
// worker pool computes the others straight from the mmap'd GGUF (the page cache is the RAM tier, the SSD sits behind
// it). Per layer application the graph publishes the router's choice to mapped host memory, the GPU computes its
// cached experts while the CPU computes the rest, and a wait kernel adds the CPU's result. Between tokens a cache
// manager swaps experts by decayed read counts.
//
// C API: ik_new, ik_new2, ik_reset, ik_decode, ik_n_vocab, ik_free, ik_last_error, ik_stats.

#include <cstdint>

extern "C" {
void ik_cpu_gate_up(int type, const uint8_t * w, float scale, const float * x, float * h, int n_embd, int n_ff, float beta, float lb,
                    int r0, int r1);
void ik_cpu_down(int type, const uint8_t * w, float c, const float * h, float * y, int n_embd, int n_ff, int r0, int r1);
void ik_cpu_gate_up_batch(int type, const uint8_t * w, float scale, const float * const * xs, float * const * hs, int n_tok,
                          int n_embd, int n_ff, float beta, float lb, int r0, int r1);
void ik_cpu_down_batch(int type, const uint8_t * w, const float * cs, const float * const * hs, float * const * ys, int n_tok,
                       int n_embd, int n_ff, int r0, int r1);
}

#include "gguf.h"
#include "ik_dspark.cuh"
#include "ik_kernels.cuh"
#include "ik_prefill.cuh"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <cmath>
#include <cstring>
#include <functional>
#include <immintrin.h>
#include <memory>
#include <numeric>
#include <thread>
#ifdef _WIN32
#include "compat_win32.h"
#else
#include <fcntl.h>
#include <unistd.h>
#endif
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::string g_err;

void ck(cudaError_t e, const char * what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}
void ck(int e, const char * what) { ck((cudaError_t) e, what); }

// Scatter hidden state src [T][E] into 4-tap buffer dst [T][4][E] at tap index tap (0..3)
__global__ void scatter_tap_kernel(const float * __restrict__ src, float * __restrict__ dst, int tap, int T, int E) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int t = blockIdx.y;
    if (t < T && i < E) dst[(size_t(t) * 4 + tap) * E + i] = src[size_t(t) * E + i];
}

// Single-launch conv-history rollback across all 48 slots (replaces 192 eager cudaMemcpyAsync calls)
struct HistPtrs { float * hk, * hv, * ho, * hf; };
__global__ void rollback_snaps_kernel(const HistPtrs * __restrict__ dsts, const float * __restrict__ snaps,
                                      size_t snap_stride, int vf_T, int acc_idx, int kv, int E) {
    const int s = blockIdx.y;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const float * sb = snaps + size_t(s) * snap_stride;
    const int nk = 3 * kv, ne = 3 * E;
    if (i < nk) {
        dsts[s].hk[i] = sb[size_t(acc_idx) * nk + i];
        dsts[s].hv[i] = sb[size_t(vf_T) * nk + size_t(acc_idx) * nk + i];
    }
    if (i < ne) {
        dsts[s].ho[i] = sb[size_t(vf_T) * 2 * nk + size_t(acc_idx) * ne + i];
        dsts[s].hf[i] = sb[size_t(vf_T) * (2 * nk + ne) + size_t(acc_idx) * ne + i];
    }
}

// a dense matrix: Q8_0 repacked into the fast kernels' layout, any other type in its GGUF blocks
struct Q8 {
    void * buf = nullptr;
    int rows = 0, n = 0;
    int gtype = gguf::Q8_0;
    bool repacked = true;
};

struct Experts {
    int type = 0;                 // fast path: 0 NVFP4, 1 Q8_0 (gate/up and down both, repacked)
    bool fast = true;             // false: other types, kept in GGUF blocks (the gguf_kernels.cu path)
    int gu_gtype = 0, dn_gtype = 0;  // ggml types of the two matrices
    int pool = -1;                // offload: the slot pool of this layer's expert format
    void ** gu = nullptr;         // device array of 128 device pointers (null: not in VRAM)
    void ** dn = nullptr;
    float * gu_scale = nullptr;
    float * dn_scale = nullptr;
    // offload mode
    std::vector<const uint8_t *> gu_raw, dn_raw;  // GGUF blocks of each expert (mmap)
    size_t gu_bytes = 0, dn_bytes = 0;            // GGUF bytes per expert
    std::vector<float> h_gu_scale, h_dn_scale;
    std::vector<int> slot;                        // VRAM slot of each expert, -1 if none
    // zero-copy share: device pointers to the pinned GGUF blocks of CPU-side experts (null: not pinned)
    void ** zc_gu = nullptr;
    void ** zc_dn = nullptr;
    std::vector<char> zc_ok;
};

// a spin-waiting worker pool: parallel_for runs fn(i) for i in [0, n) on the workers and the calling thread
class SpinPool {
public:
    explicit SpinPool(int n_workers) {
        for (int t = 0; t < n_workers; ++t) threads_.emplace_back([this] { work(); });
    }
    ~SpinPool() {
        stop_ = true;
        gen_.fetch_add(1);
        for (auto & t : threads_) t.join();
    }
    void parallel_for(int n, const std::function<void(int)> & fn) {
        fn_ = &fn;
        n_ = n;
        next_.store(0);
        busy_.store((int) threads_.size());
        gen_.fetch_add(1, std::memory_order_release);
        run();
        while (busy_.load(std::memory_order_acquire) > 0) _mm_pause();
    }
private:
    std::vector<std::thread> threads_;
    std::atomic<int> gen_{0}, next_{0}, busy_{0};
    std::atomic<bool> stop_{false};
    const std::function<void(int)> * fn_ = nullptr;
    int n_ = 0;
    void run() {
        for (int i; (i = next_.fetch_add(1)) < n_;) (*fn_)(i);
    }
    void work() {
        int seen = 0;
        for (;;) {
            int spins = 0;
            while (gen_.load(std::memory_order_acquire) == seen) {
                if (++spins < 20000) _mm_pause();
                else std::this_thread::yield();
            }
            seen = gen_.load();
            if (stop_) return;
            run();
            busy_.fetch_sub(1, std::memory_order_release);
        }
    }
};

__global__ void set_pos_kernel(int * d_pos, int v) { *d_pos = v; }

struct Layer {
    bool swa = false;
    Q8 q, k, v, gate, rel, out;
    float * attn_norm = nullptr, * q_norm = nullptr, * k_norm = nullptr, * v_norm = nullptr;
    float * res_query = nullptr;  // [1 or 3][E], null for layer 0
    float * ffn_res_query = nullptr;
    float * q_clip = nullptr, * rel_clip = nullptr, * rel_bias = nullptr, * sink = nullptr, * kcp = nullptr, * vcp = nullptr;
    float * k_conv = nullptr, * v_conv = nullptr, * post_conv = nullptr, * ffn_post_conv = nullptr;
    float * ffn_norm = nullptr, * router = nullptr, * probs_b = nullptr;
    Experts ex;
    // loop controllers (core layers): q, k, v, ffn
    float * msg_query = nullptr;
    float * down[4] = {}, * gain[4] = {}, * forget[4] = {};
};

constexpr int PF_T = 256;  // the largest prefill chunk (sliding-window rings hold window + PF_T rows)

struct Slot {
    float * k = nullptr, * v = nullptr;
    int8_t * kq = nullptr, * vq = nullptr;
    __half * ks = nullptr, * vs = nullptr;
    int window = 0, ring = 0;
    float * hist_k = nullptr, * hist_v = nullptr, * hist_o = nullptr, * hist_f = nullptr;
};

class Engine {
public:
    Engine(const std::string & path, int ctx, long long vram_expert_bytes = -1, const char * profile = nullptr, int n_threads = 0)
        : f_(path), path_(path), ctx_(ctx), offload_(vram_expert_bytes >= 0), budget_(vram_expert_bytes) {
        if (const char * kv = getenv("IK_KV"))
            kv_q8_ = !(strcmp(kv, "fp32") == 0 || strcmp(kv, "f32") == 0 || strcmp(kv, "0") == 0);
        ck(cudaStreamCreateWithFlags(&st_, cudaStreamNonBlocking), "stream");
        load_hparams();
        if (offload_) load_profile(profile);
        load_weights();
        alloc_state();
        if (offload_) start_offload(n_threads);
        build_graph();
    }

    ~Engine() {
        if (ram_fd_ >= 0) close(ram_fd_);
#ifdef _WIN32
        _aligned_free(ram_arena_);
#else
        free(ram_arena_);
#endif
        for (void * p : mmap_pinned_) cudaHostUnregister(p);
        for (uint8_t * p : zc_chunks_) cudaFreeHost(p);
        if (offload_) {
            stop_ = true;
            swap_cv_.notify_all();
            if (swapper_.joinable()) swapper_.join();
            if (coord_.joinable()) coord_.join();
            if (cp_thread_.joinable()) cp_thread_.join();
            if (getenv("IK_COPY_DEBUG") && dbg_jobs_)
                fprintf(stderr, "copy path: %lld jobs, per job: issue %.0f us (staging memcpy %.0f us), sync %.0f us, coordinator waited %.0f us\n",
                        (long long) dbg_jobs_, dbg_issue_ns_ / 1e3 / dbg_jobs_, dbg_stage_ns_ / 1e3 / dbg_jobs_, dbg_sync_ns_ / 1e3 / dbg_jobs_,
                        dbg_idle_wait_ns_ / 1e3 / dbg_jobs_);
            if (cp_stage_) cudaFreeHost(cp_stage_);
            if (cp_host_) cudaFreeHost(cp_host_);
            for (int b = 0; b < 2; ++b) if (swap_pinned_[b]) cudaFreeHost(swap_pinned_[b]);
            pool_.reset();
            if (h_x_) cudaFreeHost(h_x_);
        }
        if (exec_) cudaGraphExecDestroy(exec_);
        if (graph_) cudaGraphDestroy(graph_);
        for (int t = 0; t <= VF_MAX_T; ++t) {
            if (vf_exec_[t]) cudaGraphExecDestroy(vf_exec_[t]);
            if (vf_graph_[t]) cudaGraphDestroy(vf_graph_[t]);
        }
        dspark_.reset();
        for (void * p : owned_) cudaFree(p);
        if (pf_rows_h_) cudaFreeHost(pf_rows_h_);
        for (int b = 0; b < 2; ++b) {
            if (pf_pin_[b]) cudaFreeHost(pf_pin_[b]);
            if (pf_htab_gu_[b]) cudaFreeHost(pf_htab_gu_[b]);
            if (pf_htab_dn_[b]) cudaFreeHost(pf_htab_dn_[b]);
            if (pf_copy_ev_[b]) cudaEventDestroy(pf_copy_ev_[b]);
            if (pf_gpu_ev_[b]) cudaEventDestroy(pf_gpu_ev_[b]);
        }
        if (pf_copy_st_) cudaStreamDestroy(pf_copy_st_);
        if (pf_ids_ev_) cudaEventDestroy(pf_ids_ev_);
        if (pf_h_ids_) cudaFreeHost(pf_h_ids_);
        if (vf_ev_) cudaEventDestroy(vf_ev_);
        if (vf_h_ids_) cudaFreeHost(vf_h_ids_);
        if (vf_h_wts_) cudaFreeHost(vf_h_wts_);
        if (vf_h_x_) cudaFreeHost(vf_h_x_);
        if (vf_h_y_) cudaFreeHost(vf_h_y_);
        if (h_row_) cudaFreeHost(h_row_);
        if (h_logits_) cudaFreeHost(h_logits_);
        if (h_argmax_idx_) cudaFreeHost(h_argmax_idx_);
        if (h_argmax_val_) cudaFreeHost(h_argmax_val_);
        if (h_filt_ids_) cudaFreeHost(h_filt_ids_);
        if (h_filt_vals_) cudaFreeHost(h_filt_vals_);
        if (h_filt_cnt_) cudaFreeHost(h_filt_cnt_);
        if (vf_argmax_h_) cudaFreeHost(vf_argmax_h_);
        if (vf_argmax_val_h_) cudaFreeHost(vf_argmax_val_h_);
        if (h_pos_) cudaFreeHost(h_pos_);
        if (st_) cudaStreamDestroy(st_);
    }

    int n_vocab() const { return n_vocab_; }

    void reset() {
        for (auto & s : slots_) {
            ck(cudaMemsetAsync(s.hist_k, 0, sizeof(float) * 3 * kv_, st_), "reset");
            ck(cudaMemsetAsync(s.hist_v, 0, sizeof(float) * 3 * kv_, st_), "reset");
            ck(cudaMemsetAsync(s.hist_o, 0, sizeof(float) * 3 * E_, st_), "reset");
            ck(cudaMemsetAsync(s.hist_f, 0, sizeof(float) * 3 * E_, st_), "reset");
        }
        ck(cudaMemsetAsync(d_pos_, 0, sizeof(int), st_), "reset");
        ck(cudaStreamSynchronize(st_), "reset");
        reset_flags();
        pos_ = 0;
        last_maintain_pos_ = 0;
        vf_T_ = 0;
        if (dspark_) dspark_->reset();
        for (auto & c : ck_pool_) { c.pos = -1; c.pin = false; c.pin_t = 0; }
        pin_req_.clear();
        planned_.clear();
    }

    // feed n tokens in chunks of up to PF_T, batched (no logits); decode() continues after them
    void prefill(const int * toks, int n) {
        for (int i = 0; i < n; ++i)
            if (toks[i] < 0 || toks[i] >= n_vocab_) throw std::runtime_error("token out of range");
        if (pos_ + n > ctx_) throw std::runtime_error("context full");
        static const int chunk = getenv("IK_PF_CHUNK") ? std::max(1, std::min(PF_T, atoi(getenv("IK_PF_CHUNK")))) : PF_T;
        for (int i = 0; i < n; i += chunk) prefill_chunk(toks + i, std::min(chunk, n - i));
    }

    // ---- checkpoint pool in host RAM (modeled on megakernel/ck_pool.cu: coverage eviction + pinned system prompts).
    // Each checkpoint holds the 40 sliding-window rings and all 48 convolution histories (~110 MiB in q8_0, 0 MiB VRAM,
    // lazily allocated in pageable host RAM so WSL's pinned-memory cap is untouched). Full-attention KV caches are
    // positional and need nothing saved.
    size_t ck_bytes() const {
        size_t b = 0;
        for (const Slot & S : slots_) {
            if (S.ring) {
                if (kv_q8_) b += 2 * size_t(S.ring) * kv_ + 2 * sizeof(__half) * S.ring * (kv_ / 32);
                else b += 2 * sizeof(float) * S.ring * kv_;
            }
            b += sizeof(float) * 3 * (2 * kv_ + 2 * E_);
        }
        if (dspark_) b += dspark_->ck_bytes();
        return b;
    }

    void ck_init_pool() {
        int nslots = ram_ ? 2 : 4;
        if (const char * e = getenv("IK_CK_SLOTS")) nslots = std::max(1, std::min(32, atoi(e)));
        ck_pool_.resize(nslots);
        pin_max_ = nslots >= 4 ? std::min(2, nslots / 4) : (nslots >= 2 ? 1 : 0);
    }

    int ck_find(int pos) const {
        for (int i = 0; i < (int) ck_pool_.size(); ++i)
            if (ck_pool_[i].pos == pos && pos >= 0) return i;
        return -1;
    }

    int pin_count() const {
        int n = 0;
        for (const auto & k : ck_pool_) n += (k.pos >= 0 && k.pin);
        return n;
    }

    int pin_slot(int P) const {
        int best = -1;
        for (int i = 0; i < (int) ck_pool_.size(); ++i)
            if (ck_pool_[i].pos > P - 32 && ck_pool_[i].pos <= P && (best < 0 || ck_pool_[i].pos > ck_pool_[best].pos)) best = i;
        return best;
    }

    void pin_set(int si, unsigned t) {
        if (pin_max_ <= 0) return;
        ck_pool_[si].pin = true;
        ck_pool_[si].pin_t = t;
        while (pin_count() > pin_max_) {
            int old = -1;
            for (int i = 0; i < (int) ck_pool_.size(); ++i)
                if (i != si && ck_pool_[i].pos >= 0 && ck_pool_[i].pin &&
                    (old < 0 || ck_pool_[i].pin_t < ck_pool_[old].pin_t ||
                     (ck_pool_[i].pin_t == ck_pool_[old].pin_t && ck_pool_[i].pos < ck_pool_[old].pos)))
                    old = i;
            if (old < 0) break;
            ck_pool_[old].pin = false;
        }
    }

    // megakernel/ck_pool.cu's coverage eviction: thin older checkpoints geometrically, keep recent & pinned ones
    std::vector<int> ck_assign(const std::vector<int> & want, int keep, int E, const std::vector<char> & wpin = {}) {
        struct C { int pos, slot, wi; };
        std::vector<C> c;
        for (int i = 0; i < (int) ck_pool_.size(); ++i)
            if (ck_pool_[i].pos >= 0) c.push_back({ck_pool_[i].pos, i, -1});
        for (int i = 0; i < (int) want.size(); ++i) c.push_back({want[i], -1, i});
        std::stable_sort(c.begin(), c.end(), [](const C & a, const C & b) { return a.pos < b.pos; });
        while (c.size() > ck_pool_.size()) {
            int best = -1;
            double bs = 1e300;
            for (int i = 0; i < (int) c.size(); ++i) {
                if (c[i].wi >= 0 ? c[i].pos == keep || (c[i].wi < (int) wpin.size() && wpin[c[i].wi]) : ck_pool_[c[i].slot].pin) continue;
                const double prev = i ? c[i - 1].pos : 0, next = i + 1 < (int) c.size() ? c[i + 1].pos : E;
                const double sc = (next - prev) / std::max(1.0, E - prev);
                if (sc < bs) bs = sc, best = i;
            }
            if (best < 0) break;
            c.erase(c.begin() + best);
        }
        std::vector<char> used(ck_pool_.size(), 0);
        for (const auto & x : c) if (x.slot >= 0) used[x.slot] = 1;
        for (int i = 0; i < (int) ck_pool_.size(); ++i)
            if (!used[i]) { ck_pool_[i].pos = -1; ck_pool_[i].pin = false; ck_pool_[i].pin_t = 0; }
        std::vector<int> out(want.size(), -1);
        int f = 0;
        for (const auto & x : c)
            if (x.wi >= 0) {
                while (f < (int) ck_pool_.size() && used[f]) f++;
                if (f == (int) ck_pool_.size()) break;
                used[f] = 1;
                out[x.wi] = f;
            }
        return out;
    }

    std::vector<int> plan_checkpoints(const int * cks, int n_cks, int ck_exact, int base, int end) {
        planned_.clear();
        struct Cand { int pos; unsigned pin_t; };
        std::vector<Cand> cands;
        auto add = [&](int p, unsigned pt) {
            if (p <= base || p > end) return;
            for (auto & x : cands) { if (x.pos == p) { x.pin_t = std::max(x.pin_t, pt); return; } }
            cands.push_back({p, pt});
        };
        for (int i = 0; i < n_cks; ++i) add(cks[i], 0);
        const bool has_exact = ck_exact > base && ck_exact <= end;
        if (has_exact) add(ck_exact, 0);
        else add(end, 0);
        for (const auto & pr : pin_req_) add(pr.first, pr.second);
        pin_req_.clear();
        std::sort(cands.begin(), cands.end(), [](const Cand & a, const Cand & b) { return a.pos < b.pos; });
        std::vector<int> want;
        std::vector<char> wpin;
        std::vector<unsigned> wpt;
        for (const auto & x : cands) {
            const int ex = ck_find(x.pos);
            if (ex >= 0) {
                if (x.pin_t) pin_set(ex, x.pin_t);
                continue;
            }
            want.push_back(x.pos);
            wpin.push_back(x.pin_t != 0 ? 1 : 0);
            wpt.push_back(x.pin_t);
        }
        const int keep = has_exact ? ck_exact : end;
        const std::vector<int> slots = ck_assign(want, keep, end, wpin);
        std::vector<int> res;
        for (size_t i = 0; i < want.size(); ++i)
            if (slots[i] >= 0) {
                planned_.push_back({want[i], slots[i], wpt[i]});
                res.push_back(want[i]);
            }
        return res;
    }

    void save_checkpoint_slot(int si, unsigned pin_t) {
        CkEntry & e = ck_pool_[si];
        const size_t need = ck_bytes();
        if (e.data.size() != need) e.data.resize(need);
        uint8_t * dst = e.data.data();
        for (const Slot & S : slots_) {
            if (S.ring) {
                if (kv_q8_) {
                    const size_t qb = size_t(S.ring) * kv_, sb = sizeof(__half) * S.ring * (kv_ / 32);
                    ck(cudaMemcpyAsync(dst, S.kq, qb, cudaMemcpyDeviceToHost, st_), "checkpoint"); dst += qb;
                    ck(cudaMemcpyAsync(dst, S.vq, qb, cudaMemcpyDeviceToHost, st_), "checkpoint"); dst += qb;
                    ck(cudaMemcpyAsync(dst, S.ks, sb, cudaMemcpyDeviceToHost, st_), "checkpoint"); dst += sb;
                    ck(cudaMemcpyAsync(dst, S.vs, sb, cudaMemcpyDeviceToHost, st_), "checkpoint"); dst += sb;
                } else {
                    const size_t fb = sizeof(float) * S.ring * kv_;
                    ck(cudaMemcpyAsync(dst, S.k, fb, cudaMemcpyDeviceToHost, st_), "checkpoint"); dst += fb;
                    ck(cudaMemcpyAsync(dst, S.v, fb, cudaMemcpyDeviceToHost, st_), "checkpoint"); dst += fb;
                }
            }
            const size_t hk = sizeof(float) * 3 * kv_, he = sizeof(float) * 3 * E_;
            ck(cudaMemcpyAsync(dst, S.hist_k, hk, cudaMemcpyDeviceToHost, st_), "checkpoint"); dst += hk;
            ck(cudaMemcpyAsync(dst, S.hist_v, hk, cudaMemcpyDeviceToHost, st_), "checkpoint"); dst += hk;
            ck(cudaMemcpyAsync(dst, S.hist_o, he, cudaMemcpyDeviceToHost, st_), "checkpoint"); dst += he;
            ck(cudaMemcpyAsync(dst, S.hist_f, he, cudaMemcpyDeviceToHost, st_), "checkpoint"); dst += he;
        }
        if (dspark_) dspark_->ck_save(dst, st_);
        ck(cudaStreamSynchronize(st_), "checkpoint");
        e.pos = pos_;
        e.pin = false;
        e.pin_t = 0;
        if (pin_t) pin_set(si, pin_t);
    }

    void save_checkpoint() {
        int si = -1;
        unsigned pin_t = 0;
        for (size_t i = 0; i < planned_.size(); ++i) {
            if (planned_[i].pos == pos_) {
                si = planned_[i].slot;
                pin_t = planned_[i].pin_t;
                planned_.erase(planned_.begin() + i);
                break;
            }
        }
        if (si < 0) si = ck_find(pos_);
        if (si < 0) {
            const auto sl = ck_assign({pos_}, pos_, pos_);
            si = sl[0];
        }
        if (si < 0) return;
        save_checkpoint_slot(si, pin_t);
    }

    // the host-slot and copy flags hold position + 1 of the last pass: a rewind would let an old value pass for a new one
    void reset_flags() {
        for (auto & h : hslots_) { *h.h_ready = 0; *h.h_done = 0; }
        if (cp_host_) for (size_t a = 0; a < hslots_.size(); ++a) cp_host_[a].flag = 0;
    }

    bool restore_checkpoint_pos(int pos) {
        const int si = ck_find(pos);
        if (si < 0) return false;
        reset_flags();
        const uint8_t * src = ck_pool_[si].data.data();
        for (Slot & S : slots_) {
            if (S.ring) {
                if (kv_q8_) {
                    const size_t qb = size_t(S.ring) * kv_, sb = sizeof(__half) * S.ring * (kv_ / 32);
                    ck(cudaMemcpyAsync(S.kq, src, qb, cudaMemcpyHostToDevice, st_), "restore"); src += qb;
                    ck(cudaMemcpyAsync(S.vq, src, qb, cudaMemcpyHostToDevice, st_), "restore"); src += qb;
                    ck(cudaMemcpyAsync(S.ks, src, sb, cudaMemcpyHostToDevice, st_), "restore"); src += sb;
                    ck(cudaMemcpyAsync(S.vs, src, sb, cudaMemcpyHostToDevice, st_), "restore"); src += sb;
                } else {
                    const size_t fb = sizeof(float) * S.ring * kv_;
                    ck(cudaMemcpyAsync(S.k, src, fb, cudaMemcpyHostToDevice, st_), "restore"); src += fb;
                    ck(cudaMemcpyAsync(S.v, src, fb, cudaMemcpyHostToDevice, st_), "restore"); src += fb;
                }
            }
            const size_t hk = sizeof(float) * 3 * kv_, he = sizeof(float) * 3 * E_;
            ck(cudaMemcpyAsync(S.hist_k, src, hk, cudaMemcpyHostToDevice, st_), "restore"); src += hk;
            ck(cudaMemcpyAsync(S.hist_v, src, hk, cudaMemcpyHostToDevice, st_), "restore"); src += hk;
            ck(cudaMemcpyAsync(S.hist_o, src, he, cudaMemcpyHostToDevice, st_), "restore"); src += he;
            ck(cudaMemcpyAsync(S.hist_f, src, he, cudaMemcpyHostToDevice, st_), "restore"); src += he;
        }
        if (dspark_) dspark_->ck_restore(src, st_);
        pos_ = pos;
        last_maintain_pos_ = pos;
        vf_T_ = 0;
        ck(cudaMemcpyAsync(d_pos_, &pos_, sizeof(int), cudaMemcpyHostToDevice, st_), "restore");
        ck(cudaStreamSynchronize(st_), "restore");
        for (auto & o : ck_pool_)
            if (o.pos > pos_) { o.pos = -1; o.pin = false; o.pin_t = 0; }
        planned_.clear();
        return true;
    }

    bool restore_checkpoint() {
        const int p = checkpoint_pos();
        return p >= 0 && restore_checkpoint_pos(p);
    }

    int checkpoint_pos() const {
        int best = -1;
        for (const auto & e : ck_pool_) if (e.pos > best) best = e.pos;
        return best;
    }

    std::vector<int> checkpoints() const {
        std::vector<int> v;
        for (const auto & e : ck_pool_) if (e.pos >= 0) v.push_back(e.pos);
        std::sort(v.begin(), v.end());
        return v;
    }

    int ck_pin(const int * positions, int n) {
        pin_req_.clear();
        for (int i = 0; i < n; ++i) {
            const int P = positions[i];
            if (P <= 0) continue;
            const unsigned t = ++pin_clock_;
            const int s = pin_slot(P);
            if (s >= 0) pin_set(s, t);
            else pin_req_.push_back({P, t});
        }
        return pin_count();
    }

    std::vector<int> ck_pinned() const {
        std::vector<int> v;
        for (const auto & e : ck_pool_) if (e.pos >= 0 && e.pin) v.push_back(e.pos);
        std::sort(v.begin(), v.end());
        return v;
    }

    // debugging: non-finite values in the state (every slot's KV rows up to the position, convolution histories);
    // prints the first few
    long long check_state() {
        ck(cudaStreamSynchronize(st_), "check");
        long long bad = 0;
        auto scan = [&](const char * what, int slot, const float * p, size_t n) {
            std::vector<float> h(n);
            ck(cudaMemcpy(h.data(), p, n * 4, cudaMemcpyDeviceToHost), "check");
            for (size_t i = 0; i < n; ++i)
                if (!std::isfinite(h[i])) {
                    if (bad < 5) fprintf(stderr, "check_state: %s slot %d index %zu (row %zu)\n", what, slot, i, i / (std::string(what)[0] == 'h' ? 1 : kv_));
                    bad++;
                }
        };
        auto scan_h = [&](const char * what, int slot, const __half * p, size_t n) {
            std::vector<__half> h(n);
            ck(cudaMemcpy(h.data(), p, n * sizeof(__half), cudaMemcpyDeviceToHost), "check");
            for (size_t i = 0; i < n; ++i)
                if (!std::isfinite(__half2float(h[i]))) {
                    if (bad < 5) fprintf(stderr, "check_state: %s slot %d index %zu\n", what, slot, i);
                    bad++;
                }
        };
        for (size_t s = 0; s < slots_.size(); ++s) {
            const Slot & S = slots_[s];
            const size_t rows = S.ring ? std::min<size_t>(S.ring, pos_) : pos_;
            if (kv_q8_) {
                scan_h("k_s", (int) s, S.ks, rows * (kv_ / 32));
                scan_h("v_s", (int) s, S.vs, rows * (kv_ / 32));
            } else {
                scan("k", (int) s, S.k, rows * kv_);
                scan("v", (int) s, S.v, rows * kv_);
            }
            scan("hist_k", (int) s, S.hist_k, 3 * kv_);
            scan("hist_o", (int) s, S.hist_o, 3 * E_);
            scan("hist_f", (int) s, S.hist_f, 3 * E_);
        }
        return bad;
    }

    // offload statistics since the last call: reads served from VRAM, computed by the CPU, experts swapped in
    void stats(long long * out) {
        out[0] = n_gpu_.exchange(0); out[1] = n_cpu_.exchange(0); out[2] = n_swaps_; n_swaps_ = 0;
        out[3] = cpu_ns_.exchange(0); out[4] = cpu_apps_.exchange(0);
        out[5] = ssd_reads_.exchange(0); out[6] = ssd_bytes_.exchange(0); out[7] = ssd_ns_.exchange(0);
        out[8] = n_copy_.exchange(0); out[9] = cp_ns_.exchange(0);
    }

    void launch_decode(int token) {
        if (token < 0 || token >= n_vocab_) throw std::runtime_error("token out of range");
        if (pos_ >= ctx_) throw std::runtime_error("context full");
        const gguf::Tensor & emb = f_.get("token_embd.weight");
        const size_t rb = gguf::type_row_bytes(emb.type, E_);
        memcpy(h_row_, emb.data + size_t(token) * rb, rb);
        if (offload_) maintain();
        if (getenv("IK_EAGER")) record_step();  // debugging: run the step without the graph (allocates every call)
        else ck(cudaGraphLaunch(exec_, st_), "graph launch");
    }

    void decode(int token, float * logits) {
        launch_decode(token);
        if (dspark_) dspark_->store_context(d_taps_1_, pos_, 1, st_);
        ck(cudaMemcpyAsync(h_logits_, d_logits_, sizeof(float) * n_vocab_, cudaMemcpyDeviceToHost, st_), "logits");
        ck(cudaStreamSynchronize(st_), "decode");
        memcpy(logits, h_logits_, sizeof(float) * n_vocab_);
        pos_++;
    }

    // Zero-Logits-D2H greedy decode: runs the CUDA graph (with fused softcap + penalties + argmax) and reads 4 bytes
    int decode_greedy(int token) {
        launch_decode(token);
        if (dspark_) dspark_->store_context(d_taps_1_, pos_, 1, st_);
        ck(cudaStreamSynchronize(st_), "decode greedy");
        pos_++;
        return *h_argmax_idx_;
    }

    // Zero-Logits-D2H filtered decode for stochastic sampling: returns the number of tokens with logit >= max - margin.
    // When count <= cap (4096), ids_out/vals_out hold all surviving (id, logit) pairs; when count > cap, call read_logits().
    int decode_filtered(int token, float margin, int32_t * ids_out, float * vals_out, int cap, float * max_out) {
        launch_decode(token);
        if (dspark_) dspark_->store_context(d_taps_1_, pos_, 1, st_);
        const int c = std::min(cap, FILT_CAP);
        ik::filter_logits(d_logits_, n_vocab_, d_argmax_out_val_, margin, d_filt_ids_, d_filt_vals_, d_filt_cnt_, c, st_);
        ck(cudaStreamSynchronize(st_), "decode filtered");
        pos_++;
        if (max_out) *max_out = *h_argmax_val_;
        const int n = *h_filt_cnt_;
        if (n <= c) {
            memcpy(ids_out, h_filt_ids_, sizeof(int32_t) * n);
            memcpy(vals_out, h_filt_vals_, sizeof(float) * n);
        }
        return n;
    }

    void load_mtp(const std::string & path) {
        prefill_alloc();
        auto lm_fn = [this](const float * X, float * Y, int T, cudaStream_t s) {
            (void) s;
            pf_tmp_.used = 0;
            mv_pf(out_, X, Y, T);
        };
        dspark_ = std::make_unique<ik::DSpark>(path, ctx_, n_vocab_, n_vocab_pad_, softcap_, lm_fn, st_);
    }

    bool has_mtp() const { return dspark_ != nullptr; }

    int draft(int anchor_tok, int n_draft, int * out_drafts) {
        if (!dspark_ || pos_ <= 0 || n_draft <= 0) return 0;
        if (anchor_tok < 0 || anchor_tok >= n_vocab_) throw std::runtime_error("draft: token out of range");
        const int nd = std::min({n_draft, ik::DSpark::N_SPEC, ctx_ - pos_ - 1});
        if (nd <= 0) return 0;
        const gguf::Tensor & emb = f_.get("token_embd.weight");
        memcpy(h_row_, emb.data + size_t(anchor_tok) * emb_rb_, emb_rb_);
        ck(cudaMemcpyAsync(d_row_, h_row_, emb_rb_, cudaMemcpyHostToDevice, st_), "draft row");
        ik::embed(d_row_, tok_norm_, d_anchor_emb_, E_, eps_, st_, emb_type_);
        auto lm_fn = [this](const float * X, float * Y, int T, cudaStream_t s) {
            (void) s;
            pf_tmp_.used = 0;
            mv_pf(out_, X, Y, T);
        };
        dspark_->draft(d_anchor_emb_, anchor_tok, pos_, nd, lm_fn, out_drafts, st_);
        return nd;
    }

    void read_logits(float * logits) {
        ck(cudaMemcpyAsync(h_logits_, d_logits_, sizeof(float) * n_vocab_, cudaMemcpyDeviceToHost, st_), "read logits");
        ck(cudaStreamSynchronize(st_), "read logits");
        memcpy(logits, h_logits_, sizeof(float) * n_vocab_);
    }

    void set_penalty(float pres, float freq) {
        const float p[2] = {pres, freq};
        ck(cudaMemcpyAsync(d_pen_params_, p, sizeof(p), cudaMemcpyHostToDevice, st_), "pen params");
        ck(cudaStreamSynchronize(st_), "pen params");
    }

    void pen_clear() {
        ck(cudaMemsetAsync(d_pen_counts_, 0, sizeof(int32_t) * n_vocab_, st_), "pen clear");
    }

    void pen_inc(int tok) {
        if (tok >= 0 && tok < n_vocab_) ik::pen_inc(d_pen_counts_, tok, st_);
    }

    // exact multi-token verification (1 <= T <= VF_MAX_T): computes logits[0..T-1] and saves per-token conv-history
    // snapshots inside pf_keep_ (no extra VRAM). Call verify_commit(acc) with 1 <= acc <= T to commit the accepted
    // prefix and roll back the conv histories cleanly.
    static constexpr int VF_MAX_T = 8;

    void verify_record(int T, bool use_dpos) {
        pf_keep_.used = 0;
        vf_snap_stride_ = size_t(T) * 3 * (2 * size_t(kv_) + 2 * size_t(E_));
        vf_snaps_ = pf_keep_.take(slots_.size() * vf_snap_stride_);
        ck(cudaMemcpyAsync(pf_rows_d_, pf_rows_h_, emb_rb_ * T, cudaMemcpyHostToDevice, st_), "vf rows");
        const int pos0 = pos_;
        const int * dpos = use_dpos ? d_pos_ : nullptr;
        float * x0 = pf_keep_.take(size_t(T) * E_);
        ik::pf::embed(pf_rows_d_, tok_norm_, x0, E_, eps_, T, st_, emb_type_);
        const float * cur = x0;
        std::vector<const float *> blocks;
        auto tap = [&](const float * src, int tidx) {
            scatter_tap_kernel<<<dim3((E_ + 255) / 256, T), 256, 0, st_>>>(src, vf_taps_, tidx, T, E_);
        };
        for (int il = 0; il < core_start_; ++il) {
            if (il == 5) tap(cur, 0);
            cur = layer_pf(il, il, il, cur, blocks, nullptr, T, pos0, true, 0, dpos);
        }
        const float * prelude = cur;
        const std::vector<const float *> prelude_blocks = blocks;
        const int n_core = core_end_ - core_start_;
        for (int ps = 0; ps < n_passes_; ++ps) {
            std::vector<const float *> bank;
            const float * bank_out = nullptr;
            if (ps > 0) {
                bank.assign(blocks.begin() + prelude_blocks.size() + 1, blocks.end());
                bank_out = cur;
                cur = prelude;
                blocks = prelude_blocks;
            }
            for (int ip = core_start_; ip < core_end_; ++ip) {
                const int slot = ps == 0 ? ip : n_layer_ + (ps - 1) * n_core + (ip - core_start_);
                float * msgs = nullptr;
                if (ps > 0) {
                    msgs = pf_msgs_;
                    ik::pf::mix(states(bank, bank_out), layers_[ip].msg_query, 4, msgs, E_, T, st_);
                }
                if (ps == 1 && ip == 11) tap(cur, 1);
                if (ps == 2 && ip == 17) tap(cur, 2);
                cur = layer_pf(ip, ip, slot, cur, blocks, msgs, T, pos0, true, ps + 1, dpos);
            }
        }
        for (int il = core_end_; il < n_layer_; ++il) cur = layer_pf(il, il, il, cur, blocks, nullptr, T, pos0, true, 0, dpos);
        pf_tmp_.used = 0;
        float * x = pf_tmp_.take(size_t(T) * E_);
        ik::pf::mix(states(blocks, cur), out_res_query_, 1, x, E_, T, st_);
        float * xn = pf_tmp_.take(size_t(T) * E_);
        {
            ik::NormArgs na{};
            na.x[0] = x; na.w[0] = out_norm_; na.out[0] = xn;
            ik::pf::rmsnorm(na, 1, E_, E_, E_, eps_, T, st_);
        }
        tap(xn, 3);
        float * raw = pf_tmp_.take(size_t(T) * n_vocab_pad_);
        mv_pf(out_, xn, raw, T);
        for (int t = 0; t < T; ++t) {
            ik::softcap_pen_argmax(raw + size_t(t) * n_vocab_pad_, vf_out_logits_ + size_t(t) * n_vocab_,
                                   d_pen_counts_, d_pen_zero_, n_vocab_, softcap_,
                                   d_argmax_val_, d_argmax_idx_, vf_argmax_d_ + t, vf_argmax_val_d_ + t, st_);
        }
    }

    void run_verify(const int * toks, int T) {
        if (T < 1 || T > VF_MAX_T) throw std::runtime_error("verify: batch size out of range");
        for (int i = 0; i < T; ++i)
            if (toks[i] < 0 || toks[i] >= n_vocab_) throw std::runtime_error("token out of range");
        if (pos_ + T > ctx_) throw std::runtime_error("context full");
        if (offload_) maintain();
        prefill_alloc();
        vf_pos0_ = pos_;
        vf_T_ = T;
        vf_snap_stride_ = size_t(T) * 3 * (2 * size_t(kv_) + 2 * size_t(E_));
        vf_snaps_ = pf_keep_.base;  // first allocation in verify_record(T) after pf_keep_.used = 0
        const gguf::Tensor & emb = f_.get("token_embd.weight");
        const size_t rb = gguf::type_row_bytes(emb.type, E_);
        for (int t = 0; t < T; ++t) memcpy(pf_rows_h_ + rb * t, emb.data + size_t(toks[t]) * rb, rb);
        if (!offload_ && !getenv("IK_EAGER")) {
            if (!vf_exec_[T]) {
                ck(cudaStreamBeginCapture(st_, cudaStreamCaptureModeRelaxed), "vf capture");
                try {
                    verify_record(T, true);
                } catch (...) {
                    cudaGraph_t g;
                    cudaStreamEndCapture(st_, &g);
                    if (g) cudaGraphDestroy(g);
                    throw;
                }
                ck(cudaStreamEndCapture(st_, &vf_graph_[T]), "vf end capture");
                ck(cudaGraphInstantiate(&vf_exec_[T], vf_graph_[T], 0), "vf instantiate");
            }
            ck(cudaGraphLaunch(vf_exec_[T], st_), "vf launch");
        } else {
            verify_record(T, false);
        }
    }

    void verify(const int * toks, int T, float * logits_out) {
        run_verify(toks, T);
        ck(cudaMemcpyAsync(logits_out, vf_out_logits_, sizeof(float) * T * n_vocab_, cudaMemcpyDeviceToHost, st_), "vf logits");
        ck(cudaStreamSynchronize(st_), "verify");
    }

    void verify_greedy(const int * toks, int T, int * argmax_out) {
        run_verify(toks, T);
        ck(cudaStreamSynchronize(st_), "verify greedy");
        for (int t = 0; t < T; ++t) argmax_out[t] = vf_argmax_h_[t];
    }

    void verify_commit(int acc) {
        if (vf_T_ <= 0 || acc < 1 || acc > vf_T_) throw std::runtime_error("verify_commit: invalid count");
        if (acc < vf_T_) {
            const int n_max = std::max(3 * kv_, 3 * E_);
            rollback_snaps_kernel<<<dim3((n_max + 255) / 256, (int) slots_.size()), 256, 0, st_>>>(
                d_hist_ptrs_, vf_snaps_, vf_snap_stride_, vf_T_, acc - 1, kv_, E_);
        }
        if (dspark_) dspark_->store_context(vf_taps_, vf_pos0_, acc, st_);
        pos_ = vf_pos0_ + acc;
        vf_T_ = 0;
        set_pos_kernel<<<1, 1, 0, st_>>>(d_pos_, pos_);
    }

private:
    gguf::File f_;
    std::string path_;
    int ctx_;
    cudaStream_t st_ = nullptr;
    std::vector<void *> owned_;

    int n_layer_ = 0, E_ = 0, H_ = 0, HK_ = 0, D_ = 0, n_expert_ = 0, n_used_ = 0, n_ff_ = 0, n_swa_ = 0, block_ = 0;
    int n_rel_ = 0, extent_ = 0, core_start_ = 0, core_end_ = 0, n_passes_ = 0, rank_ = 0, n_vocab_ = 0, n_vocab_pad_ = 0, kv_ = 0;
    float eps_ = 0, w_scale_ = 0, beta_ = 0, lb_ = 0, softcap_ = 0, qk_clip_ = 0, rel_clip_ = 0, log_floor_ = 0, log_alpha_ = 0;
    std::vector<int> swa_;

    std::vector<Layer> layers_;
    Q8 out_;
    float * tok_norm_ = nullptr, * out_norm_ = nullptr, * out_res_query_ = nullptr;
    std::vector<Slot> slots_;

    // ------------------------------------------------------------ offload state
    bool offload_ = false;
    long long budget_ = -1;
    std::vector<float> profile_;          // reads per token of each (layer, expert); empty: none
    // one pool per expert format (fast type, or the two GGUF types): slots of equal size
    struct Pool { bool fast; int type, gu_gtype, dn_gtype; size_t gu_bytes, dn_bytes, slot_bytes; uint8_t * base = nullptr;
                  std::vector<std::pair<int, int>> owner; };
    std::vector<Pool> pools_;
    struct HSlot { volatile int * h_ready; volatile int * h_done; int32_t * h_ids; float * h_wts; float * h_x; float * h_y; ik::HostSlot dev; int layer; };
    std::vector<HSlot> hslots_;           // one per layer application
    float * h_x_ = nullptr;               // the mapped allocation behind hslots_
    std::unique_ptr<SpinPool> pool_;
    std::thread coord_;
    std::atomic<bool> stop_{false};
    std::vector<float> reads_;            // reads per (layer, expert) since the last maintenance (coordinator thread)
    std::vector<float> score_;            // decayed reads (cache manager)
    std::atomic<long long> n_gpu_{0}, n_cpu_{0}, cpu_ns_{0}, cpu_apps_{0};
    long long n_swaps_ = 0;
    int app_ = 0;                         // layer application being recorded
    uint8_t * swap_staging_ = nullptr;
    // asynchronous swaps: a thread copies an incoming expert from the mmap into pinned memory and uploads it on its own
    // stream while decoding goes on; the tables point at it once the copy has landed (commit_swaps)
    struct SwapJob { int il, e, slot; cudaEvent_t done = nullptr; };
    std::thread swapper_;
    std::mutex swap_mu_;
    std::condition_variable swap_cv_;
    std::deque<SwapJob> swap_queue_;      // to copy (swap thread)
    std::vector<SwapJob> swap_flight_;    // copying or copied, not yet committed (decode thread)
    std::vector<char> swap_busy_;         // per (layer, expert): queued or in flight
    cudaStream_t copy_st_ = nullptr;
    uint8_t * swap_pinned_[2] = {nullptr, nullptr};
    uint8_t * swap_dev_[2] = {nullptr, nullptr};
    cudaEvent_t swap_free_[2] = {nullptr, nullptr}, swap_landed_[2] = {nullptr, nullptr};
    bool swap_async_ = true;
    // zero-copy share of the misses (IK_ZC_MIB of pinned expert pages, IK_ZC_SHARE of the misses that have them)
    bool zc_ = false;
    float zc_share_ = 0.f;
    std::vector<uint8_t *> zc_chunks_;                         // pinned copies of the zero-copy experts
    std::vector<void *> mmap_pinned_;                          // native Linux: mmap'd ranges pinned via cudaHostRegister

    // prefill: persistent (blocks, layer outputs) and per-layer scratch arenas, token-row staging, expert streaming
    struct Arena {
        float * base = nullptr; size_t cap = 0, used = 0;
        float * take(size_t n) {
            n = (n + 63) / 64 * 64;
            if (used + n > cap) throw std::runtime_error("prefill arena exhausted");
            float * p = base + used; used += n; return p;
        }
    };
    Arena pf_keep_, pf_tmp_;
    float * pf_msgs_ = nullptr;              // a loop pass's messages [PF_T][4][E], used by the layer that follows
    uint8_t * pf_rows_h_ = nullptr, * pf_rows_d_ = nullptr;
    int * pf_offsets_ = nullptr, * pf_list_ = nullptr, * pf_active_ = nullptr, * pf_n_active_ = nullptr;
    int * h_pos_ = nullptr;
    uint8_t * pf_stage_ = nullptr;           // offload: experts streamed in for a chunk (GGUF blocks + repacked)
    uint8_t * pf_pin_[2] = {nullptr, nullptr};  // offload: 2-way double-buffered pinned staging for async DMA
    void ** pf_htab_gu_[2] = {nullptr, nullptr}, ** pf_htab_dn_[2] = {nullptr, nullptr};
    void ** pf_tab_gu_[2] = {nullptr, nullptr}, ** pf_tab_dn_[2] = {nullptr, nullptr};
    cudaStream_t pf_copy_st_ = nullptr;
    cudaEvent_t pf_copy_ev_[2] = {nullptr, nullptr}, pf_gpu_ev_[2] = {nullptr, nullptr}, pf_ids_ev_ = nullptr;
    int32_t * pf_h_ids_ = nullptr;
    bool pf_gpu_busy_[2] = {false, false};
    int pf_cur_ = 0;
    size_t pf_stage_slot_ = 0;
    int pf_stage_n_ = 0, pf_pin_n_ = 0;
    // multi-token verification state (uses pf_keep_ / pf_tmp_, no extra VRAM)
    int vf_pos0_ = 0, vf_T_ = 0, last_maintain_pos_ = 0;
    float * vf_snaps_ = nullptr;
    size_t vf_snap_stride_ = 0;
    int32_t * vf_h_ids_ = nullptr;
    float * vf_h_wts_ = nullptr, * vf_h_x_ = nullptr, * vf_h_y_ = nullptr;
    cudaEvent_t vf_ev_ = nullptr;
    cudaGraph_t vf_graph_[VF_MAX_T + 1] = {};
    cudaGraphExec_t vf_exec_[VF_MAX_T + 1] = {};
    float * vf_out_logits_ = nullptr;
    float * vf_taps_ = nullptr;
    float * pf_taps_ = nullptr;
    float * d_taps_1_ = nullptr;
    float * d_anchor_emb_ = nullptr;
    float * d_pen_zero_ = nullptr;
    int32_t * vf_argmax_h_ = nullptr, * vf_argmax_d_ = nullptr;
    float * vf_argmax_val_h_ = nullptr, * vf_argmax_val_d_ = nullptr;
    HistPtrs * d_hist_ptrs_ = nullptr;
    std::unique_ptr<ik::DSpark> dspark_;

    // copy path (IK_COPY=1, off by default): each layer application's missing experts are split between the CPU, which
    // computes its share in RAM, and a copy thread, which sends the rest to VRAM scratch slots for the GPU to compute:
    // those in the pinned arena (IK_ZC_MIB, default 768 with the copy path), and with IK_COPY_STAGE=1 others staged
    // through a pinned buffer. The split balances the two sides' measured costs. Measured on WSL2 (RTX 3090, DDR4) it
    // loses: staging from the page cache ran at 3.7 GB/s beside the CPU kernels, slower than computing the expert, and
    // the arena's RAM pushed other experts out of the page cache (inkling/engine/README.md). It is meant for systems
    // where copies can come straight from pinned memory without that cost.
    bool cp_ = false;
    cudaStream_t cp_st_ = nullptr;
    uint8_t * cp_dev_ = nullptr, * cp_stage_ = nullptr;  // n_used_ slots each
    size_t cp_slot_ = 0, cp_gu_area_ = 0;
    // per layer application, in mapped host memory: the selection, then the flag (= position + 1) once the copies have
    // landed. (Written by the host: under WSL a kernel on another stream does not run while the graph's wait spins.)
    struct CopyFlag { volatile int flag; ik::CopySel sel; char pad[64 - sizeof(int) - sizeof(ik::CopySel)]; };
    CopyFlag * cp_host_ = nullptr;
    int * cp_flag_ = nullptr;                             // device pointers into cp_host_
    ik::CopySel * cp_sel_ = nullptr;
    struct CopyJob { int app = 0, value = 0, n = 0; int k[8]; const uint8_t * g[8], * d[8]; bool pinned[8]; size_t gb = 0, db = 0; };
    CopyJob cp_job_;
    std::atomic<uint64_t> cp_seq_{0};
    std::atomic<bool> cp_idle_{true};
    std::thread cp_thread_;
    float cp_frac_ = -1.f;                                // IK_COPY_FRAC: a fixed share instead of the cost model
    bool cp_stage_on_ = false;                            // IK_COPY_STAGE: also copy experts staged from the page cache
    // cost model (ns): CPU per expert of cost 1 (cpu_cost), copy per byte from the staging path and from pinned memory
    std::atomic<double> cpu_unit_ns_{400e3}, stage_ns_b_{1.0 / 11.0}, pin_ns_b_{1.0 / 24.0};
    std::atomic<long long> n_copy_{0}, cp_ns_{0};
    std::atomic<long long> dbg_jobs_{0}, dbg_issue_ns_{0}, dbg_sync_ns_{0}, dbg_idle_wait_ns_{0}, dbg_stage_ns_{0};
    // SSD tier (IK_RAM_MIB): a RAM cache of whole experts, filled with block-aligned pread (O_DIRECT where the file
    // system allows) instead of page faults through the mmap; least recently used slots are evicted
    bool ram_ = false;
    int ram_fd_ = -1;
    bool ram_direct_ = false;
    uint8_t * ram_arena_ = nullptr;
    size_t ram_slot_bytes_ = 0, ram_gu_area_ = 0;
    std::vector<int> ram_slot_of_, ram_owner_;
    std::vector<uint64_t> ram_stamp_;
    std::vector<char> ram_loop_pin_;
    uint64_t ram_clock_ = 0;
    std::atomic<long long> ssd_bytes_{0}, ssd_reads_{0}, ssd_ns_{0};
    bool kv_q8_ = true;  // q8_0 KV cache (default, matching megakernel); IK_KV=fp32 for fp32
    struct CkEntry { std::vector<uint8_t> data; int pos = -1; bool pin = false; unsigned pin_t = 0; };
    struct CkPlan { int pos, slot; unsigned pin_t; };
    std::vector<CkEntry> ck_pool_;
    int pin_max_ = 0;
    unsigned pin_clock_ = 0;
    std::vector<std::pair<int, unsigned>> pin_req_;
    std::vector<CkPlan> planned_;

    int * d_pos_ = nullptr;
    int emb_type_ = gguf::Q8_0;
    size_t emb_rb_ = 0;
    uint8_t * h_row_ = nullptr, * d_row_ = nullptr;
    float * h_logits_ = nullptr, * d_logits_ = nullptr;
    // Zero-Logits-D2H on-GPU penalties, argmax and threshold filter
    static constexpr int FILT_CAP = 4096;
    int32_t * d_pen_counts_ = nullptr;
    float * d_pen_params_ = nullptr;
    float * d_argmax_val_ = nullptr;
    int32_t * d_argmax_idx_ = nullptr;
    int32_t * h_argmax_idx_ = nullptr, * d_argmax_out_idx_ = nullptr;
    float * h_argmax_val_ = nullptr, * d_argmax_out_val_ = nullptr;
    int32_t * h_filt_ids_ = nullptr, * d_filt_ids_ = nullptr;
    float * h_filt_vals_ = nullptr, * d_filt_vals_ = nullptr;
    int * h_filt_cnt_ = nullptr, * d_filt_cnt_ = nullptr;
    int pos_ = 0;

    cudaGraph_t graph_ = nullptr;
    cudaGraphExec_t exec_ = nullptr;

public:
    // named buffers of the recorded step, for debugging (ik_debug_*)
    struct Dbg { std::string name; const float * p; int n; };
    std::vector<Dbg> dbg_;
private:
    bool recording_dbg_ = true;
    void dbg(const std::string & name, const float * p, int n) { if (recording_dbg_) dbg_.push_back({name, p, n}); }

    // ------------------------------------------------------------ loading

    template <typename T> T * dalloc(size_t n) {
        void * p = nullptr;
        ck(cudaMalloc(&p, sizeof(T) * n), "cudaMalloc");
        owned_.push_back(p);
        return (T *) p;
    }

    std::string key(const char * k) const { return std::string("inkling2_loop.") + k; }
    uint64_t geti(const char * k) const {
        auto it = f_.ints.find(key(k));
        if (it == f_.ints.end()) throw std::runtime_error(std::string("missing ") + key(k));
        return it->second;
    }
    float getf(const char * k) const {
        auto it = f_.floats.find(key(k));
        if (it == f_.floats.end()) throw std::runtime_error(std::string("missing ") + key(k));
        return it->second;
    }
    const std::vector<int64_t> & geta(const char * k) const {
        auto it = f_.iarrs.find(key(k));
        if (it == f_.iarrs.end()) throw std::runtime_error(std::string("missing ") + key(k));
        return it->second;
    }

    void load_hparams() {
        auto arch = f_.strs.find("general.architecture");
        if (arch == f_.strs.end() || arch->second != "inkling2_loop")
            throw std::runtime_error("not an inkling2_loop GGUF (use the Mainline_Support files of unsloth/Inkling-2-Mini-GGUF)");
        n_layer_ = (int) geti("block_count");
        E_ = (int) geti("embedding_length");
        H_ = (int) geta("attention.head_count")[0];
        HK_ = (int) geta("attention.head_count_kv")[0];
        D_ = (int) geti("attention.key_length");
        n_expert_ = (int) geti("expert_count");
        n_used_ = (int) geti("expert_used_count");
        n_ff_ = (int) geti("expert_feed_forward_length");
        eps_ = getf("attention.layer_norm_rms_epsilon");
        n_swa_ = (int) geti("attention.sliding_window");
        for (int64_t v : geta("attention.sliding_window_pattern")) swa_.push_back((int) v);
        w_scale_ = getf("expert_weights_scale");
        beta_ = getf("activation.situ_beta");
        lb_ = getf("activation.situ_linear_beta");
        block_ = (int) geti("attn_res.block_size");
        softcap_ = getf("final_logit_softcapping");
        n_rel_ = (int) geti("attention.relative_length");
        extent_ = (int) geti("attention.relative_extent");
        qk_clip_ = getf("attention.qk_clip");
        rel_clip_ = getf("attention.relative_clip");
        log_floor_ = (float) geti("attention.log_scale_floor");
        log_alpha_ = getf("attention.log_scale_alpha");
        core_start_ = (int) geti("loop.start");
        core_end_ = (int) geti("loop.end");
        n_passes_ = (int) geti("loop.count");
        rank_ = (int) geti("loop.gate_rank");
        n_vocab_ = (int) geti("vocab_size_unpadded");
        n_vocab_pad_ = (int) geti("vocab_size");
        kv_ = HK_ * D_;
        for (int64_t h : geta("attention.head_count")) if (h != H_) throw std::runtime_error("per-layer head counts differ");
        if ((int) swa_.size() != n_layer_) throw std::runtime_error("sliding_window_pattern length");
        if (D_ != 128 || n_rel_ != 16 || E_ != 2304 || n_ff_ != 1152 || H_ % HK_ || (int) geti("shortconv.l_cache") != 4 || rank_ > 64)
            throw std::runtime_error("unsupported Inkling dimensions (the kernels are sized for Inkling 2 Mini)");
    }

    // F32 or BF16 tensor to device fp32
    float * f32(const std::string & name, bool required = true) {
        auto it = f_.tensors.find(name);
        if (it == f_.tensors.end()) {
            if (required) throw std::runtime_error("missing tensor " + name);
            return nullptr;
        }
        const gguf::Tensor & t = it->second;
        size_t n = 1;
        for (uint64_t d : t.ne) n *= d;
        std::vector<float> h(n);
        if (t.type == gguf::F32) {
            memcpy(h.data(), t.data, n * 4);
        } else if (t.type == gguf::BF16) {
            const uint16_t * s = (const uint16_t *) t.data;
            for (size_t i = 0; i < n; ++i) { uint32_t b = uint32_t(s[i]) << 16; memcpy(&h[i], &b, 4); }
        } else {
            throw std::runtime_error(name + ": expected F32 or BF16");
        }
        float * d = dalloc<float>(n);
        upload(d, h.data(), n * 4);
        return d;
    }

    // host -> device on the engine's stream, finished before returning. A plain cudaMemcpy from pageable memory (the
    // mmap'd GGUF) can return before its DMA lands, and kernels on the non-blocking stream would not wait for it.
    void upload(void * dst, const void * src, size_t bytes) {
        ck(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, st_), "upload");
        ck(cudaStreamSynchronize(st_), "upload");
    }

    // raw GGUF blocks -> repacked device matrix
    void * repack(const uint8_t * src, size_t bytes, int type, int rows, int n, uint8_t * staging) {
        upload(staging, src, bytes);
        void * dst = dalloc<uint8_t>(ik_matrix_bytes(type, rows, n));
        ck(ik_repack(type, staging, dst, rows, n, st_), "repack");
        ck(cudaStreamSynchronize(st_), "repack");
        return dst;
    }

    Q8 q8(const std::string & name, uint8_t * staging) {
        const gguf::Tensor & t = f_.get(name);
        Q8 m;
        m.n = (int) t.ne[0];
        m.rows = (int) t.ne[1];
        m.gtype = (int) t.type;
        if (t.type == gguf::Q8_0) {
            m.buf = repack(t.data, t.nbytes, 1, m.rows, m.n, staging);
        } else if (ik_gguf_type_supported((int) t.type)) {
            m.repacked = false;
            m.buf = dalloc<uint8_t>(t.nbytes);
            upload(m.buf, t.data, t.nbytes);
        } else {
            throw std::runtime_error(name + ": unsupported tensor type " + std::to_string(t.type));
        }
        return m;
    }

    // device bytes of one expert's matrices (repacked or GGUF), 256-aligned
    size_t gu_dev_bytes(const Experts & x) const {
        const size_t b = x.fast ? ik_matrix_bytes(x.type, 2 * n_ff_, E_) : x.gu_bytes;
        return (b + 255) / 256 * 256;
    }
    size_t dn_dev_bytes(const Experts & x) const {
        const size_t b = x.fast ? ik_matrix_bytes(x.type, E_, n_ff_) : x.dn_bytes;
        return (b + 255) / 256 * 256;
    }

    // expert e's GGUF blocks into the device matrices gu / dn (repacked on the fast path)
    void put_expert(const Experts & x, int e, uint8_t * gu, uint8_t * dn, uint8_t * staging) {
        if (x.fast) {
            ck(cudaMemcpyAsync(staging, x.gu_raw[e], x.gu_bytes, cudaMemcpyHostToDevice, st_), "upload");
            ck(ik_repack(x.type, staging, gu, 2 * n_ff_, E_, st_), "repack");
            ck(cudaMemcpyAsync(staging + x.gu_bytes, x.dn_raw[e], x.dn_bytes, cudaMemcpyHostToDevice, st_), "upload");
            ck(ik_repack(x.type, staging + x.gu_bytes, dn, E_, n_ff_, st_), "repack");
        } else {
            ck(cudaMemcpyAsync(gu, x.gu_raw[e], x.gu_bytes, cudaMemcpyHostToDevice, st_), "upload");
            ck(cudaMemcpyAsync(dn, x.dn_raw[e], x.dn_bytes, cudaMemcpyHostToDevice, st_), "upload");
        }
        ck(cudaStreamSynchronize(st_), "expert upload");
    }

    void load_weights() {
        const gguf::Tensor & out = f_.get("output.weight");
        uint8_t * staging = nullptr;
        ck(cudaMalloc(&staging, out.nbytes), "staging");  // the largest single matrix
        layers_.resize(n_layer_);
        for (int il = 0; il < n_layer_; ++il) {
            Layer & L = layers_[il];
            const std::string p = "blk." + std::to_string(il) + ".";
            L.swa = swa_[il] != 0;
            L.q = q8(p + "attn_q.weight", staging);
            L.k = q8(p + "attn_k.weight", staging);
            L.v = q8(p + "attn_v.weight", staging);
            L.gate = q8(p + "attn_gate.weight", staging);
            L.rel = q8(p + "attn_rel.weight", staging);
            L.out = q8(p + "attn_output.weight", staging);
            if (L.swa) {
                L.attn_norm = f32(p + "attn_norm.weight");
            } else {
                L.q_norm = f32(p + "attn_q_inp_norm.weight");
                L.k_norm = f32(p + "attn_k_inp_norm.weight");
                L.v_norm = f32(p + "attn_v_inp_norm.weight");
                L.kcp = f32(p + "attn_k_conv_proj.weight");
                L.vcp = f32(p + "attn_v_conv_proj.weight");
            }
            if (il > 0) L.res_query = f32(p + "attn_res_query.weight");
            L.ffn_res_query = f32(p + "ffn_res_query.weight");
            L.q_clip = f32(p + "attn_q_clip.weight");
            L.rel_clip = f32(p + "attn_rel_clip.weight");
            L.rel_bias = f32(p + "attn_rel_bias.weight");
            L.sink = f32(p + "attn_sink_proj.weight");
            L.k_conv = f32(p + "attn_k_conv.weight");
            L.v_conv = f32(p + "attn_v_conv.weight");
            L.post_conv = f32(p + "attn_post_conv.weight");
            L.ffn_post_conv = f32(p + "ffn_post_conv.weight");
            L.ffn_norm = f32(p + "ffn_norm.weight");
            L.router = f32(p + "ffn_gate_inp.weight");
            L.probs_b = f32(p + "exp_probs_b.bias", false);
            if (il >= core_start_ && il < core_end_) {
                L.msg_query = f32(p + "loop_msg_query.weight");
                const char * names[4] = {"q", "k", "v", "ffn"};
                for (int c = 0; c < 4; ++c) {
                    L.down[c] = f32(p + "loop_" + names[c] + "_ctrl_down.weight");
                    L.gain[c] = f32(p + "loop_" + names[c] + "_ctrl_gain.weight");
                    L.forget[c] = f32(p + "loop_" + names[c] + "_ctrl_forget.weight");
                }
            }
            // experts: one repacked matrix per expert, addressed through a pointer table
            const gguf::Tensor & gu = f_.get(p + "ffn_gate_up_exps.weight", {(uint64_t) E_, (uint64_t) 2 * n_ff_, (uint64_t) n_expert_});
            const gguf::Tensor & dn = f_.get(p + "ffn_down_exps.weight", {(uint64_t) n_ff_, (uint64_t) E_, (uint64_t) n_expert_});
            if (!ik_gguf_type_supported((int) gu.type) || !ik_gguf_type_supported((int) dn.type))
                throw std::runtime_error(p + "experts: unsupported tensor type");
            L.ex.gu_gtype = (int) gu.type;
            L.ex.dn_gtype = (int) dn.type;
            L.ex.fast = gu.type == dn.type && (gu.type == gguf::NVFP4 || gu.type == gguf::Q8_0);
            L.ex.type = gu.type == gguf::NVFP4 ? 0 : 1;
            const size_t gub = gu.nbytes / n_expert_, dnb = dn.nbytes / n_expert_;
            L.ex.gu_bytes = gub;
            L.ex.dn_bytes = dnb;
            for (int e = 0; e < n_expert_; ++e) {
                L.ex.gu_raw.push_back(gu.data + e * gub);
                L.ex.dn_raw.push_back(dn.data + e * dnb);
            }
            L.ex.gu = dalloc<void *>(n_expert_);
            L.ex.dn = dalloc<void *>(n_expert_);
            L.ex.gu_scale = f32(p + "ffn_gate_up_exps.scale", false);
            L.ex.dn_scale = f32(p + "ffn_down_exps.scale", false);
            if (offload_) {
                L.ex.h_gu_scale = host_f32(p + "ffn_gate_up_exps.scale", n_expert_);
                L.ex.h_dn_scale = host_f32(p + "ffn_down_exps.scale", n_expert_);
                L.ex.slot.assign(n_expert_, -1);
                ck(cudaMemset(L.ex.gu, 0, sizeof(void *) * n_expert_), "memset");
                ck(cudaMemset(L.ex.dn, 0, sizeof(void *) * n_expert_), "memset");
                continue;
            }
            // one allocation per layer: separate cudaMallocs round each matrix up to 2 MiB granules (measured: ~5 GB
            // lost over 6,144 matrices)
            const size_t gsz = gu_dev_bytes(L.ex), dsz = dn_dev_bytes(L.ex);
            uint8_t * arena = dalloc<uint8_t>((gsz + dsz) * n_expert_);
            std::vector<void *> gp(n_expert_), dp(n_expert_);
            for (int e = 0; e < n_expert_; ++e) {
                gp[e] = arena + (gsz + dsz) * e;
                dp[e] = arena + (gsz + dsz) * e + gsz;
                put_expert(L.ex, e, (uint8_t *) gp[e], (uint8_t *) dp[e], staging);
            }
            upload(L.ex.gu, gp.data(), sizeof(void *) * n_expert_);
            upload(L.ex.dn, dp.data(), sizeof(void *) * n_expert_);
        }
        out_ = q8("output.weight", staging);
        const gguf::Tensor & emb = f_.get("token_embd.weight");
        emb_type_ = (int) emb.type;
        if (emb_type_ != gguf::Q8_0 && emb_type_ != gguf::Q4_K && emb_type_ != gguf::Q5_K && emb_type_ != gguf::Q6_K)
            throw std::runtime_error("token_embd.weight: unsupported tensor type " + std::to_string(emb_type_));
        emb_rb_ = gguf::type_row_bytes(emb.type, E_);
        tok_norm_ = f32("token_embd_norm.weight");
        out_norm_ = f32("output_norm.weight");
        out_res_query_ = f32("output_res_query.weight");
        cudaFree(staging);
        if (offload_) {
            fill_cache();
            setup_zero_copy();
            setup_ram_tier();
            if (!ram_ && (!getenv("IK_PREFAULT") || atoi(getenv("IK_PREFAULT")))) prefault_experts();
            start_swapper();
        }
    }

    // Map the CPU-side experts' GGUF pages into the process now: the CPU reads experts through the mmap, and first touches
    // cost a page fault per 4 KB page (measured: decode 26 -> 41 tok/s once warm at a 12 GB budget). If RAM cannot hold
    // them, the page cache evicts some and those are read from the SSD when needed.
    void prefault_experts() {
        // only the experts the CPU computes (not in VRAM): the others would just take RAM from them
        std::vector<std::pair<const uint8_t *, size_t>> ranges;
        for (const Layer & L : layers_) {
            for (int e = 0; e < n_expert_; ++e) {
                if (L.ex.slot[e] >= 0) continue;
                ranges.push_back({L.ex.gu_raw[e], L.ex.gu_bytes});
                ranges.push_back({L.ex.dn_raw[e], L.ex.dn_bytes});
            }
        }
        const uintptr_t page = 4096;
#ifndef _WIN32
        for (auto & r : ranges) {
            const uintptr_t a = uintptr_t(r.first) & ~(page - 1);
            madvise((void *) a, uintptr_t(r.first) + r.second - a, MADV_WILLNEED);
        }
#endif
        const int nt = std::max(1, (int) std::thread::hardware_concurrency() / 2);
        std::atomic<size_t> next{0};
        std::atomic<uint64_t> sink{0};
        std::vector<std::thread> ts;
        for (int t = 0; t < nt; ++t) ts.emplace_back([&] {
            uint64_t acc = 0;
            for (size_t i; (i = next.fetch_add(1)) < ranges.size();) {
                const volatile uint8_t * p = ranges[i].first;
                for (size_t o = 0; o < ranges[i].second; o += page) acc += p[o];
            }
            sink += acc;
        });
        for (auto & t : ts) t.join();
    }

    // ------------------------------------------------------------ offload: cache

    std::vector<float> host_f32(const std::string & name, int n) {
        std::vector<float> v(n, 1.f);
        auto it = f_.tensors.find(name);
        if (it != f_.tensors.end()) memcpy(v.data(), it->second.data, sizeof(float) * n);
        return v;
    }

    void load_profile(const char * path) {
        if (!path || !*path) return;
        FILE * fp = fopen(path, "rb");
        if (!fp) throw std::runtime_error(std::string("cannot open profile ") + path);
        profile_.resize(size_t(n_layer_) * n_expert_);
        const size_t got = fread(profile_.data(), sizeof(float), profile_.size(), fp);
        fclose(fp);
        if (got != profile_.size()) throw std::runtime_error("profile size does not match the model");
    }

    // what a CPU-side read of one of this layer's experts costs relative to NVFP4 / Q8_0 (AVX2): Q5_0, IQ4_NL, IQ4_XS,
    // Q4_K and Q5_K now also run AVX2 on CPU; only types still on the scalar path get weighted higher
    float cpu_cost(int il) const {
        const auto avx2 = [](int t) {
            return t == gguf::NVFP4 || t == gguf::Q8_0 || t == gguf::Q5_0 || t == gguf::IQ4_NL ||
                   t == gguf::IQ4_XS || t == gguf::Q4_K || t == gguf::Q5_K ||
                   t == gguf::IQ2_XXS || t == gguf::IQ2_XS || t == gguf::IQ3_XXS || t == gguf::IQ2_S;
        };
        return (avx2(layers_[il].ex.gu_gtype) && avx2(layers_[il].ex.dn_gtype)) ? 1.f : 8.f;
    }

    // reads per token of each (layer, expert): the profile, else the loop structure alone (a core expert is read 3x)
    float prior(int il, int e) const {
        if (!profile_.empty()) return profile_[size_t(il) * n_expert_ + e];
        return (il >= core_start_ && il < core_end_ ? float(n_passes_) : 1.f) * n_used_ / n_expert_;
    }

    // place expert e of layer il into slot `slot` of its pool (upload GGUF blocks, repack, point the tables at it)
    void place(int il, int e, int slot) {
        Layer & L = layers_[il];
        Pool & P = pools_[L.ex.pool];
        uint8_t * gu = P.base + size_t(slot) * P.slot_bytes;
        uint8_t * dn = gu + P.gu_bytes;
        put_expert(L.ex, e, gu, dn, swap_staging_);
        void * pg = gu, * pd = dn;
        ck(cudaMemcpyAsync(L.ex.gu + e, &pg, sizeof(void *), cudaMemcpyHostToDevice, st_), "table");
        ck(cudaMemcpyAsync(L.ex.dn + e, &pd, sizeof(void *), cudaMemcpyHostToDevice, st_), "table");
        ck(cudaStreamSynchronize(st_), "place");
        L.ex.slot[e] = slot;
        P.owner[slot] = {il, e};
    }

    // touch every page of [p, p + n) so the CPU's later reads do not fault
    static void touch(const uint8_t * p, size_t n) {
        const uintptr_t page = 4096;
#ifndef _WIN32
        const uintptr_t a = uintptr_t(p) & ~(page - 1);
        madvise((void *) a, uintptr_t(p) + n - a, MADV_WILLNEED);
#endif
        uint64_t acc = 0;
        for (size_t o = 0; o < n; o += page) acc += ((const volatile uint8_t *) p)[o];
        static std::atomic<uint64_t> sink{0};
        sink += acc;
    }

    void evict(int il, int e, bool fault_in = true) {
        Layer & L = layers_[il];
        void * z = nullptr;
        ck(cudaMemcpyAsync(L.ex.gu + e, &z, sizeof(void *), cudaMemcpyHostToDevice, st_), "table");
        ck(cudaMemcpyAsync(L.ex.dn + e, &z, sizeof(void *), cudaMemcpyHostToDevice, st_), "table");
        ck(cudaStreamSynchronize(st_), "table");
        L.ex.slot[e] = -1;
        if (fault_in && !ram_) {
            touch(L.ex.gu_raw[e], L.ex.gu_bytes);  // the CPU computes it from now on
            touch(L.ex.dn_raw[e], L.ex.dn_bytes);
        }
    }

    // ------------------------------------------------------------ offload: SSD tier

    void setup_ram_tier() {
        const long long mib = getenv("IK_RAM_MIB") ? atoll(getenv("IK_RAM_MIB")) : 0;
        if (mib <= 0) return;
        constexpr size_t A = 4096;
        size_t gu = 0, dn = 0;
        for (const Layer & L : layers_) { gu = std::max(gu, L.ex.gu_bytes); dn = std::max(dn, L.ex.dn_bytes); }
        ram_gu_area_ = (gu + 2 * A + A - 1) / A * A;
        ram_slot_bytes_ = ram_gu_area_ + (dn + 2 * A + A - 1) / A * A;
        const int n = (int) std::max<long long>(16, (mib << 20) / (long long) ram_slot_bytes_);
#ifdef _WIN32
        ram_arena_ = (uint8_t *) _aligned_malloc(ram_slot_bytes_ * n, A);
        if (!ram_arena_) throw std::runtime_error("SSD tier: cannot allocate its RAM cache");
        ram_fd_ = mk_open(path_.c_str(), O_RDONLY);
        ram_direct_ = false;
#else
        if (posix_memalign((void **) &ram_arena_, A, ram_slot_bytes_ * n) != 0) throw std::runtime_error("SSD tier: cannot allocate its RAM cache");
        ram_fd_ = open(path_.c_str(), O_RDONLY | O_DIRECT);
        ram_direct_ = ram_fd_ >= 0;
        if (ram_fd_ < 0) ram_fd_ = open(path_.c_str(), O_RDONLY);
#endif
        if (ram_fd_ < 0) throw std::runtime_error("SSD tier: cannot open " + path_);
        ram_slot_of_.assign(size_t(n_layer_) * n_expert_, -1);
        ram_owner_.assign(n, -1);
        ram_stamp_.assign(n, 0);
        ram_loop_pin_.assign(n, 0);
        ram_ = true;
        fprintf(stderr, "inkling: SSD tier: %d experts in %lld MiB of RAM, reads %s\n", n, (long long) (ram_slot_bytes_ * n) >> 20,
                ram_direct_ ? "O_DIRECT" : "through the page cache");
    }

    // the CPU-side experts of one layer application, in RAM: hits refresh their stamp, misses take the least recently
    // used slots not needed by this application (preferring unpinned slots over loop-pinned pass-1 experts) and are
    // read in parallel; writes the matrices' addresses. loop_pass: 1 = pin for passes 2-3, 3 = unpin, 0 = non-core.
    void ram_fetch(const Layer & L, int il, int loop_pass, const int * es, int n, const uint8_t ** gu, const uint8_t ** dn) {
        std::vector<int> slots(n), todo;
        for (int i = 0; i < n; ++i) {
            const size_t key = size_t(il) * n_expert_ + es[i];
            int s = ram_slot_of_[key];
            if (s < 0) {
                uint64_t best = UINT64_MAX;
                for (int pass = 0; pass < 2 && s < 0; ++pass) {
                    for (int j = 0; j < (int) ram_owner_.size(); ++j) {  // (a few hundred slots: a scan is cheap next to a read)
                        if (pass == 0 && ram_loop_pin_[j]) continue;     // keep pass-1 core experts alive for passes 2 & 3
                        bool used = false;
                        for (int q = 0; q < i; ++q) used |= slots[q] == j;
                        if (!used && ram_stamp_[j] < best) { best = ram_stamp_[j]; s = j; }
                    }
                }
                if (ram_owner_[s] >= 0) ram_slot_of_[ram_owner_[s]] = -1;
                ram_owner_[s] = (int) key;
                ram_slot_of_[key] = s;
                ram_loop_pin_[s] = 0;
                todo.push_back(i);
            }
            slots[i] = s;
            ram_stamp_[s] = ++ram_clock_;
            if (loop_pass == 1) ram_loop_pin_[s] = 1;
            else if (loop_pass == n_passes_) ram_loop_pin_[s] = 0;
        }
        if (!todo.empty()) {
            const auto t0 = std::chrono::steady_clock::now();
            // every expert's two block-aligned ranges, cut into 512 KiB reads spread over the worker threads: a single
            // stream leaves the drive's queue mostly idle (measured 0.8 GB/s vs 1.7 GB/s with 8 readers on WSL)
            constexpr size_t A = 4096, PIECE = size_t(512) << 10;
            struct Rd { uint8_t * dst; size_t off, len; };
            std::vector<Rd> rds;
            for (int i : todo) {
                const int e = es[i];
                uint8_t * base = ram_arena_ + size_t(slots[i]) * ram_slot_bytes_;
                for (auto r : {std::make_pair(L.ex.gu_raw[e], std::make_pair(L.ex.gu_bytes, base)),
                               std::make_pair(L.ex.dn_raw[e], std::make_pair(L.ex.dn_bytes, base + ram_gu_area_))}) {
                    const size_t off = size_t(r.first - f_.base), a0 = off / A * A, a1 = (off + r.second.first + A - 1) / A * A;
                    for (size_t o = a0; o < a1; o += PIECE) rds.push_back({r.second.second + (o - a0), o, std::min(PIECE, a1 - o)});
                }
            }
            std::atomic<bool> fallback{false};
            pool_->parallel_for((int) rds.size(), [&](int j) {
                const Rd & r = rds[j];
                size_t done = 0;
                while (done < r.len) {
                    const ssize_t got = pread(ram_fd_, r.dst + done, r.len - done, (off_t) (r.off + done));
                    if (got <= 0) { fallback = true; return; }
                    done += (size_t) got;
                }
            });
            if (fallback) {  // a read failed (O_DIRECT refused at run time, say): copy those experts from the mmap
                for (int i : todo) {
                    const int e = es[i];
                    uint8_t * base = ram_arena_ + size_t(slots[i]) * ram_slot_bytes_;
                    memcpy(base + size_t(L.ex.gu_raw[e] - f_.base) % A, L.ex.gu_raw[e], L.ex.gu_bytes);
                    memcpy(base + ram_gu_area_ + size_t(L.ex.dn_raw[e] - f_.base) % A, L.ex.dn_raw[e], L.ex.dn_bytes);
                }
            }
            ssd_reads_ += (long long) todo.size();
            for (int i : todo) ssd_bytes_ += (long long) (L.ex.gu_bytes + L.ex.dn_bytes);
            ssd_ns_ += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count();
        }
        constexpr size_t A = 4096;
        for (int i = 0; i < n; ++i) {  // the in-slot offset is the file offset's remainder
            const int e = es[i];
            const uint8_t * base = ram_arena_ + size_t(slots[i]) * ram_slot_bytes_;
            gu[i] = base + size_t(L.ex.gu_raw[e] - f_.base) % A;
            dn[i] = base + ram_gu_area_ + size_t(L.ex.dn_raw[e] - f_.base) % A;
        }
    }

    // ------------------------------------------------------------ offload: zero-copy share of the misses

    // copy the hottest CPU-side experts (up to IK_ZC_MIB) into pinned, mapped host memory: the GPU reads IK_ZC_SHARE of
    // each layer's missing experts from there over PCIe while the CPU computes the rest (from the same copies, which
    // never page-fault). Pinning the mmap'd file directly (cudaHostRegister) is refused on some systems (WSL).
    void setup_zero_copy() {
        const char * cp_env = getenv("IK_COPY");
        const bool cp_disabled = cp_env && atoi(cp_env) == 0;
        cp_ = cp_env && atoi(cp_env) != 0;
        if (getenv("IK_COPY_FRAC")) cp_frac_ = (float) atof(getenv("IK_COPY_FRAC"));
        cp_stage_on_ = getenv("IK_COPY_STAGE") && atoi(getenv("IK_COPY_STAGE"));
        for (Layer & L : layers_) {
            L.ex.zc_gu = dalloc<void *>(n_expert_);
            L.ex.zc_dn = dalloc<void *>(n_expert_);
            ck(cudaMemset(L.ex.zc_gu, 0, sizeof(void *) * n_expert_), "memset");
            ck(cudaMemset(L.ex.zc_dn, 0, sizeof(void *) * n_expert_), "memset");
            L.ex.zc_ok.assign(n_expert_, 0);
        }
        // Native Linux auto-pin: try cudaHostRegister directly on the mmap'd GGUF expert tensors (0 extra RAM!).
        // WSL2 returns cudaErrorNotSupported on the probe and skips cleanly.
        if (!cp_disabled && !(getenv("IK_RAM_MIB") && atoll(getenv("IK_RAM_MIB")) > 0)) {
            const uintptr_t page = 4096;
            size_t n_reg = 0;
            bool probe_ok = true;
            for (int il = 0; il < n_layer_ && probe_ok; ++il) {
                Experts & X = layers_[il].ex;
                bool any_cpu = false;
                for (int e = 0; e < n_expert_; ++e) if (X.slot[e] < 0) { any_cpu = true; break; }
                if (!any_cpu) continue;
                const uint8_t * g0 = X.gu_raw[0], * d0 = X.dn_raw[0];
                const size_t gtot = X.gu_bytes * n_expert_, dtot = X.dn_bytes * n_expert_;
                auto try_reg = [&](const uint8_t * p, size_t len) -> bool {
                    const uintptr_t a0 = uintptr_t(p) & ~(page - 1);
                    const size_t alen = ((uintptr_t(p) + len - a0) + page - 1) & ~(page - 1);
                    const cudaError_t err = cudaHostRegister((void *) a0, alen, cudaHostRegisterReadOnly | cudaHostRegisterMapped);
                    if (err != cudaSuccess) { cudaGetLastError(); return false; }
                    mmap_pinned_.push_back((void *) a0);
                    return true;
                };
                if (!try_reg(g0, gtot) || !try_reg(d0, dtot)) {
                    probe_ok = false;
                    break;
                }
                for (int e = 0; e < n_expert_; ++e) {
                    if (X.slot[e] < 0) { X.zc_ok[e] = 1; n_reg++; }
                }
            }
            if (n_reg > 0) {
                size_t gu = 0, dn = 0;
                for (const Layer & L : layers_) { gu = std::max(gu, L.ex.gu_bytes); dn = std::max(dn, L.ex.dn_bytes); }
                cp_gu_area_ = (gu + 255) / 256 * 256;
                cp_slot_ = cp_gu_area_ + (dn + 255) / 256 * 256;
                cp_ = true;
                fprintf(stderr, "inkling: native mmap pin: %zu CPU-side experts registered in place for PCIe copy assist\n", n_reg);
                return;
            }
            for (void * p : mmap_pinned_) cudaHostUnregister(p);
            mmap_pinned_.clear();
        }
        if (cp_) {  // the staging buffer first: pinned memory is scarce on some systems (about 1 GiB under WSL)
            size_t gu = 0, dn = 0;
            for (const Layer & L : layers_) { gu = std::max(gu, L.ex.gu_bytes); dn = std::max(dn, L.ex.dn_bytes); }
            cp_gu_area_ = (gu + 255) / 256 * 256;
            cp_slot_ = cp_gu_area_ + (dn + 255) / 256 * 256;
            if (cudaHostAlloc((void **) &cp_stage_, cp_slot_ * n_used_, cudaHostAllocDefault) != cudaSuccess) {
                cudaGetLastError();
                cp_stage_ = nullptr;
                cp_ = false;
                fprintf(stderr, "inkling: copy path off: no pinned memory for its staging buffer\n");
            }
        }
        // in copy mode the arena is the copy path's pinned source (default 768 MiB); otherwise the GPU reads it directly
        const long long mib = getenv("IK_ZC_MIB") ? atoll(getenv("IK_ZC_MIB")) : (cp_ ? 768 : 0);
        zc_share_ = cp_ ? 1.f : getenv("IK_ZC_SHARE") ? (float) atof(getenv("IK_ZC_SHARE")) : 0.5f;
        if (mib <= 0 || zc_share_ <= 0.f) return;
        struct C { float v; int il, e; };
        std::vector<C> c;
        for (int il = 0; il < n_layer_; ++il)
            for (int e = 0; e < n_expert_; ++e)
                if (layers_[il].ex.slot[e] < 0) c.push_back({prior(il, e) * cpu_cost(il), il, e});
        std::sort(c.begin(), c.end(), [](const C & a, const C & b) { return a.v > b.v; });
        long long left = mib << 20;
        std::vector<C> chosen;
        size_t total = 0;
        for (const C & x : c) {
            const Experts & X = layers_[x.il].ex;
            const long long b = (long long) ((X.gu_bytes + 255) / 256 * 256 + (X.dn_bytes + 255) / 256 * 256);
            if (b > left) continue;
            left -= b;
            total += (size_t) b;
            chosen.push_back(x);
        }
        if (chosen.empty()) return;
        // pinned in chunks of 64 MiB (large pinned allocations fail on some systems, WSL among them)
        constexpr size_t CHUNK = size_t(64) << 20;
        size_t used = CHUNK, pinned = 0;
        uint8_t * cur = nullptr;
        size_t n_done = 0;
        for (const C & x : chosen) {
            Experts & X = layers_[x.il].ex;
            const size_t gb = (X.gu_bytes + 255) / 256 * 256, db = (X.dn_bytes + 255) / 256 * 256;
            if (used + gb + db > CHUNK) {
                uint8_t * p = nullptr;
                const cudaError_t err = cudaHostAlloc((void **) &p, CHUNK, cudaHostAllocMapped);
                if (err != cudaSuccess) {
                    cudaGetLastError();
                    fprintf(stderr, "inkling: pinned expert arena stops at %zu MiB (%s)\n", pinned >> 20, cudaGetErrorString(err));
                    break;
                }
                zc_chunks_.push_back(p);
                cur = p;
                used = 0;
                pinned += CHUNK;
            }
            uint8_t * g = cur + used;
            uint8_t * d = g + gb;
            used += gb + db;
            memcpy(g, X.gu_raw[x.e], X.gu_bytes);
            memcpy(d, X.dn_raw[x.e], X.dn_bytes);
            X.gu_raw[x.e] = g;  // the CPU reads the pinned copy too
            X.dn_raw[x.e] = d;
            void * dg = nullptr, * dd = nullptr;
            ck(cudaHostGetDevicePointer(&dg, g, 0), "zc pointer");
            ck(cudaHostGetDevicePointer(&dd, d, 0), "zc pointer");
            upload(X.zc_gu + x.e, &dg, sizeof(void *));
            upload(X.zc_dn + x.e, &dd, sizeof(void *));
            X.zc_ok[x.e] = 1;
            n_done++;
        }
        if (!n_done) return;
        total = pinned;
        chosen.resize(n_done);
        zc_ = !cp_;
        if (cp_) fprintf(stderr, "inkling: copy path: %zu CPU-side experts in %zu MiB of pinned memory\n", chosen.size(), total >> 20);
        else fprintf(stderr, "inkling: zero-copy share: %zu experts in %zu MiB of pinned memory, GPU share of their misses %.2f\n",
                     chosen.size(), total >> 20, zc_share_);
    }

    // ------------------------------------------------------------ offload: asynchronous swaps

    void start_swapper() {
        swap_async_ = !getenv("IK_SWAP_SYNC");
        if (!swap_async_) return;
        size_t most = 0;
        for (const Pool & P : pools_) most = std::max(most, P.slot_bytes);
        ck(cudaStreamCreateWithFlags(&copy_st_, cudaStreamNonBlocking), "copy stream");
        for (int b = 0; b < 2; ++b) {
            ck(cudaHostAlloc((void **) &swap_pinned_[b], most, cudaHostAllocDefault), "pinned");
            swap_dev_[b] = dalloc<uint8_t>(most);
            ck(cudaEventCreateWithFlags(&swap_free_[b], cudaEventDisableTiming), "event");
            ck(cudaEventCreateWithFlags(&swap_landed_[b], cudaEventDisableTiming), "event");
            ck(cudaEventRecord(swap_free_[b], copy_st_), "event");
        }
        swap_busy_.assign(size_t(n_layer_) * n_expert_, 0);
        swapper_ = std::thread([this] { swap_loop(); });
    }

    void swap_loop() {
        int b = 0;
        for (;;) {
            SwapJob j;
            {
                std::unique_lock<std::mutex> lk(swap_mu_);
                swap_cv_.wait(lk, [this] { return stop_ || !swap_queue_.empty(); });
                if (stop_) return;
                j = swap_queue_.front();
                swap_queue_.pop_front();
            }
            const Experts & x = layers_[j.il].ex;
            if (j.slot < 0) {  // an evicted expert: fault its pages in for the CPU (the SSD tier reads its own)
                if (ram_) continue;
                touch(x.gu_raw[j.e], x.gu_bytes);
                touch(x.dn_raw[j.e], x.dn_bytes);
                continue;
            }
            const Pool & P = pools_[x.pool];
            uint8_t * gu = P.base + size_t(j.slot) * P.slot_bytes;
            uint8_t * dn = gu + P.gu_bytes;
            ck(cudaEventSynchronize(swap_free_[b]), "swap buffer");  // the previous copy from this buffer has landed
            memcpy(swap_pinned_[b], x.gu_raw[j.e], x.gu_bytes);
            memcpy(swap_pinned_[b] + x.gu_bytes, x.dn_raw[j.e], x.dn_bytes);
            // the copies on the copy stream; the repack kernels on the decode stream, where they run between tokens. (Under
            // WSL a kernel on another stream cannot start while the decode graph waits on the host, and copies queued
            // behind it then stall too, which deadlocks the copy path's wait.)
            // (only with the copy path on: otherwise the repack overlaps the decode on the copy stream)
            cudaStream_t done_st = copy_st_;
            if (x.fast) {
                cudaStream_t rp = cp_ ? st_ : copy_st_;
                ck(cudaMemcpyAsync(swap_dev_[b], swap_pinned_[b], x.gu_bytes + x.dn_bytes, cudaMemcpyHostToDevice, copy_st_), "swap copy");
                if (cp_) {
                    ck(cudaEventRecord(swap_landed_[b], copy_st_), "event");
                    ck(cudaStreamWaitEvent(st_, swap_landed_[b], 0), "event");
                }
                ck(ik_repack(x.type, swap_dev_[b], gu, 2 * n_ff_, E_, rp), "swap repack");
                ck(ik_repack(x.type, swap_dev_[b] + x.gu_bytes, dn, E_, n_ff_, rp), "swap repack");
                done_st = rp;
            } else {
                ck(cudaMemcpyAsync(gu, swap_pinned_[b], x.gu_bytes, cudaMemcpyHostToDevice, copy_st_), "swap copy");
                ck(cudaMemcpyAsync(dn, swap_pinned_[b] + x.gu_bytes, x.dn_bytes, cudaMemcpyHostToDevice, copy_st_), "swap copy");
            }
            ck(cudaEventRecord(swap_free_[b], done_st), "event");
            ck(cudaEventCreateWithFlags(&j.done, cudaEventDisableTiming), "event");
            ck(cudaEventRecord(j.done, done_st), "event");
            {
                std::lock_guard<std::mutex> lk(swap_mu_);
                swap_flight_.push_back(j);
            }
            b ^= 1;
        }
    }

    // point the tables at the experts whose copies have landed (between tokens, on the decode thread)
    void commit_swaps() {
        std::vector<SwapJob> ready;
        {
            std::lock_guard<std::mutex> lk(swap_mu_);
            for (size_t i = 0; i < swap_flight_.size();) {
                if (cudaEventQuery(swap_flight_[i].done) == cudaSuccess) {
                    ready.push_back(swap_flight_[i]);
                    swap_flight_[i] = swap_flight_.back();
                    swap_flight_.pop_back();
                } else {
                    ++i;
                }
            }
        }
        if (ready.empty()) return;
        for (const SwapJob & j : ready) {
            Layer & L = layers_[j.il];
            Pool & P = pools_[L.ex.pool];
            void * pg = P.base + size_t(j.slot) * P.slot_bytes;
            void * pd = (uint8_t *) pg + P.gu_bytes;
            ck(cudaMemcpyAsync(L.ex.gu + j.e, &pg, sizeof(void *), cudaMemcpyHostToDevice, st_), "table");
            ck(cudaMemcpyAsync(L.ex.dn + j.e, &pd, sizeof(void *), cudaMemcpyHostToDevice, st_), "table");
            ck(cudaStreamSynchronize(st_), "table");
            L.ex.slot[j.e] = j.slot;
            P.owner[j.slot] = {j.il, j.e};
            swap_busy_[size_t(j.il) * n_expert_ + j.e] = 0;
            cudaEventDestroy(j.done);
        }
    }

    // initial resident set: the experts with the most reads per byte, until the budget is spent
    void fill_cache() {
        for (Layer & L : layers_) {  // a pool per expert format
            const Experts & x = L.ex;
            int found = -1;
            for (size_t i = 0; i < pools_.size(); ++i) {
                const Pool & P = pools_[i];
                if (P.fast == x.fast && (x.fast ? P.type == x.type : (P.gu_gtype == x.gu_gtype && P.dn_gtype == x.dn_gtype))) found = (int) i;
            }
            if (found < 0) {
                Pool P{};
                P.fast = x.fast; P.type = x.type; P.gu_gtype = x.gu_gtype; P.dn_gtype = x.dn_gtype;
                P.gu_bytes = gu_dev_bytes(x);
                P.dn_bytes = dn_dev_bytes(x);
                P.slot_bytes = P.gu_bytes + P.dn_bytes;
                pools_.push_back(P);
                found = (int) pools_.size() - 1;
            }
            L.ex.pool = found;
        }
        struct Cand { float v; int il, e; };
        std::vector<Cand> c;
        for (int il = 0; il < n_layer_; ++il)
            for (int e = 0; e < n_expert_; ++e) c.push_back({prior(il, e) * cpu_cost(il) / float(pools_[layers_[il].ex.pool].slot_bytes), il, e});
        std::sort(c.begin(), c.end(), [](const Cand & a, const Cand & b) { return a.v > b.v; });
        long long left = budget_;
        std::vector<int> n_slots(pools_.size(), 0), used(pools_.size(), 0);
        std::vector<Cand> chosen;
        for (const Cand & x : c) {
            const size_t b = pools_[layers_[x.il].ex.pool].slot_bytes;
            if ((long long) b > left) continue;
            left -= (long long) b;
            n_slots[layers_[x.il].ex.pool]++;
            chosen.push_back(x);
        }
        size_t staging = 0;
        for (size_t t = 0; t < pools_.size(); ++t) {
            if (n_slots[t]) pools_[t].base = dalloc<uint8_t>(size_t(n_slots[t]) * pools_[t].slot_bytes);
            pools_[t].owner.assign(n_slots[t], {-1, -1});
            staging = std::max(staging, pools_[t].slot_bytes);
        }
        swap_staging_ = dalloc<uint8_t>(staging);
        for (const Cand & x : chosen) place(x.il, x.e, used[layers_[x.il].ex.pool]++);
        score_.assign(size_t(n_layer_) * n_expert_, 0.f);
        for (int il = 0; il < n_layer_; ++il)
            for (int e = 0; e < n_expert_; ++e) score_[size_t(il) * n_expert_ + e] = prior(il, e) * 16.f;  // prior worth ~16 tokens
        reads_.assign(size_t(n_layer_) * n_expert_, 0.f);
    }

    // between tokens: fold recent reads into decayed scores; every few tokens swap in experts that clearly
    // out-score the coldest resident expert of their pool
    void maintain() {
        static const int every = getenv("IK_SWAP_EVERY") ? atoi(getenv("IK_SWAP_EVERY")) : 8;  // measured best: 4 swaps every 8 tokens
        static const int max_swaps = getenv("IK_SWAP_MAX") ? atoi(getenv("IK_SWAP_MAX")) : 4;
        if (swap_async_) commit_swaps();
        if (every <= 0 || max_swaps <= 0 || pos_ == 0 || pos_ - last_maintain_pos_ < every) return;
        last_maintain_pos_ = pos_;
        std::atomic_thread_fence(std::memory_order_acquire);
        const float decay = std::pow(0.97f, float(every));
        for (size_t i = 0; i < score_.size(); ++i) { score_[i] = score_[i] * decay + reads_[i]; reads_[i] = 0.f; }
        for (size_t t = 0; t < pools_.size(); ++t) {
            Pool & P = pools_[t];
            if (P.owner.empty()) continue;
            struct Item { float s; int il, e; };
            std::vector<Item> out, in;
            for (int il = 0; il < n_layer_; ++il) {
                if (layers_[il].ex.pool != (int) t) continue;
                for (int e = 0; e < n_expert_; ++e) {
                    if (swap_async_ && swap_busy_[size_t(il) * n_expert_ + e]) continue;  // on its way in
                    const float sc = score_[size_t(il) * n_expert_ + e] * cpu_cost(il);
                    (layers_[il].ex.slot[e] < 0 ? out : in).push_back({sc, il, e});
                }
            }
            const int k = std::min<int>(max_swaps, (int) std::min(out.size(), in.size()));
            std::partial_sort(out.begin(), out.begin() + k, out.end(), [](const Item & a, const Item & b) { return a.s > b.s; });
            std::partial_sort(in.begin(), in.begin() + k, in.end(), [](const Item & a, const Item & b) { return a.s < b.s; });
            for (int j = 0; j < k; ++j) {
                if (out[j].s <= 1.5f * in[j].s + 1e-6f) break;
                const int slot = layers_[in[j].il].ex.slot[in[j].e];
                if (swap_async_) {
                    evict(in[j].il, in[j].e, false);  // the swap thread faults its pages in for the CPU
                    P.owner[slot] = {-2, -2};          // reserved until the incoming expert lands
                    swap_busy_[size_t(out[j].il) * n_expert_ + out[j].e] = 1;
                    {
                        std::lock_guard<std::mutex> lk(swap_mu_);
                        swap_queue_.push_back({out[j].il, out[j].e, slot, nullptr});
                        swap_queue_.push_back({in[j].il, in[j].e, -1, nullptr});  // slot -1: fault in the evicted one
                    }
                    swap_cv_.notify_one();
                } else {
                    evict(in[j].il, in[j].e);
                    place(out[j].il, out[j].e, slot);
                }
                n_swaps_++;
            }
        }
    }

    // ------------------------------------------------------------ offload: CPU side

    void start_offload(int n_threads) {
        const int n_apps = n_layer_ + (n_passes_ - 1) * (core_end_ - core_start_);
        const size_t per = 2 * 64 + 2 * 64 + sizeof(float) * (n_used_ * 2 + 2 * E_) + 256;  // flags, ids, wts, x, y
        ck(cudaHostAlloc((void **) &h_x_, per * n_apps, cudaHostAllocMapped), "mapped");
        memset(h_x_, 0, per * n_apps);
        hslots_.resize(n_apps);
        for (int a = 0; a < n_apps; ++a) {
            uint8_t * b = (uint8_t *) h_x_ + per * a;
            HSlot & h = hslots_[a];
            h.h_ready = (volatile int *) b;
            h.h_done = (volatile int *) (b + 64);
            h.h_ids = (int32_t *) (b + 128);
            h.h_wts = (float *) (b + 256);
            h.h_x = (float *) (b + 384);
            h.h_y = h.h_x + E_;
            auto dev = [](void * p) { void * d = nullptr; ck(cudaHostGetDevicePointer(&d, p, 0), "mapped pointer"); return d; };
            h.dev.ready = (volatile int *) dev((void *) h.h_ready);
            h.dev.done = (volatile int *) dev((void *) h.h_done);
            h.dev.ids = (int32_t *) dev(h.h_ids);
            h.dev.wts = (float *) dev(h.h_wts);
            h.dev.x = (float *) dev(h.h_x);
            h.dev.y = (volatile float *) dev(h.h_y);
        }
        // layer of each application, in execution order
        std::vector<int> order;
        for (int il = 0; il < core_start_; ++il) order.push_back(il);
        for (int ps = 0; ps < n_passes_; ++ps) for (int il = core_start_; il < core_end_; ++il) order.push_back(il);
        for (int il = core_end_; il < n_layer_; ++il) order.push_back(il);
        for (int a = 0; a < n_apps; ++a) hslots_[a].layer = order[a];
        if (n_threads <= 0) n_threads = std::max(1, (int) std::thread::hardware_concurrency() / 2);
        if (cp_) {
            cp_dev_ = dalloc<uint8_t>(cp_slot_ * n_used_);
            static_assert(sizeof(CopyFlag) == 64, "CopyFlag");
            ck(cudaHostAlloc((void **) &cp_host_, sizeof(CopyFlag) * n_apps, cudaHostAllocMapped), "mapped");
            memset((void *) cp_host_, 0, sizeof(CopyFlag) * n_apps);
            CopyFlag * dev = nullptr;
            ck(cudaHostGetDevicePointer((void **) &dev, cp_host_, 0), "mapped pointer");
            cp_flag_ = (int *) &dev[0].flag;
            cp_sel_ = &dev[0].sel;
            ck(cudaStreamCreateWithFlags(&cp_st_, cudaStreamNonBlocking), "copy stream");
            cp_thread_ = std::thread([this] { copy_loop(); });
        }
        pool_ = std::make_unique<SpinPool>(n_threads - 1);
        coord_ = std::thread([this] { coordinate(); });
    }

    // the copy thread: sends a layer application's copied experts to the scratch slots, then sets its flag
    void copy_loop() {
        uint64_t seen = 0;
        constexpr size_t PIECE = size_t(1) << 20;
        while (!stop_) {
            int spins = 0;
            while (cp_seq_.load(std::memory_order_acquire) == seen) {
                if (stop_) return;
                if (++spins < 50000) _mm_pause();
                else std::this_thread::yield();
            }
            seen = cp_seq_.load(std::memory_order_acquire);
            const CopyJob & J = cp_job_;
            const auto t0 = std::chrono::steady_clock::now();
            size_t staged = 0, pinned = 0;
            for (int i = 0; i < J.n; ++i) {
                for (int part = 0; part < 2; ++part) {
                    const uint8_t * src = part ? J.d[i] : J.g[i];
                    const size_t n = part ? J.db : J.gb, off = size_t(i) * cp_slot_ + (part ? cp_gu_area_ : 0);
                    if (J.pinned[i]) {
                        ck(cudaMemcpyAsync(cp_dev_ + off, src, n, cudaMemcpyHostToDevice, cp_st_), "copy path");
                        pinned += n;
                        continue;
                    }
                    for (size_t o = 0; o < n; o += PIECE) {  // staged in pieces, each sent as soon as it is in
                        const size_t len = std::min(PIECE, n - o);
                        const auto m0 = std::chrono::steady_clock::now();
                        memcpy(cp_stage_ + off + o, src + o, len);
                        dbg_stage_ns_ += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - m0).count();
                        ck(cudaMemcpyAsync(cp_dev_ + off + o, cp_stage_ + off + o, len, cudaMemcpyHostToDevice, cp_st_), "copy path");
                    }
                    staged += n;
                }
            }
            const auto s0 = std::chrono::steady_clock::now();
            ck(cudaStreamSynchronize(cp_st_), "copy path");
            const auto s1 = std::chrono::steady_clock::now();
            if (J.n) {
                dbg_jobs_++;
                dbg_issue_ns_ += std::chrono::duration_cast<std::chrono::nanoseconds>(s0 - t0).count();
                dbg_sync_ns_ += std::chrono::duration_cast<std::chrono::nanoseconds>(s1 - s0).count();
            }
            CopyFlag & F = cp_host_[J.app];
            F.sel.n = J.n;
            for (int i = 0; i < J.n; ++i) F.sel.k[i] = J.k[i];
            std::atomic_thread_fence(std::memory_order_release);
            F.flag = J.value;
            if (J.n) {
                const double ns = (double) std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count();
                n_copy_ += J.n;
                cp_ns_ += (long long) ns;
                if (staged && !pinned) stage_ns_b_.store(0.95 * stage_ns_b_.load() + 0.05 * ns / double(staged));
                if (pinned && !staged) pin_ns_b_.store(0.95 * pin_ns_b_.load() + 0.05 * ns / double(pinned));
            }
            cp_idle_.store(true, std::memory_order_release);
        }
    }

    // the CPU half of every layer application, in execution order
    void coordinate() {
        std::vector<float> h(size_t(n_used_) * n_ff_);
        std::vector<float> x_loc(E_);
        int a = 0;
        while (!stop_) {
            HSlot & hs = hslots_[a];
            int spins = 0;
            int ready;
            while ((ready = *hs.h_ready) == *hs.h_done || ready == 0) {
                if (stop_) return;
                if (++spins < 50000) _mm_pause();
                else std::this_thread::yield();
            }
            std::atomic_thread_fence(std::memory_order_acquire);
            const Layer & L = layers_[hs.layer];
            int miss[8], nm = 0, m_all = 0;
            for (int k = 0; k < n_used_; ++k) {
                const int e = hs.h_ids[k];
                reads_[size_t(hs.layer) * n_expert_ + e] += 1.f;
                if (L.ex.slot[e] < 0) {
                    // the zero-copy rule of zc_select_kernel: the GPU reads this one over PCIe
                    const bool gpu = zc_ && L.ex.zc_ok[e] && std::floor((m_all + 1) * zc_share_) > std::floor(m_all * zc_share_);
                    m_all++;
                    if (!gpu) miss[nm++] = k;
                }
            }
            n_gpu_ += n_used_ - nm;
            // where each missing expert's GGUF blocks are: the mmap (page cache), a pinned copy, or the SSD tier
            const uint8_t * gp[8], * dp[8];
            if (nm) {
                int es[8], n_ram = 0, ram_idx[8];
                for (int m = 0; m < nm; ++m) {
                    const int e = hs.h_ids[miss[m]];
                    gp[m] = L.ex.gu_raw[e];
                    dp[m] = L.ex.dn_raw[e];
                    if (ram_ && !L.ex.zc_ok[e]) { ram_idx[n_ram] = m; es[n_ram++] = e; }
                }
                if (n_ram) {
                    const int n_core = core_end_ - core_start_;
                    const int loop_pass = (a >= core_start_ && a < core_start_ + n_passes_ * n_core)
                                              ? 1 + (a - core_start_) / n_core : 0;
                    const uint8_t * rg[8], * rd[8];
                    ram_fetch(L, hs.layer, loop_pass, es, n_ram, rg, rd);
                    for (int i = 0; i < n_ram; ++i) { gp[ram_idx[i]] = rg[i]; dp[ram_idx[i]] = rd[i]; }
                }
            }
            // the CPU's share (indices into miss); the rest goes to the copy thread
            int cpu[8], nc = 0;
            if (cp_) {
                const auto w0 = std::chrono::steady_clock::now();
                while (!cp_idle_.load(std::memory_order_acquire)) {  // (the previous application's copies have landed)
                    if (stop_) return;
                    _mm_pause();
                }
                dbg_idle_wait_ns_ += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - w0).count();
                CopyJob & J = cp_job_;
                J.app = a; J.value = ready; J.n = 0; J.gb = L.ex.gu_bytes; J.db = L.ex.dn_bytes;
                const double bytes = double(L.ex.gu_bytes + L.ex.dn_bytes);
                const double c_cpu = cpu_unit_ns_.load() * cpu_cost(hs.layer);
                const double c_stage = bytes * stage_ns_b_.load(), c_pin = bytes * pin_ns_b_.load();
                double t_cp = 0, t_cpu = 0;
                int order[8], no = 0;  // pinned ones first: the cheapest to copy
                for (int m = 0; m < nm; ++m) if (L.ex.zc_ok[hs.h_ids[miss[m]]]) order[no++] = m;
                for (int m = 0; m < nm; ++m) if (!L.ex.zc_ok[hs.h_ids[miss[m]]]) order[no++] = m;
                for (int i = 0; i < nm; ++i) {
                    const int m = order[i];
                    const bool pin = L.ex.zc_ok[hs.h_ids[miss[m]]];
                    const double c = pin ? c_pin : c_stage;
                    const bool copy = (pin || cp_stage_on_) &&
                                      (cp_frac_ >= 0.f ? std::floor((i + 1) * cp_frac_) > std::floor(i * cp_frac_)
                                                       : t_cp + c <= t_cpu + c_cpu);
                    if (copy) {
                        J.k[J.n] = miss[m]; J.g[J.n] = gp[m]; J.d[J.n] = dp[m]; J.pinned[J.n] = pin; J.n++;
                        t_cp += c;
                    } else {
                        cpu[nc++] = m;
                        t_cpu += c_cpu;
                    }
                }
                cp_idle_.store(false, std::memory_order_relaxed);
                cp_seq_.fetch_add(1, std::memory_order_release);  // (posted even with nothing to copy: the GPU waits for its flag)
            } else {
                for (int m = 0; m < nm; ++m) cpu[nc++] = m;
            }
            n_cpu_ += nc;
            if (nc) {
                const auto t0 = std::chrono::steady_clock::now();
                constexpr int GU_ROWS = 64, DN_ROWS = 128;
                const int gu_chunks = (n_ff_ + GU_ROWS - 1) / GU_ROWS, dn_chunks = (E_ + DN_ROWS - 1) / DN_ROWS;
                // Stage x out of mapped host memory into L1-resident RAM once; accumulate down in stack y_loc before one writeback.
                memcpy(x_loc.data(), hs.h_x, sizeof(float) * E_);
                pool_->parallel_for(nc * gu_chunks, [&](int i) {
                    const int j = i / gu_chunks, c = i % gu_chunks, m = cpu[j], e = hs.h_ids[miss[m]];
                    const int r0 = c * GU_ROWS, r1 = std::min(n_ff_, (c + 1) * GU_ROWS);
                    ik_cpu_gate_up(L.ex.gu_gtype, gp[m], L.ex.h_gu_scale[e], x_loc.data(), h.data() + size_t(j) * n_ff_, E_, n_ff_, beta_, lb_,
                                   r0, r1);
                });
                pool_->parallel_for(dn_chunks, [&](int c) {
                    const int r0 = c * DN_ROWS, r1 = std::min(E_, (c + 1) * DN_ROWS);
                    float y_loc[DN_ROWS];
                    memset(y_loc, 0, sizeof(float) * (r1 - r0));
                    for (int j = 0; j < nc; ++j) {
                        const int m = cpu[j], k = miss[m], e = hs.h_ids[k];
                        const float coeff = hs.h_wts[k] * L.ex.h_dn_scale[e];
                        ik_cpu_down(L.ex.dn_gtype, dp[m], coeff, h.data() + size_t(j) * n_ff_, y_loc - r0, E_, n_ff_, r0, r1);
                    }
                    memcpy(hs.h_y + r0, y_loc, sizeof(float) * (r1 - r0));
                });
                const long long ns = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count();
                cpu_ns_ += ns;
                cpu_apps_++;
                if (cp_) cpu_unit_ns_.store(0.95 * cpu_unit_ns_.load() + 0.05 * double(ns) / (nc * cpu_cost(hs.layer)));
            } else {
                memset(hs.h_y, 0, sizeof(float) * E_);
            }
            std::atomic_thread_fence(std::memory_order_release);
            *hs.h_done = ready;
            a = (a + 1) % (int) hslots_.size();
        }
    }

    void alloc_state() {
        const int n_slots = n_layer_ + (n_passes_ - 1) * (core_end_ - core_start_);
        slots_.resize(n_slots);
        for (int s = 0; s < n_slots; ++s) {
            const int il = s < n_layer_ ? s : core_start_ + (s - n_layer_) % (core_end_ - core_start_);
            Slot & S = slots_[s];
            S.window = swa_[il] ? n_swa_ : 0;
            S.ring = swa_[il] ? n_swa_ + PF_T : 0;
            const size_t rows = S.ring ? (size_t) S.ring : (size_t) ctx_;
            if (kv_q8_) {
                S.kq = dalloc<int8_t>(rows * kv_);
                S.vq = dalloc<int8_t>(rows * kv_);
                S.ks = dalloc<__half>(rows * (kv_ / 32));
                S.vs = dalloc<__half>(rows * (kv_ / 32));
            } else {
                S.k = dalloc<float>(rows * kv_);
                S.v = dalloc<float>(rows * kv_);
            }
            S.hist_k = dalloc<float>(3 * kv_);
            S.hist_v = dalloc<float>(3 * kv_);
            S.hist_o = dalloc<float>(3 * E_);
            S.hist_f = dalloc<float>(3 * E_);
        }
        {
            std::vector<HistPtrs> hp(n_slots);
            for (int s = 0; s < n_slots; ++s) hp[s] = {slots_[s].hist_k, slots_[s].hist_v, slots_[s].hist_o, slots_[s].hist_f};
            d_hist_ptrs_ = dalloc<HistPtrs>(n_slots);
            upload(d_hist_ptrs_, hp.data(), sizeof(HistPtrs) * n_slots);
        }
        d_pos_ = dalloc<int>(1);
        ck(cudaHostAlloc((void **) &h_pos_, sizeof(int), cudaHostAllocDefault), "h_pos");
        ck(cudaMallocHost(&h_row_, emb_rb_), "pinned");
        d_row_ = dalloc<uint8_t>(emb_rb_);
        ck(cudaMallocHost(&h_logits_, sizeof(float) * n_vocab_), "pinned");
        d_logits_ = dalloc<float>(n_vocab_pad_);
        d_pen_counts_ = dalloc<int32_t>(n_vocab_);
        d_pen_params_ = dalloc<float>(2);
        d_pen_zero_ = dalloc<float>(2);
        d_taps_1_ = dalloc<float>(4 * E_);
        d_anchor_emb_ = dalloc<float>(E_);
        ck(cudaMemset(d_pen_counts_, 0, sizeof(int32_t) * n_vocab_), "pen");
        ck(cudaMemset(d_pen_params_, 0, sizeof(float) * 2), "pen");
        ck(cudaMemset(d_pen_zero_, 0, sizeof(float) * 2), "pen zero");
        ck(cudaHostAlloc((void **) &vf_argmax_h_, sizeof(int32_t) * VF_MAX_T, cudaHostAllocMapped), "vf argmax");
        ck(cudaHostAlloc((void **) &vf_argmax_val_h_, sizeof(float) * VF_MAX_T, cudaHostAllocMapped), "vf argmax val");
        ck(cudaHostGetDevicePointer((void **) &vf_argmax_d_, vf_argmax_h_, 0), "vf argmax");
        ck(cudaHostGetDevicePointer((void **) &vf_argmax_val_d_, vf_argmax_val_h_, 0), "vf argmax val");
        d_argmax_val_ = dalloc<float>(64);
        d_argmax_idx_ = dalloc<int32_t>(64);
        ck(cudaHostAlloc((void **) &h_argmax_idx_, sizeof(int32_t), cudaHostAllocMapped), "argmax");
        ck(cudaHostAlloc((void **) &h_argmax_val_, sizeof(float), cudaHostAllocMapped), "argmax");
        ck(cudaHostGetDevicePointer((void **) &d_argmax_out_idx_, h_argmax_idx_, 0), "argmax");
        ck(cudaHostGetDevicePointer((void **) &d_argmax_out_val_, h_argmax_val_, 0), "argmax");
        ck(cudaHostAlloc((void **) &h_filt_ids_, sizeof(int32_t) * FILT_CAP, cudaHostAllocMapped), "filt");
        ck(cudaHostAlloc((void **) &h_filt_vals_, sizeof(float) * FILT_CAP, cudaHostAllocMapped), "filt");
        ck(cudaHostAlloc((void **) &h_filt_cnt_, sizeof(int), cudaHostAllocMapped), "filt");
        ck(cudaHostGetDevicePointer((void **) &d_filt_ids_, h_filt_ids_, 0), "filt");
        ck(cudaHostGetDevicePointer((void **) &d_filt_vals_, h_filt_vals_, 0), "filt");
        ck(cudaHostGetDevicePointer((void **) &d_filt_cnt_, h_filt_cnt_, 0), "filt");
        ck_init_pool();
        reset();
    }

    // ------------------------------------------------------------ the decode step

    float * buf(size_t n) { return dalloc<float>(n); }

    ik::MixStates states(const std::vector<const float *> & blocks, const float * cur) {
        ik::MixStates s{};
        if (blocks.size() + 1 > (size_t) ik::MAX_STATES) throw std::runtime_error("too many depth-mixing states");
        for (size_t i = 0; i < blocks.size(); ++i) s.p[i] = blocks[i];
        s.p[blocks.size()] = cur;
        s.n = (int) blocks.size() + 1;
        return s;
    }

    void mv(const Q8 & m, const float * x, float * y) {
        if (m.repacked) ck(ik_matvec_q8_0(m.buf, x, y, m.rows, m.n, st_), "matvec");
        else ck(ik_matvec_gguf(m.gtype, m.buf, x, y, m.rows, m.n, st_), "matvec");
    }

    const float * layer(int il, int ip, int slot_idx, const float * cur, std::vector<const float *> & blocks, const float * msgs) {
        const Layer & L = layers_[il];
        Slot & S = slots_[slot_idx];
        const float * inpSA = cur;
        const float * in[3] = {cur, cur, cur};
        if (L.res_query && !msgs) {
            const int nq = L.swa ? 1 : 3;
            float * normed = buf(size_t(nq) * E_);
            const float * nw[3] = {L.swa ? L.attn_norm : L.q_norm, L.k_norm, L.v_norm};
            ik::mix_norm(states(blocks, cur), L.res_query, nq, nw, eps_, normed, E_, st_);
            for (int j = 0; j < 3; ++j) in[j] = normed + (L.swa ? 0 : j) * E_;
        } else {
            if (L.res_query) {
                const int nq = L.swa ? 1 : 3;
                float * mixed = buf(size_t(nq) * E_);
                ik::mix(states(blocks, cur), L.res_query, nq, mixed, E_, st_);
                for (int j = 0; j < 3; ++j) in[j] = mixed + (L.swa ? 0 : j) * E_;
            }
            if (msgs) {
                ik::ControlArgs c{};
                float * o = buf(3 * size_t(E_));
                for (int j = 0; j < 3; ++j) {
                    c.cur[j] = in[j]; c.msg[j] = msgs + j * E_;
                    c.down[j] = L.down[j]; c.gain[j] = L.gain[j]; c.forget[j] = L.forget[j];
                    c.out[j] = o + j * E_;
                }
                ik::control(c, 3, E_, rank_, buf(3 * 64), st_);
                for (int j = 0; j < 3; ++j) in[j] = o + j * E_;
            }
            {
                ik::NormArgs na{};
                float * o = buf(3 * size_t(E_));
                int count = 3;
                if (L.swa && !msgs) count = 1;
                for (int j = 0; j < 3; ++j) {
                    na.x[j] = in[j];
                    na.w[j] = L.swa ? L.attn_norm : (j == 0 ? L.q_norm : j == 1 ? L.k_norm : L.v_norm);
                    na.out[j] = o + j * E_;
                }
                ik::rmsnorm(na, count, E_, eps_, st_);
                for (int j = 0; j < 3; ++j) in[j] = o + (count == 1 ? 0 : j) * E_;
            }
        }
        float * q = buf(H_ * D_), * k = buf(kv_), * v = buf(kv_), * r = buf(H_ * n_rel_), * g = buf(H_ * D_), * sk = buf(H_);
        // fuse shared-input repacked Q8_0 projections into one launch (5 on SWA pass 0, 3 otherwise)
        if (in[0] == in[1] && in[0] == in[2] && L.q.repacked && L.k.repacked && L.v.repacked && L.rel.repacked && L.gate.repacked) {
            const void * ws[5] = {L.q.buf, L.k.buf, L.v.buf, L.rel.buf, L.gate.buf};
            float * ys[5] = {q, k, v, r, g};
            const int rs[5] = {L.q.rows, L.k.rows, L.v.rows, L.rel.rows, L.gate.rows};
            ck(ik_matvec_q8_0_multi(5, ws, in[0], ys, rs, E_, st_), "matvec multi5");
        } else if (L.q.repacked && L.rel.repacked && L.gate.repacked) {
            const void * ws[3] = {L.q.buf, L.rel.buf, L.gate.buf};
            float * ys[3] = {q, r, g};
            const int rs[3] = {L.q.rows, L.rel.rows, L.gate.rows};
            ck(ik_matvec_q8_0_multi(3, ws, in[0], ys, rs, E_, st_), "matvec multi3");
            mv(L.k, in[1], k);
            mv(L.v, in[2], v);
        } else {
            mv(L.q, in[0], q);
            mv(L.k, in[1], k);
            mv(L.v, in[2], v);
            mv(L.rel, in[0], r);
            mv(L.gate, in[0], g);
        }
        ik::matvec_f32(L.sink, in[0], sk, H_, E_, st_);
        float * kdyn = nullptr, * vdyn = nullptr;
        if (!L.swa) {
            kdyn = buf(HK_ * 4);
            vdyn = buf(HK_ * 4);
            ik::matvec_f32(L.kcp, in[1], kdyn, HK_ * 4, E_, st_);
            ik::matvec_f32(L.vcp, in[2], vdyn, HK_ * 4, E_, st_);
        }
        float * qo = buf(H_ * D_), * ro = buf(H_ * n_rel_);
        ik::AttnPrepArgs pa{};
        pa.q = q; pa.k = k; pa.v = v; pa.r = r; pa.kdyn = kdyn; pa.vdyn = vdyn;
        pa.k_conv = L.k_conv; pa.v_conv = L.v_conv; pa.hist_k = S.hist_k; pa.hist_v = S.hist_v;
        pa.q_clip = L.q_clip; pa.rel_clip = L.rel_clip; pa.q_out = qo; pa.r_out = ro;
        pa.k_cache = S.k; pa.v_cache = S.v; pa.k_q = S.kq; pa.v_q = S.vq; pa.k_s = S.ks; pa.v_s = S.vs; pa.kv_q8 = kv_q8_;
        pa.pos = d_pos_; pa.window = S.window; pa.ring = S.ring; pa.global = !L.swa;
        pa.n_head = H_; pa.n_head_kv = HK_; pa.d_head = D_; pa.n_rel = n_rel_;
        pa.qk_clip = qk_clip_; pa.rel_clip_thr = rel_clip_; pa.eps = eps_; pa.log_alpha = log_alpha_; pa.log_floor = log_floor_;
        ik::attn_prep(pa, st_);
        float * o = buf(H_ * D_);
        ik::AttnArgs aa{};
        aa.q = qo; aa.r = ro; aa.sinks = sk; aa.gate = g; aa.rel_bias = L.rel_bias;
        aa.k_cache = S.k; aa.v_cache = S.v; aa.k_q = S.kq; aa.v_q = S.vq; aa.k_s = S.ks; aa.v_s = S.vs; aa.kv_q8 = kv_q8_;
        aa.out = o; aa.pos = d_pos_;
        aa.window = S.window; aa.ring = S.ring; aa.extent = L.swa ? n_swa_ : extent_; aa.n_head = H_; aa.n_head_kv = HK_;
        aa.scratch = buf(ik::attention_scratch_floats(H_));
        ik::attention(aa, st_);
        float * o2 = buf(E_);
        mv(L.out, o, o2);
        const bool block_start = ip % block_ == 0;
        float * c2 = buf(E_);
        ik::conv_residual(o2, S.hist_o, L.post_conv, block_start ? nullptr : inpSA, c2, E_, st_);
        if (block_start) blocks.push_back(inpSA);
        const float * ffn_inp = c2;

        float * xn = buf(E_);
        if (!msgs) {
            const float * nw[1] = {L.ffn_norm};
            ik::mix_norm(states(blocks, c2), L.ffn_res_query, 1, nw, eps_, xn, E_, st_);
        } else {
            float * x = buf(E_), * oc = buf(E_);
            ik::mix(states(blocks, c2), L.ffn_res_query, 1, x, E_, st_);
            ik::ControlArgs c{};
            c.cur[0] = x; c.msg[0] = msgs + 3 * E_;
            c.down[0] = L.down[3]; c.gain[0] = L.gain[3]; c.forget[0] = L.forget[3]; c.out[0] = oc;
            ik::control(c, 1, E_, rank_, buf(64), st_);
            ik::NormArgs na{};
            na.x[0] = oc; na.w[0] = L.ffn_norm; na.out[0] = xn;
            ik::rmsnorm(na, 1, E_, eps_, st_);
        }
        int32_t * ids = dalloc<int32_t>(n_used_);
        float * wts = buf(n_used_);
        ik::router(L.router, L.probs_b, xn, ids, wts, n_expert_, E_, n_used_, w_scale_, buf(n_expert_), st_);
        if (offload_) ik::publish(ids, wts, xn, hslots_[app_].dev, d_pos_, n_used_, E_, st_);
        float * h = buf(size_t(n_used_) * n_ff_), * part = buf(size_t(n_used_) * E_), * y = buf(E_);
        if (L.ex.fast) {
            ck(ik_experts_gate_up(L.ex.type, L.ex.gu, L.ex.gu_scale, ids, n_used_, xn, h, E_, n_ff_, beta_, lb_, st_), "gate_up");
            ck(ik_experts_down(L.ex.type, L.ex.dn, L.ex.dn_scale, ids, wts, n_used_, h, part, y, E_, n_ff_, st_), "down");
        } else {
            ck(ik_experts_gate_up_gguf(L.ex.gu_gtype, L.ex.gu, L.ex.gu_scale, ids, n_used_, xn, h, E_, n_ff_, beta_, lb_, st_), "gate_up");
            ck(ik_experts_down_gguf(L.ex.dn_gtype, L.ex.dn, L.ex.dn_scale, ids, wts, n_used_, h, part, y, E_, n_ff_, st_), "down");
        }
        float * yz = nullptr;
        if (zc_) {  // the GPU's share of the misses, read from pinned host memory while the CPU computes its own
            const void ** tg = (const void **) dalloc<void *>(n_expert_);
            const void ** td = (const void **) dalloc<void *>(n_expert_);
            ik::zc_select(L.ex.gu, L.ex.zc_gu, L.ex.zc_dn, ids, n_used_, zc_share_, tg, td, st_);
            float * hz = buf(size_t(n_used_) * n_ff_), * pz = buf(size_t(n_used_) * E_);
            yz = buf(E_);
            ck(ik_experts_gate_up_gguf(L.ex.gu_gtype, (const void * const *) tg, L.ex.gu_scale, ids, n_used_, xn, hz, E_, n_ff_, beta_, lb_, st_), "zc gate_up");
            ck(ik_experts_down_gguf(L.ex.dn_gtype, (const void * const *) td, L.ex.dn_scale, ids, wts, n_used_, hz, pz, yz, E_, n_ff_, st_), "zc down");
        }
        float * yc = nullptr;
        if (cp_) {  // the copied experts: wait for their copies to land, then compute them from the scratch slots
            const void ** tg = (const void **) dalloc<void *>(n_expert_);
            const void ** td = (const void **) dalloc<void *>(n_expert_);
            ik::copy_wait(L.ex.gu, ids, n_used_, d_pos_, (const volatile int *) ((CopyFlag *) cp_flag_ + app_),
                          (const ik::CopySel *) ((const char *) cp_sel_ + sizeof(CopyFlag) * app_), cp_dev_, cp_slot_, cp_gu_area_, tg, td, st_);
            float * hc = buf(size_t(n_used_) * n_ff_), * pc = buf(size_t(n_used_) * E_);
            yc = buf(E_);
            ck(ik_experts_gate_up_gguf(L.ex.gu_gtype, (const void * const *) tg, L.ex.gu_scale, ids, n_used_, xn, hc, E_, n_ff_, beta_, lb_, st_), "copy gate_up");
            ck(ik_experts_down_gguf(L.ex.dn_gtype, (const void * const *) td, L.ex.dn_scale, ids, wts, n_used_, hc, pc, yc, E_, n_ff_, st_), "copy down");
        }
        if (offload_) ik::wait_add(hslots_[app_].dev, L.ex.gu, ids, n_used_, d_pos_, y, E_, st_);
        if (yz) ik::add(y, yz, E_, st_);
        if (yc) ik::add(y, yc, E_, st_);
        app_++;
        float * c3 = buf(E_);
        ik::conv_residual(y, S.hist_f, L.ffn_post_conv, ffn_inp, c3, E_, st_);
        const std::string tag = "s" + std::to_string(slot_idx) + ".";
        dbg(tag + "in0", in[0], E_); dbg(tag + "q", q, H_ * D_); dbg(tag + "k", k, kv_); dbg(tag + "o", o, H_ * D_);
        dbg(tag + "attn_out", c2, E_); dbg(tag + "ffn_x", xn, E_); dbg(tag + "moe", y, E_); dbg(tag + "out", c3, E_);
        return c3;
    }

    void record_step() {
        app_ = 0;
        ck(cudaMemcpyAsync(d_row_, h_row_, emb_rb_, cudaMemcpyHostToDevice, st_), "row");
        float * x0 = buf(E_);
        ik::embed(d_row_, tok_norm_, x0, E_, eps_, st_, emb_type_);
        dbg("embed", x0, E_);
        const float * cur = x0;
        std::vector<const float *> blocks;
        auto tap1 = [&](const float * src, int tidx) {
            ck(cudaMemcpyAsync(d_taps_1_ + size_t(tidx) * E_, src, sizeof(float) * E_, cudaMemcpyDeviceToDevice, st_), "tap1");
        };
        for (int il = 0; il < core_start_; ++il) {
            if (il == 5) tap1(cur, 0);
            cur = layer(il, il, il, cur, blocks, nullptr);
        }
        const float * prelude = cur;
        const std::vector<const float *> prelude_blocks = blocks;
        const int n_core = core_end_ - core_start_;
        for (int ps = 0; ps < n_passes_; ++ps) {
            std::vector<const float *> bank;
            const float * bank_out = nullptr;
            if (ps > 0) {
                bank.assign(blocks.begin() + prelude_blocks.size() + 1, blocks.end());
                bank_out = cur;
                cur = prelude;
                blocks = prelude_blocks;
            }
            for (int ip = core_start_; ip < core_end_; ++ip) {
                const int slot = ps == 0 ? ip : n_layer_ + (ps - 1) * n_core + (ip - core_start_);
                float * msgs = nullptr;
                if (ps > 0) {
                    msgs = buf(4 * size_t(E_));
                    ik::mix(states(bank, bank_out), layers_[ip].msg_query, 4, msgs, E_, st_);
                }
                if (ps == 1 && ip == 11) tap1(cur, 1);
                if (ps == 2 && ip == 17) tap1(cur, 2);
                cur = layer(ip, ip, slot, cur, blocks, msgs);
            }
        }
        for (int il = core_end_; il < n_layer_; ++il) cur = layer(il, il, il, cur, blocks, nullptr);
        float * xn = buf(E_), * raw = buf(n_vocab_pad_);
        {
            const float * nw[1] = {out_norm_};
            ik::mix_norm(states(blocks, cur), out_res_query_, 1, nw, eps_, xn, E_, st_);
        }
        tap1(xn, 3);
        mv(out_, xn, raw);
        ik::softcap_pen_argmax(raw, d_logits_, d_pen_counts_, d_pen_params_, n_vocab_, softcap_,
                               d_argmax_val_, d_argmax_idx_, d_argmax_out_idx_, d_argmax_out_val_, st_);
        ik::add_pos(d_pos_, 1, st_);
    }

    // ------------------------------------------------------------ prefill

    void prefill_alloc() {
        if (pf_keep_.base) return;
        const size_t te = size_t(PF_T) * E_;
        const int n_apps = n_layer_ + (n_passes_ - 1) * (core_end_ - core_start_);
        pf_keep_.cap = te * (2 * n_apps + 8);
        pf_tmp_.cap = te * 50 + size_t(PF_T) * n_used_ * (n_ff_ + E_) * 2 + size_t(PF_T) * (n_expert_ + 2 * n_used_) + (1 << 20);
        pf_keep_.base = dalloc<float>(pf_keep_.cap);
        pf_msgs_ = dalloc<float>(te * 4);
        pf_tmp_.base = dalloc<float>(pf_tmp_.cap);
        ck(cudaMallocHost(&pf_rows_h_, emb_rb_ * PF_T), "pinned");
        pf_rows_d_ = dalloc<uint8_t>(emb_rb_ * PF_T);
        pf_offsets_ = dalloc<int>(n_expert_ + 1);
        pf_list_ = dalloc<int>(size_t(PF_T) * n_used_);
        pf_active_ = dalloc<int>(n_expert_);
        pf_n_active_ = dalloc<int>(1);
        vf_out_logits_ = dalloc<float>(size_t(VF_MAX_T) * n_vocab_);
        vf_taps_ = dalloc<float>(size_t(VF_MAX_T) * 4 * E_);
        pf_taps_ = dalloc<float>(size_t(PF_T) * 4 * E_);
        if (offload_) {
            size_t most = 0;
            for (const Layer & L : layers_) most = std::max(most, (L.ex.gu_bytes + 255) / 256 * 256 + (L.ex.dn_bytes + 255) / 256 * 256);
            pf_stage_slot_ = most;
            pf_stage_n_ = getenv("IK_PF_STAGE") ? std::max(1, atoi(getenv("IK_PF_STAGE"))) : 16;
            pf_stage_ = dalloc<uint8_t>(pf_stage_slot_ * pf_stage_n_);
            ck(cudaStreamCreateWithFlags(&pf_copy_st_, cudaStreamNonBlocking), "pf copy stream");
            for (int b = 0; b < 2; ++b) {
                pf_tab_gu_[b] = dalloc<void *>(n_expert_);
                pf_tab_dn_[b] = dalloc<void *>(n_expert_);
                ck(cudaHostAlloc((void **) &pf_htab_gu_[b], sizeof(void *) * n_expert_, cudaHostAllocDefault), "pf htab");
                ck(cudaHostAlloc((void **) &pf_htab_dn_[b], sizeof(void *) * n_expert_, cudaHostAllocDefault), "pf htab");
                ck(cudaEventCreateWithFlags(&pf_copy_ev_[b], cudaEventDisableTiming), "pf ev");
                ck(cudaEventCreateWithFlags(&pf_gpu_ev_[b], cudaEventDisableTiming), "pf ev");
            }
            // 2-way double-buffered pinned staging (4 experts x 2 = ~68 MiB, falls back to 2 x 2 = ~34 MiB)
            pf_pin_n_ = std::min(4, std::max(1, pf_stage_n_ / 4));
            for (int try_n = pf_pin_n_; try_n >= 1; try_n /= 2) {
                bool ok = true;
                for (int b = 0; b < 2; ++b) {
                    if (cudaHostAlloc((void **) &pf_pin_[b], pf_stage_slot_ * try_n, cudaHostAllocDefault) != cudaSuccess) {
                        cudaGetLastError();
                        ok = false;
                        break;
                    }
                }
                if (ok) { pf_pin_n_ = try_n; break; }
                for (int b = 0; b < 2; ++b) if (pf_pin_[b]) { cudaFreeHost(pf_pin_[b]); pf_pin_[b] = nullptr; }
                pf_pin_n_ = 0;
            }
            ck(cudaEventCreateWithFlags(&pf_ids_ev_, cudaEventDisableTiming), "pf ids ev");
            ck(cudaHostAlloc((void **) &pf_h_ids_, sizeof(int32_t) * PF_T * n_used_, cudaHostAllocDefault), "pf h_ids");
            ck(cudaEventCreateWithFlags(&vf_ev_, cudaEventDisableTiming), "vf ev");
            ck(cudaHostAlloc((void **) &vf_h_ids_, sizeof(int32_t) * VF_MAX_T * n_used_, cudaHostAllocDefault), "vf ids");
            ck(cudaHostAlloc((void **) &vf_h_wts_, sizeof(float) * VF_MAX_T * n_used_, cudaHostAllocDefault), "vf wts");
            ck(cudaHostAlloc((void **) &vf_h_x_, sizeof(float) * VF_MAX_T * E_, cudaHostAllocDefault), "vf x");
            ck(cudaHostAlloc((void **) &vf_h_y_, sizeof(float) * VF_MAX_T * E_, cudaHostAllocDefault), "vf y");
        }
    }

    // IK_PF_DEBUG=1: report the first non-finite value after each prefill op (slow)
    void nan_check(const char * what, const float * p, size_t n, int il, int pos0) {
        static const bool on = getenv("IK_PF_DEBUG") != nullptr;
        static bool reported = false;
        if (!on || reported) return;
        std::vector<float> h(n);
        ck(cudaMemcpyAsync(h.data(), p, n * 4, cudaMemcpyDeviceToHost, st_), "debug");
        ck(cudaStreamSynchronize(st_), "debug");
        for (size_t i = 0; i < n; ++i)
            if (!std::isfinite(h[i])) {
                fprintf(stderr, "prefill: first non-finite in %s, layer %d, chunk at %d, index %zu of %zu\n", what, il, pos0, i, n);
                reported = true;
                return;
            }
    }

    void mv_pf(const Q8 & m, const float * X, float * Y, int T, const void * Xh = nullptr) {
        static const bool tc_on = !getenv("IK_PF_TC") || atoi(getenv("IK_PF_TC")) != 0;
        if (tc_on && T >= 16 && m.rows % 128 == 0 && m.n % 128 == 0) {
            const void * xp = Xh;
            if (!xp) {
                const int nb = T <= 32 ? 32 : (T <= 64 ? 64 : 128);
                const int tpad = (T + nb - 1) / nb * nb;
                __half * tmp = reinterpret_cast<__half *>(pf_tmp_.take((size_t(tpad) * m.n + 1) / 2));
                ik_pf_f32_to_hperm(X, tmp, T, tpad, m.n, st_);
                xp = tmp;
            }
            if (m.repacked) {
                ck(ik_pf_gemm_q8r(m.buf, xp, Y, m.rows, m.n, T, st_), "pf tc q8");
                return;
            }
            if (m.gtype == gguf::Q4_K || m.gtype == gguf::Q5_K) {
                ck(ik_pf_gemm_gguf(m.gtype, m.buf, xp, Y, m.rows, m.n, T, st_), "pf tc gguf");
                return;
            }
        }
        if (m.repacked) ck(ik::pf::matvec_q8r(m.buf, X, Y, m.rows, m.n, T, st_), "pf matvec");
        else ck(ik_pf_matvec_gguf(m.gtype, m.buf, X, Y, m.rows, m.n, T, st_), "pf matvec");
    }

    // experts for the chunk: resident ones from VRAM; in offload mode, normal prefill streams missing experts into
    // staging in double-buffered batches, while verify_mode (T <= 8) deduplicates misses across tokens and computes
    // them on CPU with batched AVX2 kernels in parallel with the GPU's resident experts.
    void experts_pf(int il, const Layer & L, const int32_t * ids, const float * wts, const float * xn, float * H, float * partial,
                    float * y, int T, bool verify_mode = false, int loop_pass = 0) {
        static const bool tc_on = !getenv("IK_PF_TC") || atoi(getenv("IK_PF_TC")) != 0;
        const bool use_tc = L.ex.fast && tc_on && T >= 16;
        const bool compact = !use_tc && L.ex.fast && T <= 8;
        ik::pf::group(ids, T, n_used_, n_expert_, pf_offsets_, pf_list_, st_,
                      compact ? pf_active_ : nullptr, compact ? pf_n_active_ : nullptr);
        if (offload_ && !verify_mode) {
            // Overlap router id readback + batch-0 CPU staging/DMA on pf_copy_st_ with resident experts on st_.
            ck(cudaEventRecord(pf_ids_ev_, st_), "pf ids ev");
            ck(cudaStreamWaitEvent(pf_copy_st_, pf_ids_ev_, 0), "pf ids wait");
            ck(cudaMemcpyAsync(pf_h_ids_, ids, sizeof(int32_t) * T * n_used_, cudaMemcpyDeviceToHost, pf_copy_st_), "pf ids");
        }
        if (verify_mode && offload_) {
            ck(cudaMemsetAsync(partial, 0, sizeof(float) * T * n_used_ * E_, st_), "vf zero partial");
            ck(cudaEventRecord(vf_ev_, st_), "vf ev");
            ck(cudaStreamWaitEvent(pf_copy_st_, vf_ev_, 0), "vf wait");
            ck(cudaMemcpyAsync(vf_h_ids_, ids, sizeof(int32_t) * T * n_used_, cudaMemcpyDeviceToHost, pf_copy_st_), "vf ids");
            ck(cudaMemcpyAsync(vf_h_wts_, wts, sizeof(float) * T * n_used_, cudaMemcpyDeviceToHost, pf_copy_st_), "vf wts");
            ck(cudaMemcpyAsync(vf_h_x_, xn, sizeof(float) * T * E_, cudaMemcpyDeviceToHost, pf_copy_st_), "vf x");
        }
        void * Xg = nullptr, * Hg = nullptr;
        if (use_tc) {
            const int n_entries = T * n_used_;
            Xg = pf_tmp_.take((size_t(n_entries) * E_ + 1) / 2);
            Hg = pf_tmp_.take((size_t(n_entries) * n_ff_ + 1) / 2);
            ik_pf_gather_hperm(xn, pf_list_, n_used_, n_entries, E_, Xg, st_);
            ck(ik_pf_experts_tc(L.ex.type, (const void * const *) L.ex.gu, (const void * const *) L.ex.dn, L.ex.gu_scale, L.ex.dn_scale,
                                pf_offsets_, pf_list_, wts, Xg, Hg, partial, E_, n_ff_, n_expert_, beta_, lb_, st_), "pf experts tc");
        } else if (L.ex.fast) {
            ck(ik::pf::experts_fast(L.ex.type, (const void * const *) L.ex.gu, (const void * const *) L.ex.dn, L.ex.gu_scale, L.ex.dn_scale,
                                    pf_offsets_, pf_list_, n_used_, wts, xn, H, partial, E_, n_ff_, n_expert_, beta_, lb_, st_,
                                    compact ? pf_active_ : nullptr, compact ? pf_n_active_ : nullptr,
                                    compact ? std::min(n_expert_, T * n_used_) : 0), "pf experts");
        } else
            ck(ik_pf_experts_gguf(L.ex.gu_gtype, L.ex.dn_gtype, (const void * const *) L.ex.gu, (const void * const *) L.ex.dn, L.ex.gu_scale,
                                  L.ex.dn_scale, pf_offsets_, pf_list_, n_used_, wts, xn, H, partial, E_, n_ff_, n_expert_, beta_, lb_, st_),
               "pf experts");
        if (!offload_) {
            ik::pf::sum_entries(partial, y, E_, n_used_, T, st_);
            return;
        }
        if (verify_mode) {
            ik::pf::sum_entries(partial, y, E_, n_used_, T, st_);
            ck(cudaStreamSynchronize(pf_copy_st_), "vf d2h");
            struct MissGrp { int e, n, h_off; int tok[VF_MAX_T]; float wt[VF_MAX_T]; };
            int idx_of[256];
            for (int e = 0; e < n_expert_; ++e) idx_of[e] = -1;
            MissGrp grps[VF_MAX_T * 8];
            int n_grps = 0, total_miss = 0;
            for (int t = 0; t < T; ++t) {
                for (int k = 0; k < n_used_; ++k) {
                    const int e = vf_h_ids_[t * n_used_ + k];
                    reads_[size_t(il) * n_expert_ + e] += 1.f;
                    if (L.ex.slot[e] < 0) {
                        int g = idx_of[e];
                        if (g < 0) {
                            g = n_grps++;
                            idx_of[e] = g;
                            grps[g].e = e;
                            grps[g].n = 0;
                        }
                        grps[g].tok[grps[g].n] = t;
                        grps[g].wt[grps[g].n] = vf_h_wts_[t * n_used_ + k];
                        grps[g].n++;
                        total_miss++;
                    }
                }
            }
            n_gpu_ += T * n_used_ - total_miss;
            n_cpu_ += total_miss;
            if (n_grps == 0) return;
            int h_cur = 0;
            for (int g = 0; g < n_grps; ++g) { grps[g].h_off = h_cur; h_cur += grps[g].n; }
            std::vector<float> h_buf(size_t(total_miss) * n_ff_);
            memset(vf_h_y_, 0, sizeof(float) * T * E_);
            const auto t0 = std::chrono::steady_clock::now();
            constexpr int GU_ROWS = 64, DN_ROWS = 128;
            const int gu_chunks = (n_ff_ + GU_ROWS - 1) / GU_ROWS, dn_chunks = (E_ + DN_ROWS - 1) / DN_ROWS;
            const int max_b = ram_ ? std::max(1, (int) ram_owner_.size()) : n_grps;
            for (int g0 = 0; g0 < n_grps; g0 += max_b) {
                const int nb = std::min(max_b, n_grps - g0);
                std::vector<int> es(nb);
                std::vector<const uint8_t *> gp(nb), dp(nb);
                for (int i = 0; i < nb; ++i) {
                    const int e = grps[g0 + i].e;
                    es[i] = e;
                    gp[i] = L.ex.gu_raw[e];
                    dp[i] = L.ex.dn_raw[e];
                }
                if (ram_) ram_fetch(L, il, loop_pass, es.data(), nb, gp.data(), dp.data());
                pool_->parallel_for(nb * gu_chunks, [&](int idx) {
                    const int i = idx / gu_chunks, c = idx % gu_chunks;
                    const MissGrp & G = grps[g0 + i];
                    const float * xs[VF_MAX_T];
                    float * hs[VF_MAX_T];
                    for (int j = 0; j < G.n; ++j) {
                        xs[j] = vf_h_x_ + size_t(G.tok[j]) * E_;
                        hs[j] = h_buf.data() + size_t(G.h_off + j) * n_ff_;
                    }
                    ik_cpu_gate_up_batch(L.ex.gu_gtype, gp[i], L.ex.h_gu_scale[G.e], xs, hs, G.n, E_, n_ff_, beta_, lb_,
                                         c * GU_ROWS, std::min(n_ff_, (c + 1) * GU_ROWS));
                });
                pool_->parallel_for(dn_chunks, [&](int c) {
                    const int r0 = c * DN_ROWS, r1 = std::min(E_, (c + 1) * DN_ROWS);
                    for (int i = 0; i < nb; ++i) {
                        const MissGrp & G = grps[g0 + i];
                        float cs[VF_MAX_T];
                        const float * hs[VF_MAX_T];
                        float * ys[VF_MAX_T];
                        const float sd = L.ex.h_dn_scale[G.e];
                        for (int j = 0; j < G.n; ++j) {
                            cs[j] = G.wt[j] * sd;
                            hs[j] = h_buf.data() + size_t(G.h_off + j) * n_ff_;
                            ys[j] = vf_h_y_ + size_t(G.tok[j]) * E_;
                        }
                        ik_cpu_down_batch(L.ex.dn_gtype, dp[i], cs, hs, ys, G.n, E_, n_ff_, r0, r1);
                    }
                });
            }
            cpu_ns_ += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count();
            cpu_apps_++;
            float * y_cpu = pf_tmp_.take(size_t(T) * E_);
            ck(cudaMemcpyAsync(y_cpu, vf_h_y_, sizeof(float) * T * E_, cudaMemcpyHostToDevice, st_), "vf y_cpu");
            ik::add(y, y_cpu, T * E_, st_);
            return;
        }
        // Resident experts are already queued on st_; wait only for pf_h_ids_ D2H on pf_copy_st_.
        ck(cudaStreamSynchronize(pf_copy_st_), "pf ids sync");
        std::vector<char> seen(n_expert_, 0);
        std::vector<int> miss;
        const size_t n_id = size_t(T) * n_used_;
        for (size_t i = 0; i < n_id; ++i) {
            const int32_t e = pf_h_ids_[i];
            if (!seen[e]) { seen[e] = 1; if (L.ex.slot[e] < 0) miss.push_back(e); }
        }
        if (miss.empty()) {
            ik::pf::sum_entries(partial, y, E_, n_used_, T, st_);
            return;
        }
        const bool stream_tc = use_tc && pf_stage_n_ >= 4;
        const bool dbl = (pf_pin_[0] && pf_pin_[1] && pf_pin_n_ >= 1 && pf_stage_n_ >= (stream_tc ? 4 : 2));
        const int half_slots = pf_stage_n_ / 2;
        const int batch_n = dbl ? std::min(pf_pin_n_, stream_tc ? half_slots / 2 : half_slots)
                                : (stream_tc ? half_slots : pf_stage_n_);
        const size_t gu_aln = (L.ex.gu_bytes + 255) / 256 * 256;
        for (size_t b = 0; b < miss.size(); b += batch_n) {
            const int buf = dbl ? pf_cur_ : 0;
            if (pf_gpu_busy_[buf]) {
                ck(cudaEventSynchronize(pf_gpu_ev_[buf]), "pf wait gpu");
                pf_gpu_busy_[buf] = false;
            }
            const size_t cnt = std::min<size_t>(miss.size() - b, batch_n);
            if (dbl && pool_) {
                pool_->parallel_for((int) cnt, [&](int k) {
                    const int e = miss[b + k];
                    uint8_t * pg = pf_pin_[buf] + pf_stage_slot_ * k;
                    memcpy(pg, L.ex.gu_raw[e], L.ex.gu_bytes);
                    memcpy(pg + gu_aln, L.ex.dn_raw[e], L.ex.dn_bytes);
                });
            }
            memset(pf_htab_gu_[buf], 0, sizeof(void *) * n_expert_);
            memset(pf_htab_dn_[buf], 0, sizeof(void *) * n_expert_);
            const int slot_base = dbl ? buf * half_slots : 0;
            const int rep_base = slot_base + (stream_tc ? batch_n : 0);
            cudaStream_t cst = dbl ? pf_copy_st_ : st_;
            for (size_t k = 0; k < cnt; ++k) {
                const int e = miss[b + k];
                uint8_t * g = pf_stage_ + pf_stage_slot_ * (slot_base + k);
                uint8_t * d = g + gu_aln;
                const uint8_t * sg = dbl ? (pf_pin_[buf] + pf_stage_slot_ * k) : L.ex.gu_raw[e];
                const uint8_t * sd = dbl ? (pf_pin_[buf] + pf_stage_slot_ * k + gu_aln) : L.ex.dn_raw[e];
                ck(cudaMemcpyAsync(g, sg, L.ex.gu_bytes, cudaMemcpyHostToDevice, cst), "pf stream");
                ck(cudaMemcpyAsync(d, sd, L.ex.dn_bytes, cudaMemcpyHostToDevice, cst), "pf stream");
                if (stream_tc) {
                    uint8_t * rg = pf_stage_ + pf_stage_slot_ * (rep_base + k);
                    uint8_t * rd = rg + gu_dev_bytes(L.ex);
                    pf_htab_gu_[buf][e] = rg;
                    pf_htab_dn_[buf][e] = rd;
                } else {
                    pf_htab_gu_[buf][e] = g;
                    pf_htab_dn_[buf][e] = d;
                }
            }
            ck(cudaMemcpyAsync(pf_tab_gu_[buf], pf_htab_gu_[buf], sizeof(void *) * n_expert_, cudaMemcpyHostToDevice, cst), "pf tab");
            ck(cudaMemcpyAsync(pf_tab_dn_[buf], pf_htab_dn_[buf], sizeof(void *) * n_expert_, cudaMemcpyHostToDevice, cst), "pf tab");
            if (dbl) {
                ck(cudaEventRecord(pf_copy_ev_[buf], pf_copy_st_), "pf ev");
                ck(cudaStreamWaitEvent(st_, pf_copy_ev_[buf], 0), "pf wait copy");
            }
            if (stream_tc) {
                for (size_t k = 0; k < cnt; ++k) {
                    uint8_t * g = pf_stage_ + pf_stage_slot_ * (slot_base + k);
                    uint8_t * d = g + gu_aln;
                    uint8_t * rg = pf_stage_ + pf_stage_slot_ * (rep_base + k);
                    uint8_t * rd = rg + gu_dev_bytes(L.ex);
                    ck(ik_repack(L.ex.type, g, rg, 2 * n_ff_, E_, st_), "pf repack");
                    ck(ik_repack(L.ex.type, d, rd, E_, n_ff_, st_), "pf repack");
                }
                ck(ik_pf_experts_tc(L.ex.type, (const void * const *) pf_tab_gu_[buf], (const void * const *) pf_tab_dn_[buf],
                                    L.ex.gu_scale, L.ex.dn_scale, pf_offsets_, pf_list_, wts, Xg, Hg, partial,
                                    E_, n_ff_, n_expert_, beta_, lb_, st_), "pf streamed experts tc");
            } else {
                ck(ik_pf_experts_gguf(L.ex.gu_gtype, L.ex.dn_gtype, (const void * const *) pf_tab_gu_[buf], (const void * const *) pf_tab_dn_[buf],
                                      L.ex.gu_scale, L.ex.dn_scale, pf_offsets_, pf_list_, n_used_, wts, xn, H, partial, E_, n_ff_, n_expert_,
                                      beta_, lb_, st_), "pf streamed experts");
            }
            ck(cudaEventRecord(pf_gpu_ev_[buf], st_), "pf ev");
            pf_gpu_busy_[buf] = true;
            if (dbl) pf_cur_ ^= 1;
        }
        ik::pf::sum_entries(partial, y, E_, n_used_, T, st_);
    }

    const float * layer_pf(int il, int ip, int slot_idx, const float * cur, std::vector<const float *> & blocks, const float * msgs, int T,
                           int pos0, bool verify_mode = false, int loop_pass = 0, const int * d_pos = nullptr) {
        const Layer & L = layers_[il];
        Slot & S = slots_[slot_idx];
        float * sn_k = verify_mode ? vf_snaps_ + size_t(slot_idx) * vf_snap_stride_ : nullptr;
        float * sn_v = sn_k ? sn_k + size_t(T) * 3 * kv_ : nullptr;
        float * sn_o = sn_k ? sn_v + size_t(T) * 3 * kv_ : nullptr;
        float * sn_f = sn_k ? sn_o + size_t(T) * 3 * E_ : nullptr;
        pf_tmp_.used = 0;
        const size_t TE = size_t(T) * E_;
        const float * inpSA = cur;
        const float * in[3] = {cur, cur, cur};
        size_t in_stride = E_;
        if (L.res_query) {
            const int nq = L.swa ? 1 : 3;
            float * mixed = pf_tmp_.take(size_t(T) * nq * E_);
            ik::pf::mix(states(blocks, cur), L.res_query, nq, mixed, E_, T, st_);
            for (int j = 0; j < 3; ++j) in[j] = mixed + (L.swa ? 0 : j) * E_;
            in_stride = size_t(nq) * E_;
        }
        if (msgs) {
            ik::ControlArgs c{};
            float * o = pf_tmp_.take(3 * TE);
            for (int j = 0; j < 3; ++j) {
                c.cur[j] = in[j]; c.msg[j] = msgs + j * E_;
                c.down[j] = L.down[j]; c.gain[j] = L.gain[j]; c.forget[j] = L.forget[j];
                c.out[j] = o + j * TE;
            }
            // cur strides: in_stride; msgs [T][4][E]
            ik::pf::control(c, 3, in_stride, 4 * size_t(E_), E_, E_, rank_, T, pf_tmp_.take(size_t(T) * 3 * 64), st_);
            for (int j = 0; j < 3; ++j) in[j] = o + j * TE;
            in_stride = E_;
        }
        float * normed = pf_tmp_.take(3 * TE);
        {
            ik::NormArgs na{};
            int count = (L.swa && !msgs) ? 1 : 3;
            for (int j = 0; j < 3; ++j) {
                na.x[j] = in[j];
                na.w[j] = L.swa ? L.attn_norm : (j == 0 ? L.q_norm : j == 1 ? L.k_norm : L.v_norm);
                na.out[j] = normed + j * TE;
            }
            ik::pf::rmsnorm(na, count, in_stride, E_, E_, eps_, T, st_);
            for (int j = 0; j < 3; ++j) in[j] = normed + (count == 1 ? 0 : j) * TE;
        }
        float * q = pf_tmp_.take(size_t(T) * H_ * D_), * k = pf_tmp_.take(size_t(T) * kv_), * v = pf_tmp_.take(size_t(T) * kv_);
        float * r = pf_tmp_.take(size_t(T) * H_ * n_rel_), * g = pf_tmp_.take(size_t(T) * H_ * D_), * sk = pf_tmp_.take(size_t(T) * H_);
        nan_check("attention input", in[0], size_t(T) * E_, il, pos0);
        const void * xh[3] = {nullptr, nullptr, nullptr};
        static const bool tc_on = !getenv("IK_PF_TC") || atoi(getenv("IK_PF_TC")) != 0;
        if (tc_on && T >= 16) {
            const int nb = T <= 32 ? 32 : (T <= 64 ? 64 : 128);
            const int tpad = (T + nb - 1) / nb * nb;
            for (int j = 0; j < 3; ++j) {
                if (j > 0 && in[j] == in[0]) { xh[j] = xh[0]; continue; }
                void * p = pf_tmp_.take((size_t(tpad) * E_ + 1) / 2);
                ik_pf_f32_to_hperm(in[j], p, T, tpad, E_, st_);
                xh[j] = p;
            }
        }
        if (T <= 8 && in[0] == in[1] && in[0] == in[2] && L.q.repacked && L.k.repacked && L.v.repacked && L.rel.repacked && L.gate.repacked) {
            const void * ws[5] = {L.q.buf, L.k.buf, L.v.buf, L.rel.buf, L.gate.buf};
            float * ys[5] = {q, k, v, r, g};
            const int rs[5] = {L.q.rows, L.k.rows, L.v.rows, L.rel.rows, L.gate.rows};
            ck(ik::pf::matvec_q8r_multi(5, ws, in[0], ys, rs, E_, T, st_), "pf matvec multi5");
        } else if (T <= 8 && L.q.repacked && L.rel.repacked && L.gate.repacked) {
            const void * ws[3] = {L.q.buf, L.rel.buf, L.gate.buf};
            float * ys[3] = {q, r, g};
            const int rs[3] = {L.q.rows, L.rel.rows, L.gate.rows};
            ck(ik::pf::matvec_q8r_multi(3, ws, in[0], ys, rs, E_, T, st_), "pf matvec multi3");
            mv_pf(L.k, in[1], k, T, xh[1]);
            mv_pf(L.v, in[2], v, T, xh[2]);
        } else {
            mv_pf(L.q, in[0], q, T, xh[0]);
            mv_pf(L.k, in[1], k, T, xh[1]);
            mv_pf(L.v, in[2], v, T, xh[2]);
            mv_pf(L.rel, in[0], r, T, xh[0]);
            mv_pf(L.gate, in[0], g, T, xh[0]);
        }
        ik::pf::matvec_f32(L.sink, in[0], sk, H_, E_, T, st_);
        float * kdyn = nullptr, * vdyn = nullptr;
        if (!L.swa) {
            kdyn = pf_tmp_.take(size_t(T) * HK_ * 4);
            vdyn = pf_tmp_.take(size_t(T) * HK_ * 4);
            ik::pf::matvec_f32(L.kcp, in[1], kdyn, HK_ * 4, E_, T, st_);
            ik::pf::matvec_f32(L.vcp, in[2], vdyn, HK_ * 4, E_, T, st_);
        }
        const float norm_k = 1.f / std::sqrt(float((L.swa ? 1 : 2) * 4 + 1));
        float * kc = pf_tmp_.take(size_t(T) * kv_), * vc = pf_tmp_.take(size_t(T) * kv_);
        ik::pf::conv(k, S.hist_k, L.k_conv, kdyn, HK_, kc, kv_, T, norm_k, st_, sn_k);
        ik::pf::conv(v, S.hist_v, L.v_conv, vdyn, HK_, vc, kv_, T, norm_k, st_, sn_v);
        float * qo = pf_tmp_.take(size_t(T) * H_ * D_), * ro = pf_tmp_.take(size_t(T) * H_ * n_rel_);
        ik::AttnPrepArgs pa{};
        pa.q = q; pa.r = r; pa.q_clip = L.q_clip; pa.rel_clip = L.rel_clip; pa.q_out = qo; pa.r_out = ro;
        pa.k_cache = S.k; pa.v_cache = S.v; pa.k_q = S.kq; pa.v_q = S.vq; pa.k_s = S.ks; pa.v_s = S.vs; pa.kv_q8 = kv_q8_;
        pa.window = S.window; pa.ring = S.ring; pa.global = !L.swa;
        pa.n_head = H_; pa.n_head_kv = HK_; pa.d_head = D_; pa.n_rel = n_rel_;
        pa.qk_clip = qk_clip_; pa.rel_clip_thr = rel_clip_; pa.eps = eps_; pa.log_alpha = log_alpha_; pa.log_floor = log_floor_;
        nan_check("k conv", kc, size_t(T) * kv_, il, pos0);
        ik::pf::attn_prep2(pa, kc, vc, pos0, T, st_, d_pos);
        nan_check("q out", qo, size_t(T) * H_ * D_, il, pos0);
        float * o = pf_tmp_.take(size_t(T) * H_ * D_);
        ik::AttnArgs aa{};
        aa.q = qo; aa.r = ro; aa.sinks = sk; aa.gate = g; aa.rel_bias = L.rel_bias;
        aa.k_cache = S.k; aa.v_cache = S.v; aa.k_q = S.kq; aa.v_q = S.vq; aa.k_s = S.ks; aa.v_s = S.vs; aa.kv_q8 = kv_q8_;
        aa.out = o;
        aa.window = S.window; aa.ring = S.ring; aa.extent = L.swa ? n_swa_ : extent_; aa.n_head = H_; aa.n_head_kv = HK_;
        ik::pf::attention(aa, pos0, T, st_, d_pos);
        nan_check("attention", o, size_t(T) * H_ * D_, il, pos0);
        float * o2 = pf_tmp_.take(TE);
        mv_pf(L.out, o, o2, T);
        const bool block_start = ip % block_ == 0;
        float * c2 = pf_keep_.take(TE);
        ik::pf::conv_res(o2, S.hist_o, L.post_conv, block_start ? nullptr : inpSA, c2, E_, T, st_, sn_o);
        if (block_start) blocks.push_back(inpSA);
        const float * ffn_inp = c2;

        float * x = pf_tmp_.take(TE);
        ik::pf::mix(states(blocks, c2), L.ffn_res_query, 1, x, E_, T, st_);
        const float * xin = x;
        if (msgs) {
            ik::ControlArgs c{};
            float * oc = pf_tmp_.take(TE);
            c.cur[0] = x; c.msg[0] = msgs + 3 * E_;
            c.down[0] = L.down[3]; c.gain[0] = L.gain[3]; c.forget[0] = L.forget[3]; c.out[0] = oc;
            ik::pf::control(c, 1, E_, 4 * size_t(E_), E_, E_, rank_, T, pf_tmp_.take(size_t(T) * 64), st_);
            xin = oc;
        }
        float * xn = pf_tmp_.take(TE);
        {
            ik::NormArgs na{};
            na.x[0] = xin; na.w[0] = L.ffn_norm; na.out[0] = xn;
            ik::pf::rmsnorm(na, 1, E_, E_, E_, eps_, T, st_);
        }
        int32_t * ids = (int32_t *) pf_tmp_.take(size_t(T) * n_used_);
        float * wts = pf_tmp_.take(size_t(T) * n_used_);
        float * probs = pf_tmp_.take(size_t(T) * n_expert_);
        nan_check("ffn input", xn, size_t(T) * E_, il, pos0);
        ik::pf::router(L.router, L.probs_b, xn, ids, wts, probs, n_expert_, E_, n_used_, w_scale_, T, st_);
        float * h = pf_tmp_.take(size_t(T) * n_used_ * n_ff_), * part = pf_tmp_.take(size_t(T) * n_used_ * E_), * y = pf_tmp_.take(TE);
        experts_pf(il, L, ids, wts, xn, h, part, y, T, verify_mode, loop_pass);
        nan_check("experts", y, size_t(T) * E_, il, pos0);
        float * c3 = pf_keep_.take(TE);
        ik::pf::conv_res(y, S.hist_f, L.ffn_post_conv, ffn_inp, c3, E_, T, st_, sn_f);
        return c3;
    }

    void prefill_chunk(const int * toks, int T) {
        prefill_alloc();
        pf_keep_.used = 0;
        const gguf::Tensor & emb = f_.get("token_embd.weight");
        for (int t = 0; t < T; ++t) memcpy(pf_rows_h_ + emb_rb_ * t, emb.data + size_t(toks[t]) * emb_rb_, emb_rb_);
        ck(cudaMemcpyAsync(pf_rows_d_, pf_rows_h_, emb_rb_ * T, cudaMemcpyHostToDevice, st_), "rows");
        const int pos0 = pos_;
        float * x0 = pf_keep_.take(size_t(T) * E_);
        ik::pf::embed(pf_rows_d_, tok_norm_, x0, E_, eps_, T, st_, emb_type_);
        const float * cur = x0;
        std::vector<const float *> blocks;
        auto tap_pf = [&](const float * src, int tidx) {
            if (dspark_) scatter_tap_kernel<<<dim3((E_ + 255) / 256, T), 256, 0, st_>>>(src, pf_taps_, tidx, T, E_);
        };
        for (int il = 0; il < core_start_; ++il) {
            if (il == 5) tap_pf(cur, 0);
            cur = layer_pf(il, il, il, cur, blocks, nullptr, T, pos0);
        }
        const float * prelude = cur;
        const std::vector<const float *> prelude_blocks = blocks;
        const int n_core = core_end_ - core_start_;
        for (int ps = 0; ps < n_passes_; ++ps) {
            std::vector<const float *> bank;
            const float * bank_out = nullptr;
            if (ps > 0) {
                bank.assign(blocks.begin() + prelude_blocks.size() + 1, blocks.end());
                bank_out = cur;
                cur = prelude;
                blocks = prelude_blocks;
            }
            for (int ip = core_start_; ip < core_end_; ++ip) {
                const int slot = ps == 0 ? ip : n_layer_ + (ps - 1) * n_core + (ip - core_start_);
                float * msgs = nullptr;
                if (ps > 0) {
                    msgs = pf_msgs_;
                    ik::pf::mix(states(bank, bank_out), layers_[ip].msg_query, 4, msgs, E_, T, st_);
                }
                if (ps == 1 && ip == 11) tap_pf(cur, 1);
                if (ps == 2 && ip == 17) tap_pf(cur, 2);
                cur = layer_pf(ip, ip, slot, cur, blocks, msgs, T, pos0);
            }
        }
        for (int il = core_end_; il < n_layer_; ++il) cur = layer_pf(il, il, il, cur, blocks, nullptr, T, pos0);
        if (dspark_) {
            pf_tmp_.used = 0;
            float * x = pf_tmp_.take(size_t(T) * E_);
            ik::pf::mix(states(blocks, cur), out_res_query_, 1, x, E_, T, st_);
            float * xn = pf_tmp_.take(size_t(T) * E_);
            ik::NormArgs na{};
            na.x[0] = x; na.w[0] = out_norm_; na.out[0] = xn;
            ik::pf::rmsnorm(na, 1, E_, E_, E_, eps_, T, st_);
            tap_pf(xn, 3);
            dspark_->store_context(pf_taps_, pos0, T, st_);
        }
        pos_ += T;
        ck(cudaMemcpyAsync(d_pos_, &pos_, sizeof(int), cudaMemcpyHostToDevice, st_), "pos");
        ck(cudaStreamSynchronize(st_), "prefill");
        pf_gpu_busy_[0] = pf_gpu_busy_[1] = false;
    }

    void build_graph() {
        // relaxed: buffers are allocated while recording
        ck(cudaStreamBeginCapture(st_, cudaStreamCaptureModeRelaxed), "capture");
        try {
            record_step();
        } catch (...) {
            cudaGraph_t g;
            cudaStreamEndCapture(st_, &g);
            if (g) cudaGraphDestroy(g);
            throw;
        }
        ck(cudaStreamEndCapture(st_, &graph_), "end capture");
        recording_dbg_ = false;
        ck(cudaGraphInstantiate(&exec_, graph_, 0), "instantiate");
    }
};

}  // namespace

extern "C" {

const char * ik_last_error() { return g_err.c_str(); }

void * ik_new(const char * path, int ctx) {
    try {
        return new Engine(path, ctx);
    } catch (const std::exception & e) {
        g_err = e.what();
        return nullptr;
    }
}

// offload mode: vram_expert_mib of expert cache in VRAM (the rest computed by the CPU from the mmap'd GGUF);
// profile: float32 [n_layer][n_expert] reads per token (inkling/routing/make_profile.py) or null; n_threads: CPU
// threads for experts (0: half the hardware threads)
void * ik_new2(const char * path, int ctx, int vram_expert_mib, const char * profile, int n_threads) {
    try {
        return new Engine(path, ctx, (long long) vram_expert_mib << 20, profile, n_threads);
    } catch (const std::exception & e) {
        g_err = e.what();
        return nullptr;
    }
}

// [reads served from VRAM, experts computed by the CPU, experts swapped in, CPU compute ns, layer applications with
// CPU work, SSD-tier reads, SSD-tier bytes, SSD-tier read ns, experts copied to VRAM for the GPU, copy ns] since the
// last call (10 values)
void ik_stats(void * h, long long * out) { static_cast<Engine *>(h)->stats(out); }

int ik_reset(void * h) {
    try {
        static_cast<Engine *>(h)->reset();
        return 0;
    } catch (const std::exception & e) {
        g_err = e.what();
        return -1;
    }
}

// feed one token; writes the next-token logits (n_vocab floats) to `logits`
int ik_decode(void * h, int token, float * logits) {
    try {
        static_cast<Engine *>(h)->decode(token, logits);
        return 0;
    } catch (const std::exception & e) {
        g_err = e.what();
        return -1;
    }
}

// Zero-Logits-D2H greedy decode: returns winning token id (>= 0), or -1 on error
int ik_decode_greedy(void * h, int token) {
    try {
        return static_cast<Engine *>(h)->decode_greedy(token);
    } catch (const std::exception & e) {
        g_err = e.what();
        return -1;
    }
}

// Zero-Logits-D2H filtered decode for stochastic sampling: returns candidate count (>= 0; > cap => call ik_read_logits)
int ik_decode_filtered(void * h, int token, float margin, int32_t * ids_out, float * vals_out, int cap, float * max_out) {
    try {
        return static_cast<Engine *>(h)->decode_filtered(token, margin, ids_out, vals_out, cap, max_out);
    } catch (const std::exception & e) {
        g_err = e.what();
        return -1;
    }
}

int ik_read_logits(void * h, float * logits) {
    try {
        static_cast<Engine *>(h)->read_logits(logits);
        return 0;
    } catch (const std::exception & e) {
        g_err = e.what();
        return -1;
    }
}

int ik_set_penalty(void * h, float pres, float freq) {
    try {
        static_cast<Engine *>(h)->set_penalty(pres, freq);
        return 0;
    } catch (const std::exception & e) {
        g_err = e.what();
        return -1;
    }
}

int ik_pen_clear(void * h) {
    try {
        static_cast<Engine *>(h)->pen_clear();
        return 0;
    } catch (const std::exception & e) {
        g_err = e.what();
        return -1;
    }
}

int ik_pen_inc(void * h, int tok) {
    try {
        static_cast<Engine *>(h)->pen_inc(tok);
        return 0;
    } catch (const std::exception & e) {
        g_err = e.what();
        return -1;
    }
}

// exact multi-token verification (1 <= n <= 8): writes n * n_vocab logits; call ik_verify_commit(acc) afterwards
int ik_verify(void * h, const int * toks, int n, float * logits) {
    try {
        static_cast<Engine *>(h)->verify(toks, n, logits);
        return 0;
    } catch (const std::exception & e) {
        g_err = e.what();
        return -1;
    }
}

int ik_verify_commit(void * h, int acc) {
    try {
        static_cast<Engine *>(h)->verify_commit(acc);
        return 0;
    } catch (const std::exception & e) {
        g_err = e.what();
        return -1;
    }
}

int ik_verify_greedy(void * h, const int * toks, int n, int * argmax_out) {
    try {
        static_cast<Engine *>(h)->verify_greedy(toks, n, argmax_out);
        return 0;
    } catch (const std::exception & e) {
        g_err = e.what();
        return -1;
    }
}

int ik_load_mtp(void * h, const char * mtp_path) {
    try {
        static_cast<Engine *>(h)->load_mtp(mtp_path);
        return 0;
    } catch (const std::exception & e) {
        g_err = e.what();
        return -1;
    }
}

int ik_has_mtp(void * h) { return static_cast<Engine *>(h)->has_mtp() ? 1 : 0; }

int ik_draft(void * h, int anchor_tok, int n_draft, int * out_drafts) {
    try {
        return static_cast<Engine *>(h)->draft(anchor_tok, n_draft, out_drafts);
    } catch (const std::exception & e) {
        g_err = e.what();
        return -1;
    }
}

int ik_n_vocab(void * h) { return static_cast<Engine *>(h)->n_vocab(); }

// conversation checkpoint pool: save at current pos, restore latest or specific pos, list, pin, plan
int ik_checkpoint_save(void * h) {
    try { static_cast<Engine *>(h)->save_checkpoint(); return 0; } catch (const std::exception & e) { g_err = e.what(); return -1; }
}
int ik_checkpoint_restore(void * h) {
    try { return static_cast<Engine *>(h)->restore_checkpoint() ? 0 : -1; } catch (const std::exception & e) { g_err = e.what(); return -1; }
}
int ik_checkpoint_restore_pos(void * h, int pos) {
    try { return static_cast<Engine *>(h)->restore_checkpoint_pos(pos) ? 0 : -1; } catch (const std::exception & e) { g_err = e.what(); return -1; }
}
int ik_checkpoint_pos(void * h) { return static_cast<Engine *>(h)->checkpoint_pos(); }
int ik_checkpoints(void * h, int * out, int cap) {
    const auto v = static_cast<Engine *>(h)->checkpoints();
    if (out) for (int i = 0; i < std::min((int) v.size(), cap); ++i) out[i] = v[i];
    return (int) v.size();
}
int ik_checkpoint_pin(void * h, const int * pos, int n) {
    try { return static_cast<Engine *>(h)->ck_pin(pos, n); } catch (const std::exception & e) { g_err = e.what(); return -1; }
}
int ik_checkpoint_pinned(void * h, int * out, int cap) {
    const auto v = static_cast<Engine *>(h)->ck_pinned();
    if (out) for (int i = 0; i < std::min((int) v.size(), cap); ++i) out[i] = v[i];
    return (int) v.size();
}
int ik_checkpoint_plan(void * h, const int * cks, int n_cks, int ck_exact, int base, int end, int * out, int cap) {
    try {
        const auto v = static_cast<Engine *>(h)->plan_checkpoints(cks, n_cks, ck_exact, base, end);
        if (out) for (int i = 0; i < std::min((int) v.size(), cap); ++i) out[i] = v[i];
        return (int) v.size();
    } catch (const std::exception & e) {
        g_err = e.what();
        return -1;
    }
}

long long ik_debug_check_state(void * h) { return static_cast<Engine *>(h)->check_state(); }
int ik_debug_device_sync() { return (int) cudaDeviceSynchronize(); }

// feed n tokens batched (no logits); the next ik_decode continues after them
int ik_prefill(void * h, const int * toks, int n) {
    try {
        static_cast<Engine *>(h)->prefill(toks, n);
        return 0;
    } catch (const std::exception & e) {
        g_err = e.what();
        return -1;
    }
}

int ik_debug_count(void * h) { return (int) static_cast<Engine *>(h)->dbg_.size(); }
const char * ik_debug_name(void * h, int i) { return static_cast<Engine *>(h)->dbg_[i].name.c_str(); }
int ik_debug_read(void * h, int i, float * out) {
    const auto & d = static_cast<Engine *>(h)->dbg_[i];
    cudaDeviceSynchronize();
    return cudaMemcpy(out, d.p, sizeof(float) * d.n, cudaMemcpyDeviceToHost) == cudaSuccess ? d.n : -1;
}

void ik_free(void * h) { delete static_cast<Engine *>(h); }

}  // extern "C"
