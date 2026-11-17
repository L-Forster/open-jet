// C API over MKEngine plus libllama's tokenizer (vocab only), for the HTTP server (server/mk_server.py).
#include <algorithm>
#include <chrono>
#include <climits>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <cuda_runtime.h>

#include "llama.h"
#include "qwen38_mk.h"

// the build's commit and engine flags for mk_build_info: written by the build into its output directory (Makefile:
// build/, build.ps1: build_win\), which it puts on the include path of this file
#if defined(__has_include)
#if __has_include("mk_build_info.h")
#include "mk_build_info.h"
#endif
#endif
#ifndef MK_BUILD_COMMIT
#define MK_BUILD_COMMIT "unknown"
#endif
#ifndef MK_BUILD_FLAGS
#define MK_BUILD_FLAGS "unknown"
#endif
#define MK_STR2(...) #__VA_ARGS__
#define MK_STR(...) MK_STR2(__VA_ARGS__)

namespace {
// ---- keep-warm (MK_KEEP_WARM_S) ----
// A few seconds without GPU work and the driver drops the clocks (P8); a reply that starts there runs its first ~2 s at
// ramping clocks. mk_keep_warm(h, 1), which the server calls once ready
// and after every request, keeps a light load on the GPU for MK_KEEP_WARM_S seconds (default 120; 0 = off), then lets
// the card idle. The driver's power settings stay as they are. Every C API call that runs GPU work stops the load
// first: the kernel polls a host-mapped stop word and ends within microseconds, so it never runs next to an engine
// launch (mk_cycle is a cooperative launch over every SM; a missed stop could only delay it, each warm kernel ends by
// itself), and it touches no engine memory. Programs that never call mk_keep_warm (the CLI, benchmarks, tests) never
// warm. MK_KEEP_WARM_LOAD = "blocks,threads,on_ms,off_ms,spin" (measurements): kernels of on_ms each (far below WDDM's
// 2 s timeout) with off_ms without a kernel between them; spin 1 = every thread busy-loops, 0 = one polling thread.
// On an RTX 3090 the driver keys the P-state to utilization, so only a continuous load holds P2 (a 50% duty
// cycle sits at P5 with 810 MHz memory, 20% stays at P8), and at P2 it runs full boost whatever the load: the default
// (one polling thread) draws ~165 W against ~30 W idle, one block spinning ~166 W, every SM ~268 W.
struct WarmLoad {
    int blocks = 1, threads = 32, spin = 0;
    double on_ms = 50.0, off_ms = 0.0;
};

// "blocks,threads,on_ms,off_ms,spin" (missing fields keep their defaults); false on a malformed or out-of-range spec
bool parse_load(const char * s, WarmLoad & w) {
    WarmLoad v;
    if (s && *s) {
        int spin = v.spin;
        const int n = sscanf(s, "%d,%d,%lf,%lf,%d", &v.blocks, &v.threads, &v.on_ms, &v.off_ms, &spin);
        v.spin = spin;
        if (n < 1 || v.blocks < 1 || v.blocks > 1024 || v.threads < 1 || v.threads > 1024 || !(v.on_ms > 0.0) ||
            v.on_ms > 1000.0 || !(v.off_ms >= 0.0) || v.off_ms > 10000.0 || (v.spin != 0 && v.spin != 1))
            return false;
    }
    w = v;
    return true;
}

class Warmer {
    using Clock = std::chrono::steady_clock;

  public:
    Warmer(double seconds, const WarmLoad & load) : seconds_(seconds), load_(load) {}
    ~Warmer() {
        {
            std::lock_guard<std::mutex> lk(m_);
            quit_ = true;
            armed_ = false;
            if (inflight_) *stop_h_ = 1;
        }
        cv_.notify_all();
        if (th_.joinable()) th_.join();
        if (ev_) cudaEventDestroy(ev_);
        if (st_) cudaStreamDestroy(st_);
        if (stop_h_) cudaFreeHost((void *) stop_h_);
    }
    double seconds() const { return seconds_; }
    const WarmLoad & load() const { return load_; }

    // (re)starts the window: the load runs until seconds() from now, or until pause()
    void arm() {
        std::lock_guard<std::mutex> lk(m_);
        if (failed_ || (!th_.joinable() && !start())) return;
        armed_ = true;
        until_ = Clock::now() + std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(seconds_));
        cv_.notify_all();
    }

    // stops the load; returns once no warm kernel runs (the wait in microseconds)
    double pause() {
        std::unique_lock<std::mutex> lk(m_);
        armed_ = false;
        if (!inflight_) return 0.0;
        *stop_h_ = 1;
        const cudaEvent_t ev = ev_;
        lk.unlock();
        // spin on the event (the warm thread itself waits with a blocking sync): no OS wake-up latency here. A query
        // that finds the kernel running is not an engine error: clear it, unless an error was already pending.
        const auto t0 = Clock::now();
        const cudaError_t before = cudaPeekAtLastError();
        while (cudaEventQuery(ev) == cudaErrorNotReady) {
        }
        if (before == cudaSuccess) cudaGetLastError();
        return std::chrono::duration<double, std::micro>(Clock::now() - t0).count();
    }

  private:
    // CUDA resources and the thread, on the first arm (under m_); on a failure the warmer stays off
    bool start() {
        const cudaError_t before = cudaPeekAtLastError();
        int dev = 0;
        void * p = nullptr;
        cudaError_t e = cudaGetDevice(&dev);
        if (e == cudaSuccess) e = cudaHostAlloc(&p, sizeof(unsigned), cudaHostAllocMapped);
        if (e == cudaSuccess) {
            stop_h_ = static_cast<volatile unsigned *>(p);
            *stop_h_ = 0;
            e = cudaHostGetDevicePointer(reinterpret_cast<void **>(&stop_d_), p, 0);
        }
        if (e == cudaSuccess) e = cudaStreamCreateWithFlags(&st_, cudaStreamNonBlocking);
        if (e == cudaSuccess) e = cudaEventCreateWithFlags(&ev_, cudaEventDisableTiming | cudaEventBlockingSync);
        if (e != cudaSuccess) {
            fprintf(stderr, "[mk] keep-warm: off (%s)\n", cudaGetErrorString(e));
            if (before == cudaSuccess) cudaGetLastError(); // (not an engine error)
            failed_ = true;
            return false;
        }
        try {
            th_ = std::thread([this, dev] { run(dev); });
        } catch (const std::exception & ex) { // (no thread to be had: this runs inside a C API call, nothing may escape it)
            fprintf(stderr, "[mk] keep-warm: off (%s)\n", ex.what());
            failed_ = true;
            return false;
        }
        return true;
    }

    void run(int dev) {
        cudaSetDevice(dev);
        std::unique_lock<std::mutex> lk(m_);
        for (;;) {
            cv_.wait(lk, [&] { return quit_ || armed_; });
            if (quit_) return;
            const auto now = Clock::now();
            if (now >= until_) { // the window is over: idle until the next arm
                armed_ = false;
                continue;
            }
            const double left = std::chrono::duration<double, std::milli>(until_ - now).count();
            *stop_h_ = 0;
            int e = MKEngine::warm_launch(st_, stop_d_, std::min(load_.on_ms, left), load_.blocks, load_.threads, load_.spin);
            if (e == 0) e = (int) cudaEventRecord(ev_, st_);
            if (e != 0) {
                fprintf(stderr, "[mk] keep-warm: off (%s)\n", cudaGetErrorString((cudaError_t) e));
                failed_ = true;
                armed_ = false;
                return;
            }
            inflight_ = true; // (launched and recorded under m_: pause() either sees this or stops the next launch)
            lk.unlock();
            cudaEventSynchronize(ev_);
            lk.lock();
            inflight_ = false;
            if (load_.off_ms > 0.0)
                cv_.wait_for(lk, std::chrono::duration<double, std::milli>(load_.off_ms), [&] { return quit_ || !armed_; });
        }
    }

    const double seconds_;
    const WarmLoad load_;
    std::mutex m_;
    std::condition_variable cv_;
    std::thread th_;
    bool quit_ = false, armed_ = false, inflight_ = false, failed_ = false;
    Clock::time_point until_{};
    volatile unsigned * stop_h_ = nullptr; // host-mapped stop word (stop_d_: its device address)
    unsigned * stop_d_ = nullptr;
    cudaStream_t st_ = nullptr;
    cudaEvent_t ev_ = nullptr;
};

struct Handle {
    MKEngine * eng = nullptr;
    llama_model * lm = nullptr;
    const llama_vocab * vocab = nullptr;
    Warmer * warm = nullptr; // MK_KEEP_WARM_S > 0
    std::string err;
};
thread_local std::string g_err;

// the engine of hp, after stopping the keep-warm load: every call that runs GPU work gets the engine through here
MKEngine * gpu_engine(void * hp) {
    auto * h = static_cast<Handle *>(hp);
    if (h->warm) h->warm->pause();
    return h->eng;
}

// The reset after a CUDA error failed too: the error repeats after it was cleared (mk_cuda_fail's test for a sticky
// one), and a later retry would run calls a caller planned against the pre-error sequence() on an emptied engine.
[[noreturn]] void reset_failed(const char * why) {
    fprintf(stderr, "[mk] the reset after a CUDA error failed too (%s): exiting with code 4\n", why);
    fflush(stdout);
    fflush(stderr);
    std::_Exit(4);
}

// Runs an engine call; an exception becomes -1 with mk_last_error(). A CUDA failure (MKCudaError: the engine
// threw instead of exiting) leaves the sequence state undefined, so the engine is reset: the next prompt is
// prefilled in full. (Sticky CUDA errors never get here: the engine exits with code 4, as when that reset fails.)
template <class F> int guarded(void * hp, F f) {
    auto * h = static_cast<Handle *>(hp);
    try {
        return f(gpu_engine(hp));
    } catch (const MKCudaError & e) {
        g_err = e.what();
        try {
            if (h->eng->side_active()) h->eng->side_drop(); // (a side request's kept sequence goes too: never restored over an undefined state)
            else h->eng->reset();
        } catch (const std::exception & e2) {
            reset_failed(e2.what());
        } catch (...) {
            reset_failed("unknown exception");
        }
        return -1;
    } catch (const MKCancelled & e) { // (mk_cancel; the engine put its state back itself)
        g_err = e.what();
        return -2;
    } catch (const std::exception & e) {
        g_err = e.what();
        return -1;
    }
}

// a caller's array of n ints (n <= 0: none)
std::vector<int> vec(const int * p, int n) { return n > 0 ? std::vector<int>(p, p + n) : std::vector<int>(); }

// libllama's tokenizer calls throw (std::out_of_range for an id outside the vocabulary, bad_alloc): INT_MIN, which is
// also llama_tokenize's own error value, with mk_last_error
template <class F> int vocab_call(F f) {
    try {
        return f();
    } catch (const std::exception & e) {
        g_err = e.what();
        return INT_MIN;
    }
}
}

extern "C" {

const char * mk_last_error() { return g_err.c_str(); }

// mtp: the MTP layer's source (MKEngine's mtp): NULL / "" the model file (or MK_NO_MTP / MK_MTP), a GGUF path, "none"
void * mk_new_mtp(const char * model, int ctx, int n_draft, const char * draft_vocab, int kv_mode, const char * mtp) {
    Handle * h = nullptr;
    try {
        llama_backend_init();
        llama_model_params mp = llama_model_default_params();
        mp.vocab_only = true;
        h = new Handle;
        h->lm = llama_model_load_from_file(model, mp);
        if (!h->lm) throw std::runtime_error("failed to load the vocabulary");
        h->vocab = llama_model_get_vocab(h->lm);
        h->eng = new MKEngine(model, ctx, n_draft, draft_vocab ? draft_vocab : "", kv_mode, mtp ? mtp : "");
        // keep-warm (MK_KEEP_WARM_S seconds, default 120; 0 = off): armed by mk_keep_warm only
        const char * ks = getenv("MK_KEEP_WARM_S");
        const double sec = ks && *ks ? atof(ks) : 120.0;
        WarmLoad load;
        if (!parse_load(getenv("MK_KEEP_WARM_LOAD"), load))
            fprintf(stderr, "[mk] keep-warm: MK_KEEP_WARM_LOAD '%s' ignored (blocks,threads,on_ms,off_ms,spin)\n",
                    getenv("MK_KEEP_WARM_LOAD"));
        if (sec > 0.0) {
            h->warm = new Warmer(sec, load);
            fprintf(stderr, "[mk] keep-warm: %g s after each request (MK_KEEP_WARM_S; 0 = off), %d x %d threads, %g ms on / %g ms off%s\n",
                    sec, load.blocks, load.threads, load.on_ms, load.off_ms, load.spin ? ", spinning" : "");
        } else {
            fprintf(stderr, "[mk] keep-warm: off (MK_KEEP_WARM_S=%s)\n", ks);
        }
        return h;
    } catch (const std::exception & e) {
        g_err = e.what();
        if (h) { // (what was made before the failure, in mk_free's order)
            delete h->warm;
            delete h->eng;
            if (h->lm) llama_model_free(h->lm);
            delete h;
        }
        return nullptr;
    }
}

void * mk_new(const char * model, int ctx, int n_draft, const char * draft_vocab, int kv_mode) {
    return mk_new_mtp(model, ctx, n_draft, draft_vocab, kv_mode, nullptr);
}

// open-jet's mk_new2: kv_ctx > 0 = the engine's own sequence's positions (parallel slots hold their own; ctx then only
// bounds what the caller expects of a slot)
void * mk_new2(const char * model, int ctx, int n_draft, const char * draft_vocab, int kv_mode, int kv_ctx) {
    return mk_new_mtp(model, kv_ctx > 0 ? kv_ctx : ctx, n_draft, draft_vocab, kv_mode, nullptr);
}

// ---- parallel slots (open-jet's slot API; MKEngine::slots_open) ----
// A CUDA error (-1) fails the slots the call ran on: mk_slot_prefill / mk_slots_step(_mixed) refuse them (-1) until
// mk_slot_begin2 starts them from scratch (reused 0); the engine's own sequence is not reset.
// n slots of ctx positions each (0: the most that fit); returns ctx (a slot's positions), or -1
int mk_slots_open2(void * hp, int n, int ctx, int seg_tokens) {
    try {
        return gpu_engine(hp)->slots_open(n, ctx, seg_tokens);
    } catch (const std::exception & e) {
        g_err = e.what();
        return -1;
    }
}
int mk_slots_open(void * hp, int n, int ctx) { return mk_slots_open2(hp, n, ctx, 0); }

// slot <- the prompt (temp 0: greedy), reusing what it can: returns the reused count (or -1); mk_slot_prefill feeds the rest
// (MK_EXACT: a bit-exact follow-up needs two calls, see MKEngine::slot_begin in qwen38_mk.h)
int mk_slot_begin2(void * hp, int slot, const int * toks, int n, float temp, float top_p, int top_k, float min_p, unsigned long long seed,
                   int share_len, int pin) {
    try {
        return gpu_engine(hp)->slot_begin(slot, vec(toks, n), temp, top_p, top_k, min_p, seed, share_len, pin != 0);
    } catch (const std::exception & e) {
        g_err = e.what();
        return -1;
    }
}
int mk_slot_begin(void * hp, int slot, const int * toks, int n, float temp, float top_p, int top_k, float min_p, unsigned long long seed) {
    return mk_slot_begin2(hp, slot, toks, n, temp, top_p, top_k, min_p, seed, 0, 0);
}

// feeds up to budget prompt tokens: returns how many remain (0: ready to step; -1: error; -2: mk_cancel stopped it, the
// rest from the sequence's checkpoint on is queued again)
int mk_slot_prefill(void * hp, int slot, int budget) {
    try {
        return gpu_engine(hp)->slot_prefill(slot, budget);
    } catch (const MKCancelled & e) {
        g_err = e.what();
        return -2;
    } catch (const std::exception & e) {
        g_err = e.what();
        return -1;
    }
}

// one step of the n slots: out[i * per + j] = the j-th token slot i emitted, counts[i] how many (0: its context is full);
// a slot listed twice fails the call (-1) before anything runs
static void slots_out(const std::vector<std::vector<int>> & em, int n, int * out, int per, int * counts) {
    for (int i = 0; i < n; i++) {
        counts[i] = std::min((int) em[i].size(), per);
        for (int j = 0; j < counts[i]; j++) out[i * per + j] = em[i][j];
    }
}
int mk_slots_step(void * hp, const int * slots, int n, int * out, int per, int * counts) {
    try {
        slots_out(gpu_engine(hp)->slots_step(vec(slots, n)), n, out, per, counts);
        return 0;
    } catch (const std::exception & e) {
        g_err = e.what();
        return -1;
    }
}

// the next budget tokens of slot pslot's prompt, then mk_slots_step (a failed chunk steps nothing); *remain <- the
// prompt tokens left
int mk_slots_step_mixed(void * hp, int pslot, int budget, const int * slots, int n, int * out, int per, int * counts, int * remain) {
    try {
        int rem = 0;
        slots_out(gpu_engine(hp)->slots_step_mixed(pslot, budget, vec(slots, n), rem), n, out, per, counts);
        *remain = rem;
        return 0;
    } catch (const std::exception & e) {
        g_err = e.what();
        return -1;
    }
}

int mk_slot_segment(void * hp, int slot) { return static_cast<Handle *>(hp)->eng->slot_segment(slot); }

// (test) the slot's sequence -> out (up to cap); returns its length
int mk_slot_sequence(void * hp, int slot, int * out, int cap) {
    try {
        const auto s = gpu_engine(hp)->slot_sequence(slot); // (lane_in / lane_out synchronise)
        for (int i = 0; i < std::min(cap, (int) s.size()); i++) out[i] = s[i];
        return (int) s.size();
    } catch (const std::exception & e) {
        g_err = e.what();
        return -1;
    }
}

// (test) the hidden state of the slot's last processed position -> out[0 .. n), returns its size
int mk_slot_h(void * hp, int slot, float * out, int n) {
    try {
        const auto h = gpu_engine(hp)->slot_h(slot);
        for (int i = 0; i < std::min(n, (int) h.size()); i++) out[i] = h[i];
        return (int) h.size();
    } catch (const std::exception & e) {
        g_err = e.what();
        return -1;
    }
}

// 1: the engine has an MTP layer (drafting), 0: plain decoding (depth 0)
int mk_has_mtp(void * hp) { return static_cast<Handle *>(hp)->eng->has_mtp() ? 1 : 0; }

void mk_free(void * hp) {
    auto * h = static_cast<Handle *>(hp);
    delete h->warm; // (stops its thread before the engine goes)
    delete h->eng;  // (the destructor does not throw)
    llama_model_free(h->lm);
    delete h;
}

// chat template stored in the GGUF (tokenizer.chat_template); returns its length, copies up to cap (-1: none, or an
// error with mk_last_error)
int mk_chat_template(void * hp, char * buf, int cap) {
    try {
        auto * h = static_cast<Handle *>(hp);
        const char * t = llama_model_chat_template(h->lm, nullptr);
        if (!t) return -1;
        const int n = (int) strlen(t);
        if (buf && cap > 0) memcpy(buf, t, std::min(n, cap));
        return n;
    } catch (const std::exception & e) {
        g_err = e.what();
        return -1;
    }
}

// text -> token ids (special tokens such as <|im_start|> parsed); returns the count (negative: -needed; INT_MIN: error)
int mk_tokenize(void * hp, const char * text, int len, int * out, int cap) {
    auto * h = static_cast<Handle *>(hp);
    return vocab_call([&] { return llama_tokenize(h->vocab, text, len, out, cap, false, true); });
}

// mk_tokenize with llama_tokenize's flags: add_special: the vocabulary's BOS / EOS where it adds them; parse_special: the
// text of control tokens (<|im_start|>, <|im_end|>, ...) becomes those tokens, else it is plain text (user-defined tokens
// such as <think> are matched either way, as in llama.cpp and Hugging Face tokenizers)
int mk_tokenize2(void * hp, const char * text, int len, int * out, int cap, int add_special, int parse_special) {
    auto * h = static_cast<Handle *>(hp);
    return vocab_call([&] { return llama_tokenize(h->vocab, text, len, out, cap, add_special != 0, parse_special != 0); });
}

// a token's llama_token_attr bits (LLAMA_TOKEN_ATTR_CONTROL 8, _USER_DEFINED 16, _UNKNOWN 1, ...; the control and unknown
// ones are what parse_special decides on); -1 for an id outside the vocabulary
int mk_token_attr(void * hp, int tok) {
    const llama_vocab * vocab = static_cast<Handle *>(hp)->vocab;
    if (tok < 0 || tok >= llama_vocab_n_tokens(vocab)) return -1;
    return (int) llama_vocab_get_attr(vocab, tok);
}

// token -> its bytes (special tokens rendered as text); returns the byte count (INT_MIN: an id outside the vocabulary)
int mk_piece(void * hp, int tok, char * buf, int cap) {
    auto * h = static_cast<Handle *>(hp);
    return vocab_call([&] { return llama_token_to_piece(h->vocab, tok, buf, cap, 0, true); });
}

int mk_is_eog(void * hp, int tok) { return llama_vocab_is_eog(static_cast<Handle *>(hp)->vocab, tok) ? 1 : 0; }

// the vocabulary's size: valid token ids are 0 .. n - 1
int mk_n_vocab(void * hp) { return llama_vocab_n_tokens(static_cast<Handle *>(hp)->vocab); }

int mk_set_sampling(void * hp, float temp, float top_p, int top_k, float min_p, unsigned long long seed) {
    try {
        static_cast<Handle *>(hp)->eng->set_sampling(temp, top_p, top_k, min_p, seed);
        return 0;
    } catch (const std::exception & e) {
        g_err = e.what();
        return -1;
    }
}

// presence / frequency penalty (MKEngine::set_penalty: the engine's own sequence, never the slots); returns 0, or -1
int mk_set_penalty(void * hp, float presence, float frequency) {
    try {
        gpu_engine(hp)->set_penalty(presence, frequency); // (the first non-zero penalty allocates its counts)
        return 0;
    } catch (const std::exception & e) {
        g_err = e.what();
        return -1;
    }
}

// draft-depth policy "key=value,..." (MKEngine::set_policy); returns 0, or -1 on a bad spec
int mk_set_policy(void * hp, const char * spec) {
    try { // (dq4= copies to the draft head's constants: drafts only, the sequence needs no reset after a failure)
        gpu_engine(hp)->set_policy(spec ? spec : "");
        return 0;
    } catch (const std::exception & e) {
        g_err = e.what();
        return -1;
    }
}

// appends tokens to the sequence; returns the first generated token (negative on error)
int mk_prefill(void * hp, const int * toks, int n) {
    return guarded(hp, [&](MKEngine * e) { return e->prefill(vec(toks, n)); });
}

// mk_prefill with the rewind checkpoint saved at absolute position ck_at (inside this prompt) instead of its end
int mk_prefill_ck(void * hp, const int * toks, int n, int ck_at) {
    return guarded(hp, [&](MKEngine * e) { return e->prefill(vec(toks, n), ck_at); });
}

// mk_prefill with checkpoints at the absolute positions cks[0..ncks) (ck_exact: the one kept exact, or -1;
// MKEngine::prefill(tokens, cks, ck_exact))
int mk_prefill_cks(void * hp, const int * toks, int n, const int * cks, int ncks, int ck_exact) {
    return guarded(hp, [&](MKEngine * e) {
        return e->prefill(vec(toks, n), vec(cks, ncks), ck_exact);
    });
}

// mk_prefill_cks with nimg images: image i covers the absolute positions pos[i] .. pos[i] + nx[i] * ny[i] - 1 with the
// rows embd[i] ([nx[i] * ny[i]][mk_n_embd()] fp32); MKEngine::prefill(tokens, cks, ck_exact, images)
int mk_prefill_img(void * hp, const int * toks, int n, const int * cks, int ncks, int ck_exact, const int * pos,
                   const int * nx, const int * ny, const float * const * embd, int nimg) {
    return guarded(hp, [&](MKEngine * e) {
        std::vector<MKImage> im;
        for (int i = 0; i < nimg; i++) im.push_back({pos[i], nx[i], ny[i], embd[i]});
        return e->prefill(vec(toks, n), vec(cks, ncks), ck_exact, im);
    });
}

// width of an image row (the model's embedding size)
int mk_n_embd() { return MKEngine::n_embd(); }

// on 1: a prefill running on another thread (or the next one) stops at its next chunk boundary and returns -2, with the
// sequence back at the last checkpoint before it (MKEngine::cancel; a host flag, no GPU work, never waits); 0 clears it
// (every prefill clears it when it returns). The server calls it when the client disconnects during a long prompt.
int mk_cancel(void * hp, int on) {
    static_cast<Handle *>(hp)->eng->cancel(on != 0);
    return 0;
}

// progress of a prefill running on another thread: *done of the *total prompt positions of its batched part
// (MKEngine::prefill_progress; host counters, never waits). The server shows it while a long prompt is processed.
void mk_prefill_progress(void * hp, int * done, int * total) {
    static_cast<Handle *>(hp)->eng->prefill_progress(done, total);
}

// one decode cycle; writes 1..n_draft+1 tokens; returns the count (negative on error)
int mk_step(void * hp, int * out, int cap) {
    return guarded(hp, [&](MKEngine * e) {
        std::vector<int> v;
        e->step(v);
        if ((int) v.size() > cap) throw std::runtime_error("mk_step: " + std::to_string(v.size()) + " tokens, the buffer holds " + std::to_string(cap));
        for (int i = 0; i < (int) v.size(); i++) out[i] = v[i];
        return (int) v.size();
    });
}

// (an error is only visible through mk_last_error)
void mk_reset(void * hp) {
    guarded(hp, [](MKEngine * e) {
        e->reset();
        return 0;
    });
}

int mk_truncate(void * hp, int len) {
    return guarded(hp, [&](MKEngine * e) {
        e->truncate(len);
        return 0;
    });
}

// copies the sequence so far (up to cap tokens); returns its length
int mk_sequence(void * hp, int * out, int cap) {
    const auto & s = static_cast<Handle *>(hp)->eng->sequence();
    const int n = (int) s.size();
    if (out)
        for (int i = 0; i < std::min(n, cap); i++) out[i] = s[i];
    return n;
}

int mk_committed(void * hp) { return static_cast<Handle *>(hp)->eng->committed(); }
int mk_checkpoint_pos(void * hp) { return static_cast<Handle *>(hp)->eng->checkpoint_pos(); }
int mk_rewind_to_checkpoint(void * hp) {
    return guarded(hp, [](MKEngine * e) { return e->rewind_to_checkpoint() ? 0 : -1; });
}
// positions of the checkpoints (ascending, up to cap); returns their count
int mk_checkpoints(void * hp, int * out, int cap) {
    const auto v = static_cast<Handle *>(hp)->eng->checkpoints();
    if (out)
        for (int i = 0; i < std::min((int) v.size(), cap); i++) out[i] = v[i];
    return (int) v.size();
}
// restores the checkpoint at pos: 0, or -1 (none there)
int mk_rewind_to(void * hp, int pos) {
    return guarded(hp, [&](MKEngine * e) { return e->rewind_to(pos) ? 0 : -1; });
}
// pins the checkpoints at the positions pos[0..n) (each: the last one in (P - 32, P], now or as the next prefill saves
// it; MKEngine::ck_pin): never evicted, parked with the sequence; n = 0 pins nothing new. Returns how many were pinned
// now (-1 error)
int mk_ck_pin(void * hp, const int * pos, int n) {
    try { // (host bookkeeping only: no GPU work)
        return static_cast<Handle *>(hp)->eng->ck_pin(n > 0 ? std::vector<int>(pos, pos + n) : std::vector<int>{});
    } catch (const std::exception & e) {
        g_err = e.what();
        return -1;
    }
}
// positions of the pinned checkpoints (ascending, up to cap); returns their count
int mk_ck_pinned(void * hp, int * out, int cap) {
    const auto v = static_cast<Handle *>(hp)->eng->ck_pinned();
    if (out)
        for (int i = 0; i < std::min((int) v.size(), cap); i++) out[i] = v[i];
    return (int) v.size();
}
// 1 if the engine runs MK_EXACT (exact reuse: restore a checkpoint instead of appending to decoded rows)
int mk_exact(void * hp) { return static_cast<Handle *>(hp)->eng->exact() ? 1 : 0; }
// test helper: MKEngine::draft_vocab (the device's draft-vocabulary rows, checked against the host's list) into out (up
// to cap ids); returns the count (0: no draft vocabulary), -1 on error
int mk_draft_vocab(void * hp, int * out, int cap) {
    return guarded(hp, [&](MKEngine * e) {
        const auto v = e->draft_vocab();
        if (out)
            for (int i = 0; i < std::min((int) v.size(), cap); i++) out[i] = v[i];
        return (int) v.size();
    });
}
// test helper: MKEngine::state_digest into out (up to cap words); returns the word count (negative on error)
int mk_state_digest(void * hp, unsigned long long * out, int cap) {
    return guarded(hp, [&](MKEngine * e) {
        const auto v = e->state_digest();
        if (out)
            for (int i = 0; i < std::min((int) v.size(), cap); i++) out[i] = v[i];
        return (int) v.size();
    });
}

// ---- conversation state in host memory (MKEngine::store_budget, side_begin, park; see qwen38_mk.h) ----
// host store budget in bytes (0: off; the default); returns 0
int mk_store_budget(void * hp, long long bytes) {
    return guarded(hp, [&](MKEngine * e) {
        e->store_budget((size_t) std::max(bytes, 0LL));
        return 0;
    });
}
// allocates host-store chunks now until `bytes` are free (within the budget); returns 0
int mk_store_reserve(void * hp, long long bytes) {
    return guarded(hp, [&](MKEngine * e) {
        e->store_reserve((size_t) std::max(bytes, 0LL));
        return 0;
    });
}
// out[0..9]: budget, allocated, used bytes, parked entries, the last side_end / park / unpark's bytes and microseconds,
// the side request's copy-out microseconds, bytes per KV token, the last side_end_park's park of the side request's
// conversation: bytes and microseconds; returns the count written (up to cap)
int mk_store_info(void * hp, long long * out, int cap) {
    const MKEngine * e = static_cast<Handle *>(hp)->eng;
    const MKEngine::StoreInfo s = e->store_info();
    size_t pt, rings, state, ck;
    e->kv_sizes(&pt, &rings, &state, &ck);
    const long long v[] = {(long long) s.budget, (long long) s.allocated, (long long) s.used, s.parked, (long long) s.last_bytes,
                           (long long) (s.last_ms * 1000.0), (long long) (s.side_ms * 1000.0), (long long) pt,
                           (long long) s.side_park_bytes, (long long) (s.side_park_ms * 1000.0)};
    const int n = std::min(cap, (int) (sizeof v / sizeof v[0]));
    for (int i = 0; i < n; i++) out[i] = v[i];
    return n;
}
// bytes: per KV token, the window rings, the recurrent state, one checkpoint (MKEngine::kv_sizes)
void mk_kv_sizes(void * hp, long long * per_token, long long * rings, long long * state, long long * checkpoint) {
    size_t a, b, c, d;
    static_cast<Handle *>(hp)->eng->kv_sizes(&a, &b, &c, &d);
    *per_token = (long long) a;
    *rings = (long long) b;
    *state = (long long) c;
    *checkpoint = (long long) d;
}
// side request: 0 started, 1 no room in the host store (nothing changed), -1 error
int mk_side_begin(void * hp) {
    return guarded(hp, [](MKEngine * e) { return e->side_begin() ? 0 : 1; });
}
// 1 the kept sequence is back, 0 it was dropped (the store ran full; the engine is reset), -1 error (dropped too).
// Then (as after mk_side_unpark, mk_unpark) continue with a prefill: mk_step right away only where qwen38_mk.h
// says (park)
int mk_side_end(void * hp) {
    return guarded(hp, [](MKEngine * e) { return e->side_end(); });
}
// mk_side_end, with the side request's own conversation parked first (as mk_park(max_checkpoints)): *id its entry (0:
// not parked: empty, or no room). 1 the kept sequence is back; 0 it was lost (the store ran full during the side
// request): the side request's conversation stays as the engine's sequence; -1 error (a CUDA error drops both)
int mk_side_end_park(void * hp, int max_checkpoints, int * id) {
    *id = 0;
    return guarded(hp, [&](MKEngine * e) { return e->side_end_park(max_checkpoints, id); });
}
// right after mk_side_begin: the side request continues parked conversation id (the engine becomes it, the entry is
// freed; the kept sequence stays aside): 0, 1 (no room in the store: nothing changed), -1 (no such entry / error)
int mk_side_unpark(void * hp, int id) {
    return guarded(hp, [&](MKEngine * e) {
        const int r = e->side_unpark(id);
        if (r < 0) g_err = "no parked conversation " + std::to_string(id);
        return r < 0 ? -1 : r == 1 ? 0 : 1;
    });
}
int mk_side_drop(void * hp) {
    return guarded(hp, [](MKEngine * e) {
        e->side_drop();
        return 0;
    });
}
int mk_side_active(void * hp) { return static_cast<Handle *>(hp)->eng->side_active() ? 1 : 0; }
// parks a snapshot of the sequence (its newest max_checkpoints checkpoints and, if max_checkpoints > 0, the pinned ones):
// its id (> 0), 0 (empty sequence), -3 (no room in the store), -1 error
int mk_park(void * hp, int max_checkpoints) {
    return guarded(hp, [&](MKEngine * e) {
        const int id = e->park(max_checkpoints);
        return id < 0 ? -3 : id;
    });
}
// the engine becomes parked conversation id (entry freed): 0, or -1 (no such entry / error); then a prefill
// (mk_side_end)
int mk_unpark(void * hp, int id) {
    return guarded(hp, [&](MKEngine * e) {
        if (e->unpark(id)) return 0;
        g_err = "no parked conversation " + std::to_string(id);
        return -1;
    });
}
int mk_park_drop(void * hp, int id) {
    return guarded(hp, [&](MKEngine * e) { return e->park_drop(id) ? 0 : -1; });
}
long long mk_park_need(void * hp, int max_checkpoints) {
    return (long long) static_cast<Handle *>(hp)->eng->park_need(max_checkpoints);
}
long long mk_park_bytes(void * hp, int id) { return (long long) static_cast<Handle *>(hp)->eng->park_bytes(id); }

// keep-warm (MK_KEEP_WARM_S): on 1 (re)starts the window now (the server: once ready and after every request), 0 stops
// the load now. Returns the window in seconds (0: off).
double mk_keep_warm(void * hp, int on) {
    auto * h = static_cast<Handle *>(hp);
    if (!h->warm) return 0.0;
    if (on) h->warm->arm();
    else h->warm->pause();
    return h->warm->seconds();
}

// measurement helpers: a keep-warm load without an engine (window in seconds, load as MK_KEEP_WARM_LOAD; null: the
// default); mk_warm_set(w, 1) starts the window, mk_warm_set(w, 0) stops the load and returns the microseconds it
// waited for the running kernel to end (what a request waits)
void * mk_warm_new(double seconds, const char * load) {
    WarmLoad w;
    if (!(seconds > 0.0) || !parse_load(load, w)) {
        g_err = "mk_warm_new: seconds must be > 0 and load 'blocks,threads,on_ms,off_ms,spin'";
        return nullptr;
    }
    return new Warmer(seconds, w);
}
double mk_warm_set(void * w, int on) {
    auto * x = static_cast<Warmer *>(w);
    if (on) {
        x->arm();
        return 0.0;
    }
    return x->pause();
}
void mk_warm_free(void * w) { delete static_cast<Warmer *>(w); }

void mk_stats(void * hp, long * cycles, long * drafted, long * accepted) {
    const auto & s = static_cast<Handle *>(hp)->eng->stats;
    *cycles = s.cycles;
    *drafted = s.drafted;
    *accepted = s.accepted;
}

// the last step: tokens verified (T), drafts produced for the next step, drafts accepted
void mk_last(void * hp, int * T, int * nd, int * acc) {
    const auto & s = static_cast<Handle *>(hp)->eng->stats;
    *T = s.last_T;
    *nd = s.last_nd;
    *acc = s.last_acc;
}

// the library's build as JSON: {"commit", "flags" (the engine's MKFLAGS), "cuda" (the runtime it was built against),
// "arch" (nvcc's GPU architectures, when nvcc says)} for the server's /props; commit and flags are "unknown" when the
// build did not write mk_build_info.h
const char * mk_build_info() {
    static const std::string s = [] {
        std::string j = "{\"commit\": \"" MK_BUILD_COMMIT "\", \"flags\": \"" MK_BUILD_FLAGS "\", \"cuda\": \"" +
                        std::to_string(CUDART_VERSION / 1000) + "." + std::to_string(CUDART_VERSION % 1000 / 10) + "\"";
#ifdef __CUDA_ARCH_LIST__
        j += ", \"arch\": \"" MK_STR(__CUDA_ARCH_LIST__) "\"";
#endif
        return j + "}";
    }();
    return s.c_str();
}

} // extern "C"
