// C API over MKEngine plus libllama's tokenizer (vocab only), for the HTTP server (server/mk_server.py).
#include <algorithm>
#include <cstring>
#include <exception>
#include <stdexcept>
#include <string>
#include <vector>

#include "llama.h"
#include "qwen38_mk.h"

namespace {
struct Handle {
    MKEngine * eng = nullptr;
    llama_model * lm = nullptr;
    const llama_vocab * vocab = nullptr;
    std::string err;
};
thread_local std::string g_err;
}

// MSVC exports nothing from a DLL unless asked
#ifdef _WIN32
#define MK_API __declspec(dllexport)
#else
#define MK_API
#endif

extern "C" {

MK_API const char * mk_last_error() { return g_err.c_str(); }
MK_API void * mk_new2(const char * model, int ctx, int n_draft, const char * draft_vocab, int kv_mode, int kv_ctx);
MK_API int mk_slots_open2(void * hp, int n, int ctx, int seg_tokens);
MK_API int mk_slot_begin2(void * hp, int slot, const int * toks, int n, float temp, float top_p, int top_k, float min_p, unsigned long long seed,
                   int share_len, int pin);

MK_API void * mk_new(const char * model, int ctx, int n_draft, const char * draft_vocab, int kv_mode) {
    return mk_new2(model, ctx, n_draft, draft_vocab, kv_mode, 0);
}

// as mk_new, with kv_ctx: positions of the engine's own K/V cache (0: ctx; parallel slots have their own)
MK_API void * mk_new2(const char * model, int ctx, int n_draft, const char * draft_vocab, int kv_mode, int kv_ctx) {
    try {
        llama_backend_init();
        llama_model_params mp = llama_model_default_params();
        mp.vocab_only = true;
        auto * h = new Handle;
        h->lm = llama_model_load_from_file(model, mp);
        if (!h->lm) throw std::runtime_error("failed to load the vocabulary");
        h->vocab = llama_model_get_vocab(h->lm);
        h->eng = new MKEngine(model, ctx, n_draft, draft_vocab ? draft_vocab : "", kv_mode, kv_ctx);
        return h;
    } catch (const std::exception & e) {
        g_err = e.what();
        return nullptr;
    }
}

MK_API void mk_free(void * hp) {
    auto * h = static_cast<Handle *>(hp);
    delete h->eng;
    llama_model_free(h->lm);
    delete h;
}

// chat template stored in the GGUF (tokenizer.chat_template); returns its length, copies up to cap
MK_API int mk_chat_template(void * hp, char * buf, int cap) {
    auto * h = static_cast<Handle *>(hp);
    const char * t = llama_model_chat_template(h->lm, nullptr);
    if (!t) return -1;
    const int n = (int) strlen(t);
    if (buf && cap > 0) memcpy(buf, t, std::min(n, cap));
    return n;
}

// text -> token ids (special tokens such as <|im_start|> parsed); returns the count (negative: -needed)
MK_API int mk_tokenize(void * hp, const char * text, int len, int * out, int cap) {
    auto * h = static_cast<Handle *>(hp);
    return llama_tokenize(h->vocab, text, len, out, cap, false, true);
}

// token -> its bytes (special tokens rendered as text); returns the byte count
MK_API int mk_piece(void * hp, int tok, char * buf, int cap) {
    auto * h = static_cast<Handle *>(hp);
    return llama_token_to_piece(h->vocab, tok, buf, cap, 0, true);
}

MK_API int mk_is_eog(void * hp, int tok) { return llama_vocab_is_eog(static_cast<Handle *>(hp)->vocab, tok) ? 1 : 0; }

MK_API int mk_set_sampling(void * hp, float temp, float top_p, int top_k, float min_p, unsigned long long seed) {
    try {
        static_cast<Handle *>(hp)->eng->set_sampling(temp, top_p, top_k, min_p, seed);
        return 0;
    } catch (const std::exception & e) {
        g_err = e.what();
        return -1;
    }
}

// appends tokens to the sequence; returns the first generated token (negative on error)
MK_API int mk_prefill(void * hp, const int * toks, int n) {
    try {
        return static_cast<Handle *>(hp)->eng->prefill(std::vector<int>(toks, toks + n));
    } catch (const std::exception & e) {
        g_err = e.what();
        return -1;
    }
}

// one decode cycle; writes 1..n_draft+1 tokens; returns the count (negative on error)
MK_API int mk_step(void * hp, int * out, int cap) {
    try {
        std::vector<int> v;
        static_cast<Handle *>(hp)->eng->step(v);
        const int n = std::min((int) v.size(), cap);
        for (int i = 0; i < n; i++) out[i] = v[i];
        return n;
    } catch (const std::exception & e) {
        g_err = e.what();
        return -1;
    }
}

MK_API void mk_reset(void * hp) { static_cast<Handle *>(hp)->eng->reset(); }

MK_API int mk_truncate(void * hp, int len) {
    try {
        static_cast<Handle *>(hp)->eng->truncate(len);
        return 0;
    } catch (const std::exception & e) {
        g_err = e.what();
        return -1;
    }
}

// copies the sequence so far (up to cap tokens); returns its length
MK_API int mk_sequence(void * hp, int * out, int cap) {
    const auto & s = static_cast<Handle *>(hp)->eng->sequence();
    const int n = (int) s.size();
    if (out)
        for (int i = 0; i < std::min(n, cap); i++) out[i] = s[i];
    return n;
}

MK_API int mk_committed(void * hp) { return static_cast<Handle *>(hp)->eng->committed(); }
MK_API int mk_checkpoint_pos(void * hp) { return static_cast<Handle *>(hp)->eng->checkpoint_pos(); }
MK_API int mk_rewind_to_checkpoint(void * hp) { return static_cast<Handle *>(hp)->eng->rewind_to_checkpoint() ? 0 : -1; }

MK_API void mk_stats(void * hp, long * cycles, long * drafted, long * accepted) {
    const auto & s = static_cast<Handle *>(hp)->eng->stats;
    *cycles = s.cycles;
    *drafted = s.drafted;
    *accepted = s.accepted;
}

// slots (parallel decoding): open n persistent sequences drawing on one K/V pool of n x ctx positions (0: the most
// that fit in device memory); returns the pool's positions, or -1
MK_API int mk_slots_open(void * hp, int n, int ctx) { return mk_slots_open2(hp, n, ctx, 0); }

// as mk_slots_open; seg_tokens > 0: shared segments (prefixes stored once) too
MK_API int mk_slots_open2(void * hp, int n, int ctx, int seg_tokens) {
    try {
        return static_cast<Handle *>(hp)->eng->slots_open(n, ctx, seg_tokens);
    } catch (const std::exception & e) {
        g_err = e.what();
        return -1;
    }
}

// slot <- the prompt (prefilled, reusing a matching prefix; first draft ready); temp 0: greedy. Returns the reused count
MK_API int mk_slot_start(void * hp, int slot, const int * toks, int n, float temp, float top_p, int top_k, float min_p, unsigned long long seed) {
    try {
        return static_cast<Handle *>(hp)->eng->slot_start(slot, std::vector<int>(toks, toks + n), temp, top_p, top_k, min_p, seed);
    } catch (const std::exception & e) {
        g_err = e.what();
        return -1;
    }
}

// positions of the shared segment the slot reads (0: none)
MK_API int mk_slot_segment(void * hp, int slot) { return static_cast<Handle *>(hp)->eng->slot_segment(slot); }

// slot_start in parts: mk_slot_begin returns the reused count (or -1), mk_slot_prefill feeds up to `budget` prompt
// tokens and returns how many remain (0: ready to step; -1: error)
MK_API int mk_slot_begin(void * hp, int slot, const int * toks, int n, float temp, float top_p, int top_k, float min_p, unsigned long long seed) {
    return mk_slot_begin2(hp, slot, toks, n, temp, top_p, top_k, min_p, seed, 0, 0);
}

// as mk_slot_begin; share_len: a prefix other prompts may share (a segment), pin: keep that segment while unused
MK_API int mk_slot_begin2(void * hp, int slot, const int * toks, int n, float temp, float top_p, int top_k, float min_p, unsigned long long seed,
                   int share_len, int pin) {
    try {
        return static_cast<Handle *>(hp)->eng->slot_begin(slot, std::vector<int>(toks, toks + n), temp, top_p, top_k, min_p, seed, share_len, pin != 0);
    } catch (const std::exception & e) {
        g_err = e.what();
        return -1;
    }
}

MK_API int mk_slot_prefill(void * hp, int slot, int budget) {
    try {
        return static_cast<Handle *>(hp)->eng->slot_prefill(slot, budget);
    } catch (const std::exception & e) {
        g_err = e.what();
        return -1;
    }
}

// one step of the n slots: out[i * per + j] = the j-th token slot i emitted, counts[i] how many (<= per; 0: the
// slot's context is full)
MK_API int mk_slots_step(void * hp, const int * slots, int n, int * out, int per, int * counts) {
    try {
        auto em = static_cast<Handle *>(hp)->eng->slots_step(std::vector<int>(slots, slots + n));
        for (int i = 0; i < n; i++) {
            counts[i] = std::min((int) em[i].size(), per);
            for (int j = 0; j < counts[i]; j++) out[i * per + j] = em[i][j];
        }
        return 0;
    } catch (const std::exception & e) {
        g_err = e.what();
        return -1;
    }
}

// (MK_PF_PROFILE) the phase times collected since the last report, to stderr
MK_API void mk_profile_report(void * hp) { static_cast<Handle *>(hp)->eng->profile_report(); }

// (test) the hidden state of the slot's last processed position -> out[0 .. n), returns its size
MK_API int mk_slot_h(void * hp, int slot, float * out, int n) {
    try {
        auto h = static_cast<Handle *>(hp)->eng->slot_h(slot);
        for (int i = 0; i < std::min(n, (int) h.size()); i++) out[i] = h[i];
        return (int) h.size();
    } catch (const std::exception & e) {
        g_err = e.what();
        return -1;
    }
}

// mk_slots_step with the next chunk of slot pslot's prompt in the same pass (budget rows with the step's); *remain <-
// the prompt tokens left, as mk_slot_prefill returns
MK_API int mk_slots_step_mixed(void * hp, int pslot, int budget, const int * slots, int n, int * out, int per, int * counts, int * remain) {
    try {
        auto em = static_cast<Handle *>(hp)->eng->slots_step_mixed(pslot, budget, std::vector<int>(slots, slots + n), *remain);
        for (int i = 0; i < n; i++) {
            counts[i] = std::min((int) em[i].size(), per);
            for (int j = 0; j < counts[i]; j++) out[i * per + j] = em[i][j];
        }
        return 0;
    } catch (const std::exception & e) {
        g_err = e.what();
        return -1;
    }
}

} // extern "C"
