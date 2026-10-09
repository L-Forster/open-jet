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

extern "C" {

const char * mk_last_error() { return g_err.c_str(); }

void * mk_new(const char * model, int ctx, int n_draft, const char * draft_vocab, int kv_mode) {
    try {
        llama_backend_init();
        llama_model_params mp = llama_model_default_params();
        mp.vocab_only = true;
        auto * h = new Handle;
        h->lm = llama_model_load_from_file(model, mp);
        if (!h->lm) throw std::runtime_error("failed to load the vocabulary");
        h->vocab = llama_model_get_vocab(h->lm);
        h->eng = new MKEngine(model, ctx, n_draft, draft_vocab ? draft_vocab : "", kv_mode);
        return h;
    } catch (const std::exception & e) {
        g_err = e.what();
        return nullptr;
    }
}

void mk_free(void * hp) {
    auto * h = static_cast<Handle *>(hp);
    delete h->eng;
    llama_model_free(h->lm);
    delete h;
}

// chat template stored in the GGUF (tokenizer.chat_template); returns its length, copies up to cap
int mk_chat_template(void * hp, char * buf, int cap) {
    auto * h = static_cast<Handle *>(hp);
    const char * t = llama_model_chat_template(h->lm, nullptr);
    if (!t) return -1;
    const int n = (int) strlen(t);
    if (buf && cap > 0) memcpy(buf, t, std::min(n, cap));
    return n;
}

// text -> token ids (special tokens such as <|im_start|> parsed); returns the count (negative: -needed)
int mk_tokenize(void * hp, const char * text, int len, int * out, int cap) {
    auto * h = static_cast<Handle *>(hp);
    return llama_tokenize(h->vocab, text, len, out, cap, false, true);
}

// token -> its bytes (special tokens rendered as text); returns the byte count
int mk_piece(void * hp, int tok, char * buf, int cap) {
    auto * h = static_cast<Handle *>(hp);
    return llama_token_to_piece(h->vocab, tok, buf, cap, 0, true);
}

int mk_is_eog(void * hp, int tok) { return llama_vocab_is_eog(static_cast<Handle *>(hp)->vocab, tok) ? 1 : 0; }

int mk_set_sampling(void * hp, float temp, float top_p, int top_k, float min_p, unsigned long long seed) {
    try {
        static_cast<Handle *>(hp)->eng->set_sampling(temp, top_p, top_k, min_p, seed);
        return 0;
    } catch (const std::exception & e) {
        g_err = e.what();
        return -1;
    }
}

// appends tokens to the sequence; returns the first generated token (negative on error)
int mk_prefill(void * hp, const int * toks, int n) {
    try {
        return static_cast<Handle *>(hp)->eng->prefill(std::vector<int>(toks, toks + n));
    } catch (const std::exception & e) {
        g_err = e.what();
        return -1;
    }
}

// one decode cycle; writes 1..n_draft+1 tokens; returns the count (negative on error)
int mk_step(void * hp, int * out, int cap) {
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

void mk_reset(void * hp) { static_cast<Handle *>(hp)->eng->reset(); }

int mk_truncate(void * hp, int len) {
    try {
        static_cast<Handle *>(hp)->eng->truncate(len);
        return 0;
    } catch (const std::exception & e) {
        g_err = e.what();
        return -1;
    }
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
int mk_rewind_to_checkpoint(void * hp) { return static_cast<Handle *>(hp)->eng->rewind_to_checkpoint() ? 0 : -1; }

void mk_stats(void * hp, long * cycles, long * drafted, long * accepted) {
    const auto & s = static_cast<Handle *>(hp)->eng->stats;
    *cycles = s.cycles;
    *drafted = s.drafted;
    *accepted = s.accepted;
}

} // extern "C"
