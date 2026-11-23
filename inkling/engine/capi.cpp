// The mk_* C API of OpenJet's engine server (megakernel/server/mk_engine.py) on top of the Inkling engine, so
// mk_server.py serves Inkling 2 Mini with --lib libinkling.so: OpenAI Chat Completions, Responses and Anthropic
// Messages, unchanged.
//
// Tokenizer and chat template come from libllama with the model loaded vocabulary-only (a llama.cpp build that knows
// inkling2_loop). Sampling (penalties, top-k, top-p, min-p, temperature, seed) runs on the host over the logits.
// Prompts are fed in batches (ik_prefill), the last token through a decode step for its logits. One rewind checkpoint
// is kept: at ck_at when the server gives it, else at the end of each prompt. A follow-up request that re-renders the
// reply differently (a client dropping the reasoning, say) resumes there instead of re-reading the conversation.
//
// Offload mode: IK_VRAM_MIB=<expert cache MiB> (unset: every expert in VRAM); IK_PROFILE=<profile> (default:
// profiles/default.f32 next to the library); IK_THREADS.

#include "llama.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#ifdef _WIN32
#include "compat_win32.h"
#else
#include <dlfcn.h>
#endif
#include <random>
#include <regex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

extern "C" {
void * ik_new(const char * path, int ctx);
void * ik_new2(const char * path, int ctx, int vram_expert_mib, const char * profile, int n_threads);
int ik_reset(void * h);
int ik_decode(void * h, int token, float * logits);
int ik_decode_greedy(void * h, int token);
int ik_decode_filtered(void * h, int token, float margin, int32_t * ids_out, float * vals_out, int cap, float * max_out);
int ik_read_logits(void * h, float * logits);
int ik_set_penalty(void * h, float pres, float freq);
int ik_pen_clear(void * h);
int ik_pen_inc(void * h, int tok);
int ik_verify(void * h, const int * toks, int n, float * logits);
int ik_verify_greedy(void * h, const int * toks, int n, int * argmax_out);
int ik_verify_commit(void * h, int acc);
int ik_load_mtp(void * h, const char * mtp_path);
int ik_has_mtp(void * h);
int ik_draft(void * h, int anchor_tok, int n_draft, int * out_drafts);
int ik_prefill(void * h, const int * toks, int n);
int ik_checkpoint_save(void * h);
int ik_checkpoint_restore(void * h);
int ik_checkpoint_restore_pos(void * h, int pos);
int ik_checkpoint_pos(void * h);
int ik_checkpoints(void * h, int * out, int cap);
int ik_checkpoint_pin(void * h, const int * pos, int n);
int ik_checkpoint_pinned(void * h, int * out, int cap);
int ik_checkpoint_plan(void * h, const int * cks, int n_cks, int ck_exact, int base, int end, int * out, int cap);
int ik_n_vocab(void * h);
void ik_free(void * h);
const char * ik_last_error();
}

namespace {

std::string g_err;

struct Handle {
    void * eng = nullptr;
    llama_model * lm = nullptr;
    const llama_vocab * vocab = nullptr;
    int ctx = 0, n_vocab = 0;
    std::vector<int> seq;       // fed tokens, then the last emitted one (not yet fed)
    int fed = 0;                // tokens whose state is in the engine
    std::vector<float> logits;     // after the last fed token
    std::vector<float> vf_logits;  // multi-token verification logits [T][n_vocab]
    // sampling
    float temp = 0.f, top_p = 1.f, min_p = 0.f, presence = 0.f, frequency = 0.f;
    int top_k = 0;
    std::mt19937_64 rng{0};
    std::unordered_map<int, int> counts;  // generated tokens, for the penalties
    // prompt-lookup speculative drafting (default: on in full VRAM, off in offload where 128-expert MoE
    // multi-token batches multiply unique CPU expert reads; override with IK_LOOKUP=0|1, IK_DRAFT=1..7)
    int lookup_on = 1;
    int max_draft = 4;
    int cur_draft = 4;
    bool adapt_draft = false;
    int lookup_cooldown = 0;
    std::vector<float> pen_buf;  // reusable scratch for penalized logits (avoids 800 KB malloc per token)
    long cycles = 0, drafted = 0, accepted = 0;
    std::atomic<bool> cancel{false};
    std::atomic<int> pf_done{0}, pf_total{0};
};

template <class F> int guarded(Handle * h, F f) {
    try {
        return f(h);
    } catch (const std::exception & e) {
        g_err = e.what();
        return -1;
    }
}

std::string lib_dir() {
#ifdef _WIN32
    char buf[PATH_MAX];
    if (mk_win_module_path((void *) &lib_dir, buf, sizeof(buf))) {
        std::string p = buf;
        const size_t s = p.find_last_of("\\/");
        return s == std::string::npos ? "." : p.substr(0, s);
    }
    return ".";
#else
    Dl_info info;
    if (dladdr((void *) &lib_dir, &info) && info.dli_fname) {
        std::string p = info.dli_fname;
        const size_t s = p.rfind('/');
        return s == std::string::npos ? "." : p.substr(0, s);
    }
    return ".";
#endif
}

// build exact filtered sampling distribution (top-k, min-p, top-p) for raw_logits + optional penalties.
// Pre-filters logits >= max - margin in a linear scan so we sort ~20-100 survivors instead of partial-sorting 200K tokens.
void build_dist(Handle * h, const float * raw_logits, std::vector<int> & idx, std::vector<double> & p, size_t & n, double & tot,
                bool apply_pen = true) {
    const int V = h->n_vocab;
    const float * l = raw_logits;
    if (apply_pen && !h->counts.empty() && (h->presence != 0.f || h->frequency != 0.f)) {
        h->pen_buf.assign(raw_logits, raw_logits + V);
        for (const auto & [t, c] : h->counts) h->pen_buf[t] -= h->presence + h->frequency * c;
        l = h->pen_buf.data();
    }
    float mx_f = l[0];
    for (int i = 1; i < V; ++i)
        if (l[i] > mx_f) mx_f = l[i];
    const float margin = h->min_p > 0.f ? (-h->temp * std::log(h->min_p) + 1e-3f) : (20.f * h->temp);
    const float thresh = mx_f - margin;
    idx.clear();
    for (int i = 0; i < V; ++i)
        if (l[i] >= thresh) idx.push_back(i);
    if (idx.empty()) idx.push_back(0);
    size_t k = h->top_k > 0 ? std::min<size_t>(h->top_k, idx.size()) : idx.size();
    const size_t keep = std::min<size_t>(k, 1024);
    if (idx.size() > keep) {
        std::partial_sort(idx.begin(), idx.begin() + keep, idx.end(), [&](int a, int b) {
            return l[a] > l[b] || (l[a] == l[b] && a < b);
        });
        idx.resize(keep);
    } else {
        std::sort(idx.begin(), idx.end(), [&](int a, int b) {
            return l[a] > l[b] || (l[a] == l[b] && a < b);
        });
    }
    p.resize(idx.size());
    const double mx = l[idx[0]];
    for (size_t i = 0; i < idx.size(); ++i) p[i] = std::exp((l[idx[i]] - mx) / h->temp);
    n = idx.size();
    if (h->min_p > 0.f) {
        const double floor = p[0] * h->min_p;
        while (n > 1 && p[n - 1] < floor) n--;
    }
    if (h->top_p < 1.f) {
        double acc = 0, s = 0;
        for (size_t i = 0; i < n; ++i) s += p[i];
        size_t m = 0;
        while (m < n) { acc += p[m++]; if (acc >= h->top_p * s) break; }
        n = m;
    }
    tot = 0;
    for (size_t i = 0; i < n; ++i) tot += p[i];
}

int sample_from(Handle * h, const float * raw_logits, bool apply_pen = true) {
    if (h->temp <= 0.f) {
        std::vector<float> l(raw_logits, raw_logits + h->n_vocab);
        if (apply_pen) for (const auto & [t, c] : h->counts) l[t] -= h->presence + h->frequency * c;
        return int(std::max_element(l.begin(), l.end()) - l.begin());
    }
    std::vector<int> idx;
    std::vector<double> p;
    size_t n = 0;
    double tot = 0;
    build_dist(h, raw_logits, idx, p, n, tot, apply_pen);
    std::uniform_real_distribution<double> u(0.0, tot);
    double r = u(h->rng);
    for (size_t i = 0; i < n; ++i) { r -= p[i]; if (r <= 0) return idx[i]; }
    return idx[n - 1];
}

// exact sampling over GPU-filtered candidate set (penalties already applied on GPU)
int sample_filtered(Handle * h, const int32_t * ids, const float * vals, int cnt) {
    std::vector<int> order(cnt);
    for (int i = 0; i < cnt; ++i) order[i] = i;
    size_t keep = (size_t) cnt;
    if (h->top_k > 0) keep = std::min<size_t>(keep, (size_t) h->top_k);
    keep = std::min<size_t>(keep, 1024);
    std::partial_sort(order.begin(), order.begin() + keep, order.end(), [&](int a, int b) {
        return vals[a] > vals[b] || (vals[a] == vals[b] && ids[a] < ids[b]);
    });
    order.resize(keep);
    std::vector<double> p(keep);
    const double mx = vals[order[0]];
    for (size_t i = 0; i < keep; ++i) p[i] = std::exp((vals[order[i]] - mx) / h->temp);
    size_t n = keep;
    if (h->min_p > 0.f) {
        const double floor = p[0] * h->min_p;
        while (n > 1 && p[n - 1] < floor) n--;
    }
    if (h->top_p < 1.f) {
        double acc = 0, s = 0;
        for (size_t i = 0; i < n; ++i) s += p[i];
        size_t m = 0;
        while (m < n) { acc += p[m++]; if (acc >= h->top_p * s) break; }
        n = m;
    }
    double tot = 0;
    for (size_t i = 0; i < n; ++i) tot += p[i];
    std::uniform_real_distribution<double> u(0.0, tot);
    double r = u(h->rng);
    for (size_t i = 0; i < n; ++i) { r -= p[i]; if (r <= 0) return ids[order[i]]; }
    return ids[order[n - 1]];
}

// Zero-Logits-D2H single-token step: feeds tok, samples next token (on GPU for greedy, via compact GPU filter for temp > 0),
// records count on host & GPU, and appends to h->seq.
int feed_and_emit(Handle * h, int tok) {
    if (h->fed >= h->ctx) throw std::runtime_error("context full (" + std::to_string(h->ctx) + " tokens)");
    int next = -1;
    if (h->temp <= 0.f) {
        next = ik_decode_greedy(h->eng, tok);
        if (next < 0) throw std::runtime_error(ik_last_error());
    } else {
        constexpr int CAP = 4096;
        int32_t ids[CAP];
        float vals[CAP];
        float mx = 0.f;
        const float margin = h->min_p > 0.f ? (-h->temp * std::log(h->min_p) + 1e-3f) : (20.f * h->temp);
        const int cnt = ik_decode_filtered(h->eng, tok, margin, ids, vals, CAP, &mx);
        if (cnt < 0) throw std::runtime_error(ik_last_error());
        if (cnt > 0 && cnt <= CAP) {
            next = sample_filtered(h, ids, vals, cnt);
        } else {
            if (ik_read_logits(h->eng, h->logits.data()) != 0) throw std::runtime_error(ik_last_error());
            next = sample_from(h, h->logits.data(), false);
        }
    }
    h->fed++;
    h->counts[next]++;
    ik_pen_inc(h->eng, next);
    h->seq.push_back(next);
    return next;
}

// exact speculative verification of one deterministic draft token against raw_logits:
// returns true if accepted (out_tok == draft_tok), false if rejected (out_tok sampled from exact residual distribution).
bool verify_token(Handle * h, const float * raw_logits, int draft_tok, int & out_tok) {
    if (h->temp <= 0.f) {
        out_tok = sample_from(h, raw_logits);
        return out_tok == draft_tok;
    }
    std::vector<int> idx;
    std::vector<double> p;
    size_t n = 0;
    double tot = 0;
    build_dist(h, raw_logits, idx, p, n, tot);
    size_t found = n;
    for (size_t i = 0; i < n; ++i) {
        if (idx[i] == draft_tok) { found = i; break; }
    }
    const double pd = (found < n && tot > 0.0) ? (p[found] / tot) : 0.0;
    std::uniform_real_distribution<double> u01(0.0, 1.0);
    if (u01(h->rng) < pd) {
        out_tok = draft_tok;
        return true;
    }
    double rem = tot;
    if (found < n) {
        rem -= p[found];
        p[found] = 0.0;
    }
    if (rem <= 0.0) {
        out_tok = draft_tok;
        return true;
    }
    std::uniform_real_distribution<double> ur(0.0, rem);
    double r = ur(h->rng);
    int last_nonzero = idx[0];
    for (size_t i = 0; i < n; ++i) {
        if (p[i] <= 0.0) continue;
        last_nonzero = idx[i];
        r -= p[i];
        if (r <= 0.0) { out_tok = idx[i]; return false; }
    }
    out_tok = last_nonzero;
    return false;
}

// prompt-lookup n-gram drafter: scans h->seq backwards for a 4-gram match at the suffix and returns up to
// max_d continuation tokens (stopping before any EOG token).
int draft_lookup(const Handle * h, int * out, int max_d) {
    const int L = (int) h->seq.size();
    constexpr int ng = 4;
    if (max_d <= 0 || L <= ng) return 0;
    const int * s = h->seq.data();
    const int * pat = s + (L - ng);
    const int last = pat[ng - 1];
    for (int pos = L - ng - 1; pos >= 0; --pos) {
        if (s[pos + ng - 1] != last) continue;
        if (s[pos] != pat[0] || s[pos + 1] != pat[1] || s[pos + 2] != pat[2]) continue;
        int nd = 0;
        for (int k = 0; k < max_d && pos + ng + k < L; ++k) {
            const int tok = s[pos + ng + k];
            if (tok < 0 || tok >= h->n_vocab || llama_vocab_is_eog(h->vocab, tok)) break;
            out[nd++] = tok;
        }
        if (nd > 0) return nd;
    }
    return 0;
}

int emit(Handle * h) {
    const int t = sample_from(h, h->logits.data());
    h->counts[t]++;
    h->seq.push_back(t);
    return t;
}

}  // namespace

extern "C" {

const char * mk_last_error() { return g_err.c_str(); }

void * mk_new_mtp(const char * model, int ctx, int n_draft, const char * /*draft_vocab*/, int /*kv_mode*/, const char * mtp) {
    auto * h = new Handle();
    try {
        h->ctx = ctx > 0 ? ctx : 8192;
        const char * vram = getenv("IK_VRAM_MIB");
        const int vram_mib = (vram && *vram) ? atoi(vram) : -1;
        if (vram_mib >= 0) {
            std::string prof = getenv("IK_PROFILE") ? getenv("IK_PROFILE") : lib_dir() + "/profiles/default.f32";
            FILE * fp = fopen(prof.c_str(), "rb");
            if (fp) fclose(fp); else prof.clear();
            h->eng = ik_new2(model, h->ctx, vram_mib, prof.empty() ? nullptr : prof.c_str(),
                             getenv("IK_THREADS") ? atoi(getenv("IK_THREADS")) : 0);
        } else {
            h->eng = ik_new(model, h->ctx);
        }
        if (!h->eng) throw std::runtime_error(ik_last_error());
        // Optional DSpark MTP drafter: explicit --mtp / IK_MTP, or auto-discovered mtp.safetensors next to the model
        // in full-VRAM mode (offload mode multiplies unique CPU expert misses on multi-token verify, so stay off unless
        // explicitly requested; override with IK_MTP=<path> or disable with IK_NO_MTP=1 / --no-mtp)
        const bool no_mtp = (mtp && strcmp(mtp, "none") == 0) ||
                            (getenv("IK_NO_MTP") && atoi(getenv("IK_NO_MTP")) != 0) ||
                            (getenv("MK_NO_MTP") && atoi(getenv("MK_NO_MTP")) != 0);
        if (!no_mtp) {
            std::string mp_path;
            if (mtp && *mtp) {
                mp_path = mtp;
            } else if (const char * em = getenv("IK_MTP"); em && *em) {
                mp_path = em;
            } else if (vram_mib < 0) {
                std::string m = model;
                const size_t slash = m.find_last_of("/\\");
                std::string cand = (slash == std::string::npos ? "" : m.substr(0, slash + 1)) + "mtp.safetensors";
                FILE * fp = fopen(cand.c_str(), "rb");
                if (fp) { fclose(fp); mp_path = cand; }
            }
            if (!mp_path.empty()) {
                if (ik_load_mtp(h->eng, mp_path.c_str()) != 0) {
                    if ((mtp && *mtp) || (getenv("IK_MTP") && *getenv("IK_MTP")))
                        throw std::runtime_error(ik_last_error());
                    fprintf(stderr, "inkling: DSpark MTP skipped (%s)\n", ik_last_error());
                }
            }
        }
        llama_backend_init();
        llama_model_params mp = llama_model_default_params();
        mp.vocab_only = true;
        h->lm = llama_model_load_from_file(model, mp);
        if (!h->lm) throw std::runtime_error("cannot load the vocabulary of " + std::string(model));
        h->vocab = llama_model_get_vocab(h->lm);
        h->n_vocab = ik_n_vocab(h->eng);
        h->logits.assign(h->n_vocab, 0.f);
        h->lookup_on = (vram_mib >= 0) ? 0 : 1;
        if (const char * lk = getenv("IK_LOOKUP")) h->lookup_on = atoi(lk) != 0 ? 1 : 0;
        const bool has_mtp = ik_has_mtp(h->eng) != 0;
        const char * dr = getenv("IK_DRAFT");
        h->max_draft = has_mtp ? (n_draft > 0 ? std::min(7, n_draft) : 7) : 4;
        if (dr) h->max_draft = std::max(0, std::min(7, atoi(dr)));
        // Adaptive draft depth in {2, 4, 7} when DSpark is active and draft count wasn't pinned via IK_DRAFT / --n-draft
        h->adapt_draft = has_mtp && !dr && n_draft <= 0 && (!getenv("IK_ADAPT") || atoi(getenv("IK_ADAPT")) != 0);
        h->cur_draft = h->adapt_draft ? std::min(h->max_draft, 4) : h->max_draft;
        return h;
    } catch (const std::exception & e) {
        g_err = e.what();
        if (h->eng) ik_free(h->eng);
        delete h;
        return nullptr;
    }
}

void * mk_new(const char * model, int ctx, int n_draft, const char * draft_vocab, int kv_mode) {
    return mk_new_mtp(model, ctx, n_draft, draft_vocab, kv_mode, nullptr);
}

void mk_free(void * hp) {
    auto * h = static_cast<Handle *>(hp);
    if (h->eng) ik_free(h->eng);
    if (h->lm) llama_model_free(h->lm);
    delete h;
}

// the GGUF's chat template, without Hugging Face's {% generation %} / {% endgeneration %} markers (Inkling 2 Mini's
// template marks the assistant's text with them for training masks; transformers renders the body unchanged, and
// plain Jinja does not know the tag)
int mk_chat_template(void * hp, char * buf, int cap) {
    const char * t = llama_model_chat_template(static_cast<Handle *>(hp)->lm, nullptr);
    if (!t) return -1;
    static const std::regex marker(R"(\{%-?\s*(end)?generation\s*-?%\})");
    const std::string out = std::regex_replace(std::string(t), marker, "");
    const int n = (int) out.size();
    if (buf && cap > 0) memcpy(buf, out.data(), std::min(n, cap));
    return n;
}

int mk_tokenize2(void * hp, const char * text, int len, int * out, int cap, int add_special, int parse_special) {
    return llama_tokenize(static_cast<Handle *>(hp)->vocab, text, len, out, cap, add_special != 0, parse_special != 0);
}

int mk_tokenize(void * hp, const char * text, int len, int * out, int cap) { return mk_tokenize2(hp, text, len, out, cap, 0, 1); }

int mk_token_attr(void * hp, int tok) { return (int) llama_vocab_get_attr(static_cast<Handle *>(hp)->vocab, tok); }

int mk_piece(void * hp, int tok, char * buf, int cap) {
    return llama_token_to_piece(static_cast<Handle *>(hp)->vocab, tok, buf, cap, 0, true);
}

int mk_is_eog(void * hp, int tok) { return llama_vocab_is_eog(static_cast<Handle *>(hp)->vocab, tok) ? 1 : 0; }

int mk_n_vocab(void * hp) { return static_cast<Handle *>(hp)->n_vocab; }

int mk_set_sampling(void * hp, float temp, float top_p, int top_k, float min_p, unsigned long long seed) {
    auto * h = static_cast<Handle *>(hp);
    h->temp = temp; h->top_p = top_p; h->top_k = top_k; h->min_p = min_p;
    h->rng.seed(seed);
    return 0;
}

int mk_set_penalty(void * hp, float presence, float frequency) {
    auto * h = static_cast<Handle *>(hp);
    h->presence = presence; h->frequency = frequency;
    return ik_set_penalty(h->eng, presence, frequency);
}

// appends tokens to the sequence (after the last emitted one, which is fed first); returns the first generated token.
// Saves checkpoints at the surviving positions of `cks` + `ck_exact` (or end-of-prompt when ck_exact < 0) + pending pins.
int mk_prefill_cks(void * hp, const int * toks, int n, const int * cks, int n_cks, int ck_exact) {
    return guarded(static_cast<Handle *>(hp), [&](Handle * h) {
        h->counts.clear();
        ik_pen_clear(h->eng);
        h->lookup_cooldown = 0;
        h->cur_draft = h->adapt_draft ? std::min(h->max_draft, 4) : h->max_draft;
        const int base = h->fed;
        std::vector<int> todo(h->seq.begin() + base, h->seq.end());
        todo.insert(todo.end(), toks, toks + n);
        if (todo.empty()) throw std::runtime_error("prefill: no tokens");
        for (int t : todo)
            if (t < 0 || t >= h->n_vocab) throw std::runtime_error("token id out of range");
        h->seq.resize(base);
        h->pf_total = (int) todo.size();
        h->pf_done = 0;
        const int end = base + (int) todo.size();
        if (end > h->ctx) throw std::runtime_error("context full (" + std::to_string(h->ctx) + " tokens)");
        // plan_checkpoints returns at most ck_pool_.size() (<= 32) positions
        std::vector<int> plan(64);
        int np = ik_checkpoint_plan(h->eng, cks, n_cks, ck_exact, base, end, plan.data(), (int) plan.size());
        if (np < 0) throw std::runtime_error(ik_last_error());
        plan.resize(std::min(np, (int) plan.size()));
        // all but the last token batched, in pieces so a cancel, progress, or checkpoint lands between them
        constexpr int PIECE = 1024;
        const int n_batch = (int) todo.size() - 1;
        size_t pi = 0;
        for (int i = 0; i < n_batch;) {
            if (h->cancel.load()) {  // the sequence keeps what was fed
                g_err = "prefill cancelled";
                return -2;
            }
            int m = std::min(PIECE, n_batch - i);
            if (pi < plan.size()) {
                const int rel = plan[pi] - base;
                if (rel > i && rel < i + m) m = rel - i;
            }
            if (ik_prefill(h->eng, todo.data() + i, m) != 0) throw std::runtime_error(ik_last_error());
            h->fed += m;
            h->seq.insert(h->seq.end(), todo.begin() + i, todo.begin() + i + m);
            i += m;
            h->pf_done = i;
            if (pi < plan.size() && h->fed == plan[pi]) {
                if (ik_checkpoint_save(h->eng) != 0) throw std::runtime_error(ik_last_error());
                pi++;
            }
        }
        h->seq.push_back(todo.back());
        const int first = feed_and_emit(h, todo.back());  // Zero-Logits-D2H decode gives the first generated token
        h->pf_done = (int) todo.size();
        if (pi < plan.size() && h->fed == plan[pi]) {
            if (ik_checkpoint_save(h->eng) != 0) throw std::runtime_error(ik_last_error());
        }
        return first;
    });
}

int mk_prefill_ck(void * hp, const int * toks, int n, int ck_at) { return mk_prefill_cks(hp, toks, n, nullptr, 0, ck_at); }

int mk_prefill(void * hp, const int * toks, int n) { return mk_prefill_cks(hp, toks, n, nullptr, 0, -1); }

// one decode cycle: feeds the last emitted token (plus up to IK_DRAFT DSpark MTP or prompt-lookup drafts),
// writes 1..1+nd exact output tokens
int mk_step(void * hp, int * out, int cap) {
    return guarded(static_cast<Handle *>(hp), [&](Handle * h) {
        if (cap < 1) throw std::runtime_error("mk_step: empty buffer");
        if ((int) h->seq.size() != h->fed + 1) throw std::runtime_error("mk_step: nothing to decode (prefill first)");
        const int want_d = h->adapt_draft ? h->cur_draft : h->max_draft;
        const int max_d = std::min({want_d, cap - 1, h->ctx - h->fed - 1, 7});
        int draft[8];
        int nd = 0;
        const bool use_mtp = ik_has_mtp(h->eng) != 0 && max_d >= 1;
        if (use_mtp) {
            nd = ik_draft(h->eng, h->seq.back(), max_d, draft);
            if (nd < 0) throw std::runtime_error(ik_last_error());
        } else if (h->lookup_on && max_d >= 1) {
            if (h->lookup_cooldown > 0) h->lookup_cooldown--;
            else nd = draft_lookup(h, draft, max_d);
        }
        if (nd == 0) {
            out[0] = feed_and_emit(h, h->seq.back());
            h->cycles++;
            return 1;
        }
        int toks[8];
        toks[0] = h->seq.back();
        for (int k = 0; k < nd; ++k) toks[1 + k] = draft[k];
        const int T = 1 + nd;
        int emitted = 0, n_acc = 0;
        if (h->temp <= 0.f && h->presence == 0.f && h->frequency == 0.f) {
            // Zero-Logits-D2H exact greedy verification (32 bytes D2H instead of 6.4 MB)
            int argmax[8];
            if (ik_verify_greedy(h->eng, toks, T, argmax) != 0) throw std::runtime_error(ik_last_error());
            for (int k = 0; k < nd; ++k) {
                const int tok = argmax[k];
                out[emitted++] = tok;
                h->counts[tok]++;
                ik_pen_inc(h->eng, tok);
                h->seq.push_back(tok);
                if (tok == draft[k]) n_acc++;
                else break;
            }
            if (n_acc == nd) {
                const int bonus = argmax[nd];
                out[emitted++] = bonus;
                h->counts[bonus]++;
                ik_pen_inc(h->eng, bonus);
                h->seq.push_back(bonus);
            }
        } else {
            if (h->vf_logits.size() < size_t(T) * h->n_vocab) h->vf_logits.resize(size_t(T) * h->n_vocab);
            if (ik_verify(h->eng, toks, T, h->vf_logits.data()) != 0) throw std::runtime_error(ik_last_error());
            for (int k = 0; k < nd; ++k) {
                int tok = 0;
                const bool acc = verify_token(h, h->vf_logits.data() + size_t(k) * h->n_vocab, draft[k], tok);
                out[emitted++] = tok;
                h->counts[tok]++;
                ik_pen_inc(h->eng, tok);
                h->seq.push_back(tok);
                if (acc) n_acc++;
                else break;
            }
            if (n_acc == nd) {
                const int bonus = sample_from(h, h->vf_logits.data() + size_t(nd) * h->n_vocab);
                out[emitted++] = bonus;
                h->counts[bonus]++;
                ik_pen_inc(h->eng, bonus);
                h->seq.push_back(bonus);
            }
        }
        if (ik_verify_commit(h->eng, 1 + n_acc) != 0) throw std::runtime_error(ik_last_error());
        h->fed += 1 + n_acc;
        if (use_mtp && h->adapt_draft) {
            if (n_acc == nd) {
                h->cur_draft = (h->cur_draft < 4) ? 4 : std::min(h->max_draft, 7);
            } else if (h->cur_draft > 4 && n_acc <= 3) {
                h->cur_draft = 4;
            } else if (h->cur_draft == 4 && n_acc <= 1) {
                h->cur_draft = 2;
            }
        }
        if (!use_mtp) h->lookup_cooldown = (n_acc == 0) ? 8 : 0;
        h->cycles++;
        h->drafted += nd;
        h->accepted += n_acc;
        return emitted;
    });
}

void mk_reset(void * hp) {
    guarded(static_cast<Handle *>(hp), [](Handle * h) {
        if (ik_reset(h->eng) != 0) throw std::runtime_error(ik_last_error());
        h->seq.clear();
        h->fed = 0;
        h->counts.clear();
        ik_pen_clear(h->eng);
        h->lookup_cooldown = 0;
        h->cur_draft = h->adapt_draft ? std::min(h->max_draft, 4) : h->max_draft;
        return 0;
    });
}

// drops emitted tokens that were not fed yet (len >= committed)
int mk_truncate(void * hp, int len) {
    return guarded(static_cast<Handle *>(hp), [&](Handle * h) {
        if (len < h->fed || len > (int) h->seq.size())
            throw std::runtime_error("truncate: can only drop tokens emitted after the committed position");
        h->seq.resize(len);
        return 0;
    });
}

int mk_sequence(void * hp, int * out, int cap) {
    const auto & s = static_cast<Handle *>(hp)->seq;
    if (out)
        for (int i = 0; i < std::min((int) s.size(), cap); ++i) out[i] = s[i];
    return (int) s.size();
}

int mk_committed(void * hp) { return static_cast<Handle *>(hp)->fed; }
int mk_checkpoint_pos(void * hp) { return ik_checkpoint_pos(static_cast<Handle *>(hp)->eng); }
int mk_checkpoints(void * hp, int * out, int cap) { return ik_checkpoints(static_cast<Handle *>(hp)->eng, out, cap); }

// back to the checkpoint at `pos` (0, or -1 when there is none)
int mk_rewind_to(void * hp, int pos) {
    return guarded(static_cast<Handle *>(hp), [&](Handle * h) {
        if (pos < 0 || pos > (int) h->seq.size()) { g_err = "no checkpoint"; return -1; }
        if (ik_checkpoint_restore_pos(h->eng, pos) != 0) { g_err = "no checkpoint"; return -1; }
        h->seq.resize(pos);
        h->fed = pos;
        h->counts.clear();
        ik_pen_clear(h->eng);
        h->lookup_cooldown = 0;
        h->cur_draft = h->adapt_draft ? std::min(h->max_draft, 4) : h->max_draft;
        return 0;
    });
}

// back to the latest checkpoint: the sequence is cut there (0, or -1 when there is none)
int mk_rewind_to_checkpoint(void * hp) {
    const int p = mk_checkpoint_pos(hp);
    return p < 0 ? (g_err = "no checkpoint", -1) : mk_rewind_to(hp, p);
}

int mk_ck_pin(void * hp, const int * pos, int n) {
    return guarded(static_cast<Handle *>(hp), [&](Handle * h) { return ik_checkpoint_pin(h->eng, pos, n); });
}

int mk_ck_pinned(void * hp, int * out, int cap) { return ik_checkpoint_pinned(static_cast<Handle *>(hp)->eng, out, cap); }

int mk_has_mtp(void * hp) { return ik_has_mtp(static_cast<Handle *>(hp)->eng); }

void mk_stats(void * hp, long * cycles, long * drafted, long * accepted) {
    const auto * h = static_cast<Handle *>(hp);
    if (cycles) *cycles = h->cycles;
    if (drafted) *drafted = h->drafted;
    if (accepted) *accepted = h->accepted;
}

int mk_cancel(void * hp, int on) {
    static_cast<Handle *>(hp)->cancel = on != 0;
    return 0;
}

void mk_prefill_progress(void * hp, int * done, int * total) {
    *done = static_cast<Handle *>(hp)->pf_done;
    *total = static_cast<Handle *>(hp)->pf_total;
}

const char * mk_build_info() { return "{\"engine\": \"inkling\", \"model\": \"Inkling 2 Mini\"}"; }

}  // extern "C"
