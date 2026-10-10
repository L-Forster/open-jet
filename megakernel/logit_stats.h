// Reference next-token distributions and their comparison (KL divergence, perplexity, top-token
// agreement), shared by the megakernel CLI and tools/llama_ref (the llama.cpp side).
//
// A reference file holds token sequences and, for every scored position i (the distribution of
// token i + 1 after tokens[0..i]), the REF_K most likely tokens with their log-probabilities, the
// probability of all other tokens together, and the log-probability of the actual next token.
// KL(ref || engine) treats the tokens outside the reference's REF_K as one bucket, which makes it a
// lower bound of the full-vocabulary value (tight while that bucket is small; its mean is reported).
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <numeric>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace ls {

constexpr int REF_K = 256;
constexpr char MAGIC[8] = "MKREF01";

struct Seq {
    std::vector<int> tok;
    int first; // first scored position; the last one is tok.size() - 2
    int scored() const { return std::max(0, (int) tok.size() - 1 - first); }
};

struct Rec {
    float lp_next; // log p(actual next token)
    float tail;    // probability of the tokens outside id[]
    int id[REF_K]; // most likely tokens, descending
    float lp[REF_K];
};

struct Cmp {
    float kld, nll, ref_nll;
    int top, ref_top;
    float ref_gap, gap; // on a top-token mismatch: how far each side's log p of its own choice is above the other's
};

// f(r) for r in [0, n) on the machine's hardware threads
template <class F> void par_rows(int n, F f) {
    const int nt = std::max(1, std::min(n, (int) std::thread::hardware_concurrency()));
    std::vector<std::thread> th;
    for (int k = 0; k < nt; k++)
        th.emplace_back([=] { for (int r = k; r < n; r += nt) f(r); });
    for (auto & t : th) t.join();
}

// argmax (lowest id on ties) and log sum exp of a logits row
inline int row_lse(const float * l, int nv, double & lse) {
    int top = 0;
    for (int i = 1; i < nv; i++)
        if (l[i] > l[top]) top = i;
    double s = 0;
    for (int i = 0; i < nv; i++) s += std::exp(l[i] - l[top]);
    lse = l[top] + std::log(s);
    return top;
}

inline void make_rec(const float * l, int nv, int next, Rec & r) {
    double lse;
    row_lse(l, nv, lse);
    static thread_local std::vector<int> idx;
    idx.resize(nv);
    std::iota(idx.begin(), idx.end(), 0);
    const auto above = [l](int a, int b) { return l[a] > l[b] || (l[a] == l[b] && a < b); };
    std::nth_element(idx.begin(), idx.begin() + REF_K, idx.end(), above);
    std::sort(idx.begin(), idx.begin() + REF_K, above);
    double mass = 0;
    for (int k = 0; k < REF_K; k++) {
        r.id[k] = idx[k];
        r.lp[k] = (float) (l[idx[k]] - lse);
        mass += std::exp(l[idx[k]] - lse);
    }
    r.tail = (float) std::max(0.0, 1.0 - mass);
    r.lp_next = (float) (l[next] - lse);
}

inline Cmp compare(const float * l, int nv, int next, const Rec & ref) {
    double lse;
    Cmp c{};
    c.top = row_lse(l, nv, lse);
    c.ref_top = ref.id[0];
    double kld = 0, mass = 0;
    for (int k = 0; k < REF_K; k++) {
        const double lq = l[ref.id[k]] - lse;
        kld += std::exp((double) ref.lp[k]) * (ref.lp[k] - lq);
        mass += std::exp(lq);
    }
    if (ref.tail > 0) kld += ref.tail * (std::log((double) ref.tail) - std::log(std::max(1.0 - mass, 1e-12)));
    c.kld = (float) kld;
    c.nll = (float) (lse - l[next]);
    c.ref_nll = -ref.lp_next;
    if (c.top != c.ref_top) {
        // the engine's choice in the reference's list; beyond the list it is at least as far down as the last entry
        float ref_lp_top = ref.lp[REF_K - 1];
        for (int k = 1; k < REF_K; k++)
            if (ref.id[k] == c.top) ref_lp_top = ref.lp[k];
        c.ref_gap = ref.lp[0] - ref_lp_top;
        c.gap = l[c.top] - l[c.ref_top];
    }
    return c;
}

inline float pct(std::vector<float> & v, double p) {
    if (v.empty()) return 0;
    const size_t k = std::min(v.size() - 1, (size_t) (p * v.size()));
    std::nth_element(v.begin(), v.begin() + k, v.end());
    return v[k];
}

// Consumes an engine's logits rows sequence by sequence, in position order: compares them with the
// reference file `kld_path` and / or writes them as a new reference file `out_path`.  The sequences
// come from the reference file when there is one, else from `seqs`.
class Session {
  public:
    Session(const std::string & kld_path, const std::string & out_path, std::vector<Seq> seqs, int n_vocab)
        : seqs_(std::move(seqs)), nv_(n_vocab) {
        if (!kld_path.empty()) {
            in_ = fopen(kld_path.c_str(), "rb");
            if (!in_) throw std::runtime_error("cannot open " + kld_path);
            char magic[8];
            int hdr[3];
            get(magic, 8);
            get(hdr, sizeof(hdr));
            if (memcmp(magic, MAGIC, 8) || hdr[1] != REF_K) throw std::runtime_error(kld_path + ": not a reference file");
            if (hdr[0] != nv_) throw std::runtime_error(kld_path + ": vocabulary size differs");
            seqs_.resize(hdr[2]);
            for (Seq & s : seqs_) {
                int nf[2];
                get(nf, sizeof(nf));
                s.tok.resize(nf[0]);
                s.first = nf[1];
                get(s.tok.data(), s.tok.size() * sizeof(int));
            }
        }
        if (!out_path.empty()) {
            out_ = fopen(out_path.c_str(), "wb");
            if (!out_) throw std::runtime_error("cannot write " + out_path);
            const int hdr[3] = {nv_, REF_K, (int) seqs_.size()};
            fwrite(MAGIC, 1, 8, out_);
            fwrite(hdr, sizeof(hdr), 1, out_);
            for (const Seq & s : seqs_) {
                const int nf[2] = {(int) s.tok.size(), s.first};
                fwrite(nf, sizeof(nf), 1, out_);
                fwrite(s.tok.data(), sizeof(int), s.tok.size(), out_);
            }
        }
    }
    ~Session() {
        if (in_) fclose(in_);
        if (out_) fclose(out_);
    }

    const std::vector<Seq> & seqs() const { return seqs_; }

    // logits ([n][n_vocab]) after positions i0 .. i0 + n - 1 of sequence s; unscored positions are skipped
    void rows(int s, int i0, int n, const float * logits) {
        const Seq & q = seqs_[s];
        const int lo = std::max(i0, q.first), hi = std::min(i0 + n, (int) q.tok.size() - 1);
        if (lo >= hi) return;
        const int m = hi - lo;
        std::vector<Rec> ref(in_ ? m : 0), own(out_ ? m : 0);
        std::vector<Cmp> cmp(in_ ? m : 0);
        if (in_) get(ref.data(), m * sizeof(Rec));
        par_rows(m, [&](int r) {
            const float * l = logits + (size_t) (lo - i0 + r) * nv_;
            const int next = q.tok[lo + r + 1];
            if (in_) cmp[r] = compare(l, nv_, next, ref[r]);
            if (out_) make_rec(l, nv_, next, own[r]);
        });
        if (out_) fwrite(own.data(), sizeof(Rec), m, out_);
        for (int r = 0; r < (int) cmp.size(); r++) {
            const Cmp & c = cmp[r];
            kld_.push_back(c.kld);
            nll_ += c.nll;
            ref_nll_ += c.ref_nll;
            const double d = (double) c.nll - c.ref_nll;
            d_ += d;
            d2_ += d * d;
            tail_ += ref[r].tail;
            if (c.top != c.ref_top) miss_.push_back({s, lo + r, c});
        }
    }

    // piece(id): token text for the list of the worst mismatches
    void report(const std::function<std::string(int)> & piece) {
        const double n = (double) kld_.size();
        if (n == 0) return;
        const double mean = std::accumulate(kld_.begin(), kld_.end(), 0.0) / n;
        printf("positions          %.0f in %zu sequences\n", n, seqs_.size());
        printf("top token agrees   %.0f / %.0f (%.3f%%)\n", n - miss_.size(), n, 100.0 * (n - miss_.size()) / n);
        printf("KL(ref || this)    mean %.3g  median %.3g  p99 %.3g  max %.3g nats (reference mass outside its top %d: mean %.2g)\n",
               mean, pct(kld_, 0.5), pct(kld_, 0.99), *std::max_element(kld_.begin(), kld_.end()), REF_K, tail_ / n);
        const double dm = d_ / n, se = std::sqrt(std::max(0.0, d2_ / n - dm * dm) / n);
        printf("perplexity         ref %.4f  this %.4f  (mean log-loss difference %+.2g +- %.1g)\n", std::exp(ref_nll_ / n),
               std::exp(nll_ / n), dm, se);
        if (miss_.empty()) return;
        // a near-tie has small gaps on both sides; a wide reference gap means the engines really disagree
        std::vector<float> g;
        int wide[3] = {0, 0, 0};
        for (const Miss & m : miss_) {
            g.push_back(m.c.ref_gap);
            wide[0] += m.c.ref_gap > 0.1f;
            wide[1] += m.c.ref_gap > 0.5f;
            wide[2] += m.c.ref_gap > 1.0f;
        }
        printf("mismatches         %zu; reference's lead of its own top token over this engine's: median %.3g  p90 %.3g  max %.3g nats"
               " (> 0.1: %d, > 0.5: %d, > 1: %d)\n",
               miss_.size(), pct(g, 0.5), pct(g, 0.9), *std::max_element(g.begin(), g.end()), wide[0], wide[1], wide[2]);
        std::sort(miss_.begin(), miss_.end(), [](const Miss & a, const Miss & b) { return a.c.ref_gap > b.c.ref_gap; });
        for (size_t k = 0; k < std::min<size_t>(10, miss_.size()); k++) {
            const Miss & m = miss_[k];
            printf("  seq %d pos %d: ref %d %s (lead %.3g)  this %d %s (lead %.3g)\n", m.s, m.pos, m.c.ref_top,
                   quoted(piece(m.c.ref_top)).c_str(), m.c.ref_gap, m.c.top, quoted(piece(m.c.top)).c_str(), m.c.gap);
        }
    }

  private:
    struct Miss {
        int s, pos;
        Cmp c;
    };
    static std::string quoted(const std::string & s) {
        std::string o = "\"";
        for (char ch : s) o += ch == '\n' ? std::string("\\n") : std::string(1, ch);
        return o + "\"";
    }
    void get(void * p, size_t n) {
        if (fread(p, 1, n, in_) != n) throw std::runtime_error("reference file is truncated");
    }

    std::vector<Seq> seqs_;
    int nv_;
    FILE *in_ = nullptr, *out_ = nullptr;
    std::vector<float> kld_;
    std::vector<Miss> miss_;
    double nll_ = 0, ref_nll_ = 0, d_ = 0, d2_ = 0, tail_ = 0;
};

} // namespace ls
