// CLI for the Qwen3.8-27B megakernel: tokenises with libllama (vocab only), runs greedy
// speculative decoding and reports decode throughput.
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "llama.h"
#include "logit_stats.h"
#include "qwen38_mk.h"

static void usage() {
    fprintf(stderr,
            "usage: qwen38_mk -m model.gguf [-p prompt | -f prompt_file] [-n n_predict] [-d n_draft]\n"
            "                 [-c ctx] [--draft-vocab ids.bin] [--raw] [--ids] [--quiet]\n"
            "                 [--turn2 text | --turn2-file f]   (a second user turn, appended to the first)\n"
            "                 [--prompt-ids f]   (prompt given as whitespace-separated token ids)\n"
            "                 [--think [xhigh|medium|low]]   (reasoning mode, the chat template's default effort is xhigh)\n"
            "                 [--temp t] [--top-p p] [--top-k k] [--min-p p] [--seed s]   (default: greedy)\n"
            "                 [--kv f16|q8|auto]   (KV cache format; auto: q8_0 only when fp16 does not fit)\n"
            "                 [--kld ref [--ref-out f] [--score-path prefill|decode|mixed]]   (KL divergence, perplexity and\n"
            "                     top-token agreement against a reference file of tools/llama_ref or of --ref-out)\n");
    exit(1);
}

int main(int argc, char ** argv) {
    std::string model, prompt = "Write a Python function that checks whether a number is prime.";
    std::string draft_vocab;
    int n_predict = 256, n_draft = 2, ctx = 8192;
    bool raw = false, print_ids = false, quiet = false, bench_phases = false;
    std::string score_file; // file with whitespace-separated continuation token ids to teacher-force
    std::string turn2;      // optional second user turn
    std::string prompt_ids_file;
    std::string think;      // reasoning effort; empty: thinking disabled (empty think block)
    // teacher-forced comparison with a reference file; its sequences go through the batched prefill
    // kernels, the decode megakernel, or (mixed) the unscored prefix through the former, like a prompt,
    // and the scored positions through the latter (the first half and the rest when all are scored)
    std::string kld_file, ref_out, score_path = "prefill";
    float temp = 0.f, top_p = 1.f, min_p = 0.f;
    int top_k = 32;
    unsigned long long seed = 1;
    int kv_mode = -1;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() -> std::string { if (i + 1 >= argc) usage(); return argv[++i]; };
        if (a == "-m") model = next();
        else if (a == "-p") prompt = next();
        else if (a == "-f") { std::ifstream f(next()); std::stringstream ss; ss << f.rdbuf(); prompt = ss.str(); }
        else if (a == "-n") n_predict = atoi(next().c_str());
        else if (a == "-d") n_draft = atoi(next().c_str());
        else if (a == "-c") ctx = atoi(next().c_str());
        else if (a == "--draft-vocab") draft_vocab = next();
        else if (a == "--raw") raw = true;
        else if (a == "--ids") print_ids = true;
        else if (a == "--quiet") quiet = true;
        else if (a == "--bench-phases") bench_phases = true;
        else if (a == "--score") score_file = next();
        else if (a == "--kld") kld_file = next();
        else if (a == "--ref-out") ref_out = next();
        else if (a == "--score-path") score_path = next();
        else if (a == "--turn2") turn2 = next();
        else if (a == "--prompt-ids") prompt_ids_file = next();
        else if (a == "--think") think = (i + 1 < argc && argv[i + 1][0] != '-') ? next() : "xhigh";
        else if (a == "--temp") temp = atof(next().c_str());
        else if (a == "--top-p") top_p = atof(next().c_str());
        else if (a == "--top-k") top_k = atoi(next().c_str());
        else if (a == "--min-p") min_p = atof(next().c_str());
        else if (a == "--seed") seed = strtoull(next().c_str(), nullptr, 10);
        else if (a == "--kv") { const std::string k = next(); kv_mode = k == "f16" ? 0 : k == "q8" ? 1 : -1; }
        else if (a == "--turn2-file") { std::ifstream f(next()); std::stringstream ss; ss << f.rdbuf(); turn2 = ss.str(); }
        else usage();
    }
    if (model.empty() || (kld_file.empty() && !ref_out.empty())) usage();
    if (score_path != "prefill" && score_path != "decode" && score_path != "mixed") usage();

    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.vocab_only = true;
    llama_model * lm = llama_model_load_from_file(model.c_str(), mp);
    if (!lm) { fprintf(stderr, "failed to load vocab\n"); return 1; }
    const llama_vocab * vocab = llama_model_get_vocab(lm);

    // chat template of the model (llama.cpp renders the same): with thinking, an optional reasoning-effort
    // system message and an open <think> block; without, an empty think block
    auto reasoning_line = [&]() -> std::string {
        if (think == "xhigh")
            return "Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, "
                   "consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.";
        if (think == "low")
            return "Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to the conclusion "
                   "without unnecessary elaboration.";
        if (!think.empty() && think != "medium") { fprintf(stderr, "--think: xhigh, medium or low\n"); exit(1); }
        return "";
    };
    const std::string gen_prompt = think.empty() ? "<|im_start|>assistant\n<think>\n\n</think>\n\n" : "<|im_start|>assistant\n<think>\n";
    const std::string sys = reasoning_line().empty() ? "" : "<|im_start|>system\n" + reasoning_line() + "<|im_end|>\n";
    std::string text = raw ? prompt : sys + "<|im_start|>user\n" + prompt + "<|im_end|>\n" + gen_prompt;
    std::vector<llama_token> toks(text.size() + 16);
    int nt = llama_tokenize(vocab, text.c_str(), (int) text.size(), toks.data(), (int) toks.size(), false, true);
    if (nt < 0) { fprintf(stderr, "tokenize failed\n"); return 1; }
    toks.resize(nt);

    auto piece = [&](int id) {
        char buf[256];
        int n = llama_token_to_piece(vocab, id, buf, sizeof(buf), 0, false);
        return std::string(buf, n > 0 ? n : 0);
    };

    if (!kld_file.empty()) {
        ls::Session ses(kld_file, ref_out, {}, MKEngine::n_vocab());
        size_t longest = 0;
        for (const ls::Seq & q : ses.seqs()) longest = std::max(longest, q.tok.size());
        MKEngine eng(model, std::max(ctx, (int) longest + 64), 0, "", kv_mode);
        for (size_t s = 0; s < ses.seqs().size(); s++) {
            const ls::Seq & q = ses.seqs()[s];
            const int N = (int) q.tok.size();
            const int mk_from = score_path == "decode" ? 0 : score_path == "prefill" ? N : q.first > 0 ? q.first : N / 2;
            eng.score_logits(q.tok, mk_from, q.first, [&](int i0, int n, const float * l) { ses.rows((int) s, i0, n, l); });
            fprintf(stderr, "\r%zu / %zu sequences", s + 1, ses.seqs().size());
        }
        fprintf(stderr, "\n");
        ses.report(piece);
        return 0;
    }

    MKEngine eng(model, ctx, n_draft, draft_vocab, kv_mode);
    eng.set_sampling(temp, top_p, top_k, min_p, seed);
    if (bench_phases) {
        eng.bench_phases();
        return 0;
    }

    std::vector<int> prompt_ids(toks.begin(), toks.end());
    if (!prompt_ids_file.empty()) {
        std::ifstream f(prompt_ids_file);
        prompt_ids.clear();
        int id;
        while (f >> id) prompt_ids.push_back(id);
    }
    if (!score_file.empty()) {
        std::ifstream f(score_file);
        std::vector<int> cont;
        int id;
        while (f >> id) cont.push_back(id);
        std::vector<int> all = prompt_ids;
        all.insert(all.end(), cont.begin(), cont.end());
        std::vector<int> pred = eng.score(all);
        int agree = 0, total = 0, first_bad = -1;
        for (size_t i = prompt_ids.size() - 1; i + 1 < all.size(); i++) {
            total++;
            if (pred[i] == all[i + 1]) agree++;
            else if (first_bad < 0) first_bad = (int) (i + 1 - prompt_ids.size());
        }
        printf("SCORE agree %d / %d (%.2f%%), first disagreement at continuation index %d\n", agree, total,
               100.0 * agree / total, first_bad);
        return 0;
    }
    auto tokenize = [&](const std::string & t) {
        std::vector<llama_token> v(t.size() + 16);
        const int k = llama_tokenize(vocab, t.c_str(), (int) t.size(), v.data(), (int) v.size(), false, true);
        if (k < 0) { fprintf(stderr, "tokenize failed\n"); exit(1); }
        return std::vector<int>(v.begin(), v.begin() + k);
    };
    // one turn: prefill (or append) `in`, then decode up to n_predict tokens or end of turn
    auto turn = [&](const std::vector<int> & in, const char * tag) {
        const MKStats s0 = eng.stats;
        auto t0 = std::chrono::steady_clock::now();
        int first = eng.prefill(in);
        auto t1 = std::chrono::steady_clock::now();
        std::vector<int> out = {first};
        bool eog = llama_vocab_is_eog(vocab, first);
        while (!eog && (int) out.size() < n_predict) {
            size_t before = out.size();
            eng.step(out);
            for (size_t i = before; i < out.size(); i++) {
                if (llama_vocab_is_eog(vocab, out[i])) {
                    out.resize(i + 1);
                    eog = true;
                    break;
                }
            }
        }
        if ((int) out.size() > n_predict) out.resize(n_predict);
        auto t2 = std::chrono::steady_clock::now();
        if (!quiet) {
            std::string str;
            for (int id : out) str += piece(id);
            printf("%s\n", str.c_str());
        }
        if (print_ids) {
            printf("IN_%s:", tag);
            for (int id : in) printf(" %d", id);
            printf("\n%s:", tag);
            for (int id : out) printf(" %d", id);
            printf("\n");
        }
        const double pf = std::chrono::duration<double>(t1 - t0).count();
        const double dc = std::chrono::duration<double>(t2 - t1).count();
        const int n_dec = (int) out.size() - 1;
        const long cyc = eng.stats.cycles - s0.cycles, dr = eng.stats.drafted - s0.drafted, ac = eng.stats.accepted - s0.accepted;
        fprintf(stderr, "\n[mk] prompt %d tok in %.3f s (%.1f tok/s)\n", (int) in.size(), pf, in.size() / pf);
        fprintf(stderr, "[mk] decode %d tok in %.3f s = %.2f tok/s | cycles %ld, accepted %ld/%ld drafts (%.1f%%), %.2f tok/cycle\n",
                n_dec, dc, n_dec / dc, cyc, ac, dr, dr ? 100.0 * ac / dr : 0.0, cyc ? (double) n_dec / cyc : 0.0);
        return std::make_pair(out, eog);
    };
    const int plen = (int) prompt_ids.size();
    auto [out1, eog1] = turn(prompt_ids, "IDS");
    if (!turn2.empty()) {
        // the engine may have run past the end of turn (or n_predict) within the last cycle
        eng.truncate(plen + (int) out1.size());
        const std::string t = (eog1 ? "\n" : "<|im_end|>\n") + std::string("<|im_start|>user\n") + turn2 + "<|im_end|>\n" + gen_prompt;
        turn(tokenize(t), "IDS2");
    }
    llama_model_free(lm);
    return 0;
}
