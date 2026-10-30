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
    int ms_n = 0, ms_new = 64; // --ms S: greedy decode of S prompt variants at once (test path), --ms-new tokens each
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
        else if (a == "--ms") ms_n = atoi(next().c_str());
        else if (a == "--ms-new") ms_new = atoi(next().c_str());
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

    if (ms_n > 0) {
        auto tok_text = [&](const std::string & t) {
            std::vector<llama_token> v(t.size() + 16);
            const int k = llama_tokenize(vocab, t.c_str(), (int) t.size(), v.data(), (int) v.size(), false, true);
            if (k < 0) { fprintf(stderr, "tokenize failed\n"); exit(1); }
            return std::vector<int>(v.begin(), v.begin() + k);
        };
        const char * topics[] = {"a lighthouse keeper", "a robot learning to paint", "the history of tea", "a city under the sea",
                                 "a chess tournament", "a mountain rescue", "a time-travelling librarian", "a desert caravan"};
        std::vector<std::vector<int>> prompts;
        for (int s = 0; s < ms_n; s++) {
            const std::string p = prompt + " The topic: " + topics[s % 8] + ".";
            prompts.push_back(tok_text(raw ? p : sys + "<|im_start|>user\n" + p + "<|im_end|>\n" + gen_prompt));
        }
        size_t longest = 0;
        for (auto & pr : prompts) longest = std::max(longest, pr.size());
        const int ctx_s = (int) longest + 3 * ms_new + 128; // room for the second turn
        double secs = 0;
        const bool tc = getenv("MS_TC") && atoi(getenv("MS_TC"));
        auto outs = eng.ms_generate(prompts, ms_new, ctx_s, secs, tc);
        printf("MS batched (%s): S=%d, %d tokens each: %.2f s, %.1f tok/s per sequence, %.1f tok/s total\n", tc ? "tensor-core" : "fp16 gemm",
               ms_n, ms_new, secs, ms_new / secs, ms_n * ms_new / secs);
        if (tc) { // agreement with the fp16-GEMM path
            double s2 = 0;
            auto ref = eng.ms_generate(prompts, ms_new, ctx_s, s2, false);
            for (int s = 0; s < ms_n; s++) {
                int k = 0;
                while (k < ms_new && ref[s][k] == outs[s][k]) k++;
                printf("  seq %d: tensor-core == fp16-gemm path for the first %d / %d tokens\n", s, k, ms_new);
            }
        }
        int same = 0;
        double secs1 = 0;
        for (int s = 0; s < ms_n; s++) {
            double t1 = 0;
            auto one = eng.ms_generate({prompts[s]}, ms_new, ctx_s, t1, tc);
            secs1 += t1;
            int k = 0;
            while (k < ms_new && one[0][k] == outs[s][k]) k++;
            same += k == ms_new;
            printf("  seq %d: batched == alone for %d / %d tokens\n", s, k, ms_new);
        }
        printf("MS alone: %.1f tok/s per sequence; %d / %d sequences identical\n", ms_n * ms_new / secs1, same, ms_n);
        if (getenv("MS_SPEC") && temp > 0.f) { // sampled speculative decode: text and speed
            double ss = 0, ar = 0;
            auto sp = eng.ms_generate_spec(prompts, ms_new, ctx_s + 8, ss, 0, nullptr, &ar);
            printf("MS spec sampled (temp %.2f top_p %.2f top_k %d): %.1f tok/s per sequence, %.1f tok/s total, acceptance %.2f\n", temp, top_p,
                   top_k, ms_new / ss, ms_n * ms_new / ss, ar);
            for (int s = 0; s < ms_n; s++) {
                std::string t;
                for (int id : sp[s]) t += piece(id);
                printf("  [%d] %s\n", s, t.c_str());
            }
        } else if (getenv("MS_SPEC")) { // speculative path: oracle drafts must reproduce the greedy tokens exactly
            for (int oracle : {1, 2, 0}) {
                double ss = 0, ar = 0;
                auto sp = eng.ms_generate_spec(prompts, ms_new, ctx_s + 8, ss, oracle, &outs, &ar);
                int same = 0;
                for (int s = 0; s < ms_n; s++) {
                    int k = 0;
                    while (k < ms_new && sp[s][k] == outs[s][k]) k++;
                    same += k;
                    printf("  spec (%s) seq %d: == greedy for the first %d / %d tokens\n", oracle == 1 ? "oracle-right" : oracle == 2 ? "oracle-wrong" : "MTP", s, k, ms_new);
                }
                printf("MS spec (%s drafts): %.1f tok/s per sequence, %.1f tok/s total, acceptance %.2f, %d / %d tokens equal greedy\n",
                       oracle == 1 ? "oracle-right" : oracle == 2 ? "oracle-wrong" : "MTP", ms_new / ss, ms_n * ms_new / ss, ar, same, ms_n * ms_new);
            }
            // a second turn extending each slot's sequence: reuses the slot (prefills only the new tokens). Not bitwise a
            // fresh prefill: the delta-net scan rounds its state to fp16 per 32-token chunk, so moving the split point
            // changes h by ~3e-3 (checked below against two full prefills, which are bitwise equal) and greedy text
            // can flip at near-ties
            std::vector<std::vector<int>> p2;
            const std::vector<int> more = tok_text("<|im_end|>\n<|im_start|>user\nNow describe the main character in one sentence.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n");
            for (int s = 0; s < ms_n; s++) {
                auto q = eng.slot_sequence(s);
                q.insert(q.end(), more.begin(), more.end());
                p2.push_back(q);
            }
            size_t l2 = 0;
            for (auto & q : p2) l2 = std::max(l2, q.size());
            double s2a = 0, s2b = 0;
            const auto t2 = std::chrono::steady_clock::now();
            auto turn2 = eng.ms_generate_spec(p2, ms_new, ctx_s + 8, s2a);
            const double tt2 = std::chrono::duration<double>(std::chrono::steady_clock::now() - t2).count();
            auto fresh = eng.ms_generate(p2, ms_new, ctx_s + 8, s2b, true); // (same slot context: no reallocation)
            int eq = 0;
            for (int s = 0; s < ms_n; s++) {
                int k = 0;
                while (k < ms_new && turn2[s][k] == fresh[s][k]) k++;
                eq += k;
                printf("  turn 2 seq %d (%zu prompt tokens): reusing slot == fresh prefill for %d / %d tokens\n", s, p2[s].size(), k, ms_new);
            }
            printf("MS turn 2: %d / %d tokens equal (near-tie flips expected), %.2f s including prefill of the new tokens\n", eq, ms_n * ms_new, tt2);
            auto rel = [](const std::vector<float> & a, const std::vector<float> & b) {
                double d = 0, n = 0;
                for (size_t i = 0; i < a.size(); i++) { d += (a[i] - b[i]) * (a[i] - b[i]); n += b[i] * b[i]; }
                return std::sqrt(d / n);
            };
            for (int s = 0; s < ms_n; s++) {
                std::vector<int> a(p2[s].begin(), p2[s].end() - (long) more.size());
                const std::vector<int> x = {a[0] == 0 ? 1 : 0};
                eng.slot_start(s, x, 0.f, 1.f, 0, 0.f, 0); eng.slot_start(s, p2[s], 0.f, 1.f, 0, 0.f, 0);
                auto hf = eng.slot_h(s);
                eng.slot_start(s, x, 0.f, 1.f, 0, 0.f, 0); eng.slot_start(s, p2[s], 0.f, 1.f, 0, 0.f, 0);
                auto hf2 = eng.slot_h(s);
                eng.slot_start(s, x, 0.f, 1.f, 0, 0.f, 0); eng.slot_start(s, a, 0.f, 1.f, 0, 0.f, 0);
                const int r = eng.slot_start(s, p2[s], 0.f, 1.f, 0, 0.f, 0);
                auto he = eng.slot_h(s);
                std::vector<int> b(p2[s].begin(), p2[s].end() - 7);
                eng.slot_start(s, x, 0.f, 1.f, 0, 0.f, 0); eng.slot_start(s, b, 0.f, 1.f, 0, 0.f, 0);
                eng.slot_start(s, p2[s], 0.f, 1.f, 0, 0.f, 0);
                auto hb7 = eng.slot_h(s);
                printf("  h rel diff seq %d: full vs full %.2e, extend (from %d) vs full %.2e, extend (last 7) vs full %.2e\n", s, rel(hf2, hf), r, rel(he, hf), rel(hb7, hf));
            }
            if (getenv("MS_SEG_TEST")) { // shared segments: one slot makes the shared prefix a segment, the others attach
                std::string body;
                const int nrules = getenv("MS_SEG_RULES") ? atoi(getenv("MS_SEG_RULES")) : 160;
                for (int i = 0; i < nrules; i++) body += "Rule " + std::to_string(i) + ": every agent must log step " + std::to_string(i * 7 % 13) + ". ";
                const std::vector<int> pre = tok_text("<|im_start|>system\n" + body + "<|im_end|>\n");
                std::vector<std::vector<int>> ps;
                for (int s = 0; s < ms_n; s++) {
                    auto q = pre;
                    auto u = tok_text(std::string("<|im_start|>user\nTask ") + std::to_string(s) + ": summarise rule " + std::to_string(3 * s + 1) +
                                      ".<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n");
                    q.insert(q.end(), u.begin(), u.end());
                    ps.push_back(q);
                }
                const int share = (int) pre.size();
                eng.slots_open(ms_n, (int) pre.size() + 2048, (int) pre.size() + 512); // (room for the long prompts, and a segment)
                // reference: fresh prefills, no segment (share_len 0; a prompt differing in the first token drops the slot's state)
                std::vector<std::vector<float>> href(ms_n);
                std::vector<std::vector<int>> gref(ms_n);
                for (int s = 0; s < ms_n; s++) {
                    eng.slot_start(s, {ps[s][1]}, 0.f, 1.f, 0, 0.f, 0);
                    eng.slot_begin(s, ps[s], 0.f, 1.f, 0, 0.f, 0);
                    while (eng.slot_prefill(s, 512)) {}
                    href[s] = eng.slot_h(s);
                }
                std::vector<int> all(ms_n);
                for (int s = 0; s < ms_n; s++) all[s] = s;
                while ((int) gref[0].size() < ms_new) {
                    auto em = eng.slots_step(all);
                    for (int s = 0; s < ms_n; s++) gref[s].insert(gref[s].end(), em[s].begin(), em[s].end());
                }
                // shared: slot 0 makes the segment, the others attach
                std::vector<int> reused(ms_n);
                std::vector<std::vector<float>> hseg(ms_n);
                const auto t0 = std::chrono::steady_clock::now();
                for (int s = 0; s < ms_n; s++) {
                    eng.slot_start(s, {ps[s][1]}, 0.f, 1.f, 0, 0.f, 0);
                    reused[s] = eng.slot_begin(s, ps[s], 0.f, 1.f, 0, 0.f, 0, share);
                    while (eng.slot_prefill(s, 512)) {}
                    hseg[s] = eng.slot_h(s);
                }
                const double tp = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
                if (getenv("MK_PF_PROFILE")) eng.profile_report(); // (reset: the decode below alone)
                std::vector<std::vector<int>> gseg(ms_n);
                int nsteps = 0;
                const auto td = std::chrono::steady_clock::now();
                while ((int) gseg[0].size() < ms_new) {
                    auto em = eng.slots_step(all);
                    nsteps++;
                    for (int s = 0; s < ms_n; s++) gseg[s].insert(gseg[s].end(), em[s].begin(), em[s].end());
                }
                const double tdd = std::chrono::duration<double>(std::chrono::steady_clock::now() - td).count();
                printf("MS segment decode: %d steps, %.1f ms per step, %.1f tok/s per sequence\n", nsteps, 1e3 * tdd / nsteps, gseg[0].size() / tdd);
                if (getenv("MK_PF_PROFILE")) eng.profile_report();
                auto rel = [](const std::vector<float> & a, const std::vector<float> & b) {
                    double d = 0, n = 0;
                    for (size_t i = 0; i < a.size(); i++) { d += (a[i] - b[i]) * (a[i] - b[i]); n += b[i] * b[i]; }
                    return std::sqrt(d / n);
                };
                // the same split without a segment (prefill the prefix, then extend): the segment must change nothing
                std::vector<std::vector<float>> hsplit(ms_n);
                {
                    std::vector<int> keep(ms_n);
                    for (int s = 0; s < ms_n; s++) keep[s] = eng.slot_segment(s);
                    for (int s = 0; s < ms_n; s++) eng.slot_start(s, {ps[s][1]}, 0.f, 1.f, 0, 0.f, 0); // (detach)
                    eng.segments_drop();
                    for (int s = 0; s < ms_n; s++) {
                        std::vector<int> head(ps[s].begin(), ps[s].begin() + share + 1);
                        eng.slot_begin(s, head, 0.f, 1.f, 0, 0.f, 0);
                        while (eng.slot_prefill(s, 512)) {}
                        eng.slot_begin(s, ps[s], 0.f, 1.f, 0, 0.f, 0);
                        while (eng.slot_prefill(s, 512)) {}
                        hsplit[s] = eng.slot_h(s);
                    }
                    (void) keep;
                    std::vector<std::vector<int>> gsplit(ms_n);
                    while ((int) gsplit[0].size() < ms_new) {
                        auto em = eng.slots_step(all);
                        for (int s = 0; s < ms_n; s++) gsplit[s].insert(gsplit[s].end(), em[s].begin(), em[s].end());
                    }
                    for (int s = 0; s < ms_n; s++) {
                        int k = 0;
                        while (k < ms_new && gsplit[s][k] == gseg[s][k]) k++;
                        printf("  seg seq %d: decode with segment == same split without for %d / %d\n", s, k, ms_new);
                    }
                }
                for (int s = 0; s < ms_n; s++) printf("  seg seq %d: h with segment vs the same split without: rel diff %.2e\n", s, rel(hseg[s], hsplit[s]));
                for (int s = 0; s < ms_n; s++) {
                    int k = 0;
                    while (k < ms_new && gseg[s][k] == gref[s][k]) k++;
                    printf("  seg seq %d: prompt %zu, shared %d, reused %d, segment %d; h rel diff %.2e; greedy == unshared for %d / %d\n", s,
                           ps[s].size(), share, reused[s], eng.slot_segment(s), rel(hseg[s], href[s]), k, ms_new);
                }
                printf("MS segments: %d prompts of %zu tokens (%d shared) prefilled in %.2f s\n", ms_n, ps[0].size(), share, tp);
                // solo decode on a segment (one slot: the megakernel reads the segment), against batched on the segment
                for (int s = 0; s < ms_n; s++) eng.slot_start(s, {ps[s][1]}, 0.f, 1.f, 0, 0.f, 0);
                eng.segments_drop();
                for (int s = 0; s < 2; s++) { eng.slot_begin(s, ps[s], 0.f, 1.f, 0, 0.f, 0, share); while (eng.slot_prefill(s, 512)) {} }
                {
                    std::vector<int> b1;
                    while ((int) b1.size() < ms_new) { const auto em = eng.slots_step({0, 1}); b1.insert(b1.end(), em[1].begin(), em[1].end()); }
                    gseg[1] = b1;
                }
                eng.slot_begin(1, ps[1], 0.f, 1.f, 0, 0.f, 0, share);
                while (eng.slot_prefill(1, 512)) {}
                std::vector<int> g1;
                while ((int) g1.size() < ms_new) { const auto em = eng.slots_step({1}); g1.insert(g1.end(), em[0].begin(), em[0].end()); }
                int k = 0;
                while (k < ms_new && g1[k] == gseg[1][k]) k++;
                printf("MS segment solo (seq 1, segment %d): == batched for %d / %d\n", eng.slot_segment(1), k, ms_new);
            }
            // solo: a lone slot decodes through the megakernel, then hands back to the batched path mid-sequence
            {
                eng.slot_start(0, p2[0], 0.f, 1.f, 0, 0.f, 0);
                eng.slot_start(1, p2[1], 0.f, 1.f, 0, 0.f, 0);
                std::vector<int> a, b;
                const auto t3 = std::chrono::steady_clock::now();
                while ((int) a.size() < ms_new / 2) {
                    const auto em = eng.slots_step({0});
                    a.insert(a.end(), em[0].begin(), em[0].end());
                }
                const double ts = std::chrono::duration<double>(std::chrono::steady_clock::now() - t3).count();
                while ((int) a.size() < ms_new || (int) b.size() < ms_new) {
                    auto em = eng.slots_step({0, 1});
                    a.insert(a.end(), em[0].begin(), em[0].end());
                    b.insert(b.end(), em[1].begin(), em[1].end());
                }
                int ka = 0, kb = 0;
                while (ka < ms_new && a[ka] == fresh[0][ka]) ka++;
                while (kb < ms_new && b[kb] == fresh[1][kb]) kb++;
                printf("MS solo: %d tokens alone at %.1f tok/s, then batched; == fresh for %d / %d (slot 0), %d / %d (slot 1)\n",
                       ms_new / 2, ms_new / 2 / ts, ka, ms_new, kb, ms_new);
            }
        }
        std::string str;
        for (int id : outs[0]) str += piece(id);
        printf("seq 0: %s\n", str.c_str());
        // the megakernel's own greedy decode of prompt 0 (other arithmetic: agreement, not identity), timed
        std::vector<int> ref = {eng.prefill(prompts[0])};
        const auto tk0 = std::chrono::steady_clock::now();
        while ((int) ref.size() < ms_new) eng.step(ref);
        const double tk = std::chrono::duration<double>(std::chrono::steady_clock::now() - tk0).count();
        printf("megakernel single stream (-d %d): %.1f tok/s\n", n_draft, (ref.size() - 1) / tk);
        int k = 0;
        while (k < ms_new && k < (int) ref.size() && ref[k] == outs[0][k]) k++;
        printf("seq 0 vs megakernel greedy decode: first %d / %d tokens agree\n", k, ms_new);
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
