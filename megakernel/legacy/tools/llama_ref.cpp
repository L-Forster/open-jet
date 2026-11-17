// The llama.cpp side of the megakernel's accuracy check: teacher-forces token sequences through
// libllama and writes its next-token distributions as a reference file, and / or compares them with
// an existing reference file (KL divergence, perplexity, top-token agreement; see logit_stats.h).
//
//   llama_ref -m model.gguf -f text [-f text ...] --ref-out llama.ref    reference from text files
//   llama_ref -m model.gguf --kld llama.ref -ngl 0                       another llama.cpp setup against it
//   qwen38_mk -m model.gguf --kld llama.ref                              the megakernel against it
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "../logit_stats.h"
#include "llama.h"

static void usage() {
    fprintf(stderr,
            "usage: llama_ref -m model.gguf (-f text ... --ref-out f | --kld ref [--ref-out f])\n"
            "                 [--chunk n]        (tokens per sequence when reading text, default 2048)\n"
            "                 [--max-chunks n]   (default: all)\n"
            "                 [--score-from i]   (first scored position of each sequence, default 0)\n"
            "                 [-ngl n]           (GPU layers, default all; 0: CPU)\n"
            "                 [--kv f16|q8]      (KV cache format, default q8 as open-jet runs llama-server)\n"
            "                 [-b n]             (scored tokens per batch, default 512; 1: token by token, as llama.cpp\n"
            "                                     generates; the positions before --score-from are always batched, like a prompt)\n"
            "                 [-t threads]\n");
    exit(1);
}

int main(int argc, char ** argv) {
    std::string model, kld_file, ref_out, text;
    int chunk = 2048, max_chunks = 0, first = 0, ngl = -1, threads = 0, nb = 512;
    bool kv_q8 = true;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() -> std::string { if (i + 1 >= argc) usage(); return argv[++i]; };
        if (a == "-m") model = next();
        else if (a == "-f") { std::ifstream f(next(), std::ios::binary); std::stringstream ss; ss << f.rdbuf(); text += ss.str(); }
        else if (a == "--kld") kld_file = next();
        else if (a == "--ref-out") ref_out = next();
        else if (a == "--chunk") chunk = atoi(next().c_str());
        else if (a == "--max-chunks") max_chunks = atoi(next().c_str());
        else if (a == "--score-from") first = atoi(next().c_str());
        else if (a == "-ngl") ngl = atoi(next().c_str());
        else if (a == "--kv") kv_q8 = next() == "q8";
        else if (a == "-b") nb = atoi(next().c_str());
        else if (a == "-t") threads = atoi(next().c_str());
        else usage();
    }
    if (model.empty() || (kld_file.empty() && (text.empty() || ref_out.empty()))) usage();

    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = ngl;
    llama_model * lm = llama_model_load_from_file(model.c_str(), mp);
    if (!lm) { fprintf(stderr, "failed to load the model\n"); return 1; }
    const llama_vocab * vocab = llama_model_get_vocab(lm);
    const int nv = llama_vocab_n_tokens(vocab);

    std::vector<ls::Seq> seqs;
    if (kld_file.empty()) {
        std::vector<llama_token> toks(text.size() + 16);
        const int nt = llama_tokenize(vocab, text.c_str(), (int) text.size(), toks.data(), (int) toks.size(), false, false);
        if (nt < 0) { fprintf(stderr, "tokenize failed\n"); return 1; }
        for (int o = 0; o < nt && (max_chunks == 0 || (int) seqs.size() < max_chunks); o += chunk) {
            const int n = std::min(chunk, nt - o);
            if (n > first + 1) seqs.push_back({std::vector<int>(toks.begin() + o, toks.begin() + o + n), first});
        }
        if (seqs.empty()) { fprintf(stderr, "the text is too short\n"); return 1; }
    }
    ls::Session ses(kld_file, ref_out, seqs, nv);

    size_t longest = 0;
    for (const ls::Seq & q : ses.seqs()) longest = std::max(longest, q.tok.size());
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = (uint32_t) longest + 16;
    const int pre = 512; // batch size of the unscored prefix
    cp.n_batch = cp.n_ubatch = std::max(pre, nb);
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    cp.type_k = cp.type_v = kv_q8 ? GGML_TYPE_Q8_0 : GGML_TYPE_F16;
    if (threads > 0) cp.n_threads = cp.n_threads_batch = threads;
    llama_context * ctx = llama_init_from_model(lm, cp);
    if (!ctx) { fprintf(stderr, "failed to create the context\n"); return 1; }

    llama_batch batch = llama_batch_init(std::max(pre, nb), 0, 1);
    for (size_t s = 0; s < ses.seqs().size(); s++) {
        const ls::Seq & q = ses.seqs()[s];
        const int N = (int) q.tok.size();
        llama_memory_clear(llama_get_memory(ctx), true);
        for (int i0 = 0, n; i0 < N; i0 += n) {
            n = i0 < q.first ? std::min(pre, q.first - i0) : std::min(nb, N - i0);
            // logits only for the scored positions, which are consecutive within a batch
            const int lo = std::max(i0, q.first), hi = std::min(i0 + n, N - 1);
            batch.n_tokens = n;
            for (int k = 0; k < n; k++) {
                batch.token[k] = q.tok[i0 + k];
                batch.pos[k] = i0 + k;
                batch.n_seq_id[k] = 1;
                batch.seq_id[k][0] = 0;
                batch.logits[k] = i0 + k >= lo && i0 + k < hi;
            }
            if (llama_decode(ctx, batch) != 0) { fprintf(stderr, "llama_decode failed\n"); return 1; }
            if (lo < hi) ses.rows((int) s, lo, hi - lo, llama_get_logits(ctx));
        }
        fprintf(stderr, "\r%zu / %zu sequences", s + 1, ses.seqs().size());
    }
    fprintf(stderr, "\n");
    ses.report([&](int id) {
        char buf[256];
        const int n = llama_token_to_piece(vocab, id, buf, sizeof(buf), 0, false);
        return std::string(buf, n > 0 ? n : 0);
    });
    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(lm);
    return 0;
}
