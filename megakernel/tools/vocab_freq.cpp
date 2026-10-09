// Count token frequencies over a list of text files (one path per line on stdin) and write the
// most frequent ids (plus all special/control tokens) as the MTP draft vocabulary.
//   vocab_freq model.gguf N out.bin < files.txt
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "llama.h"

int main(int argc, char ** argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: vocab_freq model.gguf N out.bin < files.txt\n");
        return 1;
    }
    const int N = atoi(argv[2]);
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.vocab_only = true;
    llama_model * m = llama_model_load_from_file(argv[1], mp);
    const llama_vocab * v = llama_model_get_vocab(m);
    const int nv = llama_vocab_n_tokens(v);
    std::vector<long> cnt(nv, 0);
    std::string path;
    long total = 0, files = 0;
    std::vector<llama_token> toks;
    while (std::getline(std::cin, path)) {
        std::ifstream f(path, std::ios::binary);
        if (!f) continue;
        std::stringstream ss;
        ss << f.rdbuf();
        std::string text = ss.str();
        size_t off = 0;
        while (off < text.size()) {
            size_t end = std::min(text.size(), off + (1 << 16));
            if (end < text.size()) {
                size_t nl = text.rfind('\n', end);
                if (nl != std::string::npos && nl > off) end = nl + 1;
            }
            std::string chunk = text.substr(off, end - off);
            off = end;
            toks.resize(chunk.size() + 16);
            int n = -1;
            try {
                n = llama_tokenize(v, chunk.c_str(), (int) chunk.size(), toks.data(), (int) toks.size(), false, false);
            } catch (...) {
                continue; // not valid UTF-8
            }
            if (n < 0) continue;
            for (int i = 0; i < n; i++) cnt[toks[i]]++;
            total += n;
        }
        files++;
    }
    std::vector<int> ids(nv);
    for (int i = 0; i < nv; i++) ids[i] = i;
    std::stable_sort(ids.begin(), ids.end(), [&](int a, int b) { return cnt[a] > cnt[b]; });
    std::vector<int> out;
    std::vector<char> used(nv, 0);
    for (int i = 0; i < nv; i++) { // control / special tokens always draftable
        if (llama_vocab_get_attr(v, i) & (LLAMA_TOKEN_ATTR_CONTROL | LLAMA_TOKEN_ATTR_USER_DEFINED)) {
            out.push_back(i);
            used[i] = 1;
        }
    }
    for (int id : ids) {
        if ((int) out.size() >= N) break;
        if (!used[id]) {
            out.push_back(id);
            used[id] = 1;
        }
    }
    std::sort(out.begin(), out.end());
    long covered = 0;
    for (int id : out) covered += cnt[id];
    FILE * fo = fopen(argv[3], "wb");
    fwrite(out.data(), sizeof(int), out.size(), fo);
    fclose(fo);
    fprintf(stderr, "%ld files, %ld tokens, %zu ids written, coverage %.3f%%\n", files, total, out.size(),
            100.0 * covered / std::max(1L, total));
    llama_model_free(m);
    return 0;
}
