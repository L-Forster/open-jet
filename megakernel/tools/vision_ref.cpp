// The llama.cpp side of the megakernel's image-input accuracy check: an image prompt through libllama + libmtmd
// (llama-server's code path: mtmd_tokenize, the CPU image encoder, mtmd_helper_decode_image_chunk with its M-RoPE
// positions), teacher-forced, its next-token distributions written as a reference file (logit_stats.h) and the
// image rows it fed as an images file, so that the megakernel scores the same tokens with the same rows:
//
//   vision_ref -m model.gguf --mmproj mmproj.gguf --prompt-file p.txt --image a.png [--image b.jpg ...]
//              [--cont-file answer.txt] --ref-out ref --images-out imgs [--kv q8|f16] [-ngl n] [-t threads]
//              [--rows-in imgs]   (the images' rows from an images file instead of libmtmd's encoder)
//   qwen38_mk -m model.gguf --kld ref --kld-images imgs [--score-path prefill|mixed]
//
// p.txt: a rendered chat prompt with one <__media__> marker per --image (the image goes there, between
// <|vision_start|> and <|vision_end|>, as llama-server renders image_url parts); cont-file: text appended after it
// (e.g. an answer to teacher-force). Scored: every position from the last image's end on.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "../logit_stats.h"
#include "llama.h"
#include "mtmd-helper.h"
#include "mtmd.h"

static void usage() {
    fprintf(stderr, "usage: vision_ref -m model.gguf --mmproj mmproj.gguf --prompt-file p.txt --image f [--image f ...]\n"
                    "                  [--cont-file f] --ref-out ref --images-out imgs [--kv q8|f16] [-ngl n] [-t threads]\n"
                    "                  [--image-max-tokens n]\n");
    exit(1);
}

static std::string slurp(const std::string & p) {
    std::ifstream f(p, std::ios::binary);
    if (!f) {
        fprintf(stderr, "cannot read %s\n", p.c_str());
        exit(1);
    }
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

int main(int argc, char ** argv) {
    std::string model, mmproj, prompt, cont, ref_out, img_out;
    std::vector<std::string> images;
    int ngl = -1, threads = 0, max_tok = -1, tb = 512;
    std::string rows_file;
    bool kv_q8 = true;
    for (int i = 1; i < argc; i++) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string { if (i + 1 >= argc) usage(); return argv[++i]; };
        if (a == "-m") model = next();
        else if (a == "--mmproj") mmproj = next();
        else if (a == "--prompt-file") prompt = slurp(next());
        else if (a == "--cont-file") cont = slurp(next());
        else if (a == "--image") images.push_back(next());
        else if (a == "--ref-out") ref_out = next();
        else if (a == "--images-out") img_out = next();
        else if (a == "--kv") kv_q8 = next() == "q8";
        else if (a == "-ngl") ngl = atoi(next().c_str());
        else if (a == "-t") threads = atoi(next().c_str());
        else if (a == "--image-max-tokens") max_tok = atoi(next().c_str());
        else if (a == "--text-batch") tb = atoi(next().c_str()); // (diagnostics: text decoded in batches of tb)
        else if (a == "--rows-in") rows_file = next(); // the images' rows from an images file (MKIMG01), not the encoder
        else usage();
    }
    if (model.empty() || mmproj.empty() || prompt.empty() || images.empty() || ref_out.empty() || img_out.empty()) usage();

    std::vector<std::vector<float>> rows_in;
    if (!rows_file.empty()) {
        FILE * f = fopen(rows_file.c_str(), "rb");
        char magic[8];
        int hdr[2];
        if (!f || fread(magic, 1, 8, f) != 8 || memcmp(magic, "MKIMG01", 8) || fread(hdr, sizeof(hdr), 1, f) != 1) {
            fprintf(stderr, "bad --rows-in\n");
            return 1;
        }
        for (int k = 0; k < hdr[1]; k++) {
            int v[4];
            bool ok = fread(v, sizeof(v), 1, f) == 1;
            if (ok) {
                rows_in.emplace_back((size_t) v[2] * v[3] * hdr[0]);
                ok = fread(rows_in.back().data(), sizeof(float), rows_in.back().size(), f) == rows_in.back().size();
            }
            if (!ok) {
                fprintf(stderr, "truncated --rows-in\n");
                return 1;
            }
        }
        fclose(f);
    }
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = ngl;
    llama_model * lm = llama_model_load_from_file(model.c_str(), mp);
    if (!lm) { fprintf(stderr, "failed to load the model\n"); return 1; }
    const llama_vocab * vocab = llama_model_get_vocab(lm);
    const int nv = llama_vocab_n_tokens(vocab);

    mtmd_context_params mcp = mtmd_context_params_default();
    mcp.use_gpu = false; // the megakernel server's encoder runs on the CPU too (same rows either way: they are saved)
    mcp.print_timings = true;
    mcp.n_threads = threads > 0 ? threads : 16;
    mcp.image_max_tokens = max_tok;
    mtmd_context * mt = mtmd_init_from_file(mmproj.c_str(), lm, mcp);
    if (!mt) { fprintf(stderr, "failed to load the mmproj\n"); return 1; }

    // tokenize as llama-server does
    std::vector<mtmd_bitmap *> bms;
    for (const std::string & p : images) {
        mtmd_helper_bitmap_wrapper bw = mtmd_helper_bitmap_init_from_file(mt, p.c_str(), false);
        if (!bw.bitmap) { fprintf(stderr, "cannot load %s\n", p.c_str()); return 1; }
        bms.push_back(bw.bitmap);
    }
    const std::string text = prompt + cont;
    mtmd_input_chunks * chunks = mtmd_input_chunks_init();
    mtmd_input_text it = {text.c_str(), text.size(), true, true};
    if (mtmd_tokenize(mt, chunks, &it, const_cast<const mtmd_bitmap **>(bms.data()), bms.size()) != 0) {
        fprintf(stderr, "mtmd_tokenize failed (one <__media__> marker per --image?)\n");
        return 1;
    }
    std::vector<llama_token> pad(1);
    if (llama_tokenize(vocab, "<|image_pad|>", 13, pad.data(), 1, false, true) != 1) { fprintf(stderr, "no <|image_pad|>\n"); return 1; }

    // the sequence (image rows as <|image_pad|> placeholders) and the images' places in it
    struct Img {
        int pos, nx, ny;
        std::vector<float> rows;
    };
    std::vector<int> tok;
    std::vector<Img> imgs;
    for (size_t c = 0; c < mtmd_input_chunks_size(chunks); c++) {
        const mtmd_input_chunk * ch = mtmd_input_chunks_get(chunks, c);
        if (mtmd_input_chunk_get_type(ch) == MTMD_INPUT_CHUNK_TYPE_TEXT) {
            size_t n;
            const llama_token * t = mtmd_input_chunk_get_tokens_text(ch, &n);
            tok.insert(tok.end(), t, t + n);
        } else {
            const mtmd_image_tokens * im = mtmd_input_chunk_get_tokens_image(ch);
            const size_t n = mtmd_image_tokens_get_n_tokens(im);
            const mtmd_decoder_pos last = mtmd_image_tokens_get_decoder_pos(im, 0, n - 1);
            imgs.push_back({(int) tok.size(), (int) last.x + 1, (int) last.y + 1, {}});
            tok.insert(tok.end(), n, pad[0]);
        }
    }
    const int N = (int) tok.size();
    const int first = imgs.back().pos + imgs.back().nx * imgs.back().ny; // <|vision_end|> of the last image
    fprintf(stderr, "%d tokens, %zu images, scoring positions %d .. %d\n", N, imgs.size(), first, N - 2);
    ls::Session ses("", ref_out, {{tok, first}}, nv);

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = (uint32_t) N + 16;
    const int nb = 512;
    cp.n_batch = cp.n_ubatch = nb;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    cp.type_k = cp.type_v = kv_q8 ? GGML_TYPE_Q8_0 : GGML_TYPE_F16;
    llama_context * ctx = llama_init_from_model(lm, cp);
    if (!ctx) { fprintf(stderr, "failed to create the context\n"); return 1; }

    llama_batch batch = llama_batch_init(nb, 0, 1);
    llama_pos n_past = 0; // decoder (M-RoPE) position: an image advances it by max(nx, ny), not by its rows
    int i = 0, k_img = 0;     // sequence index
    for (size_t c = 0; c < mtmd_input_chunks_size(chunks); c++) {
        const mtmd_input_chunk * ch = mtmd_input_chunks_get(chunks, c);
        if (mtmd_input_chunk_get_type(ch) == MTMD_INPUT_CHUNK_TYPE_TEXT) {
            size_t m;
            const llama_token * t = mtmd_input_chunk_get_tokens_text(ch, &m);
            for (size_t o = 0; o < m;) {
                const int n = (int) std::min<size_t>(std::min(nb, tb), m - o), i0 = i + (int) o;
                const int lo = std::max(i0, first), hi = std::min(i0 + n, N - 1);
                batch.n_tokens = n;
                for (int k = 0; k < n; k++) {
                    batch.token[k] = t[o + k];
                    batch.pos[k] = n_past + (llama_pos) o + k;
                    batch.n_seq_id[k] = 1;
                    batch.seq_id[k][0] = 0;
                    batch.logits[k] = i0 + k >= lo && i0 + k < hi;
                }
                if (llama_decode(ctx, batch) != 0) { fprintf(stderr, "llama_decode failed\n"); return 1; }
                if (lo < hi) ses.rows(0, lo, hi - lo, llama_get_logits(ctx));
                o += n;
            }
            i += (int) m;
            n_past += (llama_pos) m;
        } else {
            Img & im = imgs[k_img++];
            const size_t n = (size_t) im.nx * im.ny, ne = (size_t) llama_model_n_embd_inp(lm);
            float * e;
            if (!rows_in.empty()) { // given rows (e.g. an fp64 forward's) instead of libmtmd's encoder
                if (k_img > (int) rows_in.size() || rows_in[k_img - 1].size() != n * ne) { fprintf(stderr, "--rows-in: image %d does not match\n", k_img - 1); return 1; }
                e = rows_in[k_img - 1].data();
            } else {
                if (mtmd_encode_chunk(mt, ch) != 0) { fprintf(stderr, "encode failed\n"); return 1; }
                e = mtmd_get_output_embd(mt);
            }
            im.rows.assign(e, e + n * ne);
            llama_pos np = 0;
            if (mtmd_helper_decode_image_chunk(mt, ctx, ch, e, n_past, 0, nb, &np, nullptr, nullptr) != 0) {
                fprintf(stderr, "image decode failed\n");
                return 1;
            }
            fprintf(stderr, "image %d: %d x %d at sequence position %d, decoder position %d -> %d\n", k_img - 1, im.nx, im.ny, i,
                    (int) n_past, (int) np);
            n_past = np;
            i += (int) n;
        }
    }
    ses.report([&](int id) {
        char buf[256];
        const int n = llama_token_to_piece(vocab, id, buf, sizeof(buf), 0, false);
        return std::string(buf, n > 0 ? n : 0);
    });

    FILE * f = fopen(img_out.c_str(), "wb");
    const char magic[8] = "MKIMG01";
    const int hdr[2] = {llama_model_n_embd_inp(lm), (int) imgs.size()};
    bool ok = f && fwrite(magic, 1, 8, f) == 8 && fwrite(hdr, sizeof(hdr), 1, f) == 1;
    for (const Img & im : imgs) {
        const int v[4] = {0, im.pos, im.nx, im.ny};
        ok = ok && fwrite(v, sizeof(v), 1, f) == 1 && fwrite(im.rows.data(), sizeof(float), im.rows.size(), f) == im.rows.size();
    }
    if (f) fclose(f);
    if (!ok) { fprintf(stderr, "cannot write %s\n", img_out.c_str()); return 1; }

    llama_batch_free(batch);
    for (mtmd_bitmap * b : bms) mtmd_bitmap_free(b);
    mtmd_input_chunks_free(chunks);
    mtmd_free(mt);
    llama_free(ctx);
    llama_model_free(lm);
    return 0;
}
