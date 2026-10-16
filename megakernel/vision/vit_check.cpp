// Test tool (not part of the library): the CPU encoder (vit_cpu) against libmtmd on the same images.
//   vit_check -m model.gguf --mmproj mmproj.gguf [-t threads] [-r reps] [--dump prefix] [--max-tokens n] image...
// Per image: (a) libmtmd's own path (decode, preprocess, encode: what llama-server feeds), (b) libmtmd's encoder on
// vit_cpu's preprocessed pixels (clip_image_encode; a == b bit for bit means the preprocessing is llama.cpp's),
// (c) vit_cpu. Reports the differences and times; --dump writes <prefix>.<i>.{img,mtmd,vit}.f32 (+ .txt dims) for
// the fp64 reference (vit_ref.py). Uses libmtmd's internal clip API (clip.h, clip-impl.h): a test, nothing ships it.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "clip-impl.h"
#include "clip.h"
#include "llama.h"
#include "mtmd-helper.h"
#include "mtmd.h"
#include "mtmd-image.h"
#include "vit_cpu.h"

#include <chrono>

static double now_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

static std::vector<unsigned char> read_file(const char * p) {
    std::vector<unsigned char> b;
    FILE * f = fopen(p, "rb");
    if (!f) return b;
    fseek(f, 0, SEEK_END);
    b.resize((size_t) ftell(f));
    fseek(f, 0, SEEK_SET);
    if (fread(b.data(), 1, b.size(), f) != b.size()) b.clear();
    fclose(f);
    return b;
}

static void dump(const std::string & path, const float * p, size_t n) {
    FILE * f = fopen(path.c_str(), "wb");
    if (!f || fwrite(p, sizeof(float), n, f) != n) {
        fprintf(stderr, "cannot write %s\n", path.c_str());
        exit(1);
    }
    fclose(f);
}

struct Diff {
    double rel = 0, maxabs = 0, mincos = 1;
};
static Diff compare(const float * a, const float * b, int rows, int ne) {
    Diff d;
    double num = 0, den = 0;
    for (int r = 0; r < rows; r++) {
        double dot = 0, na = 0, nb = 0;
        for (int e = 0; e < ne; e++) {
            const double x = a[(size_t) r * ne + e], y = b[(size_t) r * ne + e];
            num += (x - y) * (x - y);
            den += x * x;
            dot += x * y, na += x * x, nb += y * y;
            d.maxabs = std::max(d.maxabs, std::fabs(x - y));
        }
        d.mincos = std::min(d.mincos, dot / std::sqrt(na * nb + 1e-300));
    }
    d.rel = std::sqrt(num / (den + 1e-300));
    return d;
}

int main(int argc, char ** argv) {
    const char * model = nullptr, * mmproj = nullptr, * prefix = nullptr;
    int threads = 0, reps = 1, max_tok = 0;
    std::vector<const char *> imgs;
    for (int i = 1; i < argc; i++) {
        const std::string a = argv[i];
        auto next = [&]() { if (i + 1 >= argc) exit(1); return argv[++i]; };
        if (a == "-m") model = next();
        else if (a == "--mmproj") mmproj = next();
        else if (a == "-t") threads = atoi(next());
        else if (a == "-r") reps = atoi(next());
        else if (a == "--dump") prefix = next();
        else if (a == "--max-tokens") max_tok = atoi(next());
        else imgs.push_back(argv[i]);
    }
    if (!model || !mmproj || imgs.empty()) {
        fprintf(stderr, "usage: vit_check -m model --mmproj mmproj [-t threads] [-r reps] [--dump prefix] [--max-tokens n] image...\n");
        return 1;
    }
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.vocab_only = true;
    llama_model * lm = llama_model_load_from_file(model, mp);
    mtmd_context_params cp = mtmd_context_params_default();
    cp.use_gpu = false;
    cp.print_timings = false;
    cp.warmup = false;
    cp.n_threads = threads > 0 ? threads : 32;
    cp.image_max_tokens = max_tok > 0 ? max_tok : -1;
    mtmd_context * mt = mtmd_init_from_file(mmproj, lm, cp);
    clip_context_params ccp{};
    ccp.use_gpu = false;
    ccp.flash_attn_type = CLIP_FLASH_ATTN_TYPE_AUTO;
    ccp.image_min_tokens = -1;
    ccp.image_max_tokens = max_tok > 0 ? max_tok : -1;
    ccp.warmup = false;
    clip_ctx * cc = clip_init(mmproj, ccp).ctx_v;
    if (!lm || !mt || !cc) {
        fprintf(stderr, "load failed\n");
        return 2;
    }
    mkv::Vit::Params vp;
    vp.n_threads = threads;
    vp.max_tokens = max_tok;
    const double tl = now_ms();
    mkv::Vit vit(mmproj, vp);
    printf("vit_cpu: %d threads, loaded in %.0f ms\n", vit.n_threads(), now_ms() - tl);
    const int ne = vit.n_embd();
    for (size_t k = 0; k < imgs.size(); k++) {
        const std::vector<unsigned char> b = read_file(imgs[k]);
        mtmd_helper_bitmap_wrapper bw = mtmd_helper_bitmap_init_from_buf(mt, b.data(), b.size(), false);
        if (!bw.bitmap) {
            fprintf(stderr, "%s: cannot decode\n", imgs[k]);
            return 3;
        }
        const int nx = (int) mtmd_bitmap_get_nx(bw.bitmap), ny = (int) mtmd_bitmap_get_ny(bw.bitmap);
        // (a) libmtmd's own path
        mtmd_input_chunks * chunks = mtmd_input_chunks_init();
        mtmd_input_text txt = {mtmd_default_marker(), strlen(mtmd_default_marker()), false, true};
        const mtmd_bitmap * bms[1] = {bw.bitmap};
        if (mtmd_tokenize(mt, chunks, &txt, bms, 1) != 0) return 4;
        const mtmd_input_chunk * img = nullptr;
        for (size_t i = 0; i < mtmd_input_chunks_size(chunks); i++)
            if (mtmd_input_chunk_get_type(mtmd_input_chunks_get(chunks, i)) == MTMD_INPUT_CHUNK_TYPE_IMAGE) img = mtmd_input_chunks_get(chunks, i);
        const size_t nt = mtmd_image_tokens_get_n_tokens(mtmd_input_chunk_get_tokens_image(img));
        double ta = 1e30;
        std::vector<float> ra;
        for (int r = 0; r < reps; r++) {
            const double t0 = now_ms();
            if (mtmd_encode_chunk(mt, img) != 0) return 5;
            ta = std::min(ta, now_ms() - t0);
        }
        ra.assign(mtmd_get_output_embd(mt), mtmd_get_output_embd(mt) + nt * ne);
        // vit_cpu's preprocessing
        std::vector<float> pix;
        int tw, th;
        vit.preprocess(mtmd_bitmap_get_data(bw.bitmap), nx, ny, pix, tw, th);
        const int nv = (tw / 32) * (th / 32);
        {   // pixels against libmtmd's preprocessor itself
            mtmd_image_preprocessor_dyn_size pp(cc);
            clip_image_u8 u8;
            u8.set_size({nx, ny}, false);
            const unsigned char * d = mtmd_bitmap_get_data(bw.bitmap);
            u8.cpy_buf(std::vector<uint8_t>(d, d + (size_t) nx * ny * 3));
            const mtmd_image_preproc_out po = pp.preprocess(u8);
            const std::vector<float> & ref = po.entries[0].get_ro_buf();
            size_t bad = 0, first = SIZE_MAX;
            for (size_t i = 0; i < std::min(ref.size(), pix.size()); i++)
                if (ref[i] != pix[i]) bad++, first = std::min(first, i);
            printf("   pixels vs libmtmd's preprocessor (%d x %d): %zu of %zu differ", po.entries[0].nx(), po.entries[0].ny(), bad, ref.size());
            if (bad) printf(" (first at x %zu y %zu c %zu: %.6f vs %.6f)", first / 3 % tw, first / 3 / tw, first % 3, ref[first], pix[first]);
            printf("\n");
        }
        // (b) libmtmd's encoder on those pixels
        clip_image_f32 cim;
        cim.set_size({tw, th}, false, false);
        cim.cpy_buf(pix);
        std::vector<float> rb(nt * ne); // (sized by the caller)
        if (!clip_image_encode(cc, cp.n_threads, &cim, rb)) return 6;
        // (c) vit_cpu
        std::vector<float> rc((size_t) nv * ne);
        double tc = 1e30, ph[5] = {};
        for (int r = 0; r < reps; r++) {
            vit.encode(pix.data(), tw, th, rc.data());
            if (vit.ms[4] < tc) {
                tc = vit.ms[4];
                memcpy(ph, vit.ms, sizeof(ph));
            }
        }
        printf("%s: %d x %d -> %d x %d px, %zu tokens (vit %d)\n", imgs[k], nx, ny, tw, th, nt, nv);
        if ((size_t) nv != nt || rb.size() != nt * ne) {
            printf("   TOKEN COUNT MISMATCH\n");
            return 7;
        }
        const Diff ab = compare(ra.data(), rb.data(), (int) nt, ne), ac = compare(ra.data(), rc.data(), (int) nt, ne);
        printf("   preprocessing: mtmd path vs mtmd encoder on vit_cpu pixels: %s (rel %.3g, max |d| %.3g)\n",
               ab.maxabs == 0 ? "IDENTICAL" : "differ", ab.rel, ab.maxabs);
        printf("   vit_cpu vs mtmd: rel L2 %.4g, max |d| %.4g, min row cosine %.6f\n", ac.rel, ac.maxabs, ac.mincos);
        printf("   time: mtmd %.0f ms (%d threads), vit_cpu %.0f ms (%d threads: patch %.0f, gemm %.0f, attn %.0f, merger %.0f)\n",
               ta, cp.n_threads, tc, vit.n_threads(), ph[0], ph[1], ph[2], ph[3]);
        if (prefix) {
            const std::string p = std::string(prefix) + "." + std::to_string(k);
            dump(p + ".img.f32", pix.data(), pix.size());
            dump(p + ".mtmd.f32", ra.data(), ra.size());
            dump(p + ".vit.f32", rc.data(), rc.size());
            FILE * f = fopen((p + ".txt").c_str(), "w");
            fprintf(f, "%d %d %zu %d\n", tw, th, nt, ne);
            fclose(f);
        }
        fflush(stdout);
        mtmd_input_chunks_free(chunks);
        mtmd_bitmap_free(bw.bitmap);
    }
    clip_free(cc);
    mtmd_free(mt);
    llama_model_free(lm);
    return 0;
}
