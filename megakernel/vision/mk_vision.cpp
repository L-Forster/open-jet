// CPU vision encoder (see mk_vision.h). Two backends:
// - cpu (vit_cpu.cpp, x86-64 with AVX2 + FMA): llama.cpp's preprocessing bit for bit, then the tower in fp32 with
//   hand-written kernels; faster than libmtmd's and ~1000x closer to an fp64 forward of the checkpoint
// - mtmd: libmtmd's own encoder (any CPU llama.cpp builds for)
#include "mk_vision.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "gguf.h"
#include "llama.h"
#include "mtmd-helper.h"
#include "mtmd.h"

#ifndef MKV_NO_VIT
#include "vit_cpu.h"
#endif

// image decoding: the stb_image that llama.cpp vendors and libmtmd decodes with (mtmd_helper_bitmap_init_from_buf:
// stbi_load_from_memory, 3 channels), compiled here as private static functions
#define STB_IMAGE_STATIC
#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"

namespace {
thread_local std::string g_err;

bool g_verbose = false;
void log_filtered(ggml_log_level level, const char * text, void *) {
    if (g_verbose || level == GGML_LOG_LEVEL_ERROR || level == GGML_LOG_LEVEL_WARN) fputs(text, stderr);
}

double now_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

bool cpu_has_avx2_fma() {
#if defined(__x86_64__) || defined(_M_X64)
    return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
#else
    return false;
#endif
}
} // namespace

struct mkv_ctx {
    int backend = 0; // MKV_BACKEND_CPU or MKV_BACKEND_MTMD
    int n_embd = 0;
    int n_threads = 0;
    // mtmd
    llama_model * vocab = nullptr; // the text model, vocabulary only (mtmd reads its special tokens and RoPE type)
    mtmd_context * mt = nullptr;
#ifndef MKV_NO_VIT
    std::unique_ptr<mkv::Vit> vit;
#endif
};

struct mkv_image {
    int nx = 0, ny = 0, n = 0;
    mtmd_input_chunk * chunk = nullptr; // mtmd: the image chunk of mtmd_tokenize (owned copy)
    std::vector<float> pix;             // cpu: the preprocessed image (HWC)
    int tw = 0, th = 0;
    std::vector<float> rows;
    double encode_ms = 0;
};

extern "C" {

struct mkv_params mkv_params_default(void) {
    mkv_params p;
    p.n_threads = 0;
    p.min_tokens = 0;
    p.max_tokens = 0;
    p.flash_attn = -1;
    p.verbose = 0;
    p.backend = MKV_BACKEND_AUTO;
    return p;
}

const char * mkv_last_error(void) { return g_err.c_str(); }

mkv_ctx * mkv_new(const char * mmproj, const char * text_model, struct mkv_params params) {
    g_err.clear();
    auto * c = new mkv_ctx;
    try {
        g_verbose = params.verbose != 0;
        const int threads = params.n_threads > 0 ? params.n_threads : (int) std::max(1u, std::thread::hardware_concurrency());
        std::string why_not_cpu;
#ifndef MKV_NO_VIT
        if (params.backend != MKV_BACKEND_MTMD) {
            if (!cpu_has_avx2_fma()) {
                why_not_cpu = "the CPU has no AVX2 + FMA";
            } else {
                try {
                    mkv::Vit::Params vp;
                    vp.n_threads = threads; // (one per logical CPU: SMT siblings hide the kernels' latencies)
                    vp.min_tokens = params.min_tokens;
                    vp.max_tokens = params.max_tokens;
                    c->vit.reset(new mkv::Vit(mmproj, vp));
                    c->backend = MKV_BACKEND_CPU;
                    c->n_embd = c->vit->n_embd();
                    c->n_threads = c->vit->n_threads();
                    return c;
                } catch (const std::exception & e) {
                    why_not_cpu = e.what();
                }
            }
            if (params.backend == MKV_BACKEND_CPU) throw std::runtime_error("cpu backend: " + why_not_cpu);
            fprintf(stderr, "mk_vision: cpu backend unavailable (%s), using libmtmd\n", why_not_cpu.c_str());
        }
#else
        if (params.backend == MKV_BACKEND_CPU) throw std::runtime_error("built without the cpu backend (MKV_NO_VIT)");
#endif
        c->backend = MKV_BACKEND_MTMD;
        mtmd_helper_log_set(log_filtered, nullptr);
        llama_log_set(log_filtered, nullptr);
        llama_backend_init();
        llama_model_params mp = llama_model_default_params();
        mp.vocab_only = true;
        c->vocab = llama_model_load_from_file(text_model, mp);
        llama_log_set(nullptr, nullptr); // (back to libllama's default for the engine's own logging)
        if (!c->vocab) throw std::runtime_error(std::string("cannot load the text model's vocabulary: ") + text_model);
        mtmd_context_params cp = mtmd_context_params_default();
        cp.use_gpu = false;
        cp.print_timings = params.verbose != 0;
        c->n_threads = threads;
        cp.n_threads = threads;
        cp.flash_attn_type = params.flash_attn < 0 ? LLAMA_FLASH_ATTN_TYPE_AUTO
                             : params.flash_attn  ? LLAMA_FLASH_ATTN_TYPE_ENABLED
                                                  : LLAMA_FLASH_ATTN_TYPE_DISABLED;
        cp.warmup = false; // (the first encode pays the graph allocation, ~10 ms)
        cp.image_min_tokens = params.min_tokens > 0 ? params.min_tokens : -1;
        cp.image_max_tokens = params.max_tokens > 0 ? params.max_tokens : -1;
        c->mt = mtmd_init_from_file(mmproj, c->vocab, cp);
        if (!c->mt) throw std::runtime_error(std::string("cannot load the mmproj: ") + mmproj);
        if (!mtmd_support_vision(c->mt)) throw std::runtime_error("the mmproj has no vision encoder");
        if (!mtmd_decode_use_mrope(c->mt)) throw std::runtime_error("the mmproj does not use M-RoPE positions (not a Qwen-VL projector)");
        // the row width: the mmproj's projection size (mtmd sizes its output by it; a vocabulary-only text model carries
        // no hyperparameters to check it against)
        gguf_init_params gp = {true, nullptr};
        gguf_context * g = gguf_init_from_file(mmproj, gp);
        if (!g) throw std::runtime_error(std::string("cannot read the mmproj's metadata: ") + mmproj);
        const int64_t k = gguf_find_key(g, "clip.vision.projection_dim");
        c->n_embd = k >= 0 ? (int) gguf_get_val_u32(g, k) : 0;
        gguf_free(g);
        if (c->n_embd <= 0) throw std::runtime_error("the mmproj has no clip.vision.projection_dim");
        return c;
    } catch (const std::exception & e) {
        g_err = e.what();
        mkv_free(c);
        return nullptr;
    }
}

void mkv_free(mkv_ctx * c) {
    if (!c) return;
    if (c->mt) mtmd_free(c->mt);
    if (c->vocab) llama_model_free(c->vocab);
    delete c;
}

int mkv_n_embd(const mkv_ctx * c) { return c->n_embd; }
int mkv_n_threads(const mkv_ctx * c) { return c->n_threads; }
int mkv_backend(const mkv_ctx * c) { return c->backend; }

mkv_image * mkv_image_load(mkv_ctx * c, const unsigned char * data, size_t len) {
    g_err.clear();
#ifndef MKV_NO_VIT
    if (c->backend == MKV_BACKEND_CPU) {
        int nx, ny, nc;
        unsigned char * rgb = stbi_load_from_memory(data, (int) len, &nx, &ny, &nc, 3);
        if (!rgb) {
            g_err = "cannot decode the image";
            return nullptr;
        }
        auto * im = new mkv_image;
        try {
            c->vit->preprocess(rgb, nx, ny, im->pix, im->tw, im->th);
        } catch (const std::exception & e) {
            g_err = e.what();
            stbi_image_free(rgb);
            delete im;
            return nullptr;
        }
        stbi_image_free(rgb);
        im->nx = im->tw / 32;
        im->ny = im->th / 32;
        im->n = im->nx * im->ny;
        return im;
    }
#endif
    mtmd_helper_bitmap_wrapper bw = mtmd_helper_bitmap_init_from_buf(c->mt, data, len, false);
    if (!bw.bitmap) {
        g_err = "cannot decode the image";
        return nullptr;
    }
    mtmd_input_chunks * chunks = mtmd_input_chunks_init();
    const char * marker = mtmd_default_marker();
    mtmd_input_text txt = {marker, strlen(marker), false, true};
    const mtmd_bitmap * bms[1] = {bw.bitmap};
    const int32_t r = mtmd_tokenize(c->mt, chunks, &txt, bms, 1);
    mtmd_bitmap_free(bw.bitmap);
    mkv_image * im = nullptr;
    if (r != 0) {
        g_err = "image preprocessing failed (mtmd_tokenize " + std::to_string(r) + ")";
    } else {
        for (size_t i = 0; i < mtmd_input_chunks_size(chunks); i++) {
            const mtmd_input_chunk * ch = mtmd_input_chunks_get(chunks, i);
            if (mtmd_input_chunk_get_type(ch) != MTMD_INPUT_CHUNK_TYPE_IMAGE) continue;
            const mtmd_image_tokens * it = mtmd_input_chunk_get_tokens_image(ch);
            const size_t n = mtmd_image_tokens_get_n_tokens(it);
            const mtmd_decoder_pos last = mtmd_image_tokens_get_decoder_pos(it, 0, n - 1);
            im = new mkv_image;
            im->chunk = mtmd_input_chunk_copy(ch);
            im->nx = (int) last.x + 1; // (pos_0 = 0: the last raster row's x / y are nx - 1 / ny - 1)
            im->ny = (int) last.y + 1;
            im->n = (int) n;
            if ((size_t) im->nx * im->ny != n) {
                g_err = "unexpected image token layout";
                mkv_image_free(im);
                im = nullptr;
            }
            break;
        }
        if (!im && g_err.empty()) g_err = "no image chunk";
    }
    mtmd_input_chunks_free(chunks);
    return im;
}

int mkv_image_grid(const mkv_image * im, int * nx, int * ny) {
    if (nx) *nx = im->nx;
    if (ny) *ny = im->ny;
    return im->n;
}

int mkv_image_encode(mkv_ctx * c, mkv_image * im) {
    g_err.clear();
    if (!im->rows.empty()) return 0;
    const double t0 = now_ms();
#ifndef MKV_NO_VIT
    if (c->backend == MKV_BACKEND_CPU) {
        try {
            im->rows.resize((size_t) im->n * c->n_embd);
            c->vit->encode(im->pix.data(), im->tw, im->th, im->rows.data());
        } catch (const std::exception & e) {
            im->rows.clear();
            g_err = e.what();
            return -1;
        }
        std::vector<float>().swap(im->pix); // (not needed again)
        im->encode_ms = now_ms() - t0;
        return 0;
    }
#endif
    if (mtmd_encode_chunk(c->mt, im->chunk) != 0) {
        g_err = "image encode failed";
        return -1;
    }
    const float * e = mtmd_get_output_embd(c->mt); // n rows of the projection size
    im->rows.assign(e, e + (size_t) im->n * c->n_embd);
    im->encode_ms = now_ms() - t0;
    return 0;
}

const float * mkv_image_rows(const mkv_image * im) { return im->rows.empty() ? nullptr : im->rows.data(); }
double mkv_image_encode_ms(const mkv_image * im) { return im->encode_ms; }

void mkv_image_free(mkv_image * im) {
    if (!im) return;
    if (im->chunk) mtmd_input_chunk_free(im->chunk);
    delete im;
}

} // extern "C"
