// CPU vision encoder (see mk_vision.h). Two backends:
// - cpu (vit_cpu.cpp, x86-64 with AVX2 + FMA): llama.cpp's preprocessing bit for bit, then the tower in fp32 with
//   hand-written kernels; faster than libmtmd's and ~1000x closer to an fp64 forward of the checkpoint
// - mtmd: libmtmd's own encoder (any CPU llama.cpp builds for)
#include "mk_vision.h"

#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
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
#ifdef _WIN32
#include "win32.h"
#endif

// image decoding for both backends: llama.cpp's vendored stb_image (llama-server's decode: stbi_load_from_memory, 3
// channels) with PNG, JPEG, GIF and BMP only; its TGA / HDR / PIC loaders have unfixed bugs (GHSL-2023-147 / -149)
#define STB_IMAGE_STATIC
#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_PNG
#define STBI_ONLY_JPEG
#define STBI_ONLY_GIF
#define STBI_ONLY_BMP
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

// MK_IMAGE_MAX_PIXELS: the largest image (width x height) mkv_image_load decodes. Larger ones are refused from the
// file's header, before any pixel is decoded: a PNG of a few hundred KB can declare 20000 x 20000 pixels, whose
// decode allocates ~2.4 GB. 100 megapixels covers phone cameras; the encoder resizes to at most 4096 tokens (~4 MP).
long long max_pixels_env() {
    const char * s = getenv("MK_IMAGE_MAX_PIXELS");
    if (!s || !*s) return 100000000;
    char * end = nullptr;
    const long long v = strtoll(s, &end, 10);
    if (*end || v <= 0) throw std::runtime_error(std::string("MK_IMAGE_MAX_PIXELS: expected a positive number, not ") + s);
    return v;
}

// the encoder's default thread count: every logical CPU but two (8 or more; else all but one). The cpu backend's workers
// are pinned and meet at a barrier after every step, so a single other busy thread on one of their CPUs (e.g. the
// server's keep-warm load) about doubles the encode time, while an idle machine encodes as fast with two CPUs fewer.
// The rows do not depend on the thread count.
int default_threads() {
    const int hw = (int) std::max(1u, std::thread::hardware_concurrency());
    return hw >= 8 ? hw - 2 : std::max(1, hw - 1);
}

// the image tokens of libmtmd's smart resize of a w x h image (mtmd-image.cpp calc_size_preserved_ratio, float arithmetic
// as there; Qwen-VL: 32 x 32 pixels per token). Past an aspect ratio of max_tokens its clamp of the short side to 32
// gives more than max_tokens (1,000,000 x 1 -> 64,000 tokens of 4096, hours of encoding), so the mtmd backend refuses
// those images and frames; the cpu backend cuts them to fit (Vit::target_size).
long long mtmd_resize_tokens(int width, int height, int min_tokens, int max_tokens) {
    const int f = 32;
    const long long max_pixels = (long long) max_tokens * f * f, min_pixels = (long long) min_tokens * f * f;
    auto round_by = [f](float x) { return static_cast<int>(std::round(x / static_cast<float>(f))) * f; };
    auto ceil_by = [f](float x) { return static_cast<int>(std::ceil(x / static_cast<float>(f))) * f; };
    auto floor_by = [f](float x) { return static_cast<int>(std::floor(x / static_cast<float>(f))) * f; };
    int h_bar = std::max(f, round_by((float) height));
    int w_bar = std::max(f, round_by((float) width));
    if ((long long) h_bar * w_bar > max_pixels) {
        const auto beta = std::sqrt(static_cast<float>((long long) height * width) / max_pixels);
        h_bar = std::max(f, floor_by(height / beta));
        w_bar = std::max(f, floor_by(width / beta));
    } else if ((long long) h_bar * w_bar < min_pixels) {
        const auto beta = std::sqrt(static_cast<float>(min_pixels) / ((long long) height * width));
        h_bar = ceil_by(height * beta);
        w_bar = ceil_by(width * beta);
    }
    return (long long) (h_bar / f) * (w_bar / f);
}

// the mtmd backend's check of an image or frame size (an empty string: fine)
std::string mtmd_size_error(const char * what, int w, int h, int min_tokens, int max_tokens) {
    const long long n = mtmd_resize_tokens(w, h, min_tokens, max_tokens);
    if (n <= max_tokens) return "";
    return std::string(what) + " of " + std::to_string(w) + " x " + std::to_string(h) + " pixels: its aspect ratio gives " +
           std::to_string(n) + " image tokens with the mtmd vision backend, over the limit of " + std::to_string(max_tokens) +
           " (the cpu backend cuts such images to fit)";
}

bool cpu_has_avx2_fma() {
#if defined(_WIN32) && defined(_M_X64)
    return mkv_win::avx2_fma();
#elif defined(__x86_64__) || defined(_M_X64)
    return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
#else
    return false;
#endif
}

// only the Qwen3.x projector reaches libmtmd (others trust file sizes: llama.cpp #27166), read as clip.cpp reads it;
// returns libmtmd's row width for it (clip_n_mmproj_embd: mm.2.bias times 1 + deepstack layers, which are refused)
int check_mtmd_projector(const char * mmproj) {
    ggml_context * meta = nullptr;
    gguf_init_params gp = {true, &meta};
    std::unique_ptr<gguf_context, void (*)(gguf_context *)> g(gguf_init_from_file(mmproj, gp), gguf_free);
    std::unique_ptr<ggml_context, void (*)(ggml_context *)> m(meta, ggml_free);
    if (!g || !m) throw std::runtime_error(std::string("cannot read the mmproj's metadata: ") + mmproj);
    auto find = [&](const char * k, gguf_type t) {
        const int64_t i = gguf_find_key(g.get(), k);
        if (i >= 0 && gguf_get_kv_type(g.get(), i) != t) throw std::runtime_error(std::string("mmproj: ") + k + " has an unexpected type");
        return i;
    };
    const int64_t hv = find("clip.has_vision_encoder", GGUF_TYPE_BOOL), ha = find("clip.has_audio_encoder", GGUF_TYPE_BOOL);
    if (hv < 0 || !gguf_get_val_bool(g.get(), hv)) throw std::runtime_error("the mmproj has no vision encoder");
    if (ha >= 0 && gguf_get_val_bool(g.get(), ha)) throw std::runtime_error("the mmproj has an audio encoder (only a vision-only qwen3vl_merger mmproj is supported)");
    std::string type;
    for (const char * k : {"clip.projector_type", "clip.vision.projector_type"}) {
        const int64_t i = find(k, GGUF_TYPE_STRING);
        if (i >= 0) type = gguf_get_val_str(g.get(), i);
        if (!type.empty()) break;
    }
    if (type != "qwen3vl_merger") throw std::runtime_error("mmproj projector \"" + type + "\": only qwen3vl_merger (Qwen3.x vision) is supported");
    for (ggml_tensor * t = ggml_get_first_tensor(m.get()); t; t = ggml_get_next_tensor(m.get(), t))
        if (strncmp(ggml_get_name(t), "v.deepstack.", 12) == 0) throw std::runtime_error("the mmproj has deepstack layers (not supported)");
    const ggml_tensor * b = ggml_get_tensor(m.get(), "mm.2.bias");
    const int64_t pd = find("clip.vision.projection_dim", GGUF_TYPE_UINT32);
    if (!b || ggml_n_dims(b) != 1 || b->ne[0] < 1 || b->ne[0] > (1 << 16) || pd < 0 || b->ne[0] != (int64_t) gguf_get_val_u32(g.get(), pd))
        throw std::runtime_error("mmproj: mm.2.bias does not match clip.vision.projection_dim");
    return (int) b->ne[0];
}
} // namespace

struct mkv_ctx {
    int backend = 0; // MKV_BACKEND_CPU or MKV_BACKEND_MTMD
    long long max_pixels = 0; // MK_IMAGE_MAX_PIXELS
    int min_tokens = 8, max_tokens = 4096; // mtmd: the resize's image-token limits (mtmd_size_error)
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
    std::vector<uint8_t> f0, f1;        // cpu, video: the resized frame(s), u8 HWC (normalized at encode: 1/4 the memory)
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
        c->max_pixels = max_pixels_env();
        g_verbose = params.verbose != 0;
        const int threads = params.n_threads > 0 ? params.n_threads : default_threads();
        std::string why_not_cpu;
#ifndef MKV_NO_VIT
        if (params.backend != MKV_BACKEND_MTMD) {
            if (!cpu_has_avx2_fma()) {
                why_not_cpu = "the CPU has no AVX2 + FMA";
            } else {
                try {
                    mkv::Vit::Params vp;
                    vp.n_threads = threads; // (logical CPUs: SMT siblings hide the kernels' latencies)
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
        c->n_embd = check_mtmd_projector(mmproj);
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
        if (params.min_tokens > 0) c->min_tokens = params.min_tokens; // (else the projector's 8 / 4096, as above)
        if (params.max_tokens > 0) c->max_tokens = params.max_tokens;
        c->mt = mtmd_init_from_file(mmproj, c->vocab, cp);
        if (!c->mt) throw std::runtime_error(std::string("cannot load the mmproj: ") + mmproj);
        if (!mtmd_support_vision(c->mt)) throw std::runtime_error("the mmproj has no vision encoder");
        if (!mtmd_decode_use_mrope(c->mt)) throw std::runtime_error("the text model does not use M-RoPE positions (not a Qwen3.x model)");
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
    // the header first (both backends decode with stb_image): an image over MK_IMAGE_MAX_PIXELS is refused undecoded
    int w = 0, h = 0, comp = 0;
    if (!data || len == 0 || len > (size_t) INT_MAX || !stbi_info_from_memory(data, (int) len, &w, &h, &comp)) {
        g_err = len > (size_t) INT_MAX ? "cannot decode the image (over 2 GB)" : "cannot decode the image (PNG, JPEG, GIF or BMP)";
        return nullptr;
    }
    if (w <= 0 || h <= 0 || (long long) w * h > c->max_pixels) {
        g_err = "image of " + std::to_string(w) + " x " + std::to_string(h) + " pixels: over the limit of " +
                std::to_string(c->max_pixels) + " pixels (MK_IMAGE_MAX_PIXELS)";
        return nullptr;
    }
#ifndef MKV_NO_VIT
    if (c->backend == MKV_BACKEND_CPU) {
        int nx = 0, ny = 0, nc = 0;
        unsigned char * rgb = stbi_load_from_memory(data, (int) len, &nx, &ny, &nc, 3);
        if (!rgb || nx <= 0 || ny <= 0 || (long long) nx * ny > c->max_pixels) {
            if (rgb) stbi_image_free(rgb);
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
    g_err = mtmd_size_error("image", w, h, c->min_tokens, c->max_tokens);
    if (!g_err.empty()) return nullptr;
    // (not mtmd_helper_bitmap_init_from_buf: its stb has every loader, and a failed decode goes to an ffmpeg on PATH)
    int nx = 0, ny = 0, nc = 0;
    unsigned char * rgb = stbi_load_from_memory(data, (int) len, &nx, &ny, &nc, 3);
    if (!rgb || nx <= 0 || ny <= 0 || (long long) nx * ny > c->max_pixels) {
        if (rgb) stbi_image_free(rgb);
        g_err = "cannot decode the image";
        return nullptr;
    }
    mtmd_bitmap * bm = mtmd_bitmap_init((uint32_t) nx, (uint32_t) ny, rgb);
    stbi_image_free(rgb);
    mtmd_input_chunks * chunks = mtmd_input_chunks_init();
    const char * marker = mtmd_default_marker();
    mtmd_input_text txt = {marker, strlen(marker), false, true};
    const mtmd_bitmap * bms[1] = {bm};
    const int32_t r = mtmd_tokenize(c->mt, chunks, &txt, bms, 1);
    mtmd_bitmap_free(bm);
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

// mtmd backend: libmtmd's own merge of consecutive same-size bitmaps (mtmd_tokenize, [QWEN_VIDEO]) into one image chunk
static mkv_image * frames_mtmd(mkv_ctx * c, const unsigned char * rgb0, const unsigned char * rgb1, int nx, int ny) {
    mtmd_bitmap * bm[2] = {mtmd_bitmap_init((uint32_t) nx, (uint32_t) ny, rgb0), rgb1 ? mtmd_bitmap_init((uint32_t) nx, (uint32_t) ny, rgb1) : nullptr};
    const int nb = rgb1 ? 2 : 1;
    std::string text;
    for (int i = 0; i < nb; i++) text += mtmd_default_marker();
    mtmd_input_chunks * chunks = mtmd_input_chunks_init();
    mtmd_input_text txt = {text.c_str(), text.size(), false, true};
    const mtmd_bitmap * bms[2] = {bm[0], bm[1]};
    const int32_t r = mtmd_tokenize(c->mt, chunks, &txt, bms, (size_t) nb);
    for (mtmd_bitmap * b : bm)
        if (b) mtmd_bitmap_free(b);
    mkv_image * im = nullptr;
    size_t n_img = 0;
    if (r != 0) g_err = "frame preprocessing failed (mtmd_tokenize " + std::to_string(r) + ")";
    for (size_t i = 0; r == 0 && i < mtmd_input_chunks_size(chunks); i++) {
        const mtmd_input_chunk * ch = mtmd_input_chunks_get(chunks, i);
        if (mtmd_input_chunk_get_type(ch) != MTMD_INPUT_CHUNK_TYPE_IMAGE) continue;
        if (n_img++) break; // (two chunks: libmtmd did not merge the frames)
        const mtmd_image_tokens * it = mtmd_input_chunk_get_tokens_image(ch);
        const size_t n = mtmd_image_tokens_get_n_tokens(it);
        const mtmd_decoder_pos last = mtmd_image_tokens_get_decoder_pos(it, 0, n - 1);
        im = new mkv_image;
        im->chunk = mtmd_input_chunk_copy(ch);
        im->nx = (int) last.x + 1;
        im->ny = (int) last.y + 1;
        im->n = (int) n;
    }
    mtmd_input_chunks_free(chunks);
    if (im && (n_img != 1 || (size_t) im->nx * im->ny != (size_t) im->n)) {
        g_err = n_img != 1 ? "libmtmd did not merge the frame pair" : "unexpected frame token layout";
        mkv_image_free(im);
        im = nullptr;
    }
    if (!im && g_err.empty()) g_err = "no image chunk";
    return im;
}

mkv_image * mkv_frames_load(mkv_ctx * c, const unsigned char * rgb0, const unsigned char * rgb1, int nx, int ny, int max_tokens) {
    g_err.clear();
    if (!rgb0 || nx <= 0 || ny <= 0 || (long long) nx * ny > c->max_pixels) {
        g_err = !rgb0 || nx <= 0 || ny <= 0 ? "no frame"
                                           : "frame of " + std::to_string(nx) + " x " + std::to_string(ny) + " pixels: over the limit of " +
                                                 std::to_string(c->max_pixels) + " pixels (MK_IMAGE_MAX_PIXELS)";
        return nullptr;
    }
#ifndef MKV_NO_VIT
    if (c->backend == MKV_BACKEND_CPU) {
        auto * im = new mkv_image;
        try {
            int tw1 = 0, th1 = 0;
            c->vit->resize(rgb0, nx, ny, im->f0, im->tw, im->th, max_tokens);
            if (rgb1) c->vit->resize(rgb1, nx, ny, im->f1, tw1, th1, max_tokens);
        } catch (const std::exception & e) {
            g_err = e.what();
            delete im;
            return nullptr;
        }
        im->nx = im->tw / 32;
        im->ny = im->th / 32;
        im->n = im->nx * im->ny;
        return im;
    }
#endif
    if (max_tokens > 0) {
        g_err = "the mtmd backend resizes video frames with the image token limit only (no per-frame max_tokens)";
        return nullptr;
    }
    g_err = mtmd_size_error("frame", nx, ny, c->min_tokens, c->max_tokens);
    if (!g_err.empty()) return nullptr;
    try {
        return frames_mtmd(c, rgb0, rgb1, nx, ny);
    } catch (const std::exception & e) {
        g_err = e.what();
        return nullptr;
    }
}

int mkv_image_grid(const mkv_image * im, int * nx, int * ny) {
    if (nx) *nx = im ? im->nx : 0;
    if (ny) *ny = im ? im->ny : 0;
    return im ? im->n : 0;
}

int mkv_image_encode(mkv_ctx * c, mkv_image * im) {
    g_err.clear();
    if (!im) {
        g_err = "no image";
        return -1;
    }
    if (!im->rows.empty()) return 0;
    const double t0 = now_ms();
#ifndef MKV_NO_VIT
    if (c->backend == MKV_BACKEND_CPU) {
        try {
            im->rows.resize((size_t) im->n * c->n_embd);
            if (!im->f0.empty()) { // video: the frame(s) normalized now (as preprocess does)
                std::vector<float> p0(im->f0.size()), p1(im->f1.size());
                c->vit->normalize(im->f0.data(), im->f0.size(), p0.data());
                if (!im->f1.empty()) {
                    c->vit->normalize(im->f1.data(), im->f1.size(), p1.data());
                    c->vit->encode_pair(p0.data(), p1.data(), im->tw, im->th, im->rows.data());
                } else {
                    c->vit->encode(p0.data(), im->tw, im->th, im->rows.data());
                }
            } else {
                c->vit->encode(im->pix.data(), im->tw, im->th, im->rows.data());
            }
        } catch (const std::exception & e) {
            im->rows.clear();
            g_err = e.what();
            return -1;
        }
        std::vector<float>().swap(im->pix); // (not needed again)
        std::vector<uint8_t>().swap(im->f0);
        std::vector<uint8_t>().swap(im->f1);
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

const float * mkv_image_rows(const mkv_image * im) { return !im || im->rows.empty() ? nullptr : im->rows.data(); }
double mkv_image_encode_ms(const mkv_image * im) { return im ? im->encode_ms : 0; }

void mkv_image_free(mkv_image * im) {
    if (!im) return;
    if (im->chunk) mtmd_input_chunk_free(im->chunk);
    delete im;
}

} // extern "C"
