// CPU vision encoder for the megakernel server: an image file in memory -> input-embedding rows for MKEngine's
// prefill (MKImage), computed entirely on the CPU (no GPU memory). Projector "qwen3vl_merger" (the Qwen3.5 / 3.6 /
// 3.8 vision towers). Decoding (stb_image) and preprocessing (smart resize, padded bilinear, normalisation) are
// llama-server's, bit for bit, so an image gets the same grid and pixels as with llama-server --mmproj.
// Backends:
//   cpu   vit_cpu.cpp: the tower in fp32 with AVX2 / FMA kernels (x86-64 with AVX2 + FMA: Zen 2+, Haswell+), weights
//         widened from the mmproj's type (BF16 / F16 exactly). On a Ryzen 9 5900XT ~1.5x faster than libmtmd and
//         within ~1e-5 (relative) of an fp64 forward, where libmtmd (BF16 GEMM inputs, fp16 K / V) is ~2e-2.
//   mtmd  libmtmd's encoder (llama.cpp's own, any CPU it builds for).
//   auto  cpu when the CPU and the mmproj allow it, else mtmd.
//
// Lifecycle: mkv_new (load the mmproj) -> per image mkv_image_load (decode + preprocess: the token grid is known here,
// before any encoding) -> mkv_image_encode (the expensive part; once per image) -> mkv_image_rows -> mkv_image_free.
// Images are independent of each other and of the context; one encode runs at a time per context.
#pragma once

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct mkv_ctx mkv_ctx;
typedef struct mkv_image mkv_image;

enum { MKV_BACKEND_AUTO = 0, MKV_BACKEND_MTMD = 1, MKV_BACKEND_CPU = 2 };

struct mkv_params {
    int n_threads;   // encoder threads (<= 0: one per logical CPU, which is fastest on Zen 3 with SMT on)
    int min_tokens;  // image-token limits of the resize (<= 0: the projector's defaults, as llama-server: 8 / 4096)
    int max_tokens;
    int flash_attn;  // mtmd backend: -1 auto (llama-server's default), 0 off, 1 on
    int verbose;     // 1: libmtmd's log on stderr
    int backend;     // MKV_BACKEND_*
};

struct mkv_params mkv_params_default(void);

// mmproj: the projector GGUF; text_model: the text model GGUF (mtmd backend: its vocabulary and architecture)
mkv_ctx * mkv_new(const char * mmproj, const char * text_model, struct mkv_params params);
void mkv_free(mkv_ctx * ctx);
// the last error of this thread (empty: none)
const char * mkv_last_error(void);
// width of an embedding row (the text model's embedding size)
int mkv_n_embd(const mkv_ctx * ctx);
// threads the encoder uses
int mkv_n_threads(const mkv_ctx * ctx);
// the backend in use (MKV_BACKEND_CPU or MKV_BACKEND_MTMD)
int mkv_backend(const mkv_ctx * ctx);

// Decodes (stb_image: PNG, JPEG, BMP, GIF, ...) and preprocesses an image; NULL on error. The grid is final here.
mkv_image * mkv_image_load(mkv_ctx * ctx, const unsigned char * data, size_t len);
// grid of merged image tokens (columns, rows); returns nx * ny, the number of embedding rows
int mkv_image_grid(const mkv_image * im, int * nx, int * ny);
// Runs the encoder (CPU); 0, or -1 (mkv_last_error). A second call returns at once.
int mkv_image_encode(mkv_ctx * ctx, mkv_image * im);
// [nx * ny][mkv_n_embd] fp32 rows in raster order after mkv_image_encode, else NULL; valid until mkv_image_free
const float * mkv_image_rows(const mkv_image * im);
// wall time of the encode in ms (0 before it)
double mkv_image_encode_ms(const mkv_image * im);
void mkv_image_free(mkv_image * im);

#ifdef __cplusplus
}
#endif
