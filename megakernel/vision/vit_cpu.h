// The Qwen3.x vision tower (mmproj projector "qwen3vl_merger") on the CPU, in fp32 with hand-written AVX2 / FMA
// kernels: llama.cpp's image preprocessing bit for bit, then the ViT (patch embedding, learned position embedding,
// 27 pre-LN blocks with 2D vision RoPE and full attention, GELU MLP) and the 2x2 patch merger.
//
// Why not libmtmd's encoder: on Zen 3 its ggml kernels run at ~25% of the cores' fp32 peak (latency-bound: 32 threads
// beat 16, and its static work split stalls on the slowest thread). Here every matrix product runs through a packed
// 6x16 FMA micro-kernel and attention through a tiled online-softmax kernel, both at the FMA peak of one core per
// thread, on a pool of threads pinned one per physical core that take work dynamically.
//
// Arithmetic: weights as stored, widened to fp32 (BF16 / F16 exactly, Q8_0 dequantised); activations, attention and
// GELU (tanh form, as llama.cpp's graph uses) in fp32 throughout: closer to the checkpoint than libmtmd's path, which
// rounds GEMM inputs to the weight type, K / V to fp16 and GELU through an fp16 table.
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace mkv {

class Vit {
  public:
    struct Params {
        int n_threads = 0;   // <= 0: one per physical core
        bool pin = true;     // pin worker i to physical core i (first SMT sibling; the second siblings after that)
        int min_tokens = 0;  // image-token limits of the resize (<= 0: llama.cpp's defaults for the projector, 8 / 4096)
        int max_tokens = 0;
    };
    Vit(const std::string & mmproj, const Params & p);
    ~Vit();

    // llama.cpp's smart resize: the size an image of w x h is resized to (multiples of 32)
    void target_size(int w, int h, int & tw, int & th) const;
    // RGB8 (nx * ny * 3, as stb_image decodes it) -> llama.cpp's preprocessed image: bilinear resize to target_size,
    // u8, then (x / 255 - mean) / std; fp32, HWC
    void preprocess(const uint8_t * rgb, int nx, int ny, std::vector<float> & img, int & tw, int & th) const;
    // preprocessed image (HWC, tw and th multiples of 32) -> (tw / 32) * (th / 32) rows of n_embd(), raster order of the
    // merged 2x2 patch groups (row-major, tw / 32 per row)
    void encode(const float * img, int tw, int th, float * out);

    int n_embd() const;    // projection size (the text model's embedding size)
    int n_threads() const;
    // wall time of the last encode's phases in ms: patch embedding, blocks (GEMMs), blocks (attention), merger, total
    double ms[5] = {};

  private:
    struct Impl;
    std::unique_ptr<Impl> p_;
};

} // namespace mkv
