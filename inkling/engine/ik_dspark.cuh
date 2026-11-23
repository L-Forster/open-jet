// Inkling 2 Mini DSpark (mtp.safetensors) block-parallel + recurrent Markov drafter.
#pragma once

#include <cstddef>
#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <functional>
#include <string>

namespace ik {

class DSpark {
public:
    static constexpr int N_LAYERS = 3;
    static constexpr int N_TAPS = 4;
    static constexpr int E = 2304;
    static constexpr int H = 24;
    static constexpr int HK = 8;
    static constexpr int D = 128;
    static constexpr int KV = HK * D;       // 1024
    static constexpr int FF = 9216;
    static constexpr int BLOCK = 8;         // 1 anchor + 7 draft masks
    static constexpr int N_SPEC = BLOCK - 1;
    static constexpr int RANK = 256;
    static constexpr int SWA = 1024;
    static constexpr int RING = SWA + BLOCK;  // 1032

    // out_proj(X_dev [T, E], Y_dev [T, n_vocab_pad], T, st): applies the target model's output.weight.
    using LmHeadFn = std::function<void(const float * X, float * Y, int T, cudaStream_t st)>;

    DSpark(const std::string & path, int ctx, int n_vocab, int n_vocab_pad, float softcap,
           const LmHeadFn & lm_head, cudaStream_t st);
    ~DSpark();

    DSpark(const DSpark &) = delete;
    DSpark & operator=(const DSpark &) = delete;

    // Store T target tokens' 4-tap hidden states [T][4][E] at sequence positions [pos0, pos0 + T)
    // and update last_taps_ to token T-1 (used for f_rnn).
    void store_context(const float * d_taps, int pos0, int T, cudaStream_t st);

    // Draft up to n_draft (1..7) speculative tokens starting at pos+1 given:
    //   d_anchor_emb: [E] target's normalized embedding of anchor_tok (at position pos)
    //   anchor_tok:   anchor token id (for Markov step 0)
    //   pos:          sequence position of anchor_tok (context KV holds 0..pos-1)
    //   lm_head:      target's output.weight matmul for base_logits [7, n_vocab_pad]
    // Writes n_draft token ids into out_drafts[0..n_draft-1].
    void draft(const float * d_anchor_emb, int anchor_tok, int pos, int n_draft,
               const LmHeadFn & lm_head, int * out_drafts, cudaStream_t st);

    void reset();

    // Checkpoint support: saves/restores the 2 local sliding-window rings + last_taps_.
    size_t ck_bytes() const;
    void ck_save(uint8_t * dst, cudaStream_t st) const;
    void ck_restore(const uint8_t * src, cudaStream_t st);

private:
    struct Impl;
    Impl * p_ = nullptr;
};

}  // namespace ik
