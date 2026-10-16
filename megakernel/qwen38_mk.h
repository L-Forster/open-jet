// Qwen3.8-27B (llama.cpp arch "qwen35", shared with Qwen3.5/3.6) single-launch-per-cycle CUDA megakernel decoder.
#pragma once

#include <functional>
#include <string>
#include <vector>

// An image in a prompt (vision input, e.g. an mmproj encoder's output): positions pos .. pos + nx * ny - 1 take these
// input-embedding rows instead of their tokens' embeddings (the tokens there are placeholders, e.g. <|image_pad|>),
// with llama.cpp's M-RoPE positions for Qwen-VL: the grid's rows in raster order, nx merged patches per row.
struct MKImage {
    int pos;            // absolute sequence position of the first row
    int nx, ny;         // grid: columns, rows
    const float * embd; // [nx * ny][5120] fp32 (read during the prefill call)
};

struct MKStats {
    long cycles = 0;
    long drafted = 0;
    long accepted = 0;
};

class MKEngine {
  public:
    // n_draft: MTP draft tokens per cycle (0..4). ctx: KV-cache capacity in tokens.
    // draft_vocab: optional file of int32 token ids; the MTP draft head is restricted to them.
    // kv_mode: KV cache format, 0 fp16, 1 q8_0 (half the memory, ~1-2% slower), -1 auto (q8_0 only
    // when the fp16 cache for ctx positions does not fit in the free GPU memory).
    MKEngine(const std::string & gguf_path, int ctx, int n_draft, const std::string & draft_vocab = "", int kv_mode = -1);
    ~MKEngine();

    // Clears all sequence state (KV, recurrent state, positions).
    void reset();
    // Processes the prompt, or appends `tokens` to the sequence so far (prompt + emitted tokens,
    // e.g. the next user turn); returns the first generated token (greedy).
    int prefill(const std::vector<int> & tokens);
    // prefill() with images inside the prompt (each ends before its last token; batched prefill only). The sequence
    // keeps their positions for its RoPE rows until a reset, rewind or truncate drops them; after a rewind into an
    // image, the next prefill must pass that image again (same pos and grid).
    int prefill(const std::vector<int> & tokens, const std::vector<MKImage> & images);
    // Sampling: temp 0 = greedy (default); otherwise llama.cpp's order top_k (1..32), top_p, min_p,
    // temperature.  Speculative verification keeps the sampled distribution exact.
    void set_sampling(float temp, float top_p, int top_k, float min_p, unsigned long long seed);
    // Drops emitted tokens beyond `len` sequence positions (e.g. those generated after an
    // end-of-turn token within the last cycle) before the next prefill.
    void truncate(int len);
    // The sequence so far (prompts and emitted tokens; position = index) and the number of its
    // positions folded into the recurrent state (truncate() can only go back to there).
    const std::vector<int> & sequence() const;
    int committed() const;
    // Each prefill saves a checkpoint at the end of its prompt; rewinding to it drops everything
    // after (also tokens before the committed position), so a prompt that diverges anywhere after
    // the checkpoint can be re-fed from there.  Returns false when there is no checkpoint.
    int checkpoint_pos() const;
    bool rewind_to_checkpoint();
    // Runs one verify+draft cycle; appends 1..n_draft+1 accepted tokens to `out`.
    int step(std::vector<int> & out);
    // Debug helper: copy of the last main-model logits row selection (argmax ids) is not kept;
    // instead expose the final normalised hidden row of the last processed position.
    std::vector<float> last_hidden();

    int position() const;
    // Teacher-forced greedy predictions: pred[i] = argmax of the next-token distribution after
    // tokens[0..i] (resets the sequence).
    std::vector<int> score(const std::vector<int> & tokens);
    // Teacher-forced next-token logits (resets the sequence): sink(i0, n, logits) receives the logits
    // ([n][n_vocab()], valid during the call) after tokens[0..i] for i = i0 .. i0 + n - 1, in position
    // order, for all positions >= first.  Positions below mk_from are processed by the batched
    // prefill kernels, the others by the decode megakernel.
    using LogitSink = std::function<void(int i0, int n, const float * logits)>;
    void score_logits(const std::vector<int> & tokens, int mk_from, int first, const LogitSink & sink);
    // score_logits with images (inside the batched part: below mk_from)
    void score_logits(const std::vector<int> & tokens, int mk_from, int first, const LogitSink & sink,
                      const std::vector<MKImage> & images);
    static int n_vocab();
    static int n_embd(); // width of an MKImage row
    void bench_phases();
    int n_draft() const { return n_draft_; }
    MKStats stats;

  private:
    struct Impl;
    Impl * p_;
    int n_draft_;
};
