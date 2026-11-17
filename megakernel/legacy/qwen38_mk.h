// Qwen3.8-27B (llama.cpp arch "qwen35", shared with Qwen3.5/3.6) single-launch-per-cycle CUDA megakernel decoder.
#pragma once

#include <functional>
#include <string>
#include <vector>

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
    // when the fp16 cache for ctx positions does not fit in the free GPU memory); -n: auto for n copies
    // (the engine's and n - 1 parallel slots of ctx positions).
    // kv_ctx: positions of the engine's own K/V cache (default ctx); parallel slots (slots_open) have their own, so a
    // server running only slots can keep it small
    MKEngine(const std::string & gguf_path, int ctx, int n_draft, const std::string & draft_vocab = "", int kv_mode = -1, int kv_ctx = 0);
    ~MKEngine();

    // Clears all sequence state (KV, recurrent state, positions).
    void reset();
    // Processes the prompt, or appends `tokens` to the sequence so far (prompt + emitted tokens,
    // e.g. the next user turn); returns the first generated token (greedy).
    int prefill(const std::vector<int> & tokens);
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
    static int n_vocab();
    void bench_phases();
    // test path: greedy decode of several sequences at once (batched weight GEMMs, per-sequence state); returns the
    // n_new tokens of each, step_secs = time of the decode steps
    std::vector<std::vector<int>> ms_generate(const std::vector<std::vector<int>> & prompts, int n_new, int ctx_s, double & step_secs, bool tc = false);
    // slots (server): persistent sequences decoded together; slots_step runs one speculative step of the given slots
    // and returns the tokens each emitted (greedy, or sampled with the slot's parameters: exact speculative sampling)
    // The slots' K/V is one pool of n x ctx positions (ctx 0: all the device memory left) in pages any slot draws on
    // as its context grows (idle slots' caches are reclaimed); seg_tokens > 0: shared segments (prompt prefixes
    // stored once, see slot_begin) too. Returns the pool's positions; a slot the pool cannot extend emits nothing.
    int slots_open(int n, int ctx, int seg_tokens = 0);
    int slot_segment(int slot) const; // positions of the shared segment slot `slot` reads (0: none)
    int segments_drop(); // frees the segments no slot reads (pinned ones too); returns how many
    void profile_report(); // (MK_PF_PROFILE) the phase times collected since the last report
    int slot_start(int slot, const std::vector<int> & prompt, float temp, float top_p, int top_k, float min_p, unsigned long long seed); // returns reused tokens
    // slot_start in parts, so other slots can step between prefill chunks: slot_begin returns the reused count,
    // slot_prefill feeds up to `budget` prompt tokens and returns how many remain (0: ready to step; budget 0 abandons)
    // share_len: positions < share_len are a prefix other prompts may share; unless a segment holds it already (then
    // the slot attaches to it: no prefill of it), the prefill makes it one (pin: kept while unused)
    int slot_begin(int slot, const std::vector<int> & prompt, float temp, float top_p, int top_k, float min_p, unsigned long long seed,
                   int share_len = 0, bool pin = false);
    int slot_prefill(int slot, int budget);
    std::vector<std::vector<int>> slots_step(const std::vector<int> & slots);
    // slots_step with the next chunk of slot `slot`'s prompt in the same pass (the weights read once for both): the
    // chunk and the step's rows fill `budget` rows; remain <- the prompt tokens left, as slot_prefill returns
    std::vector<std::vector<int>> slots_step_mixed(int slot, int budget, const std::vector<int> & slots, int & remain);
    std::vector<int> slot_sequence(int slot) const; // the slot's tokens (prompt, emitted ones)
    std::vector<float> slot_h(int slot) const; // (test) hidden state of the slot's last processed position
    // ... with one MTP draft per sequence and step (oracle 1 / 2: drafts from `ref`, right / wrong, for testing)
    std::vector<std::vector<int>> ms_generate_spec(const std::vector<std::vector<int>> & prompts, int n_new, int ctx_s, double & step_secs,
                                                   int oracle = 0, const std::vector<std::vector<int>> * ref = nullptr, double * acc_rate = nullptr);
    int n_draft() const { return n_draft_; }
    MKStats stats;

  private:
    struct Impl;
    Impl * p_;
    int n_draft_;
};
