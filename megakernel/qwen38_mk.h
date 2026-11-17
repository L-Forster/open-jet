// Qwen3.8-27B (llama.cpp arch "qwen35", shared with Qwen3.5/3.6) single-launch-per-cycle CUDA megakernel decoder.
#pragma once

#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

// A failed (non-sticky) CUDA call inside the engine, or no finite logit. The sequence it ran on is then undefined:
// reset() it before the next use (the C API does; a slot call marks its slots failed, see slots_open). Sticky errors
// (the CUDA context is lost) exit the process with code 4.
struct MKCudaError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

// prefill() stopped by MKEngine::cancel(). The engine is consistent (see cancel()): no reset needed.
struct MKCancelled : std::runtime_error {
    using std::runtime_error::runtime_error;
};

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
    int last_T = 0, last_nd = 0, last_acc = 0; // the last step: tokens verified, drafts produced, accepted
};

class MKEngine {
  public:
    // n_draft: MTP draft tokens per cycle (0 .. MAXD - 1: 7 in the served build). ctx: KV-cache capacity in tokens.
    // draft_vocab: optional file of int32 token ids; the MTP draft head is restricted to them.
    // kv_mode: KV cache format, 0 fp16, 1 q8_0 (half the memory, ~1-2% slower), -1 auto (q8_0 only
    // when the fp16 cache for ctx positions does not fit in the free GPU memory).
    // mtp: where the MTP draft layer (blk.64) comes from: "" the model file (else the env vars MK_NO_MTP=1 / MK_MTP=FILE),
    // a path: that GGUF's MTP layer (it must belong to this model; it replaces the model file's own), "none": no MTP layer.
    // Without one (also a model file that has none) the engine decodes without drafting: n_draft is 0 whatever is asked.
    MKEngine(const std::string & gguf_path, int ctx, int n_draft, const std::string & draft_vocab = "", int kv_mode = -1,
             const std::string & mtp = "");
    ~MKEngine();

    // Clears all sequence state (KV, recurrent state, positions).
    void reset();
    // Processes the prompt, or appends `tokens` to the sequence so far (prompt + emitted tokens,
    // e.g. the next user turn); returns the first generated token (greedy).
    // ck_at (absolute position inside the fed range): save the rewind checkpoint there instead of at the end of the
    // prompt, e.g. at the start of the generation prompt, which a follow-up turn's template may render differently
    // (thinking mode re-renders the previous turn as <think>\n\n</think>, one token before the end of the prompt).
    int prefill(const std::vector<int> & tokens, int ck_at = -1);
    // prefill() saving checkpoints at the absolute positions cks (inside this prompt) into the checkpoint pool:
    // MK_CK_SLOTS (default 8) checkpoints of 0.17 GB of pinned host memory each, evicted by coverage when full
    // (older ones thin out, recent ones stay dense). ck_exact (or -1) is kept at its exact position; the others move
    // down by up to 31 tokens onto the batched prefill's scan grid. Saving changes nothing the prefill computes.
    // Unless ck_exact is inside the batched part, a checkpoint at the end of the prompt is saved too (as prefill()).
    // MK_EXACT: ck_exact moves onto the grid like the others; no end-of-prompt checkpoint (MK_PF_CAP below 384: none).
    int prefill(const std::vector<int> & tokens, const std::vector<int> & cks, int ck_exact);
    // prefill(tokens, cks, ck_exact) with images inside the prompt (each ends before its last token; batched prefill
    // only). The sequence keeps their positions for its RoPE rows until a reset, rewind or truncate drops them; after a
    // rewind into an image, the next prefill must pass that image again (same pos and grid).
    int prefill(const std::vector<int> & tokens, const std::vector<int> & cks, int ck_exact, const std::vector<MKImage> & images);
    // Sampling: temp 0 = greedy (default); otherwise llama.cpp's order top_k (1..32), top_p, min_p,
    // temperature.  Speculative verification keeps the sampled distribution exact.
    void set_sampling(float temp, float top_p, int top_k, float min_p, unsigned long long seed);
    // Presence / frequency penalty (OpenAI parameters, vLLM's semantics): each generated token's logit is lowered by
    // frequency * c + presence * (c > 0), c = its occurrences among the tokens generated so far in this response
    // (since the last prefill; prompt tokens do not count). Greedy and sampled; speculative verification stays exact
    // (a verify row counts the drafts before it). 0, 0 (default): off. The engine's own sequence only: parallel slots
    // never apply them.
    void set_penalty(float presence, float frequency);
    // Draft-depth policy, "key=value,..." (see engine_impl.h); "" restores the default. The env var
    // MK_POLICY sets it at construction.
    void set_policy(const std::string & spec);
    // KV-cache quantisation emulation (builds with -DMK_KV_EMU=1; accuracy experiments): the MK_KV_EMU_SPEC syntax
    // (see engine_impl.h), applied to the rows stored from now on. Throws on other builds.
    void set_kv_emu(const std::string & spec);
    // Drops emitted tokens beyond `len` sequence positions (e.g. those generated after an
    // end-of-turn token within the last cycle) before the next prefill.
    void truncate(int len);
    // The sequence so far (prompts and emitted tokens; position = index) and the number of its
    // positions folded into the recurrent state (truncate() can only go back to there).
    const std::vector<int> & sequence() const;
    int committed() const;
    // Each prefill saves a checkpoint at the end of its prompt (or at ck_at, and at cks); rewinding to one
    // drops everything after it (also tokens before the committed position) and the checkpoints after it, so a
    // prompt that diverges anywhere after the checkpoint can be re-fed from there. A rewind or truncate also sets the
    // draft vocabulary to what a fresh prefill of the kept prefix builds, and the next prefill extends it as a fresh
    // prefill of the whole sequence would: the drafts (and so sampled replies) match the same prompt sent fresh.
    // checkpoints(): positions of the checkpoints (ascending); rewind_to(pos): restores the one at pos (false: none);
    // checkpoint_pos() / rewind_to_checkpoint(): the last one (-1 / false: none).
    std::vector<int> checkpoints() const;
    bool rewind_to(int pos);
    int checkpoint_pos() const;
    bool rewind_to_checkpoint();
    // Pinned checkpoints: ck_pin(positions) pins, for each position P, the checkpoint at the last position in (P - 32, P]
    // (where a prefill's cks move P onto the scan grid): one there now, else the one the next prefill saves there (it
    // saves it whether or not cks has P). A pinned checkpoint is never evicted from the pool and goes with every park of
    // the sequence (besides the newest max_checkpoints); a rewind before it or a reset drops it like any checkpoint. The
    // caller pins the end of a prompt's shared prefix (a client's system prompt and tools), so a new session resumes
    // there however long the cached conversation grew. At most min(2, MK_CK_SLOTS / 4) are pinned (none with fewer than
    // 4 slots); more release the least recently requested. Returns how many were pinned now. ck_pinned(): their
    // positions (ascending).
    int ck_pin(const std::vector<int> & positions);
    std::vector<int> ck_pinned() const;
    // Asks a batched prefill running on another thread (or the next one) to stop at its next chunk boundary: a host flag
    // read between chunks, no GPU work, callable from any thread. That prefill() then waits for the chunks already on
    // the GPU and throws MKCancelled, with nothing of it kept: the sequence, the recurrent state and the checkpoints go
    // back to the last checkpoint that existed before the call (or to a reset when there is none), the draft vocabulary
    // and the image spans to what they were at that position. Each prefill() clears the flag when it returns;
    // cancel(false) clears it too.
    void cancel(bool on = true);
    // Progress of the prefill() running on another thread (callable from any thread; host counters, no GPU work): *done
    // of the *total positions of its batched part are processed. After it returns: that prefill's last values.
    void prefill_progress(int * done, int * total) const;
    // Runs one verify+draft cycle; appends 1..n_draft+1 accepted tokens to `out`.
    int step(std::vector<int> & out);
    // Debug helper: the final normalised hidden row of the last processed position (the logits are not kept).
    std::vector<float> last_hidden();
    // Test helper (exact-reuse probes): 64-bit digests of the sequence state, layout in engine.cu (state_digest).
    std::vector<unsigned long long> state_digest();
    // Test helper: the draft vocabulary's rows (token ids) as the device holds them, checked against the host's list
    // (throws if they differ); empty without a draft vocabulary.
    std::vector<int> draft_vocab();
    // MK_EXACT=1 at construction: prefill results independent of the chunking (see engine_impl.h `exact`).
    bool exact() const;
    // false: no MTP layer (see the constructor's mtp): depth 0 only
    bool has_mtp() const;

    // ---- conversation state in host memory: side requests and parked conversations ----
    // Host store: pinned host memory in chunks of MK_STORE_CHUNK_MIB (default 32) MiB, allocated on first use and then kept
    // for reuse (pinned allocation runs at ~1 GB/s under WSL2), at most `bytes` in all; 0 (the default) disables
    // side_begin and park. A lower budget frees the unused chunks above it.
    void store_budget(size_t bytes);
    // allocates chunks now (within the budget) until `bytes` are free, so the next copies need not allocate
    void store_reserve(size_t bytes);
    struct StoreInfo {
        size_t budget, allocated, used; // bytes: the budget, chunks allocated, chunks holding entries / a side request's rows
        int parked;                     // parked conversations
        double last_ms;                 // the last side_end / park / unpark: wall time, bytes copied
        size_t last_bytes;
        double side_ms; // the running (or last) side request: time spent copying the kept sequence's rows out
        double side_park_ms;    // the last side_end_park: parking the side request's conversation (0: not parked)
        size_t side_park_bytes;
    };
    StoreInfo store_info() const;
    // bytes per token of the KV cache (every slot, K and V), of the MK_KV_WIN rings, of the recurrent state (DeltaNet /
    // conv / h ring), of one checkpoint (with its window rings under MK_EXACT)
    void kv_sizes(size_t * per_token, size_t * rings, size_t * state, size_t * checkpoint) const;
    // Side request (a short unrelated prompt without losing the cached sequence): side_begin() keeps the current
    // sequence aside (its recurrent state is copied to the host store now; its KV rows and window-ring rows only when a
    // launch of the side request is about to overwrite them, 1024 rows at a time) and leaves the engine as reset() does,
    // with its checkpoints hidden. The side request saves checkpoints of its own in the pool: a slot holding one of the
    // kept sequence's is copied to the store first. side_end() copies everything back: the kept sequence is then bit for
    // bit as before side_begin (returns 1), or, if the store ran out of room during the side request, the engine is reset
    // (0: the kept sequence is lost, the next prompt is prefilled in full). side_drop(): discard it (reset).
    // false from side_begin: no room in the store (nothing changed).
    // side_end_park(max_checkpoints, id): as side_end, but the side request's own conversation is parked first (as
    // park(max_checkpoints): *id its entry, 0 when it is empty or the store has no room for it); if the kept sequence
    // was lost (returns 0), the side request's conversation stays as the engine's sequence instead.
    // side_unpark(id): right after side_begin, the side request continues parked conversation id: the engine becomes it
    // bit for bit (as unpark; the entry is freed) while the kept sequence stays aside (the rows, window rings and
    // checkpoint blocks it overwrites are copied to the store first). 1 done, 0 no room in the store (nothing changed),
    // -1 no such entry.
    bool side_begin();
    int side_end();
    int side_end_park(int max_checkpoints, int * id);
    int side_unpark(int id);
    void side_drop();
    bool side_active() const;
    // Parked conversations: park() copies the whole sequence state (KV rows, window rings, recurrent state, the sequence,
    // image spans, draft vocabulary, the newest max_checkpoints checkpoints and, if max_checkpoints > 0, the pinned ones)
    // into a new store entry and returns its id
    // (> 0); the engine itself is unchanged. 0: the sequence is empty; -1: no room in the store. unpark(id): the engine
    // becomes that conversation bit for bit (its current sequence is discarded) and the entry is freed; false: no such
    // entry. park_need: the bytes park() would take now.
    // "Bit for bit" (unpark, side_unpark, side_end) is the sequence state: KV rows, window rings, recurrent state,
    // checkpoints, bookkeeping. A decode step also reads device state that is not kept: the DeltaNet replay rows of the
    // drafts the last cycle accepted, the drafts' q' (sampled drafts) and the response's penalty counts. Continue with a
    // prefill (the next turn's tokens: it refeeds from the committed position and rebuilds them, as the server does).
    // step() before that throws where it would read them: the first step unless no accepted draft is pending (as right
    // after a prefill) and the drafts are greedy-verified or point-mass, and any step with penalties.
    int park(int max_checkpoints);
    bool unpark(int id);
    bool park_drop(int id);
    size_t park_need(int max_checkpoints) const;
    size_t park_bytes(int id) const;

    // Draft depth for the following prefills and steps, 0 .. the n_draft given at construction (CLI batch mode: an engine
    // built for the deepest job runs each job at its own depth; the same launches as an engine built for that depth).
    // Without an MTP layer any depth becomes 0 (a notice, not an error).
    void set_n_draft(int n);
    // Post-prefill state files (test runs: decode A-B-As at long context without re-prefilling). prefill(tokens) on an
    // empty sequence that first looks in dir for a state file of the same batched prefill (every prompt token but the
    // last, which then goes through the megakernel as in prefill()) and restores it instead of running the prefill; on a
    // miss it runs the prefill and saves the file. The key covers the binary (its file hash), the GPU, the model file,
    // the context, the KV format and every MK_* setting that is not known to be decode-only, so a restore leaves the state
    // bit for bit as the prefill would. *how: 1 restored, 2 saved (0: not applicable, plain prefill).
    int prefill_state(const std::vector<int> & tokens, const std::string & dir, int * how = nullptr);

    int position() const;

    // Parallel slots (open-jet's slot API): n persistent sequences (lanes) of ctx positions each (0: the most that fit),
    // decoded together by batched launches; returns ctx. Needs the plain q8 KV cache and a model whose
    // delta-net projections are all in the wv layout (plain Q4_K_M). The draft vocabulary is the base list from then on.
    // slot_begin: the slot takes the prompt (sampling as set_sampling), reusing the longest prefix it can (a truncate of
    // emitted tokens, else its newest checkpoint before the divergence); returns the reused tokens, slot_prefill feeds
    // up to budget of the rest per call and returns how many remain (0: ready; budget 0: abandon the rest). The prompt's
    // last chunk saves a checkpoint before its last token (a repeat of the prompt resumes there). MK_EXACT: a chunk
    // before the last ends on the 32-token scan grid and a last chunk of 32 tokens or fewer joins the one before (a
    // budget of 32 or more: chunks within 32 tokens of it); the checkpoint is the grid point at or before the last token.
    // MK_EXACT follow-ups: a prompt that reaches the last token of the slot's last prompt (last prompt + reply + new turn,
    // or that prompt again) reuses none of the rows from there on (the megakernel's, whose bits depend on the draft
    // schedule): the slot goes back to its newest checkpoint inside the last prompt (else empties it) and prefills the
    // rest, so the reply is the one a fresh slot gives (calling slot_begin(slot, that last prompt) first, as mk_server's
    // --agents scheduler does, changes nothing). slots_step: one step of the given slots (two or more: one batched
    // launch, each slot up to 1 + MAXT / n - 1 tokens), the tokens each emitted (a slot's first call after its prefill:
    // the prefill's token only; empty: its context is full).
    // slots_step_mixed: pslot's next prompt chunk, then slots_step (a failed chunk steps nothing: no token is lost).
    // Every slot call leaves the engine's own sequence current again. An MKCudaError fails the slots the call ran on:
    // slot_prefill / slots_step refuse them (std::runtime_error) until slot_begin, which starts them from scratch
    // (reuses nothing). A slot_prefill stopped by cancel() (MKCancelled) queues the prompt's rest again from where the
    // slot's sequence went back to.
    int slots_open(int n, int ctx, int seg_tokens = 0);
    int slots_count() const;
    int slot_begin(int slot, const std::vector<int> & toks, float temp, float top_p, int top_k, float min_p, unsigned long long seed,
                   int share_len = 0, bool pin = false);
    int slot_prefill(int slot, int budget);
    std::vector<std::vector<int>> slots_step(const std::vector<int> & slots);
    std::vector<std::vector<int>> slots_step_mixed(int pslot, int budget, const std::vector<int> & slots, int & remain);
    int slot_segment(int) const { return 0; } // (no shared segments)
    std::vector<int> slot_sequence(int slot);
    std::vector<float> slot_h(int slot);
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
    // Keep-warm load (capi.cpp, MK_KEEP_WARM_S): launches one kernel of blocks x threads on `stream` (a cudaStream_t)
    // that touches no engine memory and ends when the word at stop (device pointer of host-mapped memory) becomes
    // non-zero, or after ms milliseconds; spin 1 = every thread busy-loops, 0 = one thread polls with __nanosleep.
    // Returns the launch's cudaError_t (0: launched).
    static int warm_launch(void * stream, const unsigned * stop, double ms, int blocks, int threads, int spin);
    void bench_phases();
    int n_draft() const { return n_draft_; }
    MKStats stats;

  private:
    struct Impl;
    Impl * p_;
    int n_draft_;
};
