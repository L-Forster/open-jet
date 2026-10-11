# Qwen3.8-27B megakernel (RTX 30/40/50 series)

A specialised CUDA inference engine for `Qwen3.8-27B-Q4_K_M.gguf` (llama.cpp arch `qwen35`:
48 Gated-DeltaNet + 16 gated-attention layers, one MTP block) on RTX 30, 40 and 50-series GPUs
with 24 GB+ (sm_86, sm_89, sm_120; tuned and benchmarked on an RTX 3090), with an OpenAI-compatible server that open-jet uses in place of `llama-server`
(`llama_backend: auto`, see [docs/runtimes/llama-cpp.md](../docs/runtimes/llama-cpp.md)).

## Quick guide

1. Run `openjet setup` and choose "Qwen3.8 27B Q4_K_M MTP". It downloads the model and builds the
   megakernel (by hand: [Build / run](#build--run)).
2. Run `openjet`. It serves the model through the megakernel.

Which model file runs on it (details in [The GGUF](#the-gguf)):

| Model file | Runs on |
|---|---|
| unsloth's original Qwen3.8-27B `Q4_K_M`, the one the catalog downloads | megakernel |
| a Qwen3.8-27B `Q4_K_M` from lmstudio-community, mradermacher, Abiray or KikoCis | megakernel, semi compatible |
| any other quant (unsloth `UD-*`, bartowski, GSQ-RCO, ...) | llama.cpp |

## How it works

Decoding runs an entire speculative cycle in one persistent cooperative kernel launch:

```
main model over T = 1 + n_draft tokens  ->  lm_head + verification on device (greedy, or exact
speculative sampling)  ->  MTP catch-up rows for the accepted tokens  ->  n_draft MTP draft steps
```

The ~340 phases of a cycle (projections, DeltaNet core, attention, FFN, lm_head, MTP) are separated
by a software grid barrier; the host only reads back the accepted tokens. Prompts go through
batched tensor-core kernels instead (see Prefill).

## Results

RTX 3090, against `llama-server` with open-jet's flags (`--spec-type draft-mtp --spec-draft-n-max 2
-fa on -ctk q8_0 -ctv q8_0`). Back-to-back on an otherwise idle GPU; numbers vary with prompt
predictability (draft acceptance) and other GPU load.

**Reasoning mode as open-jet runs it** (thinking on, xhigh effort, temperature 1.0, top_p 0.95,
top_k 20; 5 coding prompts x 2 seeds, up to 1024 tokens each):

| | llama.cpp | megakernel | |
|---|---|---|---|
| decode through the HTTP servers (`/v1/chat/completions`) | 50.9 tok/s | **78.0 tok/s** | **1.53x** |
| decode, CLI (`bench/think_bench.py`, `-d 4`) | 50.7 tok/s | 77.4 tok/s | 1.53x |

**Greedy decode, 4 chat prompts x 512 tokens** (`bench/code_bench.py`):

| prompt | llama.cpp | megakernel `-d 3` | megakernel `-d 4` |
|---|---|---|---|
| Python function (code) | 72.5 | 126.3 | **139.7** (1.93x) |
| hash map explanation | 64.1 | 107.4 | 106.5 |
| short story | 52.8 | 66.8 | 70.7 |
| Flask to FastAPI plan | 63.2 | 102.6 | 109.8 |
| **mean** | 63.1 | 100.8 | **106.7** (1.69x) |

Without speculation: llama.cpp 40.4 tok/s, megakernel 42.9 (`-d 0`): the decode itself is close to
the memory-bandwidth limit for both; the lead comes from verifying 4-5 drafted tokens per pass at
almost no extra cost.

**A source file in the prompt, then a coding task** (greedy, 256 tokens; prompt = batched prefill of
the whole prompt incl. the first token):

| context | decode llama.cpp | decode megakernel | prompt llama.cpp | prompt megakernel |
|---|---|---|---|---|
| 1.8K | 57.4 | 84.8 | 1050 | 1595 |
| 3.6K | 56.3 | 95.3 | 1124 | 1595 |
| 7.8K | 54.8 | 85.8 | 1140 | 1533 |
| 15.9K | 55.7 | 86.5 | 1126 | 1466 |
| 29.5K | 52.3 | 72.7 | 1062 | 1236 |

(tok/s; measured before the sequence-token draft vocabulary and double-buffered attention below,
which add 1-8% to the megakernel's decode here.)

**Pasting a file into the next turn of a conversation** (prompt processing time):

| new tokens | llama.cpp (prompt cache) | megakernel |
|---|---|---|
| 2.7K | 2.41 s | 1.68 s |
| 12K | 10.45 s | 8.10 s |

With the q8_0 KV cache (chosen automatically for contexts above ~80K tokens, e.g. open-jet's 100K)
decode is ~1-2% and prefill 2-7% slower than with fp16.

**Correctness.** The output is not byte-identical to llama.cpp's: the arithmetic is ordered and
rounded differently, so where two tokens are close the other one can win, and a generation then
continues differently. Measured against llama.cpp's own next-token distributions (same GGUF,
13,299 positions of code and prose from this repo, each after a 1024-token prompt; reference:
llama.cpp generating token by token with its q8_0 KV cache, as open-jet runs it):

| against llama.cpp generating token by token | top token agrees | KL divergence (mean, nats) | perplexity (reference 2.4263) |
|---|---|---|---|
| llama.cpp, batched instead | 98.68% | 0.0019 | 2.4283 |
| llama.cpp, fp16 KV cache instead | 99.17% | 0.0007 | 2.4284 |
| megakernel (prefill kernels, then the decode kernel), q8_0 KV | 99.08% | 0.0009 | 2.4274 |
| megakernel, fp16 KV | 99.11% | 0.0008 | 2.4273 |
| megakernel, prefill kernels only, q8_0 KV | 99.01% | 0.0009 | 2.4275 |

The megakernel is as close to llama.cpp as llama.cpp's own configurations are to each other, and
its log-loss differs by +0.0005 +- 0.0004 (not significant). Most top-token disagreements are
near-ties (median lead of llama.cpp's token 0.03-0.04 nats); in 2-3 of ~120 it led by more than
0.5 nats, as it does in llama.cpp against itself (4 of 176).

At a 30K context (one sequence, 1,999 positions scored after a 28,000-token prompt, same
reference setup) the megakernel agrees on 99.65% of top tokens with KL 0.0007 (q8_0 KV; fp16:
99.55%, 0.0007), llama.cpp batched on 99.35% with KL 0.0009; log-loss against the reference is
+0.0018 +- 0.001 (fp16 KV +0.0027), llama.cpp batched +0.0017. At 96K (open-jet's setting; 1,999
positions after a 94,000-token prompt, q8_0 KV) the megakernel agrees on 99.60% with KL 0.0008 and
log-loss +0.0012 +- 0.001, llama.cpp batched on 99.45% with KL 0.0014 and +0.0015 +- 0.002. Each
long-context figure is from a single sequence of this repo's code. To reproduce:

```sh
make tools/llama_ref
tools/llama_ref -m model.gguf -f code.txt -f prose.txt --score-from 1024 -b 1 --ref-out llama.ref
tools/llama_ref -m model.gguf --kld llama.ref                            # llama.cpp batched: the noise floor
./qwen38_mk -m model.gguf --kld llama.ref --score-path mixed --kv q8     # the megakernel
```

`llama_ref` teacher-forces 2048-token chunks of the text (`--chunk`) and stores, per scored
position, the 256 most likely tokens (`logit_stats.h`); `--kld` compares an engine with such a file
(`--ref-out` also writes its own). `--score-path` selects the megakernel's batched prefill kernels
(`prefill`, the default), the decode kernel (`decode`), or the prefill kernels for the unscored
prompt and the decode kernel for the scored positions (`mixed`, what a generation runs).

Speculative verification is exact: greedy output is the main model's greedy choice, and with
sampling the token distribution is the same as without drafts (2000-sample test per position).
Verifying 1 vs 4 tokens per pass rounds slightly differently, so on near-ties the output can differ
between `-d` settings.

## Build / run

Requires CUDA 12+ (12.8+ for RTX 50 series; built with 13.2), an sm_86, sm_89 or sm_120 GPU with 24 GB, and the repo's llama.cpp build (only
`libllama`, for tokenization, loaded vocab-only). `openjet setup` builds `libqwen38mk.so` automatically
(this source ships in the open-jet wheel); building by hand:

```sh
cd megakernel
make        # qwen38_mk (CLI), libqwen38mk.so (server library), tools/vocab_freq; ~2 min
./qwen38_mk -m ../models/Qwen3.8-27B-Q4_K_M.gguf -p "Write quicksort in C" -n 512 \
            -d 4 --draft-vocab draft_vocab_32k.bin --think --temp 1.0 --top-p 0.95 --top-k 20
```

CLI flags: `-d` drafts per cycle (0-4; with 4 the depth drops to 3 for a cycle after a rejected
draft), `-c` KV capacity, `--kv f16|q8|auto` (KV cache format), `--think [xhigh|medium|low]`
(reasoning mode; the chat template's default effort is xhigh), `--temp` / `--top-p` / `--top-k` /
`--min-p` / `--seed` (sampling; default greedy), `--raw` (no chat template), `--ids`,
`--prompt-ids f` (prompt as token ids), `--turn2 text` / `--turn2-file f` (a second user turn
appended to the conversation), `--score ids.txt` (teacher-forced agreement), `--kld ref` (KL
divergence, perplexity and agreement against a reference file, see Correctness), `MK_PROFILE=1`
(per-phase timing, including a "min-composed" estimate that filters out time-slicing by other GPU
users), `MK_PF_PROFILE=1` (per-kernel prefill timing), `MK_PREFILL=mk` (prefill through the decode
megakernel instead).

### The GGUF

The kernels are compiled for unsloth's `Qwen3.8-27B-Q4_K_M.gguf` quant mix, and the loader checks
every tensor against it:

| tensors | type |
|---|---|
| `ffn_gate`, `ffn_up`, `attn_q`, `attn_k`, `attn_output`, `attn_gate`, `token_embd` | Q4_K |
| `ffn_down`, `attn_v`, `attn_qkv` | Q4_K or Q6_K |
| `ssm_out` | Q5_K |
| `ssm_alpha`, `ssm_beta` | F32 |
| `nextn.eh_proj` | Q8_0 |
| `output` | Q6_K |

A plain `llama-quantize ... Q4_K_M` stores `ssm_out`, `ssm_alpha`, `ssm_beta` and `nextn.eh_proj` as
Q4_K; the loader re-encodes those to the table's types (145 tensors) and the file then runs the same
kernels. A file with any other mix is refused, naming the first tensor that does not match, and
open-jet starts `llama-server`. Which files that covers is in the [quick guide](#quick-guide).

To build a file with unsloth's mix from the BF16 GGUF (unsloth's imatrix optional):

```sh
llama-quantize --imatrix imatrix_unsloth.gguf \
    --tensor-type ssm_out=q5_k --tensor-type ssm_alpha=f32 --tensor-type ssm_beta=f32 \
    --tensor-type nextn.eh_proj=q8_0 \
    Qwen3.8-27B-BF16-00001-of-00002.gguf Qwen3.8-27B-Q4_K_M.gguf Q4_K_M
```

Benchmarks: `bench/think_bench.py` (reasoning mode, both engines), `bench/code_bench.py` (greedy
decode, context sweep, second turn), `bench/bench.py` (decode on the 4 chat prompts);
`bench/*_bench.cu` are GEMV / GEMM microbenchmarks.

## Server

`server/mk_server.py` serves the engine (through `libqwen38mk.so`, the C API in `capi.cpp`) with an
OpenAI-compatible `/v1/chat/completions`, streaming or not, plus `/health` and `/v1/models`.
open-jet starts it itself; standalone:

```sh
python3 server/mk_server.py -m ../models/Qwen3.8-27B-Q4_K_M.gguf --port 18080 -c 100000
```

- Prompts are rendered with the chat template stored in the GGUF (jinja2): `chat_template_kwargs`
  `enable_thinking` / `preserve_thinking` / `reasoning_effort`, OpenAI `tools`. Reasoning is streamed
  as `reasoning_content`, Qwen's XML tool calls are returned as structured `tool_calls`.
- Successive requests of a conversation only prefill what is new: the prompt is matched against the
  engine's sequence (prompts + emitted tokens), and when the re-rendered history diverges after the
  end of the previous prompt (e.g. a reformatted tool call) the engine rewinds to a checkpoint
  saved there. Anything else starts over (batched prefill, ~1500 tok/s).
- Sampling: temperature, top_p, top_k (<= 32), min_p, seed; presence / repetition penalties are
  ignored (with a warning). One request at a time.
- KV cache: fp16, or q8_0 when fp16 does not fit the requested context (`--kv auto|f16|q8`).

## How it gets its speed

- **Weights stay in their GGUF k-quant encoding** (Q4_K / Q5_K / Q6_K / Q8_0), read once per
  cycle regardless of T. Q6_K and Q8_0 are re-laid-out at load time (values unchanged) so every
  lane does aligned 16/8-byte loads; Q6_K blocks carry their scales and `d` in one 8-byte word.
- **dp4a GEMV with shared activations**: activations are int8 (per-16/32 scales, as llama.cpp's
  q8_1), each warp streams 2-4 weight rows at once and reuses every shared-memory activation
  load across them, so T = 4 costs almost the same memory time as T = 1. Weight loads bypass L1
  and are marked L2 evict-first.
- **No launches, few round trips**: activations are re-quantised by the producing phase
  (including the next RMS norm's weight), so consumers only async-copy (`cp.async`) them.
- **Gated-DeltaNet rollback by lazy replay**: the recurrent state is committed only through the
  surely-accepted token; accepted drafts are replayed from a small (k, v, g, beta) ring at the
  start of the next cycle, so rejected drafts cost no state snapshots or extra state traffic.
- **Attention**: split-K flash decoding with tensor-core MMAs (Q.K^T and P.V in fp16 with fp32
  accumulation, online softmax), double-buffered 32-position K/V tiles in shared memory (the next
  tile streams in while the current one is processed: ~5% faster cycles at 30K context).
- **MTP drafting on device**, with the draft lm_head restricted to the 32,768 most frequent
  tokens (`draft_vocab_32k.bin`, built by `tools/vocab_freq` from local code + docs; covers
  ~99.4% of tokens), plus the tokens of the current sequence that are missing from it, added on the
  fly (identifiers of pasted code: +3-8% decode speed with code in context). Verification always
  uses the full vocabulary, so this affects only the acceptance rate, never the output.
- **Sampling** keeps the 32 best logits of each verified row; a draft is accepted with probability
  p(draft) and a rejected one is resampled from the rest of the distribution, which keeps the
  sampled distribution exact.
- **L2 warm-up**: before each barrier, blocks prefetch (evict-last) what their next latency-bound
  phase reads first (DeltaNet state, conv weights, norm weights, RoPE rows, first K/V tile).

## Prefill

A prompt (all but its last token) is processed in chunks of up to 2048 tokens by ordinary kernel
launches (`prefill.cuh`), which leave exactly the state the megakernel expects (trunk and MTP K/V
caches, committed DeltaNet state, conv ring, hidden-state history); the last prompt token then goes
through the megakernel, which produces the first token and the first MTP drafts. The same path
appends to a running conversation (`MKEngine::prefill` on a non-empty sequence, e.g. the next user
turn): it starts at the committed DeltaNet position and also re-feeds the last emitted token and any
accepted drafts not yet folded into the recurrent state; the output is token-identical to
processing the whole conversation as one fresh prompt.

- **GEMMs on fp16 tensor cores straight from the k-quants** (`prefill_gemm.cuh`, ~85% of prefill
  time): 128 x 128 tiles, weights stay in their GGUF blocks and are dequantised to fp16 in
  registers (exact small integers via the `0x6400` trick, then one fp16 scale and min per
  sub-block); activations are fp16 with the elements of each 16-group permuted so that a 4-byte
  weight word is directly an MMA A fragment. Products accumulate in fp16 over k = 128, then in fp32.
  ~100-108 TFLOPS on all four quant types (`bench/mmq_bench.cu`). FFN gate and up share one GEMM
  whose epilogue writes `silu(gate) * up` as the down projection's fp16 input.
- **DeltaNet in chunked (WY) form on tensor cores**: per 32-token chunk, `A = tril(beta K K^T)`
  with the gate decays, `T = (I + A)^-1` (one warp, forward substitution), then
  `U = T beta V - T beta gamma (K S)`, `O = gamma Q S + P U` and `S <- gamma_last S + K^T (decay U)`.
  Value columns are independent, so a block owns (head, 64 columns) and each warp keeps its 128 x 8
  slice of the state in fp32 MMA accumulators, turning it (and K S, U) into fp16 MMA operands
  in registers with `movmatrix`; gates are carried as log g, so tiny or zero gates are exact.
  ~2% of prefill time; 3.5e-4 relative error against a float64 reference of the recurrence.
- **Attention**: causal flash attention over the KV cache on tensor cores.

## Limitations

- Specialised to this model's shapes and quant mix (27B, Q4_K_M as published by unsloth, see
  [The GGUF](#the-gguf)), plus a plain `llama-quantize` Q4_K_M; the loader rejects any other model or mix. Tuned on sm_86 (RTX 3090); open-jet enables it on sm_86, sm_89 and sm_120
  GPUs with 24 GB or more. `make` builds for the installed GPU (`make ARCH=sm_89` to override).
- Sampling supports temperature, top-k (<= 32), top-p and min-p (llama.cpp's order); no presence /
  repetition penalties yet (open-jet's non-thinking preset uses a presence penalty of 1.5).
- Text only; one sequence and one checkpoint: a conversation is reused while it grows (or diverges
  only after the end of the previous prompt); editing an earlier part, e.g. context compaction,
  re-prefills it. No slot save / restore (open-jet's low-memory shell swap falls back to
  re-prompting).

## Files

- `qwen38_mk.cu` / `qwen38_mk.h` - loader, repacking, the megakernel and the `MKEngine` API
- `prefill.cuh` / `prefill_gemm.cuh` - batched prefill kernels and their quantised fp16 GEMM
- `capi.cpp` - C API (engine + libllama tokenizer) for `libqwen38mk.so`
- `server/mk_server.py` - OpenAI-compatible HTTP server
- `main.cu` - CLI (tokenization via libllama)
- `gguf.h` - minimal mmap GGUF reader
- `draft_vocab_32k.bin` - MTP draft-head token subset (int32 ids)
- `tools/vocab_freq.cpp` - builds a draft vocabulary from a list of text files
- `tools/llama_ref.cpp` / `logit_stats.h` - llama.cpp's next-token distributions and the comparison
  with them (KL divergence, perplexity, top-token agreement)
- `bench/` - benchmarks (see Build / run)
