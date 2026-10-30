# OpenJet's Qwen3.8-27B engine

OpenJet serves Qwen3.8-27B on NVIDIA RTX 30, 40 and 50-series GPUs with 24 GB+ (sm_86, sm_89,
sm_120) with its own CUDA engine. It combines persistent megakernel decoding (each decode step is
one kernel launch), on-device MTP speculative verification, and batched tensor-core prefill, behind
an HTTP server that speaks the OpenAI Chat Completions, OpenAI Responses and Anthropic Messages APIs.

This page documents the engine on its own: how to serve it from your application without OpenJet's
agent, its API, how it works, and its measurements. The source lives in `megakernel/`; the binaries,
C API symbols and OpenJet's `llama_backend: megakernel` setting keep that name.

On an RTX 3090 with Qwen3.8-27B Q4_K_M, reasoning decode is **92 vs 60 tok/s** for llama.cpp
with MTP on short prompts and **79 vs 37 tok/s** with 98K tokens in context, and a 98K-token
prompt is read in **82 vs 112 s**. An earlier build measured greedy decode at **107 vs 63 tok/s**
averaged over four prompts, and greedy code generation at **140 vs 73 tok/s**.
These are single-conversation results on that GPU. With four agents sharing it (`--parallel 4`),
each decodes at **64-72 vs 18-20 tok/s** for `llama-server --parallel 4`, and at 52-55 vs 15-16
with 23K tokens of context each. See [Results](#results) for settings, reproduction scripts, and
correctness comparisons.

## Standalone installation

The quickest route is `pipx install open-jet`, then `openjet setup`, which downloads the engine
prebuilt for your GPU, and `openjet serve` (see [Coding tools](../docs/clients.md)). The rest of
this section builds and serves the engine from source without the OpenJet CLI.

Building from source requires Linux, a supported NVIDIA GPU with 24 GB+ VRAM, Python 3.10+, a C++
compiler, CMake, Make, and the CUDA toolkit (`nvcc`: CUDA 12+, or 12.8+ for RTX 50).
Supply a supported Qwen3.8-27B GGUF yourself; see the [model table](#quick-guide).

From a source checkout, build llama.cpp for the tokenizer, install the engine's Python
server, and compile the original Q4_K_M engine:

```sh
git clone https://github.com/l-forster/open-jet.git
cd open-jet
git clone --depth 1 --branch b10246 https://github.com/ggml-org/llama.cpp.git llama.cpp
cmake -S llama.cpp -B llama.cpp/build -DGGML_CUDA=ON -DLLAMA_CURL=OFF
cmake --build llama.cpp/build --config Release -j 2
python3 -m venv .venv-engine
. .venv-engine/bin/activate
python -m pip install -e ./megakernel
make -C megakernel NVCC="$(command -v nvcc)" libqwen38mk.so
openjet-engine --model /absolute/path/to/Qwen3.8-27B-Q4_K_M.gguf --port 18080
```

This installs the local `openjet-engine` distribution with only Jinja2 as a Python runtime
dependency. The editable install keeps the server beside the compiled library and bundled
draft vocabulary. Keep the checkout and llama.cpp build in place. This is a source-install
entry point; it does not require a separately published PyPI package or the OpenJet CLI.
`openjet-engine --help` lists the context, sampling-draft, library and KV-cache options.
You can also launch it with `python -m megakernel`.

For UD-Q4_K_XL, replace the `make` and server commands with:

```sh
make -C megakernel NVCC="$(command -v nvcc)" -j1 libqwen38mk_mixed.so
openjet-engine --model /absolute/path/to/Qwen3.8-27B-UD-Q4_K_XL.gguf \
  --lib "$PWD/megakernel/libqwen38mk_mixed.so" --port 18080
```

The mixed build used roughly 11 GiB of system RAM in testing; 32 GB is recommended
for build headroom. See [Build / run](#build--run) for manual CLI builds and quant details.

## HTTP API

The server listens on `127.0.0.1:18080` by default. Use `http://127.0.0.1:18080/v1` as the base
URL for OpenAI clients and `http://127.0.0.1:18080` for Anthropic clients.

```sh
curl http://127.0.0.1:18080/health
curl http://127.0.0.1:18080/v1/models
curl http://127.0.0.1:18080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen3.8-27B-Q4_K_M.gguf","messages":[{"role":"user","content":"Write quicksort in Python."}],"max_tokens":512,"temperature":0.6,"top_k":20,"stream":false}'
curl http://127.0.0.1:18080/v1/messages \
  -H 'Content-Type: application/json' \
  -d '{"model":"any","max_tokens":512,"messages":[{"role":"user","content":"Write quicksort in Python."}]}'
```

The server serves one model and ignores the `model` field; `/v1/models` returns its ID.

| Endpoint | Behavior |
|---|---|
| `GET /health` | Readiness after the model has loaded |
| `GET /v1/models` | Loaded model ID |
| `POST /v1/chat/completions` | OpenAI Chat Completions; `stream: true` returns SSE ending in `[DONE]` |
| `POST /v1/responses` | OpenAI Responses (Codex CLI); streaming or not |
| `POST /v1/messages` | Anthropic Messages (Claude Code); streaming or not |
| `POST /v1/messages/count_tokens` | Anthropic token count, without waiting for a running request |

Chat completions include `choices`, `usage`, and engine `timings`. Thinking output uses
`reasoning_content`; tool calls use structured `tool_calls`. Requests accept `tools`,
`chat_template_kwargs`, `temperature`, `top_p`, `top_k` (at most 32), `min_p`, and `seed`.

The Responses and Messages endpoints translate the request to a chat completion and the result
back: reasoning becomes a `reasoning` item or a `thinking` block, tool calls become `function_call`
(or `custom_tool_call`) items or `tool_use` blocks, and cached prompt tokens are reported as
`cached_tokens` / `cache_read_input_tokens`. Responses' `reasoning.effort` selects the chat
template's thinking level, and Messages' `thinking: {"type": "disabled"}` turns thinking off. A
prompt longer than the context returns Anthropic's `prompt is too long` error on `/v1/messages`, so
Claude Code compacts and retries. See [Coding tools](../docs/clients.md) for client setup and the
limits of each API.

One request runs at a time unless the server is started with `--parallel N`
([Parallel serving](#parallel-serving)). Text only; presence and repetition penalties are ignored with
a warning. See [Server](#server) and [Limitations](#limitations) for conversation reuse
and other API constraints.

## Use with OpenJet

The full `open-jet` package also installs `openjet-engine`. Run `openjet setup` to download
a model and the matching engine library automatically, then run `openjet` for the coding agent or
`openjet serve` for other tools. OpenJet selects this engine when supported and falls back to
llama.cpp otherwise; see [runtime configuration](../docs/runtimes/llama-cpp.md).

To serve a library installed by setup directly, pass `openjet-engine --lib /path/to/libqwen38mk.so`
(or the mixed library) together with `--model`. Standalone serving loads the requested engine
and reports errors directly; automatic runtime selection belongs to OpenJet setup.

## Quick guide

Which model file runs on it (details in [The GGUF](#the-gguf)):

| Model file | Runs on |
|---|---|
| unsloth's original Qwen3.8-27B `Q4_K_M`, the one the catalog downloads | OpenJet |
| a Qwen3.8-27B `Q4_K_M` from lmstudio-community, mradermacher, Abiray or KikoCis | OpenJet, semi compatible |
| unsloth Qwen3.8-27B `UD-Q4_K_XL` | OpenJet's separate mixed-quant build |
| any other quant (other unsloth `UD-*`, bartowski, GSQ-RCO, ...) | llama.cpp |

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

**Current engine, reasoning mode** (29 Oct 2026, `bench/run_compare.sh`; thinking on, xhigh effort,
temperature 1.0, top_p 0.95, top_k 20; 5 coding prompts x 4 seeds and one 98K-token prompt of this
repository's source x 4 seeds, up to 1024 tokens each; 106,496-token context, q8_0 KV cache for both;
llama.cpp b11982 with `--spec-type draft-mtp --spec-draft-n-max 2`):

| | llama.cpp | OpenJet (default, `pk_attn8`) | OpenJet, `MK_ATTN_S8=0` (`pk_attn2`) |
|---|---|---|---|
| decode, short prompts | 59.9 tok/s | **91.8 tok/s** (1.53x) | 91.5 tok/s |
| decode, 98K tokens in context | 36.5 tok/s | **79.3 tok/s** (2.17x) | 85.6 tok/s |
| reading the 98K-token prompt (cold) | 112 s (876 tok/s) | **82 s** (1,198 tok/s) | 96 s (1,018 tok/s) |

`pk_attn8` reads long prompts faster; `pk_attn2` decodes faster at depth.

**Reasoning mode, earlier build** (same settings; 5 coding prompts x 2 seeds):

| | llama.cpp | OpenJet | |
|---|---|---|---|
| decode through the HTTP servers (`/v1/chat/completions`) | 50.9 tok/s | **78.0 tok/s** | **1.53x** |
| decode, CLI (`bench/think_bench.py`, `-d 4`) | 50.7 tok/s | 77.4 tok/s | 1.53x |

**Greedy decode, 4 chat prompts x 512 tokens** (`bench/code_bench.py`):

| prompt | llama.cpp | OpenJet `-d 3` | OpenJet `-d 4` |
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

| context | decode llama.cpp | decode OpenJet | prompt llama.cpp | prompt OpenJet |
|---|---|---|---|---|
| 1.8K | 57.4 | 84.8 | 1050 | 1595 |
| 3.6K | 56.3 | 95.3 | 1124 | 1595 |
| 7.8K | 54.8 | 85.8 | 1140 | 1533 |
| 15.9K | 55.7 | 86.5 | 1126 | 1466 |
| 29.5K | 52.3 | 72.7 | 1062 | 1236 |

(tok/s; measured before the sequence-token draft vocabulary and double-buffered attention below,
which add 1-8% to the megakernel's decode here.)

**Pasting a file into the next turn of a conversation** (prompt processing time):

| new tokens | llama.cpp (prompt cache) | OpenJet |
|---|---|---|
| 2.7K | 2.41 s | 1.68 s |
| 12K | 10.45 s | 8.10 s |

With the q8_0 KV cache (chosen automatically for contexts above ~80K tokens, e.g. open-jet's 100K)
decode is ~1-2% and prefill 2-7% slower than with fp16.

**Four agents at once** ([Parallel serving](#parallel-serving); `bench/parallel_bench.py`, coding
tasks, 512 tokens, temperature 0.6, thinking off). llama.cpp as `llama-server -ngl 99 -fa on
-c 131072 --parallel 4 -ctk q8_0 -ctv q8_0 --jinja`, the megakernel as `mk_server.py -c 0
--parallel 4`; same GPU, back to back:

| per agent | llama.cpp | megakernel |
|---|---|---|
| decode, short prompts | 18-20 tok/s | **64-72 tok/s** |
| decode, 23K tokens of context each | 15-16 tok/s | **52-55 tok/s** |
| first token of the next request on those contexts | 13.4 s | **0.2-0.5 s** |
| a 20K-token prompt arrives: its first token | 20.9 s | 20.4 s |
| ... an agent decoding meanwhile | 0.6 tok/s | **9.5 tok/s** |

Context capacity is about equal: llama.cpp was given 4 x 32K positions, the megakernel's pool holds
132K.

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

Requires CUDA 12+ (12.8+ for RTX 50 series; built with 13.2), an sm_86, sm_89 or sm_120 GPU with 24 GB
(or, experimentally, sm_75: see [Turing](#turing-rtx-8000)), and the repo's llama.cpp build (only
`libllama`, for tokenization, loaded vocab-only). `openjet setup` reads the selected model's GGUF
header and builds only the library it needs (the source ships in the open-jet wheel):

- `libqwen38mk.so` / `qwen38_mk.cu`: the original Q4_K_M engine, including its existing
  conversions for plain llama-quantize Q4_K_M. Its original quant-specialized kernels are retained.
- `libqwen38mk_mixed.so` / `qwen38_mixed.cu`: additional quant kernels for UD-Q4_K_XL and
  supported mixed tensor recipes. This is compiled only when the selected model needs it.

Selection checks tensor roles and types, rather than assuming the quant recipe from the filename.
Both libraries and their architecture/build records are retained independently when switching models.
The default `make` target builds only the original engine.

The mixed engine has a heavier compilation: in an RTX 3090 / CUDA 13.2 test, compilation consumed
roughly 11 GiB of available system RAM and succeeded with 16 GiB assigned to WSL. This is observed
usage, not a proven minimum; 32 GB system RAM is recommended for headroom when building the mixed
engine. GPU VRAM is separate. Existing Q4_K_M users do not compile the mixed engine.

Building the original engine by hand:

```sh
cd megakernel
make        # qwen38_mk (CLI), libqwen38mk.so (server library), tools/vocab_freq; ~2 min
./qwen38_mk -m ../models/Qwen3.8-27B-Q4_K_M.gguf -p "Write quicksort in C" -n 512 \
            -d 4 --draft-vocab draft_vocab_32k.bin --think --temp 1.0 --top-p 0.95 --top-k 20
```

To build and serve the mixed engine manually:

```sh
make -j1 qwen38_mixed libqwen38mk_mixed.so
python server/mk_server.py -m /path/to/Qwen3.8-27B-UD-Q4_K_XL.gguf --lib ./libqwen38mk_mixed.so
```

The original CLI remains Q4_K_M-only; use `./qwen38_mixed` or the server with `--lib` for the mixed engine.
Benchmark scripts accept `--model /path/to/model.gguf --mk-bin ./qwen38_mixed` for this executable.
`openjet` passes the correct library automatically. The mixed loader accepts Q4_K, Q5_K, Q6_K,
Q8_0 and IQ4_XS projections, and converts Q3_K / IQ3_S to Q6_K and IQ4_NL to Q8_0.
Both engines still require Q4_K token embeddings and a Q6_K output head.

CLI flags: `-d` drafts per cycle (0-4; with 4 the depth drops to 3 for a cycle after a rejected
draft), `-c` KV capacity, `--kv f16|q8|auto` (KV cache format), `--think [xhigh|medium|low]`
(reasoning mode; the chat template's default effort is xhigh), `--temp` / `--top-p` / `--top-k` /
`--min-p` / `--seed` (sampling; default greedy), `--raw` (no chat template), `--ids`,
`--prompt-ids f` (prompt as token ids), `--turn2 text` / `--turn2-file f` (a second user turn
appended to the conversation), `--score ids.txt` (teacher-forced agreement), `--kld ref` (KL
divergence, perplexity and agreement against a reference file, see Correctness), `MK_PROFILE=1`
(per-phase timing, including a "min-composed" estimate that filters out time-slicing by other GPU
users), `MK_PF_PROFILE=1` (per-kernel prefill timing), `MK_PREFILL=mk` (prefill through the decode
megakernel instead), `MK_PF_CAP=1024` (prefill chunks of 1024 tokens; default 2048,
rounded down to a multiple of 128 and clamped to 128–2048), `MK_EMBD_DEVICE=1` (keep
`token_embd` in GPU memory), `MK_DRAFT_SAMPLING=greedy` (at temperature > 0, take each MTP draft as
the draft head's argmax and accept it with p(d), instead of the default: sample it from the draft
head's distribution and verify by rejection sampling, `min(1, p/q)` with residual `max(0, p-q)`;
both leave the output distribution unchanged, the default accepts more drafts).

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
kernels. The original engine refuses any other mix, naming the first tensor that does not match.
OpenJet selects the mixed engine for supported extended recipes and otherwise starts `llama-server`. Which files that covers is in the [quick guide](#quick-guide).

To build a file with unsloth's mix from the BF16 GGUF (unsloth's imatrix optional):

```sh
llama-quantize --imatrix imatrix_unsloth.gguf \
    --tensor-type ssm_out=q5_k --tensor-type ssm_alpha=f32 --tensor-type ssm_beta=f32 \
    --tensor-type nextn.eh_proj=q8_0 \
    Qwen3.8-27B-BF16-00001-of-00002.gguf Qwen3.8-27B-Q4_K_M.gguf Q4_K_M
```

Benchmarks: `bench/think_bench.py` (reasoning mode, both engines), `bench/code_bench.py` (greedy
decode, context sweep, second turn), `bench/bench.py` (decode on the 4 chat prompts),
`bench/parallel_bench.py` (several agents on any OpenAI-compatible server);
`bench/*_bench.cu` are GEMV / GEMM microbenchmarks.

## Server

`server/mk_server.py` serves the engine (through `libqwen38mk.so`, the C API in `capi.cpp`) with
OpenAI's `/v1/chat/completions` and `/v1/responses` and Anthropic's `/v1/messages`, streaming or not,
plus `/health` and `/v1/models` (see [HTTP API](#http-api)).
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
  ignored (with a warning). One request at a time, or several with `--parallel N` (below).
- KV cache: fp16, or q8_0 when fp16 does not fit the requested context (`--kv auto|f16|q8`).
- Both engines keep the ~0.72 GB token embedding table in mapped pinned system RAM by default,
  freeing about 680 MiB of GPU VRAM. `MK_EMBD_DEVICE=1` restores GPU residency.
- `MK_PF_CAP=1024` halves prefill scratch, freeing about another 325 MiB of VRAM.
  The default is 2048, and 1024 under `--parallel`, where the memory goes to context (a 9.1K-token
  prompt measured 1248-1508 tok/s at either cap). Smaller caps also limit scoring/logit tiles to
  the scratch allocation. These options apply to both quant engines.
- The MTP layer's matrices stored wider than Q4_K are re-quantized to Q4_K at load (`attn_v` and
  `ffn_down` in unsloth's Q4_K_M; the whole layer and `nextn.eh_proj` in UD-Q4_K_XL). Only the drafts
  change: the main model verifies every draft, so greedy output is unchanged and sampling keeps its
  distribution; acceptance can move. Each MTP pass streams fewer bytes. `MK_MTP_Q4=0` keeps the
  stored types.

### Parallel serving

`--agents N` serves up to 8 conversations at once (two MTP drafts each per step up to 5, one
above), e.g. several coding agents on one GPU; it is `--parallel N -c 0`:

```sh
openjet-engine --model /absolute/path/to/Qwen3.8-27B-Q4_K_M.gguf --agents 4
```

- Each conversation lives in a slot. A request goes to the slot already holding its conversation,
  which then processes only what is new; otherwise to the free slot that loses least (an empty one
  before another conversation's cache).
- `-c 0` gives the slots all the memory left: their K/V (q8_0) is one pool of pages that any slot
  draws on as its context grows, 132K positions on a 24 GB RTX 3090 (about 33K per agent with 4).
- A system message plus tools of 1024 tokens or more is stored once and shared by the
  conversations that start with it (`--no-shared-prefix` turns this off).
- The active slots decode together: one pass over the weights for all their rows (each slot's
  current token and its MTP / prompt-lookup drafts).
- A new prompt rides along with those steps: each pass holds the step's rows plus the next chunk
  of the prompt, 256 rows in all (`MK_PF_MIX`; `MK_PF_MIX=0` instead prefills chunks of
  `MK_PF_CHUNK`, 512, between steps). The other agents keep decoding while it is processed, with
  no cost to the prompt's own speed, and its state is bit-identical to prefilling it alone.
- Q4_K_M engine only; the mixed-quant library serves one conversation at a time.

## KV cache presets (MK_KV)

The KV cache is q8_0 (or fp16, `--kv`). `MK_KV` stores it in fewer bits, per attention layer and for K and V
separately, and `MK_KV_WIN` keeps the most recent positions at q8_0, where most of a low-bit cache's error would come
from (attention weight concentrates on recent positions). Single-stream engine, both quant engines, sm_80+.

| preset | environment | K / V bits | cache bytes per position |
|---|---|---|---|
| default | (none) | 8 / 8 | 36,992 |
| D | `MK_KV='K*=6,V*=5' MK_KV_WIN=2048` | 6 / 5 | 26,112 |
| F | `MK_KV='K*=5r,V*=5' MK_KV_WIN=2048` | 5 / 5, K Hadamard-rotated | 23,936 |

plus 114 MiB of q8_0 window rings for `MK_KV_WIN=2048`.

`openjet setup --long-context` (optionally with `--context-tokens N`) configures OpenJet to start the megakernel with
preset F and a context past the q8_0 limit; by hand:

```sh
MK_KV='K*=5r,V*=5' MK_KV_WIN=2048 python3 server/mk_server.py -m ../models/Qwen3.8-27B-Q4_K_M.gguf -c 131072
```

On a 24 GB RTX 3090 (Q4_K_M, about 1 GiB used by the desktop) preset F holds 250,000 positions at full decode speed:
23,962 of 24,576 MiB peak in a short conversation, 24,094 MiB with 229,353 tokens in context; about 100 tok/s decode
in a short conversation and 68 tok/s with the 229,353 tokens (the prompt read at 755 tok/s). Setup offers about
234,000, which leaves about 1 GiB.
The room comes from device memory the engine no longer keeps: buffers from the CUDA memory pool instead of one
`cudaMalloc` each, a 624-byte `mk_cycle` stack frame instead of 2,496 (the driver reserves it for every thread slot),
the draft head reading lm_head rows through its id list instead of a copy, and with `MK_KV` the prompt checkpoint in
host memory. The low-bit KV cache, its recent window and these memory savings are Benjamin Cutler's work
([qwen38-megakernel](https://github.com/cutlerbenjamin1-cmd/qwen38-megakernel)).

- `MK_KV`: comma-separated `<K|V|*><slot>=<bits>[r]`. Bits 4 to 8 (8 = q8_0); slot 0-15 = the attention layers 3, 7,
  .., 63, 16 = the MTP layer, empty or `*` = all of them; `r` = stored Hadamard-rotated (a 256-point Walsh-Hadamard
  transform, orthonormal: for K the queries get the same rotation, a rotated V's attention output is rotated back). The
  formats are q4_0-style grids of 32 values with one fp16 scale; 5 to 7 bits store the 4-bit nibbles plus 1 to 3 high
  bits. `D'` = `K*=6,V*=5,*16=8` keeps the MTP layer at q8_0 (+640 bytes per position; its drafts are accepted
  more often).
- `MK_KV_WIN=W`: every low-bit layer also keeps its last W to W + 640 positions in a q8_0 ring. The window start
  moves in 640-position steps, like the attention split, so greedy output stays independent of the draft depth. A
  rewind (checkpoint restore, truncate) refills the window rows the abandoned tokens overwrote from the low-bit cache.
- A non-empty `MK_KV` selects the q8_0 cache in `--kv auto`; `--kv f16` with `MK_KV` is an error, as are `--parallel` /
  `--agents` (for now), Turing and `MK_ATTN=1`.
- The accuracy check is the usual one (see Correctness) with the variables set:
  `MK_KV='K*=5r,V*=5' MK_KV_WIN=2048 ./qwen38_mk -m model.gguf --kld llama.ref --score-path mixed --kv q8`.

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
  accumulation, online softmax). The split is a function of the context length in 640-position steps
  (the host never lets one cycle's rows straddle a step), so greedy output does not depend on the
  draft depth. Past 640 positions the q8_0 cache runs `attn_core_q8`, a three-deep software pipeline
  with one block barrier per 32-position tile: warps 0-3 compute Q.K^T, 4-11 P.V, 12-15 the online
  softmax, K / V stream through separate `cp.async` rings and are converted to fp16 MMA operands in
  registers; the partials are bit-identical to the simple kernel's (`MK_DATTN=0` selects it). Low-bit
  `MK_KV` layers run the same kernel with their tiles expanded to int8 in shared memory.
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
- **Attention**: causal flash attention over the KV cache on tensor cores. On sm_80+ the engine's own
  cache runs `pk_attn8` (`prefill_attn.cuh`): Q.K^T as int8 tensor-core MMAs on the q8_0 (or expanded
  low-bit) keys with per-32 Q scales, P.V in fp16, 12 warps per (16 queries, kv head), and the last
  partial wave of work split along the keys and merged (`pk_attn2_comb`). Not bit-exact with `pk_attn`;
  KL against llama.cpp is unchanged. `MK_ATTN_S8=0` selects the fp16 `pk_attn2`, `MK_ATTN=1` the
  original `pk_attn` (also used for a parallel slot's paged cache and on Turing); `MK_ATTN_SPLIT=0`
  and `MK_ATTN_PV16=0` turn off the tail split and fp16 P.V accumulation.
- **Small chunks** (a prompt riding along with parallel steps, or a prompt's tail): a GEMM whose
  128-row tiles would leave most SMs idle (5120 output rows: 40 CTAs) splits K in two, the fp32
  halves summed in a fixed order, and attention of few queries over a long context splits the
  cached positions over more blocks, merging the partial softmax results in order. Both are
  1.6-2x faster at 128 tokens; KL against llama.cpp is unchanged.

## Turing (RTX 8000)

Experimental, contributed and tested on a 48 GB Quadro RTX 8000. Only the original Q4_K_M engine
has a Turing path; the mixed engine needs sm_80+, so open-jet serves UD-Q4_K_XL with llama.cpp on
Turing. `make ARCH=sm_75` (or setup, on a Turing GPU) defines `QWEN_MK_SM75`, which swaps in
[`cuda_compat.cuh`](cuda_compat.cuh): synchronous copies instead of `cp.async`, k16 tensor-core MMAs
split into two k8 ones, no L2 cache-policy hints, and tiles shrunk to Turing's 64 KiB of shared
memory. Builds for other GPUs are unchanged. CUDA 12 and 13 both compile sm_75. The int8 MMAs of
the parallel path are built from m8n8k16 there (compiled, not yet run on Turing hardware).

| RTX 8000, Qwen3.8-27B Q4_K_M | llama.cpp (MTP 2) | OpenJet |
|---|---|---|
| decode, reasoning mode | 32.9 tok/s | **54.1 tok/s** (1.65x) |
| prompt processing, 2-8K tokens | 515-634 tok/s | **597-731 tok/s** (1.10-1.16x) |

The two engines ran on two RTX 8000 cards in one machine. Method and raw samples:
[benchmarking.md](../docs/benchmarking.md#matched-reasoning-comparison-on-two-gpus),
[decode](../docs/benchmarks/rtx8000-turing.json),
[prefill](../docs/benchmarks/rtx8000-turing-prefill.json). There is no Turing KLD or long-context
validation yet. `bench/turing_check.cu` (`make bench/turing_check`) checks the Turing MMA, GEMM
and attention paths against references without a model. On multi-GPU machines, pin the GPU with
`CUDA_VISIBLE_DEVICES=<uuid>` for both `openjet setup` and `openjet`.

## Limitations

- Specialised to Qwen3.8-27B shapes and the supported original or mixed quant recipes
  described in [The GGUF](#the-gguf); the loader rejects unsupported models or tensor mixes. Tuned on sm_86 (RTX 3090); open-jet enables it on sm_86, sm_89 and sm_120
  GPUs with 24 GB or more, and experimentally on sm_75. `make` builds for the installed GPU (`make ARCH=sm_89` to override).
- Sampling supports temperature, top-k (<= 32), top-p and min-p (llama.cpp's order); no presence /
  repetition penalties yet (open-jet's non-thinking preset uses a presence penalty of 1.5).
- Text only; one sequence and one checkpoint: a conversation is reused while it grows (or diverges
  only after the end of the previous prompt); editing an earlier part, e.g. context compaction,
  re-prefills it. No slot save / restore (open-jet's low-memory shell swap falls back to
  re-prompting).
- `--parallel`: at most 8 conversations (16 rows per step), Q4_K_M engine only. With 4 agents on
  a 24 GB card, each has about 33K tokens of context.
- `MK_KV` low-bit caches: single conversation, sm_80+ (see [KV cache presets](#kv-cache-presets-mk_kv)).

## Files

- `qwen38_mk.cu` / `qwen38_mk.h` - loader, repacking, the megakernel and the `MKEngine` API
- `prefill.cuh` / `prefill_gemm.cuh` - batched prefill kernels and their quantised fp16 GEMM
- `prefill_attn.cuh` - prefill attention `pk_attn2` / `pk_attn8` (sm_80+)
- `kv_lowbit.cuh` / `kv_spec.h` - `MK_KV` cache formats, the window ring, their parsing and allocation
- `capi.cpp` - C API (engine + libllama tokenizer) for `libqwen38mk.so`
- `server/mk_server.py` - HTTP server: OpenAI Chat Completions and Responses, Anthropic Messages
- `main.cu` - CLI (tokenization via libllama)
- `gguf.h` - minimal mmap GGUF reader
- `draft_vocab_32k.bin` - MTP draft-head token subset (int32 ids)
- `tools/vocab_freq.cpp` - builds a draft vocabulary from a list of text files
- `tools/llama_ref.cpp` / `logit_stats.h` - llama.cpp's next-token distributions and the comparison
  with them (KL divergence, perplexity, top-token agreement)
- `bench/` - benchmarks (see Build / run)
