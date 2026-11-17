# qwen38-megakernel

**Qwen3.8-27B on a single 24 GB GPU, decoding 1.8-2x faster than llama.cpp.**

A CUDA engine and server built for one model. On an RTX 3090 it decodes 166 tok/s on short prompts and 116 tok/s with
100k tokens of context, and fits the model's full 262k context on the card: older K / V drop to 5-6 bits (K
Hadamard-rotated in preset F) while the newest 2,048 positions stay in a q8_0 ring, where attention puts most of its
weight. It speaks the OpenAI, Anthropic and OpenAI Responses APIs, so coding agents and chat UIs connect to it
directly.

**And it's exact, bit for bit.** Greedy output is the same at 0 drafts or 7, a cached, edited or resumed conversation
answers exactly like the same request sent fresh, and the Windows and Linux builds give identical tokens. Speculation
and caching never change the answer ([how](#exact-output)).

[Quick start](#quick-start) | [Benchmarks](docs/BENCHMARKS.md) | [Supported clients](docs/CLIENTS.md) |
[Configuration](docs/CONFIGURATION.md) | [Releases](https://github.com/cutlerbenjamin1-cmd/qwen38-megakernel/releases)

Built on the megakernel from [open-jet](https://github.com/L-Forster/open-jet) by Louis Forster
([credits](#credits-and-license)).

## Speed

Against llama.cpp's `llama-server` (b12024, tuned, q8_0 KV cache, at 4 MTP drafts, its fastest setting), same session,
same client, prompts and model file. RTX 3090, unsloth's UD-Q4_K_XL, 120k context, release v2026.11.05 at its default 7
drafts, with the q8_0 KV cache and with [preset F](#the-full-262k-context-on-24-gb).

| | llama.cpp | qwen38-megakernel, q8_0 | qwen38-megakernel, preset F |
|---|---|---|---|
| greedy decode, short prompts | 87.7 tok/s | **166.5 tok/s** (1.9x) | **160.7 tok/s** (1.8x) |
| sampled decode, thinking on | 73.2 tok/s | **146.9 tok/s** (2.0x) | **147.4 tok/s** (2.0x) |
| decode at 38k context | 66.2 tok/s | **127.8 tok/s** (1.9x) | **111.3 tok/s** (1.7x) |
| decode at 101.6k context | 55.2 tok/s | **116.4 tok/s** (2.1x) | **123.2 tok/s** (2.2x) |
| read a 101.6k-token prompt | 95.6 s | **68.5 s** (1.4x) | **70.7 s** (1.4x) |
| first token, short prompt | 117 ms | **61 ms** (1.9x) | **61 ms** (1.9x) |
| edit an earlier message in an 11.7k-token chat | 302 ms | **69 ms** (4.4x) | **71 ms** (4.3x) |
| 282 benchmark questions (GSM8K, MMLU-Pro, ARC-C, MATH-500, HumanEval, NIAH, LongBench v2) | 229 right, 2,534 s | **230 right, 1,563 s** (1.6x) | **229 right, 1,596 s** (1.6x) |
| VRAM, idle at 120k context | 22.9 GiB | **21.2 GiB** | **19.8 GiB** |

With drafting off on both sides it's still 11% faster on short prompts and 45% faster at 101.6k context, so the gain
isn't only speculation. Preset F's replies differ from q8_0's, so its decode rows aren't like-for-like. llama.cpp's
message-edit time moves between sessions (247 ms on 2026-11-03). Token-weighted rates, the no-draft table and the
method are in [docs/BENCHMARKS.md](docs/BENCHMARKS.md). Every row is one session.

## What you get

- **Speculative decoding with the model's own MTP head**, up to 7 drafts per step (15 with the opt-in
  [deep-pass build](docs/EXPERIMENTAL.md#the-deep-pass), q8_0 KV cache only), verified on the GPU.
- **[Exact output](#exact-output):** greedy output is identical at any draft count, from a cached conversation or a
  fresh one, on Windows or Linux.
- **[The full 262k context on 24 GB](#the-full-262k-context-on-24-gb)** with near-lossless low-bit KV presets.
- **[A conversation cache](#picks-up-where-you-left-off)** that survives follow-ups, edits, new agent sessions and
  switching between conversations.
- **[OpenAI, Anthropic and Responses APIs](#works-with-your-coding-agent)**, tested with 11 coding agents and chat UIs.
- **Images and video**, encoded on the CPU, so the encoder takes no VRAM.
- **Several agents at once:** `--agents 4` serves 424 tok/s in total on short prompts (Q4_K_M), 360 with ~23k tokens
  of context each, against 137 and 97 for llama.cpp's `llama-server --parallel 4` measured on 2026-11-02, sampled
  ([parallel agents](docs/BENCHMARKS.md#parallel-agents)).
- **Prebuilt packages** for Windows and Linux.
- **[In active development](docs/EXPERIMENTAL.md):** the [deep pass](docs/EXPERIMENTAL.md#the-deep-pass) (up to 15 drafts per step, q8_0 KV
  cache only for now; low-bit KV presets with it are on the list), int8 decode attention
  ([`-DMK_ATT_I8=1`](docs/CONFIGURATION.md#engine--d-flags), q8_0 KV cache) and
  [L2 prefetch budgets](docs/EXPERIMENTAL.md#l2-prefetch-budgets) you can set at run time, to tune a GPU other than the
  RTX 3090. All are opt-in source builds, not in the regular packages; the L2 build also has a
  [Windows pre-release](https://github.com/cutlerbenjamin1-cmd/qwen38-megakernel/releases/tag/v2026.11.04-l2tune).

## Quick start

**Prebuilt packages** for Windows and Linux are on the
[releases page](https://github.com/cutlerbenjamin1-cmd/qwen38-megakernel/releases), built for RTX 30-series GPUs (sm_86;
RTX 40-series cards can run that code but are untested). Unpack one, `pip install -r requirements.txt`, download the
model as below and run `scripts/serve.sh` or `scripts\serve.bat`. What they need:
[docs/BUILDING.md](docs/BUILDING.md#prebuilt-packages).

**From source on Linux or WSL2** (CUDA 13.2, llama.cpp next to this repository for its tokenizer and image library):

```sh
git clone https://github.com/ggml-org/llama.cpp ../llama.cpp && git -C ../llama.cpp checkout b10246
cmake -S ../llama.cpp -B ../llama.cpp/build -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON -DGGML_CUDA=ON
cmake --build ../llama.cpp/build -j
make -j2                          # the served configuration: about 14 minutes, up to ~21 GiB of RAM in ptxas
pip install -r requirements.txt
hf download unsloth/Qwen3.8-27B-GGUF Qwen3.8-27B-UD-Q4_K_XL.gguf mmproj-BF16.gguf --local-dir ~/models
scripts/serve.sh                  # http://127.0.0.1:8080: 120k context, 7 drafts, exact reuse, images on
```

**From source on Windows 10 / 11** (Visual Studio 2022 Build Tools, CUDA 13.2, CMake + Ninja, Python), in PowerShell
from an "x64 Native Tools" prompt, then the server from cmd:

```powershell
git clone https://github.com/ggml-org/llama.cpp ..\llama.cpp; git -C ..\llama.cpp checkout b10246
cmake -S ..\llama.cpp -B ..\llama.cpp\build -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON -DGGML_CUDA=OFF -DLLAMA_CURL=OFF
cmake --build ..\llama.cpp\build --target llama mtmd
.\build.ps1                       # uses $env:CUDA_PATH, or -Cuda <toolkit folder>
python -m venv .venv; .venv\Scripts\pip install -r requirements.txt
hf download unsloth/Qwen3.8-27B-GGUF Qwen3.8-27B-UD-Q4_K_XL.gguf mmproj-BF16.gguf --local-dir D:\models
```

```bat
set MK_MODEL=D:\models\Qwen3.8-27B-UD-Q4_K_XL.gguf
set MK_PYTHON=%CD%\.venv\Scripts\python.exe
scripts\serve.bat
```

Then ask it something:

```sh
curl -s http://127.0.0.1:8080/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "messages": [{"role": "user", "content": "Write quicksort in C."}], "max_tokens": 512}'
```

The launchers take the port and the context size as arguments (`scripts/serve.sh 8080 64000`) and everything else from
environment variables (`MK_HOST=0.0.0.0` for the LAN, `MK_KV` / `MK_KV_WIN` for preset D, ...). Building in detail:
[docs/BUILDING.md](docs/BUILDING.md). Every setting: [docs/CONFIGURATION.md](docs/CONFIGURATION.md).

## Exact output

Speculative decoding is lossless on paper, but verifying several tokens in one pass rounds differently from verifying
one, so in practice greedy output moves with drafting: on the 282 benchmark questions, llama.cpp b12024's greedy answer
differs from its own without drafts on 113 at 4 drafts and on 113 at 7, and between 4 and 7 on 9 (8 of them reproducibly
from the depth). A reused cache or another compiler can drift the same way. This engine is built so none of them change
the output. All 282 of its answers are the same at 7 drafts as with none:

- **Drafting never changes the answer.** Greedy token ids are identical at 0, 4 and 7 drafts, over about 2,000 tokens of
  output and under every draft policy. Sampled output keeps the distribution of sampling without drafts (chi-square
  tests of a CPU replica of the loop and a PIT test on the engine, 2026-10-17; not part of the gate).
- **A cached conversation answers like a fresh one.** With exact reuse (the launchers' default), follow-ups, edits,
  resumed sessions and conversations restored from host memory all continue bit for bit like the same request sent
  fresh, greedy or sampled with the same seed.
- **Windows and Linux agree bit for bit.** The MSVC and g++ builds give the same token ids, server replies and
  image-encoder rows.

So an agent's run can be reproduced, and an odd reply is never the speculation's or the cache's doing. These hold on q8_0
and on the presets, each against itself. Every release is gated on the drafting and cache checks, and you can run
them on your own build: `scripts/exactness.py` for the exact-output checks, `scripts/gate.sh` for those and the rest of
the GPU checks ([checking a build](docs/BUILDING.md#checking-a-build)). Windows against Linux is `scripts/exactness.py`
run on both builds and compared with `--compare`, by hand.

Not bit-identical, by design:
- **llama.cpp.** Its arithmetic is ordered and rounded differently. Against llama.cpp's own token-by-token
  distributions (13,299 positions of code and prose), this engine picks the same top token at 98.74% of positions with
  a mean KL divergence of 0.00119 nats. llama.cpp's own batched mode scores 98.49% / 0.00166 against the same reference.
- **The low-bit KV presets** ([below](#the-full-262k-context-on-24-gb)): near-lossless and opt-in, except that
  `scripts\serve.bat` uses preset F above 240k context unless `MK_KV` is set.
- **`MK_EXACT_REUSE=0`**, which makes follow-up turns faster by appending to the previous reply instead of reading it
  again.
- **Sampled replies beside other requests with `--agents`**: they keep their distribution but not their tokens. Greedy
  replies are the same beside others as alone.

Evidence and the tests behind each line: [docs/BENCHMARKS.md](docs/BENCHMARKS.md#exactness).

## The full 262k context on 24 GB

Qwen3.8-27B reads up to 262,144 tokens, but a q8_0 KV cache that long doesn't fit next to the weights on a 24 GB card.
The low-bit presets store older K / V at 5-6 bits and keep the newest 2,048 positions at 8 bits, because that's where
attention puts most of its weight and where low-bit storage costs accuracy.

| | q8_0 (default) | preset D | preset F |
|---|---|---|---|
| K / V bits (newest 2k positions at 8) | 8 / 8 | 6 / 5 | 5, rotated / 5 |
| VRAM, idle server at 120k context | 21.2 GiB | 20.1 GiB | 19.8 GiB |
| full 262,144-token context on 24 GB | no | yes | yes |
| KL against llama.cpp at 32k / 100k context | 0.00219 / 0.00217 | 0.00236 / 0.00226 | 0.00242 / 0.00218 |
| 282 benchmark questions | 230 right | 230 right | 229 right |
| read a 101.6k-token prompt | 68.5 s | 71.1 s | 70.7 s |

At the full context, preset F reads a 260,896-token prompt in about 4.6 minutes and decodes at that depth at 78 tok/s,
and 4 of 4 needle-in-a-haystack questions came back right. A conversation shorter than the window never leaves 8 bits, so
preset D's output there is q8_0's.

```sh
MK_KV='K*=6,V*=5' MK_KV_WIN=2048 scripts/serve.sh 8080 262144     # preset D, the full context
```

**Recommendation:** for long-context work, or to free a gigabyte for something else, turn D on. The measured cost is
small; it stays opt-in only because it isn't bit-identical to q8_0. On native Windows with the desktop on the same card,
D at the full context goes past what the driver keeps resident, so `scripts\serve.bat` switches to F above 240k. Every
preset, the window and rotation sweeps, and KL at 160k: [docs/BENCHMARKS.md](docs/BENCHMARKS.md#kv-cache-presets).

## Picks up where you left off

The server keeps whole conversations cached between requests, and with exact reuse anything it restores answers as the
same request sent fresh would.

| prompt time | llama.cpp | qwen38-megakernel |
|---|---|---|
| follow-up turn in an 11.8k-token chat | 248 ms | **169 ms** |
| edit an earlier message in an 11.7k-token chat | 302 ms | **69 ms** |
| edit the end of a 9.1k-token system prompt | 882 ms | **82 ms** |

- **Follow-ups and edits** resume from DeltaNet checkpoints, one at every message start and every 4,096 tokens. The
  follow-up turn re-reads the previous reply, which is what keeps it exact.
- **New sessions** of the same client resume after its system prompt and tools, whose checkpoint is pinned. A new
  Claude Code session starts with 8,288 of its 11,850 prompt tokens already cached, a new Codex CLI session with 8,672
  of 8,847.
- **Several conversations** take turns: the inactive ones are parked in host memory. A 120k-token conversation comes
  back in 0.18 s instead of the 98 s it took to read it again (2026-10-19).
- **Title, tag and summary requests** run beside the session (side requests) instead of throwing its cache away.

How it works and how to size it: [docs/SERVER.md](docs/SERVER.md#the-conversation-cache).

## Works with your coding agent

The server implements OpenAI chat completions (`/v1/chat/completions`), the Anthropic Messages API (`/v1/messages`,
with token counting), the OpenAI Responses API (`/v1/responses`) and llama-server's endpoints (`/health`, `/props`,
`/v1/models`, `/slots`, `/metrics`, `/tokenize`, `/detokenize`, `/apply-template`), with streaming, tool calls streamed
as they're generated, thinking, images and video. These clients ran agent tasks against it (create a file, find and
fix a failing test, follow-ups, questions about a large file, an image):

| client | version tested | client | version tested |
|---|---|---|---|
| Claude Code (Anthropic Messages) | 2.1.306 | Goose | 1.55.0 |
| Codex CLI (OpenAI Responses) | 0.163.0 | Hermes Agent | v0.22.3 |
| Qwen Code | 0.25.3 | Aider | 0.86.2 |
| Pi | 1.3.0 | Open WebUI | 0.12.3 |
| DeepSeek Harness (`dsh`) | 0.2.1-rc.2 | OpenCode | 2.0.32 |
| Crush | v0.98.1 | | |

The others use chat completions. Each needs the base URL, any API key and the model id (the GGUF's file name); most
also need the context window set by hand, some the image flag or the thinking level. Per-client settings and quirks:
[docs/CLIENTS.md](docs/CLIENTS.md).

## How it's fast

- **One kernel launch per decode cycle.** The main model pass over up to 8 tokens (16 in the experimental deep-pass
  build), lm_head, draft verification, the MTP catch-up and the next drafts (about 340 phases) run in one persistent
  cooperative kernel. The host only reads back the accepted tokens.
- **Verification on int8 tensor cores.** The main pass runs as int8 GEMMs, so a cycle that verifies 8 tokens costs
  about 29 ms against 22 ms for one (a 2026-10-16 build: [ENGINE.md](docs/ENGINE.md#how-it-works)).
- **Weights stay in their GGUF k-quant encoding**, re-laid-out at load for aligned loads (values unchanged) and read
  once per cycle however many tokens are verified.
- **DeltaNet rollback by lazy replay.** The recurrent state commits only through the surely-accepted token; rejected
  drafts cost no state snapshots.
- **Attention straight from the q8_0 cache:** a software-pipelined decode kernel, and Q K^T on int8 tensor cores in
  prefill. The batched tensor-core prefill reads a 101.6k-token prompt in 68.5 s.

The full tour: [docs/ENGINE.md](docs/ENGINE.md).

## Requirements

- **GPU:** NVIDIA Ampere or newer with 24 GB, and at least 80 SMs for more than 4 drafts (50 for up to 4). Tuned and
  tested on an RTX 3090 (sm_86); other architectures build but are untested. `--agents` with 2 or more slots needs 80
  SMs at any draft count. Making its batched passes work on 50-79 SMs (A10, L4, RTX A5000) is open, and a pull request
  is welcome.
- **CUDA 13.2** (older 12.x versions are untested), Linux, WSL2 or Windows 10 / 11 x64.
- **RAM:** ~22 GiB free to compile the served configuration. At run time, page-locked host memory for parked
  conversations up to `MK_PARK_MIB` (8 GiB by default, allocated as needed).
- **llama.cpp b10246** next to this repository, **Python 3.10+** with jinja2, an x86-64 CPU with AVX2 + FMA for the
  fast image encoder, and ffmpeg for video (optional).

Details: [docs/BUILDING.md](docs/BUILDING.md#requirements).

## Documentation

- [docs/BUILDING.md](docs/BUILDING.md): requirements, the Linux and Windows builds, model files, checking a build.
- [docs/CONFIGURATION.md](docs/CONFIGURATION.md): every launcher setting, server option, environment variable and
  build flag.
- [docs/SERVER.md](docs/SERVER.md): the APIs, the conversation cache (exact reuse, checkpoints, parking, side
  requests), the console, limits and security.
- [docs/CLIENTS.md](docs/CLIENTS.md): settings for each coding agent and chat UI.
- [docs/BENCHMARKS.md](docs/BENCHMARKS.md): headline numbers, then each measurement with its date, commit and method.
- [docs/ENGINE.md](docs/ENGINE.md): how the engine works inside.

## Credits and license

This engine started as the megakernel of [open-jet](https://github.com/L-Forster/open-jet) by Louis Forster
(L-Forster), who wrote its design and first version (the first commits of this history, 4d92fd2 to 0f40d41): the
persistent kernel that runs a whole speculative decode cycle in one launch, MTP drafting with on-device verification
(greedy and exact speculative sampling), the batched tensor-core prefill, the Q4_K_M weight kernels, the draft
vocabulary, the C API, CLI and OpenAI-compatible server with prefix reuse, the benchmark scripts, and the accuracy
method used above (`tools/llama_ref`: KL against llama.cpp's token-by-token distributions, with llama.cpp's batched run
for scale). Evan Light contributed the load-time tensor-type check and the server warm-up (open-jet PRs #14 and #16).

This repository is open-jet's `megakernel/` directory, split out with its history from our fork's
[`qwen38-perf`](https://github.com/cutlerbenjamin1-cmd/open-jet/tree/qwen38-perf) branch, and development continues
here. Much of the work has gone back upstream, including UD-Q4_K_XL support, draft-depth-independent greedy output,
the pipelined decode attention, the int8 prefill attention, the low-bit KV presets and parallel agents:
[our pull requests to open-jet](https://github.com/L-Forster/open-jet/pulls?q=is%3Apr+author%3Acutlerbenjamin1-cmd).

It also stands on [llama.cpp](https://github.com/ggml-org/llama.cpp): its tokenizer (libllama), its image library
(libmtmd: the fallback encoder and the reference tools), its GGUF quantization formats, whose dequantization the
kernels follow, and its prompt layout for images and video. The model files and the default sampling settings are
[Unsloth](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF)'s; the model, Qwen3.8-27B, is Qwen's.

Licensed under the GNU AGPL v3, like open-jet ([LICENSE](LICENSE)).
