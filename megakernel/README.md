# OpenJet's Qwen3.8-27B engine

A CUDA engine and server built for one model, Qwen3.8-27B. Each decode step runs in one kernel launch: the model's own
MTP head drafts up to 7 tokens and the GPU checks them in the same pass. The server speaks OpenAI Chat Completions,
OpenAI Responses and Anthropic Messages, so coding agents connect to it directly. OpenJet installs and starts it for you
(`openjet setup`, then `openjet` or `openjet serve`); this page is for running, building or changing it yourself.

This directory holds [qwen38-megakernel](https://github.com/cutlerbenjamin1-cmd/qwen38-megakernel) by Benjamin Cutler,
at commit `ed0d237`. It began as this directory's engine and was developed further in that repository; OpenJet adopted
it in November 2026 ([credits](#credits-and-license)). The engine OpenJet shipped before is in [`legacy/`](legacy/README.md),
and still serves what this one doesn't: Turing GPUs, UD-Q3_K_XL and cards with less than 24 GB.

## Quick guide

Which engine OpenJet starts:

| GPU | Model file | Engine |
|---|---|---|
| RTX 30, 40 or 50-series with 24 GB or more (sm_86, sm_89, sm_120) | unsloth Qwen3.8-27B `Q4_K_M` or `UD-Q4_K_XL` | this one |
| the same | a `Q4_K_M` from lmstudio-community, mradermacher, Abiray or KikoCis | this one, semi compatible: it reads the file, but it isn't benchmarked |
| the same | unsloth `UD-Q3_K_XL` | the previous engine's mixed-quant build ([legacy/](legacy/README.md)) |
| sm_86, sm_89 or sm_120 with 16 GB | unsloth `UD-Q3_K_XL` (12.4 GiB on the GPU, about 48K context) | the previous engine's mixed-quant build |
| Turing with 24 GB or more (sm_75) | `Q4_K_M` | the previous engine, experimental ([Turing](legacy/README.md#turing-rtx-8000)) |
| anything else | any other quant | llama.cpp |

`_megakernel_variant` in [`src/llama_server.py`](../src/llama_server.py) makes this choice from the GPU and the GGUF header.

## Results

### Speed

On one RTX 3090 against the previous engine (OpenJet 76b939d), each started as OpenJet starts it. Same unsloth
`Q4_K_M` file, 106,496-token context, q8_0 KV cache, thinking on, sampled (temperature 1.0, top_p 0.95, top_k 20),
1,024 tokens generated per request. Short: 5 prompts of 77 to 3,088 tokens; long: one 98,092-token prompt. 8 seeds
each, in 4 interleaved rounds. This engine is the build of this source from the engine workflow (`engine.yml`).

| | Previous engine | This engine | Same seed, this engine / previous (median) |
|---|---|---|---|
| Decode, short prompts (40 requests) | 90.1 tok/s | **121.2 tok/s** | **1.36x**, faster in 40 of 40 |
| Decode with 98K tokens in context (8 requests) | 74.5 tok/s | **95.9 tok/s** | **1.25x**, faster in 8 of 8 |
| Read the 98K-token prompt, cold | 80-98 s | **72-83 s** | |

- 2026-11-17, driver 610.62. Something else used the GPU in the last round, which slowed all three engines in it (the
  previous engine's short prompts from about 95 to 75 tok/s); the same-seed ratios are the reliable part.
- The previous engine runs at its own defaults (4 drafts); this engine at OpenJet's (7 drafts, exact reuse).
- The release binary this source comes from (qwen38-megakernel v2026.11.11) ran in the same session: this build's
  same-seed median was 1.00x its speed on short prompts and 0.99x at 98K, and the two gave the same 12 of 12 replies to
  seeded requests.
- Reproduce with [`bench/engine_compare.py`](bench/engine_compare.py)
  `--seeds 8 --rounds 4 -n 1024 --suite short,long --output speed.json`, one `--engine NAME=PORT=COMMAND` per
  server.

Against llama.cpp, the measurements in OpenJet's [README](../README.md#speed) are of the previous engine. Benjamin
Cutler's own measurements of this engine against llama.cpp (UD-Q4_K_XL, llama.cpp b12024, his release v2026.11.05) are in
[docs/UPSTREAM_README.md](docs/UPSTREAM_README.md#speed).

### Accuracy

Next-token distributions against llama.cpp generating one token at a time on the same weights, q8_0 KV cache. llama.cpp
in its own batched mode, against the same reference, shows the noise floor.

| | Top token agrees | Mean KL divergence |
|---|---|---|
| **Short context:** 13 chunks of 2,048 tokens (7 code, 6 prose), scored from position 1,024; 13,299 positions | | |
| llama.cpp batched | 98.25% | 0.00225 |
| previous engine | 98.66% | 0.00161 |
| this engine | **98.74%** | **0.00162** |
| **After a 94K-token prompt:** 1,980 positions | | |
| llama.cpp batched | 99.49% | 0.00111 |
| previous engine | 99.70% | 0.00092 |
| this engine | **99.70%** | **0.00093** |

Measured 2026-11-16 with the v2026.11.11 release binary, which gives the same replies as this build (above). The method
is `tools/llama_ref` against `qwen38_mk --kld` ([docs/BUILDING.md](docs/BUILDING.md#checking-a-build)).

## Use with OpenJet

`openjet setup` downloads this engine prebuilt for your GPU, or compiles it from a source checkout. OpenJet then starts
`server/mk_server.py` with:

- `--kv q8 --exact-reuse -d 7`: the q8_0 KV cache, cached conversations answering exactly as fresh ones, 7 drafts. A GPU
  with fewer than 80 SMs refuses 7 drafts at load, and OpenJet starts it again with `-d 4`.
- `--park-mib 0` on machines with less than 32 GiB of RAM: parking other conversations pins up to 8 GiB of host
  memory, which next to the model file's page cache runs a 16 GB machine out of memory.
- `MK_PF_CAP=1024` above 96K context: smaller prefill chunks, 0.34 GB less GPU memory.
- With `openjet setup --long-context`, KV preset F ([below](#kv-cache-presets-mk_kv)).

`pgrep -af mk_server.py` shows the running command.

## Standalone installation

Linux, an RTX 30/40/50-series GPU with 24 GB, the CUDA toolkit (13.2) and a machine with 24 GiB of RAM or more for the
compile:

```sh
git clone https://github.com/l-forster/open-jet.git
cd open-jet
git clone --depth 1 --branch b10246 https://github.com/ggml-org/llama.cpp.git llama.cpp
cmake -S llama.cpp -B llama.cpp/build -DGGML_CUDA=ON -DLLAMA_CURL=OFF
cmake --build llama.cpp/build --config Release -j 2
python3 -m venv .venv-engine
. .venv-engine/bin/activate
python -m pip install -e ./megakernel
make -C megakernel -j1 NVCC="$(command -v nvcc)" libqwen38mk.so
openjet-engine --model /absolute/path/to/Qwen3.8-27B-Q4_K_M.gguf --port 18080 -c 106496 --kv q8 --exact-reuse -d 7
```

The server's own default is 4 drafts, so pass `-d 7` as above. `openjet-engine --help` lists every option;
`python -m megakernel` does the same. Instead of compiling, you can unpack a release's
`openjet-engine-linux-x64-sm86.tar.gz` (or `sm89`, `sm120`) into `megakernel/`: it holds `libqwen38mk.so` and the
llama.cpp libraries it needs.

## Build / run

- **Compiling takes about 21 GiB of RAM**, almost all of it ptxas on the megakernel, so build with `-j1`. It takes
  up to half an hour. `openjet setup` doesn't try on a machine with less than 24 GiB and serves llama.cpp there instead.
- `make` builds for the installed GPU; `make ARCH=sm_89` overrides it. `make` with no target also builds the
  `qwen38_mk` command line and the tools.
- Windows: [`build.ps1`](build.ps1) (`-Targets engine -Arch sm_86 -Llama <llama.cpp> -Cuda <CUDA>`), which sets up
  MSVC 2022 itself.
- The release archives are built by [`.github/workflows/engine.yml`](../.github/workflows/engine.yml) with CUDA 13.2,
  so they need NVIDIA driver 580 or newer.

Requirements, both builds, the model files and the checks to run after a build:
[docs/BUILDING.md](docs/BUILDING.md).

### The GGUF

The loader reads F32, F16, Q4_0, Q8_0, Q3_K, Q4_K, Q5_K, Q6_K, IQ4_NL, IQ3_S and IQ4_XS ([`gguf.h`](gguf.h)), checks each
tensor's type for its role, and names the first one it refuses. unsloth's `Q4_K_M` and `UD-Q4_K_XL` are the files it's
measured on. A plain `llama-quantize ... Q4_K_M` also loads; OpenJet labels it semi compatible. `UD-Q3_K_XL` holds
Q2_K, IQ2_XXS, IQ2_XS, IQ2_S and IQ3_XXS tensors, which this engine has no kernels for, so it runs on the previous
engine.

## KV cache presets (MK_KV)

The KV cache is q8_0 by default. `MK_KV` stores older positions in fewer bits, and `MK_KV_WIN` keeps the newest ones at
q8_0, where attention puts most of its weight:

| Preset | Environment | K / V bits |
|---|---|---|
| default | (none) | 8 / 8 |
| D | `MK_KV='K*=6,V*=5' MK_KV_WIN=2048` | 6 / 5 |
| F | `MK_KV='K*=5r,V*=5' MK_KV_WIN=2048` | 5 / 5, K Hadamard-rotated |

`openjet setup --long-context` starts the engine with preset F and offers about 234K tokens of context on a 24 GB card,
against about 120K with q8_0. Started as OpenJet starts it on an RTX 3090 with about 1.7 GiB already used by the
desktop, with one short request, peak GPU memory was:

| Context | KV cache | Peak, of 24,576 MiB |
|---|---|---|
| 120,000 | q8_0 | 22,383 MiB |
| 234,000 | preset F | 23,433 MiB |
| 262,144 (the model's maximum) | preset F | 23,855 MiB |

Only short conversations were run at these sizes. Benjamin Cutler's KL and benchmark measurements of each preset are
in [docs/BENCHMARKS.md](docs/BENCHMARKS.md#kv-cache-presets).

## Limitations

- Qwen3.8-27B only, and Ampere or newer (sm_80+): Turing runs the [previous engine](legacy/README.md).
- More than 4 drafts needs a GPU with at least 80 SMs (50 for up to 4). `--agents` with 2 or more slots needs 80 SMs
  at any draft count.
- Tuned and measured on an RTX 3090 (sm_86). sm_89 and sm_120 build in CI but haven't been measured.
- `top_k` runs 1 to 32; `repetition_penalty` is ignored with a warning. `--agents` is text only and refuses the
  presence and frequency penalties.
- Compiling needs about 21 GiB of RAM (see [Build / run](#build--run)).

## Documentation

[`docs/`](docs/) holds the documentation written for the standalone project. It uses that project's name and launch
scripts (`scripts/serve.sh`), and its numbers are Benjamin Cutler's measurements, not OpenJet's.

- [docs/BUILDING.md](docs/BUILDING.md): requirements, the Linux and Windows builds, model files, checking a build.
- [docs/CONFIGURATION.md](docs/CONFIGURATION.md): every server option, environment variable and build flag.
- [docs/SERVER.md](docs/SERVER.md): the APIs, the conversation cache (exact reuse, checkpoints, parking), limits.
- [docs/CLIENTS.md](docs/CLIENTS.md): settings for each coding agent and chat UI.
- [docs/BENCHMARKS.md](docs/BENCHMARKS.md): his measurements, each with its date, commit and method.
- [docs/ENGINE.md](docs/ENGINE.md): how the engine works inside.
- [docs/UPSTREAM_README.md](docs/UPSTREAM_README.md): the standalone project's README.

## Credits and license

Louis Forster (L-Forster) wrote this engine's design and first version in OpenJet: the persistent kernel that runs a
whole speculative decode cycle in one launch, MTP drafting verified on the GPU, the batched prefill, the Q4_K_M weight
kernels, the C API and the server. Benjamin Cutler developed it from there in
[qwen38-megakernel](https://github.com/cutlerbenjamin1-cmd/qwen38-megakernel), including the UD-Q4_K_XL support, the
low-bit KV presets, the conversation cache, parallel agents and the 7-draft decode; much of that work also went into
OpenJet through his pull requests before this directory took the whole engine. Evan Light contributed the load-time
tensor-type check and the server warm-up. Third-party code is listed in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

It uses [llama.cpp](https://github.com/ggml-org/llama.cpp) for the tokenizer (libllama) and follows its GGUF
quantization formats. The model files and default sampling settings are
[Unsloth](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF)'s; the model is Qwen's.

GNU AGPL v3, like OpenJet ([LICENSE](LICENSE)).
