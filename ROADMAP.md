# Roadmap

Open engineering work, roughly in priority order. Each item is sized for one contributor or a
short series of pull requests. Comment on the matching issue, or open one, before starting a
large item, and follow [What A Pull Request Needs](CONTRIBUTING.md#what-a-pull-request-needs):
engine changes come with before/after benchmarks, and new quants with accuracy numbers.

## 1. Anthropic Messages and OpenAI Responses APIs on the engine server

`megakernel/server/mk_server.py` serves `/v1/chat/completions`, `/v1/models` and `/v1/health`.
Claude Code talks to `/v1/messages` (Anthropic Messages); Codex CLI talks to `/v1/responses`
(OpenAI Responses). Neither can use the engine directly today.

- Add `POST /v1/messages`: system prompt, content blocks, `tool_use` / `tool_result`,
  thinking blocks mapped to `reasoning_content`, streaming as Anthropic SSE events
  (`message_start`, `content_block_delta`, `message_stop`).
- Add `POST /v1/responses`: input items, function calls and their outputs, reasoning items,
  streaming events.
- Both reuse the existing chat-completions path (template rendering, tool-call parsing,
  sampling), not a second implementation.
- Done when: a recipe in `docs/clients/` shows Claude Code and Codex CLI each completing a task
  with tool calls against the engine, plus tests for the request/response mapping.

## 2. Prebuilt engine binaries (no `nvcc` on the user's machine)

The wheel ships the engine as source and `openjet setup` compiles it, so users need the CUDA
toolkit. Most only have the NVIDIA driver.

- Build `libqwen38mk.so` (and `libqwen38mk_mixed.so`) in CI as one fat binary for sm_75, sm_86,
  sm_89 and sm_120, and ship it in the Linux x86_64 wheel or as release assets.
- The library links `libllama` / `libggml*` (tokenizer, via `capi.cpp`) and cudart / cuBLAS /
  cuBLASLt. Bundle the llama.cpp libraries, or link them statically. For CUDA, depend on
  NVIDIA's pip wheels: `nvidia-cuda-runtime` 13.4.92 and `nvidia-cublas` 13.8.2.12 both publish
  `manylinux` x86_64 wheels.
- Setup uses the prebuilt library when its architecture matches and falls back to compiling
  from source otherwise.
- Done when: on a clean Linux machine with only the driver, `pipx install open-jet && openjet
  setup` serves Qwen3.8-27B with the engine, and `pgrep -af mk_server.py` shows it.

## 3. Windows build of the engine

The engine is Linux-only; Windows users fall back to llama.cpp.

- Build `qwen38_mk.cu`, `capi.cpp` and the server under MSVC + nvcc, and load the DLL from
  `mk_server.py`.
- `nvidia-cuda-runtime` 13 has a `win_amd64` wheel; `nvidia-cublas` 13 does not (Linux only),
  so cuBLAS / cuBLASLt DLLs have to be bundled or taken from the CUDA 12 wheels.
- Done when: a Windows machine with an RTX 30/40/50 card and 24 GB+ runs `openjet setup` and
  serves with the engine; decode speed reported against the same card under Linux or WSL.

## 4. i-quant support: 16 GB and 12 GB cards

Every Qwen3.8-27B GGUF under ~15 GiB on Hugging Face is a dynamic mix of many weight types, not one
(read from the GGUF headers; GiB of layer weights per type):

| File | Size | Largest weight types | lm_head / token_embd |
|---|---:|---|---|
| unsloth `UD-IQ4_XS` | 13.27 GiB | IQ4_XS 6.72, IQ3_S 1.50, Q4_K 1.18, Q5_K 1.14, IQ3_XXS 0.51, Q3_K 0.24, IQ2_S 0.19, IQ2_XS, Q2_K, IQ4_NL | Q5_K / Q3_K |
| unsloth `UD-Q3_K_XL` | 12.24 GiB | IQ4_XS 4.63, IQ3_S 3.29, IQ3_XXS 0.84, Q4_K 0.60, IQ2_S 0.38, Q3_K 0.30, Q5_K 0.28, IQ2_XS, Q2_K, IQ2_XXS, IQ4_NL | Q5_K / Q3_K |
| unsloth `UD-IQ3_S` | 11.21 GiB | IQ3_S 3.46, IQ4_XS 2.42, IQ3_XXS 1.73, IQ2_S 0.54, Q3_K 0.37, IQ2_XS 0.27, IQ2_XXS 0.25, Q2_K 0.11, IQ1_S 0.03 | Q5_K / Q3_K |
| ISTA-DASLab `GSQ-RCO-IQ3_S-mtp` | 11.29 GiB | IQ3_S 3.39, IQ4_XS 2.81, IQ3_XXS 1.93, Q4_K 0.81, IQ2_S 0.37, Q2_K 0.22, IQ2_XS 0.21, IQ2_XXS, IQ1_M | Q4_K / IQ2_S |
| unsloth `UD-IQ3_XXS` | 10.18 GiB | IQ3_XXS 2.78, IQ3_S 2.42, IQ4_XS 1.34, IQ2_S 0.78, IQ2_XXS 0.44, IQ1_S 0.13, IQ1_M 0.05, ... | Q4_K / Q2_K |

The mixed build (`qwen38_mixed.cu`) reads IQ4_XS natively and, on branch `iq3s-native`, IQ3_S, IQ3_XXS,
IQ2_XXS, IQ2_XS, IQ2_S, IQ1_S, IQ1_M, Q3_K and Q2_K (repacked at load into GEMV-friendly blocks: IQ3 and
Q3_K at 3.5 bpw, IQ2, IQ1 and Q2_K at 3 bpw; `tools/iq_repack_check` verifies every layout bit-exact
against ggml), a Q5_K / Q4_K / Q8_0 lm_head and a token_embd of any of those types. Every tensor of all
five files above is then of a supported type. On UD-Q3_K_XL, KL divergence against llama.cpp token by token is
0.00136 (decode) / 0.00139 (prefill), at llama.cpp's own batched noise floor (0.00145); setup offers it on 16 GB
cards (per-model VRAM gate, ~48K context). Remaining:

1. **Faster codebook lookups**: the i-quant kernels read their grids through L1 (`__ldg`); a
   shared-memory copy may be faster. Measure decode and prefill against IQ4_XS first.
2. **Smaller blocks for IQ2 / IQ1 / Q2_K**: the 96-byte layout is 3 bpw, against 2.06-2.63 bpw
   in the GGUF; it costs ~0.1-0.4 GiB on the files above.

Done when, for each file: the loader accepts it, decode and prefill speeds are reported, and
top-token agreement and KL divergence against llama.cpp on the same file are reported with the
commands under Correctness in the [engine README](megakernel/README.md).

## 5. Fit the engine on 16 GB cards

Done for UD-Q3_K_XL on branch `iq3s-native` (per-model VRAM gate from the GGUF header, measured engine
overhead ~1.25 GiB); still to do: a run on a real 16 GB card, and the same for UD-IQ4_XS (~25K context).

**Needs a contributor with a 16 GB card.** The engine is developed and tuned on an RTX 3090 (sm_86).
Accuracy can be checked there, but the 16 GB cards are Ada (sm_89) or Blackwell (sm_120), with
different SM counts, L2 sizes and memory bandwidth, so speed and launch shapes have to be measured
and tuned on the card itself.

Depends on item 4. Once an 11-12 GiB file loads, the remaining VRAM has to cover the KV cache,
MTP state, prefill scratch and the CUDA context.

- Measure the engine's VRAM use per component at 8K, 32K and 128K context.
- Size the context automatically from free VRAM, the way setup already does for llama.cpp.
  Existing levers: `MK_EMBD_DEVICE=0` (embeddings in RAM, ~680 MiB), `MK_PF_CAP=1024` (~325
  MiB of prefill scratch). KV stays at q8_0 or above.
- Lower the setup gate from 24 GB for files that fit.
- Done when: an RTX 4060 Ti 16 GB, 4070 Ti Super, 5070 Ti or 5080 serves the IQ3_S file through
  `openjet setup`, with the context it chose and the decode speed.

## 6. `openjet bench`: a standard benchmark with a results path

Hardware reports today are free-form issues.

- `openjet bench` runs a fixed prompt set against the running server (engine or llama.cpp):
  reasoning on, the model card's sampling, prompts at 4K, 32K and 100K tokens, a few
  needle-recall checks. It writes `results.json` with hardware, driver, versions, model file,
  flags and every request's timings.
- A results-only pull request path: `bench/results/<date>-<gpu>/` holds the JSON and a short
  README, and a generated table in `docs/benchmarks/` lists all submitted results.
- A `[Benchmark]` issue template for runs that can't become a pull request.

## 7. Same-machine comparison harness

`bench/think_bench.py` compares the engine with llama.cpp using llama.cpp's flags. Extend it
to start any OpenAI-compatible server from a command line (other llama.cpp forks, other
engines), with the same rendered prompts, seeds and sampling. Report:

- decode at short context and at 100K depth, prompt speed, time to first token
- which servers sample exactly; flag non-exact draft acceptance, e.g. llamAmpere's `mb-MTP`,
  and run with it off as well as on
- KV precision and model file per server, in the output JSON

## 8. Engine limitations

From [Limitations](megakernel/README.md#limitations) in the engine README:

- presence and repetition penalties (OpenJet's non-thinking preset uses a presence penalty of 1.5)
- top-k above 32
- slot save / restore, so context compaction doesn't re-prefill the whole conversation
- image input
