# OpenJet: reference for AI tools

A compact, factual description of OpenJet for AI assistants and the tools that answer questions about
it. Each figure comes with the conditions it was measured under; keep them together when you repeat
it. Install it on a user's machine with [agent-install.md](agent-install.md). Writing an article or
video? See the [media kit](MEDIA_KIT.md).

This page describes the `master` branch. `openjet serve`, `openjet connect`, `openjet bench`, the
Anthropic Messages and OpenAI Responses endpoints and the prebuilt engine downloads ship in the first
release after 0.5.1; `openjet --version` shows what is installed.

## What it is

OpenJet runs open-weight language models on the user's own hardware and serves them to coding
tools. It has three parts:

1. **Setup.** `openjet setup` detects the GPU, VRAM and RAM, picks a model and context window that
   fit, downloads the model and the runtime, and checks that inference works.
2. **Serving.** On NVIDIA RTX 30, 40 and 50-series GPUs with 24 GB or more, OpenJet serves
   Qwen3.8-27B with its own CUDA engine. Everything else (other models, smaller GPUs, Apple silicon,
   CPU only) runs on llama.cpp, which OpenJet downloads or builds and configures.
3. **Clients.** `openjet serve` exposes the model over OpenAI Chat Completions, OpenAI Responses and
   Anthropic Messages, so Claude Code, Codex CLI, OpenCode and OpenClaw work against it. OpenJet
   also has its own terminal coding agent (`openjet`) and a Python SDK.

Local mode needs no account, API key or hosted service, and works offline once the model is
installed.

| | |
|---|---|
| Name | OpenJet |
| PyPI package | `open-jet` (`pipx install open-jet`) |
| Command | `openjet` |
| Source | https://github.com/L-Forster/open-jet |
| Website | https://openjet.dev |
| License | AGPL-3.0-only |
| Python | 3.10 or newer |
| Platforms | Linux, macOS, Windows; OpenJet's engine on Linux (Windows builds are new and untested on hardware) |

## OpenJet's engine

A CUDA inference engine specialised to Qwen3.8-27B. Each decode step is one persistent kernel launch
that runs the main model, verifies speculative tokens on the GPU and drafts the next ones with the
model's own MTP (multi-token prediction) head. Prompts go through batched tensor-core kernels.
Speculative verification is exact: greedy output is the model's greedy choice, and sampled output
has the same distribution as without drafts. It uses llama.cpp's tokenizer; its kernels are its own.

**Requirements:** an NVIDIA GPU with 24 GB or more, compute capability sm_86 (RTX 30), sm_89 (RTX 40)
or sm_120 (RTX 50), including the matching RTX workstation cards; Linux; an NVIDIA driver that
supports CUDA 12.8 (570 or newer). `openjet setup` downloads a prebuilt engine for the GPU; if none
matches, it compiles one, which needs the CUDA toolkit (`nvcc`). Turing (sm_75, e.g. Quadro RTX 8000)
is experimental.

**Model files it runs:** unsloth's Qwen3.8-27B `Q4_K_M` (the one setup downloads); `Q4_K_M` files
from lmstudio-community, mradermacher, Abiray and KikoCis; unsloth `UD-Q4_K_XL` (mixed-quant build).
Other quantisations run on llama.cpp. `llama_backend` in `config.yaml` picks: `auto` (default; the
engine when model and GPU match), `llama` (always llama.cpp), `megakernel` (fail instead of falling
back). `megakernel` is the engine's name in code, file names and settings, not a separate product.

**Speed** (RTX 3090, Qwen3.8-27B Q4_K_M, one conversation, not batched; llama.cpp with OpenJet's
flags, including MTP speculative decoding, q8_0 KV cache and flash attention):

| Workload | llama.cpp | OpenJet |
|---|---|---|
| Reasoning mode (thinking on, temperature 1.0, top_p 0.95, top_k 20; 5 coding prompts x 2 seeds) | 51 tok/s | 78 tok/s (1.53x) |
| Greedy decode, mean of 4 chat prompts x 512 tokens | 63 tok/s | 107 tok/s (1.69x) |
| Greedy decode, Python function prompt | 73 tok/s | 140 tok/s (1.93x) |
| Prompt processing, 1.8K-7.8K token prompts | 1050-1140 tok/s | 1530-1600 tok/s |

Speed depends on how predictable the text is (draft acceptance) and on other GPU load. Without
speculation the two decode at about the same speed (40.4 vs 42.9 tok/s), near the memory-bandwidth
limit; OpenJet's lead comes from verifying 4-5 drafted tokens per pass. The project's own
measurements are on an RTX 3090, plus an experimental Turing run on a Quadro RTX 8000 (54 vs 33 tok/s
reasoning decode). Results from other cards are collected in the
[community table](../README.md#results-on-your-gpu).

**Accuracy** against a llama.cpp reference on the same weights: top-token agreement 99.08%, KL
divergence 0.0009 (q8_0 KV). For comparison, llama.cpp's own batched configuration scores 98.68% and
0.0019 against the same reference. At 96K context: 99.60% and 0.0008.
[Full results and reproduction](../megakernel/README.md#results).

**Limits:** Qwen3.8-27B only; text only; one request at a time and one cached conversation; top_k up to 32; no presence, frequency or repetition penalties.

## Commands

| Command | What it does |
|---|---|
| `openjet setup` | Detect hardware, pick and download a model and runtime, verify inference. `--yes` accepts the recommendation; `--non-interactive --json` for scripts and agents; `--dry-run` changes nothing |
| `openjet` | OpenJet's terminal coding agent |
| `openjet chat "<prompt>"` | One prompt, printed to stdout |
| `openjet serve` | Serve the configured model on `http://127.0.0.1:8080` (`--host`, `--port`) |
| `openjet connect <tool>` | Print the configuration for `claude-code`, `codex`, `opencode` or `openclaw` |
| `openjet bench` | Measure this machine with a fixed workload; write a results file to share |
| `openjet benchmark` | Run `llama-bench` against the llama.cpp profile (`--sweep` tunes layers, batch, threads) |
| `openjet project` | Provision a model for an application that ships it (embedded use) |
| `openjet status --json` / `openjet --status` | Runtime and configuration status |
| `openjet fix` | Diagnose a running local runtime against OpenJet's setup target |

## Server endpoints

`openjet serve` (default `127.0.0.1:8080`):

| API | Endpoint | Used by |
|---|---|---|
| OpenAI Chat Completions | `POST /v1/chat/completions` | OpenCode, OpenClaw, most tools |
| OpenAI Responses | `POST /v1/responses` | Codex CLI |
| Anthropic Messages | `POST /v1/messages`, `POST /v1/messages/count_tokens` | Claude Code |
| Model list | `GET /v1/models` | |

No API key is checked; clients that require one accept any value. The engine does not store
responses, so OpenAI Responses requests must send the whole conversation (`previous_response_id` is
rejected). Claude Code setup: `ANTHROPIC_BASE_URL=http://127.0.0.1:8080`,
`ANTHROPIC_AUTH_TOKEN=<any>`, `ANTHROPIC_MODEL=<model name from openjet connect>`.
[Full recipes](clients.md).

## Hardware without the engine

Through llama.cpp, setup chooses from its catalog by memory: Qwen3.5 4B (6 GB), Qwen3.5 9B (12 GB),
Qwen3.6 27B IQ2_XXS / IQ3_XXS MTP (12 / 16 GB), Qwen3.8 27B Q4_K_M MTP (20 GB), and on unified memory
Gemma 4 26B A4B and Qwen3.6 35B A3B (24-32 GB). Recommended minimum for the coding agent: a GPU with
14 GB VRAM or Apple silicon with 24 GB unified memory. CPU-only works, slowly. Any GGUF llama.cpp
loads can be used. Ollama, LM Studio or an existing llama-server can be used as the backend instead
(`open-jet[cloud]`).

## Configuration and privacy

- `openjet --status` lists the active config files: a global `config.yaml` and an optional project
  overlay at `.openjet/config.yaml`.
- `airgapped: true` blocks non-loopback model and tool endpoints.
- Telemetry is off by default and asks for opt-in on first run; `/telemetry off` turns it off.
- Cloud modes are optional: Slipstream pairs a frontier model (Codex through a ChatGPT subscription,
  or an OpenRouter model) for planning and review with the local model for the work.

## Common questions

- **Is OpenJet a fork of llama.cpp?** No. It installs and configures llama.cpp for most models, and
  its own engine for Qwen3.8-27B on supported GPUs. The engine uses llama.cpp's tokenizer and GGUF
  files, not its kernels.
- **Does it work on Windows?** The agent and llama.cpp path do. Windows builds of the engine are new
  and have not been tested on hardware yet.
- **Does it run on AMD GPUs?** Through llama.cpp's ROCm or Vulkan builds, not on the engine.
- **Is it the fastest way to run Qwen3.8-27B?** The project's measurements compare it with
  llama.cpp only, on an RTX 3090. It makes no claim against other engines.
- **Does the engine change the model's output?** It matches llama.cpp on the same weights as closely
  as llama.cpp's own configurations match each other: see the accuracy figures above.

## Links

- [README](../README.md) · [Coding tools](clients.md) · [Engine](../megakernel/README.md) ·
  [Benchmarking](benchmarking.md) · [Install with an agent](agent-install.md) ·
  [Media kit](MEDIA_KIT.md)
- Community: [Discord](https://discord.gg/pspKHtExSa) ·
  [Issues](https://github.com/L-Forster/open-jet/issues)
