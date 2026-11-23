<h1 align="center"><img src="docs/assets/logo.png" width="498" alt="OpenJet"></h1>

<p align="center"><b>Run Qwen3.8-27B and Inkling 2 Mini locally, 2-3x faster than llama.cpp.</b></p>

<p align="center">NVIDIA RTX 30, 40 and 50-series GPUs (6-24 GB VRAM), plus AMD and Intel Vulkan laptops. Works with Claude Code, Codex, OpenCode, OpenClaw or OpenJet's own agent. Runs offline.</p>

<p align="center"><b>Coding agents in the cloud:</b> <a href="https://www.openjet.dev/cloud">deploy a rented RTX 3090</a> with OpenJet and
Qwen3.8-27B ready: up to 4 agents decoding together, 424 tok/s in total, about $0.20/hour on Vast.ai, with no per-token billing or usage caps.</p>

<p align="center">
  <a href="#models">Models</a> · <a href="#speed">Speed</a> · <a href="#install">Install</a> · <a href="#what-you-need">Hardware</a> ·
  <a href="#use-it-with-your-coding-tools">Coding tools</a> · <a href="#4-coding-agents-on-one-gpu">4 agents</a> ·
  <a href="#docs">Docs</a>
</p>

<p align="center">
  <a href="https://github.com/L-Forster/open-jet/stargazers"><img src="https://img.shields.io/github/stars/L-Forster/open-jet" alt="Stars"></a>
  <a href="https://pypi.org/project/open-jet/"><img src="https://img.shields.io/pypi/v/open-jet" alt="PyPI"></a>
  <a href="LICENSE"><img src="https://img.shields.io/github/license/L-Forster/open-jet" alt="License"></a>
  <a href="https://discord.com/invite/pspKHtExSa"><img src="https://img.shields.io/badge/Discord-join-5865F2" alt="Discord"></a>
</p>

<p align="center">English · <a href="README.zh-CN.md">简体中文</a></p>

https://github.com/user-attachments/assets/e91b1cd1-d8fe-4657-b2b1-5e44a71592bf

## Models

<details>
<summary><b>Qwen3.8-27B</b> (24 GB+ NVIDIA RTX 30, 40 and 50-series)</summary>

| Quant | Your VRAM | Runtime |
|---|---|---|
| `Q4_K_M` (default) | 24 GB+ | OpenJet's engine |
| `UD-Q4_K_XL` | 24 GB+ | OpenJet's engine |
| `UD-Q3_K_XL` | 16 GB+ | OpenJet's engine (mixed) or `llama.cpp` |

[Qwen3.8-27B GGUFs on Hugging Face](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF)

</details>

<details>
<summary><b>Inkling 2 Mini</b> (6–24 GB NVIDIA RTX 30/40/50 & 16 GB AMD/Intel Vulkan laptops)</summary>

| Quant | Your RAM | Graphics card | Runtime |
|---|---|---|---|
| `UD-IQ3_XXS` (10.9 GiB) | 16 GB+ | 6 GB NVIDIA, or 8 GB NVIDIA (16 GB RAM) | OpenJet's Inkling engine |
| `UD-IQ3_XXS` (10.9 GiB) | 16 GB+ | AMD / Intel Vulkan laptop (6–10 GB heap) | `llama.cpp` Vulkan (`--cpu-moe`) |
| `UD-IQ4_XS` (14.7 GiB) | 16–32 GB+ | 8 GB NVIDIA (32 GB RAM), 10–12 GB or 24 GB NVIDIA | OpenJet's Inkling engine |
| `UD-Q4_K_XL` (16.4 GiB) | 16 GB+ | 24 GB NVIDIA | OpenJet's Inkling engine |

[Inkling 2 Mini Mainline GGUFs on Hugging Face](https://huggingface.co/unsloth/Inkling-2-Mini-GGUF/tree/main/Mainline_Support) · [Inkling guide](docs/inkling.md)

</details>

Not sure what fits? [Choose a model for your computer](docs/models.md).

## Speed

### Qwen3.8-27B (24 GB GPU)

| RTX 3090, Qwen3.8-27B Q4_K_M | llama.cpp | OpenJet |
|---|---|---|
| Thinking on, short prompts | 61 tok/s | **121 tok/s** (2.0x) |
| Thinking on, 98K tokens in context | 37 tok/s | **96 tok/s** (2.6x) |
| Reading a 98K-token prompt | 112 s | **76 s** (1.5x) |

- Single conversation on an RTX 3090, 106K context, `q8_0` KV cache, thinking on, temperature 1.0, `top_p` 0.95, `top_k` 20, 1,024 tokens per reply; 5 short prompts and one 98K prompt ([`engine_compare.py`](megakernel/bench/engine_compare.py)).
- llama.cpp `b11982` (2 drafts, flash attention, measured 2026-10-29) vs OpenJet's engine (7 drafts, measured 2026-11-17). Same harness and settings; one short prompt and the 98K prompt text differ between runs.
- Output matches llama.cpp on the same weights: 98.74% top-token agreement, 0.0016 KL divergence, vs 98.25% and 0.0023 for llama.cpp's own batched mode ([results](megakernel/README.md#results)).
- Context on 24 GB: about **120K tokens**, or about **234K** with `openjet setup --long-context` ([KV cache presets](megakernel/README.md#kv-cache-presets-mk_kv)).

### Inkling 2 Mini (8-24 GB GPU)

| RTX 3090 + Ryzen 7 5800X, `UD-IQ4_XS` (256 greedy tok) | llama.cpp (`UD-Q4_K_XL`) | OpenJet |
|---|---:|---:|
| 8 GB card budget (`6000 MiB` expert cache) | 17.7-19.6 tok/s | **47.8-51.1 tok/s** (2.6x) |
| 12 GB card budget (`9000 MiB` expert cache) | 23.0-24.7 tok/s | **65.0-67.1 tok/s** (2.7x) |
| Full VRAM (no MTP) | 41.6-48.9 tok/s | **82.0 tok/s** (1.8x) |
| Full VRAM + DSpark MTP (`mtp.safetensors`) | - | **75-140 tok/s** (95.8-139.7 code, 75.0 prose) |

- Measured on RTX 3090 + Ryzen 7 5800X (DDR4, 15 GB RAM, warm page cache). Smaller cards simulated by capping `IK_VRAM_MIB`; prefill runs at **157 tok/s** (8 GB), **225 tok/s** (12 GB), and **566 tok/s** (full VRAM). Full breakdown: [docs/inkling.md](docs/inkling.md#measured-speed).

## Install

```bash
pip install open-jet
openjet setup      # finds your GPU, downloads the model and OpenJet's prebuilt engine for it
openjet            # OpenJet's own coding agent
openjet serve      # or serve the model to Claude Code, Codex, OpenCode or OpenClaw
```

More: [docs/installation.md](docs/installation.md)

## What you need

- **24 GB NVIDIA RTX 30, 40 or 50-series (Linux):** runs Qwen3.8-27B on OpenJet's engine (`sm_86`, `sm_89`, `sm_120`, driver 580+).
- **6-12 GB NVIDIA RTX 30, 40 or 50-series (Linux, 16 GB+ RAM):** runs Inkling 2 Mini on OpenJet's Inkling engine with VRAM expert caching + CPU offload.
- **16 GB AMD/Intel Vulkan laptops, Apple silicon, Windows, or CPU-only:** `openjet setup` installs `llama.cpp` and picks the strongest model that fits your memory.

`openjet setup` downloads the prebuilt engine for your GPU ([details](docs/installation.md#prebuilt-engines)). No local GPU? [Deploy a rented RTX 3090](https://www.openjet.dev/cloud) with OpenJet and Qwen3.8-27B ready (up to 4 agents decoding together, 424 tok/s total, about $0.20/hour on Vast.ai).

More: [docs/installation.md](docs/installation.md#what-you-need)

## 4 coding agents on one GPU

https://github.com/user-attachments/assets/0df36ad0-b484-463e-a4f7-530e5a768cd9

Up to 4 agents decode in the same pass over the weights: about 420 tok/s in total on an RTX 3090 on short
prompts, 360 tok/s with about 23K tokens of context each.

More: [docs/features.md](docs/features.md#4-coding-agents-on-one-gpu)

## Results on your GPU

`openjet bench --github <your-handle>` measures your card with a fixed workload. Send the file in a
pull request to [`benchmarks/community/`](benchmarks/community/) ([what a run measures](benchmarks/community/README.md#what-a-run-measures)).

<!-- community-results:start -->
No results yet. Run `openjet bench` and send yours; the first row could be your card.
<!-- community-results:end -->

More: [benchmarks/community/README.md](benchmarks/community/README.md)

## Use it with your coding tools

`openjet serve` speaks Anthropic Messages, OpenAI Responses and OpenAI Chat Completions;
`openjet connect claude-code` (or `codex`, `opencode`, `openclaw`) prints the settings.

More: [docs/features.md](docs/features.md#use-it-with-your-coding-tools)

## OpenJet's terminal agent

A coding agent built for a model that fits on your GPU: bounded context, automatic condensing,
approved shell commands, resumable sessions, [skills](docs/usage/skills.md) and [MCP](docs/usage/mcp.md).

More: [docs/features.md](docs/features.md#openjets-terminal-agent)

## Python SDK

Embed a local model in your application.

More: [docs/features.md](docs/features.md#python-sdk-embed-a-local-llm-in-your-application)

Also: [tuned to the hardware](docs/features.md#tuned-to-the-hardware) · [Ollama, LM Studio or llama-server](docs/features.md#use-ollama-lm-studio-or-an-existing-llama-server) · [FAQ](docs/features.md#faq)

## Built on OpenJet

- [geezuz-qwen3.8-27b-dual-gpu](https://github.com/ohgeebtw/geezuz-qwen3.8-27b-dual-gpu): OpenJet's engine split across 2 GPUs (tensor-parallel, 2x RTX 3090 over P2P)
- [qwen38-megakernel](https://github.com/cutlerbenjamin1-cmd/qwen38-megakernel): OpenJet's engine developed further as a standalone server, with a CPU vision encoder; OpenJet's engine now comes from it

Built something on OpenJet? Open a pull request to add it here.

## Docs

- Start: [Getting started](docs/getting-started.md) · [Quickstart](docs/quickstart.md) ·
  [Installation](docs/installation.md) · [Configuration](docs/configuration.md) ·
  [Choosing a model](docs/models.md) · [Overview](docs/overview.md)
- Coding tools: [Claude Code, Codex, OpenCode and OpenClaw](docs/clients.md)
- Runtime: [OpenJet's engine](megakernel/README.md) · [llama.cpp](docs/runtimes/llama-cpp.md) ·
  [Benchmarking](docs/benchmarking.md)
- Agent: [CLI](docs/usage/cli.md) · [Slash commands](docs/usage/slash-commands.md) ·
  [Slipstream](docs/usage/slipstream.md) · [Skills](docs/usage/skills.md) ·
  [MCP](docs/usage/mcp.md) · [Device sources](docs/usage/device-sources.md) ·
  [Workflow harness](docs/usage/workflow-harness.md) ·
  [Session state and logging](docs/usage/session-state-and-logging.md)
- SDK: [Quickstart](docs/sdk/quickstart.md) · [Python SDK](docs/sdk/python-sdk.md)
- For AI tools and writers: [AGENTS.md](AGENTS.md) · [Install with an agent](docs/agent-install.md) ·
  [AI reference](docs/AI_REFERENCE.md) · [Media kit](docs/MEDIA_KIT.md)
- Deployment: [Jetson](docs/deployment/jetson.md) ·
  [Linux x86 + NVIDIA](docs/deployment/linux-x86-nvidia.md) · [CPU-only](docs/deployment/cpu-only.md)

## Community and license

[Discord](https://discord.gg/pspKHtExSa) · [X / Twitter](https://x.com/FlouisLF) · `AGPL-3.0-only` ([licensing](docs/licensing.md), [CONTRIBUTING.md](CONTRIBUTING.md), [CLA.md](CLA.md))
