<h1 align="center"><img src="https://raw.githubusercontent.com/L-Forster/open-jet/master/docs/assets/logo.png" width="498" alt="OpenJet"></h1>

<p align="center"><b>Run Qwen3.8-27B and Inkling 2 Mini locally, 2-3x faster than llama.cpp.</b></p>

<p align="center">NVIDIA RTX 30, 40 and 50-series GPUs (6-24 GB VRAM), plus AMD and Intel Vulkan laptops. Works with Claude Code, Codex, OpenCode, OpenClaw or OpenJet's own agent. Runs offline.</p>

<p align="center"><b>Coding agents in the cloud:</b> <a href="https://www.openjet.dev/cloud">deploy a rented RTX 3090</a> with OpenJet and
Qwen3.8-27B ready: up to 4 agents decoding together, 424 tok/s in total, about $0.20/hour on Vast.ai, with no per-token billing or usage caps.</p>

<p align="center">
  <a href="https://github.com/L-Forster/open-jet#models">Models</a> · <a href="https://github.com/L-Forster/open-jet#speed">Speed</a> · <a href="https://github.com/L-Forster/open-jet#install">Install</a> · <a href="https://github.com/L-Forster/open-jet#what-you-need">Hardware</a> ·
  <a href="https://github.com/L-Forster/open-jet#use-it-with-your-coding-tools">Coding tools</a> · <a href="https://github.com/L-Forster/open-jet#4-coding-agents-on-one-gpu">4 agents</a> ·
  <a href="https://github.com/L-Forster/open-jet#docs">Docs</a>
</p>

<p align="center">
  <a href="https://github.com/L-Forster/open-jet/stargazers"><img src="https://img.shields.io/github/stars/L-Forster/open-jet" alt="Stars"></a>
  <a href="https://pypi.org/project/open-jet/"><img src="https://img.shields.io/pypi/v/open-jet" alt="PyPI"></a>
  <a href="https://github.com/l-forster/open-jet/blob/master/LICENSE"><img src="https://img.shields.io/github/license/L-Forster/open-jet" alt="License"></a>
  <a href="https://discord.com/invite/pspKHtExSa"><img src="https://img.shields.io/badge/Discord-join-5865F2" alt="Discord"></a>
</p>

<p align="center">English · <a href="https://github.com/l-forster/open-jet/blob/master/README.zh-CN.md">简体中文</a></p>

https://github.com/user-attachments/assets/e91b1cd1-d8fe-4657-b2b1-5e44a71592bf

## Models

`openjet setup` checks your VRAM and RAM and picks the best fit automatically:

| Your hardware | Default model | Runtime |
|---|---|---|
| 24 GB+ NVIDIA RTX 30/40/50 (Linux) | **Qwen3.8-27B** (`Q4_K_M`) | OpenJet's engine |
| 8 GB NVIDIA (32 GB RAM) or 10-12 GB NVIDIA | **Inkling 2 Mini** (`UD-IQ4_XS`, 14.7 GiB) | OpenJet's Inkling engine |
| 6 GB NVIDIA or 8 GB NVIDIA (16 GB RAM) | **Inkling 2 Mini** (`UD-IQ3_XXS`, 10.9 GiB) | OpenJet's Inkling engine |
| 16 GB AMD / Intel Vulkan laptop (6-10 GB heap) | **Inkling 2 Mini** (`UD-IQ3_XXS`, 10.9 GiB) | `llama.cpp` Vulkan (`--cpu-moe`) |
| 12 GB discrete Vulkan / Apple silicon / CPU | Best coding model that fits | `llama.cpp` |

Full table and manual download links: [Choosing a model](https://github.com/l-forster/open-jet/blob/master/docs/models.md) · [Inkling 2 Mini guide](https://github.com/l-forster/open-jet/blob/master/docs/inkling.md).

## Speed

### Qwen3.8-27B (24 GB GPU)

| RTX 3090, Qwen3.8-27B Q4_K_M | llama.cpp | OpenJet |
|---|---|---|
| Thinking on, short prompts | 61 tok/s | **121 tok/s** (2.0x) |
| Thinking on, 98K tokens in context | 37 tok/s | **96 tok/s** (2.6x) |
| Reading a 98K-token prompt | 112 s | **76 s** (1.5x) |

- Single conversation on an RTX 3090, 106K context, `q8_0` KV cache, thinking on, temperature 1.0, `top_p` 0.95, `top_k` 20, 1,024 tokens per reply; 5 short prompts and one 98K prompt ([`engine_compare.py`](https://github.com/l-forster/open-jet/blob/master/megakernel/bench/engine_compare.py)).
- llama.cpp `b11982` (2 drafts, flash attention, measured 2026-10-29) vs OpenJet's engine (7 drafts, measured 2026-11-17). Same harness and settings; one short prompt and the 98K prompt text differ between runs.
- Output matches llama.cpp on the same weights: 98.74% top-token agreement, 0.0016 KL divergence, vs 98.25% and 0.0023 for llama.cpp's own batched mode ([results](https://github.com/l-forster/open-jet/blob/master/megakernel/README.md#results)).
- Context on 24 GB: about **120K tokens**, or about **234K** with `openjet setup --long-context` ([KV cache presets](https://github.com/l-forster/open-jet/blob/master/megakernel/README.md#kv-cache-presets-mk_kv)).

### Inkling 2 Mini (8-24 GB GPU)

| RTX 3090 + Ryzen 7 5800X, `UD-IQ4_XS` (256 greedy tok) | llama.cpp (`UD-Q4_K_XL`) | OpenJet |
|---|---:|---:|
| 8 GB card budget (`6000 MiB` expert cache) | 17.7-19.6 tok/s | **47.8-51.1 tok/s** (2.6x) |
| 12 GB card budget (`9000 MiB` expert cache) | 23.0-24.7 tok/s | **65.0-67.1 tok/s** (2.7x) |
| Full VRAM (no MTP) | 41.6-48.9 tok/s | **82.0 tok/s** (1.8x) |
| Full VRAM + DSpark MTP (`mtp.safetensors`) | - | **75-140 tok/s** (95.8-139.7 code, 75.0 prose) |

- Measured on RTX 3090 + Ryzen 7 5800X (DDR4, 15 GB RAM, warm page cache). Smaller cards simulated by capping `IK_VRAM_MIB`; prefill runs at **157 tok/s** (8 GB), **225 tok/s** (12 GB), and **566 tok/s** (full VRAM). Full breakdown: [docs/inkling.md](https://github.com/l-forster/open-jet/blob/master/docs/inkling.md#measured-speed).

## [Install](https://github.com/l-forster/open-jet/blob/master/docs/installation.md)

Linux and macOS:

```bash
curl -fsSL https://github.com/L-Forster/open-jet/releases/latest/download/install.sh | sh
```

Windows PowerShell:

```powershell
irm https://github.com/L-Forster/open-jet/releases/latest/download/install.ps1 | iex
```

Then, in a new terminal:

```bash
openjet setup      # finds your GPU, downloads the model and the engine built for it
openjet            # OpenJet's own coding agent
openjet serve      # or serve the model to Claude Code, Codex, OpenCode or OpenClaw
```

Or have Claude Code or Codex [install it for you](https://github.com/l-forster/open-jet/blob/master/docs/agent-install.md).

## [What you need](https://github.com/l-forster/open-jet/blob/master/docs/installation.md#what-you-need)

| Your hardware | What runs | Guide |
|---|---|---|
| NVIDIA RTX 30, 40 or 50-series with 24 GB+, Linux | Qwen3.8-27B on OpenJet's engine | [Engine for your GPU](https://github.com/l-forster/open-jet/blob/master/docs/installation.md#prebuilt-engines) |
| NVIDIA RTX 30, 40 or 50-series with 6-12 GB, Linux (16 GB+ RAM) | Inkling 2 Mini (`UD-IQ4_XS` or `UD-IQ3_XXS`) on OpenJet's Inkling engine | [Inkling guide](https://github.com/l-forster/open-jet/blob/master/docs/inkling.md) |
| 16 GB AMD/Intel Vulkan laptop (Windows or Linux) | Inkling 2 Mini `UD-IQ3_XXS` on llama.cpp Vulkan (`--cpu-moe`) | [Choosing a model](https://github.com/l-forster/open-jet/blob/master/docs/models.md#pick-by-memory) |
| No 24 GB NVIDIA card (cloud option) | Qwen3.8-27B on OpenJet's engine on a rented RTX 3090 (~$0.20/hour on Vast.ai) | [Deploy](https://www.openjet.dev/cloud) |
| NVIDIA Turing with 24 GB+, Linux | OpenJet's legacy engine (experimental) | [Turing](https://github.com/l-forster/open-jet/blob/master/megakernel/legacy/README.md#turing-rtx-8000) |
| Any other GPU, Apple silicon, Jetson, or CPU only | Best model that fits on llama.cpp | [Pick by memory](https://github.com/l-forster/open-jet/blob/master/docs/models.md#pick-by-memory) |

## [4 coding agents on one GPU](https://github.com/l-forster/open-jet/blob/master/docs/features.md#4-coding-agents-on-one-gpu)

https://github.com/user-attachments/assets/0df36ad0-b484-463e-a4f7-530e5a768cd9

Up to 4 agents decode in the same pass over the weights: about 420 tok/s in total on an RTX 3090 on short
prompts, 360 tok/s with about 23K tokens of context each.

## [Results on your GPU](https://github.com/l-forster/open-jet/blob/master/benchmarks/community/README.md)

`openjet bench --github <your-handle>` measures your card with a fixed workload. Send the file in a
pull request to [`benchmarks/community/`](https://github.com/l-forster/open-jet/blob/master/benchmarks/community/) ([what a run measures](https://github.com/l-forster/open-jet/blob/master/benchmarks/community/README.md#what-a-run-measures)).

<!-- community-results:start -->
No results yet. Run `openjet bench` and send yours; the first row could be your card.
<!-- community-results:end -->

## [Use it with your coding tools](https://github.com/l-forster/open-jet/blob/master/docs/features.md#use-it-with-your-coding-tools)

`openjet serve` speaks Anthropic Messages, OpenAI Responses and OpenAI Chat Completions;
`openjet connect claude-code` (or `codex`, `opencode`, `openclaw`) prints the settings.

## [OpenJet's terminal agent](https://github.com/l-forster/open-jet/blob/master/docs/features.md#openjets-terminal-agent)

A coding agent built for a model that fits on your GPU: bounded context, automatic condensing,
approved shell commands, resumable sessions, [skills](https://github.com/l-forster/open-jet/blob/master/docs/usage/skills.md) and [MCP](https://github.com/l-forster/open-jet/blob/master/docs/usage/mcp.md).

## [Python SDK](https://github.com/l-forster/open-jet/blob/master/docs/features.md#python-sdk-embed-a-local-llm-in-your-application)

Embed a local model in your application.

Also: [tuned to the hardware](https://github.com/l-forster/open-jet/blob/master/docs/features.md#tuned-to-the-hardware) · [Ollama, LM Studio or llama-server](https://github.com/l-forster/open-jet/blob/master/docs/features.md#use-ollama-lm-studio-or-an-existing-llama-server) · [FAQ](https://github.com/l-forster/open-jet/blob/master/docs/features.md#faq)

## Docs

- Start: [Getting started](https://github.com/l-forster/open-jet/blob/master/docs/getting-started.md) · [Quickstart](https://github.com/l-forster/open-jet/blob/master/docs/quickstart.md) ·
  [Installation](https://github.com/l-forster/open-jet/blob/master/docs/installation.md) · [Configuration](https://github.com/l-forster/open-jet/blob/master/docs/configuration.md) ·
  [Choosing a model](https://github.com/l-forster/open-jet/blob/master/docs/models.md) · [Overview](https://github.com/l-forster/open-jet/blob/master/docs/overview.md)
- Coding tools: [Claude Code, Codex, OpenCode and OpenClaw](https://github.com/l-forster/open-jet/blob/master/docs/clients.md)
- Runtime: [OpenJet's engine](https://github.com/l-forster/open-jet/blob/master/megakernel/README.md) · [llama.cpp](https://github.com/l-forster/open-jet/blob/master/docs/runtimes/llama-cpp.md) ·
  [Benchmarking](https://github.com/l-forster/open-jet/blob/master/docs/benchmarking.md)
- Agent: [CLI](https://github.com/l-forster/open-jet/blob/master/docs/usage/cli.md) · [Slash commands](https://github.com/l-forster/open-jet/blob/master/docs/usage/slash-commands.md) ·
  [Slipstream](https://github.com/l-forster/open-jet/blob/master/docs/usage/slipstream.md) · [Skills](https://github.com/l-forster/open-jet/blob/master/docs/usage/skills.md) ·
  [MCP](https://github.com/l-forster/open-jet/blob/master/docs/usage/mcp.md) · [Device sources](https://github.com/l-forster/open-jet/blob/master/docs/usage/device-sources.md) ·
  [Workflow harness](https://github.com/l-forster/open-jet/blob/master/docs/usage/workflow-harness.md) ·
  [Session state and logging](https://github.com/l-forster/open-jet/blob/master/docs/usage/session-state-and-logging.md)
- SDK: [Quickstart](https://github.com/l-forster/open-jet/blob/master/docs/sdk/quickstart.md) · [Python SDK](https://github.com/l-forster/open-jet/blob/master/docs/sdk/python-sdk.md)
- For AI tools and writers: [AGENTS.md](https://github.com/l-forster/open-jet/blob/master/AGENTS.md) · [Install with an agent](https://github.com/l-forster/open-jet/blob/master/docs/agent-install.md) ·
  [AI reference](https://github.com/l-forster/open-jet/blob/master/docs/AI_REFERENCE.md) · [Media kit](https://github.com/l-forster/open-jet/blob/master/docs/MEDIA_KIT.md)
- Deployment: [Jetson](https://github.com/l-forster/open-jet/blob/master/docs/deployment/jetson.md) ·
  [Linux x86 + NVIDIA](https://github.com/l-forster/open-jet/blob/master/docs/deployment/linux-x86-nvidia.md) · [CPU-only](https://github.com/l-forster/open-jet/blob/master/docs/deployment/cpu-only.md)

## Community and license

[Discord](https://discord.gg/pspKHtExSa) · [X / Twitter](https://x.com/FlouisLF) · `AGPL-3.0-only` ([licensing](https://github.com/l-forster/open-jet/blob/master/docs/licensing.md), [CONTRIBUTING.md](https://github.com/l-forster/open-jet/blob/master/CONTRIBUTING.md), [CLA.md](https://github.com/l-forster/open-jet/blob/master/CLA.md))
