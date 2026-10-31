<h1 align="center"><img src="https://raw.githubusercontent.com/L-Forster/open-jet/master/docs/assets/logo.png" width="498" alt="OpenJet"></h1>

<p align="center"><b>Run Qwen3.8-27B on your own GPU, 1.5x faster than llama.cpp.</b></p>

<p align="center">For NVIDIA RTX 30, 40 and 50-series cards with 24 GB. Use it with Claude Code, Codex, OpenCode,
OpenClaw or OpenJet's own agent. Runs offline.</p>

<p align="center">
  <a href="https://github.com/L-Forster/open-jet#speed">Speed</a> · <a href="https://github.com/L-Forster/open-jet#install">Install</a> · <a href="https://github.com/L-Forster/open-jet#what-you-need">Hardware</a> ·
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

## Speed

| RTX 3090, Qwen3.8-27B Q4_K_M | llama.cpp | OpenJet |
|---|---|---|
| Thinking on, short prompts | 60 tok/s | **92 tok/s** (1.5x) |
| Thinking on, 98K tokens in context | 37 tok/s | **79 tok/s** (2.2x) |
| Reading a 98K-token prompt | 112 s | **82 s** (1.4x) |
| Greedy, average of 4 prompts | 63 tok/s | **107 tok/s** (1.7x) |
| Greedy, writing code | 73 tok/s | **140 tok/s** (1.9x) |
| Reading a 2-8K-token prompt | 1050-1140 tok/s | **1530-1600 tok/s** |

- One conversation on one RTX 3090, not a batch.
- llama.cpp b11982 runs with the same flags, including MTP speculative decoding.
- Rows 1-3: current engine, thinking at xhigh, temperature 1.0, 5 prompts x 4 seeds and one 98K-token prompt x 4 seeds
  ([`run_compare.sh`](https://github.com/l-forster/open-jet/blob/master/megakernel/bench/run_compare.sh)). Other rows: an earlier engine build.
- Same output as llama.cpp on the same weights: KL divergence 0.0009, top-token agreement 99.08%
  ([results](https://github.com/l-forster/open-jet/blob/master/megakernel/README.md#results)).

Context on a 24 GB card: about **120K tokens**, or about **234K** with `openjet setup --long-context`
([KV cache presets](https://github.com/l-forster/open-jet/blob/master/megakernel/README.md#kv-cache-presets-mk_kv)).

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
| NVIDIA RTX 30, 40 or 50-series with 24 GB, Linux | Qwen3.8-27B on OpenJet's engine | [Engine for your GPU](https://github.com/l-forster/open-jet/blob/master/docs/installation.md#prebuilt-engines) |
| NVIDIA Turing with 24 GB or more, Linux | OpenJet's engine, experimental | [Turing](https://github.com/l-forster/open-jet/blob/master/megakernel/README.md#turing-rtx-8000) |
| Windows | llama.cpp; OpenJet's engine on Windows is in progress | [Installation](https://github.com/l-forster/open-jet/blob/master/docs/installation.md) |
| Any other GPU, or Apple silicon | The best model that fits, on llama.cpp | [Pick by memory](https://github.com/l-forster/open-jet/blob/master/docs/models.md#pick-by-memory) |
| Jetson | llama.cpp | [Jetson](https://github.com/l-forster/open-jet/blob/master/docs/deployment/jetson.md) |
| CPU only | llama.cpp | [CPU-only](https://github.com/l-forster/open-jet/blob/master/docs/deployment/cpu-only.md) |

## [4 coding agents on one GPU](https://github.com/l-forster/open-jet/blob/master/docs/features.md#4-coding-agents-on-one-gpu)

https://github.com/user-attachments/assets/0df36ad0-b484-463e-a4f7-530e5a768cd9

Up to 4 agents decode in the same pass over the weights: about 290 tok/s in total on an RTX 3090 at
short context, 150-200 tok/s with real coding agents.

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
