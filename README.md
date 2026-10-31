<h1 align="center">OpenJet</h1>

<p align="center"><b>Run Qwen3.8-27B on your own GPU, 1.5x faster than llama.cpp.</b><br>
RTX 30, 40 and 50-series with 24 GB · works with Claude Code, Codex, OpenCode and OpenClaw · offline · open source</p>

<p align="center">
  <a href="https://github.com/L-Forster/open-jet/stargazers"><img src="https://img.shields.io/github/stars/L-Forster/open-jet" alt="Stars"></a>
  <a href="https://pypi.org/project/open-jet/"><img src="https://img.shields.io/pypi/v/open-jet" alt="PyPI"></a>
  <a href="LICENSE"><img src="https://img.shields.io/github/license/L-Forster/open-jet" alt="License"></a>
  <a href="https://discord.com/invite/pspKHtExSa"><img src="https://img.shields.io/badge/Discord-join-5865F2" alt="Discord"></a>
</p>



https://github.com/user-attachments/assets/c9ac0b21-67b1-4f20-af70-21741accc774


| RTX 3090, Qwen3.8-27B Q4_K_M | llama.cpp | OpenJet |
|---|---|---|
| Reasoning mode, short prompts (what coding agents use) | 60 tok/s | **92 tok/s** (1.5x) |
| Reasoning mode, 98K tokens in context | 37 tok/s | **79 tok/s** (2.2x) |
| Reading a 98K-token prompt | 112 s | **82 s** (1.4x) |
| Greedy, average of 4 prompts | 63 tok/s | **107 tok/s** (1.7x) |
| Greedy, writing code | 73 tok/s | **140 tok/s** (1.9x) |
| Reading your prompt, 2-8K tokens | 1050-1140 tok/s | **1530-1600 tok/s** |

Each figure is one conversation on one RTX 3090, not a batch. llama.cpp runs with OpenJet's flags,
including MTP speculative decoding. The first three rows are the current engine against llama.cpp
b11982, measured with [`megakernel/bench/run_compare.sh`](megakernel/bench/run_compare.sh): thinking on
at xhigh, temperature 1.0, 5 prompts x 4 seeds, and one 98K-token prompt x 4 seeds. The other rows are
from an earlier engine build. OpenJet's output matches llama.cpp's on the same weights: KL divergence 0.0009, top-token
agreement 99.08% ([results and reproduction scripts](megakernel/README.md#results)).

## Install

```bash
pip install open-jet
openjet setup      # finds your GPU, downloads the model and OpenJet's prebuilt engine for it
openjet            # OpenJet's own coding agent
openjet serve      # or serve the model to Claude Code, Codex, OpenCode or OpenClaw
```

More: [docs/installation.md](docs/installation.md)

## What you need

| Your hardware | What runs | Guide |
|---|---|---|
| NVIDIA RTX 30, 40 or 50-series with 24 GB, Linux | Qwen3.8-27B on OpenJet's engine | [Engine for your GPU](docs/installation.md#prebuilt-engines) |
| NVIDIA Turing with 24 GB or more, Linux | OpenJet's engine, experimental | [Turing](megakernel/README.md#turing-rtx-8000) |
| Windows | llama.cpp; OpenJet's engine on Windows is in progress | [Installation](docs/installation.md) |
| Any other GPU, or Apple silicon | The best model that fits, on llama.cpp | [Pick by memory](docs/models.md#pick-by-memory) |
| Jetson | llama.cpp | [Jetson](docs/deployment/jetson.md) |
| CPU only | llama.cpp | [CPU-only](docs/deployment/cpu-only.md) |

More: [docs/installation.md](docs/installation.md#what-you-need)

## 4 coding agents on one GPU

https://github.com/user-attachments/assets/a7fc5010-110d-45fd-9774-a42afa1f3eb9

Up to 4 agents decode in the same pass over the weights: about 290 tok/s in total on an RTX 3090 at
short context, 150-200 tok/s with real coding agents.

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

## Slipstream

A frontier model plans and reviews; your local model reads, edits and tests.

More: [docs/features.md](docs/features.md#slipstream-a-frontier-model-plans-your-local-model-works)

## Python SDK

Embed a local model in your application.

More: [docs/features.md](docs/features.md#python-sdk-embed-a-local-llm-in-your-application)

Also: [tuned to the hardware](docs/features.md#tuned-to-the-hardware) · [Ollama, LM Studio or llama-server](docs/features.md#use-ollama-lm-studio-or-an-existing-llama-server) · [FAQ](docs/features.md#faq)

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
