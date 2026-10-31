# OpenJet: run Qwen3.8-27B on your own GPU, 1.5x faster than llama.cpp

OpenJet sets up Qwen3.8-27B, or another open model that fits your hardware, and serves it to your
coding tools: Claude Code, Codex, OpenCode, OpenClaw, OpenJet's own terminal agent, or your
application. On NVIDIA RTX 30, 40 and 50-series GPUs with 24 GB, it serves Qwen3.8-27B with its own
CUDA engine. Local mode needs no account, API key or hosted service, and works offline once the
model is installed.

[GitHub repository](https://github.com/l-forster/open-jet) ·
[Documentation](https://github.com/l-forster/open-jet/blob/master/docs/getting-started.md) ·
[Coding tools](https://github.com/l-forster/open-jet/blob/master/docs/clients.md) ·
[How the engine works](https://github.com/l-forster/open-jet/tree/master/megakernel) ·
[Discord](https://discord.gg/pspKHtExSa)

| RTX 3090, Qwen3.8-27B Q4_K_M | llama.cpp | OpenJet |
|---|---|---|
| Reasoning mode, short prompts (what coding agents use) | 60 tok/s | **92 tok/s** (1.5x) |
| Reasoning mode, 98K tokens in context | 37 tok/s | **79 tok/s** (2.2x) |
| Reading a 98K-token prompt | 112 s | **82 s** (1.4x) |
| Greedy, average of 4 prompts | 63 tok/s | **107 tok/s** (1.7x) |
| Greedy, writing code | 73 tok/s | **140 tok/s** (1.9x) |
| Reading your prompt, 2-8K tokens | 1050-1140 tok/s | **1530-1600 tok/s** |

Each figure is one conversation on one RTX 3090. llama.cpp runs with OpenJet's flags, including
MTP speculative decoding. The first three rows are the current engine (thinking on at xhigh,
temperature 1.0, 5 prompts x 4 seeds, and one 98K-token prompt); the others are from an earlier build.

## Quick Start

Linux and macOS:

```bash
curl -fsSL https://github.com/L-Forster/open-jet/releases/latest/download/install.sh | sh
```

Windows PowerShell:

```powershell
irm https://github.com/L-Forster/open-jet/releases/latest/download/install.ps1 | iex
```

The installer brings its own Python, so you don't need Python or pipx. Then, in a new terminal:

```bash
openjet setup      # finds your GPU, downloads the model and the engine built for it
openjet            # OpenJet's own coding agent
```

Already use Python? `pipx install open-jet` works too.

Or serve the model to the coding tool you already use:

```bash
openjet serve                     # OpenAI and Anthropic APIs on http://127.0.0.1:8080
openjet connect claude-code       # or codex, opencode, openclaw
```

| Tool | API | Set up with |
|---|---|---|
| Claude Code | Anthropic Messages (`/v1/messages`) | `openjet connect claude-code` |
| Codex CLI | OpenAI Responses (`/v1/responses`) | `openjet connect codex` |
| OpenCode | OpenAI Chat Completions | `openjet connect opencode` |
| OpenClaw | OpenAI Chat Completions | `openjet connect openclaw` |

Or paste this into Claude Code, Codex or another agent with terminal access:

> Install OpenJet on this machine by following https://github.com/l-forster/open-jet/blob/master/docs/agent-install.md. Inspect my hardware, choose a suitable local coding model, configure it without the setup wizard, verify inference works, and tell me how to use OpenJet.

To accept the hardware recommendation without the wizard, run `openjet setup --yes`.

## What you need

| Your hardware | What runs | Guide |
|---|---|---|
| NVIDIA RTX 30, 40 or 50-series with 24 GB, Linux | Qwen3.8-27B on OpenJet's engine | [Engine for your GPU](https://github.com/l-forster/open-jet/blob/master/docs/installation.md#prebuilt-engines) · [OpenJet's engine](https://github.com/l-forster/open-jet/blob/master/megakernel/README.md) |
| NVIDIA Turing with 24 GB or more, Linux | OpenJet's engine, experimental | [Engine for your GPU](https://github.com/l-forster/open-jet/blob/master/docs/installation.md#prebuilt-engines) · [Turing](https://github.com/l-forster/open-jet/blob/master/megakernel/README.md#turing-rtx-8000) |
| Windows | llama.cpp; OpenJet's engine on Windows is in progress | [Windows install](https://github.com/l-forster/open-jet/blob/master/docs/installation.md#one-line-install) |
| Any other GPU, or Apple silicon | The best model that fits, on llama.cpp | [Pick by memory](https://github.com/l-forster/open-jet/blob/master/docs/models.md#pick-by-memory) |
| Jetson | llama.cpp | [Jetson](https://github.com/l-forster/open-jet/blob/master/docs/deployment/jetson.md) |
| CPU only | llama.cpp | [CPU-only](https://github.com/l-forster/open-jet/blob/master/docs/deployment/cpu-only.md) |

The engine's model is unsloth's [Qwen3.8-27B-Q4_K_M.gguf](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/blob/566347e99d832d31f6245a577198d10d5bdd389e/Qwen3.8-27B-Q4_K_M.gguf), which `openjet setup` downloads for
you. [All requirements](https://github.com/l-forster/open-jet/blob/master/docs/installation.md#what-you-need) · [Every model file](https://github.com/l-forster/open-jet/blob/master/docs/models.md)

## CLI

The primary entrypoint is the `openjet` command:

```bash
openjet                     # start interactive session
openjet setup               # provision a coding model for this machine
openjet project             # provision the model your application ships with
openjet bench               # measure this machine and write a results file to share
openjet benchmark --sweep   # tune llama.cpp settings with llama-bench
```

The CLI is a full terminal agent that can:

- **Work with local files** — search, read, create, and update files in your projects
- **Run shell commands** — with explicit approval before execution
- **Run workflows and checks** — inspect results, update files, and try again against the local model
- **Resume sessions** — close the terminal and pick up where you left off
- **Work on constrained hardware** — automatic context condensing and model unload/reload around heavy tasks
- **Connect to devices** — cameras, microphones, GPIO, and remote devices for edge and embedded workflows
- **Connect tools** — trusted MCP servers are exposed through the normal OpenJet tool registry

## Embed a local LLM in your own application

Provision the model once, at build time, in the project that will ship it:

```bash
cd your-project
openjet project --use-case dialogue --target handheld --budget 4
```

Selection is driven by what the model is for, the device you ship to (declared, not
detected — the machine you build on is not the machine your users have), and the memory
your application can concede. The weights land in `.openjet/models/` so your build can
bundle them, and `.openjet/config.yaml` pins them for the SDK.

```python
import asyncio

from openjet.sdk import create_inference_session


async def main() -> None:
    session = await create_inference_session()
    try:
        result = await session.run("Summarise this support ticket in one line.")
        print(result.text)
    finally:
        await session.close()


asyncio.run(main())
```

`create_inference_session()` refuses every tool — text in, text out — so an embedded model
cannot reach the shell or the filesystem no matter what it generates. Use `create_agent()`
when you do want the full agent surface.

Nothing downloads at runtime. A missing model raises immediately rather than reaching for
the network on a user's machine, so your users never fetch a file, see a config, or need
an internet connection.

## Python SDK

Use `openjet.sdk` to embed the same runtime in your own Python application.

```python
from openjet.sdk import OpenJetSession, create_inference_session, recommend_hardware_config
```

### Session API

Run agent sessions programmatically:

```python
import asyncio
from openjet.sdk import OpenJetSession


async def main() -> None:
    session = await OpenJetSession.create()
    try:
        result = await session.run("Summarize the current README")
        print(result.text)
    finally:
        await session.close()


asyncio.run(main())
```

The session API exposes:

- `OpenJetSession.create(...)` — create a session
- `session.run(...)` — run a prompt and get the result
- `session.stream(...)` — stream events as they arrive
- `session.set_airgapped(...)` — toggle air-gapped mode
- `session.add_turn_context(...)` / `session.clear_turn_context(...)` — manage turn context
- `create_agent(...)` — construct an agent instance

### Hardware Recommendation API

Get model and runtime recommendations for your hardware:

```python
from openjet.sdk import recommend_hardware_config

result = recommend_hardware_config(
    {
        "total_ram_gb": 16,
        "gpu": "cuda",
        "vram_mb": 24576,
        "label": "RTX 4090 box",
    }
)

print(result.model.label)
print(result.model.target_path)
print(result.llama.device)
print(result.llama.gpu_layers)
print(result.llama.context_window_tokens)
```

Typed input is also supported:

```python
from openjet.sdk import HardwareRecommendationInput, recommend_hardware_config

result = recommend_hardware_config(
    HardwareRecommendationInput(
        total_ram_gb=8.0,
        gpu="cpu",
        hardware_profile="other",
        hardware_override="desktop_8",
    )
)
```

### SDK Surface

The supported public SDK surface:

```python
from openjet.sdk import (
    HardwareRecommendation,
    HardwareRecommendationInput,
    OpenJetSession,
    RecommendedLlamaConfig,
    RecommendedModel,
    SDKEvent,
    SDKEventKind,
    SDKResponse,
    ToolResult,
    create_agent,
    create_inference_session,
    recommend_hardware_config,
)
```

## Package Contents

This wheel includes the full OpenJet package:

- `openjet.sdk` — Python integrations
- CLI entrypoint `openjet` (installed as `open-jet` on PyPI)
- Benchmark entrypoints via `openjet bench` and `openjet benchmark`
- The local/session runtime shared by the SDK and CLI
- OpenJet's Qwen3.8-27B engine server and source (`megakernel/`); `openjet setup` downloads the
  prebuilt library for your GPU, or compiles it when none matches

## Frequently asked questions

**Can I use OpenJet with Ollama or LM Studio?** Yes. Install `open-jet[cloud]` for the
LiteLLM integration and configure an OpenAI-compatible local server profile. See
[local server configuration](https://github.com/l-forster/open-jet#use-ollama-lm-studio-or-an-existing-llama-server).
The default managed llama.cpp integration does not require this extra.

**Does OpenJet work offline?** Yes. After installing the package, runtime and model,
local inference works offline. Set `airgapped: true` to block non-loopback model and tool
endpoints. Cloud modes are optional.

**Which hardware can run OpenJet?** A GPU with 14 GB+ VRAM or Apple silicon with 24 GB+
unified memory is recommended for the coding agent. CPU-only inference is also supported.
OpenJet's engine requires Linux, a supported RTX 30, 40 or 50-series GPU with 24 GB+,
a recent NVIDIA driver and a compatible Qwen3.8-27B model.

**Can I use it without OpenJet's coding agent?** Yes. `openjet serve` serves the model to Claude
Code, Codex, OpenCode, OpenClaw or any OpenAI- or Anthropic-compatible client; see
[Coding tools](https://github.com/l-forster/open-jet/blob/master/docs/clients.md). The
[engine docs](https://github.com/l-forster/open-jet/tree/master/megakernel) cover the standalone
`openjet-engine` command and its HTTP API.

## Documentation and community

- [GitHub repository](https://github.com/l-forster/open-jet)
- [Coding tools: Claude Code, Codex, OpenCode, OpenClaw](https://github.com/l-forster/open-jet/blob/master/docs/clients.md)
- [Installation guide](https://github.com/l-forster/open-jet/blob/master/docs/installation.md)
- [Model selection and GGUF support](https://github.com/l-forster/open-jet/blob/master/docs/models.md)
- [Python SDK quickstart](https://github.com/l-forster/open-jet/blob/master/docs/sdk/quickstart.md)
- [Local inference benchmarks](https://github.com/l-forster/open-jet/blob/master/docs/benchmarking.md)
- [AI reference](https://github.com/l-forster/open-jet/blob/master/docs/AI_REFERENCE.md) and
  [media kit](https://github.com/l-forster/open-jet/blob/master/docs/MEDIA_KIT.md), for AI tools and writers
- [Issues](https://github.com/l-forster/open-jet/issues)
- [Discord](https://discord.gg/pspKHtExSa)

## License

OpenJet core is licensed under `AGPL-3.0-only`.

This package covers the AGPL-licensed core SDK and CLI. Future hosted, team, or enterprise offerings may be licensed separately.

External contributions are accepted under the contributor terms in the repository's `CONTRIBUTING.md` and `CLA.md`.
