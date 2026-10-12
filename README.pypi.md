# OpenJet — Local AI Coding Agent and LLM Inference Engine

OpenJet is an open-source, self-hosted AI coding assistant and local LLM runtime. Run a
coding agent in your terminal with llama.cpp, Ollama or LM Studio, embed an on-device model
with the Python SDK, or serve Qwen3.8-27B with the standalone CUDA inference engine.

The `open-jet` Python package installs the `openjet` CLI, `openjet.sdk` and the
`openjet-engine` server. Local mode needs no account, API key or hosted service.
Once your model is installed, it works offline.

[GitHub repository](https://github.com/l-forster/open-jet) ·
[Documentation](https://github.com/l-forster/open-jet/blob/master/docs/getting-started.md) ·
[OpenJet Engine](https://github.com/l-forster/open-jet/tree/master/megakernel) ·
[Discord](https://discord.gg/pspKHtExSa)

Install from PyPI:

```bash
pip install open-jet
```

## Local AI coding assistant, inference server and Python SDK

- **Terminal coding agent** — read and edit files, run approved shell commands, use MCP
  tools, and resume sessions on your machine.
- **Hardware-aware setup** — detect GPU and memory, download a GGUF model that fits,
  and configure local inference without assembling a separate AI stack.
- **Local server support** — connect to Ollama, LM Studio, llama.cpp and other
  OpenAI-compatible servers.
- **CUDA inference engine** — serve Qwen3.8-27B on supported NVIDIA RTX GPUs through
  the standalone OpenJet Engine, with an HTTP API for your own applications.
- **Python SDK** — embed local LLM inference, provision on-device models at build time,
  and automate agent sessions.
- **Benchmarking tools** — measure prompt processing and generation on your hardware.

## Quick Start

Use your coding agent to install OpenJet. Paste this into Codex, Claude Code, or another agent with terminal access:

> Install OpenJet on this machine by following https://github.com/l-forster/open-jet/blob/master/docs/agent-install.md. Inspect my hardware, choose a suitable local coding model, configure it without the setup wizard, verify inference works, and tell me how to use OpenJet.

Your agent checks GPU and memory, downloads a model that fits, and verifies the local runtime. [Agent installation instructions](https://github.com/l-forster/open-jet/blob/master/docs/agent-install.md).

For manual installation:

```bash
pip install open-jet
openjet setup
openjet
```

To accept the hardware recommendation without the wizard, run `openjet setup --yes`.

`openjet setup` detects your hardware, picks a model that fits your RAM, downloads it, and gets everything running. Already have a `.gguf`? It finds that too.

Recommended hardware: Apple silicon with 24GB+ unified memory, or a GPU with 14GB+ VRAM.

## OpenJet Engine: CUDA LLM inference on NVIDIA RTX GPUs

For Qwen3.8-27B Q4_K_M on Linux with an RTX 30, 40 or 50-series GPU that has 24GB or more, OpenJet serves the model through its own CUDA engine instead of `llama-server`. On an RTX 3090 it decodes at 1.5x llama.cpp's speed in the agent's reasoning mode (78 vs 51 tok/s) and up to 1.9x when writing code (140 vs 73 tok/s).

The engine ships in this package as source, from version 0.4.32. `openjet setup` compiles it for your GPU:

```bash
pip install --upgrade open-jet
openjet setup        # choose "Qwen3.8 27B Q4_K_M MTP"
openjet
```

You need the CUDA toolkit installed, for `nvcc` (CUDA 12 or newer; 12.8 or newer for RTX 50). The compile takes about 2 minutes and ends with `Megakernel built: OpenJet will serve Qwen3.8-27B with it.` If you set OpenJet up before 0.4.32, upgrade and run `openjet setup` again.

If setup can't build it, it prints `Skipping the Qwen3.8-27B megakernel (...)` with the reason and OpenJet uses `llama-server` instead, so setup still finishes. While OpenJet is running, `pgrep -af mk_server.py` prints a line when the megakernel is serving.

It has been benchmarked on an RTX 3090 only. Benchmarks and details: [megakernel README](https://github.com/l-forster/open-jet/tree/master/megakernel).

## CLI

The primary entrypoint is the `openjet` command:

```bash
openjet                     # start interactive session
openjet setup               # provision a coding model for this machine
openjet project             # provision the model your application ships with
openjet benchmark --sweep   # run a hardware benchmark sweep
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
- Benchmark entrypoints via `openjet benchmark`
- The local/session runtime shared by the SDK and CLI
- Source for the Qwen3.8-27B megakernel (`megakernel/`), compiled by `openjet setup`

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
The CUDA engine requires Linux, a supported RTX 30, 40 or 50-series GPU with 24 GB+,
the CUDA toolkit and a compatible Qwen3.8-27B model.

**Can I use the inference engine without the coding agent?** Yes. The
[OpenJet Engine quickstart](https://github.com/l-forster/open-jet/tree/master/megakernel)
covers standalone installation, the `openjet-engine` command and HTTP API usage.

## Documentation and community

- [GitHub repository](https://github.com/l-forster/open-jet)
- [Installation guide](https://github.com/l-forster/open-jet/blob/master/docs/installation.md)
- [Model selection and GGUF support](https://github.com/l-forster/open-jet/blob/master/docs/models.md)
- [Python SDK quickstart](https://github.com/l-forster/open-jet/blob/master/docs/sdk/quickstart.md)
- [Local inference benchmarks](https://github.com/l-forster/open-jet/blob/master/docs/benchmarking.md)
- [Issues](https://github.com/l-forster/open-jet/issues)
- [Discord](https://discord.gg/pspKHtExSa)

## License

OpenJet core is licensed under `AGPL-3.0-only`.

This package covers the AGPL-licensed core SDK and CLI. Future hosted, team, or enterprise offerings may be licensed separately.

External contributions are accepted under the contributor terms in the repository's `CONTRIBUTING.md` and `CLA.md`.
