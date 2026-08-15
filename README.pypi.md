# OpenJet

## Self-hosted local AI coding agent for Ollama, LM Studio, and llama.cpp

**Use your existing local LLM server, or let OpenJet provision and host the right GGUF
model for your hardware. Then use the same runtime through the terminal agent or Python SDK.**

OpenJet is an open-source, private AI coding assistant that reads and edits files, runs
approved shell commands, uses tools, and keeps sessions on your machine. It connects to
Ollama, LM Studio, llama.cpp, and other OpenAI-compatible local servers. If you do not
already have a local AI stack, `openjet setup` downloads, configures, and serves one for you.

RTX 3090 + Qwen 27B: **33 tok/s -> 70 tok/s** with MTP. Open source.

[Discord](https://discord.gg/pspKHtExSa)

**In your terminal**, OpenJet is a self-hosted coding agent:

```text
files -> tools -> shell approval -> workflows -> local model
```

**In your code**, it is a way to ship an on-device model inside your own application —
an in-app assistant, an NPC that talks, a classifier, an extraction pipeline. Run
`openjet project` in your project at build time and the model is chosen for your target
device, downloaded into the project, and bundled with your build. The SDK never
downloads at runtime, so your users never fetch a file or see a config.

No hosted API is required. With a local profile, no code or prompt data leaves your machine.

RTX 3090 + Qwen 27B: **33 tok/s -> 70 tok/s** with MTP.

Open source.

If you are new to local LLMs, OpenJet is the fastest way to get started without spending hours figuring out models, runtimes, and config. If you have already tried local LLMs and got frustrated piecing together a model backend, a frontend, and an actual agent workflow, OpenJet removes that setup tax.

This is the PyPI package for OpenJet. Install it with:

```bash
pip install open-jet
```

The package installs:

- **CLI + chat TUI** — interactive local agent work in the terminal
- **Python SDK** — embed sessions, profile hardware, and automate workflows
- **Benchmarking tools** — measure prompt and generation performance on your hardware

## Quick Start

```bash
pip install open-jet
openjet setup
openjet
```

`openjet setup` detects your hardware, picks a model that fits your RAM, downloads it, and gets everything running. Already have a `.gguf`? It finds that too.

Recommended hardware: Apple silicon with 24GB+ unified memory, or a GPU with 14GB+ VRAM.

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

## Repository

- [github.com/l-forster/open-jet](https://github.com/l-forster/open-jet)
- [Issues](https://github.com/l-forster/open-jet/issues)
- [Discord](https://discord.gg/pspKHtExSa)

## License

OpenJet core is licensed under `AGPL-3.0-only`.

This package covers the AGPL-licensed core SDK and CLI. Future hosted, team, or enterprise offerings may be licensed separately.

External contributions are accepted under the contributor terms in the repository's `CONTRIBUTING.md` and `CLA.md`.
