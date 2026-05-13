# OpenJet

## Claude Code for local LLMs

**The easiest and fastest way to get a local coding agent running.**

OpenJet sets up the local model backend for your hardware and gives you a Claude-Code-style coding agent that can read files, edit code, and run commands fully on your own machine.

[Discord](https://discord.gg/pspKHtExSa)

If you are new to local LLMs, OpenJet is the fastest way to get started without spending hours figuring out models, runtimes, and config.

If you have already tried local LLMs and got frustrated piecing together a model backend, a frontend, and an actual coding agent, OpenJet removes that setup tax.

OpenJet is built for people looking for a **Claude Code alternative, easy local LLM setup, or a self-hosted local coding agent.**

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
openjet                  # start interactive session
openjet benchmark --sweep   # run a hardware benchmark sweep
```

The CLI is a full terminal agent that can:

- **Read and edit your code** — search files, apply edits, write new ones
- **Run shell commands** — with explicit approval before execution
- **Resume sessions** — close the terminal and pick up where you left off
- **Work on constrained hardware** — automatic context condensing and model unload/reload around heavy tasks
- **Connect to devices** — cameras, microphones, GPIO, and remote devices for edge and embedded workflows
- **Connect MCP tools** — optional trusted MCP servers are exposed through the normal OpenJet tool registry

## Python SDK

Use `openjet.sdk` to embed the same runtime in your own Python application.

```python
from openjet.sdk import OpenJetSession, recommend_hardware_config
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

OpenJet core is licensed under `Apache-2.0`.

This package covers the permissive core SDK and CLI. Future hosted, team, or enterprise offerings may be licensed separately.

External contributions are accepted under the contributor terms in the repository's `CONTRIBUTING.md` and `CLA.md`.
