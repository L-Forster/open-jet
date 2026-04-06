# Open-Jet

[GitHub](https://github.com/l-forster/open-jet) | [X / Twitter](https://x.com/FlouisLF) | [Discord](https://discord.gg/pspKHtExSa)

**An AI coding agent that runs entirely on your machine.**

  This is Claude Code for local LLMs. OpenJet handles the model, runtime, and setup for you, so you can run a terminal coding agent on your own machine without fighting configuration. It reads files, edits code, runs commands, and keeps your work out of the cloud.
  
`open-jet` ships the full OpenJet package:

- the CLI and chat TUI
- the Python SDK
- the benchmarking entrypoints

Install it with:

```bash
pip install open-jet
```

OpenJet handles the model, runtime, and local setup without making you wire together `llama.cpp`, model files, and device-specific settings by hand. You get a coding agent in your terminal that can read files, edit code, run commands, and stay local.

This is not an SDK-only wheel. It installs the full OpenJet package, with `openjet.sdk` exposed as a supported import surface.

## Product Surfaces

OpenJet has three primary surfaces in one package:

- **CLI + chat TUI** for interactive local agent work
- **Python SDK** for embedded sessions, hardware profiling, and auto-configuration
- **Benchmarking** for `llama-bench` runs and sweep comparisons

Typical entrypoints:

```bash
open-jet
open-jet benchmark --sweep
```

```python
from openjet.sdk import OpenJetSession, recommend_hardware_config
```

## What You Get

- **Read and edit local code** from a terminal agent
- **Run shell commands** through the same session flow
- **Resume sessions** instead of losing context when the terminal closes
- **Work on constrained hardware** with hardware-aware model selection
- **Use the Python SDK** to embed the same runtime in your own app
- **Run benchmarks** against your current model/runtime profile

## SDK Import Path

Use:

```python
from openjet.sdk import recommend_hardware_config
```

or:

```python
from openjet.sdk import OpenJetSession, create_agent
```

## SDK Surface

The supported SDK surface includes:

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

That covers two main use cases:

- hardware/model recommendation for local `llama.cpp` setups
- embedded session/chat usage from your own Python application

## Hardware Recommendation API

`recommend_hardware_config()` takes hardware input and returns:

- a recommended model
- recommended llama device settings
- recommended GPU layer count
- recommended context window size
- a token generation estimate for the recommended setup

Example:

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

Typed input also works:

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

## Session API

Use `OpenJetSession` when you want to embed OpenJet into another Python service, worker, or app.

Basic example:

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

The session API includes:

- `OpenJetSession.create(...)`
- `session.stream(...)`
- `session.run(...)`
- `session.set_airgapped(...)`
- `session.add_turn_context(...)`
- `session.clear_turn_context(...)`
- `create_agent(...)`

The event/response types exposed for integrations are:

- `SDKEvent`
- `SDKEventKind`
- `SDKResponse`
- `ToolResult`

## CLI

This package also installs the CLI:

```bash
openjet
```

or:

```bash
open-jet
```

The CLI and the SDK share the same underlying package and runtime code.

## Package Contents

This wheel currently includes:

- `openjet.sdk` for Python integrations
- CLI entrypoints: `openjet` and `open-jet`
- benchmark entrypoints via `open-jet benchmark`
- the local/session runtime used by both the SDK and the CLI

If you only need one narrow SDK feature, the package still installs the full declared dependency set for this distribution.

## Repository

- Repository: [github.com/l-forster/open-jet](https://github.com/l-forster/open-jet)
- Issues: [github.com/l-forster/open-jet/issues](https://github.com/l-forster/open-jet/issues)

## License

`open-jet` core is licensed under `Apache-2.0`.

This package covers the permissive core SDK and CLI. Any future hosted, team,
or enterprise offerings may be licensed separately.

External contributions are accepted under the contributor terms in
the repository's `CONTRIBUTING.md` and `CLA.md`.
