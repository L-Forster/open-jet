# open-jet

`open-jet` ships both:

- the OpenJet CLI and chat TUI
- the OpenJet Python SDK
- the OpenJet benchmarking entrypoints

Install it with:

```bash
pip install open-jet
```

## What Is In The Package

This package currently includes:

- `openjet.sdk` for Python integrations
- CLI entrypoints: `openjet` and `open-jet`
- benchmark entrypoints via `open-jet benchmark`
- the local/session runtime used by both the SDK and the CLI

So this is not an SDK-only wheel. It is the full OpenJet package, with the SDK exposed as a supported import surface.

## Product Surfaces

OpenJet is one package with three primary surfaces:

- **CLI + chat TUI** for interactive local agent use
- **Python SDK** for embedded sessions and hardware profiling
- **Benchmarking** for `llama-bench` runs and sweeps

Typical entrypoints:

```bash
open-jet
open-jet benchmark --sweep
```

```python
from openjet.sdk import OpenJetSession, recommend_hardware_config
```

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

The currently exported SDK surface is:

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

## Notes

- This wheel currently ships the wider OpenJet application, not just the SDK subset.
- The SDK is exposed through `openjet.sdk`.
- If you only need one narrow SDK feature, the package still installs the full declared dependency set for this distribution.

## Repository

- Repository: [github.com/l-forster/open-jet](https://github.com/l-forster/open-jet)
- Issues: [github.com/l-forster/open-jet/issues](https://github.com/l-forster/open-jet/issues)

## License

`open-jet` core is licensed under `Apache-2.0`.

This package covers the permissive core SDK and CLI. Any future hosted, team,
or enterprise offerings may be licensed separately.

External contributions are accepted under the contributor terms in
the repository's `CONTRIBUTING.md` and `CLA.md`.
