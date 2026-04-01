# Product Surfaces

OpenJet is one product with three primary surfaces.

## 1. CLI + Chat TUI

Use this when you want an interactive local coding agent in your terminal.

Primary entrypoints:

```bash
open-jet
open-jet chat
open-jet chat Explain the repo layout briefly.
```

This surface is for:

- individual developers using OpenJet directly
- local-first coding and debugging sessions
- device-aware chat and workflow execution

Related docs:

- [CLI usage](usage/cli.md)
- [Slash commands](usage/slash-commands.md)
- [Device sources](usage/device-sources.md)
- [Workflow harness](usage/workflow-harness.md)

## 2. Python SDK

Use this when you want to embed OpenJet in another Python app, worker, toolchain,
or agent system.

Primary entrypoints:

```python
from openjet.sdk import OpenJetSession, create_agent, recommend_hardware_config
```

This surface is for:

- embedding bounded-memory local sessions in another app
- reusing OpenJet's approval and tool-execution model
- hardware profiling and auto-`llama.cpp` configuration

Related docs:

- [Python SDK](sdk/python-sdk.md)
- [Configuration](configuration.md)
- [Runtime: llama.cpp](runtimes/llama-cpp.md)

## 3. Benchmarking

Use this when you want to measure throughput and compare runtime settings for the
currently configured model profile.

Primary entrypoints:

```bash
open-jet benchmark
open-jet benchmark --sweep
```

This surface is for:

- prompt/gen throughput checks with `llama-bench`
- tuning GPU layers, batch size, and thread count
- comparing local hardware and model presets

Related docs:

- [Benchmarking](benchmarking.md)
- [Configuration](configuration.md)
- [Runtime: llama.cpp](runtimes/llama-cpp.md)

## Why one repo

OpenJet keeps these surfaces in one repo because they share:

- the same local runtime model configuration
- the same `llama.cpp` integration
- the same hardware detection and model-profile logic
- the same session/runtime implementation used by both CLI and SDK

That keeps the product coherent for users:

- use `open-jet` for the app
- import `openjet.sdk` to build on it
- run `open-jet benchmark` to measure it

## When to split packages later

Keep one repo unless a surface grows into a clearly separate product.

The likely future split is packaging, not repositories:

- `open-jet` for the main CLI + SDK distribution
- `openjet-benchmark` as an optional package if benchmarking grows a larger independent audience

If that happens, keep the same docs and brand architecture so the surfaces still
feel like parts of OpenJet rather than unrelated projects.
