# Product Surfaces

New to OpenJet? [Getting started](getting-started.md) forks by what you are building —
an agent in your terminal, or a model inside your own application.

OpenJet is one product with three primary surfaces.

## 1. CLI + Chat TUI

Use this when you want an interactive local coding agent in your terminal.

Primary entrypoints:

```bash
openjet
openjet chat
openjet chat Explain the repo layout briefly.
```

This surface is for:

- individual developers using OpenJet directly
- local-first coding and debugging sessions
- device-aware chat and workflow execution

Related docs:

- [CLI usage](usage/cli.md)
- [Slash commands](usage/slash-commands.md)
- [Hybrid mode](usage/hybrid-mode.md)
- [Device sources](usage/device-sources.md)
- [Workflow harness](usage/workflow-harness.md)

## 2. Python SDK

Use this when you want to embed OpenJet in another Python app, worker, toolchain,
or agent system.

Primary entrypoints:

```bash
openjet project    # build-time: provision the model this application ships with
```

```python
from openjet.sdk import (
    OpenJetSession,
    create_agent,
    create_inference_session,
    recommend_hardware_config,
)
```

This surface is for:

- shipping an on-device model inside an application of your own — an in-app assistant, an
  NPC that talks, a classifier, an extraction pipeline
- embedding bounded-memory local sessions in another app
- reusing OpenJet's approval and tool-execution model
- hardware profiling and auto-`llama.cpp` configuration

Model acquisition belongs to `openjet project` and happens once, at build time. Nothing in
the SDK downloads at runtime; a missing model raises instead.

Related docs:

- [SDK quickstart](sdk/quickstart.md)
- [Python SDK](sdk/python-sdk.md)
- [Choosing a model](models.md)
- [Configuration](configuration.md)
- [Runtime: llama.cpp](runtimes/llama-cpp.md)

## 3. Benchmarking

Use this when you want to measure throughput and compare runtime settings for the
currently configured model profile.

Primary entrypoints:

```bash
openjet benchmark
openjet benchmark --sweep
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

- use `openjet` for the app
- import `openjet.sdk` to build on it
- run `openjet benchmark` to measure it

## When to split packages later

Keep one repo unless a surface grows into a clearly separate product.

The likely future split is packaging, not repositories:

- `openjet` for the main CLI + SDK distribution
- `openjet-benchmark` as an optional package if benchmarking grows a larger independent audience

If that happens, keep the same docs and brand architecture so the surfaces still
feel like parts of OpenJet rather than unrelated projects.
