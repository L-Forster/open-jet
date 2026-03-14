# OpenJet
<img width="286" height="114" alt="Screenshot_20260227_144411" src="https://github.com/user-attachments/assets/70d90ca5-5841-44d8-a1d0-ef3e08acd95b" />

`open-jet` is an offline-first agent runtime for Jetson-class and other edge Linux systems with tight memory budgets.

It is built for cases where the hard part is not just running a local model, but keeping the agent useful under constrained RAM, limited context windows, interrupted sessions, and hardware-specific failure modes.

## Quickstart

```bash
pip install open-jet
open-jet --setup
```

## What it provides

- bounded-context local chat with your on-device model
- automatic context condensing under pressure
- session resume and harness state recovery
- replayable JSONL event traces
- hardware-aware runtime setup for Jetson and edge Linux
- controlled tool use and slash commands
- Python SDK access to the same backend

## Why OpenJet exists

Most local LLM tools stop at “chat with a model on your box.” That breaks down on edge hardware.

OpenJet is built around:

- limited prompt memory on-device
- interrupted work and session recovery
- deterministic tool and approval boundaries
- real traces for evaluation
- reliable operator workflows on constrained local models

## Docs

- [Quickstart](docs/quickstart.md)
- [Installation](docs/installation.md)
- [Runtime: llama.cpp](docs/runtimes/llama-cpp.md)
- [Runtime: SGLang](docs/runtimes/sglang.md)
- [Runtime: TensorRT-LLM](docs/runtimes/tensorrt-llm.md)
- [Usage: CLI](docs/usage/cli.md)
- [Usage: Slash commands](docs/usage/slash-commands.md)
- [Usage: Workflow harness](docs/usage/workflow-harness.md)
- [Usage: Session state and logging](docs/usage/session-state-and-logging.md)
- [Python SDK](docs/sdk/python-sdk.md)
- [Deployment: Jetson](docs/deployment/jetson.md)
- [Deployment: Linux x86 + NVIDIA](docs/deployment/linux-x86-nvidia.md)
- [Deployment: CPU-only](docs/deployment/cpu-only.md)
- [Configuration](docs/configuration.md)
- [Licensing](docs/licensing.md)

## License

`AGPL-3.0-only`, with commercial licensing available under separate terms.
