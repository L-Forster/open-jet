# OpenJet
<img height="225" alt="image" src="https://github.com/user-attachments/assets/660f7969-b38b-4a90-8b2c-ae872105cc13" />

**OpenJet is a local `llama.cpp` agent runtime with real system and device I/O.**

`open-jet` is for running local GGUF models that need to interact with shell workflows, files, cameras, microphones, GPIO, and simple sensors without turning into a brittle demo.

## Why use OpenJet

- local GGUF models with real tool access
- explicit approvals for risky actions
- session resume and KV-cache recovery
- low-memory model unload/reload around heavy shell work
- hardware-aware setup for local Linux and Jetson-style systems
- the same local runtime from the TUI and Python SDK

## Quickstart

```bash
git clone https://github.com/l-forster/open-jet.git
cd open-jet
./install.sh
open-jet --setup
open-jet
```

Setup is local-only and sequential. It will:

1. detect hardware
2. let you pick an existing `.gguf` or download a recommended one
3. reuse or build `llama-server`
4. save the final flat config in `config.yaml`

Automatic provisioning still needs normal local prerequisites:

- network access for model download and `llama.cpp` clone/build
- `git`
- `cmake`
- a working compiler toolchain

## Config

The canonical local config fields are:

- `llama_model`
- `llama_server_path`
- `device`
- `gpu_layers`
- `context_window_tokens`
- `setup_recommendations.direct_models`
- `active_model_profile`
- `model_profiles`

See [Configuration](docs/configuration.md) for examples.

## What it provides

- bounded local inference through `llama-server`
- controlled tool use and slash commands
- automatic context condensing under pressure
- resumable chat/session state
- OpenTelemetry instrumentation with optional collector export
- device registry and workflow support for local machines

## Docs

- [Quickstart](docs/quickstart.md)
- [Installation](docs/installation.md)
- [Configuration](docs/configuration.md)
- [Runtime: llama.cpp](docs/runtimes/llama-cpp.md)
- [Python SDK](docs/sdk/python-sdk.md)
- [Usage: CLI](docs/usage/cli.md)
- [Usage: Device sources](docs/usage/device-sources.md)
- [Usage: Slash commands](docs/usage/slash-commands.md)
- [Usage: Workflow harness](docs/usage/workflow-harness.md)
- [Usage: Session state and logging](docs/usage/session-state-and-logging.md)
- [Examples](docs/examples/README.md)
- [Telemetry](docs/telemetry.md)
- [Deployment: Jetson](docs/deployment/jetson.md)
- [Deployment: Linux x86 + NVIDIA](docs/deployment/linux-x86-nvidia.md)
- [Deployment: CPU-only](docs/deployment/cpu-only.md)

## License

`AGPL-3.0-only`, with commercial licensing available under separate terms.



