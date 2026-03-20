# OpenJet
<img height="225" alt="image" src="https://github.com/user-attachments/assets/660f7969-b38b-4a90-8b2c-ae872105cc13" />


`open-jet` is an offline-first local agent you can actually use for real work.

It gives you a practical agent interface on top of your own local model and runtime, without making you assemble chat, tools, session recovery, and prompt management yourself.

OpenJet is designed to stay useful when local AI gets messy: tight memory, short context windows, interrupted sessions, and hardware-specific runtime issues.

## Why use OpenJet

- start with a local or self-hosted runtime you control
- keep working without losing the thread when sessions get interrupted
- use tools with explicit approvals and predictable behavior
- run the same backend from the TUI or Python SDK
- avoid building your own glue layer around local inference

## Quickstart

Clone the repo and run the installer:

```bash
git clone https://github.com/l-forster/open-jet.git
cd open-jet
./install.sh
```

OpenJet is positioned for local and self-hosted use first. Hosted APIs are supported, but they are the fallback path, not the main story.

### Recommended start: local `llama.cpp`

If you want the intended OpenJet path, start with:

```bash
open-jet --setup
```

Choose `Local model: llama.cpp (GGUF)` and follow the prompts for:

1. model source
2. model path or Ollama model
3. context window size
4. GPU offload configuration

On the local path, setup can work behind the scenes:

- reuse an existing `llama-server` if one is already installed
- provision `llama.cpp` and build `llama-server` if it is missing
- use a local `.gguf`, resolve an installed Ollama model, pull an Ollama model, or download a recommended GGUF

Automatic provisioning still relies on normal local prerequisites:

- network access for downloads and git clone operations
- `git`, `cmake`, and a working build toolchain when `llama.cpp` needs to be built
- enough disk space for the model and build artifacts

### Self-hosted or existing API gateway

If you already have a self-hosted gateway or another OpenAI-compatible endpoint, OpenJet can use that too.

OpenAI-compatible API:

```bash
export OPENAI_API_KEY=your-key
open-jet --setup
```

In setup, choose `Self-hosted API: OpenAI-compatible`, then enter:

1. model id
2. base URL
3. API key env var name

### Optional hosted fallback: OpenRouter

If you want a hosted profile alongside your local or self-hosted setup:

```bash
export OPENROUTER_API_KEY=your-key
open-jet --setup
```

In setup, choose `Hosted API: OpenRouter`, then enter:

1. model id
2. API key env var name

Then launch OpenJet normally:

```bash
open-jet
```

Supported runtimes in this simplified build:

- `llama.cpp` for local/offline use
- OpenAI-compatible APIs for self-hosted gateways and compatible services
- OpenRouter as an optional hosted profile

`SGLang` and `TensorRT-LLM` are intentionally disabled to keep setup simpler.

If you want full install details for each runtime, jump to [Installation](#installation) or the [Quickstart docs](docs/quickstart.md).

## Installation

### 1. Clone the repo and install

```bash
git clone https://github.com/l-forster/open-jet.git
cd open-jet
./install.sh
```

This creates a local virtualenv, installs the repo, and links `open-jet` / `openjet` into `~/.local/bin`.

### 2. Choose a runtime path

Recommended local/self-hosted path:

- run `open-jet --setup`
- let setup reuse or provision `llama-server`
- use a local `.gguf`, an installed Ollama model, an Ollama pull, or a recommended GGUF download

Self-hosted API path:

- use `Self-hosted API: OpenAI-compatible` in setup
- point OpenJet at your existing gateway or compatible endpoint
- keep the API key in an environment variable if your gateway requires one

Optional hosted path:

- use `Hosted API: OpenRouter` in setup
- keep your key in `OPENROUTER_API_KEY`

### 3. Optional: use an existing local model

If you already have a local model, setup can use one of:

- a local `.gguf` file
- an installed Ollama model
- an Ollama tag you want setup to pull for you
- or nothing up front if you want setup to download a recommended GGUF

### 4. Run setup

```bash
open-jet --setup
```

The setup flow guides you through:

1. hardware detection/profile
2. runtime selection
3. runtime-specific model or API details
4. context window size
5. GPU offload configuration for local `llama.cpp`

After setup:

```bash
open-jet
```

## What it provides

- bounded-context local chat with your on-device model
- self-hosted and hosted OpenAI-compatible API support through the same session layer
- automatic context condensing under pressure
- low-memory shell execution that can unload and reload `llama.cpp` models around heavy commands
- session resume and harness state recovery
- OpenTelemetry instrumentation with collector export
- hardware-aware runtime setup for local Linux systems, including Jetson
- controlled tool use and slash commands
- Python SDK access to the same backend, including streaming, approvals, tool limits, turn context, and image inputs

## Why OpenJet exists

Most local LLM tools stop at "run a model locally." That is not enough if you want an agent that stays usable across real sessions.

OpenJet is built around:

- limited prompt memory on-device
- interrupted work and session recovery
- deterministic tool and approval boundaries
- real traces for evaluation
- reliable operator workflows on constrained local models

## Docs

- [Quickstart](docs/quickstart.md)
- [Installation](docs/installation.md)
- [Examples](docs/examples/README.md)
- [Runtime: llama.cpp](docs/runtimes/llama-cpp.md)
- [Runtime: OpenAI-compatible](docs/runtimes/openai-compatible.md)
- [Runtime: OpenRouter](docs/runtimes/openrouter.md)
- [Usage: CLI](docs/usage/cli.md)
- [Usage: Slash commands](docs/usage/slash-commands.md)
- [Usage: Workflow harness](docs/usage/workflow-harness.md)
- [Usage: Session state and logging](docs/usage/session-state-and-logging.md)
- [Telemetry](docs/telemetry.md)
- [Python SDK](docs/sdk/python-sdk.md)
- [Deployment: Jetson](docs/deployment/jetson.md)
- [Deployment: Linux x86 + NVIDIA](docs/deployment/linux-x86-nvidia.md)
- [Deployment: CPU-only](docs/deployment/cpu-only.md)
- [Configuration](docs/configuration.md)
- [Licensing](docs/licensing.md)

## License

`AGPL-3.0-only`, with commercial licensing available under separate terms.
