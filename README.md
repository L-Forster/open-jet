# OpenJet
<img  height="150" alt="Screenshot_20260227_144411" src="https://github.com/user-attachments/assets/70d90ca5-5841-44d8-a1d0-ef3e08acd95b" />


`open-jet` is an offline-first local agent you can actually use for real work.

It gives you a practical agent interface on top of your own local model and runtime, without making you assemble chat, tools, session recovery, and prompt management yourself.

OpenJet is designed to stay useful when local AI gets messy: tight memory, short context windows, interrupted sessions, and hardware-specific runtime issues.

## Why use OpenJet

- start quickly with a local runtime and model you already control
- keep working without losing the thread when sessions get interrupted
- use tools with explicit approvals and predictable behavior
- run the same backend from the TUI or Python SDK
- avoid building your own glue layer around local inference

## Quickstart

Install the package:

```bash
pip install open-jet
```

On first run, OpenJet walks you through connecting a local runtime and model.

Start setup:

```bash
open-jet --setup
```

The setup flow guides you through:

1. runtime selection
2. model source or path
3. context window size
4. GPU offload configuration

Then launch OpenJet normally:

```bash
open-jet
```

By default, OpenJet uses `llama-server` from [`llama.cpp`](https://github.com/ggerganov/llama.cpp), and setup can also configure `SGLang` or `TensorRT-LLM`.

If you want full install details for each runtime, jump to [Installation](#installation) or the [Quickstart docs](docs/quickstart.md).

## Installation

### 1. Install the Python package

```bash
pip install open-jet
```

### 2. Install `llama-server` from `llama.cpp`

`llama.cpp` setup is required for the default OpenJet runtime. If you choose
`SGLang` or `TensorRT-LLM` in `open-jet --setup`, follow the runtime-specific
model/runtime setup for those backends instead.

Linux x86 + NVIDIA:

```bash
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
mkdir build && cd build

cmake .. -DGGML_CUDA=ON -DGGML_CUDA_FA_ALL_QUANTS=ON
cmake --build . --target llama-server -j$(nproc)
```

CPU-only:

```bash
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
mkdir build && cd build

cmake ..
cmake --build . --target llama-server -j$(nproc)
```

Jetson:

```bash
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
mkdir build && cd build

cmake .. \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=87 \
  -DGGML_CUDA_FA_ALL_QUANTS=ON

cmake --build . --target llama-server -j$(nproc)
```

OpenJet looks for `llama-server` on `PATH` first, then at
`~/llama.cpp/build/bin/llama-server`.

### 3. Have a local model ready

You need a model reference for the selected runtime:

- `llama.cpp`: typically a local `.gguf` file, or an Ollama-backed model selected during setup
- `SGLang`: a local model directory or supported HF model id
- `TensorRT-LLM`: a local model directory or supported HF model id

### 4. Run setup

```bash
open-jet --setup
```

The setup flow guides you through:

1. hardware detection/profile
2. model source selection
3. model path or download choice
4. context window size
5. GPU offload configuration

After setup:

```bash
open-jet
```

## What it provides

- bounded-context local chat with your on-device model
- automatic context condensing under pressure
- session resume and harness state recovery
- OpenTelemetry instrumentation with collector export
- hardware-aware runtime setup for local Linux systems, including Jetson
- controlled tool use and slash commands
- Python SDK access to the same backend

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
- [Runtime: SGLang](docs/runtimes/sglang.md)
- [Runtime: TensorRT-LLM](docs/runtimes/tensorrt-llm.md)
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
