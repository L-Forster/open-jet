# OpenJet
<img  height="150" alt="Screenshot_20260227_144411" src="https://github.com/user-attachments/assets/70d90ca5-5841-44d8-a1d0-ef3e08acd95b" />


`open-jet` is an offline-first agent runtime for Jetson-class and other edge Linux systems with tight memory budgets.

It is built for cases where the hard part is not just running a local model, but keeping the agent useful under constrained RAM, limited context windows, interrupted sessions, and hardware-specific failure modes.

## Quickstart

```bash
pip install open-jet
```

OpenJet does not bundle an inference runtime. By default it uses `llama-server`
from [`llama.cpp`](https://github.com/ggerganov/llama.cpp), though setup can
also configure `SGLang` or `TensorRT-LLM` instead.

For the default `llama.cpp` path, you need:

1. the `open-jet` Python package
2. a working `llama-server` binary
3. a local model file such as a `.gguf`

Then run:

```bash
open-jet --setup
open-jet
```

## Installation

### 1. Install the Python package

```bash
pip install open-jet
```

### 2. Install `llama-server` from `llama.cpp`

`llama.cpp` setup is required for the default OpenJet runtime. If you choose
`SGLang` or `TensorRT-LLM` in `open-jet --setup`, follow the runtime-specific
model/runtime setup for those backends instead.

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
