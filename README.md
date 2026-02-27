# open-jet

`open-jet` is an offline-first terminal app for running local LLM workflows on edge Linux devices (including Jetson-class hardware).

It provides:
- local chat with your on-device model
- safe file-context loading with token/memory guards
- slash commands for session control
- first-run setup for model/runtime configuration
- optional session logging and resume

## Requirements

- `llama-server` from `llama.cpp` built for your device (see below)
- a local `.gguf` model file, or `ollama` installed for model download

## Building llama-server

`open-jet` uses `llama-server` from [llama.cpp](https://github.com/ggerganov/llama.cpp) as its inference backend. Pre-built binaries are available for x86, but on Jetson/ARM64 you need to build from source with the right flags.

### Jetson (Orin Nano, Orin NX, AGX Orin)

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

Key flags:

| Flag | Why |
|------|-----|
| `GGML_CUDA=ON` | Enable CUDA backend |
| `CMAKE_CUDA_ARCHITECTURES=87` | Target SM 8.7 (Orin). Use `72` for Xavier. |
| `GGML_CUDA_FA_ALL_QUANTS=ON` | Enable flash attention for all KV cache quantizations (q8_0, q4_0), not just f16. Required for fast inference with quantized KV cache. |

The built binary will be at `build/bin/llama-server`. Either add it to your `PATH` or leave it at `~/llama.cpp/build/bin/` where `open-jet` will find it automatically.

### Other Linux (x86_64 with NVIDIA GPU)

```bash
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
mkdir build && cd build

cmake .. -DGGML_CUDA=ON -DGGML_CUDA_FA_ALL_QUANTS=ON
cmake --build . --target llama-server -j$(nproc)
```

### CPU only

```bash
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
mkdir build && cd build

cmake ..
cmake --build . --target llama-server -j$(nproc)
```

## Install

```bash
pip install open-jet
```

## Start

```bash
open-jet
```

Optional setup screen on launch:

```bash
open-jet --setup
```

## First-Run Setup

On first run, `open-jet` guides you through:
1. hardware detection/profile
2. model source selection
3. model path or download choice
4. context window size
5. GPU offload configuration

It then saves your configuration and starts the runtime.

## Basic Use

- Type normally and press Enter to chat
- Use `@file` or `@[path with spaces]` to add file content to context
- Type `/` to open slash-command suggestions
- `Tab`/`Enter` can autocomplete slash commands and file mentions
- `Ctrl+C` or `/exit` quits

## Slash Commands

- `/help` show commands
- `/exit` quit app
- `/clear` clear chat and restart runtime (flush KV cache)
- `/clear-chat` clear chat only
- `/status` show context/RAM status
- `/condense` condense older context
- `/load <path>` load a file into context
- `/resume` load previous saved session
- `/setup` reopen setup wizard

## Configuration

Main settings are stored in `config.yaml`, including:
- context window size
- memory guard limits
- logging settings
- session state/resume settings

### TensorRT-LLM (PyTorch runtime) with Qwen

`open-jet` can run against `trtllm-serve` instead of `llama-server`.

1. Install TensorRT-LLM so `trtllm-serve` is on your `PATH`.
2. Set your config to use the TensorRT-LLM runtime.

Example `config.yaml` values:

```yaml
runtime: trtllm_pytorch
model: Qwen/Qwen2.5-7B-Instruct
trtllm_backend: pytorch
trtllm_trust_remote_code: true
# optional: pass a trtllm-serve YAML file
# trtllm_config_path: /home/you/qwen-fast.yml
context_window_tokens: 4096
gpu_layers: 0
```

When `runtime` is `trtllm_pytorch`, `open-jet` launches:

```bash
trtllm-serve <model> --backend pytorch --host 127.0.0.1 --port 8080
```

and then connects through the same OpenAI-compatible chat API path.

## Logging and Session State

When enabled:
- session events are written to `session_logs/*.events.jsonl`
- system metrics are written to `session_logs/*.metrics.jsonl`
- conversation state is saved to `session_state.json`

## Contact

- Website: https://www.openjet.dev/
- X: https://x.com/flouislf
