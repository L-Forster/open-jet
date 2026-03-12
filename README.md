# open-jet

`open-jet` is an offline-first terminal app for running local LLM workflows on edge Linux devices (including Jetson-class hardware).

It provides:
- local chat with your on-device model
- safe file-context loading with token/memory guards
- slash commands for session control
- first-run setup for model/runtime configuration
- optional session logging and resume
- a Python SDK for driving the agent backend without the TUI

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

## Python SDK

`open-jet` also exposes a programmatic session API so you can drive the same agent backend from your own scripts.

```python
import asyncio

from src import OpenJetSession


async def main() -> None:
    session = await OpenJetSession.create()
    try:
        response = await session.run("Summarize the current README")
        print(response.text)

        async for event in session.stream("Inspect README.md with tools if needed"):
            if event.text:
                print(event.text, end="")
            if event.tool_result:
                print(f"\n[{event.tool_result.tool_call.name}] {event.tool_result.output}")
    finally:
        await session.close()


asyncio.run(main())
```

Tools that mutate state or run shell commands require an approval handler:

```python
session = await OpenJetSession.create(
    approval_handler=lambda tool_call: tool_call.name == "shell"
)
```

You can also restrict the tool surface for embedded use:

```python
session = await OpenJetSession.create(allowed_tools={"read_file", "load_file", "grep"})
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

### SGLang

`open-jet` can also connect to `SGLang` through its OpenAI-compatible server.

On Jetson, prefer running SGLang in a local container. This keeps inference fully local while avoiding host-side Python dependency issues like missing `triton`.

Example `config.yaml` values:

```yaml
runtime: sglang
model: /home/you/models/Qwen3.5-4B-AWQ-4bit
sglang_model: /home/you/models/Qwen3.5-4B-AWQ-4bit
sglang_launch_mode: docker
sglang_base_url: http://127.0.0.1:30000
sglang_docker_image: your-local-sglang-image
sglang_docker_container_name: open-jet-sglang
sglang_docker_runtime: nvidia
sglang_served_model_name: local
sglang_reasoning_parser: qwen3
sglang_tool_call_parser: qwen3_coder
sglang_mem_fraction_static: 0.8
context_window_tokens: 8192
gpu_layers: 0
```

In `docker` mode, `open-jet` starts the local container itself and waits for the OpenAI-compatible API on `127.0.0.1`.

In `external` mode, `open-jet` does not import or launch SGLang from the host environment. It only connects to an already-running local server:

```bash
http://127.0.0.1:30000
```

Use `managed` mode only when SGLang is installed in the same Python environment as `open-jet`.

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
