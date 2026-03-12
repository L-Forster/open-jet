# open-jet

`open-jet` is an offline-first agent runtime for edge Linux devices with tight memory budgets.

It is built for Jetson-class and other edge systems where the hard part is not just running a local model, but keeping the agent useful under constrained RAM, limited context windows, interrupted sessions, and hardware-specific failure modes.

It provides:
- bounded-context local chat with your on-device model
- safe file and doc loading with token and memory guards
- automatic context condensing under pressure
- session resume and harness state recovery
- replayable JSONL event traces for evals and debugging
- hardware-aware runtime setup for Jetson and edge Linux
- slash commands and harness modes for controlled workflows
- a Python SDK for driving the same agent backend without the TUI

`open-jet` is positioned around five practical problems:
- managing limited prompt memory on-device
- resuming interrupted work instead of starting over
- enforcing deterministic tool and approval boundaries
- capturing real traces for evaluation instead of guessing from vibes
- turning constrained local models into reliable operator workflows

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

## Why It Exists

Most local LLM tools stop at "chat with a model on your box." That breaks down quickly on edge hardware:
- context windows are small relative to the task
- available RAM moves around under real workloads
- long tasks get interrupted
- shell and file actions need deterministic approval paths
- failures differ by runtime, model, quant, and device profile

`open-jet` is designed around those constraints. The goal is to keep an on-device agent productive when memory is bounded and recovery matters more than demos.

## Python SDK

`open-jet` also exposes a programmatic session API so you can drive the same bounded-memory agent backend from your own scripts.

```python
import asyncio

from src import OpenJetSession


async def main() -> None:
    session = await OpenJetSession.create()
    try:
        response = await session.run("Summarize the current README")
        print(response.text)

        vision = await session.run("Describe this image", image_paths=["./example.png"])
        print(vision.text)

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

It then saves your configuration and starts the runtime with a device-appropriate memory profile.

## Basic Use

- Type normally and press Enter to chat
- Use `@file` or `@[path with spaces]` to add file content to context
- Use `@image.png` or paste local image file paths into the prompt to attach images to the next turn
- Type `/` to open slash-command suggestions
- `Tab`/`Enter` can autocomplete slash commands and file mentions
- `Ctrl+C` or `/exit` quits

The app is designed to keep work decomposed into small recoverable turns instead of trying to hold an entire task in prompt memory at once.

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

## Workflow Harness

`open-jet` includes a lightweight harness layer for keeping agent work structured under constrained context:
- modes for `chat`, `code`, `review`, and `debug`
- step-oriented state so the agent can continue work across turns
- skill docs and project docs loaded into bounded turn context
- persistent harness state stored under `.openjet/`

This is there to reduce prompt drift and keep limited-context models on the current task.

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

The event log is the main reliability artifact. It captures replayable traces for things like:
- tool call success rate
- approval and denial decisions
- interrupted generation and resumed sessions
- time-to-resolution
- token usage for successful tasks
- hallucinated or low-value command proposals
- hardware and runtime-specific failure analysis

This is meant to support evaluation from real traces, not just subjective testing.

## Benchmarks and Eval Traces

`open-jet` now includes benchmark environments and harness-driven eval cases under `benchmarks/`.

The point of these is not abstract leaderboard work. They are for checking whether the agent:
- stays useful under limited context
- resumes interrupted work
- follows deterministic tool constraints
- succeeds across different local workflows
- behaves differently across hardware, runtimes, and model profiles

If you care about edge agents in production, replayable traces and task artifacts matter more than one-off demo prompts.

## Contact

- Website: https://www.openjet.dev/
- X: https://x.com/flouislf
