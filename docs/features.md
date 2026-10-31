# Features

English · [简体中文](zh/features.md)

## Use it with your coding tools

`openjet serve` answers the three APIs coding tools speak, from the same local model:

| Tool | API | Set up with |
|---|---|---|
| Claude Code | Anthropic Messages (`/v1/messages`) | `openjet connect claude-code` |
| Codex CLI | OpenAI Responses (`/v1/responses`) | `openjet connect codex` |
| OpenCode | OpenAI Chat Completions (`/v1/chat/completions`) | `openjet connect opencode` |
| OpenClaw | OpenAI Chat Completions | `openjet connect openclaw` |
| OpenJet's terminal agent | built in | `openjet` |
| Your own app | any of the three, or the [Python SDK](sdk/quickstart.md) | |

Each `connect` command prints the exact configuration for your server address and model. See
[Coding tools](clients.md) for the full recipes and known limits.

## 4 coding agents on one GPU

https://github.com/user-attachments/assets/0df36ad0-b484-463e-a4f7-530e5a768cd9

Same hardware as OpenJet's engine above. Build the engine from a source checkout:

```bash
git clone https://github.com/l-forster/open-jet.git && cd open-jet
git clone --depth 1 --branch b10246 https://github.com/ggml-org/llama.cpp.git llama.cpp
cmake -S llama.cpp -B llama.cpp/build -DGGML_CUDA=ON -DLLAMA_CURL=OFF
cmake --build llama.cpp/build --config Release -j 2
make -C megakernel libqwen38mk.so
pip install jinja2 rich  # rich: for mk_top.py
```

Start the server with the Qwen3.8-27B Q4_K_M GGUF (if `openjet setup` downloaded it, its path is
`llama_model` in your OpenJet `config.yaml`):

```bash
python3 megakernel/server/mk_server.py -m /path/to/Qwen3.8-27B-Q4_K_M.gguf --agents 4
```

Then point up to 4 agents, anything that speaks the OpenAI chat API, at `http://127.0.0.1:18080/v1`.
They decode in the same pass over the weights, so they share the GPU instead of taking turns:

| RTX 3090, Qwen3.8-27B Q4_K_M | Per agent | Total |
|---|---|---|
| 1 agent generating | 117 tok/s | 117 tok/s |
| 4 agents generating, short context | ~75 tok/s | ~290 tok/s |
| 4 agents generating, 10K context each | ~40 tok/s | ~160 tok/s |
| 4 real coding agents (they pause to run tools and tests) | 40-75 tok/s | 150-200 tok/s |

The 4 share one 132K-token KV cache (q8_0), so one agent can use most of it while the others are short.
A system prompt and tool list they have in common is processed once. To watch them live (tok/s, context
and task per agent):

```bash
python3 megakernel/tools/mk_top.py
```

## OpenJet's terminal agent

Most coding agents assume a frontier model with a huge context window. OpenJet's agent assumes a
model that fits on your GPU, whose context is limited and drifts over a long task. So the harness
carries more of the load:

- `chat`, `code`, `review` and `debug` modes
- step state that persists across turns
- project and skill docs loaded into a bounded per-turn context
- automatic context condensing
- shell commands only run after you approve them
- sessions you can close and resume
- MCP tools, and device sources such as cameras, microphones and GPIO

![Qwen3.6-27B Terminal-Bench 2.0](https://img.shields.io/badge/Qwen3.6--27B%20Terminal--Bench%202.0-59.3-blue)

The terminal UI is TypeScript, built on the
Pi terminal framework; Python handles hardware, provisioning and the model lifecycle. Release wheels
bundle the UI, so you don't need Node.js or Bun. See
[TypeScript terminal UI](architecture/typescript-tui.md).

With a local model, no code or prompts leave your machine. `airgapped: true` also blocks every
non-loopback model and tool endpoint.

## Tuned to the hardware

On the managed llama.cpp path, OpenJet owns the whole model lifecycle and sets it up for the machine
it's on:

- **Model choice.** Picks the model, quantisation and context window from measured VRAM and RAM.
- **GPU offload.** Sets GPU layers, and on unified-memory machines offloads MoE experts to the CPU.
- **Device quirks.** Caps the context on Vulkan to stay under single-allocation limits, and
  configures Jetson's unified memory and clocks.
- **Memory pressure.** Unloads and reloads the model around heavy shell commands on constrained
  machines.
- **KV cache.** Saves and restores the cache between turns, so a conversation isn't re-processed
  from scratch.
- **Benchmarks.** `openjet bench` measures the configured model with a fixed workload and writes a
  results file to share ([Results on your GPU](../README.md#results-on-your-gpu)). `openjet benchmark --sweep`
  tunes llama.cpp's GPU layers, batch sizes and thread counts with `llama-bench`
  ([Benchmarking](benchmarking.md)).

## Slipstream: a frontier model plans, your local model works

Slipstream pairs a frontier model (OpenAI Codex through your ChatGPT subscription, or any
OpenRouter model) with your local model. The frontier model handles intent, architecture,
decomposition and the final review. The local model reads the repo, edits files, runs tests and
works through failures. Only its conclusions go back to the frontier model, not its exploration.

The target is a 20% frontier share of tokens, about 5x more work from the same plan. The footer
shows your actual share every session; small or high-risk tasks use more. The local side isn't
free either: it costs GPU time.

Start it with `/mode` → **Slipstream**. Connect with `/connect openai-codex`, or `/login` /
`/connect openrouter`. See [Usage: Slipstream](usage/slipstream.md).

## Use Ollama, LM Studio or an existing llama-server

If you already run a model server, OpenJet's agent can connect to it instead. That server keeps
control of the model, so OpenJet's hardware tuning and engine don't apply.

```bash
pipx install 'open-jet[cloud]'
```

Add a profile to `config.yaml` (`openjet --status` shows where it is), then select it with
`/model <name>`:

```yaml
model_profiles:
  - name: ollama-qwen
    runtime: litellm
    provider: openai-compatible
    model: openai/qwen3.5:9b
    base_url: http://127.0.0.1:11434/v1
    context_window_tokens: 32768
```

| Server | `base_url` | `model` |
|---|---|---|
| Ollama | `http://127.0.0.1:11434/v1` | `openai/<ollama model>` |
| LM Studio | `http://127.0.0.1:1234/v1` | `openai/<LM Studio model ID>` |
| llama-server | `http://127.0.0.1:8080/v1` | `openai/local` |

That server keeps control of the model, so OpenJet's hardware tuning and its CUDA engine don't apply.

## Python SDK: embed a local LLM in your application

Ship a local model inside your app. Size it for the target device at build time:

```bash
pip install open-jet
cd your-project
openjet project --use-case dialogue --target handheld --budget 4
```

```python
import asyncio

from openjet.sdk import create_inference_session

async def main() -> None:
    session = await create_inference_session()
    try:
        result = await session.run("Summarise this support ticket in one line.")
        print(result.text)
    finally:
        await session.close()

asyncio.run(main())
```

The weights go in `.openjet/models/` for your build to bundle; nothing downloads on a user's machine.
See the [SDK quickstart](sdk/quickstart.md).

## FAQ

**What is OpenJet?** A fast way to run Qwen3.8-27B, and other open models, on your own GPU. It
sets up the model and the engine for your hardware and serves it to Claude Code, Codex, OpenCode,
OpenClaw, its own terminal agent, or your application. Install the `open-jet` package from PyPI.

**How is this different from Ollama or LM Studio?** They run and serve models. OpenJet does that
too, and on supported NVIDIA GPUs serves Qwen3.8-27B with its own engine,
1.5x faster than llama.cpp. It also speaks the Anthropic Messages API, so Claude Code connects
without a proxy.

**Does it work offline?** Yes. Once the model is on disk, everything runs locally.
`airgapped: true` blocks cloud runtimes and remote MCP servers too.

**Can it use a cloud model?** Only if you choose one. In OpenJet's agent, `/mode` switches between
Local, Codex, OpenRouter and Slipstream; local is the default.

**Does the engine work on other GPUs or models?** It serves Qwen3.8-27B Q4_K_M and UD-Q4_K_XL on
RTX 30, 40 and 50-series GPUs with 24 GB or more (sm_86, sm_89, sm_120). It's tuned on an RTX 3090.
Everything else runs on llama.cpp.
