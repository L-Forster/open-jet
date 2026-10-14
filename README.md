# OpenJet — Local AI Coding Agent and LLM Inference Engine

![Stars](https://img.shields.io/github/stars/L-Forster/open-jet)
![License](https://img.shields.io/github/license/L-Forster/open-jet)
[![PyPI](https://img.shields.io/pypi/v/open-jet)](https://pypi.org/project/open-jet/)

OpenJet is an open-source, self-hosted AI coding assistant and local LLM runtime. Run a
coding agent in your terminal with llama.cpp, Ollama or LM Studio, embed an on-device model
with the Python SDK, or serve Qwen3.8-27B with the standalone CUDA inference engine.

OpenJet profiles your hardware, picks a GGUF model that fits, downloads it, starts the inference
server, and gives you an agent that edits files and runs approved commands. Local mode needs
no account, API key or hosted service. Once your model is installed, it works offline.

[Install OpenJet](#install) · [Documentation](docs/getting-started.md) ·
[Python package](https://pypi.org/project/open-jet/) · [OpenJet Engine](megakernel/README.md) ·
[Discord](https://discord.com/invite/pspKHtExSa)

![Qwen3.6-27B Terminal-Bench 2.0](https://img.shields.io/badge/Qwen3.6--27B%20Terminal--Bench%202.0-59.3-blue)

<p align="center">
  <img width="1672" height="941" alt="OpenJet coding agent running in a terminal" src="https://github.com/user-attachments/assets/b06b0b8f-1bbc-443d-920e-bd70bff1479c" />
</p>

On NVIDIA **RTX 30, 40 and 50-series GPUs with 24 GB+**, OpenJet serves Qwen3.8-27B through its own
CUDA engine at **1.5x the speed of llama.cpp**. On an RTX 3090: 78 tok/s in reasoning mode, up to
140 tok/s writing code.

**Inference engine:** [OpenJet Engine](megakernel/README.md) has its own standalone installation,
`openjet-engine` server command, HTTP API quick start, benchmarks, and systems documentation.
Use it directly from your own application without running the coding agent.

## Install

Use your coding agent to install OpenJet. Paste this into Codex, Claude Code, or another agent with terminal access:

> Install OpenJet on this machine by following https://github.com/l-forster/open-jet/blob/master/docs/agent-install.md. Inspect my hardware, choose a suitable local coding model, configure it without the setup wizard, verify inference works, and tell me how to use OpenJet.

Your agent checks GPU and memory, downloads a model that fits, and verifies the local runtime. See the [agent installation instructions](docs/agent-install.md).

For manual installation:

```bash
pipx install open-jet
openjet setup
openjet
```

To accept the hardware recommendation without the wizard, run `openjet setup --yes`.

`openjet setup` detects your GPU and memory, picks a model and quantisation that fit, downloads the
GGUF, builds or reuses `llama-server`, and sets the context window and GPU offload. Already have a
`.gguf`? Pick **Use a local .gguf model file** during setup.

Recommended: a GPU with 14 GB+ VRAM, or Apple silicon with 24 GB+ unified memory. CPU-only works
too ([Deployment: CPU-only](docs/deployment/cpu-only.md)).

## OpenJet Engine: CUDA LLM inference on NVIDIA RTX GPUs

For Qwen3.8-27B Q4_K_M on an RTX 30, 40 or 50-series GPU with 24 GB or more (sm_86, sm_89, sm_120,
including the matching RTX workstation cards), OpenJet replaces `llama-server` with
[`megakernel/`](megakernel/README.md), a CUDA engine written for that one model. Benchmarks below are
from an RTX 3090. Each
decode step runs as a single persistent kernel launch and verifies 4-5 drafted tokens per pass.

| RTX 3090, Qwen3.8-27B Q4_K_M | llama.cpp | OpenJet megakernel |
|---|---|---|
| Decode, reasoning mode (what the agent uses) | 51 tok/s | **78 tok/s** (1.5x) |
| Decode, greedy, average of 4 prompts | 63 tok/s | **107 tok/s** (1.7x) |
| Decode, greedy, writing code | 73 tok/s | **140 tok/s** (1.9x) |
| Prompt processing, 2-8K tokens | 1050-1140 tok/s | **1530-1600 tok/s** |

Each figure is a single conversation, not a batch. llama.cpp runs with OpenJet's flags, including
MTP speculative decoding.

Turing GPUs (sm_75, e.g. the 48 GB Quadro RTX 8000) are supported experimentally for Q4_K_M: a
contributor measured 54 vs 33 tok/s reasoning decode against llama.cpp. See
[Turing (RTX 8000)](megakernel/README.md#turing-rtx-8000).

### Getting the megakernel

It ships as source in the PyPI package (0.4.32 and later) and `openjet setup` compiles it on your
machine, for your GPU:

```bash
pipx install open-jet     # already installed? pipx upgrade open-jet
openjet setup             # choose "Qwen3.8 27B Q4_K_M MTP"
openjet
```

You need:

- Linux with an NVIDIA RTX 30, 40 or 50-series GPU that has 24 GB or more
- the CUDA toolkit, for `nvcc` (CUDA 12 or newer; 12.8 or newer for RTX 50)
- the Qwen3.8 27B Q4_K_M model from the setup catalog

Which model file runs on it — `openjet setup` shows the engine in brackets next to each model:

| In setup | Model file |
|---|---|
| `(OpenJet megakernel, 1.5x faster)` | unsloth's original Qwen3.8-27B `Q4_K_M`, the one the catalog downloads |
| `(OpenJet megakernel, semi compatible)` | a Qwen3.8-27B `Q4_K_M` from lmstudio-community, mradermacher, Abiray or KikoCis |
| `(OpenJet megakernel, mixed quant)` | unsloth Qwen3.8-27B `UD-Q4_K_XL`, using a separate build |
| `(llama.cpp)` | everything else, including other Qwen3.8-27B quants (other unsloth `UD-*`, bartowski, GSQ-RCO) |

If none of your models is the megakernel's own, setup offers to download it.

Setup builds llama.cpp with CUDA, then compiles only the megakernel library needed by the selected
model, based on its GGUF tensor types. Existing Q4_K_M users keep the smaller build. The mixed
engine takes longer and uses roughly 11 GiB of system RAM during compilation; see the
[build requirements](megakernel/README.md#build--run). Both engines keep the embedding table in
system RAM by default, saving about 680 MiB of GPU VRAM.

For Q4_K_M, setup compiles the megakernel against llama.cpp. The compile takes about
2 minutes and ends with `Megakernel built: OpenJet will serve Qwen3.8-27B with it.` If you set
OpenJet up before 0.4.32, upgrade and run `openjet setup` again.

If setup can't build it, it prints `Skipping the Qwen3.8-27B megakernel (...); using llama-server.`
with the reason, most often a missing CUDA toolkit, and OpenJet serves the model with `llama-server`
instead. To see which engine is serving while OpenJet is running:

```bash
pgrep -af mk_server.py    # prints a line when the megakernel is serving
```

`llama_backend` in `config.yaml` controls this: `auto` (the default) uses the megakernel whenever the
model and GPU match, `llama` always uses `llama-server`, and `megakernel` fails instead of falling
back. From a source checkout you can also build it by hand with `cd megakernel && make`. See
[the llama.cpp runtime docs](docs/runtimes/llama-cpp.md#qwen38-27b-megakernel-backend-rtx-3090).

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
- **Benchmarks.** `openjet benchmark --sweep` measures GPU layers, batch sizes and thread counts on
  your machine ([Benchmarking](docs/benchmarking.md)).

### Models in the setup catalog

| Model | Target memory (GB) |
|---|---|
| Qwen3.5 4B | 6 |
| Qwen3.5 9B | 12 |
| Qwen3.6 27B UD-IQ2_XXS MTP | 12 |
| Qwen3.6 27B UD-IQ3_XXS MTP | 16 |
| Qwen3.8 27B Q4_K_M MTP | 20 |
| Gemma 4 26B A4B (unified memory only) | 24 |
| Qwen3.6 35B A3B UD-IQ2_XXS MTP (unified memory only) | 24 |
| Qwen3.6 35B A3B UD-Q3_K_XL MTP (unified memory only) | 32 |

Any GGUF that llama.cpp loads also works. See [Choosing a model](docs/models.md).

## Local AI coding assistant features

Most coding agents assume a frontier model with a huge context window. OpenJet assumes a model that
fits on your GPU, whose context is limited and drifts over a long task. So the harness carries more
of the load:

- `chat`, `code`, `review` and `debug` modes
- step state that persists across turns
- project and skill docs loaded into a bounded per-turn context
- automatic context condensing
- shell commands only run after you approve them
- sessions you can close and resume
- MCP tools, and device sources such as cameras, microphones and GPIO

The terminal UI is TypeScript, built on the Pi terminal framework; Python handles hardware,
provisioning and the model lifecycle. Release wheels bundle the UI, so you don't need Node.js or
Bun. See [TypeScript terminal UI](docs/architecture/typescript-tui.md).

With a local model, no code or prompts leave your machine. `airgapped: true` also blocks every
non-loopback model and tool endpoint.

## Slipstream: a frontier model plans, your local model works

Slipstream pairs a frontier model (OpenAI Codex through your ChatGPT subscription, or any
OpenRouter model) with your local model. The frontier model handles intent, architecture,
decomposition and the final review. The local model reads the repo, edits files, runs tests and
works through failures. Only its conclusions go back to the frontier model, not its exploration.

The target is a 20% frontier share of tokens, about 5x more work from the same plan. The footer
shows your actual share every session; small or high-risk tasks use more. The local side isn't
free either: it costs GPU time.

Start it with `/mode` → **Slipstream**. Connect with `/connect openai-codex`, or `/login` /
`/connect openrouter`. See [Usage: Slipstream](docs/usage/slipstream.md).

## Use Ollama, LM Studio or an existing llama-server

If you already run a model server, OpenJet can connect to it instead. That server keeps control of
the model, so OpenJet's hardware tuning and the megakernel don't apply.

```bash
pipx install 'open-jet[cloud]'
```

Add a profile to the `config.yaml` that `openjet status` reports, then select it with
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

Loopback servers get a placeholder API key automatically.

## Python SDK: embed a local LLM in your application

The same runtime ships inside your software: an in-app assistant, a game NPC, an offline
classifier. You provision the model at build time for the device you ship to:

```bash
pip install open-jet
cd your-project
openjet project --use-case dialogue --target handheld --budget 4
```

Unlike `openjet setup`, this sizes the model for the target device and the memory your app gives
it, optimises for first-token latency, and puts the weights in `.openjet/models/` so your build
bundles them.

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

Nothing downloads at runtime: if the model is missing, the session raises an error rather than
fetching it on a user's machine. See the [SDK quickstart](docs/sdk/quickstart.md).

## FAQ

**What is OpenJet?** An open-source local AI coding agent, Python SDK and LLM inference runtime.
Install the `open-jet` package from PyPI and run the `openjet` command to get started.

**How is this different from Ollama or LM Studio?** They run and serve models. OpenJet adds the
agent (file edits, approved commands, tools, sessions, an SDK), and can also replace them by
running `llama-server` or the megakernel itself.

**Does it work offline?** Yes. Once the model is on disk, everything runs locally.
`airgapped: true` blocks cloud runtimes and remote MCP servers too.

**Can it use a cloud model?** Only if you choose one. `/mode` switches between Local, Codex,
OpenRouter and Slipstream; local llama.cpp is the default.

**Does the megakernel work on other GPUs or models?** It serves Qwen3.8-27B Q4_K_M and UD-Q4_K_XL on RTX 30, 40 and
50-series GPUs with 24 GB or more (sm_86, sm_89, sm_120). It's tuned on an RTX 3090; `openjet setup` builds it
for whichever of these GPUs you have. Everything else uses `llama-server`.

## Docs

- Start: [Getting started](docs/getting-started.md) · [Quickstart](docs/quickstart.md) ·
  [Installation](docs/installation.md) · [Configuration](docs/configuration.md) ·
  [Choosing a model](docs/models.md) · [Overview](docs/overview.md)
- Runtime: [llama.cpp and the megakernel](docs/runtimes/llama-cpp.md) ·
  [Benchmarking](docs/benchmarking.md)
- Usage: [CLI](docs/usage/cli.md) · [Slash commands](docs/usage/slash-commands.md) ·
  [Slipstream](docs/usage/slipstream.md) · [Skills](docs/usage/skills.md) ·
  [MCP](docs/usage/mcp.md) · [Device sources](docs/usage/device-sources.md) ·
  [Workflow harness](docs/usage/workflow-harness.md) ·
  [Session state and logging](docs/usage/session-state-and-logging.md)
- SDK: [Quickstart](docs/sdk/quickstart.md) · [Python SDK](docs/sdk/python-sdk.md)
- Deployment: [Jetson](docs/deployment/jetson.md) ·
  [Linux x86 + NVIDIA](docs/deployment/linux-x86-nvidia.md) · [CPU-only](docs/deployment/cpu-only.md) ·
  [Examples](docs/examples/README.md)

## Community

[Discord](https://discord.gg/pspKHtExSa) · [X / Twitter](https://x.com/FlouisLF). Benchmarks from
other hardware are welcome.

## License

OpenJet core is `AGPL-3.0-only`: you can use, modify and redistribute the SDK and CLI under the
AGPL v3, including its network-use source requirements. Paid hosted, team or enterprise features
may ship separately under commercial terms. Contributions are accepted under
[CONTRIBUTING.md](CONTRIBUTING.md) and [CLA.md](CLA.md).
