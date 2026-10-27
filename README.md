# OpenJet — Local AI Coding Agent and LLM Inference Engine

![Stars](https://img.shields.io/github/stars/L-Forster/open-jet)
![License](https://img.shields.io/github/license/L-Forster/open-jet)
[![PyPI](https://img.shields.io/pypi/v/open-jet)](https://pypi.org/project/open-jet/)

Coding agents that run on your own GPU. Numbers are Qwen3.8-27B Q4_K_M on one RTX 3090:

| What you run | Speed | Context | Setup |
|---|---|---|---|
| **1 coding agent** in your terminal | 78 tok/s reasoning, up to 140 tok/s writing code (1.5-1.9x llama.cpp) | 100K tokens | [3 commands](#1-agent-on-an-rtx-3090-4090-or-5090) |
| **4 coding agents at once** on one GPU | ~290 tok/s total, ~75 per agent; 150-200 total in real agent runs | 132K tokens shared by the 4 | [from source](#4-agents-on-one-gpu) |
| **1 agent on any other GPU, Mac or CPU** | llama.cpp speed for your hardware | sized to your memory | [3 commands](#1-agent-on-any-other-hardware) |
| **A frontier model plans, your local model codes** (Slipstream) | local speed above | as above | [`/mode`](#slipstream) |
| **A local model inside your app** (Python SDK) | as above | as above | [`pip install`](#python-sdk) |

Everything runs on your machine: no account, no API key, nothing leaves it, and it works offline once
the model is downloaded.



https://github.com/user-attachments/assets/a7fc5010-110d-45fd-9774-a42afa1f3eb9



## Setup

### 1 agent on an RTX 3090, 4090 or 5090

You need Linux, an NVIDIA RTX 30, 40 or 50-series GPU with 24 GB or more, and the CUDA toolkit
(`nvcc`, CUDA 12 or newer; 12.8 or newer for RTX 50).

```bash
pipx install open-jet
openjet setup        # choose "Qwen3.8 27B Q4_K_M MTP"
openjet
```

`openjet setup` downloads the model (17 GB), builds llama.cpp, then compiles OpenJet's own CUDA engine
for your GPU in about 2 minutes. It ends with:

```
Megakernel built: OpenJet will serve Qwen3.8-27B with it.
```

If it prints `Skipping the Qwen3.8-27B megakernel (...)` instead, the reason is in the brackets (usually
no CUDA toolkit), and OpenJet falls back to `llama-server`: it still works, at llama.cpp speed. To check
which engine is serving while OpenJet runs, `pgrep -af mk_server.py` prints a line for the megakernel.

### 1 agent on any other hardware

The same three commands:

```bash
pipx install open-jet
openjet setup        # accept the recommended model, or: openjet setup --yes
openjet
```

Setup measures your VRAM and RAM, picks a model and quantisation that fit, downloads it, and sets the
context window and GPU offload. Recommended: a GPU with 14 GB+ VRAM, or Apple silicon with 24 GB+.
CPU-only works too ([CPU-only](docs/deployment/cpu-only.md)). Already have a `.gguf`? Pick
**Use a local .gguf model file** during setup. Models it chooses from: [Choosing a model](docs/models.md).

### 4 agents on one GPU

Same hardware as the RTX 3090/4090/5090 setup. Build the engine from a source checkout:

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

### Or let your coding agent set it up

Paste this into Codex, Claude Code, or another agent with terminal access:

> Install OpenJet on this machine by following https://github.com/l-forster/open-jet/blob/master/docs/agent-install.md. Inspect my hardware, choose a suitable local coding model, configure it without the setup wizard, verify inference works, and tell me how to use OpenJet.

## Slipstream

A frontier model (OpenAI Codex through your ChatGPT subscription, or any OpenRouter model) plans and
reviews; your local model reads the repo, edits files and runs the tests. Only the local model's
conclusions go back to the frontier model, so it uses about 20% of the tokens. Switch with `/mode` →
**Slipstream**, after `/connect openai-codex` or `/connect openrouter`.
See [Slipstream](docs/usage/slipstream.md).

## Use Ollama, LM Studio or your own llama-server

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

## Python SDK

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
See the [SDK quickstart](docs/sdk/quickstart.md).

## Benchmarks: OpenJet's engine vs llama.cpp

One conversation (not a batch), RTX 3090, Qwen3.8-27B Q4_K_M. llama.cpp runs with OpenJet's flags,
including MTP speculative decoding.

| | llama.cpp | OpenJet engine |
|---|---|---|
| Decode, reasoning mode (what the agent uses) | 51 tok/s | **78 tok/s** (1.5x) |
| Decode, greedy, average of 4 prompts | 63 tok/s | **107 tok/s** (1.7x) |
| Decode, greedy, writing code | 73 tok/s | **140 tok/s** (1.9x) |
| Prompt processing, 2-8K tokens | 1050-1140 tok/s | **1530-1600 tok/s** |

The engine serves Qwen3.8-27B Q4_K_M and UD-Q4_K_XL; every other model uses `llama-server`. How it
works, other quants and Turing GPUs: [OpenJet Engine](megakernel/README.md).

## Docs

- Start: [Getting started](docs/getting-started.md) · [Installation](docs/installation.md) ·
  [Configuration](docs/configuration.md) · [Choosing a model](docs/models.md)
- Runtime: [llama.cpp and the megakernel](docs/runtimes/llama-cpp.md) ·
  [Benchmarking](docs/benchmarking.md)
- Usage: [CLI](docs/usage/cli.md) · [Slash commands](docs/usage/slash-commands.md) ·
  [Slipstream](docs/usage/slipstream.md) · [Skills](docs/usage/skills.md) · [MCP](docs/usage/mcp.md)
- SDK: [Quickstart](docs/sdk/quickstart.md) · [Python SDK](docs/sdk/python-sdk.md)
- Deployment: [Jetson](docs/deployment/jetson.md) ·
  [Linux x86 + NVIDIA](docs/deployment/linux-x86-nvidia.md) · [CPU-only](docs/deployment/cpu-only.md)

## Community

[Discord](https://discord.gg/pspKHtExSa) · [X / Twitter](https://x.com/FlouisLF). Benchmarks from
other hardware are welcome.

## License

OpenJet core is `AGPL-3.0-only`: you can use, modify and redistribute the SDK and CLI under the
AGPL v3, including its network-use source requirements. Paid hosted, team or enterprise features
may ship separately under commercial terms. Contributions are accepted under
[CONTRIBUTING.md](CONTRIBUTING.md) and [CLA.md](CLA.md).
