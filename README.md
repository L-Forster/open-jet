# OpenJet

![Stars](https://img.shields.io/github/stars/L-Forster/open-jet)
![License](https://img.shields.io/github/license/L-Forster/open-jet)
![Qwen3.6-27B Terminal-Bench 2.0](https://img.shields.io/badge/Qwen3.6--27B%20Terminal--Bench%202.0-59.3-blue)

<p align="center">
  <img width="1672" height="941" alt="OpenJet running a local LLM coding agent in a terminal" src="https://github.com/user-attachments/assets/b06b0b8f-1bbc-443d-920e-bd70bff1479c" />
</p>

<br />

<h1 align="center">Run local LLMs in your terminal and in your code</h1>

<h3 align="center">
  A terminal coding agent, and a Python SDK for embedding on-device models in your own apps.
</h3>

<p align="center">
  OpenJet runs the model and the agent loop on your machine. No API keys. No code or data upload.
</p>

<p align="center">
  RTX 3090 + Qwen 27B: <strong>33 tok/s -> 70 tok/s</strong> with MTP. Open source.
</p>

<p align="center">
  <a href="https://discord.com/invite/pspKHtExSa">Discord</a>
</p>

**An agent in your terminal:**

```bash
pipx install open-jet
openjet setup
```

Setup profiles your hardware, picks a coding model that fits, downloads it, configures `llama.cpp`, and drops you into the agent. Then run `openjet`.

**A model inside your own application:**

```bash
pip install open-jet
cd your-project
openjet project
```

`openjet project` provisions a model for what you are shipping — chosen by use case, target device, and the memory budget your app can spare — and downloads it into your project so the build can bundle it. The SDK never downloads at runtime.

```python
from openjet.sdk import create_inference_session

session = await create_inference_session(system_prompt="You are a shopkeeper. Two sentences max.")
print((await session.run("The player asks what you have for sale.")).text)
```

Text in, text out, with every tool refused — an embedded model cannot reach the shell or the filesystem no matter what it generates.

New here? [Getting started](docs/getting-started.md) forks by which one you want.

## Built for small models, not shrunk down from big ones

Most coding agents assume a frontier model with a huge context window, then let you point them at a local endpoint. OpenJet assumes the opposite: a model that fits on your GPU, with limited context that drifts over a long task.

So the harness does the work — `chat` / `code` / `review` / `debug` modes, step-oriented state that persists across turns, project and skill docs loaded into bounded turn context, automatic context condensing, and model unload/reload on constrained hardware. The point is that the loop still knows what it is doing at step twenty.

No API calls. No code or data upload.

If you are new to local LLMs, OpenJet is the fastest way to get started without spending hours figuring out models, runtimes, and config. If you have already tried local LLMs and got frustrated piecing together a model backend, a frontend, and an actual agent workflow, OpenJet removes that setup tax.

## Install

### Recommended

```bash
pipx install open-jet
openjet setup
```

If you do not use `pipx`, install with Python directly:

```bash
python -m pip install --user open-jet
openjet setup
```

The PyPI package is `open-jet`; the installed command is `openjet`.

Recommended hardware: Apple silicon with 24GB+ unified memory, or a GPU with 14GB+ VRAM.

### Recommended hardware and models

The tables below list the setup catalog entries from `src/config.py`. `max_ram_gb`
is the configured setup target for that row. For embedding a model in your own
application, see [Choosing a model](docs/models.md).

**General (any GPU/RAM — no `unified_memory_only` flag):**

| Model | Configured `max_ram_gb` |
|---|---|
| Qwen3.5 4B | 6.0 |
| Qwen3.5 9B | 12.0 |
| Qwen3.6 27B UD-IQ2_XXS MTP | 12.0 |
| Qwen3.6 27B UD-IQ3_XXS MTP | 16.0 |
| Qwen3.6 27B Q4_K_M MTP | 20.0 |

**Unified memory only (`unified_memory_only: True`, `llama_cpu_moe: True`):**

| Model | Configured `max_ram_gb` |
|---|---|
| Gemma 4 26B A4B | 24.0 |
| Qwen3.6 35B A3B UD-IQ2_XXS MTP | 24.0 |
| Qwen3.6 35B A3B UD-Q3_K_XL MTP | 32.0 |

Setup detects your hardware, picks a model that fits your RAM, downloads it, and gets everything running. Already have a `.gguf`? It finds that too.

Then run:

```bash
openjet
```

Other entrypoints from the same install:

```bash
openjet benchmark --sweep
```

```bash
openjet fix
```

```python
from openjet.sdk import OpenJetSession, recommend_hardware_config
```

## Embed a local LLM in your own application

The terminal agent is one half. The other is shipping an on-device model inside something you build — an in-app assistant, a game NPC that talks, an offline classifier, a document extraction pipeline. No API keys, no per-token bill, no user data leaving the device.

```bash
pip install open-jet
cd your-project
openjet project --use-case dialogue --target handheld --budget 4
```

That is a build-time step, and it is different from `openjet setup` in the questions it asks:

| | `openjet setup` | `openjet project` |
|---|---|---|
| Hardware | detected on this machine | declared: the device you ship to |
| Memory for the model | whatever the machine has | the slice your application concedes |
| Optimises for | coding capability | use case and first-token latency |
| Model lands in | a machine-wide store | `.openjet/models/` in the project, so your build bundles it |

Then, in your code:

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

**Nothing downloads at runtime.** If the provisioned model is missing, the session raises immediately rather than reaching for the network on a user's machine. Your users never fetch a file, never see a config, and never need an internet connection.

- [SDK quickstart](docs/sdk/quickstart.md) — provision, embed, ship
- [Choosing a model](docs/models.md) — use cases, target devices, and the embedded catalog
- [Project configuration](docs/configuration.md#project-configuration-openjetconfigyaml) — the `.openjet/` overlay and config precedence

## Why OpenJet

| What it does | Why it matters |
|---|---|
| **Easy local LLM setup** | Get a working local agent without manually learning the entire backend and runtime stack first |
| **Unified backend + harness** | One local system instead of separately wiring together a model runtime, config layer, frontend, and agent workflow |
| **Local agent loop** | Work with files, approve shell commands, connect tools, and iterate against a local model |
| **Hardware-aware setup** | OpenJet picks sensible defaults for your machine instead of leaving you to trial-and-error every setting |
| **No API calls or data upload** | Keep the agent loop on your machine instead of sending work to a hosted model provider |
| **Remote execution support** | Run the model on one machine and execute on another |
| **SDK + benchmarks included** | Script the same runtime from Python and measure performance on your own hardware |

## What OpenJet combines

| Layer | What OpenJet provides |
| --- | --- |
| Local model runtime | Model discovery, download, and `llama.cpp` configuration |
| Agent interface | Terminal TUI for file work, commands, tools, and session continuity |
| Hardware setup | RAM / VRAM profiling and sensible defaults for the current machine |
| Workflow harness | Repeatable runs from the CLI, SDK, or background workflow runner |
| Device and tool access | MCP tools, cameras, microphones, GPIO, and remote execution targets |

## What you get

An agent in your terminal that can actually do useful work:

- **Work with local files**  
  Search, read, create, and update files in your projects

- **Run shell commands**  
  Explicit approval before commands execute

- **Run workflows and checks**  
  Let the agent inspect results, update files, and try again against the local model

- **Resume sessions**  
  Close the terminal, come back later, keep going

- **Work on constrained hardware**  
  Automatic context condensing and model unload / reload around heavy tasks

- **Connect to devices**  
  Cameras, microphones, GPIO, and remote devices for edge and embedded workflows

- **Connect tools**  
  Expose trusted MCP server tools through OpenJet's normal tool registry

- **Use the Python SDK**  
  Automate the same runtime from scripts and external apps

- **Auto-configure local inference**  
  Hardware profiling and recommended settings for local `llama.cpp`

- **Benchmark your setup**  
  Sweep GPU layers, batch sizes, and thread counts on your own hardware

## One runtime, three interfaces

### CLI + chat TUI
Interactive local agent work in the terminal.

### Python SDK
Ship an on-device model inside your own application, embed sessions, profile hardware, and automate workflows from Python.

```python
from openjet.sdk import OpenJetSession, create_inference_session, recommend_hardware_config
```

### Benchmarking tools
Measure prompt and generation performance on your active model profile.

```bash
openjet benchmark --sweep
```

## Why this exists

Cloud agents need API keys, send your work to someone else's server, and charge per token.

Most local tools stop at chat. You can run a model, but you still do not have a real agent loop for files, commands, tools, devices, and repeatable workflows.

OpenJet closes that gap. It is built for people who want the speed, control, and privacy of local LLMs without becoming experts in runtimes, config, and frontend/backend glue just to get started.

Everything runs on your machine.

## FAQ

### How do I run an LLM locally without an API key?

`pipx install open-jet && openjet setup`. Setup profiles your GPU and RAM, picks a GGUF model that fits, downloads it, builds or reuses `llama-server` from `llama.cpp`, and drops you into the agent. There is no account, no key, and no hosted endpoint anywhere in the path.

### Can I embed a local LLM in an app I ship to users?

Yes — that is what `openjet project` and `openjet.sdk` are for. You provision the model once at build time, bundle the weights with your build, and call `create_inference_session()` from your code. See [Embed a local LLM in your own application](#embed-a-local-llm-in-your-own-application).

### Will my users have to download a model?

No. Model acquisition happens once, on your machine, before you ship. The SDK never fetches anything at runtime; a missing model is an error, not a download.

### Does it work offline / air-gapped?

Yes. Once the model is on disk, inference is entirely local. `airgapped: true` additionally blocks every non-loopback network path, including cloud runtimes and remote MCP servers. See [Configuration](docs/configuration.md).

### What hardware do I need?

Apple silicon with 24GB+ unified memory or a GPU with 14GB+ VRAM for the terminal coding agent. Embedded models go much smaller — a 4B at Q4_K_M is about 2.8GB resident at 4k context and runs on an 8GB laptop or a handheld. CPU-only works too; see [Deployment: CPU-only](docs/deployment/cpu-only.md).

### Which models are supported?

Any GGUF that `llama.cpp` can load. The curated catalogs cover Qwen3.5 4B / 9B, Qwen3.6 27B and 35B-A3B with MTP, and Gemma 4 26B A4B — see [Choosing a model](docs/models.md). Point setup at a `.gguf` you already have and it will use that instead.

### Can it use a cloud model too?

Optionally, and never automatically. `/connect` plus a model profile routes to OpenAI, Anthropic, OpenRouter, or Codex OAuth when you switch to it by hand. Local `llama.cpp` remains the default.

### Is my code or data sent anywhere?

No. The model, the agent loop, tool execution, and session state all stay on your machine.

## Docs

### Start here
- [Getting started](docs/getting-started.md) — terminal agent, or a model inside your app
- [Choosing a model](docs/models.md)
- [Quickstart](docs/quickstart.md)
- [Installation](docs/installation.md)
- [Configuration](docs/configuration.md)
- [Product overview](docs/overview.md)
- [Runtime: llama.cpp](docs/runtimes/llama-cpp.md)

### CLI + chat TUI
- [Usage: CLI](docs/usage/cli.md)
- [Usage: Slash commands](docs/usage/slash-commands.md)
- [Usage: Skills](docs/usage/skills.md)
- [Usage: MCP](docs/usage/mcp.md)
- [Usage: Device sources](docs/usage/device-sources.md)
- [Usage: Workflow harness](docs/usage/workflow-harness.md)
- [Usage: Session state and logging](docs/usage/session-state-and-logging.md)

### SDK
- [SDK quickstart](docs/sdk/quickstart.md)
- [Python SDK](docs/sdk/python-sdk.md)

### Benchmarking
- [Benchmarking](docs/benchmarking.md)

### Examples and deployment
- [Examples](docs/examples/README.md)
- [Deployment: Jetson](docs/deployment/jetson.md)
- [Deployment: Linux x86 + NVIDIA](docs/deployment/linux-x86-nvidia.md)
- [Deployment: CPU-only](docs/deployment/cpu-only.md)

## Community

- [Discord](https://discord.gg/pspKHtExSa)
- [X / Twitter](https://x.com/FlouisLF)

Benchmarkers and testers are appreciated.

## License

OpenJet core is licensed under `AGPL-3.0-only`.

That means individual developers and companies can use, modify, and redistribute the core SDK and CLI under the GNU Affero General Public License v3.0 terms, including its network-use source availability requirements. Future paid offerings for hosted, team, or enterprise functionality may be shipped separately under commercial terms.

External contributions are accepted under the contributor terms in [CONTRIBUTING.md](CONTRIBUTING.md) and [CLA.md](CLA.md).
