# OpenJet

![Stars](https://img.shields.io/github/stars/L-Forster/open-jet)
![License](https://img.shields.io/github/license/L-Forster/open-jet)
![Terminal-Bench 2.0](https://img.shields.io/badge/Terminal--Bench%202.0-59.3-blue)

<p align="center">
  <img width="1672" height="941" alt="OpenJet screenshot" src="https://github.com/user-attachments/assets/b06b0b8f-1bbc-443d-920e-bd70bff1479c" />
</p>

<br />

<h1 align="center">The local agent for your own GPU</h1>

<h3 align="center">
  Files -> tools -> shell approval -> workflows -> local model.
</h3>

<p align="center">
  OpenJet runs the agent loop locally. No API calls. No code or data upload.
</p>

<p align="center">
  RTX 3090 + Qwen 27B: <strong>33 tok/s -> 70 tok/s</strong> with MTP. Open source.
</p>

<p align="center">
  <a href="https://discord.com/invite/pspKHtExSa">Discord</a>
</p>

Someone built a local AI agent that runs on your own GPU.

OpenJet runs the agent loop locally:

```text
files -> tools -> shell approval -> workflows -> local model
```

No API calls. No code or data upload.

RTX 3090 + Qwen 27B: **33 tok/s -> 70 tok/s** with MTP.

Open source.

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
is the configured setup target for that row.

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
| Qwen3.6 35B A3B UD-Q3_K_XL MTP | 24.0 |
| Qwen3.6 35B A3B MTP | 32.0 |

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
Embed sessions, profile hardware, and automate workflows from Python.

```python
from openjet.sdk import OpenJetSession, recommend_hardware_config
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

## Docs

### Start here
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
