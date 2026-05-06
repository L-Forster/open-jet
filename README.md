# OpenJet

![Stars](https://img.shields.io/github/stars/L-Forster/open-jet)
![License](https://img.shields.io/github/license/L-Forster/open-jet)
![Terminal-Bench 2.0](https://img.shields.io/badge/Terminal--Bench%202.0-59.3-blue)

<p align="center">
  <img width="1672" height="941" alt="OpenJet screenshot" src="https://github.com/user-attachments/assets/b06b0b8f-1bbc-443d-920e-bd70bff1479c" />
</p>

<br />

<h1 align="center">Claude Code for local LLMs</h1>

<h3 align="center">
  The easiest and fastest way to get a local coding agent running.
</h3>

<p align="center">
  OpenJet sets up the local model backend for your hardware and gives you a Claude-Code-style coding agent that can read files, edit code, and run commands fully on your own machine.
</p>

<p align="center">
  <a href="https://discord.com/invite/pspKHtExSa">Discord</a>
</p>

If you are new to local LLMs, OpenJet is the fastest way to get started without spending hours figuring out models, runtimes, and config.

If you have already tried local LLMs and got frustrated piecing together a model backend, a frontend, and an actual coding agent, OpenJet removes that setup tax.

OpenJet is built for people looking for a **Claude Code alternative, easy local LLM setup, or a self-hosted local coding agent.**

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

**General (any GPU/RAM — no `unified_memory_only` flag):**

| RAM | Model | max_ram_gb |
|---|---|---|
| < 6 GB | Qwen3.5 4B | 6.0 |
| < 12 GB | Qwen3.5 9B | 12.0 |
| < 12 GB | Qwen3.6 27B UD-IQ2_XXS | 12.0 |
| < 16 GB | Qwen3.6 27B UD-IQ3_XXS | 16.0 |
| < 20 GB | Qwen3.6 27B Q4_K_M | 20.0 |

**Unified memory only (`unified_memory_only: True`, `llama_cpu_moe: True`):**

| RAM | Model | max_ram_gb |
|---|---|---|
| < 24 GB | Gemma 4 26B A4B | 24.0 |
| < 24 GB | Qwen3.6 35B A3B UD-Q3_K_XL | 24.0 |
| < 32 GB | Qwen3.6 35B A3B UD-Q4_K_M | 32.0 |

Setup detects your hardware, picks a model that fits your RAM, downloads it, and gets everything running. Already have a `.gguf`? It finds that too.

Then run:

```bash
openjet
```

Other entrypoints from the same install:

```bash
openjet benchmark --sweep
```

```python
from openjet.sdk import OpenJetSession, recommend_hardware_config
```

## Why OpenJet

| What it does | Why it matters |
|---|---|
| **Easy local LLM setup** | Get a working local coding agent without manually learning the entire backend and runtime stack first |
| **Unified backend + harness** | One local system instead of separately wiring together a model runtime, config layer, frontend, and agent workflow |
| **Claude-Code-style workflow** | Read files, edit code, run commands, and work in a terminal agent instead of a plain chat window |
| **Hardware-aware setup** | OpenJet picks sensible defaults for your machine instead of leaving you to trial-and-error every setting |
| **Fully local** | Your code stays on your machine, with no cloud dependency required |
| **Remote execution support** | Run the model on one machine and execute on another |
| **SDK + benchmarks included** | Script the same runtime from Python and measure performance on your own hardware |

## How it compares

| Tool | Backend setup | Local runtime provisioning | Hardware auto-config | Terminal agent | Memory persistence |
| --- | --- | --- | --- | --- | --- |
| OpenJet | Built in: install + `openjet setup` | Yes: model discovery/download + `llama.cpp` config | Yes | Full TUI | Yes: global + project memory |
| Aider | Manual: choose API, local endpoint, or provider config | No | No | Terminal chat | No persistent agent memory |
| Cline | Manual: extension/CLI plus provider or local model config | No | No | Editor-first; CLI available | Yes: Memory Bank/rules |
| OpenCode | Manual: install CLI plus provider/local model config | No | No | Full TUI | Sessions/config persist |

## What you get

An agent in your terminal that can actually do useful work:

- **Read and edit your code**  
  Search files, apply edits, and write new ones

- **Run shell commands**  
  Explicit approval before commands execute

- **Resume sessions**  
  Close the terminal, come back later, keep going

- **Work on constrained hardware**  
  Automatic context condensing and model unload / reload around heavy tasks

- **Connect to devices**  
  Cameras, microphones, GPIO, and remote devices for edge and embedded workflows

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

Cloud coding agents need API keys, send your code to someone else's server, and charge per token.

Most local tools stop at chat. You can run a model, but you still do not have a real coding workflow.

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

OpenJet core is licensed under `Apache-2.0`.

That means individual developers and companies can use, modify, and embed the core SDK and CLI freely under the Apache terms. Future paid offerings for hosted, team, or enterprise functionality may be shipped separately under commercial terms.

External contributions are accepted under the contributor terms in [CONTRIBUTING.md](CONTRIBUTING.md) and [CLA.md](CLA.md).
