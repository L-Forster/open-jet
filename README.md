````markdown
# OpenJet

<div align="center">

<img width="100%" alt="OpenJet screenshot" src="https://github.com/user-attachments/assets/660f7969-b38b-4a90-8b2c-ae872105cc13" />

## Claude Code for local LLMs

**The easiest way to get a local coding agent running.**

OpenJet is a **unified local agent runtime** that handles the backend, setup, and coding harness for you.

[Discord](https://discord.com/invite/pspKHtExSa) · [X/Twitter](https://x.com/FlouisLF)

</div>

OpenJet gives you a terminal coding agent that can **read files, edit code, run commands, and stay fully on your machine**, without manually piecing together a local model backend, runtime config, and agent harness.

If you are new to local LLMs, OpenJet is the fastest way to get a Claude-Code-style workflow running locally.

If you have already tried local LLMs and got annoyed wiring together a backend, frontend, and usable coding loop, OpenJet removes that setup tax.

## Install

### macOS / Linux

```bash
curl -fsSL https://www.openjet.dev/install.sh | bash
````

### Windows

```bat
curl -L https://www.openjet.dev/install.bat -o install.bat && install.bat
```

After install:

```bash
openjet setup
```

Setup detects your hardware, picks a model that fits your RAM, downloads it, and gets everything running. Already have a `.gguf`? It finds that too.

Then run:

```bash
openjet
```

## Why OpenJet

| What it does                   | Why it matters                                                                                         |
| ------------------------------ | ------------------------------------------------------------------------------------------------------ |
| **Unified backend + harness**  | One local coding system instead of separately wiring a model runtime, config layer, and agent workflow |
| **Hardware-aware setup**       | Gets a working local stack running without config archaeology                                          |
| **Claude-Code-style workflow** | Read files, edit code, run commands, and work in a terminal agent instead of a plain chat window       |
| **Fully local**                | Your code stays on your machine, with no cloud dependency required                                     |
| **Remote execution support**   | Run the model on one machine and execute on another                                                    |
| **SDK + benchmarks included**  | Script the same runtime from Python and measure performance on your own hardware                       |

## What you get

An agent in your terminal that can actually do useful work:

* **Read and edit your code**
  Search files, apply edits, and write new ones

* **Run shell commands**
  Explicit approval before commands execute

* **Resume sessions**
  Close the terminal, come back later, keep going

* **Work on constrained hardware**
  Automatic context condensing and model unload / reload around heavy tasks

* **Connect to devices**
  Cameras, microphones, GPIO, and remote devices for edge and embedded workflows

* **Use the Python SDK**
  Automate the same runtime from scripts and external apps

* **Auto-configure local inference**
  Hardware profiling and recommended settings for local `llama.cpp`

* **Benchmark your setup**
  Sweep GPU layers, batch sizes, and thread counts on your own hardware

## One runtime, three interfaces

### 1. CLI + chat TUI

Interactive local agent work in the terminal.

### 2. Python SDK

Embed sessions, profile hardware, and automate workflows from Python.

```python
from openjet.sdk import OpenJetSession, recommend_hardware_config
```

### 3. Benchmarking tools

Measure prompt and generation performance on your active model profile.

```bash
openjet benchmark --sweep
```

## Why this exists

Cloud coding agents need API keys, send your code to someone else's server, and charge per token.

Most local tools stop at chat. You get a model window, but not a real coding workflow.

OpenJet closes that gap. It is built for local models on real hardware, where memory is tight, context windows are short, and sessions get interrupted. Everything runs on your machine.

## Docs

### Start here

* [Quickstart](docs/quickstart.md)
* [Installation](docs/installation.md)
* [Configuration](docs/configuration.md)
* [Product overview](docs/overview.md)
* [Runtime: llama.cpp](docs/runtimes/llama-cpp.md)

### CLI + chat TUI

* [Usage: CLI](docs/usage/cli.md)
* [Usage: Slash commands](docs/usage/slash-commands.md)
* [Usage: Device sources](docs/usage/device-sources.md)
* [Usage: Workflow harness](docs/usage/workflow-harness.md)
* [Usage: Session state and logging](docs/usage/session-state-and-logging.md)

### SDK

* [Python SDK](docs/sdk/python-sdk.md)

### Benchmarking

* [Benchmarking](docs/benchmarking.md)

### Examples and deployment

* [Examples](docs/examples/README.md)
* [Deployment: Jetson](docs/deployment/jetson.md)
* [Deployment: Linux x86 + NVIDIA](docs/deployment/linux-x86-nvidia.md)
* [Deployment: CPU-only](docs/deployment/cpu-only.md)

## Community

* [Discord](https://discord.gg/pspKHtExSa)
* [X / Twitter](https://x.com/FlouisLF)

Benchmarkers and testers are appreciated.

## License

OpenJet core is licensed under `Apache-2.0`.

That means individual developers and companies can use, modify, and embed the core SDK and CLI freely under the Apache terms. Future paid offerings for hosted, team, or enterprise functionality may be shipped separately under commercial terms.

External contributions are accepted under the contributor terms in [CONTRIBUTING.md](CONTRIBUTING.md) and [CLA.md](CLA.md).

```

Big changes:
- stronger top-line message
- screenshot kept at the top, but social links made secondary
- clearer split for newcomers vs frustrated local users
- stronger explanation of the unified backend + harness idea
- "one runtime, three interfaces" instead of "three surfaces in one repo"
- "Why OpenJet" table added near the top so the value is obvious fast
```
