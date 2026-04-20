# OpenJet
<div align="center">

<img height="500" alt="image" src="https://github.com/user-attachments/assets/660f7969-b38b-4a90-8b2c-ae872105cc13" />

**An AI coding agent that runs entirely on your machine.**
  <h1><strong><a href="https://discord.com/invite/pspKHtExSa">Discord</a></strong><strong>, <a href="https://openjet.dev/">Website</a></strong>, <strong><a href="https://x.com/FlouisLF">X/Twitter</a></strong></h1>


</div>
This is Claude Code for local LLMs. OpenJet handles the model, the runtime, and the setup without having to manually wrangle complex confirgurations. You get a coding agent in your terminal that reads your files, edits your code, runs commands, and stays out of the cloud.

OpenJet can connect to remote devices, so the agent can run on one device, but execute and write code on another (even if it isn't powerful enough to run the model)

OpenJet has three primary surfaces in one repo:

- **CLI + chat TUI** for interactive local agent work
- **Python SDK** for embedding sessions, hardware profiling, and auto-`llama.cpp` configuration
- **Benchmarking tools** for running `llama-bench` and sweep comparisons against your active model profile

## Get started

```bash
curl -fsSL https://www.openjet.dev/install.sh | bash
```

```bash
wget -qO- https://www.openjet.dev/install.sh | bash
```

On Windows:

```bat
curl -L https://www.openjet.dev/install.bat -o install.bat && install.bat
```

That installs OpenJet and starts setup. On Linux and macOS, run setup after install:

```bash
open-jet setup
```

That's it. Setup detects your hardware, picks a model that fits your RAM, downloads it, and gets everything running. Already have a `.gguf`? It finds that too.

Then just:

```bash
open-jet
```

Other entrypoints from the same install:

```bash
open-jet benchmark --sweep
```

```python
from openjet.sdk import OpenJetSession, recommend_hardware_config
```

## What you get

An agent in your terminal that can actually do things:

- **Read and edit your code** — search files, apply edits, write new ones
- **Run shell commands** — with explicit approval before anything executes
- **Resume sessions** — close the terminal, come back later, pick up where you left off
- **Work on constrained hardware** — automatic context condensing, model unload/reload around heavy tasks
- **Device access** — cameras, microphones, GPIO for edge and embedded work
- **Python SDK** — automate the same agent from scripts and external apps
- **Hardware profiling + auto-config** — recommend model/runtime settings for local `llama.cpp`
- **Benchmark sweeps** — compare prompt/gen throughput across GPU layers, batch sizes, and thread counts

## Why this exists

Cloud coding agents need API keys, send your code to someone else's server, and cost money per token. Local chat tools give you a chat window but not an agent — no file access, no shell, no session recovery.

OpenJet closes that gap. It's built for local models on real hardware, where memory is tight, context windows are short, and sessions get interrupted. Everything runs on your machine, nothing leaves it.

## Docs

- [Product surfaces](docs/overview.md)
- [Quickstart](docs/quickstart.md)
- [Installation](docs/installation.md)
- [Configuration](docs/configuration.md)
- [Runtime: llama.cpp](docs/runtimes/llama-cpp.md)

### CLI + chat TUI

- [Usage: CLI](docs/usage/cli.md)
- [Usage: Slash commands](docs/usage/slash-commands.md)
- [Usage: Device sources](docs/usage/device-sources.md)
- [Usage: Workflow harness](docs/usage/workflow-harness.md)
- [Usage: Session state and logging](docs/usage/session-state-and-logging.md)

### SDK + hardware profiling

- [Python SDK](docs/sdk/python-sdk.md)

### Benchmarking

- [Benchmarking](docs/benchmarking.md)

### Examples and deployment

- [Examples](docs/examples/README.md)
- [Deployment: Jetson](docs/deployment/jetson.md)
- [Deployment: Linux x86 + NVIDIA](docs/deployment/linux-x86-nvidia.md)
- [Deployment: CPU-only](docs/deployment/cpu-only.md)


## Community

Discord: https://discord.gg/pspKHtExSa

Benchmarkers and testers are appreciated.

X: https://x.com/FlouisLF

## License

OpenJet core is licensed under `Apache-2.0`.

That means individual developers and companies can use, modify, and embed the
core SDK and CLI freely under the Apache terms. Future paid offerings for
hosted, team, or enterprise functionality may be shipped separately under
commercial terms.

External contributions are accepted under the contributor terms in
[CONTRIBUTING.md](CONTRIBUTING.md) and [CLA.md](CLA.md).
