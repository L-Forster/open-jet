# OpenJet

<img height="225" alt="image" src="https://github.com/user-attachments/assets/660f7969-b38b-4a90-8b2c-ae872105cc13" />

**An AI coding agent that runs entirely on your machine.**

This is Claude Code for local LLMs. OpenJet handles the model, the runtime, and the setup without having to manually wrangle complex confirgurations. You get a coding agent in your terminal that reads your files, edits your code, runs commands, and stays out of the cloud.

## Get started

```bash
git clone https://github.com/l-forster/open-jet.git
cd open-jet
./install.sh
open-jet --setup
```

That's it. Setup detects your hardware, picks a model that fits your RAM, downloads it, and gets everything running. Already have a `.gguf`? It finds that too.

Then just:

```bash
open-jet
```

## What you get

An agent in your terminal that can actually do things:

- **Read and edit your code** — search files, apply edits, write new ones
- **Run shell commands** — with explicit approval before anything executes
- **Resume sessions** — close the terminal, come back later, pick up where you left off
- **Work on constrained hardware** — automatic context condensing, model unload/reload around heavy tasks
- **Device access** — cameras, microphones, GPIO for edge and embedded work
- **Python SDK** — automate the same agent from scripts

## Why this exists

Cloud coding agents need API keys, send your code to someone else's server, and cost money per token. Local alternatives like Ollama give you a chat window but not an agent — no file access, no shell, no session recovery.

OpenJet closes that gap. It's built for local models on real hardware, where memory is tight, context windows are short, and sessions get interrupted. Everything runs on your machine, nothing leaves it.

## Docs

- [Quickstart](docs/quickstart.md)
- [Installation](docs/installation.md)
- [Configuration](docs/configuration.md)
- [Runtime: llama.cpp](docs/runtimes/llama-cpp.md)
- [Python SDK](docs/sdk/python-sdk.md)
- [Usage: CLI](docs/usage/cli.md)
- [Usage: Slash commands](docs/usage/slash-commands.md)
- [Usage: Device sources](docs/usage/device-sources.md)
- [Usage: Workflow harness](docs/usage/workflow-harness.md)
- [Usage: Session state and logging](docs/usage/session-state-and-logging.md)
- [Examples](docs/examples/README.md)
- [Deployment: Jetson](docs/deployment/jetson.md)
- [Deployment: Linux x86 + NVIDIA](docs/deployment/linux-x86-nvidia.md)
- [Deployment: CPU-only](docs/deployment/cpu-only.md)

## License

`AGPL-3.0-only`, with commercial licensing available under separate terms.

