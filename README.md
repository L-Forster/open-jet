# OpenJet

![Stars](https://img.shields.io/github/stars/L-Forster/open-jet)
![License](https://img.shields.io/github/license/L-Forster/open-jet)
![Qwen3.6-27B Terminal-Bench 2.0](https://img.shields.io/badge/Qwen3.6--27B%20Terminal--Bench%202.0-59.3-blue)

<p align="center">
  <img width="1672" height="941" alt="OpenJet self-hosted local AI coding agent running in a terminal" src="https://github.com/user-attachments/assets/b06b0b8f-1bbc-443d-920e-bd70bff1479c" />
</p>

<br />

<h1 align="center">The self-hosted AI coding agent for local LLMs</h1>

<h3 align="center">
  Use Ollama, LM Studio, or llama.cpp — or let OpenJet provision and host the right GGUF model for your hardware.
</h3>

<p align="center">
  A private, offline coding assistant, hardware-aware local inference host, and Python SDK for embedding on-device models in your own applications.
</p>

<p align="center">
  RTX 3090 + Qwen 27B: <strong>33 tok/s -> 70 tok/s</strong> with MTP. Open source.
</p>

<p align="center">
  <a href="https://discord.com/invite/pspKHtExSa">Discord</a>
</p>

OpenJet is an open-source local AI coding agent that reads and edits files, runs approved
shell commands, uses tools, and keeps sessions on your machine. Connect an existing local
LLM server or let OpenJet download, configure, and serve a model itself—without requiring
an OpenJet account, subscription, or hosted API.

## Local AI coding agent for Ollama, LM Studio, and llama.cpp

OpenJet does not make you rebuild your local LLM setup. Use the server you already run,
or let OpenJet provision the complete stack for you.

| You have | How OpenJet uses it |
|---|---|
| **Nothing installed yet** | Profiles your hardware, selects and downloads a GGUF, provisions `llama-server`, and starts the agent |
| **Ollama** | Connects to Ollama's local OpenAI-compatible endpoint |
| **LM Studio** | Connects to the local server exposed by LM Studio |
| **llama.cpp + a GGUF** | Reuses your model and `llama-server`, with native lifecycle and KV-cache integration |
| **Any OpenAI-compatible local server** | Uses its loopback `/v1` endpoint without requiring a real API key |

### Let OpenJet provision and host the model

```bash
pipx install open-jet
openjet setup
```

Setup profiles your GPU and memory, picks a coding model that fits, downloads it, provisions
or reuses `llama-server`, and configures the context window and GPU offload. OpenJet then
starts and manages the model server itself.

```bash
openjet
```

Already have a GGUF? Choose **Use a local .gguf model file** during setup. If
`llama-server` is on `PATH` or installed under `~/llama.cpp/build/bin/`, OpenJet reuses it.

This native path gives OpenJet control of the full model lifecycle: server startup, streaming,
reasoning mode, KV-cache reset/save/restore, and unload/reload on memory-constrained machines.

### Use Ollama, LM Studio, or a running llama.cpp server

Install the OpenAI-compatible runtime adapter:

```bash
pipx install 'open-jet[cloud]'
```

Run `openjet status` to see which `config.yaml` OpenJet is using. Add one of these
profiles to that file, start OpenJet, then select it with `/model <profile-name>`.

#### Ollama local coding agent

First run a tool-capable model, for example `ollama run qwen3.5:9b`:

```yaml
model_profiles:
  - name: ollama-qwen
    runtime: litellm
    provider: openai-compatible
    model: openai/qwen3.5:9b
    base_url: http://127.0.0.1:11434/v1
    context_window_tokens: 32768
```

#### LM Studio local coding agent

Start the LM Studio local server and replace the model ID below with the one it exposes:

```yaml
model_profiles:
  - name: lm-studio
    runtime: litellm
    provider: openai-compatible
    model: "openai/<lm-studio-model-id>"
    base_url: http://127.0.0.1:1234/v1
    context_window_tokens: 32768
```

#### llama.cpp local coding agent

Connect a running `llama-server`; its default port is normally `8080`:

```yaml
model_profiles:
  - name: llama-server
    runtime: litellm
    provider: openai-compatible
    model: openai/local
    base_url: http://127.0.0.1:8080/v1
    context_window_tokens: 32768
```

Loopback servers receive a local placeholder key automatically; OpenJet does not require or
store a real API key for them. Set `airgapped: true` if you want OpenJet to reject every
non-loopback model and tool endpoint.

> Connecting to an already-running server gives OpenJet the agent interface while that server
> owns the model process. Use the native managed `llama.cpp` path when you want OpenJet to
> provision, start, stop, unload, reload, and optimize the model itself.

## Why OpenJet

- **Use your runtime or bring none.** Connect Ollama, LM Studio, llama.cpp, or another local
  OpenAI-compatible server. If you have no stack yet, one setup command creates it.
- **Own the model and the agent loop.** Files, commands, sessions, inference, and model weights
  can all remain on your machine, with no hosted OpenJet service in the path.
- **Built for local models.** The harness uses bounded context, automatic condensing, persistent
  step state, and low-memory model swapping instead of assuming frontier-model resources.
- **Hardware-aware provisioning.** OpenJet selects a model, quantization, context window, and GPU
  offload for the memory actually available on the machine.
- **Local inference hosting included.** On the native llama.cpp path, OpenJet provisions and manages
  `llama-server`; you do not need a separate model manager or inference service.
- **An SDK, not only a terminal.** Use the same local runtime from Python, or provision and bundle
  an on-device model inside software you ship.
- **Tools without surrendering control.** Work with files and shell commands, connect MCP tools
  and devices, and keep side effects visible and permission-gated.

## Built for small models, not shrunk down from big ones

Most coding agents assume a frontier model with a huge context window, then let you point them at a local endpoint. OpenJet assumes the opposite: a model that fits on your GPU, with limited context that drifts over a long task.

So the harness does the work — `chat` / `code` / `review` / `debug` modes, step-oriented state that persists across turns, project and skill docs loaded into bounded turn context, automatic context condensing, and model unload/reload on constrained hardware. The point is that the loop still knows what it is doing at step twenty.

No hosted API is required. With a local profile, no code or prompt data leaves your machine.

If you are new to local LLMs, OpenJet is the fastest way to get started without spending hours figuring out models, runtimes, and config. If you have already tried local LLMs and got frustrated piecing together a model backend, a frontend, and an actual agent workflow, OpenJet removes that setup tax.

## Managed llama.cpp hardware and models

For the native managed runtime, recommended hardware is Apple silicon with 24GB+ unified
memory or a GPU with 14GB+ VRAM. Existing Ollama, LM Studio, and OpenAI-compatible servers
remain responsible for their own model requirements.

### Provisioning catalog

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
| Qwen3.8 27B Q4_K_M MTP | 20.0 |

**Unified memory only (`unified_memory_only: True`, `llama_cpu_moe: True`):**

| Model | Configured `max_ram_gb` |
|---|---|
| Gemma 4 26B A4B | 24.0 |
| Qwen3.6 35B A3B UD-IQ2_XXS MTP | 24.0 |
| Qwen3.6 35B A3B UD-Q3_K_XL MTP | 32.0 |

Setup detects your hardware, picks a model that fits your RAM, downloads it, and gets
everything running. Already have a `.gguf`? It finds that too.

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

## What OpenJet combines

| Layer | What OpenJet provides |
| --- | --- |
| Local model runtime | Managed model discovery, download, and `llama.cpp` hosting, or connection to Ollama, LM Studio, and OpenAI-compatible servers |
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

### Does OpenJet work with Ollama and LM Studio?

Yes. OpenJet connects to the local OpenAI-compatible endpoints exposed by Ollama and
LM Studio, as well as a running `llama-server` or another compatible server. See
[Use Ollama, LM Studio, or a running llama.cpp server](#use-ollama-lm-studio-or-a-running-llamacpp-server).

### What is the difference between OpenJet and Ollama or LM Studio?

Ollama and LM Studio primarily run and serve models. OpenJet adds the coding-agent loop:
file editing, approved shell commands, tools, persistent sessions, bounded context, workflows,
and an SDK. OpenJet can use those servers, or replace that layer by provisioning and managing
`llama-server` itself.

### Is OpenJet fully self-hosted and offline?

Yes when you select a local profile. The model, agent loop, tool execution, and session state
remain on hardware you control. `airgapped: true` additionally rejects non-loopback endpoints.

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

Not when you use a local profile. The model, agent loop, tool execution, and session state
stay on your machine. If you manually select an optional cloud profile, prompts are sent only
to that profile's configured provider.

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
