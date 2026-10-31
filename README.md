<h1 align="center">OpenJet</h1>

<p align="center"><b>Run Qwen3.8-27B on your own GPU, 1.5x faster than llama.cpp.</b><br>
RTX 30, 40 and 50-series with 24 GB · works with Claude Code, Codex, OpenCode and OpenClaw · offline · open source</p>

<p align="center">
  <a href="https://github.com/L-Forster/open-jet/stargazers"><img src="https://img.shields.io/github/stars/L-Forster/open-jet" alt="Stars"></a>
  <a href="https://pypi.org/project/open-jet/"><img src="https://img.shields.io/pypi/v/open-jet" alt="PyPI"></a>
  <a href="LICENSE"><img src="https://img.shields.io/github/license/L-Forster/open-jet" alt="License"></a>
  <a href="https://discord.com/invite/pspKHtExSa"><img src="https://img.shields.io/badge/Discord-join-5865F2" alt="Discord"></a>
</p>



https://github.com/user-attachments/assets/c9ac0b21-67b1-4f20-af70-21741accc774


| RTX 3090, Qwen3.8-27B Q4_K_M | llama.cpp | OpenJet |
|---|---|---|
| Reasoning mode, short prompts (what coding agents use) | 60 tok/s | **92 tok/s** (1.5x) |
| Reasoning mode, 98K tokens in context | 37 tok/s | **79 tok/s** (2.2x) |
| Reading a 98K-token prompt | 112 s | **82 s** (1.4x) |
| Greedy, average of 4 prompts | 63 tok/s | **107 tok/s** (1.7x) |
| Greedy, writing code | 73 tok/s | **140 tok/s** (1.9x) |
| Reading your prompt, 2-8K tokens | 1050-1140 tok/s | **1530-1600 tok/s** |

Each figure is one conversation on one RTX 3090, not a batch. llama.cpp runs with OpenJet's flags,
including MTP speculative decoding. The first three rows are the current engine against llama.cpp
b11982, measured with [`megakernel/bench/run_compare.sh`](megakernel/bench/run_compare.sh): thinking on
at xhigh, temperature 1.0, 5 prompts x 4 seeds, and one 98K-token prompt x 4 seeds. The other rows are
from an earlier engine build. OpenJet's output matches llama.cpp's on the same weights: KL divergence 0.0009, top-token
agreement 99.08% ([results and reproduction scripts](megakernel/README.md#results)).

## Install

Linux and macOS:

```bash
curl -fsSL https://github.com/L-Forster/open-jet/releases/latest/download/install.sh | sh
```

Windows PowerShell:

```powershell
irm https://github.com/L-Forster/open-jet/releases/latest/download/install.ps1 | iex
```

The installer brings its own Python, so you don't need Python or pipx. Then, in a new terminal:

```bash
openjet setup      # finds your GPU, downloads the model and the engine built for it
openjet            # OpenJet's own coding agent
```

Already use Python? `pipx install open-jet` works too.

Or serve the model to the coding tool you already use:

```bash
openjet serve                     # OpenAI and Anthropic APIs on http://127.0.0.1:8080
openjet connect claude-code       # prints the settings for Claude Code (or codex, opencode, openclaw)
```

Or paste this into Claude Code, Codex or another agent with terminal access:

> Install OpenJet on this machine by following https://github.com/l-forster/open-jet/blob/master/docs/agent-install.md. Inspect my hardware, choose a suitable local coding model, configure it without the setup wizard, verify inference works, and tell me how to use OpenJet.

To accept the hardware recommendation without the wizard, run `openjet setup --yes`.

## Use it with your coding tools

`openjet serve` answers the three APIs coding tools speak, from the same local model:

| Tool | API | Set up with |
|---|---|---|
| Claude Code | Anthropic Messages (`/v1/messages`) | `openjet connect claude-code` |
| Codex CLI | OpenAI Responses (`/v1/responses`) | `openjet connect codex` |
| OpenCode | OpenAI Chat Completions (`/v1/chat/completions`) | `openjet connect opencode` |
| OpenClaw | OpenAI Chat Completions | `openjet connect openclaw` |
| OpenJet's terminal agent | built in | `openjet` |
| Your own app | any of the three, or the [Python SDK](docs/sdk/quickstart.md) | |

Each `connect` command prints the exact configuration for your server address and model. See
[Coding tools](docs/clients.md) for the full recipes and known limits.

## What you need

| Your hardware | What runs | Guide |
|---|---|---|
| NVIDIA RTX 30, 40 or 50-series with 24 GB, Linux | Qwen3.8-27B on OpenJet's engine | [Engine for your GPU](docs/installation.md#prebuilt-engines) · [OpenJet's engine](megakernel/README.md) |
| NVIDIA Turing with 24 GB or more, Linux | OpenJet's engine, experimental | [Engine for your GPU](docs/installation.md#prebuilt-engines) · [Turing](megakernel/README.md#turing-rtx-8000) |
| Windows | llama.cpp; OpenJet's engine on Windows is in progress | [Windows install](docs/installation.md#one-line-install) |
| Any other GPU, or Apple silicon | The best model that fits, on llama.cpp | [Pick by memory](docs/models.md#pick-by-memory) |
| Jetson | llama.cpp | [Jetson](docs/deployment/jetson.md) |
| CPU only | llama.cpp | [CPU-only](docs/deployment/cpu-only.md) |

The engine's model is unsloth's [Qwen3.8-27B-Q4_K_M.gguf](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/blob/566347e99d832d31f6245a577198d10d5bdd389e/Qwen3.8-27B-Q4_K_M.gguf), which `openjet setup` downloads for
you. [All requirements](docs/installation.md#what-you-need) · [Every model file](docs/models.md)

## 4 coding agents on one GPU

https://github.com/user-attachments/assets/a7fc5010-110d-45fd-9774-a42afa1f3eb9


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

## Results on your GPU

The numbers above are from one RTX 3090. These are from everyone else's cards, measured with the same
fixed workload:

```bash
openjet bench --github <your-handle>
```

It runs the model `openjet setup` configured, on whichever engine serves it, and writes a results file.
Add the file to [`benchmarks/community/`](benchmarks/community/) in a pull request, or paste it into a
[Benchmark issue](https://github.com/l-forster/open-jet/issues/new?template=benchmark.yml). Results-only
pull requests are merged the same day, and contributors are named in the release notes.

<!-- community-results:start -->
No results yet. Run `openjet bench` and send yours; the first row could be your card.
<!-- community-results:end -->

Reasoning: thinking on, OpenJet's thinking preset, 4 coding prompts. Greedy: thinking off, 4 chat
prompts. At 32K: a fresh 32K-token prompt, then decoding at that depth. Recall: a line hidden halfway
into a 4K, 32K or 96K prompt, found or not. Each row is the median of its runs; rows differ in driver,
power limit, context window and OpenJet version, which each file records. See
[what a run measures](benchmarks/community/README.md#what-a-run-measures).

## More

### OpenJet's terminal agent


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
[TypeScript terminal UI](docs/architecture/typescript-tui.md).

With a local model, no code or prompts leave your machine. `airgapped: true` also blocks every
non-loopback model and tool endpoint.

### Tuned to the hardware

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
  results file to share ([Results on your GPU](#results-on-your-gpu)). `openjet benchmark --sweep`
  tunes llama.cpp's GPU layers, batch sizes and thread counts with `llama-bench`
  ([Benchmarking](docs/benchmarking.md)).

### Slipstream: a frontier model plans, your local model works

Slipstream pairs a frontier model (OpenAI Codex through your ChatGPT subscription, or any
OpenRouter model) with your local model. The frontier model handles intent, architecture,
decomposition and the final review. The local model reads the repo, edits files, runs tests and
works through failures. Only its conclusions go back to the frontier model, not its exploration.

The target is a 20% frontier share of tokens, about 5x more work from the same plan. The footer
shows your actual share every session; small or high-risk tasks use more. The local side isn't
free either: it costs GPU time.

Start it with `/mode` → **Slipstream**. Connect with `/connect openai-codex`, or `/login` /
`/connect openrouter`. See [Usage: Slipstream](docs/usage/slipstream.md).

### Use Ollama, LM Studio or an existing llama-server

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

### Python SDK: embed a local LLM in your application

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

### FAQ

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

### Docs

- Start: [Getting started](docs/getting-started.md) · [Quickstart](docs/quickstart.md) ·
  [Installation](docs/installation.md) · [Configuration](docs/configuration.md) ·
  [Choosing a model](docs/models.md) · [Overview](docs/overview.md)
- Coding tools: [Claude Code, Codex, OpenCode and OpenClaw](docs/clients.md)
- Runtime: [OpenJet's engine](megakernel/README.md) · [llama.cpp](docs/runtimes/llama-cpp.md) ·
  [Benchmarking](docs/benchmarking.md)
- Agent: [CLI](docs/usage/cli.md) · [Slash commands](docs/usage/slash-commands.md) ·
  [Slipstream](docs/usage/slipstream.md) · [Skills](docs/usage/skills.md) ·
  [MCP](docs/usage/mcp.md) · [Device sources](docs/usage/device-sources.md) ·
  [Workflow harness](docs/usage/workflow-harness.md) ·
  [Session state and logging](docs/usage/session-state-and-logging.md)
- SDK: [Quickstart](docs/sdk/quickstart.md) · [Python SDK](docs/sdk/python-sdk.md)
- For AI tools and writers: [AGENTS.md](AGENTS.md) · [Install with an agent](docs/agent-install.md) ·
  [AI reference](docs/AI_REFERENCE.md) · [Media kit](docs/MEDIA_KIT.md)
- Deployment: [Jetson](docs/deployment/jetson.md) ·
  [Linux x86 + NVIDIA](docs/deployment/linux-x86-nvidia.md) · [CPU-only](docs/deployment/cpu-only.md)

### Community

[Discord](https://discord.gg/pspKHtExSa) · [X / Twitter](https://x.com/FlouisLF). Results from other
hardware are welcome: run `openjet bench` and send the file ([Results on your GPU](#results-on-your-gpu)).

### License

OpenJet core is `AGPL-3.0-only`: you can use, modify and redistribute the SDK and CLI under the
AGPL v3, including its network-use source requirements. Paid hosted, team or enterprise features
may ship separately under commercial terms. Contributions are accepted under
[CONTRIBUTING.md](CONTRIBUTING.md) and [CLA.md](CLA.md).
