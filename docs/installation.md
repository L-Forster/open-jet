# Installation

English · [简体中文](zh/installation.md)

## What you need

| Computer | What runs |
|---|---|
| Linux, NVIDIA RTX 30, 40 or 50-series with 24 GB+ (`sm_86`, `sm_89`, `sm_120`) | Qwen3.8-27B on [OpenJet's engine](../megakernel/README.md) |
| Linux, NVIDIA RTX 30, 40 or 50-series with 6-12 GB and 16 GB+ RAM | Inkling 2 Mini (`UD-IQ4_XS` or `UD-IQ3_XXS`) on [OpenJet's Inkling engine](inkling.md) |
| 16 GB AMD or Intel Vulkan laptop (Windows or Linux, 6-10 GB Vulkan heap) | Inkling 2 Mini `UD-IQ3_XXS` on `llama.cpp` (`b12651`) Vulkan with `--cpu-moe` |
| No 24 GB NVIDIA card (cloud option) | Qwen3.8-27B on OpenJet's engine, on a rented RTX 3090 (~$0.20/hour on Vast.ai, [deploy](https://www.openjet.dev/pricing#deploy)) |
| Windows | `llama.cpp` prebuilt binaries (`openjet setup` fetches Vulkan/CUDA builds); Windows builds of OpenJet's engine are in progress |
| NVIDIA Turing with 24 GB+ (`sm_75`, e.g. Quadro RTX 8000) | OpenJet's legacy engine, experimental ([Turing](../megakernel/legacy/README.md#turing-rtx-8000)) |
| Any other GPU, Apple silicon, Jetson, or CPU only | Best model that fits on `llama.cpp` ([pick by memory](models.md#pick-by-memory), [Jetson](deployment/jetson.md), [CPU-only](deployment/cpu-only.md)) |

For OpenJet's engine you also need an NVIDIA driver that supports CUDA 13.2 (580 or newer); the
legacy engine, which serves Turing, needs CUDA 12.8 (570 or newer). `openjet setup` downloads the
engine prebuilt for your GPU ([which one](#prebuilt-engines)), so you don't need the CUDA toolkit. If
no prebuilt engine matches, setup compiles it, which needs `nvcc` (CUDA 13.2) and about 22 GiB of RAM;
on a machine with less than 24 GiB setup doesn't try. If the engine can't be installed, setup says why
and serves the model with llama.cpp instead.

Context on a 24 GB card is about 120K tokens with the default q8_0 KV cache, or up to about 234K
with `openjet setup --long-context`, which stores the cache in 5 bits and keeps the most recent 2K
positions at q8_0 ([KV cache presets](../megakernel/README.md#kv-cache-presets-mk_kv)).

Which model files run where, with download links: [Choosing a model](models.md).

## Prebuilt engines

Each [GitHub release](https://github.com/L-Forster/open-jet/releases) carries one prebuilt engine bundle per OS
holding every supported GPU architecture (`sm_75`, `sm_86`, `sm_89`, `sm_120`):

- **Linux:** [openjet-engine-linux-x64.tar.gz](https://github.com/L-Forster/open-jet/releases/latest/download/openjet-engine-linux-x64.tar.gz) - Qwen3.8-27B (current and legacy/mixed), Inkling 2 Mini (`sm_86`, `sm_89`, `sm_120`, with `profiles/default.f32`), and CPU-only `libllama` tokenizers.
- **Windows:** `openjet-engine-windows-x64.zip` - Qwen3.8-27B (in progress; until released, OpenJet on Windows serves models with prebuilt `llama.cpp`).

`openjet setup` reads your GPU's compute capability (`nvidia-smi --query-gpu=name,compute_cap --format=csv`, where 8.6 is `sm_86`), downloads the bundle for your OS from the matching `open-jet` release, and extracts only your GPU's slice automatically. CUDA is linked in, so only the NVIDIA driver is needed.

## One-line install

Linux and macOS:

```bash
curl -fsSL https://github.com/L-Forster/open-jet/releases/latest/download/install.sh | sh
```

Windows PowerShell:

```powershell
irm https://github.com/L-Forster/open-jet/releases/latest/download/install.ps1 | iex
```

The installer uses [uv](https://docs.astral.sh/uv/) to download its own Python 3.12, separate from
your system's, and installs the `open-jet` package from PyPI with it. `openjet` goes in
`~/.local/bin` (`%USERPROFILE%\.local\bin` on Windows), which the installer adds to your PATH. To
install a particular version, set `OPENJET_VERSION` first (for example `OPENJET_VERSION=0.5.4`).
Run the same command again to update. With Python already installed, `pipx install open-jet` does
the same.

## Install with your agent

Ask Codex or Claude Code to [follow the agent installation instructions](agent-install.md). It can inspect your hardware, choose a model, configure OpenJet without the wizard, verify inference, and give you usage commands.

Once installed, the noninteractive path is:

```bash
openjet setup --non-interactive --dry-run --json
openjet setup --non-interactive --yes --json
```

## Clone and install

```bash
git clone https://github.com/l-forster/open-jet.git
cd open-jet
./install.sh
```

On macOS, use:

```bash
./install-macos.sh
```

On Windows, run:

```bat
curl -L https://www.openjet.dev/install.bat -o install.bat && install.bat
```

This creates a local virtualenv, installs the repo, creates `open-jet` / `openjet` command launchers, and exits. Run `openjet setup --yes` to accept the hardware recommendation or `openjet setup` for the wizard.
It is an editable Python install, not a compiled release build.

If stale in-place compiled extension artifacts exist under `src/`, the installer removes them so imports continue to use the source tree.

## Optional compiled wheel

If you want a compiled package artifact, build it separately:

```bash
./scripts/build-compiled.sh
```

This produces a wheel in `dist/` and keeps the normal editable install path separate.

## Runtime

OpenJet runs GGUF models locally: Qwen3.8-27B on OpenJet's engine where the GPU supports it,
everything else on `llama-server` from `llama.cpp`.

## Recommended path

Start with:

```bash
openjet setup
```

OpenJet will first try to reuse an existing local setup. If pieces are missing, setup can provision them for you.

Setup can:

- reuse `llama-server` if it is already installed
- clone and build `llama.cpp` to produce `llama-server`
- use a local `.gguf`
- download a recommended GGUF directly

Automatic provisioning still needs:

- network access
- `git`
- `cmake`
- a working build toolchain

Manual build guides are still available if you want to control the runtime yourself:

- [Jetson setup](deployment/jetson.md)
- [Linux x86 + NVIDIA](deployment/linux-x86-nvidia.md)
- [CPU-only](deployment/cpu-only.md)

## Public SDK import

For scripts and integrations, use:

```python
from openjet.sdk import OpenJetSession
```
