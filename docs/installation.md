# Installation

## Clone and install

```bash
git clone https://github.com/l-forster/open-jet.git
cd open-jet
./install.sh
```

This creates a local virtualenv, installs the repo, and links `open-jet` / `openjet` into `~/.local/bin`.

## Supported runtimes

- `llama.cpp` for local/offline inference
- `openai_compatible` for self-hosted or compatible OpenAI-style APIs
- `openrouter` as an optional hosted profile

OpenJet is positioned for local and self-hosted use first.

## Recommended path: local `llama.cpp`

Start with:

```bash
open-jet --setup
```

Choose `Local model: llama.cpp (GGUF)`.

OpenJet will first try to reuse an existing local setup. If pieces are missing, setup can provision them for you.

The local path can:

- reuse `llama-server` if it is already installed
- clone and build `llama.cpp` to produce `llama-server`
- use a local `.gguf`
- resolve an installed Ollama model
- pull an Ollama model
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

## Self-hosted API path

OpenAI-compatible:

```bash
export OPENAI_API_KEY=your-key
open-jet --setup
```

Choose `Self-hosted API: OpenAI-compatible`.

## Optional hosted path

OpenRouter:

```bash
export OPENROUTER_API_KEY=your-key
open-jet --setup
```

Choose `Hosted API: OpenRouter`.

## Public SDK import

For scripts and integrations, use:

```python
from open_jet import OpenJetSession
```
