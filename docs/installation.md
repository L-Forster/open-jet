# Installation

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

This creates a local virtualenv, installs the repo, creates `openjet` / `openjet` command launchers, and starts setup.
It is an editable Python install, not a compiled release build.

If stale in-place compiled extension artifacts exist under `src/`, the installer removes them so imports continue to use the source tree.

## Optional compiled wheel

If you want a compiled package artifact, build it separately:

```bash
./scripts/build-compiled.sh
```

This produces a wheel in `dist/` and keeps the normal editable install path separate.

## Runtime

OpenJet is local-only. It uses `llama-server` from `llama.cpp` with GGUF models.

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
