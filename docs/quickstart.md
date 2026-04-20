# Quickstart

## 1. Install

```bash
git clone https://github.com/l-forster/open-jet.git
cd open-jet
./install.sh
```

This is the normal editable-source install path.

## 2. Run setup

Start with:

```bash
openjet setup
```

If you already have local runtime pieces, setup will reuse them. If not, setup can:

- use a local `.gguf` model file
- download a recommended GGUF
- provision `llama-server` from `llama.cpp`

Automatic provisioning still depends on normal local prerequisites:

- network access for downloads and git clone operations
- `git`, `cmake`, and a working build toolchain if `llama.cpp` needs to be built

## 3. Launch OpenJet

```bash
openjet
```

Other primary entrypoints from the same install:

```bash
openjet benchmark --sweep
```

```python
from openjet.sdk import OpenJetSession, recommend_hardware_config
```

## 4. What setup configures

The setup wizard now focuses on:

1. hardware profile
2. local model selection or direct download
3. `llama-server` reuse or build
4. context window
5. GPU layers

## 5. Next docs

- [Product surfaces](overview.md)
- [Installation](installation.md)
- [Configuration](configuration.md)
- [Backend Workflows](usage/backend-workflows.md)
- [Runtime: llama.cpp](runtimes/llama-cpp.md)
- [Python SDK](sdk/python-sdk.md)
- [Benchmarking](benchmarking.md)
