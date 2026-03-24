# Quickstart

## 1. Install

```bash
git clone https://github.com/l-forster/open-jet.git
cd open-jet
./install.sh
```

`open-jet` now supports a deliberately smaller runtime set:

- `llama.cpp` for local/offline use
- `openai_compatible` for self-hosted or compatible OpenAI-style APIs
- `openrouter` for OpenRouter

`SGLang` and `TensorRT-LLM` are disabled in this simplified build.

OpenJet is positioned for local and self-hosted use first.

## 2. Recommended path: local `llama.cpp`

Start with:

```bash
open-jet --setup
```

Choose `Local model: llama.cpp (GGUF)`.

If you already have local runtime pieces, setup will reuse them. If not, setup can:

- use a local `.gguf` model file
- resolve an installed Ollama model
- pull a recommended Ollama model
- download a recommended GGUF
- provision `llama-server` from `llama.cpp`

Automatic provisioning still depends on normal local prerequisites:

- network access for downloads and git clone operations
- `git`, `cmake`, and a working build toolchain if `llama.cpp` needs to be built

## 3. Self-hosted gateway path

```bash
export OPENAI_API_KEY=your-key
open-jet --setup
```

Choose `Self-hosted API: OpenAI-compatible` and enter:

1. model id
2. base URL
3. API key env var name

This path is for people who already run an OpenAI-compatible gateway or service.

## 4. Optional hosted fallback

```bash
export OPENROUTER_API_KEY=your-key
open-jet --setup
```

Choose `Hosted API: OpenRouter` and enter:

1. model id
2. API key env var name

## 5. Launch OpenJet

```bash
open-jet
```

## 6. What setup configures

The setup wizard now focuses on:

1. hardware profile
2. runtime selection
3. model or API configuration
4. context window
5. GPU layers for local `llama.cpp`

## 7. Next docs

- [Installation](installation.md)
- [Configuration](configuration.md)
- [Backend Workflows](usage/backend-workflows.md)
- [Runtime: llama.cpp](runtimes/llama-cpp.md)
- [Runtime: OpenAI-compatible](runtimes/openai-compatible.md)
- [Runtime: OpenRouter](runtimes/openrouter.md)
- [Python SDK](sdk/python-sdk.md)
