# Runtime: llama.cpp

`open-jet` uses `llama-server` from [llama.cpp](https://github.com/ggerganov/llama.cpp) as its local/offline inference backend.

In this simplified build, `llama.cpp` is the only supported local runtime.

Pre-built binaries are available for x86, while Jetson/ARM64 commonly requires source build with device-appropriate flags.

Use the deployment-specific build guides:

- [Jetson](../deployment/jetson.md)
- [Linux x86 + NVIDIA](../deployment/linux-x86-nvidia.md)
- [CPU-only](../deployment/cpu-only.md)

The `llama-server` binary can be added to `PATH` or left in `~/llama.cpp/build/bin/`.

## What setup automates

When you run `open-jet --setup` for the local runtime, OpenJet can:

- reuse an existing `llama-server` binary
- clone and build `llama.cpp` if `llama-server` is missing
- use a local `.gguf`
- resolve an installed Ollama model
- pull an Ollama model
- download a recommended GGUF if you do not have a local model ready

Automatic provisioning still depends on local prerequisites like network access, `git`, `cmake`, and a working compiler toolchain.

## Current OpenJet integration

When `llama.cpp` is the active runtime, OpenJet can:

- stream chat and tool calls through `llama-server`
- toggle reasoning mode per request with `/reasoning`
- clear or rebuild the KV cache with `/clear`
- save and restore slot state for low-memory shell workflows

## Low-memory shell swap behavior

For memory-heavy shell commands, OpenJet can temporarily unload the active `llama-server` process, run the command, then reload the model.

Current behavior:

- this path is only wired up for the `llama.cpp` runtime
- OpenJet starts `llama-server` with `--slot-save-path .openjet/state/swap`
- before unload, it saves conversation messages and attempts KV-cache save through the llama slot API
- after the shell command completes, it reloads the server and attempts KV-cache restore
- if KV restore fails, OpenJet keeps the restored messages and falls back to re-prompting the conversation history

Swap state is stored under `.openjet/state/swap/`.
