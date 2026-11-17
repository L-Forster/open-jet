# Runtime: llama.cpp

`openjet` uses `llama-server` from [llama.cpp](https://github.com/ggerganov/llama.cpp) as its local/offline inference backend.

For supported Qwen3.8-27B models OpenJet uses [its own engine](../../megakernel/README.md) instead.
Its README covers standalone installation and the HTTP API.

Pre-built binaries are available for x86, while Jetson/ARM64 commonly requires source build with device-appropriate flags.

Use the deployment-specific build guides:

- [Jetson](../deployment/jetson.md)
- [Linux x86 + NVIDIA](../deployment/linux-x86-nvidia.md)
- [CPU-only](../deployment/cpu-only.md)

The `llama-server` binary can be added to `PATH` or left in `~/llama.cpp/build/bin/`.

## OpenJet's engine (RTX 30/40/50 series)

For one specific setup, Qwen3.8-27B Q4_K_M or UD-Q4_K_XL on an RTX 30, 40 or 50-series GPU with 24 GB+
(sm_86, sm_89, sm_120), OpenJet ships a specialised CUDA engine in [`megakernel/`](../../megakernel/README.md)
and serves the model through it instead of `llama-server`. The previous engine, in
[`megakernel/legacy/`](../../megakernel/legacy/README.md), serves Turing (sm_75, experimental, Q4_K_M only;
see [Turing (RTX 8000)](../../megakernel/legacy/README.md#turing-rtx-8000)) and UD-Q3_K_XL. Measured with
the previous engine on an RTX 3090 against `llama-server` with OpenJet's flags:

| | llama.cpp | OpenJet, previous engine |
|---|---|---|
| reasoning mode, OpenJet's thinking preset (decode) | 60 tok/s | 92 tok/s (1.5x) |
| reasoning mode, 98K tokens in context (decode) | 37 tok/s | 79 tok/s (2.2x) |
| reading a 98K-token prompt | 112 s | 82 s (1.4x) |
| greedy decode, chat prompts (mean; earlier build) | 63 tok/s | 107 tok/s (1.7x) |
| prompt processing, 2-8K tokens | 1050-1140 tok/s | 1530-1600 tok/s (1.3-1.5x) |
| pasting a 12K-token file into the next turn | 10.5 s | 8.1 s |

The current engine decodes 1.36x as fast as the previous one on short prompts and 1.25x with 98K tokens
in context, thinking on, sampled, on the same RTX 3090 ([results](../../megakernel/README.md#results)).

`openjet setup` installs it when it selects this model on a supported GPU. An installed wheel
downloads the engine prebuilt for that GPU's architecture from the matching GitHub release, which
needs only the NVIDIA driver. A source checkout, or a release without a matching archive, compiles
the engine source for that architecture against the CUDA `llama.cpp` build setup makes (for its
tokenizer). That needs the CUDA toolkit (`nvcc`, CUDA 13.2), about 21 GiB of RAM and up to half an
hour; on a machine with less than 24 GiB of RAM, or without `nvcc`, setup says so and OpenJet uses
`llama-server`. On Windows the engine only comes prebuilt. In a source checkout you can also build
by hand:

```bash
make -C megakernel -j1 libqwen38mk.so
```

The previous engine builds separately (`make -C megakernel/legacy libqwen38mk.so`, or
`libqwen38mk_mixed.so` for UD-Q3_K_XL), in a few minutes and under 12 GiB of RAM. Setup picks the
engine from the GPU and the downloaded GGUF's tensor types, builds only that one, and caches each
build separately ([build requirements](../../megakernel/README.md#build--run)).

With `llama_backend: auto` (the default) OpenJet then starts the engine's `server/mk_server.py`
whenever the configured model is Qwen3.8-27B Q4_K_M, UD-Q4_K_XL or UD-Q3_K_XL, the engine it needs is
installed, the GPU is one that engine supports and `jinja2` is installed; in every other case, or if the megakernel fails to start,
it starts `llama-server` as usual. `llama_backend: llama` turns it off, `llama_backend: megakernel`
makes it required. `OPENJET_MEGAKERNEL_DIR` points at a megakernel build elsewhere.

The server speaks OpenAI Chat Completions and Responses and Anthropic Messages (see
[Coding tools](../clients.md)). For OpenJet's agent: streaming chat completions rendered with
the model's own chat template, reasoning as `reasoning_content`, tool calls as structured
`tool_calls`, and reuse of the processed conversation between turns. Differences from
`llama-server`:

- sampling supports temperature, top_k (at most 32), top_p, min_p and the presence and frequency
  penalties; `repetition_penalty` is ignored
- OpenJet starts it with the q8_0 KV cache
- no slot save / restore: the low-memory shell swap below falls back to re-prompting
- editing an earlier part of a conversation (e.g. context compaction) resumes from the nearest saved
  checkpoint before the edit and re-processes the rest

## What setup automates

When you run `openjet setup`, OpenJet can:

- reuse an existing `llama-server` binary
- clone and build `llama.cpp` if `llama-server` is missing
- use a local `.gguf`
- download a recommended GGUF if you do not have a local model ready

Automatic provisioning still depends on local prerequisites like network access, `git`, `cmake`, and a working compiler toolchain.

## Current OpenJet integration

When `llama.cpp` is the active runtime, OpenJet can:

- stream chat and tool calls through `llama-server` (or the megakernel server, see above)
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
