# Benchmarking

OpenJet includes a benchmarking surface built around `llama-bench` and the
currently active model profile from `config.yaml`.

Use it when you want to answer questions like:

- how fast does this model run on this machine
- what `gpu_layers` setting is best here
- how does throughput change with batch size or thread count

## Single run

Run one benchmark against the active model profile:

```bash
openjet benchmark
openjet benchmark --mode standard
```

Useful flags:

```bash
openjet benchmark -p 1024 -n 256 -r 3 -o md
openjet benchmark -- -ub 512
```

OpenJet passes the active model path, configured device, and GPU layer count to
`llama-bench`, then prints the result directly.

## Turbo DFlash mode

OpenJet also has an experimental benchmark path for DFlash speculative decoding:

```bash
openjet turbo benchmark
openjet benchmark --mode turbo
openjet benchmark --mode thinking
```

Turbo mode can set up the Lucebox backend itself. With
`--backend-kind lucebox`, OpenJet will use `.openjet/turbo/lucebox-hub`, clone
`Luce-Org/lucebox-hub` if needed, build `dflash/build/test_dflash`, install the
local HTTP server Python dependencies, reuse the active OpenJet target GGUF,
download the z-lab draft when no draft path is configured, then run the
benchmark.

The benchmark starts a temporary local server, sends a fixed local-agent coding
prompt through the OpenAI-compatible API, and prints:

- hardware and CUDA status
- target and draft model names
- context size
- thinking mode on/off
- backend path
- prompt eval tok/s
- generation tok/s
- speedup versus a provided baseline

`--mode turbo` disables Qwen thinking mode. `--mode thinking` enables it for
comparison, but this is expected to hurt DFlash acceptance with the current
Qwen3.6-27B drafter because it was not trained on `<think>...</think>` output.

Supported backends:

- `llama-server`: DFlash-capable `llama-server` such as `spiritbuun/buun-llama-cpp`
- `lucebox`: `Luce-Org/lucebox-hub/dflash`, using its daemonized `scripts/server.py`
  around `build/test_dflash`

Required setup for `llama-server` mode:

- target GGUF: Qwen3.6-27B, preferably Q4_K_M or similar 4-bit quant
- draft GGUF: `spiritbuun/Qwen3.6-27B-DFlash-GGUF`, preferably
  `dflash-draft-3.6-q8_0.gguf`
- backend: `spiritbuun/buun-llama-cpp`; DFlash is not expected in upstream
  `llama.cpp`
- CUDA/NVIDIA GPU, such as RTX 3090-class hardware

Automatic setup for `lucebox` mode needs:

- `git`, `cmake`, a CUDA compiler/toolkit, and an sm_86+ NVIDIA GPU
- an existing Qwen3.5/Qwen3.6 27B target GGUF from `openjet setup` or
  `--target-model`; OpenJet will not download a second target GGUF
- enough disk for the checkout plus draft download
- Hugging Face access for gated draft repos; set `HF_TOKEN` first if required

Configuration can live in `config.yaml`:

```yaml
turbo:
  backend_kind: llama-server
  llama_server_path: /home/you/buun-llama-cpp/build/bin/llama-server
  target_model: /models/Qwen3.6-27B-Q4_K_M.gguf
  draft_model: /models/dflash-draft-3.6-q8_0.gguf
  context_window_tokens: 6048
  baseline_tok_s: 34.5
```

For Lucebox:

```yaml
turbo:
  backend_kind: lucebox
  context_window_tokens: 6048
```

That is enough for the default auto-provisioned path. You can override paths if
you already have files elsewhere:

```yaml
turbo:
  backend_kind: lucebox
  lucebox_root: /home/you/lucebox-hub/dflash
  target_model: /models/Qwen3.6-27B-Q4_K_M.gguf
  draft_model: /home/you/lucebox-hub/dflash/models/draft/model.safetensors
```

The same values can be provided on the command line:

```bash
openjet turbo benchmark \
  --backend-kind llama-server \
  --backend-path /home/you/buun-llama-cpp/build/bin/llama-server \
  --target-model /models/Qwen3.6-27B-Q4_K_M.gguf \
  --draft-model /models/dflash-draft-3.6-q8_0.gguf \
  --context 6048 \
  --baseline-tok-s 34.5
```

```bash
openjet turbo benchmark \
  --backend-kind lucebox \
  --context 6048
```

## Sweep mode

Run a one-variable-at-a-time sweep:

```bash
openjet benchmark --sweep
```

Sweep mode currently checks:

- GPU layer offloading when the active device is not CPU
- batch size
- thread scaling

This is the quickest way to compare settings on the same machine without
rewriting config by hand.

## Prerequisites

OpenJet expects `llama-bench` to exist in one of these places:

- next to the configured `llama_server_path`
- on `PATH`
- `~/llama.cpp/build/bin/llama-bench`

If it is missing, build it from your `llama.cpp` checkout:

```bash
cd ~/llama.cpp
cmake --build build --target llama-bench
```

## How it picks settings

Benchmarking reuses the same configuration model as the CLI and SDK:

- `llama_model`
- `device`
- `gpu_layers`
- the active model profile selected in `config.yaml`

That means the benchmark surface measures the same runtime shape that the TUI
and SDK would use.

## Related surfaces

- If you want to **use** OpenJet interactively, go to [CLI usage](usage/cli.md).
- If you want to **embed** OpenJet in another app, go to [Python SDK](sdk/python-sdk.md).
- If you want to **tune** the runtime before benchmarking, see [Configuration](configuration.md).
