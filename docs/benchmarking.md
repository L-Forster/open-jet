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
open-jet benchmark
```

Useful flags:

```bash
open-jet benchmark -p 1024 -n 256 -r 3 -o md
open-jet benchmark -- -ub 512
```

OpenJet passes the active model path, configured device, and GPU layer count to
`llama-bench`, then prints the result directly.

## Sweep mode

Run a one-variable-at-a-time sweep:

```bash
open-jet benchmark --sweep
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
