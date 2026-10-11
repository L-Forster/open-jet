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
```

Useful flags:

```bash
openjet benchmark -p 1024 -n 256 -r 3 -o md
openjet benchmark -- -ub 512
```

OpenJet passes the active model path, configured device, and GPU layer count to
`llama-bench`, then prints the result directly.

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

## Megakernel benchmarks

`openjet benchmark` measures `llama-bench`. The Qwen3.8 megakernel has its own benchmarks
against `llama-server` in `megakernel/bench/`: `think_bench.py` (reasoning mode with OpenJet's
sampling preset), `code_bench.py` (greedy decode, a context-length sweep and second-turn prompt
time) and `bench.py`. See [megakernel/README.md](../megakernel/README.md).

### Matched reasoning comparison on two GPUs

Run from the repository root with a built megakernel and the original unsloth
`Qwen3.8-27B-Q4_K_M.gguf` (catalog revision `566347e99d832d31f6245a577198d10d5bdd389e`).
For Turing, build with CUDA 12.x and `ARCH=sm_75`; CUDA 13 no longer compiles this architecture.
See [Turing build and validation](../megakernel/README.md#turing-rtx-8000).

Use full UUIDs from `nvidia-smi -L` rather than assuming CUDA ordinals match NVIDIA indexes:

```bash
export LLAMA_GPU_UUID="GPU-REPLACE-WITH-BASELINE-FULL-UUID"
export MK_GPU_UUID="GPU-REPLACE-WITH-MEGAKERNEL-FULL-UUID"
python megakernel/bench/think_bench.py \
    --model models/Qwen3.8-27B-Q4_K_M.gguf \
    --llama-server /path/to/llama.cpp/build/bin/llama-server \
    --megakernel megakernel/qwen38_mk \
    --llama-gpu "$LLAMA_GPU_UUID" --megakernel-gpu "$MK_GPU_UUID" \
    --kv q8 --context 16384 --seeds 2 -n 256 --output rtx8000-repeat.json
```

`--model`, `--llama-server` and `--megakernel` override the GGUF and executable paths.
`--llama-gpu` and `--megakernel-gpu` select each process's device independently, accepting one
full UUID or numeric PCI-bus-order ordinal and overriding inherited `CUDA_VISIBLE_DEVICES`.
Without these flags, each process inherits the CUDA environment. The llama-server port 8094
must be free; the script refuses to reuse an existing server.

`--context` sets both engines' context **capacity**, not the number of prompt tokens.
`--kv q8` matches the megakernel to llama.cpp's fixed q8_0 KV cache; the megakernel default
`auto` may select fp16. `--seeds 2` uses seeds 1000 and 1001 for each of five identical rendered
prompts, and `-n 256` caps generation per sample. Thinking/xhigh uses temperature 1.0,
top_p 0.95, top_k 20 and min_p 0.0. The script runs llama.cpp with MTP 2 and the megakernel
with both 3 and 4 drafts. One prompt includes local `prefill.cuh` source, so retain the JSON
prompts when comparing different source revisions.

`--output` writes settings, rendered prompts and all raw timing samples to JSON (its parent
directory must already exist). Per-prompt table entries average all seeds; overall throughput
is total timed decode tokens divided by total decode seconds, not best-of. Timing is decode-only:
the megakernel's first generated token belongs to prefill and is excluded. In the RTX 8000 run,
this meant 2,550 timed megakernel tokens per draft setting versus 2,560 for llama.cpp.
Model load, prefill and end-to-end chat latency are not included.

The short, UUID-pinned RTX 8000 run measured 54.14 tok/s with 3 drafts and 53.69 with 4 on
GPU 3, versus llama.cpp MTP 2 at 32.88 on GPU 2 (both Quadro RTX 8000). Cross-card performance
can depend on clocks and other load; these results do not establish Turing KLD, model-quality
equivalence or long-context behavior. The baseline reported `0 (unknown)`, not a verified
`b10246` build. The [sanitized measurement record](benchmarks/rtx8000-turing.json) publishes
all 30 timing samples, model and binary fingerprints, aggregates and validation limits.
Full logs, prompts and UUIDs under `benchmark_results/` remain ignored machine-local artifacts.
Save executable/library fingerprints and GPU UUIDs alongside repeat results rather than
inferring a baseline version.

### Fresh-context prefill comparison on two GPUs

`megakernel/bench/prefill_bench.py` measures 2,048, 4,096 and 8,192 **actual input tokens**
at a 16,384-token context capacity. Prepare shared token IDs once, then run each backend on its
own GPU using the same workload file:

```bash
python megakernel/bench/prefill_bench.py --backend prepare \
    --model models/Qwen3.8-27B-Q4_K_M.gguf --gpu "$MK_GPU_UUID" \
    --output prefill-workloads.json
python megakernel/bench/prefill_bench.py --backend llama \
    --model models/Qwen3.8-27B-Q4_K_M.gguf \
    --llama-server /path/to/llama.cpp/build/bin/llama-server --gpu "$LLAMA_GPU_UUID" \
    --workload prefill-workloads.json --output prefill-llama.json
python megakernel/bench/prefill_bench.py --backend megakernel \
    --model models/Qwen3.8-27B-Q4_K_M.gguf --gpu "$MK_GPU_UUID" \
    --workload prefill-workloads.json --output prefill-megakernel.json
```

The preparation command loads the existing megakernel library to tokenize an original CC0
synthetic garden ledger/Python-code corpus; it does not run inference. The measured run's IDs
were prepared with vocab-only libllama on CPU. Both use the same model tokenizer, with special
tokens parsed and no automatic special-token insertion. Each length is a prefix of one raw
tokenization; the JSON records exact IDs, model identity and token hashes. Keep the workload file
when comparing different tokenizer/library builds.

Each backend loads its model once. One excluded, same-process warmup per input length precedes
three timed repetitions. The megakernel uses the existing synchronous C API, q8 KV, four initial
MTP drafts and the 32K draft vocabulary; reset and empty-sequence checks precede every request.
llama.cpp uses q8_0 KV, MTP 2, batch 2,048 and microbatch 512. Its owned server erases slot 0
before every request, disables prompt caching and checks that the full input was evaluated
without truncation. Port 8095 must be free. `--slot-save-path` is provided so slot erasure is
enabled, not merely slot monitoring.

These are **warmed, fresh-context prompt-processing timings**, not pure GEMM speed or
end-to-end HTTP time-to-first-token. Both evaluate the whole input and produce one token, with
no subsequent decode. llama.cpp's native prompt timer ends after synchronized first-token
sampling in the available server source; the measured binary still reports an unknown version.
The megakernel timer wraps synchronous `Engine.prefill`, including Python/ctypes input conversion,
C++ vector copying, the last prompt token's persistent-kernel step, checkpoint work and initial
draft setup. Model loading, reset/slot erase and warmup allocation are outside the timed samples.
The scopes are not identical, so throughput ratios describe these paths rather than isolated
prefill kernels.

All three repetitions contribute to each result: total input tokens divided by total measured
seconds, not best-of. Outputs retain raw samples and fingerprints. `--mk-mode cli` is an optional
alternative with fresh processes; it includes per-process lazy allocation and is **not** the
warmed C API method used here.

Measured on GPU 2 (llama.cpp) and GPU 3 (megakernel), both Quadro RTX 8000:

| Input tokens | llama.cpp mean s | llama.cpp tok/s | megakernel mean s | megakernel tok/s | Measured-path ratio |
|---|---:|---:|---:|---:|---:|
| 2,048 | 3.228 | 634.40 | 2.802 | 730.80 | 1.15x |
| 4,096 | 6.906 | 593.12 | 6.281 | 652.17 | 1.10x |
| 8,192 | 15.922 | 514.51 | 13.720 | 597.10 | 1.16x |

The [sanitized prefill record](benchmarks/rtx8000-turing-prefill.json) contains all 18 timed
samples, exact input hashes, settings and final-library fingerprints. The repeat preparation
command was verified to reproduce the measured CPU-tokenized inputs exactly. These measurements
use the final rebuilt megakernel library, unlike the earlier reasoning-decode measurements.
Results from one synthetic workload do not establish KLD, output quality, long-context quality
or universal speedups; cross-card clocks/load and the different timer overheads remain caveats.

## Related surfaces

- If you want to **use** OpenJet interactively, go to [CLI usage](usage/cli.md).
- If you want to **embed** OpenJet in another app, go to [Python SDK](sdk/python-sdk.md).
- If you want to **tune** the runtime before benchmarking, see [Configuration](configuration.md).
