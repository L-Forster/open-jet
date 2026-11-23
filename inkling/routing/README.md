# Inkling 2 Mini expert routing study

Before writing an engine that streams Inkling 2 Mini's experts on small GPUs (the way Strata does
for Qwen3.8-Flash-Next), this measures what such an engine depends on: how unevenly the model
uses its 24 × 128 experts, how much the three passes of the looped layers (6–17) and consecutive
tokens reuse the same experts, and how well VRAM and RAM caches of a given size would hit.

## Files

| File | What it does |
|---|---|
| `llama-route-trace.patch` | llama.cpp patch (base `19bc41d`): with `LLAMA_ROUTE_TRACE=<file>`, llama-server appends every `ffn_moe_topk-<slot>` and `ffn_moe_weights-<slot>` tensor to the file. Slots 0–23 are the physical layers; 24–47 are passes 2 and 3 of layers 6–17 |
| `workload.py` | Sends 12 OpenJet-style requests (code review, debugging, CUDA, TypeScript, long context, multi-turn, general chat, Chinese, maths) and records each request's byte range in the trace |
| `run_trace.sh` | Starts the traced server, runs the workload, stops the server |
| `analyze.py` | Expert-use skew, loop-pass and token-to-token overlap, cache replays (static profile, adaptive, LRU, Belady bound) and per-device decode estimates |

## Reproduce

```bash
cd llama.cpp && git checkout 19bc41d && git apply open-jet/inkling/routing/llama-route-trace.patch
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 && cmake --build build -j3 --target llama-server
# mainline llama.cpp needs the inkling2_loop layout: unsloth/Inkling-2-Mini-GGUF's Mainline_Support/ files
# (renamed from .gguf_file to .gguf); the root-folder files use Unsloth's own layout and fail to load
inkling/routing/run_trace.sh
python3 inkling/routing/analyze.py --trace results/<run>/trace.bin --requests results/<run>/requests.jsonl \
  --gguf Inkling-2-Mini-UD-Q4_K_XL.gguf --out results/<run>
```

## Method notes

- Only decode tokens are analysed. Prompt tokens go through in batches of 512, where nearly every expert is used.
- A "read" is one use of an expert by one layer application, so a looped layer can read the same expert up to three
  times per token. The VRAM hit rate counts reads, because a GPU or CPU kernel reads the weights once per application.
- The SSD tier counts unique experts per token, because an expert fetched once stays in RAM for the other passes.
- Static profiles are leave-one-request-out: each request is scored against a profile built from the other requests.
- Device estimates are a model, not measurements. GPU work runs at 75% of the card's bandwidth, CPU expert work at the
  stated RAM rate, and SSD reads are not overlapped with compute.

## Results (2026-11-21, `results/2026-11-21-q4kxl/`)

The run used Unsloth's UD-Q4_K_XL (mainline layout) on an RTX 3090 with llama.cpp `19bc41d` plus the trace patch, at
temperature 1.0. It covered 12 requests, 27.5K prompt tokens and 10,736 decode tokens. Experts are 4.48 MB each in
layers 0–19 and 8.46 MB in layers 20–23, so the expert pool is 15.8 GB. A decode token reads 1.85 GB of experts,
counting each pass.

**Routing is uneven.** The most-read 12.7% of expert bytes serve 50% of reads, 36% serve 80% and 52% serve 90%. Per
layer, the top quarter of experts takes 47–74% of that layer's reads.

**The loop passes reuse experts.** On average pass 2 picks 4.8 of pass 1's 8 experts (60%), and pass 3 picks 4.3 of
pass 2's (54%). A looped layer needs 14.3 different experts per token, against 22.5 if routing were independent.
Pass 1's choices therefore predict most of what passes 2 and 3 will need, a layer ahead.

**Consecutive tokens reuse experts.** Of the experts a token needs, 48% were also needed by the previous token, 70% by
one of the previous 4, 88% within 16 and 97% within 64.

**Cache replay** (share of expert reads served by a cache holding the given share of the expert pool):

| Cache size | Static profile | Adaptive (profile + swaps every 4 tokens) | LRU | Belady (upper bound) |
|---|---|---|---|---|
| 10% | 40% | 56% | 53% | 75% |
| 20% | 59% | 73% | 73% | 87% |
| 30% | 71% | 82% | 84% | 93% |
| 40% | 80% | 88% | 90% | 96% |
| 50% | 87% | 92% | 94% | 98% |
| 70% | 95% | 97% | 98% | 99% |

Adapting to the conversation adds 8–16 points over a fixed profile at 10–40% cache. Strata reports the same effect:
50% to 72% on Qwen3.8-Flash-Next. LRU does as well as the adaptive cache once the cache holds more than one token's
working set (about 1.3 GB). Below that it thrashes (0% at 5%).

**SSD traffic** for the part of the pool that fits in neither VRAM nor RAM (LRU, unique experts per token): 419 MB per
token with 20% of the pool in RAM and VRAM, 160 MB at 40%, 57 MB at 60% and 12 MB at 80%.

**Device estimates** (`summary.json` → `devices`; a model, not measurements; batch 1, no multi-token prediction):

| Device | Experts in VRAM | VRAM read hit | Decode |
|---|---|---|---|
| 6 GB GPU + 16 GB DDR4 | 26% | 79% | ~59 tok/s |
| 8 GB GPU + 16 GB DDR4 | 39% | 88% | ~84 tok/s |
| 8 GB GPU + 8 GB DDR4, rest from a 3 GB/s SSD | 39% (60% with RAM) | 88% | ~32 tok/s (58 MB/token from the SSD) |
| 12 GB GPU (360 GB/s) + 16 GB DDR5 | 66% | 96% | ~66 tok/s (GPU-bound) |
| 16 GB GPU (448 GB/s) + 16 GB DDR5 | 94% | 99.8% | ~79 tok/s (GPU-bound) |

For comparison, the same traced llama.cpp build decoded at 33–35 tok/s with the whole model on the RTX 3090. The trace
callback adds synchronisation, so untraced llama.cpp may be somewhat faster. The 3090's bandwidth bound for the 3.5 GB
read per token is about 200 tok/s at 75% efficiency.

**llama.cpp baseline** (`bench_offload.sh`, `results/2026-11-21-offload/`): RTX 3090, Ryzen 7 5800X, DDR4, WSL
with 15 GB RAM, llama.cpp `19bc41d`. Decode tok/s for a 1,536-token and a 1,024-token answer. Only expert placement
changes, to simulate smaller cards; dense weights stay on the GPU.

| Placement | Experts on the GPU | Decode tok/s |
|---|---|---|
| Everything on the GPU | 15.8 GB | 48.9 / 41.6 |
| 12 GB card, `--n-cpu-moe 11` (layers 0–10 on the CPU) | 9.5 GB | 19.5 / 18.4 |
| 12 GB card, loop layers kept: layers 0, 1, 18–23 on the CPU | 9.2 GB | 24.4 / 24.7 |
| Layers 6–17 on the GPU, the rest on the CPU, no cache | 6.9 GB | 21.4 / 16.3 |
| Same, plus `--moe-cache-mib 2400` (llama.cpp's expert cache; 81% decode hit rate, misses uploaded) | 6.9 + 2.4 GB | 23.0 / 24.5 |
| 8 GB card, `--n-cpu-moe 18` | 5.5 GB | 11.9 / 9.5 |
| 8 GB card, loop layers kept: layers 0–5, 16–23 on the CPU | 5.7 GB | 17.7 / 19.6 |

Keeping the loop layers on the GPU is the biggest single gain in llama.cpp: +27–34% at 12 GB and +50–100% at 8 GB.
`--cpu-moe` with the cache could not be measured here. It needs all 15.8 GB of experts in RAM, and on this 15 GB
machine the page cache thrashed to the SSD (about 3 tok/s), so that run measured the SSD, not the cache.

**Gate weights** are spread fairly evenly over the 8 chosen experts: rank 1 carries 22% and rank 8 carries 7%. Skipping
low-ranked experts would drop real weight, and accuracy-for-speed trades are not accepted, so the engine computes all 8.

**Limits.** The trace covers one quant, one machine and 12 requests at temperature 1.0. Agent-style prompts with no
tools defined ended early, after 89–192 tokens, so most decode tokens come from long single answers. A trace of real
tool-calling agent sessions would be the next check.
