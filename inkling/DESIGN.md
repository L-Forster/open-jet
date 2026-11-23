# OpenJet's engine for Inkling 2 Mini on small GPUs

Goal: run Inkling 2 Mini (26.4B parameters, 128 experts, 8 per token, layers 6–17 run three times) at usable speed on
GPUs that cannot hold it (6–16 GB), with every expert computed exactly. The engine streams the experts that don't fit
from RAM, and from the SSD when RAM is short too, the way Strata does for Qwen3.8-Flash-Next. It uses Inkling's loop
structure to read less.

Measurements behind this design: `routing/README.md` (expert routing over 10.7K decode tokens) and
`routing/results/*-offload/` (llama.cpp with experts on the CPU).

## What has to be read per token

From the UD-Q4_K_XL GGUF, which keeps the model's own NVFP4 experts and puts everything else in Q8_0:

| Part | Stored | Read per decode token |
|---|---|---|
| Experts, layers 0–19 (NVFP4, 4.48 MB each) | 11.5 GB | 8 per layer application |
| Experts, layers 20–23 (Q8_0, 8.46 MB each) | 4.3 GB | 8 per layer application |
| Attention, convolutions, routers, depth mixing (Q8_0/F32/BF16) | 0.72 GB | 1.7 GB: the core layers' weights are read on each of the three passes |
| Loop controllers (passes 2 and 3) | 0.11 GB | 0.11 GB |
| Output head (Q8_0) | 0.49 GB | 0.49 GB |
| Token embedding (Q8_0) | 0.49 GB | one row (stays in RAM) |

A token reads 1.85 GB of experts, counting each pass, and about 2.3 GB of everything else, which stays on the GPU. At
75% of an RTX 3090's bandwidth that is a ceiling of about 170–200 tok/s with everything in VRAM.

## Measured facts the design relies on

1. **Uneven routing.** The most-read 12.7% of expert bytes serve 50% of reads, and 36% serve 80%.
2. **Pass-to-pass reuse.** Passes 2 and 3 of a looped layer reuse 54–60% of the previous pass's experts. So once pass
   1 of layer *l* has routed, most of what passes 2 and 3 of *l* will need is known 12 layer applications early.
3. **Token-to-token reuse.** 48% of a token's experts were used by the previous token, and 88% within 16 tokens.
4. **An adaptive cache beats a fixed profile.** With 20% of the pool in VRAM, a profile alone serves 59% of reads and
   profile plus conversation-driven swaps serve 73%. The Belady bound is 87%.
5. **llama.cpp today** (RTX 3090, Ryzen 7 5800X, DDR4): 42–49 tok/s with everything on the GPU. With 12 GB of the card
   used, 18–20 tok/s with `--n-cpu-moe` and 24–25 with layers 6–17 pinned on the GPU. With 8 GB, 10–12 versus 18–20.

## Memory tiers

- **VRAM**: all non-expert weights (about 1.3 GB), the output head, the KV cache (8 full-attention passes × 2 KB per
  token at 8-bit, plus 1024-token windows elsewhere), the convolution states, and an expert cache filling the rest. The
  cache is sized in bytes, because layers 20–23 have experts twice the size of the others.
- **RAM**: an expert arena in pinned memory, holding every expert when RAM allows, otherwise the hottest experts up to a
  budget.
- **SSD**: the GGUF itself. Missing experts are read with `pread`/`O_DIRECT` into the arena; no mmap, so reads aren't
  page-sized and the OS cache can't evict the arena.

## One layer application

1. The GPU runs depth mixing, loop control, attention and the router, then writes the 8 expert ids and weights to
   mapped host memory.
2. A CPU thread spinning on that memory splits the experts. Cached ones go to the GPU. Missing ones are either computed
   by CPU threads in place in RAM, or copied to the GPU by its copy engine, following the measured split that keeps
   both sides busy (Strata found 20–55% by copy, depending on how much RAM bandwidth the CPU kernels use).
3. The GPU computes its experts while the CPU computes its own. The CPU writes results back to mapped memory, the GPU
   sums them, and the next layer application starts.

Decode runs as CUDA graphs with no driver synchronisation per layer. Prompt processing copies each layer's missing
experts to the GPU and computes there, because large batches use nearly every expert.

## Using the loop

- **Prefetch for later passes.** When pass 1 of a core layer routes, its experts are needed again by passes 2–3 with
  54–60% probability, 12 layer applications later. They stay pinned in VRAM until pass 3 of that layer finishes. When
  they had to come from the SSD, the read starts then rather than when pass 2 asks for them.
- **Weighting.** A core-layer expert is read up to three times per token. The cache scores experts by reads, not
  tokens, so the core layers get VRAM first. This made static placement in llama.cpp 27–100% faster on its own.

## Cache policy

The VRAM cache starts from a profile (rank by reads over a mixed workload) and adapts to the conversation:
- decayed read counts per expert;
- every few tokens, swaps a bounded number of experts whose decayed counts clearly exceed the coldest cached ones;
- each swap's copy is issued at once, and the expert becomes usable when the copy lands (Strata's finding 10).

`routing/analyze.py` replays traces against candidate policies. Any change to the policy should be replayed against
the trace before it is built.

## Kernels

- **NVFP4 matrix-vector on sm_86 and older.** These GPUs have no FP4 tensor cores, so the kernel decodes 16-element
  sub-blocks (UE4M3 scale × E2M1 lookup) in registers, multiplies by the per-expert `.scale`, and accumulates in fp32.
  It is bandwidth-bound.
- **Q8_0 matrix-vector** for the dense weights and the Q8_0 experts.
- **Attention with the relative bias fused.** Compute `r · rel_bias` once per head and query, then gather by
  distance inside the attention kernel; llama.cpp materialises it as a large F32 tensor. Sinks fold into the softmax
  denominator.
- **CPU expert kernels** (AVX2, plus AVX-512 where present): NVFP4 decoded through a 16-entry lookup table and Q8_0,
  for several tokens at once when a verify window holds them.

## Exactness

The engine computes all 8 experts with the model's weights. It never skips low-weight experts or substitutes a cached
expert for a missing one; the gate weights are spread too evenly for that (rank 8 carries 7%). `reference/` holds an
fp32 reference forward pass checked against llama.cpp. Every kernel change reports top-token agreement and KL
divergence against it, as the repo's rules require for engine changes.

## Status (2026-11-21)

Steps 1–5 and 7 below are built, plus batched prefill and split-K decode attention. Step 5 is the cache with
asynchronous swaps, the CPU-cost weighting, and the opt-in zero-copy GPU share. Step 6 (multi-token prediction) is
not built. Measurements:
[`engine/README.md`](engine/README.md).

## Build order

1. Reference forward pass, validated against llama.cpp (`reference/`).
2. NVFP4 and Q8_0 GPU kernels with unit tests against the reference dequantisation.
3. Everything in VRAM (16–24 GB cards): the full GPU decode path against the reference. Speed against llama.cpp's
   42–49 tok/s on the 3090.
4. Expert cache with CPU computation of misses (8–12 GB cards), against llama.cpp's 18–25 tok/s at the same VRAM.
5. SSD tier with loop prefetch (cards and machines too small for the model).
6. Multi-token prediction (the model ships a 3-layer MTP head), with exact verification.
7. OpenJet integration: a `llama_backend` engine choice picked by `src/llama_server.py` for Inkling GGUFs, served
   through the same OpenAI/Anthropic-compatible server as the current engine.
