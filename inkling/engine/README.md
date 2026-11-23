# OpenJet's Inkling 2 Mini engine

A CUDA engine for Inkling 2 Mini built around its looped MoE structure. It runs everything from VRAM on 24 GB cards.
On smaller cards it keeps part of the experts in a VRAM cache and computes the rest on the CPU, straight from the
mmap'd GGUF. Design: [`../DESIGN.md`](../DESIGN.md). Routing measurements behind it: [`../routing/`](../routing/).

Status: decode works in both modes and matches the fp32 reference for UD-Q4_K_XL and UD-IQ4_XS, and supports all tensor
types in UD-IQ3_XXS (`IQ2_XXS`, `IQ2_XS`, `IQ3_XXS`, `IQ2_S`, `NVFP4`, `Q4_K`/`Q5_K`/`Q6_K`/`Q8_0` embeddings). OpenJet starts it
for Inkling GGUFs (`src/llama_server.py`), and `../server/ik_server.py` serves it through OpenJet's engine server,
with reasoning and tool calls. Prompts are processed in batches (`ik_prefill`), and decode attention is split across
blocks so long contexts keep their speed. There is no SSD-aware read-ahead.

## Files

| File | What it is |
|---|---|
| `ik_engine.cu` | The engine: loading, cache slots, the decode step as one CUDA graph, offload coordination, C API (`ik_new`, `ik_new2`, `ik_decode`, …) |
| `ik_prefill.cu/.cuh` | Batched prefill kernels: attention, convolutions, routing, grouping, and fp32 fallback matvecs |
| `ik_prefill_tc.cu` | Tensor-core (`m16n8k16`) prefill GEMMs reusing `megakernel/prefill_gemm.cuh` for GGUF Q4_K/Q5_K, plus linear Q8_0 dense and grouped NVFP4/Q8_0 MoE expert kernels |
| `ik_dspark.cu/.cuh` | Native CUDA DSpark MTP drafter (`mtp.safetensors`): load-time `Q8_0` 3-layer block transformer + rank-256 recurrent Markov head with folded `lm_head`, CUDA-graphed `draft` and `store_context` |
| `ik_kernels.cu/.cuh` | Per-token kernels: embedding, depth mixing, loop control, attention preparation with `q8_0` KV quantization (same split int8 + fp16-per-32 format as `megakernel/qwen38_mk.cu`), split-K attention with relative bias and sinks fused, router, convolutions, CPU handoff |
| `expert_kernels.cu` | NVFP4 and Q8_0 expert and dense matrix-vector kernels, and the repack into their layout (the fast paths) |
| `gguf_kernels.cu` | Coalesced warp-level GPU kernels for Q4_K, Q5_K, Q6_K, IQ4_XS, IQ4_NL, IQ3_XXS, IQ2_S, IQ2_XS, IQ2_XXS, Q5_0, Q8_0 and NVFP4 in their GGUF blocks |
| `cpu_experts.cpp` | CPU expert kernels reading GGUF blocks directly: AVX2 + FMA for NVFP4, Q8_0, Q5_0, IQ4_NL, IQ4_XS, IQ3_XXS, IQ2_S, IQ2_XS, IQ2_XXS, Q4_K and Q5_K, scalar fallback for other CPUs/types |
| `iq_tables.h` | Canonical ggml IQ2/IQ3 lookup tables (`k_ksigns_iq2xs`, `k_iq2xxs_grid`, `k_iq2xs_grid`, `k_iq2s_grid`, `k_iq3xxs_grid`) shared by `gguf_kernels.cu` and `cpu_experts.cpp` |
| `capi.cpp` | The `mk_*` C API of `megakernel/server` on top of the engine: tokenizer and chat template (libllama, vocabulary only), sampling, multi-slot checkpoint pool (`mk_prefill_cks`, `mk_checkpoints`, `mk_rewind_to`, `mk_ck_pin`, `mk_ck_pinned`) |
| `gguf.h` | GGUF parser (copy of `megakernel/gguf.h` plus BF16, NVFP4 and longer arrays) |
| `engine_v0.py` | PyTorch prototype of the decode algorithm (slow; it fixed the algorithm before the C++ engine) |
| `profiles/default.f32` | Expert read profile from `../routing/make_profile.py`, for the initial cache fill |
| `tests/` | Kernel tests against PyTorch (`test_*.py`), and engine checks against the reference (`check_engine.py`) |

## Build and test

```bash
make -C inkling/engine                       # libinkling.so plus the test libraries (needs ~6 GB RAM, -j1)
python -m pytest inkling/engine/tests -q     # kernel tests (CUDA GPU; the GGUF for the expert tests)
python inkling/engine/tests/check_engine.py ref      # reference logits for the validation prompts
python inkling/engine/tests/check_engine.py engine   # everything in VRAM
IK_VRAM_MIB=9000 python inkling/engine/tests/check_engine.py engine   # offload, 9000 MiB of experts in VRAM
```

Environment knobs for offload: `IK_SWAP_EVERY` (tokens between cache swaps, default 8), `IK_SWAP_MAX` (swaps per
round, default 4, measured best), `IK_SWAP_SYNC=1` (synchronous swaps), `IK_PREFAULT=0` (don't fault in the
CPU-side expert pages at load), `IK_THREADS`, `IK_ZC_MIB` / `IK_ZC_SHARE` (the GPU's share of the misses, below).
`IK_GGUF` picks the model for `check_engine.py`.

## Results (RTX 3090, Ryzen 7 5800X, DDR4, WSL with 15 GB RAM, 2026-11-21)

**Accuracy.** Decoding the three validation prompts (116 positions) token by token against the fp32 reference forward
pass gives top-1 agreement of 100%, max |logit difference| of 4–5e-5, and mean KL of 4e-12 to 2e-11. That holds for
both files, with everything in VRAM and in offload mode, where the CPU computes part of the experts. The kernels for
every ggml type match gguf-py's dequantisation to within 5e-7 relative error, on GPU and CPU (`tests/`).

**Decode speed**, 256 greedy tokens from a short prompt, second run after the page cache has warmed. Offload uses
asynchronous cache swaps (4 every 8 tokens). The GPU was idle; runs while it was in other use measured up to 60%
lower.

| Setup | UD-Q4_K_XL (17.6 GB) | UD-IQ4_XS (15.7 GB) | llama.cpp `19bc41d`, UD-Q4_K_XL, best placement measured |
|---|---|---|---|
| Everything in VRAM | 76–88 tok/s | 67–72 tok/s | 41.6–48.9 tok/s |
| 12 GB card budget (9000 MiB expert cache) | 65.6 tok/s (VRAM hit 95.1%) | 59.9 tok/s (94.7%) | 23.0–24.7 tok/s |
| 8 GB card budget (5000 MiB expert cache) | 23.4 tok/s (79.9%) | 34.5 tok/s (77.0%) | 17.7–19.6 tok/s |

The llama.cpp numbers come from 1,024- and 1,536-token answers to two prompts (`../routing/README.md`). There the
12 GB and 8 GB budgets are simulated by expert placement; here they are the engine's cache size. The first run
after loading is slower while the page cache warms: 43 and 48 tok/s at 12 GB, 12.5 and 18.3 at 8 GB.

- IQ4_XS is the better file on small cards. Layers 20–23's experts are about half the size of Q4_K_XL's Q8_0 ones,
  so more of the model fits in the cache. Its full-VRAM decode is slower because the Q6_K output head and the
  K-quant matrices use the generic kernels.
- Asynchronous swaps were 61.3 tok/s against 57.5 synchronous at the same rate. Faster swapping (8 every 4) was
  slower, at 42 tok/s: the CPU then faults on just-evicted experts.
- The cache weighs each expert's reads by its CPU cost: 8x for the types on the scalar CPU path. Without that,
  IQ4_XS at 12 GB left layers 20–23 on the CPU and decoded at 13.8 tok/s instead of 59.9.
- Prefill (batches of 256 tokens, `tests/check_long.py`: 1,300 tokens matching the reference exactly). It runs at
  178–188 tok/s with every expert in VRAM, independent of context length, against ~85 token by token. At the 12 GB
  budget it is 64 tok/s, with the missing experts streamed in per batch. Decode at 9K context is 81.3 tok/s (88.1 at
  512) since attention was split across 16 blocks per head; it was 37 tok/s with one block per head.
- Conversation checkpoint (`tests/check_checkpoint.py`). The state after a prompt (convolution histories and
  sliding-window rings, ~425 MiB) is saved, and the server rewinds to it when a follow-up re-renders the reply
  differently. Restored and continued, it matches a fresh run bit for bit with every expert in VRAM; in offload mode
  it matches within 8e-6, since cache swaps move some experts between CPU and GPU summation. Through the server, the
  second turn of a 5.7K-token conversation whose client dropped the reasoning took 1.3 s (5,741 tokens resumed, 72
  prefilled) instead of 36 s.
- SSD tier (`IK_RAM_MIB`, set by OpenJet when RAM can't hold the CPU-side experts). Whole experts are read into a RAM
  cache with block-aligned `pread` (O_DIRECT), cut into 512 KiB pieces spread over the CPU threads, and least recently
  used slots are evicted. Tested on a simulated small machine: 8 GB card budget (10.8 GB of experts on the CPU side)
  with the process capped at 6 GB of RAM, page cache included. It reached 6.3 tok/s (31 expert reads per token at
  1.28 GB/s) against 5.4 tok/s for the mmap path, both matching the reference. WSL's virtual disk peaks at 1.7 GB/s
  with 8 parallel readers (0.7–0.9 GB/s for one), so a native NVMe drive should do better; not measured.
- Copy path (`IK_COPY=1`, off by default). Each layer's missing experts are split between the CPU and a copy thread
  that sends the rest to VRAM scratch slots, where the GPU computes them; the split follows the two sides' measured
  costs, and the copy thread raises a flag in mapped host memory once the copies land (under WSL a kernel on another
  stream does not start while the decode graph waits on the host, so swap repacks then run on the decode stream).
  Exact against the reference. It loses on this machine, at the 8 GB budget (UD-Q4_K_XL, one prompt):
  20.7 tok/s without, 11.1 with copies from an 832 MiB pinned arena (5 copies per token at 0.4 ms), and 8.3 with
  copies also staged from the page cache. Why: WSL refuses to pin the mapped file (`cudaHostRegister`: operation not
  supported) and caps pinned memory at about 1 GiB. Staging copies ran at 3.7 GB/s next to the CPU kernels (1.2 ms
  per expert, slower than computing it), and the arena duplicates pages the page cache also holds, which pushed
  other experts out (CPU time per expert 469 -> 1072 us). Pinned copies run at 25.7 GB/s here (0.18 ms per NVFP4
  expert), so a system that can pin the file mapping (native Linux) could copy without either cost; not measured.
- A bug the prefill work exposed, fixed in both paths: depth mixing summed over all 8 shared-memory slots, multiplying
  the unused ones by zero, but those slots are uninitialised, and 0 × NaN = NaN. It showed as intermittent NaN floods
  in long prefills, depending on what ran before on the SM.
- At 8 GB this machine is the limit: 10–11 GB of experts live outside VRAM, in the page cache of a 15 GB machine.
- GPU share of the misses (`IK_ZC_MIB`, off by default). The hottest CPU-side experts are copied into pinned, mapped
  host memory. The GPU reads `IK_ZC_SHARE` of each layer's missing experts from there over PCIe, while the CPU
  computes the rest. Both apply the same rule, so there is no extra round trip. It matches the reference exactly. Here
  it was slower: 58.5 against 65.3 tok/s at 12 GB with share 0.5. WSL lets a process pin only about 1 GiB, which
  holds 112 experts, so the GPU took about one miss per token. That saving is smaller than the four extra kernels per
  layer application cost. Pinning the mmap'd file directly (`cudaHostRegister`) is refused on WSL. It needs
  measuring on native Linux with more pinned memory.

## Updates (2026-11-22: `megakernel/` component reuse)

- **`q8_0` KV cache (default; `IK_KV=fp32` to revert).** Ported `megakernel/qwen38_mk.cu`'s split `int8_t` + `__half` per-32
  scale KV format (`1.0625 B/elem` vs `4 B/elem`). Shrinks the 40 sliding-window rings from `400 -> 106 MiB` and the 8
  full-attention caches from `64 -> 17 KiB/tok`. Combined with moving checkpoints off-GPU, total VRAM at 8K context dropped
  from `18,583 -> 17,438 MiB` (**~1.1 GiB freed** for the expert cache on 8–12 GB cards). On an 8 GB card budget, giving
  that extra 1 GiB to the expert cache (`5000 -> 6000 MiB`) raised VRAM hit rate from `79.3% -> 85.4%` (`UD-IQ4_XS`,
  `28.3 -> 32.3 tok/s`) and `78.2% -> 84.0%` (`UD-Q4_K_XL`, `20.0 -> 22.9 tok/s`). Accuracy against the fp32 reference:
  `100%` top-1 agreement across all 3 validation prompts and 1,300-token `check_long.py` (`kl_mean` `0.00033–0.00105`).
- **Host-RAM multi-slot checkpoint pool (`IK_CK_SLOTS`, default 4, 2 with SSD tier).** Modeled on `megakernel/ck_pool.cu`
  (coverage-based eviction + `mk_ck_pin` / `mk_ck_pinned` + `mk_prefill_cks` / `mk_checkpoints` / `mk_rewind_to`). Each
  checkpoint is `~110 MiB` in lazily-allocated pageable RAM (0 MiB VRAM, 0 MiB pinned memory). Bit-identical restore
  verified against fresh runs (`0.00e+00` logit diff in full VRAM and offload without swaps).
- **Tensor-core prefill (`ik_prefill_tc.cu`, `IK_PF_TC=0` to revert).** Directly includes `megakernel/prefill_gemm.cuh` for
  GGUF `Q4_K`/`Q5_K` dense projections and uses the same `cp.async` + `hperm` + `ldsm_x4` + `m16n8k16` pipeline (with fp32
  MMA accumulation) for linear repacked `Q8_0` dense projections and grouped `NVFP4`/`Q8_0` MoE experts. On `check_long.py`
  (1,300 tokens, `UD-Q4_K_XL`): full-VRAM prefill jumped **`183.9 -> 566.1 tok/s` (3.08×)**; 12 GB offload prefill (with
  2-way double-buffered async pinned streaming overlapping CPU copies + H2D DMA with GPU repack + TC experts) jumped
  **`64 -> 122.2 tok/s` (1.91×)**, both with `100%` top-1 agreement and `kl_mean = 0.00071–0.00082`.
- **GPU decode fusion & register-cached K-quants (`ik_kernels.cu`, `expert_kernels.cu`, `gguf_kernels.cu`).** Fused
  `mix + rmsnorm` (`ik::mix_norm`, saving 48 launches/token), fused up to 5 shared-input repacked `Q8_0` projections into
  one launch (`ik_matvec_q8_0_multi`, saving ~136 launches/token), and register-cached `x` for `Q6_K`/`Q4_K`/`Q5_K`
  (`matvec_k_reg_kernel`). Full-VRAM `UD-IQ4_XS` decode jumped **`69.4 -> 82.0 tok/s` (+18%)**, matching `UD-Q4_K_XL`.
- **Fast AVX2 CPU kernels (`cpu_experts.cpp`).** Every `UD-IQ4_XS` expert format (`Q5_0`, `IQ4_NL`, `IQ4_XS`, `Q4_K`,
  `Q5_K`) runs AVX2 on CPU (`rel err 5e-7`), and `NVFP4` now uses a 256-bit `vpshufb` nibble LUT + branchless 128-entry
  `k_ue4m3` scale LUT (**`0.86 -> 0.58 ms`** single-thread, **+48%** faster per expert). On an 8 GB budget
  (`IK_VRAM_MIB=6000`), CPU compute dropped from `310 -> 158–165 µs/expert`, pushing 8 GB decode from
  **`22.9 -> 35.7 tok/s`** (`UD-Q4_K_XL`) and **`32.3 -> 39.4 tok/s`** (`UD-IQ4_XS`) — ~2× faster than llama.cpp
  (`17.7–19.6 tok/s`).
- **Exact multi-token verification & prompt-lookup drafting (`ik_verify` / `ik_verify_commit`, `capi.cpp`).** Exact
  conv-history rollback snapshots in `pf_keep_` (0 extra VRAM, `5.7e-6–9.5e-6` max logit diff vs `ik_decode` in fp32 KV)
  with batched CPU expert kernels (`ik_cpu_gate_up_batch` / `ik_cpu_down_batch`) and exact greedy/rejection verification in
  `mk_step` + `mk_stats`. Default `IK_LOOKUP` is on in full VRAM (4-gram match + adaptive cooldown: **+10–18%** on
  code-editing prompts with 100% exact token match) and off in offload (where 128-expert MoE multi-token batches multiply
  unique CPU expert reads); override with `IK_LOOKUP=0|1` and `IK_DRAFT=1..7`.
- **Small-GPU offload & pipelined prefill (`cpu_experts.cpp`, `ik_engine.cu`).** Exact fp32 2-row fused AVX2 `NVFP4`
  (`dot_nvfp4_2rows`, MSVC AVX2 guard fixed), L1-local `x`/`y` staging in `coordinate()`, Zero-Logits-D2H greedy and
  filtered sampling, and pipelined offload prefill (`pf_ids_ev_`, persistent double-buffered `pf_gpu_busy_`). On RTX 3090
  (`UD-IQ4_XS`, `q8_0` KV): 8 GB decode reached **47.8** greedy / **51.1 tok/s** `temp=0.7` (**157.0 tok/s** prefill),
  and 12 GB decode reached **65.0** greedy / **67.1 tok/s** `temp=0.7` (**225.3 tok/s** prefill, +84% vs 122.2 before),
  with `100.0%` top-1 agreement (`kl_mean 0.00014–0.00114`).
- **Native CUDA DSpark MTP speculative decoding (`ik_dspark.cu/.cuh`, `ik_prefill.cu`, `gguf_kernels.cu`, `capi.cpp`).**
  Loads `thinkingmachines/Inkling-2-Mini`'s `mtp.safetensors` (607 MB; auto-discovered next to the GGUF in full-VRAM mode
  or via `--mtp` / `IK_MTP`), quantizing its 3 dense layers and folded `[n_vocab, 256]` Markov head to `Q8_0` at load time
  (~480 MB VRAM, identical acceptance to BF16). Pairs CUDA-graphed `draft` / `store_context` with coalesced
  `dot_row_multi`, 128-thread parallel `group_kernel` + compacted active-expert dispatch, register-cached `cnt==1` /
  fused-`X` `cnt<=4` expert paths, fused `matvec_q8r_multi`, single-kernel conv-history rollback, pre-filtered stochastic
  `build_dist`, and adaptive draft depth (`{2, 4, 7}`, override with `IK_DRAFT=1..7` / `IK_ADAPT=0`). On RTX 3090 full
  VRAM (`UD-IQ4_XS`, under light ~34% host load where no-MTP baseline was `75.5–75.8 tok/s`): adaptive DSpark hit
  **95.8 tok/s** on code (`4.05 tok/cyc`, 100% greedy match), **139.7 tok/s** on structured code continuation
  (`6.64 tok/cyc`, 100% match), **75.0 tok/s** on prose, **80.9 tok/s** at `temp=0.7, top_p=0.95` (exact rejection
  sampling), and `exact=True` checkpoint rewind. Stays off by default in offload mode (`IK_VRAM_MIB >= 0`), where
  12 GB offload measured slower with MTP (`20.0 -> 18.1` code, `27.2 -> 13.2` prose) due to unique CPU expert misses.

## Next

1. Measure native Linux (higher pinned-memory ceiling for copy path, native NVMe for SSD tier) and real 8–12 GB cards.
2. Router-predicted SSD read-ahead across core passes 1 -> 2–3.
3. Tensor-core `mma.sync` verification kernels for `T = 5..8` to further reduce full-VRAM MoE verify latency.
