# Inkling 2 Mini on OpenJet

[Inkling 2 Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini) is a 26.4B MoE model (128 experts, 8 active per token). On NVIDIA RTX 30, 40 and 50-series GPUs, OpenJet runs it on its own CUDA engine (`libinkling.so` on Linux, `inkling.dll` on Windows). When all experts fit in VRAM, everything stays on the GPU. On 6-12 GB cards, the engine keeps the hottest experts in a VRAM cache and computes cache misses on the CPU directly from RAM. Nothing is pruned or approximated.

On 16 GB AMD and Intel Vulkan laptops (such as Radeon 740M/760M/780M iGPUs), OpenJet serves `UD-IQ3_XXS` through `llama.cpp` (`b12651`) with Vulkan and `--cpu-moe`.

## Which file to use

Use a mainline-layout GGUF from the `Mainline_Support/` folder of [unsloth/Inkling-2-Mini-GGUF](https://huggingface.co/unsloth/Inkling-2-Mini-GGUF) (rename `.gguf_file` to `.gguf`), or ggml-org's files. Unsloth's root-folder files use a different layout and only run on llama.cpp.

`openjet setup` picks one automatically based on your hardware:

| Hardware | Default model | Size | Runtime |
|---|---|---:|---|
| 6 GB NVIDIA (e.g. RTX 4050) or 8 GB NVIDIA + 16 GB RAM | `UD-IQ3_XXS` (`KLD 0.0269`, 94.7% top-1) | 10.9 GiB | OpenJet Inkling engine |
| 8 GB NVIDIA + 32 GB RAM, or 10-12 GB NVIDIA (e.g. RTX 3060) | `UD-IQ4_XS` (`KLD 0.0109`, 97.7% top-1) | 14.7 GiB | OpenJet Inkling engine |
| 16 GB AMD / Intel Vulkan laptop (6-10 GB Vulkan heap) | `UD-IQ3_XXS` | 10.9 GiB | `llama.cpp` (`b12651`) Vulkan + `--cpu-moe` |
| 24 GB NVIDIA (manual choice; default is Qwen3.8-27B) | `UD-IQ4_XS` or `UD-Q4_K_XL` | 14.7-16.4 GiB | OpenJet Inkling engine |

The engine reads `NVFP4`, `Q8_0`, `Q4_K`, `Q5_K`, `Q6_K`, `IQ4_XS`, `IQ4_NL`, `IQ3_XXS`, `IQ2_S`, `IQ2_XS`, `IQ2_XXS`, and `Q5_0` tensors (including `Q4_K`, `Q5_K`, `Q6_K`, and `Q8_0` token embeddings), covering `UD-IQ3_XXS`, `UD-IQ4_XS`, and `UD-Q4_K_XL`.

## How VRAM budgeting works

With `llama_backend: auto` (the default), OpenJet reserves VRAM for non-expert state before sizing the expert cache:

- `2300 MiB` for dense weights (attention, 200K output head, loop controllers), activation buffers, and CUDA context
- `17 KiB` per context position for the `q8_0` KV cache (`~136 MiB` at 8K context)
- `512 MiB` for desktop headroom

Everything left goes to the expert cache (`IK_VRAM_MIB`). On a 6 GB card that leaves about `3.2 GB` for experts; on 8 GB about `5.2-6.0 GB`; on 12 GB about `9.0 GB`. If all experts fit, offload turns off completely.

If system RAM cannot hold the remaining CPU-side experts with `3 GB` to spare, OpenJet turns on the SSD tier (`IK_RAM_MIB`), which reads experts in parallel blocks into a fixed RAM cache instead of page-faulting through `mmap`.

Useful environment variables:

- `IK_VRAM_MIB`: override the expert cache size in MiB
- `IK_RAM_MIB`: override the SSD-tier RAM cache size in MiB
- `IK_KV=fp32`: switch KV cache from `q8_0` to fp32
- `IK_MTP=/path/to/mtp.safetensors` or `IK_NO_MTP=1`: force DSpark speculative decoding on or off
- `IK_CK_SLOTS`: host-RAM conversation checkpoint slots (default 4, or 2 with SSD tier; ~110 MiB RAM each, 0 VRAM)

## Measured speed

Measured on an RTX 3090 + Ryzen 7 5800X (DDR4, 15 GB RAM under WSL2), 256 greedy tokens from a short prompt after the page cache warmed. Smaller cards are simulated by capping `IK_VRAM_MIB`. The llama.cpp column is `19bc41d` on the same machine (`UD-Q4_K_XL`, 1,024- and 1,536-token outputs).

| Setup | `UD-IQ4_XS` (OpenJet) | `UD-Q4_K_XL` (OpenJet) | `llama.cpp` (`UD-Q4_K_XL`, best placement) |
|:---:|:---:|:---:|:---:|
| Full VRAM (no MTP) | 82.0 tok/s | 81-91 tok/s | 41.6-48.9 tok/s |
| Full VRAM + DSpark MTP (`mtp.safetensors`) | 75-140 tok/s* | - | - |
| 12 GB card (`9000 MiB` expert cache) | 65.0-67.1 tok/s | 57.1-65.6 tok/s | 23.0-24.7 tok/s |
| 8 GB card (`6000 MiB` expert cache) | 47.8-51.1 tok/s | 35.7 tok/s | 17.7-19.6 tok/s |

\* DSpark MTP measured on RTX 3090 under light host GPU load (`75.5-75.8 tok/s` baseline without MTP): adaptive draft depth `{2, 4, 7}` hit **95.8 tok/s** on code (`4.05 tok/cyc`, 100% greedy match), **139.7 tok/s** on structured code continuation (`6.64 tok/cyc`, 100% match), **75.0 tok/s** on prose, and **80.9 tok/s** at `temp=0.7, top_p=0.95` with exact rejection sampling.

Prefill (256-token batches, fp16 tensor-core GEMMs): **566 tok/s** in full VRAM, **225.3 tok/s** at 12 GB (pipelined double-buffered streaming), and **157.0 tok/s** at 8 GB. Long-context decode (`UD-Q4_K_XL`, full VRAM) runs at 70-81 tok/s at 8K-9K context.

`UD-IQ3_XXS` GPU and CPU kernels are verified bit-accurately against `libggml-base.so` (`< 1e-5` relative error; `101-111 us` per 8-expert GPU layer on RTX 3090 vs `70 us` for repacked `NVFP4`), and end-to-end live tok/s on `UD-IQ3_XXS` is not measured yet.

## Speculative decoding (DSpark MTP)

Put the official [`mtp.safetensors`](https://huggingface.co/thinkingmachines/Inkling-2-Mini) (607 MB) next to your GGUF, or pass `--mtp /path/to/mtp.safetensors`. In full-VRAM mode, the engine loads it automatically, quantizes the 3 DSpark layers and folded Markov head to `Q8_0` on GPU (~480 MB VRAM), and drafts 2, 4, or 7 tokens per cycle with exact target verification. In offload mode, DSpark stays off by default because verifying multiple tokens multiplies CPU expert reads.

## Building from source

Linux (`sm_86`, `sm_89`, `sm_120`; needs ~6 GB RAM and a llama.cpp build with `inkling2_loop` for the tokenizer):

```bash
make -C inkling/engine LLAMA=/path/to/llama.cpp ARCH=sm_86
```

Windows (MSVC 2022 + CUDA toolkit):

```powershell
.\inkling\engine\build.ps1 -Llama C:\llama.cpp -Arch sm_86
```

Details and implementation notes: [`inkling/engine/README.md`](../inkling/engine/README.md).
