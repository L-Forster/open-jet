# Inkling 2 Mini reference forward pass

`inkling_ref.py` is an fp32 PyTorch forward pass of Inkling 2 Mini, read straight from a mainline-layout
(`inkling2_loop`) GGUF. It follows llama.cpp's `src/models/inkling2.cpp` and `inkling2-loop.cpp` (base `19bc41d`) op
for op:
- depth mixing over the residual blocks;
- loop controllers on passes 2 and 3;
- short causal convolutions, with input-dependent taps on the full-attention layers;
- Q/K L2 clipping;
- the relative-position bias with per-query sinks;
- the sliding window;
- sigmoid routing with selection bias, 8 of 128 experts, SITU activation, and per-expert NVFP4 scales;
- the final softcap.

It processes a whole sequence per layer, with no KV cache, and dequantises only the experts each layer selects. The
engine's kernels are checked against it.

## Check against llama.cpp

`validate.py` runs llama.cpp (`19bc41d` plus `../routing/llama-route-trace.patch`) greedily on three prompts (code,
English prose, Chinese) for 24 tokens each. It keeps the top-10 next-token probabilities and, through the routing
trace, the experts chosen for every token and layer application. The reference then runs on the same tokens.

Results for 2026-11-21, UD-Q4_K_XL mainline file (`results/2026-11-21-validate.json`):

| Prompt | Top-1 agreement | Mean / max abs. difference, top-5 probabilities | Experts matching llama.cpp | Token-layers with all 8 matching |
|---|---|---|---|---|
| Python function | 24/24 | 0.09% / 1.7% | 98.3% | 86.6% |
| English prose | 24/24 | 0.22% / 3.8% | 98.6% | 89.3% |
| Chinese | 24/24 | 0.18% / 2.3% | 98.5% | 88.0% |

The remaining expert differences are near-ties in routing. llama.cpp rounds activations to 8-bit for its Q8_0 matrix
products and accumulates partly in fp16; the reference stays in fp32 throughout.
