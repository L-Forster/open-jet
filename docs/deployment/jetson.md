# Deployment: Jetson

For Jetson (Orin Nano, Orin NX, AGX Orin), build `llama-server` from source with Jetson-appropriate CUDA flags.

```bash
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
mkdir build && cd build

cmake .. \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=87 \
  -DGGML_CUDA_FA_ALL_QUANTS=ON

cmake --build . --target llama-server -j$(nproc)
```

## Key flags

| Flag | Why |
|------|-----|
| `GGML_CUDA=ON` | Enable CUDA backend |
| `CMAKE_CUDA_ARCHITECTURES=87` | Target SM 8.7 (Orin). Use `72` for Xavier. |
| `GGML_CUDA_FA_ALL_QUANTS=ON` | Enable flash attention for all KV cache quantizations (q8_0, q4_0), not only f16. |

The resulting binary is typically at `build/bin/llama-server`.
