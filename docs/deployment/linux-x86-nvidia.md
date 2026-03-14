# Deployment: Linux x86 + NVIDIA

Build `llama-server` from source with CUDA enabled:

```bash
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
mkdir build && cd build

cmake .. -DGGML_CUDA=ON -DGGML_CUDA_FA_ALL_QUANTS=ON
cmake --build . --target llama-server -j$(nproc)
```

Add `build/bin/` to your `PATH` or keep the binary in a known location and reference it in your runtime configuration.
