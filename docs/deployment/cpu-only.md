# Deployment: CPU-only

Build `llama-server` without CUDA flags:

```bash
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
mkdir build && cd build

cmake ..
cmake --build . --target llama-server -j$(nproc)
```
