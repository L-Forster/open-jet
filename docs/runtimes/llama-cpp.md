# Runtime: llama.cpp

`open-jet` uses `llama-server` from [llama.cpp](https://github.com/ggerganov/llama.cpp) as its default inference backend.

Pre-built binaries are available for x86, while Jetson/ARM64 commonly requires source build with device-appropriate flags.

Use the deployment-specific build guides:

- [Jetson](../deployment/jetson.md)
- [Linux x86 + NVIDIA](../deployment/linux-x86-nvidia.md)
- [CPU-only](../deployment/cpu-only.md)

The `llama-server` binary can be added to `PATH` or left in `~/llama.cpp/build/bin/`.
