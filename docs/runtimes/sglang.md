# Runtime: SGLang

`open-jet` can connect to SGLang through its OpenAI-compatible server.

On Jetson, prefer running SGLang in a local container to keep inference local while avoiding host-side Python dependency issues.

## Example configuration

```yaml
runtime: sglang
model: /home/you/models/Qwen3.5-4B-AWQ-4bit
sglang_model: /home/you/models/Qwen3.5-4B-AWQ-4bit
sglang_launch_mode: docker
sglang_base_url: http://127.0.0.1:30000
sglang_docker_image: your-local-sglang-image
sglang_docker_container_name: open-jet-sglang
sglang_docker_runtime: nvidia
sglang_served_model_name: local
sglang_reasoning_parser: qwen3
sglang_tool_call_parser: qwen3_coder
sglang_mem_fraction_static: 0.8
context_window_tokens: 8192
gpu_layers: 0
```

## Launch modes

- `docker`: `open-jet` starts the local SGLang container and waits for `127.0.0.1:30000`.
- `external`: `open-jet` connects to an already-running local SGLang server and does not import/launch SGLang from the host Python environment.
- `managed`: use only when SGLang is installed in the same Python environment as `open-jet`.
