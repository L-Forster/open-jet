# Runtime: TensorRT-LLM

`open-jet` can run against `trtllm-serve` instead of `llama-server`.

## Setup

1. Install TensorRT-LLM so `trtllm-serve` is available on `PATH`.
2. Set runtime config to TensorRT-LLM.

## Example configuration

```yaml
runtime: trtllm_pytorch
model: Qwen/Qwen2.5-7B-Instruct
trtllm_backend: pytorch
trtllm_trust_remote_code: true
# optional: pass a trtllm-serve YAML file
# trtllm_config_path: /home/you/qwen-fast.yml
context_window_tokens: 4096
gpu_layers: 0
```

When `runtime` is `trtllm_pytorch`, `open-jet` launches:

```bash
trtllm-serve <model> --backend pytorch --host 127.0.0.1 --port 8080
```

and then connects through the same OpenAI-compatible chat API path.
