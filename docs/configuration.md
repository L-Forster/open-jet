# Configuration

Main settings live in `config.yaml`.

## Minimal local config

```yaml
llama_model: /home/you/models/Qwen3.5-4B-Q4_K_M.gguf
model_source: local
llama_server_path: /home/you/llama.cpp/build/bin/llama-server
context_window_tokens: 4096
device: cuda
gpu_layers: 99
```

`llama_server_path` is optional. Setup will populate it automatically when it provisions or discovers `llama-server`.

If setup starts from a direct GGUF download, it will download into the OpenJet models directory and then persist the resolved `llama_model` path for future runs.

You can override the setup wizard's direct-download recommendations in `config.yaml`:

```yaml
setup_recommendations:
  direct_models:
    - max_ram_gb: 6
      label: Qwen3.5 4B
      filename: Qwen3.5-4B-Q4_K_M.gguf
      url: https://huggingface.co/unsloth/Qwen3.5-4B-GGUF/resolve/main/Qwen3.5-4B-Q4_K_M.gguf?download=true
    - max_ram_gb: 12
      label: Qwen3.5 9B
      filename: Qwen3.5-9B-Q4_K_M.gguf
      url: https://huggingface.co/unsloth/Qwen3.5-9B-GGUF/resolve/main/Qwen3.5-9B-Q4_K_M.gguf?download=true
    - max_ram_gb: 24
      label: Qwen3.5 27B
      filename: Qwen3.5-27B-Q4_K_M.gguf
      url: https://huggingface.co/unsloth/Qwen3.5-27B-GGUF/resolve/main/Qwen3.5-27B-Q4_K_M.gguf?download=true
```

Rows are matched by `max_ram_gb`, and the last row is used as the fallback above the highest configured RAM band.

## General settings

Common local settings:

```yaml
airgapped: false
context_window_tokens: 4096
system_prompt: |
  You are concise and tool-aware.
memory_guard:
  context_reserved_tokens: 768
  min_prompt_tokens: 256
  condense_target_tokens: 900
  keep_last_messages: 6
```

## Model profiles

Setup stores reusable model presets under `model_profiles`. The active preset is tracked by `active_model_profile`.

This is the recommended way to switch between local GGUF presets with different paths, context windows, or GPU offload settings.

## Related docs

- [Quickstart](quickstart.md)
- [Runtime: llama.cpp](runtimes/llama-cpp.md)
- [Python SDK](sdk/python-sdk.md)
