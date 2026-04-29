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
    - max_ram_gb: 12
      label: Qwen3.6 27B UD-IQ2_XXS
      filename: Qwen3.6-27B-UD-IQ2_XXS.gguf
      url: https://huggingface.co/unsloth/Qwen3.6-27B-GGUF/resolve/main/Qwen3.6-27B-UD-IQ2_XXS.gguf?download=true
      model_size_mb: 9626
      kv_bytes_per_token: 34816
    - max_ram_gb: 16
      label: Qwen3.6 27B UD-IQ3_XXS
      filename: Qwen3.6-27B-UD-IQ3_XXS.gguf
      url: https://huggingface.co/unsloth/Qwen3.6-27B-GGUF/resolve/main/Qwen3.6-27B-UD-IQ3_XXS.gguf?download=true
      model_size_mb: 12288
      kv_bytes_per_token: 34816
    - max_ram_gb: 20
      label: Qwen3.6 27B Q4_K_M
      filename: Qwen3.6-27B-Q4_K_M.gguf
      url: https://huggingface.co/lmstudio-community/Qwen3.6-27B-GGUF/resolve/main/Qwen3.6-27B-Q4_K_M.gguf?download=true
      model_size_mb: 16896
      kv_bytes_per_token: 34816
    - max_ram_gb: 24
      label: Qwen3.6 35B A3B UD-Q3_K_XL
      filename: Qwen3.6-35B-A3B-UD-Q3_K_XL.gguf
      url: https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF/resolve/main/Qwen3.6-35B-A3B-UD-Q3_K_XL.gguf?download=true
      model_size_mb: 17203
      active_model_size_mb: 3072
      kv_bytes_per_token: 24576
      unified_memory_only: true
      llama_cpu_moe: true
      llama_n_cpu_moe: 0
    - max_ram_gb: 32
      label: Qwen3.6 35B A3B
      filename: Qwen3.6-35B-A3B-UD-Q4_K_M.gguf
      url: https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF/resolve/main/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf?download=true
      model_size_mb: 22630
      active_model_size_mb: 3072
      kv_bytes_per_token: 24576
      unified_memory_only: true
      llama_cpu_moe: true
      llama_n_cpu_moe: 0
```

Rows are matched by `max_ram_gb`, and the last row is used as the fallback above the highest configured RAM band.
For unified-memory MoE rows, setup keeps a 4GB system reserve before applying the normal model/KV headroom, so Q4_K_M remains preferred when it fits and UD-Q3_K_XL is the smaller fallback.

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

## Shell Targets

You can expose named remote shell targets for the `shell` tool. OpenJet stays on the local machine, and when the model sets `target` on a shell call, OpenJet writes a temporary script locally, copies it to the target with `scp`, runs it with `ssh`, and returns stdout/stderr.

Example:

```yaml
shell_targets:
  jetson:
    description: Jetson for running scripts and commands.
    ssh_command: ssh -p 2222 louis@localhost
    scp_command: scp -P 2222
    scp_target: louis@localhost
    remote_tmp_dir: /tmp
    control_path: ~/.openjet/state/ssh-jetson.sock
    control_persist: 10m
```

Behavior:

- omit `target` or use `local` to run on the machine hosting OpenJet
- use `target: jetson` to run on the configured Jetson target
- one persistent OpenSSH control connection is reused across commands
- file writes and edits remain local unless you separately use a mounted filesystem such as `sshfs`

## Model profiles

Setup stores reusable model presets under `model_profiles`. The active preset is tracked by `active_model_profile`.

This is the recommended way to switch between local GGUF presets with different paths, context windows, or GPU offload settings.

## Related docs

- [Quickstart](quickstart.md)
- [Runtime: llama.cpp](runtimes/llama-cpp.md)
- [Python SDK](sdk/python-sdk.md)
