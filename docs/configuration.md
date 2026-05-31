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
      label: Qwen3.6 27B UD-IQ2_XXS MTP
      filename: Qwen3.6-27B-UD-IQ2_XXS-MTP.gguf
      url: https://huggingface.co/unsloth/Qwen3.6-27B-MTP-GGUF/resolve/main/Qwen3.6-27B-UD-IQ2_XXS.gguf?download=true
      llama_cpp_ref: b9189
      llama_mtp: true
      model_size_mb: 9626
      kv_bytes_per_token: 34816
    - max_ram_gb: 16
      label: Qwen3.6 27B UD-IQ3_XXS MTP
      filename: Qwen3.6-27B-UD-IQ3_XXS-MTP.gguf
      url: https://huggingface.co/unsloth/Qwen3.6-27B-MTP-GGUF/resolve/main/Qwen3.6-27B-UD-IQ3_XXS.gguf?download=true
      llama_cpp_ref: b9189
      llama_mtp: true
      model_size_mb: 12288
      kv_bytes_per_token: 34816
    - max_ram_gb: 20
      label: Qwen3.6 27B Q4_K_M MTP
      filename: Qwen3.6-27B-Q4_K_M-MTP.gguf
      url: https://huggingface.co/unsloth/Qwen3.6-27B-MTP-GGUF/resolve/main/Qwen3.6-27B-Q4_K_M.gguf?download=true
      llama_cpp_ref: b9189
      llama_mtp: true
      model_size_mb: 16896
      kv_bytes_per_token: 34816
    - max_ram_gb: 24
      label: Qwen3.6 35B A3B UD-Q3_K_XL MTP
      filename: Qwen3.6-35B-A3B-UD-Q3_K_XL-MTP.gguf
      url: https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF/resolve/main/Qwen3.6-35B-A3B-UD-Q3_K_XL.gguf?download=true
      llama_cpp_ref: b9189
      llama_mtp: true
      model_size_mb: 17203
      active_model_size_mb: 3072
      kv_bytes_per_token: 24576
      unified_memory_only: true
      llama_cpu_moe: true
      llama_n_cpu_moe: 0
    - max_ram_gb: 32
      label: Qwen3.6 35B A3B MTP
      filename: Qwen3.6-35B-A3B-UD-Q4_K_M-MTP.gguf
      url: https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF/resolve/main/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf?download=true
      llama_cpp_ref: b9189
      llama_mtp: true
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

Profiles may use the local `llama_cpp` runtime or the OpenAI Codex OAuth runtime:

```yaml
model_profiles:
  - name: local-qwen
    runtime: llama_cpp
    llama_model: /home/you/models/Qwen.gguf
    context_window_tokens: 32768
    gpu_layers: 99

  - name: codex
    runtime: openai_codex
    provider: openai-codex
    model: gpt-5.5
    context_window_tokens: 272000
    reasoning_effort: medium
    reasoning_summary: auto
    text_verbosity: medium
```

Use `/connect openai-codex` to sign in through the official Codex CLI ChatGPT OAuth flow, or `/connect openai-codex --device-auth` for Codex CLI's device-code flow on SSH/headless systems, then `/runtime cloud` or `/cloud` to switch manually. OpenJet never routes prompts to Codex automatically. Codex OAuth is not API-key auth: OpenJet reads the Codex CLI OAuth session from `$CODEX_HOME/auth.json` or `~/.codex/auth.json` and sends requests to the Codex backend. `airgapped: true` disables Codex login and Codex runtime startup while preserving local llama.cpp profiles.

API-key providers use the optional LiteLLM runtime. Install it with `pip install open-jet[cloud]`, save credentials with `/connect openai`, `/connect anthropic`, or `/connect openrouter`, then switch manually with `/cloud <name>` or `/model <name>`:

OpenJet does not write provider secrets to OpenJet-owned JSON files. API-key `/connect` uses OS keyring storage; environment variables such as `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, and `OPENROUTER_API_KEY` take precedence and are the recommended path for headless systems.

```yaml
model_profiles:
  - name: openai-api
    runtime: litellm
    provider: openai
    model: openai/gpt-5.2
    api_key_env: OPENAI_API_KEY
    context_window_tokens: 272000

  - name: claude-api
    runtime: litellm
    provider: anthropic
    model: anthropic/claude-sonnet-4-5-20250929
    api_key_env: ANTHROPIC_API_KEY

  - name: openrouter
    runtime: litellm
    provider: openrouter
    model: openrouter/anthropic/claude-sonnet-4-5
    api_key_env: OPENROUTER_API_KEY

  - name: lmstudio
    runtime: litellm
    provider: openai-compatible
    model: openai/local-model
    base_url: http://127.0.0.1:1234/v1
```

`airgapped: true` blocks remote LiteLLM providers. Loopback `base_url` profiles such as LM Studio, Ollama, and vLLM remain allowed.

This is the recommended way to switch between local GGUF presets with different paths, context windows, or GPU offload settings.

## Related docs

- [Quickstart](quickstart.md)
- [Runtime: llama.cpp](runtimes/llama-cpp.md)
- [Python SDK](sdk/python-sdk.md)
