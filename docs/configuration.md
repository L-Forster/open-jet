# Configuration

Main settings live in `config.yaml`. A project that embeds the SDK can pin its own model
on top of that with a [project overlay](#project-configuration-openjetconfigyaml).

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
      llama_mtp: true
      model_size_mb: 9626
      kv_bytes_per_token: 34816
    - max_ram_gb: 16
      label: Qwen3.6 27B UD-IQ3_XXS MTP
      filename: Qwen3.6-27B-UD-IQ3_XXS-MTP.gguf
      url: https://huggingface.co/unsloth/Qwen3.6-27B-MTP-GGUF/resolve/main/Qwen3.6-27B-UD-IQ3_XXS.gguf?download=true
      llama_mtp: true
      model_size_mb: 12288
      kv_bytes_per_token: 34816
    - max_ram_gb: 20
      label: Qwen3.8 27B Q4_K_M MTP
      filename: Qwen3.8-27B-Q4_K_M.gguf
      url: https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/resolve/main/Qwen3.8-27B-Q4_K_M.gguf?download=true
      llama_mtp: true
      model_size_mb: 16896
      kv_bytes_per_token: 34816
    - max_ram_gb: 24
      label: Qwen3.6 35B A3B UD-Q3_K_XL MTP
      filename: Qwen3.6-35B-A3B-UD-Q3_K_XL-MTP.gguf
      url: https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF/resolve/main/Qwen3.6-35B-A3B-UD-Q3_K_XL.gguf?download=true
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

## Project configuration (`.openjet/config.yaml`)

`openjet project` writes a second, smaller config inside the project it is run in. It is
an overlay, not a replacement: the machine-wide `config.yaml` still supplies the device
profile, `llama_server_path`, telemetry consent, MCP servers, and everything else, and
the project file overrides only the model.

```yaml
# your-project/.openjet/config.yaml
project:
  model_id: qwen35-4b-q4km
  use_case: dialogue
  target: handheld
  budget_gb: 4.0
model_source: direct
llama_model: /home/you/your-project/.openjet/models/Qwen3.5-4B-Q4_K_M.gguf
context_window_tokens: 4096
filename: Qwen3.5-4B-Q4_K_M.gguf
model_size_mb: 2806
model_download_url: https://huggingface.co/unsloth/Qwen3.5-4B-GGUF/resolve/main/Qwen3.5-4B-Q4_K_M.gguf?download=true
model_download_path: /home/you/your-project/.openjet/models/Qwen3.5-4B-Q4_K_M.gguf
```

### Which keys a project may own

Only model selection keys: `project`, `model_source`, `llama_model`,
`context_window_tokens`, `llama_mtp`, `llama_cpu_moe`, `llama_n_cpu_moe`,
`model_download_url`, `model_download_path`, `model_size_mb`, `filename`. Anything else in
the file is ignored on write.

The scope is deliberate. A project overlay describes *what model this application ships
with*, which is portable. Device profile, GPU layers, and shell targets describe *the
machine you happen to be on*, and freezing a snapshot of one developer's machine into a
project would drift the moment anyone else built it.

### How the overlay is found

`OpenJetSession` and every `openjet` subcommand walk up from the current directory
looking for a `.openjet/` directory, and stop at the first `.git/` they reach. So the
overlay applies from anywhere inside the project — `src/`, a test, your app's
entrypoint — and never leaks into a sibling checkout.

### Precedence

Lowest to highest:

1. `./config.yaml` in the working directory, if present, otherwise the installed
   `config.yaml`
2. `.openjet/config.yaml` from the nearest enclosing project

The project overlay is applied twice: once before normalization, and again after, so that
a release migration re-deriving the model from managed paths cannot outrank an explicit
project pin.

`openjet --status` prints the files feeding the active configuration, lowest precedence
first:

```
Project model: qwen35-4b-q4km (use case: dialogue, target: handheld)
Config: /home/you/.openjet/config.yaml <- /home/you/your-project/.openjet/config.yaml
```

One `Config:` entry means no overlay is in effect.

### Version control

`.openjet/` ignores itself — `openjet project` writes `.openjet/.gitignore` containing
`*` on first run. Weights do not belong in a repository, and the overlay records absolute
machine-local paths. Model choice is a hardware decision rather than a project one, so
each developer and each build target runs `openjet project` for the machine in front of
them.

See [SDK quickstart](sdk/quickstart.md) and [Choosing a model](models.md).

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

Profiles may use the local `llama_cpp` runtime, OpenAI Codex OAuth, or LiteLLM-hosted providers such as OpenRouter:

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
    model: gpt-5.6-sol
    context_window_tokens: 1050000
    reasoning_effort: medium
    reasoning_summary: auto
    text_verbosity: medium

  - name: ox-alpha
    runtime: litellm
    provider: openrouter
    model: openrouter/stealth/ox-alpha
    api_key_env: OPENROUTER_API_KEY
    context_window_tokens: 1048576
```

### Codex OAuth

Use `/connect openai-codex` to sign in through the official Codex CLI ChatGPT OAuth flow, or `/connect openai-codex --device-auth` for Codex CLI's device-code flow on SSH/headless systems, then `/mode` to choose Local, Codex, OpenRouter, or Slipstream. Codex OAuth is not API-key auth: OpenJet reads the Codex CLI OAuth session from `$CODEX_HOME/auth.json` or `~/.codex/auth.json` and sends requests to the Codex backend. `airgapped: true` disables Codex login and cloud agent modes while preserving local llama.cpp profiles.

### OpenRouter and other API keys

API-key providers use the optional LiteLLM runtime. Install it with `pip install open-jet[cloud]`.

In the Pi TUI:

- `/login` or `/connect openrouter` saves an OpenRouter API key to the OS keyring
- `/cloud` opens the curated OpenRouter picker (featured free model: Ox Alpha)
- `/mode openrouter` runs OpenRouter only (local llama.cpp is not loaded)
- `/mode slipstream` (or `hybrid`) can use OpenRouter or Codex as the orchestrator with a local worker

OpenJet does not write provider secrets to OpenJet-owned JSON files. API-key `/connect` uses OS keyring storage; environment variables such as `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, and `OPENROUTER_API_KEY` take precedence and are the recommended path for headless systems. The TUI sends OpenRouter keys on a dedicated `apiKey` RPC field rather than embedding them in slash-command text.

Curated OpenRouter presets (also listed in `/cloud`) live in `src/openrouter_catalog.py` and are emitted into the TUI during `scripts/build_tui.py`:

```yaml
model_profiles:
  - name: openai-api
    runtime: litellm
    provider: openai
    model: openai/gpt-5.5
    api_key_env: OPENAI_API_KEY
    context_window_tokens: 272000

  - name: claude-api
    runtime: litellm
    provider: anthropic
    model: anthropic/claude-opus-4-8
    api_key_env: ANTHROPIC_API_KEY

  - name: ox-alpha
    runtime: litellm
    provider: openrouter
    model: openrouter/stealth/ox-alpha
    api_key_env: OPENROUTER_API_KEY
    context_window_tokens: 1048576

  - name: openrouter-free
    runtime: litellm
    provider: openrouter
    model: openrouter/openrouter/free
    api_key_env: OPENROUTER_API_KEY

  - name: claude-opus-openrouter
    runtime: litellm
    provider: openrouter
    model: openrouter/anthropic/claude-opus-4.8
    api_key_env: OPENROUTER_API_KEY

  - name: gemini-openrouter
    runtime: litellm
    provider: openrouter
    model: openrouter/google/gemini-3.1-pro-preview
    api_key_env: OPENROUTER_API_KEY

  - name: grok-openrouter
    runtime: litellm
    provider: openrouter
    model: openrouter/x-ai/grok-4.20
    api_key_env: OPENROUTER_API_KEY

  - name: deepseek-v4-openrouter
    runtime: litellm
    provider: openrouter
    model: openrouter/deepseek/deepseek-v4-pro
    api_key_env: OPENROUTER_API_KEY

  - name: glm-openrouter
    runtime: litellm
    provider: openrouter
    model: openrouter/z-ai/glm-5.1
    api_key_env: OPENROUTER_API_KEY
    context_window_tokens: 202752

  - name: kimi-openrouter
    runtime: litellm
    provider: openrouter
    model: openrouter/moonshotai/kimi-k2.5
    api_key_env: OPENROUTER_API_KEY
    context_window_tokens: 262144
```

`airgapped: true` blocks remote LiteLLM providers. Loopback `base_url` profiles remain allowed.

This is the recommended way to switch between local GGUF presets with different paths, context windows, or GPU offload settings.

## Related docs

- [Quickstart](quickstart.md)
- [Runtime: llama.cpp](runtimes/llama-cpp.md)
- [Python SDK](sdk/python-sdk.md)
