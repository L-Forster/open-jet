# Configuration

Main settings live in `config.yaml`.

## Supported runtimes

- `llama_cpp`
- `openai_compatible`
- `openrouter`

`sglang` and `trtllm_pytorch` are disabled in this simplified build.

OpenJet is positioned for local and self-hosted use first. `openrouter` is the optional hosted profile.

## Minimal local/offline config

```yaml
runtime: llama_cpp
llama_model: /home/you/models/Qwen3.5-4B-Q4_K_M.gguf
model_source: local
llama_server_path: /home/you/llama.cpp/build/bin/llama-server
context_window_tokens: 4096
device: cuda
gpu_layers: 99
```

`llama_server_path` is optional. Setup will populate it automatically when it provisions or discovers `llama-server`.

Ollama-backed local setup:

```yaml
runtime: llama_cpp
model_source: ollama
ollama_model: qwen3:4b
recommended_llm: qwen3:4b
context_window_tokens: 4096
device: cuda
gpu_layers: 99
```

If setup starts from a direct GGUF download, it will download into the OpenJet models directory and then persist the resolved `llama_model` path for future runs.

## Minimal self-hosted OpenAI-compatible config

```yaml
runtime: openai_compatible
openai_compatible_model: gpt-4o-mini
openai_compatible_base_url: https://api.openai.com
openai_compatible_api_key_env: OPENAI_API_KEY
context_window_tokens: 8192
```

Useful optional fields:

```yaml
openai_compatible_headers:
  X-Team: local-dev
openai_compatible_extra_body:
  reasoning:
    effort: medium
openai_compatible_verify_connection: false
```

## Optional OpenRouter config

```yaml
runtime: openrouter
openrouter_model: openai/gpt-4o-mini
openrouter_api_key_env: OPENROUTER_API_KEY
openrouter_base_url: https://openrouter.ai/api/v1
context_window_tokens: 8192
```

Useful optional fields:

```yaml
openrouter_site_url: https://example.com
openrouter_app_name: OpenJet
openrouter_headers:
  X-Team: local-dev
openrouter_extra_body:
  provider:
    sort: latency
openrouter_verify_connection: false
```

## General settings

Common fields across runtimes:

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

This is the recommended way to switch between:

- local `llama.cpp`
- a self-hosted OpenAI-compatible API
- an optional OpenRouter fallback

## Related docs

- [Quickstart](quickstart.md)
- [Runtime: llama.cpp](runtimes/llama-cpp.md)
- [Runtime: OpenAI-compatible](runtimes/openai-compatible.md)
- [Runtime: OpenRouter](runtimes/openrouter.md)
- [Python SDK](sdk/python-sdk.md)
