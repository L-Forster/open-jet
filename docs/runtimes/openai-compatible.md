# Runtime: OpenAI-compatible

Use this runtime when you already have a self-hosted gateway or another OpenAI-compatible endpoint.

This runtime is intended primarily for self-hosted use, though hosted OpenAI-style APIs also work.

Supported targets include:

- OpenAI
- local gateways that expose an OpenAI-style `/v1/chat/completions` API
- self-hosted services with OpenAI-compatible request and streaming behavior

## Minimal config

```yaml
runtime: openai_compatible
openai_compatible_model: gpt-4o-mini
openai_compatible_base_url: https://api.openai.com
openai_compatible_api_key_env: OPENAI_API_KEY
context_window_tokens: 8192
```

## Local gateway example

```yaml
runtime: openai_compatible
openai_compatible_model: local
openai_compatible_base_url: http://127.0.0.1:9000
openai_compatible_api_key_env: OPENAI_API_KEY
context_window_tokens: 8192
```

## Optional fields

```yaml
openai_compatible_headers:
  X-Team: local-dev
openai_compatible_extra_body:
  reasoning:
    effort: medium
openai_compatible_verify_connection: false
```

## Setup wizard

`open-jet --setup` can configure this runtime directly. Choose `Self-hosted API: OpenAI-compatible`.
