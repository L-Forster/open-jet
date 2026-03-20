# Runtime: OpenRouter

Use OpenRouter when you want an optional hosted profile alongside your local or self-hosted setup.

## Minimal config

```yaml
runtime: openrouter
openrouter_model: openai/gpt-4o-mini
openrouter_api_key_env: OPENROUTER_API_KEY
openrouter_base_url: https://openrouter.ai/api/v1
context_window_tokens: 8192
```

## Optional fields

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

## Setup wizard

`open-jet --setup` can configure this runtime directly. Choose `Hosted API: OpenRouter`.
