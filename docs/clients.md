# Coding tools: Claude Code, Codex, OpenCode and OpenClaw

`openjet serve` runs the model `openjet setup` configured, on OpenJet's engine when your GPU and model
support it and on llama.cpp otherwise, and answers three APIs:

| API | Endpoint | Used by |
|---|---|---|
| Anthropic Messages | `POST /v1/messages`, `POST /v1/messages/count_tokens` | Claude Code |
| OpenAI Responses | `POST /v1/responses` | Codex CLI |
| OpenAI Chat Completions | `POST /v1/chat/completions` | OpenCode, OpenClaw, most other tools |

```bash
openjet serve                 # http://127.0.0.1:8080; --host and --port change it
```

Leave it running in its own terminal. It prints which engine is serving. Stop OpenJet's own agent
first: both load the model onto the same GPU.

`openjet connect <tool>` prints the configuration below with your server address, model name and
context window filled in. Pass the same `--host` / `--port` you gave `openjet serve`.

## Claude Code

```bash
export ANTHROPIC_BASE_URL=http://127.0.0.1:8080
export ANTHROPIC_AUTH_TOKEN=openjet
export ANTHROPIC_MODEL=Qwen3.8-27B-Q4_K_M
export ANTHROPIC_DEFAULT_HAIKU_MODEL=Qwen3.8-27B-Q4_K_M
claude
```

The token can be any value; the server does not check it. Setting the Haiku model sends Claude
Code's background requests, such as session titles, to the same server. To keep the settings, put
the same keys in the `env` block of `~/.claude/settings.json`.

Claude Code's system prompt and tool definitions are large. Give the server the largest context
your GPU fits with `openjet context <tokens>`, then restart `openjet serve`. When a prompt does not
fit, the server answers with Anthropic's `prompt is too long` error, and Claude Code compacts the
conversation and retries.

## Codex CLI

Add the provider to `~/.codex/config.toml`:

```toml
[model_providers.openjet]
name = "OpenJet"
base_url = "http://127.0.0.1:8080/v1"
wire_api = "responses"
```

Save this as `~/.codex/openjet.config.toml`:

```toml
model = "Qwen3.8-27B-Q4_K_M"
model_provider = "openjet"
model_context_window = 100000
```

Then run `codex --profile openjet`. Codex's `model_reasoning_effort` maps to the model's thinking
levels: `low` and `minimal` to low, `medium` to medium, `high` and `xhigh` to xhigh, and `none` turns
thinking off.

## OpenCode

Merge this into `opencode.json` in your project, or `~/.config/opencode/opencode.json`:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "openjet": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "OpenJet (local)",
      "options": { "baseURL": "http://127.0.0.1:8080/v1" },
      "models": {
        "Qwen3.8-27B-Q4_K_M": {
          "name": "Qwen3.8-27B-Q4_K_M (OpenJet)",
          "limit": { "context": 100000, "output": 32768 }
        }
      }
    }
  }
}
```

Run `opencode` and pick `openjet/Qwen3.8-27B-Q4_K_M` with `/models`.

## OpenClaw

Merge this into `~/.openclaw/openclaw.json`:

```json
{
  "models": {
    "providers": {
      "openjet": {
        "baseUrl": "http://127.0.0.1:8080/v1",
        "apiKey": "openjet",
        "api": "openai-completions",
        "models": [
          { "id": "Qwen3.8-27B-Q4_K_M", "name": "Qwen3.8-27B-Q4_K_M (OpenJet)",
            "contextWindow": 100000, "maxTokens": 32768 }
        ]
      }
    }
  },
  "agents": { "defaults": { "model": { "primary": "openjet/Qwen3.8-27B-Q4_K_M" } } }
}
```

## Limits on OpenJet's engine

On llama.cpp, these endpoints are llama-server's own. On OpenJet's engine:

- **One request at a time.** Requests that arrive together wait for each other.
- **One cached conversation.** A request that continues the previous conversation only processes
  the new tokens. A request from another conversation, such as a background title request,
  processes its whole prompt, and the next request of the first conversation does too.
- **Text only.** Images and documents in a request are replaced with a short placeholder.
- **Anthropic Messages.** Thinking blocks carry an empty signature. `stop_sequences` and
  `tool_choice` are ignored. Server tools such as web search are dropped from the tool list.
- **OpenAI Responses.** No stored responses: `previous_response_id` is rejected, so the client must
  send the whole conversation in `input`. Function and custom (freeform) tools are supported;
  other tool types, such as web search, are skipped with a warning in the server log.
- **Sampling.** `top_k` is capped at 32; presence, frequency and repetition penalties are ignored.
