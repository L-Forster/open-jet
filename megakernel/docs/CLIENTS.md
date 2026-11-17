# Clients

The server speaks the three APIs that coding agents and chat UIs use: OpenAI chat completions, the Anthropic Messages
API and the OpenAI Responses API ([SERVER.md](SERVER.md#apis-and-endpoints)). Each client below ran the same set of
real tasks against it ([how](#how-the-clients-were-tested)). For each one: the settings that work, what goes wrong
without them, and how the server handles the small requests the client sends on the side.

With the settings below every client completed the tasks. Most need two or three settings that no client can guess for a
local model: its context window, whether it takes images, and how to turn thinking off. None needs a setting for the
cache: follow-up turns, new sessions, several sessions at once and the clients' title or summary requests are handled by
the server ([the conversation cache](SERVER.md#the-conversation-cache)), and the replies stay those of the same request
sent fresh.

## What every client needs

- **Base URL:** `http://127.0.0.1:8080/v1` with the launchers' default port; Claude Code takes it without `/v1`.
- **API key:** any non-empty value; the server checks none.
- **Model id:** the GGUF's file name, as `GET /v1/models` reports it (`Qwen3.8-27B-UD-Q4_K_XL.gguf` below). The server
  serves one model and does not check the `model` field, but clients look the id up in their own model tables and find
  nothing, so:
- **Context window:** set it by hand to the server's context (`-c`, the launchers' second argument; the configs below
  use the launchers' default, 120000). A client that assumes less compacts early; one that assumes more sends prompts
  the server refuses with a 400 ("exceeds the available context size", or Anthropic's "prompt is too long").
- **Images:** turn the client's vision flag on when the server runs with an mmproj (the launchers do when they find
  one). Most clients take an unknown model for text-only and leave images out of the request.
- **Thinking:** the model thinks unless told otherwise, at its highest effort (the chat template's default). Some
  clients' "off" level sends nothing (Pi, DeepSeek Harness), which means the most thinking: where the client allows it,
  map "off" to the `reasoning_effort` value `none` (chat completions and Responses; on `/v1/messages` thinking follows
  the request's `thinking` field, off when it is absent). The effort text sits at the very start of the rendered system
  prompt, so changing the level in the middle of a session re-reads the whole prompt once.

| client | tested | API | needed | recommended |
|---|---|---|---|---|
| [Claude Code](#claude-code) | 2.1.306 | Anthropic `/v1/messages` | base URL, key, the five model ids | context tokens; cache-friendly system prompt |
| [Codex CLI](#codex-cli) | 0.163.0 | OpenAI `/v1/responses` | a provider with `wire_api = "responses"` | context window; raw reasoning shown |
| [Qwen Code](#qwen-code) | 0.25.3 | chat completions | context window; image modality | managed auto-memory off |
| [Pi](#pi) | 1.3.0 | chat completions | a `models.json` provider; `off` mapped to `none` | - |
| [DeepSeek Harness](#deepseek-harness-dsh) | 0.2.1-rc.2 | chat completions | a provider in `cordis.patch.yml` | effort map with `off: none` |
| [Crush](#crush) | v0.98.1 | chat completions | provider + model with context window | images flag; network tools denied |
| [Goose](#goose) | 1.55.0 | chat completions | the `openai` provider (custom provider for images) | session naming off |
| [Hermes Agent](#hermes-agent) | v0.22.3 | chat completions | a provider with `api_mode: chat_completions` | vision on; title generation off |
| [Aider](#aider) | 0.86.2 | chat completions | model settings (edit format) and metadata (window, vision) | pass the chat's files on every call |
| [Open WebUI](#open-webui) | 0.12.3 | chat completions | the connection in the admin settings | task settings; builtin tools off; vision capability |
| [OpenCode](#opencode) | 2.0.32 | chat completions | an openai-compatible provider | model limits and modalities |

## Claude Code

Tested: 2.1.306. Route: `POST /v1/messages` (streamed, `?beta=true`) and
`/v1/messages/count_tokens`.

```sh
export ANTHROPIC_BASE_URL=http://127.0.0.1:8080          # no /v1
export ANTHROPIC_API_KEY=sk-local                        # any value
export ANTHROPIC_MODEL=Qwen3.8-27B-UD-Q4_K_XL.gguf
export ANTHROPIC_SMALL_FAST_MODEL=$ANTHROPIC_MODEL
export ANTHROPIC_DEFAULT_HAIKU_MODEL=$ANTHROPIC_MODEL
export ANTHROPIC_DEFAULT_SONNET_MODEL=$ANTHROPIC_MODEL
export ANTHROPIC_DEFAULT_OPUS_MODEL=$ANTHROPIC_MODEL
export CLAUDE_CODE_MAX_CONTEXT_TOKENS=120000             # the server's -c (recommended)
export CLAUDE_CODE_ATTRIBUTION_HEADER=0                  # recommended, with the flag below
claude --exclude-dynamic-system-prompt-sections
```

- **Needed:** the base URL, a key, and all five model variables set to the server's model, so subagents and small
  calls reach the same model.
- **Recommended:** `CLAUDE_CODE_MAX_CONTEXT_TOKENS` = the server's `-c`: for an unknown model Claude Code assumes a
  200,000-token window, so it compacts early on a larger context, and on a smaller one it sends prompts the server
  refuses. `CLAUDE_CODE_ATTRIBUTION_HEADER=0` with `--exclude-dynamic-system-prompt-sections`: by default each session's
  system prompt starts with a billing header that differs per session, and carries the working directory and memory
  path; with these two the system prompt is the same in every directory, and a new session reuses its ~9k tokens of
  system prompt and tools (9,024 of 11,809 tokens instead of 8,288 of 11,850 with the defaults; measured in
  [SERVER.md](SERVER.md#new-sessions-pinned-prefix-checkpoints)).
- **Effort:** `/effort low` and `/effort medium` set the template's reasoning effort; `high` (the default) and `max`
  are its highest, `xhigh`.
- **Side requests:** in `claude -p` runs Claude Code sent no title or other small requests. Subagents (the Agent tool)
  are conversations of their own, with tools: each is parked like any conversation, and the main one comes back from
  host memory when the subagent is done. `/compact` is one request that continues the session. The Read tool counts a
  file's tokens with `count_tokens` first; the console logs one line per count.
- **Quirks:** Claude Code 2.1 sends its environment as a system message after the first user message; the server
  renders it as a user message in place (Qwen's template takes a system message only at the start), so turns still
  match the cache to the last token. Thinking blocks come back with an empty signature and re-render identically.
  Every run prints `[claude-code:unrecognized_model]` to stderr and prices the run at an unknown-model default:
  cosmetic. Claude Code's Read stopped at 2,240 of a 4,903-line file and the model went on with Grep.
  `--dangerously-skip-permissions` runs tools unattended in `-p` mode.

## Codex CLI

Tested: 0.163.0. Route: `POST /v1/responses` only (Codex removed its chat-completions wire API).

In `~/.codex/config.toml`:

```toml
model = "Qwen3.8-27B-UD-Q4_K_XL.gguf"
model_provider = "mk"
model_context_window = 120000      # the server's -c: Codex has no metadata for this model (it warns, harmless)
show_raw_agent_reasoning = true    # show the thinking (not in exec --json output)

[model_providers.mk]
name = "qwen38-megakernel"
base_url = "http://127.0.0.1:8080/v1"
wire_api = "responses"
env_key = "MK_API_KEY"             # set MK_API_KEY to any value
```

- **Needed:** the provider block with `wire_api = "responses"`.
- **Recommended:** `model_context_window` (Codex falls back to default metadata and prints "Model metadata ... not
  found"); `show_raw_agent_reasoning` to see the thinking: the server returns the reasoning as text and generates no
  reasoning summaries.
- **Effort:** `model_reasoning_effort` low, medium, high (= xhigh, the template's default) and xhigh work; minimal is
  low; max and ultra are taken as xhigh; none turns thinking off.
- **Side requests:** none seen; every Codex request carries its tools.
- **Tasks run:** create a file, find and fix a failing test, a `codex exec resume` follow-up that adds a function and
  its test, a question about a 4,900-line file read in chunks (to 69k tokens of context), an image (`-i` and Codex's
  `view_image` tool). The first session prefilled ~8.8k tokens (Codex's instructions and tools); then each turn a few
  hundred tokens plus its tool output; `codex exec resume` reused the whole earlier conversation (10,016 cached + 174
  prefilled). With pinned checkpoints a later new session reuses 8,672 of its 8,847 prompt tokens
  ([SERVER.md](SERVER.md#new-sessions-pinned-prefix-checkpoints)).
- **Quirks:** in these runs `-s workspace-write` refused every command ("blocked by policy"); the tasks ran with
  `--dangerously-bypass-approvals-and-sandbox` in scratch directories. Codex always sends a `web_search` tool, which
  the server does not offer to the model (a console warning).

## Qwen Code

Tested: 0.25.3. Route: chat completions, streamed.

Environment: `OPENAI_BASE_URL=http://127.0.0.1:8080/v1`, `OPENAI_API_KEY` (any value),
`OPENAI_MODEL=Qwen3.8-27B-UD-Q4_K_XL.gguf`, and auth type `openai` (`qwen --auth-type openai`). In
`~/.qwen/settings.json`:

```json
{
  "security": { "auth": { "selectedType": "openai" } },
  "general": { "enableAutoUpdate": false },
  "memory": { "enableManagedAutoMemory": false },
  "model": { "generationConfig": { "contextWindowSize": 120000, "modalities": { "image": true } } }
}
```

- **Needed:** `contextWindowSize` = the server's `-c`. Qwen Code maps model ids that start with `qwen3.` to a
  1,000,000-token window, so it never compacts before the server's limit; at the limit its compaction request carries
  the whole oversized history and is refused too. `modalities.image: true` for images: ids that start with "qwen" are
  taken as text-only, and the model then decoded a PNG by hand with Python over 25 turns.
- **Recommended:** `memory.enableManagedAutoMemory: false`. The managed memory extractor runs after every reply as a
  second conversation with its own tools (4-66 s here), and the memory catalog it adds moves to the end of every
  request, so each turn re-reads the previous reply. Without it, follow-ups match the cached prompt and reply to the
  last token. `general.enableAutoUpdate: false` and `QWEN_CODE_MODELS_DEV=off` (it downloads a model catalog at startup)
  if you want no outbound traffic.
- **Side requests:** Qwen Code sends no one-shots; the memory extractor (above) is a conversation of its own, with
  tools, and is parked like one. The compaction summary (no tools, the whole conversation as messages, thinking off) is
  longer than a side request may be; the conversation it summarises is parked with low priority (normal again if the
  answer is shorter than a summary's 256 tokens).
- **Quirks:** `--resume` / `--continue` do not re-send the conversation byte for byte (the folder listing and a date
  reminder move), so a resumed session re-reads from its first user message; for long work keep one process running.
  After an auto-compaction on a small window, Qwen Code once sent its next request with `max_tokens` 1 (a client bug,
  likely only below a ~67k window): give it a large context.

## Pi

Tested: 1.3.0. Route: chat completions, streamed (system prompt as role `developer`).

In `~/.pi/agent/models.json`:

```json
{
  "providers": {
    "mk": {
      "baseUrl": "http://127.0.0.1:8080/v1",
      "api": "openai-completions",
      "apiKey": "$MK_API_KEY",
      "models": [
        {
          "id": "Qwen3.8-27B-UD-Q4_K_XL.gguf",
          "name": "Qwen3.8-27B (megakernel)",
          "reasoning": true,
          "input": ["text", "image"],
          "contextWindow": 120000,
          "maxTokens": 32768,
          "thinkingLevelMap": { "off": "none" }
        }
      ]
    }
  }
}
```

and in `settings.json` `"defaultProvider": "mk"`, `"defaultModel": "Qwen3.8-27B-UD-Q4_K_XL.gguf"`.

- **Needed for `--thinking off`:** `"thinkingLevelMap": {"off": "none"}`. Pi sends no `reasoning_effort` for off, and
  the template's default then thinks the most.
- **Side requests:** a compaction summary (no tools; 3.0k tokens for a 49.8k-token conversation) runs beside the
  conversation it summarises while it fits `--side-max`; the summarised conversation, which Pi replaces by the summary,
  is parked with low priority when the next request replaces it (a longer summary parks it at once with low priority,
  normal again if the answer is shorter than 256 tokens).
- **Quirks:** Pi runs tools without asking and has no turn limit: bound unattended runs yourself. Images work with
  `input` listing `image` (`pi -p @image.png "..."`).

## DeepSeek Harness (dsh)

Tested: 0.2.1-rc.2, headless profile. Route: chat completions, streamed.

In `~/.dsh/cordis.patch.yml`:

```yaml
- id: llm-pi-ai
  config:
    providers:
      mk:
        displayName: Qwen3.8 megakernel
        apiKeyEnv: MK_API_KEY
        api: openai-completions
        baseURL: http://127.0.0.1:8080/v1
        models:
          - id: Qwen3.8-27B-UD-Q4_K_XL.gguf
            name: Qwen3.8-27B (megakernel)
            contextWindow: 120000
            input: [text, image]
            reasoningEfforts:
              off: none
              low: low
              medium: medium
              high: high
- id: agent-default-model
  config:
    provider: mk
    model: Qwen3.8-27B-UD-Q4_K_XL.gguf
    reasoningEffort: medium
```

- **Needed:** the provider (`contextWindow`, `input` with `image` for images) and the default model.
- **Recommended:** the `reasoningEfforts` map: without it dsh sends no `reasoning_effort` and the model thinks at its
  highest. The keys are dsh's levels and the values the wire spelling, so `off: none` is the line that turns thinking
  off (a `none:` key is rejected).
- **Side requests:** every new session sends a title request (no tools, `max_completion_tokens` 4096): it runs beside
  the session, so it needs no cap. Compaction requests carry the 25 tools and are normal requests.
- **Quirks:** the default permission mode (workspace-write, a Windows ACL sandbox) ran every tool unattended inside the
  task directory. There is no turn limit in headless runs.

## Crush

Tested: v0.98.1. Route: chat completions, streamed.

```sh
crush provider add mk --type openai-compat --base-url "http://127.0.0.1:8080/v1" --api-key "$MK_API_KEY"
crush model add mk/Qwen3.8-27B-UD-Q4_K_XL.gguf --name "Qwen3.8-27B (megakernel)" --context-window 120000 \
    --default-max-tokens 32768 --supports-images true
```

(These also work as lines of the `crushrc` file, with LF line ends, plus `option provider-auto-update false`.)

- **Needed:** the provider and the model with its context window; `--supports-images true` for images (without it
  Crush does not send them).
- **Recommended:** `permissions deny fetch download agentic_fetch sourcegraph` removes the network tools (26 to 22
  tools); `permissions allow bash edit multiedit view write grep glob ls todos job_output job_kill` is the way to run
  tools unattended in `crush run`.
- **Side requests:** titles (no tools) run beside the session. Crush ends them with `/no_think` and an empty
  `<think></think>`, Qwen3's soft switches, which the server applies (Qwen3.8's template ignores them), so titles come
  back short without turning thinking off for the whole provider. Auto-summarize (no tools, the conversation as
  messages) runs beside the session while it fits `--side-max`; a longer one parks the summarised conversation with low
  priority (normal again if the answer is shorter than 256 tokens).
- **Quirks:** `--can-reason` is not needed (the model thinks anyway). Crush stores absolute paths in its session
  database: a copied project directory continues editing the original.

## Goose

Tested: 1.55.0. Route: chat completions, streamed.

Environment for the built-in `openai` provider:

```sh
export GOOSE_PROVIDER=openai
export GOOSE_MODEL=Qwen3.8-27B-UD-Q4_K_XL.gguf
export OPENAI_HOST=http://127.0.0.1:8080
export OPENAI_BASE_PATH=v1/chat/completions
export OPENAI_API_KEY=sk-local
export GOOSE_MODE=auto                       # run tools unattended
export GOOSE_DISABLE_SESSION_NAMING=true     # optional, below
```

For images, a custom provider instead, `custom_providers/mk_local.json` in Goose's config directory, used with
`--provider mk_local --model Qwen3.8-27B-UD-Q4_K_XL.gguf`:

```json
{"name": "mk_local", "engine": "openai", "display_name": "MK local", "description": "megakernel server",
 "api_key_env": "OPENAI_API_KEY", "base_url": "http://127.0.0.1:8080/v1/chat/completions",
 "models": [{"name": "Qwen3.8-27B-UD-Q4_K_XL.gguf", "context_limit": 120000, "supports_vision": true}],
 "supports_streaming": true}
```

- **Needed for images:** the custom provider with `supports_vision: true`; otherwise Goose drops images ("omitted as
  the model does not support vision").
- **Optional:** `GOOSE_DISABLE_SESSION_NAMING=true` (or `-n <name>`) saves the naming request of each new session
  (1-3 s with thinking on); it runs beside the session either way.
- **Side requests:** session naming (no tools, no `max_tokens`) runs beside the session; compaction summaries (no
  tools, the conversation in the system message) run beside it too while they fit `--side-max`.
- **Quirks:** Goose's compaction summaries think as long as they like (85 s once): the `openai` provider has no
  per-request thinking switch (`GOOSE_THINKING_EFFORT` was not tried). Goose loads the skills and agents of other tools
  from the user profile (`~/.claude/skills`, `~/.claude/agents`, ...) into its system prompt (~3k tokens in testing);
  on Windows it finds the profile through the Known Folder API, so a redirected `USERPROFILE` does not change that. Its
  `cmd.exe` shell cuts multi-line commands at the first newline (`GOOSE_SHELL=powershell` avoids it).

## Hermes Agent

Tested: v0.22.3. Route: chat completions, streamed; at startup it probes `/props` and a few other routes
(three harmless 404s) and takes the server for llama-server.

In `~/.hermes/config.yaml`:

```yaml
model:
  provider: mk
  default: Qwen3.8-27B-UD-Q4_K_XL.gguf
  base_url: http://127.0.0.1:8080/v1
  api_mode: chat_completions
  supports_vision: true          # recommended
providers:
  mk:
    api: http://127.0.0.1:8080/v1
    api_key: ${MK_API_KEY}
    discover_models: false
auxiliary:
  title_generation:
    enabled: false               # recommended
```

- **Needed:** the provider with `api_mode: chat_completions`. The context size comes from `/props`, so
  `context_length` under the model is optional.
- **Recommended:** `model.supports_vision: true`: by default Hermes sends each image in a separate describe-the-image
  request and the main model sees only the text description. `auxiliary.title_generation.enabled: false`: its title
  request asks for a JSON schema, which the server refuses with a 400 (it has no constrained sampling); Hermes logs a
  warning and goes on, once per run.
- **Side requests:** the image describer and compaction summaries (no tools, one message) run beside the session while
  they fit `--side-max`.
- **Quirks:** `--resume` needs the same `--max-turns` as the session's first run. Its compaction waits only 10 s for
  the summary, so the user's next turn can queue behind it.

## Aider

Tested: 0.86.2. Route: chat completions, streamed; aider sends no tools and no `max_tokens`.

Environment: `OPENAI_API_BASE=http://127.0.0.1:8080/v1`, `OPENAI_API_KEY` (any value),
`AIDER_MODEL=openai/Qwen3.8-27B-UD-Q4_K_XL.gguf`. In `.aider.model.settings.yml`:

```yaml
- name: openai/Qwen3.8-27B-UD-Q4_K_XL.gguf
  edit_format: diff
  use_repo_map: true
  weak_model_name: openai/Qwen3.8-27B-UD-Q4_K_XL.gguf
  examples_as_sys_msg: true
```

and in `.aider.model.metadata.json`:

```json
{"openai/Qwen3.8-27B-UD-Q4_K_XL.gguf": {"max_input_tokens": 120000, "max_output_tokens": 32768,
  "input_cost_per_token": 0, "output_cost_per_token": 0, "litellm_provider": "openai", "mode": "chat",
  "supports_vision": true}}
```

- **Needed:** both files. For an unknown model aider uses the "whole" edit format, which here produced stray files and
  long thinking; with `diff` the tasks went right. The metadata gives the window (and a larger repo map) and turns
  images on. `LITELLM_LOCAL_MODEL_COST_MAP=True` stops a price-list download at startup.
- **Pass the chat's files on every call:** `--restore-chat-history` restores the messages but not the files, and the
  model then edits nothing. Keep the flags of a session identical too (`--test-cmd` changes the system prompt).
- **Side requests:** commit messages and history summaries (no tools, a first turn) run beside the chat. Aider's main
  requests always carry an assistant message (the repo map is a user / "Ok" pair), so they are never taken for
  one-shots.
- **Quirks:** aider shows the thinking and keeps it in the chat history, re-sent with every later request. Adding a
  file moves everything after the repo map, so the next request re-reads from there.

## Open WebUI

Tested: 0.12.3, API path and UI tasks. Route: `GET /v1/models` and chat completions; the browser talks to
Open WebUI's backend, not to the server, so no CORS setting is needed.

- **Needed:** the connection under Admin > Settings > Connections > OpenAI: `http://127.0.0.1:8080/v1`, any key.
  `OPENAI_API_BASE_URL` in the environment only counts on Open WebUI's first start; afterwards every setting lives in
  its database.
- **Recommended, task settings** (Admin > Settings > Interface): task model parameters `{"reasoning_effort": "none"}`
  (follow-up suggestions think 1-2k tokens otherwise), and title / tags templates that quote only the end of the chat:
  the default templates with `{{MESSAGES:END:2|middletruncate:1500}}` (title) and
  `{{MESSAGES:END:6|middletruncate:1500}}` (tags). By default each task quotes the last messages whole, so a long chat
  is read again for every task (after a 52k-token paste: +63 s per turn); truncated, the tasks stay small and run beside
  the chat. Tags and follow-ups can also be turned off.
- **Recommended, the model** (Admin > Workspace > Models, id = the GGUF's file name): capability Vision on (the UI
  hides image upload otherwise), Builtin Tools off: version 0.12.3 adds 28 built-in tools (~5.4k tokens) to every chat
  request, and the model used them instead of looking at an attached image. Advanced parameter `reasoning_effort` (low
  / none) to think less.
- **Side requests:** title, tags and follow-up requests (no tools, one user message) run beside the chat while their
  prompt fits `--side-max` (8,192 tokens), which the truncated templates ensure. With the default templates a long
  chat's tasks are longer: each is read in full and parks the chat, which its next turn restores (their answers are
  short, so the chat keeps normal priority); the title task's own conversation is parked only into room that side and
  low-priority conversations leave. A chat without tools (Builtin Tools off) runs beside an agent's conversation while
  it fits `--side-max`, and keeps its own conversation cached for the next turn.
- **Quirks:** the emoji picker sends `max_tokens` 4 without the task parameters, so the model spends them thinking and
  returns an empty reply (cosmetic). Open WebUI does not limit what it sends to the model's context.

## OpenCode

Tested: 2.0.32 (native Windows and WSL2). Route: chat completions, streamed.

In `~/.config/opencode/opencode.json`:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "qwen38": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Qwen3.8-27B megakernel (local)",
      "options": { "baseURL": "http://127.0.0.1:8080/v1" },
      "models": {
        "qwen3.8-27b": {
          "name": "Qwen3.8 27B megakernel",
          "reasoning": true,
          "interleaved": { "field": "reasoning_content" },
          "tool_call": true,
          "attachment": true,
          "temperature": true,
          "modalities": { "input": ["text", "image"], "output": ["text"] },
          "limit": { "context": 120000, "output": 32768 }
        }
      }
    }
  },
  "model": "qwen38/qwen3.8-27b",
  "small_model": "qwen38/qwen3.8-27b"
}
```

- **Needed:** an openai-compatible provider with the base URL. The model key may be any name (the server does not check
  it). The model entry gives the context limit, images and the reasoning field, which round-trips as
  `reasoning_content`.
- **Side requests:** titles run beside the session and are kept as side conversations. Two sessions at once share the
  engine through parking: 110 s with ~72k tokens prefilled against 242 s and ~232k with the host store off
  (230,000-token context, preset D).
- **Quirks:** OpenCode's background service caches its config: run `opencode service restart` after an edit. OpenCode
  sends no sampling fields, so the server's defaults (Unsloth's Qwen3.8 thinking preset) apply.

## Other clients

Any client of one of the three APIs should work; the server's tests use the official OpenAI and Anthropic Python SDKs,
streams included. Three things to check: the context window and vision flag (above); that the client does not need a
feature the server refuses with a 400 (JSON schema output, forced tool choice, log probabilities, `n` > 1: see
[SERVER.md](SERVER.md#sampling)); and, for a browser app served from another origin, `--cors-origin`
([SERVER.md](SERVER.md#security)).

## How the clients were tested

- **Live runs:** native Windows, RTX 3090; the test server ran with a 230,000-token context, preset D, 7
  drafts, exact reuse, images and the 8 GiB host store. Each client was installed in its own directory with its own
  home, config and caches. Tasks: create a file with tests, find and fix a failing test, a follow-up in the same
  session, a question about a 4,900-line (~30k-token) file, an image, switching between sessions, and each client's own
  features (compaction, subagents, titles, thinking levels). With the settings above every client completed the tasks;
  without them, images failed in Qwen Code, Goose and Hermes (described or decoded by hand instead) and Aider's default
  edit format left stray files. The model slipped twice: Aider's arithmetic on the long-file question, and a line
  number off by two in Pi and dsh.
- **Server version:** the live runs predate some of the current cache rules: one-shot requests without tools run
  beside the conversation whatever their `max_tokens`, a request that only edits a conversation's later history parks
  nothing, compaction leftovers are parked with low priority, new sessions resume at pinned checkpoints, Qwen3's soft
  switches are applied, and Codex's max / ultra efforts are accepted. Those were checked by replaying the captured
  requests of the live runs ([SERVER.md](SERVER.md#measured-on-real-client-traffic)) and with requests of each
  client's shapes against the real model.
- **What the configs above change:** the base URL (port 8080 instead of the test port), the context (120000, the
  launchers' default, instead of 230000), the API key and display names are placeholders; the shell lines are written
  for a POSIX shell (the runs used `set` in `cmd`); settings that only served the test setup (telemetry and update
  checks off, a logging proxy, Codex's approval and sandbox modes) are left out. The rest is what ran.
