# open-jet

Minimal agentic TUI for offline edge devices (Jetson-class targets).

## Setup

1. Install prerequisites:
- `llama-server` from `llama.cpp` must be on `PATH` (or at `~/llama.cpp/build/bin/llama-server`).
- If you plan to use local models, place a `.gguf` file on disk.
- If you plan to download models in setup, install `ollama`.

2. Create a virtualenv and install:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

3. Start the app:

```bash
open-jet
```

4. Optional: force setup wizard even if config already exists:

```bash
open-jet --setup
```

First run opens a setup wizard and writes `config.yaml` in this order:
1. Hardware detection:
Use detected hardware or choose manual.
2. Hardware override (manual only):
Select Jetson + RAM profile.
3. Model source:
Choose local `.gguf` or Ollama download.
4. Local model file (local only):
Pick a detected `.gguf` or enter a path.
5. Context size:
Set prompt context window.
6. GPU offload:
Set GPU layer count.

Controls: `Up/Down` to change option, `Tab/Enter` next, `Shift+Tab` back, `Enter` on final step saves and restarts.

## Usage

- Chat normally with Enter.
- `@file` or `@[path with spaces]` loads file content into context automatically.
- Typing `/` shows slash command suggestions; `Up/Down` cycles and `Tab` autocompletes.
- `Tab` also autocompletes `@` file paths from the current workspace.
- A live token counter is shown under the chat input.
- Tool calls that can change system state still require approval (`y` / `n`).
- Tool approvals include a compact preview of the exact command/write action.
- `Ctrl+C` or `/exit` quits.

## Slash Commands

- `/help`: list commands.
- `/exit`: quit the app.
- `/clear`: clear chat and restart `llama-server` (flush KV cache).
- `/clear-chat`: clear chat only.
- `/status`: show context and RAM status.
- `/condense`: manually condense older context.
- `/load <path>`: preload a text/code file into context.
- `/setup`: reopen setup wizard and restart runtime with new config.

## Edge Device Notes

- File loading is runtime-budgeted using current `MemAvailable` and prompt budget.
- Only text/code files are allowed for context loading.
- Large files are truncated and marked with `...[truncated for context safety]`.
- `load_file` tool calls are clamped to remaining prompt budget at runtime.

## Configuration

Key options in `config.yaml`:

```yaml
context_window_tokens: 2048
memory_guard:
  context_reserved_tokens: null
  min_prompt_tokens: 256
  min_available_mb: null
  max_used_percent: null
  check_interval_chunks: 16
  condense_target_tokens: 900
  keep_last_messages: 6
```

## Session Logging

- `open-jet` now writes structured session logs by default to `session_logs/`.
- Each run creates two labeled files:
- `*.events.jsonl`: user messages, assistant output, tool requests, approvals, tool results, and errors.
- `*.metrics.jsonl`: timestamped system samples (CPU/load average/memory/process RSS) at a fixed interval.
- Configure this in `config.yaml`:

```yaml
logging:
  enabled: true
  directory: session_logs
  label: open-jet
  metrics_interval_seconds: 5
```

## Session Resume

- Conversation state is autosaved to `session_state.json` by default.
- On startup, open-jet does not resume previous session context unless enabled.
- Configure in `config.yaml`:

```yaml
state:
  auto_resume: false
  enabled: true
  path: session_state.json
```
