# open-jet

Minimal agentic TUI for offline edge devices (Jetson-class targets).

## Setup

1. Install `llama-server` (from `llama.cpp`) on your device and ensure it is on `PATH` (or at `~/llama.cpp/build/bin/llama-server`).
2. Put your `.gguf` model on disk.
3. Create a venv and install:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

4. Run:

```bash
open-jet
```

To force the setup wizard at startup:

```bash
open-jet --setup
```

On first run, the TUI runs a setup wizard and saves config to `config.yaml`:
- model selection/path
- device (`auto` / `cuda` / `cpu`)
- context window tokens (recommended from current RAM/headless runtime)
- GPU layers
- setup navigation uses TUI selects (Tab + Up/Down), with `Ctrl+S` to save

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
