# open-jet

`open-jet` is an offline-first terminal app for running local LLM workflows on edge Linux devices (including Jetson-class hardware).

It provides:
- local chat with your on-device model
- safe file-context loading with token/memory guards
- slash commands for session control
- first-run setup for model/runtime configuration
- optional session logging and resume

## Requirements

Before running `open-jet`, make sure:
- `llama-server` from `llama.cpp` is installed and available on `PATH`
- you have a local `.gguf` model file, or `ollama` installed for model download

## Install

```bash
pip install open-jet
```

## Start

```bash
open-jet
```

Optional setup screen on launch:

```bash
open-jet --setup
```

## First-Run Setup

On first run, `open-jet` guides you through:
1. hardware detection/profile
2. model source selection
3. model path or download choice
4. context window size
5. GPU offload configuration

It then saves your configuration and starts the runtime.

## Basic Use

- Type normally and press Enter to chat
- Use `@file` or `@[path with spaces]` to add file content to context
- Type `/` to open slash-command suggestions
- `Tab`/`Enter` can autocomplete slash commands and file mentions
- `Ctrl+C` or `/exit` quits

## Slash Commands

- `/help` show commands
- `/exit` quit app
- `/clear` clear chat and restart `llama-server`
- `/clear-chat` clear chat only
- `/status` show context/RAM status
- `/condense` condense older context
- `/load <path>` load a file into context
- `/resume` load previous saved session
- `/setup` reopen setup wizard

## Configuration

Main settings are stored in `config.yaml`, including:
- context window size
- memory guard limits
- logging settings
- session state/resume settings

## Logging and Session State

When enabled:
- session events are written to `session_logs/*.events.jsonl`
- system metrics are written to `session_logs/*.metrics.jsonl`
- conversation state is saved to `session_state.json`
