# CLI Usage

## Start commands

```bash
open-jet
open-jet chat
open-jet chat Explain the repo layout briefly.
open-jet benchmark
open-jet benchmark --sweep
```

`open-jet chat <text>` runs one prompt through the shared SDK/runtime backend and prints the final response to stdout without launching the TUI.

OpenJet's CLI surface covers three adjacent jobs:

- interactive local agent usage through the chat TUI
- operational helpers like setup, status, models, and workflows
- benchmark entrypoints for `llama-bench`

Optional setup flow:

```bash
open-jet --setup
open-jet setup
```

Setup supports:

- `Local model: llama.cpp (GGUF)`
- `Self-hosted API: OpenAI-compatible`
- `Hosted API: OpenRouter`

Read-only helpers:

```bash
open-jet status
open-jet models
open-jet commands
open-jet version
open-jet update
```

`open-jet update` pulls the latest remote repo commit from the tracked branch.

Benchmark helpers:

```bash
open-jet benchmark
open-jet benchmark --sweep
open-jet benchmark -p 1024 -n 256 -r 3
```

Benchmarking reuses the active model profile from `config.yaml`. See
[benchmarking.md](../benchmarking.md).

Persistent device setup:

```bash
open-jet device list
open-jet device add <existing_id> <new_id>
open-jet device on <id>
open-jet device off <id>
```

Run `open-jet device list` first. Use the current id shown on the left as `<existing_id>` if you want to rename a device for chat.

Backend workflow commands:

```bash
open-jet workflow list
open-jet workflow show <name>
open-jet workflow run <name>
open-jet workflow start <name>
open-jet workflow stop <name>
open-jet workflow status [name]
open-jet workflow logs <name>
open-jet workflow assign <name> <device_id>...
```

Workflow files are Markdown files under `workflows/` or `.openjet/workflows/`. See [backend-workflows.md](backend-workflows.md).

## Basic interaction

- Type normally and press Enter to chat
- Use `@file` or `@[path with spaces]` to add file content to context
- Use `@camera0`, `@mic0`, `@gpio0`, or another concrete device id from `open-jet device list` or `/device`
- Use `@image.png` or paste local image file paths to attach images to the next turn
- Type `/` to open slash-command suggestions
- `Tab`/`Enter` can autocomplete slash commands and file mentions
- `Ctrl+C` or `/exit` quits

For slash command reference, see [slash-commands.md](slash-commands.md).
