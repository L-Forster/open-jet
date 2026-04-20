# CLI Usage

## Start commands

```bash
openjet
openjet chat
openjet chat Explain the repo layout briefly.
openjet benchmark
openjet benchmark --sweep
```

`openjet chat <text>` runs one prompt through the shared SDK/runtime backend and prints the final response to stdout without launching the TUI.

OpenJet's CLI surface covers three adjacent jobs:

- interactive local agent usage through the chat TUI
- operational helpers like setup, status, models, and workflows
- benchmark entrypoints for `llama-bench`

Optional setup flow:

```bash
openjet setup
```

Setup supports:

- `Local model: llama.cpp (GGUF)`
- `Self-hosted API: OpenAI-compatible`
- `Hosted API: OpenRouter`

Read-only helpers:

```bash
openjet status
openjet models
openjet commands
openjet version
openjet update
```

`openjet update` pulls the latest remote repo commit from the tracked branch.

Benchmark helpers:

```bash
openjet benchmark
openjet benchmark --sweep
openjet benchmark -p 1024 -n 256 -r 3
```

Benchmarking reuses the active model profile from `config.yaml`. See
[benchmarking.md](../benchmarking.md).

Persistent device setup:

```bash
openjet device list
openjet device add <existing_id> <new_id>
openjet device on <id>
openjet device off <id>
```

Run `openjet device list` first. Use the current id shown on the left as `<existing_id>` if you want to rename a device for chat.

Backend workflow commands:

```bash
openjet workflow list
openjet workflow show <name>
openjet workflow run <name>
openjet workflow start <name>
openjet workflow stop <name>
openjet workflow status [name]
openjet workflow logs <name>
openjet workflow assign <name> <device_id>...
```

Workflow files are Markdown files under `workflows/` or `.openjet/workflows/`. See [backend-workflows.md](backend-workflows.md).

## Basic interaction

- Type normally and press Enter to chat
- Use `@file` or `@[path with spaces]` to add file content to context
- Use `@camera0`, `@mic0`, `@gpio0`, or another concrete device id from `openjet device list` or `/device`
- Use `@image.png` or paste local image file paths to attach images to the next turn
- Type `/` to open slash-command suggestions
- `Tab`/`Enter` can autocomplete slash commands and file mentions
- `Ctrl+C` or `/exit` quits

For slash command reference, see [slash-commands.md](slash-commands.md).
