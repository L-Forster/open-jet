# CLI Usage

## Start commands

```bash
open-jet
open-jet chat
```

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

Persistent device setup:

```bash
open-jet device list
open-jet device add <existing_id> <new_id>
open-jet device on <id>
open-jet device off <id>
```

Run `open-jet device list` first. Use the current id shown on the left as `<existing_id>` if you want to rename a device for chat.

## Basic interaction

- Type normally and press Enter to chat
- Use `@file` or `@[path with spaces]` to add file content to context
- Use `@camera0`, `@mic0`, `@gpio0`, or another concrete device id from `open-jet device list` or `/device`
- Use `@image.png` or paste local image file paths to attach images to the next turn
- Type `/` to open slash-command suggestions
- `Tab`/`Enter` can autocomplete slash commands and file mentions
- `Ctrl+C` or `/exit` quits

For slash command reference, see [slash-commands.md](slash-commands.md).
