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

Read-only helpers:

```bash
open-jet status
open-jet models
open-jet commands
open-jet version
```

## Basic interaction

- Type normally and press Enter to chat
- Use `@file` or `@[path with spaces]` to add file content to context
- Use `@image.png` or paste local image file paths to attach images to the next turn
- Type `/` to open slash-command suggestions
- `Tab`/`Enter` can autocomplete slash commands and file mentions
- `Ctrl+C` or `/exit` quits

For slash command reference, see [slash-commands.md](slash-commands.md).
