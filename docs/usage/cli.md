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

Project provisioning, for embedding a model in an application you ship:

```bash
openjet project
openjet project --use-case dialogue --target handheld --budget 4 --context-tokens 4096
```

`openjet project` is a build-time step, distinct from `openjet setup`. Setup detects this
machine and picks a coding model for it; `openjet project` asks which device you are
shipping to, how much memory your application will concede to the model, and what the
model is for, then downloads it into `.openjet/models/` in the project.

| Flag | Meaning | Omitted |
|---|---|---|
| `--use-case` | `chat`, `dialogue`, `extract`, `classify`, `summarize` | prompts |
| `--target` | device you ship to, not this machine — `handheld`, `laptop_8`, `laptop_16`, `desktop_gpu_8`, `desktop_gpu_16`, `desktop_gpu_24`, `server` | prompts |
| `--budget` | memory in GB the model may take on the target | the target preset's default |
| `--context-tokens` | context window to size the KV cache for | 4096 |

Pass all four to run it non-interactively from a build script or Dockerfile. Selection
fails loudly rather than silently downgrading: if nothing in the catalog meets the use
case's first-token deadline inside the budget, the command exits with the reason each
candidate was rejected.

Written output, both gitignored:

```text
your-project/.openjet/models/<model>.gguf   # weights, for your build to bundle
your-project/.openjet/config.yaml           # the model pin, overlaid on config.yaml
```

See the [SDK quickstart](../sdk/quickstart.md), [Choosing a model](../models.md), and
[project configuration](../configuration.md#project-configuration-openjetconfigyaml).

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

MCP server helpers:

```bash
openjet mcp list
openjet mcp test <server>
openjet mcp add-stdio <name> -- <command> [args...]
openjet mcp remove <server>
```

MCP is disabled by default. These helpers read layered MCP config from `~/.openjet/mcp.yaml`, `.openjet/mcp.yaml`, and legacy `config.yaml` entries. `add-stdio` and `remove` write the project `.openjet/mcp.yaml`. See [mcp.md](mcp.md).

Skill helpers:

```bash
openjet skill list
openjet skill view <name> [file_path]
openjet skill create <name>
openjet skill validate <name>
openjet skill doctor
```

Skills are loaded from `.openjet/skills`, `.agents/skills`, `~/.openjet/skills`, `~/.agents/skills`, and bundled install skills. `create` writes project skills to `.openjet/skills`. See [skills.md](skills.md).

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
- `Ctrl+C` clears a non-empty input; when the input is already empty, it closes OpenJet
- `Ctrl+V` or `Alt+V` pastes clipboard text or an image
- `Esc` stops the active turn
- Press `Esc` twice within 500 ms to clear the input and pending attachments
- `/exit` or `/quit` closes OpenJet

For slash command reference, see [slash-commands.md](slash-commands.md).
