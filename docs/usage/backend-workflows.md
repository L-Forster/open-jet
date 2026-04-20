# Backend Workflows

OpenJet supports backend workflows defined as Markdown files.

Writing good agent constraints is hard. The community shares optimized `.openjet` workflow files for specific languages and frameworks in the Discord.

These workflows reuse the same device backend as the chat TUI:

- device discovery and aliases from `devices.md`
- device tools like camera, microphone, GPIO, and sensor reads
- observation payload storage under `.openjet/state/observations/`
- the normal OpenJet runtime and harness context

## Quickstart

Create a workflow file under `workflows/`:

```md
---
name: smoke
mode: chat
interval_seconds: 60
devices:
  - gpio0
allow_shell: false
---
Inspect the bound GPIO device when relevant and return a short status summary.
If no useful reading is available, say so clearly.
```

List discovered workflows:

```bash
openjet workflow list
```

Run a workflow once:

```bash
openjet workflow run smoke
```

Start it as a background runner:

```bash
openjet workflow start smoke
```

Check status and logs:

```bash
openjet workflow status smoke
openjet workflow logs smoke
```

Stop it immediately:

```bash
openjet workflow stop smoke
```

## Markdown Contract

Workflow files are plain Markdown with optional YAML frontmatter.

Supported frontmatter fields:

- `name`: workflow id; defaults to the filename stem
- `mode`: `chat`, `code`, `review`, or `debug`
- `devices`: optional list of device ids
- `interval_seconds`: polling interval for background runs
- `allow_shell`: allow shell tool access for this workflow
- `require_plan`: keep the workflow in read-only planning mode until a plan is produced and approved
- `require_verification`: require post-edit verification handling for workflow runs; defaults to the mode-specific runtime behavior when omitted
- `skills`: optional list of harness skills
- `files`: optional list of files to preload into workflow context

The Markdown body is the workflow instruction document.

## File Locations

OpenJet discovers workflows from:

- `workflows/*.md`
- `.openjet/workflows/*.md`

Local overrides in `.openjet/workflows/` win over repo workflows with the same workflow name.

OpenJet stores workflow runtime state under:

- `.openjet/state/workflows/<name>/pid.json`
- `.openjet/state/workflows/<name>/status.json`
- `.openjet/state/workflows/<name>/last-run.md`
- `.openjet/state/workflows/<name>/runs/*.md`

## Devices and Context

Workflows do not preload device logs or payload files into the prompt.

Instead, OpenJet gives the agent:

- the workflow Markdown document
- the absolute path to `devices.md`
- the bound device ids for that run

The agent can then open `devices.md` and call device tools only when needed.

## Device Binding

Workflows can bind devices three ways, in this precedence order:

1. CLI override:

```bash
openjet workflow run smoke --device gpio0
openjet workflow start smoke --device camera0
```

2. Saved local assignment:

```bash
openjet workflow assign smoke gpio0 camera0
```

3. `devices:` in the workflow frontmatter

Use `openjet device list` first so you do not have to guess ids.

## Tool Policy

By default, workflows get:

- device tools
- read-only file/context tools

They do not get write/edit tools by default.

Shell is available only when `allow_shell: true`.

## Stop Semantics

`openjet workflow stop <name>` is intended to stop the background runner immediately.

The stop path terminates the detached workflow process and, if needed, force-kills it so state does not remain stuck in `running`.

## Typical Flow

1. Add or rename devices with `openjet device ...`
2. Create `workflows/<name>.md`
3. Run `openjet workflow list`
4. Test with `openjet workflow run <name>`
5. Start background mode with `openjet workflow start <name>`
6. Inspect `status`, `logs`, and `last-run.md`
