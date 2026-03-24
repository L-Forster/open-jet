# Workflow Harness

`open-jet` includes a lightweight harness layer to keep agent work structured under constrained context.

It provides:

- modes for `chat`, `code`, `review`, and `debug`
- step-oriented state so the agent can continue across turns
- skill docs and project docs loaded into bounded turn context
- persistent harness state under `.openjet/`

This helps reduce prompt drift and keep limited-context models focused on the active task.

For backend workflow files and CLI execution, see [backend-workflows.md](backend-workflows.md).
