# Session State and Logging

When local logging is enabled, each session gets its own directory under `.openjet/state/sessions/YYYY/MM/DD/`.

The session folder contains:

- `session.json`: manifest with session id, install id, runtime context, and collector configuration
- `.openjet/state/session_state.json`: chat/session state if state persistence is enabled

Normal chat turns do not dump full prompt payloads into the session directory. Full runtime prompt dumps are only written in debug mode under `.openjet/state/debug_prompts/`.

Telemetry is no longer persisted by the app itself. `open-jet` emits OTLP to a collector, and the collector decides where logs, traces, and metrics are stored.

The emitted payload is intentionally metadata-only:

- prompt text is not emitted
- tool output, stdout, stderr, file paths, and model paths are not exported
- model identifiers are reduced to safe names
- errors are exported as redacted summaries plus hashes

This keeps telemetry useful for reliability analysis without turning it into a second conversation transcript.

To keep local session metadata bounded:

- `logging.retention_days` prunes old session folders
- `logging.max_sessions` keeps only the newest session directories

For remote telemetry broadcasting, see [Telemetry](../telemetry.md).
