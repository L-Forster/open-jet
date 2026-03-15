# Session State and Logging

When local logging is enabled, each session gets its own directory under `session_logs/YYYY/MM/DD/`.

The session folder contains:

- `session.json`: manifest with session id, install id, runtime context, and collector configuration
- `session_state.json`: chat/session state if state persistence is enabled

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
