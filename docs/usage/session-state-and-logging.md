# Session State and Logging

When enabled:

- session events are written to `session_logs/*.events.jsonl`
- system metrics are written to `session_logs/*.metrics.jsonl`
- conversation state is saved to `session_state.json`

The event log captures replayable traces for:

- tool call success rate
- approval and denial decisions
- interrupted generation and resumed sessions
- time-to-resolution
- token usage for successful tasks
- hallucinated or low-value command proposals
- hardware/runtime-specific failure analysis

Use this data to evaluate reliability from real traces, not only subjective testing.
