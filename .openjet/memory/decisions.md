# decisions

Append-only architectural decisions.

## 2026-03-12

- Use a two-layer harness: markdown for editable behavior, Python for hard enforcement.
- Keep one core agent and specialize by mode instead of building a heavy multi-agent graph.
- Budget context from remaining prompt headroom and available RAM, not static token caps alone.
