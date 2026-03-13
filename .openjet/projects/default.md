Project defaults for `open-jet`.

Context:
- offline-first terminal app for local LLM workflows
- targets edge Linux devices including Jetson-class hardware
- must remain usable on constrained RAM and small context windows

Codebase priorities:
- local-first behavior
- bounded context loading
- safe tool use
- low-overhead runtime behavior
- fast, focused verification over heavyweight flows

Important areas:
- `src/app.py`: TUI orchestration and turn loop
- `src/agent.py`: agent loop and context condensation
- `src/harness.py`: multi-turn harness, budgeting, doc loading
- `src/runtime_limits.py`: token and RAM budgeting
- `src/executor.py`: tool execution layer
- `src/session_state.py`: saved session state

When editing:
- preserve simple startup flow
- preserve approval gates for risky tools
- avoid introducing heavyweight dependencies without a strong reason
