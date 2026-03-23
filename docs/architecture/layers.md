# Layered Project Structure

`open-jet` is easiest to maintain when each major concern has one owner and each user surface composes those owners instead of embedding their logic.

## Target Sections

- `src/context/`: context budgets, harness state, multimodal user content, persistent memory, and prompt shaping.
- `src/runtime/`: runtime registry, protocol adaptation, and transport clients.
- `src/tools/`: tool dispatch plus filesystem and shell helpers.
- `src/peripherals/`: hardware discovery, shared observation types, and device adapters for cameras, audio, GPIO, buses, and sensors.
- `src/observation/`: saved payloads, basic input processing, and agent-facing conversion of frames, speech events, and text buffers.
- `src/surfaces/`: user-facing entry points such as TUI and CLI launchers.
- `src/sdk.py`: programmatic session API that composes `context`, `runtime`, and `tools` without depending on the TUI.
- `docs/`: operational and user documentation.
- `tests/`: layer-focused tests, plus cross-layer integration tests where needed.

## Dependency Rules

- `context` may depend on `runtime` token budgeting and protocol-neutral message shapes, but not on TUI or CLI code.
- `tools` must not depend on TUI or CLI code.
- `runtime` must not depend on harness state or UI concerns.
- `surfaces` compose the lower layers and may be missing independently.
- `sdk` should depend only on the lower layers needed to run a session.

## Failure Model

Each section should degrade independently:

- Without `context`, sessions still run but lose automatic context loading or condensation.
- Without `tools`, chat still works but tool calls should be denied or surfaced as unavailable.
- Without `surfaces.tui`, non-TUI CLI commands and the SDK should still import and run.
- Without `sdk`, the CLI and TUI should still function.

## Current Incremental Split

This repo now exposes facade packages for `context`, `runtime`, `tools`, and `surfaces`, and the CLI entry point is isolated in `src/cli.py` with lazy TUI loading. That keeps non-TUI commands from importing the full TUI stack.

The next step is to move concrete modules behind those facades in small batches, starting with:

1. move TUI-only helpers out of `src/app.py`
2. move slash-command orchestration under `src/surfaces/`
3. move harness and memory helpers under `src/context/`
4. move runtime clients under `src/runtime/`
5. introduce `src/peripherals/` for edge-device input flows described in [Edge Inputs Phase 1](edge-inputs-phase-1.md)
6. keep lightweight processing in `src/observation/` before adding workflows or background schedulers
