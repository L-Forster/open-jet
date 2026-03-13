# Layered Context Benchmarks

## What This Benchmarks

The `open-jet` harness context path is a layered document-admission system in `src/harness.py`, not a generic conversation-memory summarizer.

For each turn it:
- computes a `TurnBudget` from the effective context window, generation/tool/system reserves, available RAM factor, and current persistent context tokens
- builds a structured harness state summary first
- generates ordered candidate docs across layer1, layer2, and layer3
- admits a candidate only if it fits both the remaining global docs budget and the remaining per-layer budget
- skips oversized docs instead of truncating them inside the layered harness path

Persistent memory in `src/persistent_memory.py` is separate. It is merged into the base system prompt and is not part of the layered doc-admission pipeline.

## Commands

Run context-only tests:

```bash
open-jet-bench context-tests
```

Run one named suite:

```bash
open-jet-bench context-suite jetson_4k_baseline
```

Run the Jetson-focused 4k suites:

```bash
open-jet-bench jetson-4k
```

Compare saved runs:

```bash
open-jet-bench compare benchmark_results/context/<run_a> benchmark_results/context/<run_b>
```

Print a concise summary from saved artifacts:

```bash
open-jet-bench summary benchmark_results/context/<run>
```

## Suites

- `jetson_4k_baseline`: healthy 4k coding-task baseline
- `jetson_4k_ram_pressure`: same 4k state under different RAM snapshots
- `jetson_4k_layer_compare`: same 4k state across layer-toggle variants
- `long_debug_session`: multi-turn debug loop with recent verification pressure
- `skill_heavy_code_session`: ranking and survival of skill docs
- `candidate_starvation_case`: ordered starvation where an early admitted doc blocks a later one

## Artifact Files

Each run writes:

- `config.json`: run metadata, layered config, git commit, platform, workspace manifest
- `summary.json`: aggregate metrics and final-context summary
- `turns.jsonl`: one JSON record per turn with budgets, candidate order, admitted docs, skipped reasons, layer usage, and latency
- `summary.md`: concise human-readable run report
- `timeline.txt`: terminal-readable turn-by-turn context evolution
- `compare_ready_metrics.json`: pre-aggregated metrics used by run comparison

## Turn Record Fields

Each `turns.jsonl` record includes:

- `turn_index`
- `title`
- `current_context_tokens`
- `memory_snapshot`
- `turn_budget`
- `state_summary`
- `state_summary_tokens`
- `candidate_labels_in_order`
- `candidate_metadata`
- `admitted_docs`
- `skipped_docs`
- `layer_token_totals`
- `layer_loaded_docs`
- `budget_alerts`
- `latency_ms`

## How To Inspect Evolution

Use `timeline.txt` first when you want a quick terminal view of:
- usable budget and docs budget by turn
- layer usage vs layer budgets
- admitted docs
- skipped docs and skip reasons
- budget alerts
- doc-set deltas from the previous turn

Use `summary.md` for a compact report and `compare_ready_metrics.json` plus `open-jet-bench compare` for side-by-side comparisons.

## Limitations

- These benchmarks measure harness context assembly only; they do not run model inference.
- Token counts are estimated with the configured tokenizer.
- Candidate ordering and admission behavior are intentional implementation details, so benchmark output reflects that exact code path rather than an abstract planner.
