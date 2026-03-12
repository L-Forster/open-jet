# Open-Jet Benchmark Judge Instructions

Judge whether the agent successfully completed the benchmark task in the provided eval environment.

## Scoring

- Return `passed: true` only if the task intent is satisfied in the final filesystem and final response.
- Return `score` from `0.0` to `1.0`.
- Prefer final filesystem state and tool trace over intermediate intent.
- Penalize misuse of tools that violates the task or environment constraints.
- Use deterministic checks as strong evidence, but you may adjust slightly if the full trace clearly justifies it.

## Required Output Fields

- `score`
- `passed`
- `reasoning_summary`
- `key_evidence`
- `missed_expectations`
