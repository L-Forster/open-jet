# Slipstream Mode

Slipstream pairs a frontier cloud model with a warm local model so you get
frontier-level results for a fraction of the frontier tokens. The orchestrator owns the thinking
that decides whether the work is any good — intent, architecture, risk, decomposition,
and independent final review. The local model does the high-volume work that consumes
most of the tokens in practice: reading the repository, editing files, running tests,
and iterating on failures.

The orchestrator can be **OpenAI Codex** (subscription OAuth) or **OpenRouter** (API key),
including free models such as **Ox Alpha** (`openrouter/stealth/ox-alpha`).

The name is the aviation one. A trailing aircraft flying in the leader's wake burns far
less fuel to hold the same speed. Slipstream does the same thing with tokens: you stay
at the frontier model's altitude while your local model absorbs the drag.

## What it costs

Slipstream targets a **20% Codex token share** across paired work, which is roughly
**5x more work from the same Codex plan**. The target lives in code as
`TARGET_CODEX_SHARE` in `src/hybrid.py`, and the footer measures your actual share from
the current session's attributed model usage.

Two honest caveats:

- 20% is an optimization target, not a guarantee. Small, ambiguous, or high-risk tasks
  appropriately use a larger Codex share, and the footer will show that.
- The quality goal is at least 99% of Codex-only result quality, achieved by keeping
  planning and final review with Codex. That is a design target measured across paired
  work, not a per-task promise.

The local side is not free either — it costs you GPU time and memory rather than plan
quota. Slipstream trades a resource you own for one you meter.

## Start Slipstream

Run `/mode` and choose **Slipstream** (`/mode slipstream` also works, and `/mode hybrid`
remains accepted). OpenJet uses the selected orchestrator and local profiles, validates
cloud authentication when needed, and immediately starts and health-checks the local runtime. It does not
wait for the first delegated task to start llama.cpp. The footer reports the active pair:

```text
SLIPSTREAM · stealth/ox-alpha + Qwen3.5-27B · Orchestrator 18%
```

Run `/model` while Slipstream is active to configure both sides of the pair: Codex or OpenRouter
model (and Codex effort when Codex is the orchestrator), followed by local model and reasoning mode.
`/effort` changes only the Codex reasoning effort. Selecting Local or Codex from `/mode` returns to a single-model mode.
The local picker includes saved profiles, detected GGUF files, and a manual GGUF path, so
adding a local model does not require rebuilding the Slipstream configuration.

Slipstream needs a Codex or OpenRouter orchestrator profile and a saved local model
profile. Run `/connect openrouter` or `/connect openai-codex`, then `/setup` if the local model is missing.

## Division of work

Codex or OpenRouter owns intent, architecture, risk decisions, decomposition, and independent final
review. It delegates repository exploration, implementation, repetitive edits, tests,
and debugging through `delegate_local`. Each delegation supplies a bounded task and
acceptance criteria. The local worker edits and tests in the same project, then returns
only a concise handoff rather than sending its full transcript back through the orchestrator.

That handoff is where the saving actually comes from. The worker's exploration, failed
attempts, and test output never enter the orchestrator context — only the conclusion does.

Existing OpenJet tool approvals still apply to the local worker. The delegation tool is
registered only while Slipstream is healthy, and the local agent cannot recursively
invoke it.

The transcript inserts `CODEX · <model>` and `LOCAL · <model>` lane headers whenever
control changes models. Local reads, edits, commands, tests, and final handoff stay in
the Local lane; Codex orchestration and review stay in the Codex lane. The same boundary
is written to the standard OpenJet session trace with a shared turn ID and delegation
call ID, including per-model tool activity and token usage without logging prompt
contents.

## Measuring savings

The footer records Codex and local exchanges against their actual model references and
shows the current session's Codex share alongside the tokens the local model absorbed:

```text
412K tokens saved by local model · Codex 18%
```

Watch that share over a few sessions rather than a single task. If it sits well above
20%, the work is likely dominated by decisions rather than implementation, which is a
signal about the task rather than a fault in the mode.

## Naming note

`slipstream` is the user-facing name. The stored configuration value, the `agentMode`
field in the TUI protocol, and the internal identifiers remain `hybrid`, so existing
`config.yaml` files and any pinned frontend keep working unchanged.
