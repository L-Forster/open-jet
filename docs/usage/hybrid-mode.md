# Hybrid Mode

Hybrid mode keeps Codex as the primary reasoning and quality agent while a warm local
model performs the high-volume implementation loop. The design targets about 20%
Codex tokens across paired work and at least 99% of Codex-only result quality through
Codex-owned planning and final review.
The footer measures that target from the current session's attributed model usage; it is
not a claim that every task has an identical token split or quality score.

## Start Hybrid

Run `/mode` and choose **Hybrid**. OpenJet uses the selected Codex and local profiles,
validates Codex authentication, and immediately starts and health-checks the local
runtime. It does not wait for the first delegated task to start llama.cpp. The footer
reports the active pair, for example:

```text
HYBRID · gpt-5.6-sol medium + Qwen3.5-27B
```

Run `/model` while Hybrid is active to configure both sides of the pair: Codex model and
effort, followed by local model and reasoning mode. `/effort` changes only the Codex
reasoning effort. Selecting Local
or Codex from `/mode` returns to a single-model mode. The local picker includes saved
profiles, detected GGUF files, and a manual GGUF path, so adding a local model does not
require rebuilding the Hybrid configuration.

## Division of work

Codex owns intent, architecture, risk decisions, decomposition, and independent final
review. It delegates repository exploration, implementation, repetitive edits, tests,
and debugging through `delegate_local`. Each delegation supplies a bounded task and
acceptance criteria. The local worker edits and tests in the same project, then returns
only a concise handoff rather than sending its full transcript back through Codex.

Existing OpenJet tool approvals still apply to the local worker. The delegation tool is
registered only while Hybrid is healthy, and the local agent cannot recursively invoke it.

The transcript inserts `CODEX · <model>` and `LOCAL · <model>` lane headers whenever
control changes models. Local reads, edits, commands, tests, and final handoff stay in the
Local lane; Codex orchestration and review stay in the Codex lane. The same boundary is
written to the standard OpenJet session trace with a shared turn ID and delegation call
ID, including per-model tool activity and token usage without logging prompt contents.

The rotating startup hints advertise `/mode` and explain this division of work.

## Measuring savings

The footer records Codex and local exchanges against their actual model references and
shows the current Hybrid session's Codex share. The 20% figure is an optimization target;
small, ambiguous, or high-risk tasks may appropriately use a larger Codex share.
