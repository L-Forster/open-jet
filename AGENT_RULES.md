# Rules for AI agents working on OpenJet

Read this before changing anything. [AGENTS.md](AGENTS.md) covers the layout and commands; this file covers how to work.
Each rule comes from a mistake an agent made here.

## Git

- Do not commit, push, rebase, cherry-pick or amend unless the maintainer asks for that step. Finish with the changes in
  the working tree and say what is ready. If a step creates commits by itself (a rebase), ask before running it.
- No AI attribution in commits or pull requests: no `Co-Authored-By: Claude ...`, no "Generated with ..." lines, even
  when your tooling adds them by default.
- One pull request does one thing. Don't mix a refactor, a feature and a docs rewrite.

## Accuracy comes first

- No speedup that lowers output quality, however small. An int8-activation prefill (+15 % prefill, KL 0.0014 -> 0.0019)
  was rejected and removed. Scheduling, batching and kernel efficiency are fine; approximations are not.
- The KV cache floor is q8_0. Do not propose q4 KV.
- Prove it: a refactor gives byte-identical greedy output and an identical `--kld` report; a kernel change reports top-token
  agreement and KL divergence against llama.cpp generating token by token, next to llama.cpp's own batched noise floor
  (commands under Correctness in [megakernel/README.md](megakernel/README.md)).
- Sampled runs use the model card's settings. Never run Qwen3.8 with thinking on at temperature 0: it loops.

## Claims and measurements

- Never present a guess as a fact. Label estimates as estimates, or leave them out. Search before writing "nobody has
  published X".
- Benchmark only the current code: `git fetch`, compare with `origin/master`, rebuild if the library is older than the last
  engine commit.
- Compare engines on the same machine, same model file, same prompts, same sampling, with runs interleaved and the GPU
  otherwise idle (a game or video on the same GPU moved results by 50 %). Record the GPU state with every sample.
- Quote OpenJet's full range (78 / 107 / 140 tok/s on an RTX 3090, with the workload), list every condition that differs,
  and show whole benchmark tables, not chosen rows. If conditions differ, the answer is "unknown until a same-machine run".
- Do not name small competing projects in the repo's docs or README. Comparisons with them are internal.

## The machine you run on

- Ask before using a GPU (benchmarks, test agents, requests to a running `mk_server`). Say what will run and for how long,
  and wait for a clear yes. Never send test traffic to someone's live server.
- Memory: build the engine one target at a time, under a hard cap so the build dies instead of the machine:
  `systemd-run --user --scope -p MemoryMax=12G -p MemorySwapMax=0 make -j1 ...`. Check `free -m` first. If a build hits
  the cap, report it; do not raise the cap and retry. Build only the library the model needs (`libqwen38mk.so` for Q4_K_M).
- While a model loads or serves, run nothing else that reads a model file, compiles or downloads. On a 15 GB WSL machine
  this combination killed the VM and lost hours of results.
- Agent runs that auto-approve tools can crawl the disk (`find /` walks the Windows drive under WSL). Check `ps` for orphans
  after any agent run.
- Outputs (scripts, results, recordings) go in the repo next to their kind, e.g. `megakernel/bench/results/<date>-<name>/`,
  never in `/tmp` (WSL restarts wipe it). Model files and other large binaries stay out of git.

## Scope

- "Plan" means a written plan: no code, no files, until asked to implement.
- Before building something from a spec, say in a sentence or two whether its premise holds and why it might not.
- Fix bugs in the repo's source, never in local config (`config.yaml`, `.openjet/`): config edits hide the bug and fix
  nothing for users.
- When the maintainer changes one setting in your proposal, change only that one; everything else stays as they had it.
- Don't start new packages or projects that weren't asked for.

## Code

- Match the surrounding code: its naming, comment density and idioms. Python 3.10.
- No defensive `try/except` or `getattr` in internal code; catch only where an error is turned into a message for the user.
- Don't duplicate an engine source file to add a variant; use a compile-time switch in the same file.

## Talking to the maintainer

- Answer in a few plain sentences, recommendation first. No walls of headers and bullets, no drafts nobody asked for.
- Shell commands on one line.
- When a regression is reported, believe it: name the exact target or setting involved and fix it. Don't argue, and don't
  rerun checks to prove it wrong.
- When the maintainer is discussing an idea, reply in text and run nothing until told to.
