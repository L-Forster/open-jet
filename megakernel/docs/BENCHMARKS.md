# Benchmarks

Every number here was measured on one machine: an RTX 3090 (24 GB, 480 W limit), a Ryzen 9 5900XT and 64 GB of RAM,
Windows 11 with WSL2, CUDA 13.2, with unsloth's `Qwen3.8-27B-UD-Q4_K_XL.gguf` unless a section says otherwise. Each
result carries its date and the commit it was measured on.

Bit-level results are tied to the GPU's SM count: the kernel runs one block per SM and some sums are split by the grid
(lm_head's per-block softmax sums; without exact reuse, the prefill attention's last-wave split, `a2_plan`). A GPU
with another SM count (the 3090 has 82, an RTX 4090 128) can differ in the last bits and, at near-ties, in greedy
tokens, so its replies are not expected to match these runs token for token.

How to read them:
- **Sessions.** The same binary runs several percent faster or slower from one session to the next with GPU temperature
  and clocks (up to ~11% between sessions on 2026-10-12). Absolute numbers are only compared within one session;
  changes made after a full run are measured A-B-A (old build, new build, old build again, in one session), never added
  onto older absolutes. The comparison with llama.cpp and the full run are one session (2026-11-06); llama.cpp's
  column in the parallel agents' table is from 2026-11-02 and carries its date.
- **Rates.** Decode "tok/s" is generated tokens over decode time, as the server reports it (`timings`). The
  short-prompt rows are means of the per-request rates, as the client reports them; token-weighted rates (total tokens
  over total time) are given next to them.
- **VRAM** is `nvidia-smi`'s memory used on the whole card, other processes included, unless a row says otherwise.
  GiB = MiB / 1,024.
- **Other engines** are only compared where they ran on this machine: llama.cpp b12024 (tuned) for speed,
  llama.cpp b10246 as the accuracy reference, and upstream open-jet at 72ba4a7.

Contents: [headline numbers](#headline-numbers) - [against llama.cpp, 2026-11-06](#against-llamacpp-2026-11-06) -
[full run 2026-11-06](#full-run-2026-11-06) - [at a 250 W limit](#at-a-250-w-limit) -
[parallel agents](#parallel-agents) - [KV cache presets](#kv-cache-presets) -
[Q4_K_M and upstream's scripts](#q4_k_m-and-upstreams-scripts) - [exactness](#exactness) -
[the conversation cache](#the-conversation-cache) - [keep-warm](#keep-warm) - [images and video](#images-and-video) -
[wide verify](#wide-verify-2026-10-13) - [earlier full runs](#earlier-full-runs) - [reproducing](#reproducing).

## Headline numbers

The latest result in each area. The links lead to the details, and the previous full run is at the
[end of the file](#earlier-full-runs).

**Against llama.cpp** ([2026-11-06](#against-llamacpp-2026-11-06)): one session, UD-Q4_K_XL, the server at 120k context,
the same client, prompts and model file. This engine: the release v2026.11.05 (3ed5ce5) at its default 7 drafts, with
the q8_0 KV cache and with preset F. llama.cpp: b12024's `llama-server`, tuned, q8_0 KV cache, at 4 MTP drafts, its
fastest.

| | llama.cpp, 4 drafts | this engine, q8_0 | this engine, preset F |
|---|---|---|---|
| greedy decode, short prompts (4 x 512 tokens) | 87.7 tok/s | 166.5 tok/s | 160.7 tok/s |
| sampled decode, thinking on (8 runs) | 73.2 tok/s | 146.9 tok/s | 147.4 tok/s |
| decode at 38k / 101.6k context | 66.2 / 55.2 tok/s | 127.8 / 116.4 tok/s | 111.3 / 123.2 tok/s |
| read a 101.6k-token prompt from scratch | 95.6 s | 68.5 s | 70.7 s |
| first token, short prompt | 117 ms | 61 ms | 61 ms |
| follow-up turn in an 11.8k-token chat, prompt time | 248 ms | 169 ms | 178 ms |
| 282 benchmark questions: right, total time | 229, 2,534 s | 230, 1,563 s | 229, 1,596 s |
| VRAM, idle server at 120k context | 22.9 GiB | 21.2 GiB | 19.8 GiB |

Preset F's replies differ from q8_0's, and with them the share of accepted drafts, so its decode rows are not
like-for-like with the q8_0 column ([full run](#full-run-2026-11-06)).

Without drafts on either side (q8_0): llama.cpp 43.8 against 48.7 tok/s on the short prompts, 36.3 against 45.2 at 38k
and 28.0 against 40.7 at 101.6k context ([without drafts](#without-drafts)).

**Everything else:**

| | result | measured | details |
|---|---|---|---|
| 4 parallel agents (`--agents 4`, Q4_K_M, q8_0 KV, sampled) | 423.7 tok/s in total on short prompts, 360.0 with a ~23k-token context each; llama.cpp b12024's `llama-server --parallel 4` on 2026-11-02: 137.2 and 96.8 | 2026-11-06 | [parallel agents](#parallel-agents) |
| q8_0 at a 250 W limit | greedy 130.3 tok/s on short prompts, decode at 38k 94.5, the 282 questions 230 right in 2,237 s; 2.33 J per token on short prompts against 3.47 at 480 W and llama.cpp's 6.96 | 2026-11-06 | [at a 250 W limit](#at-a-250-w-limit) |
| upstream open-jet 72ba4a7, its own scripts (Q4_K_M, 4 drafts each) | writing code 163.8 against 132.6 tok/s; with 30k tokens of code in context 118.1 against 86.9; prefill 1,878 against 1,528 tok/s | 2026-10-25 | [Q4_K_M and upstream's scripts](#q4_k_m-and-upstreams-scripts) |
| accuracy against llama.cpp generating token by token | top token agrees on 98.737% of positions, mean KL 0.00119 nats (llama.cpp's own batched run: 98.489%, 0.00166) | 2026-10-17, unchanged through v2026.11.05 | [against llama.cpp](#against-llamacpp) |
| KV presets D / F: KL at 100k context | 0.00226 / 0.00218, against q8_0's 0.00217 | 2026-10-25 | [KV cache presets](#kv-cache-presets) |
| the same formats without the 8-bit ring: KL at 100k context | 0.00254 / 0.00281 (4-bit K and V: 0.00469) | 2026-11-02 | [without the ring](#without-the-ring) |
| the full 262,144-token context, preset F | a 260,896-token prompt in 279 s, decode at that depth 77.8 tok/s, 23,589 MiB on the card | 2026-11-06 | [the full context](#the-full-context) |
| greedy token ids at 0, 4 and 7 drafts, and under every draft policy | identical | each release (`scripts/gate.sh`) | [speculation](#speculation) |
| the 282 questions at no drafts against drafted | this engine at 7 drafts: 0 differ. llama.cpp b12024 at 4 drafts: 113 differ, at 7: 113 (4 against 7: 9) | 2026-11-06 | [speculation](#speculation) |
| Windows build against Linux build | identical output | each release (by hand: `scripts/exactness.py --compare`) | [Windows and Linux](#windows-and-linux) |
| reused, parked and restored requests (exact reuse) | bit for bit equal to the same request sent fresh, q8_0, D and F, up to ~255k. Second turns on 20 conversations against a fresh server: 0 differ; llama.cpp b12024: 20 | 2026-11-06 | [exact reuse](#exact-reuse) |

## Against llama.cpp, 2026-11-06

One session on 2026-11-06: llama.cpp b12024's `llama-server` (ddf48f85c, built from source on Windows with CUDA 12.8
for sm_86) at 4 MTP drafts and at 0, then this engine, the release v2026.11.05 (3ed5ce5, its Linux build on WSL2 as
`scripts/serve.sh` starts it) at 0 drafts on q8_0 and at its default 7 drafts on q8_0, preset D and preset F. All at
120,000 tokens of context, driven by [bench/suite](../bench/suite/README.md)'s clients with the same prompts and model
file; llama.cpp with a q8_0 KV cache and the tuned command in
[bench/suite](../bench/suite/README.md#2-start-a-server). Preset D's column is in the [full run](#full-run-2026-11-06).

| | llama.cpp, 4 drafts | this engine, q8_0 | this engine, preset F |
|---|---|---|---|
| greedy decode, short prompts (4 x 512 tokens), mean of request rates | 87.7 tok/s | 166.5 | 160.7 |
| the same, token-weighted | 81.1 | 141.4 | 135.6 |
| sampled decode, thinking on (8 runs), mean of request rates | 73.2 tok/s | 146.9 | 147.4 |
| the same, token-weighted | 68.2 | 140.2 | 139.1 |
| decode at 11.6k context (200 tokens) | 73.3 tok/s | 148.6 | 133.9 |
| decode at 38k context (256 tokens) | 66.2 tok/s | 127.8 | 111.3 |
| decode at 101.6k context (256 tokens) | 55.2 tok/s | 116.4 | 123.2 |
| read an 11.6k-token prompt | 7.61 s | 5.77 s | 5.85 s |
| read a 38k-token prompt | 27.6 s | 22.0 s | 22.7 s |
| read a 101.6k-token prompt from scratch | 95.6 s | 68.5 s | 70.7 s |
| first token, short prompt | 117 ms | 61 ms | 61 ms |
| follow-up turn in an 11.8k-token chat, prompt time | 248 ms | 169 ms | 178 ms |
| edit an earlier message in an 11.7k-token chat, prompt time | 302 ms | 69 ms | 71 ms |
| edit the end of a 9.1k-token system prompt, prompt time | 882 ms | 82 ms | 93 ms |
| VRAM, whole card, server idle | 22.9 GiB | 21.2 GiB | 19.8 GiB |
| 282 benchmark questions: right, total time | 229, 2,534 s | 230, 1,563 s | 229, 1,596 s |

The four short prompts one by one (greedy; llama.cpp's and q8_0's replies have the same length within a token, so those
columns time the same amount of work; preset F's replies differ, and its story stops at 505 tokens):

| prompt (reply tokens) | llama.cpp, 4 drafts | this engine, q8_0 | this engine, preset F | q8_0 against llama.cpp |
|---|---|---|---|---|
| a Python function that checks whether a number is prime, with a docstring and doctests (273) | 112.4 tok/s | 247.7 | 246.7 | 2.20x |
| how a hash map works internally, for a junior developer (512) | 86.6 tok/s | 154.2 | 150.2 | 1.78x |
| a ~300-word short story about a lighthouse keeper (510; q8_0 511) | 61.8 tok/s | 96.2 | 93.3 | 1.56x |
| a step-by-step plan for migrating a Flask REST API to FastAPI (512) | 90.1 tok/s | 167.9 | 152.5 | 1.86x |

Code drafts best and prose worst, for both engines: llama.cpp accepted 213 of 236 drafted tokens on the prime check at 4
drafts and 306 of 811 on the story; this engine on q8_0 at 7 drafts accepted 80% of 292 and 34% of 838.

### Without drafts

The same clients with no drafting on either side (llama.cpp without `--spec-type`, this engine with `MK_DRAFT=0`, both
on q8_0):

| | llama.cpp | this engine | this engine ahead |
|---|---|---|---|
| greedy decode, short prompts, mean of request rates | 43.8 tok/s | 48.7 | 11.3% |
| sampled decode, thinking on, mean of request rates | 43.5 tok/s | 48.5 | 11.3% |
| decode at 11.6k / 38k / 101.6k context | 41.1 / 36.3 / 28.0 tok/s | 47.4 / 45.2 / 40.7 | 15% / 24% / 45% |
| VRAM, whole card, server idle | 21.2 GiB | 21.2 GiB | |

Per prompt, greedy: llama.cpp 43.7 to 43.9 tok/s, this engine 48.6 to 48.9 (11.1% to 11.4% ahead). llama.cpp's greedy
story without drafts was 460 tokens long, against 510 at 4 drafts ([speculation](#speculation)); this engine's was 511
at 0 and at 7.

Notes:
- **Draft depth.** 4 drafts is llama.cpp's fastest setting here, as it was on b11548. Measured on this build in one
  session on 2026-10-31: 2, 5, 6 and 7 drafts were slower than 4 on every decode row above (greedy short prompts 80.0,
  82.1, 76.7 and 73.4 against 86.8 tok/s) except 2 drafts' token-weighted sampled rate (69.8 against 67.8); 2 drafts was
  also faster on the story prompt alone (66.9 against 61.4 tok/s, greedy). At 7 drafts llama.cpp decoded 15% to 24%
  slower than at 4 and read prompts within 2% of it. Its greedy answers also change with the depth
  ([speculation](#speculation)).
- **llama.cpp against its 2026-11-03 numbers**, the same binary and model file: at 4 drafts the decode rows 0.9% to 1.9%
  faster, the prompts within 1.3% and the question time 1.4% shorter, with all 282 answers byte-identical; at 0 drafts
  the decode rows 0.6% to 1.0% faster. The GPU ran about 6 C cooler at the same points of the run, so read small
  differences from that run against these. Two short prompt-time rows moved more: the follow-up turn 239 -> 248 ms and
  the message edit 247 -> 302 ms (at 0 drafts 227 -> 265 ms).
- **Against b11548**, the previous llama.cpp column (the best of three sessions on 2026-10-12 / 13; in this file's
  history at d034e3a): b12024 at 4 drafts (2026-10-31) decoded 2% to 11% faster, read prompts 5% to 9% faster and
  answered the follow-up turn in 222 instead of 323 ms. On the four greedy prompts through a simpler client, an A-B-A in
  one session gave b11548 85.2, b12024 89.5, b11548 84.9 tok/s.
- **The questions:** llama.cpp at 4 drafts got 229 (GSM8K 50 / 50, MMLU-Pro 45 / 70, ARC-C 47 / 50, MATH-500 28 / 40 and
  9 / 10 with thinking on, HumanEval 38 / 41, NIAH 9 / 9, LongBench v2 3 / 12), in 2,534 s. This engine's answers per
  set: [the full run](#full-run-2026-11-06).
- **Sampling:** the sampled requests set temperature 1.0, top_p 0.95, top_k 20 and a seed, not `min_p`, so each server
  used its own default: this engine 0, llama-server llama.cpp's 0.05 (b10246's `common.h`; the GGUF sets no
  `general.sampling.min_p`). llama-server's `preserve_thinking: true` is what Qwen3.8's template does when the key is
  absent, as on this server: both keep earlier turns' thinking.
- **Conditions:** fans pinned at 100% (they read 93-98% at the start and end of each speed pass); GPU 44-69 C and SM
  clock 1,860-2,025 MHz at those points; memory clock 9,851 MHz. llama.cpp ran natively on Windows, this engine under
  WSL2. No pass was repeated.

## Full run 2026-11-06

The release v2026.11.05 (3ed5ce5), its Linux build on WSL2: the server at 120,000 tokens of context as
`scripts/serve.sh` starts it (wide verify, 7 drafts, exact reuse, side requests and parked conversations on, prefill
chunk 1,024), driven by [bench/suite](../bench/suite/README.md) in one session. The comparison with llama.cpp is in
[against llama.cpp, 2026-11-06](#against-llamacpp-2026-11-06).

| | q8_0 KV | preset D | preset F |
|---|---|---|---|
| greedy decode, short prompts (4 x 512 tokens), mean of request rates | 166.5 tok/s | 165.9 | 160.7 |
| the same, token-weighted | 141.4 | 141.1 | 135.6 |
| sampled decode, thinking on (8 runs), mean of request rates | 146.9 tok/s | 146.4 | 147.4 |
| the same, token-weighted | 140.2 | 139.8 | 139.1 |
| decode at 11.6k context (200 tokens) | 148.6 tok/s | 147.5 | 133.9 |
| decode at 38k context (256 tokens) | 127.8 tok/s | 111.6 | 111.3 |
| decode at 101.6k context (256 tokens) | 116.4 tok/s | 110.6 | 123.2 |
| read an 11.6k-token prompt | 5.77 s | 5.86 s | 5.85 s |
| read a 38k-token prompt | 22.0 s | 22.8 s | 22.7 s |
| read a 101.6k-token prompt from scratch | 68.5 s | 71.1 s | 70.7 s |
| first token, short prompt | 61 ms | 62 ms | 61 ms |
| follow-up turn in an 11.8k-token chat, prompt time | 169 ms | 177 ms | 178 ms |
| edit an earlier message in an 11.7k-token chat, prompt time | 69 ms | 70 ms | 71 ms |
| edit the end of a 9.1k-token system prompt, prompt time | 82 ms | 93 ms | 93 ms |
| VRAM, whole card, server idle | 21.2 GiB | 20.1 GiB | 19.8 GiB |
| 282 benchmark questions: right, total time | 230, 1,563 s | 230, 1,600 s | 229, 1,596 s |

Notes:
- **The follow-up turn** re-reads the previous reply because exact reuse is on; `MK_EXACT_REUSE=0` appends to it
  instead (~80 ms on 2026-10-14, but not bit-identical to a fresh request).
- **The 101.6k-token prompt from scratch** ran on a separate server with side requests and parking off
  (`MK_SIDE_MAX=0 MK_PARK_MIB=0`): with them on, the client's cache-clearing request runs beside the cached
  conversation instead of evicting it, and the "from scratch" request would be served from the cache.
- **The presets' decode rows** are not like-for-like with q8_0's: preset D's replies differ from q8_0's from 2,048
  tokens on, preset F's from the start (its rotated K changes the window's rows too), and with the replies the share of
  accepted drafts changes. The time per engine cycle compares the kernels directly (reply time / (reply tokens / tokens
  per cycle), from the server's console):

  | context | q8_0: tokens per cycle, ms per cycle | preset D | preset F | D / F, ms per cycle against q8_0 |
  |---|---|---|---|---|
  | 11.6k (200 tokens) | 3.75, 25.3 | 3.83, 26.0 | 3.37, 25.1 | +2.8% / -0.8% |
  | 38k (256 tokens) | 3.53, 27.6 | 2.98, 26.7 | 3.06, 27.5 | -3.3% / -0.4% |
  | 101.6k (256 tokens) | 3.61, 31.0 | 3.40, 30.7 | 3.81, 31.0 | -1.0% / 0.0% |

  One reply each, timed to 0.01 s, so each derived value is good to about 0.5%. On the short prompts preset D's replies
  and cycles were q8_0's; preset F's cycles were within 1% of them, and most of its lower greedy rate there comes from
  accepting fewer drafts (3.80 instead of 4.16 tokens per cycle on one of the four prompts).
- **The questions**, per set: each configuration's 282 answers are byte-identical to its answers on 2026-11-03.
  q8_0 and preset D: GSM8K 50 / 50, MMLU-Pro 47 / 70, ARC-C 47 / 50, MATH-500 29 / 40 (and 9 / 10 with thinking on),
  HumanEval 37 / 41, NIAH 9 / 9, LongBench v2 2 / 12. Preset F differs in the same three sets as on 2026-10-25
  ([preset F](#preset-f)): MMLU-Pro 46, MATH-500 30, HumanEval 36. This client gives LongBench v2 answers 128 tokens,
  which cut 8 of its 12 answers on 2026-10-14; the 262,144-token runs give them 1,024. The total time includes the side
  requests and parks the questions cause beside the cached long conversation.
- **VRAM:** the Windows desktop held 627 MiB of the card in this session, against 581 MiB on 2026-11-03. The engine's own
  memory (q8_0 at 120k, idle) was about 21,040 MiB in both, so the higher whole-card figures come from the desktop.
- **Conditions:** fans pinned at 100% (they read 93-98%); GPU 51-65 C and SM clock 1,965-2,010 MHz (`nvidia-smi` at the
  start and end of each speed pass); memory clock 9,851 MHz.
- **Against the previous full run** ([2026-11-03, 605f82a](#2026-11-03-605f82a)), with llama.cpp's own 0.9% to 1.9% gain
  between the two sessions in mind: decode past the presets' 8-bit window moved most, preset D +3.2% at 38k and +5.5% at
  101.6k, preset F +2.5% and +5.5%, with the time per cycle 2.5% to 5.2% lower; preset F at 262,144 tokens decodes 11.6%
  faster and reads its prompt 10.6% faster ([the full context](#the-full-context)). q8_0's decode rows are +0.7% to +0.9%,
  no more than the session's. Prompt reads gain with their length on every cache: -1.4% to -1.7% at 11.6k, -3.0% to
  -3.7% at 38k, -5.6% to -6.0% at 101.6k. The questions: 1,607 -> 1,563 s (q8_0), 1,644 -> 1,600 s (D), 1,647 -> 1,596 s
  (F) with the same answers. The 4 agents, on q8_0, stayed level: 424.0 -> 423.7 and 366.0 -> 360.0 tok/s, sampled, so
  not the same replies. Each change in between was measured on its own ([below](#changes-since-the-2026-10-20-run)).

### How the run was measured

- **Client:** [bench/suite](../bench/suite/README.md)'s `speed_client.py` against `/v1/chat/completions` without
  streaming: the requests of the 2026-10-12 / 13 llama.cpp sessions plus a clock warm-up. It reads the server's
  `timings` (prompt and decode time) and `usage` (cached tokens). One warm-up request, then ~3 s of decode, so the
  timed requests start at full clocks.
- **Short prompts:** four chat prompts (a Python prime check with doctests, how a hash map works, a 300-word story, a
  Flask-to-FastAPI migration plan): greedy without thinking, 512 tokens each; then sampled with thinking (temperature
  1.0, top_p 0.95, top_k 20) with seeds 11 and 22, up to 768 tokens each. First token: three one-word questions,
  `max_tokens` 1.
- **Long context:** an 11.6k-token excerpt of a 38k-token document with a summary request (greedy, thinking on, 200
  tokens) and a follow-up question; the whole 38k-token document, then a 101.6k-token one, each with a 256-token answer
  (greedy, no thinking). In the main pass the 101.6k prompt reuses the 38k one's prefix; the from-scratch row is the
  separate server described above.
- **Edits:** a three-turn chat over 30k characters of the document, then turn 2's user message edited; a 9.1k-token
  system prompt, then its last word changed. The row is the prompt time of the edited request.
- **Questions:** 282 items, greedy, zero-shot, the same prompts for both engines, thinking off except for the 10
  MATH-500 items with thinking; scored offline. The llama.cpp answers come from the same client in the same session.

### Changes since the 2026-10-20 run

Each change to the GPU code was measured against the build before it, the two builds alternating in one session, with
the CLI under Linux / WSL2 (UD-Q4_K_XL, q8_0 KV cache, 7 drafts unless noted). None of them changed the greedy token
ids or the accuracy report.

| change | effect | measured |
|---|---|---|
| preset F's own 5-bit prefill kernels (fad007f) | F's prompt time -3.4% at 101.6k, -1.5% at 38k (exact mode), its KL report byte-identical; the other caches' kernels unchanged | 2026-10-25 |
| parallel slots (`--agents`, f09ef59) | one conversation decodes 0.6% slower (UD-Q4_K_XL), 1.3% slower (Q4_K_M) | 2026-10-28 |
| compile trim (752368c): code no served setting reaches is no longer compiled; lm_head's penalties and output type became runtime switches | ms per cycle -0.28% on short prompts, unchanged at 38k and with 4 slots, ~+0.2% with a presence / frequency penalty (estimated); ptxas peak ~36 -> 20.2 GiB, build 28.5 -> 13 minutes | 2026-10-29 |
| the megakernel's stack frame 2,016 -> 816 bytes (df0d418) | 144 MiB less local memory reserved; speed unchanged at 38k | 2026-10-30 |
| local memory and spills, frame 816 -> 752 bytes (6f3b257) | ms per cycle -1.55% at 0 drafts, -0.91% at 38k, -1.19% at 101.6k; 4 slots +7.4% tok/s | 2026-10-30 |
| decode and prefill speedups from Nsight runs, frame 752 -> 704 bytes (03daaa9) | against v2026.11.01: +5.5% tok/s on short prompts, +5.7% at 40.5k, the same tokens | 2026-11-03 |
| attention kernels for prefill (`pk_attn9`) and for decode on low-bit K / V (3ed5ce5) | sustained, against fbb8efb: preset F's ms per cycle -11.2% at 258k with the default draft gates, -7.5% at 258k with 7 drafts, -4.1% at 101k; prefill against `MK_PFX=64`: F -2.3 / -6.5 / -9.9% and D -2.8 / -6.5 / -10.3% at 40k / 101k / 258k, q8_0 -2.4 / -5.0% at 40k / 101k; the same tokens and saved states | 2026-11-05 |
| resid's K2 combine interleaved, the wv phase tops' dealing table, the projection wrappers' loop state rebuilt after each tile call, the D / F T = 8 attention schedule, lm_head<8>'s staged drafts, prefill attention (row sums, copies, exponentials) and two small prefill kernels, PTX-model fences (exp-w6, 6f6a3de) | against 1ecb507, interleaved per-barrier minima: ms per cycle -1.43% at 7 drafts on short prompts, -1.14% at 101k (q8_0), -1.95% at 101k (D), -2.19% at 258k (F); sustained at 101k +1.0% tok/s; prefill attention -4.4%, 101k prompt time -1.2%; the same tokens, KL report and saved states | 2026-11-09 |

### At a 250 W limit

The release v2026.11.05 (3ed5ce5) with the settings and client of the full run, one session on 2026-11-06, with the card's
power limit set to 250 W (`nvidia-smi -pl 250`, as Windows admin; this card allows 100 to 480 W, default 350 W). The 480 W
column is a speed and from-scratch pass at the end of the same session, with the question time from the full run.
llama.cpp b12024 ran in the same session at 480 W only.

| | q8_0, 250 W | q8_0, 480 W | 250 W against 480 W | llama.cpp, 4 drafts, 480 W |
|---|---|---|---|---|
| greedy decode, short prompts (4 x 512 tokens), mean of request rates | 130.3 tok/s | 165.7 | -21% | 84.8 |
| the same, token-weighted | 110.8 | 140.9 | -21% | 78.5 |
| sampled decode, thinking on (8 runs), mean of request rates | 113.7 tok/s | 146.4 | -22% | 71.4 |
| the same, token-weighted | 108.7 | 139.8 | -22% | 66.7 |
| decode at 11.6k context (200 tokens) | 113.1 tok/s | 148.2 | -24% | 71.3 |
| decode at 38k context (256 tokens) | 94.5 tok/s | 127.6 | -26% | 65.3 |
| decode at 101.6k context (256 tokens) | 81.9 tok/s | 115.7 | -29% | 54.2 |
| read an 11.6k-token prompt | 9.04 s | 5.77 s | +57% | 7.99 s |
| read a 38k-token prompt | 33.8 s | 22.1 s | +52% | 27.9 s |
| read a 101.6k-token prompt from scratch | 105.8 s | 68.6 s | +54% | 96.6 s |
| first token, short prompt | 77 ms | 61 ms | +25% | 142 ms |
| follow-up turn in an 11.8k-token chat, prompt time | 233 ms | 170 ms | +37% | 279 ms |
| edit an earlier message in an 11.7k-token chat, prompt time | 85 ms | 70 ms | +21% | 340 ms |
| edit the end of a 9.1k-token system prompt, prompt time | 105 ms | 85 ms | +24% | 882 ms |
| VRAM, whole card, server idle | 21.1 GiB | 21.1 GiB | | 22.8 GiB |
| 282 benchmark questions: right, total time | 230, 2,237 s | 230, 1,563 s | +43% | 229, 2,554 s |

Power and energy (`nvidia-smi` once a second, whole card, the desktop's idle draw included; energy = mean power x time):

| | q8_0, 250 W | q8_0, 480 W | llama.cpp, 4 drafts, 480 W |
|---|---|---|---|
| short prompts, greedy and sampled: tokens | 7,724 | 7,724 | 7,231 |
| the same: mean power | 249.6 W | 476.7 W | 463.0 W |
| the same: mean SM clock | 816 MHz | 1,962 MHz | 1,889 MHz |
| the same: energy per token | 2.33 J | 3.47 J | 6.96 J |
| the 101.6k-token read: mean power | 249.4 W | 445.5 W | 446.0 W |
| the same: mean SM clock | 1,126 MHz | 1,764 MHz | 1,854 MHz |
| the same: rate | 960 tok/s | 1,481 tok/s | 1,052 tok/s |
| the same: energy per prompt token | 0.260 J | 0.301 J | 0.424 J |
| the 282 questions: mean power | 249.2 W | | 457.9 W |
| the same: energy | 557.5 kJ | | 1,169.4 kJ |
| the same: energy per reply token | 4.31 J | | 9.03 J |

- **Output:** all 282 answers at 250 W are byte-identical to the 480 W run's, and to the same engine's at no drafts.
- **The session did not drift:** the 480 W pass at its end matched the full run's q8_0 column within 0.5% on decode at short
  prompts and 38k (0.6% at 101.6k), and the 101.6k-token read took 68.6 against 68.5 s. llama.cpp at no drafts matched its
  column within 0.8%; at 4 drafts it read 1.4 to 3.3% slower than in the full run, with all 282 answers byte-identical, so
  the full run's llama.cpp column is the safer comparison for decode.
- **Prompt reads are the one place llama.cpp at 480 W beats this engine at 250 W.** It also uses 1.6x the energy for it.
- **The limit held:** every sample in the capped passes read an enforced limit of 250 W; the highest power reading was
  256.1 W.
- **Clocks:** under load the memory clock stayed at 9,851 MHz, so the limit takes the SM clock: 58% lower on short-prompt
  decode for 21% less speed, 36% lower on the 101.6k-token read for 35% less.
- **Conditions:** fans at 98-99%; GPU 26-39 C through the capped passes, 34-70 C for llama.cpp; the desktop held about
  520 MiB of the card.

## Parallel agents

`--agents N` serves up to four requests at once, each in a parallel slot of the engine with its own context; every slot
that is decoding runs in the same batched launch ([SERVER.md](SERVER.md#several-agents-at-once---agents)). Measured
with [bench/suite/parallel_client.py](../bench/suite/README.md) (the prompts, sampling, context text and measurements
of open-jet's `bench/parallel_bench.py`) on unsloth's plain `Qwen3.8-27B-Q4_K_M.gguf`, in the full run's session on
2026-11-06 (WSL2, v2026.11.05, `--agents 4 -d 7 --kv q8`, q8_0 KV cache, exact mode):

| test | result |
|---|---|
| 4 agents, short coding prompts: total decode (better of 2 runs) | 423.7 tok/s |
| 4 agents, each with its own ~23k-token context: total decode | 360.0 tok/s |
| the same: the slowest first token | 0.52 s |
| a ~20k-token prompt sent while another request decodes: its first token | 13.1 s |
| the other request meanwhile | 14.1 tok/s |
| 1 agent, short prompts (better of 2 runs) | 194.8 tok/s |
| 1 agent, ~23k-token context | 173.2 tok/s |
| VRAM at ready, whole card | 22,597 MiB |

- A total adds the requests' decode rates (512 tokens each, temperature 0.6; one of the four ~23k-token replies stopped
  at 504). The two short passes gave 423.7 and 413.2 tok/s with 4 agents, 194.8 and 194.1 with one.
- One agent's short-prompt rate follows how many sampled drafts it accepts: twelve passes on 2026-10-30 (v2026.10.30)
  ran 167 to 204 tok/s (median 187) at the same 26.2 to 26.9 ms per engine cycle, with 4.4 to 5.5 tokens per cycle.
- The "other request" row counts that request's streamed chunks while the long prompt is read; it varies between
  sessions (10.8 to 14.3 tok/s since 2026-10-28).
- Each slot's greedy reply is the one the request gets alone, also when its prompt arrives while others decode; seeded
  sampling beside other slots keeps its distribution, not its tokens
  ([SERVER.md](SERVER.md#several-agents-at-once---agents)).

### Against open-jet's server

Open-jet's `bench/parallel_bench.py` (unchanged) against this server and against open-jet's own server (f246ff0) running
the same engine library: Q4_K_M, `--agents 4 -d 7`, one session on an RTX 3090 with open-jet's server first and last,
2026-10-28. A total adds the four requests' decode rates (512 tokens each, temperature 0.6).

| | open-jet's server | this server | this server, `MK_EXACT=0` | open-jet's server again |
|---|---|---|---|---|
| 4 short requests at once, total tok/s (2 runs) | 288 / 350 | 328 / 354 | 341 / 357 | 291 / 344 |
| 4 requests with ~23k-token contexts, total tok/s | 286 | 305 | 302 | 304 |
| a 20k-token prompt sent while another request decodes: its first token after | 15.7 s | 15.8 s | 15.8 s | 15.7 s |
| the other request meanwhile | 11.5 tok/s | 12.9 tok/s | 12.8 tok/s | 10.6 tok/s |
| 1 short request alone, tok/s (2 runs) | 193 / 184 | 190 / 201 | 185 / 205 | 163 / 187 |
| 1 request with a ~23k-token context alone, tok/s | 161 | 203 | 158 | 173 |

The decoding is the engine's either way, so the totals differ by run-to-run noise (one request alone with a long
context varies most). Exact mode (`MK_EXACT=1`, the `--agents` default) costs nothing measurable here. Feeding a
waiting prompt whole instead (`MK_PF_MIX=0 MK_PF_CHUNK=0`; another session that day, open-jet's server 15.7 s and
11.5 / 10.7 tok/s there) got the 20k-token prompt's first token after 11.0 s, while the other request made 0.6 tok/s
meanwhile.

### Against llama.cpp's server

The same client against llama.cpp b12024's `llama-server --parallel 4` (the tuned command of
[bench/suite](../bench/suite/README.md#2-start-a-server) with `-np 4` and `-c 100000`: 25,000 positions per slot; 4 MTP
drafts, q8_0 KV cache, natively on Windows), then this server (v2026.10.30, `--agents 4 -d 7 --kv q8`: 33,280 positions
per slot, WSL2), back to back in one session on 2026-11-02, unsloth's plain `Qwen3.8-27B-Q4_K_M.gguf`. llama-server was
not run again for v2026.11.05 (2026-11-06, the table above), so the last two columns compare across sessions:

| | llama-server --parallel 4 (2026-11-02) | this server, v2026.10.30 (2026-11-02) | this server, v2026.11.05 (2026-11-06) | v2026.11.05 against llama-server |
|---|---|---|---|---|
| 4 short requests at once, total tok/s (better of 2 runs) | 137.2 | 381.6 | 423.7 | 3.09x |
| 4 requests with ~23k-token contexts, total tok/s | 96.8 | 315.4 | 360.0 | 3.72x |
| the same: the slowest first token | 0.61 s | 0.54 s | 0.52 s | |
| a ~20k-token prompt sent while another request decodes: its first token | 16.4 s | 13.7 s | 13.1 s | |
| the other request meanwhile | 2.3 tok/s | 14.3 tok/s | 14.1 tok/s | |
| 1 short request alone, tok/s (better of 2 runs) | 61.1 | 190.8 | 194.8 | 3.19x |
| 1 request with a ~23k-token context alone, tok/s | 61.0 | 159.7 | 173.2 | 2.84x |
| VRAM at ready, whole card | 24,014 MiB | 22,500 MiB | 22,597 MiB | |

- A total adds the requests' decode rates (512 tokens each, sampled at temperature 0.6, top_p 0.95, top_k 20).
  llama.cpp's per-request token counts come from its stream's `timings.predicted_n`, as for this server.
- **Context:** with `-np 4` and MTP drafting, llama-server does not load at `-c 120000` on this card: its MTP draft
  context's KV cache (472 MiB) fails to allocate. The ctx test's prompts are 23,316 to 23,323 tokens, plus 512
  generated.
- **Checked after the run, the same session:** llama-server's own timings match the client's (60.3 / 61.9 tok/s against
  60.5 / 62.1). One request decodes at the same ~61 tok/s with `-np 4 -c 64000` (1.4 GB less VRAM in use) and with
  `-np 1 -c 100000`, and 4 at once give 131 tok/s at `-c 64000` against 126 / 132 at `-c 100000`. Its process's shared
  GPU memory stays at 280 to 640 MiB in every configuration (pinned host buffers), so the `-c 100000` run did not page
  into system memory. Its SM clock read 780 to 1,245 MHz right after its runs (this server: 1,950 to 1,965 MHz, but
  its keep-warm load holds the clocks up for 120 s after a request, `MK_KEEP_WARM_S`, so the readings do not compare
  like for like).
- The gaps are larger than single-stream ([against llama.cpp](#against-llamacpp-2026-11-06)) because this client
  samples: llama.cpp's sampled decode with MTP drafts is its slower path.
- v2026.10.30's own pass earlier in the 2026-11-02 session, before llama-server's: 371.0 / 331.0 / 183.0 / 179.0 tok/s.
- v2026.11.05 against v2026.10.30's column, across the two sessions: +11% and +14% with 4 agents, +2% and +8% alone.

## KV cache presets

`MK_KV` stores K and V below 8 bits, and `MK_KV_WIN=2048` keeps the most recent 2,048 positions in an 8-bit ring next
to the low-bit cache (the presets C, D, D', E and F and their syntax:
[CONFIGURATION.md](CONFIGURATION.md#kv-cache-presets)). While a whole sequence fits in the window, every attention read
comes from the 8-bit ring, so preset D's output is q8_0's: its short-prompt accuracy report is byte-identical to
q8_0's. Preset F stores K rotated, which changes the ring's rows too.

Accuracy against llama.cpp (the reference in [against llama.cpp](#against-llamacpp)), top-token agreement / mean KL in
nats. The fp16 cache, more precise than q8_0, marks the floor of the measurement:

| KV cache | short prompts (13,299 positions) | ~32k (4,094) | 64k (4,095) | 100k (4,095) |
|---|---|---|---|---|
| fp16 (2026-10-12 / 10-13) | 98.774% / 0.00113 | 99.096% / 0.00226 | 0.00161 | |
| q8_0 | 98.737% / 0.00119 (2026-10-17, 7322fbd 10-20) | 99.121% / 0.00219 (10-17) | 99.267% / 0.00163 (10-15) | 99.121% / 0.00217 (10-25, 0278d09) |
| D: `K*=6,V*=5` + window | = q8_0 (2026-10-20, 7322fbd) | 98.950% / 0.00236 (10-20, 7322fbd) | 99.267% / 0.00161 (2026-11-06) | 99.121% / 0.00226 (10-25, 0278d09) |
| D': D + 8-bit MTP layer | = q8_0 (10-15, 7d67c5d) | 0.00236 (10-15) | 0.00161 (10-15) | |
| F: `K*=5r,V*=5` + window | 98.714% / 0.00113 (2026-10-25, 7322fbd) | 98.974% / 0.00242 (10-25, 7322fbd) | 99.341% / 0.00163 (10-25, 7322fbd) | 99.170% / 0.00218 (10-25, 0278d09) |
| C: `V*=5` + window | = q8_0 (10-13) | 99.072% / 0.00228 (10-13) | 99.316% / 0.00159 (10-13) | |
| 4-bit everywhere: `*=4`, no window | 0.00547 (10-15) | | | |

On v2026.11.05 (2026-11-06) every q8_0, D and F cell above, and their [100k and 160k rows](#100k-and-160k-context), read as
listed. The one older cell, D at 64k, moved from 0.00160 to 0.00161 and has the new date.

- The MTP layer's cache only feeds the drafts, so D and D' score the same wherever both were measured.
- The fp16 cache was no better than q8_0: on its 2026-10-12 / 13 builds q8_0 read 0.00113 on the short set and 0.00219
  at 32k (the reference itself was generated with a q8_0 cache).
- The short set's values moved by up to 0.00006 between builds as the kernels changed, which is why each cell carries
  its date. The 32k set's noise is about 4% either way.
- **Memory:** the KV cache at 120k / 262,144 context takes 3.25 / 6.97 GB with preset D and 2.99 / 6.39 GB with F
  (2026-10-25). Whole-card VRAM, speed and the 282 questions under each preset: the
  [full run](#full-run-2026-11-06).

### C, D and E side by side

One session on 2026-10-13 (1ed60cd, 4 drafts, A-B-A against q8_0; KL against llama.cpp's reference, relative to
q8_0's own). Memory is compared at 120k, the launchers' default context: the third column is how much context fits in
the VRAM q8_0 uses at 120k, not how far a preset was tested.

| Preset | Frees at 120k | Context that fits in q8_0's 120k VRAM | KL: short / 32k / 64k | Decode, ms per cycle at 38k / 101.6k | Prefill speed at 38k / 101.6k |
|---|---|---|---|---|---|
| C | 691 MiB | 144k | same / +4% / +2% | +1.2% / +3.7% | -1.3% / -2.7% |
| D | 1,132 MiB | 165k | same / +1% / +5% | +2.9% / +5.0% | -3.3% / -6.7% |
| E | 1,381 MiB | 180k | +2% / +5% / +10% | +2.4% / +4.9% | -2.0% / -4.2% |

D' is D with the MTP layer's cache at 8 bits (~70 MB more; the drafts read q8_0 K / V): at 7 drafts it decoded 1.2% /
2.9% faster than D at 101.6k / 38k with the same output ids and 32k KL as D (2026-10-13, 9dede7f).

### Preset F

K at 5 bits, stored Hadamard-rotated, and V at 5 bits, with the 2,048 window (`MK_KV='K*=5r,V*=5' MK_KV_WIN=2048`). In
the rounding emulator, rotated 5-bit K under the window read at the fp16 cache's floor (+4% at 32k, -2% at 64k). With
the real kernels (2026-10-25, 7322fbd):

- **Short prompts** fit in the window, but F's rotated K changes the ring's rows too, so its short-set KL (0.00113) is
  not q8_0's; the difference is within the 0.00006 the set moves between builds.
- **The 282 questions:** 229 right against q8_0's and D's 230 (2026-10-25, the 2026-10-20 client and settings; the
  2026-10-30, 2026-11-03 and 2026-11-06 runs gave the same 229, set for set). The answers match per set except three
  items: one more
  MATH-500 problem right, one MMLU-Pro answer cut by its 1,024-token cap before it named a choice, and one HumanEval task
  wrong. F's answers match D's word for word on 167 of the 282 (all of ARC-C and the needles, 36 of 41 HumanEval tasks):
  with rotated K even a short prompt's rows differ from q8_0's, so longer answers take other words on the way.
- **A larger window did not help:** at 32k, F read 0.00242 / 0.00239 / 0.00245 / 0.00240 with `MK_KV_WIN` 2048 / 2688
  / 3328 / 4608 (rings of 114 / 137 / 159 / 204 MiB), all within the set's noise. Its excess over q8_0 sits in the
  older 5-bit rows.
- **Prompt speed:** F reads prompts as fast as D: a 101.6k-token prompt in 76.97 s against D's 77.40 s (2026-10-25,
  fad007f, one session).

### The full context

The model's full 262,144-token context with presets D and F, under WSL2 and natively on Windows (where the desktop
shares the card):

| | D, WSL2 (2026-10-20, 7322fbd) | F, WSL2 (2026-11-06, 3ed5ce5) | F, native Windows (2026-10-25) | D, native Windows, 512-token chunk (2026-10-24) |
|---|---|---|---|---|
| read a 260,896-token prompt from scratch | 326 s | 279 s | 340 s | 366 s |
| decode at that depth | 66 tok/s | 77.8 tok/s | 66.5 tok/s | 64.3 tok/s |
| follow-up turn, prompt time | 1.28 s | 1.07 s | 1.3 s | 1.6 s |
| VRAM, whole card | 24,080 MiB after the prompt | 23,589 MiB after the prompt | peak 23,622 of 24,576 MiB | peak 23,771 of 24,576 MiB |
| needle in a haystack (4 needles, asks one), at 190k and at the longest length that fits | 4 / 4 | 4 / 4 | found at 190k | found at 190k |
| LongBench v2 at that length (6 questions, 1,024 answer tokens) | 4 / 6 | 3 / 6 | | |

- **Natively on Windows** the default settings (1,024-token prefill chunk, the draft head's Q4_K copy) leave preset D
  just past what the Windows driver keeps resident at this length, and the engine pages: a 442-token reply decoded at
  16 tok/s against 82.5 under WSL2 (the same tokens and drafts). A 512-token chunk with the draft head on lm_head
  (0.17 GB less, the same output) keeps it resident: the same reply at 81.6 tok/s, and the last column. Preset F stays
  resident with the defaults, so `scripts\serve.bat` uses F above 240k unless `MK_KV` is set.
- **Against 2026-11-03** (605f82a, the same client): F read the prompt in 312 s and decoded at 69.7 tok/s, with a 1.23 s
  follow-up; the same answers to all 10 questions. llama.cpp's same binary gained 0.9% to 1.9% on its decode rows between
  the two sessions.
- **LongBench:** the item F missed was still reasoning toward the right choice when it reached the 1,024-token answer
  cap; D answered it in 658 tokens.

### 100k and 160k context

The 32k / 64k measurement, further out (2026-10-25, the v2026.10.25 Linux package, 0278d09): llama.cpp b10246 with its
q8_0 cache reads one sequence and stores its last 4,096 positions (`tools/llama_ref --chunk N --max-chunks 1
--score-from N-4096 -b 512`), and the engine's `mixed` path is scored against that. Two texts: `pf101k.txt` (code, the
text of the 32k / 64k sets) and War and Peace (`pg2600.txt` from `bench/suite/fetch.py`, line ends converted to LF, from
its first byte). Prose has flatter next-token distributions than code, so it reads higher KL at any cache precision,
q8_0 included: compare within a row. Top-token agreement / mean KL / perplexity:

| text, tokens (scored positions) | q8_0 | D | F |
|---|---|---|---|
| code, 100,352 (96,256-100,351); reference perplexity 1.6250 | 99.121% / 0.00217 / 1.6226 | 99.121% / 0.00226 / 1.6210 | 99.170% / 0.00218 / 1.6230 |
| War and Peace, 100,352 (96,256-100,351); 9.4888 | 96.996% / 0.00311 / 9.4830 | 96.874% / 0.00317 / 9.4884 | 96.825% / 0.00317 / 9.4898 |
| War and Peace, 163,840 (159,744-163,839); 7.7330 | 96.361% / 0.00467 / 7.7244 | 96.581% / 0.00468 / 7.7282 | 96.459% / 0.00472 / 7.7329 |
| the same 4,096 tokens behind 96,256 (the novel from token 63,488); 7.7474 | 96.606% / 0.00336 / 7.7318 | 96.825% / 0.00335 / 7.7339 | 96.850% / 0.00344 / 7.7296 |

- **D and F read q8_0's KL at every depth here**, within the 32k set's ~4% noise; the largest gap is D's +4% at 100k
  on code.
- **The KL grows with depth for every cache, q8_0 included:** the same 4,096 tokens read 0.0034 behind ~100k of context
  and 0.0047 behind ~160k for q8_0, D and F alike (the last two rows), so the growth is between this engine's arithmetic
  and llama.cpp's, not in the cache formats. For scale, llama.cpp's own distributions for those tokens differ by
  0.00864 between the two depths. The disagreements stay near-ties: at 160k the median lead of llama.cpp's token over
  the engine's is 0.09 nats, and 1 of the 140-149 disagreements exceeds 0.5 nats for each cache.
- **160k** is about as far as llama.cpp's q8_0 cache fits on a 24 GB card to make the reference (34 KB per token).
  This engine's q8_0 cache held the 163,904 positions in 6.06 GB (the `--kld` CLI under WSL2: no server, no draft
  head).

### Without the ring

The presets' formats with no 8-bit ring (`MK_KV` without `MK_KV_WIN`) on the real kernels: the v2026.10.25 Linux package
(the build of the 100k / 160k rows above), the same references and scoring, 2026-11-02. Mean KL in nats, lower is
better, with the excess over q8_0's in the same row:

| set | q8_0 | `K*=6,V*=5`, no ring | `K*=5r,V*=5`, no ring | `K*=4,V*=4`, no ring |
|---|---|---|---|---|
| ~32k, code | 0.00219 | 0.00258 (+18%) | 0.00302 (+38%) | 0.00525 (2.4x) |
| 64k, code | 0.00163 | 0.00183 (+12%) | 0.00197 (+21%) | 0.00334 (2.0x) |
| 100k, code | 0.00217 | 0.00254 (+17%) | 0.00281 (+29%) | 0.00469 (2.2x) |
| 100k, War and Peace | 0.00311 | 0.00342 (+10%) | 0.00376 (+21%) | 0.00602 (1.9x) |
| 160k, War and Peace | 0.00467 | 0.00496 (+6%) | 0.00515 (+10%) | 0.00740 (1.6x) |

- With the 2,048 ring the same formats read q8_0's numbers from 64k out (presets D and F above): the ring is what makes
  5-bit K usable.
- Continuity: preset F with the ring at ~32k on this package gives the published report byte for byte (0.00242), and
  4-bit K and V at ~32k / 64k equal the 2026-10-15 build's 0.00525 / 0.00334.

### The window size and rotation

Why the presets use a 2,048-position window and, except for F's K, unrotated rows. All rows below come from the
rounding emulator (2026-10-12 / 13), which stores the cache in each format and is scored like the real kernels; it
keeps exactly W recent positions at 8 bits, where the kernels round the window up to 640-position steps
([CONFIGURATION.md](CONFIGURATION.md#choosing-the-window-and-rotation)). KL is relative to q8_0's on the same build
(0.00113 on the short prompts, 0.00219 at 32k); the fp16 cache reads +0% / +3%, and the 32k set's noise is about 4%
either way. The low-bit loss of K and V sits almost entirely in the most recent ~0.5-2k positions.

The window. The short prompts are ~1k-token sequences, so any window of 1k or more holds all of them, and only the 32k
set can tell 1k from 2k:

| low-bit sides | short prompts: no window / 512 | 32k: no window / 512 / 1,024 / 2,048 |
|---|---|---|
| V at 5 bits, rotated | +27% / +6% | +16% / +5.5% / +4.6% / 0% |
| K at 6 bits, rotated | +12% / +2.7% | +4% / +7% / - / +3% |
| K at 6 and V at 5 bits, rotated | +42% / - | - / - / +6% / +2.7% |
| K at 6 and V at 5 bits (preset D) | - / - | - / - / - / +1.4% |

Attention sinks instead of a window (the first 4 positions kept at 8 bits) did nothing: K at 6 bits, rotated, read +11%
on the short prompts against +12% without.

Rotation helps without a window (on the short prompts, K at 6 bits read +23% plain and +12% rotated, V at 5 bits +40%
and +27%), but not under it: with the 2,048 window, K at 6 and V at 5 bits read +1.4% plain and +2.7% rotated at 32k,
the same within the set's noise, so preset D drops the rotation (no rotation of the queries, no rotating back of the
attention output). Other formats tried in the emulator (3-bit, codebooks, per-row scales, random signs, angle coding)
scored worse than the plain per-32-value grid.

## Q4_K_M and upstream's scripts

unsloth's plain `Qwen3.8-27B-Q4_K_M.gguf` runs on the same engine. Its last full run was on 2026-10-14 (b3be928), in one
session with UD-Q4_K_XL (q8_0 KV, 7 drafts, 120k context, WSL2):

| | UD-Q4_K_XL | Q4_K_M |
|---|---|---|
| greedy decode, short prompts | 144.7 tok/s | 148.1 tok/s |
| decode at 11.6k / 38k / 101.6k context | 128.6 / 109.9 / 102.0 | 145.7 / 109.4 / 106.0 |
| read a 101.6k-token prompt from scratch | 79.0 s | 77.6 s |
| VRAM, whole card | 21.1 GB | 20.7 GB |
| 282 questions right | 230 | 231 |

The two files disagreed on 15 of the 282 answers and had the same perplexity. Accuracy against llama.cpp on Q4_K_M:
98.887% / 0.00127 (a wide-verify build with up to 4 drafts, 2026-10-13; llama.cpp batched on that file: 98.489% /
0.00184).

Upstream open-jet's own benchmark scripts at its master 72ba4a7 (`bench/code_bench.py`, `bench/think_bench.py`: their
megakernel halves, best of 2, greedy code runs, the thinking runs at temperature 1.0 with seeds 1000 / 1001), on
Q4_K_M, in one session on 2026-10-25 (WSL2): upstream's engine (`make qwen38_mk` with its default flags, 4 drafts, the
CLI's default KV cache) between two passes of this engine (7322fbd, the release's GPU code, q8_0 cache). The table gives
the lower of this engine's two passes, which agreed within 0.7%:

| test | open-jet 72ba4a7, 4 drafts | this engine, 4 drafts | this engine, 7 drafts |
|---|---|---|---|
| writing code (no context) | 132.6 tok/s | 163.8 tok/s | 184.9 tok/s |
| 30k tokens of code in context | 86.9 | 118.1 | 126.9 |
| short prompts (mean of 4) | 117.1 | 140.2 | 157.2 |
| reasoning mode (4 prompts x 2 seeds, 1,024 tokens) | 102.5 | 119.4 | 130.1 |
| prefill, 29.6k-token prompt | 1,528 tok/s | 1,878 tok/s | 1,877 tok/s |

Draft acceptance in reasoning mode: open-jet 55.6%, this engine 61.1% with 4 drafts and 51.8% with 7. Upstream has
since merged some of this engine's kernels (open-jet #24 to #27), so the comparison is with 72ba4a7 as it was.
`code_bench.py` takes its context prompt from the checkout's own `qwen38_mk.cu`, `prefill.cuh` and `prefill_gemm.cuh`,
which the device / host split rewrote, so a later run of the script is not comparable with these rows either.

## Exactness

### Speculation

- **Greedy output does not depend on the draft depth.** The attention's split-K partition depends only on the context
  length, and the verify GEMMs compute every row the same way whatever the number of rows, so the same prompt gives the
  same token ids at 0, 4 and 7 drafts. Every release is checked this way (v2026.10.30: the Windows build's ids at 0 and
  7 drafts equal the Linux build's at 7). On v2026.11.05 (2026-11-06):
  - the 282 questions at no drafts against 7 drafts: 0 answers differ, 129,511 tokens on both sides, 41 answers past 1,000
    tokens (not in the gate: about an hour);
  - about 2,000 tokens of greedy output (1,874 to 2,048) at no drafts against 7, on q8_0 and on presets D and F past their
    window: identical
    (`scripts/exactness.py`, the long check);
  - the same essay under five draft policies, from no gates to tight gates: identical to no drafts (the policy check).
- **llama.cpp's greedy output depends on drafting.** Its greedy answers to the 282 questions (b12024, 2026-11-06) differ
  from its own at no drafts on 113 at 4 drafts and 113 at 7; 72 of the 113 at 4 drafts are under 1,000 tokens. It repeats
  itself: a fresh no-draft server gave the first 120 questions' answers back byte for byte, the 73 of them that differ from
  4 drafts included, and two sessions of 4 drafts agree on all 282. Quality doesn't visibly move: 231 right at no drafts,
  229 at 4 and at 7, and 6 of the 113 at 4 drafts flipped right or wrong (4 for no drafts, 2 for 4 drafts). With no
  drafts every token goes through the single-token kernels and with drafts most go through the batched ones (its code picks
  kernels by batch size), the likely reason the split is so much wider than between two draft counts. Between those, its answers at 4 and at 7 MTP
  drafts (b12024, 2026-10-31) differ on 9: none of the 236 answers under 1,000 tokens, 9 of the 46 longer ones. On
  MMLU-Pro (3) and MATH-500 (3) the first difference is 1,936 to 4,550 characters in; on MATH-500 with thinking on (3)
  the reasoning itself went another way (2,228 against 4,795 tokens on one). Asked again twice at 4 drafts on a fresh
  server, the 9 got the same answer both times (reasoning text included), and on 8 of them the first run's: those 8
  differences come from the draft depth. The ninth's first answer came after the other questions on the same server and
  is not reproduced without them (3,692 against 4,405 tokens). The speed run's short story shows it too: 460 tokens
  without drafts, 510 at 4 and at 7 (2026-10-31). A preregistered study of llama.cpp's MTP on one RTX 3090 (Hsiu-Chi
  Tsai, 2026-08-29, [repository](https://github.com/thc1006/qwen3.8-speculative-decoding-rtx3090),
  [archive](https://doi.org/10.5281/zenodo.22149941)) found its output the same from pass to pass but different from
  decoding without drafts in 23 to 25 of 25 prompts by 1,600 tokens.
- **Sampling keeps the distribution of sampling without drafts.** A draft is accepted with probability p(draft) and a
  rejected one is resampled from the rest of the distribution. Checked with chi-square and PIT tests; the counter-based
  random numbers, keyed by seed and position, were checked with a CPU replica of the loop (396k re-drafted positions)
  and a PIT test on 13,401 re-drafted rows of the engine at 7 drafts (KS p = 0.70; 2026-10-17). Two opt-in policy
  settings that were biased were fixed or restricted then ([ENGINE.md](ENGINE.md#correctness)). These are one-off runs,
  not part of the gate: the replica is `tests/spec_rng_sim.py` (needs scipy; pytest does not collect it), and the PIT
  test's script is not in the repository. The gate's sampled runs check that a seeded run is reproducible and that the
  cache does not change it, not the distribution.
- **The MTP layer only drafts.** A model file without one runs at depth 0, and its greedy output equals the drafted
  engine's bit for bit (UD-Q4_K_XL, 2026-10-17).

### Exact reuse

With `--exact-reuse` (`MK_EXACT=1`; the launchers set it), a reused request is answered bit for bit like the same
request sent fresh. The test is a state digest after the prompt (every K / V row, every row of the 8-bit window ring,
the DeltaNet and convolution state), compared with a fresh prefill of the whole prompt.

| case | result | date |
|---|---|---|
| follow-up turns, edits, abandoned replies, 38k context, q8_0 and preset D', several chunk sizes | digest bitwise equal to fresh | 2026-10-14 |
| sampled follow-up / edit / restored park with a fixed seed (the draft vocabulary rebuilt as a fresh prefill builds it) | 9 / 9 equal to fresh (before the change, 5 of 9 differed) | 2026-10-20 |
| a side request beside a cached conversation, a park and restore | digest after the side request / restore bitwise equal to before; the next replies equal to the conversation alone on a fresh server, greedy and sampled (8 / 8) | 2026-10-19 |
| a new session resumed at a pinned checkpoint (live, parked, after a side request, after the conversation grew to 63k) | replies equal to the same request on a fresh server, q8_0 and preset D, greedy and sampled | 2026-10-21 |
| a resume after decoding, a branch and back, a park and restore, a side request, ~11.8k tokens, q8_0, D and F (`scripts/capi_checks.py`) | state bitwise equal to a fresh prefill, and the next 64 greedy ids equal; 50 checks | 2026-11-06 |
| a follow-up, an edit, a sampled follow-up, three parked conversations, a side request and new sessions at a pinned checkpoint, q8_0, D and F (`scripts/exactness.py`) | replies equal to a fresh server's | 2026-11-06 |
| the same turns at ~99.5k with the launchers' settings (120k, a 1,024-token chunk) plus an edit inside the prompt, q8_0, D and F; a follow-up at ~255k on F at 262,144 | replies equal to a fresh server's | 2026-11-06 |
| 20 two-turn conversations, greedy, 2,048 tokens: the second turn after the first, against the same request on a fresh server | 0 of 20 differ | 2026-11-06 |

What it costs (2026-10-14): a follow-up turn re-reads the previous reply (11.6k-token chat, 7 drafts: 167 instead of
78 ms; with a thinking reply kept in the history: 469 instead of 86 ms). Edits cost the same either way. A fresh
98k-token prompt took 1.3% longer, and restoring a checkpoint and reading 32 / 128 / 512 tokens after it took 2.6% and
1.6% less and 0.7% more time (A-B-A, q8_0). The accuracy report is the same within noise: 98.662% / 0.00118 on the
short set and 99.121% / 0.00231 at 32k with exact reuse, 98.654% / 0.00116 and 99.023% / 0.00232 without. The CLI
measurements in [against llama.cpp](#against-llamacpp) run without exact reuse (the CLI's default); the launchers'
server runs with it.

The same 20 conversations on llama.cpp b12024 at 4 drafts (`cache_prompt` on, its default): 20 of 20 differ from a fresh
server, the first difference a median of about 355 characters in (4 to 1,671). They repeat: on another fresh server, five of them gave
the same warm answers and the same cold ones. With `cache_prompt` false on a running server, 3 of 20 still differ from a
fresh server. Its [server docs](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md) say the option
can change results.

### fp16 range in the prefill GEMMs

The prefill GEMMs add two 64-wide k stages (128 products) in fp16 before the fp32 accumulators, and store their inputs
as fp16 clamped to +-65504. A debug build with `MK_PF_INFCHK=1` (73fd978; compiled out of the served build) counts,
per layer and GEMM, the fp16 partial sums that are inf / NaN, the non-finite outputs, and the inputs the clamp
saturates or turns from NaN into -65504. Since 47f8b0d it also counts the DeltaNet scan's fp16 MMA operands (S, K S,
U, U d) past 65504, which are not clamped; the runs below predate those counters.

| prompt | result | date |
|---|---|---|
| a 38,048- and a 101,583-token prompt, `-d 7`, q8_0 KV | every counter 0, in every layer | 2026-10-17 |
| positive control: an 81-token prompt with a NaN image row | every counter fires; the engine stops with "no finite logit" | 2026-10-17 |

### Against llama.cpp

The engine is not bit-identical to llama.cpp: the arithmetic is ordered and rounded differently, so where two tokens are
close the other one can win. It is scored against llama.cpp's own next-token distributions: llama.cpp b10246 generating
token by token with its q8_0 KV cache, on 13 sequences of this repository's code and docs (13,299 positions scored after
a 1,024-token prompt each), with the 256 most likely tokens stored per position (`tools/llama_ref`, `--kld`).

| against llama.cpp generating token by token | top token agrees | KL divergence (mean, nats) | perplexity (reference 2.8099) |
|---|---|---|---|
| llama.cpp, batched instead (2026-10-11) | 98.489% | 0.00166 | 2.8116 |
| this engine, prefill then decode kernels (`mixed`), q8_0 KV (2026-10-17) | 98.737% | 0.00119 | 2.8085 |
| this engine, prefill kernels only, q8_0 KV (2026-10-17) | 98.744% | 0.00124 | 2.8073 |

The `mixed` report has been the same since 2026-10-15: none of the GPU changes up to v2026.11.05 changed it, and the
native Windows builds give it byte for byte. Most disagreements are near-ties: the median lead of llama.cpp's token over
the engine's is 0.043 nats (llama.cpp batched: 0.051), and 1 of the 168 disagreements exceeds 0.5 nats. At long context
(same reference setup, q8_0 KV): 99.121% / 0.00219 over 4,094 positions of 2 sequences at ~32k (2026-10-17), 99.267% /
0.00163 over 4,095 positions of one 64k sequence (2026-10-15, 7d67c5d), and 99.121% / 0.00217 over 4,095 positions of
one 100k sequence (2026-10-25, 0278d09; more in [100k and 160k context](#100k-and-160k-context)).

### Windows and Linux

The Windows build (MSVC + nvcc) against the Linux build (g++ + nvcc) of the same source (fdad907), on the same model
file, 2026-10-16:

| check | Windows vs Linux |
|---|---|
| greedy token ids, 7 and 0 drafts | identical |
| `--kld` report against llama.cpp (13,299 positions) | identical |
| server replies: text, thinking, an image, an image follow-up turn | identical |
| ptxas registers and spills, every function | identical |
| decode (38k context, chat prompts) and prefill (38k tokens) | within 1.5% |
| image encode, 196 / 300 / 1,024 tokens | 0.59 / 0.81 / 3.56 s vs 0.50 / 0.80 / 3.95 s |

Every release since has passed the same identity checks natively, and since 2026-10-19 also the side-request and
parked-conversation scenarios under q8_0 and preset D and the title scenario under preset D (v2026.10.30: all identical
to the Linux build). These comparisons are made by hand, a run on each build (`scripts/exactness.py --compare`,
[checking a build](BUILDING.md#checking-a-build)); `scripts/gate.sh` runs on Linux / WSL2 only.

The image encoder's rows first differed between the two builds at fp32 rounding level: its attention took e^x from the
C library, and glibc's and Microsoft's `expf` differ in the last bit on 193,924 of the 2^32 inputs. Since 2026-10-18
the encoder computes it with glibc's algorithm on both, and the Windows rows equal the Linux rows bit for bit (57
images: stills and every frame pair of 4 video clips, at 7 / 8 / 30 / 32 threads).

## The conversation cache

How it works: [SERVER.md](SERVER.md). Costs (WSL2, q8_0 unless noted, host memory already allocated, 2026-10-19):

| operation | 12k tokens | 38k tokens | 101.6k tokens | 120k tokens |
|---|---|---|---|---|
| park a conversation in host memory | 31 ms | 62 ms (D: 63 ms) | 151 ms | 177 ms (4.99 GB) |
| restore it | 29 ms | 65 ms (D: 74-92 ms) | 152 ms | 178 ms |
| read it again instead | 7.4 s | 23.9 s | 78.7 s | 98.4 s |

- The first park of a large conversation pays for page-locked host memory at ~1.1 s per GB (5.7 s for 120k tokens); it
  is kept for reuse until the server exits.
- A side request (a client's title or summary request beside a long conversation) moves only the rows it overwrites:
  12 ms each way for a 2.7k-token side prompt, 20 / 19 ms for 8k (preset D: 17-22 ms), whatever the length of the cached
  conversation (2026-10-19). Its own conversation is parked afterwards: 10-15 ms for a title, ~10 ms back, 15-21 ms for
  ~400 tokens (2026-10-20).
- Two OpenCode 2.0.32 sessions working at the same time on native Windows (230k context, preset D) finished in 110 s
  with ~72k prompt tokens read, against 242 s and ~232k tokens with the host store off (2026-10-19).

### Pinned prefix checkpoints

A new session of a client that keeps its system prompt and tools resumes after them. Measured with each client's
captured first requests (preset D, 131,072 context, native Windows, 2026-10-21): the server before the change (538d441,
with 7322fbd's engine) against the change (merged as 2e659e0 + 4eb7b51). "Live": a 52-55k-token conversation of the
same client is cached; "parked": another 9.8k-token conversation has parked it. Times are the client's wall time,
including the park of the long conversation and ~2.5-3.5 s of first-use host-memory allocation.

| new session (prompt tokens) | | before: cached / time | after: cached / time |
|---|---|---|---|
| Claude Code, default settings (11,850) | live | 0 / 10.12 s | 8,288 / 5.92 s |
| | parked | 0 / 8.67 s | 8,288 / 3.98 s |
| Claude Code with its cache-friendly settings (11,809) | live | 0 / 9.96 s | 9,024 / 5.11 s |
| | parked | 0 / 8.28 s | 9,024 / 3.27 s |
| Codex CLI, `/v1/responses` (8,847) | live | 0 / 8.01 s | 8,672 / 3.44 s |
| | parked | 32 / 6.64 s | 8,672 / 2.00 s |
| Crush, another directory (11,168) | live | 0 / 9.30 s | 10,304 / 4.11 s |
| | parked | 32 / 7.67 s | 10,304 / 2.12 s |

### Replayed client traffic

The requests 9 clients sent during their test drives on 2026-10-21, replayed back to back against a native Windows
server (preset D, 230k context): ef4a6dd (the 2026-10-21 release) against the server before its pinned checkpoints and
side-request rules (538d441, the same engine). The server samples its own replies, so reply lengths differ slightly
from the live runs.

| | before (538d441) | ef4a6dd |
|---|---|---|
| 9 single-client sequences: time reading prompts | 885 s | 800 s |
| tokens read | 1,267k | 1,150k |
| first turns of new agent sessions (50 requests) | 173 s | 120 s |
| DeepSeek Harness / Crush / Qwen Code / Pi / Goose / Claude Code first turns | 29.8 / 30.5 / 62.3 / 8.4 / 5.7 / 27.9 s | 11.3 / 20.2 / 47.2 / 5.2 / 3.1 / 24.3 s |
| Open WebUI first turns | 92 s | 75 s |
| Claude Code + Open WebUI interleaved: tokens read, cache misses | 508k, 28 | 489k, 10 |
| DeepSeek Harness + Aider interleaved: tokens read, cache misses | 266k, 32 | 218k, 7 |

- **The cost:** every parked conversation carries its pinned checkpoint (+288 MiB with preset D), so fewer fit in the
  host store (Claude Code's sequence held 4 instead of 6, and two of its requests cached 4-6k fewer tokens).
- **Title requests** that quote a whole long chat (Open WebUI's default) are too long to run beside it. Since 2efba87
  they no longer cost the chat its place in the cache: replayed on 2efba87 (2026-10-24), five of the sequences spent
  698 s reading prompts instead of 856 s. The trade-off: a chat whose title request looks the same but is never
  continued keeps its place too, and in one sequence it crowded out the pinned system prompt of two new Claude Code
  sessions (11.9k tokens read instead of 3.5k, each). With Open WebUI's title and tags templates set as in
  [CLIENTS.md](CLIENTS.md) (quoting only the end of the chat), these requests are short and run beside the chat.

`--park-min` (among its uses: conversations of fewer tokens are not parked; [SERVER.md](SERVER.md)), both interleaved
sequences, ef4a6dd, fixed seeds:

| `--park-min` | parks | GiB copied | real conversations evicted | tokens read | time reading prompts |
|---|---|---|---|---|---|
| 1024 (the default) | 68 | 106.5 | 15 | 706,771 | 489 s |
| 4096 | 51 | 81.4 | 8 | 709,651 | 489 s |

4096 halved the evictions of real conversations and copied a quarter less for the same prompt time; the default stays
1024.

## Keep-warm

After each request the engine keeps the GPU's clocks up for `MK_KEEP_WARM_S` seconds (default 120). On native Windows
at 147054b, a 255-token reply after 65 s of idle started in ~65 ms instead of 391-397 ms and decoded at 80.2 instead of
61-63 tok/s (2026-10-17). Holding the clocks costs power: 164-173 W instead of 30-35 W at idle on this card
(2026-10-17).

## Images and video

On the Ryzen 9 5900XT (32 threads):
- **Images** (2026-10-16): a 196 / 300 / 1,024-token image encodes in 0.51 / 0.78 / 3.56 s back to back; llama.cpp's
  libmtmd on the same CPU took 0.80 / 1.29 / 6.07 s. A new image after the CPU has been idle takes ~1.2-1.5 s for 300
  tokens while the cores ramp up. The encoder's rows are within 4e-6 to 4e-5 (relative) of an fp64 forward of the vision
  tower as llama.cpp's graph runs it, tanh GELU in the patch merger included (libmtmd: 4e-3 to 2.7e-2). On six image
  prompts, the mean KL against llama.cpp fed the same image rows is 0.0006-0.0059 (text only: 0.0012). Prompt tokens
  equal llama-server's on 7 of 7 image prompts.
- **Video** (2026-10-18; frames sampled at 4 fps as llama.cpp does, two consecutive frames per grid of tokens): a 640 x
  360 clip costs ~450 prompt tokens per second of video (228 at 2 fps) and its frame pairs encode in 0.56 s each, about
  real time at 4 fps; 320 x 240: 167 tokens per second, 0.22 s per pair; 1280 x 720: 1,861 tokens per second, ~3.2 s
  per pair. The prompt token counts equal llama-server's on four clips, and the frame-pair rows are within 5e-6 to
  1.1e-5 of an fp64 forward (libmtmd's: 5.5e-3 to 2.2e-2).

The image and video encoder runs on the CPU, so it takes no VRAM: with an image projector loaded, the idle server's
VRAM at 120k context did not grow (2026-10-16).

## Wide verify (2026-10-13)

The int8 tensor-core verify pass (wide verify, up to 7 drafts) when it was first served, 72a58cd at 7 drafts, against
the dp4a verify pass before it, 999fc68 at its most, 4 drafts: greedy decode on the short prompts 17.6% faster, sampled
with thinking 19.5% faster, decode at 101.6k context 9.2% faster. The dp4a pass summed in another order (greedy output
still independent of the draft depth) and was neither released nor gated; its build went with the other compile-time
switches ([CONFIGURATION.md](CONFIGURATION.md#former-flags)).

## Earlier full runs

Older runs (2026-10-14 and 10-15) are in this file's history at fe21771; the 2026-10-20 run (7322fbd) and the previous
comparison with llama.cpp (2026-10-31: b12024 at 4 and 7 drafts, v2026.10.30) at 605f82a; the 2026-10-30 run (6f3b257)
and the whole 2026-11-03 comparison at 3ed5ce5; and the llama.cpp column before that (b11548, the best of three
sessions on 2026-10-12 / 13) at d034e3a. The 2026-10-14 run with charts and raw
data, and the gates behind each change up to 2026-10-16, are in the open-jet fork's
[perf docs](https://github.com/cutlerbenjamin1-cmd/open-jet/tree/qwen38-perf/perf), where this engine was developed
until it moved here.

### 2026-11-03 (605f82a)

The same client and settings as the 2026-11-06 run, one session, fans at 97-98%, 581 MiB of the card the desktop's.

| | llama.cpp, 4 drafts | q8_0 KV | preset D | preset F |
|---|---|---|---|---|
| greedy decode, short prompts (mean / token-weighted) | 86.5 / 79.9 tok/s | 165.3 / 140.6 | 165.3 / 140.5 | 160.1 / 135.1 |
| sampled decode, thinking on (mean / token-weighted) | 72.4 / 67.4 tok/s | 145.9 / 139.4 | 145.8 / 139.2 | 146.7 / 138.5 |
| decode at 11.6k / 38k / 101.6k context | 71.9 / 65.7 / 54.5 tok/s | 147.5 / 126.9 / 115.4 | 145.9 / 108.1 / 104.9 | 132.5 / 108.6 / 116.7 |
| read an 11.6k / 38k / 101.6k-token prompt | 7.59 / 28.0 / 96.7 s | 5.87 / 22.7 / 72.5 s | 5.95 / 23.7 / 75.6 s | 5.94 / 23.4 / 75.2 s |
| first token, short prompt | 118 ms | 61 ms | 61 ms | 61 ms |
| follow-up turn (11.8k chat) / edit a message / edit the system prompt, prompt time | 239 / 247 / 883 ms | 174 / 71 / 84 ms | 180 / 71 / 94 ms | 181 / 72 / 94 ms |
| VRAM, whole card, server idle | 22.8 GiB | 21.1 GiB | 20.0 GiB | 19.8 GiB |
| 282 questions | 229 right, 2,570 s | 230 right, 1,607 s | 230 right, 1,644 s | 229 right, 1,647 s |

Time per engine cycle, q8_0 / D / F: 25.5 / 26.2 / 25.4 ms at 11.6k, 27.9 / 27.6 / 28.2 at 38k, 31.3 / 32.4 / 32.6 at
101.6k. The 4 agents on Q4_K_M: 424.0 tok/s on short prompts, 366.0 with ~23k-token contexts, 203.5 and 170.7 alone.

## Reproducing

- **Server benchmarks:** [bench/suite](../bench/suite/README.md) has the clients behind the server tables (speed, the
  282 questions, the from-scratch prefill, the 262,144-token run), the scorer and the summary script. The datasets are
  not in this repository: `python -I bench/suite/fetch.py` downloads them and checks every item against the hashes of
  the items these runs used (`bench/suite/items.json`). HumanEval answers are executed only with `score.py
  --exec-humaneval`. The 2026-11-06 run is the suite as published, and its summary script made the table above;
  re-scoring the saved 2026-10-20 answers with the suite's scorer gives the same 230 right answers for q8_0 and
  preset D, item for item.
- **Parallel agents:** the parallel step of [bench/suite](../bench/suite/README.md) (`parallel_client.py`, its context
  fetched by `fetch.py`); without a server, `python3 bench/slots_bench.py <gguf> all` checks the engine's slots against
  each conversation alone and times them.
- **Speed, CLI:** `python3 bench/bench.py mk --draft 4 -n 512 --draft-vocab draft_vocab_32k.bin --model <gguf>` (the
  four chat prompts through the CLI; it runs the CLI's default KV cache, so at most 4 drafts); `bench/code_bench.py` and
  `bench/think_bench.py` are upstream's scripts; `bench/*.cu` are kernel microbenchmarks.
- **Accuracy against llama.cpp:**

  ```sh
  make tools/llama_ref
  tools/llama_ref -m model.gguf -f code.txt -f prose.txt --score-from 1024 -b 1 --ref-out llama.ref
  tools/llama_ref -m model.gguf --kld llama.ref                            # llama.cpp batched, for scale
  ./qwen38_mk -m model.gguf --kld llama.ref --score-path mixed --kv q8     # this engine
  # long context: one sequence of N tokens, its last 4,096 positions scored (here N = 100,352)
  tools/llama_ref -m model.gguf -f pf101k.txt --chunk 100352 --max-chunks 1 --score-from 96256 -b 512 --ref-out long.ref
  MK_KV='K*=5r,V*=5' MK_KV_WIN=2048 ./qwen38_mk -m model.gguf --kld long.ref --score-path mixed --kv q8   # preset F
  ```

- **Draft-depth independence:** `./qwen38_mk -m <gguf> -p "..." -n 256 -d 7 --kv q8 --draft-vocab draft_vocab_32k.bin
  --ids`, then the same with `-d 4` and `-d 0`: the ids must be identical.

More on each check: [BUILDING.md](BUILDING.md#checking-a-build).
