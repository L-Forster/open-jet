# The server

`server/mk_server.py` puts the engine behind an HTTP server that speaks three APIs: OpenAI chat completions, the
Anthropic Messages API and the OpenAI Responses API, plus llama-server's own endpoints. Coding agents, chat UIs and the
official SDKs talk to it as they would to llama-server or a hosted model; [CLIENTS.md](CLIENTS.md) has tested settings
for 11 of them. It serves one model, runs one request at a time (or up to four at once with
[`--agents`](#several-agents-at-once---agents)), and keeps whole conversations cached between requests: on the GPU, and
in host memory when several conversations take turns.

**The cache never changes an answer.** With exact reuse on (the launchers' default), a reply served from the cache is,
bit for bit, the reply the same request gets on a freshly started server. That holds for a follow-up turn, an edited
conversation, a new session that resumes after the client's system prompt, a conversation brought back from host memory,
and a conversation kept aside while a client's title request ran. [The conversation cache](#the-conversation-cache)
explains how, what it costs and how it was checked.

Every option is listed with its default in [CONFIGURATION.md](CONFIGURATION.md); this page says what the options do.

- [Running it](#running-it)
- [The conversation cache](#the-conversation-cache)
- [Several agents at once: `--agents`](#several-agents-at-once---agents)
- [APIs and endpoints](#apis-and-endpoints)
- [Request features](#request-features): streaming, tools, thinking, stop strings, sampling, templates, images, video
- [Keep-warm](#keep-warm)
- [Monitoring](#monitoring): the console, `/metrics`, `/props`, `/slots`
- [Errors, cancels and disconnects](#errors-cancels-and-disconnects)
- [Security](#security): Host / Origin checks, CORS, special tokens in content
- [Limits](#limits)
- [Engine calls behind the server](#engine-calls-behind-the-server)
- [Windows](#windows)

## Running it

The launchers start the server with the recommended settings:

```sh
scripts/serve.sh                     # Linux / WSL2: http://127.0.0.1:8080, 120,000-token context
scripts/serve.sh 8080 64000          # port, context
MK_KV='K*=6,V*=5' MK_KV_WIN=2048 scripts/serve.sh 8080 262144   # preset D: the model's full context on 24 GB
MK_HOST=0.0.0.0 scripts/serve.sh     # reachable from other machines on the LAN
```

```bat
set MK_MODEL=D:\models\Qwen3.8-27B-UD-Q4_K_XL.gguf
scripts\serve.bat                    :: native Windows, the same defaults and settings
```

`MK_MODEL` names the GGUF; the settings each launcher reads are listed at its top and in
[CONFIGURATION.md](CONFIGURATION.md). The server itself can also be started directly; its own defaults are more
conservative than the launchers':

```sh
python3 server/mk_server.py -m Qwen3.8-27B-UD-Q4_K_XL.gguf -c 120000 --kv q8 -d 7 --exact-reuse --mmproj mmproj-BF16.gguf
```

| setting | launchers (`serve.sh`, `serve.bat`) | `mk_server.py` alone |
|---|---|---|
| port | 8080 (first argument) | 18080 (`--port`) |
| context | 120,000 tokens (second argument) | 32,768 (`-c`) |
| KV cache | q8_0; a low-bit preset with `MK_KV` / `MK_KV_WIN`; `serve.bat` preset F above 240k unless `MK_KV` is set | `--kv auto`: fp16 when it fits, else q8_0 |
| MTP drafts per cycle | 7 (`MK_DRAFT`) | 4 (`-d`) |
| exact reuse | on (`MK_EXACT_REUSE=0` turns it off) | off (`--exact-reuse` turns it on) |
| prefill chunk | 1,024 tokens above 96k context, else 2,048 (`MK_PF_CAP`); `serve.bat` 512 above 240k with your own `MK_KV` | 2,048 |
| images | on when the mmproj file and the vision library are found | `--mmproj FILE` |
| listen address | 127.0.0.1 (`MK_HOST`) | 127.0.0.1 (`--host`) |

Side requests, parking, checkpoints and keep-warm use the server's defaults in both cases.

The port opens before the model loads: until the load is done every request gets llama-server's 503 `Loading model`
(`/health` answers 200 `{"status": "ok"}` once ready), and a port that is already taken fails at once instead of after
the load. With the model on a local disk the server is ready in about 12 s under WSL2 (model on ext4) and about 7 s
natively on Windows (NVMe). Under WSL2, keep the GGUF on the Linux filesystem: files on `/mnt/c` load several times
slower. Before it listens, the server runs one prefill and one decode cycle, so a context too large for the card fails
at startup rather than on the first request (on a driver with the CUDA sysmem fallback, Windows and WSL2, it runs,
slowly, and the engine warns at load: [ENGINE.md](ENGINE.md#kv-cache)).

```sh
curl -s http://127.0.0.1:8080/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "messages": [{"role": "user", "content": "Write quicksort in C."}], "max_tokens": 512}'
```

### Options at a glance

Defaults in brackets; each is described in full in [CONFIGURATION.md](CONFIGURATION.md), and the sections below say
what they do.

| topic | options (environment twin) |
|---|---|
| several agents at once | `--agents N` [0] (`MK_AGENTS`), `--parallel N` / `-np N`: [several agents at once](#several-agents-at-once---agents) |
| model and engine | `-m` / `--model` (required), `-c` / `--ctx` [32768], `-d` / `--draft` [4], `--kv auto\|f16\|q8` [auto], `--draft-vocab` [draft_vocab_32k.bin], `--mtp FILE` (`MK_MTP`), `--no-mtp` (`MK_NO_MTP=1`), `--lib` |
| network | `--host` [127.0.0.1], `--port` [18080], `--allow-host NAME` (`MK_ALLOW_HOSTS`), `--cors-origin ORIGIN` (`MK_CORS_ORIGINS`), `--api-key KEY` (`MK_API_KEY`): [Security](#security) |
| conversation cache | `--exact-reuse` (`MK_EXACT=1`), `--ck-every` [4096], `--ck-msgs` [1], `--ck-pin` [1], `--side-max` [8192] (`MK_SIDE_MAX`), `--park-mib` [8192] (`MK_PARK_MIB`), `--park-min` [1024], `--park-cks` [2]: [the conversation cache](#the-conversation-cache) |
| templates and output | `--chat-template-file PATH`, `--chat-template TEMPLATE`, `--chat-template-kwargs JSON` (`MK_CHAT_TEMPLATE_KWARGS`), `--reasoning-format` [auto] (`MK_REASONING_FORMAT`), `--content-specials` [text] (`MK_CONTENT_SPECIALS`) |
| images and video | `--mmproj FILE`, `--vision-lib`, `--vision-backend` [auto], `--vision-threads` [0: every logical CPU but two], `--image-min-tokens` / `--image-max-tokens` [0: the mmproj's], `--image-cache` [8], `--video-fps` [4], `--video-max-tokens` [0: the image limit], `--video-cache` [2], `--ffmpeg` (`MK_FFMPEG`) |
| engine environment | `MK_CK_SLOTS` [8; 1 with `--agents`], `MK_KEEP_WARM_S` [120], `MK_STORE_CHUNK_MIB` [32], `MK_TIMEOUT_S` [600], `MK_REQUEST_TIMEOUT_S` [300], `MK_MAX_BODY_MB` [128], `MK_MAX_INFLIGHT_MB` [256], `MK_MAX_CONNECTIONS` [256], `MK_MAX_IMAGES` [32], `MK_IMAGE_MAX_PIXELS` [100000000], `MK_MAX_VIDEO_FRAMES` [256] |

## The conversation cache

A coding agent re-sends its whole conversation with every request: system prompt, tool definitions, every earlier turn
and tool result. Reading all of that again each time would cost seconds to minutes per turn, so the server keeps what
the engine has already processed and prefills only what is new. It does that across follow-up turns, edits, new
sessions of the same client, several conversations that take turns, and the small requests clients send on the side
(titles, tags, commit messages). None of it changes a reply.

### Exact, and how it is checked

With exact reuse on, the server never builds on rows that the decode kernels wrote (their arithmetic order differs from
the batched prefill's). It restores the last checkpoint at or before the point where the new prompt differs from the
cached conversation and prefills from there; the prefill computes rows that do not depend on where its chunks start
(attention over absolute key segments, checkpoints on the absolute 32-token grid of the recurrent scan), so the result
is the state a prefill of the whole prompt from position 0 would produce. Everything else in this section
moves that state around (to host memory and back) without touching its bits.

| situation | reply is identical, bit for bit, to | checked by |
|---|---|---|
| follow-up turn; edited message or system prompt | the same request on a fresh server | `scripts/exactness.py` (q8_0, D, F: at 32k, and at ~100k with the launchers' settings, F also at ~255k) and the state digest of `scripts/capi_checks.py` (every K / V row, window-ring row and DeltaNet / conv state, bitwise against a fresh prefill) |
| new session resumed at a pinned checkpoint | the same request on a fresh server (greedy, and temperature 1.0 with fixed seeds) | `scripts/exactness.py`, q8_0, D and F: a new session after a live and after a parked conversation; by hand, after a 63k-token one |
| parked conversation brought back | each conversation alone on a fresh server | `scripts/exactness.py`, q8_0, D and F: three conversations, greedy and sampled; the state digest; by hand on each release (not in the scripts), 8 interleaved requests |
| side request, and the conversation kept beside it | each of them alone on a fresh server | `scripts/exactness.py`, q8_0, D and F; the state digest; by hand on each release (not in the scripts), 5 side-request scenarios and 40 titles beside two conversations |
| a prefill stopped because the client left | a server that never saw that request | by hand on each release (not in the scripts): the requests after a stopped prefill |

Sampled replies are identical too when the request sets a `seed`: the draft vocabulary the MTP head uses is rebuilt
after a rewind exactly as a fresh prefill builds it. Without a `seed`, each request draws a random one.
`scripts/exactness.py` runs on Linux and on the native Windows build (`scripts/gate.sh` on Linux / WSL2), and the two
builds give the same replies (two runs compared with `--compare`: [BENCHMARKS.md](BENCHMARKS.md#windows-and-linux)).

What is not exact, and against what:
- **Exact reuse off** (`MK_EXACT_REUSE=0` in the launchers, no `--exact-reuse` on the server): a follow-up appends to
  the cached rows instead of re-prefilling the previous reply, and those rows came from the decode kernels, whose
  arithmetic order differs from the prefill's. Replies are then not bit-identical to the same request sent fresh.
  Parking and side requests still restore the engine's state bit for bit. With a preset's window (`MK_KV_WIN`), a
  rewind to a checkpoint also refills the window rows it cannot restore from the low-bit main cache, so they keep the
  low-bit precision until they leave the window (with exact reuse, each checkpoint keeps the rings).
- **Low-bit KV presets** (`MK_KV`) are near-lossless against the q8_0 cache, not identical to it; each preset is exact
  with itself. Their accuracy is in [BENCHMARKS.md](BENCHMARKS.md).
- **llama.cpp** gives different bits (different arithmetic order); the distance is measured in
  [BENCHMARKS.md](BENCHMARKS.md).

Why it matters: an agent's run can be replayed request by request; a reply never depends on what the server happened
to have cached; and an answer you want to debug comes back the same when you send the request again (greedy, or with
the same seed).

### How a request finds its cache

The engine's cached sequence is every prompt token and every generated token of the conversation it last served. When
a request arrives, the server:

1. Compares the prompt's tokens with that sequence. With exact reuse it rewinds to the last checkpoint at or before the
   first difference and prefills the rest (`resumed at the checkpoint at P`). Without exact reuse it continues right
   after the last matching token when it can (`continued from there`), and from a checkpoint otherwise.
2. Looks at the parked conversations: one that this prompt continues better, by at least `--park-min` tokens [1024], is
   brought back and the current one parked ([parking](#several-conversations-parking)).
3. Runs a client's one-shot request (a title, tags, a commit message, a summary) as a
   [side request](#one-shot-requests-side-requests) beside the cached conversation.
4. Parks the cached conversation first when the request would throw away at least `--park-min` tokens of it without
   continuing it.
5. Prefills from position 0 when nothing matches (`CACHE MISS` on the console, with how much of the cached sequence the
   prompt matched).

Requests run one at a time, in arrival order, and the server switches conversations only between replies (a reply in
progress is never paused). An image counts as part of the match: the reused prefix ends at the first image that is not
the same image at the same place.

### Follow-ups and edits

A follow-up turn re-sends the conversation plus the model's last reply. With exact reuse, the prompt resumes at the
checkpoint saved at the previous generation prompt, and the previous reply is prefilled again as part of the prompt:
that re-prefill is what exactness costs. An edit to an earlier message, or to the end of the system prompt, resumes at
the last checkpoint before the edit.

Prompt processing time of each request (WSL2 server, 120,000-token context, 7 drafts, exact reuse on; the
[full run](BENCHMARKS.md#full-run-2026-11-06) in BENCHMARKS.md):

| request | q8_0 KV | preset D | preset F |
|---|---|---|---|
| follow-up turn of an 11.8k-token chat | 169 ms | 177 ms | 178 ms |
| edit an earlier message of an 11.7k-token chat | 69 ms | 70 ms | 71 ms |
| edit the end of a 9.1k-token system prompt | 82 ms | 93 ms | 93 ms |
| for scale: the 11.6k-token prompt from scratch | 5.77 s | 5.86 s | 5.85 s |

What exactness costs against appending (exact reuse off; 11.6k-token chat, 7 drafts): a follow-up turn took 167 ms
against 78 ms, a follow-up that keeps the previous turn's thinking 469 against 86 ms, and edits the same either way.
The longer the previous reply, the more the follow-up re-prefills.

### Checkpoints

The model's 48 recurrent layers (Gated DeltaNet) carry a state that cannot be rewound by dropping rows, so the engine
saves checkpoints of it during the batched prefill (with exact reuse, of the low-bit window rings too) into a pool of
8 slots in pinned host memory (`MK_CK_SLOTS`). It saves one at the generation prompt (the next turn re-renders the reply
from there), one at the start of each message (`--ck-msgs 1`) and one every `--ck-every` [4096] prompt tokens, on the
absolute 32-token grid of the recurrent scan. When the pool is full it keeps the most useful ones and thins the oldest:
in a conversation that grows past ~20-35k tokens, the checkpoints near its start are gone (that is what pinning,
below, fixes). A checkpoint holds 174 MiB with the q8_0 cache and 288 MiB with preset D (the server prints the size at
startup).

### New sessions: pinned prefix checkpoints

Every new session of a coding agent starts with the same system prompt and tool definitions, often 8-12k tokens. The
server pins the checkpoint at the end of a prompt's system message (where the tools end too: the chat template renders
them inside it), so the pool never evicts it, and parks it with its conversation. A new session of the same client, an
agent's next task or another subagent of the same kind, then resumes after the system prompt instead of reading it
again (`--ck-pin 1`, the default).

When a client's system prompt ends with per-session text (a date, the working directory, a session id), a new session
leaves the cached conversation inside the system message. The server then also pins that point, and the client's later
sessions resume there. At most two pins per conversation, inside the pool (no memory of their own; with `MK_CK_SLOTS`
under 8 one, under 4 none); prompts without a system message, or with one under 1,024 tokens, pin nothing. A parked
conversation that a new session restores only for its system prompt is parked again at once, so it is not lost.

Measured with the clients' captured first requests (WSL2, preset D, 131,072-token context). The new session arrives
after a 52-55k-token conversation of the same client that is either still live or parked; times are the client's wall
time and include parking the long conversation, ~2.5-3.5 s of which was the first allocation of host memory:

| client (prompt of the new session) | without pins: cached / time | with pins: cached / time |
|---|---|---|
| Claude Code, default settings (11,850 tokens), live | 0 / 10.1 s | 8,288 / 5.9 s |
| Claude Code, default settings, parked | 0 / 8.7 s | 8,288 / 4.0 s |
| Claude Code, cache-friendly settings (11,809), live | 0 / 10.0 s | 9,024 / 5.1 s |
| Claude Code, cache-friendly settings, parked | 0 / 8.3 s | 9,024 / 3.3 s |
| Codex CLI, `/v1/responses` (8,847), live | 0 / 8.0 s | 8,672 / 3.4 s |
| Codex CLI, parked | 32 / 6.6 s | 8,672 / 2.0 s |
| Crush, another directory (11,168), live | 0 / 9.3 s | 10,304 / 4.1 s |
| Crush, parked | 32 / 7.7 s | 10,304 / 2.1 s |

Replies are those of the same request on a fresh server. Claude Code's default settings and Crush resume at the learned
second pin (Claude Code's per-session billing header; Crush's working directory); Claude Code's cache-friendly
settings are in [CLIENTS.md](CLIENTS.md#claude-code). The cost: a pin occupies one slot of the pool, and a parked
conversation with a pin carries one more checkpoint (+174 MiB with q8_0, +288 MiB with preset D).

### Several conversations: parking

When a request belongs to another conversation, the cached one is copied to pinned host memory first (parked): its KV
rows, window rings, recurrent state, token sequence, image positions, draft vocabulary, its newest `--park-cks` [2]
checkpoints and its pinned ones. The engine keeps it too, so a shared system prompt is still reused. A later request
that continues a parked conversation better than the cached one, by at least `--park-min` tokens, parks the cached one
and brings the parked one back: the engine is then bit for bit as it was when the conversation was parked, and its
replies are the ones it would have given had it never left.

"Another conversation" means a request that would throw away at least `--park-min` tokens of the cached one, does not
continue its last turn, and differs from it before its first reply. A request that repeats the cached conversation up to
its first assistant message and differs only later is that same conversation with its history edited (a client moving
or dropping a block, clearing old tool results, a chat message edited): it is served from the engine's state, and no
copy is parked that would never be continued (`not parked: the request continues the cached conversation with its
history edited` on the console).

Parked conversations share a host-memory budget, `--park-mib` [8192 MiB]. Memory is allocated on first use and kept
for reuse until the server exits. When the budget is full, the least recently parked conversation is evicted (it is
prefilled again if it comes back), but two kinds go first, least recently parked first among them:
- side conversations (below), at most 4 of which are kept;
- conversations a client is unlikely to continue: one it has just summarised to compact it, and the conversation of the
  summary request itself (`low priority: <why>` on the park line). A request without tools that replaces a conversation
  with tools is taken for such a summary unless it answers in fewer than 256 tokens (thinking aside): after such a short
  answer (a client's title or tags request that quotes the whole chat), the replaced conversation keeps normal priority
  (`keeps normal priority` on the console). One that comes back anyway is a normal conversation again.

A conversation parked with low priority would be the first to go, so it only takes the room that side conversations and
other low-priority ones leave: when that is not enough it is not parked (`not parked: ... have low priority` on the
console) and nothing is evicted for it. The one exception is the conversation a request without tools has just
replaced: its low priority waits for that request's answer, so it is parked like any conversation.

What a parked conversation costs: its rows (36,992 bytes per token with the q8_0 cache, 26,112 with preset D), the
recurrent state (174 MiB) and its checkpoints (174 or 288 MiB each). A 38k-token conversation with the q8_0 cache and
two checkpoints takes 1.9 GB.

Park and restore against reading the conversation again (WSL2, q8_0 cache, memory already allocated):

| conversation | park | restore | prefill it again |
|---|---|---|---|
| 12k tokens | 31 ms | 29 ms | 7.4 s |
| 38k tokens (1.9 GB) | 62 ms | 65 ms | 23.9 s |
| 101.6k tokens (4.3 GB) | 151 ms | 152 ms | 78.7 s |
| 120k tokens (4.99 GB) | 177 ms | 178 ms | 98.4 s |

Preset D's window rings add a little (38k: 63 ms to park, 74-92 ms to restore). The copies run at ~26 GB/s, but the
first park of a large conversation is slower, because page-locked memory is allocated at ~1 GB/s (5.7 s for the first
120k-token park); after that the memory is reused.

In practice: with OpenCode 2.0.32 on native Windows (230,000-token context, preset D), two coding sessions running at
the same time finished in 110 s with ~72k prompt tokens prefilled, against 242 s and ~232k tokens with the store off.

### One-shot requests: side requests

Clients send small requests next to the real conversation: a title for the session, tags, follow-up suggestions, a
commit message, a session name, a summary. Served the normal way, each would push the long conversation out of the
engine and the next real turn would wait for it to come back. The server runs them beside it instead (side requests):

- **Which requests:** a prompt of at most `--side-max` [8192] tokens with no earlier assistant reply in it that would
  otherwise throw away at least twice its length (and at least `--park-min` tokens) of the cached conversation. With
  parking on, it must also be a one-shot: a request without tools, whatever its `max_tokens`, or one that asks for at
  most 1,024 tokens. Every tested agent client sends its one-shots without tools and every new session's first turn
  with tools, so a new agent session is parked like any conversation and finds its own next turns cached. With parking
  on, any later request without tools (a chat without tools, a summary that quotes the conversation as messages) runs
  beside a conversation whose last request had tools in the same way.
- **What happens:** the engine keeps the cached sequence aside and runs the request from position 0, as a reset engine
  would (its reply is the one a fresh server gives). Only what the side request would overwrite is copied to host
  memory: the recurrent state at once, the KV and window-ring rows just before a launch would overwrite them, 1,024
  rows at a time; the cached sequence's checkpoints are hidden, and a slot the side request needs is copied out first.
  After the reply, also after a disconnect or an error, everything is copied back and the cached conversation is bit
  for bit as before.
- **Cost:** ~10-20 ms each way under WSL2 for a 2.7k-8k-token side prompt, whatever the length of the cached
  conversation.
- **Room:** before it starts, the server makes room in the host store (other side conversations go first) for the rows
  of its prompt and reply: `max_tokens`, or 8,192 tokens when the request sets none. If the store still runs out, or
  the engine fails, the cached sequence is dropped and the next prompt is prefilled in full (logged; never a corrupt
  cache).
- **Side conversations:** with parking on, the side request's own conversation is parked after its reply, with one
  checkpoint (its generation prompt). A later request that continues it and would be a side request but for its earlier
  turns runs as a side request again, resumed from there, while the long conversation stays on the GPU: a chat without
  tools keeps its prefix cached beside an agent's conversation. At most 4 side conversations are kept (the oldest goes
  first; most are titles, never continued), they are evicted before the other parked conversations, and parking one
  never evicts anything else. Steady state for titles (WSL2, q8_0): kept aside in 10-15 ms, back in 10 ms,
  its own ~400-token conversation parked (363 MiB) in 15-21 ms. Continuing a 1.5k-token side conversation beside a
  9k-token one took 27-34 ms with q8_0 (425 MB copied) and 86 ms with preset D (new host memory included), and parking
  it again with the long conversation's rows back 35-41 ms (q8_0) and 70 ms (preset D).
- **With parking off** (`--park-mib 0`), side requests still work, with a 1 GiB store: every first turn of at most
  `--side-max` tokens that would throw away a much longer conversation runs beside it, and its own conversation is not
  kept.

A side request starts from position 0: it does not reuse the cached conversation's prefix, even when it shares the
system prompt. A side request without tools that quotes at least `--park-min` tokens and answers in at least 256 (a
summary; titles, tags, follow-ups and commit messages answer in tens) marks the conversation beside it as summarised: if
the next request replaces that conversation instead of continuing it, it is parked with low priority. A request without
tools too long to run beside a conversation with tools replaces it: the replaced conversation is parked with low
priority, and gets normal priority back if that request answers in fewer than 256 tokens.

### Sizing the host store

Parked conversations, side requests and the checkpoint pool live in page-locked host memory, which the OS cannot swap
out. Three server options size it:

- `--park-mib` [8192]: the budget. On the replayed client traffic ([below](#measured-on-real-client-traffic)), both
  mixed sequences filled 8 GiB.
- `--park-cks` [2]: checkpoints kept per parked conversation besides its pins. With exact reuse any checkpoint gives
  the same bits, so fewer only cost re-prefill after an edit. With 1, less is copied (47.9 instead of
  62.0 GiB over the two mixed sequences), but a parked conversation can no longer resume before its last turn: 7% more
  tokens were prefilled (measured without pinned checkpoints).
- `--park-min` [1024]: conversations shorter than this are not parked, and a parked conversation comes back only when
  the prompt reuses this many more tokens of it than of the cached one. It also bounds side requests from below (a side
  request must save at least this much).

`--park-min 4096` against the default (native Windows, 230,000-token context, preset D, 8 GiB store; two replayed
mixed-client sequences, 201 requests, fixed seeds):

| `--park-min` | parks | GiB copied | real conversations evicted | restores | cache misses | tokens prefilled | prefill time |
|---|---|---|---|---|---|---|---|
| 1024 (server default) | 68 | 106.5 | 15 | 25 | 17 | 706,771 | 489 s |
| 4096 | 51 | 81.4 | 8 | 19 | 18 | 709,651 | 489 s |

4096 halved the evictions of real conversations at the same prefill time; five requests cached 1.0-3.3k tokens less (a
small parked conversation was no longer brought back for them) and one 6.0k more. The server's default is 1024, and the
launchers do not change it. Do not raise it to 8192: side requests are tied to it, so titles beside a 5-6k-token
conversation stop being side requests and that conversation is not parked either (on the same sequences: 2.6 times the
cache misses and 15% more tokens prefilled).

### What the console shows

Each generation request logs three lines: what it asks for, its prompt, and its reply. The prompt line says how the
cache was used, and each park, restore, eviction and side request logs a line of its own with tokens, bytes and copy
time. From a run with several clients (native Windows, preset D):

```text
#24 parked the cached conversation (another conversation came back): 12,022 tok, 1,165 MiB copied in 118 ms
#24 restored the parked conversation of 14,031 tok: 1,215 MiB copied in 39 ms | 4 parked, 5,504 MiB in use
#24 prompt: 13,888 cached + 158 prefilled in 0.34 s | restored a parked conversation (14,031 tok); the prompt matches 14,029 of the 14,031 cached tok: resumed at the checkpoint at 13,888
#26 prompt: 9,056 cached + 2,765 prefilled in 1.88 s | the prompt matches 9,126 of the 12,053 cached tok: resumed at the checkpoint at 9,056
#11 side request done: the cached 12,987 tok are back (236 MiB: kept in 90 ms, back in 12 ms)
#20 prompt: 0 cached + 11,894 prefilled in 6.83 s | CACHE MISS: the prompt matches 8,355 of the 40,669 cached tok, no checkpoint before that
```

(Each line starts with `[mk-server] HH:MM:SS`.) `/props` (`engine.host_cache`) lists the parked conversations and their
sizes, and `/metrics` counts parks, restores, evictions and side requests: see [Monitoring](#monitoring).

### Measured on real client traffic

The requests captured from the client test runs ([CLIENTS.md](CLIENTS.md)), replayed against the server without and
with the cache rules above for one-shots, edited histories, compaction leftovers and pinned checkpoints (native
Windows, 230,000-token context, preset D, 8 GiB store; the server generates its own replies, so reply lengths differ
from the live runs):

| replay | without these rules | with them |
|---|---|---|
| 9 single-client sequences: prefill time | 885 s | 800 s |
| 9 single-client sequences: tokens prefilled | 1,267k | 1,150k |
| agent clients' new-session first turns (50 requests): prefill time | 173 s | 120 s |
| Claude Code + Open WebUI interleaved: tokens prefilled / cache misses / real evictions | 508k / 28 / 20 | 489k / 10 / 13 |
| DeepSeek Harness + Aider interleaved: tokens prefilled / cache misses / real evictions | 266k / 32 / 19 | 218k / 7 / 2 |

No request got a 4xx or 5xx. The cost side: a parked conversation with a pin is 288 MiB larger, so the 8 GiB store held
fewer of them (Claude Code's sequence: 4 conversations instead of 6, and two requests cached 4-6k tokens less).

### Turning parts off

| option | effect |
|---|---|
| `MK_EXACT_REUSE=0` (launchers) / no `--exact-reuse` | follow-ups append instead of re-prefilling the last reply: faster, not bit-identical to fresh |
| `--ck-pin 0` | no pinned checkpoints (new sessions re-read the system prompt once the pool has thinned it out) |
| `--park-mib 0` (`MK_PARK_MIB`) | no parking; side requests keep a 1 GiB store |
| `--side-max 0` (`MK_SIDE_MAX`) | no side requests |
| `--side-max 0 --park-mib 0` | the whole host cache off: what a from-scratch benchmark needs (`cache_prompt: false` is accepted and ignored) |
| `--ck-every 0`, `--ck-msgs 0` | fewer checkpoints: more re-prefill after an edit |

An engine library built before a feature existed turns it off by itself, with a console line that says to rebuild.

## Several agents at once: `--agents`

`--agents N` (env `MK_AGENTS`) serves up to N requests at once: several coding agents, or one agent's subagents, on one
GPU. Each request runs in a parallel slot of the engine, a conversation with its own KV cache and recurrent state, and
every slot that is decoding runs in the same engine cycle: one batched launch for all of them. Each agent gets fewer
tokens per second than it would alone, all of them together far more. The engine has 4 slots (N = 1..4). Two or more
slots need a GPU with at least 80 SMs at any number of drafts (the batched passes run 8 rows); the engine refuses fewer
at startup, and 1 slot runs wherever the server does.

```sh
python3 server/mk_server.py -m Qwen3.8-27B-UD-Q4_K_XL.gguf -d 7 --agents 4   # 4 slots, the most context that fits
python3 server/mk_server.py -m Qwen3.8-27B-UD-Q4_K_XL.gguf -d 7 -np 4 -c 16384   # 4 slots of 16,384 positions
MK_AGENTS=4 scripts/serve.sh 8080 0                                        # the launcher: context 0, the most that fits
```

`--parallel N` (`-np N`, llama-server's and open-jet's option) is `--agents N` with `-c` defaulting to 32,768 per slot.

- **Context:** with `--agents`, `-c` is each slot's context, and its default is 0: the most that fits next to
  `MK_SLOT_RESERVE_MIB` [1536] of working memory. On an RTX 3090 (24 GB):

  | model | KV cache | `--agents 2` | `--agents 4` |
  |---|---|---|---|
  | Q4_K_M | q8_0 | 2 x 70,400 | 4 x 32,000 |
  | Q4_K_M | preset D' | 2 x 91,520 | 4 x 39,680 |
  | UD-Q4_K_XL | q8_0 | 2 x 65,280 | 4 x 29,440 |
  | UD-Q4_K_XL | preset D' | 2 x 83,840 | 4 x 35,840 |

  Slots that need more device memory than is free are refused at startup (the error names both; `MK_SLOT_SPILL=1`
  skips the check, and a Windows / WSL2 driver's sysmem fallback then puts the rest in system memory). A prompt that
  does not fit its slot gets the usual 400. The KV cache is q8_0 (`--kv f16` is refused at startup); the presets
  (`MK_KV`, `MK_KV_WIN`) apply to every slot.
- **The same replies:** a greedy reply is the one the request gets when it runs alone on the server, bit for bit,
  whatever runs beside it, its follow-up turns included (one exception, rare: beside a sampled request, a greedy one
  picks among its top candidates after the final norm scale, so two logits within one float step of each other can
  tie there and pick the smaller token id, where alone it compares them unscaled). Each slot decodes as it decodes
  alone, and `--agents` turns on `MK_EXACT` (unless it is set), so a prompt never reuses the rows a slot decoded (their
  bits depend on how many drafts each cycle verified, which depends on the slots beside it): a follow-up that reaches
  into its last reply resumes at a checkpoint inside its last prompt, as with
  [exact reuse](#exact-and-how-it-is-checked) (the scheduler takes the slot back there first: the engine's own
  `slot_begin` would keep the reply's decoded rows, `qwen38_mk.h`). A prompt that arrives while other slots decode is
  fed in chunks beside their cycles; with `MK_EXACT` its rows are those of the same prompt fed in one go (the engine
  computes every chunk but the prompt's last with the batched prefill kernels and ends it on their 32-token scan grid,
  so its rows do not depend on where the prompt is cut; a last chunk of 32 tokens or fewer joins the one before, so
  the prompt keeps its checkpoint).
  Sampled replies (temperature above 0 with a `seed`) are reproducible alone, and beside other
  requests keep their distribution but not their tokens: speculative sampling's tokens depend on the draft schedule,
  and a slot's drafts depend on how many slots share the cycle (an engine limitation). Tested with Q4_K_M across
  OpenAI chat and Anthropic Messages, streamed or not, with and without tools, 2 to 4 at once, an 8k-token prompt
  arriving while three others decode, follow-up turns, a client that disconnects mid-stream, stop strings and
  `max_tokens`: every greedy comparison was identical (45 of 45, also with UD-Q4_K_XL), every sampled reply alone was
  reproducible, and 13 of the 15 sampled replies decoded beside others differed. A 1.6k-token prompt decoded to the
  end of a 2,048-position slot beside three others gave the greedy reply it gets alone, and long prompts (3.8k-5.7k
  tokens) fed in 256-token chunks between other slots' cycles gave the replies of the same prompts fed whole, with q8_0
  and preset D', on Q4_K_M and UD-Q4_K_XL.
- **Scheduling:** requests wait for a free slot in arrival order. A request takes the free slot whose cached sequence
  shares the most with its prompt, less what that discards of the slot's last prompt: a conversation's next turn goes
  back to its own slot, and a new conversation takes an empty slot before it evicts another one for the few template
  tokens they share. While slots decode, a waiting prompt is fed 256 rows per cycle, right before their step
  (`MK_PF_MIX`; 0: `MK_PF_CHUNK` [512] tokens between the cycles, 0 = the whole prompt), so they keep decoding, slower,
  while a long prompt is read; with no slot decoding, a prompt is fed whole (`MK_PF_SOLO`: tokens per call). A stop
  string, `max_tokens`, `t_max_predict_ms` or a client that disconnects takes a request out of the next cycle; a client
  that leaves while its request waits is dropped from the queue, and one that leaves while its prompt is fed in chunks
  stops it at the next chunk.
- **Refused or off:** image and video input and the presence / frequency penalties get a 400 that names `--agents`
  (slot prompts are token ids; slots sample with temperature, top_p, top_k, min_p and seed), and `--mmproj` is ignored.
  The single-conversation cache features are off: side requests, parked conversations, pinned checkpoints and the
  `--ck-*` checkpoints (each slot keeps `MK_SLOT_CK` [2] checkpoints of its own). The keep-warm window starts when every
  slot is idle. Prefill chunks are 1,024 tokens (`MK_PF_CAP`, unless set): more context for the slots.
- **Monitoring:** each request's prompt line names its slot (`slot 2 (after 1.30 s in the queue)`); there is no status
  line. `GET /slots` lists every slot with its state (`idle`, `prefill`, `decode`), its request, prompt, cached and
  decoded tokens and the prompt tokens left (`?fail_on_no_slot=1`: 503 when a new request would wait). `/props` has
  `total_slots` N and an `agents` block in `engine`; in `/metrics`, `requests_processing` counts the requests in slots,
  `requests_deferred` those waiting for one, and `decode_cycles` the engine cycles of all slots together (drafts are not
  counted per slot).

### Speed with several agents

Four agents at once (Q4_K_M, `--agents 4 -d 7 --kv q8`, RTX 3090) decode 424 tok/s in total on short coding prompts and
360 tok/s with a ~23k-token context each; one agent alone ran 194 and 195 tok/s on short prompts in two passes, which
depends on how many sampled drafts it accepts. Exact mode (`MK_EXACT=1`, the `--agents` default) costs nothing measurable,
and open-jet's own server running the same engine library measures the same within run-to-run noise. Numbers and method:
[BENCHMARKS.md](BENCHMARKS.md#parallel-agents).

A prompt that arrives while others decode is fed beside their cycles (`MK_PF_MIX`), so they keep going: a ~20k-token
prompt got its first token after 13.1 s while the other request decoded at 14.1 tok/s. Feeding it whole instead
(`MK_PF_MIX=0 MK_PF_CHUNK=0`) gets the first token about 30% sooner, but the other request all but stops meanwhile
(0.6 tok/s).

## APIs and endpoints

| route | what |
|---|---|
| `POST /v1/chat/completions` (also `/chat/completions`) | OpenAI chat completions, streamed or not |
| `POST /v1/messages` | Anthropic Messages API, streamed or not |
| `POST /v1/responses` (also `/responses`) | OpenAI Responses API, stateless, streamed or not |
| `POST /v1/messages/count_tokens`, `/v1/responses/input_tokens`, `/v1/chat/completions/input_tokens` | the prompt's token count (`input_tokens`), without generating |
| `POST /tokenize`, `/detokenize`, `/apply-template` | llama-server's: token ids, text, the rendered prompt |
| `GET /health` (also `/v1/health`) | 503 `Loading model` until the model is loaded, then 200 `{"status": "ok"}` |
| `GET /v1/models` (also `/models`) | the one model, with llama-server's `meta` (`n_ctx`, file size, `n_embd`) |
| `GET /props` | llama-server's fields plus an `engine` block ([Monitoring](#monitoring)) |
| `GET /slots` | one slot, `is_processing` (`?fail_on_no_slot=1`: 503 while busy); with `--agents`, every slot and its state |
| `GET /metrics` | Prometheus text, always on |

The model is the GGUF the server loaded; its id in `/v1/models` is the file name. The `model` field of a request is
not checked (it shows on the console), so a client may send any name. API keys are checked only when the server has one
(`--api-key`, [Security](#security)); without it, clients that insist on a key can send any value.

llama-server's other routes (`/completion`, `/v1/completions`, `/v1/embeddings`, `/infill`, `/v1/rerank`, audio,
LoRA adapters, `POST /slots` for slot save / restore, `POST /props`) and OpenAI's `/v1/responses/compact` answer 501
naming the route, so a client sees "not supported" rather than a bare 404; any other path is a 404.

### OpenAI chat completions

The reference route: the other two APIs are translated into a chat request and generated by the same code, so they
share the cache, side requests and parking, and a greedy reply through `/v1/messages` is the same as through
`/v1/chat/completions`. Prompts are rendered with the chat template stored in the GGUF (jinja2), with
OpenAI `tools`, `chat_template_kwargs` and the controls in [Request features](#request-features). The reasoning comes
back as `reasoning_content` (streamed too), Qwen's XML tool calls as structured `tool_calls`. `max_tokens`,
`max_completion_tokens` and llama-server's `n_predict` all set the reply's limit; without one, the reply runs to its end
or to the context's. A `developer` message (OpenAI's newer name for a system message) is a `system` message for every
template without GPT-OSS's `<|channel|>`, as llama-server maps it. `usage` reports
`completion_tokens_details.reasoning_tokens`. A reply cut by its limit inside a character whose bytes span tokens ends
before that character.

### Anthropic Messages API

`POST /v1/messages` (streamed or not) and `POST /v1/messages/count_tokens`, as Claude Code and the Anthropic SDKs use
them; the request is translated to a chat request, as llama-server does.

- **`system`**: a string or text blocks, joined by blank lines. Claude Code's per-request billing header `cch=` value is
  normalized as in llama-server, so its long system prompt is reused.
- **Messages** of the roles `user`, `assistant` and `system`. Claude Code 2.1 sends its environment as a system message
  after the first user message; with a template that takes a system message only at the start (Qwen's), it is rendered
  as a user message in its place, where llama-server answers the template's 400. A request that ends with an assistant
  message (Anthropic's prefill) gets it rendered as a finished turn, and the reply is a new turn, not its continuation
  (a one-time console warning).
- **Content blocks:** `text`, `image` (base64), `thinking` (the turn's reasoning), `tool_use`, `tool_result` (tool
  messages, placed before the rest of their user message), `document` with a plain-text or content source (title,
  context and text, as text). Other blocks (PDF documents, server tool blocks, search results, ...) are skipped, as
  llama-server does, with a one-time console warning per kind.
- **`tools`**: each becomes a function with its `input_schema`; Anthropic's typed tools too (`bash_*`, `text_editor_*`,
  `web_search_*`, ...: as in llama-server, a call to one comes back as a `tool_use` block for the client to run).
  `tool_choice` auto / none (any / tool: 400, they need constrained sampling), `disable_parallel_tool_use`.
- **Generation:** `max_tokens` [4096, as llama-server], `stop_sequences`, `temperature` / `top_p` / `top_k`. `thinking`
  `enabled` or `adaptive` turns thinking on; `disabled` or absent turns it off (Anthropic's default); `budget_tokens` is
  accepted but not enforced (the reasoning ends when the model ends it, within `max_tokens`; the console warns once);
  `chat_template_kwargs` win. So `--chat-template-kwargs '{"enable_thinking": ...}'` and Qwen's soft switches do not
  reach `/v1/messages` ([Thinking](#thinking)). Earlier turns' thinking blocks that a client sends back (Claude Code
  does) stay in the prompt: Qwen3.8's template keeps them unless `preserve_thinking` is false
  (`--chat-template-kwargs '{"preserve_thinking": false}'` drops those before the last user message).
- **Effort:** `output_config.effort` (Claude Code's `/effort`) sets the template's reasoning effort while thinking is
  on: `low` and `medium` as named, `high` (what Claude Code sends by default) and `max` as `xhigh`, the template's
  default, so a default session's prompt is the one it gets without the field. `chat_template_kwargs` win.
- **Fields the server does not act on:** `output_format`, `output_config`'s format, `context_management` and
  `container` are accepted with a one-time console warning (clients send some of them on every request): a
  structured-output request gets plain text here, where `response_format` on the chat route is a 400.
- **Replies:** content blocks `thinking` (with an empty signature), `text` and `tool_use` (parsed `input`);
  `stop_reason` `end_turn`, `max_tokens`, `stop_sequence` (with `stop_sequence`) or `tool_use`; `usage` with
  `input_tokens` (prefilled), `cache_read_input_tokens` (reused from the cache) and `output_tokens`.
- **Streams:** `message_start`, `ping`, `content_block_start` / `content_block_delta` (`thinking_delta`, `text_delta`,
  and `input_json_delta` pieces of each tool call's input as it is generated) / `content_block_stop`, `message_delta`,
  `message_stop`; an error after the start is an `error` event.
- **Errors** in Anthropic's shape: `{"type": "error", "error": {"type": "invalid_request_error", ...}}`.

### OpenAI Responses API

`POST /v1/responses` (streamed or not) and `POST /v1/responses/input_tokens`, as Codex CLI uses them. The server is
stateless: no response is stored, so a client sends the whole conversation every time (`store: false`, as Codex does),
and `previous_response_id`, `conversation`, `background`, stored `prompt`s and `item_reference` items are a 400. The
request is translated to a chat request and generated by the same code, with the same reuse, side requests and parking
(console lines `#N responses ...`).

- **`input`**: a string (one user message) or items: messages (roles `user`, `assistant`, `system`, and `developer`,
  read as system; parts `input_text`, `output_text`, `refusal`, `input_image` with a base64 `data:` URL; `input_file`
  and `file_id` images: 400), `reasoning` (its `reasoning_text` content, else this server's `encrypted_content`, else
  its summary: the turn's reasoning), `function_call` / `custom_tool_call` and their outputs (an output may hold images,
  as Codex's `view_image` returns). `instructions` and the leading system / developer messages become one system message
  (their trimmed texts joined by newlines, as Qwen's templates merge them); a later one is rendered as a user message
  in its place. The items of one assistant turn (reasoning, text, calls) become one assistant message, so the turn
  renders as it was generated and turn N's prompt is the start of turn N+1's: Codex's turns and `codex exec resume`
  follow-ups reuse the whole conversation. Other item types (`web_search_call`, ...) are skipped with a one-time console
  warning.
- **`tools`**: `function`; `namespace` (its functions under their plain names; a call to one gets its `namespace` back);
  `custom` (free-form input: offered as a function with one string parameter `input`, and the call comes back as a
  `custom_tool_call`). Tools the server would have to run itself (`web_search`, `file_search`, `mcp`, ...) are not
  offered to the model (console warning; Codex always sends `web_search`). `tool_choice` auto / none (`required`, a
  named or an allowed tool: 400), `parallel_tool_calls`.
- **Generation:** `max_output_tokens`, `temperature` / `top_p` (and the chat request's `top_k`, `min_p`, `seed`,
  penalties, `chat_template_kwargs`), `reasoning.effort` (as `reasoning_effort` below), `text.format` text only (JSON
  schema / object: 400), `include` (`message.output_text.logprobs` and `top_logprobs`: 400). `store`, `metadata`,
  `prompt_cache_key`, `user`, `safety_identifier`, `service_tier`, `truncation` and `max_tool_calls` are accepted;
  `text.verbosity` and reasoning summaries are not generated (console warning).
- **Replies:** a `response` object whose `output` holds `reasoning` (the thinking, trimmed, as `reasoning_text` content;
  `summary` empty; with `include: ["reasoning.encrypted_content"]` also `encrypted_content`, the same text compressed:
  not secret, it brings the reasoning back from clients that return only that), `message` (`output_text`) and
  `function_call` / `custom_tool_call` items; `status` completed, or incomplete with `incomplete_details.reason`
  max_output_tokens; `usage` with `input_tokens` (the whole prompt), `input_tokens_details.cached_tokens` (reused),
  `output_tokens` and `output_tokens_details.reasoning_tokens`. All of a request's `encrypted_content` together may
  inflate to at most what the model's full context could hold (its GGUF context length times its longest token), else
  a 400.
- **Streams:** `response.created` and `response.in_progress` when the prefill starts (a refused request gets its HTTP
  status instead), `response.in_progress` again every 10 s while a long prefill runs (clients drop a silent stream:
  Codex after 300 s), then per output item `response.output_item.added`, `response.content_part.added`, its deltas
  (`response.reasoning_text.delta`, `response.output_text.delta`, `response.function_call_arguments.delta` as the call
  is generated; a custom tool's input whole at its end), the done events and `response.output_item.done`, and last
  `response.completed` (or `response.incomplete`) with the usage; every event has its `sequence_number`. An error after
  the start is `response.failed` (code `server_error`).
- **Errors** in OpenAI's shape: `{"error": {"message": ..., "type": "invalid_request_error", "param": null, "code":
  null}}`.

### Token counts and llama-server's tools

These do not generate and do not wait for a running request (except to encode images): `POST /tokenize` (`content`: a
string, or strings and ids; `with_pieces`, `parse_special` [true] and `add_special` [false, the first string only], as
in llama-server), `/detokenize`, `/apply-template` (the rendered prompt), and the three token-count routes, which render
the prompt exactly as a generation request would (template, tools, images as their tokens) and return
`input_tokens`. Each count logs a console line without a request number (tokens, messages, tools, ms).

## Request features

### Streaming and tool calls

Streams follow each API's event format (above). Tool calls are parsed as the model writes them:

- A parameter's value runs to its `</parameter>`, so it may hold any text, tool-call markup included (a file that
  contains it, as llama.cpp's parser reads it). A call counts only with both closing tags (`</function>`,
  `</tool_call>`): one cut by `max_tokens` or left open is returned as text at the end of `content`, and the reply ends
  with `finish_reason` "length" (complete calls before it are still returned).
- Streams send each call as it is generated, as OpenAI and llama-server do: a `delta.tool_calls` entry with `index`,
  `id`, `type` and `function.name` (`arguments` "") once the function name is complete, then entries with `index` and
  pieces of `function.arguments`. A string parameter's value goes out as it is generated (JSON-escaped); other values
  (schema type not "string"; without a type, a value that starts like JSON) whole at their end, decoded as in the
  non-streamed reply; the closing brace comes with the call's closing tags. The pieces of a complete call add up to the
  non-streamed reply's `arguments` byte for byte (a parameter the model gives twice: both values are sent, and a JSON
  parser keeps the last, as the reply does).
- A call the stream started that is then cut (`max_tokens`, a stop string) or left open keeps arguments that are not
  valid JSON, its markup is not repeated as `content`, and `finish_reason` is "length" / "stop" as without streaming.
  A client should check `finish_reason` / `stop_reason` before it runs a call: an SDK that parses partial JSON
  (Anthropic's stream accumulator) shows such a call as a `tool_use` block with the input parsed so far (`{}`), where
  the non-streamed reply has the markup as text; the OpenAI SDK's `stream()` helper raises `LengthFinishReasonError` on
  "length", as it does against OpenAI.
- Text the model writes after a complete call (between calls or after the last) is not returned: a reply with calls
  carries the calls (Qwen3.8's template tells the model not to write there). The console logs how many characters were
  dropped.
- `parallel_tool_calls: false`: no call is started after the first complete one. `tool_choice: "none"`: the tools stay
  in the prompt and calls come back as text, as in llama-server.

### Thinking

Qwen3.8 thinks before it answers unless told not to. The controls, strongest first:

| control | effect |
|---|---|
| `chat_template_kwargs`: `enable_thinking`, `reasoning_effort`, `preserve_thinking` | handed to the template as given |
| Anthropic `thinking` | `enabled` / `adaptive`: on; `disabled` or absent: off |
| `reasoning_effort` (OpenAI's top-level field; Responses `reasoning.effort`) | `none` turns thinking off, as in llama-server; `minimal` = `low`; `max` / `ultra` (Codex's) = `xhigh`, with a console note; others (`low`, `medium`, `high`, `xhigh`) go to the template |
| Qwen3's soft switches | a last user message (else a leading system message) that ends in `/no_think` or an empty `<think></think>`, or a trailing assistant message that is one, turns thinking off for that request; `/think` turns it on. The prompt keeps the text as sent. Qwen3.8's template ignores them; the server applies them |
| `--chat-template-kwargs '{"enable_thinking": false}'` (`MK_CHAT_TEMPLATE_KWARGS`) | server-wide defaults under every request's own, for clients that cannot send `chat_template_kwargs` (keys: `enable_thinking`, `preserve_thinking`, `reasoning_effort`; others are refused at startup) |
| nothing | the template's default: Unsloth's Qwen3.8 template thinks, at `xhigh`, its highest effort (it knows `low`, `medium` and `xhigh`, and reads `high` as `xhigh`) |

Where the thinking goes: `--reasoning-format` / a request's `reasoning_format` (llama-server's option, `auto` by
default). `auto` and `deepseek` return it as `reasoning_content` (streamed too); `none` leaves it in `content` as
generated (with the `<think>` the prompt opened it with, then `</think>` and the answer); `deepseek-legacy` does both (a
stream sends it in `content` only, as llama-server). `/v1/messages` replies keep their thinking blocks.

### Stop strings

`stop` (a string or a list; at most 64 strings of at most 1,024 characters; empty ones are dropped). As in
llama-server they are matched on the whole generated text, reasoning and tool-call markup included: the reply ends
before the first occurrence, with `finish_reason` "stop" ("tool_calls" when complete calls came before it). A stream
holds back text that could be the start of a stop string until the next token decides it. Tokens the engine accepted in
the same cycle after the stop (MTP drafts) are dropped; the next request continues from the reply that was sent (the
reuse finds the divergence, as after `max_tokens`).

### Sampling

Absent values default to Unsloth's thinking-mode preset for Qwen3.8: `temperature` [1.0], `top_p` [0.95], `top_k` [20],
`min_p` [0]. `top_k` runs 1..32 (<= 0 means 32, the engine's maximum; above 32 is clamped with a one-time warning).
`seed` (absent or negative: a random one per request), `presence_penalty` / `frequency_penalty` (-2..2, vLLM's
semantics: over the tokens generated in this response). `repetition_penalty` is ignored, with a warning. Temperature 0
is greedy, and greedy replies are the same at any draft depth; sampled replies keep the distribution of sampling
without drafts ([BENCHMARKS.md](BENCHMARKS.md) has the tests).

Request fields the engine cannot honour are a 400 naming the field, never ignored silently (`/v1/messages` aside:
[Anthropic Messages API](#anthropic-messages-api)): `n` / `n_cmpl` > 1, `logprobs` / `top_logprobs` / `n_probs`,
`logit_bias`, `response_format` other than `{"type": "text"}`, `json_schema`, `grammar`, `tool_choice` other than "auto"
/ "none" (forcing a call needs constrained sampling), legacy `functions`, `ignore_eos`, audio output,
`add_generation_prompt: false` / `continue_final_message`, llama-server's `id_slot` other than 0 / -1 (there is one
slot) and `n_indent`, and llama.cpp's other samplers (`mirostat`, `typical_p` / `typ_p`, `tfs_z`, `xtc_probability`,
`dry_multiplier`, `top_n_sigma`, `dynatemp_range`, `min_tokens`) unless they have their "off" value (llama.cpp's web UI
sends them all). `samplers` (names, or a string of their letters) is accepted only when it asks for what the engine
does: top_k, top_p, min_p and temperature in llama.cpp's default order (others in their default places, penalties first
when one is set), e.g. the web UI's "edkypmxt". A body that is not a JSON object, non-numeric sampling values and a
missing `messages` array are 400s too, and so is a tool that is not an object whose function (wrapped or flat) is an
object with a non-empty string `name` and `parameters` absent, null or an object (Anthropic: `input_schema`), checked
before the prompt is rendered. A lone UTF-16 surrogate in a request body (a `\ud800`-`\udfff` escape without its pair,
or its raw UTF-8 bytes: JS clients send them when they cut a string inside a pair) is replaced with U+FFFD, as a JS
client's UTF-8 encoder would, and the request runs (a one-time console warning).

Handled by the server: llama-server's `return_progress` (a stream gets `prompt_progress` chunks while the prompt is
processed: `{"total", "cache", "processed", "time_ms"}` with a `{"role": "assistant", "content": null}` delta, at the
start, about every 0.5 s and at the end, as llama-server's web UI shows them), `timings_per_token` (a stream's chunks
carry the `timings` so far) and `t_max_predict_ms` (past it, the reply ends at the next token with a newline,
`finish_reason` "length", as in llama-server). Accepted without effect on the output: `user`, `metadata`, `store`,
`cache_prompt` (the cache is always used), `stream_options`, llama-server's `n_keep` / `n_discard` (there is no context
shift: a prompt too long for the context is a 400), `response_fields` / `return_tokens` (they shape llama-server's
`/completion` replies only), `n_cache_reuse`, `sse_ping_interval`, `verbose`, and other bookkeeping fields.

### Chat template

The template stored in the GGUF renders every prompt; a GGUF without one gets ChatML, as in llama-server.
`--chat-template-file PATH` (the file's text, UTF-8; CRLF line ends are read as LF, as llama-server on Windows reads
them) or `--chat-template TEMPLATE` (the Jinja source itself, or `chatml`) replaces it everywhere: chat and Anthropic
requests, `/apply-template`, token counts and `/props` `chat_template`; the last of the two options given wins
(`MK_CHAT_TEMPLATE_FILE` in the launchers). The server does not start (it exits before it opens the port) when the file
cannot be read, the template does not compile, or it fails on a short conversation (system, user, assistant, user).
Use: a fine-tune of the same model whose GGUF ships Qwen's original template, which answers a second system message, a
turn without a user query or `reasoning_effort` "high" with a 400; Unsloth's template (the one in its Qwen3.8-27B
GGUFs) takes all three. (A `developer` message is a system message under any template without `<|channel|>`.)

### Images

With `--mmproj <mmproj GGUF>` (projector `qwen3vl_merger`, e.g. Unsloth's `mmproj-BF16.gguf`; the launchers pass it when
they find it) the server accepts OpenAI `image_url` parts, Anthropic `image` blocks and Responses `input_image` parts,
base64 only (`data:` URLs; Anthropic's base64 source; no remote URLs), and `/props` reports `modalities.vision`. The
vision tower runs on the CPU, so the encoder takes no GPU memory (an image's tokens take KV cache positions, as text
does).

```sh
IMG=$(base64 -w0 photo.jpg)
curl -s http://127.0.0.1:8080/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "messages": [{"role": "user", "content": [{"type": "text", "text": "What is in this picture?"},
    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,'"$IMG"'"}}]}]}'
```

- **The prompt is llama-server's:** a message whose content is a list of parts is flattened llama-server's way (text
  parts joined by newlines, none next to an image or video), with or without images, so Claude Code's text blocks and
  Codex's `input_text` parts are never run together. Each image becomes `<|vision_start|>`, its rows,
  `<|vision_end|>`, and decoding and preprocessing (stb_image; smart resize to multiples of 32 px within 8 .. 4,096
  tokens, the padded bilinear resize, normalisation) are llama.cpp's bit for bit: the same token counts and pixels as
  `llama-server --mmproj`, except for two images of the same size with no text between them, which llama-server merges
  into one two-frame temporal pair (half the tokens) and this server keeps as two stills, as Hugging Face transformers
  does. EXIF orientation is not applied (as in llama.cpp; Hugging Face's loader applies it): rotate phone photos before
  sending them.
  `--image-min-tokens` / `--image-max-tokens` set the limits (fewer tokens encode faster).
- **Image formats:** PNG, JPEG, GIF (its first frame) and BMP. The other formats stb_image reads (TGA, PSD, HDR, PIC,
  PNM) are refused with a 400: their loaders have unfixed memory bugs a crafted file reaches (GHSL-2023-147, -149).
- **One exception**, where llama.cpp's smart resize breaks its own limit: past an aspect ratio of the token limit
  (4096:1 by default) it clamps the short side up to 32 px after scaling, so a 1,000,000 x 1 PNG of a few KB became
  64,000 tokens (hours of CPU encoding, ~16 GB). Here the long side is cut so that the image gets at most the limit
  (the cpu encoder; video frames too), and the mtmd backend, which resizes inside libmtmd, refuses such an image (400).
  Every image within the limit keeps llama.cpp's size and pixels.
- **Reuse:** an image's placeholder tokens are all alike, so the reused prefix ends at the first image that is not the
  same file (content hash) at the same place. Decoded and encoded images are cached (`--image-cache` [8] images between
  requests; a request holds every image it references until it ends, so it may use more).
- **Encode time** (Ryzen 9 5900XT, 16 cores, BF16 mmproj, steady state): 0.51 s for a 448 x 448 image (196
  tokens), 0.78 s for 640 x 480 (300), 3.56 s for 1024 x 1024 (1,024). The first encode after the load costs ~0.5 s
  more, which the server absorbs with a warm-up encode at startup; a new image after the CPU has been idle takes
  ~1.2-1.5 s for 300 tokens while the cores ramp up. The encoder uses every logical CPU but two (`--vision-threads`):
  its workers meet at a barrier after every step, so one other busy thread on a CPU they use doubles the encode time,
  and the keep-warm load is such a thread (a 220-token frame pair took 1.15 s next to it with all 32 threads, 0.65 s
  without it).
- `--vision-backend auto|cpu|mtmd` picks the encoder; the encoder itself and its accuracy are in
  [ENGINE.md](ENGINE.md#images-vision).

### Video

With `--mmproj` and an ffmpeg (`--ffmpeg` / `MK_FFMPEG`: the executable or its directory; else `ffmpeg` on `PATH`;
else the binary of the `imageio-ffmpeg` Python package, which bundles one for Windows, Linux and macOS) the server
takes video clips: llama-server's `{"type": "input_video", "input_video": {"data": <base64>}}` and vLLM-style
`{"type": "video_url", "video_url": {"url": "data:video/mp4;base64,..."}}` (no remote or file URLs). The containers
read are MP4 / MOV (3GP, M4V), Matroska / WebM, AVI, MPEG-TS, MPEG-PS, Ogg, FLV and GIF (ffmpeg's `-format_whitelist`),
with any codec the ffmpeg reads; anything else is a 400, including DASH manifests, HLS playlists and concat lists, which
would make ffmpeg open other files or URLs. `/props` `modalities.video` says whether video works in this process;
without ffmpeg, or with an older vision library, a video request gets a 400 naming what is missing. There is no audio.

```sh
VID=$(base64 -w0 clip.mp4)
curl -s http://127.0.0.1:8080/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "messages": [{"role": "user", "content": [{"type": "text", "text": "What happens in this video?"},
    {"type": "input_video", "input_video": {"data": "'"$VID"'"}}]}]}'
```

- **Frames:** ffmpeg decodes the clip (written to a temporary file, read with the file protocol only) through its `fps`
  filter at `--video-fps` [4, llama.cpp's default] into RGB24, as llama.cpp's mtmd-helper does (`-vf fps=4.000000
  -pix_fmt rgb24`), read as a PPM stream so that no ffprobe is needed: the same frames, byte for byte, as llama.cpp's
  command. Each frame is resized like an image; `--video-max-tokens` caps a frame's tokens (default: the image limit).
  A decode that takes over 300 s, or a clip ffmpeg cannot read, is a 400.
- **Prompt, as llama.cpp b10246 builds it:** the clip becomes the text `Video:`, then the frames, with a timestamp
  text after each frame that reaches the next 5 s mark (`[0m0.00s]` after frame 0, `[0m5.00s]` after the frame at 5 s,
  ...). Consecutive frames merge in pairs into one grid of tokens (Qwen-VL's temporal patch of 2): frame 0 stands alone,
  and a run between two timestamps with an odd number of frames ends with a lone frame. Checked against llama.cpp's own
  path on 4 clips (H.264, VP9, 12 to 40 frames): the same texts and grids, and llama-server reported the same prompt
  token counts. This is llama.cpp's layout, not the model's own (Hugging Face's processor puts a `<t seconds>` text
  before each frame pair, at 2 fps).
- **Reuse:** each frame or pair is an image to the reuse logic (its key: the clip's hash, the frame rate, the token
  limit and its place), so a follow-up turn reuses the whole clip. Decoded and encoded clips are cached
  (`--video-cache` [2] clips between requests); a request holds all of its own. A client that disconnects stops the
  encodes between pairs.
- **Limits:** at most `MK_MAX_VIDEO_FRAMES` [256] frames per request (all of its videos; 64 s at 4 fps), and frames of
  more than `MK_IMAGE_MAX_PIXELS` pixels are refused from their header ([Limits](#limits)).

Cost (Ryzen 9 5900XT, BF16 mmproj, 30 threads, steady state; tokens per second of video include each frame
or pair's `<|vision_start|>` / `<|vision_end|>` and the timestamps):

| frames | tokens per pair | encode per pair | tokens per second of video at 1 / 2 / 4 fps |
|---|---|---|---|
| 320 x 240 | 80 | 0.22 s | 51 / 85 / 167 |
| 640 x 360 | 220 | 0.56 s | 135 / 228 / 450 |
| 1280 x 720 | 920 | 3.1-3.7 s | 555 / 939 / 1,861 |

At 1 fps a clip costs more than half its 2 fps tokens: every 5 s run then has an odd number of frames, and its last
frame is a grid of its own. A 640 x 360 clip at 4 fps encodes at about real time.

**Live feeds** need no extra API: cut the feed into short segments and append each as a new user message (with the
stream time as a text part) to the conversation you re-send; send "watching" requests with `max_tokens: 1` and drop
their one-token reply, and ask a question whenever you want an answer. Each request then encodes and prefills only the
new segment (the earlier ones are reused up to the checkpoint at the generation prompt), so a tick costs the same at
any stream age. Set `--video-cache` to at least the number of segments in the conversation, or every request decodes
the earlier segments again; cached segments keep their rows (~20 KB per token, bounded by the context). When the
context fills, start a new conversation from a summary plus the latest segments (one prefill, no re-encode). Tested
with 3 s segments of 480 x 272 at 4 fps: 10 ticks of 3.7-4.2 s each (3.0 s of encoding), and the model placed a color
change at the right time after 30 s of feed.

## Keep-warm

The driver lowers the GPU's clocks after a few seconds without work, and a reply that starts from there runs its first
~2 s at ramping clocks. After each request (and at startup) the engine therefore keeps a light load on the GPU, a
one-block polling kernel relaunched every 50 ms, for `MK_KEEP_WARM_S` seconds [120; 0 = off]; the next request stops it
within microseconds. The driver keys its power state to utilization, so only a continuous load holds full clocks; the
driver's power settings are left alone. On an RTX 3090: ~165 W while warm against 30-35 W idle; from
idle the clocks take 2.2-2.7 s to come up. An image the request encodes on the CPU re-arms it first, so the GPU stays
warm until the prefill's first launch. `MK_KEEP_WARM_LOAD` changes the load (for measurements;
[CONFIGURATION.md](CONFIGURATION.md)).

## Monitoring

### The console

Every line on stderr is `[mk-server] HH:MM:SS ...`. A generation request is numbered (`#12`) and logs three lines:

- **request:** route, stream, the `model` field, prompt tokens, messages, tools, images / videos, thinking,
  `max_tokens`, the sampling used;
- **prompt:** `N cached + M prefilled in s (prefill tok/s)` and how the cache was used (continued, resumed at a
  checkpoint, or `CACHE MISS` with how much of the cached sequence the prompt matched; after a side request or a
  restored conversation when one was used); images and their CPU encode time;
- **reply:** `tok in s = tok/s` over the decode (from the first token to the end of the reply), the tokens of the
  thinking / answer / tool calls, `tok/cycle` (tokens per engine cycle: 1 plus the accepted MTP drafts), the share of
  drafts accepted, the finish reason and the whole request's time. A one-token reply, made by the prefill, shows no
  rate.

Each park, restore, eviction and side request logs a line of its own ([examples](#what-the-console-shows)). A request
that waits for the engine, 4xx answers and disconnects are logged with the request number; each token count logs a line
without one. On a terminal, one status line under the log is rewritten in place (a carriage return, no escape sequences)
while a request runs: image encodes, the prefill's progress with its rate and the time left, then the decode's tokens,
tok/s and tok/cycle. Written to a file, only the log lines appear. A client resetting an idle keep-alive connection is
not logged.

### `/props`, `/slots`, `/v1/models`

`GET /props` has llama-server's fields: `default_generation_settings` (`n_ctx`, the sampling defaults), `total_slots` 1
(`--agents`: N),
`model_alias`, `model_path`, `modalities` (vision, video, audio), `chat_template`, `build_info` (the commit and engine
flags of the engine library, written by the Makefile and `build.ps1`; "unknown" for libraries built otherwise). Its
`engine` block holds what the server knows for sure: context, the KV format asked for (`auto` is resolved inside the
engine), MTP layer and draft depth, draft vocabulary, exact reuse, the checkpoint settings (pool, every, messages,
pinned, slots), the keep-warm window, `host_cache` (budget, side-request and parking settings, the parked conversations
and their token counts, how many are side conversations or low priority, whether side conversations are kept, host
memory in use and allocated), the vision settings, the limits, the `MK_*` environment (`MK_API_KEY` as `(set)`, not
the key), busy and uptime. Some clients (Hermes Agent) read the context size from it. `GET /slots`: one slot with
`is_processing`. `GET /v1/models`: the model with `meta`.

### `/metrics`

Prometheus text, always on, with llama-server's `llamacpp:` names where they mean the same and `megakernel:` names for
the rest:

| metric | meaning |
|---|---|
| `llamacpp:prompt_tokens_total`, `llamacpp:prompt_seconds_total` | prompt tokens prefilled (reused ones not) and their time |
| `llamacpp:tokens_predicted_total`, `llamacpp:tokens_predicted_seconds_total` | generated tokens and their time |
| `llamacpp:prompt_tokens_seconds`, `llamacpp:predicted_tokens_seconds` | average prefill and decode throughput |
| `llamacpp:requests_processing`, `llamacpp:requests_deferred` | the running request; requests waiting for the engine |
| `megakernel:prompt_tokens_reused_total` | prompt tokens reused from the cache |
| `megakernel:decode_cycles_total`, `megakernel:draft_tokens_total`, `megakernel:draft_tokens_accepted_total` | engine cycles, MTP drafts made and accepted |
| `megakernel:requests_total`, `..._bad_total`, `..._failed_total`, `..._cancelled_total` | generation requests: received, answered 4xx, failed 5xx, client disconnected |
| `megakernel:side_requests_total`, `megakernel:side_requests_dropped_total` | side requests; those after which the cached conversation was lost |
| `megakernel:side_conversations_parked_total`, `megakernel:side_conversations_resumed_total` | side conversations kept; continued as side requests |
| `megakernel:parks_total`, `megakernel:park_restores_total`, `megakernel:park_evictions_total` | conversations parked, brought back, evicted |
| `megakernel:parked_conversations`, `megakernel:host_cache_bytes` | parked conversations now, and their host memory |

## Errors, cancels and disconnects

- **Engine errors:** a CUDA error or non-finite logits make the engine throw; the server answers HTTP 500 and keeps
  serving. In a stream that has started (status 200 sent), the error is the last event, OpenAI style: `data: {"error":
  {"message": ..., "type": "server_error", "code": 500}}`, then `data: [DONE]` (Anthropic: an `error` event; Responses:
  `response.failed`). Errors carry llama-server's `code` (the HTTP status) next to `message` and `type`.
- **A prompt longer than the context** (the reply needs 16 tokens of it) is a 400 in llama-server's machine-readable
  shape, `{"error": {"code": 400, "type": "exceed_context_size_error", "message": "request (N tokens) exceeds the
  available context size (C tokens), try increasing it", "n_prompt_tokens": N, "n_ctx": C}}`: clients that compact or
  trim key on the type and the two numbers. With images the counting stops at the first image past the context ("at
  least N tokens"), and no more images are decoded once the prompt cannot fit. On `/v1/messages` it is Anthropic's
  `invalid_request_error` "prompt is too long: N tokens > M maximum" (the text Claude Code reads before it compacts;
  M = C - 17) with the same two fields; on `/v1/responses`, OpenAI's `invalid_request_error` with code
  `context_length_exceeded` and the two fields. Streams get it as the status, before the stream starts.
- **Disconnects:** before the prefill and before each decode cycle the server checks whether the client closed its
  connection, and stops. A client that stops reading its reply (streamed or not) counts as a disconnect once a write has
  made no progress for 60 s ([Limits](#limits)). During the prefill a watcher thread checks every 50 ms and asks the
  engine to stop at its next prefill chunk (a host flag read between chunks of 2,048 tokens, no device sync); the chunk
  running and the one queued behind it finish, so the engine is free 1-2 chunks later. Nothing of a stopped prefill
  counts: the engine drops the checkpoints it saved and goes back to the last checkpoint from before it (or resets),
  with the draft vocabulary as it was at that position, so the next request answers as on a server that never saw it. A
  client that leaves during decoding leaves what was generated in the cache, as after a reply cut by `max_tokens`, and
  the next request reuses or rewinds it as usual. A disconnect during a side request brings the kept conversation back
  like a finished one. (On Windows an aborted client connection raises a different error than on Linux; both are counted
  as disconnects.)
- **Closing:** when the server ends a connection while the client may still be sending (a 400 / 411 / 413 for a body it
  does not read), it shuts down its write side, then reads and drops what the client still sends (at most 1 GiB and
  30 s, 5 s without data, as nginx's lingering close) before closing. Closing with unread data makes the kernel send a
  reset, and a client that sends its whole body before reading (Python's http.client, curl without `Expect:
  100-continue`) would then see a connection error instead of the answer.

## Security

The server is meant for one machine or a trusted LAN. It checks an API key only when given one; what it always checks
is that a web page you happen to open cannot use it.

- **API key** (`--api-key KEY`, env `MK_API_KEY`; default none): every request but `/health` and CORS preflights must
  carry the key, as `Authorization: Bearer KEY` (the OpenAI clients' `api_key`) or `x-api-key: KEY` (the Anthropic
  clients'); others get a 401 with `WWW-Authenticate: Bearer` (llama-server's error shape; Anthropic's on
  `/v1/messages`) before the body is read. Prefer the environment variable: an option shows in the process list. Over
  plain HTTP the key crosses the network in clear text: beyond a trusted LAN, put a TLS reverse proxy in front. A server
  listening on anything but a loopback address without a key says so in a warning at startup: anyone who can reach the
  port can use the model and read its replies.

- **Host names (DNS rebinding):** a request whose `Host` is not `localhost`, an IP address, the `--host` name or an
  `--allow-host` name (any port; repeatable or comma-separated, env `MK_ALLOW_HOSTS`) gets a 403 naming the option. A
  web page that points its own domain at this machine would otherwise be same-origin with the server and could read
  `/props`, `/metrics` and the replies. `--allow-host '*'` accepts any name (e.g. behind a reverse proxy) and turns this
  check off. Reaching the server by its machine name from the LAN needs that name in `MK_ALLOW_HOSTS`.
- **Origins (other sites' pages):** a request with an `Origin` header (browsers send one with every POST and every
  cross-origin request) is answered only when that origin is the server's own (Origin's host and port equal `Host`'s)
  or listed with `--cors-origin` (repeatable or comma-separated, env `MK_CORS_ORIGINS`; default none; `*` is refused:
  llama-server reflects any origin, this server does not). Pages of other sites, of another port on this machine (any
  other local web app) and `file:` pages (`Origin: null`, unless `null` is listed) get a 403, also for the simple
  requests (a `text/plain` POST) a page may send without a preflight, so a page cannot start a long prefill, evict the
  cached conversation or read anything. The 403 comes before the body is read and the connection closes.
- **CORS:** a listed origin gets `Access-Control-Allow-Origin` + `Vary: Origin` on every response (errors, streams and
  the 503 while loading too), and its preflights (`OPTIONS`) are answered 200 with `Access-Control-Allow-Methods: GET,
  POST, OPTIONS`, the headers it asked for and `Access-Control-Max-Age: 86400` (Chrome's
  `Access-Control-Request-Private-Network` too). Use it for a browser UI served from another origin, e.g.
  `--cors-origin http://localhost:5173`. Clients that are not web pages (curl, the OpenAI / Anthropic SDKs, coding
  agents, Open WebUI's backend) send no `Origin` and are not affected; the `Content-Type` is not checked (`curl -d`
  without `-H` sends a form type), as the Origin check already stops pages.
- **Special-token text in content** (`--content-specials text|parse` [text], env `MK_CONTENT_SPECIALS`): the text of a
  control token (`<|im_end|>`, `<|im_start|>`, `<|endoftext|>`, `<|vision_start|>`, ...: the vocabulary's control
  tokens) inside messages of any role (user, system, tool results, assistant replies and their reasoning), tool-call
  arguments and tool definitions is tokenized as plain text, while the special tokens the template writes stay special.
  So a file an agent reads (this repository's own sources contain `<|im_end|>`), a web page or a tool result cannot
  forge a turn (`...<|im_end|>\n<|im_start|>system\n...`). llama-server (and `parse`) tokenizes the whole rendered
  prompt with special tokens parsed, content included.
  - How: the template renders the request as usual (the prompt text is unchanged: `/apply-template`, token counts), and
    once more with each such text replaced by a private-use character; when that render, with the texts put back, is
    the same text, the prompt is tokenized piece by piece between the template's control tokens, a piece holding
    content's special-token text as plain text (one run of text, as llama.cpp tokenizes the text between special
    tokens), every other piece as before. A template that renders the marked request differently keeps `parse` for it
    (logged once). User-defined tokens (`<think>`, `</think>`, `<tool_call>`, `<tool_response>`) are tokens in any
    text, as in llama.cpp and Hugging Face tokenizers.
  - Prompts without such text in their content get exactly the token ids they got before (checked on 237 requests: the
    test suite's, the repro and API test requests, Claude Code's and OpenCode-shaped agent sessions of 10-25k tokens).
    Assistant turns are content too: a reply that quoted `<|im_end|>` as text comes back as the same text tokens, so
    exact reuse keeps it; only a control token the model emitted mid-reply (none of the end-of-turn ones: those end the
    reply) would re-prefill from there.
- **Bodies that are not read:** only POST bodies are read. Any other request with a body (a `Content-Length` other than
  0, or a `Transfer-Encoding`), e.g. a `GET`, is answered with `Connection: close` and the connection closes, so the
  body is never taken for the next request (behind a reverse proxy: a request the proxy never saw).
- **Untrusted uploads:** the limits below refuse oversized bodies, images and video frames before they are decoded.

## Limits

Read from the environment at startup ([CONFIGURATION.md](CONFIGURATION.md)):

| limit | default | answer |
|---|---|---|
| request body, `MK_MAX_BODY_MB` | 128 MB | 413, refused before the body is read (also on `Expect: 100-continue`) |
| request bodies in memory at once, all connections, `MK_MAX_INFLIGHT_MB` | 256 MB (at least `MK_MAX_BODY_MB`) | 503 with `Retry-After`, and the connection closes |
| a connection's read or write without progress, `MK_TIMEOUT_S` | 600 s (llama-server's `--timeout`; 0: none) | the connection closes: an idle keep-alive one, a stalled request, or a client that stopped reading a reply (a disconnect: the generation stops) |
| a reply's write without progress, streamed or not (fixed) | 60 s, or `MK_TIMEOUT_S` when that is shorter and not 0 | the client counts as gone, as after a disconnect: the generation stops and the engine is free for the requests waiting for it |
| a request's arrival, first byte to the last of its body, `MK_REQUEST_TIMEOUT_S` | 300 s (0: none) | the connection closes. A client sending a byte at a time never hits `MK_TIMEOUT_S`, but hits this; generating the reply does not count |
| open connections, `MK_MAX_CONNECTIONS` | 256 | 503 with `Retry-After` as the connection is accepted, and it closes |
| images per request, `MK_MAX_IMAGES` | 32 | 400 |
| pixels per image or video frame, `MK_IMAGE_MAX_PIXELS` | 100,000,000 | 400, checked from the header before decoding |
| video frames per request, `MK_MAX_VIDEO_FRAMES` | 256 (64 s at 4 fps) | 400 as soon as the frame past it is decoded |
| stop strings | 64, of at most 1,024 characters | 400 |

- **Bodies:** a request body needs a `Content-Length` (411 without one; 400 when malformed) or `Transfer-Encoding:
  chunked` (Node's `fetch` with a stream body, Go clients, proxies that do not buffer; as llama-server). Chunks are
  decoded into a buffer of at most `MK_MAX_BODY_MB` (413 as soon as a chunk would pass it); extensions and trailers are
  dropped; a malformed size line (not hex, over 16 digits, no CRLF within 4,096 bytes), chunk data without its CRLF or
  more than 64 trailer lines is a 400 that closes the connection; another transfer coding is a 501, and a
  `Content-Length` next to it a 400. A refused upload still gets its answer (the server drains the rest, bounded).
- **Bodies in memory:** a body counts as it arrives, not by the size it declares, until its request ends, queued or
  running. Parsed JSON takes up to 24 times the bytes (`[[],[],...]`, measured; text about once), so the default keeps
  all requests together under about 6 GB of Python objects; ten bodies of 128 MB at once would be 30 GB.
- **What the connection limits are for:** they keep the process alive, they do not keep a port open to strangers usable.
  The API key is checked once the headers have arrived, so anyone who can reach the port can hold connections up to
  `MK_MAX_CONNECTIONS` until they time out.
- **Why the pixel limit:** without it, a 446 KB PNG declaring 12000 x 12000 pixels took 39 s and
  0.9 GB of server memory (decode, then 4,096 tokens encoded); a 20000 x 20000 one asks stb_image for ~2.4 GB. For
  video, ffmpeg does not decode frames past the limit plus 8 Mpx (its `-max_pixels`; the slack covers its decoders' row
  alignment; its default takes frames up to 2^31 pixels, so a clip of a few hundred KB declaring 40,000 x 40,000 frames
  means 4.8 GB of RGB per frame). Without the header check, a clip of 11000 x 10000 frames took 5.2 s, 0.66 GB in the
  server and 2 GB in ffmpeg before its frames were refused; with it, 0.05 s and ~50 MB.
- **Context:** no more images are decoded once the prompt cannot fit the context (see the 400 above).

## Engine calls behind the server

The server loads the engine library (`libqwen38mk.so`, `qwen38mk.dll` on Windows; `--lib`) through ctypes and uses its
C API (`capi.cpp`). Where each feature lives, for reading the code:

| feature | C API | engine (`MKEngine`) |
|---|---|---|
| prompt reuse, checkpoints | `mk_prefill_cks`, `mk_prefill_img`, `mk_checkpoints`, `mk_rewind_to`, `mk_truncate`, `mk_sequence`, `mk_committed` | `MKEngine::prefill` and the checkpoint pool |
| exact reuse | `mk_exact` (`MK_EXACT=1` is read when the engine is built) | the prefill's exact attention form |
| pinned checkpoints | `mk_ck_pin`, `mk_ck_pinned` | `MKEngine::ck_pin` |
| parking | `mk_park`, `mk_unpark`, `mk_park_drop`, `mk_park_need`, `mk_store_budget`, `mk_store_info`, `mk_kv_sizes` | `MKEngine::park` / `unpark`, the host store of pinned chunks |
| side requests | `mk_side_begin`, `mk_side_end`, `mk_side_drop`, `mk_side_active`, `mk_side_end_park`, `mk_side_unpark` | `MKEngine::side_begin` / `side_end` / `side_end_park` / `side_unpark` |
| stopping a prefill | `mk_cancel` (a host flag read between prefill chunks, no device sync) | `MKEngine::cancel` |
| the console's prefill progress | `mk_prefill_progress` (host counters per chunk, no device sync) | `MKEngine::prefill_progress` |
| keep-warm | `mk_keep_warm` | `capi.cpp` |
| `--agents` | `mk_slots_open2`, `mk_slot_begin2`, `mk_slot_prefill`, `mk_slots_step`, `mk_slots_step_mixed` (open-jet's slot API) | `MKEngine::slots_open`, the lanes and their batched launch |
| `/props` `build_info` | `mk_build_info` (from `mk_build_info.h`, which the Makefile and `build.ps1` write) | - |
| special-token text as text | `mk_tokenize2`, `mk_token_attr` | - |
| the reuse probe | `mk_state_digest` | - |

A library built before one of these existed lacks the export: the server then turns that feature off with a startup
line that says to rebuild (with `--mmproj`, a library without `mk_prefill_img` stops it from starting).

## Windows

- `scripts\serve.bat` takes the same settings as `scripts/serve.sh` (`set MK_KV=K*=6,V*=5` and `set MK_KV_WIN=2048` for
  preset D, `MK_HOST=0.0.0.0` for the LAN, `MK_MMPROJ=off` for no images); images are on when `mmproj-BF16.gguf` sits
  next to the model or `MK_MMPROJ` names it. Set `MK_PYTHON` to a Python with the server's requirements.
- The Windows build gives the same replies as the Linux build, bit for bit: text, thinking, images and follow-up turns.
  Every release is checked for it, side-request and parked-conversation gates under q8_0 and preset D included
  ([BENCHMARKS.md](BENCHMARKS.md#windows-and-linux)).
- The port is bound exclusively, so a second server cannot share it silently: it fails at startup with a message.
- A client inside WSL2 with NAT networking does not reach a server running natively on Windows at `127.0.0.1`; run the
  client on Windows, or the server inside WSL2.
- Memory: a native server with the 8 GiB store took ~35 GB of Windows commit charge during the client replays
  (the model's host copy, the store and the checkpoint pool), so leave room for it next to other large
  programs.
