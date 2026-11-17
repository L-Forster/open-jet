# Configuration

Every setting of the build, the launchers, the server and the engine: what it does, its default, what the served
configuration uses, and whether it changes the output. Most people need only the first two sections; the rest is
reference.

The list is checked against every name the code reads, so a setting that is missing here is a documentation bug.
Measurements with their dates and conditions are in [BENCHMARKS.md](BENCHMARKS.md).

- [How settings reach the engine](#how-settings-reach-the-engine)
- [You probably only need these](#you-probably-only-need-these)
- [KV cache presets](#kv-cache-presets)
- [Launchers: scripts/serve.sh and scripts/serve.bat](#launchers-scriptsservesh-and-scriptsservebat)
- [Server options](#server-options)
- [Engine environment variables](#engine-environment-variables)
- [Compile-time flags](#compile-time-flags)
- [The CLI: qwen38_mk](#the-cli-qwen38_mk)
- [Tools and benchmarks](#tools-and-benchmarks)
- [Measurement, test and debug switches](#measurement-test-and-debug-switches)
- [Not settings](#not-settings)

## How settings reach the engine

There are four layers. Each one passes its settings down to the next:

1. **Build** (`make` / `build.ps1`): compile-time flags (`MKFLAGS`, `-D...`) decide which kernels exist in the binary
   ([Compile-time flags](#compile-time-flags)).
2. **Launchers** (`scripts/serve.sh`, `scripts/serve.bat`): read `MK_*` environment variables, turn some of them into
   server options, and start the server with the rest of the environment unchanged.
3. **Server** (`server/mk_server.py`): command-line options. Some have an environment twin (`--side-max` /
   `MK_SIDE_MAX`); an option given on the command line wins over its twin. The host-name and CORS lists are the
   exception: `MK_ALLOW_HOSTS` / `MK_CORS_ORIGINS` and the options are combined.
4. **Engine** (`libqwen38mk.so` / `qwen38mk.dll`, also inside the CLI `qwen38_mk`): reads `MK_*` environment variables
   when the model loads. The server's `--exact-reuse` works by setting `MK_EXACT=1` for it.

A running server shows what it uses in its `/props` reply: the build's commit and `MKFLAGS` (`build_info`), the
resolved server settings, and every `MK_*` variable of its environment (`engine.env`).

Two columns appear in most tables:

- **Output**: `same` = the output is bit-identical with the setting on or off (checked by the gates named in the row,
  or by construction where said so); `drafts only` = only the drafts change, so greedy output is identical and sampled
  output keeps its distribution (the sampled tokens themselves can differ); `changes` = the output differs (another
  arithmetic order, or a lossy format); `-` = no effect on generation; `not checked` = no record either way.
- **Status**: `served` (what the launchers and the served build use), `option` (supported, off by default or a value
  to tune), `opt-in` (experimental, off by default), `slower` (measured slower, kept for A/B), `earlier path` (the
  code path before a change, kept for A/B), `debug` (tests, measurement, diagnostics).

## You probably only need these

The launchers start the server with 120,000 tokens of context, the q8_0 KV cache, 7 drafts, exact reuse on, and images
on when the vision projector is found. These are the settings worth knowing:

| What | Launcher variable (`serve.sh` / `serve.bat`) | Server option | Default | When to change it |
|---|---|---|---|---|
| Model file | `MK_MODEL` | `-m` | `$HOME/models/Qwen3.8-27B-UD-Q4_K_XL.gguf` (Windows: `%USERPROFILE%\models\...`) | another GGUF or another folder |
| Port, context | 1st and 2nd argument | `--port`, `-c` | 8080, 120000 | context up to 262,144 tokens with preset D or F (`serve.bat` picks F above 240k) |
| KV cache preset | `MK_KV`, `MK_KV_WIN` | (engine) | q8_0 | [preset D](#kv-cache-presets) frees ~1.1 GB at 120k and fits the full 262,144-token context; near-lossless, not bit-identical. Above 240k `serve.bat` uses preset F unless `MK_KV` is set |
| Drafts per cycle | `MK_DRAFT` | `-d` | 7 | 0 turns speculation off; 8-15 only with the opt-in [deep-pass build](EXPERIMENTAL.md#the-deep-pass) and no `MK_KV` preset (so not above 240k with `serve.bat`, which picks preset F there) |
| Listen address | `MK_HOST` | `--host` | 127.0.0.1 | `0.0.0.0` for other machines on the LAN (then also `MK_ALLOW_HOSTS`) |
| Host names, web pages | `MK_ALLOW_HOSTS`, `MK_CORS_ORIGINS` | `--allow-host`, `--cors-origin` | none | a LAN name or a reverse proxy; a browser UI on another origin |
| API key | `MK_API_KEY` | `--api-key` | none | other machines on the LAN: clients then send it as their API key (listening beyond this machine without one: a warning at startup) |
| Images and video | `MK_MMPROJ` | `--mmproj` | `mmproj-BF16.gguf` next to the model | another projector file, or images off |
| Exact reuse | `MK_EXACT_REUSE=0` turns it off | `--exact-reuse` | on | off: a follow-up turn appends to the previous reply instead of re-reading it (faster, but not bit-identical to the same request sent fresh) |
| Keep-warm | `MK_KEEP_WARM_S` | (engine) | 120 s | 0: no extra draw after a request (an RTX 3090 held ~165 W instead of 30-35 W), but a reply after a pause starts at low clocks |
| Parked conversations | `MK_PARK_MIB` | `--park-mib` | 8192 MiB of host memory | less RAM to spare, or more conversations to keep |
| Side requests | `MK_SIDE_MAX` | `--side-max` | 8192 tokens | 0 turns them off (benchmarks of from-scratch prefill) |
| Smallest parked conversation | (not passed) | `--park-min` | 1024 tokens | 4096 halved the real evictions in a replay of recorded client sessions |
| Several agents at once | `MK_AGENTS` (with context `0`) | `--agents N` | off | serve up to 4 requests at once, each in a slot of its own context ([SERVER.md](SERVER.md#several-agents-at-once---agents)) |
| Prefill chunk | `MK_PF_CAP` | (engine) | 2048; the launchers set 1024 above 96k context, `serve.bat` 512 above 240k with your own `MK_KV` | 1024 frees ~0.34 GB of VRAM |
| Thinking off by default | `MK_CHAT_TEMPLATE_KWARGS` | `--chat-template-kwargs` | thinking on | `{"enable_thinking": false}` |
| Python | `MK_PYTHON` | - | `python3` (Windows: `python`) | a virtual environment with jinja2 |

Running `server/mk_server.py` directly instead of a launcher: its own defaults are more conservative (port 18080,
context 32768, 4 drafts, `--kv auto`, exact reuse off, no images unless `--mmproj`). More than 4 drafts needs the q8_0
cache, which `--kv auto` picks only when fp16 does not fit or `MK_KV` is set, so pass `--kv q8` with `-d 7`.

### Examples

Linux / WSL2:

```sh
scripts/serve.sh                                                # 127.0.0.1:8080, 120k context, q8_0, 7 drafts
scripts/serve.sh 8081 64000                                     # port 8081, 64k context
MK_KV='K*=6,V*=5' MK_KV_WIN=2048 scripts/serve.sh 8080 262144   # preset D, the model's full context
MK_HOST=0.0.0.0 MK_ALLOW_HOSTS=gpu-box scripts/serve.sh         # the LAN, reached as http://gpu-box:8080
MK_CHAT_TEMPLATE_KWARGS='{"enable_thinking": false}' scripts/serve.sh   # thinking off unless a request asks
MK_PARK_MIB=4096 MK_KEEP_WARM_S=0 scripts/serve.sh              # less host memory, no keep-warm
```

Windows (`cmd`; `set` keeps the value for the rest of the window):

```bat
REM the model and a Python with jinja2, then the defaults
set MK_MODEL=D:\models\Qwen3.8-27B-UD-Q4_K_XL.gguf
set MK_PYTHON=%CD%\.venv\Scripts\python.exe
scripts\serve.bat

REM the model's full context: above 240k serve.bat uses preset F unless MK_KV is set
scripts\serve.bat 8080 262144

REM thinking off unless a request asks (the quotes around the whole assignment keep the inner ones)
set "MK_CHAT_TEMPLATE_KWARGS={"enable_thinking": false}"
```

The server directly, with the launchers' settings plus two that the launchers do not pass:

```sh
python3 server/mk_server.py -m Qwen3.8-27B-UD-Q4_K_XL.gguf --port 8080 -c 120000 --kv q8 -d 7 \
    --exact-reuse --mmproj mmproj-BF16.gguf --park-min 4096 --vision-threads 8
```

## KV cache presets

The launchers' KV cache is q8_0 (8 bits per value, one fp16 scale per 32); `--kv auto`, the CLI's default, picks fp16 when it
fits and q8_0 otherwise. `MK_KV` stores chosen layers'
K and V at fewer bits, and `MK_KV_WIN` keeps the most recent positions at 8 bits in a ring on top. The presets are
opt-in and near-lossless: measured by KL divergence against llama.cpp's reference distributions, not bit-identical to
the q8_0 cache. With C, D and D', a prompt that fits in the window reads only q8_0 rows, and the short-prompt KL report
is byte-identical to q8_0's; E and F store rotated rows (F only K), which changes it slightly. A
[deep-pass build](EXPERIMENTAL.md#the-deep-pass) runs the presets at up to 7 drafts only: its drafts 8-15 need the
plain q8_0 cache (the deep pass is in active development; low-bit support is on the list).

| Preset | Settings | K / V bits |
|---|---|---|
| C | `MK_KV='V*=5' MK_KV_WIN=2048` | 8 / 5 |
| **D** | `MK_KV='K*=6,V*=5' MK_KV_WIN=2048` | 6 / 5 |
| D' | `MK_KV='K*=6,V*=5,*16=8' MK_KV_WIN=2048` | 6 / 5, the MTP layer at 8 / 8 |
| E | `MK_KV='K*=6r,V*=4r' MK_KV_WIN=2048` | 6 / 4, both stored Hadamard-rotated |
| F | `MK_KV='K*=5r,V*=5' MK_KV_WIN=2048` | 5 / 5, K stored Hadamard-rotated; `serve.bat` above 240k |

C, D and E side by side (what each frees at 120k, the launchers' default context, how much context fits in the VRAM
q8_0 uses there, KL, decode and prefill cost): [BENCHMARKS.md](BENCHMARKS.md#c-d-and-e-side-by-side). D and F run the
model's full 262,144 tokens (below, and [BENCHMARKS.md](BENCHMARKS.md#the-full-context)).

D' is D with the MTP layer's cache at 8 bits: ~70 MB more, and the drafts read q8_0 K / V (its speed against D:
[BENCHMARKS.md](BENCHMARKS.md#c-d-and-e-side-by-side)).

**Preset D** frees ~1.1 GB at 120k context and fits the model's full 262,144-token context on 24 GB. A prompt that fits
in the window reads only q8_0 rows, so its output is q8_0's; past the window, KL at 32k is 0.00236 against q8_0's
0.00219, the 282 questions score the same 230, and prompts read about 4% slower. Decode tok/s also depends on the text:
with D the reply differs from q8_0's, and with it the draft acceptance. At the full context (2026-10-20, 7322fbd) a
260,896-token prompt was read in 326 s, decode at that depth ran at 66 tok/s, a follow-up turn took 1.28 s, the card
held 23.5 GB, and 4 of 4 needle-in-a-haystack and 4 of 6 LongBench v2 questions at that length were right. Every number,
with the fp16 cache's own KL floor: [BENCHMARKS.md](BENCHMARKS.md#kv-cache-presets).

**Preset F** stores K at 5 bits, Hadamard-rotated, and V at 5 bits, with the same window: 0.26 GB less KV cache than D
at 120k context (2.99 against 3.25 GB) and 0.58 GB less at 262,144 (6.39 against 6.97 GB). Its KL equals q8_0's at 64k
(0.00163), at 100k (0.00218 against 0.00217; D 0.00226) and on 160k tokens of prose (0.00472 against 0.00467), and is
10% above q8_0's at 32k (0.00242; D 0.00236; the set's noise is about 0.00008 either way), and a larger window did not
change the 32k number. On the 282 questions it answered 229 right (D and q8_0: 230; three items
differ, one of them an answer cut by its token cap). It reads prompts as fast as D. `scripts\serve.bat` uses it above
240k unless `MK_KV` is set: natively on Windows, with the desktop on the same card, preset D at the full context passes
what the driver keeps resident, and F stays inside it with the default prefill chunk and draft head. Numbers:
[BENCHMARKS.md](BENCHMARKS.md#preset-f).

### MK_KV syntax

`MK_KV` is a comma-separated list of `<K|V|*><slot>=<bits>[r]`:

- `K`, `V` or `*` (both); the slot is 0-15 for the 16 attention layers in model order (layer 4 x slot + 3) and 16 for
  the MTP layer; no slot or `*` means every slot. Later items override earlier ones for the same slot and side.
- bits 4-8 (8 = q8_0, 4 = q4); `r` stores the side Hadamard-rotated (K: the queries get the same rotation; V: the
  attention output is rotated back).
- It needs the q8_0 cache: `--kv q8`, or `--kv auto`, which picks q8_0 whenever `MK_KV` is set; `--kv f16` refuses it.
- `MK_KV_WIN=W` (only with a non-empty `MK_KV`) keeps every low-bit side's most recent W or more positions as q8_0 in
  a ring (sized in steps of 640 positions). With exact reuse, every checkpoint also holds the rings as of its position
  in pinned host memory (preset D': +106 MiB per checkpoint slot), and a prefill chunk is at most the ring's length:
  the engine lowers `MK_PF_CAP` to 1,280 for W up to 640 and to 1,920 up to 1,280 (the presets' 2,048 is not lowered).
- With exact reuse on, a reused or restored conversation under a preset continues bit for bit as the same request sent
  fresh under that preset (`scripts/exactness.py` and the state digest of `scripts/capi_checks.py`, on q8_0, D and F).

Per-layer scans showed K about 2.3 times as sensitive as V, and no setting that saved 0.5 GB or more
stayed within the KL noise of the q8_0 cache, which is why the presets are opt-in. A plain 4-bit cache
(`MK_KV='*=4'`, no window) scores 0.00547 on the short prompts.

### Choosing the window and rotation

Every preset uses `MK_KV_WIN=2048` because the accuracy a low-bit cache loses comes almost entirely from the most
recent ~0.5-2k positions, where attention puts most of its weight. Keeping those at 8 bits removes it. Of the windows
tried (512, 1,024 and 2,048 positions), 2,048 was the smallest at which K at 6 bits and V at 5 bits measured at the
noise floor of the 32k test set; at 1,024 they were still near-lossless, but measurably worse. The numbers:
[BENCHMARKS.md](BENCHMARKS.md#the-window-size-and-rotation).

- **The ring is small and does not grow with the context.** With preset D (K and V low-bit in the 16 attention layers
  and the MTP layer: 34 rings) it takes 46 / 69 / 114 / 182 MiB at `MK_KV_WIN` 512 / 1024 / 2048 / 4096, at 32k context
  or at 262k; preset C (V only) half of that. The presets' savings
  ([BENCHMARKS.md](BENCHMARKS.md#c-d-and-e-side-by-side)) are net of it. A smaller window saves tens of MiB at a
  measurable cost in accuracy. Larger windows were not measured: at 2,048 preset D already read at the noise floor of
  the 32k set in the emulator.
- **The window is rounded up to steps of 640 positions**, the attention kernel's tile partition (which also keeps
  greedy output independent of the draft depth): `MK_KV_WIN=2048` keeps the last 2,561 to 3,200 positions at 8 bits,
  and any value from 1,921 to 2,560 gives the same ring. The startup line `[mk] KV recent window: ...` prints the range
  and the ring's size.
- **Rotation (`r`)** spreads each block's outliers over the block before rounding, so the one scale per 32 values fits
  the rest of the block better. Without a window it halved 6-bit K's extra KL, and it helped V on short prompts but not
  at 32k. Under the 2,048 window it made no measurable difference, so preset D stores K and V unrotated (no rotation of
  the queries, no rotating back of the attention output). It earns its place when going lower: preset E stores V at 4
  bits, rotated.

## Launchers: scripts/serve.sh and scripts/serve.bat

`scripts/serve.sh [port] [ctx]` (Linux / WSL2) and `scripts\serve.bat [port] [ctx]` (native Windows) check that the
engine library and the model file exist, print one line with the settings, and start the server as

```text
<MK_PYTHON> server/mk_server.py -m <MK_MODEL> --host <MK_HOST> --port <port> -c <ctx> --kv q8 -d <MK_DRAFT>
    --exact-reuse                                   (unless MK_EXACT_REUSE=0)
    --mmproj <file> [--image-max-tokens N] [--video-fps F] [--video-max-tokens N]   (when images are on)
    --chat-template-file <MK_CHAT_TEMPLATE_FILE>    (when set)
```

with the environment passed on unchanged. The defaults are port 8080 and 120,000 tokens of context; the KV cache is
always q8_0 (`--kv q8`), with or without a low-bit preset on top.

These variables are read by the launchers themselves and turned into options:

| Variable | Default | What it does |
|---|---|---|
| `MK_MODEL` | `$HOME/models/Qwen3.8-27B-UD-Q4_K_XL.gguf`; Windows `%USERPROFILE%\models\Qwen3.8-27B-UD-Q4_K_XL.gguf` | the GGUF (`-m`). Under WSL2 keep it on the Linux filesystem: files on a Windows drive (`/mnt/c/...`) load several times slower |
| `MK_HOST` | `127.0.0.1` | listen address (`--host`); `0.0.0.0` for the LAN |
| `MK_DRAFT` | `7` | drafts per cycle (`-d`); 8-15 need a [deep-pass build](EXPERIMENTAL.md#the-deep-pass) and no `MK_KV`, so with `serve.bat` above 240,000 (preset F) such a load fails |
| `MK_EXACT_REUSE` | on | `0` leaves out `--exact-reuse`; any other value keeps it |
| `MK_MMPROJ` | serve.sh: the first of `<model folder>/mmproj-BF16.gguf`, `$HOME/models/mmproj-BF16.gguf`, `$HOME/models/mmproj-Qwen3.8-27B-BF16.gguf` that exists; serve.bat: `<model folder>\mmproj-BF16.gguf` | the vision projector (`--mmproj`). Images are on when the file and the vision library (`libqwen38vision.so` / `qwen38vision.dll`) exist. Off: `MK_MMPROJ=` (set and empty) in serve.sh, `MK_MMPROJ=off` in serve.bat |
| `MK_IMAGE_MAX_TOKENS` | the projector's (4096) | `--image-max-tokens`, with images on |
| `MK_VIDEO_FPS` | 4 | `--video-fps`, with images on |
| `MK_VIDEO_MAX_TOKENS` | the image limit | `--video-max-tokens`, with images on |
| `MK_CHAT_TEMPLATE_FILE` | the GGUF's template | `--chat-template-file` |
| `MK_PYTHON` | `python3`; Windows `python` | the Python that runs the server (needs jinja2) |
| `MK_KV`, `MK_KV_WIN` | unset (q8_0): `serve.bat` sets preset F (`K*=5r,V*=5`, window `2048`) above 240,000 | read by the engine ([KV cache presets](#kv-cache-presets)) |
| `MK_PF_CAP` | unset: the launchers set `1024` when the context is above 96,000; `serve.bat` sets `512` above 240,000 when `MK_KV` was set by you | read by the engine ([Prefill](#prefill)) |
| `MK_DRAFT_Q4` | unset (the engine's default, on): `serve.bat` sets `0` above 240,000 when `MK_KV` was set by you | read by the engine ([Speculation](#speculation-drafts)) |

Everything else is read from the environment by the server or the engine, so it works the same with or without a
launcher: `MK_KV`, `MK_KV_WIN`, `MK_KEEP_WARM_S`, `MK_DRAFT_Q4`, `MK_MTP`, `MK_NO_MTP`, `MK_SIDE_MAX`, `MK_PARK_MIB`,
`MK_ALLOW_HOSTS`, `MK_CORS_ORIGINS`, `MK_API_KEY`, `MK_CHAT_TEMPLATE_KWARGS`, `MK_CONTENT_SPECIALS`,
`MK_REASONING_FORMAT`, `MK_TIMEOUT_S`, `MK_REQUEST_TIMEOUT_S`, `MK_MAX_BODY_MB`, `MK_MAX_INFLIGHT_MB`,
`MK_MAX_CONNECTIONS`, `MK_MAX_IMAGES`, `MK_IMAGE_MAX_PIXELS`, `MK_MAX_VIDEO_FRAMES`, `MK_FFMPEG` (all listed at the top
of `scripts/serve.sh`) and every engine variable below.

The launchers do not pass `--park-min`, `--park-cks`, `--ck-every`, `--ck-msgs`, `--ck-pin`, `--image-min-tokens`,
`--image-cache`, `--video-cache`, `--vision-threads`, `--vision-backend`, `--chat-template`, `--lib`, `--vision-lib`
or `--draft-vocab`: those keep the server's defaults unless the server is started directly.

## Server options

`server/mk_server.py` options, in the order of `--help`. "Env" is the environment twin: the option given on the
command line wins, except for the two lists, which are combined. Most server options do not touch the arithmetic; the
behaviour behind the conversation-cache options (checkpoints, pinned checkpoints, parking, side requests) is described
in [SERVER.md](SERVER.md#the-conversation-cache), the security options in [SERVER.md](SERVER.md#security) and the
limits in [SERVER.md](SERVER.md#limits).

| Option | Env | Default | What it does | Output |
|---|---|---|---|---|
| `-m`, `--model` | - | required | the GGUF | - |
| `--host` | - | `127.0.0.1` | listen address | - |
| `--port` | - | `18080` (launchers 8080) | listen port; a port in use ends the server before the model loads | - |
| `--allow-host NAME` | `MK_ALLOW_HOSTS` | none | host names clients may use besides localhost, IP addresses and `--host`'s (repeatable or comma-separated; `*` = any name, which turns the DNS-rebinding check off); other names get a 403 | - |
| `--cors-origin`, `--cors-origins ORIGIN` | `MK_CORS_ORIGINS` | none | web origins (`scheme://host[:port]`) whose pages may call the server (CORS headers, preflights; `*` is refused); pages of other origins get a 403. Clients that are not web pages send no Origin and are not affected | - |
| `--api-key KEY` | `MK_API_KEY` | none | every request but `/health` and CORS preflights must carry the key (`Authorization: Bearer KEY` or `x-api-key: KEY`); others get a 401 | - |
| `-c`, `--ctx` | - | `32768` (launchers 120000); with `--agents` `0` (with `--parallel` 32768) | context in tokens, at least 1 and at most the model's full context of 262,144 (else the engine refuses to start; a KV cache larger than the free device memory loads with a warning where the driver falls back to system memory, Windows and WSL2, and elsewhere stops the load with an error naming what fits: [ENGINE.md](ENGINE.md#kv-cache)); with `--agents` each slot's, 0 = the most that fits | - |
| `-d`, `--draft` | - | `4` (launchers 7) | MTP drafts per cycle: 0-7, or 0-15 in a [deep-pass build](EXPERIMENTAL.md#the-deep-pass); more than 4 needs the q8_0 cache, more than 7 also no `MK_KV` | drafts only (8-15: [deep-pass build](EXPERIMENTAL.md#the-deep-pass)) |
| `--draft-vocab` | - | `draft_vocab_32k.bin` in the repo | the draft head's vocabulary (the 32k most frequent tokens; tokens of the current conversation are added) | drafts only |
| `--mtp FILE` | `MK_MTP` (engine) | the model file's MTP layer | the MTP draft layer from another GGUF of the same model (checked against it) | drafts only |
| `--no-mtp` | `MK_NO_MTP=1` (engine) | off | no MTP layer: plain decoding at depth 0, no MTP weights or KV slot | drafts only |
| `--lib` | - | `libqwen38mk.so` / `qwen38mk.dll` in the repo root | the engine library | - |
| `--kv` | - | `auto` (launchers `q8`) | KV cache format: `f16`, `q8` (q8_0), `auto` (q8_0 when fp16 does not fit next to ~1.5 GB of working memory, or when `MK_KV` is set; with `--agents` q8_0, which the slots need) | changes (f16 vs q8_0) |
| `--agents N` | `MK_AGENTS` | `0`: one request at a time | up to N requests (1-4) at once, each in a parallel slot of the engine with its own context (`-c`, at most the model's context length, 262,144), decoded together in one batched launch per cycle; images, video and the presence / frequency penalties get a 400, side requests, parking and pinned checkpoints are off ([SERVER.md](SERVER.md#several-agents-at-once---agents)) | greedy: the reply alone, bit for bit (a prompt fed in chunks beside other slots: see `MK_PF_MIX`; beside a sampled request, rarely not: two logits one float step apart can tie after the norm scale, [SERVER.md](SERVER.md#several-agents-at-once---agents)); sampled: the same distribution, not the same tokens beside other requests |
| `--parallel N`, `-np N` | - | `0` | `--agents N` with `-c` defaulting to 32,768 per slot (llama-server's and open-jet's option) | as `--agents` |
| `--ck-every N` | - | `4096` | a DeltaNet checkpoint every N prompt tokens (0: off); the pool has `MK_CK_SLOTS` slots | same with exact reuse |
| `--ck-msgs` | - | `1` | a checkpoint at each message start (1 / 0) | same with exact reuse |
| `--ck-pin` | - | `1` | pin the checkpoint at the end of the system message (and where new sessions left the cached one) against eviction, and park it with the conversation: a client's new session resumes after its system prompt and tools | same with exact reuse |
| `--exact-reuse` | `MK_EXACT=1` (engine) | off (launchers on) | reused requests answer exactly as fresh ones: restore a checkpoint and prefill, instead of appending to decoded rows ([MK_EXACT](#exact-reuse-and-checkpoints)) | makes reuse bit-exact |
| `--side-max N` | `MK_SIDE_MAX` | `8192` | side requests: a first-turn prompt of at most N tokens (with parking on: one without tools or with `max_tokens` <= 1024; also any request without tools beside a conversation whose last request had tools) that would discard at least twice its length, and at least `--park-min` tokens, of the cached conversation runs beside it (0: off) | same with exact reuse |
| `--park-mib MIB` | `MK_PARK_MIB` | `8192` | host memory for parked conversations, allocated on first use (0: off; side requests alone then use 1024 MiB) | same with exact reuse |
| `--park-min N` | - | `1024` | conversations of fewer tokens are not parked; a parked one comes back when its prompt reuses at least N more tokens than the cached one's; a side request must spare at least N cached tokens | same with exact reuse |
| `--park-cks N` | - | `2` | checkpoints kept with a parked conversation (the newest ones, besides its pinned ones) | same with exact reuse |
| `--content-specials` | `MK_CONTENT_SPECIALS` | `text` | special-token text (`<\|im_end\|>`, ...) inside messages, tool results, tool calls and tool definitions: `text` (tokenized as text, so content cannot forge turns) or `parse` (as special tokens, as llama-server does) | changes (tokenization) |
| `--chat-template-file PATH` | `MK_CHAT_TEMPLATE_FILE` (launchers) | the GGUF's template | a Jinja chat template file instead of the GGUF's (llama-server's option) | changes (prompt) |
| `--chat-template TEMPLATE` | - | the GGUF's template | the template itself (`chatml`: llama.cpp's ChatML); the last of the two options wins | changes (prompt) |
| `--reasoning-format` | `MK_REASONING_FORMAT` | `auto` | llama-server's: `auto` / `deepseek` (thinking as `reasoning_content`), `deepseek-legacy`, `none` (left in the content); a request's `reasoning_format` wins | - (formatting) |
| `--chat-template-kwargs JSON` | `MK_CHAT_TEMPLATE_KWARGS` | none | defaults under every request's `chat_template_kwargs` (keys `enable_thinking`, `preserve_thinking`, `reasoning_effort`); the request's own win | changes (prompt) |
| `--mmproj` | `MK_MMPROJ` (launchers) | none: no images | the vision projector GGUF: image and video input, encoded on the CPU (no VRAM) | - |
| `--vision-lib` | - | `libqwen38vision.so` / `qwen38vision.dll` in the repo root | the vision library | - |
| `--vision-threads` | - | `0`: every logical CPU but two | CPU threads of the image encoder | same |
| `--vision-backend` | - | `auto` | `cpu` (the AVX2 / FMA encoder), `mtmd` (llama.cpp's), `auto` (cpu when the CPU has AVX2 and FMA) | changes (image rows) |
| `--image-min-tokens` | - | `0`: the projector's | fewest tokens per image | changes (image rows) |
| `--image-max-tokens` | `MK_IMAGE_MAX_TOKENS` (launchers) | `0`: the projector's (4096 for Qwen3.x) | most tokens per image; fewer encode faster | changes (image rows) |
| `--image-cache` | - | `8` | images kept decoded / encoded for reuse by later requests | - |
| `--video-fps` | `MK_VIDEO_FPS` (launchers) | `4.0` (llama.cpp's) | video frames sampled per second, in (0, 30]; two frames make one grid of tokens | changes (video rows) |
| `--video-max-tokens` | `MK_VIDEO_MAX_TOKENS` (launchers) | `0`: `--image-max-tokens` | most tokens per frame pair (0 or at least 8; the cpu backend only); fewer fit longer clips | changes (video rows) |
| `--video-cache` | - | `2` | videos kept decoded / encoded for the next requests of a conversation | - |
| `--ffmpeg` | `MK_FFMPEG` | `ffmpeg` on PATH, else the imageio-ffmpeg package's | the ffmpeg executable (or its folder) for video input; without one, video requests get a 400 | - |

"Same with exact reuse": with `--exact-reuse` these options decide only how much is re-read, never the output;
without it, a reused prompt appends to the rows the previous reply decoded, which is not bit-identical to a fresh
request. Measured effects: `--park-min 4096` against 1024 halved the real evictions (15 to 8) at equal prefill in a
replay of recorded client sessions; pinned checkpoints took Claude Code's new sessions from 0 to
8,288 cached prompt tokens and Codex's from 0 to 8,672 (preset D). More in
[SERVER.md](SERVER.md#the-conversation-cache).

### Server environment variables

Read by the server only (there is no option for them):

| Variable | Default | What it does |
|---|---|---|
| `MK_TIMEOUT_S` | `600` | seconds a connection's read or write may make no progress before the connection closes (llama-server's `--timeout`; `0`: none): idle keep-alive connections, stalled requests, a client that stopped reading a reply. A stream's write that makes no progress for 60 s (this, when it is shorter and not `0`) counts as a disconnect: the generation stops and the engine is free |
| `MK_REQUEST_TIMEOUT_S` | `300` | seconds a request may take to arrive, from its first byte to the last of its body, before the connection closes (`0`: no limit): a client sending a byte at a time never hits `MK_TIMEOUT_S`. Generating the reply does not count |
| `MK_MAX_BODY_MB` | `128` | largest request body in MB, Content-Length or chunked; larger ones get a 413 |
| `MK_MAX_INFLIGHT_MB` | `256` | request bodies in memory at once in MB, all connections (at least `MK_MAX_BODY_MB`); a body past it gets a 503 |
| `MK_MAX_CONNECTIONS` | `256` | open connections; a connection past it gets a 503 |
| `MK_MAX_IMAGES` | `32` | images per request; more get a 400 |
| `MK_MAX_VIDEO_FRAMES` | `256` | video frames per request, all of its videos (64 s at 4 fps) |
| `MK_PF_MIX` | `256` | `--agents`: rows of a waiting prompt fed right before each decode cycle of the decoding slots (`mk_slots_step_mixed`: the chunk, then the step, as separate launches), so they keep decoding while a long prompt is read; `0`: `MK_PF_CHUNK` tokens between the cycles. With `MK_EXACT` (the `--agents` default) a chunk before the prompt's last ends on the 32-token scan grid (up to 31 tokens off this budget), a last chunk of 32 tokens or fewer joins the one before (so the prompt keeps its checkpoint), and the chunking changes no reply |
| `MK_PF_CHUNK` | `512` | `--agents` with `MK_PF_MIX=0`: prompt tokens fed between two decode cycles (`0`: the whole prompt, the other slots wait); with `MK_EXACT`, on the 32-token grid as `MK_PF_MIX` |
| `MK_PF_SOLO` | `0` | `--agents`: prompt tokens fed per call while no slot decodes (`0`: the whole prompt); with `MK_EXACT`, on the 32-token grid as `MK_PF_MIX` |
| `MK_IMAGE_MAX_PIXELS` | `100000000` | largest image or video frame in pixels, checked from the file header before decoding; also read by the vision library. The server reads it only with `--mmproj` |

`MK_MAX_BODY_MB`, `MK_MAX_INFLIGHT_MB`, `MK_MAX_CONNECTIONS`, `MK_MAX_IMAGES`, `MK_MAX_VIDEO_FRAMES` and
`MK_IMAGE_MAX_PIXELS` must be positive integers, the three `MK_PF_*` integers >= 0: anything else ends the server at
startup with a message. `MK_SIDE_MAX`, `MK_PARK_MIB`, `MK_AGENTS`, `MK_TIMEOUT_S` and `MK_REQUEST_TIMEOUT_S` accept `0`
(off). The server sets `MK_EXACT=1` for the engine when `--exact-reuse` is given, and with `--agents` sets `MK_EXACT=1`,
`MK_PF_CAP=1024` and `MK_CK_SLOTS=1` unless they are set; it reports `MK_CK_SLOTS` in `/props`.
llama-server's `LLAMA_ARG_*` variables are not read.

## Engine environment variables

Read by the engine library (server) and the CLI when the model loads, unless said otherwise. Each row shows the values
the code checks: `0` turns a default-on switch off, `1` turns a default-off one on, and "set" means any value turns it
on (even `0`). The measurement and test switches are in [their own section](#measurement-test-and-debug-switches).

### KV cache

| Variable | Default | What it does | Output | Status |
|---|---|---|---|---|
| `MK_KV` | empty: q8_0 everywhere | per-layer K / V formats on top of the q8_0 cache ([presets, syntax](#kv-cache-presets)) | changes (lossy, KL-gated) | opt-in |
| `MK_KV_WIN` | `0`: no window | with `MK_KV`: the most recent W positions of every low-bit side also kept as q8_0 in a ring | changes (closer to q8_0) | opt-in |
| `MK_MTP_WIN` | `0`: off | the MTP layer's decode attention sees only the first 32 positions and the last N (up to N + 31: the window starts on a 32-position tile). Draft acceptance fell from 97% to 88-90% at 101.6k context for any window of 1k-16k, 5-7% fewer tok/s | drafts only | slower |

### Exact reuse and checkpoints

| Variable | Default | What it does | Output | Status |
|---|---|---|---|---|
| `MK_EXACT` | `0`; the server sets `1` with `--exact-reuse`, and with `--agents` unless set | exact reuse: every prefill row comes out the same however the sequence was cut into prefill calls and chunks, as long as a prefill starts on the absolute 32-token grid (the engine warns when one doesn't; the server's checkpoints are all on it) (prefill attention over absolute key segments, checkpoints on that grid, window rings saved with each checkpoint). The server then restores the last checkpoint at or before the point where a prompt diverges and prefills from there instead of appending to the reply's decoded rows. Cost: a 98k prefill from scratch +1.3%; a follow-up turn re-reads the previous reply (11.6k chat: 167 instead of 78 ms); edits cost the same | reuse = fresh, bit for bit; the prefill arithmetic differs from `0` (KL within noise: 98.662% / 0.00118 vs 98.654% / 0.00116) | served |
| `MK_EXACT_SEG` | `4096` | exact mode: positions per absolute key segment of the prefill attention (rounded down to a multiple of 32). Against 8192 on a 98k prompt: the same time from scratch, faster for a restore plus 32 tokens | changes (another fold order); reuse stays exact at any value | option |
| `MK_EXACT_SPLIT` | `1` | exact mode: the last wave of attention work is split into segment units to balance the GPU; `0`: whole items only | same (every split gives the same rows) | option |
| `MK_CK_SLOTS` | `8`; the CLI and `--agents` keep 1 unless set | DeltaNet checkpoint slots in pinned host memory (1-64; ~0.17 GB each). Pinned checkpoints (`--ck-pin`) take slots from this pool. With `--agents` the engine's own sequence only warms up: each slot has its own checkpoints (`MK_SLOT_CK`). When pinned host memory runs out, the pool keeps the slots it got (with none: one slot in device memory), with a warning | same with exact reuse | served |
| `MK_CK_DEVICE` | `0` | `1`: a single checkpoint slot in device memory instead of the pinned-host pool (the original layout; refused when `MK_EXACT` and `MK_KV_WIN` are both set) | same | earlier path |

### Speculation (drafts)

| Variable | Default | What it does | Output | Status |
|---|---|---|---|---|
| `MK_POLICY` | empty: the defaults | the draft-depth policy as `key=value,...` ([keys](#mk_policy-keys)); the CLI's `--policy` sets it per job | drafts only | option |
| `MK_DRAFT_Q4` | `1` | the draft head reads a Q4_K copy of its lm_head rows, built at load (+90 MiB of VRAM, +0.83 s of load); `0`: lm_head itself. Short greedy replies +2.1% tok/s, long context no measurable change | drafts only | served |
| `MK_MTP_Q4` | `1` | the MTP layer's matrices stored wider than Q4_K are re-quantized to Q4_K at load (-107 MiB), IQ4_NL too. Always on: `0` (as stored) and `ab` (both copies) are refused ([why](ENGINE.md#the-gguf)) | drafts only | served |
| `MK_MTP` | unset | the MTP layer from another GGUF of the same model (as `--mtp`; the CLI's and server's options win) | drafts only | option |
| `MK_NO_MTP` | unset | `1`: no MTP layer, plain decoding at depth 0 (as `--no-mtp`) | drafts only | option |
| `MK_MTP_KVC` | `1` | the MTP catch-up rows before the last 5 compute K / V only; `0`: the full pass. +0.9% at 38k, +0.7% at 101.6k | same (identical drafts) | served |

### MK_POLICY keys

`MK_POLICY` (and the CLI's `--policy`) take comma-separated `key=value` items; an empty string means the defaults.
Every key changes only which drafts are made and verified, never the output (greedy ids identical; sampled output
keeps its distribution). The defaults were tuned on replayed logs and live A/B runs.

| Key | Default | What it does |
|---|---|---|
| `depth` | `1` | `1`: always up to the `-d` drafts; `0`: one fewer (at least 3) after a cycle with a rejected draft; `2`: one more than the last cycle accepted, within [`dmin`, `-d`] |
| `dmin` | `1` | the least drafts for `depth=2` |
| `stop` | 0.4 from draft 2 on | stop drafting after a draft when the running product of the MTP's top-1 probabilities falls below this (every draft) |
| `drop` | none (the deepest draft: `droplast`) | drop a draft whose running product falls below this (every draft) |
| `stopK`, `dropK` | - | the same for draft K alone (K = 1-7; 1-15 in a deep-pass build), e.g. `drop4=0.4` |
| `droplast` | `0.4` | drop the deepest draft below this |
| `g7` | `1` | the gates for depths above 5: stop after drafts 2-4 below 0.25, drop draft 5 below 0.3, stop after drafts 5-6 below 0.3, never drop the deepest draft. Any explicit `stop` / `drop` key turns it off. +2.0% at 38k and +4.1% at 101.6k greedy, together with `MK_MTP_KVC` |
| `dstop` | `0.5` | [deep-pass build](EXPERIMENTAL.md#the-deep-pass) at depths above 7, with `g7`: stop after drafts 8 and later below this |
| `tier` | `3` | deep-pass build: a cycle verifies its drafts past the 7th (the 16-row pass) only when the sum of the running products of their MTP probabilities, the MTP's own estimate of how many of them will be accepted, is at least this; else up to 7 drafts. `0`: no such condition. With sampled drafts verified by sampling (`h1=1`), the sum uses only what is known before the 8th draft is drawn: the running product through the 8th draft's probability, then that probability again for each further draft up to the depth (choosing by the later drafts' own probabilities which drafts get verified would bias the output) |
| `dctx` | off | from position N on, at most 4 drafts |
| `chain` | `1` | what the gates compare: `1` the running product of the top-1 probabilities, `0` each draft's own top-1 probability, `2` with sampled drafts the product of the drafted tokens' own probabilities (drop gates off) |
| `h1` | `1` | when sampling, draft by sampling the MTP head (`0`: its top-1 token) |
| `dq4` | - | with the draft head's Q4_K copy (`MK_DRAFT_Q4`, on by default): `1` the copy, `0` lm_head (from the next launch) |

No gates: `depth=0,drop=0,stop=0,chain=0,h1=0` (adaptive depth, top-1 drafts), the A/B baseline for the defaults.

### Prefill

| Variable | Default | What it does | Output | Status |
|---|---|---|---|---|
| `MK_PF_CAP` | `2048`; the launchers use `1024` above 96k context, `serve.bat` `512` above 240k with your own `MK_KV` | tokens per prefill chunk, 128-2048 in steps of 128. The scratch is ~0.33 MB per token: 1024 frees ~0.34 GB, and prefill was 0.2-0.45% slower than at 2048 on UD-Q4_K_XL (1ecb507, 2026-11-08). With exact reuse and `MK_KV_WIN` up to 1,280 the engine lowers it to the window ring's length (1,280 or 1,920: a chunk must not wrap the rings a checkpoint saves) and says so at load | same with exact reuse; not checked without | served |
| `MK_ATTN_S8` | `1` | q8_0 cache, prefill attention: `1` int8 Q K^T with one Q scale per 32 dims (+16.7% prefill at 101.6k); `2` one Q scale per row (+5% in the kernel, 1.6 times the Q rounding noise); `0` fp16 tensor cores | changes | served (`1`) |
| `MK_ATTN_SPLIT` | `1` | `0`: no tail split of long rows in the prefill attention (exact mode never splits) | changes | earlier path |
| `MK_ATTN_PV16` | `1` | `0`: the attention's P V accumulated in fp32 instead of fp16 | changes | earlier path |
| `MK_ATTN` | `0` | `1`: the original prefill attention kernel (q8_0 only, no `MK_KV`). Its successor was 41.6% faster at 101.6k and gives the same bits with `MK_ATTN_S8=0 MK_ATTN_SPLIT=0 MK_ATTN_PV16=0` | changes | earlier path |
| `MK_PREFILL` | batched | `mk`: prompts go through the decode megakernel instead of the batched prefill kernels; slower even at 42 tokens (460 vs 404 ms) | changes | earlier path (upstream's) |
| `MK_PF_GLU_MIX` | `1` | `0`: a gate / up pair of two weight types runs as two GEMMs plus a kernel instead of one fused GEMM | same | earlier path |
| `MK_PF_Y1` | `0` | `1`: exact mode keeps the full-size y1 prefill buffer instead of the smaller one it needs when every gate / up pair is fused (about -44 MiB) | same | debug |
| `MK_PF_BN` | `0`: by prompt size | force one token tile for every prefill GEMM: 32, 64 or 128 | same (32 checked byte-identical on Q4_K_M) | debug |
| `MK_PF_SYNC` | `0` | `1`: wait for each prefill chunk before queuing the next (the original flow; GPU idle per prefill at 38k 11.8 ms vs 0.19 ms on native Windows, no measurable change in prefill time) | same | earlier path |
| `MK_K8` | `6` | prefill small kernels, bits: `2` pk_norm_quant reads its input once, `4` pk_dn_prep handles `MK_K8_NT` tokens per block (each -27..-29% in isolation, no measurable prefill change), `1` pk_ab on a second stream (opt-in, not measured) | same | served (`6`) |
| `MK_K8_NT` | `4` | tokens per block for `MK_K8` bit 4: 4 or 8 (8 was slower) | same | option |
| `MK_PFX` | `0` | prefill launch order and kernels, bits (a set bit keeps the earlier path): `1` no side streams for a layer's independent GEMMs (the delta-net gate next to qkv, attention k / v next to q), `2` the delta-net scan without the `pk_dn_tri` pre-pass of its state-independent chunk matrices, `4` `pk_ab` after the in-projection join, `8` attention k and v both on side streams, `16` `pk_ab` instead of `pk_ab2` (the same values in fewer shared-memory loads), `64` `pk_attn8` instead of `pk_attn9` (the same values, more instructions per tile), `128` `pk_conv_ring` copying one float per thread and step (not ten float4 loads before the stores: 14.8 vs 3.4 us per delta-net layer at 1024 tokens), `256` `pk_attn2_comb` merging 12 rows per warp (not one row per warp: 45.8 vs 8.6 us per attention layer) | same | earlier path |

### Decode kernels

These switch single decode kernels back to an earlier form, for A/B measurements. All defaults are the served ones.

| Variable | Default | What it does | Output | Status |
|---|---|---|---|---|
| `MK_DATTN` | `1` | `0`: the earlier, unpipelined decode attention for q8_0 layers (the pipelined one was +18.7% at 101.6k and +7.3% at 38k) | same | earlier path |
| `MK_K1` | `1` | `0`: DeltaNet input projection deals its alpha / beta work the earlier way | same | earlier path |
| `MK_K2` | `1` | `0`: the split-K residual combine loads its partial sums token by token (K1 + K2 together: decode +0.98..+1.58%) | same | earlier path |
| `MK_K6` | `6` | Q8_0 k / v tiles of the attention projection, bits: `4` each tile as two 8-row items (88.5 to 78.0 us per call); `2` the items dealt across the blocks instead of in contiguous runs (per cycle -0.04 ms at 16 rows, -0.06 at 8, -0.02 at 40k context) | same | served (`6`) |

### Parallel slots (`--agents`)

| Variable | Default | What it does | Output | Status |
|---|---|---|---|---|
| `MK_SLOT_RESERVE_MIB` | `1536` | device memory left free when the slots take the most context that fits (`-c 0`), for the prefill and the runtime | - | option |
| `MK_SLOT_SPILL` | `0` | `1`: no check that the slots fit in the free device memory (plus the memory pool's unused reserve) before they are allocated. Without it a `-c` that does not fit is refused at once, naming what it needs and what is free; with it a Windows / WSL2 driver's CUDA sysmem fallback (on by default) puts the rest in system memory (charged to Windows commit; the KV cache is then read over PCIe) | same | option |
| `MK_SLOT_CK` | `2` | DeltaNet checkpoints per slot (where a slot's next prompt resumes when it diverges inside the slot's sequence), in pinned host memory: when that runs out, a slot keeps fewer, with a warning | same with exact reuse | option |
| `MK_PAR_RMAX` | unset | a value up to 8: batched slot passes of 8 rows (the verify pass's width) instead of the build's `MK_PAR_RMAX` (16). The engine prints which it runs at load (`parallel slots: batched passes of ...`) | same | debug |

### Memory and host store

| Variable | Default | What it does | Output | Status |
|---|---|---|---|---|
| `MK_STORE_CHUNK_MIB` | `32` | chunk size of the host store that holds parked conversations and side requests: pinned chunks allocated on first use and reused | - | option |
| `MK_CUDAMALLOC` | `0` | `1`: one `cudaMalloc` per buffer instead of the stream-ordered memory pool, which packs the allocations (cudaMalloc rounds each to 2 MiB pages: ~0.5 GB over the ~900 weight tensors) | same | earlier path |
| `MK_STACK_LIMIT` | `0` | the CUDA per-thread stack limit in bytes. The driver reserves local memory for every thread slot (1,536 per SM) by the larger of this and the kernels' own frames: on an RTX 3090 the megakernel's 720-byte frame (736 in the deep-pass build) reserves ~86 MiB, the driver's 1 KB default would reserve ~123 MiB | - | earlier path |
| `MK_RESERVE_EARLY` | `1` | `0`: 1ecb507's load order. `1` (default): device memory placement for a nearly full GPU. The stack limit is set to the megakernel's frame right after the CUDA context is made (before the weights; `MK_STACK_LIMIT` raises it) and kept, and the kernels' code (loaded on first use) loads before the KV cache, so neither is placed last. The KV cache is then checked against the free memory: its bytes plus the RoPE table, the prefill scratch and `MK_VRAM_MARGIN` must fit, else the load stops with the `-c` that fits. At load end and after the first launch a full device is reported (the CUDA sysmem fallback of Windows / WSL2 then keeps part of the process in system memory: slower decode that varies per process) | same | option |
| `MK_VRAM_MARGIN` | `384` | MiB that `MK_RESERVE_EARLY`'s KV fit check keeps free past what it counts: the memory pool's overshoot (~100 MiB at a full cache) and the free figure's movement with the desktop's own use (up to ~150 MiB between processes on WSL2) | same | option |
| `MK_VRAM_OVERCOMMIT` | `0` | `1`: `MK_RESERVE_EARLY`'s fit check warns instead of stopping the load | same | option |
| `MK_EMBD_DEVICE` | `0` | `1`: token_embd in device memory (+0.72 GB of VRAM) instead of mapped pinned host memory, whose rows the kernels fetch over PCIe (no measurable time difference) | not checked (the same rows) | debug |

### Model load

The first six only change how the load runs: the weights on the device are the same bytes (a weight digest compared
them). With them the CLI loads in 11.9 s cold / 7.4-8.3 s warm (59.7 / 32.4 s without), and the server is ready in
8.5 / 12.2 s (30.1 / 59.7 s without). The next three are earlier weight formats, kept for A/B.

| Variable | Default | What it does | Output | Status |
|---|---|---|---|---|
| `MK_LOAD_IO` | `8` | read-ahead threads over the GGUF, at most 3 GiB ahead (`0`: off) | same | served |
| `MK_LOAD_ASYNC` | `1` | `0`: weight uploads on the loading thread instead of a worker thread | same | earlier path |
| `MK_LOAD_GPURL` | `1` | `0`: the wide-verify weight re-layout on the CPU instead of the GPU | same | earlier path |
| `MK_LOAD_PINBG` | `1` | `0`: token_embd's pinned host table allocated in turn instead of on a helper thread | same | earlier path |
| `MK_LOAD_RQBG` | `1` | `0`: the MTP layer's Q4_K re-quantization in turn instead of in the background | same | earlier path |
| `MK_LOAD_POOL` | `1` | `0`: no pool of row-worker threads (the pool gained nothing measurable) | same | option |
| `MK_IQ4NL_Q8` | unset | set: IQ4_NL tensors converted to Q8_0 at load (the path before native IQ4_NL, same arithmetic) | same | earlier path |
| `MK_REENC3` | `0` | `1`: Q3_K and IQ3_S re-encoded to Q6_K at load (the path before the native kernels, which are bit-identical) | same | earlier path |
| `MK_AB_F32` | unset | set: the Q8_0 DeltaNet alpha / beta weights folded into fp32 rows at load (+66 MiB) instead of int8 rows formed on the fly | same | earlier path |
| `MK_VOCAB_SERIAL` | `0` | CLI: `1` loads the tokenizer before the engine instead of next to it | - | debug |
| `MK_STATE_WARM_MS` | `1000` | CLI `--state-dir`: GPU warm-up in ms before decoding from a restored state file (`0`: off). Decode right after a 38k restore: 117 / 166 tok/s without, 168 / 168 with | same | served |

### Keep-warm and watchdog

| Variable | Default | What it does | Output | Status |
|---|---|---|---|---|
| `MK_KEEP_WARM_S` | `120` | seconds the engine keeps the GPU's clocks up after each server request with a one-block, 32-thread polling kernel (`0`: off; the CLI never warms). A short reply after 65 s idle started in ~65 ms instead of 391-397 ms and decoded at 80.2 instead of 61-63 tok/s (native Windows); an RTX 3090 draws ~165 W while warm instead of 30-35 W idle | same (reply text identical) | served |
| `MK_KEEP_WARM_LOAD` | `1,32,50,0,0` | the warm load as `blocks,threads,on_ms,off_ms,spin` (measurement knob: only a continuous load holds the clocks) | - | debug |
| `MK_WATCHDOG` | `60` | seconds a decode launch may take before the engine stops with an error and exit code 3 (prefill chunks, checkpoint copies and the weight upload get at least 120 s; `0`: plain blocking waits). It turns a GPU fault the runtime does not report into a message instead of a hang | - | served |
| `MK_WATCHDOG_SPIN` | `0.25` | seconds a device wait spins before it sleeps between polls | - | option |

## Compile-time flags

### The served build

`make` and `build.ps1` compile the served engine without flags: `MKFLAGS` is empty by default and takes only the debug
and experiment flags below and the opt-in deep-pass build's `MK_MAXD=16`. A running server reports them in `/props`
(`build_info`).

The engine always runs wide verify (the verify pass on int8 tensor cores, up to 7 drafts; up to 15 with `MK_MAXD=16`),
every main-layer Q6_K matrix fragment-ordered on the tensor-core path (ms per cycle -1.5% at 101.6k and -2.6% at 38k
on q8_0), the split DeltaNet barrier, the per-chunk ffn_down waits and the host egress ([ENGINE.md](ENGINE.md)). ptxas
peaked at 21.0 GiB of RAM for it on Linux / WSL2 ([BUILDING.md](BUILDING.md#requirements)).

Wide verify replaced a dp4a verify pass of at most 4 drafts, which summed in another order; what it gained:
[BENCHMARKS.md](BENCHMARKS.md#wide-verify-2026-10-13). The dp4a build is gone, with the other compile-time A/B switches
([Former flags](#former-flags)).

### Former flags

On 2026-11-01 these compile-time switches were removed and their served values built in: the other arms were not
released or gated, and nothing measured them any more. A `-D` for one of them stops the build with an error that names
it.

| Former flag | Built in | What it does | Output against the removed arm | Since |
|---|---|---|---|---|
| `MK_WV` | `1` | wide verify: the main-pass matrix products run as int8 tensor-core GEMMs on a weight layout made at load, at every token count | changes (another summation order; still depth-invariant) | 44e319f, 2026-10-13 |
| `MK_MAXT` | `8` | most tokens of one verify pass, 1 + drafts: 5 allowed 4 drafts, 8 allows 7 (more: [`MK_MAXD`](#engine--d-flags)). Above 5, every ffn_down runs on the tensor-core path | changes | 408fa84, 2026-10-13 |
| `MK_C1` | `1` | per-chunk waits instead of a grid barrier between the ffn gate / up and ffn_down phases: +0.41% at 38k, +0.59% at 101.6k, +0.44% short (2026-10-20) | same | e7fec18, 2026-10-20 (on by default since c4210da) |
| `MK_EGRESS` | `1` | each launch's control block reaches the host through mapped memory, barrier counts carry across launches, draft-vocabulary uploads run asynchronously (+2 MiB): +0.74..+1.34% on Linux over two holds, +0.8..+1.0% native Windows (2026-10-20) | same | 816aced, 2026-10-20 (on by default since 7322fbd) |
| `MK_WV_Q6F` | `1` | every main-layer Q6_K matrix fragment-ordered on the tensor-core path: ms per cycle -1.5% at 101.6k, -2.6% at 38k (q8_0, 2026-10-14) | changes (KL 0.00119 mixed after it) | f30440d, 2026-10-14 |
| `MK_K5` | `1` | the DeltaNet projection -> core grid barrier becomes a split barrier: the 48 heads go to the first 48 blocks that finish, which load their state and replay accepted drafts before they wait. +1.11% at 38k, +0.75% at 101.6k (2026-10-17) | same | 4f2d6f9, 2026-10-17 |
| `MK_GSYNC` | `0` | the grid barrier's form: `0` upstream's; `1`-`3` flag-broadcast and relaxed-polling variants, measured slower (0.61-0.94 us against 0.51-0.55 us per barrier, 2026-10-15). The barrier benchmark (`MK_GSB`) still runs every form | same | e374f27, 2026-10-15 |
| `MK_K7` | `1` | the DeltaNet state laid out so that each warp access is contiguous: the core phase 26.7 -> 18.3 us per call (2026-10-15); `0`: upstream's layout | same | 38d0c26, 2026-10-15 |
| `MK_W48` | `1` | decode attention for 6-8 tokens in one pass over all 48 query rows instead of two head-group passes: 101.6k at 7 drafts 41.2 -> 38.0 ms per cycle on q8_0 (2026-10-13) | same | 9dede7f, 2026-10-13 |
| `MK_DA_ROUTE` | `1` | decode attention chunks of one K / V tile (contexts up to 640) on the simpler kernel: short-context decode +0.4-0.5% (2026-10-12) | same | c482b97, 2026-10-12 |
| `MK_DA_Q4` | `1` | low-bit `MK_KV` layers on the pipelined decode attention (q4 budgets were 9.8-18.4% faster at 101.6k than on the earlier kernel, 2026-10-12) | same | 6287651, 2026-10-12 |
| `MK_AQ_EARLY` | `0` | the removed arm issued low-bit decode attention's K / V copies at the top of each interval: 2% slower per cycle at 101.6k (2026-10-13) | same | 5532c21, 2026-10-13 |
| `MK_XP_SM` | `0` | low-bit tile expansion rows on all 12 non-softmax warps (the removed arm put some on the softmax warps; no speed difference, 2026-10-13) | same | 9deb9c0, 2026-10-13 |
| `MK_K6_LD` | `1` | Q8_0 weight tiles loaded 16 bytes at a time plus one shuffle: the attention projection of the Q8_0-v layers 96 -> 84 us (2026-10-15) | same | 38d0c26, 2026-10-15 |
| `MK_W48S` | `1` | the one-pass attention keeps its per-call scalars in shared memory (fewer register spills) | same | 6cc40a7, 2026-10-14 |
| `MK_MTP_KVC` (the `-D`) | `1` | the K / V-only MTP catch-up rows compiled in; the `MK_MTP_KVC` environment variable still turns them off at run time ([Speculation](#speculation-drafts)) | same | 455017a, 2026-10-14 |
| `MK_WV_FMK` | `1` | Q4_K / Q5_K main-pass matrices fragment-ordered; `0`: tile-major whole blocks (decode -5.8% at 0 drafts, no gain at 4, 2026-10-13) | not checked | 408fa84, 2026-10-13 |
| `MK_WV_OUT` | `0` | the removed arm put the out / ssm_out matrices of the offset types (Q3_K, IQ3_S) on the tensor-core path at every T: per cycle +1.4% at 38k, no change at 101.6k (2026-10-14) | changes (KL 0.00116 mixed) | 911d288, 2026-10-14 |

### Engine -D flags

| Flag | Default | Served | What it does | Output | Status |
|---|---|---|---|---|---|
| `MK_MAXD` | `8` | `8` | the most tokens of one decode verify pass: `8` (up to 7 drafts) or `16`, the [deep-pass build](EXPERIMENTAL.md#the-deep-pass): `-d` up to 15, cycles of 9-16 tokens verified as one 16-row pass, drafts 8-15 only with the q8_0 cache and no `MK_KV` / `MK_KV_WIN`. Needs `-DMK_PAR_RMAX` >= 16 (its default) | [EXPERIMENTAL.md](EXPERIMENTAL.md#the-deep-pass) | opt-in (in development) |
| `MK_ATT_I8` | `0` | `0` | experiment build ([EXPERIMENTAL.md](EXPERIMENTAL.md#int8-decode-attention)): int8 Q K^T in the decode attention of the all-q8_0 slots, used when the environment variable `MK_ATT_I8=1` is set. One score helper serves every attention kernel, so draft depths and parallel slots stay exact with it; at 40.5k tokens of context attention took 12% less time per 8-row pass, 3-5 rows unchanged (2026-11-02). In active development and on the list for the served build; not part of it yet, its KL divergence gate is not settled | changes | experiment (in development) |
| `MK_L2_TUNE` | `0` | `0` | experiment build ([EXPERIMENTAL.md](EXPERIMENTAL.md#l2-prefetch-budgets)): the L2 prefetch budgets read from the `MK_L2_*` environment variables, to tune them on a GPU with another L2 size. Without it the device code is unchanged. Hints only: no value changes a token. On an RTX 3090 no setting beat the defaults (2026-11-04) | same | experiment (in development) |
| `MK_FD_NOINL` | `0x1c` | `0x1c` | bit mask of the token counts whose ffn_down rows run out of line (T 2-4); inline they spilled and the phase ran 0.17-0.25 ms per pass slower | same | served |
| `MK_W48X` | `1` | `1` | the decode attention of low-bit `MK_KV` slots with the cheaper K / V copies and expansion (`attn_core_q8m`, `attn_core_w48m`: the same staged bytes and partials); `0`: the earlier path. Compile-time because a run-time switch moves ptxas's register allocation | same | served |
| `MK_W48X_Q8` | `0` | `0` | with `MK_W48X=1`: the all-q8_0 slots run those kernels too, instead of their own (no q8_0 instances in the build) | not checked | option |
| `A9_RS`, `A9_CP`, `A9_EX` | `1`, `1`, `2` | same | prefill attention (`pk_attn9`, `prefill_attn.cuh`): `A9_RS` the row-sum chain of each warp pair's second half runs in every warp without a branch, inside the P V block; `A9_CP` bit 1 issues a tile's copies inside the S loop instead of at the barrier (bit 2: each copy's source from its base, no gain); `A9_EX=2` the softmax's exponentials as bare `ex2.approx.ftz` when no argument of the warp is below -126, where `exp2f` is exactly that instruction, decided from each lane's row minima (`1`: decided after a speculative pass, 3-11% slower when scores need `exp2f`'s subnormal path). All three: prefill attention -4.5% at 101k (q8_0 and D, real prompt); with the server's exact reuse (`MK_EXACT=1`, whose forms of the kernel these are too) -5.3% at 101k on q8_0, -3.7% on D, -3.5% at 258k on F; the same outputs (attn_bench's matrix with and without warp jitter, the exactness suite, 40k token ids F and D). `0`, `0`, `0`: the earlier kernel, instruction for instruction | same | served |
| `A9_MX` | `0` | `0` | `pk_attn9`: row maxima from the pair's lane maxima without shuffle rounds (slower on F) | same | option |
| `MK_Q32N` | `1` | `1` | resid_wv / resid_ck_last's K2 combine: a batch's `q32_store` chains interleaved (`q32_store_n`) instead of one token after another; `0`: the earlier order | same | served |
| `MK_FORMAL_SYNC` | `1` | `1` | the PTX memory model's fences and `__syncwarp`s at three hand-offs: lm_head's last-block ticket, the softmax groups' partials, resid's last arrival. `0`: the earlier code, which relied on the hardware's ordering there | same | served |
| `MK_FFNA` | `7` | `7` | bits for the wv phase tops: 1 a per-block dealing table (`cDeal`, checked against the device formulas at load: the log says so) and ffn_gu_wv's packed loop word, 2 the ssq loads first, 4 the meta-first fill, 8 the table's dealing for dn_proj_wv / attn_proj_wv too (off: their T = 8 instances spilled) | same | served |
| `MK_WVREMAT` | `1` | `1` | needs `MK_FFNA` bit 1 or 8 (its dealing table): dn_proj_wv / attn_proj_wv take the layer index and rebuild their loop state after each tile call, so only a packed word stays live across it, however ptxas allocates the rest | same | served |
| `MK_W48Z` | `1` | `1` | preset D / F decode attention at T = 8 (`attn_core_w48m`): the accumulator tile its S warps never read is re-zeroed and the next k-step's Q loads issue ahead of this one's MMAs | same | served |
| `MK_K6X` | `1` | `1` | preset D at T = 8: the 6-bit K expansion runs on the P V warps | same | served |
| `MK_LMH8` | `1` | `1` | lm_head at T = 8: the pen drafts staged in shared memory and loaded per row pair, so their addresses are not spilled across the row loop | same | served |
| `MK_SILUPAD` | `1` | `1` | ffn_gu_wv at T < 8: the padding columns feed silu a 1 (the division's fast path) instead of 0; the real columns are unchanged | same | served |
| `MK_NSEQ` | `4` | `4` | the most parallel slots the engine opens (`--agents` 1 to this): the lanes of a batched launch, each with its own views of the row buffers | same | served |
| `MK_PAR_RMAX` | `16` | `16` | rows of a batched slot pass (16 beat 24 at 3-4 slots: 345 vs 322 tok/s at 4, temperature 0.6); a value up to 8 makes the passes 8 rows (not with `MK_MAXD=16`). The wide pass also needs every main matrix but ffn gate / up in the wide-verify layout and enough SMs | same | served |
| `MK_K2_TB` | `8` | `8` | tokens per batch of the split-K residual combine (8 beat 4 by 0.5-0.8 us per call at 38k) | same | option |
| `MK_CK_PFL` | `12` | `12` | 128-byte lines per warp and matrix that the 16-row phases prefetch to L2 of the next sub-chunk's weights, before each activation fill | same | option |
| `MK_K6_DEF`, `MK_K8_DEF`, `MK_K8_NT` | `6`, `6`, `4` | same | the defaults of the `MK_K6`, `MK_K8`, `MK_K8_NT` environment variables | - | option |
| `MK_PF_FRAG_G` | `-1` | `-1` | prefill GEMMs: the MMA group of a stage after which the next stage's scales and fragments are made (`-1`: the kernel's own) | not checked | option |
| `MK_PF_NS_IQ4` | `4` | `4` | cp.async stages of the IQ4_XS prefill GEMMs (3 stages were 2-5% slower) | not checked | option |
| `MK_PF_INFCHK` | `0` | `0` | debug build: counts fp16 overflows (inf / NaN, clamps) per prefill GEMM and layer, and the DeltaNet scan's fp16 MMA operands past 65504 (not clamped), printed after each prefill; none on 38k and 101.6k prompts (2026-10-17, before the scan's counters) | same | debug |
| `MK_SPEC_LOG` | `0` | `0` | analysis build: with `MK_SPEC_LOG_FILE=path`, one JSON line per decode cycle (drafts, their MTP top logits, verify rows, outcome) | not checked | debug |
| `MK_TEST_NAN` | `0` | `0` | test build: `MK_TEST_NAN_AT=N` writes NaN K scales into the first attention layer before decode launch N; the engine must stop with an error and the server answer HTTP 500 (or an error event in a started stream) and recover | - | debug |
| `MK_TEST_SLOTS_FAIL` | `0` | `0` | test build: `MK_TEST_SLOTS_FAIL_AT=k` makes `slots_open` fail (an `MKCudaError`) once slot k is built, so its roll-back runs over real slots (an out-of-memory failure cannot be forced where allocations spill past the VRAM); the next open must succeed | - | debug |
| `MK_KV_EMU` | `0` | `0` | accuracy build: `MK_KV_EMU_SPEC` rounds chosen KV layers to a lower-bit grid inside the q8_0 layout, so formats can be scored per layer before kernels exist (CLI `--kld-specs`) | changes | debug |
| `MKV_NO_VIT` | unset | unset on x86-64 | vision library without the AVX2 / FMA encoder (libmtmd only); the Makefile sets it on other CPUs | changes (image rows) | option |

### Make and build.ps1 variables

| Variable | Default | What it does |
|---|---|---|
| `MKFLAGS` (make) / `-MkFlags` (build.ps1) | empty (the served build) | extra engine `-D` flags (debug and experiment builds; `-DMK_MAXD=16`, the deep-pass build) |
| `ARCH` (make) / `-Arch` (build.ps1) | the first GPU's compute capability from nvidia-smi, else `sm_86` | the CUDA architecture, `sm_89` or `89` |
| `LLAMA` (make) / `-Llama` (build.ps1) | `../llama.cpp` | the llama.cpp checkout and build (tokenizer, libmtmd, reference tools) |
| `NVCC` (make) | `/usr/local/cuda/bin/nvcc` | the CUDA compiler |
| `-Cuda` (build.ps1), `CUDA_PATH` | `$env:CUDA_PATH` | the CUDA toolkit folder (with `bin\nvcc.exe`) |
| `-Targets` (build.ps1) | `all` | any of `all`, `engine` (DLL + CLI), `vision`, `tools` |
| `GLIBC_TUNABLES` (make, exported) | `glibc.malloc.hugetlb=1` | lets ptxas's malloc use transparent huge pages: the Linux build runs ~20% faster, same machine code (needs THP in `madvise` or `always` mode; glibc 2.35+) |

`build.ps1` skips the engine's compile when its object is newer than every engine source and was built
with the same flags. Make targets: `all` (the default: CLI, engine library, tools, and the vision library when
llama.cpp's build has libmtmd), `vision`, `check` (the kernel benches' self-checks, no model), `exactness` and `gate`
(`scripts/exactness.py` and `scripts/gate.sh` on the build, with `MK_MODEL=<gguf>`;
[BUILDING.md](BUILDING.md#checking-a-build)), `tools/vit_check`, `tests/gguf_fuzz`, `clean`. `NVFLAGS`, `VISION_INC`,
`VISION_LIBS`, `VIT_OBJ` and `VIT_DEF` are the Makefile's internal variables (`VIT_DEF` is `-DMKV_NO_VIT` on CPUs
other than x86-64); BUILDING.md has the build itself.

## The CLI: qwen38_mk

`qwen38_mk -m model.gguf [options]` runs one prompt (or a batch of them) through the engine and reports the decode
speed; it also scores the engine against reference files. It reads the same engine environment variables as the
server, keeps 1 checkpoint slot unless `MK_CK_SLOTS` is set, and never runs the keep-warm load. Sampling is greedy
unless `--temp` is given. The first 8 commits of this repository are upstream open-jet's; "upstream" marks options
that come from there.

Model and engine (one per process; in `--batch` mode these stay the command line's):

| Option | Default | What it does |
|---|---|---|
| `-m FILE` | required | the GGUF |
| `-c N` | `8192` | context (KV capacity) in tokens, at least 1 and at most the model's context length (262,144) |
| `--kv f16\|q8\|auto` | `auto` | KV cache format; `auto`: q8_0 when fp16 does not fit, or when `MK_KV` is set |
| `--draft-vocab FILE` | none: the full lm_head | the draft head's vocabulary (`draft_vocab_32k.bin`) |
| `--mtp FILE`, `--no-mtp` | the model file's MTP layer | as the server's options (env `MK_MTP`, `MK_NO_MTP=1`) |
| `--batch FILE` | - | one job per line (the job options below plus `-o <prefix>`: stdout to `<prefix>.out`, stderr to `<prefix>.err`), one model load, the engine reset between jobs: each job's output equals a separate run's; `#` lines are comments |

Generation job (the command line's job, and the defaults of every `--batch` line):

| Option | Default | What it does |
|---|---|---|
| `-p TEXT`, `-f FILE` | a prime-number coding prompt | the prompt (wrapped in the chat template unless `--raw`) |
| `--prompt-ids FILE` | - | the prompt as whitespace-separated token ids (checked against the vocabulary) |
| `-n N` | `256` | tokens to generate |
| `-d N` | `2` | drafts per cycle: 0-7, or 0-15 in a [deep-pass build](EXPERIMENTAL.md#the-deep-pass) (more than 4 needs `--kv q8`, more than 7 also no `MK_KV`) |
| `--raw` | off | the prompt as given, without the chat template |
| `--think [xhigh\|medium\|low]` | off; `xhigh` when given without a value | thinking on; `xhigh` and `low` add the template's reasoning-effort system line, `medium` adds none |
| `--turn2 TEXT`, `--turn2-file FILE` | - | a second user turn appended after the first reply (the follow-up path) |
| `--temp T`, `--top-p P`, `--top-k K`, `--min-p P`, `--seed S` | `0` (greedy), `1`, `32`, `0`, `1` | sampling; top-k at most 32 |
| `--presence-penalty P`, `--frequency-penalty F` | `0` | penalties on the generated tokens (vLLM's semantics; exact under speculation) |
| `--policy SPEC` | `MK_POLICY` | the draft policy for this job ([keys](#mk_policy-keys)); in a batch it stays for the jobs after it |
| `--state-dir DIR` | - | post-prefill state files: the first prompt's prefill is restored from DIR when a file of the same prompt, binary, GPU, model and settings is there, else saved there (`MK_STATE_WARM_MS` warms the GPU after a restore) |
| `--digest` | off | print a hash of the sequence state after the first prefill (tests) |
| `--ids` | off | print the prompt's and the generated token ids |
| `--quiet` | off | no generated text, only the statistics |
| `-o PREFIX` | - | `--batch` lines only: where the job's output goes |

Scoring and measurement:

| Option | Default | What it does |
|---|---|---|
| `--kld REF` | - | KL divergence, perplexity and top-token agreement against a reference file of `tools/llama_ref` (or of `--ref-out`) |
| `--ref-out FILE` | - | with `--kld`: also write the engine's own distributions as a reference |
| `--score-path prefill\|decode\|mixed` | `prefill` | which kernels score: the batched prefill, the decode megakernel, or the unscored prefix through the prefill and the scored positions through decode |
| `--kld-images FILE` | - | with `--kld`: the reference's image rows (`tools/vision_ref --images-out`) |
| `--kld-specs SPECS` | - | `-DMK_KV_EMU=1` builds: score the reference once per `;`-separated `MK_KV_EMU_SPEC` value, one model load |
| `--score FILE` | - | teacher-forced top-1 agreement of the prompt plus the token ids in FILE |
| `--bench-phases` | off | per-phase timing of the decode megakernel (with `MK_GSB`: the grid-barrier benchmark) |

Exit codes: 1 for an error the engine reports (bad arguments, a prompt over the context), 3 for a watchdog timeout,
4 for a sticky CUDA error.

A `--state-dir` file is keyed by the binary, the GPU (name, compute capability, SM count), the model file, the context,
the KV formats, the prompt, and every `MK_*` variable in the environment except those that only change decode,
logging, timing or the load path: names starting with `MK_STATE`, `MK_POLICY`, `MK_PROFILE`, `MK_SPEC_LOG`,
`MK_MEMLOG`, `MK_LOADLOG`, `MK_LOAD_`, `MK_WDIGEST`, `MK_WATCHDOG`, `MK_CK_SLOTS`, `MK_BENCH`, `MK_STACK_LIMIT`,
`MK_RESERVE_EARLY`, `MK_VRAM_`, `MK_GSB`, `MK_PF_PROFILE`, `MK_PF_GAPS`, `MK_PF_SYNC`, `MK_CK_DEVICE`, `MK_STORE_`,
`MK_SLOT_RESERVE`, `MK_CUDAMALLOC`, and `MK_K1`, `MK_K2`, `MK_K6`. Any other change saves a new file instead of restoring a
stale one.

## Tools and benchmarks

These are for measuring and checking, not for serving. Each one's usage text is at the top of its source.

| Tool | Options |
|---|---|
| `tools/llama_ref` (llama.cpp's own distributions, the accuracy reference) | `-m FILE`, `-f TEXT` (repeatable), `--ref-out FILE`, `--kld REF` (score another llama.cpp setup against a reference), `--chunk N` (tokens per sequence, default 2048), `--max-chunks N`, `--score-from I` (first scored position, default 0), `-ngl N` (GPU layers, default all), `--kv f16\|q8` (default q8), `-b N` (scored tokens per batch, default 512; 1 = token by token, as llama.cpp generates), `-t N` (threads) |
| `tools/vision_ref` (llama.cpp's distributions for an image prompt, and the image rows it fed) | `-m`, `--mmproj`, `--prompt-file`, `--image` (repeatable), `--ref-out`, `--images-out`, `--cont-file FILE` (continuation text), `--kv f16\|q8`, `-ngl`, `-t`, `--image-max-tokens`, `--text-batch N` (diagnostics: text decoded in batches of N), `--rows-in FILE` (the images' rows from a file instead of the encoder) |
| `tools/mkv_cli` (the vision library on image files: rows, time) | `-m`, `--mmproj`, `-t N`, `--min-tokens N`, `--max-tokens N`, `--fa -1\|0\|1` (libmtmd's flash attention), `--backend auto\|cpu\|mtmd`, `-r N` (repeats), `-o PREFIX` (the rows to `<prefix>.<k>.f32`), `-v` (libmtmd's log), then the images |
| `tools/vit_check` (the CPU encoder against libmtmd: pixels, rows, time; `make tools/vit_check`) | `-m`, `--mmproj`, `-t N`, `-r N`, `--dump PREFIX`, `--max-tokens N`, `--pair` (video frame pairs), then the images |
| `tools/vocab_freq` (builds a draft vocabulary) | `vocab_freq model.gguf N out.bin < files.txt`: the N most frequent tokens of the listed text files, plus every special and control token |
| `tests/gguf_fuzz` (the GGUF parser on malformed files, AddressSanitizer / UBSan) | no options: every check; `--dump FILE`: the parsed keys and tensors of a GGUF; env `GGUF_FUZZ_ONLY="<check>"` runs one check |
| `bench/bench.py` (decode speed, llama-server against the engine, same prompts) | `llama\|mk`, `--spec N` (llama-server's MTP drafts, default 2), `--draft N` (default 3), `-n N` (512), `--ctx N` (8192), `--reps N`, `--port N` (8091), `--dump DIR` (the first repeat's replies as JSON), `--draft-vocab FILE`, `--model FILE`; env `MK_BIN` (the engine binary, default `./qwen38_mk`) |
| `bench/code_bench.py` (code workloads, both engines) | `--ctx-sizes LIST` (default 0,2000,4000,8000,16000,30000), `--reps N` (2), `-n N` (256), `--model FILE` |
| `bench/think_bench.py` (thinking-mode decode, both engines) | `--seeds N` (2), `-n N` (1024) |
| `bench/wv_extract.py` | `[qwen38_mk.cu] [out.cuh]`: copies the engine's dp4a GEMV code from `mk_shared.cuh` and `qwen38_mk.cu` into `bench/wv_ref.cuh` for `wvgemm_bench` (`make check` runs it first) |
| `scripts/gate.sh` (every GPU check of a build: [checking a build](BUILDING.md#checking-a-build)) | `-m GGUF` (default `MK_MODEL`), `--record DIR` (keep the outputs as a reference) or `--ref DIR` (every output, and the inherited `MK_*` environment, must equal the reference's; a check it did not run is a skip), `--out DIR` (default `gate-out`; one holding a recorded reference is refused), `--legs LIST` (default `benches,exactness,pfcap,cli,tinv,slots,capi,kld`), `--model2 GGUF`, `--kld-ref FILE` (the kld leg), `--capi FILE` (default `scripts/capi_checks.py`), `--capi-cases "CASE ..."` (one or more), `--python PY` (with jinja2), `--port N` (default 18190), `--timeout S` (per run, default 1800); exit 1 if a check failed, 2 for bad arguments, a stale build or a refused `--out` |
| `scripts/exactness.py` (the exact-output checks: [checking a build](BUILDING.md#checking-a-build)) | `-m GGUF` (default `MK_MODEL`), `--bin FILE` (the CLI), `--draft-vocab FILE`, `--depths LIST` (default `0,4,7`), `--legs LIST` (default every leg: `depth,mtp,long,policy,chunks,server,park,resume,deep,full,agents`), `--presets LIST` (default `q8,D,F`), `--long-tokens N` (2048), `--long-min N` (1000), `--deep-tokens N` (100000), `--full-tokens N` (258000), `--out DIR` (default `exactness-out`), `--compare DIR` (another run's results must be identical), `--port N` (18190), `--python PY` (the server's, with jinja2), `--timeout S` (1800, per run, server start or request); exit 1 on a FAIL |
| `bench/slots_bench.py` (parallel slots over the C API: speed and exactness; the gate's slots leg) | `MODEL`, `speed\|exact\|stagger\|all` (default `speed`), `--lib FILE`, `--draft-vocab FILE` (`''`: none), `--n N` (slots, default 4), `--nmin N` (speed: the fewest, 1), `--gen N` (tokens per slot, 256), `--draft N` (7), `--ctx N` (positions per slot, 8192), `--temp T` (speed, 0.6), `--reps N` (1), `--long` (3.8k-5.7k-token prompts, past the presets' window), `--budget N` (stagger: the late slot's chunk, 1024), `--lead N` (stagger: steps before it, 8), `--ref-chunked`; exit 1 when an exactness check differs |
| `scripts/capi_checks.py` (the C API: engine errors and roll-backs, out-of-memory inside the free VRAM, penalties, policies, a slot listed twice, a model's values the kernel does not have, the state after a resume, a park and a side request against a fresh prefill, a step right after a restore, an `MK_EXACT` slot's checkpoint) | `LIB MODEL DRAFT_VOCAB [CASE ...]`: the given cases (default: all), each in its own process, PASS / FAIL / SKIP, exit 1 on a FAIL and 2 for an unknown case; env `CAPI_DEVOOM=0` (skip the device out-of-memory cases), `CAPI_COMMIT_MIN_GB` (Windows commit those wait for, default 45) |
| `bench/suite/fetch.py` (downloads the benchmark items and checks each against `bench/suite/items.json`) | `--data DIR` (default `bench/suite/data`), `--sets NAME ...` (default all, and `pf101k`), `--check` (check the files already there, download nothing) |
| `bench/suite/speed_client.py` (decode, prefill, first token, follow-up turns, prompt edits, VRAM; one server) | `--label L`, `--out FILE` (JSON lines, appended), `--url URL` (default `http://127.0.0.1:8080`), `--mode all\|edit` (default all; edit: the warm-up and the edit scenarios only), `--data DIR`, `--gpu N\|none` (nvidia-smi GPU for the VRAM and clock records, default 0) |
| `bench/suite/qa_client.py` (the 282 questions) | `--label`, `--out`, `--url`, `--data`, `--limit N` (the first N items of each set only), then set names (default all) |
| `bench/suite/scratch_client.py` (the 101.6k-token prompt from scratch; a server with `--side-max 0 --park-mib 0`) | `--label`, `--out`, `--url`, `--data` |
| `bench/suite/maxctx_client.py` (speed, needles and LongBench v2 at the full context) | `--ctx N` (the server's context; decides which items fit), `--label`, `--out`, `--url`, `--data`, `--gpu`, `--limit N` |
| `bench/suite/score.py` (scores the question runs offline) | the clients' files, `--data DIR`, `--labels L ...`, `--exec-humaneval` (runs the model-written HumanEval code; without it HumanEval is not scored), `--timeout S` (per HumanEval task, default 15), `--out FILE` (default `scores.json` next to the first file) |
| `bench/suite/summarize.py` (the published tables from the clients' files) | the clients' files (each one a session; a label in several files gets its best session per row), `--scores FILE ...`, `--col LABEL[=TITLE]` (repeatable), `--sub LABEL:KEY[,KEY]=OTHER` (rows of a column from another label), `--out FILE` |

The CUDA microbenchmarks in `bench/` are built by hand (`nvcc ...`, the line is at the top of each file; `attn_bench.cu`:
`make bench/attn_bench`) and take these switches:

| Switch | File | What it does |
|---|---|---|
| `AB_VARS=<digits>` (env) | `attn_bench.cu` | prefill attention kernels to run, the first is the reference (0 `pk_attn8`, 1 `pk_attn9`; default `01`) |
| `EXA=1\|2`, `QROW=1`, `QSCALE`, `SPECIAL=1`, `SEED`, `OUT_REPS`, `BETWEEN`, `DIFF_HIST=1`, `WARM_MS`, `NCU_N` (env) | `attn_bench.cu` | MK_EXACT's form (2: split), the per-row Q scales, inputs, repeated comparisons under other launch timing, where outputs differ; the top of the file |
| `CTX`, `KQ4=1`, `VQ4=1`, `KROT=1`, `TINV=1` (env) | `dattn_bench.cu` | decode attention: cache size, q4 K / V, rotated K, the token-count invariance check |
| `HINT` (-D) | `ffn_bench.cu` | the load cache hint of the replica (0 none, 1 evict-first L2 policy, 2 256-byte L2 fetch) |
| `WV_L2PF`, `WV_LD256`, `WV_NOPOL`, `WV_TILEMAJ`, `WV_SBU`, `WV_SPS`, `WV_RH`, `WV_PF`, `WV_Q6FM`, `WV_Q6HP`, `WV2_ROLL`, `WV_V2` (-D) | `wvgemm_bench.cu` | variants of the tensor-core verify GEMM under test: L2 prefetch distance, 256-byte fetch hint, no evict-first policy, tile-major layout, super-blocks per batch, stage / row tiling parameters, prefetch form, Q6_K fragment-ordered forms, rolling prefetch, the v2 kernel |
| `WV_MODE` (-D) | `wvgemm_bench.cu` | diagnostics: 0 normal, 1 loads only, 2 compute only |
| `WV_QUICK`, `WV_T8ONLY`, `WV_DPMAX5`, `WV_DPONLY`, `WV_Q6ONLY` (-D) | `wvgemm_bench.cu` | shorter experiment builds: fewer token counts, dp4a up to T 5 only, dp4a only, Q6_K only |
| `KVB_Q8M=1`, `KVB_WARM=S`, `KVB_STRESS_OLD` (env); `KVB_NEW_X` (-D, default 1) | `kvattn_bench.cu` (with `bench/kvattn_gen.py`, which cuts the kernels of two revisions out of `qwen38_mk.cu`) | q8_0 data through the new revision's low-bit kernel; seconds of warm-up before timing (default 4.5; 0 under ncu); the stress mode repeats the old kernel instead of the new; the new revision's `MK_W48X` |
| `MMQ_SET=1`, `MMQ_CAND='"file"'` (-D, through `MMQFLAGS`) | `mmq_bench.cu` | plain Q5_K only; candidate kernels from a file, each checked against the kernel it must match |
| `VIT_PROF` (env) | the vision library (`vision/vit_cpu.cpp`) | print a per-stage profile of the CPU image encoder |

## Measurement, test and debug switches

Engine environment variables for profiling, tests and fault finding. None of them is needed to serve. A name without
a value acts when it is set to anything.

| Variable | What it does |
|---|---|
| `MK_PROFILE` | per-phase timing of the decode launches, printed at the end, including a "min-composed" estimate that filters out time-slicing by other GPU users |
| `MK_PROFILE_SEQ` | with `MK_PROFILE`: every barrier position's phase and minimum time (per-layer costs) |
| `MK_PROF_DUMP=N` | with `MK_PROFILE`: the raw barrier stamps (every barrier's phase, end, duration and each block's arrival, in ns) of launch N, counting every megakernel launch and dumped when it is a decode launch, to `MK_PROF_DUMP_FILE` (default stderr); `MK_PROF_DUMP_N=k`: launches N to N + k - 1 (with a file, one per launch: `<file>.<n>`), so a phase's minimum over them filters out an occasional whole-GPU pause; `%p` in the file name becomes the process id, so each pass of Nsight Compute's application replay (one run of the program per pass) keeps its own stamps |
| `MK_PF_PROFILE` | per-category GPU time of the batched prefill; `MK_PF_PROFILE=2`: idempotent kernels are also timed alone (5 repeats, minimum) |
| `MK_PF_GAPS=1` | GPU idle time per prefill |
| `MK_MEMLOG` | device memory per load step: bytes allocated against the drop in free memory (under WSL2 cudaMemGetInfo reads ~550 MiB low, so the in-process deltas are what to trust) |
| `MK_LOADLOG` | the load's time per phase |
| `MK_WDIGEST` | a digest of the weights on the device after the load (to compare loaders) |
| `MK_POISON=1` | every device allocation filled with 0xFF bytes (NaN in fp32 / fp16) instead of zeros, so an output that reads memory nothing wrote changes; the counters that start at zero by design stay zeroed. The served paths give the same output with it |
| `MK_POISON_SKIP=i,j,...` | with `MK_POISON`: allocation numbers to zero instead (bisecting) |
| `MK_POISON_LOG` | with `MK_POISON`: log every allocation's number and size |
| `MK_FAULT=cuda:N`, `sticky:N` | the N-th prefill fails half-way through its layers: with a rejected CUDA call (the engine resets, the server answers with an error and keeps serving) or a trapping kernel (sticky: the process exits with code 4) |
| `MK_TEST_NAN_AT=N` | `-DMK_TEST_NAN=1` builds: NaN K scales in the first attention layer before decode launch N |
| `MK_TEST_SLOTS_FAIL_AT=k` | `-DMK_TEST_SLOTS_FAIL=1` builds: `slots_open` fails once slot k (from 0) is built |
| `MK_KV_EMU_SPEC` | `-DMK_KV_EMU=1` builds: the layers and formats to emulate, as `MK_KV` items `<K\|V\|*><slot>=<bits 2-8><modes>` (modes `r` Hadamard rotation, `s` random signs + rotation, `a` q4_0-style scale, `l` Lloyd-Max codebook with an rms scale, `n` one scale per 256-value row, `t` / `T` variants of `s`; `8` without modes = plain q8_0) plus `sink=N` (positions below N stay q8_0) and `win=N` (an emulated recent window) |
| `MK_SPEC_LOG_FILE=path` | `-DMK_SPEC_LOG=1` builds: append one JSON line per decode cycle |
| `MK_ATT_I8=1` | `-DMK_ATT_I8=1` builds: int8 Q K^T in the decode attention of the all-q8_0 slots ([Engine -D flags](#engine--d-flags)); other builds print that it is ignored |
| `MK_L2_DN`, `MK_L2_ATTN`, `MK_L2_ATTN_CTX`, `MK_L2_TOP`, `MK_L2_CK`, `MK_L2_MTP`, `MK_L2_MTP_R`, `MK_L2_MTP_KEEP` | `-DMK_L2_TUNE=1` builds: the L2 prefetch budgets ([EXPERIMENTAL.md](EXPERIMENTAL.md#l2-prefetch-budgets)); other builds print that they are ignored |
| `MK_KV_WIN_ALL=1` | with `MK_KV_WIN`: rings for the q8_0 sides too; their rows are the same q8_0 rows, so the output must stay byte-identical (a test of the ring paths) |
| `MK_GSB=N` | CLI `--bench-phases`: the grid-barrier benchmark on N blocks instead, every form: `0` upstream's (the engine's), `1`-`3` flag-broadcast and relaxed-polling variants (measured slower: 0.61-0.94 us against 0.51-0.55 us per barrier) |
| `MK_GSB_REPS` | repeats of the barrier benchmark (at least 3, default 15) |

## Not settings

Names in the code that look like settings but are not, and fixed values that cannot be changed without editing the
code:

| Name | What it is |
|---|---|
| `MK_BUILD_COMMIT`, `MK_BUILD_FLAGS` | written by the build into `build/mk_build_info.h` (`build_win\` on Windows): the commit (with `-dirty` for local changes) and `MKFLAGS`. The server reports them as `/props` `build_info` |
| `MK_MAX_GRID` | constant 128: the most blocks of the persistent kernel; a GPU with more SMs runs 128 |
| `MK_BAR_LT`, `MK_EPOCH`, `MK_C1E` | helper macros of the barrier counters that run on across launches (egress, C1) |
| `MK_GLU_MIX_PAIRS`, `MK_GLU_ATTR`, `MK_GLU_GO` | macro lists of the fused gate / up weight-type pairs |
| `MK_SHARED_NS` | the namespace `mk_shared.cuh` puts its definitions in: `mk` in the host files, anonymous in `qwen38_mk.cu` (which defines it empty) |
| `MK_VIEW_INIT` | macro at the entry of every device kernel: the block reads which buffer view (`cView`) it runs on |
| `MK_LANE_OUT` | macro in `lanes.cu`: a scope guard that leaves the engine's own sequence live again when a slot call returns |
| `MK_STR`, `MK_STR2` | stringify helpers |
| `MTP_KVC` | derived: not `MK_KV_EMU` (the K / V-only MTP catch-up rows are compiled in) |
| `MK_STATE`, `MK_LOAD_`, `MK_BENCH` | prefixes in the state-file key's list of ignored variables (`MK_STATE_WARM_MS`, the `MK_LOAD_*` family). No variable named `MK_BENCH...` is read anywhere |
| `environ` | a Windows compatibility name (`compat_win32.h`) |
| `_CRT_SECURE_NO_WARNINGS`, `_CRT_NONSTDC_NO_WARNINGS` | MSVC warning switches in `build.ps1` |
| `NVFLAGS`, `VISION_INC`, `VISION_LIBS`, `VIT_OBJ`, `VIT_DEF` | Makefile internals ([Make and build.ps1 variables](#make-and-buildps1-variables)) |

`MK_API_KEY` in the client examples is only the variable a client (Codex) reads its API key from: the server checks
no keys.

Fixed in the server: at most 64 stop strings per request of at most 1024 characters each; a video's ffmpeg decode
stops after 300 s; the sampling defaults for fields a request leaves out are Unsloth's thinking-mode preset for Qwen3.8
(temperature 1.0, top_p 0.95, top_k 20, min_p 0); top_k is at most 32 (0 or less means 32); one request runs at a
time. Conversation cache: with parking on, a side request with tools must cap its reply at 1024 tokens; a side request
that does not cap its reply reserves room for 8192 rows in the host store; a side request's conversation is parked
with 1 checkpoint, and at most 4 of them are kept; a request without tools is taken for a summary by its answer, 256
tokens or more (thinking aside): a side request that quotes at least `--park-min` tokens, of the conversation beside it;
a longer one that replaced a conversation with tools, of that conversation (a shorter answer gives it normal priority
back); a conversation parked with low priority evicts only side conversations and other low-priority ones (it is not
parked when they do not make room), except the conversation a request without tools has just replaced; a shared prefix
shorter than 1024 tokens is not pinned. A Responses API stream sends a progress event every 10 s during a long
prefill; a reply's write that makes no progress for 60 s ends the request (its client counts as gone); browsers may
keep a CORS preflight's answer for 86400 s.
