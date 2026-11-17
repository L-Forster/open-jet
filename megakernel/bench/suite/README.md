# Benchmark suite

The clients, items and scoring behind the server tables in [docs/BENCHMARKS.md](../../docs/BENCHMARKS.md) (the full run
of 2026-11-06, with preset F's 262,144-token run) and the README: the same prompts, request parameters, item
selection, warm-up requests, timing fields and scoring rules. Point them at this engine's server or at llama.cpp's
`llama-server` (the comparison column) and the numbers are comparable with the published ones, within the
session-to-session spread described there.

No dataset is part of this repository. `fetch.py` downloads them on your machine, picks the items listed in
`items.json` and checks each one against the hash recorded there.

| File | What it is |
|---|---|
| `items.json` | the manifest: the source of each dataset (pinned where the host allows it), per set the id and upstream row of every item and the sha256 of its content; for the built items (needle-in-a-haystack prompts, `pf101k.txt`) the construction parameters and the hash of the finished prompt |
| `fetch.py` | downloads, selects, builds and checks the items |
| `speed_client.py` | decode, prefill, first token, follow-up turns, prompt edits, VRAM |
| `qa_client.py` | the 282 questions |
| `scratch_client.py` | the ~101.6k-token prompt read from scratch |
| `maxctx_client.py` | speed and accuracy at the largest context (262,144 tokens with preset D or F) |
| `parallel_client.py` | several agents at once: decode per request, first token on long contexts, a long prompt arriving while another request decodes (a port of open-jet's `bench/parallel_bench.py`) |
| `score.py` | scores the answers offline (HumanEval only with `--exec-humaneval`) |
| `summarize.py` | the tables |
| `prompts/pf38k.txt` | the 38k-token prompt: the first 100,000 bytes of an earlier version of this engine's `qwen38_mk.cu` |
| `common.py` | shared code |

Python 3.10 or later, standard library only, Linux or Windows. Every path is an argument; the defaults are relative to
this directory. Run the scripts that read downloaded data with `python -I`, so that no module is imported from the
current directory.

## 1. Get the data

```sh
python -I bench/suite/fetch.py            # into bench/suite/data (ignored by git); --data DIR for another place
python -I bench/suite/fetch.py --check    # later: check what is there, download nothing
```

About 470 MB of downloads (465 MB of them LongBench v2), 485 MB on disk; reading LongBench v2 takes ~4 GB of RAM for a
few seconds. The downloads stay under `data/raw/`, so a second run downloads nothing; delete `data/raw/` afterwards if
you need the space.

| Set | Items | Source |
|---|---|---|
| GSM8K | 50 | `openai/gsm8k`, test split |
| MMLU-Pro | 70 (5 per category) | `TIGER-Lab/MMLU-Pro`, test split |
| ARC-Challenge | 50 | `allenai/ai2_arc`, test split |
| MATH-500 | 40 (8 per level), and 10 level-5 problems asked with thinking on | `HuggingFaceH4/MATH-500` |
| HumanEval | 41 (every 4th task) | `openai/openai_humaneval` |
| needle in a haystack | 9 at ~32k / 64k / 100k tokens; 6 at ~190k / 248k / 261k | built from Project Gutenberg #2600 (War and Peace) |
| LongBench v2 | 12 that fit a 120k context; 6 at 125k-258k tokens | `THUDM/LongBench-v2` at revision `2b48e49`, `data.json` |
| `pf101k.txt` | the ~101.6k-token prompt | `prompts/pf38k.txt` followed by llama.cpp b10246's `src/llama-model.cpp` and `src/llama-vocab.cpp`, cut to 325,100 bytes |
| `oj100k.txt` | the parallel client's ~23k-token context | open-jet f246ff0's `src/**/*.py` in sorted order (the text its `parallel_bench.py` sends), cut to 100,000 characters |

The five small sets come from the Hugging Face dataset viewer's rows API, by row number. Each dataset is under its own
license and terms (see its page). The items were chosen once by seeded sampling, on 2026-10-13 (the full-context sets
on 2026-10-14); `fetch.py` does not sample again, it takes the rows the manifest names. The needle prompts are rebuilt
from the manifest's character offsets, so no tokenizer is needed; their token counts, measured with llama.cpp's
tokenizer on the model file, are in the manifest.

An item whose content differs from the manifest is an error that names it: the dataset changed upstream since the
published runs, and results on it would not be comparable. If that happens, open an issue.

If a download fails:

- **HTTP 429** from the rows API: it limits requests per minute; `fetch.py` waits and retries, which can take a few
  minutes.
- **`CERTIFICATE_VERIFY_FAILED`** on Windows: Python checks certificates against the roots Windows has already
  installed, and Windows installs some roots only when a program it trusts first needs them. Open the URL once in a
  browser (or `curl -I <url>`), then run `fetch.py` again.

## 2. Start a server

Each step needs its own server. With this engine, the launcher's defaults are the settings of the published runs (q8_0
KV cache, 7 drafts, exact reuse, side requests and parked conversations on, a 1,024-token prefill chunk above 96k
context). The VRAM rows are for the whole card, so nothing else should run on it.

| Step | Linux / WSL2 | Windows (`cmd`) |
|---|---|---|
| q8_0, 120k context | `scripts/serve.sh 8080 120000` | `scripts\serve.bat 8080 120000` |
| preset D, 120k context | `MK_KV='K*=6,V*=5' MK_KV_WIN=2048 scripts/serve.sh 8080 120000` | `set MK_KV=K*=6,V*=5` and `set MK_KV_WIN=2048`, then `scripts\serve.bat 8080 120000` |
| preset F, 120k context | `MK_KV='K*=5r,V*=5' MK_KV_WIN=2048 scripts/serve.sh 8080 120000` | `set MK_KV=K*=5r,V*=5` and `set MK_KV_WIN=2048`, then `scripts\serve.bat 8080 120000` |
| preset D, 262,144 tokens | `MK_KV='K*=6,V*=5' MK_KV_WIN=2048 scripts/serve.sh 8080 262144` | as above with `262144` |
| preset F, 262,144 tokens | `MK_KV='K*=5r,V*=5' MK_KV_WIN=2048 scripts/serve.sh 8080 262144` | as above with `262144` |
| from-scratch prefill | add `MK_SIDE_MAX=0 MK_PARK_MIB=0` to the line of the configuration | `set MK_SIDE_MAX=0` and `set MK_PARK_MIB=0` as well |

The from-scratch step needs side requests and parking off: with them on, the short request the client sends to clear
the cache runs beside the cached conversation and keeps it, and the long prompt is then served from the cache.

The llama.cpp columns were measured with llama.cpp b12024's `llama-server` (2026-11-06), tuned (q8_0 KV cache, 4 MTP
drafts; no `--spec-type` without drafts), on the same model file:

```sh
llama-server -m Qwen3.8-27B-UD-Q4_K_XL.gguf --mmproj mmproj-BF16.gguf --no-mmproj-offload -ngl 99 -c 120000 -np 1 \
    -fa on -ctk q8_0 -ctv q8_0 -b 2048 -ub 1024 --keep 4096 --spec-type draft-mtp --spec-draft-n-max 4 --jinja \
    --chat-template-kwargs '{"preserve_thinking": true}' --temp 1 --top-p 0.95 --presence-penalty 0.0 \
    --host 127.0.0.1 --port 8080
```

llama-server honours `cache_prompt: false`, so the speed client's own from-scratch request is valid there and no
separate step is needed.

The parallel step serves several agents at once, on unsloth's plain `Qwen3.8-27B-Q4_K_M.gguf` (upstream open-jet's
model). This engine: `python3 server/mk_server.py -m Qwen3.8-27B-Q4_K_M.gguf --agents 4 -d 7 --kv q8 --port 8080`. Upstream,
from an open-jet checkout at f246ff0 with its megakernel built: `python3 megakernel/server/mk_server.py -m
Qwen3.8-27B-Q4_K_M.gguf --agents 4 --port 8080`, and with `--lib <this engine's libqwen38mk.so> -d 7` added, open-jet's
server driving this engine. llama-server: `--parallel 4`, with the context it needs for four 23k-token conversations.

## 3. Run the steps

All clients take `--url` (default `http://127.0.0.1:8080`), `--label` (a name for the configuration, written into every
record) and `--out` (a JSON-lines file; records are appended). Times are for an RTX 3090 on 2026-11-06.

| Step | Server | Command | Time |
|---|---|---|---|
| speed | 120k (q8_0 or preset D) | `python -I bench/suite/speed_client.py --label q8 --out speed.jsonl` | ~3 min |
| questions | the same server, after the speed step | `python -I bench/suite/qa_client.py --label q8 --out qa.jsonl` | ~26 min |
| from-scratch prefill | 120k, side requests and parking off | `python -I bench/suite/scratch_client.py --label q8 --out scratch.jsonl` | ~2 min |
| full context | preset F at 262,144 | `python -I bench/suite/maxctx_client.py --label max_f --ctx 262144 --out max.jsonl` | ~37 min |
| parallel agents | `--agents 4`, Q4_K_M | `python -I bench/suite/parallel_client.py --label agents --n 4 --out parallel.jsonl`, then the same with `--n 1 --tests short,ctx` | ~2 min |
| scores | none | `python -I bench/suite/score.py qa.jsonl max.jsonl --exec-humaneval --out scores.json` | seconds |
| tables | none | `python bench/suite/summarize.py speed.jsonl scratch.jsonl max.jsonl parallel.jsonl --scores scores.json` | seconds |

Repeat the speed, questions and from-scratch steps for each configuration with its own label (the published table: q8_0,
preset D and preset F). Use the same label for a configuration's speed and from-scratch runs: the summary takes the
from-scratch row from `scratch_client.py`'s record of that label. `qa_client.py` resumes an interrupted run (it skips
items already answered under the label), and takes set names to run only some sets. `--limit N` on `qa_client.py` and
`maxctx_client.py` asks only the first N items of each set: a quick check of a setup, not comparable with the published
totals. `speed_client.py --mode edit` repeats only the warm-up and the prompt-edit requests, for re-running those rows.

Each 120k server needs a 24 GB card (whole-card VRAM at idle on the reference machine on 2026-11-06: 21.2 GB with q8_0,
20.1 GB with preset D, 19.8 GB with preset F, 0.6 GB of it the desktop's); preset F's 262,144-token server reached 23.0 GB.
`maxctx_client.py --ctx` must be the server's context: it decides which items fit (the smallest needle item needs
190,191 tokens).

### HumanEval executes model-written code

Without `--exec-humaneval`, `score.py` does not run HumanEval and reports it as "not scored (needs --exec-humaneval)";
the totals then count the other sets only. With it, each completion, together with the task's tests, runs as a Python
program on your machine: a fresh temporary directory per task, `python -I`, 15 seconds (`--timeout`), and no other
isolation. Use a virtual machine or a container if that is not acceptable.

### Reading the results

`summarize.py` prints the published table's rows for each label (`--col LABEL=TITLE` to choose and name the columns),
the question totals and the full-context table, and writes them as JSON with `--out`.

- **Sessions.** The same build runs several percent faster or slower from one session to the next with GPU temperature
  and clocks. The speed client records nvidia-smi's temperature, clocks, fan and power at the start and the end of a
  pass (`--gpu none` turns VRAM and GPU records off, e.g. for a server on another machine). A label found in several
  files counts as several sessions, and each row shows the best one; give each run its own label unless that is what
  you want.
- **Re-runs.** `--sub LABEL:KEY[,KEY]=OTHER` takes some rows of a column from another label, e.g. a step re-run after
  something disturbed it. Keys: the row keys listed in `summarize.py`'s usage, `qa` (all question sets) or `qa.SET`.
  The 2026-11-06 table needed none; the 2026-10-20 one was made with
  `--sub D:edit_msg,edit_sys,qa.gsm8k,qa.mmlu_pro=D_rerun` (a browser opened during the first pass) and its from-scratch
  rows came from separate labels (`--sub q8:pf101k=q8_scratch`).
- **Question time** includes, on this engine, the side requests and parks the questions cause beside the cached long
  conversation of the speed step.
- **Preset D's long-context decode rows** are not like-for-like with q8_0's: the replies differ from 2,048 tokens on,
  and with them the share of accepted drafts (see BENCHMARKS.md).
- **Parallel agents.** Totals are the sums of the per-request decode rates; the short test runs twice and the table
  shows the better pass. The ctx test's first tokens show whether the server kept each conversation (a server that keeps
  four 23k-token conversations answers each in a fraction of a second). In the stagger test the other request's rate
  counts its streamed chunks while the long prompt is processed, as open-jet's script does: one chunk per token on
  servers that stream token by token. Each request's header carries the run's start time, so nothing is reused across
  runs.

The accuracy against llama.cpp's own distributions (KL divergence) is measured with the CLI, not the server: see
[BENCHMARKS.md](../../docs/BENCHMARKS.md#reproducing). The engine's parallel slots can also be checked and timed without
a server: `python3 bench/slots_bench.py <gguf> all` (batched output against each conversation alone, then decode speed
with 1 to 4 slots).
