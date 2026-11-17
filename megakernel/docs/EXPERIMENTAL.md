# Experimental builds

Two opt-in source builds are in active development and on the list for the served build: the deep pass (up to 15 drafts
per step) and int8 decode attention. A third, the L2 prefetch budgets, is a tool for tuning on other GPUs. None is in
the regular release packages; each builds from source with one flag ([BUILDING.md](BUILDING.md) has the rest of the
build), and the L2 budgets also have a [Windows pre-release](https://github.com/cutlerbenjamin1-cmd/qwen38-megakernel/releases/tag/v2026.11.04-l2tune).
The default build stays the served one: 7 drafts, every KV preset, the full 262,144-token context.

Contents: [the deep pass](#the-deep-pass) - [int8 decode attention](#int8-decode-attention) - [both
together](#both-together) - [speed](#speed) - [how it was measured](#how-it-was-measured) - [L2 prefetch
budgets](#l2-prefetch-budgets).

## The deep pass

The default build verifies at most 8 tokens per cycle (`-d` up to 7; the launchers use 7). A build with `MK_MAXD=16`
also takes `-d` 8 to 15 and verifies those 9 to 16 tokens in one 16-row pass (`main_forward_d`: the matrix products in
the 16-row phases of the `--agents` batched passes, attention as two 8-row calls). It is opt-in and in active
development: the served configuration and the release packages are the default build, so the deep pass needs a source
build, and its support for the low-bit KV presets is on the list.

- **Build:** `make -j2 MKFLAGS=-DMK_MAXD=16` (Linux / WSL2) or `.\build.ps1 -MkFlags "-DMK_MAXD=16"` (Windows). Both
  give the flag to the device object and the seven host objects alike, as the engine requires (it refuses at load a
  device object built with other settings than its host files), and a change of flags recompiles all of them, ptxas
  included ([its RAM and time](BUILDING.md#requirements)). `-DMK_PAR_RMAX` must stay at 16 or more (its default is 16).
  A server reports the build in `/props` (`build_info`: `-DMK_MAXD=16`).
- **Drafts 8-15 need** the q8_0 KV cache (`--kv q8`) with 8-bit K and V in the 16 attention layers: no `MK_KV` preset
  (C, D, D', E or F) and no `MK_KV_WIN` rings. Low-bit presets with the deep pass are on the list, not supported yet.
  They also need every main matrix but ffn gate / up in the wide-verify layout, and at least 80 SMs. With `-d` above 7
  and one of these missing, the load fails with `n_draft > 7 (the deep pass) needs <the first one missing>`. At 7 drafts
  or fewer none of this is checked: a deep-pass build loads with the presets as the default build does.
- **`scripts\serve.bat` above 240,000 tokens of context** sets preset F unless `MK_KV` is set, so with `MK_DRAFT`
  above 7 a deep-pass build fails to load there. Keep `MK_DRAFT` at 7 or below for contexts that need a preset (the
  full 262,144 tokens on a 24 GB card).
- **When a cycle takes the deep pass:** only when the MTP head's own probabilities expect at least 3 of the drafts past
  the 7th to be accepted (`MK_POLICY` key `tier`, [CONFIGURATION.md](CONFIGURATION.md#mk_policy-keys); `tier=0` drops
  this condition; with sampled drafts the estimate uses only the probabilities up to the 8th draft's since 2e64a2c:
  deciding on the later ones biased sampled output, and the change's speed is not measured yet) and its 16 positions
  lie in one attention partition, as the 8-row pass's 8 must; other cycles verify up to 8 tokens, as the default build
  does. Parallel slots (`--agents`) that decode together in one batched pass verify at most 8 tokens each in either
  build; a slot that decodes alone (one busy agent, or `--agents 1`) steps as a single sequence and can take the deep
  pass.
- **Output:** the default build's. Greedy output at 0, 7 and 15 drafts, and a seeded sampled run, matched the default
  build's at 7 drafts on short prompts and on a 40.5k-token prompt (512 tokens), also with every cycle forced through
  the 16-row pass (2026-11-03). `MK_POLICY=stop=0,drop=0,droplast=0,tier=0 scripts/exactness.py -m <gguf> --depths
  0,7,15` runs the [exactness checks](BUILDING.md#checking-a-build) that way: every cycle at 15 drafts then takes the
  16-row pass.
- **Speed:** [below](#speed).
- **On the list:** the low-bit KV presets with drafts 8-15, and int8 attention in the 16-row pass ([both
  together](#both-together)).

## Int8 decode attention

The decode attention computes Q K^T in int8 for the attention layers whose K and V are both 8-bit (the q8_0 cache): each
query row is quantized per 32 dimensions as the cache's K is, and a score is the sum of the blocks' exact int32 dot
products times the two scales. One function computes the scores for every attention kernel, so the engine's exactness
holds with it on: the same output at every draft count, from parallel slots as from a single sequence. It is not
bit-identical to the default fp16 Q.

- **Build and run:** `make -j2 MKFLAGS=-DMK_ATT_I8=1` (Linux / WSL2) or `.\build.ps1 -MkFlags "-DMK_ATT_I8=1"`
  (Windows), then `MK_ATT_I8=1` in the environment at run time (the q8_0 KV cache; slots of a low-bit `MK_KV` preset
  keep fp16 Q). Without the variable an int8 build runs the default attention and gives the default build's output. A
  default build prints that `MK_ATT_I8` is ignored.
- **Cost to compile:** ptxas peaked at 26.1 GiB in 17 minutes, against 21.0 GiB in 12 for the default build (its eleven
  extra attention kernels). This is why it is not in the default build.
- **Accuracy** (2026-11-02, an earlier build of the same code): against llama.cpp's reference distributions on five
  reference sets, KL divergence within +0.00002 of the fp16 path, and top-1 agreement with llama.cpp as high as fp16's
  (100k context: 0.05 points lower). Its top token matched the fp16 path's at 99.27% of positions (99.24% in decode),
  under the 99.5% that was proposed as the bar. The bar is not settled; until it is, int8 attention stays opt-in.
- **Exactness** (2026-11-03): with `MK_ATT_I8=1`, greedy output identical at 0, 4 and 7 drafts and without the MTP
  layer, the same tokens as an earlier int8 build of this branch, and four parallel slots identical to each conversation
  alone. With the variable unset, the 40.5k-token prompt's 512 tokens matched the default build's.
- **Speed** ([below](#speed)): 12% less attention time per 8-row pass at 40k tokens of context (2026-11-02); decode
  +1.4% after a 40.5k-token prompt, but 2.7% slower on short prompts, about half of that from building it in.
- **On the list:** a settled accuracy bar, then int8 attention in the served build; the deep pass with it is below.

## Both together

`make -j2 MKFLAGS="-DMK_MAXD=16 -DMK_ATT_I8=1"` builds the deep pass with int8 attention
(`.\build.ps1 -MkFlags "-DMK_MAXD=16 -DMK_ATT_I8=1"` on Windows). ptxas peaked at 27.0 GiB in 17 minutes. With
`MK_ATT_I8=1` its greedy output at 0, 7 and 15 drafts matched an earlier deep build of this branch with int8 attention,
and with the variable unset it matched the deep build (2026-11-03). Speed: [below](#speed).

## Speed

Decode rates on an RTX 3090, two sessions on 2026-11-03 ([how](#how-it-was-measured)); each row against the default
build of the same session.

| Build | Drafts | Short prompts | vs default | After a 40.5k-token prompt | vs default |
|---|---|---|---|---|---|
| v2026.11.01's engine (default build) | 7 | 155.9 tok/s | -5.2% | 115.9 tok/s | -5.4% |
| **this release, default build** | 7 | **164.4 tok/s** | | **122.5 tok/s** | |
| deep pass | 15 | 168.8 tok/s | +2.7% | 121.9 tok/s | -0.5% |
| int8 attention (`MK_ATT_I8=1`) | 7 | 159.9 tok/s | -2.7% | 124.2 tok/s | +1.4% |
| *second session:* default build | 7 | 165.0 tok/s | | 122.1 tok/s | |
| int8 build, `MK_ATT_I8` unset | 7 | 162.7 tok/s | -1.4% | 122.0 tok/s | -0.1% |
| deep pass | 15 | 170.1 tok/s | +3.1% | 122.2 tok/s | +0.1% |
| deep pass and int8 attention | 15 | 163.4 tok/s | -1.0% | 124.3 tok/s | +1.8% |

- **The default build** of this release decodes 5.5% faster on short prompts and 5.7% faster after the 40.5k-token
  prompt than v2026.11.01's engine, and reads that prompt in 22.6 s instead of 23.3 s. Every experimental build reads it
  in the same 22.5-22.8 s: neither option changes prompt processing.
- **The deep pass** gains on short prompts (+2.7% and +3.1%); after the 40.5k-token prompt it runs at the default
  build's rate.
- **Int8 attention** gains after the long prompt (+1.4%, +1.8% with the deep pass) and loses on short prompts: -2.7%
  with it on, -1.4% with it only built in, so about half of that loss comes from the extra kernels in the build.

## How it was measured

An RTX 3090 (driver 610.62, 480 W power limit) under WSL2, the UD-Q4_K_XL model, the q8_0 KV cache, and the CLI with
the default draft policy, greedy: `qwen38_mk -m <gguf> --kv q8 -n 512 -d 7` (`-d 15` for the deep pass).

- **Short prompts:** the four prompts of `bench/bench.py` (`python3 bench/bench.py mk --draft N -n 512 --reps 2`, which
  passes `-c 8192`), the mean of the request rates.
- **The 40.5k-token prompt:** 40,509 tokens (the start of an earlier version of this engine's `qwen38_mk.cu`, then a
  question), `-c 65536`: the decode rate of the 512 tokens and the time to read the prompt.
- **Sessions:** each ran its builds interleaved, two rounds, with nothing else on the GPU; the rows are the means of the
  two rounds, which agreed within about 1%. A build's rate moves by a few percent from one session to another with GPU
  clocks and temperature ([BENCHMARKS.md](BENCHMARKS.md)), so compare rows within a session.

## L2 prefetch budgets

While DRAM would otherwise sit idle, the engine prefetches weights to L2. How much it prefetches is set by constants
sized for the 6 MB L2 of an RTX 3090, the only GPU the engine has been tuned on. A build with `MK_L2_TUNE=1` reads them
from the environment, so a GPU with another L2 (an RTX 4090 has 72 MB) can try other values without a device rebuild for
each one. This is a tool for tuning, not a faster build: on an RTX 3090 no setting beat the defaults
([below](#on-an-rtx-3090)), and nobody has measured it on another GPU yet.

- **Build and run:** `make -j2 MKFLAGS=-DMK_L2_TUNE=1` (Linux / WSL2) or `.\build.ps1 -MkFlags "-DMK_L2_TUNE=1"`
  (Windows), then the variables below at run time. The load prints the values in use (`[mk] MK_L2_TUNE: ...`) and
  refuses a value out of range. A server reports the build in `/props` (`build_info`). A default build prints that the
  variables are ignored; its device code is unchanged, the same PTX as without this feature.
- **Cost to compile:** ptxas peaked at 21.5 GiB in 13 minutes, against 21.0 GiB in 12 for the default build. The numeric
  budgets compile to almost the same code: 8 of 587 functions changed their spills, by 20 bytes or less.
- **Not covered:** prompt processing, which has its own kernels. Also not the eviction hint on the weight loads
  themselves: they load `evict_first`, compiled in, because a run-time switch in the hottest loads would slow the thing
  being measured.

| Variable | Default | Largest | What it sets |
|---|---|---|---|
| `MK_L2_DN` | `4096` | `65536` | bytes of `ssm_out` per item (1,280 items, the phase-top unit included) that `dn_core` prefetches for `resid_sout`. On Q4_K 11,520 is the whole matrix, 14.7 MB |
| `MK_L2_ATTN` | `8` | `256` | 128-byte lines of `resid_o` per item, past the phase-top unit, that the blocks without an attention item prefetch. On Q4_K 72 is all of the matrix past those units |
| `MK_L2_ATTN_CTX` | `1280` | `1073741824` | the longest context at which those blocks prefetch |
| `MK_L2_TOP` | `1` | `20` | weight units per warp that each phase prefetches at its top, before its activation fill |
| `MK_L2_CK` | `12` | `32` | lines per warp and matrix before a sub-chunk fill of the 16-row phases (the batched slots of `--agents`, the deep pass); other passes do not use it |
| `MK_L2_MTP` | `11` | `64` | lines per row of a warp's first row pair in the MTP's row-pair phases (a Q4_K row is 23 lines) |
| `MK_L2_MTP_R` | `12` | `128` | lines per row of a block's first 32 rows in the MTP's other phases |
| `MK_L2_MTP_KEEP` | `0` | `1` | `1`: the MTP's prefetches go out `evict_last`, to keep those lines in L2 for the next draft step, which reads the same weights |

`0` turns a budget off. A budget larger than a matrix holds stops at the end of what that prefetch reads. The values
beyond the constants' old reach are the point: one prefetched unit per item, or one line per thread, was all a 6 MB L2
could use.

Attention deals at most 80 items (`attn_dims`, 4 KV heads x 20 chunks), so a GPU with more than 80 SMs has blocks
without an item at every context: at least 48 of them with 128 SMs, at least 2 with 82. That is why `MK_L2_ATTN_CTX` may
matter more there.

`MK_L2_MTP_KEEP` is a prefetch hint and not a CUDA persisting L2 window because a window does not reach these weights.
On an RTX 3090 (a 1.5 MB slice read back after three passes of a 1 GiB stream, one thread, cycles per load; 130 is a
hit, 194 a miss), a window kept the slice for plain loads (130) but not for loads with the weights' explicit
`evict_first` hint (194). An `evict_last` prefetch kept it through those same loads and an `evict_first` stream (130),
and held up to about 4.5 MB of the 6 MB L2 (139; 152 at 5.6 MB, 194 at 8 MB).

### On an RTX 3090

Measured on 2026-11-04 (driver 610.62, 480 W power limit, WSL2, the UD-Q4_K_XL model, the q8_0 KV cache, 7 drafts,
greedy).

- **Output:** prefetches are hints, and no value changed a token. The ids of 13 runs (three prompts at 0, 4 and 7
  drafts, the same prompts sampled, a 22.6k-token prompt) equal v2026.11.03's with every budget at its default, at `0`,
  at its largest and at a middle set, on UD-Q4_K_XL; at the default, `0` and the largest on Q4_K_M; at the largest with
  preset D. [`scripts/gate.sh`](BUILDING.md#checking-a-build) passes against the same reference as v2026.11.03, and
  `scripts/exactness.py` gives the same results at `0` and at the largest values as at the defaults. The slot checks of
  the gate pass at the largest values.
- **The build at its defaults costs nothing measurable:** three interleaved rounds against the default build, 163.5 vs
  163.6 tok/s on short prompts, 122.2 vs 122.6 tok/s after a 57k-token prompt, reading it at 1,652 vs 1,654 tok/s.
- **No setting beats the defaults:** `scripts/l2_sweep.py --grid full` (18 settings, two rounds) leaves all of them
  within the defaults' own change from run to run, 1.3% on short prompts and 1.8% after a 51.8k-token prompt. The
  largest mean changes were +0.5% and -1.2% (`MK_L2_TOP=5`, whose two rounds differed by 2.1%). Every budget at its
  largest does cost something where it shows: one run of 4 slots gave 388 tok/s against 413, of 2 slots 217 against 246.
- **The settings do act on the phases they target,** so that flat table is a result and not a setting that does nothing.
  `MK_PROFILE` on one short prompt, the change in time per call against three runs of the defaults:

| Setting | Slower | Faster |
|---|---|---|
| `MK_L2_DN=65536` | `dn_core` 12.4 to 17.2 us, issuing the prefetches | `resid_sout` 36.2 to 35.1 us |
| `MK_L2_DN=0` | `resid_sout` +2.4% | `dn_core` -2.3% |
| `MK_L2_ATTN=72` | `attn_core` 17.7 to 19.8 us | `resid_o` -1.0% (within its noise) |
| `MK_L2_ATTN=0` | `resid_o` +1.3% (within its noise) | `attn_core` -5.0% |
| `MK_L2_TOP=0` | `attn_proj`, `dn_proj`, `ffn_down` +1.4 to +1.7% | |
| `MK_L2_MTP=0,MK_L2_MTP_R=0` | `mtp_attn_proj`, `mtp_resid_o` +4% | `mtp_m0` -22%, `mtp_attn_core` -17% |

On 6 MB these trades cancel: `MK_L2_DN=65536` costs 48 `dn_core` calls about 230 us per cycle and gives back about 50 us
in `resid_sout` (the matrix is 14.7 MB, more than that L2 holds). What a larger L2 does to that balance is the open
question.

### Trying it on another GPU

The [v2026.11.04-l2tune pre-release](https://github.com/cutlerbenjamin1-cmd/qwen38-megakernel/releases/tag/v2026.11.04-l2tune) is this build for Windows, so
this takes no compiler: `scripts/l2_sweep.py` needs Python, the build's CLI and the GPU to itself. On an RTX 40 card an sm_86
build runs the same code as an sm_89 one: with CUDA 13.2.2 all 157 code sections of this build came out byte-identical
for the two architectures (2026-11-04), and only the ELF header's architecture differs.

```
python scripts/l2_sweep.py -m <gguf>                  # 7 settings, about 25 minutes
python scripts/l2_sweep.py -m <gguf> --grid full      # 18 settings, about 55 minutes
python scripts/l2_sweep.py -m <gguf> --set MK_L2_DN=11520,MK_L2_TOP=2 --set MK_L2_MTP_KEEP=1
```

Each setting runs in its own process, compared with runs of the defaults on both sides of it. It decodes four short
prompts and a long one greedily, and a change smaller than the defaults' own change from run to run is marked `~`. Every
setting must give the same tokens: a difference is reported as `OUTPUT CHANGED`, which would be a bug. `--profile` adds
the phase tables above, one process per setting. The table goes to `results.md` in `--out` (default `l2_sweep_out`): it
is the thing to send, with the GPU's name in its first line.
