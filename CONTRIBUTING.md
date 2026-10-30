# Contributing

OpenJet accepts bug reports, hardware results, docs fixes, client recipes, and code
contributions. You do not need to write CUDA to help: most of the tasks below need a
machine, a client you already use, and some care with the exact commands.

## License And Contributor Terms

By submitting a pull request, patch, or other contribution to this repository,
you agree that:

- your contribution is your original work or you otherwise have the right to submit it
- you are granting the project the rights described in [CLA.md](CLA.md)
- your contribution may be included in this AGPL-3.0-only repository and in future
  OpenJet commercial, hosted, or enterprise offerings

If you are contributing on behalf of an employer or client, make sure they
allow you to agree to those terms before submitting work.

## Build And Test

The Python package and its tests, as CI runs them (Python 3.10, 3.11 and 3.12):

```bash
git clone --recurse-submodules https://github.com/l-forster/open-jet.git && cd open-jet
python -m venv .venv && . .venv/bin/activate
pip install -e . pytest
python -m pytest tests/ -q
```

The terminal UI and agent harness (Node 24) live in
[open-jet-harness](https://github.com/L-Forster/open-jet-harness), checked out
as the `ui/` submodule (`git submodule update --init ui` in an existing clone):

```bash
cd ui
npm ci --ignore-scripts --legacy-peer-deps
npm run build && npm test
```

The engine (Linux, NVIDIA GPU, CUDA 12+, or 12.8+ for RTX 50). It needs the repo's
llama.cpp build for tokenization:

```bash
git clone --depth 1 --branch b10246 https://github.com/ggml-org/llama.cpp.git llama.cpp
cmake -S llama.cpp -B llama.cpp/build -DGGML_CUDA=ON -DLLAMA_CURL=OFF
cmake --build llama.cpp/build --config Release -j 2
make -C megakernel NVCC="$(command -v nvcc)"     # qwen38_mk CLI, libqwen38mk.so, tools; ~2 min
make -C megakernel NVCC="$(command -v nvcc)" -j1 libqwen38mk_mixed.so   # UD-Q4_K_XL only; ~11 GiB RAM
```

`make` builds for the installed GPU; `make ARCH=sm_89` overrides it. Serving, benchmarks and
correctness checks are documented in the [engine README](megakernel/README.md).

## Pick A Task

Larger engineering work, in priority order, is in [ROADMAP.md](ROADMAP.md).

### Send your card's numbers

Results from GPUs other than the RTX 3090 are the most useful thing a newcomer can send. After
`openjet setup`, run:

```bash
openjet bench --github <your-handle>
```

It takes a few minutes and writes `openjet-bench-<gpu>-<time>.json`. Open a pull request that adds
only that file to [`benchmarks/community/`](benchmarks/community/), or paste it into a
[Benchmark issue](https://github.com/l-forster/open-jet/issues/new?template=benchmark.yml). It goes
into the community table in the README, and you are named in the next release notes.

### Report a failed run

If setup, the engine or `openjet bench` fails on your machine, that is worth reporting too. Open an
issue titled `Hardware report: <GPU>` and paste this, filled in:

````markdown
**Hardware**
- GPU / VRAM:
- Driver / CUDA (`nvidia-smi`, `nvcc --version`):
- CPU / system RAM:
- OS:

**Setup**
- OpenJet commit or version (`git rev-parse --short HEAD` or `pip show open-jet`):
- How installed (`openjet setup`, source build, `openjet-engine`):
- Model file (repo and filename):
- Engine serving it (megakernel, mixed megakernel, llama.cpp; `pgrep -af mk_server.py`):
- Server command and flags:

**Result**
- Did setup finish? If not, the exact step and error:
- First request worked (`curl http://127.0.0.1:18080/v1/models`, then a chat request):
- Decode speed, with the command that produced it:
- Other GPU load during the run:

**Anything that surprised you**
````

For engine development numbers beyond `openjet bench`, run from `megakernel/`:

```bash
python3 bench/bench.py mk --draft 4 -n 512 --draft-vocab draft_vocab_32k.bin   # one conversation
python3 bench/code_bench.py                                                      # also runs llama.cpp
python3 bench/parallel_bench.py http://127.0.0.1:18080 --n 4                     # several agents, any OpenAI-compatible server
```

### Write a client recipe

The engine serves an OpenAI-compatible API at `http://127.0.0.1:18080/v1`. A recipe shows one
coding client (an editor extension, a CLI agent, a chat UI) working against it, and goes in
`docs/clients/<client>.md`. Pick a client you already use. A recipe should include:

- the client name and version you tested, and the OpenJet commit
- the server command (`--agents 4` if the client runs several sessions)
- the exact client configuration: base URL, model ID as returned by `/v1/models`, and any API
  key placeholder the client insists on
- a streamed request that worked, and a tool call that worked if the client uses tools
- what did not work (sampling settings it ignores, features it expects that the engine lacks;
  see Limitations in the engine README)
- for several sessions, the monitor command: `python3 megakernel/tools/mk_top.py`

Only document what you ran. A short recipe with one verified client is more useful than a
long one covering several untested ones.

### Fix the installation docs

Follow [README setup](README.md#setup), [Installation](docs/installation.md), or the
[engine's standalone installation](megakernel/README.md#standalone-installation) on a machine
that has not had OpenJet installed. Stop at the first step that fails or is unclear, then open a
pull request that fixes that instruction. Include the machine, the command you ran, and the
exact error output. A single corrected step is a complete contribution.

### Tune or extend the engine

The engine ([megakernel/](megakernel/README.md)) is tuned on an RTX 3090 (sm_86); OpenJet
enables it on sm_86, sm_89 and sm_120 cards with 24 GB+, and experimentally on sm_75. It serves
Qwen3.8-27B in two builds:

- `libqwen38mk.so`: unsloth's Q4_K_M mix, plus plain `llama-quantize` Q4_K_M files, which the
  loader re-encodes to that mix. Parallel serving (`--parallel` / `--agents`) runs on this
  build only.
- `libqwen38mk_mixed.so`: UD-Q4_K_XL and other supported mixed tensor recipes.

Other quants fall back to llama.cpp. The [quick guide](megakernel/README.md#quick-guide) and
[The GGUF](megakernel/README.md#the-gguf) list which files run where. Especially welcome:

- tunes for specific hardware (RTX 4090, 5090 and other 24 GB+ cards)
- support for further quantisations of Qwen3.8-27B
- the gaps under Limitations in the engine README

For a large change, open an issue describing the approach before writing it, and split the
work into pull requests that can be reviewed and benchmarked one at a time.

## What A Pull Request Needs

Every pull request:

- does one thing, and says what it changes and why
- passes CI (`pytest` on Python 3.10-3.12 and the UI build and tests)
- adds or updates tests when practical, and updates docs when public behavior changes
- contains no secrets, tokens, model files, build outputs or private data

An engine pull request also includes benchmarks from the same machine before and after the
change:

- the GPU, driver and CUDA version
- decode speed with the commands above; `parallel_bench.py` too if it touches parallel serving
- for a change to the kernels or a new quant, accuracy against llama.cpp: top token agreement
  and KL divergence, with the commands under "Correctness" in the engine README
- for a new quant, the repo and filename of the GGUF you tested

Review checks those numbers against the existing RTX 3090 path. A change must not slow down
or change the output of that path unless the pull request says so and shows the numbers.
Changes that trade accuracy for speed are not accepted.

## How Outside Pull Requests Are Handled

- Pull requests from outside contributors get a review, or are merged, the same day they are opened.
- A results-only pull request (only files in `benchmarks/community/`) is merged once the
  `Check benchmark results` job passes. After merging, `python scripts/bench_table.py` rebuilds the
  README table; commit that on `master`.
- Label pull requests that add results `benchmark`, engine changes `engine` and fixes `bug`. Release
  notes are made with GitHub's **Generate release notes**, which names the author of every merged pull
  request ([.github/release.yml](.github/release.yml) sets the sections). Results sent as issues are
  credited by handle in the same release notes.
