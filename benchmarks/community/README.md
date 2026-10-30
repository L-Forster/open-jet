# Community benchmark results

Each file here is one `openjet bench` run on someone's machine. `scripts/bench_table.py` builds the
[community table in the README](../../README.md#results-on-your-gpu) from them.

## Add your result

1. Run `openjet bench --github <your-handle>` with nothing else using the GPU. It takes a few minutes
   and writes `openjet-bench-<gpu>-<time>.json` in the current directory.
2. Open a pull request that adds only that file to this folder. Keep the file as `openjet bench` wrote
   it; put anything worth knowing (power limit, overclock, laptop GPU) in `--notes` when you run it.
3. CI checks the file and shows the table with your row in the run summary. Pull requests that only
   add a results file are merged the same day, and every contributor is named in the next release's
   notes.

No pull request? Open a [Benchmark issue](https://github.com/l-forster/open-jet/issues/new?template=benchmark.yml)
and paste the file there.

## What a run measures

- **Reasoning tok/s**: 4 short coding prompts with thinking on and OpenJet's thinking preset (temperature
  1.0, top_p 0.95, top_k 20, min_p 0, seed 1000), 512 tokens each. This is how coding agents use the
  model. Total generated tokens divided by total decode time.
- **Greedy tok/s**: 4 short chat prompts, thinking off, greedy, 256 tokens each.
- **Prompt and decode at 32K**: a fresh 32K-token prompt of synthetic source code with a coding task:
  prompt processing speed, then decode speed (thinking on, 256 tokens) at that depth. The file also has
  4K and 96K when the configured context window fits them.
- **Recall**: at each depth, a question about a line hidden halfway into the code. Passed when the
  model's answer or reasoning contains it.

Speeds are the timings the model server reports, so HTTP is not included. The same model file, engine
and settings give comparable numbers; the table groups runs by GPU, model file, engine and OS and shows
the median.

Each file records the GPU, VRAM, driver and power limit, OS, CPU, RAM, OpenJet version, model file
name and size, engine and context window, and every request's timings. It holds no host name, user
name or file paths.
