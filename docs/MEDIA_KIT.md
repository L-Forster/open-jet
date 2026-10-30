# OpenJet media kit

For people and AI tools writing an article, overview or video about OpenJet. It has ready-to-use
descriptions, the facts with the conditions they were measured under, and what not to claim. For a
fuller technical description, see the [AI reference](AI_REFERENCE.md).

## Name

- **OpenJet**: one word, capital O and J. Not "Open Jet", "Openjet" or "OpenJET".
- The command is `openjet`; the PyPI package is `open-jet`.
- The CUDA engine is "OpenJet's engine", part of OpenJet, not a separate product. "Megakernel"
  describes how it works (one persistent kernel launch per decode step), and is the engine's name in
  the source code; please don't use it as a product name.

## Descriptions

**One line**

> OpenJet runs Qwen3.8-27B on your own NVIDIA GPU, 1.5x faster than llama.cpp, and serves it to Claude
> Code, Codex, OpenCode and OpenClaw.

**About 50 words**

> OpenJet is an open-source tool that sets up a local coding model on your own computer and serves it
> to the coding tools you already use. On RTX 30, 40 and 50-series GPUs with 24 GB, it runs
> Qwen3.8-27B with its own CUDA engine, 1.5x faster than llama.cpp in reasoning mode on an RTX 3090.

**About 150 words**

> OpenJet is an open-source (AGPL-3.0) tool for running coding models locally. `openjet setup`
> detects your hardware, picks a model that fits, downloads it, and checks that it works.
> `openjet serve` then serves it over the OpenAI and Anthropic APIs, so Claude Code, Codex CLI,
> OpenCode and OpenClaw can use it without a cloud account; `openjet connect` prints the settings for
> each.
>
> On NVIDIA RTX 30, 40 and 50-series GPUs with 24 GB or more, OpenJet serves Qwen3.8-27B with its own
> CUDA engine: each decode step runs as one persistent kernel launch that also verifies speculative
> tokens drafted by the model's own multi-token-prediction head. On an RTX 3090 it decodes at 78
> tokens per second in reasoning mode, against 51 for llama.cpp with the same speculative decoding,
> with output matching llama.cpp's (KL divergence 0.0009). Other models, smaller GPUs, Macs and CPUs
> run through llama.cpp.

## Facts you can quote

Keep each figure with its conditions. All speed figures are from one RTX 3090 running one
conversation, with Qwen3.8-27B Q4_K_M, against llama.cpp with OpenJet's flags (MTP speculative
decoding, flash attention, q8_0 KV cache).

| Claim | Figure | Condition |
|---|---|---|
| Reasoning-mode decode | 78 vs 51 tok/s (1.5x) | thinking on, temperature 1.0; how coding agents use the model |
| Greedy decode | 107 vs 63 tok/s (1.7x) | mean of 4 chat prompts |
| Greedy code generation | 140 vs 73 tok/s (1.9x) | one prompt: writing a Python function |
| Prompt processing | 1530-1600 vs 1050-1140 tok/s | 2-8K token prompts |
| Output fidelity | KL divergence 0.0009, top-token agreement 99.08% | against a llama.cpp reference on the same weights |
| Supported GPUs (engine) | RTX 30, 40, 50-series, 24 GB+ | Linux, NVIDIA driver 570+; Turing experimental |
| Supported clients | Claude Code, Codex CLI, OpenCode, OpenClaw | via `openjet serve`; any OpenAI- or Anthropic-compatible tool |
| License | AGPL-3.0-only | |

Results from other GPUs, sent by users with `openjet bench`, are in the
[community table](../README.md#results-on-your-gpu). Quote them as user-reported, with the GPU.

## Please don't claim

- **"The fastest way to run Qwen3.8-27B" or "faster than vLLM / SGLang / ExLlama".** OpenJet's
  measurements compare it with llama.cpp only.
- **"2x faster".** 1.9x is the best single prompt; the reasoning-mode figure, which is what coding
  agents see, is 1.5x.
- **"Runs on any GPU" or "runs on 16 GB cards".** The engine needs 24 GB and an RTX 30, 40 or 50-series
  card. Other hardware runs llama.cpp, at llama.cpp's speed.
- **"Works on Windows" for the engine.** Windows builds of the engine are new and untested on
  hardware. The agent and the llama.cpp path run on Windows.
- **"A llama.cpp fork".** The engine has its own kernels; it uses llama.cpp's tokenizer and GGUF
  files. OpenJet installs llama.cpp for other models.
- **"Runs Claude" or "a free Claude Code".** It runs open-weight models (Qwen3.8-27B by default) and
  lets Claude Code, Codex and other tools use them instead of a cloud model.
- **"Same quality as frontier models".** Qwen3.8-27B is a 27-billion-parameter open model; OpenJet
  makes no quality comparison with hosted models.

## Showing it in a video

A demo that works on a supported GPU, start to finish:

```bash
pipx install open-jet
openjet setup --yes            # detects the GPU, downloads the model and engine, tests it
openjet                        # OpenJet's terminal agent: ask it to fix a bug in a repo
openjet serve                  # in another terminal: serve the model to other tools
openjet connect claude-code    # prints the settings; paste them, then run `claude`
openjet bench                  # measures this machine and writes a results file
```

The model download is about 17 GB, so record setup before the session or cut it. `pgrep -af
mk_server.py` prints a line while OpenJet's engine (rather than llama.cpp) is serving; showing it
makes clear which engine produced the speed on screen.

## Assets

- Logo: https://openjet.dev/icon.svg
- Social card: https://openjet.dev/opengraph-image
- Terminal demo still: https://openjet.dev/terminal-demo-poster.jpg

You may use these to refer to OpenJet. Screenshots of your own runs are welcome.

## Links and contact

- Source: https://github.com/L-Forster/open-jet
- Website: https://openjet.dev
- PyPI: https://pypi.org/project/open-jet/
- Engine details and measurements: [megakernel/README.md](../megakernel/README.md)
- Created by Louis ([L-Forster](https://github.com/L-Forster) on GitHub,
  [@FlouisLF](https://x.com/FlouisLF) on X)
- Questions and corrections: [Discord](https://discord.gg/pspKHtExSa) or a
  [GitHub issue](https://github.com/L-Forster/open-jet/issues). If something you published is wrong,
  open an issue and we'll help you fix it.
