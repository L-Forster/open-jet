# Contributing

OpenJet accepts bug reports, docs fixes, examples, and code contributions.

## License And Contributor Terms

By submitting a pull request, patch, or other contribution to this repository,
you agree that:

- your contribution is your original work or you otherwise have the right to submit it
- you are granting the project the rights described in [CLA.md](CLA.md)
- your contribution may be included in this AGPL-3.0-only repository and in future
  OpenJet commercial, hosted, or enterprise offerings

If you are contributing on behalf of an employer or client, make sure they
allow you to agree to those terms before submitting work.

## Development

- Keep changes focused and well scoped.
- Update docs when the public behavior changes.
- Add or update tests when practical.
- Do not include secrets, tokens, or private data in commits.

## Megakernel

The megakernel ([megakernel/](megakernel/README.md)) has only been tuned on an
RTX 3090 and runs one quant mix of Qwen3.8-27B. Two kinds of contribution are
especially welcome:

- tunes for specific hardware (RTX 4090, 5090 and other 24 GB+ cards)
- support for other quantisations of Qwen3.8-27B

Include benchmarks in the pull request, from the same machine before and after
your change:

- the GPU, driver and CUDA version
- decode speed: `python3 bench/bench.py mk --draft 4 -n 512 --draft-vocab draft_vocab_32k.bin`
  (run from `megakernel/`), or `python3 bench/code_bench.py` to compare with
  llama.cpp as well
- for a change to the kernels or a new quant, accuracy against llama.cpp: top
  token agreement and KL divergence, with the commands under "Correctness" in
  the megakernel README
- for a new quant, the repo and filename of the GGUF you tested

A change must not slow down or change the output of the existing RTX 3090 path
unless the pull request says so and shows the numbers.
