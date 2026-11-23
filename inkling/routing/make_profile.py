"""Expert profile for the engine's VRAM cache: reads per decode token of every (layer, expert), from a routing trace.

    python inkling/routing/make_profile.py --trace T.bin --requests R.jsonl --out inkling/engine/profiles/default.f32

Output: float32 [n_layer][n_expert], little-endian. A looped layer's three passes each count as reads, so the
engine's cache ranks the core layers' experts by how often they are actually read.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import analyze


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", type=Path, required=True)
    ap.add_argument("--requests", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    reqs = analyze.load_requests(args.trace, args.requests)
    counts = analyze.usage_counts(reqs)
    n_tok = sum(r.ids.shape[0] for r in reqs)
    prof = (counts / n_tok).astype(np.float32)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    prof.tofile(args.out)
    print(f"{args.out}: {prof.shape}, {n_tok} decode tokens, mean reads/token {prof.sum():.1f}")


if __name__ == "__main__":
    main()
