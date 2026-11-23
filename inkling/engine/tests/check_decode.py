"""Token-by-token decode (engine_v0) against the full-sequence reference, on the validation prompts.

    python inkling/engine/tests/check_decode.py [--gguf model.gguf] [--n 2]

Reports, per position: top-1 agreement, max |logit difference|, and KL(reference || engine).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ENGINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENGINE))
from engine_v0 import InklingEngineV0  # noqa: E402

VALIDATION = ENGINE.parent / "reference" / "results" / "2026-11-21-validate.llama.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", default="/home/louis/models/inkling/Inkling-2-Mini-UD-Q4_K_XL-mainline.gguf")
    ap.add_argument("--n", type=int, default=3, help="number of validation prompts")
    args = ap.parse_args()

    runs = json.loads(VALIDATION.read_text())[: args.n]
    t0 = time.time()
    eng = InklingEngineV0(args.gguf, ctx=256)
    print(f"loaded in {time.time() - t0:.0f} s", flush=True)

    seqs = [run["prompt_tokens"] + run["gen_tokens"][:-1] for run in runs]
    refs = []
    for seq in seqs:
        logits, _ = eng.ref.forward(seq)
        refs.append(logits.cpu())
    # drop the reference's dequantised dense weights before decoding
    for name in [n for n in eng.ref._cache if n in eng.q8]:
        del eng.ref._cache[name]
    torch.cuda.empty_cache()

    for run, seq, ref in zip(runs, seqs, refs):
        eng.reset()
        out = []
        t0 = time.time()
        for tok in seq:
            out.append(eng.decode(tok).cpu())
        dt = (time.time() - t0) / len(seq)
        eng_l = torch.stack(out)
        top1 = (eng_l.argmax(-1) == ref.argmax(-1)).float().mean().item()
        maxdiff = (eng_l - ref).abs().max().item()
        lp_r, lp_e = torch.log_softmax(ref.double(), -1), torch.log_softmax(eng_l.double(), -1)
        kl = (lp_r.exp() * (lp_r - lp_e)).sum(-1)
        print(json.dumps({"prompt": run["prompt"][:30], "tokens": len(seq), "top1_agreement": round(top1, 4),
                          "max_abs_logit_diff": round(maxdiff, 5), "kl_mean": float(kl.mean()), "kl_max": float(kl.max()),
                          "ms_per_token": round(dt * 1e3, 1)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
