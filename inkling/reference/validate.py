"""Check the reference forward pass against llama.cpp on the same tokens.

1. llama-server (with the routing-trace patch) runs a prompt and greedily continues it, returning the top-10
   next-token probabilities at each generated position and, through LLAMA_ROUTE_TRACE, the experts it chose.
2. The reference runs prompt + continuation in one pass.
3. Reported: top-1 agreement and probability differences at the generated positions, and how many of each
   layer application's 8 experts match llama.cpp's choice, for prompt and generated tokens.

    python inkling/reference/validate.py --gguf model.gguf --out inkling/reference/results/validate.json
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "routing"))

from inkling_ref import InklingRef  # noqa: E402
import analyze  # noqa: E402

PROMPTS = [
    "def merge_sorted(a, b):\n    \"\"\"Merge two sorted lists into one sorted list.\"\"\"\n",
    "The three main reasons the Roman Republic collapsed were",
    "请用一句话解释什么是混合专家模型：",
]


def post(url, body):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())


def run_llama(args, trace: Path) -> list[dict]:
    trace.write_bytes(b"")
    env = dict(os.environ, LLAMA_ROUTE_TRACE=str(trace))
    log = open(trace.with_suffix(".server.log"), "w")
    srv = subprocess.Popen([args.llama, "-m", args.gguf, "-ngl", "99", "-c", "4096", "-np", "1", "--port", str(args.port),
                            "--host", "127.0.0.1", "--no-warmup"], env=env, stdout=log, stderr=subprocess.STDOUT)
    try:
        url = f"http://127.0.0.1:{args.port}"
        for _ in range(300):
            try:
                urllib.request.urlopen(url + "/health", timeout=2)
                break
            except Exception:
                if srv.poll() is not None:
                    raise RuntimeError("llama-server exited")
                time.sleep(2)
        out = []
        for p in PROMPTS:
            toks = post(url + "/tokenize", {"content": p, "add_special": True})["tokens"]
            start = trace.stat().st_size
            r = post(url + "/completion", {"prompt": toks, "n_predict": args.n_gen, "temperature": 0.0, "top_k": 1,
                                           "n_probs": 10, "post_sampling_probs": False, "cache_prompt": False,
                                           "return_tokens": True, "samplers": ["top_k"]})
            end = trace.stat().st_size
            gen = r["tokens"]
            probs = [[(c["id"], c["prob"] if "prob" in c else float(np.exp(c["logprob"]))) for c in (step.get("top_logprobs") or step.get("top_probs") or [])]
                     for step in r["completion_probabilities"]]
            out.append({"prompt": p, "prompt_tokens": toks, "gen_tokens": gen, "top": probs,
                        "trace_start": start, "trace_end": end})
        return out
    finally:
        srv.send_signal(signal.SIGINT)
        srv.wait(timeout=60)
        log.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--llama", default="/home/louis/bench-engines/llama.cpp-inkling/build/bin/llama-server")
    ap.add_argument("--port", type=int, default=8097)
    ap.add_argument("--n-gen", type=int, default=24)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--reuse", action="store_true", help="reuse the llama.cpp results saved next to --out")
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    trace = args.out.with_suffix(".trace.bin")
    saved = args.out.with_suffix(".llama.json")

    if args.reuse and saved.exists():
        runs = json.loads(saved.read_text())
    else:
        runs = run_llama(args, trace)
        saved.write_text(json.dumps(runs, ensure_ascii=False))

    m = InklingRef(args.gguf)
    report = []
    for run in runs:
        P, G = len(run["prompt_tokens"]), len(run["gen_tokens"])
        seq = run["prompt_tokens"] + run["gen_tokens"][:-1]
        logits, routing = m.forward(seq)
        probs = torch.softmax(logits.double(), dim=-1)

        # next-token agreement at the generated positions: position P-1+i predicts gen token i
        top1 = dmax = 0
        diffs = []
        for i in range(G):
            row = probs[P - 1 + i]
            ref_top = int(torch.argmax(row))
            top1 += ref_top == run["gen_tokens"][i]
            for tid, pl in run["top"][i][:5]:
                diffs.append(abs(float(row[tid]) - pl))
        # expert agreement: llama.cpp evaluations are the prompt batch, then one per generated token
        evals = analyze.read_evals(trace, run["trace_start"], run["trace_end"])
        ll = np.concatenate([e[0] for e in evals])               # [P + G - 1, 48, 8]
        ref = np.stack([r.numpy() for r in routing], axis=1)     # [P + G - 1, 48, 8]
        n = min(len(ll), len(ref))
        overlap = np.array([[len(set(ll[t, a]) & set(ref[t, a])) / 8 for a in range(48)] for t in range(n)])
        report.append({
            "prompt": run["prompt"][:40], "prompt_tokens": P, "gen_tokens": G,
            "top1_agreement": top1 / G,
            "mean_abs_prob_diff_top5": float(np.mean(diffs)), "max_abs_prob_diff_top5": float(np.max(diffs)),
            "expert_overlap_mean": float(overlap.mean()),
            "expert_overlap_min_app": float(overlap.mean(0).min()),
            "expert_exact_share": float((overlap == 1).mean()),
            "expert_overlap_by_app": [round(float(x), 4) for x in overlap.mean(0)],
        })
        print(json.dumps({k: v for k, v in report[-1].items() if k != "expert_overlap_by_app"}, ensure_ascii=False), flush=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
