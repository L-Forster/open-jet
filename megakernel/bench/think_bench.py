#!/usr/bin/env python3
"""Reasoning-mode benchmark under open-jet's Qwen3.8 thinking preset (temperature 1.0, top_p 0.95,
top_k 20, thinking on with the chat template's default xhigh effort): llama.cpp llama-server with
open-jet's flags vs the megakernel.  Both get the identically rendered prompt.

  python3 bench/think_bench.py [--seeds 2] [-n 1024]
"""
import argparse
import json
import os
import re
import statistics
import subprocess
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
MKDIR = os.path.dirname(HERE)
ROOT = os.path.dirname(MKDIR)
MODEL = os.path.join(ROOT, "models", "Qwen3.8-27B-Q4_K_M.gguf")
LLAMA_SERVER = os.path.join(ROOT, "llama.cpp", "build", "bin", "llama-server")
MK = os.path.join(MKDIR, "qwen38_mk")
DV = os.path.join(MKDIR, "draft_vocab_32k.bin")
TMP = os.path.join(HERE, "think_bench_tmp")
SAMPLING = dict(temperature=1.0, top_p=0.95, top_k=20, min_p=0.0)
XHIGH = ("Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, "
         "consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.")

PROMPTS = [
    "Write a Python function that merges overlapping intervals in a list of [start, end] pairs, with a few pytest tests.",
    "Write a bash script that finds the 10 largest files under a directory (argument), skipping .git, and prints sizes in human-readable form.",
    "In C, implement a fixed-capacity ring buffer of ints with push, pop and is_full, safe for one producer and one consumer thread.",
    "Explain the difference between a mutex and a semaphore, and when to use each, with a short Python example of each.",
]


def file_prompt():
    src = open(os.path.join(MKDIR, "prefill.cuh")).read()[:8000]
    return "Here is a CUDA source file:\n\n" + src + "\n\nFind one possible bug or fragile spot in this file and suggest a fix."


def render(p):
    return f"<|im_start|>system\n{XHIGH}<|im_end|>\n<|im_start|>user\n{p}<|im_end|>\n<|im_start|>assistant\n<think>\n"


def post(url, payload):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        return json.loads(r.read())


def run_llama(prompts, seeds, n):
    port = 8094
    cmd = [LLAMA_SERVER, "-m", MODEL, "--host", "127.0.0.1", "--port", str(port), "--parallel", "1",
           "--flash-attn", "on", "-ctk", "q8_0", "-ctv", "q8_0", "-ngl", "99", "-c", "16384",
           "--spec-type", "draft-mtp", "--spec-draft-n-max", "2"]
    proc = subprocess.Popen(cmd, stdout=open(os.path.join(TMP, "llama_server.log"), "w"), stderr=subprocess.STDOUT)
    res = {}
    try:
        for _ in range(600):
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as r:
                    if r.status == 200:
                        break
            except Exception:
                time.sleep(1)
        for i, p in enumerate(prompts):
            for s in range(seeds):
                r = post(f"http://127.0.0.1:{port}/completion", dict(prompt=render(p), n_predict=n, seed=1000 + s,
                                                                      cache_prompt=False, **SAMPLING))
                t = r["timings"]
                res[(i, s)] = dict(n=t["predicted_n"], tps=t["predicted_per_second"],
                                   acc=(t.get("draft_n_accepted", 0), t.get("draft_n", 0)))
                print(f"[llama] p{i} seed{s}: {t['predicted_n']} tok, {t['predicted_per_second']:.1f} tok/s, "
                      f"drafts {t.get('draft_n_accepted')}/{t.get('draft_n')}", flush=True)
    finally:
        proc.terminate()
        proc.wait()
    return res


def run_mk(prompts, seeds, n, d):
    res = {}
    for i, p in enumerate(prompts):
        f = os.path.join(TMP, f"p{i}.txt")
        open(f, "w").write(render(p))
        for s in range(seeds):
            out = subprocess.run([MK, "-m", MODEL, "-f", f, "--raw", "-n", str(n), "-d", str(d), "-c", "16384",
                                  "--draft-vocab", DV, "--temp", "1.0", "--top-p", "0.95", "--top-k", "20",
                                  "--seed", str(1000 + s), "--quiet"], capture_output=True, text=True).stderr
            m = re.search(r"decode (\d+) tok in [\d.]+ s = ([\d.]+) tok/s.*accepted (\d+)/(\d+).*?([\d.]+) tok/cycle", out)
            if not m:
                raise RuntimeError(out[-2000:])
            res[(i, s)] = dict(n=int(m.group(1)) + 1, tps=float(m.group(2)), acc=(int(m.group(3)), int(m.group(4))),
                               tpc=float(m.group(5)))
            print(f"[mk d={d}] p{i} seed{s}: {m.group(1)} tok, {float(m.group(2)):.1f} tok/s, {m.group(5)} tok/cycle",
                  flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("-n", type=int, default=1024)
    args = ap.parse_args()
    os.makedirs(TMP, exist_ok=True)
    prompts = PROMPTS + [file_prompt()]
    ll = run_llama(prompts, args.seeds, args.n)
    m3 = run_mk(prompts, args.seeds, args.n, 3)
    m4 = run_mk(prompts, args.seeds, args.n, 4)

    def tok_weighted(r):  # total tokens / total time
        return sum(v["n"] for v in r.values()) / sum(v["n"] / v["tps"] for v in r.values())

    print("\n## Thinking mode, open-jet sampling preset (decode tok/s, mean over seeds)\n")
    print("| prompt | llama.cpp (MTP 2) | megakernel -d 3 | megakernel -d 4 |\n|---|---|---|---|")
    names = ["merge intervals (py)", "largest files (bash)", "ring buffer (C)", "mutex vs semaphore", "bug in pasted file (~3K tok)"]
    for i, nm in enumerate(names):
        row = [statistics.mean(r[(i, s)]["tps"] for s in range(args.seeds)) for r in (ll, m3, m4)]
        print(f"| {nm} | {row[0]:.1f} | {row[1]:.1f} | {row[2]:.1f} |")
    t = [tok_weighted(r) for r in (ll, m3, m4)]
    print(f"| **all (tokens / time)** | **{t[0]:.1f}** | **{t[1]:.1f}** ({t[1] / t[0]:.2f}x) | **{t[2]:.1f}** ({t[2] / t[0]:.2f}x) |")
    acc = lambda r: sum(v["acc"][0] for v in r.values()) / max(1, sum(v["acc"][1] for v in r.values()))
    print(f"\ndraft acceptance: llama {acc(ll):.1%}, mk -d 3 {acc(m3):.1%}, mk -d 4 {acc(m4):.1%}")


if __name__ == "__main__":
    main()
