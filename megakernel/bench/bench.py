#!/usr/bin/env python3
"""Decode-speed benchmark: llama.cpp llama-server (open-jet launch flags) vs the megakernel.

Both engines get the identical chat-templated prompt and greedy sampling, so their outputs
can be compared token for token.

  python3 bench/bench.py llama --spec 2      # llama-server with MTP (open-jet default)
  python3 bench/bench.py llama --spec 0      # llama-server without speculation
  python3 bench/bench.py mk --draft 3        # megakernel
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
MODEL = os.path.join(ROOT, "models", "Qwen3.8-27B-Q4_K_M.gguf")
LLAMA_SERVER = os.path.join(ROOT, "llama.cpp", "build", "bin", "llama-server")
MK = os.path.join(os.path.dirname(HERE), "qwen38_mk")


def chat(prompt: str) -> str:
    return f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def prompts():
    with open(os.path.join(HERE, "prompts.txt")) as f:
        return [l.strip() for l in f if l.strip()]


def post(url, payload):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())


def run_llama(args):
    cmd = [LLAMA_SERVER, "-m", MODEL, "--host", "127.0.0.1", "--port", str(args.port), "--parallel", "1",
           "--flash-attn", "on", "-ctk", "q8_0", "-ctv", "q8_0", "-ngl", "99", "-c", str(args.ctx)]
    if args.spec > 0:
        cmd += ["--spec-type", "draft-mtp", "--spec-draft-n-max", str(args.spec)]
    log = open(os.path.join(HERE, "llama_server.log"), "w")
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
    try:
        for _ in range(600):
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{args.port}/health", timeout=2) as r:
                    if r.status == 200:
                        break
            except Exception:
                time.sleep(1)
        results = []
        for i, p in enumerate(prompts()):
            for rep in range(args.reps):
                r = post(f"http://127.0.0.1:{args.port}/completion", {
                    "prompt": chat(p), "n_predict": args.n, "temperature": 0.0, "top_k": 1,
                    "cache_prompt": False, "return_tokens": True,
                })
                t = r["timings"]
                results.append(t["predicted_per_second"])
                print(f"llama spec={args.spec} prompt{i} rep{rep}: {t['predicted_n']} tok, "
                      f"{t['predicted_per_second']:.2f} tok/s"
                      + (f", draft accepted {t.get('draft_n_accepted')}/{t.get('draft_n')}" if 'draft_n' in t else ""))
                if rep == 0 and args.dump:
                    with open(os.path.join(args.dump, f"llama_p{i}.json"), "w") as f:
                        json.dump({"tokens": r.get("tokens"), "content": r["content"]}, f)
        print(f"llama spec={args.spec}: mean {sum(results)/len(results):.2f} tok/s")
    finally:
        proc.terminate()
        proc.wait()


def run_mk(args):
    results = []
    for i, p in enumerate(prompts()):
        for rep in range(args.reps):
            cmd = [MK, "-m", MODEL, "-p", p, "-n", str(args.n), "-d", str(args.draft), "-c", str(args.ctx), "--ids", "--quiet"]
            if args.draft_vocab:
                cmd += ["--draft-vocab", args.draft_vocab]
            out = subprocess.run(cmd, capture_output=True, text=True)
            m = re.search(r"decode (\d+) tok in ([\d.]+) s = ([\d.]+) tok/s.*accepted (\d+)/(\d+)", out.stderr)
            if not m:
                print(out.stderr[-2000:])
                sys.exit(1)
            tps = float(m.group(3))
            results.append(tps)
            print(f"mk draft={args.draft} prompt{i} rep{rep}: {m.group(1)} tok, {tps:.2f} tok/s, "
                  f"accepted {m.group(4)}/{m.group(5)}")
            if rep == 0 and args.dump:
                ids = [int(x) for x in re.search(r"IDS:(.*)", out.stdout).group(1).split()]
                with open(os.path.join(args.dump, f"mk_p{i}.json"), "w") as f:
                    json.dump({"tokens": ids}, f)
    print(f"mk draft={args.draft}: mean {sum(results)/len(results):.2f} tok/s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("engine", choices=["llama", "mk"])
    ap.add_argument("--spec", type=int, default=2)
    ap.add_argument("--draft", type=int, default=3)
    ap.add_argument("-n", type=int, default=512)
    ap.add_argument("--ctx", type=int, default=8192)
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--port", type=int, default=8091)
    ap.add_argument("--dump", default="")
    ap.add_argument("--draft-vocab", default="")
    args = ap.parse_args()
    if args.dump:
        os.makedirs(args.dump, exist_ok=True)
    (run_llama if args.engine == "llama" else run_mk)(args)


if __name__ == "__main__":
    main()
