#!/usr/bin/env python3
"""Code-workload benchmark: llama.cpp llama-server (open-jet flags) vs the megakernel.

  1. decode on the 4 chat prompts of prompts.txt (llama spec 2, megakernel 3 and 4 drafts)
  2. context sweep: a source file of ~N tokens pasted into the prompt plus a coding task;
     prompt-processing and decode speed at each N
  3. second turn: a pasted file appended to a running conversation (llama with prompt cache,
     megakernel batched append, megakernel through the decode kernel)

  python3 bench/code_bench.py [--ctx-sizes 0,2000,4000,8000,16000,30000] [--reps 2]
"""
import argparse
import json
import os
import re
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
TMP = os.path.join(HERE, "code_bench_tmp")
CTX = 34000
CHARS_PER_TOK = 2.6  # this source code tokenises at ~2.6 characters per token

TASK = "Write a C++ unit test (plain asserts, no framework) for one self-contained helper function from the file above."
TURN1 = "Write a C function that reverses a string in place."
TURN2 = "Here is a source file I am working on:\n\n{code}\n\nWhat does the first kernel in this file do? Answer briefly."


def chat(p):
    return f"<|im_start|>user\n{p}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def code_text(n_tok):
    src = ""
    for f in ["qwen38_mk.cu", "prefill.cuh", "prefill_gemm.cuh"]:
        src += open(os.path.join(MKDIR, f)).read() + "\n"
    return src[: int(n_tok * CHARS_PER_TOK)]


def ctx_prompt(n_tok):
    if n_tok == 0:
        return "Write a C++ function that parses a GGUF file header (magic, version, tensor count) from a byte buffer, with error checks."
    return "Here is a source file:\n\n" + code_text(n_tok) + "\n\n" + TASK


def post(url, payload):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        return json.loads(r.read())


class Llama:
    def __init__(self, port=8093):
        self.url = f"http://127.0.0.1:{port}"
        cmd = [LLAMA_SERVER, "-m", MODEL, "--host", "127.0.0.1", "--port", str(port), "--parallel", "1",
               "--flash-attn", "on", "-ctk", "q8_0", "-ctv", "q8_0", "-ngl", "99", "-c", str(CTX),
               "--spec-type", "draft-mtp", "--spec-draft-n-max", "2"]
        self.log = open(os.path.join(TMP, "llama_server.log"), "w")
        self.proc = subprocess.Popen(cmd, stdout=self.log, stderr=subprocess.STDOUT)
        for _ in range(600):
            try:
                with urllib.request.urlopen(self.url + "/health", timeout=2) as r:
                    if r.status == 200:
                        return
            except Exception:
                time.sleep(1)
        raise RuntimeError("llama-server did not start")

    def complete(self, prompt, n, cache=False):
        r = post(self.url + "/completion", {"prompt": prompt, "n_predict": n, "temperature": 0.0, "top_k": 1,
                                            "cache_prompt": cache})
        return r["content"], r["timings"]

    def close(self):
        self.proc.terminate()
        self.proc.wait()


def mk_run(args, env=None):
    out = subprocess.run([MK, "-m", MODEL, "-c", str(CTX), "--draft-vocab", DV, "--quiet"] + args,
                         capture_output=True, text=True, env=dict(os.environ, **(env or {})))
    turns = re.findall(r"prompt (\d+) tok in ([\d.]+) s \(([\d.]+) tok/s\)\n.*decode (\d+) tok in ([\d.]+) s = ([\d.]+) tok/s"
                       r".*?, ([\d.]+) tok/cycle", out.stderr)
    if not turns:
        raise RuntimeError(out.stderr[-2000:])
    return [dict(pn=int(t[0]), ps=float(t[1]), ptps=float(t[2]), dn=int(t[3]), dtps=float(t[5]), tpc=float(t[6]))
            for t in turns]


def best(runs, key):
    return max(runs, key=lambda r: r[key])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx-sizes", default="0,2000,4000,8000,16000,30000")
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("-n", type=int, default=256)
    args = ap.parse_args()
    os.makedirs(TMP, exist_ok=True)
    sizes = [int(x) for x in args.ctx_sizes.split(",")]
    prompts = [l.strip() for l in open(os.path.join(HERE, "prompts.txt")) if l.strip()]
    files = {}
    for n in sizes:
        files[n] = os.path.join(TMP, f"ctx_{n}.txt")
        open(files[n], "w").write(ctx_prompt(n))
    t2files = {}
    for n in [3000, 12000]:
        t2files[n] = os.path.join(TMP, f"turn2_{n}.txt")
        open(t2files[n], "w").write(TURN2.format(code=code_text(n)))

    res = {}
    # ---------------- llama.cpp ----------------
    ll = Llama()
    try:
        for i, p in enumerate(prompts):
            runs = []
            for _ in range(args.reps):
                _, t = ll.complete(chat(p), 512)
                runs.append(t["predicted_per_second"])
            res[("short", "llama", i)] = max(runs)
            print(f"[llama] prompt{i}: {max(runs):.1f} tok/s", flush=True)
        for n in sizes:
            runs = []
            for _ in range(args.reps):
                _, t = ll.complete(chat(open(files[n]).read()), args.n)
                runs.append(dict(pn=t["prompt_n"], ptps=t["prompt_per_second"], dtps=t["predicted_per_second"]))
            r = dict(pn=runs[0]["pn"], ptps=max(x["ptps"] for x in runs), dtps=max(x["dtps"] for x in runs))
            res[("ctx", "llama", n)] = r
            print(f"[llama] ctx {r['pn']} tok: prompt {r['ptps']:.0f} tok/s, decode {r['dtps']:.1f} tok/s", flush=True)
        for n, f in t2files.items():
            best_ms = None
            for _ in range(args.reps):
                p1 = chat(TURN1)
                a1, _ = ll.complete(p1, 128, cache=True)
                conv = p1 + a1 + "<|im_end|>\n" + chat(open(f).read())
                _, t = ll.complete(conv, 64, cache=True)
                if best_ms is None or t["prompt_ms"] < best_ms[1]:
                    best_ms = (t["prompt_n"], t["prompt_ms"])
            res[("turn2", "llama", n)] = dict(pn=best_ms[0], s=best_ms[1] / 1000)
            print(f"[llama] turn2 ~{n}: {best_ms[0]} new tok in {best_ms[1] / 1000:.2f} s", flush=True)
    finally:
        ll.close()

    # ---------------- megakernel ----------------
    for d in (3, 4):
        for i, p in enumerate(prompts):
            runs = [mk_run(["-p", p, "-n", "512", "-d", str(d)])[0] for _ in range(args.reps)]
            res[("short", f"mk{d}", i)] = best(runs, "dtps")["dtps"]
            print(f"[mk d={d}] prompt{i}: {res[('short', f'mk{d}', i)]:.1f} tok/s", flush=True)
        for n in sizes:
            runs = [mk_run(["-f", files[n], "-n", str(args.n), "-d", str(d)])[0] for _ in range(args.reps)]
            r = dict(pn=runs[0]["pn"], ptps=best(runs, "ptps")["ptps"], dtps=best(runs, "dtps")["dtps"],
                     tpc=best(runs, "dtps")["tpc"])
            res[("ctx", f"mk{d}", n)] = r
            print(f"[mk d={d}] ctx {r['pn']} tok: prompt {r['ptps']:.0f} tok/s, decode {r['dtps']:.1f} tok/s "
                  f"({r['tpc']:.2f} tok/cycle)", flush=True)
    for n, f in t2files.items():
        for lab, env, reps in (("mk", None, args.reps), ("mk_old", {"MK_PREFILL": "mk"}, 1)):
            runs = [mk_run(["-p", TURN1, "-n", "128", "-d", "4", "--turn2-file", f], env)[1] for _ in range(reps)]
            r = min(runs, key=lambda x: x["ps"])
            res[("turn2", lab, n)] = dict(pn=r["pn"], s=r["ps"])
            print(f"[{lab}] turn2 ~{n}: {r['pn']} new tok in {r['ps']:.2f} s", flush=True)

    # ---------------- tables ----------------
    print("\n## Decode, short chat prompts (tok/s, best of %d)\n" % args.reps)
    print("| prompt | llama.cpp (MTP 2) | megakernel -d 3 | megakernel -d 4 |\n|---|---|---|---|")
    for i, p in enumerate(prompts):
        print(f"| {p[:40]}... | {res[('short', 'llama', i)]:.1f} | {res[('short', 'mk3', i)]:.1f} | {res[('short', 'mk4', i)]:.1f} |")
    m = {k: sum(res[('short', k, i)] for i in range(len(prompts))) / len(prompts) for k in ("llama", "mk3", "mk4")}
    print(f"| **mean** | **{m['llama']:.1f}** | **{m['mk3']:.1f}** | **{m['mk4']:.1f}** |")
    print("\n## Code in context: prompt processing and decode (%d generated tokens)\n" % args.n)
    print("| context (tok) | llama prompt tok/s | mk prompt tok/s | llama decode | mk -d 3 decode | mk -d 4 decode |\n|---|---|---|---|---|---|")
    for n in sizes:
        a, b, c = res[("ctx", "llama", n)], res[("ctx", "mk3", n)], res[("ctx", "mk4", n)]
        print(f"| {b['pn']} | {a['ptps']:.0f} | {max(b['ptps'], c['ptps']):.0f} | {a['dtps']:.1f} | {b['dtps']:.1f} | {c['dtps']:.1f} |")
    print("\n## Second turn: pasted file appended to a conversation (prompt time)\n")
    print("| new tokens | llama.cpp (prompt cache) | megakernel (batched append) | megakernel (decode kernel) |\n|---|---|---|---|")
    for n in t2files:
        a, b, c = res[("turn2", "llama", n)], res[("turn2", "mk", n)], res[("turn2", "mk_old", n)]
        print(f"| {b['pn']} (llama: {a['pn']}) | {a['s']:.2f} s | {b['s']:.2f} s | {c['s']:.2f} s |")


if __name__ == "__main__":
    main()
