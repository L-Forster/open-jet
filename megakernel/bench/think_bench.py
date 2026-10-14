#!/usr/bin/env python3
"""Reasoning-mode benchmark under open-jet's Qwen3.8 thinking preset (temperature 1.0, top_p 0.95,
top_k 20, thinking on with the chat template's default xhigh effort): llama.cpp llama-server with
open-jet's flags vs the megakernel.  Both get the identically rendered prompt.

  python3 bench/think_bench.py [--seeds 2] [-n 1024] [--output results.json]

Optional --model, --llama-server and --megakernel override executable/model paths.
--llama-gpu and --megakernel-gpu each select one GPU UUID or numeric PCI_BUS_ID
ordinal, independently of inherited CUDA_VISIBLE_DEVICES. Without pins, the inherited
CUDA environment is preserved. JSON output retains every seed, not a best-of result.
Use --kv q8 to match llama.cpp's q8_0 KV precision; the legacy megakernel
default is auto. --context sets the same capacity for both backends (default 16384).
The llama-server port (8094) must be free; occupied ports fail before startup.
"""
import argparse
import json
import os
import re
import socket
import statistics
import subprocess
import time
import urllib.error
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
    with open(os.path.join(MKDIR, "prefill.cuh")) as f:
        src = f.read()[:8000]
    return "Here is a CUDA source file:\n\n" + src + "\n\nFind one possible bug or fragile spot in this file and suggest a fix."


def render(p):
    return f"<|im_start|>system\n{XHIGH}<|im_end|>\n<|im_start|>user\n{p}<|im_end|>\n<|im_start|>assistant\n<think>\n"


def post(url, payload):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        return json.loads(r.read())


def gpu_identifier(value):
    if not re.fullmatch(r"\d+|(?:GPU|MIG)-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}", value):
        raise argparse.ArgumentTypeError("GPU must be one numeric PCI_BUS_ID ordinal or GPU/MIG UUID")
    return value


def gpu_env(gpu):
    env = os.environ.copy()
    if gpu is not None:
        env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        env["CUDA_VISIBLE_DEVICES"] = gpu_identifier(gpu)
    return env


def positive_int(value):
    n = int(value)
    if n <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return n


def run_llama(prompts, seeds, n, model=MODEL, llama_server=LLAMA_SERVER, gpu=None, context=16384):
    port = 8094
    # Do not mistake another shared-host server's health endpoint for our own.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind(("127.0.0.1", port))
        except OSError as exc:
            raise RuntimeError(f"llama-server port {port} is unavailable; refusing to use an existing server") from exc
    cmd = [llama_server, "-m", model, "--host", "127.0.0.1", "--port", str(port), "--parallel", "1",
           "--flash-attn", "on", "-ctk", "q8_0", "-ctv", "q8_0", "-ngl", "99", "-c", str(context),
           "--spec-type", "draft-mtp", "--spec-draft-n-max", "2"]
    log = open(os.path.join(TMP, "llama_server.log"), "w")
    try:
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=gpu_env(gpu))
    finally:
        log.close()
    res = {}
    try:
        for _ in range(600):
            if proc.poll() is not None:
                raise RuntimeError(f"llama-server exited with status {proc.returncode}; see {TMP}/llama_server.log")
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as r:
                    if r.status == 200:
                        if proc.poll() is not None:
                            raise RuntimeError(f"llama-server exited with status {proc.returncode}")
                        break
            except (urllib.error.URLError, TimeoutError):
                pass
            time.sleep(1)
        else:
            raise TimeoutError(f"llama-server did not become healthy; see {TMP}/llama_server.log")
        for i, p in enumerate(prompts):
            for s in range(seeds):
                if proc.poll() is not None:
                    raise RuntimeError(f"llama-server exited with status {proc.returncode}")
                r = post(f"http://127.0.0.1:{port}/completion", dict(prompt=render(p), n_predict=n, seed=1000 + s,
                                                                      cache_prompt=False, **SAMPLING))
                if proc.poll() is not None:
                    raise RuntimeError(f"llama-server exited with status {proc.returncode}")
                t = r["timings"]
                res[(i, s)] = dict(n=t["predicted_n"], tps=t["predicted_per_second"],
                                   seconds=t["predicted_ms"] / 1000, timings=t,
                                   acc=(t.get("draft_n_accepted", 0), t.get("draft_n", 0)))
                print(f"[llama] p{i} seed{s}: {t['predicted_n']} tok, {t['predicted_per_second']:.1f} tok/s, "
                      f"drafts {t.get('draft_n_accepted')}/{t.get('draft_n')}", flush=True)
    finally:
        if proc.poll() is None:
            proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
    return res


def run_mk(prompts, seeds, n, d, model=MODEL, megakernel=MK, gpu=None, kv="auto", context=16384):
    res = {}
    for i, p in enumerate(prompts):
        f = os.path.join(TMP, f"p{i}.txt")
        with open(f, "w") as prompt_file:
            prompt_file.write(render(p))
        for s in range(seeds):
            out = subprocess.run([megakernel, "-m", model, "-f", f, "--raw", "-n", str(n), "-d", str(d), "-c", str(context),
                                  "--draft-vocab", DV, "--temp", "1.0", "--top-p", "0.95", "--top-k", "20",
                                  "--min-p", "0.0", "--seed", str(1000 + s), "--kv", kv, "--quiet"],
                                 capture_output=True, text=True, check=True, env=gpu_env(gpu)).stderr
            m = re.search(r"decode (\d+) tok in ([\d.]+) s = ([\d.]+) tok/s.*accepted (\d+)/(\d+).*?([\d.]+) tok/cycle", out)
            if not m:
                raise RuntimeError(out[-2000:])
            # The first generated token belongs to prefill, not the timed decode.
            res[(i, s)] = dict(n=int(m.group(1)), generated_n=int(m.group(1)) + 1,
                               seconds=float(m.group(2)), tps=float(m.group(3)),
                               acc=(int(m.group(4)), int(m.group(5))), tpc=float(m.group(6)), stderr=out)
            print(f"[mk d={d}] p{i} seed{s}: {m.group(1)} tok, {float(m.group(3)):.1f} tok/s, {m.group(6)} tok/cycle",
                  flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=positive_int, default=2)
    ap.add_argument("-n", type=positive_int, default=1024, help="maximum generated tokens per sample")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--llama-server", default=LLAMA_SERVER)
    ap.add_argument("--megakernel", default=MK)
    ap.add_argument("--llama-gpu", type=gpu_identifier)
    ap.add_argument("--megakernel-gpu", type=gpu_identifier)
    ap.add_argument("--kv", choices=("f16", "q8", "auto"), default="auto",
                    help="megakernel KV precision; q8 matches llama.cpp's fixed q8_0")
    ap.add_argument("--context", type=positive_int, default=16384, help="context capacity for both backends")
    ap.add_argument("--output", help="write settings, prompts and all raw timing samples as JSON")
    args = ap.parse_args()
    os.makedirs(TMP, exist_ok=True)
    prompts = PROMPTS + [file_prompt()]
    ll = run_llama(prompts, args.seeds, args.n, model=args.model, llama_server=args.llama_server,
                   gpu=args.llama_gpu, context=args.context)
    m3 = run_mk(prompts, args.seeds, args.n, 3, model=args.model, megakernel=args.megakernel,
                gpu=args.megakernel_gpu, kv=args.kv, context=args.context)
    m4 = run_mk(prompts, args.seeds, args.n, 4, model=args.model, megakernel=args.megakernel,
                gpu=args.megakernel_gpu, kv=args.kv, context=args.context)
    if args.output:
        settings = dict(vars(args), sampling=SAMPLING, seeds_used=list(range(1000, 1000 + args.seeds)),
                        prompt_count=len(prompts), llama_draft_max=2, megakernel_drafts=[3, 4], draft_vocab=DV,
                        llama_kv="q8_0", megakernel_kv=args.kv,
                        timing_scope="decode only; megakernel first token is in prefill",
                        aggregation="per-prompt arithmetic mean over all seeds; total decode tokens / total decode seconds")
        settings["cuda_environments"] = {
            name: {k: env.get(k) for k in ("CUDA_DEVICE_ORDER", "CUDA_VISIBLE_DEVICES")}
            for name, env in (("llama", gpu_env(args.llama_gpu)), ("megakernel", gpu_env(args.megakernel_gpu)))
        }
        samples = {
            name: [dict(prompt_index=i, seed=1000 + s, **v) for (i, s), v in results.items()]
            for name, results in (("llama", ll), ("megakernel_d3", m3), ("megakernel_d4", m4))
        }
        with open(args.output, "w") as output:
            json.dump(dict(settings=settings, prompts=prompts, rendered_prompts=[render(p) for p in prompts],
                           samples=samples), output, indent=2, allow_nan=False)
            output.write("\n")

    def tok_weighted(r):  # total tokens / total time
        return sum(v["n"] for v in r.values()) / sum(v["seconds"] for v in r.values())

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
