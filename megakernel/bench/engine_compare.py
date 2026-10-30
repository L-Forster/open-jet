#!/usr/bin/env python3
"""Same-machine engine comparison through each engine's OpenAI-compatible HTTP server.

Every engine gets the same GGUF, the same chat requests (thinking on, reasoning_effort xhigh,
open-jet's thinking sampling: temperature 1.0, top_p 0.95, top_k 20, min_p 0), the same seeds
and the same context size. Two suites:

  short  think_bench.py's five prompts (four coding questions and an 8K-character file)
  long   about 100K tokens of this repository's source followed by one question

Timing is measured on the client from the streamed response, the same way for every engine:
time to first token (TTFT) covers the prompt read, decode = (completion tokens - 1) / (time of
last chunk - time of first chunk). Each server's own timings are kept too. The long suite's
first request per server is the cold prompt read; later seeds may reuse the cached prompt, so
only the first counts for TTFT.

  python3 bench/engine_compare.py --engine NAME=PORT=CMD... [--suite short,long] [--seeds 2]
      [--rounds 1] [-n 1024] [--tokenizer tokenizer.json] --output results.json

Each --engine is NAME=PORT=COMMAND (the command is split with shlex and must start a server on
PORT). Engines run one after another; each server is stopped before the next starts. With
--rounds R the seeds are split into R slices and every round runs each engine on one slice, so
background GPU load and clock drift fall on all engines alike instead of on one block.

Before each request the GPU's utilisation, SM clock and power are sampled (nvidia-smi) and kept
with the sample; the load before each server starts is printed. Every file named in an engine's
command is fingerprinted (size, mtime, sha256) so the results record which build ran.
"""
import argparse
import hashlib
import json
import os
import shlex
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
MKDIR = os.path.dirname(HERE)
ROOT = os.path.dirname(MKDIR)
sys.path.insert(0, HERE)
from think_bench import PROMPTS, file_prompt  # noqa: E402

SAMPLING = dict(temperature=1.0, top_p=0.95, top_k=20, min_p=0.0)
TEMPLATE_KWARGS = dict(enable_thinking=True, reasoning_effort="xhigh")
LONG_SOURCES = ["megakernel/qwen38_mk.cu", "megakernel/prefill.cuh", "megakernel/prefill_gemm.cuh",
                "megakernel/qwen38_mixed.cu", "megakernel/main.cu", "megakernel/capi.cpp",
                "megakernel/server/mk_server.py", "megakernel/README.md", "src/llama_server.py",
                "src/hybrid.py", "src/runtime_protocol.py", "README.md"]
LONG_QUESTION = ("\n\nAbove is source code from one project. Explain how the decode path verifies drafted tokens, "
                 "naming the functions involved, and point out one place where a bug could hide.")


def long_prompt(target_tokens, tokenizer_path):
    text = ""
    for rel in LONG_SOURCES:
        with open(os.path.join(ROOT, rel), errors="replace") as f:
            text += f"\n\n===== {rel} =====\n\n" + f.read()
    if tokenizer_path:
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(tokenizer_path)
        ids = tok.encode(text, add_special_tokens=False).ids
        if len(ids) < target_tokens:
            raise RuntimeError(f"sources give {len(ids)} tokens, fewer than {target_tokens}")
        text = tok.decode(ids[:target_tokens])
    else:
        text = text[:target_tokens * 3]
    return "Here is a codebase:\n" + text + LONG_QUESTION


def stream_chat(port, prompt, n, seed):
    body = dict(model="bench", messages=[{"role": "user", "content": prompt}], max_tokens=n, seed=seed,
                stream=True, stream_options={"include_usage": True}, chat_template_kwargs=TEMPLATE_KWARGS,
                cache_prompt=True, **SAMPLING)
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    t_first = t_last = None
    usage = timings = None
    chars = 0
    with urllib.request.urlopen(req, timeout=3600) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            obj = json.loads(data)
            usage = obj.get("usage") or usage
            timings = obj.get("timings") or timings
            for ch in obj.get("choices") or []:
                d = ch.get("delta") or {}
                piece = (d.get("content") or "") + (d.get("reasoning_content") or "")
                if piece:
                    now = time.time()
                    t_first = t_first or now
                    t_last = now
                    chars += len(piece)
    if not usage or t_first is None:
        raise RuntimeError(f"no usage or no tokens in the response (usage={usage})")
    n_out = usage["completion_tokens"]
    return dict(prompt_tokens=usage["prompt_tokens"], completion_tokens=n_out,
                cached_tokens=(usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
                ttft_s=t_first - t0, decode_s=t_last - t_first,
                decode_tps=(n_out - 1) / max(t_last - t_first, 1e-9), output_chars=chars, server_timings=timings)


def gpu_state():
    out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,clocks.sm,power.draw,memory.used",
                          "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout.split(",")
    try:
        return dict(util=float(out[0]), sm_mhz=float(out[1]), power_w=float(out[2]), mem_mib=float(out[3]))
    except (IndexError, ValueError):
        return None


def fingerprint(cmd):
    files = {}
    for arg in cmd:
        if os.path.isfile(arg):
            st = os.stat(arg)
            h = hashlib.sha256()
            with open(arg, "rb") as f:  # the GGUF is hashed by its first 64 MiB only
                h.update(f.read(64 << 20))
            files[arg] = dict(size=st.st_size, mtime=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(st.st_mtime)),
                              sha256_head=h.hexdigest())
    return files


def wait_ready(proc, port, log_path):
    for _ in range(1200):
        if proc.poll() is not None:
            raise RuntimeError(f"server exited with status {proc.returncode}; see {log_path}")
        for path in ("/health", "/v1/models"):
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=2) as r:
                    if r.status == 200:
                        return
            except (urllib.error.URLError, TimeoutError, ConnectionError):
                pass
        time.sleep(1)
    raise TimeoutError(f"server did not become ready; see {log_path}")


def port_free(port, wait=60):
    # A server stopped a moment ago can hold the port for a few seconds; a live one never lets go.
    for _ in range(wait):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind(("127.0.0.1", port))
                return True
            except OSError:
                pass
        time.sleep(1)
    return False


def run_engine(name, port, cmd, suites, seeds, n, logdir, save, rnd=0):
    if not port_free(port):
        raise RuntimeError(f"port {port} is in use; refusing to benchmark an existing server")
    idle = gpu_state()
    print(f"[{name}] round {rnd}: GPU before start {idle}", flush=True)
    log_path = os.path.join(logdir, f"{name}.r{rnd}.log")
    log = open(log_path, "w")
    t_start = time.time()
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
    log.close()
    samples = []
    try:
        wait_ready(proc, port, log_path)
        print(f"[{name}] ready after {time.time() - t_start:.0f} s", flush=True)
        for suite, prompts in suites:
            for i, p in enumerate(prompts):
                for s in seeds:
                    g = gpu_state()
                    r = stream_chat(port, p, n, s)
                    r.update(engine=name, suite=suite, prompt_index=i, seed=s, round=rnd, gpu_before=g)
                    samples.append(r)
                    save(samples)
                    print(f"[{name}] {suite} p{i} seed{s}: prompt {r['prompt_tokens']} "
                          f"(cached {r['cached_tokens']}), TTFT {r['ttft_s']:.2f} s, "
                          f"{r['completion_tokens']} tok at {r['decode_tps']:.1f} tok/s", flush=True)
    finally:
        if proc.poll() is None:
            proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
    return samples


def summarize(samples):
    out = {}
    for e in dict.fromkeys(x["engine"] for x in samples):
        for suite in ("short", "long"):
            xs = [x for x in samples if x["engine"] == e and x["suite"] == suite]
            if not xs:
                continue
            tok = sum(x["completion_tokens"] - 1 for x in xs)
            sec = sum(x["decode_s"] for x in xs)
            row = dict(samples=len(xs), decode_tps=tok / sec, prompt_tokens=sorted({x["prompt_tokens"] for x in xs}))
            cold = [x for x in xs if (x["cached_tokens"] or 0) < x["prompt_tokens"] // 2]
            if suite == "long" and cold:  # requests that read the whole prompt (a new server, or no reuse)
                ttft = sum(x["ttft_s"] for x in cold) / len(cold)
                row.update(cold_ttft_s=ttft, cold_prompt_tps=cold[0]["prompt_tokens"] / ttft, cold_samples=len(cold))
            out[f"{e}/{suite}"] = row
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", action="append", required=True, help="NAME=PORT=COMMAND")
    ap.add_argument("--suite", default="short,long")
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--rounds", type=int, default=1, help="split the seeds into this many interleaved rounds")
    ap.add_argument("--from-round", type=int, default=0, help="resume: run only rounds from this one on (their seeds)")
    ap.add_argument("-n", type=int, default=1024)
    ap.add_argument("--long-tokens", type=int, default=98000)
    ap.add_argument("--tokenizer", help="Qwen3.8 tokenizer.json, to cut the long prompt to an exact token count")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    logdir = os.path.splitext(args.output)[0] + "_logs"
    os.makedirs(logdir, exist_ok=True)
    wanted = args.suite.split(",")
    suites = []
    if "short" in wanted:
        suites.append(("short", PROMPTS + [file_prompt()]))
    if "long" in wanted:
        lp = long_prompt(args.long_tokens, args.tokenizer)
        with open(os.path.join(logdir, "long_prompt.txt"), "w") as f:
            f.write(lp)
        suites.append(("long", [lp]))
    meta = dict(settings=dict(vars(args), sampling=SAMPLING, chat_template_kwargs=TEMPLATE_KWARGS,
                              seeds_used=list(range(1000, 1000 + args.seeds))),
                prompt_sha256={name: [hashlib.sha256(p.encode()).hexdigest() for p in ps] for name, ps in suites},
                gpu=subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,power.limit,clocks.max.sm",
                                    "--format=csv,noheader"], capture_output=True, text=True).stdout.strip(),
                engines={})
    all_samples = []

    def save(engine_samples):
        with open(args.output, "w") as f:
            json.dump(dict(meta, samples=all_samples + engine_samples,
                           summary=summarize(all_samples + engine_samples)), f, indent=1)

    if args.seeds % args.rounds:
        ap.error("--seeds must be a multiple of --rounds")
    engines = []
    for spec in args.engine:
        name, port, cmd = spec.split("=", 2)
        argv = shlex.split(cmd)
        meta["engines"][name] = dict(port=int(port), command=cmd, files=fingerprint(argv))
        engines.append((name, int(port), argv))
    per = args.seeds // args.rounds
    for rnd in range(args.from_round, args.rounds):
        seeds = list(range(1000 + rnd * per, 1000 + (rnd + 1) * per))
        for name, port, argv in engines:
            all_samples += run_engine(name, port, argv, suites, seeds, args.n, logdir, save, rnd)
            save([])
    for k, v in summarize(all_samples).items():
        extra = f", cold TTFT {v['cold_ttft_s']:.1f} s ({v['cold_prompt_tps']:.0f} tok/s)" if "cold_ttft_s" in v else ""
        print(f"{k}: {v['decode_tps']:.1f} tok/s decode over {v['samples']} samples, prompt {v['prompt_tokens']}{extra}")


if __name__ == "__main__":
    main()
