#!/usr/bin/env python3
"""Several agents on one server: per-request decode speed and waits (any OpenAI-compatible server).

  short    N concurrent coding requests with short prompts
  ctx      N concurrent requests, each with its own ~CTX-token context (this repo's source, a distinct
           prefix per request) sent once first, then again with a coding task: decode per request and
           time to first token (a server that keeps each conversation answers at once)
  stagger  one request decodes a long answer; 4 s in, another sends a ~20K-token prompt: the first
           one's speed while that prompt is processed, and the prompt's time to first token

  python3 server/mk_server.py -m model.gguf -c 0 --parallel 4 --port 18080
  llama-server -m model.gguf -ngl 99 -fa on -c 131072 --parallel 4 -ctk q8_0 -ctv q8_0 --jinja --port 18080
  python3 bench/parallel_bench.py http://127.0.0.1:18080 [--n 4] [--max 512] [--ctx-chars 100000]
"""
import argparse
import glob
import json
import os
import threading
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TASKS = ["Write a Python module implementing an LRU cache with TTL expiry, thread safety and unit tests.",
         "Write a Rust command-line tool that counts word frequencies in files, with argument parsing and error handling.",
         "Write a TypeScript Express server with JWT authentication, a users CRUD API and input validation.",
         "Write a C++ class for a thread pool with a task queue, futures and graceful shutdown, plus an example main."]


def source_text(chars):
    out = ""
    for f in sorted(glob.glob(os.path.join(ROOT, "src", "**", "*.py"), recursive=True)):
        out += open(f, encoding="utf-8", errors="replace").read()
        if len(out) >= chars:
            break
    return out[:chars]


def request(url, content, max_tokens, temperature=0.6):
    """streams one chat completion: (decoded tokens, decode seconds, seconds to first token, token times)"""
    body = {"messages": [{"role": "user", "content": content}], "max_tokens": max_tokens, "temperature": temperature,
            "top_p": 0.95, "top_k": 20, "stream": True, "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0, times, n = time.time(), [], 0
    with urllib.request.urlopen(req, timeout=1800) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data: {"):
                continue
            j = json.loads(line[6:])
            if any((c["delta"].get("content") or "") for c in j.get("choices", [])):
                times.append(time.time())
            if j.get("timings"):
                n = j["timings"]["predicted_n"]
    first = times[0] if times else t0
    return n, time.time() - first, first - t0, times


def concurrent(url, contents, max_tokens):
    res = [None] * len(contents)

    def run(i):
        res[i] = request(url, contents[i], max_tokens)
    th = [threading.Thread(target=run, args=(i,)) for i in range(len(contents))]
    for t in th:
        t.start()
    for t in th:
        t.join()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--max", type=int, default=512)
    ap.add_argument("--ctx-chars", type=int, default=100000, help="per-request context for `ctx` (~23K tokens)")
    ap.add_argument("--tests", default="short,ctx,stagger")
    a = ap.parse_args()
    stamp = time.time()
    tests = a.tests.split(",")
    if "short" in tests:
        for rep in range(2):
            res = concurrent(a.url, [TASKS[i % 4] + " Output only code." for i in range(a.n)], a.max)
            print(f"short x{a.n}: decode per request " + ", ".join(f"{r[0] / r[1]:.0f}" for r in res) + " tok/s")
    if "ctx" in tests:
        ctx = source_text(a.ctx_chars)
        prompts = [f"[session {i} {stamp}]\n{ctx}\n\n{TASKS[i % 4]} Output only code." for i in range(a.n)]
        for p in prompts:  # (each context processed once first)
            request(a.url, p, 1)
        res = concurrent(a.url, prompts, a.max)
        print(f"ctx x{a.n}: decode per request " + ", ".join(f"{r[0] / r[1]:.0f}" for r in res) +
              " tok/s; first token " + ", ".join(f"{r[2]:.1f}" for r in res) + " s")
    if "stagger" in tests:
        ctx = source_text(88000)  # ~20K tokens
        out = {}
        ta = threading.Thread(target=lambda: out.__setitem__("A", request(
            a.url, f"[{stamp}] Write a very long, detailed story about a lighthouse keeper.", 3000, 0.0)))
        ta.start()
        time.sleep(4)
        b0 = time.time()
        out["B"] = request(a.url, f"[{stamp}]\n{ctx}\n\nSummarise the code above in one paragraph.", 32, 0.0)
        b1 = b0 + out["B"][2]
        ta.join()
        during = sum(1 for t in out["A"][3] if b0 <= t <= b1)
        print(f"stagger: the long prompt's first token after {b1 - b0:.1f} s; the other request meanwhile "
              f"{during / (b1 - b0):.1f} tok/s")


if __name__ == "__main__":
    main()
