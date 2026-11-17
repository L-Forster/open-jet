"""Parallel-agents client: several requests at once on one server, as open-jet's bench/parallel_bench.py measures them.
usage: python -I bench/suite/parallel_client.py --label L --out parallel.jsonl [--url URL] [--n 4] [--max 512]
                                                [--tests short,ctx,stagger] [--data DIR] [--gpu N|none]
  short    N concurrent coding requests with short prompts, twice
  ctx      N concurrent requests, each with its own ~23k-token context (oj100k.txt behind a header that differs per
           request and per run) sent once first (1 token), then again with a coding task: decode per request and time to
           first token (a server that keeps each conversation answers at once)
  stagger  one request decodes a long answer (up to 3,000 tokens, greedy); 4 s in, another sends a ~20k-token prompt
           (oj100k.txt's first 88,000 characters, greedy, 32 tokens): the first request's speed while that prompt is
           processed, and the prompt's time to first token
A port of open-jet f246ff0's megakernel/bench/parallel_bench.py (AGPL-3.0, like this repository): the same prompts,
request parameters (temperature 0.6, top_p 0.95, top_k 20, thinking off, streamed; greedy for stagger), context text
(its src/**/*.py, rebuilt by fetch.py) and measurements: a request's tokens are the stream's timings.predicted_n, its
decode time runs from the first content chunk to the end of the stream; the stagger's "other request" rate counts its
content chunks inside the window (one per token on servers that stream token by token). Totals are the sums of the
per-request rates.
One JSON line per request and per test in --out; summarize.py prints the parallel table from them.
Servers: this engine's `server/mk_server.py --agents N`, open-jet's `mk_server.py --agents N`, or llama-server
`--parallel N` (needs the stream's `timings`)."""
import argparse
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

TASKS = ["Write a Python module implementing an LRU cache with TTL expiry, thread safety and unit tests.",
         "Write a Rust command-line tool that counts word frequencies in files, with argument parsing and error handling.",
         "Write a TypeScript Express server with JWT authentication, a users CRUD API and input validation.",
         "Write a C++ class for a thread pool with a task queue, futures and graceful shutdown, plus an example main."]
STAGGER_CHARS = 88000  # open-jet's stagger context (~20k tokens); ctx sends all 100,000 characters (~23k tokens)


def request(url, content, max_tokens, temperature=0.6):
    """Streams one chat completion: (decoded tokens, decode seconds, seconds to first token, content-chunk times)."""
    body = {"messages": [{"role": "user", "content": content}], "max_tokens": max_tokens, "temperature": temperature,
            "top_p": 0.95, "top_k": 20, "stream": True, "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0, times, n = time.time(), [], None
    with urllib.request.urlopen(req, timeout=1800) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data: {"):
                continue
            j = json.loads(line[6:])
            if any((c.get("delta", {}).get("content") or "") for c in j.get("choices", [])):
                times.append(time.time())
            if j.get("timings"):
                n = j["timings"].get("predicted_n")
    if n is None:
        raise RuntimeError("the stream carried no timings.predicted_n (the token count this client reports)")
    first = times[0] if times else t0
    return n, time.time() - first, first - t0, times


def concurrent(url, contents, max_tokens):
    res, errs = [None] * len(contents), []

    def run(i):
        try:
            res[i] = request(url, contents[i], max_tokens)
        except (urllib.error.URLError, OSError, RuntimeError, ValueError) as e:
            errs.append(f"request {i}: {e}")
    th = [threading.Thread(target=run, args=(i,)) for i in range(len(contents))]
    for t in th:
        t.start()
    for t in th:
        t.join()
    if errs:
        raise SystemExit("; ".join(errs))
    return res


class Out:
    def __init__(self, a):
        self.label, self.n, self.gpu = a.label, a.n, C.Gpu(a.gpu)
        self.f = open(a.out, "a", encoding="utf-8")

    def write(self, rec):
        self.f.write(json.dumps({"label": self.label, "n": self.n, **rec}) + "\n")
        self.f.flush()

    def per_request(self, test, rep, res):
        for i, (tok, dec, ttft, _) in enumerate(res):
            self.write({"test": test, "rep": rep, "i": i, "tokens": tok, "decode_s": round(dec, 3), "ttft_s": round(ttft, 3),
                        "tps": round(tok / dec, 2) if dec > 0 else None})
        tps = [tok / dec for tok, dec, _, _ in res if dec > 0]
        self.write({"test": f"{test}_total", "rep": rep, "tps": round(sum(tps), 1),
                    "ttft_max_s": round(max(r[2] for r in res), 3)})

    def gpu_state(self, tag):
        v = self.gpu.query("temperature.gpu,clocks.sm,clocks.mem,memory.used")
        if v is not None:
            self.write({"test": f"gpu_{tag}", "temp_c": v[0], "sm_mhz": v[1], "mem_mhz": v[2], "mib": v[3]})


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    C.add_common_args(ap)
    ap.add_argument("--n", type=int, default=4, help="concurrent requests (the server's agents / parallel slots)")
    ap.add_argument("--max", type=int, default=512, help="max_tokens of the short and ctx requests")
    ap.add_argument("--tests", default="short,ctx,stagger")
    a = ap.parse_args()
    tests = a.tests.split(",")
    bad = [t for t in tests if t not in ("short", "ctx", "stagger")]
    if bad:
        raise SystemExit(f"unknown test(s) {bad}: short, ctx, stagger")
    ctx = C.load_prompt(a.data, "oj100k") if "ctx" in tests or "stagger" in tests else None
    url, out, stamp = C.chat_url(a.url), Out(a), time.time()
    out.gpu_state("start")
    if "short" in tests:
        for rep in range(2):
            res = concurrent(url, [TASKS[i % 4] + " Output only code." for i in range(a.n)], a.max)
            out.per_request("short", rep, res)
            print(f"{a.label}: short x{a.n}: decode per request " + ", ".join(f"{r[0] / r[1]:.0f}" for r in res) +
                  f" tok/s (total {sum(r[0] / r[1] for r in res):.0f})", flush=True)
    if "ctx" in tests:
        prompts = [f"[session {i} {stamp}]\n{ctx}\n\n{TASKS[i % 4]} Output only code." for i in range(a.n)]
        for p in prompts:  # (each context processed once first)
            request(url, p, 1)
        res = concurrent(url, prompts, a.max)
        out.per_request("ctx", 0, res)
        out.gpu_state("after_ctx")
        print(f"{a.label}: ctx x{a.n}: decode per request " + ", ".join(f"{r[0] / r[1]:.0f}" for r in res) +
              f" tok/s (total {sum(r[0] / r[1] for r in res):.0f}); first token " + ", ".join(f"{r[2]:.1f}" for r in res) +
              " s", flush=True)
    if "stagger" in tests:
        got = {}

        def long_answer():
            try:
                got["A"] = request(url, f"[{stamp}] Write a very long, detailed story about a lighthouse keeper.", 3000, 0.0)
            except (urllib.error.URLError, OSError, RuntimeError, ValueError) as e:
                got["A_err"] = e
        ta = threading.Thread(target=long_answer)
        ta.start()
        time.sleep(4)
        b0 = time.time()
        got["B"] = request(url, f"[{stamp}]\n{ctx[:STAGGER_CHARS]}\n\nSummarise the code above in one paragraph.", 32, 0.0)
        b1 = b0 + got["B"][2]
        ta.join()
        if "A_err" in got:
            raise SystemExit(f"stagger: the long answer failed: {got['A_err']}")
        during = sum(1 for t in got["A"][3] if b0 <= t <= b1)
        out.write({"test": "stagger", "ttft_s": round(b1 - b0, 3), "other_tps": round(during / (b1 - b0), 2)})
        print(f"{a.label}: stagger: the long prompt's first token after {b1 - b0:.1f} s; the other request meanwhile "
              f"{during / (b1 - b0):.1f} tok/s", flush=True)
    out.gpu_state("end")
    print("CLIENT_DONE", flush=True)


if __name__ == "__main__":
    main()
