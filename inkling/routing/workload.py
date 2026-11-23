"""Drive a traced llama-server with OpenJet-style requests and record where each request's
routing records sit in the trace file.

The server must run with LLAMA_ROUTE_TRACE=<trace> (see llama.cpp patch in ../README.md).
Each request is sent once, sequentially; after it finishes the trace file size is final for it,
so the byte range [before, after) is that request's routing.

    python inkling/routing/workload.py --trace /path/trace.bin --out inkling/routing/results/requests.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def read(rel: str, start: int = 0, lines: int | None = None) -> str:
    text = (REPO / rel).read_text(encoding="utf-8", errors="replace").splitlines()
    end = None if lines is None else start + lines
    return "\n".join(text[start:end])


AGENT_SYSTEM = (
    "You are a coding agent working in the user's repository. You can read files, run shell "
    "commands and edit code. Think before acting, keep changes minimal, and explain what you changed."
)


def build_requests() -> list[dict]:
    """Each item: name, messages, max_tokens. Mix of coding-agent work (OpenJet's main use) and general chat."""
    reqs: list[dict] = []

    reqs.append({
        "name": "review_connect",
        "messages": [
            {"role": "system", "content": AGENT_SYSTEM},
            {"role": "user", "content": "Review this file for bugs and propose a patch as a unified diff.\n\n```python\n"
             + read("src/connect.py") + "\n```"},
        ],
        "max_tokens": 1536,
    })
    reqs.append({
        "name": "explain_llama_server",
        "messages": [
            {"role": "system", "content": AGENT_SYSTEM},
            {"role": "user", "content": "Explain how this module decides which model server to start, and list the "
             "failure modes it handles.\n\n```python\n" + read("src/llama_server.py", 0, 420) + "\n```"},
        ],
        "max_tokens": 1536,
    })
    reqs.append({
        "name": "write_function_tests",
        "messages": [
            {"role": "user", "content": "Write a Python function that parses a byte-range HTTP header "
             "(`Range: bytes=...`) including multiple ranges and suffix ranges, with full pytest tests. "
             "Handle invalid input by raising ValueError."},
        ],
        "max_tokens": 1536,
    })
    reqs.append({
        "name": "debug_traceback",
        "messages": [
            {"role": "system", "content": AGENT_SYSTEM},
            {"role": "user", "content": "My test run fails with this. What's wrong and how do I fix it?\n\n"
             "```\nTraceback (most recent call last):\n  File \"tests/test_setup.py\", line 88, in test_detect_gpu\n"
             "    info = detect_gpu(run=fake_run)\n  File \"src/setup/gpu.py\", line 41, in detect_gpu\n"
             "    mem = int(fields[2].strip().split()[0])\nIndexError: list index out of range\n```\n"
             "fake_run returns `NVIDIA GeForce RTX 3090, 535.104.05` for the nvidia-smi query "
             "`--query-gpu=name,driver_version,memory.total --format=csv,noheader`."},
        ],
        "max_tokens": 1024,
    })
    reqs.append({
        "name": "cuda_kernel",
        "messages": [
            {"role": "user", "content": "Write a CUDA kernel for a fused RMSNorm followed by a matrix-vector "
             "product with 4-bit block-quantized weights (32 weights per block, one fp16 scale). Explain "
             "the memory access pattern and how to make it bandwidth-bound."},
        ],
        "max_tokens": 1536,
    })
    reqs.append({
        "name": "typescript_ui",
        "messages": [
            {"role": "system", "content": AGENT_SYSTEM},
            {"role": "user", "content": "Add a `--json` flag to this CLI entry that prints the result as JSON "
             "instead of text. Show the full changed code.\n\n```ts\n" + read("ui/src/index.ts", 0, 200) + "\n```"},
        ],
        "max_tokens": 1536,
    })
    reqs.append({
        "name": "general_explain",
        "messages": [
            {"role": "user", "content": "Explain how mixture-of-experts language models work to someone who "
             "knows basic linear algebra, and why they are faster than dense models of the same size."},
        ],
        "max_tokens": 1024,
    })
    reqs.append({
        "name": "chinese",
        "messages": [
            {"role": "user", "content": "请用中文解释如何在只有 8GB 显存的电脑上运行大语言模型，并比较量化、"
             "CPU 卸载和从 SSD 流式加载三种方法的优缺点。"},
        ],
        "max_tokens": 1024,
    })
    reqs.append({
        "name": "math",
        "messages": [
            {"role": "user", "content": "A tank fills through pipe A in 6 hours and through pipe B in 4 hours, "
             "and drains through pipe C in 3 hours. All three open at noon with the tank one quarter full. "
             "When is it full or empty? Show your reasoning."},
        ],
        "max_tokens": 1024,
    })
    reqs.append({
        "name": "long_context_mk_server",
        "messages": [
            {"role": "system", "content": AGENT_SYSTEM},
            {"role": "user", "content": "Here is the engine's HTTP server and its README. Summarise the request "
             "flow from HTTP request to generated token, then suggest two concrete improvements.\n\n```python\n"
             + read("megakernel/server/mk_server.py") + "\n```\n\n" + read("README.md", 0, 250)},
        ],
        "max_tokens": 1536,
    })
    # multi-turn follow-up: the agent continues the review after "running the tests"
    reqs.append({
        "name": "review_connect_followup",
        "messages": [
            {"role": "system", "content": AGENT_SYSTEM},
            {"role": "user", "content": "Review this file for bugs.\n\n```python\n" + read("src/connect.py") + "\n```"},
            {"role": "assistant", "content": "I found two issues: the timeout is not passed to the request, and the "
             "error message drops the status code. I'll patch both."},
            {"role": "user", "content": "I applied your patch and ran `pytest tests/ -q`:\n\n```\n"
             "FAILED tests/test_connect.py::test_connect_timeout - TypeError: urlopen() got an unexpected keyword "
             "argument 'timeout_s'\n1 failed, 412 passed in 31.2s\n```\nFix it."},
        ],
        "max_tokens": 1024,
    })
    reqs.append({
        "name": "shell_task",
        "messages": [
            {"role": "system", "content": AGENT_SYSTEM},
            {"role": "user", "content": "Write a bash script that finds the 10 largest files under a directory, "
             "skipping .git and node_modules, prints human-readable sizes, and works on both Linux and macOS."},
        ],
        "max_tokens": 1024,
    })
    return reqs


def post(url: str, body: dict, timeout: float) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8099")
    ap.add_argument("--trace", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--temperature", type=float, default=1.0)  # generation_config.json default
    ap.add_argument("--only", nargs="*")
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for i, r in enumerate(build_requests()):
        if args.only and r["name"] not in args.only:
            continue
        before = os.path.getsize(args.trace)
        t0 = time.time()
        resp = post(args.url + "/v1/chat/completions", {
            "messages": r["messages"], "max_tokens": r["max_tokens"],
            "temperature": args.temperature, "seed": 1000 + i, "cache_prompt": False,
        }, timeout=1800)
        after = os.path.getsize(args.trace)
        usage = resp.get("usage", {})
        msg = resp["choices"][0]["message"]
        rec = {
            "name": r["name"], "trace_start": before, "trace_end": after, "seconds": round(time.time() - t0, 1),
            "prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": usage.get("completion_tokens"),
            "finish_reason": resp["choices"][0].get("finish_reason"),
            "reply_head": ((msg.get("reasoning_content") or "")[:200] + " || " + (msg.get("content") or "")[:300]),
        }
        with out.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"{r['name']}: prompt {rec['prompt_tokens']} gen {rec['completion_tokens']} in {rec['seconds']}s", flush=True)


if __name__ == "__main__":
    main()
