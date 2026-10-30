"""`openjet bench`: measure the configured local model on this machine and write a results file.

The file is what people share in a [Benchmark] issue or add to benchmarks/community/ in a pull request;
scripts/bench_table.py builds the README's community table from those files. Every run uses the same
fixed workload, so files from different machines compare:

  - reasoning: 4 short coding prompts with thinking on and OpenJet's thinking preset (temperature 1.0,
    top_p 0.95, top_k 20, min_p 0, fixed seed), 512 tokens each. This is how coding agents use the model.
  - greedy: 4 short chat prompts with thinking off and greedy decoding, 256 tokens each.
  - depth: at 4K, 32K and 100K tokens of context (those that fit the configured context window), a fresh
    prompt of synthetic source code with a coding task, thinking on: prompt speed, then decode speed at that
    depth. Then a question about a line hidden halfway into the same code checks that the model recalls it.

Speeds come from the timings the server reports, so they measure the model, not HTTP. The file holds every
request's timings, the GPU, driver, OS, CPU, RAM, model file name and runtime settings; no host name, user
name or paths.
"""

from __future__ import annotations

import asyncio
import json
import platform
import random
import re
import shutil
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

SCHEMA = "openjet-bench/1"
SEED = 1000
REASONING_TOKENS = 512
GREEDY_TOKENS = 256
DEPTHS = (4096, 32768, 98304)  # 96K fits a 100K context window
DEPTH_TOKENS = 256
RECALL_TOKENS = 512
REASONING_SAMPLING = {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0}
GREEDY_SAMPLING = {"temperature": 0.0, "top_k": 1}

REASONING_PROMPTS = (
    "Write a Python function that merges overlapping intervals in a list of [start, end] pairs, with a few pytest tests.",
    "Write a bash script that finds the 10 largest files under a directory (argument), skipping .git, and prints sizes in human-readable form.",
    "In C, implement a fixed-capacity ring buffer of ints with push, pop and is_full, safe for one producer and one consumer thread.",
    "Explain the difference between a mutex and a semaphore, and when to use each, with a short Python example of each.",
)
GREEDY_PROMPTS = (
    "Write a Python function that checks whether a number is prime, with a docstring and a few doctests.",
    "Explain how a hash map works internally, including collision handling and resizing, for a junior developer.",
    "Write a short story (about 300 words) about a lighthouse keeper who discovers a message in a bottle.",
    "Give me a step-by-step plan for migrating a Flask REST API to FastAPI, including testing and deployment.",
)
DEPTH_TASK = "Write a pytest test for the first function in the file above."
RECALL_TASK = "A comment in the file above gives the access code for the {name} deployment. What is the code?"
_CHARS_PER_TOKEN = 3.28  # Qwen3.8's tokenizer: 3.28-3.31 characters per token on the synthetic code below
_VERBS = ("load", "parse", "merge", "count", "score", "filter", "render", "flush", "index", "split")
_NOUNS = ("orders", "frames", "tokens", "events", "rows", "blocks", "spans", "users", "pages", "edges")


def depth_text(target_tokens: int, salt: str, needle: str) -> str:
    """Synthetic Python source of about *target_tokens* tokens with *needle* as a comment halfway in. *salt*
    starts the text, so no two runs share a cached prefix; the code itself is the same on every machine."""
    rnd = random.Random(target_tokens)
    parts = [f"# benchmark source {salt}\n"]
    size, i, placed = len(parts[0]), 0, False
    while size < target_tokens * _CHARS_PER_TOKEN:
        if not placed and size >= target_tokens * _CHARS_PER_TOKEN / 2:
            parts.append(f"# {needle}\n\n")
            placed = True
        verb, noun, field = rnd.choice(_VERBS), rnd.choice(_NOUNS), rnd.choice(_NOUNS)
        part = (
            f"def {verb}_{noun}_{i}(items, limit={rnd.randint(1, 999)}):\n"
            f'    """{verb.capitalize()} up to `limit` {noun} and return the weighted {field} total."""\n'
            f"    total = 0\n"
            f"    for item in items[:limit]:\n"
            f"        total += item.{field} * {rnd.randint(2, 97)} - {rnd.randint(0, 9)}\n"
            f"    return total\n\n"
        )
        parts.append(part)
        size += len(part)
        i += 1
    return "".join(parts)


def _run(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=15, check=True).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def _number(text: str) -> int | None:
    """nvidia-smi prints `[N/A]` for values a card does not report."""
    try:
        return round(float(text))
    except ValueError:
        return None


def _nvidia_gpus() -> list[dict]:
    if not shutil.which("nvidia-smi"):
        return []
    out = _run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version,power.limit",
                "--format=csv,noheader,nounits"])
    gpus = []
    for line in out.splitlines():
        fields = [f.strip() for f in line.split(",")]
        if len(fields) != 4:
            continue
        name, vram, driver, power = fields
        gpu = {"name": name, "vram_mb": _number(vram), "driver": driver}
        if _number(power) is not None:
            gpu["power_limit_w"] = _number(power)
        gpus.append(gpu)
    return gpus


def _os_name() -> str:
    if sys.platform == "linux":
        pretty = ""
        try:
            for line in Path("/etc/os-release").read_text().splitlines():
                if line.startswith("PRETTY_NAME="):
                    pretty = line.split("=", 1)[1].strip().strip('"')
        except OSError:
            pass
        wsl = " (WSL2)" if "microsoft" in platform.release().lower() else ""
        return f"{pretty or 'Linux'}{wsl}"
    if sys.platform == "darwin":
        return f"macOS {platform.mac_ver()[0]}"
    if sys.platform == "win32":
        return f"Windows {platform.release()} ({platform.version()})"
    return platform.platform()


def _cpu_name() -> str:
    if sys.platform == "linux":
        try:
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
        except OSError:
            pass
    if sys.platform == "darwin":
        return _run(["sysctl", "-n", "machdep.cpu.brand_string"]) or platform.machine()
    return platform.processor() or platform.machine()


def system_info(engine_gpu: str | None = None) -> dict:
    """Hardware the run used. *engine_gpu* is the nvidia-smi index OpenJet's engine was pinned to."""
    from .hardware import detect_hardware_info

    hw = detect_hardware_info()
    gpus = _nvidia_gpus()
    if gpus:
        index = int(engine_gpu) if engine_gpu is not None and engine_gpu.isdigit() else None
        if index is None or index >= len(gpus):
            index = max(range(len(gpus)), key=lambda i: gpus[i]["vram_mb"] or 0)
        gpu = dict(gpus[index])
    elif hw.has_metal:
        gpu = {"name": hw.label, "vram_mb": None, "unified_memory_gb": round(hw.total_ram_gb)}
    elif hw.has_cuda or hw.has_rocm or hw.has_vulkan:
        gpu = {"name": hw.label, "vram_mb": int(hw.vram_mb) or None}
    else:
        gpu = {"name": "CPU only", "vram_mb": None}
    return {"gpu": gpu, "gpu_count": max(len(gpus), 1), "os": _os_name(), "arch": platform.machine(),
            "cpu": _cpu_name(), "ram_gb": round(hw.total_ram_gb)}


def gpu_slug(name: str) -> str:
    name = re.sub(r"^(nvidia\s+)?(geforce\s+)?", "", name.strip(), flags=re.I)
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "gpu"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


async def _complete(http: httpx.AsyncClient, url: str, body: dict) -> tuple[dict, str]:
    r = await http.post(url, json=body)
    r.raise_for_status()
    data = r.json()
    timings = data.get("timings")
    if not timings or not timings.get("predicted_ms"):
        raise RuntimeError("the model server did not report timings, so the run cannot be measured")
    message = data["choices"][0]["message"]
    text = (message.get("reasoning_content") or "") + (message.get("content") or "")
    return {"prompt_tokens": int(timings["prompt_n"]), "prompt_ms": round(float(timings["prompt_ms"]), 1),
            "tokens": int(timings["predicted_n"]), "ms": round(float(timings["predicted_ms"]), 1)}, text


def _rate(tokens: int, ms: float) -> float:
    return round(tokens / ms * 1e3, 2) if ms > 0 else 0.0


def _decode_summary(runs: list[dict]) -> dict:
    tokens = sum(r["tokens"] for r in runs)
    ms = sum(r["ms"] for r in runs)
    return {"tok_s": _rate(tokens, ms), "tokens": tokens, "seconds": round(ms / 1e3, 3), "runs": runs}


async def run_workload(base_url: str, model: str, context_tokens: int, log=print) -> dict:
    """The fixed workload against an OpenAI-compatible server that reports llama.cpp-style timings."""
    url = f"{base_url}/v1/chat/completions"
    salt = f"{time.time_ns():x}"
    async with httpx.AsyncClient(timeout=httpx.Timeout(3600.0, connect=10.0)) as http:
        def body(prompt: str, sampling: dict, thinking: bool, max_tokens: int, fresh: bool = True) -> dict:
            # cache_prompt is llama-server's; OpenJet's engine reuses only a shared prefix, which fresh prompts lack
            return {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
                    "seed": SEED, "cache_prompt": not fresh, "chat_template_kwargs": {"enable_thinking": thinking},
                    **sampling}

        log("Warming up...")
        await _complete(http, url, body("Say hello.", GREEDY_SAMPLING, False, 16))
        results: dict = {}
        for name, prompts, sampling, thinking, n in (
            ("reasoning", REASONING_PROMPTS, REASONING_SAMPLING, True, REASONING_TOKENS),
            ("greedy", GREEDY_PROMPTS, GREEDY_SAMPLING, False, GREEDY_TOKENS),
        ):
            runs = []
            for i, prompt in enumerate(prompts):
                run, _ = await _complete(http, url, body(prompt, sampling, thinking, n))
                runs.append({"prompt": i, **run})
                log(f"  {name} {i + 1}/{len(prompts)}: {_rate(run['tokens'], run['ms']):.1f} tok/s")
            results[name] = _decode_summary(runs)
        results["depth"] = []
        for depth in DEPTHS:
            if depth + DEPTH_TOKENS + RECALL_TOKENS + 512 > context_tokens:
                log(f"  {depth // 1024}K context: skipped, the configured context window is {context_tokens} tokens")
                continue
            rnd = random.Random(f"{salt}-{depth}")
            name, code = f"{rnd.choice(_NOUNS)}-{rnd.randint(100, 999)}", str(rnd.randint(100000, 999999))
            source = depth_text(depth, f"{salt}-{depth}", f"The access code for the {name} deployment is {code}.")
            run, _ = await _complete(http, url, body(f"{source}\n{DEPTH_TASK}", REASONING_SAMPLING, True, DEPTH_TOKENS))
            # the same source again: the server reuses it, so this request only processes the question
            recall, text = await _complete(http, url, body(f"{source}\n{RECALL_TASK.format(name=name)}",
                                                           GREEDY_SAMPLING, True, RECALL_TOKENS, fresh=False))
            entry = {"depth": depth, **run, "prefill_tok_s": _rate(run["prompt_tokens"], run["prompt_ms"]),
                     "decode_tok_s": _rate(run["tokens"], run["ms"]), "recall": code in text,
                     "recall_run": recall}
            results["depth"].append(entry)
            log(f"  {run['prompt_tokens']} tokens of context: prompt {entry['prefill_tok_s']:.0f} tok/s, "
                f"decode {entry['decode_tok_s']:.1f} tok/s, recall {'passed' if entry['recall'] else 'FAILED'}")
    return results


def _runtime_info(cfg: dict, backend: str, model: str, context: int) -> dict:
    from .llama_server import model_runtime_label

    info = {"engine": "openjet" if backend == "megakernel" else "llama.cpp", "context_tokens": context}
    if backend == "megakernel":
        info["variant"] = model_runtime_label(model)
    else:
        info["device"] = str(cfg.get("device") or "auto")
        if cfg.get("gpu_layers") is not None:
            info["gpu_layers"] = cfg.get("gpu_layers")
    return info


async def bench(*, output: Path | None, contributor: str | None, notes: str | None) -> Path:
    from .config import load_config
    from .llama_server import _megakernel_gpu
    from .megakernel_build import _openjet_version
    from .runtime_registry import DEFAULT_RUNTIME, active_model_ref, active_runtime, create_runtime_client

    cfg = load_config()
    if active_runtime(cfg) != DEFAULT_RUNTIME:
        raise SystemExit("`openjet bench` measures a local model; the active profile uses a remote runtime. "
                         "Pick a local model with `/model` or `openjet setup`.")
    model = str(active_model_ref(cfg) or "")
    port = _free_port()
    client = create_runtime_client(cfg, host="127.0.0.1", port=port)
    print(f"Loading {Path(model).name}...", flush=True)
    await client.start()
    try:
        backend = client.active_backend
        print(f"Benchmarking with {'OpenJet engine' if backend == 'megakernel' else 'llama.cpp'}. "
              "This takes a few minutes; keep other GPU work stopped.", flush=True)
        context = client.context_window_tokens
        results = await run_workload(f"http://127.0.0.1:{port}", Path(model).stem, context,
                                     log=lambda m: print(m, flush=True))
    finally:
        await client.close()
    model_path = Path(model)
    record = {
        "schema": SCHEMA,
        "created": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "contributor": contributor,
        "openjet_version": _openjet_version(),
        "system": system_info(_megakernel_gpu() if backend == "megakernel" else None),
        "model": {"file": model_path.name,
                  "size_bytes": model_path.stat().st_size if model_path.is_file() else None},
        "runtime": _runtime_info(cfg, backend, model, context),
        "settings": {"seed": SEED, "reasoning": {**REASONING_SAMPLING, "thinking": True, "max_tokens": REASONING_TOKENS},
                     "greedy": {**GREEDY_SAMPLING, "thinking": False, "max_tokens": GREEDY_TOKENS},
                     "depth": {**REASONING_SAMPLING, "thinking": True, "max_tokens": DEPTH_TOKENS},
                     "recall": {**GREEDY_SAMPLING, "thinking": True, "max_tokens": RECALL_TOKENS}},
        "results": results,
    }
    if notes:
        record["notes"] = notes
    if output is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M")
        output = Path(f"openjet-bench-{gpu_slug(record['system']['gpu']['name'])}-{stamp}.json")
    output.write_text(json.dumps(record, indent=2) + "\n")
    return output


def summary(record: dict) -> str:
    res = record["results"]
    lines = [f"{record['system']['gpu']['name']}, {record['model']['file']}, "
             f"{'OpenJet engine' if record['runtime']['engine'] == 'openjet' else 'llama.cpp'}",
             f"  reasoning   {res['reasoning']['tok_s']:.1f} tok/s",
             f"  greedy      {res['greedy']['tok_s']:.1f} tok/s"]
    for d in res["depth"]:
        lines.append(f"  {d['depth'] // 1024}K context  prompt {d['prefill_tok_s']:.0f} tok/s, decode "
                     f"{d['decode_tok_s']:.1f} tok/s, recall {'passed' if d['recall'] else 'failed'}")
    return "\n".join(lines)


SHARE = """
Share it so your card shows up in the community table:
  - open a [Benchmark] issue and attach or paste the file:
    https://github.com/l-forster/open-jet/issues/new?template=benchmark.yml
  - or open a pull request that adds only this file to benchmarks/community/
"""


def run_bench(*, output: str | None, contributor: str | None, notes: str | None) -> None:
    try:
        path = asyncio.run(bench(output=Path(output) if output else None, contributor=contributor, notes=notes))
    except KeyboardInterrupt:
        raise SystemExit(130)
    record = json.loads(path.read_text())
    print("\n" + summary(record))
    print(f"\nWrote {path}")
    if not contributor:
        print("Add `--github <your-handle>` to be credited in the table and release notes.")
    print(SHARE.rstrip())
