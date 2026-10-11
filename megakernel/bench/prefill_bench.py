#!/usr/bin/env python3
"""Matched, uncached prefill-to-first-token benchmark (no decode throughput).

Prepare shared token IDs once (--backend prepare), then run each backend
independently, pinned to a GPU UUID::

    python3 megakernel/bench/prefill_bench.py --backend megakernel \
        --gpu GPU-00000000-0000-0000-0000-000000000003 --kv q8 \
        --workload benchmark_results/prefill-workloads.json \
        --output benchmark_results/prefill-megakernel.json

Each length gets one unreported warmup and three timed repetitions. llama owns
its server and erases slot 0 before EVERY request. timings.prompt_n must equal
the input length; tokens_cached is final occupancy, not a cache-hit count.
Megakernel defaults to its existing synchronous C API, loaded once and reset
before each request; one same-process warmup per length excludes lazy allocation.
--mk-mode cli uses fresh processes (-n 1), whose GPU/file-cache-only warmup does
NOT exclude per-process lazy allocation. --draft 4 includes initial MTP drafts,
without generating any extra output tokens.
Both native timers include final prompt evaluation/lm-head/first-token sampling;
MK additionally includes checkpoint work, and -d 4 includes initial MTP drafts.
Model loading, HTTP transfer, and slot erasure are outside the native timers.
"""

import argparse
from contextlib import contextmanager
import ctypes
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

_SPEC = importlib.util.spec_from_file_location("prefill_think_helpers", Path(__file__).with_name("think_bench.py"))
_HELPERS = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_HELPERS)
gpu_env = _HELPERS.gpu_env
ROOT = Path(__file__).resolve().parents[2]
LENGTHS = (2048, 4096, 8192)


def gpu_uuid(value):
    value = _HELPERS.gpu_identifier(value)
    if not value.startswith(("GPU-", "MIG-")):
        raise argparse.ArgumentTypeError("real prefill runs require a GPU/MIG UUID, not an ordinal")
    return value


def benchmark_env(gpu):
    env = gpu_env(gpu_uuid(gpu))
    for key in ("MK_PREFILL", "MK_PROFILE", "MK_PF_PROFILE"):
        env.pop(key, None)
    return env


def token_hash(ids):
    return hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()


def fingerprint(path):
    path = Path(path).resolve(strict=True)
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(f"file changed while hashing: {path}")
    return dict(path=str(path), size_bytes=after.st_size, mtime_ns=after.st_mtime_ns, sha256=digest.hexdigest())


def workload_from_ids(ids, model, source):
    if len(ids) < max(LENGTHS):
        raise ValueError("tokenizer must produce at least 8192 tokens")
    return dict(schema_version=1, source=source, model=model,
                prompts=[dict(n=n, token_ids=ids[:n], token_sha256=token_hash(ids[:n])) for n in LENGTHS])


def source_text():
    text = ("Synthetic benchmark ledger and Python arithmetic examples. This original generated text "
            "is dedicated to the public domain under CC0.\n\n")
    for i in range(1024):
        text += (f"Record {i:04d}: A garden contains {i % 19 + 3} rows of beans and {i % 13 + 2} rows of peas. "
                 "Count each row once; keep the ledger sorted by record number.\n")
        text += (f"def ledger_{i:04d}(beans, peas):\n    total = beans + peas\n"
                 f'    return {{"record": {i}, "total": total, "difference": beans - peas}}\n\n')
    return text


def prepare_workload(args, model):
    previous = os.environ.copy()
    env = benchmark_env(args.gpu)
    os.environ.clear()
    os.environ.update(env)
    engine = None
    try:
        engine = engine_class()(args.megakernel_lib, args.model, args.context, 0, "", 1)
        engine.L.mk_free.restype = None
        engine.L.mk_free.argtypes = [ctypes.c_void_p]
        text = source_text()
        ids = engine.tokenize(text)
        return workload_from_ids(ids, model, dict(
            description="Original deterministic CC0 garden ledger and Python code; raw tokenization prefix slices",
            text_sha256=hashlib.sha256(text.encode()).hexdigest(),
            add_special=False, parse_special=True, total_source_tokens=len(ids)))
    finally:
        if engine is not None:
            engine.L.mk_free(engine.h)
        os.environ.clear()
        os.environ.update(previous)


def load_workload(path, model):
    data = json.loads(Path(path).read_text())
    if data.get("schema_version") != 1:
        raise ValueError("unsupported workload schema")
    if data["model"]["sha256"] != model["sha256"] or data["model"]["size_bytes"] != model["size_bytes"]:
        raise ValueError("workload tokenizer model differs from benchmark model")
    if [p["n"] for p in data["prompts"]] != list(LENGTHS):
        raise ValueError("workload must contain exact 2048/4096/8192-token inputs")
    for prompt in data["prompts"]:
        ids = prompt["token_ids"]
        if len(ids) != prompt["n"] or any(type(i) is not int or i < 0 for i in ids):
            raise ValueError("invalid token IDs or prompt count")
        if token_hash(ids) != prompt["token_sha256"]:
            raise ValueError("workload token hash mismatch")
    return data


def positive_time(seconds):
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("prefill seconds must be finite and positive")
    return seconds


def parse_mk(stderr, expected):
    matches = re.findall(r"\[mk\] prompt (\d+) tok in ([\d.eE+-]+) s \(([\d.eE+-]+) tok/s\)", stderr)
    if len(matches) != 1:
        raise ValueError("expected exactly one megakernel prompt timing")
    n, seconds, reported = matches[0]
    if int(n) != expected:
        raise ValueError(f"megakernel processed {n} tokens, expected {expected}")
    seconds = positive_time(float(seconds))
    positive_time(float(reported))
    decode = re.findall(r"\[mk\] decode (\d+) tok", stderr)
    if decode != ["0"]:
        raise ValueError("megakernel must generate only the first token, with zero timed decode tokens")
    return dict(n=expected, seconds=seconds, tokens_per_second=expected / seconds,
                reported_tokens_per_second=float(reported), stderr=stderr)


def parse_llama(response, expected):
    timings = response["timings"]
    n = timings["prompt_n"]
    if type(n) is not int or n != expected:
        raise ValueError(f"llama evaluated {n} prompt tokens, expected {expected}; cached reuse is not prefill")
    if response.get("truncated") is not False or response.get("tokens_evaluated") != expected:
        raise ValueError("llama input was truncated or its evaluated count differs")
    if timings.get("predicted_n") != 1 or response.get("tokens_predicted") != 1:
        raise ValueError("llama must generate exactly one token")
    # In this server, tokens_cached reports the final slot contents, not input reuse.
    if response.get("tokens_cached") != expected:
        raise ValueError("unexpected llama final slot occupancy")
    seconds = positive_time(float(timings["prompt_ms"]) / 1000)
    return dict(n=n, seconds=seconds, tokens_per_second=n / seconds, response=response)


def aggregate(samples):
    if not samples:
        raise ValueError("no timed prefill samples")
    for sample in samples:
        if type(sample["n"]) is not int or sample["n"] <= 0:
            raise ValueError("processed-token count must be positive")
        positive_time(sample["seconds"])
    tokens = sum(s["n"] for s in samples)
    seconds = sum(s["seconds"] for s in samples)
    return dict(processed_tokens=tokens, seconds=seconds, tokens_per_second=tokens / seconds,
                sample_count=len(samples))


def post(url, payload):
    request = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=1800) as response:
        return json.loads(response.read())


def stop_child(proc):
    if proc.poll() is None:
        proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


@contextmanager
def llama_server(args, artifact_dir):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind(("127.0.0.1", args.port))
        except OSError as exc:
            raise RuntimeError(f"port {args.port} is occupied; refusing existing server") from exc
    log_path = artifact_dir / "llama-server.log"
    command = [args.llama_server, "-m", args.model, "--host", "127.0.0.1", "--port", str(args.port),
               "--parallel", "1", "--slots", "--slot-save-path", str(artifact_dir.resolve()),
               "--flash-attn", "on", "-ctk", "q8_0", "-ctv", "q8_0",
               "-ngl", "99", "-c", str(args.context), "--spec-type", "draft-mtp",
               "--spec-draft-n-max", "2", "--no-warmup",
               "--cache-ram", "0", "-b", "2048", "-ub", "512"]
    with log_path.open("w") as log:
        proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=benchmark_env(args.gpu))
    url = f"http://127.0.0.1:{args.port}"
    try:
        deadline = time.monotonic() + args.startup_timeout
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                raise RuntimeError(f"llama-server exited {proc.returncode}; see {log_path}")
            try:
                with urllib.request.urlopen(url + "/health", timeout=2) as response:
                    if response.status == 200:
                        if proc.poll() is not None:
                            raise RuntimeError(f"llama-server exited; see {log_path}")
                        break
            except (urllib.error.URLError, TimeoutError):
                pass
            time.sleep(1)
        else:
            raise TimeoutError(f"llama-server startup timeout; see {log_path}")
        yield url, proc, command
    finally:
        stop_child(proc)


def run_llama(args, workload, artifact_dir):
    samples = []
    with llama_server(args, artifact_dir) as (url, proc, command):
        for prompt in workload["prompts"]:
            for iteration in range(4):
                if proc.poll() is not None:
                    raise RuntimeError(f"owned llama-server exited {proc.returncode}")
                erased = post(url + "/slots/0?action=erase", {})
                if erased.get("id_slot") != 0 or type(erased.get("n_erased")) is not int or erased["n_erased"] < 0:
                    raise ValueError("slot erasure was not acknowledged")
                response = post(url + "/completion",
                                dict(prompt=prompt["token_ids"], n_predict=1, cache_prompt=False, id_slot=0,
                                     temperature=0.0, top_k=1, seed=1000, stream=False))
                if proc.poll() is not None:
                    raise RuntimeError(f"owned llama-server exited {proc.returncode}")
                sample = parse_llama(response, prompt["n"])
                if iteration:
                    samples.append(dict(repetition=iteration, token_sha256=prompt["token_sha256"],
                                        slot_erase=erased, **sample))
                    print(f"[llama] {prompt['n']} rep {iteration}: {sample['tokens_per_second']:.2f} tok/s", flush=True)
    return samples, command


def run_mk(args, workload, artifact_dir):
    samples, commands = [], []
    for prompt in workload["prompts"]:
        ids_path = artifact_dir / f"prompt-{prompt['n']}.ids"
        ids_path.write_text(" ".join(map(str, prompt["token_ids"])) + "\n")
        command = [args.megakernel, "-m", args.model, "--prompt-ids", str(ids_path),
                   "--raw", "-n", "1", "-d", str(args.draft), "-c", str(args.context), "--kv", "q8",
                   "--draft-vocab", args.draft_vocab, "--temp", "0", "--quiet"]
        commands.append(command)
        for iteration in range(4):
            result = subprocess.run(command, capture_output=True, text=True, env=benchmark_env(args.gpu),
                                    timeout=1800)
            (artifact_dir / f"mk-{prompt['n']}-{'warmup' if not iteration else iteration}.stderr.log").write_text(result.stderr)
            if result.returncode:
                raise RuntimeError(f"megakernel exited {result.returncode}: {result.stderr[-2000:]}")
            sample = parse_mk(result.stderr, prompt["n"])
            if iteration:
                samples.append(dict(repetition=iteration, token_sha256=prompt["token_sha256"],
                                    stdout=result.stdout, **sample))
                print(f"[megakernel] {prompt['n']} rep {iteration}: {sample['tokens_per_second']:.2f} tok/s", flush=True)
    return samples, commands


def engine_class():
    spec = importlib.util.spec_from_file_location("prefill_mk_server", ROOT / "megakernel" / "server" / "mk_server.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.Engine


def run_mk_api(args, workload, artifact_dir):
    # No CUDA library is loaded before this point. Pin before Engine loads it.
    previous = os.environ.copy()
    env = benchmark_env(args.gpu)
    os.environ.clear()
    os.environ.update(env)
    engine = None
    samples = []
    try:
        engine = engine_class()(args.megakernel_lib, args.model, args.context, args.draft, args.draft_vocab, 1)
        engine.L.mk_free.restype = None
        engine.L.mk_free.argtypes = [ctypes.c_void_p]
        engine.set_sampling(0.0, 1.0, 1, 0.0, 1000)
        for prompt in workload["prompts"]:
            for iteration in range(4):
                engine.reset()
                if engine.sequence():
                    raise ValueError("megakernel reset left cached input")
                start = time.perf_counter()
                first_token = engine.prefill(prompt["token_ids"])
                seconds = positive_time(time.perf_counter() - start)
                sequence = engine.sequence()
                if sequence != prompt["token_ids"] + [first_token] or engine.committed() != prompt["n"]:
                    raise ValueError("megakernel prefill input/processed-token count differs")
                if iteration:
                    sample = dict(n=prompt["n"], seconds=seconds,
                                  tokens_per_second=prompt["n"] / seconds, first_token=first_token,
                                  processed_prompt_tokens=engine.committed(), repetition=iteration,
                                  token_sha256=prompt["token_sha256"])
                    samples.append(sample)
                    print(f"[megakernel API] {prompt['n']} rep {iteration}: {sample['tokens_per_second']:.2f} tok/s",
                          flush=True)
    finally:
        if engine is not None:
            engine.L.mk_free(engine.h)
        os.environ.clear()
        os.environ.update(previous)
    return samples, dict(api="mk_new/reset/prefill/free", library=args.megakernel_lib,
                         context=args.context, n_draft=args.draft, draft_vocab=args.draft_vocab,
                         kv_mode=1, sampling="greedy")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backend", required=True, choices=("prepare", "llama", "megakernel"))
    parser.add_argument("--model", default=str(ROOT / "models" / "Qwen3.8-27B-Q4_K_M.gguf"))
    parser.add_argument("--llama-server", default=str(ROOT / "llama.cpp" / "build" / "bin" / "llama-server"))
    parser.add_argument("--megakernel", default=str(ROOT / "megakernel" / "qwen38_mk"))
    parser.add_argument("--megakernel-lib", default=str(ROOT / "megakernel" / "libqwen38mk.so"))
    parser.add_argument("--mk-mode", choices=("api", "cli"), default="api")
    parser.add_argument("--gpu", required=True, type=gpu_uuid)
    parser.add_argument("--context", type=_HELPERS.positive_int, default=16384)
    parser.add_argument("--kv", choices=("q8",), default="q8")
    parser.add_argument("--draft", type=int, choices=(0, 4), default=4)
    parser.add_argument("--draft-vocab", default=str(ROOT / "megakernel" / "draft_vocab_32k.bin"))
    parser.add_argument("--workload", default=str(ROOT / "benchmark_results" / "prefill-workloads.json"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--port", type=int, default=8095)
    parser.add_argument("--startup-timeout", type=_HELPERS.positive_int, default=600)
    args = parser.parse_args()
    if args.context < max(LENGTHS) + 9:
        parser.error("context must leave room beyond 8192 prompt tokens (megakernel requires nine)")
    if not 1 <= args.port <= 65535:
        parser.error("port must be in 1..65535")
    output = Path(args.output)
    artifact_dir = output.parent / (output.stem + "-artifacts")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    model = fingerprint(args.model)
    if args.backend == "prepare":
        output.write_text(json.dumps(prepare_workload(args, model), indent=2, allow_nan=False) + "\n")
        print(f"Prepared shared 2048/4096/8192-token workloads: {output}")
        return
    workload = load_workload(args.workload, model)
    executable = fingerprint(args.llama_server if args.backend == "llama" else
                             args.megakernel_lib if args.mk_mode == "api" else args.megakernel)
    env = benchmark_env(args.gpu)
    runner = run_llama if args.backend == "llama" else run_mk_api if args.mk_mode == "api" else run_mk
    samples, commands = runner(args, workload, artifact_dir)
    scope = ("Prefill-to-first-token timer; all prompt tokens evaluated including "
             "lm-head. Excludes model loading, HTTP transfer and slot erase. "
             + ("Includes final prompt token through decode megakernel, checkpoint work, and "
                f"initial {args.draft} MTP drafts. "
                + ("Python perf_counter around synchronous Engine.prefill; includes ctypes input conversion and "
                   "C API vector copy; reset and count validation excluded; same-process allocation warmup."
                   if args.mk_mode == "api" else
                   "CLI native timer includes lazy per-process allocation; warmup only warms GPU/file cache.")
                if args.backend == "megakernel"
                else "Server times from prompt-processing start through synchronized first-token sampling."))
    report = dict(schema_version=1, settings=dict(vars(args), repetitions=3, warmups_per_length=1,
                  n_predict=1, timing_scope=scope,
                  aggregation="total processed prompt tokens / total measured prompt-processing seconds",
                  cache_policy="slot erase before every request" if args.backend == "llama" else
                  "engine reset before every request" if args.mk_mode == "api" else "fresh process per request",
                  cuda_environment={k: env.get(k) for k in ("CUDA_DEVICE_ORDER", "CUDA_VISIBLE_DEVICES")},
                  cleared_environment=["MK_PREFILL", "MK_PROFILE", "MK_PF_PROFILE"],
                  llama_speculation="draft-mtp", llama_draft_max=2,
                  llama_batch=2048, llama_ubatch=512, mk_prefill_cap=2048),
                  model=model, executable=executable, workload=fingerprint(args.workload),
                  inputs=[{k: p[k] for k in ("n", "token_sha256")} for p in workload["prompts"]],
                  commands=commands, samples=samples, aggregate=aggregate(samples),
                  by_length={str(n): aggregate([s for s in samples if s["n"] == n]) for n in LENGTHS})
    if args.backend == "megakernel":
        report["draft_vocabulary"] = fingerprint(args.draft_vocab)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report["by_length"], indent=2))


if __name__ == "__main__":
    main()
