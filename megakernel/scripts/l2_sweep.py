#!/usr/bin/env python3
"""Times the L2 prefetch budgets of a -DMK_L2_TUNE=1 build on this GPU against its defaults (docs/EXPERIMENTAL.md)."""
import argparse
import glob
import hashlib
import json
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WIN = os.name == "nt"
KNOBS = {"MK_L2_DN": 4096, "MK_L2_ATTN": 8, "MK_L2_ATTN_CTX": 1280, "MK_L2_TOP": 1, "MK_L2_CK": 12, "MK_L2_MTP": 11,
         "MK_L2_MTP_R": 12, "MK_L2_MTP_KEEP": 0}
PROMPTS = ["Write a Python function that checks whether a number is prime, with a docstring and a few doctests.",
           "Explain how a hash map works internally, including collision handling and resizing, for a junior developer.",
           "Write a short story (about 300 words) about a lighthouse keeper who discovers a message in a bottle.",
           "Give me a step-by-step plan for migrating a Flask REST API to FastAPI, including testing and deployment."]
GRID_QUICK = ["MK_L2_DN=8192", "MK_L2_DN=65536", "MK_L2_TOP=3", "MK_L2_ATTN=18,MK_L2_ATTN_CTX=1000000",
              "MK_L2_MTP=23,MK_L2_MTP_R=23", "MK_L2_MTP_KEEP=1", "MK_L2_MTP=23,MK_L2_MTP_R=23,MK_L2_MTP_KEEP=1"]
GRID_FULL = ["MK_L2_DN=0", "MK_L2_DN=8192", "MK_L2_DN=65536", "MK_L2_TOP=0", "MK_L2_TOP=2", "MK_L2_TOP=3", "MK_L2_TOP=5",
             "MK_L2_ATTN=0", "MK_L2_ATTN=18", "MK_L2_ATTN=72", "MK_L2_ATTN=8,MK_L2_ATTN_CTX=1000000",
             "MK_L2_ATTN=18,MK_L2_ATTN_CTX=1000000", "MK_L2_ATTN=72,MK_L2_ATTN_CTX=1000000", "MK_L2_MTP=0,MK_L2_MTP_R=0",
             "MK_L2_MTP=23,MK_L2_MTP_R=23", "MK_L2_MTP=6,MK_L2_MTP_R=6,MK_L2_MTP_KEEP=1", "MK_L2_MTP_KEEP=1",
             "MK_L2_MTP=23,MK_L2_MTP_R=23,MK_L2_MTP_KEEP=1"]
DESCRIPTION = """\
Times L2 prefetch settings against the build's defaults, one setting per process, and prints a table to paste into a
report. Needs a -DMK_L2_TUNE=1 build and the GPU to itself. Each process loads the model, warms the GPU up, decodes the
four short prompts greedily and, unless --no-long, a long prompt. Runs alternate with runs of the defaults, and each
setting is compared with the defaults on both sides of it, over --rounds rounds (the second in reverse order). Prefetches
are hints, so every setting must give the same tokens: a difference is reported as OUTPUT CHANGED. A change smaller than
the defaults' own run-to-run change is marked as noise with ~. Raw logs and results.json stay in --out."""


def parse_setting(text):
    """'MK_L2_DN=8192,MK_L2_TOP=2' -> {'MK_L2_DN': '8192', 'MK_L2_TOP': '2'}; a typo must not time the defaults twice"""
    out = {}
    for part in filter(None, text.split(",")):
        name, sep, value = part.strip().partition("=")
        if not sep or name not in KNOBS or not value.isdigit():
            raise ValueError(f"{part!r}: expected NAME=integer with NAME one of {', '.join(KNOBS)}")
        out[name] = value
    return out


def schedule(settings, rounds, every=2):
    """run order: (round, index of the setting or None for the defaults), the defaults first, last and every `every` settings"""
    order = []
    for r in range(rounds):
        idx = list(range(len(settings)))
        if r % 2:
            idx.reverse()
        order.append((r, None))
        for k, i in enumerate(idx):
            order.append((r, i))
            if (k + 1) % every == 0 and k + 1 < len(idx):
                order.append((r, None))
        order.append((r, None))
    return order


def base_around(runs, pos, metric):
    """the defaults' value at run `pos`: the mean of the nearest runs of the defaults before and after it"""
    def nearest(rng):
        for k in rng:
            if runs[k]["setting"] is None and runs[k].get(metric) is not None:
                return runs[k][metric]
        return None
    vals = [v for v in (nearest(range(pos - 1, -1, -1)), nearest(range(pos + 1, len(runs)))) if v is not None]
    return sum(vals) / len(vals) if vals else None


def noise_of(runs, metric):
    """the largest change between neighbouring runs of the defaults"""
    b = [r[metric] for r in runs if r["setting"] is None and r.get(metric) is not None]
    return max((abs(y / x - 1) for x, y in zip(b, b[1:])), default=0.0)


def analyse(runs, settings):
    rows = []
    for i, s in enumerate(settings):
        row = {"setting": s, "metrics": {}, "ids_same": True}
        for metric in ("short", "long", "prefill"):
            vals, ratios = [], []
            for pos, r in enumerate(runs):
                if r["setting"] == i and r.get(metric) is not None:
                    base = base_around(runs, pos, metric)
                    vals.append(r[metric])
                    if base:
                        ratios.append(r[metric] / base - 1)
            if ratios:
                row["metrics"][metric] = {"value": sum(vals) / len(vals), "delta": sum(ratios) / len(ratios),
                                          "spread": max(ratios) - min(ratios)}
        ref = next((r["ids"] for r in runs if r["setting"] is None), None)
        row["ids_same"] = all(r["ids"] == ref for r in runs if r["setting"] == i)
        rows.append(row)
    return rows, {m: noise_of(runs, m) for m in ("short", "long", "prefill")}


def fmt_delta(m, noise):
    if not m:
        return ""
    return f"{'~' if abs(m['delta']) <= noise else ''}{m['delta'] * 100:+.1f}%"


def table(rows, runs, noise):
    def mean(metric):
        v = [r[metric] for r in runs if r["setting"] is None and r.get(metric) is not None]
        return sum(v) / len(v) if v else None
    long_on = mean("long") is not None
    head = ["setting", "short prompts, tok/s", "vs defaults"] + (["after the long prompt, tok/s", "vs defaults", "reading it, tok/s", "vs defaults"] if long_on else [])
    lines = ["| " + " | ".join(head + ["tokens"]) + " |", "|" + "---|" * (len(head) + 1)]
    def cells(name, short, long_, prefill):
        c = [name, *short] + ([*long_, *prefill] if long_on else [])
        return c
    b = [f"{mean('short'):.1f}" if mean("short") else "", ""]
    lb, pb = [f"{mean('long'):.1f}", ""] if long_on else [], [f"{mean('prefill'):.0f}", ""] if long_on else []
    lines.append("| " + " | ".join(cells("defaults", b, lb, pb) + [""]) + " |")
    for row in rows:
        m = row["metrics"]
        def pair(metric, f):
            return [f.format(m[metric]["value"]), fmt_delta(m[metric], noise[metric])] if metric in m else ["", ""]
        name = ", ".join(f"{k.replace('MK_L2_', '')}={v}" for k, v in parse_setting(row["setting"]).items())
        lines.append("| " + " | ".join(cells(name, pair("short", "{:.1f}"), pair("long", "{:.1f}"), pair("prefill", "{:.0f}"))
                                       + ["same" if row["ids_same"] else "**OUTPUT CHANGED**"]) + " |")
    lines.append("")
    lines.append("The defaults' own run-to-run change: " + ", ".join(f"{k} {v * 100:.1f}%" for k, v in noise.items() if v or k == "short")
                 + "; a smaller change is marked ~.")
    return "\n".join(lines)


def long_text(chars):
    """the package's own documentation, repeated up to `chars` characters, then a question"""
    parts = []
    for p in sorted(glob.glob(os.path.join(HERE, "docs", "*.md"))) + [os.path.join(HERE, "README.md")]:
        if os.path.isfile(p):
            with open(p, encoding="utf-8", errors="replace") as f:
                parts.append(f.read())
    text = "\n\n".join(parts) or "\n".join(PROMPTS)
    text = (text * (chars // len(text) + 1))[:chars]
    return text + "\n\nSummarize the text above: what is this engine, and how is it configured?"


def quoted(s):
    """one argument of a --batch line: '...' is taken literally there"""
    if "'" in s:
        raise ValueError(f"a path with a quote cannot go through --batch: {s}")
    return f"'{s}'"


def job_results(prefix):
    with open(prefix + ".err", encoding="utf-8", errors="replace") as f:
        err = f.read()
    with open(prefix + ".out", encoding="utf-8", errors="replace") as f:
        out = f.read()
    d = re.search(r"decode (\d+) tok in ([\d.]+) s = ([\d.]+) tok/s.*accepted (\d+)/(\d+)", err)
    p = re.search(r"prompt (\d+) tok in ([\d.]+) s \(([\d.]+) tok/s\)", err)
    ids = re.search(r"^IDS:(.*)$", out, re.M)
    if not d or not p or not ids:
        raise RuntimeError(f"no timing or token ids in {prefix}.err / .out:\n{err[-1500:]}")
    return {"decode": float(d.group(3)), "prompt_tokens": int(p.group(1)), "prefill": float(p.group(3)),
            "accepted": int(d.group(4)), "drafted": int(d.group(5)), "ids": hashlib.sha256(ids.group(1).encode()).hexdigest()[:16]}


def run_process(a, setting, tag, files, profile=False):
    """one process of the CLI under `setting`; returns the run's rates and a hash of every job's tokens"""
    env = {k: v for k, v in os.environ.items() if k not in KNOBS and k != "MK_PROFILE"}
    env.update(setting)
    if profile:
        env["MK_PROFILE"] = "1"
    prefix = os.path.join(a.out, tag)
    jobs = [("warm", files["short"][0], 512)]
    jobs += [(f"p{i}", f, a.n) for i, f in enumerate(files["short"])]
    if files["long"]:
        jobs.append(("long", files["long"], a.n))
    batch = prefix + ".jobs"
    with open(batch, "w", encoding="utf-8", newline="\n") as f:
        for name, path, n in jobs:
            f.write(f"-f {quoted(path)} -n {n} -d {a.draft} --ids --quiet -o {quoted(prefix + '_' + name)}\n")
    cmd = [a.exe, "-m", a.model, "--kv", a.kv, "-c", str(a.ctx), "--batch", batch]
    if a.draft_vocab:
        cmd += ["--draft-vocab", a.draft_vocab]
    if a.dry_run:
        print(" ".join(f"{k}={v}" for k, v in setting.items()), " ".join(cmd))
        return None
    r = subprocess.run(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env, timeout=a.timeout,
                       encoding="utf-8", errors="replace")
    with open(prefix + ".log", "w", encoding="utf-8") as f:
        f.write(r.stderr)
    if r.returncode:
        raise RuntimeError(f"{tag}: the CLI ended with {r.returncode}:\n{r.stderr[-1500:]}")
    m = re.search(r"\[mk\] MK_L2_TUNE: (.*)", r.stderr)
    if not m:
        raise RuntimeError("this build has no MK_L2_* settings: it needs -DMK_L2_TUNE=1 (docs/EXPERIMENTAL.md)")
    applied = dict(kv.split("=") for kv in m.group(1).split())
    for k, v in setting.items():
        if applied.get(k) != str(int(v)):
            raise RuntimeError(f"{tag}: asked for {k}={v}, the engine reports {applied.get(k)}")
    res = {name: job_results(prefix + "_" + name) for name, _, _ in jobs}
    out = {"short": sum(res[f"p{i}"]["decode"] for i in range(len(files["short"]))) / len(files["short"]),
           "ids": [res[name]["ids"] for name, _, _ in jobs if name != "warm"],
           "applied": applied}
    if files["long"]:
        out.update(long=res["long"]["decode"], prefill=res["long"]["prefill"], long_tokens=res["long"]["prompt_tokens"],
                   long_accepted=res["long"]["accepted"], long_drafted=res["long"]["drafted"])
    return out


def gpu_info():
    try:
        q = "name,driver_version,power.limit,temperature.gpu,clocks.max.memory,utilization.gpu"
        r = subprocess.run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader"], capture_output=True, text=True, timeout=30)
        return r.stdout.strip().splitlines()[0] if r.returncode == 0 and r.stdout.strip() else ""
    except (OSError, subprocess.SubprocessError):
        return ""


def main():
    ap = argparse.ArgumentParser(description=DESCRIPTION, formatter_class=argparse.RawDescriptionHelpFormatter)
    dv = os.path.join(HERE, "draft_vocab_32k.bin")
    ap.add_argument("-m", "--model", default=os.environ.get("MK_MODEL"), help="the GGUF (default: MK_MODEL)")
    ap.add_argument("--exe", default=os.path.join(HERE, "qwen38_mk.exe" if WIN else "qwen38_mk"), help="the CLI of the build")
    ap.add_argument("--draft-vocab", default=dv if os.path.isfile(dv) else "", help="(default: draft_vocab_32k.bin)")
    ap.add_argument("--grid", choices=("quick", "full"), default="quick", help="the settings tried: quick 7 (about 25 minutes at two rounds), full 18 (about 55)")
    ap.add_argument("--set", action="append", default=[], metavar="NAME=N[,NAME=N]", help="try this setting instead of a grid (repeatable)")
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("-d", "--draft", type=int, default=7)
    ap.add_argument("-n", type=int, default=512, help="tokens per prompt")
    ap.add_argument("-c", "--ctx", type=int, default=65536)
    ap.add_argument("--kv", default="q8")
    ap.add_argument("--long-chars", type=int, default=150000, help="size of the long prompt (about 50k tokens)")
    ap.add_argument("--long-file", default="", help="a text file as the long prompt, instead of the package's documentation")
    ap.add_argument("--no-long", action="store_true", help="only the short prompts")
    ap.add_argument("--profile", action="store_true", help="afterwards, one MK_PROFILE process per setting: phase tables in --out (not timed)")
    ap.add_argument("--out", default="l2_sweep_out")
    ap.add_argument("--timeout", type=float, default=1800, help="seconds per process")
    ap.add_argument("--dry-run", action="store_true", help="print the commands, run nothing")
    a = ap.parse_args()
    if not a.model or not os.path.isfile(a.model):
        ap.error("no model file (-m <gguf> or MK_MODEL)")
    if not os.path.isfile(a.exe):
        ap.error(f"no CLI at {a.exe} (--exe)")
    a.model, a.exe = os.path.abspath(a.model), os.path.abspath(a.exe)
    a.draft_vocab = os.path.abspath(a.draft_vocab) if a.draft_vocab else ""
    a.out = os.path.abspath(a.out)
    grid = a.set or (GRID_FULL if a.grid == "full" else GRID_QUICK)
    try:
        for s in grid:
            parse_setting(s)
    except ValueError as e:
        ap.error(str(e))
    os.makedirs(a.out, exist_ok=True)
    files = {"short": [], "long": ""}
    for i, p in enumerate(PROMPTS):
        files["short"].append(os.path.join(a.out, f"prompt{i}.txt"))
        with open(files["short"][-1], "w", encoding="utf-8", newline="\n") as f:
            f.write(p)
    if not a.no_long:
        files["long"] = os.path.abspath(a.long_file) if a.long_file else os.path.join(a.out, "prompt_long.txt")
        if not a.long_file:
            with open(files["long"], "w", encoding="utf-8", newline="\n") as f:
                f.write(long_text(a.long_chars))
    order = schedule(grid, a.rounds)
    info = {"gpu_before": gpu_info(), "model": os.path.basename(a.model), "draft": a.draft, "n": a.n, "ctx": a.ctx, "kv": a.kv,
            "env": {k: v for k, v in sorted(os.environ.items()) if k.startswith("MK_") and k != "MK_API_KEY"}}
    build = os.path.join(HERE, "BUILD_INFO.txt")
    if os.path.isfile(build):
        with open(build, encoding="utf-8-sig", errors="replace") as f:
            info["build"] = f.read()
    print(f"{len(order)} runs: {len(grid)} settings, {a.rounds} rounds. GPU: {info['gpu_before'] or 'unknown'}", flush=True)
    if not a.dry_run:
        run_process(a, {}, "warmup", files)  # (the session's first process runs cold, ~1.5% low: not counted)
    runs = []
    t0 = time.time()
    for k, (r, i) in enumerate(order):
        setting = parse_setting(grid[i]) if i is not None else {}
        got = run_process(a, setting, f"run{k:03d}", files)
        if got is None:
            continue
        got.update(setting=i, round=r)
        runs.append(got)
        with open(os.path.join(a.out, "results.json"), "w", encoding="utf-8") as f:
            json.dump({"info": info, "grid": grid, "runs": runs}, f, indent=1)
        print(f"[{k + 1}/{len(order)} {time.time() - t0:.0f} s] {grid[i] if i is not None else 'defaults'}: short {got['short']:.1f}"
              + (f", long {got['long']:.1f}, reading {got['prefill']:.0f}" if "long" in got else "") + " tok/s", flush=True)
    if a.dry_run:
        return 0
    info["gpu_after"] = gpu_info()
    rows, noise = analyse(runs, grid)
    report = f"GPU: {info['gpu_after']}\nmodel {info['model']}, {a.draft} drafts, --kv {a.kv}, greedy, {a.n} tokens per prompt"
    if "long_tokens" in runs[0]:
        report += f", long prompt {runs[0]['long_tokens']} tokens"
    report += f", {a.rounds} rounds\n\n" + table(rows, runs, noise)
    print("\n" + report)
    with open(os.path.join(a.out, "results.md"), "w", encoding="utf-8") as f:
        f.write(report + "\n")
    with open(os.path.join(a.out, "results.json"), "w", encoding="utf-8") as f:
        json.dump({"info": info, "grid": grid, "runs": runs, "rows": rows, "noise": noise}, f, indent=1)
    if a.profile:
        for k, s in enumerate([""] + grid):
            run_process(a, parse_setting(s), f"profile{k:02d}", files, profile=True)
        print(f"phase tables: {os.path.join(a.out, 'profile*.log')}")
    changed = [r["setting"] for r in rows if not r["ids_same"]]
    if changed:
        print("OUTPUT CHANGED under: " + "; ".join(changed) + "\nPrefetches must not change tokens: please report this.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
