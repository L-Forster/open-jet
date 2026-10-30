#!/usr/bin/env python3
"""Check the `openjet bench` results in benchmarks/community/ and rebuild the README's community table.

  python scripts/bench_table.py           # validate every file, then rewrite the table in README.md
  python scripts/bench_table.py --check   # validate only; exits 1 on a bad file (CI runs this on pull requests)

Runs on one GPU, model, engine and OS are grouped into one row, which shows the median of each figure.
Needs only the standard library.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "benchmarks" / "community"
README = ROOT / "README.md"
START, END = "<!-- community-results:start -->", "<!-- community-results:end -->"
SCHEMA = "openjet-bench/1"
DEPTHS = (4096, 32768, 98304)
MAX_BYTES = 256 * 1024
HANDLE = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,37}[A-Za-z0-9])?")


def _num(value, low: float, high: float) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and low < value < high


def validate(record: dict) -> list[str]:
    """Problems with one results file; empty when it can go in the table."""
    errors = []
    if record.get("schema") != SCHEMA:
        return [f"schema is {record.get('schema')!r}, expected {SCHEMA!r}; rerun with the current `openjet bench`"]
    contributor = record.get("contributor")
    if contributor is not None and not (isinstance(contributor, str) and HANDLE.fullmatch(contributor)):
        errors.append("contributor must be a GitHub handle or null")
    gpu = (record.get("system") or {}).get("gpu") or {}
    if not isinstance(gpu.get("name"), str) or not 0 < len(gpu["name"]) <= 100:
        errors.append("system.gpu.name is missing")
    if not str((record.get("model") or {}).get("file", "")).endswith(".gguf"):
        errors.append("model.file must be a .gguf file name")
    if (record.get("runtime") or {}).get("engine") not in ("openjet", "llama.cpp"):
        errors.append("runtime.engine must be openjet or llama.cpp")
    results = record.get("results") or {}
    for name, count in (("reasoning", 4), ("greedy", 4)):
        part = results.get(name) or {}
        runs = part.get("runs") or []
        if len(runs) != count or not all(_num(r.get("tokens"), 0, 1e6) and _num(r.get("ms"), 0, 1e9) for r in runs):
            errors.append(f"results.{name} needs {count} runs with tokens and ms")
            continue
        tok_s = sum(r["tokens"] for r in runs) / sum(r["ms"] for r in runs) * 1e3
        if not _num(part.get("tok_s"), 0, 2000) or abs(part["tok_s"] - tok_s) > max(0.01 * tok_s, 0.02):
            errors.append(f"results.{name}.tok_s does not match its runs")
    for d in results.get("depth") or []:
        ok = (d.get("depth") in DEPTHS and _num(d.get("prompt_tokens"), 0, 1e6) and _num(d.get("prompt_ms"), 0, 1e9)
              and _num(d.get("prefill_tok_s"), 0, 1e6) and _num(d.get("decode_tok_s"), 0, 2000)
              and isinstance(d.get("recall"), bool))
        if not ok:
            errors.append(f"results.depth entry {d.get('depth')!r} is incomplete")
        elif abs(d["prefill_tok_s"] - d["prompt_tokens"] / d["prompt_ms"] * 1e3) > 0.01 * d["prefill_tok_s"] + 0.02:
            errors.append(f"results.depth {d['depth']} prefill_tok_s does not match prompt_tokens and prompt_ms")
    if len(str(record.get("notes") or "")) > 500:
        errors.append("notes are longer than 500 characters")
    return errors


def load(directory: Path = RESULTS) -> tuple[list[dict], list[str]]:
    records, errors = [], []
    for path in sorted(directory.glob("*.json")):
        rel = path.relative_to(ROOT) if path.is_relative_to(ROOT) else path
        if path.stat().st_size > MAX_BYTES:
            errors.append(f"{rel}: larger than {MAX_BYTES // 1024} KB")
            continue
        try:
            record = json.loads(path.read_text())
        except (ValueError, UnicodeDecodeError) as exc:
            errors.append(f"{rel}: not valid JSON ({exc})")
            continue
        problems = validate(record) if isinstance(record, dict) else ["not a JSON object"]
        errors.extend(f"{rel}: {p}" for p in problems)
        if not problems:
            records.append(record)
    return records, errors


def _gpu(record: dict) -> str:
    gpu = record["system"]["gpu"]
    name = re.sub(r"^NVIDIA\s+(GeForce\s+)?", "", gpu["name"]).strip()
    if gpu.get("vram_mb"):
        name += f" {round(gpu['vram_mb'] / 1024)} GB"
    elif gpu.get("unified_memory_gb"):
        name += f" {gpu['unified_memory_gb']} GB"
    return name


def _os(record: dict) -> str:
    os_name = str(record["system"].get("os", ""))
    for family in ("WSL2", "Windows", "macOS"):
        if family in os_name:
            return family
    return "Linux" if os_name else "?"


def _engine(record: dict) -> str:
    runtime = record["runtime"]
    if runtime["engine"] == "openjet":
        return "OpenJet (mixed quant)" if "mixed" in str(runtime.get("variant", "")).lower() else "OpenJet"
    return f"llama.cpp ({runtime['device']})" if runtime.get("device") not in (None, "auto") else "llama.cpp"


def _cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def _median(values: list[float], digits: int = 0) -> str:
    if not values:
        return "–"
    value = statistics.median(values)
    return f"{value:.{digits}f}"


def rows(records: list[dict]) -> list[list[str]]:
    groups: dict[tuple, list[dict]] = {}
    for r in records:
        groups.setdefault((_gpu(r), r["model"]["file"].removesuffix(".gguf"), _engine(r), _os(r)), []).append(r)
    table = []
    for (gpu, model, engine, os_family), runs in groups.items():
        depth = {d: [e for r in runs for e in r["results"].get("depth", []) if e["depth"] == d] for d in DEPTHS}
        checks = [e["recall"] for entries in depth.values() for e in entries]
        names = sorted({r["contributor"] for r in runs if r.get("contributor")}, key=str.lower)
        credit = ", ".join(f"[{n}](https://github.com/{n})" for n in names[:3])
        if len(names) > 3:
            credit += f" +{len(names) - 3}"
        reasoning = [r["results"]["reasoning"]["tok_s"] for r in runs]
        table.append([
            _cell(gpu), _cell(model), engine, os_family,
            _median(reasoning, 1), _median([r["results"]["greedy"]["tok_s"] for r in runs], 1),
            _median([e["prefill_tok_s"] for e in depth[32768]]), _median([e["decode_tok_s"] for e in depth[32768]], 1),
            f"{sum(checks)}/{len(checks)}" if checks else "–", str(len(runs)), credit or "–",
            statistics.median(reasoning),
        ])
    table.sort(key=lambda row: -row[-1])
    return [row[:-1] for row in table]


HEADER = ["GPU", "Model", "Engine", "OS", "Reasoning tok/s", "Greedy tok/s", "Prompt tok/s at 32K",
          "Decode tok/s at 32K", "Recall", "Runs", "Sent by"]


def render(records: list[dict]) -> str:
    if not records:
        return "No results yet. Run `openjet bench` and send yours; the first row could be your card.\n"
    lines = ["| " + " | ".join(HEADER) + " |", "|" + "---|" * 4 + "---:|" * 6 + "---|"]
    lines += ["| " + " | ".join(row) + " |" for row in rows(records)]
    return "\n".join(lines) + "\n"


def update_readme(table: str, readme: Path = README) -> bool:
    text = readme.read_text()
    start, end = text.index(START) + len(START), text.index(END)
    new = text[:start] + "\n" + table + text[end:]
    if new == text:
        return False
    readme.write_text(new)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="validate only; do not touch README.md")
    args = parser.parse_args()
    records, errors = load()
    for error in errors:
        print(f"error: {error}", file=sys.stderr)
    table = render(records)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a") as f:
            f.write(f"### Community table with this pull request\n\n{table}\n")
    if errors:
        return 1
    if args.check:
        print(f"{len(records)} results files are valid.")
        return 0
    print("README.md updated." if update_readme(table) else "README.md is already up to date.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
