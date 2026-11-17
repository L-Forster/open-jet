"""Score the question runs offline, every label in the given files with the same rules (those of the published runs).
usage: python -I bench/suite/score.py FILE [FILE ...] [--data DIR] [--labels L ...] [--exec-humaneval] [--timeout S]
                                      [--out scores.json]
  FILE             qa_client.py / maxctx_client.py output (records with a set and an id; a later file's answer to the
                   same label, set and id replaces an earlier one)
  --exec-humaneval run HumanEval: each completion is MODEL-WRITTEN CODE, executed on this machine (one fresh temporary
                   directory per task, python -I, --timeout seconds, default 15; no other isolation: use a VM or a
                   container if that is not acceptable). Without it HumanEval is reported as not scored
  --out            default: scores.json next to the first FILE
Rules: GSM8K the number after the last 'Answer:' (else the last number); MMLU-Pro the last 'answer is (X)' (else the
last '(X)'); ARC-C the first letter A-E / digit 1-5 of the reply; MATH-500 the last \\boxed{} against the reference
after light LaTeX normalisation (or numerically equal); HumanEval the task's own tests; NIAH the number anywhere in the
reply; LongBench v2 the first 'correct answer is (X)' (else the first '(X)').
Writes {labels, sets[set][label] = {n, correct}, timing[set][label], items[]} and prints the table."""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

SETS = ["gsm8k", "mmlu_pro", "arc_c", "math500", "math500_think", "humaneval", "niah", "longbench_v2", "niah_max",
        "longbench_max"]
NOT_SCORED = "not scored (needs --exec-humaneval)"


def num(s):
    s = s.replace(",", "").replace("$", "").strip().rstrip(".")
    try:
        return float(s)
    except ValueError:
        return None


def s_gsm8k(it, out):
    m = re.findall(r"Answer:\s*\$?\s*(-?[\d,]*\.?\d+)", out)
    if not m:
        m = re.findall(r"-?[\d,]*\.?\d+", out)
    v, g = (num(m[-1]) if m else None), num(it["answer"])
    return v is not None and g is not None and abs(v - g) < 1e-6


def s_mmlu_pro(it, out):
    m = re.findall(r"answer is \(?([A-J])\)?", out) or re.findall(r"\(([A-J])\)", out)
    return bool(m) and m[-1] == it["answer"]


def s_arc_c(it, out):
    m = re.search(r"[A-E1-5]", out.strip())
    return bool(m) and m.group(0) == it["answer"]


def boxed(out):
    i = out.rfind("\\boxed")
    if i < 0:
        return None
    j = out.find("{", i)
    if j < 0:
        return None
    depth, k = 0, j
    while k < len(out):
        if out[k] == "{":
            depth += 1
        elif out[k] == "}":
            depth -= 1
            if depth == 0:
                return out[j + 1:k]
        k += 1
    return None


def norm(a):
    a = a.strip().strip("$").strip()
    for x in ("\\left", "\\right", "\\!", "\\,", "\\;", "\\ ", "^\\circ", "^{\\circ}", "\\%", "%", "\\$", "dfrac", "tfrac"):
        a = a.replace(x, "frac" if x in ("dfrac", "tfrac") else "")
    a = re.sub(r"\\text\{([^}]*)\}", r"\1", a)
    a = re.sub(r"\\mbox\{([^}]*)\}", r"\1", a)
    a = a.replace(" ", "").rstrip(".")
    return a


def s_math(it, out):
    b = boxed(out)
    if b is None:
        return False
    x, g = norm(b), norm(it["answer"])
    if x == g:
        return True
    nx, ng = num(x), num(g)
    return nx is not None and ng is not None and abs(nx - ng) < 1e-6


def s_humaneval(it, out, timeout):
    """Model-written code: run only under --exec-humaneval, in a fresh temporary directory, isolated mode, a timeout."""
    m = re.findall(r"```(?:python)?\n(.*?)```", out, re.S)
    code = m[0] if m else out
    prog = (code if f"def {it['entry_point']}" in code else it["prompt"] + code) + "\n\n" + it["test"] + \
        f"\n\ncheck({it['entry_point']})\n"
    d = tempfile.mkdtemp(prefix="he_")
    try:
        p = os.path.join(d, "task.py")
        with open(p, "w", encoding="utf-8") as f:
            f.write(prog)
        r = subprocess.run([sys.executable, "-I", p], cwd=d, stdin=subprocess.DEVNULL, capture_output=True, timeout=timeout)
        return r.returncode == 0
    except subprocess.TimeoutExpired:
        return False
    finally:
        shutil.rmtree(d, ignore_errors=True)


def s_niah(it, out):
    return it["answer"] in out


def s_long(it, out):
    m = re.findall(r"correct answer is \(?([A-D])\)?", out) or re.findall(r"\(([A-D])\)", out)
    return bool(m) and m[0] == it["answer"]


SC = {"gsm8k": s_gsm8k, "mmlu_pro": s_mmlu_pro, "arc_c": s_arc_c, "math500": s_math, "math500_think": s_math,
      "niah": s_niah, "longbench_v2": s_long, "niah_max": s_niah, "longbench_max": s_long}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("files", nargs="+")
    ap.add_argument("--data", default=C.DATA)
    ap.add_argument("--labels", nargs="+")
    ap.add_argument("--exec-humaneval", action="store_true")
    ap.add_argument("--timeout", type=float, default=15.0)
    ap.add_argument("--out")
    a = ap.parse_args()

    recs = {}
    for path in a.files:
        with open(path, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                if not r.get("ok") or r.get("set") not in SETS or "id" not in r:
                    continue
                if a.labels and r["label"] not in a.labels:
                    continue
                recs[(r["label"], r["set"], r["id"])] = r
    man = C.load_manifest()
    items = {}
    for s in sorted({k[1] for k in recs}):
        for it in C.load_set(a.data, s, man):
            items[(s, it["id"])] = it
    res = {}
    for (lb, s, i), r in recs.items():
        it = items.get((s, i))
        if it is None:
            raise SystemExit(f"{lb} {s} {i}: not an item of the manifest")
        if s == "humaneval":
            res[(lb, s, i)] = s_humaneval(it, r["content"], a.timeout) if a.exec_humaneval else None
        else:
            res[(lb, s, i)] = SC[s](it, r["content"])

    labels = sorted({lb for lb, _, _ in res})
    out = {"labels": labels, "sets": {}, "timing": {}, "items": []}
    for s in SETS:
        for lb in labels:
            ks = [k for k in res if k[0] == lb and k[1] == s]
            if not ks:
                continue
            rs = [recs[k] for k in ks]
            pt, pms = sum(r.get("prompt_tokens") or 0 for r in rs), sum(r.get("prompt_ms") or 0 for r in rs)
            pn, dms = sum(r.get("pred_n") or 0 for r in rs), sum(r.get("pred_ms") or 0 for r in rs)
            scored = all(res[k] is not None for k in ks)
            e = {"n": len(ks), "correct": sum(res[k] for k in ks) if scored else None}
            if not scored:
                e["note"] = NOT_SCORED
            out["sets"].setdefault(s, {})[lb] = e
            out["timing"].setdefault(s, {})[lb] = {"wall_s": round(sum(r["wall_s"] for r in rs), 1),
                                                  "prefill_tok_s": round(pt / (pms / 1e3), 1) if pms else None,
                                                  "decode_tok_s": round(pn / (dms / 1e3), 1) if dms else None,
                                                  "truncated": sum(1 for r in rs if r.get("finish") == "length")}
    out["items"] = [{"label": lb, "set": s, "id": i, "correct": ok} for (lb, s, i), ok in sorted(res.items())]
    dest = a.out or os.path.join(os.path.dirname(os.path.abspath(a.files[0])), "scores.json")
    with open(dest, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)

    def cell(e):
        return f"{e['correct']}/{e['n']}" if e["correct"] is not None else f"-/{e['n']} (not run)"

    print("| set | " + " | ".join(labels) + " |")
    print("|---|" + "---|" * len(labels))
    for s in SETS:
        if s in out["sets"]:
            print(f"| {s} | " + " | ".join(cell(out["sets"][s][lb]) if lb in out["sets"][s] else "-" for lb in labels) + " |")
    if not a.exec_humaneval and "humaneval" in out["sets"]:
        print(f"humaneval: {NOT_SCORED}")
    print(f"scores -> {dest}")


if __name__ == "__main__":
    main()
