"""Tables of the published runs from the clients' JSON lines and score.py's scores.
usage: python bench/suite/summarize.py FILE [FILE ...] [--scores scores.json ...] [--col LABEL[=TITLE] ...]
                                       [--sub LABEL:KEY[,KEY...]=OTHER ...] [--out results.json]
  FILE    speed_client.py / scratch_client.py / maxctx_client.py / parallel_client.py output. Each file is one session:
          a label found in several files gets the best session's value in each row (the published llama.cpp column is
          the best of three sessions); give every run its own label unless that is what you want
  --col   the columns, in order (default: every label with short-prompt decode records)
  --sub   take some rows of LABEL's column from another label, e.g. a re-run of a disturbed step. KEY: a row key below,
          qa (all question sets) or qa.SET (one set: its answers and its time)
The from-scratch prefill row prefers scratch_client.py's record of the label (any file), then the speed client's own
"fresh" request when the server reports no cached tokens for it, then a long_pf101k request without cached tokens.
Row keys: greedy greedy_tw sampled sampled_tw dec11k dec38k dec101k ttft turn2 pf11k pf38k pf101k edit_msg edit_sys
vram_idle vram_101k. parallel_client.py's records make a table of their own (per label and request count, the best
session). Prints markdown tables; --out writes them as JSON."""
import argparse
import json
import statistics as st


def load(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def recs(src, lb, test, exact=True):
    return [r for r in src if r.get("label") == lb and (r.get("test") == test if exact else str(r.get("test")).startswith(test))]


def mean(src, lb, prefix, key, scale=1.0):  # mean of the per-request values ("fresh" requests excluded)
    v = [r[key] * scale for r in recs(src, lb, prefix, False) if r.get(key) is not None and not r["test"].endswith("_fresh")]
    return round(st.mean(v), 2) if v else None


def tokw(src, lb, prefix):  # token-weighted decode rate (total tokens / total decode time)
    rs = [r for r in recs(src, lb, prefix, False) if r.get("pred_n") and r.get("pred_ms")]
    return round(sum(r["pred_n"] for r in rs) / (sum(r["pred_ms"] for r in rs) / 1e3), 2) if rs else None


def one(src, lb, test, key, scale=1.0):
    rs = recs(src, lb, test)
    return round(rs[0][key] * scale, 3) if rs and rs[0].get(key) is not None else None


def pf_fresh(src, lb):
    rs = [r for r in recs(src, lb, "long_pf101k_fresh") if not r.get("cached")] or \
        [r for r in recs(src, lb, "long_pf101k") if not r.get("cached")]
    return round(rs[0]["prompt_ms"] / 1e3, 2) if rs and rs[0].get("prompt_ms") else None


def gb(src, lb, test):
    rs = recs(src, lb, test)
    return round(rs[0]["mib"] / 1024, 2) if rs and rs[0].get("mib") is not None else None


METRICS = [
    ("greedy", "greedy decode, short prompts (mean of 4)", "tok/s", "up", lambda s, l: mean(s, l, "greedy_nothink", "decode_tps")),
    ("greedy_tw", "greedy decode, short prompts, token-weighted", "tok/s", "up", lambda s, l: tokw(s, l, "greedy_nothink")),
    ("sampled", "sampled decode, thinking on (mean of 8)", "tok/s", "up", lambda s, l: mean(s, l, "sampled_think", "decode_tps")),
    ("sampled_tw", "sampled decode, thinking on, token-weighted", "tok/s", "up", lambda s, l: tokw(s, l, "sampled_think")),
    ("dec11k", "decode at 11.6k context", "tok/s", "up", lambda s, l: mean(s, l, "followup_turn1", "decode_tps")),
    ("dec38k", "decode at 38k context", "tok/s", "up", lambda s, l: mean(s, l, "long_pf38k", "decode_tps")),
    ("dec101k", "decode at 101.6k context", "tok/s", "up", lambda s, l: mean(s, l, "long_pf101k", "decode_tps")),
    ("ttft", "time to first token, short prompt", "ms", "down", lambda s, l: mean(s, l, "ttft_short", "prompt_ms")),
    ("turn2", "follow-up turn, prompt time (11.8k chat)", "ms", "down", lambda s, l: one(s, l, "followup_turn2", "prompt_ms")),
    ("pf11k", "prefill 11.6k tokens", "s", "down", lambda s, l: one(s, l, "followup_turn1", "prompt_ms", 1e-3)),
    ("pf38k", "prefill 38k tokens", "s", "down", lambda s, l: one(s, l, "long_pf38k", "prompt_ms", 1e-3)),
    ("pf101k", "prefill 101.6k tokens, from scratch", "s", "down", None),
    ("edit_msg", "edit an earlier message (11.7k-token chat)", "ms", "down", lambda s, l: one(s, l, "edit_turn2_edited", "prompt_ms")),
    ("edit_sys", "edit the end of a 9.1k-token system prompt", "ms", "down", lambda s, l: one(s, l, "edit_system_edited", "prompt_ms")),
    ("vram_idle", "VRAM, server idle (whole card)", "GB", "down", lambda s, l: gb(s, l, "vram_idle")),
    ("vram_101k", "VRAM after the 101.6k prompt (whole card)", "GB", "down", lambda s, l: gb(s, l, "vram_after_pf101k")),
]


def best(vals, better):
    v = [x for x in vals if x is not None]
    return (max(v) if better == "up" else min(v)) if v else None


PARALLEL = [  # (key, metric, unit, better, test, field)
    ("short", "short prompts, total decode", "tok/s", "up", "short_total", "tps"),
    ("ctx", "23k-token contexts, total decode", "tok/s", "up", "ctx_total", "tps"),
    ("ctx_ttft", "23k-token contexts, slowest first token", "s", "down", "ctx_total", "ttft_max_s"),
    ("stagger_ttft", "stagger: the 20k-token prompt's first token", "s", "down", "stagger", "ttft_s"),
    ("stagger_other", "stagger: the other request meanwhile", "tok/s", "up", "stagger", "other_tps"),
]


def parallel(sessions, labels):
    """parallel_client.py's records: {(key, n): {label: best session's value}} (a session's value: its best rep)"""
    out = {}
    for key, _, _, better, test, field in PARALLEL:
        for s in sessions:
            for r in s:
                if r.get("test") == test and r.get("label") in labels and r.get(field) is not None:
                    cell = out.setdefault((key, r.get("n")), {})
                    cell[r["label"]] = best([cell.get(r["label"]), r[field]], better)
    return out


def metric(sessions, lb, key, fn, better):
    if key == "pf101k":
        scratch = [one(s, lb, "long_pf101k_scratch", "prompt_ms", 1e-3) for s in sessions]
        if any(x is not None for x in scratch):
            return best(scratch, better)
        return best([pf_fresh(s, lb) for s in sessions], better)
    return best([fn(s, lb) for s in sessions], better)


def qa_total(scores, lb, sub):
    """Questions of the 120k sets: right / n over the scored sets, total wall time (layout of score.py's output)."""
    sets = {}
    for s, v in scores.get("sets", {}).items():
        if s.endswith("_max"):
            continue
        src = sub.get(f"qa.{s}") or sub.get("qa") or lb
        if src in v:
            sets[s] = (src, v[src])
    if not sets:
        return None
    scored = [e for _, e in sets.values() if e.get("correct") is not None]
    wall = sum(scores.get("timing", {}).get(s, {}).get(src, {}).get("wall_s", 0) for s, (src, _) in sets.items())
    out = {"right": sum(e["correct"] for e in scored), "n": sum(e["n"] for e in scored), "wall_s": round(wall, 1),
           "sets": {s: (f"{e['correct']}/{e['n']}" if e.get("correct") is not None else f"not scored ({e['n']})")
                    for s, (_, e) in sets.items()}}
    if any(src != lb for src, _ in sets.values()):
        out["from"] = {s: src for s, (src, _) in sets.items() if src != lb}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("files", nargs="+")
    ap.add_argument("--scores", nargs="+", default=[])
    ap.add_argument("--col", action="append", default=[])
    ap.add_argument("--sub", action="append", default=[])
    ap.add_argument("--out")
    a = ap.parse_args()
    sessions = [load(p) for p in a.files]
    cols = []
    for c in a.col:
        lb, _, title = c.partition("=")
        cols.append((lb, title or lb))
    par_labels = []
    for s in sessions:
        for r in s:
            if r.get("test") in ("short_total", "ctx_total", "stagger") and r.get("label") not in par_labels:
                par_labels.append(r["label"])
    par_cols = [(lb, t) for lb, t in cols if lb in par_labels] if cols else [(lb, lb) for lb in par_labels]
    if not cols:
        seen = []
        for s in sessions:
            for r in s:
                if str(r.get("test", "")).startswith("greedy_nothink") and r.get("label") not in seen:
                    seen.append(r["label"])
        cols = [(lb, lb) for lb in seen]
    subs = {}
    for x in a.sub:
        try:
            left, other = x.split("=", 1)
            lb, keys = left.split(":", 1)
        except ValueError:
            raise SystemExit(f"--sub {x!r}: expected LABEL:KEY[,KEY...]=OTHER")
        for k in keys.split(","):
            subs.setdefault(lb, {})[k.strip()] = other
    scores = {"sets": {}, "timing": {}}
    for p in a.scores:  # several score files: merged, a later file's label wins
        with open(p, encoding="utf-8") as f:
            sc = json.load(f)
        for part in ("sets", "timing"):
            for s, v in sc.get(part, {}).items():
                scores[part].setdefault(s, {}).update(v)

    rows = []
    for key, label, unit, better, fn in METRICS:
        row = {"key": key, "metric": label, "unit": unit}
        for lb, _ in cols:
            row[lb] = metric(sessions, subs.get(lb, {}).get(key, lb), key, fn, better)
        rows.append(row)
    qa = {lb: qa_total(scores, lb, subs.get(lb, {})) for lb, _ in cols}

    maxctx = {}
    for s in sessions:
        for lb in {r.get("label") for r in s if r.get("test") == "speed_max_turn1"}:
            rs = [r for r in s if r.get("label") == lb]
            g = lambda t: next((r for r in rs if r.get("test") == t), None)  # noqa: E731
            t1, t2, vr = g("speed_max_turn1"), g("speed_max_turn2"), g("vram_after_max")
            mx = {"ctx": rs[0].get("ctx"), "prefill_tokens": t1.get("prompt_tokens"),
                  "prefill_s": t1.get("prompt_ms") and round(t1["prompt_ms"] / 1e3, 1), "decode_tok_s": t1.get("decode_tps"),
                  "turn2_prompt_ms": t2 and t2.get("prompt_ms"),
                  "vram_after_max_gb": round(vr["mib"] / 1024, 2) if vr and vr.get("mib") is not None else None}
            for s_name in ("niah_max", "longbench_max"):
                e = scores["sets"].get(s_name, {}).get(lb)
                if e:
                    mx[s_name] = f"{e['correct']}/{e['n']}"
            maxctx[lb] = mx  # a later file's run of the same label replaces an earlier one
    gpu = [r for s in sessions for r in s if str(r.get("test", "")).startswith("gpu_") and r.get("label") in dict(cols)]
    par = parallel(sessions, {lb for lb, _ in par_cols})
    par_rows = []
    for n in sorted({n for _, n in par if n is not None}, reverse=True):
        for key, label, unit, _, _, _ in PARALLEL:
            if (key, n) in par:
                name = label if key.startswith("stagger") else f"{label}, {n} at once"
                par_rows.append({"key": key, "n": n, "metric": name, "unit": unit,
                                 **{lb: par[(key, n)].get(lb) for lb, _ in par_cols}})
    res = {"columns": [{"label": lb, "title": t} for lb, t in cols], "subs": subs, "rows": rows, "qa": qa,
           "maxctx": maxctx, "gpu": gpu,
           "parallel": {"columns": [{"label": lb, "title": t} for lb, t in par_cols], "rows": par_rows}}
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=1)

    def fmt(x):
        return "-" if x is None else str(x)

    if any(r[lb] is not None for r in rows for lb, _ in cols) or any(qa.values()):
        print("| metric | " + " | ".join(t for _, t in cols) + " |")
        print("|---|" + "---|" * len(cols))
        for r in rows:
            print(f"| {r['metric']} ({r['unit']}) | " + " | ".join(fmt(r[lb]) for lb, _ in cols) + " |")
        print("| questions: right / scored, total time (s) | " +
              " | ".join(f"{q['right']} / {q['n']}, {q['wall_s']}" if q else "-" for q in (qa[lb] for lb, _ in cols)) + " |")
    if par_rows:
        print("")
        print("| parallel agents | " + " | ".join(t for _, t in par_cols) + " |")
        print("|---|" + "---|" * len(par_cols))
        for r in par_rows:
            print(f"| {r['metric']} ({r['unit']}) | " + " | ".join(fmt(r[lb]) for lb, _ in par_cols) + " |")
    for lb, _ in cols:
        if qa[lb]:
            print(f"questions {lb}: {qa[lb]['sets']}" + (f", from {qa[lb]['from']}" if qa[lb].get("from") else ""))
    for lb, mx in maxctx.items():
        print(f"max context {lb}: {mx}")
    if a.out:
        print(f"results -> {a.out}")


if __name__ == "__main__":
    main()
