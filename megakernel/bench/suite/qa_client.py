"""Question client of the published runs: the 282 questions, one server (OpenAI-compatible /v1/chat/completions).
usage: python -I bench/suite/qa_client.py --label L --out qa.jsonl [--url URL] [--data DIR] [--limit N] [SET ...]
  SET: gsm8k mmlu_pro arc_c math500 humaneval niah longbench_v2 math500_think (default: all, in this order)
  --limit N: only the first N items of each set (a quick check of a setup; not comparable with the published totals)
Greedy (temperature 0, top_k 1), zero-shot, thinking off except math500_think; the answer text, token counts and the
server's timings are recorded per item and scored offline (score.py). Items already answered under the same label in
--out are skipped, so an interrupted run resumes. Needs a server at 120,000 tokens of context (NIAH and LongBench v2
prompts reach ~100k tokens); ~28 minutes on the reference machine.
Item files come from fetch.py (untrusted downloads): their text only goes into prompts, nothing is executed here."""
import argparse
import json
import os
import sys
import time
import urllib.error

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    C.add_common_args(ap, gpu=False)
    ap.add_argument("--limit", type=int, help="only the first N items of each set (a quick check)")
    ap.add_argument("sets", nargs="*", help="sets to ask (default all)")
    a = ap.parse_args()
    names = [s for s, _ in C.QA_SETS]
    bad = [s for s in a.sets if s not in names]
    if bad:
        raise SystemExit(f"unknown set(s) {bad}; sets: {' '.join(names)}")
    want = a.sets or names
    url = C.chat_url(a.url)
    man = C.load_manifest()
    sets = [(name, fn, C.load_set(a.data, name, man)) for name, fn in C.QA_SETS if name in want]
    done = set()
    if os.path.exists(a.out):  # resume: skip items this label already answered
        with open(a.out, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                    if r.get("label") == a.label and r.get("ok"):
                        done.add((r["set"], r["id"]))
                except (ValueError, KeyError):
                    pass
    out = open(a.out, "a", encoding="utf-8")
    for name, fn, items in sets:
        items = items[:a.limit] if a.limit else items
        print(f"== {a.label} {name}: {len(items)} items {time.strftime('%H:%M:%S')}", flush=True)
        for it in items:
            if (name, it["id"]) in done:
                continue
            prompt, max_tokens, think = fn(it)
            body = {"messages": [{"role": "user", "content": prompt}], "stream": False, "max_tokens": max_tokens, **C.GREEDY}
            if not think:
                body.update(C.NO_THINK)
            rec = {"label": a.label, "set": name, "id": it["id"]}
            t0 = time.perf_counter()
            try:
                d, wall = C.post(url, body, 3600)
                msg = d["choices"][0]["message"]
                t, u = d.get("timings", {}), d.get("usage", {})
                rec.update(ok=True, wall_s=round(wall, 3), content=msg.get("content") or "",
                           reasoning_len=len(msg.get("reasoning_content") or ""), finish=d["choices"][0].get("finish_reason"),
                           prompt_tokens=u.get("prompt_tokens"), completion_tokens=u.get("completion_tokens"),
                           prompt_ms=t.get("prompt_ms"), pred_n=t.get("predicted_n"), pred_ms=t.get("predicted_ms"))
            except (urllib.error.URLError, OSError, ValueError, KeyError, IndexError) as e:
                rec.update(ok=False, wall_s=round(time.perf_counter() - t0, 3), error=str(e)[:300])
            out.write(json.dumps(rec) + "\n")
            out.flush()
            print(f"{a.label:6s} {name:13s} {it['id'][:28]:28s} {'ok ' if rec['ok'] else 'ERR'} {rec.get('prompt_tokens')} -> "
                  f"{rec.get('completion_tokens')} tok, {rec['wall_s']} s", flush=True)
    print("QA_CLIENT_DONE", flush=True)


if __name__ == "__main__":
    main()
