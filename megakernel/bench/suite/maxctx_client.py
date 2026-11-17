"""Maximum-context client of the published runs: one server at the largest context (262,144 tokens with preset D);
every long item is read from scratch (a short unrelated request first), greedy, no thinking.
usage: python -I bench/suite/maxctx_client.py --label L --out max.jsonl --ctx N [--url URL] [--data DIR] [--gpu N|none]
                                              [--limit N]
  speed : the longest NIAH haystack that fits, asked for a 512-token summary: prefill from scratch at ~the full context
          and decode at that depth; then a follow-up turn on the same chat (prompt reuse at the full context); VRAM after
  niah  : the niah_max items (4 needles, asks one; max_tokens 64): the ~190k pair and the pair at the largest length
          that fits (items of ~190k / ~248k / ~261k tokens)
  lb    : the longbench_max items that fit with 1,024 answer tokens: LongBench v2's own no-CoT prompt (as qa_client.py)
          but max_tokens 1,024 (the question client's 128 cut most long answers)
--ctx is the server's context; it decides which items fit. ~45 minutes at 262,144 on the reference machine.
--limit N asks only the first N needle and N LongBench items that fit (a quick check; not the published selection).
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
    C.add_common_args(ap)
    ap.add_argument("--ctx", type=int, required=True, help="the server's context size in tokens")
    ap.add_argument("--limit", type=int, help="only the first N needle and N LongBench items that fit (a quick check)")
    a = ap.parse_args()
    url, label, ctx, gpu = C.chat_url(a.url), a.label, a.ctx, C.Gpu(a.gpu)
    man = C.load_manifest()
    niah = C.load_set(a.data, "niah_max", man)
    lb = C.load_set(a.data, "longbench_max", man)
    out = open(a.out, "a", encoding="utf-8")

    def log(rec):
        out.write(json.dumps({"label": label, "ctx": ctx, **rec}) + "\n")
        out.flush()

    def req(test, messages, max_tokens, extra=None):
        body = {"messages": messages, "stream": False, "max_tokens": max_tokens, **C.GREEDY, **C.NO_THINK}
        rec = {"test": test, **(extra or {})}
        t0 = time.perf_counter()
        try:
            d, wall = C.post(url, body, 3600)
            msg = d["choices"][0]["message"]
            t, u = d.get("timings", {}), d.get("usage", {})
            pn, pms = t.get("predicted_n") or u.get("completion_tokens"), t.get("predicted_ms")
            rec.update(ok=True, wall_s=round(wall, 3), content=msg.get("content") or "",
                       finish=d["choices"][0].get("finish_reason"), prompt_tokens=u.get("prompt_tokens"),
                       cached=(u.get("prompt_tokens_details") or {}).get("cached_tokens"), prompt_ms=t.get("prompt_ms"),
                       pred_n=pn, pred_ms=pms, decode_tps=round(pn / (pms / 1e3), 2) if pn and pms else None)
        except (urllib.error.URLError, OSError, ValueError, KeyError, IndexError) as e:
            msg = {}
            rec.update(ok=False, wall_s=round(time.perf_counter() - t0, 3), error=str(e)[:300])
        log(rec)
        pt, pms = rec.get("prompt_tokens"), rec.get("prompt_ms")
        rate = f" = {pt / (pms / 1e3):.0f} tok/s" if pt and pms else ""
        print(f"{label:8s} {test:28s} {'ok ' if rec['ok'] else 'ERR'} prompt {pt} (cached {rec.get('cached')}) "
              f"{pms and round(pms)} ms{rate} | {rec.get('pred_n')} tok {rec.get('decode_tps')} tok/s | "
              f"wall {rec['wall_s']} s {rec.get('error', '')}", flush=True)
        return msg, rec

    def other():  # short unrelated request: the next prompt shares no prefix with the engine's cache
        req("reset_short", [{"role": "user", "content": "Say hi."}], 4)

    log({"test": "vram_idle", "mib": gpu.vram_mib()})
    req("warmup", [{"role": "user", "content": "Say hi."}], 8)
    req("warmup_clocks", [{"role": "user", "content": "Write a short story (about 300 words) about a lighthouse keeper."}], 512)

    fit_n = [it for it in niah if it["tokens"] + 64 + 64 <= ctx]
    if not fit_n:
        raise SystemExit(f"no NIAH item fits a {ctx}-token context (the shortest needs "
                         f"{min(it['tokens'] for it in niah) + 128})")
    top_len = max(it["length"] for it in fit_n)  # the 190k pair + the pair at the largest length that fits
    fit_n = [it for it in fit_n if it["length"] in (190_000, top_len)]
    fit_lb = [it for it in lb if it["approx_tokens"] + 1024 + 64 <= ctx]
    big = max(fit_n, key=lambda it: it["tokens"])
    print(f"== {label} ctx {ctx}: niah {len(fit_n)} / {len(niah)} fit, longbench {len(fit_lb)} / {len(lb)} fit "
          f"{time.strftime('%H:%M:%S')}", flush=True)
    if a.limit:
        fit_n, fit_lb = fit_n[:a.limit], fit_lb[:a.limit]
        print(f"   --limit {a.limit}: asking {len(fit_n)} needle and {len(fit_lb)} LongBench items", flush=True)

    # speed at the full context: the longest NIAH haystack, a long answer, then a follow-up turn
    text = big["prompt"].rsplit("\n\nWhat is the special magic number", 1)[0]
    m = [{"role": "user", "content": text + "\n\nSummarize the story told in the text above in detail."}]
    other()
    ans, _ = req("speed_max_turn1", m, 512, {"src": big["id"]})
    log({"test": "vram_after_max", "mib": gpu.vram_mib()})
    m += [{"role": "assistant", "content": ans.get("content") or ""},
          {"role": "user", "content": "Now name five characters from the text and one sentence about each."}]
    req("speed_max_turn2", m, 256)

    for it in fit_n:
        other()
        _, r = req(f"niah_{it['id']}", [{"role": "user", "content": it["prompt"]}], 64,
                   {"set": "niah_max", "id": it["id"], "answer": it["answer"], "tokens_est": it["tokens"]})
        hit = "HIT" if it["answer"] in (r.get("content") or "") else "MISS"
        print(f"   niah {it['id']}: answer {it['answer']} -> {hit}", flush=True)

    for it in fit_lb:
        other()
        req(f"lb_{it['id'][:12]}", [{"role": "user", "content": C.longbench_prompt(it)}], 1024,
            {"set": "longbench_max", "id": it["id"], "answer": it["answer"], "domain": it["domain"],
             "difficulty": it["difficulty"], "tokens_est": it["approx_tokens"]})
    log({"test": "vram_end", "mib": gpu.vram_mib()})
    print("MAXCTX_CLIENT_DONE", flush=True)


if __name__ == "__main__":
    main()
