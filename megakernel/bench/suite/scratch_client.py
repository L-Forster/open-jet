"""From-scratch prefill of the ~101.6k-token prompt (the "read a 101.6k-token prompt from scratch" row).
usage: python -I bench/suite/scratch_client.py --label L --out scratch.jsonl [--url URL] [--data DIR]
A warm-up, ~3 s of decode, a short unrelated request, then speed_client.py's pf101k request (the same messages, 16
tokens). On this engine's server, start it with side requests and parking off (MK_SIDE_MAX=0 MK_PARK_MIB=0, or
--side-max 0 --park-mib 0): otherwise the short request runs beside the cached conversation and keeps it, and the long
prompt is served from the cache. Use the same --label as the speed run of that configuration: summarize.py takes the
row from here. Needs a server at 120,000 tokens of context and pf101k.txt from fetch.py."""
import argparse
import json
import os
import sys
import urllib.error

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    C.add_common_args(ap, gpu=False)
    a = ap.parse_args()
    url, label = C.chat_url(a.url), a.label
    doc = C.load_prompt(a.data, "pf101k")
    out = open(a.out, "a", encoding="utf-8")

    def req(test, messages, **kw):
        try:
            d, wall = C.post(url, {"messages": messages, "stream": False, **kw}, 1800)
        except (urllib.error.URLError, OSError) as e:
            raise SystemExit(f"{test}: request failed: {e}")
        t, u = d.get("timings", {}), d.get("usage", {})
        rec = {"label": label, "test": test, "wall_s": round(wall, 3), "prompt_tokens": u.get("prompt_tokens"),
               "cached": (u.get("prompt_tokens_details") or {}).get("cached_tokens"), "prompt_ms": t.get("prompt_ms"),
               "pred_n": t.get("predicted_n"), "pred_ms": t.get("predicted_ms")}
        out.write(json.dumps(rec) + "\n")
        out.flush()
        print(f"{label} {test:18s} prompt {rec['prompt_tokens']} (cached {rec['cached']}) "
              f"{rec['prompt_ms'] and round(rec['prompt_ms'])} ms", flush=True)
        return rec

    req("warmup", [{"role": "user", "content": "Say hi."}], max_tokens=8, **C.GREEDY, **C.NO_THINK)
    req("warmup_clocks", [{"role": "user", "content": "Write a short story (about 300 words) about a lighthouse keeper who "
                                                       "discovers a message in a bottle."}],
        max_tokens=512, **C.GREEDY, **C.NO_THINK)
    req("reset_short", [{"role": "user", "content": "Say hi."}], max_tokens=4, **C.GREEDY, **C.NO_THINK)
    rec = req("long_pf101k_scratch", [{"role": "user", "content": doc + "\n\nWhat is the main purpose of the code above? "
                                                                       "Answer in detail."}],
              max_tokens=16, **C.GREEDY, **C.NO_THINK)
    if rec.get("cached"):
        print(f"warning: {rec['cached']} of the prompt's tokens came from the cache: not a from-scratch prefill "
              "(side requests or parking on?)", flush=True)
    print("SCRATCH_DONE", flush=True)


if __name__ == "__main__":
    main()
