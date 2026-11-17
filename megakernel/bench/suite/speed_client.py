"""Speed client of the published runs: decode, prefill, first token, follow-up turns and prompt edits, one server.
usage: python -I bench/suite/speed_client.py --label L --out speed.jsonl [--url URL] [--mode all|edit] [--data DIR]
                                             [--gpu N|none]
  all : every test, in the order of the 2026-10-20 run (~3 minutes)
  edit: the warm-up requests + the prompt-edit scenarios only (to re-run those rows)
Requests (non-streaming /v1/chat/completions): a warm-up, ~3 s of decode so the timed requests start at full clocks,
three one-word questions (max_tokens 1), four short prompts greedy without thinking (512 tokens), the same four sampled
with thinking (temperature 1.0, top_p 0.95, top_k 20, seeds 11 and 22, 768 tokens), an 11.6k-token summary request
(greedy, thinking on, 200 tokens) and its follow-up turn, pf38k and pf101k (256 tokens, no thinking), then the edit
scenarios. Times come from the server's `timings` (prompt_ms, predicted_n, predicted_ms) and `usage` (cached tokens);
VRAM and GPU clocks from nvidia-smi on this machine when present. One JSON line per request in --out.
Needs a server at 120,000 tokens of context (pf101k is ~101.6k tokens) and pf101k.txt from fetch.py."""
import argparse
import json
import os
import sys
import urllib.error

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

PROMPTS = [
    "Write a Python function that checks whether a number is prime, with a docstring and a few doctests.",
    "Explain how a hash map works internally, including collision handling and resizing, for a junior developer.",
    "Write a short story (about 300 words) about a lighthouse keeper who discovers a message in a bottle.",
    "Give me a step-by-step plan for migrating a Flask REST API to FastAPI, including testing and deployment.",
]
NO_THINK, GREEDY = C.NO_THINK, C.GREEDY


class Client:
    def __init__(self, a):
        self.url, self.label, self.gpu = C.chat_url(a.url), a.label, C.Gpu(a.gpu)
        self.out = open(a.out, "a", encoding="utf-8")

    def write(self, rec):
        self.out.write(json.dumps(rec) + "\n")
        self.out.flush()

    def vram(self, test):
        mib = self.gpu.vram_mib()
        self.write({"label": self.label, "test": test, "mib": mib})
        return mib

    def gpu_state(self, tag):  # cooling moves whole sessions by several percent: record it with every run
        v = self.gpu.query("temperature.gpu,clocks.sm,clocks.mem,fan.speed,power.draw")
        if v is None:
            print(f"== {self.label}: GPU {tag}: no nvidia-smi reading", flush=True)
            return
        t, sm, mem, fan, pw = v
        self.write({"label": self.label, "test": f"gpu_{tag}", "temp_c": t, "sm_mhz": sm, "mem_mhz": mem, "fan_pct": fan,
                    "power_w": pw})
        print(f"== {self.label}: GPU {tag}: {t} C, SM {sm} MHz, mem {mem} MHz, fan {fan}%, {pw} W", flush=True)

    def req(self, test, messages, **kw):
        body = {"messages": messages, "stream": False, **kw}
        try:
            d, wall = C.post(self.url, body, 1800)
        except (urllib.error.URLError, OSError) as e:
            raise SystemExit(f"{test}: request failed: {e}")
        t = d.get("timings", {})
        u = d.get("usage", {})
        msg = d["choices"][0]["message"]
        pn, pms = t.get("predicted_n") or u.get("completion_tokens"), t.get("predicted_ms")
        rec = {"label": self.label, "test": test, "wall_s": round(wall, 3),
               "prompt_tokens": u.get("prompt_tokens"), "cached": (u.get("prompt_tokens_details") or {}).get("cached_tokens"),
               "prompt_ms": t.get("prompt_ms"), "pred_n": pn, "pred_ms": pms,
               "decode_tps": round(pn / (pms / 1e3), 2) if pn and pms else None,
               "draft_n": t.get("draft_n"), "draft_acc": t.get("draft_n_accepted"),
               "finish": d["choices"][0].get("finish_reason")}
        self.write(rec)
        print(f"{self.label:9s} {test:22s} prompt {rec['prompt_tokens']} (cached {rec['cached']}) "
              f"{rec['prompt_ms'] and round(rec['prompt_ms'])} ms | {pn} tok {rec['decode_tps']} tok/s | wall {rec['wall_s']} s",
              flush=True)
        return msg, rec

    def other(self):  # short unrelated request: the next prompt shares no prefix with the engine's cache
        self.req("reset_short", [{"role": "user", "content": "Say hi."}], max_tokens=4, **GREEDY, **NO_THINK)


def edit(c, pf38k):
    """Prompt reuse when the conversation changes behind the last turn."""
    doc = pf38k[:30000]
    kw = dict(max_tokens=96, **GREEDY, **NO_THINK)
    c.other()
    m = [{"role": "user", "content": doc + "\n\nSummarize the text above in three bullet points."}]
    a, _ = c.req("edit_turn1", m, **kw)
    m += [{"role": "assistant", "content": a.get("content") or ""}, {"role": "user", "content": "List three risks the text mentions."}]
    a, _ = c.req("edit_turn2", m, **kw)
    m += [{"role": "assistant", "content": a.get("content") or ""}, {"role": "user", "content": "Which one is the worst, and why?"}]
    c.req("edit_turn3", m, **kw)
    m2 = m[:3] + [{"role": "user", "content": "List three benefits the text mentions."}]  # turn 2's user message edited
    c.req("edit_turn2_edited", m2, **kw)
    c.other()
    c.req("edit_turn2_edited_fresh", m2, cache_prompt=False, **kw)  # the same prompt without reuse
    sysmsg = doc[:24000] + "\n\nYou are a helpful assistant. Today is Monday."
    ms = [{"role": "system", "content": sysmsg}, {"role": "user", "content": "What day is it, and what is the text about?"}]
    c.req("edit_system", ms, **kw)
    ms2 = [{"role": "system", "content": sysmsg.replace("Monday", "Tuesday")}, ms[1]]
    c.req("edit_system_edited", ms2, **kw)
    c.other()
    c.req("edit_system_edited_fresh", ms2, cache_prompt=False, **kw)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    C.add_common_args(ap)
    ap.add_argument("--mode", choices=["all", "edit"], default="all")
    a = ap.parse_args()
    man = C.load_manifest()
    pf38k = C.load_prompt(a.data, "pf38k", man)
    pf101k = C.load_prompt(a.data, "pf101k", man) if a.mode == "all" else None
    c = Client(a)
    print(f"== {c.label}: idle VRAM {c.vram('vram_idle')} MiB", flush=True)
    c.req("warmup", [{"role": "user", "content": "Say hi."}], max_tokens=8, **GREEDY, **NO_THINK)
    # a card that idles at low clocks while the server loads would slow the first timed requests: ~3 s of decode first
    c.req("warmup_clocks", [{"role": "user", "content": PROMPTS[2]}], max_tokens=512, **GREEDY, **NO_THINK)
    c.gpu_state("start")
    if a.mode == "all":
        for i, q in enumerate(["What is the capital of Australia? One word.", "Name a prime number above 100. Just the number.",
                               "Translate 'good morning' into Spanish. Only the translation."]):
            c.req(f"ttft_short_{i}", [{"role": "user", "content": q}], max_tokens=1, **GREEDY, **NO_THINK)
        for i, p in enumerate(PROMPTS):
            c.req(f"greedy_nothink_p{i}", [{"role": "user", "content": p}], max_tokens=512, **GREEDY, **NO_THINK)
        for s in (11, 22):
            for i, p in enumerate(PROMPTS):
                c.req(f"sampled_think_p{i}_s{s}", [{"role": "user", "content": p}], max_tokens=768,
                      temperature=1.0, top_p=0.95, top_k=20, seed=s)
        m = [{"role": "user", "content": pf38k[:30000] + "\n\nSummarize the text above in three bullet points."}]
        r1, _ = c.req("followup_turn1", m, max_tokens=200, **GREEDY)
        m += [{"role": "assistant", "content": r1.get("content") or "", "reasoning_content": r1.get("reasoning_content") or ""},
              {"role": "user", "content": "Now name two risks the text mentions."}]
        c.req("followup_turn2", m, max_tokens=200, **GREEDY)
        q101 = rec = None
        for name, doc in (("pf38k", pf38k), ("pf101k", pf101k)):
            q101 = [{"role": "user", "content": doc + "\n\nWhat is the main purpose of the code above? Answer in detail."}]
            _, rec = c.req(f"long_{name}", q101, max_tokens=256, **GREEDY, **NO_THINK)
            c.vram(f"vram_after_{name}")
        if rec.get("cached"):  # pf101k reused pf38k's prefix: also time it "from scratch" (see scratch_client.py)
            c.other()
            c.req("long_pf101k_fresh", q101, max_tokens=16, cache_prompt=False, **GREEDY, **NO_THINK)
        print(f"== {c.label}: VRAM after long prompts {c.gpu.vram_mib()} MiB", flush=True)
    edit(c, pf38k)
    c.gpu_state("end")
    print("CLIENT_DONE", flush=True)


if __name__ == "__main__":
    main()
