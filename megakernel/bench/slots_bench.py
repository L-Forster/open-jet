#!/usr/bin/env python3
"""Parallel slots over the C API (libqwen38mk.so): speed of n conversations decoded together, and their exactness.

  speed:   n = --nmin .. --n slots on short coding prompts (temperature 0.6, top_p 0.95, top_k 20), each decoding --gen
           tokens: total tok/s over the batched steps (prefill excluded), after a warm-up run per slot count
  exact:   each prompt alone on a slot (greedy), then all of them together on other slots, then again on the same slots
           (prefix reuse): every slot's tokens must equal its solo run's
  stagger: the last prompt begins after --lead batched steps of the others and is prefilled in --budget-token chunks
           between their steps (mk_slots_step_mixed): every slot's tokens must equal its solo run's, whose prompt is
           prefilled in one piece (with MK_EXACT=1 the chunking changes no row; --ref-chunked: the solo run of the late
           prompt fed in the same chunks, for engines or modes where it does)
  all:     exact, then speed

  python3 bench/slots_bench.py MODEL [speed|exact|stagger|all] [--n 4] [--gen 256] [--draft 7] [--long] [--lib LIB]

--long: prompts of 3,780 / 4,400 / 5,050 / 5,700 tokens (four slices of bench/suite/prompts/pf38k.txt), past the
2,048-row window of the KV presets (MK_KV / MK_KV_WIN; run those with MK_EXACT=1). The engine's environment applies as
usual (MK_KV, MK_KV_WIN, MK_PAR_RMAX, ...). Exit status 1 when an exactness check differs.
"""
import argparse
import ctypes
import hashlib
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
PROMPTS = [
    "Write a Python module implementing an LRU cache with TTL expiry, thread safety and unit tests.",
    "Write a Rust command-line tool that counts word frequencies in files, with argument parsing and error handling.",
    "Write a TypeScript Express server with JWT authentication, a users CRUD API and input validation.",
    "Write a C++ class for a thread pool with a task queue, futures and graceful shutdown, plus an example main.",
]
LONG_TOK = [3780, 4400, 5050, 5700]  # (a 192-token decode with drafts crosses 3840 / 4480 / 5120 / 5760)
LONG_LEAD = ["Explain", "Summarize", "Review", "Document"]  # (different first words: no prefix shared between them)
LONG_SLICE = 25000  # characters of pf38k.txt per long prompt (slice i starts at i * LONG_SLICE)


class Eng:
    def __init__(self, lib, model, n_draft, dv):
        L = self.L = ctypes.CDLL(lib)
        vp, ip, cp = ctypes.c_void_p, ctypes.POINTER(ctypes.c_int), ctypes.c_char_p
        sig = {
            "mk_new2": (vp, [cp, ctypes.c_int, ctypes.c_int, cp, ctypes.c_int, ctypes.c_int]),
            "mk_last_error": (cp, []),
            "mk_tokenize": (ctypes.c_int, [vp, cp, ctypes.c_int, ip, ctypes.c_int]),
            "mk_slots_open2": (ctypes.c_int, [vp, ctypes.c_int, ctypes.c_int, ctypes.c_int]),
            "mk_slot_begin2": (ctypes.c_int, [vp, ctypes.c_int, ip, ctypes.c_int, ctypes.c_float, ctypes.c_float, ctypes.c_int,
                                              ctypes.c_float, ctypes.c_ulonglong, ctypes.c_int, ctypes.c_int]),
            "mk_slot_prefill": (ctypes.c_int, [vp, ctypes.c_int, ctypes.c_int]),
            "mk_slots_step": (ctypes.c_int, [vp, ip, ctypes.c_int, ip, ctypes.c_int, ip]),
            "mk_slots_step_mixed": (ctypes.c_int, [vp, ctypes.c_int, ctypes.c_int, ip, ctypes.c_int, ip, ctypes.c_int, ip, ip]),
            "mk_free": (None, [vp]),
        }
        for name, (res, args) in sig.items():
            f = getattr(L, name)
            f.restype, f.argtypes = res, args
        # (the engine's own sequence only needs a small context: the slots hold the conversations)
        self.h = L.mk_new2(model.encode(), 1024, n_draft, dv.encode() if dv else None, 1, 1024)
        if not self.h:
            raise SystemExit(f"mk_new2: {self.err()}")

    def err(self):
        return (self.L.mk_last_error() or b"").decode()

    def check(self, r, what):
        if r < 0:
            raise SystemExit(f"{what}: {self.err()}")
        return r

    def tokenize(self, text):
        b = text.encode()
        out = (ctypes.c_int * (len(b) + 16))()
        return list(out[:self.L.mk_tokenize(self.h, b, len(b), out, len(b) + 16)])

    def slots_open(self, n, ctx):
        return self.check(self.L.mk_slots_open2(self.h, n, ctx, 0), "mk_slots_open2")

    def begin(self, slot, toks, temp, seed):
        arr = (ctypes.c_int * len(toks))(*toks)
        return self.check(self.L.mk_slot_begin2(self.h, slot, arr, len(toks), temp, 0.95, 20, 0.0, seed, 0, 0), "mk_slot_begin2")

    def prefill(self, slot, budget=1 << 30):
        return self.check(self.L.mk_slot_prefill(self.h, slot, budget), "mk_slot_prefill")

    def step(self, slots, per=16):
        n = len(slots)
        out, cnt = (ctypes.c_int * (n * per))(), (ctypes.c_int * n)()
        self.check(self.L.mk_slots_step(self.h, (ctypes.c_int * n)(*slots), n, out, per, cnt), "mk_slots_step")
        return [list(out[i * per:i * per + cnt[i]]) for i in range(n)]

    def step_mixed(self, pslot, budget, slots, per=16):
        """a step of slots and the next budget prompt tokens of pslot: (tokens per slot, prompt tokens left)"""
        n = len(slots)
        out, cnt, rem = (ctypes.c_int * (n * per))(), (ctypes.c_int * n)(), ctypes.c_int(0)
        self.check(self.L.mk_slots_step_mixed(self.h, pslot, budget, (ctypes.c_int * n)(*slots), n, out, per, cnt,
                                              ctypes.byref(rem)), "mk_slots_step_mixed")
        return [list(out[i * per:i * per + cnt[i]]) for i in range(n)], rem.value


def wrap(t):  # one user turn, thinking off (the chat template's shape)
    return f"<|im_start|>user\n{t}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def long_prompt(e, i, src):
    """long prompt i: exactly LONG_TOK[i] tokens, a slice of pf38k.txt between the turn's head and tail"""
    head = e.tokenize(f"<|im_start|>user\n{LONG_LEAD[i]} the following CUDA source:\n```\n")
    tail = e.tokenize("\n```\nGo through it section by section.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")
    body = e.tokenize(src[i * LONG_SLICE:(i + 1) * LONG_SLICE])
    k = LONG_TOK[i] - len(head) - len(tail)
    if len(body) < k:
        raise SystemExit(f"long prompt {i}: the slice has {len(body)} tokens, {k} needed")
    return head + body[:k] + tail


def gen(e, slots, prompts, n_new, temp=0.0, budget=1 << 30):
    """starts the slots on their prompts and steps them together until each has n_new tokens: (tokens, step seconds)"""
    for s, p in zip(slots, prompts):
        e.begin(s, p, temp, 1234 + s)
        while e.prefill(s, budget) > 0:
            pass
    outs, t, active = {s: [] for s in slots}, 0.0, list(slots)
    while active:
        t0 = time.perf_counter()
        em = e.step(active)
        t += time.perf_counter() - t0
        for s, toks in zip(active, em):
            outs[s] += toks
        active = [s for s, toks in zip(active, em) if len(outs[s]) < n_new and toks]
    return {s: outs[s][:n_new] for s in slots}, t


def stagger(e, slots, prompts, n_new, budget, lead):
    """slots[:-1] start together; after `lead` steps the last one begins, its prompt fed beside their steps"""
    early, late = slots[:-1], slots[-1]
    for s, p in zip(early, prompts[:-1]):
        e.begin(s, p, 0.0, 1234 + s)
        while e.prefill(s) > 0:
            pass
    outs = {s: [] for s in slots}

    def take(act, em):
        for s, toks in zip(act, em):
            outs[s] += toks
        return [s for s, toks in zip(act, em) if len(outs[s]) < n_new and toks]

    active = list(early)
    for _ in range(lead):
        if active:
            active = take(active, e.step(active))
    rem, chunks = len(prompts[-1]) - e.begin(late, prompts[-1], 0.0, 1234 + late), 0
    while rem > 0:
        if active:
            em, rem = e.step_mixed(late, budget, active)
            active = take(active, em)
        else:
            rem = e.prefill(late, budget)
        chunks += 1
    active.append(late)
    while active:
        active = take(active, e.step(active))
    return {s: outs[s][:n_new] for s in slots}, chunks


def compare(tag, got, ref):
    k = next((j for j in range(min(len(got), len(ref))) if got[j] != ref[j]), None)
    same = got == ref
    print(f"{tag}: {'IDENTICAL' if same else 'DIFFER at ' + str(k if k is not None else min(len(got), len(ref)))}"
          f" ({len(got)} tokens)", flush=True)
    return not same


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("model")
    ap.add_argument("what", nargs="?", default="speed", choices=["speed", "exact", "stagger", "all"])
    ap.add_argument("--lib", default=os.path.join(REPO, "libqwen38mk.so"))
    ap.add_argument("--draft-vocab", default=os.path.join(REPO, "draft_vocab_32k.bin"), help="'' = none")
    ap.add_argument("--n", type=int, default=4, help="slots (speed: the largest count)")
    ap.add_argument("--nmin", type=int, default=1, help="speed: the smallest slot count")
    ap.add_argument("--gen", type=int, default=256, help="tokens per slot")
    ap.add_argument("--draft", type=int, default=7)
    ap.add_argument("--ctx", type=int, default=8192, help="positions per slot")
    ap.add_argument("--temp", type=float, default=0.6, help="speed: temperature (0: greedy, the same tokens in every build)")
    ap.add_argument("--reps", type=int, default=1, help="speed: timed runs per slot count")
    ap.add_argument("--long", action="store_true", help="the long prompts (see above)")
    ap.add_argument("--budget", type=int, default=1024, help="stagger: the late slot's prefill chunk")
    ap.add_argument("--lead", type=int, default=8, help="stagger: steps of the others before the late slot begins")
    ap.add_argument("--ref-chunked", action="store_true", help="stagger: the late prompt's solo run fed in the same chunks")
    a = ap.parse_args()
    e = Eng(a.lib, a.model, a.draft, a.draft_vocab or None)
    print(f"slots: {a.n} x {e.slots_open(a.n, a.ctx)} positions", flush=True)
    src = None
    if a.long:
        with open(os.path.join(HERE, "suite", "prompts", "pf38k.txt"), encoding="utf-8") as f:
            src = f.read()

    def prompts(n, extra=lambda i: ""):
        if a.long:
            return [long_prompt(e, i % 4, src) for i in range(n)]
        return [e.tokenize(wrap(PROMPTS[i % 4] + extra(i))) for i in range(n)]

    bad = 0
    if a.what != "speed":
        pr = prompts(a.n)
        print("prompt tokens:", [len(p) for p in pr], flush=True)
        slots = [(i + 1) % a.n for i in range(a.n)]  # (together: each prompt on another slot than its solo run's)
        if a.what in ("exact", "all"):
            ref = {i: gen(e, [i], [pr[i]], a.gen)[0][i] for i in range(a.n)}
            o, _ = gen(e, slots, pr, a.gen)
            bad += sum(compare(f"prompt {i}: slot {slots[i]} batched vs solo", o[slots[i]], ref[i]) for i in range(a.n))
            o, _ = gen(e, slots, pr, a.gen)
            bad += sum(compare(f"prompt {i}: slot {slots[i]} repeat (prefix reuse) vs solo", o[slots[i]], ref[i])
                       for i in range(a.n))
        else:  # stagger
            ref = {i: gen(e, [i], [pr[i]], a.gen, budget=a.budget if a.ref_chunked and i == a.n - 1 else 1 << 30)[0][i]
                   for i in range(a.n)}
            o, chunks = stagger(e, slots, pr, a.gen, a.budget, a.lead)
            print(f"stagger: slot {slots[-1]} began after {a.lead} steps, prefilled in {chunks} chunks of <= {a.budget}",
                  flush=True)
            bad += sum(compare(f"prompt {i}: slot {slots[i]} staggered vs solo", o[slots[i]], ref[i]) for i in range(a.n))
    if a.what in ("speed", "all"):
        for n in range(a.nmin, a.n + 1):
            slots = list(range(n))
            gen(e, slots, prompts(n, lambda i: f" (variant {100 + i})"), 32, temp=a.temp)  # (warm-up: clocks up)
            for rep in range(a.reps):
                o, t = gen(e, slots, prompts(n, lambda i: f" (take {200 + 10 * rep + i})"), a.gen, temp=a.temp)
                tot = sum(len(v) for v in o.values())
                h = hashlib.sha1(repr(sorted(o.items())).encode()).hexdigest()[:10]
                print(f"speed n={n}: {tot} tokens in {t:.2f} s of steps = {tot / t:.1f} tok/s total, {tot / t / n:.1f} per slot "
                      f"[outputs {h}]", flush=True)
    print("RESULT:", "FAIL" if bad else "OK", flush=True)
    e.L.mk_free(e.h)  # (MK_PROFILE prints its tables here)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
