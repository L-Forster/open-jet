#!/usr/bin/env python3
"""OpenJet Engine: OpenAI-compatible chat server for the Qwen3.8-27B megakernel (a drop-in for the parts of
llama-server's API that open-jet uses).

  python3 server/mk_server.py -m model.gguf --port 18080 [--ctx 32768] [--draft 4]

POST /v1/chat/completions (streaming or not): the request is rendered with the chat template stored
in the GGUF (enable_thinking / preserve_thinking / reasoning_effort via chat_template_kwargs, tools
via `tools`), reasoning is returned as `reasoning_content`, Qwen's XML tool calls as structured
`tool_calls`.  Sampling: temperature, top_p, top_k (<= 32), min_p, seed.  Successive requests of a
conversation reuse the processed tokens: the new prompt is matched against the sequence so far and
only the rest is prefilled (rewinding to the end of the previous prompt if it diverges after that).
GET /health, GET /v1/models, GET /slots (--parallel: each slot's state, tok/s, context, task; see tools/mk_top.py).
One request at a time unless --parallel.
"""
import argparse
import codecs
import ctypes
import json
import os
import random
import re
import sys
import contextlib
import queue
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import jinja2
from jinja2.sandbox import ImmutableSandboxedEnvironment

HERE = os.path.dirname(os.path.abspath(__file__))
TOPK_MAX = 32


def log(*a):
    print("[mk-server]", *a, file=sys.stderr, flush=True)


def under_wsl():
    try:
        with open("/proc/version") as f:
            return "microsoft" in f.read().lower()
    except OSError:
        return False


def drop_file_cache(path):
    """releases the page cache holding a file (the GGUF once its weights are on the GPU, which the engine unmaps after
    loading). Only under WSL, where cached pages stay counted against Windows, a whole model's worth of RAM held for
    nothing: elsewhere the cache is free and makes the next start fast."""
    if not under_wsl():
        return
    fd = os.open(path, os.O_RDONLY)
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)


class Engine:
    """ctypes binding of capi.cpp"""

    def __init__(self, lib, model, ctx, n_draft, draft_vocab, kv_mode=-1, kv_ctx=0):
        L = self.L = ctypes.CDLL(lib)
        vp, ip, cp = ctypes.c_void_p, ctypes.POINTER(ctypes.c_int), ctypes.c_char_p
        sig = {
            "mk_new2": (vp, [cp, ctypes.c_int, ctypes.c_int, cp, ctypes.c_int, ctypes.c_int]),
            "mk_last_error": (cp, []),
            "mk_chat_template": (ctypes.c_int, [vp, cp, ctypes.c_int]),
            "mk_tokenize": (ctypes.c_int, [vp, cp, ctypes.c_int, ip, ctypes.c_int]),
            "mk_piece": (ctypes.c_int, [vp, ctypes.c_int, cp, ctypes.c_int]),
            "mk_is_eog": (ctypes.c_int, [vp, ctypes.c_int]),
            "mk_set_sampling": (ctypes.c_int, [vp, ctypes.c_float, ctypes.c_float, ctypes.c_int, ctypes.c_float,
                                               ctypes.c_ulonglong]),
            "mk_prefill": (ctypes.c_int, [vp, ip, ctypes.c_int]),
            "mk_step": (ctypes.c_int, [vp, ip, ctypes.c_int]),
            "mk_reset": (None, [vp]),
            "mk_truncate": (ctypes.c_int, [vp, ctypes.c_int]),
            "mk_sequence": (ctypes.c_int, [vp, ip, ctypes.c_int]),
            "mk_committed": (ctypes.c_int, [vp]),
            "mk_checkpoint_pos": (ctypes.c_int, [vp]),
            "mk_rewind_to_checkpoint": (ctypes.c_int, [vp]),
            "mk_slots_open": (ctypes.c_int, [vp, ctypes.c_int, ctypes.c_int]),
            "mk_slot_begin2": (ctypes.c_int, [vp, ctypes.c_int, ip, ctypes.c_int, ctypes.c_float, ctypes.c_float, ctypes.c_int,
                                              ctypes.c_float, ctypes.c_ulonglong, ctypes.c_int, ctypes.c_int]),
            "mk_slots_open2": (ctypes.c_int, [vp, ctypes.c_int, ctypes.c_int, ctypes.c_int]),
            "mk_slot_segment": (ctypes.c_int, [vp, ctypes.c_int]),
            "mk_slot_prefill": (ctypes.c_int, [vp, ctypes.c_int, ctypes.c_int]),
            "mk_slots_step": (ctypes.c_int, [vp, ip, ctypes.c_int, ip, ctypes.c_int, ip]),
            "mk_slots_step_mixed": (ctypes.c_int, [vp, ctypes.c_int, ctypes.c_int, ip, ctypes.c_int, ip, ctypes.c_int, ip, ip]),
            "mk_slot_h": (ctypes.c_int, [vp, ctypes.c_int, ctypes.POINTER(ctypes.c_float), ctypes.c_int]),
        }
        for name, (res, args) in sig.items():
            f = getattr(L, name)
            f.restype, f.argtypes = res, args
        self.h = L.mk_new2(model.encode(), ctx, n_draft, draft_vocab.encode() if draft_vocab else None, kv_mode, kv_ctx)
        if not self.h:
            raise RuntimeError(L.mk_last_error().decode())
        self.ctx = ctx
        self._pieces = {}

    def _err(self, what):
        return RuntimeError(f"{what}: {self.L.mk_last_error().decode()}")

    def chat_template(self):
        n = self.L.mk_chat_template(self.h, None, 0)
        buf = ctypes.create_string_buffer(n + 1)
        self.L.mk_chat_template(self.h, buf, n)
        return buf.raw[:n].decode()

    def tokenize(self, text):
        b = text.encode()
        cap = len(b) + 16
        out = (ctypes.c_int * cap)()
        n = self.L.mk_tokenize(self.h, b, len(b), out, cap)
        if n < 0:
            cap = -n
            out = (ctypes.c_int * cap)()
            n = self.L.mk_tokenize(self.h, b, len(b), out, cap)
        return list(out[:n])

    def piece(self, tok):
        p = self._pieces.get(tok)
        if p is None:
            buf = ctypes.create_string_buffer(256)
            n = self.L.mk_piece(self.h, tok, buf, 256)
            p = self._pieces[tok] = buf.raw[:max(n, 0)]
        return p

    def is_eog(self, tok):
        return bool(self.L.mk_is_eog(self.h, tok))

    def set_sampling(self, temp, top_p, top_k, min_p, seed):
        if self.L.mk_set_sampling(self.h, temp, top_p, top_k, min_p, seed) < 0:
            raise self._err("sampling")

    def prefill(self, toks):
        arr = (ctypes.c_int * len(toks))(*toks)
        r = self.L.mk_prefill(self.h, arr, len(toks))
        if r < 0:
            raise self._err("prefill")
        return r

    def step(self):
        out = (ctypes.c_int * 8)()
        n = self.L.mk_step(self.h, out, 8)
        if n < 0:
            raise self._err("decode")
        return list(out[:n])

    def reset(self):
        self.L.mk_reset(self.h)

    def truncate(self, n):
        if self.L.mk_truncate(self.h, n) < 0:
            raise self._err("truncate")

    def sequence(self):
        n = self.L.mk_sequence(self.h, None, 0)
        out = (ctypes.c_int * max(n, 1))()
        self.L.mk_sequence(self.h, out, n)
        return list(out[:n])

    def committed(self):
        return self.L.mk_committed(self.h)

    def slots_open(self, n, ctx, seg_tokens=0):
        """opens n slots drawing on one K/V pool of n x ctx positions (0: the most that fit), shared segments
        (seg_tokens > 0) too; returns the pool's positions"""
        r = self.L.mk_slots_open2(self.h, n, ctx, seg_tokens)
        if r < 0:
            raise self._err("slots")
        return r

    def slot_begin(self, slot, toks, temp, top_p, top_k, min_p, seed, share_len=0, pin=False):
        """returns the reused prompt tokens; slot_prefill then feeds the rest. share_len: a prefix other prompts may
        share (stored once, as a segment), pin: keep that segment while unused"""
        arr = (ctypes.c_int * len(toks))(*toks)
        reused = self.L.mk_slot_begin2(self.h, slot, arr, len(toks), temp, top_p, top_k, min_p, seed, share_len, int(pin))
        if reused < 0:
            raise self._err("slot prefill")
        return reused

    def slot_segment(self, slot):
        """positions of the shared segment the slot reads (0: none)"""
        return self.L.mk_slot_segment(self.h, slot)

    def slot_prefill(self, slot, budget):
        """feeds up to `budget` prompt tokens; returns how many remain (0: ready to step). Budget 0 abandons."""
        rem = self.L.mk_slot_prefill(self.h, slot, budget)
        if rem < 0:
            raise self._err("slot prefill")
        return rem

    def slots_step(self, slots, per=8):  # (a lone slot decodes through the megakernel: up to 1 + n_draft tokens)
        n = len(slots)
        sa = (ctypes.c_int * n)(*slots)
        out = (ctypes.c_int * (n * per))()
        cnt = (ctypes.c_int * n)()
        if self.L.mk_slots_step(self.h, sa, n, out, per, cnt) < 0:
            raise self._err("decode")
        return [list(out[i * per:i * per + cnt[i]]) for i in range(n)]

    def slots_step_mixed(self, pslot, budget, slots, per=8):
        """slots_step with the next chunk of slot pslot's prompt in the same pass (budget rows with the step's);
        returns (the slots' tokens, the prompt tokens left)"""
        n = len(slots)
        sa = (ctypes.c_int * n)(*slots)
        out = (ctypes.c_int * (n * per))()
        cnt = (ctypes.c_int * n)()
        rem = ctypes.c_int()
        if self.L.mk_slots_step_mixed(self.h, pslot, budget, sa, n, out, per, cnt, ctypes.byref(rem)) < 0:
            raise self._err("decode")
        return [list(out[i * per:i * per + cnt[i]]) for i in range(n)], rem.value

    def slot_h(self, slot):  # (test)
        buf = (ctypes.c_float * 8192)()
        n = self.L.mk_slot_h(self.h, slot, buf, 8192)
        if n < 0:
            raise self._err("slot_h")
        return list(buf[:n])

    def checkpoint_pos(self):
        return self.L.mk_checkpoint_pos(self.h)

    def rewind_to_checkpoint(self):
        return self.L.mk_rewind_to_checkpoint(self.h) == 0


# ---------------------------------------------------------------------------------------------
# chat template and output parsing


def make_template(src):
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=["jinja2.ext.loopcontrols"])

    def raise_exception(msg):
        raise jinja2.exceptions.TemplateError(msg)

    env.globals["raise_exception"] = raise_exception
    env.filters["tojson"] = lambda x, indent=None, ensure_ascii=False, sort_keys=False: json.dumps(
        x, indent=indent, ensure_ascii=ensure_ascii, sort_keys=sort_keys)
    return env.from_string(src)


class BadRequest(Exception):
    pass


def normalize_messages(messages):
    out = []
    for m in messages:
        m = dict(m)
        content = m.get("content")
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") not in (None, "text"):
                    raise BadRequest(f"unsupported content part type {part.get('type')!r} (text only)")
        if content is None:
            m["content"] = ""
        if m.get("role") == "assistant" and m.get("tool_calls"):
            calls = []
            for tc in m["tool_calls"]:
                tc = dict(tc)
                fn = dict(tc.get("function") or {})
                args = fn.get("arguments")
                if isinstance(args, str):
                    try:
                        args = json.loads(args) if args.strip() else {}
                    except json.JSONDecodeError:
                        args = {}
                fn["arguments"] = args if isinstance(args, dict) else {}
                tc["function"] = fn
                calls.append(tc)
            m["tool_calls"] = calls
        out.append(m)
    return out


TOOL_CALL_RE = re.compile(r"<tool_call>\s*<function=([^>\n]+)>(.*?)(?:</function>\s*(?:</tool_call>|$)|$)", re.S)
PARAM_RE = re.compile(r"<parameter=([^>\n]+)>\n?(.*?)\n?</parameter>", re.S)


def parse_tool_calls(text, tools):
    schemas = {}
    for t in tools or []:
        fn = t.get("function", t)
        schemas[fn.get("name")] = (fn.get("parameters") or {}).get("properties") or {}
    calls = []
    for m in TOOL_CALL_RE.finditer(text):
        name = m.group(1).strip()
        props = schemas.get(name, {})
        args = {}
        for p in PARAM_RE.finditer(m.group(2)):
            key, val = p.group(1).strip(), p.group(2)
            typ = (props.get(key) or {}).get("type")
            if typ == "string" or (typ is None and not val.strip()[:1] in "[{0123456789-tfn"):
                args[key] = val
            else:
                try:
                    args[key] = json.loads(val)
                except json.JSONDecodeError:
                    args[key] = val
        calls.append({"id": "call_" + uuid.uuid4().hex[:24], "type": "function",
                      "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)}})
    return calls


class OutputParser:
    """Splits the generated text into reasoning (before </think>), content and tool-call markup."""

    def __init__(self, thinking, tools):
        self.mode = "reasoning" if thinking else "content_start"
        self.tools = tools
        self.buf = ""
        self.tool_text = ""

    def feed(self, text, final=False):
        """returns [(kind, text)] with kind in {reasoning, content}"""
        out = []
        self.buf += text
        while True:
            if self.mode == "reasoning":
                i = self.buf.find("</think>")
                if i >= 0:
                    out.append(("reasoning", self.buf[:i]))
                    self.buf = self.buf[i + len("</think>"):]
                    self.mode = "content_start"
                    continue
                keep = 0 if final else len("</think>") - 1
                if len(self.buf) > keep:
                    out.append(("reasoning", self.buf[:len(self.buf) - keep]))
                    self.buf = self.buf[len(self.buf) - keep:]
                break
            if self.mode == "content_start":
                self.buf = self.buf.lstrip()
                if not self.buf:
                    break
                self.mode = "content"
                continue
            if self.mode == "content":
                if self.tools:
                    i = self.buf.find("<tool_call>")
                    if i >= 0:
                        out.append(("content", self.buf[:i].rstrip()))
                        self.tool_text = self.buf[i:]
                        self.buf = ""
                        self.mode = "tools"
                        break
                    keep = 0 if final else len("<tool_call>") - 1
                else:
                    keep = 0
                if final:
                    out.append(("content", self.buf.rstrip()))
                    self.buf = ""
                elif len(self.buf) > keep:
                    # trailing whitespace waits for more text: it is dropped before a tool call or the end
                    seg = self.buf[:len(self.buf) - keep]
                    head = seg.rstrip()
                    out.append(("content", head))
                    self.buf = seg[len(head):] + self.buf[len(self.buf) - keep:]
                break
            # tools: collect everything
            self.tool_text += self.buf
            self.buf = ""
            break
        return [(k, t) for k, t in out if t]


# ---------------------------------------------------------------------------------------------


def task_label(body):
    """(task, agent) a request shows in GET /slots: metadata.task / metadata.agent (or the OpenAI `user` field) when
    the client sends them, else the first line of the first user message"""
    meta = body.get("metadata") if isinstance(body.get("metadata"), dict) else {}
    task = str(meta.get("task") or "")
    if not task:
        for m in body.get("messages") or []:
            if m.get("role") == "user":
                c = m.get("content")
                if isinstance(c, list):
                    c = " ".join(p.get("text", "") for p in c if isinstance(p, dict))
                # (skipping headings such as "Delegated task:")
                task = next((ln.strip() for ln in str(c or "").splitlines() if ln.strip() and not ln.strip().endswith(":")), "")
                break
    return task[:200], str(meta.get("agent") or body.get("user") or "")


def common_prefix(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


class SlotScheduler:
    """--parallel N: one thread owns the engine and decodes every active request together (slots_step); requests
    wait for a free slot and get their tokens through a queue. A request goes to the free slot holding the longest
    prefix of it (the engine reuses that prefix: the same conversation's next turn prefills only its new tokens);
    the rest of its prompt is prefilled in chunks that ride along with the active requests' steps: one pass of MIX_ROWS
    rows, the step's rows and a chunk of the prompt (MK_PF_MIX=0: chunks of PF_CHUNK tokens between the steps)."""

    PF_CHUNK = int(os.environ.get("MK_PF_CHUNK", "512"))
    MIX_ROWS = int(os.environ.get("MK_PF_MIX", "256"))

    class Req:
        def __init__(self, toks, sp, max_new, share_len=0, pin=False, task="", agent=""):
            self.toks, self.sp, self.max_new = toks, sp, max_new
            self.share_len, self.pin = share_len, pin
            self.task, self.agent = task, agent
            self.tail = ""  # the last of its output text (GET /slots)
            self.t_end = None
            self.q = queue.Queue()
            self.n = 0
            self.reused = 0  # prompt tokens the slot already held (set when its prefill starts)
            self.cancelled = False

    RATE_WINDOW = 3.0  # seconds of emitted tokens a slot's tok/s is measured over

    def __init__(self, eng, n, ctx):
        self.eng, self.n, self.ctx = eng, n, ctx
        self.cv = threading.Condition()
        self.pending, self.active, self.free = [], {}, list(range(n))
        self.filling = {}  # slot -> (request, start time, reused) while its prompt is prefilled
        self.building = {}  # shared prefix -> the slot prefilling it into a segment (requests sharing it wait)
        self.built = set()  # shared prefixes made into segments
        self.seq = [[] for _ in range(n)]  # each slot's tokens (prompt, emitted), as the engine holds them
        # GET /slots: each slot's latest request, (time, tokens) of its recent steps, prompt tokens left to prefill
        self.last = [None] * n
        self.emitted = [[] for _ in range(n)]
        self.pf_rem = [0] * n
        self.total_gen = 0
        threading.Thread(target=self.run, daemon=True).start()

    def submit(self, toks, sp, max_new, share_len=0, pin=False, task="", agent=""):
        r = SlotScheduler.Req(toks, sp, max_new, share_len, pin, task, agent)
        with self.cv:
            self.pending.append(r)
            self.cv.notify()
        return r

    @staticmethod
    def share_key(r):
        return hash(tuple(r.toks[:r.share_len]))

    def _finish(self, slot):
        r = self.active.pop(slot)
        self.free.append(slot)
        r.q.put(None)
        r.t_end = time.time()

    def snapshot(self):
        """per-slot state for GET /slots (read from the HTTP threads while the scheduler runs)"""
        now = time.time()
        active, filling = dict(self.active), dict(self.filling)
        slots = []
        for s in range(self.n):
            r = self.last[s]
            ev = [e for e in list(self.emitted[s]) if now - e[0] <= self.RATE_WINDOW]
            # the tokens of the steps after the window's first over the time since it (the first step's tokens were
            # made before its timestamp); nothing to measure before a second step
            rate = sum(e[1] for e in ev[1:]) / (now - ev[0][0]) if len(ev) > 1 else 0.0
            st = "decode" if s in active else "prefill" if s in filling else "idle"
            slots.append({
                "id": s, "state": st,
                "ctx": len(self.seq[s]) - (self.pf_rem[s] if st == "prefill" else 0),
                "tok_s": rate if st == "decode" else 0.0,
                "prompt": len(r.toks) if r else 0, "reused": r.reused if r else 0, "prefill_left": self.pf_rem[s],
                "generated": r.n if r else 0, "task": r.task if r else "", "agent": r.agent if r else "",
                "tail": r.tail if r else "", "idle_s": now - r.t_end if r and st == "idle" and r.t_end else 0.0,
            })
        return {"slots": slots, "pending": len(self.pending), "ctx": self.ctx, "total_generated": self.total_gen}

    def _fail(self, slot, r, e):
        r.q.put(e)
        r.t_end = time.time()
        self.seq[slot] = []
        self.pf_rem[slot] = 0
        self.free.append(slot)

    def run(self):
        eng = self.eng
        while True:
            with self.cv:
                while not self.pending and not self.active and not self.filling:
                    self.cv.wait()
                admit = []
                waiting = []
                building = set(self.building)
                while self.pending and self.free:
                    r = self.pending.pop(0)
                    key = self.share_key(r) if r.share_len else None
                    if key is not None and key in building:  # (attaches once that build is done)
                        waiting.append(r)
                        continue
                    if key is not None and key not in self.built:  # this one builds it; others sharing it wait
                        building.add(key)
                    # the slot holding this conversation; else the one whose loss costs least (an empty one before
                    # evicting another conversation for the few template tokens it shares): reused - discarded tokens
                    slot = max(self.free, key=lambda q: 2 * common_prefix(self.seq[q], r.toks) - len(self.seq[q]))
                    self.free.remove(slot)
                    admit.append((slot, r))
                self.pending[:0] = waiting
            for slot, r in admit:
                try:
                    reused = r.reused = eng.slot_begin(slot, r.toks, *r.sp, r.share_len, r.pin)
                    self.filling[slot] = (r, time.time(), reused)
                    self.seq[slot] = list(r.toks)
                    self.last[slot], self.emitted[slot], self.pf_rem[slot] = r, [], len(r.toks) - reused
                    if r.share_len and reused < r.share_len:  # this slot prefills the shared prefix (or no room)
                        self.building[self.share_key(r)] = slot
                    elif r.share_len:
                        self.built.add(self.share_key(r))
                except Exception as e:  # report to the request, keep serving
                    self._fail(slot, r, e)
            for slot in [s for s, r in self.active.items() if r.cancelled]:
                self._finish(slot)
            stepped = None  # (slots, their tokens) when the prompt chunk rode along with their step
            if self.filling:  # one chunk of the oldest admitted prompt (all of it when nothing is decoding)
                slot = next(iter(self.filling))
                r, t0, reused = self.filling[slot]
                try:
                    if r.cancelled:
                        eng.slot_prefill(slot, 0)
                        del self.filling[slot]
                        self.building = {k: v for k, v in self.building.items() if v != slot}
                        self.seq[slot] = []
                        self.pf_rem[slot] = 0
                        self.free.append(slot)
                        r.q.put(None)
                        r.t_end = time.time()
                    else:
                        if self.active and self.MIX_ROWS > 0:
                            slots = sorted(self.active)
                            em, rem = eng.slots_step_mixed(slot, self.MIX_ROWS, slots)
                            stepped = (slots, em)
                        else:
                            rem = eng.slot_prefill(slot, self.PF_CHUNK if self.active else 1 << 30)
                        self.pf_rem[slot] = rem
                        if rem == 0:
                            self._prefilled(slot, r, t0, reused)
                except Exception as e:
                    self.filling.pop(slot, None)
                    self.building = {k: v for k, v in self.building.items() if v != slot}
                    self._fail(slot, r, e)
                    if stepped is None and self.active and self.MIX_ROWS > 0:  # (the step failed with it)
                        for s in sorted(self.active):
                            self.active[s].q.put(e)
                            self._finish(s)
            if stepped is None:
                if not self.active:
                    continue
                slots = sorted(self.active)
                try:
                    stepped = (slots, eng.slots_step(slots))
                except Exception as e:
                    for s in slots:
                        self.active[s].q.put(e)
                        self._finish(s)
                    continue
            for s, toks in zip(*stepped):
                r = self.active[s]
                if not toks:  # (the K/V pool is full: the request ends at its length)
                    self._finish(s)
                    continue
                self.seq[s] += toks
                now = time.time()
                self.emitted[s] = [e for e in self.emitted[s] if now - e[0] <= self.RATE_WINDOW] + [(now, len(toks))]
                self.total_gen += len(toks)
                for t in toks:
                    r.q.put(t)
                    r.n += 1
                    if eng.is_eog(t) or r.n >= r.max_new:
                        self._finish(s)
                        break

    def _prefilled(self, slot, r, t0, reused):
        """the slot's prompt is in: it decodes from the next step on"""
        eng = self.eng
        del self.filling[slot]
        for k in [k for k, v in self.building.items() if v == slot]:
            del self.building[k]
            if r.share_len and eng.slot_segment(slot) >= r.share_len:
                self.built.add(k)
        seg = eng.slot_segment(slot)
        # at most what the K/V pool holds (shared with the other slots and segments: a slot that runs out first stops
        # early)
        r.max_new = min(r.max_new, self.ctx - len(r.toks) - 16)
        log(f"slot {slot}: {len(r.toks)} prompt tokens ({r.share_len} shareable), {reused} reused, segment {seg}, "
            f"prefill {time.time() - t0:.2f} s")
        if r.max_new <= 0:  # (nothing to generate: a shared-context prefill, or no room)
            self.free.append(slot)
            r.q.put(None)
            r.t_end = time.time()
        else:
            self.active[slot] = r

class Server:
    def __init__(self, args):
        self.args = args
        t0 = time.time()
        # --parallel: the slots hold the conversations (the engine's own K/V only serves the warm-up), auto K/V is
        # q8_0 (half the memory: context is what the slots run out of), and -c 0 sizes the slots to the memory left
        # (the engine's ctx then only bounds the RoPE table)
        parallel = args.parallel > 1
        kv = {"f16": 0, "q8": 1, "auto": 1 if parallel else -1}[args.kv]
        eng_ctx = args.ctx if args.ctx > 0 else 131072
        if parallel:  # prefill chunks of 1024: half the scratch (~0.34 GB, ~9k more pool positions), same prefill speed
            os.environ.setdefault("MK_PF_CAP", "1024")
        self.eng = Engine(args.lib, args.model, eng_ctx, args.draft, args.draft_vocab, kv, 1024 if parallel else 0)
        self.template = make_template(self.eng.chat_template())
        self.warm_up()
        self.model_id = os.path.basename(args.model)
        self.lock = threading.Lock()
        self.warned = set()
        self.sched = None
        self.shared_text = None
        if args.shared_context:
            with open(args.shared_context, encoding="utf-8") as f:
                self.shared_text = f.read()
        if args.parallel > 1:
            # one K/V pool for the slots and the shared segments: the shared context (--shared-context) and
            # automatically shared system prompts
            shared_toks = len(self.eng.tokenize(self.shared_text)) if self.shared_text else 0
            args.ctx = self.eng.slots_open(args.parallel, args.ctx, 0 if args.no_shared_prefix and not self.shared_text else 1)
            self.sched = SlotScheduler(self.eng, args.parallel, args.ctx)
            if self.shared_text:
                log(f"shared context: {shared_toks} tokens, prefilled once (with the first request) and shared by all")
        drop_file_cache(args.model)
        log(f"ready in {time.time() - t0:.1f} s: {self.model_id}, ctx {args.ctx}, {args.draft} drafts")

    def warm_up(self):
        """Runs one prefill and one decode cycle before the server listens.

        Loading allocates the weights and the KV cache, but the first kernel launches still need
        device memory of their own. When the context is too large for that, loading succeeds, /health
        answers, and the first request ends the process with a CUDA out-of-memory (CK exits), which
        the client sees as a dropped connection. Warming up here moves that failure to startup.
        The prompt must be at least two tokens: only then does prefill take the batched path, whose
        buffers are allocated on first use."""
        toks = self.eng.tokenize("Hello, how are you today?")
        assert len(toks) >= 2
        self.eng.prefill(toks)
        self.eng.step()
        self.eng.reset()

    def warn_once(self, key, msg):
        if key not in self.warned:
            self.warned.add(key)
            log("warning:", msg)

    def with_shared(self, messages):
        """--shared-context: every conversation's system message starts with the shared text"""
        if not self.shared_text:
            return messages
        if messages and messages[0].get("role") == "system" and isinstance(messages[0].get("content"), str):
            return [dict(messages[0], content=self.shared_text + "\n\n" + messages[0]["content"])] + messages[1:]
        return [{"role": "system", "content": self.shared_text}] + messages

    def share_len(self, body, toks):
        """tokens of the conversation's shareable prefix: the shared context (--shared-context), else the system message
        and tools (which agents repeat); what the prompt has in common with that prefix rendered alone (same template
        arguments), less a margin for tokens merging across its end; 0 if short. A segment (stored once, see
        slot_begin) holds it."""
        if self.shared_text:
            sys_msgs = [{"role": "system", "content": self.shared_text}]
        elif self.args.no_shared_prefix:
            return 0
        else:
            msgs = normalize_messages(body.get("messages") or [])
            sys_msgs = msgs[:1] if msgs and msgs[0].get("role") == "system" else []
        if not sys_msgs and not body.get("tools"):
            return 0
        try:  # (the template wants a user message: a placeholder one, past the common prefix)
            ctx = self.template_args(body, sys_msgs + [{"role": "user", "content": "\u0001"}], False)
            ptoks = self.eng.tokenize(self.template.render(**ctx))
        except Exception as e:
            self.warn_once("share", f"no prompt sharing: rendering the system prefix failed: {e!r}")
            return 0
        n = 0
        for a, b in zip(ptoks, toks):
            if a != b:
                break
            n += 1
        n -= 8
        return n if n >= 1024 else 0

    def template_args(self, body, messages, generation_prompt):
        kw = dict(body.get("chat_template_kwargs") or {})
        ctx = dict(messages=messages, add_generation_prompt=generation_prompt)
        if body.get("tools"):
            ctx["tools"] = body["tools"]
        for k in ("enable_thinking", "preserve_thinking", "reasoning_effort"):
            if k in kw:
                ctx[k] = kw[k]
        return ctx

    def render(self, body, generation_prompt=True):
        ctx = self.template_args(body, self.with_shared(normalize_messages(body.get("messages") or [])), generation_prompt)
        try:
            text = self.template.render(**ctx)
        except jinja2.exceptions.TemplateError as e:
            raise BadRequest(f"chat template: {e}")
        thinking = text.endswith("<think>\n")
        return text, thinking

    def load_prompt(self, toks):
        """brings the engine to `toks` processed, reusing what it has; returns (first token, reused count)"""
        eng = self.eng
        seq = eng.sequence()
        lcp = 0
        for a, b in zip(seq, toks):
            if a != b:
                break
            lcp += 1
        lcp = min(lcp, len(toks) - 1)  # at least one token is fed (it produces the next one)
        if seq and lcp >= eng.committed():
            eng.truncate(lcp)
            return eng.prefill(toks[lcp:]), lcp
        ck = eng.checkpoint_pos()
        if 0 < ck <= lcp and eng.rewind_to_checkpoint():
            return eng.prefill(toks[ck:]), ck
        eng.reset()
        return eng.prefill(toks), 0

    def sampling(self, body):
        temp = float(body.get("temperature", 0.8) if body.get("temperature") is not None else 0.8)
        top_p = float(body.get("top_p", 0.95) if body.get("top_p") is not None else 0.95)
        top_k = int(body.get("top_k", 40) if body.get("top_k") is not None else 40)
        min_p = float(body.get("min_p", 0.05) if body.get("min_p") is not None else 0.05)
        seed = body.get("seed")
        seed = random.getrandbits(63) if seed is None or int(seed) < 0 else int(seed)
        if top_k <= 0 or top_k > TOPK_MAX:
            self.warn_once("top_k", f"top_k {top_k} -> {TOPK_MAX} (the megakernel samples from at most {TOPK_MAX} candidates)")
            top_k = TOPK_MAX
        for k in ("presence_penalty", "frequency_penalty"):
            if float(body.get(k) or 0) != 0:
                self.warn_once(k, f"{k} is not supported by the megakernel and is ignored")
        if float(body.get("repetition_penalty") or body.get("repeat_penalty") or 1.0) != 1.0:
            self.warn_once("rep", "repetition_penalty is not supported by the megakernel and is ignored")
        if temp <= 0:
            temp, top_p, top_k, min_p = 0.0, 1.0, TOPK_MAX, 0.0
        if self.sched is None:
            self.eng.set_sampling(temp, top_p, top_k, min_p, seed)
        return temp, top_p, top_k, min_p, seed

    def generate(self, body, emit):
        """runs one chat completion; emit(kind, payload) is called with ('reasoning'|'content', text) deltas.
        Returns (finish_reason, tool_calls, usage). emit returning False aborts."""
        text, thinking = self.render(body)
        toks = self.eng.tokenize(text)
        max_new = body.get("max_tokens") or body.get("max_completion_tokens") or body.get("n_predict") or -1
        max_new = int(max_new)
        # (parallel: a shared segment holds part of the prompt; the scheduler checks the slot's own room)
        room = self.args.ctx  # (parallel: the K/V pool, shared segments included)
        limit = room - len(toks) - 16
        if limit <= 0:
            raise BadRequest(f"prompt of {len(toks)} tokens does not fit the context of {room}")
        max_new = limit if max_new <= 0 else min(max_new, limit)
        tools = body.get("tools") or []
        sp = self.sampling(body)
        t0 = time.time()
        parser = OutputParser(thinking, tools)
        dec = codecs.getincrementaldecoder("utf-8")(errors="replace")
        n_gen, finish, aborted = 0, "length", False
        if self.sched is not None:  # parallel: the scheduler decodes this request with the others
            req = self.sched.submit(toks, sp, max_new, self.share_len(body, toks), self.shared_text is not None,
                                    *task_label(body))
            t1 = None
            while True:
                t = req.q.get()
                if t is None:
                    break
                if isinstance(t, Exception):
                    raise t
                if t1 is None:
                    t1 = time.time()
                n_gen += 1
                if self.eng.is_eog(t):
                    finish = "stop"
                    continue
                if aborted:
                    continue
                text = dec.decode(self.eng.piece(t))
                req.tail = (req.tail + text)[-400:]
                for kind, delta in parser.feed(text):
                    if emit(kind, delta) is False:
                        aborted = True
                        req.cancelled = True
                        break
            t1 = t1 or time.time()
            reused = req.reused
        else:
            tok, reused = self.load_prompt(toks)
            t1 = time.time()
            pending = [tok]
            while n_gen < max_new and not aborted:
                if not pending:
                    pending = self.eng.step()
                t = pending.pop(0)
                n_gen += 1
                if self.eng.is_eog(t):
                    finish = "stop"
                    break
                for kind, delta in parser.feed(dec.decode(self.eng.piece(t))):
                    if emit(kind, delta) is False:
                        aborted = True
                        break
        for kind, delta in parser.feed(dec.decode(b"", final=True), final=True):
            emit(kind, delta)
        calls = parse_tool_calls(parser.tool_text, tools) if parser.tool_text else []
        if calls:
            finish = "tool_calls"
        t2 = time.time()
        log(f"prompt {len(toks)} tok ({reused} reused) in {t1 - t0:.2f} s, {n_gen} tok in {t2 - t1:.2f} s "
            f"({n_gen / max(t2 - t1, 1e-9):.1f} tok/s), {finish}")
        usage = {"prompt_tokens": len(toks), "completion_tokens": n_gen, "total_tokens": len(toks) + n_gen,
                 "prompt_tokens_details": {"cached_tokens": reused}}
        timings = {"prompt_n": len(toks) - reused, "prompt_ms": (t1 - t0) * 1e3, "predicted_n": n_gen,
                   "predicted_ms": (t2 - t1) * 1e3, "predicted_per_second": n_gen / max(t2 - t1, 1e-9)}
        return finish, calls, usage, timings


def make_handler(srv):
    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):
            pass

        def _json(self, code, obj):
            data = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            path = self.path.split("?")[0]
            if path in ("/health", "/v1/health"):
                self._json(200, {"status": "ok"})
            elif path in ("/v1/models", "/models"):
                self._json(200, {"object": "list", "data": [{"id": srv.model_id, "object": "model", "owned_by": "openjet"}]})
            elif path == "/props":
                self._json(200, {"default_generation_settings": {"n_ctx": srv.args.ctx}, "total_slots": max(1, srv.args.parallel)})
            elif path == "/slots" and srv.sched is not None:
                self._json(200, {"model": srv.model_id, **srv.sched.snapshot()})
            else:
                self._json(404, {"error": {"message": "not found", "type": "not_found_error"}})

        def do_POST(self):
            path = self.path.split("?")[0]
            n = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(n) if n else b""
            if path not in ("/v1/chat/completions", "/chat/completions"):
                self._json(404 if not path.startswith("/slots") else 501,
                           {"error": {"message": f"{path} is not supported by the megakernel server", "type": "not_supported"}})
                return
            try:
                body = json.loads(raw or b"{}")
            except json.JSONDecodeError:
                self._json(400, {"error": {"message": "invalid JSON", "type": "invalid_request_error"}})
                return
            with (srv.lock if srv.sched is None else contextlib.nullcontext()):
                try:
                    if body.get("stream"):
                        self._stream(body)
                    else:
                        self._complete(body)
                except BadRequest as e:
                    self._json(400, {"error": {"message": str(e), "type": "invalid_request_error"}})
                except (BrokenPipeError, ConnectionResetError):
                    pass
                except Exception as e:  # engine failure: report it, keep serving
                    log("error:", repr(e))
                    try:
                        self._json(500, {"error": {"message": str(e), "type": "server_error"}})
                    except Exception:
                        pass

        def _complete(self, body):
            parts = {"reasoning": [], "content": []}
            finish, calls, usage, timings = srv.generate(body, lambda k, t: parts[k].append(t))
            msg = {"role": "assistant", "content": "".join(parts["content"]) or (None if calls else "")}
            if parts["reasoning"]:
                msg["reasoning_content"] = "".join(parts["reasoning"]).strip()
            if calls:
                msg["tool_calls"] = calls
            self._json(200, {"id": "chatcmpl-" + uuid.uuid4().hex, "object": "chat.completion", "created": int(time.time()),
                             "model": srv.model_id, "choices": [{"index": 0, "message": msg, "finish_reason": finish}],
                             "usage": usage, "timings": timings})

        def _stream(self, body):
            cid, created = "chatcmpl-" + uuid.uuid4().hex, int(time.time())
            started = [False]

            def chunk(delta, finish=None, usage=None, timings=None):
                obj = {"id": cid, "object": "chat.completion.chunk", "created": created, "model": srv.model_id,
                       "choices": [] if delta is None else [{"index": 0, "delta": delta, "finish_reason": finish}]}
                if usage is not None:
                    obj["usage"] = usage
                if timings is not None:
                    obj["timings"] = timings
                data = b"data: " + json.dumps(obj, ensure_ascii=False).encode() + b"\n\n"
                self.wfile.write(b"%x\r\n" % len(data) + data + b"\r\n")
                self.wfile.flush()

            def head():
                if not started[0]:
                    started[0] = True
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Cache-Control", "no-cache")
                    self.send_header("Transfer-Encoding", "chunked")
                    self.end_headers()
                    chunk({"role": "assistant", "content": None})

            def emit(kind, text):
                head()
                try:
                    chunk({"reasoning_content": text} if kind == "reasoning" else {"content": text})
                except (BrokenPipeError, ConnectionResetError):
                    return False
                return True

            # render / tokenize errors are reported as a 400 before the stream starts
            srv.render(body)
            finish, calls, usage, timings = srv.generate(body, emit)
            head()
            for i, c in enumerate(calls):
                chunk({"tool_calls": [{"index": i, **c}]})
            chunk({}, finish=finish, timings=timings)
            if (body.get("stream_options") or {}).get("include_usage"):
                chunk(None, usage=usage)
            data = b"data: [DONE]\n\n"
            self.wfile.write(b"%x\r\n" % len(data) + data + b"\r\n0\r\n\r\n")
            self.wfile.flush()

    return H


def main():
    ap = argparse.ArgumentParser(
        description="OpenJet Engine: CUDA inference for Qwen3.8-27B with an OpenAI-compatible HTTP API."
    )
    ap.add_argument("-m", "--model", required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=18080)
    ap.add_argument("-c", "--ctx", type=int, default=None,
                    help="context positions (default 32768; with --parallel: per slot, all of them in one pool any slot draws "
                         "on; 0: the most that fit)")
    ap.add_argument("--no-shared-prefix", action="store_true",
                    help="with --parallel: don't store shared prompt prefixes once (by default system prompts and tools repeated "
                         "across conversations are prefilled once and shared)")
    ap.add_argument("--shared-context", default=None,
                    help="with --parallel: a text file every conversation starts with (in its system message), prefilled once "
                         "at startup and stored once: all agents share that context")
    ap.add_argument("-d", "--draft", type=int, default=4)
    ap.add_argument("--draft-vocab", default=os.path.join(os.path.dirname(HERE), "draft_vocab_32k.bin"))
    ap.add_argument("--lib", default=os.path.join(os.path.dirname(HERE), "libqwen38mk.so"))
    ap.add_argument("--parallel", type=int, default=1,
                    help="requests decoded together (slots; each holds its own KV cache of --ctx positions); 1: one at a time")
    ap.add_argument("--agents", type=int, default=0,
                    help="serve N agents at once: --parallel N with all the memory left as their context (-c 0 unless given)")
    ap.add_argument("--kv", choices=["auto", "f16", "q8"], default="auto",
                    help="KV cache format (auto: q8_0 only when the fp16 cache does not fit)")
    args = ap.parse_args()
    if args.agents > 0:
        args.parallel = args.agents
        if args.ctx is None:
            args.ctx = 0
    if args.ctx is None:
        args.ctx = 32768
    srv = Server(args)
    httpd = ThreadingHTTPServer((args.host, args.port), make_handler(srv))
    httpd.daemon_threads = True
    log(f"listening on http://{args.host}:{args.port}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
