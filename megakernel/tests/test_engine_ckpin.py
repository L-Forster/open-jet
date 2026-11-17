"""Tests of server/mk_server.py's pinned checkpoints without a GPU: the end of a prompt's system message
(and where a new session left the cached one inside it) is pinned (Engine.ck_pin), so a new session of the same client
resumes there although the pool thinned the conversation's older checkpoints or the conversation was parked; a parked
conversation restored only for its system prompt stays parked. Fake engines and the test server come from
test_mk_server.py / test_engine_side.py."""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_engine_side as side  # noqa: E402  (its helpers; its test cases are not collected from here)
import test_mk_server as base  # noqa: E402

IM, END = 1, 2  # <|im_start|>, <|im_end|>: one token each, as in the real vocabulary


class _PinEngine(side._SideEngine):
    """the store semantics of _SideEngine, a pool that keeps only the newest SLOTS checkpoints besides the pinned ones
    (the engine's thinning at its worst), and MKEngine::ck_pin / ck_pinned: a request for P pins the checkpoint in
    (P - 32, P] now, else the one the next prefill saves at P; parks carry the pinned ones besides the newest k"""

    ck_pins = True
    SLOTS = 3

    def __init__(self, pieces, **kw):
        super().__init__(pieces, **kw)
        self.pins, self.pin_req, self.pin_calls, self.entry_pins = set(), [], [], {}

    def tokenize(self, text):
        out = []
        for i, piece in enumerate(text.split("<|im_start|>")):
            if i:
                out.append(IM)
            for j, p in enumerate(piece.split("<|im_end|>")):
                if j:
                    out.append(END)
                out += [ord(c) for c in p]
        return out

    def ck_pin(self, positions):
        self.pin_calls.append(list(positions))
        self.pin_req = []
        for P in positions[:2]:
            c = max((p for p in self.cks if P - 32 < p <= P), default=None)
            if c is not None:
                self.pins.add(c)
            elif P > 0:
                self.pin_req.append(P)

    def pinned(self):
        return sorted(p for p in self.pins if p in self.cks)

    def prefill(self, toks, ck_at=-1, cks=(), images=()):
        n0, old = len(self.seq), set(self.cks)
        req, self.pin_req = [P for P in self.pin_req if n0 < P < n0 + len(toks)], []
        r = super().prefill(toks, ck_at, tuple(cks) + tuple(req), images)
        self.pins -= set(self.cks) - old  # (a new checkpoint is not pinned unless requested)
        self.pins |= set(req)
        keep = sorted(set(self.cks) - set(self.pinned()))[-self.SLOTS:]
        self.cks = sorted(set(keep) | set(self.pinned()))
        return r

    def park(self, k):
        pid = super().park(k)
        if pid is not None:
            seq, dn_c, cks, n = self.entries[pid]
            pins = self.pinned() if k else []
            self.entries[pid] = (seq, dn_c, sorted(set(cks) | set(pins)), n)
            self.entry_pins[pid] = set(pins)
        return pid

    def unpark(self, pid):
        pins = self.entry_pins.pop(pid, set())
        super().unpark(pid)
        self.pins = set(pins)

    def side_begin(self):
        self.kept_pins = set(self.pins)
        self.pins = set()
        return super().side_begin()

    def side_end_park(self, k):
        ok, pid = super().side_end_park(k)
        if ok:
            self.pins = self.kept_pins
        return ok, pid

    def side_end(self):
        ok = super().side_end()
        self.pins = self.kept_pins if ok else set()
        return ok


class _OldEngine(_PinEngine):
    """a library without mk_ck_pin (the deployed DLLs): the server must not call it"""

    ck_pins = False

    def ck_pin(self, positions):
        raise AssertionError("ck_pin on a library without it")


SYS = "You are a coding agent. Tools: read, write, run. " * 23  # (1,127 tokens with the test tokenizer: over PIN_MIN)


class PinnedCheckpointTests(base._HttpCase):
    chat, metrics, quiet = base.HostCacheTests.chat, base.HostCacheTests.metrics, base.HostCacheTests.quiet
    follow_up = staticmethod(side.SideConversationTests.follow_up)

    def serve_pin(self, eng, ck_pin=1, park_mib=8):
        srv, http_ = base.HostCacheTests.serve_store(self, eng, 200, park_mib, 50, 2)
        srv.im_start, srv.sys_role, srv.args.ck_pin = IM, eng.tokenize("system"), ck_pin
        srv.args.ctx = 16384
        return srv, http_

    @staticmethod
    def session(task, system=SYS):
        return [{"role": "system", "content": system}, {"role": "user", "content": task}]

    @staticmethod
    def anchor(system=SYS):
        return 1 + len("system\n") + len(system) + 2  # (<|im_start|>system\n ... <|im_end|>\n: the next <|im_start|>)

    @staticmethod
    def cached(r):
        return r["usage"]["prompt_tokens_details"]["cached_tokens"]

    def grow(self, http_, msgs, turns=4):
        """a conversation of `turns` turns from msgs (each adds a reply and a user message): its last prompt's reply"""
        r = self.chat(http_, msgs)
        for i in range(turns - 1):
            msgs = self.follow_up(msgs, r, f"next step {i}: " + "details " * 30)
            r = self.chat(http_, msgs)
        return msgs, r

    def test_pins_are_the_system_messages_end_only_with_a_long_system_message_and_a_new_library(self) -> None:
        eng = _PinEngine(["a"])
        srv, _ = self.serve_pin(eng)
        toks = eng.tokenize(f"<|im_start|>system\n{SYS}<|im_end|>\n<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n")
        self.assertEqual(srv.ck_pins(toks, 0, []), [self.anchor()])
        short = eng.tokenize("<|im_start|>system\nbe brief<|im_end|>\n<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n")
        self.assertEqual(srv.ck_pins(short, 0, []), [])  # (under PIN_MIN: not worth a slot)
        user = eng.tokenize(f"<|im_start|>user\n{SYS}<|im_end|>\n<|im_start|>assistant\n")
        self.assertEqual(srv.ck_pins(user, 0, []), [])  # (no system message: nothing every session shares)
        self.assertEqual(srv.ck_pins(toks, 1100, []), [1100, self.anchor()])  # (left the cached one inside it)
        self.assertEqual(srv.ck_pins(toks, 500, []), [self.anchor()])  # (under PIN_MIN)
        srv.args.ck_pin = 0
        self.assertEqual(srv.ck_pins(toks, 0, []), [])
        srv.args.ck_pin, srv.eng = 1, _OldEngine(["a"])
        self.assertEqual(srv.ck_pins(toks, 0, []), [])
        self.assertTrue(srv.engine_info()["checkpoints"]["pinned"] is False)

    def test_a_new_session_resumes_at_the_pinned_system_end_after_the_conversation_grew(self) -> None:
        for cls, ck_pin in ((_PinEngine, 1), (_PinEngine, 0), (_OldEngine, 1)):
            eng = cls(["a", "b"])
            srv, http_ = self.serve_pin(eng, ck_pin=ck_pin)
            out = {}

            def run():
                self.grow(http_, self.session("task one: " + "fix the bug " * 20), turns=5)
                out["b"] = self.chat(http_, self.session("task two: write the docs"))
            log = self.quiet(run)
            want = self.anchor() if cls is _PinEngine and ck_pin else 0
            self.assertEqual(self.cached(out["b"]), want, (cls.__name__, ck_pin, log))
            if want:
                self.assertIn(f"resumed at the checkpoint at {want:,}", log)
                self.assertEqual(eng.pinned(), [want])  # (the new session pins it again)
            else:
                self.assertIn("CACHE MISS", log)
            if cls is _OldEngine:
                self.assertEqual(eng.pin_calls, [])

    def test_a_parked_conversation_restored_for_its_system_prompt_stays_parked(self) -> None:
        eng = _PinEngine(["a", "b"])
        srv, http_ = self.serve_pin(eng)
        out = {}

        def run():
            a, ra = self.grow(http_, self.session("task one: " + "fix the bug " * 20), turns=3)
            out["a_len"] = len(eng.seq)
            self.chat(http_, self.session("something else entirely " * 10, system="another client. " * 80))  # (A parked)
            out["b"] = self.chat(http_, self.session("task two: write the docs"))  # (A's system prompt)
            out["parked"] = sorted(len(e.seq) for e in srv.parked.values())
            out["a"] = self.chat(http_, self.follow_up(a, ra, "and now the tests"))  # (A comes back, whole)
        log = self.quiet(run)
        self.assertEqual(self.cached(out["b"]), self.anchor(), log)
        self.assertIn("parked the cached conversation (a new session reuses its system prompt)", log)
        self.assertIn(out["a_len"], out["parked"], log)  # (A is still parked after B used its prefix)
        self.assertGreaterEqual(self.cached(out["a"]), out["a_len"] - 1, log)
        a_entry = [o for o in eng.ops if o[0] == "park" and o[2] == out["a_len"]]
        self.assertEqual(len(a_entry), 2, log)  # (parked when the other client came, again when B took its prefix)

    def test_a_low_priority_conversation_parked_again_for_a_new_session_stays_low(self) -> None:
        eng = _PinEngine(["a", "b"])
        srv, http_ = self.serve_pin(eng)
        out = {}

        def run():
            self.grow(http_, self.session("task one: " + "fix the bug " * 20), turns=3)
            out["a_len"] = len(eng.seq)
            self.chat(http_, self.session("something else entirely " * 10, system="another client. " * 80))  # (A parked)
            for e in srv.parked.values():
                if len(e.seq) == out["a_len"]:
                    e.low = True  # (as compaction debris: Server.cache_switch)
            self.chat(http_, self.session("task two: write the docs"))  # (A's system prompt: restored, parked again)
            out["low"] = [e.low for e in srv.parked.values() if len(e.seq) == out["a_len"]]
        log = self.quiet(run)
        self.assertIn("parked the cached conversation (a new session reuses its system prompt, low priority", log)
        self.assertEqual(out["low"], [True], log)

    def test_without_pins_a_parked_conversation_is_not_restored_for_a_new_session(self) -> None:
        eng = _OldEngine(["a", "b"])
        srv, http_ = self.serve_pin(eng)
        out = {}

        def run():
            self.grow(http_, self.session("task one: " + "fix the bug " * 20), turns=3)
            self.chat(http_, self.session("something else entirely " * 10, system="another client. " * 80))
            out["b"] = self.chat(http_, self.session("task two: write the docs"))
        log = self.quiet(run)
        self.assertEqual(self.cached(out["b"]), 0, log)
        self.assertNotIn("unpark", [o[0] for o in eng.ops], log)
        self.assertTrue(all(e.pins == () for e in srv.parked.values()))

    def test_a_learned_pin_where_sessions_leave_the_system_prompt(self) -> None:
        eng = _PinEngine(["a", "b"])
        srv, http_ = self.serve_pin(eng)
        base_sys = SYS[:1100]
        out = {}

        def run():  # (the system prompt ends with per-session text: its working directory)
            for i, d in enumerate(("/work/alpha", "/work/bravo", "/work/charlie")):
                out[i] = self.chat(http_, self.session("look around", system=base_sys + f" cwd {d}"))
                out[f"pins{i}"] = eng.pin_calls[-1]
        log = self.quiet(run)
        cut = 1 + len("system\n") + len(base_sys + " cwd /work/")
        self.assertEqual(out["pins0"], [self.anchor(base_sys + " cwd /work/alpha")], log)
        self.assertEqual(out["pins1"], [cut, self.anchor(base_sys + " cwd /work/bravo")], log)
        self.assertEqual(self.cached(out[2]), cut, log)  # (the third session resumes where they part)

    def test_a_side_request_in_between_keeps_the_pin(self) -> None:
        eng = _PinEngine(["a", "b"])
        srv, http_ = self.serve_pin(eng)
        out = {}

        def run():
            self.grow(http_, self.session("task one: " + "fix the bug " * 20), turns=3)
            out["before"] = eng.pinned()
            self.chat(http_, [{"role": "user", "content": "a title please"}], max_tokens=4)
            out["after"] = eng.pinned()
            out["b"] = self.chat(http_, self.session("task two: write the docs"))
        log = self.quiet(run)
        self.assertEqual(out["before"], [self.anchor()], log)
        self.assertEqual(out["after"], out["before"], log)
        self.assertIn("side_begin", [o[0] for o in eng.ops], log)
        self.assertEqual(self.cached(out["b"]), self.anchor(), log)


if __name__ == "__main__":
    unittest.main()
