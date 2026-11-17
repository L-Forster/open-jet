"""Tests of server/mk_server.py's side conversations without a GPU: a side request's own conversation is parked
after its reply (Engine.side_end_park), a later bounded request that continues it runs beside the live conversation again
(Engine.side_unpark), and those entries are evicted before any other parked conversation. Fake engines and the test
server come from test_mk_server.py."""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_mk_server as base  # noqa: E402  (its helpers; its test cases are not collected from here)


class _SideEngine(base._StoreEngine):
    """the store semantics of base._StoreEngine plus MKEngine::side_end_park / side_unpark. As the real engine: without
    MK_EXACT every prefill also keeps a checkpoint at the end of its prompt, and a side request's copies of the kept
    sequence (its recurrent state and the rows the side request overwrites) take store room until it ends"""

    side_keep = True

    def prefill(self, toks, ck_at=-1, cks=(), images=()):
        n0 = len(self.seq)
        r = super().prefill(toks, ck_at, cks, images)
        if n0 + len(toks) not in self.cks:
            self.cks.append(n0 + len(toks))
        return r

    def store_info(self):
        d = super().store_info()
        if self.kept is not None:
            d["used"] += self.STATE + self.PT * min(len(self.kept[0]), 1024)
        return d

    def side_end_park(self, k):
        if not self.side_ok:  # (the store ran full during it: the kept sequence is lost, this conversation stays)
            self.kept = None
            self.ops.append(("side_end_park", None, False))
            return False, None
        pid = self.park(k) if self.seq else None
        (seq, dn_c, cks), self.kept = self.kept, None
        self.seq, self.dn_c, self.cks = seq, dn_c, cks
        self.ops.append(("side_end_park", pid, True))
        return True, pid

    def side_unpark(self, pid):
        if self.kept is None or self.seq:
            raise RuntimeError("side_unpark: not right after side_begin")
        seq, dn_c, cks, _ = self.entries.pop(pid)
        self.seq, self.dn_c, self.cks = list(seq), dn_c, list(cks)
        self.ops.append(("side_unpark", pid, len(seq)))
        return True


class SideConversationTests(base._HttpCase):
    LONG = base.HostCacheTests.LONG
    chat, metrics, quiet = base.HostCacheTests.chat, base.HostCacheTests.metrics, base.HostCacheTests.quiet

    def serve_store(self, eng, side_max=200, park_mib=2, park_min=50, park_cks=2):
        srv, http_ = base.HostCacheTests.serve_store(self, eng, side_max, park_mib, park_min, park_cks)
        srv.im_start = ord("<")  # (the test tokenizer: one token per character; the last '<' starts the generation prompt)
        return srv, http_

    @staticmethod
    def follow_up(msgs, reply, text="more"):
        return msgs + [{"role": "assistant", "content": reply["choices"][0]["message"]["content"]}, {"role": "user", "content": text}]

    @staticmethod
    def side_ops(eng):
        return [o[0] for o in eng.ops if o[0].startswith("side")]

    def test_a_side_request_keeps_its_conversation_and_its_follow_up_resumes_beside_the_live_one(self) -> None:
        eng = _SideEngine(["a", "b"])
        srv, http_ = self.serve_store(eng)
        a1, s1 = [{"role": "user", "content": self.LONG}], [{"role": "user", "content": "a title please"}]
        out = {}

        def run():
            ra = self.chat(http_, a1)
            kept = list(eng.seq)
            rs = self.chat(http_, s1, max_tokens=4)
            out["s1_seq"] = list(eng.entries[1][0])
            self.assertEqual(eng.seq, kept)  # (the live conversation is back)
            out["s2"] = self.chat(http_, self.follow_up(s1, rs), max_tokens=4)
            self.assertEqual(eng.seq, kept)
            out["a2"] = self.chat(http_, self.follow_up(a1, ra))
        log = self.quiet(run)
        self.assertEqual([o[0] for o in eng.ops], ["side_begin", "park", "side_end_park", "side_begin", "side_unpark", "park",
                                                   "side_end_park"], log)
        self.assertEqual(eng.ops[4], ("side_unpark", 1, len(out["s1_seq"])), log)
        # the follow-up reused the side conversation's prompt; the live one's next turn reused the long prompt
        self.assertEqual(out["s2"]["usage"]["prompt_tokens_details"]["cached_tokens"], len(out["s1_seq"]) - 1, log)
        self.assertGreaterEqual(out["a2"]["usage"]["prompt_tokens_details"]["cached_tokens"], len(self.LONG), log)
        self.assertIn("its own conversation (", log)
        self.assertIn("its parked conversation (", log)
        m = self.metrics(http_)
        self.assertEqual([m[f"megakernel:{k}"] for k in ("side_requests_total", "side_conversations_parked_total",
                                                          "side_conversations_resumed_total", "parked_conversations")],
                         ["2", "2", "1", "1"], log)
        info = srv.engine_info()["host_cache"]
        self.assertEqual((info["parked"], info["parked_side"], info["side_conversations"]), (1, 1, True))

    def test_a_different_prompt_does_not_take_a_side_conversation(self) -> None:
        eng = _SideEngine(["a", "b"])
        srv, http_ = self.serve_store(eng)

        def run():
            self.chat(http_, [{"role": "user", "content": self.LONG}])
            self.chat(http_, [{"role": "user", "content": "a title please"}], max_tokens=4)
            self.chat(http_, [{"role": "user", "content": "a title for this one"}], max_tokens=4)
        log = self.quiet(run)
        self.assertEqual(self.side_ops(eng), ["side_begin", "side_end_park"] * 2, log)  # (no side_unpark: both stay)
        self.assertEqual(len(eng.entries), 2, log)

    def test_side_conversations_are_evicted_first_and_never_push_out_a_parked_conversation(self) -> None:
        eng = _SideEngine(["a", "b"])
        srv, http_ = self.serve_store(eng, park_mib=2)  # (2 MiB: two long conversations, or one and some short ones)
        a1, b1 = [{"role": "user", "content": self.LONG}], [{"role": "user", "content": self.LONG.upper()}]
        out = {}

        def run():
            ra = self.chat(http_, a1)
            self.chat(http_, b1)  # (A is parked: a conversation's entry)
            for i in range(40):  # titles beside B: each kept as a side entry, the oldest evicted when the store is full
                self.chat(http_, [{"role": "user", "content": f"title number {i:02d} please"}], max_tokens=4)
            out["a2"] = self.chat(http_, self.follow_up(a1, ra))  # (B is parked, A comes back)
        log = self.quiet(run)
        real = sorted(o[1] for o in eng.ops if o[0] == "park" and o[2] >= len(self.LONG))
        dropped = [o[1] for o in eng.ops if o[0] == "drop"]
        self.assertEqual(len(real), 2, log)  # A, then B when A came back
        self.assertTrue(dropped and not set(real) & set(dropped), (dropped, real))  # side entries went, never A or B
        self.assertIn(("unpark", real[0]), [o[:2] for o in eng.ops], log)
        self.assertGreaterEqual(out["a2"]["usage"]["prompt_tokens_details"]["cached_tokens"], len(self.LONG), log)
        self.assertEqual(self.side_ops(eng).count("side_end_park"), 40, log)
        # at most SIDE_KEEP_MAX side conversations stay: the oldest goes before the next one is parked
        self.assertEqual(len(dropped), 40 - self.mk.SIDE_KEEP_MAX, log)
        self.assertIn(f"the oldest of {self.mk.SIDE_KEEP_MAX} kept side request conversations", log)
        self.assertEqual(sum(e.side for e in srv.parked.values()), self.mk.SIDE_KEEP_MAX)

    def test_no_room_for_the_side_conversation_keeps_only_the_live_one(self) -> None:
        eng = _SideEngine(["a", "b"])
        srv, http_ = self.serve_store(eng, park_mib=1)  # (1 MiB: a parked conversation, the side copies, no more)
        a1, b1 = [{"role": "user", "content": self.LONG[:400]}], [{"role": "user", "content": self.LONG[:400].upper()}]

        def run():
            self.chat(http_, a1)
            self.chat(http_, b1)  # (A is parked)
            self.chat(http_, [{"role": "user", "content": "please write a title " * 7}], max_tokens=4)
        log = self.quiet(run)
        self.assertEqual([o[0] for o in eng.ops], ["park", "side_begin", "side_end"], log)
        self.assertIn("its own conversation was not parked (no room in the host store)", log)

    def test_parking_off_side_requests_off_or_an_old_library_give_the_previous_behaviour(self) -> None:
        for side_max, park_mib, cls, want in ((200, 0, _SideEngine, ["side_begin", "side_end"]),
                                              (0, 2, _SideEngine, []),
                                              (200, 2, base._StoreEngine, ["side_begin", "side_end"])):
            eng = cls(["a", "b"])
            srv, http_ = self.serve_store(eng, side_max=side_max, park_mib=park_mib)
            s1 = [{"role": "user", "content": "a title please"}]

            def run():
                self.chat(http_, [{"role": "user", "content": self.LONG}])
                rs = self.chat(http_, s1, max_tokens=4)
                self.chat(http_, self.follow_up(s1, rs), max_tokens=4)
            log = self.quiet(run)
            self.assertEqual(self.side_ops(eng), want, (side_max, park_mib, cls.__name__, log))

    def test_a_lost_kept_sequence_leaves_the_side_conversation_live(self) -> None:
        eng = _SideEngine(["a", "b"])
        srv, http_ = self.serve_store(eng)
        s1 = [{"role": "user", "content": "a title please"}]
        out = {}

        def run():
            self.chat(http_, [{"role": "user", "content": self.LONG}])
            eng.side_ok = False  # (the store runs full during the side request: the long conversation is lost)
            rs = self.chat(http_, s1, max_tokens=4)
            out["seq"] = list(eng.seq)
            eng.side_ok = True
            out["s2"] = self.chat(http_, self.follow_up(s1, rs), max_tokens=4)
        log = self.quiet(run)
        self.assertEqual([o[0] for o in eng.ops], ["side_begin", "side_end_park"], log)
        self.assertEqual(out["seq"][:3], [ord(c) for c in "<|i"], log)  # (the side conversation is the live one)
        self.assertEqual(out["s2"]["usage"]["prompt_tokens_details"]["cached_tokens"], len(out["seq"]) - 1, log)
        self.assertIn("stays cached", log)
        self.assertEqual(self.metrics(http_)["megakernel:side_requests_dropped_total"], "1")

    def test_make_room_evicts_side_conversations_first(self) -> None:
        eng = _SideEngine(["a"])
        srv, _ = self.serve_store(eng, park_mib=1)
        mk = self.mk
        for pid, side in ((1, False), (2, True), (3, False), (4, True)):
            eng.entries[pid] = ([1] * 300, 0, [], 300000)
            srv.parked[pid] = mk.Parked(pid, [1] * 300, [], 0, [], -1, 300000, side=side)
        self.quiet(lambda: self.assertTrue(srv.make_room(100000, 1 << 20)))
        self.assertEqual(list(srv.parked), [1, 3, 4])  # (side entry 2 first, though 1 is older)
        self.quiet(lambda: self.assertFalse(srv.make_room(700000, 1 << 20, side_only=True)))
        self.assertEqual(list(srv.parked), [1, 3])  # (side entry 4 went; the others are not side entries)
        self.quiet(lambda: self.assertTrue(srv.make_room(700000, 1 << 20, keep=1)))
        self.assertEqual(list(srv.parked), [1])


if __name__ == "__main__":
    unittest.main()
