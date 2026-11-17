"""Tests of server/mk_server.py's --agents mode (SlotServer, SlotScheduler) without a GPU, against a fake engine with the
slot API whose replies depend on the prompt and its sampling only, as the real engine's do (a slot decodes as it does
alone), while how many tokens each cycle emits depends on how many slots decode together (as the real depth policy)."""
import http.client
import json
import socket
import threading
import time
import types
import unittest
import unittest.mock
import zlib

from server_modules import load as _load_mk_server

TEMPLATE = ("{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{% endfor %}"
            "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}")
CALL = "<tool_call>\n<function=read_file>\n<parameter=path>\n/etc/hostname\n</parameter>\n</function>\n</tool_call>"
TOOLS = [{"type": "function", "function": {"name": "read_file", "parameters": {
    "type": "object", "properties": {"path": {"type": "string"}}}}}]
WORDS = ("alpha", "beta", "gamma", "delta", "kappa", "lambda", "sigma", "omega", "tau", "rho", "zeta", "eta")


def _reply(prompt, temp, seed):
    """the fake model: a reply that depends on the prompt and (sampled) on the seed only"""
    if "use the tool" in prompt.rsplit("<|im_start|>user", 1)[-1]:
        return "Let me look.\n" + CALL
    h = zlib.crc32(prompt.encode()) ^ (0 if temp == 0 else (seed * 2654435761) & 0xffffffff)
    n = 150 if "long" in prompt else 8 + h % 24
    text = " ".join(WORDS[(h >> (i % 20)) % len(WORDS)] + str(i % 7) for i in range(n))
    if "stopme" in prompt:
        text = text[:20] + " STOP " + text[20:]
    return text


class _SlotEngine:
    """MKEngine's slot API without a GPU. A slot's sequence, prompt queue and first token as the engine keeps them;
    each cycle a slot emits 1-3 tokens of its script (more with fewer slots in the cycle), filler after the end of its
    reply; [] once its context is full. Every call is checked to come from one thread at a time."""

    EOG, FILL, BASE = 3_000_000, 3_000_001, 2_000_000
    exact, has_mtp, slots, n_vocab = True, True, True, 0

    def __init__(self, ctx=4096, delay=0.002, prefill_delay=0.0):
        self.ctx, self.delay, self.prefill_delay = ctx, delay, prefill_delay
        self.seqs, self.pend, self.first, self.script, self.plen, self.sp = {}, {}, {}, {}, {}, {}
        self.calls, self.warm, self.threads = [], [], set()
        self.max_batch, self.fail_steps, self.fail_feeds = 0, 0, 0
        self._busy = threading.Lock()
        self.overlap = False

    def _enter(self):
        if not self._busy.acquire(blocking=False):
            self.overlap = True
            self._busy.acquire()
        self.threads.add(threading.current_thread().name)

    # tokenizer and vocabulary (from any thread)
    def tokenize(self, text, special=True, add_special=False):
        return [ord(c) for c in text]

    def piece(self, tok):
        if tok == self.FILL:
            return b" x"
        return chr(tok - self.BASE).encode() if tok >= self.BASE else chr(tok).encode()

    def is_eog(self, tok):
        return tok == self.EOG

    def keep_warm(self, on=True):
        self.warm.append((on, threading.current_thread().name))
        return 0.0

    def slots_open(self, n, ctx):
        self.n = n
        return ctx or self.ctx

    def slot_begin(self, slot, toks, temp, top_p, top_k, min_p, seed):
        self._enter()
        try:
            seq = self.seqs.get(slot, [])
            c = 0
            while c < len(seq) and c < len(toks) and seq[c] == toks[c]:
                c += 1
            reused = min(c, len(toks) - 1)
            self.seqs[slot], self.pend[slot], self.first[slot] = list(toks[:reused]), list(toks[reused:]), None
            self.sp[slot] = (temp, seed)
            self.calls.append(("begin", slot, reused))
            return reused
        finally:
            self._busy.release()

    def _feed(self, slot, budget):
        if budget <= 0:
            self.pend[slot] = []
            self.calls.append(("abandon", slot))
            return 0
        time.sleep(self.prefill_delay)
        chunk, self.pend[slot] = self.pend[slot][:budget], self.pend[slot][budget:]
        self.seqs[slot] += chunk
        self.calls.append(("prefill", slot, len(chunk)))
        if not self.pend[slot]:
            prompt = "".join(chr(t) for t in self.seqs[slot])
            self.script[slot] = [self.BASE + ord(ch) for ch in _reply(prompt, *self.sp[slot])] + [self.EOG]
            self.plen[slot] = len(self.seqs[slot])
            self.first[slot] = self.script[slot][0]
            self.seqs[slot].append(self.first[slot])
        return len(self.pend[slot])

    def slot_prefill(self, slot, budget):
        self._enter()
        try:
            if self.fail_feeds and budget > 0:
                self.fail_feeds -= 1
                raise RuntimeError("prefill: out of memory (test)")
            return self._feed(slot, budget)
        finally:
            self._busy.release()

    def _step(self, slots):
        if self.fail_steps:
            self.fail_steps -= 1
            raise RuntimeError("decode: no finite logit (test)")
        time.sleep(self.delay)
        self.max_batch = max(self.max_batch, len(slots))
        out = []
        for s in slots:
            if self.first[s] is not None:
                out.append([self.first[s]])
                self.first[s] = None
                continue
            if len(self.seqs[s]) + 3 > self.ctx:
                out.append([])
                continue
            pos = len(self.seqs[s]) - self.plen[s]
            k = 1 + (4 - len(slots) + pos) % 3
            sc = self.script[s]
            toks = [sc[i] if i < len(sc) else self.FILL for i in range(pos, pos + k)]
            self.seqs[s] += toks
            out.append(toks)
        self.calls.append(("step", tuple(slots)))
        return out

    def slots_step(self, slots):
        self._enter()
        try:
            return self._step(slots)
        finally:
            self._busy.release()


def _slot_server(mk, eng, n=4, ctx=4096):
    srv = object.__new__(mk.SlotServer)
    srv.local = threading.local()
    srv.eng, srv.im_start, srv.vision = eng, None, None
    srv.template = mk.make_template(TEMPLATE)
    srv.args = types.SimpleNamespace(ctx=ctx, agents=n, ck_every=0, ck_msgs=0, model="test.gguf", lib="libqwen38mk.so",
                                     kv="q8", draft=4, mtp=None, draft_vocab=None, mmproj=None)
    srv.warned, srv.model_id = set(), "test"
    srv.setup_slots(srv.args)
    return srv


class _Http:
    def __init__(self, mk, srv):
        self.httpd = mk.HTTPServer(("127.0.0.1", 0), mk.make_handler(srv))
        self.port = self.httpd.server_address[1]
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()

    def post(self, body, path="/v1/chat/completions", timeout=60):
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=timeout)
        c.request("POST", path, json.dumps(body), {"Content-Type": "application/json"})
        r = c.getresponse()
        data = r.read()
        c.close()
        return r.status, data

    def get(self, path):
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        c.request("GET", path)
        r = c.getresponse()
        data = r.read()
        c.close()
        return r.status, data


def _reply_of(path, body, status, data):
    """what a reply says (content, reasoning, tool calls, finish), from any of the routes and stream forms; ids and
    times left out"""
    if status != 200:
        return status, data
    if path == "/v1/messages":
        if body.get("stream"):
            text, stop = [], None
            for ln in data.decode().split("\n"):
                if ln.startswith("data: "):
                    ev = json.loads(ln[6:])
                    d = ev.get("delta") or {}
                    text.append(d.get("text") or d.get("thinking") or d.get("partial_json") or "")
                    if ev.get("type") == "content_block_start" and ev["content_block"]["type"] == "tool_use":
                        text.append("<use " + ev["content_block"]["name"] + ">")
                    stop = d.get("stop_reason") or stop
            return "".join(text), stop
        j = json.loads(data)
        return [{k: v for k, v in b.items() if k != "id"} for b in j["content"]], j["stop_reason"], j["usage"]["output_tokens"]
    if body.get("stream"):
        text, calls, finish = [], [], None
        for ln in data.decode().split("\n"):
            if ln.startswith("data: {"):
                ch = json.loads(ln[6:])
                for c in ch.get("choices") or []:
                    d = c.get("delta") or {}
                    text.append((d.get("content") or "") + (d.get("reasoning_content") or ""))
                    for tc in d.get("tool_calls") or []:
                        calls.append((tc.get("function") or {}).get("name") or "")
                        calls.append((tc.get("function") or {}).get("arguments") or "")
                    finish = c.get("finish_reason") or finish
        return "".join(text), "".join(calls), finish
    j = json.loads(data)
    m = j["choices"][0]["message"]
    calls = [(c["function"]["name"], c["function"]["arguments"]) for c in m.get("tool_calls") or []]
    return m.get("content"), m.get("reasoning_content"), calls, j["choices"][0]["finish_reason"], j["usage"]["completion_tokens"]


def _chat(text, **kw):
    return {"messages": [{"role": "system", "content": "You are a test."}, {"role": "user", "content": text}],
            "max_tokens": 400, "temperature": 0, **kw}


class AgentsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.mk = _load_mk_server()

    def serve(self, eng, n=4, ctx=4096):
        srv = _slot_server(self.mk, eng, n, ctx)
        http_ = _Http(self.mk, srv)
        self.addCleanup(http_.close)
        return srv, http_

    def run_all(self, http_, reqs):
        """sends (path, body) requests at once; returns their replies (_reply_of) in order"""
        out = [None] * len(reqs)

        def one(i):
            path, body = reqs[i]
            out[i] = _reply_of(path, body, *http_.post(body, path))
        th = [threading.Thread(target=one, args=(i,)) for i in range(len(reqs))]
        for t in th:
            t.start()
        for t in th:
            t.join(60)
        return out

    def alone(self, http_, reqs):
        return [self.run_all(http_, [r])[0] for r in reqs]

    def idle(self, srv, timeout=5.0):
        t0 = time.time()
        while srv.sched.counts() != (0, 0) or len(srv.sched.free) != srv.sched.n:
            self.assertLess(time.time() - t0, timeout, "the scheduler did not go idle")
            time.sleep(0.01)

    def test_concurrent_replies_equal_alone(self) -> None:
        eng = _SlotEngine()
        srv, http_ = self.serve(eng)
        reqs = [("/v1/chat/completions", _chat("first question")),
                ("/v1/chat/completions", _chat("second question", temperature=0.7, seed=11)),
                ("/v1/chat/completions", _chat("third question", stream=True)),
                ("/v1/chat/completions", _chat("fourth question", temperature=0.9, seed=5, stream=True))]
        ref = self.alone(http_, reqs)
        self.assertEqual(eng.max_batch, 1)
        for n in (2, 3, 4):
            self.assertEqual(self.run_all(http_, reqs[:n]), ref[:n], f"{n} at once")
        self.assertGreaterEqual(eng.max_batch, 2)
        self.assertEqual(eng.threads, {"mk-slots"})  # (the engine runs on the scheduler's thread only)
        self.assertFalse(eng.overlap)
        self.idle(srv)

    def test_anthropic_and_tools_concurrent(self) -> None:
        eng = _SlotEngine()
        srv, http_ = self.serve(eng)
        anth = {"model": "m", "max_tokens": 300, "temperature": 0,
                "messages": [{"role": "user", "content": "hello there"}]}
        reqs = [("/v1/messages", anth),
                ("/v1/messages", dict(anth, stream=True, messages=[{"role": "user", "content": "please use the tool"}],
                                      tools=[{"name": "read_file", "input_schema": {"type": "object"}}])),
                ("/v1/chat/completions", _chat("please use the tool", tools=TOOLS)),
                ("/v1/chat/completions", _chat("please use the tool", tools=TOOLS, stream=True))]
        ref = self.alone(http_, reqs)
        self.assertEqual(ref[2][2], [("read_file", '{"path": "/etc/hostname"}')])
        self.assertEqual(ref[2][3], "tool_calls")
        self.assertEqual(self.run_all(http_, reqs), ref)
        self.idle(srv)

    def test_arrival_mid_decode(self) -> None:
        """a request arriving while another decodes: its prompt rides along with the other's cycles (a chunk right before
        each step), both replies as alone"""
        eng = _SlotEngine(delay=0.003)
        srv, http_ = self.serve(eng)
        srv.sched.mix = 64
        a = ("/v1/chat/completions", _chat("a long story", stream=True))
        b = ("/v1/chat/completions", _chat("summarise: " + "lorem ipsum " * 200))
        ref = self.alone(self.serve(_SlotEngine())[1], [a, b])  # (another server: here B's prompt is not cached)
        out = [None, None]
        ta = threading.Thread(target=lambda: out.__setitem__(0, _reply_of(a[0], a[1], *http_.post(a[1]))))
        eng.calls.clear()
        ta.start()
        t0 = time.time()
        while sum(1 for c in eng.calls if c[0] == "step") < 20:  # (A decodes)
            self.assertLess(time.time() - t0, 10)
            time.sleep(0.005)
        out[1] = _reply_of(b[0], b[1], *http_.post(b[1]))
        ta.join(30)
        self.assertEqual(out, ref)
        fed = [i for i, c in enumerate(eng.calls) if c[0] == "prefill" and c[2] == 64]
        self.assertGreater(len(fed), 5)  # (B's ~2,450-token prompt in 64-row chunks beside A's cycles)
        self.assertTrue(all(eng.calls[i + 1][0] == "step" for i in fed))
        self.idle(srv)

    def test_a_failed_chunk_fails_its_own_request_only(self) -> None:
        """a prompt chunk that fails while another request decodes (the engine stepped nothing with it): that request
        gets the error, the decoding one goes on and answers as alone"""
        eng = _SlotEngine(delay=0.003)
        srv, http_ = self.serve(eng)
        srv.sched.mix = 64
        a = ("/v1/chat/completions", _chat("a long story", stream=True))
        ref = self.alone(self.serve(_SlotEngine())[1], [a])
        out = [None]
        ta = threading.Thread(target=lambda: out.__setitem__(0, _reply_of(a[0], a[1], *http_.post(a[1]))))
        eng.calls.clear()
        ta.start()
        t0 = time.time()
        while sum(1 for c in eng.calls if c[0] == "step") < 20:  # (A decodes)
            self.assertLess(time.time() - t0, 10)
            time.sleep(0.005)
        eng.fail_feeds = 1
        self.assertEqual(http_.post(_chat("summarise: " + "lorem ipsum " * 200))[0], 500)
        ta.join(30)
        self.assertEqual(out, ref)
        self.idle(srv)

    def test_prompt_feeding_settings(self) -> None:
        """open-jet's feeding by default (256 rows per mixed cycle, 512 tokens between cycles, the whole prompt while none
        decodes); MK_PF_MIX=0 MK_PF_CHUNK=0: a prompt arriving while another request decodes goes in one engine call"""
        with unittest.mock.patch.dict("os.environ", {"MK_PF_MIX": "", "MK_PF_CHUNK": "", "MK_PF_SOLO": ""}):
            s = self.mk.SlotScheduler(_SlotEngine(), 2, 4096)
        self.assertEqual((s.mix, s.chunk, s.solo), (256, 512, 0))
        eng = _SlotEngine(delay=0.003)
        with unittest.mock.patch.dict("os.environ", {"MK_PF_MIX": "0", "MK_PF_CHUNK": "0"}):
            srv, http_ = self.serve(eng)
        self.assertEqual((srv.sched.mix, srv.sched.chunk, srv.sched.solo), (0, 0, 0))
        a = ("/v1/chat/completions", _chat("a long story", stream=True))
        b = _chat("summarise: " + "lorem ipsum " * 200)
        ta = threading.Thread(target=http_.post, args=(a[1],))
        ta.start()
        t0 = time.time()
        while sum(1 for c in eng.calls if c[0] == "step") < 20:
            self.assertLess(time.time() - t0, 10)
            time.sleep(0.005)
        n0 = len(eng.calls)
        self.assertEqual(http_.post(b)[0], 200)
        ta.join(30)
        fed = [c for c in eng.calls[n0:] if c[0] == "prefill"]
        self.assertEqual(fed, [("prefill", fed[0][1], len(srv.render(b)[0]))])  # (one call: the whole prompt)
        with unittest.mock.patch.dict("os.environ", {"MK_PF_MIX": "128", "MK_PF_SOLO": "1024"}):
            s = self.mk.SlotScheduler(_SlotEngine(), 2, 4096)
        self.assertEqual((s.mix, s.chunk, s.solo), (128, 512, 1024))
        self.idle(srv)

    def test_follow_up_goes_back_to_its_slot(self) -> None:
        eng = _SlotEngine()
        srv, http_ = self.serve(eng, n=3)
        convs = [_chat(f"conversation {k}: tell me something") for k in range(3)]
        first = self.run_all(http_, [("/v1/chat/completions", c) for c in convs])
        turn1 = len(srv.render(convs[0])[0])  # (the fake tokenizer: one token per character)
        follow = [dict(c, messages=c["messages"] + [{"role": "assistant", "content": f[0]},
                                                    {"role": "user", "content": "and then?"}])
                  for c, f in zip(convs, first)]
        eng.calls.clear()
        out = self.run_all(http_, [("/v1/chat/completions", f) for f in follow])
        # each follow-up went back to its conversation's slot: it reaches past the first prompt into the reply (the
        # reply's tokens are the fake engine's own ids, never the prompt's), so the slot is first taken back inside the
        # first prompt (exact reuse: never the decoded rows), then the follow-up reuses that: another slot would give
        # only the system message and "conversation "
        begins = [c for c in eng.calls if c[0] == "begin"]
        self.assertEqual(len(begins), 6)
        for a, b in zip(begins[::2], begins[1::2]):
            self.assertEqual((a[1], a[2], b[2]), (b[1], turn1 - 1, turn1 - 1))
        ref = self.alone(http_, [("/v1/chat/completions", f) for f in follow])
        self.assertEqual(out, ref)
        self.idle(srv)

    def test_no_reuse_of_decoded_rows_without_exact(self) -> None:
        """MK_EXACT off: the follow-up's prompt goes to the engine at once (slot_begin appends to the decoded rows)"""
        eng = _SlotEngine()
        eng.exact = False
        srv, http_ = self.serve(eng, n=1)
        conv = _chat("one conversation")
        first = self.run_all(http_, [("/v1/chat/completions", conv)])[0]
        follow = dict(conv, messages=conv["messages"] + [{"role": "assistant", "content": first[0]},
                                                         {"role": "user", "content": "more"}])
        eng.calls.clear()
        self.assertEqual(http_.post(follow)[0], 200)
        self.assertEqual(len([c for c in eng.calls if c[0] == "begin"]), 1)
        self.idle(srv)

    def test_more_requests_than_slots(self) -> None:
        eng = _SlotEngine()
        srv, http_ = self.serve(eng, n=2)
        reqs = [("/v1/chat/completions", _chat(f"question number {k}", temperature=0.5, seed=k)) for k in range(5)]
        ref = self.alone(http_, reqs)
        self.assertEqual(self.run_all(http_, reqs), ref)
        self.assertEqual(eng.max_batch, 2)
        self.idle(srv)

    def test_stop_strings_and_max_tokens(self) -> None:
        eng = _SlotEngine()
        srv, http_ = self.serve(eng)
        reqs = [("/v1/chat/completions", _chat("stopme please", stop=["STOP"])),
                ("/v1/chat/completions", _chat("stopme please", stop=["STOP"], stream=True)),
                ("/v1/chat/completions", _chat("a long answer", max_tokens=17)),
                ("/v1/messages", {"model": "m", "max_tokens": 300, "temperature": 0, "stop_sequences": ["STOP"],
                                  "messages": [{"role": "user", "content": "stopme too"}]})]
        ref = self.alone(http_, reqs)
        self.assertEqual(ref[0][3], "stop")
        self.assertNotIn("STOP", ref[0][0])
        self.assertEqual((ref[2][3], ref[2][4]), ("length", 17))
        self.assertEqual(ref[3][1], "stop_sequence")
        self.assertEqual(self.run_all(http_, reqs), ref)
        self.idle(srv)  # (the slots of replies cut by a stop string are free again)

    def test_context_full_and_overflow(self) -> None:
        eng = _SlotEngine(ctx=500)  # (the engine's slot fills up before the server's limit: it emits [] then)
        srv, http_ = self.serve(eng, ctx=600)
        body = _chat("a long answer please " + "x" * 300, max_tokens=-1)
        r = _reply_of("/v1/chat/completions", body, *http_.post(body))
        self.assertEqual(r[3], "length")
        self.assertLess(r[4], 600 - 380 - 16)
        status, data = http_.post(_chat("y" * 700))
        self.assertEqual(status, 400)
        self.assertEqual(json.loads(data)["error"]["n_ctx"], 600)
        self.idle(srv)

    def test_refused_requests(self) -> None:
        eng = _SlotEngine()
        srv, http_ = self.serve(eng)
        for k in ("presence_penalty", "frequency_penalty"):
            status, data = http_.post(_chat("hi", **{k: 0.5}))
            self.assertEqual(status, 400)
            self.assertIn(f"{k} is not available with --agents", json.loads(data)["error"]["message"])
        img = {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBO"}}
        status, data = http_.post({"messages": [{"role": "user", "content": [{"type": "text", "text": "what?"}, img]}]})
        self.assertEqual(status, 400)
        self.assertIn("image and video input are not available with --agents", json.loads(data)["error"]["message"])
        status, data = http_.post({"model": "m", "max_tokens": 10, "messages": [{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "iVBO"}}]}]},
            "/v1/messages")
        self.assertEqual(status, 400)
        self.assertEqual(json.loads(data)["error"]["type"], "invalid_request_error")
        self.assertEqual(http_.post(_chat("still fine"))[0], 200)
        self.assertEqual(len([c for c in eng.calls if c[0] == "begin"]), 1)  # (refused before the engine)

    def test_disconnect_frees_its_slot(self) -> None:
        eng = _SlotEngine(delay=0.004)
        srv, http_ = self.serve(eng, n=2)
        others = [("/v1/chat/completions", _chat("short one")), ("/v1/chat/completions", _chat("short two", stream=True))]
        ref = self.alone(http_, others)
        body = json.dumps(_chat("a long story", stream=True)).encode()
        s = socket.create_connection(("127.0.0.1", http_.port), timeout=10)
        s.sendall(b"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\n"
                  b"Content-Length: " + str(len(body)).encode() + b"\r\n\r\n" + body)
        got = b""
        while got.count(b"data: ") < 5:
            got += s.recv(65536)
        out = [None]
        th = threading.Thread(target=lambda: out.__setitem__(0, self.run_all(http_, others)))
        th.start()
        time.sleep(0.05)
        s.close()  # (mid-stream, beside the others)
        th.join(30)
        self.assertEqual(out[0], ref)
        self.idle(srv)
        t0 = time.time()
        while srv.counters()["requests_cancelled"] != 1:
            self.assertLess(time.time() - t0, 5)
            time.sleep(0.01)

    def test_close_while_queued_or_prefilling(self) -> None:
        eng = _SlotEngine(prefill_delay=0.01)
        srv = _slot_server(self.mk, eng, n=1)
        sched = srv.sched
        sched.solo = 8
        a = sched.submit(list(range(100, 300)), (0.0, 1.0, 32, 0.0, 1), 50, 1)
        b = sched.submit(list(range(100, 140)), (0.0, 1.0, 32, 0.0, 1), 50, 2)
        t0 = time.time()
        while a.state != "prefill" or a.left > 150:
            self.assertLess(time.time() - t0, 5)
            time.sleep(0.002)
        b.close()  # (queued: dropped)
        a.close()  # (prefilling: abandoned at the next chunk)
        self.assertEqual(b.q.get(timeout=5), [])
        self.assertEqual(a.q.get(timeout=5), [])
        self.idle(srv)
        self.assertIn(("abandon", 0), eng.calls)
        self.assertEqual([c for c in eng.calls if c[0] == "begin"], [("begin", 0, 0)])  # (b never reached the engine)
        self.assertEqual(sched.seq[0], list(range(100, 300))[:200 - a.left])  # (what the slot keeps)

    def test_engine_error_reaches_its_requests(self) -> None:
        eng = _SlotEngine()
        srv, http_ = self.serve(eng)
        eng.fail_steps = 1
        status, data = http_.post(_chat("this one fails"))
        self.assertEqual(status, 500)
        self.assertIn("no finite logit", json.loads(data)["error"]["message"])
        self.assertEqual(http_.post(_chat("this one works"))[0], 200)
        self.idle(srv)

    def test_reporting(self) -> None:
        eng = _SlotEngine(delay=0.004)
        srv, http_ = self.serve(eng, n=3)
        p = json.loads(http_.get("/props")[1])
        self.assertEqual((p["total_slots"], p["default_generation_settings"]["n_ctx"]), (3, 4096))
        self.assertEqual(p["engine"]["agents"]["slots"], 3)
        th = threading.Thread(target=http_.post, args=(_chat("a long story"),))
        th.start()
        t0 = time.time()
        while not any(s["state"] == "decode" for s in json.loads(http_.get("/slots")[1])):
            self.assertLess(time.time() - t0, 5)
            time.sleep(0.005)
        slots = json.loads(http_.get("/slots")[1])
        self.assertEqual(len(slots), 3)
        self.assertEqual(sum(s["is_processing"] for s in slots), 1)
        self.assertIn("llamacpp:requests_processing 1\n", http_.get("/metrics")[1].decode())
        th.join(30)
        self.idle(srv)

    def test_keep_warm_is_the_schedulers(self) -> None:
        """armed when the scheduler goes idle, stopped before its next GPU work; never from a request's thread"""
        eng = _SlotEngine()
        srv, http_ = self.serve(eng)
        def entries(n):
            t0 = time.time()
            while len(eng.warm) < n:
                self.assertLess(time.time() - t0, 5)
                time.sleep(0.005)
        entries(1)
        http_.post(_chat("one"))
        entries(3)
        http_.post(_chat("two"))
        entries(5)
        time.sleep(0.05)
        self.assertEqual([on for on, _ in eng.warm], [True, False, True, False, True])
        self.assertEqual({t for _, t in eng.warm}, {"mk-slots"})


if __name__ == "__main__":
    unittest.main()
