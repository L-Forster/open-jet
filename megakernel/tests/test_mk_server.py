"""Tests of server/mk_server.py without a GPU: request parsing, prompt reuse and image handling against fake engines."""
import base64
import contextlib
import http.client
import io
import json
import os
import socket
import sys
import tempfile
import threading
import time
import types
import unittest
from pathlib import Path
from unittest import mock

from server_modules import load as _load_mk_server

ROOT = Path(__file__).resolve().parent.parent


class MegakernelServerParsingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.mk = _load_mk_server()

    def _feed_all(self, parser, chunks):
        out = []
        for c in chunks:
            out += parser.feed(c)
        out += parser.feed("", final=True)
        merged = {"reasoning": "", "content": ""}
        for kind, text in out:
            merged[kind] += text
        return merged

    def test_reasoning_and_content_are_split_across_chunk_boundaries(self) -> None:
        parser = self.mk.OutputParser(thinking=True, tools=[])
        text = "Think about it.\n</think>\n\nThe answer is 4."
        merged = self._feed_all(parser, [text[i:i + 3] for i in range(0, len(text), 3)])
        self.assertEqual(merged["reasoning"], "Think about it.\n")
        self.assertEqual(merged["content"], "The answer is 4.")

    def test_tool_call_markup_is_held_back_and_parsed(self) -> None:
        tools = [{"type": "function", "function": {"name": "read_file", "parameters": {
            "type": "object", "properties": {"path": {"type": "string"}, "max_lines": {"type": "integer"}}}}}]
        parser = self.mk.OutputParser(thinking=False, tools=tools)
        text = ("Let me look.\n\n<tool_call>\n<function=read_file>\n<parameter=path>\n/etc/hostname\n</parameter>\n"
                "<parameter=max_lines>\n10\n</parameter>\n</function>\n</tool_call>")
        merged = self._feed_all(parser, [text[i:i + 5] for i in range(0, len(text), 5)])
        self.assertEqual(merged["content"], "Let me look.")
        calls = self.mk.parse_tool_calls(parser.tool_text, tools)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["function"]["name"], "read_file")
        self.assertEqual(json.loads(calls[0]["function"]["arguments"]), {"path": "/etc/hostname", "max_lines": 10})

    def test_string_parameters_are_not_json_decoded(self) -> None:
        tools = [{"type": "function", "function": {"name": "run", "parameters": {
            "type": "object", "properties": {"cmd": {"type": "string"}}}}}]
        calls = self.mk.parse_tool_calls("<tool_call>\n<function=run>\n<parameter=cmd>\n123\n</parameter>\n</function>\n</tool_call>", tools)
        self.assertEqual(json.loads(calls[0]["function"]["arguments"]), {"cmd": "123"})

    def test_assistant_tool_call_arguments_are_decoded_for_the_template(self) -> None:
        msgs = self.mk.normalize_messages([{"role": "assistant", "content": None, "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{\"a\": 1}"}}]}])
        self.assertEqual(msgs[0]["content"], "")
        self.assertEqual(msgs[0]["tool_calls"][0]["function"]["arguments"], {"a": 1})

    def test_images_are_rejected(self) -> None:
        with self.assertRaises(self.mk.BadRequest):
            self.mk.normalize_messages([{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "x"}}]}])



class _FakeEngine:
    """MKEngine's sequence / checkpoint semantics, without a GPU"""

    exact = False  # (an MK_EXACT engine never appends; see Server.load_prompt)

    def __init__(self):
        self.seq, self.dn_c, self.cks, self.calls = [], 0, [], []

    def sequence(self):
        return list(self.seq)

    def committed(self):
        return self.dn_c

    def checkpoints(self):
        return sorted(self.cks)

    def truncate(self, n):
        self.seq = self.seq[:n]

    def reset(self):
        self.seq, self.dn_c, self.cks = [], 0, []
        self.calls.append(("reset",))

    def rewind_to(self, pos):
        if pos not in self.cks:
            return False
        self.seq, self.dn_c = self.seq[:pos], pos
        self.cks = [c for c in self.cks if c <= pos]
        self.calls.append(("rewind", pos))
        return True

    def prefill(self, toks, ck_at=-1, cks=()):
        n0 = len(self.seq)
        self.seq += toks
        self.cks += [c for c in [*cks, ck_at] if n0 < c < len(self.seq)]
        self.dn_c = len(self.seq)
        self.calls.append(("prefill", n0, len(toks), ck_at, tuple(cks)))
        self.seq.append(7)  # the first generated token
        return 7

    @contextlib.contextmanager
    def cancel_when(self, gone):  # (Engine.cancel_when; _ChunkedEngine runs the real one)
        yield


class MegakernelPromptReuseTests(unittest.TestCase):
    IM = 1  # <|im_start|>

    @classmethod
    def setUpClass(cls) -> None:
        cls.mk = _load_mk_server()

    def _server(self, every=8):
        srv = object.__new__(self.mk.Server)
        srv.eng, srv.im_start = _FakeEngine(), self.IM
        srv.args = mock.Mock(ck_every=every, ck_msgs=1)
        return srv

    def test_checkpoints_at_message_starts_the_generation_prompt_and_every_n_tokens(self) -> None:
        srv = self._server()
        toks = [self.IM, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, self.IM, 5, 5, self.IM, 9, 9]
        self.assertEqual(srv.load_prompt(toks), (7, 0))
        self.assertEqual(srv.eng.calls, [("reset",), ("prefill", 0, len(toks), 15, (8, 12, 15, 16))])

    def test_a_divergence_rewinds_to_the_last_checkpoint_before_it(self) -> None:
        srv = self._server()
        first = [self.IM, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, self.IM, 5, 5, self.IM, 9, 9]
        srv.load_prompt(first)
        srv.eng.dn_c = len(srv.eng.seq) + 5  # decoding went on
        second = first[:14] + [6, 6, self.IM, 9, 9]  # diverges at 14: checkpoints 8, 12 are before it
        self.assertEqual(srv.load_prompt(second), (7, 12))
        self.assertEqual(srv.eng.calls[-2:], [("rewind", 12), ("prefill", 12, len(second) - 12, 16, (16,))])

    def test_an_extension_of_the_sequence_is_appended_without_a_rewind(self) -> None:
        srv = self._server(every=0)
        first = [self.IM, 5, 5, self.IM, 9]
        srv.load_prompt(first)
        second = srv.eng.sequence() + [self.IM, 6, self.IM, 9]
        self.assertEqual(srv.load_prompt(second), (7, len(first) + 1))
        self.assertEqual(srv.eng.calls[-1], ("prefill", len(first) + 1, 4, len(second) - 2, (len(second) - 2,)))

    def test_no_checkpoint_before_the_divergence_means_a_full_prefill(self) -> None:
        srv = self._server(every=0)
        srv.load_prompt([self.IM, 5, 5, self.IM, 9])
        srv.eng.dn_c += 3
        self.assertEqual(srv.load_prompt([self.IM, 6, 6, self.IM, 9]), (7, 0))
        self.assertEqual(srv.eng.calls[-2][0], "reset")


class _FakeSamplingEngine:
    """records what Server.sampling passes to the engine"""

    def __init__(self):
        self.sampling = self.penalty = None

    def set_penalty(self, presence, frequency):
        self.penalty = (presence, frequency)

    def set_sampling(self, temp, top_p, top_k, min_p, seed):
        self.sampling = (temp, top_p, top_k, min_p, seed)


class MegakernelSamplingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.mk = _load_mk_server()

    def _top_k(self, body):
        """(the top_k the engine gets, the warnings given) for a sampled request with these fields"""
        srv = object.__new__(self.mk.Server)
        srv.eng, srv.warned = _FakeSamplingEngine(), set()
        srv.sampling({"temperature": 0.7, "seed": 1, **body})
        return srv.eng.sampling[2], srv.warned

    def test_top_k_absent_is_the_default_without_a_warning(self) -> None:
        for body in ({}, {"top_k": None}):
            self.assertEqual(self._top_k(body), (self.mk.DEF_TOP_K, set()), body)

    def test_top_k_off_is_the_engine_maximum_without_a_warning(self) -> None:
        for body in ({"top_k": 0}, {"top_k": -1}):
            self.assertEqual(self._top_k(body), (self.mk.TOPK_MAX, set()), body)

    def test_top_k_in_range_is_kept(self) -> None:
        self.assertEqual(self._top_k({"top_k": 20}), (20, set()))
        self.assertEqual(self._top_k({"top_k": self.mk.TOPK_MAX}), (self.mk.TOPK_MAX, set()))

    def test_top_k_above_the_maximum_is_clamped_with_a_warning(self) -> None:
        self.assertEqual(self._top_k({"top_k": 64}), (self.mk.TOPK_MAX, {"top_k"}))


class _FakeVisionEngine(_FakeEngine):
    """_FakeEngine with mk_prefill_img's images argument and a character tokenizer"""

    def tokenize(self, text):
        return [ord(c) for c in text]

    def prefill(self, toks, ck_at=-1, cks=(), images=()):
        self.calls.append(("images", tuple((p, nx, ny) for p, nx, ny, _ in images)))
        return super().prefill(toks, ck_at, cks)

    def keep_warm(self, on=True):
        self.calls.append(("warm",))
        return 120.0


class _FakeImage:
    def __init__(self, key, nx, ny):
        self.key, self.nx, self.ny, self.n = key, nx, ny, nx * ny
        self.encoded = 0
        self.encode_s = None  # (Vision.Image: seconds of its CPU encode, None until encoded)
        self.log = None

    def rows(self):
        self.encoded += 1
        if self.encode_s is None and self.log is not None:
            self.log.append(("encode", self.key))
        self.encode_s = 0.5
        return 0  # (an address)


class MegakernelVisionTests(unittest.TestCase):
    VS, VE, PAD, IM = 1001, 1002, 1003, 1

    @classmethod
    def setUpClass(cls) -> None:
        cls.mk = _load_mk_server()

    def _media(self, content):
        media = []
        msgs = self.mk.normalize_messages([{"role": "user", "content": content}], media)
        return msgs[0]["content"], media

    def test_image_parts_are_flattened_as_llama_server_does(self) -> None:
        url = {"url": "data:image/png;base64,iVBORw0KGgo="}
        img = {"type": "image_url", "image_url": url}
        m = self.mk.MEDIA_MARKER
        self.assertEqual(self._media([{"type": "text", "text": "a"}, img, {"type": "text", "text": "b"}])[0], f"a{m}b")
        self.assertEqual(self._media([img, {"type": "text", "text": "b"}])[0], f"{m}b")
        self.assertEqual(self._media([{"type": "text", "text": "a"}, {"type": "text", "text": "b"}, img])[0], f"a\nb{m}")
        text, media = self._media([img, img])
        self.assertEqual((text, len(media)), (m + m, 2))
        self.assertEqual(media[0][:4], bytes([0x89, 0x50, 0x4E, 0x47]))  # (the decoded data: URL)
        # a message without images is flattened the same way: its text parts joined by newlines, not run together
        self.assertEqual(self._media([{"type": "text", "text": "a"}])[0], "a")
        self.assertEqual(self._media([{"type": "text", "text": "a"}, {"type": "text", "text": "b"}])[0], "a\nb")
        self.assertEqual(self._media([{"type": "text", "text": ""}, {"text": "b"}, {"type": "text", "text": "c"}])[0],
                         "b\nc")
        self.assertEqual(self._media([])[0], "")

    def test_remote_image_urls_are_rejected(self) -> None:
        with self.assertRaises(self.mk.BadRequest):
            self._media([{"type": "image_url", "image_url": {"url": "https://example.com/x.png"}}])

    def _server(self, images):
        srv = object.__new__(self.mk.Server)
        srv.eng, srv.im_start = _FakeVisionEngine(), self.IM
        srv.args = mock.Mock(ck_every=0, ck_msgs=1, ctx=4096)
        srv.vis_start, srv.vis_end, srv.img_pad = self.VS, self.VE, self.PAD
        srv.vision = mock.Mock()
        srv.vision.image.side_effect = lambda data: images[data]
        return srv

    def test_each_image_becomes_its_rows_between_vision_start_and_end(self) -> None:
        im = _FakeImage("k", 3, 2)
        srv = self._server({b"png": im})
        m = self.mk.MEDIA_MARKER
        toks, images = srv.tokenize_prompt(f"ab{m}c", [b"png"])
        self.assertEqual(toks, [ord("a"), ord("b"), self.VS] + [self.PAD] * 6 + [self.VE, ord("c")])
        self.assertEqual(images, [(3, im)])

    def test_a_different_image_at_the_same_place_is_not_reused(self) -> None:
        a, b = _FakeImage("a", 2, 2), _FakeImage("b", 2, 2)
        srv = self._server({b"A": a, b"B": b})
        m = self.mk.MEDIA_MARKER
        toks, images = srv.tokenize_prompt(f"x{m}yz", [b"A"])
        srv.load_prompt(toks, images)
        self.assertEqual(srv.eng.calls[-2], ("images", ((2, 2, 2),)))
        toks2, images2 = srv.tokenize_prompt(f"x{m}yz", [b"B"])
        self.assertEqual(toks2, toks)  # (the placeholders are alike)
        _, reused = srv.load_prompt(toks2, images2)
        self.assertEqual(reused, 0)  # (no checkpoint before the image: a full prefill)
        self.assertEqual(b.encoded, 1)

    def test_the_same_image_is_reused_and_a_rewind_into_it_feeds_it_again(self) -> None:
        a = _FakeImage("a", 4, 4)
        srv = self._server({b"A": a})
        m = self.mk.MEDIA_MARKER
        toks, images = srv.tokenize_prompt(f"x{m}yz", [b"A"])
        srv.load_prompt(toks, images)
        srv.eng.cks.append(10)  # a checkpoint inside the image (positions 2 .. 17)
        srv.eng.dn_c = len(srv.eng.seq) + 3
        toks2, images2 = srv.tokenize_prompt(f"x{m}yw", [b"A"])
        _, reused = srv.load_prompt(toks2, images2)
        self.assertEqual(reused, 10)
        self.assertEqual(srv.eng.calls[-3:-1], [("rewind", 10), ("images", ((2, 4, 4),))])

    def test_the_gpu_is_kept_warm_across_an_image_encode(self) -> None:
        """the engine stops its keep-warm load at the request's first GPU call; an image encoded on the CPU afterwards
        re-arms it first, so the GPU's clocks are up when the prefill launches (and only then: a cached image does not)"""
        a = _FakeImage("a", 2, 2)
        srv = self._server({b"A": a})
        a.log = srv.eng.calls
        m = self.mk.MEDIA_MARKER
        toks, images = srv.tokenize_prompt(f"x{m}yz", [b"A"])
        srv.load_prompt(toks, images)
        self.assertEqual([c[0] for c in srv.eng.calls], ["reset", "warm", "encode", "images", "prefill"])
        srv.eng.calls.clear()
        srv.eng.reset()  # (the image's rows are needed again; it is encoded already)
        srv.load_prompt(toks, images)
        self.assertEqual([c[0] for c in srv.eng.calls], ["reset", "reset", "images", "prefill"])


# === HTTP harness ===
# The HTTP handler end to end (make_handler on a local port) over a scripted engine: no GPU, no model.

TEMPLATE = ("{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{% endfor %}"
            "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}")


class _ScriptEngine(_FakeEngine):
    """generates a fixed text: one token per piece (the first from the prefill, two per decode cycle), then an
    end-of-generation token (eog=False: filler pieces forever); delay: seconds per decode cycle; fail_at: the decode
    cycle that raises, as the engine does on an error"""

    EOG, FILL = 900000, 900001

    def __init__(self, pieces, eog=True, delay=0.0, fail_at=None):
        super().__init__()
        self.pieces, self.eog, self.delay, self.fail_at = [p.encode() for p in pieces], eog, delay, fail_at
        self.i = self.cycles = 0

    def tokenize(self, text):
        return [ord(c) for c in text]

    def set_penalty(self, *a):
        pass

    def set_sampling(self, *a):
        pass

    def keep_warm(self, on=True):
        return 0.0

    def _next(self):
        self.i += 1
        if self.i <= len(self.pieces):
            return 1000 + self.i - 1
        return self.EOG if self.eog else self.FILL

    def prefill(self, toks, ck_at=-1, cks=(), images=()):
        super().prefill(toks, ck_at, cks)
        self.i = 0  # (each request replays the script)
        return self._next()

    def step(self):
        self.cycles += 1
        if self.cycles == self.fail_at:
            raise RuntimeError("decode: no finite logit (test)")
        time.sleep(self.delay)
        return [self._next(), self._next()]

    def is_eog(self, tok):
        return tok == self.EOG

    def piece(self, tok):
        return b" x" if tok == self.FILL else self.pieces[tok - 1000]


def _chat_server(mk, eng, ctx=4096):
    srv = object.__new__(mk.Server)
    srv.eng, srv.im_start, srv.vision = eng, None, None
    srv.template = mk.make_template(TEMPLATE)
    srv.args = types.SimpleNamespace(ctx=ctx, ck_every=0, ck_msgs=0, model="test.gguf", lib="libqwen38mk.so", kv="q8",
                                     draft=4, mtp=None, draft_vocab=None, mmproj=None)
    srv.lock, srv.warned, srv.model_id = threading.Lock(), set(), "test"
    return srv


class _Http:
    """make_handler(srv) on 127.0.0.1:<free port>"""

    def __init__(self, mk, srv):
        self.httpd = mk.HTTPServer(("127.0.0.1", 0), mk.make_handler(srv))  # (the server's class: lingering close)
        self.port = self.httpd.server_address[1]
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()

    def post(self, body, timeout=20):
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=timeout)
        c.request("POST", "/v1/chat/completions", json.dumps(body), {"Content-Type": "application/json"})
        r = c.getresponse()
        data = r.read()
        c.close()
        return r.status, data

    def raw(self, data, timeout=3.0, until=None):
        """sends raw bytes; returns what comes back until the server closes, `until` ends the data, or the timeout"""
        with socket.create_connection(("127.0.0.1", self.port), timeout=timeout) as s:
            s.sendall(data)
            out = b""
            try:
                while not (until and out.endswith(until)):
                    b = s.recv(65536)
                    if not b:
                        break
                    out += b
            except (socket.timeout, ConnectionResetError):  # (a reset after the answer: an unread request body)
                pass
            return out


def _request(body, headers=None):
    data = json.dumps(body).encode()
    h = {"Content-Type": "application/json", "Content-Length": str(len(data)), **(headers or {})}
    return (b"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\n" + b"".join(f"{k}: {v}\r\n".encode() for k, v in h.items())
            + b"\r\n" + data)


def _sse(raw):
    """the data: payloads of a chunked SSE response, decoded strictly (a status line inside the body fails here)"""
    head, _, body = raw.partition(b"\r\n\r\n")
    events, pos = [], 0
    while True:
        eol = body.index(b"\r\n", pos)
        n = int(body[pos:eol], 16)
        if n == 0:
            return head, events
        chunk = body[eol + 2:eol + 2 + n]
        assert body[eol + 2 + n:eol + 4 + n] == b"\r\n"
        events += [ln[6:] for ln in chunk.split(b"\n") if ln.startswith(b"data: ")]
        pos = eol + 4 + n


class _HttpCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.mk = _load_mk_server()

    def serve(self, eng, ctx=4096):
        srv = _chat_server(self.mk, eng, ctx)
        http_ = _Http(self.mk, srv)
        self.addCleanup(http_.close)
        return srv, http_


# === api: tool calls cut by max_tokens ===
TOOLS = [{"type": "function", "function": {"name": "read_file", "parameters": {
    "type": "object", "properties": {"path": {"type": "string"}}}}}]
CALL = "<tool_call>\n<function=read_file>\n<parameter=path>\n/etc/hostname\n</parameter>\n</function>\n</tool_call>"


class ToolCallCutTests(_HttpCase):
    """a tool call counts only with its closing tags: one cut by max_tokens is text and the reply ends with 'length' (a
    stream started it as it was generated: its arguments never become valid JSON)"""

    def test_only_a_closed_call_is_parsed(self) -> None:
        self.assertEqual(len(self.mk.parse_tool_calls(CALL, TOOLS)), 1)
        for n in range(len(CALL)):
            self.assertEqual(self.mk.parse_tool_calls(CALL[:n], TOOLS), [], CALL[:n])
            self.assertEqual(self.mk.unparsed_tool_call(CALL[:n]), CALL[CALL.find("<tool_call>"):n] if n >= 11 else "")
        self.assertEqual(self.mk.unparsed_tool_call(CALL + "\n" + CALL[:30]), CALL[:30])

    PIECES = (lambda t: [t[i:i + 3] for i in range(0, len(t), 3)])("Let me look.\n\n" + CALL)

    def _reply(self, max_tokens, stream):
        pieces = self.PIECES
        _, http_ = self.serve(_ScriptEngine(pieces))
        body = {"messages": [{"role": "user", "content": "hostname?"}], "tools": TOOLS, "max_tokens": max_tokens,
                "stream": stream}
        if not stream:
            status, data = http_.post(body)
            self.assertEqual(status, 200)
            o = json.loads(data)
            c = o["choices"][0]
            return pieces, c["finish_reason"], c["message"].get("content"), c["message"].get("tool_calls") or []
        _, events = _sse(http_.raw(_request(body), until=b"0\r\n\r\n"))
        self.assertEqual(events[-1], b"[DONE]")
        deltas = [c for e in events[:-1] for c in json.loads(e)["choices"]]
        content = "".join(d["delta"].get("content") or "" for d in deltas)
        calls = {}  # (accumulated by index, as the SDKs do)
        for tc in [tc for d in deltas for tc in d["delta"].get("tool_calls") or []]:
            c = calls.setdefault(tc["index"], {"id": tc.get("id"), "function": {"name": "", "arguments": ""}})
            for k in ("name", "arguments"):
                c["function"][k] += tc["function"].get(k) or ""
        calls = [calls[i] for i in sorted(calls)]
        return pieces, [d["finish_reason"] for d in deltas if d["finish_reason"]][-1], content, calls

    def test_a_complete_call_is_a_tool_call(self) -> None:
        for stream in (False, True):
            _, finish, content, calls = self._reply(500, stream)
            self.assertEqual((finish, content, len(calls)), ("tool_calls", "Let me look.", 1), stream)
            self.assertEqual(json.loads(calls[0]["function"]["arguments"]), {"path": "/etc/hostname"})

    def test_a_call_cut_by_max_tokens_is_text_and_length(self) -> None:
        n = len(self.PIECES)
        for cut in (1, 2, 4, 12):  # inside </tool_call>, </function>, the parameter
            for stream in (False, True):
                pieces, finish, content, calls = self._reply(n - cut, stream)
                if not stream:
                    self.assertEqual((finish, calls), ("length", []), cut)
                    self.assertEqual(content, "".join(pieces[:n - cut]), cut)  # (all of the generated text)
                    continue
                self.assertEqual((finish, content, len(calls)), ("length", "Let me look.", 1), cut)
                self.assertEqual(calls[0]["function"]["name"], "read_file")
                with self.assertRaises(ValueError):  # (never valid JSON: a client cannot run it)
                    json.loads(calls[0]["function"]["arguments"])


# === srv: image ownership, MK_MAX_IMAGES ===
class _FakeVisionLib:
    """libqwen38vision's C API over Python objects; using a freed image is recorded (the real library crashes)"""

    def __init__(self):
        self.live, self.freed, self.misuse, self.next = {}, [], [], 1

    def mkv_image_load(self, ctx, data, n):
        if data.startswith(b"bad"):
            return None
        h, self.next = self.next, self.next + 1
        self.live[h] = data
        return h

    def mkv_last_error(self):
        return b"cannot decode the image"

    def mkv_image_grid(self, h, nx, ny):
        nx._obj.value, ny._obj.value = 2, 2
        return 4

    def mkv_image_encode(self, ctx, h):
        if h not in self.live:
            self.misuse.append(("encode", h))
            return -1
        return 0

    def mkv_image_encode_ms(self, h):
        return 1.0

    def mkv_image_rows(self, h):
        if h not in self.live:
            self.misuse.append(("rows", h))
        return 4096 + 16 * (h or 0)

    def mkv_image_free(self, h):
        self.freed.append(self.live.pop(h))


class ImagePinTests(_HttpCase):
    """a request owns every image it references until it is done; at most MK_MAX_IMAGES images per request"""

    def serve_vision(self, cache=8):
        eng = _ScriptEngine(["ok"])
        srv, http_ = self.serve(eng)
        lib = _FakeVisionLib()
        v = object.__new__(self.mk.Vision)
        v.L, v.ctx, v.cache, v.cache_max = lib, 1, {}, cache
        srv.vision, srv.vis_start, srv.vis_end, srv.img_pad = v, 1001, 1002, 1003
        return v, lib, http_

    def _images(self, datas):
        parts = [{"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(d).decode()}}
                 for d in datas]
        return {"messages": [{"role": "user", "content": [{"type": "text", "text": "what?"}] + parts}], "max_tokens": 4}

    def test_more_images_than_the_cache_in_one_request(self) -> None:
        v, lib, http_ = self.serve_vision(cache=8)
        datas = [b"img%d" % i for i in range(9)]
        status, data = http_.post(self._images(datas))
        self.assertEqual((status, lib.misuse, lib.freed), (200, [], [b"img0"]), data[:300])  # (freed after the request)
        self.assertEqual(len(v.cache), 8)
        self.assertTrue(all(im.pins == 0 for im in v.cache.values()))
        status, _ = http_.post(self._images(datas))  # again: 8 cached, the first decoded again
        self.assertEqual((status, lib.misuse, len(v.cache)), (200, [], 8))

    def test_the_same_image_twice_in_a_request(self) -> None:
        v, lib, http_ = self.serve_vision(cache=1)
        status, _ = http_.post(self._images([b"a", b"b", b"a"]))
        self.assertEqual((status, lib.misuse, lib.freed, list(v.cache)), (200, [], [b"b"], list(v.cache)))
        self.assertEqual(len(v.cache), 1)

    def test_a_failed_request_releases_its_images(self) -> None:
        v, lib, http_ = self.serve_vision(cache=2)
        status, data = http_.post(self._images([b"a", b"b", b"c", b"bad"]))
        self.assertEqual(status, 400)
        self.assertIn(b"cannot decode the image", data)
        self.assertEqual((len(v.cache), lib.freed, lib.misuse), (2, [b"a"], []))
        self.assertTrue(all(im.pins == 0 for im in v.cache.values()))

    def test_too_many_images_is_a_400(self) -> None:
        v, lib, http_ = self.serve_vision()
        with mock.patch.object(self.mk, "MAX_IMAGES", 4):
            status, data = http_.post(self._images([b"i%d" % i for i in range(5)]))
            self.assertEqual(status, 400)
            self.assertIn(b"MK_MAX_IMAGES", data)
            self.assertEqual(lib.live, {})  # (refused before decoding any)
            self.assertEqual(http_.post(self._images([b"i%d" % i for i in range(4)]))[0], 200)


# === srv: engine errors in a stream ===
class StreamErrorTests(_HttpCase):
    """an engine error after the stream started is an SSE error event and [DONE], not a second status line"""

    def test_an_error_mid_stream_is_an_event(self) -> None:
        srv, http_ = self.serve(_ScriptEngine(["a"] * 40, fail_at=3))
        body = {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 30, "stream": True}
        raw = http_.raw(_request(body), until=b"0\r\n\r\n")
        self.assertEqual(raw.count(b"HTTP/1.1 "), 1, raw[-400:])
        head, events = _sse(raw)
        self.assertTrue(head.startswith(b"HTTP/1.1 200"))
        self.assertEqual(events[-1], b"[DONE]")
        err = json.loads(events[-2])["error"]
        self.assertEqual((err["type"], err["code"]), ("server_error", 500))
        self.assertIn("no finite logit", err["message"])
        self.assertTrue(any(json.loads(e)["choices"][0]["delta"].get("content") for e in events[1:-2]))  # (text first)
        srv.eng.fail_at = None
        self.assertEqual(http_.post({**body, "stream": False})[0], 200)  # (keeps serving)

    def test_an_error_before_the_stream_starts_is_a_status(self) -> None:
        srv, http_ = self.serve(_ScriptEngine(["a"] * 40, fail_at=1))
        status, data = http_.post({"messages": [{"role": "user", "content": "hi"}], "max_tokens": 30})
        self.assertEqual((status, json.loads(data)["error"]["type"]), (500, "server_error"))


# === srv: client disconnects ===
class DisconnectTests(_HttpCase):
    """a client that closes its connection stops the generation at the next engine cycle"""

    def _abandon(self, eng, body, ctx=1024):
        srv, http_ = self.serve(eng, ctx)
        s = socket.create_connection(("127.0.0.1", http_.port))
        s.sendall(_request(body))
        t0 = time.time()
        while eng.cycles < 5 and time.time() - t0 < 5:
            time.sleep(0.01)
        s.close()
        time.sleep(0.3)
        c1 = eng.cycles
        time.sleep(0.3)
        self.assertEqual(eng.cycles, c1, "still generating after the client left")
        t = time.time()
        status, _ = http_.post({"messages": [{"role": "user", "content": "next"}], "max_tokens": 4})
        self.assertEqual(status, 200)  # (the engine is free again)
        self.assertLess(time.time() - t, 2.0)

    def test_non_streaming(self) -> None:
        self._abandon(_ScriptEngine(["a"], eog=False, delay=0.01),
                      {"messages": [{"role": "user", "content": "long"}], "max_tokens": 100000})

    def test_streaming_while_a_tool_call_is_held_back(self) -> None:
        self._abandon(_ScriptEngine(["<tool_call>\n<function=read_file>\n<parameter=path>\n"], eog=False, delay=0.01),
                      {"messages": [{"role": "user", "content": "long"}], "tools": TOOLS, "max_tokens": 100000,
                       "stream": True})


# === srv: request body limits ===
class BodyLimitTests(_HttpCase):
    """Content-Length: required, numeric, at most MK_MAX_BODY_MB; a refused body is not read and the connection closes"""

    HEAD = b"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\n"

    def _status(self, raw):
        line = raw.split(b"\r\n", 1)[0]
        return int(line.split()[1]) if line.startswith(b"HTTP/1.1 ") else None

    def test_bad_lengths(self) -> None:
        _, http_ = self.serve(_ScriptEngine(["ok"]))
        body = b'{"messages": [{"role": "user", "content": "hi"}]}'
        for cl, want in ((b"abc", 400), (b"-5", 400), (b"1e3", 400), (b" 12x", 400), (b"99999999999999999999999", 413)):
            raw = http_.raw(self.HEAD + b"Content-Length: " + cl + b"\r\n\r\n" + body)
            self.assertEqual(self._status(raw), want, (cl, raw[:200]))
            self.assertIn(b"Connection: close", raw)
        raw = http_.raw(self.HEAD + b"\r\n" + body)  # no Content-Length
        self.assertEqual(self._status(raw), 411, raw[:200])
        raw = http_.raw(self.HEAD + b"Transfer-Encoding: chunked\r\n\r\n%x\r\n%s\r\n0\r\n\r\n" % (len(body), body), until=b"}")
        self.assertEqual(self._status(raw), 200, raw[:200])  # (chunked bodies: tests/test_server_api.py)
        raw = http_.raw(self.HEAD + b"Content-Length: 5\r\nContent-Length: 6\r\n\r\n" + body)
        self.assertEqual(self._status(raw), 400, raw[:200])
        self.assertEqual(http_.post({"messages": [{"role": "user", "content": "hi"}], "max_tokens": 2})[0], 200)

    def test_an_oversized_body_is_refused_unread(self) -> None:
        _, http_ = self.serve(_ScriptEngine(["ok"]))
        with mock.patch.object(self.mk, "MAX_BODY_MB", 1):
            t = time.time()
            raw = http_.raw(self.HEAD + b"Content-Length: %d\r\n\r\n" % (2 << 20))  # (the body never comes)
            self.assertEqual(self._status(raw), 413, raw[:200])
            self.assertIn(b"MK_MAX_BODY_MB", raw)
            self.assertLess(time.time() - t, 2.0)
            raw = http_.raw(self.HEAD + b"Content-Length: %d\r\nExpect: 100-continue\r\n\r\n" % (2 << 20))
            self.assertEqual(self._status(raw), 413, raw[:200])  # (no 100 Continue first)
            body = json.dumps({"messages": [{"role": "user", "content": "x" * 1000}], "max_tokens": 2}).encode()
            raw = http_.raw(self.HEAD + b"Content-Length: %d\r\n\r\n" % len(body) + body, until=b"}")
            self.assertEqual(self._status(raw), 200, raw[:200])


# === srv: disconnect during a long prefill ===
class _ChunkedEngine(_ScriptEngine):
    """_ScriptEngine whose prefill takes n_chunks chunks of chunk_s seconds and, as the engine does, stops between two
    of them once mk_cancel(h, 1) was called (the real Engine.cancel_when over a fake library): nothing of the prefill is
    kept and PrefillCancelled is raised"""

    def __init__(self, mk, pieces, n_chunks=20, chunk_s=0.1, **kw):
        super().__init__(pieces, **kw)
        self.mk, self.n_chunks, self.chunk_s = mk, n_chunks, chunk_s
        self.flag, self.chunks_run, self.cancelled = 0, 0, 0
        self._cancel, self.h = True, None
        self.L = types.SimpleNamespace(mk_cancel=self._mk_cancel)

    def _mk_cancel(self, h, on):
        self.flag = on
        return 0

    def cancel_when(self, gone, every=0.05):
        return self.mk.Engine.cancel_when(self, gone, every)

    def prefill(self, toks, ck_at=-1, cks=(), images=()):
        for c in range(self.n_chunks):
            if c and self.flag:
                self.cancelled += 1
                self.calls.append(("cancelled", len(self.seq), c))
                raise self.mk.PrefillCancelled("test")
            time.sleep(self.chunk_s)
            self.chunks_run += 1
        return super().prefill(toks, ck_at, cks, images)


class PrefillDisconnectTests(_HttpCase):
    """a client that leaves during a long prefill stops it at the next chunk; nothing of it is kept"""

    def _abandon(self, stream):
        eng = _ChunkedEngine(self.mk, ["a", "b"])
        srv, http_ = self.serve(eng)
        body = {"messages": [{"role": "user", "content": "long prompt"}], "max_tokens": 8, "stream": stream}
        s = socket.create_connection(("127.0.0.1", http_.port))
        s.sendall(_request(body))
        time.sleep(0.35)  # (3-4 chunks of 20 in)
        s.close()
        t0 = time.time()
        while not eng.cancelled and time.time() - t0 < 3:
            time.sleep(0.01)
        self.assertEqual(eng.cancelled, 1, "the prefill ran on after the client left")
        self.assertLess(time.time() - t0, 0.5)  # (watcher 50 ms + the chunk in progress)
        self.assertLess(eng.chunks_run, 10)
        self.assertEqual(eng.sequence(), [])  # (nothing of it kept)
        eng.n_chunks = 1
        t = time.time()
        status, data = http_.post({**body, "stream": False})
        self.assertEqual(status, 200, data[:300])
        self.assertLess(time.time() - t, 1.0)
        self.assertEqual(json.loads(data)["choices"][0]["message"]["content"], "ab")
        prefills = [c for c in eng.calls if c[0] == "prefill"]
        self.assertEqual(prefills[-1][1], 0)  # (the next prompt from position 0: no reuse of the abandoned one)
        self.assertEqual(eng.flag, 0)  # (cleared after the call)

    def test_non_streaming(self) -> None:
        self._abandon(False)

    def test_streaming(self) -> None:
        self._abandon(True)

    def test_a_client_that_stays_is_not_cancelled(self) -> None:
        eng = _ChunkedEngine(self.mk, ["a", "b"], n_chunks=6, chunk_s=0.05)
        srv, http_ = self.serve(eng)
        status, data = http_.post({"messages": [{"role": "user", "content": "hi"}], "max_tokens": 8})
        self.assertEqual((status, eng.cancelled, eng.chunks_run, eng.flag), (200, 0, 6, 0), data[:300])


# === srv: a 4xx while the client is still uploading ===
class LingerTests(_HttpCase):
    """after an early 4xx the server half-closes and drains the rest of the upload (bounded) instead of closing with
    unread data, which resets the connection: the client, sending its whole body first as http.client does, then reads
    the answer"""

    HEAD = BodyLimitTests.HEAD

    def _upload(self, http_, declared, send, rate=None):
        """declares `declared` bytes, sends `send` of them (MiB/s rate or as fast as possible), then reads the answer;
        returns (status line or None, error)"""
        with socket.create_connection(("127.0.0.1", http_.port), timeout=10) as s:
            s.sendall(self.HEAD + b"Content-Length: %d\r\n\r\n" % declared)
            time.sleep(0.2)  # (the 4xx is out before the body starts)
            chunk, t0 = b"x" * (1 << 20), time.time()
            try:
                for i in range(send >> 20):
                    s.sendall(chunk)
                    if rate:
                        time.sleep(max(0.0, t0 + (i + 1) / rate - time.time()))
                s.shutdown(socket.SHUT_WR)
                raw = b""
                while True:
                    b = s.recv(65536)
                    if not b:
                        break
                    raw += b
            except OSError as e:  # (a reset: the answer is lost)
                return None, repr(e)
            return raw.split(b"\r\n", 1)[0], None

    def test_the_413_survives_the_upload(self) -> None:
        _, http_ = self.serve(_ScriptEngine(["ok"]))
        with mock.patch.object(self.mk, "MAX_BODY_MB", 1):
            line, err = self._upload(http_, 64 << 20, 64 << 20)
            self.assertEqual(((line or b"")[:12], err), (b"HTTP/1.1 413", None))
            line, err = self._upload(http_, 24 << 20, 24 << 20, rate=40)  # (slowly: 0.6 s)
            self.assertEqual(((line or b"")[:12], err), (b"HTTP/1.1 413", None))
        self.assertEqual(http_.post({"messages": [{"role": "user", "content": "hi"}], "max_tokens": 2})[0], 200)

    def test_the_drain_is_bounded(self) -> None:
        _, http_ = self.serve(_ScriptEngine(["ok"]))
        with mock.patch.object(self.mk, "MAX_BODY_MB", 1), mock.patch.object(self.mk, "LINGER_BYTES", 4 << 20):
            t = time.time()
            line, err = self._upload(http_, 256 << 20, 256 << 20)
            self.assertIsNone(line)  # (reset after LINGER_BYTES, 4 MiB: past the drain's bound the client gets a reset)
            self.assertLess(time.time() - t, 5.0)
        with mock.patch.object(self.mk, "MAX_BODY_MB", 1), mock.patch.object(self.mk, "LINGER_S", 0.5):
            t = time.time()
            line, err = self._upload(http_, 256 << 20, 256 << 20, rate=8)  # (would take 32 s)
            self.assertIsNone(line)
            self.assertLess(time.time() - t, 3.0)


# === api: stop strings ===
class StopStringTests(_HttpCase):
    """`stop`: the reply ends before the first stop string (matched on the raw generated text, across tokens), the rest
    of the engine cycle's tokens dropped; a stream never sends a stop string's first characters"""

    def _cut(self, text, stops):
        at = [(text.find(s), k) for k, s in enumerate(stops) if s and text.find(s) >= 0]
        if not at:
            return text, None
        i, k = min(at)
        return text[:i], stops[k]

    def test_matcher_equals_the_cut_text_for_any_chunking(self) -> None:
        import random as rnd
        r = rnd.Random(5)
        for _ in range(3000):
            text = "".join(r.choice("ab c\n") for _ in range(r.randint(0, 30)))
            stops = ["".join(r.choice("ab c\n") for _ in range(r.randint(1, 4))) for _ in range(r.randint(1, 3))]
            m, out, i = self.mk.StopMatcher(stops), "", 0
            while i < len(text):
                n = r.randint(1, 5)
                out += m.feed(text[i:i + n])
                i += n
                if m.word is not None:
                    break
            out += m.feed("", final=True)
            want, word = self._cut(text, stops)
            if len(stops) == 1 or word is None:
                self.assertEqual((out, m.word), (want, word), (text, stops))
            else:  # (as llama-server: the first stop string to COMPLETE wins, so with overlapping ones, e.g. "bcc" and
                # "c" on "bcc", the cut depends on the token boundaries; it is always at an occurrence, none before it)
                self.assertTrue(text.startswith(out) and text.startswith(m.word, len(out)), (text, stops, out))
                self.assertFalse(any(s in out for s in stops), (text, stops, out))

    def test_stop_values(self) -> None:
        self.assertEqual(self.mk.stop_strings(None), [])
        self.assertEqual(self.mk.stop_strings("x"), ["x"])
        self.assertEqual(self.mk.stop_strings(["a", "", "b"]), ["a", "b"])
        for bad in (5, ["a", 1], {"a": 1}, ["x"] * 65, ["x" * 1025]):
            with self.assertRaises(self.mk.BadRequest):
                self.mk.stop_strings(bad)

    PIECES = ["Hello", " wor", "ld", ". The", " END", " is", " near", "."]

    def _reply(self, stop, stream, pieces=None, template=None, max_tokens=50):
        eng = _ScriptEngine(pieces or self.PIECES)
        srv, http_ = self.serve(eng)
        if template:
            srv.template = self.mk.make_template(template)
        body = {"messages": [{"role": "user", "content": "hi"}], "max_tokens": max_tokens, "stream": stream}
        if stop is not None:
            body["stop"] = stop
        if not stream:
            status, data = http_.post(body)
            self.assertEqual(status, 200, data[:300])
            o = json.loads(data)
            m = o["choices"][0]["message"]
            return (m.get("content") or "", m.get("reasoning_content") or "", o["choices"][0]["finish_reason"],
                    o["usage"]["completion_tokens"], srv)
        _, events = _sse(http_.raw(_request({**body, "stream_options": {"include_usage": True}}), until=b"0\r\n\r\n"))
        self.assertEqual(events[-1], b"[DONE]")
        objs = [json.loads(e) for e in events[:-1]]
        deltas = [c for o in objs for c in o["choices"]]
        content = "".join(d["delta"].get("content") or "" for d in deltas)
        reasoning = "".join(d["delta"].get("reasoning_content") or "" for d in deltas)
        for s in ([stop] if isinstance(stop, str) else stop or []):
            for d in deltas:  # (no delta carries a stop string's start that was then cut)
                self.assertNotIn(s, d["delta"].get("content") or "")
        finish = [d["finish_reason"] for d in deltas if d["finish_reason"]][-1]
        return content, reasoning, finish, objs[-1]["usage"]["completion_tokens"], srv

    def test_the_reply_is_the_no_stop_reply_cut_at_the_first_occurrence(self) -> None:
        full = "".join(self.PIECES)
        for stream in (False, True):
            base = self._reply(None, stream)
            self.assertEqual(base[:3], (full, "", "stop"))
            for stop in ("END", ["near", " wor"], "ld. T", ["o"], "Hello", "zzz", ["is n", "ear"]):
                content, _, finish, n, srv = self._reply(stop, stream)
                want, word = self._cut(full, [stop] if isinstance(stop, str) else stop)
                self.assertEqual(content, want.rstrip(), (stop, stream))  # (the end of a reply is trimmed)
                self.assertEqual(finish, "stop", (stop, stream))
                self.assertEqual(srv.stop_word, word, (stop, stream))
                if word is not None:  # (tokens up to the one that completed the stop string)
                    k = next(i for i in range(1, len(self.PIECES) + 1) if word in "".join(self.PIECES[:i])
                             and "".join(self.PIECES[:i]).find(word) == len(want))
                    self.assertEqual(n, k, (stop, stream))

    def test_a_stop_in_a_cycle_drops_the_cycles_later_tokens(self) -> None:
        # _ScriptEngine: token 1 from the prefill, then two per cycle (2-3, 4-5, ...): ". The" is token 4, " END" 5
        for stream in (False, True):
            content, _, finish, n, srv = self._reply("The", stream)  # (token 4: token 5 of the same cycle dropped)
            self.assertEqual((content, finish, n, srv.eng.cycles), ("Hello world.", "stop", 4, 2), stream)
            content, _, finish, n, srv = self._reply(" EN", stream)  # (completed inside token 5)
            self.assertEqual((content, finish, n, srv.eng.cycles), ("Hello world. The", "stop", 5, 2), stream)

    def test_max_tokens_releases_held_text(self) -> None:
        for stream in (False, True):
            content, _, finish, n, _ = self._reply("ld!", stream, max_tokens=3)  # ("ld" held, then the end)
            self.assertEqual((content, finish, n), ("Hello world", "length", 3))

    def test_stop_applies_to_the_reasoning_too(self) -> None:
        think = TEMPLATE.replace("assistant\n{% endif %}", "assistant\n<think>\n{% endif %}")
        pieces = ["Let me", " think", " hard.", "</think>", "\n\nAnswer", ": 4"]
        for stream in (False, True):
            content, reasoning, finish, _, _ = self._reply(None, stream, pieces, think)
            self.assertEqual((content, reasoning.strip(), finish), ("Answer: 4", "Let me think hard.", "stop"))
            content, reasoning, finish, _, _ = self._reply("hard", stream, pieces, think)
            self.assertEqual((content, reasoning.strip(), finish), ("", "Let me think", "stop"))
            content, reasoning, finish, _, _ = self._reply(": ", stream, pieces, think)
            self.assertEqual((content, reasoning.strip(), finish), ("Answer", "Let me think hard.", "stop"))

    def test_a_stop_inside_a_tool_call_leaves_text(self) -> None:
        pieces = ToolCallCutTests.PIECES
        _, http_ = self.serve(_ScriptEngine(pieces))
        body = {"messages": [{"role": "user", "content": "hostname?"}], "tools": TOOLS, "max_tokens": 500}
        o = json.loads(http_.post({**body, "stop": "hostname\n"})[1])
        c = o["choices"][0]
        self.assertEqual((c["finish_reason"], c["message"].get("tool_calls")), ("stop", None))
        self.assertTrue(c["message"]["content"].endswith("/etc/"), c["message"]["content"])
        o = json.loads(http_.post({**body, "stop": "</tool_call>\n"})[1])  # (never matches: the call is whole)
        self.assertEqual(o["choices"][0]["finish_reason"], "tool_calls")

    def test_bad_stop_is_a_400(self) -> None:
        _, http_ = self.serve(_ScriptEngine(["ok"]))
        for stream in (False, True):
            status, data = http_.post({"messages": [{"role": "user", "content": "hi"}], "stop": [1], "stream": stream})
            self.assertEqual(status, 400, data[:200])
            self.assertIn(b"stop must be", data)


# === api: reporting ===
class ReportingTests(_HttpCase):
    """/health, /props, /v1/models, /slots, /metrics in llama-server's shapes (plus the engine block)"""

    def _get(self, http_, path):
        c = http_.httpd.server_address[1]
        conn = http.client.HTTPConnection("127.0.0.1", c, timeout=10)
        conn.request("GET", path)
        r = conn.getresponse()
        data = r.read()
        conn.close()
        return r.status, data, r

    def test_props_models_slots(self) -> None:
        srv, http_ = self.serve(_ScriptEngine(["ok"]))
        self.assertEqual(self._get(http_, "/health")[:2], (200, b'{"status": "ok"}'))
        status, data, _ = self._get(http_, "/props")
        p = json.loads(data)
        self.assertEqual(status, 200)
        self.assertEqual((p["total_slots"], p["model_alias"], p["default_generation_settings"]["n_ctx"]), (1, "test", 4096))
        self.assertEqual(p["modalities"], {"vision": False, "video": False, "audio": False})
        self.assertEqual(p["default_generation_settings"]["params"]["temperature"], self.mk.DEF_TEMP)
        self.assertEqual((p["build_info"], p["engine"]["busy"], p["engine"]["kv"]), ("unknown", False, "q8"))
        srv.video_ok = True  # (set by the video track when its decoder works)
        srv.eng.build = {"commit": "abc1234", "flags": "-DMK_WV=1", "cuda": "12.8"}
        p = json.loads(self._get(http_, "/props")[1])
        self.assertEqual((p["modalities"]["video"], p["build_info"]), (True, "abc1234 -DMK_WV=1"))
        self.assertEqual(p["engine"]["build"]["cuda"], "12.8")
        m = json.loads(self._get(http_, "/v1/models")[1])
        self.assertEqual((m["data"][0]["id"], m["data"][0]["meta"]["n_ctx"], m["models"][0]["name"]), ("test", 4096, "test"))
        self.assertEqual(json.loads(self._get(http_, "/slots")[1])[0]["is_processing"], False)
        self.assertEqual(self._get(http_, "/slots?fail_on_no_slot=1")[0], 200)
        c = http.client.HTTPConnection("127.0.0.1", http_.port, timeout=10)
        c.request("POST", "/props", b"{}", {"Content-Type": "application/json"})
        self.assertEqual(c.getresponse().status, 501)

    def test_busy_slot_and_metrics(self) -> None:
        eng = _ScriptEngine(["a"] * 30, delay=0.05)
        srv, http_ = self.serve(eng)
        t = threading.Thread(target=http_.post, args=({"messages": [{"role": "user", "content": "hi"}], "max_tokens": 20},))
        t.start()
        t0 = time.time()
        while not srv.busy() and time.time() - t0 < 5:
            time.sleep(0.01)
        self.assertTrue(json.loads(self._get(http_, "/slots")[1])[0]["is_processing"])
        self.assertEqual(self._get(http_, "/slots?fail_on_no_slot=1")[0], 503)
        t.join()
        http_.post({"messages": [{"role": "user", "content": "hi"}], "max_tokens": 2, "stop": 5})  # (a 400)
        status, data, r = self._get(http_, "/metrics")
        self.assertEqual(status, 200)
        self.assertTrue(r.getheader("Content-Type").startswith("text/plain"))
        vals = dict(ln.split(" ", 1) for ln in data.decode().splitlines() if ln and not ln.startswith("#"))
        self.assertEqual(vals["llamacpp:tokens_predicted_total"], "20")
        self.assertEqual((vals["megakernel:requests_total"], vals["megakernel:requests_bad_total"]), ("2", "1"))
        self.assertEqual(vals["llamacpp:requests_processing"], "0")
        self.assertIn("# TYPE llamacpp:prompt_tokens_total counter", data.decode())

    def test_loading_answers_503_until_the_swap(self) -> None:
        httpd = self.mk.HTTPServer(("127.0.0.1", 0), self.mk.make_loading_handler())
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        self.addCleanup(httpd.server_close)
        self.addCleanup(httpd.shutdown)
        port = httpd.server_address[1]
        c = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
        c.request("GET", "/health")
        r = c.getresponse()
        self.assertEqual((r.status, json.loads(r.read())["error"]["message"]), (503, "Loading model"))
        raw = _Http.raw(types.SimpleNamespace(port=port), _request({"messages": []}))
        self.assertTrue(raw.startswith(b"HTTP/1.1 503"), raw[:100])
        httpd.RequestHandlerClass = self.mk.make_handler(_chat_server(self.mk, _ScriptEngine(["ok"])))
        c = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
        c.request("GET", "/health")
        self.assertEqual(c.getresponse().status, 200)


# === api: request parameters ===
class RequestParamTests(_HttpCase):
    """fields the server cannot honour are a 400 (never ignored silently); what it can do alone, it does"""

    MSG = [{"role": "user", "content": "hi"}]

    def test_unsupported_fields_are_a_400(self) -> None:
        _, http_ = self.serve(_ScriptEngine(["ok"]))
        for extra, field in (({"n": 2}, "n:"), ({"logprobs": True}, "logprobs"), ({"top_logprobs": 3}, "logprobs"),
                             ({"logit_bias": {"5": 10}}, "logit_bias"),
                             ({"response_format": {"type": "json_object"}}, "response_format"),
                             ({"response_format": {"type": "json_schema", "json_schema": {}}}, "response_format"),
                             ({"grammar": "root ::= x"}, "grammar"), ({"json_schema": {"type": "object"}}, "json_schema"),
                             ({"tool_choice": "required"}, "tool_choice"),
                             ({"tool_choice": {"type": "function", "function": {"name": "f"}}}, "tool_choice"),
                             ({"functions": [{"name": "f"}]}, "functions"), ({"ignore_eos": True}, "ignore_eos"),
                             ({"mirostat": 2}, "mirostat"), ({"typical_p": 0.9}, "typical_p"), ({"xtc_probability": 0.5}, "xtc"),
                             ({"dry_multiplier": 0.8}, "dry_multiplier"), ({"n_probs": 5}, "n_probs"),
                             ({"temperature": "hot"}, "must be numbers"), ({"max_tokens": "many"}, "max_tokens"),
                             ({"messages": "hi"}, "messages"), ({"add_generation_prompt": False}, "add_generation_prompt")):
            for stream in (False, True):
                status, data = http_.post({"messages": self.MSG, "stream": stream, **extra})
                self.assertEqual(status, 400, (extra, stream, data[:200]))
                self.assertIn(field, json.loads(data)["error"]["message"], extra)
        for raw in (b"[1, 2]", b"\"x\"", b"{bad"):
            c = http.client.HTTPConnection("127.0.0.1", http_.port, timeout=10)
            c.request("POST", "/v1/chat/completions", raw, {"Content-Type": "application/json"})
            self.assertEqual(c.getresponse().status, 400, raw)

    def test_harmless_and_neutral_fields_are_accepted(self) -> None:
        _, http_ = self.serve(_ScriptEngine(["ok"]))
        webui = {"n": 1, "response_format": {"type": "text"}, "tool_choice": "auto", "parallel_tool_calls": True,
                 "user": "u", "metadata": {"a": "b"}, "store": False, "samplers": "edkypmxt", "typ_p": 1, "typical_p": 1.0,
                 "xtc_probability": 0, "dry_multiplier": 0, "mirostat": 0, "top_n_sigma": -1, "dynatemp_range": 0,
                 "cache_prompt": True, "logprobs": False, "top_logprobs": None, "logit_bias": {}, "timings_per_token": False}
        status, data = http_.post({"messages": self.MSG, **webui})
        self.assertEqual(status, 200, data[:300])

    def test_tool_choice_none_returns_calls_as_text(self) -> None:
        _, http_ = self.serve(_ScriptEngine(ToolCallCutTests.PIECES))
        body = {"messages": self.MSG, "tools": TOOLS, "max_tokens": 500}
        c = json.loads(http_.post(body)[1])["choices"][0]
        self.assertEqual((c["finish_reason"], len(c["message"]["tool_calls"])), ("tool_calls", 1))
        c = json.loads(http_.post({**body, "tool_choice": "none"})[1])["choices"][0]
        self.assertEqual((c["finish_reason"], c["message"].get("tool_calls")), ("stop", None))
        self.assertEqual(c["message"]["content"], "Let me look.\n\n" + CALL)

    def test_parallel_tool_calls_false_keeps_the_first_call(self) -> None:
        two = "Two.\n\n" + CALL + "\n" + CALL.replace("hostname", "hosts")
        _, http_ = self.serve(_ScriptEngine([two[i:i + 4] for i in range(0, len(two), 4)]))
        body = {"messages": self.MSG, "tools": TOOLS, "max_tokens": 500}
        self.assertEqual(len(json.loads(http_.post(body)[1])["choices"][0]["message"]["tool_calls"]), 2)
        calls = json.loads(http_.post({**body, "parallel_tool_calls": False})[1])["choices"][0]["message"]["tool_calls"]
        self.assertEqual([json.loads(c["function"]["arguments"])["path"] for c in calls], ["/etc/hostname"])

    def test_top_level_reasoning_effort(self) -> None:
        srv = _chat_server(self.mk, _ScriptEngine(["ok"]))
        srv.template = self.mk.make_template("{{ reasoning_effort|default('-') }}|{{ enable_thinking|default('-') }}")
        r = lambda body: srv.render({"messages": self.MSG, **body})[0]  # noqa: E731
        self.assertEqual(r({}), "-|-")
        self.assertEqual(r({"reasoning_effort": "none"}), "-|False")
        self.assertEqual(r({"reasoning_effort": "minimal"}), "low|-")
        self.assertEqual(r({"reasoning_effort": "high"}), "high|-")
        self.assertEqual(r({"reasoning_effort": "high", "chat_template_kwargs": {"reasoning_effort": "medium"}}), "medium|-")


# === api: Anthropic Messages ===
def _sse_events(raw):
    """(event name, data object) of a chunked Anthropic SSE response"""
    head, payloads = b"", []
    head, _, body = raw.partition(b"\r\n\r\n")
    text, pos = b"", 0
    while True:
        eol = body.index(b"\r\n", pos)
        n = int(body[pos:eol], 16)
        if n == 0:
            break
        text += body[eol + 2:eol + 2 + n]
        pos = eol + 4 + n
    for ev in text.decode().split("\n\n"):
        if ev.strip():
            lines = dict(ln.split(": ", 1) for ln in ev.split("\n"))
            payloads.append((lines["event"], json.loads(lines["data"])))
    return head, payloads


class AnthropicTests(_HttpCase):
    """POST /v1/messages (+ stream) and /v1/messages/count_tokens over the chat path"""

    def test_request_translation(self) -> None:
        a2c = self.mk.anthropic_to_chat
        body = {"model": "claude-x", "max_tokens": 100, "stop_sequences": ["END"], "temperature": 0.2, "top_k": 5,
                "system": [{"type": "text", "text": "x-anthropic-billing-header: cc_version=2.1; cch=a5145;You are X."},
                           {"type": "text", "text": "Be brief.", "cache_control": {"type": "ephemeral"}}],
                "tools": [{"name": "read_file", "description": "reads", "input_schema": {"type": "object"}}],
                "tool_choice": {"type": "auto", "disable_parallel_tool_use": True},
                "thinking": {"type": "enabled", "budget_tokens": 2048},
                "messages": [
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": [{"type": "thinking", "thinking": "hmm", "signature": ""},
                                                      {"type": "text", "text": "Let me look."},
                                                      {"type": "tool_use", "id": "toolu_1", "name": "read_file",
                                                       "input": {"path": "/x"}}]},
                    {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_1",
                                                  "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]},
                                                 {"type": "text", "text": "and?"}]},
                    {"role": "user", "content": [{"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                                                         "data": "iVBO"}}]}]}
        c = a2c(body)
        m = c["messages"]
        self.assertEqual(m[0], {"role": "system", "content": "x-anthropic-billing-header: cc_version=2.1; cch=fffff;You are X."
                                                                "\n\nBe brief."})
        self.assertEqual(m[1], {"role": "user", "content": "hi"})
        self.assertEqual((m[2]["content"], m[2]["reasoning_content"], m[2]["tool_calls"][0]["function"]),
                         ("Let me look.", "hmm", {"name": "read_file", "arguments": '{"path": "/x"}'}))
        self.assertEqual(m[3], {"role": "tool", "tool_call_id": "toolu_1", "content": "a\nb"})  # (results first)
        self.assertEqual(m[4], {"role": "user", "content": "and?"})
        self.assertEqual(m[5]["content"], [{"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBO"}}])
        self.assertEqual((c["max_tokens"], c["stop"], c["temperature"], c["top_k"], c["tool_choice"],
                          c["parallel_tool_calls"], c["chat_template_kwargs"]),
                         (100, ["END"], 0.2, 5, "auto", False, {"enable_thinking": True}))
        self.assertEqual(c["tools"][0]["function"], {"name": "read_file", "description": "reads",
                                                     "parameters": {"type": "object"}})
        self.assertEqual(a2c({"messages": []})["chat_template_kwargs"], {"enable_thinking": False})  # (absent: off)
        self.assertEqual(a2c({"messages": [], "thinking": {"type": "disabled"}})["chat_template_kwargs"],
                         {"enable_thinking": False})
        self.assertEqual(a2c({"messages": [], "chat_template_kwargs": {"enable_thinking": True}})["chat_template_kwargs"],
                         {"enable_thinking": True})
        for bad in ({"messages": "x"}, {"messages": [{"role": "tool", "content": "x"}]},
                    {"messages": [{"role": "user", "content": ["text"]}]}, {"messages": [], "tools": [{"type": "bash"}]},
                    {"messages": [], "tool_choice": {"type": "any"}}, {"messages": [], "thinking": {"type": "on"}},
                    {"messages": [], "stop_sequences": "x" * 2000}):
            with self.assertRaises(self.mk.BadRequest, msg=str(bad)):
                a2c(bad)

    def _msg(self, eng, body, template=None):
        srv, http_ = self.serve(eng)
        if template:
            srv.template = self.mk.make_template(template)
        if body.get("stream"):
            raw = http_.raw(_request(body).replace(b"/v1/chat/completions", b"/v1/messages"), until=b"0\r\n\r\n")
            return int(raw.split()[1]), _sse_events(raw)[1]
        c = http.client.HTTPConnection("127.0.0.1", http_.port, timeout=20)
        c.request("POST", "/v1/messages", json.dumps(body), {"Content-Type": "application/json", "x-api-key": "k",
                                                              "anthropic-version": "2023-06-01"})
        r = c.getresponse()
        return r.status, json.loads(r.read())

    THINK = TEMPLATE.replace("assistant\n{% endif %}",
                             "assistant\n{% if enable_thinking %}<think>\n{% endif %}{% endif %}")

    def test_non_streaming_replies(self) -> None:
        base = {"model": "claude-x", "max_tokens": 50, "messages": [{"role": "user", "content": "hi"}]}
        status, o = self._msg(_ScriptEngine(["Hello", " there", "."]), base)
        self.assertEqual(status, 200, o)
        self.assertEqual((o["type"], o["role"], o["model"], o["content"], o["stop_reason"], o["stop_sequence"]),
                         ("message", "assistant", "claude-x", [{"type": "text", "text": "Hello there."}], "end_turn", None))
        self.assertEqual(o["usage"]["output_tokens"], 4)  # (3 pieces + the end of generation)
        self.assertEqual((o["usage"]["cache_read_input_tokens"], o["usage"]["input_tokens"] > 0), (0, True))
        _, o = self._msg(_ScriptEngine(["a"] * 20, eog=False), {**base, "max_tokens": 5})
        self.assertEqual((o["stop_reason"], o["usage"]["output_tokens"]), ("max_tokens", 5))
        _, o = self._msg(_ScriptEngine(["Hello", " there", "."]), {**base, "stop_sequences": ["ere"]})
        self.assertEqual((o["content"][0]["text"], o["stop_reason"], o["stop_sequence"]), ("Hello th", "stop_sequence", "ere"))
        _, o = self._msg(_ScriptEngine(ToolCallCutTests.PIECES),
                         {**base, "tools": [{"name": "read_file", "input_schema": TOOLS[0]["function"]["parameters"]}]})
        self.assertEqual(o["stop_reason"], "tool_use")
        self.assertEqual(o["content"][0], {"type": "text", "text": "Let me look."})
        tu = o["content"][1]
        self.assertEqual((tu["type"], tu["name"], tu["input"], tu["id"][:6]), ("tool_use", "read_file", {"path": "/etc/hostname"}, "toolu_"))
        _, o = self._msg(_ScriptEngine(["Hmm", ".</think>", "\n\nFour"]), {**base, "thinking": {"type": "enabled", "budget_tokens": 1024}},
                         self.THINK)
        self.assertEqual(o["content"], [{"type": "thinking", "thinking": "Hmm.", "signature": ""}, {"type": "text", "text": "Four"}])

    def test_thinking_budget_is_noted_as_not_enforced(self) -> None:
        def keys(thinking):
            notes = []
            self.mk.anthropic_to_chat({"max_tokens": 50, "messages": [{"role": "user", "content": "hi"}], "thinking": thinking},
                                      notes)
            return [k for k, _ in notes]
        self.assertEqual(keys({"type": "enabled", "budget_tokens": 1024}), ["anthropic-thinking-budget"])
        self.assertEqual(keys({"type": "adaptive", "budget_tokens": 1024}), ["anthropic-thinking-budget"])
        self.assertEqual(keys({"type": "enabled"}), [])
        self.assertEqual(keys({"type": "disabled", "budget_tokens": 1024}), [])
        self.assertEqual(keys(None), [])

    def test_streaming_events(self) -> None:
        base = {"model": "claude-x", "max_tokens": 50, "messages": [{"role": "user", "content": "hi"}], "stream": True}
        _, ev = self._msg(_ScriptEngine(["\n", "Hmm", ".\n", "</think>", "\n\nLet me look.\n\n"]
                                        + [CALL[i:i + 3] for i in range(0, len(CALL), 3)]),
                          {**base, "thinking": {"type": "enabled", "budget_tokens": 1024},
                           "tools": [{"name": "read_file", "input_schema": TOOLS[0]["function"]["parameters"]}]}, self.THINK)
        names = [n for n, _ in ev]
        self.assertEqual(names[:3], ["message_start", "ping", "content_block_start"])
        self.assertEqual(names[-2:], ["message_delta", "message_stop"])
        starts = [d["content_block"]["type"] for n, d in ev if n == "content_block_start"]
        self.assertEqual(starts, ["thinking", "text", "tool_use"])
        self.assertEqual(names.count("content_block_stop"), 3)
        idx = [d["index"] for n, d in ev if n.startswith("content_block")]
        self.assertEqual(sorted(set(idx)), [0, 1, 2])
        think = "".join(d["delta"]["thinking"] for n, d in ev if n == "content_block_delta" and d["delta"]["type"] == "thinking_delta")
        text = "".join(d["delta"]["text"] for n, d in ev if n == "content_block_delta" and d["delta"]["type"] == "text_delta")
        js = "".join(d["delta"]["partial_json"] for n, d in ev if n == "content_block_delta" and d["delta"]["type"] == "input_json_delta")
        self.assertEqual((think, text, json.loads(js)), ("Hmm.", "Let me look.", {"path": "/etc/hostname"}))  # (trimmed)
        md = ev[-2][1]
        self.assertEqual((md["delta"]["stop_reason"], md["usage"]["output_tokens"] > 0), ("tool_use", True))
        self.assertEqual(ev[0][1]["message"]["usage"]["output_tokens"], 0)

    def test_errors_and_count_tokens(self) -> None:
        srv, http_ = self.serve(_ScriptEngine(["ok"]))

        def post(path, body):
            c = http.client.HTTPConnection("127.0.0.1", http_.port, timeout=10)
            c.request("POST", path, json.dumps(body) if not isinstance(body, bytes) else body)
            r = c.getresponse()
            return r.status, json.loads(r.read())
        for body in ({"messages": "x"}, {"max_tokens": 5, "messages": [], "tool_choice": {"type": "any"}},
                     {"max_tokens": "x", "messages": []}, b"[1]"):
            for stream in (False, True):
                if isinstance(body, dict):
                    body = {**body, "stream": stream}
                status, o = post("/v1/messages", body)
                self.assertEqual((status, o["type"], o["error"]["type"]), (400, "error", "invalid_request_error"), body)
        msgs = {"system": "Be brief.", "messages": [{"role": "user", "content": "How many tokens?"}]}
        status, o = post("/v1/messages/count_tokens", msgs)
        want = len(srv.render(self.mk.anthropic_to_chat(msgs))[0])  # (a character tokenizer)
        self.assertEqual((status, o), (200, {"input_tokens": want}))
        status, o = post("/v1/messages/batches", {})
        self.assertEqual((status, o["type"]), (404, "error"))


# === api: routes without generation, unsupported routes ===
class ExtraRouteTests(_HttpCase):
    def _post(self, http_, path, body):
        c = http.client.HTTPConnection("127.0.0.1", http_.port, timeout=10)
        c.request("POST", path, json.dumps(body))
        r = c.getresponse()
        return r.status, json.loads(r.read())

    def test_tokenize_detokenize_template_input_tokens(self) -> None:
        srv, http_ = self.serve(_ScriptEngine(["ok"]))
        srv.eng.n_vocab = 200
        srv.eng.piece = lambda t: bytes([t]) if t < 128 else b"\xff"
        self.assertEqual(self._post(http_, "/tokenize", {"content": "hi"}), (200, {"tokens": [104, 105]}))
        self.assertEqual(self._post(http_, "/tokenize", {"content": ["h", 5, 300], "with_pieces": True})[1]["tokens"],
                         [{"id": 104, "piece": "h"}, {"id": 5, "piece": "\x05"}, {"id": 300, "piece": ""}])
        self.assertEqual(self._post(http_, "/tokenize", {"content": [150], "with_pieces": True})[1]["tokens"],
                         [{"id": 150, "piece": [255]}])
        self.assertEqual(self._post(http_, "/detokenize", {"tokens": [104, 105]}), (200, {"content": "hi"}))
        self.assertEqual(self._post(http_, "/detokenize", {"tokens": [104, 999]})[0], 400)
        body = {"messages": [{"role": "user", "content": "hi"}]}
        prompt = "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n"
        self.assertEqual(self._post(http_, "/apply-template", body), (200, {"prompt": prompt}))
        self.assertEqual(self._post(http_, "/v1/chat/completions/input_tokens", body),
                         (200, {"input_tokens": len(prompt), "object": "response.input_tokens"}))
        srv.eng.n_vocab = 0  # (a library without mk_n_vocab)
        self.assertEqual(self._post(http_, "/detokenize", {"tokens": [104]})[0], 400)

    def test_malformed_fields_are_400(self) -> None:
        _, http_ = self.serve(_ScriptEngine(["ok"]))
        hi = [{"role": "user", "content": "hi"}]
        for path, body in (("/v1/chat/completions", {"messages": [1]}), ("/apply-template", {"messages": [None]}),
                           ("/v1/chat/completions/input_tokens", {"messages": [1]}),
                           ("/v1/chat/completions", {"messages": hi, "repetition_penalty": "x"}),
                           ("/v1/chat/completions", {"messages": hi, "max_tokens": 0}),
                           ("/v1/chat/completions", {"messages": hi, "max_tokens": 0, "stream": True}),
                           ("/v1/messages", {"messages": hi, "max_tokens": 0})):
            self.assertEqual(self._post(http_, path, body)[0], 400, (path, body))

    def test_http10_stream_is_not_chunked(self) -> None:
        _, http_ = self.serve(_ScriptEngine(["ok"]))
        b = json.dumps({"messages": [{"role": "user", "content": "hi"}], "stream": True}).encode()
        with socket.create_connection(("127.0.0.1", http_.port), timeout=10) as s:
            s.sendall(b"POST /v1/chat/completions HTTP/1.0\r\nHost: 127.0.0.1\r\nConnection: keep-alive\r\n"
                      b"Content-Length: %d\r\n\r\n" % len(b) + b)
            data = b""
            while True:  # (the body ends when the server closes the connection)
                d = s.recv(65536)
                if not d:
                    break
                data += d
        head, _, body = data.partition(b"\r\n\r\n")
        self.assertNotIn(b"transfer-encoding", head.lower())
        self.assertIn(b"connection: close", head.lower())
        self.assertTrue(body.startswith(b"data: {") and body.endswith(b"data: [DONE]\n\n"), body[-80:])

    def test_unsupported_routes_are_501_unknown_404(self) -> None:
        _, http_ = self.serve(_ScriptEngine(["ok"]))
        for path in ("/v1/completions", "/completion", "/v1/embeddings", "/infill", "/v1/rerank", "/v1/responses/compact",
                     "/slots/0?action=save"):
            status, o = self._post(http_, path, {})
            self.assertEqual(status, 501, path)
            self.assertIn(path.split("?")[0], o["error"]["message"])
        self.assertEqual(self._post(http_, "/nope", {})[0], 404)


# === api: --chat-template-file / --chat-template ===
# tests/templates: the chat templates of real GGUFs (tokenizer.chat_template, exact bytes): qwen38_unsloth.jinja =
# unsloth's Qwen3.8-27B GGUFs (developer role, merged system messages, tool-only turns), qwen38_original.jinja = Qwen's
# original, as a fine-tune's GGUF ships it (Swift-1.5-Qwen3.8-27B)
TEMPLATES = ROOT / "tests" / "templates"


class _GgufEngine(_ScriptEngine):
    """what Server.__init__ needs of mk.Engine (patched in): the GGUF's chat template, a warm-up, stats"""

    has_mtp = True

    def __init__(self, gguf_template):
        super().__init__(["ok"])
        self.gguf_template = gguf_template

    def chat_template(self):
        return self.gguf_template

    def stats(self):
        return (0, 0, 0)


class ChatTemplateOptionTests(_HttpCase):
    UNSLOTH = (TEMPLATES / "qwen38_unsloth.jinja").read_bytes().decode()
    ORIGINAL = (TEMPLATES / "qwen38_original.jinja").read_bytes().decode()
    CALL = {"id": "c1", "type": "function", "function": {"name": "read_file", "arguments": '{"path": "/etc/hostname"}'}}

    def _tmpdir(self):
        t = tempfile.TemporaryDirectory()
        self.addCleanup(t.cleanup)
        return t.name

    def _file(self, text, crlf=False, bom=False):
        p = Path(self._tmpdir()) / "t.jinja"
        p.write_bytes((b"\xef\xbb\xbf" if bom else b"") + (text.replace("\n", "\r\n") if crlf else text).encode())
        return str(p)

    def _server(self, gguf_template, kind=None, value=None):
        """Server(args) as main builds it (mk.Engine replaced), the template option read by chat_template_source"""
        args = types.SimpleNamespace(lib="libqwen38mk.so", model="test.gguf", ctx=4096, draft=4, draft_vocab=None, kv="q8",
                                     no_mtp=False, mtp=None, mmproj=None, ck_every=0, ck_msgs=0, chat_template_src=None)
        if kind:
            args.chat_template_src = self.mk.chat_template_source(kind, value)
        with mock.patch.object(self.mk, "Engine", lambda *a: _GgufEngine(gguf_template)):
            srv = self.mk.Server(args)
        http_ = _Http(self.mk, srv)
        self.addCleanup(http_.close)
        return srv, http_

    def _post(self, http_, path, body):
        c = http.client.HTTPConnection("127.0.0.1", http_.port, timeout=20)
        c.request("POST", path, json.dumps(body), {"Content-Type": "application/json"})
        r = c.getresponse()
        return r.status, json.loads(r.read())

    def _props(self, http_):
        c = http.client.HTTPConnection("127.0.0.1", http_.port, timeout=20)
        c.request("GET", "/props")
        return json.loads(c.getresponse().read())

    def _prompt_of_last_prefill(self, eng):
        n = [c for c in eng.calls if c[0] == "prefill"][-1][2]
        return "".join(chr(t) for t in eng.seq[:n])  # (a character tokenizer; one request per server here)

    def test_the_file_replaces_the_gguf_template_everywhere(self) -> None:
        mine = ("{% for m in messages %}[{{ m.role }}]{{ m.content }}\n{% endfor %}"
                "{% if add_generation_prompt %}[assistant]{% endif %}")
        path = self._file(mine, crlf=True, bom=True)  # (a file saved on Windows: CRLF line ends and a BOM)
        srv, http_ = self._server(TEMPLATE, "file", path)
        self.assertEqual(srv.template_src, mine)
        self.assertEqual(self._props(http_)["chat_template"], mine)
        body = {"messages": [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "hi"}],
                "max_tokens": 5}
        prompt = "[system]Be brief.\n[user]hi\n[assistant]"
        self.assertEqual(self._post(http_, "/apply-template", body), (200, {"prompt": prompt}))
        self.assertEqual(self._post(http_, "/v1/chat/completions/input_tokens", body)[1]["input_tokens"], len(prompt))
        anth = {"model": "m", "system": "Be brief.", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 5}
        self.assertEqual(self._post(http_, "/v1/messages/count_tokens", anth), (200, {"input_tokens": len(prompt)}))
        status, o = self._post(http_, "/v1/chat/completions", body)
        self.assertEqual((status, o["choices"][0]["message"]["content"]), (200, "ok"))
        self.assertEqual(self._prompt_of_last_prefill(srv.eng), prompt)
        srv2, http2 = self._server(TEMPLATE, "file", path)
        status, o = self._post(http2, "/v1/messages", anth)
        self.assertEqual((status, o["content"]), (200, [{"type": "text", "text": "ok"}]))
        self.assertEqual(self._prompt_of_last_prefill(srv2.eng), prompt)
        # without the option: the GGUF's
        srv3, http3 = self._server(TEMPLATE)
        self.assertEqual((srv3.template_src, self._props(http3)["chat_template"]), (TEMPLATE, TEMPLATE))
        self.assertEqual(self._post(http3, "/apply-template", body)[1]["prompt"],
                         "<|im_start|>system\nBe brief.<|im_end|>\n<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n")

    def test_requests_the_original_template_refuses_render_under_unsloths(self) -> None:
        """a fine-tune shipping Qwen's original template: 400s for a second system message, a turn with no user query,
        reasoning_effort high; with --chat-template-file of unsloth's template the same requests render. A developer
        message is a system message under either (llama-server's mapping for templates without <|channel|>)"""
        user = {"role": "user", "content": "hi"}
        dev = {"messages": [{"role": "developer", "content": "Be brief."}, user]}
        bodies = {
            "System message must be at the beginning": {"messages": [{"role": "system", "content": "a"},
                                                                      {"role": "system", "content": "b"}, user]},
            "No user query found": {"messages": [{"role": "system", "content": "s"},
                                                 {"role": "assistant", "content": "", "tool_calls": [self.CALL]},
                                                 {"role": "tool", "tool_call_id": "c1", "content": "host"}],
                                    "tools": TOOLS},
            "Unexpected reasoning effort high": {"messages": [user], "reasoning_effort": "high"},
        }
        _, http_ = self._server(self.ORIGINAL)
        for why, body in bodies.items():
            status, o = self._post(http_, "/v1/chat/completions", {**body, "max_tokens": 5})
            self.assertEqual(status, 400, why)
            self.assertIn(why, o["error"]["message"])
        status, o = self._post(http_, "/v1/chat/completions", {**dev, "max_tokens": 5})
        self.assertEqual(status, 200, o)
        as_system = {"messages": [{"role": "system", "content": "Be brief."}, user]}
        self.assertEqual(self._post(http_, "/apply-template", dev)[1]["prompt"],
                         self._post(http_, "/apply-template", as_system)[1]["prompt"])
        srv, http_ = self._server(self.ORIGINAL, "file", str(TEMPLATES / "qwen38_unsloth.jinja"))
        self.assertEqual(self._props(http_)["chat_template"], self.UNSLOTH)
        for why, body in bodies.items():
            status, o = self._post(http_, "/apply-template", body)
            self.assertEqual(status, 200, why)
            self.assertTrue(o["prompt"].endswith("<|im_start|>assistant\n<think>\n"), why)
        p = self._post(http_, "/apply-template", dev)[1]["prompt"]
        self.assertTrue(p.startswith("<|im_start|>system\n") and "Be brief.<|im_end|>" in p, p)  # (developer = system)
        p = self._post(http_, "/apply-template", bodies["Unexpected reasoning effort high"])[1]["prompt"]
        self.assertIn("Reasoning effort is set to xhigh", p)  # (unsloth's template maps high to xhigh)
        status, o = self._post(http_, "/v1/chat/completions", {**bodies["No user query found"], "max_tokens": 5})
        self.assertEqual(status, 200, o)
        # (a template with GPT-OSS's channels has a developer role of its own: the message keeps it, as in llama-server)
        _, http_ = self._server("{# <|channel|> #}{% for m in messages %}<|{{ m.role }}|>{{ m.content }}{% endfor %}")
        self.assertEqual(self._post(http_, "/apply-template", dev)[1]["prompt"], "<|developer|>Be brief.<|user|>hi")

    def test_a_template_that_fails_on_the_requests_data_is_a_400(self) -> None:
        # (the template's own Python error, e.g. a value it concatenates of another type: the request's fault, not a 500)
        _, http_ = self._server("{% if tools %}{{ tools[0] + 1 }}{% endif %}{% for m in messages %}{{ m.content }}{% endfor %}")
        status, o = self._post(http_, "/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}],
                                                                "tools": TOOLS, "max_tokens": 5})
        self.assertEqual(status, 400, o)
        self.assertIn("chat template: TypeError", o["error"]["message"])

    def test_the_gguf_template_as_a_file_renders_byte_identical_prompts(self) -> None:
        """the GGUF's own template given as a file (written with CRLF line ends, as an extraction on Windows writes it):
        every prompt is byte-identical to the one rendered from the GGUF"""
        gguf, _ = self._server(self.UNSLOTH)
        filed, http_ = self._server(self.UNSLOTH, "file", self._file(self.UNSLOTH, crlf=True))
        self.assertEqual((filed.template_src, self._props(http_)["chat_template"]), (self.UNSLOTH, self.UNSLOTH))
        user, sys_ = {"role": "user", "content": "What is 2+2?"}, {"role": "system", "content": "Be brief."}
        turn = [sys_, user, {"role": "assistant", "content": "4", "reasoning_content": "Add them."},
                {"role": "user", "content": "And 3+3?"}]
        tool_turn = [user, {"role": "assistant", "content": "Let me look.", "tool_calls": [self.CALL]},
                     {"role": "tool", "tool_call_id": "c1", "content": "box"}]
        bodies = [{"messages": [user]}, {"messages": turn},
                  {"messages": turn, "chat_template_kwargs": {"preserve_thinking": False}},
                  {"messages": [sys_, user], "chat_template_kwargs": {"enable_thinking": False}},
                  {"messages": tool_turn, "tools": TOOLS}]
        bodies += [{"messages": turn, "reasoning_effort": e} for e in ("none", "minimal", "low", "medium", "high", "xhigh")]
        bodies.append(self.mk.anthropic_to_chat({"system": "Be brief.", "max_tokens": 5,
                                                 "thinking": {"type": "enabled", "budget_tokens": 64},
                                                 "messages": [{"role": "user", "content": "hi"}]}))
        for body in bodies:
            self.assertEqual(filed.render(body), gguf.render(body), body)

    def test_a_gguf_without_a_template_gets_chatml_and_chatml_is_built_in(self) -> None:
        srv, http_ = self._server("")
        self.assertEqual(srv.template_src, self.mk.CHATML_TEMPLATE)
        chatml, hi = "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n", {"messages": [{"role": "user", "content": "hi"}]}
        self.assertEqual(self._post(http_, "/apply-template", hi)[1]["prompt"], chatml)
        srv, http_ = self._server(self.UNSLOTH, "inline", "chatml")
        self.assertEqual(self._props(http_)["chat_template"], self.mk.CHATML_TEMPLATE)
        self.assertEqual(self._post(http_, "/apply-template", hi)[1]["prompt"], chatml)
        srv, _ = self._server(self.UNSLOTH, "inline", TEMPLATE)  # (Jinja source inline)
        self.assertEqual(srv.template_src, TEMPLATE)

    def _main(self, *argv):
        """mk.main() with these options up to the point where it opens the port (HTTPServer stops it there); returns
        (exit code or None if it got there, stderr, the chat_template_source calls)"""
        class Reached(Exception):
            pass
        err = io.StringIO()
        spy = mock.Mock(wraps=self.mk.chat_template_source)
        with mock.patch.object(sys, "argv", ["mk_server.py", "-m", "test.gguf", *argv]), \
                mock.patch.object(self.mk, "HTTPServer", side_effect=Reached), \
                mock.patch.object(self.mk, "chat_template_source", spy), contextlib.redirect_stderr(err):
            try:
                self.mk.main()
            except SystemExit as e:
                return e.code, err.getvalue(), spy.call_args_list
            except Reached:
                return None, err.getvalue(), spy.call_args_list
        self.fail("main() returned")

    def test_startup_errors_name_the_problem_before_the_port_opens(self) -> None:
        d = Path(self._tmpdir())
        (d / "syntax.jinja").write_text("{% for m in messages %}{{ m.content }}", encoding="utf-8")
        (d / "raises.jinja").write_text("{{ raise_exception('no system messages here') }}", encoding="utf-8")
        (d / "latin1.jinja").write_bytes("{{ messages }} caf\xe9".encode("latin-1"))
        (d / "ok.jinja").write_text(TEMPLATE, encoding="utf-8")
        cases = [(("--chat-template-file", str(d / "missing.jinja")), "--chat-template-file: cannot read"),
                 (("--chat-template-file", str(d / "syntax.jinja")), "does not compile"),
                 (("--chat-template-file", str(d / "raises.jinja")), "no system messages here"),
                 (("--chat-template-file", str(d / "latin1.jinja")), "is not UTF-8"),
                 (("--chat-template", "llama3"), "the only built-in name is chatml"),
                 (("--chat-template", "{{ nope("), "--chat-template: the template does not compile"),
                 (("--chat-template", "chatml", "--chat-template-file", str(d / "syntax.jinja")), "does not compile")]
        for argv, want in cases:
            code, err, _ = self._main(*argv)
            self.assertEqual(code, 2, argv)
            self.assertIn(want, err, argv)
        # the last of the two options wins (one setting, as in llama-server)
        code, err, calls = self._main("--chat-template-file", str(d / "syntax.jinja"), "--chat-template", "chatml")
        self.assertEqual((code, [c.args for c in calls]), (None, [("inline", "chatml")]), err)
        code, err, calls = self._main("--chat-template", "llama3", "--chat-template-file", str(d / "ok.jinja"))
        self.assertEqual((code, [c.args for c in calls]), (None, [("file", str(d / "ok.jinja"))]), err)
        self.assertIn("chat template: " + str(d / "ok.jinja"), err)
        self.assertEqual(self._main()[0:3:2], (None, []))  # (no option: the GGUF's, nothing read)

    def test_agents_keep_one_checkpoint_slot_for_the_engines_own_sequence(self) -> None:
        # (its 1024 positions run the warm-up only; each slot has a pool of its own, MK_SLOT_CK)
        keys = ("MK_CK_SLOTS", "MK_PF_CAP", "MK_EXACT")
        clean = {k: v for k, v in os.environ.items() if k not in keys}
        for argv, preset, want in ((("--agents", "2"), None, "1"), (("--parallel", "2"), None, "1"),
                                   (("--agents", "2"), "4", "4"), ((), None, None)):
            with mock.patch.dict(os.environ, {**clean, **({"MK_CK_SLOTS": preset} if preset else {})}, clear=True):
                code, err, _ = self._main(*argv)
                self.assertEqual((code, os.environ.get("MK_CK_SLOTS")), (None, want), (argv, preset, err))


# === con: the console (log lines with request numbers, the live status line) ===
class _Tty(io.StringIO):
    def isatty(self):
        return True


def _screen(out):
    """what a terminal shows after `out`: a carriage return goes back to the start of the row, a newline starts one"""
    rows, col = [[]], 0
    for ch in out:
        if ch == "\r":
            col = 0
        elif ch == "\n":
            rows.append([])
            col = 0
        else:
            row = rows[-1]
            if col < len(row):
                row[col] = ch
            else:
                row.append(ch)
            col += 1
    return ["".join(r).rstrip() for r in rows]


class ConsoleTests(unittest.TestCase):
    IM = 1

    @classmethod
    def setUpClass(cls) -> None:
        cls.mk = _load_mk_server()

    def _run(self, stream, steps, columns=120):
        con = self.mk.Console()
        with mock.patch.object(sys, "stderr", stream), \
                mock.patch.object(self.mk.shutil, "get_terminal_size", return_value=types.SimpleNamespace(columns=columns)):
            for kind, text in steps:
                (con.line if kind == "line" else con.status)(text)
        return stream.getvalue()

    def test_off_a_terminal_only_the_log_lines_are_written(self) -> None:
        out = self._run(io.StringIO(), [("status", "#1 prefill"), ("line", "a"), ("status", "#1 decoding"), ("line", "b"),
                                        ("status", "")])
        self.assertEqual(out, "a\nb\n")

    def test_on_a_terminal_the_status_line_is_rewritten_in_place_under_the_log_lines(self) -> None:
        steps = [("status", "#1 prefill 10 / 100 tok"), ("status", "#1 prefill 50 / 100 tok"), ("line", "[mk-server] #1 prompt"),
                 ("status", "#1 decoding")]
        out = self._run(_Tty(), steps)
        self.assertEqual(_screen(out), ["[mk-server] #1 prompt", "#1 decoding"])  # (the shorter text padded over)
        self.assertNotIn("\x1b", out)  # (no escape sequences: any console)
        out = self._run(_Tty(), steps + [("line", "[mk-server] #1 reply"), ("status", "")])
        self.assertEqual(_screen(out), ["[mk-server] #1 prompt", "[mk-server] #1 reply", ""])

    def test_the_status_line_is_cut_to_the_terminal_width(self) -> None:
        text = "#1 prefill 123,456 / 200,000 tok (61%), 1,600 tok/s"
        out = self._run(_Tty(), [("status", text)], columns=30)
        self.assertEqual(_screen(out), [text[:29].rstrip()])  # (a full row would wrap, and \r returns to its last row only)
        self.assertLessEqual(max(len(seg) for seg in out.split("\r")), 29)

    def _server(self, every):
        srv = object.__new__(self.mk.Server)
        srv.eng, srv.im_start = _FakeEngine(), self.IM
        srv.args = mock.Mock(ck_every=every, ck_msgs=1)
        return srv

    def test_the_prompt_line_says_how_the_cache_was_used(self) -> None:
        IM = self.IM
        first = [IM, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, IM, 5, 5, IM, 9, 9]
        srv = self._server(8)
        srv.load_prompt(first)
        self.assertEqual(srv.cache_note, "nothing cached")
        srv.load_prompt(srv.eng.sequence() + [IM, 6, IM, 9])
        self.assertEqual(srv.cache_note, "the prompt matches 19 of the 19 cached tok: continued from there")
        srv = self._server(8)
        srv.load_prompt(first)
        srv.eng.dn_c = len(srv.eng.seq) + 5  # (decoding went on: no append)
        srv.load_prompt(first[:14] + [6, 6, IM, 9, 9])
        self.assertEqual(srv.cache_note, "the prompt matches 14 of the 19 cached tok: resumed at the checkpoint at 12")
        srv = self._server(0)
        srv.load_prompt([IM, 5, 5, IM, 9])
        srv.eng.dn_c += 3
        srv.load_prompt([IM, 6, 6, IM, 9])
        self.assertEqual(srv.cache_note, "CACHE MISS: the prompt matches 1 of the 6 cached tok, no checkpoint before that")

    def test_progress_texts(self) -> None:
        srv = object.__new__(self.mk.Server)
        srv.rid = 3
        self.assertEqual(srv.prefill_text(100, (40, 100), 2.0), "#3 prefill 40 / 100 tok (40%), 20 tok/s, ~3.00 s left | 2.00 s")
        self.assertEqual(srv.prefill_text(100, (0, 100), 0.3), "#3 prefill 0 / 100 tok (0%) | 0.30 s")
        for prog in (None, (0, 0)):  # (a library without mk_prefill_progress; a prefill without a batched part)
            self.assertEqual(srv.prefill_text(100, prog, 1.5), "#3 prefill 100 tok | 1.50 s")
        self.assertEqual(srv.decode_text(120, {"thinking": 100, "answer": 19, "tool call": 0}, 40, 119, 1.2),
                         "#3 decoding 120 tok, 100.0 tok/s, 3.0 tok/cycle | thinking 100, answer 19 | 1.20 s")
        self.assertEqual([self.mk.secs(s) for s in (0.4, 12.34, 125)], ["0.40 s", "12.3 s", "2 min 05 s"])

    def test_the_engine_reads_the_librarys_progress_counters(self) -> None:
        eng = object.__new__(self.mk.Engine)
        eng.h, eng._progress = None, True

        def prog(h, done, total):  # (ctypes.byref arguments: ._obj is the c_int)
            done._obj.value, total._obj.value = 5, 9
        eng.L = types.SimpleNamespace(mk_prefill_progress=prog)
        self.assertEqual(eng.prefill_progress(), (5, 9))
        eng._progress = False  # (a library before mk_prefill_progress)
        self.assertIsNone(eng.prefill_progress())

    def test_a_reset_connection_is_not_reported_other_errors_are(self) -> None:
        httpd = object.__new__(self.mk.HTTPServer)

        def report(exc):
            err = io.StringIO()
            with mock.patch.object(sys, "stderr", err), mock.patch.object(self.mk, "CON", self.mk.Console()):
                try:
                    raise exc
                except Exception:
                    httpd.handle_error(None, ("127.0.0.1", 5))
            return err.getvalue()
        for e in (ConnectionResetError(10054, "reset"), ConnectionAbortedError(), BrokenPipeError()):
            self.assertEqual(report(e), "", e)
        out = report(ValueError("boom"))
        self.assertIn("error on the connection from ('127.0.0.1', 5)", out)
        self.assertIn("ValueError: boom", out)


class ConsoleHttpTests(_HttpCase):
    def _log(self, fn, stream=None):
        err = io.StringIO() if stream is None else stream
        with mock.patch.object(sys, "stderr", err), mock.patch.object(self.mk, "CON", self.mk.Console()):
            fn()
        return err.getvalue()

    def test_each_request_logs_what_it_is_its_prompt_and_its_reply(self) -> None:
        srv, http_ = self.serve(_ScriptEngine(["a", "b"]))
        msgs = [{"role": "user", "content": "hi"}]

        def run():
            self.assertEqual(http_.post({"messages": msgs, "model": "m1", "max_tokens": 8})[0], 200)
            more = msgs + [{"role": "assistant", "content": "ab"}, {"role": "user", "content": "more"}]
            self.assertEqual(http_.post({"messages": more, "stream": True})[0], 200)
            self.assertEqual(http_.post({"messages": msgs, "top_k": "x"})[0], 400)
        lines = [ln.split(" ", 2)[2] for ln in self._log(run).splitlines() if ln.startswith("[mk-server] ")]
        self.assertEqual(len(lines), 7, lines)
        n = int(lines[0].split()[0][1:])
        samp = "temp 1 top_p 0.95 top_k 20 min_p 0"
        want = [rf"#{n} chat 'm1': \d+ tok, 1 message \| no thinking, max_tokens 8 \| {samp}",
                rf"#{n} prompt: 0 cached \+ \d+ prefilled in [\d.]+ s \([\d,]+ tok/s\) \| nothing cached",
                rf"#{n} reply: 3 tok in [\d.]+ s = [\d.]+ tok/s \| answer 2 \| 2.00 tok/cycle \| finish stop \| request [\d.]+ s",
                rf"#{n + 1} chat stream: \d+ tok, 3 messages \| no thinking \| {samp}",
                rf"#{n + 1} prompt: (\d+) cached \+ \d+ prefilled .* \| the prompt matches \1 of the \d+ cached tok: "
                r"continued from there",
                rf"#{n + 1} reply: 3 tok in .* \| finish stop \| request [\d.]+ s",
                rf"#{n + 2} 400: temperature, top_p, top_k, min_p and seed must be numbers"]
        for w, ln in zip(want, lines):
            self.assertRegex(ln, "^" + w + "$")

    def test_draft_counts_are_per_reply_across_an_engine_reset(self) -> None:
        """MKEngine::reset (a full re-prefill) zeroes the engine's counters: the reply line and /metrics count each reply's
        decode cycles anyway"""
        class Eng(_ScriptEngine):
            st = (0, 0, 0)

            def stats(self):
                return self.st

            def reset(self):
                super().reset()
                self.st = (0, 0, 0)

            def step(self):
                c, d, a = self.st
                self.st = (c + 1, d + 3, a + 1)  # (3 drafts per cycle, 1 accepted)
                return super().step()
        srv, http_ = self.serve(Eng(["a", "b", "c", "d", "e"]))

        def run():
            for content in ("hi", "other"):  # (the second prompt diverges early: no checkpoint, the engine is reset)
                self.assertEqual(http_.post({"messages": [{"role": "user", "content": content}]})[0], 200)
        out = self._log(run)
        replies = [ln for ln in out.splitlines() if " reply: " in ln]
        self.assertEqual(len(replies), 2, out)
        for ln in replies:
            self.assertIn("| 2.00 tok/cycle, 33% of 9 drafts accepted |", ln)
        self.assertIn("CACHE MISS", out)
        c = http.client.HTTPConnection("127.0.0.1", http_.port, timeout=10)
        c.request("GET", "/metrics")
        vals = dict(ln.split(" ", 1) for ln in c.getresponse().read().decode().splitlines() if ln and not ln.startswith("#"))
        self.assertEqual([vals[f"megakernel:{k}"] for k in ("decode_cycles_total", "draft_tokens_total",
                                                            "draft_tokens_accepted_total")], ["6", "18", "6"])

    def test_a_terminal_shows_a_live_status_line_and_it_is_gone_after_the_request(self) -> None:
        class Eng(_ChunkedEngine):
            def prefill_progress(self):
                return self.chunks_run * 1000, self.n_chunks * 1000
        eng = Eng(self.mk, list("abcdefghij"), n_chunks=6, chunk_s=0.1, delay=0.1)
        srv, http_ = self.serve(eng)

        def run():
            self.assertEqual(http_.post({"messages": [{"role": "user", "content": "hi"}]})[0], 200)
            t = time.time()
            while srv.busy() and time.time() - t < 3:  # (the status line is cleared after the reply is sent)
                time.sleep(0.01)
        out = self._log(run, _Tty())
        self.assertRegex(out, r"\r#\d+ prefill [1-5],000 / 6,000 tok \(\d+%\), [\d,]+ tok/s, ~[\d.]+ s left \| [\d.]+ s")
        self.assertRegex(out, r"\r#\d+ decoding \d+ tok, [\d.]+ tok/s, 2.0 tok/cycle \| answer \d+ \| [\d.]+ s")
        screen = _screen(out)
        self.assertEqual(screen[-1], "", screen)  # (the status line is gone)
        self.assertEqual([r.split()[2].startswith("#") for r in screen[:-1]], [True] * 3, screen)
        self.assertTrue(all(r.startswith("[mk-server] ") for r in screen[:-1]), screen)


class _StoreEngine(_ScriptEngine):
    """the host store's semantics (MKEngine::side_begin / park): snapshots of the sequence, its committed position and its
    newest checkpoints within a byte budget (PT bytes per token, STATE of recurrent state, CK per checkpoint)"""

    store = True
    PT, STATE, CK = 1000, 1000, 500

    def __init__(self, pieces, **kw):
        super().__init__(pieces, **kw)
        self.budget, self.entries, self.next_id, self.kept, self.side_ok, self.ops = 0, {}, 1, None, True, []

    def store_budget(self, nbytes):
        self.budget = nbytes

    def store_info(self):
        used = sum(e[3] for e in self.entries.values())
        return {"budget": self.budget, "allocated": used, "used": used, "parked": len(self.entries), "last_bytes": 0,
                "last_ms": 0.0, "side_ms": 0.0, "kv_token_bytes": self.PT}

    def kv_sizes(self):
        return self.PT, 0, self.STATE, self.CK

    def park_need(self, k):
        return len(self.seq) * self.PT + self.STATE + min(k, len(self.cks)) * self.CK

    def park(self, k):
        n = self.park_need(k)
        if self.store_info()["used"] + n > self.budget:
            return None
        pid, self.next_id = self.next_id, self.next_id + 1
        self.entries[pid] = (list(self.seq), self.dn_c, sorted(self.cks)[-k:] if k else [], n)
        self.ops.append(("park", pid, len(self.seq)))
        return pid

    def unpark(self, pid):
        seq, dn_c, cks, _ = self.entries.pop(pid)
        self.seq, self.dn_c, self.cks = list(seq), dn_c, list(cks)
        self.ops.append(("unpark", pid, len(seq)))

    def park_drop(self, pid):
        self.entries.pop(pid)
        self.ops.append(("drop", pid))

    def side_begin(self):
        self.kept = (list(self.seq), self.dn_c, list(self.cks))
        self.seq, self.dn_c, self.cks = [], 0, []
        self.ops.append(("side_begin", len(self.kept[0])))
        return True

    def side_active(self):
        return self.kept is not None

    def side_drop(self):
        self.kept, self.seq, self.dn_c, self.cks = None, [], 0, []
        self.ops.append(("side_drop",))

    def side_end(self):
        (seq, dn_c, cks), self.kept = self.kept, None
        self.ops.append(("side_end",))
        if not self.side_ok:  # (the store ran full: the engine is reset)
            self.seq, self.dn_c, self.cks = [], 0, []
            return False
        self.seq, self.dn_c, self.cks = seq, dn_c, cks
        return True


class HostCacheTests(_HttpCase):
    """side requests (--side-max) and parked conversations (--park-mib) over the store's semantics"""

    LONG = "a long document. " * 40  # (~690 prompt tokens with the test tokenizer: one per character)

    def serve_store(self, eng, side_max=200, park_mib=0, park_min=50, park_cks=2):
        srv, http_ = self.serve(eng)
        srv.args.side_max, srv.args.park_mib, srv.args.park_min, srv.args.park_cks = side_max, park_mib, park_min, park_cks
        with mock.patch.object(sys, "stderr", io.StringIO()):
            srv.setup_store(srv.args)
        srv.lock = self.mk.FifoLock()
        return srv, http_

    def chat(self, http_, msgs, **kw):
        status, data = http_.post({"messages": msgs, **kw})
        self.assertEqual(status, 200, data)
        return json.loads(data)

    def metrics(self, http_):
        c = http.client.HTTPConnection("127.0.0.1", http_.port, timeout=10)
        c.request("GET", "/metrics")
        text = c.getresponse().read().decode()
        c.close()
        return dict(ln.split(" ", 1) for ln in text.splitlines() if ln and not ln.startswith("#"))

    def quiet(self, fn):
        err = io.StringIO()
        with mock.patch.object(sys, "stderr", err), mock.patch.object(self.mk, "CON", self.mk.Console()):
            fn()
        return err.getvalue()

    def test_a_short_first_turn_prompt_runs_as_a_side_request_and_the_long_sequence_stays(self) -> None:
        eng = _StoreEngine(["a", "b"])
        srv, http_ = self.serve_store(eng)
        a1 = [{"role": "user", "content": self.LONG}]
        out = {}

        def run():
            out["r"] = self.chat(http_, a1)
            kept = list(eng.seq)
            self.chat(http_, [{"role": "user", "content": "a title please"}], max_tokens=4)
            self.assertEqual(eng.seq, kept)  # (the long sequence is back)
            a2 = a1 + [{"role": "assistant", "content": out["r"]["choices"][0]["message"]["content"]},
                       {"role": "user", "content": "more"}]
            out["r2"] = self.chat(http_, a2)
        log = self.quiet(run)
        self.assertEqual([o[0] for o in eng.ops], ["side_begin", "side_end"], log)
        self.assertGreaterEqual(out["r2"]["usage"]["prompt_tokens_details"]["cached_tokens"], len(self.LONG), log)
        self.assertIn("side request: the cached", log)
        self.assertIn("side request done: the cached", log)
        m = self.metrics(http_)
        self.assertEqual((m["megakernel:side_requests_total"], m["megakernel:side_requests_dropped_total"]), ("1", "0"))

    def test_no_side_request_when_off_after_an_assistant_reply_or_for_a_short_cached_sequence(self) -> None:
        for side_max, msgs in ((0, [{"role": "user", "content": "hi"}]),
                               (200, [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "x"},
                                      {"role": "user", "content": "more"}])):
            eng = _StoreEngine(["a", "b"])
            srv, http_ = self.serve_store(eng, side_max=side_max)
            self.quiet(lambda: (self.chat(http_, [{"role": "user", "content": self.LONG}]), self.chat(http_, msgs)))
            self.assertEqual(eng.ops, [], side_max)
        eng = _StoreEngine(["a", "b"])
        srv, http_ = self.serve_store(eng)
        self.quiet(lambda: (self.chat(http_, [{"role": "user", "content": "a short one"}]),
                            self.chat(http_, [{"role": "user", "content": "other"}])))
        self.assertEqual(eng.ops, [])  # (what it would discard is less than twice its own length)

    def test_a_side_request_that_fails_or_overflows_the_store_still_ends_it(self) -> None:
        eng = _StoreEngine(["a", "b", "c", "d"], fail_at=1)
        srv, http_ = self.serve_store(eng)
        a1 = [{"role": "user", "content": self.LONG}]

        def run():
            eng.fail_at = None
            self.chat(http_, a1)
            kept = list(eng.seq)
            eng.fail_at, eng.cycles = 1, 0  # (the side request's first decode cycle fails)
            self.assertEqual(http_.post({"messages": [{"role": "user", "content": "x"}]})[0], 500)
            self.assertEqual(eng.seq, kept)  # (the engine's state stayed consistent: the sequence comes back)
            eng.fail_at, eng.side_ok = None, False
            self.chat(http_, [{"role": "user", "content": "y"}])
            self.assertEqual(eng.seq, [])  # (the store ran full during it: dropped, reset)
        log = self.quiet(run)
        self.assertEqual([o[0] for o in eng.ops], ["side_begin", "side_end"] * 2, log)
        self.assertIn("were dropped", log)
        self.assertEqual(self.metrics(http_)["megakernel:side_requests_dropped_total"], "1")

    def test_conversations_are_parked_and_come_back_at_request_boundaries(self) -> None:
        eng = _StoreEngine(["a", "b"])
        srv, http_ = self.serve_store(eng, side_max=0, park_mib=4)
        a1 = [{"role": "user", "content": self.LONG}]
        b1 = [{"role": "user", "content": self.LONG.upper()}]
        out = {}

        def run():
            ra = self.chat(http_, a1)
            kept_a = list(eng.seq)
            self.chat(http_, b1)  # (another conversation: A is parked first)
            kept_b = list(eng.seq)
            a2 = a1 + [{"role": "assistant", "content": ra["choices"][0]["message"]["content"]}, {"role": "user", "content": "more"}]
            out["a2"] = self.chat(http_, a2)  # (B is parked, A comes back and continues)
            out["kept"] = (kept_a, kept_b)
        log = self.quiet(run)
        kept_a, kept_b = out["kept"]
        self.assertEqual(eng.ops, [("park", 1, len(kept_a)), ("park", 2, len(kept_b)), ("unpark", 1, len(kept_a))], log)
        self.assertEqual(out["a2"]["usage"]["prompt_tokens_details"]["cached_tokens"], len(kept_a) - 1, log)
        self.assertIn("parked the cached conversation (a new conversation)", log)
        self.assertIn("restored the parked conversation of", log)
        self.assertRegex(log, r"prompt: [\d,]+ cached \+ \d+ prefilled .* \| restored a parked conversation")
        m = self.metrics(http_)
        self.assertEqual([m[f"megakernel:{k}"] for k in ("parks_total", "park_restores_total", "parked_conversations")],
                         ["2", "1", "1"])
        info = srv.engine_info()["host_cache"]
        self.assertEqual((info["parked"], info["parked_tokens"], info["park_mib"]), (1, [len(kept_b)], 4))

    def test_with_parking_on_a_side_request_has_no_tools_or_a_bounded_reply(self) -> None:
        eng = _StoreEngine(["a", "b"])
        srv, http_ = self.serve_store(eng, park_mib=4)

        def run():
            self.chat(http_, [{"role": "user", "content": self.LONG}])
            self.chat(http_, [{"role": "user", "content": "a title please"}], max_tokens=8)  # (side request)
            self.chat(http_, [{"role": "user", "content": "a title, uncapped"}])  # (no tools: a side request too)
            self.chat(http_, [{"role": "user", "content": "a new task"}], tools=TOOLS)  # (tools: the cached one parks)
        log = self.quiet(run)
        self.assertEqual([o[0] for o in eng.ops], ["side_begin", "side_end"] * 2 + ["park"], log)

    def test_the_conversation_coming_back_is_never_evicted_to_park_the_cached_one(self) -> None:
        eng = _StoreEngine(["a", "b"])
        srv, http_ = self.serve_store(eng, side_max=0, park_mib=1)  # (one ~0.7 MB entry fits)
        a1 = [{"role": "user", "content": "A" + self.LONG}]

        def run():
            ra = self.chat(http_, a1)
            self.chat(http_, [{"role": "user", "content": "B" + self.LONG}])  # (A parked)
            a2 = a1 + [{"role": "assistant", "content": ra["choices"][0]["message"]["content"]}, {"role": "user", "content": "more"}]
            self.chat(http_, a2)  # (B does not fit next to A: B is not parked, A comes back)
        log = self.quiet(run)
        self.assertEqual([o[:2] for o in eng.ops], [("park", 1), ("unpark", 1)], log)
        self.assertIn("not parked", log)

    def test_the_least_recently_parked_conversation_is_evicted_for_room(self) -> None:
        eng = _StoreEngine(["a", "b"])
        srv, http_ = self.serve_store(eng, side_max=0, park_mib=1)  # (one ~0.7 MB entry fits)

        def run():
            for c in ("A", "B", "C"):
                self.chat(http_, [{"role": "user", "content": c + self.LONG}])
        log = self.quiet(run)
        self.assertEqual([o[:2] for o in eng.ops], [("park", 1), ("drop", 1), ("park", 2)], log)
        self.assertIn("evicted the parked conversation of", log)
        self.assertEqual(self.metrics(http_)["megakernel:park_evictions_total"], "1")

    def test_the_environment_can_turn_the_host_cache_off(self) -> None:
        import os
        with mock.patch.dict(os.environ, {"MK_SIDE_MAX": "0", "MK_MAX_BODY_MB": "0"}):
            self.assertEqual(self.mk.env_int("MK_SIDE_MAX", 8192, 0), 0)
            with self.assertRaises(SystemExit):  # (the other settings stay positive)
                self.mk.env_int("MK_MAX_BODY_MB", 128)

    def test_the_engine_lock_serves_waiting_requests_in_arrival_order(self) -> None:
        lock, order, started = self.mk.FifoLock(), [], []
        lock.acquire()

        def worker(i):
            started.append(i)
            with lock:
                order.append(i)
        ths = []
        for i in range(6):
            th = threading.Thread(target=worker, args=(i,))
            th.start()
            ths.append(th)
            while len(started) <= i:  # (each waits before the next arrives)
                time.sleep(0.001)
            time.sleep(0.02)
        self.assertTrue(lock.locked())
        lock.release()
        for th in ths:
            th.join(5)
        self.assertEqual(order, list(range(6)))
        self.assertFalse(lock.locked())


if __name__ == "__main__":
    unittest.main()
