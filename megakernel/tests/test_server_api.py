"""API completeness of server/mk_server.py without a GPU:
tool calls whose values hold tool-call markup, tool-call arguments streamed as they are generated (OpenAI and Anthropic
shapes, the official SDKs' stream accumulators when installed), Anthropic typed tools and unknown blocks, the context
overflow error, reasoning format and template defaults, llama-server's extra request fields, chunked request bodies."""
import base64
import http.client
import io
import json
import socket
import sys
import time
import unittest
from unittest import mock

from test_mk_server import (CALL, ROOT, TEMPLATE, TOOLS, _FakeVisionLib, _HttpCase, _ScriptEngine, _load_mk_server, _request,
                            _sse, _sse_events)

WRITE = [{"type": "function", "function": {"name": "write", "parameters": {"type": "object", "properties": {
    "path": {"type": "string"}, "content": {"type": "string"}, "mode": {"type": "integer"}, "opts": {"type": "object"}}}}}]


def _call(name, **params):
    """Qwen's markup of a call (values as given: strings raw, others as their JSON text)"""
    body = "".join(f"<parameter={k}>\n{v if isinstance(v, str) else json.dumps(v)}\n</parameter>\n" for k, v in params.items())
    return f"<tool_call>\n<function={name}>\n{body}</function>\n</tool_call>"


def _pieces(text, n=3):
    return [text[i:i + n] for i in range(0, len(text), n)]


def _accumulate(deltas):
    """an OpenAI stream's tool_calls deltas by index, as the SDKs accumulate them: (calls, the deltas of each index)"""
    calls, per = {}, {}
    for tc in deltas:
        per.setdefault(tc["index"], []).append(tc)
        c = calls.setdefault(tc["index"], {"id": tc.get("id"), "type": tc.get("type"), "function": {"name": "", "arguments": ""}})
        for k in ("name", "arguments"):
            c["function"][k] += tc["function"].get(k) or ""
    return [calls[i] for i in sorted(calls)], per


class ToolCallParserTests(unittest.TestCase):
    """a value runs to its </parameter>; the incremental parser's decisions do not depend on the pieces"""

    @classmethod
    def setUpClass(cls) -> None:
        cls.mk = _load_mk_server()

    def test_a_value_may_hold_tool_call_markup(self) -> None:
        mk = self.mk
        # (a value that holds a <tool_call> tag: the call is parsed, nothing is left as text)
        probe = "<tool_call>\n<function=f>\n<parameter=x>\n<tool_call>\n</parameter>\n</function>\n</tool_call>"
        calls = mk.parse_tool_calls(probe, [])
        self.assertEqual([(c["function"]["name"], json.loads(c["function"]["arguments"])) for c in calls], [("f", {"x": "<tool_call>"})])
        self.assertEqual(mk.unparsed_tool_call(probe), "")
        doc = "Calls look like <tool_call>\n<function=read>\n<parameter=path>\n/x\n</parameter>\n</function>\n</tool_call>, see."
        text = _call("write", path="notes.md", content=doc) + "\n" + _call("write", path="b", content="2")
        calls = mk.parse_tool_calls(text, WRITE)
        self.assertEqual([json.loads(c["function"]["arguments"])["content"] for c in calls], [doc.split("\n</parameter>")[0], "2"])
        # an example call inside a value that never closes (cut by max_tokens) is not run
        cut = "<tool_call>\n<function=write>\n<parameter=content>\nlike <tool_call>\n<function=rm>\n</function>\n</tool_call>\n"
        self.assertEqual((mk.parse_tool_calls(cut, WRITE), mk.unparsed_tool_call(cut)), ([], cut))
        # a new <tool_call> between the parameters (not in a value): the first was left open, the second is a call
        text = "<tool_call>\n<function=a>\n<parameter=x>\n1\n</parameter>\n" + _call("b", y="2")
        self.assertEqual([c["function"]["name"] for c in mk.parse_tool_calls(text, [])], ["b"])

    def test_values_and_types(self) -> None:
        args = lambda text, tools=WRITE: [json.loads(c["function"]["arguments"]) for c in self.mk.parse_tool_calls(text, tools)]
        self.assertEqual(args(_call("write", path="123", mode=7, opts={"a": [1]})), [{"path": "123", "mode": 7, "opts": {"a": [1]}}])
        self.assertEqual(args(_call("write", mode="seven")), [{"mode": "seven"}])  # (not JSON: the text)
        self.assertEqual(args(_call("other", n="12", s="text", b="true", e="")), [{"n": 12, "s": "text", "b": True, "e": ""}])
        self.assertEqual(args("<tool_call>\n<function=write>\n<parameter=content>\n\na\n\n</parameter>\n</function>\n</tool_call>"),
                         [{"content": "\na\n"}])  # (one newline at each end belongs to the markup)
        self.assertEqual(args("<tool_call><function=write><parameter=path>p</parameter></function></tool_call>"), [{"path": "p"}])
        self.assertEqual(args("<tool_call>\n<function=free>\n</function>\n</tool_call>", []), [{}])

    def _events(self, text, tools, n):
        p = self.mk.ToolCallParser(tools, events=True)
        evs = []
        for piece in _pieces(text, n):
            evs += p.feed(piece)
        evs += p.feed("", final=True)
        return p, evs

    def test_pieces_do_not_matter_and_fragments_add_up(self) -> None:
        texts = [_call("write", path="a\\b \"q\" é", content="x\ny\n" * 5, mode=3) + "\n\n" + _call("free"),
                 "<tool_call>\n<function=write>\n<parameter=content>\n<tool_call></parameter>\n</function>\n</tool_call>",
                 _call("write", path="p") + "\n<tool_call>\n<function=write>\n<parameter=path>\ncut",
                 "<tool_call>\n<function=write>\n<parameter=opts>\n{\"k\": 1}\n</parameter>\n</function>\nx</function>\n</tool_call>"]
        for text in texts:
            whole = self.mk.ToolCallParser(WRITE)
            whole.feed(text, final=True)
            for n in range(1, 9):
                p, evs = self._events(text, WRITE, n)
                self.assertEqual([c["function"] for c in p.calls], [c["function"] for c in whole.calls], (text, n))
                self.assertEqual(p.unparsed(), whole.unparsed(), (text, n))
                frags = {}
                for ev in evs:
                    frags[ev[1]] = frags.get(ev[1], "") + (ev[2] if ev[0] == "args" else "")
                ids = [ev[2] for ev in evs if ev[0] == "start"]
                for i, cid in enumerate(ids):
                    done = [c for c in p.calls if c["id"] == cid]
                    if done:
                        self.assertEqual(frags[i], done[0]["function"]["arguments"], (text, n))
                    else:
                        with self.assertRaises(ValueError):  # (a call left open never has valid JSON)
                            json.loads(frags[i])

    def test_strings_stream_typed_values_come_whole(self) -> None:
        text = _call("write", content="z" * 60, mode=12345)
        _, evs = self._events(text, WRITE, 2)
        frags = [ev[2] for ev in evs if ev[0] == "args"]
        self.assertGreater(sum("z" in f for f in frags), 20)  # (the string as it is generated)
        self.assertIn(', "mode": 12345', frags)  # (the integer whole, at its end)
        self.assertEqual(frags[-1], "}")


class ToolCallStreamTests(_HttpCase):
    """the stream's tool calls as they are generated, == the non-streamed reply's"""

    def _openai(self, pieces, body):
        _, http_ = self.serve(_ScriptEngine(pieces))
        base = {"messages": [{"role": "user", "content": "go"}], "tools": WRITE, "max_tokens": 500, **body}
        status, data = http_.post({**base, "stream": False})
        self.assertEqual(status, 200, data)
        whole = json.loads(data)["choices"][0]
        _, events = _sse(http_.raw(_request({**base, "stream": True}), until=b"0\r\n\r\n"))
        self.assertEqual(events[-1], b"[DONE]")
        deltas = [c for e in events[:-1] for c in json.loads(e)["choices"]]
        tcs = [tc for d in deltas for tc in d["delta"].get("tool_calls") or []]
        content = "".join(d["delta"].get("content") or "" for d in deltas)
        return whole, deltas, tcs, content, [d["finish_reason"] for d in deltas if d["finish_reason"]][-1]

    def test_openai_stream_shape_and_identity(self) -> None:
        text = "Writing.\n\n" + _call("write", path="/tmp/a.txt", content="line \"1\"\nline 2\né中", mode=420) + "\n" + \
            _call("write", path="/tmp/b", opts={"x": [1, 2]})
        whole, deltas, tcs, content, finish = self._openai(_pieces(text), {})
        calls, per = _accumulate(tcs)
        self.assertEqual((finish, whole["finish_reason"], content, whole["message"]["content"]),
                         ("tool_calls", "tool_calls", "Writing.", "Writing."))
        self.assertEqual(len(calls), 2)
        for i, (c, w) in enumerate(zip(calls, whole["message"]["tool_calls"])):
            first = per[i][0]
            self.assertEqual(set(first), {"index", "id", "type", "function"})
            self.assertEqual((first["type"], first["function"]), ("function", {"name": "write", "arguments": ""}))
            self.assertTrue(all(set(d) == {"index", "function"} and set(d["function"]) == {"arguments"} for d in per[i][1:]))
            self.assertGreater(len(per[i]), 4)  # (pieces, not one delta at the end)
            self.assertEqual(c["function"], w["function"])  # (byte for byte, the ids aside)
            self.assertTrue(c["id"].startswith("call_"))

    def test_parallel_tool_calls_false_and_stop_strings(self) -> None:
        text = _call("write", path="a") + "\n" + _call("write", path="b")
        whole, _, tcs, _, finish = self._openai(_pieces(text), {"parallel_tool_calls": False})
        calls, _ = _accumulate(tcs)
        self.assertEqual(([c["function"] for c in calls], finish), ([whole["message"]["tool_calls"][0]["function"]], "tool_calls"))
        self.assertEqual(json.loads(calls[0]["function"]["arguments"]), {"path": "a"})
        # a stop string inside a call: the call is left open (finish stop; streamed: arguments never valid JSON)
        whole, _, tcs, content, finish = self._openai(_pieces(_call("write", path="abcSTOPdef")), {"stop": ["STOP"]})
        self.assertEqual((whole["finish_reason"], whole["message"].get("tool_calls")), ("stop", None))
        self.assertTrue(whole["message"]["content"].startswith("<tool_call>"))
        calls, _ = _accumulate(tcs)
        self.assertEqual((finish, content, len(calls)), ("stop", "", 1))
        self.assertEqual(calls[0]["function"]["arguments"], '{"path": "abc')

    def test_markup_inside_a_value_streams_too(self) -> None:
        doc = "a <tool_call> b </function> c"
        whole, _, tcs, content, finish = self._openai(_pieces(_call("write", content=doc)), {})
        calls, _ = _accumulate(tcs)
        self.assertEqual((finish, content), ("tool_calls", ""))
        self.assertEqual(json.loads(calls[0]["function"]["arguments"]), {"content": doc})
        self.assertEqual(calls[0]["function"], whole["message"]["tool_calls"][0]["function"])

    def test_text_after_a_call_is_dropped_and_logged(self) -> None:
        # (Qwen's template tells the model not to write after a call; when it does, the reply carries the calls only)
        text = ("Before.\n\n" + _call("write", path="a") + "\nbetween\n" + _call("write", path="b") + "\n\nAfter the call.\n")
        for n in (1, 3, 8):
            p = self.mk.OutputParser(False, WRITE)
            for piece in _pieces(text, n):
                p.feed(piece)
            p.feed("", final=True)
            self.assertEqual((len(p.calls), p.stray(), p.unparsed()), (2, ["between", "After the call."], ""), n)
        cut = self.mk.OutputParser(False, WRITE)  # (markup after the calls is the reply's text, not dropped)
        cut.feed(_call("write", path="a") + "\n<tool_call>\n<function=write>\n<parameter=path>\nx", final=True)
        self.assertEqual((cut.stray(), cut.unparsed()[:11]), ([], "<tool_call>"))
        self.assertEqual(self.mk.OutputParser(False, []).stray(), [])
        err = io.StringIO()
        with mock.patch.object(sys, "stderr", err), mock.patch.object(self.mk, "CON", self.mk.Console()):
            whole, _, tcs, content, finish = self._openai(_pieces(text), {})
        self.assertEqual((whole["message"]["content"], content, finish, len(tcs) > 2), ("Before.", "Before.", "tool_calls", True))
        self.assertEqual(err.getvalue().count("dropped 22 chars of text the model wrote between or after its tool calls"), 2)

    def test_anthropic_stream(self) -> None:
        text = "Let me write.\n\n" + _call("write", path="/x", content="q" * 40, mode=1) + "\n" + _call("write", path="/y")
        tools = [{"name": "write", "input_schema": WRITE[0]["function"]["parameters"]}]
        body = {"model": "m", "max_tokens": 500, "messages": [{"role": "user", "content": "go"}], "tools": tools}
        _, http_ = self.serve(_ScriptEngine(_pieces(text)))
        c = http.client.HTTPConnection("127.0.0.1", http_.port, timeout=20)
        c.request("POST", "/v1/messages", json.dumps(body))
        whole = json.loads(c.getresponse().read())
        c.close()
        raw = http_.raw(_request({**body, "stream": True}).replace(b"/v1/chat/completions", b"/v1/messages"), until=b"0\r\n\r\n")
        _, ev = _sse_events(raw)
        starts = [(d["index"], d["content_block"]) for n, d in ev if n == "content_block_start"]
        self.assertEqual([b["type"] for _, b in starts], ["text", "tool_use", "tool_use"])
        inputs = {}
        for n, d in ev:
            if n == "content_block_delta" and d["delta"]["type"] == "input_json_delta":
                inputs.setdefault(d["index"], []).append(d["delta"]["partial_json"])
        self.assertGreater(len(inputs[1]), 4)
        uses = [b for b in whole["content"] if b["type"] == "tool_use"]
        for (i, b), u in zip(starts[1:], uses):
            self.assertEqual((b["name"], b["input"], b["id"][:6]), ("write", {}, "toolu_"))
            self.assertEqual(json.loads("".join(inputs[i])), u["input"])
        self.assertEqual(ev[-2][1]["delta"]["stop_reason"], "tool_use")
        self.assertEqual([n for n, _ in ev].count("content_block_stop"), 3)


class AnthropicBlockTests(_HttpCase):
    """Anthropic typed tools become functions, unknown content blocks are skipped (as llama-server), documents with
    a text source become text; each kind is a one-time console warning"""

    def test_typed_tools(self) -> None:
        notes = []
        tools = [{"name": "read", "description": "reads", "input_schema": {"type": "object", "properties": {}}},
                 {"type": "custom", "name": "c", "input_schema": {"type": "object"}},
                 {"type": "bash_20250124", "name": "bash"}, {"type": "text_editor_20250728", "name": "str_replace_based_edit_tool"},
                 {"type": "web_search_20250305", "name": "web_search", "max_uses": 5}]
        out = self.mk.anthropic_to_chat({"messages": [], "tools": tools}, notes)
        self.assertEqual([t["function"] for t in out["tools"]], [
            {"name": "read", "description": "reads", "parameters": {"type": "object", "properties": {}}},
            {"name": "c", "description": "", "parameters": {"type": "object"}},
            {"name": "bash", "description": "", "parameters": {}},
            {"name": "str_replace_based_edit_tool", "description": "", "parameters": {}},
            {"name": "web_search", "description": "", "parameters": {}}])
        self.assertEqual([k for k, _ in notes], ["anthropic-tool-bash_20250124", "anthropic-tool-text_editor_20250728",
                                                 "anthropic-tool-web_search_20250305"])
        for bad in ([{"type": "bash_20250124"}], ["bash"], [{"name": ""}]):
            with self.assertRaises(self.mk.BadRequest):
                self.mk.anthropic_to_chat({"messages": [], "tools": bad})

    def test_blocks(self) -> None:
        notes = []
        content = [{"type": "text", "text": "Summarize."},
                   {"type": "document", "title": "notes.txt", "source": {"type": "text", "media_type": "text/plain", "data": "A B"}},
                   {"type": "document", "source": {"type": "content", "content": [{"type": "text", "text": "one"},
                                                                              {"type": "text", "text": "two"}]},
                    "context": "ctx"},
                   {"type": "document", "source": {"type": "base64", "media_type": "application/pdf", "data": "JVBERi0="}},
                   {"type": "search_result", "source": "https://x", "title": "t", "content": [{"type": "text", "text": "s"}]},
                   {"type": "container_upload", "file_id": "f"}]
        msgs = [{"role": "user", "content": content},
                {"role": "assistant", "content": [
                    {"type": "server_tool_use", "id": "srvtoolu_1", "name": "web_search", "input": {"query": "q"}},
                    {"type": "web_search_tool_result", "tool_use_id": "srvtoolu_1", "content": []},
                    {"type": "text", "text": "Found it."},
                    {"type": "tool_use", "id": "toolu_9", "name": "read", "input": {"p": 1}}]},
                {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_9", "content": [
                    {"type": "text", "text": "r1"}, {"type": "document", "source": {"type": "text", "data": "d1"}},
                    {"type": "search_result", "source": "u", "title": "t", "content": []}]}]}]
        m = self.mk.anthropic_to_chat({"messages": msgs}, notes)["messages"]
        self.assertEqual(m[0], {"role": "user", "content": [{"type": "text", "text": "Summarize."},
                                                             {"type": "text", "text": "notes.txt\n\nA B"},
                                                             {"type": "text", "text": "ctx\n\none\ntwo"}]})
        self.assertEqual((m[1]["content"], [c["function"]["name"] for c in m[1]["tool_calls"]]), ("Found it.", ["read"]))
        self.assertEqual(m[2], {"role": "tool", "tool_call_id": "toolu_9", "content": "r1\nd1"})
        self.assertEqual(sorted({k for k, _ in notes}), [
            "anthropic-block-container_upload", "anthropic-block-search_result", "anthropic-block-server_tool_use",
            "anthropic-block-web_search_tool_result", "anthropic-doc-base64"])

    def test_a_prefill_and_fields_the_server_ignores_are_warnings(self) -> None:
        # (accepted, as before: Claude Code sends some of these fields; never silent)
        body = {"max_tokens": 5, "messages": [{"role": "user", "content": "Q"}, {"role": "assistant", "content": "{"}],
                "output_format": {"type": "json_schema", "schema": {"type": "object"}},
                "output_config": {"effort": "high", "format": {"type": "json_schema"}},
                "context_management": {"edits": [{"type": "clear_thinking_20251015"}]}, "container": "c1",
                "metadata": {"user_id": "u"}, "service_tier": "auto"}
        want = {"anthropic-prefill", "anthropic-field-output_format", "anthropic-field-output_config",
                "anthropic-field-context_management", "anthropic-field-container"}
        notes = []
        self.mk.anthropic_to_chat(body, notes)
        self.assertEqual({k for k, _ in notes}, want)
        notes = []
        self.mk.anthropic_to_chat({"messages": body["messages"][:1], "metadata": {"user_id": "u"}}, notes)
        self.assertEqual(notes, [])
        srv, http_ = self.serve(_ScriptEngine(["ok"]))
        for path in ("/v1/messages", "/v1/messages/count_tokens"):
            status, data = _post(http_.port, path, body)
            self.assertEqual(status, 200, data)
        self.assertTrue(want <= srv.warned)

    def test_output_config_effort_is_the_templates_reasoning_effort(self) -> None:
        # (Claude Code's /effort; it sends high by default: the template's default xhigh, so the same prompt as without it)
        base = {"max_tokens": 5, "messages": [{"role": "user", "content": "Q"}], "thinking": {"type": "adaptive"}}
        for eff, want in (("low", "low"), ("medium", "medium"), ("high", "xhigh"), ("max", "xhigh")):
            notes = []
            self.assertEqual(self.mk.anthropic_to_chat({**base, "output_config": {"effort": eff}}, notes)["reasoning_effort"],
                             want)
            self.assertEqual(notes, [])
        off = {**base, "output_config": {"effort": "low"}, "thinking": {"type": "disabled"}}
        self.assertNotIn("reasoning_effort", self.mk.anthropic_to_chat(off, []))
        notes = []
        self.assertNotIn("reasoning_effort", self.mk.anthropic_to_chat({**base, "output_config": {"effort": "x"}}, notes))
        self.assertEqual([k for k, _ in notes], ["anthropic-field-output_config"])

    def test_a_history_tool_call_name_that_is_not_a_string_is_a_400(self) -> None:
        srv, http_ = self.serve(_ScriptEngine(["ok"]))
        for name in (5, ["x"], True):
            call = {"id": "c1", "type": "function", "function": {"name": name, "arguments": "{}"}}
            body = {"max_tokens": 2, "messages": [{"role": "user", "content": "hi"},
                                                  {"role": "assistant", "content": "", "tool_calls": [call]},
                                                  {"role": "tool", "tool_call_id": "c1", "content": "r"}]}
            status, data = _post(http_.port, "/v1/chat/completions", body)
            self.assertEqual(status, 400, data)
            self.assertIn("function name must be a string", json.loads(data)["error"]["message"])
        self.assertEqual(srv.counters()["requests_failed"], 0)

    def test_system_messages_inside_the_conversation(self) -> None:
        # (Claude Code 2.1 sends its reminders as a system message after the first user message)
        body = {"system": [{"type": "text", "text": "S"}], "messages": [
            {"role": "user", "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]},
            {"role": "system", "content": [{"type": "text", "text": "reminder", "cache_control": {"type": "ephemeral"}}]},
            {"role": "system", "content": "plain"}]}
        m = self.mk.anthropic_to_chat(body)["messages"]
        self.assertEqual([x["role"] for x in m], ["system", "user", "system", "system"])
        self.assertEqual((m[2]["content"], m[3]["content"]), ("reminder", "plain"))
        notes = []
        m = self.mk.anthropic_to_chat(body, notes, mid_system=False)["messages"]  # (Qwen's templates)
        self.assertEqual([x["role"] for x in m], ["system", "user", "user", "user"])
        self.assertEqual({k for k, _ in notes}, {"anthropic-mid-system"})
        # the server asks its template once
        strict = ("{% for m in messages %}{% if m.role == 'system' and not loop.first %}"
                  "{{ raise_exception('System message must be at the beginning.') }}{% endif %}"
                  "<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{% endfor %}"
                  "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}")
        self.assertFalse(self.mk.takes_mid_system(self.mk.make_template(strict)))
        self.assertTrue(self.mk.takes_mid_system(self.mk.make_template(TEMPLATE)))
        srv, http_ = self.serve(_ScriptEngine(["ok"]))
        srv.template = self.mk.make_template(strict)
        c = http.client.HTTPConnection("127.0.0.1", http_.port, timeout=10)
        c.request("POST", "/v1/messages", json.dumps({**body, "max_tokens": 5}))
        r = c.getresponse()
        self.assertEqual((r.status, json.loads(r.read())["stop_reason"]), (200, "end_turn"))
        c.close()
        self.assertEqual((srv.mid_system, "anthropic-mid-system" in srv.warned), (False, True))

    def test_http(self) -> None:
        srv, http_ = self.serve(_ScriptEngine(["Hello", "."]))
        body = {"model": "m", "max_tokens": 20, "tools": [{"type": "web_fetch_20250910", "name": "web_fetch"}],
                "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"},
                                                          {"type": "redacted_thinking", "data": "x"},
                                                          {"type": "mcp_tool_use", "id": "m", "name": "n", "input": {}}]}]}
        for path in ("/v1/messages", "/v1/messages", "/v1/messages/count_tokens"):
            c = http.client.HTTPConnection("127.0.0.1", http_.port, timeout=10)
            c.request("POST", path, json.dumps(body))
            r = c.getresponse()
            o = json.loads(r.read())
            c.close()
            self.assertEqual(r.status, 200, o)
        self.assertEqual(o, {"input_tokens": len(srv.render(self.mk.anthropic_to_chat(body))[0])})
        self.assertTrue({"anthropic-tool-web_fetch_20250910", "anthropic-block-mcp_tool_use"} <= srv.warned)


def _post(port, path, body, headers=None):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=20)
    c.request("POST", path, json.dumps(body) if not isinstance(body, bytes) else body, headers or {})
    r = c.getresponse()
    data = r.read()
    c.close()
    return r.status, data


class TextPartTests(_HttpCase):
    """text-only part lists (Claude Code's text blocks, Codex's input_text parts, AI-SDK clients' text parts) are joined
    by newlines, as with images and as llama-server joins them: never run together; the count routes count that prompt"""

    PARTS = [{"type": "text", "text": "<system-reminder>r</system-reminder>"}, {"type": "text", "text": "do it"}]
    JOINED = "<|im_start|>user\n<system-reminder>r</system-reminder>\ndo it<|im_end|>"

    def fed(self, srv):
        """the prompt the engine got (the test tokenizer: one token per character)"""
        n = next(c[2] for c in srv.eng.calls if c[0] == "prefill")
        return "".join(map(chr, srv.eng.seq[:n]))

    def check(self, path, count_path, body):
        srv, http_ = self.serve(_ScriptEngine(["ok"]))
        status, data = _post(http_.port, path, body)
        self.assertEqual(status, 200, data)
        prompt = self.fed(srv)
        self.assertIn(self.JOINED, prompt)
        status, data = _post(http_.port, count_path, body)
        self.assertEqual((status, json.loads(data)["input_tokens"]), (200, len(prompt)))
        return srv, http_, prompt

    def test_openai_text_parts(self) -> None:
        body = {"messages": [{"role": "user", "content": self.PARTS}], "max_tokens": 2}
        srv, http_, prompt = self.check("/v1/chat/completions", "/v1/chat/completions/input_tokens", body)
        self.assertEqual(json.loads(_post(http_.port, "/apply-template", body)[1])["prompt"], prompt)
        srv.template = self.mk.make_template((ROOT / "tests" / "templates" / "qwen38_unsloth.jinja").read_text(encoding="utf-8"))
        self.assertIn(self.JOINED, srv.render(body)[0])  # (Qwen3.8's template too: it would run a list's parts together)

    def test_anthropic_text_blocks(self) -> None:
        self.check("/v1/messages", "/v1/messages/count_tokens",
                   {"model": "m", "max_tokens": 2, "messages": [{"role": "user", "content": self.PARTS}]})

    def test_responses_input_text_parts(self) -> None:
        parts = [{"type": "input_text", "text": p["text"]} for p in self.PARTS]
        self.check("/v1/responses", "/v1/responses/input_tokens",
                   {"max_output_tokens": 2, "input": [{"type": "message", "role": "user", "content": parts}]})


class CutCharacterTests(_HttpCase):
    """a reply cut by max_tokens inside a character whose bytes span tokens (byte-fallback pieces) ends before that
    character: no U+FFFD for bytes the model never finished"""

    BODY = {"messages": [{"role": "user", "content": "hi"}], "chat_template_kwargs": {"enable_thinking": False}}

    def reply(self, raw, **kw):
        eng = _ScriptEngine([])
        eng.pieces = list(raw)
        srv, http_ = self.serve(eng)
        srv.template = self.mk.make_template((ROOT / "tests" / "templates" / "qwen38_unsloth.jinja").read_text(encoding="utf-8"))
        status, data = _post(http_.port, "/v1/chat/completions", {**self.BODY, **kw})
        self.assertEqual(status, 200, data)
        if not kw.get("stream"):
            return json.loads(data)["choices"][0]["message"]["content"]
        events = [json.loads(ln[6:]) for ln in data.decode().split("\n") if ln.startswith("data: {")]
        return "".join(e["choices"][0]["delta"].get("content") or "" for e in events if e.get("choices"))

    def test_the_cut_character_is_dropped(self) -> None:
        for raw, whole in (([b"ok \xf0\x9f", b"\x98\x80 done"], "ok \U0001F600 done"),
                           ([b"ok \xe2", b"\x82\xac done"], "ok \u20ac done")):
            self.assertEqual(self.reply(raw, max_tokens=8), whole)
            for stream in (False, True):
                self.assertEqual(self.reply(raw, max_tokens=1, stream=stream).rstrip(), "ok", (raw, stream))


class ContextOverflowTests(_HttpCase):
    """a prompt longer than the context: llama-server's exceed_context_size_error with n_prompt_tokens / n_ctx, and
    Anthropic's 'prompt is too long' on /v1/messages; streams too (the error comes before the stream starts)"""

    def test_openai_and_anthropic_shapes(self) -> None:
        srv, http_ = self.serve(_ScriptEngine(["ok"]), ctx=100)
        msgs = [{"role": "user", "content": "x" * 100}]
        n = len(srv.render({"messages": msgs})[0])  # (a character tokenizer)
        for stream in (False, True):
            status, data = _post(http_.port, "/v1/chat/completions", {"messages": msgs, "stream": stream})
            self.assertEqual((status, json.loads(data)), (400, {"error": {
                "code": 400, "type": "exceed_context_size_error", "n_prompt_tokens": n, "n_ctx": 100,
                "message": f"request ({n} tokens) exceeds the available context size (100 tokens), try increasing it"}}))
            status, data = _post(http_.port, "/v1/messages", {"max_tokens": 5, "messages": msgs, "stream": stream})
            n_a = len(srv.render(self.mk.anthropic_to_chat({"messages": msgs}))[0])
            self.assertEqual((status, json.loads(data)), (400, {"type": "error", "error": {
                "type": "invalid_request_error", "message": f"prompt is too long: {n_a} tokens > 83 maximum",
                "n_prompt_tokens": n_a, "n_ctx": 100}}))
        # (the largest prompt that runs: 83 tokens, 16 left for the reply)
        over = len(srv.render({"messages": [{"role": "user", "content": ""}]})[0])
        fit = [{"role": "user", "content": "x" * (83 - over)}]
        status, data = _post(http_.port, "/v1/chat/completions", {"messages": fit})
        self.assertEqual(status, 200, data)
        status, data = _post(http_.port, "/v1/chat/completions", {"messages": [{"role": "user", "content": fit[0]["content"] + "x"}]})
        self.assertEqual(json.loads(data)["error"]["n_prompt_tokens"], 84)
        # other 400s keep their type, with llama-server's code field
        status, data = _post(http_.port, "/v1/chat/completions", {"messages": msgs, "n": 2})
        self.assertEqual((status, json.loads(data)["error"]["code"], json.loads(data)["error"]["type"]), (400, 400, "invalid_request_error"))

    def test_images_stop_at_the_context(self) -> None:
        srv, http_ = self.serve(_ScriptEngine(["ok"]), ctx=48)
        lib = _FakeVisionLib()  # (each image: 2 x 2 = 4 tokens + start / end)
        v = object.__new__(self.mk.Vision)
        v.L, v.ctx, v.cache, v.cache_max = lib, 1, {}, 8
        srv.vision, srv.vis_start, srv.vis_end, srv.img_pad = v, 1001, 1002, 1003
        parts = [{"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(b"i%d" % i).decode()}}
                 for i in range(8)]
        status, data = _post(http_.port, "/v1/chat/completions", {"messages": [{"role": "user", "content": parts}]})
        err = json.loads(data)["error"]
        self.assertEqual((status, err["type"], err["n_ctx"]), (400, "exceed_context_size_error", 48))
        self.assertTrue(err["message"].startswith(f"request (at least {err['n_prompt_tokens']} tokens)"), err)
        self.assertLess(len(lib.freed) + len(lib.live), 8)  # (stopped decoding images)


THINK = TEMPLATE.replace("assistant\n{% endif %}", "assistant\n{% if enable_thinking %}<think>\n{% endif %}{% endif %}")


class ReasoningFormatTests(_HttpCase):
    """--reasoning-format / reasoning_format (auto = deepseek, the default), --chat-template-kwargs
    defaults under the request's"""

    PIECES = ["Hmm", ".</think>", "\n\nFour"]

    def _chat(self, http_, body, stream):
        body = {"messages": [{"role": "user", "content": "2+2?"}], "chat_template_kwargs": {"enable_thinking": True},
                "max_tokens": 50, **body}
        if not stream:
            status, data = http_.post(body)
            self.assertEqual(status, 200, data)
            m = json.loads(data)["choices"][0]["message"]
            return m.get("content"), m.get("reasoning_content")
        _, events = _sse(http_.raw(_request({**body, "stream": True}), until=b"0\r\n\r\n"))
        deltas = [c["delta"] for e in events[:-1] for c in json.loads(e)["choices"]]
        return ("".join(d.get("content") or "" for d in deltas),
                "".join(d.get("reasoning_content") or "" for d in deltas) or None)

    def test_formats(self) -> None:
        srv, http_ = self.serve(_ScriptEngine(self.PIECES))
        srv.template = self.mk.make_template(THINK)
        inline = "<think>\nHmm.</think>\n\nFour"
        want = {None: {False: ("Four", "Hmm."), True: ("Four", "Hmm.")},
                "auto": {False: ("Four", "Hmm."), True: ("Four", "Hmm.")},
                "deepseek": {False: ("Four", "Hmm."), True: ("Four", "Hmm.")},
                "none": {False: (inline, None), True: (inline, None)},
                "deepseek-legacy": {False: (inline, "Hmm."), True: (inline, None)}}
        for fmt, by_stream in want.items():
            for stream, w in by_stream.items():
                body = {} if fmt is None else {"reasoning_format": fmt}
                self.assertEqual(self._chat(http_, body, stream), w, (fmt, stream))
        # the server's default; a request's own wins; thinking off: nothing to keep inline
        srv.args.reasoning_format = "none"
        self.assertEqual(self._chat(http_, {}, False), (inline, None))
        self.assertEqual(self._chat(http_, {"reasoning_format": "auto"}, True), ("Four", "Hmm."))
        self.assertEqual(self._chat(http_, {"chat_template_kwargs": {"enable_thinking": False}}, False)[1], None)
        # Anthropic replies keep their thinking blocks
        c = http.client.HTTPConnection("127.0.0.1", http_.port, timeout=10)
        c.request("POST", "/v1/messages", json.dumps({"max_tokens": 50, "thinking": {"type": "enabled", "budget_tokens": 1024},
                                                     "messages": [{"role": "user", "content": "2+2?"}]}))
        o = json.loads(c.getresponse().read())
        c.close()
        self.assertEqual([b["type"] for b in o["content"]], ["thinking", "text"])
        status, data = http_.post({"messages": [{"role": "user", "content": "x"}], "reasoning_format": "raw"})
        self.assertEqual((status, json.loads(data)["error"]["message"].split(":")[0]), (400, "reasoning_format"))

    def test_template_kwargs_defaults(self) -> None:
        srv, http_ = self.serve(_ScriptEngine(["ok"]))
        srv.template = self.mk.make_template(THINK.replace("{% endfor %}", "{% endfor %}[{{ reasoning_effort }}]"))
        prompt = lambda body: json.loads(_post(http_.port, "/apply-template", {"messages": [{"role": "user", "content": "q"}],
                                                                               **body})[1])["prompt"]
        self.assertTrue(prompt({}).endswith("[]<|im_start|>assistant\n"))
        srv.args.chat_template_kwargs = {"enable_thinking": True, "reasoning_effort": "low"}
        self.assertTrue(prompt({}).endswith("[low]<|im_start|>assistant\n<think>\n"))
        self.assertTrue(prompt({"chat_template_kwargs": {"enable_thinking": False}}).endswith("[low]<|im_start|>assistant\n"))
        self.assertTrue(prompt({"reasoning_effort": "high"}).endswith("[high]<|im_start|>assistant\n<think>\n"))
        self.assertTrue(prompt({"reasoning_effort": "none"}).endswith("[low]<|im_start|>assistant\n"))
        status, data = _post(http_.port, "/apply-template", {"messages": [], "chat_template_kwargs": "x"})
        self.assertEqual(status, 400, data)

    def test_options(self) -> None:
        import contextlib
        import io
        from unittest import mock
        for argv in (["--reasoning-format", "raw"], ["--chat-template-kwargs", "{"], ["--chat-template-kwargs", "[1]"],
                     ["--chat-template-kwargs", '{"add_vision_id": true}']):
            err = io.StringIO()
            with mock.patch.object(self.mk.sys, "argv", ["mk_server.py", "-m", "x.gguf", *argv]), \
                    contextlib.redirect_stderr(err), self.assertRaises(SystemExit) as cm:
                self.mk.main()
            self.assertEqual(cm.exception.code, 2, argv)
            self.assertIn(argv[0], err.getvalue())


class _ProgressEngine(_ScriptEngine):
    """a prefill of n_chunks chunks of chunk_s seconds that reports its progress as mk_prefill_progress does"""

    def __init__(self, pieces, n_chunks=6, chunk_s=0.25, **kw):
        super().__init__(pieces, **kw)
        self.n_chunks, self.chunk_s, self.prog = n_chunks, chunk_s, (0, 0)

    def prefill_progress(self):
        return self.prog

    def prefill(self, toks, ck_at=-1, cks=(), images=()):
        for c in range(self.n_chunks):
            time.sleep(self.chunk_s)
            self.prog = (len(toks) * (c + 1) // self.n_chunks, len(toks))
        return super().prefill(toks, ck_at, cks, images)


class RequestFieldTests(_HttpCase):
    """llama-server's return_progress, timings_per_token and t_max_predict_ms; its other fields accepted when
    harmless, a 400 when they would change the output"""

    def _stream(self, http_, body):
        _, events = _sse(http_.raw(_request({"messages": [{"role": "user", "content": "hello"}], "max_tokens": 20,
                                             "stream": True, **body}), until=b"0\r\n\r\n", timeout=10))
        self.assertEqual(events[-1], b"[DONE]")
        return [json.loads(e) for e in events[:-1]]

    def test_return_progress(self) -> None:
        srv, http_ = self.serve(_ProgressEngine(["a", "b", "c"]))
        evs = self._stream(http_, {"return_progress": True})
        prog = [e["prompt_progress"] for e in evs if "prompt_progress" in e]
        self.assertGreaterEqual(len(prog), 3)  # (start, at least one while it runs, end)
        total = prog[0]["total"]
        self.assertEqual((prog[0]["cache"], prog[0]["processed"], prog[-1]["processed"]), (0, 0, total))
        self.assertEqual([p["processed"] for p in prog], sorted(p["processed"] for p in prog))
        self.assertTrue(all(set(p) == {"total", "cache", "processed", "time_ms"} for p in prog))
        self.assertTrue(all(e["choices"][0]["delta"] == {"role": "assistant", "content": None} for e in evs if "prompt_progress" in e))
        last = max(i for i, e in enumerate(evs) if "prompt_progress" in e)
        self.assertEqual("".join(c["delta"].get("content") or "" for e in evs[last:] for c in e["choices"]), "abc")
        # a follow-up that reuses the cached prompt reports it as cache
        prog = [e["prompt_progress"] for e in self._stream(http_, {"return_progress": True, "messages": [
            {"role": "user", "content": "hello"}, {"role": "assistant", "content": "abc"}, {"role": "user", "content": "more"}]})
            if "prompt_progress" in e]
        self.assertGreater(prog[0]["cache"], 0)
        self.assertEqual(prog[0]["processed"], prog[0]["cache"])
        self.assertIsNone(srv.on_progress)
        self.assertFalse(any("prompt_progress" in e for e in self._stream(http_, {})))  # (only when asked)

    def test_timings_per_token(self) -> None:
        _, http_ = self.serve(_ScriptEngine(["a", "b", "c", "d"]))
        evs = [e for e in self._stream(http_, {"timings_per_token": True}) if e["choices"] and e["choices"][0]["delta"].get("content")]
        n = [e["timings"]["predicted_n"] for e in evs]
        self.assertEqual(n, sorted(n))
        self.assertEqual(len(n), 4)
        self.assertTrue(all({"prompt_n", "prompt_ms", "predicted_n", "predicted_ms", "predicted_per_second"} <= set(e["timings"])
                            for e in evs))
        self.assertFalse(any("timings" in e for e in self._stream(http_, {}) if e["choices"] and e["choices"][0]["delta"].get("content")))

    def test_t_max_predict_ms(self) -> None:
        _, http_ = self.serve(_ScriptEngine(["line\n"] * 200, delay=0.02))
        body = {"messages": [{"role": "user", "content": "go"}], "max_tokens": 150}
        status, data = http_.post({**body, "t_max_predict_ms": 100})
        o = json.loads(data)
        self.assertEqual((status, o["choices"][0]["finish_reason"]), (200, "length"))
        self.assertLess(o["usage"]["completion_tokens"], 60)  # (~2 tokens per 20 ms cycle)
        self.assertTrue(o["choices"][0]["message"]["content"].endswith("line"))  # (ended on a newline token)
        o = json.loads(http_.post({**body, "t_max_predict_ms": 0})[1])
        self.assertEqual(o["usage"]["completion_tokens"], 150)
        _, http_ = self.serve(_ScriptEngine(["word "] * 100, delay=0.02))  # (no newline: no stop, as llama-server)
        o = json.loads(http_.post({**body, "max_tokens": 60, "t_max_predict_ms": 50})[1])
        self.assertEqual(o["usage"]["completion_tokens"], 60)

    def test_other_fields(self) -> None:
        _, http_ = self.serve(_ScriptEngine(["ok"]))
        base = {"messages": [{"role": "user", "content": "x"}], "max_tokens": 4}
        for extra in ({"n_keep": 10, "n_discard": 0, "cache_prompt": False, "id_slot": 0, "response_fields": ["content"],
                       "return_tokens": True, "n_cache_reuse": 256, "sse_ping_interval": 30, "verbose": True},
                      {"id_slot": -1, "n_cmpl": 1, "n_indent": 0, "t_max_predict_ms": -1},
                      {"samplers": "edkypmxt"}, {"samplers": "kpmt"}, {"samplers": ["top_k", "top-p", "minp", "temp"]},
                      {"samplers": ["penalties", "dry", "top_n_sigma", "top_k", "typ_p", "top_p", "min_p", "xtc", "temperature"]},
                      {"samplers": "ekpmt", "presence_penalty": 0.5}, {"samplers": ["top_k", "unknown", "top_p", "min_p", "temperature"]}):
            status, data = http_.post({**base, **extra})
            self.assertEqual(status, 200, (extra, data))
        for extra, field in (({"id_slot": 1}, "id_slot"), ({"n_cmpl": 2}, "n_cmpl"), ({"n_indent": 4}, "n_indent"),
                             ({"t_max_predict_ms": "soon"}, "t_max_predict_ms"), ({"samplers": "tkpm"}, "samplers"),
                             ({"samplers": "kpm"}, "samplers"), ({"samplers": "kpmt", "frequency_penalty": 1}, "samplers"),
                             ({"samplers": ["top_k", "adaptive_p", "top_p", "min_p", "temperature"]}, "samplers"),
                             ({"samplers": "kkpmt"}, "samplers"), ({"samplers": 3}, "samplers")):
            status, data = http_.post({**base, **extra})
            self.assertEqual((status, json.loads(data)["error"]["message"].split(":")[0]), (400, field), extra)


class ChunkedBodyTests(_HttpCase):
    """Transfer-Encoding: chunked request bodies (Node fetch with a stream body, Go, proxies), decoded into a buffer
    capped at MK_MAX_BODY_MB; malformed framing is a 400 and the connection closes"""

    HEAD = b"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\n"
    BODY = json.dumps({"messages": [{"role": "user", "content": "hi"}], "max_tokens": 4}).encode()

    def _status(self, raw):
        line = raw.split(b"\r\n", 1)[0]
        return int(line.split()[1]) if line.startswith(b"HTTP/1.1 ") else None

    @staticmethod
    def chunks(data, sizes, ext=b"", upper=False, trailers=b""):
        out, i = b"", 0
        for n in sizes:
            part, i = data[i:i + n], i + n
            out += (b"%X" if upper else b"%x") % len(part) + ext + b"\r\n" + part + b"\r\n"
        return out + b"0" + ext + b"\r\n" + trailers + b"\r\n"

    def test_valid_bodies(self) -> None:
        _, http_ = self.serve(_ScriptEngine(["ok"]))
        b = self.BODY
        for enc in (self.chunks(b, [len(b)]), self.chunks(b, [1] * len(b)), self.chunks(b, [7, 13, 1000], upper=True),
                    self.chunks(b, [10, 1000], ext=b" ;name=value", trailers=b"X-Checksum: abc\r\nX-More: 1\r\n")):
            for te in (b"chunked", b"Chunked", b" chunked "):
                raw = http_.raw(self.HEAD + b"Transfer-Encoding: " + te + b"\r\n\r\n" + enc, until=b"}")
                self.assertEqual(self._status(raw), 200, (enc[:60], raw[:300]))
                self.assertIn(b'"content": "ok"', raw)
        # keep-alive: the next request on the connection after a chunked one (body and trailers fully read)
        req = self.HEAD + b"Transfer-Encoding: chunked\r\n\r\n" + self.chunks(b, [5, 1000], trailers=b"T: 1\r\n")
        raw = http_.raw(req + req, until=b"}")
        if raw.count(b"HTTP/1.1 200") < 2:
            with socket.create_connection(("127.0.0.1", http_.port), timeout=5) as s:
                s.sendall(req + req)
                raw, t0 = b"", time.time()
                while raw.count(b'"content": "ok"') < 2 and time.time() - t0 < 5:
                    raw += s.recv(65536)
        self.assertEqual(raw.count(b"HTTP/1.1 200"), 2, raw[:600])
        # Expect: 100-continue, then the chunks
        with socket.create_connection(("127.0.0.1", http_.port), timeout=5) as s:
            s.sendall(self.HEAD + b"Transfer-Encoding: chunked\r\nExpect: 100-continue\r\n\r\n")
            self.assertTrue(s.recv(100).startswith(b"HTTP/1.1 100"))
            s.sendall(self.chunks(b, [20, 1000]))
            raw, t0 = b"", time.time()
            while b'"content": "ok"' not in raw and time.time() - t0 < 5:
                raw += s.recv(65536)
            self.assertEqual(self._status(raw), 200, raw[:300])

    def test_malformed_framing(self) -> None:
        _, http_ = self.serve(_ScriptEngine(["ok"]))
        b = self.BODY
        good = self.chunks(b, [len(b)])
        cases = [(b"zz\r\n" + b + b"\r\n0\r\n\r\n", 400, b"bad chunk size"),  # (not hex)
                 (b"\r\n" + b, 400, b"bad chunk size"),  # (empty size)
                 (b"%x\r\n%sXY0\r\n\r\n" % (len(b), b), 400, b"without its CRLF"),  # (no CRLF after the data)
                 (b"0" * 17 + b"1\r\n", 400, b"bad chunk size"),  # (over 16 hex digits)
                 (b"1" * 5000, 400, b"chunk-size line"),  # (a size line that never ends)
                 (b"-5\r\n", 400, b"bad chunk size"),
                 (b"%x\n%s\r\n0\r\n\r\n" % (len(b), b), 400, b"chunk-size line"),  # (LF only)
                 (b"%x\r\n%s\r\n0\r\n" % (len(b), b) + b"X: y\r\n" * 70 + b"\r\n", 400, b"trailer lines"),
                 (b"%x\r\n%s\r\n0\r\nX: " % (len(b), b) + b"y" * 5000, 400, b"trailer line")]
        for enc, code, words in cases:
            raw = http_.raw(self.HEAD + b"Transfer-Encoding: chunked\r\n\r\n" + enc)
            self.assertEqual(self._status(raw), code, (enc[:40], raw[:300]))
            self.assertIn(words, raw)
            self.assertIn(b"Connection: close", raw)
        for head, code in ((b"Transfer-Encoding: gzip, chunked\r\n", 501), (b"Transfer-Encoding: deflate\r\n", 501),
                           (b"Transfer-Encoding: chunked\r\nContent-Length: %d\r\n" % len(good), 400)):
            raw = http_.raw(self.HEAD + head + b"\r\n" + good)
            self.assertEqual(self._status(raw), code, (head, raw[:300]))
        self.assertEqual(http_.post({"messages": [{"role": "user", "content": "hi"}], "max_tokens": 2})[0], 200)

    def test_the_body_limit(self) -> None:
        from unittest import mock
        _, http_ = self.serve(_ScriptEngine(["ok"]))
        with mock.patch.object(self.mk, "MAX_BODY_MB", 1):
            t = time.time()
            raw = http_.raw(self.HEAD + b"Transfer-Encoding: chunked\r\n\r\n%x\r\n" % ((1 << 20) + 1))  # (refused unread)
            self.assertEqual((self._status(raw), b"MK_MAX_BODY_MB" in raw), (413, True), raw[:300])
            self.assertLess(time.time() - t, 2.0)
            # over the limit across chunks: refused when the chunk that passes it is announced
            piece = b"x" * 300000
            raw = http_.raw(self.HEAD + b"Transfer-Encoding: chunked\r\n\r\n" + (b"%x\r\n%s\r\n" % (len(piece), piece)) * 4)
            self.assertEqual(self._status(raw), 413, raw[:300])
            big = json.dumps({"messages": [{"role": "user", "content": "x" * 900000}], "max_tokens": 2}).encode()
            raw = http_.raw(self.HEAD + b"Transfer-Encoding: chunked\r\n\r\n" + self.chunks(big, [65536] * 16), until=b"}",
                            timeout=20)
            self.assertNotEqual(self._status(raw), 413, raw[:300])  # (under the limit: read whole)

    def test_a_client_that_leaves_mid_body(self) -> None:
        srv, http_ = self.serve(_ScriptEngine(["ok"]))
        with socket.create_connection(("127.0.0.1", http_.port), timeout=5) as s:
            s.sendall(self.HEAD + b"Transfer-Encoding: chunked\r\n\r\n100\r\nabc")
        time.sleep(0.2)
        self.assertEqual(http_.post({"messages": [{"role": "user", "content": "hi"}], "max_tokens": 2})[0], 200)


class SdkStreamTests(_HttpCase):
    """the official SDKs' stream accumulators rebuild the streamed calls (skipped when the SDK is not installed)"""

    TEXT = "Writing.\n\n" + _call("write", path="/tmp/a", content="x \"y\"\n" * 8, mode=7) + "\n" + _call("write", path="/b")

    def test_openai_sdk(self) -> None:
        try:
            import openai
        except ImportError:
            self.skipTest("openai is not installed")
        _, http_ = self.serve(_ScriptEngine(_pieces(self.TEXT)))
        client = openai.OpenAI(base_url=f"http://127.0.0.1:{http_.port}/v1", api_key="x", max_retries=0)
        self.addCleanup(client.close)
        kw = dict(model="m", messages=[{"role": "user", "content": "go"}], tools=WRITE)
        whole = client.chat.completions.create(**kw)
        stream = getattr(client.chat.completions, "stream", None) or client.beta.chat.completions.stream
        with stream(**kw) as s:
            final = s.get_final_completion()
        a, b = whole.choices[0], final.choices[0]
        self.assertEqual((b.finish_reason, b.message.content), (a.finish_reason, a.message.content))
        self.assertEqual([(t.function.name, t.function.arguments) for t in b.message.tool_calls],
                         [(t.function.name, t.function.arguments) for t in a.message.tool_calls])
        self.assertEqual(json.loads(b.message.tool_calls[0].function.arguments)["mode"], 7)

    def test_anthropic_sdk(self) -> None:
        try:
            import anthropic
        except ImportError:
            self.skipTest("anthropic is not installed")
        _, http_ = self.serve(_ScriptEngine(_pieces(self.TEXT)))
        client = anthropic.Anthropic(base_url=f"http://127.0.0.1:{http_.port}", api_key="x", max_retries=0)
        self.addCleanup(client.close)
        kw = dict(model="m", max_tokens=500, messages=[{"role": "user", "content": "go"}],
                  tools=[{"name": "write", "input_schema": WRITE[0]["function"]["parameters"]}])
        whole = client.messages.create(**kw)
        with client.messages.stream(**kw) as s:
            final = s.get_final_message()
        self.assertEqual(final.stop_reason, whole.stop_reason)
        self.assertEqual([b.type for b in final.content], [b.type for b in whole.content])
        self.assertEqual([b.input for b in final.content if b.type == "tool_use"],
                         [b.input for b in whole.content if b.type == "tool_use"])


if __name__ == "__main__":
    unittest.main()
