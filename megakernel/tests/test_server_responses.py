"""The OpenAI Responses API of server/mk_server.py (POST /v1/responses, /v1/responses/input_tokens) without a GPU, against
the fake engines of test_mk_server.py: the request mapping (every input item type, Codex's request), the tool-call round
trip, the streamed event sequence, the non-streamed Response, the errors, the rendered prompt's prefix stability over a
Codex-like exchange (stateless: every turn resends the whole conversation), and the official openai SDK's parsers when
installed."""
import base64
import http.client
import io
import json
import os
import struct
import sys
import tempfile
import time
import unittest
import zlib
from unittest import mock

from test_mk_server import ROOT, _HttpCase, _ScriptEngine, _load_mk_server

UNSLOTH = (ROOT / "tests" / "templates" / "qwen38_unsloth.jinja").read_text(encoding="utf-8")
ORIGINAL = (ROOT / "tests" / "templates" / "qwen38_original.jinja").read_text(encoding="utf-8")

EXEC = {"type": "function", "name": "exec_command", "description": "Runs a command in a PTY.", "strict": False,
        "parameters": {"type": "object", "properties": {"cmd": {"type": "string"}, "yield_time_ms": {"type": "number"}},
                       "required": ["cmd"], "additionalProperties": False}}
AGENTS = {"type": "namespace", "name": "multi_agent_v1", "description": "Tools for sub-agents.", "tools": [
    {"type": "function", "name": "spawn_agent", "description": "Spawns an agent.", "strict": False,
     "parameters": {"type": "object", "properties": {"message": {"type": "string"}}}}]}
PATCH = {"type": "custom", "name": "apply_patch", "description": "Edits files.",
         "format": {"type": "grammar", "syntax": "lark", "definition": "start: \"*** Begin Patch\" ..."}}
# Codex CLI 0.163.0's request (harness capture, texts shortened): its base prompt is a developer message, not instructions
CODEX_INPUT = [
    {"type": "message", "id": "msg_1", "role": "developer", "content": [
        {"type": "input_text", "text": "You are a coding agent running in the Codex CLI.\r\n\r\nBe precise.\r\n"}]},
    {"type": "message", "id": "msg_2", "role": "developer", "content": [
        {"type": "input_text", "text": "<skills_instructions>\n## Skills\n</skills_instructions>"},
        {"type": "input_text", "text": "<permissions instructions>\nsandbox_mode is workspace-write.\n</permissions instructions>"}]},
    {"type": "message", "id": "msg_3", "role": "user", "content": [
        {"type": "input_text", "text": "<environment_context>\n  <cwd>C:\\work</cwd>\n</environment_context>"}]},
    {"type": "message", "id": "msg_4", "role": "user", "content": [
        {"type": "input_text", "text": "Create a file named hello.txt containing the word hi, then stop."}]}]


def _codex(items, **kw):
    return {"model": "Qwen3.8-27B-UD-Q4_K_XL.gguf", "input": items, "tools": [EXEC, AGENTS, {"type": "web_search",
            "external_web_access": False}], "tool_choice": "auto", "parallel_tool_calls": True,
            "reasoning": {"summary": "auto"}, "store": False, "stream": True, "include": ["reasoning.encrypted_content"],
            "prompt_cache_key": "01a1", "client_metadata": {"turn_id": "t1"}, **kw}


def _call(name, **params):
    """Qwen's markup of a call (values as given: strings raw, others as their JSON text)"""
    body = "".join(f"<parameter={k}>\n{v if isinstance(v, str) else json.dumps(v)}\n</parameter>\n" for k, v in params.items())
    return f"<tool_call>\n<function={name}>\n{body}</function>\n</tool_call>"


def _pieces(text, n=3):
    return [text[i:i + n] for i in range(0, len(text), n)]


def _post(port, path, body, timeout=20):
    """(status, Content-Type, body: a JSON object, or a stream's [(event, data)])"""
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    c.request("POST", path, json.dumps(body) if not isinstance(body, bytes) else body, {"Content-Type": "application/json"})
    r = c.getresponse()
    data, ctype = r.read(), r.getheader("Content-Type")
    c.close()
    if ctype == "text/event-stream":
        events = []
        for block in data.decode().split("\n\n"):
            if block.strip():
                lines = dict(ln.split(": ", 1) for ln in block.split("\n"))
                events.append((lines["event"], json.loads(lines["data"])))
        return r.status, ctype, events
    return r.status, ctype, json.loads(data)


class RequestMappingTests(unittest.TestCase):
    """responses_to_chat: Responses requests as the server's chat requests"""

    @classmethod
    def setUpClass(cls) -> None:
        cls.mk = _load_mk_server()

    def test_codex_request(self) -> None:
        notes = []
        chat, info = self.mk.responses_to_chat(_codex(CODEX_INPUT), notes, mid_system=False)
        m = chat["messages"]
        self.assertEqual([x["role"] for x in m], ["system", "user", "user"])
        # the two developer messages: one system message, trimmed texts joined by a newline (parts: by newlines too)
        self.assertEqual(m[0]["content"], "You are a coding agent running in the Codex CLI.\r\n\r\nBe precise.\n"
                                          "<skills_instructions>\n## Skills\n</skills_instructions>\n<permissions instructions>\n"
                                          "sandbox_mode is workspace-write.\n</permissions instructions>")
        self.assertEqual(m[2]["content"], CODEX_INPUT[3]["content"][0]["text"])
        self.assertEqual(sorted(chat), ["messages", "stream", "tool_choice", "tools"])
        self.assertEqual((chat["stream"], chat["tool_choice"]), (True, "auto"))
        self.assertEqual([t["function"]["name"] for t in chat["tools"]], ["exec_command", "spawn_agent"])
        self.assertEqual(chat["tools"][0], {"type": "function", "function": {
            "name": "exec_command", "description": "Runs a command in a PTY.", "parameters": EXEC["parameters"]}})
        self.assertEqual(info, {"ns": {"spawn_agent": "multi_agent_v1"}, "custom": set()})
        self.assertEqual(sorted(k for k, _ in notes), ["responses-summary", "responses-tool-web_search"])
        # the unsloth template merges leading system / developer messages the same way: the prompt is unchanged
        tmpl = self.mk.make_template(UNSLOTH)
        sep = [{"role": "system" if x["role"] == "developer" else "user", "content": "\n".join(p["text"] for p in x["content"])}
               for x in CODEX_INPUT]
        self.assertEqual(tmpl.render(messages=m, tools=chat["tools"], add_generation_prompt=True),
                         tmpl.render(messages=sep, tools=chat["tools"], add_generation_prompt=True))
        # Qwen's original template takes one system message and has no developer role
        self.mk.make_template(ORIGINAL).render(messages=m, tools=chat["tools"], add_generation_prompt=True)

    def test_every_input_item_type(self) -> None:
        mk = self.mk
        img = "data:image/png;base64,iVBO"
        items = [
            {"role": "developer", "content": "Be brief."},  # (an easy input message: no type)
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "look: "},
                                                            {"type": "input_image", "image_url": img, "detail": "auto"}]},
            {"type": "reasoning", "id": "rs_1", "summary": [], "encrypted_content": None,
             "content": [{"type": "reasoning_text", "text": "I should read it."}]},
            {"type": "message", "role": "assistant", "content": [
                {"type": "output_text", "text": "Reading.", "annotations": [], "logprobs": []}]},
            {"type": "function_call", "id": "fc_1", "call_id": "call_1", "name": "exec_command", "arguments": "{\"cmd\": \"ls\"}"},
            {"type": "function_call", "call_id": "call_2", "name": "spawn_agent", "namespace": "multi_agent_v1",
             "arguments": "{}"},
            {"type": "function_call_output", "call_id": "call_1", "output": "a.txt"},
            {"type": "function_call_output", "call_id": "call_2", "output": [{"type": "input_text", "text": "x"},
                                                                             {"type": "input_text", "text": "y"}]},
            {"type": "reasoning", "summary": [], "encrypted_content": mk.reasoning_seal("Sealed thought.")},
            {"type": "custom_tool_call", "call_id": "call_3", "name": "apply_patch", "input": "*** Begin Patch"},
            {"type": "custom_tool_call_output", "call_id": "call_3", "output": "Done!"},
            {"type": "function_call_output", "call_id": "call_4", "output": [{"type": "input_image", "image_url": img}]},
            {"type": "reasoning", "summary": [{"type": "summary_text", "text": "S1"}, {"type": "summary_text", "text": "S2"}],
             "encrypted_content": "gAAAAopaque"},  # (another server's: its summary)
            {"type": "message", "role": "assistant", "content": "Next."},
            {"type": "message", "role": "developer", "content": "Mode changed."},  # (mid-conversation)
            {"type": "web_search_call", "id": "ws_1", "status": "completed"}]
        notes = []
        chat, _ = mk.responses_to_chat({"instructions": "You are X.", "input": items, "tools": [EXEC, AGENTS, PATCH]},
                                       notes, mid_system=False)
        want = [{"role": "system", "content": "You are X.\nBe brief."},
                {"role": "user", "content": [{"type": "text", "text": "look: "}, {"type": "image_url", "image_url": {"url": img}}]},
                {"role": "assistant", "content": "Reading.", "reasoning_content": "I should read it.", "tool_calls": [
                    {"id": "call_1", "type": "function", "function": {"name": "exec_command", "arguments": "{\"cmd\": \"ls\"}"}},
                    {"id": "call_2", "type": "function", "function": {"name": "spawn_agent", "arguments": "{}"}}]},
                {"role": "tool", "tool_call_id": "call_1", "content": "a.txt"},
                {"role": "tool", "tool_call_id": "call_2", "content": "x\ny"},
                {"role": "assistant", "content": "", "reasoning_content": "Sealed thought.", "tool_calls": [
                    {"id": "call_3", "type": "function", "function": {"name": "apply_patch",
                                                                      "arguments": "{\"input\": \"*** Begin Patch\"}"}}]},
                {"role": "tool", "tool_call_id": "call_3", "content": "Done!"},
                {"role": "tool", "tool_call_id": "call_4", "content": [{"type": "image_url", "image_url": {"url": img}}]},
                {"role": "assistant", "content": "Next.", "reasoning_content": "S1\n\nS2"},
                {"role": "user", "content": "Mode changed."}]
        self.assertEqual(chat["messages"], want)
        self.assertEqual(sorted({k for k, _ in notes}), ["responses-item-web_search_call", "responses-mid-system"])
        # a string input is one user message; mid_system: a template that takes it keeps the system message
        self.assertEqual(mk.responses_to_chat({"input": "hi"})[0]["messages"], [{"role": "user", "content": "hi"}])
        late = mk.responses_to_chat({"input": [{"role": "user", "content": "a"}, {"role": "system", "content": "b"}]})[0]
        self.assertEqual([x["role"] for x in late["messages"]], ["user", "system"])

    def test_assistant_items_split_where_the_model_could_not_have_generated_them_together(self) -> None:
        call = {"type": "function_call", "call_id": "c", "name": "f", "arguments": "{}"}
        say = {"type": "message", "role": "assistant", "content": "t"}
        think = {"type": "reasoning", "summary": [], "content": [{"type": "reasoning_text", "text": "r"}]}
        for items, n in (([think, say, call, call], 1), ([say, say], 2), ([call, say], 2), ([say, think], 2),
                         ([think, think], 2), ([call, think, call], 2)):
            m = self.mk.responses_to_chat({"input": items})[0]["messages"]
            self.assertEqual(len(m), n, items)

    def test_parameters(self) -> None:
        mk = self.mk
        body = {"input": "hi", "max_output_tokens": 300, "temperature": 0.2, "top_p": 0.9, "top_k": 5, "seed": 3,
                "parallel_tool_calls": False, "tool_choice": "none", "reasoning": {"effort": "low", "summary": None},
                "text": {"format": {"type": "text"}, "verbosity": "medium"}, "store": True, "metadata": {"a": "b"},
                "user": "u", "safety_identifier": "s", "service_tier": "auto", "truncation": "auto", "max_tool_calls": 3,
                "include": ["reasoning.encrypted_content", "web_search_call.action.sources"], "stream_options": {}}
        notes = []
        chat, _ = mk.responses_to_chat(body, notes)
        self.assertEqual({k: v for k, v in chat.items() if k != "messages"}, {
            "stream": False, "max_tokens": 300, "temperature": 0.2, "top_p": 0.9, "top_k": 5, "seed": 3,
            "parallel_tool_calls": False, "tool_choice": "none", "reasoning_effort": "low"})
        self.assertEqual(notes, [])
        mk.responses_to_chat({**body, "text": {"verbosity": "low"}}, notes)
        self.assertEqual([k for k, _ in notes], ["responses-verbosity"])

    def test_what_cannot_be_honoured_is_a_400(self) -> None:
        for bad in ({"input": "x", "previous_response_id": "resp_1"}, {"input": "x", "conversation": "conv_1"},
                    {"input": "x", "background": True}, {"input": "x", "prompt": {"id": "pmpt_1"}}, {"input": 5}, {},
                    {"input": [{"type": "item_reference", "id": "msg_1"}]}, {"input": "x", "tool_choice": "required"},
                    {"input": "x", "tool_choice": {"type": "function", "name": "f"}},
                    {"input": "x", "text": {"format": {"type": "json_schema", "name": "s", "schema": {}}}},
                    {"input": "x", "text": {"format": {"type": "json_object"}}},
                    {"input": "x", "include": ["message.output_text.logprobs"]},
                    {"input": [{"role": "user", "content": [{"type": "input_file", "file_id": "file_1"}]}]},
                    {"input": [{"role": "user", "content": [{"type": "input_image", "file_id": "file_1"}]}]},
                    {"input": [{"role": "user", "content": [{"type": "input_audio"}]}]},
                    {"input": [{"role": "tool", "content": "x"}]}, {"input": ["x"]},
                    {"input": [{"type": "function_call", "call_id": "c", "arguments": "{}"}]},
                    {"input": "x", "tools": [EXEC, EXEC]}, {"input": "x", "tools": [{"type": "function"}]},
                    {"input": "x", "tools": {"type": "function"}}, {"input": "x", "instructions": ["x"]},
                    {"input": "x", "reasoning": "high"}):
            with self.assertRaises(self.mk.BadRequest, msg=str(bad)):
                self.mk.responses_to_chat(bad)
        # (top_logprobs reaches check_params, as on the chat path)
        with self.assertRaises(self.mk.BadRequest):
            self.mk.check_params(self.mk.responses_to_chat({"input": "x", "top_logprobs": 3})[0])

    def test_custom_tools_are_functions_with_one_string_input(self) -> None:
        mk = self.mk
        tools, ns, custom = mk.responses_tools([PATCH], [])
        f = tools[0]["function"]
        self.assertEqual((f["name"], f["parameters"]["required"], custom, ns), ("apply_patch", ["input"], {"apply_patch"}, {}))
        self.assertIn(PATCH["format"]["definition"], f["parameters"]["properties"]["input"]["description"])
        item = mk.responses_call("call_9", "apply_patch", json.dumps({"input": "*** Begin Patch\n"}),
                                 {"ns": {}, "custom": custom})
        self.assertEqual(item, {"type": "custom_tool_call", "id": "ctc_9", "call_id": "call_9", "name": "apply_patch",
                                "input": "*** Begin Patch\n", "status": "completed"})

    def test_reasoning_seal(self) -> None:
        mk = self.mk
        text = "Thinking é中 " * 50
        self.assertEqual(mk.reasoning_open(mk.reasoning_seal(text)), text)
        sealed = mk.reasoning_seal(text)
        cut = mk.RESP_REASONING + base64.b64encode(base64.b64decode(sealed[len(mk.RESP_REASONING):])[:-6]).decode()
        for s in (None, "gAAAAB", mk.RESP_REASONING + "!!", mk.RESP_REASONING + "aGk=", 5, cut):  # (cut: truncated)
            self.assertIsNone(mk.reasoning_open(s), s)
        # a zlib stream that inflates past what the model's full context holds: a 400, never a large allocation
        with mock.patch.object(mk, "REASONING_MAX_BYTES", 1 << 20):
            bomb = mk.RESP_REASONING + base64.b64encode(zlib.compress(b"x" * ((1 << 20) + 10))).decode()
            with self.assertRaises(mk.BadRequest):
                mk.reasoning_open(bomb)

    def test_reasoning_budget_is_per_request(self) -> None:
        mk = self.mk
        item = {"type": "reasoning", "summary": [], "encrypted_content": mk.reasoning_seal("y" * 600)}
        user = {"type": "message", "role": "user", "content": "hi"}
        with mock.patch.object(mk, "REASONING_MAX_BYTES", 1000):
            chat, _ = mk.responses_to_chat({"model": "m", "input": [item, user]})
            self.assertEqual(chat["messages"][0]["reasoning_content"], "y" * 600)
            # each item fits alone, together they pass the budget: many small items cannot inflate to gigabytes
            with self.assertRaises(mk.BadRequest):
                mk.responses_to_chat({"model": "m", "input": [item, user, item, user]})

    def test_gguf_text_limits(self) -> None:
        mk = self.mk

        def s(b):
            return struct.pack("<Q", len(b)) + b
        kv = [s(b"general.architecture") + struct.pack("<I", 8) + s(b"qwen35"),
              s(b"llama.context_length") + struct.pack("<II", 4, 4096),  # (another architecture's: ignored)
              s(b"qwen35.context_length") + struct.pack("<II", 4, 262144),
              s(b"tokenizer.ggml.scores") + struct.pack("<IIQ3f", 9, 6, 3, 0, 0, 0),
              s(b"tokenizer.ggml.tokens") + struct.pack("<IIQ", 9, 8, 3) + s(b"a") + s("ĠĠĠ".encode())
              + s(b"<|im_start|>")]
        fd, path = tempfile.mkstemp(suffix=".gguf")
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(b"GGUF" + struct.pack("<IQQ", 3, 0, len(kv)) + b"".join(kv))
            self.assertEqual(mk.gguf_text_limits(path), (262144, 12))
            with open(path, "r+b") as f:  # (truncated inside the token list: what was read before it)
                f.truncate(len(b"GGUF") + 20 + sum(map(len, kv[:4])) + 30)
            self.assertEqual(mk.gguf_text_limits(path), (262144, None))
            with open(path, "wb") as f:
                f.write(b"not a gguf file")
            self.assertEqual(mk.gguf_text_limits(path), (None, None))
        finally:
            os.remove(path)
        self.assertEqual(mk.gguf_text_limits(path), (None, None))


def _items(events):
    """the output items of a stream's output_item.done events, by output_index"""
    return {d["output_index"]: d["item"] for n, d in events if n == "response.output_item.done"}


class ResponsesHttpTests(_HttpCase):
    """POST /v1/responses (+ stream) and /v1/responses/input_tokens over the chat path"""

    THINK = "The user wants a file.\n</think>\n\n"
    WRITE = _call("exec_command", cmd="Set-Content hello.txt hi")
    REPLY = THINK + "Creating it.\n\n" + WRITE

    def serve_unsloth(self, pieces, **kw):
        srv, http_ = self.serve(_ScriptEngine(pieces, **kw))
        srv.template = self.mk.make_template(UNSLOTH)
        return srv, http_

    def check_stream(self, events):
        """the stream's invariants (what Codex and the openai SDK rely on); returns the final response"""
        names = [n for n, _ in events]
        self.assertEqual(names[:2], ["response.created", "response.in_progress"])
        self.assertIn(names[-1], ("response.completed", "response.incomplete", "response.failed"))
        self.assertEqual([d["sequence_number"] for _, d in events], list(range(len(events))))
        for n, d in events:
            self.assertEqual(d["type"], n)
        final = events[-1][1]["response"]
        self.assertEqual({d["response"]["id"] for n, d in events if "response" in d}, {final["id"]})
        added, done, opened = {}, {}, None
        for n, d in events:
            if n == "response.output_item.added":
                self.assertIsNone(opened, "an item opened before the previous one was done")
                self.assertEqual(d["output_index"], len(added))
                added[d["output_index"]], opened = d["item"], d["output_index"]
            elif n == "response.output_item.done":
                self.assertEqual(d["output_index"], opened)
                self.assertEqual((d["item"]["type"], d["item"]["id"]), (added[opened]["type"], added[opened]["id"]))
                done[opened], opened = d["item"], None
            elif "output_index" in d:  # (a content event: for the open item, by its id)
                self.assertEqual((d["output_index"], d["item_id"]), (opened, added[opened]["id"]), n)
        if final["status"] != "failed":
            self.assertIsNone(opened)
            self.assertEqual(final["output"], [done[i] for i in sorted(done)])
        # deltas add up to the done texts
        for i, item in done.items():
            deltas = "".join(d.get("delta", "") for n, d in events if d.get("output_index") == i and n.endswith(".delta"))
            whole = {"reasoning": lambda x: x["content"][0]["text"], "message": lambda x: x["content"][0]["text"],
                     "function_call": lambda x: x["arguments"], "custom_tool_call": lambda x: x["input"]}[item["type"]](item)
            self.assertEqual(deltas, whole, item["type"])
        return final

    def test_non_streamed_response(self) -> None:
        srv, http_ = self.serve_unsloth(_pieces(self.REPLY))
        body = _codex(CODEX_INPUT, stream=False)
        status, ctype, o = _post(http_.port, "/v1/responses", body)
        self.assertEqual((status, ctype), (200, "application/json"), o)
        self.assertEqual((o["object"], o["status"], o["model"], o["store"], o["error"], o["incomplete_details"]),
                         ("response", "completed", "test", False, None, None))
        self.assertTrue(o["id"].startswith("resp_"))
        self.assertEqual([x["type"] for x in o["output"]], ["reasoning", "message", "function_call"])
        rs, msg, fc = o["output"]
        self.assertEqual((rs["summary"], rs["content"], self.mk.reasoning_open(rs["encrypted_content"])),
                         ([], [{"type": "reasoning_text", "text": "The user wants a file."}], "The user wants a file."))
        self.assertEqual((msg["role"], msg["status"], msg["content"]), ("assistant", "completed", [
            {"type": "output_text", "text": "Creating it.", "annotations": [], "logprobs": []}]))
        self.assertEqual((fc["name"], json.loads(fc["arguments"]), fc["status"], fc["call_id"][:5], fc["id"][:3]),
                         ("exec_command", {"cmd": "Set-Content hello.txt hi"}, "completed", "call_", "fc_"))
        self.assertNotIn("namespace", fc)
        u = o["usage"]
        n = len(srv.render(self.mk.responses_to_chat(body)[0])[0])  # (a character tokenizer)
        pieces = _pieces(self.REPLY)
        think = next(k for k in range(len(pieces)) if "</think>" in "".join(pieces[:k]))  # (to the one that ends it)
        self.assertEqual(u, {"input_tokens": n, "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
                             "output_tokens": len(pieces) + 1, "output_tokens_details": {"reasoning_tokens": think},
                             "total_tokens": n + len(pieces) + 1})
        self.assertEqual((o["tools"], o["parallel_tool_calls"], o["tool_choice"], o["reasoning"]),
                         (body["tools"], True, "auto", {"effort": None, "summary": "auto"}))
        # without include: no encrypted_content
        _, _, o = _post(http_.port, "/v1/responses", {**body, "include": []})
        self.assertIsNone(o["output"][0]["encrypted_content"])

    def test_streamed_events(self) -> None:
        srv, http_ = self.serve_unsloth(_pieces(self.REPLY))
        status, ctype, ev = _post(http_.port, "/v1/responses", _codex(CODEX_INPUT))
        self.assertEqual((status, ctype), (200, "text/event-stream"))
        final = self.check_stream(ev)
        self.assertEqual(final["status"], "completed")
        names = [n for n, _ in ev if not n.endswith(".delta")]
        self.assertEqual(names, [
            "response.created", "response.in_progress",
            "response.output_item.added", "response.content_part.added", "response.reasoning_text.done",
            "response.content_part.done", "response.output_item.done",
            "response.output_item.added", "response.content_part.added", "response.output_text.done",
            "response.content_part.done", "response.output_item.done",
            "response.output_item.added", "response.function_call_arguments.done", "response.output_item.done",
            "response.completed"])
        _, _, whole = _post(http_.port, "/v1/responses", _codex(CODEX_INPUT, stream=False))
        strip = ("id", "call_id", "encrypted_content")
        self.assertEqual([{k: v for k, v in x.items() if k not in strip} for x in final["output"]],
                         [{k: v for k, v in x.items() if k not in strip} for x in whole["output"]])
        self.assertEqual(final["usage"]["output_tokens"], whole["usage"]["output_tokens"])
        rs = final["output"][0]
        self.assertEqual(self.mk.reasoning_open(rs["encrypted_content"]), rs["content"][0]["text"])
        added = [d["item"] for n, d in ev if n == "response.output_item.added"]
        self.assertEqual([(x["type"], x["status"]) for x in added], [("reasoning", "in_progress"), ("message", "in_progress"),
                                                                      ("function_call", "in_progress")])
        self.assertEqual(added[2]["arguments"], "")

    def test_a_namespaced_function_and_a_custom_tool(self) -> None:
        text = self.THINK + _call("spawn_agent", message="go") + "\n" + _call("apply_patch", input="*** Begin Patch\n*** End Patch")
        srv, http_ = self.serve_unsloth(_pieces(text))
        body = _codex(CODEX_INPUT, tools=[EXEC, AGENTS, PATCH])
        _, _, ev = _post(http_.port, "/v1/responses", body)
        final = self.check_stream(ev)
        fc, ctc = final["output"][1:]
        self.assertEqual((fc["type"], fc["name"], fc["namespace"], json.loads(fc["arguments"])),
                         ("function_call", "spawn_agent", "multi_agent_v1", {"message": "go"}))
        self.assertEqual((ctc["type"], ctc["name"], ctc["input"]), ("custom_tool_call", "apply_patch", "*** Begin Patch\n*** End Patch"))
        names = [n for n, _ in ev]
        self.assertEqual(names.count("response.custom_tool_call_input.done"), 1)
        self.assertNotIn("response.function_call_arguments.delta",
                         [n for n, d in ev if d.get("output_index") == 2])
        _, _, o = _post(http_.port, "/v1/responses", {**body, "stream": False})
        self.assertEqual([(x["type"], x.get("namespace"), x.get("input")) for x in o["output"][1:]],
                         [("function_call", "multi_agent_v1", None), ("custom_tool_call", None, "*** Begin Patch\n*** End Patch")])

    def test_max_output_tokens_is_incomplete(self) -> None:
        srv, http_ = self.serve_unsloth(_pieces(self.REPLY))
        body = _codex(CODEX_INPUT, max_output_tokens=12)
        _, _, ev = _post(http_.port, "/v1/responses", body)
        final = self.check_stream(ev)
        self.assertEqual((ev[-1][0], final["status"], final["incomplete_details"]),
                         ("response.incomplete", "incomplete", {"reason": "max_output_tokens"}))
        _, _, o = _post(http_.port, "/v1/responses", {**body, "stream": False})
        self.assertEqual((o["status"], o["incomplete_details"], o["usage"]["output_tokens"]),
                         ("incomplete", {"reason": "max_output_tokens"}, 12))
        # a call cut by max_output_tokens: started in the stream, done as incomplete, its markup returned as text
        cut = len(_pieces(self.THINK + "Creating it.\n\n")) + 16  # (inside its first parameter)
        _, _, ev = _post(http_.port, "/v1/responses", {**body, "max_output_tokens": cut})
        final = self.check_stream(ev)
        self.assertEqual([(x["type"], x["status"]) for x in final["output"]],
                         [("reasoning", "completed"), ("message", "completed"), ("function_call", "incomplete")])

    def test_errors(self) -> None:
        srv, http_ = self.serve_unsloth(["ok"])
        for body in ({"input": "x", "previous_response_id": "resp_1"}, {"input": "x", "tool_choice": "required"},
                     {"input": "x", "text": {"format": {"type": "json_schema"}}}, {"input": "x", "n": 2, "top_logprobs": 2},
                     {"input": "x", "max_output_tokens": "x"}, {"input": "x", "temperature": "hot"}):
            for stream in (False, True):
                status, ctype, o = _post(http_.port, "/v1/responses", {**body, "stream": stream})
                self.assertEqual((status, ctype, sorted(o["error"]), o["error"]["type"], o["error"]["code"]),
                                 (400, "application/json", ["code", "message", "param", "type"], "invalid_request_error", None), body)
        status, _, o = _post(http_.port, "/v1/responses", b"[1]")
        self.assertEqual((status, o["error"]["type"], o["error"]["code"]), (400, "invalid_request_error", None))
        for path in ("/v1/responses/compact", "/responses/compact"):
            status, _, o = _post(http_.port, path, {"input": "x"})
            self.assertEqual((status, o["error"]["type"], o["error"]["code"]), (501, "not_supported", None))
        # a prompt over the context: OpenAI's code, before the stream starts
        srv.args.ctx = 100
        for stream in (False, True):
            status, _, o = _post(http_.port, "/v1/responses", {"input": "x" * 200, "stream": stream})
            self.assertEqual((status, o["error"]["code"], o["error"]["n_ctx"]), (400, "context_length_exceeded", 100))
        srv.args.ctx = 4096
        # an engine failure once the stream has started: response.failed (Codex retries a server_error)
        srv.eng = _ScriptEngine(_pieces(self.REPLY), fail_at=3)
        _, _, ev = _post(http_.port, "/v1/responses", _codex(CODEX_INPUT))
        final = self.check_stream(ev)
        self.assertEqual((ev[-1][0], final["status"], final["error"]["code"]), ("response.failed", "failed", "server_error"))
        self.assertIn("no finite logit", final["error"]["message"])
        srv.eng = _ScriptEngine(_pieces(self.REPLY), fail_at=3)
        status, _, o = _post(http_.port, "/v1/responses", _codex(CODEX_INPUT, stream=False))
        self.assertEqual((status, o["error"]["type"]), (500, "server_error"))

    def test_a_long_prefill_sends_in_progress_events(self) -> None:
        class SlowPrefill(_ScriptEngine):  # (1.2 s of prefill, its progress readable meanwhile)
            t0 = None

            def prefill(self, toks, ck_at=-1, cks=(), images=()):
                self.t0 = time.monotonic()
                time.sleep(1.2)
                return super().prefill(toks, ck_at, cks, images)

            def prefill_progress(self):
                return (int((time.monotonic() - self.t0) * 1000) if self.t0 else 0, 2000)
        srv, http_ = self.serve(SlowPrefill(_pieces(self.REPLY)))
        srv.template = self.mk.make_template(UNSLOTH)
        with mock.patch.object(self.mk, "RESP_KEEPALIVE_S", 0.3):
            _, _, ev = _post(http_.port, "/v1/responses", _codex(CODEX_INPUT))
        self.check_stream(ev)
        names = [n for n, _ in ev]
        first = names.index("response.output_item.added")
        self.assertGreaterEqual(names[:first].count("response.in_progress"), 3, names[:first])
        self.assertEqual(set(names[1:first]), {"response.in_progress"})

    def test_input_tokens(self) -> None:
        srv, http_ = self.serve_unsloth(["ok"])
        body = _codex(CODEX_INPUT)
        for path in ("/v1/responses/input_tokens", "/responses/input_tokens"):
            status, _, o = _post(http_.port, path, body)
            self.assertEqual((status, o), (200, {"object": "response.input_tokens", "input_tokens": len(
                srv.render(self.mk.responses_to_chat(body)[0])[0])}))
        status, _, o = _post(http_.port, "/v1/responses/input_tokens", {"input": "x", "conversation": "c"})
        self.assertEqual((status, o["error"]["type"]), (400, "invalid_request_error"))

    def test_console_lines(self) -> None:
        srv, http_ = self.serve_unsloth(_pieces(self.REPLY))
        err = io.StringIO()
        with mock.patch.object(sys, "stderr", err), mock.patch.object(self.mk, "CON", self.mk.Console()):
            _post(http_.port, "/v1/responses", _codex(CODEX_INPUT))
            _post(http_.port, "/v1/responses", _codex(CODEX_INPUT))
        lines = [ln.split(" ", 2)[2] for ln in err.getvalue().splitlines() if ln.startswith("[mk-server] ")]
        n = int(lines[2].split()[0][1:])
        self.assertTrue(lines[0].startswith("warning: Responses reasoning summaries are not generated"), lines)
        self.assertTrue(lines[1].startswith("warning: Responses tool type 'web_search' is not supported"), lines)
        self.assertRegex(lines[2], rf"^#{n} responses stream 'Qwen3.8-27B-UD-Q4_K_XL.gguf': [\d,]+ tok, 3 messages, 2 tools "
                                   r"\| thinking \| temp 1 top_p 0.95 top_k 20 min_p 0$")
        self.assertRegex(lines[3], rf"^#{n} prompt: 0 cached \+ [\d,]+ prefilled")
        self.assertRegex(lines[4], rf"^#{n} reply: .* \| thinking \d+, answer \d+, tool call \d+ \| .*finish tool_calls \(1 call\)")
        self.assertRegex(lines[5], rf"^#{n + 1} responses stream")  # (the warnings: once)


class PrefixStabilityTests(_HttpCase):
    """Codex is stateless: every turn resends the whole conversation (its output items as received, the tool outputs,
    new user messages). Turn N+1's rendered prompt must begin with turn N's prompt followed by turn N's reply as it
    was generated, so the engine reuses all of it (here: a character tokenizer and the unsloth template)"""

    REPLIES = ["The user wants hello.txt.\n</think>\n\n" + _call("exec_command", cmd="Set-Content hello.txt hi"),
               "It worked.\n</think>\n\nCreated hello.txt containing hi.",
               "Add mean_abs.\n</think>\n\nAdding it.\n\n" + _call("exec_command", cmd="type calc.py", yield_time_ms=10000)
               + "\n" + _call("spawn_agent", message="write the test"),
               "Both done.\n</think>\n\nDone."]

    def _exchange(self, keep):
        """runs the 4 turns (a follow-up user message before the third); keep(item) -> the item Codex sends back.
        Returns [(prompt text, the response)]"""
        srv, http_ = self.serve(_ScriptEngine([]))
        srv.template = self.mk.make_template(UNSLOTH)
        items, out = list(CODEX_INPUT), []
        for k, reply in enumerate(self.REPLIES):
            if k == 2:
                items.append({"type": "message", "role": "user", "content": [
                    {"type": "input_text", "text": "Now add mean_abs(values) to calc.py with a test."}]})
            srv.eng.pieces = [p.encode() for p in _pieces(reply)]
            body = _codex(items)
            prompt = srv.render(self.mk.responses_to_chat(body)[0])[0]
            _, _, ev = _post(http_.port, "/v1/responses", body)
            out.append((prompt, ev[-1][1]["response"]))
            done = list(_items(ev).values())
            items += [keep(item) for item in done]
            items += [{"type": "function_call_output", "call_id": item["call_id"], "output": "Exit code: 0\nOutput:\n"}
                      for item in done if item["type"] == "function_call"]
        return out

    def check(self, out):
        for k in range(1, len(out)):
            prev, cur = out[k - 1][0], out[k][0]
            self.assertTrue(cur.startswith(prev + self.REPLIES[k - 1] + "<|im_end|>\n"), (k, cur[len(prev):len(prev) + 300]))
            self.assertEqual(out[k][1]["usage"]["input_tokens_details"]["cached_tokens"], len(prev), k)
            self.assertEqual(out[k][1]["usage"]["input_tokens"], len(cur), k)

    def test_codex_like_exchange(self) -> None:
        out = self._exchange(lambda item: item)
        self.check(out)
        self.assertEqual([[x["type"] for x in r["output"]] for _, r in out], [
            ["reasoning", "function_call"], ["reasoning", "message"],
            ["reasoning", "message", "function_call", "function_call"], ["reasoning", "message"]])

    def test_reasoning_returned_only_as_encrypted_content(self) -> None:
        def keep(item):
            if item["type"] == "reasoning":
                return {"type": "reasoning", "id": item["id"], "summary": [], "encrypted_content": item["encrypted_content"]}
            return {k: v for k, v in item.items() if k != "status"}
        self.check(self._exchange(keep))

    def test_the_original_template_too(self) -> None:
        srv, http_ = self.serve(_ScriptEngine(_pieces(self.REPLIES[0])))
        srv.template = self.mk.make_template(ORIGINAL)
        status, _, ev = _post(http_.port, "/v1/responses", _codex(CODEX_INPUT))
        self.assertEqual((status, ev[-1][0]), (200, "response.completed"))


class OpenAISdkTests(_HttpCase):
    """the official openai SDK (a strict stream accumulator) reads the responses, streamed and not"""

    def test_openai_sdk(self) -> None:
        try:
            import openai
        except ImportError:
            self.skipTest("openai is not installed")
        reply = ResponsesHttpTests.REPLY + "\n" + _call("spawn_agent", message="go")
        srv, http_ = self.serve(_ScriptEngine(_pieces(reply)))
        srv.template = self.mk.make_template(UNSLOTH)
        client = openai.OpenAI(base_url=f"http://127.0.0.1:{http_.port}/v1", api_key="x", max_retries=0)
        self.addCleanup(client.close)
        kw = {k: v for k, v in _codex(CODEX_INPUT).items() if k not in ("stream", "client_metadata")}
        kw["tools"] = [EXEC, AGENTS]  # (the SDK's tool types; web_search is dropped by the server anyway)
        whole = client.responses.create(**kw)
        self.assertEqual([x.type for x in whole.output], ["reasoning", "message", "function_call", "function_call"])
        self.assertEqual(whole.output_text, "Creating it.")
        self.assertEqual((whole.output[3].name, whole.output[3].namespace), ("spawn_agent", "multi_agent_v1"))
        self.assertEqual(whole.usage.output_tokens_details.reasoning_tokens, srv.reasoning_tokens)
        with client.responses.stream(**kw) as stream:
            kinds = [e.type for e in stream]
            final = stream.get_final_response()
        self.assertEqual(kinds[0], "response.created")
        self.assertEqual([x.type for x in final.output], [x.type for x in whole.output])
        self.assertEqual(final.output_text, whole.output_text)
        self.assertEqual([json.loads(x.arguments) for x in final.output[2:]], [json.loads(x.arguments) for x in whole.output[2:]])
        self.assertEqual(final.output[0].content[0].text, whole.output[0].content[0].text)
        raw = list(client.responses.create(**kw, stream=True))
        self.assertEqual(raw[-1].type, "response.completed")
        # the stream's items go back as input, as a stateless client sends them
        nxt = client.responses.create(**{**kw, "input": CODEX_INPUT + [x.model_dump(exclude_none=True) for x in final.output] + [
            {"type": "function_call_output", "call_id": final.output[2].call_id, "output": "ok"},
            {"type": "function_call_output", "call_id": final.output[3].call_id, "output": "ok"}]})
        self.assertEqual(nxt.status, "completed")
        self.assertGreater(nxt.usage.input_tokens_details.cached_tokens, 0)
        tc = client.responses.input_tokens.count(model="m", input="hi") if hasattr(client.responses, "input_tokens") else None
        if tc is not None:
            self.assertEqual(tc.input_tokens, len(srv.render({"messages": [{"role": "user", "content": "hi"}]})[0]))


if __name__ == "__main__":
    unittest.main()
