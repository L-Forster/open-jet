"""The engine server's Anthropic Messages and OpenAI Responses endpoints, over HTTP with a scripted engine."""
import http.client
import importlib.util
import json
import sys
import threading
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent

TEMPLATE = (
    "{% for m in messages %}<|{{ m.role }}|>{{ m.content }}"
    "{% if m.reasoning_content %}[r:{{ m.reasoning_content }}]{% endif %}"
    "{% for c in m.tool_calls or [] %}[call:{{ c.function.name }}:{{ c.function.arguments | tojson }}]{% endfor %}\n"
    "{% endfor %}{% if tools %}[tools:{% for t in tools %}{{ t.function.name }},{% endfor %}]{% endif %}"
    "<|assistant|>{% if enable_thinking is not defined or enable_thinking %}<think>\n{% endif %}"
)
REPLY = ("Check the file.\n</think>\n\nReading it now.\n\n<tool_call>\n<function=read_file>\n<parameter=path>\n"
         "a.txt\n</parameter>\n</function>\n</tool_call>")
PATCH_REPLY = ("</think>\n\n<tool_call>\n<function=apply_patch>\n<parameter=input>\n*** Begin Patch\n*** End Patch\n"
               "</parameter>\n</function>\n</tool_call>")


def _load_mk_server():
    path = ROOT / "megakernel" / "server" / "mk_server.py"
    spec = importlib.util.spec_from_file_location("mk_server_api_under_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class ScriptedEngine:
    """tokens are code points; generation replays `reply` three tokens per step, then end of generation"""

    def __init__(self):
        self.reply = REPLY
        self.prompts = []

    def tokenize(self, text):
        return [ord(c) for c in text]

    def piece(self, tok):
        return chr(tok).encode()

    def is_eog(self, tok):
        return tok == 0

    def start(self, toks):
        self.prompts.append("".join(chr(t) for t in toks))
        self.out = [ord(c) for c in self.reply] + [0]
        return self.out.pop(0)

    def step(self):
        taken, self.out = self.out[:3], self.out[3:]
        return taken


def _events(raw):
    """(event, data) pairs of a server-sent event stream"""
    out = []
    for block in raw.decode().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines() if ": " in line)
        if "data" in lines:
            out.append((lines.get("event"), json.loads(lines["data"])))
    return out


class EngineApiCompatTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.mk = mk = _load_mk_server()
        srv = object.__new__(mk.Server)
        srv.args = SimpleNamespace(ctx=4096)
        srv.eng = cls.eng = ScriptedEngine()
        srv.template = mk.make_template(TEMPLATE)
        srv.model_id = "Qwen3.8-27B-Q4_K_M.gguf"
        srv.lock = threading.Lock()
        srv.sched = None  # one request at a time (no --parallel)
        srv.shared_text = None
        srv.warned = set()
        srv.sampling = lambda body: None
        srv.load_prompt = lambda toks: (cls.eng.start(toks), 2)
        cls.httpd = ThreadingHTTPServer(("127.0.0.1", 0), mk.make_handler(srv))
        cls.httpd.daemon_threads = True
        threading.Thread(target=cls.httpd.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.httpd.shutdown()
        cls.httpd.server_close()

    def setUp(self) -> None:
        self.eng.reply = REPLY

    def _post(self, path, body):
        conn = http.client.HTTPConnection("127.0.0.1", self.httpd.server_address[1], timeout=10)
        conn.request("POST", path, json.dumps(body), {"Content-Type": "application/json"})
        resp = conn.getresponse()
        data = resp.read()
        conn.close()
        return resp.status, data

    # Anthropic Messages (Claude Code)

    claude_request = {
        "model": "claude-sonnet-4-5", "max_tokens": 512, "temperature": 1,
        "system": [{"type": "text", "text": "x-anthropic-billing-header: cc_version=2.1; cch=a5145;You are Claude Code."},
                   {"type": "text", "text": "Be brief.", "cache_control": {"type": "ephemeral"}}],
        "tools": [{"name": "read_file", "description": "Read a file",
                   "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}}}],
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "<system-reminder>ctx</system-reminder>"},
                                         {"type": "text", "text": "What is in a.txt?"}]},
            {"role": "assistant", "content": [{"type": "thinking", "thinking": "look", "signature": ""},
                                              {"type": "tool_use", "id": "toolu_1", "name": "read_file", "input": {"path": "a.txt"}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_1",
                                          "content": [{"type": "text", "text": "hello"}, {"type": "image", "source": {}}]}]},
        ],
    }

    def test_anthropic_request_maps_to_chat_messages(self) -> None:
        chat = self.mk.anthropic_to_chat(self.claude_request)
        system, user, assistant, tool = chat["messages"]
        self.assertTrue(system["content"].startswith("x-anthropic-billing-header: cc_version=2.1; cch=fffff;You are"))
        self.assertTrue(system["content"].endswith("Be brief."))
        self.assertEqual(user, {"role": "user", "content": "<system-reminder>ctx</system-reminder>\nWhat is in a.txt?"})
        self.assertEqual(assistant["reasoning_content"], "look")
        self.assertEqual(assistant["tool_calls"][0]["function"], {"name": "read_file", "arguments": {"path": "a.txt"}})
        self.assertEqual(tool, {"role": "tool", "tool_call_id": "toolu_1", "content": "hello\n" + self.mk.IMAGE_PLACEHOLDER})
        self.assertEqual(chat["tools"][0]["function"]["parameters"]["properties"], {"path": {"type": "string"}})
        self.assertEqual(chat["max_tokens"], 512)

    def test_anthropic_thinking_disabled_turns_thinking_off(self) -> None:
        chat = self.mk.anthropic_to_chat({"messages": [], "thinking": {"type": "disabled"}})
        self.assertEqual(chat["chat_template_kwargs"], {"enable_thinking": False})

    def test_anthropic_message(self) -> None:
        status, data = self._post("/v1/messages", self.claude_request)
        self.assertEqual(status, 200)
        msg = json.loads(data)
        self.assertEqual([b["type"] for b in msg["content"]], ["thinking", "text", "tool_use"])
        self.assertEqual(msg["content"][0]["thinking"], "Check the file.")
        self.assertEqual(msg["content"][1]["text"], "Reading it now.")
        self.assertEqual(msg["content"][2]["input"], {"path": "a.txt"})
        self.assertTrue(msg["content"][2]["id"].startswith("toolu_"))
        self.assertEqual(msg["stop_reason"], "tool_use")
        self.assertEqual(msg["usage"]["cache_read_input_tokens"], 2)
        self.assertEqual(msg["usage"]["input_tokens"] + 2, len(self.eng.prompts[-1]))

    def test_anthropic_stream(self) -> None:
        status, data = self._post("/v1/messages", {**self.claude_request, "stream": True})
        self.assertEqual(status, 200)
        events = _events(data)
        self.assertTrue(all(name == obj["type"] for name, obj in events))
        names = [name for name, _ in events]
        self.assertEqual(names[0], "message_start")
        self.assertEqual(names[-2:], ["message_delta", "message_stop"])
        starts = [obj["content_block"]["type"] for name, obj in events if name == "content_block_start"]
        self.assertEqual(starts, ["thinking", "text", "tool_use"])
        self.assertEqual(names.count("content_block_stop"), 3)
        deltas = [obj["delta"] for name, obj in events if name == "content_block_delta"]
        self.assertEqual("".join(d.get("thinking", "") for d in deltas), "Check the file.\n")
        self.assertEqual("".join(d.get("text", "") for d in deltas), "Reading it now.")
        self.assertIn({"type": "signature_delta", "signature": ""}, deltas)
        self.assertEqual(json.loads("".join(d.get("partial_json", "") for d in deltas)), {"path": "a.txt"})
        self.assertEqual(events[-2][1]["delta"]["stop_reason"], "tool_use")

    def test_count_tokens(self) -> None:
        status, data = self._post("/v1/messages/count_tokens", self.claude_request)
        self.assertEqual(status, 200)
        self.assertGreater(json.loads(data)["input_tokens"], 0)

    def test_anthropic_overflow_uses_anthropic_wording(self) -> None:
        big = {"model": "m", "max_tokens": 10, "messages": [{"role": "user", "content": "x" * 5000}]}
        status, data = self._post("/v1/messages", big)
        self.assertEqual(status, 400)
        err = json.loads(data)
        self.assertEqual(err["type"], "error")
        self.assertRegex(err["error"]["message"], r"^prompt is too long: \d+ tokens > 4096 maximum$")

    # OpenAI Responses (Codex CLI)

    codex_request = {
        "model": "qwen3.8-27b", "instructions": "You are Codex.", "store": False,
        "reasoning": {"effort": "medium", "summary": "auto"},
        "tools": [{"type": "function", "name": "read_file", "description": "Read a file", "strict": False,
                   "parameters": {"type": "object", "properties": {"path": {"type": "string"}}}},
                  {"type": "custom", "name": "apply_patch", "description": "Edit files",
                   "format": {"type": "grammar", "syntax": "lark", "definition": "start: ..."}},
                  {"type": "web_search"}],
        "input": [
            {"type": "message", "role": "developer", "content": [{"type": "input_text", "text": "Sandbox: on"}]},
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "What is in a.txt?"}]},
            {"type": "reasoning", "summary": [], "content": [{"type": "reasoning_text", "text": "look"}]},
            {"type": "function_call", "call_id": "call_1", "name": "read_file", "arguments": "{\"path\": \"a.txt\"}"},
            {"type": "function_call_output", "call_id": "call_1", "output": "hello"},
            {"type": "custom_tool_call", "call_id": "call_2", "name": "apply_patch", "input": "*** Begin Patch"},
            {"type": "custom_tool_call_output", "call_id": "call_2", "output": "Done"},
        ],
    }

    def test_responses_request_maps_to_chat_messages(self) -> None:
        chat = self.mk.responses_to_chat(self.codex_request)
        roles = [m["role"] for m in chat["messages"]]
        self.assertEqual(roles, ["system", "system", "user", "assistant", "tool", "assistant", "tool"])
        call = chat["messages"][3]
        self.assertEqual(call["reasoning_content"], "look")
        self.assertEqual(call["tool_calls"][0]["function"], {"name": "read_file", "arguments": "{\"path\": \"a.txt\"}"})
        self.assertEqual(chat["messages"][5]["tool_calls"][0]["function"]["arguments"], {"input": "*** Begin Patch"})
        self.assertEqual([t["function"]["name"] for t in chat["tools"]], ["read_file", "apply_patch"])
        self.assertEqual(chat["chat_template_kwargs"], {"reasoning_effort": "medium"})

    def test_responses_reject_previous_response_id(self) -> None:
        status, data = self._post("/v1/responses", {"input": "hi", "previous_response_id": "resp_1"})
        self.assertEqual(status, 400)
        self.assertIn("previous_response_id", json.loads(data)["error"]["message"])

    def test_response(self) -> None:
        status, data = self._post("/v1/responses", self.codex_request)
        self.assertEqual(status, 200)
        resp = json.loads(data)
        self.assertEqual(resp["status"], "completed")
        self.assertEqual([o["type"] for o in resp["output"]], ["reasoning", "message", "function_call"])
        self.assertEqual(resp["output"][1]["content"][0]["text"], "Reading it now.")
        self.assertEqual(json.loads(resp["output"][2]["arguments"]), {"path": "a.txt"})
        self.assertEqual(resp["usage"]["input_tokens_details"]["cached_tokens"], 2)

    def test_responses_stream(self) -> None:
        status, data = self._post("/v1/responses", {**self.codex_request, "stream": True})
        self.assertEqual(status, 200)
        events = _events(data)
        self.assertTrue(all(name == obj["type"] for name, obj in events))
        self.assertEqual([obj["sequence_number"] for _, obj in events], list(range(len(events))))
        names = [name for name, _ in events]
        self.assertEqual(names[:2], ["response.created", "response.in_progress"])
        self.assertEqual(names[-1], "response.completed")
        reasoning = "".join(o["delta"] for n, o in events if n == "response.reasoning_text.delta")
        text = "".join(o["delta"] for n, o in events if n == "response.output_text.delta")
        self.assertEqual((reasoning, text), ("Check the file.\n", "Reading it now."))
        done = [o["item"]["type"] for n, o in events if n == "response.output_item.done"]
        self.assertEqual(done, ["reasoning", "message", "function_call"])
        final = events[-1][1]["response"]
        self.assertEqual([o["type"] for o in final["output"]], done)
        self.assertEqual(final["usage"]["output_tokens"], len(REPLY) + 1)

    def test_custom_tool_calls_come_back_as_custom_tool_calls(self) -> None:
        self.eng.reply = PATCH_REPLY
        status, data = self._post("/v1/responses", {**self.codex_request, "stream": True})
        self.assertEqual(status, 200)
        item = [o["item"] for n, o in _events(data) if n == "response.output_item.done"][-1]
        self.assertEqual(item["type"], "custom_tool_call")
        self.assertEqual(item["name"], "apply_patch")
        self.assertEqual(item["input"], "*** Begin Patch\n*** End Patch")


if __name__ == "__main__":
    unittest.main()
