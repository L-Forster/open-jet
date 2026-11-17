"""Tests of server/mk_server.py's smaller behaviours without a GPU: an exclusive listening port (Windows), a console
line per token-count request, no decode rate for a one-token reply, Codex's reasoning efforts max / ultra, and Qwen3's
soft switches (/no_think, /think, an empty <think></think>). Fake engines and the test server come from
test_mk_server.py."""
import http.client
import http.server
import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_mk_server as base  # noqa: E402  (its helpers; its test cases are not collected from here)

UNSLOTH = (base.TEMPLATES / "qwen38_unsloth.jinja").read_bytes().decode()


class PortTests(base._HttpCase):
    def test_a_second_server_on_a_port_in_use_fails_with_a_message(self) -> None:
        mk = self.mk
        first = mk.HTTPServer(("127.0.0.1", 0), http.server.BaseHTTPRequestHandler)
        self.addCleanup(first.server_close)
        port = first.server_address[1]
        with self.assertRaises(OSError):  # (Windows: the exclusive bind; Linux: a listening port)
            mk.HTTPServer(("127.0.0.1", port), http.server.BaseHTTPRequestHandler).server_close()
        with self.assertRaises(SystemExit) as cm:
            mk.listen("127.0.0.1", port, http.server.BaseHTTPRequestHandler)
        self.assertIn(f"cannot listen on 127.0.0.1:{port}", str(cm.exception.code))
        self.assertIn("another server is using the port", str(cm.exception.code))
        self.assertEqual(first.allow_reuse_address, os.name != "nt")


class ConsoleTests(base._HttpCase):
    quiet = base.HostCacheTests.quiet

    def post(self, http_, path, body):
        c = http.client.HTTPConnection("127.0.0.1", http_.port, timeout=10)
        c.request("POST", path, json.dumps(body), {"Content-Type": "application/json"})
        r = c.getresponse()
        data = r.read()
        c.close()
        self.assertEqual(r.status, 200, data)
        return json.loads(data)

    def test_each_token_count_request_logs_a_line(self) -> None:
        srv, http_ = self.serve(base._ScriptEngine(["a"]))
        msgs = [{"role": "user", "content": "count me"}]
        out = {}

        def run():
            out["chat"] = self.post(http_, "/v1/chat/completions/input_tokens", {"messages": msgs, "tools": base.TOOLS})
            out["anth"] = self.post(http_, "/v1/messages/count_tokens", {"model": "m", "messages": msgs})
            out["resp"] = self.post(http_, "/v1/responses/input_tokens", {"model": "m", "input": "count me"})
        log = self.quiet(run)
        n = {k: f"{v['input_tokens']:,}" for k, v in out.items()}
        self.assertIn(f"/v1/chat/completions/input_tokens: {n['chat']} tok (1 message, 1 tool) in ", log)
        self.assertIn(f"/v1/messages/count_tokens: {n['anth']} tok (1 message) in ", log)
        self.assertIn(f"/v1/responses/input_tokens: {n['resp']} tok (a text input) in ", log)

    def test_a_one_token_reply_has_no_decode_rate(self) -> None:
        for pieces, rate in (([], False), (["a", "b", "c"], True)):
            srv, http_ = self.serve(base._ScriptEngine(pieces))
            body = {"messages": [{"role": "user", "content": "hi"}]}
            log = self.quiet(lambda: self.assertEqual(http_.post(body)[0], 200))
            reply = next(ln for ln in log.splitlines() if " reply: " in ln)
            self.assertEqual("tok/s" in reply, rate, reply)


class _AbortEngine(base._ScriptEngine):
    """a decode cycle that meets Windows' WinError 10053 (the client aborted the connection while the reply was sent)"""

    def step(self):
        raise ConnectionAbortedError(10053, "An established connection was aborted by the software in your host machine")


class DisconnectTests(base._HttpCase):
    quiet = ConsoleTests.quiet
    metrics = base.HostCacheTests.metrics

    def test_a_connection_aborted_by_the_client_is_a_disconnect_not_an_error(self) -> None:
        srv, http_ = self.serve(_AbortEngine(["a", "b", "c"]))

        def run():
            with self.assertRaises(http.client.HTTPException):  # (no answer: the connection is closed)
                http_.post({"messages": [{"role": "user", "content": "hi"}]})
        log = self.quiet(run)
        self.assertIn("the client disconnected (ConnectionAbortedError)", log)
        self.assertNotIn(" error: ", log)
        m = self.metrics(http_)
        self.assertEqual((m["megakernel:requests_cancelled_total"], m["megakernel:requests_failed_total"]), ("1", "0"))


class ThinkingTests(base._HttpCase):
    def server(self, **kwargs):
        srv = base._chat_server(self.mk, base._ScriptEngine(["a"]))
        srv.template = self.mk.make_template(UNSLOTH)
        srv.args.chat_template_kwargs = kwargs
        return srv

    def test_codex_efforts_max_and_ultra_are_the_templates_highest(self) -> None:
        srv = self.server()
        msgs = [{"role": "user", "content": "hi"}]
        want = srv.render({"messages": msgs, "reasoning_effort": "xhigh"})[0]
        for eff in ("max", "ultra"):
            self.assertEqual(srv.render({"messages": msgs, "reasoning_effort": eff})[0], want, eff)
        with self.assertRaises(self.mk.BadRequest):  # (other unknown values: the template's own error)
            srv.render({"messages": msgs, "reasoning_effort": "extreme"})

    def test_soft_switches_set_the_requests_thinking(self) -> None:
        mk = self.mk
        crush = [{"role": "system", "content": "You will generate a short title.\n\n /no_think"},
                 {"role": "user", "content": "Generate a concise title for: fizzbuzz\n <think>\n\n</think>"}]
        parts = [{"type": "text", "text": "a"}, {"type": "text", "text": "b /no_think"}]
        cases = [
            ("crush title", crush, {}, {}, False),
            ("system /no_think", [crush[0], {"role": "user", "content": "hi"}], {}, {}, False),
            ("user /no_think wins over the system's /think",
             [{"role": "system", "content": "s /think"}, {"role": "user", "content": "hi /no_think"}], {}, {}, False),
            ("/think over a server default off", [{"role": "user", "content": "hi /think"}], {},
             {"enable_thinking": False}, True),
            ("parts", [{"role": "user", "content": parts}], {}, {}, False),
            ("a mention is no switch", [{"role": "user", "content": "what does /no_think do?"}], {}, {}, True),
            ("chat_template_kwargs win", crush, {"chat_template_kwargs": {"enable_thinking": True}}, {}, True),
            ("reasoning_effort none wins", [{"role": "user", "content": "hi /think"}], {"reasoning_effort": "none"}, {},
             False),
        ]
        for name, msgs, extra, default, want in cases:
            srv = self.server(**default)
            text, thinking, _ = srv.render({"messages": msgs, **extra})
            self.assertEqual(thinking, want, name)
            last = msgs[-1]["content"]
            self.assertIn(last if isinstance(last, str) else last[-1]["text"], text, name)  # (the text as sent)
        prefill = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "<think></think>"}]
        self.assertIs(mk.soft_switch(prefill), False)
        self.assertIsNone(mk.soft_switch([{"role": "user", "content": "hi"}]))


if __name__ == "__main__":
    unittest.main()
