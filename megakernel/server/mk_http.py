"""The HTTP layer: routes, the request handlers, the listening socket."""
import errno
import json
import os
import re
import select
import socket
import sys
import threading
import time
import traceback
import uuid
from http.server import ThreadingHTTPServer
from urllib.parse import parse_qs

import mk_limits
from mk_anthropic import (
    ANTHROPIC_ERRORS, anthropic_stop_reason, anthropic_to_chat, anthropic_tool_use, anthropic_usage,
    takes_mid_system,
)
from mk_chat import check_params
from mk_errors import BadRequest, ClientGone, ContextOverflow, DISCONNECTS
from mk_gate import BodyBudget, GatedHandler
from mk_limits import LINGER_BYTES, LINGER_IDLE_S, LINGER_S
from mk_log import log, plural
from mk_responses import (
    RESP_KEEPALIVE_S, response_object, responses_call, responses_message, responses_reasoning, responses_to_chat,
    responses_usage,
)

POST_ROUTES = {"/v1/chat/completions": "chat", "/chat/completions": "chat", "/v1/messages": "messages",
               "/v1/messages/count_tokens": "count_tokens", "/tokenize": "tokenize", "/detokenize": "detokenize",
               "/apply-template": "apply_template", "/v1/chat/completions/input_tokens": "input_tokens",
               "/chat/completions/input_tokens": "input_tokens", "/v1/responses": "responses", "/responses": "responses",
               "/v1/responses/input_tokens": "responses_input_tokens", "/responses/input_tokens": "responses_input_tokens"}
# llama-server's other POST routes (and OpenAI's compaction route): 501 (not supported here) instead of 404
UNSUPPORTED = {"/completion", "/completions", "/v1/completions", "/v1/responses/compact", "/responses/compact",
               "/infill", "/embedding", "/embeddings", "/v1/embeddings", "/rerank", "/reranking",
               "/v1/rerank", "/v1/reranking", "/v1/audio/transcriptions", "/audio/transcriptions", "/lora-adapters",
               "/v1/chat/completions/control", "/models", "/v1/streams/lookup", "/tools", "/cors-proxy"}
# a surrogate in a request body: its JSON escape, or its UTF-8 form (json.loads takes both)
SURROGATE_RAW = re.compile(rb"\\u[dD][89a-fA-F]|\xed[\xa0-\xbf]")
SURROGATE = re.compile("[\ud800-\udfff]")


def with_trace(e):
    """an exception for the console line of a 500 (in its except block): its repr and where it was raised"""
    return f"{e!r}\n{traceback.format_exc().rstrip()}"


def fix_surrogates(raw, body):
    """(body, n): the parsed body with each lone UTF-16 surrogate in its keys and strings replaced by U+FFFD (a pair
    is one character by then, untouched), as a JS client's own UTF-8 encoder would (they cut strings inside a pair):
    text without a UTF-8 form fails the tokenizer and the stream events. raw (the body's bytes) is looked at first"""
    if not SURROGATE_RAW.search(raw):
        return body, 0
    try:
        text, n = SURROGATE.subn("\ufffd", json.dumps(body, ensure_ascii=False))
        return (json.loads(text) if n else body), n
    except RecursionError:  # (nested about as deep as json.loads takes: left as it is)
        return body, 0


def make_handler(srv):
    class H(GatedHandler):
        def client_gone(self):
            """True once the client has closed its connection (end of file or reset). The request was read in full, so
            anything readable is the next request of a keep-alive client (still connected) or the close."""
            try:
                if not select.select([self.connection], [], [], 0)[0]:
                    return False
                return self.connection.recv(1, socket.MSG_PEEK) == b""
            except (OSError, ValueError):
                return True

        def _json(self, code, obj, close=False, headers=()):
            """close: then close the connection (a request body that was not read must not be taken for the next request)"""
            data = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            for k, v in headers:
                self.send_header(k, v)
            if close:
                self.send_header("Connection", "close")
                self.close_connection = True
            self.end_headers()
            self._timed_write(data)

        def _body_length(self):
            """(the request body's length, or "chunked", None), or (None, (status, message)) when it is not acceptable:
            no Content-Length (411), a malformed one (400), over MK_MAX_BODY_MB (413, refused before reading); a
            transfer coding other than chunked (501), chunked with a Content-Length too (400: two framings)"""
            te = [c.strip().lower() for v in self.headers.get_all("Transfer-Encoding") or [] for c in v.split(",")]
            if any(te):
                if [c for c in te if c] != ["chunked"]:
                    return None, (501, f"Transfer-Encoding {', '.join(c for c in te if c)!r}: only chunked is supported")
                if self.headers.get_all("Content-Length"):
                    return None, (400, "Content-Length and Transfer-Encoding: chunked together")
                return "chunked", None
            vals = [v.strip() for v in self.headers.get_all("Content-Length") or []]
            if not vals:
                return None, (411, "a Content-Length header is required")
            if len(set(vals)) > 1 or not re.fullmatch(r"[0-9]+", vals[0]):
                return None, (400, f"bad Content-Length {vals[0][:40]!r}")
            n = int(vals[0]) if len(vals[0]) <= 18 else 1 << 62
            if n > mk_limits.MAX_BODY_MB << 20:
                return None, (413, f"request body of {n} bytes: over the limit of {mk_limits.MAX_BODY_MB} MB (MK_MAX_BODY_MB)")
            return n, None

        CHUNK_LINE, TRAILER_LINES = 4096, 64  # (a chunk-size line with its extensions; trailer field lines)
        READ_PIECE = 1 << 20  # (a body is read and charged to the server's budget in pieces of this size)

        def _refuse(self, code, msg, kind="invalid_request_error", headers=()):
            """answers a request whose body is not read (further) and closes the connection"""
            log(f"{self.command} {self.path.split('?')[0]} -> {code}: {msg}")
            self._json(code, {"error": {"code": code, "message": msg, "type": kind}}, close=True, headers=headers)

        def _read_into(self, buf, n):
            """n more bytes of the request's body into buf, charged to the server's budget as they arrive (do_POST
            releases them). False: the client left mid-body, or the budget is full (answered with a 503)"""
            while n > 0:
                piece = min(n, self.READ_PIECE)
                data = self.rfile.read(piece)
                if len(data) < piece:  # (the client closed the connection mid-body)
                    self.close_connection = True
                    return False
                if not self.server.bodies.charge(piece):
                    self._refuse(503, f"the server holds {self.server.bodies.cap >> 20} MB of request bodies at once "
                                      "(MK_MAX_INFLIGHT_MB): try again later", "unavailable_error",
                                 [("Retry-After", "5")])
                    return False
                self.charged += piece
                buf += data
                n -= piece
            return True

        def _read_chunked(self):
            """a Transfer-Encoding: chunked body (RFC 9112 7.1), decoded into a buffer of at most MK_MAX_BODY_MB (413 as
            soon as a chunk would pass it); a malformed size line (not hex, over 16 digits, no CRLF), data without its
            CRLF or a malformed trailer section is a 400; trailers are read and dropped. Both close the connection (the
            lingering close drains what the client still sends). None: answered, or the client left mid-body"""
            cap, buf, refuse = mk_limits.MAX_BODY_MB << 20, bytearray(), self._refuse
            while True:
                line = self.rfile.readline(self.CHUNK_LINE + 1)
                if not line:  # (the client closed the connection mid-body)
                    self.close_connection = True
                    return None
                if not line.endswith(b"\r\n"):
                    return refuse(400, "chunked body: a chunk-size line without its CRLF (or over 4096 bytes)")
                size = line[:-2].split(b";", 1)[0].rstrip(b" \t")
                if not re.fullmatch(rb"[0-9A-Fa-f]{1,16}", size):
                    return refuse(400, f"chunked body: bad chunk size {size[:40].decode('latin-1')!r}")
                n = int(size, 16)
                if n == 0:
                    break
                if len(buf) + n > cap:
                    return refuse(413, f"chunked request body over the limit of {mk_limits.MAX_BODY_MB} MB (MK_MAX_BODY_MB)")
                if not self._read_into(buf, n):
                    return None
                end = self.rfile.read(2)
                if end != b"\r\n":
                    if len(end) < 2:
                        self.close_connection = True
                        return None
                    return refuse(400, "chunked body: chunk data without its CRLF")
            for _ in range(self.TRAILER_LINES + 1):  # (trailer fields: dropped)
                line = self.rfile.readline(self.CHUNK_LINE + 1)
                if line == b"\r\n":
                    return buf
                if not line:
                    self.close_connection = True
                    return None
                if not line.endswith(b"\r\n"):
                    return refuse(400, "chunked body: a trailer line without its CRLF (or over 4096 bytes)")
            return refuse(400, f"chunked body: more than {self.TRAILER_LINES} trailer lines")

        def _refuse_body(self):
            """answers an unacceptable request body (_body_length) without reading it; True if it did"""
            refused = self._body_length()[1]
            if refused:
                self._refuse(*refused)
            return bool(refused)

        def handle_expect_100(self):
            # (curl sends large bodies after a 100 Continue: a refused one is answered before it is sent)
            if not self.admit() or self.command == "POST" and self._refuse_body():
                return False
            return super().handle_expect_100()

        def _text(self, code, text, ctype="text/plain; charset=utf-8", headers=()):
            data = text.encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            for k, v in headers:
                self.send_header(k, v)
            self.end_headers()
            self._timed_write(data)

        def do_GET(self):
            path, _, query = self.path.partition("?")
            if path in ("/health", "/v1/health"):
                self._json(200, {"status": "ok"})
            elif path in ("/v1/models", "/models"):
                self._json(200, srv.models())
            elif path == "/props":
                self._json(200, srv.props())
            elif path == "/slots":
                if parse_qs(query).get("fail_on_no_slot", [""])[0] and srv.busy():
                    self._json(503, {"error": {"code": 503, "message": "no slot available", "type": "unavailable_error"}})
                else:
                    self._json(200, srv.slots())
            elif path == "/metrics":
                self._text(200, srv.metrics_text(), "text/plain; version=0.0.4",
                           [("Process-Start-Time-Unix", str(int(srv.t_start)))])
            else:
                self._error(404, "not found")

        head_sent = False  # the running POST's status line and headers are out (a stream's: its body follows)

        def end_headers(self):
            super().end_headers()
            self.head_sent = True

        def do_POST(self):
            self.charged, self.head_sent = 0, False
            try:
                self._post()
            finally:  # (the body and what was parsed from it are gone with the request)
                self.server.bodies.release(self.charged)

        def _post(self):
            path = self.path.split("?")[0]
            if self._refuse_body():
                return
            n = self._body_length()[0]
            if n == "chunked":
                raw = self._read_chunked()
                if raw is None:  # (refused, or the client left)
                    return
            else:
                raw = bytearray()
                if not self._read_into(raw, n):
                    return
            self.reader.done()
            route = POST_ROUTES.get(path)
            # (errors in Anthropic's shape on /v1/messages, OpenAI's on /v1/responses: _error)
            anth = "responses" if path.startswith(("/v1/responses", "/responses")) else path.startswith("/v1/messages")
            if route is None:
                known = path in UNSUPPORTED or path.startswith(("/slots", "/props", "/models/"))
                self._error(501 if known else 404, f"{path} is not supported by the megakernel server", anth)
                return
            try:
                body = json.loads(raw or b"{}")
            except (ValueError, RecursionError):  # (not JSON or UTF-8, an integer past the digit limit, nested too deep)
                body = None
            if not isinstance(body, dict):
                self._error(400, "the request body must be a JSON object", anth)
                return
            body, n = fix_surrogates(raw, body)
            if n:
                srv.warn_once("lone-surrogate", f"{path}: {plural(n, 'lone UTF-16 surrogate')} in a request body (a "
                                                f"\\ud800-\\udfff escape without its pair) replaced with U+FFFD")
            if route in ("tokenize", "detokenize", "apply_template", "input_tokens", "count_tokens", "responses_input_tokens"):
                try:  # (no generation: the tokenizer, from this thread; images take the engine's lock)
                    t0 = time.time()
                    out = getattr(self, "_" + route)(body)
                    if route.endswith("_tokens"):  # (a client counting its prompt: one console line each)
                        msgs = body.get("messages") if route != "responses_input_tokens" else body.get("input")
                        what = [plural(len(msgs), "message" if route != "responses_input_tokens" else "item")
                                if isinstance(msgs, list) else "a text input"]
                        if body.get("tools"):
                            what.append(plural(len(body["tools"]), "tool"))
                        ms = (time.time() - t0) * 1e3
                        log(f"{path}: {out['input_tokens']:,} tok ({', '.join(what)}) in {ms:.0f} ms")
                    self._json(200, out)
                except BadRequest as e:
                    self._bad(e, anth)
                except DISCONNECTS:
                    self.close_connection = True
                except Exception as e:  # (answered, not a dropped connection)
                    log(f"{path} error:", with_trace(e))
                    try:
                        self._error(500, str(e), anth)
                    except Exception:
                        pass
                return
            rid = next(srv.rids)
            if srv.sched is None and srv.busy():
                log(f"#{rid} {route}: waiting for #{srv.rid} (the engine runs one request at a time)")
            srv.count(requests=1, deferred=1)  # (deferred: waiting for the engine)
            with srv.lock:
                srv.count(deferred=-1)
                srv.rid, srv.req = rid, (route, body.get("model"), time.time())
                try:
                    if route == "messages":
                        self._anthropic(body)
                    elif route == "responses":
                        self._responses(body)
                    elif body.get("stream"):
                        self._stream(body)
                    else:
                        self._complete(body)
                except BadRequest as e:
                    srv.count(requests_bad=1)
                    if self.head_sent:  # (a stream's status and part of its body are out: no second response in it)
                        log(f"#{rid} 400 after the stream started: {e}; closing the connection")
                        self.close_connection = True
                    else:
                        self._bad(e, anth, rid)
                except (*DISCONNECTS, ClientGone) as e:
                    srv.count(requests_cancelled=1)
                    self.close_connection = True
                    if not isinstance(e, ClientGone):  # (generate reports the disconnects it notices itself)
                        log(f"#{rid} the client disconnected ({type(e).__name__})")
                except Exception as e:  # engine failure: report it, keep serving
                    srv.count(requests_failed=1)
                    log(f"#{rid} error:", with_trace(e))
                    if self.head_sent:  # (as above: the client sees the stream end without its last chunk)
                        log(f"#{rid} the stream had started: closing the connection")
                        self.close_connection = True
                    else:
                        try:
                            self._error(500, str(e), anth)
                        except Exception:
                            pass
                finally:
                    srv.request_done()

        def _error(self, code, message, anthropic=False, rid=None, kind=None, extra=None):
            """answers with an error (OpenAI / llama-server's shape with its code, Anthropic's, or with anthropic
            "responses" the OpenAI API's: code a string or null); a 4xx is also logged (a 5xx was logged with its
            exception). kind: the type instead of the code's; extra: more fields"""
            if code < 500:
                log(f"#{rid} {code}: {message}" if rid else f"{self.command} {self.path.split('?')[0]} -> {code}: {message}")
            if anthropic == "responses":
                kind = kind or {400: "invalid_request_error", 404: "not_found_error", 501: "not_supported"}.get(code, "server_error")
                self._json(code, {"error": {"message": message, "type": kind, "param": None, "code": None, **(extra or {})}})
            elif anthropic:
                err = {"type": kind or ANTHROPIC_ERRORS.get(code, "api_error"), "message": message, **(extra or {})}
                self._json(code, {"type": "error", "error": err})
            else:
                kind = kind or {400: "invalid_request_error", 404: "not_found_error", 501: "not_supported"}.get(code, "server_error")
                self._json(code, {"error": {"code": code, "message": message, "type": kind, **(extra or {})}})

        def _bad(self, e, anthropic=False, rid=None):
            """answers a BadRequest (400); a prompt longer than the context in llama-server's / Anthropic's shape"""
            if not isinstance(e, ContextOverflow):
                return self._error(400, str(e), anthropic, rid)
            extra = {"n_prompt_tokens": e.n_prompt, "n_ctx": e.n_ctx}
            if anthropic == "responses":  # (OpenAI's error code for it)
                return self._error(400, str(e), anthropic, rid, extra={**extra, "code": "context_length_exceeded"})
            if anthropic:
                return self._error(400, e.anthropic, True, rid, extra=extra)
            self._error(400, str(e), False, rid, "exceed_context_size_error", extra)

        # ---- routes without generation (llama-server's) ----
        def _tokenize(self, body):
            content = body.get("content")
            # llama-server's flags: parse_special [true], add_special [false; for the first element only, if a string]
            special = body.get("parse_special") is not False
            add = body.get("add_special") is True
            if isinstance(content, str):
                content = [content]
            if not isinstance(content, list) or not all(isinstance(c, (str, int)) and not isinstance(c, bool) for c in content):
                raise BadRequest("content: a string, or an array of strings and token ids")

            def tok(i, c):  # (ids of the tokenizer's own, or ids given in the content: those only in the vocabulary get a piece)
                if not isinstance(c, str):
                    return [c]
                first = add and i == 0
                return srv.eng.tokenize(c) if special and not first else srv.eng.tokenize(c, special, first)
            try:
                toks = [(t, isinstance(c, str)) for i, c in enumerate(content) for t in tok(i, c)]
            except ValueError as e:  # (a library without mk_tokenize2)
                raise BadRequest(str(e))
            if not body.get("with_pieces"):
                return {"tokens": [t for t, _ in toks]}
            out = []
            for t, own in toks:
                p = srv.eng.piece(t) if own or 0 <= t < srv.n_vocab() else b""
                try:
                    out.append({"id": t, "piece": p.decode()})
                except UnicodeDecodeError:
                    out.append({"id": t, "piece": list(p)})
            return {"tokens": out}

        def _detokenize(self, body):
            toks = body.get("tokens") or []
            nv = srv.n_vocab()
            if not nv:
                raise BadRequest("/detokenize needs a newer libqwen38mk (mk_n_vocab)")
            if not isinstance(toks, list) or not all(isinstance(t, int) and not isinstance(t, bool) and 0 <= t < nv for t in toks):
                raise BadRequest(f"tokens: an array of token ids in [0, {nv})")
            return {"content": b"".join(srv.eng.piece(t) for t in toks).decode(errors="replace")}

        def _apply_template(self, body):
            check_params(body)
            return {"prompt": srv.render(body)[0]}

        def _input_tokens(self, body):
            check_params(body)
            return {"input_tokens": srv.prompt_tokens(body), "object": "response.input_tokens"}

        def _count_tokens(self, body):
            return {"input_tokens": srv.prompt_tokens(self._anthropic_chat(body))}

        # ---- Anthropic Messages ----
        def _anthropic_chat(self, body):
            """anthropic_to_chat; what it skipped or converted is a one-time console warning per kind"""
            notes = []
            if getattr(srv, "mid_system", None) is None:  # (the template's answer, once)
                srv.mid_system = takes_mid_system(srv.template)
            chat = anthropic_to_chat(body, notes, srv.mid_system)
            for key, msg in notes:
                srv.warn_once(key, msg)
            return chat

        def _anthropic(self, body):
            chat = self._anthropic_chat(body)
            model = body.get("model") if isinstance(body.get("model"), str) and body.get("model") else srv.model_id
            if chat["stream"]:
                return self._anthropic_stream(chat, model)
            parts = {"reasoning": [], "content": []}
            finish, calls, usage, _ = srv.generate(chat, lambda k, t: parts[k].append(t), self.client_gone)
            blocks = []
            reasoning = "".join(parts["reasoning"]).strip()
            if reasoning:
                blocks.append({"type": "thinking", "thinking": reasoning, "signature": ""})
            if parts["content"]:
                blocks.append({"type": "text", "text": "".join(parts["content"])})
            blocks += [anthropic_tool_use(c) for c in calls]
            self._json(200, {"id": "msg_" + uuid.uuid4().hex, "type": "message", "role": "assistant", "model": model,
                             "content": blocks, "stop_reason": anthropic_stop_reason(finish, srv.stop_word),
                             "stop_sequence": srv.stop_word, "usage": anthropic_usage(usage)})

        def _stream_head(self):
            """ends an event stream's headers: chunked, or for HTTP/1.0 (no chunked) a body that ends with the connection"""
            if self.request_version == "HTTP/1.0":
                self.send_header("Connection", "close")
                self.close_connection = True
            else:
                self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()

        WRITE_PIECE = 1 << 16  # (a reply's body goes out this much at a time: each piece has STREAM_WRITE_S)

        def _timed_write(self, data):
            """a reply's body (streamed or not: either is written while its request holds the engine). A write that makes
            no progress for mk_limits.STREAM_WRITE_S times out (TimeoutError: the client is gone, as after a reset),
            whatever MK_TIMEOUT_S is: that stays the limit of the connection's other reads and writes"""
            sock, view = self.connection, memoryview(data)
            idle = sock.gettimeout()
            sock.settimeout(mk_limits.STREAM_WRITE_S if idle is None else min(idle, mk_limits.STREAM_WRITE_S))
            try:  # (a write's timeout counts from its start: a large one goes out in pieces, each one timed)
                for i in range(0, len(view), self.WRITE_PIECE):
                    self.wfile.write(view[i:i + self.WRITE_PIECE])
            finally:
                sock.settimeout(idle)

        def _stream_send(self, data, last=False):
            """one piece of an event stream's body (_stream_head's framing); last: then the end of the body"""
            if self.request_version != "HTTP/1.0":
                data = b"%x\r\n" % len(data) + data + (b"\r\n0\r\n\r\n" if last else b"\r\n")
            self._timed_write(data)
            self.wfile.flush()

        def _anthropic_stream(self, chat, model):
            """Anthropic's SSE: message_start, ping, content blocks (thinking / text deltas as they come, tool_use blocks
            with their input as input_json_delta pieces as it is generated), message_delta, message_stop"""
            mid = "msg_" + uuid.uuid4().hex
            st = {"started": False, "index": -1, "open": None, "ws": ""}  # ws: the thinking's held trailing whitespace
            send = self._stream_send

            def event(name, obj, last=False):
                send(f"event: {name}\ndata: {json.dumps(obj, ensure_ascii=False)}\n\n".encode(), last)

            def start():
                if st["started"]:
                    return
                st["started"] = True
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self._stream_head()
                n, reused = srv.last_prompt
                event("message_start", {"type": "message_start", "message": {
                    "id": mid, "type": "message", "role": "assistant", "model": model, "content": [], "stop_reason": None,
                    "stop_sequence": None, "usage": {"input_tokens": n - reused, "cache_creation_input_tokens": 0,
                                                     "cache_read_input_tokens": reused, "output_tokens": 0}}})
                event("ping", {"type": "ping"})

            def close():
                if st["open"]:
                    event("content_block_stop", {"type": "content_block_stop", "index": st["index"]})
                    st["open"] = None

            def block(cb):
                close()
                st["index"] += 1
                st["open"] = cb["type"]
                event("content_block_start", {"type": "content_block_start", "index": st["index"], "content_block": cb})

            def delta(d):
                event("content_block_delta", {"type": "content_block_delta", "index": st["index"], "delta": d})

            def emit(kind, text):
                try:
                    start()
                    if kind == "reasoning":  # (trimmed at both ends, as the non-streamed reply's thinking block)
                        t = st["ws"] + text
                        if st["open"] != "thinking":
                            t = t.lstrip()
                            if not t:
                                return True
                            block({"type": "thinking", "thinking": "", "signature": ""})
                        body = t.rstrip()
                        st["ws"] = t[len(body):]
                        if body:
                            delta({"type": "thinking_delta", "thinking": body})
                    elif kind == "tool_call":  # (ToolCallParser's events: a call's start, then its input's JSON text)
                        if "id" in text:
                            block({"type": "tool_use", "id": "toolu_" + text["id"].removeprefix("call_"),
                                   "name": text["function"]["name"], "input": {}})
                        if text["function"].get("arguments"):
                            delta({"type": "input_json_delta", "partial_json": text["function"]["arguments"]})
                    else:
                        if st["open"] != "text":
                            block({"type": "text", "text": ""})
                        delta({"type": "text_delta", "text": text})
                except DISCONNECTS:
                    return False
                return True

            check_params(chat)  # (parameter / render errors: a 400 before the stream starts)
            srv.render(chat)
            try:
                finish, calls, usage, _ = srv.generate(chat, emit, self.client_gone)
            except Exception as e:
                if not st["started"] or isinstance(e, (*DISCONNECTS, ClientGone)):
                    raise
                bad = isinstance(e, BadRequest)
                log(f"#{srv.rid} 400: {e}" if bad else f"#{srv.rid} error: {with_trace(e)}")
                event("error", {"type": "error", "error": {"type": "invalid_request_error" if bad else "api_error",
                                                         "message": str(e)}}, last=True)
                return
            start()
            close()
            event("message_delta", {"type": "message_delta", "delta": {
                "stop_reason": anthropic_stop_reason(finish, srv.stop_word), "stop_sequence": srv.stop_word},
                "usage": anthropic_usage(usage)})
            event("message_stop", {"type": "message_stop"}, last=True)

        # ---- OpenAI Responses ----
        def _responses_chat(self, body):
            """responses_to_chat; what it dropped or converted is a one-time console warning per kind"""
            notes = []
            if getattr(srv, "mid_system", None) is None:  # (the template's answer, once)
                srv.mid_system = takes_mid_system(srv.template)
            chat, info = responses_to_chat(body, notes, srv.mid_system)
            for key, msg in notes:
                srv.warn_once(key, msg)
            return chat, info

        def _responses_input_tokens(self, body):
            return {"object": "response.input_tokens", "input_tokens": srv.prompt_tokens(self._responses_chat(body)[0])}

        def _responses(self, body):
            chat, info = self._responses_chat(body)
            rid, created = "resp_" + uuid.uuid4().hex, int(time.time())
            sealed = "reasoning.encrypted_content" in (body.get("include") or [])
            if chat["stream"]:
                return self._responses_stream(body, chat, info, rid, created, sealed)
            parts = {"reasoning": [], "content": []}
            finish, calls, usage, _ = srv.generate(chat, lambda k, t: parts[k].append(t), self.client_gone)
            output = []
            reasoning = "".join(parts["reasoning"]).strip()
            if reasoning:
                output.append(responses_reasoning(reasoning, "rs_" + uuid.uuid4().hex, sealed))
            if parts["content"]:
                output.append(responses_message("".join(parts["content"]), "msg_" + uuid.uuid4().hex))
            output += [responses_call(c["id"], c["function"]["name"], c["function"]["arguments"], info) for c in calls]
            done = finish != "length"
            self._json(200, response_object(
                rid, created, body, srv.model_id, "completed" if done else "incomplete", output,
                responses_usage(usage, srv.reasoning_tokens), incomplete=None if done else {"reason": "max_output_tokens"}))

        def _responses_stream(self, body, chat, info, rid, created, sealed):
            """the Responses API's SSE: response.created and response.in_progress when the prefill starts (a request
            refused before that is an HTTP error), response.in_progress every RESP_KEEPALIVE_S during a long prefill (a
            client times out a silent stream: Codex after 300 s), then the output items as they are generated, each
            opened with output_item.added and closed with its done events: reasoning (reasoning_text deltas, trimmed at
            both ends as the non-streamed item), message (output_text deltas), function_call (its arguments' JSON
            text as it is generated) or custom_tool_call (its input whole at the end); then response.completed
            (response.incomplete after max_output_tokens) with the usage. An error once it has started is
            response.failed (code server_error, invalid_prompt for a bad request). Every event has its sequence_number."""
            st = {"started": False, "seq": 0, "items": [], "open": None, "ws": "", "t": 0.0}
            send = self._stream_send

            def event(obj, last=False):
                obj["sequence_number"] = st["seq"]
                st["seq"] += 1
                st["t"] = time.monotonic()
                send(f"event: {obj['type']}\ndata: {json.dumps(obj, ensure_ascii=False)}\n\n".encode(), last)

            def response(status, **kw):
                return response_object(rid, created, body, srv.model_id, status, list(st["items"]), **kw)

            def start():
                if st["started"]:
                    return
                st["started"] = True
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self._stream_head()
                event({"type": "response.created", "response": response("in_progress")})
                event({"type": "response.in_progress", "response": response("in_progress")})

            def progress(cached, processed, total):  # (Server.prompt_progress: the prefill's start, every 0.5 s, its end)
                try:
                    if not st["started"]:
                        start()
                    elif time.monotonic() - st["t"] >= RESP_KEEPALIVE_S:
                        event({"type": "response.in_progress", "response": response("in_progress")})
                except DISCONNECTS:  # (generate notices the disconnect)
                    pass

            def where():
                o = st["open"]
                return {"item_id": o["item"]["id"], "output_index": o["index"]}

            def open_item(item):
                close()
                st["open"] = {"index": len(st["items"]), "item": item, "text": ""}
                st["items"].append(item)
                event({"type": "response.output_item.added", "output_index": st["open"]["index"], "item": item})

            def close():
                o = st["open"]
                if o is None:
                    return
                item, text, at = o["item"], o["text"], where()
                if item["type"] == "reasoning":
                    event({"type": "response.reasoning_text.done", **at, "content_index": 0, "text": text})
                    event({"type": "response.content_part.done", **at, "content_index": 0,
                           "part": {"type": "reasoning_text", "text": text}})
                    done = responses_reasoning(text, item["id"], sealed)
                elif item["type"] == "message":
                    done = responses_message(text, item["id"])
                    event({"type": "response.output_text.done", **at, "content_index": 0, "text": text, "logprobs": []})
                    event({"type": "response.content_part.done", **at, "content_index": 0, "part": done["content"][0]})
                else:  # (a call's arguments are valid JSON only when it is complete: ToolCallParser)
                    try:
                        json.loads(text)
                        status = "completed"
                    except ValueError:
                        status = "incomplete"
                    done = responses_call(item["call_id"], item["name"], text, info, status)
                    if done["type"] == "custom_tool_call":
                        if done["input"]:
                            event({"type": "response.custom_tool_call_input.delta", **at, "delta": done["input"]})
                        event({"type": "response.custom_tool_call_input.done", **at, "input": done["input"]})
                    else:
                        event({"type": "response.function_call_arguments.done", **at, "name": item["name"],
                               "arguments": text})
                st["items"][o["index"]] = done
                st["open"] = None
                event({"type": "response.output_item.done", "output_index": o["index"], "item": done})

            def emit(kind, x):
                try:
                    start()
                    o = st["open"]
                    if kind == "reasoning":
                        t = st["ws"] + x
                        if not o or o["item"]["type"] != "reasoning":
                            t = t.lstrip()
                            if not t:
                                return True
                            open_item({"type": "reasoning", "id": "rs_" + uuid.uuid4().hex, "summary": [], "content": [],
                                       "encrypted_content": None, "status": "in_progress"})
                            event({"type": "response.content_part.added", **where(), "content_index": 0,
                                   "part": {"type": "reasoning_text", "text": ""}})
                        piece = t.rstrip()
                        st["ws"] = t[len(piece):]
                        if piece:
                            st["open"]["text"] += piece
                            event({"type": "response.reasoning_text.delta", **where(), "content_index": 0, "delta": piece})
                    elif kind == "tool_call":  # (ToolCallParser's events: a call's start, then its arguments' JSON text)
                        if "id" in x:
                            item = responses_call(x["id"], x["function"]["name"], "", info, "in_progress")
                            if item["type"] == "custom_tool_call":
                                item["input"] = ""
                            open_item(item)
                        frag = x["function"].get("arguments")
                        if frag:
                            st["open"]["text"] += frag
                            if st["open"]["item"]["type"] == "function_call":  # (a custom tool's input: at its end)
                                event({"type": "response.function_call_arguments.delta", **where(), "delta": frag})
                    else:
                        if not o or o["item"]["type"] != "message":
                            open_item(responses_message("", "msg_" + uuid.uuid4().hex, "in_progress") | {"content": []})
                            event({"type": "response.content_part.added", **where(), "content_index": 0,
                                   "part": {"type": "output_text", "text": "", "annotations": [], "logprobs": []}})
                        st["open"]["text"] += x
                        event({"type": "response.output_text.delta", **where(), "content_index": 0, "delta": x,
                               "logprobs": []})
                except DISCONNECTS:
                    return False
                return True

            check_params(chat)  # (parameter / render errors: a 400 before the stream starts)
            srv.render(chat)
            srv.on_progress = progress
            try:
                finish, calls, usage, _ = srv.generate(chat, emit, self.client_gone)
                start()
                close()
            except Exception as e:
                if not st["started"] or isinstance(e, (*DISCONNECTS, ClientGone)):
                    raise  # nothing sent yet: do_POST answers with a status (400 / 500)
                bad = isinstance(e, BadRequest)
                log(f"#{srv.rid} 400: {e}" if bad else f"#{srv.rid} error: {with_trace(e)}")
                event({"type": "response.failed", "response": response("failed", error={
                    "code": "invalid_prompt" if bad else "server_error", "message": str(e)})}, last=True)
                return
            finally:
                srv.on_progress = None
            usage = responses_usage(usage, srv.reasoning_tokens)
            if finish == "length":
                event({"type": "response.incomplete", "response": response(
                    "incomplete", usage=usage, incomplete={"reason": "max_output_tokens"})}, last=True)
            else:
                event({"type": "response.completed", "response": response("completed", usage=usage)}, last=True)

        def _complete(self, body):
            parts = {"reasoning": [], "content": []}
            finish, calls, usage, timings = srv.generate(body, lambda k, t: parts[k].append(t), self.client_gone)
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
            send = self._stream_send

            def chunk(delta, finish=None, usage=None, timings=None, progress=None):
                obj = {"id": cid, "object": "chat.completion.chunk", "created": created, "model": srv.model_id,
                       "choices": [] if delta is None else [{"index": 0, "delta": delta, "finish_reason": finish}]}
                if usage is not None:
                    obj["usage"] = usage
                if timings is not None:
                    obj["timings"] = timings
                if progress is not None:
                    obj["prompt_progress"] = progress
                send(b"data: " + json.dumps(obj, ensure_ascii=False).encode() + b"\n\n")

            def head():
                if not started[0]:
                    started[0] = True
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Cache-Control", "no-cache")
                    self._stream_head()
                    chunk({"role": "assistant", "content": None})

            per_token = bool(body.get("timings_per_token"))  # (llama-server's: the timings so far in every chunk)

            def emit(kind, text):
                try:
                    head()
                    chunk({"tool_calls": [text]} if kind == "tool_call" else
                          {"reasoning_content": text} if kind == "reasoning" else {"content": text},
                          timings=srv.timings_now() if per_token else None)
                except DISCONNECTS:
                    return False
                return True

            t_p = []

            def progress(cached, processed, total):  # (return_progress: llama-server's prompt_progress chunks)
                t_p[:] = t_p or [time.monotonic()]
                try:
                    head()
                    chunk({"role": "assistant", "content": None}, progress={
                        "total": total, "cache": cached, "processed": processed,
                        "time_ms": int((time.monotonic() - t_p[0]) * 1e3)})
                except DISCONNECTS:  # (generate notices the disconnect)
                    pass

            # parameter / render errors are reported as a 400 before the stream starts
            check_params(body)
            srv.render(body)
            srv.on_progress = progress if body.get("return_progress") else None
            try:
                finish, calls, usage, timings = srv.generate(body, emit, self.client_gone)
            except Exception as e:
                if not started[0] or isinstance(e, (*DISCONNECTS, ClientGone)):
                    raise  # nothing sent yet: do_POST answers with a status (400 / 500)
                # the status (200) and part of the reply are out: the error is the stream's last event, OpenAI style
                bad = isinstance(e, BadRequest)
                log(f"#{srv.rid} 400: {e}" if bad else f"#{srv.rid} error: {with_trace(e)}")
                err = {"message": str(e), "type": "invalid_request_error" if bad else "server_error", "code": 400 if bad else 500}
                send(b"data: " + json.dumps({"error": err}, ensure_ascii=False).encode() + b"\n\n")
                send(b"data: [DONE]\n\n", last=True)
                return
            finally:
                srv.on_progress = None
            head()  # (the tool calls went out as they were generated)
            chunk({}, finish=finish, timings=timings)
            if (body.get("stream_options") or {}).get("include_usage"):
                chunk(None, usage=usage)
            send(b"data: [DONE]\n\n", last=True)

    return H


def make_loading_handler():
    """the handler while the model loads (the port opens first, as llama-server's does): every request gets llama-server's
    503 "Loading model" and the connection closes (a request body is not read: the lingering close drains it); the gate
    and CORS preflights as in the real handler"""
    class Loading(GatedHandler):
        def _loading(self):
            data = json.dumps({"error": {"code": 503, "message": "Loading model", "type": "unavailable_error"}}).encode()
            self.send_response(503)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Connection", "close")
            self.close_connection = True
            self.end_headers()
            self.wfile.write(data)

        def handle_expect_100(self):
            if self.admit():
                self._loading()
            return False

        do_GET = do_POST = do_PUT = do_DELETE = do_HEAD = _loading

    return Loading


def linger(sock, seconds=None, idle=None, cap=None):
    """after shutdown(SHUT_WR): reads and drops what the client still sends until it closes, at most `cap` bytes and
    `seconds` (`idle` seconds without data). Closing a socket with unread data makes the kernel send a reset, which can
    reach the client before it has read the answer."""
    end = time.monotonic() + (LINGER_S if seconds is None else seconds)
    left = LINGER_BYTES if cap is None else cap
    idle = LINGER_IDLE_S if idle is None else idle
    buf = bytearray(1 << 16)
    try:
        while left > 0:
            t = end - time.monotonic()
            if t <= 0:
                break
            sock.settimeout(min(t, idle))
            n = sock.recv_into(buf, min(left, len(buf)))
            if not n:
                break
            left -= n
    except OSError:  # (timeout, reset)
        pass


class HTTPServer(ThreadingHTTPServer):
    """ThreadingHTTPServer with a lingering close: when the server ends a connection (an early 400 / 411 / 413 for a
    request body it does not read, the base class's errors, Connection: close), it half-closes it (the client reads the
    whole answer and the end of the stream), then drains what the client still sends (linger) before closing, so the
    client is not reset in the middle of its upload and gets the answer (the reason) instead of a connection error."""
    daemon_threads = True
    gate = None  # the request Gate (main sets it; None: every request is answered)
    def __init__(self, *a, **kw):
        self.bodies = BodyBudget(max(mk_limits.MAX_INFLIGHT_MB, mk_limits.MAX_BODY_MB) << 20)
        self.slots = threading.BoundedSemaphore(mk_limits.MAX_CONNECTIONS)
        self.full_logged = 0.0
        super().__init__(*a, **kw)

    def process_request(self, request, client_address):
        """a thread for the connection, or past MK_MAX_CONNECTIONS a 503 without one (this is the accept loop: nothing
        here may wait, so no lingering close)"""
        if self.slots.acquire(blocking=False):
            try:
                super().process_request(request, client_address)
            except BaseException:
                self.slots.release()
                raise
            return
        if time.monotonic() - self.full_logged >= 60:
            self.full_logged = time.monotonic()
            log(f"the limit of {mk_limits.MAX_CONNECTIONS} connections is reached (MK_MAX_CONNECTIONS): new ones get a 503")
        why = f"more than {mk_limits.MAX_CONNECTIONS} connections (MK_MAX_CONNECTIONS): try again later"
        data = json.dumps({"error": {"code": 503, "message": why, "type": "unavailable_error"}}).encode()
        try:
            request.setblocking(False)
            request.send(b"HTTP/1.1 503 Service Unavailable\r\nContent-Type: application/json\r\nRetry-After: 5\r\n"
                         b"Connection: close\r\nContent-Length: %d\r\n\r\n%b" % (len(data), data))
        except OSError:
            pass
        self.close_request(request)

    def process_request_thread(self, request, client_address):
        try:
            super().process_request_thread(request, client_address)
        finally:
            self.slots.release()

    # Windows: SO_REUSEADDR (the base class's allow_reuse_address) lets a second server bind a port that is in use with
    # no error, and requests then reach either one; an exclusive bind makes the second one fail at startup instead.
    # (Linux: SO_REUSEADDR only allows a restart over connections in TIME_WAIT; a port in use still fails to bind.)
    allow_reuse_address = os.name != "nt"

    def server_bind(self):
        if os.name == "nt" and hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        super().server_bind()

    def handle_error(self, request, client_address):
        """a client that resets its connection (an idle keep-alive one at the end of its session) is not an error; other
        exceptions are reported as the base class does, on the console"""
        if isinstance(sys.exc_info()[1], (ConnectionResetError, ConnectionAbortedError, BrokenPipeError)):
            return
        log(f"error on the connection from {client_address}:\n{traceback.format_exc().rstrip()}")

    def shutdown_request(self, request):
        try:
            request.shutdown(socket.SHUT_WR)
        except OSError:
            pass
        else:
            linger(request)
        self.close_request(request)


def listen(host, port, handler):
    """the HTTPServer on host:port; a port that is taken (another server on it) ends the process with a message"""
    try:
        return HTTPServer((host, port), handler)
    except OSError as e:
        taken = (e.errno in (errno.EADDRINUSE, getattr(errno, "WSAEADDRINUSE", None))
                 or getattr(e, "winerror", 0) in (10048, 10013))  # (Windows: in use / held exclusively)
        sys.exit(f"cannot listen on {host}:{port}: {e.strerror or e}" + (
            " (another server is using the port: choose another with --port, or the scripts' first argument)"
            if taken else ""))
