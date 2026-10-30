#!/usr/bin/env python3
"""OpenJet's Qwen3.8-27B server: OpenAI- and Anthropic-compatible HTTP API in front of the CUDA engine
(a drop-in for the parts of llama-server's API that open-jet uses).

  python3 server/mk_server.py -m model.gguf --port 18080 [--ctx 32768] [--draft 4]

POST /v1/chat/completions (streaming or not): the request is rendered with the chat template stored
in the GGUF (enable_thinking / preserve_thinking / reasoning_effort via chat_template_kwargs, tools
via `tools`), reasoning is returned as `reasoning_content`, Qwen's XML tool calls as structured
`tool_calls`.  Sampling: temperature, top_p, top_k (<= 32), min_p, seed.  Successive requests of a
conversation reuse the processed tokens: the new prompt is matched against the sequence so far and
only the rest is prefilled (rewinding to the end of the previous prompt if it diverges after that).
POST /v1/messages (+ /count_tokens, Anthropic Messages: Claude Code) and POST /v1/responses (OpenAI
Responses: Codex CLI) are translated to a chat completion and back, streaming or not.
GET /health, GET /v1/models.  One request at a time.
"""
import argparse
import codecs
import ctypes
import json
import os
import random
import re
import sys
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


class Engine:
    """ctypes binding of capi.cpp"""

    def __init__(self, lib, model, ctx, n_draft, draft_vocab, kv_mode=-1):
        L = self.L = ctypes.CDLL(lib)
        vp, ip, cp = ctypes.c_void_p, ctypes.POINTER(ctypes.c_int), ctypes.c_char_p
        sig = {
            "mk_new": (vp, [cp, ctypes.c_int, ctypes.c_int, cp, ctypes.c_int]),
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
        }
        for name, (res, args) in sig.items():
            f = getattr(L, name)
            f.restype, f.argtypes = res, args
        self.h = L.mk_new(model.encode(), ctx, n_draft, draft_vocab.encode() if draft_vocab else None, kv_mode)
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


class ContextOverflow(BadRequest):
    def __init__(self, n_prompt, ctx):
        # llama-server's wording: agent clients (Pi) match it to compact and retry
        super().__init__(f"the request exceeds the available context size ({n_prompt} prompt tokens, context {ctx})")
        self.n_prompt, self.ctx = n_prompt, ctx


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
# Anthropic Messages and OpenAI Responses, translated to and from a chat completion

IMAGE_PLACEHOLDER = "[image omitted: this model reads text only]"
# Claude Code stamps its system prompt with a per-request `cch=xxxxx;`, which would end prompt reuse right
# there (llama.cpp normalises it the same way, ggml-org/llama.cpp#21793)
CCH_RE = re.compile(r"^(x-anthropic-billing-header:[^\n]*?cch=)[0-9a-f]{5};")
# the chat template accepts xhigh (its default), medium and low
EFFORTS = {"minimal": "low", "low": "low", "medium": "medium", "high": "xhigh", "xhigh": "xhigh"}


def _blocks_text(content, text_types=("text",)):
    """joins the text of a string or a list of content blocks; images become a placeholder"""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    parts = []
    for b in content:
        if isinstance(b, str):
            parts.append(b)
        elif b.get("type") in text_types:
            parts.append(b.get("text") or "")
        elif b.get("type") in ("image", "input_image", "image_url", "document", "input_file"):
            parts.append(IMAGE_PLACEHOLDER)
    return "\n".join(p for p in parts if p)


def _sampling_fields(body, out):
    for k in ("temperature", "top_p", "top_k", "stream"):
        if body.get(k) is not None:
            out[k] = body[k]
    return out


def anthropic_to_chat(body):
    """Anthropic Messages request -> chat completion request"""
    msgs = []
    system = body.get("system")
    if isinstance(system, list):
        system = _blocks_text(system)
    if system:
        msgs.append({"role": "system", "content": CCH_RE.sub(r"\1fffff;", system)})
    for m in body.get("messages") or []:
        role, content = m.get("role"), m.get("content")
        if isinstance(content, str):
            msgs.append({"role": role, "content": content})
            continue
        text, reasoning, calls = [], [], []
        for b in content or []:
            t = b.get("type")
            if t == "thinking":
                reasoning.append(b.get("thinking") or "")
            elif t == "tool_use":
                calls.append({"id": b.get("id"), "type": "function",
                              "function": {"name": b.get("name"), "arguments": b.get("input") or {}}})
            elif t == "tool_result":
                result = _blocks_text(b.get("content"))
                msgs.append({"role": "tool", "tool_call_id": b.get("tool_use_id"),
                             "content": ("Error: " + result) if b.get("is_error") else result})
            elif t != "redacted_thinking":
                text.append(_blocks_text([b]))
        if role == "assistant":
            msg = {"role": "assistant", "content": "".join(text)}
            if reasoning:
                msg["reasoning_content"] = "".join(reasoning)
            if calls:
                msg["tool_calls"] = calls
            msgs.append(msg)
        elif any(text):
            msgs.append({"role": role, "content": "\n".join(t for t in text if t)})
    out = {"messages": msgs, "max_tokens": body.get("max_tokens")}
    tools = [{"type": "function", "function": {"name": t["name"], "description": t.get("description") or "",
                                                 "parameters": t.get("input_schema") or {"type": "object", "properties": {}}}}
             for t in body.get("tools") or [] if t.get("name") and (t.get("input_schema") or t.get("type") in (None, "custom"))]
    if tools:
        out["tools"] = tools
    thinking = (body.get("thinking") or {}).get("type")
    if thinking:
        out["chat_template_kwargs"] = {"enable_thinking": thinking != "disabled"}
    return _sampling_fields(body, out)


def _call_args(call):
    try:
        args = json.loads(call["function"]["arguments"] or "{}")
    except json.JSONDecodeError:
        args = {}
    return args if isinstance(args, dict) else {}


def anthropic_tool_use(call):
    return {"type": "tool_use", "id": "toolu_" + call["id"].removeprefix("call_"), "name": call["function"]["name"],
            "input": _call_args(call)}


ANTHROPIC_STOP = {"stop": "end_turn", "length": "max_tokens", "tool_calls": "tool_use"}


def anthropic_usage(usage):
    cached = usage["prompt_tokens_details"]["cached_tokens"]
    return {"input_tokens": usage["prompt_tokens"] - cached, "cache_read_input_tokens": cached,
            "cache_creation_input_tokens": 0, "output_tokens": usage["completion_tokens"]}


def anthropic_message(mid, model, reasoning, content, calls, finish, usage):
    blocks = []
    if reasoning:
        blocks.append({"type": "thinking", "thinking": reasoning, "signature": ""})
    if content:
        blocks.append({"type": "text", "text": content})
    blocks += [anthropic_tool_use(c) for c in calls]
    return {"id": mid, "type": "message", "role": "assistant", "model": model, "content": blocks,
            "stop_reason": ANTHROPIC_STOP.get(finish, "end_turn"), "stop_sequence": None, "usage": anthropic_usage(usage)}


class AnthropicStream:
    """Anthropic SSE events for one message; send(event, data) writes one event"""

    def __init__(self, send, mid, model):
        self.send, self.mid, self.model = send, mid, model
        self.index, self.kind = -1, None

    def start(self, n_prompt, reused):
        usage = {"input_tokens": n_prompt - reused, "cache_read_input_tokens": reused,
                 "cache_creation_input_tokens": 0, "output_tokens": 0}
        self.send("message_start", {"type": "message_start", "message": {
            "id": self.mid, "type": "message", "role": "assistant", "model": self.model, "content": [],
            "stop_reason": None, "stop_sequence": None, "usage": usage}})

    def _close(self):
        if self.kind == "reasoning":
            self.send("content_block_delta", {"type": "content_block_delta", "index": self.index,
                                              "delta": {"type": "signature_delta", "signature": ""}})
        if self.kind is not None:
            self.send("content_block_stop", {"type": "content_block_stop", "index": self.index})
        self.kind = None

    def _open(self, block):
        self.index += 1
        self.send("content_block_start", {"type": "content_block_start", "index": self.index, "content_block": block})

    def delta(self, kind, text):
        if kind != self.kind:
            self._close()
            self._open({"type": "thinking", "thinking": "", "signature": ""} if kind == "reasoning"
                       else {"type": "text", "text": ""})
            self.kind = kind
        delta = {"type": "thinking_delta", "thinking": text} if kind == "reasoning" else {"type": "text_delta", "text": text}
        self.send("content_block_delta", {"type": "content_block_delta", "index": self.index, "delta": delta})

    def finish(self, finish, calls, usage):
        self._close()
        for c in calls:
            block = anthropic_tool_use(c)
            args = json.dumps(block["input"], ensure_ascii=False)
            self._open({**block, "input": {}})
            self.send("content_block_delta", {"type": "content_block_delta", "index": self.index,
                                              "delta": {"type": "input_json_delta", "partial_json": args}})
            self.send("content_block_stop", {"type": "content_block_stop", "index": self.index})
        self.send("message_delta", {"type": "message_delta",
                                    "delta": {"stop_reason": ANTHROPIC_STOP.get(finish, "end_turn"), "stop_sequence": None},
                                    "usage": anthropic_usage(usage)})
        self.send("message_stop", {"type": "message_stop"})


def responses_to_chat(body, warn=lambda key, msg: None):
    """OpenAI Responses request -> chat completion request. Custom (freeform) tools become functions with a
    single string argument `input`."""
    if body.get("previous_response_id"):
        raise BadRequest("previous_response_id is not supported: send the whole conversation in `input`")
    msgs = []
    if body.get("instructions"):
        msgs.append({"role": "system", "content": body["instructions"]})
    items = body.get("input")
    if isinstance(items, str):
        items = [{"role": "user", "content": items}]
    reasoning = None
    text_types = ("input_text", "output_text", "text", "refusal")

    def assistant():
        nonlocal reasoning
        last = msgs[-1] if msgs else None
        if last is not None and last["role"] == "assistant" and reasoning is None:
            return last
        msg = {"role": "assistant", "content": ""}
        if reasoning is not None:
            msg["reasoning_content"] = reasoning
            reasoning = None
        msgs.append(msg)
        return msg

    for item in items or []:
        t = item.get("type") or ("message" if "role" in item else None)
        if t == "message":
            role = "system" if item.get("role") == "developer" else item.get("role")
            text = _blocks_text(item.get("content"), text_types)
            if role == "assistant":
                msg = assistant()
                msg["content"] += text
            else:
                msgs.append({"role": role, "content": text})
        elif t in ("function_call", "custom_tool_call"):
            args = item.get("arguments") if t == "function_call" else {"input": item.get("input") or ""}
            assistant().setdefault("tool_calls", []).append(
                {"id": item.get("call_id"), "type": "function", "function": {"name": item.get("name"), "arguments": args}})
        elif t in ("function_call_output", "custom_tool_call_output"):
            msgs.append({"role": "tool", "tool_call_id": item.get("call_id"),
                         "content": _blocks_text(item.get("output"), text_types)})
        elif t == "reasoning":
            reasoning = _blocks_text(item.get("content"), ("reasoning_text",)) or \
                _blocks_text(item.get("summary"), ("summary_text",))
        else:
            warn(f"item:{t}", f"Responses input item type {t!r} is not supported and is skipped")
    out = {"messages": msgs, "max_tokens": body.get("max_output_tokens")}
    tools = []
    for t in body.get("tools") or []:
        if t.get("type") == "function":
            fn = {k: t[k] for k in ("name", "description", "parameters") if k in t}
            tools.append({"type": "function", "function": fn})
        elif t.get("type") == "custom":
            desc = (t.get("description") or "") + "\nPass the raw input as the `input` string."
            tools.append({"type": "function", "function": {
                "name": t["name"], "description": desc.strip(),
                "parameters": {"type": "object", "properties": {"input": {"type": "string"}}, "required": ["input"]}}})
        else:
            warn(f"tool:{t.get('type')}", f"Responses tool type {t.get('type')!r} is not supported and is skipped")
    if tools:
        out["tools"] = tools
    effort = (body.get("reasoning") or {}).get("effort")
    if effort == "none":
        out["chat_template_kwargs"] = {"enable_thinking": False}
    elif effort in EFFORTS:
        out["chat_template_kwargs"] = {"reasoning_effort": EFFORTS[effort]}
    return _sampling_fields(body, out)


def responses_tool_item(call, custom):
    """a chat tool call -> a Responses output item (a custom_tool_call for the tools in `custom`)"""
    cid, name = call["id"], call["function"]["name"]
    if name in custom:
        return {"type": "custom_tool_call", "id": "ctc_" + cid.removeprefix("call_"), "call_id": cid, "name": name,
                "input": str(_call_args(call).get("input", "")), "status": "completed"}
    return {"type": "function_call", "id": "fc_" + cid.removeprefix("call_"), "call_id": cid, "name": name,
            "arguments": call["function"]["arguments"], "status": "completed"}


def responses_usage(usage):
    return {"input_tokens": usage["prompt_tokens"], "input_tokens_details": usage["prompt_tokens_details"],
            "output_tokens": usage["completion_tokens"], "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": usage["total_tokens"]}


def response_object(rid, model, created, output, finish=None, usage=None):
    obj = {"id": rid, "object": "response", "created_at": created, "model": model, "output": output,
           "status": "in_progress" if finish is None else "incomplete" if finish == "length" else "completed",
           "incomplete_details": {"reason": "max_output_tokens"} if finish == "length" else None,
           "parallel_tool_calls": True, "tool_choice": "auto", "tools": []}
    if usage is not None:
        obj["usage"] = responses_usage(usage)
    return obj


def _reasoning_item(rid, text):
    return {"type": "reasoning", "id": rid, "summary": [], "content": [{"type": "reasoning_text", "text": text}]}


def _message_item(mid, text, status="completed"):
    return {"type": "message", "id": mid, "status": status, "role": "assistant",
            "content": [{"type": "output_text", "text": text, "annotations": []}] if text is not None else []}


def responses_output(reasoning, content, calls, custom):
    out = []
    if reasoning:
        out.append(_reasoning_item("rs_" + uuid.uuid4().hex, reasoning))
    if content:
        out.append(_message_item("msg_" + uuid.uuid4().hex, content))
    return out + [responses_tool_item(c, custom) for c in calls]


class ResponsesStream:
    """OpenAI Responses SSE events for one response; send(event, data) writes one event"""

    def __init__(self, send, rid, model, custom):
        self.send, self.rid, self.model, self.custom = send, rid, model, custom
        self.created, self.seq = int(time.time()), 0
        self.output, self.kind, self.item, self.text = [], None, None, []

    def _send(self, event, **data):
        data = {"type": event, "sequence_number": self.seq, **data}
        self.seq += 1
        self.send(event, data)

    def start(self, n_prompt, reused):
        response = response_object(self.rid, self.model, self.created, [])
        self._send("response.created", response=response)
        self._send("response.in_progress", response=response)

    def _close(self):
        if self.kind is None:
            return
        text, i = "".join(self.text), len(self.output)
        if self.kind == "reasoning":
            item = _reasoning_item(self.item, text)
            self._send("response.reasoning_text.done", item_id=self.item, output_index=i, content_index=0, text=text)
        else:
            item = _message_item(self.item, text)
            part = item["content"][0]
            self._send("response.output_text.done", item_id=self.item, output_index=i, content_index=0, text=text)
            self._send("response.content_part.done", item_id=self.item, output_index=i, content_index=0, part=part)
        self._send("response.output_item.done", output_index=i, item=item)
        self.output.append(item)
        self.kind, self.text = None, []

    def delta(self, kind, text):
        i = len(self.output)
        if kind != self.kind:
            self._close()
            i = len(self.output)
            self.kind = kind
            if kind == "reasoning":
                self.item = "rs_" + uuid.uuid4().hex
                self._send("response.output_item.added", output_index=i,
                           item={"type": "reasoning", "id": self.item, "summary": [], "content": []})
            else:
                self.item = "msg_" + uuid.uuid4().hex
                self._send("response.output_item.added", output_index=i, item=_message_item(self.item, None, "in_progress"))
                self._send("response.content_part.added", item_id=self.item, output_index=i, content_index=0,
                           part={"type": "output_text", "text": "", "annotations": []})
        self.text.append(text)
        event = "response.reasoning_text.delta" if kind == "reasoning" else "response.output_text.delta"
        self._send(event, item_id=self.item, output_index=i, content_index=0, delta=text)

    def finish(self, finish, calls, usage):
        self._close()
        for c in calls:
            item, i = responses_tool_item(c, self.custom), len(self.output)
            self._send("response.output_item.added", output_index=i, item={**item, "status": "in_progress",
                       **({"input": ""} if item["type"] == "custom_tool_call" else {"arguments": ""})})
            if item["type"] == "custom_tool_call":
                self._send("response.custom_tool_call_input.delta", item_id=item["id"], output_index=i, delta=item["input"])
                self._send("response.custom_tool_call_input.done", item_id=item["id"], output_index=i, input=item["input"])
            else:
                self._send("response.function_call_arguments.delta", item_id=item["id"], output_index=i,
                           delta=item["arguments"])
                self._send("response.function_call_arguments.done", item_id=item["id"], output_index=i,
                           arguments=item["arguments"])
            self._send("response.output_item.done", output_index=i, item=item)
            self.output.append(item)
        response = response_object(self.rid, self.model, self.created, self.output, finish, usage)
        self._send("response.incomplete" if finish == "length" else "response.completed", response=response)


# ---------------------------------------------------------------------------------------------


class Server:
    def __init__(self, args):
        self.args = args
        t0 = time.time()
        self.eng = Engine(args.lib, args.model, args.ctx, args.draft, args.draft_vocab,
                          {"f16": 0, "q8": 1, "auto": -1}[args.kv])
        self.template = make_template(self.eng.chat_template())
        self.warm_up()
        self.model_id = os.path.basename(args.model)
        self.lock = threading.Lock()
        self.warned = set()
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

    def render(self, body):
        kw = dict(body.get("chat_template_kwargs") or {})
        ctx = dict(messages=normalize_messages(body.get("messages") or []), add_generation_prompt=True)
        if body.get("tools"):
            ctx["tools"] = body["tools"]
        for k in ("enable_thinking", "preserve_thinking", "reasoning_effort"):
            if k in kw:
                ctx[k] = kw[k]
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
            self.eng.set_sampling(0.0, 1.0, TOPK_MAX, 0.0, seed)
        else:
            self.eng.set_sampling(temp, top_p, top_k, min_p, seed)

    def generate(self, body, emit, on_prompt=None):
        """runs one chat completion; emit(kind, payload) is called with ('reasoning'|'content', text) deltas.
        on_prompt(prompt tokens, reused tokens) is called once the prompt is processed, before the first delta.
        Returns (finish_reason, tool_calls, usage, timings). emit returning False aborts."""
        text, thinking = self.render(body)
        toks = self.eng.tokenize(text)
        max_new = body.get("max_tokens") or body.get("max_completion_tokens") or body.get("n_predict") or -1
        max_new = int(max_new)
        limit = self.args.ctx - len(toks) - 16
        if limit <= 0:
            raise ContextOverflow(len(toks), self.args.ctx)
        max_new = limit if max_new <= 0 else min(max_new, limit)
        tools = body.get("tools") or []
        self.sampling(body)
        t0 = time.time()
        tok, reused = self.load_prompt(toks)
        t1 = time.time()
        if on_prompt is not None:
            on_prompt(len(toks), reused)
        parser = OutputParser(thinking, tools)
        dec = codecs.getincrementaldecoder("utf-8")(errors="replace")
        n_gen, finish, pending, aborted = 0, "length", [tok], False
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
                self._json(200, {"default_generation_settings": {"n_ctx": srv.args.ctx}, "total_slots": 1})
            else:
                self._json(404, {"error": {"message": "not found", "type": "not_found_error"}})

        def _error(self, code, kind, message, anthropic):
            if anthropic:
                self._json(code, {"type": "error", "error": {"type": kind, "message": message}})
            else:
                self._json(code, {"error": {"message": message, "type": kind}})

        def do_POST(self):
            path = self.path.split("?")[0]
            n = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(n) if n else b""
            routes = {"/v1/chat/completions": self._chat, "/chat/completions": self._chat,
                      "/v1/messages": self._messages, "/v1/messages/count_tokens": self._count_tokens,
                      "/v1/responses": self._responses, "/responses": self._responses}
            anthropic = path.startswith("/v1/messages")
            if path not in routes:
                self._error(404 if not path.startswith("/slots") else 501, "not_supported",
                            f"{path} is not supported by the OpenJet engine server", anthropic)
                return
            try:
                body = json.loads(raw or b"{}")
            except json.JSONDecodeError:
                self._error(400, "invalid_request_error", "invalid JSON", anthropic)
                return
            if path == "/v1/messages/count_tokens":  # tokenizer only: does not wait for a running generation
                try:
                    self._count_tokens(body)
                except BadRequest as e:
                    self._error(400, "invalid_request_error", str(e), True)
                return
            self._sse_started = False
            with srv.lock:
                try:
                    routes[path](body)
                except BadRequest as e:
                    if anthropic and isinstance(e, ContextOverflow):  # Claude Code compacts on this wording
                        e = BadRequest(f"prompt is too long: {e.n_prompt} tokens > {e.ctx} maximum")
                    self._fail(400, "invalid_request_error", str(e), anthropic)
                except (BrokenPipeError, ConnectionResetError):
                    pass
                except Exception as e:  # engine failure: report it, keep serving
                    log("error:", repr(e))
                    try:
                        self._fail(500, "api_error" if anthropic else "server_error", str(e), anthropic)
                    except Exception:
                        pass

        def _fail(self, code, kind, message, anthropic):
            """an error response, or an error event when an Anthropic / Responses stream has started"""
            if not self._sse_started:
                self._error(code, kind, message, anthropic)
            else:
                self._sse("error", {"type": "error", "error": {"type": kind, "message": message}} if anthropic
                          else {"type": "error", "code": kind, "message": message, "param": None})
                self._sse_end()

        # Server-sent events with chunked transfer encoding, for /v1/messages and /v1/responses

        def _sse_head(self):
            if not self._sse_started:
                self._sse_started = True
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()

        def _sse(self, event, obj):
            self._sse_head()
            data = f"event: {event}\ndata: ".encode() + json.dumps(obj, ensure_ascii=False).encode() + b"\n\n"
            self.wfile.write(b"%x\r\n" % len(data) + data + b"\r\n")
            self.wfile.flush()

        def _sse_end(self):
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()

        def _run_stream(self, chat, stream):
            """runs `chat` through a protocol stream (AnthropicStream / ResponsesStream)"""
            def emit(kind, text):
                try:
                    stream.delta(kind, text)
                except (BrokenPipeError, ConnectionResetError):
                    return False
                return True
            finish, calls, usage, _ = srv.generate(chat, emit, on_prompt=stream.start)
            stream.finish(finish, calls, usage)
            self._sse_end()

        def _collect(self, chat):
            parts = {"reasoning": [], "content": []}
            finish, calls, usage, timings = srv.generate(chat, lambda k, t: parts[k].append(t))
            return "".join(parts["reasoning"]).strip(), "".join(parts["content"]), calls, finish, usage, timings

        def _messages(self, body):
            chat = anthropic_to_chat(body)
            mid = "msg_" + uuid.uuid4().hex
            if body.get("stop_sequences"):
                srv.warn_once("stop_sequences", "stop_sequences is not supported by the OpenJet engine and is ignored")
            if chat.get("stream"):
                self._run_stream(chat, AnthropicStream(self._sse, mid, srv.model_id))
                return
            reasoning, content, calls, finish, usage, _ = self._collect(chat)
            self._json(200, anthropic_message(mid, srv.model_id, reasoning, content, calls, finish, usage))

        def _count_tokens(self, body):
            text, _ = srv.render(anthropic_to_chat(body))
            self._json(200, {"input_tokens": len(srv.eng.tokenize(text))})

        def _responses(self, body):
            chat = responses_to_chat(body, srv.warn_once)
            custom = {t["name"] for t in body.get("tools") or [] if t.get("type") == "custom"}
            rid = "resp_" + uuid.uuid4().hex
            if chat.get("stream"):
                self._run_stream(chat, ResponsesStream(self._sse, rid, srv.model_id, custom))
                return
            reasoning, content, calls, finish, usage, _ = self._collect(chat)
            output = responses_output(reasoning, content, calls, custom)
            self._json(200, response_object(rid, srv.model_id, int(time.time()), output, finish, usage))

        def _chat(self, body):
            if body.get("stream"):
                self._stream(body)
            else:
                self._complete(body)

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
        description="OpenJet: CUDA inference for Qwen3.8-27B with OpenAI- and Anthropic-compatible HTTP APIs."
    )
    ap.add_argument("-m", "--model", required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=18080)
    ap.add_argument("-c", "--ctx", type=int, default=32768)
    ap.add_argument("-d", "--draft", type=int, default=4)
    ap.add_argument("--draft-vocab", default=os.path.join(os.path.dirname(HERE), "draft_vocab_32k.bin"))
    ap.add_argument("--lib", default=os.path.join(os.path.dirname(HERE), "qwen38mk.dll" if os.name == "nt" else "libqwen38mk.so"))
    ap.add_argument("--kv", choices=["auto", "f16", "q8"], default="auto",
                    help="KV cache format (auto: q8_0 only when the fp16 cache does not fit)")
    args = ap.parse_args()
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
