"""Translation between the OpenAI Responses API and the chat request."""
import base64
import binascii
import json
import time
import zlib

import mk_limits
from mk_errors import BadRequest
from mk_limits import DEF_TEMP, DEF_TOP_P


# ---------------------------------------------------------------------------------------------
# OpenAI Responses API (POST /v1/responses, /v1/responses/input_tokens): translated to the chat request and back,
# as the Anthropic Messages API is (llama-server's server_chat_convert_responses_to_chatcmpl). Stateless: the client
# sends the whole conversation every time (store false, as Codex does); nothing is stored, so previous_response_id,
# conversation and item_reference are a 400.

# a reasoning item's encrypted_content (include reasoning.encrypted_content): this prefix + the reasoning text, zlib +
# base64. Not secret (the item has the text as reasoning_text content too): it brings the reasoning back for clients that
# return only encrypted_content, so the next prompt renders the turn as it was generated
RESP_REASONING = "mk-reasoning-1:"
RESP_KEEPALIVE_S = 10.0  # a stream's response.in_progress events during a long prefill (clients time out idle streams)


def reasoning_seal(text):
    return RESP_REASONING + base64.b64encode(zlib.compress(text.encode(), 6)).decode()


def reasoning_open(s, budget=None):
    """this server's encrypted_content as text, else None; budget: [bytes left for the request], past it a BadRequest"""
    if not isinstance(s, str) or not s.startswith(RESP_REASONING):
        return None
    budget = [mk_limits.REASONING_MAX_BYTES] if budget is None else budget
    d = zlib.decompressobj()
    try:  # (max_length budget + 1 > 0: zlib reads 0 as no limit)
        raw = d.decompress(base64.b64decode(s[len(RESP_REASONING):], validate=True), budget[0] + 1)
    except (binascii.Error, zlib.error, ValueError):
        return None
    if len(raw) > budget[0]:
        raise BadRequest(f"reasoning encrypted_content: the request's reasoning inflates to more than the model's full "
                         f"context could hold ({mk_limits.REASONING_MAX_BYTES} bytes in all)")
    budget[0] -= len(raw)
    if not d.eof or d.unconsumed_tail or d.unused_data:
        return None
    try:
        return raw.decode()
    except UnicodeDecodeError:
        return None


def _resp_content(content, where):
    """a message item's or function output's content (a string, or input_text / output_text / refusal / input_image
    parts) as chat content: one string (text parts joined by newlines, as normalize_messages joins a chat part list),
    or with images the chat parts (normalize_messages flattens them as llama-server does)"""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        raise BadRequest(f"{where}: a string or an array of content parts")
    parts = []
    for p in content:
        if not isinstance(p, dict):
            raise BadRequest(f"{where}: content parts must be objects")
        t = p.get("type")
        if t in ("input_text", "output_text", "text"):
            parts.append({"type": "text", "text": str(p.get("text") or "")})
        elif t == "refusal":
            parts.append({"type": "text", "text": str(p.get("refusal") or "")})
        elif t == "input_image":
            url = p.get("image_url")
            if not url and p.get("file_id"):
                raise BadRequest("input_image: file_id is not supported (the server stores no files): send the image as "
                                 "a data: URL in image_url")
            parts.append({"type": "image_url", "image_url": {"url": str((url.get("url") if isinstance(url, dict) else url)
                                                                     or "")}})
        elif t == "input_file":
            raise BadRequest("input_file is not supported: send the file's text as input_text (images as input_image)")
        else:
            raise BadRequest(f"{where}: content part type {t!r} is not supported (input_text, output_text, refusal, "
                             "input_image)")
    if any(p["type"] == "image_url" for p in parts):
        return parts
    text = ""
    for p in parts:
        text += ("\n" if text else "") + p["text"]
    return text


def responses_tools(tools, notes):
    """a Responses request's tools as chat tools: (tools, {function name: its namespace}, {custom tool names}).
    function -> function; namespace -> its functions (plain names; a call to one gets its namespace back); custom
    (freeform input) -> a function with one string parameter "input"; the others (web_search, file_search, mcp, ...:
    tools the server would have to run) are dropped with a note"""
    out, ns, custom = [], {}, set()

    def add(t, space=None):
        name = t.get("name")
        if not isinstance(name, str) or not name:
            raise BadRequest(f"tools: a {t.get('type')} tool needs a name")
        if any(o["function"]["name"] == name for o in out):
            raise BadRequest(f"tools: two tools are named {name!r} (namespaces are flattened)")
        f = {"name": name, "description": str(t.get("description") or "")}
        if t.get("type") == "custom":
            fmt = t.get("format") if isinstance(t.get("format"), dict) else {}
            what = "The tool's input, free-form text"
            if fmt.get("type") == "grammar" and fmt.get("definition"):
                what += " in this " + (f"{fmt['syntax']} " if fmt.get("syntax") else "") + f"grammar:\n{fmt['definition']}"
            f["parameters"] = {"type": "object", "properties": {"input": {"type": "string", "description": what}},
                               "required": ["input"]}
            custom.add(name)
        else:
            f["parameters"] = t.get("parameters") if isinstance(t.get("parameters"), dict) else {}
        if space:
            ns[name] = space
        out.append({"type": "function", "function": f})

    for t in tools:
        if not isinstance(t, dict):
            raise BadRequest("tools: each tool must be an object")
        kind = t.get("type")
        if kind in ("function", "custom"):
            add(t)
        elif kind == "namespace":
            for sub in t.get("tools") or []:
                if not isinstance(sub, dict) or sub.get("type") not in ("function", "custom"):
                    raise BadRequest(f"tools: namespace {t.get('name')!r}: function or custom tools only")
                add(sub, str(t.get("name") or ""))
        else:
            notes.append((f"responses-tool-{kind}", f"Responses tool type {kind!r} is not supported (the server runs no "
                          "tools): it is not offered to the model"))
    return out, ns, custom


def responses_to_chat(body, notes=None, mid_system=True):
    """a Responses API request as the server's chat request: (chat body, {"ns": function namespaces, "custom": custom
    tool names}: what the output items need). input: a string (one user message) or items: messages (roles user,
    assistant, system, developer -> system), reasoning (-> reasoning_content: its reasoning_text content, else this
    server's encrypted_content, else its summary), function_call / custom_tool_call (-> tool_calls) and their outputs
    (-> tool messages); the assistant-side items of one turn become one assistant message in generation order.
    instructions and the leading system / developer messages become one system message (their trimmed texts joined by
    newlines, as Qwen's templates merge them); a later one becomes a user message in its place when the template takes
    a system message only at the start (mid_system False). Whatever would change the output and cannot be honoured is
    a BadRequest (stored state, forced tool calls, JSON output, logprobs, files). notes: gets (key, message) for what was
    dropped or converted (the console's one-time warnings)"""
    notes = [] if notes is None else notes

    def bad(field, why):
        raise BadRequest(f"{field}: {why}")
    for k in ("previous_response_id", "conversation"):
        if body.get(k):
            bad(k, "not supported: the server stores no responses (send the whole conversation in input, with store "
                   "false)")
    if body.get("background"):
        bad("background", "not supported: the server stores no responses")
    if body.get("prompt"):
        bad("prompt", "stored prompt templates are not supported")
    inp = body.get("input")
    items = [{"type": "message", "role": "user", "content": inp}] if isinstance(inp, str) else inp
    if not isinstance(items, list):
        bad("input", "a string or an array of input items is required")
    msgs = []
    budget = [mk_limits.REASONING_MAX_BYTES]  # (one decompression budget for all of the request's encrypted_content)
    if body.get("instructions") is not None:
        if not isinstance(body["instructions"], str):
            bad("instructions", "a string")
        msgs.append({"role": "system", "content": body["instructions"]})
    # stage of the last message when it is an assistant message: 0 reasoning, 1 text, 2 tool calls (the order the model
    # generates them in, one reasoning and one text each); an item that cannot come next starts a new assistant message
    stage = [None]

    def turn(s):
        if stage[0] is None or stage[0] > s or (stage[0] == s and s < 2):
            msgs.append({"role": "assistant", "content": ""})
        stage[0] = s
        return msgs[-1]

    def other(m):
        msgs.append(m)
        stage[0] = None

    for it in items:
        if not isinstance(it, dict):
            bad("input", "input items must be objects")
        t = it.get("type") or ("message" if "role" in it else None)
        if t == "message":
            role = it.get("role")
            if role not in ("user", "assistant", "system", "developer"):
                bad("input", f"message role {role!r}: user, assistant, system or developer")
            content = _resp_content(it.get("content"), "input message content")
            if role == "assistant":
                turn(1)["content"] = content
            else:
                other({"role": "system" if role == "developer" else role, "content": content})
        elif t == "reasoning":
            parts = it.get("content") if isinstance(it.get("content"), list) else []
            texts = [str(p.get("text") or "") for p in parts if isinstance(p, dict) and p.get("type") == "reasoning_text"]
            text = "".join(texts) if texts else reasoning_open(it.get("encrypted_content"), budget)
            if text is None:
                summary = it.get("summary") if isinstance(it.get("summary"), list) else []
                text = "\n\n".join(str(p.get("text") or "") for p in summary if isinstance(p, dict))
            turn(0)["reasoning_content"] = text
        elif t in ("function_call", "custom_tool_call"):
            if not isinstance(it.get("name"), str) or not it.get("name"):
                bad("input", f"a {t} item needs a name")
            args = it.get("arguments") if t == "function_call" else json.dumps({"input": str(it.get("input") or "")},
                                                                                ensure_ascii=False)
            turn(2).setdefault("tool_calls", []).append({"id": str(it.get("call_id") or ""), "type": "function", "function": {
                "name": it["name"], "arguments": args if isinstance(args, str) else json.dumps(args or {}, ensure_ascii=False)}})
        elif t in ("function_call_output", "custom_tool_call_output"):
            other({"role": "tool", "tool_call_id": str(it.get("call_id") or ""),
                   "content": _resp_content(it.get("output") if it.get("output") is not None else "", f"{t} output")})
        elif t == "item_reference":
            bad("input", "item_reference is not supported: the server stores no items (send the items themselves)")
        else:  # (built-in tool calls the server never makes: web_search_call, local_shell_call, compaction, ...)
            notes.append((f"responses-item-{t}", f"Responses input item type {t!r} is not supported: skipped"))
    # one system message at the start (Qwen's templates merge leading system / developer messages this way; the original
    # template takes only the first)
    lead = 0
    while lead < len(msgs) and msgs[lead]["role"] == "system" and isinstance(msgs[lead]["content"], str):
        lead += 1
    if lead > 1:
        msgs[:lead] = [{"role": "system", "content": "\n".join(
            s for s in (m["content"].strip() for m in msgs[:lead]) if s)}]
    if not mid_system:
        for i, m in enumerate(msgs):
            if i and m["role"] == "system":
                m["role"] = "user"
                notes.append(("responses-mid-system", "Responses system / developer messages inside the conversation are "
                              "rendered as user messages in their place: the chat template takes a system message only "
                              "at the start"))
    out = {"messages": msgs, "stream": bool(body.get("stream"))}
    if body.get("max_output_tokens") is not None:
        out["max_tokens"] = body["max_output_tokens"]
    # (temperature, top_p: the API's; the others: the chat path's extensions, as llama-server passes the body through)
    for k in ("temperature", "top_p", "top_k", "min_p", "seed", "presence_penalty", "frequency_penalty",
              "chat_template_kwargs", "top_logprobs"):
        if body.get(k) is not None:
            out[k] = body[k]
    include = body.get("include") or []
    if not isinstance(include, list):
        bad("include", "an array of strings")
    if "message.output_text.logprobs" in include:
        bad("include", "message.output_text.logprobs: token log-probabilities are not available from the megakernel")
    text = body.get("text")
    if text is not None:
        if not isinstance(text, dict):
            bad("text", "an object")
        fmt = text.get("format")
        if fmt is not None and not (isinstance(fmt, dict) and fmt.get("type", "text") == "text"):
            bad("text.format", "only {\"type\": \"text\"}: JSON output needs constrained sampling (a grammar), which the "
                               "megakernel does not have")
        if text.get("verbosity") not in (None, "medium"):
            notes.append(("responses-verbosity", "Responses text.verbosity is not supported (the model has no such "
                          "control): the reply is generated as usual"))
    reasoning = body.get("reasoning")
    if reasoning is not None:
        if not isinstance(reasoning, dict):
            bad("reasoning", "an object")
        if isinstance(reasoning.get("effort"), str) and reasoning["effort"]:
            out["reasoning_effort"] = reasoning["effort"]  # (Server.render: none = thinking off, minimal = low)
        if (reasoning.get("summary") or reasoning.get("generate_summary")) not in (None, "none"):
            notes.append(("responses-summary", "Responses reasoning summaries are not generated: the reasoning is "
                          "returned whole as reasoning_text content (Codex shows it with show_raw_agent_reasoning)"))
    tools = body.get("tools")
    info = {"ns": {}, "custom": set()}
    if tools:
        if not isinstance(tools, list):
            bad("tools", "an array of tools")
        out["tools"], info["ns"], info["custom"] = responses_tools(tools, notes)
    tc = body.get("tool_choice")
    if tc not in (None, "auto", "none"):
        bad("tool_choice", "\"auto\" or \"none\": forcing a tool call (required / a named or allowed tool) needs "
                           "constrained sampling, which the megakernel does not have")
    if tc is not None:
        out["tool_choice"] = tc
    if body.get("parallel_tool_calls") is False:
        out["parallel_tool_calls"] = False
    return out, info


def responses_usage(usage, reasoning_tokens=0):
    """the chat usage as a Response's (input_tokens counts the reused ones, as OpenAI's does)"""
    return {"input_tokens": usage["prompt_tokens"],
            "input_tokens_details": {"cached_tokens": usage["prompt_tokens_details"]["cached_tokens"], "cache_write_tokens": 0},
            "output_tokens": usage["completion_tokens"], "output_tokens_details": {"reasoning_tokens": reasoning_tokens},
            "total_tokens": usage["total_tokens"]}


def responses_reasoning(text, rid, sealed, status="completed"):
    item = {"type": "reasoning", "id": rid, "summary": [], "content": [{"type": "reasoning_text", "text": text}],
            "encrypted_content": reasoning_seal(text) if sealed else None, "status": status}
    return item


def responses_message(text, mid, status="completed"):
    return {"type": "message", "id": mid, "status": status, "role": "assistant",
            "content": [{"type": "output_text", "text": text, "annotations": [], "logprobs": []}]}


def responses_call(call_id, name, arguments, info, status="completed"):
    """a tool call as a function_call item (with its namespace), or a custom_tool_call for a custom tool (its input:
    the call's "input" argument)"""
    key = call_id.removeprefix("call_")
    if name in info["custom"]:
        try:
            inp = json.loads(arguments).get("input")
        except (ValueError, AttributeError):
            inp = None
        return {"type": "custom_tool_call", "id": "ctc_" + key, "call_id": call_id, "name": name,
                "input": inp if isinstance(inp, str) else "", "status": status}
    item = {"type": "function_call", "id": "fc_" + key, "call_id": call_id, "name": name, "arguments": arguments,
            "status": status}
    if name in info["ns"]:
        item["namespace"] = info["ns"][name]
    return item


def response_object(rid, created, body, model, status, output, usage=None, error=None, incomplete=None):
    """a Response object (the request's settings echoed as OpenAI does; store false: nothing is kept)"""
    r = body.get("reasoning") if isinstance(body.get("reasoning"), dict) else {}
    text = body.get("text") if isinstance(body.get("text"), dict) else {}
    return {"id": rid, "object": "response", "created_at": created, "status": status, "background": False,
            "completed_at": int(time.time()) if status == "completed" else None, "error": error,
            "incomplete_details": incomplete, "instructions": body.get("instructions"),
            "max_output_tokens": body.get("max_output_tokens"), "max_tool_calls": body.get("max_tool_calls"),
            "model": model, "output": output, "parallel_tool_calls": body.get("parallel_tool_calls") is not False,
            "previous_response_id": None, "reasoning": {"effort": r.get("effort"), "summary": r.get("summary")},
            "service_tier": "default", "store": False,
            "temperature": body["temperature"] if body.get("temperature") is not None else DEF_TEMP,
            "text": {"format": {"type": "text"}, "verbosity": text.get("verbosity") or "medium"},
            "tool_choice": body.get("tool_choice") or "auto", "tools": body.get("tools") or [],
            "top_p": body["top_p"] if body.get("top_p") is not None else DEF_TOP_P,
            "truncation": body.get("truncation") or "disabled", "usage": usage, "user": body.get("user"),
            "metadata": body.get("metadata") or {}}
