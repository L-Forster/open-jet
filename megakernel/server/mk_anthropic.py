"""Translation between the Anthropic Messages API and the chat request."""
import json
import re

from mk_chat import stop_strings
from mk_errors import BadRequest


# ---------------------------------------------------------------------------------------------
# Anthropic Messages API (POST /v1/messages, /v1/messages/count_tokens): translated to the chat request and back,
# so one code path generates for both (as llama-server's server_chat_convert_anthropic_to_oai)

# Claude Code's system prompt starts with a billing header whose cch= value changes on every request: normalized (as
# llama-server does), or the prompt would differ from its first tokens on and no prefix could be reused
BILLING_RE = re.compile(r"^(x-anthropic-billing-header:.*?cch=)[^;\s]{5};")
# Anthropic request fields the server does not act on: accepted (clients send some on every request), each a one-time
# console warning
IGNORED_FIELDS = {
    "output_format": "structured output needs constrained sampling, which the megakernel does not have: the reply is "
                     "not held to the format",
    "output_config": "its format is not applied (its effort is: the chat template's reasoning effort)",
    "context_management": "context editing is not done: the conversation is rendered as sent",
    "container": "the server runs no code-execution container"}
# output_config.effort (Claude Code's /effort; Claude Code sends high by default) -> the chat template's reasoning_effort,
# whose levels are low, medium and xhigh (the default)
EFFORT = {"low": "low", "medium": "medium", "high": "xhigh", "max": "xhigh"}


def _anthropic_image(src):
    if not isinstance(src, dict):
        raise BadRequest("image: a source object is required")
    if src.get("type") == "base64":
        return f"data:{src.get('media_type') or 'image/jpeg'};base64,{src.get('data') or ''}"
    if src.get("type") == "url":
        return str(src.get("url") or "")  # (refused by image_bytes: only data: URLs)
    raise BadRequest(f"image source type {src.get('type')!r} is not supported (base64)")


def _anthropic_document(b, notes):
    """a document block's text: plain-text and content sources (title and context first); None for the others (PDF,
    URL, file: skipped with a note)"""
    src = b.get("source") if isinstance(b.get("source"), dict) else {}
    if src.get("type") == "text":
        text = str(src.get("data") or "")
    elif src.get("type") == "content" and isinstance(src.get("content"), (str, list)):
        c = src["content"]
        text = c if isinstance(c, str) else "\n".join(
            str(x.get("text") or "") for x in c if isinstance(x, dict) and x.get("type") == "text")
    else:
        notes.append(("anthropic-doc-" + str(src.get("type")), f"Anthropic document source {src.get('type')!r} is not "
                       "supported (text and content sources are): the document is skipped"))
        return None
    return "\n\n".join(s for s in (str(b.get("title") or ""), str(b.get("context") or ""), text) if s)


def _skipped_block(t, notes, where="content"):
    notes.append((f"anthropic-block-{t}", f"Anthropic {where} block type {t!r} is not supported: skipped (as llama-server)"))


def _tool_result(content, notes):
    """a tool_result block's content as an OpenAI tool message's: a string, or parts when it has images"""
    if content is None or isinstance(content, str):
        return content or ""
    if not isinstance(content, list):
        raise BadRequest("tool_result content: a string or an array of blocks")
    parts = []
    for c in content:
        if not isinstance(c, dict):
            raise BadRequest("tool_result content: blocks must be objects")
        t = c.get("type")
        if t == "text":
            parts.append({"type": "text", "text": str(c.get("text") or "")})
        elif t == "image":
            parts.append({"type": "image_url", "image_url": {"url": _anthropic_image(c.get("source"))}})
        elif t == "document":
            text = _anthropic_document(c, notes)
            if text is not None:
                parts.append({"type": "text", "text": text})
        else:
            _skipped_block(t, notes, "tool_result content")
    if any(p["type"] == "image_url" for p in parts):
        return parts
    return "\n".join(p["text"] for p in parts)


def takes_mid_system(template):
    """whether a chat template renders a system message after the first message (Qwen's raise "System message must be
    at the beginning")"""
    try:
        return "MIDSYS" in template.render(messages=[{"role": "system", "content": "S"}, {"role": "user", "content": "U"},
                                                     {"role": "system", "content": "MIDSYS"}], add_generation_prompt=True)
    except Exception:
        return False


def anthropic_to_chat(body, notes=None, mid_system=True):
    """an Anthropic Messages request as the server's chat request. system: a string or text blocks (joined by blank
    lines); content blocks text / image / thinking (-> reasoning_content) / tool_use (-> tool_calls) / tool_result (->
    tool messages, before the rest of their message: they answer the previous turn's calls) / document (plain-text and
    content sources -> text); other block types are skipped, as llama-server does; tools input_schema -> parameters,
    any tool type (Anthropic's typed client and server tools too) as a function with its input_schema or none, as
    llama-server does; tool_choice auto / none / any / tool; stop_sequences -> stop; thinking enabled / adaptive ->
    thinking on, disabled or absent -> off (Anthropic's default; chat_template_kwargs win). notes: gets (key, message)
    for what was skipped or converted (the console's one-time warnings). mid_system False (the chat template takes a
    system message only at the start: takes_mid_system): a system message inside the conversation becomes a user
    message in its place"""
    notes = [] if notes is None else notes
    if not isinstance(body.get("messages"), list):
        raise BadRequest("messages: an array of messages is required")
    msgs = []
    system = body.get("system")
    if system is not None:
        if isinstance(system, str):
            texts = [system]
        elif isinstance(system, list) and all(isinstance(b, dict) and b.get("type") == "text" for b in system):
            texts = [str(b.get("text") or "") for b in system]
        else:
            raise BadRequest("system: a string or an array of text blocks")
        msgs.append({"role": "system", "content": "\n\n".join(BILLING_RE.sub(r"\1fffff;", t) for t in texts)})
    for m in body["messages"]:
        # (a system message inside the conversation: Claude Code sends some; the chat template decides whether it can
        # render one there, as llama-server leaves it to the template)
        if not isinstance(m, dict) or m.get("role") not in ("user", "assistant", "system"):
            raise BadRequest("messages: each message needs the role user, assistant or system")
        role, content = m["role"], m.get("content")
        if isinstance(content, str):
            msgs.append({"role": role, "content": content})
            continue
        if not isinstance(content, list):
            raise BadRequest("message content: a string or an array of content blocks")
        parts, calls, results, thinking = [], [], [], ""
        for b in content:
            if not isinstance(b, dict):
                raise BadRequest("message content: content blocks must be objects")
            t = b.get("type")
            if t == "text":
                parts.append({"type": "text", "text": str(b.get("text") or "")})
            elif t == "image":
                parts.append({"type": "image_url", "image_url": {"url": _anthropic_image(b.get("source"))}})
            elif t == "thinking":
                thinking += str(b.get("thinking") or "")
            elif t == "redacted_thinking":
                pass
            elif t == "tool_use":
                calls.append({"id": str(b.get("id") or ""), "type": "function", "function": {
                    "name": str(b.get("name") or ""), "arguments": json.dumps(b.get("input") or {}, ensure_ascii=False)}})
            elif t == "tool_result":
                results.append({"role": "tool", "tool_call_id": str(b.get("tool_use_id") or ""),
                                "content": _tool_result(b.get("content"), notes)})
            elif t == "document":
                text = _anthropic_document(b, notes)
                if text is not None:
                    parts.append({"type": "text", "text": text})
            else:  # (server tool blocks, search results, container uploads, ...)
                _skipped_block(t, notes)
        msgs += results
        if parts or calls or thinking:
            nm = {"role": role, "content": parts[0]["text"] if len(parts) == 1 and parts[0]["type"] == "text"
                  else (parts or "")}
            if calls:
                nm["tool_calls"] = calls
            if thinking:
                nm["reasoning_content"] = thinking
            msgs.append(nm)
    if body["messages"] and body["messages"][-1]["role"] == "assistant":
        notes.append(("anthropic-prefill", "an Anthropic request that ends with an assistant message (a prefill) is not "
                      "continued: the message is rendered as a finished turn and the reply starts a new one"))
    for k, why in IGNORED_FIELDS.items():
        v = body.get(k)
        if v is not None and not (k == "output_config" and isinstance(v, dict) and set(v) <= {"effort"}
                                  and v.get("effort") in EFFORT):
            notes.append((f"anthropic-field-{k}", f"Anthropic field {k!r} is ignored: {why}"))
    if not mid_system:  # (Claude Code sends its environment and skills list as a system message after the first one)
        for i, m in enumerate(msgs):
            if i and m["role"] == "system":
                m["role"] = "user"
                notes.append(("anthropic-mid-system", "Anthropic system messages inside the conversation are rendered as "
                              "user messages in their place: the chat template takes a system message only at the start"))
    out = {"messages": msgs, "stream": bool(body.get("stream"))}
    out["max_tokens"] = body.get("max_tokens") if body.get("max_tokens") is not None else 4096  # (llama-server's)
    for k in ("temperature", "top_p", "top_k"):
        if body.get(k) is not None:
            out[k] = body[k]
    if body.get("stop_sequences") is not None:
        out["stop"] = stop_strings(body["stop_sequences"], "stop_sequences")
    tools = body.get("tools")
    if tools:
        if not isinstance(tools, list):
            raise BadRequest("tools: an array of tools")
        out["tools"] = []
        for t in tools:
            if not isinstance(t, dict) or not t.get("name"):
                raise BadRequest(f"tools: each tool needs a name, not {str(t)[:80]!r}")
            if t.get("input_schema") is not None and not isinstance(t["input_schema"], dict):
                raise BadRequest(f"tools: {str(t['name'])[:80]}: input_schema must be an object (a JSON Schema)")
            if t.get("type") not in (None, "custom"):  # (e.g. bash_20250124, web_search_20250305)
                notes.append((f"anthropic-tool-{t['type']}", f"Anthropic tool type {t['type']!r} ({t['name']}) is offered "
                              f"to the model as a plain function{'' if t.get('input_schema') else ' without parameters'}"
                              " (as llama-server): a call to it comes back as a tool_use block for the client"))
            out["tools"].append({"type": "function", "function": {
                "name": str(t["name"]), "description": t.get("description") or "", "parameters": t.get("input_schema") or {}}})
    tc = body.get("tool_choice")
    if tc is not None:
        kind = tc.get("type") if isinstance(tc, dict) else None
        if kind in ("any", "tool"):
            raise BadRequest(f"tool_choice {kind}: forcing a tool call needs constrained sampling, which the megakernel "
                             "does not have (auto, none)")
        if kind not in ("auto", "none"):
            raise BadRequest("tool_choice: {\"type\": \"auto\"} or {\"type\": \"none\"}")
        out["tool_choice"] = kind
        if tc.get("disable_parallel_tool_use"):
            out["parallel_tool_calls"] = False
    if body.get("mcp_servers"):
        raise BadRequest("mcp_servers: not supported (the client runs its tools)")
    think = body.get("thinking")
    kw = dict(body.get("chat_template_kwargs") or {})
    if think is not None and not (isinstance(think, dict) and think.get("type") in ("enabled", "adaptive", "disabled")):
        raise BadRequest("thinking: {\"type\": \"enabled\" | \"adaptive\" | \"disabled\"}")
    kw.setdefault("enable_thinking", bool(think) and think["type"] != "disabled")
    oc = body.get("output_config")
    eff = EFFORT.get(oc.get("effort")) if isinstance(oc, dict) else None
    if eff and kw["enable_thinking"]:  # (chat_template_kwargs' reasoning_effort wins, as on the chat route)
        out["reasoning_effort"] = eff
    if think and think["type"] != "disabled" and think.get("budget_tokens") is not None:
        notes.append(("anthropic-thinking-budget", "Anthropic thinking.budget_tokens is not enforced: the reasoning ends "
                      "when the model ends it, within max_tokens"))
    out["chat_template_kwargs"] = kw
    return out


def anthropic_stop_reason(finish, stop_word):
    if finish == "length":
        return "max_tokens"
    if finish == "tool_calls":
        return "tool_use"
    return "stop_sequence" if stop_word is not None else "end_turn"


def anthropic_usage(usage):
    reused = usage["prompt_tokens_details"]["cached_tokens"]
    return {"input_tokens": usage["prompt_tokens"] - reused, "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": reused, "output_tokens": usage["completion_tokens"]}


def anthropic_tool_use(call):
    try:
        args = json.loads(call["function"]["arguments"])
    except ValueError:
        args = {}
    return {"type": "tool_use", "id": "toolu_" + call["id"].removeprefix("call_"), "name": call["function"]["name"],
            "input": args if isinstance(args, dict) else {}}


ANTHROPIC_ERRORS = {400: "invalid_request_error", 404: "not_found_error", 413: "request_too_large",
                    500: "api_error", 501: "api_error", 503: "overloaded_error"}
