"""The chat request and reply: message normalizing, parameter checks, tool-call and output parsing."""
import json
import math
import re
import uuid

from mk_errors import BadRequest
from mk_limits import MAX_STOP, MAX_STOP_LEN
from mk_media import VIDEO_PARTS, image_bytes, video_bytes

MEDIA_MARKER = "<__media__>"  # llama-server's (mtmd_default_marker)


def normalize_messages(messages, media=None, notes=None):
    """media (a list: the server has a vision encoder): image_url parts and video parts (input_video / video_url)
    become MEDIA_MARKER and their file bytes are appended to media in order (videos as VideoBytes). A part list is
    flattened to one string as llama-server does (text parts joined by newlines, no newline next to an image or
    video), so the prompt around each is llama-server's; text parts are not run together. notes: gets (key, message)
    for what was converted (the console's one-time warnings)"""
    notes = [] if notes is None else notes
    out = []
    for m in messages:
        if not isinstance(m, dict):
            raise BadRequest("messages: each message must be an object")
        m = dict(m)
        content = m.get("content")
        if isinstance(content, list):
            for part in content:
                if not isinstance(part, dict):
                    raise BadRequest("messages: each content part must be an object")
                if part.get("type") not in (None, "text", "image_url", *VIDEO_PARTS):
                    raise BadRequest(f"unsupported content part type {part.get('type')!r} (text, image_url, "
                                     f"input_video and video_url only)")
            kinds = {p.get("type") for p in content}
            if kinds & {"image_url", *VIDEO_PARTS} and media is None:
                what = "image" if "image_url" in kinds else "video"
                raise BadRequest(f"{what} input is not supported: start the server with --mmproj")
            text, after_media = "", False
            for part in content:
                if part.get("type") in VIDEO_PARTS:
                    media.append(video_bytes(part))
                    text += MEDIA_MARKER
                    after_media = True
                elif part.get("type") == "image_url":
                    iu = part.get("image_url")
                    media.append(image_bytes(iu.get("url") if isinstance(iu, dict) else iu))
                    text += MEDIA_MARKER
                    after_media = True
                else:
                    if not after_media and text:
                        text += "\n"
                    text += str(part.get("text") or "")
                    after_media = False
            m["content"] = text
        if content is None:
            m["content"] = ""
        if m.get("role") == "assistant" and m.get("tool_calls"):
            calls = []
            if not isinstance(m["tool_calls"], list):
                raise BadRequest("messages: tool_calls must be an array")
            for tc in m["tool_calls"]:
                if not isinstance(tc, dict) or not isinstance(tc.get("function") or {}, dict):
                    raise BadRequest("messages: each of tool_calls must be an object, with an object as its function")
                tc = dict(tc)
                fn = dict(tc.get("function") or {})
                if fn.get("name") is not None and not isinstance(fn["name"], str):  # (none: the template's own 400)
                    raise BadRequest("messages: a tool call's function name must be a string")
                args = given = fn.get("arguments")
                if isinstance(args, str):
                    try:
                        args = json.loads(args) if args.strip() else {}
                    except (ValueError, RecursionError):  # (not JSON, an integer past the digit limit, nested too deep)
                        args = None
                if not isinstance(args, dict) and given is not None:
                    notes.append(("tool-call-arguments", f"a tool call's arguments in a request's history are not a JSON "
                                  f"object ({str(given)[:40]!r}...): the call is rendered without arguments"))
                fn["arguments"] = args if isinstance(args, dict) else {}
                tc["function"] = fn
                calls.append(tc)
            m["tool_calls"] = calls
        out.append(m)
    return out


# Qwen's XML tool calls: <tool_call>\n<function=NAME>\n<parameter=KEY>\nVALUE\n</parameter>\n...</function>\n</tool_call>.
# A call counts only with both closing tags: one cut by max_tokens (or left open) is not a call. A value runs to the next
# </parameter> (as llama.cpp's parser reads it), so it may contain any other markup, <tool_call> included.
TOOL_OPEN = "<tool_call>"
TOOL_HEAD_RE = re.compile(r"<tool_call>\s*<function=([^>\n]+)>")
TOOL_END_RE = re.compile(r"</function>\s*</tool_call>")
PARAM_HEAD_RE = re.compile(r"<parameter=([^>\n]+)>")
TAG_NAME_RE = re.compile(r"[^>\n]*")  # (a name that may still be completing: nothing that ends it yet)
SPACE_RE = re.compile(r"\s*")


class ToolCallParser:
    """Qwen's tool calls in the generated text (from its first <tool_call>), parsed as the text arrives. `calls`: the
    complete calls in OpenAI's shape. With events, feed() also returns each call as it is generated: ("start", index,
    id, name) once its function name is complete, then ("args", index, fragment) pieces of its arguments' JSON text: a
    string parameter's value as it is generated (JSON-escaped), any other value whole at its end (it is decoded as JSON
    there), the closing brace with the call's closing tags, so the arguments become valid JSON only when the call is
    complete. A complete call's fragments add up to its arguments byte for byte (a parameter given twice: both values
    are sent, the arguments hold the last one, as a JSON parser reads the sent text). A parameter is a string when
    its schema type is "string", or it has no type and does not start like a JSON value.
    one (parallel_tool_calls false): no call is started in the stream after a complete one. Decisions do not depend on
    how the text is split into pieces: a tag that may still be completing waits for more text."""

    def __init__(self, tools, events=False, one=False):
        self.schemas = {}
        for t in tools or []:
            fn = t.get("function", t) if isinstance(t, dict) else {}
            props = (fn.get("parameters") or {}).get("properties") if isinstance(fn, dict) else None
            self.schemas[fn.get("name") if isinstance(fn, dict) else None] = props if isinstance(props, dict) else {}
        self.events, self.one = events, one
        self.text, self.pos, self.state = "", 0, "seek"
        self.calls = []
        self.spans = []  # (start, end) of each complete call in text
        self.end = 0  # where the last complete call ends in text
        self.started = 0  # calls started in the stream
        self.open = []  # where the calls started in the stream that did not complete begin
        self.cur = None  # the call being read

    def feed(self, text, final=False):
        """the stream events of text (all of them by the final call)"""
        self.text += text
        out = []
        step = {"seek": self._seek, "head": self._head, "between": self._between, "value": self._value}
        while self.state != "done" and step[self.state](final, out):
            pass
        if final:
            self.state = "done"
        return out

    def unparsed(self, streamed=False):
        """the tool-call markup after the last complete call that is not a call itself (cut by max_tokens, left open or
        malformed), from its <tool_call> to the end; empty if there is none. streamed: only the part before a call that
        was started in the stream (the client has that one as a call already)"""
        u = self.text.find(TOOL_OPEN, self.end)
        if u < 0:
            return ""
        cut = min((s for s in self.open if s >= u), default=len(self.text)) if streamed else len(self.text)
        return self.text[u:cut]

    def stray(self):
        """the text no reply carries: what lies outside the complete calls before the unparsed markup (between the calls,
        after the last one), whitespace-only stretches aside"""
        u = self.text.find(TOOL_OPEN, self.end)
        bits, pos = [], 0
        for s, e in self.spans:
            bits.append(self.text[pos:s])
            pos = e
        bits.append(self.text[pos:u if u >= 0 else len(self.text)])
        return [b.strip() for b in bits if b.strip()]

    def _seek(self, final, out):
        i = self.text.find(TOOL_OPEN, self.pos)
        if i < 0:
            self.pos = max(self.pos, len(self.text) - len(TOOL_OPEN) + 1)
            return False
        self.start, self.pos, self.state = i, i, "head"
        return True

    def _head(self, final, out):
        t = self.text
        m = TOOL_HEAD_RE.match(t, self.start)
        if m is None:
            j = SPACE_RE.match(t, self.start + len(TOOL_OPEN)).end()
            if not final and ("<function=".startswith(t[j:]) if len(t) - j < 10 else
                              t.startswith("<function=", j) and TAG_NAME_RE.fullmatch(t, j + 10)):
                return False  # (it may still become a call)
            self.pos, self.state = self.start + len(TOOL_OPEN), "seek"  # (not a call)
            return True
        name = m.group(1).strip()
        props = self.schemas.get(name, {})
        self.cur = c = {"start": self.start, "name": name, "id": "call_" + uuid.uuid4().hex[:24], "props": props,
                        "args": {}, "n": 0, "index": None}
        if self.events and not (self.one and self.calls):
            c["index"], self.started = self.started, self.started + 1
            out.append(("start", c["index"], c["id"], name))
        self.pos, self.state = m.end(), "between"
        return True

    def _between(self, final, out):
        """after the function name or a parameter: the next parameter or the closing tags (other text is skipped, a new
        <tool_call> makes this one not a call)"""
        t, c = self.text, self.cur
        hits = [(j, tag) for tag in ("<parameter=", "</function>", TOOL_OPEN) for j in (t.find(tag, self.pos),) if j >= 0]
        if not hits:
            if final:
                return self._drop()  # (left open)
            self.pos = max(self.pos, len(t) - 10)
            return False
        j, tag = min(hits)
        if tag == TOOL_OPEN:  # (a new call before this one's closing tags: this one was left open)
            self._drop()
            self.pos, self.state = j, "seek"
            return True
        if tag == "</function>":
            m = TOOL_END_RE.match(t, j)
            if m is not None:
                self._arg("}" if c["n"] else "{}", out)
                self.calls.append({"id": c["id"], "type": "function", "function": {
                    "name": c["name"], "arguments": json.dumps(c["args"], ensure_ascii=False)}})
                self.spans.append((c["start"], m.end()))
                self.cur, self.end, self.pos, self.state = None, m.end(), m.end(), "seek"
                return True
            k = SPACE_RE.match(t, j + 11).end()
            if not final and len(t) - k < 12 and "</tool_call>".startswith(t[k:]):
                self.pos = j
                return False
            self.pos = j + 11
            return True
        m = PARAM_HEAD_RE.match(t, j)
        if m is None:
            if not final and TAG_NAME_RE.fullmatch(t, j + 11):
                self.pos = j
                return False
            self.pos = j + 11
            return True
        vs = m.end()
        if vs == len(t):  # (a newline right after the tag is not part of the value)
            if final:
                return self._drop()
            self.pos = j
            return False
        vs += t[vs] == "\n"
        key = m.group(1).strip()
        spec = c["props"].get(key)
        typ = spec.get("type") if isinstance(spec, dict) else None
        c.update(key=key, vstart=vs, vscan=vs, vsent=vs, kind="string" if typ == "string" else None if typ is None else "json")
        self.state = "value"
        return True

    def _value(self, final, out):
        t, c = self.text, self.cur
        k = t.find("</parameter>", c["vscan"])
        if k < 0:
            if final:  # (a stream gets the string's text to the end, the call stays incomplete)
                self._send_value(len(t), False, out)
                return self._drop()
            c["vscan"] = max(c["vstart"], len(t) - 11)
            self._send_value(max(c["vstart"], len(t) - 12), False, out)  # (a newline and the closing tag may follow)
            return False
        end = k - 1 if k > c["vstart"] and t[k - 1] == "\n" else k
        self._send_value(end, True, out)
        self.pos, self.state = k + len("</parameter>"), "between"
        return True

    def _send_value(self, upto, done, out):
        """the value's text up to `upto` (done: its end): a string's as it comes, another value's at its end"""
        t, c = self.text, self.cur
        if c["kind"] is None:  # (no type in the schema: decided by its first character)
            head = t[c["vstart"]:upto].strip()[:1]
            if not head and not done:
                return
            c["kind"] = "json" if head in "[{0123456789-tfn" else "string"
        key = json.dumps(c["key"], ensure_ascii=False)
        if c["kind"] == "string":
            if c["vsent"] == c["vstart"] and not c.get("opened"):
                c["opened"] = True
                self._arg((", " if c["n"] else "{") + key + ': "', out)
                c["n"] += 1
            if upto > c["vsent"]:
                self._arg(json.dumps(t[c["vsent"]:upto], ensure_ascii=False)[1:-1], out)
                c["vsent"] = upto
            if done:
                self._arg('"', out)
                c["args"][c["key"]] = t[c["vstart"]:upto]
                c["opened"] = False
            return
        if not done:
            return
        val = t[c["vstart"]:upto]
        try:
            val = json.loads(val)
        except (ValueError, RecursionError):  # (not JSON: the text itself)
            pass
        self._arg((", " if c["n"] else "{") + key + ": " + json.dumps(val, ensure_ascii=False), out)
        c["n"] += 1
        c["args"][c["key"]] = val

    def _arg(self, frag, out):
        if self.cur["index"] is not None:
            out.append(("args", self.cur["index"], frag))

    def _drop(self):
        """the call being read is not a call: a new <tool_call> before its closing tags, or (returns False: the end)
        left open at the end of the text, whose unclosed value or tag may hold anything (no call is looked for in it)"""
        if self.cur["index"] is not None:
            self.open.append(self.cur["start"])
        self.cur, self.state = None, "seek"
        return False


def unparsed_tool_call(text):
    """the tool-call markup after the last complete call that is not a call itself (ToolCallParser.unparsed)"""
    p = ToolCallParser([])
    p.feed(text, final=True)
    return p.unparsed()


def parse_tool_calls(text, tools):
    """the complete tool calls in text (ToolCallParser)"""
    p = ToolCallParser(tools)
    p.feed(text, final=True)
    return p.calls


def _off(limit, above=False):
    """a check that a sampler field has its 'off' value: None, or <= limit (above: >= limit)"""
    def ok(v):
        if v is None:
            return True
        try:
            return float(v) >= limit if above else float(v) <= limit
        except (TypeError, ValueError, OverflowError):
            return False
    return ok


def finite(x):
    """the number x is finite as a float (an integer past a float's range is not)"""
    try:
        return math.isfinite(x)
    except OverflowError:
        return False


# llama.cpp sampler fields the engine does not have (it samples with temperature, top_k, top_p, min_p and the presence /
# frequency penalties): accepted only with the value that turns them off (llama.cpp's web UI sends them all)
SAMPLER_OFF = {"mirostat": _off(0), "typical_p": _off(1, True), "typ_p": _off(1, True), "tfs_z": _off(1, True),
               "xtc_probability": _off(0), "dry_multiplier": _off(0), "top_n_sigma": _off(0), "dynatemp_range": _off(0),
               "min_tokens": _off(0), "n_probs": _off(0)}


# llama-server's --reasoning-format values: auto = deepseek (reasoning_content, streamed too: the default),
# deepseek-legacy (the content keeps the thinking block too; a stream only there), none (the thinking stays in the content)
REASONING_FORMATS = ("auto", "deepseek", "deepseek-legacy", "none")
# the chat_template_kwargs the server hands to the template
TEMPLATE_KWARGS = ("enable_thinking", "preserve_thinking", "reasoning_effort")
SOFT_SWITCH = re.compile(r"(/no_think|/think|<think>\s*</think>)\s*$")


def soft_switch(messages):
    """Qwen3's soft switches as a client sends them (Qwen3.8's template ignores them, so the server applies them): the
    last user message's text, else the leading system / developer message's, ending in /no_think (or an empty
    <think></think> block, as after the last user text or as a trailing assistant message) -> False, in /think -> True;
    None: no switch. The prompt keeps the text as sent."""
    def text(m):
        c = m.get("content") if isinstance(m, dict) else None
        if isinstance(c, list):  # (parts: the last text one)
            c = next((p["text"] for p in reversed(c) if isinstance(p, dict) and isinstance(p.get("text"), str)), None)
        return c if isinstance(c, str) else ""

    def switch(s):
        m = SOFT_SWITCH.search(s)
        return None if m is None else m.group(1) == "/think"
    msgs = [m for m in messages if isinstance(m, dict)]
    if msgs and msgs[-1].get("role") == "assistant" and re.fullmatch(r"\s*<think>\s*</think>\s*", text(msgs[-1])):
        return False
    user = next((m for m in reversed(msgs) if m.get("role") == "user"), None)
    s = switch(text(user)) if user is not None else None
    if s is None and msgs and msgs[0].get("role") in ("system", "developer"):
        s = switch(text(msgs[0]))
    return s


# llama.cpp's default sampler chain (its `samplers` field: names, or a string of their letters); the engine applies
# penalties, top_k, top_p, min_p and temperature in this order
SAMPLER_CHAIN = ("penalties", "dry", "top_n_sigma", "top_k", "typ_p", "top_p", "min_p", "xtc", "temperature")
SAMPLER_NAMES = {**{a: n for n in SAMPLER_CHAIN + ("infill", "adaptive_p") for a in (n, n.replace("_", "-"), n.replace("_", ""))},
                 **dict(zip("edskypmxti", SAMPLER_CHAIN + ("infill",))), "nucleus": "top_p", "temp": "temperature",
                 "typ": "typ_p"}


def check_samplers(v, penalties):
    """a request's `samplers` is accepted when it asks for what the engine does: top_k, top_p, min_p and temperature in
    llama.cpp's order (the other samplers in their default places: their parameters are checked to be off), penalties
    too when a penalty is set; unknown names are dropped, as llama.cpp does"""
    if isinstance(v, str):
        names = [SAMPLER_NAMES.get(c) for c in v]
    elif isinstance(v, list) and all(isinstance(n, str) for n in v):
        names = [SAMPLER_NAMES.get(n.lower()) for n in v]
    else:
        raise BadRequest("samplers: an array of sampler names or a string of sampler letters")
    names = [n for n in names if n]
    order = [SAMPLER_CHAIN.index(n) if n in SAMPLER_CHAIN else -1 for n in names]
    need = {"top_k", "top_p", "min_p", "temperature"} | ({"penalties"} if penalties else set())
    if min(order, default=0) < 0 or order != sorted(set(order)) or not need <= set(names):
        raise BadRequest(f"samplers: {v!r} is not supported: the megakernel samples with "
                         f"{', '.join(n for n in SAMPLER_CHAIN if n in need)} in llama.cpp's order (its default chain)")


def check_params(body):
    """400 for the request fields the server cannot honour (they would change the output and need engine work, e.g.
    logprobs, n > 1, constrained sampling): never ignored silently. Harmless ones (user, metadata, store, cache_prompt,
    llama-server's n_keep / n_discard (no context shift), response_fields / return_tokens (its /completion only),
    n_cache_reuse, sse_ping_interval, verbose) are accepted; samplers only in the order the engine samples."""
    def bad(field, why):
        raise BadRequest(f"{field}: {why}")
    if not isinstance(body.get("messages"), list):
        bad("messages", "an array of messages is required")
    for k in ("n", "n_cmpl"):
        if body.get(k) not in (None, 1):
            bad(k, "one choice per request (the engine runs one sequence): send several requests instead")
    if body.get("id_slot") not in (None, -1, 0):
        bad("id_slot", "the server has one slot, 0 (-1: any)")
    if body.get("n_indent") not in (None, 0):
        bad("n_indent", "not supported (llama-server's infill option: it ends a reply on indentation)")
    t = body.get("t_max_predict_ms")
    if t is not None and (isinstance(t, bool) or not isinstance(t, (int, float)) or not finite(t)):
        bad("t_max_predict_ms", "a number of milliseconds (<= 0: no limit)")
    if body.get("samplers") is not None:
        try:
            pen = any(float(body.get(k) or 0) for k in ("presence_penalty", "frequency_penalty"))
        except (TypeError, ValueError, OverflowError):
            pen = False  # (sampling() refuses it)
        check_samplers(body["samplers"], pen)
    if body.get("logprobs") or body.get("top_logprobs") not in (None, 0):
        bad("logprobs", "token log-probabilities are not available from the megakernel (they need engine support)")
    if body.get("logit_bias"):
        bad("logit_bias", "not supported by the megakernel's sampler")
    rf = body.get("response_format")
    if rf is not None and not (isinstance(rf, dict) and rf.get("type", "text") == "text"):
        bad("response_format", "only {\"type\": \"text\"}: JSON modes need constrained sampling (a grammar), which the "
                               "megakernel does not have")
    for k in ("json_schema", "grammar"):
        if body.get(k):
            bad(k, "constrained sampling (a grammar) is not supported by the megakernel")
    tc = body.get("tool_choice")
    if tc not in (None, "auto", "none"):
        bad("tool_choice", "\"auto\" or \"none\": forcing a tool call (required / a named function) needs constrained "
                           "sampling, which the megakernel does not have")
    if body.get("tools") is not None and not isinstance(body["tools"], list):
        bad("tools", "an array of tools")
    for t in body.get("tools") or []:  # (as ToolCallParser reads them: a function object, or the tool itself)
        fn = t.get("function", t) if isinstance(t, dict) else None
        if not isinstance(fn, dict):
            bad("tools", f"each tool must be an object with an object as its function, not {str(t)[:80]!r}")
        if not isinstance(fn.get("name"), str) or not fn["name"]:
            bad("tools", f"each tool's function needs a name (a non-empty string), not {str(fn.get('name'))[:80]!r}")
        if fn.get("parameters") is not None and not isinstance(fn["parameters"], dict):
            bad("tools", f"{fn['name'][:80]}: parameters must be an object (a JSON Schema), not "
                         f"{str(fn['parameters'])[:80]!r}")
    if body.get("functions") or body.get("function_call"):
        bad("functions", "the legacy function-calling fields are not supported: use tools")
    if body.get("ignore_eos"):
        bad("ignore_eos", "not supported by the megakernel")
    if body.get("audio") or "audio" in (body.get("modalities") or []):
        bad("audio", "audio output is not supported")
    if body.get("add_generation_prompt") is False or body.get("continue_final_message"):
        bad("add_generation_prompt", "continuing the final message is not supported")
    for k, ok in SAMPLER_OFF.items():
        if not ok(body.get(k)):
            bad(k, f"{body.get(k)!r} is not supported: the megakernel samples with temperature, top_k, top_p, min_p "
                   "and the presence / frequency penalties only")
    if body.get("reasoning_format") is not None and body["reasoning_format"] not in REASONING_FORMATS:
        bad("reasoning_format", f"one of {', '.join(REASONING_FORMATS)}")
    if body.get("chat_template_kwargs") is not None and not isinstance(body["chat_template_kwargs"], dict):
        bad("chat_template_kwargs", "an object")


def stop_strings(v, field="stop"):
    """the stop strings of a request: a string or a list of strings (empty ones dropped, as llama-server does)"""
    if v is None:
        return []
    if isinstance(v, str):
        v = [v]
    if not isinstance(v, list) or not all(isinstance(s, str) for s in v):
        raise BadRequest(f"{field} must be a string or a list of strings")
    v = [s for s in v if s]
    if len(v) > MAX_STOP or any(len(s) > MAX_STOP_LEN for s in v):
        raise BadRequest(f"{field}: at most {MAX_STOP} strings of at most {MAX_STOP_LEN} characters")
    return v


class StopMatcher:
    """Cuts the generated text at the first occurrence of a stop string, as llama-server does: on the raw text (reasoning,
    content and tool-call markup alike), the stop string and everything after it dropped. Text that could still become
    the start of a stop string is held back until the next piece decides it (or the end releases it)."""

    def __init__(self, stops):
        self.stops = stops
        self.held = ""
        self.word = None  # the stop string found

    def _partial(self, buf):
        """length of the longest suffix of buf that is a proper prefix of a stop string"""
        best = 0
        for s in self.stops:
            j = buf.find(s[0], max(0, len(buf) - len(s) + 1))  # (scanned from the longest candidate)
            while 0 <= j and len(buf) - j > best:
                if s.startswith(buf[j:]):
                    best = len(buf) - j
                    break
                j = buf.find(s[0], j + 1)
        return best

    def feed(self, text, final=False):
        """the text that can go out now; after a match (self.word set) nothing more"""
        if not self.stops or self.word is not None:
            return text if self.word is None else ""
        buf = self.held + text
        # (a match cannot start in text released before: its start would have been held)
        at = -1
        for s in self.stops:
            i = buf.find(s)
            if i >= 0 and (at < 0 or i < at):
                at, self.word = i, s
        if at >= 0:
            self.held = ""
            return buf[:at]
        keep = 0 if final else self._partial(buf)
        self.held = buf[len(buf) - keep:] if keep else ""
        return buf[:len(buf) - keep]


class OutputParser:
    """Splits the generated text into reasoning (before </think>), content and tool calls (from the first <tool_call>:
    ToolCallParser). calls: the tool calls' stream events too, as ("tool_call", delta) with delta in the shape of an
    OpenAI stream's delta.tool_calls[] entry; one: parallel_tool_calls false. inline (reasoning_format): "only" (none):
    the thinking block stays in the content as generated, with the <think> the prompt opened it with; "both"
    (deepseek-legacy): there and as reasoning."""

    def __init__(self, thinking, tools, calls=False, one=False, inline=None):
        self.mode = "reasoning" if thinking else "content_start"
        self.inline, self.opened = inline if thinking else None, False
        self.tools = tools
        self.tc = ToolCallParser(tools, calls, one) if tools else None
        self.buf = ""
        self.tool_text = ""
        self.tool_sep = ""  # the whitespace between the content and the first <tool_call>
        self.content_seen = False

    @property
    def calls(self):
        """the complete tool calls (OpenAI's shape)"""
        return self.tc.calls if self.tc else []

    def feed(self, text, final=False):
        """returns [(kind, text)] with kind in {reasoning, content}, and ("tool_call", delta) (calls)"""
        out = []
        self.buf += text
        while True:
            if self.mode == "reasoning":
                i = self.buf.find("</think>")
                if i >= 0:
                    self._think(out, self.buf[:i], True)
                    self.buf = self.buf[i + len("</think>"):]
                    self.mode = "content" if self.inline else "content_start"  # (inline: the text as generated)
                    continue
                keep = 0 if final else len("</think>") - 1
                if len(self.buf) > keep:
                    self._think(out, self.buf[:len(self.buf) - keep], False)
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
                        head = self.buf[:i].rstrip()
                        out.append(("content", head))
                        self.tool_sep = self.buf[len(head):i]
                        self.buf = self.buf[i:]
                        self.mode = "tools"
                        continue
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
            # tools: everything from the first <tool_call>
            self.tool_text += self.buf
            for ev in self.tc.feed(self.buf, final):
                if ev[0] == "start":
                    out.append(("tool_call", {"index": ev[1], "id": ev[2], "type": "function",
                                              "function": {"name": ev[3], "arguments": ""}}))
                else:
                    out.append(("tool_call", {"index": ev[1], "function": {"arguments": ev[2]}}))
            self.buf = ""
            break
        out = [(k, t) for k, t in out if t]
        self.content_seen = self.content_seen or any(k == "content" for k, _ in out)
        return out

    def _think(self, out, text, end):
        """reasoning text (end: up to its </think>)"""
        if self.inline != "only":
            out.append(("reasoning", text))
        if self.inline:
            out.append(("content", ("" if self.opened else "<think>\n") + text + ("</think>" if end else "")))
            self.opened = True

    def unparsed(self, streamed=False):
        """the tool-call markup that is not a complete call (ToolCallParser.unparsed), as the rest of the content: the
        client gets the text, as from llama-server, instead of a call it would execute half-written. streamed: without
        a call the stream started (its client has it as a call whose arguments never became valid JSON)"""
        rest = self.tc.unparsed(streamed) if self.tc else ""
        return (self.tool_sep if self.content_seen else "") + rest if rest else ""

    def stray(self):
        """the text between or after the tool calls that no reply carries (ToolCallParser.stray)"""
        return self.tc.stray() if self.tc else []
