from __future__ import annotations

import html
import json
import re
from dataclasses import dataclass, field
from typing import AsyncIterator

import httpx

from .runtime_limits import estimate_tokens
from .tools.registry import all_tool_names, runtime_tool_schemas

TOOLS = runtime_tool_schemas()
TOOL_NAMES = frozenset(all_tool_names())
TOOL_PARAM_TYPES = {
    str(function.get("name", "")).strip(): {
        str(param_name): str(spec.get("type", "")).strip()
        for param_name, spec in (
            (function.get("parameters", {}) if isinstance(function, dict) else {}).get("properties", {})
            if isinstance((function.get("parameters", {}) if isinstance(function, dict) else {}).get("properties", {}), dict)
            else {}
        ).items()
        if isinstance(spec, dict)
    }
    for tool in TOOLS
    for function in [tool.get("function", {}) if isinstance(tool, dict) else {}]
    if isinstance(function, dict) and str(function.get("name", "")).strip()
}


def tool_guidelines_xml() -> str:
    lines = [
        "<tool_guidelines>",
        "  <format><tool_call><function=tool_name><parameter=parameter_name>value</parameter></function></tool_call></format>",
        "  <tools>",
    ]
    for tool in TOOLS:
        function = tool.get("function", {}) if isinstance(tool, dict) else {}
        if not isinstance(function, dict):
            continue
        name = str(function.get("name", "")).strip()
        if not name:
            continue
        description = str(function.get("description", "")).strip()
        parameters = function.get("parameters", {})
        properties = parameters.get("properties", {}) if isinstance(parameters, dict) else {}
        required = set(parameters.get("required", []) if isinstance(parameters, dict) else [])
        params = []
        if isinstance(properties, dict):
            for param_name, spec in properties.items():
                param_type = str(spec.get("type", "")).strip() if isinstance(spec, dict) else ""
                required_suffix = "*" if param_name in required else ""
                params.append(
                    f'{html.escape(str(param_name))}{required_suffix}{f":{html.escape(param_type)}" if param_type else ""}'
                )
        lines.append(f'    <tool name="{html.escape(name)}" params="{html.escape(", ".join(params))}">')
        if description:
            lines.append(f"      {html.escape(description)}")
        lines.append("    </tool>")
    lines.extend(["  </tools>", "</tool_guidelines>"])
    return "\n".join(lines)


TOOL_GUIDELINES_XML = tool_guidelines_xml()


@dataclass
class ToolCall:
    name: str
    arguments: dict
    id: str | None = None


@dataclass
class StreamChunk:
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    done: bool = False
    reasoning: str = ""
    tool_args_delta: str = ""


_TOOL_CALL_BLOCK_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.IGNORECASE | re.DOTALL)
_TOOL_CALL_OPEN_RE = re.compile(r"<tool_call>\s*", re.IGNORECASE)
_FUNCTION_BLOCK_RE = re.compile(r"<function=([^>\n]+)>\s*(.*?)\s*</function>", re.IGNORECASE | re.DOTALL)
_PARAMETER_BLOCK_RE = re.compile(r"<parameter=([^>\n]+)>\s*(.*?)\s*</parameter>", re.IGNORECASE | re.DOTALL)
_TOOL_CALL_MARKER = "<tool_call"
_TOOL_CALL_MARKER_OVERLAP = len(_TOOL_CALL_MARKER) - 1


def tool_schema_token_estimate() -> int:
    return estimate_tokens(TOOL_GUIDELINES_XML)


def _messages_with_tool_guidelines(messages: list[dict]) -> list[dict]:
    if not messages:
        return [{"role": "system", "content": TOOL_GUIDELINES_XML}]
    first = messages[0]
    if isinstance(first, dict) and first.get("role") == "system":
        merged = dict(first)
        merged["content"] = f"{first.get('content', '')}\n\n{TOOL_GUIDELINES_XML}".strip()
        return [merged, *messages[1:]]
    return [{"role": "system", "content": TOOL_GUIDELINES_XML}, *messages]


def parse_tool_calls(text: str) -> list[ToolCall]:
    def coerce(tool_name: str, param_name: str, raw: str) -> object:
        value = html.unescape(raw.strip())
        param_type = TOOL_PARAM_TYPES.get(tool_name, {}).get(param_name, "")
        if param_type == "integer" and re.fullmatch(r"-?\d+", value):
            return int(value)
        if param_type == "boolean" and value.lower() in {"true", "false"}:
            return value.lower() == "true"
        return value

    tool_calls: list[ToolCall] = []
    if not text.strip():
        return tool_calls

    consumed_spans: list[tuple[int, int]] = []
    for match in _TOOL_CALL_BLOCK_RE.finditer(text):
        consumed_spans.append(match.span())
        block = match.group(1)
        function_match = _FUNCTION_BLOCK_RE.search(block)
        if function_match:
            name = function_match.group(1).strip()
            if not name or name not in TOOL_NAMES:
                continue
            arguments: dict[str, object] = {}
            for param_name, param_value in _PARAMETER_BLOCK_RE.findall(function_match.group(2)):
                key = param_name.strip()
                if not key:
                    continue
                arguments[key] = coerce(name, key, param_value)
            tool_calls.append(ToolCall(name=name, arguments=arguments))
            continue

        json_call = _parse_json_tool_call(block, TOOL_NAMES)
        if json_call is not None:
            tool_calls.append(json_call)

    # Unclosed <tool_call> ... (model truncated, missing </tool_call>): salvage trailing block.
    for open_match in _TOOL_CALL_OPEN_RE.finditer(text):
        start = open_match.start()
        if any(s <= start < e for s, e in consumed_spans):
            continue
        tail = text[open_match.end():]
        json_call = _parse_json_tool_call(tail, TOOL_NAMES)
        if json_call is not None:
            tool_calls.append(json_call)
    return tool_calls


def _parse_json_tool_call(block: str, tool_names: frozenset[str]) -> ToolCall | None:
    """Qwen-native fallback: <tool_call>{"name": ..., "arguments": {...}}</tool_call>."""
    stripped = block.strip()
    if not stripped:
        return None
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start < 0 or end <= start:
        return None
    candidate = stripped[start : end + 1]
    try:
        data = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    name = str(data.get("name", "")).strip()
    if not name or name not in tool_names:
        return None
    raw_args = data.get("arguments", data.get("parameters", {}))
    if isinstance(raw_args, str):
        try:
            raw_args = json.loads(raw_args)
        except json.JSONDecodeError:
            raw_args = {}
    if not isinstance(raw_args, dict):
        raw_args = {}
    return ToolCall(name=name, arguments=raw_args)


async def stream_openai_chat(
    http: httpx.AsyncClient,
    *,
    base_url: str,
    model: str,
    messages: list[dict],
    use_tools: bool = True,
    headers: dict[str, str] | None = None,
    extra_body: dict | None = None,
) -> AsyncIterator[StreamChunk]:
    payload: dict = {
        "model": model,
        "messages": messages,
        "stream": True,
    }
    if use_tools:
        payload["messages"] = _messages_with_tool_guidelines(messages)
    if extra_body:
        payload.update(extra_body)

    reasoning_buf = ""
    content_buf = ""
    text_emit_buffer = ""
    saw_content_tool_markup = False
    structured_tool_args: dict[int, str] = {}
    structured_tool_names: dict[int, str] = {}
    structured_tool_ids: dict[int, str] = {}

    async with http.stream(
        "POST",
        f"{base_url}/v1/chat/completions",
        json=payload,
        headers=headers,
    ) as resp:
        try:
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            detail = ""
            try:
                body = await resp.aread()
                text = body.decode("utf-8", errors="replace").strip()
                if text:
                    detail = f": {text[:500]}"
            except Exception:
                detail = ""
            raise RuntimeError(f"{exc}{detail}") from exc

        async for line in resp.aiter_lines():
            line = line.strip()
            if not line or line == "data: [DONE]":
                continue
            if line.startswith("data: "):
                line = line[6:]
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            choices = data.get("choices", [])
            if not choices:
                continue

            choice = choices[0]
            delta = choice.get("delta", {}) or {}
            finish_reason = choice.get("finish_reason")

            text = delta.get("content", "") or ""
            if text:
                content_buf += text
                if not use_tools:
                    yield StreamChunk(text=text)
                elif saw_content_tool_markup:
                    yield StreamChunk(tool_args_delta=text)
                else:
                    text_emit_buffer += text
                    marker_index = text_emit_buffer.lower().find(_TOOL_CALL_MARKER)
                    if marker_index >= 0:
                        visible = text_emit_buffer[:marker_index]
                        if visible:
                            yield StreamChunk(text=visible)
                        hidden = text_emit_buffer[marker_index:]
                        if hidden:
                            yield StreamChunk(tool_args_delta=hidden)
                        text_emit_buffer = ""
                        saw_content_tool_markup = True
                    elif len(text_emit_buffer) > _TOOL_CALL_MARKER_OVERLAP:
                        visible = text_emit_buffer[:-_TOOL_CALL_MARKER_OVERLAP]
                        text_emit_buffer = text_emit_buffer[-_TOOL_CALL_MARKER_OVERLAP:]
                        if visible:
                            yield StreamChunk(text=visible)

            reasoning_text = delta.get("reasoning_content", "") or ""
            if reasoning_text:
                reasoning_buf += reasoning_text
                yield StreamChunk(reasoning=reasoning_text)

            for tc in delta.get("tool_calls", []) or []:
                if not isinstance(tc, dict):
                    continue
                idx = int(tc.get("index", 0))
                tc_id = tc.get("id")
                if isinstance(tc_id, str) and tc_id:
                    structured_tool_ids[idx] = tc_id
                fn = tc.get("function", {}) or {}
                if not isinstance(fn, dict):
                    continue
                name_part = fn.get("name")
                if isinstance(name_part, str) and name_part:
                    structured_tool_names[idx] = name_part
                args_part = fn.get("arguments")
                if isinstance(args_part, str) and args_part:
                    structured_tool_args[idx] = structured_tool_args.get(idx, "") + args_part

            if finish_reason is None:
                continue

            if use_tools and not saw_content_tool_markup and text_emit_buffer:
                yield StreamChunk(text=text_emit_buffer)
                text_emit_buffer = ""

            tool_calls = parse_tool_calls(f"{content_buf}\n{reasoning_buf}") if use_tools else []
            for idx in sorted(structured_tool_names):
                name = structured_tool_names.get(idx, "").strip()
                if not name or name not in TOOL_NAMES:
                    continue
                args_raw = structured_tool_args.get(idx, "")
                if args_raw:
                    try:
                        args = json.loads(args_raw)
                    except json.JSONDecodeError:
                        args = {"raw": args_raw}
                else:
                    args = {}
                if not isinstance(args, dict):
                    args = {"value": args}
                tool_calls.append(ToolCall(name=name, arguments=args, id=structured_tool_ids.get(idx)))

            yield StreamChunk(tool_calls=tool_calls, done=True)
            return
