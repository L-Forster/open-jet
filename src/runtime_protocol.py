from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import AsyncIterator

import httpx

from .runtime_limits import estimate_tokens
from .tools.registry import all_tool_names, runtime_tool_schemas

TOOLS = runtime_tool_schemas()
TOOL_NAMES = frozenset(all_tool_names())


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


_TOOL_CALL_BLOCK_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.IGNORECASE | re.DOTALL)
_FUNCTION_BLOCK_RE = re.compile(r"<function=([^>\n]+)>\s*(.*?)\s*</function>", re.IGNORECASE | re.DOTALL)
_PARAMETER_BLOCK_RE = re.compile(r"<parameter=([^>\n]+)>\s*(.*?)\s*</parameter>", re.IGNORECASE | re.DOTALL)
_TOOL_CALL_MARKER = "<tool_call"
_TOOL_CALL_MARKER_OVERLAP = len(_TOOL_CALL_MARKER) - 1


def tool_schema_token_estimate() -> int:
    payload = json.dumps(TOOLS, ensure_ascii=True, separators=(",", ":"))
    return estimate_tokens(payload)


def _coerce_reasoning_parameter(raw: str) -> object:
    value = raw.strip()
    if not value:
        return ""
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        pass
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered == "null":
        return None
    if re.fullmatch(r"-?\d+", value):
        try:
            return int(value)
        except ValueError:
            return value
    if re.fullmatch(r"-?\d+\.\d+", value):
        try:
            return float(value)
        except ValueError:
            return value
    return value


def _extract_xml_tool_calls(text: str) -> list[ToolCall]:
    tool_calls: list[ToolCall] = []
    if not text.strip():
        return tool_calls

    for block in _TOOL_CALL_BLOCK_RE.findall(text):
        match = _FUNCTION_BLOCK_RE.search(block)
        if not match:
            continue
        name = match.group(1).strip()
        if not name or name not in TOOL_NAMES:
            continue
        arguments: dict[str, object] = {}
        for param_name, param_value in _PARAMETER_BLOCK_RE.findall(match.group(2)):
            key = param_name.strip()
            if not key:
                continue
            arguments[key] = _coerce_reasoning_parameter(param_value)
        tool_calls.append(ToolCall(name=name, arguments=arguments))
    return tool_calls


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
        payload["tools"] = TOOLS
    if extra_body:
        payload.update(extra_body)

    tool_id_by_index: dict[int, str] = {}
    tool_name_by_index: dict[int, str] = {}
    tool_args_by_index: dict[int, str] = {}
    reasoning_buf = ""
    content_buf = ""
    text_emit_buffer = ""
    saw_content_tool_markup = False

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
                    pass
                else:
                    text_emit_buffer += text
                    marker_index = text_emit_buffer.lower().find(_TOOL_CALL_MARKER)
                    if marker_index >= 0:
                        visible = text_emit_buffer[:marker_index]
                        if visible:
                            yield StreamChunk(text=visible)
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
                idx = int(tc.get("index", 0))
                tc_id = tc.get("id", "")
                if tc_id:
                    tool_id_by_index[idx] = tc_id
                fn = tc.get("function", {}) or {}
                name_part = fn.get("name", "")
                args_part = fn.get("arguments", "")
                if name_part:
                    tool_name_by_index[idx] = name_part
                if isinstance(args_part, str) and args_part:
                    tool_args_by_index[idx] = tool_args_by_index.get(idx, "") + args_part

            if finish_reason is None:
                continue

            if use_tools and not saw_content_tool_markup and text_emit_buffer:
                yield StreamChunk(text=text_emit_buffer)
                text_emit_buffer = ""

            tool_calls: list[ToolCall] = []
            for idx in sorted(tool_name_by_index):
                name = tool_name_by_index.get(idx, "")
                if not name:
                    continue
                args_raw = tool_args_by_index.get(idx, "")
                if args_raw:
                    try:
                        args = json.loads(args_raw)
                    except json.JSONDecodeError:
                        args = {"raw": args_raw}
                else:
                    args = {}
                tool_calls.append(
                    ToolCall(
                        name=name,
                        arguments=args,
                        id=tool_id_by_index.get(idx),
                    )
                )

            if use_tools and not tool_calls and content_buf:
                tool_calls = _extract_xml_tool_calls(content_buf)

            if use_tools and not tool_calls and reasoning_buf:
                tool_calls = _extract_xml_tool_calls(reasoning_buf)

            yield StreamChunk(tool_calls=tool_calls, done=True)
            return
