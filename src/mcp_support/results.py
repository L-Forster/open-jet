from __future__ import annotations

import json
from typing import Any, Mapping

from ..tool_executor import ToolExecutionResult


def mcp_result_to_tool_execution_result(result: object, *, server_name: str, tool_name: str) -> ToolExecutionResult:
    is_error = _bool_attr(result, "isError", "is_error")
    structured = _value(result, "structuredContent")
    if structured is None:
        structured = _value(result, "structured_content")

    lines: list[str] = []
    for block in _content_blocks(result):
        rendered = _render_content_block(block)
        if rendered:
            lines.append(rendered)

    if structured is not None:
        rendered_structured = _json_text(structured)
        if rendered_structured and rendered_structured not in lines:
            lines.append(f"Structured content:\n{rendered_structured}")

    output = "\n".join(line for line in lines if line).strip()
    if not output:
        output = "MCP tool returned no content."
    return ToolExecutionResult(
        output=output,
        meta={
            "ok": not is_error,
            "status": "failed" if is_error else "completed",
            "mcp": True,
            "server": server_name,
            "tool": tool_name,
            "is_error": is_error,
            "structured": structured,
        },
    )


def mcp_error_result(*, server_name: str, tool_name: str, message: str, status: str = "exception") -> ToolExecutionResult:
    return ToolExecutionResult(
        output=f"MCP tool {server_name}/{tool_name} failed: {message}",
        meta={
            "ok": False,
            "status": status,
            "mcp": True,
            "server": server_name,
            "tool": tool_name,
            "error": message,
        },
    )


def _content_blocks(result: object) -> list[object]:
    content = _value(result, "content")
    if content is None:
        return []
    if isinstance(content, list):
        return content
    return [content]


def _render_content_block(block: object) -> str:
    kind = str(_value(block, "type") or type(block).__name__).strip()
    lowered = kind.lower()
    if lowered in {"text", "textcontent"} or _value(block, "text") is not None:
        return str(_value(block, "text") or "")
    if lowered in {"image", "imagecontent"}:
        return _media_summary("image", block)
    if lowered in {"audio", "audiocontent"}:
        return _media_summary("audio", block)
    if lowered in {"embeddedresource", "resource"} or _value(block, "resource") is not None:
        return _resource_summary(block)
    if lowered in {"blob", "blobcontent"}:
        return _media_summary("blob", block)
    return f"[unsupported MCP content block: {kind or type(block).__name__}]"


def _media_summary(label: str, block: object) -> str:
    mime = _value(block, "mimeType") or _value(block, "mime_type") or "unknown"
    data = _value(block, "data")
    size = len(data) if isinstance(data, (bytes, str)) else None
    size_text = f", bytes={size}" if size is not None else ""
    return f"[{label} returned: mime={mime}{size_text}]"


def _resource_summary(block: object) -> str:
    resource = _value(block, "resource") or block
    uri = _value(resource, "uri") or "unknown"
    mime = _value(resource, "mimeType") or _value(resource, "mime_type") or "unknown"
    text = _value(resource, "text")
    if isinstance(text, str) and text.strip():
        return f"[resource returned: uri={uri}, mime={mime}]\n{text}"
    blob = _value(resource, "blob")
    size = len(blob) if isinstance(blob, (bytes, str)) else None
    size_text = f", bytes={size}" if size is not None else ""
    return f"[resource returned: uri={uri}, mime={mime}{size_text}]"


def _json_text(value: object) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str)
    except (TypeError, ValueError):
        return str(value)


def _bool_attr(value: object, *names: str) -> bool:
    for name in names:
        raw = _value(value, name)
        if raw is not None:
            return bool(raw)
    return False


def _value(value: object, name: str) -> object:
    if isinstance(value, Mapping):
        return value.get(name)
    return getattr(value, name, None)
