from __future__ import annotations

import hashlib
import re
from copy import deepcopy
from typing import Any, Callable, Mapping

from ..tools.registry import ToolSpec
from .config import MCPServerConfig


_IDENTIFIER_RE = re.compile(r"[^A-Za-z0-9]+")
_MAX_TOOL_NAME_LENGTH = 64


def sanitize_identifier(value: str, *, fallback: str = "tool") -> str:
    text = _IDENTIFIER_RE.sub("_", str(value or "").strip().lower())
    text = re.sub(r"_+", "_", text).strip("_")
    return text or fallback


def sanitize_mcp_tool_name(server_name: str, tool_name: str) -> str:
    server = sanitize_identifier(server_name, fallback="server")
    tool = sanitize_identifier(tool_name, fallback="tool")
    generated = f"mcp_{server}_{tool}"
    if len(generated) <= _MAX_TOOL_NAME_LENGTH:
        return generated
    digest = hashlib.sha1(generated.encode("utf-8")).hexdigest()[:8]
    keep = _MAX_TOOL_NAME_LENGTH - len(digest) - 1
    return f"{generated[:keep].rstrip('_')}_{digest}"


def mcp_tool_to_spec(
    server: MCPServerConfig,
    mcp_tool: object,
    *,
    generated_name: str,
    executor: Callable[[dict[str, Any]], Any],
) -> ToolSpec:
    original_name = _tool_attr(mcp_tool, "name")
    if not original_name:
        raise ValueError("MCP tool is missing name")
    description = _tool_attr(mcp_tool, "description")
    input_schema = _input_schema(mcp_tool)
    properties = input_schema.get("properties", {})
    if not isinstance(properties, Mapping):
        properties = {}
    required = input_schema.get("required", ())
    if not isinstance(required, list):
        required = []
    explicit_tags = server.explicit_tags_for(original_name)
    tags = frozenset({"mcp", f"mcp:{server.name.lower()}", *explicit_tags})
    return ToolSpec(
        name=generated_name,
        description=_description(server.name, original_name, description),
        parameters=deepcopy(dict(properties)),
        required=tuple(str(item) for item in required if isinstance(item, str)),
        confirmation_required=server.confirmation_required,
        tags=tags,
        metadata={
            "mcp": True,
            "mcp_server_name": server.name,
            "mcp_tool_name": original_name,
            "mcp_transport": server.transport,
        },
        executor=executor,
    )


def tool_original_name(mcp_tool: object) -> str:
    return _tool_attr(mcp_tool, "name")


def _description(server_name: str, tool_name: str, description: str) -> str:
    prefix = f"[MCP: {server_name}/{tool_name}]"
    detail = " ".join(str(description or "").split())
    return f"{prefix} {detail}".strip()


def _input_schema(mcp_tool: object) -> dict[str, Any]:
    raw = _tool_value(mcp_tool, "inputSchema")
    if raw is None:
        raw = _tool_value(mcp_tool, "input_schema")
    if not isinstance(raw, Mapping):
        return {"type": "object", "properties": {}, "required": []}
    schema = deepcopy(dict(raw))
    if schema.get("type") != "object":
        schema["type"] = "object"
    if not isinstance(schema.get("properties"), Mapping):
        schema["properties"] = {}
    if not isinstance(schema.get("required"), list):
        schema["required"] = []
    return schema


def _tool_attr(mcp_tool: object, name: str) -> str:
    value = _tool_value(mcp_tool, name)
    return str(value or "").strip()


def _tool_value(mcp_tool: object, name: str) -> object:
    if isinstance(mcp_tool, Mapping):
        return mcp_tool.get(name)
    return getattr(mcp_tool, name, None)
