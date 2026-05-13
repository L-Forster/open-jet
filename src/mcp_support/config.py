from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml


DEFAULT_MCP_TIMEOUT_SECONDS = 30.0
PROJECT_MCP_CONFIG = ".openjet/mcp.yaml"
USER_MCP_CONFIG = ".openjet/mcp.yaml"
_ENV_VAR_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
_VALID_TRANSPORTS = {"stdio", "http", "streamable_http", "streamable-http"}
_READ_WRITE_TAGS = frozenset({"read", "write"})


@dataclass(frozen=True)
class MCPServerConfig:
    name: str
    transport: str
    enabled: bool = True
    command: str | None = None
    args: tuple[str, ...] = ()
    env: Mapping[str, str] = field(default_factory=dict)
    url: str | None = None
    headers: Mapping[str, str] = field(default_factory=dict)
    confirmation_required: bool = True
    timeout_seconds: float = DEFAULT_MCP_TIMEOUT_SECONDS
    include_tools: frozenset[str] = frozenset()
    exclude_tools: frozenset[str] = frozenset()
    default_tags: frozenset[str] = frozenset()
    tool_tags: Mapping[str, frozenset[str]] = field(default_factory=dict)

    def allows_tool(self, tool_name: str) -> bool:
        name = str(tool_name).strip()
        if not name:
            return False
        if self.include_tools:
            return name in self.include_tools
        return name not in self.exclude_tools

    def explicit_tags_for(self, tool_name: str) -> frozenset[str]:
        tags = set(self.default_tags)
        tags.update(self.tool_tags.get(str(tool_name).strip(), frozenset()))
        return frozenset(tag for tag in tags if tag in _READ_WRITE_TAGS)


@dataclass(frozen=True)
class MCPConfig:
    enabled: bool = False
    default_timeout_seconds: float = DEFAULT_MCP_TIMEOUT_SECONDS
    servers: tuple[MCPServerConfig, ...] = ()
    errors: tuple[str, ...] = ()

    def server(self, name: str) -> MCPServerConfig | None:
        needle = str(name).strip().lower()
        for server in self.servers:
            if server.name.lower() == needle:
                return server
        return None


def parse_mcp_config(cfg: Mapping[str, object] | None, *, strict: bool = False) -> MCPConfig:
    root = cfg if isinstance(cfg, Mapping) else {}
    section = root.get("mcp", root if _looks_like_mcp_section(root) else {})
    if not isinstance(section, Mapping):
        return MCPConfig(errors=("mcp config must be a mapping",)) if not strict else _raise("mcp config must be a mapping")

    enabled = _bool(section.get("enabled"), default=False)
    default_timeout = _positive_float(section.get("default_timeout_seconds"), DEFAULT_MCP_TIMEOUT_SECONDS)
    raw_servers = section.get("servers", {})
    raw_errors = section.get("_errors", ())
    errors = [str(error) for error in raw_errors] if isinstance(raw_errors, list) else []
    servers: list[MCPServerConfig] = []
    if raw_servers is None:
        raw_servers = {}
    if not isinstance(raw_servers, Mapping):
        message = "mcp.servers must be a mapping"
        if strict:
            raise ValueError(message)
        errors.append(message)
        raw_servers = {}

    for name, raw_server in raw_servers.items():
        try:
            servers.append(_parse_server(str(name), raw_server, default_timeout_seconds=default_timeout))
        except ValueError as exc:
            message = f"{name}: {exc}"
            if strict:
                raise ValueError(message) from exc
            errors.append(message)

    return MCPConfig(
        enabled=enabled,
        default_timeout_seconds=default_timeout,
        servers=tuple(servers),
        errors=tuple(errors),
    )


def project_mcp_config_path(root: Path | None = None) -> Path:
    return Path(root or Path.cwd()).expanduser().resolve() / PROJECT_MCP_CONFIG


def user_mcp_config_path(home: Path | None = None) -> Path:
    return Path(home or Path.home()).expanduser() / USER_MCP_CONFIG


def load_mcp_config_sources(
    *,
    root: Path | None = None,
    runtime_cfg: Mapping[str, object] | None = None,
    home: Path | None = None,
) -> dict[str, object]:
    merged: dict[str, object] = {}
    if isinstance(runtime_cfg, Mapping) and isinstance(runtime_cfg.get("mcp"), Mapping):
        merged = _merge_mcp_sections(merged, runtime_cfg.get("mcp"))  # type: ignore[arg-type]
    for path in (user_mcp_config_path(home), project_mcp_config_path(root)):
        section = _read_mcp_yaml(path)
        if section is not None:
            merged = _merge_mcp_sections(merged, section)
    return {"mcp": merged} if merged else {}


def load_project_mcp_config(root: Path | None = None) -> dict[str, object]:
    section = _read_mcp_yaml(project_mcp_config_path(root))
    return {"mcp": section} if section is not None else {}


def save_project_mcp_config(cfg: Mapping[str, object], root: Path | None = None) -> Path:
    section = cfg.get("mcp", cfg if _looks_like_mcp_section(cfg) else {})
    if not isinstance(section, Mapping):
        raise ValueError("mcp config must be a mapping")
    path = project_mcp_config_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(dict(section), default_flow_style=False, sort_keys=False), encoding="utf-8")
    return path


def add_stdio_server_config(cfg: dict[str, object], name: str, command_argv: list[str]) -> dict[str, object]:
    server_name = _safe_server_name(name)
    argv = [str(part) for part in command_argv if str(part).strip()]
    if not argv:
        raise ValueError("add-stdio requires a command after --")
    updated = dict(cfg)
    section = dict(updated.get("mcp") or {})
    servers = dict(section.get("servers") or {})
    servers[server_name] = {
        "enabled": True,
        "transport": "stdio",
        "command": argv[0],
        "args": argv[1:],
        "env": {},
        "confirmation_required": True,
        "tools": {"include": [], "exclude": []},
    }
    section["enabled"] = True
    section["servers"] = servers
    updated["mcp"] = section
    return updated


def remove_server_config(cfg: dict[str, object], name: str) -> dict[str, object]:
    server_name = str(name).strip()
    if not server_name:
        raise ValueError("server name is required")
    updated = dict(cfg)
    section = dict(updated.get("mcp") or {})
    servers = dict(section.get("servers") or {})
    if server_name not in servers:
        raise ValueError(f"unknown MCP server: {server_name}")
    servers.pop(server_name, None)
    section["servers"] = servers
    updated["mcp"] = section
    return updated


def expand_env_value(value: object) -> str:
    text = str(value or "")
    return _ENV_VAR_RE.sub(lambda match: os.environ.get(match.group(1), ""), text)


def _parse_server(name: str, raw: object, *, default_timeout_seconds: float) -> MCPServerConfig:
    server_name = _safe_server_name(name)
    if not isinstance(raw, Mapping):
        raise ValueError("server config must be a mapping")

    enabled = _bool(raw.get("enabled"), default=True)
    transport = str(raw.get("transport") or "stdio").strip().lower().replace("-", "_")
    if transport not in _VALID_TRANSPORTS:
        raise ValueError("transport must be stdio or http")
    if transport == "streamable-http":
        transport = "streamable_http"

    timeout = _positive_float(raw.get("timeout_seconds"), default_timeout_seconds)
    confirmation_required = _bool(raw.get("confirmation_required"), default=True)
    tools = raw.get("tools", {})
    tools_map = tools if isinstance(tools, Mapping) else {}

    env = _string_mapping(raw.get("env") or {}, expand_values=True, label="env")
    headers = _string_mapping(raw.get("headers") or {}, expand_values=True, label="headers")
    include = frozenset(_string_list(tools_map.get("include", ()), label="tools.include"))
    exclude = frozenset(_string_list(tools_map.get("exclude", ()), label="tools.exclude"))
    default_tags = _normalize_tags(raw.get("tags", ()))
    tool_tags = _parse_tool_tags(tools_map.get("tags", {}))

    if transport == "stdio":
        command = raw.get("command")
        if not isinstance(command, str) or not command.strip():
            raise ValueError("stdio servers require command")
        args = tuple(_string_list(raw.get("args", ()), label="args"))
        return MCPServerConfig(
            name=server_name,
            transport="stdio",
            enabled=enabled,
            command=command.strip(),
            args=args,
            env=env,
            confirmation_required=confirmation_required,
            timeout_seconds=timeout,
            include_tools=include,
            exclude_tools=exclude,
            default_tags=default_tags,
            tool_tags=tool_tags,
        )

    url = raw.get("url")
    if not isinstance(url, str) or not url.strip():
        raise ValueError("http servers require url")
    return MCPServerConfig(
        name=server_name,
        transport="streamable_http",
        enabled=enabled,
        url=url.strip(),
        headers=headers,
        confirmation_required=confirmation_required,
        timeout_seconds=timeout,
        include_tools=include,
        exclude_tools=exclude,
        default_tags=default_tags,
        tool_tags=tool_tags,
    )


def _looks_like_mcp_section(value: Mapping[str, object]) -> bool:
    return "servers" in value or "enabled" in value or "default_timeout_seconds" in value


def _read_mcp_yaml(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        return {"_errors": [f"{path}: malformed YAML: {str(exc).strip()}"]}
    if not isinstance(raw, Mapping):
        return {"_errors": [f"{path}: must contain a YAML mapping"]}
    section = raw.get("mcp", raw if _looks_like_mcp_section(raw) else {})
    if not isinstance(section, Mapping):
        return {"_errors": [f"{path}: mcp section must be a mapping"]}
    return dict(section)


def _merge_mcp_sections(base: Mapping[str, object], overlay: Mapping[str, object]) -> dict[str, object]:
    merged = dict(base)
    for key, value in overlay.items():
        if key == "servers":
            base_servers = merged.get("servers")
            overlay_servers = value
            servers = dict(base_servers) if isinstance(base_servers, Mapping) else {}
            if isinstance(overlay_servers, Mapping):
                servers.update(dict(overlay_servers))
            merged["servers"] = servers
            continue
        if key == "_errors":
            errors = list(merged.get("_errors", ())) if isinstance(merged.get("_errors"), list) else []
            if isinstance(value, list):
                errors.extend(str(error) for error in value)
            merged["_errors"] = errors
            continue
        merged[str(key)] = value
    return merged


def _raise(message: str) -> MCPConfig:
    raise ValueError(message)


def _safe_server_name(name: str) -> str:
    text = str(name or "").strip()
    if not text:
        raise ValueError("server name cannot be empty")
    return text


def _bool(value: object, *, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    return bool(value)


def _positive_float(value: object, default: float) -> float:
    if value is None:
        return default
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("timeout must be numeric") from exc
    if parsed <= 0:
        raise ValueError("timeout must be > 0")
    return parsed


def _string_list(value: object, *, label: str) -> list[str]:
    if value in (None, ""):
        return []
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{label} must be a list")
    items: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise ValueError(f"{label} entries must be strings")
        text = item.strip()
        if text:
            items.append(text)
    return items


def _string_mapping(value: object, *, expand_values: bool, label: str) -> dict[str, str]:
    if value in (None, ""):
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    parsed: dict[str, str] = {}
    for key, raw_value in value.items():
        if not isinstance(key, str):
            raise ValueError(f"{label} keys must be strings")
        parsed[key] = expand_env_value(raw_value) if expand_values else str(raw_value)
    return parsed


def _normalize_tags(value: object) -> frozenset[str]:
    try:
        raw_tags = _string_list(value, label="tags")
    except ValueError:
        return frozenset()
    return frozenset(tag.strip().lower() for tag in raw_tags if tag.strip().lower() in _READ_WRITE_TAGS)


def _parse_tool_tags(value: object) -> dict[str, frozenset[str]]:
    if value in (None, ""):
        return {}
    if not isinstance(value, Mapping):
        return {}
    parsed: dict[str, frozenset[str]] = {}
    for tool_name, raw_tags in value.items():
        name = str(tool_name).strip()
        if not name:
            continue
        parsed[name] = _normalize_tags(raw_tags)
    return parsed
