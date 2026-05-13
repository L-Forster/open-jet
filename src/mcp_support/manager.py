from __future__ import annotations

import asyncio
import contextvars
import inspect
from contextlib import AsyncExitStack, contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping

from ..tools.registry import TOOL_REGISTRY, get_tool_spec, unregister_tool
from .config import MCPConfig, MCPServerConfig, load_mcp_config_sources, parse_mcp_config
from .redaction import redact_mapping, redact_text
from .results import mcp_error_result, mcp_result_to_tool_execution_result
from .schema import mcp_tool_to_spec, sanitize_mcp_tool_name, tool_original_name


class MCPSDKUnavailable(RuntimeError):
    pass


@dataclass(frozen=True)
class MCPToolRef:
    openjet_name: str
    server_name: str
    mcp_tool_name: str
    timeout_seconds: float


@dataclass
class MCPServerRuntime:
    config: MCPServerConfig
    session: object
    stack: AsyncExitStack | None = None
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


@dataclass(frozen=True)
class MCPServerStatus:
    name: str
    enabled: bool
    ok: bool
    message: str
    tools: tuple[str, ...] = ()


SessionFactory = Callable[[MCPServerConfig], object]


@dataclass
class _GlobalToolRegistration:
    owners: list["MCPManager"] = field(default_factory=list)
    metadata: Mapping[str, object] = field(default_factory=dict)
    server_signature: tuple[object, ...] = ()


_GLOBAL_TOOL_REGISTRATIONS: dict[str, _GlobalToolRegistration] = {}
_ACTIVE_MCP_MANAGER: contextvars.ContextVar["MCPManager | None"] = contextvars.ContextVar(
    "openjet_active_mcp_manager",
    default=None,
)


@contextmanager
def active_mcp_manager(manager: "MCPManager | None"):
    token = _ACTIVE_MCP_MANAGER.set(manager)
    try:
        yield
    finally:
        _ACTIVE_MCP_MANAGER.reset(token)


class MCPManager:
    def __init__(self, config: MCPConfig, *, session_factory: SessionFactory | None = None) -> None:
        self.config = config
        self._session_factory = session_factory
        self._servers: dict[str, MCPServerRuntime] = {}
        self._tool_refs: dict[str, MCPToolRef] = {}
        self._registered_names: list[str] = []
        self._statuses: dict[str, MCPServerStatus] = {}
        self._initialized = False

    @classmethod
    def from_config(cls, cfg: dict[str, object] | None, *, session_factory: SessionFactory | None = None) -> "MCPManager":
        return cls(parse_mcp_config(cfg or {}), session_factory=session_factory)

    @classmethod
    def from_sources(
        cls,
        *,
        root=None,
        runtime_cfg: Mapping[str, object] | None = None,
        session_factory: SessionFactory | None = None,
    ) -> "MCPManager":
        return cls(
            parse_mcp_config(load_mcp_config_sources(root=root, runtime_cfg=runtime_cfg)),
            session_factory=session_factory,
        )

    async def initialize(self, *, server_names: Iterable[str] | None = None) -> None:
        if self._initialized and server_names is None:
            return
        requested = {str(name).strip().lower() for name in server_names or () if str(name).strip()}
        if not self.config.enabled:
            self._statuses["mcp"] = MCPServerStatus(
                name="mcp",
                enabled=False,
                ok=True,
                message="MCP is disabled.",
            )
            self._initialized = True
            return

        for error in self.config.errors:
            self._statuses[f"config:{len(self._statuses)}"] = MCPServerStatus(
                name="config",
                enabled=False,
                ok=False,
                message=error,
            )

        for server in self.config.servers:
            if requested and server.name.lower() not in requested:
                continue
            if not server.enabled:
                self._statuses[server.name] = MCPServerStatus(
                    name=server.name,
                    enabled=False,
                    ok=True,
                    message="Server is disabled.",
                )
                continue
            await self._initialize_server(server)
        self._initialized = True

    async def call_tool(self, openjet_tool_name: str, arguments: dict[str, Any]) -> object:
        ref = self._tool_refs.get(str(openjet_tool_name).strip())
        if ref is None:
            return mcp_error_result(
                server_name="unknown",
                tool_name=str(openjet_tool_name),
                message="MCP tool is not registered",
                status="unknown_tool",
            )
        runtime = self._servers.get(ref.server_name)
        if runtime is None:
            return mcp_error_result(
                server_name=ref.server_name,
                tool_name=ref.mcp_tool_name,
                message="MCP server session is not available",
                status="server_unavailable",
            )
        try:
            async with runtime.lock:
                result = await asyncio.wait_for(
                    runtime.session.call_tool(ref.mcp_tool_name, arguments=arguments),
                    timeout=ref.timeout_seconds,
                )
        except asyncio.TimeoutError:
            return mcp_error_result(
                server_name=ref.server_name,
                tool_name=ref.mcp_tool_name,
                message=f"timed out after {ref.timeout_seconds:g}s",
                status="timeout",
            )
        except Exception as exc:
            return mcp_error_result(
                server_name=ref.server_name,
                tool_name=ref.mcp_tool_name,
                message=self._redact_server_error(runtime.config, exc),
            )
        return mcp_result_to_tool_execution_result(
            result,
            server_name=ref.server_name,
            tool_name=ref.mcp_tool_name,
        )

    async def aclose(self) -> None:
        for name in reversed(self._registered_names):
            _release_global_tool(self, name)
        self._registered_names.clear()
        self._tool_refs.clear()

        for runtime in list(self._servers.values()):
            if runtime.stack is not None:
                try:
                    await runtime.stack.aclose()
                except Exception:
                    pass
                continue
            close = getattr(runtime.session, "aclose", None) or getattr(runtime.session, "close", None)
            if callable(close):
                result = close()
                if inspect.isawaitable(result):
                    try:
                        await result
                    except Exception:
                        pass
        self._servers.clear()
        self._initialized = False

    def statuses(self) -> tuple[MCPServerStatus, ...]:
        return tuple(self._statuses.values())

    def registered_tool_names(self) -> tuple[str, ...]:
        return tuple(self._registered_names)

    def format_status(self) -> str:
        lines: list[str] = []
        if not self.config.enabled:
            return "MCP is disabled. Set `mcp.enabled: true` in config.yaml to enable MCP servers."
        if not self.config.servers:
            lines.append("MCP is enabled, but no servers are configured.")
        else:
            lines.append("MCP servers:")
            for server in self.config.servers:
                status = self._statuses.get(server.name)
                if status is None:
                    state = "configured" if server.enabled else "disabled"
                    detail = self._server_config_detail(server)
                    lines.append(f"- {server.name}: {state}{detail}")
                    continue
                state = "ok" if status.ok else "failed"
                if not status.enabled:
                    state = "disabled"
                tool_suffix = f" | tools={', '.join(status.tools)}" if status.tools else ""
                lines.append(f"- {status.name}: {state} | {status.message}{tool_suffix}")
        if self.config.errors:
            lines.append("Config errors:")
            lines.extend(f"- {error}" for error in self.config.errors)
        return "\n".join(lines)

    async def _initialize_server(self, server: MCPServerConfig) -> None:
        runtime: MCPServerRuntime | None = None
        try:
            runtime = await self._connect_server(server)
            tools = await self._list_tools(runtime)
            registered = self._register_tools(server, tools)
        except Exception as exc:
            if runtime is not None:
                await _close_runtime(runtime)
            self._statuses[server.name] = MCPServerStatus(
                name=server.name,
                enabled=True,
                ok=False,
                message=self._redact_server_error(server, exc),
            )
            return
        self._servers[server.name] = runtime
        self._statuses[server.name] = MCPServerStatus(
            name=server.name,
            enabled=True,
            ok=True,
            message=f"Connected via {server.transport}.",
            tools=tuple(registered),
        )

    async def _connect_server(self, server: MCPServerConfig) -> MCPServerRuntime:
        if self._session_factory is not None:
            session = self._session_factory(server)
            if inspect.isawaitable(session):
                session = await session
            if hasattr(session, "__aenter__") and hasattr(session, "__aexit__"):
                stack = AsyncExitStack()
                try:
                    session = await stack.enter_async_context(session)
                    await self._maybe_initialize_session(session, server.timeout_seconds)
                except Exception:
                    await stack.aclose()
                    raise
                return MCPServerRuntime(config=server, session=session, stack=stack)
            await self._maybe_initialize_session(session, server.timeout_seconds)
            return MCPServerRuntime(config=server, session=session)
        return await self._connect_with_sdk(server)

    async def _connect_with_sdk(self, server: MCPServerConfig) -> MCPServerRuntime:
        sdk = _load_mcp_sdk(server.transport)
        stack = AsyncExitStack()
        try:
            if server.transport == "stdio":
                params = sdk["StdioServerParameters"](
                    command=server.command,
                    args=list(server.args),
                    env=dict(server.env) or None,
                )
                read_stream, write_stream = await stack.enter_async_context(sdk["stdio_client"](params))
            else:
                transport_cm = _streamable_http_context(
                    sdk["streamable_http_client"],
                    str(server.url),
                    headers=dict(server.headers),
                    timeout_seconds=server.timeout_seconds,
                )
                streams = await stack.enter_async_context(transport_cm)
                read_stream, write_stream = streams[0], streams[1]
            session = await stack.enter_async_context(sdk["ClientSession"](read_stream, write_stream))
            await self._maybe_initialize_session(session, server.timeout_seconds)
            return MCPServerRuntime(config=server, session=session, stack=stack)
        except Exception:
            await stack.aclose()
            raise

    async def _maybe_initialize_session(self, session: object, timeout_seconds: float) -> None:
        initialize = getattr(session, "initialize", None)
        if callable(initialize):
            await asyncio.wait_for(initialize(), timeout=timeout_seconds)

    async def _list_tools(self, runtime: MCPServerRuntime) -> list[object]:
        response = await asyncio.wait_for(runtime.session.list_tools(), timeout=runtime.config.timeout_seconds)
        tools = getattr(response, "tools", None)
        if tools is None and isinstance(response, dict):
            tools = response.get("tools")
        if tools is None:
            return []
        return list(tools)

    def _register_tools(self, server: MCPServerConfig, tools: list[object]) -> list[str]:
        filtered = [tool for tool in tools if server.allows_tool(tool_original_name(tool))]
        generated_names: set[str] = set()
        specs = []
        for tool in filtered:
            original_name = tool_original_name(tool)
            generated_name = sanitize_mcp_tool_name(server.name, original_name)
            if generated_name in generated_names:
                raise ValueError(f"duplicate generated MCP tool name: {generated_name}")
            generated_names.add(generated_name)

            async def executor(args: dict[str, Any], *, _name: str = generated_name):
                return await _execute_global_mcp_tool(_name, args)

            specs.append(
                mcp_tool_to_spec(
                    server,
                    tool,
                    generated_name=generated_name,
                    executor=executor,
                )
            )

        registered: list[str] = []
        try:
            for spec in specs:
                _acquire_global_tool(self, server, spec)
                ref = MCPToolRef(
                    openjet_name=spec.name,
                    server_name=server.name,
                    mcp_tool_name=str(spec.metadata.get("mcp_tool_name", "")),
                    timeout_seconds=server.timeout_seconds,
                )
                self._tool_refs[spec.name] = ref
                self._registered_names.append(spec.name)
                registered.append(spec.name)
        except Exception:
            for name in registered:
                _release_global_tool(self, name)
                self._registered_names.remove(name)
                self._tool_refs.pop(name, None)
            raise
        return registered

    @staticmethod
    def _redact_server_error(server: MCPServerConfig, exc: Exception) -> str:
        raw = str(exc).strip() or type(exc).__name__
        secret_values = {**dict(server.env), **dict(server.headers)}
        return redact_text(raw, secret_values)

    @staticmethod
    def _server_config_detail(server: MCPServerConfig) -> str:
        if server.transport == "stdio":
            env = redact_mapping(server.env)
            env_text = f" env={env}" if env else ""
            return f" | stdio command={server.command} args={list(server.args)}{env_text}"
        headers = redact_mapping(server.headers)
        headers_text = f" headers={headers}" if headers else ""
        return f" | http url={server.url}{headers_text}"


def _load_mcp_sdk(transport: str) -> dict[str, object]:
    try:
        try:
            from mcp import ClientSession, StdioServerParameters
        except ImportError:
            from mcp.client.session import ClientSession
            from mcp.client.stdio import StdioServerParameters
        from mcp.client.stdio import stdio_client
    except ImportError as exc:
        raise MCPSDKUnavailable("MCP Python SDK is not installed. Install `open-jet[mcp]` or `mcp`.") from exc

    sdk: dict[str, object] = {
        "ClientSession": ClientSession,
        "StdioServerParameters": StdioServerParameters,
        "stdio_client": stdio_client,
    }
    if transport != "stdio":
        try:
            from mcp.client.streamable_http import streamable_http_client
        except ImportError as exc:
            raise MCPSDKUnavailable("MCP SDK streamable HTTP client is not available.") from exc
        sdk["streamable_http_client"] = streamable_http_client
    return sdk


def _streamable_http_context(streamable_http_client: object, url: str, *, headers: dict[str, str], timeout_seconds: float):
    client = streamable_http_client
    try:
        return client(url, headers=headers or None, timeout=timeout_seconds)
    except TypeError:
        return client(url, headers=headers or None)


async def _execute_global_mcp_tool(openjet_tool_name: str, args: dict[str, Any]):
    registration = _GLOBAL_TOOL_REGISTRATIONS.get(openjet_tool_name)
    if registration is None:
        return mcp_error_result(
            server_name="unknown",
            tool_name=openjet_tool_name,
            message="MCP tool is not registered",
            status="unknown_tool",
        )
    active_manager = _ACTIVE_MCP_MANAGER.get()
    if active_manager is not None and openjet_tool_name in active_manager._tool_refs:
        return await active_manager.call_tool(openjet_tool_name, args)
    owners = [owner for owner in registration.owners if openjet_tool_name in owner._tool_refs]
    if len(owners) == 1:
        return await owners[0].call_tool(openjet_tool_name, args)
    if len(owners) > 1:
        return mcp_error_result(
            server_name="unknown",
            tool_name=openjet_tool_name,
            message="MCP tool has multiple active server sessions and no active caller context",
            status="ambiguous_session",
        )
    return mcp_error_result(
        server_name="unknown",
        tool_name=openjet_tool_name,
        message="MCP tool has no active server session",
        status="server_unavailable",
    )


def _acquire_global_tool(manager: MCPManager, server: MCPServerConfig, spec) -> None:
    signature = _server_signature(server)
    existing = _GLOBAL_TOOL_REGISTRATIONS.get(spec.name)
    if existing is not None:
        if existing.metadata != dict(spec.metadata) or existing.server_signature != signature:
            raise ValueError(f"duplicate generated MCP tool name: {spec.name}")
        if manager not in existing.owners:
            existing.owners.append(manager)
        return

    existing_spec = get_tool_spec(spec.name)
    if existing_spec is not None:
        raise ValueError(f"duplicate generated MCP tool name: {spec.name}")
    TOOL_REGISTRY.register(spec)
    _GLOBAL_TOOL_REGISTRATIONS[spec.name] = _GlobalToolRegistration(
        owners=[manager],
        metadata=dict(spec.metadata),
        server_signature=signature,
    )


def _release_global_tool(manager: MCPManager, openjet_tool_name: str) -> None:
    registration = _GLOBAL_TOOL_REGISTRATIONS.get(openjet_tool_name)
    if registration is None:
        return
    registration.owners = [owner for owner in registration.owners if owner is not manager]
    if registration.owners:
        return
    unregister_tool(openjet_tool_name)
    _GLOBAL_TOOL_REGISTRATIONS.pop(openjet_tool_name, None)


def _server_signature(server: MCPServerConfig) -> tuple[object, ...]:
    return (
        server.name,
        server.transport,
        server.command,
        tuple(server.args),
        server.url,
        tuple(sorted(server.env.items())),
        tuple(sorted(server.headers.items())),
        server.confirmation_required,
        server.timeout_seconds,
    )


async def _close_runtime(runtime: MCPServerRuntime) -> None:
    if runtime.stack is not None:
        try:
            await runtime.stack.aclose()
        except Exception:
            pass
        return
    close = getattr(runtime.session, "aclose", None) or getattr(runtime.session, "close", None)
    if callable(close):
        result = close()
        if inspect.isawaitable(result):
            try:
                await result
            except Exception:
                pass
