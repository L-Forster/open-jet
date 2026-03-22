from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Awaitable, Callable

from .airgap import airgapped_from_cfg, set_airgapped
from .agent import ActionKind, Agent
from .config import load_config
from .multimodal import build_user_content, content_to_plain_text
from .persistent_memory import build_system_prompt
from .runtime_limits import derive_context_budget, estimate_tokens
from .runtime_protocol import ToolCall
from .runtime_registry import create_runtime_client
from .tool_executor import ToolExecutionResult, execute_tool


ApprovalHandler = Callable[[ToolCall], bool | Awaitable[bool]]


class SDKEventKind(Enum):
    TEXT = auto()
    TOOL_REQUEST = auto()
    TOOL_RESULT = auto()
    CONDENSE = auto()
    DONE = auto()
    ERROR = auto()


@dataclass(frozen=True)
class ToolResult:
    tool_call: ToolCall
    output: str
    meta: dict = field(default_factory=dict)
    approved: bool = True

    @property
    def ok(self) -> bool:
        return bool(self.meta.get("ok", False))


@dataclass(frozen=True)
class SDKEvent:
    kind: SDKEventKind
    text: str = ""
    tool_call: ToolCall | None = None
    tool_result: ToolResult | None = None


@dataclass(frozen=True)
class SDKResponse:
    text: str
    tool_results: list[ToolResult] = field(default_factory=list)
    condense_messages: list[str] = field(default_factory=list)


class OpenJetSession:
    def __init__(
        self,
        agent: Agent,
        *,
        approval_handler: ApprovalHandler | None = None,
        allowed_tools: set[str] | None = None,
        airgapped: bool = False,
    ) -> None:
        self.agent = agent
        self._approval_handler = approval_handler
        self._allowed_tools = allowed_tools
        self.airgapped = bool(airgapped)
        set_airgapped(self.airgapped)

    @classmethod
    async def create(
        cls,
        *,
        cfg: dict | None = None,
        system_prompt: str | None = None,
        approval_handler: ApprovalHandler | None = None,
        allowed_tools: set[str] | None = None,
        airgapped: bool | None = None,
    ) -> OpenJetSession:
        resolved_cfg = dict(cfg or load_config())
        resolved_cfg["airgapped"] = airgapped_from_cfg(resolved_cfg, override=airgapped)
        set_airgapped(bool(resolved_cfg["airgapped"]))
        client = create_runtime_client(resolved_cfg)
        mem_cfg = resolved_cfg.get("memory_guard", {})
        await client.start()
        agent = Agent(
            client=client,
            system_prompt=await build_system_prompt(
                system_prompt if system_prompt is not None else str(resolved_cfg.get("system_prompt", "")),
                Path.cwd(),
                cfg=resolved_cfg,
            ),
            context_window_tokens=client.context_window_tokens,
            context_reserved_tokens=(
                int(mem_cfg["context_reserved_tokens"])
                if mem_cfg.get("context_reserved_tokens") is not None
                else None
            ),
            min_prompt_tokens=int(mem_cfg.get("min_prompt_tokens", 256)),
            min_available_mb=(
                int(mem_cfg["min_available_mb"])
                if mem_cfg.get("min_available_mb") is not None
                else None
            ),
            max_used_percent=(
                float(mem_cfg["max_used_percent"])
                if mem_cfg.get("max_used_percent") is not None
                else None
            ),
            memory_check_interval_chunks=int(mem_cfg.get("check_interval_chunks", 16)),
            condense_target_tokens=int(mem_cfg.get("condense_target_tokens", 900)),
            keep_last_messages=int(mem_cfg.get("keep_last_messages", 6)),
        )
        return cls(
            agent,
            approval_handler=approval_handler,
            allowed_tools=allowed_tools,
            airgapped=bool(resolved_cfg["airgapped"]),
        )

    async def close(self) -> None:
        await self.agent.client.close()

    def set_airgapped(self, enabled: bool) -> None:
        self.airgapped = bool(enabled)
        set_airgapped(self.airgapped)

    def add_turn_context(self, messages: list[dict]) -> None:
        self.agent.set_turn_context(messages)

    def clear_turn_context(self) -> None:
        self.agent.clear_turn_context()

    async def stream(self, prompt: str, *, image_paths: list[str] | None = None):
        self.agent.add_user_message(prompt, image_paths=image_paths)
        while True:
            pending_tool_calls: list[ToolCall] = []
            async for event in self.agent.run_turn():
                if event.kind == ActionKind.TEXT:
                    yield SDKEvent(kind=SDKEventKind.TEXT, text=event.text)
                    continue
                if event.kind == ActionKind.TOOL_REQUEST and event.tool_call:
                    pending_tool_calls.append(event.tool_call)
                    yield SDKEvent(kind=SDKEventKind.TOOL_REQUEST, tool_call=event.tool_call)
                    continue
                if event.kind == ActionKind.CONDENSE:
                    message = await self.agent.condense_context()
                    yield SDKEvent(kind=SDKEventKind.CONDENSE, text=message)
                    break
                if event.kind == ActionKind.ERROR:
                    yield SDKEvent(kind=SDKEventKind.ERROR, text=event.text)
                    return
                if event.kind == ActionKind.DONE:
                    yield SDKEvent(kind=SDKEventKind.DONE)
                    return
            else:
                if not pending_tool_calls:
                    yield SDKEvent(kind=SDKEventKind.DONE)
                    return

            if not pending_tool_calls:
                continue

            for tool_call in pending_tool_calls:
                tool_result = await self._handle_tool_call(tool_call)
                if tool_result is None:
                    continue
                yield SDKEvent(
                    kind=SDKEventKind.TOOL_RESULT,
                    tool_call=tool_call,
                    tool_result=tool_result,
                )

    async def run(self, prompt: str, *, image_paths: list[str] | None = None) -> SDKResponse:
        text_parts: list[str] = []
        tool_results: list[ToolResult] = []
        condense_messages: list[str] = []

        async for event in self.stream(prompt, image_paths=image_paths):
            if event.kind == SDKEventKind.TEXT:
                text_parts.append(event.text)
            elif event.kind == SDKEventKind.TOOL_RESULT and event.tool_result:
                tool_results.append(event.tool_result)
            elif event.kind == SDKEventKind.CONDENSE:
                condense_messages.append(event.text)
            elif event.kind == SDKEventKind.ERROR:
                raise RuntimeError(event.text or "Agent turn failed.")

        return SDKResponse(
            text="".join(text_parts),
            tool_results=tool_results,
            condense_messages=condense_messages,
        )

    async def _handle_tool_call(self, tool_call: ToolCall) -> ToolResult | None:
        if self._allowed_tools is not None and tool_call.name not in self._allowed_tools:
            output = f"Tool {tool_call.name} is not allowed in this session."
            self.agent.complete_tool_call(tool_call, output)
            return ToolResult(
                tool_call=tool_call,
                output=output,
                meta={"ok": False, "denied": True},
                approved=False,
            )

        if self.agent.needs_confirmation(tool_call):
            approved = await self._approve(tool_call)
            if not approved:
                output = "User denied this action."
                self.agent.complete_tool_call(tool_call, output)
                return ToolResult(
                    tool_call=tool_call,
                    output=output,
                    meta={"ok": False, "denied": True},
                    approved=False,
                )

        if tool_call.name == "load_file":
            self._clamp_load_file_tool_budget(tool_call)

        result = await execute_tool(tool_call)
        context_output = self._fit_tool_result_content_to_budget(result)
        self.agent.complete_tool_call(tool_call, context_output)
        if bool(result.meta.get("internal_retry")):
            return None
        return ToolResult(tool_call=tool_call, output=result.output, meta=result.meta, approved=True)

    async def _approve(self, tool_call: ToolCall) -> bool:
        if self._approval_handler is None:
            return False
        decision = self._approval_handler(tool_call)
        if inspect.isawaitable(decision):
            return bool(await decision)
        return bool(decision)

    def _clamp_load_file_tool_budget(self, tool_call: ToolCall) -> None:
        if not isinstance(tool_call.arguments, dict):
            return
        remaining = self._remaining_prompt_tokens(reserve_next_turn_overhead=True)
        current = tool_call.arguments.get("max_tokens")
        if not isinstance(current, int):
            tool_call.arguments["max_tokens"] = remaining
            return
        tool_call.arguments["max_tokens"] = max(128, min(current, remaining))

    def _remaining_prompt_tokens(self, *, reserve_next_turn_overhead: bool = False) -> int:
        current = self.agent.estimated_context_tokens()
        budget = self.agent.context_budget()
        if not budget:
            window = self.agent.context_window_tokens or 2048
            budget = derive_context_budget(window)
        runtime_overhead = (
            self.agent.runtime_overhead_tokens(force_post_tool_continuation=True)
            if reserve_next_turn_overhead
            else 0
        )
        return max(0, budget.prompt_tokens - current - runtime_overhead)

    def _fit_tool_result_to_budget(self, result: str) -> str:
        if not result:
            return result

        budget_tokens = self._remaining_prompt_tokens(reserve_next_turn_overhead=True)
        runtime_overhead = self.agent.runtime_overhead_tokens(force_post_tool_continuation=True)
        result_tokens = estimate_tokens(result)
        if budget_tokens <= 0:
            return (
                "...[tool output omitted: no prompt budget remaining]\n"
                "[tool context limit reached: "
                f"tool~{result_tokens}t, available~0t after overhead~{runtime_overhead}t; "
                "narrow the next tool call or condense context]"
            )
        if result_tokens <= budget_tokens:
            return result

        prefix = "...[tool output truncated]\n"
        suffix = (
            "\n[tool context limit reached: "
            f"tool~{result_tokens}t, available~{budget_tokens}t after overhead~{runtime_overhead}t; "
            "narrow the next tool call or condense context]"
        )
        max_chars = max(256, budget_tokens * 4)
        clipped = result[-max_chars:]
        candidate = prefix + clipped + suffix

        while estimate_tokens(candidate) > budget_tokens and len(clipped) > 64:
            clipped = clipped[max(64, int(len(clipped) * 0.85)):]
            candidate = prefix + clipped + suffix

        return candidate

    def _fit_tool_result_content_to_budget(self, result: ToolExecutionResult) -> object:
        content = result.context_content
        if content is None:
            return self._fit_tool_result_to_budget(result.output)
        if isinstance(content, str):
            return self._fit_tool_result_to_budget(content)

        image_paths: list[str] = []
        for block in content:
            if not isinstance(block, dict):
                continue
            if str(block.get("type", "")).strip().lower() != "input_image":
                continue
            path = str(block.get("path", "")).strip()
            if path:
                image_paths.append(path)

        text = self._fit_tool_result_to_budget(content_to_plain_text(content))
        return build_user_content(text, image_paths or None)


async def create_agent(
    *,
    cfg: dict | None = None,
    system_prompt: str | None = None,
    approval_handler: ApprovalHandler | None = None,
    allowed_tools: set[str] | None = None,
    airgapped: bool | None = None,
) -> OpenJetSession:
    return await OpenJetSession.create(
        cfg=cfg,
        system_prompt=system_prompt,
        approval_handler=approval_handler,
        allowed_tools=allowed_tools,
        airgapped=airgapped,
    )
