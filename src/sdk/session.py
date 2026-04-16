from __future__ import annotations

import json
import inspect
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import TYPE_CHECKING, Any, Awaitable, Callable

from ..airgap import airgapped_from_cfg, set_airgapped
from ..agent import ActionKind, Agent
from ..config import load_config
from ..memory_reflection import (
    build_recorded_turn_payload,
    reflect_agent_persistent_memory,
    refresh_agent_system_prompt,
)
from ..multimodal import build_user_content, content_to_plain_text
from ..runtime_limits import derive_context_budget, estimate_tokens
from ..runtime_protocol import ToolCall
from ..tool_executor import ToolExecutionResult, execute_tool
from ..harness import (
    HarnessState,
    clear_todos,
    complete_todo,
    pre_edit_gate_message,
    exit_plan_mode as harness_exit_plan_mode,
    record_verification_skip,
    upsert_todos,
    verification_gate_message,
)

if TYPE_CHECKING:
    from ..session_logging import SessionLogger


ApprovalHandler = Callable[[ToolCall], bool | Awaitable[bool]]


class SDKEventKind(Enum):
    TEXT = auto()
    REASONING = auto()
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
    context_output: object = ""
    context_output_text: str = ""

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


@dataclass
class _TurnArtifactState:
    turn_index: int
    prompt_index: int
    user_prompt: str
    mode: str
    active_step: str | None
    runtime_request: dict[str, Any]
    extra: dict[str, Any] = field(default_factory=dict)
    assistant_chunks: list[str] = field(default_factory=list)
    assistant_text: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    tool_results: list[dict[str, Any]] = field(default_factory=list)
    condense_reason: str | None = None
    condense_result: str | None = None
    condense_report: dict[str, Any] | None = None
    error: str | None = None
    status: str = "in_progress"


class _SessionArtifactRecorder:
    def __init__(self, session_dir: Path) -> None:
        self.session_dir = session_dir
        self.chat_log_path = self.session_dir / "chat_log.md"
        self.turns_dir = self.session_dir / "turns"
        self.turn_state_paths: list[str] = []
        self._turns: dict[int, _TurnArtifactState] = {}
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.turns_dir.mkdir(parents=True, exist_ok=True)

    def record_user_prompt(self, *, prompt_index: int, text: str) -> None:
        self._append_markdown(
            [
                f"## Prompt {prompt_index} User",
                "",
                text,
                "",
            ]
        )

    def start_turn(
        self,
        *,
        turn_index: int,
        prompt_index: int,
        user_prompt: str,
        mode: str | None,
        active_step: str | None,
        runtime_request: dict[str, Any],
        extra: dict[str, Any] | None = None,
    ) -> None:
        self._turns[turn_index] = _TurnArtifactState(
            turn_index=turn_index,
            prompt_index=prompt_index,
            user_prompt=user_prompt,
            mode=str(mode or "chat"),
            active_step=active_step,
            runtime_request=runtime_request,
            extra=dict(extra or {}),
        )

    def record_text_chunk(self, *, turn_index: int, text: str) -> None:
        if not text:
            return
        state = self._turns.get(turn_index)
        if state is None:
            return
        state.assistant_chunks.append(text)
        state.assistant_text += text

    def record_tool_request(self, *, turn_index: int, tool_call: ToolCall) -> None:
        state = self._turns.get(turn_index)
        if state is None:
            return
        payload = {
            "tool": tool_call.name,
            "arguments": dict(tool_call.arguments),
            "id": tool_call.id,
        }
        state.tool_calls.append(payload)
        self._append_markdown(
            [
                f"## Prompt {state.prompt_index} / Turn {turn_index} Tool Request",
                "",
                f"- tool: `{tool_call.name}`",
                f"- arguments: `{json.dumps(tool_call.arguments, ensure_ascii=False)}`",
                "",
            ]
        )

    def record_tool_result(
        self,
        *,
        turn_index: int,
        tool_call: ToolCall,
        tool_result: ToolResult,
    ) -> None:
        state = self._turns.get(turn_index)
        if state is None:
            return
        payload = {
            "tool": tool_call.name,
            "arguments": dict(tool_call.arguments),
            "id": tool_call.id,
            "approved": tool_result.approved,
            "status": str(tool_result.meta.get("status", "")),
            "ok": bool(tool_result.ok),
            "meta": dict(tool_result.meta),
            "raw_output": tool_result.output,
            "context_output": tool_result.context_output,
            "context_output_text": tool_result.context_output_text,
        }
        state.tool_results.append(payload)
        self._append_markdown(
            [
                f"## Prompt {state.prompt_index} / Turn {turn_index} Tool Result",
                "",
                f"- tool: `{tool_call.name}`",
                f"- ok: `{'true' if tool_result.ok else 'false'}`",
                f"- approved: `{'true' if tool_result.approved else 'false'}`",
                f"- status: `{tool_result.meta.get('status', '')}`",
                "- raw_output:",
                "```text",
                tool_result.output,
                "```",
                "- context_output:",
                "```text",
                tool_result.context_output_text,
                "```",
                "",
            ]
        )

    def record_condense(self, *, turn_index: int, reason: str, result: str, report: dict[str, Any] | None) -> None:
        state = self._turns.get(turn_index)
        if state is None:
            return
        state.condense_reason = reason
        state.condense_result = result
        state.condense_report = report
        self._append_markdown(
            [
                f"## Prompt {state.prompt_index} / Turn {turn_index} Condense",
                "",
                f"- reason: `{reason}`",
                "```text",
                result,
                "```",
                "",
            ]
        )

    def record_error(self, *, turn_index: int, error: str) -> None:
        state = self._turns.get(turn_index)
        if state is None:
            return
        state.error = error

    def finish_turn(
        self,
        *,
        turn_index: int,
        status: str,
        conversation_messages_after_turn: list[dict[str, Any]],
        extra: dict[str, Any] | None = None,
    ) -> str:
        state = self._turns.get(turn_index)
        if state is None:
            raise KeyError(f"Unknown turn {turn_index}")
        state.status = status
        if extra:
            state.extra.update(extra)
        if state.assistant_text.strip():
            self._append_markdown(
                [
                    f"## Prompt {state.prompt_index} / Turn {turn_index} Assistant",
                    "",
                    state.assistant_text,
                    "",
                ]
            )
        if state.error:
            self._append_markdown(
                [
                    f"## Prompt {state.prompt_index} / Turn {turn_index} Error",
                    "",
                    state.error,
                    "",
                ]
            )
        payload = {
            "schema_version": 1,
            "turn_index": state.turn_index,
            "prompt_index": state.prompt_index,
            "user_prompt": state.user_prompt,
            "mode": state.mode,
            "active_step": state.active_step,
            "status": state.status,
            "assistant_text": state.assistant_text,
            "assistant_chunks": list(state.assistant_chunks),
            "tool_calls": list(state.tool_calls),
            "tool_results": list(state.tool_results),
            "condense_reason": state.condense_reason,
            "condense_result": state.condense_result,
            "condense_report": state.condense_report,
            "runtime_request": state.runtime_request,
            "conversation_messages_after_turn": conversation_messages_after_turn,
            "error": state.error,
            "extra": state.extra,
        }
        target = self.turns_dir / f"turn_{turn_index:03d}.json"
        target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        path_text = str(target)
        if path_text not in self.turn_state_paths:
            self.turn_state_paths.append(path_text)
        return path_text

    def latest_turn_payload(self) -> dict[str, Any] | None:
        if not self._turns:
            return None
        turn_index = max(self._turns)
        state = self._turns.get(turn_index)
        if state is None:
            return None
        return build_recorded_turn_payload(
            user_prompt=state.user_prompt,
            assistant_text=state.assistant_text,
            tool_calls=list(state.tool_calls),
            tool_results=list(state.tool_results),
            extra={
                "turn_index": state.turn_index,
                "prompt_index": state.prompt_index,
                "status": state.status,
            },
        )

    def _append_markdown(self, lines: list[str]) -> None:
        with self.chat_log_path.open("a", encoding="utf-8") as handle:
            handle.write("\n".join(lines).rstrip() + "\n")


class OpenJetSession:
    def __init__(
        self,
        agent: Agent,
        *,
        approval_handler: ApprovalHandler | None = None,
        allowed_tools: set[str] | None = None,
        harness_state_getter: Callable[[], HarnessState | None] | None = None,
        harness_state_setter: Callable[[HarnessState], None] | None = None,
        airgapped: bool = False,
        session_logger: "SessionLogger | None" = None,
    ) -> None:
        self.agent = agent
        self._approval_handler = approval_handler
        self._allowed_tools = allowed_tools
        self._harness_state_getter = harness_state_getter
        self._harness_state_setter = harness_state_setter
        self.airgapped = bool(airgapped)
        self.session_logger = session_logger
        self._artifacts = _SessionArtifactRecorder(session_logger.session_dir) if session_logger else None
        set_airgapped(self.airgapped)

    @classmethod
    async def create(
        cls,
        *,
        cfg: dict | None = None,
        system_prompt: str | None = None,
        root: Path | None = None,
        approval_handler: ApprovalHandler | None = None,
        allowed_tools: set[str] | None = None,
        harness_state_getter: Callable[[], HarnessState | None] | None = None,
        harness_state_setter: Callable[[HarnessState], None] | None = None,
        airgapped: bool | None = None,
    ) -> OpenJetSession:
        from . import build_system_prompt, create_runtime_client

        resolved_cfg = dict(cfg or load_config())
        resolved_root = Path(root or Path.cwd()).resolve()
        resolved_cfg["airgapped"] = airgapped_from_cfg(resolved_cfg, override=airgapped)
        set_airgapped(bool(resolved_cfg["airgapped"]))
        client = create_runtime_client(resolved_cfg)
        mem_cfg = resolved_cfg.get("memory_guard", {})
        agent = Agent(
            client=client,
            system_prompt=await build_system_prompt(
                system_prompt or "",
                resolved_root,
                cfg=resolved_cfg,
            ),
            base_system_prompt=system_prompt or "",
            project_root=resolved_root,
            prompt_cfg=resolved_cfg,
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
            harness_state_getter=harness_state_getter,
            harness_state_setter=harness_state_setter,
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
        async for event in self.stream_existing_turn():
            yield event

    @property
    def chat_log_path(self) -> str:
        return str(self._artifacts.chat_log_path) if self._artifacts else ""

    @property
    def turn_state_paths(self) -> list[str]:
        return list(self._artifacts.turn_state_paths) if self._artifacts else []

    @property
    def session_dir(self) -> str:
        return str(self.session_logger.session_dir) if self.session_logger else ""

    @property
    def session_manifest_path(self) -> str:
        return str(self.session_logger.manifest_path) if self.session_logger else ""

    def record_user_prompt(self, *, prompt_index: int, text: str, mode: str | None = None) -> None:
        if self._artifacts:
            self._artifacts.record_user_prompt(prompt_index=prompt_index, text=text)

    def begin_turn_trace(
        self,
        *,
        turn_index: int,
        prompt_index: int,
        prompt_text: str,
        mode: str | None,
        active_step: str | None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        if not self._artifacts:
            return
        self._artifacts.start_turn(
            turn_index=turn_index,
            prompt_index=prompt_index,
            user_prompt=prompt_text,
            mode=mode,
            active_step=active_step,
            runtime_request=self.agent.runtime_request_snapshot(),
            extra=extra,
        )

    def record_text_chunk(self, *, turn_index: int, text: str) -> None:
        if self._artifacts:
            self._artifacts.record_text_chunk(turn_index=turn_index, text=text)

    def record_tool_request(self, *, turn_index: int, tool_call: ToolCall) -> None:
        if self._artifacts:
            self._artifacts.record_tool_request(turn_index=turn_index, tool_call=tool_call)

    def record_turn_error(self, *, turn_index: int, error: str) -> None:
        if self._artifacts:
            self._artifacts.record_error(turn_index=turn_index, error=error)

    def finish_turn_trace(self, *, turn_index: int, status: str, extra: dict[str, Any] | None = None) -> str:
        if not self._artifacts:
            return ""
        return self._artifacts.finish_turn(
            turn_index=turn_index,
            status=status,
            conversation_messages_after_turn=[dict(message) for message in self.agent.messages],
            extra=extra,
        )

    def record_condense_result(self, *, turn_index: int, reason: str, result: str) -> None:
        if not self._artifacts:
            return
        report = self.agent.last_condense_report()
        self._artifacts.record_condense(
            turn_index=turn_index,
            reason=reason,
            result=result,
            report=None if report is None else {
                "original_messages": report.original_messages,
                "final_messages": report.final_messages,
                "total_before_tokens": report.total_before_tokens,
                "total_after_tokens": report.total_after_tokens,
                "target_tokens": report.target_tokens,
                "summary_tokens": report.summary_tokens,
                "tighter_summary_used": report.tighter_summary_used,
                "kept_latest_user": report.kept_latest_user,
                "forced": report.forced,
            },
        )

    async def stream_existing_turn(self):
        completion_gate_retry_used = False
        while True:
            pending_tool_calls: list[ToolCall] = []
            async for event in self.agent.run_turn():
                if event.kind == ActionKind.TEXT:
                    yield SDKEvent(kind=SDKEventKind.TEXT, text=event.text)
                    continue
                if event.kind == ActionKind.REASONING:
                    yield SDKEvent(kind=SDKEventKind.REASONING, text=event.text)
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
                    gate_message = self._completion_gate_message()
                    if gate_message:
                        if completion_gate_retry_used:
                            yield SDKEvent(
                                kind=SDKEventKind.ERROR,
                                text=gate_message,
                            )
                            return
                        completion_gate_retry_used = True
                        self.agent.turn_context_messages.append({"role": "system", "content": gate_message})
                        break
                    try:
                        recorded_turn = self._artifacts.latest_turn_payload() if self._artifacts else None
                        report = await reflect_agent_persistent_memory(
                            self.agent,
                            recorded_turn=recorded_turn,
                        )
                    except Exception as exc:
                        yield SDKEvent(
                            kind=SDKEventKind.ERROR,
                            text=f"Persistent memory update failed: {exc}",
                        )
                        return
                    if isinstance(report, dict) and not bool(report.get("ok", True)):
                        reason = str(report.get("reason", "unknown failure")).strip() or "unknown failure"
                        yield SDKEvent(
                            kind=SDKEventKind.ERROR,
                            text=f"Persistent memory update failed: {reason}",
                        )
                        return
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
                meta={"ok": False, "denied": True, "status": "disallowed"},
                approved=False,
            )

        state = self._current_harness_state()
        if state is not None:
            gate_message = pre_edit_gate_message(state, tool_name=tool_call.name)
            if gate_message:
                self.agent.complete_tool_call(tool_call, gate_message)
                return ToolResult(
                    tool_call=tool_call,
                    output=gate_message,
                    meta={"ok": False, "denied": True, "status": "blocked_by_harness"},
                    approved=False,
                    context_output=gate_message,
                    context_output_text=gate_message,
                )

        if tool_call.name == "exit_plan_mode":
            result = await self._handle_exit_plan_mode(tool_call)
            if result is not None:
                self.agent.complete_tool_call(tool_call, result.context_output or result.output)
            return result

        control_result = self._handle_control_tool(tool_call)
        if control_result is not None:
            self.agent.complete_tool_call(tool_call, control_result.context_output or control_result.output)
            return control_result

        needs_confirmation = getattr(self.agent, "needs_confirmation", None)
        if callable(needs_confirmation) and needs_confirmation(tool_call):
            approved = await self._approve(tool_call)
            if not approved:
                output = "User denied this action."
                self.agent.complete_tool_call(tool_call, output)
                return ToolResult(
                    tool_call=tool_call,
                    output=output,
                    meta={"ok": False, "denied": True, "status": "denied"},
                    approved=False,
                )

        if tool_call.name == "load_file":
            self._clamp_load_file_tool_budget(tool_call)

        t0 = time.monotonic()
        try:
            result = await execute_tool(tool_call)
        except Exception as exc:
            output = f"Tool execution failed: {exc}"
            self.agent.complete_tool_call(tool_call, output)
            return ToolResult(
                tool_call=tool_call,
                output=output,
                meta={
                    "ok": False,
                    "status": "exception",
                    "error": str(exc).strip() or type(exc).__name__,
                    "duration_ms": round((time.monotonic() - t0) * 1000.0, 2),
                },
                approved=True,
            )

        if result.meta.get("swapped") and not result.meta.get("swap_restore_ok", True):
            try:
                await self.agent.client.reset_kv_cache()
            except Exception:
                pass
        if tool_call.name == "memory" and result.ok:
            try:
                await refresh_agent_system_prompt(self.agent)
            except Exception:
                pass

        context_output, output_truncated = self._fit_tool_result_content_to_budget(result)
        self.agent.complete_tool_call(tool_call, context_output)
        context_output_text = content_to_plain_text(context_output).strip() if context_output else ""
        result_meta = dict(result.meta)
        result_meta.setdefault("ok", bool(result.ok))
        result_meta.setdefault("status", "completed" if result.ok else "failed")
        result_meta["duration_ms"] = round((time.monotonic() - t0) * 1000.0, 2)
        result_meta["output_truncated"] = output_truncated
        if bool(result_meta.get("internal_retry")):
            return None
        return ToolResult(
            tool_call=tool_call,
            output=result.output,
            meta=result_meta,
            approved=True,
            context_output=context_output,
            context_output_text=context_output_text,
        )

    def _current_harness_state(self) -> HarnessState | None:
        if self._harness_state_getter is None:
            return None
        try:
            state = self._harness_state_getter()
        except Exception:
            return None
        return state if isinstance(state, HarnessState) else None

    def _set_harness_state(self, state: HarnessState) -> None:
        if self._harness_state_setter is None:
            return
        self._harness_state_setter(state)

    def _handle_control_tool(self, tool_call: ToolCall) -> ToolResult | None:
        state = self._current_harness_state()
        if state is None:
            return None
        if not isinstance(tool_call.arguments, dict):
            return None
        if tool_call.name == "todo_write":
            todos = tool_call.arguments.get("todos")
            if not isinstance(todos, list):
                output = "todo_write requires a todos array."
                return ToolResult(
                    tool_call=tool_call,
                    output=output,
                    meta={"ok": False, "status": "invalid"},
                    approved=True,
                    context_output=output,
                    context_output_text=output,
                )
            try:
                updated = upsert_todos(state, todos)
            except Exception as exc:
                output = f"todo_write failed: {exc}"
                return ToolResult(
                    tool_call=tool_call,
                    output=output,
                    meta={"ok": False, "status": "invalid"},
                    approved=True,
                    context_output=output,
                    context_output_text=output,
                )
            self._set_harness_state(updated)
            output = "Todo ledger updated."
            return ToolResult(
                tool_call=tool_call,
                output=output,
                meta={"ok": True, "status": "completed"},
                approved=True,
                context_output=output,
                context_output_text=output,
            )
        if tool_call.name == "todo_complete":
            try:
                updated = complete_todo(state, str(tool_call.arguments.get("id", "")))
            except Exception as exc:
                output = f"todo_complete failed: {exc}"
                return ToolResult(
                    tool_call=tool_call,
                    output=output,
                    meta={"ok": False, "status": "invalid"},
                    approved=True,
                    context_output=output,
                    context_output_text=output,
                )
            self._set_harness_state(updated)
            output = "Todo marked completed."
            return ToolResult(
                tool_call=tool_call,
                output=output,
                meta={"ok": True, "status": "completed"},
                approved=True,
                context_output=output,
                context_output_text=output,
            )
        if tool_call.name == "todo_clear":
            updated = clear_todos(state)
            self._set_harness_state(updated)
            output = "Todo list cleared."
            return ToolResult(
                tool_call=tool_call,
                output=output,
                meta={"ok": True, "status": "completed"},
                approved=True,
                context_output=output,
                context_output_text=output,
            )
        if tool_call.name == "verify_skip":
            try:
                updated = record_verification_skip(
                    state,
                    reason=str(tool_call.arguments.get("reason", "")),
                    next_command=str(tool_call.arguments.get("next_command", "")),
                )
            except Exception as exc:
                output = f"verify_skip failed: {exc}"
                return ToolResult(
                    tool_call=tool_call,
                    output=output,
                    meta={"ok": False, "status": "invalid"},
                    approved=True,
                    context_output=output,
                    context_output_text=output,
                )
            self._set_harness_state(updated)
            output = "Verification skip recorded."
            return ToolResult(
                tool_call=tool_call,
                output=output,
                meta={"ok": True, "status": "completed"},
                approved=True,
                context_output=output,
                context_output_text=output,
            )
        return None

    async def _handle_exit_plan_mode(self, tool_call: ToolCall) -> ToolResult | None:
        state = self._current_harness_state()
        if state is None or not isinstance(tool_call.arguments, dict):
            return None
        summary = str(tool_call.arguments.get("plan_summary", ""))
        approved = False
        status = "pending_approval"
        if self._approval_handler is not None:
            approved = await self._approve(tool_call)
            status = "approved" if approved else "rejected"
        try:
            updated = harness_exit_plan_mode(
                state,
                plan_summary=summary,
                approved=approved,
            )
        except Exception as exc:
            output = f"exit_plan_mode failed: {exc}"
            return ToolResult(
                tool_call=tool_call,
                output=output,
                meta={"ok": False, "status": "invalid"},
                approved=True,
                context_output=output,
                context_output_text=output,
            )
        self._set_harness_state(updated)
        if approved:
            output = "Plan approved. Plan mode exited and edit tools are available."
        elif self._approval_handler is not None:
            output = "Plan was not approved. Remaining in plan mode."
        else:
            output = "Plan summary recorded. Approval is still required before edits."
        return ToolResult(
            tool_call=tool_call,
            output=output,
            meta={"ok": True, "status": status, "plan_approved": approved},
            approved=approved if self._approval_handler is not None else True,
            context_output=output,
            context_output_text=output,
        )

    def _completion_gate_message(self) -> str | None:
        state = self._current_harness_state()
        if state is None:
            return None
        return verification_gate_message(state)

    async def handle_tool_call(self, tool_call: ToolCall, *, turn_index: int | None = None) -> ToolResult | None:
        tool_result = await self._handle_tool_call(tool_call)
        if tool_result is not None and self._artifacts and turn_index is not None:
            self._artifacts.record_tool_result(
                turn_index=turn_index,
                tool_call=tool_call,
                tool_result=tool_result,
            )
        return tool_result

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
        if remaining <= 0:
            tool_call.arguments["max_tokens"] = 0
            return
        current = tool_call.arguments.get("max_tokens")
        if not isinstance(current, int):
            tool_call.arguments["max_tokens"] = remaining
            return
        tool_call.arguments["max_tokens"] = min(current, remaining)

    def _remaining_prompt_tokens(self, *, reserve_next_turn_overhead: bool = False) -> int:
        estimate_context = getattr(self.agent, "estimated_context_tokens", None)
        if callable(estimate_context):
            current = int(estimate_context())
        else:
            persistent_context = getattr(self.agent, "persistent_context_tokens", None)
            current = int(persistent_context()) if callable(persistent_context) else 0
        context_budget = getattr(self.agent, "context_budget", None)
        budget = context_budget() if callable(context_budget) else None
        if not budget:
            window = int(getattr(self.agent, "context_window_tokens", 2048) or 2048)
            budget = derive_context_budget(window)
        runtime_overhead = 0
        if reserve_next_turn_overhead:
            overhead_fn = getattr(self.agent, "runtime_overhead_tokens", None)
            if callable(overhead_fn):
                try:
                    runtime_overhead = int(overhead_fn(force_post_tool_continuation=True))
                except TypeError:
                    runtime_overhead = int(overhead_fn())
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

    def _fit_tool_result_content_to_budget(self, result: ToolExecutionResult) -> tuple[object, bool]:
        content = result.context_content
        if content is None:
            fitted = self._fit_tool_result_to_budget(result.output)
            return fitted, fitted != result.output
        if isinstance(content, str):
            fitted = self._fit_tool_result_to_budget(content)
            return fitted, fitted != content

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
        return build_user_content(text, image_paths or None), text != content_to_plain_text(content)


async def create_agent(
    *,
    cfg: dict | None = None,
    system_prompt: str | None = None,
    root: Path | None = None,
    approval_handler: ApprovalHandler | None = None,
    allowed_tools: set[str] | None = None,
    airgapped: bool | None = None,
) -> OpenJetSession:
    return await OpenJetSession.create(
        cfg=cfg,
        system_prompt=system_prompt,
        root=root,
        approval_handler=approval_handler,
        allowed_tools=allowed_tools,
        airgapped=airgapped,
    )
