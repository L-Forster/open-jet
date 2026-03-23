"""Agent loop: manages conversation, calls runtime, handles tool proposals."""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass
from enum import Enum, auto
from typing import AsyncIterator, Callable

from .multimodal import content_to_plain_text, estimate_message_content_tokens, runtime_content
from .runtime_protocol import ToolCall, tool_schema_token_estimate
from .runtime_client import RuntimeClient
from .runtime_limits import ContextBudget, derive_context_budget, estimate_tokens, read_memory_snapshot


class ActionKind(Enum):
    TEXT = auto()         # plain text token to display
    CONDENSE = auto()     # internal request to condense context
    TOOL_REQUEST = auto() # model wants to run a tool — needs approval
    TOOL_RESULT = auto()  # result after tool execution
    DONE = auto()         # turn finished
    ERROR = auto()        # something went wrong
    UNLOAD = auto()       # signal that model should be unloaded for a heavy task


@dataclass
class AgentEvent:
    kind: ActionKind
    text: str = ""
    tool_call: ToolCall | None = None


@dataclass(frozen=True)
class CondenseReport:
    original_messages: int
    final_messages: int
    total_before_tokens: int
    total_after_tokens: int
    target_tokens: int
    summary_tokens: int
    tighter_summary_used: bool
    kept_latest_user: bool
    forced: bool


# Tools that require user confirmation before execution
CONFIRM_TOOLS = {"shell", "write_file", "edit_file", "memory"}
POST_TOOL_CONTINUATION_NOTE = (
    "Tool results are available in the conversation. Analyze them before deciding the next action. "
    "If they answer the user's request, answer directly. Otherwise request the next precise tool. "
    "Do not end the turn with an empty reply."
)
EMPTY_COMPLETION_RETRY_NOTE = (
    "Your previous completion for this turn was empty. Continue the same turn now. "
    "Either answer the user directly or request the next tool. Do not return an empty reply."
)
MAX_EMPTY_COMPLETION_RETRIES = 2
RUNTIME_REQUEST_SLACK_TOKENS = 128


class Agent:
    """Manages conversation history and drives the chat loop."""

    def __init__(
        self,
        client: RuntimeClient,
        system_prompt: str,
        *,
        context_window_tokens: int | None = None,
        context_reserved_tokens: int | None = None,
        min_prompt_tokens: int = 256,
        min_available_mb: int | None = None,
        max_used_percent: float | None = None,
        memory_check_interval_chunks: int = 16,
        condense_target_tokens: int = 900,
        keep_last_messages: int = 6,
        trace_hook: Callable[[str, dict[str, object]], None] | None = None,
    ) -> None:
        self.client = client
        self.system_prompt = system_prompt
        self.messages: list[dict] = [{"role": "system", "content": system_prompt}]
        self.context_window_tokens = context_window_tokens
        self.context_reserved_tokens = context_reserved_tokens
        self.min_prompt_tokens = max(128, int(min_prompt_tokens))
        self.min_available_mb = min_available_mb
        self.max_used_percent = max_used_percent
        self.memory_check_interval_chunks = max(1, int(memory_check_interval_chunks))
        self.condense_target_tokens = condense_target_tokens
        self.keep_last_messages = keep_last_messages
        self.turn_context_messages: list[dict] = []
        self.trace_hook = trace_hook
        self._last_condense_report: CondenseReport | None = None

    def add_user_message(self, text: str, *, image_paths: list[str] | None = None) -> None:
        from .multimodal import build_user_content

        self.messages.append({"role": "user", "content": build_user_content(text, image_paths)})

    def reset_conversation(self) -> None:
        self.messages = [{"role": "system", "content": self.system_prompt}]

    def conversation_message_count(self) -> int:
        return max(0, len(self.messages) - 1)

    def estimated_context_tokens(self) -> int:
        return self._estimated_context_tokens(include_turn_context=True)

    def persistent_context_tokens(self) -> int:
        return self._estimated_context_tokens(include_turn_context=False)

    def context_budget(self) -> ContextBudget | None:
        if not self.context_window_tokens:
            return None
        return derive_context_budget(
            self.context_window_tokens,
            reserve_tokens=self.context_reserved_tokens,
            min_prompt_tokens=self.min_prompt_tokens,
        )

    async def run_turn(self) -> AsyncIterator[AgentEvent]:
        """Run one model turn. Yields events the TUI should handle.

        When a CONDENSE event is yielded, the caller should run condense_context()
        and then call run_turn() again.
        When a TOOL_REQUEST event is yielded, the caller must ask for approval
        (if needed), call complete_tool_call() with the result, and then call
        run_turn() again so the model can continue.
        """
        self._trace(
            "run_turn_start",
            message_count=len(self.messages),
            turn_context_count=len(self.turn_context_messages),
            estimated_tokens=self._estimated_context_tokens(include_turn_context=True),
            runtime_overhead_tokens=self.runtime_overhead_tokens(),
        )
        pressure_reason = self._resource_pressure_reason()
        if pressure_reason:
            self._trace("run_turn_condense_before_stream", reason=pressure_reason)
            yield AgentEvent(
                kind=ActionKind.TEXT,
                text=f"(resource pressure, condensing chat: {pressure_reason})",
            )
            yield AgentEvent(kind=ActionKind.CONDENSE, text=pressure_reason)
            return

        empty_retry_count = 0
        while True:
            collected_text = ""
            pending_tool_calls: list[ToolCall] = []
            chunk_count = 0
            extra_system_note = self._continuation_note_for_runtime(empty_retry_count)
            runtime_messages = self._messages_for_runtime(extra_system_note=extra_system_note)
            prompt_tokens = self._estimate_runtime_request_tokens(runtime_messages, use_tools=True)
            self._trace(
                "runtime_request",
                request_kind="turn",
                prompt_tokens=prompt_tokens,
                use_tools=True,
                message_count=len(runtime_messages),
                empty_retry_count=empty_retry_count,
            )

            try:
                async for chunk in self.client.chat_stream(runtime_messages):
                    chunk_count += 1
                    if chunk_count == 1:
                        self._trace(
                            "stream_first_chunk",
                            text_len=len(chunk.text),
                            tool_call_count=len(chunk.tool_calls),
                            done=chunk.done,
                            empty_retry_count=empty_retry_count,
                            has_extra_system_note=bool(extra_system_note),
                        )
                    if chunk.text:
                        collected_text += chunk.text
                        yield AgentEvent(kind=ActionKind.TEXT, text=chunk.text)

                    if chunk.tool_calls:
                        pending_tool_calls.extend(chunk.tool_calls)

                    if chunk_count % self.memory_check_interval_chunks == 0:
                        pressure_reason = self._resource_pressure_reason()
                        if pressure_reason:
                            if collected_text:
                                self.messages.append({"role": "assistant", "content": collected_text})
                            self._trace(
                                "run_turn_condense_mid_stream",
                                reason=pressure_reason,
                                chunk_count=chunk_count,
                                collected_text_len=len(collected_text),
                            )
                            yield AgentEvent(
                                kind=ActionKind.TEXT,
                                text=(
                                    "\n(resource pressure, condensing chat: "
                                    f"{pressure_reason})"
                                ),
                            )
                            yield AgentEvent(kind=ActionKind.CONDENSE, text=pressure_reason)
                            return

            except Exception as e:
                self._trace(
                    "run_turn_error",
                    error=str(e),
                    chunk_count=chunk_count,
                    collected_text_len=len(collected_text),
                )
                yield AgentEvent(kind=ActionKind.ERROR, text=str(e))
                return

            assistant_text = collected_text.strip()
            if pending_tool_calls:
                assistant_msg: dict = {"role": "assistant", "content": collected_text}
                for tc in pending_tool_calls:
                    if not tc.id:
                        tc.id = f"call_{uuid.uuid4().hex[:12]}"
                assistant_msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments, ensure_ascii=True),
                        }
                    }
                    for tc in pending_tool_calls
                ]
                self.messages.append(assistant_msg)
                completion_tokens = self._estimate_runtime_output_tokens(
                    text=collected_text,
                    tool_calls=assistant_msg["tool_calls"],
                )
                self._trace(
                    "runtime_exchange_complete",
                    request_kind="turn",
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    use_tools=True,
                    message_count=len(runtime_messages),
                    tool_call_count=len(pending_tool_calls),
                    empty_retry_count=empty_retry_count,
                )
                self._trace(
                    "run_turn_complete",
                    chunk_count=chunk_count,
                    collected_text_len=len(collected_text),
                    pending_tool_call_count=len(pending_tool_calls),
                )
                for tc in pending_tool_calls:
                    yield AgentEvent(kind=ActionKind.TOOL_REQUEST, tool_call=tc)
                return

            if assistant_text:
                self.messages.append({"role": "assistant", "content": collected_text})
                completion_tokens = self._estimate_runtime_output_tokens(text=collected_text)
                self._trace(
                    "runtime_exchange_complete",
                    request_kind="turn",
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    use_tools=True,
                    message_count=len(runtime_messages),
                    tool_call_count=0,
                    empty_retry_count=empty_retry_count,
                )
                self._trace(
                    "run_turn_complete",
                    chunk_count=chunk_count,
                    collected_text_len=len(collected_text),
                    pending_tool_call_count=0,
                )
                self._trace("run_turn_done", assistant_text_len=len(collected_text))
                yield AgentEvent(kind=ActionKind.DONE)
                return

            empty_retry_count += 1
            self._trace(
                "run_turn_empty_completion",
                empty_retry_count=empty_retry_count,
                chunk_count=chunk_count,
                had_tool_context=self._last_non_system_role() == "tool",
            )
            if empty_retry_count > MAX_EMPTY_COMPLETION_RETRIES:
                error = (
                    "Model returned an empty completion after tool analysis. "
                    "Turn aborted instead of silently ending."
                )
                self._trace(
                    "run_turn_error",
                    error=error,
                    chunk_count=chunk_count,
                    collected_text_len=len(collected_text),
                )
                yield AgentEvent(kind=ActionKind.ERROR, text=error)
                return

    def _trace(self, event: str, **data: object) -> None:
        if not self.trace_hook:
            return
        try:
            self.trace_hook(event, dict(data))
        except Exception:
            return

    def complete_tool_call(self, tool_call: ToolCall, result: object) -> None:
        """Record a tool result in conversation history so the model sees it."""
        msg = {"role": "tool", "content": result}
        if tool_call.id:
            msg["tool_call_id"] = tool_call.id
        self.messages.append(msg)

    def needs_confirmation(self, tool_call: ToolCall) -> bool:
        """Return True if this tool call requires user approval."""
        return tool_call.name in CONFIRM_TOOLS

    def last_condense_report(self) -> CondenseReport | None:
        return self._last_condense_report

    async def condense_context(self, *, force: bool = False) -> str:
        """Condense message history with an LLM-generated summary."""
        self._last_condense_report = None
        if len(self.messages) <= 1:
            return "No message history to condense."

        total_before = self._estimated_context_tokens(include_turn_context=True)
        original_messages = len(self.messages)
        history = self.messages[1:]
        transcript = self._history_as_text(history)
        provenance_note = self._build_condense_provenance(history)
        provenance_tokens = estimate_tokens(provenance_note) if provenance_note else 0
        target_cap = max(96, self.condense_target_tokens - min(provenance_tokens, max(96, self.condense_target_tokens // 2)))
        target_tokens = max(96, min(target_cap, total_before // 2))
        latest_user = self._latest_user_message()
        try:
            summary = await self._summarize_text(transcript, target_tokens=target_tokens)
        except Exception as exc:
            self._trace("condense_context_failed", error=str(exc), target_tokens=target_tokens, forced=force)
            return f"Condense failed: {exc}"
        if not summary:
            self._trace("condense_context_failed", error="empty summary", target_tokens=target_tokens, forced=force)
            return "Condense failed: model returned empty summary."

        summary_tokens = estimate_tokens(summary)
        summary_msg = {"role": "system", "content": self._compose_condensed_context(summary, provenance_note)}
        self.messages = [self.messages[0], summary_msg]
        if latest_user is not None:
            self.messages.append(latest_user)

        total_after = self._estimated_context_tokens(include_turn_context=True)
        tighter_summary_used = False
        if total_after >= total_before:
            tighter_target = max(64, target_tokens // 2)
            try:
                tighter_summary = await self._summarize_text(summary, target_tokens=tighter_target)
            except Exception:
                tighter_summary = ""
            if tighter_summary:
                tighter_summary_used = True
                summary_tokens = estimate_tokens(tighter_summary)
                self.messages = [
                    self.messages[0],
                    {"role": "system", "content": self._compose_condensed_context(tighter_summary, provenance_note)},
                ]
                if latest_user is not None:
                    self.messages.append(latest_user)
                total_after = self._estimated_context_tokens(include_turn_context=True)

        report = CondenseReport(
            original_messages=original_messages,
            final_messages=len(self.messages),
            total_before_tokens=total_before,
            total_after_tokens=total_after,
            target_tokens=target_tokens,
            summary_tokens=summary_tokens,
            tighter_summary_used=tighter_summary_used,
            kept_latest_user=latest_user is not None,
            forced=force,
        )
        self._last_condense_report = report
        self._trace(
            "condense_context_complete",
            original_messages=report.original_messages,
            final_messages=report.final_messages,
            total_before_tokens=report.total_before_tokens,
            total_after_tokens=report.total_after_tokens,
            target_tokens=report.target_tokens,
            summary_tokens=report.summary_tokens,
            tighter_summary_used=report.tighter_summary_used,
            kept_latest_user=report.kept_latest_user,
            forced=report.forced,
        )
        return (
            "Context condensed automatically. "
            f"messages: {report.original_messages} -> {report.final_messages}, "
            f"tokens(est): {report.total_before_tokens} -> {report.total_after_tokens}, "
            f"target<={report.target_tokens}t, summary~{report.summary_tokens}t, "
            f"kept_latest_user={'yes' if report.kept_latest_user else 'no'}."
        )

    def set_turn_context(self, messages: list[dict]) -> None:
        self.turn_context_messages = [msg for msg in messages if isinstance(msg, dict)]

    def clear_turn_context(self) -> None:
        self.turn_context_messages = []

    def _messages_for_runtime(self, extra_system_note: str | None = None) -> list[dict]:
        combined = self.messages + self.turn_context_messages
        system_parts: list[str] = []
        runtime_messages: list[dict] = []

        for msg in combined:
            if not isinstance(msg, dict):
                continue
            role = str(msg.get("role", ""))
            if role == "system":
                content = content_to_plain_text(msg.get("content", "")).strip()
                if content:
                    system_parts.append(content)
                continue
            runtime_msg = dict(msg)
            runtime_msg["content"] = runtime_content(msg.get("content", ""))
            runtime_messages.append(runtime_msg)

        extra_system_note = (extra_system_note or "").strip()
        if extra_system_note:
            system_parts.append(extra_system_note)

        if not system_parts:
            return runtime_messages

        merged_system = {"role": "system", "content": "\n\n".join(system_parts)}
        return [merged_system, *runtime_messages]

    def runtime_overhead_tokens(
        self,
        *,
        empty_retry_count: int = 0,
        force_post_tool_continuation: bool = False,
    ) -> int:
        total = tool_schema_token_estimate() + RUNTIME_REQUEST_SLACK_TOKENS
        continuation_note = self._continuation_note_for_runtime(empty_retry_count)
        if force_post_tool_continuation and POST_TOOL_CONTINUATION_NOTE not in continuation_note:
            continuation_note = (
                f"{continuation_note}\n\n{POST_TOOL_CONTINUATION_NOTE}".strip()
                if continuation_note
                else POST_TOOL_CONTINUATION_NOTE
            )
        if continuation_note:
            total += estimate_tokens(continuation_note) + 8
        return total

    def _continuation_note_for_runtime(self, empty_retry_count: int) -> str:
        notes: list[str] = []
        if self._last_non_system_role() == "tool":
            notes.append(POST_TOOL_CONTINUATION_NOTE)
        if empty_retry_count > 0:
            notes.append(EMPTY_COMPLETION_RETRY_NOTE)
        return "\n\n".join(notes)

    def _last_non_system_role(self) -> str | None:
        for msg in reversed(self.messages):
            role = str(msg.get("role", ""))
            if role and role != "system":
                return role
        return None

    def _latest_user_message(self) -> dict | None:
        for msg in reversed(self.messages):
            if str(msg.get("role", "")) != "user":
                continue
            return {"role": "user", "content": msg.get("content", "")}
        return None

    def _estimated_context_tokens(self, *, include_turn_context: bool = True) -> int:
        total = 0
        message_sets = [self.messages]
        if include_turn_context:
            message_sets.append(self.turn_context_messages)
        for batch in message_sets:
            for msg in batch:
                total += estimate_message_content_tokens(msg.get("content", "")) + 8
                tool_calls = msg.get("tool_calls")
                if isinstance(tool_calls, list) and tool_calls:
                    total += self._estimate_tool_call_tokens(tool_calls)
        return total

    def _estimate_tool_call_tokens(self, tool_calls: list[dict]) -> int:
        try:
            payload = json.dumps(tool_calls, ensure_ascii=True, separators=(",", ":"))
        except (TypeError, ValueError):
            payload = str(tool_calls)
        return estimate_tokens(payload)

    def _estimate_runtime_request_tokens(self, messages: list[dict], *, use_tools: bool) -> int:
        total = 0
        for msg in messages:
            total += estimate_message_content_tokens(msg.get("content", "")) + 8
            tool_calls = msg.get("tool_calls")
            if isinstance(tool_calls, list) and tool_calls:
                total += self._estimate_tool_call_tokens(tool_calls)
        if use_tools:
            total += tool_schema_token_estimate()
        return total

    def _estimate_runtime_output_tokens(self, *, text: str, tool_calls: list[dict] | None = None) -> int:
        total = estimate_tokens(text)
        if tool_calls:
            total += self._estimate_tool_call_tokens(tool_calls)
        return total

    def _resource_pressure_reason(self) -> str | None:
        context_pressure = self._context_pressure_reason()
        if context_pressure:
            return context_pressure
        return self._memory_pressure_reason()

    def resource_pressure_reason(self) -> str | None:
        return self._resource_pressure_reason()

    def _context_pressure_reason(self) -> str | None:
        budget = self.context_budget()
        if not budget:
            return None
        current = self._estimated_context_tokens(include_turn_context=True) + self.runtime_overhead_tokens()
        if current <= budget.prompt_tokens:
            return None
        return (
            f"context={current}t exceeds prompt budget "
            f"{budget.prompt_tokens}t/{budget.window_tokens}t "
            f"(reserve={budget.reserve_tokens}t)"
        )

    def _memory_pressure_reason(self) -> str | None:
        snapshot = read_memory_snapshot()
        if not snapshot:
            return None
        total_mb = snapshot.total_mb
        available_mb = snapshot.available_mb
        used_percent = snapshot.used_percent
        if total_mb is None or available_mb is None or used_percent is None:
            return None

        min_available_mb = self.min_available_mb
        if min_available_mb is None:
            min_available_mb = max(512, int(total_mb * 0.08))

        if available_mb < float(min_available_mb):
            return (
                f"mem_available={available_mb:.0f}MB < {min_available_mb}MB "
                f"(mem_used={used_percent:.1f}%)"
            )

        if self.max_used_percent is not None and used_percent > self.max_used_percent:
            return (
                f"mem_used={used_percent:.1f}% > {self.max_used_percent:.1f}% "
                f"(mem_available={available_mb:.0f}MB)"
            )
        return None

    def _history_as_text(self, older_messages: list[dict]) -> str:
        lines: list[str] = [
            "CONDENSE INPUT",
            "Preserve concrete findings, decisions, and exact source provenance from tools/files.",
        ]
        pending_tool_calls: list[tuple[str | None, str]] = []
        for msg in older_messages:
            role = str(msg.get("role", "unknown")).upper()
            if role == "ASSISTANT":
                text = self._compact_history_text(content_to_plain_text(msg.get("content", "")))
                if text:
                    lines.append(f"ASSISTANT: {text}")
                tool_calls = msg.get("tool_calls")
                if isinstance(tool_calls, list):
                    for raw_tool_call in tool_calls:
                        tool_id, descriptor = self._tool_call_descriptor(raw_tool_call)
                        pending_tool_calls.append((tool_id, descriptor))
                        lines.append(f"TOOL_CALL: {descriptor}")
                continue
            if role == "TOOL":
                descriptor = self._consume_pending_tool_descriptor(
                    pending_tool_calls,
                    str(msg.get("tool_call_id", "")).strip() or None,
                )
                lines.append(f"TOOL_RESULT: {descriptor}")
                for detail in self._tool_result_snapshot(content_to_plain_text(msg.get("content", "")), descriptor):
                    lines.append(f"  {detail}")
                continue
            text = self._compact_history_text(content_to_plain_text(msg.get("content", "")))
            if text:
                lines.append(f"{role}: {text}")
        return "\n".join(lines)

    def _compose_condensed_context(self, summary: str, provenance_note: str) -> str:
        parts = ["Condensed conversation context:"]
        if provenance_note:
            parts.append(provenance_note)
        parts.append("Working summary:")
        parts.append(summary)
        return "\n".join(parts)

    def _build_condense_provenance(self, older_messages: list[dict], *, max_items: int = 6) -> str:
        entries: list[str] = []
        pending_tool_calls: list[tuple[str | None, str]] = []
        for msg in older_messages:
            role = str(msg.get("role", "")).lower()
            if role == "assistant":
                tool_calls = msg.get("tool_calls")
                if isinstance(tool_calls, list):
                    for raw_tool_call in tool_calls:
                        pending_tool_calls.append(self._tool_call_descriptor(raw_tool_call))
                continue
            if role != "tool":
                continue
            descriptor = self._consume_pending_tool_descriptor(
                pending_tool_calls,
                str(msg.get("tool_call_id", "")).strip() or None,
            )
            snapshot = self._tool_result_snapshot(content_to_plain_text(msg.get("content", "")), descriptor)
            entry_lines = [f"- {descriptor}"]
            entry_lines.extend(f"  {line}" for line in snapshot)
            entries.append("\n".join(entry_lines))
        if not entries:
            return ""
        return "Source trail to preserve:\n" + "\n".join(entries[-max_items:])

    def _compact_history_text(self, text: str, *, max_len: int = 360) -> str:
        compact = " ".join(text.split())
        if len(compact) <= max_len:
            return compact
        head = compact[: max_len - 40].rstrip()
        tail = compact[-24:].lstrip()
        return f"{head} ... {tail}"

    def _tool_call_descriptor(self, raw_tool_call: dict) -> tuple[str | None, str]:
        if not isinstance(raw_tool_call, dict):
            return None, self._compact_history_text(str(raw_tool_call), max_len=120)
        tool_id = raw_tool_call.get("id")
        function = raw_tool_call.get("function") if isinstance(raw_tool_call.get("function"), dict) else {}
        name = str(function.get("name") or raw_tool_call.get("name") or "tool").strip() or "tool"
        raw_arguments = function.get("arguments", raw_tool_call.get("arguments", {}))
        arguments: dict | str
        if isinstance(raw_arguments, str):
            try:
                parsed = json.loads(raw_arguments)
            except Exception:
                parsed = raw_arguments
            arguments = parsed
        else:
            arguments = raw_arguments if isinstance(raw_arguments, dict) else str(raw_arguments)
        descriptor = self._format_tool_call_descriptor(name, arguments)
        return (str(tool_id).strip() or None, descriptor)

    def _format_tool_call_descriptor(self, name: str, arguments: dict | str) -> str:
        if isinstance(arguments, dict):
            if name in {"read_file", "write_file", "load_file", "edit_file"}:
                return f"{name} path={str(arguments.get('path', '')).strip() or '<unknown>'}"
            if name in {"camera_snapshot", "microphone_record", "sensor_read", "gpio_read"}:
                source = str(arguments.get("source", "")).strip() or "<auto>"
                if name == "microphone_record":
                    duration_seconds = arguments.get("duration_seconds")
                    if isinstance(duration_seconds, int):
                        return f"{name} source={source} duration={duration_seconds}s"
                return f"{name} source={source}"
            if name == "microphone_set_enabled":
                source = str(arguments.get("source", "")).strip() or "<auto>"
                enabled = arguments.get("enabled")
                return f"{name} source={source} enabled={enabled}"
            if name == "device_list":
                kind = str(arguments.get("kind", "")).strip() or "all"
                return f"{name} kind={kind}"
            if name == "grep":
                pattern = self._compact_history_text(str(arguments.get("pattern", "")).strip(), max_len=80)
                path = str(arguments.get("path", "")).strip()
                suffix = f" path={path}" if path else ""
                return f"{name} pattern={pattern}{suffix}"
            if name == "glob":
                pattern = self._compact_history_text(str(arguments.get("pattern", "")).strip(), max_len=80)
                path = str(arguments.get("path", "")).strip()
                suffix = f" path={path}" if path else ""
                return f"{name} pattern={pattern}{suffix}"
            if name == "list_directory":
                return f"{name} path={str(arguments.get('path', '.') or '.').strip()}"
            if name == "shell":
                command = self._compact_history_text(str(arguments.get("command", "")).strip(), max_len=100)
                return f"{name} command={command}"
            if name == "system_info":
                scope = str(arguments.get("scope", "summary") or "summary").strip()
                return f"{name} scope={scope}"
        return self._compact_history_text(f"{name} {arguments}", max_len=140)

    def _consume_pending_tool_descriptor(
        self,
        pending_tool_calls: list[tuple[str | None, str]],
        tool_call_id: str | None,
    ) -> str:
        if tool_call_id:
            for index, (candidate_id, descriptor) in enumerate(pending_tool_calls):
                if candidate_id == tool_call_id:
                    pending_tool_calls.pop(index)
                    return descriptor
        if pending_tool_calls:
            _, descriptor = pending_tool_calls.pop(0)
            return descriptor
        return "tool result"

    def _tool_result_snapshot(self, text: str, descriptor: str) -> list[str]:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not lines:
            return ["summary: empty tool result"]
        markers: list[str] = []
        lowered = text.lower()
        if "tool context limit reached" in lowered:
            markers.append("context-limited")
        if "tool output truncated" in lowered or "...[tool output truncated]" in lowered:
            markers.append("truncated")
        if "no matches for pattern" in lowered:
            markers.append("no-matches")
        if any(line.lower().startswith("error:") for line in lines):
            markers.append("error")

        summary_line = next((line for line in lines if line.lower() != "content:"), lines[0])
        details = [f"summary: {self._compact_history_text(summary_line, max_len=220)}"]
        file_paths = self._extract_paths_from_text(text)
        if not file_paths:
            file_paths = self._extract_paths_from_text(descriptor)
        if file_paths:
            details.append(f"files: {', '.join(file_paths[:6])}")
        detail_line = next(
            (
                line
                for line in lines[1:]
                if line != summary_line
                and line.lower() != "content:"
                and not line.startswith("[tool")
            ),
            "",
        )
        if detail_line:
            details.append(f"detail: {self._compact_history_text(detail_line, max_len=220)}")
        if markers:
            details.append(f"markers: {', '.join(markers)}")
        return details[:4]

    def _extract_paths_from_text(self, text: str, *, max_paths: int = 6) -> list[str]:
        seen: set[str] = set()
        found: list[str] = []
        path_pattern = re.compile(r"(?<!\w)(?:/[\w.\-]+(?:/[\w.\-]+)+|\.?/?[\w.\-]+(?:/[\w.\-]+)+)")
        for raw in path_pattern.findall(text):
            candidate = raw.strip().rstrip(",:;)]}")
            if not candidate or "/" not in candidate or candidate in seen:
                continue
            seen.add(candidate)
            found.append(candidate)
            if len(found) >= max_paths:
                break
        return found

    async def _summarize_text(self, text: str, *, target_tokens: int) -> str:
        prompt = (
            "Summarize the conversation history for future context.\n"
            f"Keep key facts, constraints, decisions, and learned findings. Target under {target_tokens} tokens.\n"
            "Use this exact structure:\n"
            "GOAL:\n"
            "KEY FINDINGS:\n"
            "- include concrete facts already learned\n"
            "- when a fact came from a tool or file, cite it inline like [source: read_file path=src/train.py]\n"
            "FILES / EVIDENCE:\n"
            "- list the important files, commands, or tool outputs already inspected\n"
            "OPEN QUESTIONS:\n"
            "NEXT ACTION:\n"
            "Do not restart exploration that has already happened. Preserve source provenance. Return only the summary."
        )
        messages = [
            {"role": "system", "content": "You summarize conversation context for an offline coding agent."},
            {"role": "user", "content": f"{prompt}\n\nConversation:\n{text}"},
        ]
        prompt_tokens = self._estimate_runtime_request_tokens(messages, use_tools=False)
        self._trace(
            "runtime_request",
            request_kind="condense_summary",
            prompt_tokens=prompt_tokens,
            use_tools=False,
            message_count=len(messages),
        )
        chunks: list[str] = []
        async for chunk in self.client.chat_stream(messages, use_tools=False):
            if chunk.text:
                chunks.append(chunk.text)
        summary = "".join(chunks).strip()
        self._trace(
            "runtime_exchange_complete",
            request_kind="condense_summary",
            prompt_tokens=prompt_tokens,
            completion_tokens=self._estimate_runtime_output_tokens(text=summary),
            use_tools=False,
            message_count=len(messages),
            tool_call_count=0,
        )
        return summary
