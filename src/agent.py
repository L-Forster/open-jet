"""Agent loop: manages conversation, calls llama-server, handles tool proposals."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from enum import Enum, auto
from typing import AsyncIterator

from .llama_server import LlamaServerClient, ToolCall
from .runtime_limits import ContextBudget, derive_context_budget, estimate_tokens, read_memory_snapshot


class ActionKind(Enum):
    TEXT = auto()         # plain text token to display
    CONDENSE = auto()     # internal request to condense context
    TOOL_REQUEST = auto() # model wants to run a tool — needs approval
    TOOL_RESULT = auto()  # result after tool execution
    DONE = auto()         # turn finished
    ERROR = auto()        # something went wrong


@dataclass
class AgentEvent:
    kind: ActionKind
    text: str = ""
    tool_call: ToolCall | None = None


# Tools that require user confirmation before execution
CONFIRM_TOOLS = {"shell", "write_file", "edit_file"}


class Agent:
    """Manages conversation history and drives the llama-server chat loop."""

    def __init__(
        self,
        client: LlamaServerClient,
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

    def add_user_message(self, text: str) -> None:
        self.messages.append({"role": "user", "content": text})

    def reset_conversation(self) -> None:
        self.messages = [{"role": "system", "content": self.system_prompt}]

    def conversation_message_count(self) -> int:
        return max(0, len(self.messages) - 1)

    def estimated_context_tokens(self) -> int:
        return self._estimated_context_tokens()

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
        pressure_reason = self._resource_pressure_reason()
        if pressure_reason:
            yield AgentEvent(
                kind=ActionKind.TEXT,
                text=f"(resource pressure, condensing chat: {pressure_reason})",
            )
            yield AgentEvent(kind=ActionKind.CONDENSE, text=pressure_reason)
            return

        collected_text = ""
        pending_tool_calls: list[ToolCall] = []
        chunk_count = 0

        try:
            async for chunk in self.client.chat_stream(self.messages):
                chunk_count += 1
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
            yield AgentEvent(kind=ActionKind.ERROR, text=str(e))
            return

        # Record assistant message in history
        assistant_msg: dict = {"role": "assistant", "content": collected_text}
        if pending_tool_calls:
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

        # Yield tool requests for the TUI to handle
        for tc in pending_tool_calls:
            yield AgentEvent(kind=ActionKind.TOOL_REQUEST, tool_call=tc)

        if not pending_tool_calls:
            yield AgentEvent(kind=ActionKind.DONE)

    def complete_tool_call(self, tool_call: ToolCall, result: str) -> None:
        """Record a tool result in conversation history so the model sees it."""
        msg = {"role": "tool", "content": result}
        if tool_call.id:
            msg["tool_call_id"] = tool_call.id
        self.messages.append(msg)

    def needs_confirmation(self, tool_call: ToolCall) -> bool:
        """Return True if this tool call requires user approval."""
        return tool_call.name in CONFIRM_TOOLS

    async def condense_context(self, *, force: bool = False) -> str:
        """Condense message history with an LLM-generated summary."""
        if len(self.messages) <= 1:
            return "No message history to condense."

        total_before = self._estimated_context_tokens()
        original_messages = len(self.messages)
        history = self.messages[1:]
        transcript = self._history_as_text(history)
        target_tokens = max(96, min(self.condense_target_tokens, total_before // 2))
        try:
            summary = await self._summarize_text(transcript, target_tokens=target_tokens)
        except Exception as exc:
            return f"Condense failed: {exc}"
        if not summary:
            return "Condense failed: model returned empty summary."

        summary_msg = {
            "role": "system",
            "content": (
                "Condensed conversation context:\n"
                f"{summary}"
            ),
        }
        self.messages = [self.messages[0], summary_msg]

        total_after = self._estimated_context_tokens()
        if total_after >= total_before:
            tighter_target = max(64, target_tokens // 2)
            try:
                tighter_summary = await self._summarize_text(summary, target_tokens=tighter_target)
            except Exception:
                tighter_summary = ""
            if tighter_summary:
                self.messages = [
                    self.messages[0],
                    {
                        "role": "system",
                        "content": (
                            "Condensed conversation context:\n"
                            f"{tighter_summary}"
                        ),
                    },
                ]
                total_after = self._estimated_context_tokens()

        return (
            "Context condensed automatically. "
            f"messages: {original_messages} -> {len(self.messages)}, "
            f"tokens(est): {total_before} -> {total_after}."
        )

    def _estimated_context_tokens(self) -> int:
        total = 0
        for msg in self.messages:
            content = str(msg.get("content", ""))
            total += estimate_tokens(content) + 8
        return total

    def _resource_pressure_reason(self) -> str | None:
        context_pressure = self._context_pressure_reason()
        if context_pressure:
            return context_pressure
        return self._memory_pressure_reason()

    def _context_pressure_reason(self) -> str | None:
        budget = self.context_budget()
        if not budget:
            return None
        current = self._estimated_context_tokens()
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
        lines: list[str] = []
        for msg in older_messages:
            role = str(msg.get("role", "unknown")).upper()
            text = " ".join(str(msg.get("content", "")).split())
            if text:
                lines.append(f"{role}: {text}")
        return "\n".join(lines)

    async def _summarize_text(self, text: str, *, target_tokens: int) -> str:
        prompt = (
            "Summarize the conversation history for future context.\n"
            f"Keep key facts, constraints, and decisions. Target under {target_tokens} tokens.\n"
            "Remove repetition and fluff. Return only the summary."
        )
        messages = [
            {"role": "system", "content": "You summarize conversation context for an offline coding agent."},
            {"role": "user", "content": f"{prompt}\n\nConversation:\n{text}"},
        ]
        chunks: list[str] = []
        async for chunk in self.client.chat_stream(messages, use_tools=False):
            if chunk.text:
                chunks.append(chunk.text)
        return "".join(chunks).strip()
