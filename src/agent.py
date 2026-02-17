"""Agent loop: manages conversation, calls Ollama, handles tool proposals."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from enum import Enum, auto
from typing import AsyncIterator

from .ollama_client import OllamaClient, ToolCall


class ActionKind(Enum):
    TEXT = auto()         # plain text token to display
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
CONFIRM_TOOLS = {"shell", "write_file"}
CONDENSE_COMMAND = "open-jet-condense-context"


class Agent:
    """Manages conversation history and drives the Ollama chat loop."""

    def __init__(
        self,
        client: OllamaClient,
        system_prompt: str,
        *,
        min_available_mb: int | None = None,
        max_used_percent: float | None = None,
        memory_check_interval_chunks: int = 16,
        condense_target_tokens: int = 900,
        keep_last_messages: int = 6,
    ) -> None:
        self.client = client
        self.messages: list[dict] = [{"role": "system", "content": system_prompt}]
        self.min_available_mb = min_available_mb
        self.max_used_percent = max_used_percent
        self.memory_check_interval_chunks = max(1, int(memory_check_interval_chunks))
        self.condense_target_tokens = condense_target_tokens
        self.keep_last_messages = keep_last_messages

    def add_user_message(self, text: str) -> None:
        self.messages.append({"role": "user", "content": text})

    async def run_turn(self) -> AsyncIterator[AgentEvent]:
        """Run one model turn. Yields events the TUI should handle.

        When a TOOL_REQUEST event is yielded, the caller must:
        1. Ask the user for approval (if needed).
        2. Call complete_tool_call() with the result.
        3. Call run_turn() again to let the model see the result.
        """
        pressure_reason = self._memory_pressure_reason()
        if pressure_reason:
            tool_call = self._build_condense_tool_call()
            self._append_assistant_tool_call("", tool_call)
            yield AgentEvent(
                kind=ActionKind.TEXT,
                text=f"(out of memory, condensing chat to reduce kvcache: {pressure_reason})",
            )
            yield AgentEvent(
                kind=ActionKind.TOOL_REQUEST,
                tool_call=tool_call,
            )
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
                    pressure_reason = self._memory_pressure_reason()
                    if pressure_reason:
                        tool_call = self._build_condense_tool_call()
                        self._append_assistant_tool_call(collected_text, tool_call)
                        yield AgentEvent(
                            kind=ActionKind.TEXT,
                            text=(
                                "\n(out of memory, condensing chat to reduce kvcache: "
                                f"{pressure_reason})"
                            ),
                        )
                        yield AgentEvent(kind=ActionKind.TOOL_REQUEST, tool_call=tool_call)
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

    def _build_condense_tool_call(self) -> ToolCall:
        return ToolCall(
            name="shell",
            arguments={"command": CONDENSE_COMMAND},
            id=f"call_{uuid.uuid4().hex[:12]}",
        )

    def _append_assistant_tool_call(self, content: str, tool_call: ToolCall) -> None:
        self.messages.append(
            {
                "role": "assistant",
                "content": content,
                "tool_calls": [
                    {
                        "id": tool_call.id,
                        "type": "function",
                        "function": {
                            "name": tool_call.name,
                            "arguments": json.dumps(tool_call.arguments, ensure_ascii=True),
                        }
                    }
                ],
            }
        )

    def needs_confirmation(self, tool_call: ToolCall) -> bool:
        """Return True if this tool call requires user approval."""
        if self.is_internal_condense_tool(tool_call):
            return False
        return tool_call.name in CONFIRM_TOOLS

    def is_internal_condense_tool(self, tool_call: ToolCall) -> bool:
        if tool_call.name != "shell":
            return False
        return tool_call.arguments.get("command", "").strip() == CONDENSE_COMMAND

    def condense_context(self) -> str:
        """Condense older messages into a compact system summary."""
        if len(self.messages) <= 2:
            return "No condensation needed."

        total_before = self._estimated_context_tokens()
        original_messages = len(self.messages)
        notes: list[str] = []

        # Keep system prompt plus only the newest N messages.
        keep = max(2, self.keep_last_messages)
        recent = self.messages[-keep:]
        older = self.messages[1:-keep]

        if older:
            summary = self._build_summary(older)
            summary_msg = {
                "role": "system",
                "content": (
                    "Condensed conversation context (older turns):\n"
                    f"{summary}"
                ),
            }
            self.messages = [self.messages[0], summary_msg, *recent]
            notes.append("summarized older turns")
        else:
            notes.append("no older turns to summarize")

        # If still large, progressively trim oldest remaining non-system messages.
        while (
            self._estimated_context_tokens() > self.condense_target_tokens
            and len(self.messages) > 4
        ):
            del self.messages[2]
            notes.append("dropped oldest detailed turn")

        # Final safeguard: truncate oldest large payload if still over target.
        if self._estimated_context_tokens() > self.condense_target_tokens:
            for idx in range(1, max(1, len(self.messages) - 1)):
                content = str(self.messages[idx].get("content", ""))
                if len(content) > 800:
                    self.messages[idx]["content"] = content[:800] + "\n...[truncated]"
                    notes.append("truncated oversized message")
                    break

        total_after = self._estimated_context_tokens()
        return (
            "Context condensed via command tool. "
            f"messages: {original_messages} -> {len(self.messages)}, "
            f"tokens(est): {total_before} -> {total_after}. "
            f"actions: {', '.join(notes)}"
        )

    def _estimated_context_tokens(self) -> int:
        total = 0
        for msg in self.messages:
            content = str(msg.get("content", ""))
            # Fast estimate: roughly 1 token ~= 4 chars + protocol overhead.
            total += max(1, len(content) // 4) + 8
        return total

    def _memory_pressure_reason(self) -> str | None:
        total_mb, available_mb, used_percent = self._read_memory_info_mb()
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

    def _read_memory_info_mb(self) -> tuple[float | None, float | None, float | None]:
        mem_total_kb: int | None = None
        mem_available_kb: int | None = None
        try:
            with open("/proc/meminfo", "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        mem_total_kb = int(line.split()[1])
                    elif line.startswith("MemAvailable:"):
                        mem_available_kb = int(line.split()[1])
        except OSError:
            return None, None, None

        if not mem_total_kb or mem_available_kb is None:
            return None, None, None

        total_mb = mem_total_kb / 1024.0
        available_mb = mem_available_kb / 1024.0
        used_percent = ((mem_total_kb - mem_available_kb) / mem_total_kb) * 100.0
        return total_mb, available_mb, used_percent

    def _build_summary(self, older_messages: list[dict]) -> str:
        lines: list[str] = []
        for msg in older_messages[-20:]:
            role = str(msg.get("role", "unknown")).upper()
            text = " ".join(str(msg.get("content", "")).split())
            if len(text) > 220:
                text = text[:217] + "..."
            if text:
                lines.append(f"- {role}: {text}")
        if not lines:
            return "- (no prior text content)"
        return "\n".join(lines)
