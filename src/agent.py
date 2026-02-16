"""Agent loop: manages conversation, calls Ollama, handles tool proposals."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import AsyncIterator, Callable, Awaitable

from .ollama_client import OllamaClient, StreamChunk, ToolCall


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


class Agent:
    """Manages conversation history and drives the Ollama chat loop."""

    def __init__(self, client: OllamaClient, system_prompt: str) -> None:
        self.client = client
        self.messages: list[dict] = [{"role": "system", "content": system_prompt}]

    def add_user_message(self, text: str) -> None:
        self.messages.append({"role": "user", "content": text})

    async def run_turn(self) -> AsyncIterator[AgentEvent]:
        """Run one model turn. Yields events the TUI should handle.

        When a TOOL_REQUEST event is yielded, the caller must:
        1. Ask the user for approval (if needed).
        2. Call complete_tool_call() with the result.
        3. Call run_turn() again to let the model see the result.
        """
        collected_text = ""
        pending_tool_calls: list[ToolCall] = []

        try:
            async for chunk in self.client.chat_stream(self.messages):
                if chunk.text:
                    collected_text += chunk.text
                    yield AgentEvent(kind=ActionKind.TEXT, text=chunk.text)

                if chunk.tool_calls:
                    pending_tool_calls.extend(chunk.tool_calls)

        except Exception as e:
            yield AgentEvent(kind=ActionKind.ERROR, text=str(e))
            return

        # Record assistant message in history
        assistant_msg: dict = {"role": "assistant", "content": collected_text}
        if pending_tool_calls:
            assistant_msg["tool_calls"] = [
                {
                    "function": {
                        "name": tc.name,
                        "arguments": tc.arguments,
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
        self.messages.append(
            {
                "role": "tool",
                "content": result,
            }
        )

    def needs_confirmation(self, tool_call: ToolCall) -> bool:
        """Return True if this tool call requires user approval."""
        return tool_call.name in CONFIRM_TOOLS
