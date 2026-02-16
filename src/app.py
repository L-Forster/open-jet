"""open-jetson TUI: chat pane, output pane, confirmation modal."""

from __future__ import annotations

import asyncio
from pathlib import Path

import yaml
from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Footer, Header, Input, RichLog, Static

from .agent import ActionKind, Agent, AgentEvent, ToolCall
from .executor import ExecResult, read_file, run_shell, write_file
from .ollama_client import OllamaClient


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

def load_config() -> dict:
    """Load config.yaml from the project root or cwd."""
    for candidate in [Path("config.yaml"), Path(__file__).resolve().parent.parent / "config.yaml"]:
        if candidate.exists():
            return yaml.safe_load(candidate.read_text())
    return {
        "ollama": {"base_url": "http://localhost:11434", "model": "llama3.2"},
        "system_prompt": "You are an AI assistant on a Jetson device.",
    }


# ---------------------------------------------------------------------------
# Confirmation modal
# ---------------------------------------------------------------------------

class ConfirmModal(ModalScreen[bool]):
    """Simple y/n approval dialog for risky tool calls."""

    BINDINGS = [
        Binding("y", "approve", "Approve"),
        Binding("n", "deny", "Deny"),
        Binding("escape", "deny", "Deny"),
    ]

    def __init__(self, tool_call: ToolCall) -> None:
        super().__init__()
        self.tool_call = tool_call

    def compose(self) -> ComposeResult:
        name = self.tool_call.name
        args = self.tool_call.arguments
        if name == "shell":
            detail = args.get("command", str(args))
        elif name == "write_file":
            path = args.get("path", "?")
            content = args.get("content", "")
            preview = content[:200] + ("..." if len(content) > 200 else "")
            detail = f"{path}\n---\n{preview}"
        else:
            detail = str(args)

        yield Vertical(
            Static(f"[bold yellow] Tool: {name} [/]", id="confirm-title"),
            Static(detail, id="confirm-detail"),
            Static("[bold green][y][/] Approve  [bold red][n][/] Deny", id="confirm-hint"),
            id="confirm-box",
        )

    def action_approve(self) -> None:
        self.dismiss(True)

    def action_deny(self) -> None:
        self.dismiss(False)


# ---------------------------------------------------------------------------
# Main TUI app
# ---------------------------------------------------------------------------

CSS = """
#confirm-box {
    width: 60%;
    height: auto;
    max-height: 70%;
    margin: 4 2;
    padding: 1 2;
    border: thick $accent;
    background: $surface;
    align: center middle;
}
#confirm-title {
    text-align: center;
    margin-bottom: 1;
}
#confirm-detail {
    margin-bottom: 1;
    max-height: 20;
    overflow-y: auto;
}
#confirm-hint {
    text-align: center;
}
#chat-pane {
    width: 2fr;
    border-right: solid $accent;
}
#output-pane {
    width: 1fr;
}
#input-box {
    dock: bottom;
    height: 3;
}
"""


class OpenJetsonApp(App):
    """Minimal agentic TUI for Jetson."""

    TITLE = "open-jetson"
    CSS = CSS

    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.cfg = load_config()
        ollama_cfg = self.cfg.get("ollama", {})
        self.client = OllamaClient(
            base_url=ollama_cfg.get("base_url", "http://localhost:11434"),
            model=ollama_cfg.get("model", "llama3.2"),
        )
        self.agent = Agent(
            client=self.client,
            system_prompt=self.cfg.get("system_prompt", ""),
        )

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal():
            yield RichLog(id="chat-pane", wrap=True, markup=True)
            yield RichLog(id="output-pane", wrap=True, markup=True)
        yield Input(placeholder="Type a message…", id="input-box")
        yield Footer()

    def on_mount(self) -> None:
        chat = self.query_one("#chat-pane", RichLog)
        chat.write("[bold cyan]open-jetson[/] ready. Connected to "
                   f"[bold]{self.client.model}[/] at {self.client.base_url}")
        chat.write("Type a message and press Enter.\n")

    # -- Input handling ------------------------------------------------------

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        if not text:
            return
        event.input.value = ""

        chat = self.query_one("#chat-pane", RichLog)
        chat.write(f"\n[bold green]You:[/] {text}")

        self.agent.add_user_message(text)
        self.run_agent_turn()

    # -- Agent turn (background worker) --------------------------------------

    @work(exclusive=True)
    async def run_agent_turn(self) -> None:
        """Drive one agent turn, handling tool requests inline."""
        chat = self.query_one("#chat-pane", RichLog)
        output = self.query_one("#output-pane", RichLog)

        chat.write("[bold magenta]Assistant:[/] ", end="")

        pending_tool_calls: list[ToolCall] = []

        async for event in self.agent.run_turn():
            if event.kind == ActionKind.TEXT:
                chat.write(event.text, end="")
            elif event.kind == ActionKind.TOOL_REQUEST:
                pending_tool_calls.append(event.tool_call)
            elif event.kind == ActionKind.ERROR:
                chat.write(f"\n[bold red]Error:[/] {event.text}")
                return
            elif event.kind == ActionKind.DONE:
                chat.write("")  # newline

        # Process tool calls sequentially
        for tc in pending_tool_calls:
            await self._handle_tool_call(tc, chat, output)

        # If we handled tool calls, run another turn so the model sees results
        if pending_tool_calls:
            self.run_agent_turn()

    async def _handle_tool_call(
        self, tc: ToolCall, chat: RichLog, output: RichLog
    ) -> None:
        """Execute a single tool call, prompting for confirmation if needed."""
        needs_confirm = self.agent.needs_confirmation(tc)

        if needs_confirm:
            chat.write(f"\n[yellow]Tool request:[/] {tc.name}({_fmt_args(tc)})")
            approved = await self.push_screen_wait(ConfirmModal(tc))
            if not approved:
                chat.write("[red]Denied.[/]")
                self.agent.complete_tool_call(tc, "User denied this action.")
                return
            chat.write("[green]Approved.[/]")

        # Execute the tool
        result = await _execute_tool(tc)
        output.write(f"[bold]{tc.name}[/]: {result[:500]}")
        self.agent.complete_tool_call(tc, result)


# ---------------------------------------------------------------------------
# Tool execution dispatch
# ---------------------------------------------------------------------------

async def _execute_tool(tc: ToolCall) -> str:
    if tc.name == "shell":
        res = await run_shell(tc.arguments.get("command", ""))
        return res.summary
    elif tc.name == "read_file":
        return await read_file(tc.arguments.get("path", ""))
    elif tc.name == "write_file":
        return await write_file(
            tc.arguments.get("path", ""),
            tc.arguments.get("content", ""),
        )
    return f"Unknown tool: {tc.name}"


def _fmt_args(tc: ToolCall) -> str:
    if tc.name == "shell":
        return tc.arguments.get("command", str(tc.arguments))
    if tc.name == "read_file":
        return tc.arguments.get("path", str(tc.arguments))
    if tc.name == "write_file":
        return tc.arguments.get("path", str(tc.arguments))
    return str(tc.arguments)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    app = OpenJetsonApp()
    app.run()


if __name__ == "__main__":
    main()
