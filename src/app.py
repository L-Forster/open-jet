"""open-jet TUI: single-pane chat with block title."""

from __future__ import annotations

from pathlib import Path

import yaml
from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Input, RichLog, Static

from .agent import ActionKind, Agent, ToolCall
from .executor import read_file, run_shell, write_file
from .ollama_client import OllamaClient


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config() -> dict:
    for candidate in [Path("config.yaml"), Path(__file__).resolve().parent.parent / "config.yaml"]:
        if candidate.exists():
            return yaml.safe_load(candidate.read_text())
    return {
        "ollama": {"base_url": "http://localhost:11434", "model": "llama3.2"},
        "system_prompt": "You are an AI assistant on a Jetson device.",
    }


# ---------------------------------------------------------------------------
# Block title banner
# ---------------------------------------------------------------------------

BANNER = r"""[bold cyan]
  open-jet
[/]"""


# ---------------------------------------------------------------------------
# Confirmation modal
# ---------------------------------------------------------------------------

class ConfirmModal(ModalScreen[bool]):
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
            detail = f"  $ {args.get('command', str(args))}"
        elif name == "write_file":
            path = args.get("path", "?")
            content = args.get("content", "")
            preview = content[:300] + ("..." if len(content) > 300 else "")
            detail = f"  {path}\n  ---\n  {preview}"
        else:
            detail = f"  {args}"

        yield Vertical(
            Static(f"[bold yellow]  {name} [/]", id="confirm-title"),
            Static(detail, id="confirm-detail"),
            Static(
                "  [bold green][y][/] Approve  [bold red][n][/] Deny",
                id="confirm-hint",
            ),
            id="confirm-box",
        )

    def action_approve(self) -> None:
        self.dismiss(True)

    def action_deny(self) -> None:
        self.dismiss(False)


# ---------------------------------------------------------------------------
# Styles
# ---------------------------------------------------------------------------

CSS = """
Screen {
    background: $background;
}
#chat-log {
    width: 100%;
    height: 1fr;
    padding: 0 2;
}
#prompt {
    dock: bottom;
    height: 3;
    margin: 0 2;
}
#confirm-box {
    width: 70%;
    height: auto;
    max-height: 60%;
    margin: 2 4;
    padding: 1 2;
    border: heavy $warning;
    background: $surface;
}
#confirm-title {
    margin-bottom: 1;
}
#confirm-detail {
    margin-bottom: 1;
    max-height: 16;
    overflow-y: auto;
}
#confirm-hint {
    margin-top: 1;
}
"""


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

class OpenJetApp(App):
    TITLE = "open-jet"
    CSS = CSS
    BINDINGS = [Binding("ctrl+c", "quit", "Quit")]

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
        yield RichLog(id="chat-log", wrap=True, markup=True)
        yield Input(placeholder="> ", id="prompt")

    def on_mount(self) -> None:
        log = self.query_one("#chat-log", RichLog)
        log.write(BANNER)
        model = self.client.model
        url = self.client.base_url
        log.write(f"  [dim]{model} · {url}[/]")
        log.write("")

    # -- Input ---------------------------------------------------------------

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        if not text:
            return
        event.input.value = ""

        log = self.query_one("#chat-log", RichLog)
        log.write(f"[bold green]> [/]{text}")
        log.write("")

        self.agent.add_user_message(text)
        self.run_agent_turn()

    # -- Agent turn ----------------------------------------------------------

    @work(exclusive=True)
    async def run_agent_turn(self) -> None:
        log = self.query_one("#chat-log", RichLog)
        pending_tool_calls: list[ToolCall] = []

        async for event in self.agent.run_turn():
            if event.kind == ActionKind.TEXT:
                log.write(event.text, end="")
            elif event.kind == ActionKind.TOOL_REQUEST:
                pending_tool_calls.append(event.tool_call)
            elif event.kind == ActionKind.ERROR:
                log.write(f"\n[bold red]error:[/] {event.text}")
                return
            elif event.kind == ActionKind.DONE:
                log.write("")

        for tc in pending_tool_calls:
            await self._handle_tool_call(tc, log)

        if pending_tool_calls:
            self.run_agent_turn()

    async def _handle_tool_call(self, tc: ToolCall, log: RichLog) -> None:
        needs_confirm = self.agent.needs_confirmation(tc)

        if needs_confirm:
            log.write(f"[yellow]{tc.name}:[/] {_fmt_args(tc)}")
            approved = await self.push_screen_wait(ConfirmModal(tc))
            if not approved:
                log.write("[dim red]  denied[/]")
                log.write("")
                self.agent.complete_tool_call(tc, "User denied this action.")
                return
            log.write("[dim green]  approved[/]")

        result = await _execute_tool(tc)
        # Show output inline in the chat
        for line in result.splitlines()[:20]:
            log.write(f"  [dim]{line}[/]")
        if len(result.splitlines()) > 20:
            log.write(f"  [dim]... ({len(result.splitlines()) - 20} more lines)[/]")
        log.write("")
        self.agent.complete_tool_call(tc, result)


# ---------------------------------------------------------------------------
# Tool dispatch
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
        return f"$ {tc.arguments.get('command', str(tc.arguments))}"
    if tc.name == "read_file":
        return tc.arguments.get("path", str(tc.arguments))
    if tc.name == "write_file":
        return tc.arguments.get("path", str(tc.arguments))
    return str(tc.arguments)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    app = OpenJetApp()
    app.run()


if __name__ == "__main__":
    main()
