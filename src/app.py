"""open-jet TUI: single-pane chat with block title."""

from __future__ import annotations

import asyncio
import re
import time
from pathlib import Path

import yaml
from textual import events, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Input, RichLog, Static
from textual.widgets._input import Selection

from .agent import ActionKind, Agent, ToolCall
from .commands import SlashCommandHandler
from .completion import CompletionEngine, FileMentionCompletionProvider, SlashCompletionProvider
from .executor import load_file, read_file, run_shell, write_file
from .ollama_client import OllamaClient
from .runtime_limits import derive_context_budget, estimate_tokens, read_memory_snapshot
from .session_logging import SessionLogger


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"


def load_config() -> dict:
    for candidate in [Path("config.yaml"), CONFIG_PATH]:
        if candidate.exists():
            return yaml.safe_load(candidate.read_text()) or {}
    return {}


def save_config(cfg: dict) -> None:
    CONFIG_PATH.write_text(yaml.dump(cfg, default_flow_style=False))


# ---------------------------------------------------------------------------
# Block title banner
# ---------------------------------------------------------------------------

BANNER = r"""[bold green]
   ___                   _        _   
  / _ \ _ __   ___ _ __  (_) ___  | |_ 
 | | | | '_ \ / _ \ '_ \ | |/ _ \ | __|
 | |_| | |_) |  __/ | | || |  __/ | |_ 
  \___/| .__/ \___|_| |_|/ |\___|  \__|
       |_|              |__/           
[/]"""


class SetupScreen(ModalScreen[dict]):
    BINDINGS = [Binding("escape", "cancel", "Cancel")]

    def compose(self) -> ComposeResult:
        yield Vertical(
            Static("[bold green]First-run setup[/]"),
            Static(""),
            Static("Path to your .gguf model file:"),
            Input(placeholder="/path/to/model.gguf", id="setup-model"),
            Static(""),
            Static("[dim]Press enter to save[/]"),
            id="setup-box",
        )

    def on_mount(self) -> None:
        self.query_one("#setup-model", Input).focus()

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        model_path = event.value.strip()
        if not model_path:
            return
        if not Path(model_path).is_file():
            self.query_one("#setup-model", Input).value = ""
            return
        self.dismiss({"model": model_path})

    def action_cancel(self) -> None:
        self.app.exit()


# ---------------------------------------------------------------------------
# Styles
# ---------------------------------------------------------------------------

CSS = """
Screen {
    background: $background;
}
#chat-log {
    width: 100%;
    height: auto;
    padding: 0 2;
}
#prompt {
    height: 3;
    margin: 0 2;
}
#command-suggestions {
    height: auto;
    margin: 0 2;
    color: $text-muted;
}
#token-counter {
    height: 1;
    margin: 0 2;
    color: $text-muted;
}
#assistant-status {
    height: 1;
    margin: 0 2;
    color: $accent;
}
#approval-bar {
    height: auto;
    min-height: 3;
    margin: 0 2;
    padding: 0 1;
    border: round $warning;
    background: $surface;
}
.hidden {
    display: none;
}
#setup-box {
    width: 70%;
    height: auto;
    margin: 4 8;
    padding: 1 2;
    border: heavy $accent;
    background: $surface;
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
        self.client: OllamaClient | None = None
        self.agent: Agent | None = None
        self.session_logger: SessionLogger | None = None
        self._thinking_timer = None
        self._thinking_idx = 0
        self._thinking_token = 0
        self._awaiting_approval = False
        self._approval_choice = 0
        self._approval_future: asyncio.Future[bool] | None = None
        self._approval_tool_call: ToolCall | None = None
        self.commands = SlashCommandHandler(self, banner=BANNER)
        self.completion = CompletionEngine(
            [
                SlashCompletionProvider(self.commands),
                FileMentionCompletionProvider(Path.cwd()),
            ]
        )

    async def _init_client(self) -> None:
        mem_cfg = self.cfg.get("memory_guard", {})
        self.client = OllamaClient(
            model=self.cfg["model"],
            context_window_tokens=int(self.cfg.get("context_window_tokens", 2048)),
        )
        await self.client.start()
        self.agent = Agent(
            client=self.client,
            system_prompt=self.cfg.get("system_prompt", ""),
            context_window_tokens=self.client.context_window_tokens,
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

    def compose(self) -> ComposeResult:
        yield RichLog(id="chat-log", wrap=True, markup=True)
        yield Static("", id="assistant-status", classes="hidden")
        yield Static("", id="approval-bar", classes="hidden")
        yield Input(placeholder="> ", id="prompt")
        yield Static("", id="command-suggestions", classes="hidden")
        yield Static("", id="token-counter")

    async def on_mount(self) -> None:
        log = self.query_one("#chat-log", RichLog)
        prompt = self.query_one("#prompt", Input)
        log.write(BANNER)
        log_cfg = self.cfg.get("logging", {})
        if log_cfg.get("enabled", True):
            self.session_logger = SessionLogger(
                base_dir=Path(log_cfg.get("directory", "session_logs")),
                label=str(log_cfg.get("label", "open-jet")),
                metrics_interval_seconds=float(log_cfg.get("metrics_interval_seconds", 5)),
            )
            await self.session_logger.start()
            self.session_logger.log_event("app_mount", cwd=str(Path.cwd()))

        if not self.cfg.get("model"):
            result = await self.push_screen_wait(SetupScreen())
            self.cfg["model"] = result["model"]
            save_config(self.cfg)

        log.write(f"  [dim]Loading {Path(self.cfg['model']).name}...[/]")
        try:
            await self._init_client()
        except Exception as e:
            log.write(f"\n[bold red]Failed to start LLM:[/] {e}")
            if self.session_logger:
                self.session_logger.log_event("llm_start_error", error=str(e))
            prompt.focus()
            return
        if self.session_logger:
            self.session_logger.log_event("llm_ready", model=self.cfg["model"])
        log.write(f"  [dim]Ready.[/]")
        log.write("")
        self._render_token_counter()
        prompt.focus()

    async def action_quit(self) -> None:
        if self.client:
            await self.client.close()
        if self.session_logger:
            await self.session_logger.stop()
        self.exit()

    def on_key(self, event: events.Key) -> None:
        if self._awaiting_approval:
            if event.key in ("left", "right"):
                self._approval_choice = 0 if event.key == "left" else 1
                self._render_approval_bar()
                event.stop()
                return
            if event.key == "y":
                self._approval_choice = 0
                self._resolve_approval(True)
                event.stop()
                return
            if event.key in ("n", "escape"):
                self._approval_choice = 1
                self._resolve_approval(False)
                event.stop()
                return
            if event.key == "enter":
                self._resolve_approval(self._approval_choice == 0)
                event.stop()
            return

        prompt = self.query_one("#prompt", Input)
        if self.focused is not prompt:
            return

        if event.key == "up" and self.completion.state:
            event.prevent_default()
            self.completion.cycle(-1)
            self._render_completion_suggestions()
            event.stop()
            return
        if event.key == "down" and self.completion.state:
            event.prevent_default()
            self.completion.cycle(1)
            self._render_completion_suggestions()
            event.stop()
            return
        if event.key == "tab":
            event.prevent_default()
            if self.completion.state:
                prompt.value = self.completion.apply_selected(prompt.value)
                self._update_completion_suggestions(prompt.value)
                prompt.action_end(select=False)
                self.call_after_refresh(self._collapse_prompt_selection)
                prompt.focus()
            elif prompt.value.lstrip().startswith(("/", "@")):
                # Keep focus in chat input when tab is used for completion contexts.
                prompt.focus()
                self.call_after_refresh(self._collapse_prompt_selection)
            event.stop()
            return

    # -- Input ---------------------------------------------------------------

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id != "prompt":
            return
        self._update_completion_suggestions(event.value)
        self._render_token_counter(event.value)

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        if self._awaiting_approval:
            event.input.value = ""
            return

        text = event.value.strip()
        if not text:
            return
        event.input.value = ""
        self._clear_completion_suggestions()

        if await self.commands.maybe_handle(text):
            self._render_token_counter()
            return

        log = self.query_one("#chat-log", RichLog)
        log.write(f"[bold green]> [/]{text}")
        log.write("")
        if self.session_logger:
            self.session_logger.log_event("user_message", text=text)

        await self._load_mentioned_files_into_context(text, log)
        self.agent.add_user_message(text)
        self._render_token_counter()
        self.run_agent_turn()

    async def _load_mentioned_files_into_context(self, text: str, log: RichLog) -> None:
        if not self.agent:
            return

        paths = _extract_file_mentions(text)
        if not paths:
            return

        for mention_path in paths:
            await self.load_context_file(mention_path, log)
        self._render_token_counter()

    async def load_context_file(self, path: str, log: RichLog) -> bool:
        if not self.agent:
            return False
        mention_path = path.strip()
        if not mention_path:
            log.write("[yellow]load:[/] empty path")
            return False

        current_tokens = self.agent.estimated_context_tokens()
        remaining_tokens = self._remaining_prompt_tokens()
        result = await load_file(mention_path, max_tokens=remaining_tokens)
        if not result.ok:
            log.write(f"[yellow]@{mention_path}:[/] {result.detail}")
            return False

        context_text = (
            "User-loaded file context:\n"
            f"path: {result.path}\n"
            f"tokens_estimated: {result.estimated_tokens}\n"
            f"tokens_loaded: {result.returned_tokens}\n"
            f"token_budget: {result.token_budget}\n"
            f"truncated: {'yes' if result.truncated else 'no'}\n"
            "content:\n"
            f"{result.content}"
        )
        self.agent.messages.append({"role": "system", "content": context_text})
        log.write(f"[dim]Loaded @{mention_path} into context ({result.summary}).[/]")
        if self.session_logger:
            self.session_logger.log_event(
                "context_file_loaded",
                mention_path=mention_path,
                resolved_path=result.path,
                context_tokens_before=current_tokens,
                estimated_tokens=result.estimated_tokens,
                returned_tokens=result.returned_tokens,
                token_budget=result.token_budget,
                remaining_prompt_tokens=remaining_tokens,
                truncated=result.truncated,
                mem_available_mb=result.mem_available_mb,
            )
        self._render_token_counter()
        return True

    def _update_completion_suggestions(self, raw_value: str) -> None:
        if self._awaiting_approval:
            self._clear_completion_suggestions()
            return
        state = self.completion.refresh(raw_value)
        if not state:
            self._clear_completion_suggestions()
            return
        self._render_completion_suggestions()

    def _render_completion_suggestions(self) -> None:
        bar = self.query_one("#command-suggestions", Static)
        state = self.completion.state
        if not state:
            bar.add_class("hidden")
            bar.update("")
            return

        lines: list[str] = []
        for idx, item in enumerate(state.items):
            if idx == state.index:
                lines.append(f"[black on green] {item.label} [/]")
            else:
                lines.append(f"[bold green]{item.label}[/]")
            if item.detail:
                lines[-1] += f" [dim]- {item.detail}[/]"
        bar.remove_class("hidden")
        bar.update("\n".join(lines) + "\n[dim]Up/Down to select, Tab to autocomplete[/]")

    def _clear_completion_suggestions(self) -> None:
        self.completion.clear()
        bar = self.query_one("#command-suggestions", Static)
        bar.add_class("hidden")
        bar.update("")

    def _render_token_counter(self, draft_text: str = "") -> None:
        counter = self.query_one("#token-counter", Static)
        if not self.agent:
            counter.update("[dim]tokens: 0/0[/]")
            return
        current = self.agent.estimated_context_tokens()
        draft = estimate_tokens(draft_text)
        total = current + draft
        window = self.client.context_window_tokens if self.client else int(self.cfg.get("context_window_tokens", 2048))
        budget = self.agent.context_budget() or derive_context_budget(window)
        remaining = max(0, budget.prompt_tokens - total)
        if total > budget.prompt_tokens:
            color = "red"
        elif remaining <= 256:
            color = "yellow"
        else:
            color = "dim"
        counter.update(
            f"[{color}]tokens: {total}/{window} | prompt<= {budget.prompt_tokens} | remaining: {remaining}[/]"
        )

    def runtime_status_snapshot(self) -> dict:
        if not self.agent:
            return {"ready": False}

        window = self.client.context_window_tokens if self.client else int(self.cfg.get("context_window_tokens", 2048))
        budget = self.agent.context_budget() or derive_context_budget(window)
        current = self.agent.estimated_context_tokens()
        remaining = max(0, budget.prompt_tokens - current)
        mem = read_memory_snapshot()
        return {
            "ready": True,
            "messages": self.agent.conversation_message_count(),
            "generating": self._thinking_timer is not None,
            "context_tokens": current,
            "context_window_tokens": window,
            "prompt_budget_tokens": budget.prompt_tokens,
            "reserve_tokens": budget.reserve_tokens,
            "remaining_prompt_tokens": remaining,
            "memory_total_mb": mem.total_mb if mem else None,
            "memory_available_mb": mem.available_mb if mem else None,
            "memory_used_percent": mem.used_percent if mem else None,
        }

    def refresh_token_counter(self) -> None:
        self._render_token_counter()

    # -- Agent turn ----------------------------------------------------------

    @work(exclusive=True)
    async def run_agent_turn(self) -> None:
        log = self.query_one("#chat-log", RichLog)
        pending_tool_calls: list[ToolCall] = []
        text_buf = ""
        assistant_turn_text = ""
        thinking_token = self._start_thinking()

        try:
            async for event in self.agent.run_turn():
                if event.kind == ActionKind.TEXT:
                    text_buf += event.text
                    assistant_turn_text += event.text
                    # Flush complete lines as they arrive
                    while "\n" in text_buf:
                        line, text_buf = text_buf.split("\n", 1)
                        log.write(line)
                elif event.kind == ActionKind.TOOL_REQUEST:
                    pending_tool_calls.append(event.tool_call)
                    if self.session_logger and event.tool_call:
                        self.session_logger.log_event(
                            "tool_request",
                            tool=event.tool_call.name,
                            arguments=event.tool_call.arguments,
                        )
                elif event.kind == ActionKind.ERROR:
                    log.write(f"\n[bold red]error:[/] {event.text}")
                    if self.session_logger:
                        self.session_logger.log_event("agent_error", error=event.text)
                    return
                elif event.kind == ActionKind.DONE:
                    if text_buf:
                        log.write(text_buf)
                        text_buf = ""
                    log.write("")

            # Flush any remaining text
            if text_buf:
                log.write(text_buf)
        finally:
            self._stop_thinking(thinking_token)

        if self.session_logger and assistant_turn_text.strip():
            self.session_logger.log_event("assistant_message", text=assistant_turn_text)

        for tc in pending_tool_calls:
            await self._handle_tool_call(tc, log)

        if pending_tool_calls:
            self.run_agent_turn()
        else:
            self._render_token_counter()

    async def _handle_tool_call(self, tc: ToolCall, log: RichLog) -> None:
        if self.agent.is_internal_condense_tool(tc):
            log.write(f"[yellow]{tc.name}:[/] {_fmt_args(tc)}")
            result = await self.agent.condense_context()
            log.write(f"  [dim]{result}[/]")
            log.write("")
            self.agent.complete_tool_call(tc, result)
            return

        needs_confirm = self.agent.needs_confirmation(tc)

        if needs_confirm:
            log.write(f"[yellow]{tc.name}:[/] {_fmt_args(tc)}")
            approved = await self._wait_for_tool_approval(tc)
            if not approved:
                log.write("[dim red]  denied[/]")
                log.write("")
                if self.session_logger:
                    self.session_logger.log_event(
                        "tool_approval",
                        tool=tc.name,
                        approved=False,
                        arguments=tc.arguments,
                    )
                self.agent.complete_tool_call(tc, "User denied this action.")
                return
            log.write("[dim green]  approved[/]")
            if self.session_logger:
                self.session_logger.log_event(
                    "tool_approval",
                    tool=tc.name,
                    approved=True,
                    arguments=tc.arguments,
                )

        if tc.name == "load_file":
            self._clamp_load_file_tool_budget(tc)

        t0 = time.monotonic()
        result, meta = await _execute_tool(tc)
        duration_ms = round((time.monotonic() - t0) * 1000.0, 2)
        if self.session_logger:
            self.session_logger.log_tool_result(
                tc.name,
                result,
                duration_ms=duration_ms,
                arguments=tc.arguments,
                **meta,
            )
        # Show output inline in the chat
        for line in result.splitlines()[:20]:
            log.write(f"  [dim]{line}[/]")
        if len(result.splitlines()) > 20:
            log.write(f"  [dim]... ({len(result.splitlines()) - 20} more lines)[/]")
        log.write("")
        self.agent.complete_tool_call(tc, result)

    def _clamp_load_file_tool_budget(self, tc: ToolCall) -> None:
        if not isinstance(tc.arguments, dict):
            return
        remaining = self._remaining_prompt_tokens()
        current = tc.arguments.get("max_tokens")
        if not isinstance(current, int):
            tc.arguments["max_tokens"] = remaining
            return
        tc.arguments["max_tokens"] = max(128, min(current, remaining))

    def _remaining_prompt_tokens(self) -> int:
        if not self.agent:
            return 128
        current = self.agent.estimated_context_tokens()
        budget = self.agent.context_budget()
        if not budget:
            window = self.client.context_window_tokens if self.client else int(self.cfg.get("context_window_tokens", 2048))
            budget = derive_context_budget(window)
        return max(128, budget.prompt_tokens - current)

    def _start_thinking(self) -> int:
        status = self.query_one("#assistant-status", Static)
        self._thinking_token += 1
        self._thinking_idx = 0
        status.remove_class("hidden")
        status.update("[bold green]Generating .[/]")
        if self._thinking_timer:
            self._thinking_timer.stop()
        self._thinking_timer = self.set_interval(0.4, self._tick_thinking)
        return self._thinking_token

    def _tick_thinking(self) -> None:
        status = self.query_one("#assistant-status", Static)
        dots = [".", "..", "..."][self._thinking_idx % 3]
        status.update(f"[bold green]Generating {dots}[/]")
        self._thinking_idx += 1

    def _stop_thinking(self, token: int | None = None) -> None:
        if token is not None and token != self._thinking_token:
            return
        status = self.query_one("#assistant-status", Static)
        if self._thinking_timer:
            self._thinking_timer.stop()
            self._thinking_timer = None
        status.add_class("hidden")
        status.update("")

    async def _wait_for_tool_approval(self, tc: ToolCall) -> bool:
        bar = self.query_one("#approval-bar", Static)
        prompt = self.query_one("#prompt", Input)

        self._awaiting_approval = True
        self._approval_choice = 0
        self._approval_tool_call = tc
        self._approval_future = asyncio.get_running_loop().create_future()
        bar.remove_class("hidden")
        prompt.disabled = True
        self._render_approval_bar()

        try:
            return await self._approval_future
        finally:
            self._awaiting_approval = False
            self._approval_tool_call = None
            self._approval_future = None
            bar.add_class("hidden")
            bar.update("")
            prompt.disabled = False
            prompt.focus()

    def _render_approval_bar(self) -> None:
        if not self._awaiting_approval or not self._approval_tool_call:
            return
        bar = self.query_one("#approval-bar", Static)
        detail = _fmt_args(self._approval_tool_call)
        if len(detail) > 100:
            detail = detail[:97] + "..."
        approve = (
            "[black on green] Approve [/]"
            if self._approval_choice == 0
            else "[bold green]Approve[/]"
        )
        deny = (
            "[black on red] Deny [/]"
            if self._approval_choice == 1
            else "[bold red]Deny[/]"
        )
        bar.update(
            f"[bold yellow]Tool request:[/] {detail}\n"
            f"Use [bold]←[/]/[bold]→[/] then [bold]Enter[/]   {approve}  {deny}"
        )

    def _resolve_approval(self, approved: bool) -> None:
        if self._approval_future and not self._approval_future.done():
            self._approval_future.set_result(approved)

    def _collapse_prompt_selection(self) -> None:
        prompt = self.query_one("#prompt", Input)
        cursor = prompt.cursor_position
        prompt.selection = Selection(cursor, cursor)


# ---------------------------------------------------------------------------
# Tool dispatch
# ---------------------------------------------------------------------------

async def _execute_tool(tc: ToolCall) -> tuple[str, dict]:
    if not isinstance(tc.arguments, dict):
        return f"Error: invalid arguments for {tc.name}", {"ok": False}

    if tc.name == "shell":
        command = tc.arguments.get("command", "")
        if not isinstance(command, str) or not command.strip():
            return "Error: invalid arguments for shell (required: command)", {"ok": False}
        res = await run_shell(command)
        return res.summary, {
            "ok": res.ok,
            "exit_code": res.exit_code,
            "stdout": res.stdout,
            "stderr": res.stderr,
        }
    elif tc.name == "read_file":
        path = tc.arguments.get("path", "")
        if not isinstance(path, str) or not path.strip():
            return "Error: invalid arguments for read_file (required: path)", {"ok": False}
        text = await read_file(path)
        return text, {"ok": not text.startswith("Error:")}
    elif tc.name == "write_file":
        path = tc.arguments.get("path", "")
        content = tc.arguments.get("content", "")
        if not isinstance(path, str) or not path.strip():
            return "Error: invalid arguments for write_file (required: path, content)", {"ok": False}
        if not isinstance(content, str):
            return "Error: invalid arguments for write_file (required: path, content)", {"ok": False}
        text = await write_file(
            path,
            content,
        )
        return text, {"ok": not text.startswith("Error")}
    elif tc.name == "load_file":
        path = tc.arguments.get("path", "")
        max_tokens = tc.arguments.get("max_tokens")
        if not isinstance(path, str) or not path.strip():
            return "Error: invalid arguments for load_file (required: path)", {"ok": False}
        if max_tokens is not None and not isinstance(max_tokens, int):
            return "Error: invalid arguments for load_file (max_tokens must be int)", {"ok": False}
        loaded = await load_file(path, max_tokens=max_tokens)
        if not loaded.ok:
            return loaded.detail, {"ok": False}
        payload = (
            f"{loaded.summary}\n"
            f"content:\n{loaded.content}"
        )
        return payload, {
            "ok": True,
            "truncated": loaded.truncated,
            "estimated_tokens": loaded.estimated_tokens,
            "returned_tokens": loaded.returned_tokens,
            "token_budget": loaded.token_budget,
            "mem_available_mb": loaded.mem_available_mb,
        }
    return f"Unknown tool: {tc.name}", {"ok": False}


def _fmt_args(tc: ToolCall) -> str:
    if tc.name == "shell":
        return f"$ {tc.arguments.get('command', str(tc.arguments))}"
    if tc.name == "read_file":
        return tc.arguments.get("path", str(tc.arguments))
    if tc.name == "write_file":
        return tc.arguments.get("path", str(tc.arguments))
    if tc.name == "load_file":
        return tc.arguments.get("path", str(tc.arguments))
    return str(tc.arguments)


def _extract_file_mentions(text: str) -> list[str]:
    cleaned: list[str] = []
    seen: set[str] = set()
    for match in re.finditer(r"@\[([^\]]+)\]|(?<!\S)@([^\s]+)", text):
        bracketed = match.group(1)
        bare = match.group(2)
        candidate = (bracketed if bracketed is not None else bare or "").strip()
        if bracketed is None:
            candidate = candidate.rstrip(".,;:!?)]}")
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        cleaned.append(candidate)
    return cleaned


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    app = OpenJetApp()
    app.run()


if __name__ == "__main__":
    main()
