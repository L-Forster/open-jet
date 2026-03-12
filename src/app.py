"""open-jet TUI: single-pane chat with block title."""

from __future__ import annotations

import argparse
import asyncio
import os
import re
import shlex
import shutil
import time
from pathlib import Path

from rich.markup import escape
from textual import events, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container
from textual.screen import Screen
from textual.worker import Worker, get_current_worker
from textual.widgets import Input, RichLog, Static
from textual.widgets.input import Selection

from .agent import ActionKind, Agent, ToolCall
from .commands import SlashCommandHandler
from .completion import CompletionEngine, FileMentionCompletionProvider, SlashCompletionProvider
from .executor import (
    DEFAULT_SHELL_TIMEOUT_SECONDS,
    edit_file,
    glob_files,
    grep_files,
    list_directory,
    load_file,
    read_file,
    run_shell,
    write_file,
)
from .harness import (
    HarnessSessionStore,
    HarnessState,
    active_step,
    advance_step,
    allowed_tools_for_mode,
    append_memory_entry,
    available_skill_names,
    build_turn_context,
    clear_preferred_skills,
    normalize_skill_name,
    set_mode,
    set_preferred_skills,
    shell_command_is_verification,
    split_active_step,
    update_state_after_turn,
    update_state_for_user_message,
)
from .config import load_config, save_config
from .hardware import (
    detect_hardware_info,
    recommended_context_window_tokens,
    recommended_device,
    recommended_gpu_layers,
)
from .ollama_setup import discover_installed_ollama_models, materialize_setup_model
from .runtime_client import RuntimeClient
from .runtime_limits import derive_context_budget, estimate_tokens, read_memory_snapshot
from .runtime_registry import active_model_ref, create_runtime_client
from .session_logging import SessionLogger
from .session_state import SessionStateStore
from .setup import ACCENT_GREEN, SetupScreen, discover_model_files
from .system_metrics import SystemMetricsReader, format_hours
from .tool_executor import execute_tool, format_tool_args


def _format_error(exc: Exception) -> str:
    text = str(exc).strip()
    if text:
        return text
    return f"{type(exc).__name__} (no message)"


_SHELL_BUILTINS = {
    ".", ":", "alias", "bg", "bind", "break", "builtin", "caller", "cd", "command",
    "compgen", "complete", "compopt", "continue", "declare", "dirs", "disown", "echo",
    "enable", "eval", "exec", "exit", "export", "fc", "fg", "getopts", "hash", "help",
    "history", "jobs", "kill", "let", "local", "logout", "mapfile", "popd", "printf",
    "pushd", "pwd", "read", "readarray", "readonly", "return", "set", "shift", "shopt",
    "source", "suspend", "test", "times", "trap", "type", "typeset", "ulimit", "umask",
    "unalias", "unset", "wait",
}


def _classify_shell_command(command: str) -> dict[str, object]:
    stripped = command.strip()
    if not stripped:
        return {
            "primary_command": "",
            "classified_verification": False,
            "hallucinated_command": False,
            "false_positive_proposal": True,
            "classification_reason": "empty command",
        }

    try:
        parts = shlex.split(stripped)
    except ValueError:
        parts = stripped.split()
    primary = parts[0] if parts else ""
    verification = shell_command_is_verification(stripped)
    builtin = primary in _SHELL_BUILTINS
    executable_found = builtin or bool(shutil.which(primary)) or "/" in primary
    false_positive = False
    reasons: list[str] = []

    if primary in {"cat", "ls", "find", "grep"}:
        false_positive = True
        reasons.append("covered by dedicated tool")
    if primary == "echo" and not verification:
        false_positive = True
        reasons.append("non-actionable shell proposal")

    if not executable_found:
        reasons.append("command not found on PATH")

    return {
        "primary_command": primary,
        "classified_verification": verification,
        "hallucinated_command": not executable_found,
        "false_positive_proposal": false_positive,
        "classification_reason": ", ".join(reasons) if reasons else None,
    }


# ---------------------------------------------------------------------------
# Block title banner
# ---------------------------------------------------------------------------

BANNER = r"""[bold green]
   ___                    _        _   
  / _ \ _ __   ___ _ __  (_) ___  | |_ 
 | | | | '_ \ / _ \ '_ \ | |/ _ \ | __|
 | |_| | |_) |  __/ | | || |  __/ | |_ 
  \___/| .__/ \___|_| |_|/ |\___|  \__|
       |_|              |__/           
[/]"""


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

class OpenJetApp(App):
    TITLE = "open-jet"
    CSS_PATH = "app.tcss"
    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit"),
        Binding("escape", "stop_generation", "Stop"),
    ]

    def __init__(self, *, force_setup: bool = False) -> None:
        super().__init__()
        self.force_setup = force_setup
        self.cfg = load_config()
        self.client: RuntimeClient | None = None
        self.agent: Agent | None = None
        self.session_logger: SessionLogger | None = None
        state_cfg = self.cfg.get("state", {})
        self.state_store = SessionStateStore(
            path=Path(state_cfg.get("path", "session_state.json")),
            enabled=bool(state_cfg.get("enabled", True)),
        )
        self.harness_store = HarnessSessionStore()
        self.harness_state = HarnessState()
        self._turn_context_docs: list[str] = []
        self._turn_context_tokens = 0
        self.auto_resume = bool(state_cfg.get("auto_resume", False))
        self._session_was_resumed = False
        self._turn_counter = 0
        self._active_turn_id: str | None = None
        self._active_turn_started_at: float | None = None
        self._active_turn_prompt = ""
        self._active_turn_generation_tokens = 0
        self._active_turn_tool_attempts = 0
        self._active_turn_tool_successes = 0
        self._active_turn_approval_requests = 0
        self._active_turn_approval_grants = 0
        self._active_turn_false_positive_commands = 0
        self._active_turn_hallucinated_commands = 0
        self._active_turn_recovered_after_resume = False
        self.loaded_files: dict[str, dict] = {}
        self._thinking_timer = None
        self._thinking_idx = 0
        self._thinking_token = 0
        self._awaiting_approval = False
        self._approval_choice = 0
        self._approval_future: asyncio.Future[bool] | None = None
        self._approval_tool_call: ToolCall | None = None
        self._approval_started_at: float | None = None
        self._generation_worker: Worker | None = None
        self.commands = SlashCommandHandler(self, banner=BANNER)
        self.completion = CompletionEngine(
            [
                SlashCompletionProvider(self.commands),
                FileMentionCompletionProvider(Path.cwd()),
            ]
        )
        self._prompt_history: list[str] = []
        self._prompt_history_index: int | None = None
        self._prompt_history_draft = ""
        self._history_navigation_active = False
        self._ignore_prompt_change_events = 0
        self._utilization_timer = None
        self._utilization_visible = True
        self.metrics = SystemMetricsReader()
        self._power_min_watts: float | None = None
        self._power_max_watts: float | None = None
        self._generation_started_at: float | None = None
        self._generation_tokens_streamed = 0
        self._last_generation_tps: float | None = None

    def _active_model_ref(self) -> str:
        return active_model_ref(self.cfg)

    def _trace_runtime_context(self) -> dict[str, object]:
        return {
            "runtime": self.cfg.get("runtime", "llama_cpp"),
            "model": self._active_model_ref(),
            "device_profile": self.cfg.get("device", "auto"),
            "context_window_tokens": self.client.context_window_tokens if self.client else self.cfg.get("context_window_tokens", 2048),
            "gpu_layers": self.cfg.get("gpu_layers", 0),
            "host_arch": os.uname().machine if hasattr(os, "uname") else None,
        }

    def _begin_turn_trace(self, prompt: str) -> None:
        self._turn_counter += 1
        self._active_turn_id = f"turn-{int(time.time() * 1000)}-{self._turn_counter}"
        self._active_turn_started_at = time.monotonic()
        self._active_turn_prompt = prompt
        self._active_turn_generation_tokens = 0
        self._active_turn_tool_attempts = 0
        self._active_turn_tool_successes = 0
        self._active_turn_approval_requests = 0
        self._active_turn_approval_grants = 0
        self._active_turn_false_positive_commands = 0
        self._active_turn_hallucinated_commands = 0
        self._active_turn_recovered_after_resume = False
        if self.session_logger:
            self.session_logger.log_event(
                "turn_start",
                turn_id=self._active_turn_id,
                goal=prompt,
                mode=self.harness_state.mode,
                resumed_session=self._session_was_resumed,
                active_step=active.title if (active := active_step(self.harness_state)) else None,
                files_in_play=self.harness_state.files_in_play,
                **self._trace_runtime_context(),
            )

    def _finish_turn_trace(self, *, success: bool, status: str, error: str | None = None) -> None:
        if not self._active_turn_id:
            return
        resolution_ms = None
        if self._active_turn_started_at is not None:
            resolution_ms = round((time.monotonic() - self._active_turn_started_at) * 1000.0, 2)
        if self.session_logger:
            self.session_logger.log_event(
                "task_outcome",
                turn_id=self._active_turn_id,
                goal=self._active_turn_prompt,
                success=success,
                status=status,
                error=error,
                resolution_ms=resolution_ms,
                generation_tokens=self._active_turn_generation_tokens,
                tool_attempts=self._active_turn_tool_attempts,
                tool_successes=self._active_turn_tool_successes,
                approval_requests=self._active_turn_approval_requests,
                approval_grants=self._active_turn_approval_grants,
                false_positive_command_proposals=self._active_turn_false_positive_commands,
                hallucinated_command_proposals=self._active_turn_hallucinated_commands,
                recovered_after_resumed_session=self._active_turn_recovered_after_resume,
                **self._trace_runtime_context(),
            )
        self._active_turn_id = None
        self._active_turn_started_at = None
        self._active_turn_prompt = ""
        self._session_was_resumed = False

    def _has_any_configured_model(self) -> bool:
        return bool(self._active_model_ref())

    async def _init_client(self) -> None:
        mem_cfg = self.cfg.get("memory_guard", {})
        configured_ctx = int(self.cfg.get("context_window_tokens", 2048))
        configured_gpu_layers = int(self.cfg.get("gpu_layers", 99))
        self.client = create_runtime_client(self.cfg)
        if self.client.gpu_layers == 0:
            configured_gpu_layers = 0
        await self.client.start()
        if (
            self.client.context_window_tokens != configured_ctx
            or self.client.gpu_layers != configured_gpu_layers
        ):
            self.cfg["context_window_tokens"] = self.client.context_window_tokens
            self.cfg["gpu_layers"] = self.client.gpu_layers
            save_config(self.cfg)
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

    def _build_setup_screen(self, *, exit_on_cancel: bool) -> SetupScreen:
        return SetupScreen(
            model_options=discover_model_files(),
            installed_ollama_models=discover_installed_ollama_models(),
            hardware_info=detect_hardware_info(),
            recommended_ctx=recommended_context_window_tokens(),
            exit_on_cancel=exit_on_cancel,
        )

    async def _materialize_setup_model(self, setup_result: dict, log: RichLog) -> dict:
        status = self.query_one("#assistant-status", Static)

        def _set_status(text: str) -> None:
            status.remove_class("hidden")
            status.update(text)

        def _clear_status() -> None:
            status.update("")
            status.add_class("hidden")

        return await materialize_setup_model(
            setup_result,
            log,
            set_status=_set_status,
            clear_status=_clear_status,
        )

    async def _wait_for_screen_result(self, screen: Screen) -> object:
        """Wait for a screen result without requiring a worker context."""
        loop = asyncio.get_running_loop()
        result_future: asyncio.Future[object] = loop.create_future()

        def _on_dismiss(result: object) -> None:
            if not result_future.done():
                result_future.set_result(result)

        self.push_screen(screen, callback=_on_dismiss)
        return await result_future

    async def run_setup_command(self, log: RichLog) -> bool:
        previous_cfg = dict(self.cfg)
        had_runtime = bool(self.client or self.agent)
        if self.agent:
            self.persist_session_state(reason="setup_command_start")

        if self.client:
            try:
                await self.client.close()
            except Exception as exc:
                log.write(f"[yellow]Runtime stop warning:[/] {exc}")
                if self.session_logger:
                    self.session_logger.log_event("setup_runtime_stop_warning", error=str(exc))
        self.client = None
        self.agent = None
        self.loaded_files.clear()
        self.set_focus(None)
        self._render_token_counter()

        result = await self._wait_for_screen_result(self._build_setup_screen(exit_on_cancel=False))
        if not isinstance(result, dict) or not result.get("setup_complete"):
            if had_runtime:
                try:
                    await self._init_client()
                    log.write("[bold bright_white]Setup cancelled. Previous runtime restored.[/]")
                except Exception as exc:
                    log.write(f"[bold red]Setup cancelled; runtime restore failed:[/] {_format_error(exc)}")
                    if self.session_logger:
                        self.session_logger.log_event("setup_restore_failed", error=_format_error(exc))
            else:
                log.write("[bold bright_white]Setup cancelled.[/]")
            log.write("")
            prompt = self.query_one("#prompt", Input)
            prompt.disabled = False
            prompt.focus()
            self._render_token_counter(prompt.value)
            return False

        try:
            resolved_result = await self._materialize_setup_model(result, log)
        except Exception as exc:
            log.write(f"[bold red]Setup failed:[/] {_format_error(exc)}")
            log.write("")
            if self.session_logger:
                self.session_logger.log_event("setup_apply_failed", error=_format_error(exc))
            return False

        self.cfg.update(resolved_result)
        save_config(self.cfg)

        model_name = Path(self._active_model_ref()).name or self._active_model_ref() or "model"
        log.write(f"  [bold bright_white]Applying setup and loading {escape(model_name)}...[/]")
        status = self.query_one("#assistant-status", Static)
        status.update(f"[bold {ACCENT_GREEN}]loading {escape(model_name)}...[/]")
        status.remove_class("hidden")

        try:
            await self._init_client()
        except Exception as exc:
            status.update("")
            status.add_class("hidden")
            self.cfg = previous_cfg
            save_config(self.cfg)
            try:
                await self._init_client()
            except Exception:
                pass
            log.write(f"[bold red]Setup failed:[/] {_format_error(exc)}")
            log.write("")
            if self.session_logger:
                self.session_logger.log_event("setup_apply_failed", error=_format_error(exc))
            return False

        status.update("")
        status.add_class("hidden")
        self.loaded_files.clear()
        self.persist_session_state(reason="setup_command")
        self._render_token_counter()
        prompt = self.query_one("#prompt", Input)
        prompt.focus()
        log.write("[bold bright_white]Setup applied. Runtime restarted and context reset.[/]")
        log.write("")
        if self.session_logger:
            self.session_logger.log_event(
                "setup_applied",
                runtime=self.cfg.get("runtime", "llama_cpp"),
                model=self.cfg.get("model"),
                model_source=self.cfg.get("model_source", "local"),
                ollama_model=self.cfg.get("ollama_model"),
                device=self.cfg.get("device"),
                context_window_tokens=self.cfg.get("context_window_tokens"),
                gpu_layers=self.cfg.get("gpu_layers"),
            )
        return True

    @work(exclusive=True)
    async def run_setup_command_worker(self) -> None:
        log = self.query_one("#chat-log", RichLog)
        await self.run_setup_command(log)

    def compose(self) -> ComposeResult:
        yield RichLog(id="chat-log", wrap=True, markup=True, auto_scroll=True)
        yield Static("", id="assistant-status", classes="hidden")
        yield Static("", id="approval-bar", classes="hidden")
        with Container(id="bottom-stack"):
            yield Input(placeholder="> ", id="prompt")
            yield Static("", id="utilization-bar")
            yield Static("", id="token-counter")
            yield Static("", id="command-suggestions", classes="hidden")

    def on_mount(self) -> None:
        log = self.query_one("#chat-log", RichLog)
        log.write(BANNER)
        self._start_utilization_updates()
        self._render_utilization_bar()
        self._startup_sequence()

    @work(exclusive=True)
    async def _startup_sequence(self) -> None:
        log = self.query_one("#chat-log", RichLog)
        prompt = self.query_one("#prompt", Input)
        log_cfg = self.cfg.get("logging", {})
        if log_cfg.get("enabled", True):
            self.session_logger = SessionLogger(
                base_dir=Path(log_cfg.get("directory", "session_logs")),
                label=str(log_cfg.get("label", "open-jet")),
                metrics_interval_seconds=float(log_cfg.get("metrics_interval_seconds", 5)),
            )
            await self.session_logger.start()
            self.session_logger.log_event("app_mount", cwd=str(Path.cwd()))

        if self.force_setup or not self._has_any_configured_model():
            # First run without model: cancel exits app.
            # Explicit --setup mode: cancel returns to startup using existing config.
            setup_result = await self._wait_for_screen_result(
                self._build_setup_screen(exit_on_cancel=not self._has_any_configured_model())
            )
            if isinstance(setup_result, dict) and setup_result.get("setup_complete"):
                try:
                    setup_result = await self._materialize_setup_model(setup_result, log)
                except Exception as exc:
                    log.write(f"[bold red]Setup failed:[/] {_format_error(exc)}")
                    log.write("")
                    return
                self.cfg.update(setup_result)
                save_config(self.cfg)
            elif not self._has_any_configured_model():
                return
        elif not self.cfg.get("setup_complete"):
            # Backfill defaults for older configs created before setup wizard existed.
            self.cfg["setup_complete"] = True
            self.cfg.setdefault("model_source", "local")
            self.cfg.setdefault("runtime", "llama_cpp")
            self.cfg.setdefault("device", recommended_device())
            self.cfg.setdefault("context_window_tokens", recommended_context_window_tokens())
            self.cfg.setdefault("gpu_layers", recommended_gpu_layers(str(self.cfg.get("device", "auto"))))
            save_config(self.cfg)

        if (
            self.cfg.get("model_source") == "ollama"
            and self.cfg.get("ollama_model")
            and (not self.cfg.get("model") or not Path(str(self.cfg.get("model"))).is_file())
        ):
            try:
                resolved = await self._materialize_setup_model(dict(self.cfg), log)
            except Exception as exc:
                log.write(f"[bold red]Failed to resolve Ollama model:[/] {_format_error(exc)}")
                log.write("")
                return
            self.cfg.update(resolved)
            save_config(self.cfg)

        active_model = self._active_model_ref()
        log.write(f"  [bold bright_white]Loading {Path(active_model).name or active_model}...[/]")
        try:
            await self._init_client()
        except Exception as e:
            log.write(f"\n[bold red]Failed to start LLM:[/] {_format_error(e)}")
            if self.session_logger:
                self.session_logger.log_event("llm_start_error", error=_format_error(e))
            prompt.focus()
            return
        if self.session_logger:
            self.session_logger.log_event("llm_ready", model=active_model)
            self.session_logger.log_event(
                "llm_runtime_config",
                **self._trace_runtime_context(),
            )
        log.write(f"  [bold bright_white]Ready.[/]")
        if self.auto_resume:
            self._restore_session_state(log)
        self._restore_harness_state()
        log.write("")
        self._render_token_counter()
        prompt.focus()

    async def action_quit(self) -> None:
        if self._active_turn_id:
            self._finish_turn_trace(success=False, status="abandoned", error="application quit")
        self.persist_session_state(reason="quit")
        if self._utilization_timer:
            self._utilization_timer.stop()
            self._utilization_timer = None
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

        # While setup screen is open, route navigation keys directly to setup
        # actions so prompt focus can't steal them.
        if isinstance(self.screen, SetupScreen):
            setup = self.screen
            if event.key == "up":
                setup.action_prev_option()
                event.stop()
                return
            if event.key == "down":
                setup.action_next_option()
                event.stop()
                return
            if event.key in ("enter", "tab"):
                setup.action_advance()
                event.stop()
                return
            if event.key == "shift+tab":
                setup.action_back()
                event.stop()
                return
            if event.key == "escape":
                setup.action_cancel()
                event.stop()
                return
            return

        prompt = self.query_one("#prompt", Input)
        if self.focused is not prompt:
            return

        # While navigating history, arrows should keep navigating history.
        if event.key == "up" and self._prompt_history_index is not None:
            event.prevent_default()
            self._navigate_prompt_history(prompt, direction=-1)
            event.stop()
            return
        if event.key == "down" and self._prompt_history_index is not None:
            event.prevent_default()
            self._navigate_prompt_history(prompt, direction=1)
            event.stop()
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
        if event.key == "up":
            event.prevent_default()
            self._navigate_prompt_history(prompt, direction=-1)
            event.stop()
            return
        if event.key == "down":
            event.prevent_default()
            self._navigate_prompt_history(prompt, direction=1)
            event.stop()
            return
        if event.key in ("tab", "enter") and self.completion.state:
            next_value = self.completion.apply_selected(prompt.value)
            if next_value == prompt.value:
                # If completion has nothing new to apply, allow Enter to submit.
                if event.key == "enter":
                    return
                event.prevent_default()
                event.stop()
                return
            event.prevent_default()
            prompt.value = next_value
            self._update_completion_suggestions(prompt.value)
            prompt.action_end(select=False)
            self.call_after_refresh(self._collapse_prompt_selection)
            if event.key == "tab":
                prompt.focus()
            event.stop()
            return
        if event.key == "tab":
            event.prevent_default()
            if prompt.value.lstrip().startswith(("/", "@")):
                # Keep focus in chat input when tab is used for completion contexts.
                prompt.focus()
                self.call_after_refresh(self._collapse_prompt_selection)
            event.stop()
            return

    # -- Input ---------------------------------------------------------------

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id != "prompt":
            return
        if self._history_navigation_active:
            return
        if self._ignore_prompt_change_events > 0:
            self._ignore_prompt_change_events -= 1
            self._update_completion_suggestions(event.value)
            self._render_token_counter(event.value)
            return
        if self._prompt_history_index is not None:
            self._prompt_history_index = None
            self._prompt_history_draft = event.value
        self._update_completion_suggestions(event.value)
        self._render_token_counter(event.value)

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        if self._awaiting_approval:
            event.input.value = ""
            return

        text = event.value.strip()
        if not text:
            return
        self._record_prompt_history(text)
        event.input.value = ""
        self._clear_completion_suggestions()

        if await self.commands.maybe_handle(text):
            self._render_token_counter()
            return

        log = self.query_one("#chat-log", RichLog)
        if not self.agent:
            log.write("[yellow]LLM is not ready yet. Wait for Ready, or run /setup.[/]")
            log.write("")
            self._render_token_counter()
            return

        log.write(f"[bold green]> [/]{text}")
        log.write("")

        mentioned_files = _extract_file_mentions(text)
        self.harness_state = update_state_for_user_message(
            self.harness_state,
            text,
            files=mentioned_files,
        )
        self._begin_turn_trace(text)
        self.persist_harness_state()
        if self.session_logger:
            self.session_logger.log_event(
                "user_message",
                turn_id=self._active_turn_id,
                text=text,
                mentioned_files=mentioned_files,
                mode=self.harness_state.mode,
            )
        await self._load_mentioned_files_into_context(text, log)
        self.agent.add_user_message(text)
        self.persist_session_state(reason="user_message")
        self._render_token_counter()
        self._start_agent_turn()

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
        self.loaded_files[result.path] = {
            "path": result.path,
            "estimated_tokens": result.estimated_tokens,
            "loaded_tokens": result.returned_tokens,
            "truncated": result.truncated,
        }
        if result.path not in self.harness_state.files_in_play:
            self.harness_state.files_in_play.append(result.path)
            active = active_step(self.harness_state)
            if active and result.path not in active.files:
                active.files.append(result.path)
            self.persist_harness_state()
        log.write(f"[bold bright_white]Loaded @{mention_path} into context ({result.summary}).[/]")
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
        self.persist_session_state(reason="context_file_loaded")
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
                lines.append(f"[bold {ACCENT_GREEN}][underline]{item.label}[/underline][/]")
            else:
                lines.append(f"[bold {ACCENT_GREEN}]{item.label}[/]")
            if item.detail:
                lines[-1] += f" [bold bright_white]- {item.detail}[/]"
        bar.remove_class("hidden")
        bar.update("\n".join(lines) + "\n[bold bright_white]Up/Down to select, Tab or Enter to autocomplete[/]")

    def _clear_completion_suggestions(self) -> None:
        self.completion.clear()
        bar = self.query_one("#command-suggestions", Static)
        bar.add_class("hidden")
        bar.update("")

    def _render_token_counter(self, draft_text: str = "") -> None:
        counter = self.query_one("#token-counter", Static)
        if not self.agent:
            counter.update("[bold bright_white]tokens: 0/0[/]")
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
            color = "bright_white"
        counter.update(
            f"[{color}]tokens: {total}/{window} | prompt<= {budget.prompt_tokens} | remaining: {remaining} | harness: {self._turn_context_tokens}[/]"
        )

    def _start_utilization_updates(self) -> None:
        if self._utilization_timer:
            self._utilization_timer.stop()
        self._utilization_timer = self.set_interval(2.0, self._render_utilization_bar)

    def _render_utilization_bar(self) -> None:
        bar = self.query_one("#utilization-bar", Static)
        if not self._utilization_visible:
            bar.add_class("hidden")
            bar.update("")
            return
        bar.remove_class("hidden")
        cpu_pct = self.metrics.read_cpu_percent()
        mem = read_memory_snapshot()
        battery = self.metrics.read_battery_metrics()
        power_watts, power_pct = self.metrics.read_power_metrics()
        self._update_power_minmax(power_watts)

        cpu_text = self._format_percent("cpu", cpu_pct)
        mem_text = self._format_percent("mem", mem.used_percent if mem else None)
        tps_text = self._format_tps_text()
        power_text = self._format_power_text(power_watts, power_pct, battery)

        mem_detail = ""
        if mem and mem.total_mb is not None and mem.available_mb is not None:
            used_mb = max(0.0, mem.total_mb - mem.available_mb)
            mem_detail = f" ({used_mb / 1024.0:.1f}/{mem.total_mb / 1024.0:.1f} GB)"

        bar.update(
            f"[dim]util: {cpu_text} | {mem_text}{mem_detail} | {tps_text} | {power_text}[/]"
        )

    def _format_percent(self, label: str, pct: float | None) -> str:
        if pct is None:
            return f"{label} n/a"
        clamped = max(0.0, min(100.0, pct))
        return f"{label} {clamped:4.1f}%"

    def _format_power_text(
        self,
        watts: float | None,
        pct: float | None,
        battery: dict[str, float | str | None] | None = None,
    ) -> str:
        minmax = self._format_power_minmax()
        if battery:
            status_raw = str(battery.get("status") or "").strip().lower()
            capacity = battery.get("capacity_pct")
            remaining_hours = battery.get("remaining_hours")
            watts_now = battery.get("watts")

            base = "batt"
            if isinstance(capacity, (int, float)):
                base += f" {float(capacity):4.1f}%"

            if status_raw == "discharging" and isinstance(remaining_hours, (int, float)):
                base += f" {format_hours(float(remaining_hours))} left"
            elif status_raw == "charging" and isinstance(remaining_hours, (int, float)):
                base += f" {format_hours(float(remaining_hours))} to full"
            elif status_raw in {"full", "not charging"}:
                base += " full"
            elif status_raw:
                base += f" {status_raw}"

            if isinstance(watts_now, (int, float)):
                base += f" ({float(watts_now):.1f}W)"
            if minmax:
                base += f" {minmax}"
            return base

        if watts is None:
            return f"pwr n/a{(' ' + minmax) if minmax else ''}"
        if pct is not None:
            base = f"pwr {pct:4.1f}% ({watts:.1f}W)"
        else:
            base = f"pwr {watts:.1f}W"
        if minmax:
            base += f" {minmax}"
        return base

    def _format_tps_text(self) -> str:
        tps = self._current_tps()
        if tps is None:
            return "tps n/a"
        return f"tps {tps:.1f}"

    def _current_tps(self) -> float | None:
        if self._thinking_timer is not None and self._generation_started_at is not None:
            elapsed = time.monotonic() - self._generation_started_at
            if elapsed <= 0:
                return None
            return self._generation_tokens_streamed / elapsed
        return self._last_generation_tps

    def _update_power_minmax(self, watts: float | None) -> None:
        if watts is None:
            return
        if self._power_min_watts is None or watts < self._power_min_watts:
            self._power_min_watts = watts
        if self._power_max_watts is None or watts > self._power_max_watts:
            self._power_max_watts = watts

    def _format_power_minmax(self) -> str:
        if self._power_min_watts is None or self._power_max_watts is None:
            return ""
        return f"[min {self._power_min_watts:.1f}W max {self._power_max_watts:.1f}W]"

    def set_utilization_visible(self, visible: bool) -> None:
        self._utilization_visible = bool(visible)
        self._render_utilization_bar()

    def toggle_utilization_visible(self) -> bool:
        self._utilization_visible = not self._utilization_visible
        self._render_utilization_bar()
        return self._utilization_visible

    def is_utilization_visible(self) -> bool:
        return self._utilization_visible

    def runtime_status_snapshot(self) -> dict:
        if not self.agent:
            return {"ready": False}

        window = self.client.context_window_tokens if self.client else int(self.cfg.get("context_window_tokens", 2048))
        budget = self.agent.context_budget() or derive_context_budget(window)
        current = self.agent.estimated_context_tokens()
        remaining = max(0, budget.prompt_tokens - current)
        mem = read_memory_snapshot()
        active = active_step(self.harness_state)
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
            "harness_mode": self.harness_state.mode,
            "harness_active_step": active.title if active else None,
            "harness_docs": list(self._turn_context_docs),
            "harness_doc_tokens": self._turn_context_tokens,
        }

    def refresh_token_counter(self) -> None:
        self._render_token_counter()

    def _restore_session_state(self, log: RichLog) -> bool:
        if not self.agent:
            return False
        state = self.state_store.load()
        if not state:
            return False
        messages = state.get("messages")
        if not isinstance(messages, list) or not messages:
            return False
        valid_messages: list[dict] = []
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            role = msg.get("role")
            if not isinstance(role, str):
                continue
            if "content" in msg and not isinstance(msg.get("content"), str):
                continue
            valid_messages.append(msg)
        if not valid_messages:
            return False

        first = valid_messages[0]
        if first.get("role") != "system":
            valid_messages = [{"role": "system", "content": self.cfg.get("system_prompt", "")}, *valid_messages]

        self.agent.messages = valid_messages
        self._replay_restored_history(log, self.agent.messages)
        self._seed_prompt_history_from_messages(self.agent.messages)
        loaded_files = state.get("loaded_files")
        if isinstance(loaded_files, dict):
            self.loaded_files = loaded_files
        else:
            self.loaded_files = {}
        harness_payload = state.get("harness_state")
        if isinstance(harness_payload, dict):
            self.harness_state = HarnessState.from_dict(harness_payload)
        log.write(
            "  [bold bright_white]"
            f"Resumed previous session: {max(0, len(self.agent.messages) - 1)} messages, "
            f"{len(self.loaded_files)} loaded files."
            "[/]"
        )
        if self.session_logger:
            self.session_logger.log_event(
                "session_resume",
                messages=max(0, len(self.agent.messages) - 1),
                loaded_files=len(self.loaded_files),
                state_path=str(self.state_store.path),
                restored_ok=True,
                prior_session_id=state.get("session_id"),
                prior_saved_at=state.get("saved_at"),
                resume_gap_seconds=(
                    round(max(0.0, time.time() - float(state.get("saved_at", 0.0))), 2)
                    if state.get("saved_at")
                    else None
                ),
                **self._trace_runtime_context(),
            )
        self._session_was_resumed = True
        return True

    def _restore_harness_state(self) -> None:
        try:
            self.harness_state = self.harness_store.load()
        except Exception:
            self.harness_state = HarnessState()

    def _replay_restored_history(self, log: RichLog, messages: list[dict]) -> None:
        for msg in messages:
            role = msg.get("role")
            if role == "user":
                text = msg.get("content", "")
                if isinstance(text, str) and text.strip():
                    log.write(f"[bold green]> [/]{text}")
                    log.write("")
                continue

            if role == "assistant":
                text = msg.get("content", "")
                if isinstance(text, str) and text:
                    self._write_text_block(log, text)
                if not msg.get("tool_calls"):
                    log.write("")
                continue

            if role == "tool":
                text = msg.get("content", "")
                if isinstance(text, str) and text:
                    self._write_tool_result(log, text)

    def _seed_prompt_history_from_messages(self, messages: list[dict]) -> None:
        self._prompt_history = []
        for msg in messages:
            if msg.get("role") != "user":
                continue
            text = msg.get("content")
            if isinstance(text, str):
                normalized = text.strip()
                if normalized:
                    self._prompt_history.append(normalized)
        self._prompt_history_index = None
        self._prompt_history_draft = ""

    def _record_prompt_history(self, text: str) -> None:
        normalized = text.strip()
        if not normalized:
            return
        self._prompt_history.append(normalized)
        self._prompt_history_index = None
        self._prompt_history_draft = ""

    def _navigate_prompt_history(self, prompt: Input, *, direction: int) -> None:
        if not self._prompt_history:
            return
        if direction not in (-1, 1):
            return

        if direction == -1:
            if self._prompt_history_index is None:
                self._prompt_history_draft = prompt.value
                self._prompt_history_index = len(self._prompt_history) - 1
            elif self._prompt_history_index > 0:
                self._prompt_history_index -= 1
            next_value = self._prompt_history[self._prompt_history_index]
        else:
            if self._prompt_history_index is None:
                return
            if self._prompt_history_index < len(self._prompt_history) - 1:
                self._prompt_history_index += 1
                next_value = self._prompt_history[self._prompt_history_index]
            else:
                self._prompt_history_index = None
                next_value = self._prompt_history_draft

        self._history_navigation_active = True
        try:
            self._ignore_prompt_change_events += 1
            prompt.value = next_value
            self._update_completion_suggestions(prompt.value)
            self._render_token_counter(prompt.value)
            prompt.action_end(select=False)
            self.call_after_refresh(self._collapse_prompt_selection)
        finally:
            self._history_navigation_active = False

    def _write_text_block(self, log: RichLog, text: str) -> None:
        buf = text
        while "\n" in buf:
            line, buf = buf.split("\n", 1)
            log.write(line)
        if buf:
            log.write(buf)

    def _write_tool_result(self, log: RichLog, result: str) -> None:
        lines = result.splitlines()
        for line in lines[:20]:
            log.write(f"  [bold bright_white]{line}[/]")
        if len(lines) > 20:
            log.write(f"  [bold bright_white]... ({len(lines) - 20} more lines)[/]")
        log.write("")

    def persist_session_state(self, *, reason: str) -> None:
        if not self.agent:
            return
        payload = {
            "version": 1,
            "saved_at": time.time(),
            "reason": reason,
            "session_id": self.session_logger.session_id if self.session_logger else None,
            "model": self.cfg.get("model"),
            "device": self.cfg.get("device", "auto"),
            "context_window_tokens": self.client.context_window_tokens if self.client else self.cfg.get("context_window_tokens", 2048),
            "messages": self.agent.messages,
            "loaded_files": self.loaded_files,
            "harness_state": self.harness_state.to_dict(),
        }
        try:
            self.state_store.save(payload)
        except Exception as exc:
            if self.session_logger:
                self.session_logger.log_event("session_state_save_error", reason=reason, error=str(exc))

    def persist_harness_state(self) -> None:
        self.harness_state.updated_at = time.time()
        try:
            self.harness_store.save(self.harness_state)
        except Exception as exc:
            if self.session_logger:
                self.session_logger.log_event("harness_state_save_error", error=str(exc))

    def available_harness_skills(self) -> list[str]:
        return available_skill_names(Path.cwd())

    def set_harness_mode(self, mode: str) -> None:
        self.harness_state = set_mode(self.harness_state, mode)
        self.persist_harness_state()
        self.persist_session_state(reason=f"harness_mode:{mode}")
        self._render_token_counter()

    def set_harness_skills(self, names: list[str]) -> tuple[list[str], list[str]]:
        available = set(self.available_harness_skills())
        normalized = [normalize_skill_name(name) for name in names]
        applied = [name for name in normalized if name in available]
        missing = [name for name in normalized if name and name not in available]
        self.harness_state = set_preferred_skills(self.harness_state, applied)
        self.persist_harness_state()
        self.persist_session_state(reason="harness_skills_set")
        self._render_token_counter()
        return applied, missing

    def clear_harness_skills(self) -> None:
        self.harness_state = clear_preferred_skills(self.harness_state)
        self.persist_harness_state()
        self.persist_session_state(reason="harness_skills_cleared")
        self._render_token_counter()

    def harness_active_step(self) -> str | None:
        active = active_step(self.harness_state)
        return active.title if active else None

    def advance_harness_step(self) -> None:
        self.harness_state = advance_step(self.harness_state)
        self.persist_harness_state()
        self.persist_session_state(reason="harness_step_advanced")
        self._render_token_counter()

    def split_harness_step(self) -> None:
        self.harness_state = split_active_step(self.harness_state)
        self.persist_harness_state()
        self.persist_session_state(reason="harness_step_split")
        self._render_token_counter()

    def _append_harness_memory(
        self,
        previous_state: HarnessState,
        current_state: HarnessState,
        *,
        tool_events: list[dict],
        assistant_text: str,
    ) -> None:
        root = Path.cwd()
        prev_active = active_step(previous_state)
        curr_active = active_step(current_state)
        if prev_active and prev_active.status != "done":
            now_done = next((step for step in current_state.plan if step.id == prev_active.id and step.status == "done"), None)
            if now_done:
                append_memory_entry(
                    root,
                    "session.md",
                    [
                        f"- completed_step: {prev_active.title}",
                        f"- mode: {current_state.mode}",
                        f"- files: {', '.join(prev_active.files) if prev_active.files else 'n/a'}",
                        f"- next: {curr_active.title if curr_active else 'n/a'}",
                    ],
                )
        verification = current_state.last_verification
        if verification.get("status") == "fail":
            append_memory_entry(
                root,
                "failures.md",
                [
                    f"- mode: {current_state.mode}",
                    f"- step: {prev_active.title if prev_active else 'n/a'}",
                    f"- pattern: verification failed",
                    f"- detail: {verification.get('summary', 'n/a')}",
                    f"- command: {verification.get('command', 'n/a')}",
                ],
            )
        if any(token in current_state.goal.lower() for token in ("architecture", "design", "harness")) and tool_events:
            append_memory_entry(
                root,
                "decisions.md",
                [
                    f"- goal: {current_state.goal}",
                    f"- last_action: {current_state.last_action.get('summary', 'n/a')}",
                    f"- verification: {verification.get('status', 'not_run')}",
                ],
            )

    # -- Agent turn ----------------------------------------------------------

    def _start_agent_turn(self, recovery_attempted: bool = False) -> None:
        self._prepare_turn_context()
        self._generation_worker = self.run_agent_turn(recovery_attempted=recovery_attempted)

    def _prepare_turn_context(self) -> None:
        if not self.agent:
            return
        window = self.client.context_window_tokens if self.client else int(self.cfg.get("context_window_tokens", 2048))
        context = build_turn_context(
            root=Path.cwd(),
            state=self.harness_state,
            current_context_tokens=self.agent.persistent_context_tokens(),
            effective_window=window,
            memory_snapshot=read_memory_snapshot(),
        )
        self.agent.set_turn_context(context.messages)
        self._turn_context_docs = context.docs_loaded
        self._turn_context_tokens = context.docs_tokens
        if self.session_logger:
            self.session_logger.log_event(
                "turn_context_prepared",
                mode=self.harness_state.mode,
                active_step=active.title if (active := active_step(self.harness_state)) else None,
                docs=context.docs_loaded,
                docs_tokens=context.docs_tokens,
                usable_prompt_budget=context.budget.usable_prompt_budget,
                remaining_budget=context.budget.remaining_budget,
                ram_factor=context.budget.ram_factor,
            )

    def action_stop_generation(self) -> None:
        if self._awaiting_approval or isinstance(self.screen, SetupScreen):
            return
        if not self._generation_worker or self._generation_worker.is_finished:
            return
        self._generation_worker.cancel()
        log = self.query_one("#chat-log", RichLog)
        log.write("[yellow]Generation stopped.[/]")
        log.write("")
        if self.session_logger:
            self.session_logger.log_event(
                "generation_interrupted",
                turn_id=self._active_turn_id,
                source="escape",
                during_generation=True,
                during_tool=self._awaiting_approval,
            )
        self._finish_turn_trace(success=False, status="interrupted", error="generation stopped by user")
        self._render_token_counter()

    @work(exclusive=True)
    async def run_agent_turn(self, recovery_attempted: bool = False) -> None:
        log = self.query_one("#chat-log", RichLog)
        pending_tool_calls: list[ToolCall] = []
        tool_events: list[dict] = []
        condense_requested = False
        text_buf = ""
        assistant_turn_text = ""
        thinking_token = self._start_thinking()
        current_worker = get_current_worker()

        try:
            async for event in self.agent.run_turn():
                if event.kind == ActionKind.TEXT:
                    text_buf += event.text
                    assistant_turn_text += event.text
                    self._generation_tokens_streamed += estimate_tokens(event.text)
                    self._active_turn_generation_tokens += estimate_tokens(event.text)
                    # Flush complete lines as they arrive
                    while "\n" in text_buf:
                        line, text_buf = text_buf.split("\n", 1)
                        log.write(line)
                elif event.kind == ActionKind.TOOL_REQUEST:
                    pending_tool_calls.append(event.tool_call)
                    if self.session_logger and event.tool_call:
                        event_data = {
                            "turn_id": self._active_turn_id,
                            "proposal_id": event.tool_call.id,
                            "tool": event.tool_call.name,
                            "arguments": event.tool_call.arguments,
                        }
                        if event.tool_call.name == "shell":
                            classification = _classify_shell_command(str(event.tool_call.arguments.get("command", "")))
                            if classification["false_positive_proposal"]:
                                self._active_turn_false_positive_commands += 1
                            if classification["hallucinated_command"]:
                                self._active_turn_hallucinated_commands += 1
                            event_data.update(classification)
                            self.session_logger.log_event("command_eval", **event_data)
                        self.session_logger.log_event(
                            "tool_request",
                            **event_data,
                        )
                elif event.kind == ActionKind.CONDENSE:
                    condense_requested = True
                    if self.session_logger:
                        self.session_logger.log_event("auto_condense_requested", turn_id=self._active_turn_id, reason=event.text)
                elif event.kind == ActionKind.ERROR:
                    if not recovery_attempted and self._is_recoverable_runtime_error(event.text):
                        recovered = await self._recover_runtime(log, event.text)
                        if recovered:
                            self._active_turn_recovered_after_resume = self._session_was_resumed
                            self._start_agent_turn(recovery_attempted=True)
                            return
                    log.write(f"\n[bold red]error:[/] {event.text}")
                    if self.session_logger:
                        self.session_logger.log_event("agent_error", turn_id=self._active_turn_id, error=event.text)
                    self._finish_turn_trace(success=False, status="agent_error", error=event.text)
                    return
                elif event.kind == ActionKind.DONE:
                    if text_buf:
                        log.write(text_buf)
                        text_buf = ""
                    log.write("")

            # Flush any remaining text
            if text_buf:
                log.write(text_buf)
        except asyncio.CancelledError:
            return
        finally:
            self._stop_thinking(thinking_token)
            if self._generation_worker is current_worker:
                self._generation_worker = None

        if self.session_logger and assistant_turn_text.strip():
            self.session_logger.log_event(
                "assistant_message",
                turn_id=self._active_turn_id,
                text=assistant_turn_text,
                output_tokens=estimate_tokens(assistant_turn_text),
            )

        if condense_requested:
            result = await self.agent.condense_context()
            log.write(f"  [bold bright_white]{result}[/]")
            log.write("")
            self.persist_session_state(reason="auto_condense")
            self.persist_harness_state()
            self._start_agent_turn()
            return

        for tc in pending_tool_calls:
            try:
                event = await self._handle_tool_call(tc, log)
                if event:
                    tool_events.append(event)
            except Exception as exc:
                log.write(f"[bold red]tool error ({tc.name}):[/] {exc}")
                log.write("")
                if self.session_logger:
                    self.session_logger.log_event(
                        "tool_error",
                        tool=tc.name,
                        arguments=tc.arguments,
                        error=str(exc),
                    )
                if self.agent:
                    self.agent.complete_tool_call(tc, f"Tool execution failed: {exc}")

        if pending_tool_calls:
            previous_state = HarnessState.from_dict(self.harness_state.to_dict())
            self.harness_state = update_state_after_turn(
                self.harness_state,
                tool_events=tool_events,
                assistant_text=assistant_turn_text,
            )
            self._append_harness_memory(previous_state, self.harness_state, tool_events=tool_events, assistant_text=assistant_turn_text)
            self.persist_harness_state()
            self.persist_session_state(reason="assistant_turn_with_tools")
            self._start_agent_turn()
        else:
            previous_state = HarnessState.from_dict(self.harness_state.to_dict())
            self.harness_state = update_state_after_turn(
                self.harness_state,
                tool_events=tool_events,
                assistant_text=assistant_turn_text,
            )
            self._append_harness_memory(previous_state, self.harness_state, tool_events=tool_events, assistant_text=assistant_turn_text)
            self.persist_harness_state()
            self.persist_session_state(reason="assistant_turn_done")
            self._finish_turn_trace(success=True, status="completed")
            self._render_token_counter()

    async def _handle_tool_call(self, tc: ToolCall, log: RichLog) -> dict | None:
        if tc.name not in allowed_tools_for_mode(self.harness_state.mode):
            denied = f"Tool {tc.name} is not allowed in {self.harness_state.mode} mode."
            log.write(f"[yellow]{denied}[/]")
            log.write("")
            if self.agent:
                self.agent.complete_tool_call(tc, denied)
            return {"tool": tc.name, "ok": False, "summary": denied, "target": format_tool_args(tc)}
        needs_confirm = self.agent.needs_confirmation(tc)

        if needs_confirm:
            self._active_turn_approval_requests += 1
            log.write(f"[yellow]{tc.name}:[/]")
            for preview_line in self._tool_preview_lines(tc):
                log.write(f"  [bold bright_white]{preview_line}[/]")
            approved = await self._wait_for_tool_approval(tc)
            approval_latency_ms = None
            if self._approval_started_at is not None:
                approval_latency_ms = round((time.monotonic() - self._approval_started_at) * 1000.0, 2)
            self._approval_started_at = None
            if not approved:
                log.write("[red]  denied[/]")
                log.write("")
                if self.session_logger:
                    self.session_logger.log_event(
                        "tool_approval",
                        turn_id=self._active_turn_id,
                        proposal_id=tc.id,
                        tool=tc.name,
                        approved=False,
                        human_latency_ms=approval_latency_ms,
                        arguments=tc.arguments,
                    )
                self.agent.complete_tool_call(tc, "User denied this action.")
                self.persist_session_state(reason=f"tool_denied:{tc.name}")
                return {
                    "tool": tc.name,
                    "ok": False,
                    "summary": "User denied this action.",
                    "target": format_tool_args(tc),
                }
            log.write("[green]  approved[/]")
            self._active_turn_approval_grants += 1
            if self.session_logger:
                self.session_logger.log_event(
                    "tool_approval",
                    turn_id=self._active_turn_id,
                    proposal_id=tc.id,
                    tool=tc.name,
                    approved=True,
                    human_latency_ms=approval_latency_ms,
                    arguments=tc.arguments,
                )

        if tc.name == "load_file":
            self._clamp_load_file_tool_budget(tc)

        self._active_turn_tool_attempts += 1
        t0 = time.monotonic()
        execution = await execute_tool(tc)
        result = execution.output
        meta = execution.meta
        result_for_context, clipped_tool_result = self._fit_tool_result_to_budget(result)
        duration_ms = round((time.monotonic() - t0) * 1000.0, 2)
        if execution.ok:
            self._active_turn_tool_successes += 1
        if self.session_logger:
            self.session_logger.log_tool_result(
                tc.name,
                result,
                turn_id=self._active_turn_id,
                proposal_id=tc.id,
                duration_ms=duration_ms,
                arguments=tc.arguments,
                context_result_clipped=clipped_tool_result,
                **meta,
            )
        # Show output inline in the chat
        for line in result_for_context.splitlines()[:20]:
            log.write(f"  [bold bright_white]{line}[/]")
        if len(result_for_context.splitlines()) > 20:
            log.write(
                "  [bold bright_white]"
                f"... ({len(result_for_context.splitlines()) - 20} more lines)[/]"
            )
        log.write("")
        log.scroll_end(animate=False)
        self.agent.complete_tool_call(tc, result_for_context)
        self.persist_session_state(reason=f"tool_result:{tc.name}")
        return {
            "tool": tc.name,
            "ok": bool(meta.get("ok", True)),
            "summary": result.splitlines()[0] if result else "",
            "target": format_tool_args(tc),
            "verification": tc.name == "shell" and shell_command_is_verification(str(tc.arguments.get("command", ""))),
            "command": tc.arguments.get("command") if isinstance(tc.arguments, dict) else None,
        }

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

    def _fit_tool_result_to_budget(self, result: str) -> tuple[str, bool]:
        if not result:
            return result, False

        budget_tokens = self._remaining_prompt_tokens()
        if estimate_tokens(result) <= budget_tokens:
            return result, False

        prefix = "...[tool output truncated]\n"
        max_chars = max(256, budget_tokens * 4)
        clipped = result[-max_chars:]
        candidate = prefix + clipped

        while estimate_tokens(candidate) > budget_tokens and len(clipped) > 64:
            clipped = clipped[max(64, int(len(clipped) * 0.85)):]
            candidate = prefix + clipped

        return candidate, True

    def _is_recoverable_runtime_error(self, error_text: str) -> bool:
        lowered = error_text.lower()
        needles = (
            "connecterror",
            "connection refused",
            "connection reset",
            "remoteprotocolerror",
            "readtimeout",
            "timed out",
            "server disconnected",
            "llama-server exited",
            "trtllm-serve exited",
            "502",
            "503",
            "504",
        )
        return any(needle in lowered for needle in needles)

    async def _recover_runtime(self, log: RichLog, error_text: str) -> bool:
        if not self.client:
            return False
        log.write("[yellow]LLM runtime interrupted. Restarting runtime once and retrying...[/]")
        if self.session_logger:
            self.session_logger.log_event("llm_recovery_attempt", error=error_text)
        try:
            await self.client.reset_kv_cache()
        except Exception as exc:
            log.write(f"[bold red]Runtime recovery failed:[/] {exc}")
            log.write("")
            if self.session_logger:
                self.session_logger.log_event("llm_recovery_failed", error=str(exc))
            return False
        log.write("[bold bright_white]Runtime recovered. Retrying turn.[/]")
        log.write("")
        if self.session_logger:
            self.session_logger.log_event("llm_recovery_succeeded")
        return True

    def _start_thinking(self) -> int:
        status = self.query_one("#assistant-status", Static)
        self._thinking_token += 1
        self._thinking_idx = 0
        self._generation_started_at = time.monotonic()
        self._generation_tokens_streamed = 0
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
        if self._generation_started_at is not None:
            elapsed = time.monotonic() - self._generation_started_at
            if elapsed > 0 and self._generation_tokens_streamed > 0:
                self._last_generation_tps = self._generation_tokens_streamed / elapsed
        self._generation_started_at = None
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
        self._approval_started_at = time.monotonic()
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
            self._approval_started_at = None
            bar.add_class("hidden")
            bar.update("")
            prompt.disabled = False
            prompt.focus()

    def _render_approval_bar(self) -> None:
        if not self._awaiting_approval or not self._approval_tool_call:
            return
        bar = self.query_one("#approval-bar", Static)
        summary = self._approval_summary_text(self._approval_tool_call)
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
            f"[bold yellow]Tool request:[/] {summary}\n"
            f"Use [bold]←[/]/[bold]→[/] then [bold]Enter[/]   {approve}  {deny}"
        )

    def _approval_summary_text(self, tc: ToolCall) -> str:
        if tc.name == "write_file":
            path = str(tc.arguments.get("path", "")).strip()
            content = str(tc.arguments.get("content", ""))
            return f"write_file -> {escape(path)} ({len(content)} bytes)"
        if tc.name == "edit_file":
            path = str(tc.arguments.get("path", "")).strip()
            return f"edit_file -> {escape(path)}"
        if tc.name == "shell":
            command = str(tc.arguments.get("command", "")).strip()
            if len(command) > 120:
                command = command[:117] + "..."
            return f"shell -> {escape(command)}"
        return escape(f"{tc.name} -> {format_tool_args(tc)}")

    def _resolve_approval(self, approved: bool) -> None:
        if self._approval_future and not self._approval_future.done():
            self._approval_future.set_result(approved)

    def _collapse_prompt_selection(self) -> None:
        prompt = self.query_one("#prompt", Input)
        cursor = prompt.cursor_position
        prompt.selection = Selection(cursor, cursor)

    def _tool_preview_lines(self, tc: ToolCall) -> list[str]:
        if tc.name == "shell":
            command = str(tc.arguments.get("command", "")).strip()
            timeout_seconds = tc.arguments.get("timeout_seconds")
            if len(command) > 200:
                command = command[:197] + "..."
            if isinstance(timeout_seconds, int):
                return [f"command: {command}", f"timeout_seconds: {timeout_seconds}"]
            return [f"command: {command}"]
        if tc.name == "write_file":
            path = str(tc.arguments.get("path", "")).strip()
            content = str(tc.arguments.get("content", ""))
            preview = content.replace("\r\n", "\n").replace("\r", "\n")
            lines = [f"path: {path}", f"bytes: {len(content)}", "content:"]
            lines.extend(escape(line) for line in preview.split("\n"))
            return lines
        if tc.name == "edit_file":
            path = str(tc.arguments.get("path", "")).strip()
            old_s = str(tc.arguments.get("old_string", ""))
            new_s = str(tc.arguments.get("new_string", ""))
            replace_all_flag = tc.arguments.get("replace_all", False)
            lines = [f"path: {path}"]
            if replace_all_flag:
                lines.append("replace_all: true")
            lines.append("old_string:")
            lines.extend(f"  {escape(l)}" for l in old_s.split("\n"))
            lines.append("new_string:")
            lines.extend(f"  {escape(l)}" for l in new_s.split("\n"))
            return lines
        return [str(format_tool_args(tc))]

    def _approval_preview_text(self, tc: ToolCall) -> str:
        lines = self._tool_preview_lines(tc)
        joined = "\n".join(lines)
        if tc.name in ("write_file", "edit_file"):
            return joined
        if len(joined) > 280:
            return joined[:277] + "..."
        return joined


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

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="open-jet offline agentic TUI")
    parser.add_argument(
        "--setup",
        action="store_true",
        help="start in setup wizard mode before launching the chat UI",
    )
    args = parser.parse_args(argv)

    app = OpenJetApp(force_setup=args.setup)
    app.run(mouse=False, inline=True, inline_no_clear=True)


if __name__ == "__main__":
    main()
