"""open-jet terminal chat UI with native terminal scrollback."""

from __future__ import annotations

import argparse
import asyncio
import html
import os
import re
import shlex
import shutil
import signal
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.filters import Condition
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.patch_stdout import patch_stdout
from prompt_toolkit.styles import Style
from rich.console import Console
from rich.markup import escape
from rich.rule import Rule

from .airgap import airgapped_from_cfg, set_airgapped
from .agent import ActionKind, Agent, ToolCall
from .commands import SlashCommandHandler
from .completion import CompletionEngine, FileMentionCompletionProvider, SlashCompletionProvider
from .config import load_config, save_config
from .executor import load_file
from .harness import (
    HarnessSessionStore,
    HarnessState,
    active_step,
    advance_step,
    allowed_tools_for_mode,
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
from .harness_debug import write_debug_context_snapshot, write_debug_runtime_messages
from .hardware import (
    detect_hardware_info,
    effective_hardware_info,
    recommended_context_window_tokens,
    recommended_device,
    recommended_gpu_layers,
)
from .multimodal import (
    build_user_content,
    content_to_plain_text,
    estimate_message_content_tokens,
    extract_pasted_image_paths,
    is_image_path,
    is_supported_message_content,
)
from .ollama_setup import discover_installed_ollama_models, materialize_setup_model
from .persistent_memory import build_system_prompt
from .runtime_client import RuntimeClient
from .runtime_limits import derive_context_budget, estimate_tokens, read_memory_snapshot
from .runtime_registry import active_model_ref, create_runtime_client
from .session_logging import BroadcastConfig, SessionLogger
from .session_state import SessionStateStore
from .setup import ACCENT_GREEN, discover_model_files, run_setup_wizard
from .system_metrics import SystemMetricsReader, format_hours
from .tool_executor import execute_tool, format_tool_args


def _format_error(exc: Exception) -> str:
    text = str(exc).strip()
    return text or f"{type(exc).__name__} (no message)"


def _plain_markup(text: str) -> str:
    return re.sub(r"\[[^\]]*\]", "", text)


def _normalize_telemetry_slug(value: str | None, *, default: str = "unknown") -> str:
    text = (value or "").strip().lower()
    if not text:
        return default
    cleaned = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    return cleaned or default


def _telemetry_backend(cfg: dict[str, Any]) -> str:
    model_source = str(cfg.get("model_source", "") or "").strip().lower()
    runtime = str(cfg.get("runtime", "llama_cpp") or "").strip().lower()
    if model_source:
        return _normalize_telemetry_slug(model_source)
    if runtime == "openai_compatible":
        return "openai_compatible"
    if runtime == "openrouter":
        return "openrouter"
    return _normalize_telemetry_slug(runtime)


def _telemetry_model_fields(model_ref: str) -> tuple[str, str]:
    model_name = Path(model_ref).name or model_ref or "unknown"
    base = re.sub(r"\.(gguf|bin|safetensors)$", "", model_name, flags=re.IGNORECASE)
    parts = [part for part in re.split(r"[-_]+", base) if part]
    variant_tokens: list[str] = []
    id_tokens: list[str] = []
    variant_started = False
    for part in parts:
        lower = part.lower()
        if variant_started or re.fullmatch(r"(q\d.*|awq|gptq|fp\d+|bf16|int\d+)", lower):
            variant_started = True
            variant_tokens.append(lower)
        else:
            id_tokens.append(lower)
    model_id = _normalize_telemetry_slug("-".join(id_tokens) or base)
    model_variant = _normalize_telemetry_slug("-".join(variant_tokens), default="unknown")
    return model_id, model_variant


def _telemetry_hardware_fields(cfg: dict[str, Any]) -> dict[str, object]:
    detected = detect_hardware_info()
    hardware = effective_hardware_info(
        str(cfg.get("hardware_profile", "auto")),
        detected,
        str(cfg.get("hardware_override", "")).strip() or None,
    )
    label = hardware.label.strip() or "unknown"
    lowered = label.lower()
    if "jetson" in lowered:
        family = "jetson"
    elif hardware.has_cuda:
        family = "cuda"
    else:
        family = "cpu"
    return {
        "hardware_class": _normalize_telemetry_slug(label),
        "hardware_family": family,
        "accelerator": "cuda" if hardware.has_cuda else "cpu",
        "system_memory_total_mb": round(hardware.total_ram_gb * 1024.0, 2),
    }


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


def _shell_command_category(primary_command: str) -> str:
    primary = primary_command.strip().lower()
    if not primary:
        return "empty"
    if primary in _SHELL_BUILTINS:
        return "builtin"
    if primary in {"git", "gh"}:
        return "git"
    if primary in {"pytest", "unittest", "nose"}:
        return "test"
    if primary in {"python", "python3", "uv", "pip", "pip3"}:
        return "python"
    if primary in {"cargo", "rustc"}:
        return "rust"
    if primary in {"npm", "pnpm", "yarn", "node"}:
        return "node"
    if primary in {"make", "cmake", "ninja"}:
        return "build"
    if primary in {"bash", "sh", "zsh"}:
        return "shell"
    if primary in {"ls", "cat", "find", "grep", "rg"}:
        return "filesystem"
    return "other"


BANNER = r"""[bold green]
   ___                    _        _
  / _ \ _ __   ___ _ __  (_) ___  | |_
 | | | | '_ \ / _ \ '_ \ | |/ _ \ | __|
 | |_| | |_) |  __/ | | || |  __/ | |_
  \___/| .__/ \___|_| |_|/ |\___|  \__|
       |_|              |__/
[/]"""


@dataclass
class Selection:
    start: int
    end: int


class LogView:
    def __init__(self, console: Console) -> None:
        self.console = console
        self._entries: list[object] = []

    def write(self, content: object, **_: object) -> "LogView":
        self._entries.append(content)
        self.console.print(content)
        return self

    def clear(self) -> None:
        self._entries.clear()

    def scroll_page_up(self, animate: bool = False) -> None:
        return

    def scroll_page_down(self, animate: bool = False) -> None:
        return

    def scroll_home(self, animate: bool = False) -> None:
        return

    def scroll_end(self, animate: bool = False) -> None:
        return


class StatusWidget:
    def __init__(self) -> None:
        self.text = ""
        self.hidden = True

    def update(self, text: str) -> None:
        self.text = text

    def add_class(self, name: str) -> None:
        if name == "hidden":
            self.hidden = True

    def remove_class(self, name: str) -> None:
        if name == "hidden":
            self.hidden = False


class PromptWidget:
    def __init__(self, placeholder: str = "> ") -> None:
        self.placeholder = placeholder
        self.value = ""
        self.disabled = False
        self.selection = Selection(0, 0)

    @property
    def cursor_position(self) -> int:
        return len(self.value)

    def action_end(self, select: bool = False) -> None:
        return

    def focus(self) -> None:
        return


class OpenJetCompleter(Completer):
    def __init__(self, app: "OpenJetApp") -> None:
        self.app = app

    def get_completions(self, document, complete_event):
        del complete_event
        state = self.app.completion.refresh(document.text)
        if not state:
            return
        for item in state.items:
            yield Completion(
                item.insert[state.start:],
                start_position=state.start - len(document.text),
                display=item.label,
                display_meta=item.detail,
            )


class OpenJetApp:
    TITLE = "open-jet"

    def __init__(self, *, force_setup: bool = False) -> None:
        self.force_setup = force_setup
        self.cfg = load_config()
        self.cfg["airgapped"] = airgapped_from_cfg(self.cfg)
        set_airgapped(bool(self.cfg["airgapped"]))
        self.client: RuntimeClient | None = None
        self.agent: Agent | None = None
        self.session_logger: SessionLogger | None = None
        self.console = Console()
        self._style = Style.from_dict(
            {
                "prompt": "bg:#14213d #e5e7eb bold",
                "brand": "bg:#88D83F #111111 bold",
                "prompt-airgapped": "bg:#7c2d12 #ffedd5 bold",
                "brand-airgapped": "bg:#c2410c #fff7ed bold",
                "bottom-toolbar": "bg:#0f172a #cbd5e1",
                "completion-menu.completion": "bg:#111827 #cbd5e1",
                "completion-menu.completion.current": "bg:#88D83F #111111 bold",
            }
        )
        self._session: PromptSession[str] | None = None
        self._toolbar_task: asyncio.Task[None] | None = None
        self._generation_worker: asyncio.Task[None] | None = None
        self._setup_task: asyncio.Task[None] | None = None
        self._quit_requested = False
        self.screen = None
        self.focused: object | None = None
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
        self._thinking_token = 0
        self._assistant_status_kind: str | None = None
        self._assistant_status_command: str | None = None
        self._awaiting_approval = False
        self._approval_choice = 0
        self._approval_future: asyncio.Future[bool] | None = None
        self._approval_tool_call: ToolCall | None = None
        self._approval_started_at: float | None = None
        self.commands = SlashCommandHandler(self, banner=BANNER)
        self.completion = CompletionEngine(
            [
                SlashCompletionProvider(self.commands),
                FileMentionCompletionProvider(Path.cwd()),
            ]
        )
        self._prompt_history: list[str] = []
        self._utilization_visible = True
        self.metrics = SystemMetricsReader()
        self._power_min_watts: float | None = None
        self._power_max_watts: float | None = None
        self._generation_started_at: float | None = None
        self._generation_tokens_streamed = 0
        self._last_generation_tps: float | None = None
        self._pending_image_paths: list[str] = []
        self._widgets = {
            "#chat-log": LogView(self.console),
            "#assistant-status": StatusWidget(),
            "#approval-bar": StatusWidget(),
            "#command-suggestions": StatusWidget(),
            "#token-counter": StatusWidget(),
            "#utilization-bar": StatusWidget(),
            "#prompt": PromptWidget(),
        }
        self._widgets["#token-counter"].hidden = False
        self._widgets["#utilization-bar"].hidden = False
        self.focused = self._widgets["#prompt"]

    def query_one(self, selector: str, _expected_type: object | None = None) -> Any:
        return self._widgets[selector]

    def set_focus(self, target: object | None) -> None:
        self.focused = target

    def call_after_refresh(self, callback) -> None:
        callback()

    def exit(self) -> None:
        self._quit_requested = True

    def _request_terminal_exit(self, prompt_app: object | None = None) -> None:
        if self._quit_requested:
            return
        app = prompt_app
        if app is None and self._session is not None:
            app = getattr(self._session, "app", None)
        if app is not None and getattr(app, "is_running", False):
            app.exit(exception=KeyboardInterrupt())
            return
        asyncio.create_task(self.action_quit())

    @staticmethod
    def _install_quit_signal_handlers(callback) -> list[signal.Signals]:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return []

        installed: list[signal.Signals] = []
        for sig in (signal.SIGINT, signal.SIGTSTP):
            try:
                loop.add_signal_handler(sig, callback)
            except (NotImplementedError, RuntimeError, ValueError):
                continue
            installed.append(sig)
        return installed

    @staticmethod
    def _remove_signal_handlers(signals_to_remove: list[signal.Signals]) -> None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        for sig in signals_to_remove:
            try:
                loop.remove_signal_handler(sig)
            except (NotImplementedError, RuntimeError, ValueError):
                continue

    def _active_model_ref(self) -> str:
        return active_model_ref(self.cfg)

    def is_airgapped(self) -> bool:
        return bool(self.cfg.get("airgapped", False))

    def set_airgapped(self, enabled: bool, *, persist: bool = True) -> bool:
        normalized = bool(enabled)
        changed = normalized != self.is_airgapped()
        self.cfg["airgapped"] = normalized
        set_airgapped(normalized)
        if persist:
            save_config(self.cfg)
            self.persist_session_state(reason=f"airgapped:{str(normalized).lower()}")
        if self.session_logger:
            self.session_logger.broadcast = self._effective_broadcast_config()
        if self._session and self._session.app:
            self._session.app.invalidate()
        self._render_token_counter()
        return changed

    def _effective_broadcast_config(self) -> BroadcastConfig:
        telemetry_cfg = self.cfg.get("telemetry", {})
        broadcast_cfg = telemetry_cfg.get("broadcast", {})
        enabled = bool(broadcast_cfg.get("enabled", False)) and not self.is_airgapped()
        return BroadcastConfig(
            enabled=enabled,
            endpoint=str(broadcast_cfg.get("endpoint", "")).strip() or None,
            headers=broadcast_cfg.get("headers") if isinstance(broadcast_cfg.get("headers"), dict) else None,
            timeout_seconds=float(broadcast_cfg.get("timeout_seconds", 3.0)),
            export_logs=bool(broadcast_cfg.get("export_logs", True)),
            export_metrics=bool(broadcast_cfg.get("export_metrics", True)),
            export_traces=bool(broadcast_cfg.get("export_traces", True)),
        )

    def _trace_runtime_context(self) -> dict[str, object]:
        model_ref = self._active_model_ref()
        model_id, model_variant = _telemetry_model_fields(model_ref)
        return {
            "runtime": self.cfg.get("runtime", "llama_cpp"),
            "backend": _telemetry_backend(self.cfg),
            "model": model_ref,
            "model_id": model_id,
            "model_variant": model_variant,
            "device_profile": self.cfg.get("device", "auto"),
            "os_type": os.uname().sysname.lower() if hasattr(os, "uname") else os.name,
            "context_window_tokens": self.client.context_window_tokens if self.client else self.cfg.get("context_window_tokens", 2048),
            "gpu_layers": self.cfg.get("gpu_layers", 0),
            "host_arch": os.uname().machine if hasattr(os, "uname") else None,
            "use_case_tag": self.cfg.get("telemetry", {}).get("use_case_tag"),
            **_telemetry_hardware_fields(self.cfg),
        }

    def _log_trace_event(self, event_type: str, **data: object) -> None:
        if self.session_logger:
            self.session_logger.log_event(event_type, turn_id=self._active_turn_id, **data)

    def _agent_trace(self, event: str, data: dict[str, object]) -> None:
        if self.session_logger:
            self.session_logger.record_agent_trace(event, data, turn_id=self._active_turn_id)

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
            self.session_logger.start_turn(
                turn_id=self._active_turn_id,
                prompt=prompt,
                mode=self.harness_state.mode,
                resumed_session=self._session_was_resumed,
                active_step=active.title if (active := active_step(self.harness_state)) else None,
                files_in_play=self.harness_state.files_in_play,
                runtime_context=self._trace_runtime_context(),
            )

    def _tool_telemetry_attributes(self, tc: ToolCall) -> dict[str, object]:
        attrs: dict[str, object] = {}
        if isinstance(tc.arguments, dict):
            attrs["openjet.tool.arg_keys"] = sorted(str(key) for key in tc.arguments.keys())
        if tc.name == "shell" and isinstance(tc.arguments, dict):
            command = str(tc.arguments.get("command", "")).strip()
            classified = _classify_shell_command(command)
            attrs.update(
                {
                    "openjet.tool.shell.command_category": _shell_command_category(
                        str(classified.get("primary_command", ""))
                    ),
                    "openjet.tool.shell.classified_verification": bool(
                        classified.get("classified_verification")
                    ),
                    "openjet.tool.shell.hallucinated_command": bool(
                        classified.get("hallucinated_command")
                    ),
                    "openjet.tool.shell.false_positive_proposal": bool(
                        classified.get("false_positive_proposal")
                    ),
                }
            )
        elif tc.name in {"read_file", "write_file", "load_file", "edit_file"} and isinstance(tc.arguments, dict):
            path = Path(str(tc.arguments.get("path", "")).strip())
            attrs.update(
                {
                    "openjet.tool.target_extension": path.suffix.lower() or "<none>",
                    "openjet.tool.target_is_absolute": path.is_absolute(),
                }
            )
        elif tc.name in {"glob", "grep"} and isinstance(tc.arguments, dict):
            attrs["openjet.tool.pattern_length"] = len(str(tc.arguments.get("pattern", "")))
        elif tc.name == "memory" and isinstance(tc.arguments, dict):
            attrs["openjet.tool.memory.scope"] = str(tc.arguments.get("scope", "")).strip() or "<empty>"
            attrs["openjet.tool.memory.action"] = str(tc.arguments.get("action", "")).strip() or "<empty>"
        return attrs

    def _tool_result_telemetry_attributes(
        self,
        tc: ToolCall,
        meta: dict[str, object],
        *,
        duration_ms: float,
        output_truncated: bool,
    ) -> dict[str, object]:
        attrs: dict[str, object] = {
            "openjet.tool.duration_ms": duration_ms,
            "openjet.tool.output_truncated": output_truncated,
        }
        if tc.name == "shell":
            attrs["openjet.tool.shell.timed_out"] = bool(meta.get("timed_out"))
            exit_code = meta.get("exit_code")
            if isinstance(exit_code, int):
                attrs["openjet.tool.shell.exit_code"] = exit_code
            timeout_seconds = meta.get("timeout_seconds")
            if isinstance(timeout_seconds, int):
                attrs["openjet.tool.shell.timeout_seconds"] = timeout_seconds
        elif tc.name == "load_file":
            for key in ("truncated", "estimated_tokens", "returned_tokens", "token_budget", "mem_available_mb"):
                if key in meta:
                    attrs[f"openjet.tool.load_file.{key}"] = meta[key]
        elif tc.name == "edit_file":
            for key in ("internal_retry", "replacements", "match_strategy"):
                if key in meta:
                    attrs[f"openjet.tool.edit_file.{key}"] = meta[key]
            attrs["openjet.tool.edit_file.validation_error_present"] = bool(meta.get("validation_error"))
        return attrs

    def _finish_turn_trace(self, *, success: bool, status: str, error: str | None = None) -> None:
        if not self._active_turn_id:
            return
        if self.session_logger:
            self.session_logger.finish_turn(
                self._active_turn_id,
                success=success,
                status=status,
                error=error,
                generation_tokens=self._active_turn_generation_tokens,
                tool_attempts=self._active_turn_tool_attempts,
                tool_successes=self._active_turn_tool_successes,
                approval_requests=self._active_turn_approval_requests,
                approval_grants=self._active_turn_approval_grants,
                false_positive_command_proposals=self._active_turn_false_positive_commands,
                hallucinated_command_proposals=self._active_turn_hallucinated_commands,
                recovered_after_resumed_session=self._active_turn_recovered_after_resume,
                runtime_context=self._trace_runtime_context(),
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
        if self.client.context_window_tokens != configured_ctx or self.client.gpu_layers != configured_gpu_layers:
            self.cfg["context_window_tokens"] = self.client.context_window_tokens
            self.cfg["gpu_layers"] = self.client.gpu_layers
            save_config(self.cfg)
        self.agent = Agent(
            client=self.client,
            system_prompt=await build_system_prompt(str(self.cfg.get("system_prompt", "")), Path.cwd()),
            context_window_tokens=self.client.context_window_tokens,
            context_reserved_tokens=int(mem_cfg["context_reserved_tokens"]) if mem_cfg.get("context_reserved_tokens") is not None else None,
            min_prompt_tokens=int(mem_cfg.get("min_prompt_tokens", 256)),
            min_available_mb=int(mem_cfg["min_available_mb"]) if mem_cfg.get("min_available_mb") is not None else None,
            max_used_percent=float(mem_cfg["max_used_percent"]) if mem_cfg.get("max_used_percent") is not None else None,
            memory_check_interval_chunks=int(mem_cfg.get("check_interval_chunks", 16)),
            condense_target_tokens=int(mem_cfg.get("condense_target_tokens", 900)),
            keep_last_messages=int(mem_cfg.get("keep_last_messages", 6)),
            trace_hook=self._agent_trace,
        )

    async def _materialize_setup_model(self, setup_result: dict, log: LogView) -> dict:
        status = self.query_one("#assistant-status")

        def _set_status(text: str) -> None:
            status.remove_class("hidden")
            status.update(text)

        def _clear_status() -> None:
            status.update("")
            status.add_class("hidden")

        return await materialize_setup_model(setup_result, log, set_status=_set_status, clear_status=_clear_status)

    async def _run_setup_wizard(self) -> dict | None:
        return await run_setup_wizard(
            session=self._session,
            console=self.console,
            hardware_info=detect_hardware_info(),
            recommended_ctx=recommended_context_window_tokens(),
            current_cfg=self.cfg,
        )

    async def run_setup_command(self, log: LogView) -> bool:
        previous_cfg = dict(self.cfg)
        had_runtime = bool(self.client or self.agent)
        if self.agent:
            self.persist_session_state(reason="setup_command_start")
        if self.client:
            try:
                await self.client.close()
            except Exception as exc:
                log.write(f"[yellow]Runtime stop warning:[/] {exc}")
        self.client = None
        self.agent = None
        self.loaded_files.clear()
        self._render_token_counter()

        try:
            result = await self._run_setup_wizard()
        except (EOFError, KeyboardInterrupt):
            result = None

        if not isinstance(result, dict):
            if had_runtime:
                try:
                    await self._init_client()
                    log.write("[bold bright_white]Setup cancelled. Previous runtime restored.[/]")
                except Exception as exc:
                    if self.session_logger:
                        self.session_logger.record_exception("setup_restore_failed", exc, component="setup")
                    log.write(f"[bold red]Setup cancelled; runtime restore failed:[/] {_format_error(exc)}")
            else:
                log.write("[bold bright_white]Setup cancelled.[/]")
            log.write("")
            self._render_token_counter()
            return False

        try:
            resolved_result = await self._materialize_setup_model(result, log)
        except Exception as exc:
            if self.session_logger:
                self.session_logger.record_exception("setup_apply_failed", exc, component="setup")
            log.write(f"[bold red]Setup failed:[/] {_format_error(exc)}")
            log.write("")
            return False

        self.cfg.update(resolved_result)
        save_config(self.cfg)
        model_name = Path(self._active_model_ref()).name or self._active_model_ref() or "model"
        log.write(f"  [bold bright_white]Applying setup and loading {escape(model_name)}...[/]")
        status = self.query_one("#assistant-status")
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
            if self.session_logger:
                self.session_logger.record_exception("setup_reload_failed", exc, component="setup")
            log.write(f"[bold red]Setup failed:[/] {_format_error(exc)}")
            log.write("")
            return False
        status.update("")
        status.add_class("hidden")
        self.loaded_files.clear()
        self.persist_session_state(reason="setup_command")
        self._render_token_counter()
        log.write("[bold bright_white]Setup applied. Runtime restarted and context reset.[/]")
        log.write("")
        return True

    def run_setup_command_worker(self) -> None:
        log = self.query_one("#chat-log")
        if self._setup_task and not self._setup_task.done():
            return
        self._setup_task = asyncio.create_task(self.run_setup_command(log))

    async def _startup_sequence(self) -> None:
        log = self.query_one("#chat-log")
        log.write(BANNER)
        log_cfg = self.cfg.get("logging", {})
        if log_cfg.get("enabled", True):
            self.session_logger = SessionLogger(
                base_dir=Path(log_cfg.get("directory", "session_logs")),
                label=str(log_cfg.get("label", "open-jet")),
                metrics_interval_seconds=float(log_cfg.get("metrics_interval_seconds", 5)),
                install_id_path=Path(
                    self.cfg.get("telemetry", {}).get("install_id_path", ".openjet/state/telemetry_identity.json")
                ),
                retention_days=int(log_cfg.get("retention_days", 30)) if log_cfg.get("retention_days") is not None else None,
                max_sessions=int(log_cfg.get("max_sessions", 100)) if log_cfg.get("max_sessions") is not None else None,
                broadcast=self._effective_broadcast_config(),
            )
            await self.session_logger.start()
            self.session_logger.log_event("app_mount", cwd=str(Path.cwd()))
        if self.force_setup or not self._has_any_configured_model():
            try:
                setup_result = await self._run_setup_wizard()
            except (EOFError, KeyboardInterrupt):
                setup_result = None
            if isinstance(setup_result, dict):
                try:
                    setup_result = await self._materialize_setup_model(setup_result, log)
                except Exception as exc:
                    if self.session_logger:
                        self.session_logger.record_exception("setup_failed", exc, component="setup")
                    log.write(f"[bold red]Setup failed:[/] {_format_error(exc)}")
                    log.write("")
                    self._quit_requested = True
                    return
                self.cfg.update(setup_result)
                save_config(self.cfg)
            elif not self._has_any_configured_model():
                self._quit_requested = True
                return
        elif not self.cfg.get("setup_complete"):
            self.cfg["setup_complete"] = True
            self.cfg.setdefault("model_source", "local")
            self.cfg.setdefault("runtime", "llama_cpp")
            self.cfg.setdefault("device", recommended_device())
            self.cfg.setdefault("context_window_tokens", recommended_context_window_tokens())
            self.cfg.setdefault("gpu_layers", recommended_gpu_layers(str(self.cfg.get("device", "auto"))))
            save_config(self.cfg)

        if self.cfg.get("model_source") == "ollama" and self.cfg.get("ollama_model") and (not self.cfg.get("model") or not Path(str(self.cfg.get("model"))).is_file()):
            try:
                resolved = await self._materialize_setup_model(dict(self.cfg), log)
            except Exception as exc:
                if self.session_logger:
                    self.session_logger.record_exception("setup_resolve_model_failed", exc, component="setup")
                log.write(f"[bold red]Failed to resolve Ollama model:[/] {_format_error(exc)}")
                log.write("")
                self._quit_requested = True
                return
            self.cfg.update(resolved)
            save_config(self.cfg)

        active_model = self._active_model_ref()
        log.write(f"  [bold bright_white]Loading {Path(active_model).name or active_model}...[/]")
        try:
            await self._init_client()
        except Exception as exc:
            if self.session_logger:
                self.session_logger.record_exception("runtime_start_failed", exc, component="runtime")
            log.write(f"\n[bold red]Failed to start LLM:[/] {_format_error(exc)}")
            self._quit_requested = True
            return
        if self.session_logger:
            self.session_logger.record_runtime_ready(**self._trace_runtime_context())
        log.write("  [bold bright_white]Ready.[/]")
        if self.is_airgapped():
            log.write("  [bold #c2410c]Air-gapped mode is enabled. External network access is blocked.[/]")
        if self.auto_resume:
            self._restore_session_state(log)
        self._restore_harness_state()
        log.write("")
        self._render_token_counter()

    async def action_quit(self) -> None:
        if self._quit_requested:
            return
        self._quit_requested = True
        if self._active_turn_id:
            self._finish_turn_trace(success=False, status="abandoned", error="application quit")
        self.persist_session_state(reason="quit")
        if self._generation_worker and not self._generation_worker.done():
            self._generation_worker.cancel()
        if self.client:
            await self.client.close()
        if self.session_logger:
            await self.session_logger.stop()

    def _record_prompt_history(self, text: str) -> None:
        normalized = text.strip()
        if normalized:
            self._prompt_history.append(normalized)

    async def submit_text(self, text: str) -> None:
        prompt = self.query_one("#prompt")
        prompt.value = text
        if self._awaiting_approval or (self._generation_worker and not self._generation_worker.done()):
            self.query_one("#chat-log").write("[yellow]Wait for the current generation to finish, then retry.[/]")
            self.query_one("#chat-log").write("")
            prompt.value = ""
            return

        attached_images = list(self._pending_image_paths)
        stripped = text.strip()
        image_only = extract_pasted_image_paths(text) if not stripped else []
        for image_path in image_only:
            if image_path not in attached_images:
                attached_images.append(image_path)
        if not stripped and not attached_images:
            return
        history_text = stripped if stripped else "\n".join(f"Attached image: {path}" for path in attached_images)
        self._record_prompt_history(history_text)
        prompt.value = ""
        self._pending_image_paths.clear()

        if stripped and await self.commands.maybe_handle(stripped):
            self._render_token_counter()
            return

        log = self.query_one("#chat-log")
        if not self.agent:
            log.write("[yellow]LLM is not ready yet. Wait for Ready, or run /setup.[/]")
            log.write("")
            self._render_token_counter()
            return

        display_text = stripped if stripped else "[image attachment]"
        log.write(Rule("[bold green]User", style="green"))
        log.write(f"[bold green]> [/]{display_text}")
        for image_path in attached_images:
            log.write(f"  [bold bright_white]attached image:[/] {escape(image_path)}")
        log.write("")

        mentioned_files = _extract_file_mentions(stripped)
        attached_from_mentions = [
            str(Path(path).expanduser().resolve())
            for path in mentioned_files
            if is_image_path(path)
        ]
        for image_path in attached_from_mentions:
            if image_path not in attached_images:
                attached_images.append(image_path)
        self.harness_state = update_state_for_user_message(self.harness_state, history_text, files=mentioned_files)
        self._begin_turn_trace(history_text)
        self.persist_harness_state()
        if self.session_logger:
            self.session_logger.record_user_message(
                turn_id=self._active_turn_id,
                text=history_text,
                mentioned_files=mentioned_files,
                attached_images=attached_images,
                mode=self.harness_state.mode,
            )
        await self._load_mentioned_files_into_context(stripped, log)
        self.agent.add_user_message(stripped, image_paths=attached_images)
        self.persist_session_state(reason="user_message")
        self._render_token_counter()
        self._start_agent_turn()

    async def _load_mentioned_files_into_context(self, text: str, log: LogView) -> None:
        if not self.agent:
            return
        for mention_path in _extract_file_mentions(text):
            if is_image_path(mention_path):
                continue
            await self.load_context_file(mention_path, log)
        self._render_token_counter()

    async def load_context_file(self, path: str, log: LogView) -> bool:
        if not self.agent:
            return False
        mention_path = path.strip()
        if not mention_path:
            log.write("[yellow]load:[/] empty path")
            return False
        if is_image_path(mention_path):
            log.write(f"[yellow]load:[/] {escape(mention_path)} is an image. Attach it in a chat turn with @path.")
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
            self.session_logger.record_context_file_loaded(
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

    def _render_token_counter(self, draft_text: str = "") -> None:
        counter = self.query_one("#token-counter")
        if not self.agent:
            counter.update("tokens: 0/0")
            return
        current = self.agent.estimated_context_tokens()
        draft = estimate_message_content_tokens(build_user_content(draft_text, self._pending_image_paths))
        total = current + draft
        window = self.client.context_window_tokens if self.client else int(self.cfg.get("context_window_tokens", 2048))
        budget = self.agent.context_budget() or derive_context_budget(window)
        remaining = max(0, budget.prompt_tokens - total)
        counter.update(
            f"tokens: {total}/{window} | prompt<= {budget.prompt_tokens} | remaining: {remaining} | harness: {self._turn_context_tokens}"
        )

    def _format_percent(self, label: str, pct: float | None) -> str:
        if pct is None:
            return f"{label} n/a"
        clamped = max(0.0, min(100.0, pct))
        return f"{label} {clamped:4.1f}%"

    def _format_power_minmax(self) -> str:
        if self._power_min_watts is None or self._power_max_watts is None:
            return ""
        return f"[min {self._power_min_watts:.1f}W max {self._power_max_watts:.1f}W]"

    def _format_power_text(self, watts: float | None, pct: float | None, battery: dict[str, float | str | None] | None = None) -> str:
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
        base = f"pwr {pct:4.1f}% ({watts:.1f}W)" if pct is not None else f"pwr {watts:.1f}W"
        if minmax:
            base += f" {minmax}"
        return base

    def _format_tps_text(self) -> str:
        tps = self._current_tps()
        return "tps n/a" if tps is None else f"tps {tps:.1f}"

    def _current_tps(self) -> float | None:
        if self._thinking_timer is not None and self._generation_started_at is not None:
            elapsed = time.monotonic() - self._generation_started_at
            if elapsed > 0:
                return self._generation_tokens_streamed / elapsed
        return self._last_generation_tps

    def _update_power_minmax(self, watts: float | None) -> None:
        if watts is None:
            return
        if self._power_min_watts is None or watts < self._power_min_watts:
            self._power_min_watts = watts
        if self._power_max_watts is None or watts > self._power_max_watts:
            self._power_max_watts = watts

    def _toolbar_text(self) -> str:
        rows: list[str] = []
        rows.append("mode: AIR-GAPPED" if self.is_airgapped() else "mode: internet")
        if not self.query_one("#assistant-status").hidden and self.query_one("#assistant-status").text:
            rows.append(_plain_markup(self.query_one("#assistant-status").text))
        if not self.query_one("#approval-bar").hidden and self.query_one("#approval-bar").text:
            rows.append(_plain_markup(self.query_one("#approval-bar").text))
        if not self.query_one("#token-counter").hidden and self.query_one("#token-counter").text:
            rows.append(self.query_one("#token-counter").text)
        if self._utilization_visible:
            cpu_pct = self.metrics.read_cpu_percent()
            mem = read_memory_snapshot()
            battery = self.metrics.read_battery_metrics()
            power_watts, power_pct = self.metrics.read_power_metrics()
            self._update_power_minmax(power_watts)
            mem_text = self._format_percent("mem", mem.used_percent if mem else None)
            cpu_text = self._format_percent("cpu", cpu_pct)
            power_text = self._format_power_text(power_watts, power_pct, battery)
            rows.append(f"util: {cpu_text} | {mem_text} | {self._format_tps_text()} | {power_text}")
        return "\n".join(row for row in rows if row)

    def _prompt_message(self) -> HTML:
        if self.is_airgapped():
            return HTML("<brand-airgapped> open-jet air-gap </brand-airgapped><prompt-airgapped>  > </prompt-airgapped>")
        return HTML("<brand> open-jet </brand><prompt>  > </prompt>")

    def set_utilization_visible(self, visible: bool) -> None:
        self._utilization_visible = bool(visible)

    def toggle_utilization_visible(self) -> bool:
        self._utilization_visible = not self._utilization_visible
        return self._utilization_visible

    def is_utilization_visible(self) -> bool:
        return self._utilization_visible

    def runtime_status_snapshot(self) -> dict:
        if not self.agent:
            return {"ready": False, "airgapped": self.is_airgapped()}
        window = self.client.context_window_tokens if self.client else int(self.cfg.get("context_window_tokens", 2048))
        budget = self.agent.context_budget() or derive_context_budget(window)
        current = self.agent.estimated_context_tokens()
        remaining = max(0, budget.prompt_tokens - current)
        mem = read_memory_snapshot()
        active = active_step(self.harness_state)
        reasoning_mode = None
        if self.client and hasattr(self.client, "reasoning_status"):
            try:
                reasoning_mode = getattr(self.client, "reasoning_status")()
            except Exception:
                reasoning_mode = None
        return {
            "ready": True,
            "messages": self.agent.conversation_message_count(),
            "airgapped": self.is_airgapped(),
            "generating": self._thinking_timer is not None,
            "command_in_progress": self._assistant_status_kind == "command",
            "active_command": self._assistant_status_command,
            "reasoning_mode": reasoning_mode,
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

    def _restore_session_state(self, log: LogView) -> bool:
        if not self.agent:
            return False
        state = self.state_store.load()
        if not state:
            return False
        airgapped = state.get("airgapped")
        if isinstance(airgapped, bool):
            self.set_airgapped(airgapped, persist=False)
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
            if "content" in msg and not is_supported_message_content(msg.get("content")):
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
        self.loaded_files = loaded_files if isinstance(loaded_files, dict) else {}
        harness_payload = state.get("harness_state")
        if isinstance(harness_payload, dict):
            self.harness_state = HarnessState.from_dict(harness_payload)
        log.write(
            "  [bold bright_white]"
            f"Resumed previous session: {max(0, len(self.agent.messages) - 1)} messages, "
            f"{len(self.loaded_files)} loaded files.[/]"
        )
        self._session_was_resumed = True
        return True

    def _restore_harness_state(self) -> None:
        try:
            self.harness_state = self.harness_store.load()
        except Exception:
            self.harness_state = HarnessState()

    def _replay_restored_history(self, log: LogView, messages: list[dict]) -> None:
        for msg in messages:
            role = msg.get("role")
            if role == "user":
                text = content_to_plain_text(msg.get("content", ""))
                if text.strip():
                    log.write(f"[bold green]> [/]{text}")
                    log.write("")
            elif role == "assistant":
                text = content_to_plain_text(msg.get("content", ""))
                if text:
                    self._write_text_block(log, text)
                if not msg.get("tool_calls"):
                    log.write("")
            elif role == "tool":
                text = content_to_plain_text(msg.get("content", ""))
                if text:
                    self._write_tool_result(log, text)

    def _seed_prompt_history_from_messages(self, messages: list[dict]) -> None:
        self._prompt_history = []
        for msg in messages:
            if msg.get("role") == "user":
                normalized = content_to_plain_text(msg.get("content", "")).strip()
                if normalized:
                    self._prompt_history.append(normalized)

    def handle_prompt_paste(self, text: str) -> bool:
        image_paths = extract_pasted_image_paths(text)
        if not image_paths:
            return False
        added = 0
        for path in image_paths:
            if path not in self._pending_image_paths:
                self._pending_image_paths.append(path)
                added += 1
        if added > 0:
            self._render_token_counter()
            status = self.query_one("#assistant-status")
            noun = "image" if added == 1 else "images"
            status.remove_class("hidden")
            status.update(f"Attached {added} pasted {noun}. Enter a prompt to analyze them.")
        return True

    def _write_text_block(self, log: LogView, text: str) -> None:
        buf = text
        while "\n" in buf:
            line, buf = buf.split("\n", 1)
            log.write(line)
        if buf:
            log.write(buf)

    def _write_tool_result(self, log: LogView, result: str) -> None:
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
            "airgapped": self.is_airgapped(),
            "model": self.cfg.get("model"),
            "device": self.cfg.get("device", "auto"),
            "context_window_tokens": self.client.context_window_tokens if self.client else self.cfg.get("context_window_tokens", 2048),
            "messages": self.agent.messages,
            "loaded_files": self.loaded_files,
            "harness_state": self.harness_state.to_dict(),
        }
        try:
            self.state_store.save(payload)
        except Exception:
            return

    def persist_harness_state(self) -> None:
        self.harness_state.updated_at = time.time()
        try:
            self.harness_store.save(self.harness_state)
        except Exception:
            return

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

    def _start_agent_turn(self, recovery_attempted: bool = False) -> None:
        if self._generation_worker and not self._generation_worker.done():
            self._log_trace_event("turn_replaced", replaced_by_new_turn=True)
        self._prepare_turn_context()
        self._generation_worker = asyncio.create_task(self.run_agent_turn(recovery_attempted=recovery_attempted))

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
            layered_config=self.cfg.get("layered_context", {}),
        )
        self.agent.set_turn_context(context.messages)
        self._turn_context_docs = context.docs_loaded
        self._turn_context_tokens = context.docs_tokens
        if self.harness_state.mode == "debug":
            runtime_messages = self.agent._messages_for_runtime()
            if runtime_messages:
                write_debug_runtime_messages(root=Path.cwd(), turn_id=self._active_turn_id or "pending-turn", messages=runtime_messages)
                write_debug_context_snapshot(
                    root=Path.cwd(),
                    turn_id=self._active_turn_id or "pending-turn",
                    snapshot={
                        "docs_loaded": context.docs_loaded,
                        "docs_tokens": context.docs_tokens,
                        "state_summary": context.state_summary,
                        "state_summary_tokens": context.state_summary_tokens,
                        "layer_tokens": context.layer_tokens,
                        "layer_docs": context.layer_docs,
                        "budget_alerts": context.budget_alerts,
                        "candidate_decisions": context.candidate_decisions,
                        "budget": {
                            "effective_window": context.budget.effective_window,
                            "usable_prompt_budget": context.budget.usable_prompt_budget,
                            "remaining_budget": context.budget.remaining_budget,
                            "docs_budget": context.budget.docs_budget,
                            "layer1_budget": context.budget.layer1_budget,
                            "layer2_budget": context.budget.layer2_budget,
                            "layer3_budget": context.budget.layer3_budget,
                            "layer_alert_tokens": context.budget.layer_alert_tokens,
                        },
                    },
                )

    def action_stop_generation(self) -> None:
        if self._generation_worker and not self._generation_worker.done():
            self._generation_worker.cancel()
            log = self.query_one("#chat-log")
            log.write("[yellow]Generation stopped.[/]")
            log.write("")
            self._finish_turn_trace(success=False, status="interrupted", error="generation stopped by user")
            self._render_token_counter()

    async def run_agent_turn(self, recovery_attempted: bool = False) -> None:
        log = self.query_one("#chat-log")
        pending_tool_calls: list[ToolCall] = []
        tool_events: list[dict] = []
        condense_requested = False
        text_buf = ""
        assistant_turn_text = ""
        assistant_header_written = False
        thinking_token = self._start_thinking()
        self._log_trace_event("run_agent_turn_started", recovery_attempted=recovery_attempted)
        try:
            assert self.agent is not None
            async for event in self.agent.run_turn():
                if event.kind == ActionKind.TEXT:
                    if not assistant_header_written:
                        log.write(Rule("[bold cyan]Assistant", style="cyan"))
                        assistant_header_written = True
                    text_buf += event.text
                    assistant_turn_text += event.text
                    tokens = estimate_tokens(event.text)
                    self._generation_tokens_streamed += tokens
                    self._active_turn_generation_tokens += tokens
                    while "\n" in text_buf:
                        line, text_buf = text_buf.split("\n", 1)
                        log.write(line)
                elif event.kind == ActionKind.TOOL_REQUEST:
                    pending_tool_calls.append(event.tool_call)
                elif event.kind == ActionKind.CONDENSE:
                    condense_requested = True
                elif event.kind == ActionKind.ERROR:
                    if not recovery_attempted and self._is_recoverable_runtime_error(event.text):
                        recovered = await self._recover_runtime(log, event.text)
                        if recovered:
                            self._start_agent_turn(recovery_attempted=True)
                            return
                    log.write(f"\n[bold red]error:[/] {event.text}")
                    self._finish_turn_trace(success=False, status="agent_error", error=event.text)
                    return
                elif event.kind == ActionKind.DONE:
                    if text_buf:
                        log.write(text_buf)
                        text_buf = ""
                    log.write("")
        except asyncio.CancelledError:
            return
        finally:
            self._stop_thinking(thinking_token)
            if self._generation_worker and self._generation_worker.done():
                self._generation_worker = None

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
                if self.session_logger:
                    self.session_logger.record_exception(
                        "tool_call_failed",
                        exc,
                        component="tool",
                        turn_id=self._active_turn_id,
                        tool_key=tc.id,
                        extra_attributes={"openjet.tool.name": tc.name},
                    )
                log.write(f"[bold red]tool error ({tc.name}):[/] {exc}")
                log.write("")
                if self.agent:
                    self.agent.complete_tool_call(tc, f"Tool execution failed: {exc}")

        self.harness_state = update_state_after_turn(self.harness_state, tool_events=tool_events, assistant_text=assistant_turn_text)
        self.persist_harness_state()
        if pending_tool_calls:
            self.persist_session_state(reason="assistant_turn_with_tools")
            self._start_agent_turn()
        else:
            self.persist_session_state(reason="assistant_turn_done")
            self._finish_turn_trace(success=True, status="completed")
            self._render_token_counter()

    async def _handle_tool_call(self, tc: ToolCall, log: LogView) -> dict | None:
        tool_key = tc.id or f"{tc.name}-{int(time.time() * 1000)}"
        tc.id = tool_key
        tool_attrs = self._tool_telemetry_attributes(tc)
        if tc.name == "shell":
            if bool(tool_attrs.get("openjet.tool.shell.false_positive_proposal")):
                self._active_turn_false_positive_commands += 1
            if bool(tool_attrs.get("openjet.tool.shell.hallucinated_command")):
                self._active_turn_hallucinated_commands += 1
        if tc.name not in allowed_tools_for_mode(self.harness_state.mode):
            denied = f"Tool {tc.name} is not available for this request. Use an approved tool or ask for a different approach."
            log.write(f"[yellow]{denied}[/]")
            log.write("")
            if self.session_logger and self._active_turn_id:
                self.session_logger.start_tool_call(
                    turn_id=self._active_turn_id,
                    tool_key=tool_key,
                    tool_name=tc.name,
                    attributes=tool_attrs,
                    needs_confirmation=False,
                )
                self.session_logger.finish_tool_call(
                    tool_key,
                    ok=False,
                    approved=False,
                    duration_ms=None,
                    status="disallowed",
                    attributes={"openjet.tool.denial_reason": "mode_restricted"},
                )
            if self.agent:
                self.agent.complete_tool_call(tc, denied)
            return {"tool": tc.name, "ok": False, "summary": denied, "target": format_tool_args(tc)}
        needs_confirm = self.agent.needs_confirmation(tc) if self.agent else False
        if self.session_logger and self._active_turn_id:
            self.session_logger.start_tool_call(
                turn_id=self._active_turn_id,
                tool_key=tool_key,
                tool_name=tc.name,
                attributes=tool_attrs,
                needs_confirmation=needs_confirm,
            )
        if needs_confirm:
            self._active_turn_approval_requests += 1
            log.write(Rule(f"[bold yellow]Tool Request: {tc.name}", style="yellow"))
            log.write(f"[yellow]{tc.name}:[/]")
            for preview_line in self._tool_preview_lines(tc):
                log.write(f"  [bold bright_white]{preview_line}[/]")
            approved = await self._wait_for_tool_approval(tc)
            decision_ms = None
            if self._approval_started_at is not None:
                decision_ms = round((time.monotonic() - self._approval_started_at) * 1000.0, 2)
            self._approval_started_at = None
            if self.session_logger:
                self.session_logger.record_tool_approval(
                    tool_key=tool_key,
                    approved=approved,
                    decision_ms=decision_ms,
                )
            if not approved:
                log.write("[red]  denied[/]")
                log.write("")
                assert self.agent is not None
                self.agent.complete_tool_call(tc, "User denied this action.")
                self.persist_session_state(reason=f"tool_denied:{tc.name}")
                if self.session_logger:
                    self.session_logger.finish_tool_call(
                        tool_key,
                        ok=False,
                        approved=False,
                        duration_ms=decision_ms,
                        status="denied",
                    )
                return {"tool": tc.name, "ok": False, "summary": "User denied this action.", "target": format_tool_args(tc)}
            log.write("[green]  approved[/]")
            self._active_turn_approval_grants += 1

        if tc.name == "load_file":
            self._clamp_load_file_tool_budget(tc)
        self._active_turn_tool_attempts += 1
        t0 = time.monotonic()
        tool_status_token = self._start_tool_status(tc)
        try:
            execution = await execute_tool(tc)
        except Exception as exc:
            duration_ms = round((time.monotonic() - t0) * 1000.0, 2)
            if self.session_logger:
                self.session_logger.record_exception(
                    "tool_execute_exception",
                    exc,
                    component="tool",
                    turn_id=self._active_turn_id,
                    tool_key=tool_key,
                    extra_attributes={"openjet.tool.name": tc.name},
                )
                self.session_logger.finish_tool_call(
                    tool_key,
                    ok=False,
                    approved=True,
                    duration_ms=duration_ms,
                    status="exception",
                )
            raise
        finally:
            self._stop_tool_status(tool_status_token)
        result = execution.output
        meta = execution.meta
        result_for_context, output_truncated = self._fit_tool_result_to_budget(result)
        if execution.ok:
            self._active_turn_tool_successes += 1
        for line in result_for_context.splitlines()[:20]:
            log.write(f"  [bold bright_white]{line}[/]")
        if len(result_for_context.splitlines()) > 20:
            log.write(f"  [bold bright_white]... ({len(result_for_context.splitlines()) - 20} more lines)[/]")
        log.write("")
        assert self.agent is not None
        self.agent.complete_tool_call(tc, result_for_context)
        self.persist_session_state(reason=f"tool_result:{tc.name}")
        duration_ms = round((time.monotonic() - t0) * 1000.0, 2)
        if self.session_logger:
            self.session_logger.finish_tool_call(
                tool_key,
                ok=execution.ok,
                approved=True,
                duration_ms=duration_ms,
                status="completed" if execution.ok else "failed",
                attributes=self._tool_result_telemetry_attributes(
                    tc,
                    meta,
                    duration_ms=duration_ms,
                    output_truncated=output_truncated,
                ),
            )
        return {
            "tool": tc.name,
            "ok": bool(meta.get("ok", True)),
            "summary": result.splitlines()[0] if result else "",
            "target": format_tool_args(tc),
            "verification": tc.name == "shell" and shell_command_is_verification(str(tc.arguments.get("command", ""))),
            "command": tc.arguments.get("command") if isinstance(tc.arguments, dict) else None,
            "duration_ms": duration_ms,
        }

    def _clamp_load_file_tool_budget(self, tc: ToolCall) -> None:
        if isinstance(tc.arguments, dict):
            remaining = self._remaining_prompt_tokens()
            current = tc.arguments.get("max_tokens")
            tc.arguments["max_tokens"] = max(128, min(current, remaining)) if isinstance(current, int) else remaining

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
        needles = ("connecterror", "connection refused", "connection reset", "remoteprotocolerror", "readtimeout", "timed out", "server disconnected", "llama-server exited", "trtllm-serve exited", "502", "503", "504")
        return any(needle in lowered for needle in needles)

    async def _recover_runtime(self, log: LogView, error_text: str) -> bool:
        if not self.client:
            return False
        log.write("[yellow]LLM runtime interrupted. Restarting runtime once and retrying...[/]")
        self._log_trace_event("runtime_recovery_attempt", recoverable_error=error_text)
        try:
            await self.client.reset_kv_cache()
        except Exception as exc:
            if self.session_logger:
                self.session_logger.record_exception(
                    "runtime_recovery_failed",
                    exc,
                    component="runtime",
                    turn_id=self._active_turn_id,
                )
            log.write(f"[bold red]Runtime recovery failed:[/] {exc}")
            log.write("")
            return False
        if self.session_logger:
            self.session_logger.log_event("runtime_recovery_succeeded")
        log.write("[bold bright_white]Runtime recovered. Retrying turn.[/]")
        log.write("")
        return True

    def _start_thinking(self) -> int:
        self._thinking_token += 1
        self._assistant_status_kind = "generating"
        self._assistant_status_command = None
        self._generation_started_at = time.monotonic()
        self._generation_tokens_streamed = 0
        self._thinking_timer = True
        self._render_assistant_status()
        return self._thinking_token

    def _stop_thinking(self, token: int | None = None) -> None:
        if token is not None and token != self._thinking_token:
            return
        if self._generation_started_at is not None:
            elapsed = time.monotonic() - self._generation_started_at
            if elapsed > 0 and self._generation_tokens_streamed > 0:
                self._last_generation_tps = self._generation_tokens_streamed / elapsed
        self._generation_started_at = None
        self._thinking_timer = None
        self._assistant_status_kind = None
        self._assistant_status_command = None
        self._clear_assistant_status()

    def _start_tool_status(self, tc: ToolCall) -> int | None:
        if tc.name != "shell" or not isinstance(tc.arguments, dict):
            return None
        command = str(tc.arguments.get("command", "")).strip()
        if not command:
            return None
        self._thinking_token += 1
        self._assistant_status_kind = "command"
        self._assistant_status_command = command
        self._thinking_timer = True
        self._render_assistant_status()
        return self._thinking_token

    def _stop_tool_status(self, token: int | None = None) -> None:
        if token is None or token != self._thinking_token:
            return
        self._thinking_timer = None
        self._assistant_status_kind = None
        self._assistant_status_command = None
        self._clear_assistant_status()

    def _render_assistant_status(self) -> None:
        status = self.query_one("#assistant-status")
        if self._assistant_status_kind == "command" and self._assistant_status_command:
            status.remove_class("hidden")
            status.update(f"Running {self._format_command_status_label(self._assistant_status_command)}")
            return
        if self._assistant_status_kind == "generating":
            status.remove_class("hidden")
            status.update("Generating")
            return
        self._clear_assistant_status()

    def _clear_assistant_status(self) -> None:
        status = self.query_one("#assistant-status")
        status.add_class("hidden")
        status.update("")

    @staticmethod
    def _format_command_status_label(command: str, max_len: int = 72) -> str:
        compact = " ".join(command.split())
        if len(compact) <= max_len:
            return compact
        return compact[: max_len - 3] + "..."

    async def _wait_for_tool_approval(self, tc: ToolCall) -> bool:
        bar = self.query_one("#approval-bar")
        self._awaiting_approval = True
        self._approval_choice = 0
        self._approval_tool_call = tc
        self._approval_started_at = time.monotonic()
        self._approval_future = asyncio.get_running_loop().create_future()
        bar.remove_class("hidden")
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

    def _render_approval_bar(self) -> None:
        if not self._awaiting_approval or not self._approval_tool_call:
            return
        bar = self.query_one("#approval-bar")
        summary = self._approval_summary_text(self._approval_tool_call)
        approve = "[Approve]" if self._approval_choice == 0 else "Approve"
        deny = "[Deny]" if self._approval_choice == 1 else "Deny"
        bar.update(f"Tool request: {summary} | Left/Right choose | Enter confirm | y/n quick reply | {approve} {deny}")

    def _approval_summary_text(self, tc: ToolCall) -> str:
        if tc.name == "write_file":
            path = str(tc.arguments.get("path", "")).strip()
            content = str(tc.arguments.get("content", ""))
            return f"write_file -> {path} ({len(content)} bytes)"
        if tc.name == "edit_file":
            return f"edit_file -> {str(tc.arguments.get('path', '')).strip()}"
        if tc.name == "memory":
            return f"memory -> {str(tc.arguments.get('action', '')).strip()} {str(tc.arguments.get('scope', '')).strip()}".strip()
        if tc.name == "shell":
            command = str(tc.arguments.get("command", "")).strip()
            if len(command) > 120:
                command = command[:117] + "..."
            return f"shell -> {command}"
        return f"{tc.name} -> {format_tool_args(tc)}"

    def _resolve_approval(self, approved: bool) -> None:
        if self._approval_future and not self._approval_future.done():
            self._approval_future.set_result(approved)

    def _tool_preview_lines(self, tc: ToolCall) -> list[str]:
        if tc.name == "shell":
            command = str(tc.arguments.get("command", "")).strip()
            timeout_seconds = tc.arguments.get("timeout_seconds")
            lines = [f"command: {command[:200] + ('...' if len(command) > 200 else '')}"]
            if isinstance(timeout_seconds, int):
                lines.append(f"timeout_seconds: {timeout_seconds}")
            return lines
        if tc.name == "write_file":
            path = str(tc.arguments.get("path", "")).strip()
            return [f"path: {path}", f"bytes: {len(str(tc.arguments.get('content', '')))}"]
        if tc.name == "edit_file":
            return [f"path: {str(tc.arguments.get('path', '')).strip()}"]
        if tc.name == "memory":
            return [f"scope: {str(tc.arguments.get('scope', '')).strip()}", f"action: {str(tc.arguments.get('action', '')).strip()}"]
        return [str(format_tool_args(tc))]

    def _bindings(self) -> KeyBindings:
        bindings = KeyBindings()
        awaiting_approval = Condition(lambda: self._awaiting_approval)
        generating = Condition(lambda: bool(self._generation_worker and not self._generation_worker.done()))

        @bindings.add("c-c")
        def _ctrl_c(event) -> None:
            self._request_terminal_exit(event.app)

        @bindings.add("c-z")
        def _ctrl_z(event) -> None:
            self._request_terminal_exit(event.app)

        @bindings.add("escape", filter=awaiting_approval)
        def _escape_approval(event) -> None:
            self._approval_choice = 1
            self._resolve_approval(False)
            event.current_buffer.reset()

        @bindings.add("escape", filter=generating)
        def _escape_generation(event) -> None:
            self.action_stop_generation()
            event.current_buffer.reset()

        @bindings.add("left", filter=awaiting_approval)
        def _left(event) -> None:
            self._approval_choice = 0
            self._render_approval_bar()
            event.current_buffer.reset()

        @bindings.add("right", filter=awaiting_approval)
        def _right(event) -> None:
            self._approval_choice = 1
            self._render_approval_bar()
            event.current_buffer.reset()

        @bindings.add("y", filter=awaiting_approval)
        def _yes(event) -> None:
            self._approval_choice = 0
            self._resolve_approval(True)
            event.current_buffer.reset()

        @bindings.add("n", filter=awaiting_approval)
        def _no(event) -> None:
            self._approval_choice = 1
            self._resolve_approval(False)
            event.current_buffer.reset()

        @bindings.add("enter", filter=awaiting_approval)
        def _enter_approval(event) -> None:
            self._resolve_approval(self._approval_choice == 0)
            event.current_buffer.reset()

        @bindings.add("enter", filter=generating)
        def _enter_generation(event) -> None:
            event.current_buffer.reset()

        @bindings.add("enter")
        def _enter_default(event) -> None:
            event.current_buffer.validate_and_handle()

        return bindings

    async def _toolbar_updater(self) -> None:
        while not self._quit_requested:
            await asyncio.sleep(1.0)
            if self._session and self._session.app:
                self._session.app.invalidate()

    async def run_async(self) -> None:
        installed_signals = self._install_quit_signal_handlers(self._request_terminal_exit)
        self._session = PromptSession(
            history=InMemoryHistory(),
            completer=OpenJetCompleter(self),
            complete_while_typing=True,
            complete_in_thread=True,
            key_bindings=self._bindings(),
            style=self._style,
            bottom_toolbar=self._toolbar_text,
        )
        self._toolbar_task = asyncio.create_task(self._toolbar_updater())
        try:
            try:
                await self._startup_sequence()
            except (EOFError, KeyboardInterrupt):
                await self.action_quit()
                return
            if self._quit_requested:
                return
            with patch_stdout(raw=True):
                while not self._quit_requested:
                    try:
                        text = await self._session.prompt_async(
                            self._prompt_message,
                            enable_suspend=False,
                        )
                    except (EOFError, KeyboardInterrupt):
                        await self.action_quit()
                        break
                    await self.submit_text(text)
        finally:
            self._remove_signal_handlers(installed_signals)
            if self._toolbar_task:
                self._toolbar_task.cancel()


def _extract_file_mentions(text: str) -> list[str]:
    cleaned: list[str] = []
    seen: set[str] = set()
    for match in re.finditer(r"@\[([^\]]+)\]|(?<!\S)@([^\s]+)", text):
        bracketed = match.group(1)
        bare = match.group(2)
        candidate = (bracketed if bracketed is not None else bare or "").strip()
        if bracketed is None:
            candidate = candidate.rstrip(".,;:!?)]}")
        if candidate and candidate not in seen:
            seen.add(candidate)
            cleaned.append(candidate)
    return cleaned


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="open-jet offline agentic terminal UI")
    parser.add_argument("--setup", action="store_true", help="start in setup wizard mode before launching the chat UI")
    args = parser.parse_args(argv)
    asyncio.run(OpenJetApp(force_setup=args.setup).run_async())


if __name__ == "__main__":
    main()
