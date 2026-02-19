"""open-jet TUI: single-pane chat with block title."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import yaml
from rich.markup import escape
from textual import events, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import ModalScreen, Screen
from textual.worker import Worker, get_current_worker
from textual.widgets import Input, RichLog, Static
from textual.widgets._input import Selection

from .agent import ActionKind, Agent, ToolCall
from .commands import SlashCommandHandler
from .completion import CompletionEngine, FileMentionCompletionProvider, SlashCompletionProvider
from .executor import load_file, read_file, run_shell, write_file
from .ollama_client import OllamaClient
from .runtime_limits import derive_context_budget, estimate_tokens, read_memory_snapshot
from .session_state import SessionStateStore
from .session_logging import SessionLogger


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"


@dataclass(frozen=True)
class HardwareInfo:
    label: str
    total_ram_gb: float
    has_cuda: bool


# Curated, size-banded shortlist used by setup recommendations.
RECOMMENDED_LLM_BANDS: tuple[tuple[float, tuple[tuple[str, float, str], ...]], ...] = (
    (
        2.0,
        (
            ("qwen2.5:1.5b", 1.5, "Qwen2.5 1.5B"),
            ("deepseek-r1:1.5b", 1.5, "DeepSeek R1 1.5B"),
            ("gemma2:2b", 2.0, "Gemma 2 2B"),
        ),
    ),
    (
        4.0,
        (
            ("qwen2.5:3b", 3.0, "Qwen2.5 3B"),
            ("qwen2.5:3b-instruct", 3.0, "Qwen2.5 3B Instruct"),
            ("gemma2:2b", 2.0, "Gemma 2 2B"),
        ),
    ),
    (
        8.0,
        (
            ("qwen2.5:7b", 7.0, "Qwen2.5 7B"),
            ("mistral:7b", 7.0, "Mistral 7B"),
            ("deepseek-r1:7b", 7.0, "DeepSeek R1 7B"),
        ),
    ),
    (
        14.0,
        (
            ("qwen2.5:14b", 14.0, "Qwen2.5 14B"),
            ("deepseek-r1:14b", 14.0, "DeepSeek R1 14B"),
            ("gemma2:9b", 9.0, "Gemma 2 9B"),
        ),
    ),
    (
        32.0,
        (
            ("qwen2.5:32b", 32.0, "Qwen2.5 32B"),
            ("qwen2.5-coder:32b", 32.0, "Qwen2.5 Coder 32B"),
            ("gemma2:27b", 27.0, "Gemma 2 27B"),
        ),
    ),
)

JETSON_OVERRIDE_OPTIONS: tuple[tuple[str, str, float], ...] = (
    ("jetson_nano_4", "Jetson Nano (4GB RAM)", 4.0),
    ("jetson_xavier_nx_8", "Jetson Xavier NX (8GB RAM)", 8.0),
    ("jetson_orin_nano_8", "Jetson Orin Nano (8GB RAM)", 8.0),
    ("jetson_orin_nx_16", "Jetson Orin NX (16GB RAM)", 16.0),
    ("jetson_agx_orin_32", "Jetson AGX Orin (32GB RAM)", 32.0),
    ("jetson_agx_orin_64", "Jetson AGX Orin (64GB RAM)", 64.0),
)

TOOL_RESULT_SAFE_PADDING_TOKENS = 256
TOOL_RESULT_MIN_BUDGET_TOKENS = 64


def _find_ollama_cli() -> str | None:
    found = shutil.which("ollama")
    if found:
        return found
    # Common install locations when PATH isn't inherited by TUI launchers.
    for candidate in (
        "/usr/local/bin/ollama",
        "/usr/bin/ollama",
        "/opt/homebrew/bin/ollama",
        "/snap/bin/ollama",
    ):
        path = Path(candidate)
        if path.is_file() and os.access(path, os.X_OK):
            return str(path)
    return None


def load_config() -> dict:
    for candidate in [Path("config.yaml"), CONFIG_PATH]:
        if candidate.exists():
            return yaml.safe_load(candidate.read_text()) or {}
    return {}


def save_config(cfg: dict) -> None:
    CONFIG_PATH.write_text(yaml.dump(cfg, default_flow_style=False))


def _discover_model_files() -> list[str]:
    roots = [
        Path.cwd(),
        Path.cwd() / "models",
        Path.home() / "Downloads",
        Path.home() / "models",
    ]
    found: set[str] = set()
    for root in roots:
        if not root.exists() or not root.is_dir():
            continue
        for path in root.glob("*.gguf"):
            found.add(str(path.resolve()))
    return sorted(found)


def _recommended_device() -> str:
    if Path("/usr/local/cuda").exists() or Path("/dev/nvhost-gpu").exists():
        return "cuda"
    return "cpu"


def _read_device_model() -> str | None:
    try:
        raw = Path("/proc/device-tree/model").read_bytes()
    except OSError:
        return None
    text = raw.decode("utf-8", errors="ignore").replace("\x00", "").strip()
    return text or None


def _detect_hardware_info() -> HardwareInfo:
    mem = read_memory_snapshot()
    total_ram_gb = (mem.total_mb / 1024.0) if mem else 0.0
    has_cuda = bool(Path("/usr/local/cuda").exists() or Path("/dev/nvhost-gpu").exists())
    board = _read_device_model()
    if board:
        label = board
    elif has_cuda:
        label = "CUDA-capable device"
    else:
        label = "CPU-only device"
    return HardwareInfo(label=label, total_ram_gb=total_ram_gb, has_cuda=has_cuda)


def _effective_hardware_info(profile: str, detected: HardwareInfo, override_key: str | None = None) -> HardwareInfo:
    if profile != "other":
        return detected
    for key, label, ram_gb in JETSON_OVERRIDE_OPTIONS:
        if key == override_key:
            clean_label = label.split(" (", 1)[0]
            return HardwareInfo(label=clean_label, total_ram_gb=ram_gb, has_cuda=True)
    return detected


def _recommended_device_for_hardware(profile: str, detected: HardwareInfo, override_key: str | None = None) -> str:
    hw = _effective_hardware_info(profile, detected, override_key)
    return "cuda" if hw.has_cuda else "cpu"


def _recommended_param_budget_b(profile: str, detected: HardwareInfo, override_key: str | None = None) -> float:
    hw = _effective_hardware_info(profile, detected, override_key)
    total_gb = hw.total_ram_gb
    if total_gb < 6:
        cap = 2.0
    elif total_gb < 12:
        cap = 4.0
    elif total_gb < 24:
        cap = 8.0
    elif total_gb < 48:
        cap = 14.0
    else:
        cap = 32.0
    if not hw.has_cuda:
        cap = min(cap, 8.0)
    return cap


def _recommended_llm_models(max_params_b: float) -> list[tuple[str, str]]:
    for band_limit, models in RECOMMENDED_LLM_BANDS:
        if max_params_b <= band_limit:
            return [(f"{title} ({params:g}B params)", tag) for tag, params, title in models]
    models = RECOMMENDED_LLM_BANDS[-1][1]
    return [(f"{title} ({params:g}B params)", tag) for tag, params, title in models]


def _recommended_context_window_tokens() -> int:
    headless = not bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    mem = read_memory_snapshot()
    if not mem:
        return 4096 if headless else 2048
    total_gb = mem.total_mb / 1024.0
    return _recommended_context_window_tokens_from_total(
        total_gb,
        headless=headless,
        available_mb=mem.available_mb,
    )


def _recommended_context_window_tokens_from_total(
    total_gb: float,
    *,
    headless: bool,
    available_mb: float | None = None,
) -> int:
    if total_gb >= 48:
        rec = 12288
    elif total_gb >= 24:
        rec = 8192
    elif total_gb >= 12:
        rec = 6144
    elif total_gb >= 7:
        rec = 4096 if headless else 3072
    elif total_gb >= 4:
        rec = 2048
    else:
        rec = 1024

    if available_mb is not None and available_mb < 1200:
        rec = min(rec, 2048)
    return rec


def _recommended_gpu_layers(device: str, total_ram_gb: float | None = None) -> int:
    if device == "cpu":
        return 0
    if total_ram_gb is None or total_ram_gb < 16:
        return 20
    if total_ram_gb < 32:
        return 28
    return 35


_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")
_PULL_PERCENT_RE = re.compile(r"(?P<pct>\d{1,3})%")
_PULL_SIZE_RE = re.compile(
    r"(?P<done>\d+(?:\.\d+)?)\s*(?P<done_unit>[KMGTP]B)\s*/\s*(?P<total>\d+(?:\.\d+)?)\s*(?P<total_unit>[KMGTP]B)",
    flags=re.IGNORECASE,
)
_PULL_SPEED_RE = re.compile(r"(?P<speed>\d+(?:\.\d+)?)\s*(?P<speed_unit>[KMGTP]B/s)", flags=re.IGNORECASE)
_PULL_ETA_RE = re.compile(r"(?P<eta>(?:\d+h)?(?:\d+m)?(?:\d+s)|\d+ms)$", flags=re.IGNORECASE)


def _extract_pull_progress(line: str) -> dict[str, str] | None:
    pct_match = _PULL_PERCENT_RE.search(line)
    if not pct_match:
        return None
    pct = max(0, min(100, int(pct_match.group("pct"))))

    size_match = _PULL_SIZE_RE.search(line)
    speed_match = _PULL_SPEED_RE.search(line)
    eta_match = _PULL_ETA_RE.search(line.strip())

    payload: dict[str, str] = {"pct": str(pct)}
    if size_match:
        payload["done"] = f"{size_match.group('done')} {size_match.group('done_unit').upper()}"
        payload["total"] = f"{size_match.group('total')} {size_match.group('total_unit').upper()}"
    if speed_match:
        payload["speed"] = f"{speed_match.group('speed')} {speed_match.group('speed_unit').upper()}"
    if eta_match:
        payload["eta"] = eta_match.group("eta")
    return payload


def _render_progress_bar(percent: int, width: int = 24) -> str:
    clamped = max(0, min(100, percent))
    filled = int(round((clamped / 100) * width))
    return "#" * filled + "-" * (width - filled)


def _is_jetson_label(label: str | None) -> bool:
    if not label:
        return False
    return "jetson" in label.lower()


def _running_on_jetson() -> bool:
    if _is_jetson_label(_read_device_model()):
        return True
    return Path("/etc/nv_tegra_release").exists()


def _context_window_options(recommended: int) -> list[int]:
    options = [1024, 1536, 2048, 3072, 4096, 6144, 8192]
    if recommended not in options:
        options.append(recommended)
    return sorted(set(options))


def _gpu_layer_options(device: str, recommended: int) -> list[int]:
    base = [0] if device == "cpu" else [0, 10, 20, 28, 35]
    if recommended not in base:
        base.append(recommended)
    return sorted(set(base))


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

ACCENT_GREEN = "#88D83F"


class SetupScreen(ModalScreen[dict]):
    SETUP_ACCENT_OPEN = f"[bold {ACCENT_GREEN}]"
    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit"),
        Binding("q", "quit", "Quit"),
        Binding("up", "prev_option", "Prev"),
        Binding("down", "next_option", "Next"),
        Binding("enter", "advance", "Next"),
        Binding("tab", "advance", "Next"),
        Binding("shift+tab", "back", "Back"),
        Binding("escape", "cancel", "Cancel"),
    ]

    def __init__(
        self,
        *,
        model_options: list[str],
        hardware_info: HardwareInfo,
        recommended_ctx: int,
        exit_on_cancel: bool = True,
    ) -> None:
        super().__init__()
        self.model_options = model_options
        self.hardware_info = hardware_info
        self.recommended_ctx = max(512, int(recommended_ctx))
        self.exit_on_cancel = exit_on_cancel
        self.ollama_cli = _find_ollama_cli()
        self._steps: list[dict] = []
        self._step_index = 0
        self._indices: dict[str, int] = {}
        self._selections: dict[str, object] = {}

    def compose(self) -> ComposeResult:
        yield Vertical(
            Static(f"{self.SETUP_ACCENT_OPEN}First-run setup[/]"),
            Static("", id="setup-description"),
            Static("", id="setup-step"),
            Static("", id="setup-options"),
            Static("Manual model path (.gguf):", id="setup-model-path-label", classes="hidden"),
            Input(placeholder="/path/to/model.gguf", id="setup-model-path", classes="hidden"),
            Static(""),
            Static("", id="setup-error"),
            Static(
                "[dim]Up/Down select • Tab/Enter next • Shift+Tab back • Enter on final step saves and restarts[/]",
                id="setup-hint",
            ),
            id="setup-box",
        )

    def on_mount(self) -> None:
        self._init_steps()
        self._render_step()
        self.set_focus(None)

    def on_key(self, event: events.Key) -> None:
        if event.key == "up":
            self.action_prev_option()
            event.stop()
            return
        if event.key == "down":
            self.action_next_option()
            event.stop()
            return
        if event.key in ("enter", "tab"):
            self.action_advance()
            event.stop()
            return
        if event.key == "shift+tab":
            self.action_back()
            event.stop()
            return
        if event.key == "escape":
            self.action_cancel()
            event.stop()
            return

    def _init_steps(self) -> None:
        ram_text = f"{self.hardware_info.total_ram_gb:.1f} GB RAM" if self.hardware_info.total_ram_gb > 0 else "RAM unknown"
        hardware_rows: list[tuple[str, str]] = [
            (f"Use detected hardware ({self.hardware_info.label}, {ram_text})", "auto"),
            ("Pick hardware profile manually", "other"),
        ]
        hardware_override_rows: list[tuple[str, str]] = [
            (label, key) for key, label, _ram in JETSON_OVERRIDE_OPTIONS
        ]
        local_model_rows: list[tuple[str, str]] = [(Path(model).name, model) for model in self.model_options]
        local_model_rows.append(("Manual path", "__manual__"))
        context_rows = [
            (f"{value} (recommended)" if value == self.recommended_ctx else str(value), value)
            for value in _context_window_options(self.recommended_ctx)
        ]

        self._steps = [
            {"key": "hardware", "title": "Hardware Detection", "options": hardware_rows},
            {"key": "hardware_override", "title": "Hardware Override", "options": hardware_override_rows},
            {"key": "model_plan", "title": "Model Source", "options": []},
            {"key": "local_model", "title": "Local Model File", "options": local_model_rows},
            {"key": "context_window_tokens", "title": "Context Size", "options": context_rows},
            {"key": "gpu_layers", "title": "GPU Offload", "options": []},
        ]

        defaults = {
            "hardware": "auto",
            "hardware_override": hardware_override_rows[0][1] if hardware_override_rows else "",
            "local_model": local_model_rows[0][1] if local_model_rows else "__manual__",
            "context_window_tokens": self.recommended_ctx,
        }
        for step in self._steps:
            key = str(step["key"])
            options = list(step["options"])
            if not options:
                continue
            idx = 0
            for i, (_label, value) in enumerate(options):
                if value == defaults.get(key):
                    idx = i
                    break
            self._indices[key] = idx
            self._selections[key] = options[idx][1]

        manual_input = self.query_one("#setup-model-path", Input)
        if self.model_options:
            manual_input.value = self.model_options[0]
        self._sync_dynamic_steps()

    def _step_by_key(self, key: str) -> dict | None:
        return next((step for step in self._steps if step["key"] == key), None)

    def _selected_hardware_profile(self) -> str:
        return str(self._selections.get("hardware", "auto"))

    def _selected_hardware_override(self) -> str:
        return str(self._selections.get("hardware_override", ""))

    def _effective_hardware(self) -> HardwareInfo:
        return _effective_hardware_info(
            self._selected_hardware_profile(),
            self.hardware_info,
            self._selected_hardware_override(),
        )

    def _jetson_target_selected(self) -> bool:
        profile = self._selected_hardware_profile()
        override = self._selected_hardware_override()
        if profile == "other" and override.startswith("jetson_"):
            return True
        return _is_jetson_label(self._effective_hardware().label)

    def _recommended_context_for_current_hardware(self) -> int:
        if self._selected_hardware_profile() == "auto":
            return _recommended_context_window_tokens()
        hw = self._effective_hardware()
        headless = not bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
        return _recommended_context_window_tokens_from_total(hw.total_ram_gb, headless=headless)

    def _sync_dynamic_steps(self) -> None:
        self._sync_model_plan_step()
        self._sync_context_step()
        self._sync_gpu_step()

    def _sync_model_plan_step(self) -> None:
        model_step = self._step_by_key("model_plan")
        if not model_step:
            return
        old_value = self._selections.get("model_plan")
        max_b = _recommended_param_budget_b(
            self._selected_hardware_profile(),
            self.hardware_info,
            self._selected_hardware_override(),
        )
        download_rows = [
            (f"Download with Ollama: {label}", tag)
            for label, tag in _recommended_llm_models(max_b)[:3]
        ]
        rows = [("Use a local .gguf model file", "__local__"), *download_rows]
        model_step["options"] = rows
        values = [value for _label, value in rows]
        if old_value in values:
            new_value = old_value
        elif rows:
            new_value = rows[0][1]
        else:
            new_value = "__local__"
        self._indices["model_plan"] = values.index(new_value)
        self._selections["model_plan"] = new_value

    def _sync_context_step(self) -> None:
        context_step = self._step_by_key("context_window_tokens")
        if not context_step:
            return
        old_value = self._selections.get("context_window_tokens")
        recommended = self._recommended_context_for_current_hardware()
        values = _context_window_options(recommended)
        context_step["options"] = [
            (f"{value} (recommended)" if value == recommended else str(value), value)
            for value in values
        ]
        new_value = old_value if old_value in values else recommended
        self._indices["context_window_tokens"] = values.index(new_value)
        self._selections["context_window_tokens"] = new_value

    def _current_step(self) -> dict:
        return self._steps[self._step_index]

    def _is_step_enabled(self, key: str) -> bool:
        if key == "hardware_override":
            return str(self._selections.get("hardware", "auto")) == "other"
        if key == "local_model":
            return str(self._selections.get("model_plan", "")) == "__local__"
        return True

    def _visible_step_indices(self) -> list[int]:
        return [i for i, step in enumerate(self._steps) if self._is_step_enabled(str(step["key"]))]

    def _next_step_index(self, *, forward: bool) -> int | None:
        if forward:
            for idx in range(self._step_index + 1, len(self._steps)):
                if self._is_step_enabled(str(self._steps[idx]["key"])):
                    return idx
            return None
        for idx in range(self._step_index - 1, -1, -1):
            if self._is_step_enabled(str(self._steps[idx]["key"])):
                return idx
        return None

    def _render_step(self) -> None:
        current_key = str(self._steps[self._step_index]["key"])
        if not self._is_step_enabled(current_key):
            next_idx = self._next_step_index(forward=True)
            if next_idx is not None:
                self._step_index = next_idx
            else:
                prev_idx = self._next_step_index(forward=False)
                if prev_idx is not None:
                    self._step_index = prev_idx
        step = self._current_step()
        key = str(step["key"])
        options = list(step["options"])
        idx = int(self._indices[key])
        idx = max(0, min(idx, len(options) - 1))
        self._indices[key] = idx
        self._selections[key] = options[idx][1]

        header = self.query_one("#setup-step", Static)
        description = self.query_one("#setup-description", Static)
        visible = self._visible_step_indices()
        visible_pos = visible.index(self._step_index) + 1 if self._step_index in visible else self._step_index + 1
        header.update(
            f"{self.SETUP_ACCENT_OPEN}Step {visible_pos}/{len(visible)}[/]"
            f" {self.SETUP_ACCENT_OPEN}{step['title']}[/]"
        )
        description.update(f"[dim]{self._step_description(key)}[/]")

        lines: list[str] = []
        for i, (label, value) in enumerate(options):
            pretty_label = self._compact_text(str(label), 34)
            if i == idx:
                line = f"{self.SETUP_ACCENT_OPEN}[underline]> {pretty_label}[/underline][/]"
                detail = self._option_detail(key, value)
                if detail:
                    line += f" [dim]- {self._compact_text(detail, 26)}[/]"
            else:
                line = f"{self.SETUP_ACCENT_OPEN}  {pretty_label}[/]"
            lines.append(line)
        self.query_one("#setup-options", Static).update("\n".join(lines))

        manual_label = self.query_one("#setup-model-path-label", Static)
        manual_input = self.query_one("#setup-model-path", Input)
        show_manual = key == "local_model" and self._selections.get("local_model") == "__manual__"
        if show_manual:
            manual_label.remove_class("hidden")
            manual_input.remove_class("hidden")
            manual_input.focus()
        else:
            manual_label.add_class("hidden")
            manual_input.add_class("hidden")
            if self.focused is manual_input:
                self.set_focus(None)
        self.query_one("#setup-error", Static).update("")

    def action_next_option(self) -> None:
        step = self._current_step()
        key = str(step["key"])
        options = list(step["options"])
        self._indices[key] = (self._indices[key] + 1) % len(options)
        self._selections[key] = options[self._indices[key]][1]
        if key in {"hardware", "hardware_override"}:
            self._sync_dynamic_steps()
        self._render_step()

    def action_prev_option(self) -> None:
        step = self._current_step()
        key = str(step["key"])
        options = list(step["options"])
        self._indices[key] = (self._indices[key] - 1) % len(options)
        self._selections[key] = options[self._indices[key]][1]
        if key in {"hardware", "hardware_override"}:
            self._sync_dynamic_steps()
        self._render_step()

    def action_advance(self) -> None:
        next_idx = self._next_step_index(forward=True)
        if next_idx is not None:
            self._step_index = next_idx
            self._render_step()
            return
        self.action_save()

    def action_back(self) -> None:
        prev_idx = self._next_step_index(forward=False)
        if prev_idx is None:
            return
        self._step_index = prev_idx
        self._render_step()

    def _sync_gpu_step(self) -> None:
        hw = self._effective_hardware()
        device = "cuda" if hw.has_cuda else "cpu"
        recommended = _recommended_gpu_layers(device, hw.total_ram_gb)
        gpu_step = next((step for step in self._steps if step["key"] == "gpu_layers"), None)
        if not gpu_step:
            return
        old_value = self._selections.get("gpu_layers")
        values = _gpu_layer_options(device, recommended)
        gpu_step["options"] = [
            (f"{value} (recommended)" if value == recommended else str(value), value)
            for value in values
        ]
        if old_value in values:
            new_value = old_value
        else:
            new_value = recommended
        self._indices["gpu_layers"] = values.index(new_value)
        self._selections["gpu_layers"] = new_value

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        self.action_advance()

    def action_save(self) -> None:
        self.ollama_cli = _find_ollama_cli()
        model_plan = str(self._selections.get("model_plan", ""))
        model_source = "local" if model_plan == "__local__" else "ollama"
        profile = self._selected_hardware_profile()
        override_key = str(self._selections.get("hardware_override", ""))
        device = _recommended_device_for_hardware(profile, self.hardware_info, override_key)
        hw = self._effective_hardware()
        payload: dict[str, object] = {
            "model_source": model_source,
            "hardware_profile": profile,
            "hardware_override": override_key if profile == "other" else "",
        }
        if model_source == "ollama":
            if not self.ollama_cli:
                self._set_error("Ollama CLI not found. Install Ollama (https://ollama.com/download), then retry.")
                return
            selected_llm = self._selected_model_plan()
            payload["recommended_llm"] = selected_llm
            ollama_model = selected_llm
            if not ollama_model:
                self._set_error("Ollama model tag is required.")
                return
            payload["ollama_model"] = ollama_model
        else:
            model_path = self._selected_model_path()
            if not model_path:
                self._set_error("Model path is required.")
                return
            model_file = Path(model_path).expanduser()
            if not model_file.is_file():
                self._set_error("Model file does not exist.")
                return
            if model_file.suffix.lower() != ".gguf":
                self._set_error("Model file must end with .gguf.")
                return
            payload["model"] = str(model_file)

        context_value = int(self._selections.get("context_window_tokens", self.recommended_ctx))
        if context_value < 512:
            self._set_error("Select a valid context window.")
            return

        gpu_value = int(self._selections.get("gpu_layers", _recommended_gpu_layers(device, hw.total_ram_gb)))
        if gpu_value < 0:
            self._set_error("Select a valid GPU layer value.")
            return

        payload.update(
            {
                "device": device,
                "context_window_tokens": context_value,
                "gpu_layers": gpu_value,
                "setup_complete": True,
            }
        )
        self.dismiss(payload)

    def _selected_model_path(self) -> str:
        selected = self._selections.get("local_model")
        if isinstance(selected, str) and selected != "__manual__":
            return selected
        return self.query_one("#setup-model-path", Input).value.strip()

    def _selected_model_plan(self) -> str:
        selected = self._selections.get("model_plan")
        if isinstance(selected, str) and selected != "__local__":
            return selected
        return ""

    def _option_detail(self, key: str, value: object) -> str:
        if key == "hardware":
            if value == "auto":
                return "best default"
            return "manual override"
        if key == "hardware_override":
            return "sets memory profile"
        if key == "model_plan":
            if value == "__local__":
                return "use existing file"
            if not self.ollama_cli:
                return "install ollama first"
            if self._jetson_target_selected():
                return "quantized GGUF required for Jetson"
            cap = _recommended_param_budget_b(
                self._selected_hardware_profile(),
                self.hardware_info,
                self._selected_hardware_override(),
            )
            return f"fit for this hardware (~{cap:g}B)"
        if key == "local_model":
            if value == "__manual__":
                return "enter full path"
            return "detected file"
        if key == "context_window_tokens":
            if value == self._recommended_context_for_current_hardware():
                return "best default"
            return "higher uses more memory"
        if key == "gpu_layers":
            return "higher may be faster"
        return ""

    def _step_description(self, key: str) -> str:
        if key == "hardware":
            return "Pick auto-detected hardware or switch to manual."
        if key == "hardware_override":
            return "Select the closest Jetson and RAM profile."
        if key == "model_plan":
            if not self.ollama_cli:
                return "Choose local file, or install Ollama to enable downloads."
            if self._jetson_target_selected():
                return "Choose local file or Ollama download (quantized GGUF only on Jetson)."
            return "Choose local file or Ollama download."
        if key == "local_model":
            return "Select a detected .gguf or enter a path."
        if key == "context_window_tokens":
            return "Set context size."
        if key == "gpu_layers":
            return "Set GPU offload depth."
        return ""

    def _compact_text(self, text: str, limit: int) -> str:
        src = " ".join(text.split())
        if len(src) <= limit:
            return src
        if limit <= 1:
            return src[:limit]
        return src[: limit - 1] + "…"

    def _set_error(self, message: str) -> None:
        self.query_one("#setup-error", Static).update(f"[bold red]{message}[/]")

    def action_cancel(self) -> None:
        if self.exit_on_cancel:
            self.app.exit()
            return
        self.dismiss({})

    async def action_quit(self) -> None:
        await self.app.action_quit()


# ---------------------------------------------------------------------------
# Styles
# ---------------------------------------------------------------------------

CSS = """
App {
    background: #2a2a2a;
}
SetupScreen {
    background: #2a2a2a;
    tint: transparent;
}
Screen {
    background: #2a2a2a;
}
Vertical {
    background: transparent;
}
RichLog {
    background: transparent;
}
Input {
    background: transparent;
}
Static {
    background: transparent;
}
#chat-log {
    width: 100%;
    height: auto;
    padding: 0 2;
    color: #ffffff;
}
#prompt {
    height: 3;
    margin: 0 2;
    color: #ffffff;
}
#command-suggestions {
    height: auto;
    margin: 0 2;
    color: #ffffff;
}
#token-counter {
    height: 1;
    margin: 0 2;
    color: #ffffff;
}
#utilization-bar {
    height: 1;
    margin: 0 2;
    color: #ffffff;
}
#assistant-status {
    height: 1;
    margin: 0 2;
    color: #ffffff;
}
#approval-bar {
    height: auto;
    min-height: 3;
    margin: 0 2;
    padding: 0 1;
    border: round $warning;
    background: transparent;
}
.hidden {
    display: none;
}
#setup-box {
    width: 100%;
    height: auto;
    margin: 0 2;
    padding: 1 2;
    background: transparent;
}
#setup-step {
    margin: 0 0;
}
#setup-description {
    margin: 0 0 1 0;
}
#setup-options {
    margin: 1 0 0 0;
}
#setup-model-path-label {
    margin: 0 0;
}
#setup-model-path {
    margin: 0 0;
}
#setup-error {
    margin: 0 0;
}
#setup-hint {
    margin: 1 0 0 0;
}
"""


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

class OpenJetApp(App):
    TITLE = "open-jet"
    CSS = CSS
    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit"),
        Binding("escape", "stop_generation", "Stop"),
    ]

    def __init__(self, *, force_setup: bool = False) -> None:
        super().__init__()
        self.force_setup = force_setup
        self.cfg = load_config()
        self.client: OllamaClient | None = None
        self.agent: Agent | None = None
        self.session_logger: SessionLogger | None = None
        state_cfg = self.cfg.get("state", {})
        self.state_store = SessionStateStore(
            path=Path(state_cfg.get("path", "session_state.json")),
            enabled=bool(state_cfg.get("enabled", True)),
        )
        self.auto_resume = bool(state_cfg.get("auto_resume", False))
        self.loaded_files: dict[str, dict] = {}
        self._thinking_timer = None
        self._thinking_idx = 0
        self._thinking_token = 0
        self._awaiting_approval = False
        self._approval_choice = 0
        self._approval_future: asyncio.Future[bool] | None = None
        self._approval_tool_call: ToolCall | None = None
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
        self._prev_cpu_sample: tuple[int, int] | None = None
        self._prev_powercap_sample: dict[str, tuple[float, float, float | None]] = {}
        self._power_min_watts: float | None = None
        self._power_max_watts: float | None = None
        self._generation_started_at: float | None = None
        self._generation_tokens_streamed = 0
        self._last_generation_tps: float | None = None

    async def _init_client(self) -> None:
        mem_cfg = self.cfg.get("memory_guard", {})
        configured_ctx = int(self.cfg.get("context_window_tokens", 2048))
        configured_gpu_layers = int(self.cfg.get("gpu_layers", 20))
        self.client = OllamaClient(
            model=self.cfg["model"],
            context_window_tokens=configured_ctx,
            device=str(self.cfg.get("device", "auto")),
            gpu_layers=configured_gpu_layers,
        )
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
            model_options=_discover_model_files(),
            hardware_info=_detect_hardware_info(),
            recommended_ctx=_recommended_context_window_tokens(),
            exit_on_cancel=exit_on_cancel,
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

    async def _run_command_capture(self, *args: str) -> tuple[int, str, str]:
        proc = await asyncio.create_subprocess_exec(
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        out_raw, err_raw = await proc.communicate()
        return (proc.returncode or 0, out_raw.decode("utf-8", errors="ignore"), err_raw.decode("utf-8", errors="ignore"))

    async def _run_command_stream(
        self,
        *args: str,
        on_chunk: Callable[[str, bool], None] | None = None,
    ) -> tuple[int, str, str]:
        proc = await asyncio.create_subprocess_exec(
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        out_chunks: list[str] = []
        err_chunks: list[str] = []

        async def _pump(stream: asyncio.StreamReader | None, sink: list[str], is_stderr: bool) -> None:
            if stream is None:
                return
            while True:
                raw = await stream.read(4096)
                if not raw:
                    break
                text = raw.decode("utf-8", errors="ignore")
                sink.append(text)
                if on_chunk:
                    on_chunk(text, is_stderr)

        await asyncio.gather(
            _pump(proc.stdout, out_chunks, False),
            _pump(proc.stderr, err_chunks, True),
        )
        rc = await proc.wait()
        return rc, "".join(out_chunks), "".join(err_chunks)

    async def _resolve_ollama_model_file(self, ollama_model: str) -> str:
        ollama_cli = _find_ollama_cli()
        if not ollama_cli:
            raise RuntimeError("`ollama` CLI is not installed or not discoverable.")
        rc, out, err = await self._run_command_capture(ollama_cli, "show", ollama_model, "--modelfile")
        if rc != 0:
            detail = (err or out).strip()[:500]
            raise RuntimeError(f"Unable to inspect pulled model '{ollama_model}': {detail or 'unknown error'}")

        from_ref = ""
        for line in out.splitlines():
            stripped = line.strip()
            if stripped.upper().startswith("FROM "):
                from_ref = stripped.split(maxsplit=1)[1].strip()
                break
        if not from_ref:
            raise RuntimeError("Could not resolve Ollama model file path from modelfile output.")

        candidates: list[Path] = []
        ref_path = Path(from_ref).expanduser()
        if ref_path.is_absolute():
            candidates.append(ref_path)
        if from_ref.startswith("sha256:"):
            candidates.append(Path.home() / ".ollama" / "models" / "blobs" / from_ref.replace("sha256:", "sha256-"))
        if from_ref.startswith("sha256-"):
            candidates.append(Path.home() / ".ollama" / "models" / "blobs" / from_ref)

        for candidate in candidates:
            if candidate.is_file():
                return str(candidate.resolve())
        raise RuntimeError(
            "Pulled model was resolved by Ollama, but no local GGUF/blob path could be found for llama-server."
        )

    async def _read_ollama_model_details(self, ollama_cli: str, ollama_model: str) -> dict[str, str]:
        rc, out, err = await self._run_command_capture(ollama_cli, "show", ollama_model, "--json")
        if rc != 0:
            detail = (err or out).strip()[:400]
            raise RuntimeError(f"Unable to inspect pulled model metadata for '{ollama_model}': {detail or 'unknown error'}")
        try:
            payload = json.loads(out)
        except json.JSONDecodeError:
            return {}
        details = payload.get("details")
        if not isinstance(details, dict):
            return {}
        extracted: dict[str, str] = {}
        for key in ("format", "family", "parameter_size", "quantization_level"):
            val = details.get(key)
            if isinstance(val, str):
                extracted[key] = val.strip()
        return extracted

    def _file_is_gguf(self, path: str) -> bool:
        try:
            with Path(path).open("rb") as fh:
                return fh.read(4) == b"GGUF"
        except OSError:
            return False

    def _jetson_constraints_required(self, setup_result: dict) -> bool:
        profile = str(setup_result.get("hardware_profile", "")).strip()
        override = str(setup_result.get("hardware_override", "")).strip()
        if profile == "other" and override.startswith("jetson_"):
            return True
        return _running_on_jetson()

    async def _validate_jetson_ollama_model(
        self,
        *,
        setup_result: dict,
        ollama_cli: str,
        ollama_model: str,
        resolved_model: str,
    ) -> None:
        if not self._jetson_constraints_required(setup_result):
            return

        if not self._file_is_gguf(resolved_model):
            raise RuntimeError(
                "Jetson setup requires a quantized GGUF model, but pulled artifact is not GGUF."
            )

        details = await self._read_ollama_model_details(ollama_cli, ollama_model)
        model_format = details.get("format", "").lower()
        quant = details.get("quantization_level", "").upper()

        if model_format and model_format != "gguf":
            raise RuntimeError(
                f"Jetson setup requires GGUF format, but Ollama reports format '{details.get('format', 'unknown')}'."
            )
        if not quant or quant.startswith(("F16", "F32", "BF16")):
            raise RuntimeError(
                "Jetson setup requires a quantized Ollama model (e.g. Q4/Q5 GGUF), not a base/unquantized variant."
            )

    async def _materialize_setup_model(self, setup_result: dict, log: RichLog) -> dict:
        if str(setup_result.get("model_source", "local")) != "ollama":
            return setup_result

        ollama_model = str(setup_result.get("ollama_model", "")).strip()
        if not ollama_model:
            raise RuntimeError("Ollama model tag is missing.")
        ollama_cli = _find_ollama_cli()
        if not ollama_cli:
            raise RuntimeError("`ollama` CLI is not installed or not discoverable. Install it and retry setup.")

        log.write(
            f"[bold bright_white]Pulling {escape(ollama_model)} from Ollama...[/] "
            "[dim](this can take several minutes for larger models)[/]"
        )
        status = self.query_one("#assistant-status", Static)
        status.remove_class("hidden")

        def _set_pull_status(text: str) -> None:
            status.update(f"[bold {ACCENT_GREEN}]{escape(text)}[/]")

        progress_buffer = ""
        last_percent = -1
        last_rendered = ""
        last_emit_t = 0.0
        last_status = ""

        def _emit_progress(text: str, _is_stderr: bool) -> None:
            nonlocal progress_buffer, last_percent, last_rendered, last_emit_t, last_status
            progress_buffer += _ANSI_ESCAPE_RE.sub("", text)
            chunks = re.split(r"[\r\n]+", progress_buffer)
            progress_buffer = chunks.pop() if chunks else ""

            for raw_line in chunks:
                line = raw_line.strip()
                if not line:
                    continue

                parsed = _extract_pull_progress(line)
                now = time.monotonic()
                if parsed:
                    pct = int(parsed["pct"])
                    if pct < last_percent and pct < 5:
                        # Some pulls report per-layer progress and reset at the next layer.
                        last_percent = -1
                    if pct <= last_percent and (now - last_emit_t) < 1.0:
                        continue
                    last_percent = pct
                    bar = _render_progress_bar(pct)
                    detail = f"{parsed.get('done', '?')}/{parsed.get('total', '?')}"
                    speed = parsed.get("speed", "?")
                    eta = parsed.get("eta", "?")
                    rendered = f"pull {pct:3d}% |{bar}| {detail} {speed} ETA {eta}"
                    if rendered != last_rendered or (now - last_emit_t) >= 1.0:
                        _set_pull_status(rendered)
                        last_rendered = rendered
                        last_emit_t = now
                    continue

                line_low = line.lower()
                if (
                    line_low.startswith("pulling")
                    or line_low.startswith("verifying")
                    or line_low.startswith("processing")
                    or line_low.startswith("writing")
                    or line_low.startswith("success")
                ) and line != last_status:
                    _set_pull_status(line)
                    last_status = line
                    last_emit_t = now

        try:
            rc, out, err = await self._run_command_stream(ollama_cli, "pull", ollama_model, on_chunk=_emit_progress)
            if rc != 0:
                detail = (err or out).strip()[:700]
                raise RuntimeError(f"Ollama pull failed for '{ollama_model}': {detail or 'unknown error'}")
            resolved_model = await self._resolve_ollama_model_file(ollama_model)
            await self._validate_jetson_ollama_model(
                setup_result=setup_result,
                ollama_cli=ollama_cli,
                ollama_model=ollama_model,
                resolved_model=resolved_model,
            )
            log.write(f"[bold bright_white]Pulled {escape(ollama_model)} and resolved local model file.[/]")
        finally:
            status.update("")
            status.add_class("hidden")

        merged = dict(setup_result)
        merged["model"] = resolved_model
        return merged

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
                    log.write(f"[bold red]Setup cancelled; runtime restore failed:[/] {exc}")
                    if self.session_logger:
                        self.session_logger.log_event("setup_restore_failed", error=str(exc))
            else:
                log.write("[bold bright_white]Setup cancelled.[/]")
            log.write("")
            return False

        try:
            resolved_result = await self._materialize_setup_model(result, log)
        except Exception as exc:
            log.write(f"[bold red]Setup failed:[/] {exc}")
            log.write("")
            if self.session_logger:
                self.session_logger.log_event("setup_apply_failed", error=str(exc))
            return False

        self.cfg.update(resolved_result)
        save_config(self.cfg)

        try:
            await self._init_client()
        except Exception as exc:
            self.cfg = previous_cfg
            save_config(self.cfg)
            try:
                await self._init_client()
            except Exception:
                pass
            log.write(f"[bold red]Setup failed:[/] {exc}")
            log.write("")
            if self.session_logger:
                self.session_logger.log_event("setup_apply_failed", error=str(exc))
            return False

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
        yield RichLog(id="chat-log", wrap=True, markup=True)
        yield Static("", id="assistant-status", classes="hidden")
        yield Static("", id="approval-bar", classes="hidden")
        yield Input(placeholder="> ", id="prompt")
        yield Static("", id="command-suggestions", classes="hidden")
        yield Static("", id="token-counter")
        yield Static("", id="utilization-bar")

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

        if self.force_setup or not self.cfg.get("model"):
            # First run without model: cancel exits app.
            # Explicit --setup mode: cancel returns to startup using existing config.
            setup_result = await self._wait_for_screen_result(
                self._build_setup_screen(exit_on_cancel=not bool(self.cfg.get("model")))
            )
            if isinstance(setup_result, dict) and setup_result.get("setup_complete"):
                try:
                    setup_result = await self._materialize_setup_model(setup_result, log)
                except Exception as exc:
                    log.write(f"[bold red]Setup failed:[/] {exc}")
                    log.write("")
                    return
                self.cfg.update(setup_result)
                save_config(self.cfg)
            elif not self.cfg.get("model"):
                return
        elif not self.cfg.get("setup_complete"):
            # Backfill defaults for older configs created before setup wizard existed.
            self.cfg["setup_complete"] = True
            self.cfg.setdefault("model_source", "local")
            self.cfg.setdefault("device", _recommended_device())
            self.cfg.setdefault("context_window_tokens", _recommended_context_window_tokens())
            self.cfg.setdefault("gpu_layers", _recommended_gpu_layers(str(self.cfg.get("device", "auto"))))
            save_config(self.cfg)

        if (
            self.cfg.get("model_source") == "ollama"
            and self.cfg.get("ollama_model")
            and (not self.cfg.get("model") or not Path(str(self.cfg.get("model"))).is_file())
        ):
            try:
                resolved = await self._materialize_setup_model(dict(self.cfg), log)
            except Exception as exc:
                log.write(f"[bold red]Failed to resolve Ollama model:[/] {exc}")
                log.write("")
                return
            self.cfg.update(resolved)
            save_config(self.cfg)

        log.write(f"  [bold bright_white]Loading {Path(self.cfg['model']).name}...[/]")
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
            self.session_logger.log_event(
                "llm_runtime_config",
                device=self.cfg.get("device", "auto"),
                gpu_layers=self.cfg.get("gpu_layers", 20),
                context_window_tokens=self.cfg.get("context_window_tokens", 2048),
            )
        log.write(f"  [bold bright_white]Ready.[/]")
        if self.auto_resume:
            self._restore_session_state(log)
        log.write("")
        self._render_token_counter()
        prompt.focus()

    async def action_quit(self) -> None:
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
        if self.session_logger:
            self.session_logger.log_event("user_message", text=text)

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
        bar.update("\n".join(lines) + "\n[bold bright_white]Up/Down to select, Tab to autocomplete[/]")

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
            color = "dim"
        counter.update(
            f"[{color}]tokens: {total}/{window} | prompt<= {budget.prompt_tokens} | remaining: {remaining}[/]"
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
        cpu_pct = self._read_cpu_percent()
        mem = read_memory_snapshot()
        battery = self._read_battery_metrics()
        power_watts, power_pct = self._read_power_metrics()
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
                base += f" {self._format_hours(float(remaining_hours))} left"
            elif status_raw == "charging" and isinstance(remaining_hours, (int, float)):
                base += f" {self._format_hours(float(remaining_hours))} to full"
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

    def _format_hours(self, hours: float) -> str:
        if hours <= 0:
            return "0m"
        total_minutes = int(round(hours * 60.0))
        h, m = divmod(total_minutes, 60)
        if h <= 0:
            return f"{m}m"
        return f"{h}h{m:02d}m"

    def _read_battery_metrics(self) -> dict[str, float | str | None] | None:
        root = Path("/sys/class/power_supply")
        if not root.exists() or not root.is_dir():
            return None

        for dev in root.iterdir():
            if not dev.is_dir():
                continue
            dev_type = (self._read_text_file(dev / "type") or "").strip().lower()
            is_battery = dev_type == "battery" or dev.name.upper().startswith("BAT")
            if not is_battery:
                continue

            status = self._read_text_file(dev / "status")
            capacity = self._read_number_file(dev / "capacity")
            if capacity is None:
                energy_now = self._read_number_file(dev / "energy_now")
                energy_full = self._read_number_file(dev / "energy_full")
                charge_now = self._read_number_file(dev / "charge_now")
                charge_full = self._read_number_file(dev / "charge_full")
                if energy_now is not None and energy_full and energy_full > 0:
                    capacity = (energy_now / energy_full) * 100.0
                elif charge_now is not None and charge_full and charge_full > 0:
                    capacity = (charge_now / charge_full) * 100.0

            watts = self._read_device_watts(dev)
            remaining_hours = self._estimate_battery_remaining_hours(dev, status)
            return {
                "status": status,
                "capacity_pct": round(capacity, 1) if capacity is not None else None,
                "remaining_hours": remaining_hours,
                "watts": round(watts, 2) if watts is not None else None,
            }

        return None

    def _estimate_battery_remaining_hours(self, dev: Path, status: str | None) -> float | None:
        status_raw = (status or "").strip().lower()

        # Prefer kernel-provided seconds when available.
        time_to_empty = self._read_number_file(dev / "time_to_empty_now")
        if status_raw == "discharging" and time_to_empty is not None:
            return max(0.0, time_to_empty / 3600.0)
        time_to_full = self._read_number_file(dev / "time_to_full_now")
        if status_raw == "charging" and time_to_full is not None:
            return max(0.0, time_to_full / 3600.0)

        if status_raw not in {"discharging", "charging"}:
            return None

        energy_now = self._read_number_file(dev / "energy_now")
        energy_full = self._read_number_file(dev / "energy_full")
        power_now = self._read_number_file(dev / "power_now")
        if power_now and power_now > 0:
            if status_raw == "discharging" and energy_now is not None:
                return max(0.0, energy_now / power_now)
            if status_raw == "charging" and energy_now is not None and energy_full is not None:
                return max(0.0, (energy_full - energy_now) / power_now)

        charge_now = self._read_number_file(dev / "charge_now")
        charge_full = self._read_number_file(dev / "charge_full")
        current_now = self._read_number_file(dev / "current_now")
        if current_now and current_now > 0:
            if status_raw == "discharging" and charge_now is not None:
                return max(0.0, charge_now / current_now)
            if status_raw == "charging" and charge_now is not None and charge_full is not None:
                return max(0.0, (charge_full - charge_now) / current_now)

        return None

    def _read_cpu_percent(self) -> float | None:
        try:
            first = Path("/proc/stat").read_text(encoding="utf-8").splitlines()[0]
        except (OSError, IndexError):
            return None

        parts = first.split()
        if len(parts) < 5 or parts[0] != "cpu":
            return None
        try:
            nums = [int(v) for v in parts[1:]]
        except ValueError:
            return None

        idle = nums[3] + (nums[4] if len(nums) > 4 else 0)
        total = sum(nums)
        prev = self._prev_cpu_sample
        self._prev_cpu_sample = (total, idle)
        if prev is None:
            return None

        total_delta = total - prev[0]
        idle_delta = idle - prev[1]
        if total_delta <= 0:
            return None
        busy = total_delta - idle_delta
        return round((busy / total_delta) * 100.0, 1)

    def _read_power_metrics(self) -> tuple[float | None, float | None]:
        root = Path("/sys/class/power_supply")
        if not root.exists() or not root.is_dir():
            rapl_watts = self._read_powercap_watts()
            if rapl_watts is None:
                rapl_watts = self._read_hwmon_watts()
            return (rapl_watts, None)

        watts_total = 0.0
        watts_max_total = 0.0
        found_watts = False

        for dev in root.iterdir():
            if not dev.is_dir():
                continue
            watts = self._read_device_watts(dev)
            if watts is not None:
                watts_total += watts
                found_watts = True
            max_watts = self._read_device_max_watts(dev)
            if max_watts is not None:
                watts_max_total += max_watts

        if not found_watts:
            rapl_watts = self._read_powercap_watts()
            if rapl_watts is None:
                rapl_watts = self._read_hwmon_watts()
            return (rapl_watts, None)

        pct: float | None = None
        if watts_max_total > 0:
            pct = round(min(100.0, (watts_total / watts_max_total) * 100.0), 1)
        return (round(watts_total, 2), pct)

    def _read_powercap_watts(self) -> float | None:
        root = Path("/sys/class/powercap")
        if not root.exists() or not root.is_dir():
            return None

        now = time.monotonic()
        total_watts = 0.0
        have_delta = False

        for zone in root.glob("intel-rapl:*"):
            # Ignore nested subdomains (e.g. intel-rapl:0:0) to avoid double counting.
            if zone.name.count(":") != 1:
                continue
            energy_uj = self._read_number_file(zone / "energy_uj")
            if energy_uj is None:
                continue
            max_energy_uj = self._read_number_file(zone / "max_energy_range_uj")
            key = str(zone)
            prev = self._prev_powercap_sample.get(key)
            self._prev_powercap_sample[key] = (energy_uj, now, max_energy_uj)
            if prev is None:
                continue

            prev_energy, prev_time, prev_max = prev
            dt = now - prev_time
            if dt <= 0:
                continue
            delta = energy_uj - prev_energy
            wrap = max_energy_uj if max_energy_uj is not None else prev_max
            if delta < 0 and wrap and wrap > 0:
                delta += wrap
            if delta < 0:
                continue

            watts = (delta / 1_000_000.0) / dt
            if watts >= 0:
                total_watts += watts
                have_delta = True

        if not have_delta:
            return None
        return round(total_watts, 2)

    def _read_hwmon_watts(self) -> float | None:
        root = Path("/sys/class/hwmon")
        if not root.exists() or not root.is_dir():
            return None

        # First choice: direct power sensors, typically in microwatts.
        direct_watts = 0.0
        found_direct = False
        for dev in root.glob("hwmon*"):
            if not dev.is_dir():
                continue
            for path in dev.glob("power*_input"):
                value = self._read_number_file(path)
                if value is None:
                    continue
                direct_watts += value / 1_000_000.0
                found_direct = True
        if found_direct:
            return round(direct_watts, 2)

        # Fallback for INA3221-style rails: W = (mV * mA) / 1_000_000.
        # Prefer the VDD_IN rail if available; it is usually total input draw.
        total_pairs_watts = 0.0
        found_pairs = False
        for dev in root.glob("hwmon*"):
            if not dev.is_dir():
                continue
            if (self._read_text_file(dev / "name") or "").strip().lower() != "ina3221":
                continue

            rail_watts: dict[int, float] = {}
            for in_path in dev.glob("in*_input"):
                suffix = in_path.name[len("in") : -len("_input")]
                if not suffix.isdigit():
                    continue
                idx = int(suffix)
                curr_path = dev / f"curr{idx}_input"
                if not curr_path.is_file():
                    continue
                mv = self._read_number_file(in_path)
                ma = self._read_number_file(curr_path)
                if mv is None or ma is None:
                    continue
                rail_watts[idx] = (mv * ma) / 1_000_000.0

            if not rail_watts:
                continue

            for idx, watts in rail_watts.items():
                label = (self._read_text_file(dev / f"in{idx}_label") or "").strip().upper()
                if "VDD_IN" in label:
                    return round(max(0.0, watts), 2)

            for idx, watts in rail_watts.items():
                label = (self._read_text_file(dev / f"in{idx}_label") or "").strip().lower()
                if "sum of shunt" in label:
                    continue
                total_pairs_watts += max(0.0, watts)
                found_pairs = True

        if found_pairs:
            return round(total_pairs_watts, 2)
        return None

    def _read_device_watts(self, dev: Path) -> float | None:
        power_now = self._read_number_file(dev / "power_now")
        if power_now is not None:
            return power_now / 1_000_000.0

        current_now = self._read_number_file(dev / "current_now")
        voltage_now = self._read_number_file(dev / "voltage_now")
        if current_now is not None and voltage_now is not None:
            return (current_now * voltage_now) / 1_000_000_000_000.0
        return None

    def _read_device_max_watts(self, dev: Path) -> float | None:
        power_max = self._read_number_file(dev / "power_max_design")
        if power_max is not None:
            return power_max / 1_000_000.0

        current_max = self._read_number_file(dev / "current_max")
        voltage_max = self._read_number_file(dev / "voltage_max")
        if current_max is not None and voltage_max is not None:
            return (current_max * voltage_max) / 1_000_000_000_000.0
        return None

    def _read_number_file(self, path: Path) -> float | None:
        try:
            raw = path.read_text(encoding="utf-8").strip()
        except OSError:
            return None
        try:
            return float(raw)
        except ValueError:
            return None

    def _read_text_file(self, path: Path) -> str | None:
        try:
            return path.read_text(encoding="utf-8").strip()
        except OSError:
            return None

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
        log.write(
            "  [bold bright_white]"
            f"Resumed previous session: {max(0, len(self.agent.messages) - 1)} messages, "
            f"{len(self.loaded_files)} loaded files."
            "[/]"
        )
        if self.session_logger:
            self.session_logger.log_event(
                "session_state_restored",
                messages=max(0, len(self.agent.messages) - 1),
                loaded_files=len(self.loaded_files),
                state_path=str(self.state_store.path),
            )
        return True

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
            "model": self.cfg.get("model"),
            "device": self.cfg.get("device", "auto"),
            "context_window_tokens": self.client.context_window_tokens if self.client else self.cfg.get("context_window_tokens", 2048),
            "messages": self.agent.messages,
            "loaded_files": self.loaded_files,
        }
        try:
            self.state_store.save(payload)
        except Exception as exc:
            if self.session_logger:
                self.session_logger.log_event("session_state_save_error", reason=reason, error=str(exc))

    # -- Agent turn ----------------------------------------------------------

    def _start_agent_turn(self, recovery_attempted: bool = False) -> None:
        self._generation_worker = self.run_agent_turn(recovery_attempted=recovery_attempted)

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
            self.session_logger.log_event("generation_stopped", source="escape")
        self._render_token_counter()

    @work(exclusive=True)
    async def run_agent_turn(self, recovery_attempted: bool = False) -> None:
        log = self.query_one("#chat-log", RichLog)
        pending_tool_calls: list[ToolCall] = []
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
                    if not recovery_attempted and self._is_recoverable_runtime_error(event.text):
                        recovered = await self._recover_runtime(log, event.text)
                        if recovered:
                            self._start_agent_turn(recovery_attempted=True)
                            return
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
        except asyncio.CancelledError:
            return
        finally:
            self._stop_thinking(thinking_token)
            if self._generation_worker is current_worker:
                self._generation_worker = None

        if self.session_logger and assistant_turn_text.strip():
            self.session_logger.log_event("assistant_message", text=assistant_turn_text)

        for tc in pending_tool_calls:
            try:
                await self._handle_tool_call(tc, log)
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
            self.persist_session_state(reason="assistant_turn_with_tools")
            self._start_agent_turn()
        else:
            self.persist_session_state(reason="assistant_turn_done")
            self._render_token_counter()

    async def _handle_tool_call(self, tc: ToolCall, log: RichLog) -> None:
        if self.agent.is_internal_condense_tool(tc):
            log.write(f"[yellow]{tc.name}:[/] {_fmt_args(tc)}")
            result = await self.agent.condense_context()
            log.write(f"  [bold bright_white]{result}[/]")
            log.write("")
            self.agent.complete_tool_call(tc, result)
            self.persist_session_state(reason="auto_condense")
            return

        needs_confirm = self.agent.needs_confirmation(tc)

        if needs_confirm:
            log.write(f"[yellow]{tc.name}:[/]")
            for preview_line in self._tool_preview_lines(tc):
                log.write(f"  [bold bright_white]{preview_line}[/]")
            approved = await self._wait_for_tool_approval(tc)
            if not approved:
                log.write("[red]  denied[/]")
                log.write("")
                if self.session_logger:
                    self.session_logger.log_event(
                        "tool_approval",
                        tool=tc.name,
                        approved=False,
                        arguments=tc.arguments,
                    )
                self.agent.complete_tool_call(tc, "User denied this action.")
                self.persist_session_state(reason=f"tool_denied:{tc.name}")
                return
            log.write("[green]  approved[/]")
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
        result_for_context, clipped_tool_result = self._fit_tool_result_to_budget(result)
        duration_ms = round((time.monotonic() - t0) * 1000.0, 2)
        if self.session_logger:
            self.session_logger.log_tool_result(
                tc.name,
                result,
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
        self.agent.complete_tool_call(tc, result_for_context)
        self.persist_session_state(reason=f"tool_result:{tc.name}")

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

        remaining = self._remaining_prompt_tokens()
        budget_tokens = max(
            TOOL_RESULT_MIN_BUDGET_TOKENS,
            remaining - TOOL_RESULT_SAFE_PADDING_TOKENS,
        )
        if estimate_tokens(result) <= budget_tokens:
            return result, False

        suffix = "\n...[tool output truncated for context safety]"
        max_chars = max(256, budget_tokens * 4)
        clipped = result[:max_chars]
        candidate = clipped + suffix

        # Tighten iteratively so the final payload stays within budget.
        while estimate_tokens(candidate) > budget_tokens and len(clipped) > 64:
            clipped = clipped[: max(64, int(len(clipped) * 0.85))]
            candidate = clipped + suffix

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
            "502",
            "503",
            "504",
        )
        return any(needle in lowered for needle in needles)

    async def _recover_runtime(self, log: RichLog, error_text: str) -> bool:
        if not self.client:
            return False
        log.write("[yellow]LLM runtime interrupted. Restarting llama-server once and retrying...[/]")
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
        preview = self._approval_preview_text(self._approval_tool_call)
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
            f"[bold yellow]Tool request:[/]\n{preview}\n"
            f"Use [bold]←[/]/[bold]→[/] then [bold]Enter[/]   {approve}  {deny}"
        )

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
            if len(command) > 200:
                command = command[:197] + "..."
            return [f"command: {command}"]
        if tc.name == "write_file":
            path = str(tc.arguments.get("path", "")).strip()
            content = str(tc.arguments.get("content", ""))
            preview = content.replace("\r\n", "\n").replace("\r", "\n")
            lines = [f"path: {path}", f"bytes: {len(content)}", "content:"]
            lines.extend(escape(line) for line in preview.split("\n"))
            return lines
        return [str(_fmt_args(tc))]

    def _approval_preview_text(self, tc: ToolCall) -> str:
        lines = self._tool_preview_lines(tc)
        joined = "\n".join(lines)
        if tc.name == "write_file":
            return joined
        if len(joined) > 280:
            return joined[:277] + "..."
        return joined


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

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="open-jet offline agentic TUI")
    parser.add_argument(
        "--setup",
        action="store_true",
        help="start in setup wizard mode before launching the chat UI",
    )
    args = parser.parse_args(argv)

    app = OpenJetApp(force_setup=args.setup)
    app.run(mouse=False)


if __name__ == "__main__":
    main()
