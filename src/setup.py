from __future__ import annotations

import os
import re
from pathlib import Path

from textual import events
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Input, Static

from .config import JETSON_OVERRIDE_OPTIONS
from .hardware import (
    HardwareInfo,
    effective_hardware_info,
    is_jetson_label,
    recommended_context_window_tokens,
    recommended_context_window_tokens_from_total,
    recommended_device_for_hardware,
    recommended_gpu_layers,
    recommended_llm_models,
    recommended_param_budget_b,
)
from .ollama_setup import discover_installed_ollama_models, find_ollama_cli

ACCENT_GREEN = "#88D83F"


def discover_model_files() -> list[str]:
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


def discover_trt_model_dirs() -> list[str]:
    roots = [
        Path.cwd(),
        Path.cwd() / "models",
        Path.home() / "models",
    ]
    found: set[str] = set()
    for root in roots:
        if not root.exists() or not root.is_dir():
            continue
        for path in root.iterdir():
            if not path.is_dir():
                continue
            # Typical HF local checkpoints include config + tokenizer files.
            if (path / "config.json").is_file() and (path / "tokenizer_config.json").is_file():
                found.add(str(path.resolve()))
    return sorted(found)


def estimate_model_params_b_from_text(text: str) -> float | None:
    src = text.strip().lower()
    if not src:
        return None
    match = re.search(r"(?<!\d)(\d+(?:\.\d+)?)\s*([bm])(?!\w)", src)
    if not match:
        return None
    try:
        value = float(match.group(1))
    except ValueError:
        return None
    unit = match.group(2)
    if unit == "b":
        return value
    return value / 1000.0


def context_window_options(recommended: int) -> list[int]:
    options = [1024, 1536, 2048, 3072, 4096, 6144, 8192]
    if recommended not in options:
        options.append(recommended)
    return sorted(set(options))


def gpu_layer_options(device: str, recommended: int) -> list[int]:
    base = [0] if device == "cpu" else [0, 10, 20, 28, 35]
    if recommended not in base:
        base.append(recommended)
    return sorted(set(base))


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
        installed_ollama_models: list[str],
        hardware_info: HardwareInfo,
        recommended_ctx: int,
        exit_on_cancel: bool = True,
    ) -> None:
        super().__init__()
        self.model_options = model_options
        self.installed_ollama_models = sorted(set(installed_ollama_models))
        self.hardware_info = hardware_info
        self.recommended_ctx = max(512, int(recommended_ctx))
        self.exit_on_cancel = exit_on_cancel
        self.ollama_cli = find_ollama_cli()
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
            Static("", id="setup-warning"),
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
        runtime_rows: list[tuple[str, str]] = [
            ("llama.cpp (GGUF)", "llama_cpp"),
            ("TensorRT-LLM (PyTorch runtime)", "trtllm_pytorch"),
        ]
        local_model_rows: list[tuple[str, str]] = [(Path(model).name, model) for model in self.model_options]
        local_model_rows.append(("Manual path", "__manual__"))
        trt_model_rows: list[tuple[str, str]] = [
            (Path(model).name, model) for model in discover_trt_model_dirs()
        ]
        trt_model_rows.append(("Manual path or HF model id", "__manual__"))
        context_rows = [
            (f"{value} (recommended)" if value == self.recommended_ctx else str(value), value)
            for value in context_window_options(self.recommended_ctx)
        ]

        self._steps = [
            {"key": "hardware", "title": "Hardware Detection", "options": hardware_rows},
            {"key": "hardware_override", "title": "Hardware Override", "options": hardware_override_rows},
            {"key": "runtime", "title": "Runtime", "options": runtime_rows},
            {"key": "model_plan", "title": "Model Source", "options": []},
            {"key": "local_model", "title": "Local Model File", "options": local_model_rows},
            {"key": "trt_model", "title": "TensorRT Model", "options": trt_model_rows},
            {"key": "context_window_tokens", "title": "Context Size", "options": context_rows},
            {"key": "gpu_layers", "title": "GPU Offload", "options": []},
        ]

        defaults = {
            "hardware": "auto",
            "hardware_override": hardware_override_rows[0][1] if hardware_override_rows else "",
            "runtime": "llama_cpp",
            "local_model": local_model_rows[0][1] if local_model_rows else "__manual__",
            "trt_model": trt_model_rows[0][1] if trt_model_rows else "__manual__",
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
        elif trt_model_rows:
            manual_input.value = trt_model_rows[0][1]
        self._sync_dynamic_steps()

    def _step_by_key(self, key: str) -> dict | None:
        return next((step for step in self._steps if step["key"] == key), None)

    def _selected_hardware_profile(self) -> str:
        return str(self._selections.get("hardware", "auto"))

    def _selected_hardware_override(self) -> str:
        return str(self._selections.get("hardware_override", ""))

    def _selected_runtime(self) -> str:
        return str(self._selections.get("runtime", "llama_cpp"))

    def _effective_hardware(self) -> HardwareInfo:
        return effective_hardware_info(
            self._selected_hardware_profile(),
            self.hardware_info,
            self._selected_hardware_override(),
        )

    def _jetson_target_selected(self) -> bool:
        profile = self._selected_hardware_profile()
        override = self._selected_hardware_override()
        if profile == "other" and override.startswith("jetson_"):
            return True
        return is_jetson_label(self._effective_hardware().label)

    def _recommended_context_for_current_hardware(self) -> int:
        if self._selected_hardware_profile() == "auto":
            return recommended_context_window_tokens()
        hw = self._effective_hardware()
        headless = not bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
        return recommended_context_window_tokens_from_total(hw.total_ram_gb, headless=headless)

    def _sync_dynamic_steps(self) -> None:
        self._sync_model_plan_step()
        self._sync_context_step()
        self._sync_gpu_step()

    def _sync_model_plan_step(self) -> None:
        model_step = self._step_by_key("model_plan")
        if not model_step:
            return
        # TRT runtime branch temporarily disabled.
        old_value = self._selections.get("model_plan")
        max_b = recommended_param_budget_b(
            self._selected_hardware_profile(),
            self.hardware_info,
            self._selected_hardware_override(),
        )
        download_rows = [
            (f"Download with Ollama: {label}", tag)
            for label, tag in recommended_llm_models(max_b)[:3]
        ]
        installed_rows = [
            (f"Use installed Ollama model: {tag}", tag)
            for tag in self.installed_ollama_models
        ]
        installed_tags = {tag for _label, tag in installed_rows}
        filtered_download_rows = [row for row in download_rows if row[1] not in installed_tags]
        rows = [("Use a local .gguf model file", "__local__"), *installed_rows, *filtered_download_rows]
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
        values = context_window_options(recommended)
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
        # TRT runtime temporarily disabled — always use llama.cpp / GGUF.
        if key == "runtime":
            return False
        if key == "trt_model":
            return False
        if key == "model_plan":
            return True
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
        show_manual = False
        if key == "local_model" and self._selections.get("local_model") == "__manual__":
            show_manual = True
            manual_label.update("Manual model path (.gguf):")
            manual_input.placeholder = "/path/to/model.gguf"
        elif key == "trt_model" and self._selections.get("trt_model") == "__manual__":
            show_manual = True
            manual_label.update("TensorRT model path or HF model id:")
            manual_input.placeholder = "/path/to/model-dir or org/model"
        if show_manual:
            manual_label.remove_class("hidden")
            manual_input.remove_class("hidden")
            manual_input.focus()
        else:
            manual_label.add_class("hidden")
            manual_input.add_class("hidden")
            if self.focused is manual_input:
                self.set_focus(None)
        self.query_one("#setup-warning", Static).update(self._warning_text())
        self.query_one("#setup-error", Static).update("")

    def action_next_option(self) -> None:
        step = self._current_step()
        key = str(step["key"])
        options = list(step["options"])
        self._indices[key] = (self._indices[key] + 1) % len(options)
        self._selections[key] = options[self._indices[key]][1]
        if key in {"hardware", "hardware_override", "runtime"}:
            self._sync_dynamic_steps()
        self._render_step()

    def action_prev_option(self) -> None:
        step = self._current_step()
        key = str(step["key"])
        options = list(step["options"])
        self._indices[key] = (self._indices[key] - 1) % len(options)
        self._selections[key] = options[self._indices[key]][1]
        if key in {"hardware", "hardware_override", "runtime"}:
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
        recommended = recommended_gpu_layers(device, hw.total_ram_gb)
        gpu_step = next((step for step in self._steps if step["key"] == "gpu_layers"), None)
        if not gpu_step:
            return
        old_value = self._selections.get("gpu_layers")
        values = gpu_layer_options(device, recommended)
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

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id != "setup-model-path":
            return
        self._render_step()

    def action_save(self) -> None:
        self.ollama_cli = find_ollama_cli()
        # TRT runtime temporarily disabled — always use llama_cpp.
        runtime = "llama_cpp"
        model_plan = str(self._selections.get("model_plan", ""))
        model_source = "local" if model_plan == "__local__" else "ollama"
        profile = self._selected_hardware_profile()
        override_key = str(self._selections.get("hardware_override", ""))
        device = recommended_device_for_hardware(profile, self.hardware_info, override_key)
        hw = self._effective_hardware()
        payload: dict[str, object] = {
            "runtime": runtime,
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
            payload["llama_model"] = str(model_file)

        context_value = int(self._selections.get("context_window_tokens", self.recommended_ctx))
        if context_value < 512:
            self._set_error("Select a valid context window.")
            return

        gpu_value = int(self._selections.get("gpu_layers", recommended_gpu_layers(device, hw.total_ram_gb)))
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

    def _selected_trt_model(self) -> str:
        selected = self._selections.get("trt_model")
        if isinstance(selected, str) and selected != "__manual__":
            return selected
        return self.query_one("#setup-model-path", Input).value.strip()

    def _selected_params_b(self) -> float | None:
        model_plan = str(self._selections.get("model_plan", ""))
        if model_plan and model_plan != "__local__":
            return estimate_model_params_b_from_text(model_plan)

        local_path = self._selected_model_path()
        if local_path:
            return estimate_model_params_b_from_text(Path(local_path).name)
        return None

    def _warning_text(self) -> str:
        params_b = self._selected_params_b()
        if params_b is None:
            return ""
        if params_b <= 2.0:
            return "[bold yellow]Warning:[/] [yellow]Models with 2B parameters or smaller are not recommended for accurate results.[/]"
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
            if str(value) in self.installed_ollama_models:
                return "already downloaded"
            if not self.ollama_cli:
                return "install ollama first"
            if self._jetson_target_selected():
                return "quantized GGUF required for Jetson"
            cap = recommended_param_budget_b(
                self._selected_hardware_profile(),
                self.hardware_info,
                self._selected_hardware_override(),
            )
            return f"fit for this hardware (~{cap:g}B)"
        if key == "local_model":
            if value == "__manual__":
                return "enter full path"
            return "detected file"
        if key == "trt_model":
            if value == "__manual__":
                return "path or HF model id"
            return "detected checkpoint"
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
        if key == "runtime":
            return "Choose inference backend runtime."
        if key == "model_plan":
            if not self.ollama_cli:
                return "Choose local file, or install Ollama to enable downloads."
            if self._jetson_target_selected():
                return "Choose local file, installed Ollama model, or download (quantized GGUF only on Jetson)."
            return "Choose local file, installed Ollama model, or download."
        if key == "local_model":
            return "Select a detected .gguf or enter a path."
        if key == "trt_model":
            return "Select a local checkpoint dir or enter a HF model id/path."
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
