from __future__ import annotations

import os
import re
from pathlib import Path
from typing import TYPE_CHECKING, Mapping

from rich.console import Console
from rich.markup import escape

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
from .model_profiles import default_profile_name
from .ollama_setup import discover_installed_ollama_models, find_ollama_cli
from .runtime_registry import runtime_options, runtime_spec

if TYPE_CHECKING:
    from prompt_toolkit import PromptSession

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
            if (path / "config.json").is_file() and (path / "tokenizer_config.json").is_file():
                found.add(str(path.resolve()))
    return sorted(found)


def discover_sglang_model_dirs() -> list[str]:
    return discover_trt_model_dirs()


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
    return value if match.group(2) == "b" else value / 1000.0


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


def _dedupe_refs(refs: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for ref in refs:
        value = str(ref).strip()
        if not value or value in seen:
            continue
        seen.add(value)
        unique.append(value)
    return unique


def _saved_model_refs(current_cfg: Mapping[str, object] | None, runtime: str) -> list[str]:
    if not isinstance(current_cfg, Mapping):
        return []
    refs: list[str] = []
    history = current_cfg.get("setup_model_history")
    if isinstance(history, dict):
        saved = history.get(runtime)
        if isinstance(saved, list):
            refs.extend(str(item).strip() for item in saved)
    model_key = runtime_spec(runtime).model_config_key
    active_ref = str(current_cfg.get(model_key) or "").strip()
    if active_ref:
        refs.insert(0, active_ref)
    return _dedupe_refs(refs)


def _remember_model_ref(
    payload: dict[str, object],
    current_cfg: Mapping[str, object] | None,
    runtime: str,
    model_ref: str,
) -> None:
    ref = str(model_ref).strip()
    if not ref:
        return
    history_payload: dict[str, list[str]] = {}
    existing = current_cfg.get("setup_model_history") if isinstance(current_cfg, Mapping) else None
    if isinstance(existing, dict):
        for key, value in existing.items():
            if isinstance(value, list):
                history_payload[str(key)] = _dedupe_refs([str(item) for item in value])
    history_payload[runtime] = _dedupe_refs([ref, *history_payload.get(runtime, [])])
    payload["setup_model_history"] = history_payload


async def _prompt_text(
    session: PromptSession[object] | None,
    prompt: str,
    *,
    default: str = "",
) -> str:
    if session is not None:
        result = await session.prompt_async(prompt, default=default)
        return result.strip()
    return input(f"{prompt}{default}").strip()


async def _prompt_choice(
    session: PromptSession[object] | None,
    console: Console,
    title: str,
    options: list[tuple[str, object]],
    *,
    default_index: int = 0,
    detail: str | None = None,
) -> object:
    console.print(f"[bold {ACCENT_GREEN}]{escape(title)}[/]")
    if detail:
        console.print(f"[dim]{escape(detail)}[/]")
    for idx, (label, _value) in enumerate(options, start=1):
        marker = " (default)" if idx - 1 == default_index else ""
        console.print(f"  [bold]{idx}.[/] {escape(str(label))}{marker}")
    while True:
        raw = await _prompt_text(session, "choice> ", default=str(default_index + 1))
        if not raw:
            return options[default_index][1]
        try:
            picked = int(raw) - 1
        except ValueError:
            console.print("[yellow]Enter a number from the list.[/]")
            continue
        if 0 <= picked < len(options):
            return options[picked][1]
        console.print("[yellow]Choice out of range.[/]")


async def run_setup_wizard(
    *,
    session: PromptSession[object] | None,
    console: Console,
    hardware_info: HardwareInfo,
    recommended_ctx: int,
    current_cfg: Mapping[str, object] | None = None,
) -> dict | None:
    ram_text = f"{hardware_info.total_ram_gb:.1f} GB RAM" if hardware_info.total_ram_gb > 0 else "RAM unknown"
    console.print(f"[bold {ACCENT_GREEN}]open-jet setup[/]")
    console.print(f"[dim]Detected hardware: {escape(hardware_info.label)} ({escape(ram_text)})[/]")

    hardware = await _prompt_choice(
        session,
        console,
        "Hardware profile",
        [
            (f"Use detected hardware ({hardware_info.label}, {ram_text})", "auto"),
            ("Pick hardware profile manually", "other"),
        ],
    )

    hardware_override = ""
    if hardware == "other":
        hardware_override = str(
            await _prompt_choice(
                session,
                console,
                "Hardware override",
                [(label, key) for key, label, _ram in JETSON_OVERRIDE_OPTIONS],
            )
        )

    runtime = str(
        await _prompt_choice(
            session,
            console,
            "Runtime",
            runtime_options(),
        )
    )

    payload: dict[str, object] = {
        "runtime": runtime,
        "hardware_profile": hardware,
        "hardware_override": hardware_override if hardware == "other" else "",
    }

    effective_hw = effective_hardware_info(str(hardware), hardware_info, hardware_override)
    jetson_target = (hardware == "other" and hardware_override.startswith("jetson_")) or is_jetson_label(effective_hw.label)
    device = recommended_device_for_hardware(str(hardware), hardware_info, hardware_override)

    if runtime == "llama_cpp":
        model_files = discover_model_files()
        saved_model_files = _saved_model_refs(current_cfg, runtime)
        ollama_cli = find_ollama_cli()
        installed_ollama = discover_installed_ollama_models() if ollama_cli else []
        max_b = recommended_param_budget_b(str(hardware), hardware_info, hardware_override)
        download_rows = [
            (f"Download with Ollama: {label}", tag)
            for label, tag in recommended_llm_models(max_b)[:3]
        ]
        installed_rows = [
            (f"Use installed Ollama model: {tag}", tag)
            for tag in installed_ollama
        ]
        installed_tags = {tag for _label, tag in installed_rows}
        download_rows = [row for row in download_rows if row[1] not in installed_tags]
        model_plan_options: list[tuple[str, object]] = [("Use a local .gguf model file", "__local__")]
        model_plan_options.extend(installed_rows)
        model_plan_options.extend(download_rows)
        detail = (
            "Choose local GGUF, an installed Ollama model, or a recommended Ollama download."
            if ollama_cli
            else "Ollama not found; using a local GGUF model file."
        )
        model_plan = "__local__"
        if len(model_plan_options) > 1:
            model_plan = str(
                await _prompt_choice(
                    session,
                    console,
                    "Model source",
                    model_plan_options,
                    detail=detail,
                )
            )
        if model_plan == "__local__":
            local_rows = [(Path(model).name, model) for model in model_files]
            local_rows.extend(
                (f"{Path(model).name} (saved)", model)
                for model in saved_model_files
                if model not in model_files
            )
            local_rows.append(("Manual path", "__manual__"))
            local_choice = await _prompt_choice(
                session,
                console,
                "Local model",
                local_rows,
                detail="Select a detected GGUF file or choose Manual path.",
            )
            if local_choice == "__manual__":
                model_path = await _prompt_text(session, "model path> ")
            else:
                model_path = str(local_choice)
            model_file = Path(model_path).expanduser()
            if not model_file.is_file():
                raise RuntimeError("Model file does not exist.")
            if model_file.suffix.lower() != ".gguf":
                raise RuntimeError("Model file must end with .gguf.")
            payload["model_source"] = "local"
            payload["model"] = str(model_file)
            payload["llama_model"] = str(model_file)
            _remember_model_ref(payload, current_cfg, runtime, str(model_file))
        else:
            if not ollama_cli:
                raise RuntimeError("Ollama CLI not found.")
            if jetson_target:
                console.print("[dim]Jetson targets require quantized GGUF-backed Ollama models.[/]")
            payload["model_source"] = "ollama"
            payload["recommended_llm"] = model_plan
            payload["ollama_model"] = model_plan
    elif runtime == "sglang":
        discovered = discover_sglang_model_dirs()
        saved_refs = _saved_model_refs(current_cfg, runtime)
        rows = [(Path(model).name, model) for model in discovered]
        rows.extend(
            (f"{Path(model).name or model} (saved)", model)
            for model in saved_refs
            if model not in discovered
        )
        rows.append(("Manual path or HF model id", "__manual__"))
        choice = await _prompt_choice(session, console, "SGLang model", rows)
        model_ref = await _prompt_text(session, "sglang model> ") if choice == "__manual__" else str(choice)
        if not model_ref:
            raise RuntimeError("SGLang model path or HF model id is required.")
        payload["model_source"] = "local"
        payload["model"] = model_ref
        payload["sglang_model"] = model_ref
        payload["sglang_launch_mode"] = "jetson_container" if jetson_target else "managed"
        payload["sglang_served_model_name"] = "local"
        _remember_model_ref(payload, current_cfg, runtime, model_ref)
    else:
        discovered = discover_trt_model_dirs()
        saved_refs = _saved_model_refs(current_cfg, runtime)
        rows = [(Path(model).name, model) for model in discovered]
        rows.extend(
            (f"{Path(model).name or model} (saved)", model)
            for model in saved_refs
            if model not in discovered
        )
        rows.append(("Manual path or HF model id", "__manual__"))
        choice = await _prompt_choice(session, console, "TensorRT model", rows)
        model_ref = await _prompt_text(session, "trt model> ") if choice == "__manual__" else str(choice)
        if not model_ref:
            raise RuntimeError("TensorRT model path or HF model id is required.")
        payload["model_source"] = "local"
        payload["model"] = model_ref
        payload["trtllm_model"] = model_ref
        _remember_model_ref(payload, current_cfg, runtime, model_ref)

    headless = not bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    if hardware == "auto":
        recommended_ctx_value = recommended_context_window_tokens()
    else:
        recommended_ctx_value = recommended_context_window_tokens_from_total(
            effective_hw.total_ram_gb,
            headless=headless,
        )
    recommended_ctx_value = max(recommended_ctx, recommended_ctx_value)
    context_value = int(
        await _prompt_choice(
            session,
            console,
            "Context window",
            [
                (f"{value} (recommended)" if value == recommended_ctx_value else str(value), value)
                for value in context_window_options(recommended_ctx_value)
            ],
        )
    )

    gpu_value = 0
    if runtime == "llama_cpp":
        recommended_gpu = recommended_gpu_layers(device, effective_hw.total_ram_gb)
        gpu_value = int(
            await _prompt_choice(
                session,
                console,
                "GPU layers",
                [
                    (f"{value} (recommended)" if value == recommended_gpu else str(value), value)
                    for value in gpu_layer_options(device, recommended_gpu)
                ],
                detail="Higher offload can be faster but uses more memory.",
            )
        )

    payload.update(
        {
            "device": device,
            "context_window_tokens": context_value,
            "gpu_layers": gpu_value if runtime == "llama_cpp" else 0,
            "setup_complete": True,
        }
    )
    payload["model_profile_name"] = await _prompt_text(
        session,
        "model name> ",
        default=default_profile_name(payload),
    )
    return payload
