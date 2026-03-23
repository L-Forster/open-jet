from __future__ import annotations

import os
import re
import shutil
from html import escape as html_escape
from pathlib import Path
from typing import TYPE_CHECKING, Mapping

from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.key_binding import KeyBindings
from rich.console import Console
from rich.markup import escape

from .config import HARDWARE_OVERRIDE_OPTIONS
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
from .provisioning import recommend_direct_model
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
    options = [1024, 2048, 4096, 8192, 16384, 32768]
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


def _default_option_index(options: list[tuple[str, object]], value: object, *, fallback: int = 0) -> int:
    for idx, (_label, option_value) in enumerate(options):
        if option_value == value:
            return idx
    return fallback


def _configured_runtime(current_cfg: Mapping[str, object] | None) -> str:
    value = ""
    if isinstance(current_cfg, Mapping):
        value = str(current_cfg.get("runtime") or "").strip()
    spec = runtime_spec(value)
    return spec.key if spec.show_in_setup and spec.enabled else "llama_cpp"


def _discover_llama_server() -> str | None:
    found = shutil.which("llama-server")
    if found:
        return found
    candidate = Path.home() / "llama.cpp" / "build" / "bin" / "llama-server"
    if candidate.is_file():
        return str(candidate)
    return None


def _runtime_prompt_options(
    current_cfg: Mapping[str, object] | None,
    *,
    llama_ready: bool,
) -> list[tuple[str, str]]:
    configured_runtime = _configured_runtime(current_cfg)
    options: list[tuple[str, str]] = []
    for label, key in runtime_options():
        suffix = ""
        if key == "llama_cpp":
            suffix = " (recommended)" if llama_ready else " (recommended, setup can provision llama-server)"
        elif key == "openai_compatible":
            suffix = " (self-hosted gateway or compatible API)"
        elif key == "openrouter":
            suffix = " (optional hosted fallback)"
        if key == configured_runtime:
            suffix = f"{suffix}, current" if suffix else " (current)"
        options.append((f"{label}{suffix}", key))
    return options


def _current_string(current_cfg: Mapping[str, object] | None, key: str) -> str:
    if not isinstance(current_cfg, Mapping):
        return ""
    return str(current_cfg.get(key) or "").strip()


def _current_int(current_cfg: Mapping[str, object] | None, key: str) -> int | None:
    if not isinstance(current_cfg, Mapping):
        return None
    value = current_cfg.get(key)
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError:
            return None
    return None


def _current_bool(current_cfg: Mapping[str, object] | None, key: str) -> bool | None:
    if not isinstance(current_cfg, Mapping):
        return None
    value = current_cfg.get(key)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "on"}:
            return True
        if lowered in {"false", "0", "no", "off"}:
            return False
    return None


def _recommended_setup_runtime(current_cfg: Mapping[str, object] | None) -> str:
    raw_runtime = _current_string(current_cfg, "runtime")
    if raw_runtime:
        spec = runtime_spec(raw_runtime)
        if spec.enabled and spec.show_in_setup:
            return spec.key
    return "llama_cpp"


def _default_remote_model(runtime: str) -> str:
    if runtime == "openrouter":
        return "openai/gpt-4o-mini"
    return "gpt-4o-mini"


def _recommended_model_choice(
    *,
    hardware_info: HardwareInfo,
    current_cfg: Mapping[str, object] | None,
    runtime: str,
    model_files: list[str],
    saved_model_files: list[str],
    installed_ollama: list[str],
    ollama_cli: str | None,
    max_b: float,
) -> tuple[str, dict[str, object]]:
    current_model_source = _current_string(current_cfg, "model_source").lower()
    current_ollama_model = _current_string(current_cfg, "ollama_model")
    current_local_model = _current_string(current_cfg, runtime_spec(runtime).model_config_key)
    if current_model_source == "ollama" and current_ollama_model:
        return "ollama", {
            "model_source": "ollama",
            "recommended_llm": current_ollama_model,
            "ollama_model": current_ollama_model,
        }
    if current_local_model and Path(current_local_model).is_file():
        return "local", {
            "model_source": "local",
            "model": current_local_model,
            "llama_model": current_local_model,
        }
    if model_files:
        model_path = model_files[0]
        return "local", {
            "model_source": "local",
            "model": model_path,
            "llama_model": model_path,
        }
    existing_saved = [p for p in saved_model_files if Path(p).is_file()]
    if existing_saved:
        model_path = existing_saved[0]
        return "local", {
            "model_source": "local",
            "model": model_path,
            "llama_model": model_path,
        }
    if installed_ollama:
        tag = installed_ollama[0]
        return "ollama", {
            "model_source": "ollama",
            "recommended_llm": tag,
            "ollama_model": tag,
        }
    recommended = recommended_llm_models(max_b)
    if ollama_cli and recommended:
        tag = recommended[0][1]
        return "ollama", {
            "model_source": "ollama",
            "recommended_llm": tag,
            "ollama_model": tag,
        }
    direct = recommend_direct_model(hardware_info)
    return "direct", {
        "model_source": "direct",
        "model_download_url": direct["url"],
        "model_download_path": direct["target_path"],
        "recommended_llm": direct["label"],
        "setup_missing_model": True,
    }


def build_recommended_payload(
    *,
    hardware_info: HardwareInfo,
    recommended_ctx: int,
    current_cfg: Mapping[str, object] | None = None,
) -> dict[str, object]:
    hardware_profile = _current_string(current_cfg, "hardware_profile") or "auto"
    if hardware_profile != "other":
        hardware_profile = "auto"
    hardware_override = _current_string(current_cfg, "hardware_override") if hardware_profile == "other" else ""
    effective_hw = effective_hardware_info(hardware_profile, hardware_info, hardware_override)
    runtime = _recommended_setup_runtime(current_cfg)
    device = recommended_device_for_hardware(hardware_profile, hardware_info, hardware_override)

    payload: dict[str, object] = {
        "runtime": runtime,
        "hardware_profile": hardware_profile,
        "hardware_override": hardware_override,
    }

    if runtime == "llama_cpp":
        model_files = discover_model_files()
        saved_model_files = _saved_model_refs(current_cfg, runtime)
        ollama_cli = find_ollama_cli()
        installed_ollama = discover_installed_ollama_models() if ollama_cli else []
        max_b = recommended_param_budget_b(hardware_profile, hardware_info, hardware_override)
        _source, model_payload = _recommended_model_choice(
            hardware_info=hardware_info,
            current_cfg=current_cfg,
            runtime=runtime,
            model_files=model_files,
            saved_model_files=saved_model_files,
            installed_ollama=installed_ollama,
            ollama_cli=ollama_cli,
            max_b=max_b,
        )
        payload.update(model_payload)
        payload["setup_missing_runtime"] = _discover_llama_server() is None
        if "llama_model" in payload:
            _remember_model_ref(payload, current_cfg, runtime, str(payload["llama_model"]))
    elif runtime == "openai_compatible":
        model_ref = _current_string(current_cfg, "openai_compatible_model") or _default_remote_model(runtime)
        base_url = _current_string(current_cfg, "openai_compatible_base_url") or os.environ.get("OPENAI_BASE_URL", "").strip() or "https://api.openai.com"
        api_key_env = _current_string(current_cfg, "openai_compatible_api_key_env") or "OPENAI_API_KEY"
        payload.update(
            {
                "model_source": "remote",
                "model": model_ref,
                "openai_compatible_model": model_ref,
                "openai_compatible_base_url": base_url,
                "openai_compatible_api_key_env": api_key_env,
                "openai_compatible_verify_connection": _current_bool(current_cfg, "openai_compatible_verify_connection") or False,
                "setup_missing_api_key": not bool(os.environ.get(api_key_env, "").strip()),
            }
        )
    else:
        model_ref = _current_string(current_cfg, "openrouter_model") or _default_remote_model(runtime)
        api_key_env = _current_string(current_cfg, "openrouter_api_key_env") or "OPENROUTER_API_KEY"
        payload.update(
            {
                "model_source": "remote",
                "model": model_ref,
                "openrouter_model": model_ref,
                "openrouter_base_url": _current_string(current_cfg, "openrouter_base_url") or "https://openrouter.ai/api/v1",
                "openrouter_api_key_env": api_key_env,
                "openrouter_verify_connection": _current_bool(current_cfg, "openrouter_verify_connection") or False,
                "setup_missing_api_key": not bool(os.environ.get(api_key_env, "").strip()),
            }
        )

    headless = not bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    detected_recommended_ctx = (
        recommended_context_window_tokens()
        if hardware_profile == "auto"
        else recommended_context_window_tokens_from_total(
            effective_hw.total_ram_gb,
            headless=headless,
        )
    )
    context_value = _current_int(current_cfg, "context_window_tokens") or max(recommended_ctx, detected_recommended_ctx)
    gpu_value = 0
    if runtime == "llama_cpp":
        gpu_value = _current_int(current_cfg, "gpu_layers")
        if gpu_value is None:
            gpu_value = recommended_gpu_layers(device, effective_hw.total_ram_gb)

    payload.update(
        {
            "device": device,
            "context_window_tokens": context_value,
            "gpu_layers": gpu_value if runtime == "llama_cpp" else 0,
            "setup_complete": True,
            "model_profile_name": default_profile_name(payload),
        }
    )
    return payload


def _recommended_summary(payload: Mapping[str, object]) -> str:
    runtime = str(payload.get("runtime") or "llama_cpp")
    model_source = str(payload.get("model_source") or "local")
    model_ref = str(
        payload.get("ollama_model")
        or payload.get(runtime_spec(runtime).model_config_key)
        or payload.get("model")
        or ""
    ).strip()
    model_text = model_ref or "missing"
    notes: list[str] = []
    if payload.get("setup_missing_runtime"):
        notes.append("llama-server will be provisioned")
    if payload.get("setup_missing_model"):
        if model_source == "direct":
            notes.append("recommended GGUF will be downloaded")
        else:
            notes.append("model missing")
    if payload.get("setup_missing_api_key"):
        notes.append("API key env missing")
    note_suffix = f" [{', '.join(notes)}]" if notes else ""
    return (
        f"runtime={runtime}, model_source={model_source}, model={model_text}, "
        f"device={payload.get('device', 'auto')}, ctx={payload.get('context_window_tokens', '?')}, "
        f"gpu_layers={payload.get('gpu_layers', 0)}{note_suffix}"
    )


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
        return str(result).strip()
    return input(f"{prompt}{default}").strip()


def _choice_prompt_html(
    title: str,
    options: list[tuple[str, object]],
    *,
    selected_index: int,
    detail: str | None = None,
) -> HTML:
    lines = [f"<style fg='{ACCENT_GREEN}' bold='true'>{html_escape(title)}</style>"]
    if detail:
        lines.append(f"<style fg='ansibrightblack'>{html_escape(detail)}</style>")
    for idx, (label, _value) in enumerate(options):
        marker = "›" if idx == selected_index else " "
        style = f" fg='{ACCENT_GREEN}' bold='true'" if idx == selected_index else ""
        lines.append(
            f"{marker} <style{style}>{idx + 1}. {html_escape(str(label))}</style>"
        )
    return HTML("\n".join(lines) + "\n\nchoice> ")


async def _prompt_choice(
    session: PromptSession[object] | None,
    console: Console,
    title: str,
    options: list[tuple[str, object]],
    *,
    default_index: int = 0,
    detail: str | None = None,
) -> object:
    default_index = max(0, min(default_index, len(options) - 1))
    if session is not None:
        selected_index = default_index
        bindings = KeyBindings()

        def _set_index(next_index: int) -> None:
            nonlocal selected_index
            selected_index = next_index % len(options)

        @bindings.add("up")
        @bindings.add("c-p")
        def _prev_choice(event) -> None:
            _set_index(selected_index - 1)
            event.current_buffer.reset()
            event.app.invalidate()

        @bindings.add("down")
        @bindings.add("c-n")
        def _next_choice(event) -> None:
            _set_index(selected_index + 1)
            event.current_buffer.reset()
            event.app.invalidate()

        @bindings.add("pageup")
        def _first_choice(event) -> None:
            _set_index(0)
            event.current_buffer.reset()
            event.app.invalidate()

        @bindings.add("pagedown")
        def _last_choice(event) -> None:
            _set_index(len(options) - 1)
            event.current_buffer.reset()
            event.app.invalidate()

        @bindings.add("enter")
        def _submit_choice(event) -> None:
            event.app.exit(result=options[selected_index][1])

        result = await session.prompt_async(
            lambda: _choice_prompt_html(
                title,
                options,
                selected_index=selected_index,
                detail=detail,
            ),
            default="",
            key_bindings=bindings,
            complete_while_typing=False,
            enable_suspend=False,
        )
        raw = str(result).strip()
        if not raw:
            return options[selected_index][1]
        try:
            picked = int(raw) - 1
        except ValueError:
            return options[selected_index][1]
        if 0 <= picked < len(options):
            return options[picked][1]
        return options[selected_index][1]

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
    console.print(
        "[dim]Local setup can reuse an existing llama.cpp or Ollama install, "
        "or provision llama-server and a recommended model when the required tools and network access are available.[/]"
    )

    recommended_payload = build_recommended_payload(
        hardware_info=hardware_info,
        recommended_ctx=recommended_ctx,
        current_cfg=current_cfg,
    )
    setup_mode_options = [
        ("Use recommended setup", "recommended"),
        ("Review and edit recommended setup", "guided"),
        ("Manual setup", "manual"),
    ]
    setup_mode = str(
        await _prompt_choice(
            session,
            console,
            "Setup mode",
            setup_mode_options,
            default_index=0,
            detail=_recommended_summary(recommended_payload),
        )
    )
    if setup_mode == "recommended":
        return recommended_payload

    hardware_options = [
        (f"Use detected hardware ({hardware_info.label}, {ram_text})", "auto"),
        ("Pick hardware profile manually", "other"),
    ]
    default_hardware = "other" if setup_mode == "manual" and _current_string(current_cfg, "hardware_profile") == "other" else str(recommended_payload.get("hardware_profile", "auto"))
    hardware = await _prompt_choice(
        session,
        console,
        "Hardware profile",
        hardware_options,
        default_index=_default_option_index(
            hardware_options,
            default_hardware,
        ),
    )

    hardware_override = ""
    if hardware == "other":
        override_options = [(label, key) for key, label, _ram, _cuda in HARDWARE_OVERRIDE_OPTIONS]
        hardware_override = str(
            await _prompt_choice(
                session,
                console,
                "Hardware override",
                override_options,
                default_index=_default_option_index(
                    override_options,
                    _current_string(current_cfg, "hardware_override"),
                ),
            )
        )

    runtime_prompt_options = _runtime_prompt_options(
        current_cfg,
        llama_ready=_discover_llama_server() is not None,
    )
    runtime_default = _configured_runtime(current_cfg) if setup_mode == "manual" else str(recommended_payload.get("runtime", _configured_runtime(current_cfg)))
    runtime = str(
        await _prompt_choice(
            session,
            console,
            "Runtime",
            runtime_prompt_options,
            default_index=_default_option_index(
                runtime_prompt_options,
                runtime_default,
            ),
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
        direct = recommend_direct_model(effective_hw)
        direct_plan = "__direct__"
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
        model_plan_options.append((f"Download recommended GGUF: {direct['label']}", direct_plan))
        detail = (
            "Choose a local GGUF, an installed Ollama model, or let setup pull or download a recommended local model."
            if ollama_cli
            else "Choose a local GGUF or let setup download a recommended local GGUF."
        )
        current_model_source = _current_string(current_cfg, "model_source").lower()
        current_ollama_model = _current_string(current_cfg, "ollama_model")
        current_llama_model = _current_string(current_cfg, "llama_model")
        default_model_plan = "__local__"
        if setup_mode != "manual":
            recommended_source = str(recommended_payload.get("model_source") or "local")
            if recommended_source == "ollama":
                default_model_plan = str(recommended_payload.get("ollama_model") or current_ollama_model or "__local__")
            elif recommended_source == "direct":
                default_model_plan = direct_plan
        elif current_model_source == "ollama" and current_ollama_model:
            default_model_plan = current_ollama_model
        elif current_llama_model or model_files or saved_model_files:
            default_model_plan = "__local__"
        elif installed_ollama:
            default_model_plan = installed_ollama[0]
        elif download_rows:
            default_model_plan = str(download_rows[0][1])
        else:
            default_model_plan = direct_plan
        model_plan = "__local__"
        if len(model_plan_options) > 1:
            model_plan = str(
                await _prompt_choice(
                    session,
                    console,
                    "Model source",
                    model_plan_options,
                    default_index=_default_option_index(model_plan_options, default_model_plan),
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
            preferred_local_model = current_llama_model if current_llama_model else None
            if setup_mode != "manual":
                preferred_local_model = str(recommended_payload.get("llama_model") or preferred_local_model or "")
                preferred_local_model = preferred_local_model or None
            if preferred_local_model and preferred_local_model not in [value for _label, value in local_rows]:
                local_rows.insert(0, (f"{Path(preferred_local_model).name or preferred_local_model} (current)", preferred_local_model))
            local_choice = await _prompt_choice(
                session,
                console,
                "Local model",
                local_rows,
                default_index=_default_option_index(
                    local_rows,
                    preferred_local_model or (local_rows[0][1] if local_rows else "__manual__"),
                ),
                detail="Select a detected GGUF file or choose Manual path.",
            )
            if local_choice == "__manual__":
                model_path = await _prompt_text(session, "model path> ", default=current_llama_model)
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
        elif model_plan == direct_plan:
            payload["model_source"] = "direct"
            payload["model_download_url"] = str(direct["url"])
            payload["model_download_path"] = str(direct["target_path"])
            payload["recommended_llm"] = str(direct["label"])
            payload["setup_missing_model"] = True
        else:
            if not ollama_cli:
                raise RuntimeError("Ollama CLI not found.")
            if jetson_target:
                console.print("[dim]Jetson targets require quantized GGUF-backed Ollama models.[/]")
            payload["model_source"] = "ollama"
            payload["recommended_llm"] = model_plan
            payload["ollama_model"] = model_plan
    elif runtime == "openai_compatible":
        current_model_ref = _current_string(current_cfg, "openai_compatible_model")
        current_base_url = _current_string(current_cfg, "openai_compatible_base_url") or "https://api.openai.com"
        current_api_env = _current_string(current_cfg, "openai_compatible_api_key_env") or "OPENAI_API_KEY"
        if setup_mode != "manual":
            current_model_ref = str(recommended_payload.get("openai_compatible_model") or current_model_ref or "")
            current_base_url = str(recommended_payload.get("openai_compatible_base_url") or current_base_url)
            current_api_env = str(recommended_payload.get("openai_compatible_api_key_env") or current_api_env)
        model_ref = await _prompt_text(
            session,
            "model id> ",
            default=current_model_ref or _default_remote_model(runtime),
        )
        if not model_ref:
            raise RuntimeError("OpenAI-compatible runtime requires a model id.")
        base_url = await _prompt_text(
            session,
            "base url> ",
            default=current_base_url,
        )
        if not base_url:
            raise RuntimeError("OpenAI-compatible runtime requires a base URL.")
        api_key_env = await _prompt_text(
            session,
            "api key env> ",
            default=current_api_env,
        )
        payload["model_source"] = "remote"
        payload["model"] = model_ref
        payload["openai_compatible_model"] = model_ref
        payload["openai_compatible_base_url"] = base_url
        payload["openai_compatible_api_key_env"] = api_key_env or "OPENAI_API_KEY"
        payload["openai_compatible_verify_connection"] = False
    else:
        current_model_ref = _current_string(current_cfg, "openrouter_model")
        current_api_env = _current_string(current_cfg, "openrouter_api_key_env") or "OPENROUTER_API_KEY"
        if setup_mode != "manual":
            current_model_ref = str(recommended_payload.get("openrouter_model") or current_model_ref or "")
            current_api_env = str(recommended_payload.get("openrouter_api_key_env") or current_api_env)
        model_ref = await _prompt_text(
            session,
            "model id> ",
            default=current_model_ref or _default_remote_model(runtime),
        )
        if not model_ref:
            raise RuntimeError("OpenRouter runtime requires a model id.")
        api_key_env = await _prompt_text(
            session,
            "api key env> ",
            default=current_api_env,
        )
        payload["model_source"] = "remote"
        payload["model"] = model_ref
        payload["openrouter_model"] = model_ref
        payload["openrouter_base_url"] = "https://openrouter.ai/api/v1"
        payload["openrouter_api_key_env"] = api_key_env or "OPENROUTER_API_KEY"
        payload["openrouter_verify_connection"] = False

    headless = not bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    if hardware == "auto":
        recommended_ctx_value = recommended_context_window_tokens()
    else:
        recommended_ctx_value = recommended_context_window_tokens_from_total(
            effective_hw.total_ram_gb,
            headless=headless,
        )
    recommended_ctx_value = max(recommended_ctx, recommended_ctx_value)
    context_options = [
        (f"{value} (recommended)" if value == recommended_ctx_value else str(value), value)
        for value in context_window_options(recommended_ctx_value)
    ]
    context_default = _current_int(current_cfg, "context_window_tokens") or recommended_ctx_value
    if setup_mode != "manual":
        context_default = int(recommended_payload.get("context_window_tokens", context_default))
    context_value = int(
        await _prompt_choice(
            session,
            console,
            "Context window",
            context_options,
            default_index=_default_option_index(
                context_options,
                context_default,
            ),
        )
    )

    gpu_value = 0
    if runtime == "llama_cpp":
        recommended_gpu = recommended_gpu_layers(device, effective_hw.total_ram_gb)
        gpu_options = [
            (f"{value} (recommended)" if value == recommended_gpu else str(value), value)
            for value in gpu_layer_options(device, recommended_gpu)
        ]
        gpu_default = _current_int(current_cfg, "gpu_layers") if _current_int(current_cfg, "gpu_layers") is not None else recommended_gpu
        if setup_mode != "manual":
            gpu_default = int(recommended_payload.get("gpu_layers", gpu_default))
        gpu_value = int(
            await _prompt_choice(
                session,
                console,
                "GPU layers",
                gpu_options,
                default_index=_default_option_index(
                    gpu_options,
                    gpu_default,
                ),
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
        default=str(recommended_payload.get("model_profile_name") or default_profile_name(payload)),
    )
    return payload
