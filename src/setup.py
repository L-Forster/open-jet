from __future__ import annotations

import os
import shutil
from html import escape as html_escape
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

from rich.console import Console
from rich.markup import escape

from .config import HARDWARE_OVERRIDE_OPTIONS, setup_direct_model_catalog
from .hardware import (
    HardwareInfo,
    effective_hardware_info,
    recommended_context_window_tokens,
    recommended_context_window_tokens_from_total,
    recommended_device_for_hardware,
    recommended_gpu_layers,
)
from .model_profiles import default_profile_name
from .provisioning import MODELS_DIR, recommend_direct_model
from .setup_memory import recommend_context_window_for_model, recommend_setup_context_window

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


def _discover_llama_server() -> str | None:
    found = shutil.which("llama-server")
    if found:
        return found
    candidate = Path.home() / "llama.cpp" / "build" / "bin" / "llama-server"
    if candidate.is_file():
        return str(candidate)
    return None


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
    accent_color: str = ACCENT_GREEN,
) -> Any:
    from prompt_toolkit.formatted_text import HTML

    lines = [f"<style fg='{accent_color}' bold='true'>{html_escape(title)}</style>"]
    if detail:
        lines.append(f"<style fg='ansibrightblack'>{html_escape(detail)}</style>")
    for idx, (label, _value) in enumerate(options):
        marker = "›" if idx == selected_index else " "
        style = f" fg='{accent_color}' bold='true'" if idx == selected_index else ""
        lines.append(f"{marker} <style{style}>{idx + 1}. {html_escape(str(label))}</style>")
    return HTML("\n".join(lines) + "\n\nchoice> ")


async def _prompt_choice(
    session: PromptSession[object] | None,
    console: Console,
    title: str,
    options: list[tuple[str, object]],
    *,
    default_index: int = 0,
    detail: str | None = None,
    accent_color: str = ACCENT_GREEN,
) -> object:
    from prompt_toolkit.key_binding import KeyBindings

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
                accent_color=accent_color,
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

    console.print(f"[bold {accent_color}]{escape(title)}[/]")
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


def _default_option_index(options: list[tuple[str, object]], value: object, *, fallback: int = 0) -> int:
    for idx, (_label, option_value) in enumerate(options):
        if option_value == value:
            return idx
    return fallback


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
    active_ref = _current_string(current_cfg, "llama_model")
    if active_ref:
        refs.insert(0, active_ref)
    return _dedupe_refs(refs)


def _setup_model_refs(payload: Mapping[str, object] | None) -> list[str]:
    if not isinstance(payload, Mapping):
        return []
    refs = [
        payload.get("llama_model"),
        payload.get("model_download_path"),
    ]
    unique: list[str] = []
    seen: set[str] = set()
    for ref in refs:
        value = str(ref or "").strip()
        if not value or value in seen:
            continue
        seen.add(value)
        unique.append(value)
    return unique


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


def _runtime_prompt_options(
    current_cfg: Mapping[str, object] | None,
    *,
    llama_ready: bool,
) -> list[tuple[str, str]]:
    suffix = " (recommended)" if llama_ready else " (recommended, setup can provision llama-server)"
    return [(f"Local model: llama.cpp (GGUF){suffix}", "llama_cpp")]


def _recommended_summary(payload: Mapping[str, object]) -> str:
    model_source = str(payload.get("model_source") or "local")
    model_ref = str(payload.get("llama_model") or "").strip()
    model_text = model_ref or "missing"
    notes: list[str] = []
    if payload.get("setup_missing_runtime"):
        notes.append("llama-server will be provisioned")
    if payload.get("setup_missing_model"):
        notes.append("recommended GGUF will be downloaded")
    note_suffix = f" [{', '.join(notes)}]" if notes else ""
    return (
        f"runtime=llama_cpp, model_source={model_source}, model={model_text}, "
        f"device={payload.get('device', 'auto')}, ctx={payload.get('context_window_tokens', '?')}, "
        f"gpu_layers={payload.get('gpu_layers', 0)}{note_suffix}"
    )


def _recommended_local_payload(
    *,
    current_cfg: Mapping[str, object] | None,
    model_files: list[str],
    saved_model_files: list[str],
    direct: Mapping[str, str],
) -> dict[str, object]:
    current_model = _current_string(current_cfg, "llama_model")
    if current_model and Path(current_model).expanduser().is_file():
        model_path = str(Path(current_model).expanduser())
        return {
            "model_source": "local",
            "llama_model": model_path,
        }
    if model_files:
        model_path = str(Path(model_files[0]).expanduser())
        return {
            "model_source": "local",
            "llama_model": model_path,
        }
    existing_saved = [path for path in saved_model_files if Path(path).expanduser().is_file()]
    if existing_saved:
        model_path = str(Path(existing_saved[0]).expanduser())
        return {
            "model_source": "local",
            "llama_model": model_path,
        }
    return {
        "model_source": "direct",
        "model_download_url": str(direct["url"]),
        "model_download_path": str(direct["target_path"]),
        "setup_missing_model": True,
    }


def _direct_catalog_payload(row: Mapping[str, object]) -> dict[str, object]:
    filename = str(row["filename"])
    payload: dict[str, object] = {
        "model_source": "direct",
        "model_download_url": str(row["url"]),
        "model_download_path": str(MODELS_DIR / filename),
        "setup_missing_model": True,
        "model_profile_name": str(row.get("label") or "").strip() or Path(filename).stem,
    }
    for key in ("model_size_mb", "kv_bytes_per_token", "resident_model_size_mb", "active_model_size_mb"):
        if row.get(key) is not None:
            payload[key] = row[key]
    if "unified_memory_only" in row:
        payload["unified_memory_only"] = bool(row.get("unified_memory_only"))
    if bool(row.get("unified_memory_only")) and row.get("active_model_size_mb") is not None:
        payload["llama_cpu_moe"] = False
        payload["llama_n_cpu_moe"] = 0
    return payload


def _direct_catalog_label(row: Mapping[str, object], recommended_url: str) -> str:
    label = str(row.get("label") or row.get("filename") or "Model")
    filename = str(row.get("filename") or "").strip()
    max_ram = float(row.get("max_ram_gb") or 0.0)
    size_mb = float(row.get("model_size_mb") or 0.0)
    details: list[str] = []
    if max_ram > 0:
        details.append(f"{max_ram:g}GB tier")
    if size_mb > 0:
        details.append(f"{size_mb / 1024.0:.1f}GB download")
    if bool(row.get("unified_memory_only")):
        details.append("unified memory")
    if str(row.get("url") or "") == recommended_url:
        details.append("recommended")
    suffix = f" ({', '.join(details)})" if details else ""
    if filename and filename != label:
        return f"{label}: {filename}{suffix}"
    return f"{label}{suffix}"


def _recommended_context_for_payload(
    payload: Mapping[str, object],
    *,
    device: str,
    fallback_tokens: int,
    total_vram_mb: float,
) -> int:
    if str(payload.get("model_source") or "") == "direct":
        try:
            model_size_mb = float(payload.get("model_size_mb") or 0.0)
            kv_bytes_per_token = float(payload.get("kv_bytes_per_token") or 0.0)
        except (TypeError, ValueError):
            model_size_mb = 0.0
            kv_bytes_per_token = 0.0
        for resident_key in ("resident_model_size_mb", "active_model_size_mb"):
            try:
                resident_mb = float(payload.get(resident_key) or 0.0)
            except (TypeError, ValueError):
                resident_mb = 0.0
            if resident_mb > 0:
                model_size_mb = resident_mb
                break
        if model_size_mb > 0 and kv_bytes_per_token > 0:
            return recommend_context_window_for_model(
                device=device,
                fallback_tokens=fallback_tokens,
                model_size_mb=model_size_mb,
                kv_bytes_per_token=kv_bytes_per_token,
                total_vram_mb=total_vram_mb,
            )
    return recommend_setup_context_window(
        runtime="llama_cpp",
        device=device,
        fallback_tokens=fallback_tokens,
        model_refs=_setup_model_refs(payload),
        total_vram_mb=total_vram_mb,
    )


def _payload_matches_current_model(
    payload: Mapping[str, object],
    current_cfg: Mapping[str, object] | None,
) -> bool:
    if not isinstance(current_cfg, Mapping):
        return False
    return (
        str(payload.get("model_source") or "") == str(current_cfg.get("model_source") or "")
        and str(payload.get("llama_model") or "") == str(current_cfg.get("llama_model") or "")
        and str(payload.get("model_download_url") or "") == str(current_cfg.get("model_download_url") or "")
    )


def build_recommended_payload(
    *,
    hardware_info: HardwareInfo,
    recommended_ctx: int,
    current_cfg: Mapping[str, object] | None = None,
) -> dict[str, object]:
    hardware_profile = "other" if _current_string(current_cfg, "hardware_profile") == "other" else "auto"
    hardware_override = _current_string(current_cfg, "hardware_override") if hardware_profile == "other" else ""
    effective_hw = effective_hardware_info(hardware_profile, hardware_info, hardware_override)
    device = recommended_device_for_hardware(hardware_profile, hardware_info, hardware_override)

    payload: dict[str, object] = {
        "hardware_profile": hardware_profile,
        "hardware_override": hardware_override,
        "device": device,
        "setup_complete": True,
    }

    direct = recommend_direct_model(effective_hw, cfg=current_cfg)
    payload.update(
        _recommended_local_payload(
            current_cfg=current_cfg,
            model_files=discover_model_files(),
            saved_model_files=_saved_model_refs(current_cfg, "llama_cpp"),
            direct=direct,
        )
    )

    llama_model = str(payload.get("llama_model") or "").strip()
    if llama_model:
        _remember_model_ref(payload, current_cfg, "llama_cpp", llama_model)

    payload["setup_missing_runtime"] = _discover_llama_server() is None

    headless = not bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    fallback_ctx = (
        int(recommended_ctx)
        if recommended_ctx > 0
        else (
            recommended_context_window_tokens()
            if hardware_profile == "auto"
            else recommended_context_window_tokens_from_total(
                effective_hw.total_ram_gb,
                headless=headless,
            )
        )
    )
    payload["context_window_tokens"] = _recommended_context_for_payload(
        payload,
        device=device,
        fallback_tokens=fallback_ctx,
        total_vram_mb=effective_hw.vram_mb,
    )
    payload["gpu_layers"] = (
        _current_int(current_cfg, "gpu_layers")
        if _current_int(current_cfg, "gpu_layers") is not None
        else recommended_gpu_layers(device, effective_hw.total_ram_gb)
    )
    payload["model_profile_name"] = default_profile_name(payload)
    return payload


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
    console.print("[dim]Setup is local-only: pick or download a GGUF and compile llama.cpp when needed.[/]")

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
    default_hardware = (
        "other"
        if setup_mode == "manual" and _current_string(current_cfg, "hardware_profile") == "other"
        else str(recommended_payload.get("hardware_profile", "auto"))
    )
    hardware = str(
        await _prompt_choice(
            session,
            console,
            "Hardware profile",
            hardware_options,
            default_index=_default_option_index(hardware_options, default_hardware),
        )
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

    payload: dict[str, object] = {
        "hardware_profile": hardware,
        "hardware_override": hardware_override if hardware == "other" else "",
    }

    effective_hw = effective_hardware_info(hardware, hardware_info, hardware_override)
    device = recommended_device_for_hardware(hardware, hardware_info, hardware_override)
    model_files = discover_model_files()
    saved_model_files = _saved_model_refs(current_cfg, "llama_cpp")
    direct = recommend_direct_model(effective_hw, cfg=current_cfg)
    direct_catalog = list(setup_direct_model_catalog(current_cfg))
    model_plan_options: list[tuple[str, object]] = [
        ("Use a local .gguf model file", "__local__"),
        ("Download a GGUF from the model catalog", "__direct__"),
    ]
    detail = f"Choose a local GGUF or download a catalog model. Recommended: {direct['label']}."
    current_llama_model = _current_string(current_cfg, "llama_model")
    default_model_plan = "__local__"
    if setup_mode != "manual":
        default_model_plan = "__direct__" if recommended_payload.get("model_source") == "direct" else "__local__"
    elif not (current_llama_model or model_files or saved_model_files):
        default_model_plan = "__direct__"

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
        preferred_local_model = current_llama_model or None
        if setup_mode != "manual":
            preferred_local_model = str(recommended_payload.get("llama_model") or preferred_local_model or "")
            preferred_local_model = preferred_local_model or None
        option_values = [value for _label, value in local_rows]
        if preferred_local_model and preferred_local_model not in option_values:
            local_rows.insert(
                0,
                (f"{Path(preferred_local_model).name or preferred_local_model} (current)", preferred_local_model),
            )
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
        payload["llama_model"] = str(model_file)
        _remember_model_ref(payload, current_cfg, "llama_cpp", str(model_file))
    else:
        recommended_url = str(direct["url"])
        catalog_options = [
            (_direct_catalog_label(row, recommended_url), row)
            for row in direct_catalog
        ]
        selected_direct = await _prompt_choice(
            session,
            console,
            "Model download",
            catalog_options,
            default_index=next(
                (
                    index
                    for index, (_label, row) in enumerate(catalog_options)
                    if str(row.get("url") or "") == recommended_url
                ),
                0,
            ),
            detail="Select any configured GGUF catalog entry to download.",
        )
        if not isinstance(selected_direct, Mapping):
            raise RuntimeError("Invalid model catalog selection.")
        payload.update(_direct_catalog_payload(selected_direct))

    headless = not bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    fallback_ctx = (
        int(recommended_ctx)
        if recommended_ctx > 0 and hardware == "auto"
        else (
            recommended_context_window_tokens()
            if hardware == "auto"
            else recommended_context_window_tokens_from_total(
                effective_hw.total_ram_gb,
                headless=headless,
            )
        )
    )
    recommended_ctx_value = _recommended_context_for_payload(
        payload,
        device=device,
        fallback_tokens=fallback_ctx,
        total_vram_mb=effective_hw.vram_mb,
    )
    current_model_matches_payload = _payload_matches_current_model(payload, current_cfg)
    context_default = (
        _current_int(current_cfg, "context_window_tokens")
        if current_model_matches_payload and _current_int(current_cfg, "context_window_tokens") is not None
        else recommended_ctx_value
    )
    model_matches_recommendation = (
        payload.get("model_source") == recommended_payload.get("model_source")
        and str(payload.get("llama_model") or "") == str(recommended_payload.get("llama_model") or "")
        and str(payload.get("model_download_url") or "") == str(recommended_payload.get("model_download_url") or "")
    )
    if setup_mode != "manual" and model_matches_recommendation:
        context_default = int(recommended_payload.get("context_window_tokens", context_default))
    context_values = context_window_options(recommended_ctx_value)
    if context_default not in context_values:
        context_values = sorted(set([*context_values, context_default]))
    context_options = [
        (f"{value} (recommended)" if value == recommended_ctx_value else str(value), value)
        for value in context_values
    ]
    context_value = int(
        await _prompt_choice(
            session,
            console,
            "Context window",
            context_options,
            default_index=_default_option_index(context_options, context_default),
        )
    )

    recommended_gpu = recommended_gpu_layers(device, effective_hw.total_ram_gb)
    gpu_default = (
        _current_int(current_cfg, "gpu_layers")
        if current_model_matches_payload and _current_int(current_cfg, "gpu_layers") is not None
        else recommended_gpu
    )
    if setup_mode != "manual" and model_matches_recommendation:
        gpu_default = int(recommended_payload.get("gpu_layers", gpu_default))
    gpu_values = gpu_layer_options(device, recommended_gpu)
    if gpu_default not in gpu_values:
        gpu_values = sorted(set([*gpu_values, gpu_default]))
    gpu_options = [
        (f"{value} (recommended)" if value == recommended_gpu else str(value), value)
        for value in gpu_values
    ]
    gpu_value = int(
        await _prompt_choice(
            session,
            console,
            "GPU layers",
            gpu_options,
            default_index=_default_option_index(gpu_options, gpu_default),
            detail="Higher offload can be faster but uses more memory.",
        )
    )

    payload.update(
        {
            "device": device,
            "context_window_tokens": context_value,
            "gpu_layers": gpu_value,
            "setup_complete": True,
            "setup_missing_runtime": _discover_llama_server() is None,
        }
    )
    payload["model_profile_name"] = await _prompt_text(
        session,
        "model name> ",
        default=str(payload.get("model_profile_name") or default_profile_name(payload)),
    )
    return payload
