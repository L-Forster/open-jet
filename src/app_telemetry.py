"""Telemetry, tracing, and shell command classification helpers.

Extracted from app.py to reduce its size. These are pure functions and
static helpers used by OpenJetApp for telemetry and diagnostics.
"""

from __future__ import annotations

import re
import shlex
import shutil
from pathlib import Path
from typing import Any

from .hardware import detect_hardware_info, effective_hardware_info


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

    from .harness import shell_command_is_verification

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
