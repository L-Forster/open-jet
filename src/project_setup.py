"""Provisioning flow for `open-jet project`.

Distinct from `open-jet setup` in what it asks and what it optimises for. Setup profiles
the machine it runs on and picks the strongest coding model that machine can hold.
Project provisioning is a build-time step for an application that embeds the SDK: the
target device is declared rather than detected, the budget is a slice the host
application concedes rather than all available headroom, and selection is driven by use
case and latency.

Downloading happens here and only here. Nothing in the SDK fetches at runtime.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Any, Mapping

from .app_paths import project_models_dir, project_openjet_root
from .config import save_project_config
from .embed_catalog import (
    DEFAULT_EMBED_CONTEXT_TOKENS,
    EMBED_USE_CASES,
    TARGET_DEVICE_PRESETS,
    describe_row,
    recommend_embed_model,
    rejection_reasons,
    resident_mb,
    target_preset,
    use_case,
)
from .provisioning import ensure_direct_model

_MARKUP = re.compile(r"\[/?[^\[\]]*\]")


class _ConsoleLog:
    """`ensure_direct_model` writes Rich markup; this renders it to a plain terminal."""

    def write(self, text: str) -> None:
        print(_MARKUP.sub("", str(text)))


def _status_width() -> int:
    return max(20, shutil.get_terminal_size(fallback=(100, 24)).columns - 4)


def _set_status(text: str) -> None:
    """Render one overwriting progress line.

    `ensure_direct_model` composes status text for a TUI widget, so it arrives already
    truncated and can carry fragments of a path. Clear the line to its full width before
    writing, and cap to the real terminal, so nothing from the previous frame survives.
    """
    width = _status_width()
    line = " ".join(str(text).split())[:width]
    print("\r" + " " * (width + 2) + "\r  " + line, end="", flush=True)


def _clear_status() -> None:
    print("\r" + " " * (_status_width() + 2) + "\r", end="", flush=True)


def _choose(prompt: str, rows: tuple[dict[str, object], ...], detail_key: str | None = None) -> dict[str, object]:
    print(f"\n{prompt}")
    for index, row in enumerate(rows, start=1):
        detail = f" — {row[detail_key]}" if detail_key and row.get(detail_key) else ""
        print(f"  {index}. {row['label']}{detail}")
    while True:
        raw = input(f"Select 1-{len(rows)}: ").strip()
        if raw.isdigit() and 1 <= int(raw) <= len(rows):
            return dict(rows[int(raw) - 1])
        print("  Enter one of the listed numbers.")


def _ask_float(prompt: str, default: float) -> float:
    raw = input(f"{prompt} [{default:g}]: ").strip()
    if not raw:
        return default
    return float(raw)


def _ask_int(prompt: str, default: int) -> int:
    raw = input(f"{prompt} [{default}]: ").strip()
    if not raw:
        return default
    return int(raw)


def _gitignore_project_dir(root: Path) -> None:
    """Self-ignore .openjet so weights and machine-local config never reach the repo."""
    marker = project_openjet_root(root) / ".gitignore"
    marker.parent.mkdir(parents=True, exist_ok=True)
    if not marker.exists():
        marker.write_text("*\n")


def resolve_selection(
    *,
    use_case_id: str,
    target_id: str | None,
    budget_gb: float,
    context_tokens: int,
) -> dict[str, Any]:
    row = recommend_embed_model(
        use_case_id=use_case_id,
        budget_gb=budget_gb,
        context_tokens=context_tokens,
    )
    if row is None:
        case = use_case(use_case_id)
        lines = [
            f"No catalog model satisfies {case['label']} within {budget_gb:.1f}GB at {context_tokens} ctx.",
            "",
            *(f"  - {reason}" for reason in rejection_reasons(
                use_case_id=use_case_id,
                budget_gb=budget_gb,
                context_tokens=context_tokens,
            )),
            "",
            "Raise the budget, shorten the context, or choose a use case with a looser latency target.",
        ]
        raise RuntimeError("\n".join(lines))
    selected = dict(row)
    selected["use_case"] = use_case_id
    selected["target"] = target_id
    selected["budget_gb"] = budget_gb
    selected["context_tokens"] = context_tokens
    return selected


def build_overlay(selection: Mapping[str, Any], model_path: Path) -> dict[str, Any]:
    overlay: dict[str, Any] = {
        "project": {
            "model_id": selection["id"],
            "use_case": selection["use_case"],
            "target": selection["target"],
            "budget_gb": selection["budget_gb"],
        },
        "model_source": "direct",
        "llama_model": str(model_path),
        "context_window_tokens": int(selection["context_tokens"]),
        "model_download_url": selection["url"],
        "model_download_path": str(model_path),
        "model_size_mb": selection["model_size_mb"],
        "filename": selection["filename"],
    }
    for flag in ("llama_mtp", "llama_cpu_moe", "llama_n_cpu_moe"):
        if flag in selection:
            overlay[flag] = selection[flag]
    return overlay


async def provision_project(
    root: Path | None = None,
    *,
    use_case_id: str | None = None,
    target_id: str | None = None,
    budget_gb: float | None = None,
    context_tokens: int | None = None,
) -> Path:
    project_root = Path(root or Path.cwd()).expanduser().resolve()

    print(f"Provisioning a model for {project_root}")
    print("This runs once, at build time. The SDK never downloads at runtime.")

    if use_case_id is None:
        use_case_id = str(_choose("What is the model for?", EMBED_USE_CASES, detail_key="detail")["id"])
    case = use_case(use_case_id)

    if target_id is None:
        print("\nThe target device is where your application runs, not this machine.")
        target_id = str(_choose("Which device are you shipping to?", TARGET_DEVICE_PRESETS)["id"])
    target = target_preset(target_id)

    if budget_gb is None:
        print(
            f"\n{target['label']} has ~{target['total_gb']:g}GB. Your application owns the rest of it;"
            "\nonly the slice you name here goes to the model."
        )
        budget_gb = _ask_float("Memory budget for the model in GB", float(target["budget_gb"]))

    if context_tokens is None:
        context_tokens = _ask_int("Context window in tokens", DEFAULT_EMBED_CONTEXT_TOKENS)

    selection = resolve_selection(
        use_case_id=use_case_id,
        target_id=target_id,
        budget_gb=budget_gb,
        context_tokens=context_tokens,
    )

    print(f"\nSelected: {describe_row(selection, context_tokens)}")
    print(f"  use case:  {case['label']} (first token must stay under {case['max_ttft_ms']}ms)")
    print(f"  target:    {target['label']}")
    print(f"  budget:    {resident_mb(selection, context_tokens) / 1024.0:.1f}GB of {budget_gb:.1f}GB")
    print("  latency figures are estimates — run `openjet benchmark` on real target hardware.")

    models_dir = project_models_dir(project_root)
    models_dir.mkdir(parents=True, exist_ok=True)
    _gitignore_project_dir(project_root)
    model_path = models_dir / str(selection["filename"])

    setup_result = {
        "model_source": "direct",
        "model_download_url": selection["url"],
        "model_download_path": str(model_path),
        "model_size_mb": selection["model_size_mb"],
    }
    print()
    resolved = await ensure_direct_model(
        setup_result,
        log=_ConsoleLog(),
        set_status=_set_status,
        clear_status=_clear_status,
    )

    final_path = Path(str(resolved.get("llama_model") or model_path))
    config_path = save_project_config(build_overlay(selection, final_path), project_root)

    print(f"\nModel:  {final_path}")
    print(f"Config: {config_path}")
    print("\nThe model lives in the project so your build can bundle it. Both are gitignored.")
    print("`open-jet` from anywhere in this project now uses it, and so does OpenJetSession.")
    return config_path
