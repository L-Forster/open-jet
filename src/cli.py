"""CLI surface for open-jet with lazy TUI loading."""

from __future__ import annotations

import argparse
import importlib.metadata

from .airgap import airgapped_from_cfg
from .config import load_config
from .model_profiles import list_model_profiles
from .runtime_registry import active_model_ref, runtime_spec
from .surfaces import launch_tui
from .surfaces.command_specs import COMMANDS


def _active_profile_name(cfg: dict[str, object]) -> str:
    return str(cfg.get("active_model_profile") or "").strip() or "none"


def _format_model_profiles_summary(cfg: dict[str, object]) -> str:
    profiles = list_model_profiles(cfg)
    if not profiles:
        return "No saved model presets yet. Run `open-jet setup` or `/setup` to add one."

    active_name = _active_profile_name(cfg).lower()
    lines = [f"Active model preset: {_active_profile_name(cfg)}"]
    for profile in profiles:
        runtime = str(profile.get("runtime", "llama_cpp"))
        model_ref = active_model_ref(dict(profile)) or "n/a"
        marker = " (active)" if str(profile["name"]).strip().lower() == active_name else ""
        lines.append(
            f"- {profile['name']}{marker}: runtime={runtime} "
            f"context={profile.get('context_window_tokens', 'n/a')} "
            f"gpu={profile.get('gpu_layers', 'n/a')} "
            f"model={model_ref}"
        )
    return "\n".join(lines)


def _format_cli_status(cfg: dict[str, object]) -> str:
    runtime = str(cfg.get("runtime", "llama_cpp"))
    spec = runtime_spec(runtime)
    model_ref = active_model_ref(cfg) or "n/a"
    lines = [
        f"Runtime: {spec.label} ({runtime})",
        f"Active model preset: {_active_profile_name(cfg)}",
        f"Model ref: {model_ref}",
        f"Context window: {cfg.get('context_window_tokens', 'n/a')}",
        f"GPU layers: {cfg.get('gpu_layers', 'n/a')}",
        f"Air-gapped: {'true' if airgapped_from_cfg(cfg) else 'false'}",
    ]
    return "\n".join(lines)


def _format_slash_commands_summary() -> str:
    lines = ["Slash commands:"]
    for spec in COMMANDS:
        aliases = f" (aliases: {', '.join(f'/{alias}' for alias in spec.aliases)})" if spec.aliases else ""
        lines.append(f"- /{spec.name}: {spec.description}{aliases}")
    return "\n".join(lines)


def _open_jet_version() -> str:
    try:
        return importlib.metadata.version("open-jet")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="open-jet offline agentic terminal UI")
    parser.add_argument("--setup", action="store_true", help="start in setup wizard mode before launching the chat UI")
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("chat", help="start the interactive chat UI")
    subparsers.add_parser("setup", help="run setup wizard before launching the chat UI")
    subparsers.add_parser("models", help="list saved model presets")
    subparsers.add_parser("commands", help="list available slash commands")
    subparsers.add_parser("status", help="show runtime and configuration status")
    subparsers.add_parser("version", help="show version information")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    if args.command == "models":
        print(_format_model_profiles_summary(load_config()))
        return
    if args.command == "commands":
        print(_format_slash_commands_summary())
        return
    if args.command == "status":
        print(_format_cli_status(load_config()))
        return
    if args.command == "version":
        print(f"open-jet {_open_jet_version()}")
        return

    force_setup = bool(args.setup or args.command == "setup")
    launch_tui(force_setup=force_setup)
