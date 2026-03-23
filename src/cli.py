"""CLI surface for open-jet with lazy TUI loading."""

from __future__ import annotations

import argparse
import importlib.metadata

from .airgap import airgapped_from_cfg
from .config import load_config, save_config
from .device_sources import assign_device_alias, list_device_sources, set_device_enabled, sync_devices_registry
from .model_profiles import list_model_profiles, sync_active_model_profile
from .peripherals.system import device_discovery_hint
from .runtime_registry import active_model_ref, runtime_spec
from .self_update import update_from_latest_release
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
        if spec.hidden:
            continue
        aliases = f" (aliases: {', '.join(f'/{alias}' for alias in spec.aliases)})" if spec.aliases else ""
        lines.append(f"- /{spec.name}: {spec.description}{aliases}")
    return "\n".join(lines)


def _open_jet_version() -> str:
    try:
        return importlib.metadata.version("open-jet")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _update_context_window(cfg: dict[str, object], context_tokens: int) -> str:
    if context_tokens < 512:
        raise ValueError("Context window must be at least 512 tokens.")
    updated_cfg = dict(cfg)
    updated_cfg["context_window_tokens"] = context_tokens
    sync_active_model_profile(updated_cfg)
    save_config(updated_cfg)
    return (
        f"Context window set to {context_tokens} tokens for active config"
        if not str(updated_cfg.get("active_model_profile") or "").strip()
        else f"Context window set to {context_tokens} tokens for preset {updated_cfg['active_model_profile']}"
    )


def _format_device_list(cfg: dict[str, object]) -> str:
    registry_path = sync_devices_registry(cfg)
    sources = list_device_sources(cfg)
    lines = [f"Device registry: {registry_path}"]
    if not sources:
        lines.append("No devices detected.")
        hint = device_discovery_hint()
        if hint:
            lines.append(hint)
        lines.append("When devices are visible to Linux, run `open-jet device list` again, then `open-jet device add <existing_id> <new_id>` if you want a stable chat id.")
        return "\n".join(lines)

    lines.append("Discovered devices:")
    for source in sources:
        refs = ", ".join(f"@{ref}" for ref in source.refs)
        lines.append(
            f"- {source.primary_ref}: {source.device.label} | tag=@{source.primary_ref} | "
            f"kind={source.device.kind.value} | transport={source.device.transport.value} | "
            f"state={'enabled' if source.enabled else 'disabled'} | refs={refs}"
        )
    lines.append("Use the current id on the left as `<existing_id>` in `open-jet device add <existing_id> <new_id>`.")
    lines.append("The TUI also accepts `/device ...`, but persistent device setup is usually clearer from the CLI.")
    return "\n".join(lines)


def _add_device_id(cfg: dict[str, object], existing_id: str, new_id: str) -> str:
    source = assign_device_alias(cfg, reference=existing_id, alias=new_id)
    save_config(cfg)
    registry_path = sync_devices_registry(cfg)
    return (
        f"Saved device id {source.primary_ref} for {source.device.label}.\n"
        f"Device registry: {registry_path}\n"
        f"Use @{source.primary_ref} in chat to reference this device."
    )


def _set_device_state(cfg: dict[str, object], existing_id: str, *, enabled: bool) -> str:
    source = set_device_enabled(cfg, reference=existing_id, enabled=enabled)
    save_config(cfg)
    registry_path = sync_devices_registry(cfg)
    state = "enabled" if source.enabled else "disabled"
    return f"Device {source.primary_ref} is now {state}.\nDevice registry: {registry_path}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="open-jet offline agentic terminal UI",
        epilog=(
            "Useful commands: `open-jet commands` for slash commands, `open-jet status` for runtime/config status, "
            "and `open-jet device list` for persistent device setup. In the TUI, use `/device` and `@device_id` tags "
            "for device inputs. See docs/usage/device-sources.md."
        ),
    )
    parser.add_argument("--setup", action="store_true", help="start in setup wizard mode before launching the chat UI")
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("chat", help="start the interactive chat UI")
    subparsers.add_parser("setup", help="run setup wizard before launching the chat UI")
    subparsers.add_parser("models", help="list saved model presets")
    subparsers.add_parser("commands", help="list available slash commands")
    subparsers.add_parser("status", help="show runtime and configuration status")
    subparsers.add_parser("version", help="show version information")
    subparsers.add_parser("update", help="install the latest GitHub release into the current environment")
    context_parser = subparsers.add_parser("context", help="set the configured context window token count")
    context_parser.add_argument("tokens", type=int, help="new context window token count")
    device_parser = subparsers.add_parser("device", aliases=("devices",), help="list and configure persistent device ids")
    device_subparsers = device_parser.add_subparsers(dest="device_action")
    device_subparsers.add_parser("list", help="list discovered devices and current ids")
    device_add_parser = device_subparsers.add_parser("add", help="assign a stable id to an existing device")
    device_add_parser.add_argument("existing_id", help="current device id from `open-jet device list`")
    device_add_parser.add_argument("new_id", help="new stable device id to use in chat")
    device_on_parser = device_subparsers.add_parser("on", help="enable a device")
    device_on_parser.add_argument("device_id", help="current device id")
    device_off_parser = device_subparsers.add_parser("off", help="disable a device")
    device_off_parser.add_argument("device_id", help="current device id")
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
    if args.command == "context":
        try:
            print(_update_context_window(load_config(), int(args.tokens)))
        except ValueError as exc:
            raise SystemExit(str(exc))
        return
    if args.command in {"device", "devices"}:
        cfg = load_config()
        action = str(getattr(args, "device_action", "") or "list").strip().lower()
        try:
            if action == "list":
                print(_format_device_list(cfg))
                return
            if action == "add":
                print(_add_device_id(cfg, str(args.existing_id), str(args.new_id)))
                return
            if action == "on":
                print(_set_device_state(cfg, str(args.device_id), enabled=True))
                return
            if action == "off":
                print(_set_device_state(cfg, str(args.device_id), enabled=False))
                return
        except ValueError as exc:
            raise SystemExit(str(exc))
        raise SystemExit("Usage: open-jet device [list|add <existing_id> <new_id>|on <id>|off <id>]")
    if args.command == "version":
        print(f"open-jet {_open_jet_version()}")
        return
    if args.command == "update":
        try:
            print(update_from_latest_release(current_version=_open_jet_version()))
        except RuntimeError as exc:
            raise SystemExit(str(exc))
        return

    force_setup = bool(args.setup or args.command == "setup")
    launch_tui(force_setup=force_setup)
