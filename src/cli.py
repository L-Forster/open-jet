"""CLI surface for open-jet with lazy TUI loading."""

from __future__ import annotations

import argparse
import asyncio
import importlib.metadata
from pathlib import Path

from .airgap import airgapped_from_cfg
from .config import load_config, save_config
from .device_sources import assign_device_alias, list_device_sources, set_device_enabled, sync_devices_registry
from .model_profiles import list_model_profiles, sync_active_model_profile
from .peripherals.system import device_discovery_hint
from .runtime_registry import RUNTIME_LABEL, active_model_ref
from .self_update import update_from_latest_release
from .surfaces import launch_tui
from .surfaces.command_specs import COMMANDS
from .workflows import (
    WorkflowStatus,
    discover_workflow_index,
    load_workflow_spec,
    run_workflow,
    run_workflow_daemon,
    start_workflow_daemon,
    stop_workflow_daemon,
    validate_workflow_device_ids,
    write_workflow_run_report,
)
from .workflows.state import (
    list_workflow_statuses,
    load_workflow_status,
    save_workflow_assignment,
    save_workflow_status,
    workflow_last_run_path,
    workflow_runs_dir,
)


def _active_profile_name(cfg: dict[str, object]) -> str:
    return str(cfg.get("active_model_profile") or "").strip() or "none"


def _format_model_profiles_summary(cfg: dict[str, object]) -> str:
    profiles = list_model_profiles(cfg)
    if not profiles:
        return "No saved model presets yet. Run `open-jet setup` or `/setup` to add one."

    active_name = _active_profile_name(cfg).lower()
    lines = [f"Active model preset: {_active_profile_name(cfg)}"]
    for profile in profiles:
        model_ref = active_model_ref(dict(profile)) or "n/a"
        marker = " (active)" if str(profile["name"]).strip().lower() == active_name else ""
        lines.append(
            f"- {profile['name']}{marker}: "
            f"context={profile.get('context_window_tokens', 'n/a')} "
            f"gpu={profile.get('gpu_layers', 'n/a')} "
            f"model={model_ref}"
        )
    return "\n".join(lines)


def _format_cli_status(cfg: dict[str, object]) -> str:
    model_ref = active_model_ref(cfg) or "n/a"
    lines = [
        f"Runtime: {RUNTIME_LABEL}",
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


def _workflow_root() -> Path:
    return Path.cwd().resolve()


def discover_workflow_specs(root: Path):
    return discover_workflow_index(root)[0]


def discover_workflow_issues(root: Path):
    return discover_workflow_index(root)[1]


def _format_workflow_list(root: Path) -> str:
    specs = discover_workflow_specs(root)
    issues = discover_workflow_issues(root)
    if not specs:
        if not issues:
            return "No workflows discovered. Add Markdown files under `workflows/` or `.openjet/workflows/`."
        lines = ["No valid workflows discovered."]
        lines.extend(f"- skipped {issue.path}: {issue.error}" for issue in issues)
        return "\n".join(lines)
    status_by_name = {status.name.lower(): status for status in list_workflow_statuses(root)}
    lines = ["Discovered workflows:"]
    for spec in specs:
        status = status_by_name.get(spec.name.lower())
        state = "running" if status and status.running else "idle"
        interval = status.interval_seconds if status and status.interval_seconds is not None else spec.interval_seconds
        devices = ", ".join(status.bound_devices if status and status.bound_devices else spec.devices) or "none"
        lines.append(
            f"- {spec.name}: mode={spec.mode} | state={state} | devices={devices} | "
            f"interval={interval if interval is not None else 'n/a'} | path={spec.path}"
        )
    if issues:
        lines.append("Skipped invalid workflow files:")
        lines.extend(f"- {issue.path}: {issue.error}" for issue in issues)
    return "\n".join(lines)


def _format_workflow_show(spec) -> str:
    lines = [
        f"Workflow: {spec.name}",
        f"Path: {spec.path}",
        f"Source: {spec.source}",
        f"Mode: {spec.mode}",
        f"Devices: {', '.join(spec.devices) if spec.devices else 'none'}",
        f"Interval: {spec.interval_seconds if spec.interval_seconds is not None else 'n/a'}",
        f"Allow shell: {'true' if spec.allow_shell else 'false'}",
        f"Skills: {', '.join(spec.skills) if spec.skills else 'none'}",
        f"Files: {', '.join(spec.files) if spec.files else 'none'}",
        "",
        spec.body.strip() or "(empty workflow body)",
    ]
    return "\n".join(lines)


def _format_single_workflow_status(spec, status: WorkflowStatus | None) -> str:
    if status is None:
        return (
            f"Workflow: {spec.name}\n"
            f"Path: {spec.path}\n"
            "State: idle\n"
            "No runs recorded yet."
        )
    lines = [
        f"Workflow: {spec.name}",
        f"Path: {spec.path}",
        f"State: {'running' if status.running else 'idle'}",
        f"PID: {status.pid if status.pid is not None else 'n/a'}",
        f"Interval: {status.interval_seconds if status.interval_seconds is not None else spec.interval_seconds or 'n/a'}",
        f"Bound devices: {', '.join(status.bound_devices) if status.bound_devices else 'none'}",
        f"Last success: {status.last_success if status.last_success is not None else 'n/a'}",
        f"Last started: {status.last_started_at or 'n/a'}",
        f"Last finished: {status.last_finished_at or 'n/a'}",
        f"Last report: {status.last_report_path or 'n/a'}",
    ]
    if status.last_error:
        lines.append(f"Last error: {status.last_error}")
    return "\n".join(lines)


def _format_workflow_status(root: Path, name: str | None = None) -> str:
    specs = discover_workflow_specs(root)
    issues = discover_workflow_issues(root)
    if name:
        spec = load_workflow_spec(root, name)
        return _format_single_workflow_status(spec, load_workflow_status(root, spec.name))
    if not specs:
        if not issues:
            return "No workflows discovered."
        lines = ["No valid workflows discovered."]
        lines.extend(f"- skipped {issue.path}: {issue.error}" for issue in issues)
        return "\n".join(lines)
    status_by_name = {status.name.lower(): status for status in list_workflow_statuses(root)}
    lines = ["Workflow status:"]
    for spec in specs:
        status = status_by_name.get(spec.name.lower())
        state = "running" if status and status.running else "idle"
        last = "n/a" if status is None or status.last_success is None else ("success" if status.last_success else "failure")
        lines.append(f"- {spec.name}: state={state} | last={last} | path={spec.path}")
    if issues:
        lines.append("Skipped invalid workflow files:")
        lines.extend(f"- {issue.path}: {issue.error}" for issue in issues)
    return "\n".join(lines)


def _read_workflow_logs(root: Path, name: str, tail: int) -> str:
    if tail <= 0:
        raise ValueError("--tail must be greater than zero")
    spec = load_workflow_spec(root, name)
    run_files = sorted(workflow_runs_dir(root, spec.name).glob("*.md"))
    selected = run_files[-tail:]
    if not selected:
        latest = workflow_last_run_path(root, spec.name)
        if latest.is_file():
            selected = [latest]
    if not selected:
        return f"No workflow logs recorded yet for {spec.name}."
    chunks: list[str] = []
    for path in selected:
        chunks.append(f"# Log File: {path}\n\n{path.read_text(encoding='utf-8').strip()}")
    return "\n\n".join(chunks)


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
    active_name = str(updated_cfg.get("active_model_profile") or "").strip().lower()
    raw_profiles = updated_cfg.get("model_profiles")
    if active_name and isinstance(raw_profiles, list):
        updated_profiles: list[object] = []
        matched_active = False
        for item in raw_profiles:
            if isinstance(item, dict) and str(item.get("name") or "").strip().lower() == active_name:
                revised = dict(item)
                revised["context_window_tokens"] = context_tokens
                updated_profiles.append(revised)
                matched_active = True
                continue
            updated_profiles.append(item)
        if matched_active:
            updated_cfg["model_profiles"] = updated_profiles
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


async def _run_workflow_once(
    root: Path,
    spec_name: str,
    *,
    device_ids: list[str] | None = None,
) -> tuple[object, Path]:
    spec = load_workflow_spec(root, spec_name)
    cfg = load_config()
    result = await run_workflow(root, spec, cfg=cfg, cli_device_ids=device_ids)
    report_path = write_workflow_run_report(root, spec, result)
    save_workflow_status(
        root,
        WorkflowStatus(
            name=spec.name,
            running=False,
            pid=None,
            interval_seconds=spec.interval_seconds,
            bound_devices=result.bound_devices,
            last_started_at=result.started_at,
            last_finished_at=result.finished_at,
            last_success=result.success,
            last_error=result.error,
            last_report_path=str(report_path),
            updated_at=result.finished_at,
        ),
    )
    return result, report_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="open-jet offline agentic terminal UI",
        epilog=(
            "Useful commands: `open-jet commands` for slash commands, `open-jet status` for runtime/config status, "
            "`open-jet device list` for persistent device setup, and `open-jet workflow list` for Markdown workflows. "
            "In the TUI, use `/device` and `@device_id` tags for device inputs. See docs/usage/device-sources.md."
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
    subparsers.add_parser("update", help="pull the latest repo commit from the tracked remote branch")
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
    workflow_parser = subparsers.add_parser("workflow", help="run Markdown-defined workflows over the shared device backend")
    workflow_subparsers = workflow_parser.add_subparsers(dest="workflow_action")
    workflow_subparsers.add_parser("list", help="list discovered workflow Markdown files")
    workflow_show_parser = workflow_subparsers.add_parser("show", help="show a workflow definition")
    workflow_show_parser.add_argument("name", help="workflow name")
    workflow_run_parser = workflow_subparsers.add_parser("run", help="execute a workflow once")
    workflow_run_parser.add_argument("name", help="workflow name")
    workflow_run_parser.add_argument("--device", dest="device_ids", action="append", default=[], help="bind a device id for this run")
    workflow_start_parser = workflow_subparsers.add_parser("start", help="start a background workflow runner")
    workflow_start_parser.add_argument("name", help="workflow name")
    workflow_start_parser.add_argument("--device", dest="device_ids", action="append", default=[], help="bind a device id for this runner")
    workflow_start_parser.add_argument("--interval", type=int, default=None, help="poll interval in seconds")
    workflow_stop_parser = workflow_subparsers.add_parser("stop", help="stop a running workflow")
    workflow_stop_parser.add_argument("name", help="workflow name")
    workflow_status_parser = workflow_subparsers.add_parser("status", help="show workflow status")
    workflow_status_parser.add_argument("name", nargs="?", help="workflow name")
    workflow_logs_parser = workflow_subparsers.add_parser("logs", help="show workflow Markdown reports")
    workflow_logs_parser.add_argument("name", help="workflow name")
    workflow_logs_parser.add_argument("--tail", type=int, default=1, help="number of report files to show")
    workflow_assign_parser = workflow_subparsers.add_parser("assign", help="persist local device bindings for a workflow")
    workflow_assign_parser.add_argument("name", help="workflow name")
    workflow_assign_parser.add_argument("device_ids", nargs="+", help="one or more device ids from `open-jet device list`")
    workflow_runner_parser = subparsers.add_parser("workflow-runner", help=argparse.SUPPRESS)
    workflow_runner_parser.add_argument("name", help=argparse.SUPPRESS)
    workflow_runner_parser.add_argument("--device", dest="device_ids", action="append", default=[], help=argparse.SUPPRESS)
    workflow_runner_parser.add_argument("--interval", type=int, default=None, help=argparse.SUPPRESS)
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
    if args.command == "workflow":
        root = _workflow_root()
        action = str(getattr(args, "workflow_action", "") or "list").strip().lower()
        try:
            if action == "list":
                print(_format_workflow_list(root))
                return
            if action == "show":
                print(_format_workflow_show(load_workflow_spec(root, str(args.name))))
                return
            if action == "run":
                if args.device_ids:
                    validate_workflow_device_ids(load_config(), list(args.device_ids))
                result, report_path = asyncio.run(
                    _run_workflow_once(root, str(args.name), device_ids=list(args.device_ids) or None)
                )
                status = "success" if result.success else "failure"
                print(
                    f"Workflow {result.name} completed with status={status}.\n"
                    f"Report: {report_path}\n"
                    f"Bound devices: {', '.join(result.bound_devices) if result.bound_devices else 'none'}"
                )
                if result.error:
                    print(f"Error: {result.error}")
                return
            if action == "start":
                spec = load_workflow_spec(root, str(args.name))
                if args.device_ids:
                    validate_workflow_device_ids(load_config(), list(args.device_ids))
                interval = int(args.interval) if args.interval is not None else spec.interval_seconds
                if interval is None or interval <= 0:
                    raise ValueError("workflow start requires --interval or interval_seconds in the workflow file")
                pid = start_workflow_daemon(root, spec.name, device_ids=list(args.device_ids), interval_seconds=interval)
                print(f"Started workflow {spec.name} with pid={pid}.\nRunner log: {workflow_last_run_path(root, spec.name).parent / 'runner.log'}")
                return
            if action == "stop":
                spec = load_workflow_spec(root, str(args.name))
                stopped = stop_workflow_daemon(root, spec.name)
                if stopped:
                    print(f"Stopped workflow {spec.name}.")
                else:
                    print(f"Failed to stop workflow {spec.name} or it was not running.")
                return
            if action == "status":
                print(_format_workflow_status(root, str(args.name) if getattr(args, "name", None) else None))
                return
            if action == "logs":
                print(_read_workflow_logs(root, str(args.name), int(args.tail)))
                return
            if action == "assign":
                spec = load_workflow_spec(root, str(args.name))
                validate_workflow_device_ids(load_config(), list(args.device_ids))
                assignment_path = save_workflow_assignment(root, spec.name, [str(item).strip() for item in args.device_ids if str(item).strip()])
                print(
                    f"Saved workflow device bindings for {spec.name}.\n"
                    f"Assignment file: {assignment_path}"
                )
                return
        except ValueError as exc:
            raise SystemExit(str(exc))
        raise SystemExit(
            "Usage: open-jet workflow [list|show <name>|run <name>|start <name>|stop <name>|status [name]|logs <name>|assign <name> <device_id>...]"
        )
    if args.command == "workflow-runner":
        asyncio.run(
            run_workflow_daemon(
                _workflow_root(),
                str(args.name),
                device_ids=list(args.device_ids),
                interval_seconds=args.interval,
            )
        )
        return
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
