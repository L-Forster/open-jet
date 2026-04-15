from __future__ import annotations

import asyncio
import os
import shlex
import tempfile
import uuid
from dataclasses import dataclass
from typing import Mapping

from .config import load_config


@dataclass(frozen=True)
class ShellTarget:
    name: str
    ssh_command: str
    scp_command: str
    scp_target: str
    remote_tmp_dir: str
    description: str


def configured_shell_targets(cfg: Mapping[str, object] | None = None) -> tuple[ShellTarget, ...]:
    resolved_cfg = cfg if isinstance(cfg, Mapping) else load_config()
    raw_targets = resolved_cfg.get("shell_targets")
    if not isinstance(raw_targets, Mapping):
        return ()

    targets: list[ShellTarget] = []
    for raw_name, raw_value in raw_targets.items():
        name = str(raw_name or "").strip()
        if not name or not isinstance(raw_value, Mapping):
            continue
        ssh_command = str(raw_value.get("ssh_command", "") or "").strip()
        scp_command = str(raw_value.get("scp_command", "") or "").strip()
        scp_target = str(raw_value.get("scp_target", "") or "").strip()
        remote_tmp_dir = str(raw_value.get("remote_tmp_dir", "/tmp") or "/tmp").strip() or "/tmp"
        description = str(raw_value.get("description", "") or "").strip() or f"{name} remote shell target"
        if not ssh_command or not scp_command or not scp_target:
            continue
        targets.append(
            ShellTarget(
                name=name,
                ssh_command=ssh_command,
                scp_command=scp_command,
                scp_target=scp_target,
                remote_tmp_dir=remote_tmp_dir,
                description=description,
            )
        )
    return tuple(targets)


def resolve_shell_target(name: str | None, cfg: Mapping[str, object] | None = None) -> ShellTarget | None:
    normalized = str(name or "").strip()
    if not normalized or normalized == "local":
        return None
    targets = configured_shell_targets(cfg)
    for target in targets:
        if target.name == normalized:
            return target
    available = ", ".join(["local", *[target.name for target in targets]])
    raise ValueError(f"Unknown shell target: {normalized}. Available targets: {available}")


def shell_targets_prompt(cfg: Mapping[str, object] | None = None) -> str:
    targets = configured_shell_targets(cfg)
    if not targets:
        return ""
    lines = [
        "Shell execution targets:",
        "- Use the shell tool's optional `target` field to choose where a command runs.",
        "- Omit `target` or use `local` to run on the machine hosting open-jet.",
    ]
    for target in targets:
        lines.append(f"- {target.name}: {target.description}")
    return "\n".join(lines)


async def run_shell_via_target(
    command: str,
    *,
    target: ShellTarget,
    timeout_seconds: int,
) -> tuple[int, str, str, bool]:
    local_script_path: str | None = None
    remote_script_path = f"{target.remote_tmp_dir.rstrip('/')}/openjet-{uuid.uuid4().hex}.sh"
    timed_out = False
    try:
        with tempfile.NamedTemporaryFile("w", delete=False, prefix="openjet-", suffix=".sh", encoding="utf-8") as handle:
            handle.write("#!/usr/bin/env bash\nset -euo pipefail\n")
            handle.write(command)
            handle.write("\n")
            local_script_path = handle.name

        remote_dest = f"{target.scp_target}:{remote_script_path}"
        await _run_command(f"{target.scp_command} {shlex.quote(local_script_path)} {shlex.quote(remote_dest)}")
        remote_cmd = (
            f"chmod 700 {shlex.quote(remote_script_path)} && "
            f"bash {shlex.quote(remote_script_path)}; "
            f"status=$?; rm -f {shlex.quote(remote_script_path)}; exit $status"
        )
        exit_code, stdout, stderr, timed_out = await _run_command(
            f"{target.ssh_command} {shlex.quote(remote_cmd)}",
            timeout_seconds=timeout_seconds,
        )
        return exit_code, stdout, stderr, timed_out
    finally:
        if local_script_path:
            try:
                os.unlink(local_script_path)
            except FileNotFoundError:
                pass


async def _run_command(command: str, *, timeout_seconds: int | None = None) -> tuple[int, str, str, bool]:
    proc = await asyncio.create_subprocess_shell(
        command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    timed_out = False
    try:
        if timeout_seconds is None:
            stdout_bytes, stderr_bytes = await proc.communicate()
        else:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(proc.communicate(), timeout=timeout_seconds)
    except asyncio.TimeoutError:
        timed_out = True
        proc.kill()
        stdout_bytes, stderr_bytes = await proc.communicate()
        timeout_note = f"Command timed out after {timeout_seconds}s and was terminated.".encode()
        if stderr_bytes:
            stderr_bytes = stderr_bytes + b"\n" + timeout_note
        else:
            stderr_bytes = timeout_note
    exit_code = proc.returncode if proc.returncode is not None else 0
    return exit_code, stdout_bytes.decode(errors="replace"), stderr_bytes.decode(errors="replace"), timed_out
