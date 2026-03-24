from __future__ import annotations

import asyncio
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from .reports import write_workflow_run_report
from .runner import WorkflowRunResult, run_workflow
from .specs import load_workflow_spec
from .state import (
    WorkflowStatus,
    clear_workflow_pid,
    load_workflow_pid,
    load_workflow_status,
    pid_is_running,
    save_workflow_pid,
    save_workflow_status,
    workflow_runner_log_path,
)

STOP_GRACE_SECONDS = 1.5
STOP_FORCE_TIMEOUT_SECONDS = 2.0


def start_workflow_daemon(
    root: Path,
    workflow_name: str,
    *,
    device_ids: Sequence[str],
    interval_seconds: int,
) -> int:
    existing = load_workflow_pid(root, workflow_name)
    existing_pid = int(existing["pid"]) if isinstance(existing, dict) and existing.get("pid") is not None else None
    if pid_is_running(existing_pid):
        raise ValueError(f"workflow {workflow_name} is already running")
    log_path = workflow_runner_log_path(root, workflow_name)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    runner_bootstrap = _runner_bootstrap()
    command = [
        sys.executable,
        "-c",
        runner_bootstrap,
        "workflow-runner",
        workflow_name,
        "--interval",
        str(interval_seconds),
    ]
    for device_id in device_ids:
        command.extend(["--device", str(device_id)])
    with log_path.open("a", encoding="utf-8") as handle:
        process = subprocess.Popen(
            command,
            cwd=str(root),
            stdin=subprocess.DEVNULL,
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    save_workflow_pid(
        root,
        workflow_name,
        pid=process.pid,
        interval_seconds=interval_seconds,
        device_ids=[str(item).strip() for item in device_ids if str(item).strip()],
        updated_at=_utcnow(),
    )
    return process.pid


async def run_workflow_daemon(
    root: Path,
    workflow_name: str,
    *,
    device_ids: Sequence[str],
    interval_seconds: int | None,
) -> None:
    spec = load_workflow_spec(root, workflow_name)
    effective_interval = interval_seconds or spec.interval_seconds
    if effective_interval is None or effective_interval <= 0:
        raise ValueError("workflow start requires an interval override or interval_seconds in the workflow file")

    stop_event = asyncio.Event()
    last_result: WorkflowRunResult | None = None
    last_report_path: str | None = None

    def _request_stop(*_args: object) -> None:
        stop_event.set()

    for signame in ("SIGTERM", "SIGINT"):
        signum = getattr(signal, signame, None)
        if signum is None:
            continue
        signal.signal(signum, _request_stop)

    save_workflow_pid(
        root,
        spec.name,
        pid=os.getpid(),
        interval_seconds=effective_interval,
        device_ids=[str(item).strip() for item in device_ids if str(item).strip()],
        updated_at=_utcnow(),
    )
    try:
        while not stop_event.is_set():
            last_result = await run_workflow(
                root,
                spec,
                cli_device_ids=[str(item).strip() for item in device_ids if str(item).strip()] or None,
            )
            last_report = write_workflow_run_report(root, spec, last_result)
            last_report_path = str(last_report)
            save_workflow_status(
                root,
                WorkflowStatus(
                    name=spec.name,
                    running=True,
                    pid=os.getpid(),
                    interval_seconds=effective_interval,
                    bound_devices=last_result.bound_devices,
                    last_started_at=last_result.started_at,
                    last_finished_at=last_result.finished_at,
                    last_success=last_result.success,
                    last_error=last_result.error,
                    last_report_path=last_report_path,
                    updated_at=_utcnow(),
                ),
            )
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=effective_interval)
            except asyncio.TimeoutError:
                continue
    finally:
        clear_workflow_pid(root, spec.name)
        save_workflow_status(
            root,
            WorkflowStatus(
                name=spec.name,
                running=False,
                pid=None,
                interval_seconds=effective_interval,
                bound_devices=last_result.bound_devices if last_result else (),
                last_started_at=last_result.started_at if last_result else None,
                last_finished_at=last_result.finished_at if last_result else None,
                last_success=last_result.success if last_result else None,
                last_error=last_result.error if last_result else None,
                last_report_path=last_report_path,
                updated_at=_utcnow(),
            ),
        )


def stop_workflow_daemon(root: Path, workflow_name: str) -> bool:
    payload = load_workflow_pid(root, workflow_name)
    pid = int(payload["pid"]) if isinstance(payload, dict) and payload.get("pid") is not None else None
    if not pid_is_running(pid):
        clear_workflow_pid(root, workflow_name)
        return False
    _signal_workflow_process(pid, signal.SIGTERM)
    if not _wait_for_pid_exit(pid, timeout_seconds=STOP_GRACE_SECONDS):
        force_signal = getattr(signal, "SIGKILL", signal.SIGTERM)
        _signal_workflow_process(pid, force_signal)
        if not _wait_for_pid_exit(pid, timeout_seconds=STOP_FORCE_TIMEOUT_SECONDS):
            return False
    clear_workflow_pid(root, workflow_name)
    _save_stopped_workflow_status(root, workflow_name, payload)
    return True


def _utcnow() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _signal_workflow_process(pid: int | None, signum: int) -> None:
    if pid is None or pid <= 0:
        return
    try:
        if hasattr(os, "killpg"):
            os.killpg(pid, signum)
        else:
            os.kill(pid, signum)
    except ProcessLookupError:
        return


def _wait_for_pid_exit(pid: int | None, *, timeout_seconds: float) -> bool:
    if pid is None or pid <= 0:
        return True
    deadline = time.monotonic() + max(0.0, timeout_seconds)
    while time.monotonic() < deadline:
        if not pid_is_running(pid):
            return True
        time.sleep(0.1)
    return not pid_is_running(pid)


def _save_stopped_workflow_status(root: Path, workflow_name: str, pid_payload: dict | None) -> None:
    status = load_workflow_status(root, workflow_name)
    interval_seconds = (
        int(pid_payload["interval_seconds"])
        if isinstance(pid_payload, dict) and pid_payload.get("interval_seconds") is not None
        else (status.interval_seconds if status else None)
    )
    bound_devices = (
        tuple(str(item).strip() for item in pid_payload.get("device_ids", []) if str(item).strip())
        if isinstance(pid_payload, dict) and isinstance(pid_payload.get("device_ids"), list)
        else (status.bound_devices if status else ())
    )
    save_workflow_status(
        root,
        WorkflowStatus(
            name=workflow_name,
            running=False,
            pid=None,
            interval_seconds=interval_seconds,
            bound_devices=bound_devices,
            last_started_at=status.last_started_at if status else None,
            last_finished_at=status.last_finished_at if status else None,
            last_success=status.last_success if status else None,
            last_error=status.last_error if status else None,
            last_report_path=status.last_report_path if status else None,
            updated_at=_utcnow(),
        ),
    )


def _runner_bootstrap() -> str:
    package_root = Path(__file__).resolve().parents[2]
    return (
        "import sys; "
        f"sys.path.insert(0, {str(package_root)!r}); "
        "from src.cli import main; "
        "main()"
    )
