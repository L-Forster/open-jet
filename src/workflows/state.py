from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class WorkflowStatus:
    name: str
    running: bool = False
    pid: int | None = None
    interval_seconds: int | None = None
    bound_devices: tuple[str, ...] = field(default_factory=tuple)
    last_started_at: str | None = None
    last_finished_at: str | None = None
    last_success: bool | None = None
    last_error: str | None = None
    last_report_path: str | None = None
    updated_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["bound_devices"] = list(self.bound_devices)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> WorkflowStatus | None:
        if not isinstance(payload, dict):
            return None
        return cls(
            name=str(payload.get("name", "")).strip(),
            running=bool(payload.get("running", False)),
            pid=int(payload["pid"]) if payload.get("pid") is not None else None,
            interval_seconds=(
                int(payload["interval_seconds"])
                if payload.get("interval_seconds") is not None
                else None
            ),
            bound_devices=tuple(
                str(item).strip()
                for item in payload.get("bound_devices", [])
                if str(item).strip()
            ),
            last_started_at=str(payload["last_started_at"]) if payload.get("last_started_at") else None,
            last_finished_at=str(payload["last_finished_at"]) if payload.get("last_finished_at") else None,
            last_success=(
                bool(payload["last_success"])
                if payload.get("last_success") is not None
                else None
            ),
            last_error=str(payload["last_error"]) if payload.get("last_error") else None,
            last_report_path=str(payload["last_report_path"]) if payload.get("last_report_path") else None,
            updated_at=str(payload["updated_at"]) if payload.get("updated_at") else None,
        )


def workflow_state_root(root: Path) -> Path:
    return root / ".openjet" / "state" / "workflows"


def workflow_state_dir(root: Path, name: str) -> Path:
    return workflow_state_root(root) / _safe_name(name)


def workflow_runs_dir(root: Path, name: str) -> Path:
    return workflow_state_dir(root, name) / "runs"


def workflow_last_run_path(root: Path, name: str) -> Path:
    return workflow_state_dir(root, name) / "last-run.md"


def workflow_assignment_path(root: Path, name: str) -> Path:
    return workflow_state_dir(root, name) / "assignment.json"


def workflow_pid_path(root: Path, name: str) -> Path:
    return workflow_state_dir(root, name) / "pid.json"


def workflow_status_path(root: Path, name: str) -> Path:
    return workflow_state_dir(root, name) / "status.json"


def workflow_runner_log_path(root: Path, name: str) -> Path:
    return workflow_state_dir(root, name) / "runner.log"


def load_workflow_assignment(root: Path, name: str) -> tuple[str, ...]:
    payload = _load_json(workflow_assignment_path(root, name))
    if not isinstance(payload, dict):
        return ()
    normalized: list[str] = []
    seen: set[str] = set()
    for item in payload.get("device_ids", []):
        text = str(item).strip()
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        normalized.append(text)
    return tuple(normalized)


def save_workflow_assignment(root: Path, name: str, device_ids: list[str]) -> Path:
    path = workflow_assignment_path(root, name)
    _save_json(
        path,
        {
            "workflow": name,
            "device_ids": device_ids,
        },
    )
    return path


def load_workflow_pid(root: Path, name: str) -> dict[str, Any] | None:
    payload = _load_json(workflow_pid_path(root, name))
    if not isinstance(payload, dict):
        return None
    return payload


def save_workflow_pid(
    root: Path,
    name: str,
    *,
    pid: int,
    interval_seconds: int,
    device_ids: list[str],
    updated_at: str,
) -> Path:
    path = workflow_pid_path(root, name)
    _save_json(
        path,
        {
            "workflow": name,
            "pid": pid,
            "interval_seconds": interval_seconds,
            "device_ids": device_ids,
            "updated_at": updated_at,
        },
    )
    return path


def clear_workflow_pid(root: Path, name: str) -> None:
    try:
        workflow_pid_path(root, name).unlink()
    except FileNotFoundError:
        return


def load_workflow_status(root: Path, name: str) -> WorkflowStatus | None:
    status = WorkflowStatus.from_dict(_load_json(workflow_status_path(root, name)))
    pid_payload = load_workflow_pid(root, name)
    if status is None:
        if not isinstance(pid_payload, dict):
            return None
        pid = int(pid_payload["pid"]) if pid_payload.get("pid") is not None else None
        return WorkflowStatus(
            name=name,
            running=pid_is_running(pid),
            pid=pid,
            interval_seconds=(
                int(pid_payload["interval_seconds"])
                if pid_payload.get("interval_seconds") is not None
                else None
            ),
            bound_devices=tuple(
                str(item).strip()
                for item in pid_payload.get("device_ids", [])
                if str(item).strip()
            ),
            updated_at=str(pid_payload["updated_at"]) if pid_payload.get("updated_at") else None,
        )
    pid = int(pid_payload["pid"]) if isinstance(pid_payload, dict) and pid_payload.get("pid") is not None else status.pid
    running = pid_is_running(pid) if pid is not None else False
    return WorkflowStatus(
        name=status.name or name,
        running=running,
        pid=pid,
        interval_seconds=(
            int(pid_payload["interval_seconds"])
            if isinstance(pid_payload, dict) and pid_payload.get("interval_seconds") is not None
            else status.interval_seconds
        ),
        bound_devices=(
            tuple(str(item).strip() for item in pid_payload.get("device_ids", []) if str(item).strip())
            if isinstance(pid_payload, dict) and isinstance(pid_payload.get("device_ids"), list)
            else status.bound_devices
        ),
        last_started_at=status.last_started_at,
        last_finished_at=status.last_finished_at,
        last_success=status.last_success,
        last_error=status.last_error,
        last_report_path=status.last_report_path,
        updated_at=status.updated_at,
    )


def save_workflow_status(root: Path, status: WorkflowStatus) -> Path:
    path = workflow_status_path(root, status.name)
    _save_json(path, status.to_dict())
    return path


def list_workflow_statuses(root: Path) -> list[WorkflowStatus]:
    statuses: list[WorkflowStatus] = []
    state_root = workflow_state_root(root)
    if not state_root.exists():
        return statuses
    for child in sorted(state_root.iterdir()):
        if not child.is_dir():
            continue
        status = load_workflow_status(root, child.name)
        if status is not None:
            statuses.append(status)
    return statuses


def pid_is_running(pid: int | None) -> bool:
    if pid is None or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def _safe_name(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in "._-" else "-" for ch in value.strip())
    return cleaned.strip("-") or "workflow"
