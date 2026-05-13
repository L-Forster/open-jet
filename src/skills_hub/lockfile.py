from __future__ import annotations

import json
from pathlib import Path

from .model import HubInstallRecord, HubLockfile, default_hub_root


def ensure_hub_layout(hub_root: Path | None = None) -> Path:
    root = Path(hub_root or default_hub_root()).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    for directory in ("quarantine", "index-cache"):
        (root / directory).mkdir(exist_ok=True)
    for file_name, default_content in (("lock.json", "{}\n"), ("taps.json", "[]\n"), ("audit.log", "")):
        path = root / file_name
        if not path.exists():
            path.write_text(default_content, encoding="utf-8")
    return root


def load_lockfile(hub_root: Path | None = None) -> HubLockfile:
    root = ensure_hub_layout(hub_root)
    path = root / "lock.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8") or "{}")
    except json.JSONDecodeError:
        payload = {}
    if not isinstance(payload, dict):
        payload = {}
    return HubLockfile.from_dict(payload)


def save_lockfile(lockfile: HubLockfile, hub_root: Path | None = None) -> Path:
    root = ensure_hub_layout(hub_root)
    path = root / "lock.json"
    temp = path.with_suffix(".json.tmp")
    temp.write_text(json.dumps(lockfile.to_dict(), ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temp.replace(path)
    return path


def update_lockfile(record: HubInstallRecord, hub_root: Path | None = None) -> HubLockfile:
    current = load_lockfile(hub_root)
    updated = dict(current.skills)
    updated[record.name] = record
    lockfile = HubLockfile(version=current.version, skills=updated)
    save_lockfile(lockfile, hub_root)
    return lockfile
