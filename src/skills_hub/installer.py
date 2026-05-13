from __future__ import annotations

import hashlib
import shutil
from datetime import datetime, timezone
from pathlib import Path

from ..skills.validate import validate_skill_name
from .lockfile import ensure_hub_layout, update_lockfile
from .model import HubInstallRecord, default_hub_root
from .scanner import scan_skill_root


def install_local_skill(
    source_root: Path,
    *,
    name: str | None = None,
    hub_root: Path | None = None,
    skills_root: Path | None = None,
    force: bool = False,
) -> HubInstallRecord:
    source = Path(source_root).expanduser()
    if not source.exists() or not source.is_dir():
        raise ValueError(f"source skill directory does not exist: {source}")
    skill_name = (name or source.name).strip().lower()
    if not validate_skill_name(skill_name):
        raise ValueError(f"invalid skill name: {skill_name}")
    hub = ensure_hub_layout(hub_root or default_hub_root())
    destination_root = Path(skills_root or hub.parent).expanduser()
    install_root = destination_root / skill_name
    _assert_child(destination_root, install_root)

    quarantine_root = hub / "quarantine" / f"{skill_name}-{_utc_stamp()}"
    _copy_tree_safely(source, quarantine_root)
    # fetch -> quarantine -> validate -> scan -> install -> lockfile update
    if not (quarantine_root / "SKILL.md").is_file():
        raise ValueError("quarantined skill is missing SKILL.md")
    report = scan_skill_root(quarantine_root)
    if report.blocked:
        raise ValueError("dangerous scanner findings block install")
    if report.warnings and not force:
        raise ValueError("scanner warnings require force")
    if install_root.exists():
        raise ValueError(f"skill already installed: {skill_name}")
    _copy_tree_safely(quarantine_root, install_root)
    record = HubInstallRecord(
        name=skill_name,
        source=str(source),
        installed_at=datetime.now(timezone.utc).isoformat(),
        checksum=_tree_checksum(install_root),
        path=str(install_root),
        findings=report.findings,
    )
    update_lockfile(record, hub)
    _append_audit(hub, f"installed {skill_name} from {source}")
    return record


def _copy_tree_safely(source: Path, destination: Path) -> None:
    source_resolved = source.resolve(strict=True)
    if destination.exists():
        raise ValueError(f"destination already exists: {destination}")
    for path in source.rglob("*"):
        resolved = path.resolve(strict=True)
        _assert_child(source_resolved, resolved)
        relative = resolved.relative_to(source_resolved)
        if any(part in {"", ".", ".."} for part in relative.parts):
            raise ValueError(f"unsafe path in skill tree: {path}")
        target = destination / relative
        _assert_child(destination, target)
        if path.is_symlink():
            raise ValueError(f"symlinks are not installed from hub skills: {path}")
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)


def _assert_child(root: Path, path: Path) -> None:
    root_resolved = root.resolve() if root.exists() else root.absolute()
    path_resolved = path.resolve() if path.exists() else path.absolute()
    try:
        path_resolved.relative_to(root_resolved)
    except ValueError as exc:
        raise ValueError(f"path escapes root: {path}") from exc


def _tree_checksum(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest.update(str(path.relative_to(root)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return f"sha256:{digest.hexdigest()}"


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")


def _append_audit(hub_root: Path, line: str) -> None:
    timestamp = datetime.now(timezone.utc).isoformat()
    with (hub_root / "audit.log").open("a", encoding="utf-8") as handle:
        handle.write(f"{timestamp} {line}\n")
