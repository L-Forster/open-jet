from __future__ import annotations

from dataclasses import dataclass
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import yaml

from .config import CONFIG_PATH, normalize_config

RELEASES_LATEST_URL = "https://api.github.com/repos/l-forster/open-jet/releases/latest"
USER_AGENT = "open-jet-updater"
DEFAULT_RELEASE_METADATA_TIMEOUT_SECONDS = 4.0
DEFAULT_RELEASE_DOWNLOAD_TIMEOUT_SECONDS = 30.0


@dataclass(frozen=True)
class ReleaseInfo:
    tag_name: str
    version: str
    tarball_url: str


def _active_config_snapshot() -> tuple[Path, str] | None:
    for candidate in (Path("config.yaml"), CONFIG_PATH):
        if candidate.exists():
            return candidate, candidate.read_text()
    return None


def _restore_config(snapshot: tuple[Path, str] | None) -> None:
    if snapshot is None:
        return
    path, content = snapshot
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def _load_yaml_dict(raw: str) -> dict[str, object]:
    data = yaml.safe_load(raw) or {}
    if not isinstance(data, dict):
        raise RuntimeError("Config file must contain a YAML mapping at the top level.")
    return data


def _deep_merge(base: dict[str, object], overlay: dict[str, object]) -> dict[str, object]:
    merged = dict(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(
                dict(merged[key]),
                value,
            )
            continue
        merged[key] = value
    return merged


def _backup_path(path: Path, version: str) -> Path:
    safe_version = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in version)
    return path.with_name(f"{path.name}.bak.pre-update-{safe_version}")


def _preserve_user_config(snapshot: tuple[Path, str] | None, *, latest_version: str) -> None:
    if snapshot is None:
        return
    path, previous_raw = snapshot
    previous_cfg = _load_yaml_dict(previous_raw)
    path.parent.mkdir(parents=True, exist_ok=True)
    backup = _backup_path(path, latest_version)
    backup.write_text(previous_raw)

    if path.exists():
        current_raw = path.read_text()
        current_cfg = _load_yaml_dict(current_raw)
        merged_cfg = _deep_merge(current_cfg, previous_cfg)
    else:
        merged_cfg = previous_cfg

    path.write_text(yaml.safe_dump(normalize_config(merged_cfg), sort_keys=False))


def _latest_release(*, timeout_seconds: float = DEFAULT_RELEASE_METADATA_TIMEOUT_SECONDS) -> dict[str, object]:
    request = Request(
        RELEASES_LATEST_URL,
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": USER_AGENT,
        },
    )
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            payload = response.read().decode("utf-8")
    except HTTPError as exc:
        raise RuntimeError(f"Failed to fetch latest release metadata: HTTP {exc.code}.") from exc
    except URLError as exc:
        raise RuntimeError(f"Failed to fetch latest release metadata: {exc.reason}.") from exc

    try:
        release = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise RuntimeError("Failed to parse latest release metadata.") from exc

    if not isinstance(release, dict):
        raise RuntimeError("Latest release metadata was not a JSON object.")
    return release


def _download_release_archive(
    url: str,
    destination: Path,
    *,
    timeout_seconds: float = DEFAULT_RELEASE_DOWNLOAD_TIMEOUT_SECONDS,
) -> Path:
    request = Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            destination.write_bytes(response.read())
    except HTTPError as exc:
        raise RuntimeError(f"Failed to download release archive: HTTP {exc.code}.") from exc
    except URLError as exc:
        raise RuntimeError(f"Failed to download release archive: {exc.reason}.") from exc
    return destination


def latest_release_info(*, timeout_seconds: float = DEFAULT_RELEASE_METADATA_TIMEOUT_SECONDS) -> ReleaseInfo:
    release = _latest_release(timeout_seconds=timeout_seconds)
    tag_name = str(release.get("tag_name") or "").strip()
    tarball_url = str(release.get("tarball_url") or "").strip()
    if not tag_name or not tarball_url:
        raise RuntimeError("Latest release metadata is missing tag_name or tarball_url.")
    version = tag_name[1:] if tag_name.startswith("v") else tag_name
    return ReleaseInfo(tag_name=tag_name, version=version, tarball_url=tarball_url)


def available_release_update(
    *,
    current_version: str | None = None,
    timeout_seconds: float = DEFAULT_RELEASE_METADATA_TIMEOUT_SECONDS,
) -> ReleaseInfo | None:
    installed_version = str(current_version or "").strip()
    if not installed_version or installed_version.lower() == "unknown":
        return None
    release = latest_release_info(timeout_seconds=timeout_seconds)
    if installed_version == release.version:
        return None
    return release


def install_release(release: ReleaseInfo, *, current_version: str | None = None) -> str:
    latest_version = release.version
    tarball_url = release.tarball_url

    installed_version = str(current_version or "").strip()
    if installed_version and installed_version == latest_version:
        return f"open-jet {installed_version} is already up to date."

    snapshot = _active_config_snapshot()
    try:
        with tempfile.TemporaryDirectory(prefix="openjet-update-") as tmpdir:
            archive_path = Path(tmpdir) / f"open-jet-{latest_version}.tar.gz"
            _download_release_archive(tarball_url, archive_path)
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "--upgrade", str(archive_path)],
                check=True,
            )
    except subprocess.CalledProcessError as exc:
        _restore_config(snapshot)
        raise RuntimeError(f"Failed to install open-jet {latest_version}.") from exc

    _preserve_user_config(snapshot, latest_version=latest_version)
    if installed_version:
        return f"Updated open-jet from {installed_version} to {latest_version}."
    return f"Installed open-jet {latest_version}."


def update_from_latest_release(*, current_version: str | None = None) -> str:
    release = latest_release_info()
    return install_release(release, current_version=current_version)
