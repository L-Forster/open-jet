#!/usr/bin/env python3
"""Compare current git state with the latest published PyPI release.

This script:
1) reads the package name from pyproject.toml (or --package),
2) fetches latest release metadata from PyPI,
3) maps that release version to a git tag/ref (vX.Y.Z or X.Y.Z),
4) compares that release commit with a target commit (default: HEAD).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Map latest PyPI release to git and compare with current commit."
    )
    parser.add_argument(
        "--package",
        help="PyPI package name. Defaults to [project].name in pyproject.toml.",
    )
    parser.add_argument(
        "--pyproject",
        default="pyproject.toml",
        help="Path to pyproject.toml (default: pyproject.toml).",
    )
    parser.add_argument(
        "--to",
        default="HEAD",
        help="Target git ref/commit to compare against (default: HEAD).",
    )
    parser.add_argument(
        "--from-ref",
        help="Override base git ref directly. Skips PyPI lookup if set.",
    )
    parser.add_argument(
        "--show-patch",
        action="store_true",
        help="Include full git patch after summary output.",
    )
    return parser.parse_args()


def read_package_name(pyproject_path: Path) -> str:
    if not pyproject_path.exists():
        raise SystemExit(f"pyproject not found: {pyproject_path}")
    text = pyproject_path.read_text(encoding="utf-8")
    in_project = False
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line.startswith("[") and line.endswith("]"):
            in_project = line == "[project]"
            continue
        if in_project and line.startswith("name"):
            parts = line.split("=", maxsplit=1)
            if len(parts) != 2:
                continue
            value = parts[1].strip().strip('"').strip("'")
            if value:
                return value
    raise SystemExit("Could not read [project].name from pyproject.toml")


def fetch_pypi_json(package: str) -> dict:
    url = f"https://pypi.org/pypi/{package}/json"
    try:
        with urllib.request.urlopen(url, timeout=15) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise SystemExit(f"PyPI lookup failed for '{package}': HTTP {exc.code}") from exc
    except urllib.error.URLError as exc:
        raise SystemExit(f"PyPI lookup failed for '{package}': {exc.reason}") from exc


def latest_release_version(payload: dict) -> tuple[str, str | None]:
    releases = payload.get("releases", {})
    latest_version = None
    latest_upload = None

    for version, files in releases.items():
        if not files:
            continue
        for file_meta in files:
            stamp = file_meta.get("upload_time_iso_8601") or file_meta.get("upload_time")
            if not stamp:
                continue
            stamp_cmp = stamp.replace("Z", "+00:00")
            if latest_upload is None or stamp_cmp > latest_upload:
                latest_upload = stamp_cmp
                latest_version = version

    if latest_version is None:
        raise SystemExit("No published release files found on PyPI.")
    return latest_version, latest_upload


def run_git(args: list[str]) -> str:
    proc = subprocess.run(
        ["git", *args],
        check=False,
        text=True,
        capture_output=True,
    )
    if proc.returncode != 0:
        msg = proc.stderr.strip() or proc.stdout.strip() or "git command failed"
        raise SystemExit(f"git {' '.join(args)} failed: {msg}")
    return proc.stdout.strip()


def resolve_commit(ref: str) -> str:
    return run_git(["rev-parse", "--verify", f"{ref}^{{commit}}"])


def resolve_release_ref(version: str) -> tuple[str, str]:
    candidates = [f"v{version}", version]
    for candidate in candidates:
        try:
            sha = resolve_commit(candidate)
            return candidate, sha
        except SystemExit:
            continue
    raise SystemExit(
        "Could not map PyPI release to git tag/ref. Tried: "
        + ", ".join(candidates)
        + ". Create one of these tags to enable automatic mapping."
    )


def fmt_timestamp(stamp: str | None) -> str:
    if not stamp:
        return "unknown"
    try:
        dt = datetime.fromisoformat(stamp)
    except ValueError:
        return stamp
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def print_section(title: str, content: str) -> None:
    print(f"\n== {title} ==")
    print(content if content else "(none)")


def main() -> None:
    args = parse_args()
    package = args.package or read_package_name(Path(args.pyproject))

    to_ref = args.to
    to_sha = resolve_commit(to_ref)

    if args.from_ref:
        from_ref = args.from_ref
        from_sha = resolve_commit(from_ref)
        release_version = "manual"
        upload_stamp = None
    else:
        payload = fetch_pypi_json(package)
        release_version, upload_stamp = latest_release_version(payload)
        from_ref, from_sha = resolve_release_ref(release_version)

    print(f"Package: {package}")
    print(f"PyPI latest release: {release_version} ({fmt_timestamp(upload_stamp)})")
    print(f"Base ref: {from_ref} ({from_sha})")
    print(f"Target ref: {to_ref} ({to_sha})")

    log_text = run_git(["log", "--oneline", "--no-merges", f"{from_sha}..{to_sha}"])
    names_text = run_git(["diff", "--name-status", from_sha, to_sha])
    stat_text = run_git(["diff", "--stat", from_sha, to_sha])

    print_section("Commits Since Release", log_text)
    print_section("Changed Files", names_text)
    print_section("Diffstat", stat_text)

    if args.show_patch:
        patch_text = run_git(["diff", from_sha, to_sha])
        print_section("Patch", patch_text)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
