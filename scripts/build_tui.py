#!/usr/bin/env python3
"""Build the host OpenJet TUI executable and stage it for wheel packaging."""

from __future__ import annotations

import argparse
import hashlib
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
UI = ROOT / "ui"
TARGETS: dict[str, tuple[str, str]] = {
    "linux-x64": ("bun-linux-x64", "openjet-tui"),
    "linux-arm64": ("bun-linux-arm64", "openjet-tui"),
    "macos-x64": ("bun-darwin-x64", "openjet-tui"),
    "macos-arm64": ("bun-darwin-arm64", "openjet-tui"),
    "windows-x64": ("bun-windows-x64", "openjet-tui.exe"),
}


def target_tag() -> str:
    systems = {"linux": "linux", "darwin": "macos", "windows": "windows"}
    machines = {"x86_64": "x64", "amd64": "x64", "aarch64": "arm64", "arm64": "arm64"}
    system = platform.system().lower()
    machine = platform.machine().lower()
    if system not in systems or machine not in machines:
        raise SystemExit(f"unsupported TUI build target: {platform.system()} {platform.machine()}")
    return f"{systems[system]}-{machines[machine]}"


def copy_pi_theme_assets(next_to_binary: Path) -> None:
    """Pi's compiled TUI reads theme/dark.json next to the executable."""
    source = (
        UI
        / "node_modules"
        / "@earendil-works"
        / "pi-coding-agent"
        / "dist"
        / "modes"
        / "interactive"
        / "theme"
    )
    if not source.is_dir():
        raise SystemExit(f"Pi theme assets are missing: {source}")
    destination = next_to_binary / "theme"
    destination.mkdir(parents=True, exist_ok=True)
    for name in ("dark.json", "light.json"):
        shutil.copy2(source / name, destination / name)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-install", action="store_true")
    parser.add_argument("--target", choices=tuple(TARGETS), default=target_tag())
    args = parser.parse_args()
    # The toolchain (typescript, bun) lives in devDependencies, so the build
    # breaks on machines whose npm config or NODE_ENV omits dev installs.
    env = {**os.environ, "NODE_ENV": "development", "NPM_CONFIG_OMIT": ""}
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from src.openrouter_catalog import write_openrouter_catalog_ts

    write_openrouter_catalog_ts()
    if not args.skip_install:
        subprocess.run(
            ["npm", "ci", "--ignore-scripts", "--legacy-peer-deps", "--include=dev"],
            cwd=UI,
            check=True,
            env=env,
        )
    subprocess.run(["npm", "run", "build"], cwd=UI, check=True, env=env)
    local_bun = UI / "node_modules" / ".bin" / ("bun.exe" if os.name == "nt" else "bun")
    bun = shutil.which("bun") or (str(local_bun) if local_bun.is_file() else None)
    if not bun:
        raise SystemExit("Bun 1.3.10 is required to build the standalone TUI (https://bun.sh).")
    bun_target, executable = TARGETS[args.target]
    output = f"./dist/{executable}"
    subprocess.run(
        [bun, "build", "--compile", f"--target={bun_target}", "./dist/index.js", "--outfile", output],
        cwd=UI,
        check=True,
    )
    built = UI / "dist" / executable
    if not built.is_file():
        raise SystemExit(f"TUI build did not produce {built}")
    copy_pi_theme_assets(UI / "dist")
    destination = ROOT / "src" / "tui_bin" / args.target
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    target = destination / executable
    shutil.copy2(built, target)
    if os.name != "nt":
        target.chmod(0o755)
    copy_pi_theme_assets(destination)
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    (destination / "SHA256").write_text(f"{digest}  {executable}\n", encoding="utf-8")
    release_dir = ROOT / "release"
    release_dir.mkdir(exist_ok=True)
    release_name = f"openjet-tui-{args.target}{'.exe' if executable.endswith('.exe') else ''}"
    release_target = release_dir / release_name
    shutil.copy2(target, release_target)
    (release_dir / f"{release_name}.sha256").write_text(f"{digest}  {release_name}\n", encoding="utf-8")
    print(target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
