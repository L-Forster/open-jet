#!/usr/bin/env python3
"""Install the checksum-verified native TUI for a source checkout."""

from __future__ import annotations

import hashlib
import os
import platform
import shutil
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
RELEASE_BASE = "https://github.com/L-Forster/open-jet/releases/latest/download"


def tag() -> str:
    systems = {"linux": "linux", "darwin": "macos", "windows": "windows"}
    machines = {"x86_64": "x64", "amd64": "x64", "aarch64": "arm64", "arm64": "arm64"}
    system = platform.system().lower()
    machine = platform.machine().lower()
    if system not in systems or machine not in machines:
        raise SystemExit(f"No prebuilt OpenJet TUI is available for {platform.system()} {platform.machine()}.")
    return f"{systems[system]}-{machines[machine]}"


def main() -> int:
    executable = "openjet-tui.exe" if os.name == "nt" else "openjet-tui"
    local_build = ROOT / "ui" / "dist" / executable
    if local_build.is_file():
        return 0
    asset_name = f"openjet-tui-{tag()}{'.exe' if os.name == 'nt' else ''}"
    destination = ROOT / "ui" / "dist"
    destination.mkdir(parents=True, exist_ok=True)
    temporary = destination / f".{asset_name}.download"
    try:
        with urllib.request.urlopen(f"{RELEASE_BASE}/{asset_name}", timeout=60) as response:
            temporary.write_bytes(response.read())
        with urllib.request.urlopen(f"{RELEASE_BASE}/{asset_name}.sha256", timeout=30) as response:
            expected = response.read().decode("utf-8").split()[0].strip().lower()
        actual = hashlib.sha256(temporary.read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(f"checksum mismatch for {asset_name}")
        shutil.move(temporary, local_build)
        if os.name != "nt":
            local_build.chmod(0o755)
    except Exception as exc:
        temporary.unlink(missing_ok=True)
        raise SystemExit(
            f"Could not install the OpenJet TUI ({exc}). Install Node.js 22.19+ and Bun, "
            "then run `cd ui && npm ci --ignore-scripts --legacy-peer-deps && npm run build:binary`."
        ) from exc
    print(f"Installed {local_build}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
