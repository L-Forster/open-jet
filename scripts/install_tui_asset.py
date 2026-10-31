#!/usr/bin/env python3
"""Install the native TUI for a source checkout, taken from the platform wheel on PyPI."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
PYPI_JSON = "https://pypi.org/pypi/open-jet/json"
# TUI tag -> the platform tag of the wheel that bundles it (.github/workflows/release.yml)
WHEEL_PLATFORMS = {
    "linux-x64": "manylinux_2_17_x86_64",
    "linux-arm64": "manylinux_2_17_aarch64",
    "macos-x64": "macosx_10_15_x86_64",
    "macos-arm64": "macosx_11_0_arm64",
    "windows-x64": "win_amd64",
}


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
    destination = ROOT / "ui" / "dist"
    if (destination / executable).is_file():
        return 0
    target = tag()
    try:
        with urllib.request.urlopen(PYPI_JSON, timeout=30) as response:
            release = json.load(response)
        wheel = next(u for u in release["urls"] if u["filename"].endswith(f"-{WHEEL_PLATFORMS[target]}.whl"))
        with tempfile.TemporaryDirectory() as tmp:
            archive = Path(tmp) / wheel["filename"]
            with urllib.request.urlopen(wheel["url"], timeout=300) as response:
                archive.write_bytes(response.read())
            if hashlib.sha256(archive.read_bytes()).hexdigest() != wheel["digests"]["sha256"]:
                raise RuntimeError(f"checksum mismatch for {wheel['filename']}")
            # the binary, its SHA256 file and Pi's theme assets, which it reads from next to itself
            # (under open_jet-<version>.data/purelib/ in the wheel)
            prefix = f"src/tui_bin/{target}/"
            with zipfile.ZipFile(archive) as wheel_zip:
                for name in wheel_zip.namelist():
                    if prefix in name and not name.endswith("/"):
                        out = destination / name.split(prefix, 1)[1]
                        out.parent.mkdir(parents=True, exist_ok=True)
                        with wheel_zip.open(name) as src, open(out, "wb") as dst:
                            shutil.copyfileobj(src, dst)
        if not (destination / executable).is_file():
            raise RuntimeError(f"{wheel['filename']} has no {prefix}{executable}")
        if os.name != "nt":
            (destination / executable).chmod(0o755)
    except Exception as exc:
        raise SystemExit(
            f"Could not install the OpenJet TUI ({exc}). Install Node.js 22.19+ and Bun, "
            "then run `git submodule update --init ui && python scripts/build_tui.py`."
        ) from exc
    print(f"Installed {destination / executable} from open-jet {release['info']['version']} on PyPI")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
