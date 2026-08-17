"""Locate and launch the bundled OpenJet TypeScript terminal frontend."""

from __future__ import annotations

import os
import platform
import hashlib
import subprocess
import sys
from importlib.resources import files
from pathlib import Path


class TuiLaunchError(RuntimeError):
    pass


def _wsl_workspace(path: Path | None = None) -> tuple[str, str] | None:
    """Return (distribution, Linux cwd) for a Windows WSL UNC workspace."""
    if os.name != "nt":
        return None
    raw = str(path or Path.cwd()).replace("/", "\\")
    parts = raw.split("\\")
    if len(parts) < 5 or parts[0:2] != ["", ""]:
        return None
    if parts[2].lower() not in {"wsl.localhost", "wsl$"} or not parts[3]:
        return None
    linux_cwd = "/" + "/".join(part for part in parts[4:] if part)
    return parts[3], linux_cwd or "/"


def configure_backend_environment(env: dict[str, str], *, cwd: Path | None = None) -> None:
    workspace = _wsl_workspace(cwd)
    if workspace is None:
        env["OPENJET_PYTHON"] = sys.executable
        return
    distribution, linux_cwd = workspace
    local_venv = (cwd or Path.cwd()) / ".venv" / "bin" / "python"
    env["OPENJET_WSL_DISTRO"] = distribution
    env["OPENJET_WSL_CWD"] = linux_cwd
    env["OPENJET_PYTHON"] = f"{linux_cwd}/.venv/bin/python" if local_venv.is_file() else "python3"


def platform_tag() -> str:
    system = platform.system().lower()
    machine = platform.machine().lower()
    system_map = {"linux": "linux", "darwin": "macos", "windows": "windows"}
    machine_map = {
        "x86_64": "x64",
        "amd64": "x64",
        "aarch64": "arm64",
        "arm64": "arm64",
    }
    if system not in system_map or machine not in machine_map:
        raise TuiLaunchError(
            f"The OpenJet TUI does not have a prebuilt binary for {platform.system()} {platform.machine()}. "
            "Install Node.js 22.19+ and Bun, run `python scripts/build_tui.py`, then set "
            "OPENJET_TUI_BINARY to the resulting executable if it is outside this checkout."
        )
    return f"{system_map[system]}-{machine_map[machine]}"


def tui_binary_candidates() -> list[Path]:
    executable = "openjet-tui.exe" if os.name == "nt" else "openjet-tui"
    candidates: list[Path] = []
    override = os.environ.get("OPENJET_TUI_BINARY", "").strip()
    if override:
        candidates.append(Path(override).expanduser())
    repository_root = Path(__file__).resolve().parent.parent
    candidates.append(repository_root / "ui" / "dist" / executable)
    candidates.append(Path(str(files("src").joinpath("tui_bin", platform_tag(), executable))))
    return candidates


def find_tui_binary() -> Path:
    for candidate in tui_binary_candidates():
        if candidate.is_file():
            checksum_file = candidate.parent / "SHA256"
            if checksum_file.is_file():
                expected = checksum_file.read_text(encoding="utf-8").split()[0].strip().lower()
                actual = hashlib.sha256(candidate.read_bytes()).hexdigest()
                if expected != actual:
                    raise TuiLaunchError(
                        f"The bundled OpenJet TUI failed checksum validation: {candidate}. Reinstall OpenJet."
                    )
            return candidate.resolve()
    searched = "\n  - ".join(str(path) for path in tui_binary_candidates())
    raise TuiLaunchError(
        "The OpenJet TypeScript TUI is missing. Reinstall the platform wheel or install Node.js 22.19+ "
        "and Bun, then run `python scripts/build_tui.py`.\n"
        f"Searched:\n  - {searched}"
    )


def launch_tui(*, force_setup: bool = False) -> None:
    workspace = _wsl_workspace()
    if workspace is not None:
        distribution, linux_cwd = workspace
        linux_binary = Path.cwd() / "ui" / "dist" / "openjet-tui"
        if linux_binary.is_file():
            linux_python = (
                f"{linux_cwd}/.venv/bin/python"
                if (Path.cwd() / ".venv" / "bin" / "python").is_file()
                else "python3"
            )
            completed = subprocess.run(
                [
                    "wsl.exe",
                    "-d",
                    distribution,
                    "--cd",
                    linux_cwd,
                    "--",
                    "env",
                    f"OPENJET_PYTHON={linux_python}",
                    f"OPENJET_FORCE_SETUP={'1' if force_setup else '0'}",
                    f"{linux_cwd}/ui/dist/openjet-tui",
                ],
                check=False,
            )
            if completed.returncode:
                raise SystemExit(completed.returncode)
            return
    binary = find_tui_binary()
    env = dict(os.environ)
    configure_backend_environment(env)
    env["OPENJET_FORCE_SETUP"] = "1" if force_setup else "0"
    args = [str(binary)]
    if os.name != "nt":
        os.execve(str(binary), args, env)
    completed = subprocess.run(args, env=env, check=False)
    if completed.returncode:
        raise SystemExit(completed.returncode)
