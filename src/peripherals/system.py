from __future__ import annotations

import glob
import platform
import shutil
import subprocess
from typing import Sequence

from .types import CommandResult


def run_command(args: Sequence[str], *, timeout_seconds: int = 5) -> CommandResult:
    proc = subprocess.run(
        list(args),
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        check=False,
    )
    return CommandResult(
        args=tuple(str(part) for part in args),
        returncode=proc.returncode,
        stdout=proc.stdout or "",
        stderr=proc.stderr or "",
    )


def resolve_binary(name: str) -> str | None:
    return shutil.which(name)


def glob_paths(pattern: str) -> list[str]:
    return sorted(glob.glob(pattern))


def device_discovery_hint() -> str | None:
    release = platform.release().lower()
    if "microsoft" in release or "wsl" in release:
        return (
            "Running inside WSL2. OpenJet only detects devices exposed to Linux as /dev nodes or audio endpoints. "
            "USB, cameras, GPIO, I2C, and serial adapters are not auto-forwarded into WSL."
        )
    return None
