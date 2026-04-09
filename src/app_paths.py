from __future__ import annotations

from pathlib import Path


def openjet_install_root() -> Path:
    return Path(__file__).resolve().parent.parent


def global_openjet_root() -> Path:
    return openjet_install_root()


def project_openjet_root(root: Path | None = None) -> Path:
    base = Path(root or Path.cwd()).expanduser().resolve()
    return base / ".openjet"
