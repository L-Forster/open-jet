from __future__ import annotations

from pathlib import Path


def openjet_install_root() -> Path:
    return Path(__file__).resolve().parent.parent


def global_openjet_root() -> Path:
    return openjet_install_root()


def project_openjet_root(root: Path | None = None) -> Path:
    base = Path(root or Path.cwd()).expanduser().resolve()
    return base / ".openjet"


def find_project_root(start: Path | None = None) -> Path | None:
    """Nearest ancestor holding a provisioned .openjet directory.

    Walks up so `open-jet` works from any subdirectory of the project, and stops at the
    repository root so a provisioned project never leaks its model into a sibling.
    """
    current = Path(start or Path.cwd()).expanduser().resolve()
    for candidate in [current, *current.parents]:
        if (candidate / ".openjet").is_dir():
            return candidate
        if (candidate / ".git").exists():
            return None
    return None


def project_config_path(root: Path | None = None) -> Path:
    return project_openjet_root(root) / "config.yaml"


def project_models_dir(root: Path | None = None) -> Path:
    return project_openjet_root(root) / "models"
