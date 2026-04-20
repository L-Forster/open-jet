from __future__ import annotations

from dataclasses import dataclass
import os
import shutil
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_INSTALL_SCRIPT = _REPO_ROOT / "install.sh"
_WINDOWS_INSTALL_SCRIPT = _REPO_ROOT / "install.bat"
_INSTALL_RELEVANT_FILES = {
    "install.bat",
    "install.sh",
    "pyproject.toml",
    "setup.py",
}


@dataclass(frozen=True)
class RepoUpdateInfo:
    remote: str
    branch: str
    local_commit: str
    remote_commit: str

    @property
    def local_short(self) -> str:
        return self.local_commit[:7]

    @property
    def remote_short(self) -> str:
        return self.remote_commit[:7]


def _git_output(*args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=_REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def _git_ok(*args: str) -> bool:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=_REPO_ROOT,
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except OSError:
        return False
    return result.returncode == 0


def _git_capture(*args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=_REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def _has_tracked_changes(cwd: Path) -> bool | None:
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return bool(result.stdout.strip())


def _llama_git_output(cwd: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def _sync_managed_llama_cpp_after_update() -> str | None:
    from .hardware import detect_hardware_info
    from .provisioning import LLAMA_CPP_DIR, _llama_cmake_args, _needs_rebuild, managed_llama_cpp_ref

    git_dir = LLAMA_CPP_DIR / ".git"
    if not git_dir.is_dir():
        return None
    if sys.platform == "darwin":
        return None

    status = _has_tracked_changes(LLAMA_CPP_DIR)
    if status is None:
        raise RuntimeError("Failed to inspect managed llama.cpp checkout before update.")
    if status:
        raise RuntimeError("Cannot update managed llama.cpp checkout with local changes. Commit or stash them first.")

    target_ref = managed_llama_cpp_ref()
    current_ref = _llama_git_output(LLAMA_CPP_DIR, "rev-parse", "HEAD")
    if current_ref is None:
        raise RuntimeError("Failed to read managed llama.cpp revision before update.")

    binary_name = "llama-server.exe" if sys.platform == "win32" else "llama-server"
    binary = LLAMA_CPP_DIR / "build" / "bin" / binary_name
    hardware_info = detect_hardware_info()
    needs_rebuild = not binary.is_file() or _needs_rebuild(hardware_info, str(binary))
    if current_ref == target_ref and not needs_rebuild:
        return None
    if shutil.which("cmake") is None:
        raise RuntimeError("cmake not found on PATH. Install CMake, e.g. `brew install cmake`, then rerun `openjet setup`.")

    try:
        subprocess.run(
            ["git", "fetch", "--tags", "--prune", "origin"],
            cwd=LLAMA_CPP_DIR,
            check=True,
        )
        subprocess.run(
            ["git", "checkout", "--detach", target_ref],
            cwd=LLAMA_CPP_DIR,
            check=True,
        )
        build_dir = LLAMA_CPP_DIR / "build"
        build_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            _llama_cmake_args(hardware_info),
            cwd=build_dir,
            check=True,
        )
        subprocess.run(
            ["cmake", "--build", ".", "--target", "llama-server", "-j4"],
            cwd=build_dir,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"Failed to sync managed llama.cpp checkout to {target_ref[:7]}."
        ) from exc
    return target_ref[:7]


def _changed_files(base: str, head: str) -> list[str] | None:
    output = _git_output("diff", "--name-only", base, head)
    if output is None:
        return None
    return [line.strip() for line in output.splitlines() if line.strip()]


def _update_requires_install(update: RepoUpdateInfo) -> bool:
    changed_files = _changed_files(update.local_commit, update.remote_commit)
    if changed_files is None:
        return True
    return any(path in _INSTALL_RELEVANT_FILES for path in changed_files)


def _install_command() -> list[str]:
    if sys.platform == "win32":
        return ["cmd", "/c", str(_WINDOWS_INSTALL_SCRIPT)]
    return ["bash", str(_INSTALL_SCRIPT)]


def _repo_tracking_target() -> tuple[str, str] | None:
    upstream = _git_output("rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{upstream}")
    if upstream:
        remote, _, branch = upstream.partition("/")
        if remote and branch:
            return remote, branch
    ref = _git_output("symbolic-ref", "--quiet", "refs/remotes/origin/HEAD")
    if not ref:
        return None
    prefix = "refs/remotes/"
    if not ref.startswith(prefix):
        return None
    tail = ref[len(prefix):]
    remote, _, branch = tail.partition("/")
    if not remote or not branch:
        return None
    return remote, branch


def available_update(*, current_version: str | None = None, timeout_seconds: float = 0.0) -> RepoUpdateInfo | None:
    del current_version, timeout_seconds
    target = _repo_tracking_target()
    if target is None:
        return None
    remote, branch = target
    if _git_capture("fetch", "--quiet", remote, branch) is None:
        return None
    local_commit = _git_output("rev-parse", "HEAD")
    remote_commit = _git_output("rev-parse", "FETCH_HEAD")
    if not local_commit or not remote_commit:
        return None
    remote_sha = remote_commit.split()[0].strip()
    if not remote_sha or local_commit == remote_sha:
        return None
    if not _git_ok("merge-base", "--is-ancestor", local_commit, remote_sha):
        return None
    return RepoUpdateInfo(
        remote=remote,
        branch=branch,
        local_commit=local_commit,
        remote_commit=remote_sha,
    )

def install_update(update: RepoUpdateInfo, *, current_version: str | None = None) -> str:
    del current_version
    status = _has_tracked_changes(_REPO_ROOT)
    if status is None:
        raise RuntimeError("Failed to inspect repo status before update.")
    if status:
        raise RuntimeError("Cannot update repo checkout with local changes. Commit or stash them first.")
    reinstall_required = _update_requires_install(update)
    try:
        subprocess.run(
            ["git", "pull", "--ff-only", update.remote, update.branch],
            cwd=_REPO_ROOT,
            check=True,
        )
        subprocess.run(
            _install_command(),
            cwd=_REPO_ROOT,
            env={
                **os.environ,
                "OPENJET_INSTALL_MODE": "update",
                "OPENJET_UPDATE_REINSTALL": "1" if reinstall_required else "0",
            },
            check=True,
        )
        llama_ref = _sync_managed_llama_cpp_after_update()
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"Failed to update repo checkout from {update.local_short} to {update.remote_short}."
        ) from exc
    message = f"Updated open-jet repo from {update.local_short} to {update.remote_short}."
    if llama_ref:
        message += f" Synced managed llama.cpp to {llama_ref}."
    return message


def update_from_latest_release(*, current_version: str | None = None) -> str:
    update = available_update(current_version=current_version)
    if update is None:
        return "open-jet repo is already up to date."
    return install_update(update, current_version=current_version)
