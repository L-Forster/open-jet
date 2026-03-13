from __future__ import annotations

from pathlib import Path

from src.runtime_limits import MemorySnapshot


def memory_snapshot(total_mb: float, available_mb: float) -> MemorySnapshot:
    used_percent = ((total_mb - available_mb) / total_mb) * 100.0 if total_mb else 0.0
    return MemorySnapshot(total_mb=total_mb, available_mb=available_mb, used_percent=used_percent)


def skill_doc(*, tags: list[str], body: str, mode: str | None = None) -> str:
    lines = ["---", "tags:"]
    for tag in tags:
        lines.append(f"  - {tag}")
    if mode:
        lines.append(f"mode: {mode}")
    lines.extend(["---", body.strip()])
    return "\n".join(lines)


def write_repo_fixture(
    root: Path,
    *,
    architecture_lines: list[str],
    role_docs: dict[str, str] | None = None,
    project_doc: str = "project guidance",
    skills: dict[str, str] | None = None,
    repo_files: dict[str, str] | None = None,
) -> None:
    agents = root / ".openjet" / "agents"
    projects = root / ".openjet" / "projects"
    skills_dir = root / ".openjet" / "skills"
    agents.mkdir(parents=True, exist_ok=True)
    projects.mkdir(parents=True, exist_ok=True)
    skills_dir.mkdir(parents=True, exist_ok=True)

    agents_md = (
        "## What This Project Is\n"
        "- offline-first local agent\n"
        "- layered context admission pipeline\n\n"
        "## Engineering Rules\n"
        "- keep context narrow\n"
        "- verify focused changes\n\n"
        "## Hardware And Performance Assumptions\n"
        "- Jetson-class memory pressure matters\n"
        "- 4k context windows are common\n\n"
        "## Core Architecture\n"
        + "\n".join(architecture_lines)
        + "\n"
    )
    (root / "AGENTS.md").write_text(agents_md, encoding="utf-8")
    (agents / "base.md").write_text("base guidance", encoding="utf-8")
    default_roles = {"coder": "coder guidance", "debugger": "debugger guidance", "reviewer": "reviewer guidance"}
    for role_name, body in {**default_roles, **(role_docs or {})}.items():
        (agents / f"{role_name}.md").write_text(body, encoding="utf-8")
    (projects / "default.md").write_text(project_doc, encoding="utf-8")

    for name, body in (skills or {}).items():
        (skills_dir / name).write_text(body, encoding="utf-8")

    for relative_path, content in (repo_files or {}).items():
        path = root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
