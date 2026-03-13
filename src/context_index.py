from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re


@dataclass(frozen=True)
class RepoFileSummary:
    path: str
    purpose: str
    related_tests: tuple[str, ...] = ()


@dataclass(frozen=True)
class RepoContextIndex:
    project_summary: str
    files: dict[str, RepoFileSummary]


def load_repo_context_index(root: Path) -> RepoContextIndex:
    cached = _load_cached_index(root)
    if cached is not None:
        return cached
    built = build_repo_context_index(root)
    _write_cached_index(root, built)
    return built


def build_repo_context_index(root: Path) -> RepoContextIndex:
    project_docs = _load_project_docs(root)
    file_summaries = _extract_file_summaries(project_docs)
    project_summary = _extract_project_summary(project_docs)
    return RepoContextIndex(project_summary=project_summary, files=file_summaries)


def lookup_file_summary(index: RepoContextIndex, path: str) -> RepoFileSummary | None:
    normalized = _normalize_path(path)
    if not normalized:
        return None
    if normalized in index.files:
        return index.files[normalized]
    if normalized.startswith("tests/") or normalized.startswith("../open-jet-internal/tests/"):
        return RepoFileSummary(
            path=normalized,
            purpose="Test file used for focused verification in the current task scope.",
            related_tests=(normalized,),
        )
    return None


def _load_project_docs(root: Path) -> str:
    chunks: list[str] = []
    agents = _read_text(root / "AGENTS.md")
    if agents:
        chunks.append(agents)
    projects_dir = root / ".openjet" / "projects"
    if projects_dir.exists():
        for path in sorted(projects_dir.glob("*.md")):
            body = _read_text(path)
            if body:
                chunks.append(body)
    return "\n\n".join(chunks)


def _extract_project_summary(text: str) -> str:
    if not text.strip():
        return ""
    lines = ["PROJECT CONTEXT SUMMARY"]
    for heading in ("## What This Project Is", "## Engineering Rules", "## Hardware And Performance Assumptions"):
        section = _section_lines(text, heading)
        cleaned = _compact_lines(section, limit=4 if heading == "## What This Project Is" else 3)
        if not cleaned:
            continue
        title = heading.replace("## ", "")
        lines.append(f"{title}:")
        lines.extend(cleaned)
    return "\n".join(lines)


def _extract_file_summaries(text: str) -> dict[str, RepoFileSummary]:
    results: dict[str, RepoFileSummary] = {}
    for line in text.splitlines():
        match = re.match(r"\s*-\s*`([^`]+)`:\s*(.+)", line)
        if not match:
            continue
        raw_path, purpose = match.groups()
        normalized = _normalize_path(raw_path)
        if not normalized:
            continue
        results[normalized] = RepoFileSummary(
            path=normalized,
            purpose=" ".join(purpose.split()),
            related_tests=tuple(_related_tests_for_file(normalized)),
        )
    return results


def _load_cached_index(root: Path) -> RepoContextIndex | None:
    path = root / ".openjet" / "state" / "repo_context_index.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    project_summary = str(payload.get("project_summary", "")).strip()
    raw_files = payload.get("files")
    if not isinstance(raw_files, dict):
        return None
    files: dict[str, RepoFileSummary] = {}
    for key, item in raw_files.items():
        if not isinstance(item, dict):
            continue
        normalized = _normalize_path(str(item.get("path", key)))
        purpose = str(item.get("purpose", "")).strip()
        if not normalized or not purpose:
            continue
        related = tuple(str(value) for value in item.get("related_tests", []) if str(value).strip())
        files[normalized] = RepoFileSummary(path=normalized, purpose=purpose, related_tests=related)
    return RepoContextIndex(project_summary=project_summary, files=files)


def _write_cached_index(root: Path, index: RepoContextIndex) -> None:
    path = root / ".openjet" / "state" / "repo_context_index.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "project_summary": index.project_summary,
        "files": {
            key: {
                "path": value.path,
                "purpose": value.purpose,
                "related_tests": list(value.related_tests),
            }
            for key, value in index.files.items()
        },
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _section_lines(text: str, heading: str) -> list[str]:
    lines = text.splitlines()
    capture = False
    collected: list[str] = []
    for line in lines:
        if line.startswith("## "):
            if line.strip() == heading:
                capture = True
                continue
            if capture:
                break
        if capture:
            collected.append(line)
    return collected


def _compact_lines(lines: list[str], *, limit: int) -> list[str]:
    cleaned: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("- "):
            stripped = stripped[2:].strip()
        cleaned.append(stripped)
        if len(cleaned) >= limit:
            break
    return cleaned


def _related_tests_for_file(path: str) -> list[str]:
    if path.startswith("src/") and path.endswith(".py"):
        stem = Path(path).stem
        repo_root = Path(__file__).resolve().parent.parent
        local_test = repo_root / "tests" / f"test_{stem}.py"
        if local_test.exists():
            return [f"tests/test_{stem}.py"]
        internal_test = repo_root.parent / "open-jet-internal" / "tests" / f"test_{stem}.py"
        if internal_test.exists():
            return [f"../open-jet-internal/tests/test_{stem}.py"]
        return [f"tests/test_{stem}.py"]
    if path.startswith("tests/") or path.startswith("../open-jet-internal/tests/"):
        return [path]
    return []


def _normalize_path(path: str) -> str:
    normalized = path.strip().replace("\\", "/")
    if not normalized:
        return ""
    return normalized


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return ""
