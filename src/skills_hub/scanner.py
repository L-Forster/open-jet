from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

from .model import HubFinding


_TEXT_SUFFIXES = {".md", ".txt", ".sh", ".bash", ".zsh", ".py", ".js", ".ts", ".json", ".yaml", ".yml"}
_RULES: tuple[tuple[str, str, str, re.Pattern[str]], ...] = (
    ("destructive_command", "danger", "Destructive command pattern found.", re.compile(r"\b(rm\s+-rf|mkfs|dd\s+if=|drop\s+table|git\s+reset\s+--hard)\b", re.I)),
    ("curl_pipe_bash", "danger", "curl/wget piped into a shell found.", re.compile(r"\b(curl|wget)\b[^\n|;]*\|\s*(bash|sh|zsh)\b", re.I)),
    ("prompt_injection", "danger", "Prompt-injection instruction found.", re.compile(r"\b(ignore|override|bypass)\b.{0,40}\b(previous|system|developer)\b.{0,40}\b(instruction|message|prompt)s?\b", re.I)),
    ("secret_exfiltration", "danger", "Secret exfiltration pattern found.", re.compile(r"\b(api[_-]?key|secret|token|password|ssh[_-]?key)\b.{0,80}\b(upload|exfiltrate|send|post|curl)\b", re.I)),
    ("suspicious_network_upload", "warning", "Suspicious network upload pattern found.", re.compile(r"\b(curl|wget|httpx|requests\.post|fetch)\b.{0,120}\b(upload|multipart|form-data|pastebin|webhook|ngrok)\b", re.I)),
)


@dataclass(frozen=True)
class ScanReport:
    findings: tuple[HubFinding, ...]

    @property
    def blocked(self) -> bool:
        return any(finding.severity == "danger" for finding in self.findings)

    @property
    def warnings(self) -> tuple[HubFinding, ...]:
        return tuple(finding for finding in self.findings if finding.severity == "warning")


def scan_skill_root(skill_root: Path) -> ScanReport:
    root = Path(skill_root).expanduser()
    findings: list[HubFinding] = []
    if not root.exists() or not root.is_dir():
        return ScanReport((HubFinding("missing_root", "danger", f"Skill root does not exist: {root}"),))
    root_resolved = root.resolve()
    for current, dirs, files in os.walk(root):
        current_path = Path(current)
        safe_dirs: list[str] = []
        for directory in dirs:
            child = current_path / directory
            if _unsafe_path(root_resolved, child):
                findings.append(HubFinding("unsafe_symlink", "danger", "Directory symlink escapes the skill root.", str(child)))
                continue
            safe_dirs.append(directory)
        dirs[:] = safe_dirs
        for file_name in files:
            path = current_path / file_name
            if _unsafe_path(root_resolved, path):
                findings.append(HubFinding("unsafe_symlink", "danger", "File symlink escapes the skill root.", str(path)))
                continue
            if path.suffix.lower() not in _TEXT_SUFFIXES:
                continue
            findings.extend(_scan_file(root_resolved, path))
    return ScanReport(tuple(findings))


def _scan_file(root: Path, path: Path) -> list[HubFinding]:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return [HubFinding("read_error", "warning", f"Could not read file: {exc}", str(path))]
    relative = str(path.relative_to(root))
    findings: list[HubFinding] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        for rule_id, severity, message, pattern in _RULES:
            if pattern.search(line):
                findings.append(HubFinding(rule_id, severity, message, relative, line_number))  # type: ignore[arg-type]
    return findings


def _unsafe_path(root: Path, path: Path) -> bool:
    try:
        resolved = path.resolve(strict=True)
    except OSError:
        return True
    try:
        resolved.relative_to(root)
        return False
    except ValueError:
        return True
