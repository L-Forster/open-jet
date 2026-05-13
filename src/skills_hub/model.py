from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


FindingSeverity = Literal["info", "warning", "danger"]


def default_hub_root(home: Path | None = None) -> Path:
    return Path(home or Path.home()).expanduser() / ".openjet" / "skills" / ".hub"


@dataclass(frozen=True)
class HubFinding:
    rule_id: str
    severity: FindingSeverity
    message: str
    path: str = ""
    line: int | None = None

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "rule_id": self.rule_id,
            "severity": self.severity,
            "message": self.message,
        }
        if self.path:
            payload["path"] = self.path
        if self.line is not None:
            payload["line"] = self.line
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "HubFinding":
        return cls(
            rule_id=str(payload.get("rule_id", "")),
            severity=str(payload.get("severity", "warning")),  # type: ignore[arg-type]
            message=str(payload.get("message", "")),
            path=str(payload.get("path", "")),
            line=int(payload["line"]) if isinstance(payload.get("line"), int) else None,
        )


@dataclass(frozen=True)
class HubInstallRecord:
    name: str
    version: str = ""
    source: str = ""
    installed_at: str = ""
    checksum: str = ""
    path: str = ""
    findings: tuple[HubFinding, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "version": self.version,
            "source": self.source,
            "installed_at": self.installed_at,
            "checksum": self.checksum,
            "path": self.path,
            "findings": [finding.to_dict() for finding in self.findings],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "HubInstallRecord":
        raw_findings = payload.get("findings")
        findings = tuple(
            HubFinding.from_dict(item)
            for item in raw_findings
            if isinstance(item, dict)
        ) if isinstance(raw_findings, list) else ()
        return cls(
            name=str(payload.get("name", "")),
            version=str(payload.get("version", "")),
            source=str(payload.get("source", "")),
            installed_at=str(payload.get("installed_at", "")),
            checksum=str(payload.get("checksum", "")),
            path=str(payload.get("path", "")),
            findings=findings,
        )


@dataclass(frozen=True)
class HubLockfile:
    version: int = 1
    skills: dict[str, HubInstallRecord] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "skills": {name: record.to_dict() for name, record in sorted(self.skills.items())},
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "HubLockfile":
        raw_skills = payload.get("skills")
        skills = {
            str(name): HubInstallRecord.from_dict(record)
            for name, record in raw_skills.items()
            if isinstance(record, dict)
        } if isinstance(raw_skills, dict) else {}
        return cls(version=int(payload.get("version", 1) or 1), skills=skills)


@dataclass(frozen=True)
class HubTap:
    name: str
    url: str
    trusted: bool = False

    def to_dict(self) -> dict[str, object]:
        return {"name": self.name, "url": self.url, "trusted": self.trusted}

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "HubTap":
        return cls(
            name=str(payload.get("name", "")),
            url=str(payload.get("url", "")),
            trusted=bool(payload.get("trusted", False)),
        )
