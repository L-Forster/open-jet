from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SkillSource:
    reference: str

    def fetch(self, destination: Path) -> Path:
        raise NotImplementedError("remote skill sources are not enabled in the offline-safe foundation")
