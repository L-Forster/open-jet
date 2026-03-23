from __future__ import annotations

import json
import re
import shutil
from dataclasses import replace
from pathlib import Path

from ..peripherals.types import Observation


class ObservationStore:
    def __init__(self, root: str | Path = ".openjet/state/observations") -> None:
        self.root = Path(root)

    def source_dir(self, source_id: str) -> Path:
        return self.root / "sources" / _safe_name(source_id)

    def persist(self, observation: Observation, *, copy_payload: bool = False) -> Observation:
        stored = self.copy_payload(observation) if copy_payload and observation.payload_ref else observation
        source_dir = self.source_dir(stored.source_id)
        source_dir.mkdir(parents=True, exist_ok=True)
        events_path = source_dir / "events.jsonl"
        latest_path = source_dir / "latest.json"
        with events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(stored.as_dict(), ensure_ascii=True) + "\n")
        latest_path.write_text(json.dumps(stored.as_dict(), ensure_ascii=True, indent=2), encoding="utf-8")
        return stored

    def copy_payload(self, observation: Observation) -> Observation:
        payload_ref = observation.payload_ref
        if not payload_ref:
            return observation
        source = Path(payload_ref)
        if not source.is_file():
            raise RuntimeError(f"payload file does not exist: {source}")
        timestamp = observation.timestamp.strftime("%Y%m%dT%H%M%SZ")
        destination_dir = (
            self.root
            / "payloads"
            / observation.modality.value
            / observation.timestamp.strftime("%Y")
            / observation.timestamp.strftime("%m")
            / observation.timestamp.strftime("%d")
            / _safe_name(observation.source_id)
        )
        destination_dir.mkdir(parents=True, exist_ok=True)
        destination = destination_dir / f"{timestamp}{source.suffix or _default_suffix(source.name)}"
        if source.resolve() != destination.resolve():
            shutil.copy2(source, destination)
        return replace(observation, payload_ref=str(destination))

    def append_text_buffer(
        self,
        source_id: str,
        line: str,
        *,
        buffer_name: str = "buffer.txt",
        max_lines: int = 200,
    ) -> Path:
        source_dir = self.source_dir(source_id)
        source_dir.mkdir(parents=True, exist_ok=True)
        buffer_path = source_dir / buffer_name
        lines: list[str] = []
        if buffer_path.exists():
            lines = buffer_path.read_text(encoding="utf-8").splitlines()
        lines.append(line.rstrip())
        if max_lines > 0 and len(lines) > max_lines:
            lines = lines[-max_lines:]
        buffer_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        return buffer_path

    def read_text_buffer(self, path: str | Path, *, max_chars: int = 4000) -> str:
        text = Path(path).read_text(encoding="utf-8")
        if max_chars <= 0 or len(text) <= max_chars:
            return text
        return text[-max_chars:]


def _safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip())
    return cleaned.strip("-") or "unknown"


def _default_suffix(name: str) -> str:
    return Path(name).suffix or ".bin"
