"""Session and system metrics logging for open-jet."""

from __future__ import annotations

import asyncio
import json
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .runtime_limits import CpuSample, read_cpu_percent, read_memory_info


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _session_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _sanitize_label(label: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_.-]+", "-", label.strip())
    return cleaned or "session"


def _truncate(value: str, limit: int = 8000) -> str:
    if len(value) <= limit:
        return value
    return value[:limit] + f"\n... [truncated {len(value) - limit} chars]"


class SessionLogger:
    def __init__(
        self,
        base_dir: Path,
        label: str,
        metrics_interval_seconds: float = 5.0,
    ) -> None:
        self.base_dir = base_dir
        self.label = _sanitize_label(label)
        self.metrics_interval_seconds = max(1.0, float(metrics_interval_seconds))
        self.session_id = uuid.uuid4().hex
        self.session_stamp = _session_stamp()
        self.events_path = self.base_dir / (
            f"{self.session_stamp}_{self.label}_{self.session_id[:8]}.events.jsonl"
        )
        self.metrics_path = self.base_dir / (
            f"{self.session_stamp}_{self.label}_{self.session_id[:8]}.metrics.jsonl"
        )
        self._metrics_task: asyncio.Task[None] | None = None
        self._stop_metrics = asyncio.Event()
        self._prev_cpu: CpuSample | None = None

    async def start(self) -> None:
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.log_event(
            "session_start",
            session_id=self.session_id,
            label=self.label,
            events_file=str(self.events_path),
            metrics_file=str(self.metrics_path),
        )
        self._metrics_task = asyncio.create_task(self._metrics_loop())

    async def stop(self) -> None:
        self._stop_metrics.set()
        if self._metrics_task:
            self._metrics_task.cancel()
            try:
                await self._metrics_task
            except asyncio.CancelledError:
                pass
        self.log_event("session_end", session_id=self.session_id)

    def log_event(self, event_type: str, **data: Any) -> None:
        payload = {
            "timestamp": _utc_now(),
            "session_id": self.session_id,
            "type": event_type,
            "data": self._normalize(data),
        }
        self._write_jsonl(self.events_path, payload)

    def log_tool_result(self, tool_name: str, result: str, **meta: Any) -> None:
        self.log_event(
            "tool_result",
            tool=tool_name,
            result=_truncate(result),
            **meta,
        )

    def _normalize(self, value: Any) -> Any:
        if isinstance(value, dict):
            return {str(k): self._normalize(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self._normalize(v) for v in value]
        if isinstance(value, str):
            return _truncate(value)
        if isinstance(value, (int, float, bool)) or value is None:
            return value
        return _truncate(str(value))

    def _write_jsonl(self, path: Path, payload: dict[str, Any]) -> None:
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=True) + "\n")

    async def _metrics_loop(self) -> None:
        while not self._stop_metrics.is_set():
            sample = self._build_metrics_sample()
            self._write_jsonl(self.metrics_path, sample)
            try:
                await asyncio.wait_for(
                    self._stop_metrics.wait(),
                    timeout=self.metrics_interval_seconds,
                )
            except asyncio.TimeoutError:
                continue

    def _build_metrics_sample(self) -> dict[str, Any]:
        cpu_pct, self._prev_cpu = read_cpu_percent(self._prev_cpu, precision=2)
        mem = read_memory_info()
        proc = self._read_process_info()
        load_avg = os.getloadavg() if hasattr(os, "getloadavg") else (None, None, None)
        return {
            "timestamp": _utc_now(),
            "session_id": self.session_id,
            "cpu_percent": cpu_pct,
            "load_avg_1m": load_avg[0],
            "load_avg_5m": load_avg[1],
            "load_avg_15m": load_avg[2],
            **mem,
            **proc,
        }

    def _read_process_info(self) -> dict[str, Any]:
        rss_kb: int | None = None
        try:
            with open("/proc/self/status", "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        rss_kb = int(line.split()[1])
                        break
        except OSError:
            return {"process_rss_mb": None}
        return {"process_rss_mb": round((rss_kb or 0) / 1024.0, 2)}
