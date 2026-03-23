"""OpenTelemetry-backed session telemetry for open-jet."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from importlib.metadata import PackageNotFoundError, version as package_version
from pathlib import Path
from typing import Any, Mapping

from opentelemetry._logs import SeverityNumber
from opentelemetry.context import Context
from opentelemetry.metrics import Observation
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk._logs.export import (
    BatchLogRecordProcessor,
)
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import (
    PeriodicExportingMetricReader,
)
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
)
from opentelemetry.trace import Span, Status, StatusCode, set_span_in_context

from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

from .runtime_limits import CpuSample, estimate_tokens, read_cpu_percent, read_memory_info


LOG_SCHEMA_VERSION = 3
SERVICE_NAME = "open-jet"
SERVICE_NAMESPACE = "openjet"
DEFAULT_INSTALL_ID_PATH = Path(".openjet/state/telemetry_identity.json")
_PATH_TOKEN_RE = re.compile(r"(/[^\\s:]+)+")
_URL_RE = re.compile(r"https?://\\S+")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _session_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _sanitize_label(label: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_.-]+", "-", label.strip())
    return cleaned or "session"


def _truncate(value: str, limit: int = 512) -> str:
    if len(value) <= limit:
        return value
    return value[:limit] + f"... [truncated {len(value) - limit} chars]"


def _hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def _package_version() -> str:
    try:
        return package_version(SERVICE_NAME)
    except PackageNotFoundError:
        return "0.0.0"


def _compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"))


def _safe_model_name(model: str | None) -> str | None:
    if not model:
        return None
    return Path(str(model)).name or str(model)


def _normalize_slug(value: str | None, *, default: str = "unknown") -> str:
    text = (value or "").strip().lower()
    if not text:
        return default
    cleaned = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    return cleaned or default


def _coerce_optional_text(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


def _signal_endpoint(base_endpoint: str, signal: str) -> str:
    return f"{base_endpoint.rstrip('/')}/v1/{signal}"


def _normalize_headers(headers: Mapping[str, str] | None) -> dict[str, str]:
    if not headers:
        return {}
    normalized: dict[str, str] = {}
    for key, value in headers.items():
        if key is None or value is None:
            continue
        normalized[str(key)] = str(value)
    return normalized


def _load_or_create_install_id(path: Path) -> str:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        install_id = str(payload.get("install_id", "")).strip()
        if install_id:
            return install_id
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        pass

    install_id = uuid.uuid4().hex
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "install_id": install_id,
        "created_at": _utc_now(),
        "schema_version": LOG_SCHEMA_VERSION,
    }
    path.write_text(_compact_json(payload) + "\n", encoding="utf-8")
    return install_id


def _safe_error_details(error: str) -> dict[str, Any]:
    text = (error or "").strip()
    if not text:
        return {}
    redacted = _PATH_TOKEN_RE.sub("<path>", text)
    redacted = _URL_RE.sub("<url>", redacted)
    redacted = _truncate(redacted, limit=200)
    return {
        "error.message_hash": _hash_text(text),
        "error.message_redacted": redacted,
    }


def _normalize_attribute_value(value: Any) -> str | bool | int | float | list[str] | list[bool] | list[int] | list[float]:
    if isinstance(value, (str, bool, int, float)):
        return value
    if value is None:
        return "null"
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple, set)):
        items = list(value)
        if not items:
            return []
        if all(isinstance(item, str) for item in items):
            return [str(item) for item in items]
        if all(isinstance(item, bool) for item in items):
            return [bool(item) for item in items]
        if all(isinstance(item, int) and not isinstance(item, bool) for item in items):
            return [int(item) for item in items]
        if all(isinstance(item, (int, float)) and not isinstance(item, bool) for item in items):
            return [float(item) for item in items]
        return [_truncate(str(item), limit=120) for item in items]
    return _truncate(str(value), limit=160)


@dataclass(frozen=True)
class BroadcastConfig:
    enabled: bool = False
    endpoint: str | None = None
    headers: dict[str, str] | None = None
    timeout_seconds: float = 3.0
    export_logs: bool = True
    export_metrics: bool = True
    export_traces: bool = True


class SessionLogger:
    def __init__(
        self,
        base_dir: Path,
        label: str,
        metrics_interval_seconds: float = 5.0,
        *,
        install_id_path: Path = DEFAULT_INSTALL_ID_PATH,
        retention_days: int | None = 30,
        max_sessions: int | None = 100,
        broadcast: BroadcastConfig | None = None,
    ) -> None:
        self.base_dir = base_dir
        self.label = _sanitize_label(label)
        self.metrics_interval_seconds = max(1.0, float(metrics_interval_seconds))
        self.install_id_path = install_id_path
        self.retention_days = retention_days if retention_days is None else max(1, int(retention_days))
        self.max_sessions = max_sessions if max_sessions is None else max(1, int(max_sessions))
        self.broadcast = broadcast or BroadcastConfig()

        self.session_id = uuid.uuid4().hex
        self.install_id = _load_or_create_install_id(self.install_id_path)
        self.session_stamp = _session_stamp()
        date_prefix = datetime.now(timezone.utc).strftime("%Y/%m/%d")
        self.session_dir = self.base_dir / date_prefix / (
            f"{self.session_stamp}_{self.label}_{self.session_id[:8]}"
        )
        self.manifest_path = self.session_dir / "session.json"

        self._service_version = _package_version()
        self._event_index = 0
        self._prev_cpu: CpuSample | None = None
        self._last_observed_at = 0.0
        self._last_system_sample: dict[str, float | None] | None = None
        self._started = False
        self._runtime_context: dict[str, Any] = {}
        self._turn_started_at: dict[str, float] = {}
        self._turn_ttft_recorded: set[str] = set()
        self._turn_spans: dict[str, Span] = {}
        self._tool_spans: dict[str, Span] = {}
        self._tool_to_turn: dict[str, str] = {}
        self._session_span: Span | None = None
        self._session_context: Context | None = None

        resource = Resource.create(
            {
                "service.name": SERVICE_NAME,
                "service.namespace": SERVICE_NAMESPACE,
                "service.version": self._service_version,
                "deployment.environment": "local",
                "host.arch": os.uname().machine if hasattr(os, "uname") else None,
                "os.type": os.name,
            }
        )
        self._logger_provider = LoggerProvider(resource=resource)

        self._tracer_provider = TracerProvider(resource=resource)
        self._metric_readers: list[PeriodicExportingMetricReader] = []

        base_endpoint = (self.broadcast.endpoint or "").strip() if self.broadcast.enabled else ""
        headers = _normalize_headers(self.broadcast.headers)
        if self.broadcast.enabled and base_endpoint:
            if self.broadcast.export_logs:
                self._logger_provider.add_log_record_processor(
                    BatchLogRecordProcessor(
                        OTLPLogExporter(
                            endpoint=_signal_endpoint(base_endpoint, "logs"),
                            headers=headers,
                            timeout=float(self.broadcast.timeout_seconds),
                        ),
                        schedule_delay_millis=5000,
                        export_timeout_millis=int(self.broadcast.timeout_seconds * 1000),
                    )
                )
            if self.broadcast.export_traces:
                self._tracer_provider.add_span_processor(
                    BatchSpanProcessor(
                        OTLPSpanExporter(
                            endpoint=_signal_endpoint(base_endpoint, "traces"),
                            headers=headers,
                            timeout=float(self.broadcast.timeout_seconds),
                        ),
                        schedule_delay_millis=5000,
                        export_timeout_millis=int(self.broadcast.timeout_seconds * 1000),
                    )
                )
            if self.broadcast.export_metrics:
                self._metric_readers.append(
                    PeriodicExportingMetricReader(
                        OTLPMetricExporter(
                            endpoint=_signal_endpoint(base_endpoint, "metrics"),
                            headers=headers,
                            timeout=float(self.broadcast.timeout_seconds),
                        ),
                        export_interval_millis=int(self.metrics_interval_seconds * 1000),
                        export_timeout_millis=int(self.broadcast.timeout_seconds * 1000),
                    )
                )

        self._meter_provider = MeterProvider(metric_readers=tuple(self._metric_readers), resource=resource)
        self._logger = self._logger_provider.get_logger("openjet.telemetry", self._service_version)
        self._tracer = self._tracer_provider.get_tracer("openjet.telemetry", self._service_version)
        self._meter = self._meter_provider.get_meter("openjet.telemetry", self._service_version)

        self._sessions_started = self._meter.create_counter(
            "openjet.sessions.started",
            unit="{session}",
            description="Sessions started by open-jet.",
        )
        self._sessions_ended = self._meter.create_counter(
            "openjet.sessions.ended",
            unit="{session}",
            description="Sessions ended by open-jet.",
        )
        self._turns_started = self._meter.create_counter(
            "openjet.turns.started",
            unit="{turn}",
            description="Turns started by open-jet.",
        )
        self._turns_finished = self._meter.create_counter(
            "openjet.turns.finished",
            unit="{turn}",
            description="Turns finished by open-jet.",
        )
        self._tools_called = self._meter.create_counter(
            "openjet.tools.called",
            unit="{call}",
            description="Tool calls requested by the agent.",
        )
        self._approval_decisions = self._meter.create_counter(
            "openjet.tool.approvals",
            unit="{decision}",
            description="Tool approval decisions.",
        )
        self._errors = self._meter.create_counter(
            "openjet.errors",
            unit="{error}",
            description="Sanitized application errors.",
        )
        self._turn_duration = self._meter.create_histogram(
            "openjet.turn.duration",
            unit="ms",
            description="End-to-end turn duration.",
        )
        self._turn_ttft = self._meter.create_histogram(
            "openjet.turn.ttft",
            unit="ms",
            description="Time to first assistant response chunk.",
        )
        self._tool_duration = self._meter.create_histogram(
            "openjet.tool.duration",
            unit="ms",
            description="Tool execution duration.",
        )

        self._meter.create_observable_gauge(
            "openjet.system.cpu.percent",
            callbacks=[self._observe_cpu_percent],
            unit="%",
            description="Approximate system CPU utilization.",
        )
        self._meter.create_observable_gauge(
            "openjet.system.load_average.1m",
            callbacks=[self._observe_load_1m],
            unit="1",
            description="System load average over one minute.",
        )
        self._meter.create_observable_gauge(
            "openjet.system.load_average.5m",
            callbacks=[self._observe_load_5m],
            unit="1",
            description="System load average over five minutes.",
        )
        self._meter.create_observable_gauge(
            "openjet.system.load_average.15m",
            callbacks=[self._observe_load_15m],
            unit="1",
            description="System load average over fifteen minutes.",
        )
        self._meter.create_observable_gauge(
            "openjet.system.memory.total",
            callbacks=[self._observe_mem_total],
            unit="MiBy",
            description="Total system memory.",
        )
        self._meter.create_observable_gauge(
            "openjet.system.memory.available",
            callbacks=[self._observe_mem_available],
            unit="MiBy",
            description="Available system memory.",
        )
        self._meter.create_observable_gauge(
            "openjet.system.memory.used.percent",
            callbacks=[self._observe_mem_used_pct],
            unit="%",
            description="Used system memory percentage.",
        )
        self._meter.create_observable_gauge(
            "openjet.process.memory.rss",
            callbacks=[self._observe_process_rss],
            unit="MiBy",
            description="Resident set size of the current process.",
        )

    async def start(self) -> None:
        if self._started:
            return
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self._prune_sessions()
        self._started_at = _utc_now()
        self._session_span = self._tracer.start_span(
            "openjet.session",
            attributes=self._base_span_attributes()
            | {
                "openjet.label": self.label,
                "openjet.started_at": self._started_at,
            },
        )
        self._session_context = set_span_in_context(self._session_span)
        self._sessions_started.add(1, {"openjet.label": self.label})
        self._started = True
        self._write_manifest()
        self.emit_event(
            "openjet.session.start",
            body="session started",
            attributes={
                "openjet.label": self.label,
                "openjet.metrics_interval_seconds": self.metrics_interval_seconds,
                "openjet.collector_configured": self.broadcast.enabled and bool(self.broadcast.endpoint),
            },
        )

    async def stop(self) -> None:
        if not self._started:
            return
        for tool_key in list(self._tool_spans):
            self.finish_tool_call(
                tool_key,
                ok=False,
                approved=False,
                duration_ms=None,
                status="abandoned",
            )
        for turn_id in list(self._turn_spans):
            self.finish_turn(turn_id, success=False, status="abandoned")
        self.emit_event("openjet.session.end", body="session ended")
        self._sessions_ended.add(1, {"openjet.label": self.label})
        if self._session_span is not None:
            self._session_span.end()
            self._session_span = None
            self._session_context = None
        self._logger_provider.force_flush()
        self._tracer_provider.force_flush()
        self._meter_provider.force_flush()
        self._write_manifest(ended_at=_utc_now())
        self._meter_provider.shutdown()
        self._logger_provider.shutdown()
        self._tracer_provider.shutdown()
        self._started = False

    def set_runtime_context(self, **runtime_context: Any) -> None:
        sanitized = self._sanitize_runtime_context(runtime_context)
        self._runtime_context.update(sanitized)
        self._write_manifest()

    def log_event(self, event_type: str, **data: Any) -> None:
        payload = self._sanitize_generic_event(event_type, data)
        severity = SeverityNumber.ERROR if event_type.endswith("error") else SeverityNumber.INFO
        body = payload.pop("event.body", event_type.replace("_", " "))
        turn_id = str(payload.pop("openjet.turn.id", "")) or None
        self.emit_event(
            self._normalize_event_name(event_type),
            body=body,
            severity=severity,
            attributes=payload,
            turn_id=turn_id,
        )

    def record_runtime_ready(self, **runtime_context: Any) -> None:
        attrs = self._sanitize_runtime_context(runtime_context)
        self.set_runtime_context(**attrs)
        self.emit_event("openjet.runtime.ready", body="runtime ready", attributes=attrs)

    def record_user_message(
        self,
        *,
        turn_id: str,
        text: str,
        mentioned_files: list[str],
        attached_images: list[str],
        mode: str,
    ) -> None:
        self.emit_event(
            "openjet.user.message",
            body="user message",
            turn_id=turn_id,
            attributes={
                "openjet.mode": mode,
                "openjet.prompt.char_count": len(text),
                "openjet.prompt.token_estimate": estimate_tokens(text),
                "openjet.mentioned_file_count": len(mentioned_files),
                "openjet.attached_image_count": len(attached_images),
            },
        )

    def record_context_file_loaded(
        self,
        *,
        mention_path: str,
        resolved_path: str,
        context_tokens_before: int,
        estimated_tokens: int,
        returned_tokens: int,
        token_budget: int,
        remaining_prompt_tokens: int,
        truncated: bool,
        mem_available_mb: float | None,
    ) -> None:
        resolved = Path(resolved_path)
        self.emit_event(
            "openjet.context.file_loaded",
            body="context file loaded",
            attributes={
                "openjet.file_extension": resolved.suffix.lower() or "<none>",
                "openjet.file_is_absolute": resolved.is_absolute(),
                "openjet.file_name_hash": _hash_text(resolved.name or mention_path or resolved_path),
                "openjet.context_tokens_before": context_tokens_before,
                "openjet.estimated_tokens": estimated_tokens,
                "openjet.returned_tokens": returned_tokens,
                "openjet.token_budget": token_budget,
                "openjet.remaining_prompt_tokens": remaining_prompt_tokens,
                "openjet.truncated": truncated,
                "openjet.mem_available_mb": mem_available_mb if mem_available_mb is not None else "null",
            },
        )

    def record_slash_command(self, text: str) -> None:
        command, _, raw_arg = text[1:].partition(" ")
        self.emit_event(
            "openjet.command.slash",
            body="slash command",
            attributes={
                "openjet.command": command.strip() or "<empty>",
                "openjet.has_args": bool(raw_arg.strip()),
                "openjet.arg_char_count": len(raw_arg.strip()),
            },
        )

    def record_manual_condense(self, summary: str) -> None:
        self.emit_event(
            "openjet.context.condense",
            body="manual condense",
            attributes={
                "openjet.summary_char_count": len(summary),
                "openjet.summary_token_estimate": estimate_tokens(summary),
            },
        )

    def start_turn(
        self,
        *,
        turn_id: str,
        prompt: str,
        mode: str,
        resumed_session: bool,
        active_step: str | None,
        files_in_play: list[str],
        runtime_context: Mapping[str, Any],
    ) -> None:
        attrs = {
            **self._sanitize_runtime_context(runtime_context),
            "openjet.mode": mode,
            "openjet.resumed_session": resumed_session,
            "openjet.active_step_present": bool(active_step),
            "openjet.files_in_play_count": len(files_in_play),
            "openjet.prompt.char_count": len(prompt),
            "openjet.prompt.token_estimate": estimate_tokens(prompt),
        }
        span = self._tracer.start_span(
            "openjet.turn",
            context=self._session_context,
            attributes=self._base_span_attributes(turn_id=turn_id) | attrs,
        )
        self._turn_spans[turn_id] = span
        self._turn_started_at[turn_id] = time.monotonic()
        self._turns_started.add(1, {"openjet.mode": mode})
        self.emit_event(
            "openjet.turn.start",
            body="turn started",
            turn_id=turn_id,
            attributes=attrs,
        )

    def finish_turn(
        self,
        turn_id: str,
        *,
        success: bool,
        status: str,
        error: str | None = None,
        generation_tokens: int | None = None,
        tool_attempts: int | None = None,
        tool_successes: int | None = None,
        approval_requests: int | None = None,
        approval_grants: int | None = None,
        false_positive_command_proposals: int | None = None,
        hallucinated_command_proposals: int | None = None,
        recovered_after_resumed_session: bool | None = None,
        runtime_context: Mapping[str, Any] | None = None,
    ) -> None:
        started_at = self._turn_started_at.pop(turn_id, None)
        duration_ms = None
        if started_at is not None:
            duration_ms = round((time.monotonic() - started_at) * 1000.0, 2)
        span = self._turn_spans.pop(turn_id, None)
        attrs = {
            "openjet.success": success,
            "openjet.status": status,
            "openjet.duration_ms": duration_ms if duration_ms is not None else "null",
            "openjet.generation_tokens": generation_tokens if generation_tokens is not None else "null",
            "openjet.tool_attempts": tool_attempts if tool_attempts is not None else "null",
            "openjet.tool_successes": tool_successes if tool_successes is not None else "null",
            "openjet.approval_requests": approval_requests if approval_requests is not None else "null",
            "openjet.approval_grants": approval_grants if approval_grants is not None else "null",
            "openjet.false_positive_command_proposals": false_positive_command_proposals
            if false_positive_command_proposals is not None
            else "null",
            "openjet.hallucinated_command_proposals": hallucinated_command_proposals
            if hallucinated_command_proposals is not None
            else "null",
            "openjet.recovered_after_resumed_session": recovered_after_resumed_session
            if recovered_after_resumed_session is not None
            else "null",
        }
        if runtime_context:
            attrs.update(self._sanitize_runtime_context(runtime_context))
        attrs.update(_safe_error_details(error or ""))
        if duration_ms is not None:
            self._turn_duration.record(duration_ms, {"openjet.status": status})
        self._turns_finished.add(1, {"openjet.status": status, "openjet.success": success})
        if error:
            self._errors.add(1, {"openjet.component": "turn", "openjet.status": status})
        self.emit_event(
            "openjet.turn.finish",
            body="turn finished",
            severity=SeverityNumber.ERROR if not success else SeverityNumber.INFO,
            turn_id=turn_id,
            attributes=attrs,
        )
        if span is not None:
            for key, value in attrs.items():
                if value != "null":
                    span.set_attribute(key, value)
            if error:
                details = _safe_error_details(error)
                for key, value in details.items():
                    span.set_attribute(key, value)
                span.set_status(Status(StatusCode.ERROR, status))
            elif not success:
                span.set_status(Status(StatusCode.ERROR, status))
            else:
                span.set_status(Status(StatusCode.OK))
            span.end()
        self._turn_ttft_recorded.discard(turn_id)

    def record_first_response(
        self,
        *,
        turn_id: str,
        response_kind: str,
        text_len: int = 0,
        tool_call_count: int = 0,
    ) -> None:
        if turn_id in self._turn_ttft_recorded:
            return
        started_at = self._turn_started_at.get(turn_id)
        if started_at is None:
            return
        ttft_ms = round((time.monotonic() - started_at) * 1000.0, 2)
        self._turn_ttft.record(ttft_ms, {"openjet.response_kind": response_kind})
        self._turn_ttft_recorded.add(turn_id)
        self.emit_event(
            "openjet.turn.first_response",
            body="first response received",
            turn_id=turn_id,
            attributes={
                "openjet.response_kind": response_kind,
                "openjet.ttft_ms": ttft_ms,
                "openjet.text_len": text_len,
                "openjet.tool_call_count": tool_call_count,
            },
        )
        span = self._turn_spans.get(turn_id)
        if span is not None:
            span.add_event(
                "openjet.turn.first_response",
                {
                    "openjet.response_kind": response_kind,
                    "openjet.ttft_ms": ttft_ms,
                    "openjet.text_len": text_len,
                    "openjet.tool_call_count": tool_call_count,
                },
            )

    def record_agent_trace(self, event: str, data: Mapping[str, Any], *, turn_id: str | None = None) -> None:
        if event == "stream_first_chunk":
            text_len = int(data.get("text_len", 0) or 0)
            tool_call_count = int(data.get("tool_call_count", 0) or 0)
            response_kind = "text" if text_len > 0 else "tool_call" if tool_call_count > 0 else "empty"
            if turn_id:
                self.record_first_response(
                    turn_id=turn_id,
                    response_kind=response_kind,
                    text_len=text_len,
                    tool_call_count=tool_call_count,
                )
        attrs = {
            f"openjet.trace.{key}": _normalize_attribute_value(value)
            for key, value in data.items()
        }
        self.emit_event(
            f"openjet.agent.{event}",
            body=event.replace("_", " "),
            turn_id=turn_id,
            attributes=attrs,
        )

    def start_tool_call(
        self,
        *,
        turn_id: str,
        tool_key: str,
        tool_name: str,
        attributes: Mapping[str, Any] | None = None,
        needs_confirmation: bool = False,
    ) -> None:
        attrs = {
            "openjet.tool.name": tool_name,
            "openjet.tool.needs_confirmation": needs_confirmation,
        }
        if attributes:
            attrs.update(
                {
                    key: _normalize_attribute_value(value)
                    for key, value in attributes.items()
                }
            )
        parent = self._turn_spans.get(turn_id)
        context = set_span_in_context(parent) if parent is not None else self._session_context
        span = self._tracer.start_span(
            "openjet.tool",
            context=context,
            attributes=self._base_span_attributes(turn_id=turn_id, tool_key=tool_key) | attrs,
        )
        self._tool_spans[tool_key] = span
        self._tool_to_turn[tool_key] = turn_id
        self._tools_called.add(1, {"openjet.tool.name": tool_name, "openjet.needs_confirmation": needs_confirmation})
        self.emit_event(
            "openjet.tool.request",
            body="tool requested",
            turn_id=turn_id,
            tool_key=tool_key,
            attributes=attrs,
        )

    def record_tool_approval(
        self,
        *,
        tool_key: str,
        approved: bool,
        decision_ms: float | None = None,
    ) -> None:
        turn_id = self._tool_to_turn.get(tool_key)
        attrs = {
            "openjet.approved": approved,
            "openjet.decision_ms": decision_ms if decision_ms is not None else "null",
        }
        self._approval_decisions.add(1, {"openjet.approved": approved})
        self.emit_event(
            "openjet.tool.approval",
            body="tool approval decided",
            turn_id=turn_id,
            tool_key=tool_key,
            attributes=attrs,
        )
        span = self._tool_spans.get(tool_key)
        if span is not None:
            span.add_event("openjet.tool.approval", attrs)

    def finish_tool_call(
        self,
        tool_key: str,
        *,
        ok: bool,
        approved: bool,
        duration_ms: float | None,
        status: str,
        attributes: Mapping[str, Any] | None = None,
    ) -> None:
        turn_id = self._tool_to_turn.pop(tool_key, None)
        span = self._tool_spans.pop(tool_key, None)
        attrs = {
            "openjet.ok": ok,
            "openjet.approved": approved,
            "openjet.status": status,
            "openjet.duration_ms": duration_ms if duration_ms is not None else "null",
        }
        if attributes:
            attrs.update(
                {
                    key: _normalize_attribute_value(value)
                    for key, value in attributes.items()
                }
            )
        if duration_ms is not None and span is not None:
            tool_name = str(span.attributes.get("openjet.tool.name", "unknown"))
            self._tool_duration.record(duration_ms, {"openjet.tool.name": tool_name, "openjet.ok": ok})
        if not ok:
            self._errors.add(1, {"openjet.component": "tool", "openjet.status": status})
        self.emit_event(
            "openjet.tool.finish",
            body="tool finished",
            severity=SeverityNumber.ERROR if not ok else SeverityNumber.INFO,
            turn_id=turn_id,
            tool_key=tool_key,
            attributes=attrs,
        )
        if span is not None:
            for key, value in attrs.items():
                if value != "null":
                    span.set_attribute(key, value)
            if not ok:
                span.set_status(Status(StatusCode.ERROR, status))
            else:
                span.set_status(Status(StatusCode.OK))
            span.end()

    def record_exception(
        self,
        event_name: str,
        exc: Exception,
        *,
        component: str,
        turn_id: str | None = None,
        tool_key: str | None = None,
        extra_attributes: Mapping[str, Any] | None = None,
    ) -> None:
        attrs = {
            "error.type": type(exc).__name__,
            **_safe_error_details(str(exc)),
            "openjet.component": component,
        }
        if extra_attributes:
            attrs.update(
                {
                    key: _normalize_attribute_value(value)
                    for key, value in extra_attributes.items()
                }
            )
        self._errors.add(1, {"openjet.component": component, "error.type": type(exc).__name__})
        self.emit_event(
            self._normalize_event_name(event_name),
            body="error recorded",
            severity=SeverityNumber.ERROR,
            turn_id=turn_id,
            tool_key=tool_key,
            attributes=attrs,
        )
        span = self._resolve_span(turn_id=turn_id, tool_key=tool_key)
        if span is not None:
            span.record_exception(exc, attrs)
            span.set_status(Status(StatusCode.ERROR, type(exc).__name__))

    def emit_event(
        self,
        event_name: str,
        *,
        body: Any,
        attributes: Mapping[str, Any] | None = None,
        severity: SeverityNumber = SeverityNumber.INFO,
        turn_id: str | None = None,
        tool_key: str | None = None,
    ) -> None:
        self._event_index += 1
        attrs: dict[str, Any] = {
            "openjet.schema_version": LOG_SCHEMA_VERSION,
            "openjet.session.id": self.session_id,
            "openjet.install.id": self.install_id,
            "openjet.event.sequence": self._event_index,
            **self._runtime_context,
        }
        if turn_id:
            attrs["openjet.turn.id"] = turn_id
        if tool_key:
            attrs["openjet.tool.id"] = tool_key
        if attributes:
            for key, value in attributes.items():
                attrs[key] = _normalize_attribute_value(value)
        context = None
        span = self._resolve_span(turn_id=turn_id, tool_key=tool_key)
        if span is not None:
            context = set_span_in_context(span)
        elif self._session_span is not None:
            context = self._session_context
        self._logger.emit(
            timestamp=time.time_ns(),
            severity_number=severity,
            severity_text=severity.name.replace("SEVERITYNUMBER.", ""),
            body=body,
            attributes=attrs,
            event_name=event_name,
            context=context,
        )

    def _resolve_span(self, *, turn_id: str | None, tool_key: str | None) -> Span | None:
        if tool_key and tool_key in self._tool_spans:
            return self._tool_spans[tool_key]
        if turn_id and turn_id in self._turn_spans:
            return self._turn_spans[turn_id]
        return self._session_span

    def _base_span_attributes(self, *, turn_id: str | None = None, tool_key: str | None = None) -> dict[str, Any]:
        attrs = {
            "openjet.session.id": self.session_id,
            "openjet.install.id": self.install_id,
            **self._runtime_context,
        }
        if turn_id:
            attrs["openjet.turn.id"] = turn_id
        if tool_key:
            attrs["openjet.tool.id"] = tool_key
        return attrs

    def _sanitize_runtime_context(self, data: Mapping[str, Any]) -> dict[str, Any]:
        model_name = _safe_model_name(_coerce_optional_text(data.get("model"))) or "unknown"
        model_id = _normalize_slug(_coerce_optional_text(data.get("model_id")) or model_name)
        model_variant = _normalize_slug(_coerce_optional_text(data.get("model_variant")), default="unknown")
        use_case_tag = _normalize_slug(_coerce_optional_text(data.get("use_case_tag")), default="unknown")
        return {
            "openjet.app.version": self._service_version,
            "openjet.runtime": data.get("runtime", "unknown"),
            "openjet.backend": data.get("backend", "unknown"),
            "openjet.model.name": model_name,
            "openjet.model.id": model_id,
            "openjet.model.variant": model_variant,
            "openjet.hardware.class": data.get("hardware_class", "unknown"),
            "openjet.hardware.family": data.get("hardware_family", "unknown"),
            "openjet.hardware.accelerator": data.get("accelerator", data.get("device_profile", "auto")),
            "openjet.device_profile": data.get("device_profile", "auto"),
            "openjet.os.type": data.get("os_type", "unknown"),
            "openjet.context_window_tokens": int(data.get("context_window_tokens", 0) or 0),
            "openjet.gpu_layers": int(data.get("gpu_layers", 0) or 0),
            "openjet.system.memory.total_mb": round(float(data.get("system_memory_total_mb", 0.0) or 0.0), 2),
            "openjet.host_arch": data.get("host_arch") or "unknown",
            "openjet.use_case_tag": use_case_tag,
        }

    def _sanitize_generic_event(self, event_type: str, data: Mapping[str, Any]) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        text = str(data.get("text", "") or "")
        goal = str(data.get("goal", "") or "")
        error = str(data.get("error", "") or "")
        command = str(data.get("command", "") or "")
        summary = str(data.get("summary", "") or "")
        model = data.get("model")

        if "turn_id" in data:
            payload["openjet.turn.id"] = str(data["turn_id"])
        if text:
            payload["openjet.text_char_count"] = len(text)
            payload["openjet.text_token_estimate"] = estimate_tokens(text)
        if goal:
            payload["openjet.goal_char_count"] = len(goal)
            payload["openjet.goal_token_estimate"] = estimate_tokens(goal)
        if error:
            payload.update(_safe_error_details(error))
        if summary:
            payload["openjet.summary_char_count"] = len(summary)
            payload["openjet.summary_token_estimate"] = estimate_tokens(summary)
        if command:
            payload["openjet.command_char_count"] = len(command)
        if model:
            payload["openjet.model.name"] = _safe_model_name(str(model)) or "unknown"
        if "cwd" in data:
            payload["openjet.cwd_name"] = Path(str(data["cwd"])).name or "<root>"
        if "mentioned_files" in data:
            payload["openjet.mentioned_file_count"] = len(list(data.get("mentioned_files") or []))
        if "attached_images" in data:
            payload["openjet.attached_image_count"] = len(list(data.get("attached_images") or []))
        if "files_in_play" in data:
            payload["openjet.files_in_play_count"] = len(list(data.get("files_in_play") or []))

        for key, value in data.items():
            if key in {
                "turn_id",
                "text",
                "goal",
                "error",
                "summary",
                "command",
                "model",
                "cwd",
                "mentioned_files",
                "attached_images",
                "files_in_play",
            }:
                continue
            payload[f"openjet.{key}"] = value
        payload["event.body"] = event_type.replace("_", " ")
        return payload

    def _normalize_event_name(self, event_type: str) -> str:
        if event_type.startswith("openjet."):
            return event_type
        return f"openjet.{event_type.replace('_', '.')}"

    def _system_sample(self) -> dict[str, float | None]:
        now = time.monotonic()
        if self._last_system_sample is not None and now - self._last_observed_at < 0.2:
            return dict(self._last_system_sample)
        cpu_pct, self._prev_cpu = read_cpu_percent(self._prev_cpu, precision=2)
        mem = read_memory_info()
        load_avg = os.getloadavg() if hasattr(os, "getloadavg") else (None, None, None)
        sample = {
            "cpu_percent": cpu_pct,
            "load_avg_1m": load_avg[0],
            "load_avg_5m": load_avg[1],
            "load_avg_15m": load_avg[2],
            "mem_total_mb": mem.get("mem_total_mb"),
            "mem_available_mb": mem.get("mem_available_mb"),
            "mem_used_percent": mem.get("mem_used_percent"),
            "process_rss_mb": self._read_process_rss_mb(),
        }
        self._last_system_sample = sample
        self._last_observed_at = now
        return dict(sample)

    def _read_process_rss_mb(self) -> float | None:
        rss_kb: int | None = None
        try:
            with open("/proc/self/status", "r", encoding="utf-8") as handle:
                for line in handle:
                    if line.startswith("VmRSS:"):
                        rss_kb = int(line.split()[1])
                        break
        except OSError:
            return None
        if rss_kb is None:
            return None
        return round(rss_kb / 1024.0, 2)

    def _observe_cpu_percent(self, _options: Any) -> list[Observation]:
        sample = self._system_sample()
        value = sample.get("cpu_percent")
        return [] if value is None else [Observation(float(value))]

    def _observe_load_1m(self, _options: Any) -> list[Observation]:
        sample = self._system_sample()
        value = sample.get("load_avg_1m")
        return [] if value is None else [Observation(float(value))]

    def _observe_load_5m(self, _options: Any) -> list[Observation]:
        sample = self._system_sample()
        value = sample.get("load_avg_5m")
        return [] if value is None else [Observation(float(value))]

    def _observe_load_15m(self, _options: Any) -> list[Observation]:
        sample = self._system_sample()
        value = sample.get("load_avg_15m")
        return [] if value is None else [Observation(float(value))]

    def _observe_mem_total(self, _options: Any) -> list[Observation]:
        sample = self._system_sample()
        value = sample.get("mem_total_mb")
        return [] if value is None else [Observation(float(value))]

    def _observe_mem_available(self, _options: Any) -> list[Observation]:
        sample = self._system_sample()
        value = sample.get("mem_available_mb")
        return [] if value is None else [Observation(float(value))]

    def _observe_mem_used_pct(self, _options: Any) -> list[Observation]:
        sample = self._system_sample()
        value = sample.get("mem_used_percent")
        return [] if value is None else [Observation(float(value))]

    def _observe_process_rss(self, _options: Any) -> list[Observation]:
        sample = self._system_sample()
        value = sample.get("process_rss_mb")
        return [] if value is None else [Observation(float(value))]

    def _prune_sessions(self) -> None:
        self.base_dir.mkdir(parents=True, exist_ok=True)
        manifests = sorted(
            self.base_dir.glob("*/*/*/*/session.json"),
            key=lambda path: path.stat().st_mtime if path.exists() else 0.0,
            reverse=True,
        )
        to_remove: list[Path] = []
        if self.max_sessions is not None and len(manifests) > self.max_sessions:
            to_remove.extend(path.parent for path in manifests[self.max_sessions :])
        if self.retention_days is not None:
            cutoff = datetime.now(timezone.utc) - timedelta(days=self.retention_days)
            for manifest in manifests:
                try:
                    modified = datetime.fromtimestamp(manifest.stat().st_mtime, tz=timezone.utc)
                except OSError:
                    continue
                if modified < cutoff:
                    to_remove.append(manifest.parent)
        for session_dir in {path for path in to_remove if path != self.session_dir and path.exists()}:
            shutil.rmtree(session_dir, ignore_errors=True)

    def _write_manifest(self, *, ended_at: str | None = None) -> None:
        payload = {
            "schema_version": LOG_SCHEMA_VERSION,
            "session_id": self.session_id,
            "install_id": self.install_id,
            "label": self.label,
            "service_name": SERVICE_NAME,
            "service_version": self._service_version,
            "started_at": getattr(self, "_started_at", None) or _utc_now(),
            "ended_at": ended_at,
            "session_dir": str(self.session_dir),
            "telemetry": {
                "enabled": bool(self.broadcast.enabled and self.broadcast.endpoint),
                "collector_endpoint": self.broadcast.endpoint or None,
                "export_logs": self.broadcast.export_logs,
                "export_metrics": self.broadcast.export_metrics,
                "export_traces": self.broadcast.export_traces,
            },
            "runtime_context": self._runtime_context,
        }
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        self.manifest_path.write_text(_compact_json(payload) + "\n", encoding="utf-8")
