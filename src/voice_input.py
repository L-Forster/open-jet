from __future__ import annotations

import asyncio
import re
from collections.abc import Awaitable, Callable, Mapping
from typing import Any

from .device_sources import DeviceSource, collect_device_observation
from .observation import ObservationStore


class VoiceInputController:
    DEFAULT_CHUNK_SECONDS = 3
    SEND_COMMAND_HINT = "send message"
    CLEAR_COMMAND_HINT = "clear message"
    STOP_COMMAND_HINT = "stop listening"

    def __init__(
        self,
        *,
        observation_store: ObservationStore,
        cfg_provider: Callable[[], Mapping[str, object] | None],
        list_sources: Callable[[], list[DeviceSource]],
        resolve_source: Callable[[str], DeviceSource | None],
        submit_text: Callable[[str], Awaitable[None]],
        is_busy: Callable[[], bool],
        is_quit_requested: Callable[[], bool],
        log_getter: Callable[[], Any],
        log_event: Callable[[str, Any], None] | None = None,
    ) -> None:
        self._observation_store = observation_store
        self._cfg_provider = cfg_provider
        self._list_sources = list_sources
        self._resolve_source = resolve_source
        self._submit_text = submit_text
        self._is_busy = is_busy
        self._is_quit_requested = is_quit_requested
        self._log_getter = log_getter
        self._log_event = log_event

        self._task: asyncio.Task[None] | None = None
        self._source_ref: str | None = None
        self._chunk_seconds = self.DEFAULT_CHUNK_SECONDS
        self._last_error: str | None = None
        self._warned_missing_transcript = False
        self._draft_segments: list[str] = []

    @classmethod
    def stop_command_hint(cls) -> str:
        return cls.STOP_COMMAND_HINT

    @classmethod
    def send_command_hint(cls) -> str:
        return cls.SEND_COMMAND_HINT

    @classmethod
    def clear_command_hint(cls) -> str:
        return cls.CLEAR_COMMAND_HINT

    @staticmethod
    def normalize_transcript(text: str) -> str:
        return " ".join(re.findall(r"[a-z0-9]+", str(text).lower()))

    @classmethod
    def transcript_requests_stop(cls, transcript: str) -> bool:
        normalized = cls.normalize_transcript(transcript)
        return normalized in {
            "stop voice",
            "stop listening",
            "stop recording",
            "voice off",
            "disable voice",
        }

    @classmethod
    def transcript_requests_send(cls, transcript: str) -> bool:
        normalized = cls.normalize_transcript(transcript)
        return normalized in {
            "send",
            "send message",
            "submit",
            "submit message",
        }

    @classmethod
    def transcript_requests_clear(cls, transcript: str) -> bool:
        normalized = cls.normalize_transcript(transcript)
        return normalized in {
            "clear",
            "clear message",
            "cancel message",
            "discard message",
        }

    def status_snapshot(self) -> dict[str, Any]:
        active = bool(self._task and not self._task.done())
        return {
            "active": active,
            "source_ref": self._source_ref if active else None,
            "chunk_seconds": self._chunk_seconds if active else None,
            "last_error": self._last_error,
            "draft_pending": bool(self._draft_segments),
            "draft_preview": self._draft_preview(),
            "send_command": self.SEND_COMMAND_HINT,
            "clear_command": self.CLEAR_COMMAND_HINT,
            "stop_command": self.STOP_COMMAND_HINT,
        }

    async def start(self, *, source_ref: str | None = None) -> dict[str, Any]:
        current = self.status_snapshot()
        if current["active"]:
            active_source = str(current["source_ref"] or "").strip()
            requested_source = str(source_ref or "").strip()
            if requested_source and active_source.lower() != requested_source.lower():
                raise RuntimeError(f"Voice input is already running on @{active_source}. Stop it before switching sources.")
            return current

        source = self._resolve_voice_source(source_ref)
        self._source_ref = source.primary_ref
        self._chunk_seconds = self.DEFAULT_CHUNK_SECONDS
        self._last_error = None
        self._warned_missing_transcript = False
        self._task = asyncio.create_task(
            self._capture_loop(source_ref=source.primary_ref, chunk_seconds=self._chunk_seconds)
        )
        self._emit_event(
            "voice_started",
            source_ref=source.primary_ref,
            device_id=source.device.id,
            chunk_seconds=self._chunk_seconds,
        )
        return self.status_snapshot()

    async def stop(self) -> bool:
        return await self._cancel(record_event_name="voice_stopped")

    async def shutdown(self) -> None:
        await self._cancel(record_event_name=None)

    async def _cancel(self, *, record_event_name: str | None) -> bool:
        task = self._task
        source_ref = self._source_ref
        self._task = None
        self._source_ref = None
        self._warned_missing_transcript = False
        self._draft_segments.clear()
        if not task or task.done():
            return False
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        if record_event_name:
            self._emit_event(record_event_name, source_ref=source_ref)
        return True

    def _resolve_voice_source(self, source_ref: str | None = None) -> DeviceSource:
        requested = str(source_ref or "").strip()
        if requested:
            source = self._resolve_source(requested)
            if source is None:
                raise ValueError(f"unknown device reference: {requested}")
            if source.device.kind.value != "microphone":
                raise ValueError(f"device @{source.primary_ref} is not a microphone")
            if not source.enabled:
                raise ValueError(f"device @{source.primary_ref} is disabled")
            return source

        for source in self._list_sources():
            if source.device.kind.value == "microphone" and source.enabled:
                return source
        raise ValueError("no enabled microphone sources are available")

    def _draft_text(self) -> str:
        return " ".join(segment.strip() for segment in self._draft_segments if segment.strip()).strip()

    def _draft_preview(self, *, max_len: int = 80) -> str | None:
        draft = self._draft_text()
        if not draft:
            return None
        if len(draft) <= max_len:
            return draft
        return f"{draft[:max_len].rstrip()}..."

    def _append_draft(self, utterance: str) -> None:
        text = utterance.strip()
        if text:
            self._draft_segments.append(text)

    def _log_lines(self, *lines: str) -> None:
        log = self._log_getter()
        for line in lines:
            log.write(line)
        log.write("")

    def _emit_event(self, event_type: str, **data: Any) -> None:
        if self._log_event is not None:
            self._log_event(event_type, **data)

    async def _capture_loop(self, *, source_ref: str, chunk_seconds: int) -> None:
        utterance_segments: list[str] = []
        try:
            while not self._is_quit_requested():
                if self._is_busy():
                    await asyncio.sleep(0.2)
                    continue

                source = self._resolve_source(source_ref)
                if source is None:
                    raise RuntimeError(f"voice source @{source_ref} is no longer available")
                if not source.enabled:
                    raise RuntimeError(f"voice source @{source_ref} is disabled")

                observation = await asyncio.to_thread(
                    collect_device_observation,
                    source,
                    store=self._observation_store,
                    cfg=self._cfg_provider(),
                    duration_seconds=chunk_seconds,
                )
                transcript = str(observation.metadata.get("transcript_text") or "").strip()
                if transcript:
                    utterance_segments.append(transcript)
                    self._warned_missing_transcript = False
                    continue

                speech_detected = bool(observation.metadata.get("speech_detected"))
                if speech_detected and not self._warned_missing_transcript:
                    self._log_lines(
                        "[yellow]Voice heard speech, but no transcript text was produced by the local speech pipeline. Skipping that chunk.[/]"
                    )
                    self._warned_missing_transcript = True
                    continue

                if not utterance_segments:
                    continue

                utterance = " ".join(segment.strip() for segment in utterance_segments if segment.strip()).strip()
                utterance_segments.clear()
                if not utterance:
                    continue
                if self.transcript_requests_stop(utterance):
                    self._draft_segments.clear()
                    self._emit_event(
                        "voice_stopped_by_speech",
                        source_ref=source_ref,
                        transcript=utterance,
                    )
                    self._log_lines(
                        f"[bold bright_white]Voice input stopped by spoken command ({self.stop_command_hint()}).[/]"
                    )
                    break
                if self.transcript_requests_clear(utterance):
                    self._draft_segments.clear()
                    self._log_lines("[bold bright_white]Voice draft cleared.[/]")
                    continue
                if self.transcript_requests_send(utterance):
                    draft = self._draft_text()
                    if not draft:
                        self._log_lines(
                            f"[yellow]Voice draft is empty. Speak a message before saying '{self.send_command_hint()}'.[/]"
                        )
                        continue
                    self._draft_segments.clear()
                    await self._submit_text(draft)
                    continue

                self._append_draft(utterance)
                preview = self._draft_preview()
                lines = [
                    (
                        "[bold bright_white]"
                        f"Voice draft updated. Say '{self.send_command_hint()}' to send, "
                        f"'{self.clear_command_hint()}' to discard, or '{self.stop_command_hint()}' to exit."
                        "[/]"
                    )
                ]
                if preview:
                    lines.append(f"[dim]{preview}[/]")
                self._log_lines(*lines)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._last_error = str(exc).strip() or f"{type(exc).__name__} (no message)"
            self._emit_event("voice_failed", source_ref=source_ref, error=self._last_error)
            self._log_lines(f"[yellow]Voice input stopped:[/] {self._last_error}")
        finally:
            self._warned_missing_transcript = False
            current_task = asyncio.current_task()
            if self._task is current_task or (self._task and self._task.done()):
                self._task = None
            if self._task is None:
                self._source_ref = None
