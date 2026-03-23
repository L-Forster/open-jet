"""Persistent session state for resume on restart."""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .multimodal import content_to_plain_text


def _load_payload(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def _save_payload(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def _normalize_chat_id(value: object) -> str:
    text = "".join(ch for ch in str(value or "").strip().lower() if ch.isalnum())
    return text or uuid.uuid4().hex


def _message_count(messages: object) -> int:
    if not isinstance(messages, list):
        return 0
    return sum(1 for message in messages if isinstance(message, dict) and message.get("role") != "system")


def _loaded_file_count(payload: dict[str, Any]) -> int:
    loaded = payload.get("loaded_files")
    return len(loaded) if isinstance(loaded, dict) else 0


def _saved_at(payload: dict[str, Any], path: Path) -> float:
    value = payload.get("saved_at")
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return path.stat().st_mtime
    except OSError:
        return time.time()


def _preview_from_messages(messages: object) -> str:
    if not isinstance(messages, list):
        return ""
    for role in ("user", "assistant", "tool"):
        for message in messages:
            if not isinstance(message, dict) or message.get("role") != role:
                continue
            text = content_to_plain_text(message.get("content", "")).strip()
            if text:
                single_line = " ".join(text.split())
                return single_line[:96]
    return ""


@dataclass(frozen=True)
class SavedChatEntry:
    chat_id: str
    state_path: Path
    saved_at: float
    reason: str
    preview: str
    message_count: int
    loaded_file_count: int
    runtime: str
    model_ref: str
    uses_resume_checkpoint: bool
    kv_cache_available: bool


def build_saved_chat_entry(
    *,
    chat_id: str,
    payload: dict[str, Any],
    state_path: Path,
    uses_resume_checkpoint: bool,
    kv_cache_available: bool,
) -> SavedChatEntry | None:
    messages = payload.get("messages")
    message_count = _message_count(messages)
    if message_count <= 0:
        return None
    return SavedChatEntry(
        chat_id=_normalize_chat_id(chat_id),
        state_path=state_path,
        saved_at=_saved_at(payload, state_path),
        reason=str(payload.get("reason", "")).strip(),
        preview=_preview_from_messages(messages),
        message_count=message_count,
        loaded_file_count=_loaded_file_count(payload),
        runtime=str(payload.get("runtime", "")).strip(),
        model_ref=str(payload.get("model_ref") or payload.get("model") or "").strip(),
        uses_resume_checkpoint=uses_resume_checkpoint,
        kv_cache_available=kv_cache_available,
    )


@dataclass
class SessionStateStore:
    path: Path
    enabled: bool = True

    def load(self) -> dict[str, Any] | None:
        if not self.enabled:
            return None
        return _load_payload(self.path)

    def save(self, payload: dict[str, Any]) -> None:
        if not self.enabled:
            return
        _save_payload(self.path, payload)


@dataclass
class ChatArchiveStore:
    root: Path
    enabled: bool = True

    @classmethod
    def from_session_state_path(cls, path: Path, *, enabled: bool = True) -> "ChatArchiveStore":
        return cls(root=path.parent / "chats", enabled=enabled)

    @staticmethod
    def new_chat_id() -> str:
        return uuid.uuid4().hex

    def live_state_path(self, chat_id: str) -> Path:
        return self.root / _normalize_chat_id(chat_id) / "session_state.json"

    def resume_state_path(self, chat_id: str) -> Path:
        return self.root / _normalize_chat_id(chat_id) / "resume_state.json"

    def kv_cache_path(self, chat_id: str) -> Path:
        return self.root.parent / "swap" / f"resume_{_normalize_chat_id(chat_id)}.bin"

    def save_live_state(self, chat_id: str, payload: dict[str, Any]) -> None:
        if not self.enabled:
            return
        _save_payload(self.live_state_path(chat_id), payload)

    def save_resume_state(self, chat_id: str, payload: dict[str, Any]) -> None:
        if not self.enabled:
            return
        _save_payload(self.resume_state_path(chat_id), payload)

    def load_live_state(self, chat_id: str) -> dict[str, Any] | None:
        if not self.enabled:
            return None
        return _load_payload(self.live_state_path(chat_id))

    def load_resume_state(self, chat_id: str) -> dict[str, Any] | None:
        if not self.enabled:
            return None
        return _load_payload(self.resume_state_path(chat_id))

    def load_chat(self, chat_id: str, *, prefer_resume_checkpoint: bool = True) -> tuple[dict[str, Any] | None, bool]:
        if not self.enabled:
            return None, False
        if prefer_resume_checkpoint:
            payload = self.load_resume_state(chat_id)
            if payload is not None:
                return payload, self.kv_cache_path(chat_id).exists()
        return self.load_live_state(chat_id), False

    def delete_kv_cache(self, chat_id: str) -> None:
        if not self.enabled:
            return
        try:
            self.kv_cache_path(chat_id).unlink()
        except FileNotFoundError:
            return
        except OSError:
            return

    def list_chats(self) -> list[SavedChatEntry]:
        if not self.enabled or not self.root.exists():
            return []
        entries: list[SavedChatEntry] = []
        for child in self.root.iterdir():
            if not child.is_dir():
                continue
            chat_id = child.name
            resume_path = self.resume_state_path(chat_id)
            live_path = self.live_state_path(chat_id)
            uses_resume_checkpoint = resume_path.exists()
            state_path = resume_path if uses_resume_checkpoint else live_path
            payload = _load_payload(state_path)
            if payload is None:
                continue
            entry = build_saved_chat_entry(
                chat_id=chat_id,
                payload=payload,
                state_path=state_path,
                uses_resume_checkpoint=uses_resume_checkpoint,
                kv_cache_available=uses_resume_checkpoint and self.kv_cache_path(chat_id).exists(),
            )
            if entry is not None:
                entries.append(entry)
        entries.sort(key=lambda entry: (entry.saved_at, entry.state_path.name), reverse=True)
        return entries
