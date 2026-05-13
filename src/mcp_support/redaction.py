from __future__ import annotations

from typing import Mapping


SECRET_KEY_FRAGMENTS = (
    "api_key",
    "apikey",
    "auth",
    "bearer",
    "credential",
    "password",
    "private",
    "secret",
    "token",
)


def looks_secret_key(key: str) -> bool:
    lowered = str(key or "").strip().lower()
    return any(fragment in lowered for fragment in SECRET_KEY_FRAGMENTS)


def looks_secret_value(value: object) -> bool:
    text = str(value or "").strip()
    if not text:
        return False
    if text.lower().startswith(("bearer ", "token ")):
        return True
    return len(text) >= 32 and " " not in text


def redact_secret_value(key: str, value: object) -> str:
    if looks_secret_key(key) or looks_secret_value(value):
        return "<redacted>"
    return str(value)


def redact_mapping(values: Mapping[str, object]) -> dict[str, str]:
    return {str(key): redact_secret_value(str(key), value) for key, value in values.items()}


def redact_text(text: object, values: Mapping[str, object]) -> str:
    redacted = str(text or "")
    for key, value in values.items():
        raw = str(value or "")
        if not raw:
            continue
        if looks_secret_key(str(key)) or looks_secret_value(raw):
            redacted = redacted.replace(raw, "<redacted>")
    return redacted
