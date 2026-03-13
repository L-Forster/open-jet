from __future__ import annotations

import base64
import mimetypes
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from .runtime_limits import estimate_tokens


IMAGE_SUFFIXES = {
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".gif",
    ".bmp",
    ".tif",
    ".tiff",
}
IMAGE_TOKEN_COST = 512


def is_image_path(path: str | Path) -> bool:
    return Path(path).suffix.lower() in IMAGE_SUFFIXES


def resolve_local_path(raw: str) -> Path:
    text = raw.strip()
    if text.startswith(("'", '"')) and text.endswith(("'", '"')) and len(text) >= 2:
        text = text[1:-1].strip()
    if text.startswith("file://"):
        parsed = urlparse(text)
        text = unquote(parsed.path)
    return Path(text).expanduser().resolve()


def extract_pasted_image_paths(text: str) -> list[str]:
    paths: list[str] = []
    seen: set[str] = set()
    for line in text.splitlines():
        candidate = line.strip()
        if not candidate:
            continue
        try:
            resolved = resolve_local_path(candidate)
        except Exception:
            return []
        if not resolved.is_file() or not is_image_path(resolved):
            return []
        normalized = str(resolved)
        if normalized in seen:
            continue
        seen.add(normalized)
        paths.append(normalized)
    return paths


def build_user_content(text: str, image_paths: list[str] | None = None) -> str | list[dict[str, Any]]:
    normalized_text = text.strip()
    normalized_images = _normalize_image_paths(image_paths or [])
    if not normalized_images:
        return normalized_text

    lines = [normalized_text] if normalized_text else []
    for path in normalized_images:
        lines.append(f"Attached image: {path}")
    blocks: list[dict[str, Any]] = [{"type": "text", "text": "\n".join(lines).strip()}]
    for path in normalized_images:
        mime_type, _ = mimetypes.guess_type(path)
        blocks.append(
            {
                "type": "input_image",
                "path": path,
                "mime_type": mime_type or "application/octet-stream",
            }
        )
    return blocks


def content_to_plain_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content)

    text_parts = [
        str(block.get("text", "")).strip()
        for block in content
        if isinstance(block, dict) and str(block.get("type", "")).strip().lower() == "text"
    ]
    if any(text_parts):
        return "\n".join(part for part in text_parts if part).strip()

    parts: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        block_type = str(block.get("type", "")).strip().lower()
        if block_type in {"input_image", "image_url"}:
            path = str(block.get("path", "")).strip()
            if path:
                parts.append(f"[image: {path}]")
            else:
                parts.append("[image attached]")
    return "\n".join(part for part in parts if part).strip()


def estimate_message_content_tokens(content: Any) -> int:
    if isinstance(content, str):
        return estimate_tokens(content)
    if not isinstance(content, list):
        return estimate_tokens(str(content))

    total = 0
    for block in content:
        if not isinstance(block, dict):
            total += estimate_tokens(str(block))
            continue
        block_type = str(block.get("type", "")).strip().lower()
        if block_type == "text":
            total += estimate_tokens(str(block.get("text", "")))
        elif block_type in {"input_image", "image_url"}:
            total += IMAGE_TOKEN_COST
    return total


def runtime_content(content: Any) -> Any:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return content

    runtime_blocks: list[dict[str, Any]] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        block_type = str(block.get("type", "")).strip().lower()
        if block_type == "text":
            runtime_blocks.append({"type": "text", "text": str(block.get("text", ""))})
            continue
        if block_type == "input_image":
            path = str(block.get("path", "")).strip()
            if not path:
                continue
            mime_type = str(block.get("mime_type", "")).strip() or "application/octet-stream"
            runtime_blocks.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": _image_path_to_data_url(path=path, mime_type=mime_type),
                    },
                }
            )
            continue
        if block_type == "image_url":
            runtime_blocks.append(block)
    return runtime_blocks


def is_supported_message_content(content: Any) -> bool:
    if isinstance(content, str):
        return True
    if not isinstance(content, list):
        return False
    for block in content:
        if not isinstance(block, dict):
            return False
        block_type = str(block.get("type", "")).strip().lower()
        if block_type == "text":
            if not isinstance(block.get("text"), str):
                return False
            continue
        if block_type == "input_image":
            if not isinstance(block.get("path"), str):
                return False
            continue
        return False
    return True


def _normalize_image_paths(image_paths: list[str]) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for raw in image_paths:
        resolved = str(resolve_local_path(raw))
        if resolved in seen:
            continue
        seen.add(resolved)
        normalized.append(resolved)
    return normalized


def _image_path_to_data_url(*, path: str, mime_type: str) -> str:
    encoded = base64.b64encode(Path(path).read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"
