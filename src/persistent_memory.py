from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from .app_paths import global_openjet_root, project_openjet_root
from .config import load_config
from .device_sources import ensure_devices_registry, format_device_registry_prompt
from .executor import edit_file
from .runtime_limits import MIN_TOKEN_BUDGET, derive_file_token_budget, estimate_tokens, read_memory_snapshot

DEFAULT_BASE_SYSTEM_PROMPT = """You are OpenJet, a local terminal AI assistant.
- Be concise, direct, and practical.
- Follow repository conventions and inspect real files before making changes.
- Never assume libraries, frameworks, commands, or paths without checking the repo.
- For coding tasks, follow this loop: inspect -> implement narrowly -> verify.
- After code edits, run focused local verification (tests/lint/typecheck/build) before claiming success.
- For complex multi-step tasks, create and maintain todos with todo_write.
- For simple tasks, do not create unnecessary todos.
- Mark completed todo items with todo_complete and clear irrelevant todos with todo_clear.
- Treat user memory as user-owned instructions and preferences.
- Treat agent memory as learned operational context.
- Use device ids and file paths exactly as discovered.
- Do not assume device logs, files, or outputs are already loaded unless they have been explicitly read."""

_SECRET_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----", re.IGNORECASE),
    re.compile(r"\b(api[_ -]?key|token|password|passwd|secret)\b\s*[:=]", re.IGNORECASE),
    re.compile(r"\bghp_[A-Za-z0-9]{20,}\b"),
    re.compile(r"\bsk-[A-Za-z0-9]{16,}\b"),
)


@dataclass(frozen=True)
class PersistentMemorySnapshot:
    global_user: str = ""
    global_agent: str = ""
    project_user: str = ""
    project_agent: str = ""

    @property
    def user(self) -> str:
        return self.project_user

    @property
    def agent(self) -> str:
        return self.project_agent

    def read(self, *, location: str, scope: str) -> str:
        normalized_location = normalize_location(location)
        normalized_scope = normalize_scope(scope)
        key = f"{normalized_location}_{normalized_scope}"
        return str(getattr(self, key, ""))

    def as_system_prompt(self) -> str:
        sections: list[str] = []
        scope_labels = (
            ("global_user", "Global user memory"),
            ("global_agent", "Global agent memory"),
            ("project_user", "Local user memory"),
            ("project_agent", "Local agent memory"),
        )
        for attr, label in scope_labels:
            content = str(getattr(self, attr, "")).strip()
            if content:
                sections.append(f"{label}:\n{content}")
        return "\n\n".join(section for section in sections if section).strip()


def normalize_scope(scope: str) -> str:
    normalized = scope.strip().lower()
    if normalized in {"user", "preferences", "prefs"}:
        return "user"
    if normalized in {"agent", "memory", "environment"}:
        return "agent"
    raise ValueError("scope must be one of: user, agent")


def normalize_location(location: str) -> str:
    normalized = location.strip().lower()
    if normalized in {"project", "local", "repo", "cwd", ""}:
        return "project"
    if normalized in {"global", "shared", "system"}:
        return "global"
    raise ValueError("location must be one of: global, local")


def memory_file_path(
    root: Path,
    scope: str,
    *,
    location: str = "project",
    global_root: Path | None = None,
) -> Path:
    normalized_scope = normalize_scope(scope)
    normalized_location = normalize_location(location)
    filename = "USER.md" if normalized_scope == "user" else "MEMORY.md"
    target_root = _memory_location_root(root, location=normalized_location, global_root=global_root)
    return target_root / "memories" / filename


async def load_persistent_memory(
    root: Path,
    *,
    global_root: Path | None = None,
) -> PersistentMemorySnapshot:
    return PersistentMemorySnapshot(
        global_user=await _read_bounded_memory(
            memory_file_path(root, "user", location="global", global_root=global_root)
        ),
        global_agent=await _read_bounded_memory(
            memory_file_path(root, "agent", location="global", global_root=global_root)
        ),
        project_user=await _read_bounded_memory(
            memory_file_path(root, "user", location="project", global_root=global_root)
        ),
        project_agent=await _read_bounded_memory(
            memory_file_path(root, "agent", location="project", global_root=global_root)
        ),
    )


async def update_persistent_memory(
    root: Path,
    *,
    scope: str,
    action: str,
    content: str = "",
    location: str = "project",
    global_root: Path | None = None,
) -> str:
    normalized_scope = normalize_scope(scope)
    normalized_location = normalize_location(location)
    normalized_action = action.strip().lower()
    path = memory_file_path(root, normalized_scope, location=normalized_location, global_root=global_root)
    current = await _read_bounded_memory(path)
    body = content.strip()

    if normalized_action == "read":
        return current or "(empty)"
    if normalized_action == "clear":
        _write_memory(path, "")
        return f"Cleared {normalized_location} {path.name}. Changes apply to future sessions."
    if normalized_action == "replace":
        sanitized = _sanitize_memory_body(body)
        if body and not sanitized:
            return f"Skipped replace for {normalized_location} {path.name}: content looked secret or unsafe."
        written = _write_memory(path, sanitized)
        return (
            f"Replaced {normalized_location} {path.name}. "
            f"stored_tokens~{estimate_tokens(written)}. Changes apply to future sessions."
        )
    if normalized_action == "append":
        sanitized = _sanitize_memory_body(body)
        if body and not sanitized:
            return f"Skipped append to {normalized_location} {path.name}: content looked secret or unsafe."
        merged = _merge_memory_content(current, sanitized)
        written = _write_memory(path, merged)
        return (
            f"Appended to {normalized_location} {path.name}. "
            f"stored_tokens~{estimate_tokens(written)}. Changes apply to future sessions."
        )
    raise ValueError("action must be one of: read, append, replace, clear")


async def append_persistent_memory_bullet(
    root: Path,
    *,
    scope: str,
    content: str,
    location: str = "project",
    global_root: Path | None = None,
) -> str:
    normalized_scope = normalize_scope(scope)
    normalized_location = normalize_location(location)
    sanitized = _sanitize_memory_body(content)
    if content.strip() and not sanitized:
        path = memory_file_path(root, normalized_scope, location=normalized_location, global_root=global_root)
        return f"Skipped append to {normalized_location} {path.name}: content looked secret or unsafe."
    bullet = _normalize_memory_bullet(sanitized)
    path = memory_file_path(root, normalized_scope, location=normalized_location, global_root=global_root)
    current = await _read_bounded_memory(path)
    merged = _merge_memory_content(current, bullet)
    if merged == current.strip():
        return f"Skipped append to {normalized_location} {path.name}: content already stored."

    if not path.exists():
        written = _write_memory(path, merged)
        return (
            f"Appended to {normalized_location} {path.name}. "
            f"stored_tokens~{estimate_tokens(written)}. Changes apply to future sessions."
        )

    current_disk = path.read_text(encoding="utf-8", errors="replace")
    replacement = merged + ("\n" if merged else "")
    result = await edit_file(
        str(path),
        old_string=current_disk,
        new_string=replacement,
        return_result=True,
    )
    if not result.ok:
        return f"Error editing {path.name}: {result.output}"
    written = _clip_text_to_dynamic_budget(merged)
    return (
        f"Appended to {normalized_location} {path.name}. "
        f"stored_tokens~{estimate_tokens(written)}. Changes apply to future sessions."
    )


async def build_system_prompt(
    base_prompt: str,
    root: Path,
    *,
    cfg: Mapping[str, object] | None = None,
    global_root: Path | None = None,
) -> str:
    resolved_cfg = cfg if isinstance(cfg, Mapping) else load_config()
    resolved_base_prompt = _resolve_base_system_prompt(base_prompt, cfg=resolved_cfg)
    snapshot = await load_persistent_memory(root, global_root=global_root)
    memory_prompt = snapshot.as_system_prompt()
    devices_prompt = _device_registry_prompt(root, cfg=resolved_cfg)
    sections = [resolved_base_prompt, memory_prompt, devices_prompt]
    return "\n\n".join(section for section in sections if section).strip()


async def _read_bounded_memory(path: Path) -> str:
    if not path.exists():
        return ""
    try:
        text = path.read_text(encoding="utf-8").strip()
    except Exception:
        return ""
    return _clip_text_to_dynamic_budget(text)


def _memory_location_root(
    root: Path,
    *,
    location: str,
    global_root: Path | None = None,
) -> Path:
    normalized_location = normalize_location(location)
    if normalized_location == "global":
        return Path(global_root or global_openjet_root()).expanduser().resolve()
    return project_openjet_root(root)


def _write_memory(path: Path, content: str) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    bounded = _clip_text_to_dynamic_budget(content.strip())
    path.write_text(bounded + ("\n" if bounded else ""), encoding="utf-8")
    return bounded


def _sanitize_memory_body(content: str) -> str:
    body = content.strip()
    if not body:
        return ""
    if any(pattern.search(body) for pattern in _SECRET_PATTERNS):
        return ""
    kept_lines = [line.rstrip() for line in body.splitlines() if line.strip()]
    if not kept_lines:
        return ""
    safe_lines = [line for line in kept_lines if not any(pattern.search(line) for pattern in _SECRET_PATTERNS)]
    return "\n".join(safe_lines).strip()


def _normalize_memory_bullet(content: str) -> str:
    body = " ".join(content.strip().split())
    if not body:
        return ""
    return body if body.startswith("- ") else f"- {body.lstrip('-* ').strip()}"


def _merge_memory_content(current: str, incoming: str) -> str:
    if not incoming:
        return current.strip()
    current_blocks = [block.strip() for block in re.split(r"\n\s*\n", current.strip()) if block.strip()]
    current_keys = {_normalize_memory_block(block) for block in current_blocks}
    for block in re.split(r"\n\s*\n", incoming.strip()):
        cleaned = block.strip()
        if not cleaned:
            continue
        normalized = _normalize_memory_block(cleaned)
        if normalized in current_keys:
            continue
        current_blocks.append(cleaned)
        current_keys.add(normalized)
    return "\n\n".join(current_blocks).strip()


def _normalize_memory_block(block: str) -> str:
    return "\n".join(line.strip() for line in block.splitlines() if line.strip()).lower()


def _clip_text_to_dynamic_budget(text: str) -> str:
    if not text:
        return ""
    mem = read_memory_snapshot()
    mem_available_mb = mem.available_mb if mem else None
    token_budget = max(MIN_TOKEN_BUDGET, derive_file_token_budget(mem_available_mb))
    if estimate_tokens(text) <= token_budget:
        return text

    prefix = text[: max(128, token_budget)]
    suffix = text[-(token_budget * 3) :]
    candidate = f"{prefix}\n\n...[persistent memory truncated]...\n\n{suffix}".strip()
    while estimate_tokens(candidate) > token_budget and len(suffix) > 64:
        drop = max(32, int(len(suffix) * 0.15))
        if drop >= len(suffix):
            break
        suffix = suffix[drop:]
        candidate = f"{prefix}\n\n...[persistent memory truncated]...\n\n{suffix}".strip()
    if estimate_tokens(candidate) <= token_budget:
        return candidate

    suffix_only = text[-(token_budget * 4) :]
    candidate = f"...[persistent memory truncated]\n{suffix_only}"
    while estimate_tokens(candidate) > token_budget and suffix_only:
        drop = max(1, int(len(suffix_only) * 0.15))
        if drop >= len(suffix_only):
            break
        suffix_only = suffix_only[drop:]
        candidate = f"...[persistent memory truncated]\n{suffix_only}"
    return candidate


def _device_registry_prompt(
    root: Path,
    *,
    cfg: Mapping[str, object] | None = None,
) -> str:
    registry_path = ensure_devices_registry(root, cfg=cfg)
    if registry_path is None:
        return ""
    return format_device_registry_prompt(registry_path)


def _resolve_base_system_prompt(
    base_prompt: str,
    *,
    cfg: Mapping[str, object] | None = None,
) -> str:
    explicit = str(base_prompt or "").strip()
    if explicit:
        return explicit
    configured = ""
    if isinstance(cfg, Mapping):
        configured = str(cfg.get("system_prompt", "") or "").strip()
    return configured or DEFAULT_BASE_SYSTEM_PROMPT
