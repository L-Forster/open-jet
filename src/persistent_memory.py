from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .runtime_limits import MIN_TOKEN_BUDGET, derive_file_token_budget, estimate_tokens, read_memory_snapshot


@dataclass(frozen=True)
class PersistentMemorySnapshot:
    user: str
    agent: str

    def as_system_prompt(self) -> str:
        sections: list[str] = []
        if self.user:
            sections.append(f"Persistent user preferences:\n{self.user}")
        if self.agent:
            sections.append(f"Persistent agent memory:\n{self.agent}")
        return "\n\n".join(sections).strip()


def memory_file_path(root: Path, scope: str) -> Path:
    normalized = normalize_scope(scope)
    filename = "USER.md" if normalized == "user" else "MEMORY.md"
    return root / ".openjet" / "memories" / filename


def normalize_scope(scope: str) -> str:
    normalized = scope.strip().lower()
    if normalized in {"user", "preferences", "prefs"}:
        return "user"
    if normalized in {"agent", "memory", "environment"}:
        return "agent"
    raise ValueError("scope must be one of: user, agent")


async def load_persistent_memory(root: Path) -> PersistentMemorySnapshot:
    return PersistentMemorySnapshot(
        user=await _read_bounded_memory(memory_file_path(root, "user")),
        agent=await _read_bounded_memory(memory_file_path(root, "agent")),
    )


async def update_persistent_memory(
    root: Path,
    *,
    scope: str,
    action: str,
    content: str = "",
) -> str:
    normalized_scope = normalize_scope(scope)
    normalized_action = action.strip().lower()
    path = memory_file_path(root, normalized_scope)
    current = await _read_bounded_memory(path)
    body = content.strip()

    if normalized_action == "read":
        return current or "(empty)"
    if normalized_action == "clear":
        _write_memory(path, "")
        return f"Cleared {path.name}. Changes apply to future sessions."
    if normalized_action == "replace":
        written = _write_memory(path, body)
        return (
            f"Replaced {path.name}. "
            f"stored_tokens~{estimate_tokens(written)}. Changes apply to future sessions."
        )
    if normalized_action == "append":
        merged = f"{current}\n\n{body}".strip() if current and body else (body or current)
        written = _write_memory(path, merged)
        return (
            f"Appended to {path.name}. "
            f"stored_tokens~{estimate_tokens(written)}. Changes apply to future sessions."
        )
    raise ValueError("action must be one of: read, append, replace, clear")


async def build_system_prompt(base_prompt: str, root: Path) -> str:
    snapshot = await load_persistent_memory(root)
    memory_prompt = snapshot.as_system_prompt()
    if not memory_prompt:
        return base_prompt
    if not base_prompt.strip():
        return memory_prompt
    return f"{base_prompt.rstrip()}\n\n{memory_prompt}"


async def _read_bounded_memory(path: Path) -> str:
    if not path.exists():
        return ""
    try:
        text = path.read_text(encoding="utf-8").strip()
    except Exception:
        return ""
    return _clip_text_to_dynamic_budget(text)


def _write_memory(path: Path, content: str) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    bounded = _clip_text_to_dynamic_budget(content.strip())
    path.write_text(bounded + ("\n" if bounded else ""), encoding="utf-8")
    return bounded


def _clip_text_to_dynamic_budget(text: str) -> str:
    if not text:
        return ""
    mem = read_memory_snapshot()
    mem_available_mb = mem.available_mb if mem else None
    token_budget = max(MIN_TOKEN_BUDGET, derive_file_token_budget(mem_available_mb))
    if estimate_tokens(text) <= token_budget:
        return text

    suffix = text[-(token_budget * 4) :]
    candidate = f"...[persistent memory truncated]\n{suffix}"
    while estimate_tokens(candidate) > token_budget and suffix:
        drop = max(1, int(len(suffix) * 0.15))
        if drop >= len(suffix):
            break
        suffix = suffix[drop:]
        candidate = f"...[persistent memory truncated]\n{suffix}"
    return candidate
