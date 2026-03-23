from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from .config import load_config
from .device_sources import sync_devices_registry
from .observation import ObservationStore
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


async def build_system_prompt(
    base_prompt: str,
    root: Path,
    *,
    cfg: Mapping[str, object] | None = None,
) -> str:
    snapshot = await load_persistent_memory(root)
    memory_prompt = snapshot.as_system_prompt()
    devices_prompt = _device_registry_prompt(root, cfg=cfg)
    sections = [base_prompt.strip(), memory_prompt, devices_prompt]
    return "\n\n".join(section for section in sections if section).strip()


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


def _device_registry_prompt(
    root: Path,
    *,
    cfg: Mapping[str, object] | None = None,
) -> str:
    registry_path = _write_device_registry(root, cfg=cfg)
    if registry_path is None:
        return ""
    return (
        f"IO device registry located in {registry_path}.\n"
        "Open it if wanting to interact with devices.\n"
        "Do not assume any device logs or payload files are already loaded.\n"
        "Available device tools include `device_list`, `camera_snapshot`, "
        "`microphone_record`, `microphone_set_enabled`, `gpio_read`, and `sensor_read`."
    )


def _write_device_registry(
    root: Path,
    *,
    cfg: Mapping[str, object] | None = None,
) -> Path | None:
    output_path = root / ".openjet" / "state" / "devices.md"
    try:
        resolved_cfg = cfg if isinstance(cfg, Mapping) else load_config()
        store = ObservationStore(root / ".openjet" / "state" / "observations")
        return sync_devices_registry(resolved_cfg, store=store, output_path=output_path)
    except Exception:
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(
                "# Devices\n\n"
                "Device discovery failed while refreshing this registry.\n",
                encoding="utf-8",
            )
            return output_path
        except Exception:
            return None
