from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from ..multimodal import build_user_content
from ..peripherals.types import Observation, ObservationModality
from .store import ObservationStore


def observation_to_agent_content(
    observation: Observation,
    *,
    prompt_text: str = "",
    store: ObservationStore | None = None,
    max_chars: int = 4000,
) -> str | list[dict[str, Any]]:
    return observations_to_agent_content(
        [observation],
        prompt_text=prompt_text,
        store=store,
        max_chars=max_chars,
    )


def observations_to_agent_content(
    observations: Sequence[Observation],
    *,
    prompt_text: str = "",
    store: ObservationStore | None = None,
    max_chars: int = 4000,
) -> str | list[dict[str, Any]]:
    text_parts: list[str] = []
    image_paths: list[str] = []

    prompt = prompt_text.strip()
    if prompt:
        text_parts.append(prompt)

    for observation in observations:
        if observation.modality is ObservationModality.IMAGE and observation.payload_ref:
            payload_path = Path(observation.payload_ref)
            if payload_path.is_file():
                image_paths.append(str(payload_path))
            text_parts.append(observation.summary)
            continue
        text_parts.append(_text_context_for_observation(observation, store=store, max_chars=max_chars))

    merged_text = "\n\n".join(part for part in text_parts if part).strip()
    return build_user_content(merged_text, image_paths or None)


def _text_context_for_observation(
    observation: Observation,
    *,
    store: ObservationStore | None,
    max_chars: int,
) -> str:
    payload_ref = observation.payload_ref
    if payload_ref and _is_text_buffer_path(payload_ref) and Path(payload_ref).is_file():
        text = store.read_text_buffer(payload_ref, max_chars=max_chars) if store else Path(payload_ref).read_text(encoding="utf-8")[-max_chars:]
        return f"{observation.summary}\n\n{text.strip()}".strip()
    return observation.summary


def _is_text_buffer_path(path: str) -> bool:
    return Path(path).suffix.lower() in {".txt", ".log", ".md", ".jsonl"}
