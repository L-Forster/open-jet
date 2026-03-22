"""Thin observation processing layer above raw peripherals.

This layer keeps raw device access in ``src/peripherals`` and adds only the
temporary processing needed for the first edge-agent path:
saved frames, optional local audio transcription or speech detection, and GPIO
text buffering.
"""

from .bridge import observation_to_agent_content, observations_to_agent_content
from .processors import (
    append_gpio_text_buffer,
    detect_speech_activity,
    process_audio_observation,
    provision_default_faster_whisper_model,
    save_frame_observation,
    transcribe_audio_clip,
)
from .store import ObservationStore

__all__ = [
    "ObservationStore",
    "append_gpio_text_buffer",
    "detect_speech_activity",
    "observation_to_agent_content",
    "observations_to_agent_content",
    "process_audio_observation",
    "provision_default_faster_whisper_model",
    "save_frame_observation",
    "transcribe_audio_clip",
]
