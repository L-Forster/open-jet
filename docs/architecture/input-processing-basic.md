# Input Processing Basic

This is the temporary processing layer above `src/peripherals/`.

It keeps the first edge-agent path small:

- save camera frames to a stable local store
- transcribe microphone clips locally by default, otherwise fall back to speech detection when transcription is unavailable
- append GPIO and sensor state into a rolling text buffer
- convert processed observations into agent-ready text or image content

## Layer Split

- `src/peripherals/`: raw device discovery and capture
- `src/observation/store.py`: payload and event persistence
- `src/observation/processors.py`: basic input processing
- `src/observation/bridge.py`: conversion into agent-readable content

## Temporary Scope

This layer does not implement:

- continuous workflow runners
- realtime streaming into the chat loop
- video reasoning from full streams

It is only the minimum needed to bridge local device input into the model:

- camera -> saved frame -> image input
- mic -> local transcription by default, or speech detection fallback -> text event
- GPIO/sensor state -> text buffer -> text input

## File Layout

Saved data lives under:

```text
.openjet/state/observations/
  payloads/
  sources/<source_id>/events.jsonl
  sources/<source_id>/latest.json
  sources/<source_id>/gpio-buffer.txt
```

## Agent Bridge

The bridge layer is intentionally simple.

- image observations become `build_user_content(..., [image_path])`
- text observations stay text
- buffer-backed text observations load the recent buffer contents

That is enough to wire Qwen image input and text-based sensor/audio summaries without pulling raw device streams straight into the prompt.
