# Edge Inputs Phase 1

This document defines the first implementation slice for turning OpenJet into a real-world edge agent runtime: input devices and data ingestion on Ubuntu systems.

Phase 1 does not add full background workflows, sub-agents, or voice conversations yet. It establishes the smallest modular device-facing layer those later features will depend on.

## Scope

Phase 1 covers:

- hardware-agnostic device discovery on Ubuntu
- a small set of supported input and voice I/O device classes
- normalization of raw device data into a shared observation shape
- explicit boundaries between device adapters, observation normalization, and skills
- a test plan that can be implemented incrementally

Phase 1 does not cover:

- full realtime workflow orchestration
- autonomous actuation policies
- complex robotics middleware
- raw video or audio streaming directly into the main chat loop

## Design Principles

- Keep the current TUI and SDK interaction model intact.
- Treat devices as local data sources, not as prompt text by default.
- Convert raw input into compact observations before it enters model context.
- Keep hardware access in code and behavior policy in skills.
- Prefer Ubuntu-wide primitives over Jetson-specific APIs where possible.
- Keep the first implementation small and plug-and-play.

## Supported Device Classes

Phase 1 should support these device families first.

### Cameras

- USB UVC cameras
- CSI cameras that appear through V4L2 on Ubuntu
- any camera exposed as `/dev/video*`

Primary access path:

- V4L2 device discovery
- `ffmpeg` or OpenCV-backed frame capture

Phase 1 behavior:

- capture still frame
- sample one frame from a camera on demand
- save frame locally and optionally pass it through the existing image-input path

### Microphones

- USB microphones
- USB audio interfaces
- built-in analog microphones exposed through ALSA/PipeWire/PulseAudio

Primary access path:

- PipeWire or PulseAudio device enumeration when available
- ALSA fallback when PipeWire/PulseAudio metadata is unavailable

Phase 1 behavior:

- detect available input devices
- record short local clips
- save clip locally for later STT or workflow processing

### Speakers

Speakers are output devices, but they must be discovered in the same voice I/O layer as microphones.

- USB speakers
- USB audio interfaces
- built-in analog output devices exposed through ALSA/PipeWire/PulseAudio

Primary access path:

- PipeWire or PulseAudio sink enumeration when available
- ALSA fallback

Phase 1 behavior:

- detect available output devices
- validate playback device availability
- reserve actual TTS and bidirectional voice flows for later phases

### Sensors

Phase 1 should start with simple sensors that are easy to explain and test.

- temperature/humidity sensors over I2C
- temperature/humidity sensors exposed via USB serial adapters
- GPIO-backed digital sensors where Linux access is available through `libgpiod`

Primary access path:

- `/dev/i2c-*` for I2C buses
- `/dev/ttyUSB*` and `/dev/ttyACM*` for USB serial
- `/dev/gpiochip*` for GPIO on modern Ubuntu systems

Phase 1 behavior:

- detect candidate buses and devices
- read sensor values through adapter code
- normalize readings into shared observation records

## Hardware-Agnostic Discovery Strategy

OpenJet should not assume Jetson. Ubuntu support comes first.

Discovery should be organized by transport, not by board vendor:

- video: `/dev/video*`, V4L2 metadata, capture backend check
- audio input/output: PipeWire or PulseAudio device list, ALSA fallback
- GPIO: `libgpiod` availability plus `/dev/gpiochip*`
- I2C: `/dev/i2c-*` presence and readable bus scan hooks
- serial: `/dev/ttyUSB*`, `/dev/ttyACM*`

This keeps detection portable across:

- Jetson
- x86 Ubuntu
- AMD or Intel mini PCs
- generic ARM Ubuntu systems

## Data Model

Phase 1 should introduce a shared observation shape. All device inputs should normalize into one of these modalities:

- `text`
- `image`
- `structured_state`
- `audio_clip`

Recommended observation fields:

- `source_id`
- `source_type`
- `transport`
- `timestamp`
- `modality`
- `summary`
- `payload_ref`
- `metadata`
- `changed`

Examples:

- camera frame -> `image`
- temp/humidity sample -> `structured_state` plus short `summary`
- mic recording -> `audio_clip`

## Architecture Split

Do not implement one skill file per device.

The split should be:

- `src/peripherals/types.py`: shared device and observation types
- `src/peripherals/discovery.py`: transport-based discovery adapters and registry
- `src/peripherals/camera.py`: camera capture helper
- `src/peripherals/audio.py`: microphone capture helper
- `src/peripherals/sensors.py`: sensor normalization helpers
- `skills/` or workflow files: high-level behaviors that consume observations

Recommended Phase 1 module layout:

- `src/peripherals/`

## OpenJet Integration

Phase 1 now integrates with existing systems instead of bypassing them.

- Reuse the image-content path in `src/multimodal.py` for camera frames.
- Keep device operations behind explicit tools in `src/runtime_protocol.py` and `src/tool_executor.py`.
- Allow the TUI to reference device ids directly in prompts, for example `@camera0`, `@mic0`, or `@gpio0`.
- Keep the main TUI loop conversational. Device capture is on-demand in the current implementation.

## Modularity

Phase 1 should be plug-and-play:

- discovery runs through a small registry of adapters
- each transport adapter can be added independently
- normalization uses one shared observation type
- higher-level features depend on observation output, not raw device APIs

That means a new input transport should be addable by introducing one adapter and registering it, without rewriting the TUI or the agent loop.

## Tool Surface For Phase 1

Phase 1 should stay small.

- `device_list`
- `camera_snapshot`
- `microphone_record`
- `microphone_set_enabled`
- `gpio_read`
- `sensor_read`

Speaker discovery belongs in `device_list`, but speaker playback can wait until the next phase. `sensor_read` is currently a legacy alias for `gpio_read`.

## Setup Requirements

`openjet setup` should eventually detect and validate:

- available cameras
- available audio input devices
- available audio output devices
- available GPIO chips
- available I2C buses
- available USB serial devices

Phase 1 only needs the spec and adapter boundaries defined clearly. Device setup UX can be added after the adapter layer exists.

## Test Plan

Phase 1 should be added to the test suite as small slices.

### Slice 1: discovery

- enumerate mock video devices
- enumerate mock audio devices
- enumerate mock GPIO chips
- enumerate mock I2C buses
- enumerate mock serial devices
- verify unsupported transports degrade cleanly
- verify a custom discovery adapter can plug into the registry

### Slice 2: normalization

- camera capture result normalizes to `image`
- sensor reading normalizes to `structured_state`
- microphone capture normalizes to `audio_clip`
- all observations include stable source metadata

### Slice 3: tool boundaries

- `device_list` reports known devices
- `camera_snapshot` returns saved frame metadata
- `sensor_read` returns normalized reading
- invalid or missing devices fail with clear errors

### Slice 4: Ubuntu fallback behavior

- PipeWire unavailable falls back cleanly
- PulseAudio unavailable falls back cleanly
- ALSA-only systems still detect audio devices where possible
- missing `libgpiod` or missing device nodes report unavailability, not crashes

## Acceptance Criteria

Phase 1 is complete when:

- OpenJet can discover cameras, microphones, speakers, and simple sensor transports on Ubuntu
- discovery is transport-based, not Jetson-specific
- at least one camera path can be normalized into an image observation
- microphone recordings can be normalized into short text summaries via bundled local transcription, or via local speech detection when transcription is unavailable
- GPIO-backed sources can be normalized into text observations and rolling buffers
- the feature is covered by unit tests with mocked device environments
- the architecture clearly separates adapters, normalization, and skills

## Follow-On Phases

Later phases can build on this:

- Phase 2: broaden sensor capture beyond GPIO text snapshots
- Phase 3: richer TUI affordances beyond direct `@device_id` tagging
- Phase 4: background workflows
- Phase 5: sub-agents and scheduler integration
- Phase 6: user docs and end-to-end demos
