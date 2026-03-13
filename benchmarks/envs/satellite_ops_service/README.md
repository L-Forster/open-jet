# Satellite Telemetry Ops Service

Air-gapped telemetry summarization service for orbital assets.

Constraints:
- intermittent uplink windows
- no cloud dependency
- operator notes must remain on-device

Main components:
- `src/telemetry_ingest.py`
- `src/uplink_policy.py`
- `src/summarizer.py`
- `configs/ops.yaml`
- `docs/architecture.md`
