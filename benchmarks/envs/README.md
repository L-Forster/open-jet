Environment templates for realistic benchmark cases.

Each directory is a miniature codebase representing an Open-Jet deployment
context:
- `satellite_ops_service`: air-gapped telemetry and operator workflows
- `manufacturing_inspection_pipeline`: on-prem quality inspection
- `robotics_control_stack`: disconnected robotics command layer
- `edge_runtime_diagnostics`: runtime health and memory debugging

The benchmark runner clones one of these templates into an isolated per-run
eval environment before executing the agent.
