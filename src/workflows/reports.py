from __future__ import annotations

from pathlib import Path

from .runner import WorkflowRunResult
from .specs import WorkflowSpec
from .state import workflow_last_run_path, workflow_runs_dir


def render_workflow_run_report(spec: WorkflowSpec, result: WorkflowRunResult) -> str:
    lines = [
        "# Workflow Run",
        "",
        f"- workflow: `{result.name}`",
        f"- workflow_path: `{result.spec_path}`",
        f"- started_at: `{result.started_at}`",
        f"- finished_at: `{result.finished_at}`",
        f"- status: `{'success' if result.success else 'failure'}`",
        f"- binding_source: `{result.binding_source}`",
        f"- registry_path: `{result.registry_path or 'none'}`",
        "",
    ]
    lines.append("## Bound Devices")
    if result.bound_devices:
        lines.extend(f"- `{device_id}`" for device_id in result.bound_devices)
    else:
        lines.append("- none")
    lines.append("")
    lines.append("## Workflow Instructions")
    lines.append(spec.body.strip() or "(empty workflow body)")
    lines.append("")
    if result.response_text:
        lines.append("## Result")
        lines.append(result.response_text)
        lines.append("")
    if result.error:
        lines.append("## Error")
        lines.append(result.error)
        lines.append("")
    lines.append("## Tools Used")
    if result.tool_details:
        lines.extend(f"- {detail}" for detail in result.tool_details)
    else:
        lines.append("- none")
    lines.append("")
    lines.append("## Observation Payloads")
    if result.payload_paths:
        lines.extend(f"- `{path}`" for path in result.payload_paths)
    else:
        lines.append("- none")
    lines.append("")
    if result.docs_loaded:
        lines.append("## Harness Docs")
        lines.extend(f"- {label}" for label in result.docs_loaded)
        lines.append("")
    if result.preloaded_files:
        lines.append("## Preloaded Files")
        lines.extend(f"- `{path}`" for path in result.preloaded_files)
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def write_workflow_run_report(root: Path, spec: WorkflowSpec, result: WorkflowRunResult) -> Path:
    rendered = render_workflow_run_report(spec, result)
    runs_dir = workflow_runs_dir(root, spec.name)
    runs_dir.mkdir(parents=True, exist_ok=True)
    run_path = runs_dir / f"{_timestamp_slug(result.started_at)}.md"
    run_path.write_text(rendered, encoding="utf-8")
    latest_path = workflow_last_run_path(root, spec.name)
    latest_path.parent.mkdir(parents=True, exist_ok=True)
    latest_path.write_text(rendered, encoding="utf-8")
    return run_path


def _timestamp_slug(value: str) -> str:
    return value.replace(":", "").replace("-", "").replace("+", "_").replace(".", "_")
