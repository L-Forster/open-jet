from __future__ import annotations

import json
import platform
import subprocess
import sys
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

from .agent import Agent
from .harness import HarnessState, build_turn_context, update_state_after_turn, update_state_for_user_message
from .multimodal import content_to_plain_text, estimate_message_content_tokens
from .runtime_limits import MemorySnapshot, estimate_tokens


@dataclass(frozen=True)
class ContextBenchmarkTurnInput:
    title: str
    state: HarnessState
    current_context_tokens: int | None
    memory_snapshot: MemorySnapshot | None
    note: str = ""
    effective_window: int | None = None
    layered_config: dict[str, Any] | None = None
    chat_messages: list[dict[str, Any]] | None = None
    system_prompt: str | None = None


@dataclass(frozen=True)
class ContextBenchmarkSuite:
    name: str
    description: str
    workspace: Path
    effective_window: int
    layered_config: dict[str, Any]
    turns: list[ContextBenchmarkTurnInput]


def available_context_suites() -> list[str]:
    return sorted(_CONTEXT_SUITE_BUILDERS)


def run_context_suite(name: str, *, output_root: Path | None = None) -> Path:
    builder = _CONTEXT_SUITE_BUILDERS.get(name)
    if builder is None:
        raise ValueError(f"Unknown context benchmark suite: {name}")

    root = output_root or (Path.cwd() / "benchmark_results" / "context")
    root.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    run_dir = root / f"{timestamp}_{name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    workspace = run_dir / "workspace"
    suite = builder(workspace)

    config_payload = {
        "benchmark_name": suite.name,
        "description": suite.description,
        "run_id": uuid.uuid4().hex[:12],
        "generated_at": timestamp,
        "git_commit": _git_commit(Path(__file__).resolve().parent.parent),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "effective_window": suite.effective_window,
        "layered_config": dict(suite.layered_config),
        "workspace": str(suite.workspace),
        "workspace_manifest": _workspace_manifest(suite.workspace),
    }

    turns_path = run_dir / "turns.jsonl"
    turn_records: list[dict[str, Any]] = []
    for index, turn in enumerate(suite.turns, start=1):
        start = time.perf_counter()
        effective_window = turn.effective_window or suite.effective_window
        layered_config = dict(suite.layered_config)
        layered_config.update(turn.layered_config or {})
        chat_history_profile = None
        current_context_tokens = turn.current_context_tokens
        token_source = "explicit"
        if turn.chat_messages is not None:
            chat_history_profile = _profile_chat_history(
                turn.chat_messages,
                system_prompt=turn.system_prompt,
            )
            current_context_tokens = int(chat_history_profile["persistent_context_tokens"])
            token_source = "chat_history"
        if current_context_tokens is None:
            raise ValueError(f"Turn {turn.title} must provide current_context_tokens or chat_messages.")
        context = build_turn_context(
            root=suite.workspace,
            state=turn.state,
            current_context_tokens=current_context_tokens,
            effective_window=effective_window,
            memory_snapshot=turn.memory_snapshot,
            layered_config=layered_config,
        )
        latency_ms = round((time.perf_counter() - start) * 1000.0, 3)
        record = {
            "turn_index": index,
            "title": turn.title,
            "note": turn.note,
            "effective_window": effective_window,
            "current_context_tokens": current_context_tokens,
            "current_context_token_source": token_source,
            "memory_snapshot": asdict(turn.memory_snapshot) if turn.memory_snapshot else None,
            "layered_config": layered_config,
            "turn_budget": asdict(context.budget),
            "state_summary": context.state_summary,
            "state_summary_tokens": context.state_summary_tokens,
            "chat_history_profile": chat_history_profile,
            "candidate_labels_in_order": [item["label"] for item in context.candidate_decisions],
            "candidate_metadata": context.candidate_decisions,
            "admitted_docs": list(context.docs_loaded),
            "skipped_docs": [item for item in context.candidate_decisions if not item.get("admitted")],
            "layer_token_totals": dict(context.layer_tokens),
            "layer_loaded_docs": {key: list(value) for key, value in context.layer_docs.items()},
            "budget_alerts": list(context.budget_alerts),
            "docs_tokens": context.docs_tokens,
            "latency_ms": latency_ms,
            "final_harness_context_summary": {
                "docs_loaded": list(context.docs_loaded),
                "docs_tokens": context.docs_tokens,
                "layer_tokens": dict(context.layer_tokens),
                "layer_docs": {key: list(value) for key, value in context.layer_docs.items()},
                "budget_alerts": list(context.budget_alerts),
            },
        }
        turn_records.append(record)
        with turns_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary = _build_run_summary(config_payload, turn_records)
    compare_ready = _compare_ready_metrics(summary, turn_records)

    (run_dir / "config.json").write_text(json.dumps(config_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "compare_ready_metrics.json").write_text(json.dumps(compare_ready, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "timeline.txt").write_text(_render_timeline(turn_records), encoding="utf-8")
    (run_dir / "summary.md").write_text(_render_summary_markdown(summary, turn_records), encoding="utf-8")
    return run_dir


def run_jetson_4k_suites(*, output_root: Path | None = None) -> list[Path]:
    runs: list[Path] = []
    for name in ("jetson_4k_baseline", "jetson_4k_ram_pressure", "jetson_4k_layer_compare"):
        runs.append(run_context_suite(name, output_root=output_root))
    return runs


def compare_context_runs(run_dirs: list[Path], *, markdown_path: Path | None = None) -> str:
    if len(run_dirs) < 2:
        raise ValueError("Need at least two context benchmark runs to compare.")

    loaded = [_load_context_run(path) for path in run_dirs]
    baseline = loaded[0]
    lines = ["CONTEXT BENCHMARK COMPARISON", ""]
    lines.append("Runs:")
    for item in loaded:
        lines.append(
            f"- {item['summary']['benchmark_name']}: turns={item['summary']['turn_count']} "
            f"latency_avg_ms={item['metrics']['latency_ms']['avg']} "
            f"churn_rate={item['metrics']['churn_rate']}"
        )
    lines.append("")
    lines.append("Metric deltas vs baseline:")
    for item in loaded[1:]:
        lines.extend(_compare_metric_block(baseline, item))
    lines.append("")
    lines.append("Admitted doc set deltas by turn:")
    for item in loaded[1:]:
        lines.extend(_compare_turn_doc_sets(baseline, item))

    rendered = "\n".join(lines).strip() + "\n"
    if markdown_path:
        markdown_path.write_text(_comparison_as_markdown(loaded), encoding="utf-8")
    return rendered


def summarize_context_run(run_dir: Path) -> str:
    loaded = _load_context_run(run_dir)
    summary = loaded["summary"]
    metrics = loaded["metrics"]
    lines = [
        f"{summary['benchmark_name']}: {summary['description']}",
        f"turns={summary['turn_count']} effective_window={summary['effective_window']}",
        f"latency_avg_ms={metrics['latency_ms']['avg']} churn_rate={metrics['churn_rate']}",
        f"alerts={metrics['budget_alert_turns']} file_context_survival={metrics['doc_category_survival']['file_context']}",
        f"recent_context_survival={metrics['doc_category_survival']['recent_context']}",
        f"skip_reasons={metrics['skip_reason_counts']}",
        f"run_dir={run_dir}",
    ]
    return "\n".join(lines)


def _load_context_run(run_dir: Path) -> dict[str, Any]:
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    metrics = json.loads((run_dir / "compare_ready_metrics.json").read_text(encoding="utf-8"))
    turns = [
        json.loads(line)
        for line in (run_dir / "turns.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return {"run_dir": run_dir, "summary": summary, "metrics": metrics, "turns": turns}


def _compare_metric_block(baseline: dict[str, Any], other: dict[str, Any]) -> list[str]:
    base_metrics = baseline["metrics"]
    other_metrics = other["metrics"]
    lines = [
        f"- {other['summary']['benchmark_name']} vs {baseline['summary']['benchmark_name']}:",
        f"  layer_usage={other_metrics['layer_usage_totals']} baseline={base_metrics['layer_usage_totals']}",
        f"  alerts={other_metrics['budget_alert_turns']} baseline={base_metrics['budget_alert_turns']}",
        f"  churn_rate={other_metrics['churn_rate']} baseline={base_metrics['churn_rate']}",
        f"  latency_avg_ms={other_metrics['latency_ms']['avg']} baseline={base_metrics['latency_ms']['avg']}",
        f"  file_context_survival={other_metrics['doc_category_survival']['file_context']} "
        f"baseline={base_metrics['doc_category_survival']['file_context']}",
        f"  recent_context_survival={other_metrics['doc_category_survival']['recent_context']} "
        f"baseline={base_metrics['doc_category_survival']['recent_context']}",
        f"  skip_reason_counts={other_metrics['skip_reason_counts']} baseline={base_metrics['skip_reason_counts']}",
    ]
    return lines


def _compare_turn_doc_sets(baseline: dict[str, Any], other: dict[str, Any]) -> list[str]:
    lines = [f"- {other['summary']['benchmark_name']}"]
    for base_turn, other_turn in zip(baseline["turns"], other["turns"]):
        base_docs = set(base_turn["admitted_docs"])
        other_docs = set(other_turn["admitted_docs"])
        added = sorted(other_docs - base_docs)
        removed = sorted(base_docs - other_docs)
        if not added and not removed:
            continue
        lines.append(
            f"  turn {other_turn['turn_index']}: added={added or ['none']} removed={removed or ['none']}"
        )
    return lines


def _comparison_as_markdown(loaded: list[dict[str, Any]]) -> str:
    baseline = loaded[0]
    lines = ["# Context Benchmark Comparison", ""]
    lines.append("## Runs")
    for item in loaded:
        lines.append(
            f"- **{item['summary']['benchmark_name']}**: turns={item['summary']['turn_count']}, "
            f"latency_avg_ms={item['metrics']['latency_ms']['avg']}, churn_rate={item['metrics']['churn_rate']}"
        )
    lines.append("")
    lines.append("## Deltas vs Baseline")
    for item in loaded[1:]:
        lines.extend(_compare_metric_block(baseline, item))
    lines.append("")
    lines.append("## Turn-by-Turn Admitted Docs")
    for item in loaded[1:]:
        lines.extend(_compare_turn_doc_sets(baseline, item))
    return "\n".join(lines).strip() + "\n"


def _build_run_summary(config_payload: dict[str, Any], turns: list[dict[str, Any]]) -> dict[str, Any]:
    last_turn = turns[-1] if turns else {}
    layer_totals = {"layer1": 0, "layer2": 0, "layer3": 0}
    skip_counts: dict[str, int] = {}
    budget_alert_turns = 0
    admitted_frequency: dict[str, int] = {}
    latency_values: list[float] = []
    for turn in turns:
        latency_values.append(float(turn["latency_ms"]))
        if turn["budget_alerts"]:
            budget_alert_turns += 1
        for layer, value in turn["layer_token_totals"].items():
            layer_totals[layer] = layer_totals.get(layer, 0) + int(value)
        for label in turn["admitted_docs"]:
            admitted_frequency[label] = admitted_frequency.get(label, 0) + 1
        for candidate in turn["candidate_metadata"]:
            reason = candidate.get("skipped_reason")
            if reason:
                skip_counts[reason] = skip_counts.get(reason, 0) + 1

    summary = {
        "benchmark_name": config_payload["benchmark_name"],
        "description": config_payload["description"],
        "run_id": config_payload["run_id"],
        "generated_at": config_payload["generated_at"],
        "git_commit": config_payload["git_commit"],
        "python_version": config_payload["python_version"],
        "platform": config_payload["platform"],
        "effective_window": config_payload["effective_window"],
        "layered_config": dict(config_payload["layered_config"]),
        "turn_count": len(turns),
        "layer_usage_totals": layer_totals,
        "budget_alert_turns": budget_alert_turns,
        "admitted_doc_frequency": admitted_frequency,
        "skip_reason_counts": skip_counts,
        "latency_ms": {
            "avg": round(sum(latency_values) / len(latency_values), 3) if latency_values else 0.0,
            "max": round(max(latency_values), 3) if latency_values else 0.0,
            "min": round(min(latency_values), 3) if latency_values else 0.0,
        },
        "final_context": last_turn.get("final_harness_context_summary", {}),
        "final_state_summary": last_turn.get("state_summary", ""),
        "workspace_manifest": config_payload["workspace_manifest"],
    }
    return summary


def _compare_ready_metrics(summary: dict[str, Any], turns: list[dict[str, Any]]) -> dict[str, Any]:
    previous_docs: set[str] = set()
    churn_values: list[float] = []
    category_survival = {
        "base_doc": 0,
        "role_doc": 0,
        "project_context": 0,
        "project_doc": 0,
        "file_context": 0,
        "recent_context": 0,
        "skill_doc": 0,
    }
    global_skip = 0
    per_layer_skip = 0
    for turn in turns:
        admitted = set(turn["admitted_docs"])
        union = admitted | previous_docs
        if union:
            churn_values.append(round(len(admitted ^ previous_docs) / len(union), 4))
        previous_docs = admitted
        categories_in_turn = {_doc_category(label) for label in admitted}
        for category in categories_in_turn:
            if category in category_survival:
                category_survival[category] += 1
        for candidate in turn["candidate_metadata"]:
            if candidate.get("skipped_reason") == "exceeds_remaining_global_budget":
                global_skip += 1
            if candidate.get("skipped_reason") == "exceeds_remaining_per_layer_budget":
                per_layer_skip += 1

    metrics = {
        "benchmark_name": summary["benchmark_name"],
        "turn_count": summary["turn_count"],
        "layer_usage_totals": summary["layer_usage_totals"],
        "budget_alert_turns": summary["budget_alert_turns"],
        "skip_reason_counts": dict(summary["skip_reason_counts"]),
        "doc_category_survival": category_survival,
        "latency_ms": dict(summary["latency_ms"]),
        "churn_rate": round(sum(churn_values) / len(churn_values), 4) if churn_values else 0.0,
        "turn_churn": churn_values,
        "global_budget_skips": global_skip,
        "per_layer_budget_skips": per_layer_skip,
    }
    return metrics


def _render_timeline(turns: list[dict[str, Any]]) -> str:
    previous_docs: set[str] = set()
    blocks: list[str] = []
    for turn in turns:
        admitted = set(turn["admitted_docs"])
        added = sorted(admitted - previous_docs)
        removed = sorted(previous_docs - admitted)
        skipped = [
            f"{item['label']}({item['skipped_reason']})"
            for item in turn["candidate_metadata"]
            if not item.get("admitted")
        ]
        history_line = ""
        profile = turn.get("chat_history_profile")
        if profile:
            top = [
                f"{item['role']}#{item['index']}:{item['loader_counted_tokens']}"
                for item in profile.get("top_loader_messages", [])
            ]
            history_line = (
                f"  history_messages={profile['message_count']} "
                f"loader_tokens={profile['persistent_context_tokens']} "
                f"tool_call_tokens={profile['tool_call_tokens_in_history']} "
                f"top_loader={top or ['none']}"
            )
        blocks.extend(
            [
                f"Turn {turn['turn_index']}: {turn['title']}",
                f"  context_tokens={turn['current_context_tokens']} usable={turn['turn_budget']['usable_prompt_budget']} "
                f"docs={turn['turn_budget']['docs_budget']}",
                *([history_line] if history_line else []),
                f"  layer_usage={turn['layer_token_totals']} layer_budget={{'layer1': {turn['turn_budget']['layer1_budget']}, "
                f"'layer2': {turn['turn_budget']['layer2_budget']}, 'layer3': {turn['turn_budget']['layer3_budget']}}}",
                f"  admitted={turn['admitted_docs'] or ['none']}",
                f"  skipped={skipped or ['none']}",
                f"  alerts={turn['budget_alerts'] or ['none']}",
                f"  delta_added={added or ['none']} delta_removed={removed or ['none']}",
                f"  latency_ms={turn['latency_ms']}",
                "",
            ]
        )
        previous_docs = admitted
    return "\n".join(blocks).rstrip() + "\n"


def _render_summary_markdown(summary: dict[str, Any], turns: list[dict[str, Any]]) -> str:
    metrics = _compare_ready_metrics(summary, turns)
    derived_turns = [
        turn for turn in turns if turn.get("current_context_token_source") == "chat_history"
    ]
    max_loader_tokens = max((turn["current_context_tokens"] for turn in derived_turns), default=0)
    max_tool_call_tokens = max(
        (
            int(turn["chat_history_profile"]["tool_call_tokens_in_history"])
            for turn in derived_turns
            if turn.get("chat_history_profile")
        ),
        default=0,
    )
    lines = [
        f"# {summary['benchmark_name']}",
        "",
        summary["description"],
        "",
        "## Run Summary",
        f"- turns: {summary['turn_count']}",
        f"- effective window: {summary['effective_window']}",
        f"- git commit: {summary['git_commit'] or 'unknown'}",
        f"- latency avg/max ms: {summary['latency_ms']['avg']} / {summary['latency_ms']['max']}",
        f"- budget alert turns: {summary['budget_alert_turns']}",
        f"- layer usage totals: {summary['layer_usage_totals']}",
        f"- churn rate: {metrics['churn_rate']}",
        f"- chat-history-derived turns: {len(derived_turns)}",
        f"- max loader-counted history tokens: {max_loader_tokens}",
        f"- max tool-call tokens in history: {max_tool_call_tokens}",
        "",
        "## Doc Survival",
        f"- file context turns: {metrics['doc_category_survival']['file_context']}",
        f"- recent context turns: {metrics['doc_category_survival']['recent_context']}",
        f"- skill doc turns: {metrics['doc_category_survival']['skill_doc']}",
        f"- base/role/project turns: "
        f"{metrics['doc_category_survival']['base_doc']}/"
        f"{metrics['doc_category_survival']['role_doc']}/"
        f"{metrics['doc_category_survival']['project_context']}",
        "",
        "## Skip Reasons",
        f"- {summary['skip_reason_counts'] or {'none': 0}}",
        "",
        "## Final Context",
        f"- admitted docs: {summary['final_context'].get('docs_loaded', [])}",
        f"- layer tokens: {summary['final_context'].get('layer_tokens', {})}",
        f"- alerts: {summary['final_context'].get('budget_alerts', [])}",
    ]
    return "\n".join(lines).strip() + "\n"


def _doc_category(label: str) -> str:
    if label == "project-context":
        return "project_context"
    if label == "agents/base.md":
        return "base_doc"
    if label.startswith("agents/"):
        return "role_doc"
    if label.startswith("projects/"):
        return "project_doc"
    if label.startswith("file-context:"):
        return "file_context"
    if label == "recent-context":
        return "recent_context"
    if label.startswith("skills/"):
        return "skill_doc"
    return "other"


def _git_commit(repo_root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    return result.stdout.strip() or None


def _workspace_manifest(root: Path) -> list[dict[str, Any]]:
    manifest: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(root).as_posix()
        manifest.append({"path": rel, "bytes": path.stat().st_size})
    return manifest


def _write_file(root: Path, relative_path: str, content: str) -> None:
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _skill_doc(*, tags: list[str], body: str, mode: str | None = None) -> str:
    lines = ["---", "tags:"]
    for tag in tags:
        lines.append(f"  - {tag}")
    if mode:
        lines.append(f"mode: {mode}")
    lines.extend(["---", body.strip()])
    return "\n".join(lines)


def _write_repo_fixture(
    root: Path,
    *,
    architecture_lines: list[str],
    role_docs: dict[str, str],
    project_doc: str,
    skills: dict[str, str] | None = None,
    repo_files: dict[str, str] | None = None,
) -> None:
    what_is_lines = [
        "- offline-first local agent",
        "- layered context admission under bounded prompt budgets",
    ]
    engineering_lines = [
        "- prefer narrow file context over broad repo loads",
        "- keep context stable across resumed work",
    ]
    hardware_lines = [
        "- Jetson-class memory pressure matters",
        "- 4k context windows are a primary target",
    ]
    agents_md = (
        "## What This Project Is\n"
        + "\n".join(what_is_lines)
        + "\n\n## Engineering Rules\n"
        + "\n".join(engineering_lines)
        + "\n\n## Hardware And Performance Assumptions\n"
        + "\n".join(hardware_lines)
        + "\n\n## Core Architecture\n"
        + "\n".join(architecture_lines)
        + "\n"
    )
    _write_file(root, "AGENTS.md", agents_md)
    _write_file(root, ".openjet/agents/base.md", "base harness guidance for edge Linux work.")
    for role_name, body in role_docs.items():
        _write_file(root, f".openjet/agents/{role_name}.md", body)
    _write_file(root, ".openjet/projects/default.md", project_doc)
    for relative_path, content in (repo_files or {}).items():
        _write_file(root, relative_path, content)
    for name, body in (skills or {}).items():
        _write_file(root, f".openjet/skills/{name}.md", body)


def _clone_state(state: HarnessState) -> HarnessState:
    return HarnessState.from_dict(state.to_dict())


def _turn_state(
    prompt: str,
    *,
    mode: str,
    files: list[str],
    preferred_skills: list[str] | None = None,
) -> HarnessState:
    return update_state_for_user_message(
        HarnessState(preferred_skills=list(preferred_skills or [])),
        prompt,
        mode=mode,
        files=files,
    )


def _tool_call(name: str, arguments: dict[str, Any], *, call_id: str) -> dict[str, Any]:
    return {
        "id": call_id,
        "type": "function",
        "function": {
            "name": name,
            "arguments": json.dumps(arguments, ensure_ascii=True),
        },
    }


def _chat_system_prompt() -> str:
    return (
        "You are open-jet. Work locally, keep context narrow, prefer tool results over guesses, "
        "and preserve only the most relevant state for the next coding turn."
    )


def _narrow_task_history() -> list[list[dict[str, Any]]]:
    system = {"role": "system", "content": _chat_system_prompt()}
    user1 = {
        "role": "user",
        "content": (
            "Need to adjust the layered harness budget in src/harness.py without blowing the 4k "
            "context budget. Read the budget flow, then make the smallest fix."
        ),
    }
    assistant1 = {
        "role": "assistant",
        "content": "I will inspect the harness budget path and then patch the narrowest branch.",
        "tool_calls": [
            _tool_call("read_file", {"path": "src/harness.py"}, call_id="call_read_harness"),
        ],
    }
    tool1 = {
        "role": "tool",
        "tool_call_id": "call_read_harness",
        "content": (
            "src/harness.py\n"
            "compute_turn_budget derives usable prompt budget from window, reserves, RAM factor, "
            "and current context tokens.\n"
            "build_turn_context injects a state summary first, then admits ordered docs if they fit.\n"
            "Layer 1 keeps project/base docs, layer 2 keeps role/file/skill docs, layer 3 keeps recent-context.\n"
            "Oversized docs are skipped instead of truncated."
        ),
    }
    assistant2 = {
        "role": "assistant",
        "content": (
            "The harness budget path is in src/harness.py. The summary is injected before layered docs, "
            "so any fix has to preserve that ordering while tightening budget arithmetic."
        ),
    }

    user2 = {
        "role": "user",
        "content": (
            "Also inspect src/context_index.py because file summaries affect which file-context docs "
            "show up when more files enter play."
        ),
    }
    assistant3 = {
        "role": "assistant",
        "content": "I will inspect the file summary resolver and keep the change set narrow.",
        "tool_calls": [
            _tool_call("read_file", {"path": "src/context_index.py"}, call_id="call_read_index"),
            _tool_call(
                "read_file",
                {"path": "tests/test_context_harness.py"},
                call_id="call_read_tests",
            ),
        ],
    }
    tool2 = {
        "role": "tool",
        "tool_call_id": "call_read_index",
        "content": (
            "src/context_index.py\n"
            "lookup_file_summary loads indexed file summaries and falls back to nearby tests. "
            "Related tests for src/harness.py include tests/test_harness.py and tests/test_context_harness.py. "
            "Project summary extraction reads AGENTS.md sections for architecture and engineering rules."
        ),
    }
    tool3 = {
        "role": "tool",
        "tool_call_id": "call_read_tests",
        "content": (
            "tests/test_context_harness.py\n"
            "Current tests cover layered ratios, docs budget, and candidate ordering. "
            "Need extra assertions for recent-context survival under pressure and for candidate starvation."
        ),
    }
    assistant4 = {
        "role": "assistant",
        "content": (
            "File summary resolution pulls in related tests once the scope expands. I should keep "
            "src/harness.py and src/context_index.py in play and then verify with focused tests."
        ),
    }

    user3 = {
        "role": "user",
        "content": (
            "Patch it, then run the focused harness tests. Keep the failure details visible if verification fails."
        ),
    }
    assistant5 = {
        "role": "assistant",
        "content": "I will patch the harness budget branch and run the focused test target.",
        "tool_calls": [
            _tool_call(
                "edit_file",
                {"path": "src/harness.py", "instruction": "tighten remaining budget handling"},
                call_id="call_edit_harness",
            ),
            _tool_call(
                "shell",
                {"command": "python -m unittest tests.test_context_harness"},
                call_id="call_run_tests",
            ),
        ],
    }
    tool4 = {
        "role": "tool",
        "tool_call_id": "call_edit_harness",
        "content": (
            "Applied edit to src/harness.py. Updated docs budget handling when remaining prompt headroom "
            "approaches the alert threshold. Preserved summary-first ordering and sequential doc admission."
        ),
    }
    tool5 = {
        "role": "tool",
        "tool_call_id": "call_run_tests",
        "content": (
            "python -m unittest tests.test_context_harness\n"
            "FAIL: test_debug_loop_preserves_recent_context_until_pressure_drops_it\n"
            "AssertionError: 'recent-context' missing from admitted docs on verification failure turn.\n"
            "Observed admitted docs: project-context, agents/base.md, agents/coder.md, projects/default.md, "
            "file-context:src/harness.py, file-context:src/context_index.py"
        ),
    }
    assistant6 = {
        "role": "assistant",
        "content": (
            "The verification failure matters: recent-context dropped out on the pressure turn, while "
            "file-context survived. I need the next turn to keep that failure visible without broadening the prompt."
        ),
    }

    turn1 = [system, user1, assistant1, tool1, assistant2]
    turn2 = [*turn1, user2, assistant3, tool2, tool3, assistant4]
    turn3 = [*turn2, user3, assistant5, tool4, tool5, assistant6]
    return [turn1, turn2, turn3]


def _debug_loop_history() -> list[list[dict[str, Any]]]:
    system = {"role": "system", "content": _chat_system_prompt()}
    user1 = {
        "role": "user",
        "content": "Debug why recent verification context disappears in debug mode when the harness is under pressure.",
    }
    assistant1 = {
        "role": "assistant",
        "content": "I will trace the failing debug path and keep the latest verification details visible.",
        "tool_calls": [
            _tool_call("read_file", {"path": "src/harness.py"}, call_id="dbg_read_harness"),
            _tool_call(
                "shell",
                {"command": "python -m unittest tests.test_context_harness"},
                call_id="dbg_run_tests_1",
            ),
        ],
    }
    tool1 = {
        "role": "tool",
        "tool_call_id": "dbg_read_harness",
        "content": (
            "src/harness.py\n"
            "recent-context is emitted from last_action, last_verification, and files_in_play. "
            "build_turn_context places it in layer3 after file-context and skills."
        ),
    }
    tool2 = {
        "role": "tool",
        "tool_call_id": "dbg_run_tests_1",
        "content": (
            "python -m unittest tests.test_context_harness\n"
            "FAIL: test_budget_alerts_emit_when_threshold_crosses_custom_ratio\n"
            "FAIL: test_debug_loop_preserves_recent_context_until_pressure_drops_it\n"
        ),
    }
    assistant2 = {
        "role": "assistant",
        "content": (
            "The failing verification should remain in recent-context. I will inspect app-side context snapshotting "
            "and then rerun the same focused tests."
        ),
    }
    user2 = {
        "role": "user",
        "content": "Inspect the app path too and keep the failing command/result in scope.",
    }
    assistant3 = {
        "role": "assistant",
        "content": "I will inspect src/app.py and add targeted observability only if needed.",
        "tool_calls": [
            _tool_call("read_file", {"path": "src/app.py"}, call_id="dbg_read_app"),
        ],
    }
    tool3 = {
        "role": "tool",
        "tool_call_id": "dbg_read_app",
        "content": (
            "src/app.py\n"
            "_prepare_turn_context passes agent.persistent_context_tokens() into build_turn_context and writes "
            "debug snapshots in debug mode, including docs_loaded, layer_tokens, budget_alerts, and candidate decisions."
        ),
    }
    assistant4 = {
        "role": "assistant",
        "content": (
            "App-side wiring already captures debug snapshots. The bug is in admission pressure, not in snapshot logging."
        ),
    }
    user3 = {
        "role": "user",
        "content": "Patch it and rerun the focused test.",
    }
    assistant5 = {
        "role": "assistant",
        "content": "I will patch the admission branch and rerun the focused suite.",
        "tool_calls": [
            _tool_call("edit_file", {"path": "src/harness.py", "instruction": "preserve debug verification context"}, call_id="dbg_edit"),
            _tool_call(
                "shell",
                {"command": "python -m unittest tests.test_context_harness"},
                call_id="dbg_run_tests_2",
            ),
        ],
    }
    tool4 = {
        "role": "tool",
        "tool_call_id": "dbg_edit",
        "content": (
            "Applied edit to src/harness.py. Added explicit candidate trace for skipped docs and preserved "
            "recent-context when pressure allows."
        ),
    }
    tool5 = {
        "role": "tool",
        "tool_call_id": "dbg_run_tests_2",
        "content": (
            "python -m unittest tests.test_context_harness\n"
            "OK\n"
            "Recent-context now survives until the global floor is reached."
        ),
    }
    assistant6 = {
        "role": "assistant",
        "content": "The focused debug test now passes and the recent verification details remain visible for the next turn.",
    }
    turn1 = [system, user1, assistant1, tool1, tool2, assistant2]
    turn2 = [*turn1, user2, assistant3, tool3, assistant4]
    turn3 = [*turn2, user3, assistant5, tool4, tool5, assistant6]
    return [turn1, turn2, turn3]


def _profile_chat_history(
    messages: list[dict[str, Any]],
    *,
    system_prompt: str | None = None,
) -> dict[str, Any]:
    normalized = [dict(msg) for msg in messages if isinstance(msg, dict)]
    if not normalized:
        normalized = [{"role": "system", "content": system_prompt or _chat_system_prompt()}]
    elif str(normalized[0].get("role")) != "system":
        normalized = [{"role": "system", "content": system_prompt or _chat_system_prompt()}, *normalized]

    agent = Agent(object(), system_prompt or content_to_plain_text(normalized[0].get("content", "")))
    agent.messages = normalized

    role_counts: dict[str, int] = {}
    profiled_messages: list[dict[str, Any]] = []
    tool_call_tokens_in_history = 0
    for index, msg in enumerate(normalized):
        role = str(msg.get("role", "unknown"))
        role_counts[role] = role_counts.get(role, 0) + 1
        content_text = content_to_plain_text(msg.get("content", ""))
        content_tokens = estimate_message_content_tokens(msg.get("content", ""))
        loader_counted = content_tokens + 8
        tool_calls = msg.get("tool_calls") if isinstance(msg.get("tool_calls"), list) else []
        tool_call_tokens = estimate_tokens(json.dumps(tool_calls, ensure_ascii=True)) if tool_calls else 0
        tool_call_tokens_in_history += tool_call_tokens
        profiled_messages.append(
            {
                "index": index,
                "role": role,
                "content_tokens": content_tokens,
                "overhead_tokens": 8,
                "loader_counted_tokens": loader_counted + tool_call_tokens,
                "tool_call_tokens_in_history": tool_call_tokens,
                "tool_call_count": len(tool_calls),
                "content_preview": " ".join(content_text.split())[:180],
            }
        )

    top_loader_messages = sorted(
        profiled_messages,
        key=lambda item: item["loader_counted_tokens"],
        reverse=True,
    )[:3]
    return {
        "message_count": len(normalized),
        "role_counts": role_counts,
        "persistent_context_tokens": agent.persistent_context_tokens(),
        "tool_call_tokens_in_history": tool_call_tokens_in_history,
        "top_loader_messages": top_loader_messages,
        "messages": profiled_messages,
    }


def _build_jetson_4k_baseline(workspace: Path) -> ContextBenchmarkSuite:
    _write_repo_fixture(
        workspace,
        architecture_lines=[
            "- `src/harness.py`: owns layered prompt budgeting and document admission.",
            "- `src/context_index.py`: resolves repo summaries and file context notes.",
            "- `tests/test_context_harness.py`: validates layered context scenarios.",
        ],
        role_docs={
            "coder": "Coder guidance: keep changes narrow and verify the active step.",
            "debugger": "Debugger guidance: reproduce, localize, and verify small fixes.",
        },
        project_doc="Project guidance: prefer harness summaries over broad file loads when the 4k budget is tight.",
        skills={
            "python-harness": _skill_doc(tags=["python", "harness"], mode="code", body="Skill: patch harness code paths with small focused edits."),
            "verification-loop": _skill_doc(tags=["verification", "tests"], mode="debug", body="Skill: keep failed verification details visible across turns."),
        },
        repo_files={
            "src/harness.py": "def budget():\n    return 'budget'\n",
            "src/context_index.py": "def lookup():\n    return 'summary'\n",
            "tests/test_context_harness.py": "def test_placeholder():\n    assert True\n",
        },
    )
    first = _turn_state(
        "Implement a narrow change in the harness budgeting path.",
        mode="code",
        files=["src/harness.py"],
        preferred_skills=["python-harness"],
    )
    second = update_state_after_turn(
        first,
        tool_events=[{"tool": "read_file", "ok": True, "summary": "read harness", "target": "src/harness.py"}],
        assistant_text="inspected harness budget code",
    )
    second = _clone_state(second)
    second.files_in_play = ["src/harness.py", "src/context_index.py"]
    third = update_state_after_turn(
        second,
        tool_events=[
            {"tool": "edit_file", "ok": True, "summary": "updated budget math", "target": "src/harness.py"},
            {
                "tool": "shell",
                "ok": False,
                "summary": "pytest failed on context budget assertions",
                "target": "pytest tests/test_context_harness.py",
                "verification": True,
                "command": "python -m unittest tests.test_context_harness",
            },
        ],
        assistant_text="patched harness budget handling",
    )
    turn1_history, turn2_history, turn3_history = _narrow_task_history()
    turns = [
        ContextBenchmarkTurnInput(
            "initial narrow task",
            first,
            None,
            _memory(8192, 4096),
            note="Healthy 4k baseline with actual user/tool transcript.",
            chat_messages=turn1_history,
            system_prompt=_chat_system_prompt(),
        ),
        ContextBenchmarkTurnInput(
            "expanded file scope",
            second,
            None,
            _memory(8192, 4096),
            note="Second file enters play via actual transcript growth.",
            chat_messages=turn2_history,
            system_prompt=_chat_system_prompt(),
        ),
        ContextBenchmarkTurnInput(
            "verification failure pressure",
            third,
            None,
            _memory(8192, 3584),
            note="Recent verification should stay visible while chat/tool history grows.",
            chat_messages=turn3_history,
            system_prompt=_chat_system_prompt(),
        ),
    ]
    return ContextBenchmarkSuite(
        name="jetson_4k_baseline",
        description="Baseline 4k layered-context behavior for a narrow coding task on healthy Jetson-class memory.",
        workspace=workspace,
        effective_window=4096,
        layered_config={"enabled": True, "layer1_enabled": True, "layer2_enabled": True, "layer3_enabled": True},
        turns=turns,
    )


def _build_jetson_4k_ram_pressure(workspace: Path) -> ContextBenchmarkSuite:
    suite = _build_jetson_4k_baseline(workspace)
    stressed = _clone_state(suite.turns[-1].state)
    stressed_history = suite.turns[-1].chat_messages
    turns = [
        ContextBenchmarkTurnInput(
            "high memory",
            stressed,
            None,
            _memory(8192, 4096),
            note="Healthy memory baseline with the same verification-failure transcript.",
            chat_messages=stressed_history,
            system_prompt=_chat_system_prompt(),
        ),
        ContextBenchmarkTurnInput(
            "mid memory",
            stressed,
            None,
            _memory(8192, 1400),
            note="Expect smaller usable prompt budget for the same transcript.",
            chat_messages=stressed_history,
            system_prompt=_chat_system_prompt(),
        ),
        ContextBenchmarkTurnInput(
            "low memory",
            stressed,
            None,
            _memory(8192, 620),
            note="Recent context and skills should be squeezed harder for the same transcript.",
            chat_messages=stressed_history,
            system_prompt=_chat_system_prompt(),
        ),
    ]
    return ContextBenchmarkSuite(
        name="jetson_4k_ram_pressure",
        description="Same 4k task state under different memory snapshots to show RAM-driven budget contraction.",
        workspace=workspace,
        effective_window=4096,
        layered_config=suite.layered_config,
        turns=turns,
    )


def _build_jetson_4k_layer_compare(workspace: Path) -> ContextBenchmarkSuite:
    suite = _build_jetson_4k_baseline(workspace)
    compared = _clone_state(suite.turns[-1].state)
    compared_history = suite.turns[-1].chat_messages
    turns = [
        ContextBenchmarkTurnInput(
            "all layers enabled",
            compared,
            None,
            _memory(8192, 3200),
            note="Default layered config against the same real transcript.",
            chat_messages=compared_history,
            system_prompt=_chat_system_prompt(),
        ),
        ContextBenchmarkTurnInput(
            "layer1 only",
            compared,
            None,
            _memory(8192, 3200),
            layered_config={"layer2_enabled": False, "layer3_enabled": False},
            chat_messages=compared_history,
            system_prompt=_chat_system_prompt(),
        ),
        ContextBenchmarkTurnInput(
            "layer3 disabled",
            compared,
            None,
            _memory(8192, 3200),
            layered_config={"layer3_enabled": False},
            chat_messages=compared_history,
            system_prompt=_chat_system_prompt(),
        ),
        ContextBenchmarkTurnInput(
            "layered context disabled",
            compared,
            None,
            _memory(8192, 3200),
            layered_config={"enabled": False},
            chat_messages=compared_history,
            system_prompt=_chat_system_prompt(),
        ),
    ]
    return ContextBenchmarkSuite(
        name="jetson_4k_layer_compare",
        description="Same 4k state compared across layer toggle variants.",
        workspace=workspace,
        effective_window=4096,
        layered_config=suite.layered_config,
        turns=turns,
    )


def _build_long_debug_session(workspace: Path) -> ContextBenchmarkSuite:
    _write_repo_fixture(
        workspace,
        architecture_lines=[
            "- `src/harness.py`: owns layered context and prompt budgeting.",
            "- `src/app.py`: injects harness docs into runtime messages before each turn.",
            "- `tests/test_context_harness.py`: debug-loop verification coverage.",
        ],
        role_docs={
            "coder": "Coder guidance: keep edits narrow.",
            "debugger": "Debugger guidance: keep the last failing verification in context until resolved.",
        },
        project_doc="Project guidance: do not lose failed verification context across debug turns.",
        skills={
            "debug-trace": _skill_doc(tags=["debug", "traceback"], mode="debug", body="Skill: preserve failing commands, files, and likely fault lines."),
        },
        repo_files={
            "src/harness.py": "def build_turn_context():\n    return None\n",
            "src/app.py": "def prepare_turn():\n    return None\n",
            "tests/test_context_harness.py": "def test_debug_loop():\n    assert True\n",
        },
    )
    first = _turn_state(
        "Debug why the harness drops recent verification context under pressure.",
        mode="debug",
        files=["src/harness.py"],
        preferred_skills=["debug-trace"],
    )
    second = update_state_after_turn(
        first,
        tool_events=[
            {"tool": "read_file", "ok": True, "summary": "read harness", "target": "src/harness.py"},
            {
                "tool": "shell",
                "ok": False,
                "summary": "unit test still failing in debug loop",
                "target": "python -m unittest tests.test_context_harness",
                "verification": True,
                "command": "python -m unittest tests.test_context_harness",
            },
        ],
        assistant_text="localized the failing path",
    )
    third = _clone_state(second)
    third.files_in_play = ["src/harness.py", "src/app.py", "tests/test_context_harness.py"]
    fourth = update_state_after_turn(
        third,
        tool_events=[
            {"tool": "edit_file", "ok": True, "summary": "added debug snapshot", "target": "src/app.py"},
            {
                "tool": "shell",
                "ok": True,
                "summary": "debug loop test now passes",
                "target": "python -m unittest tests.test_context_harness",
                "verification": True,
                "command": "python -m unittest tests.test_context_harness",
            },
        ],
        assistant_text="verified the debug path",
    )
    dbg_turn1, dbg_turn2, dbg_turn3 = _debug_loop_history()
    turns = [
        ContextBenchmarkTurnInput("debug turn 1", first, None, _memory(8192, 3600), chat_messages=dbg_turn1, system_prompt=_chat_system_prompt()),
        ContextBenchmarkTurnInput("debug turn 2 failed verification", second, None, _memory(8192, 2200), chat_messages=dbg_turn2, system_prompt=_chat_system_prompt()),
        ContextBenchmarkTurnInput("debug turn 3 expanded scope", third, None, _memory(8192, 1800), chat_messages=dbg_turn3, system_prompt=_chat_system_prompt()),
        ContextBenchmarkTurnInput("debug turn 4 verified", fourth, None, _memory(8192, 1500), chat_messages=dbg_turn3, system_prompt=_chat_system_prompt()),
    ]
    return ContextBenchmarkSuite(
        name="long_debug_session",
        description="Multi-turn debug session showing how recent verification context survives and then churns as pressure rises.",
        workspace=workspace,
        effective_window=4096,
        layered_config={"enabled": True, "layer1_enabled": True, "layer2_enabled": True, "layer3_enabled": True},
        turns=turns,
    )


def _build_skill_heavy_code_session(workspace: Path) -> ContextBenchmarkSuite:
    _write_repo_fixture(
        workspace,
        architecture_lines=[
            "- `src/harness.py`: layered context budgeting.",
            "- `src/context_index.py`: file-summary resolution.",
            "- `src/runtime_limits.py`: token and RAM heuristics.",
        ],
        role_docs={"coder": "Coder guidance: select the narrowest relevant skill docs."},
        project_doc="Project guidance: rank skills by explicit preference, query overlap, and mode.",
        skills={
            "python-harness": _skill_doc(tags=["python", "harness"], mode="code", body="Skill: modify harness budget code carefully."),
            "token-budget": _skill_doc(tags=["tokens", "budget"], mode="code", body="Skill: reason about prompt and docs budgets."),
            "jetson-memory": _skill_doc(tags=["jetson", "memory"], mode="code", body="Skill: adapt budgets under low-memory Jetson conditions."),
            "review-docs": _skill_doc(tags=["review", "docs"], mode="review", body="Skill: review-only guidance."),
            "context-compare": _skill_doc(tags=["context", "compare"], mode="code", body="Skill: compare admitted-doc churn across turns."),
        },
        repo_files={
            "src/harness.py": "def layered_context_config():\n    return {}\n",
            "src/context_index.py": "def load_repo_context_index():\n    return None\n",
            "src/runtime_limits.py": "def estimate_tokens(text):\n    return len(text)\n",
        },
    )
    state = _turn_state(
        "Implement a Python harness budget comparison for Jetson memory pressure.",
        mode="code",
        files=["src/harness.py", "src/runtime_limits.py"],
        preferred_skills=["python-harness"],
    )
    state.files_in_play.append("src/context_index.py")
    turns = [
        ContextBenchmarkTurnInput("wide-window skill ranking", state, 1200, _memory(16384, 8192), effective_window=16384),
        ContextBenchmarkTurnInput("mid pressure skill retention", state, 4200, _memory(16384, 4096), effective_window=16384),
        ContextBenchmarkTurnInput("high pressure skill retention", state, 6200, _memory(16384, 2400), effective_window=16384),
    ]
    return ContextBenchmarkSuite(
        name="skill_heavy_code_session",
        description="Skill-heavy coding session to expose ranking, selection limits, and survival under pressure.",
        workspace=workspace,
        effective_window=16384,
        layered_config={"enabled": True, "layer1_enabled": True, "layer2_enabled": True, "layer3_enabled": True},
        turns=turns,
    )


def _build_candidate_starvation_case(workspace: Path) -> ContextBenchmarkSuite:
    huge_purpose = " ".join(["owns the dominant context budget path"] * 20)
    _write_repo_fixture(
        workspace,
        architecture_lines=[
            f"- `src/harness.py`: {huge_purpose}",
            "- `src/context_index.py`: resolves file summaries.",
            "- `tests/test_context_harness.py`: focused starvation coverage.",
        ],
        role_docs={"coder": "Coder guidance: keep context on the active file and skill."},
        project_doc="Project guidance: order matters because docs are admitted sequentially.",
        skills={
            "late-skill": _skill_doc(
                tags=["skill", "late"],
                mode="code",
                body=" ".join(["tiny late skill"] * 50),
            ),
        },
        repo_files={
            "src/harness.py": "def candidate_starvation():\n    return True\n",
            "src/context_index.py": "def index():\n    return {}\n",
            "tests/test_context_harness.py": "def test_starvation():\n    assert True\n",
        },
    )
    state = _turn_state(
        "Implement a harness change and keep the late skill doc visible.",
        mode="code",
        files=["src/harness.py"],
        preferred_skills=["late-skill"],
    )
    state.last_action = {"type": "read_file", "target": "src/harness.py", "summary": "read large file summary"}
    turns = [
        ContextBenchmarkTurnInput("starvation turn", state, 700, _memory(4096, 4096)),
    ]
    return ContextBenchmarkSuite(
        name="candidate_starvation_case",
        description="Sequential admission case where an early large layer2 doc starves a later smaller skill doc.",
        workspace=workspace,
        effective_window=4096,
        layered_config={"enabled": True, "layer1_enabled": True, "layer2_enabled": True, "layer3_enabled": True, "layer2_ratio": 0.08},
        turns=turns,
    )


def _memory(total_mb: float, available_mb: float) -> MemorySnapshot:
    used_percent = ((total_mb - available_mb) / total_mb) * 100.0 if total_mb else 0.0
    return MemorySnapshot(total_mb=total_mb, available_mb=available_mb, used_percent=used_percent)


_CONTEXT_SUITE_BUILDERS: dict[str, Callable[[Path], ContextBenchmarkSuite]] = {
    "candidate_starvation_case": _build_candidate_starvation_case,
    "jetson_4k_baseline": _build_jetson_4k_baseline,
    "jetson_4k_layer_compare": _build_jetson_4k_layer_compare,
    "jetson_4k_ram_pressure": _build_jetson_4k_ram_pressure,
    "long_debug_session": _build_long_debug_session,
    "skill_heavy_code_session": _build_skill_heavy_code_session,
}
