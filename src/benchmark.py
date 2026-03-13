from __future__ import annotations

import argparse
import asyncio
import shutil
import hashlib
import json
import os
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

from .agent import ActionKind, Agent
from .config import load_config
from .executor import (
    DEFAULT_SHELL_TIMEOUT_SECONDS,
    edit_file,
    glob_files,
    grep_files,
    list_directory,
    load_file,
    read_file,
    run_shell,
    write_file,
)
from .harness import (
    HarnessState,
    active_step,
    allowed_tools_for_mode,
    build_turn_context,
    normalize_skill_name,
    shell_command_is_verification,
    update_state_after_turn,
    update_state_for_user_message,
)
from .runtime_limits import read_memory_snapshot
from .runtime_protocol import ToolCall
from .runtime_registry import active_model_ref, create_runtime_client


@dataclass(frozen=True)
class EvalFile:
    path: str
    content: str


@dataclass(frozen=True)
class EvalEnvironment:
    files: list[EvalFile] = field(default_factory=list)


@dataclass(frozen=True)
class EvalExpectation:
    kind: str
    target: str = ""
    value: str = ""
    negate: bool = False
    weight: float = 1.0


@dataclass(frozen=True)
class BenchmarkCase:
    case_id: str
    description: str
    prompt: str
    environment: EvalEnvironment
    expectations: list[EvalExpectation]
    template_dir: str | None = None
    mode: str | None = None
    preferred_skills: list[str] = field(default_factory=list)
    max_turns: int = 6
    tags: list[str] = field(default_factory=list)
    allowed_tools: list[str] = field(default_factory=list)


@dataclass
class TurnRecord:
    turn_index: int
    mode: str
    active_step: str | None
    harness_docs: list[str]
    harness_doc_tokens: int
    usable_prompt_budget: int
    remaining_budget: int
    assistant_text: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    tool_results: list[dict[str, Any]] = field(default_factory=list)
    condense_reason: str | None = None


@dataclass
class EvaluationResult:
    score: float
    passed: bool
    passed_weight: float
    total_weight: float
    checks: list[dict[str, Any]]


@dataclass
class BenchmarkRunArtifact:
    benchmark_version: int
    case_id: str
    run_id: str
    repeat_index: int
    started_at: float
    ended_at: float
    duration_seconds: float
    model: str
    runtime: str
    prompt: str
    mode: str
    preferred_skills: list[str]
    allowed_tools: list[str]
    workspace: str
    environment_manifest: list[dict[str, Any]]
    filesystem_snapshot: list[dict[str, Any]]
    turns: list[dict[str, Any]]
    final_messages: list[dict[str, Any]]
    final_harness_state: dict[str, Any]
    tool_counts: dict[str, int]
    final_assistant_text: str
    completed: bool
    failure_reason: str | None
    evaluation: dict[str, Any]


def build_llm_judge_packet(*, case: BenchmarkCase, artifact: BenchmarkRunArtifact) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "evaluation_type": "open_jet_benchmark_run",
        "instructions_markdown": benchmark_judge_instructions_path().read_text(encoding="utf-8"),
        "case": {
            "case_id": case.case_id,
            "description": case.description,
            "prompt": case.prompt,
            "mode": case.mode,
            "tags": list(case.tags),
            "allowed_tools": list(case.allowed_tools),
            "preferred_skills": list(case.preferred_skills),
            "expectations": [asdict(expectation) for expectation in case.expectations],
        },
        "run": {
            "run_id": artifact.run_id,
            "repeat_index": artifact.repeat_index,
            "model": artifact.model,
            "runtime": artifact.runtime,
            "duration_seconds": artifact.duration_seconds,
            "completed": artifact.completed,
            "failure_reason": artifact.failure_reason,
            "tool_counts": dict(artifact.tool_counts),
        },
        "environment": {
            "manifest": artifact.environment_manifest,
            "final_filesystem": artifact.filesystem_snapshot,
        },
        "trace": {
            "turns": artifact.turns,
            "final_assistant_text": artifact.final_assistant_text,
            "final_messages": artifact.final_messages,
            "final_harness_state": artifact.final_harness_state,
        },
        "deterministic_evaluation": artifact.evaluation,
    }


def default_benchmark_cases() -> list[BenchmarkCase]:
    cases: list[BenchmarkCase] = []
    for path in sorted(benchmark_cases_root().glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        environment_payload = payload.get("environment", {})
        files = []
        if isinstance(environment_payload, dict):
            files = [
                EvalFile(path=str(item.get("path", "")), content=str(item.get("content", "")))
                for item in environment_payload.get("files", [])
                if isinstance(item, dict) and str(item.get("path", ""))
            ]
        expectations = [
            EvalExpectation(
                kind=str(item.get("kind", "")),
                target=str(item.get("target", "")),
                value=str(item.get("value", "")),
                negate=bool(item.get("negate", False)),
                weight=float(item.get("weight", 1.0)),
            )
            for item in payload.get("expectations", [])
            if isinstance(item, dict)
        ]
        cases.append(
            BenchmarkCase(
                case_id=str(payload.get("case_id", path.stem)),
                description=str(payload.get("description", "")),
                prompt=str(payload.get("prompt", "")),
                environment=EvalEnvironment(files=files),
                expectations=expectations,
                template_dir=str(payload["template_dir"]) if payload.get("template_dir") else None,
                mode=str(payload["mode"]) if payload.get("mode") else None,
                preferred_skills=[str(item) for item in payload.get("preferred_skills", []) if str(item)],
                max_turns=int(payload.get("max_turns", 6)),
                tags=[str(item) for item in payload.get("tags", []) if str(item)],
                allowed_tools=[str(item) for item in payload.get("allowed_tools", []) if str(item)],
            )
        )
    return cases


@contextmanager
def pushd(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def seed_environment(workspace: Path, env: EvalEnvironment) -> list[dict[str, Any]]:
    workspace.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, Any]] = []
    for item in env.files:
        path = workspace / item.path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(item.content, encoding="utf-8")
        manifest.append(
            {
                "path": item.path,
                "sha256": hashlib.sha256(item.content.encode("utf-8")).hexdigest(),
                "bytes": len(item.content.encode("utf-8")),
            }
        )
    return manifest


def clone_environment_template(workspace: Path, template_dir: Path) -> None:
    if not template_dir.exists():
        return
    workspace.mkdir(parents=True, exist_ok=True)
    for path in sorted(template_dir.rglob("*")):
        rel = path.relative_to(template_dir)
        target = workspace / rel
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)


def benchmark_templates_root() -> Path:
    return Path(__file__).resolve().parent.parent / "benchmarks" / "envs"


def benchmark_cases_root() -> Path:
    return Path(__file__).resolve().parent.parent / "benchmarks" / "cases"


def benchmark_judge_instructions_path() -> Path:
    return Path(__file__).resolve().parent.parent / "benchmarks" / "judge_instructions.md"


def snapshot_workspace(workspace: Path) -> list[dict[str, Any]]:
    snapshot: list[dict[str, Any]] = []
    for path in sorted(workspace.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(workspace).as_posix()
        raw = path.read_bytes()
        snapshot.append(
            {
                "path": rel,
                "sha256": hashlib.sha256(raw).hexdigest(),
                "bytes": len(raw),
                "content": raw.decode("utf-8", errors="replace"),
            }
        )
    return snapshot


async def execute_benchmark_tool(tc: ToolCall, *, remaining_prompt_tokens: int) -> tuple[str, dict[str, Any]]:
    if tc.name == "shell":
        command = str(tc.arguments.get("command", "")).strip()
        timeout_seconds = tc.arguments.get("timeout_seconds")
        if not command:
            return "Error: invalid arguments for shell (required: command)", {"ok": False}
        res = await run_shell(command, timeout_seconds=timeout_seconds or DEFAULT_SHELL_TIMEOUT_SECONDS)
        return res.summary, {
            "ok": res.ok,
            "summary": res.summary.splitlines()[0] if res.summary else "",
            "target": command,
            "verification": shell_command_is_verification(command),
            "command": command,
        }
    if tc.name == "read_file":
        path = str(tc.arguments.get("path", "")).strip()
        text = await read_file(path)
        return text, {"ok": not text.startswith("Error"), "summary": text.splitlines()[0] if text else "", "target": path}
    if tc.name == "load_file":
        path = str(tc.arguments.get("path", "")).strip()
        max_tokens = tc.arguments.get("max_tokens")
        if not isinstance(max_tokens, int):
            max_tokens = remaining_prompt_tokens
        loaded = await load_file(path, max_tokens=max(128, min(max_tokens, remaining_prompt_tokens)))
        if not loaded.ok:
            return loaded.detail, {"ok": False, "summary": loaded.detail, "target": path}
        result = f"{loaded.summary}\ncontent:\n{loaded.content}"
        return result, {
            "ok": True,
            "summary": loaded.summary,
            "target": path,
        }
    if tc.name == "write_file":
        path = str(tc.arguments.get("path", "")).strip()
        content = str(tc.arguments.get("content", ""))
        text = await write_file(path, content)
        return text, {"ok": not text.startswith("Error"), "summary": text, "target": path}
    if tc.name == "edit_file":
        path = str(tc.arguments.get("path", "")).strip()
        old_string = str(tc.arguments.get("old_string", ""))
        new_string = str(tc.arguments.get("new_string", ""))
        replace_all = bool(tc.arguments.get("replace_all", False))
        text = await edit_file(path, old_string, new_string, replace_all=replace_all)
        return text, {"ok": not text.startswith("Error"), "summary": text, "target": path}
    if tc.name == "glob":
        pattern = str(tc.arguments.get("pattern", "")).strip()
        path = tc.arguments.get("path")
        text = await glob_files(pattern, path=path)
        return text, {"ok": not text.startswith("Error"), "summary": text.splitlines()[0] if text else "", "target": pattern}
    if tc.name == "grep":
        pattern = str(tc.arguments.get("pattern", "")).strip()
        path = tc.arguments.get("path")
        glob_filter = tc.arguments.get("glob")
        ignore_case = bool(tc.arguments.get("ignore_case", False))
        text = await grep_files(pattern, path=path, glob_filter=glob_filter, ignore_case=ignore_case)
        return text, {"ok": not text.startswith("Error"), "summary": text.splitlines()[0] if text else "", "target": pattern}
    if tc.name == "list_directory":
        path = tc.arguments.get("path")
        text = await list_directory(path=path)
        return text, {"ok": not text.startswith("Error"), "summary": text.splitlines()[0] if text else "", "target": path or "."}
    return f"Unknown tool: {tc.name}", {"ok": False, "summary": f"Unknown tool: {tc.name}", "target": tc.name}


def evaluate_case(
    *,
    case: BenchmarkCase,
    artifact: BenchmarkRunArtifact,
) -> EvaluationResult:
    checks: list[dict[str, Any]] = []
    final_text = artifact.final_assistant_text
    snapshot_by_path = {item["path"]: item for item in artifact.filesystem_snapshot}
    used_tools = set(artifact.tool_counts)
    total_weight = sum(max(0.0, expectation.weight) for expectation in case.expectations) or 1.0
    passed_weight = 0.0

    for expectation in case.expectations:
        passed = False
        detail = ""
        if expectation.kind == "tool_used":
            passed = expectation.target in used_tools
            detail = f"used_tools={sorted(used_tools)}"
        elif expectation.kind == "assistant_contains":
            passed = expectation.value in final_text
            detail = final_text
        elif expectation.kind == "file_exists":
            passed = expectation.target in snapshot_by_path
            detail = expectation.target
        elif expectation.kind == "file_contains":
            content = snapshot_by_path.get(expectation.target, {}).get("content", "")
            passed = expectation.value in content
            detail = content
        elif expectation.kind == "file_not_contains":
            content = snapshot_by_path.get(expectation.target, {}).get("content", "")
            passed = expectation.value not in content
            detail = content
        else:
            detail = f"unknown_expectation:{expectation.kind}"

        if expectation.negate:
            passed = not passed
        if passed:
            passed_weight += max(0.0, expectation.weight)
        checks.append(
            {
                "kind": expectation.kind,
                "target": expectation.target,
                "value": expectation.value,
                "weight": expectation.weight,
                "passed": passed,
                "detail": detail,
            }
        )

    score = round(passed_weight / total_weight, 4)
    return EvaluationResult(
        score=score,
        passed=(score >= 1.0 and artifact.completed and artifact.failure_reason is None),
        passed_weight=passed_weight,
        total_weight=total_weight,
        checks=checks,
    )


def _remaining_prompt_tokens(agent: Agent) -> int:
    budget = agent.context_budget()
    if not budget:
        return 128
    return max(128, budget.prompt_tokens - agent.estimated_context_tokens())


async def run_benchmark_case(
    case: BenchmarkCase,
    *,
    repeat_index: int,
    output_dir: Path,
    cfg: dict | None = None,
    client_factory: Callable[[dict], Any] | None = None,
) -> BenchmarkRunArtifact:
    cfg = dict(cfg or load_config())
    run_id = uuid.uuid4().hex[:10]
    run_dir = output_dir / case.case_id / f"run_{repeat_index:02d}_{run_id}"
    workspace = run_dir / "env"
    if case.template_dir:
        clone_environment_template(workspace, benchmark_templates_root() / case.template_dir)
    environment_manifest = seed_environment(workspace, case.environment)

    started_at = time.time()
    runtime = client_factory(cfg) if client_factory else create_runtime_client(cfg)
    await runtime.start()

    mem_cfg = cfg.get("memory_guard", {})
    agent = Agent(
        client=runtime,
        system_prompt=str(cfg.get("system_prompt", "")),
        context_window_tokens=getattr(runtime, "context_window_tokens", int(cfg.get("context_window_tokens", 2048))),
        context_reserved_tokens=(int(mem_cfg["context_reserved_tokens"]) if mem_cfg.get("context_reserved_tokens") is not None else None),
        min_prompt_tokens=int(mem_cfg.get("min_prompt_tokens", 256)),
        min_available_mb=(int(mem_cfg["min_available_mb"]) if mem_cfg.get("min_available_mb") is not None else None),
        max_used_percent=(float(mem_cfg["max_used_percent"]) if mem_cfg.get("max_used_percent") is not None else None),
        memory_check_interval_chunks=int(mem_cfg.get("check_interval_chunks", 16)),
        condense_target_tokens=int(mem_cfg.get("condense_target_tokens", 900)),
        keep_last_messages=int(mem_cfg.get("keep_last_messages", 6)),
    )

    harness_state = HarnessState(preferred_skills=[normalize_skill_name(name) for name in case.preferred_skills])
    harness_state = update_state_for_user_message(
        harness_state,
        case.prompt,
        mode=case.mode,
        files=[item.path for item in case.environment.files[:2]],
    )
    agent.add_user_message(case.prompt)

    turns: list[TurnRecord] = []
    final_assistant_text = ""
    completed = False
    failure_reason: str | None = None

    try:
        with pushd(workspace):
            for turn_index in range(1, case.max_turns + 1):
                context = build_turn_context(
                    root=workspace,
                    state=harness_state,
                    current_context_tokens=agent.persistent_context_tokens(),
                    effective_window=getattr(runtime, "context_window_tokens", int(cfg.get("context_window_tokens", 2048))),
                    memory_snapshot=read_memory_snapshot(),
                    layered_config=cfg.get("layered_context", {}),
                )
                agent.set_turn_context(context.messages)
                turn = TurnRecord(
                    turn_index=turn_index,
                    mode=harness_state.mode,
                    active_step=active_step(harness_state).title if active_step(harness_state) else None,
                    harness_docs=context.docs_loaded,
                    harness_doc_tokens=context.docs_tokens,
                    usable_prompt_budget=context.budget.usable_prompt_budget,
                    remaining_budget=context.budget.remaining_budget,
                )

                pending_tool_calls: list[ToolCall] = []
                assistant_text = ""
                condense_reason: str | None = None

                async for event in agent.run_turn():
                    if event.kind == ActionKind.TEXT:
                        assistant_text += event.text
                    elif event.kind == ActionKind.TOOL_REQUEST and event.tool_call:
                        pending_tool_calls.append(event.tool_call)
                        turn.tool_calls.append(
                            {
                                "tool": event.tool_call.name,
                                "arguments": dict(event.tool_call.arguments),
                            }
                        )
                    elif event.kind == ActionKind.CONDENSE:
                        condense_reason = event.text
                    elif event.kind == ActionKind.ERROR:
                        failure_reason = event.text
                        break

                turn.assistant_text = assistant_text
                turn.condense_reason = condense_reason
                final_assistant_text = assistant_text or final_assistant_text
                tool_events: list[dict[str, Any]] = []

                if failure_reason:
                    turns.append(turn)
                    break

                if condense_reason:
                    await agent.condense_context(force=True)
                    turns.append(turn)
                    continue

                for tool_call in pending_tool_calls:
                    if case.allowed_tools and tool_call.name not in set(case.allowed_tools):
                        denial = {
                            "tool": tool_call.name,
                            "ok": False,
                            "summary": f"Tool {tool_call.name} is outside the eval allowlist.",
                            "target": tool_call.name,
                        }
                        tool_events.append(denial)
                        turn.tool_results.append(dict(denial))
                        agent.complete_tool_call(tool_call, denial["summary"])
                        continue
                    if tool_call.name not in allowed_tools_for_mode(harness_state.mode):
                        denial = {
                            "tool": tool_call.name,
                            "ok": False,
                            "summary": f"Tool {tool_call.name} is not allowed in {harness_state.mode} mode.",
                            "target": tool_call.name,
                        }
                        tool_events.append(denial)
                        turn.tool_results.append(dict(denial))
                        agent.complete_tool_call(tool_call, denial["summary"])
                        continue

                    result, meta = await execute_benchmark_tool(
                        tool_call,
                        remaining_prompt_tokens=_remaining_prompt_tokens(agent),
                    )
                    agent.complete_tool_call(tool_call, result)
                    event_meta = {
                        "tool": tool_call.name,
                        "ok": bool(meta.get("ok", True)),
                        "summary": meta.get("summary", ""),
                        "target": meta.get("target"),
                        "verification": bool(meta.get("verification", False)),
                        "command": meta.get("command"),
                    }
                    tool_events.append(event_meta)
                    tool_result_record = dict(event_meta)
                    tool_result_record["result"] = result
                    turn.tool_results.append(tool_result_record)

                harness_state = update_state_after_turn(
                    harness_state,
                    tool_events=tool_events,
                    assistant_text=assistant_text,
                )
                turns.append(turn)
                if not pending_tool_calls:
                    completed = True
                    break
            else:
                failure_reason = f"max_turns_exceeded:{case.max_turns}"
    finally:
        await runtime.close()

    ended_at = time.time()
    tool_counts: dict[str, int] = {}
    for turn in turns:
        for result in turn.tool_results:
            tool = str(result.get("tool", ""))
            tool_counts[tool] = tool_counts.get(tool, 0) + 1

    artifact = BenchmarkRunArtifact(
        benchmark_version=1,
        case_id=case.case_id,
        run_id=run_id,
        repeat_index=repeat_index,
        started_at=started_at,
        ended_at=ended_at,
        duration_seconds=round(ended_at - started_at, 3),
        model=active_model_ref(cfg) or getattr(runtime, "model", ""),
        runtime=str(cfg.get("runtime", "llama_cpp")),
        prompt=case.prompt,
        mode=harness_state.mode,
        preferred_skills=list(harness_state.preferred_skills),
        allowed_tools=list(case.allowed_tools),
        workspace=str(workspace),
        environment_manifest=environment_manifest,
        filesystem_snapshot=snapshot_workspace(workspace),
        turns=[asdict(turn) for turn in turns],
        final_messages=agent.messages,
        final_harness_state=harness_state.to_dict(),
        tool_counts=tool_counts,
        final_assistant_text=final_assistant_text,
        completed=completed,
        failure_reason=failure_reason,
        evaluation={},
    )
    evaluation = evaluate_case(case=case, artifact=artifact)
    artifact.evaluation = asdict(evaluation)

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "result.json").write_text(json.dumps(asdict(artifact), ensure_ascii=False, indent=2), encoding="utf-8")
    judge_packet = build_llm_judge_packet(case=case, artifact=artifact)
    (run_dir / "judge_packet.json").write_text(json.dumps(judge_packet, ensure_ascii=False, indent=2), encoding="utf-8")
    return artifact


async def run_benchmark_suite(
    *,
    cases: list[BenchmarkCase],
    repeats: int,
    output_dir: Path,
    cfg: dict | None = None,
    client_factory: Callable[[dict], Any] | None = None,
) -> list[BenchmarkRunArtifact]:
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts: list[BenchmarkRunArtifact] = []
    summary_path = output_dir / "summary.jsonl"
    for case in cases:
        for repeat_index in range(1, repeats + 1):
            artifact = await run_benchmark_case(
                case,
                repeat_index=repeat_index,
                output_dir=output_dir,
                cfg=cfg,
                client_factory=client_factory,
            )
            artifacts.append(artifact)
            summary_record = {
                "case_id": artifact.case_id,
                "repeat_index": artifact.repeat_index,
                "run_id": artifact.run_id,
                "passed": artifact.evaluation["passed"],
                "score": artifact.evaluation["score"],
                "completed": artifact.completed,
                "failure_reason": artifact.failure_reason,
                "duration_seconds": artifact.duration_seconds,
                "tool_counts": artifact.tool_counts,
                "result_path": str(output_dir / case.case_id / f"run_{repeat_index:02d}_{artifact.run_id}" / "result.json"),
            }
            with summary_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(summary_record, ensure_ascii=True) + "\n")

    pass_rate = 0.0
    avg_score = 0.0
    if artifacts:
        pass_rate = sum(1 for artifact in artifacts if artifact.evaluation["passed"]) / len(artifacts)
        avg_score = sum(float(artifact.evaluation["score"]) for artifact in artifacts) / len(artifacts)

    manifest = {
        "benchmark_version": 1,
        "generated_at": time.time(),
        "repeats": repeats,
        "cases": [case.case_id for case in cases],
        "runs": len(artifacts),
        "pass_rate": round(pass_rate, 4),
        "average_score": round(avg_score, 4),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    judge_index = {
        "schema_version": 1,
        "evaluation_type": "open_jet_benchmark_suite",
        "runs": [
            {
                "case_id": artifact.case_id,
                "repeat_index": artifact.repeat_index,
                "run_id": artifact.run_id,
                "judge_packet_path": str(output_dir / artifact.case_id / f"run_{artifact.repeat_index:02d}_{artifact.run_id}" / "judge_packet.json"),
                "result_path": str(output_dir / artifact.case_id / f"run_{artifact.repeat_index:02d}_{artifact.run_id}" / "result.json"),
            }
            for artifact in artifacts
        ],
    }
    (output_dir / "judge_index.json").write_text(json.dumps(judge_index, ensure_ascii=False, indent=2), encoding="utf-8")
    return artifacts


def select_cases(names: list[str] | None) -> list[BenchmarkCase]:
    lookup = {case.case_id: case for case in default_benchmark_cases()}
    if not names:
        return list(lookup.values())
    selected: list[BenchmarkCase] = []
    for name in names:
        key = name.strip()
        if key not in lookup:
            raise ValueError(f"Unknown benchmark case: {key}")
        selected.append(lookup[key])
    return selected


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run open-jet eval benchmarks in isolated environments.")
    parser.add_argument("--cases", type=str, default="", help="Comma-separated case ids. Default: all.")
    parser.add_argument("--repeats", type=int, default=3, help="How many times to run each case.")
    parser.add_argument("--output-dir", type=str, default="benchmark_runs", help="Directory for benchmark artifacts.")
    args = parser.parse_args(argv)

    case_names = [item.strip() for item in args.cases.split(",") if item.strip()]
    cases = select_cases(case_names)
    asyncio.run(run_benchmark_suite(cases=cases, repeats=max(1, args.repeats), output_dir=Path(args.output_dir)))


if __name__ == "__main__":
    main()
