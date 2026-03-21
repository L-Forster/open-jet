from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .context_index import load_repo_context_index, lookup_file_summary
from .runtime_limits import MemorySnapshot, estimate_tokens


ROLE_BY_MODE = {
    "chat": "chat",
    "code": "coder",
    "review": "reviewer",
    "debug": "debugger",
}

TOOL_BUNDLES = {
    "chat": {"memory", "read_file", "load_file", "glob", "grep", "list_directory", "shell"},
    "code": {"memory", "read_file", "load_file", "write_file", "edit_file", "glob", "grep", "list_directory", "shell"},
    "review": {"memory", "read_file", "load_file", "glob", "grep", "list_directory", "shell"},
    "debug": {"memory", "read_file", "load_file", "write_file", "edit_file", "glob", "grep", "list_directory", "shell"},
}

DEFAULT_BASE_PROMPT = """You are operating inside open-jet on an edge Linux device.
Keep work decomposed into small turns.
Prefer exact file excerpts over broad repo context.
Prefer dedicated tools over shell unless shell is the right fit.
Preserve enough state for the next turn to continue cleanly.
"""

DEFAULT_ROLE_PROMPTS = {
    "chat": "Answer directly. Avoid unnecessary tool use.",
    "coder": "Implement the active step only. Keep edits narrow and verify the changed behavior.",
    "reviewer": "Prioritize bugs, regressions, and missing verification over style commentary.",
    "debugger": "Reproduce, localize, fix, and verify. Prefer small experiments over broad rewrites.",
}


@dataclass
class PlanStep:
    id: str
    title: str
    status: str
    kind: str
    files: list[str] = field(default_factory=list)
    skill: str | None = None
    acceptance: str = ""


@dataclass
class HarnessState:
    goal: str = ""
    mode: str = "code"
    preferred_skills: list[str] = field(default_factory=list)
    plan: list[PlanStep] = field(default_factory=list)
    active_step_id: str | None = None
    files_in_play: list[str] = field(default_factory=list)
    last_action: dict[str, Any] = field(default_factory=dict)
    last_verification: dict[str, Any] = field(default_factory=dict)
    open_questions: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=lambda: ["edge device", "bounded context"])
    next_action: str = ""
    failure_count_for_active_step: int = 0
    updated_at: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["plan"] = [asdict(step) for step in self.plan]
        return data

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> HarnessState:
        if not isinstance(payload, dict):
            return cls()
        plan_payload = payload.get("plan")
        plan: list[PlanStep] = []
        if isinstance(plan_payload, list):
            for item in plan_payload:
                if not isinstance(item, dict):
                    continue
                try:
                    plan.append(
                        PlanStep(
                            id=str(item.get("id", "")),
                            title=str(item.get("title", "")),
                            status=str(item.get("status", "pending")),
                            kind=str(item.get("kind", "work")),
                            files=[str(path) for path in item.get("files", []) if str(path)],
                            skill=str(item["skill"]) if item.get("skill") else None,
                            acceptance=str(item.get("acceptance", "")),
                        )
                    )
                except Exception:
                    continue
        return cls(
            goal=str(payload.get("goal", "")),
            mode=str(payload.get("mode", "code")),
            preferred_skills=[str(item) for item in payload.get("preferred_skills", []) if str(item)],
            plan=plan,
            active_step_id=str(payload["active_step_id"]) if payload.get("active_step_id") else None,
            files_in_play=[str(path) for path in payload.get("files_in_play", []) if str(path)],
            last_action=dict(payload.get("last_action", {})) if isinstance(payload.get("last_action"), dict) else {},
            last_verification=dict(payload.get("last_verification", {})) if isinstance(payload.get("last_verification"), dict) else {},
            open_questions=[str(item) for item in payload.get("open_questions", []) if str(item)],
            constraints=[str(item) for item in payload.get("constraints", []) if str(item)] or ["edge device", "bounded context"],
            next_action=str(payload.get("next_action", "")),
            failure_count_for_active_step=int(payload.get("failure_count_for_active_step", 0) or 0),
            updated_at=float(payload.get("updated_at", 0.0) or 0.0),
        )


@dataclass(frozen=True)
class TurnBudget:
    effective_window: int
    generation_reserve: int
    tool_reserve: int
    system_reserve: int
    base_prompt_budget: int
    ram_factor: float
    usable_prompt_budget: int
    current_context_tokens: int
    remaining_budget: int
    docs_budget: int
    layer1_budget: int
    layer2_budget: int
    layer3_budget: int
    layer_alert_tokens: int


@dataclass(frozen=True)
class LayeredContextConfig:
    enabled: bool
    layer1_enabled: bool
    layer2_enabled: bool
    layer3_enabled: bool
    layer1_ratio: float
    layer2_ratio: float
    layer3_ratio: float
    alert_ratio: float


@dataclass(frozen=True)
class HarnessContext:
    messages: list[dict[str, str]]
    state_summary: str
    state_summary_tokens: int
    docs_loaded: list[str]
    docs_tokens: int
    budget: TurnBudget
    layer_tokens: dict[str, int]
    layer_docs: dict[str, list[str]]
    budget_alerts: list[str]
    candidate_decisions: list[dict[str, Any]]


@dataclass(frozen=True)
class HarnessDocCandidate:
    layer: str
    label: str
    body: str
    token_count: int


@dataclass(frozen=True)
class HarnessDocDecision:
    layer: str
    label: str
    token_count: int
    admitted: bool
    skipped_reason: str | None = None


class HarnessSessionStore:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path or (Path.cwd() / ".openjet" / "state" / "session.json")

    def load(self) -> HarnessState:
        if not self.path.exists():
            return HarnessState()
        try:
            return HarnessState.from_dict(json.loads(self.path.read_text(encoding="utf-8")))
        except Exception:
            return HarnessState()

    def save(self, state: HarnessState) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp = self.path.with_suffix(self.path.suffix + ".tmp")
        temp.write_text(json.dumps(state.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        temp.replace(self.path)


def infer_mode(text: str) -> str:
    lowered = text.lower()
    if any(token in lowered for token in ("review", "audit", "finding", "regression")):
        return "review"
    if any(token in lowered for token in ("debug", "fix", "failing", "error", "bug", "traceback")):
        return "debug"
    if any(token in lowered for token in ("write", "edit", "change", "implement", "refactor", "@")):
        return "code"
    return "chat"


def create_plan(goal: str, mode: str, files: list[str]) -> list[PlanStep]:
    focus = ", ".join(files[:2]) if files else "the relevant area"
    if mode == "review":
        return [
            PlanStep("inspect", f"Inspect {focus}", "active", "inspect", files=files, acceptance="understand the target area"),
            PlanStep("review", f"Review {focus} for risks", "pending", "review", files=files, acceptance="identify concrete findings"),
            PlanStep("report", "Summarize findings and gaps", "pending", "report", acceptance="deliver actionable review notes"),
        ]
    if mode == "debug":
        return [
            PlanStep("inspect", f"Inspect the failing area in {focus}", "active", "inspect", files=files, acceptance="localize the likely cause"),
            PlanStep("fix", f"Implement a narrow fix for {focus}", "pending", "change", files=files, acceptance="apply the smallest useful fix"),
            PlanStep("verify", "Verify the fix", "pending", "verify", files=files, acceptance="confirm the failure is resolved"),
        ]
    if mode == "chat":
        return [
            PlanStep("answer", _shorten(goal, 72), "active", "answer", files=files, acceptance="answer the current request clearly"),
        ]
    return [
        PlanStep("inspect", f"Inspect {focus}", "active", "inspect", files=files, acceptance="understand the target area"),
        PlanStep("change", f"Implement the requested change in {focus}", "pending", "change", files=files, acceptance="complete the active code change"),
        PlanStep("verify", "Verify the changed behavior", "pending", "verify", files=files, acceptance="run a focused verification"),
    ]


def update_state_for_user_message(
    state: HarnessState,
    text: str,
    *,
    mode: str | None = None,
    files: list[str] | None = None,
) -> HarnessState:
    chosen_mode = mode or infer_mode(text)
    mentioned_files = [path for path in (files or []) if path]
    next_state = HarnessState.from_dict(state.to_dict())
    next_state.goal = text.strip()
    next_state.mode = chosen_mode
    next_state.preferred_skills = next_state.preferred_skills[:]
    next_state.plan = create_plan(next_state.goal, chosen_mode, mentioned_files)
    next_state.active_step_id = next_state.plan[0].id if next_state.plan else None
    next_state.files_in_play = mentioned_files
    next_state.last_action = {}
    next_state.last_verification = {}
    next_state.next_action = next_state.plan[0].title if next_state.plan else ""
    next_state.failure_count_for_active_step = 0
    next_state.updated_at = time.time()
    return next_state


def compute_turn_budget(
    *,
    effective_window: int,
    current_context_tokens: int,
    memory_snapshot: MemorySnapshot | None,
    layered_config: LayeredContextConfig | None = None,
) -> TurnBudget:
    config = layered_config or layered_context_config(None)
    generation_reserve = max(512, int(effective_window * 0.18))
    tool_reserve = max(128, int(effective_window * 0.04))
    system_reserve = 300 if effective_window <= 8192 else 500
    base_prompt_budget = max(512, effective_window - generation_reserve - tool_reserve - system_reserve)
    mem_available_mb = memory_snapshot.available_mb if memory_snapshot else None
    if mem_available_mb is None:
        ram_factor = 1.0
    elif mem_available_mb < 700:
        ram_factor = 0.35
    elif mem_available_mb < 1000:
        ram_factor = 0.50
    elif mem_available_mb < 1500:
        ram_factor = 0.65
    elif mem_available_mb < 2500:
        ram_factor = 0.80
    else:
        ram_factor = 1.00
    usable_prompt_budget = max(512, int(base_prompt_budget * ram_factor))
    remaining_budget = usable_prompt_budget - max(0, current_context_tokens)
    docs_budget = max(192, min(int(usable_prompt_budget * 0.28), max(192, remaining_budget // 2 if remaining_budget > 0 else 192)))
    layer1_budget = max(0, min(docs_budget, int(effective_window * config.layer1_ratio))) if config.layer1_enabled else 0
    layer2_budget = max(0, min(docs_budget, int(effective_window * config.layer2_ratio))) if config.layer2_enabled else 0
    layer3_budget = max(0, min(docs_budget, int(effective_window * config.layer3_ratio))) if config.layer3_enabled else 0
    layer_alert_tokens = max(96, int(effective_window * config.alert_ratio))
    return TurnBudget(
        effective_window=effective_window,
        generation_reserve=generation_reserve,
        tool_reserve=tool_reserve,
        system_reserve=system_reserve,
        base_prompt_budget=base_prompt_budget,
        ram_factor=ram_factor,
        usable_prompt_budget=usable_prompt_budget,
        current_context_tokens=current_context_tokens,
        remaining_budget=remaining_budget,
        docs_budget=docs_budget,
        layer1_budget=layer1_budget,
        layer2_budget=layer2_budget,
        layer3_budget=layer3_budget,
        layer_alert_tokens=layer_alert_tokens,
    )


def build_turn_context(
    *,
    root: Path,
    state: HarnessState,
    current_context_tokens: int,
    effective_window: int,
    memory_snapshot: MemorySnapshot | None,
    layered_config: dict[str, Any] | None = None,
) -> HarnessContext:
    config = layered_context_config(layered_config)
    budget = compute_turn_budget(
        effective_window=effective_window,
        current_context_tokens=current_context_tokens,
        memory_snapshot=memory_snapshot,
        layered_config=config,
    )
    messages: list[dict[str, str]] = []
    docs_loaded: list[str] = []
    docs_tokens = 0
    layer_tokens = {"layer1": 0, "layer2": 0, "layer3": 0}
    layer_docs = {"layer1": [], "layer2": [], "layer3": []}
    budget_alerts: list[str] = []
    candidate_decisions: list[dict[str, Any]] = []

    state_summary = build_state_summary(state, budget)
    state_summary_tokens = estimate_tokens(state_summary)
    messages.append({"role": "system", "content": state_summary})

    remaining = max(0, budget.docs_budget - state_summary_tokens)
    if remaining <= 0:
        return HarnessContext(
            messages=messages,
            state_summary=state_summary,
            state_summary_tokens=state_summary_tokens,
            docs_loaded=docs_loaded,
            docs_tokens=state_summary_tokens,
            budget=budget,
            layer_tokens=layer_tokens,
            layer_docs=layer_docs,
            budget_alerts=budget_alerts,
            candidate_decisions=candidate_decisions,
        )

    per_layer_remaining = {
        "layer1": min(remaining, budget.layer1_budget),
        "layer2": min(remaining, budget.layer2_budget),
        "layer3": min(remaining, budget.layer3_budget),
    }
    stop_after_global_floor = False

    for candidate in _candidate_docs(root, state, effective_window):
        if not candidate.body.strip():
            candidate_decisions.append(
                asdict(
                    HarnessDocDecision(
                        layer=candidate.layer,
                        label=candidate.label,
                        token_count=candidate.token_count,
                        admitted=False,
                        skipped_reason="empty_body",
                    )
                )
            )
            continue
        if stop_after_global_floor:
            candidate_decisions.append(
                asdict(
                    HarnessDocDecision(
                        layer=candidate.layer,
                        label=candidate.label,
                        token_count=candidate.token_count,
                        admitted=False,
                        skipped_reason="remaining_global_floor_reached",
                    )
                )
            )
            continue
        layer_name = candidate.layer
        if not _layer_enabled(config, layer_name):
            candidate_decisions.append(
                asdict(
                    HarnessDocDecision(
                        layer=layer_name,
                        label=candidate.label,
                        token_count=candidate.token_count,
                        admitted=False,
                        skipped_reason="disabled_layer",
                    )
                )
            )
            continue
        doc_tokens = candidate.token_count
        if doc_tokens > remaining:
            candidate_decisions.append(
                asdict(
                    HarnessDocDecision(
                        layer=layer_name,
                        label=candidate.label,
                        token_count=doc_tokens,
                        admitted=False,
                        skipped_reason="exceeds_remaining_global_budget",
                    )
                )
            )
            continue
        if doc_tokens > per_layer_remaining.get(layer_name, remaining):
            candidate_decisions.append(
                asdict(
                    HarnessDocDecision(
                        layer=layer_name,
                        label=candidate.label,
                        token_count=doc_tokens,
                        admitted=False,
                        skipped_reason="exceeds_remaining_per_layer_budget",
                    )
                )
            )
            continue
        messages.append({"role": "system", "content": candidate.body})
        docs_loaded.append(candidate.label)
        docs_tokens += doc_tokens
        layer_tokens[layer_name] = layer_tokens.get(layer_name, 0) + doc_tokens
        layer_docs.setdefault(layer_name, []).append(candidate.label)
        candidate_decisions.append(
            asdict(
                HarnessDocDecision(
                    layer=layer_name,
                    label=candidate.label,
                    token_count=doc_tokens,
                    admitted=True,
                )
            )
        )
        remaining -= doc_tokens
        per_layer_remaining[layer_name] = max(0, per_layer_remaining.get(layer_name, 0) - doc_tokens)
        if remaining < 128:
            stop_after_global_floor = True

    for layer_name, used in layer_tokens.items():
        if used > budget.layer_alert_tokens:
            budget_alerts.append(
                f"{layer_name} exceeded 40% of the context window: used={used} threshold={budget.layer_alert_tokens}"
            )

    docs_tokens += state_summary_tokens
    return HarnessContext(
        messages=messages,
        state_summary=state_summary,
        state_summary_tokens=state_summary_tokens,
        docs_loaded=docs_loaded,
        docs_tokens=docs_tokens,
        budget=budget,
        layer_tokens=layer_tokens,
        layer_docs=layer_docs,
        budget_alerts=budget_alerts,
        candidate_decisions=candidate_decisions,
    )


def build_state_summary(state: HarnessState, budget: TurnBudget | None = None) -> str:
    active = active_step(state)
    completed = [step.title for step in state.plan if step.status == "done"]
    lines = [
        "OPEN-JET HARNESS STATE",
        f"MODE: {state.mode}",
        f"GOAL: {state.goal or 'n/a'}",
        f"ACTIVE_STEP: {active.title if active else 'n/a'}",
        f"FILES_IN_PLAY: {', '.join(state.files_in_play) if state.files_in_play else 'n/a'}",
        f"COMPLETED_STEPS: {', '.join(completed) if completed else 'none'}",
        f"LAST_ACTION: {_compact_json(state.last_action) if state.last_action else 'n/a'}",
        f"LAST_VERIFICATION: {_compact_json(state.last_verification) if state.last_verification else 'n/a'}",
        f"NEXT_ACTION: {state.next_action or (active.title if active else 'n/a')}",
        f"OPEN_QUESTIONS: {', '.join(state.open_questions) if state.open_questions else 'none'}",
    ]
    if budget is not None:
        lines.append(
            "PROMPT_BUDGET: "
            f"usable={budget.usable_prompt_budget} "
            f"remaining={max(0, budget.remaining_budget)} "
            f"docs={budget.docs_budget} "
            f"layer1={budget.layer1_budget} "
            f"layer2={budget.layer2_budget} "
            f"layer3={budget.layer3_budget}"
        )
    lines.append("Work on the active step only. If the step is too broad, split it instead of carrying excess context.")
    return "\n".join(lines)


def update_state_after_turn(
    state: HarnessState,
    *,
    tool_events: list[dict[str, Any]],
    assistant_text: str,
) -> HarnessState:
    next_state = HarnessState.from_dict(state.to_dict())
    active = active_step(next_state)
    read_ops = {"read_file", "load_file", "glob", "grep", "list_directory"}
    write_ops = {"write_file", "edit_file"}
    saw_read = any(event.get("tool") in read_ops and event.get("ok", True) for event in tool_events)
    saw_write = any(event.get("tool") in write_ops and event.get("ok", True) for event in tool_events)
    verification = next((event for event in tool_events if event.get("verification")), None)

    if tool_events:
        last = tool_events[-1]
        next_state.last_action = {
            "type": last.get("tool"),
            "target": last.get("target"),
            "summary": last.get("summary"),
        }

    if verification:
        next_state.last_verification = {
            "status": "pass" if verification.get("ok") else "fail",
            "summary": verification.get("summary"),
            "command": verification.get("command"),
        }

    if active:
        if active.kind == "inspect" and saw_read:
            _complete_and_advance(next_state, active.id)
        elif active.kind in {"change", "fix"} and saw_write:
            _complete_and_advance(next_state, active.id)
        elif active.kind == "verify" and verification and verification.get("ok"):
            _complete_and_advance(next_state, active.id)
        elif active.kind in {"review", "report", "answer"} and assistant_text.strip():
            _complete_and_advance(next_state, active.id)
        elif verification and not verification.get("ok"):
            next_state.failure_count_for_active_step += 1

    active = active_step(next_state)
    next_state.next_action = active.title if active else "Await the next user instruction."
    next_state.updated_at = time.time()
    return next_state


def available_skill_names(root: Path) -> list[str]:
    skills_dir = root / ".openjet" / "skills"
    if not skills_dir.exists():
        return []
    return sorted(path.stem for path in skills_dir.glob("*.md"))


def set_mode(state: HarnessState, mode: str) -> HarnessState:
    next_state = HarnessState.from_dict(state.to_dict())
    next_state.mode = mode
    next_state.updated_at = time.time()
    return next_state


def set_preferred_skills(state: HarnessState, skill_names: list[str]) -> HarnessState:
    next_state = HarnessState.from_dict(state.to_dict())
    normalized: list[str] = []
    seen: set[str] = set()
    for name in skill_names:
        key = normalize_skill_name(name)
        if not key or key in seen:
            continue
        seen.add(key)
        normalized.append(key)
    next_state.preferred_skills = normalized
    next_state.updated_at = time.time()
    return next_state


def clear_preferred_skills(state: HarnessState) -> HarnessState:
    next_state = HarnessState.from_dict(state.to_dict())
    next_state.preferred_skills = []
    next_state.updated_at = time.time()
    return next_state


def advance_step(state: HarnessState) -> HarnessState:
    next_state = HarnessState.from_dict(state.to_dict())
    active = active_step(next_state)
    if active:
        _complete_and_advance(next_state, active.id)
        active = active_step(next_state)
        next_state.next_action = active.title if active else "Await the next user instruction."
    next_state.updated_at = time.time()
    return next_state


def split_active_step(state: HarnessState) -> HarnessState:
    next_state = HarnessState.from_dict(state.to_dict())
    active = active_step(next_state)
    if not active:
        return next_state
    index = next((idx for idx, step in enumerate(next_state.plan) if step.id == active.id), None)
    if index is None:
        return next_state
    active.status = "done"
    prefix = active.title.rstrip(".")
    files = active.files[:]
    skill = active.skill
    split_steps = [
        PlanStep(f"{active.id}_a", f"{prefix}: inspect narrowly", "active", "inspect", files=files, skill=skill, acceptance="narrow the target scope"),
        PlanStep(f"{active.id}_b", f"{prefix}: implement the smallest change", "pending", "change", files=files, skill=skill, acceptance="complete the narrow change"),
        PlanStep(f"{active.id}_c", f"{prefix}: verify the result", "pending", "verify", files=files, skill=skill, acceptance="confirm the narrowed step"),
    ]
    next_state.plan[index:index + 1] = split_steps
    next_state.active_step_id = split_steps[0].id
    next_state.next_action = split_steps[0].title
    next_state.failure_count_for_active_step = 0
    next_state.updated_at = time.time()
    return next_state


def normalize_skill_name(name: str) -> str:
    return Path(str(name).strip()).stem


def allowed_tools_for_mode(mode: str) -> set[str]:
    return set(TOOL_BUNDLES.get(mode, TOOL_BUNDLES["code"]))


def shell_command_is_verification(command: str) -> bool:
    lowered = command.lower()
    return any(token in lowered for token in ("pytest", "unittest", "py_compile", "ruff", "mypy", "cargo test", "go test", "npm test"))


def max_skill_docs_for_window(window_tokens: int) -> int:
    if window_tokens <= 4096:
        return 1
    if window_tokens <= 8192:
        return 1
    if window_tokens <= 16384:
        return 2
    return 3


def active_step(state: HarnessState) -> PlanStep | None:
    if state.active_step_id:
        for step in state.plan:
            if step.id == state.active_step_id:
                return step
    for step in state.plan:
        if step.status == "active":
            return step
    return None


def _complete_and_advance(state: HarnessState, step_id: str) -> None:
    found_active = False
    for step in state.plan:
        if step.id == step_id:
            step.status = "done"
            found_active = True
            continue
        if found_active and step.status == "pending":
            step.status = "active"
            state.active_step_id = step.id
            return
    state.active_step_id = None


def _candidate_docs(root: Path, state: HarnessState, window_tokens: int) -> list[HarnessDocCandidate]:
    role_name = ROLE_BY_MODE.get(state.mode, "coder")
    candidates: list[tuple[str, str, str]] = []
    repo_index = load_repo_context_index(root)
    if repo_index.project_summary:
        candidates.append(("layer1", "[project summary]", repo_index.project_summary))
    base_doc = _load_doc(root / ".openjet" / "agents" / "base.md")
    candidates.append(("layer1", ".openjet/agents/base.md", base_doc or DEFAULT_BASE_PROMPT))
    role_doc = _load_doc(root / ".openjet" / "agents" / f"{role_name}.md")
    candidates.append(("layer2", f".openjet/agents/{role_name}.md", role_doc or DEFAULT_ROLE_PROMPTS.get(role_name, "")))
    project_doc = _load_doc(root / ".openjet" / "projects" / "default.md")
    if project_doc:
        candidates.append(("layer1", ".openjet/projects/default.md", project_doc))

    for file_summary in _file_context_docs(repo_index, state):
        candidates.append(("layer2", file_summary[0], file_summary[1]))

    selected_skills = _select_skills(root, state, window_tokens)
    candidates.extend(selected_skills)
    candidates.extend(_recent_context_docs(state))

    formatted: list[HarnessDocCandidate] = []
    for layer_name, label, body in candidates:
        rendered = _format_doc(label, body) if body.strip() else ""
        formatted.append(
            HarnessDocCandidate(
                layer=layer_name,
                label=label,
                body=rendered,
                token_count=estimate_tokens(rendered),
            )
        )
    return formatted


def _select_skills(root: Path, state: HarnessState, window_tokens: int) -> list[tuple[str, str, str]]:
    skills_dir = root / ".openjet" / "skills"
    if not skills_dir.exists():
        return []
    query = " ".join(
        part
        for part in [
            state.goal,
            active_step(state).title if active_step(state) else "",
            " ".join(state.files_in_play),
        ]
        if part
    ).lower()
    query_terms = set(re.findall(r"[a-z0-9_+-]+", query))
    scored: list[tuple[int, str, str]] = []
    preferred = {normalize_skill_name(name) for name in state.preferred_skills}
    for path in sorted(skills_dir.glob("*.md")):
        body = _load_doc(path)
        if not body:
            continue
        metadata, content = _split_frontmatter(body)
        score = 0
        name_terms = set(re.findall(r"[a-z0-9_+-]+", path.stem.lower()))
        tags = {str(tag).lower() for tag in metadata.get("tags", [])} if isinstance(metadata.get("tags"), list) else set()
        score += len(query_terms & name_terms) * 4
        score += len(query_terms & tags) * 5
        if metadata.get("mode") == state.mode:
            score += 3
        if path.stem in preferred:
            score += 100
        if score <= 0:
            continue
        scored.append((score, path.name, content.strip()))
    scored.sort(key=lambda item: (-item[0], item[1]))
    limit = max_skill_docs_for_window(window_tokens)
    active = active_step(state)
    if active and active.skill:
        limit = max(limit, 1)
    return [("layer2", f"skills/{name}", body) for _, name, body in scored[:limit]]


def _file_context_docs(index: Any, state: HarnessState) -> list[tuple[str, str]]:
    docs: list[tuple[str, str]] = []
    seen: set[str] = set()
    for raw_path in state.files_in_play:
        normalized = str(raw_path).strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        summary = lookup_file_summary(index, normalized)
        if summary is None:
            continue
        lines = [
            f"FILE CONTEXT: {summary.path}",
            f"Purpose: {summary.purpose}",
        ]
        if summary.related_tests:
            lines.append(f"Related tests: {', '.join(summary.related_tests)}")
        docs.append((f"file-context:{summary.path}", "\n".join(lines)))
    return docs


def _recent_context_docs(state: HarnessState) -> list[tuple[str, str, str]]:
    lines = ["RECENT TASK CONTEXT"]
    if state.last_action:
        lines.append(f"Last action: {_compact_json(state.last_action)}")
    if state.last_verification:
        lines.append(f"Last verification: {_compact_json(state.last_verification)}")
    if state.files_in_play:
        lines.append(f"Files in play: {', '.join(state.files_in_play[:4])}")
    if len(lines) == 1:
        return []
    return [("layer3", "recent-context", "\n".join(lines))]


def layered_context_config(raw: dict[str, Any] | None) -> LayeredContextConfig:
    payload = dict(raw or {})
    enabled = bool(payload.get("enabled", True))
    return LayeredContextConfig(
        enabled=enabled,
        layer1_enabled=enabled and bool(payload.get("layer1_enabled", True)),
        layer2_enabled=enabled and bool(payload.get("layer2_enabled", True)),
        layer3_enabled=enabled and bool(payload.get("layer3_enabled", True)),
        layer1_ratio=_clamp_ratio(payload.get("layer1_ratio"), default=0.15),
        layer2_ratio=_clamp_ratio(payload.get("layer2_ratio"), default=0.20),
        layer3_ratio=_clamp_ratio(payload.get("layer3_ratio"), default=0.10),
        alert_ratio=_clamp_ratio(payload.get("alert_ratio"), default=0.40),
    )


def _layer_enabled(config: LayeredContextConfig, layer_name: str) -> bool:
    if not config.enabled:
        return False
    if layer_name == "layer1":
        return config.layer1_enabled
    if layer_name == "layer2":
        return config.layer2_enabled
    if layer_name == "layer3":
        return config.layer3_enabled
    return True


def _clamp_ratio(value: Any, *, default: float) -> float:
    try:
        ratio = float(value)
    except (TypeError, ValueError):
        return default
    return max(0.0, min(ratio, 1.0))


def _load_doc(path: Path) -> str:
    try:
        if path.exists():
            return path.read_text(encoding="utf-8").strip()
    except Exception:
        return ""
    return ""


def _split_frontmatter(body: str) -> tuple[dict[str, Any], str]:
    if not body.startswith("---\n"):
        return {}, body
    try:
        _, rest = body.split("---\n", 1)
        frontmatter, content = rest.split("\n---\n", 1)
        loaded = yaml.safe_load(frontmatter) or {}
        if isinstance(loaded, dict):
            return loaded, content
    except Exception:
        return {}, body
    return {}, body


def _format_doc(label: str, body: str) -> str:
    return f"USER-EDITABLE HARNESS DOC: {label}\n{body.strip()}"


def _compact_json(payload: dict[str, Any]) -> str:
    try:
        return json.dumps(payload, ensure_ascii=True, separators=(",", ":"))
    except Exception:
        return str(payload)


def _shorten(text: str, max_len: int) -> str:
    compact = " ".join(text.split())
    if len(compact) <= max_len:
        return compact
    return compact[: max_len - 3] + "..."
