from __future__ import annotations

from contextlib import suppress
from pathlib import Path
from typing import Any, Awaitable, Callable

from .harness import HarnessState, allowed_tools_for_state
from .model_profiles import apply_model_profile
from .sdk import OpenJetSession
from .tool_executor import ToolExecutionResult
from .tools.registry import ToolSpec, register_tool, unregister_tool


HYBRID_MODE = "hybrid"
DELEGATE_LOCAL_TOOL = "delegate_local"
TARGET_CODEX_SHARE = 0.20
TARGET_QUALITY_RETENTION = 0.99

ORCHESTRATOR_SYSTEM_PROMPT = """
You are the primary reasoning and quality agent in OpenJet Slipstream mode. Act as an
orchestrator: own intent, architecture, risk decisions, task decomposition, and final
review. Delegate the high-volume implementation loop to delegate_local whenever the
work can be described with concrete acceptance criteria. This includes repository
exploration, routine implementation, repetitive edits, test execution, debugging, and
mechanical follow-up fixes. Keep direct work for ambiguous/high-risk decisions, small
checks needed to direct the worker, and independent review of the resulting diff.

Aim to keep frontier-model token use near 20% of the combined model tokens while
targeting at least 99% of the quality of a Codex-only result. Do not ask the worker for
its transcript or reproduce its exploration. Give it a bounded task and acceptance
criteria, consume its concise result, inspect only the relevant diff/test evidence,
and delegate corrections when needed. Never claim success until you have reviewed the
implementation and verification.
""".strip()

IMPLEMENTER_SYSTEM_PROMPT = """
You are OpenJet's local implementation agent. Perform the substantial repository work
delegated by the primary orchestrator: inspect the code, edit files, run appropriate
tests, diagnose failures, and iterate until the stated acceptance criteria are met.
Work autonomously within the requested scope and obey repository instructions. Do not
delegate work to another model. Finish with a concise handoff containing: files changed,
tests and outcomes, remaining blockers or risks, and the minimum facts the orchestrator
needs to review. Do not include a transcript of your exploration.
""".strip()


def execution_mode(cfg: dict[str, Any]) -> str:
    if str(cfg.get("execution_mode") or "").strip().lower() == HYBRID_MODE:
        return HYBRID_MODE
    runtime = str(cfg.get("runtime") or "llama_cpp")
    if runtime == "llama_cpp":
        return "local"
    return "codex" if runtime == "openai_codex" else "cloud"


class HybridWorker:
    """A warm local implementation agent owned by the foreground Codex runtime."""

    def __init__(
        self,
        *,
        session: OpenJetSession,
        model_ref: str,
        profile_name: str,
        harness_state: HarnessState,
    ) -> None:
        self.session = session
        self.model_ref = model_ref
        self.profile_name = profile_name
        self.harness_state = harness_state
        self.ready = False
        self._delegating = False

    @classmethod
    async def start(
        cls,
        *,
        base_cfg: dict[str, Any],
        local_profile: dict[str, Any],
        root: Path,
        approval_handler: Callable[[Any], bool | Awaitable[bool]] | None,
        trace_hook: Callable[[str, dict[str, object]], None] | None = None,
    ) -> "HybridWorker":
        local_cfg = dict(base_cfg)
        apply_model_profile(local_cfg, local_profile)
        local_cfg["execution_mode"] = "local"
        state = HarnessState(
            mode="code",
            plan_approved=True,
            constraints=["local-first", "stay within the delegated scope", "verify changes"],
        )

        def replace_state(updated: HarnessState) -> None:
            # Keep a stable object so the worker and the SDK callbacks always observe
            # the same todo/verification state after control tools replace it.
            state.__dict__.clear()
            state.__dict__.update(updated.__dict__)

        session = await OpenJetSession.create(
            cfg=local_cfg,
            system_prompt=IMPLEMENTER_SYSTEM_PROMPT,
            root=root,
            approval_handler=approval_handler,
            allowed_tools=allowed_tools_for_state(state),
            harness_state_getter=lambda: state,
            harness_state_setter=replace_state,
        )
        worker = cls(
            session=session,
            model_ref=str(local_profile.get("llama_model") or ""),
            profile_name=str(local_profile.get("name") or "local"),
            harness_state=state,
        )
        try:
            # Slipstream is deliberately warm: selecting it starts and health-checks
            # llama.cpp now, rather than on the first delegated call.
            await session.agent.client.start()
            session.agent.trace_hook = trace_hook
            worker.ready = True
            worker._register_tool()
            return worker
        except Exception:
            with suppress(Exception):
                await session.close()
            raise

    async def delegate(self, args: dict[str, Any]) -> ToolExecutionResult:
        task = str(args.get("task") or "").strip()
        acceptance = str(args.get("acceptance_criteria") or "").strip()
        if not task:
            return ToolExecutionResult(
                output="delegate_local requires a non-empty task.",
                meta={"ok": False, "status": "invalid_arguments"},
            )
        if self._delegating:
            return ToolExecutionResult(
                output="The local implementation worker is already running.",
                meta={"ok": False, "status": "busy"},
            )

        self._delegating = True
        unregister_tool(DELEGATE_LOCAL_TOOL)
        self.harness_state.goal = task
        self.harness_state.mode = "code"
        self.harness_state.plan_approved = True
        self.session.agent.reset_conversation()
        prompt = f"Delegated task:\n{task}"
        if acceptance:
            prompt += f"\n\nAcceptance criteria:\n{acceptance}"
        prompt += "\n\nImplement, test, iterate, then return only the concise handoff requested by your system instructions."
        try:
            response = await self.session.run(prompt)
            self.ready = True
            failed_tools = [result for result in response.tool_results if not result.ok]
            return ToolExecutionResult(
                output=response.text.strip() or "Local worker completed without a textual handoff.",
                meta={
                    "ok": True,
                    "status": "completed" if not failed_tools else "completed_with_tool_failures",
                    "model_ref": self.model_ref,
                    "tool_calls": len(response.tool_results),
                    "failed_tool_calls": len(failed_tools),
                },
            )
        except Exception as exc:
            self.ready = False
            return ToolExecutionResult(
                output=f"Local worker failed: {str(exc).strip() or type(exc).__name__}",
                meta={"ok": False, "status": "failed", "model_ref": self.model_ref},
            )
        finally:
            self._delegating = False
            self._register_tool()

    async def close(self) -> None:
        self.ready = False
        unregister_tool(DELEGATE_LOCAL_TOOL)
        await self.session.close()

    def _register_tool(self) -> None:
        unregister_tool(DELEGATE_LOCAL_TOOL)
        register_tool(
            ToolSpec(
                name=DELEGATE_LOCAL_TOOL,
                description=(
                    "Delegate substantial repository exploration, implementation, edits, tests, or debugging "
                    "to the warm local implementation agent. Provide a bounded task and concrete acceptance "
                    "criteria. The result is a concise handoff, not the worker transcript."
                ),
                parameters={
                    "task": {"type": "string", "description": "The complete bounded implementation task."},
                    "acceptance_criteria": {
                        "type": "string",
                        "description": "Specific correctness and verification requirements.",
                    },
                },
                required=("task",),
                confirmation_required=False,
                modes=frozenset({"chat", "code", "review", "debug"}),
                tags=frozenset({"hybrid", "delegation"}),
                executor=self.delegate,
            )
        )
