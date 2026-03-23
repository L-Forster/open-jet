from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.harness import (
    HarnessState,
    _candidate_docs,
    _file_context_docs,
    _recent_context_docs,
    active_step,
    build_state_summary,
    build_turn_context,
    compute_turn_budget,
    layered_context_config,
    max_skill_docs_for_window,
    update_state_after_turn,
    update_state_for_user_message,
)

from tests.context_helpers import memory_snapshot, skill_doc, write_repo_fixture


def _labels(context) -> list[str]:
    return list(context.docs_loaded)


def _decision_lookup(context, label: str) -> dict:
    for item in context.candidate_decisions:
        if item["label"] == label:
            return item
    raise KeyError(label)


class LayeredContextConfigTests(unittest.TestCase):
    def test_defaults(self) -> None:
        config = layered_context_config(None)

        self.assertTrue(config.enabled)
        self.assertTrue(config.layer1_enabled)
        self.assertTrue(config.layer2_enabled)
        self.assertTrue(config.layer3_enabled)
        self.assertEqual(config.layer1_ratio, 0.15)
        self.assertEqual(config.layer2_ratio, 0.20)
        self.assertEqual(config.layer3_ratio, 0.10)
        self.assertEqual(config.alert_ratio, 0.40)

    def test_master_switch_disables_all_layers(self) -> None:
        config = layered_context_config({"enabled": False})

        self.assertFalse(config.enabled)
        self.assertFalse(config.layer1_enabled)
        self.assertFalse(config.layer2_enabled)
        self.assertFalse(config.layer3_enabled)

    def test_individual_switches_only_disable_their_layer(self) -> None:
        config = layered_context_config({"layer2_enabled": False})

        self.assertTrue(config.layer1_enabled)
        self.assertFalse(config.layer2_enabled)
        self.assertTrue(config.layer3_enabled)

    def test_ratios_and_alert_ratio_are_clamped(self) -> None:
        config = layered_context_config(
            {
                "layer1_ratio": 9,
                "layer2_ratio": -1,
                "layer3_ratio": "0.25",
                "alert_ratio": "2.2",
            }
        )

        self.assertEqual(config.layer1_ratio, 1.0)
        self.assertEqual(config.layer2_ratio, 0.0)
        self.assertEqual(config.layer3_ratio, 0.25)
        self.assertEqual(config.alert_ratio, 1.0)


class TurnBudgetTests(unittest.TestCase):
    def test_budget_derivation_for_multiple_windows(self) -> None:
        budget_4k = compute_turn_budget(
            effective_window=4096,
            current_context_tokens=0,
            memory_snapshot=memory_snapshot(8192, 4096),
        )
        budget_8k = compute_turn_budget(
            effective_window=8192,
            current_context_tokens=0,
            memory_snapshot=memory_snapshot(8192, 4096),
        )

        self.assertEqual(budget_4k.generation_reserve, 737)
        self.assertEqual(budget_4k.tool_reserve, 163)
        self.assertEqual(budget_4k.system_reserve, 300)
        self.assertEqual(budget_4k.base_prompt_budget, 2896)
        self.assertEqual(budget_4k.docs_budget, 810)
        self.assertEqual(budget_8k.system_reserve, 300)
        self.assertGreater(budget_8k.usable_prompt_budget, budget_4k.usable_prompt_budget)
        self.assertGreater(budget_8k.docs_budget, budget_4k.docs_budget)

    def test_ram_factor_thresholds(self) -> None:
        expectations = [
            (None, 1.0),
            (699, 0.35),
            (700, 0.50),
            (999, 0.50),
            (1000, 0.65),
            (1499, 0.65),
            (1500, 0.80),
            (2499, 0.80),
            (2500, 1.00),
        ]
        for available_mb, expected in expectations:
            with self.subTest(available_mb=available_mb):
                snapshot = memory_snapshot(8192, available_mb) if available_mb is not None else None
                budget = compute_turn_budget(
                    effective_window=4096,
                    current_context_tokens=0,
                    memory_snapshot=snapshot,
                )
                self.assertEqual(budget.ram_factor, expected)

    def test_docs_budget_has_floor_when_remaining_is_small_or_negative(self) -> None:
        small = compute_turn_budget(
            effective_window=4096,
            current_context_tokens=2800,
            memory_snapshot=memory_snapshot(8192, 4096),
        )
        negative = compute_turn_budget(
            effective_window=4096,
            current_context_tokens=4000,
            memory_snapshot=memory_snapshot(8192, 4096),
        )

        self.assertEqual(small.docs_budget, 192)
        self.assertEqual(negative.docs_budget, 192)
        self.assertLess(negative.remaining_budget, 0)

    def test_layer_budgets_respect_configured_ratios_and_disabled_layers(self) -> None:
        config = layered_context_config(
            {
                "layer1_enabled": False,
                "layer2_ratio": 0.05,
                "layer3_ratio": 0.02,
            }
        )
        budget = compute_turn_budget(
            effective_window=4096,
            current_context_tokens=0,
            memory_snapshot=memory_snapshot(8192, 4096),
            layered_config=config,
        )

        self.assertEqual(budget.layer1_budget, 0)
        self.assertEqual(budget.layer2_budget, int(4096 * 0.05))
        self.assertEqual(budget.layer3_budget, int(4096 * 0.02))

    def test_outputs_do_not_go_negative(self) -> None:
        budget = compute_turn_budget(
            effective_window=1024,
            current_context_tokens=999999,
            memory_snapshot=memory_snapshot(4096, 32),
        )

        self.assertGreaterEqual(budget.usable_prompt_budget, 512)
        self.assertGreaterEqual(budget.docs_budget, 192)
        self.assertGreaterEqual(budget.layer1_budget, 0)
        self.assertGreaterEqual(budget.layer2_budget, 0)
        self.assertGreaterEqual(budget.layer3_budget, 0)


class StateSummaryTests(unittest.TestCase):
    def test_state_summary_includes_expected_fields_and_budget_line(self) -> None:
        state = update_state_for_user_message(HarnessState(), "Fix the harness", mode="debug", files=["src/harness.py"])
        budget = compute_turn_budget(
            effective_window=4096,
            current_context_tokens=800,
            memory_snapshot=memory_snapshot(8192, 4096),
        )

        summary = build_state_summary(state, budget)

        self.assertIn("OPEN-JET HARNESS STATE", summary)
        self.assertIn("MODE: debug", summary)
        self.assertIn("GOAL: Fix the harness", summary)
        self.assertIn("FILES_IN_PLAY: src/harness.py", summary)
        self.assertIn("PROMPT_BUDGET:", summary)

    def test_state_summary_is_deterministic(self) -> None:
        state = update_state_for_user_message(HarnessState(), "Implement change", files=["src/harness.py"])
        budget = compute_turn_budget(
            effective_window=4096,
            current_context_tokens=400,
            memory_snapshot=memory_snapshot(8192, 4096),
        )

        self.assertEqual(build_state_summary(state, budget), build_state_summary(state, budget))


class CandidateDocTests(unittest.TestCase):
    def test_candidate_order_and_layer_assignment(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_repo_fixture(
                root,
                architecture_lines=["- `src/harness.py`: owns layered context budgeting."],
                skills={
                    "python-harness.md": skill_doc(tags=["python", "harness"], mode="code", body="python harness skill"),
                },
                repo_files={"src/harness.py": "pass\n"},
            )
            state = update_state_for_user_message(
                HarnessState(preferred_skills=["python-harness"]),
                "Implement a Python harness change",
                mode="code",
                files=["src/harness.py"],
            )
            state.last_action = {"type": "read_file", "target": "src/harness.py", "summary": "read"}
            state.last_verification = {"status": "fail", "summary": "pytest failed", "command": "pytest tests/test_context_harness.py"}

            candidates = _candidate_docs(root, state, 4096)

        self.assertEqual(
            [(candidate.layer, candidate.label) for candidate in candidates],
            [
                ("layer1", "[project summary]"),
                ("layer1", ".openjet/agents/base.md"),
                ("layer1", "skills.md"),
                ("layer2", ".openjet/agents/coder.md"),
                ("layer1", ".openjet/projects/default.md"),
                ("layer2", "file-context:src/harness.py"),
                ("layer2", "skills/python-harness.md"),
                ("layer3", "recent-context"),
            ],
        )

    def test_file_context_docs_only_exist_when_summaries_exist(self) -> None:
        index = type("Index", (), {"files": {"src/harness.py": type("Summary", (), {"path": "src/harness.py", "purpose": "owns context", "related_tests": ("tests/test_context_harness.py",)})()}})()
        state = HarnessState(files_in_play=["src/harness.py", "src/missing.py"])

        docs = _file_context_docs(index, state)

        self.assertEqual(len(docs), 1)
        self.assertEqual(docs[0][0], "file-context:src/harness.py")

    def test_recent_context_doc_absent_without_recent_state(self) -> None:
        self.assertEqual(_recent_context_docs(HarnessState()), [])

    def test_recent_context_doc_present_with_action_or_verification(self) -> None:
        state = HarnessState(
            last_action={"type": "read_file", "target": "src/harness.py", "summary": "read"},
            last_verification={"status": "fail", "summary": "pytest failed"},
            files_in_play=["src/harness.py"],
        )

        docs = _recent_context_docs(state)

        self.assertEqual(docs[0][0], "layer3")
        self.assertEqual(docs[0][1], "recent-context")
        self.assertIn("Last verification", docs[0][2])

    def test_skill_selection_honors_preference_mode_query_and_window_limit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_repo_fixture(
                root,
                architecture_lines=["- `src/harness.py`: owns layered context budgeting."],
                skills={
                    "python-harness.md": skill_doc(tags=["python", "harness"], mode="code", body="preferred harness skill"),
                    "jetson-memory.md": skill_doc(tags=["jetson", "memory"], mode="code", body="jetson memory skill"),
                    "context-compare.md": skill_doc(tags=["context", "compare"], mode="code", body="comparison skill"),
                    "review-only.md": skill_doc(tags=["review"], mode="review", body="review skill"),
                },
            )
            state = update_state_for_user_message(
                HarnessState(preferred_skills=["python-harness"]),
                "Implement a Python harness comparison for Jetson memory pressure",
                mode="code",
                files=["src/harness.py"],
            )

            small_window = [candidate.label for candidate in _candidate_docs(root, state, 4096) if candidate.label.startswith("skills/")]
            wide_window = [candidate.label for candidate in _candidate_docs(root, state, 20000) if candidate.label.startswith("skills/")]

        self.assertEqual(max_skill_docs_for_window(4096), 1)
        self.assertEqual(small_window, ["skills/python-harness.md"])
        self.assertEqual(max_skill_docs_for_window(20000), 3)
        self.assertEqual(
            wide_window,
            ["skills/python-harness.md", "skills/jetson-memory.md", "skills/context-compare.md"],
        )


class BuildTurnContextTests(unittest.TestCase):
    def test_state_summary_is_first_system_message_and_context_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_repo_fixture(
                root,
                architecture_lines=["- `src/harness.py`: owns layered context budgeting."],
                repo_files={"src/harness.py": "pass\n"},
            )
            state = update_state_for_user_message(HarnessState(), "Implement change", files=["src/harness.py"])

            first = build_turn_context(
                root=root,
                state=state,
                current_context_tokens=400,
                effective_window=4096,
                memory_snapshot=memory_snapshot(8192, 4096),
            )
            second = build_turn_context(
                root=root,
                state=state,
                current_context_tokens=400,
                effective_window=4096,
                memory_snapshot=memory_snapshot(8192, 4096),
            )

        self.assertEqual(first.messages[0]["role"], "system")
        self.assertEqual(first.messages[0]["content"], first.state_summary)
        self.assertEqual(first.docs_loaded, second.docs_loaded)
        self.assertEqual(first.layer_tokens, second.layer_tokens)
        self.assertEqual(first.candidate_decisions, second.candidate_decisions)

    def test_respects_global_docs_budget_per_layer_budgets_and_records_trace(self) -> None:
        large_skill = skill_doc(tags=["python"], mode="code", body=" ".join(["skill detail"] * 500))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_repo_fixture(
                root,
                architecture_lines=["- `src/harness.py`: owns layered context budgeting."],
                skills={"python-heavy.md": large_skill},
                repo_files={"src/harness.py": "pass\n"},
            )
            state = update_state_for_user_message(
                HarnessState(preferred_skills=["python-heavy"]),
                "Implement a Python harness change",
                files=["src/harness.py"],
            )
            state.last_action = {"type": "read_file", "target": "src/harness.py", "summary": "read"}

            context = build_turn_context(
                root=root,
                state=state,
                current_context_tokens=1200,
                effective_window=4096,
                memory_snapshot=memory_snapshot(4096, 1500),
            )

        self.assertLessEqual(context.docs_tokens, context.state_summary_tokens + context.budget.docs_budget)
        self.assertLessEqual(context.layer_tokens["layer1"], context.budget.layer1_budget)
        self.assertLessEqual(context.layer_tokens["layer2"], context.budget.layer2_budget)
        self.assertLessEqual(context.layer_tokens["layer3"], context.budget.layer3_budget)
        self.assertEqual(sorted(context.layer_docs), ["layer1", "layer2", "layer3"])
        self.assertIn("[project summary]", context.docs_loaded)
        self.assertEqual(_decision_lookup(context, "skills/python-heavy.md")["admitted"], False)

    def test_respects_enabled_and_disabled_layers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_repo_fixture(
                root,
                architecture_lines=["- `src/harness.py`: owns layered context budgeting."],
                repo_files={"src/harness.py": "pass\n"},
            )
            state = update_state_for_user_message(HarnessState(), "Implement change", files=["src/harness.py"])
            state.last_action = {"type": "read_file", "target": "src/harness.py", "summary": "read"}

            context = build_turn_context(
                root=root,
                state=state,
                current_context_tokens=400,
                effective_window=4096,
                memory_snapshot=memory_snapshot(8192, 4096),
                layered_config={"layer2_enabled": False},
            )

        self.assertFalse(any(label.startswith("file-context:") for label in context.docs_loaded))
        self.assertEqual(context.layer_tokens["layer2"], 0)
        self.assertEqual(_decision_lookup(context, "file-context:src/harness.py")["skipped_reason"], "disabled_layer")

    def test_budget_alerts_emit_when_threshold_crosses_custom_ratio(self) -> None:
        big_project_doc = " ".join(["project guidance"] * 200)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_repo_fixture(
                root,
                architecture_lines=["- `src/harness.py`: owns layered context budgeting."],
                project_doc=big_project_doc,
                repo_files={"src/harness.py": "pass\n"},
            )
            state = update_state_for_user_message(HarnessState(), "Implement change", files=["src/harness.py"])

            context = build_turn_context(
                root=root,
                state=state,
                current_context_tokens=200,
                effective_window=4096,
                memory_snapshot=memory_snapshot(8192, 4096),
                layered_config={"alert_ratio": 0.01},
            )

        self.assertTrue(context.budget_alerts)

    def test_oversized_docs_are_skipped_without_partial_inclusion(self) -> None:
        huge_project_doc = " ".join(["project guidance"] * 5000)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_repo_fixture(
                root,
                architecture_lines=["- `src/harness.py`: owns layered context budgeting."],
                project_doc=huge_project_doc,
                repo_files={"src/harness.py": "pass\n"},
            )
            state = update_state_for_user_message(HarnessState(), "Implement change", files=["src/harness.py"])

            context = build_turn_context(
                root=root,
                state=state,
                current_context_tokens=2400,
                effective_window=4096,
                memory_snapshot=memory_snapshot(8192, 4096),
            )

        self.assertNotIn(".openjet/projects/default.md", context.docs_loaded)
        self.assertEqual(_decision_lookup(context, ".openjet/projects/default.md")["admitted"], False)


class HarnessScenarioTests(unittest.TestCase):
    def test_narrow_coding_task_loads_expected_docs_in_sensible_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_repo_fixture(
                root,
                architecture_lines=[
                    "- `src/harness.py`: owns layered context budgeting.",
                    "- `src/context_index.py`: resolves file summaries.",
                ],
                role_docs={"coder": "coder guidance with focused coding steps"},
                project_doc="project doc for coding",
                skills={"python-harness.md": skill_doc(tags=["python", "harness"], mode="code", body="python harness skill")},
                repo_files={"src/harness.py": "pass\n"},
            )
            state = update_state_for_user_message(
                HarnessState(preferred_skills=["python-harness"]),
                "Implement a narrow Python harness change",
                files=["src/harness.py"],
            )

            context = build_turn_context(
                root=root,
                state=state,
                current_context_tokens=300,
                effective_window=4096,
                memory_snapshot=memory_snapshot(8192, 4096),
            )

        self.assertEqual(
            _labels(context)[:5],
            [
                "[project summary]",
                ".openjet/agents/base.md",
                "skills.md",
                ".openjet/projects/default.md",
                "file-context:src/harness.py",
            ],
        )
        self.assertIn("skills/python-harness.md", _labels(context))

    def test_debug_loop_preserves_recent_context_until_pressure_drops_it(self) -> None:
        big_project_doc = " ".join(["project guidance"] * 120)
        big_file_summary = " ".join(["owns the debug loop and failing verification handling"] * 60)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_repo_fixture(
                root,
                architecture_lines=[f"- `src/harness.py`: {big_file_summary}"],
                project_doc=big_project_doc,
                role_docs={"debugger": "debugger guidance"},
                repo_files={"src/harness.py": "pass\n"},
            )
            state = update_state_for_user_message(HarnessState(), "Debug failing verification", mode="debug", files=["src/harness.py"])
            state = update_state_after_turn(
                state,
                tool_events=[
                    {
                        "tool": "shell",
                        "ok": False,
                        "summary": "pytest failed in debug loop",
                        "target": "python -m unittest tests.test_context_harness",
                        "verification": True,
                        "command": "python -m unittest tests.test_context_harness",
                    }
                ],
                assistant_text="verification still failing",
            )

            low_pressure = build_turn_context(
                root=root,
                state=state,
                current_context_tokens=400,
                effective_window=4096,
                memory_snapshot=memory_snapshot(8192, 4096),
            )
            high_pressure = build_turn_context(
                root=root,
                state=state,
                current_context_tokens=2450,
                effective_window=4096,
                memory_snapshot=memory_snapshot(8192, 4096),
            )

        self.assertIn("recent-context", low_pressure.docs_loaded)
        self.assertNotIn("recent-context", high_pressure.docs_loaded)
        self.assertFalse(_decision_lookup(high_pressure, "recent-context")["admitted"])

    def test_expanding_file_scope_evolves_file_context_without_duplicates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_repo_fixture(
                root,
                architecture_lines=[
                    "- `src/harness.py`: owns layered context budgeting.",
                    "- `src/context_index.py`: resolves repo context summaries.",
                    "- `tests/test_context_harness.py`: verifies layered context behavior.",
                ],
                repo_files={
                    "src/harness.py": "pass\n",
                    "src/context_index.py": "pass\n",
                    "tests/test_context_harness.py": "pass\n",
                },
            )
            first_state = update_state_for_user_message(HarnessState(), "Implement change", files=["src/harness.py"])
            second_state = update_state_for_user_message(
                first_state,
                "Expand the change to related files and tests",
                files=["src/harness.py", "src/context_index.py", "tests/test_context_harness.py"],
            )

            first = build_turn_context(
                root=root,
                state=first_state,
                current_context_tokens=400,
                effective_window=4096,
                memory_snapshot=memory_snapshot(8192, 4096),
            )
            second = build_turn_context(
                root=root,
                state=second_state,
                current_context_tokens=400,
                effective_window=4096,
                memory_snapshot=memory_snapshot(8192, 4096),
            )

        self.assertEqual([label for label in first.docs_loaded if label.startswith("file-context:")], ["file-context:src/harness.py"])
        self.assertEqual(
            [label for label in second.docs_loaded if label.startswith("file-context:")],
            [
                "file-context:src/harness.py",
                "file-context:src/context_index.py",
                "file-context:tests/test_context_harness.py",
            ],
        )

    def test_high_pressure_4k_window_drops_docs_as_context_tokens_grow(self) -> None:
        large_project_doc = " ".join(["project guidance"] * 80)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_repo_fixture(
                root,
                architecture_lines=["- `src/harness.py`: owns layered context budgeting."],
                project_doc=large_project_doc,
                skills={"python-harness.md": skill_doc(tags=["python", "harness"], mode="code", body="python harness skill")},
                repo_files={"src/harness.py": "pass\n"},
            )
            state = update_state_for_user_message(
                HarnessState(preferred_skills=["python-harness"]),
                "Implement a Python harness change",
                files=["src/harness.py"],
            )
            contexts = [
                build_turn_context(
                    root=root,
                    state=state,
                    current_context_tokens=tokens,
                    effective_window=4096,
                    memory_snapshot=memory_snapshot(8192, 4096),
                )
                for tokens in (300, 1200, 2000, 2500)
            ]

        counts = [len(context.docs_loaded) for context in contexts]
        self.assertGreater(counts[0], counts[-1])
        self.assertIn("skills/python-harness.md", contexts[0].docs_loaded)
        self.assertNotIn("skills/python-harness.md", contexts[-1].docs_loaded)

    def test_ram_pressure_changes_budget_and_admitted_doc_set(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_repo_fixture(
                root,
                architecture_lines=["- `src/harness.py`: owns layered context budgeting."],
                skills={"python-harness.md": skill_doc(tags=["python", "harness"], mode="code", body="python harness skill")},
                repo_files={"src/harness.py": "pass\n"},
            )
            state = update_state_for_user_message(
                HarnessState(preferred_skills=["python-harness"]),
                "Implement a Python harness change",
                files=["src/harness.py"],
            )
            healthy = build_turn_context(
                root=root,
                state=state,
                current_context_tokens=1700,
                effective_window=4096,
                memory_snapshot=memory_snapshot(8192, 4096),
            )
            stressed = build_turn_context(
                root=root,
                state=state,
                current_context_tokens=1700,
                effective_window=4096,
                memory_snapshot=memory_snapshot(8192, 620),
            )

        self.assertGreater(healthy.budget.usable_prompt_budget, stressed.budget.usable_prompt_budget)
        self.assertNotEqual(healthy.docs_loaded, stressed.docs_loaded)

    def test_layer_toggle_comparison_is_exact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_repo_fixture(
                root,
                architecture_lines=["- `src/harness.py`: owns layered context budgeting."],
                skills={"python-harness.md": skill_doc(tags=["python", "harness"], mode="code", body="python harness skill")},
                repo_files={"src/harness.py": "pass\n"},
            )
            state = update_state_for_user_message(
                HarnessState(preferred_skills=["python-harness"]),
                "Implement a Python harness change",
                files=["src/harness.py"],
            )
            state.last_action = {"type": "read_file", "target": "src/harness.py", "summary": "read"}

            all_layers = build_turn_context(
                root=root,
                state=state,
                current_context_tokens=500,
                effective_window=4096,
                memory_snapshot=memory_snapshot(8192, 4096),
            )
            layer1_only = build_turn_context(
                root=root,
                state=state,
                current_context_tokens=500,
                effective_window=4096,
                memory_snapshot=memory_snapshot(8192, 4096),
                layered_config={"layer2_enabled": False, "layer3_enabled": False},
            )
            no_layer3 = build_turn_context(
                root=root,
                state=state,
                current_context_tokens=500,
                effective_window=4096,
                memory_snapshot=memory_snapshot(8192, 4096),
                layered_config={"layer3_enabled": False},
            )
            disabled = build_turn_context(
                root=root,
                state=state,
                current_context_tokens=500,
                effective_window=4096,
                memory_snapshot=memory_snapshot(8192, 4096),
                layered_config={"enabled": False},
            )

        self.assertTrue(any(label.startswith("file-context:") for label in all_layers.docs_loaded))
        self.assertFalse(any(label.startswith("file-context:") for label in layer1_only.docs_loaded))
        self.assertFalse(any(label == "recent-context" for label in no_layer3.docs_loaded))
        self.assertEqual(disabled.docs_loaded, [])

    def test_skill_heavy_state_ranks_relevant_skills(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_repo_fixture(
                root,
                architecture_lines=["- `src/harness.py`: owns layered context budgeting."],
                skills={
                    "python-harness.md": skill_doc(tags=["python", "harness"], mode="code", body="preferred harness skill"),
                    "jetson-memory.md": skill_doc(tags=["jetson", "memory"], mode="code", body="jetson memory skill"),
                    "context-compare.md": skill_doc(tags=["context", "compare"], mode="code", body="context compare skill"),
                    "review-only.md": skill_doc(tags=["review"], mode="review", body="review skill"),
                },
                repo_files={"src/harness.py": "pass\n"},
            )
            state = update_state_for_user_message(
                HarnessState(preferred_skills=["python-harness"]),
                "Implement a Python harness comparison for Jetson memory pressure",
                files=["src/harness.py"],
            )

            context = build_turn_context(
                root=root,
                state=state,
                current_context_tokens=1000,
                effective_window=20000,
                memory_snapshot=memory_snapshot(16384, 8192),
            )

        skills = [label for label in context.docs_loaded if label.startswith("skills/")]
        self.assertEqual(skills[:2], ["skills/python-harness.md", "skills/jetson-memory.md"])

    def test_candidate_starvation_is_visible_and_order_dependent(self) -> None:
        huge_purpose = " ".join(["owns the dominant layer2 budget"] * 20)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_repo_fixture(
                root,
                architecture_lines=[f"- `src/harness.py`: {huge_purpose}"],
                skills={
                    "late-skill.md": skill_doc(
                        tags=["late", "skill"],
                        mode="code",
                        body=" ".join(["tiny late skill"] * 50),
                    )
                },
                project_doc="project guidance",
                repo_files={"src/harness.py": "pass\n"},
            )
            state = update_state_for_user_message(
                HarnessState(preferred_skills=["late-skill"]),
                "Implement a harness change and keep the late skill visible",
                files=["src/harness.py"],
            )
            state.last_action = {"type": "read_file", "target": "src/harness.py", "summary": "read file"}

            context = build_turn_context(
                root=root,
                state=state,
                current_context_tokens=700,
                effective_window=4096,
                memory_snapshot=memory_snapshot(4096, 4096),
                layered_config={"layer2_ratio": 0.08},
            )

        self.assertIn("file-context:src/harness.py", context.docs_loaded)
        self.assertNotIn("skills/late-skill.md", context.docs_loaded)
        self.assertIn(
            _decision_lookup(context, "skills/late-skill.md")["skipped_reason"],
            {"remaining_global_floor_reached", "exceeds_remaining_global_budget"},
        )


if __name__ == "__main__":
    unittest.main()
