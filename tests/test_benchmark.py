from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.benchmark import (
    BenchmarkCase,
    EvalEnvironment,
    EvalExpectation,
    EvalFile,
    default_benchmark_cases,
    evaluate_case,
    run_benchmark_case,
    run_benchmark_suite,
)
from src.runtime_protocol import StreamChunk, ToolCall


class ScriptedRuntimeClient:
    def __init__(self, scripts: list[list[StreamChunk]]) -> None:
        self.model = "fake-model"
        self.context_window_tokens = 4096
        self.gpu_layers = 0
        self._scripts = [list(script) for script in scripts]
        self.calls: list[list[dict]] = []

    async def start(self) -> None:
        return None

    async def close(self) -> None:
        return None

    async def reset_kv_cache(self) -> None:
        return None

    async def chat_stream(self, messages: list[dict], *, use_tools: bool = True):
        self.calls.append(messages)
        script = self._scripts.pop(0)
        for chunk in script:
            yield chunk


class BenchmarkRunnerTests(unittest.IsolatedAsyncioTestCase):
    def test_default_benchmark_cases_load_from_json_files(self) -> None:
        cases = default_benchmark_cases()
        self.assertGreaterEqual(len(cases), 5)
        self.assertIn("ops_env_inventory", {case.case_id for case in cases})

    async def test_run_benchmark_case_grades_filesystem_and_tool_use(self) -> None:
        case = BenchmarkCase(
            case_id="write_case",
            description="Write a file and confirm.",
            prompt="Create note.txt with hello using write_file, then confirm.",
            environment=EvalEnvironment(),
            expectations=[
                EvalExpectation("tool_used", "write_file"),
                EvalExpectation("file_exists", "note.txt"),
                EvalExpectation("file_contains", "note.txt", "hello"),
            ],
            mode="code",
            allowed_tools=["write_file"],
        )
        scripts = [
            [StreamChunk(text="", tool_calls=[ToolCall(name="write_file", arguments={"path": "note.txt", "content": "hello"})])],
            [StreamChunk(text="Done.")],
        ]

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "out"
            artifact = await run_benchmark_case(
                case,
                repeat_index=1,
                output_dir=output_dir,
                cfg={"runtime": "llama_cpp", "model": "fake", "context_window_tokens": 4096, "system_prompt": "system"},
                client_factory=lambda cfg: ScriptedRuntimeClient(scripts),
            )

            self.assertTrue(artifact.completed)
            self.assertEqual(artifact.evaluation["score"], 1.0)
            self.assertTrue(artifact.evaluation["passed"])
            self.assertEqual(artifact.tool_counts["write_file"], 1)
            result_path = output_dir / case.case_id / f"run_01_{artifact.run_id}" / "result.json"
            payload = json.loads(result_path.read_text(encoding="utf-8"))
            self.assertIn("note.txt", {item["path"] for item in payload["filesystem_snapshot"]})
            self.assertEqual(payload["evaluation"]["score"], 1.0)
            judge_packet = json.loads((output_dir / case.case_id / f"run_01_{artifact.run_id}" / "judge_packet.json").read_text(encoding="utf-8"))
            self.assertEqual(judge_packet["case"]["case_id"], "write_case")
            self.assertEqual(judge_packet["deterministic_evaluation"]["score"], 1.0)
            self.assertIn("Judge whether the agent successfully completed", judge_packet["instructions_markdown"])

    async def test_run_benchmark_suite_writes_summary_and_manifest_scores(self) -> None:
        case = BenchmarkCase(
            case_id="read_case",
            description="Read a heading.",
            prompt="Read README.md and answer with the heading.",
            environment=EvalEnvironment(files=[EvalFile("README.md", "# Bench Heading\n")]),
            expectations=[
                EvalExpectation("tool_used", "read_file"),
                EvalExpectation("assistant_contains", value="Bench Heading"),
            ],
            mode="chat",
            allowed_tools=["read_file"],
        )

        def factory(_: dict) -> ScriptedRuntimeClient:
            return ScriptedRuntimeClient(
                [
                    [StreamChunk(text="", tool_calls=[ToolCall(name="read_file", arguments={"path": "README.md"})])],
                    [StreamChunk(text="Bench Heading")],
                ]
            )

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "suite"
            artifacts = await run_benchmark_suite(
                cases=[case],
                repeats=2,
                output_dir=output_dir,
                cfg={"runtime": "llama_cpp", "model": "fake", "context_window_tokens": 4096, "system_prompt": "system"},
                client_factory=factory,
            )

            self.assertEqual(len(artifacts), 2)
            summary_lines = (output_dir / "summary.jsonl").read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(summary_lines), 2)
            summary_record = json.loads(summary_lines[0])
            self.assertIn("score", summary_record)
            self.assertIn("passed", summary_record)
            manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["runs"], 2)
            self.assertEqual(manifest["average_score"], 1.0)
            judge_index = json.loads((output_dir / "judge_index.json").read_text(encoding="utf-8"))
            self.assertEqual(len(judge_index["runs"]), 2)

    def test_evaluate_case_supports_negative_and_content_checks(self) -> None:
        case = BenchmarkCase(
            case_id="eval_only",
            description="grading only",
            prompt="",
            environment=EvalEnvironment(),
            expectations=[
                EvalExpectation("assistant_contains", value="hello"),
                EvalExpectation("file_not_contains", "note.txt", "bad"),
            ],
        )
        artifact_payload = {
            "benchmark_version": 1,
            "case_id": "eval_only",
            "run_id": "abc",
            "repeat_index": 1,
            "started_at": 0.0,
            "ended_at": 1.0,
            "duration_seconds": 1.0,
            "model": "fake",
            "runtime": "fake",
            "prompt": "",
            "mode": "chat",
            "preferred_skills": [],
            "allowed_tools": [],
            "workspace": "/tmp/x",
            "environment_manifest": [],
            "filesystem_snapshot": [{"path": "note.txt", "sha256": "x", "bytes": 4, "content": "good"}],
            "turns": [],
            "final_messages": [],
            "final_harness_state": {},
            "tool_counts": {},
            "final_assistant_text": "hello world",
            "completed": True,
            "failure_reason": None,
            "evaluation": {},
        }
        from src.benchmark import BenchmarkRunArtifact

        evaluation = evaluate_case(case=case, artifact=BenchmarkRunArtifact(**artifact_payload))
        self.assertEqual(evaluation.score, 1.0)
        self.assertTrue(evaluation.passed)


if __name__ == "__main__":
    unittest.main()
