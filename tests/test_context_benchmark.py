from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from src.benchmark import main as benchmark_main
from src.context_benchmark import compare_context_runs, run_context_suite, run_jetson_4k_suites, summarize_context_run


class ContextBenchmarkArtifactTests(unittest.TestCase):
    def test_run_context_suite_writes_required_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = run_context_suite("candidate_starvation_case", output_root=Path(tmp))

            config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
            summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
            metrics = json.loads((run_dir / "compare_ready_metrics.json").read_text(encoding="utf-8"))
            turns = [
                json.loads(line)
                for line in (run_dir / "turns.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(config["benchmark_name"], "candidate_starvation_case")
            self.assertEqual(summary["benchmark_name"], "candidate_starvation_case")
            self.assertEqual(len(turns), 1)
            self.assertTrue((run_dir / "summary.md").exists())
            self.assertTrue((run_dir / "timeline.txt").exists())
            self.assertIn("candidate_metadata", turns[0])
            self.assertIn("state_summary_tokens", turns[0])
            self.assertIn("admitted_docs", turns[0])
            self.assertIn("skip_reason_counts", metrics)
            self.assertGreaterEqual(metrics["per_layer_budget_skips"], 1)

    def test_chat_history_driven_suite_records_loader_history_profile(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = run_context_suite("jetson_4k_baseline", output_root=Path(tmp))
            turns = [
                json.loads(line)
                for line in (run_dir / "turns.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]

            self.assertEqual(turns[0]["current_context_token_source"], "chat_history")
            self.assertIn("chat_history_profile", turns[0])
            self.assertGreater(turns[0]["chat_history_profile"]["message_count"], 1)
            self.assertGreater(turns[0]["chat_history_profile"]["persistent_context_tokens"], 0)
            self.assertIn("tool_call_tokens_in_history", turns[0]["chat_history_profile"])
            self.assertTrue(turns[0]["chat_history_profile"]["top_loader_messages"])

    def test_run_jetson_4k_suites_returns_expected_runs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runs = run_jetson_4k_suites(output_root=Path(tmp))
            self.assertEqual(len(runs), 3)
            self.assertEqual(
                sorted(path.name.split("_", 1)[1] for path in runs),
                [
                    "jetson_4k_baseline",
                    "jetson_4k_layer_compare",
                    "jetson_4k_ram_pressure",
                ],
            )

    def test_compare_context_runs_and_summary_are_terminal_readable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline = run_context_suite("jetson_4k_baseline", output_root=root)
            stressed = run_context_suite("jetson_4k_ram_pressure", output_root=root)
            markdown_path = root / "compare.md"

            comparison = compare_context_runs([baseline, stressed], markdown_path=markdown_path)
            summary_text = summarize_context_run(baseline)
            self.assertIn("CONTEXT BENCHMARK COMPARISON", comparison)
            self.assertIn("jetson_4k_ram_pressure", comparison)
            self.assertIn("skip_reason_counts", comparison)
            self.assertIn("jetson_4k_baseline", summary_text)
            self.assertTrue(markdown_path.exists())

    def test_benchmark_cli_context_commands_work(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = io.StringIO()
            with redirect_stdout(output):
                benchmark_main(["context-suite", "candidate_starvation_case", "--output-root", tmp])
            rendered = output.getvalue()
            run_dir = Path(rendered.splitlines()[0].strip())

            summary_out = io.StringIO()
            with redirect_stdout(summary_out):
                benchmark_main(["summary", str(run_dir)])

        self.assertIn("candidate_starvation_case", rendered)
        self.assertIn("candidate_starvation_case", summary_out.getvalue())


if __name__ == "__main__":
    unittest.main()
