from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.context_index import build_repo_context_index, lookup_file_summary

from tests.context_helpers import write_repo_fixture


class RepoContextIndexTests(unittest.TestCase):
    def test_project_summary_extraction_uses_expected_sections(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_repo_fixture(
                root,
                architecture_lines=["- `src/harness.py`: owns layered context."],
            )

            index = build_repo_context_index(root)

        self.assertIn("PROJECT CONTEXT SUMMARY", index.project_summary)
        self.assertIn("What This Project Is:", index.project_summary)
        self.assertIn("Engineering Rules:", index.project_summary)
        self.assertIn("Hardware And Performance Assumptions:", index.project_summary)

    def test_file_summary_extraction_uses_architecture_lines(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_repo_fixture(
                root,
                architecture_lines=[
                    "- `src/harness.py`: owns layered context budgeting and document admission.",
                    "- `src/context_index.py`: resolves project and file summaries.",
                ],
            )

            index = build_repo_context_index(root)

        self.assertIn("src/harness.py", index.files)
        self.assertEqual(
            index.files["src/context_index.py"].purpose,
            "resolves project and file summaries.",
        )

    def test_lookup_returns_indexed_file_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_repo_fixture(
                root,
                architecture_lines=["- `src/harness.py`: owns layered context budgeting."],
            )

            index = build_repo_context_index(root)
            summary = lookup_file_summary(index, "src/harness.py")

        self.assertIsNotNone(summary)
        self.assertEqual(summary.path, "src/harness.py")
        self.assertIn("budgeting", summary.purpose)

    def test_lookup_falls_back_for_test_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            index = build_repo_context_index(Path(tmp))
            summary = lookup_file_summary(index, "tests/test_context_harness.py")

        self.assertIsNotNone(summary)
        self.assertEqual(summary.path, "tests/test_context_harness.py")
        self.assertEqual(summary.related_tests, ("tests/test_context_harness.py",))


if __name__ == "__main__":
    unittest.main()
