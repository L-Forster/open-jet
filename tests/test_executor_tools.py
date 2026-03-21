from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path

from src.executor import glob_files, grep_files, list_directory


class ExecutorToolFilteringTests(unittest.TestCase):
    def test_grep_ignores_openjet_state_and_session_files_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "src").mkdir()
            (root / "src" / "model.py").write_text("input feature vector elements\n", encoding="utf-8")
            (root / ".openjet" / "state").mkdir(parents=True)
            (root / ".openjet" / "state" / "session.json").write_text(
                "input feature vector elements\n",
                encoding="utf-8",
            )
            (root / "session_state.json").write_text("input feature vector elements\n", encoding="utf-8")
            (root / "session_logs").mkdir()
            (root / "session_logs" / "turn-1.json").write_text("input feature vector elements\n", encoding="utf-8")

            result = asyncio.run(grep_files("input feature vector", path=str(root), ignore_case=True))

        self.assertIn("src/model.py", result)
        self.assertNotIn(".openjet", result)
        self.assertNotIn("session_state.json", result)
        self.assertNotIn("session_logs", result)

    def test_glob_ignores_internal_state_paths_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "src").mkdir()
            (root / "src" / "model.py").write_text("ok\n", encoding="utf-8")
            (root / ".openjet" / "state").mkdir(parents=True)
            (root / ".openjet" / "state" / "session.json").write_text("ok\n", encoding="utf-8")
            (root / "session_logs").mkdir()
            (root / "session_logs" / "turn-1.json").write_text("ok\n", encoding="utf-8")

            result = asyncio.run(glob_files("**/*.json", path=str(root)))

        self.assertNotIn(".openjet", result)
        self.assertNotIn("session_logs", result)

    def test_list_directory_hides_internal_state_entries_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "src").mkdir()
            (root / ".openjet").mkdir()
            (root / "session_logs").mkdir()
            (root / "session_state.json").write_text("{}", encoding="utf-8")

            result = asyncio.run(list_directory(str(root)))

        self.assertIn("src/", result)
        self.assertNotIn(".openjet/", result)
        self.assertNotIn("session_logs/", result)
        self.assertNotIn("session_state.json", result)


if __name__ == "__main__":
    unittest.main()
