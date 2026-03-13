from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.app import _classify_shell_command
from src.session_logging import SessionLogger


class SessionLoggerTests(unittest.TestCase):
    def test_log_event_includes_schema_version_and_event_index(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            logger = SessionLogger(base_dir=Path(tmp), label="trace-test")
            logger.log_event("tool_request", tool="shell", turn_id="turn-1")
            logger.log_event("tool_result", tool="shell", turn_id="turn-1", ok=True)

            lines = logger.events_path.read_text(encoding="utf-8").splitlines()
            first = json.loads(lines[0])
            second = json.loads(lines[1])

            self.assertEqual(first["schema_version"], 2)
            self.assertEqual(first["event_index"], 1)
            self.assertEqual(second["event_index"], 2)
            self.assertEqual(second["data"]["ok"], True)


class ShellCommandClassificationTests(unittest.TestCase):
    def test_classifies_tool_like_shell_as_false_positive(self) -> None:
        classified = _classify_shell_command("grep -n TODO src/app.py")

        self.assertEqual(classified["primary_command"], "grep")
        self.assertEqual(classified["false_positive_proposal"], True)
        self.assertEqual(classified["hallucinated_command"], False)

    def test_classifies_unknown_command_as_hallucinated(self) -> None:
        classified = _classify_shell_command("definitely-not-a-real-command --flag")

        self.assertEqual(classified["primary_command"], "definitely-not-a-real-command")
        self.assertEqual(classified["hallucinated_command"], True)


if __name__ == "__main__":
    unittest.main()
