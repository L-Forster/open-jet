from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path

from src.app import _classify_shell_command
from src.session_logging import BroadcastConfig, SessionLogger


class SessionLoggerTests(unittest.TestCase):
    def test_start_creates_session_manifest_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            logger = SessionLogger(base_dir=Path(tmp), label="trace-test")

            asyncio.run(logger.start())
            logger.log_event("runtime_recovery_succeeded")
            asyncio.run(logger.stop())

            manifest = json.loads(logger.manifest_path.read_text(encoding="utf-8"))

            self.assertTrue(logger.manifest_path.exists())
            self.assertEqual(manifest["session_id"], logger.session_id)
            self.assertEqual(Path(manifest["session_dir"]), logger.session_dir)
            self.assertEqual(manifest["telemetry"]["enabled"], False)
            self.assertFalse((logger.session_dir / "logs").exists())
            self.assertFalse((logger.session_dir / "traces").exists())
            self.assertFalse((logger.session_dir / "metrics").exists())

    def test_manifest_records_collector_endpoint_when_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            logger = SessionLogger(
                base_dir=Path(tmp),
                label="trace-test",
                broadcast=BroadcastConfig(enabled=True, endpoint="http://127.0.0.1:4318"),
            )

            asyncio.run(logger.start())
            asyncio.run(logger.stop())

            manifest = json.loads(logger.manifest_path.read_text(encoding="utf-8"))

            self.assertEqual(manifest["telemetry"]["collector_endpoint"], "http://127.0.0.1:4318")
            self.assertEqual(manifest["telemetry"]["enabled"], True)

    def test_user_message_telemetry_does_not_store_prompt_text_in_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            logger = SessionLogger(base_dir=Path(tmp), label="trace-test")
            prompt = "super secret prompt with /home/louis/private.txt"

            asyncio.run(logger.start())
            logger.record_user_message(
                turn_id="turn-1",
                text=prompt,
                mentioned_files=["/home/louis/private.txt"],
                attached_images=[],
                mode="chat",
            )
            asyncio.run(logger.stop())

            manifest = json.loads(logger.manifest_path.read_text(encoding="utf-8"))
            encoded = json.dumps(manifest)

            self.assertNotIn(prompt, encoded)
            self.assertNotIn("/home/louis/private.txt", encoded)
            self.assertEqual(manifest["runtime_context"], {})

    def test_sanitizes_generic_event_without_raw_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            logger = SessionLogger(base_dir=Path(tmp), label="trace-test")

            payload = logger._sanitize_generic_event(
                "tool_result",
                {
                    "turn_id": "turn-1",
                    "text": "sensitive prompt",
                    "command": "cat /home/louis/private.txt",
                    "cwd": "/home/louis/repo",
                },
            )

            encoded = json.dumps(payload)

            self.assertNotIn("sensitive prompt", encoded)
            self.assertNotIn("/home/louis/private.txt", encoded)
            self.assertEqual(payload["openjet.text_char_count"], len("sensitive prompt"))
            self.assertEqual(payload["openjet.command_char_count"], len("cat /home/louis/private.txt"))
            self.assertEqual(payload["openjet.cwd_name"], "repo")


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
