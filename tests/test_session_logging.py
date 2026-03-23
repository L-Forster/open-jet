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

    def test_runtime_context_includes_explicit_device_and_model_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            logger = SessionLogger(base_dir=Path(tmp), label="trace-test")

            attrs = logger._sanitize_runtime_context(
                {
                    "runtime": "llama_cpp",
                    "backend": "ollama",
                    "model": "/models/Qwen3.5-4B-Q4_K_M.gguf",
                    "model_id": "qwen3.5-4b",
                    "model_variant": "q4_k_m",
                    "hardware_class": "jetson_orin_nano_8gb",
                    "hardware_family": "jetson",
                    "accelerator": "cuda",
                    "device_profile": "cuda",
                    "os_type": "linux",
                    "context_window_tokens": 4096,
                    "gpu_layers": 99,
                    "system_memory_total_mb": 7620.5,
                    "host_arch": "aarch64",
                    "use_case_tag": "robotics",
                }
            )

            self.assertEqual(attrs["openjet.backend"], "ollama")
            self.assertEqual(attrs["openjet.model.name"], "Qwen3.5-4B-Q4_K_M.gguf")
            self.assertEqual(attrs["openjet.model.id"], "qwen3_5_4b")
            self.assertEqual(attrs["openjet.model.variant"], "q4_k_m")
            self.assertEqual(attrs["openjet.hardware.class"], "jetson_orin_nano_8gb")
            self.assertEqual(attrs["openjet.hardware.family"], "jetson")
            self.assertEqual(attrs["openjet.hardware.accelerator"], "cuda")
            self.assertEqual(attrs["openjet.os.type"], "linux")
            self.assertEqual(attrs["openjet.system.memory.total_mb"], 7620.5)
            self.assertEqual(attrs["openjet.use_case_tag"], "robotics")

    def test_none_mode_is_normalized_for_turn_and_message_logging(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            logger = SessionLogger(base_dir=Path(tmp), label="trace-test")

            asyncio.run(logger.start())
            logger.record_user_message(
                turn_id="turn-1",
                text="hello",
                mentioned_files=[],
                attached_images=[],
                mode=None,
            )
            logger.start_turn(
                turn_id="turn-1",
                prompt="hello",
                mode=None,
                resumed_session=False,
                active_step=None,
                files_in_play=[],
                runtime_context={},
            )
            asyncio.run(logger.stop())

            self.assertTrue(logger.manifest_path.exists())


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
