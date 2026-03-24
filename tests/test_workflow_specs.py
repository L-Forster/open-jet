from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from src.workflows.specs import discover_workflow_issues, discover_workflow_specs, load_workflow_spec, parse_workflow_markdown
from src.workflows.state import load_workflow_status, save_workflow_pid


class WorkflowSpecTests(unittest.TestCase):
    def test_parse_workflow_markdown_defaults_name_and_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "workflows" / "nightwatch.md"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("Check the devices and summarize findings.\n", encoding="utf-8")

            spec = parse_workflow_markdown(path)

        self.assertEqual(spec.name, "nightwatch")
        self.assertEqual(spec.mode, "chat")
        self.assertEqual(spec.devices, ())
        self.assertEqual(spec.body, "Check the devices and summarize findings.")

    def test_parse_workflow_markdown_rejects_unterminated_frontmatter(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "workflows" / "bad.md"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("---\nname: bad\nmode: chat\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "frontmatter"):
                parse_workflow_markdown(path)

    def test_discover_workflow_specs_prefers_local_override(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "workflows").mkdir(parents=True, exist_ok=True)
            (root / ".openjet" / "workflows").mkdir(parents=True, exist_ok=True)
            (root / "workflows" / "nightwatch.md").write_text(
                "---\nname: NightWatch\nmode: chat\n---\nRepo body\n",
                encoding="utf-8",
            )
            (root / ".openjet" / "workflows" / "nightwatch.md").write_text(
                "---\nname: NightWatch\nmode: review\n---\nLocal override body\n",
                encoding="utf-8",
            )

            specs = discover_workflow_specs(root)
            spec = load_workflow_spec(root, "nightwatch")

        self.assertEqual(len(specs), 1)
        self.assertEqual(spec.mode, "review")
        self.assertEqual(spec.body, "Local override body")
        self.assertEqual(spec.source, "local_override")

    def test_discover_workflow_specs_skips_invalid_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "workflows").mkdir(parents=True, exist_ok=True)
            (root / "workflows" / "valid.md").write_text(
                "---\nmode: chat\n---\nValid workflow\n",
                encoding="utf-8",
            )
            (root / "workflows" / "broken.md").write_text(
                "---\nmode: chat\n",
                encoding="utf-8",
            )

            specs = discover_workflow_specs(root)
            issues = discover_workflow_issues(root)
            spec = load_workflow_spec(root, "valid")

        self.assertEqual([item.name for item in specs], ["valid"])
        self.assertEqual(spec.name, "valid")
        self.assertEqual(len(issues), 1)
        self.assertIn("broken.md", str(issues[0].path))


class WorkflowStateTests(unittest.TestCase):
    def test_load_workflow_status_uses_pid_metadata_before_first_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            save_workflow_pid(
                root,
                "nightwatch",
                pid=os.getpid(),
                interval_seconds=30,
                device_ids=["gpio0"],
                updated_at="2026-03-24T12:00:00+00:00",
            )

            status = load_workflow_status(root, "nightwatch")

        self.assertIsNotNone(status)
        self.assertEqual(status.name, "nightwatch")
        self.assertTrue(status.running)
        self.assertEqual(status.bound_devices, ("gpio0",))
        self.assertEqual(status.interval_seconds, 30)
