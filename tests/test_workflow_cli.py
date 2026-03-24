from __future__ import annotations

import io
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from src.cli import main as cli_main
from src.workflows.specs import WorkflowSpec


class WorkflowCliTests(unittest.TestCase):
    def test_workflow_list_cli_prints_discovered_specs(self) -> None:
        stdout = io.StringIO()
        root = Path("/tmp/open-jet")
        spec = WorkflowSpec(
            name="nightwatch",
            path=root / "workflows" / "nightwatch.md",
            body="Inspect devices.",
            mode="review",
            devices=("gpio0",),
        )

        with patch("src.cli._workflow_root", return_value=root), patch(
            "src.cli.discover_workflow_specs",
            return_value=[spec],
        ), patch("src.cli.list_workflow_statuses", return_value=[]), patch("sys.stdout", stdout):
            cli_main(["workflow", "list"])

        rendered = stdout.getvalue()
        self.assertIn("Discovered workflows:", rendered)
        self.assertIn("nightwatch", rendered)
        self.assertIn("path=/tmp/open-jet/workflows/nightwatch.md", rendered)

    def test_workflow_list_cli_reports_invalid_workflow_files(self) -> None:
        stdout = io.StringIO()
        root = Path("/tmp/open-jet")
        with patch("src.cli._workflow_root", return_value=root), patch(
            "src.cli.discover_workflow_specs",
            return_value=[],
        ), patch(
            "src.cli.discover_workflow_issues",
            return_value=[SimpleNamespace(path=root / "workflows" / "broken.md", error="bad frontmatter")],
        ), patch("sys.stdout", stdout):
            cli_main(["workflow", "list"])

        rendered = stdout.getvalue()
        self.assertIn("No valid workflows discovered.", rendered)
        self.assertIn("broken.md", rendered)

    def test_workflow_assign_cli_persists_device_ids(self) -> None:
        stdout = io.StringIO()
        root = Path("/tmp/open-jet")
        spec = WorkflowSpec(
            name="nightwatch",
            path=root / "workflows" / "nightwatch.md",
            body="Inspect devices.",
        )
        with patch("src.cli._workflow_root", return_value=root), patch(
            "src.cli.load_workflow_spec",
            return_value=spec,
        ), patch("src.cli.validate_workflow_device_ids", return_value=()), patch(
            "src.cli.save_workflow_assignment",
            return_value=root / ".openjet" / "state" / "workflows" / "nightwatch" / "assignment.json",
        ) as save_assignment, patch("sys.stdout", stdout):
            cli_main(["workflow", "assign", "nightwatch", "camera0", "gpio0"])

        save_assignment.assert_called_once()
        rendered = stdout.getvalue()
        self.assertIn("Saved workflow device bindings for nightwatch.", rendered)
        self.assertIn("assignment.json", rendered)

    def test_workflow_run_cli_prints_report(self) -> None:
        stdout = io.StringIO()
        root = Path("/tmp/open-jet")
        result = SimpleNamespace(
            name="nightwatch",
            success=True,
            bound_devices=("gpio0",),
            error=None,
        )
        with patch("src.cli._workflow_root", return_value=root), patch(
            "src.cli._run_workflow_once",
            new=Mock(return_value="workflow-coro"),
        ), patch(
            "src.cli.asyncio.run",
            return_value=(result, Path("/tmp/report.md")),
        ), patch("sys.stdout", stdout):
            cli_main(["workflow", "run", "nightwatch"])

        rendered = stdout.getvalue()
        self.assertIn("Workflow nightwatch completed with status=success.", rendered)
        self.assertIn("Report: /tmp/report.md", rendered)
        self.assertIn("gpio0", rendered)

    def test_workflow_start_cli_starts_background_runner(self) -> None:
        stdout = io.StringIO()
        root = Path("/tmp/open-jet")
        spec = WorkflowSpec(
            name="nightwatch",
            path=root / "workflows" / "nightwatch.md",
            body="Inspect devices.",
            interval_seconds=30,
        )
        with patch("src.cli._workflow_root", return_value=root), patch(
            "src.cli.load_workflow_spec",
            return_value=spec,
        ), patch("src.cli.start_workflow_daemon", return_value=4321) as start_runner, patch(
            "sys.stdout",
            stdout,
        ):
            cli_main(["workflow", "start", "nightwatch"])

        start_runner.assert_called_once_with(root, "nightwatch", device_ids=[], interval_seconds=30)
        rendered = stdout.getvalue()
        self.assertIn("Started workflow nightwatch with pid=4321.", rendered)
        self.assertIn("runner.log", rendered)

    def test_hidden_workflow_runner_command_invokes_daemon(self) -> None:
        root = Path("/tmp/open-jet")
        with patch("src.cli._workflow_root", return_value=root), patch(
            "src.cli.run_workflow_daemon",
            new=Mock(return_value="daemon-coro"),
        ) as daemon_runner, patch("src.cli.asyncio.run") as run_async:
            cli_main(["workflow-runner", "nightwatch", "--interval", "60", "--device", "gpio0"])

        run_async.assert_called_once()
        daemon_runner.assert_called_once()
