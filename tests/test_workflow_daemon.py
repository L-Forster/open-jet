from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from src.workflows.daemon import stop_workflow_daemon, start_workflow_daemon
from src.workflows.state import WorkflowStatus


class WorkflowDaemonTests(unittest.TestCase):
    def test_start_workflow_daemon_bootstraps_openjet_cli_from_package_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch("src.workflows.daemon.load_workflow_pid", return_value=None), patch(
                "src.workflows.daemon.subprocess.Popen",
                return_value=SimpleNamespace(pid=4321),
            ) as popen:
                pid = start_workflow_daemon(root, "nightwatch", device_ids=["gpio0"], interval_seconds=30)

        self.assertEqual(pid, 4321)
        command = popen.call_args.args[0]
        self.assertEqual(command[0], popen.call_args.args[0][0])
        self.assertEqual(command[1], "-c")
        self.assertIn("sys.path.insert(0", command[2])
        self.assertIn("from src.cli import main", command[2])
        self.assertIn("workflow-runner", command)
        self.assertEqual(popen.call_args.kwargs["cwd"], str(root))

    def test_stop_workflow_daemon_force_stops_process_and_updates_status(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            status = WorkflowStatus(
                name="nightwatch",
                running=True,
                pid=4321,
                interval_seconds=30,
                bound_devices=("gpio0",),
                last_started_at="2026-03-24T06:00:00+00:00",
                last_finished_at=None,
                last_success=None,
                last_report_path=None,
            )
            with patch(
                "src.workflows.daemon.load_workflow_pid",
                return_value={"pid": 4321, "interval_seconds": 30, "device_ids": ["gpio0"]},
            ), patch("src.workflows.daemon.pid_is_running", return_value=True), patch(
                "src.workflows.daemon._wait_for_pid_exit",
                side_effect=[False, True],
            ), patch("src.workflows.daemon._signal_workflow_process") as signal_process, patch(
                "src.workflows.daemon.clear_workflow_pid"
            ) as clear_pid, patch(
                "src.workflows.daemon.load_workflow_status",
                return_value=status,
            ), patch(
                "src.workflows.daemon.save_workflow_status"
            ) as save_status:
                stopped = stop_workflow_daemon(root, "nightwatch")

        self.assertTrue(stopped)
        self.assertEqual(signal_process.call_count, 2)
        clear_pid.assert_called_once_with(root, "nightwatch")
        saved = save_status.call_args.args[1]
        self.assertFalse(saved.running)
        self.assertIsNone(saved.pid)
        self.assertEqual(saved.bound_devices, ("gpio0",))

    def test_stop_workflow_daemon_returns_false_when_process_wont_exit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch(
                "src.workflows.daemon.load_workflow_pid",
                return_value={"pid": 4321, "interval_seconds": 30, "device_ids": []},
            ), patch("src.workflows.daemon.pid_is_running", return_value=True), patch(
                "src.workflows.daemon._wait_for_pid_exit",
                side_effect=[False, False],
            ), patch("src.workflows.daemon._signal_workflow_process") as signal_process, patch(
                "src.workflows.daemon.clear_workflow_pid"
            ) as clear_pid, patch(
                "src.workflows.daemon.save_workflow_status"
            ) as save_status:
                stopped = stop_workflow_daemon(root, "nightwatch")

        self.assertFalse(stopped)
        self.assertEqual(signal_process.call_count, 2)
        clear_pid.assert_not_called()
        save_status.assert_not_called()
