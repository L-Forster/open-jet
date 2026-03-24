from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from src.peripherals import PeripheralDevice, PeripheralKind, PeripheralTransport
from src.runtime_protocol import ToolCall
from src.sdk import SDKResponse, ToolResult
from src.workflows.runner import run_workflow
from src.workflows.specs import load_workflow_spec


class _FakeAgent:
    def __init__(self, *, context_window_tokens: int = 8192) -> None:
        self.context_window_tokens = context_window_tokens

    def estimated_context_tokens(self) -> int:
        return 256


class _FakeSession:
    def __init__(self, response: SDKResponse, *, context_window_tokens: int = 8192) -> None:
        self.agent = _FakeAgent(context_window_tokens=context_window_tokens)
        self.response = response
        self.turn_context: list[dict[str, str]] = []
        self.prompt = ""
        self.closed = False

    def add_turn_context(self, messages: list[dict[str, str]]) -> None:
        self.turn_context = messages

    async def run(self, prompt: str, *, image_paths: list[str] | None = None) -> SDKResponse:
        self.prompt = prompt
        return self.response

    async def close(self) -> None:
        self.closed = True


class WorkflowRunnerTests(unittest.IsolatedAsyncioTestCase):
    async def test_run_workflow_builds_context_and_collects_payloads(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "workflows").mkdir(parents=True, exist_ok=True)
            (root / "README.md").write_text("repo notes\n", encoding="utf-8")
            (root / "workflows" / "nightwatch.md").write_text(
                "---\n"
                "mode: review\n"
                "devices:\n"
                "  - gpio0\n"
                "files:\n"
                "  - README.md\n"
                "---\n"
                "Inspect the GPIO sensor and summarize the current state.\n",
                encoding="utf-8",
            )
            spec = load_workflow_spec(root, "nightwatch")
            device = PeripheralDevice(
                id="sensor:/dev/gpiochip0",
                kind=PeripheralKind.SENSOR,
                transport=PeripheralTransport.GPIO,
                label="GPIO chip /dev/gpiochip0",
                path="/dev/gpiochip0",
            )
            fake_session = _FakeSession(
                SDKResponse(
                    text="GPIO looks stable.",
                    tool_results=[
                        ToolResult(
                            tool_call=ToolCall(name="gpio_read", arguments={"source": "gpio0"}),
                            output="GPIO snapshot",
                            meta={"ok": True, "payload_ref": str(root / "gpio-buffer.txt")},
                        )
                    ],
                )
            )

            with patch("src.device_sources.discover_peripherals", return_value=[device]), patch(
                "src.workflows.runner.OpenJetSession.create",
                AsyncMock(return_value=fake_session),
            ) as create_session:
                result = await run_workflow(root, spec, cfg={})

        self.assertTrue(result.success)
        self.assertEqual(result.bound_devices, ("gpio0",))
        self.assertIn(str(root / "gpio-buffer.txt"), result.payload_paths)
        self.assertTrue(fake_session.closed)
        allowed_tools = create_session.await_args.kwargs["allowed_tools"]
        self.assertIn("gpio_read", allowed_tools)
        self.assertNotIn("edit_file", allowed_tools)
        self.assertNotIn("write_file", allowed_tools)
        joined_context = "\n\n".join(message["content"] for message in fake_session.turn_context)
        self.assertIn("WORKFLOW DOCUMENT: nightwatch", joined_context)
        self.assertIn("IO device registry located in", joined_context)
        self.assertIn("Workflow-configured file context:", joined_context)

    async def test_run_workflow_allows_shell_only_when_requested(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "workflows").mkdir(parents=True, exist_ok=True)
            (root / "workflows" / "shellcheck.md").write_text(
                "---\nmode: debug\nallow_shell: true\n---\nRun a read-only hardware check.\n",
                encoding="utf-8",
            )
            spec = load_workflow_spec(root, "shellcheck")
            fake_session = _FakeSession(SDKResponse(text="ok"))

            with patch(
                "src.workflows.runner.OpenJetSession.create",
                AsyncMock(return_value=fake_session),
            ) as create_session:
                result = await run_workflow(root, spec, cfg={})

        self.assertTrue(result.success)
        self.assertIn("shell", create_session.await_args.kwargs["allowed_tools"])

    async def test_run_workflow_fails_for_disabled_bound_device(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "workflows").mkdir(parents=True, exist_ok=True)
            (root / "workflows" / "camera-check.md").write_text(
                "---\ndevices:\n  - camera0\n---\nCheck the camera.\n",
                encoding="utf-8",
            )
            spec = load_workflow_spec(root, "camera-check")
            device = PeripheralDevice(
                id="camera:/dev/video0",
                kind=PeripheralKind.CAMERA,
                transport=PeripheralTransport.V4L2,
                label="Front Camera",
                path="/dev/video0",
            )

            with patch("src.device_sources.discover_peripherals", return_value=[device]), patch(
                "src.workflows.runner.OpenJetSession.create",
                AsyncMock(side_effect=AssertionError("session should not be created")),
            ):
                result = await run_workflow(
                    root,
                    spec,
                    cfg={"disabled_device_ids": [device.id]},
                )

        self.assertFalse(result.success)
        self.assertIn("disabled", result.error or "")

    async def test_run_workflow_fails_when_file_preload_budget_is_insufficient(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "workflows").mkdir(parents=True, exist_ok=True)
            for index in range(4):
                (root / f"file{index}.md").write_text(f"file {index}\n", encoding="utf-8")
            (root / "workflows" / "budget-check.md").write_text(
                "---\n"
                "files:\n"
                "  - file0.md\n"
                "  - file1.md\n"
                "  - file2.md\n"
                "  - file3.md\n"
                "---\n"
                "Load every file.\n",
                encoding="utf-8",
            )
            spec = load_workflow_spec(root, "budget-check")
            fake_session = _FakeSession(SDKResponse(text="ok"), context_window_tokens=1024)

            with patch(
                "src.workflows.runner.OpenJetSession.create",
                AsyncMock(return_value=fake_session),
            ):
                result = await run_workflow(root, spec, cfg={})

        self.assertFalse(result.success)
        self.assertIn("preload budget", result.error or "")
