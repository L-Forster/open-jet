from __future__ import annotations

import asyncio
import os
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

from src.agent import Agent
from src.app import OpenJetApp
from src.llama_server import (
    LlamaServerClient,
    _JETSON_VMM_CHUNK_MB,
    _JETSON_VMM_RESERVE_MB,
)
from src.runtime_protocol import StreamChunk


class FakeAgent:
    def conversation_message_count(self) -> int:
        return 3

    def context_budget(self):
        return None

    def estimated_context_tokens(self) -> int:
        return 256


class FakeReasoningClient:
    context_window_tokens = 4096

    def reasoning_status(self) -> str:
        return "on"


class FakeRuntimeClient:
    context_window_tokens = 4096

    async def chat_stream(self, messages, *, use_tools=True):
        yield StreamChunk(text="hello")
        yield StreamChunk(done=True)


class AgentTraceTests(unittest.IsolatedAsyncioTestCase):
    async def test_agent_trace_hook_records_stream_lifecycle(self) -> None:
        events: list[tuple[str, dict[str, object]]] = []
        agent = Agent(
            client=FakeRuntimeClient(),
            system_prompt="system",
            context_window_tokens=4096,
            trace_hook=lambda event, data: events.append((event, data)),
        )
        agent.add_user_message("hi")

        result_kinds = []
        async for event in agent.run_turn():
            result_kinds.append(event.kind.name)

        names = [name for name, _ in events]
        self.assertIn("run_turn_start", names)
        self.assertIn("stream_first_chunk", names)
        self.assertIn("run_turn_complete", names)
        self.assertIn("run_turn_done", names)
        self.assertIn("DONE", result_kinds)

    async def test_persistent_context_tokens_counts_assistant_tool_calls(self) -> None:
        agent = Agent(
            client=FakeRuntimeClient(),
            system_prompt="system",
            context_window_tokens=4096,
        )
        base = agent.persistent_context_tokens()
        agent.messages.append(
            {
                "role": "assistant",
                "content": "Inspecting files before patching.",
                "tool_calls": [
                    {
                        "id": "call_read",
                        "type": "function",
                        "function": {
                            "name": "read_file",
                            "arguments": json.dumps(
                                {
                                    "path": "src/harness.py",
                                    "detail": "read the layered budget path and candidate admission flow",
                                },
                                ensure_ascii=True,
                            ),
                        },
                    }
                ],
            }
        )

        with_tool_calls = agent.persistent_context_tokens()

        self.assertGreater(with_tool_calls, base + 10)


class AppStatusTests(unittest.TestCase):
    def test_format_command_status_label_compacts_and_truncates(self) -> None:
        label = OpenJetApp._format_command_status_label(
            "pytest\n\n   tests/test_app.py   -k   status_indicator",
            max_len=24,
        )

        self.assertEqual(label, "pytest tests/test_app...")

    def test_runtime_status_snapshot_reports_active_command(self) -> None:
        app = OpenJetApp()
        app.agent = FakeAgent()
        app._assistant_status_kind = "command"
        app._assistant_status_command = "pytest tests/test_app.py"

        with patch("src.app.read_memory_snapshot", return_value=None):
            snapshot = app.runtime_status_snapshot()

        self.assertTrue(snapshot["command_in_progress"])
        self.assertEqual(snapshot["active_command"], "pytest tests/test_app.py")
        self.assertFalse(snapshot["generating"])

    def test_runtime_status_snapshot_reports_reasoning_mode(self) -> None:
        app = OpenJetApp()
        app.agent = FakeAgent()
        app.client = FakeReasoningClient()

        with patch("src.app.read_memory_snapshot", return_value=None):
            snapshot = app.runtime_status_snapshot()

        self.assertEqual(snapshot["reasoning_mode"], "on")


class AppQuitTests(unittest.TestCase):
    def test_request_terminal_exit_exits_active_prompt_app(self) -> None:
        app = OpenJetApp()
        prompt_app = Mock()
        prompt_app.is_running = True

        app._request_terminal_exit(prompt_app)

        prompt_app.exit.assert_called_once()
        self.assertIsInstance(prompt_app.exit.call_args.kwargs["exception"], KeyboardInterrupt)

    def test_request_terminal_exit_schedules_action_quit_without_prompt_app(self) -> None:
        app = OpenJetApp()

        with patch("asyncio.create_task") as create_task:
            app._request_terminal_exit()

        create_task.assert_called_once()
        scheduled = create_task.call_args.args[0]
        self.assertTrue(asyncio.iscoroutine(scheduled))
        scheduled.close()


class LlamaServerStartupTests(unittest.TestCase):
    def test_largest_free_block_parser_uses_highest_available_order(self) -> None:
        buddyinfo = (
            "Node 0, zone      DMA      1      0      0      0      0      0      0      0      0      0      0\n"
            "Node 0, zone   Normal   1406    708    420    250    120     65     24      8      2      0     46\n"
        )

        lfb_mb = LlamaServerClient._largest_free_block_mb_from_text(buddyinfo, page_size_kb=4)

        self.assertEqual(lfb_mb, 4.0)

    def test_startup_profile_uses_fit_off_and_no_warmup_when_fragmented(self) -> None:
        profile = LlamaServerClient._startup_profile_for_lfb(4.0)

        self.assertEqual(profile, (128, 32, True, True))


class LlamaServerLaunchEnvTests(unittest.IsolatedAsyncioTestCase):
    async def test_start_sets_jetson_vmm_env_for_cuda(self) -> None:
        client = LlamaServerClient(
            model="model.gguf",
            device="cuda",
            gpu_layers=99,
            context_window_tokens=4096,
        )
        start_once = AsyncMock()
        stop_server = AsyncMock()
        cleanup = AsyncMock()
        prepare = AsyncMock(return_value=4.0)

        with patch.object(client, "_stop_server", stop_server), patch.object(
            client, "_cleanup_stale_inference_processes", cleanup
        ), patch.object(client, "_prepare_memory_for_launch", prepare), patch.object(
            client, "_start_once", start_once
        ), patch.object(
            client, "_ensure_jetson_clocks_sudoers", return_value=None
        ), patch.object(
            client, "_maximize_gpu_clocks", return_value=None
        ), patch.object(
            client, "_is_jetson_platform", return_value=True
        ), patch(
            "src.llama_server._find_llama_server", return_value="/usr/bin/llama-server"
        ):
            await client.start()

        kwargs = start_once.await_args.kwargs
        self.assertEqual(kwargs["env"]["GGML_CUDA_VMM_CHUNK_MB"], _JETSON_VMM_CHUNK_MB)
        self.assertEqual(kwargs["env"]["GGML_CUDA_VMM_RESERVE_MB"], _JETSON_VMM_RESERVE_MB)
        self.assertEqual(kwargs["ngl"], 99)
        self.assertEqual(kwargs["ctx"], 4096)
        self.assertEqual(prepare.await_count, 1)
        self.assertEqual(cleanup.await_count, 1)


class DebugPromptLoggingTests(unittest.TestCase):
    def test_prepare_turn_context_saves_full_runtime_messages_in_debug_mode(self) -> None:
        previous_cwd = Path.cwd()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_harness_docs(root)
            os.chdir(root)
            try:
                app = OpenJetApp()
                app.client = FakeRuntimeClient()
                app.agent = Agent(client=app.client, system_prompt="base system prompt", context_window_tokens=4096)
                app.agent.add_user_message("debug this failure")
                app.harness_state.mode = "debug"
                app.harness_state.goal = "Investigate a failing debug turn"
                app._active_turn_id = "turn-debug-1"

                with patch("src.app.read_memory_snapshot", return_value=None):
                    app._prepare_turn_context()

                saved = json.loads((root / ".openjet" / "state" / "debug_prompts" / "turn-debug-1.messages.json").read_text(encoding="utf-8"))
                context_saved = json.loads((root / ".openjet" / "state" / "debug_prompts" / "turn-debug-1.context.json").read_text(encoding="utf-8"))
                latest = json.loads((root / ".openjet" / "state" / "debug_prompts" / "latest.messages.json").read_text(encoding="utf-8"))
            finally:
                os.chdir(previous_cwd)

        self.assertEqual(saved, latest)
        self.assertEqual(saved[0]["role"], "system")
        self.assertIn("base system prompt", saved[0]["content"])
        self.assertIn("OPEN-JET HARNESS STATE", saved[0]["content"])
        self.assertIn("MODE: debug", saved[0]["content"])
        self.assertIn("Investigate a failing debug turn", saved[0]["content"])
        self.assertEqual(saved[1]["role"], "user")
        self.assertEqual(saved[1]["content"], "debug this failure")
        self.assertIn("layer_tokens", context_saved)
        self.assertIn("layer_docs", context_saved)
        self.assertIn("budget", context_saved)
        self.assertIn("project-context", context_saved["docs_loaded"])

    def test_prepare_turn_context_updates_debug_payload_for_multiple_turns(self) -> None:
        previous_cwd = Path.cwd()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_harness_docs(root)
            os.chdir(root)
            try:
                app = OpenJetApp()
                app.client = FakeRuntimeClient()
                app.agent = Agent(client=app.client, system_prompt="base system prompt", context_window_tokens=4096)
                app.harness_state.mode = "debug"
                app.agent.add_user_message("first request")
                app.agent.messages.append({"role": "assistant", "content": "first reply"})
                app.agent.add_user_message("second request")

                with patch("src.app.read_memory_snapshot", return_value=None):
                    app.harness_state.goal = "First debug turn"
                    app._active_turn_id = "turn-debug-1"
                    app._prepare_turn_context()

                    app.harness_state.goal = "Second debug turn"
                    app._active_turn_id = "turn-debug-2"
                    app._prepare_turn_context()

                first = json.loads((root / ".openjet" / "state" / "debug_prompts" / "turn-debug-1.messages.json").read_text(encoding="utf-8"))
                second = json.loads((root / ".openjet" / "state" / "debug_prompts" / "turn-debug-2.messages.json").read_text(encoding="utf-8"))
                latest = json.loads((root / ".openjet" / "state" / "debug_prompts" / "latest.messages.json").read_text(encoding="utf-8"))
                latest_context = json.loads((root / ".openjet" / "state" / "debug_prompts" / "latest.context.json").read_text(encoding="utf-8"))
            finally:
                os.chdir(previous_cwd)

        self.assertEqual(first[-1]["content"], "second request")
        self.assertEqual(second[-1]["content"], "second request")
        self.assertEqual(second[1]["content"], "first request")
        self.assertEqual(second[2]["content"], "first reply")
        self.assertEqual(second[3]["content"], "second request")
        self.assertIn("First debug turn", first[0]["content"])
        self.assertIn("Second debug turn", second[0]["content"])
        self.assertEqual(second, latest)
        self.assertNotEqual(first, second)
        self.assertIn("layer_tokens", latest_context)
        self.assertIn("layer1_budget", latest_context["budget"])

    def _write_harness_docs(self, root: Path) -> None:
        (root / "AGENTS.md").write_text(
            "## What This Project Is\n"
            "- local agent for testing\n\n"
            "## Core Architecture\n"
            "- `src/demo.py`: example module\n",
            encoding="utf-8",
        )
        (root / ".openjet" / "agents").mkdir(parents=True)
        (root / ".openjet" / "projects").mkdir(parents=True)
        (root / ".openjet" / "agents" / "base.md").write_text("base guidance", encoding="utf-8")
        (root / ".openjet" / "agents" / "debugger.md").write_text("debug guidance", encoding="utf-8")
        (root / ".openjet" / "projects" / "default.md").write_text("project guidance", encoding="utf-8")
