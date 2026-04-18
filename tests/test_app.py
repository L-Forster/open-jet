from __future__ import annotations

import asyncio
import os
import json
import io
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from rich.console import Console
from rich.syntax import Syntax
from rich.text import Text

from src.agent import ActionKind, Agent
from src.app import (
    OpenJetApp,
    _format_cli_status,
    _format_model_profiles_summary,
    _format_slash_commands_summary,
    main,
)
from src.cli import main as cli_main
from src.hardware import HardwareInfo
from src.llama_server import (
    LlamaServerClient,
    _JETSON_VMM_CHUNK_MB,
    _JETSON_VMM_RESERVE_MB,
)
from src.device_sources import DeviceSource
from src.peripherals import Observation, ObservationModality, PeripheralDevice, PeripheralKind, PeripheralTransport
from src.provisioning import ensure_direct_model, recommend_direct_model
from src.runtime_protocol import StreamChunk, ToolCall
from src.sdk import ToolResult
from src.session_state import ChatArchiveStore, SavedChatEntry, SessionStateStore
from src.setup import _prompt_choice, _runtime_prompt_options, build_recommended_payload, run_setup_wizard


class FakeAgent:
    def conversation_message_count(self) -> int:
        return 3

    def context_budget(self):
        return None

    def estimated_context_tokens(self) -> int:
        return 256

    def runtime_overhead_tokens(self, *, empty_retry_count: int = 0, force_post_tool_continuation: bool = False) -> int:
        return 0

    def persistent_context_tokens(self) -> int:
        return 128

    def set_turn_context(self, messages) -> None:
        self.turn_context_messages = list(messages)

    def _messages_for_runtime(self):
        return []


class FakeBudgetAgent(FakeAgent):
    def __init__(self, *, estimated: int = 256, persistent: int = 128, overhead: int = 0, prompt_tokens: int = 1024) -> None:
        self._estimated = estimated
        self._persistent = persistent
        self._overhead = overhead
        self._prompt_tokens = prompt_tokens
        self.turn_context_messages = []

    def context_budget(self):
        return SimpleNamespace(prompt_tokens=self._prompt_tokens)

    def estimated_context_tokens(self) -> int:
        return self._estimated

    def runtime_overhead_tokens(self, *, empty_retry_count: int = 0, force_post_tool_continuation: bool = False) -> int:
        return self._overhead

    def persistent_context_tokens(self) -> int:
        return self._persistent


class FakeReasoningClient:
    context_window_tokens = 4096

    def reasoning_status(self) -> str:
        return "on"


class FakeRuntimeClient:
    context_window_tokens = 4096

    async def chat_stream(self, messages, *, use_tools=True):
        yield StreamChunk(text="hello")
        yield StreamChunk(done=True)


class FakeInitClient:
    model = "fake.gguf"
    context_window_tokens = 4096
    gpu_layers = 99

    def __init__(self) -> None:
        self.start = AsyncMock()
        self.close = AsyncMock()
        self.reset_kv_cache = AsyncMock()

    async def chat_stream(self, messages, *, use_tools=True):
        if False:
            yield messages, use_tools

    async def save_kv_cache(self, path):
        return False

    async def restore_kv_cache(self, path):
        return False


class StopSetupWizard(Exception):
    pass


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

    async def test_context_pressure_accounts_for_runtime_overhead(self) -> None:
        agent = Agent(
            client=FakeRuntimeClient(),
            system_prompt="system",
            context_window_tokens=1000,
            context_reserved_tokens=200,
        )
        agent.add_user_message("x " * 700)

        with patch("src.agent.tool_schema_token_estimate", return_value=120):
            reason = agent.resource_pressure_reason()

        self.assertIsNotNone(reason)
        self.assertIn("exceeds prompt budget", reason or "")


class AppStatusTests(unittest.TestCase):
    def test_prompt_message_includes_generating_status_above_prompt(self) -> None:
        app = OpenJetApp()
        app._start_thinking()

        prompt = app._prompt_message()

        self.assertIn("Generating...", prompt.value)
        self.assertIn("Tip: Join Discord for community support", prompt.value)
        self.assertIn("prompt-tip", prompt.value)
        self.assertIn("prompt-splash-block-", prompt.value)
        self.assertIn("open-jet", prompt.value)

    def test_generating_tip_advances_per_message_not_time(self) -> None:
        app = OpenJetApp()

        first_token = app._start_thinking()
        first_tip = app._current_generating_tip()
        app._stop_thinking(first_token)

        second_token = app._start_thinking()
        second_tip = app._current_generating_tip()
        app._stop_thinking(second_token)

        self.assertNotEqual(first_tip, second_tip)
        self.assertEqual(first_tip, app._GENERATING_TIPS[0])
        self.assertEqual(second_tip, app._GENERATING_TIPS[1])

    def test_init_client_does_not_eagerly_start_runtime(self) -> None:
        app = OpenJetApp()
        fake_client = FakeInitClient()

        async def _run() -> None:
            with patch("src.app.create_runtime_client", return_value=fake_client), patch(
                "src.app.build_system_prompt",
                new=AsyncMock(return_value="system"),
            ):
                await app._init_client()

        asyncio.run(_run())

        fake_client.start.assert_not_awaited()
        self.assertIsNotNone(app.agent)

    def test_tool_output_line_highlights_shell_commands(self) -> None:
        app = OpenJetApp()

        rendered = app._format_tool_output_line("git status --short")

        self.assertIn("[command]", rendered)

    def test_assistant_output_line_styles_inline_code(self) -> None:
        app = OpenJetApp()

        rendered, in_code_block = app._format_assistant_output_line(
            "Look in `libcity/` first.",
            in_code_block=False,
        )

        self.assertFalse(in_code_block)
        self.assertIsInstance(rendered, Text)
        self.assertEqual(rendered.plain, "Look in libcity/ first.")
        self.assertTrue(any(span.style == "code" for span in rendered.spans))

    def test_assistant_output_line_styles_markdown_bold(self) -> None:
        app = OpenJetApp()

        rendered, in_code_block = app._format_assistant_output_line(
            "Use **focused context** before broad search.",
            in_code_block=False,
        )

        self.assertFalse(in_code_block)
        self.assertIsInstance(rendered, Text)
        self.assertEqual(rendered.plain, "Use focused context before broad search.")
        self.assertTrue(any(span.style == "bold" for span in rendered.spans))

    def test_tool_result_syntax_uses_python_for_read_file(self) -> None:
        app = OpenJetApp()
        tool_call = ToolCall(name="read_file", arguments={"path": "libcity/model/demo.py"})

        rendered = app._tool_result_syntax(
            ["def demo():", "    return 1"],
            tool_call=tool_call,
        )

        self.assertIsInstance(rendered, Syntax)
        self.assertEqual(rendered.lexer.name.lower(), "python")

    def test_render_approval_bar_keeps_toolbar_bar_hidden(self) -> None:
        app = OpenJetApp()
        app._awaiting_approval = True
        app._approval_choice = 0
        app._approval_tool_call = ToolCall(name="shell", arguments={"command": "nvidia-smi"})

        app._render_approval_bar()

        approval_bar = app.query_one("#approval-bar")
        self.assertTrue(approval_bar.hidden)
        self.assertEqual(approval_bar.text, "")
        assistant_status = app.query_one("#assistant-status")
        self.assertTrue(assistant_status.hidden)
        self.assertEqual(assistant_status.text, "")

    def test_prompt_message_does_not_include_approval_status_above_prompt(self) -> None:
        app = OpenJetApp()
        app._awaiting_approval = True
        app._approval_choice = 1
        app._approval_tool_call = ToolCall(name="shell", arguments={"command": "nvidia-smi"})

        app._render_approval_bar()
        prompt = app._prompt_message()

        self.assertNotIn("Awaiting confirmation", prompt.value)
        self.assertNotIn("Generating...", prompt.value)

    def test_render_approval_bar_updates_chat_selection_line(self) -> None:
        app = OpenJetApp()
        app._awaiting_approval = True
        app._approval_choice = 0
        app._approval_tool_call = ToolCall(name="shell", arguments={"command": "nvidia-smi"})

        app._write_approval_prompt(app.query_one("#chat-log"), app._approval_tool_call)
        app._approval_choice = 1
        app._render_approval_bar()

        entries = app.query_one("#chat-log")._entries
        self.assertIn("Approve", str(entries[2]))
        self.assertIn("Deny", str(entries[2]))
        self.assertIn("deny_selected", str(entries[2]))
        self.assertIn("approve_idle", str(entries[2]))
        self.assertIn("Tab/←/→ Enter y/n", str(entries[2]))

    def test_write_approval_prompt_writes_visible_chat_block(self) -> None:
        app = OpenJetApp()
        tool_call = ToolCall(name="edit_file", arguments={"path": "src/app.py"})

        app._write_approval_prompt(app.query_one("#chat-log"), tool_call)

        entries = app.query_one("#chat-log")._entries
        self.assertGreaterEqual(len(entries), 3)
        self.assertIn("edit_file -> src/app.py", str(entries[1]))
        self.assertIn("approve_selected", str(entries[2]))
        self.assertIn("deny_idle", str(entries[2]))
        self.assertIn("Tab/←/→ Enter y/n", str(entries[2]))

    def test_set_approval_choice_updates_only_selection_line(self) -> None:
        app = OpenJetApp()
        app._awaiting_approval = True
        app._approval_choice = 0
        app._approval_tool_call = ToolCall(name="shell", arguments={"command": "nvidia-smi"})
        log = app.query_one("#chat-log")
        log.refresh_recent_line = Mock()
        app._session = SimpleNamespace(app=SimpleNamespace(invalidate=Mock()))

        app._write_approval_prompt(log, app._approval_tool_call)
        app._set_approval_choice(1)

        log.refresh_recent_line.assert_called_once()
        app._session.app.invalidate.assert_not_called()

    def test_cycle_approval_choice_toggles_between_options(self) -> None:
        app = OpenJetApp()
        app._awaiting_approval = True
        app._approval_choice = 0
        app._approval_tool_call = ToolCall(name="shell", arguments={"command": "nvidia-smi"})
        log = app.query_one("#chat-log")
        log.refresh_recent_line = Mock()

        app._write_approval_prompt(log, app._approval_tool_call)
        app._cycle_approval_choice(1)
        self.assertEqual(app._approval_choice, 1)

        app._cycle_approval_choice(1)
        self.assertEqual(app._approval_choice, 0)

        app._cycle_approval_choice(-1)
        self.assertEqual(app._approval_choice, 1)


class AppInputTests(unittest.IsolatedAsyncioTestCase):
    async def test_submit_text_while_awaiting_approval_prompts_for_confirmation(self) -> None:
        app = OpenJetApp()
        app._awaiting_approval = True

        await app.submit_text("hello")

        entries = app.query_one("#chat-log")._entries
        self.assertGreaterEqual(len(entries), 2)
        self.assertIn("Respond to the pending tool confirmation first.", str(entries[0]))

    async def test_wait_for_tool_approval_clears_generating_status(self) -> None:
        app = OpenJetApp()
        app._assistant_status_kind = "generating"
        app._render_assistant_status()
        tool_call = ToolCall(name="shell", arguments={"command": "nvidia-smi"})

        wait_task = asyncio.create_task(app._wait_for_tool_approval(tool_call))
        await asyncio.sleep(0)

        prompt = app._prompt_message()
        self.assertNotIn("Generating...", prompt.value)

        app._resolve_approval(True)
        await wait_task

    async def test_submit_text_can_bypass_slash_command_handling_when_requested(self) -> None:
        app = OpenJetApp()
        app.agent = SimpleNamespace(
            messages=[],
            set_turn_context=Mock(),
            persistent_context_tokens=Mock(return_value=0),
            runtime_overhead_tokens=Mock(return_value=0),
            _messages_for_runtime=Mock(return_value=[]),
        )
        app.client = SimpleNamespace(context_window_tokens=2048)

        with patch.object(app.commands, "maybe_handle", AsyncMock()) as maybe_handle, patch.object(
            app,
            "_load_mentioned_devices_registry_into_prompt",
            AsyncMock(side_effect=lambda text, *_args: text),
        ), patch.object(
            app,
            "_load_mentioned_files_into_context",
            AsyncMock(),
        ), patch.object(
            app,
            "_start_agent_turn",
        ), patch.object(
            app,
            "persist_session_state",
        ), patch.object(
            app,
            "persist_harness_state",
        ), patch.object(
            app,
            "_render_token_counter",
        ), patch.object(
            app,
            "_begin_turn_trace",
        ):
            await app.submit_text("/spoken literal", allow_slash_command=False)

        maybe_handle.assert_not_awaited()
        self.assertEqual(app.agent.messages[-1]["content"], "/spoken literal")
        entries = app.query_one("#chat-log")._entries
        self.assertTrue(any("User" in str(entry) for entry in entries))
        self.assertFalse(any("Slash Command" in str(entry) for entry in entries))

    def test_status_cli_does_not_import_tui_surface(self) -> None:
        original_app_module = sys.modules.pop("src.app", None)
        stdout = io.StringIO()

        try:
            with patch("src.cli.load_config", return_value={"llama_model": "model.gguf"}), patch(
                "src.cli.active_model_ref", return_value="model.gguf"
            ), patch(
                "src.cli.airgapped_from_cfg",
                return_value=False,
            ), patch("sys.stdout", stdout):
                cli_main(["status"])
        finally:
            if original_app_module is not None:
                sys.modules["src.app"] = original_app_module

        self.assertIn("Runtime: Local model: llama.cpp (GGUF)", stdout.getvalue())
        if original_app_module is None:
            self.assertNotIn("src.app", sys.modules)
        else:
            self.assertIs(sys.modules["src.app"], original_app_module)

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

    def test_runtime_status_snapshot_includes_runtime_overhead_and_context_breakdown(self) -> None:
        app = OpenJetApp()
        app.agent = FakeBudgetAgent(overhead=222)
        app.client = SimpleNamespace(context_window_tokens=4096, reasoning_status=lambda: "default")
        app._last_turn_context_snapshot = SimpleNamespace(
            state_summary_tokens=111,
            budget_alerts=["layer2 high"],
            candidate_decisions=[{"label": "a"}, {"label": "b"}],
            budget=SimpleNamespace(docs_budget=333, remaining_budget=444),
        )

        with patch("src.app.read_memory_snapshot", return_value=None):
            snapshot = app.runtime_status_snapshot()

        self.assertEqual(snapshot["runtime_overhead_tokens"], 222)
        self.assertEqual(snapshot["harness_state_summary_tokens"], 111)
        self.assertEqual(snapshot["harness_candidate_count"], 2)
        self.assertEqual(snapshot["turn_docs_budget"], 333)
        self.assertEqual(snapshot["turn_context_remaining_budget"], 444)

    def test_current_tps_uses_decode_window_not_prefill(self) -> None:
        app = OpenJetApp()
        app._thinking_timer = True
        app._generation_started_at = 100.0
        app._generation_decode_started_at = 102.0
        app._generation_tokens_streamed = 61
        app._generation_decode_tokens_streamed = 60

        with patch("src.app.time.monotonic", return_value=104.0):
            tps = app._current_tps()

        self.assertEqual(tps, 30.0)

    def test_remaining_prompt_tokens_reserves_runtime_overhead_when_requested(self) -> None:
        app = OpenJetApp()
        app.agent = FakeBudgetAgent(estimated=600, overhead=300, prompt_tokens=1400)

        self.assertEqual(app._remaining_prompt_tokens(), 800)
        self.assertEqual(app._remaining_prompt_tokens(reserve_next_turn_overhead=True), 500)

    def test_prepare_turn_context_accounts_for_runtime_overhead(self) -> None:
        app = OpenJetApp()
        app.agent = FakeBudgetAgent(persistent=900, overhead=250)
        app.client = SimpleNamespace(context_window_tokens=4096)
        app.harness_state = SimpleNamespace(mode="chat")
        app._active_turn_id = None

        with patch("src.app.build_turn_context") as build_turn_context, patch(
            "src.app.read_memory_snapshot",
            return_value=None,
        ):
            build_turn_context.return_value = SimpleNamespace(
                messages=[],
                docs_loaded=[],
                docs_tokens=0,
                state_summary="",
                state_summary_tokens=0,
                layer_tokens={},
                layer_docs={},
                budget_alerts=[],
                candidate_decisions=[],
                budget=SimpleNamespace(
                    effective_window=4096,
                    usable_prompt_budget=3000,
                    remaining_budget=1000,
                    docs_budget=300,
                    layer1_budget=100,
                    layer2_budget=100,
                    layer3_budget=100,
                    layer_alert_tokens=100,
                ),
            )

            app._prepare_turn_context()

        _, kwargs = build_turn_context.call_args
        self.assertEqual(kwargs["current_context_tokens"], 1150)


class SetupWizardTests(unittest.IsolatedAsyncioTestCase):
    def test_runtime_prompt_options_note_setup_can_provision_llama_server(self) -> None:
        options = _runtime_prompt_options({}, llama_ready=False)
        self.assertEqual(len(options), 1)
        llama_label = next(label for label, key in options if key == "llama_cpp")
        self.assertIn("setup can provision llama-server", llama_label)

    async def test_prompt_choice_supports_arrow_navigation_with_prompt_session(self) -> None:
        test_case = self

        class FakeBuffer:
            def reset(self) -> None:
                return

        class FakeApp:
            def __init__(self) -> None:
                self.result = None

            def invalidate(self) -> None:
                return

            def exit(self, *, result=None) -> None:
                self.result = result

        class FakeEvent:
            def __init__(self, app: FakeApp) -> None:
                self.app = app
                self.current_buffer = FakeBuffer()

        class FakeSession:
            async def prompt_async(self, message, **kwargs):
                app = FakeApp()
                event = FakeEvent(app)
                bindings = kwargs["key_bindings"]
                handlers = {
                    tuple(binding.keys): binding.handler
                    for binding in bindings.bindings
                }
                rendered = message()
                test_case.assertIn("Hardware profile", rendered.value)
                handlers[("down",)](event)
                enter_handler = handlers.get(("enter",))
                if enter_handler is None:
                    for keys, handler in handlers.items():
                        if keys and getattr(keys[0], "value", None) == "c-m":
                            enter_handler = handler
                            break
                test_case.assertIsNotNone(enter_handler)
                enter_handler(event)
                return app.result

        session = FakeSession()
        choice = await _prompt_choice(
            session,
            Mock(),
            "Hardware profile",
            [("Auto", "auto"), ("Manual", "manual")],
        )

        self.assertEqual(choice, "manual")

    async def test_run_setup_wizard_persists_manual_llama_model_path_in_history(self) -> None:
        console = Mock()
        hardware = HardwareInfo(label="CUDA-capable device", total_ram_gb=16.0, has_cuda=True)
        manual_model = "/models/custom.gguf"
        profile_name = "Custom Model"

        choices = iter(["manual", "auto", "__local__", "__manual__", 4096, 99])
        texts = iter([manual_model, profile_name])

        async def fake_choice(*_args, **_kwargs):
            return next(choices)

        async def fake_text(*_args, **_kwargs):
            return next(texts)

        with patch("src.setup._prompt_choice", side_effect=fake_choice), patch(
            "src.setup._prompt_text", side_effect=fake_text
        ), patch("src.setup.discover_model_files", return_value=[]), patch(
            "src.setup.recommended_context_window_tokens", return_value=4096
        ), patch(
            "src.setup.recommended_context_window_tokens_from_total", return_value=4096
        ), patch("src.setup.recommended_gpu_layers", return_value=99), patch(
            "pathlib.Path.is_file", return_value=True
        ):
            payload = await run_setup_wizard(
                session=None,
                console=console,
                hardware_info=hardware,
                recommended_ctx=4096,
                current_cfg={},
            )

        self.assertIsNotNone(payload)
        self.assertEqual(payload["llama_model"], manual_model)
        self.assertEqual(payload["setup_model_history"], {"llama_cpp": [manual_model]})
        self.assertEqual(payload["model_profile_name"], profile_name)

    async def test_run_setup_wizard_lists_saved_llama_model_path_on_next_run(self) -> None:
        console = Console(file=io.StringIO(), force_terminal=False, color_system=None)
        hardware = HardwareInfo(label="CUDA-capable device", total_ram_gb=16.0, has_cuda=True)
        saved_model = "/models/custom.gguf"
        captured_options: list[tuple[str, object]] = []

        async def fake_choice(_session, _console, title, options, **_kwargs):
            if title == "Setup mode":
                return "manual"
            if title == "Hardware profile":
                return "auto"
            if title == "Model source":
                return "__local__"
            if title == "Local model":
                captured_options.extend(options)
                raise StopSetupWizard()
            raise AssertionError(f"Unexpected setup prompt: {title}")

        with patch("src.setup._prompt_choice", side_effect=fake_choice), patch(
            "src.setup.discover_model_files", return_value=[]
        ):
            with self.assertRaises(StopSetupWizard):
                await run_setup_wizard(
                    session=None,
                    console=console,
                    hardware_info=hardware,
                    recommended_ctx=4096,
                    current_cfg={"setup_model_history": {"llama_cpp": [saved_model]}},
                )

        self.assertIn((f"{Path(saved_model).name} (saved)", saved_model), captured_options)

    async def test_run_setup_wizard_prefills_current_llama_defaults(self) -> None:
        console = Mock()
        hardware = HardwareInfo(label="CUDA-capable device", total_ram_gb=16.0, has_cuda=True)
        current_model = "/models/current.gguf"
        profile_name = "Current Model"
        captured_defaults: dict[str, object] = {}
        text_defaults: list[str] = []

        async def fake_choice(_session, _console, title, options, **kwargs):
            captured_defaults[title] = kwargs.get("default_index")
            if title == "Setup mode":
                return "guided"
            if title == "Hardware profile":
                return "auto"
            if title == "Model source":
                return "__local__"
            if title == "Local model":
                return "__manual__"
            if title == "Context window":
                return options[kwargs["default_index"]][1]
            if title == "GPU layers":
                return options[kwargs["default_index"]][1]
            raise AssertionError(f"Unexpected setup prompt: {title}")

        async def fake_text(*_args, **kwargs):
            default = str(kwargs.get("default", ""))
            text_defaults.append(default)
            if len(text_defaults) == 1:
                return current_model
            return profile_name

        with patch("src.setup._prompt_choice", side_effect=fake_choice), patch(
            "src.setup._prompt_text", side_effect=fake_text
        ), patch("src.setup.discover_model_files", return_value=[]), patch(
            "src.setup.recommended_context_window_tokens", return_value=4096
        ), patch(
            "src.setup.recommended_context_window_tokens_from_total", return_value=4096
        ), patch("src.setup.recommended_gpu_layers", return_value=99), patch(
            "pathlib.Path.is_file", return_value=True
        ):
            payload = await run_setup_wizard(
                session=None,
                console=console,
                hardware_info=hardware,
                recommended_ctx=4096,
                current_cfg={
                    "llama_model": current_model,
                    "context_window_tokens": 3072,
                    "gpu_layers": 20,
                },
            )

        self.assertIsNotNone(payload)
        self.assertEqual(payload["llama_model"], current_model)
        self.assertEqual(payload["context_window_tokens"], 4096)
        self.assertEqual(payload["gpu_layers"], 20)
        self.assertEqual(text_defaults[0], current_model)
        self.assertIsInstance(captured_defaults["Context window"], int)
        self.assertIsInstance(captured_defaults["GPU layers"], int)

    def test_build_recommended_payload_marks_missing_runtime_and_prefers_direct_model(self) -> None:
        hardware = HardwareInfo(label="CPU-only device", total_ram_gb=8.0, has_cuda=False)
        with patch("src.setup.discover_model_files", return_value=[]), patch(
            "src.setup._discover_llama_server", return_value=None
        ), patch("src.setup.recommended_context_window_tokens", return_value=2048), patch(
            "src.setup.recommended_context_window_tokens_from_total", return_value=2048
        ):
            payload = build_recommended_payload(
                hardware_info=hardware,
                recommended_ctx=2048,
                current_cfg={},
            )

        self.assertEqual(payload["model_source"], "direct")
        self.assertTrue(payload["setup_missing_runtime"])
        self.assertIn("model_download_url", payload)
        self.assertIn("model_download_path", payload)

    def test_build_recommended_payload_stays_local_first_even_with_api_key_env(self) -> None:
        hardware = HardwareInfo(label="CPU-only device", total_ram_gb=8.0, has_cuda=False)
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key", "OPENROUTER_API_KEY": "router-key"}, clear=False), patch(
            "src.setup.discover_model_files", return_value=[]
        ), patch("src.setup._discover_llama_server", return_value=None), patch(
            "src.setup.recommended_context_window_tokens", return_value=2048
        ), patch(
            "src.setup.recommended_context_window_tokens_from_total", return_value=2048
        ):
            payload = build_recommended_payload(
                hardware_info=hardware,
                recommended_ctx=2048,
                current_cfg={},
            )

    def test_build_recommended_payload_uses_configured_direct_model_catalog(self) -> None:
        hardware = HardwareInfo(label="CPU-only device", total_ram_gb=12.0, has_cuda=False)
        current_cfg = {
            "setup_recommendations": {
                "direct_models": [
                    {
                        "max_ram_gb": 6,
                        "label": "Qwen3.5 4B",
                        "filename": "Qwen_Qwen3.5-4B-Q4_K_M.gguf",
                        "url": "https://example.invalid/qwen-4b.gguf",
                    },
                    {
                        "max_ram_gb": 12,
                        "label": "Qwen3.5 9B",
                        "filename": "Qwen_Qwen3.5-9B-Q4_K_M.gguf",
                        "url": "https://example.invalid/qwen-9b.gguf",
                    },
                    {
                        "max_ram_gb": 24,
                        "label": "Qwen3.5 27B",
                        "filename": "Qwen_Qwen3.5-27B-Q4_K_M.gguf",
                        "url": "https://example.invalid/qwen-27b.gguf",
                    },
                ]
            }
        }

        with patch("src.setup.discover_model_files", return_value=[]), patch(
            "src.setup._discover_llama_server", return_value=None
        ), patch("src.setup.recommended_context_window_tokens", return_value=4096), patch(
            "src.setup.recommended_context_window_tokens_from_total", return_value=4096
        ):
            payload = build_recommended_payload(
                hardware_info=hardware,
                recommended_ctx=4096,
                current_cfg=current_cfg,
            )

        self.assertEqual(payload["model_download_url"], "https://example.invalid/qwen-9b.gguf")
        self.assertTrue(str(payload["model_download_path"]).endswith("Qwen_Qwen3.5-9B-Q4_K_M.gguf"))

    async def test_run_setup_wizard_returns_recommended_payload_immediately(self) -> None:
        console = Mock()
        hardware = HardwareInfo(label="CPU-only device", total_ram_gb=8.0, has_cuda=False)

        async def fake_choice(_session, _console, title, _options, **_kwargs):
            if title == "Setup mode":
                return "recommended"
            raise AssertionError(f"Unexpected prompt after recommended selection: {title}")

        with patch("src.setup._prompt_choice", side_effect=fake_choice), patch(
            "src.setup.build_recommended_payload",
            return_value={
                "model_source": "direct",
                "model_download_url": "https://example.invalid/model.gguf",
                "model_download_path": "/models/base.gguf",
                "device": "cpu",
                "context_window_tokens": 2048,
                "gpu_layers": 0,
                "setup_complete": True,
                "model_profile_name": "Qwen3 4B",
            },
        ):
            payload = await run_setup_wizard(
                session=None,
                console=console,
                hardware_info=hardware,
                recommended_ctx=2048,
                current_cfg={},
            )

        self.assertEqual(payload["model_source"], "direct")
        self.assertEqual(payload["model_download_path"], "/models/base.gguf")

    async def test_run_setup_wizard_supports_direct_download_local_runtime(self) -> None:
        console = Mock()
        hardware = HardwareInfo(label="CPU-only device", total_ram_gb=8.0, has_cuda=False)
        profile_name = "Downloaded Local Model"
        selected_model = {
            "max_ram_gb": 12.0,
            "label": "Qwen3.5 9B",
            "filename": "Qwen3.5-9B-Q4_K_M.gguf",
            "url": "https://huggingface.co/unsloth/Qwen3.5-9B-GGUF/resolve/main/Qwen3.5-9B-Q4_K_M.gguf?download=true",
        }

        choices = iter(["manual", "auto", "__direct__", selected_model, 2048, 0])
        texts = iter([profile_name])

        async def fake_choice(*_args, **_kwargs):
            return next(choices)

        async def fake_text(*_args, **_kwargs):
            return next(texts)

        with patch("src.setup._prompt_choice", side_effect=fake_choice), patch(
            "src.setup._prompt_text", side_effect=fake_text
        ), patch("src.setup.discover_model_files", return_value=[]), patch(
            "src.setup.recommended_context_window_tokens", return_value=2048
        ), patch(
            "src.setup.recommended_context_window_tokens_from_total", return_value=2048
        ), patch("src.setup.recommended_gpu_layers", return_value=0):
            payload = await run_setup_wizard(
                session=None,
                console=console,
                hardware_info=hardware,
                recommended_ctx=2048,
                current_cfg={},
            )

        self.assertIsNotNone(payload)
        self.assertEqual(payload["model_source"], "direct")
        self.assertTrue(payload["setup_missing_model"])
        self.assertEqual(payload["model_download_url"], selected_model["url"])
        self.assertTrue(str(payload["model_download_path"]).endswith(selected_model["filename"]))
        self.assertEqual(payload["model_profile_name"], profile_name)

    async def test_run_setup_wizard_allows_catalog_download_selection(self) -> None:
        console = Mock()
        hardware = HardwareInfo(label="CPU-only device", total_ram_gb=8.0, has_cuda=False)
        catalog = [
            {
                "max_ram_gb": 6.0,
                "label": "Small",
                "filename": "small.gguf",
                "url": "https://huggingface.co/example/small-GGUF/resolve/main/small.gguf?download=true",
            },
            {
                "max_ram_gb": 24.0,
                "label": "Large",
                "filename": "large.gguf",
                "url": "https://huggingface.co/example/large-GGUF/resolve/main/large.gguf?download=true",
            },
        ]
        prompts: list[tuple[str, list[str]]] = []
        choices = iter(["manual", "auto", "__direct__", catalog[1], 2048, 0])
        texts = iter(["Large"])

        async def fake_choice(_session, _console, title, options, **_kwargs):
            prompts.append((title, [label for label, _value in options]))
            return next(choices)

        async def fake_text(*_args, **_kwargs):
            return next(texts)

        with patch("src.setup._prompt_choice", side_effect=fake_choice), patch(
            "src.setup._prompt_text", side_effect=fake_text
        ), patch("src.setup.discover_model_files", return_value=[]), patch(
            "src.setup.setup_direct_model_catalog", return_value=tuple(catalog)
        ), patch("src.setup.recommend_direct_model", return_value={**catalog[0], "target_path": "/models/small.gguf"}), patch(
            "src.setup.recommended_context_window_tokens", return_value=2048
        ), patch(
            "src.setup.recommended_context_window_tokens_from_total", return_value=2048
        ), patch("src.setup.recommended_gpu_layers", return_value=0):
            payload = await run_setup_wizard(
                session=None,
                console=console,
                hardware_info=hardware,
                recommended_ctx=2048,
                current_cfg={},
            )

        model_download_prompt = next(labels for title, labels in prompts if title == "Model download")
        self.assertEqual(len(model_download_prompt), 2)
        self.assertTrue(any("Small" in label for label in model_download_prompt))
        self.assertTrue(any("Large" in label for label in model_download_prompt))
        self.assertEqual(payload["model_download_url"], catalog[1]["url"])
        self.assertTrue(str(payload["model_download_path"]).endswith("large.gguf"))

    async def test_run_setup_wizard_defaults_name_and_sizing_to_selected_catalog_model(self) -> None:
        console = Mock()
        hardware = HardwareInfo(
            label="CUDA-capable device",
            total_ram_gb=64.0,
            has_cuda=True,
            vram_mb=24576.0,
        )
        catalog = [
            {
                "max_ram_gb": 12.0,
                "label": "Recommended",
                "filename": "recommended.gguf",
                "url": "https://huggingface.co/example/recommended-GGUF/resolve/main/recommended.gguf",
                "model_size_mb": 4096,
                "kv_bytes_per_token": 32768,
            },
            {
                "max_ram_gb": 32.0,
                "label": "Selected MoE",
                "filename": "selected-moe.gguf",
                "url": "https://huggingface.co/example/selected-moe-GGUF/resolve/main/selected-moe.gguf",
                "model_size_mb": 22630,
                "kv_bytes_per_token": 24576,
                "unified_memory_only": True,
            },
        ]
        defaults: dict[str, object] = {}
        choices = iter(["manual", "auto", "__direct__", catalog[1], 32768, 99])

        async def fake_choice(_session, _console, title, options, **kwargs):
            defaults[title] = options[kwargs.get("default_index", 0)][1]
            return next(choices)

        async def fake_text(_session, _prompt, *, default=""):
            defaults["model name"] = default
            return default

        with patch("src.setup._prompt_choice", side_effect=fake_choice), patch(
            "src.setup._prompt_text", side_effect=fake_text
        ), patch("src.setup.discover_model_files", return_value=[]), patch(
            "src.setup.setup_direct_model_catalog", return_value=tuple(catalog)
        ), patch("src.setup.recommend_direct_model", return_value={**catalog[0], "target_path": "/models/recommended.gguf"}), patch(
            "src.setup.recommended_context_window_tokens", return_value=2048
        ), patch(
            "src.setup.recommended_context_window_tokens_from_total", return_value=2048
        ), patch("src.setup.recommended_gpu_layers", return_value=99):
            payload = await run_setup_wizard(
                session=None,
                console=console,
                hardware_info=hardware,
                recommended_ctx=2048,
                current_cfg={
                    "model_source": "direct",
                    "model_download_url": catalog[0]["url"],
                    "context_window_tokens": 2048,
                    "gpu_layers": 0,
                },
            )

        self.assertEqual(defaults["model name"], "Selected MoE")
        self.assertGreater(int(defaults["Context window"]), 2048)
        self.assertEqual(defaults["GPU layers"], 99)
        self.assertEqual(payload["model_profile_name"], "Selected MoE")


class ProvisioningTests(unittest.IsolatedAsyncioTestCase):
    def test_recommend_direct_model_uses_requested_default_q4_bands(self) -> None:
        self.assertEqual(
            recommend_direct_model(HardwareInfo(label="6GB", total_ram_gb=6.0, has_cuda=False))["label"],
            "Qwen3.5 4B",
        )
        self.assertEqual(
            recommend_direct_model(HardwareInfo(label="12GB", total_ram_gb=12.0, has_cuda=False))["label"],
            "Qwen3.5 9B",
        )
        self.assertEqual(
            recommend_direct_model(HardwareInfo(label="24GB", total_ram_gb=24.0, has_cuda=False))["label"],
            "Qwen3.5 27B",
        )

    def test_recommend_direct_model_allows_moe_catalog_entries_on_gpu(self) -> None:
        recommended = recommend_direct_model(
            HardwareInfo(
                label="CUDA-capable device",
                total_ram_gb=32.0,
                has_cuda=True,
                vram_mb=12288.0,
            )
        )

        self.assertIn(recommended["label"], {"Gemma 4 26B A4B", "Qwen3.6 35B A3B"})

    def test_recommend_direct_model_prioritizes_large_dense_before_moe(self) -> None:
        recommended = recommend_direct_model(
            HardwareInfo(
                label="RTX 3090",
                total_ram_gb=32.0,
                has_cuda=True,
                vram_mb=24576.0,
            )
        )

        self.assertEqual(recommended["label"], "Qwen3.5 27B")

    def test_recommend_direct_model_prioritizes_moe_before_small_dense(self) -> None:
        recommended = recommend_direct_model(
            HardwareInfo(
                label="RX 6700 XT",
                total_ram_gb=32.0,
                has_cuda=False,
                has_vulkan=True,
                vram_mb=12288.0,
            )
        )

        self.assertEqual(recommended["label"], "Qwen3.6 35B A3B")

    async def test_ensure_direct_model_downloads_target_file(self) -> None:
        updates: list[str] = []
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "model.gguf"
            payload = {
                "model_source": "direct",
                "model_download_url": "https://huggingface.co/example/test-GGUF/resolve/main/model.gguf?download=true",
                "model_download_path": str(target),
                "setup_missing_model": True,
            }

            async def fake_download(**kwargs):
                self.assertEqual(kwargs["repo_id"], "example/test-GGUF")
                self.assertEqual(kwargs["filename"], "model.gguf")
                self.assertEqual(kwargs["revision"], "main")
                kwargs["progress"]("model.gguf: 50%|#####     | 4.0M/8.0M")
                Path(kwargs["local_dir"], "model.gguf").write_bytes(b"test-gguf")
                return 0, "", ""

            log = Mock()
            with patch("src.provisioning._run_hf_cli_download", side_effect=fake_download) as download:
                resolved = await ensure_direct_model(
                    payload,
                    log=log,
                    set_status=updates.append,
                    clear_status=lambda: None,
                )
            written = target.read_bytes()

        self.assertEqual(written, b"test-gguf")
        self.assertEqual(resolved["llama_model"], str(target))
        self.assertFalse(resolved["setup_missing_model"])
        self.assertTrue(any(text.startswith("downloading model.gguf") for text in updates))
        self.assertTrue(any("50%" in str(call.args[0]) for call in log.write.call_args_list))
        download.assert_awaited_once()


class AppSetupOrderingTests(unittest.IsolatedAsyncioTestCase):
    async def test_run_setup_command_saves_config_before_materializing_model(self) -> None:
        app = OpenJetApp()
        app.cfg["logging"] = {"enabled": False}
        events: list[str] = []

        with patch.object(app, "_run_setup_wizard", AsyncMock(return_value={"llama_model": "/models/base.gguf"})), patch(
            "src.app.save_config", side_effect=lambda _cfg: events.append("save")
        ), patch.object(
            app,
            "_materialize_setup_model",
            AsyncMock(side_effect=lambda result, _log: events.append("materialize") or {**result, "llama_model": "/models/resolved.gguf"}),
        ), patch.object(app, "_init_client", AsyncMock(side_effect=lambda: events.append("init"))), patch.object(
            app, "_render_token_counter"
        ), patch.object(
            app, "persist_session_state"
        ):
            applied = await app.run_setup_command(app.query_one("#chat-log"))

        self.assertTrue(applied)
        self.assertEqual(events[:3], ["save", "materialize", "save"])
        self.assertEqual(events[3], "init")
        entries = app.query_one("#chat-log")._entries
        self.assertTrue(any("Setup complete. Join the Discord" in str(entry) for entry in entries))

    async def test_startup_force_setup_saves_config_before_materializing_model(self) -> None:
        app = OpenJetApp(force_setup=True)
        app.cfg["logging"] = {"enabled": False}
        events: list[str] = []

        with patch.object(app, "_run_setup_wizard", AsyncMock(return_value={"llama_model": "/models/base.gguf"})), patch(
            "src.app.save_config", side_effect=lambda _cfg: events.append("save")
        ), patch.object(
            app,
            "_materialize_setup_model",
            AsyncMock(side_effect=lambda result, _log: events.append("materialize") or {**result, "llama_model": "/models/resolved.gguf"}),
        ), patch.object(app, "_init_client", AsyncMock(side_effect=lambda: events.append("init"))), patch.object(
            app, "_maybe_prompt_for_startup_update", AsyncMock()
        ), patch.object(
            app, "_render_token_counter"
        ), patch.object(
            app, "_restore_harness_state"
        ):
            await app._startup_sequence()

        self.assertEqual(events[:3], ["save", "materialize", "save"])
        self.assertEqual(events[3], "init")
        entries = app.query_one("#chat-log")._entries
        self.assertTrue(any("Setup complete. Join the Discord" in str(entry) for entry in entries))

    def test_persist_setup_result_keeps_existing_named_profiles(self) -> None:
        app = OpenJetApp()
        app.cfg["active_model_profile"] = "Base"
        app.cfg["model_profiles"] = [
            {
                "name": "Base",
                "model_source": "local",
                "llama_model": "/models/base.gguf",
                "context_window_tokens": 4096,
                "gpu_layers": 99,
            }
        ]

        with patch("src.app.save_config"):
            app._persist_setup_result(
                {
                    "model_source": "local",
                    "llama_model": "/models/alt.gguf",
                    "context_window_tokens": 8192,
                    "gpu_layers": 70,
                    "model_profile_name": "Alt",
                }
            )

        self.assertEqual(app.cfg["active_model_profile"], "Alt")
        self.assertEqual(
            [profile["name"] for profile in app.cfg["model_profiles"]],
            ["Alt", "Base"],
        )

    def test_persist_setup_result_discards_system_prompt_from_config(self) -> None:
        app = OpenJetApp()
        app.cfg["system_prompt"] = "old prompt"

        with patch("src.app.save_config"):
            app._persist_setup_result(
                {
                    "system_prompt": "should not persist",
                }
            )

        self.assertNotIn("system_prompt", app.cfg)


class ModelCommandTests(unittest.IsolatedAsyncioTestCase):
    async def test_model_command_switches_to_saved_profile(self) -> None:
        app = OpenJetApp()
        app.cfg["model_profiles"] = [
            {
                "name": "alt",
                "model_source": "local",
                "llama_model": "/models/alt.gguf",
                "context_window_tokens": 4096,
                "gpu_layers": 99,
            }
        ]
        app.cfg["active_model_profile"] = "base"

        with patch.object(app, "activate_model_profile", AsyncMock(return_value=True)) as activate:
            handled = await app.commands.maybe_handle("/model alt")

        self.assertTrue(handled)
        activate.assert_awaited_once_with("alt", app.query_one("#chat-log"))

    async def test_model_command_without_args_opens_picker_and_switches(self) -> None:
        app = OpenJetApp()
        app.cfg["model_profiles"] = [
            {
                "name": "base",
                "model_source": "local",
                "llama_model": "/models/base.gguf",
                "context_window_tokens": 4096,
                "gpu_layers": 99,
            },
            {
                "name": "alt",
                "model_source": "local",
                "llama_model": "/models/alt.gguf",
                "context_window_tokens": 4096,
                "gpu_layers": 99,
            }
        ]

        app.cfg["active_model_profile"] = "base"

        picker = AsyncMock(return_value="alt")
        dialog = Mock(run_async=picker)
        with patch("src.commands.radiolist_dialog", return_value=dialog), patch.object(
            app, "activate_model_profile", AsyncMock(return_value=True)
        ) as activate:
            handled = await app.commands.maybe_handle("/model")

        self.assertTrue(handled)
        activate.assert_awaited_once_with("alt", app.query_one("#chat-log"))

    async def test_edit_model_updates_saved_profile(self) -> None:
        app = OpenJetApp()
        app.cfg["model_profiles"] = [
            {
                "name": "alt",
                "model_source": "local",
                "llama_model": "/models/alt.gguf",
                "context_window_tokens": 4096,
                "gpu_layers": 99,
            }
        ]
        app.cfg["active_model_profile"] = "base"

        prompts = iter(["Alt Edited", "/models/alt-v2.gguf", "8192", "70"])
        with patch("src.commands._prompt_text", side_effect=lambda *_args, **_kwargs: next(prompts)), patch(
            "src.commands.save_config"
        ) as save_cfg:
            handled = await app.commands.maybe_handle("/edit-model alt")

        self.assertTrue(handled)
        self.assertEqual(app.cfg["model_profiles"][0]["name"], "Alt Edited")
        self.assertEqual(app.cfg["model_profiles"][0]["llama_model"], "/models/alt-v2.gguf")
        self.assertEqual(app.cfg["model_profiles"][0]["context_window_tokens"], 8192)
        self.assertEqual(app.cfg["model_profiles"][0]["gpu_layers"], 70)
        save_cfg.assert_called_once()


class AppQuitTests(unittest.TestCase):
    def test_request_terminal_exit_clears_nonempty_buffer_before_exit(self) -> None:
        app = OpenJetApp()
        prompt_app = Mock()
        prompt_app.is_running = True
        current_buffer = Mock()
        current_buffer.text = "draft command"

        app._request_terminal_exit(prompt_app, current_buffer)

        current_buffer.reset.assert_called_once()
        prompt_app.exit.assert_not_called()
        entries = app.query_one("#chat-log")._entries
        self.assertTrue(any("Current input cleared" in str(entry) for entry in entries))

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


class AppQuitAsyncTests(unittest.IsolatedAsyncioTestCase):
    async def test_action_quit_writes_paid_api_equivalent_token_summary(self) -> None:
        app = OpenJetApp()
        app.agent = SimpleNamespace(messages=[{"role": "system", "content": "system"}])
        app.client = AsyncMock()
        app.session_logger = AsyncMock()
        app.state_store = Mock()
        app._session_prompt_tokens = 1234
        app._session_completion_tokens = 56
        app._session_runtime_requests = 3

        await app.action_quit()

        entries = app.query_one("#chat-log")._entries
        self.assertTrue(any("Saved with OpenJet" in str(entry) for entry in entries))
        self.assertTrue(any("1,234 input tokens • 56 output tokens • $0 API Cost" in str(entry) for entry in entries))
        self.assertTrue(any("Setup complete. Join the Discord" in str(entry) for entry in entries))
        app.client.close.assert_awaited_once()
        app.session_logger.stop.assert_awaited_once()
        app.state_store.save.assert_called_once()
        saved_payload = app.state_store.save.call_args.args[0]
        self.assertEqual(saved_payload["token_usage"]["prompt_tokens"], 1234)
        self.assertEqual(saved_payload["token_usage"]["completion_tokens"], 56)
        self.assertEqual(saved_payload["token_usage"]["runtime_requests"], 3)


class AppResumeStateTests(unittest.TestCase):
    def test_persist_session_state_writes_live_chat_archive(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state_path = root / ".openjet" / "state" / "session_state.json"

            app = OpenJetApp()
            app.state_store = SessionStateStore(path=state_path, enabled=True)
            app.chat_archive = ChatArchiveStore.from_session_state_path(state_path, enabled=True)
            app._chat_session_id = "chat123"
            app.agent = SimpleNamespace(
                messages=[
                    {"role": "system", "content": "system"},
                    {"role": "user", "content": "hello"},
                ]
            )
            app.client = SimpleNamespace(context_window_tokens=4096)
            app.loaded_files = {"src/app.py": {"path": "src/app.py"}}

            app.persist_session_state(reason="user_message")

            saved = json.loads(state_path.read_text(encoding="utf-8"))
            archived = json.loads(
                app.chat_archive.live_state_path("chat123").read_text(encoding="utf-8")
            )
            self.assertEqual(saved["chat_id"], "chat123")
            self.assertEqual(saved["reason"], "user_message")
            self.assertEqual(archived["chat_id"], "chat123")
            self.assertEqual(archived["loaded_files"]["src/app.py"]["path"], "src/app.py")

    def test_list_resume_candidates_prefers_resume_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state_path = root / ".openjet" / "state" / "session_state.json"
            app = OpenJetApp()
            app.state_store = SessionStateStore(path=state_path, enabled=True)
            app.chat_archive = ChatArchiveStore.from_session_state_path(state_path, enabled=True)

            payload = {
                "chat_id": "chat123",
                "saved_at": 100.0,
                "reason": "assistant_turn_done",
                "runtime": "llama_cpp",
                "model_ref": "/models/demo.gguf",
                "messages": [
                    {"role": "system", "content": "system"},
                    {"role": "user", "content": "hello world"},
                ],
                "loaded_files": {},
                "harness_state": {},
                "token_usage": {},
            }
            app.chat_archive.save_live_state("chat123", payload)
            app.chat_archive.save_resume_state("chat123", payload)
            app.chat_archive.kv_cache_path("chat123").parent.mkdir(parents=True, exist_ok=True)
            app.chat_archive.kv_cache_path("chat123").write_bytes(b"kv")

            entries = app.list_resume_candidates()

            self.assertEqual(len(entries), 1)
            self.assertEqual(entries[0].state_path.name, "resume_state.json")
            self.assertTrue(entries[0].kv_cache_available)


class AppResumeStateAsyncTests(unittest.IsolatedAsyncioTestCase):
    async def test_persist_resumable_session_state_saves_checkpoint_and_kv(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state_path = root / ".openjet" / "state" / "session_state.json"
            app = OpenJetApp()
            app.state_store = SessionStateStore(path=state_path, enabled=True)
            app.chat_archive = ChatArchiveStore.from_session_state_path(state_path, enabled=True)
            app._chat_session_id = "chat123"
            app.agent = SimpleNamespace(messages=[{"role": "system", "content": "system"}])
            app.client = SimpleNamespace(
                context_window_tokens=4096,
                save_kv_cache=AsyncMock(return_value=True),
            )

            saved = await app.persist_resumable_session_state(reason="assistant_turn_done")

            self.assertTrue(saved)
            self.assertTrue(app.chat_archive.resume_state_path("chat123").is_file())
            app.client.save_kv_cache.assert_awaited_once_with(
                app.chat_archive.kv_cache_path("chat123")
            )

    async def test_restore_saved_chat_restores_kv_cache_when_available(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state_path = root / ".openjet" / "state" / "session_state.json"
            app = OpenJetApp()
            app.state_store = SessionStateStore(path=state_path, enabled=True)
            app.chat_archive = ChatArchiveStore.from_session_state_path(state_path, enabled=True)
            app.cfg["llama_model"] = "/models/demo.gguf"
            app.agent = SimpleNamespace(
                system_prompt="system",
                messages=[{"role": "system", "content": "system"}],
                clear_turn_context=Mock(),
            )
            app.client = SimpleNamespace(
                context_window_tokens=4096,
                restore_kv_cache=AsyncMock(return_value=True),
                reset_kv_cache=AsyncMock(return_value=None),
            )
            payload = {
                "chat_id": "savedchat",
                "saved_at": 100.0,
                "reason": "assistant_turn_done",
                "runtime": "llama_cpp",
                "model_ref": "/models/demo.gguf",
                "messages": [
                    {"role": "system", "content": "system"},
                    {"role": "user", "content": "hello"},
                    {"role": "assistant", "content": "hi"},
                ],
                "loaded_files": {},
                "harness_state": {},
                "token_usage": {},
            }
            app.chat_archive.save_resume_state("savedchat", payload)
            app.chat_archive.kv_cache_path("savedchat").parent.mkdir(parents=True, exist_ok=True)
            app.chat_archive.kv_cache_path("savedchat").write_bytes(b"kv")

            restored = await app.restore_saved_chat(
                app.chat_archive.resume_state_path("savedchat"),
                app.query_one("#chat-log"),
            )

            self.assertTrue(restored)
            app.client.restore_kv_cache.assert_awaited_once_with(
                app.chat_archive.kv_cache_path("savedchat")
            )
            app.client.reset_kv_cache.assert_not_awaited()
            self.assertEqual(app.agent.messages[-1]["content"], "hi")
            self.assertTrue(
                any("KV cache restored" in str(entry) for entry in app.query_one("#chat-log")._entries)
            )

    async def test_restore_saved_chat_resets_runtime_when_model_differs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state_path = root / ".openjet" / "state" / "session_state.json"
            app = OpenJetApp()
            app.state_store = SessionStateStore(path=state_path, enabled=True)
            app.chat_archive = ChatArchiveStore.from_session_state_path(state_path, enabled=True)
            app.cfg["llama_model"] = "/models/current.gguf"
            app.agent = SimpleNamespace(
                system_prompt="system",
                messages=[{"role": "system", "content": "system"}],
                clear_turn_context=Mock(),
            )
            app.client = SimpleNamespace(
                context_window_tokens=4096,
                restore_kv_cache=AsyncMock(return_value=True),
                reset_kv_cache=AsyncMock(return_value=None),
            )
            payload = {
                "chat_id": "savedchat",
                "saved_at": 100.0,
                "reason": "assistant_turn_done",
                "runtime": "llama_cpp",
                "model_ref": "/models/other.gguf",
                "messages": [
                    {"role": "system", "content": "system"},
                    {"role": "user", "content": "hello"},
                ],
                "loaded_files": {},
                "harness_state": {},
                "token_usage": {},
            }
            app.chat_archive.save_resume_state("savedchat", payload)
            app.chat_archive.kv_cache_path("savedchat").parent.mkdir(parents=True, exist_ok=True)
            app.chat_archive.kv_cache_path("savedchat").write_bytes(b"kv")

            restored = await app.restore_saved_chat(
                app.chat_archive.resume_state_path("savedchat"),
                app.query_one("#chat-log"),
            )

            self.assertTrue(restored)
            app.client.restore_kv_cache.assert_not_awaited()
            app.client.reset_kv_cache.assert_awaited_once()
            self.assertTrue(
                any("active model differs" in str(entry) for entry in app.query_one("#chat-log")._entries)
            )


class SlashResumeCommandTests(unittest.IsolatedAsyncioTestCase):
    async def test_resume_command_uses_picker_and_loads_selected_chat(self) -> None:
        app = OpenJetApp()
        app.agent = SimpleNamespace(messages=[{"role": "system", "content": "system"}])
        entry = SavedChatEntry(
            chat_id="chat123",
            state_path=Path("/tmp/resume_state.json"),
            saved_at=100.0,
            reason="assistant_turn_done",
            preview="hello world",
            message_count=2,
            loaded_file_count=0,
            runtime="llama_cpp",
            model_ref="/models/demo.gguf",
            uses_resume_checkpoint=True,
            kv_cache_available=True,
        )
        picker = AsyncMock(return_value=str(entry.state_path))
        dialog = Mock(run_async=picker)

        with patch.object(app, "list_resume_candidates", return_value=[entry]), patch(
            "src.commands.radiolist_dialog",
            return_value=dialog,
        ), patch.object(app, "restore_saved_chat", AsyncMock(return_value=True)) as restore_chat:
            handled = await app.commands.maybe_handle("/resume")

        self.assertTrue(handled)
        restore_chat.assert_awaited_once_with(str(entry.state_path), app.query_one("#chat-log"))


class AppToolHandlingTests(unittest.IsolatedAsyncioTestCase):
    async def test_record_tool_result_refreshes_token_counter_immediately_after_result(self) -> None:
        app = OpenJetApp()
        tool_call = ToolCall(name="list_directory", arguments={"path": "."}, id="call-1")
        tool_result = ToolResult(
            tool_call=tool_call,
            output="listing",
            meta={"ok": True, "status": "completed"},
        )

        with patch.object(app, "_render_token_counter") as render_counter:
            event = app._record_tool_result(tool_result, app.query_one("#chat-log"), {})

        self.assertEqual(event["tool"], "list_directory")
        render_counter.assert_called_once()


class CliCommandTests(unittest.TestCase):
    def test_format_model_profiles_summary_lists_active_profile(self) -> None:
        text = _format_model_profiles_summary(
            {
                "active_model_profile": "base",
                "model_profiles": [
                    {
                        "name": "base",
                        "model_source": "local",
                        "llama_model": "/models/base.gguf",
                        "context_window_tokens": 4096,
                        "gpu_layers": 99,
                    }
                ],
            }
        )

        self.assertIn("Active model preset: base", text)
        self.assertIn("- base (active): context=4096", text)

    def test_format_cli_status_reports_runtime_and_airgap(self) -> None:
        text = _format_cli_status(
            {
                "active_model_profile": "base",
                "llama_model": "/models/base.gguf",
                "context_window_tokens": 4096,
                "gpu_layers": 99,
                "airgapped": True,
            }
        )

        self.assertIn("Runtime: Local model: llama.cpp (GGUF)", text)
        self.assertIn("Active model preset: base", text)
        self.assertIn("Air-gapped: true", text)

    def test_format_slash_commands_summary_includes_models_alias(self) -> None:
        text = _format_slash_commands_summary()

        self.assertIn("/model: Show or switch saved model presets", text)
        self.assertIn("/models", text)
        self.assertIn("/plan", text)
        self.assertIn("/todo", text)

    def test_main_context_command_updates_active_profile(self) -> None:
        cfg = {
            "active_model_profile": "base",
            "context_window_tokens": 4096,
                "model_profiles": [
                    {
                        "name": "base",
                        "model_source": "local",
                        "llama_model": "/models/base.gguf",
                        "context_window_tokens": 4096,
                        "gpu_layers": 99,
                }
            ],
        }

        with patch("src.cli.load_config", return_value=cfg), patch("src.cli.save_config") as save_cfg, patch(
            "builtins.print"
        ) as printer:
            main(["context", "8192"])

        printer.assert_called_once()
        saved_cfg = save_cfg.call_args.args[0]
        self.assertEqual(saved_cfg["context_window_tokens"], 8192)
        self.assertEqual(saved_cfg["model_profiles"][0]["context_window_tokens"], 8192)
        self.assertIn("Context window set to 8192 tokens", printer.call_args.args[0])

    def test_main_context_command_rejects_small_values(self) -> None:
        with patch("src.cli.load_config", return_value={}), patch("src.cli.save_config") as save_cfg:
            with self.assertRaises(SystemExit) as exc:
                main(["context", "128"])

        save_cfg.assert_not_called()
        self.assertIn("at least 512", str(exc.exception))

    def test_main_models_option_prints_saved_presets(self) -> None:
        cfg = {
            "active_model_profile": "base",
            "model_profiles": [
                {
                    "name": "base",
                    "model_source": "local",
                    "llama_model": "/models/base.gguf",
                    "context_window_tokens": 4096,
                    "gpu_layers": 99,
                }
            ],
        }

        with patch("src.cli.load_config", return_value=cfg), patch("builtins.print") as printer:
            main(["--models"])

        printer.assert_called_once()
        self.assertIn("Active model preset: base", printer.call_args.args[0])

    def test_main_setup_command_launches_app_in_setup_mode(self) -> None:
        with patch("src.cli.launch_tui") as launch_tui:
            main(["setup"])

        launch_tui.assert_called_once_with(force_setup=True)

    def test_main_version_option_prints_version(self) -> None:
        with patch("src.cli._open_jet_version", return_value="0.3.8"), patch("builtins.print") as printer:
            main(["--version"])

        printer.assert_called_once_with("open-jet 0.3.8")

    def test_parser_does_not_expose_setup_as_option(self) -> None:
        from src.cli import build_parser

        help_text = build_parser().format_help()

        self.assertNotIn("--setup", help_text)

    def test_parser_does_not_expose_models_status_or_update_as_commands(self) -> None:
        from src.cli import build_parser

        help_text = build_parser().format_help()

        self.assertNotIn(" setup,models,", help_text)
        self.assertNotIn(",commands,status,", help_text)
        self.assertNotIn(",update,", help_text)
        self.assertIn("--models", help_text)
        self.assertIn("--commands", help_text)
        self.assertIn("--status", help_text)
        self.assertIn("--update", help_text)


class AppCondenseTests(unittest.IsolatedAsyncioTestCase):
    async def test_run_agent_turn_stops_when_pressure_remains_after_condense(self) -> None:
        app = OpenJetApp()

        class CondenseLoopAgent:
            messages = [{"role": "system", "content": "system"}, {"role": "user", "content": "hi"}]
            turn_context_messages = []

            async def run_turn(self):
                yield SimpleNamespace(kind=ActionKind.CONDENSE, text="mem_available=300MB < 512MB")

            async def condense_context(self) -> str:
                return "Context condensed automatically. messages: 2 -> 2, tokens(est): 600 -> 580."

            def resource_pressure_reason(self) -> str | None:
                return "mem_available=300MB < 512MB"

            def persistent_context_tokens(self) -> int:
                return 0

            def runtime_overhead_tokens(self, *, force_post_tool_continuation: bool = False, empty_retry_count: int = 0) -> int:
                return 0

            def set_turn_context(self, messages) -> None:
                self.turn_context_messages = list(messages)

            def _messages_for_runtime(self):
                return list(self.turn_context_messages)

        app.agent = CondenseLoopAgent()

        with patch.object(app, "_start_agent_turn") as start_next, patch.object(
            app, "persist_session_state"
        ) as persist_session, patch.object(app, "persist_harness_state") as persist_harness, patch.object(
            app, "_finish_turn_trace"
        ) as finish_turn, patch.object(app, "_render_token_counter") as render_counter:
            await app.run_agent_turn()

        start_next.assert_not_called()
        persist_harness.assert_called_once()
        self.assertGreaterEqual(persist_session.call_count, 2)
        finish_turn.assert_called_once()
        render_counter.assert_called_once()
        self.assertTrue(
            any(
                "Auto-condense stopped" in str(entry)
                for entry in app.query_one("#chat-log")._entries
            )
        )

    async def test_run_agent_turn_continues_after_tool_calls_without_restarting_task(self) -> None:
        app = OpenJetApp()

        class ToolThenAnswerAgent:
            project_root = None

            def __init__(self) -> None:
                self.run_count = 0
                self.turn_context_messages = []
            messages = [{"role": "system", "content": "system"}, {"role": "user", "content": "inspect"}]

            async def run_turn(self):
                self.run_count += 1
                if self.run_count == 1:
                    yield SimpleNamespace(
                        kind=ActionKind.TOOL_REQUEST,
                        tool_call=ToolCall(name="list_directory", arguments={"path": "."}, id="call-1"),
                    )
                    return
                yield SimpleNamespace(kind=ActionKind.TEXT, text="done")
                yield SimpleNamespace(kind=ActionKind.DONE)

            def complete_tool_call(self, tool_call, result) -> None:
                self.messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": result})

            def persistent_context_tokens(self) -> int:
                return 0

            def runtime_overhead_tokens(self, *, force_post_tool_continuation: bool = False, empty_retry_count: int = 0) -> int:
                return 0

            def set_turn_context(self, messages) -> None:
                self.turn_context_messages = list(messages)

            def _messages_for_runtime(self):
                return list(self.turn_context_messages)

        app.agent = ToolThenAnswerAgent()

        async def handle_tool(tool_call, log):
            app.agent.complete_tool_call(tool_call, "listing")
            return {"tool": "list_directory", "ok": True}

        real_persist_session = app.persist_session_state

        with patch.object(app, "_handle_tool_call", AsyncMock(side_effect=handle_tool)), patch.object(
            app, "_start_agent_turn"
        ) as start_next, patch.object(
            app,
            "_save_live_session_payload",
        ), patch.object(
            app,
            "persist_session_state",
            side_effect=lambda *, reason: real_persist_session(reason=reason),
        ) as persist_session, patch(
            "src.app.reflect_agent_persistent_memory",
            AsyncMock(return_value={"ok": True, "applied": []}),
        ), patch.object(
            app, "persist_harness_state"
        ) as persist_harness:
            await app.run_agent_turn()

        start_next.assert_not_called()
        persist_harness.assert_called_once()
        self.assertEqual(app.agent.run_count, 2)
        persist_session.assert_any_call(reason="assistant_turn_with_tools")
        persist_session.assert_any_call(reason="assistant_turn_done")

    async def test_run_agent_turn_passes_completed_turn_output_to_memory_reflection(self) -> None:
        app = OpenJetApp()

        class AnswerAgent:
            project_root = Path.cwd()
            prompt_cfg = {}
            global_memory_root = None

            def __init__(self) -> None:
                self.turn_context_messages = []
                self.base_system_prompt = "base system"
                self.messages = [{"role": "system", "content": "system"}, {"role": "user", "content": "inspect"}]

            async def run_turn(self):
                yield SimpleNamespace(kind=ActionKind.TEXT, text="done")
                yield SimpleNamespace(kind=ActionKind.DONE)

            def persistent_context_tokens(self) -> int:
                return 0

            def runtime_overhead_tokens(self, *, force_post_tool_continuation: bool = False, empty_retry_count: int = 0) -> int:
                return 0

            def set_turn_context(self, messages) -> None:
                self.turn_context_messages = list(messages)

            def _messages_for_runtime(self):
                return list(self.turn_context_messages)

        app.agent = AnswerAgent()
        app._active_turn_prompt = "inspect"

        real_persist_session = app.persist_session_state

        with patch(
            "src.app.reflect_agent_persistent_memory",
            AsyncMock(return_value={"ok": True, "applied": []}),
        ) as reflect_mock, patch.object(
            app,
            "_save_live_session_payload",
        ), patch.object(
            app,
            "persist_session_state",
            side_effect=lambda *, reason: real_persist_session(reason=reason),
        ) as persist_session, patch.object(
            app, "persist_harness_state"
        ) as persist_harness, patch.object(
            app, "persist_resumable_session_state",
            AsyncMock(),
        ):
            await app.run_agent_turn()

        reflect_mock.assert_awaited_once()
        _, kwargs = reflect_mock.await_args
        self.assertEqual(
            kwargs["recorded_turn"],
            {
                "user_prompt": "inspect",
                "assistant_text": "done",
                "tool_calls": [],
                "tool_results": [],
            },
        )
        persist_harness.assert_called_once()
        persist_session.assert_any_call(reason="assistant_turn_done")


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

        self.assertEqual(profile, (128, 32, True, True, True))

    def test_startup_profile_uses_llama_defaults_when_memory_is_not_fragmented(self) -> None:
        profile = LlamaServerClient._startup_profile_for_lfb(None)

        self.assertEqual(profile, (2048, 512, False, False, False))


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

    async def test_start_emits_startup_diagnostics(self) -> None:
        events: list[tuple[str, dict[str, object]]] = []
        client = LlamaServerClient(
            model="model.gguf",
            device="cuda",
            gpu_layers=99,
            context_window_tokens=4096,
            diagnostics_hook=lambda event_type, data: events.append((event_type, data)),
        )
        start_once = AsyncMock()

        with patch.object(client, "_stop_server", AsyncMock()), patch.object(
            client, "_cleanup_stale_inference_processes", AsyncMock()
        ), patch.object(client, "_prepare_memory_for_launch", AsyncMock(return_value=4.0)), patch.object(
            client, "_start_once", start_once
        ), patch.object(
            client, "_ensure_jetson_clocks_sudoers", return_value=None
        ), patch.object(
            client, "_maximize_gpu_clocks", return_value=None
        ), patch.object(
            client, "_is_jetson_platform", return_value=True
        ), patch.object(
            client, "_memory_snapshot", return_value={"memavailable_mb": 8192.0, "largest_free_block_mb": 32.0}
        ), patch(
            "src.llama_server._find_llama_server", return_value="/usr/bin/llama-server"
        ):
            await client.start()

        self.assertTrue(any(name == "runtime_llama_starting" for name, _ in events))
        starting = next(data for name, data in events if name == "runtime_llama_starting")
        self.assertEqual(starting["requested_ctx"], 4096)
        self.assertEqual(starting["requested_ngl"], 99)
        self.assertEqual(starting["memavailable_mb"], 8192.0)

    async def test_start_passes_moe_options_to_launch(self) -> None:
        client = LlamaServerClient(
            model="model.gguf",
            device="cuda",
            gpu_layers=99,
            cpu_moe=True,
            n_cpu_moe=12,
        )
        start_once = AsyncMock()

        with patch.object(client, "_stop_server", AsyncMock()), patch.object(
            client, "_cleanup_stale_inference_processes", AsyncMock()
        ), patch.object(client, "_prepare_memory_for_launch", AsyncMock(return_value=None)), patch.object(
            client, "_start_once", start_once
        ), patch.object(
            client, "_ensure_jetson_clocks_sudoers", return_value=None
        ), patch.object(
            client, "_maximize_gpu_clocks", return_value=None
        ), patch.object(
            client, "_is_jetson_platform", return_value=False
        ), patch(
            "src.llama_server._find_llama_server", return_value="/usr/bin/llama-server"
        ):
            await client.start()

        kwargs = start_once.await_args.kwargs
        self.assertTrue(kwargs["cpu_moe"])
        self.assertEqual(kwargs["n_cpu_moe"], 12)

    async def test_start_once_adds_moe_cpu_flags_to_command(self) -> None:
        client = LlamaServerClient(model="model.gguf")

        class _EmptyStream:
            async def readline(self) -> bytes:
                await asyncio.sleep(60)
                return b""

        proc = SimpleNamespace(returncode=None, pid=4321, stderr=_EmptyStream())
        health = SimpleNamespace(status_code=200)

        with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=proc)) as create_proc, patch.object(
            client._http, "get", AsyncMock(return_value=health)
        ):
            await client._start_once(
                binary="/usr/bin/llama-server",
                env={},
                ngl=99,
                ctx=4096,
                batch=2048,
                ubatch=512,
                fit_off=False,
                no_warmup=False,
                no_mmap=False,
                cpu_moe=False,
                n_cpu_moe=8,
            )

        cmd = list(create_proc.await_args.args)
        self.assertIn("--n-cpu-moe", cmd)
        self.assertEqual(cmd[cmd.index("--n-cpu-moe") + 1], "8")
        self.assertNotIn("--cpu-moe", cmd)

    async def test_start_once_prefers_all_cpu_moe_over_layer_count(self) -> None:
        client = LlamaServerClient(model="model.gguf")

        class _EmptyStream:
            async def readline(self) -> bytes:
                await asyncio.sleep(60)
                return b""

        proc = SimpleNamespace(returncode=None, pid=4321, stderr=_EmptyStream())
        health = SimpleNamespace(status_code=200)

        with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=proc)) as create_proc, patch.object(
            client._http, "get", AsyncMock(return_value=health)
        ):
            await client._start_once(
                binary="/usr/bin/llama-server",
                env={},
                ngl=99,
                ctx=4096,
                batch=2048,
                ubatch=512,
                fit_off=False,
                no_warmup=False,
                no_mmap=False,
                cpu_moe=True,
                n_cpu_moe=8,
            )

        cmd = list(create_proc.await_args.args)
        self.assertIn("--cpu-moe", cmd)
        self.assertNotIn("--n-cpu-moe", cmd)

    async def test_start_clamps_requested_context_to_model_max_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_path = Path(tmp) / "model.gguf"
            model_path.write_bytes(
                b"GGUF"
                + (3).to_bytes(4, "little")
                + (0).to_bytes(8, "little")
                + (2).to_bytes(8, "little")
                + len(b"general.architecture").to_bytes(8, "little")
                + b"general.architecture"
                + (8).to_bytes(4, "little")
                + len(b"llama").to_bytes(8, "little")
                + b"llama"
                + len(b"llama.context_length").to_bytes(8, "little")
                + b"llama.context_length"
                + (4).to_bytes(4, "little")
                + (8192).to_bytes(4, "little")
            )
            client = LlamaServerClient(
                model=str(model_path),
                device="cuda",
                gpu_layers=99,
                context_window_tokens=65536,
            )
            start_once = AsyncMock()

            with patch.object(client, "_stop_server", AsyncMock()), patch.object(
                client, "_cleanup_stale_inference_processes", AsyncMock()
            ), patch.object(client, "_prepare_memory_for_launch", AsyncMock(return_value=None)), patch.object(
                client, "_start_once", start_once
            ), patch.object(
                client, "_ensure_jetson_clocks_sudoers", return_value=None
            ), patch.object(
                client, "_maximize_gpu_clocks", return_value=None
            ), patch.object(
                client, "_is_jetson_platform", return_value=False
            ), patch(
                "src.llama_server._find_llama_server", return_value="/usr/bin/llama-server"
            ):
                await client.start()

        self.assertEqual(start_once.await_args.kwargs["ctx"], 8192)
        self.assertEqual(client.context_window_tokens, 8192)

    async def test_start_once_emits_failure_diagnostics_with_stderr_tail(self) -> None:
        events: list[tuple[str, dict[str, object]]] = []
        client = LlamaServerClient(
            model="model.gguf",
            diagnostics_hook=lambda event_type, data: events.append((event_type, data)),
        )

        class _FakeStream:
            def __init__(self, lines: list[bytes]) -> None:
                self._lines = list(lines)

            async def readline(self) -> bytes:
                if self._lines:
                    return self._lines.pop(0)
                return b""

        proc = SimpleNamespace(
            returncode=1,
            pid=4321,
            stderr=_FakeStream(
                [
                    b"load_tensors: offloaded 33/33 layers to GPU\n",
                    b"NvMapMemAllocInternalTagged: 1075072515 error 12\n",
                    b"CUDA error: out of memory\n",
                ]
            ),
        )

        with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=proc)), patch.object(
            client, "_memory_snapshot", return_value={"memavailable_mb": 7424.0, "largest_free_block_mb": 16.0}
        ):
            with self.assertRaises(RuntimeError):
                await client._start_once(
                    binary="/usr/bin/llama-server",
                    env={"GGML_CUDA_ENABLE_UNIFIED_MEMORY": "1"},
                    ngl=99,
                    ctx=6144,
                    batch=128,
                    ubatch=32,
                    fit_off=True,
                    no_warmup=False,
                    no_mmap=True,
                    cpu_moe=False,
                    n_cpu_moe=0,
                )

        self.assertTrue(any(name == "runtime_llama_start_failed" for name, _ in events))
        failure = next(data for name, data in events if name == "runtime_llama_start_failed")
        self.assertEqual(failure["ctx"], 6144)
        self.assertEqual(failure["ngl"], 99)
        self.assertIn("CUDA error: out of memory", failure["error"])
        self.assertIn("NvMapMemAllocInternalTagged: 1075072515 error 12", failure["stderr_tail"])

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
