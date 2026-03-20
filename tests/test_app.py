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
from src.provisioning import ensure_direct_model
from src.runtime_protocol import StreamChunk
from src.setup import _prompt_choice, _runtime_prompt_options, build_recommended_payload, run_setup_wizard


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


class AppStatusTests(unittest.TestCase):
    def test_prompt_message_includes_generating_status_above_prompt(self) -> None:
        app = OpenJetApp()
        app._assistant_status_kind = "generating"
        app._render_assistant_status()

        prompt = app._prompt_message()

        self.assertIn("Generating...", prompt.value)
        self.assertIn("prompt-splash-block-", prompt.value)
        self.assertIn("open-jet", prompt.value)

    def test_tool_output_line_highlights_shell_commands(self) -> None:
        app = OpenJetApp()

        rendered = app._format_tool_output_line("git status --short")

        self.assertIn("[command]", rendered)

    def test_status_cli_does_not_import_tui_surface(self) -> None:
        sys.modules.pop("src.app", None)
        stdout = io.StringIO()

        with patch("src.cli.load_config", return_value={"runtime": "llama_cpp"}), patch(
            "src.cli.runtime_spec",
            return_value=SimpleNamespace(label="llama.cpp (GGUF)"),
        ), patch("src.cli.active_model_ref", return_value="model.gguf"), patch(
            "src.cli.airgapped_from_cfg",
            return_value=False,
        ), patch("sys.stdout", stdout):
            cli_main(["status"])

        self.assertIn("Runtime: llama.cpp (GGUF) (llama_cpp)", stdout.getvalue())
        self.assertNotIn("src.app", sys.modules)

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


class SetupWizardTests(unittest.IsolatedAsyncioTestCase):
    def test_runtime_prompt_options_note_setup_can_provision_llama_server(self) -> None:
        options = _runtime_prompt_options({}, llama_ready=False)
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

        choices = iter(["manual", "auto", "llama_cpp", "__local__", "__manual__", 4096, 99])
        texts = iter([manual_model, profile_name])

        async def fake_choice(*_args, **_kwargs):
            return next(choices)

        async def fake_text(*_args, **_kwargs):
            return next(texts)

        with patch("src.setup._prompt_choice", side_effect=fake_choice), patch(
            "src.setup._prompt_text", side_effect=fake_text
        ), patch("src.setup.discover_model_files", return_value=[]), patch(
            "src.setup.find_ollama_cli", return_value=None
        ), patch("src.setup.recommended_context_window_tokens", return_value=4096), patch(
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
            if title == "Runtime":
                return "llama_cpp"
            if title == "Model source":
                return "__local__"
            if title == "Local model":
                captured_options.extend(options)
                raise StopSetupWizard()
            raise AssertionError(f"Unexpected setup prompt: {title}")

        with patch("src.setup._prompt_choice", side_effect=fake_choice), patch(
            "src.setup.discover_model_files", return_value=[]
        ), patch("src.setup.find_ollama_cli", return_value=None):
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
            if title == "Runtime":
                return options[kwargs["default_index"]][1]
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
            "src.setup.find_ollama_cli", return_value=None
        ), patch("src.setup.recommended_context_window_tokens", return_value=4096), patch(
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
                    "runtime": "llama_cpp",
                    "llama_model": current_model,
                    "context_window_tokens": 3072,
                    "gpu_layers": 20,
                },
            )

        self.assertIsNotNone(payload)
        self.assertEqual(payload["llama_model"], current_model)
        self.assertEqual(payload["context_window_tokens"], 3072)
        self.assertEqual(payload["gpu_layers"], 20)
        self.assertEqual(text_defaults[0], current_model)
        self.assertIsInstance(captured_defaults["Runtime"], int)
        self.assertIsInstance(captured_defaults["Context window"], int)
        self.assertIsInstance(captured_defaults["GPU layers"], int)

    def test_build_recommended_payload_marks_missing_runtime_and_prefers_direct_model(self) -> None:
        hardware = HardwareInfo(label="CPU-only device", total_ram_gb=8.0, has_cuda=False)
        with patch("src.setup.discover_model_files", return_value=[]), patch(
            "src.setup.find_ollama_cli", return_value=None
        ), patch("src.setup.discover_installed_ollama_models", return_value=[]), patch(
            "src.setup._discover_llama_server", return_value=None
        ), patch("src.setup.recommended_context_window_tokens", return_value=2048), patch(
            "src.setup.recommended_context_window_tokens_from_total", return_value=2048
        ):
            payload = build_recommended_payload(
                hardware_info=hardware,
                recommended_ctx=2048,
                current_cfg={},
            )

        self.assertEqual(payload["runtime"], "llama_cpp")
        self.assertEqual(payload["model_source"], "direct")
        self.assertTrue(payload["setup_missing_runtime"])
        self.assertIn("model_download_url", payload)
        self.assertIn("model_download_path", payload)

    def test_build_recommended_payload_stays_local_first_even_with_api_key_env(self) -> None:
        hardware = HardwareInfo(label="CPU-only device", total_ram_gb=8.0, has_cuda=False)
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key", "OPENROUTER_API_KEY": "router-key"}, clear=False), patch(
            "src.setup.discover_model_files", return_value=[]
        ), patch("src.setup.find_ollama_cli", return_value=None), patch(
            "src.setup.discover_installed_ollama_models", return_value=[]
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

        self.assertEqual(payload["runtime"], "llama_cpp")

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
                "runtime": "llama_cpp",
                "model_source": "ollama",
                "ollama_model": "qwen3:4b",
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

        self.assertEqual(payload["ollama_model"], "qwen3:4b")

    async def test_run_setup_wizard_supports_openai_compatible_runtime(self) -> None:
        console = Mock()
        hardware = HardwareInfo(label="CPU-only device", total_ram_gb=8.0, has_cuda=False)
        profile_name = "Cloud Preset"

        choices = iter(["manual", "auto", "openai_compatible", 4096])
        texts = iter(["gpt-4o-mini", "https://api.openai.com", "OPENAI_API_KEY", profile_name])

        async def fake_choice(*_args, **_kwargs):
            return next(choices)

        async def fake_text(*_args, **_kwargs):
            return next(texts)

        with patch("src.setup._prompt_choice", side_effect=fake_choice), patch(
            "src.setup._prompt_text", side_effect=fake_text
        ), patch("src.setup.recommended_context_window_tokens", return_value=4096), patch(
            "src.setup.recommended_context_window_tokens_from_total", return_value=4096
        ):
            payload = await run_setup_wizard(
                session=None,
                console=console,
                hardware_info=hardware,
                recommended_ctx=4096,
                current_cfg={},
            )

        self.assertIsNotNone(payload)
        self.assertEqual(payload["runtime"], "openai_compatible")
        self.assertEqual(payload["model_source"], "remote")
        self.assertEqual(payload["openai_compatible_model"], "gpt-4o-mini")
        self.assertEqual(payload["openai_compatible_base_url"], "https://api.openai.com")
        self.assertEqual(payload["openai_compatible_api_key_env"], "OPENAI_API_KEY")
        self.assertEqual(payload["gpu_layers"], 0)
        self.assertEqual(payload["model_profile_name"], profile_name)

    async def test_run_setup_wizard_supports_direct_download_local_runtime(self) -> None:
        console = Mock()
        hardware = HardwareInfo(label="CPU-only device", total_ram_gb=8.0, has_cuda=False)
        profile_name = "Downloaded Local Model"

        choices = iter(["manual", "auto", "llama_cpp", "__direct__", 2048, 0])
        texts = iter([profile_name])

        async def fake_choice(*_args, **_kwargs):
            return next(choices)

        async def fake_text(*_args, **_kwargs):
            return next(texts)

        with patch("src.setup._prompt_choice", side_effect=fake_choice), patch(
            "src.setup._prompt_text", side_effect=fake_text
        ), patch("src.setup.discover_model_files", return_value=[]), patch(
            "src.setup.find_ollama_cli", return_value=None
        ), patch("src.setup.recommended_context_window_tokens", return_value=2048), patch(
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
        self.assertEqual(payload["runtime"], "llama_cpp")
        self.assertEqual(payload["model_source"], "direct")
        self.assertTrue(payload["setup_missing_model"])
        self.assertIn("model_download_url", payload)
        self.assertIn("model_download_path", payload)
        self.assertEqual(payload["model_profile_name"], profile_name)


class ProvisioningTests(unittest.IsolatedAsyncioTestCase):
    async def test_ensure_direct_model_downloads_target_file(self) -> None:
        class FakeResponse:
            status_code = 200
            headers = {"Content-Length": "8"}

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

            async def aiter_bytes(self):
                yield b"test"
                yield b"-gguf"

        class FakeClient:
            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

            def stream(self, method, url, headers=None):
                return FakeResponse()

        updates: list[str] = []
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "model.gguf"
            payload = {
                "runtime": "llama_cpp",
                "model_source": "direct",
                "model_download_url": "https://example.invalid/model.gguf",
                "model_download_path": str(target),
                "setup_missing_model": True,
            }

            with patch("src.provisioning.httpx.AsyncClient", return_value=FakeClient()):
                resolved = await ensure_direct_model(
                    payload,
                    log=Mock(),
                    set_status=updates.append,
                    clear_status=lambda: None,
                )
            written = target.read_bytes()

        self.assertEqual(written, b"test-gguf")
        self.assertEqual(resolved["model"], str(target))
        self.assertEqual(resolved["llama_model"], str(target))
        self.assertFalse(resolved["setup_missing_model"])
        self.assertTrue(any(text.startswith("downloading model.gguf") for text in updates))


class AppSetupOrderingTests(unittest.IsolatedAsyncioTestCase):
    async def test_run_setup_command_saves_config_before_materializing_model(self) -> None:
        app = OpenJetApp()
        app.cfg["logging"] = {"enabled": False}
        events: list[str] = []

        with patch.object(app, "_run_setup_wizard", AsyncMock(return_value={"model": "/models/base.gguf"})), patch(
            "src.app.save_config", side_effect=lambda _cfg: events.append("save")
        ), patch.object(
            app,
            "_materialize_setup_model",
            AsyncMock(side_effect=lambda result, _log: events.append("materialize") or {**result, "model": "/models/resolved.gguf"}),
        ), patch.object(app, "_init_client", AsyncMock(side_effect=lambda: events.append("init"))), patch.object(
            app, "_render_token_counter"
        ), patch.object(
            app, "persist_session_state"
        ):
            applied = await app.run_setup_command(app.query_one("#chat-log"))

        self.assertTrue(applied)
        self.assertEqual(events[:3], ["save", "materialize", "save"])
        self.assertEqual(events[3], "init")

    async def test_startup_force_setup_saves_config_before_materializing_model(self) -> None:
        app = OpenJetApp(force_setup=True)
        app.cfg["logging"] = {"enabled": False}
        events: list[str] = []

        with patch.object(app, "_run_setup_wizard", AsyncMock(return_value={"model": "/models/base.gguf"})), patch(
            "src.app.save_config", side_effect=lambda _cfg: events.append("save")
        ), patch.object(
            app,
            "_materialize_setup_model",
            AsyncMock(side_effect=lambda result, _log: events.append("materialize") or {**result, "model": "/models/resolved.gguf"}),
        ), patch.object(app, "_init_client", AsyncMock(side_effect=lambda: events.append("init"))), patch.object(
            app, "_render_token_counter"
        ), patch.object(
            app, "_restore_harness_state"
        ):
            await app._startup_sequence()

        self.assertEqual(events[:3], ["save", "materialize", "save"])
        self.assertEqual(events[3], "init")

    def test_persist_setup_result_keeps_existing_named_profiles(self) -> None:
        app = OpenJetApp()
        app.cfg["active_model_profile"] = "Base"
        app.cfg["model_profiles"] = [
            {
                "name": "Base",
                "runtime": "llama_cpp",
                "model_source": "local",
                "model": "/models/base.gguf",
                "llama_model": "/models/base.gguf",
                "context_window_tokens": 4096,
                "gpu_layers": 99,
            }
        ]

        with patch("src.app.save_config"):
            app._persist_setup_result(
                {
                    "runtime": "llama_cpp",
                    "model_source": "local",
                    "model": "/models/alt.gguf",
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


class ModelCommandTests(unittest.IsolatedAsyncioTestCase):
    async def test_model_command_switches_to_saved_profile(self) -> None:
        app = OpenJetApp()
        app.cfg["model_profiles"] = [
            {
                "name": "alt",
                "runtime": "llama_cpp",
                "model_source": "local",
                "model": "/models/alt.gguf",
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
                "runtime": "llama_cpp",
                "model_source": "local",
                "model": "/models/base.gguf",
                "llama_model": "/models/base.gguf",
                "context_window_tokens": 4096,
                "gpu_layers": 99,
            },
            {
                "name": "alt",
                "runtime": "llama_cpp",
                "model_source": "local",
                "model": "/models/alt.gguf",
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
                "runtime": "llama_cpp",
                "model_source": "local",
                "model": "/models/alt.gguf",
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


class CliCommandTests(unittest.TestCase):
    def test_format_model_profiles_summary_lists_active_profile(self) -> None:
        text = _format_model_profiles_summary(
            {
                "active_model_profile": "base",
                "model_profiles": [
                    {
                        "name": "base",
                        "runtime": "llama_cpp",
                        "model_source": "local",
                        "model": "/models/base.gguf",
                        "llama_model": "/models/base.gguf",
                        "context_window_tokens": 4096,
                        "gpu_layers": 99,
                    }
                ],
            }
        )

        self.assertIn("Active model preset: base", text)
        self.assertIn("- base (active): runtime=llama_cpp", text)

    def test_format_cli_status_reports_runtime_and_airgap(self) -> None:
        text = _format_cli_status(
            {
                "runtime": "llama_cpp",
                "active_model_profile": "base",
                "llama_model": "/models/base.gguf",
                "context_window_tokens": 4096,
                "gpu_layers": 99,
                "airgapped": True,
            }
        )

        self.assertIn("Runtime: Local model: llama.cpp (GGUF) (llama_cpp)", text)
        self.assertIn("Active model preset: base", text)
        self.assertIn("Air-gapped: true", text)

    def test_format_slash_commands_summary_includes_models_alias(self) -> None:
        text = _format_slash_commands_summary()

        self.assertIn("/model: Show or switch saved model presets", text)
        self.assertIn("/models", text)

    def test_main_models_command_prints_saved_presets(self) -> None:
        cfg = {
            "active_model_profile": "base",
            "model_profiles": [
                {
                    "name": "base",
                    "runtime": "llama_cpp",
                    "model_source": "local",
                    "model": "/models/base.gguf",
                    "llama_model": "/models/base.gguf",
                    "context_window_tokens": 4096,
                    "gpu_layers": 99,
                }
            ],
        }

        with patch("src.cli.load_config", return_value=cfg), patch("builtins.print") as printer:
            main(["models"])

        printer.assert_called_once()
        self.assertIn("Active model preset: base", printer.call_args.args[0])

    def test_main_setup_command_launches_app_in_setup_mode(self) -> None:
        with patch("src.cli.launch_tui") as launch_tui:
            main(["setup"])

        launch_tui.assert_called_once_with(force_setup=True)


class AppCondenseTests(unittest.IsolatedAsyncioTestCase):
    async def test_run_agent_turn_stops_when_pressure_remains_after_condense(self) -> None:
        app = OpenJetApp()

        class CondenseLoopAgent:
            messages = [{"role": "system", "content": "system"}, {"role": "user", "content": "hi"}]

            async def run_turn(self):
                yield SimpleNamespace(kind=ActionKind.CONDENSE, text="mem_available=300MB < 512MB")

            async def condense_context(self) -> str:
                return "Context condensed automatically. messages: 2 -> 2, tokens(est): 600 -> 580."

            def resource_pressure_reason(self) -> str | None:
                return "mem_available=300MB < 512MB"

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
