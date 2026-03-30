from __future__ import annotations

import sys
import types
import unittest
from contextlib import contextmanager
from unittest.mock import AsyncMock, Mock, patch


def _install_prompt_toolkit_stubs() -> None:
    if "prompt_toolkit" in sys.modules:
        return

    prompt_toolkit = types.ModuleType("prompt_toolkit")

    class PromptSession:
        pass

    prompt_toolkit.PromptSession = PromptSession
    sys.modules["prompt_toolkit"] = prompt_toolkit

    completion = types.ModuleType("prompt_toolkit.completion")

    class Completer:
        pass

    class Completion:
        def __init__(self, *args, **kwargs) -> None:
            self.args = args
            self.kwargs = kwargs

    completion.Completer = Completer
    completion.Completion = Completion
    sys.modules["prompt_toolkit.completion"] = completion

    filters = types.ModuleType("prompt_toolkit.filters")

    class Condition:
        def __init__(self, func) -> None:
            self.func = func

        def __call__(self):
            return self.func()

    filters.Condition = Condition
    sys.modules["prompt_toolkit.filters"] = filters

    formatted_text = types.ModuleType("prompt_toolkit.formatted_text")
    formatted_text.HTML = lambda text: text
    sys.modules["prompt_toolkit.formatted_text"] = formatted_text

    history = types.ModuleType("prompt_toolkit.history")

    class InMemoryHistory:
        pass

    history.InMemoryHistory = InMemoryHistory
    sys.modules["prompt_toolkit.history"] = history

    key_binding = types.ModuleType("prompt_toolkit.key_binding")

    class KeyBindings:
        pass

    key_binding.KeyBindings = KeyBindings
    sys.modules["prompt_toolkit.key_binding"] = key_binding

    patch_stdout = types.ModuleType("prompt_toolkit.patch_stdout")

    @contextmanager
    def _patch_stdout(*args, **kwargs):
        yield

    patch_stdout.patch_stdout = _patch_stdout
    sys.modules["prompt_toolkit.patch_stdout"] = patch_stdout

    shortcuts = types.ModuleType("prompt_toolkit.shortcuts")

    class _Dialog:
        async def run_async(self):
            return None

    shortcuts.radiolist_dialog = lambda *args, **kwargs: _Dialog()
    sys.modules["prompt_toolkit.shortcuts"] = shortcuts

    styles = types.ModuleType("prompt_toolkit.styles")

    class Style:
        @classmethod
        def from_dict(cls, data):
            return data

    styles.Style = Style
    sys.modules["prompt_toolkit.styles"] = styles


_install_prompt_toolkit_stubs()

if "src.session_logging" not in sys.modules:
    session_logging = types.ModuleType("src.session_logging")

    class BroadcastConfig:
        def __init__(self, *args, **kwargs) -> None:
            self.args = args
            self.kwargs = kwargs

    class SessionLogger:
        pass

    session_logging.BroadcastConfig = BroadcastConfig
    session_logging.SessionLogger = SessionLogger
    sys.modules["src.session_logging"] = session_logging

import src.app as app_module
from src.app import OpenJetApp
from src.self_update import RepoUpdateInfo


class StartupUpdateFlowTests(unittest.IsolatedAsyncioTestCase):
    async def test_startup_sequence_checks_for_update_before_runtime_init(self) -> None:
        app = OpenJetApp()
        app.cfg["logging"] = {"enabled": False}
        app.cfg["setup_complete"] = True
        events: list[str] = []

        async def fake_update_check(log) -> None:
            events.append("update")
            self.assertIs(log, app.query_one("#chat-log"))

        with patch.object(
            app,
            "_maybe_prompt_for_startup_update",
            AsyncMock(side_effect=fake_update_check),
        ) as update_check, patch.object(
            app,
            "_has_any_configured_model",
            return_value=True,
        ), patch.object(
            app,
            "_active_model_ref",
            return_value="/models/demo.gguf",
        ), patch.object(
            app,
            "_init_client",
            AsyncMock(side_effect=lambda: events.append("init")),
        ) as init_client, patch.object(
            app,
            "_restore_harness_state",
        ), patch.object(
            app,
            "_render_token_counter",
        ):
            await app._startup_sequence()

        update_check.assert_awaited_once()
        init_client.assert_awaited_once()
        self.assertEqual(events[:2], ["update", "init"])
        self.assertFalse(app._quit_requested)

    async def test_maybe_prompt_for_startup_update_skip_returns_without_quit(self) -> None:
        app = OpenJetApp()
        app.cfg["airgapped"] = False
        app.cfg["logging"] = {"enabled": False}
        update = RepoUpdateInfo(
            remote="origin",
            branch="master",
            local_commit="1111111222222333333444444555555666666777",
            remote_commit="aaaaaaa222222333333444444555556666666777",
        )

        async def immediate_to_thread(func, /, *args, **kwargs):
            return func(*args, **kwargs)

        with patch.object(app_module, "_open_jet_version", return_value="0.3.0"), patch.object(
            app_module.asyncio,
            "to_thread",
            side_effect=immediate_to_thread,
        ), patch.object(
            app_module,
            "available_update",
            return_value=update,
        ) as available_update, patch.object(
            app_module,
            "radiolist_dialog",
        ) as radiolist_dialog, patch.object(
            app_module,
            "install_update",
        ) as install_release:
            radiolist_dialog.return_value.run_async = AsyncMock(return_value="skip")
            await app._maybe_prompt_for_startup_update(app.query_one("#chat-log"))

        available_update.assert_called_once_with(
            current_version="0.3.0",
            timeout_seconds=app._STARTUP_UPDATE_CHECK_TIMEOUT_SECONDS,
        )
        radiolist_dialog.assert_called_once_with(
            title="Update open-jet repo",
            text="Newer commit available: 1111111 -> aaaaaaa",
            values=[
                ("install", "Pull origin/master and restart open-jet"),
                ("skip", "Skip for now"),
            ],
        )
        install_release.assert_not_called()
        self.assertFalse(app._quit_requested)

    async def test_maybe_prompt_for_startup_update_install_updates_and_quits_for_restart(self) -> None:
        app = OpenJetApp()
        app.cfg["airgapped"] = False
        app.cfg["logging"] = {"enabled": False}
        update = RepoUpdateInfo(
            remote="origin",
            branch="master",
            local_commit="1111111222222333333444444555555666666777",
            remote_commit="aaaaaaa222222333333444444555556666666777",
        )

        async def immediate_to_thread(func, /, *args, **kwargs):
            return func(*args, **kwargs)

        with patch.object(app_module, "_open_jet_version", return_value="0.3.0"), patch.object(
            app_module.asyncio,
            "to_thread",
            side_effect=immediate_to_thread,
        ), patch.object(
            app_module,
            "available_update",
            return_value=update,
        ), patch.object(
            app_module,
            "radiolist_dialog",
        ) as radiolist_dialog, patch.object(
            app_module,
            "install_update",
            return_value="Updated open-jet repo from 1111111 to aaaaaaa.",
        ) as install_release, patch.object(
            app,
            "_restart_process",
        ) as restart_process, patch.object(
            app,
            "_init_client",
            AsyncMock(),
        ) as init_client:
            radiolist_dialog.return_value.run_async = AsyncMock(return_value="install")
            await app._maybe_prompt_for_startup_update(app.query_one("#chat-log"))

        install_release.assert_called_once_with(update, current_version="0.3.0")
        restart_process.assert_called_once_with()
        init_client.assert_not_awaited()
        self.assertFalse(app._quit_requested)
