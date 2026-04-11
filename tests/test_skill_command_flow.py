from __future__ import annotations

import os
import sys
import tempfile
import types
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch


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
        pass

    class SessionLogger:
        pass

    session_logging.BroadcastConfig = BroadcastConfig
    session_logging.SessionLogger = SessionLogger
    sys.modules["src.session_logging"] = session_logging

from src.app import OpenJetApp
from src.commands import SlashCommandHandler


class _FakeLog:
    def __init__(self) -> None:
        self.entries: list[str] = []

    def write(self, value) -> None:
        self.entries.append(str(value))

    def clear(self) -> None:
        self.entries.clear()


class SkillCommandFlowTests(unittest.IsolatedAsyncioTestCase):
    async def test_load_skill_into_context_adds_system_message_without_touching_files_in_play(self) -> None:
        app = object.__new__(OpenJetApp)
        app.agent = SimpleNamespace(messages=[], estimated_context_tokens=lambda: 123)
        app.loaded_files = {}
        app.session_logger = None
        app.harness_state = SimpleNamespace(files_in_play=[])
        app._remaining_prompt_tokens = lambda reserve_next_turn_overhead=True: 800
        persisted: list[str] = []
        app.persist_session_state = lambda reason="": persisted.append(reason)
        app._render_token_counter = lambda: None

        log = _FakeLog()

        result = SimpleNamespace(
            ok=True,
            path="/tmp/python-refactor.md",
            estimated_tokens=42,
            returned_tokens=42,
            token_budget=800,
            truncated=False,
            content="Use this skill when refactoring Python.",
            summary="loaded fully",
            detail="",
            mem_available_mb=4096,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            skill_dir = root / ".openjet" / "skills"
            skill_dir.mkdir(parents=True)
            (skill_dir / "python-refactor.md").write_text("skill body", encoding="utf-8")
            previous_cwd = Path.cwd()
            os.chdir(root)
            try:
                with patch("src.app.load_file", AsyncMock(return_value=result)):
                    ok = await OpenJetApp.load_skill_into_context(app, "python-refactor", log)
            finally:
                os.chdir(previous_cwd)

        self.assertTrue(ok)
        self.assertEqual(app.harness_state.files_in_play, [])
        self.assertIn("skill:python-refactor", app.loaded_files)
        self.assertEqual(app.agent.messages[-1]["role"], "system")
        self.assertIn("User-loaded skill context:", app.agent.messages[-1]["content"])
        self.assertIn("skill_context_loaded", persisted)
        self.assertTrue(any("Loaded skill python-refactor into context" in entry for entry in log.entries))

    async def test_skill_command_pins_and_loads_skill_into_current_chat(self) -> None:
        log = _FakeLog()

        class FakeApp:
            def __init__(self) -> None:
                self.harness_state = SimpleNamespace(preferred_skills=["existing"])
                self.session_logger = None
                self.set_calls: list[list[str]] = []
                self.load_calls: list[str] = []

            def query_one(self, selector: str):
                return log

            def available_harness_skills(self) -> list[str]:
                return ["python-refactor"]

            def set_harness_skills(self, names: list[str]) -> tuple[list[str], list[str]]:
                self.set_calls.append(names)
                return (["python-refactor"], [])

            async def load_skill_into_context(self, name: str, _log) -> bool:
                self.load_calls.append(name)
                return True

            def clear_harness_skills(self) -> None:
                return

        handler = SlashCommandHandler(FakeApp(), banner="banner")

        with tempfile.TemporaryDirectory() as tmp:
            previous_cwd = Path.cwd()
            os.chdir(tmp)
            try:
                handled = await handler.maybe_handle("/skill python-refactor")
            finally:
                os.chdir(previous_cwd)

        self.assertTrue(handled)
        self.assertEqual(handler.app.set_calls, [["python-refactor"]])
        self.assertEqual(handler.app.load_calls, ["python-refactor"])
        joined = "\n".join(log.entries)
        self.assertIn("Selected skills: python-refactor", joined)
        self.assertIn("Loaded into current chat: python-refactor", joined)

    async def test_skill_load_command_loads_without_pinning(self) -> None:
        log = _FakeLog()

        class FakeApp:
            def __init__(self) -> None:
                self.harness_state = SimpleNamespace(preferred_skills=["existing"])
                self.session_logger = None
                self.set_calls: list[list[str]] = []
                self.load_calls: list[str] = []

            def query_one(self, selector: str):
                return log

            def available_harness_skills(self) -> list[str]:
                return ["python-refactor"]

            def set_harness_skills(self, names: list[str]) -> tuple[list[str], list[str]]:
                self.set_calls.append(names)
                return (["python-refactor"], [])

            async def load_skill_into_context(self, name: str, _log) -> bool:
                self.load_calls.append(name)
                return True

            def clear_harness_skills(self) -> None:
                return

        handler = SlashCommandHandler(FakeApp(), banner="banner")

        with tempfile.TemporaryDirectory() as tmp:
            previous_cwd = Path.cwd()
            os.chdir(tmp)
            try:
                handled = await handler.maybe_handle("/skill load python-refactor")
            finally:
                os.chdir(previous_cwd)

        self.assertTrue(handled)
        self.assertEqual(handler.app.set_calls, [])
        self.assertEqual(handler.app.load_calls, ["python-refactor"])
        joined = "\n".join(log.entries)
        self.assertNotIn("Selected skills: python-refactor", joined)
        self.assertIn("Loaded into current chat: python-refactor", joined)

    async def test_plan_and_todo_commands_use_harness_controls(self) -> None:
        log = _FakeLog()

        class FakeApp:
            def __init__(self) -> None:
                self.harness_state = SimpleNamespace(
                    preferred_skills=[],
                    plan_mode=False,
                    plan_approved=True,
                    plan_summary="",
                    todos=[],
                )
                self.session_logger = None
                self.plan_on_calls = 0
                self.plan_approve_calls = 0
                self.todo_clear_calls = 0

            def query_one(self, selector: str):
                return log

            def enter_harness_plan_mode(self) -> None:
                self.plan_on_calls += 1
                self.harness_state.plan_mode = True
                self.harness_state.plan_approved = False

            def approve_harness_plan(self) -> None:
                self.plan_approve_calls += 1
                self.harness_state.plan_mode = False
                self.harness_state.plan_approved = True

            def reject_harness_plan(self) -> None:
                return

            def clear_harness_todos(self) -> None:
                self.todo_clear_calls += 1
                self.harness_state.todos = []

        handler = SlashCommandHandler(FakeApp(), banner="banner")

        handled_plan = await handler.maybe_handle("/plan on")
        handled_todo = await handler.maybe_handle("/todo clear")

        self.assertTrue(handled_plan)
        self.assertTrue(handled_todo)
        self.assertEqual(handler.app.plan_on_calls, 1)
        self.assertEqual(handler.app.todo_clear_calls, 1)

    async def test_plan_approve_requires_recorded_summary(self) -> None:
        log = _FakeLog()

        class FakeApp:
            def __init__(self) -> None:
                self.harness_state = SimpleNamespace(
                    preferred_skills=[],
                    plan_mode=True,
                    plan_approved=False,
                    plan_summary="",
                    todos=[],
                )
                self.session_logger = None
                self.plan_approve_calls = 0

            def query_one(self, selector: str):
                return log

            def approve_harness_plan(self) -> None:
                self.plan_approve_calls += 1

        handler = SlashCommandHandler(FakeApp(), banner="banner")

        handled = await handler.maybe_handle("/plan approve")

        self.assertTrue(handled)
        self.assertEqual(handler.app.plan_approve_calls, 0)
        self.assertIn("requires a recorded plan summary", "\n".join(log.entries))
