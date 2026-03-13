from __future__ import annotations

import asyncio
import base64
import tempfile
import unittest
from pathlib import Path

from src.agent import ActionKind, Agent
from src.executor import edit_file
from src.harness import (
    HarnessState,
    active_step,
    advance_step,
    build_turn_context,
    clear_preferred_skills,
    set_preferred_skills,
    split_active_step,
    update_state_after_turn,
    update_state_for_user_message,
)
from src.persistent_memory import build_system_prompt, update_persistent_memory
from src.runtime_limits import MemorySnapshot, estimate_tokens
from src.runtime_protocol import StreamChunk, ToolCall
from src.sdk import OpenJetSession, SDKEventKind


class FakeRuntimeClient:
    def __init__(self, chunks: list[StreamChunk]) -> None:
        self.model = "fake"
        self.context_window_tokens = 4096
        self.gpu_layers = 0
        self._chunks = chunks
        self.last_messages: list[dict] | None = None

    async def start(self) -> None:
        return None

    async def close(self) -> None:
        return None

    async def reset_kv_cache(self) -> None:
        return None

    async def chat_stream(self, messages: list[dict], *, use_tools: bool = True):
        self.last_messages = messages
        for chunk in self._chunks:
            yield chunk


class SequencedRuntimeClient:
    def __init__(self, turns: list[list[StreamChunk]]) -> None:
        self.model = "fake"
        self.context_window_tokens = 4096
        self.gpu_layers = 0
        self._turns = list(turns)
        self.last_messages: list[dict] | None = None

    async def start(self) -> None:
        return None

    async def close(self) -> None:
        return None

    async def reset_kv_cache(self) -> None:
        return None

    async def chat_stream(self, messages: list[dict], *, use_tools: bool = True):
        self.last_messages = messages
        chunks = self._turns.pop(0) if self._turns else []
        for chunk in chunks:
            yield chunk


class HarnessContextTests(unittest.TestCase):
    def test_build_turn_context_shifts_file_context_between_turns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_repo_context_docs(
                root,
                architecture_lines=[
                    "- `src/file_a.py`: owns file A behavior",
                    "- `src/file_b.py`: owns file B behavior",
                ],
            )

            first_state = update_state_for_user_message(
                HarnessState(),
                "Work on file A",
                files=["src/file_a.py"],
            )
            second_state = update_state_for_user_message(
                first_state,
                "Now switch to file B",
                files=["src/file_b.py"],
            )

            first_context = build_turn_context(
                root=root,
                state=first_state,
                current_context_tokens=0,
                effective_window=4096,
                memory_snapshot=MemorySnapshot(total_mb=8192, available_mb=4096, used_percent=50.0),
            )
            second_context = build_turn_context(
                root=root,
                state=second_state,
                current_context_tokens=0,
                effective_window=4096,
                memory_snapshot=MemorySnapshot(total_mb=8192, available_mb=4096, used_percent=50.0),
            )

        first_joined = "\n".join(message["content"] for message in first_context.messages)
        second_joined = "\n".join(message["content"] for message in second_context.messages)
        self.assertIn("FILE CONTEXT: src/file_a.py", first_joined)
        self.assertNotIn("FILE CONTEXT: src/file_b.py", first_joined)
        self.assertIn("FILE CONTEXT: src/file_b.py", second_joined)
        self.assertNotIn("FILE CONTEXT: src/file_a.py", second_joined)
        self.assertIn("file-context:src/file_a.py", first_context.docs_loaded)
        self.assertIn("file-context:src/file_b.py", second_context.docs_loaded)

    def test_build_turn_context_uses_project_docs_for_file_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_repo_context_docs(
                root,
                architecture_lines=["- `src/example.py`: owns example behavior"],
            )

            state = update_state_for_user_message(
                HarnessState(),
                "Implement a change",
                files=["src/example.py"],
            )
            context = build_turn_context(
                root=root,
                state=state,
                current_context_tokens=0,
                effective_window=4096,
                memory_snapshot=MemorySnapshot(total_mb=8192, available_mb=4096, used_percent=50.0),
            )

        joined = "\n".join(message["content"] for message in context.messages)
        self.assertIn("PROJECT CONTEXT SUMMARY", joined)
        self.assertIn("FILE CONTEXT: src/example.py", joined)
        self.assertIn("owns example behavior", joined)
        self.assertIn("file-context:src/example.py", context.docs_loaded)
        self.assertIn("layer2", context.layer_docs)
        self.assertGreaterEqual(context.layer_tokens["layer2"], 1)

    def test_build_turn_context_respects_layered_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_repo_context_docs(
                root,
                architecture_lines=["- `src/example.py`: owns example behavior"],
            )
            state = update_state_for_user_message(HarnessState(), "Implement a change", files=["src/example.py"])

            context = build_turn_context(
                root=root,
                state=state,
                current_context_tokens=0,
                effective_window=4096,
                memory_snapshot=MemorySnapshot(total_mb=8192, available_mb=4096, used_percent=50.0),
                layered_config={"layer2_enabled": False},
            )

        joined = "\n".join(message["content"] for message in context.messages)
        self.assertIn("PROJECT CONTEXT SUMMARY", joined)
        self.assertNotIn("FILE CONTEXT: src/example.py", joined)
        self.assertEqual(context.layer_tokens["layer2"], 0)

    def test_build_turn_context_injects_recent_context_in_layer3(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_repo_context_docs(
                root,
                architecture_lines=["- `src/example.py`: owns example behavior"],
            )
            state = update_state_for_user_message(HarnessState(), "Implement a change", files=["src/example.py"])
            state = update_state_after_turn(
                state,
                tool_events=[
                    {
                        "tool": "shell",
                        "ok": False,
                        "summary": "pytest failed",
                        "target": "pytest tests/test_example.py",
                        "verification": True,
                        "command": "pytest tests/test_example.py",
                    }
                ],
                assistant_text="verification failed",
            )

            context = build_turn_context(
                root=root,
                state=state,
                current_context_tokens=0,
                effective_window=4096,
                memory_snapshot=MemorySnapshot(total_mb=8192, available_mb=4096, used_percent=50.0),
            )

        joined = "\n".join(message["content"] for message in context.messages)
        self.assertIn("RECENT TASK CONTEXT", joined)
        self.assertIn("pytest failed", joined)
        self.assertIn("tests/test_example.py", joined)
        self.assertIn("recent-context", context.docs_loaded)
        self.assertGreater(context.layer_tokens["layer3"], 0)

    def test_build_turn_context_enforces_overall_and_layer_budgets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_repo_context_docs(
                root,
                architecture_lines=["- `src/example.py`: owns example behavior"],
            )
            long_skill = (
                "---\n"
                "tags:\n"
                "  - python\n"
                "mode: code\n"
                "---\n"
                + ("skill detail " * 400)
            )
            (root / ".openjet" / "skills").mkdir(parents=True, exist_ok=True)
            (root / ".openjet" / "skills" / "python-heavy.md").write_text(long_skill, encoding="utf-8")
            state = update_state_for_user_message(
                HarnessState(preferred_skills=["python-heavy"]),
                "Implement a Python change",
                files=["src/example.py"],
            )
            state.last_action = {"type": "read_file", "target": "src/example.py", "summary": "read"}
            state.last_verification = {
                "status": "fail",
                "summary": "pytest failed",
                "command": "pytest tests/test_example.py",
            }

            context = build_turn_context(
                root=root,
                state=state,
                current_context_tokens=2200,
                effective_window=4096,
                memory_snapshot=MemorySnapshot(total_mb=4096, available_mb=900, used_percent=78.0),
            )

        self.assertLessEqual(context.docs_tokens, context.budget.docs_budget + estimate_tokens(context.messages[0]["content"]))
        self.assertLessEqual(context.layer_tokens["layer1"], context.budget.layer1_budget)
        self.assertLessEqual(context.layer_tokens["layer2"], context.budget.layer2_budget)
        self.assertLessEqual(context.layer_tokens["layer3"], context.budget.layer3_budget)
        self.assertNotIn("skill detail skill detail skill detail", "\n".join(message["content"] for message in context.messages))
        self.assertLessEqual(sum(context.layer_tokens.values()), context.budget.docs_budget)

    def test_build_turn_context_loads_preferred_skill_within_budget(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / ".openjet" / "agents").mkdir(parents=True)
            (root / ".openjet" / "skills").mkdir(parents=True)
            (root / ".openjet" / "projects").mkdir(parents=True)

            (root / ".openjet" / "agents" / "base.md").write_text("base guidance", encoding="utf-8")
            (root / ".openjet" / "agents" / "coder.md").write_text("coder guidance", encoding="utf-8")
            (root / ".openjet" / "skills" / "python-refactor.md").write_text(
                "---\ntags:\n  - python\n  - refactor\nmode: code\n---\npreferred refactor skill",
                encoding="utf-8",
            )
            (root / ".openjet" / "skills" / "irrelevant.md").write_text(
                "---\ntags:\n  - docs\nmode: review\n---\nreview docs",
                encoding="utf-8",
            )

            state = update_state_for_user_message(
                HarnessState(preferred_skills=["python-refactor"]),
                "Implement a Python harness change",
                files=["src/harness.py"],
            )
            context = build_turn_context(
                root=root,
                state=state,
                current_context_tokens=0,
                effective_window=8192,
                memory_snapshot=MemorySnapshot(total_mb=8192, available_mb=4096, used_percent=50.0),
            )

            joined = "\n".join(message["content"] for message in context.messages)
            self.assertIn("python-refactor.md", ",".join(context.docs_loaded))
            self.assertIn("preferred refactor skill", joined)
            self.assertNotIn("review docs", joined)
            self.assertLessEqual(context.docs_tokens, context.budget.docs_budget + 64)

    def test_clear_preferred_skills_removes_manual_skill_selection(self) -> None:
        state = HarnessState(preferred_skills=["python-refactor", "write-tests"])
        cleared = clear_preferred_skills(state)
        self.assertEqual(cleared.preferred_skills, [])

    def test_edit_file_applies_search_replace_patch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "demo.py"
            path.write_text("def greet():\n    return 'hi'\n", encoding="utf-8")
            patch = (
                "<<<<<<< SEARCH\n"
                "def greet():\n"
                "    return 'hi'\n"
                "=======\n"
                "def greet():\n"
                "    return 'hello'\n"
                ">>>>>>> REPLACE\n"
            )

            result = asyncio.run(edit_file(str(path), patch=patch, return_result=True))

            self.assertTrue(result.ok)
            self.assertIn("replacement(s) made", result.output)
            self.assertIn("hello", path.read_text(encoding="utf-8"))

    def test_edit_file_rejects_invalid_python_patch_without_writing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "demo.py"
            original = "def greet():\n    return 'hi'\n"
            path.write_text(original, encoding="utf-8")
            patch = (
                "<<<<<<< SEARCH\n"
                "def greet():\n"
                "    return 'hi'\n"
                "=======\n"
                "def greet(:\n"
                "    return 'broken'\n"
                ">>>>>>> REPLACE\n"
            )

            result = asyncio.run(edit_file(str(path), patch=patch, return_result=True))

            self.assertFalse(result.ok)
            self.assertTrue(result.internal_retry)
            self.assertIn("Python AST validation failed", result.output)
            self.assertEqual(path.read_text(encoding="utf-8"), original)

    def _write_repo_context_docs(self, root: Path, *, architecture_lines: list[str]) -> None:
        (root / ".openjet" / "agents").mkdir(parents=True)
        (root / ".openjet" / "projects").mkdir(parents=True)
        (root / "AGENTS.md").write_text(
            "## What This Project Is\n"
            "- offline-first local agent\n\n"
            "## Core Architecture\n"
            + "\n".join(architecture_lines)
            + "\n",
            encoding="utf-8",
        )
        (root / ".openjet" / "agents" / "base.md").write_text("base guidance", encoding="utf-8")
        (root / ".openjet" / "agents" / "coder.md").write_text("coder guidance", encoding="utf-8")


class HarnessStateTests(unittest.TestCase):
    def test_tool_turn_advances_from_inspect_to_change(self) -> None:
        state = update_state_for_user_message(HarnessState(), "Implement a harness change", files=["src/harness.py"])
        self.assertEqual(active_step(state).id, "inspect")

        updated = update_state_after_turn(
            state,
            tool_events=[{"tool": "read_file", "ok": True, "summary": "read", "target": "src/harness.py"}],
            assistant_text="inspected file",
        )

        self.assertEqual(active_step(updated).id, "change")
        self.assertEqual(updated.last_action["type"], "read_file")

    def test_verification_failure_stays_on_step_and_increments_failure_count(self) -> None:
        state = update_state_for_user_message(HarnessState(), "Implement a harness change", files=["src/harness.py"])
        state = update_state_after_turn(
            state,
            tool_events=[{"tool": "read_file", "ok": True, "summary": "read", "target": "src/harness.py"}],
            assistant_text="inspected file",
        )
        self.assertEqual(active_step(state).id, "change")

        failed = update_state_after_turn(
            state,
            tool_events=[
                {
                    "tool": "shell",
                    "ok": False,
                    "summary": "pytest failed",
                    "target": "pytest",
                    "verification": True,
                    "command": "pytest tests/test_harness.py",
                }
            ],
            assistant_text="verification failed",
        )

        self.assertEqual(active_step(failed).id, "change")
        self.assertEqual(failed.failure_count_for_active_step, 1)
        self.assertEqual(failed.last_verification["status"], "fail")

    def test_step_controls_advance_and_split_active_step(self) -> None:
        state = update_state_for_user_message(HarnessState(), "Implement a harness change", files=["src/harness.py"])
        advanced = advance_step(state)
        self.assertEqual(active_step(advanced).id, "change")

        split = split_active_step(advanced)
        self.assertEqual(active_step(split).id, "change_a")
        self.assertEqual(len(split.plan), 5)
        self.assertIn("inspect narrowly", active_step(split).title)

class AgentTurnContextTests(unittest.IsolatedAsyncioTestCase):
    async def test_turn_context_is_sent_to_runtime_but_not_persisted_in_history(self) -> None:
        client = FakeRuntimeClient([StreamChunk(text="done")])
        agent = Agent(client=client, system_prompt="system", context_window_tokens=4096)
        agent.add_user_message("hello")
        agent.set_turn_context([{"role": "system", "content": "temporary harness doc"}])

        events = [event async for event in agent.run_turn()]

        self.assertIsNotNone(client.last_messages)
        self.assertEqual(client.last_messages[0]["role"], "system")
        self.assertIn("system", client.last_messages[0]["content"])
        self.assertIn("temporary harness doc", client.last_messages[0]["content"])
        self.assertEqual(client.last_messages[-1]["content"], "hello")
        self.assertEqual(events[-1].kind, ActionKind.DONE)
        self.assertEqual(agent.messages[-1]["role"], "assistant")
        self.assertNotIn("temporary harness doc", [msg.get("content", "") for msg in agent.messages])

    async def test_tool_call_turn_records_tool_request_and_persistent_follow_up(self) -> None:
        tool_call = ToolCall(name="read_file", arguments={"path": "README.md"})
        client = FakeRuntimeClient([StreamChunk(text="checking", tool_calls=[tool_call])])
        agent = Agent(client=client, system_prompt="system", context_window_tokens=4096)
        agent.add_user_message("inspect README")

        events = [event async for event in agent.run_turn()]

        self.assertEqual([event.kind for event in events], [ActionKind.TEXT, ActionKind.TOOL_REQUEST])
        self.assertEqual(agent.messages[-1]["role"], "assistant")
        self.assertIn("tool_calls", agent.messages[-1])

        agent.complete_tool_call(tool_call, "README contents")
        self.assertEqual(agent.messages[-1]["role"], "tool")
        self.assertEqual(agent.messages[-1]["tool_call_id"], tool_call.id)

    async def test_image_attachment_is_serialized_for_runtime(self) -> None:
        client = FakeRuntimeClient([StreamChunk(text="vision done")])
        agent = Agent(client=client, system_prompt="system", context_window_tokens=4096)
        with tempfile.TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "sample.png"
            image_path.write_bytes(
                base64.b64decode(
                    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aS2kAAAAASUVORK5CYII="
                )
            )
            agent.add_user_message("describe this", image_paths=[str(image_path)])

            events = [event async for event in agent.run_turn()]

        self.assertEqual(events[-1].kind, ActionKind.DONE)
        self.assertIsNotNone(client.last_messages)
        content = client.last_messages[-1]["content"]
        self.assertIsInstance(content, list)
        self.assertEqual(content[0]["type"], "text")
        self.assertIn("Attached image:", content[0]["text"])
        self.assertEqual(content[1]["type"], "image_url")
        self.assertTrue(content[1]["image_url"]["url"].startswith("data:image/png;base64,"))


class SDKSessionTests(unittest.IsolatedAsyncioTestCase):
    async def test_run_collects_text_and_tool_results(self) -> None:
        tool_call = ToolCall(name="read_file", arguments={"path": "README.md"})
        client = SequencedRuntimeClient(
            [
                [StreamChunk(text="checking", tool_calls=[tool_call])],
                [StreamChunk(text=" done")],
            ]
        )
        agent = Agent(client=client, system_prompt="system", context_window_tokens=4096)
        session = OpenJetSession(agent)

        response = await session.run("inspect README")

        self.assertEqual(response.text, "checking done")
        self.assertEqual(len(response.tool_results), 1)
        self.assertEqual(response.tool_results[0].tool_call.name, "read_file")
        self.assertIn("open-jet", response.tool_results[0].output)

    async def test_stream_denies_confirmed_tool_without_handler(self) -> None:
        tool_call = ToolCall(name="shell", arguments={"command": "pwd"})
        client = SequencedRuntimeClient([[StreamChunk(tool_calls=[tool_call])], [StreamChunk(text="done")]])
        agent = Agent(client=client, system_prompt="system", context_window_tokens=4096)
        session = OpenJetSession(agent)

        events = [event async for event in session.stream("run pwd")]

        tool_results = [event.tool_result for event in events if event.kind == SDKEventKind.TOOL_RESULT]
        self.assertEqual(len(tool_results), 1)
        self.assertFalse(tool_results[0].approved)
        self.assertEqual(tool_results[0].output, "User denied this action.")

    async def test_stream_approves_confirmed_tool_with_handler(self) -> None:
        tool_call = ToolCall(name="shell", arguments={"command": "printf sdk"})
        client = SequencedRuntimeClient([[StreamChunk(tool_calls=[tool_call])], [StreamChunk(text="done")]])
        agent = Agent(client=client, system_prompt="system", context_window_tokens=4096)
        session = OpenJetSession(agent, approval_handler=lambda tc: tc.name == "shell")

        events = [event async for event in session.stream("run shell")]

        kinds = [event.kind for event in events]
        self.assertIn(SDKEventKind.TOOL_REQUEST, kinds)
        self.assertIn(SDKEventKind.TOOL_RESULT, kinds)
        tool_result = next(event.tool_result for event in events if event.kind == SDKEventKind.TOOL_RESULT)
        self.assertTrue(tool_result.approved)
        self.assertTrue(tool_result.ok)

    async def test_stream_hides_invalid_edit_retry_and_exposes_successful_retry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "demo.py"
            path.write_text("def greet():\n    return 'hi'\n", encoding="utf-8")
            bad_patch = (
                "<<<<<<< SEARCH\n"
                "def greet():\n"
                "    return 'hi'\n"
                "=======\n"
                "def greet(:\n"
                "    return 'broken'\n"
                ">>>>>>> REPLACE\n"
            )
            good_patch = (
                "<<<<<<< SEARCH\n"
                "def greet():\n"
                "    return 'hi'\n"
                "=======\n"
                "def greet():\n"
                "    return 'hello'\n"
                ">>>>>>> REPLACE\n"
            )
            client = SequencedRuntimeClient(
                [
                    [StreamChunk(tool_calls=[ToolCall(name="edit_file", arguments={"path": str(path), "patch": bad_patch})])],
                    [StreamChunk(tool_calls=[ToolCall(name="edit_file", arguments={"path": str(path), "patch": good_patch})])],
                    [StreamChunk(text="done")],
                ]
            )
            agent = Agent(client=client, system_prompt="system", context_window_tokens=4096)
            session = OpenJetSession(agent, approval_handler=lambda tc: tc.name == "edit_file")

            events = [event async for event in session.stream("fix the file")]
            tool_results = [event.tool_result for event in events if event.kind == SDKEventKind.TOOL_RESULT]
            self.assertEqual(len(tool_results), 1)
            self.assertTrue(tool_results[0].ok)
            self.assertIn("replacement(s) made", tool_results[0].output)
            self.assertEqual(path.read_text(encoding="utf-8"), "def greet():\n    return 'hello'\n")

    async def test_run_accepts_image_paths(self) -> None:
        client = SequencedRuntimeClient([[StreamChunk(text="done")]])
        agent = Agent(client=client, system_prompt="system", context_window_tokens=4096)
        session = OpenJetSession(agent)
        with tempfile.TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "sample.png"
            image_path.write_bytes(
                base64.b64decode(
                    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aS2kAAAAASUVORK5CYII="
                )
            )
            response = await session.run("inspect", image_paths=[str(image_path)])

        self.assertEqual(response.text, "done")
        self.assertIsInstance(client.last_messages[-1]["content"], list)


class PersistentMemoryTests(unittest.IsolatedAsyncioTestCase):
    async def test_build_system_prompt_includes_persistent_memory_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            await update_persistent_memory(root, scope="user", action="replace", content="- prefers concise answers")
            await update_persistent_memory(root, scope="agent", action="replace", content="- repo uses apply_patch")

            prompt = await build_system_prompt("base system", root)

        self.assertIn("Persistent user preferences", prompt)
        self.assertIn("prefers concise answers", prompt)
        self.assertIn("Persistent agent memory", prompt)
        self.assertIn("repo uses apply_patch", prompt)


if __name__ == "__main__":
    unittest.main()
