from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.agent import ActionKind, Agent
from src.memory_reflection import reflect_agent_persistent_memory
from src.persistent_memory import (
    _clip_text_to_dynamic_budget,
    append_persistent_memory_bullet,
    build_system_prompt,
    load_persistent_memory,
    memory_file_path,
    update_persistent_memory,
)
from src.peripherals import PeripheralDevice, PeripheralKind, PeripheralTransport
from src.runtime_limits import estimate_tokens
from src.runtime_protocol import StreamChunk, ToolCall
from src.tool_executor import execute_tool

from tests.context_helpers import memory_snapshot


class _FakeRuntimeClient:
    def __init__(self, turns: list[list[StreamChunk]]) -> None:
        self.model = "fake"
        self.context_window_tokens = 4096
        self.gpu_layers = 0
        self._turns = list(turns)
        self.last_messages: list[dict] = []
        self.last_use_tools: bool | None = None

    async def start(self) -> None:
        return None

    async def close(self) -> None:
        return None

    async def reset_kv_cache(self) -> None:
        return None

    async def chat_stream(self, messages: list[dict], *, use_tools: bool = True):
        self.last_messages = list(messages)
        self.last_use_tools = use_tools
        chunks = self._turns.pop(0) if self._turns else []
        for chunk in chunks:
            yield chunk


class PersistentMemoryBehaviorTests(unittest.TestCase):
    def test_clip_text_to_dynamic_budget_preserves_suffix_style_truncation(self) -> None:
        text = "\n".join(f"line {idx} persistent memory detail" for idx in range(600))
        with patch("src.persistent_memory.read_memory_snapshot", return_value=memory_snapshot(4096, 64)):
            clipped = _clip_text_to_dynamic_budget(text)

        self.assertIn("persistent memory truncated", clipped)
        self.assertIn("line 599 persistent memory detail", clipped)
        token_budget = max(128, 256)
        self.assertLessEqual(estimate_tokens(clipped), token_budget)


class PersistentMemoryFileTests(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def _recorded_turn(
        *,
        user_prompt: str,
        assistant_text: str = "",
        tool_calls: list[dict] | None = None,
        tool_results: list[dict] | None = None,
    ) -> dict[str, object]:
        return {
            "user_prompt": user_prompt,
            "assistant_text": assistant_text,
            "tool_calls": list(tool_calls or []),
            "tool_results": list(tool_results or []),
        }

    async def test_memory_file_writing_and_reading_stay_bounded(self) -> None:
        content = "\n".join(f"memory item {idx}" for idx in range(800))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch("src.persistent_memory.read_memory_snapshot", return_value=memory_snapshot(4096, 64)):
                result = await update_persistent_memory(root, scope="user", action="replace", content=content)
                readback = await update_persistent_memory(root, scope="user", action="read")
            stored = memory_file_path(root, "user").read_text(encoding="utf-8")

        self.assertIn("stored_tokens~", result)
        self.assertIn("persistent memory truncated", stored)
        self.assertEqual(readback.strip(), stored.strip())

    async def test_build_system_prompt_composes_base_prompt_and_persistent_memory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            global_root = root / "global-home"
            await update_persistent_memory(root, scope="user", action="replace", content="- prefers concise answers")
            await update_persistent_memory(
                root,
                scope="agent",
                action="replace",
                content="- ssh alias laptop is stable",
                location="global",
                global_root=global_root,
            )
            await update_persistent_memory(root, scope="agent", action="replace", content="- use apply_patch for edits")
            with patch("src.config.load_config", return_value={}):
                prompt = await build_system_prompt("base system", root, global_root=global_root)
            self.assertTrue((root / ".openjet" / "state" / "devices.md").is_file())

        self.assertIn("base system", prompt)
        self.assertIn("Local user memory", prompt)
        self.assertIn("prefers concise answers", prompt)
        self.assertIn("Global agent memory", prompt)
        self.assertIn("ssh alias laptop is stable", prompt)
        self.assertIn("Local agent memory", prompt)
        self.assertIn("apply_patch", prompt)
        self.assertIn(str(root / ".openjet" / "state" / "devices.md"), prompt)

    async def test_build_system_prompt_uses_provided_cfg_for_device_registry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            provided_cfg = {"device_aliases": {"front": "camera:/dev/video0"}}

            def _fake_registry(passed_root, *, cfg=None):
                self.assertEqual(passed_root, root)
                self.assertIs(cfg, provided_cfg)
                target = (root / ".openjet" / "state" / "devices.md").resolve()
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text("# Devices\n", encoding="utf-8")
                return target

            with patch("src.persistent_memory.ensure_devices_registry", side_effect=_fake_registry):
                prompt = await build_system_prompt("base system", root, cfg=provided_cfg)

        self.assertIn(str(root / ".openjet" / "state" / "devices.md"), prompt)

    async def test_build_system_prompt_writes_registry_with_spoofed_devices(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = {"device_aliases": {"deskcam": "camera:/dev/video0"}}
            device = PeripheralDevice(
                id="camera:/dev/video0",
                kind=PeripheralKind.CAMERA,
                transport=PeripheralTransport.V4L2,
                label="Front Camera",
                path="/dev/video0",
            )
            with patch("src.device_sources.discover_peripherals", return_value=[device]):
                prompt = await build_system_prompt("base system", root, cfg=cfg)
            registry = root / ".openjet" / "state" / "devices.md"
            rendered = registry.read_text(encoding="utf-8")

        self.assertIn(str(registry), prompt)
        self.assertIn("## deskcam", rendered)
        self.assertIn("latest_payload_file: `none`", rendered)

    async def test_build_system_prompt_uses_default_base_prompt_when_empty(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch("src.config.load_config", return_value={}):
                prompt = await build_system_prompt("", root)

        self.assertIn("You are OpenJet, a local terminal AI assistant.", prompt)
        self.assertIn("Be concise, direct, and practical.", prompt)

    async def test_build_system_prompt_omits_memory_update_policy_text(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            global_root = root / "global-home"
            await update_persistent_memory(
                root,
                scope="user",
                action="replace",
                content="- keep responses terse",
                location="global",
                global_root=global_root,
            )
            await update_persistent_memory(
                root,
                scope="agent",
                action="replace",
                content="- results are under /srv/runs/latest",
                location="project",
                global_root=global_root,
            )
            prompt = await build_system_prompt("", root, global_root=global_root)

        self.assertIn("Global user memory", prompt)
        self.assertIn("Local agent memory", prompt)
        self.assertIn("keep responses terse", prompt)
        self.assertIn("results are under /srv/runs/latest", prompt)
        self.assertNotIn("Return strict JSON only", prompt)
        self.assertNotIn("Decide whether the latest completed turn should be saved into agent memory.", prompt)
        self.assertNotIn("store at most one bullet", prompt)

    async def test_global_memory_is_shared_across_projects(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            global_root = base / "global-home"
            project_a = base / "project-a"
            project_b = base / "project-b"
            project_a.mkdir()
            project_b.mkdir()

            await update_persistent_memory(
                project_a,
                location="global",
                scope="agent",
                action="replace",
                content="- ssh alias laptop -> louis@192.168.1.8",
                global_root=global_root,
            )
            snapshot = await load_persistent_memory(project_b, global_root=global_root)

        self.assertIn("ssh alias laptop", snapshot.global_agent)
        self.assertEqual(snapshot.project_agent, "")

    async def test_append_dedupes_existing_memory_blocks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            await update_persistent_memory(root, scope="agent", action="replace", content="- use apply_patch for edits")
            await update_persistent_memory(root, scope="agent", action="append", content="- use apply_patch for edits")
            stored = await update_persistent_memory(root, scope="agent", action="read")

        self.assertEqual(stored.count("use apply_patch for edits"), 1)

    async def test_append_memory_bullet_edits_existing_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            await update_persistent_memory(root, scope="agent", action="replace", content="- first memory")

            result = await append_persistent_memory_bullet(
                root,
                scope="agent",
                location="project",
                content="- second memory",
            )
            stored = await update_persistent_memory(root, scope="agent", action="read")

        self.assertIn("Appended to project MEMORY.md", result)
        self.assertIn("first memory", stored)
        self.assertIn("second memory", stored)

    async def test_agent_memory_reflection_updates_project_memory_and_refreshes_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            global_root = root / "global-home"
            client = _FakeRuntimeClient(
                [
                    [
                        StreamChunk(
                            text='{"store":true,"location":"local","scope":"agent","bullet":"- training results live under /srv/runs/latest"}',
                            done=True,
                        ),
                    ],
                ]
            )
            agent = Agent(
                client=client,
                system_prompt="base system",
                base_system_prompt="base system",
                project_root=root,
                prompt_cfg={},
                global_memory_root=global_root,
            )
            result = await reflect_agent_persistent_memory(
                agent,
                recorded_turn=self._recorded_turn(
                    user_prompt="Where are the training results?",
                    assistant_text="Training results are under `/srv/runs/latest`.",
                    tool_calls=[
                        {"tool": "shell", "arguments": {"command": "ls /srv/runs"}, "id": "call_1"}
                    ],
                    tool_results=[
                        {
                            "tool": "shell",
                            "arguments": {"command": "ls /srv/runs"},
                            "id": "call_1",
                            "raw_output": "Training results are under /srv/runs/latest.",
                        }
                    ],
                ),
            )
            stored = await update_persistent_memory(
                root,
                location="local",
                scope="agent",
                action="read",
                global_root=global_root,
            )

        self.assertTrue(result["applied"])
        self.assertIn("/srv/runs/latest", stored)
        self.assertIn("/srv/runs/latest", agent.messages[0]["content"])

    async def test_agent_memory_reflection_prompt_contains_only_agent_memory_and_completed_turn(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            global_root = root / "global-home"
            await update_persistent_memory(
                root,
                scope="user",
                action="replace",
                content="- user-owned preference",
                location="global",
                global_root=global_root,
            )
            await update_persistent_memory(
                root,
                scope="agent",
                action="replace",
                content="- global ssh alias laptop -> louis@192.168.1.8",
                location="global",
                global_root=global_root,
            )
            await update_persistent_memory(
                root,
                scope="agent",
                action="replace",
                content="- local results path /srv/runs/latest",
                location="project",
                global_root=global_root,
            )
            client = _FakeRuntimeClient([[StreamChunk(text='{"store":false}', done=True)]])
            agent = Agent(
                client=client,
                system_prompt="base system",
                base_system_prompt="base system",
                project_root=root,
                prompt_cfg={},
                global_memory_root=global_root,
            )
            await reflect_agent_persistent_memory(
                agent,
                recorded_turn=self._recorded_turn(
                    user_prompt="Where are the training results?",
                    assistant_text="They are under /srv/runs/latest.",
                    tool_calls=[{"tool": "shell", "arguments": {"command": "ls /srv/runs"}}],
                    tool_results=[{"tool": "shell", "arguments": {"command": "ls /srv/runs"}, "raw_output": "latest"}],
                ),
            )

        self.assertIs(client.last_use_tools, False)
        self.assertEqual(len(client.last_messages), 2)
        self.assertEqual(client.last_messages[0]["role"], "system")
        self.assertEqual(client.last_messages[1]["role"], "user")
        prompt = str(client.last_messages[1]["content"])
        self.assertIn("Current global agent memory:", prompt)
        self.assertIn("global ssh alias laptop", prompt)
        self.assertIn("Current local agent memory:", prompt)
        self.assertIn("local results path /srv/runs/latest", prompt)
        self.assertIn("Latest completed turn:", prompt)
        self.assertIn("USER: Where are the training results?", prompt)
        self.assertIn("ASSISTANT: They are under /srv/runs/latest.", prompt)
        self.assertIn("TOOL_CALL: shell command=ls /srv/runs", prompt)
        self.assertIn("TOOL_RESULT: shell command=ls /srv/runs", prompt)
        self.assertNotIn("user-owned preference", prompt)

    async def test_agent_memory_reflection_ignores_user_scope(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            client = _FakeRuntimeClient(
                [
                    [
                        StreamChunk(
                            text='{"store":true,"location":"global","scope":"user","bullet":"- prefer concise answers"}',
                            done=True,
                        ),
                    ],
                ]
            )
            agent = Agent(
                client=client,
                system_prompt="base system",
                base_system_prompt="base system",
                project_root=root,
                prompt_cfg={},
                global_memory_root=root / "global-home",
            )
            result = await reflect_agent_persistent_memory(
                agent,
                recorded_turn=self._recorded_turn(
                    user_prompt="Please keep answers concise from now on.",
                    assistant_text="I will keep answers concise.",
                ),
            )
            stored_user = await update_persistent_memory(
                root,
                scope="user",
                action="read",
                location="global",
                global_root=root / "global-home",
            )
            stored_agent = await update_persistent_memory(
                root,
                scope="agent",
                action="read",
                location="global",
                global_root=root / "global-home",
            )

        self.assertEqual(result["applied"], [])
        self.assertEqual(stored_user, "(empty)")
        self.assertEqual(stored_agent, "(empty)")

    async def test_agent_memory_reflection_skips_invalid_location(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            client = _FakeRuntimeClient(
                [
                    [
                        StreamChunk(
                            text='{"store":true,"location":"nonsense","scope":"agent","bullet":"- laptop ssh alias is louis@192.168.1.8"}',
                            done=True,
                        ),
                    ],
                ]
            )
            agent = Agent(
                client=client,
                system_prompt="base system",
                base_system_prompt="base system",
                project_root=root,
                prompt_cfg={},
                global_memory_root=root / "global-home",
            )
            result = await reflect_agent_persistent_memory(
                agent,
                recorded_turn=self._recorded_turn(
                    user_prompt="Remember the ssh alias if it matters.",
                    assistant_text="The laptop ssh alias is louis@192.168.1.8.",
                ),
            )
            stored = await update_persistent_memory(
                root,
                scope="agent",
                action="read",
                location="global",
                global_root=root / "global-home",
            )

        self.assertEqual(result["applied"], [])
        self.assertEqual(stored, "(empty)")

    async def test_agent_memory_reflection_rejects_empty_bullet(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            client = _FakeRuntimeClient(
                [
                    [
                        StreamChunk(
                            text='{"store":true,"location":"local","scope":"agent","bullet":"   "}',
                            done=True,
                        ),
                    ],
                ]
            )
            agent = Agent(
                client=client,
                system_prompt="base system",
                base_system_prompt="base system",
                project_root=root,
                prompt_cfg={},
            )
            result = await reflect_agent_persistent_memory(
                agent,
                recorded_turn=self._recorded_turn(
                    user_prompt="Where are the training results?",
                    assistant_text="Training results are under /srv/runs/latest.",
                ),
            )
            stored = await update_persistent_memory(root, scope="agent", action="read")

        self.assertEqual(result["applied"], [])
        self.assertEqual(stored, "(empty)")

    async def test_agent_memory_reflection_skips_when_model_returns_no_actions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            client = _FakeRuntimeClient([[StreamChunk(text='{"store":false}', done=True)]])
            agent = Agent(
                client=client,
                system_prompt="base system",
                base_system_prompt="base system",
                project_root=root,
                prompt_cfg={},
            )
            result = await reflect_agent_persistent_memory(
                agent,
                recorded_turn=self._recorded_turn(
                    user_prompt="Where are the training results?",
                    assistant_text="Probably under /srv/runs/latest.",
                ),
            )
            stored = await update_persistent_memory(root, scope="agent", action="read")

        self.assertEqual(result["applied"], [])
        self.assertEqual(stored, "(empty)")

    async def test_memory_tool_reports_skipped_write_as_failure(self) -> None:
        result = await execute_tool(
            ToolCall(
                name="memory",
                arguments={
                    "location": "project",
                    "scope": "agent",
                    "action": "append",
                    "content": "api_key=secret-value",
                },
            )
        )

        self.assertFalse(result.ok)
        self.assertEqual(result.meta.get("status"), "skipped")
        self.assertIn("Skipped append", result.output)


if __name__ == "__main__":
    unittest.main()
