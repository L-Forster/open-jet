from __future__ import annotations

import unittest
from types import SimpleNamespace

from src.mcp_support.manager import MCPManager, active_mcp_manager
from src.runtime_protocol import ToolCall
from src.tool_executor import execute_tool
from src.tools.registry import get_tool_spec


class _FakeSession:
    def __init__(self, tools: list[object], *, label: str = "") -> None:
        self.tools = tools
        self.label = label
        self.calls: list[tuple[str, dict]] = []
        self.initialized = False

    async def initialize(self) -> None:
        self.initialized = True

    async def list_tools(self):
        return SimpleNamespace(tools=self.tools)

    async def call_tool(self, name: str, *, arguments: dict):
        self.calls.append((name, arguments))
        prefix = f"{self.label}:" if self.label else ""
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text=f"{prefix}{name}:{arguments.get('value')}")],
            isError=False,
        )


class MCPManagerTests(unittest.IsolatedAsyncioTestCase):
    async def test_fake_session_discovery_registers_and_executes_tool(self) -> None:
        session = _FakeSession(
            [
                {
                    "name": "echo",
                    "description": "Echo a value.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {"value": {"type": "string"}},
                        "required": ["value"],
                    },
                }
            ]
        )
        manager = MCPManager.from_config(
            {
                "mcp": {
                    "enabled": True,
                    "servers": {
                        "fake": {
                            "transport": "stdio",
                            "command": "fake-server",
                            "tools": {"include": ["echo"]},
                        }
                    },
                }
            },
            session_factory=lambda server: session,
        )
        try:
            await manager.initialize()
            spec = get_tool_spec("mcp_fake_echo")
            self.assertIsNotNone(spec)
            assert spec is not None
            self.assertTrue(spec.confirmation_required)
            result = await execute_tool(ToolCall(name="mcp_fake_echo", arguments={"value": "hello"}))
            self.assertTrue(result.ok)
            self.assertEqual(result.output, "echo:hello")
            self.assertEqual(session.calls, [("echo", {"value": "hello"})])
        finally:
            await manager.aclose()
        self.assertIsNone(get_tool_spec("mcp_fake_echo"))

    async def test_failed_server_does_not_crash_registration(self) -> None:
        def factory(server):
            raise RuntimeError("boom")

        manager = MCPManager.from_config(
            {
                "mcp": {
                    "enabled": True,
                    "servers": {
                        "bad": {
                            "transport": "stdio",
                            "command": "bad-server",
                        }
                    },
                }
            },
            session_factory=factory,
        )
        await manager.initialize()
        statuses = manager.statuses()
        self.assertEqual(len(statuses), 1)
        self.assertFalse(statuses[0].ok)
        self.assertIn("boom", statuses[0].message)
        await manager.aclose()

    async def test_failed_server_status_redacts_configured_secret_values(self) -> None:
        def factory(server):
            raise RuntimeError("failed with token secret-token")

        manager = MCPManager.from_config(
            {
                "mcp": {
                    "enabled": True,
                    "servers": {
                        "bad": {
                            "transport": "stdio",
                            "command": "bad-server",
                            "env": {"API_TOKEN": "secret-token"},
                        }
                    },
                }
            },
            session_factory=factory,
        )
        await manager.initialize()
        status = manager.statuses()[0]
        self.assertFalse(status.ok)
        self.assertIn("<redacted>", status.message)
        self.assertNotIn("secret-token", status.message)
        await manager.aclose()

    async def test_duplicate_manager_instances_share_registered_tool_until_all_close(self) -> None:
        tools = [
            {
                "name": "echo",
                "description": "Echo a value.",
                "inputSchema": {
                    "type": "object",
                    "properties": {"value": {"type": "string"}},
                    "required": ["value"],
                },
            }
        ]
        first_session = _FakeSession(tools, label="first")
        second_session = _FakeSession(tools, label="second")
        cfg = {
            "mcp": {
                "enabled": True,
                "servers": {
                    "fake": {
                        "transport": "stdio",
                        "command": "fake-server",
                        "tools": {"include": ["echo"]},
                    }
                },
            }
        }
        first = MCPManager.from_config(cfg, session_factory=lambda server: first_session)
        second = MCPManager.from_config(cfg, session_factory=lambda server: second_session)
        try:
            await first.initialize()
            await second.initialize()
            self.assertIsNotNone(get_tool_spec("mcp_fake_echo"))
            with active_mcp_manager(first):
                result = await execute_tool(ToolCall(name="mcp_fake_echo", arguments={"value": "hello"}))
            self.assertTrue(result.ok)
            self.assertEqual(result.output, "first:echo:hello")

            with active_mcp_manager(second):
                result = await execute_tool(ToolCall(name="mcp_fake_echo", arguments={"value": "world"}))
            self.assertTrue(result.ok)
            self.assertEqual(result.output, "second:echo:world")

            ambiguous = await execute_tool(ToolCall(name="mcp_fake_echo", arguments={"value": "ambiguous"}))
            self.assertFalse(ambiguous.ok)
            self.assertEqual(ambiguous.meta["status"], "ambiguous_session")

            await second.aclose()
            self.assertIsNotNone(get_tool_spec("mcp_fake_echo"))
            result = await execute_tool(ToolCall(name="mcp_fake_echo", arguments={"value": "again"}))
            self.assertTrue(result.ok)
            self.assertEqual(result.output, "first:echo:again")
        finally:
            await second.aclose()
            await first.aclose()
        self.assertIsNone(get_tool_spec("mcp_fake_echo"))

    async def test_duplicate_generated_names_fail_server(self) -> None:
        session = _FakeSession(
            [
                {"name": "a b", "description": "", "inputSchema": {"type": "object", "properties": {}}},
                {"name": "a_b", "description": "", "inputSchema": {"type": "object", "properties": {}}},
            ]
        )
        manager = MCPManager.from_config(
            {
                "mcp": {
                    "enabled": True,
                    "servers": {"fake": {"transport": "stdio", "command": "fake-server"}},
                }
            },
            session_factory=lambda server: session,
        )
        await manager.initialize()
        statuses = manager.statuses()
        self.assertEqual(len(statuses), 1)
        self.assertFalse(statuses[0].ok)
        self.assertIn("duplicate generated MCP tool name", statuses[0].message)
        await manager.aclose()


if __name__ == "__main__":
    unittest.main()
