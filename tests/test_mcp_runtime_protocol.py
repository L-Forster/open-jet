from __future__ import annotations

import unittest

from src.runtime_protocol import TOOLS, TOOL_GUIDELINES_XML, parse_tool_calls, tool_guidelines_xml, tool_schema_token_estimate
from src.tool_executor import ToolExecutionResult
from src.tools.registry import TOOL_REGISTRY, ToolSpec, unregister_tool


class MCPRuntimeProtocolTests(unittest.TestCase):
    def test_runtime_protocol_uses_dynamic_registry_snapshot(self) -> None:
        name = "mcp_test_dynamic"
        before = tool_schema_token_estimate()
        TOOL_REGISTRY.register(
            ToolSpec(
                name=name,
                description="Dynamic MCP test tool.",
                parameters={"count": {"type": "integer", "description": "Count"}},
                required=("count",),
                confirmation_required=True,
                tags=frozenset({"mcp", "mcp:test"}),
                executor=lambda args: ToolExecutionResult(output=str(args), meta={"ok": True}),
            )
        )
        try:
            self.assertIn(name, tool_guidelines_xml())
            self.assertIn(name, str(TOOL_GUIDELINES_XML))
            self.assertIn(name, {tool["function"]["name"] for tool in TOOLS})
            self.assertGreater(tool_schema_token_estimate(), before)
            calls = parse_tool_calls(
                "<tool_call><function=mcp_test_dynamic><parameter=count>3</parameter></function></tool_call>"
            )
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0].name, name)
            self.assertEqual(calls[0].arguments, {"count": 3})
        finally:
            unregister_tool(name)

    def test_todo_write_guidelines_include_nested_item_fields(self) -> None:
        xml = tool_guidelines_xml()
        self.assertIn('name="todo_write"', xml)
        self.assertIn("todos*:array&lt;object{id*,content*,status,kind,files,acceptance}&gt;", xml)

    def test_parse_tool_calls_deserializes_todo_write_todos_array(self) -> None:
        calls = parse_tool_calls(
            '<tool_call><function=todo_write><parameter=todos>'
            '[{"id":"t1","content":"Inspect","status":"in_progress","kind":"inspect"}]'
            "</parameter></function></tool_call>"
        )
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].name, "todo_write")
        self.assertEqual(
            calls[0].arguments["todos"],
            [{"id": "t1", "content": "Inspect", "status": "in_progress", "kind": "inspect"}],
        )


if __name__ == "__main__":
    unittest.main()
