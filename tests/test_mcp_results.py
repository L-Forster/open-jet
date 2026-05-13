from __future__ import annotations

import unittest
from types import SimpleNamespace

from src.mcp_support.results import mcp_result_to_tool_execution_result


class MCPResultTests(unittest.TestCase):
    def test_text_result(self) -> None:
        result = mcp_result_to_tool_execution_result(
            SimpleNamespace(content=[SimpleNamespace(type="text", text="hello")], isError=False),
            server_name="fake",
            tool_name="echo",
        )
        self.assertTrue(result.ok)
        self.assertEqual(result.output, "hello")
        self.assertTrue(result.meta["mcp"])

    def test_structured_result(self) -> None:
        result = mcp_result_to_tool_execution_result(
            SimpleNamespace(content=[], structuredContent={"answer": 42}, isError=False),
            server_name="fake",
            tool_name="data",
        )
        self.assertTrue(result.ok)
        self.assertIn('"answer": 42', result.output)
        self.assertEqual(result.meta["structured"], {"answer": 42})

    def test_error_result_sets_ok_false(self) -> None:
        result = mcp_result_to_tool_execution_result(
            SimpleNamespace(content=[SimpleNamespace(type="text", text="bad")], isError=True),
            server_name="fake",
            tool_name="fail",
        )
        self.assertFalse(result.ok)
        self.assertEqual(result.meta["status"], "failed")

    def test_unsupported_content_does_not_crash(self) -> None:
        result = mcp_result_to_tool_execution_result(
            SimpleNamespace(content=[SimpleNamespace(type="image", mimeType="image/png", data="abcd")], isError=False),
            server_name="fake",
            tool_name="image",
        )
        self.assertTrue(result.ok)
        self.assertIn("[image returned: mime=image/png, bytes=4]", result.output)


if __name__ == "__main__":
    unittest.main()
