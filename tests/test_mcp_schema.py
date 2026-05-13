from __future__ import annotations

import unittest

from src.mcp_support.config import parse_mcp_config
from src.mcp_support.schema import mcp_tool_to_spec, sanitize_identifier, sanitize_mcp_tool_name


class MCPSchemaTests(unittest.TestCase):
    def test_sanitize_identifier(self) -> None:
        self.assertEqual(sanitize_identifier("Git Hub!"), "git_hub")
        self.assertEqual(sanitize_mcp_tool_name("Git Hub", "List Issues"), "mcp_git_hub_list_issues")

    def test_schema_conversion_preserves_parameters_and_metadata(self) -> None:
        cfg = parse_mcp_config(
            {
                "mcp": {
                    "enabled": True,
                    "servers": {
                        "github": {
                            "transport": "stdio",
                            "command": "uvx",
                            "confirmation_required": True,
                            "tools": {"tags": {"list_issues": ["read"]}},
                        }
                    },
                }
            },
            strict=True,
        )
        server = cfg.server("github")
        self.assertIsNotNone(server)
        assert server is not None
        spec = mcp_tool_to_spec(
            server,
            {
                "name": "list_issues",
                "description": "List repository issues.",
                "inputSchema": {
                    "type": "object",
                    "properties": {"repo": {"type": "string", "description": "Repository"}},
                    "required": ["repo"],
                },
            },
            generated_name="mcp_github_list_issues",
            executor=lambda args: None,
        )
        self.assertEqual(spec.name, "mcp_github_list_issues")
        self.assertTrue(spec.confirmation_required)
        self.assertEqual(spec.required, ("repo",))
        self.assertIn("mcp", spec.tags)
        self.assertIn("mcp:github", spec.tags)
        self.assertIn("read", spec.tags)
        self.assertEqual(spec.metadata["mcp_server_name"], "github")
        self.assertEqual(spec.metadata["mcp_tool_name"], "list_issues")

    def test_include_exclude_filtering(self) -> None:
        cfg = parse_mcp_config(
            {
                "mcp": {
                    "enabled": True,
                    "servers": {
                        "docs": {
                            "transport": "stdio",
                            "command": "python",
                            "tools": {"include": ["search"], "exclude": ["search"]},
                        }
                    },
                }
            },
            strict=True,
        )
        server = cfg.server("docs")
        self.assertIsNotNone(server)
        assert server is not None
        self.assertTrue(server.allows_tool("search"))
        self.assertFalse(server.allows_tool("fetch"))


if __name__ == "__main__":
    unittest.main()
