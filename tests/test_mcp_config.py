from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.mcp_support.config import (
    add_stdio_server_config,
    load_mcp_config_sources,
    load_project_mcp_config,
    parse_mcp_config,
    remove_server_config,
    save_project_mcp_config,
)
from src.mcp_support.redaction import redact_mapping


class MCPConfigTests(unittest.TestCase):
    def test_missing_config_is_disabled(self) -> None:
        cfg = parse_mcp_config({})
        self.assertFalse(cfg.enabled)
        self.assertEqual(cfg.servers, ())

    def test_stdio_config_validation_and_defaults(self) -> None:
        cfg = parse_mcp_config(
            {
                "mcp": {
                    "enabled": True,
                    "servers": {
                        "github": {
                            "transport": "stdio",
                            "command": "uvx",
                            "args": ["mcp-server-github"],
                        }
                    },
                }
            },
            strict=True,
        )
        self.assertTrue(cfg.enabled)
        server = cfg.server("github")
        self.assertIsNotNone(server)
        assert server is not None
        self.assertTrue(server.enabled)
        self.assertEqual(server.transport, "stdio")
        self.assertTrue(server.confirmation_required)
        self.assertEqual(server.args, ("mcp-server-github",))

    def test_http_config_validation(self) -> None:
        cfg = parse_mcp_config(
            {
                "mcp": {
                    "enabled": True,
                    "servers": {
                        "docs": {
                            "transport": "http",
                            "url": "http://127.0.0.1:8000/mcp",
                            "headers": {"Authorization": "Bearer ${DOCS_TOKEN}"},
                        }
                    },
                }
            },
            strict=True,
        )
        server = cfg.server("docs")
        self.assertIsNotNone(server)
        assert server is not None
        self.assertEqual(server.transport, "streamable_http")
        self.assertEqual(server.url, "http://127.0.0.1:8000/mcp")

    def test_env_expansion_uses_current_environment(self) -> None:
        with patch.dict(os.environ, {"GITHUB_PERSONAL_ACCESS_TOKEN": "secret-token"}, clear=False):
            cfg = parse_mcp_config(
                {
                    "mcp": {
                        "enabled": True,
                        "servers": {
                            "github": {
                                "transport": "stdio",
                                "command": "uvx",
                                "env": {
                                    "GITHUB_PERSONAL_ACCESS_TOKEN": "${GITHUB_PERSONAL_ACCESS_TOKEN}",
                                    "PLAIN": "value",
                                },
                            }
                        },
                    }
                },
                strict=True,
            )
        server = cfg.server("github")
        self.assertIsNotNone(server)
        assert server is not None
        self.assertEqual(server.env["GITHUB_PERSONAL_ACCESS_TOKEN"], "secret-token")
        self.assertEqual(server.env["PLAIN"], "value")

    def test_secret_redaction(self) -> None:
        self.assertEqual(
            redact_mapping({"GITHUB_PERSONAL_ACCESS_TOKEN": "secret-token", "MODE": "read"}),
            {"GITHUB_PERSONAL_ACCESS_TOKEN": "<redacted>", "MODE": "read"},
        )

    def test_add_and_remove_stdio_server_config(self) -> None:
        cfg = add_stdio_server_config({}, "filesystem", ["npx", "-y", "@modelcontextprotocol/server-filesystem", "."])
        parsed = parse_mcp_config(cfg, strict=True)
        self.assertTrue(parsed.enabled)
        self.assertIsNotNone(parsed.server("filesystem"))
        cfg = remove_server_config(cfg, "filesystem")
        parsed = parse_mcp_config(cfg, strict=True)
        self.assertEqual(parsed.servers, ())

    def test_dedicated_mcp_yaml_layers_user_project_and_legacy_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            root = base / "project"
            home = base / "home"
            root.mkdir()
            (home / ".openjet").mkdir(parents=True)
            (root / ".openjet").mkdir()
            (home / ".openjet" / "mcp.yaml").write_text(
                "enabled: true\n"
                "servers:\n"
                "  docs:\n"
                "    transport: http\n"
                "    url: http://127.0.0.1:8000/mcp\n"
                "  shared:\n"
                "    transport: stdio\n"
                "    command: user-shared\n",
                encoding="utf-8",
            )
            (root / ".openjet" / "mcp.yaml").write_text(
                "servers:\n"
                "  shared:\n"
                "    transport: stdio\n"
                "    command: project-shared\n",
                encoding="utf-8",
            )

            cfg = load_mcp_config_sources(
                root=root,
                home=home,
                runtime_cfg={
                    "mcp": {
                        "enabled": False,
                        "servers": {
                            "legacy": {
                                "transport": "stdio",
                                "command": "legacy-server",
                            }
                        },
                    }
                },
            )
            parsed = parse_mcp_config(cfg, strict=True)

        self.assertTrue(parsed.enabled)
        self.assertIsNotNone(parsed.server("legacy"))
        self.assertEqual(parsed.server("shared").command, "project-shared")  # type: ignore[union-attr]
        self.assertEqual(parsed.server("docs").transport, "streamable_http")  # type: ignore[union-attr]

    def test_project_mcp_yaml_roundtrip_uses_openjet_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = add_stdio_server_config({}, "filesystem", ["npx", "server"])

            path = save_project_mcp_config(cfg, root)
            loaded = load_project_mcp_config(root)

        self.assertEqual(path, root / ".openjet" / "mcp.yaml")
        parsed = parse_mcp_config(loaded, strict=True)
        self.assertIsNotNone(parsed.server("filesystem"))

    def test_malformed_project_mcp_yaml_becomes_config_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / ".openjet").mkdir()
            (root / ".openjet" / "mcp.yaml").write_text("servers: [", encoding="utf-8")

            parsed = parse_mcp_config(load_mcp_config_sources(root=root))

        self.assertFalse(parsed.enabled)
        self.assertTrue(parsed.errors)
        self.assertIn("malformed YAML", parsed.errors[0])


if __name__ == "__main__":
    unittest.main()
