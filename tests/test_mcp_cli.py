from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from src.cli import main as cli_main


class MCPCLITests(unittest.TestCase):
    def test_add_stdio_writes_project_mcp_yaml(self) -> None:
        previous = Path.cwd()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            try:
                os.chdir(root)
                with patch("src.mcp_support.cli.load_config", return_value={}):
                    cli_main(["mcp", "add-stdio", "filesystem", "--", "npx", "server"])
            finally:
                os.chdir(previous)

            path = root / ".openjet" / "mcp.yaml"
            payload = yaml.safe_load(path.read_text(encoding="utf-8"))

        self.assertTrue(payload["enabled"])
        self.assertEqual(payload["servers"]["filesystem"]["command"], "npx")
        self.assertEqual(payload["servers"]["filesystem"]["args"], ["server"])


if __name__ == "__main__":
    unittest.main()
