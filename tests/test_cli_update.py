from __future__ import annotations

import io
import sys
import unittest
from unittest.mock import patch

from src.cli import main as cli_main


class CliUpdateTests(unittest.TestCase):
    def test_update_cli_uses_self_update_path_without_importing_tui_surface(self) -> None:
        sys.modules.pop("src.app", None)
        stdout = io.StringIO()

        with patch("src.cli.update_from_latest_release", return_value="Updated open-jet from 0.3.0 to 0.4.0."), patch(
            "src.cli._open_jet_version",
            return_value="0.3.0",
        ), patch("sys.stdout", stdout):
            cli_main(["update"])

        self.assertIn("Updated open-jet from 0.3.0 to 0.4.0.", stdout.getvalue())
        self.assertNotIn("src.app", sys.modules)
