from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.tui_launcher import TuiLaunchError, configure_backend_environment, find_tui_binary, platform_tag


class TuiLauncherTests(unittest.TestCase):
    def test_platform_tags(self) -> None:
        with patch("platform.system", return_value="Linux"), patch("platform.machine", return_value="x86_64"):
            self.assertEqual(platform_tag(), "linux-x64")
        with patch("platform.system", return_value="Darwin"), patch("platform.machine", return_value="arm64"):
            self.assertEqual(platform_tag(), "macos-arm64")
        with patch("platform.system", return_value="Windows"), patch("platform.machine", return_value="AMD64"):
            self.assertEqual(platform_tag(), "windows-x64")

    def test_unsupported_platform_has_build_guidance(self) -> None:
        with patch("platform.system", return_value="Plan9"), patch("platform.machine", return_value="mips"):
            with self.assertRaisesRegex(TuiLaunchError, "scripts/build_tui.py"):
                platform_tag()

    def test_environment_override_wins(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            binary = Path(tmp) / ("openjet-tui.exe" if os.name == "nt" else "openjet-tui")
            binary.write_bytes(b"test")
            with patch.dict(os.environ, {"OPENJET_TUI_BINARY": str(binary)}):
                self.assertEqual(find_tui_binary(), binary.resolve())

    def test_wsl_unc_workspace_uses_wsl_backend(self) -> None:
        env: dict[str, str] = {}
        workspace = Path(r"\\wsl.localhost\Ubuntu\home\louis\open-jet")
        with patch("src.tui_launcher.os.name", "nt"), patch.object(Path, "is_file", return_value=True):
            configure_backend_environment(env, cwd=workspace)
        self.assertEqual(env["OPENJET_WSL_DISTRO"], "Ubuntu")
        self.assertEqual(env["OPENJET_WSL_CWD"], "/home/louis/open-jet")
        self.assertEqual(env["OPENJET_PYTHON"], "/home/louis/open-jet/.venv/bin/python")


if __name__ == "__main__":
    unittest.main()
