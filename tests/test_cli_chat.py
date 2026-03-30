from __future__ import annotations

import io
import unittest
from unittest.mock import AsyncMock, patch

from src.cli import main as cli_main


class CliChatTests(unittest.TestCase):
    def test_chat_with_prompt_uses_sdk_session_path(self) -> None:
        stdout = io.StringIO()

        with patch("src.cli._run_cli_chat_prompt", AsyncMock(return_value="reply")) as run_prompt, patch(
            "src.cli.launch_tui"
        ) as launch_tui, patch("sys.stdout", stdout):
            cli_main(["chat", "hello", "world"])

        run_prompt.assert_awaited_once_with("hello world")
        launch_tui.assert_not_called()
        self.assertEqual(stdout.getvalue().strip(), "reply")

    def test_chat_without_prompt_keeps_tui_behavior(self) -> None:
        with patch("src.cli.launch_tui") as launch_tui, patch("src.cli._run_cli_chat_prompt", AsyncMock()) as run_prompt:
            cli_main(["chat"])

        launch_tui.assert_called_once_with(force_setup=False)
        run_prompt.assert_not_called()


if __name__ == "__main__":
    unittest.main()
