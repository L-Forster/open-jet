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

    def test_turbo_benchmark_command_routes_to_turbo_runner(self) -> None:
        with patch("src.benchmark.run_turbo_benchmark") as run_turbo, patch("src.cli.launch_tui") as launch_tui:
            cli_main(
                [
                    "turbo",
                    "benchmark",
                    "--target-model",
                    "/models/target.gguf",
                    "--draft-model",
                    "/models/draft.gguf",
                    "--backend-path",
                    "/opt/llama-server",
                    "--backend-kind",
                    "lucebox",
                    "--context",
                    "6048",
                    "--baseline-tok-s",
                    "30",
                    "-n",
                    "400",
                ]
            )

        launch_tui.assert_not_called()
        run_turbo.assert_called_once_with(
            thinking_enabled=False,
            n_gen=400,
            target_model="/models/target.gguf",
            draft_model="/models/draft.gguf",
            backend_path="/opt/llama-server",
            backend_kind="lucebox",
            context_size=6048,
            baseline_tok_s=30.0,
        )

    def test_benchmark_turbo_mode_routes_to_turbo_runner(self) -> None:
        with patch("src.benchmark.run_turbo_benchmark") as run_turbo, patch("src.cli.launch_tui") as launch_tui:
            cli_main(["benchmark", "--mode", "turbo"])

        launch_tui.assert_not_called()
        run_turbo.assert_called_once_with(
            thinking_enabled=False,
            n_gen=400,
            target_model=None,
            draft_model=None,
            backend_path=None,
            backend_kind="auto",
            context_size=None,
            baseline_tok_s=None,
        )

    def test_benchmark_thinking_mode_enables_thinking(self) -> None:
        with patch("src.benchmark.run_turbo_benchmark") as run_turbo, patch("src.cli.launch_tui") as launch_tui:
            cli_main(["benchmark", "--mode", "thinking", "-n", "64"])

        launch_tui.assert_not_called()
        self.assertEqual(run_turbo.call_args.kwargs["thinking_enabled"], True)
        self.assertEqual(run_turbo.call_args.kwargs["n_gen"], 64)


if __name__ == "__main__":
    unittest.main()
