import asyncio
import json
import unittest
from unittest import mock

from src import connect

CFG = {"llama_model": "/models/Qwen3.8-27B-Q4_K_M.gguf", "context_window_tokens": 65536}


class ConnectRecipeTests(unittest.TestCase):
    def test_claude_code_uses_the_anthropic_root_url(self) -> None:
        text = connect.connect_recipe("claude-code", CFG, port=9000)
        self.assertIn("export ANTHROPIC_BASE_URL=http://127.0.0.1:9000\n", text)
        self.assertIn("ANTHROPIC_MODEL=Qwen3.8-27B-Q4_K_M\n", text)
        self.assertIn("openjet serve --port 9000", text)

    def test_codex_uses_the_responses_wire_api(self) -> None:
        text = connect.connect_recipe("codex", CFG)
        self.assertIn('wire_api = "responses"', text)
        self.assertIn('base_url = "http://127.0.0.1:8080/v1"', text)
        self.assertIn("model_context_window = 65536", text)

    def test_json_recipes_are_valid_json(self) -> None:
        for client, key in (("opencode", "provider"), ("openclaw", "models")):
            text = connect.connect_recipe(client, CFG)
            start = text.index("  {")
            end = text.rindex("}") + 1
            config = json.loads(text[start:end])
            self.assertIn(key, config)
        openclaw = json.loads(connect.connect_recipe("openclaw", CFG).split(":\n\n", 2)[2])
        self.assertEqual(openclaw["agents"]["defaults"]["model"]["primary"], "openjet/Qwen3.8-27B-Q4_K_M")
        self.assertEqual(openclaw["models"]["providers"]["openjet"]["models"][0]["contextWindow"], 65536)

    def test_unknown_client_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            connect.connect_recipe("cursor", CFG)

    def test_serve_refuses_a_remote_runtime(self) -> None:
        with mock.patch.object(connect, "load_config", return_value={"runtime": "litellm", "model": "openai/gpt"}):
            with self.assertRaises(SystemExit):
                asyncio.run(connect.serve(host="127.0.0.1", port=8080))


if __name__ == "__main__":
    unittest.main()
