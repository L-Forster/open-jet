from __future__ import annotations

import unittest
from unittest.mock import patch

from src.runtime_limits import estimate_tokens


class RuntimeLimitsTests(unittest.TestCase):
    def test_estimate_tokens_uses_local_gguf_counter_for_llama_cpp_runtime(self) -> None:
        with patch("src.runtime_limits._active_local_gguf_model_ref", return_value="/tmp/model.gguf"), patch(
            "src.runtime_limits._get_local_gguf_token_counter",
            return_value=lambda text: 321,
        ):
            self.assertEqual(estimate_tokens("hello"), 321)
