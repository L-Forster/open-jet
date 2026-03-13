from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.persistent_memory import (
    _clip_text_to_dynamic_budget,
    build_system_prompt,
    memory_file_path,
    update_persistent_memory,
)
from src.runtime_limits import estimate_tokens

from tests.context_helpers import memory_snapshot


class PersistentMemoryBehaviorTests(unittest.TestCase):
    def test_clip_text_to_dynamic_budget_preserves_suffix_style_truncation(self) -> None:
        text = "\n".join(f"line {idx} persistent memory detail" for idx in range(600))
        with patch("src.persistent_memory.read_memory_snapshot", return_value=memory_snapshot(4096, 64)):
            clipped = _clip_text_to_dynamic_budget(text)

        self.assertTrue(clipped.startswith("...[persistent memory truncated]\n"))
        self.assertIn("line 599 persistent memory detail", clipped)
        token_budget = max(128, 256)
        self.assertLessEqual(estimate_tokens(clipped), token_budget)


class PersistentMemoryFileTests(unittest.IsolatedAsyncioTestCase):
    async def test_memory_file_writing_and_reading_stay_bounded(self) -> None:
        content = "\n".join(f"memory item {idx}" for idx in range(800))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch("src.persistent_memory.read_memory_snapshot", return_value=memory_snapshot(4096, 64)):
                result = await update_persistent_memory(root, scope="user", action="replace", content=content)
                readback = await update_persistent_memory(root, scope="user", action="read")
            stored = memory_file_path(root, "user").read_text(encoding="utf-8")

        self.assertIn("stored_tokens~", result)
        self.assertTrue(stored.startswith("...[persistent memory truncated]\n"))
        self.assertEqual(readback.strip(), stored.strip())

    async def test_build_system_prompt_composes_base_prompt_and_persistent_memory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            await update_persistent_memory(root, scope="user", action="replace", content="- prefers concise answers")
            await update_persistent_memory(root, scope="agent", action="replace", content="- use apply_patch for edits")

            prompt = await build_system_prompt("base system", root)

        self.assertIn("base system", prompt)
        self.assertIn("Persistent user preferences", prompt)
        self.assertIn("prefers concise answers", prompt)
        self.assertIn("Persistent agent memory", prompt)
        self.assertIn("apply_patch", prompt)


if __name__ == "__main__":
    unittest.main()
