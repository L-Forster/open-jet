from __future__ import annotations

import unittest
from pathlib import Path


class SourceTreeTests(unittest.TestCase):
    def test_top_level_src_contains_only_source_files(self) -> None:
        src_root = Path(__file__).resolve().parents[1] / "src"
        generated = sorted(
            path.name
            for path in src_root.iterdir()
            if path.is_file()
            and (
                path.suffix in {".c", ".so", ".pyd"}
                or ".cpython-" in path.name
            )
        )

        self.assertEqual(
            generated,
            [],
            f"Top-level src/ must stay source-only; found generated artifacts: {generated}",
        )
