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

    def test_remote_runtime_modules_and_docs_are_removed(self) -> None:
        root = Path(__file__).resolve().parents[1]
        removed_paths = [
            root / "src" / "openai_compatible.py",
            root / "src" / "ollama_setup.py",
            root / "docs" / "runtimes" / "openai-compatible.md",
            root / "docs" / "runtimes" / "openrouter.md",
        ]

        missing = [str(path.relative_to(root)) for path in removed_paths if path.exists()]
        self.assertEqual(missing, [], f"Remote-runtime legacy files must stay deleted: {missing}")

    def test_user_facing_docs_do_not_reference_removed_runtimes(self) -> None:
        root = Path(__file__).resolve().parents[1]
        docs = [
            root / "README.md",
            root / "docs" / "configuration.md",
            root / "docs" / "installation.md",
            root / "docs" / "quickstart.md",
            root / "docs" / "sdk" / "python-sdk.md",
            root / "docs" / "runtimes" / "llama-cpp.md",
        ]

        banned = ("openai_compatible", "openrouter", "ollama")
        offenders: list[str] = []
        for path in docs:
            text = path.read_text(encoding="utf-8").lower()
            if any(term in text for term in banned):
                offenders.append(str(path.relative_to(root)))

        self.assertEqual(offenders, [], f"User-facing docs must stay local-only: {offenders}")
