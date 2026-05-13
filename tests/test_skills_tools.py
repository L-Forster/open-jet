from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.skills.tools import skill_view, skills_list


class SkillsToolsTests(unittest.TestCase):
    def test_skills_list_returns_compact_metadata_without_body(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            skill_root = root / ".openjet" / "skills" / "python-review"
            skill_root.mkdir(parents=True)
            (skill_root / "SKILL.md").write_text(
                "---\nname: python-review\ndescription: Review Python changes.\n---\nSECRET FULL BODY\n",
                encoding="utf-8",
            )

            payload = skills_list(root=root)

        by_name = {skill["name"]: skill for skill in payload["skills"]}
        self.assertIn("python-review", by_name)
        self.assertNotIn("SECRET FULL BODY", str(payload))

    def test_skill_view_returns_body_and_single_safe_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            skill_root = root / ".openjet" / "skills" / "python-review"
            refs = skill_root / "references"
            refs.mkdir(parents=True)
            (skill_root / "SKILL.md").write_text(
                "---\nname: python-review\ndescription: Review Python changes.\n---\nFull body\n",
                encoding="utf-8",
            )
            (refs / "notes.md").write_text("Reference notes\n", encoding="utf-8")

            body = skill_view("python-review", root=root)
            reference = skill_view("python-review", file_path="references/notes.md", root=root)

        self.assertIn("Full body", body["content"])
        self.assertEqual(reference["content"], "Reference notes\n")

    def test_skill_view_blocks_path_traversal_and_absolute_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            skill_root = root / ".openjet" / "skills" / "python-review"
            skill_root.mkdir(parents=True)
            (skill_root / "SKILL.md").write_text(
                "---\nname: python-review\ndescription: Review Python changes.\n---\nFull body\n",
                encoding="utf-8",
            )
            outside = root / "outside.md"
            outside.write_text("outside", encoding="utf-8")

            with self.assertRaises(ValueError):
                skill_view("python-review", file_path="../outside.md", root=root)
            with self.assertRaises(ValueError):
                skill_view("python-review", file_path=str(outside), root=root)


if __name__ == "__main__":
    unittest.main()
