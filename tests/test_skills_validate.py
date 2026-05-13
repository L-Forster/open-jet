from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.skills.validate import create_skill_scaffold, validate_skill


class SkillsValidateTests(unittest.TestCase):
    def test_create_skill_scaffold_creates_valid_agent_skill_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            skill_root = create_skill_scaffold(root, "my-skill")
            result = validate_skill(root, "my-skill")

            self.assertTrue((skill_root / "SKILL.md").is_file())
            self.assertTrue((skill_root / "references").is_dir())
            self.assertTrue((skill_root / "scripts").is_dir())
            self.assertTrue((skill_root / "assets").is_dir())
            self.assertTrue(result.ok)

    def test_create_rejects_invalid_name(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                create_skill_scaffold(Path(tmp), "../bad")

    def test_validate_reports_missing_description(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            skill_root = root / ".openjet" / "skills" / "no-description"
            skill_root.mkdir(parents=True)
            (skill_root / "SKILL.md").write_text("---\nname: no-description\n---\nBody\n", encoding="utf-8")

            result = validate_skill(root, "no-description")

        self.assertTrue(result.ok)
        self.assertTrue(any(diagnostic.code == "missing_description" for diagnostic in result.diagnostics))


if __name__ == "__main__":
    unittest.main()
