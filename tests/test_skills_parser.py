from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.skills.parser import parse_skill_file


class SkillsParserTests(unittest.TestCase):
    def test_standard_skill_parses_agent_skills_frontmatter(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "review-sql"
            root.mkdir()
            skill_path = root / "SKILL.md"
            skill_path.write_text(
                "---\n"
                "name: review-sql\n"
                "description: Review SQL migrations safely.\n"
                "version: 1.2.3\n"
                "author: OpenJet\n"
                "license: Apache-2.0\n"
                "platforms: [linux, macos]\n"
                "required_environment_variables: [DATABASE_URL]\n"
                "allowed-tools: [read_file, grep]\n"
                "metadata:\n"
                "  openjet:\n"
                "    category: review\n"
                "  hermes:\n"
                "    compatible: true\n"
                "---\n"
                "# Review SQL\n"
                "\n"
                "Full instructions.\n",
                encoding="utf-8",
            )

            skill = parse_skill_file(
                skill_path,
                source="project_openjet",
                source_kind="project",
                source_label=".openjet/skills",
                format="standard",
            )

        self.assertEqual(skill.name, "review-sql")
        self.assertEqual(skill.metadata.description, "Review SQL migrations safely.")
        self.assertEqual(skill.metadata.version, "1.2.3")
        self.assertEqual(skill.metadata.required_environment_variables, ("DATABASE_URL",))
        self.assertEqual(skill.metadata.allowed_tools, ("read_file", "grep"))
        self.assertEqual(skill.metadata.metadata_openjet["category"], "review")
        self.assertIn("Full instructions.", skill.content)
        self.assertFalse(skill.has_errors)

    def test_malformed_yaml_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "broken"
            root.mkdir()
            skill_path = root / "SKILL.md"
            skill_path.write_text("---\nname: [broken\n---\nBody\n", encoding="utf-8")

            skill = parse_skill_file(
                skill_path,
                source="project_openjet",
                source_kind="project",
                source_label=".openjet/skills",
                format="standard",
            )

        self.assertTrue(skill.has_errors)
        self.assertTrue(any(diagnostic.code == "malformed_yaml" for diagnostic in skill.diagnostics))


if __name__ == "__main__":
    unittest.main()
