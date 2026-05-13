from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.skills.discovery import discover_skills


def _write_standard(root: Path, name: str, description: str, *, platforms: str = "") -> None:
    skill_root = root / name
    skill_root.mkdir(parents=True)
    platform_block = f"platforms: [{platforms}]\n" if platforms else ""
    (skill_root / "SKILL.md").write_text(
        "---\n"
        f"name: {name}\n"
        f"description: {description}\n"
        f"{platform_block}"
        "---\n"
        f"# {name}\n"
        "\n"
        "Body.\n",
        encoding="utf-8",
    )


class SkillsDiscoveryTests(unittest.TestCase):
    def test_discovers_standard_and_legacy_with_precedence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            project = base / "project"
            home = base / "home"
            bundled = base / "install" / "skills"
            project_openjet = project / ".openjet" / "skills"
            project_agents = project / ".agents" / "skills"
            user_openjet = home / ".openjet" / "skills"
            user_agents = home / ".agents" / "skills"
            bundled.mkdir(parents=True)
            project_openjet.mkdir(parents=True)
            project_agents.mkdir(parents=True)
            user_openjet.mkdir(parents=True)
            user_agents.mkdir(parents=True)

            (bundled / "shared.md").write_text("---\nuse: Bundled shared\n---\nBundled body", encoding="utf-8")
            _write_standard(user_agents, "agent-only", "User agent skill")
            _write_standard(user_openjet, "shared", "User shared")
            _write_standard(project_agents, "project-agent", "Project agent")
            _write_standard(project_openjet, "shared", "Project shared")

            result = discover_skills(project, home=home, bundled_dir=bundled)

        by_name = result.by_name()
        self.assertEqual(sorted(by_name), ["agent-only", "project-agent", "shared"])
        self.assertEqual(by_name["shared"].source, "project_openjet")
        self.assertEqual(by_name["shared"].description, "Project shared")
        self.assertTrue(any(diagnostic.code == "duplicate_shadowed" for diagnostic in result.diagnostics))

    def test_platform_filter_excludes_unsupported_skill(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            project_skills = base / "project" / ".openjet" / "skills"
            project_skills.mkdir(parents=True)
            _write_standard(project_skills, "windows-only", "Windows only", platforms="windows")

            result = discover_skills(base / "project", home=base / "home", bundled_dir=base / "install" / "skills")

        self.assertEqual(result.skills, ())
        self.assertTrue(any(diagnostic.code == "unsupported_platform" for diagnostic in result.diagnostics))


if __name__ == "__main__":
    unittest.main()
