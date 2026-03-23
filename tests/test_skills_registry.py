from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.skills_registry import active_skills_home, render_skills_manifest, skill_summaries, sync_skills_manifest


class SkillsRegistryTests(unittest.TestCase):
    def test_render_skills_manifest_uses_shared_home_when_project_has_no_local_skills(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "project"
            shared_home = Path(tmp) / "shared-openjet"
            (shared_home / "skills").mkdir(parents=True)
            (shared_home / "skills" / "python-refactor.md").write_text(
                "---\ntags:\n  - python\nuse: Refactor Python safely\n---\nfull skill body",
                encoding="utf-8",
            )

            with patch.dict(os.environ, {"OPENJET_HOME": str(shared_home)}):
                self.assertEqual(active_skills_home(root), shared_home)
                manifest = render_skills_manifest(root)

        self.assertIn("global_skills_dir: ~/.openjet/skills", manifest)
        self.assertIn("project_skills_dir: .openjet/skills (absent)", manifest)
        self.assertIn("name: python-refactor", manifest)
        self.assertIn("dir: ~/.openjet/skills", manifest)
        self.assertIn("source: global", manifest)
        self.assertIn("load_name: python-refactor", manifest)
        self.assertIn("use: Refactor Python safely", manifest)
        self.assertNotIn(str(shared_home), manifest)

    def test_project_skills_overlay_global_skills(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "project"
            shared_home = Path(tmp) / "shared-openjet"
            local_skills = root / ".openjet" / "skills"
            shared_skills = shared_home / "skills"
            local_skills.mkdir(parents=True)
            shared_skills.mkdir(parents=True)
            (local_skills / "local-only.md").write_text(
                "---\ntags:\n  - local\n---\nUse this local skill",
                encoding="utf-8",
            )
            (shared_skills / "shared-only.md").write_text(
                "---\ntags:\n  - shared\n---\nUse this shared skill",
                encoding="utf-8",
            )
            (shared_skills / "override-me.md").write_text(
                "---\ntags:\n  - shared\n---\nUse shared override",
                encoding="utf-8",
            )
            (local_skills / "override-me.md").write_text(
                "---\ntags:\n  - project\n---\nUse project override",
                encoding="utf-8",
            )

            with patch.dict(os.environ, {"OPENJET_HOME": str(shared_home)}):
                summaries = skill_summaries(root)
                manifest_path = sync_skills_manifest(root)
                manifest_exists = manifest_path.is_file()
                manifest = render_skills_manifest(root)

        self.assertEqual([summary.name for summary in summaries], ["local-only", "override-me", "shared-only"])
        by_name = {summary.name: summary for summary in summaries}
        self.assertEqual(by_name["local-only"].source, "project")
        self.assertEqual(by_name["shared-only"].source, "global")
        self.assertEqual(by_name["override-me"].source, "project")
        self.assertEqual(by_name["override-me"].use, "Use project override")
        self.assertEqual(manifest_path, shared_home / "skills.md")
        self.assertTrue(manifest_exists)
        self.assertIn("project_skills_dir: .openjet/skills (present)", manifest)
        self.assertIn("merge_policy: project skills overlay global skills with the same name.", manifest)
