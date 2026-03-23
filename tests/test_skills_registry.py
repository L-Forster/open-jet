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

        self.assertIn(f"skills_dir: {shared_home / 'skills'}", manifest)
        self.assertIn("name: python-refactor", manifest)
        self.assertIn("use: Refactor Python safely", manifest)

    def test_local_project_skills_override_shared_home_for_active_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "project"
            local_skills = root / ".openjet" / "skills"
            local_skills.mkdir(parents=True)
            (local_skills / "local-only.md").write_text(
                "---\ntags:\n  - local\n---\nUse this local skill",
                encoding="utf-8",
            )

            with patch.dict(os.environ, {"OPENJET_HOME": str(Path(tmp) / "shared-openjet")}):
                summaries = skill_summaries(root)
                manifest_path = sync_skills_manifest(root)
                manifest_exists = manifest_path.is_file()

        self.assertEqual([summary.name for summary in summaries], ["local-only"])
        self.assertEqual(manifest_path, root / ".openjet" / "skills.md")
        self.assertTrue(manifest_exists)
