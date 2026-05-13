from __future__ import annotations

import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.cli import main as cli_main


class SkillsCliTests(unittest.TestCase):
    def test_skill_create_list_view_validate_and_doctor(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            stdout = io.StringIO()
            with patch("src.cli._workflow_root", return_value=root), patch("sys.stdout", stdout):
                cli_main(["skill", "create", "demo-skill"])
            self.assertTrue((root / ".openjet" / "skills" / "demo-skill" / "SKILL.md").is_file())
            self.assertIn("Created skill demo-skill", stdout.getvalue())

            for argv, expected in (
                (["skill", "list"], "demo-skill"),
                (["skill", "view", "demo-skill"], "# demo-skill"),
                (["skill", "validate", "demo-skill"], "valid"),
                (["skill", "doctor"], "Skills discovered:"),
            ):
                with self.subTest(argv=argv):
                    stdout = io.StringIO()
                    with patch("src.cli._workflow_root", return_value=root), patch("sys.stdout", stdout):
                        cli_main(argv)
                    self.assertIn(expected, stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
