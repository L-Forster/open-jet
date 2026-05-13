from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.skills_hub.lockfile import ensure_hub_layout, load_lockfile, save_lockfile, update_lockfile
from src.skills_hub.model import HubInstallRecord, HubLockfile


class SkillsHubLockfileTests(unittest.TestCase):
    def test_ensure_layout_and_roundtrip_lockfile(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            hub_root = Path(tmp) / ".hub"
            ensured = ensure_hub_layout(hub_root)
            record = HubInstallRecord(name="demo", version="1.0.0", source="local")
            save_lockfile(HubLockfile(skills={"demo": record}), ensured)

            loaded = load_lockfile(ensured)

            self.assertTrue((ensured / "quarantine").is_dir())
            self.assertTrue((ensured / "index-cache").is_dir())
            self.assertEqual(loaded.skills["demo"].version, "1.0.0")

    def test_update_lockfile_adds_record(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            hub_root = Path(tmp) / ".hub"
            updated = update_lockfile(HubInstallRecord(name="demo"), hub_root)

        self.assertIn("demo", updated.skills)


if __name__ == "__main__":
    unittest.main()
