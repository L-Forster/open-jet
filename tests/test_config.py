from __future__ import annotations

import unittest

from src.config import DEFAULT_LOG_DIRECTORY, DEFAULT_SESSION_STATE_PATH, normalize_config


class ConfigNormalizationTests(unittest.TestCase):
    def test_normalize_config_migrates_legacy_root_paths(self) -> None:
        cfg = {
            "logging": {"directory": "session_logs", "enabled": True},
            "state": {"path": "session_state.json", "enabled": True},
        }

        normalized = normalize_config(cfg)

        self.assertEqual(normalized["logging"]["directory"], DEFAULT_LOG_DIRECTORY)
        self.assertEqual(normalized["state"]["path"], DEFAULT_SESSION_STATE_PATH)

    def test_normalize_config_preserves_non_legacy_custom_paths(self) -> None:
        cfg = {
            "logging": {"directory": "custom/logs", "enabled": True},
            "state": {"path": "custom/state.json", "enabled": True},
        }

        normalized = normalize_config(cfg)

        self.assertEqual(normalized["logging"]["directory"], "custom/logs")
        self.assertEqual(normalized["state"]["path"], "custom/state.json")
