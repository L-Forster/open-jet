from __future__ import annotations

import unittest

from src.config import DEFAULT_LOG_DIRECTORY, DEFAULT_SESSION_STATE_PATH, normalize_config, setup_direct_model_catalog


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

    def test_setup_direct_model_catalog_preserves_optional_metadata(self) -> None:
        catalog = setup_direct_model_catalog(
            {
                "setup_recommendations": {
                    "direct_models": [
                        {
                            "max_ram_gb": 12,
                            "label": "Qwen",
                            "filename": "qwen.gguf",
                            "url": "https://huggingface.co/example/qwen-GGUF/resolve/main/qwen.gguf",
                            "model_size_mb": 5816,
                            "active_model_size_mb": 2048,
                            "kv_bytes_per_token": 17408,
                            "unified_memory_only": True,
                            "llama_cpu_moe": True,
                            "llama_n_cpu_moe": 12,
                        }
                    ]
                }
            }
        )

        self.assertEqual(catalog[0]["model_size_mb"], 5816.0)
        self.assertEqual(catalog[0]["active_model_size_mb"], 2048.0)
        self.assertEqual(catalog[0]["kv_bytes_per_token"], 17408.0)
        self.assertTrue(catalog[0]["unified_memory_only"])
        self.assertTrue(catalog[0]["llama_cpu_moe"])
        self.assertEqual(catalog[0]["llama_n_cpu_moe"], 12)
