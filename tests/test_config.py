from __future__ import annotations

import unittest

from src.config import (
    DEFAULT_LOG_DIRECTORY,
    DEFAULT_SESSION_STATE_PATH,
    MANAGED_MODELS_DIR,
    migrate_config_for_current_release,
    normalize_config,
    setup_direct_model_catalog,
)


class ConfigNormalizationTests(unittest.TestCase):
    def test_normalize_config_converts_root_state_path_to_yaml(self) -> None:
        cfg = {
            "logging": {"directory": "session_logs", "enabled": True},
            "state": {"enabled": True},
        }

        normalized = normalize_config(cfg)

        self.assertEqual(normalized["logging"]["directory"], DEFAULT_LOG_DIRECTORY)
        self.assertEqual(normalized["state"]["path"], DEFAULT_SESSION_STATE_PATH)

    def test_normalize_config_converts_custom_json_state_paths_to_yaml(self) -> None:
        cfg = {
            "logging": {"directory": "custom/logs", "enabled": True},
            "state": {"path": "custom/state.json", "enabled": True},
        }

        normalized = normalize_config(cfg)

        self.assertEqual(normalized["logging"]["directory"], "custom/logs")
        self.assertEqual(normalized["state"]["path"], "custom/state.yaml")

    def test_normalize_config_preserves_active_cloud_runtime(self) -> None:
        cfg = {
            "runtime": "openai_codex",
            "model": "gpt-5.5",
            "active_model_profile": "codex",
        }

        normalized = normalize_config(cfg)

        self.assertEqual(normalized["runtime"], "openai_codex")
        self.assertEqual(normalized["model"], "gpt-5.5")

    def test_normalize_config_maps_legacy_qwen_mtp_filename_to_unsloth_asset(self) -> None:
        legacy_model = str(MANAGED_MODELS_DIR / "Qwen3.6-27B-Q4_K_M-mtp.gguf")
        migrated_model = str(MANAGED_MODELS_DIR / "Qwen3.6-27B-Q4_K_M-MTP.gguf")
        cfg = {
            "llama_model": legacy_model,
            "model_profiles": [
                {
                    "name": "mtp",
                    "llama_cpp_ref": "pull/22673/head",
                    "llama_model": legacy_model,
                }
            ],
        }

        normalized = normalize_config(cfg)

        self.assertEqual(normalized["llama_model"], migrated_model)
        self.assertTrue(normalized["llama_mtp"])
        self.assertNotIn("llama_cpp_ref", normalized)
        self.assertEqual(normalized["model_source"], "direct")
        self.assertTrue(normalized["setup_missing_model"])
        self.assertNotIn("setup_update_model", normalized)
        self.assertNotIn("model_update_target", normalized)
        self.assertEqual(
            normalized["model_download_url"],
            "https://huggingface.co/unsloth/Qwen3.6-27B-MTP-GGUF/resolve/main/Qwen3.6-27B-Q4_K_M.gguf?download=true",
        )
        self.assertEqual(normalized["model_profiles"][0]["llama_model"], migrated_model)
        self.assertTrue(normalized["model_profiles"][0]["llama_mtp"])
        self.assertNotIn("llama_cpp_ref", normalized["model_profiles"][0])
        self.assertEqual(normalized["model_profiles"][0]["model_source"], "direct")
        self.assertTrue(normalized["model_profiles"][0]["setup_missing_model"])
        self.assertNotIn("setup_update_model", normalized["model_profiles"][0])
        self.assertNotIn("model_update_target", normalized["model_profiles"][0])

    def test_migrate_config_for_current_release_reports_whether_it_changed_model(self) -> None:
        migrated_model = str(MANAGED_MODELS_DIR / "Qwen3.6-27B-Q4_K_M-MTP.gguf")
        cfg = {"llama_model": str(MANAGED_MODELS_DIR / "Qwen3.6-27B-Q4_K_M-mtp.gguf")}

        self.assertTrue(migrate_config_for_current_release(cfg))
        self.assertEqual(cfg["llama_model"], migrated_model)
        self.assertEqual(cfg["model_download_path"], migrated_model)
        self.assertTrue(cfg["llama_mtp"])
        self.assertEqual(cfg["model_source"], "direct")
        self.assertTrue(cfg["setup_missing_model"])
        self.assertNotIn("setup_update_model", cfg)
        self.assertNotIn("model_update_target", cfg)
        self.assertFalse(migrate_config_for_current_release(cfg))

    def test_migrate_config_for_current_release_updates_already_normalized_qwen_mtp_once(self) -> None:
        source_model = str(MANAGED_MODELS_DIR / "Qwen3.6-27B-Q4_K_M.gguf")
        migrated_model = str(MANAGED_MODELS_DIR / "Qwen3.6-27B-Q4_K_M-MTP.gguf")
        cfg = {
            "llama_model": source_model,
            "model_download_path": source_model,
            "model_download_url": "https://huggingface.co/unsloth/Qwen3.6-27B-MTP-GGUF/resolve/main/Qwen3.6-27B-Q4_K_M.gguf?download=true",
            "llama_mtp": True,
            "setup_missing_model": False,
        }

        self.assertTrue(migrate_config_for_current_release(cfg))
        self.assertEqual(cfg["llama_model"], migrated_model)
        self.assertEqual(cfg["model_download_path"], migrated_model)
        self.assertTrue(cfg["setup_missing_model"])
        self.assertNotIn("setup_update_model", cfg)
        self.assertNotIn("model_update_target", cfg)
        cfg["model_update_applied"] = "qwen36-27b-mtp-unsloth"
        cfg["setup_missing_model"] = False
        self.assertFalse(migrate_config_for_current_release(cfg))

    def test_migrate_config_for_current_release_fixes_stale_applied_marker_with_non_mtp_filename(self) -> None:
        source_model = str(MANAGED_MODELS_DIR / "Qwen3.6-27B-Q4_K_M.gguf")
        migrated_model = str(MANAGED_MODELS_DIR / "Qwen3.6-27B-Q4_K_M-MTP.gguf")
        cfg = {
            "llama_model": source_model,
            "model_download_path": source_model,
            "model_download_url": "https://huggingface.co/unsloth/Qwen3.6-27B-MTP-GGUF/resolve/main/Qwen3.6-27B-Q4_K_M.gguf?download=true",
            "llama_mtp": True,
            "model_update_applied": "qwen36-27b-mtp-unsloth",
            "setup_missing_model": False,
        }

        self.assertTrue(migrate_config_for_current_release(cfg))
        self.assertEqual(cfg["llama_model"], migrated_model)
        self.assertEqual(cfg["model_download_path"], migrated_model)
        self.assertTrue(cfg["setup_missing_model"])
        self.assertNotIn("setup_update_model", cfg)
        self.assertNotIn("model_update_target", cfg)

    def test_migrate_config_for_current_release_updates_legacy_35b_a3b_model(self) -> None:
        source_model = str(MANAGED_MODELS_DIR / "Qwen3.6-35B-A3B-UD-Q3_K_XL.gguf")
        migrated_model = str(MANAGED_MODELS_DIR / "Qwen3.6-35B-A3B-UD-Q3_K_XL-MTP.gguf")
        cfg = {
            "llama_model": source_model,
            "model_download_path": source_model,
            "model_download_url": "https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF/resolve/main/Qwen3.6-35B-A3B-UD-Q3_K_XL.gguf?download=true",
            "setup_missing_model": False,
        }

        self.assertTrue(migrate_config_for_current_release(cfg))
        self.assertEqual(cfg["llama_model"], migrated_model)
        self.assertEqual(cfg["model_download_path"], migrated_model)
        self.assertEqual(
            cfg["model_download_url"],
            "https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF/resolve/main/Qwen3.6-35B-A3B-UD-Q3_K_XL.gguf?download=true",
        )
        self.assertTrue(cfg["llama_mtp"])
        self.assertTrue(cfg["setup_missing_model"])
        self.assertNotIn("setup_update_model", cfg)
        self.assertNotIn("model_update_target", cfg)

    def test_migrate_config_for_current_release_updates_legacy_27b_quant_model(self) -> None:
        source_model = str(MANAGED_MODELS_DIR / "Qwen3.6-27B-UD-IQ2_XXS.gguf")
        migrated_model = str(MANAGED_MODELS_DIR / "Qwen3.6-27B-UD-IQ2_XXS-MTP.gguf")
        cfg = {
            "llama_model": source_model,
            "model_download_path": source_model,
            "model_download_url": "https://huggingface.co/unsloth/Qwen3.6-27B-GGUF/resolve/main/Qwen3.6-27B-UD-IQ2_XXS.gguf?download=true",
            "setup_missing_model": False,
        }

        self.assertTrue(migrate_config_for_current_release(cfg))
        self.assertEqual(cfg["llama_model"], migrated_model)
        self.assertEqual(cfg["model_download_path"], migrated_model)
        self.assertEqual(
            cfg["model_download_url"],
            "https://huggingface.co/unsloth/Qwen3.6-27B-MTP-GGUF/resolve/main/Qwen3.6-27B-UD-IQ2_XXS.gguf?download=true",
        )
        self.assertTrue(cfg["llama_mtp"])
        self.assertTrue(cfg["setup_missing_model"])
        self.assertNotIn("setup_update_model", cfg)
        self.assertNotIn("model_update_target", cfg)

    def test_migrate_config_for_current_release_ignores_manual_qwen_path_outside_managed_models_dir(self) -> None:
        manual_model = "/external/models/Qwen3.6-27B-Q4_K_M.gguf"
        cfg = {
            "model_source": "local",
            "llama_model": manual_model,
            "model_download_url": "https://huggingface.co/unsloth/Qwen3.6-27B-GGUF/resolve/main/Qwen3.6-27B-Q4_K_M.gguf?download=true",
            "setup_missing_model": False,
        }

        self.assertFalse(migrate_config_for_current_release(cfg))
        self.assertEqual(cfg["model_source"], "local")
        self.assertEqual(cfg["llama_model"], manual_model)
        self.assertFalse(cfg["setup_missing_model"])
        self.assertNotIn("model_download_path", cfg)

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
                            "llama_mtp": True,
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
        self.assertNotIn("llama_cpp_ref", catalog[0])
        self.assertTrue(catalog[0]["llama_mtp"])
