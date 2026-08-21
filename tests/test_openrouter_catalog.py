from __future__ import annotations

import unittest

from src.openrouter_catalog import (
    catalog_entry_for_model,
    ensure_openrouter_model_profiles,
    featured_openrouter_model,
    openrouter_catalog_ts_path,
    openrouter_model_option_ids,
    openrouter_picker_models,
    render_openrouter_catalog_ts,
    upsert_openrouter_profile,
)
from src.runtime_registry import LITELLM_RUNTIME


class OpenRouterCatalogTests(unittest.TestCase):
    def test_featured_model_is_ox_alpha(self) -> None:
        self.assertEqual(featured_openrouter_model(), "openrouter/stealth/ox-alpha")

    def test_catalog_entry_lookup(self) -> None:
        entry = catalog_entry_for_model("openrouter/stealth/ox-alpha")
        self.assertIsNotNone(entry)
        self.assertTrue(entry["free"])
        self.assertEqual(entry["context_window_tokens"], 1048576)

    def test_ensure_profiles_adds_curated_openrouter_presets(self) -> None:
        cfg: dict[str, object] = {"model_profiles": []}
        added = ensure_openrouter_model_profiles(cfg)
        self.assertTrue(added)
        profiles = cfg["model_profiles"]
        self.assertIsInstance(profiles, list)
        names = {str(profile["name"]) for profile in profiles}
        self.assertIn("ox-alpha", names)
        ox_alpha = next(profile for profile in profiles if profile["name"] == "ox-alpha")
        self.assertEqual(ox_alpha["runtime"], LITELLM_RUNTIME)
        self.assertEqual(ox_alpha["model"], "openrouter/stealth/ox-alpha")

    def test_option_ids_include_free_models(self) -> None:
        options = openrouter_model_option_ids()
        self.assertIn("openrouter/stealth/ox-alpha", options)
        self.assertIn("openrouter/openrouter/free", options)

    def test_picker_models_stay_small_and_include_pricing(self) -> None:
        rows = openrouter_picker_models()
        self.assertLess(len(rows), 20)
        ox = next(row for row in rows if row["id"] == "stealth/ox-alpha")
        self.assertEqual(ox["cost"]["input"], 0)
        priced = next(row for row in rows if row["cost"]["input"] > 0)
        self.assertGreater(priced["cost"]["output"], 0)

    def test_generated_typescript_catalog_matches_python(self) -> None:
        generated = openrouter_catalog_ts_path()
        self.assertTrue(generated.is_file(), f"missing generated catalog: {generated}")
        self.assertEqual(generated.read_text(encoding="utf-8"), render_openrouter_catalog_ts())

    def test_upsert_creates_profile_for_any_openrouter_id(self) -> None:
        cfg: dict[str, object] = {"model_profiles": []}
        profile = upsert_openrouter_profile(cfg, "stealth/ox-alpha")
        self.assertEqual(profile["name"], "ox-alpha")
        self.assertEqual(profile["model"], "openrouter/stealth/ox-alpha")
        self.assertEqual(profile["provider"], "openrouter")


if __name__ == "__main__":
    unittest.main()
