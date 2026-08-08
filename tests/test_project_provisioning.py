from __future__ import annotations

import asyncio
import os
import sys
import tempfile
import unittest
from pathlib import Path
import unittest.mock
from unittest.mock import patch

import yaml

from src import config as config_module
from src.app_paths import find_project_root
from src.config import load_config, save_project_config
from src.embed_catalog import (
    EMBED_MODEL_CATALOG,
    recommend_embed_model,
    resident_mb,
    use_case,
)
from src.project_setup import build_overlay, provision_project, resolve_selection
from src.runtime_registry import require_provisioned_model


class _ProjectTempDirTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._cwd = Path.cwd()
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name).resolve() / "app"
        (self.root / ".git").mkdir(parents=True)

    def tearDown(self) -> None:
        os.chdir(self._cwd)
        self._tmp.cleanup()

    def provision(self, **overlay) -> Path:
        model = self.root / ".openjet" / "models" / "m.gguf"
        model.parent.mkdir(parents=True, exist_ok=True)
        model.write_bytes(b"")
        save_project_config({"llama_model": str(model), **overlay}, self.root)
        return model


class ProjectRootDiscoveryTests(_ProjectTempDirTestCase):
    def test_found_from_a_nested_subdirectory(self) -> None:
        self.provision()
        nested = self.root / "src" / "deep" / "nested"
        nested.mkdir(parents=True)
        os.chdir(nested)
        self.assertEqual(find_project_root(), self.root)

    def test_stops_at_the_repository_root(self) -> None:
        # A provisioned directory above the repo must not leak into this one.
        (self.root.parent / ".openjet").mkdir()
        os.chdir(self.root)
        self.assertIsNone(find_project_root())

    def test_absent_outside_a_provisioned_project(self) -> None:
        os.chdir(self.root)
        self.assertIsNone(find_project_root())


class ProjectOverlayTests(_ProjectTempDirTestCase):
    def test_pin_outranks_the_global_config(self) -> None:
        model = self.provision()
        os.chdir(self.root)
        self.assertEqual(load_config()["llama_model"], str(model))

    def test_overlay_is_limited_to_project_owned_keys(self) -> None:
        save_project_config(
            {"llama_model": "/models/m.gguf", "telemetry": {"enabled": True}, "devices": ["a"]},
            self.root,
        )
        written = yaml.safe_load((self.root / ".openjet" / "config.yaml").read_text())
        self.assertEqual(sorted(written), ["llama_model"])

    def test_global_config_is_untouched_by_provisioning(self) -> None:
        before = config_module.CONFIG_PATH.read_text()
        self.provision()
        self.assertEqual(config_module.CONFIG_PATH.read_text(), before)

    def test_project_metadata_survives_normalization(self) -> None:
        self.provision(project={"model_id": "qwen35-4b-q4km", "use_case": "dialogue"})
        os.chdir(self.root)
        self.assertEqual(load_config()["project"]["use_case"], "dialogue")


class RuntimeGuardTests(_ProjectTempDirTestCase):
    def test_missing_model_fails_rather_than_downloading(self) -> None:
        model = self.provision()
        os.chdir(self.root)
        model.unlink()
        with self.assertRaises(RuntimeError) as caught:
            require_provisioned_model(load_config())
        self.assertIn("openjet project", str(caught.exception))

    def test_unconfigured_model_fails(self) -> None:
        with self.assertRaises(RuntimeError):
            require_provisioned_model({})

    def test_api_runtimes_are_not_subject_to_the_check(self) -> None:
        require_provisioned_model({"runtime": "litellm", "model": "some-model"})


class EmbedSelectionTests(unittest.TestCase):
    def test_picks_the_strongest_model_that_fits(self) -> None:
        row = recommend_embed_model(use_case_id="chat", budget_gb=6.0, context_tokens=4096)
        self.assertEqual(row["id"], "qwen35-9b-q4km")

    def test_a_smaller_budget_steps_down(self) -> None:
        row = recommend_embed_model(use_case_id="chat", budget_gb=4.0, context_tokens=4096)
        self.assertEqual(row["id"], "qwen35-4b-q4km")

    def test_latency_bound_use_case_excludes_slow_models(self) -> None:
        # A generous budget must not buy a model that misses the dialogue latency target.
        row = recommend_embed_model(use_case_id="dialogue", budget_gb=48.0, context_tokens=4096)
        self.assertEqual(row["id"], "qwen35-4b-q4km")

    def test_context_length_is_charged_against_the_budget(self) -> None:
        self.assertGreater(
            resident_mb(EMBED_MODEL_CATALOG[0], 32768),
            resident_mb(EMBED_MODEL_CATALOG[0], 4096),
        )

    def test_quant_floor_applies_to_structured_output(self) -> None:
        for row in EMBED_MODEL_CATALOG:
            if "extract" in row["use_cases"]:
                self.assertIn(row["quant"], {"Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0"})

    def test_no_fit_reports_why(self) -> None:
        with self.assertRaises(RuntimeError) as caught:
            resolve_selection(use_case_id="dialogue", target_id="laptop_8", budget_gb=0.5, context_tokens=4096)
        message = str(caught.exception)
        self.assertIn("No catalog model satisfies", message)
        self.assertIn("Qwen3.5 4B", message)

    def test_unknown_use_case_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            use_case("transcription")


class ProvisionFlowTests(_ProjectTempDirTestCase):
    def test_flow_downloads_into_the_project_and_writes_the_overlay(self) -> None:
        downloads: list[dict] = []

        async def fake_download(setup_result, *, log, set_status, clear_status):
            downloads.append(dict(setup_result))
            target = Path(setup_result["model_download_path"])
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"")
            return {**setup_result, "llama_model": str(target)}

        with patch("src.project_setup.ensure_direct_model", new=fake_download):
            config_path = asyncio.run(
                provision_project(
                    self.root,
                    use_case_id="dialogue",
                    target_id="handheld",
                    budget_gb=4.0,
                    context_tokens=4096,
                )
            )

        self.assertEqual(config_path, self.root / ".openjet" / "config.yaml")
        model_path = Path(downloads[0]["model_download_path"])
        self.assertTrue(model_path.is_file())
        self.assertTrue(model_path.is_relative_to(self.root))

        os.chdir(self.root)
        cfg = load_config()
        self.assertEqual(cfg["llama_model"], str(model_path))
        self.assertEqual(cfg["context_window_tokens"], 4096)
        self.assertEqual(cfg["project"]["use_case"], "dialogue")
        require_provisioned_model(cfg)

    def test_project_directory_is_self_ignoring(self) -> None:
        async def fake_download(setup_result, *, log, set_status, clear_status):
            target = Path(setup_result["model_download_path"])
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"")
            return {**setup_result, "llama_model": str(target)}

        with patch("src.project_setup.ensure_direct_model", new=fake_download):
            asyncio.run(
                provision_project(
                    self.root,
                    use_case_id="chat",
                    target_id="laptop_16",
                    budget_gb=6.0,
                    context_tokens=4096,
                )
            )

        self.assertEqual((self.root / ".openjet" / ".gitignore").read_text(), "*\n")


class OverlayShapeTests(unittest.TestCase):
    def test_overlay_carries_runtime_flags_from_the_selected_row(self) -> None:
        selection = dict(EMBED_MODEL_CATALOG[-1])
        selection.update(use_case="chat", target="server", budget_gb=48.0, context_tokens=8192)
        overlay = build_overlay(selection, Path("/models/m.gguf"))
        self.assertTrue(overlay["llama_mtp"])
        self.assertEqual(overlay["context_window_tokens"], 8192)
        self.assertEqual(overlay["llama_model"], "/models/m.gguf")


if __name__ == "__main__":
    unittest.main()


class GeneratedDocsTests(unittest.TestCase):
    def test_model_docs_are_regenerated_from_the_catalogs(self) -> None:
        # The tables in docs/models.md and README.md are generated. If a catalog row
        # changes, they have to be regenerated in the same commit.
        import subprocess

        repo_root = Path(__file__).resolve().parent.parent
        result = subprocess.run(
            [sys.executable, "scripts/generate_model_docs.py", "--check"],
            cwd=repo_root,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


class InferenceSessionTests(unittest.TestCase):
    def test_every_tool_is_refused(self) -> None:
        # An embedded model must not reach the shell or filesystem regardless of what it
        # generates. create_agent keeps the full tool surface and is unaffected.
        from src.runtime_protocol import ToolCall
        from src.sdk.session import OpenJetSession

        session = OpenJetSession.__new__(OpenJetSession)
        session._allowed_tools = set()
        session.agent = unittest.mock.Mock()

        result = asyncio.run(session._handle_tool_call(ToolCall(name="run_shell", arguments={})))
        self.assertFalse(result.approved)
        self.assertTrue(result.meta["denied"])
        self.assertEqual(result.meta["status"], "disallowed")

    def test_default_sessions_keep_their_tools(self) -> None:
        from src.sdk.session import OpenJetSession

        session = OpenJetSession.__new__(OpenJetSession)
        session._allowed_tools = None
        self.assertIsNone(session._allowed_tools)
