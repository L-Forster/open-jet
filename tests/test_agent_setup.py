from __future__ import annotations

import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

from src import agent_setup
from src.cli import main
from src.hardware import HardwareInfo
from src.provisioning import ensure_direct_model
from src.runtime_limits import MemorySnapshot


class AgentSetupTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.target = self.root / "config.yaml"
        self.target.write_text("original: true\n")
        self.model_dir = self.root / "models"
        self.model_dir.mkdir()
        self.row = {"label": "Test coding model", "filename": "test.gguf",
                    "url": "https://huggingface.co/test/model/resolve/main/test.gguf",
                    "max_ram_gb": 4, "model_size_mb": 1000, "kv_bytes_per_token": 1000, "max_context_tokens": 8192}
        self.direct = {**self.row, "target_path": str(self.model_dir / "test.gguf"), "llama_cpu_moe": False, "llama_n_cpu_moe": 0}
        self.hardware = HardwareInfo("CPU", total_ram_gb=16, has_cuda=False)
        for name, value in (
            ("_config_target", self.target), ("load_config", {"runtime": "llama_cpp", "unrelated": "preserved"}),
            ("inspect_hardware", (self.hardware, {"label": "CPU"})), ("recommend_direct_model", self.direct),
            ("setup_direct_model_catalog", (self.row,)),
        ):
            mocked = patch.object(agent_setup, name, return_value=value)
            mocked.start()
            self.addCleanup(mocked.stop)
        models = patch("src.setup.MODELS_DIR", self.model_dir)
        models.start()
        self.addCleanup(models.stop)

    async def test_dry_run_does_not_provision_verify_or_write(self):
        with patch.object(agent_setup, "provision_setup_artifacts", AsyncMock()) as provision, patch.object(
            agent_setup, "verify_inference", AsyncMock()
        ) as verify:
            result = await agent_setup.run_setup(dry_run=True)
        self.assertEqual(result["status"], "planned")
        self.assertEqual(result["catalog"][0]["id"], "test.gguf")
        self.assertEqual(result["model"], str(self.model_dir / "test.gguf"))
        self.assertEqual(self.target.read_text(), "original: true\n")
        provision.assert_not_awaited()
        verify.assert_not_awaited()

    async def test_mutating_setup_requires_confirmation(self):
        with self.assertRaisesRegex(agent_setup.SetupError, "--yes"):
            await agent_setup.run_setup()

    async def test_success_verifies_before_saving_profile(self):
        async def provision(payload, **kwargs):
            self.assertFalse(payload["setup_complete"])
            self.assertTrue(payload["preserve_model_files"])
            return {**payload, "llama_model": str(self.model_dir / "test.gguf")}

        async def verify(cfg):
            self.assertEqual(self.target.read_text(), "original: true\n")
            self.assertFalse(cfg["setup_complete"])
            return {"passed": True, "backend": "llama.cpp", "context_window_tokens": 2048, "gpu_layers": 0}

        with patch.object(agent_setup, "provision_setup_artifacts", side_effect=provision), patch.object(
            agent_setup, "verify_inference", side_effect=verify
        ):
            result = await agent_setup.run_setup(yes=True, model="test.gguf")
        self.assertEqual(result["status"], "ready")
        saved = agent_setup.yaml.safe_load(self.target.read_text())
        self.assertTrue(saved["setup_complete"])
        self.assertEqual(saved["unrelated"], "preserved")
        self.assertEqual(saved["model_profiles"][0]["llama_model"], str(self.model_dir / "test.gguf"))
        self.assertNotIn("preserve_model_files", saved)

    async def test_failed_verification_keeps_previous_configuration(self):
        async def provision(payload, **kwargs):
            return {**payload, "llama_model": str(self.model_dir / "test.gguf")}

        with patch.object(agent_setup, "provision_setup_artifacts", side_effect=provision), patch.object(
            agent_setup, "verify_inference", AsyncMock(side_effect=RuntimeError("no generated text"))
        ):
            with self.assertRaisesRegex(agent_setup.SetupError, "no generated text") as error:
                await agent_setup.run_setup(yes=True)
        self.assertEqual(error.exception.stage, "verification")
        self.assertEqual(self.target.read_text(), "original: true\n")

    async def test_provisioning_failure_keeps_previous_configuration(self):
        with patch.object(agent_setup, "provision_setup_artifacts", AsyncMock(side_effect=RuntimeError("build failed"))):
            with self.assertRaisesRegex(agent_setup.SetupError, "build failed") as error:
                await agent_setup.run_setup(yes=True)
        self.assertEqual(error.exception.stage, "provisioning")
        self.assertEqual(self.target.read_text(), "original: true\n")

    async def test_unknown_model_and_oversized_context_fail(self):
        with self.assertRaisesRegex(agent_setup.SetupError, "Unknown model"):
            await agent_setup.run_setup(dry_run=True, model="missing.gguf")
        with self.assertRaisesRegex(agent_setup.SetupError, "--context-tokens"):
            await agent_setup.run_setup(dry_run=True, context=999999)

    async def test_no_fit_fails_instead_of_downloading_fallback(self):
        hardware = HardwareInfo("small CPU", total_ram_gb=0.5, has_cuda=False)
        with patch.object(agent_setup, "inspect_hardware", return_value=(hardware, {})):
            with self.assertRaisesRegex(agent_setup.SetupError, "fits available memory"):
                await agent_setup.run_setup(dry_run=True)

    async def test_insufficient_disk_fails_before_provisioning(self):
        with patch.object(agent_setup.shutil, "disk_usage", return_value=Mock(free=1)):
            with self.assertRaisesRegex(agent_setup.SetupError, "disk space"):
                await agent_setup.run_setup(dry_run=True)

    async def test_existing_model_is_reused_without_changing_tuned_gpu_layers(self):
        local = self.model_dir / "existing.gguf"
        local.write_bytes(b"GGUF" + (3).to_bytes(4, "little") + b"\0" * 16)
        with patch.object(agent_setup, "load_config", return_value={"llama_model": str(local), "gpu_layers": 0, "context_window_tokens": 1024}):
            result = await agent_setup.run_setup(dry_run=True)
        self.assertEqual(result["model"], str(local))
        self.assertEqual(result["download_bytes"], 0)
        self.assertEqual(result["context_window_tokens"], 1024)

    async def test_downloading_keeps_other_models(self):
        old = self.model_dir / "old.gguf"
        old.write_bytes(b"old")
        target = self.model_dir / "test.gguf"

        async def download(**kwargs):
            target.write_bytes(b"new")
            return 0, "", ""

        with patch("src.provisioning._run_hf_cli_download", side_effect=download):
            await ensure_direct_model(
                {"model_source": "direct", "model_download_url": self.row["url"], "model_download_path": str(target), "preserve_model_files": True},
                log=Mock(), set_status=Mock(), clear_status=Mock(),
            )
        self.assertEqual(old.read_bytes(), b"old")
        self.assertEqual(target.read_bytes(), b"new")


class AgentSetupHardwareTests(unittest.TestCase):
    def test_nvidia_budget_does_not_pool_gpus_and_honors_visibility(self):
        hardware = HardwareInfo("CUDA", total_ram_gb=32, has_cuda=True, vram_mb=49152)
        output = "0, GPU-aaa, RTX A, 24576, 20000\n1, GPU-bbb, RTX B, 24576, 8000\n"
        with patch.object(agent_setup, "detect_hardware_info", return_value=hardware), patch.object(
            agent_setup, "read_memory_snapshot", return_value=MemorySnapshot(32768, 16384, 50)
        ), patch.object(agent_setup.subprocess, "check_output", return_value=output), patch.dict(
            agent_setup.os.environ, {"CUDA_VISIBLE_DEVICES": "1"}
        ):
            budget, facts = agent_setup.inspect_hardware()
        self.assertEqual(budget.vram_mb, 8000)
        self.assertEqual(budget.total_ram_gb, 16)
        self.assertEqual(facts["gpus"][0]["name"], "RTX B")

    def test_metal_budget_uses_available_unified_ram_once(self):
        hardware = HardwareInfo("Apple Silicon", total_ram_gb=32, has_cuda=False, has_metal=True, vram_mb=32768)
        with patch.object(agent_setup, "detect_hardware_info", return_value=hardware), patch.object(
            agent_setup, "read_memory_snapshot", return_value=MemorySnapshot(32768, 16384, 50)
        ):
            budget, facts = agent_setup.inspect_hardware()
        self.assertEqual(budget.total_ram_gb, 16)
        self.assertEqual(budget.vram_mb, 16384)


class AgentSetupVerificationTests(unittest.IsolatedAsyncioTestCase):
    async def _verify(self, text):
        client = Mock(base_url="http://127.0.0.1:18080", active_backend="llama.cpp", context_window_tokens=2048, gpu_layers=0)
        client.start = AsyncMock()
        client.close = AsyncMock()
        response = Mock()
        response.json.return_value = {"choices": [{"message": {"content": text}}]}
        http = AsyncMock()
        http.post.return_value = response
        context = AsyncMock()
        context.__aenter__.return_value = http
        with patch.object(agent_setup, "create_runtime_client", return_value=client) as factory, patch.object(
            agent_setup.httpx, "AsyncClient", return_value=context
        ):
            try:
                result = await agent_setup.verify_inference({"runtime": "llama_cpp", "llama_model": "/test.gguf"})
            finally:
                client.close.assert_awaited_once()
                self.assertTrue(factory.call_args.args[0]["setup_verification"])
        client.start.assert_awaited_once()
        request = http.post.call_args.kwargs["json"]
        self.assertNotIn("tools", request)
        self.assertEqual(request["max_tokens"], 64)
        return result

    async def test_verification_requires_real_generated_text_and_closes_runtime(self):
        self.assertTrue((await self._verify("Hello"))["passed"])

    async def test_empty_generation_fails_and_closes_runtime(self):
        with self.assertRaisesRegex(RuntimeError, "no generated text"):
            await self._verify("")


class AgentSetupCliTests(unittest.TestCase):
    def test_setup_returns_one_json_result_and_never_launches_tui(self):
        stdout, stderr = io.StringIO(), io.StringIO()

        async def run(**kwargs):
            print("subprocess progress")
            return {"status": "ready", "verification": {"passed": True}}

        with patch.object(agent_setup, "run_setup", side_effect=run) as setup, patch("src.cli.launch_tui") as tui, patch(
            "sys.stdout", stdout
        ), patch("sys.stderr", stderr):
            main(["setup", "--non-interactive", "--yes", "--model", "test.gguf", "--json"])
        self.assertEqual(json.loads(stdout.getvalue())["status"], "ready")
        self.assertIn("subprocess progress", stderr.getvalue())
        self.assertEqual(setup.call_args.kwargs["model"], "test.gguf")
        tui.assert_not_called()

    def test_failure_is_json_and_nonzero(self):
        stdout = io.StringIO()
        with patch.object(agent_setup, "run_setup", AsyncMock(side_effect=agent_setup.SetupError("verification", "failed"))), patch(
            "sys.stdout", stdout
        ), patch("src.cli.launch_tui") as tui:
            with self.assertRaises(SystemExit) as error:
                main(["setup", "--yes", "--json"])
        self.assertEqual(error.exception.code, 1)
        self.assertEqual(json.loads(stdout.getvalue())["stage"], "verification")
        tui.assert_not_called()

    def test_bare_setup_preserves_guided_flow(self):
        with patch("src.cli.launch_tui") as tui:
            main(["setup"])
        tui.assert_called_once_with(force_setup=True)

    def test_status_json_works_through_status_alias(self):
        stdout = io.StringIO()
        with patch("src.cli.load_config", return_value={"llama_model": "/model.gguf", "setup_complete": True}), patch(
            "sys.stdout", stdout
        ):
            main(["status", "--json"])
        self.assertTrue(json.loads(stdout.getvalue())["setup_complete"])

    def test_global_json_flag_survives_setup_parser(self):
        stdout = io.StringIO()
        with patch.object(agent_setup, "run_setup", AsyncMock(return_value={"status": "planned"})), patch("sys.stdout", stdout):
            main(["--json", "setup", "--dry-run"])
        self.assertEqual(json.loads(stdout.getvalue())["status"], "planned")

    def test_project_overlay_conflict_is_actionable(self):
        with patch.object(agent_setup, "load_project_config", return_value=(Path(".openjet/config.yaml"), {})):
            with self.assertRaisesRegex(agent_setup.SetupError, "project model overlay"):
                agent_setup._config_target()

    def test_verification_disables_jetson_sudo_prompt(self):
        from src.runtime_registry import create_runtime_client

        client = create_runtime_client({"runtime": "llama_cpp", "llama_model": "/model.gguf", "setup_verification": True})
        self.assertFalse(client.allow_system_tuning)


if __name__ == "__main__":
    unittest.main()
