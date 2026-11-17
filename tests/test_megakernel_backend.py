import asyncio
import importlib.util
import json
import subprocess
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from src import llama_server
from src.llama_server import LlamaServerClient, megakernel_unavailable_reason
from src.runtime_registry import create_runtime_client

ROOT = Path(__file__).resolve().parent.parent
MODEL = "/models/Qwen3.8-27B-Q4_K_M.gguf"


class MegakernelSelectionTests(unittest.TestCase):
    def test_other_models_are_not_served_by_the_megakernel(self) -> None:
        for model in ("/m/Qwen3.6-27B-UD-IQ3_XXS.gguf", "/m/Qwen3.8-27B-Q6_K.gguf", "/m/qwen3.8-27b-q4_k_m.bin",
                      "/m/Qwen3.8-27B-UD-Q4_K_M.gguf", "/m/Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf"):
            self.assertIn("only supports", megakernel_unavailable_reason(model))

    def test_setup_selector_names_the_runtime(self) -> None:
        with mock.patch.object(llama_server, "_megakernel_gpu_info", return_value=("0", "8.6")):
            self.assertEqual(llama_server.model_runtime_label(MODEL), "OpenJet engine, 2x faster")
            self.assertEqual(llama_server.model_runtime_label("/m/Qwen3.8-27B-UD-Q4_K_M.gguf"), "llama.cpp")
            self.assertEqual(llama_server.model_runtime_label("/m/Qwen3.5-9B-Q4_K_M.gguf"), "llama.cpp")
            self.assertEqual(llama_server.model_runtime_label("/m/mmproj-Qwen3.8-27B-BF16.gguf"), "llama.cpp")
        with mock.patch.object(llama_server, "_megakernel_gpu_info", return_value=None):
            self.assertEqual(llama_server.model_runtime_label(MODEL), "llama.cpp")

    def test_turing_and_ud_q3_k_xl_run_on_the_previous_engine(self) -> None:
        with _nvidia_smi("3, GPU-turing, 7.5, 49152"), mock.patch.dict(llama_server.os.environ, {}, clear=True), \
                mock.patch.object(llama_server, "_cuda_megakernel_gpu_info", return_value=None):
            self.assertEqual(llama_server.model_runtime_label(MODEL), llama_server.MEGAKERNEL_TURING_LABEL)
            self.assertEqual(llama_server.model_runtime_label("/models/Qwen3.8-27B-UD-Q3_K_XL.gguf"), "llama.cpp")
        with mock.patch.object(llama_server, "_megakernel_gpu_info", return_value=("0", "8.6")):
            self.assertEqual(llama_server.model_runtime_label("/models/Qwen3.8-27B-UD-Q3_K_XL.gguf"),
                             llama_server.MEGAKERNEL_MIXED_LABEL)

    def test_missing_build_is_reported(self) -> None:
        with mock.patch.object(llama_server, "_megakernel_dir", return_value=None):
            self.assertIn("not installed", megakernel_unavailable_reason(MODEL))

    def test_unsupported_gpu_is_reported(self) -> None:
        with mock.patch.object(llama_server, "_megakernel_dir", return_value=ROOT), \
                mock.patch.object(llama_server, "_megakernel_gpu_info", return_value=None):
            self.assertIn("RTX 30/40/50", megakernel_unavailable_reason(MODEL))

    def test_supported_setup_has_no_reason(self) -> None:
        with mock.patch.object(llama_server, "_megakernel_dir", return_value=ROOT), \
                mock.patch.object(llama_server, "_megakernel_gpu_info", return_value=("0", "8.6")):
            self.assertIsNone(megakernel_unavailable_reason(MODEL))

    def test_backend_setting_is_validated(self) -> None:
        with self.assertRaises(ValueError):
            LlamaServerClient(model=MODEL, backend="vllm")

    def test_registry_passes_the_long_context_setting(self) -> None:
        cfg = {"runtime": "llama_cpp", "llama_model": MODEL, "megakernel_long_context": True}
        self.assertTrue(create_runtime_client(cfg).megakernel_long_context)
        self.assertFalse(create_runtime_client({"runtime": "llama_cpp", "llama_model": MODEL}).megakernel_long_context)

    def test_long_context_needs_a_supported_model_and_an_ampere_or_newer_gpu(self) -> None:
        with mock.patch.object(llama_server, "_megakernel_gpu_info", return_value=("0", "8.6")):
            self.assertIsNone(llama_server.megakernel_long_context_reason(MODEL))
            self.assertIn("only supports", llama_server.megakernel_long_context_reason("/m/Qwen3.5-9B-Q4_K_M.gguf"))
        with mock.patch.object(llama_server, "_megakernel_gpu_info", return_value=None):
            self.assertIn("RTX 30/40/50", llama_server.megakernel_long_context_reason(MODEL))

    def test_registry_passes_the_backend_setting(self) -> None:
        cfg = {"runtime": "llama_cpp", "llama_model": MODEL, "llama_backend": "llama"}
        client = create_runtime_client(cfg)
        self.assertEqual(client.backend, "llama")
        self.assertEqual(create_runtime_client({"runtime": "llama_cpp", "llama_model": MODEL}).backend, "auto")


def _nvidia_smi(*lines: str):
    return mock.patch.object(
        llama_server.subprocess, "run",
        return_value=subprocess.CompletedProcess([], 0, stdout="\n".join(lines) + "\n", stderr=""),
    )


class MegakernelGpuDetectionTests(unittest.TestCase):
    def test_a_qualifying_gpu_after_the_first_is_found(self) -> None:
        with _nvidia_smi("0, GPU-aaa, 7.5, 8192", "1, GPU-bbb, 8.6, 24576"), \
                mock.patch.dict(llama_server.os.environ, {}, clear=True):
            self.assertEqual(llama_server._megakernel_gpu(), "1")

    def test_cuda_visible_devices_limits_the_choice(self) -> None:
        lines = ("0, GPU-aaa, 8.6, 24576", "1, GPU-bbb, 7.5, 8192")
        with _nvidia_smi(*lines), mock.patch.dict(llama_server.os.environ, {"CUDA_VISIBLE_DEVICES": "1"}):
            self.assertIsNone(llama_server._megakernel_gpu())
        with _nvidia_smi(*lines), mock.patch.dict(llama_server.os.environ, {"CUDA_VISIBLE_DEVICES": "GPU-aaa,1"}):
            self.assertEqual(llama_server._megakernel_gpu(), "0")

    def test_small_or_other_arch_gpus_do_not_qualify(self) -> None:
        with _nvidia_smi("0, GPU-aaa, 8.6, 12288", "1, GPU-bbb, 7.0, 24576", "2, GPU-ccc, 9.0, 81559"), \
                mock.patch.dict(llama_server.os.environ, {}, clear=True):
            self.assertIsNone(llama_server._megakernel_gpu())

    def test_ada_and_blackwell_24gb_gpus_qualify(self) -> None:
        for cap, mem in (("8.9", "24564"), ("12.0", "32607")):
            with _nvidia_smi(f"0, GPU-aaa, {cap}, {mem}"), \
                    mock.patch.dict(llama_server.os.environ, {}, clear=True):
                self.assertEqual(llama_server._megakernel_gpu(), "0")

    def test_a_gpu_can_be_pinned_without_selecting_another_gpu(self) -> None:
        lines = ("2, GPU-baseline, 8.6, 24576", "3, GPU-ada, 8.9, 49152")
        for visible in ("3", "GPU-ada"):
            with _nvidia_smi(*lines), \
                    mock.patch.dict(llama_server.os.environ, {"CUDA_VISIBLE_DEVICES": visible}, clear=True):
                self.assertEqual(llama_server._megakernel_gpu_info(), ("3", "8.9"))

    def test_nvml_failure_uses_cuda_instead_of_disabling_the_megakernel(self) -> None:
        failed = subprocess.CompletedProcess([], 18, stdout="", stderr="Driver/library version mismatch")
        with mock.patch.object(llama_server.subprocess, "run", return_value=failed), \
                mock.patch.object(llama_server, "_cuda_megakernel_gpu_info", return_value=("GPU-ada", "8.9")):
            self.assertEqual(llama_server._megakernel_gpu_info(), ("GPU-ada", "8.9"))

    def test_cuda_fallback_uses_visible_device_uuid_and_respects_first_device(self) -> None:
        driver = mock.Mock()
        driver.cuInit.return_value = 0

        def count(out):
            out._obj.value = 2
            return 0

        def get(out, ordinal):
            out._obj.value = ordinal
            return 0

        def capability(major, minor, device):
            major._obj.value = 8
            minor._obj.value = 9 if device == 0 else 6
            return 0

        def memory(out, device):
            out._obj.value = (49152 if device == 0 else 24576) * 1024 * 1024
            return 0

        def uuid(out, device):
            out._obj[:] = bytes.fromhex("00000000000000000000000000000003")
            return 0

        driver.cuDeviceGetCount.side_effect = count
        driver.cuDeviceGet.side_effect = get
        driver.cuDeviceComputeCapability.side_effect = capability
        driver.cuDeviceTotalMem_v2.side_effect = memory
        driver.cuDeviceGetUuid.side_effect = uuid
        with mock.patch.object(llama_server.ctypes, "CDLL", return_value=driver), \
                mock.patch.dict(llama_server.os.environ, {"CUDA_VISIBLE_DEVICES": "3,2"}, clear=True):
            self.assertEqual(llama_server._cuda_megakernel_gpu_info(),
                             ("GPU-00000000-0000-0000-0000-000000000003", "8.9"))
            driver.cuDeviceGet.assert_called_once()
            driver.cuDeviceTotalMem_v2.side_effect = lambda out, device: (
                setattr(out._obj, "value", 8192 * 1024 * 1024) or 0)
            self.assertIsNone(llama_server._cuda_megakernel_gpu_info())

    def test_cuda_detection_failure_is_reported(self) -> None:
        driver = mock.Mock()
        driver.cuInit.return_value = 803
        with mock.patch.object(llama_server.ctypes, "CDLL", return_value=driver), \
                self.assertLogs(llama_server.__name__, level="WARNING") as logs:
            self.assertIsNone(llama_server._cuda_megakernel_gpu_info())
        self.assertIn("cuInit failed with error 803", logs.output[0])


class MegakernelStartTests(unittest.TestCase):
    def _client(self, backend: str) -> LlamaServerClient:
        client = LlamaServerClient(model=MODEL, binary="/opt/llama-server", backend=backend)
        client._cleanup_stale_inference_processes = mock.AsyncMock()
        client._resolve_device = mock.Mock(return_value="vulkan")
        return client

    def _run(self, coro):
        return asyncio.run(coro)

    def test_required_megakernel_fails_on_a_non_cuda_device(self) -> None:
        client = self._client("megakernel")
        client._start_once = mock.AsyncMock()
        with mock.patch.object(llama_server, "assert_endpoint_allowed"):
            with self.assertRaisesRegex(RuntimeError, "desktop CUDA GPU"):
                self._run(client.start())
        client._start_once.assert_not_called()

    def test_required_megakernel_fails_on_jetson(self) -> None:
        client = self._client("megakernel")
        client._resolve_device.return_value = "cuda"
        client._is_jetson_platform = mock.Mock(return_value=True)
        client._ensure_jetson_clocks_sudoers = mock.Mock()
        client._maximize_gpu_clocks = mock.Mock()
        client._prepare_memory_for_launch = mock.AsyncMock(return_value=None)
        client._start_once = mock.AsyncMock()
        with mock.patch.object(llama_server, "assert_endpoint_allowed"):
            with self.assertRaisesRegex(RuntimeError, "jetson"):
                self._run(client.start())
        client._start_once.assert_not_called()

    def test_fallback_to_llama_server_resets_the_active_backend(self) -> None:
        client = self._client("auto")
        client.active_backend = "megakernel"
        client._start_once = mock.AsyncMock()
        with mock.patch.object(llama_server, "assert_endpoint_allowed"):
            self._run(client.start())
        client._start_once.assert_called_once()
        self.assertEqual(client.active_backend, "llama.cpp")

    def test_stopping_the_server_stops_the_stderr_reader(self) -> None:
        async def scenario() -> None:
            client = LlamaServerClient(model=MODEL)
            stderr = asyncio.StreamReader()
            client._proc = mock.Mock(returncode=0, stderr=stderr)
            client._stderr_task = asyncio.create_task(client._capture_startup_stderr(llama_server.deque()))
            await asyncio.sleep(0)
            task = client._stderr_task
            await client._stop_server()
            self.assertIsNone(client._proc)
            stderr.feed_data(b"late line\n")
            with self.assertRaises(asyncio.CancelledError):
                await task
            await client._http.aclose()

        self._run(scenario())

    def test_megakernel_launch_preserves_the_pinned_gpu(self) -> None:
        async def scenario() -> None:
            client = self._client("megakernel")
            client._proc = mock.Mock(pid=123)
            client._spawn_until_healthy = mock.AsyncMock(return_value="ready")
            env = {"CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": "GPU-pinned"}
            with mock.patch.object(llama_server, "_megakernel_dir", return_value=ROOT), \
                    mock.patch.object(llama_server, "_megakernel_gpu") as choose_gpu:
                await client._start_megakernel(env=env, ctx=4096)
            choose_gpu.assert_not_called()
            self.assertEqual(client._spawn_until_healthy.call_args.kwargs["env"], env)
            await client._http.aclose()

        self._run(scenario())


class MegakernelLaunchCommandTests(unittest.TestCase):
    def _launch(self, outcomes, *, ram_mb: float = 65536.0, ctx: int = 32768, stderr: str = ""):
        async def scenario():
            client = LlamaServerClient(model=MODEL, binary="/opt/llama-server", backend="megakernel")
            async def spawn(cmd, *, env, timeout, stderr_lines, keep_reading_stderr=False):
                client._proc = mock.Mock(pid=123)
                stderr_lines.append(stderr)
                return outcomes.pop(0)

            client._spawn_until_healthy = mock.AsyncMock(side_effect=spawn)
            snapshot = SimpleNamespace(total_mb=ram_mb)
            with mock.patch.object(llama_server, "_megakernel_dir", return_value=ROOT), \
                    mock.patch.object(llama_server, "_megakernel_gpu", return_value=None), \
                    mock.patch.object(llama_server, "read_memory_snapshot", return_value=snapshot):
                try:
                    await client._start_megakernel(env={}, ctx=ctx)
                finally:
                    await client._http.aclose()
            return [(call.args[0], call.kwargs["env"]) for call in client._spawn_until_healthy.call_args_list]
        return asyncio.run(scenario())

    def test_launch_uses_the_q8_cache_exact_reuse_and_seven_drafts(self) -> None:
        (cmd, env), = self._launch(["ready"])
        self.assertEqual(cmd[-2:], ["-d", "7"])
        self.assertIn("--exact-reuse", cmd)
        self.assertEqual(cmd[cmd.index("--kv") + 1], "q8")
        self.assertNotIn("--park-mib", cmd)
        self.assertNotIn("MK_PF_CAP", env)

    def test_a_gpu_with_too_few_sms_is_started_again_at_four_drafts(self) -> None:
        message = "RTX A5000 has 64 SMs; the megakernel runs one block per SM and needs at least 80"
        calls = self._launch(["exited", "ready"], stderr=message)
        self.assertEqual([cmd[-1] for cmd, _env in calls], ["7", "4"])

    def test_another_startup_failure_is_not_retried(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "out of memory"):
            self._launch(["exited"], stderr="CUDA error: out of memory")

    def test_parking_is_off_on_a_machine_with_little_ram(self) -> None:
        (cmd, _env), = self._launch(["ready"], ram_mb=15955.0)
        self.assertEqual(cmd[cmd.index("--park-mib") + 1], "0")

    def test_long_prompts_get_smaller_prefill_chunks(self) -> None:
        (_cmd, env), = self._launch(["ready"], ctx=120000)
        self.assertEqual(env["MK_PF_CAP"], "1024")


class MegakernelLongContextLaunchTests(unittest.TestCase):
    def _launch_env(self, long_context: bool, env: dict) -> dict:
        async def scenario() -> dict:
            client = LlamaServerClient(model=MODEL, binary="/opt/llama-server", backend="megakernel",
                                       megakernel_long_context=long_context)
            client._proc = mock.Mock(pid=123)
            client._spawn_until_healthy = mock.AsyncMock(return_value="ready")
            with mock.patch.object(llama_server, "_megakernel_dir", return_value=ROOT), \
                    mock.patch.object(llama_server, "_megakernel_gpu", return_value=None):
                await client._start_megakernel(env=dict(env), ctx=200000)
            await client._http.aclose()
            return client._spawn_until_healthy.call_args.kwargs["env"]
        return asyncio.run(scenario())

    def test_the_default_launch_sets_no_kv_cache_format(self) -> None:
        self.assertNotIn("MK_KV", self._launch_env(False, {}))

    def test_long_context_starts_the_compact_cache(self) -> None:
        env = self._launch_env(True, {})
        self.assertEqual(env["MK_KV"], "K*=5r,V*=5")
        self.assertEqual(env["MK_KV_WIN"], "2048")

    def test_an_explicit_mk_kv_wins(self) -> None:
        self.assertEqual(self._launch_env(True, {"MK_KV": "K*=6,V*=5"})["MK_KV"], "K*=6,V*=5")


if __name__ == "__main__":
    unittest.main()
