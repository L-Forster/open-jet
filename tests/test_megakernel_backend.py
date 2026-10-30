import asyncio
import importlib.util
import json
import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock

from src import llama_server
from src.llama_server import LlamaServerClient, megakernel_unavailable_reason
from src.runtime_registry import create_runtime_client

ROOT = Path(__file__).resolve().parent.parent
MODEL = "/models/Qwen3.8-27B-Q4_K_M.gguf"


def _load_mk_server():
    path = ROOT / "megakernel" / "server" / "mk_server.py"
    spec = importlib.util.spec_from_file_location("mk_server_under_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class MegakernelSelectionTests(unittest.TestCase):
    def test_other_models_are_not_served_by_the_megakernel(self) -> None:
        for model in ("/m/Qwen3.6-27B-UD-IQ3_XXS.gguf", "/m/Qwen3.8-27B-Q6_K.gguf", "/m/qwen3.8-27b-q4_k_m.bin",
                      "/m/Qwen3.8-27B-UD-Q4_K_M.gguf", "/m/Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf"):
            self.assertIn("only supports", megakernel_unavailable_reason(model))

    def test_setup_selector_names_the_runtime(self) -> None:
        with mock.patch.object(llama_server, "_megakernel_gpu_info", return_value=("0", "8.6")):
            self.assertEqual(llama_server.model_runtime_label(MODEL), "OpenJet megakernel, 1.5x faster")
            self.assertEqual(llama_server.model_runtime_label("/m/Qwen3.8-27B-UD-Q4_K_M.gguf"), "llama.cpp")
            self.assertEqual(llama_server.model_runtime_label("/m/Qwen3.5-9B-Q4_K_M.gguf"), "llama.cpp")
            self.assertEqual(llama_server.model_runtime_label("/m/mmproj-Qwen3.8-27B-BF16.gguf"), "llama.cpp")
        with mock.patch.object(llama_server, "_megakernel_gpu_info", return_value=None):
            self.assertEqual(llama_server.model_runtime_label(MODEL), "llama.cpp")

    def test_turing_selector_does_not_claim_ampere_benchmark_speed(self) -> None:
        with mock.patch.object(llama_server, "_megakernel_gpu_info", return_value=("3", "7.5")):
            self.assertEqual(llama_server.model_runtime_label(MODEL), llama_server.MEGAKERNEL_TURING_LABEL)
            with mock.patch.object(llama_server.Path, "is_file", return_value=True), \
                    mock.patch.object(llama_server, "_megakernel_quant_mix", return_value="native"):
                self.assertEqual(llama_server.model_runtime_label(MODEL), llama_server.MEGAKERNEL_TURING_LABEL)

    def test_missing_build_is_reported(self) -> None:
        with mock.patch.object(llama_server, "_megakernel_dir", return_value=None):
            self.assertIn("not built", megakernel_unavailable_reason(MODEL))

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
        with mock.patch.object(llama_server, "_megakernel_gpu_info", return_value=("0", "7.5")):
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

    def test_rtx_8000_can_be_pinned_without_selecting_another_gpu(self) -> None:
        lines = ("2, GPU-baseline, 8.6, 24576", "3, GPU-turing, 7.5, 49152")
        for visible in ("3", "GPU-turing"):
            with _nvidia_smi(*lines), \
                    mock.patch.dict(llama_server.os.environ, {"CUDA_VISIBLE_DEVICES": visible}, clear=True):
                self.assertEqual(llama_server._megakernel_gpu_info(), ("3", "7.5"))

    def test_nvml_failure_uses_cuda_instead_of_disabling_the_megakernel(self) -> None:
        failed = subprocess.CompletedProcess([], 18, stdout="", stderr="Driver/library version mismatch")
        with mock.patch.object(llama_server.subprocess, "run", return_value=failed), \
                mock.patch.object(llama_server, "_cuda_megakernel_gpu_info", return_value=("GPU-turing", "7.5")):
            self.assertEqual(llama_server._megakernel_gpu_info(), ("GPU-turing", "7.5"))

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
            major._obj.value = 7 if device == 0 else 8
            minor._obj.value = 5 if device == 0 else 6
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
                             ("GPU-00000000-0000-0000-0000-000000000003", "7.5"))
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

    def test_megakernel_launch_preserves_the_pinned_rtx_8000(self) -> None:
        async def scenario() -> None:
            client = self._client("megakernel")
            client._proc = mock.Mock(pid=123)
            client._spawn_until_healthy = mock.AsyncMock(return_value="ready")
            env = {"CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": "GPU-turing"}
            with mock.patch.object(llama_server, "_megakernel_dir", return_value=ROOT), \
                    mock.patch.object(llama_server, "_megakernel_gpu") as choose_gpu:
                await client._start_megakernel(env=env, ctx=4096)
            choose_gpu.assert_not_called()
            self.assertEqual(client._spawn_until_healthy.call_args.kwargs["env"], env)
            await client._http.aclose()

        self._run(scenario())


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


class MegakernelServerParsingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.mk = _load_mk_server()

    def _feed_all(self, parser, chunks):
        out = []
        for c in chunks:
            out += parser.feed(c)
        out += parser.feed("", final=True)
        merged = {"reasoning": "", "content": ""}
        for kind, text in out:
            merged[kind] += text
        return merged

    def test_reasoning_and_content_are_split_across_chunk_boundaries(self) -> None:
        parser = self.mk.OutputParser(thinking=True, tools=[])
        text = "Think about it.\n</think>\n\nThe answer is 4."
        merged = self._feed_all(parser, [text[i:i + 3] for i in range(0, len(text), 3)])
        self.assertEqual(merged["reasoning"], "Think about it.\n")
        self.assertEqual(merged["content"], "The answer is 4.")

    def test_tool_call_markup_is_held_back_and_parsed(self) -> None:
        tools = [{"type": "function", "function": {"name": "read_file", "parameters": {
            "type": "object", "properties": {"path": {"type": "string"}, "max_lines": {"type": "integer"}}}}}]
        parser = self.mk.OutputParser(thinking=False, tools=tools)
        text = ("Let me look.\n\n<tool_call>\n<function=read_file>\n<parameter=path>\n/etc/hostname\n</parameter>\n"
                "<parameter=max_lines>\n10\n</parameter>\n</function>\n</tool_call>")
        merged = self._feed_all(parser, [text[i:i + 5] for i in range(0, len(text), 5)])
        self.assertEqual(merged["content"], "Let me look.")
        calls = self.mk.parse_tool_calls(parser.tool_text, tools)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["function"]["name"], "read_file")
        self.assertEqual(json.loads(calls[0]["function"]["arguments"]), {"path": "/etc/hostname", "max_lines": 10})

    def test_string_parameters_are_not_json_decoded(self) -> None:
        tools = [{"type": "function", "function": {"name": "run", "parameters": {
            "type": "object", "properties": {"cmd": {"type": "string"}}}}}]
        calls = self.mk.parse_tool_calls("<tool_call>\n<function=run>\n<parameter=cmd>\n123\n</parameter>\n</function>\n</tool_call>", tools)
        self.assertEqual(json.loads(calls[0]["function"]["arguments"]), {"cmd": "123"})

    def test_assistant_tool_call_arguments_are_decoded_for_the_template(self) -> None:
        msgs = self.mk.normalize_messages([{"role": "assistant", "content": None, "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{\"a\": 1}"}}]}])
        self.assertEqual(msgs[0]["content"], "")
        self.assertEqual(msgs[0]["tool_calls"][0]["function"]["arguments"], {"a": 1})

    def test_images_are_rejected(self) -> None:
        with self.assertRaises(self.mk.BadRequest):
            self.mk.normalize_messages([{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "x"}}]}])

    def test_shared_prefix_ends_on_a_32_token_tile_under_mk_kv(self) -> None:
        srv = object.__new__(self.mk.Server)
        srv.shared_text = "shared context"
        srv.template_args = lambda body, messages, generation_prompt: {}
        srv.template = mock.Mock(render=lambda **kw: "")
        prefix = list(range(1108))
        srv.eng = mock.Mock(tokenize=lambda text: prefix)
        toks = prefix + [5000, 5001]
        with mock.patch.dict("os.environ", {"MK_KV": ""}):
            self.assertEqual(srv.share_len({}, toks), 1100)
        with mock.patch.dict("os.environ", {"MK_KV": "K*=5r,V*=5"}):
            self.assertEqual(srv.share_len({}, toks), 1088)


if __name__ == "__main__":
    unittest.main()
