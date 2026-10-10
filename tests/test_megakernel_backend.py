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
        for model in ("/m/Qwen3.6-27B-UD-IQ3_XXS.gguf", "/m/Qwen3.8-27B-Q6_K.gguf", "/m/qwen3.8-27b-q4_k_m.bin"):
            self.assertIn("only supports", megakernel_unavailable_reason(model))

    def test_missing_build_is_reported(self) -> None:
        with mock.patch.object(llama_server, "_megakernel_dir", return_value=None):
            self.assertIn("not built", megakernel_unavailable_reason(MODEL))

    def test_unsupported_gpu_is_reported(self) -> None:
        with mock.patch.object(llama_server, "_megakernel_dir", return_value=ROOT), \
                mock.patch.object(llama_server, "_megakernel_gpu", return_value=None):
            self.assertIn("RTX 30/40/50", megakernel_unavailable_reason(MODEL))

    def test_supported_setup_has_no_reason(self) -> None:
        with mock.patch.object(llama_server, "_megakernel_dir", return_value=ROOT), \
                mock.patch.object(llama_server, "_megakernel_gpu", return_value="0"):
            self.assertIsNone(megakernel_unavailable_reason(MODEL))

    def test_backend_setting_is_validated(self) -> None:
        with self.assertRaises(ValueError):
            LlamaServerClient(model=MODEL, backend="vllm")

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
        with _nvidia_smi("0, GPU-aaa, 8.6, 12288", "1, GPU-bbb, 7.5, 24576", "2, GPU-ccc, 9.0, 81559"), \
                mock.patch.dict(llama_server.os.environ, {}, clear=True):
            self.assertIsNone(llama_server._megakernel_gpu())

    def test_ada_and_blackwell_24gb_gpus_qualify(self) -> None:
        for cap, mem in (("8.9", "24564"), ("12.0", "32607")):
            with _nvidia_smi(f"0, GPU-aaa, {cap}, {mem}"), \
                    mock.patch.dict(llama_server.os.environ, {}, clear=True):
                self.assertEqual(llama_server._megakernel_gpu(), "0")


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


if __name__ == "__main__":
    unittest.main()
