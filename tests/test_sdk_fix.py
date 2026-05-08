from __future__ import annotations

import io
import json
import unittest
from contextlib import contextmanager
from unittest.mock import patch

from openjet.sdk import RuntimeProcess, fix, format_fix_report
from src.cli import main as cli_main
from src.hardware import HardwareInfo


class SDKFixTests(unittest.TestCase):
    def test_full_gpu_llama_cpp_below_mtp_target_is_slow(self) -> None:
        process = RuntimeProcess(
            backend="llama.cpp",
            pid=1234,
            argv=(
                "llama-server",
                "-m", "/models/qwen.gguf",
                "-ngl", "99",
                "--flash-attn", "on",
                "-c", "16384",
                "-b", "2048",
                "-ub", "512",
                "--port", "18080",
            ),
            executable="llama-server",
            port=18080,
        )
        hardware = HardwareInfo(
            label="RTX 3090",
            total_ram_gb=64.0,
            has_cuda=True,
            vram_mb=24576.0,
        )

        with patch("src.sdk.fix.detect_runtime_processes", return_value=[process]):
            report = fix(
                "llama.cpp",
                cfg={"llama_model": "/models/qwen.gguf", "device": "cuda", "gpu_layers": 99, "context_window_tokens": 16384},
                hardware=hardware,
                run_probe=False,
                observed_decode_tok_s=33.0,
            )

        rendered = format_fix_report(report)
        self.assertIn("OpenJet found why your current local LLM setup is slow.", rendered)
        self.assertIn("Current setup", rendered)
        self.assertIn("Backend", rendered)
        self.assertIn("llama.cpp", rendered)
        self.assertIn("Model", rendered)
        self.assertIn("qwen.gguf", rendered)
        self.assertIn("Context", rendered)
        self.assertIn("16k", rendered)
        self.assertIn("GPU fit", rendered)
        self.assertIn("Full", rendered)
        self.assertIn("Speed", rendered)
        self.assertIn("33 tok/s", rendered)
        self.assertIn("Problem", rendered)
        self.assertIn("OpenJet setup target", rendered)
        self.assertIn("Difference", rendered)
        self.assertIn("Speed up your model by running `openjet setup`.", rendered)

    def test_full_gpu_llama_cpp_at_mtp_target_is_optimal(self) -> None:
        process = RuntimeProcess(
            backend="llama.cpp",
            pid=1234,
            argv=("llama-server", "-m", "/models/qwen.gguf", "-ngl", "99",
                  "--flash-attn", "on", "-c", "16384", "--port", "18080"),
            executable="llama-server",
            port=18080,
        )
        hardware = HardwareInfo(label="RTX 3090", total_ram_gb=64.0, has_cuda=True, vram_mb=24576.0)

        with patch("src.sdk.fix.detect_runtime_processes", return_value=[process]):
            report = fix(
                "llama.cpp",
                cfg={"llama_model": "/models/qwen.gguf", "device": "cuda", "gpu_layers": 99},
                hardware=hardware,
                run_probe=False,
                observed_decode_tok_s=70.0,
            )

        rendered = format_fix_report(report)
        self.assertIn("Your current local LLM setup is already close to optimal.", rendered)
        self.assertIn("OpenJet setup target", rendered)
        self.assertNotIn("Possible gain", rendered)
        self.assertNotIn("5-15%", rendered)
        self.assertNotIn("Problem", rendered)
        self.assertNotIn("Difference", rendered)
        self.assertIn("Speed up your model by running `openjet setup`.", rendered)

    def test_partial_gpu_ollama_is_slow(self) -> None:
        process = RuntimeProcess(
            backend="ollama", pid=31308, argv=("ollama", "serve"),
            executable="/usr/local/bin/ollama", port=11434,
        )
        hardware = HardwareInfo(label="RTX 3090", total_ram_gb=64.0, has_cuda=True, vram_mb=24576.0)
        api_models = [{
            "name": "qwen2.5-coder:14b",
            "size": 9_000_000_000, "size_vram": 6_500_000_000,
            "context_length": 65536,
            "details": {"parameter_size": "14B", "quantization_level": "Q4_K_M"},
        }]
        with patch("src.sdk.fix.detect_runtime_processes", return_value=[process]), \
             patch("src.sdk.fix._ollama_api_ps", return_value=api_models), \
             patch("src.sdk.fix._ollama_ps_rows", return_value=[{"name": "qwen2.5-coder:14b", "processor": "72% GPU"}]), \
             patch("src.sdk.fix._ollama_list_rows", return_value=[]), \
             patch("src.sdk.fix._probe_ollama_decode", return_value=9.8):
            report = fix("ollama", cfg={}, hardware=hardware, run_probe=True)

        rendered = format_fix_report(report)
        self.assertIn("OpenJet found why your current local LLM setup is slow.", rendered)
        self.assertIn("Ollama", rendered)
        self.assertIn("qwen2.5-coder:14b", rendered)
        self.assertIn("64k", rendered)
        self.assertIn("Partial", rendered)
        self.assertIn("9.8 tok/s", rendered)
        self.assertIn("Problem", rendered)
        self.assertIn("Difference", rendered)
        self.assertIn("openjet setup", rendered)

    def test_cpu_offloaded_ollama_is_barely_usable(self) -> None:
        process = RuntimeProcess(
            backend="ollama", pid=31308, argv=("ollama", "serve"),
            executable="/usr/local/bin/ollama", port=11434,
        )
        hardware = HardwareInfo(label="RTX 3060", total_ram_gb=32.0, has_cuda=True, vram_mb=12288.0)
        api_models = [{
            "name": "qwen2.5-coder:32b",
            "size": 19_000_000_000, "size_vram": 3_000_000_000,
            "context_length": 32768,
            "details": {"parameter_size": "32B", "quantization_level": "Q4_K_M"},
        }]
        with patch("src.sdk.fix.detect_runtime_processes", return_value=[process]), \
             patch("src.sdk.fix._ollama_api_ps", return_value=api_models), \
             patch("src.sdk.fix._ollama_ps_rows", return_value=[]), \
             patch("src.sdk.fix._ollama_list_rows", return_value=[]), \
             patch("src.sdk.fix._probe_ollama_decode", return_value=3.4):
            report = fix("ollama", cfg={}, hardware=hardware, run_probe=True)

        rendered = format_fix_report(report)
        self.assertIn("OpenJet found why your current local LLM setup is barely usable.", rendered)
        self.assertIn("CPU/RAM", rendered)
        self.assertIn("3.4 tok/s", rendered)
        self.assertIn("model is too large", rendered.lower())

    def test_no_runtime_detected_outputs_target_only(self) -> None:
        hardware = HardwareInfo(label="RTX 3090", total_ram_gb=64.0, has_cuda=True, vram_mb=24576.0)
        with patch("src.sdk.fix.detect_runtime_processes", return_value=[]):
            report = fix("llama.cpp", cfg={}, hardware=hardware, run_probe=False)

        rendered = format_fix_report(report)
        self.assertIn("OpenJet did not detect a running local LLM", rendered)
        self.assertIn("none detected", rendered)
        self.assertIn("OpenJet setup target", rendered)
        self.assertIn("openjet setup", rendered)

    def test_cli_fix_passes_no_probe_flag_through(self) -> None:
        stdout = io.StringIO()
        hardware = HardwareInfo(label="CPU only", total_ram_gb=16.0, has_cuda=False)

        with patch("src.cli.load_config", return_value={}), \
             patch("src.sdk.fix.detect_runtime_processes", return_value=[]), \
             patch("src.sdk.fix.detect_hardware_info", return_value=hardware), \
             patch("sys.stdout", stdout):
            cli_main(["fix", "llama.cpp", "--no-probe"])

        rendered = stdout.getvalue()
        self.assertIn("OpenJet did not detect a running local LLM", rendered)
        self.assertIn("OpenJet setup target", rendered)
        self.assertIn("openjet setup", rendered)

    def test_llama_cpp_without_draft_flags_does_not_crash(self) -> None:
        process = RuntimeProcess(
            backend="llama.cpp",
            pid=1234,
            argv=("llama-server", "-m", "/models/qwen.gguf", "-ngl", "99", "--port", "18080"),
            executable="llama-server",
            port=18080,
        )
        hardware = HardwareInfo(label="RTX 3090", total_ram_gb=64.0, has_cuda=True, vram_mb=24576.0)

        with patch("src.sdk.fix.detect_runtime_processes", return_value=[process]):
            report = fix("llama.cpp", cfg={}, hardware=hardware, run_probe=False)

        self.assertFalse(report.observation.metadata["draft_engaged"])
        self.assertFalse(report.observation.metadata["model_has_mtp_suffix"])

    def test_llama_cpp_draft_flags_are_reported(self) -> None:
        process = RuntimeProcess(
            backend="llama.cpp",
            pid=1234,
            argv=(
                "llama-server",
                "-m", "/models/qwen-mtp.gguf",
                "--model-draft", "/models/qwen-draft.gguf",
                "--port", "18080",
            ),
            executable="llama-server",
            port=18080,
        )
        hardware = HardwareInfo(label="RTX 3090", total_ram_gb=64.0, has_cuda=True, vram_mb=24576.0)

        with patch("src.sdk.fix.detect_runtime_processes", return_value=[process]):
            report = fix("llama.cpp", cfg={}, hardware=hardware, run_probe=False)

        self.assertTrue(report.observation.metadata["draft_engaged"])
        self.assertTrue(report.observation.metadata["model_has_mtp_suffix"])

    def test_mtp_suffixed_recommended_model_is_not_reported_as_model_change(self) -> None:
        process = RuntimeProcess(
            backend="llama.cpp",
            pid=1234,
            argv=(
                "llama-server",
                "-m", "/models/Qwen3.6-27B-Q4_K_M-mtp.gguf",
                "-ngl", "99",
                "--flash-attn", "on",
                "-c", "16384",
                "--port", "18080",
            ),
            executable="llama-server",
            port=18080,
        )
        hardware = HardwareInfo(label="RTX 3090", total_ram_gb=64.0, has_cuda=True, vram_mb=24576.0)

        with patch("src.sdk.fix.detect_runtime_processes", return_value=[process]):
            report = fix("llama.cpp", cfg={}, hardware=hardware, run_probe=False, observed_decode_tok_s=21.2)

        rendered = format_fix_report(report)
        self.assertIn("Qwen3.6-27B-Q4_K_M-mtp.gguf", rendered)
        self.assertNotIn("Qwen3.6-27B-Q4_K_M-mtp.gguf  →  Qwen3.6 27B Q4_K_M", rendered)
        self.assertIn("The current model already appears to be the MTP variant.", rendered)
        self.assertNotIn("MTP speculative decoding is not enabled.", rendered)

    def test_mtp_suffixed_decode_speed_is_reported_as_effective_speed(self) -> None:
        process = RuntimeProcess(
            backend="llama.cpp",
            pid=1234,
            argv=(
                "llama-server",
                "-m", "/models/Qwen3.6-27B-Q4_K_M-mtp.gguf",
                "-ngl", "99",
                "--flash-attn", "on",
                "-c", "49152",
                "--port", "18080",
            ),
            executable="llama-server",
            port=18080,
        )
        hardware = HardwareInfo(label="RTX 3090", total_ram_gb=64.0, has_cuda=True, vram_mb=24576.0)

        with patch("src.sdk.fix.detect_runtime_processes", return_value=[process]):
            report = fix("llama.cpp", cfg={}, hardware=hardware, run_probe=False, observed_decode_tok_s=36.5)

        rendered = format_fix_report(report)
        self.assertIn("Speed      73 tok/s", rendered)
        self.assertNotIn("Problem", rendered)
        self.assertNotIn("Measured decode speed is still below OpenJet's target", rendered)

    def test_llama_cpp_probe_uses_longer_best_of_two_decode_measurement(self) -> None:
        from src.sdk.fix import _probe_llama_cpp_decode

        requests: list[object] = []
        payloads = iter([
            {"timings": {"predicted_per_second": 21.2}},
            {"timings": {"predicted_per_second": 68.7}},
        ])

        @contextmanager
        def fake_urlopen(req, timeout):
            requests.append(req)

            class Response:
                def read(self) -> bytes:
                    return json.dumps(next(payloads)).encode("utf-8")

            yield Response()

        with patch("src.sdk.fix.urlopen", side_effect=fake_urlopen):
            measured = _probe_llama_cpp_decode("127.0.0.1", 18080)

        self.assertEqual(measured, 68.7)
        self.assertEqual(len(requests), 2)
        body = json.loads(requests[0].data.decode("utf-8"))
        self.assertEqual(body["n_predict"], 160)

    def test_llama_cpp_detection_accepts_built_server_path(self) -> None:
        hardware = HardwareInfo(label="RTX 3090", total_ram_gb=64.0, has_cuda=True, vram_mb=24576.0)
        argv = [
            "/home/louis/llama.cpp/build/bin/server",
            "-m", "/models/live.gguf",
            "--ctx-size", "8192",
            "-ngl", "99",
            "--port", "8080",
        ]

        with patch(
            "src.sdk.fix._iter_process_argv",
            return_value=[(777, argv, "/home/louis/llama.cpp/build/bin/server")],
        ):
            report = fix("llama.cpp", cfg={}, hardware=hardware, run_probe=False, observed_decode_tok_s=70.0)

        rendered = format_fix_report(report)
        self.assertIn("live.gguf", rendered)
        self.assertIn("8k", rendered)

    def test_mtp_speedup_doubles_recommended_tok_s(self) -> None:
        from src.sdk.tok_s import (
            MTP_DECODE_SPEEDUP,
            estimate_token_generation_speed,
        )
        baseline = estimate_token_generation_speed(
            hardware_key="rtx_3090_24gb",
            model_key="qwen36_27b_q4_k_m",
            context_window_tokens=16384,
        )
        with_mtp = estimate_token_generation_speed(
            hardware_key="rtx_3090_24gb",
            model_key="qwen36_27b_q4_k_m",
            context_window_tokens=16384,
            decode_speedup=MTP_DECODE_SPEEDUP,
        )
        self.assertAlmostEqual(MTP_DECODE_SPEEDUP, 2.0)
        self.assertAlmostEqual(
            with_mtp.estimated_tokens_per_second / baseline.estimated_tokens_per_second,
            2.0,
            delta=0.05,
        )


if __name__ == "__main__":
    unittest.main()
