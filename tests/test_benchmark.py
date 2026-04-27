from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from src import benchmark
from src.model_profiles import build_model_profile


class BenchmarkTests(unittest.TestCase):
    def test_model_profiles_preserve_moe_and_size_settings(self) -> None:
        profile = build_model_profile(
            {
                "llama_model": "/models/qwen-moe.gguf",
                "llama_cpu_moe": False,
                "llama_n_cpu_moe": 0,
                "active_model_size_mb": 3072,
                "model_size_mb": 22630,
                "kv_bytes_per_token": 24576,
                "unified_memory_only": True,
            },
            name="qwen-moe",
        )

        self.assertIsNotNone(profile)
        assert profile is not None
        self.assertFalse(profile["llama_cpu_moe"])
        self.assertEqual(profile["llama_n_cpu_moe"], 0)
        self.assertEqual(profile["active_model_size_mb"], 3072)
        self.assertEqual(profile["model_size_mb"], 22630)
        self.assertEqual(profile["kv_bytes_per_token"], 24576)
        self.assertTrue(profile["unified_memory_only"])

    def test_run_benchmark_uses_no_repack_completion_for_moe(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp) / "model.gguf"
            model.write_bytes(b"GGUF")
            cfg = {
                "llama_model": str(model),
                "llama_server_path": "/opt/llama/bin/llama-server",
                "device": "vulkan",
                "gpu_layers": 99,
                "llama_n_cpu_moe": 40,
            }

            with patch("src.benchmark.load_config", return_value=cfg), patch(
                "src.benchmark._print_header"
            ), patch(
                "src.benchmark._find_llama_completion", return_value="/opt/llama/bin/llama-completion"
            ), patch("src.benchmark._run_completion_benchmark") as run_completion, patch(
                "src.benchmark._run_bench"
            ) as run_bench:
                benchmark.run_benchmark(repetitions=1)

        run_bench.assert_not_called()
        self.assertEqual(run_completion.call_args.kwargs["gpu_layers"], 99)
        self.assertEqual(run_completion.call_args.kwargs["n_prompt"], 512)
        self.assertEqual(run_completion.call_args.kwargs["n_gen"], 128)

    def test_run_benchmark_maps_cpu_moe_to_all_gpu_layers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp) / "model.gguf"
            model.write_bytes(b"GGUF")
            cfg = {
                "llama_model": str(model),
                "llama_server_path": "/opt/llama/bin/llama-server",
                "device": "vulkan",
                "gpu_layers": 99,
                "llama_cpu_moe": True,
                "llama_n_cpu_moe": 12,
            }

            with patch("src.benchmark.load_config", return_value=cfg), patch(
                "src.benchmark._print_header"
            ), patch(
                "src.benchmark._find_llama_completion", return_value="/opt/llama/bin/llama-completion"
            ), patch("src.benchmark._run_completion_benchmark") as run_completion:
                benchmark.run_benchmark(repetitions=1)

        self.assertTrue(run_completion.called)

    def test_no_repack_completion_command_uses_llamacpp_auto_fit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp) / "model.gguf"
            model.write_bytes(b"GGUF")
            cfg = {
                "llama_model": str(model),
                "llama_server_path": "/opt/llama/bin/llama-server",
                "device": "vulkan",
                "gpu_layers": 99,
                "llama_n_cpu_moe": 0,
                "context_window_tokens": 4096,
            }

            with patch("src.benchmark._find_llama_completion", return_value="/opt/llama/bin/llama-completion"), patch(
                "src.benchmark._moe_block_count", return_value=48
            ), patch(
                "src.benchmark._run_completion_process"
            ) as run_completion:
                benchmark._run_completion_benchmark(
                    cfg,
                    model_path=str(model),
                    device="vulkan",
                    gpu_layers=99,
                    n_prompt=512,
                    n_gen=128,
                    extra_args=None,
                )

        cmd = run_completion.call_args.args[0]
        self.assertEqual(cmd[0], "/opt/llama/bin/llama-completion")
        self.assertIn("--no-repack", cmd)
        self.assertIn("--no-host", cmd)
        self.assertNotIn("-ncmoe", cmd)
        self.assertNotIn("-ot", cmd)
        self.assertIn("-fit", cmd)
        self.assertEqual(cmd[cmd.index("-fit") + 1], "on")
        self.assertNotIn("--show-timings", cmd)
        self.assertIn("--perf", cmd)
        self.assertIn("-ngl", cmd)
        self.assertEqual(cmd[cmd.index("-ngl") + 1], "99")

    def test_explicit_n_cpu_moe_is_still_passed_without_tensor_overrides(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp) / "model.gguf"
            model.write_bytes(b"GGUF")
            cfg = {
                "llama_model": str(model),
                "llama_server_path": "/opt/llama/bin/llama-server",
                "device": "vulkan",
                "gpu_layers": 99,
                "llama_n_cpu_moe": 40,
                "context_window_tokens": 4096,
            }

            with patch("src.benchmark._find_llama_completion", return_value="/opt/llama/bin/llama-completion"), patch(
                "src.benchmark._moe_block_count", return_value=48
            ), patch(
                "src.benchmark._run_completion_process"
            ) as run_completion:
                benchmark._run_completion_benchmark(
                    cfg,
                    model_path=str(model),
                    device="vulkan",
                    gpu_layers=99,
                    n_prompt=512,
                    n_gen=128,
                    extra_args=None,
                )

        cmd = run_completion.call_args.args[0]
        self.assertIn("-ncmoe", cmd)
        self.assertEqual(cmd[cmd.index("-ncmoe") + 1], "40")
        self.assertNotIn("-ot", cmd)
        self.assertIn("-fit", cmd)
        self.assertEqual(cmd[cmd.index("-fit") + 1], "on")

    def test_no_host_is_vulkan_only_for_completion_benchmark(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp) / "model.gguf"
            model.write_bytes(b"GGUF")
            cfg = {
                "llama_model": str(model),
                "llama_server_path": "/opt/llama/bin/llama-server",
                "device": "cuda",
                "gpu_layers": 99,
                "llama_n_cpu_moe": 8,
                "context_window_tokens": 4096,
            }

            with patch("src.benchmark._find_llama_completion", return_value="/opt/llama/bin/llama-completion"), patch(
                "src.benchmark._run_completion_process"
            ) as run_completion:
                benchmark._run_completion_benchmark(
                    cfg,
                    model_path=str(model),
                    device="cuda",
                    gpu_layers=99,
                    n_prompt=512,
                    n_gen=128,
                    extra_args=None,
                )

        cmd = run_completion.call_args.args[0]
        self.assertNotIn("--no-host", cmd)
        self.assertNotIn("-ot", cmd)

    def test_auto_fit_path_does_not_disable_vulkan_fusion(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp) / "model.gguf"
            model.write_bytes(b"GGUF")
            cfg = {
                "llama_model": str(model),
                "llama_server_path": "/opt/llama/bin/llama-server",
                "device": "vulkan",
                "gpu_layers": 99,
                "llama_n_cpu_moe": 0,
                "context_window_tokens": 4096,
            }

            with patch("src.benchmark._find_llama_completion", return_value="/opt/llama/bin/llama-completion"), patch(
                "src.benchmark._moe_block_count", return_value=40
            ), patch("src.benchmark._run_completion_process") as run_completion:
                benchmark._run_completion_benchmark(
                    cfg,
                    model_path=str(model),
                    device="vulkan",
                    gpu_layers=99,
                    n_prompt=512,
                    n_gen=128,
                    extra_args=None,
                )

        env = run_completion.call_args.args[1]
        self.assertNotIn("GGML_VK_DISABLE_FUSION", env)

    def test_synthetic_prompt_does_not_repeat_benchmark_token(self) -> None:
        prompt = benchmark._synthetic_prompt(512)

        self.assertGreaterEqual(len(prompt.split()), 512)
        self.assertNotIn("benchmark benchmark benchmark", prompt)

    def test_host_resident_moe_diagnostic_explains_cpu_mapped_experts(self) -> None:
        diagnostic = benchmark._host_resident_moe_diagnostic(
            cpu_mapped_model_mib=21097.58,
            vulkan_model_mib=1523.49,
            vulkan_total_mib=12243.0,
        )

        self.assertIn("host-resident", diagnostic)
        self.assertIn("12243 MiB", diagnostic)
        self.assertIn("not dynamically streaming only active experts", diagnostic)
        self.assertIn("partial `llama_n_cpu_moe` split plus `--no-host`", diagnostic)

    def test_completion_process_reports_early_exit_without_perf(self) -> None:
        proc = Mock()
        proc.stdout = iter(["load_tensors:      Vulkan0 model buffer size =  9513.49 MiB\n"])
        proc.wait.return_value = 0

        with patch("subprocess.Popen", return_value=proc), patch("builtins.print") as mocked_print:
            benchmark._run_completion_process(["llama-completion"], {})

        printed = "\n".join(str(call.args[0]) for call in mocked_print.call_args_list if call.args)
        self.assertIn("exited before printing performance timings", printed)

    def test_completion_process_does_not_print_host_resident_diagnostic_after_abort(self) -> None:
        proc = Mock()
        proc.stdout = iter(
            [
                "load_tensors:   CPU_Mapped model buffer size = 16473.00 MiB\n",
                "load_tensors:      Vulkan0 model buffer size =  5801.00 MiB\n",
            ]
        )
        proc.wait.return_value = -6

        with patch("subprocess.Popen", return_value=proc), patch("builtins.print") as mocked_print:
            benchmark._run_completion_process(["llama-completion"], {})

        printed = "\n".join(str(call.args[0]) for call in mocked_print.call_args_list if call.args)
        self.assertIn("process exited with code -6", printed)
        self.assertNotIn("host-resident", printed)

    def test_sweep_rejects_moe_when_bench_cannot_disable_repack(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp) / "model.gguf"
            model.write_bytes(b"GGUF")
            cfg = {
                "llama_model": str(model),
                "llama_server_path": "/opt/llama/bin/llama-server",
                "device": "vulkan",
                "gpu_layers": 8,
                "llama_n_cpu_moe": 8,
            }

            with patch("src.benchmark.load_config", return_value=cfg), patch(
                "src.benchmark._print_header"
            ), patch(
                "src.benchmark._find_llama_completion", return_value="/opt/llama/bin/llama-completion"
            ):
                with self.assertRaises(SystemExit):
                    benchmark.run_benchmark_sweep(repetitions=1)

    def test_turbo_server_command_uses_dflash_and_disables_thinking(self) -> None:
        settings = benchmark.TurboBenchmarkSettings(
            target_model="/models/Qwen3.6-27B-Q4_K_M.gguf",
            draft_model="/models/dflash-draft-3.6-q8_0.gguf",
            backend_path="/opt/buun-llama-cpp/build/bin/llama-server",
            backend_kind="llama-server",
            backend_label="spiritbuun/buun-llama-cpp",
            context_size=6048,
            draft_context_size=256,
            gpu_layers=99,
            draft_gpu_layers=99,
            batch_size=256,
            ubatch_size=64,
            thinking_enabled=False,
            baseline_tok_s=None,
        )

        cmd = benchmark._turbo_server_cmd(settings, host="127.0.0.1", port=18080)

        self.assertIn("-md", cmd)
        self.assertEqual(cmd[cmd.index("-md") + 1], "/models/dflash-draft-3.6-q8_0.gguf")
        self.assertIn("--spec-type", cmd)
        self.assertEqual(cmd[cmd.index("--spec-type") + 1], "dflash")
        self.assertIn("-ngld", cmd)
        self.assertEqual(cmd[cmd.index("-ngld") + 1], "99")
        self.assertIn("-cd", cmd)
        self.assertEqual(cmd[cmd.index("-cd") + 1], "256")
        self.assertIn("--jinja", cmd)
        self.assertEqual(cmd[cmd.index("--chat-template-kwargs") + 1], '{"enable_thinking":false}')

    def test_turbo_server_command_can_enable_thinking_mode(self) -> None:
        settings = benchmark.TurboBenchmarkSettings(
            target_model="/models/target.gguf",
            draft_model="/models/draft.gguf",
            backend_path="/opt/llama-server",
            backend_kind="llama-server",
            backend_label="spiritbuun/buun-llama-cpp",
            context_size=4096,
            draft_context_size=256,
            gpu_layers=99,
            draft_gpu_layers=99,
            batch_size=256,
            ubatch_size=64,
            thinking_enabled=True,
            baseline_tok_s=None,
        )

        cmd = benchmark._turbo_server_cmd(settings, host="127.0.0.1", port=18080)

        self.assertEqual(cmd[cmd.index("--chat-template-kwargs") + 1], '{"enable_thinking":true}')

    def test_lucebox_server_command_uses_test_dflash_backend(self) -> None:
        settings = benchmark.TurboBenchmarkSettings(
            target_model="/models/Qwen3.6-27B-Q4_K_M.gguf",
            draft_model="/models/draft/model.safetensors",
            backend_path="/opt/lucebox-hub/dflash/build/test_dflash",
            backend_kind="lucebox",
            backend_label="Luce-Org/lucebox-hub dflash",
            context_size=6048,
            draft_context_size=256,
            gpu_layers=99,
            draft_gpu_layers=99,
            batch_size=256,
            ubatch_size=64,
            thinking_enabled=False,
            baseline_tok_s=None,
        )

        cmd = benchmark._lucebox_server_cmd(settings, host="127.0.0.1", port=18080)

        self.assertTrue(cmd[1].replace("\\", "/").endswith("/opt/lucebox-hub/dflash/scripts/server.py"))
        self.assertIn("--target", cmd)
        self.assertEqual(cmd[cmd.index("--target") + 1], "/models/Qwen3.6-27B-Q4_K_M.gguf")
        self.assertIn("--draft", cmd)
        self.assertEqual(cmd[cmd.index("--draft") + 1], "/models/draft/model.safetensors")
        self.assertIn("--bin", cmd)
        self.assertTrue(cmd[cmd.index("--bin") + 1].replace("\\", "/").endswith("/opt/lucebox-hub/dflash/build/test_dflash"))
        self.assertIn("--max-ctx", cmd)
        self.assertEqual(cmd[cmd.index("--max-ctx") + 1], "6048")

    def test_extract_turbo_timings_reads_llama_server_timings(self) -> None:
        timings = benchmark._extract_turbo_timings(
            {
                "usage": {"prompt_tokens": 100, "completion_tokens": 40},
                "timings": {
                    "prompt_n": 100,
                    "prompt_ms": 250.0,
                    "predicted_n": 40,
                    "predicted_ms": 1000.0,
                },
            }
        )

        self.assertEqual(timings.prompt_tokens, 100)
        self.assertEqual(timings.generation_tokens, 40)
        self.assertAlmostEqual(timings.prompt_eval_tok_s or 0, 400.0)
        self.assertAlmostEqual(timings.generation_tok_s or 0, 40.0)

    def test_load_turbo_settings_resolves_models_and_backend(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "Qwen3.6-27B-Q4_K_M.gguf"
            draft = root / "dflash-draft-3.6-q8_0.gguf"
            backend_path = root / "llama-server"
            target.write_bytes(b"GGUF")
            draft.write_bytes(b"GGUF")
            backend_path.write_text("#!/bin/sh\n", encoding="utf-8")
            cfg = {
                "llama_model": str(target),
                "llama_server_path": str(backend_path),
                "context_window_tokens": 4096,
                "gpu_layers": 42,
                "turbo": {
                    "draft_model": str(draft),
                    "assume_dflash_backend": True,
                    "baseline_tok_s": 20,
                },
            }

            settings = benchmark._load_turbo_settings(cfg, thinking_enabled=False)

        self.assertEqual(settings.target_model, str(target))
        self.assertEqual(settings.draft_model, str(draft))
        self.assertEqual(settings.backend_path, str(backend_path))
        self.assertEqual(settings.backend_kind, "llama-server")
        self.assertEqual(settings.context_size, 4096)
        self.assertEqual(settings.gpu_layers, 42)
        self.assertEqual(settings.baseline_tok_s, 20)

    def test_load_turbo_settings_supports_lucebox_backend(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "Qwen3.6-27B-Q4_K_M.gguf"
            draft = root / "models" / "draft" / "model.safetensors"
            backend_path = root / "lucebox-hub" / "dflash" / "build" / "test_dflash"
            target.write_bytes(b"GGUF")
            draft.parent.mkdir(parents=True)
            draft.write_bytes(b"draft")
            backend_path.parent.mkdir(parents=True)
            backend_path.write_text("#!/bin/sh\n", encoding="utf-8")
            cfg = {
                "llama_model": str(target),
                "turbo": {
                    "backend_kind": "lucebox",
                    "draft_model": str(draft.parent),
                    "lucebox_bin": str(backend_path),
                },
            }

            settings = benchmark._load_turbo_settings(cfg, thinking_enabled=False)

        self.assertEqual(settings.backend_kind, "lucebox")
        self.assertEqual(settings.draft_model, str(draft))
        self.assertEqual(settings.backend_path, str(backend_path))


if __name__ == "__main__":
    unittest.main()
