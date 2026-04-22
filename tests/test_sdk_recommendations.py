from __future__ import annotations

import unittest

from openjet.sdk import (
    HardwareRecommendationInput,
    build_token_generation_workload,
    estimate_recommended_token_generation_speed,
    estimate_token_generation_speed,
    estimate_token_generation_speed_for_profiles,
    estimate_token_generation_speeds_for_hardware,
    get_hardware_performance_profile,
    get_model_performance_profile,
    list_hardware_performance_profiles,
    list_model_performance_profiles,
    recommend_hardware_config,
)


class SDKRecommendationTests(unittest.TestCase):
    def test_recommend_hardware_config_returns_model_and_llama_settings(self) -> None:
        recommendation = recommend_hardware_config(
            HardwareRecommendationInput(
                total_ram_gb=8.0,
                gpu="cpu",
                hardware_profile="other",
                hardware_override="desktop_8",
            )
        )

        self.assertEqual(recommendation.model.label, "Qwen3.5 9B")
        self.assertEqual(recommendation.llama.device, "cpu")
        self.assertEqual(recommendation.llama.gpu_layers, 0)
        self.assertEqual(recommendation.llama.context_window_tokens, 3072)
        self.assertEqual(recommendation.tok_s.hardware_key, "intel_core_i9_14900k")
        self.assertEqual(recommendation.tok_s.model_key, "qwen35_9b_q4_k_m")
        self.assertEqual(recommendation.tok_s.context_window_tokens, 3072)
        self.assertEqual(recommendation.tok_s.hardware_memory_mb, 8192.0)
        self.assertEqual(recommendation.tok_s.limiting_factor, "memory")
        self.assertGreater(recommendation.tok_s.estimated_tokens_per_second, 0.0)

    def test_recommend_hardware_config_uses_passed_hardware_to_differentiate_results(self) -> None:
        low_vram_gpu = recommend_hardware_config(
            {
                "total_ram_gb": 32.0,
                "gpu": "cuda",
                "label": "RTX 4060",
                "vram_mb": 8192.0,
            }
        )
        high_vram_gpu = recommend_hardware_config(
            {
                "total_ram_gb": 32.0,
                "gpu": "cuda",
                "label": "RTX 4090",
                "vram_mb": 24576.0,
            }
        )
        cpu_only = recommend_hardware_config(
            {
                "total_ram_gb": 32.0,
                "gpu": "cpu",
                "label": "CPU only",
            }
        )

        self.assertEqual(low_vram_gpu.model.label, "Qwen3.5 9B")
        self.assertEqual(high_vram_gpu.model.label, "Qwen3.6 27B")
        self.assertEqual(cpu_only.model.label, "Qwen3.6 27B")
        self.assertEqual(low_vram_gpu.llama.context_window_tokens, 128807)
        self.assertEqual(high_vram_gpu.llama.context_window_tokens, 210314)
        self.assertEqual(cpu_only.llama.context_window_tokens, 8192)
        self.assertEqual(low_vram_gpu.llama.device, "cuda")
        self.assertEqual(high_vram_gpu.llama.device, "cuda")
        self.assertEqual(cpu_only.llama.device, "cpu")
        self.assertEqual(low_vram_gpu.llama.gpu_layers, 99)
        self.assertEqual(high_vram_gpu.llama.gpu_layers, 99)
        self.assertEqual(cpu_only.llama.gpu_layers, 0)
        self.assertEqual(low_vram_gpu.tok_s.context_window_tokens, low_vram_gpu.llama.context_window_tokens)
        self.assertEqual(high_vram_gpu.tok_s.context_window_tokens, high_vram_gpu.llama.context_window_tokens)
        self.assertEqual(cpu_only.tok_s.context_window_tokens, cpu_only.llama.context_window_tokens)
        self.assertEqual(low_vram_gpu.tok_s.hardware_key, "rtx_4060_8gb")
        self.assertEqual(high_vram_gpu.tok_s.hardware_key, "rtx_4090_24gb")
        self.assertEqual(cpu_only.tok_s.hardware_key, "intel_core_i9_14900k")
        self.assertEqual(low_vram_gpu.tok_s.hardware_memory_mb, 8192.0)
        self.assertEqual(high_vram_gpu.tok_s.hardware_memory_mb, 24576.0)
        self.assertEqual(cpu_only.tok_s.hardware_memory_mb, 32768.0)
        self.assertLess(low_vram_gpu.tok_s.estimated_tokens_per_second, high_vram_gpu.tok_s.estimated_tokens_per_second)
        self.assertLess(cpu_only.tok_s.estimated_tokens_per_second, low_vram_gpu.tok_s.estimated_tokens_per_second)
        self.assertEqual(low_vram_gpu.tok_s.limiting_factor, "memory")
        self.assertEqual(high_vram_gpu.tok_s.limiting_factor, "memory")
        self.assertEqual(cpu_only.tok_s.limiting_factor, "memory")

    def test_estimate_token_generation_speed_uses_model_decode_context_by_default(self) -> None:
        default_ctx = estimate_token_generation_speed(
            hardware_key="rtx_4060_8gb",
            model_key="qwen35_9b_q4_k_m",
        )
        explicit_ctx = estimate_token_generation_speed(
            hardware_key="rtx_4060_8gb",
            model_key="qwen35_9b_q4_k_m",
            context_window_tokens=8192,
        )

        self.assertEqual(default_ctx.context_window_tokens, 8192)
        self.assertEqual(default_ctx.estimated_tokens_per_second, explicit_ctx.estimated_tokens_per_second)
        self.assertEqual(default_ctx.context_cache_mb, explicit_ctx.context_cache_mb)

    def test_estimate_token_generation_speed_context_override_is_supported(self) -> None:
        short_ctx = estimate_token_generation_speed(
            hardware_key="rtx_4060_8gb",
            model_key="qwen35_9b_q4_k_m",
            context_window_tokens=4096,
        )
        long_ctx = estimate_token_generation_speed(
            hardware_key="rtx_4060_8gb",
            model_key="qwen35_9b_q4_k_m",
            context_window_tokens=32768,
        )

        self.assertEqual(short_ctx.context_window_tokens, 4096)
        self.assertEqual(long_ctx.context_window_tokens, 32768)

    def test_estimate_recommended_token_generation_speed_uses_override_registry(self) -> None:
        estimate = estimate_recommended_token_generation_speed(
            {
                "total_ram_gb": 8.0,
                "gpu": "cpu",
                "label": "anything",
            },
            hardware_profile="other",
            hardware_override="desktop_8",
        )

        self.assertEqual(estimate.model_key, "qwen35_9b_q4_k_m")
        self.assertEqual(estimate.hardware_key, "intel_core_i9_14900k")
        self.assertEqual(estimate.context_window_tokens, 3072)
        self.assertEqual(estimate.hardware_memory_mb, 8192.0)

    def test_estimate_token_generation_speeds_for_hardware_returns_all_models(self) -> None:
        estimates = estimate_token_generation_speeds_for_hardware(
            hardware_key="rtx_4060_8gb",
        )

        self.assertEqual(len(estimates), 5)
        self.assertEqual(estimates[0].hardware_key, "rtx_4060_8gb")
        self.assertTrue(any(item.fits_in_memory for item in estimates))
        self.assertTrue(any(not item.fits_in_memory for item in estimates))
        qwen36_27b = next(item for item in estimates if item.model_key == "qwen36_27b_q4_k_m")
        self.assertFalse(qwen36_27b.fits_in_memory)
        self.assertIsNone(qwen36_27b.estimate)

    def test_rtx_3090_qwen_27b_q4_uses_raw_spec_math(self) -> None:
        estimate = estimate_token_generation_speed(
            hardware_key="rtx_3090_24gb",
            model_key="qwen36_27b_q4_k_m",
        )

        self.assertEqual(estimate.context_window_tokens, 8192)
        self.assertAlmostEqual(estimate.estimated_tokens_per_second, 52.2, places=1)
        self.assertEqual(estimate.limiting_factor, "memory")

    def test_tok_s_registry_profiles_are_exposed_via_sdk(self) -> None:
        hardware = get_hardware_performance_profile("rtx_4060_8gb")
        model = get_model_performance_profile("qwen35_9b_q4_k_m")
        workload = build_token_generation_workload(
            hardware=hardware,
            model=model,
            context_window_tokens=4096,
        )
        estimate = estimate_token_generation_speed_for_profiles(
            hardware=hardware,
            model=model,
            context_window_tokens=4096,
        )

        self.assertEqual(hardware.key, "rtx_4060_8gb")
        self.assertEqual(model.key, "qwen35_9b_q4_k_m")
        self.assertEqual(workload.hardware_key, hardware.key)
        self.assertEqual(workload.model_key, model.key)
        self.assertGreater(workload.weight_bytes_per_token, workload.kv_cache_read_bytes_per_token)
        self.assertGreater(workload.total_flops_per_token, workload.dense_flops_per_token)
        self.assertEqual(estimate.hardware_key, hardware.key)
        self.assertEqual(estimate.model_key, model.key)
        self.assertGreater(len(list_hardware_performance_profiles(device="cuda")), 0)
        self.assertGreater(len(list_model_performance_profiles()), 0)

    def test_estimate_token_generation_speed_rejects_memory_overflow(self) -> None:
        with self.assertRaises(ValueError):
            estimate_token_generation_speed(
                hardware_key="rtx_4060_8gb",
                model_key="qwen36_27b_q4_k_m",
                context_window_tokens=8192,
            )


if __name__ == "__main__":
    unittest.main()
