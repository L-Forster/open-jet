from __future__ import annotations

import unittest

from openjet.sdk import HardwareRecommendationInput, recommend_hardware_config


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

        self.assertEqual(recommendation.model.label, "Qwen3.5 4B")
        self.assertEqual(recommendation.llama.device, "cpu")
        self.assertEqual(recommendation.llama.gpu_layers, 0)
        self.assertEqual(recommendation.llama.context_window_tokens, 3072)

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
        self.assertEqual(high_vram_gpu.model.label, "Qwen3.5 27B")
        self.assertEqual(cpu_only.model.label, "Qwen3.5 9B")
        self.assertEqual(low_vram_gpu.llama.context_window_tokens, 8192)
        self.assertEqual(high_vram_gpu.llama.context_window_tokens, 12288)
        self.assertEqual(cpu_only.llama.context_window_tokens, 8192)
        self.assertEqual(low_vram_gpu.llama.device, "cuda")
        self.assertEqual(high_vram_gpu.llama.device, "cuda")
        self.assertEqual(cpu_only.llama.device, "cpu")
        self.assertEqual(low_vram_gpu.llama.gpu_layers, 99)
        self.assertEqual(high_vram_gpu.llama.gpu_layers, 99)
        self.assertEqual(cpu_only.llama.gpu_layers, 0)


if __name__ == "__main__":
    unittest.main()
