from __future__ import annotations

import unittest
from unittest.mock import patch

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

        self.assertEqual(recommendation.model.label, "Qwen3.5 9B")
        self.assertEqual(recommendation.llama.device, "cpu")
        self.assertEqual(recommendation.llama.gpu_layers, 0)
        self.assertEqual(recommendation.llama.context_window_tokens, 3072)

    def test_recommend_hardware_config_wraps_existing_setup_builder(self) -> None:
        with patch(
            "src.sdk.recommendations.build_recommended_payload",
            return_value={
                "hardware_profile": "auto",
                "hardware_override": "",
                "device": "cuda",
                "gpu_layers": 70,
                "context_window_tokens": 12345,
                "model_download_path": "/models/Qwen3.5-27B-Q4_K_M.gguf",
                "model_download_url": "https://example.invalid/model.gguf",
            },
        ) as build_payload:
            recommendation = recommend_hardware_config(
                {
                    "total_ram_gb": 16.0,
                    "gpu": "cuda",
                    "label": "RTX test box",
                    "vram_mb": 24576.0,
                }
            )

        self.assertEqual(recommendation.llama.context_window_tokens, 12345)
        self.assertEqual(recommendation.llama.gpu_layers, 70)
        self.assertEqual(recommendation.model.target_path, "/models/Qwen3.5-27B-Q4_K_M.gguf")
        self.assertEqual(build_payload.call_args.kwargs["recommended_ctx"], 0)
        self.assertEqual(build_payload.call_args.kwargs["hardware_info"].label, "RTX test box")


if __name__ == "__main__":
    unittest.main()
