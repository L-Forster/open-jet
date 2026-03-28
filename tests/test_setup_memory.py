from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import src.setup as setup_source

from src.setup_memory import (
    _detect_amd_free_vram_mb_from_sysfs,
    detect_free_accelerator_memory_mb,
    estimate_model_memory_mb,
    recommend_context_window_from_remaining_vram_mb,
    recommend_setup_context_window,
)

class SetupMemoryTests(unittest.TestCase):
    def test_detect_free_accelerator_memory_mb_uses_max_free_gpu(self) -> None:
        proc = Mock(
            returncode=0,
            stdout="RTX 4090, 24564, 8123, 16441\nRTX 3080, 10240, 4096, 6144\n",
        )

        with patch("src.setup_memory.shutil.which", return_value="/usr/bin/nvidia-smi"), patch(
            "src.setup_memory.subprocess.run",
            return_value=proc,
        ):
            free_mb = detect_free_accelerator_memory_mb("cuda")

        self.assertEqual(free_mb, 16441.0)

    def test_estimate_model_memory_mb_prefers_real_file_size(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp) / "demo.gguf"
            model.write_bytes(b"x" * (3 * 1024 * 1024))

            estimated = estimate_model_memory_mb(str(model), "Qwen3.5 27B")

        self.assertEqual(estimated, 3.0)

    def test_detect_amd_free_vram_mb_from_sysfs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            device_dir = Path(tmp) / "card0" / "device"
            device_dir.mkdir(parents=True)
            total_path = device_dir / "mem_info_vram_total"
            used_path = device_dir / "mem_info_vram_used"
            total_path.write_text(str(8 * 1024 * 1024 * 1024), encoding="utf-8")
            used_path.write_text(str(3 * 1024 * 1024 * 1024), encoding="utf-8")

            with patch("src.setup_memory.Path.glob", return_value=[total_path]):
                free_mb = _detect_amd_free_vram_mb_from_sysfs()

        self.assertEqual(free_mb, 5120.0)

    def test_recommend_setup_context_window_uses_remaining_vram_after_model(self) -> None:
        with patch("src.setup_memory.detect_free_accelerator_memory_mb", return_value=8192.0), patch(
            "src.setup_memory.estimate_model_memory_mb",
            return_value=5120.0,
        ):
            recommended = recommend_setup_context_window(
                runtime="llama_cpp",
                device="cuda",
                fallback_tokens=8192,
                model_refs=["/models/Qwen3.5-9B-Q4_K_M.gguf"],
            )

        self.assertEqual(recommended, 4096)

    def test_recommend_context_window_from_remaining_vram_mb_has_small_floor(self) -> None:
        self.assertEqual(recommend_context_window_from_remaining_vram_mb(-1), 1024)
        self.assertEqual(recommend_context_window_from_remaining_vram_mb(1200), 1024)

    def test_detect_free_accelerator_memory_mb_uses_amd_paths_for_vulkan(self) -> None:
        with patch("src.setup_memory._detect_amd_free_vram_mb", return_value=6144.0):
            self.assertEqual(detect_free_accelerator_memory_mb("vulkan"), 6144.0)

    def test_detect_free_accelerator_memory_mb_uses_unified_memory_on_darwin(self) -> None:
        snapshot = Mock(available_mb=6144.0)
        with patch("src.setup_memory.sys.platform", "darwin"), patch(
            "src.setup_memory.read_memory_snapshot",
            return_value=snapshot,
        ):
            self.assertEqual(detect_free_accelerator_memory_mb("cpu"), 6144.0)


class SetupSourceIntegrationTests(unittest.TestCase):
    def test_build_recommended_payload_uses_setup_memory_recommendation(self) -> None:
        hardware = setup_source.HardwareInfo(label="CUDA-capable device", total_ram_gb=16.0, has_cuda=True)

        with patch.object(setup_source, "discover_model_files", return_value=["/models/demo.gguf"]), patch.object(
            setup_source,
            "_discover_llama_server",
            return_value="/usr/bin/llama-server",
        ), patch.object(
            setup_source,
            "recommended_gpu_layers",
            return_value=99,
        ), patch.object(
            setup_source,
            "recommend_setup_context_window",
            return_value=6144,
        ) as recommend_ctx:
            payload = setup_source.build_recommended_payload(
                hardware_info=hardware,
                recommended_ctx=8192,
                current_cfg={},
            )

        self.assertEqual(payload["context_window_tokens"], 6144)
        call = recommend_ctx.call_args
        self.assertIsNotNone(call)
        self.assertEqual(call.kwargs["runtime"], "llama_cpp")
        self.assertEqual(call.kwargs["device"], "cuda")
        self.assertIn("/models/demo.gguf", call.kwargs["model_refs"])
