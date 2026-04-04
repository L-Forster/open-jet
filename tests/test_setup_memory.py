from __future__ import annotations

import struct
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import src.setup as setup_source

from src.setup_memory import (
    _amd_free_vram_mb,
    _detect_free_memory_mb,
    _kv_bytes_per_token_from_gguf,
    _max_context_tokens_from_gguf,
    _max_tokens_for_memory,
    _model_file_size_mb,
    recommend_setup_context_window,
)


def _build_test_gguf(
    n_embd: int,
    n_head: int,
    n_head_kv: int,
    n_layer: int,
    *,
    arch: str = "llama",
    extras: list[tuple[str, str | int]] | None = None,
) -> bytes:
    """Build a minimal valid GGUF file with the given model architecture params."""
    buf = bytearray()
    buf += b"GGUF"
    buf += struct.pack("<I", 3)  # version
    buf += struct.pack("<Q", 0)  # tensor count
    kvs: list[tuple[str, str | int]] = [
        ("general.architecture", arch),
        (f"{arch}.embedding_length", n_embd),
        (f"{arch}.attention.head_count", n_head),
        (f"{arch}.attention.head_count_kv", n_head_kv),
        (f"{arch}.block_count", n_layer),
    ]
    if extras:
        kvs.extend(extras)
    buf += struct.pack("<Q", len(kvs))
    for key, value in kvs:
        key_bytes = key.encode("utf-8")
        buf += struct.pack("<Q", len(key_bytes))
        buf += key_bytes
        if isinstance(value, str):
            buf += struct.pack("<I", 8)  # STRING
            val_bytes = value.encode("utf-8")
            buf += struct.pack("<Q", len(val_bytes))
            buf += val_bytes
        else:
            buf += struct.pack("<I", 4)  # UINT32
            buf += struct.pack("<I", value)
    return bytes(buf)


class SetupMemoryTests(unittest.TestCase):
    def test_kv_bytes_per_token_from_gguf_llama_8b(self) -> None:
        # 32 layers, 32 heads, 8 KV heads, 4096 embd -> head_dim=128
        # q8_0: 34/32 bytes per element
        # 2 * 32 * 8 * 128 * (34/32) = 69632 bytes/token
        data = _build_test_gguf(n_embd=4096, n_head=32, n_head_kv=8, n_layer=32)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.gguf"
            path.write_bytes(data)
            self.assertEqual(_kv_bytes_per_token_from_gguf(path), 69632.0)

    def test_kv_bytes_per_token_from_gguf_llama_70b(self) -> None:
        # 80 layers, 64 heads, 8 KV heads, 8192 embd -> head_dim=128
        # 2 * 80 * 8 * 128 * (34/32) = 174080 bytes/token
        data = _build_test_gguf(n_embd=8192, n_head=64, n_head_kv=8, n_layer=80)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.gguf"
            path.write_bytes(data)
            self.assertEqual(_kv_bytes_per_token_from_gguf(path), 174080.0)

    def test_kv_bytes_per_token_from_gguf_qwen35_hybrid_attention(self) -> None:
        # Qwen3.5-style hybrid attention only stores growing KV state on every
        # 4th layer. 64 layers -> 16 KV-bearing layers.
        # 16 * 4 * (256 + 256) * (34/32) = 34816 bytes/token
        data = _build_test_gguf(
            n_embd=0,
            n_head=64,
            n_head_kv=4,
            n_layer=64,
            arch="qwen35",
            extras=[
                ("qwen35.attention.key_length", 256),
                ("qwen35.attention.value_length", 256),
                ("qwen35.full_attention_interval", 4),
            ],
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.gguf"
            path.write_bytes(data)
            self.assertEqual(_kv_bytes_per_token_from_gguf(path), 34816.0)

    def test_max_tokens_for_memory(self) -> None:
        # 8 GB with 10% reserve -> 7372.8 MiB usable -> 111025 tokens.
        self.assertEqual(_max_tokens_for_memory(8192.0, 69632), 111025)
        # 1 GB with 10% reserve -> 921.6 MiB usable -> 13878 tokens.
        self.assertEqual(_max_tokens_for_memory(1024.0, 69632), 13878)

    def test_max_tokens_for_memory_clamps_to_model_max_context(self) -> None:
        self.assertEqual(
            _max_tokens_for_memory(24576.0, 34816, max_context_tokens=131072),
            131072,
        )

    def test_max_tokens_for_memory_no_room(self) -> None:
        self.assertEqual(_max_tokens_for_memory(0.0, 69632), 1024)
        self.assertEqual(_max_tokens_for_memory(-500.0, 69632), 1024)

    def test_detect_free_memory_uses_max_free_gpu(self) -> None:
        proc = Mock(
            returncode=0,
            stdout="RTX 4090, 24564, 8123, 16441\nRTX 3080, 10240, 4096, 6144\n",
        )
        with patch("src.setup_memory.shutil.which", return_value="/usr/bin/nvidia-smi"), patch(
            "src.setup_memory.subprocess.run", return_value=proc,
        ):
            self.assertEqual(_detect_free_memory_mb("cuda"), 16441.0)

    def test_model_file_size_mb_reads_real_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp) / "demo.gguf"
            model.write_bytes(b"x" * (3 * 1024 * 1024))
            self.assertEqual(_model_file_size_mb([str(model)]), 3.0)

    def test_amd_free_vram_mb_from_sysfs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            device_dir = Path(tmp) / "card0" / "device"
            device_dir.mkdir(parents=True)
            total_path = device_dir / "mem_info_vram_total"
            used_path = device_dir / "mem_info_vram_used"
            total_path.write_text(str(8 * 1024 * 1024 * 1024), encoding="utf-8")
            used_path.write_text(str(3 * 1024 * 1024 * 1024), encoding="utf-8")
            with patch("src.setup_memory.Path.glob", return_value=[total_path]):
                self.assertEqual(_amd_free_vram_mb(), 5120.0)

    def test_recommend_uses_gguf_kv_cost(self) -> None:
        # 24576 total - 15360 model = 9216 MB remaining
        # Hold back 10% reserve -> 8294.4 MiB usable
        # 8294.4 * 1024 * 1024 / 69632 = 124903
        with patch("src.setup_memory._model_file_size_mb", return_value=15360.0), \
             patch("src.setup_memory._model_gguf_path", return_value=Path("/fake.gguf")), \
             patch("src.setup_memory._kv_bytes_per_token_from_gguf", return_value=69632), \
             patch("src.setup_memory._max_context_tokens_from_gguf", return_value=262144):
            recommended = recommend_setup_context_window(
                runtime="llama_cpp",
                device="cuda",
                fallback_tokens=8192,
                model_refs=["/models/Qwen3.5-9B-Q4_K_M.gguf"],
                total_vram_mb=24576.0,
            )
        self.assertEqual(recommended, 124903)

    def test_recommend_clamps_to_model_max_context(self) -> None:
        with patch("src.setup_memory._model_file_size_mb", return_value=15360.0), \
             patch("src.setup_memory._model_gguf_path", return_value=Path("/fake.gguf")), \
             patch("src.setup_memory._kv_bytes_per_token_from_gguf", return_value=69632), \
             patch("src.setup_memory._max_context_tokens_from_gguf", return_value=32768):
            recommended = recommend_setup_context_window(
                runtime="llama_cpp",
                device="cuda",
                fallback_tokens=8192,
                model_refs=["/models/demo.gguf"],
                total_vram_mb=24576.0,
            )
        self.assertEqual(recommended, 32768)

    def test_max_context_tokens_from_gguf_reads_context_length(self) -> None:
        data = _build_test_gguf(
            n_embd=4096,
            n_head=32,
            n_head_kv=8,
            n_layer=32,
            extras=[("llama.context_length", 131072)],
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.gguf"
            path.write_bytes(data)
            self.assertEqual(_max_context_tokens_from_gguf(path), 131072)

    def test_recommend_returns_fallback_without_gguf(self) -> None:
        with patch("src.setup_memory._model_file_size_mb", return_value=15360.0), \
             patch("src.setup_memory._model_gguf_path", return_value=None):
            recommended = recommend_setup_context_window(
                runtime="llama_cpp",
                device="cuda",
                fallback_tokens=6144,
                model_refs=["/models/model.bin"],
                total_vram_mb=24576.0,
            )
        self.assertEqual(recommended, 6144)

    def test_recommend_returns_fallback_without_model_file(self) -> None:
        recommended = recommend_setup_context_window(
            runtime="llama_cpp",
            device="cuda",
            fallback_tokens=6144,
            model_refs=["/nonexistent/model.gguf"],
            total_vram_mb=24576.0,
        )
        self.assertEqual(recommended, 6144)

    def test_detect_free_memory_uses_amd_for_vulkan(self) -> None:
        with patch("src.setup_memory._amd_free_vram_mb", return_value=6144.0):
            self.assertEqual(_detect_free_memory_mb("vulkan"), 6144.0)

    def test_detect_free_memory_uses_unified_on_darwin(self) -> None:
        snapshot = Mock(available_mb=6144.0)
        with patch("src.setup_memory.sys.platform", "darwin"), patch(
            "src.setup_memory.read_memory_snapshot", return_value=snapshot,
        ):
            self.assertEqual(_detect_free_memory_mb("cpu"), 6144.0)


class SetupSourceIntegrationTests(unittest.TestCase):
    def test_build_recommended_payload_uses_setup_memory_recommendation(self) -> None:
        hardware = setup_source.HardwareInfo(label="CUDA-capable device", total_ram_gb=16.0, has_cuda=True)

        with patch.object(setup_source, "discover_model_files", return_value=["/models/demo.gguf"]), patch.object(
            setup_source, "_discover_llama_server", return_value="/usr/bin/llama-server",
        ), patch.object(
            setup_source, "recommended_gpu_layers", return_value=99,
        ), patch.object(
            setup_source, "recommend_setup_context_window", return_value=6144,
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
        self.assertEqual(call.kwargs["total_vram_mb"], hardware.vram_mb)

    def test_build_recommended_payload_passes_detected_total_vram(self) -> None:
        hardware = setup_source.HardwareInfo(
            label="CUDA-capable device",
            total_ram_gb=32.0,
            has_cuda=True,
            vram_mb=24576.0,
        )

        with patch.object(setup_source, "discover_model_files", return_value=[]), patch.object(
            setup_source, "_discover_llama_server", return_value="/usr/bin/llama-server",
        ), patch.object(
            setup_source, "recommended_gpu_layers", return_value=99,
        ), patch.object(
            setup_source, "recommend_setup_context_window", return_value=12288,
        ) as recommend_ctx:
            payload = setup_source.build_recommended_payload(
                hardware_info=hardware,
                recommended_ctx=6144,
                current_cfg={},
            )

        self.assertEqual(payload["context_window_tokens"], 12288)
        self.assertEqual(recommend_ctx.call_args.kwargs["total_vram_mb"], 24576.0)
