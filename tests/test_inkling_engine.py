"""OpenJet's Inkling engine selection: architecture detection, expert size, VRAM budget (synthetic GGUF headers)."""

import struct
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from src import llama_server as ls


def write_gguf(path: Path, arch: str, tensors: list[tuple[str, list[int], int]]) -> None:
    """A GGUF header (no tensor data): general.architecture, then (name, dims, ggml type) tensor infos."""
    def s(x: str) -> bytes:
        b = x.encode()
        return struct.pack("<Q", len(b)) + b

    out = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), 1)
    out += s("general.architecture") + struct.pack("<I", 8) + s(arch)
    for name, dims, ttype in tensors:
        out += s(name) + struct.pack("<I", len(dims)) + b"".join(struct.pack("<Q", d) for d in dims)
        out += struct.pack("<IQ", ttype, 0)
    path.write_bytes(out)


# Inkling 2 Mini's experts: layers 0-19 NVFP4 (ggml type 40), layers 20-23 Q8_0 (type 8)
TENSORS = [(f"blk.{il}.ffn_{part}_exps.weight", [n, 2304, 128] if part == "down" else [2304, 2304, 128], 40 if il < 20 else 8)
           for il in range(24) for part, n in (("gate_up", 0), ("down", 1152))] + [("blk.0.attn_q.weight", [2304, 3072], 8)]
NVFP4_MIB = 20 * 2304 * 3456 * 128 // 64 * 36 / 2**20
Q8_MIB = 4 * 2304 * 3456 * 128 // 32 * 34 / 2**20


class InklingSelectionTest(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.model = Path(self.tmp.name) / "Inkling-2-Mini-UD-IQ4_XS.gguf"
        write_gguf(self.model, "inkling2_loop", TENSORS)

    def tearDown(self):
        self.tmp.cleanup()

    def test_architecture(self):
        self.assertEqual(ls.gguf_architecture(self.model), "inkling2_loop")
        self.assertTrue(ls.inkling_model(str(self.model)))
        other = Path(self.tmp.name) / "qwen.gguf"
        write_gguf(other, "qwen38", [])
        self.assertFalse(ls.inkling_model(str(other)))
        self.assertFalse(ls.inkling_model(str(Path(self.tmp.name) / "missing.gguf")))

    def test_expert_mib_counts_only_experts(self):
        self.assertAlmostEqual(ls.inkling_expert_mib(str(self.model)), NVFP4_MIB + Q8_MIB, places=3)

    def test_budget(self):
        experts = NVFP4_MIB + Q8_MIB
        need = ls._INKLING_BASE_MIB + 8192 * ls._INKLING_KV_MIB_PER_TOKEN + ls._INKLING_DESKTOP_MIB
        # everything fits: no cache
        self.assertIsNone(ls.inkling_expert_budget_mib(str(self.model), 8192, need + experts + 1))
        # part fits: the room left is the cache
        self.assertEqual(ls.inkling_expert_budget_mib(str(self.model), 8192, need + 3000), 3000)
        # longer context leaves less room
        self.assertLess(ls.inkling_expert_budget_mib(str(self.model), 65536, need + 7000), 7000)
        # too little for a useful cache
        self.assertEqual(ls.inkling_expert_budget_mib(str(self.model), 8192, need + 100), 0)

    def test_ram_cache(self):
        experts = NVFP4_MIB + Q8_MIB
        cache = 5000
        cpu_side = experts - cache
        # the CPU-side experts fit in RAM with room to spare: the page cache serves them
        self.assertIsNone(ls.inkling_ram_cache_mib(str(self.model), cache, cpu_side + ls._INKLING_RAM_SPARE_MIB + 1))
        # they don't: a RAM cache of what is left after the spare
        self.assertEqual(ls.inkling_ram_cache_mib(str(self.model), cache, 8000), 8000 - ls._INKLING_RAM_SPARE_MIB)
        # very little RAM: at least 1 GiB of cache
        self.assertEqual(ls.inkling_ram_cache_mib(str(self.model), cache, 3500), 1024)

    def test_unavailable_reasons(self):
        with mock.patch.object(ls, "_inkling_dir", return_value=None):
            self.assertIn("not built", ls.inkling_unavailable_reason(str(self.model)))
        with mock.patch.object(ls, "_inkling_dir", return_value=Path(self.tmp.name)), \
                mock.patch.object(ls, "_inkling_gpu", return_value=None):
            self.assertIn("sm_86", ls.inkling_unavailable_reason(str(self.model)))
        with mock.patch.object(ls, "_inkling_dir", return_value=Path(self.tmp.name)), \
                mock.patch.object(ls, "_inkling_gpu", return_value=("0", 8000.0)):
            self.assertIsNone(ls.inkling_unavailable_reason(str(self.model)))
            self.assertEqual(ls.model_runtime_label(str(self.model)), ls.INKLING_RUNTIME_LABEL)

    def test_iq3_xxs_expert_mib_and_budget(self):
        iq3_model = Path(self.tmp.name) / "Inkling-2-Mini-UD-IQ3_XXS-mainline.gguf"
        iq3_tensors = [
            ("blk.0.ffn_gate_up_exps.weight", [2304, 2304, 128], 16),   # IQ2_XXS
            ("blk.4.ffn_gate_up_exps.weight", [2304, 2304, 128], 17),   # IQ2_XS
            ("blk.6.ffn_gate_up_exps.weight", [2304, 2304, 128], 22),   # IQ2_S
            ("blk.1.ffn_gate_up_exps.weight", [2304, 2304, 128], 18),   # IQ3_XXS
        ] + [
            (f"blk.{il}.ffn_gate_up_exps.weight", [2304, 2304, 128], 18) for il in range(2, 24) if il not in (4, 6)
        ] + [
            (f"blk.{il}.ffn_down_exps.weight", [1152, 2304, 128], 40 if il < 20 else 20) for il in range(24)
        ]
        write_gguf(iq3_model, "inkling2_loop", iq3_tensors)
        mib = ls.inkling_expert_mib(str(iq3_model))
        self.assertGreater(mib, 9500.0)
        self.assertLess(mib, 10500.0)
        # 6 GB and 8 GB GPUs must get an offload cache budget, not None (which would try to fit all experts in VRAM)
        self.assertGreater(ls.inkling_expert_budget_mib(str(iq3_model), 8192, 6000.0), 2500)
        self.assertGreater(ls.inkling_expert_budget_mib(str(iq3_model), 8192, 8000.0), 4500)


if __name__ == "__main__":
    unittest.main()
