from __future__ import annotations

import unittest
from unittest.mock import patch

from src import hardware


class HardwareDetectionTests(unittest.TestCase):
    def test_detect_metal_handles_apple_silicon_under_rosetta(self) -> None:
        def fake_sysctl(name: str) -> str:
            if name == "hw.optional.arm64":
                return "1"
            return ""

        with patch("src.hardware.sys.platform", "darwin"), patch(
            "src.hardware.platform.machine", return_value="x86_64"
        ), patch("src.hardware._darwin_sysctl", side_effect=fake_sysctl):
            self.assertTrue(hardware._detect_metal())

    def test_detect_metal_rejects_intel_mac_without_arm64_flag(self) -> None:
        def fake_sysctl(name: str) -> str:
            if name == "machdep.cpu.brand_string":
                return "Intel(R) Core(TM) i9"
            return "0"

        with patch("src.hardware.sys.platform", "darwin"), patch(
            "src.hardware.platform.machine", return_value="x86_64"
        ), patch("src.hardware._darwin_sysctl", side_effect=fake_sysctl):
            self.assertFalse(hardware._detect_metal())

    def test_detect_vulkan_finds_windows_system32_dll(self) -> None:
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            dll = Path(tmp) / "System32" / "vulkan-1.dll"
            dll.parent.mkdir(parents=True)
            dll.write_bytes(b"")
            with patch("src.hardware.sys.platform", "win32"), patch(
                "src.hardware.shutil.which", return_value=None
            ), patch.dict("src.hardware.os.environ", {"SystemRoot": tmp}):
                self.assertTrue(hardware._detect_vulkan())

    def test_read_windows_vulkan_vram_uses_wddm_shared_heap_for_radeon_780m_igpu(self) -> None:
        from types import SimpleNamespace

        fake_mem = SimpleNamespace(total_mb=16384.0, available_mb=10240.0)
        fake_proc = SimpleNamespace(
            returncode=0,
            stdout='{"DriverDesc":"AMD Radeon(TM) 780M Graphics","Mem":536870912}',
        )
        with patch("src.hardware.sys.platform", "win32"), patch(
            "src.hardware.shutil.which", return_value=None
        ), patch("src.hardware.read_memory_snapshot", return_value=fake_mem), patch(
            "src.hardware.subprocess.run", return_value=fake_proc
        ):
            self.assertAlmostEqual(hardware._read_vulkan_vram_mb(), 8192.0, places=1)

    def test_read_windows_vulkan_vram_keeps_discrete_gpu_vram(self) -> None:
        from types import SimpleNamespace

        fake_mem = SimpleNamespace(total_mb=32768.0, available_mb=24000.0)
        fake_proc = SimpleNamespace(
            returncode=0,
            stdout='{"DriverDesc":"AMD Radeon RX 7800 XT","Mem":17179869184}',
        )
        with patch("src.hardware.sys.platform", "win32"), patch(
            "src.hardware.shutil.which", return_value=None
        ), patch("src.hardware.read_memory_snapshot", return_value=fake_mem), patch(
            "src.hardware.subprocess.run", return_value=fake_proc
        ):
            self.assertAlmostEqual(hardware._read_vulkan_vram_mb(), 16384.0, places=1)

    def test_windows_16gb_radeon_igpu_recommends_vulkan_and_inkling_iq3(self) -> None:
        from src import llama_server, provisioning, setup

        hw = hardware.HardwareInfo(
            label="Vulkan-capable device",
            total_ram_gb=16.0,
            has_cuda=False,
            has_vulkan=True,
            vram_mb=8192.0,
        )
        rec = provisioning.recommend_direct_model(hw)
        self.assertEqual(rec["label"], "Inkling 2 Mini UD-IQ3_XXS")
        self.assertTrue(rec["llama_cpu_moe"])
        self.assertEqual(rec["llama_cpp_ref"], "b12651")
        self.assertEqual(hardware.recommended_device_for_hardware("auto", hw), "vulkan")
        with patch("src.llama_server._inkling_gpu", return_value=None):
            self.assertEqual(llama_server.model_runtime_label(rec["filename"]), "llama.cpp")
        with patch("src.llama_server._inkling_gpu", return_value=("0", 12000.0)):
            self.assertEqual(llama_server.model_runtime_label(rec["filename"]), llama_server.INKLING_RUNTIME_LABEL)
        with patch("src.provisioning.sys.platform", "win32"), patch(
            "src.provisioning.platform.machine", return_value="AMD64"
        ):
            self.assertEqual(provisioning._prebuilt_asset_candidates(hw), ["bin-win-vulkan-x64"])
        with patch("src.setup.discover_model_files", return_value=[]), patch(
            "src.setup._discover_llama_server", return_value=None
        ):
            payload = setup.build_recommended_payload(hardware_info=hw, recommended_ctx=0)
        self.assertEqual(payload["device"], "vulkan")
        self.assertTrue(str(payload["model_download_path"]).endswith("Inkling-2-Mini-UD-IQ3_XXS-mainline.gguf"))
        self.assertTrue(payload["llama_cpu_moe"])
        self.assertEqual(payload["llama_cpp_ref"], "b12651")
        self.assertIn("Mainline_Support/Inkling-2-Mini-UD-IQ3_XXS.gguf_file", str(payload["model_download_url"]))
        self.assertGreaterEqual(int(payload["context_window_tokens"]), 32768)

    def test_nvidia_6_to_12gb_picks_iq3_on_tight_vram_or_ram_and_iq4_otherwise(self) -> None:
        from src import llama_server, provisioning

        rtx4050_6gb = hardware.HardwareInfo(label="RTX 4050", total_ram_gb=16.0, has_cuda=True, vram_mb=6144.0)
        rtx4060_16gb_ram = hardware.HardwareInfo(label="RTX 4060 Laptop", total_ram_gb=16.0, has_cuda=True, vram_mb=8192.0)
        rtx4060_32gb_ram = hardware.HardwareInfo(label="RTX 4060", total_ram_gb=32.0, has_cuda=True, vram_mb=8192.0)
        rtx3060_12gb = hardware.HardwareInfo(label="RTX 3060", total_ram_gb=16.0, has_cuda=True, vram_mb=12288.0)

        rec_4050 = provisioning.recommend_direct_model(rtx4050_6gb)
        rec_4060_16 = provisioning.recommend_direct_model(rtx4060_16gb_ram)
        rec_4060_32 = provisioning.recommend_direct_model(rtx4060_32gb_ram)
        rec_3060 = provisioning.recommend_direct_model(rtx3060_12gb)

        self.assertEqual(rec_4050["label"], "Inkling 2 Mini UD-IQ3_XXS")
        self.assertEqual(rec_4060_16["label"], "Inkling 2 Mini UD-IQ3_XXS")
        self.assertEqual(rec_4060_32["label"], "Inkling 2 Mini UD-IQ4_XS")
        self.assertEqual(rec_3060["label"], "Inkling 2 Mini UD-IQ4_XS")
        with patch("src.llama_server._inkling_gpu", return_value=("0", 6000.0)):
            self.assertEqual(llama_server.model_runtime_label(rec_4050["filename"]), llama_server.INKLING_RUNTIME_LABEL)

    def test_16gb_laptop_with_uma_carveout_13_3gb_ram_picks_inkling_iq3(self) -> None:
        from src import provisioning

        # Real 16 GB AMD/Intel laptop with ~2.7 GB BIOS UMA carve-out (13.3 GB OS RAM):
        # works both when VRAM reports the 2.7 GB UMA carve-out and when WDDM/GTT reports ~6.65 GB.
        for vram_mb in (2700.0, 6810.0):
            hw = hardware.HardwareInfo(
                label="Vulkan-capable device",
                total_ram_gb=13.3,
                has_cuda=False,
                has_vulkan=True,
                vram_mb=vram_mb,
            )
            rec = provisioning.recommend_direct_model(hw)
            self.assertEqual(rec["label"], "Inkling 2 Mini UD-IQ3_XXS")
            self.assertTrue(rec["llama_cpu_moe"])

        # 8 GB laptop must NOT pick Inkling (10.9 GiB won't fit)
        hw_8gb = hardware.HardwareInfo(
            label="Vulkan-capable device",
            total_ram_gb=7.5,
            has_cuda=False,
            has_vulkan=True,
            vram_mb=3840.0,
        )
        self.assertEqual(provisioning.recommend_direct_model(hw_8gb)["label"], "Qwen3.5 4B")

    def test_windows_vulkan_vram_boosts_small_vulkaninfo_uma_heap_and_reg_binary(self) -> None:
        from types import SimpleNamespace

        # 2 GB little-endian REG_BINARY (0x80000000) on an Arc 140V / Radeon iGPU with 13.3 GB OS RAM
        fake_mem = SimpleNamespace(total_mb=13619.0, available_mb=10900.0)
        fake_vk = SimpleNamespace(returncode=0, stdout="memoryHeaps[0]:\n  size = 2147483648\n  flags = MEMORY_HEAP_DEVICE_LOCAL_BIT\n", stderr="")
        fake_ps = SimpleNamespace(
            returncode=0,
            stdout='{"DriverDesc":"Intel(R) Arc(TM) 140V GPU","Mem":[0,0,0,128,0,0,0,0]}',
        )
        with patch("src.hardware.sys.platform", "win32"), patch(
            "src.hardware.shutil.which", return_value="vulkaninfo"
        ), patch("src.hardware.read_memory_snapshot", return_value=fake_mem), patch(
            "src.hardware.subprocess.run", side_effect=[fake_vk, fake_ps]
        ):
            self.assertAlmostEqual(hardware._read_vulkan_vram_mb(), 13619.0 * 0.5, places=1)
