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
