from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

from src.hardware import HardwareInfo
from src.provisioning import ensure_llama_server, recommend_direct_model


class ProvisioningTests(unittest.IsolatedAsyncioTestCase):
    def test_recommend_direct_model_prefers_unified_memory_moe_on_metal(self) -> None:
        hardware = HardwareInfo(
            label="Apple M-series",
            total_ram_gb=32.0,
            has_cuda=False,
            has_metal=True,
        )

        direct = recommend_direct_model(hardware)

        self.assertEqual(direct["label"], "Qwen3.6 35B A3B")

    async def test_ensure_llama_server_installs_prebuilt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bin_dir = Path(tmp) / "bin"
            installed = bin_dir / "llama-server"
            tag_file = bin_dir / "llama-server.tag"

            async def fake_install_prebuilt(*_args, **_kwargs):
                bin_dir.mkdir(parents=True, exist_ok=True)
                installed.write_text("binary", encoding="utf-8")
                tag_file.write_text("b8838", encoding="utf-8")
                return (installed, "b8838")

            log = Mock()
            hardware = HardwareInfo(label="CPU-only device", total_ram_gb=16.0, has_cuda=False)

            with patch("src.provisioning.BIN_DIR", bin_dir), patch(
                "src.provisioning.LLAMA_SERVER_BIN", installed
            ), patch(
                "src.provisioning.LLAMA_CPP_TAG_FILE", tag_file
            ), patch(
                "src.provisioning.current_llama_server_path", return_value=None
            ), patch(
                "src.provisioning._install_prebuilt_llama_server",
                AsyncMock(side_effect=fake_install_prebuilt),
            ) as install_prebuilt, patch(
                "src.provisioning._build_llama_server_from_source",
                AsyncMock(),
            ) as build_source:
                payload = await ensure_llama_server(
                    {},
                    hardware_info=hardware,
                    log=log,
                    set_status=lambda _message: None,
                    clear_status=lambda: None,
                )

        self.assertEqual(payload["llama_cpp_ref"], "b8838")
        self.assertEqual(payload["llama_server_path"], str(installed))
        self.assertFalse(payload["setup_missing_runtime"])
        install_prebuilt.assert_called_once()
        build_source.assert_not_awaited()

    async def test_ensure_llama_server_falls_back_to_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            llama_dir = Path(tmp) / "llama.cpp"
            built = llama_dir / "build" / "bin" / "llama-server"
            built.parent.mkdir(parents=True)
            built.write_text("binary", encoding="utf-8")
            bin_dir = Path(tmp) / "bin"

            log = Mock()
            hardware = HardwareInfo(label="Jetson Orin Nano", total_ram_gb=8.0, has_cuda=True)

            with patch("src.provisioning.LLAMA_CPP_DIR", llama_dir), patch(
                "src.provisioning.BIN_DIR", bin_dir
            ), patch(
                "src.provisioning.LLAMA_SERVER_BIN", bin_dir / "llama-server"
            ), patch(
                "src.provisioning.LLAMA_CPP_TAG_FILE", bin_dir / "llama-server.tag"
            ), patch(
                "src.provisioning.current_llama_server_path", return_value=None
            ), patch(
                "src.provisioning._install_prebuilt_llama_server",
                AsyncMock(return_value=None),
            ), patch(
                "src.provisioning._sync_managed_llama_cpp_checkout",
                AsyncMock(return_value="deadbee"),
            ) as sync_checkout, patch(
                "src.provisioning._run_exec",
                AsyncMock(return_value=(0, "", "")),
            ):
                payload = await ensure_llama_server(
                    {},
                    hardware_info=hardware,
                    log=log,
                    set_status=lambda _message: None,
                    clear_status=lambda: None,
                )

        self.assertEqual(payload["llama_cpp_ref"], "deadbee")
        self.assertEqual(payload["llama_server_path"], str(built))
        self.assertFalse(payload["setup_missing_runtime"])
        sync_checkout.assert_awaited_once()
