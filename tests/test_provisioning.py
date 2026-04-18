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

    async def test_ensure_llama_server_records_pinned_llama_cpp_ref(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            llama_dir = Path(tmp) / "llama.cpp"
            built = llama_dir / "build" / "bin" / "llama-server"
            built.parent.mkdir(parents=True)
            built.write_text("binary", encoding="utf-8")

            log = Mock()
            hardware = HardwareInfo(label="CPU-only device", total_ram_gb=16.0, has_cuda=False)

            with patch("src.provisioning.LLAMA_CPP_DIR", llama_dir), patch(
                "src.provisioning.current_llama_server_path",
                return_value=None,
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
