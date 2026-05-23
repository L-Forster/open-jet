from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

from src.hardware import HardwareInfo
from src.provisioning import (
    _llama_cmake_args,
    _path_without_windows_interop_entries,
    _subprocess_env,
    _sync_managed_llama_cpp_checkout,
    ensure_llama_server,
    recommend_direct_model,
)


class ProvisioningTests(unittest.IsolatedAsyncioTestCase):
    def test_recommend_direct_model_prefers_unified_memory_moe_on_metal(self) -> None:
        hardware = HardwareInfo(
            label="Apple M-series",
            total_ram_gb=32.0,
            has_cuda=False,
            has_metal=True,
        )

        direct = recommend_direct_model(hardware)

        self.assertEqual(direct["label"], "Qwen3.6 35B A3B UD-Q3_K_XL MTP")
        self.assertIn("unsloth/Qwen3.6-35B-A3B-MTP-GGUF", direct["url"])
        self.assertTrue(direct["llama_mtp"])
        self.assertTrue(direct["llama_cpu_moe"])

    def test_recommend_direct_model_uses_q3_moe_when_unified_memory_needs_system_reserve(self) -> None:
        hardware = HardwareInfo(
            label="Apple M-series",
            total_ram_gb=24.0,
            has_cuda=False,
            has_metal=True,
        )

        direct = recommend_direct_model(hardware)

        self.assertEqual(direct["label"], "Qwen3.6 35B A3B UD-IQ2_XXS MTP")
        self.assertEqual(direct["filename"], "Qwen3.6-35B-A3B-UD-IQ2_XXS-MTP.gguf")
        self.assertIn("unsloth/Qwen3.6-35B-A3B-MTP-GGUF", direct["url"])
        self.assertTrue(direct["llama_mtp"])
        self.assertTrue(direct["llama_cpu_moe"])

    def test_llama_cmake_args_honor_selected_vulkan_on_cuda_host(self) -> None:
        hardware = HardwareInfo(
            label="RTX 3090",
            total_ram_gb=64.0,
            has_cuda=True,
            has_vulkan=True,
            vram_mb=24576.0,
        )

        args = _llama_cmake_args(hardware, device="vulkan")

        self.assertIn("-DGGML_VULKAN=ON", args)
        self.assertNotIn("-DGGML_CUDA=ON", args)

    def test_llama_cmake_args_falls_back_to_cuda_when_vulkan_selected_without_vulkan(self) -> None:
        hardware = HardwareInfo(
            label="RTX 3090",
            total_ram_gb=64.0,
            has_cuda=True,
            has_vulkan=False,
            vram_mb=24576.0,
        )

        args = _llama_cmake_args(hardware, device="vulkan")

        self.assertNotIn("-DGGML_VULKAN=ON", args)
        self.assertIn("-DGGML_VULKAN=OFF", args)
        self.assertIn("-DGGML_CUDA=ON", args)

    def test_wsl_subprocess_env_drops_windows_path_entries(self) -> None:
        original_path = "/usr/local/bin:/mnt/c/Program Files/nodejs:/usr/bin:/mnt/d/CUDA/bin"

        cleaned = _path_without_windows_interop_entries(original_path)

        self.assertEqual(cleaned, "/usr/local/bin:/usr/bin")

        with patch("src.provisioning._running_under_wsl", return_value=True), patch.dict(
            os.environ,
            {"PATH": original_path},
        ):
            env = _subprocess_env()

        self.assertEqual(env["PATH"], "/usr/local/bin:/usr/bin")

    async def test_sync_managed_llama_cpp_checkout_uses_shallow_retry_fetch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            llama_dir = Path(tmp) / "llama.cpp"
            calls: list[tuple[str, ...]] = []
            fetch_attempts = 0

            async def fake_run_exec(*args: str, cwd: Path | None = None):
                nonlocal fetch_attempts
                calls.append(args)
                if "fetch" in args:
                    fetch_attempts += 1
                    if fetch_attempts == 1:
                        return 1, "", "RPC failed; curl 92 HTTP/2 stream was not closed cleanly\nfatal: early EOF"
                return 0, "", ""

            with patch("src.provisioning.LLAMA_CPP_DIR", llama_dir), patch(
                "src.provisioning._run_exec",
                AsyncMock(side_effect=fake_run_exec),
            ), patch("src.provisioning.asyncio.sleep", AsyncMock()):
                synced = await _sync_managed_llama_cpp_checkout(
                    target_ref="b9189",
                    log=Mock(),
                    set_status=lambda _message: None,
                    clear_status=lambda: None,
                )

        self.assertEqual(synced, "b9189")
        self.assertFalse(any("clone" in call for call in calls))
        fetch_calls = [call for call in calls if "fetch" in call]
        self.assertEqual(len(fetch_calls), 2)
        self.assertIn("http.version=HTTP/1.1", fetch_calls[0])
        self.assertIn("--depth=1", fetch_calls[0])
        self.assertEqual(fetch_calls[0][-2:], ("origin", "b9189"))
        self.assertTrue(any(call[-2:] == ("--detach", "FETCH_HEAD") for call in calls))

    async def test_ensure_llama_server_installs_prebuilt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            llama_dir = Path(tmp) / "llama.cpp"
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

            with patch("src.provisioning.LLAMA_CPP_DIR", llama_dir), patch(
                "src.provisioning.BIN_DIR", bin_dir
            ), patch(
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

            async def fake_build_progress(*_args, **_kwargs):
                built.parent.mkdir(parents=True, exist_ok=True)
                built.write_text("binary", encoding="utf-8")
                return 0, ""

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
            ), patch(
                "src.provisioning._run_build_with_progress",
                AsyncMock(side_effect=fake_build_progress),
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

    async def test_ensure_llama_server_builds_mtp_runtime_for_mtp_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            llama_dir = Path(tmp) / "llama.cpp"
            built = llama_dir / "build" / "bin" / "llama-server"
            built.parent.mkdir(parents=True)
            built.write_text("binary", encoding="utf-8")
            bin_dir = Path(tmp) / "bin"
            old_prebuilt = bin_dir / "llama-server"
            old_prebuilt.parent.mkdir(parents=True)
            old_prebuilt.write_text("old", encoding="utf-8")
            tag_file = bin_dir / "llama-server.tag"
            tag_file.write_text("b9999", encoding="utf-8")

            log = Mock()
            hardware = HardwareInfo(label="RTX 3090", total_ram_gb=64.0, has_cuda=True, vram_mb=24576.0)

            async def fake_build_progress(*_args, **_kwargs):
                built.parent.mkdir(parents=True, exist_ok=True)
                built.write_text("binary", encoding="utf-8")
                return 0, ""

            with patch("src.provisioning.LLAMA_CPP_DIR", llama_dir), patch(
                "src.provisioning.BIN_DIR", bin_dir
            ), patch(
                "src.provisioning.LLAMA_SERVER_BIN", old_prebuilt
            ), patch(
                "src.provisioning.LLAMA_CPP_TAG_FILE", tag_file
            ), patch(
                "src.provisioning.current_llama_server_path", return_value="/usr/bin/llama-server"
            ), patch(
                "src.provisioning._install_prebuilt_llama_server",
                AsyncMock(),
            ) as install_prebuilt, patch(
                "src.provisioning._sync_managed_llama_cpp_checkout",
                AsyncMock(return_value="b9189"),
            ) as sync_checkout, patch(
                "src.provisioning._run_exec",
                AsyncMock(return_value=(0, "", "")),
            ), patch(
                "src.provisioning._run_build_with_progress",
                AsyncMock(side_effect=fake_build_progress),
            ):
                payload = await ensure_llama_server(
                    {
                        "llama_cpp_ref": "b9072",
                        "llama_model": "/models/Qwen3.6-27B-Q4_K_M-MTP.gguf",
                        "llama_mtp": True,
                    },
                    hardware_info=hardware,
                    log=log,
                    set_status=lambda _message: None,
                    clear_status=lambda: None,
                )

        self.assertEqual(payload["llama_cpp_ref"], "b9189")
        self.assertEqual(payload["llama_server_path"], str(built))
        install_prebuilt.assert_not_awaited()
        self.assertEqual(sync_checkout.await_args.kwargs["target_ref"], "b9189")

    async def test_ensure_llama_server_reuses_configured_runtime_path_for_mtp_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            configured = Path(tmp) / "custom" / "llama-server"
            configured.parent.mkdir(parents=True)
            configured.write_text("binary", encoding="utf-8")
            llama_dir = Path(tmp) / "llama.cpp"
            bin_dir = Path(tmp) / "bin"
            log = Mock()
            hardware = HardwareInfo(label="RTX 5090", total_ram_gb=64.0, has_cuda=True, vram_mb=32768.0)

            with patch("src.provisioning.LLAMA_CPP_DIR", llama_dir), patch(
                "src.provisioning.BIN_DIR", bin_dir
            ), patch(
                "src.provisioning.LLAMA_SERVER_BIN", bin_dir / "llama-server"
            ), patch(
                "src.provisioning.LLAMA_CPP_TAG_FILE", bin_dir / "llama-server.tag"
            ), patch(
                "src.provisioning._needs_rebuild", return_value=True
            ), patch(
                "src.provisioning.current_llama_server_path", return_value=None
            ), patch(
                "src.provisioning._install_prebuilt_llama_server",
                AsyncMock(),
            ) as install_prebuilt, patch(
                "src.provisioning._build_llama_server_from_source",
                AsyncMock(),
            ) as build_source:
                payload = await ensure_llama_server(
                    {
                        "device": "cuda",
                        "llama_server_path": str(configured),
                        "llama_model": "/models/Qwen3.6-27B-Q4_K_M-MTP.gguf",
                        "llama_mtp": True,
                    },
                    hardware_info=hardware,
                    log=log,
                    set_status=lambda _message: None,
                    clear_status=lambda: None,
                )

        self.assertEqual(payload["llama_cpp_ref"], "b9189")
        self.assertEqual(payload["llama_server_path"], str(configured))
        self.assertEqual(payload["device"], "cuda")
        install_prebuilt.assert_not_awaited()
        build_source.assert_not_awaited()

    async def test_ensure_llama_server_builds_mtp_with_vulkan_on_linux_nvidia(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            llama_dir = Path(tmp) / "llama.cpp"
            built = llama_dir / "build" / "bin" / "llama-server"
            bin_dir = Path(tmp) / "bin"

            log = Mock()
            hardware = HardwareInfo(label="RTX 5090", total_ram_gb=64.0, has_cuda=True, vram_mb=32768.0)

            with patch("src.provisioning.sys.platform", "linux"), patch(
                "src.provisioning.platform.machine", return_value="x86_64"
            ), patch("src.provisioning.LLAMA_CPP_DIR", llama_dir), patch(
                "src.provisioning.BIN_DIR", bin_dir
            ), patch(
                "src.provisioning.LLAMA_SERVER_BIN", bin_dir / "llama-server"
            ), patch(
                "src.provisioning.LLAMA_CPP_TAG_FILE", bin_dir / "llama-server.tag"
            ), patch(
                "src.provisioning.current_llama_server_path", return_value=None
            ), patch(
                "src.provisioning._install_prebuilt_llama_server",
                AsyncMock(),
            ) as install_prebuilt, patch(
                "src.provisioning._build_llama_server_from_source",
                AsyncMock(return_value=(built, "b9189")),
            ) as build_source:
                payload = await ensure_llama_server(
                    {
                        "device": "cuda",
                        "llama_model": "/models/Qwen3.6-27B-Q4_K_M-MTP.gguf",
                        "llama_mtp": True,
                    },
                    hardware_info=hardware,
                    log=log,
                    set_status=lambda _message: None,
                    clear_status=lambda: None,
                )

        self.assertEqual(payload["llama_cpp_ref"], "b9189")
        self.assertEqual(payload["llama_server_path"], str(built))
        self.assertEqual(payload["device"], "vulkan")
        install_prebuilt.assert_not_awaited()
        self.assertEqual(build_source.await_args.kwargs["device"], "vulkan")

    async def test_ensure_llama_server_reuses_matching_source_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            llama_dir = Path(tmp) / "llama.cpp"
            built = llama_dir / "build" / "bin" / "llama-server"
            built.parent.mkdir(parents=True)
            built.write_text("binary", encoding="utf-8")
            (llama_dir / "build" / "openjet-llama-server.json").write_text(
                '{"device": "vulkan", "ref": "b9189"}',
                encoding="utf-8",
            )
            bin_dir = Path(tmp) / "bin"
            log = Mock()
            hardware = HardwareInfo(label="RTX 3090", total_ram_gb=64.0, has_cuda=True, vram_mb=24576.0)

            with patch("src.provisioning.LLAMA_CPP_DIR", llama_dir), patch(
                "src.provisioning.BIN_DIR", bin_dir
            ), patch(
                "src.provisioning.LLAMA_SERVER_BIN", bin_dir / "llama-server"
            ), patch(
                "src.provisioning.LLAMA_CPP_TAG_FILE", bin_dir / "llama-server.tag"
            ), patch(
                "src.provisioning._needs_rebuild", return_value=False
            ), patch(
                "src.provisioning.current_llama_server_path", return_value="/usr/bin/llama-server"
            ), patch(
                "src.provisioning._install_prebuilt_llama_server",
                AsyncMock(),
            ) as install_prebuilt, patch(
                "src.provisioning._build_llama_server_from_source",
                AsyncMock(),
            ) as build_source:
                payload = await ensure_llama_server(
                    {
                        "device": "cuda",
                        "llama_model": "/models/Qwen3.6-27B-Q4_K_M-MTP.gguf",
                        "llama_mtp": True,
                    },
                    hardware_info=hardware,
                    log=log,
                    set_status=lambda _message: None,
                    clear_status=lambda: None,
                )

        self.assertEqual(payload["llama_cpp_ref"], "b9189")
        self.assertEqual(payload["llama_server_path"], str(built))
        self.assertEqual(payload["device"], "vulkan")
        install_prebuilt.assert_not_awaited()
        build_source.assert_not_awaited()

    async def test_ensure_llama_server_reuses_matching_cuda_source_runtime_on_linux_nvidia(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            llama_dir = Path(tmp) / "llama.cpp"
            built = llama_dir / "build" / "bin" / "llama-server"
            built.parent.mkdir(parents=True)
            built.write_text("binary", encoding="utf-8")
            (llama_dir / "build" / "openjet-llama-server.json").write_text(
                '{"device": "cuda", "ref": "b9189"}',
                encoding="utf-8",
            )
            bin_dir = Path(tmp) / "bin"
            log = Mock()
            hardware = HardwareInfo(label="RTX 5090", total_ram_gb=64.0, has_cuda=True, vram_mb=32768.0)

            with patch("src.provisioning.sys.platform", "linux"), patch(
                "src.provisioning.platform.machine", return_value="x86_64"
            ), patch("src.provisioning.LLAMA_CPP_DIR", llama_dir), patch(
                "src.provisioning.BIN_DIR", bin_dir
            ), patch(
                "src.provisioning.LLAMA_SERVER_BIN", bin_dir / "llama-server"
            ), patch(
                "src.provisioning.LLAMA_CPP_TAG_FILE", bin_dir / "llama-server.tag"
            ), patch(
                "src.provisioning._needs_rebuild", return_value=True
            ), patch(
                "src.provisioning.current_llama_server_path", return_value=str(built)
            ), patch(
                "src.provisioning._install_prebuilt_llama_server",
                AsyncMock(),
            ) as install_prebuilt, patch(
                "src.provisioning._build_llama_server_from_source",
                AsyncMock(),
            ) as build_source:
                payload = await ensure_llama_server(
                    {
                        "device": "cuda",
                        "llama_model": "/models/Qwen3.6-27B-Q4_K_M-MTP.gguf",
                        "llama_mtp": True,
                    },
                    hardware_info=hardware,
                    log=log,
                    set_status=lambda _message: None,
                    clear_status=lambda: None,
                )

        self.assertEqual(payload["llama_cpp_ref"], "b9189")
        self.assertEqual(payload["llama_server_path"], str(built))
        self.assertEqual(payload["device"], "cuda")
        install_prebuilt.assert_not_awaited()
        build_source.assert_not_awaited()

    async def test_ensure_llama_server_reuses_source_runtime_when_checkout_ref_matches_without_tag(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            llama_dir = Path(tmp) / "llama.cpp"
            built = llama_dir / "build" / "bin" / "llama-server"
            built.parent.mkdir(parents=True)
            built.write_text("binary", encoding="utf-8")
            bin_dir = Path(tmp) / "bin"
            log = Mock()
            hardware = HardwareInfo(label="RTX 3090", total_ram_gb=64.0, has_cuda=True, vram_mb=24576.0)

            with patch("src.provisioning.LLAMA_CPP_DIR", llama_dir), patch(
                "src.provisioning.BIN_DIR", bin_dir
            ), patch(
                "src.provisioning.LLAMA_SERVER_BIN", bin_dir / "llama-server"
            ), patch(
                "src.provisioning.LLAMA_CPP_TAG_FILE", bin_dir / "llama-server.tag"
            ), patch(
                "src.provisioning._source_checkout_ref_matches", return_value=True
            ), patch(
                "src.provisioning._needs_rebuild", return_value=True
            ), patch(
                "src.provisioning.current_llama_server_path", return_value=str(built)
            ), patch(
                "src.provisioning._install_prebuilt_llama_server",
                AsyncMock(),
            ) as install_prebuilt, patch(
                "src.provisioning._build_llama_server_from_source",
                AsyncMock(),
            ) as build_source:
                payload = await ensure_llama_server(
                    {
                        "device": "cuda",
                        "llama_model": "/models/Qwen3.6-27B-Q4_K_M-MTP.gguf",
                        "llama_mtp": True,
                    },
                    hardware_info=hardware,
                    log=log,
                    set_status=lambda _message: None,
                    clear_status=lambda: None,
                )

        self.assertEqual(payload["llama_cpp_ref"], "b9189")
        self.assertEqual(payload["llama_server_path"], str(built))
        self.assertEqual(payload["device"], "cuda")
        install_prebuilt.assert_not_awaited()
        build_source.assert_not_awaited()
