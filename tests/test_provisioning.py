from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

from src.hardware import HardwareInfo
from src.provisioning import (
    LLAMA_CPP_MTP_REF,
    LLAMA_SERVER_EXE_NAME,
    _clamp_context_for_device,
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
                    target_ref="b9442",
                    log=Mock(),
                    set_status=lambda _message: None,
                    clear_status=lambda: None,
                )

        self.assertEqual(synced, "b9442")
        self.assertFalse(any("clone" in call for call in calls))
        fetch_calls = [call for call in calls if "fetch" in call]
        self.assertEqual(len(fetch_calls), 2)
        self.assertIn("http.version=HTTP/1.1", fetch_calls[0])
        self.assertIn("--depth=1", fetch_calls[0])
        self.assertEqual(fetch_calls[0][-2:], ("origin", "b9442"))
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
            built = llama_dir / "build" / "bin" / LLAMA_SERVER_EXE_NAME
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

    async def test_ensure_llama_server_builds_cuda_source_for_mtp_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            llama_dir = Path(tmp) / "llama.cpp"
            bin_dir = Path(tmp) / "bin"
            built = llama_dir / "build" / "bin" / LLAMA_SERVER_EXE_NAME
            old_prebuilt = bin_dir / "llama-server"
            old_prebuilt.parent.mkdir(parents=True)
            old_prebuilt.write_text("old", encoding="utf-8")
            tag_file = bin_dir / "llama-server.tag"
            tag_file.write_text("b9999", encoding="utf-8")

            log = Mock()
            hardware = HardwareInfo(label="RTX 3090", total_ram_gb=64.0, has_cuda=True, vram_mb=24576.0)

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
                "src.provisioning._build_llama_server_from_source",
                AsyncMock(return_value=(built, LLAMA_CPP_MTP_REF)),
            ) as build_source:
                payload = await ensure_llama_server(
                    {
                        "llama_model": "/models/Qwen3.6-27B-Q4_K_M-MTP.gguf",
                        "llama_mtp": True,
                    },
                    hardware_info=hardware,
                    log=log,
                    set_status=lambda _message: None,
                    clear_status=lambda: None,
                )

        self.assertEqual(payload["llama_cpp_ref"], LLAMA_CPP_MTP_REF)
        self.assertEqual(payload["llama_server_path"], str(built))
        self.assertEqual(payload["device"], "cuda")
        install_prebuilt.assert_not_awaited()
        self.assertEqual(build_source.await_args.kwargs["device"], "cuda")
        self.assertEqual(build_source.await_args.kwargs["target_ref"], LLAMA_CPP_MTP_REF)

    async def test_ensure_llama_server_builds_metal_source_for_mtp_model_on_macos(self) -> None:
        # macOS has no prebuilt MTP binary, so an MTP model must reach the source
        # build instead of tripping the "failed to install prebuilt" guard.
        with tempfile.TemporaryDirectory() as tmp:
            llama_dir = Path(tmp) / "llama.cpp"
            bin_dir = Path(tmp) / "bin"
            built = llama_dir / "build" / "bin" / LLAMA_SERVER_EXE_NAME

            log = Mock()
            hardware = HardwareInfo(label="Apple M4 Max", total_ram_gb=64.0, has_cuda=False, has_metal=True)

            with patch("src.provisioning.sys.platform", "darwin"), patch(
                "src.provisioning.LLAMA_CPP_DIR", llama_dir
            ), patch("src.provisioning.BIN_DIR", bin_dir), patch(
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
                AsyncMock(return_value=(built, LLAMA_CPP_MTP_REF)),
            ) as build_source:
                payload = await ensure_llama_server(
                    {
                        "llama_model": "/models/Qwen3.6-35B-A3B-UD-Q3_K_XL-MTP.gguf",
                        "llama_mtp": True,
                    },
                    hardware_info=hardware,
                    log=log,
                    set_status=lambda _message: None,
                    clear_status=lambda: None,
                )

        self.assertEqual(payload["llama_cpp_ref"], LLAMA_CPP_MTP_REF)
        self.assertEqual(payload["llama_server_path"], str(built))
        self.assertFalse(payload["setup_missing_runtime"])
        install_prebuilt.assert_not_awaited()
        build_source.assert_awaited_once()

    async def test_ensure_llama_server_reports_failed_macos_prebuilt_install(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            llama_dir = Path(tmp) / "llama.cpp"
            bin_dir = Path(tmp) / "bin"
            hardware = HardwareInfo(label="Apple M4 Max", total_ram_gb=64.0, has_cuda=False, has_metal=True)

            with patch("src.provisioning.sys.platform", "darwin"), patch(
                "src.provisioning.LLAMA_CPP_DIR", llama_dir
            ), patch("src.provisioning.BIN_DIR", bin_dir), patch(
                "src.provisioning.LLAMA_SERVER_BIN", bin_dir / "llama-server"
            ), patch(
                "src.provisioning.LLAMA_CPP_TAG_FILE", bin_dir / "llama-server.tag"
            ), patch(
                "src.provisioning.current_llama_server_path", return_value=None
            ), patch(
                "src.provisioning._install_prebuilt_llama_server",
                AsyncMock(return_value=None),
            ), patch(
                "src.provisioning._build_llama_server_from_source",
                AsyncMock(),
            ) as build_source:
                with self.assertRaises(RuntimeError):
                    await ensure_llama_server(
                        {"llama_model": "/models/plain.gguf"},
                        hardware_info=hardware,
                        log=Mock(),
                        set_status=lambda _message: None,
                        clear_status=lambda: None,
                    )

        build_source.assert_not_awaited()

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

        self.assertEqual(payload["llama_cpp_ref"], LLAMA_CPP_MTP_REF)
        self.assertEqual(payload["llama_server_path"], str(configured))
        self.assertEqual(payload["device"], "cuda")
        install_prebuilt.assert_not_awaited()
        build_source.assert_not_awaited()

    async def test_ensure_llama_server_builds_cuda_source_for_mtp_on_linux_nvidia(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            llama_dir = Path(tmp) / "llama.cpp"
            bin_dir = Path(tmp) / "bin"
            built = llama_dir / "build" / "bin" / LLAMA_SERVER_EXE_NAME

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
                AsyncMock(return_value=(built, LLAMA_CPP_MTP_REF)),
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

        self.assertEqual(payload["llama_cpp_ref"], LLAMA_CPP_MTP_REF)
        self.assertEqual(payload["llama_server_path"], str(built))
        self.assertEqual(payload["device"], "cuda")
        install_prebuilt.assert_not_awaited()
        self.assertEqual(build_source.await_args.kwargs["device"], "cuda")

    async def test_ensure_llama_server_reuses_matching_cuda_source_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            llama_dir = Path(tmp) / "llama.cpp"
            built = llama_dir / "build" / "bin" / LLAMA_SERVER_EXE_NAME
            built.parent.mkdir(parents=True)
            built.write_text("binary", encoding="utf-8")
            (llama_dir / "build" / "openjet-llama-server.json").write_text(
                f'{{"device": "cuda", "ref": "{LLAMA_CPP_MTP_REF}"}}',
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
                AsyncMock(return_value=None),
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

        self.assertEqual(payload["llama_cpp_ref"], LLAMA_CPP_MTP_REF)
        self.assertEqual(payload["llama_server_path"], str(built))
        self.assertEqual(payload["device"], "cuda")
        install_prebuilt.assert_not_awaited()
        build_source.assert_not_awaited()

    async def test_ensure_llama_server_does_not_reuse_cuda_source_runtime_for_explicit_vulkan_mtp(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            llama_dir = Path(tmp) / "llama.cpp"
            old_built = llama_dir / "build" / "bin" / LLAMA_SERVER_EXE_NAME
            old_built.parent.mkdir(parents=True)
            old_built.write_text("binary", encoding="utf-8")
            new_built = llama_dir / "new-build" / "bin" / LLAMA_SERVER_EXE_NAME
            (llama_dir / "build" / "openjet-llama-server.json").write_text(
                f'{{"device": "cuda", "ref": "{LLAMA_CPP_MTP_REF}"}}',
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
                "src.provisioning.current_llama_server_path", return_value=str(old_built)
            ), patch(
                "src.provisioning._install_prebuilt_llama_server",
                AsyncMock(return_value=None),
            ) as install_prebuilt, patch(
                "src.provisioning._build_llama_server_from_source",
                AsyncMock(return_value=(new_built, LLAMA_CPP_MTP_REF)),
            ) as build_source:
                payload = await ensure_llama_server(
                    {
                        "device": "vulkan",
                        "llama_model": "/models/Qwen3.6-27B-Q4_K_M-MTP.gguf",
                        "llama_mtp": True,
                    },
                    hardware_info=hardware,
                    log=log,
                    set_status=lambda _message: None,
                    clear_status=lambda: None,
                )

        self.assertEqual(payload["llama_cpp_ref"], LLAMA_CPP_MTP_REF)
        self.assertEqual(payload["llama_server_path"], str(new_built))
        self.assertEqual(payload["device"], "vulkan")
        install_prebuilt.assert_not_awaited()
        self.assertEqual(build_source.await_args.kwargs["device"], "vulkan")

    async def test_ensure_llama_server_reuses_source_runtime_when_checkout_ref_matches_without_tag(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            llama_dir = Path(tmp) / "llama.cpp"
            built = llama_dir / "build" / "bin" / LLAMA_SERVER_EXE_NAME
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
                "src.provisioning._needs_rebuild", return_value=False
            ), patch(
                "src.provisioning.current_llama_server_path", return_value=str(built)
            ), patch(
                "src.provisioning._install_prebuilt_llama_server",
                AsyncMock(return_value=None),
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

        self.assertEqual(payload["llama_cpp_ref"], LLAMA_CPP_MTP_REF)
        self.assertEqual(payload["llama_server_path"], str(built))
        self.assertEqual(payload["device"], "cuda")
        install_prebuilt.assert_not_awaited()
        build_source.assert_not_awaited()


class VulkanContextClampTests(unittest.TestCase):
    """A device override must not leave behind a context sized for the old device.

    `ensure_llama_server` can resolve to the prebuilt Vulkan runtime on a Linux
    CUDA host. Vulkan caps a single allocation at 2 GB under Dozen, so a context
    carried over from the CUDA sizing asks for a KV buffer that cannot be
    allocated and llama-server dies on startup.
    """

    def _clamp(self, **overrides: object) -> dict[str, object]:
        merged: dict[str, object] = {
            "device": "vulkan",
            "context_window_tokens": 210767,
            "llama_model": "/models/demo.gguf",
        }
        merged.update(overrides)
        with patch("src.provisioning._model_gguf_path", return_value=Path("/models/demo.gguf")), \
             patch("src.provisioning._kv_bytes_per_token_from_gguf", return_value=36992.0):
            return _clamp_context_for_device(merged)

    def test_lowers_context_past_the_vulkan_allocation_cap(self) -> None:
        self.assertEqual(self._clamp()["context_window_tokens"], 55150)

    def test_leaves_context_already_within_the_cap(self) -> None:
        self.assertEqual(self._clamp(context_window_tokens=20000)["context_window_tokens"], 20000)

    def test_ignores_non_vulkan_devices(self) -> None:
        self.assertEqual(self._clamp(device="cuda")["context_window_tokens"], 210767)

    def test_leaves_context_alone_when_the_model_cannot_be_read(self) -> None:
        merged = {"device": "vulkan", "context_window_tokens": 210767}
        with patch("src.provisioning._model_gguf_path", return_value=None):
            self.assertEqual(_clamp_context_for_device(merged)["context_window_tokens"], 210767)
