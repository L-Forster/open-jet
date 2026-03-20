"""Tests for the dynamic model swap (unload/reload) plugin system."""

from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

from src.agent import ActionKind
from src.executor import ExecResult
from src.swap_plugin import SwapPlugin
from src.swap_manager import SwapManager, SwapResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class FakeSwapPlugin(SwapPlugin):
    """In-memory plugin for testing — no llama server needed."""

    def __init__(
        self,
        *,
        available_mb: float = 1000.0,
        model_mb: float = 2500.0,
        save_ok: bool = True,
        restore_ok: bool = True,
    ) -> None:
        self._available_mb = available_mb
        self._model_mb = model_mb
        self._save_ok = save_ok
        self._restore_ok = restore_ok
        self.save_called = False
        self.unload_called = False
        self.reload_called = False
        self.restore_called = False

    async def save_state(self, state_dir: Path) -> bool:
        self.save_called = True
        return self._save_ok

    async def unload(self) -> None:
        self.unload_called = True

    async def reload(self) -> None:
        self.reload_called = True

    async def restore_state(self, state_dir: Path) -> bool:
        self.restore_called = True
        return self._restore_ok

    def available_memory_mb(self) -> float:
        return self._available_mb

    def model_memory_mb(self) -> float:
        return self._model_mb


# ---------------------------------------------------------------------------
# SwapPlugin interface tests
# ---------------------------------------------------------------------------

class SwapPluginInterfaceTests(unittest.TestCase):
    """Verify the base SwapPlugin raises NotImplementedError."""

    def test_base_class_methods_raise(self) -> None:
        plugin = SwapPlugin()
        with self.assertRaises(NotImplementedError):
            asyncio.run(plugin.save_state(Path("/tmp")))
        with self.assertRaises(NotImplementedError):
            asyncio.run(plugin.unload())
        with self.assertRaises(NotImplementedError):
            asyncio.run(plugin.reload())
        with self.assertRaises(NotImplementedError):
            asyncio.run(plugin.restore_state(Path("/tmp")))
        with self.assertRaises(NotImplementedError):
            plugin.available_memory_mb()
        with self.assertRaises(NotImplementedError):
            plugin.model_memory_mb()

    def test_fake_plugin_implements_interface(self) -> None:
        plugin = FakeSwapPlugin()
        self.assertIsInstance(plugin, SwapPlugin)
        self.assertEqual(plugin.available_memory_mb(), 1000.0)
        self.assertEqual(plugin.model_memory_mb(), 2500.0)


# ---------------------------------------------------------------------------
# SwapManager.should_unload tests
# ---------------------------------------------------------------------------

class SwapManagerShouldUnloadTests(unittest.TestCase):
    """Test memory threshold logic in should_unload."""

    def test_enough_memory_no_unload(self) -> None:
        plugin = FakeSwapPlugin(available_mb=5000.0, model_mb=2500.0)
        mgr = SwapManager(plugin)
        self.assertFalse(mgr.should_unload(estimated_need_mb=2000.0))

    def test_low_memory_triggers_unload(self) -> None:
        plugin = FakeSwapPlugin(available_mb=500.0, model_mb=2500.0)
        mgr = SwapManager(plugin)
        # 500 available, need 2000 + 200 headroom = 2200, not enough.
        # But 500 + 2500 (model) = 3000 > 2200, so unload would help.
        self.assertTrue(mgr.should_unload(estimated_need_mb=2000.0))

    def test_unload_would_not_help(self) -> None:
        # Even freeing the model wouldn't give enough memory.
        plugin = FakeSwapPlugin(available_mb=100.0, model_mb=500.0)
        mgr = SwapManager(plugin)
        # 100 + 500 = 600 < 5000 + 200 = 5200
        self.assertFalse(mgr.should_unload(estimated_need_mb=5000.0))

    def test_zero_estimated_need_low_memory(self) -> None:
        # With 0 estimated need, only triggers if available < 200 headroom.
        plugin = FakeSwapPlugin(available_mb=150.0, model_mb=2500.0)
        mgr = SwapManager(plugin)
        self.assertTrue(mgr.should_unload(estimated_need_mb=0))

    def test_zero_estimated_need_enough_memory(self) -> None:
        plugin = FakeSwapPlugin(available_mb=500.0, model_mb=2500.0)
        mgr = SwapManager(plugin)
        self.assertFalse(mgr.should_unload(estimated_need_mb=0))


# ---------------------------------------------------------------------------
# SwapManager.run_with_swap tests
# ---------------------------------------------------------------------------

class SwapManagerRunWithSwapTests(unittest.IsolatedAsyncioTestCase):
    """Test the full save → unload → run → reload → restore cycle."""

    @patch("src.swap_manager.run_shell")
    async def test_full_swap_cycle(self, mock_run_shell: AsyncMock) -> None:
        mock_run_shell.return_value = ExecResult(
            command="echo hi",
            exit_code=0,
            stdout="hi",
            stderr="",
        )
        plugin = FakeSwapPlugin()
        mgr = SwapManager(plugin)

        result = await mgr.run_with_swap("echo hi", timeout=30)

        self.assertIsInstance(result, SwapResult)
        self.assertTrue(result.unloaded)
        self.assertTrue(result.reload_ok)
        self.assertTrue(result.restore_ok)
        self.assertEqual(result.exec_result.stdout, "hi")
        self.assertTrue(result.exec_result.ok)
        self.assertGreaterEqual(result.swap_duration_s, 0)

        # Verify lifecycle order.
        self.assertTrue(plugin.save_called)
        self.assertTrue(plugin.unload_called)
        self.assertTrue(plugin.reload_called)
        self.assertTrue(plugin.restore_called)
        mock_run_shell.assert_awaited_once_with("echo hi", timeout_seconds=30)

    @patch("src.swap_manager.run_shell")
    async def test_save_failure_skips_swap(self, mock_run_shell: AsyncMock) -> None:
        mock_run_shell.return_value = ExecResult(
            command="ls", exit_code=0, stdout="files", stderr="",
        )
        plugin = FakeSwapPlugin(save_ok=False)
        mgr = SwapManager(plugin)

        result = await mgr.run_with_swap("ls", timeout=30)

        # Should run the command directly, no unload.
        self.assertFalse(result.unloaded)
        self.assertTrue(plugin.save_called)
        self.assertFalse(plugin.unload_called)
        self.assertFalse(plugin.reload_called)

    @patch("src.swap_manager.run_shell")
    async def test_restore_failure_reported(self, mock_run_shell: AsyncMock) -> None:
        mock_run_shell.return_value = ExecResult(
            command="make", exit_code=0, stdout="ok", stderr="",
        )
        plugin = FakeSwapPlugin(restore_ok=False)
        mgr = SwapManager(plugin)

        result = await mgr.run_with_swap("make", timeout=60)

        self.assertTrue(result.unloaded)
        self.assertTrue(result.reload_ok)
        self.assertFalse(result.restore_ok)
        self.assertTrue(plugin.restore_called)

    @patch("src.swap_manager.run_shell")
    async def test_reload_failure_skips_restore(self, mock_run_shell: AsyncMock) -> None:
        mock_run_shell.return_value = ExecResult(
            command="build", exit_code=0, stdout="done", stderr="",
        )
        plugin = FakeSwapPlugin()
        plugin.reload = AsyncMock(side_effect=RuntimeError("server crashed"))
        mgr = SwapManager(plugin)

        result = await mgr.run_with_swap("build", timeout=60)

        self.assertTrue(result.unloaded)
        self.assertFalse(result.reload_ok)
        self.assertFalse(result.restore_ok)
        # Restore should not be attempted if reload failed.
        self.assertFalse(plugin.restore_called)

    @patch("src.swap_manager.run_shell")
    async def test_status_hook_called(self, mock_run_shell: AsyncMock) -> None:
        mock_run_shell.return_value = ExecResult(
            command="test", exit_code=0, stdout="", stderr="",
        )
        statuses: list[str] = []
        plugin = FakeSwapPlugin()
        mgr = SwapManager(plugin, status_hook=statuses.append)

        await mgr.run_with_swap("test", timeout=10)

        self.assertTrue(any("Saving" in s for s in statuses))
        self.assertTrue(any("Unloading" in s for s in statuses))
        self.assertTrue(any("Running" in s for s in statuses))
        self.assertTrue(any("Reloading" in s for s in statuses))
        self.assertTrue(any("Restoring" in s for s in statuses))

    @patch("src.swap_manager.run_shell")
    async def test_command_failure_still_reloads(self, mock_run_shell: AsyncMock) -> None:
        mock_run_shell.return_value = ExecResult(
            command="false", exit_code=1, stdout="", stderr="fail",
        )
        plugin = FakeSwapPlugin()
        mgr = SwapManager(plugin)

        result = await mgr.run_with_swap("false", timeout=10)

        # Command failed but model should still be reloaded.
        self.assertTrue(result.unloaded)
        self.assertTrue(result.reload_ok)
        self.assertFalse(result.exec_result.ok)
        self.assertTrue(plugin.reload_called)


# ---------------------------------------------------------------------------
# LlamaSwapPlugin tests
# ---------------------------------------------------------------------------

class LlamaSwapPluginTests(unittest.IsolatedAsyncioTestCase):
    """Test the llama.cpp swap plugin with mocked server client."""

    async def test_save_state_persists_messages(self) -> None:
        from src.swap_llama import LlamaSwapPlugin

        client = Mock()
        client.save_kv_cache = AsyncMock(return_value=True)

        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ]
        plugin = LlamaSwapPlugin(client, messages=messages)

        with tempfile.TemporaryDirectory() as td:
            state_dir = Path(td) / "state"
            ok = await plugin.save_state(state_dir)

            self.assertTrue(ok)
            msgs_path = state_dir / "messages.json"
            self.assertTrue(msgs_path.exists())
            loaded = json.loads(msgs_path.read_text())
            self.assertEqual(len(loaded), 2)
            self.assertEqual(loaded[0]["role"], "system")
            client.save_kv_cache.assert_awaited_once()

    async def test_restore_state_loads_messages(self) -> None:
        from src.swap_llama import LlamaSwapPlugin

        client = Mock()
        client.restore_kv_cache = AsyncMock(return_value=True)

        messages: list[dict] = []
        plugin = LlamaSwapPlugin(client, messages=messages)

        with tempfile.TemporaryDirectory() as td:
            state_dir = Path(td) / "state"
            state_dir.mkdir()
            saved = [{"role": "user", "content": "test"}]
            (state_dir / "messages.json").write_text(json.dumps(saved))
            (state_dir / "kv_cache.bin").write_bytes(b"fake")

            ok = await plugin.restore_state(state_dir)

            self.assertTrue(ok)
            self.assertEqual(len(messages), 1)
            self.assertEqual(messages[0]["content"], "test")
            client.restore_kv_cache.assert_awaited_once()

    async def test_restore_without_kv_cache_returns_false(self) -> None:
        from src.swap_llama import LlamaSwapPlugin

        client = Mock()
        client.restore_kv_cache = AsyncMock(return_value=False)

        messages: list[dict] = []
        plugin = LlamaSwapPlugin(client, messages=messages)

        with tempfile.TemporaryDirectory() as td:
            state_dir = Path(td) / "state"
            state_dir.mkdir()
            saved = [{"role": "user", "content": "test"}]
            (state_dir / "messages.json").write_text(json.dumps(saved))
            (state_dir / "kv_cache.bin").write_bytes(b"fake")

            ok = await plugin.restore_state(state_dir)

            # Messages restored but KV cache failed.
            self.assertFalse(ok)
            self.assertEqual(len(messages), 1)

    async def test_unload_stops_server(self) -> None:
        from src.swap_llama import LlamaSwapPlugin

        client = Mock()
        client._stop_server = AsyncMock()

        plugin = LlamaSwapPlugin(client)
        await plugin.unload()

        client._stop_server.assert_awaited_once()

    async def test_reload_starts_server(self) -> None:
        from src.swap_llama import LlamaSwapPlugin

        client = Mock()
        client.start = AsyncMock()

        plugin = LlamaSwapPlugin(client)
        await plugin.reload()

        client.start.assert_awaited_once()

    @patch("src.swap_llama.read_memory_snapshot")
    async def test_available_memory_reads_proc(self, mock_snap: Mock) -> None:
        from src.runtime_limits import MemorySnapshot
        from src.swap_llama import LlamaSwapPlugin

        mock_snap.return_value = MemorySnapshot(
            total_mb=8000.0,
            available_mb=3200.0,
            used_percent=60.0,
        )
        client = Mock()
        plugin = LlamaSwapPlugin(client)

        self.assertAlmostEqual(plugin.available_memory_mb(), 3200.0)


# ---------------------------------------------------------------------------
# LlamaServerClient KV cache API tests
# ---------------------------------------------------------------------------

class LlamaServerKvCacheTests(unittest.IsolatedAsyncioTestCase):
    """Test save_kv_cache and restore_kv_cache HTTP calls."""

    async def test_save_kv_cache_posts_to_slots_api(self) -> None:
        from src.llama_server import LlamaServerClient

        client = LlamaServerClient.__new__(LlamaServerClient)
        client.base_url = "http://127.0.0.1:8080"
        client._diagnostics_hook = None

        mock_http = AsyncMock()
        mock_response = Mock()
        mock_response.status_code = 200
        mock_http.post.return_value = mock_response
        client._http = mock_http

        ok = await client.save_kv_cache(Path("/tmp/kv_cache.bin"))

        self.assertTrue(ok)
        mock_http.post.assert_awaited_once()
        call_args = mock_http.post.call_args
        self.assertIn("/slots/0?action=save", call_args[0][0])
        self.assertEqual(call_args[1]["json"]["filename"], "kv_cache.bin")

    async def test_restore_kv_cache_posts_to_slots_api(self) -> None:
        from src.llama_server import LlamaServerClient

        client = LlamaServerClient.__new__(LlamaServerClient)
        client.base_url = "http://127.0.0.1:8080"
        client._diagnostics_hook = None

        mock_http = AsyncMock()
        mock_response = Mock()
        mock_response.status_code = 200
        mock_http.post.return_value = mock_response
        client._http = mock_http

        ok = await client.restore_kv_cache(Path("/tmp/kv_cache.bin"))

        self.assertTrue(ok)
        call_args = mock_http.post.call_args
        self.assertIn("/slots/0?action=restore", call_args[0][0])

    async def test_save_kv_cache_returns_false_on_error(self) -> None:
        from src.llama_server import LlamaServerClient

        client = LlamaServerClient.__new__(LlamaServerClient)
        client.base_url = "http://127.0.0.1:8080"
        client._diagnostics_hook = None

        mock_http = AsyncMock()
        mock_http.post.side_effect = Exception("connection refused")
        client._http = mock_http

        ok = await client.save_kv_cache(Path("/tmp/kv.bin"))
        self.assertFalse(ok)

    async def test_save_kv_cache_returns_false_on_500(self) -> None:
        from src.llama_server import LlamaServerClient

        client = LlamaServerClient.__new__(LlamaServerClient)
        client.base_url = "http://127.0.0.1:8080"
        client._diagnostics_hook = None

        mock_http = AsyncMock()
        mock_response = Mock()
        mock_response.status_code = 500
        mock_http.post.return_value = mock_response
        client._http = mock_http

        ok = await client.save_kv_cache(Path("/tmp/kv.bin"))
        self.assertFalse(ok)


# ---------------------------------------------------------------------------
# Tool executor swap integration tests
# ---------------------------------------------------------------------------

class ToolExecutorSwapTests(unittest.IsolatedAsyncioTestCase):
    """Test that tool_executor routes through SwapManager when registered."""

    async def test_shell_uses_swap_when_should_unload(self) -> None:
        from src.runtime_protocol import ToolCall
        from src.swap_manager import SwapManager
        from src.tool_executor import execute_tool, set_swap_manager

        plugin = FakeSwapPlugin(available_mb=100.0, model_mb=2500.0)
        mgr = SwapManager(plugin)

        with patch.object(mgr, "run_with_swap", new_callable=AsyncMock) as mock_swap:
            mock_swap.return_value = SwapResult(
                exec_result=ExecResult(
                    command="echo hi", exit_code=0, stdout="hi", stderr="",
                ),
                unloaded=True,
                reload_ok=True,
                restore_ok=True,
                swap_duration_s=5.0,
            )
            set_swap_manager(mgr)
            try:
                tc = ToolCall(name="shell", arguments={"command": "echo hi"}, id="c1")
                result = await execute_tool(tc)

                self.assertTrue(result.ok)
                self.assertTrue(result.meta.get("swapped"))
                self.assertEqual(result.meta.get("swap_duration_s"), 5.0)
                mock_swap.assert_awaited_once()
            finally:
                set_swap_manager(None)

    async def test_shell_direct_when_enough_memory(self) -> None:
        from src.runtime_protocol import ToolCall
        from src.swap_manager import SwapManager
        from src.tool_executor import execute_tool, set_swap_manager

        plugin = FakeSwapPlugin(available_mb=5000.0, model_mb=2500.0)
        mgr = SwapManager(plugin)
        set_swap_manager(mgr)

        try:
            with patch("src.tool_executor.run_shell", new_callable=AsyncMock) as mock_shell:
                mock_shell.return_value = ExecResult(
                    command="ls", exit_code=0, stdout="file.txt", stderr="",
                )
                tc = ToolCall(name="shell", arguments={"command": "ls"}, id="c2")
                result = await execute_tool(tc)

                self.assertTrue(result.ok)
                self.assertNotIn("swapped", result.meta)
                mock_shell.assert_awaited_once()
        finally:
            set_swap_manager(None)

    async def test_shell_direct_when_no_swap_manager(self) -> None:
        from src.runtime_protocol import ToolCall
        from src.tool_executor import execute_tool, set_swap_manager

        set_swap_manager(None)

        with patch("src.tool_executor.run_shell", new_callable=AsyncMock) as mock_shell:
            mock_shell.return_value = ExecResult(
                command="pwd", exit_code=0, stdout="/home", stderr="",
            )
            tc = ToolCall(name="shell", arguments={"command": "pwd"}, id="c3")
            result = await execute_tool(tc)

            self.assertTrue(result.ok)
            mock_shell.assert_awaited_once()


# ---------------------------------------------------------------------------
# Agent UNLOAD event kind
# ---------------------------------------------------------------------------

class AgentUnloadEventTests(unittest.TestCase):
    """Verify the UNLOAD event kind exists and is usable."""

    def test_unload_event_kind_exists(self) -> None:
        self.assertEqual(ActionKind.UNLOAD.name, "UNLOAD")

    def test_agent_event_with_unload(self) -> None:
        from src.agent import AgentEvent
        event = AgentEvent(kind=ActionKind.UNLOAD, text="heavy task needs memory")
        self.assertEqual(event.kind, ActionKind.UNLOAD)
        self.assertEqual(event.text, "heavy task needs memory")


if __name__ == "__main__":
    unittest.main()
