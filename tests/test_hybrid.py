from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from src.app import OpenJetApp
from src.hybrid import (
    DELEGATE_LOCAL_TOOL,
    IMPLEMENTER_SYSTEM_PROMPT,
    ORCHESTRATOR_SYSTEM_PROMPT,
    HybridWorker,
    execution_mode,
)
from src.sdk import SDKResponse
from src.tools.registry import get_tool_spec, unregister_tool


class HybridModeTests(unittest.IsolatedAsyncioTestCase):
    def tearDown(self) -> None:
        unregister_tool(DELEGATE_LOCAL_TOOL)

    def test_execution_mode_is_explicit(self) -> None:
        self.assertEqual(execution_mode({"runtime": "llama_cpp"}), "local")
        self.assertEqual(execution_mode({"runtime": "openai_codex"}), "codex")
        self.assertEqual(
            execution_mode({"runtime": "openai_codex", "execution_mode": "hybrid"}),
            "hybrid",
        )

    def test_prompts_enforce_orchestrator_and_concise_worker_roles(self) -> None:
        self.assertIn("Act as an\norchestrator", ORCHESTRATOR_SYSTEM_PROMPT)
        self.assertIn("near 20%", ORCHESTRATOR_SYSTEM_PROMPT)
        self.assertIn("final\nreview", ORCHESTRATOR_SYSTEM_PROMPT)
        self.assertIn("concise handoff", IMPLEMENTER_SYSTEM_PROMPT)
        self.assertIn("Do not\ndelegate", IMPLEMENTER_SYSTEM_PROMPT)

    async def test_worker_is_warmed_and_registers_delegate_tool(self) -> None:
        client = SimpleNamespace(start=AsyncMock(), close=AsyncMock())
        agent = SimpleNamespace(client=client, trace_hook=None, reset_conversation=lambda: None)
        session = SimpleNamespace(agent=agent, close=AsyncMock(), run=AsyncMock())
        local_profile = {
            "name": "qwen",
            "runtime": "llama_cpp",
            "llama_model": "/models/qwen.gguf",
        }
        with patch("src.hybrid.OpenJetSession.create", AsyncMock(return_value=session)) as create:
            worker = await HybridWorker.start(
                base_cfg={"runtime": "openai_codex", "model": "gpt-5.6-sol"},
                local_profile=local_profile,
                root=Path("."),
                approval_handler=None,
            )

        client.start.assert_awaited_once_with()
        self.assertTrue(worker.ready)
        self.assertIsNotNone(get_tool_spec(DELEGATE_LOCAL_TOOL))
        create.assert_awaited_once()
        await worker.close()

    async def test_delegate_resets_context_and_returns_compact_metadata(self) -> None:
        reset = unittest.mock.Mock()
        session = SimpleNamespace(
            agent=SimpleNamespace(reset_conversation=reset),
            run=AsyncMock(return_value=SDKResponse(text="Changed x.py; tests pass.")),
            close=AsyncMock(),
        )
        worker = HybridWorker(
            session=session,
            model_ref="/models/qwen.gguf",
            profile_name="qwen",
            harness_state=SimpleNamespace(goal="", mode="code", plan_approved=True),
        )
        worker.ready = True
        worker._register_tool()

        result = await worker.delegate(
            {"task": "Implement the parser", "acceptance_criteria": "Parser tests pass"}
        )

        reset.assert_called_once_with()
        prompt = session.run.await_args.args[0]
        self.assertIn("Implement the parser", prompt)
        self.assertIn("Parser tests pass", prompt)
        self.assertTrue(result.ok)
        self.assertEqual(result.output, "Changed x.py; tests pass.")
        self.assertIsNotNone(get_tool_spec(DELEGATE_LOCAL_TOOL))

    def test_footer_starts_with_hybrid_pair(self) -> None:
        app = OpenJetApp()
        app.cfg.update(
            {
                "runtime": "openai_codex",
                "execution_mode": "hybrid",
                "model": "gpt-5.6-sol",
                "reasoning_effort": "xhigh",
            }
        )
        app.hybrid_worker = SimpleNamespace(model_ref="/models/Qwen-27B.gguf", ready=True)

        cells = app._chrome_status_cells()

        self.assertEqual(cells[0][1], "HYBRID")
        self.assertIn("gpt-5.6-sol xhigh + Qwen-27B", cells[1][1])

    def test_usage_reports_frontier_share(self) -> None:
        app = OpenJetApp()
        app.cfg.update(
            {"runtime": "openai_codex", "execution_mode": "hybrid", "model": "gpt-5.6-sol"}
        )
        app.hybrid_worker = SimpleNamespace(model_ref="/models/qwen.gguf", ready=True)
        app._record_runtime_token_usage(
            prompt_tokens=10, completion_tokens=10, model_ref="gpt-5.6-sol"
        )
        app._record_runtime_token_usage(
            prompt_tokens=40, completion_tokens=40, model_ref="/models/qwen.gguf"
        )

        hybrid = app.token_usage_snapshot()["hybrid"]

        self.assertAlmostEqual(hybrid["codex_share"], 0.2)
        self.assertTrue(hybrid["on_target"])

    async def test_model_picker_can_create_profile_from_detected_gguf(self) -> None:
        app = OpenJetApp()
        app.cfg = {"model_profiles": [], "runtime": "openai_codex", "model": "gpt-5.6-sol"}
        app._session = SimpleNamespace()
        log = app.query_one("#chat-log")
        with tempfile.TemporaryDirectory() as tmp:
            model_path = Path(tmp) / "local-coder.gguf"
            model_path.write_bytes(b"GGUF")
            with patch("src.commands.discover_model_files", return_value=[str(model_path)]), patch(
                "src.commands._prompt_choice",
                AsyncMock(return_value=("file", str(model_path))),
            ), patch("src.commands.save_config"):
                selected = await app.commands._choose_local_model_profile(log)

            self.assertEqual(selected, "local-coder")
            profile = app.cfg["model_profiles"][0]
            self.assertEqual(profile["runtime"], "llama_cpp")
            self.assertEqual(profile["llama_model"], str(model_path))
