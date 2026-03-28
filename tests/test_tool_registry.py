from __future__ import annotations

import unittest

import src.tool_executor  # noqa: F401
from src.agent import CONFIRM_TOOLS
from src.harness import CONFIRMATION_GATED_TOOLS, DEVICE_TOOLS
from src.runtime_protocol import TOOLS
from src.tools.registry import (
    TOOL_REGISTRY,
    all_tool_names,
    confirmation_required_tool_names,
    tool_names_with_tag,
    workflow_default_tool_names,
    workflow_optional_tool_names,
)
from src.workflows.runner import WORKFLOW_BASE_TOOLS, WORKFLOW_OPTIONAL_TOOLS


class ToolRegistryTests(unittest.TestCase):
    def test_runtime_tool_schema_names_match_registry_order(self) -> None:
        schema_names = [tool["function"]["name"] for tool in TOOLS]
        self.assertEqual(schema_names, list(all_tool_names()))

    def test_confirmation_policy_is_derived_from_registry(self) -> None:
        expected = set(confirmation_required_tool_names())
        self.assertEqual(expected, CONFIRM_TOOLS)
        self.assertEqual(expected, CONFIRMATION_GATED_TOOLS)

    def test_device_tool_set_is_derived_from_registry(self) -> None:
        self.assertEqual(set(tool_names_with_tag("device")), DEVICE_TOOLS)

    def test_workflow_tool_sets_are_derived_from_registry(self) -> None:
        self.assertEqual(set(workflow_default_tool_names()), WORKFLOW_BASE_TOOLS)
        self.assertEqual(set(workflow_optional_tool_names()), WORKFLOW_OPTIONAL_TOOLS)

    def test_every_registered_tool_has_an_executor_bound(self) -> None:
        missing = [spec.name for spec in TOOL_REGISTRY.all_specs() if spec.executor is None]
        self.assertEqual(missing, [])
