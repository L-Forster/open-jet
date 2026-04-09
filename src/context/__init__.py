"""Context-layer facade.

This layer owns prompt shaping, context budgeting, harness state, and durable memory.
"""

from ..agent import ActionKind, Agent, AgentEvent
from ..harness import (
    HarnessSessionStore,
    HarnessState,
    PlanStep,
    TurnBudget,
    active_step,
    advance_step,
    available_skill_names,
    build_turn_context,
    clear_preferred_skills,
    set_mode,
    set_preferred_skills,
    split_active_step,
    update_state_after_turn,
    update_state_for_user_message,
)
from ..multimodal import build_user_content, content_to_plain_text, estimate_message_content_tokens
from ..memory_reflection import reflect_agent_persistent_memory, refresh_agent_system_prompt
from ..persistent_memory import (
    build_system_prompt,
    load_persistent_memory,
    update_persistent_memory,
)
from ..runtime_limits import ContextBudget, derive_context_budget, estimate_tokens, read_memory_snapshot

__all__ = [
    "ActionKind",
    "Agent",
    "AgentEvent",
    "ContextBudget",
    "HarnessSessionStore",
    "HarnessState",
    "PlanStep",
    "TurnBudget",
    "active_step",
    "advance_step",
    "available_skill_names",
    "build_system_prompt",
    "build_turn_context",
    "build_user_content",
    "clear_preferred_skills",
    "content_to_plain_text",
    "derive_context_budget",
    "estimate_message_content_tokens",
    "estimate_tokens",
    "load_persistent_memory",
    "read_memory_snapshot",
    "reflect_agent_persistent_memory",
    "refresh_agent_system_prompt",
    "set_mode",
    "set_preferred_skills",
    "split_active_step",
    "update_persistent_memory",
    "update_state_after_turn",
    "update_state_for_user_message",
]
