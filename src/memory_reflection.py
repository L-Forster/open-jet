from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from .persistent_memory import (
    append_persistent_memory_bullet,
    build_system_prompt,
    load_persistent_memory,
    normalize_location,
)

if TYPE_CHECKING:
    from .agent import Agent


async def refresh_agent_system_prompt(agent: "Agent") -> None:
    if agent.project_root is None:
        return
    refreshed = await build_system_prompt(
        agent.base_system_prompt,
        agent.project_root,
        cfg=agent.prompt_cfg,
        global_root=agent.global_memory_root,
    )
    agent.system_prompt = refreshed
    if agent.messages and str(agent.messages[0].get("role", "")) == "system":
        agent.messages[0]["content"] = refreshed
    else:
        agent.messages.insert(0, {"role": "system", "content": refreshed})


def build_recorded_turn_payload(
    *,
    user_prompt: str,
    assistant_text: str,
    tool_calls: list[dict[str, Any]] | None = None,
    tool_results: list[dict[str, Any]] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "user_prompt": str(user_prompt or ""),
        "assistant_text": str(assistant_text or ""),
        "tool_calls": list(tool_calls or []),
        "tool_results": list(tool_results or []),
    }
    if isinstance(extra, dict):
        payload.update(extra)
    return payload


async def reflect_agent_persistent_memory(
    agent: "Agent",
    *,
    recorded_turn: dict[str, Any] | None,
) -> dict[str, object]:
    if agent.project_root is None:
        return {"ok": True, "reason": "memory unavailable", "applied": []}
    if not isinstance(recorded_turn, dict):
        return {"ok": True, "reason": "no recorded turn", "applied": []}

    transcript = _render_recorded_turn(recorded_turn)
    if not transcript:
        return {"ok": True, "reason": "empty recorded turn", "applied": []}

    snapshot = await load_persistent_memory(
        agent.project_root,
        global_root=agent.global_memory_root,
    )
    prompt = (
        "You decide whether the latest completed turn should be saved into agent memory.\n"
        "There are two memory locations:\n"
        "- global memory: facts useful across projects and future sessions\n"
        "- local memory: facts useful only for this project, repo, cwd, commands, paths, hosts, or workflows\n"
        "Examples of facts that may belong in global memory:\n"
        "- stable ssh host aliases\n"
        "- stable machine or environment facts reusable outside this repo\n"
        "- stable device naming conventions used across projects\n"
        "Examples of facts that may belong in local memory:\n"
        "- repo-specific commands\n"
        "- project-specific file paths\n"
        "- result directories\n"
        "- training or deploy workflows for this repo\n"
        "- environment facts specific to this cwd or project setup\n"
        "Examples of things that should not be stored:\n"
        "- secrets, credentials, tokens, passwords, keys\n"
        "- raw logs\n"
        "- one-off outputs\n"
        "- transient task state\n"
        "- guesses or speculation\n"
        "Decide whether there is exactly one stable fact worth remembering from the latest turn.\n"
        "Return strict JSON only in this shape:\n"
        '{"store":true|false,"location":"global|local","bullet":"- one short bullet"}\n'
        "Rules:\n"
        "- store at most one bullet\n"
        "- prefer local unless the fact is clearly useful outside this project\n"
        "- the bullet must be short and concrete\n"
        "- return {\"store\":false} if nothing should be remembered\n\n"
        "Current global agent memory:\n"
        f"{snapshot.global_agent or '(empty)'}\n\n"
        "Current local agent memory:\n"
        f"{snapshot.project_agent or '(empty)'}\n\n"
        "Latest completed turn:\n"
        f"{transcript}"
    )
    messages = [
        {
            "role": "system",
            "content": "You maintain agent memory. Return strict JSON only.",
        },
        {"role": "user", "content": prompt},
    ]
    prompt_tokens = agent._estimate_runtime_request_tokens(messages, use_tools=False)
    agent._trace(
        "runtime_request",
        request_kind="memory_reflection",
        prompt_tokens=prompt_tokens,
        use_tools=False,
        message_count=len(messages),
    )
    chunks: list[str] = []
    async for chunk in agent.client.chat_stream(messages, use_tools=False):
        if chunk.text:
            chunks.append(chunk.text)
    raw = "".join(chunks).strip()
    agent._trace(
        "runtime_exchange_complete",
        request_kind="memory_reflection",
        prompt_tokens=prompt_tokens,
        completion_tokens=agent._estimate_runtime_output_tokens(text=raw),
        use_tools=False,
        message_count=len(messages),
        tool_call_count=0,
    )

    action = _parse_memory_decision(raw)
    if not action or not action.get("store"):
        return {"ok": True, "applied": [], "raw": raw}
    requested_scope = str(action.get("scope", "")).strip().lower()
    if requested_scope and requested_scope not in {"agent", "memory", "environment"}:
        return {"ok": True, "applied": [], "raw": raw}

    try:
        location = normalize_location(str(action.get("location", "")).strip())
    except ValueError:
        return {"ok": True, "applied": [], "raw": raw}

    bullet = _normalize_reflection_bullet(str(action.get("bullet", "")))
    if not bullet:
        return {"ok": True, "applied": [], "raw": raw}

    result = await append_persistent_memory_bullet(
        agent.project_root,
        location=location,
        scope="agent",
        content=bullet,
        global_root=agent.global_memory_root,
    )
    if result.startswith("Skipped ") or result.startswith("Error"):
        return {"ok": True, "applied": [], "raw": raw, "reason": result}

    applied = [{"location": location, "scope": "agent", "content": bullet, "result": result}]
    await refresh_agent_system_prompt(agent)
    return {"ok": True, "applied": applied, "raw": raw}


def _render_recorded_turn(recorded_turn: dict[str, Any]) -> str:
    lines: list[str] = []

    user_prompt = " ".join(str(recorded_turn.get("user_prompt", "")).split()).strip()
    if user_prompt:
        lines.append(f"USER: {user_prompt}")

    assistant_text = " ".join(str(recorded_turn.get("assistant_text", "")).split()).strip()
    if assistant_text:
        lines.append(f"ASSISTANT: {assistant_text}")

    raw_tool_calls = recorded_turn.get("tool_calls", [])
    if isinstance(raw_tool_calls, list):
        for item in raw_tool_calls:
            if not isinstance(item, dict):
                continue
            tool = str(item.get("tool", "")).strip() or "tool"
            arguments = item.get("arguments", {})
            lines.append(f"TOOL_CALL: {_format_tool_call(tool, arguments)}")

    raw_tool_results = recorded_turn.get("tool_results", [])
    if isinstance(raw_tool_results, list):
        for item in raw_tool_results:
            if not isinstance(item, dict):
                continue
            tool = str(item.get("tool", "")).strip() or "tool"
            arguments = item.get("arguments", {})
            output = " ".join(str(item.get("raw_output", "")).split()).strip()
            descriptor = _format_tool_call(tool, arguments)
            if output:
                lines.append(f"TOOL_RESULT: {descriptor}")
                lines.append(output)

    return "\n".join(lines).strip()


def _format_tool_call(name: str, arguments: Any) -> str:
    if isinstance(arguments, dict):
        if name in {"read_file", "write_file", "load_file", "edit_file"}:
            return f"{name} path={str(arguments.get('path', '')).strip() or '<unknown>'}"
        if name in {"camera_snapshot", "microphone_record", "sensor_read", "gpio_read"}:
            source = str(arguments.get("source", "")).strip() or "<auto>"
            return f"{name} source={source}"
        if name == "device_list":
            kind = str(arguments.get("kind", "")).strip() or "all"
            return f"{name} kind={kind}"
        if name == "list_directory":
            return f"{name} path={str(arguments.get('path', '.') or '.').strip()}"
        if name == "shell":
            command = " ".join(str(arguments.get("command", "")).split()).strip()
            return f"{name} command={command or '<unknown>'}"
        if name == "system_info":
            scope = str(arguments.get("scope", "summary") or "summary").strip()
            return f"{name} scope={scope}"
        if name == "memory":
            location = str(arguments.get("location", "local") or "local").strip()
            scope = str(arguments.get("scope", "") or "").strip() or "<unknown>"
            action = str(arguments.get("action", "") or "").strip() or "<unknown>"
            return f"{name} action={action} location={location} scope={scope}"
    text = " ".join(str(arguments).split()).strip()
    return f"{name} {text}".strip()


def _normalize_reflection_bullet(content: str) -> str:
    normalized = " ".join(content.strip().split())
    normalized = normalized.lstrip("-* ").strip()
    if not normalized:
        return ""
    if len(normalized) > 240:
        normalized = normalized[:237].rstrip() + "..."
    return f"- {normalized}"


def _parse_memory_decision(raw: str) -> dict[str, Any] | None:
    text = raw.strip()
    if not text:
        return None
    if text.startswith("```"):
        lines = [line for line in text.splitlines() if not line.strip().startswith("```")]
        text = "\n".join(lines).strip()
    payload: object
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        left = text.find("{")
        right = text.rfind("}")
        if left == -1 or right == -1 or right <= left:
            return None
        try:
            payload = json.loads(text[left : right + 1])
        except json.JSONDecodeError:
            return None
    return payload if isinstance(payload, dict) else None
