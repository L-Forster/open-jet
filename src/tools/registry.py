from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
from typing import Any, Callable, Iterable, Mapping


ToolExecutor = Callable[[dict[str, Any]], Any]
TOOL_MODES: tuple[str, ...] = ("chat", "code", "review", "debug")
ALL_TOOL_MODES = frozenset(TOOL_MODES)


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: Mapping[str, object]
    required: tuple[str, ...] = ()
    confirmation_required: bool = False
    modes: frozenset[str] = ALL_TOOL_MODES
    workflow_default: bool = False
    workflow_optional: bool = False
    tags: frozenset[str] = frozenset()
    executor: ToolExecutor | None = None

    def runtime_schema(self) -> dict[str, object]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": deepcopy(dict(self.parameters)),
                    "required": list(self.required),
                },
            },
        }

    def with_executor(self, executor: ToolExecutor) -> "ToolSpec":
        return replace(self, executor=executor)


class ToolRegistry:
    def __init__(self, specs: Iterable[ToolSpec]) -> None:
        self._specs: dict[str, ToolSpec] = {}
        for spec in specs:
            self.register(spec)

    def register(self, spec: ToolSpec) -> None:
        name = spec.name.strip()
        if not name:
            raise ValueError("tool name cannot be empty")
        if name in self._specs:
            raise ValueError(f"duplicate tool spec: {name}")
        self._specs[name] = spec

    def bind_executor(self, name: str, executor: ToolExecutor) -> None:
        spec = self.get(name)
        if spec is None:
            raise KeyError(name)
        self._specs[name] = spec.with_executor(executor)

    def get(self, name: str) -> ToolSpec | None:
        return self._specs.get(str(name).strip())

    def all_specs(self) -> tuple[ToolSpec, ...]:
        return tuple(self._specs.values())

    def all_names(self) -> tuple[str, ...]:
        return tuple(self._specs)

    def runtime_schemas(self) -> list[dict[str, object]]:
        return [spec.runtime_schema() for spec in self._specs.values()]

    def names_requiring_confirmation(self) -> tuple[str, ...]:
        return tuple(spec.name for spec in self._specs.values() if spec.confirmation_required)

    def names_with_tag(self, tag: str) -> tuple[str, ...]:
        needle = str(tag).strip().lower()
        if not needle:
            return ()
        return tuple(spec.name for spec in self._specs.values() if needle in spec.tags)

    def names_for_mode(self, mode: str | None) -> tuple[str, ...]:
        normalized = str(mode or "").strip().lower()
        if not normalized:
            return self.all_names()
        return tuple(spec.name for spec in self._specs.values() if normalized in spec.modes)

    def bundle_names_for_mode(self, mode: str | None) -> tuple[str, ...]:
        normalized = str(mode or "").strip().lower()
        names: list[str] = []
        for spec in self._specs.values():
            if spec.confirmation_required:
                continue
            if normalized and normalized not in spec.modes:
                continue
            names.append(spec.name)
        return tuple(names)

    def workflow_default_names(self) -> tuple[str, ...]:
        return tuple(spec.name for spec in self._specs.values() if spec.workflow_default)

    def workflow_optional_names(self) -> tuple[str, ...]:
        return tuple(spec.name for spec in self._specs.values() if spec.workflow_optional)


def _param(type_name: str, description: str) -> dict[str, str]:
    return {"type": type_name, "description": description}


def _tool(
    name: str,
    description: str,
    *,
    parameters: Mapping[str, object],
    required: tuple[str, ...] = (),
    confirmation_required: bool = False,
    modes: frozenset[str] = ALL_TOOL_MODES,
    workflow_default: bool = False,
    workflow_optional: bool = False,
    tags: Iterable[str] = (),
) -> ToolSpec:
    return ToolSpec(
        name=name,
        description=description,
        parameters=dict(parameters),
        required=required,
        confirmation_required=confirmation_required,
        modes=modes,
        workflow_default=workflow_default,
        workflow_optional=workflow_optional,
        tags=frozenset(str(tag).strip().lower() for tag in tags if str(tag).strip()),
    )


TOOL_REGISTRY = ToolRegistry(
    [
        _tool(
            "device_list",
            "List active currently connected camera, microphone, speaker, and sensor sources available to the agent.",
            parameters={
                "kind": _param("string", "Optional filter: camera, microphone, speaker, or sensor"),
            },
            workflow_default=True,
            tags=("device", "read"),
        ),
        _tool(
            "camera_snapshot",
            "Capture a single image from a detected camera source and return it to the model.",
            parameters={
                "source": _param("string", "Camera source reference such as camera0 or a saved alias"),
                "output_path": _param("string", "Optional output path for the captured image"),
            },
            workflow_default=True,
            tags=("device", "read", "multimodal"),
        ),
        _tool(
            "microphone_record",
            "Record a short audio clip from a detected microphone, try bundled local transcription first, and fall back to speech-activity detection if transcription is unavailable.",
            parameters={
                "source": _param("string", "Microphone source reference such as mic0 or a saved alias"),
                "duration_seconds": _param("integer", "Recording length in seconds"),
                "output_path": _param("string", "Optional output path for the recorded WAV clip"),
            },
            workflow_default=True,
            tags=("device", "read", "multimodal"),
        ),
        _tool(
            "microphone_set_enabled",
            "Toggle a detected microphone source on or off.",
            parameters={
                "source": _param("string", "Microphone source reference such as mic0 or a saved alias"),
                "enabled": _param("boolean", "Set true to enable the microphone source, false to disable it"),
            },
            required=("enabled",),
            workflow_default=True,
            tags=("device", "write"),
        ),
        _tool(
            "gpio_read",
            "Read a GPIO-backed source and return the current text snapshot/buffer for that specific device.",
            parameters={
                "source": _param("string", "GPIO source reference such as gpio0 or a saved alias"),
            },
            workflow_default=True,
            tags=("device", "read"),
        ),
        _tool(
            "sensor_read",
            "Legacy alias for gpio_read. Current implementation supports GPIO-backed sensor text snapshots only.",
            parameters={
                "source": _param("string", "GPIO source reference such as gpio0 or a saved alias"),
            },
            workflow_default=True,
            tags=("device", "read"),
        ),
        _tool(
            "shell",
            (
                "Run a shell command. Use this for system operations such as cron setup, ssh, "
                "package managers, build tools, and other CLI workflows. For heavy local work you may set "
                "resource_mode=unload_first to unload the local model before the command "
                "and reload it afterward."
            ),
            parameters={
                "command": _param("string", "Command to run"),
                "timeout_seconds": _param("integer", "Timeout in seconds"),
                "resource_mode": _param("string", "Resource strategy: normal, auto, or unload_first"),
                "estimated_ram_mb": _param("integer", "Optional estimated RAM needed by the command in MB"),
                "estimated_vram_mb": _param("integer", "Optional estimated VRAM needed by the command in MB"),
                "reload_delay_seconds": _param(
                    "integer",
                    "Optional delay before reloading the model after an unload-run cycle",
                ),
            },
            required=("command",),
            confirmation_required=True,
            workflow_optional=True,
            tags=("system", "exec", "write"),
        ),
        _tool(
            "system_info",
            "Inspect local system resource information such as RAM, disk, load, and GPU memory when detectable. Use this before heavy shell commands.",
            parameters={
                "scope": _param("string", "Info scope: summary, memory, gpu, disk, or all"),
            },
            workflow_default=True,
            tags=("system", "read"),
        ),
        _tool(
            "memory",
            (
                "Read or update persistent cross-session memory. "
                "Use location=global for facts reusable outside the current project, "
                "and location=project for repo or cwd-specific facts."
            ),
            parameters={
                "location": _param("string", "Memory location: global or project"),
                "scope": _param("string", "Memory scope: user or agent"),
                "action": _param("string", "Operation: read, append, replace, or clear"),
                "content": _param("string", "Memory content for append or replace"),
            },
            required=("scope", "action"),
            confirmation_required=True,
            tags=("memory", "write"),
        ),
        _tool(
            "todo_write",
            "Create or replace the current todo list for a complex task. The model should write the list itself when explicit tracking is useful.",
            parameters={
                "todos": {
                    "type": "array",
                    "description": "Todo items to store for the current session.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": _param("string", "Stable todo id"),
                            "content": _param("string", "Todo text"),
                            "status": _param("string", "pending, in_progress, completed, or blocked"),
                            "kind": _param("string", "inspect, change, verify, review, or report"),
                            "files": {"type": "array", "items": {"type": "string"}, "description": "Relevant files"},
                            "acceptance": _param("string", "Acceptance criteria"),
                        },
                    },
                },
            },
            required=("todos",),
            tags=("control", "state"),
        ),
        _tool(
            "todo_complete",
            "Mark one todo item as completed. Use this after finishing a tracked task, then continue with the next remaining item.",
            parameters={
                "id": _param("string", "Todo id to mark completed"),
            },
            required=("id",),
            tags=("control", "state"),
        ),
        _tool(
            "todo_clear",
            "Clear the current todo list when the tracked work is finished or no longer relevant.",
            parameters={},
            tags=("control", "state"),
        ),
        _tool(
            "exit_plan_mode",
            "Record the current plan summary and request leaving plan mode after approval.",
            parameters={
                "plan_summary": _param("string", "Short summary of the approved plan"),
            },
            required=("plan_summary",),
            tags=("control", "state"),
        ),
        _tool(
            "verify_skip",
            "Explicitly skip verification with a concrete blocker and next command.",
            parameters={
                "reason": _param("string", "Why verification could not be run"),
                "next_command": _param("string", "The next concrete command to try later"),
            },
            required=("reason", "next_command"),
            tags=("control", "state"),
        ),
        _tool(
            "read_file",
            "Read a file.",
            parameters={
                "path": _param("string", "File path"),
            },
            required=("path",),
            workflow_default=True,
            tags=("filesystem", "read"),
        ),
        _tool(
            "load_file",
            "Load a text file into context.",
            parameters={
                "path": _param("string", "File path"),
                "max_tokens": _param("integer", "Max tokens"),
            },
            required=("path",),
            workflow_default=True,
            tags=("filesystem", "read", "context"),
        ),
        _tool(
            "write_file",
            "Write a file.",
            parameters={
                "path": _param("string", "File path"),
                "content": _param("string", "File content"),
            },
            required=("path", "content"),
            confirmation_required=True,
            tags=("filesystem", "write"),
        ),
        _tool(
            "edit_file",
            (
                "Edit an existing file using one or more strict SEARCH/REPLACE blocks. "
                "Use this exact format for every patch: "
                "<<<<<<< SEARCH\\n...existing text...\\n=======\\n...replacement text...\\n>>>>>>> REPLACE"
            ),
            parameters={
                "path": _param("string", "File path"),
                "patch": _param(
                    "string",
                    "Required patch text. Send one or more SEARCH/REPLACE blocks using the exact existing text.",
                ),
                "old_string": _param("string", "Deprecated legacy exact text to replace"),
                "new_string": _param("string", "Deprecated legacy replacement text"),
                "replace_all": _param("boolean", "Replace all matches for the legacy fields"),
            },
            required=("path", "patch"),
            confirmation_required=True,
            tags=("filesystem", "write"),
        ),
        _tool(
            "glob",
            "Find files by glob.",
            parameters={
                "pattern": _param("string", "Glob pattern"),
                "path": _param("string", "Search root"),
            },
            required=("pattern",),
            workflow_default=True,
            tags=("filesystem", "read"),
        ),
        _tool(
            "grep",
            "Search file contents.",
            parameters={
                "pattern": _param("string", "Regex pattern"),
                "path": _param("string", "Search root"),
                "glob": _param("string", "File glob"),
                "ignore_case": _param("boolean", "Ignore case"),
            },
            required=("pattern",),
            workflow_default=True,
            tags=("filesystem", "read"),
        ),
        _tool(
            "list_directory",
            "List a directory.",
            parameters={
                "path": _param("string", "Directory path"),
            },
            workflow_default=True,
            tags=("filesystem", "read"),
        ),
    ]
)


def get_tool_spec(name: str) -> ToolSpec | None:
    return TOOL_REGISTRY.get(name)


def all_tool_specs() -> tuple[ToolSpec, ...]:
    return TOOL_REGISTRY.all_specs()


def all_tool_names() -> tuple[str, ...]:
    return TOOL_REGISTRY.all_names()


def runtime_tool_schemas() -> list[dict[str, object]]:
    return TOOL_REGISTRY.runtime_schemas()


def confirmation_required_tool_names() -> tuple[str, ...]:
    return TOOL_REGISTRY.names_requiring_confirmation()


def tool_names_with_tag(tag: str) -> tuple[str, ...]:
    return TOOL_REGISTRY.names_with_tag(tag)


def tool_names_for_mode(mode: str | None) -> tuple[str, ...]:
    return TOOL_REGISTRY.names_for_mode(mode)


def tool_bundle_names_for_mode(mode: str | None) -> tuple[str, ...]:
    return TOOL_REGISTRY.bundle_names_for_mode(mode)


def workflow_default_tool_names() -> tuple[str, ...]:
    return TOOL_REGISTRY.workflow_default_names()


def workflow_optional_tool_names() -> tuple[str, ...]:
    return TOOL_REGISTRY.workflow_optional_names()


def bind_tool_executor(name: str, executor: ToolExecutor) -> None:
    TOOL_REGISTRY.bind_executor(name, executor)
