"""Input completion engine for slash commands and @file mentions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath
import re
from typing import Protocol

from .commands import SlashCommandHandler


@dataclass(frozen=True)
class CompletionItem:
    label: str
    insert: str
    detail: str = ""


@dataclass
class CompletionState:
    start: int
    end: int
    items: list[CompletionItem]
    index: int = 0

    @property
    def selected(self) -> CompletionItem:
        return self.items[self.index]


class CompletionProvider(Protocol):
    def suggest(self, text: str) -> CompletionState | None:
        ...


class SlashCompletionProvider:
    def __init__(self, commands: SlashCommandHandler) -> None:
        self.commands = commands

    def suggest(self, text: str) -> CompletionState | None:
        if not text.startswith("/"):
            return None
        if " " in text[1:]:
            return None
        prefix = text[1:]
        matches = self.commands.matching_commands(prefix)
        if not matches:
            return None
        items = [
            CompletionItem(
                label=f"/{name}",
                insert=f"/{name}",
                detail=self.commands.command_description(name),
            )
            for name in matches
        ]
        return CompletionState(start=0, end=len(text), items=items)


class DeviceMentionCompletionProvider:
    TOKEN_RE = re.compile(r"(?:^|\s)@([^\s/]*)$")
    BRACKET_TOKEN_RE = re.compile(r"(?:^|\s)@\[([^\]]*)$")

    def __init__(self, sources_provider, *, max_items: int = 20) -> None:
        self.sources_provider = sources_provider
        self.max_items = max_items

    def suggest(self, text: str) -> CompletionState | None:
        bracketed = False
        match = self.BRACKET_TOKEN_RE.search(text)
        if match:
            bracketed = True
            prefix = match.group(1)
            at_index = match.start(1) - 2
        else:
            match = self.TOKEN_RE.search(text)
            if not match:
                return None
            prefix = match.group(1)
            at_index = match.start(1) - 1
        if "/" in prefix:
            return None
        needle = prefix.strip().lower()
        matches: list[CompletionItem] = []
        for ref, detail in self.sources_provider():
            if needle and not ref.lower().startswith(needle):
                continue
            insert = f"@[{ref}]" if bracketed else f"@{ref}"
            label = insert
            matches.append(CompletionItem(label=label, insert=insert, detail=detail))
            if len(matches) >= self.max_items:
                break
        if not matches:
            return None
        return CompletionState(start=at_index, end=len(text), items=matches)


class FileMentionCompletionProvider:
    TOKEN_RE = re.compile(r"(?:^|\s)@([^\s]*)$")
    BRACKET_TOKEN_RE = re.compile(r"(?:^|\s)@\[([^\]]*)$")

    def __init__(self, cwd: Path, *, max_items: int = 20) -> None:
        self.cwd = cwd.resolve()
        self.max_items = max_items

    def suggest(self, text: str) -> CompletionState | None:
        bracketed = False
        match = self.BRACKET_TOKEN_RE.search(text)
        if match:
            bracketed = True
            prefix = match.group(1)
            at_index = match.start(1) - 2
        else:
            match = self.TOKEN_RE.search(text)
            if not match:
                return None
            prefix = match.group(1)
            at_index = match.start(1) - 1
        entries = self._matching_paths(prefix)
        if not entries:
            return None

        items = [
            CompletionItem(
                label=f"@{display}" if not bracketed else f"@[{display}]",
                insert=f"@{insert}" if not bracketed else f"@[{insert}]",
                detail=kind,
            )
            for display, insert, kind in entries
        ]
        return CompletionState(start=at_index, end=len(text), items=items)

    def _matching_paths(self, prefix: str) -> list[tuple[str, str, str]]:
        # Keep suggestions relative to current workspace.
        normalized = prefix.replace("\\", "/")
        base_prefix = PurePosixPath(normalized)

        if normalized.endswith("/"):
            parent_rel = base_prefix
            name_prefix = ""
        elif str(base_prefix.parent) in ("", "."):
            parent_rel = PurePosixPath(".")
            name_prefix = base_prefix.name
        else:
            parent_rel = base_prefix.parent
            name_prefix = base_prefix.name

        parent = (self.cwd / str(parent_rel)).resolve()
        try:
            parent.relative_to(self.cwd)
        except ValueError:
            return []
        if not parent.exists() or not parent.is_dir():
            return []

        include_hidden = name_prefix.startswith(".")
        rows: list[tuple[str, str, str]] = []
        for entry in sorted(parent.iterdir(), key=lambda p: (p.is_file(), p.name.lower())):
            if not include_hidden and entry.name.startswith("."):
                continue
            if not entry.name.lower().startswith(name_prefix.lower()):
                continue
            rel = entry.relative_to(self.cwd).as_posix()
            if entry.is_dir():
                rel = f"{rel}/"
                rows.append((rel, rel, "dir"))
            else:
                rows.append((rel, rel, "file"))
            if len(rows) >= self.max_items:
                break
        return rows


class CompletionEngine:
    def __init__(self, providers: list[CompletionProvider]) -> None:
        self.providers = providers
        self.state: CompletionState | None = None

    def refresh(self, text: str) -> CompletionState | None:
        self.state = None
        for provider in self.providers:
            state = provider.suggest(text)
            if state:
                self.state = state
                return state
        return None

    def clear(self) -> None:
        self.state = None

    def cycle(self, delta: int) -> CompletionState | None:
        if not self.state:
            return None
        self.state.index = (self.state.index + delta) % len(self.state.items)
        return self.state

    def apply_selected(self, text: str) -> str:
        if not self.state:
            return text
        item = self.state.selected
        return text[: self.state.start] + item.insert + text[self.state.end :]
