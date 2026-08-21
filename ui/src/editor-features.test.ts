import { describe, expect, it } from "vitest";
import { CombinedAutocompleteProvider, Editor, type TUI } from "@earendil-works/pi-tui";
import { editorTheme } from "./theme.js";
import { bootstrapSlashCommands, ctrlCAction, isRapidSecondEscape, isUnknownSlashCommand, slashCommandsFromPayload } from "./editor-features.js";

function fakeTui(): TUI {
  return {
    terminal: { rows: 24 },
    requestRender() {},
  } as unknown as TUI;
}

describe("Pi editor parity", () => {
  it("recalls submitted prompts with up and down arrows", () => {
    const editor = new Editor(fakeTui(), editorTheme);
    editor.addToHistory("first prompt");
    editor.addToHistory("second prompt");

    editor.handleInput("\u001b[A");
    expect(editor.getText()).toBe("second prompt");
    editor.handleInput("\u001b[A");
    expect(editor.getText()).toBe("first prompt");
    editor.handleInput("\u001b[B");
    expect(editor.getText()).toBe("second prompt");
  });

  it("offers slash-command previews before the backend is ready", async () => {
    const provider = new CombinedAutocompleteProvider(bootstrapSlashCommands, process.cwd());
    const suggestions = await provider.getSuggestions(["/"], 0, 1, { signal: new AbortController().signal });
    const names = suggestions?.items.map((item) => item.value) ?? [];

    expect(names.slice(0, 5)).toEqual(["login", "cloud", "connect", "model", "mode"]);
    expect(names).toContain("setup");
  });

  it("rejects unknown slash commands instead of treating them as prompts", () => {
    const known = bootstrapSlashCommands.map((command) => command.name);
    expect(isUnknownSlashCommand("/login", known)).toBe(false);
    expect(isUnknownSlashCommand("/cloud ox-alpha", known)).toBe(false);
    expect(isUnknownSlashCommand("/not-a-command", known)).toBe(true);
    expect(isUnknownSlashCommand("/", known)).toBe(true);
    expect(isUnknownSlashCommand("hello", known)).toBe(false);
  });

  it("adds backend commands and their aliases to completion", () => {
    const commands = slashCommandsFromPayload({
      commands: [{ name: "clear", description: "Clear the conversation", aliases: ["new"] }],
    });

    expect(commands).toContainEqual({ name: "clear", description: "Clear the conversation" });
    expect(commands).toContainEqual({ name: "new", description: "Clear the conversation (alias for /clear)" });
  });

  it("uses Ctrl+C to clear once, then exit when the input is empty", () => {
    expect(ctrlCAction("unfinished prompt")).toBe("clear");
    expect(ctrlCAction("")).toBe("exit");
  });

  it("recognizes two Escape presses within the clear-input window", () => {
    expect(isRapidSecondEscape(1_000, 1_450)).toBe(true);
    expect(isRapidSecondEscape(1_000, 1_501)).toBe(false);
    expect(isRapidSecondEscape(0, 100)).toBe(false);
  });
});
