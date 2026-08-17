import type { SlashCommand } from "@earendil-works/pi-tui";

export const bootstrapSlashCommands: SlashCommand[] = [
  { name: "status", description: "Show runtime and model status" },
  { name: "device", description: "Show the active compute device" },
  { name: "devices", description: "List available compute devices" },
  { name: "sources", description: "Show configured model sources" },
  { name: "setup", description: "Configure the local model runtime" },
  { name: "runtime", description: "Inspect or change the runtime" },
  { name: "model", description: "Inspect or change the active model" },
  { name: "mode", description: "Choose Local, Codex, or Hybrid mode" },
  { name: "effort", description: "Set Codex reasoning effort" },
  { name: "exit", description: "Exit OpenJet" },
  { name: "quit", description: "Exit OpenJet" },
];

export function ctrlCAction(inputText: string): "clear" | "exit" {
  return inputText.length > 0 ? "clear" : "exit";
}

export function isRapidSecondEscape(previousAt: number, currentAt: number, windowMs = 500): boolean {
  return previousAt > 0 && currentAt >= previousAt && currentAt - previousAt <= windowMs;
}

export function slashCommandsFromPayload(payload: Record<string, unknown>): SlashCommand[] {
  const commands = Array.isArray(payload.commands) ? payload.commands : [];
  const result = new Map(bootstrapSlashCommands.map((command) => [command.name, command]));

  for (const item of commands) {
    if (!item || typeof item !== "object") continue;
    const row = item as Record<string, unknown>;
    if (typeof row.name !== "string") continue;
    const description = typeof row.description === "string" ? row.description : undefined;
    result.set(row.name, { name: row.name, description });
    if (Array.isArray(row.aliases)) {
      for (const alias of row.aliases) {
        if (typeof alias === "string" && alias) {
          result.set(alias, { name: alias, description: description ? `${description} (alias for /${row.name})` : `Alias for /${row.name}` });
        }
      }
    }
  }

  return [...result.values()];
}
