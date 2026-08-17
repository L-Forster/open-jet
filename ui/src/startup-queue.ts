export type PromptDisposition = "send" | "queue" | "setup";

export function promptDisposition(agentReady: boolean, setupRequired: boolean | undefined): PromptDisposition {
  if (agentReady) return "send";
  if (setupRequired === true) return "setup";
  return "queue";
}
