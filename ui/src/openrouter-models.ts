import type { OpenJetPiModel } from "./pi-agent.js";
import { CURATED_OPENROUTER_MODELS as GENERATED_OPENROUTER_MODELS } from "./openrouter-models.generated.js";

export interface OpenRouterPickerModel {
  id: string;
  name: string;
  contextWindow: number;
  maxTokens: number;
  reasoning: boolean;
  cost: { input: number; output: number; cacheRead: number; cacheWrite: number };
  featured?: boolean;
}

export const CURATED_OPENROUTER_MODELS: OpenRouterPickerModel[] = GENERATED_OPENROUTER_MODELS.map((entry) => ({
  id: entry.id,
  name: entry.name,
  contextWindow: entry.contextWindow,
  maxTokens: entry.maxTokens,
  reasoning: entry.reasoning,
  cost: { ...entry.cost },
  featured: entry.featured,
}));

let snapshotModels: OpenRouterPickerModel[] = CURATED_OPENROUTER_MODELS;

export function setOpenRouterPickerModels(models: OpenRouterPickerModel[]): void {
  if (models.length) snapshotModels = models;
}

export function formatOpenRouterPrice(cost: { input: number; output: number }): string {
  if (cost.input <= 0 && cost.output <= 0) return "free";
  return `${formatPerMillion(cost.input)} in / ${formatPerMillion(cost.output)} out`;
}

export function formatOpenRouterContext(tokens: number): string {
  if (tokens >= 1_000_000) {
    const millions = tokens / 1_000_000;
    const rounded = Math.round(millions);
    return Math.abs(millions - rounded) < 0.05 ? `${rounded}M ctx` : `${millions.toFixed(1)}M ctx`;
  }
  if (tokens >= 1_000) {
    const thousands = tokens / 1_000;
    return thousands % 1 === 0 ? `${thousands}K ctx` : `${thousands.toFixed(1)}K ctx`;
  }
  return `${tokens} ctx`;
}

export function listOpenRouterPickerModels(): OpenRouterPickerModel[] {
  return snapshotModels;
}

export const OPENROUTER_SET_KEY_VALUE = "__openrouter_set_key__";

export function openRouterPickerItems(options?: { connected?: boolean }): Array<{ value: string; label: string; description: string }> {
  const models = listOpenRouterPickerModels().map((model) => ({
    value: model.id,
    label: model.featured ? `${model.name} · featured` : model.name,
    description: `${formatOpenRouterPrice(model.cost)} · ${formatOpenRouterContext(model.contextWindow)} · ${model.id}`,
  }));
  if (options?.connected) return models;
  return [
    {
      value: OPENROUTER_SET_KEY_VALUE,
      label: "Paste OpenRouter API key",
      description: "Get a key at openrouter.ai/keys — saved to the OS keyring, not a config file",
    },
    ...models,
  ];
}

export function enrichPiModel(model: OpenJetPiModel): OpenJetPiModel {
  const catalogId = model.id.replace(/^openrouter\//, "");
  const catalog = listOpenRouterPickerModels().find((entry) => entry.id === catalogId || entry.id === model.id);
  if (!catalog) return model;
  return {
    ...model,
    provider: "openrouter",
    id: catalog.id,
    name: catalog.name,
    cost: catalog.cost,
  };
}

function formatPerMillion(value: number): string {
  if (value <= 0) return "$0";
  if (value < 0.01) return `$${value.toFixed(4)}`;
  if (value < 1) return `$${value.toFixed(2)}`;
  return `$${value % 1 === 0 ? value.toFixed(0) : value.toFixed(2)}`;
}
