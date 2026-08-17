import { chmod, mkdir, readFile, writeFile } from "node:fs/promises";
import { randomUUID } from "node:crypto";
import { extname, join } from "node:path";
import {
  type AgentSession,
  type AgentSessionEvent,
  type ToolDefinition,
  ModelRuntime,
  SessionManager,
  createAgentSession,
} from "@earendil-works/pi-coding-agent";
import type { ImageContent, TextContent, ThinkingLevel } from "@earendil-works/pi-ai";
import type { TSchema } from "typebox";

export interface OpenJetPiModel {
  provider: string;
  id: string;
  name: string;
  api: "openai-completions" | "openai-codex-responses";
  apiKey: string;
  baseUrl: string;
  reasoning: boolean;
  input: Array<"text" | "image">;
  contextWindow: number;
  maxTokens: number;
  samplingParams?: Record<string, unknown>;
  nativeContextWindow?: number;
  cost: { input: number; output: number; cacheRead: number; cacheWrite: number };
  compat?: Record<string, unknown>;
  headers?: Record<string, string>;
  thinkingLevel?: ThinkingLevel;
  thinkingLevelMap?: Record<string, string | null>;
}

export type AgentMode = "local" | "codex" | "hybrid";
export type ModelLane = "codex" | "local";

export interface ModelAttribution {
  lane: ModelLane;
  model: string;
  parentCallId?: string;
}

export interface OpenJetToolDescriptor {
  name: string;
  description: string;
  parameters: Record<string, unknown>;
}

export interface OpenJetToolResponse {
  ok: boolean;
  output: string;
  meta?: Record<string, unknown>;
  contextContent?: unknown;
}

export type OpenJetToolExecutor = (
  name: string,
  arguments_: Record<string, unknown>,
  callId: string,
  signal?: AbortSignal,
) => Promise<OpenJetToolResponse>;

type PiUiEventPayload =
  | { type: "assistant_start" }
  | { type: "text_delta"; text: string }
  | { type: "reasoning_delta"; text: string }
  | { type: "generation_chunk"; text: string }
  | { type: "tool_start"; callId: string; name: string; args: unknown }
  | { type: "tool_update"; callId: string; text: string }
  | { type: "tool_end"; callId: string; text: string; ok: boolean; details?: unknown }
  | { type: "compaction_start"; reason: string }
  | { type: "compaction_end"; ok: boolean; text: string; willRetry: boolean }
  | { type: "turn_complete"; stats: ReturnType<AgentSession["getSessionStats"]> }
  | { type: "notice"; text: string; level?: "info" | "warning" | "error" }
  | { type: "trace"; event: string; turnId: string; data: Record<string, unknown> };

export type PiUiEvent = PiUiEventPayload & Partial<ModelAttribution>;

export function modelConfigPayload(model: OpenJetPiModel): Record<string, unknown> {
  return {
    providers: {
      [model.provider]: {
        baseUrl: model.baseUrl,
        api: model.api,
        apiKey: model.apiKey,
        headers: model.headers,
        compat: model.compat,
        models: [{
          id: model.id,
          name: model.name,
          reasoning: model.reasoning,
          input: model.input,
          contextWindow: model.contextWindow,
          maxTokens: model.maxTokens,
          samplingParams: model.samplingParams,
          thinkingLevelMap: model.thinkingLevelMap,
          cost: model.cost,
        }],
      },
    },
  };
}

function contentText(value: unknown): string {
  if (typeof value === "string") return value;
  if (!Array.isArray(value)) return value == null ? "" : JSON.stringify(value);
  return value.map((item) => {
    if (!item || typeof item !== "object") return String(item ?? "");
    const row = item as Record<string, unknown>;
    return typeof row.text === "string" ? row.text : typeof row.content === "string" ? row.content : "";
  }).filter(Boolean).join("\n");
}

function mimeType(path: string): string {
  switch (extname(path).toLowerCase()) {
    case ".jpg": case ".jpeg": return "image/jpeg";
    case ".gif": return "image/gif";
    case ".webp": return "image/webp";
    default: return "image/png";
  }
}

const QUOTA_ERROR_PATTERNS = [
  "usage limit",
  "usage_limit",
  "quota",
  "rate limit",
  "rate_limit",
  "too many requests",
  "insufficient_quota",
  "429",
];

export function describeModelError(error: { errorMessage?: string } | undefined): string {
  const raw = error?.errorMessage?.trim();
  if (!raw) return "Pi model error (no detail reported by the provider).";
  const lowered = raw.toLowerCase();
  if (QUOTA_ERROR_PATTERNS.some((pattern) => lowered.includes(pattern))) {
    return `Run out of tokens — Codex usage limit reached. Switch to your local model with /mode local. Details: ${raw}`;
  }
  return raw;
}

export function sessionEventToUiEvents(event: AgentSessionEvent, attribution?: ModelAttribution): PiUiEvent[] {
  const attributed = (events: PiUiEventPayload[]): PiUiEvent[] => events.map((item) => ({ ...item, ...attribution }));
  if (event.type === "message_start" && event.message.role === "assistant") {
    return attributed([{ type: "assistant_start" }]);
  }
  if (event.type === "message_update") {
    const update = event.assistantMessageEvent;
    if (update.type === "text_delta") return attributed([
      { type: "generation_chunk", text: update.delta },
      { type: "text_delta", text: update.delta },
    ]);
    if (update.type === "thinking_delta") return attributed([{ type: "reasoning_delta", text: update.delta }]);
    if (update.type === "toolcall_delta") return attributed([{ type: "generation_chunk", text: update.delta }]);
    if (update.type === "error") {
      return attributed([{ type: "notice", text: describeModelError(update.error), level: "error" }]);
    }
  } else if (event.type === "tool_execution_start") {
    return attributed([{ type: "tool_start", callId: event.toolCallId, name: event.toolName, args: event.args }]);
  } else if (event.type === "tool_execution_update") {
    return attributed([{
      type: "tool_update",
      callId: event.toolCallId,
      text: contentText(event.partialResult?.content ?? event.partialResult),
    }]);
  } else if (event.type === "tool_execution_end") {
    return attributed([{
      type: "tool_end",
      callId: event.toolCallId,
      text: contentText(event.result?.content ?? event.result),
      ok: !event.isError,
      details: event.result?.details,
    }]);
  } else if (event.type === "compaction_start") {
    return attributed([{ type: "compaction_start", reason: event.reason }]);
  } else if (event.type === "compaction_end") {
    if (event.errorMessage) {
      return attributed([{ type: "compaction_end", ok: false, text: event.errorMessage, willRetry: false }]);
    }
    if (event.aborted) {
      return attributed([{ type: "compaction_end", ok: false, text: "Context compaction was aborted.", willRetry: false }]);
    }
    const before = event.result?.tokensBefore;
    const after = event.result?.estimatedTokensAfter;
    const detail = typeof before === "number" && typeof after === "number"
      ? ` (${before.toLocaleString()} → approximately ${after.toLocaleString()} tokens)`
      : "";
    return attributed([{
      type: "compaction_end",
      ok: true,
      text: `Context compaction completed${detail}.`,
      willRetry: event.willRetry,
    }]);
  } else if (event.type === "auto_retry_start") {
    return attributed([{
      type: "notice",
      text: `Retry ${event.attempt}/${event.maxAttempts}: ${event.errorMessage}`,
      level: "warning",
    }]);
  }
  return [];
}

export class OpenJetPiAgent {
  private session?: AgentSession;
  private unsubscribe?: () => void;
  private modelsPath?: string;
  private workspace = process.cwd();
  private mode: AgentMode = "local";
  private localModel?: OpenJetPiModel;
  private primaryModel?: OpenJetPiModel;
  private turnId = "";
  private openjetTools: OpenJetToolDescriptor[] = [];

  constructor(
    private readonly emit: (event: PiUiEvent) => void,
    private readonly executeOpenJetTool: OpenJetToolExecutor,
  ) {}

  async initialize(
    model: OpenJetPiModel,
    workspace: string,
    openjetTools: OpenJetToolDescriptor[] = [],
    mode: AgentMode = "local",
    localModel?: OpenJetPiModel,
  ): Promise<void> {
    const cwd = workspace || process.cwd();
    this.workspace = cwd;
    this.mode = mode;
    this.localModel = localModel;
    this.primaryModel = model;
    this.openjetTools = openjetTools;
    const agentDir = join(cwd, ".openjet", "pi");
    const sessionsDir = join(agentDir, "sessions");
    const modelsPath = join(agentDir, "models.json");
    this.modelsPath = modelsPath;
    await mkdir(sessionsDir, { recursive: true, mode: 0o700 });
    await chmod(agentDir, 0o700);
    await this.writeModelConfig(modelsPath, [model, ...(localModel && localModel.id !== model.id ? [localModel] : [])]);

    const modelRuntime = await ModelRuntime.create({
      modelsPath,
      allowModelNetwork: false,
      refreshOnCreate: true,
    });
    const selectedModel = modelRuntime.getModel(model.provider, model.id);
    if (!selectedModel) throw new Error(`Pi could not load OpenJet model ${model.provider}/${model.id}`);
    const customTools = openjetTools.map((tool) => this.createOpenJetTool(tool));
    if (mode === "hybrid") customTools.push(this.createDelegateLocalTool());
    const created = await createAgentSession({
      cwd,
      agentDir,
      modelRuntime,
      model: selectedModel,
      thinkingLevel: model.reasoning ? model.thinkingLevel ?? "medium" : "off",
      sessionManager: SessionManager.create(cwd, sessionsDir),
      customTools,
    });
    this.session = created.session;
    this.unsubscribe = this.session.subscribe((event) => this.handleEvent(event));
    if (created.modelFallbackMessage) this.emit({ type: "notice", text: created.modelFallbackMessage, level: "warning" });
  }

  async switchModel(model: OpenJetPiModel): Promise<void> {
    if (!this.session || !this.modelsPath) throw new Error("Pi agent session is not initialized.");
    await this.writeModelConfig(this.modelsPath, [model]);
    const modelRuntime = await ModelRuntime.create({
      modelsPath: this.modelsPath,
      allowModelNetwork: false,
      refreshOnCreate: true,
    });
    const selectedModel = modelRuntime.getModel(model.provider, model.id);
    if (!selectedModel) throw new Error(`Pi could not load OpenJet model ${model.provider}/${model.id}`);
    await this.session.setModel(selectedModel);
    this.primaryModel = model;
    this.session.setThinkingLevel(model.reasoning ? model.thinkingLevel ?? "medium" : "off");
  }

  private async writeModelConfig(modelsPath: string, models: OpenJetPiModel[]): Promise<void> {
    const providers: Record<string, unknown> = {};
    for (const model of models) Object.assign(providers, (modelConfigPayload(model).providers as object));
    // Holds provider access tokens: keep it owner-only, and re-apply on rewrite
    // because writeFile's mode is ignored for an existing file.
    await writeFile(modelsPath, JSON.stringify({ providers }, null, 2), { encoding: "utf8", mode: 0o600 });
    await chmod(modelsPath, 0o600);
  }

  get ready(): boolean { return Boolean(this.session); }
  get active(): boolean { return Boolean(this.session?.isStreaming); }

  async prompt(text: string, imagePaths: string[] = []): Promise<void> {
    if (!this.session) throw new Error("Pi agent session is not initialized.");
    this.turnId = randomUUID();
    const attribution = this.primaryAttribution();
    this.emitTrace("model_turn_start", attribution, { mode: this.mode });
    const images: ImageContent[] = await Promise.all(imagePaths.map(async (path) => ({
      type: "image" as const,
      data: (await readFile(path)).toString("base64"),
      mimeType: mimeType(path),
    })));
    const prompt = this.mode === "hybrid"
      ? `<hybrid-orchestrator>Own planning, architecture, risk decisions, and final review. Delegate substantial exploration, implementation, repetitive edits, testing, and debugging to delegate_local with concrete acceptance criteria. Keep your token use compact: consume the worker's concise handoff and inspect only the relevant diff and verification evidence.</hybrid-orchestrator>\n\n${text}`
      : text;
    await this.session.prompt(prompt, { images, source: "interactive" });
  }

  async abort(): Promise<void> { await this.session?.abort(); }

  dispose(): void {
    this.unsubscribe?.();
    this.unsubscribe = undefined;
    this.session?.dispose();
    this.session = undefined;
    this.modelsPath = undefined;
  }

  private createOpenJetTool(descriptor: OpenJetToolDescriptor): ToolDefinition {
    return {
      name: descriptor.name,
      label: descriptor.name.replaceAll("_", " "),
      description: descriptor.description,
      promptSnippet: descriptor.description,
      parameters: descriptor.parameters as TSchema,
      executionMode: "sequential",
      execute: async (callId, params, signal) => {
        const response = await this.executeOpenJetTool(
          descriptor.name,
          (params ?? {}) as Record<string, unknown>,
          callId,
          signal,
        );
        if (!response.ok) throw new Error(response.output || `${descriptor.name} failed`);
        return {
          content: await this.openJetResultContent(response),
          details: response.meta ?? {},
        };
      },
    };
  }

  private createDelegateLocalTool(): ToolDefinition {
    return {
      name: "delegate_local",
      label: "Delegate locally",
      description: "Delegate substantial repository exploration, implementation, edits, tests, or debugging to the warm local model. Return only a concise implementation handoff.",
      promptSnippet: "Use delegate_local for the high-volume implementation loop; retain planning and final review.",
      parameters: {
        type: "object",
        properties: {
          task: { type: "string", description: "Complete bounded implementation task" },
          acceptance_criteria: { type: "string", description: "Required behavior and verification" },
        },
        required: ["task"],
        additionalProperties: false,
      } as TSchema,
      executionMode: "sequential",
      execute: async (callId, params) => this.runLocalWorker(callId, params as Record<string, unknown>),
    };
  }

  private async runLocalWorker(parentCallId: string, params: Record<string, unknown>): Promise<{ content: TextContent[]; details: Record<string, unknown> }> {
    if (!this.localModel || !this.modelsPath) throw new Error("Slipstream local model is not configured.");
    const runtime = await ModelRuntime.create({ modelsPath: this.modelsPath, allowModelNetwork: false, refreshOnCreate: true });
    const model = runtime.getModel(this.localModel.provider, this.localModel.id);
    if (!model) throw new Error(`Pi could not load local worker ${this.localModel.provider}/${this.localModel.id}`);
    const worker = await createAgentSession({
      cwd: this.workspace,
      agentDir: join(this.workspace, ".openjet", "pi-worker"),
      modelRuntime: runtime,
      model,
      thinkingLevel: this.localModel.reasoning ? "medium" : "off",
      sessionManager: SessionManager.inMemory(this.workspace),
      customTools: this.openjetTools.map((tool) => this.createOpenJetTool(tool)),
    });
    const task = String(params.task ?? "").trim();
    const acceptance = String(params.acceptance_criteria ?? "").trim();
    const attribution: ModelAttribution = { lane: "local", model: this.localModel.name, parentCallId };
    const unsubscribe = worker.session.subscribe((event) => {
      for (const uiEvent of sessionEventToUiEvents(event, attribution)) this.emit(uiEvent);
      if (event.type === "tool_execution_start") {
        this.emitTrace("model_tool_start", attribution, { callId: event.toolCallId, tool: event.toolName });
      } else if (event.type === "tool_execution_end") {
        this.emitTrace("model_tool_end", attribution, { callId: event.toolCallId, tool: event.toolName, ok: !event.isError });
      }
    });
    this.emitTrace("delegation_start", attribution, {
      taskChars: task.length,
      acceptanceCriteriaChars: acceptance.length,
    });
    try {
      await worker.session.prompt(
        `You are the local implementation agent. Inspect, edit, test, and iterate autonomously. Do not delegate.\n\nTask:\n${task}${acceptance ? `\n\nAcceptance criteria:\n${acceptance}` : ""}\n\nReturn only a concise handoff: files changed, tests and outcomes, and remaining risks.`,
        { source: "interactive" },
      );
      const assistant = [...worker.session.messages].reverse().find((message) => message.role === "assistant");
      const text = contentText(assistant?.content).trim() || "Local worker completed without a textual handoff.";
      const stats = worker.session.getSessionStats();
      this.emitTrace("delegation_end", attribution, { ok: true, ...tokenTraceData(stats.tokens), cost: stats.cost });
      return { content: [{ type: "text", text }], details: { local: true, stats } };
    } catch (error) {
      this.emitTrace("delegation_end", attribution, { ok: false, error: error instanceof Error ? error.message : String(error) });
      throw error;
    } finally {
      unsubscribe();
      worker.session.dispose();
    }
  }

  private async openJetResultContent(response: OpenJetToolResponse): Promise<Array<TextContent | ImageContent>> {
    const content: Array<TextContent | ImageContent> = [{ type: "text", text: response.output }];
    if (!Array.isArray(response.contextContent)) return content;
    for (const item of response.contextContent) {
      if (!item || typeof item !== "object") continue;
      const block = item as Record<string, unknown>;
      if (block.type !== "input_image" || typeof block.path !== "string") continue;
      content.push({
        type: "image",
        data: (await readFile(block.path)).toString("base64"),
        mimeType: typeof block.mime_type === "string" ? block.mime_type : mimeType(block.path),
      });
    }
    return content;
  }

  private handleEvent(event: AgentSessionEvent): void {
    const attribution = this.primaryAttribution();
    for (const uiEvent of sessionEventToUiEvents(event, attribution)) this.emit(uiEvent);
    if (event.type === "tool_execution_start") {
      this.emitTrace("model_tool_start", attribution, { callId: event.toolCallId, tool: event.toolName });
    } else if (event.type === "tool_execution_end") {
      this.emitTrace("model_tool_end", attribution, { callId: event.toolCallId, tool: event.toolName, ok: !event.isError });
    }
    if (event.type === "agent_settled" && this.session) {
      const stats = this.session.getSessionStats();
      this.emitTrace("model_turn_end", attribution, { ...tokenTraceData(stats.tokens), cost: stats.cost });
      this.emit({ type: "turn_complete", stats, ...attribution });
    }
  }

  private primaryAttribution(): ModelAttribution {
    return {
      lane: this.mode === "local" ? "local" : "codex",
      model: this.primaryModel?.name ?? "unknown model",
    };
  }

  private emitTrace(event: string, attribution: ModelAttribution, data: Record<string, unknown>): void {
    this.emit({
      type: "trace",
      event,
      turnId: this.turnId,
      data: { ...data, lane: attribution.lane, model: attribution.model, parentCallId: attribution.parentCallId ?? "" },
    });
  }
}

function tokenTraceData(tokens: ReturnType<AgentSession["getSessionStats"]>["tokens"]): Record<string, number> {
  return {
    inputTokens: tokens.input,
    outputTokens: tokens.output,
    cacheReadTokens: tokens.cacheRead,
    cacheWriteTokens: tokens.cacheWrite,
  };
}
