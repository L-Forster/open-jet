import { registerBunOAuthFlows } from "@earendil-works/pi-ai/bun-oauth";
registerBunOAuthFlows();

import chalk from "chalk";
import * as clipboard from "@mariozechner/clipboard";
import { randomUUID } from "node:crypto";
import { unlink, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import {
  CombinedAutocompleteProvider,
  Container,
  Editor,
  ProcessTerminal,
  SelectList,
  Spacer,
  Text,
  TuiMainScreen,
  matchesKey,
  type TUI,
} from "@earendil-works/pi-tui";
import { AssistantMessage, BrandHeader, PickerPanel, ReasoningBlock, SectionDivider, StatusFooter, ToolCard, UserMessage, compactModel, formatDuration, modelTokenUsage, toolLabel, totalModelTokens, type FooterState, type ModelTokenUsage } from "./components.js";
import {
  OpenJetPiAgent,
  type AgentMode,
  type OpenJetPiModel,
  type OpenJetToolDescriptor,
  type OpenJetToolResponse,
  type PiUiEvent,
} from "./pi-agent.js";
import { OpenJetRpcClient } from "./rpc-client.js";
import type { ProtocolMessage } from "./protocol.js";
import { editorTheme, palette, selectTheme } from "./theme.js";
import { promptDisposition } from "./startup-queue.js";
import { bootstrapSlashCommands, ctrlCAction, isRapidSecondEscape, isUnknownSlashCommand, slashCommandsFromPayload } from "./editor-features.js";
import { enrichPiModel, setOpenRouterPickerModels, type OpenRouterPickerModel } from "./openrouter-models.js";
import { loginOpenRouter, pickOpenRouterModel, storedOpenRouterApiKey } from "./pi-openrouter.js";

const terminal = new ProcessTerminal();
const tui: TUI = new TuiMainScreen(terminal);
const transcript = new Container();
const editorContainer = new Container();
const editor = new Editor(tui, editorTheme, { autocompleteMaxVisible: 12 });
editorContainer.addChild(editor);
editor.setAutocompleteProvider(new CombinedAutocompleteProvider(bootstrapSlashCommands, process.cwd()));
const footer = new StatusFooter({ workspace: process.cwd(), status: "starting…" });
const rpc = new OpenJetRpcClient();
const piAgent = new OpenJetPiAgent(handlePiEvent, async (name, arguments_, callId, signal) => {
  const response = await rpc.call("tool_execute", { callId, payload: { name, arguments: arguments_ } }, signal);
  return (response.payload ?? {}) as unknown as OpenJetToolResponse;
});
const toolCards = new Map<string, ToolCard>();
const toolDetails: ToolCard[] = [];
let assistant: AssistantMessage | undefined;
let assistantText = "";
let reasoningText = "";
let reasoningComponent: ReasoningBlock | undefined;
let reasoningExpanded = false;
let turnReasoningBlocks: ReasoningBlock[] = [];
let hasSubmittedTurn = false;
let activeSection: SectionDivider | undefined;
let activeSectionName = "";
let activeSectionStartedAt = 0;
let turnStartedAt = 0;
let turnActive = false;
let shuttingDown = false;
let pendingImagePaths: string[] = [];
let activeImagePaths: string[] = [];
let queuedPrompt: { text: string; imagePaths: string[]; status: Text } | undefined;
let piInitialization: Promise<void> | undefined;
let setupRequired: boolean | undefined;
let generationChunks: Array<{ text: string; timestampMs: number }> = [];
let generationFlushTimer: ReturnType<typeof setTimeout> | undefined;
let activeAgentMode: AgentMode = "local";
let activeModelLane = "";
let activeModelLabel = "";
let completedTurns = 0;
let tipIndex = 0;
let showedStartupTip = false;
let lastEscapeAt = 0;
let hybridLocalTokenUsage: ModelTokenUsage = { input: 0, cache: 0, output: 0 };
let modelProfiles: Array<{ name: string; kind: string; model: string }> = [];
let activeLocalProfile = "";
let activeCodexModel = "gpt-5.6-sol";
let activeCodexEffort = "medium";
let activeLocalReasoning = true;
let activeOpenrouterModel = "openrouter/stealth/ox-alpha";
let activeOrchestratorKind = "codex";
let openrouterConnected = false;
let codexModelOptions = ["gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna"];
const serviceCommands = new Set(["status", "device", "devices", "sources", "setup", "runtime", "model", "mode", "agent", "strategy", "effort", "reasoning", "cloud", "connect", "login", "exit", "quit"]);
// Until the backend command list arrives, pass slash commands through instead of
// rejecting them — the bootstrap list is only a subset of what the backend knows.
let backendCommandsLoaded = false;
const AGENT_TIPS = [
  "Tip: /login is Pi's OpenRouter login (account or API key). /cloud is Pi's OpenRouter model list.",
];

function clearActiveImages(): void {
  const paths = activeImagePaths;
  activeImagePaths = [];
  for (const path of paths) void unlink(path).catch(() => undefined);
}

function clearPendingImages(): void {
  const paths = pendingImagePaths;
  pendingImagePaths = [];
  for (const path of paths) void unlink(path).catch(() => undefined);
}

function handleCtrlC(): void {
  if (ctrlCAction(editor.getText()) === "clear") {
    editor.setText("");
    clearPendingImages();
    tui.requestRender();
    return;
  }
  void shutdown(0);
}

function imageExtension(bytes: Uint8Array): string {
  if (bytes[0] === 0x89 && bytes[1] === 0x50) return "png";
  if (bytes[0] === 0xff && bytes[1] === 0xd8) return "jpg";
  if (bytes[0] === 0x47 && bytes[1] === 0x49) return "gif";
  if (bytes[0] === 0x52 && bytes[1] === 0x49) return "webp";
  return "png";
}

async function pasteClipboard(): Promise<void> {
  try {
    if (clipboard.hasImage()) {
      const bytes = Uint8Array.from(await clipboard.getImageBinary());
      const path = join(tmpdir(), `openjet-paste-${randomUUID()}.${imageExtension(bytes)}`);
      await writeFile(path, bytes);
      pendingImagePaths.push(path);
      editor.insertTextAtCursor(`[image:${pendingImagePaths.length}]`);
      addNotice(`Attached image ${pendingImagePaths.length}.`);
      return;
    }
    if (clipboard.hasText()) editor.insertTextAtCursor(await clipboard.getText());
  } catch (error) {
    addNotice(`Clipboard paste failed: ${error instanceof Error ? error.message : String(error)}`, "warning");
  }
}

function add(component: Parameters<Container["addChild"]>[0]): void {
  transcript.addChild(component);
  tui.requestRender();
}

function addNotice(text: string, level: "info" | "warning" | "error" = "info"): Text {
  const color = level === "error" ? palette.errorSoft : level === "warning" ? palette.warning : palette.muted;
  const notice = new Text(chalk.hex(color)(text), 1, 0);
  add(notice);
  return notice;
}

function finishActiveSection(now = performance.now()): void {
  if (activeSection) activeSection.complete(now - activeSectionStartedAt);
  activeSection = undefined;
  activeSectionName = "";
}

function beginSection(name: "thinking" | "tools" | "response"): void {
  const label = activeModelLabel ? `${activeModelLabel} · ${name}` : name;
  if (activeSectionName === label) return;
  const now = performance.now();
  finishActiveSection(now);
  activeSection = new SectionDivider(label);
  activeSectionName = label;
  activeSectionStartedAt = now;
  add(activeSection);
}

function ensureModelLane(event: PiUiEvent): void {
  if (!event.lane || !event.model || activeModelLane === `${event.lane}:${event.model}`) return;
  finishActiveSection();
  assistant = undefined;
  assistantText = "";
  reasoningComponent = undefined;
  reasoningText = "";
  activeModelLane = `${event.lane}:${event.model}`;
  activeModelLabel = `${event.lane.toUpperCase()} · ${compactModel(event.model)}`;
}

function showNextTip(): void {
  addNotice(AGENT_TIPS[tipIndex % AGENT_TIPS.length]);
  tipIndex += 1;
}

function updateFooter(payload: Record<string, unknown> = {}, status?: string): void {
  const rawStatus = (payload.status && typeof payload.status === "object" ? payload.status : {}) as Record<string, unknown>;
  const next: FooterState = {};
  if (typeof payload.airgapped === "boolean") next.airgapped = payload.airgapped;
  if (typeof payload.workspace === "string") next.workspace = payload.workspace;
  if (typeof payload.model === "string") next.model = payload.model;
  if (typeof payload.runtime === "string") next.runtime = payload.runtime;
  if (typeof payload.agentMode === "string") next.agentMode = payload.agentMode;
  if (typeof payload.agentLocalProfile === "string") activeLocalProfile = payload.agentLocalProfile;
  if (typeof payload.localReasoning === "boolean") activeLocalReasoning = payload.localReasoning;
  if (Array.isArray(payload.codexModelOptions)) {
    codexModelOptions = payload.codexModelOptions.filter((value): value is string => typeof value === "string");
  }
  if (typeof payload.openrouterConnected === "boolean") openrouterConnected = payload.openrouterConnected;
  if (Array.isArray(payload.openrouterModels)) {
    const models = payload.openrouterModels.filter((row): row is OpenRouterPickerModel => (
      Boolean(row) && typeof row === "object"
      && typeof (row as OpenRouterPickerModel).id === "string"
      && typeof (row as OpenRouterPickerModel).name === "string"
    ));
    if (models.length) setOpenRouterPickerModels(models);
  }
  if (typeof payload.orchestratorKind === "string") {
    activeOrchestratorKind = payload.orchestratorKind;
    next.orchestratorKind = payload.orchestratorKind;
  }
  if (payload.agentChanged === true) next.codexShare = undefined;
  if (payload.agentChanged === true) next.savedTokenUsage = undefined;
  if (Array.isArray(payload.modelProfiles)) {
    modelProfiles = payload.modelProfiles.filter((row): row is { name: string; kind: string; model: string } => (
      Boolean(row) && typeof row === "object"
      && typeof (row as Record<string, unknown>).name === "string"
      && typeof (row as Record<string, unknown>).kind === "string"
      && typeof (row as Record<string, unknown>).model === "string"
    ));
  }
  if (payload.agentMode === "local" || payload.agentMode === "codex" || payload.agentMode === "hybrid") {
    activeAgentMode = payload.agentMode;
  }
  const localModel = payload.localModel as Record<string, unknown> | undefined;
  const codexModel = payload.codexModel as Record<string, unknown> | undefined;
  const orchestratorModel = payload.orchestratorModel as Record<string, unknown> | undefined;
  if (typeof localModel?.name === "string") next.localModel = localModel.name;
  if (typeof orchestratorModel?.name === "string") {
    next.orchestratorModel = orchestratorModel.name;
    if (activeOrchestratorKind === "openrouter") {
      activeOpenrouterModel = orchestratorModel.id as string || orchestratorModel.name;
    }
  }
  if (typeof codexModel?.name === "string") {
    next.codexModel = codexModel.name;
    activeCodexModel = codexModel.name;
  }
  if (typeof codexModel?.thinkingLevel === "string") {
    next.codexEffort = codexModel.thinkingLevel;
    activeCodexEffort = codexModel.thinkingLevel;
  }
  if (typeof rawStatus.contextTokens === "number") next.contextTokens = rawStatus.contextTokens;
  if (typeof rawStatus.contextWindow === "number") next.contextWindow = rawStatus.contextWindow;
  if (typeof rawStatus.completionTokens === "number") next.completionTokens = rawStatus.completionTokens;
  if (typeof rawStatus.cost === "string") next.cost = rawStatus.cost;
  if (typeof rawStatus.tps === "number") next.tps = rawStatus.tps;
  if (typeof rawStatus.powerWatts === "number") next.powerWatts = rawStatus.powerWatts;
  if (typeof rawStatus.powerPercent === "number") next.powerPercent = rawStatus.powerPercent;
  if (typeof rawStatus.cpuPercent === "number") next.cpuPercent = rawStatus.cpuPercent;
  if (typeof rawStatus.memoryPercent === "number") next.memoryPercent = rawStatus.memoryPercent;
  if (typeof rawStatus.batteryPercent === "number") next.batteryPercent = rawStatus.batteryPercent;
  if (typeof rawStatus.batteryStatus === "string") next.batteryStatus = rawStatus.batteryStatus;
  if (typeof rawStatus.temperatureC === "number") next.temperatureC = rawStatus.temperatureC;
  if (typeof rawStatus.device === "string") next.device = rawStatus.device;
  if (status !== undefined) next.status = status;
  footer.setState(next);
  footer.invalidate();
  tui.requestRender();
}

function configureCommands(payload: Record<string, unknown>): void {
  const commands = slashCommandsFromPayload(payload);
  for (const command of commands) serviceCommands.add(command.name.toLowerCase());
  backendCommandsLoaded = true;
  editor.setAutocompleteProvider(new CombinedAutocompleteProvider(commands, process.cwd()));
}

function pickValue(
  title: string,
  items: Array<{ value: string; label: string; description?: string }>,
  selectedValue?: string,
): Promise<string | undefined> {
  const selectedIndex = Math.max(0, items.findIndex((item) => item.value === selectedValue));
  const picker = new SelectList(items, Math.min(8, items.length), selectTheme);
  picker.setSelectedIndex(selectedIndex);
  const panel = new PickerPanel(title, picker);
  const overlay = tui.showOverlay(panel, {
    width: "100%",
    maxHeight: 12,
    anchor: "bottom-center",
    margin: { left: 1, right: 1, bottom: 4 },
  });
  return new Promise((resolve) => {
    picker.onSelect = (item) => {
      overlay.hide();
      resolve(item.value);
    };
    picker.onCancel = () => {
      overlay.hide();
      resolve(undefined);
    };
  });
}

async function configureCurrentModel(): Promise<void> {
  const source = await pickValue("Choose a model", [
    { value: "openrouter", label: "OpenRouter", description: "Pi hosted models. /login then /cloud." },
    { value: "local", label: "Local", description: "llama.cpp GGUF on this machine" },
    { value: "codex", label: "Codex", description: "ChatGPT subscription via /connect openai-codex" },
    { value: "hybrid-local", label: "Slipstream local worker", description: "Change only the local implementer" },
  ], activeAgentMode === "hybrid" ? "hybrid-local" : activeOrchestratorKind === "openrouter" ? "openrouter" : activeAgentMode);
  if (!source) return;
  if (source === "openrouter") {
    await configureOpenRouterModel();
    return;
  }
  if (source === "codex") {
    await configureCodexModel();
    return;
  }
  const options: string[] = [];
  const localProfiles = modelProfiles.filter((profile) => profile.kind === "local");
  if (!localProfiles.length) {
    addNotice("No saved local model profiles. Run /setup first.", "warning");
    return;
  }
  const local = await pickValue("Configure model · Local model", localProfiles.map((profile) => ({
    value: profile.name,
    label: `Local · ${profile.name}`,
    description: profile.model,
  })), activeLocalProfile);
  if (!local) return;
  const reasoning = await pickValue("Configure model · Local reasoning", [
    { value: "on", label: "Reasoning on", description: "Use the local model's thinking mode" },
    { value: "off", label: "Reasoning off", description: "Use the local instruct response mode" },
  ], activeLocalReasoning ? "on" : "off");
  if (!reasoning) return;
  options.push(`local=${JSON.stringify(local)}`, `reasoning=${reasoning}`);
  rpc.request("command", { text: `/model ${options.join(" ")}` });
}

async function configureCodexModel(): Promise<void> {
  const model = await pickValue(
    "Configure model · Codex model",
    codexModelOptions.map((value) => ({ value, label: `Codex · ${value}`, description: "ChatGPT subscription model" })),
    activeCodexModel,
  );
  if (!model) return;
  const effort = await pickValue(
    "Configure model · Codex effort",
    ["none", "low", "medium", "high", "xhigh", "max"].map((value) => ({ value, label: value, description: "Codex reasoning effort" })),
    activeCodexEffort,
  );
  if (!effort) return;
  rpc.request("command", { text: `/model codex=${model} effort=${effort}` });
}

const piUi = () => ({ tui, editorContainer, editor });

async function saveOpenRouterKey(key: string): Promise<boolean> {
  try {
    const response = await rpc.call("command", { text: "/connect openrouter", apiKey: key });
    const payload = (response.payload ?? {}) as Record<string, unknown>;
    updateFooter(payload);
    if (response.text) addNotice(response.text);
    return openrouterConnected;
  } catch (error) {
    addNotice(error instanceof Error ? error.message : String(error), "error");
    return false;
  }
}

async function ensureOpenRouterKey(): Promise<boolean> {
  const existing = storedOpenRouterApiKey();
  if (existing) return saveOpenRouterKey(existing);
  if (openrouterConnected) return true;
  try {
    const key = await loginOpenRouter(piUi());
    if (!key) return false;
    return saveOpenRouterKey(key);
  } catch (error) {
    addNotice(error instanceof Error ? error.message : String(error), "error");
    return false;
  }
}

async function handleLoginCommand(): Promise<void> {
  try {
    const key = await loginOpenRouter(piUi());
    if (!key) return;
    const saved = await saveOpenRouterKey(key);
    if (saved) addNotice("OpenRouter login saved. Use /cloud to pick a model.");
  } catch (error) {
    addNotice(error instanceof Error ? error.message : String(error), "error");
  }
}

async function configureOpenRouterModel(): Promise<void> {
  const selected = await pickOpenRouterModel({
    ...piUi(),
    cwd: process.cwd(),
    currentModelId: activeOpenrouterModel.replace(/^openrouter\//, ""),
  });
  if (!selected) return;
  if (!openrouterConnected && !(await ensureOpenRouterKey())) return;
  rpc.request("command", { text: `/cloud ${selected}` });
}

async function handleConnectCommand(text: string): Promise<void> {
  const rest = text.slice("/connect".length).trim();
  if (!rest || rest === "openrouter" || rest.startsWith("openrouter ")) {
    const key = rest.startsWith("openrouter ") ? rest.slice("openrouter ".length).trim() : "";
    if (key) {
      await saveOpenRouterKey(key);
      return;
    }
    await handleLoginCommand();
    return;
  }
  rpc.request("command", { text });
}

async function configureCodexEffort(): Promise<void> {
  if (activeOrchestratorKind !== "codex") {
    addNotice("Codex effort applies only when Codex is the orchestrator.", "warning");
    return;
  }
  const effort = await pickValue(
    "Codex effort",
    ["none", "low", "medium", "high", "xhigh", "max"].map((value) => ({ value, label: value, description: "Codex reasoning effort" })),
    activeCodexEffort,
  );
  if (effort) rpc.request("command", { text: `/effort ${effort}` });
}

async function shutdown(code = 0): Promise<never> {
  if (!shuttingDown) {
    shuttingDown = true;
    clearInterval(metricsPoll);
    if (generationFlushTimer) clearTimeout(generationFlushTimer);
    generationFlushTimer = undefined;
    generationChunks = [];
    tui.stop();
    piAgent.dispose();
    await rpc.close();
  }
  process.exit(code);
}

function handlePiEvent(event: PiUiEvent): void {
  if (event.type === "trace") {
    rpc.request("agent_trace", { payload: { event: event.event, turnId: event.turnId, data: event.data } });
    return;
  }
  ensureModelLane(event);
  if (event.type === "assistant_start") {
    // Pi emits a new assistant message after every tool-result round. Start new
    // components so later model output stays after the tools that caused it.
    assistant = undefined;
    assistantText = "";
    reasoningComponent = undefined;
    reasoningText = "";
  } else if (event.type === "generation_chunk") {
    queueGenerationChunk(event.text);
  } else if (event.type === "text_delta") {
    if (!assistant) {
      beginSection("response");
      assistantText = "";
      assistant = new AssistantMessage();
      add(assistant);
    }
    assistantText += event.text;
    assistant.setText(assistantText);
  } else if (event.type === "reasoning_delta") {
    reasoningText += event.text;
    if (!reasoningComponent) {
      reasoningComponent = new ReasoningBlock(reasoningExpanded);
      turnReasoningBlocks.push(reasoningComponent);
      add(reasoningComponent);
    }
    reasoningComponent.setText(reasoningText);
  } else if (event.type === "tool_start") {
    beginSection("tools");
    const card = new ToolCard(event.callId, event.name, event.args);
    toolCards.set(event.callId, card);
    toolDetails.push(card);
    add(card);
    updateFooter({}, `${toolLabel(event.name)}…`);
  } else if (event.type === "tool_update") {
    updateFooter({}, "tool output…");
  } else if (event.type === "tool_end") {
    const card = toolCards.get(event.callId);
    card?.complete(event.text, event.ok, true, event.details);
    if (card?.name === "delegate_local" && event.details && typeof event.details === "object") {
      const stats = (event.details as Record<string, unknown>).stats as Record<string, unknown> | undefined;
      const tokens = stats?.tokens as Record<string, unknown> | undefined;
      const usage = modelTokenUsage(tokens);
      hybridLocalTokenUsage = {
        input: hybridLocalTokenUsage.input + usage.input,
        cache: hybridLocalTokenUsage.cache + usage.cache,
        output: hybridLocalTokenUsage.output + usage.output,
      };
      footer.setState({ savedTokenUsage: hybridLocalTokenUsage });
    }
    updateFooter({}, "working…");
  } else if (event.type === "compaction_start") {
    editor.disableSubmit = true;
    addNotice(`Compacting context (${event.reason})…`);
    updateFooter({}, "compacting context…");
  } else if (event.type === "compaction_end") {
    addNotice(event.text, event.ok ? "info" : "error");
    updateFooter({}, event.ok ? (event.willRetry ? "retrying…" : "working…") : "compaction failed");
  } else if (event.type === "history_user") {
    beginSection("response");
    finishActiveSection();
    add(new UserMessage(event.text));
    hasSubmittedTurn = true;
  } else if (event.type === "history_assistant") {
    beginSection("response");
    const restored = new AssistantMessage();
    restored.setText(event.text);
    add(restored);
    finishActiveSection();
  } else if (event.type === "notice") {
    addNotice(event.text, event.level ?? "info");
  } else if (event.type === "turn_complete") {
    const now = performance.now();
    finishActiveSection(now);
    flushGenerationChunks();
    rpc.request("generation_metrics", { payload: { phase: "end" } });
    clearActiveImages();
    turnActive = false;
    editor.disableSubmit = false;
    assistant = undefined;
    reasoningComponent = undefined;
    reasoningText = "";
    activeModelLane = "";
    activeModelLabel = "";
    completedTurns += 1;
    footer.setState({
      contextTokens: event.stats.contextUsage?.tokens ?? undefined,
      contextWindow: event.stats.contextUsage?.contextWindow,
      completionTokens: event.stats.tokens.output,
      savedTokenUsage: activeAgentMode === "local"
        ? modelTokenUsage(event.stats.tokens as unknown as Record<string, unknown>)
        : activeAgentMode === "hybrid" ? hybridLocalTokenUsage : undefined,
      codexShare: activeAgentMode === "hybrid"
        ? totalModelTokens(event.stats.tokens as unknown as Record<string, unknown>)
          / Math.max(1, totalModelTokens(event.stats.tokens as unknown as Record<string, unknown>)
            + hybridLocalTokenUsage.input + hybridLocalTokenUsage.cache + hybridLocalTokenUsage.output)
        : undefined,
      cost: event.stats.cost > 0 ? `$${event.stats.cost.toFixed(4)}` : undefined,
      status: turnStartedAt > 0 ? `ready · ${formatDuration(now - turnStartedAt)}` : "ready",
    });
    if (completedTurns % 5 === 0) showNextTip();
  }
  tui.requestRender();
}

function startPiPrompt(text: string, imagePaths: string[]): void {
  turnStartedAt = performance.now();
  turnActive = true;
  editor.disableSubmit = true;
  activeImagePaths = imagePaths;
  updateFooter({}, "working…");
  rpc.request("generation_metrics", { payload: { phase: "start" } });
  void piAgent.prompt(text, activeImagePaths).catch((error) => {
    addNotice(error instanceof Error ? error.message : String(error), "error");
    clearActiveImages();
    turnActive = false;
    editor.disableSubmit = false;
    updateFooter({}, "error");
  });
}

function flushGenerationChunks(): void {
  if (generationFlushTimer) clearTimeout(generationFlushTimer);
  generationFlushTimer = undefined;
  if (generationChunks.length === 0) return;
  const chunks = generationChunks;
  generationChunks = [];
  rpc.request("generation_metrics", { payload: { phase: "chunks", chunks } });
}

function queueGenerationChunk(text: string): void {
  generationChunks.push({ text, timestampMs: performance.now() });
  generationFlushTimer ??= setTimeout(flushGenerationChunks, 100);
}

function flushQueuedPrompt(): void {
  if (!piAgent.ready || !queuedPrompt || turnActive) return;
  const pending = queuedPrompt;
  queuedPrompt = undefined;
  transcript.removeChild(pending.status);
  startPiPrompt(pending.text, pending.imagePaths);
}

function handlePiInitializationError(error: unknown): void {
  const message = error instanceof Error ? error.message : String(error);
  if (queuedPrompt) {
    transcript.removeChild(queuedPrompt.status);
    pendingImagePaths = [...queuedPrompt.imagePaths, ...pendingImagePaths];
    editor.setText(queuedPrompt.text);
    queuedPrompt = undefined;
    editor.disableSubmit = false;
    addNotice(`Model initialization failed; queued message restored: ${message}`, "error");
  } else {
    addNotice(`Model initialization failed: ${message}`, "error");
  }
  updateFooter({}, "initialization failed");
}

function initializePi(payload: Record<string, unknown>, replace = false): Promise<void> {
  if (piAgent.ready && !replace) {
    flushQueuedPrompt();
    return Promise.resolve();
  }
  if (piInitialization) return piInitialization;
  const rawModel = payload.piModel as OpenJetPiModel | undefined;
  if (!rawModel) return Promise.resolve();
  const model = enrichPiModel(rawModel);
  const localModel = payload.localModel
    ? enrichPiModel(payload.localModel as OpenJetPiModel)
    : undefined;
  if (replace) piAgent.dispose();
  updateFooter(payload, "initializing…");
  const openjetTools = Array.isArray(payload.openjetTools)
    ? payload.openjetTools as OpenJetToolDescriptor[]
    : [];
  const attempt = piAgent.initialize(
    model,
    typeof payload.workspace === "string" ? payload.workspace : process.cwd(),
    openjetTools,
    (payload.agentMode ?? "local") as AgentMode,
    localModel,
  ).then(() => {
    updateFooter(payload, "ready");
    if (!showedStartupTip) {
      showedStartupTip = true;
      showNextTip();
    }
    flushQueuedPrompt();
  }).catch(handlePiInitializationError).finally(() => {
    if (piInitialization === attempt) piInitialization = undefined;
  });
  piInitialization = attempt;
  return attempt;
}

function handleMessage(message: ProtocolMessage): void {
  const payload = message.payload ?? {};
  if (typeof payload.setupRequired === "boolean") {
    setupRequired = payload.setupRequired;
  }
  switch (message.type) {
    case "ready":
      setupRequired = payload.setupRequired === true;
      configureCommands(payload);
      updateFooter(payload, payload.setupRequired ? "setup required" : "starting Pi…");
      if (payload.setupRequired) {
        if (queuedPrompt) {
          transcript.removeChild(queuedPrompt.status);
          pendingImagePaths = [...queuedPrompt.imagePaths, ...pendingImagePaths];
          editor.setText(queuedPrompt.text);
          queuedPrompt = undefined;
          editor.disableSubmit = false;
        }
        addNotice("No model is configured. Run /setup recommended to provision the hardware-aware default.", "warning");
      } else {
        void initializePi(payload);
      }
      break;
    case "state_snapshot":
      updateFooter(payload, turnActive ? undefined : "ready");
      break;
    case "notification":
      if (message.text) addNotice(message.text, (payload.level as "info" | "warning" | "error") ?? "info");
      if (payload && typeof payload === "object") updateFooter(payload, turnActive ? "working…" : "ready");
      if (payload.openCloudPicker === true) {
        void configureOpenRouterModel();
      }
      if (payload.openConnectPicker === true || payload.needsApiKey === "openrouter") {
        void handleLoginCommand();
      }
      if (payload.piModel && payload.agentChanged === true) {
        setupRequired = false;
        hybridLocalTokenUsage = { input: 0, cache: 0, output: 0 };
        void initializePi(payload, true).then(
          () => addNotice(`Mode ready: ${String(payload.agentMode ?? "local")}.`),
          (error) => addNotice(`Mode switch failed: ${error instanceof Error ? error.message : String(error)}`, "error"),
        );
      } else if (payload.piModel && !piAgent.ready) {
        setupRequired = false;
        void initializePi(payload);
      } else if (payload.piModel && payload.modelChanged === true) {
        void piAgent.switchModel(enrichPiModel(payload.piModel as unknown as OpenJetPiModel)).then(
          () => addNotice("Model switched; the current session context was preserved."),
          (error) => addNotice(`Model switch failed: ${error instanceof Error ? error.message : String(error)}`, "error"),
        );
      }
      if (payload.exit) void shutdown(0);
      break;
    case "status_update":
      if (payload.turnActive === true) {
        turnActive = true;
        editor.disableSubmit = true;
        updateFooter({}, "working…");
      } else if (payload.turnActive === false) {
        clearActiveImages();
        turnActive = false;
        editor.disableSubmit = Boolean(queuedPrompt);
        if (!queuedPrompt) updateFooter({}, "ready");
      } else if (message.text !== undefined) {
        updateFooter({}, message.text || "working…");
      }
      break;
    case "error":
      addNotice(message.text ?? "Unknown OpenJet error.", "error");
      turnActive = false;
      editor.disableSubmit = false;
      updateFooter({}, "error");
      if (payload.fatal) clearActiveImages();
      if (payload.fatal) void shutdown(1);
      break;
  }
}

function busyTurnBlocks(action: string): boolean {
  if (!turnActive) return false;
  addNotice(`Wait for the current turn to finish before ${action}.`, "warning");
  return true;
}

async function resetConversation(): Promise<void> {
  if (busyTurnBlocks("starting a new conversation")) return;
  transcript.clear();
  toolDetails.length = 0;
  assistant = undefined;
  assistantText = "";
  reasoningText = "";
  reasoningComponent = undefined;
  reasoningExpanded = false;
  turnReasoningBlocks = [];
  hasSubmittedTurn = false;
  queuedPrompt = undefined;
  clearActiveImages();
  try {
    await piAgent.reset();
    addNotice("Started a new conversation.");
  } catch (error) {
    addNotice(`Could not start a new conversation: ${error instanceof Error ? error.message : String(error)}`, "error");
  }
}

async function resumeConversation(): Promise<void> {
  if (busyTurnBlocks("resuming a chat")) return;
  let sessions;
  try {
    sessions = await piAgent.listSessions();
  } catch (error) {
    addNotice(`Could not list saved chats: ${error instanceof Error ? error.message : String(error)}`, "error");
    return;
  }
  if (!sessions.length) {
    addNotice("No saved chats found for this workspace.", "warning");
    return;
  }
  const picker = new SelectList(
    sessions.slice(0, 20).map((session) => ({
      value: session.path,
      label: session.timestamp ? new Date(session.timestamp).toLocaleString() : session.id,
      description: session.preview || session.id,
    })),
    8,
    selectTheme,
  );
  const overlay = tui.showOverlay(picker, { width: 72, maxHeight: 10, anchor: "center" });
  picker.onSelect = (item) => {
    overlay.hide();
    void applyResume(item.value);
  };
  picker.onCancel = () => overlay.hide();
}

async function applyResume(path: string): Promise<void> {
  if (busyTurnBlocks("resuming a chat")) return;
  transcript.clear();
  toolDetails.length = 0;
  toolCards.clear();
  assistant = undefined;
  assistantText = "";
  reasoningText = "";
  reasoningComponent = undefined;
  reasoningExpanded = false;
  turnReasoningBlocks = [];
  hasSubmittedTurn = false;
  queuedPrompt = undefined;
  clearActiveImages();
  try {
    await piAgent.openSession(path);
    addNotice("Chat restored.");
  } catch (error) {
    addNotice(`Could not resume chat: ${error instanceof Error ? error.message : String(error)}`, "error");
  }
}

editor.onSubmit = (value) => {
  const text = value.trim();
  if (!text || turnActive) return;
  if (backendCommandsLoaded && isUnknownSlashCommand(text, serviceCommands)) {
    const token = text.slice(1).split(/\s+/, 1)[0] || "";
    editor.addToHistory(text);
    addNotice(token ? `Unknown command /${token}.` : "Unknown command.", "warning");
    return;
  }
  if (text === "/mode" || text === "/agent" || text === "/strategy") {
    editor.addToHistory(text);
    const picker = new SelectList([
      { value: "hybrid", label: "Slipstream", description: "Codex or OpenRouter plans and reviews; the local model implements" },
      { value: "openrouter", label: "OpenRouter", description: "Hosted models via API key, with pricing in /model and /cloud" },
      { value: "codex", label: "Codex", description: "Codex handles the complete task" },
      { value: "local", label: "Local", description: "The local model handles the complete task" },
    ], 4, selectTheme);
    const overlay = tui.showOverlay(picker, { width: 72, maxHeight: 8, anchor: "center" });
    picker.onSelect = (item) => {
      overlay.hide();
      rpc.request("command", { text: `/mode ${item.value}` });
    };
    picker.onCancel = () => overlay.hide();
    return;
  }
  if (text === "/model") {
    editor.addToHistory(text);
    void configureCurrentModel();
    return;
  }
  if (/^\/(?:clear|new)$/i.test(text)) {
    editor.addToHistory("/clear");
    void resetConversation();
    return;
  }
  if (/^\/resume$/i.test(text)) {
    editor.addToHistory("/resume");
    void resumeConversation();
    return;
  }
  if (/^\/cloud$/i.test(text)) {
    editor.addToHistory("/cloud");
    void configureOpenRouterModel();
    return;
  }
  if (/^\/login(?:\s|$)/i.test(text)) {
    editor.addToHistory("/login");
    void handleLoginCommand();
    return;
  }
  if (/^\/connect(?:\s|$)/i.test(text)) {
    // History stores the bare command only so an inline API key never lands in
    // the editor history or any transcript echo.
    editor.addToHistory("/connect");
    void handleConnectCommand(text);
    return;
  }
  if (text === "/effort") {
    editor.addToHistory(text);
    void configureCodexEffort();
    return;
  }
  editor.addToHistory(text);
  if (hasSubmittedTurn) add(new Spacer(1));
  add(new UserMessage(text));
  hasSubmittedTurn = true;
  finishActiveSection();
  assistant = undefined;
  assistantText = "";
  reasoningText = "";
  reasoningComponent = undefined;
  reasoningExpanded = false;
  turnReasoningBlocks = [];
  const command = text.startsWith("/") ? text.slice(1).split(/\s+/, 1)[0].toLowerCase() : "";
  if (command && serviceCommands.has(command)) {
    rpc.request("command", { text });
  } else {
    const imagePaths = pendingImagePaths;
    pendingImagePaths = [];
    const disposition = promptDisposition(piAgent.ready, setupRequired);
    if (disposition !== "send") {
      if (disposition === "queue") {
        const status = addNotice("Message queued — waiting for the local agent to finish initializing.");
        queuedPrompt = { text, imagePaths, status };
        editor.disableSubmit = true;
      } else {
        pendingImagePaths = imagePaths;
        addNotice("No model is configured. Run /setup recommended first.", "warning");
      }
      return;
    }
    startPiPrompt(text, imagePaths);
  }
};

tui.addInputListener((data) => {
  const escapePressed = matchesKey(data, "escape");
  if (!escapePressed) lastEscapeAt = 0;
  if (matchesKey(data, "ctrl+v") || matchesKey(data, "alt+v")) {
    void pasteClipboard();
    return { consume: true };
  }
  if (matchesKey(data, "ctrl+c")) {
    handleCtrlC();
    return { consume: true };
  }
  if (matchesKey(data, "ctrl+t")) {
    reasoningExpanded = !reasoningExpanded;
    for (const block of turnReasoningBlocks) block.setExpanded(reasoningExpanded);
    tui.requestRender();
    return { consume: true };
  }
  if (matchesKey(data, "ctrl+o")) {
    toolDetails.at(-1)?.toggle();
    tui.requestRender();
    return { consume: true };
  }
  if (escapePressed) {
    const now = performance.now();
    if (isRapidSecondEscape(lastEscapeAt, now)) {
      lastEscapeAt = 0;
      editor.setText("");
      clearPendingImages();
      tui.requestRender();
      return { consume: true };
    }
    lastEscapeAt = now;
    if (turnActive) {
      void piAgent.abort();
      return { consume: true };
    }
  }
  return undefined;
});

process.on("SIGINT", handleCtrlC);
process.on("SIGTERM", () => void shutdown(0));
process.on("uncaughtException", (error) => {
  tui.stop();
  process.stderr.write(`OpenJet TUI failure: ${error.stack ?? error.message}\n`);
  process.exit(1);
});

tui.addChild(new BrandHeader());
tui.addChild(new Spacer(1));
tui.addChild(transcript);
tui.addChild(new Spacer(1));
tui.addChild(editorContainer);
tui.addChild(footer);
tui.setFocus(editor);
rpc.onMessage(handleMessage);
rpc.start();
tui.start();
rpc.request("initialize", { width: process.stdout.columns || 80, height: process.stdout.rows || 24 });
const metricsPoll = setInterval(() => {
  if (!shuttingDown) rpc.request("status", {});
}, 1000);

process.stdout.on("resize", () => {
  rpc.request("resize", { width: process.stdout.columns || 80, height: process.stdout.rows || 24 });
});
