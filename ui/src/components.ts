import chalk from "chalk";
import { homedir } from "node:os";
import { basename, relative } from "node:path";
import type { Component } from "@earendil-works/pi-tui";
import { Markdown, truncateToWidth, visibleWidth } from "@earendil-works/pi-tui";
import { markdownTheme, palette } from "./theme.js";

const CONTENT_MAX_WIDTH = 104;
const WORDMARK = [
  " ██████╗ ██████╗ ███████╗███╗   ██╗      ██╗███████╗████████╗",
  "██╔═══██╗██╔══██╗██╔════╝████╗  ██║      ██║██╔════╝╚══██╔══╝",
  "██║   ██║██████╔╝█████╗  ██╔██╗ ██║      ██║█████╗     ██║   ",
  "██║   ██║██╔═══╝ ██╔══╝  ██║╚██╗██║ ██   ██║██╔══╝     ██║   ",
  "╚██████╔╝██║     ███████╗██║ ╚████║ ╚█████╔╝███████╗   ██║   ",
  " ╚═════╝ ╚═╝     ╚══════╝╚═╝  ╚═══╝  ╚════╝ ╚══════╝   ╚═╝   ",
];

export class BrandHeader implements Component {
  invalidate(): void {}
  render(width: number): string[] {
    const brand = chalk.hex(palette.green).bold("OPENJET");
    if (width < 72) return [` ${brand}${width >= 52 ? chalk.hex(palette.dim)("  local coding agent") : ""}`];
    return WORDMARK.map((line, index) => chalk.hex([
      palette.green,
      palette.green,
      palette.greenBright,
      palette.greenSoft,
      palette.greenPale,
      palette.dim,
    ][index])(line));
  }
}

export class UserMessage implements Component {
  constructor(private readonly text: string) {}
  invalidate(): void {}
  render(width: number): string[] {
    const contentWidth = Math.min(width, CONTENT_MAX_WIDTH);
    const rows = this.text.split("\n");
    return rows.map((row, index) => {
      const clipped = truncateToWidth(
        index === 0
        ? ` ${chalk.hex(palette.green).bold("›")} ${chalk.hex(palette.muted)(row)}`
        : `   ${chalk.hex(palette.muted)(row)}`,
        contentWidth,
      );
      const padded = clipped + " ".repeat(Math.max(0, contentWidth - visibleWidth(clipped)));
      return chalk.bgHex(palette.userSurface)(padded);
    });
  }
}

export class SectionDivider implements Component {
  private elapsedMs?: number;
  constructor(private readonly label: string) {}
  complete(elapsedMs: number): void { this.elapsedMs = Math.max(0, elapsedMs); }
  invalidate(): void {}
  render(width: number): string[] {
    const contentWidth = Math.min(width, CONTENT_MAX_WIDTH);
    const detail = this.elapsedMs == null ? this.label : `${this.label} · ${formatDuration(this.elapsedMs)}`;
    const prefix = `── ${detail} `;
    const rule = "─".repeat(Math.max(2, contentWidth - visibleWidth(prefix) - 1));
    return [chalk.hex(palette.border)(` ${prefix}${rule}`)];
  }
}

export class PickerPanel implements Component {
  constructor(
    private readonly title: string,
    private readonly picker: Component & { handleInput(data: string): void },
  ) {}
  handleInput(data: string): void { this.picker.handleInput(data); }
  invalidate(): void { this.picker.invalidate(); }
  render(width: number): string[] {
    const panelWidth = Math.max(24, width);
    const innerWidth = Math.max(18, panelWidth - 4);
    const title = truncateToWidth(this.title, Math.max(1, innerWidth - 2));
    const topFill = "─".repeat(Math.max(0, innerWidth - visibleWidth(title) - 1));
    const paint = (line: string) => chalk.bgHex(palette.surfaceRaised)(line);
    const content = this.picker.render(innerWidth).map((row) => {
      const clipped = truncateToWidth(row, innerWidth);
      const padded = clipped + " ".repeat(Math.max(0, innerWidth - visibleWidth(clipped)));
      return paint(`│ ${padded} │`);
    });
    return [
      paint(chalk.hex(palette.border)(`╭─ ${title} ${topFill}╮`)),
      ...content,
      paint(chalk.hex(palette.border)(`╰${"─".repeat(panelWidth - 2)}╯`)),
    ];
  }
}

export class AssistantMessage implements Component {
  private readonly markdown = new Markdown("", 1, 0, markdownTheme);
  setText(text: string): void { this.markdown.setText(text); }
  invalidate(): void { this.markdown.invalidate(); }
  render(width: number): string[] { return this.markdown.render(Math.min(width, CONTENT_MAX_WIDTH)); }
}

export class ReasoningBlock implements Component {
  private text = "";
  constructor(private expanded = false) {}
  setText(text: string): void { this.text = text; }
  setExpanded(expanded: boolean): void { this.expanded = expanded; }
  invalidate(): void {}
  render(width: number): string[] {
    if (!this.expanded || !this.text) return [];
    const contentWidth = Math.min(width, CONTENT_MAX_WIDTH);
    return [
      ` ${chalk.hex(palette.dim)("reasoning")}`,
      ...this.text.trim().split("\n").map((row) => truncateToWidth(`   ${chalk.hex(palette.dim)(row)}`, contentWidth)),
    ];
  }
}

export interface FooterState {
  agentMode?: string;
  localModel?: string;
  codexModel?: string;
  codexEffort?: string;
  codexShare?: number;
  airgapped?: boolean;
  workspace?: string;
  model?: string;
  runtime?: string;
  contextTokens?: number;
  contextWindow?: number;
  completionTokens?: number;
  savedTokenUsage?: ModelTokenUsage;
  cost?: string;
  tps?: number;
  powerWatts?: number;
  powerPercent?: number;
  cpuPercent?: number;
  memoryPercent?: number;
  batteryPercent?: number;
  batteryStatus?: string;
  temperatureC?: number;
  device?: string;
  status?: string;
}

export interface ModelTokenUsage {
  input: number;
  cache: number;
  output: number;
}

export function modelTokenUsage(tokens: Record<string, unknown> | undefined): ModelTokenUsage {
  const value = (key: string): number => typeof tokens?.[key] === "number" ? tokens[key] as number : 0;
  return {
    input: value("input"),
    cache: value("cacheRead") + value("cacheWrite"),
    output: value("output"),
  };
}

export function totalModelTokens(tokens: Record<string, unknown> | undefined): number {
  const usage = modelTokenUsage(tokens);
  return usage.input + usage.cache + usage.output;
}

export class StatusFooter implements Component {
  constructor(private state: FooterState = {}) {}
  setState(next: FooterState): void { this.state = { ...this.state, ...next }; }
  invalidate(): void {}
  render(width: number): string[] {
    const mode = (this.state.agentMode || "local").toUpperCase();
    const context = this.state.contextWindow
      ? `${compactNumber(this.state.contextTokens ?? 0)} / ${compactNumber(this.state.contextWindow)}`
      : "ctx —";
    const workspace = compactWorkspace(this.state.workspace ?? process.cwd());
    const model = compactModel(this.state.model || "no model");
    const runtime = compactRuntime(this.state.runtime || "local");
    const localModel = compactModel(this.state.localModel || this.state.model || "no local model");
    const codexModel = compactModel(this.state.codexModel || "Codex");
    const modelPair = mode === "HYBRID"
      ? `${codexModel} ${this.state.codexEffort || "medium"} + ${localModel}`
      : mode === "CODEX" ? `${codexModel} ${this.state.codexEffort || "medium"}` : localModel;
    const output = `${compactNumber(this.state.completionTokens ?? 0)} out${this.state.cost ? ` · ${this.state.cost}` : ""}`;
    const savedTotal = this.state.savedTokenUsage
      ? this.state.savedTokenUsage.input + this.state.savedTokenUsage.cache + this.state.savedTokenUsage.output
      : undefined;
    const saved = savedTotal == null ? "" : `${compactNumber(savedTotal)} tokens saved by local model`;
    const savedDetail = this.state.savedTokenUsage
      ? `${saved} · ${compactNumber(this.state.savedTokenUsage.input)} input, ${compactNumber(this.state.savedTokenUsage.cache)} cached, ${compactNumber(this.state.savedTokenUsage.output)} output`
      : "";
    const share = mode === "HYBRID" && this.state.codexShare != null
      ? `Codex ${(this.state.codexShare * 100).toFixed(0)}%`
      : "";
    const performance = this.state.tps == null ? "tok/s —" : `${this.state.tps.toFixed(1)} tok/s`;
    const hardware = [
      this.state.device || compactRuntime(this.state.runtime || "local").toUpperCase(),
      this.state.cpuPercent == null ? "" : `CPU ${this.state.cpuPercent.toFixed(0)}%`,
      this.state.memoryPercent == null ? "" : `RAM ${this.state.memoryPercent.toFixed(0)}%`,
      this.state.powerWatts == null ? "" : `${this.state.powerWatts.toFixed(0)} W`,
      this.state.temperatureC == null ? "" : `${this.state.temperatureC.toFixed(0)}°C`,
      this.state.batteryPercent == null ? "" : `BAT ${this.state.batteryPercent.toFixed(0)}%`,
    ].filter(Boolean).join(chalk.hex(palette.border)("  ·  "));
    const status = this.state.status || "ready";
    const statusColor = /error|fail/i.test(status)
      ? palette.errorSoft
      : /working|starting|running|initializing|streaming/i.test(status) ? palette.warning : palette.greenSoft;
    const identityParts = width >= 120
      ? [
          chalk.hex(palette.green).bold(mode),
          chalk.hex(palette.muted)(modelPair),
          chalk.hex(palette.muted)(workspace),
          chalk.hex(palette.dim)(runtime),
          chalk.hex(palette.dim)(context),
          chalk.hex(palette.dim)(output),
          chalk.hex(palette.dim)(share),
          chalk.hex(statusColor)(status),
        ]
      : width >= 80
        ? [
            chalk.hex(palette.green).bold(mode),
            chalk.hex(palette.muted)(modelPair),
            chalk.hex(palette.dim)(context),
            chalk.hex(statusColor)(status),
          ]
        : [
            chalk.hex(palette.green).bold(mode),
            chalk.hex(palette.muted)(modelPair),
            chalk.hex(statusColor)(status),
          ];
    const identity = padFooterLine(` ${identityParts.filter(Boolean).join(chalk.hex(palette.border)("  ·  "))}`, width);
    const telemetryParts = width >= 64
      ? [chalk.hex(palette.greenSoft).bold(performance), hardware]
      : [chalk.hex(palette.greenSoft).bold(performance)];
    const telemetry = padFooterLine(` ${telemetryParts.filter(Boolean).join(chalk.hex(palette.border)("  ·  "))}`, width);
    const savings = savedDetail
      ? padFooterLine(` ${chalk.hex(palette.greenSoft).bold(savedDetail)}`, width)
      : undefined;
    return savings ? [identity, savings, telemetry] : [identity, telemetry];
  }
}

function padFooterLine(line: string, width: number): string {
  const clipped = truncateToWidth(line, Math.max(1, width));
  return clipped + " ".repeat(Math.max(0, width - visibleWidth(clipped)));
}

export class ToolCard implements Component {
  private result?: { text: string; ok: boolean; approved: boolean; details?: unknown };
  private expanded = false;
  private elapsedMs?: number;
  private readonly startedAt: number;
  constructor(
    readonly callId: string,
    readonly name: string,
    readonly args: unknown,
    private readonly now: () => number = () => performance.now(),
  ) { this.startedAt = this.now(); }
  complete(text: string, ok: boolean, approved: boolean, details?: unknown): void {
    this.result = { text, ok, approved, details };
    this.elapsedMs = Math.max(0, this.now() - this.startedAt);
  }
  toggle(): void { this.expanded = !this.expanded; }
  invalidate(): void {}
  render(width: number): string[] {
    const contentWidth = Math.min(width, CONTENT_MAX_WIDTH);
    const glyph = !this.result ? "◇" : this.result.ok ? "✓" : "×";
    const color = !this.result ? palette.warning : this.result.ok ? palette.green : palette.error;
    const summary = toolCallSummary(this.name, this.args);
    const label = toolLabel(this.name);
    const timing = this.elapsedMs == null ? "" : ` · ${formatDuration(this.elapsedMs)}`;
    const lines = [truncateToWidth(` ${chalk.hex(color)(glyph)} ${chalk.hex(palette.muted)(label)}${summary ? `  ${chalk.hex(palette.dim)(summary)}` : ""}${chalk.hex(palette.dim)(timing)}`, contentWidth)];
    const diff = this.name === "edit" ? toolResultDiff(this.result?.details) : undefined;
    if (diff) {
      for (const row of renderOpenJetDiff(diff)) {
        lines.push(truncateToWidth(`  ${row}`, contentWidth));
      }
    }
    if (this.expanded) {
      const args = JSON.stringify(this.args, null, 2);
      for (const row of args.split("\n")) lines.push(truncateToWidth(chalk.hex(palette.dim)(`  ${row}`), contentWidth));
      if (this.result?.text) {
        for (const row of this.result.text.split("\n")) lines.push(truncateToWidth(`  ${row}`, contentWidth));
      }
    } else if (this.result && !this.result.ok && this.result.text) {
      lines.push(truncateToWidth(chalk.hex(palette.errorSoft)(`  ${this.result.text.split("\n")[0]}`), contentWidth));
    }
    return lines;
  }
}

function toolResultDiff(details: unknown): string | undefined {
  if (!details || typeof details !== "object" || Array.isArray(details)) return undefined;
  const diff = (details as Record<string, unknown>).diff;
  return typeof diff === "string" && diff.trim() ? diff : undefined;
}

function renderOpenJetDiff(diff: string): string[] {
  return diff.split("\n").map((row) => {
    if (row.startsWith("+")) return chalk.hex(palette.greenSoft)(row);
    if (row.startsWith("-")) return chalk.hex(palette.errorSoft)(row);
    return chalk.hex(palette.dim)(row);
  });
}

export function toolCallSummary(name: string, args: unknown): string {
  if (!args || typeof args !== "object" || Array.isArray(args)) return "";
  const values = args as Record<string, unknown>;
  if (name === "bash") return compactCommand(String(values.command ?? ""));
  if (["read", "edit", "write"].includes(name)) return compactPath(String(values.path ?? values.file_path ?? ""));
  if (["grep", "find"].includes(name)) return String(values.pattern ?? "");
  if (name === "ls") return String(values.path ?? ".");
  for (const key of ["source", "kind", "scope", "name"]) {
    if (values[key] != null && String(values[key])) return String(values[key]);
  }
  return "";
}

export function toolLabel(name: string): string {
  return ({ bash: "Run", read: "Read", edit: "Edit", write: "Write", grep: "Search", find: "Find", ls: "List" } as Record<string, string>)[name]
    ?? name.replaceAll("_", " ").replace(/^./, (letter) => letter.toUpperCase());
}

export function compactNumber(value: number): string {
  if (Math.abs(value) < 1000) return String(Math.round(value));
  const scaled = value / 1000;
  return `${scaled >= 100 ? scaled.toFixed(0) : scaled.toFixed(1).replace(/\.0$/, "")}k`;
}

export function formatDuration(milliseconds: number): string {
  if (milliseconds < 1000) return `${Math.max(0, Math.round(milliseconds))}ms`;
  const seconds = milliseconds / 1000;
  if (seconds < 10) return `${seconds.toFixed(1).replace(/\.0$/, "")}s`;
  if (seconds < 60) return `${Math.round(seconds)}s`;
  const minutes = Math.floor(seconds / 60);
  return `${minutes}m ${Math.round(seconds % 60)}s`;
}

export function compactWorkspace(path: string): string {
  return basename(path.replace(/[\\/]+$/, "")) || path;
}

export function compactModel(value: string): string {
  const name = basename(value).replace(/\.gguf$/i, "");
  return name.replace(/-Q\d+(?:_[A-Z0-9]+)+$/i, "");
}

function compactRuntime(value: string): string {
  return value.replace("llama_cpp", "llama.cpp").replace("openai_codex", "Codex");
}

function compactPath(value: string): string {
  if (!value) return "";
  const cwdRelative = relative(process.cwd(), value);
  if (cwdRelative && !cwdRelative.startsWith("..") && !cwdRelative.startsWith("/")) return cwdRelative;
  const home = homedir();
  return value.startsWith(`${home}/`) ? `~/${value.slice(home.length + 1)}` : value;
}

function compactCommand(value: string): string {
  return value.replaceAll(process.cwd(), ".").replace(/\s+/g, " ").trim();
}
