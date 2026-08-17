import { describe, expect, it } from "vitest";
import { SelectList, visibleWidth } from "@earendil-works/pi-tui";
import {
  BrandHeader,
  PickerPanel,
  ReasoningBlock,
  SectionDivider,
  StatusFooter,
  ToolCard,
  UserMessage,
  compactModel,
  compactNumber,
  formatDuration,
  toolCallSummary,
  modelTokenUsage,
  totalModelTokens,
} from "./components.js";
import { selectTheme } from "./theme.js";

describe("ToolCard", () => {
  it("renders a compact call summary without successful result noise", () => {
    const card = new ToolCard("call-1", "bash", { command: "npx tsc --noEmit" });
    card.complete("tsc exit: 0", true, true);
    const rendered = card.render(100).join("\n");

    expect(rendered).toContain("✓ Run");
    expect(rendered).toContain("npx tsc --noEmit");
    expect(rendered).not.toContain("tsc exit: 0");
    expect(rendered).not.toMatch(/[╭╮╰╯]/);
  });

  it("shows result details only after expansion", () => {
    const card = new ToolCard("call-2", "read", { path: "src/App.tsx" });
    card.complete("file contents", true, true);
    card.toggle();

    expect(card.render(100).join("\n")).toContain("file contents");
  });

  it("shows Pi edit diffs by default while keeping raw output collapsed", () => {
    const card = new ToolCard("call-edit", "edit", { path: "src/App.tsx" });
    card.complete("Successfully replaced 1 block", true, true, {
      diff: "- 10 const oldName = true;\n+ 10 const newName = true;",
      patch: "unused unified patch",
      firstChangedLine: 10,
    });
    const rendered = card.render(100).join("\n");

    expect(rendered).toContain("const oldName = true;");
    expect(rendered).toContain("const newName = true;");
    expect(rendered).not.toContain("Successfully replaced 1 block");
  });
});

describe("OpenJet visual hierarchy", () => {
  it("keeps the wordmark on wide terminals and falls back cleanly when narrow", () => {
    const compact = new BrandHeader().render(40);
    const wide = new BrandHeader().render(120);
    expect(compact).toHaveLength(1);
    expect(compact[0]).toContain("OPENJET");
    expect(wide).toHaveLength(6);
    expect(wide.join("\n")).toMatch(/[█╗╚]/);
  });

  it("renders user messages as a quiet prompt rather than a section heading", () => {
    const rendered = new UserMessage("make it cleaner").render(80).join("\n");
    expect(rendered).toContain("› make it cleaner");
    expect(rendered).not.toContain("USER");
    expect(visibleWidth(rendered)).toBe(80);
  });

  it("keeps reasoning invisible until explicitly expanded", () => {
    const reasoning = new ReasoningBlock(false);
    reasoning.setText("I should inspect the relevant source.");
    expect(reasoning.render(80)).toEqual([]);
    reasoning.setExpanded(true);
    expect(reasoning.render(80).join("\n")).toContain("reasoning");
  });

  it("prioritizes concise footer information responsively", () => {
    const footer = new StatusFooter({
      workspace: "/home/louis/rvchess",
      model: "/home/louis/open-jet/models/Qwen3.8-27B-Q4_K_M.gguf",
      runtime: "llama_cpp",
      contextTokens: 11079,
      contextWindow: 210767,
      completionTokens: 4037,
      savedTokenUsage: { input: 6000, cache: 4000, output: 2340 },
      tps: 32.4,
      powerWatts: 184,
      cpuPercent: 37,
      memoryPercent: 61,
      temperatureC: 68,
      device: "CUDA",
      status: "ready",
    });
    const wideRows = footer.render(120);
    const narrowRows = footer.render(40);
    const wide = wideRows.join("\n");
    const narrow = narrowRows.join("\n");
    expect(wideRows).toHaveLength(3);
    expect(wide).toContain("rvchess");
    expect(wide).toContain("Qwen3.8-27B");
    expect(wide).toContain("32.4 tok/s");
    expect(wide).toContain("CPU 37%");
    expect(wide).toContain("RAM 61%");
    expect(wide).toContain("184 W");
    expect(wide).toContain("12.3k tokens saved by local model");
    expect(wide).toContain("6k input, 4k cached, 2.3k output");
    expect(wide).not.toContain("/home/louis");
    expect(narrowRows.every((row) => visibleWidth(row) === 40)).toBe(true);
    expect(narrow).toContain("LOCAL");
    expect(narrow).toContain("32.4 tok/s");
    expect(narrow).toContain("ready");
  });

  it("starts the footer with the selected agent mode and model pair", () => {
    const footer = new StatusFooter({
      agentMode: "hybrid",
      codexModel: "gpt-5.6-sol",
      codexEffort: "medium",
      localModel: "Qwen3.8-27B-Q4_K_M.gguf",
      runtime: "llama_cpp",
      savedTokenUsage: { input: 12000, cache: 8000, output: 4891 },
      status: "ready",
    });
    const first = footer.render(140)[0];
    expect(first.trimStart()).toMatch(/^HYBRID/);
    expect(first).toContain("gpt-5.6-sol medium + Qwen3.8-27B");
  });

  it("formats model names and token counts without noisy precision", () => {
    expect(compactModel("/models/Qwen3.8-27B-Q4_K_M.gguf")).toBe("Qwen3.8-27B");
    expect(compactNumber(210767)).toBe("211k");
    expect(compactNumber(11079)).toBe("11.1k");
  });

  it("counts all local input, cache, and output tokens as saved", () => {
    expect(modelTokenUsage({ input: 1000, cacheRead: 400, cacheWrite: 100, output: 250 })).toEqual({
      input: 1000,
      cache: 500,
      output: 250,
    });
    expect(totalModelTokens({ input: 1000, cacheRead: 400, cacheWrite: 100, output: 250 })).toBe(1750);
  });

  it("separates transcript stages and records their elapsed time", () => {
    const divider = new SectionDivider("tools");
    expect(divider.render(80)[0]).toContain("── tools ");
    divider.complete(2340);
    const completed = divider.render(80)[0];
    expect(completed).toContain("tools · 2.3s");
    expect(visibleWidth(completed)).toBe(80);
  });

  it("renders model pickers as opaque titled panels at full overlay width", () => {
    const picker = new SelectList([
      { value: "medium", label: "medium", description: "Codex reasoning effort" },
    ], 5, selectTheme);
    const rows = new PickerPanel("Configure model · Codex effort", picker).render(72);
    expect(rows[0]).toContain("Configure model · Codex effort");
    expect(rows.every((row) => visibleWidth(row) === 72)).toBe(true);
  });

  it("records tool duration without changing the detail toggle", () => {
    let now = 1000;
    const card = new ToolCard("call-timed", "bash", { command: "npm test" }, () => now);
    now = 2650;
    card.complete("passed", true, true);
    expect(card.render(80).join("\n")).toContain("1.6s");
    card.toggle();
    expect(card.render(80).join("\n")).toContain("passed");
  });

  it("formats sub-second and long stage timings", () => {
    expect(formatDuration(240)).toBe("240ms");
    expect(formatDuration(12_400)).toBe("12s");
    expect(formatDuration(65_000)).toBe("1m 5s");
  });
});

describe("toolCallSummary", () => {
  it("selects useful arguments", () => {
    expect(toolCallSummary("read", { path: "src/App.tsx" })).toBe("src/App.tsx");
    expect(toolCallSummary("bash", { command: "npm test" })).toBe("npm test");
  });
});
