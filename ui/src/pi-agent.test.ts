import { describe, expect, it } from "vitest";
import type { AgentSessionEvent } from "@earendil-works/pi-coding-agent";
import { describeModelError, modelConfigPayload, sessionEventToUiEvents, type OpenJetPiModel } from "./pi-agent.js";

describe("Pi session transcript ordering", () => {
  it("preserves assistant-message boundaries around tool rounds", () => {
    const assistantMessage = { role: "assistant", content: [] };
    const events = [
      { type: "message_start", message: assistantMessage },
      { type: "tool_execution_start", toolCallId: "read-1", toolName: "read", args: { path: "a.ts" } },
      { type: "tool_execution_end", toolCallId: "read-1", toolName: "read", result: { content: [] }, isError: false },
      { type: "message_start", message: assistantMessage },
    ] as AgentSessionEvent[];

    expect(events.flatMap(sessionEventToUiEvents).map((event) => event.type)).toEqual([
      "assistant_start",
      "tool_start",
      "tool_end",
      "assistant_start",
    ]);
  });

  it("preserves structured edit details for the TUI diff renderer", () => {
    const [event] = sessionEventToUiEvents({
      type: "tool_execution_end",
      toolCallId: "edit-1",
      toolName: "edit",
      result: {
        content: [{ type: "text", text: "Successfully replaced 1 block" }],
        details: { diff: "- 1 old\n+ 1 new", patch: "@@ patch" },
      },
      isError: false,
    } as AgentSessionEvent);

    expect(event).toMatchObject({
      type: "tool_end",
      callId: "edit-1",
      details: { diff: "- 1 old\n+ 1 new", patch: "@@ patch" },
    });
  });

  it("attaches model attribution to local worker transcript events", () => {
    const [event] = sessionEventToUiEvents({
      type: "tool_execution_start",
      toolCallId: "local-read-1",
      toolName: "read",
      args: { path: "src/app.ts" },
    } as AgentSessionEvent, {
      lane: "local",
      model: "Qwen3.8-27B-Q4_K_M.gguf",
      parentCallId: "delegate-1",
    });
    expect(event).toMatchObject({
      type: "tool_start",
      lane: "local",
      model: "Qwen3.8-27B-Q4_K_M.gguf",
      parentCallId: "delegate-1",
    });
  });

  it("forwards legacy-meter chunks without counting hidden reasoning", () => {
    const events = sessionEventToUiEvents({
      type: "message_update",
      message: { role: "assistant" },
      assistantMessageEvent: {
        type: "text_delta",
        delta: "hello",
        partial: { usage: { output: 12 } },
      },
    } as AgentSessionEvent);

    expect(events).toContainEqual({ type: "generation_chunk", text: "hello" });
    expect(events).toContainEqual({ type: "text_delta", text: "hello" });

    const reasoning = sessionEventToUiEvents({
      type: "message_update",
      message: { role: "assistant" },
      assistantMessageEvent: { type: "thinking_delta", delta: "hidden" },
    } as AgentSessionEvent);
    expect(reasoning).not.toContainEqual({ type: "generation_chunk", text: "hidden" });
  });

  it("surfaces compaction completion and failure instead of dropping them", () => {
    expect(sessionEventToUiEvents({
      type: "compaction_end",
      reason: "overflow",
      aborted: false,
      willRetry: true,
      result: { tokensBefore: 75915, estimatedTokensAfter: 18000 },
    } as AgentSessionEvent)).toEqual([{
      type: "compaction_end",
      ok: true,
      text: "Context compaction completed (75,915 → approximately 18,000 tokens).",
      willRetry: true,
    }]);

    expect(sessionEventToUiEvents({
      type: "compaction_end",
      reason: "overflow",
      aborted: false,
      willRetry: false,
      errorMessage: "summary request failed",
    } as AgentSessionEvent)).toEqual([{
      type: "compaction_end",
      ok: false,
      text: "summary request failed",
      willRetry: false,
    }]);
  });
});

describe("Pi model sampling", () => {
  it("writes OpenJet sampling defaults into Pi's model definition", () => {
    const model: OpenJetPiModel = {
      provider: "openjet-local",
      id: "Qwen3.8-27B-Q4_K_M.gguf",
      name: "Qwen3.8-27B-Q4_K_M.gguf",
      api: "openai-completions",
      apiKey: "openjet-local",
      baseUrl: "http://127.0.0.1:18080/v1",
      reasoning: true,
      input: ["text"],
      contextWindow: 262144,
      maxTokens: 131072,
      samplingParams: { temperature: 1, top_p: 0.95, top_k: 20, min_p: 0 },
      cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
    };

    const payload = modelConfigPayload(model) as {
      providers: Record<string, { models: Array<Record<string, unknown>> }>;
    };
    const stored = payload.providers["openjet-local"].models[0];

    expect(stored.maxTokens).toBe(131072);
    expect(stored.samplingParams).toEqual(model.samplingParams);
  });

  it("preserves Codex Responses auth and thinking configuration", () => {
    const model: OpenJetPiModel = {
      provider: "openai-codex",
      id: "gpt-5.6-sol",
      name: "gpt-5.6-sol",
      api: "openai-codex-responses",
      apiKey: "oauth-token",
      baseUrl: "https://chatgpt.com/backend-api",
      headers: { "ChatGPT-Account-Id": "acct" },
      reasoning: true,
      thinkingLevel: "medium",
      thinkingLevelMap: { minimal: null, xhigh: "xhigh", max: "max" },
      input: ["text", "image"],
      contextWindow: 272000,
      maxTokens: 128000,
      cost: { input: 5, output: 30, cacheRead: 0.5, cacheWrite: 6.25 },
    };

    const payload = modelConfigPayload(model) as {
      providers: Record<string, { api: string; apiKey: string; headers: Record<string, string>; models: Array<Record<string, unknown>> }>;
    };
    const provider = payload.providers["openai-codex"];
    expect(provider.api).toBe("openai-codex-responses");
    expect(provider.apiKey).toBe("oauth-token");
    expect(provider.headers["ChatGPT-Account-Id"]).toBe("acct");
    expect(provider.models[0].thinkingLevelMap).toEqual({ minimal: null, xhigh: "xhigh", max: "max" });
  });
});

describe("model error messages", () => {
  it("tells the user to switch to the local model when Codex tokens run out", () => {
    const messages = [
      "You have hit your usage limit.",
      "429 Too Many Requests",
      "insufficient_quota: plan exhausted",
    ];
    for (const errorMessage of messages) {
      const text = describeModelError({ errorMessage });
      expect(text).toContain("Run out of tokens");
      expect(text).toContain("/mode local");
      expect(text).toContain(errorMessage);
    }
  });

  it("passes other provider errors through unchanged", () => {
    expect(describeModelError({ errorMessage: "connection reset" })).toBe("connection reset");
  });

  it("never renders an empty notice", () => {
    expect(describeModelError({ errorMessage: "   " })).toContain("no detail");
    expect(describeModelError(undefined)).toContain("no detail");
  });
});
