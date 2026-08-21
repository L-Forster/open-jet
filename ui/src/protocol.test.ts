import { describe, expect, it } from "vitest";
import { decodeMessage, PROTOCOL_VERSION } from "./protocol.js";

describe("decodeMessage", () => {
  it("accepts a v1 event", () => {
    expect(decodeMessage(JSON.stringify({ protocolVersion: PROTOCOL_VERSION, type: "ready", payload: {} })).type).toBe("ready");
  });

  it("rejects a protocol mismatch", () => {
    expect(() => decodeMessage('{"protocolVersion":2,"type":"ready"}')).toThrow(/Protocol mismatch/);
  });

  it("rejects non-object JSON", () => {
    expect(() => decodeMessage("[]")).toThrow(/must be an object/);
  });

  it("preserves an apiKey on command requests", () => {
    const message = decodeMessage(
      JSON.stringify({
        protocolVersion: PROTOCOL_VERSION,
        type: "command",
        text: "/connect openrouter",
        apiKey: "sk-or-secret",
      }),
    );
    expect(message.apiKey).toBe("sk-or-secret");
    expect(message.text).toBe("/connect openrouter");
  });
});
