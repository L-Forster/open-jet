import { describe, expect, it } from "vitest";
import { apiKeyFromCredential } from "./pi-openrouter.js";

describe("Pi OpenRouter credentials", () => {
  it("reads Pi api-key and OpenRouter OAuth access tokens", () => {
    expect(apiKeyFromCredential({ type: "api_key", key: " sk-or-key " })).toBe("sk-or-key");
    expect(apiKeyFromCredential({
      type: "oauth",
      access: "sk-or-oauth",
      refresh: "",
      expires: Number.MAX_SAFE_INTEGER,
    })).toBe("sk-or-oauth");
  });
});
