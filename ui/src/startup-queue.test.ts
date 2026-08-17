import { describe, expect, it } from "vitest";
import { promptDisposition } from "./startup-queue.js";

describe("startup prompt queue", () => {
  it("queues while the Python runtime is starting before Pi initialization exists", () => {
    expect(promptDisposition(false, undefined)).toBe("queue");
  });

  it("queues while Pi is initializing", () => {
    expect(promptDisposition(false, false)).toBe("queue");
  });

  it("sends when Pi is ready and redirects only when setup is explicitly required", () => {
    expect(promptDisposition(true, false)).toBe("send");
    expect(promptDisposition(false, true)).toBe("setup");
  });
});
