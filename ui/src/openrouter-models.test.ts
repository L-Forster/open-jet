import { describe, expect, it } from "vitest";
import { CURATED_OPENROUTER_MODELS, formatOpenRouterContext, formatOpenRouterPrice, listOpenRouterPickerModels, OPENROUTER_SET_KEY_VALUE, openRouterPickerItems } from "./openrouter-models.js";

describe("OpenRouter picker catalog", () => {
  it("lists Ox Alpha first and includes Pi catalog pricing", () => {
    const models = listOpenRouterPickerModels();
    expect(models[0]?.id).toBe("stealth/ox-alpha");
    expect(formatOpenRouterPrice(models[0]!.cost)).toBe("free");
    expect(openRouterPickerItems({ connected: true })[0]?.description).toContain("free");
    expect(openRouterPickerItems({ connected: false })[0]?.value).toBe(OPENROUTER_SET_KEY_VALUE);

    const priced = models.find((model) => model.cost.input > 0 && !model.featured);
    expect(priced).toBeDefined();
    expect(formatOpenRouterPrice(priced!.cost)).toMatch(/\$/);
    expect(openRouterPickerItems().length).toBeLessThan(20);
  });

  it("formats context windows the way Pi list-models does", () => {
    expect(formatOpenRouterContext(1_048_576)).toBe("1M ctx");
    expect(formatOpenRouterContext(200_000)).toBe("200K ctx");
  });

  it("loads the generated catalog", () => {
    expect(CURATED_OPENROUTER_MODELS.map(({ id }) => id)).toContain("stealth/ox-alpha");
  });
});
