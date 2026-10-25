import { readFileSync } from "node:fs";
import { expect, test } from "vite-plus/test";
import { ConfigSchema, DEFAULT_CONFIG, resolveConfig } from "../src/index.ts";

test("defaults satisfy the schema", () => {
  expect(ConfigSchema.parse(DEFAULT_CONFIG)).toEqual(DEFAULT_CONFIG);
});

test("defaults.json is in sync (run `vp run @doppelbot/config#export`)", () => {
  const exported = JSON.parse(readFileSync(new URL("../defaults.json", import.meta.url), "utf8"));
  expect(exported).toEqual(DEFAULT_CONFIG);
});

test("overrides merge deeply", () => {
  const config = resolveConfig({ evidence: { weights: { bot: 0.5 } } });
  expect(config.evidence.weights).toEqual({ rated: 1, casual: 0.7, bot: 0.5 });
  expect(config.recency).toEqual(DEFAULT_CONFIG.recency);
});

test("out-of-range overrides are rejected", () => {
  expect(() => resolveConfig({ recency: { halfLifeDays: 0 } })).toThrow();
  expect(() => resolveConfig({ fit: { holdoutFraction: 0.9 } })).toThrow();
});
