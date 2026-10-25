import { z } from "zod";

// Every tunable parameter in doppelbot. Shared by the web app, Lambdas, and Python
// (via defaults.json). See docs/PLAN.md "Configuration" for when each is applied.

const weight = z.number().min(0).max(1);

export const ConfigSchema = z.object({
  recency: z.object({
    // Half-life for recency-weighting move counts (read time).
    halfLifeDays: z.number().min(1).max(3650),
  }),
  blend: z.object({
    // How many games' worth of trust the model prior gets against the user's counts.
    alpha: z.number().min(0).max(1000),
  }),
  evidence: z.object({
    weights: z.object({
      rated: weight,
      casual: weight,
      bot: weight,
    }),
  }),
  filter: z.object({
    // Moves played below either clock threshold are excluded as time-pressure noise.
    minClockSeconds: z.number().min(0).max(600),
    minClockFraction: z.number().min(0).max(1),
  }),
  model: z.object({
    id: z.enum(["maia3-5m", "maia3-23m"]),
    precision: z.enum(["fp32", "fp16", "int8"]),
  }),
  fit: z.object({
    windowGames: z.number().int().min(20).max(5000),
    // Penalty for drifting off the calibrated rating point. Placeholder until Phase 2b tunes it.
    driftPenalty: z.number().min(0).max(100),
    holdoutFraction: z.number().min(0.05).max(0.5),
  }),
  bots: z.object({
    speed: z.object({ temperature: z.number().min(0).max(5) }),
    variety: z.object({ evalMarginCp: z.number().int().min(0).max(300) }),
    traps: z.object({ maxPerGame: z.number().int().min(0).max(10) }),
  }),
  drills: z.object({
    acceptMarginCp: z.number().int().min(0).max(300),
    retireProbability: z.number().min(0).max(1),
  }),
});

export type Config = z.infer<typeof ConfigSchema>;

export const DEFAULT_CONFIG: Config = {
  recency: { halfLifeDays: 120 },
  blend: { alpha: 5 },
  evidence: { weights: { rated: 1, casual: 0.8, bot: 0.3 } },
  filter: { minClockSeconds: 10, minClockFraction: 0.1 },
  model: { id: "maia3-5m", precision: "fp32" },
  fit: { windowGames: 200, driftPenalty: 1, holdoutFraction: 0.2 },
  bots: {
    speed: { temperature: 1 },
    variety: { evalMarginCp: 25 },
    traps: { maxPerGame: 1 },
  },
  drills: { acceptMarginCp: 30, retireProbability: 0.1 },
};

export type DeepPartial<T> = { [K in keyof T]?: T[K] extends object ? DeepPartial<T[K]> : T[K] };

function isPlainObject(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function deepMerge(base: unknown, override: unknown): unknown {
  if (!isPlainObject(base) || !isPlainObject(override)) {
    return override === undefined ? base : override;
  }
  const merged: Record<string, unknown> = { ...base };
  for (const [key, value] of Object.entries(override)) {
    merged[key] = deepMerge(base[key], value);
  }
  return merged;
}

/** Overlay user overrides on the defaults and validate the result. Throws on out-of-range values. */
export function resolveConfig(overrides: DeepPartial<Config> = {}): Config {
  return ConfigSchema.parse(deepMerge(DEFAULT_CONFIG, overrides));
}
