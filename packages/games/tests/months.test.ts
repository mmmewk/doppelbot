import { expect, test } from "vite-plus/test";
import { missingMonths, monthEnd, monthOf, monthStart, recentMonths } from "../src/index.ts";

const NOW = Date.UTC(2026, 9, 25, 12); // 2026-10-25

test("month helpers use UTC boundaries", () => {
  expect(monthOf(Date.UTC(2026, 0, 31, 23, 59, 59))).toBe("2026-01");
  expect(monthStart("2026-02")).toBe(Date.UTC(2026, 1, 1));
  expect(monthEnd("2026-12")).toBe(Date.UTC(2027, 0, 1));
  expect(monthEnd("2026-01")).toBe(monthStart("2026-02"));
});

test("recentMonths counts back across years, newest first", () => {
  expect(recentMonths(Date.UTC(2026, 1, 10), 4)).toEqual([
    "2026-02",
    "2026-01",
    "2025-12",
    "2025-11",
  ]);
});

test("a first refresh fetches every month in the window", () => {
  const months = missingMonths({ now: NOW, maxMonths: 3, ingested: new Map() });
  expect(months).toEqual(["2026-10", "2026-09", "2026-08"]);
});

test("closed months fetched after they ended are final; open ones are refetched", () => {
  const ingested = new Map([
    ["2026-10", { fetchedAt: Date.UTC(2026, 9, 20) }], // current month, still open
    ["2026-09", { fetchedAt: Date.UTC(2026, 9, 2) }], // fetched after September ended
    ["2026-08", { fetchedAt: Date.UTC(2026, 7, 30) }], // fetched before August ended
  ]);
  expect(missingMonths({ now: NOW, maxMonths: 4, ingested })).toEqual([
    "2026-10",
    "2026-08",
    "2026-07",
  ]);
});

test("a month fetched exactly at its end is final", () => {
  const ingested = new Map([["2026-09", { fetchedAt: monthEnd("2026-09") }]]);
  expect(missingMonths({ now: NOW, maxMonths: 2, ingested })).toEqual(["2026-10"]);
});

test("months the site can't have games for are skipped", () => {
  const archives = new Set(["2026-10", "2026-08"]);
  const months = missingMonths({
    now: NOW,
    maxMonths: 4,
    ingested: new Map(),
    available: (month) => archives.has(month),
  });
  expect(months).toEqual(["2026-10", "2026-08"]);
});

test("widening maxMonths backfills only the older months", () => {
  const ingested = new Map(
    ["2026-10", "2026-09"].map((m) => [m, { fetchedAt: Date.UTC(2027, 0, 1) }] as const),
  );
  expect(missingMonths({ now: NOW, maxMonths: 4, ingested })).toEqual(["2026-08", "2026-07"]);
});
