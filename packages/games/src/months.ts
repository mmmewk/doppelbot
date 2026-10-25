// Calendar months are UTC "yyyy-mm" strings. Lichess games land in the month they were created,
// chess.com games in the month they ended (that is how each API partitions them).

export type MonthId = string;

export function monthOf(ms: number): MonthId {
  return new Date(ms).toISOString().slice(0, 7);
}

/** Start of the month, inclusive, in ms. */
export function monthStart(month: MonthId): number {
  return Date.parse(`${month}-01T00:00:00Z`);
}

/** Start of the next month, i.e. the exclusive end, in ms. */
export function monthEnd(month: MonthId): number {
  const [year, mon] = month.split("-").map(Number) as [number, number];
  return Date.UTC(year, mon, 1);
}

/** The `count` most recent months up to and including the one containing `now`, newest first. */
export function recentMonths(now: number, count: number): MonthId[] {
  const d = new Date(now);
  let year = d.getUTCFullYear();
  let mon = d.getUTCMonth();
  const out: MonthId[] = [];
  for (let i = 0; i < count; i++) {
    out.push(`${year}-${String(mon + 1).padStart(2, "0")}`);
    if (--mon < 0) {
      mon = 11;
      year--;
    }
  }
  return out;
}

export interface MissingMonthsInput {
  now: number;
  maxMonths: number;
  /** Months already stored, with when each was fetched (ms). */
  ingested: Map<MonthId, { fetchedAt: number }>;
  /** If given, only these months can have games (chess.com archive list, Lichess account age). */
  available?: (month: MonthId) => boolean;
}

/**
 * Months a refresh must fetch, newest first: every month in the window that was never fetched,
 * plus any month fetched before it ended (the current month, or last month if it was last
 * fetched while still open). A month fetched after it ended is final and never refetched.
 */
export function missingMonths(input: MissingMonthsInput): MonthId[] {
  return recentMonths(input.now, input.maxMonths).filter((month) => {
    if (input.available && !input.available(month)) return false;
    const done = input.ingested.get(month);
    return !done || done.fetchedAt < monthEnd(month);
  });
}
