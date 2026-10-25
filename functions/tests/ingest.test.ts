import { gunzipSync } from "node:zlib";
import { DEFAULT_CONFIG } from "@doppelbot/config";
import { decodeOccurrences, gamesFromJsonl } from "@doppelbot/games";
import { expect, test } from "vite-plus/test";
import {
  MAX_ATTEMPTS,
  runIngest,
  summarize,
  type IngestDeps,
  type IngestRequest,
  type JobPatch,
  type MetaSummary,
  type MonthItem,
} from "../src/ingest.ts";

const NOW = Date.UTC(2026, 9, 25, 12);
const config = {
  ...DEFAULT_CONFIG.ingest,
  maxMonths: 3,
  lichessMinIntervalMs: 0,
  chesscomMinIntervalMs: 0,
};

// In-memory stand-ins for DynamoDB, S3, SQS and the two sites. Games are synthetic.
function setup(options: { deadline?: number; existing?: MonthItem[] } = {}) {
  const months = new Map<string, MonthItem>((options.existing ?? []).map((m) => [m.month, m]));
  const objects = new Map<string, { body: Uint8Array; contentEncoding?: string }>();
  const enqueued: IngestRequest[] = [];
  const jobs: JobPatch[] = [];
  const metas: MetaSummary[] = [];
  const requested: string[] = [];
  const deps: IngestDeps & { respond: (url: URL) => Response } = {
    store: {
      listMonths: async () => [...months.values()],
      putMonth: async (_s, _u, item) => void months.set(item.month, item),
      putMeta: async (_s, _u, summary) => void metas.push(summary),
      updateJob: async (_s, _u, _id, patch) => void jobs.push(patch),
    },
    blobs: {
      put: async (key, body, opts) =>
        void objects.set(key, { body, contentEncoding: opts.contentEncoding }),
    },
    queue: { enqueue: async (request) => void enqueued.push(request) },
    deadline: options.deadline ?? NOW + 10 * 60_000,
    now: () => NOW,
    sleep: async () => {},
    fetch: (async (url: string) => {
      requested.push(url);
      return deps.respond(new URL(url));
    }) as typeof fetch,
    respond: () => new Response("{}", { status: 404 }),
  };
  return { deps, months, objects, enqueued, jobs, metas, requested };
}

function request(site: IngestRequest["site"], username = "alice"): IngestRequest {
  return { jobId: "job1", site, username, attempt: 0 };
}

const pgn = (moves: string) =>
  `[Event "Live Chess"]\n[White "Alice"]\n[Black "Bob"]\n[Result "1-0"]\n\n${moves}`;

function chesscomGame(id: number, endTime: number) {
  return {
    url: `https://www.chess.com/game/live/${id}`,
    pgn: pgn(
      "1. e4 {[%clk 0:03:00]} 1... e5 {[%clk 0:03:00]} 2. Nf3 {[%clk 0:02:50]} 2... Nc6 {[%clk 0:02:59]} 1-0",
    ),
    time_control: "180",
    end_time: endTime,
    rated: true,
    time_class: "blitz",
    rules: "chess",
    white: { username: "Alice", rating: 1500 + id },
    black: { username: "Bob", rating: 1480 },
  };
}

function chesscomSite(archives: string[]) {
  return (url: URL) => {
    if (url.pathname.endsWith("/archives")) {
      return Response.json({
        archives: archives.map(
          (m) => `https://api.chess.com/pub/player/alice/games/${m.replace("-", "/")}`,
        ),
      });
    }
    const [year, month] = url.pathname.split("/").slice(-2);
    const start = Date.UTC(Number(year), Number(month) - 1, 5) / 1000;
    return Response.json({ games: [chesscomGame(2, start + 100), chesscomGame(1, start)] });
  };
}

test("chess.com: stores games and occurrences per month, then the META summary", async () => {
  const t = setup();
  t.deps.respond = chesscomSite(["2026-08", "2026-10"]); // no September archive
  expect(await runIngest(request("chesscom"), t.deps, config)).toBe("done");

  expect([...t.months.keys()].sort()).toEqual(["2026-08", "2026-10"]);
  const october = t.months.get("2026-10")!;
  expect(october).toMatchObject({
    gamesKey: "games/chesscom/alice/2026-10.jsonl.gz",
    occurrencesKey: "occurrences/chesscom/alice/2026-10.bin",
    games: 2,
    userMoves: 4,
    fetchedAt: NOW,
    ratings: { blitz: { rating: 1502 } },
  });

  const gamesObject = t.objects.get(october.gamesKey!)!;
  expect(gamesObject.contentEncoding).toBe("gzip");
  const games = gamesFromJsonl(gunzipSync(gamesObject.body).toString());
  expect(games.map((g) => g.id)).toEqual(["1", "2"]); // oldest first
  expect(games[0]!.clocks).toEqual([180, 180, 170, 179]);

  const rows = decodeOccurrences(t.objects.get(october.occurrencesKey!)!.body);
  expect(rows.map((r) => [r.gameIndex, r.ply, r.move])).toEqual([
    [0, 0, "e2e4"],
    [0, 2, "g1f3"],
    [1, 0, "e2e4"],
    [1, 2, "g1f3"],
  ]);
  expect(rows[1]!.clock).toBe(180); // clock before White's second move: their clock after the first
  expect(t.metas.at(-1)).toMatchObject({
    games: 4,
    userMoves: 8,
    months: 2,
    firstMonth: "2026-08",
    lastMonth: "2026-10",
  });
  expect(t.jobs.at(-1)).toEqual({ status: "done" });
});

test("refresh fetches only the open month and months it hasn't seen", async () => {
  const done = (month: string, fetchedAt: number): MonthItem => ({
    month,
    gamesKey: null,
    occurrencesKey: null,
    games: 0,
    userMoves: 0,
    fetchedAt,
    ratings: {},
  });
  const t = setup({
    existing: [done("2026-09", Date.UTC(2026, 9, 3)), done("2026-10", Date.UTC(2026, 9, 10))],
  });
  t.deps.respond = chesscomSite(["2026-08", "2026-09", "2026-10"]);
  await runIngest(request("chesscom"), t.deps, config);
  const monthUrls = t.requested.filter((url) => !url.endsWith("/archives"));
  expect(monthUrls).toEqual([
    "https://api.chess.com/pub/player/alice/games/2026/10",
    "https://api.chess.com/pub/player/alice/games/2026/08",
  ]);
  expect(t.months.get("2026-09")!.fetchedAt).toBe(Date.UTC(2026, 9, 3)); // untouched
});

test("lichess: only months since account creation, casual games kept and flagged", async () => {
  const t = setup();
  const game = (id: string, rated: boolean) =>
    JSON.stringify({
      id,
      rated,
      variant: "standard",
      speed: "rapid",
      status: "mate",
      lastMoveAt: Date.UTC(2026, 9, 5) + (rated ? 0 : 1000),
      players: {
        white: { user: { id: "alice" }, rating: 1600 },
        black: { user: { id: "bob" }, rating: 1550 },
      },
      winner: "white",
      moves: "e4 e5 Nf3 Nc6",
      clocks: [60000, 60000, 59000, 59500],
      clock: { initial: 600, increment: 0 },
    });
  t.deps.respond = (url) => {
    if (url.pathname === "/api/user/alice")
      return Response.json({ createdAt: Date.UTC(2026, 8, 15) });
    return new Response(
      url.searchParams.get("since") === String(Date.UTC(2026, 9, 1))
        ? [game("g1", true), game("g2", false)].join("\n")
        : "",
    );
  };
  expect(await runIngest(request("lichess"), t.deps, config)).toBe("done");
  expect([...t.months.keys()].sort()).toEqual(["2026-09", "2026-10"]); // August predates the account
  expect(t.months.get("2026-09")).toMatchObject({ games: 0, gamesKey: null });
  expect(t.months.get("2026-10")).toMatchObject({
    games: 2,
    userMoves: 4,
    ratings: { rapid: { rating: 1600 } },
  });
  const games = gamesFromJsonl(
    gunzipSync(t.objects.get("games/lichess/alice/2026-10.jsonl.gz")!.body).toString(),
  );
  expect(games.map((g) => [g.id, g.rated])).toEqual([
    ["g1", true],
    ["g2", false],
  ]);
  expect(t.metas.at(-1)!.ratings.rapid!.rating).toBe(1600);
});

test("an unknown user fails the job without retrying", async () => {
  const t = setup();
  expect(await runIngest(request("lichess", "nobody"), t.deps, config)).toBe("error");
  expect(t.jobs.at(-1)).toEqual({ status: "error", error: "user not found" });
  expect(t.enqueued).toEqual([]);
});

test("running low on time saves progress and re-enqueues a continuation", async () => {
  const t = setup({ deadline: NOW + 60_000 }); // under the per-month budget
  t.deps.respond = chesscomSite(["2026-10"]);
  expect(await runIngest(request("chesscom"), t.deps, config)).toBe("continued");
  expect(t.enqueued).toEqual([{ ...request("chesscom"), attempt: 1 }]);
  expect(t.jobs.at(-1)).toEqual({ status: "queued" });
  expect(t.metas).toHaveLength(1);
});

test("a rate limit with no time to wait continues later", async () => {
  // 160 s left when the month starts, but the 429 arrives 130 s in: a 60 s backoff won't fit.
  const t = setup({ deadline: NOW + 160_000 });
  let elapsed = 0;
  t.deps.now = () => NOW + elapsed;
  t.deps.respond = (url) => {
    if (url.pathname.endsWith("/archives")) {
      return Response.json({ archives: ["https://api.chess.com/pub/player/alice/games/2026/10"] });
    }
    elapsed = 130_000;
    return new Response("", { status: 429 });
  };
  expect(await runIngest(request("chesscom"), t.deps, config)).toBe("continued");
  expect(t.enqueued).toEqual([{ ...request("chesscom"), attempt: 1 }]);
});

test("gives up after too many continuations", async () => {
  const t = setup({ deadline: NOW });
  t.deps.respond = chesscomSite(["2026-10"]);
  const last = { ...request("chesscom"), attempt: MAX_ATTEMPTS - 1 };
  expect(await runIngest(last, t.deps, config)).toBe("error");
  expect(t.enqueued).toEqual([]);
});

test("summarize totals the months and keeps the latest rating per speed", () => {
  const item = (month: string, games: number, ratings: MonthItem["ratings"]): MonthItem => ({
    month,
    gamesKey: null,
    occurrencesKey: null,
    games,
    userMoves: games * 30,
    fetchedAt: 0,
    ratings,
  });
  const summary = summarize(
    [
      item("2026-10", 5, { blitz: { rating: 1400, playedAt: 200 } }),
      item("2026-08", 10, {
        blitz: { rating: 1300, playedAt: 100 },
        rapid: { rating: 1500, playedAt: 90 },
      }),
    ],
    NOW,
  );
  expect(summary).toEqual({
    games: 15,
    userMoves: 450,
    months: 2,
    firstMonth: "2026-08",
    lastMonth: "2026-10",
    ratings: { blitz: { rating: 1400, playedAt: 200 }, rapid: { rating: 1500, playedAt: 90 } },
    ingestedAt: NOW,
  });
});
