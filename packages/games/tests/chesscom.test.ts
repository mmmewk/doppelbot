import { expect, test } from "vite-plus/test";
import {
  HttpClient,
  fetchChesscomArchives,
  fetchChesscomMonth,
  parseChesscomGame,
  parsePgn,
  parseTimeControl,
} from "../src/index.ts";

// Synthetic PGN in chess.com's export style; no real data.
function pgn(moves: string, headers: Record<string, string> = {}): string {
  const all = { Event: "Live Chess", White: "Alice", Black: "Bob", Result: "1-0", ...headers };
  return (
    Object.entries(all)
      .map(([k, v]) => `[${k} "${v}"]`)
      .join("\n") + `\n\n${moves}`
  );
}

const MOVES =
  "1. e4 {[%clk 0:03:00]} 1... e5 {[%clk 0:03:00]} 2. Nf3 {[%clk 0:02:59.9]} 2... Nc6 {[%clk 0:02:57.5]} 3. Bb5 {[%clk 1:02:03.25]} 1-0";

const game = {
  url: "https://www.chess.com/game/live/123456",
  pgn: pgn(MOVES),
  time_control: "180",
  end_time: 1_700_000_300,
  rated: true,
  time_class: "blitz",
  rules: "chess",
  white: { username: "Alice", rating: 1500 },
  black: { username: "Bob", rating: 1480 },
};

test("parsePgn reads headers, SAN and clocks", () => {
  const parsed = parsePgn(pgn(MOVES, { Event: 'A \\"quoted\\" event' }));
  expect(parsed.headers.Event).toBe('A "quoted" event');
  expect(parsed.headers.Result).toBe("1-0");
  expect(parsed.sans).toEqual(["e4", "e5", "Nf3", "Nc6", "Bb5"]);
  expect(parsed.clocks).toEqual([180, 180, 179.9, 177.5, 3723.25]);
});

test("parsePgn tolerates missing clocks, annotations and a trailing result", () => {
  const parsed = parsePgn(pgn("1. e4 e5 2. Nf3! $1 Nc6?! 3. O-O-O {comment} 0-1"));
  expect(parsed.sans).toEqual(["e4", "e5", "Nf3", "Nc6", "O-O-O"]);
  expect(parsed.clocks).toEqual([null, null, null, null, null]);
});

test("parseTimeControl", () => {
  expect(parseTimeControl("180+2")).toEqual([180, 2]);
  expect(parseTimeControl("600")).toEqual([600, 0]);
  expect(parseTimeControl("1/86400")).toBeNull();
});

test("normalizes a game", () => {
  expect(parseChesscomGame(game, true)).toEqual({
    site: "chesscom",
    id: "123456",
    playedAt: 1_700_000_300,
    speed: "blitz",
    rated: true,
    base: 180,
    inc: 0,
    white: "alice",
    black: "bob",
    whiteElo: 1500,
    blackElo: 1480,
    result: "1-0",
    moves: ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5"],
    clocks: [180, 180, 179.9, 177.5, 3723.25],
  });
});

test("drops clocks unless every ply has one", () => {
  const partial = pgn("1. e4 {[%clk 0:03:00]} 1... e5 2. Nf3 {[%clk 0:02:59]} 1-0");
  expect(parseChesscomGame({ ...game, pgn: partial }, true)!.clocks).toEqual([]);
});

test("skips games the Python fetcher also skips", () => {
  expect(parseChesscomGame({ ...game, rules: "chess960" }, true)).toBeNull();
  expect(parseChesscomGame({ ...game, time_class: "bullet" }, true)).toBeNull();
  expect(
    parseChesscomGame({ ...game, time_class: "daily", time_control: "1/86400" }, true),
  ).toBeNull();
  expect(parseChesscomGame({ ...game, pgn: undefined }, true)).toBeNull();
  expect(
    parseChesscomGame({ ...game, pgn: pgn(MOVES, { FEN: "8/8/8/8/8/8/8/K6k w - - 0 1" }) }, true),
  ).toBeNull();
  expect(parseChesscomGame({ ...game, pgn: pgn("1. e4 e5 2. Qh9 1-0") }, true)).toBeNull();
  expect(parseChesscomGame({ ...game, pgn: pgn("1-0") }, true)).toBeNull();
});

test("casual games are kept only when includeCasual is set", () => {
  expect(parseChesscomGame({ ...game, rated: false }, true)!.rated).toBe(false);
  expect(parseChesscomGame({ ...game, rated: false }, false)).toBeNull();
});

function clientFor(handler: (url: URL) => Response) {
  return new HttpClient({
    minIntervalMs: 0,
    backoffMs: 0,
    sleep: async () => {},
    fetch: (async (url: string) => handler(new URL(url))) as typeof fetch,
  });
}

test("lists archive months, or null for an unknown user", async () => {
  const client = clientFor((url) =>
    url.pathname.includes("alice")
      ? Response.json({
          archives: [
            "https://api.chess.com/pub/player/alice/games/2026/08",
            "https://api.chess.com/pub/player/alice/games/2026/10",
          ],
        })
      : new Response("{}", { status: 404 }),
  );
  expect(await fetchChesscomArchives(client, "alice")).toEqual(new Set(["2026-08", "2026-10"]));
  expect(await fetchChesscomArchives(client, "nobody")).toBeNull();
});

test("fetches a month, filters, and sorts oldest first", async () => {
  const later = { ...game, url: "https://www.chess.com/game/live/2", end_time: 1_700_000_900 };
  const earlier = { ...game, url: "https://www.chess.com/game/live/1", end_time: 1_700_000_100 };
  const bullet = { ...game, url: "https://www.chess.com/game/live/3", time_class: "bullet" };
  let requested = "";
  const client = clientFor((url) => {
    requested = url.pathname;
    return Response.json({ games: [later, bullet, earlier] });
  });
  const games = await fetchChesscomMonth(client, "alice", "2026-08", { includeCasual: true });
  expect(requested).toBe("/pub/player/alice/games/2026/08");
  expect(games.map((g) => g.id)).toEqual(["1", "2"]);
});
