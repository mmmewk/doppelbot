import { expect, test } from "vite-plus/test";
import {
  HttpClient,
  fetchLichessCreatedAt,
  fetchLichessMonth,
  parseLichessGame,
  type LichessGame,
} from "../src/index.ts";

// Synthetic games shaped like the Lichess NDJSON export; no real data.
const CLOCKS = [18000, 18000, 17850, 17900, 17700, 17800];
const base: LichessGame = {
  id: "abcd1234",
  rated: true,
  variant: "standard",
  speed: "blitz",
  status: "resign",
  lastMoveAt: 1_700_000_300_500,
  players: {
    white: { user: { name: "Alice", id: "alice" }, rating: 1500 },
    black: { user: { name: "Bob", id: "bob" }, rating: 1480 },
  },
  winner: "white",
  moves: "e4 e5 Nf3 Nc6 Bb5 a6",
  clocks: CLOCKS,
  clock: { initial: 180, increment: 0, totalTime: 180 },
};

test("normalizes a game", () => {
  expect(parseLichessGame(base, true)).toEqual({
    site: "lichess",
    id: "abcd1234",
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
    moves: ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6"],
    clocks: [180, 180, 178.5, 179, 177, 178],
  });
});

test("keeps clocks aligned with plies, dropping a trailing entry and short lists", () => {
  expect(parseLichessGame({ ...base, clocks: [...CLOCKS, 100] }, true)!.clocks).toHaveLength(6);
  expect(parseLichessGame({ ...base, clocks: CLOCKS.slice(0, 5) }, true)!.clocks).toEqual([]);
  expect(parseLichessGame({ ...base, clocks: undefined }, true)!.clocks).toEqual([]);
});

test("derives the result from the winner", () => {
  expect(parseLichessGame({ ...base, winner: "black" }, true)!.result).toBe("0-1");
  expect(parseLichessGame({ ...base, winner: undefined, status: "draw" }, true)!.result).toBe(
    "1/2-1/2",
  );
});

test("skips games the Python fetcher also skips", () => {
  expect(parseLichessGame({ ...base, variant: "chess960" }, true)).toBeNull();
  expect(parseLichessGame({ ...base, clock: undefined }, true)).toBeNull();
  expect(parseLichessGame({ ...base, moves: "" }, true)).toBeNull();
  expect(parseLichessGame({ ...base, speed: "bullet" }, true)).toBeNull();
  expect(parseLichessGame({ ...base, status: "aborted" }, true)).toBeNull();
  expect(
    parseLichessGame({ ...base, players: { ...base.players, black: { rating: 1400 } } }, true),
  ).toBeNull(); // AI or anonymous opponent
  expect(parseLichessGame({ ...base, moves: "e4 e5 Qh9" }, true)).toBeNull();
});

test("casual games are kept only when includeCasual is set", () => {
  expect(parseLichessGame({ ...base, rated: false }, true)!.rated).toBe(false);
  expect(parseLichessGame({ ...base, rated: false }, false)).toBeNull();
});

function clientFor(handler: (url: URL) => Response) {
  const urls: URL[] = [];
  const client = new HttpClient({
    minIntervalMs: 0,
    backoffMs: 0,
    sleep: async () => {},
    fetch: (async (url: string) => {
      urls.push(new URL(url));
      return handler(new URL(url));
    }) as typeof fetch,
  });
  return { client, urls };
}

test("fetches one UTC month as NDJSON and drops unusable lines", async () => {
  const lines = [
    JSON.stringify(base),
    JSON.stringify({ ...base, id: "ai", players: { ...base.players, black: { aiLevel: 3 } } }),
    JSON.stringify({ ...base, id: "second", lastMoveAt: 1_700_000_900_000 }),
  ];
  const { client, urls } = clientFor(() => new Response(lines.join("\n") + "\n"));
  const games = await fetchLichessMonth(client, "alice", "2023-11", { includeCasual: true });
  expect(games.map((g) => g.id)).toEqual(["abcd1234", "second"]);
  const params = urls[0]!.searchParams;
  expect(urls[0]!.pathname).toBe("/api/games/user/alice");
  expect(params.get("since")).toBe(String(Date.UTC(2023, 10, 1)));
  expect(params.get("until")).toBe(String(Date.UTC(2023, 11, 1) - 1));
  expect(params.get("clocks")).toBe("true");
  expect(params.get("perfType")).toBe("blitz,rapid,classical");
  expect(params.has("rated")).toBe(false);
});

test("asks only for rated games when casual is excluded", async () => {
  const { client, urls } = clientFor(() => new Response(""));
  await fetchLichessMonth(client, "alice", "2023-11", { includeCasual: false });
  expect(urls[0]!.searchParams.get("rated")).toBe("true");
});

test("reads the account creation time, or null for an unknown user", async () => {
  const { client } = clientFor((url) =>
    url.pathname.endsWith("/alice")
      ? Response.json({ createdAt: 1_600_000_000_000 })
      : new Response("{}", { status: 404 }),
  );
  expect(await fetchLichessCreatedAt(client, "alice")).toBe(1_600_000_000_000);
  expect(await fetchLichessCreatedAt(client, "nobody")).toBeNull();
});
