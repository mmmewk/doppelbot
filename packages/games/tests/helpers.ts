import { sansToUci, type Game } from "../src/index.ts";

/** A synthetic game for tests; `moves` is space-separated SAN. */
export function makeGame(moves: string, overrides: Partial<Game> = {}): Game {
  const uci = sansToUci(moves.split(" "));
  if (!uci) throw new Error(`illegal test game: ${moves}`);
  return {
    site: "lichess",
    id: "game0001",
    playedAt: 1_700_000_000,
    speed: "blitz",
    rated: true,
    base: 180,
    inc: 2,
    white: "alice",
    black: "bob",
    whiteElo: 1500,
    blackElo: 1480,
    result: "1-0",
    moves: uci,
    clocks: [],
    ...overrides,
  };
}
