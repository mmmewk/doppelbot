// Normalized game shape stored in S3 (games/<site>/<username>/<yyyy-mm>.jsonl.gz, one per line).
// Mirrors python/src/doppelbot/games.py `Game`, with camelCase keys and the result added.

export type Site = "lichess" | "chesscom";

// Same speeds as the Python fetchers. Bullet and ultrabullet are noise for Maia; daily has no clocks.
export const SPEEDS = ["blitz", "rapid", "classical"] as const;
export type Speed = (typeof SPEEDS)[number];

export type GameResult = "1-0" | "0-1" | "1/2-1/2";

export interface Game {
  site: Site;
  id: string;
  /** Unix seconds, end of game. */
  playedAt: number;
  speed: Speed;
  rated: boolean;
  /** Initial clock, seconds. */
  base: number;
  /** Increment, seconds. */
  inc: number;
  /** Lowercase usernames. */
  white: string;
  black: string;
  whiteElo: number;
  blackElo: number;
  result: GameResult;
  /** UCI moves from the standard starting position. */
  moves: string[];
  /** Seconds left on the mover's clock after each ply; empty if unknown. */
  clocks: number[];
}

export function gamesToJsonl(games: Game[]): string {
  return games.map((game) => JSON.stringify(game) + "\n").join("");
}

export function gamesFromJsonl(text: string): Game[] {
  return text
    .split("\n")
    .filter((line) => line.length > 0)
    .map((line) => JSON.parse(line) as Game);
}
