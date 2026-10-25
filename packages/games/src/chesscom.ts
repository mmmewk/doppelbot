import { HttpError, type HttpClient } from "./http.ts";
import type { MonthId } from "./months.ts";
import { sansToUci } from "./replay.ts";
import { SPEEDS, type Game, type GameResult, type Speed } from "./types.ts";

// chess.com published-data API: monthly archives with PGN including [%clk] comments.
// Same filtering as fetch_chesscom in python/src/doppelbot/games.py.

interface ChesscomPlayer {
  username: string;
  rating: number;
}

export interface ChesscomGame {
  url: string;
  pgn?: string;
  time_control: string;
  end_time: number;
  rated: boolean;
  time_class: string;
  rules: string;
  white: ChesscomPlayer;
  black: ChesscomPlayer;
}

export interface ParsedPgn {
  headers: Record<string, string>;
  /** SAN moves of the main line. */
  sans: string[];
  /** Seconds from the [%clk] comment after each move, or null when it has none. */
  clocks: (number | null)[];
}

const HEADER = /^\[(\w+)\s+"((?:[^"\\]|\\.)*)"\]\s*$/gm;
const CLOCK = /\[%clk\s+(\d+):(\d{1,2}):(\d{1,2}(?:\.\d+)?)\]/;
// Comments, variations, line comments, NAGs, move numbers, then the remaining tokens.
const TOKEN = /\{[^}]*\}|\([^)]*\)|;[^\n]*|\$\d+|\d+\.(?:\.\.)?|[^\s{}();$]+/g;
const RESULTS = new Set(["1-0", "0-1", "1/2-1/2", "*"]);

/** Minimal PGN reader for chess.com exports: headers, main-line SAN and per-move clocks. */
export function parsePgn(pgn: string): ParsedPgn {
  const headers: Record<string, string> = {};
  for (const match of pgn.matchAll(HEADER)) headers[match[1]!] = match[2]!.replace(/\\(.)/g, "$1");
  const movetext = pgn.replace(HEADER, "");
  const sans: string[] = [];
  const clocks: (number | null)[] = [];
  for (const [token] of movetext.matchAll(TOKEN)) {
    if (token.startsWith("{")) {
      const clock = CLOCK.exec(token);
      if (clock && clocks.length > 0 && clocks[clocks.length - 1] === null) {
        clocks[clocks.length - 1] =
          Number(clock[1]) * 3600 + Number(clock[2]) * 60 + Number(clock[3]);
      }
    } else if (/^[(;$]/.test(token) || /^\d+\./.test(token) || RESULTS.has(token)) {
      continue;
    } else {
      sans.push(token.replace(/[!?]+$/, ""));
      clocks.push(null);
    }
  }
  return { headers, sans, clocks };
}

/** `"180+2"` -> [180, 2]; `"600"` -> [600, 0]; daily (`"1/86400"`) -> null. */
export function parseTimeControl(timeControl: string): [number, number] | null {
  if (timeControl.includes("/")) return null;
  const [base, inc] = timeControl.split("+");
  return [Number(base), Number(inc ?? 0)];
}

/** Normalize one archive game, or null if it isn't a usable standard game. */
export function parseChesscomGame(raw: ChesscomGame, includeCasual: boolean): Game | null {
  if (raw.rules !== "chess" || !SPEEDS.includes(raw.time_class as Speed)) return null;
  if (!raw.rated && !includeCasual) return null;
  const tc = parseTimeControl(raw.time_control);
  if (!tc || !raw.pgn) return null;
  const pgn = parsePgn(raw.pgn);
  if ("FEN" in pgn.headers) return null; // started from a custom position
  const moves = sansToUci(pgn.sans);
  if (!moves || moves.length === 0) return null;
  const clocks = pgn.clocks.filter((c): c is number => c !== null);
  const result = pgn.headers.Result;
  if (result !== "1-0" && result !== "0-1" && result !== "1/2-1/2") return null;
  return {
    site: "chesscom",
    id: raw.url.slice(raw.url.lastIndexOf("/") + 1),
    playedAt: raw.end_time,
    speed: raw.time_class as Speed,
    rated: raw.rated,
    base: tc[0],
    inc: tc[1],
    white: raw.white.username.toLowerCase(),
    black: raw.black.username.toLowerCase(),
    whiteElo: raw.white.rating,
    blackElo: raw.black.rating,
    result: result as GameResult,
    moves,
    clocks: clocks.length === moves.length ? clocks : [],
  };
}

/** Months with an archive, or null if the user doesn't exist. */
export async function fetchChesscomArchives(
  http: HttpClient,
  username: string,
  signal?: AbortSignal,
): Promise<Set<MonthId> | null> {
  const url = `https://api.chess.com/pub/player/${encodeURIComponent(username)}/games/archives`;
  const response = await http.get(url, { signal });
  if (response.status === 404) return null;
  if (!response.ok) throw new HttpError(response.status, url);
  const { archives } = (await response.json()) as { archives: string[] };
  return new Set(
    archives.map((archive) => {
      const [year, month] = archive.split("/").slice(-2);
      return `${year}-${month}`;
    }),
  );
}

/** Games that ended in the given UTC month, oldest first. */
export async function fetchChesscomMonth(
  http: HttpClient,
  username: string,
  month: MonthId,
  options: { includeCasual: boolean; signal?: AbortSignal },
): Promise<Game[]> {
  const [year, mon] = month.split("-");
  const url = `https://api.chess.com/pub/player/${encodeURIComponent(username)}/games/${year}/${mon}`;
  const response = await http.get(url, { signal: options.signal });
  if (!response.ok) throw new HttpError(response.status, url);
  const { games } = (await response.json()) as { games: ChesscomGame[] };
  return games
    .flatMap((raw) => parseChesscomGame(raw, options.includeCasual) ?? [])
    .sort((a, b) => a.playedAt - b.playedAt);
}
