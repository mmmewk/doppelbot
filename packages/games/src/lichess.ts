import { HttpError, type HttpClient } from "./http.ts";
import { monthEnd, monthStart, type MonthId } from "./months.ts";
import { sansToUci } from "./replay.ts";
import { SPEEDS, type Game, type GameResult, type Speed } from "./types.ts";

// Lichess export: https://lichess.org/api#tag/Games/operation/apiGamesUser (NDJSON).
// Same filtering as fetch_lichess in python/src/doppelbot/games.py.

interface LichessPlayer {
  user?: { id: string; name?: string };
  rating?: number;
}

export interface LichessGame {
  id: string;
  rated: boolean;
  variant: string;
  speed: string;
  status?: string;
  lastMoveAt: number;
  players: { white: LichessPlayer; black: LichessPlayer };
  winner?: "white" | "black";
  moves?: string;
  /** Centiseconds left after each ply. */
  clocks?: number[];
  clock?: { initial: number; increment: number; totalTime?: number };
}

const UNFINISHED = new Set(["created", "started", "aborted", "noStart"]);

/** Normalize one NDJSON game, or null if it isn't a usable standard game with a clock. */
export function parseLichessGame(raw: LichessGame, includeCasual: boolean): Game | null {
  if (raw.variant !== "standard" || !raw.clock || !raw.moves) return null;
  if (!SPEEDS.includes(raw.speed as Speed)) return null;
  if (!raw.rated && !includeCasual) return null;
  if (raw.status && UNFINISHED.has(raw.status)) return null;
  const { white, black } = raw.players;
  if (!white.user || !black.user || white.rating == null || black.rating == null) return null;
  const moves = sansToUci(raw.moves.split(" "));
  if (!moves) return null;
  // One clock per ply, sometimes plus a trailing entry; drop clocks that don't cover every ply.
  const clocks = (raw.clocks ?? []).map((c) => c / 100);
  const result: GameResult =
    raw.winner === "white" ? "1-0" : raw.winner === "black" ? "0-1" : "1/2-1/2";
  return {
    site: "lichess",
    id: raw.id,
    playedAt: Math.floor(raw.lastMoveAt / 1000),
    speed: raw.speed as Speed,
    rated: raw.rated,
    base: raw.clock.initial,
    inc: raw.clock.increment,
    white: white.user.id,
    black: black.user.id,
    whiteElo: white.rating,
    blackElo: black.rating,
    result,
    moves,
    clocks: clocks.length >= moves.length ? clocks.slice(0, moves.length) : [],
  };
}

async function* ndjsonLines(body: ReadableStream<Uint8Array>): AsyncGenerator<string> {
  const decoder = new TextDecoder();
  let buffer = "";
  for await (const chunk of body) {
    buffer += decoder.decode(chunk, { stream: true });
    let newline: number;
    while ((newline = buffer.indexOf("\n")) >= 0) {
      const line = buffer.slice(0, newline).trim();
      buffer = buffer.slice(newline + 1);
      if (line) yield line;
    }
  }
  buffer += decoder.decode();
  if (buffer.trim()) yield buffer.trim();
}

/** The account's creation time in ms, or null if the user doesn't exist. */
export async function fetchLichessCreatedAt(
  http: HttpClient,
  username: string,
  signal?: AbortSignal,
): Promise<number | null> {
  const url = `https://lichess.org/api/user/${encodeURIComponent(username)}`;
  const response = await http.get(url, { signal });
  if (response.status === 404) return null;
  if (!response.ok) throw new HttpError(response.status, url);
  const profile = (await response.json()) as { createdAt: number; closed?: boolean };
  return profile.createdAt;
}

/** Games created in the given UTC month, oldest first. */
export async function fetchLichessMonth(
  http: HttpClient,
  username: string,
  month: MonthId,
  options: { includeCasual: boolean; signal?: AbortSignal },
): Promise<Game[]> {
  const params = new URLSearchParams({
    since: String(monthStart(month)),
    until: String(monthEnd(month) - 1),
    perfType: SPEEDS.join(","),
    clocks: "true",
    moves: "true",
    opening: "false",
    sort: "dateAsc",
  });
  if (!options.includeCasual) params.set("rated", "true");
  const url = `https://lichess.org/api/games/user/${encodeURIComponent(username)}?${params.toString()}`;
  const response = await http.get(url, {
    headers: { Accept: "application/x-ndjson" },
    signal: options.signal,
  });
  if (!response.ok || !response.body) throw new HttpError(response.status, url);
  const games: Game[] = [];
  for await (const line of ndjsonLines(response.body)) {
    const game = parseLichessGame(JSON.parse(line) as LichessGame, options.includeCasual);
    if (game) games.push(game);
  }
  return games;
}
