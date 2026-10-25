import { Chess } from "chess.js";
import { positionHash, positionKey } from "./positionKey.ts";
import { SPEEDS, type Game, type Site, type Speed } from "./types.ts";

// One occurrence per move the user played: the position they faced, the move they chose, and
// the context needed to weight it at read time (age, source, clock). Nothing is filtered here:
// time-trouble moves are kept with their raw clock, and `isTimeTrouble` applies the
// `filter.*` settings when counts are built (docs/PLAN.md "Evidence and recency").

export interface Occurrence {
  /** `positionHash(positionKey(fen))` of the position before the user's move. */
  positionHash: bigint;
  /** UCI move the user played. */
  move: string;
  /** Unix seconds, end of the game (move times are not available). */
  playedAt: number;
  /** Index of the game within its month file (line number in the .jsonl). */
  gameIndex: number;
  /** Zero-based ply of the move within the game. */
  ply: number;
  /** Seconds on the user's clock before the move; null if the game has no clock data. */
  clock: number | null;
  /** Starting clock, seconds. */
  base: number;
  rated: boolean;
  site: Site;
  speed: Speed;
}

/** Replay a game and emit an occurrence for each move `username` played. */
export function buildOccurrences(game: Game, gameIndex: number, username: string): Occurrence[] {
  const user = username.toLowerCase();
  const color = game.white === user ? "w" : game.black === user ? "b" : null;
  if (color === null) return [];
  const chess = new Chess();
  const out: Occurrence[] = [];
  for (const [ply, uci] of game.moves.entries()) {
    if (chess.turn() === color) {
      out.push({
        positionHash: positionHash(positionKey(chess.fen())),
        move: uci,
        playedAt: game.playedAt,
        gameIndex,
        ply,
        clock: clockBefore(game, ply),
        base: game.base,
        rated: game.rated,
        site: game.site,
        speed: game.speed,
      });
    }
    chess.move({
      from: uci.slice(0, 2),
      to: uci.slice(2, 4),
      promotion: uci.length > 4 ? uci[4] : undefined,
    });
  }
  return out;
}

/**
 * The mover's clock before ply `ply`: their clock after their previous move (two plies back),
 * or the base time for each side's first move. Same as `_in_time_trouble` in positions.py.
 */
function clockBefore(game: Game, ply: number): number | null {
  if (game.clocks.length === 0) return null;
  return ply >= 2 ? (game.clocks[ply - 2] ?? null) : game.base;
}

export interface ClockFilter {
  minClockSeconds: number;
  minClockFraction: number;
}

/** True when the move was played under either clock threshold (matches positions.py). */
export function isTimeTrouble(occurrence: Occurrence, filter: ClockFilter): boolean {
  const { clock, base } = occurrence;
  if (clock === null) return false;
  return clock < filter.minClockSeconds || clock < filter.minClockFraction * base;
}

// Binary format: occurrences/<site>/<username>/<yyyy-mm>.bin, little-endian.
//
// Header, 16 bytes: "DPOC", u16 version (1), u16 rowSize (24), u32 count, u32 reserved.
// Row, 24 bytes:
//    0  u64  positionHash
//    8  u32  playedAt (unix seconds)
//   12  u32  gameIndex
//   16  u16  move: from | to << 6 | promotion << 12 (a1 = 0 .. h8 = 63; 0 none, 1 n, 2 b, 3 r, 4 q)
//   18  u16  clock before the move, deciseconds (0xFFFF = unknown, saturates at 0xFFFE)
//   20  u16  base time, seconds (saturates at 0xFFFF)
//   22  u8   ply (saturates at 255)
//   23  u8   flags: bit 0 rated, bit 1 chess.com, bits 2-3 speed (0 blitz, 1 rapid, 2 classical)

const MAGIC = 0x434f5044; // "DPOC" read as a little-endian u32
const VERSION = 1;
const HEADER_BYTES = 16;
const ROW_BYTES = 24;
const NO_CLOCK = 0xffff;
const PROMOTIONS = ["", "n", "b", "r", "q"];

function squareIndex(square: string): number {
  return square.charCodeAt(0) - 97 + 8 * (square.charCodeAt(1) - 49);
}

function squareName(index: number): string {
  return String.fromCharCode(97 + (index % 8), 49 + Math.floor(index / 8));
}

export function packMove(uci: string): number {
  const promotion = uci.length > 4 ? PROMOTIONS.indexOf(uci[4]!) : 0;
  return squareIndex(uci.slice(0, 2)) | (squareIndex(uci.slice(2, 4)) << 6) | (promotion << 12);
}

export function unpackMove(packed: number): string {
  return squareName(packed & 63) + squareName((packed >> 6) & 63) + PROMOTIONS[packed >> 12]!;
}

export function encodeOccurrences(rows: Occurrence[]): Uint8Array {
  const buffer = new ArrayBuffer(HEADER_BYTES + rows.length * ROW_BYTES);
  const view = new DataView(buffer);
  view.setUint32(0, MAGIC, true);
  view.setUint16(4, VERSION, true);
  view.setUint16(6, ROW_BYTES, true);
  view.setUint32(8, rows.length, true);
  for (const [i, row] of rows.entries()) {
    const at = HEADER_BYTES + i * ROW_BYTES;
    view.setBigUint64(at, row.positionHash, true);
    view.setUint32(at + 8, row.playedAt, true);
    view.setUint32(at + 12, row.gameIndex, true);
    view.setUint16(at + 16, packMove(row.move), true);
    view.setUint16(
      at + 18,
      row.clock === null ? NO_CLOCK : Math.min(Math.round(row.clock * 10), NO_CLOCK - 1),
      true,
    );
    view.setUint16(at + 20, Math.min(Math.round(row.base), 0xffff), true);
    view.setUint8(at + 22, Math.min(row.ply, 255));
    view.setUint8(
      at + 23,
      (row.rated ? 1 : 0) | (row.site === "chesscom" ? 2 : 0) | (SPEEDS.indexOf(row.speed) << 2),
    );
  }
  return new Uint8Array(buffer);
}

export function decodeOccurrences(bytes: Uint8Array): Occurrence[] {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  if (bytes.byteLength < HEADER_BYTES || view.getUint32(0, true) !== MAGIC) {
    throw new Error("not an occurrences file");
  }
  if (view.getUint16(4, true) !== VERSION || view.getUint16(6, true) !== ROW_BYTES) {
    throw new Error("unsupported occurrences version");
  }
  const count = view.getUint32(8, true);
  if (bytes.byteLength !== HEADER_BYTES + count * ROW_BYTES) throw new Error("truncated file");
  const rows: Occurrence[] = [];
  for (let i = 0; i < count; i++) {
    const at = HEADER_BYTES + i * ROW_BYTES;
    const clock = view.getUint16(at + 18, true);
    const flags = view.getUint8(at + 23);
    rows.push({
      positionHash: view.getBigUint64(at, true),
      playedAt: view.getUint32(at + 8, true),
      gameIndex: view.getUint32(at + 12, true),
      move: unpackMove(view.getUint16(at + 16, true)),
      clock: clock === NO_CLOCK ? null : clock / 10,
      base: view.getUint16(at + 20, true),
      ply: view.getUint8(at + 22),
      rated: (flags & 1) !== 0,
      site: (flags & 2) !== 0 ? "chesscom" : "lichess",
      speed: SPEEDS[(flags >> 2) & 3]!,
    });
  }
  return rows;
}
