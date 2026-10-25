import { Chess } from "chess.js";

/**
 * Position key: the first four FEN fields.
 *
 *   <piece placement> <side to move> <castling rights> <en passant square>
 *   e.g. "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq -"
 *
 * Kept: board, side to move, castling rights, en passant.
 * Dropped: halfmove clock, fullmove number, move history, clocks.
 *
 * - Transpositions merge (the key says nothing about how the position was reached).
 * - Side to move is part of the key, so "same board, other side to move" is a different position.
 * - Castling rights are kept as FEN states them (chess.js drops a right once the king or that rook
 *   moves or the rook is captured). Two boards that differ only in rights have different futures,
 *   so they stay different keys.
 * - En passant is kept only when an en passant capture is actually legal (chess.js emits the
 *   square in that case only, matching Lichess and polyglot practice). After 1.e4 the key says "-"
 *   because no black pawn can capture, so reaching that board by another move order still merges.
 *
 * `fen` must come from chess.js (`chess.fen()`) so the en passant rule above holds. For a FEN from
 * anywhere else, use `positionKeyFromFen`.
 */
export function positionKey(fen: string): string {
  const fields = fen.split(" ");
  return fields.slice(0, 4).join(" ");
}

/** Normalize an arbitrary FEN through chess.js first, so a non-capturable en passant square is dropped. */
export function positionKeyFromFen(fen: string): string {
  return positionKey(new Chess(fen).fen());
}

const FNV_OFFSET = 0xcbf29ce484222325n;
const FNV_PRIME = 0x100000001b3n;
const MASK_64 = 0xffffffffffffffffn;

/**
 * 64-bit FNV-1a of the key's UTF-8 bytes. Synchronous and dependency-free so the browser and
 * Lambda compute identical values; it is what occurrence rows store instead of the ~60-byte key.
 * With 64 bits, collisions among a single user's few million positions are negligible (~1e-7).
 * Shared, cross-user tables (Stockfish evals under POS#) should use the full key string instead.
 */
export function positionHash(key: string): bigint {
  let hash = FNV_OFFSET;
  for (const byte of new TextEncoder().encode(key)) {
    hash ^= BigInt(byte);
    hash = (hash * FNV_PRIME) & MASK_64;
  }
  return hash;
}
