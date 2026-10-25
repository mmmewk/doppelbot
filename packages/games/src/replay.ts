import { Chess } from "chess.js";

/** Replay SAN moves from the standard start and return them as UCI, or null if any is illegal. */
export function sansToUci(sans: string[]): string[] | null {
  const chess = new Chess();
  const uci: string[] = [];
  try {
    for (const san of sans) {
      const move = chess.move(san);
      uci.push(move.from + move.to + (move.promotion ?? ""));
    }
  } catch {
    return null;
  }
  return uci;
}
