import { Chess, type Color, type Square } from "chess.js";

export type UciMove = { from: string; to: string; promotion?: string };

/** A copy of the game with full move history, so React state is never mutated in place. */
export function copyGame(game: Chess): Chess {
  const copy = new Chess();
  copy.loadPgn(game.pgn());
  return copy;
}

export function withMove(game: Chess, move: UciMove | string): Chess {
  const next = copyGame(game);
  next.move(typeof move === "string" ? fromUci(move) : move);
  return next;
}

export function fromUci(uci: string): UciMove {
  return { from: uci.slice(0, 2), to: uci.slice(2, 4), promotion: uci[4] };
}

/** The last 8 positions (oldest first) Maia sees as history, rebuilt from the game's move list. */
export function recentFens(game: Chess): string[] {
  const moves = game.history({ verbose: true });
  return [moves[0]?.before ?? game.fen(), ...moves.map((move) => move.after)].slice(-8);
}

export type Outcome = { winner?: Color; reason: string };

/** How the game ended, or undefined while it is still going. */
export function outcome(game: Chess, resigned?: Color): Outcome | undefined {
  if (resigned) return { winner: resigned === "w" ? "b" : "w", reason: "resignation" };
  if (game.isCheckmate()) return { winner: game.turn() === "w" ? "b" : "w", reason: "checkmate" };
  if (game.isStalemate()) return { reason: "stalemate" };
  if (game.isThreefoldRepetition()) return { reason: "threefold repetition" };
  if (game.isInsufficientMaterial()) return { reason: "insufficient material" };
  if (game.isDrawByFiftyMoves()) return { reason: "the 50-move rule" };
  return undefined;
}

/** The squares a piece on `square` can move to. */
export function targets(game: Chess, square: string): string[] {
  return game.moves({ square: square as Square, verbose: true }).map((move) => move.to);
}

export function isPromotion(game: Chess, move: UciMove): boolean {
  return game
    .moves({ verbose: true })
    .some((legal) => legal.from === move.from && legal.to === move.to && legal.promotion);
}
