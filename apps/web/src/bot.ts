import type { RankedMove, Wdl } from "@doppelbot/maia";

/** A request for one bot move. `id` lets the page ignore replies for a game it has since reset. */
export type BotRequest = {
  id: number;
  fens: string[];
  selfElo: number;
  oppoElo: number;
  temperature: number;
};

export type BotMove = {
  id: number;
  move: string;
  top: RankedMove[];
  /** Win/draw/loss from the bot's side, for the position it moved from. */
  wdl: Wdl;
  ms: number;
};

export type BotReply = { backend: string } | { id: number; error: string } | BotMove;
