import { Chess } from "chess.js";

export type Trainer = {
  name: string;
  role: string;
  description: string;
  timeControl: string;
  /** The line shown on the card's board, in SAN from the starting position. */
  line: readonly string[];
  /** Describes the board for screen readers; not shown. */
  boardLabel: string;
  playable: boolean;
  link?: { href: string; label: string };
};

export const TRAINERS: readonly Trainer[] = [
  {
    name: "Doppel",
    role: "Your doppelgänger",
    description:
      "Doppel plays chess like you and prefers your openings. It's best for drilling the lines you already know and building speed.",
    timeControl: "Fast time controls, to sharpen your pattern recognition.",
    line: "d4 d5 Bf4 Nf6 e3 e6 Nf3 c5 c3 Nc6 Nbd2 Bd6 Bg3 O-O Bd3".split(" "),
    boardLabel: "The London System, the way you always set it up.",
    playable: false,
  },
  {
    name: "Zephyr",
    role: "Your companion",
    description:
      "Named for the Greek god of the west wind, Zephyr follows you around but explores from a different angle. It knows the same tactics you do, and like a gentle wind it blows you into new, unexpected lines you wouldn't normally play.",
    timeControl: "Medium time controls, to practice answering deviations from your plan.",
    line: "d4 d5 c4 e5 dxe5 d4 Nf3 Nc6".split(" "),
    boardLabel: "The Albin Countergambit: a Queen's Gambit you rarely see.",
    playable: false,
  },
  {
    name: "Loki",
    role: "Your nemesis",
    description:
      "Loki plays the traps you're likely to miss, and makes the blunders you're likely not to punish.",
    timeControl:
      "Longer time controls, to practice spotting the mistakes and chances you normally miss.",
    line: "e4 e5 Nf3 Nc6 Bc4 Nd4 Nxe5 Qg5 Nxf7 Qxg2 Rf1 Qxe4+ Be2 Nf3#".split(" "),
    boardLabel: "The Blackburne Shilling Gambit, ending in smothered mate.",
    playable: false,
  },
  {
    name: "Maia",
    role: "The average player",
    description:
      "Maia is a human-like model of a typical player at any rating, made by the Maia Chess team at the University of Toronto's CSSLab and trained on games from Lichess. It's the base model that Doppel, Zephyr and Loki are built on.",
    timeControl: "Any time control. Playable now, at your rating plus a stretch.",
    line: "e4 e5 Nf3 Nc6 Bc4 Bc5".split(" "),
    boardLabel: "A quiet Italian Game.",
    playable: true,
    link: { href: "https://www.maiachess.com", label: "About Maia" },
  },
];

/** Play a SAN line from the starting position. Throws on an illegal move. */
export function playLine(line: readonly string[]): Chess {
  const game = new Chess();
  for (const san of line) game.move(san);
  return game;
}
