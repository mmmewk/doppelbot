import { Chess } from "chess.js";
import { expect, test } from "vite-plus/test";
import { positionHash, positionKey, positionKeyFromFen } from "../src/index.ts";

function keyAfter(sans: string): string {
  const chess = new Chess();
  for (const san of sans.split(" ")) chess.move(san);
  return positionKey(chess.fen());
}

const START = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -";

test("the key is the first four FEN fields", () => {
  expect(keyAfter("e4")).toBe("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq -");
});

test("transpositions merge, whatever the move order or move counters", () => {
  expect(keyAfter("Nf3 d5 d4")).toBe(keyAfter("d4 d5 Nf3"));
  // Back at the start after four plies: fullmove 3 and halfmove clock 4 are ignored.
  expect(keyAfter("Nf3 Nf6 Ng1 Ng8")).toBe(START);
  expect(keyAfter("Nf3 Nf6 Ng1 Ng8 Nf3 Nf6 Ng1 Ng8")).toBe(START);
});

test("side to move is part of the key", () => {
  const board = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR";
  expect(positionKeyFromFen(`${board} w KQkq - 0 1`)).not.toBe(
    positionKeyFromFen(`${board} b KQkq - 0 1`),
  );
});

test("castling rights distinguish otherwise identical boards", () => {
  // The king walks out and back: same board and side to move, but both White rights are gone.
  const walked = keyAfter("e4 e5 Ke2 Ke7 Ke1 Ke8");
  const kept = keyAfter("e4 e5 Nf3 Nf6 Ng1 Ng8");
  expect(walked).toContain(" w - -");
  expect(kept).toContain(" w KQkq -");
  expect(walked.split(" ")[0]).toBe(kept.split(" ")[0]);
  expect(walked).not.toBe(kept);
});

test("a captured rook removes that side's castling right", () => {
  const key = positionKeyFromFen("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1");
  expect(key).toBe("r3k2r/8/8/8/8/8/8/R3K2R w KQkq -");
  const chess = new Chess("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1");
  chess.move("Rxh8");
  expect(positionKey(chess.fen())).toBe("r3k2R/8/8/8/8/8/8/R3K3 b Qq -");
});

test("en passant is in the key only when the capture is legal", () => {
  // 1.e4 a6 2.e5 d5: white can capture exd6.
  const available = keyAfter("e4 a6 e5 d5");
  expect(available).toMatch(/ w KQkq d6$/);
  // The same board a few moves later has no en passant right any more.
  expect(keyAfter("e4 a6 e5 d5 Nf3 Nf6 Ng1 Ng8")).toMatch(/ w KQkq -$/);
  // A double push that no enemy pawn can answer leaves the square out, so this merges with the
  // same board reached by a knight shuffle first.
  expect(keyAfter("e4")).toBe(keyAfter("Nf3 Nf6 Ng1 Ng8 e4"));
  // The capturing pawn is pinned along the fifth rank, so the capture is illegal.
  expect(positionKeyFromFen("8/8/8/K2pP2r/8/8/8/7k w - d6 0 1")).toBe(
    "8/8/8/K2pP2r/8/8/8/7k w - -",
  );
});

test("positionHash is stable and distinguishes keys", () => {
  expect(positionHash(START)).toBe(positionHash(keyAfter("Nf3 Nf6 Ng1 Ng8")));
  expect(positionHash(START)).not.toBe(positionHash(keyAfter("e4")));
  // Pinned value: if this changes, every stored occurrences file is invalid.
  expect(positionHash("")).toBe(0xcbf29ce484222325n);
  expect(positionHash("a")).toBe(0xaf63dc4c8601ec8cn);
});
