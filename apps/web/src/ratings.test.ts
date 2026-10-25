import { expect, test } from "vite-plus/test";
import { lichessEquivalent, pickRating } from "./ratings.ts";

test("passes Lichess ratings through", () => {
  expect(lichessEquivalent({ site: "lichess", timeControl: "rapid", rating: 1432 })).toBe(1432);
});

test("converts chess.com blitz and rapid to the Lichess scale", () => {
  expect(lichessEquivalent({ site: "chesscom", timeControl: "blitz", rating: 1200 })).toBe(1555);
  expect(lichessEquivalent({ site: "chesscom", timeControl: "blitz", rating: 1250 })).toBe(1588);
  // Rapid 1420 is blitz 1200 on chess.com.
  expect(lichessEquivalent({ site: "chesscom", timeControl: "rapid", rating: 1420 })).toBe(1555);
});

test("extends the table past its ends", () => {
  expect(lichessEquivalent({ site: "chesscom", timeControl: "blitz", rating: 400 })).toBe(1020);
  expect(lichessEquivalent({ site: "chesscom", timeControl: "blitz", rating: 3100 })).toBe(2815);
});

test("picks the time control with the most games", () => {
  expect(
    pickRating([
      { site: "lichess", timeControl: "blitz", rating: 1500, games: 40 },
      { site: "chesscom", timeControl: "rapid", rating: 1100, games: 900 },
    ]),
  ).toMatchObject({ site: "chesscom", timeControl: "rapid" });
});
