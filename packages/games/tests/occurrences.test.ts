import { expect, test } from "vite-plus/test";
import {
  buildOccurrences,
  decodeOccurrences,
  encodeOccurrences,
  isTimeTrouble,
  packMove,
  positionHash,
  positionKey,
  unpackMove,
  type Occurrence,
} from "../src/index.ts";
import { makeGame } from "./helpers.ts";

const START_HASH = positionHash("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -");

test("emits one occurrence per move by the user, keyed by the position before the move", () => {
  const game = makeGame("e4 e5 Nf3 Nc6");
  const white = buildOccurrences(game, 7, "Alice");
  expect(white.map((o) => [o.ply, o.move, o.gameIndex])).toEqual([
    [0, "e2e4", 7],
    [2, "g1f3", 7],
  ]);
  expect(white[0]!.positionHash).toBe(START_HASH);
  const black = buildOccurrences(game, 0, "bob");
  expect(black.map((o) => [o.ply, o.move])).toEqual([
    [1, "e7e5"],
    [3, "b8c6"],
  ]);
  expect(buildOccurrences(game, 0, "carol")).toEqual([]);
});

test("a transposition within a game repeats the same hash", () => {
  const rows = buildOccurrences(makeGame("Nf3 Nf6 Ng1 Ng8 d4 d5"), 0, "alice");
  expect(rows.map((o) => o.ply)).toEqual([0, 2, 4]);
  expect(rows[0]!.positionHash).toBe(START_HASH);
  expect(rows[2]!.positionHash).toBe(START_HASH); // back at the start after four plies
  expect(rows[1]!.positionHash).not.toBe(START_HASH);
});

test("hash of the first move position equals the shared key hash for both colors", () => {
  const rows = buildOccurrences(makeGame("e4 e5"), 0, "bob");
  expect(rows[0]!.positionHash).toBe(
    positionHash(positionKey("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1")),
  );
});

test("replays castling and promotion moves", () => {
  const rows = buildOccurrences(makeGame("e4 e5 Nf3 Nc6 Bc4 Bc5 O-O Nf6 d3 O-O"), 0, "alice");
  expect(rows.map((o) => o.move)).toEqual(["e2e4", "g1f3", "f1c4", "e1g1", "d2d3"]);
  const promo = buildOccurrences(makeGame("a4 b5 axb5 a6 bxa6 Nf6 a7 Ng8 axb8=Q"), 0, "alice");
  expect(promo.at(-1)!.move).toBe("a7b8q");
});

test("clock before each move is the mover's previous clock, or the base time", () => {
  const game = makeGame("e4 e5 Nf3 Nc6 Bb5", { clocks: [178, 177, 170, 169, 160] });
  expect(buildOccurrences(game, 0, "alice").map((o) => o.clock)).toEqual([180, 178, 170]);
  expect(buildOccurrences(game, 0, "bob").map((o) => o.clock)).toEqual([180, 177]);
  const noClocks = makeGame("e4 e5");
  expect(buildOccurrences(noClocks, 0, "alice")[0]!.clock).toBeNull();
});

test("time trouble matches the Python filter: under either threshold", () => {
  const filter = { minClockSeconds: 10, minClockFraction: 0.1 };
  const row = (clock: number | null, base: number) => ({ clock, base }) as unknown as Occurrence;
  expect(isTimeTrouble(row(null, 180), filter)).toBe(false);
  expect(isTimeTrouble(row(9.9, 600), filter)).toBe(true); // under 10 s
  expect(isTimeTrouble(row(10, 600), filter)).toBe(true); // under 10% of 600 s
  expect(isTimeTrouble(row(60, 600), filter)).toBe(false);
  expect(isTimeTrouble(row(20, 180), filter)).toBe(false);
  expect(isTimeTrouble(row(17, 180), filter)).toBe(true); // 17 < 18
});

test("moves pack into 15 bits and back", () => {
  for (const uci of ["a1a2", "h8a1", "e2e4", "e1g1", "a7b8q", "h2g1n", "c7c8r", "g7h8b"]) {
    expect(packMove(uci)).toBeLessThan(1 << 15);
    expect(unpackMove(packMove(uci))).toBe(uci);
  }
});

test("occurrences round-trip through the binary format", () => {
  const game = makeGame("e4 e5 Nf3 Nc6", {
    clocks: [178.4, 177, 170.04, 169],
    site: "chesscom",
    rated: false,
    speed: "classical",
    base: 900,
  });
  const rows = buildOccurrences(game, 3, "alice");
  const bytes = encodeOccurrences(rows);
  expect(bytes.byteLength).toBe(16 + 2 * 24);
  const decoded = decodeOccurrences(bytes);
  expect(decoded).toEqual(
    rows.map((row) => ({
      ...row,
      clock: row.clock === null ? null : Math.round(row.clock * 10) / 10,
    })),
  );
  expect(decoded[0]).toMatchObject({
    site: "chesscom",
    rated: false,
    speed: "classical",
    base: 900,
  });
});

test("encoding saturates instead of overflowing, and unknown clocks survive", () => {
  const row: Occurrence = {
    positionHash: 0xffffffffffffffffn,
    move: "e2e4",
    playedAt: 4_000_000_000,
    gameIndex: 4_000_000_000,
    ply: 400,
    clock: 100_000,
    base: 100_000,
    rated: true,
    site: "lichess",
    speed: "rapid",
  };
  const [back, unknown] = decodeOccurrences(encodeOccurrences([row, { ...row, clock: null }]));
  expect(back).toMatchObject({
    positionHash: row.positionHash,
    ply: 255,
    clock: 6553.4,
    base: 65535,
  });
  expect(unknown!.clock).toBeNull();
});

test("rejects files that aren't occurrences", () => {
  expect(() => decodeOccurrences(new Uint8Array(16))).toThrow("not an occurrences file");
  const bytes = encodeOccurrences(buildOccurrences(makeGame("e4"), 0, "alice"));
  expect(() => decodeOccurrences(bytes.slice(0, bytes.length - 1))).toThrow("truncated");
});
