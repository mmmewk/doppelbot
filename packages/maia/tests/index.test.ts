import { existsSync, readFileSync } from "node:fs";
import { resolve } from "node:path";
import { expect, test } from "vite-plus/test";
import {
  MaiaSession,
  allPossibleMoves,
  decodeWdl,
  historicalTokens,
  interpolateElo,
  legalMovesMask,
  maskedSoftmax,
  mirrorMove,
  tokenizeBoard,
} from "../src/index.ts";

const START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

test("tokenizes and front-pads a short history", () => {
  const frame = tokenizeBoard(START_FEN);
  const history = historicalTokens([START_FEN]);
  expect(history).toHaveLength(64 * 97);
  for (let square = 0; square < 64; square += 1) {
    for (let position = 0; position < 8; position += 1) {
      expect(history.slice(square * 97 + position * 12, square * 97 + (position + 1) * 12)).toEqual(
        frame.slice(square * 12, (square + 1) * 12),
      );
    }
  }
});

test("uses mover-relative move indices and promotion vocabulary", () => {
  const moves = allPossibleMoves();
  const indices = new Map(moves.map((move, index) => [move, index]));
  expect(moves).toHaveLength(4352);
  expect(mirrorMove("e7e5")).toBe("e2e4");
  const blackMask = legalMovesMask(
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1",
    indices,
  );
  expect(blackMask[indices.get("e2e4")!]).toBe(1);
  expect(blackMask[indices.get("e7e5")!]).toBe(0);
});

test("masks illegal moves and decodes WDL in mover order", () => {
  const probabilities = maskedSoftmax([2, 1, 0], [1, 1, 0]);
  expect(probabilities[0]).toBeCloseTo(0.7310586);
  expect(probabilities[1]).toBeCloseTo(0.2689414);
  expect(probabilities[2]).toBe(0);

  const wdl = decodeWdl([0, 1, 2]);
  expect(wdl.win).toBeCloseTo(0.66524096);
  expect(wdl.draw).toBeCloseTo(0.24472847);
  expect(wdl.loss).toBeCloseTo(0.09003057);
});

test(
  "matches Python token and legal-mask fixtures when they are available",
  { timeout: 30_000 },
  () => {
    const fixturePath = resolve(import.meta.dirname, "../../../python/data/fixtures/maia3-5m.json");
    if (!existsSync(fixturePath)) return;
    const fixture = JSON.parse(readFileSync(fixturePath, "utf8")) as {
      positions: Array<{ fens: string[]; tokens: string; legal_mask: string }>;
    };
    const moves = allPossibleMoves();
    const indices = new Map(moves.map((move, index) => [move, index]));
    for (const position of fixture.positions) {
      expect(Buffer.from(historicalTokens(position.fens))).toEqual(
        Buffer.from(position.tokens, "base64"),
      );
      expect(Buffer.from(legalMovesMask(position.fens.at(-1)!, indices))).toEqual(
        Buffer.from(position.legal_mask, "base64"),
      );
    }
  },
);

test(
  "matches Python ONNX probabilities and WDL in Node/WASM when artifacts are available",
  { timeout: 120_000 },
  async () => {
    const fixturePath = resolve(import.meta.dirname, "../../../python/data/fixtures/maia3-5m.json");
    const artifactDirectory = resolve(import.meta.dirname, "../../../python/artifacts/maia3-5m");
    const modelPath = resolve(artifactDirectory, "model.onnx");
    const embeddingsPath = resolve(artifactDirectory, "elo_emb.json");
    if (!existsSync(fixturePath) || !existsSync(modelPath) || !existsSync(embeddingsPath)) return;

    const fixture = JSON.parse(readFileSync(fixturePath, "utf8")) as {
      positions: Array<{
        legal_mask: string;
        tokens: string;
        expected: Array<{
          self_elo: number;
          oppo_elo: number;
          top_k: Array<{ move: string; probability: number }>;
          wdl: { win: number; draw: number; loss: number };
        }>;
      }>;
    };
    const embeddings = JSON.parse(readFileSync(embeddingsPath, "utf8")) as {
      low: number[];
      high: number[];
    };
    const moves = allPossibleMoves();
    const indices = new Map(moves.map((move, index) => [move, index]));
    const tokens = Uint8Array.from(
      fixture.positions.flatMap((position) => Array.from(Buffer.from(position.tokens, "base64"))),
    );
    const masks = fixture.positions.map((position) =>
      Uint8Array.from(Buffer.from(position.legal_mask, "base64")),
    );
    const session = await MaiaSession.create(modelPath);

    for (let ratingPair = 0; ratingPair < 3; ratingPair += 1) {
      const expected = fixture.positions.map((position) => position.expected[ratingPair]!);
      const selfEmb = Float32Array.from(
        expected.flatMap((entry) => Array.from(interpolateElo(entry.self_elo, embeddings))),
      );
      const oppoEmb = Float32Array.from(
        expected.flatMap((entry) => Array.from(interpolateElo(entry.oppo_elo, embeddings))),
      );
      const output = await session.run(tokens, selfEmb, oppoEmb);
      for (let positionIndex = 0; positionIndex < fixture.positions.length; positionIndex += 1) {
        const logits = output.moveLogits.subarray(positionIndex * 4352, (positionIndex + 1) * 4352);
        const probabilities = maskedSoftmax(logits, masks[positionIndex]!);
        for (const topMove of expected[positionIndex]!.top_k) {
          expect(probabilities[indices.get(topMove.move)!]).toBeCloseTo(topMove.probability, 4);
        }
        const value = output.valueLogits.subarray(positionIndex * 3, positionIndex * 3 + 3);
        const wdl = decodeWdl(value);
        expect(wdl.win).toBeCloseTo(expected[positionIndex]!.wdl.win, 4);
        expect(wdl.draw).toBeCloseTo(expected[positionIndex]!.wdl.draw, 4);
        expect(wdl.loss).toBeCloseTo(expected[positionIndex]!.wdl.loss, 4);
      }
    }
  },
);
