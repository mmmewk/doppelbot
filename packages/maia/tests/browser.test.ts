import { expect, test } from "vite-plus/test";
import {
  MaiaSession,
  allPossibleMoves,
  decodeWdl,
  interpolateElo,
  maskedSoftmax,
} from "../src/index.ts";

declare const __MAIA_FIXTURES_URL__: string;
declare const __MAIA_MODEL_URL__: string;
declare const __MAIA_EMBEDDINGS_URL__: string;

type Fixture = {
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

async function fetchLocal<T>(url: string): Promise<T> {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`could not load local parity artifact: ${response.status}`);
  return response.json() as Promise<T>;
}

function decodeBase64(value: string): Uint8Array {
  return Uint8Array.from(atob(value), (character) => character.charCodeAt(0));
}

async function assertParity(executionProvider: "wasm" | "webgpu") {
  const [fixture, embeddings, modelResponse] = await Promise.all([
    fetchLocal<Fixture>(__MAIA_FIXTURES_URL__),
    fetchLocal<{ low: number[]; high: number[] }>(__MAIA_EMBEDDINGS_URL__),
    fetch(__MAIA_MODEL_URL__),
  ]);
  if (!modelResponse.ok)
    throw new Error(`could not load local ONNX model: ${modelResponse.status}`);

  const moves = allPossibleMoves();
  const indices = new Map(moves.map((move, index) => [move, index]));
  const tokens = Uint8Array.from(
    fixture.positions.flatMap((position) => Array.from(decodeBase64(position.tokens))),
  );
  const masks = fixture.positions.map((position) => decodeBase64(position.legal_mask));
  const model = new Uint8Array(await modelResponse.arrayBuffer());
  const session = await MaiaSession.create(model, executionProvider);

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
}

test("matches Python parity fixtures in browser WASM", { timeout: 180_000 }, async () => {
  await assertParity("wasm");
});

test.skipIf(!("gpu" in navigator))(
  "matches Python parity fixtures in browser WebGPU",
  { timeout: 180_000 },
  async () => {
    await assertParity("webgpu");
  },
);
