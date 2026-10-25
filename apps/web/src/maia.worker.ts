import ortWasm from "onnxruntime-web/ort-wasm-simd-threaded.jsep.wasm?url";
import ortMjs from "onnxruntime-web/ort-wasm-simd-threaded.jsep.mjs?url";
import {
  MaiaSession,
  configureWasm,
  historicalTokens,
  interpolateElo,
  legalMovesMask,
  maskedSoftmax,
  mirrorMove,
} from "@doppelbot/maia";

configureWasm({ mjs: ortMjs, wasm: ortWasm });

let session: MaiaSession | undefined;
let moves: readonly string[] = [];
let embeddings: { low: number[]; high: number[] } | undefined;

async function load(file: string) {
  const response = await fetch(`/models/maia3-5m/${file}`);
  // A missing file falls through to the SPA's index.html, so check the type as well as the status.
  if (!response.ok || response.headers.get("content-type")?.includes("text/html")) {
    throw new Error(`model file /models/maia3-5m/${file} is missing (${response.status})`);
  }
  return response;
}

async function ready() {
  if (session && embeddings) return;
  const [model, moveList, elo] = await Promise.all([
    load("model.onnx").then((response) => response.arrayBuffer()),
    load("moves.json").then((response) => response.json() as Promise<string[]>),
    load("elo_emb.json").then(
      (response) => response.json() as Promise<{ low: number[]; high: number[] }>,
    ),
  ]);
  moves = moveList;
  embeddings = elo;
  try {
    session = await MaiaSession.create(model, "webgpu");
    postMessage({ type: "backend", backend: "WebGPU" });
  } catch {
    session = await MaiaSession.create(model, "wasm");
    postMessage({ type: "backend", backend: "WASM" });
  }
}

/** Sample proportionally to p^(1/temperature); temperature 0 means always the top move. */
function sample(ranked: { move: string; probability: number }[], temperature: number) {
  if (temperature <= 0) return ranked[0]!;
  const weights = ranked.map((item) => item.probability ** (1 / temperature));
  let r = Math.random() * weights.reduce((sum, weight) => sum + weight, 0);
  for (let i = 0; i < ranked.length; i += 1) {
    r -= weights[i]!;
    if (r <= 0) return ranked[i]!;
  }
  return ranked.at(-1)!;
}

self.onmessage = async ({
  data,
}: MessageEvent<{ fens: string[]; selfElo: number; oppoElo: number; temperature: number }>) => {
  try {
    await ready();
    const start = performance.now();
    const index = new Map(moves.map((move, position) => [move, position]));
    const output = await session!.run(
      historicalTokens(data.fens),
      interpolateElo(data.selfElo, embeddings!),
      interpolateElo(data.oppoElo, embeddings!),
    );
    const probabilities = maskedSoftmax(
      output.moveLogits,
      legalMovesMask(data.fens.at(-1)!, index),
    );
    // Moves are indexed from the mover's perspective; convert back to real squares when Black moves.
    const black = data.fens.at(-1)!.split(" ")[1] === "b";
    const ranked = moves
      .map((move, position) => ({
        move: black ? mirrorMove(move) : move,
        probability: probabilities[position]!,
      }))
      .filter((item) => item.probability > 0)
      .sort((left, right) => right.probability - left.probability);
    postMessage({
      move: sample(ranked, data.temperature).move,
      top: ranked.slice(0, 5),
      logits: Array.from(output.valueLogits),
      ms: performance.now() - start,
    });
  } catch (error) {
    postMessage({ error: error instanceof Error ? error.message : String(error) });
  }
};
