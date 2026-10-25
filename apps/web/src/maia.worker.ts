import ortWasm from "onnxruntime-web/ort-wasm-simd-threaded.jsep.wasm?url";
import ortMjs from "onnxruntime-web/ort-wasm-simd-threaded.jsep.mjs?url";
import {
  MaiaSession,
  configureWasm,
  decodeWdl,
  historicalTokens,
  interpolateElo,
  rankMoves,
  sampleMove,
} from "@doppelbot/maia";
import type { BotReply, BotRequest } from "./bot.ts";

configureWasm({ mjs: ortMjs, wasm: ortWasm });

let session: MaiaSession | undefined;
let moves: readonly string[] = [];
let index = new Map<string, number>();
let embeddings: { low: number[]; high: number[] } | undefined;

const MODEL_ID = "maia3-5m";
const CACHE_NAME = `doppelbot-models-${MODEL_ID}`;

/** Fetch a model file through the Cache API so reloads don't download the 21 MB model again. */
async function load(file: string) {
  const url = `/models/${MODEL_ID}/${file}`;
  const cache = await caches.open(CACHE_NAME).catch(() => undefined);
  const cached = await cache?.match(url);
  if (cached) return cached;
  const response = await fetch(url);
  // A missing file falls through to the SPA's index.html, so check the type as well as the status.
  if (!response.ok || response.headers.get("content-type")?.includes("text/html")) {
    throw new Error(`model file ${url} is missing (${response.status})`);
  }
  await cache?.put(url, response.clone()).catch(() => undefined);
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
  index = new Map(moves.map((move, position) => [move, position]));
  embeddings = elo;
  try {
    session = await MaiaSession.create(model, "webgpu");
    postMessage({ backend: "WebGPU" } satisfies BotReply);
  } catch {
    session = await MaiaSession.create(model, "wasm");
    postMessage({ backend: "WASM" } satisfies BotReply);
  }
}

self.onmessage = async ({ data }: MessageEvent<BotRequest>) => {
  try {
    await ready();
    const start = performance.now();
    const output = await session!.run(
      historicalTokens(data.fens),
      interpolateElo(data.selfElo, embeddings!),
      interpolateElo(data.oppoElo, embeddings!),
    );
    const ranked = rankMoves(output.moveLogits, data.fens.at(-1)!, moves, index);
    postMessage({
      id: data.id,
      move: sampleMove(ranked, data.temperature).move,
      top: ranked.slice(0, 5),
      wdl: decodeWdl(output.valueLogits),
      ms: performance.now() - start,
    } satisfies BotReply);
  } catch (error) {
    postMessage({
      id: data.id,
      error: error instanceof Error ? error.message : String(error),
    } satisfies BotReply);
  }
};
