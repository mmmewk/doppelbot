// Opt-in benchmark, not part of `test` or `test:browser`: run it with `just bench` (or `vp run bench`).
// Measures session-create time and positions/second per precision and execution provider, then logs a table.
import { expect, test } from "vite-plus/test";
import * as ort from "onnxruntime-web";
import {
  MOVE_VOCABULARY_SIZE,
  MaiaSession,
  allPossibleMoves,
  interpolateElo,
  type EloEmbeddings,
  type MaiaExecutionProvider,
} from "../src/index.ts";

declare const __MAIA_FIXTURES_URL__: string;
declare const __MAIA_ARTIFACTS_URL__: string;
declare const __MAIA_EMBEDDINGS_URL__: string;

type Fixture = {
  positions: Array<{
    legal_mask: string;
    tokens: string;
    expected: Array<{ self_elo: number; oppo_elo: number; top_k: Array<{ move: string }> }>;
  }>;
};

const PRECISIONS = ["fp32", "fp16", "int8"] as const;
const PROVIDERS: MaiaExecutionProvider[] = ["wasm", "webgpu"];
const BATCH_SIZES = [1, 16, 64, 256];
const WARMUP_RUNS = 3;
const MIN_RUNS = 5;
const MAX_RUNS = 30;
const MIN_MEASURED_MS = 3000;
const CREATE_REPEATS = 3;

const modelUrl = (precision: (typeof PRECISIONS)[number]) =>
  `${__MAIA_ARTIFACTS_URL__}/maia3-5m${precision === "fp32" ? "" : `-${precision}`}/model.onnx`;

function decodeBase64(value: string): Uint8Array {
  return Uint8Array.from(atob(value), (character) => character.charCodeAt(0));
}

function median(values: number[]): number {
  const sorted = [...values].sort((a, b) => a - b);
  const middle = sorted.length >> 1;
  return sorted.length % 2 ? sorted[middle]! : (sorted[middle - 1]! + sorted[middle]!) / 2;
}

async function webGpuAdapter(): Promise<string | undefined> {
  const gpu = (navigator as unknown as { gpu?: { requestAdapter(): Promise<unknown> } }).gpu;
  if (!gpu) return undefined;
  const adapter = (await gpu.requestAdapter()) as {
    info?: { vendor: string; architecture: string; description: string };
    features?: Set<string>;
  } | null;
  if (!adapter) return undefined;
  const info = adapter.info;
  const f16 = adapter.features?.has("shader-f16") ? ", shader-f16" : ", no shader-f16";
  return `${info?.vendor ?? "?"} ${info?.architecture ?? ""} ${info?.description ?? ""}${f16}`.trim();
}

type Row = {
  precision: string;
  provider: string;
  batch: number;
  positionsPerSecond: number;
  medianMs: number;
  runs: number;
  firstRunMs: number;
  top1VsFixtures: number;
  finite: boolean;
};

test("benchmarks positions per second", { timeout: 40 * 60_000 }, async () => {
  const [fixture, embeddings] = await Promise.all([
    fetch(__MAIA_FIXTURES_URL__).then((response) => response.json() as Promise<Fixture>),
    fetch(__MAIA_EMBEDDINGS_URL__).then((response) => response.json() as Promise<EloEmbeddings>),
  ]);
  const moves = allPossibleMoves();
  const maxBatch = Math.max(...BATCH_SIZES);
  const positions = fixture.positions.slice(0, maxBatch);
  const tokens = Uint8Array.from(positions.flatMap((p) => Array.from(decodeBase64(p.tokens))));
  const masks = positions.map((p) => decodeBase64(p.legal_mask));
  // Rating pair 0 is what the fixtures' expected top moves were recorded at.
  const pair = positions[0]!.expected[0]!;
  const embedding = (elo: number) => interpolateElo(elo, embeddings);
  const selfVector = embedding(pair.self_elo);
  const oppoVector = embedding(pair.oppo_elo);
  const tokenSize = 64 * 97;
  const slice = (batch: number) => ({
    tokens: tokens.subarray(0, batch * tokenSize),
    selfEmb: Float32Array.from({ length: batch * 128 }, (_, i) => selfVector[i % 128]!),
    oppoEmb: Float32Array.from({ length: batch * 128 }, (_, i) => oppoVector[i % 128]!),
  });

  const environment = {
    crossOriginIsolated: (globalThis as { crossOriginIsolated?: boolean }).crossOriginIsolated,
    hardwareConcurrency: navigator.hardwareConcurrency,
    userAgent: navigator.userAgent,
    webgpuAdapter: await webGpuAdapter(),
  };
  console.log(`BENCH environment ${JSON.stringify(environment)}`);

  const rows: Row[] = [];
  const creates: Array<{
    precision: string;
    provider: string;
    fetchMs?: number;
    createMs?: number;
    error?: string;
  }> = [];
  for (const provider of PROVIDERS) {
    if (provider === "webgpu" && !environment.webgpuAdapter) {
      creates.push({ precision: "all", provider, error: "no WebGPU adapter in this browser" });
      continue;
    }
    // Absorb one-off runtime start-up (WASM instantiation, GPU device) so it isn't charged to fp32.
    try {
      const warmup = await (await fetch(modelUrl("int8"))).arrayBuffer();
      await MaiaSession.create(new Uint8Array(warmup), provider);
    } catch (error) {
      creates.push({ precision: "all", provider, error: `runtime start-up: ${String(error)}` });
      continue;
    }

    for (const precision of PRECISIONS) {
      const fetchStart = performance.now();
      const bytes = new Uint8Array(await (await fetch(modelUrl(precision))).arrayBuffer());
      const fetchMs = performance.now() - fetchStart;
      let session: MaiaSession;
      const createTimes: number[] = [];
      try {
        // Time repeated creates but benchmark the last one.
        let last: MaiaSession | undefined;
        for (let repeat = 0; repeat < CREATE_REPEATS; repeat += 1) {
          const start = performance.now();
          last = await MaiaSession.create(bytes, provider);
          createTimes.push(performance.now() - start);
        }
        session = last!;
      } catch (error) {
        creates.push({ precision, provider, fetchMs, error: String(error).slice(0, 300) });
        continue;
      }
      creates.push({ precision, provider, fetchMs, createMs: median(createTimes) });

      for (const batch of BATCH_SIZES) {
        const input = slice(batch);
        try {
          const firstStart = performance.now();
          const first = await session.run(input.tokens, input.selfEmb, input.oppoEmb);
          const firstRunMs = performance.now() - firstStart;
          let matches = 0;
          for (let index = 0; index < batch; index += 1) {
            const logits = first.moveLogits.subarray(
              index * MOVE_VOCABULARY_SIZE,
              (index + 1) * MOVE_VOCABULARY_SIZE,
            );
            let best = -1;
            for (let move = 0; move < MOVE_VOCABULARY_SIZE; move += 1) {
              if (masks[index]![move] && (best < 0 || logits[move]! > logits[best]!)) best = move;
            }
            if (moves[best] === positions[index]!.expected[0]!.top_k[0]!.move) matches += 1;
          }
          const finite = first.moveLogits.every(Number.isFinite);
          for (let run = 1; run < WARMUP_RUNS; run += 1) {
            await session.run(input.tokens, input.selfEmb, input.oppoEmb);
          }
          const times: number[] = [];
          let total = 0;
          while (times.length < MIN_RUNS || (total < MIN_MEASURED_MS && times.length < MAX_RUNS)) {
            const start = performance.now();
            await session.run(input.tokens, input.selfEmb, input.oppoEmb);
            const elapsed = performance.now() - start;
            times.push(elapsed);
            total += elapsed;
          }
          const medianMs = median(times);
          rows.push({
            precision,
            provider,
            batch,
            positionsPerSecond: (batch / medianMs) * 1000,
            medianMs,
            runs: times.length,
            firstRunMs,
            top1VsFixtures: matches / batch,
            finite,
          });
        } catch (error) {
          creates.push({
            precision,
            provider,
            error: `batch ${batch}: ${String(error).slice(0, 300)}`,
          });
        }
      }
    }
  }

  console.log(`BENCH wasm numThreads after init: ${ort.env.wasm.numThreads}`);
  console.log(`BENCH creates ${JSON.stringify(creates)}`);
  console.log(`BENCH rows ${JSON.stringify(rows)}`);
  const table = [
    "| Precision | Provider | Batch | Positions/s | Median ms/run | Runs | First run ms | Top-1 vs fixtures | Finite |",
    "|---|---|---|---|---|---|---|---|---|",
    ...rows.map(
      (r) =>
        `| ${r.precision} | ${r.provider} | ${r.batch} | ${r.positionsPerSecond.toFixed(0)} | ${r.medianMs.toFixed(1)} | ${r.runs} | ${r.firstRunMs.toFixed(0)} | ${(r.top1VsFixtures * 100).toFixed(0)}% | ${r.finite} |`,
    ),
  ].join("\n");
  console.log(`BENCH table\n${table}`);
  expect(rows.length).toBeGreaterThan(0);
});
