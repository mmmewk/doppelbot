import { Chess, type Color, type PieceSymbol } from "chess.js";
import * as ort from "onnxruntime-web";

export const HISTORY = 8;
export const TOKEN_CHANNELS = 12;
export const TOKEN_WIDTH = HISTORY * TOKEN_CHANNELS + 1;
export const MOVE_VOCABULARY_SIZE = 4352;

const PIECE_CHANNEL: Record<PieceSymbol, number> = {
  p: 0,
  n: 1,
  b: 2,
  r: 3,
  q: 4,
  k: 5,
};

const PROMOTION_PIECES = ["q", "r", "b", "n"] as const;

export type Wdl = { win: number; draw: number; loss: number };
export type EloEmbeddings = { low: readonly number[]; high: readonly number[] };
export type MaiaOutput = {
  moveLogits: Float32Array;
  valueLogits: Float32Array;
  ponderLogits: Float32Array;
};
export type MaiaExecutionProvider = "wasm" | "webgpu";

/**
 * Point onnxruntime at separately served copies of its WASM glue (.mjs) and binary. Required when
 * this package is bundled into a worker: otherwise the threaded runtime spawns its pthread workers
 * from the bundle's own URL, re-running the host worker instead, and session creation never resolves.
 */
export function configureWasm(paths: typeof ort.env.wasm.wasmPaths): void {
  ort.env.wasm.wasmPaths = paths;
}

/** Encode one board from the mover's perspective, matching maia3.dataset.tokenize_board. */
export function tokenizeBoard(fen: string): Uint8Array {
  const chess = new Chess(fen);
  const mirrored = chess.turn() === "b";
  const tokens = new Uint8Array(64 * TOKEN_CHANNELS);
  const board = chess.board();

  for (let rank = 0; rank < 8; rank += 1) {
    for (let file = 0; file < 8; file += 1) {
      const piece = board[7 - rank]![file];
      if (piece === null) continue;
      const targetRank = mirrored ? 7 - rank : rank;
      const targetColor = mirrored ? opposite(piece.color) : piece.color;
      const channel = PIECE_CHANNEL[piece.type] + (targetColor === "b" ? 6 : 0);
      tokens[(targetRank * 8 + file) * TOKEN_CHANNELS + channel] = 1;
    }
  }
  return tokens;
}

/** Build the 64x97 tensor; frames are oldest first and missing frames repeat the oldest. */
export function historicalTokens(fens: readonly string[]): Uint8Array {
  if (fens.length === 0 || fens.length > HISTORY) {
    throw new RangeError(`expected 1 to ${HISTORY} FENs, received ${fens.length}`);
  }
  const frames = fens.map(tokenizeBoard);
  const tokens = new Uint8Array(64 * TOKEN_WIDTH);
  const padding = HISTORY - frames.length;
  for (let square = 0; square < 64; square += 1) {
    for (let history = 0; history < HISTORY; history += 1) {
      const frame = frames[Math.max(0, history - padding)]!;
      const destination = square * TOKEN_WIDTH + history * TOKEN_CHANNELS;
      const source = square * TOKEN_CHANNELS;
      tokens.set(frame.subarray(source, source + TOKEN_CHANNELS), destination);
    }
  }
  return tokens;
}

/** List the exact Maia-3 move vocabulary order. */
export function allPossibleMoves(): readonly string[] {
  const moves: string[] = [];
  for (let from = 0; from < 64; from += 1) {
    for (let to = 0; to < 64; to += 1) moves.push(squareName(from) + squareName(to));
  }
  for (let fromFile = 0; fromFile < 8; fromFile += 1) {
    for (let toFile = 0; toFile < 8; toFile += 1) {
      for (const promotion of PROMOTION_PIECES) {
        moves.push(`${fileName(fromFile)}7${fileName(toFile)}8${promotion}`);
      }
    }
  }
  return moves;
}

export function legalMovesMask(fen: string, moveIndex: ReadonlyMap<string, number>): Uint8Array {
  const chess = new Chess(fen);
  const mask = new Uint8Array(MOVE_VOCABULARY_SIZE);
  for (const move of chess.moves({ verbose: true })) {
    const uci = `${move.from}${move.to}${move.promotion ?? ""}`;
    const relative = chess.turn() === "w" ? uci : mirrorMove(uci);
    const index = moveIndex.get(relative);
    if (index !== undefined) mask[index] = 1;
  }
  return mask;
}

export function mirrorMove(uci: string): string {
  const promotion = uci.slice(4);
  return mirrorSquare(uci.slice(0, 2)) + mirrorSquare(uci.slice(2, 4)) + promotion;
}

export function maskedSoftmax(
  logits: ArrayLike<number>,
  legalMask: ArrayLike<number>,
): Float32Array {
  if (logits.length !== legalMask.length)
    throw new RangeError("logits and legal mask must have the same length");
  let maximum = Number.NEGATIVE_INFINITY;
  for (let index = 0; index < logits.length; index += 1) {
    if (legalMask[index] !== 0) maximum = Math.max(maximum, logits[index]!);
  }
  if (!Number.isFinite(maximum)) throw new RangeError("legal mask contains no moves");

  const probabilities = new Float32Array(logits.length);
  let total = 0;
  for (let index = 0; index < logits.length; index += 1) {
    if (legalMask[index] !== 0) {
      const probability = Math.exp(logits[index]! - maximum);
      probabilities[index] = probability;
      total += probability;
    }
  }
  for (let index = 0; index < probabilities.length; index += 1) probabilities[index] /= total;
  return probabilities;
}

export function decodeWdl(logits: ArrayLike<number>): Wdl {
  if (logits.length !== 3)
    throw new RangeError("WDL logits must contain loss, draw, and win values");
  const [loss, draw, win] = softmax(logits);
  return { win, draw, loss };
}

/** Match Maia-3's clamped linear interpolation between its two exported Elo vectors. */
export function interpolateElo(elo: number, embeddings: EloEmbeddings): Float32Array {
  if (embeddings.low.length !== 128 || embeddings.high.length !== 128) {
    throw new RangeError("Maia Elo embeddings must each contain 128 values");
  }
  const lowWeight = Math.min(5000, Math.max(0, elo)) / 5000;
  const embedding = new Float32Array(128);
  for (let index = 0; index < embedding.length; index += 1) {
    embedding[index] =
      lowWeight * embeddings.low[index]! + (1 - lowWeight) * embeddings.high[index]!;
  }
  return embedding;
}

export class MaiaSession {
  private readonly session: ort.InferenceSession;

  private constructor(session: ort.InferenceSession) {
    this.session = session;
  }

  static async create(
    model: string | Uint8Array | ArrayBuffer,
    executionProvider: MaiaExecutionProvider = "wasm",
  ): Promise<MaiaSession> {
    const options = { executionProviders: [executionProvider] };
    if (typeof model === "string") {
      return new MaiaSession(await ort.InferenceSession.create(model, options));
    }
    if (model instanceof Uint8Array) {
      return new MaiaSession(await ort.InferenceSession.create(model, options));
    }
    return new MaiaSession(await ort.InferenceSession.create(model, options));
  }

  async run(tokens: Uint8Array, selfEmb: Float32Array, oppoEmb: Float32Array): Promise<MaiaOutput> {
    if (tokens.length % (64 * TOKEN_WIDTH) !== 0)
      throw new RangeError("tokens must contain whole board batches");
    const batch = tokens.length / (64 * TOKEN_WIDTH);
    if (selfEmb.length !== batch * 128 || oppoEmb.length !== batch * 128) {
      throw new RangeError("Elo embeddings must have 128 values per token batch");
    }
    const outputs = await this.session.run({
      tokens: new ort.Tensor("float32", Float32Array.from(tokens), [batch, 64, TOKEN_WIDTH]),
      self_emb: new ort.Tensor("float32", selfEmb, [batch, 128]),
      oppo_emb: new ort.Tensor("float32", oppoEmb, [batch, 128]),
    });
    return {
      moveLogits: outputs.logits_move.data as Float32Array,
      valueLogits: outputs.logits_value.data as Float32Array,
      ponderLogits: outputs.logits_ponder.data as Float32Array,
    };
  }
}

function softmax(logits: ArrayLike<number>): [number, number, number] {
  const maximum = Math.max(...Array.from(logits));
  const values = Array.from(logits, (value) => Math.exp(value - maximum));
  const total = values[0]! + values[1]! + values[2]!;
  return [values[0]! / total, values[1]! / total, values[2]! / total];
}

function opposite(color: Color): Color {
  return color === "w" ? "b" : "w";
}

function mirrorSquare(square: string): string {
  return `${square[0]}${9 - Number(square[1])}`;
}

function fileName(file: number): string {
  return String.fromCharCode("a".charCodeAt(0) + file);
}

function squareName(square: number): string {
  return `${fileName(square % 8)}${Math.floor(square / 8) + 1}`;
}
