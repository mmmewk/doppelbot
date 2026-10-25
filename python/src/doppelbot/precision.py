"""Phase 1, step 6: compare fp16 and int8 Maia-3 exports against fp32.

Every variant runs the same inputs through ONNX Runtime on CPU, and each masked move distribution is compared
with fp32's: top-1 agreement, KL(fp32 || variant), max abs probability difference and WDL difference. Inputs
are the parity fixtures (at their three rating pairs) and a sample of held-out moves from the cached games (at
the ratings recorded in each game). On the held-out moves we also score the move the user actually played.
"""

from __future__ import annotations

import argparse
import base64
import gzip
import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from doppelbot.export import ARTIFACTS_DIR, PRECISIONS
from doppelbot.fixtures import DEFAULT_OUTPUT_PATH as FIXTURES_PATH
from doppelbot.games import DATA_DIR, load_games
from doppelbot.positions import Position, load_defaults, split_by_time, user_positions
from doppelbot.scorers import Maia3Scorer

SEED = 7
BATCH_SIZE = 128


@dataclass
class Inputs:
    tokens: np.ndarray  # float32 [N, 64, 97]
    self_emb: np.ndarray  # float32 [N, 128]
    oppo_emb: np.ndarray  # float32 [N, 128]
    legal: np.ndarray  # bool [N, 4352]
    target: np.ndarray | None = None  # int [N], the move actually played (held-out only)


@dataclass
class Outputs:
    probabilities: np.ndarray  # float64 [N, 4352], masked softmax
    wdl: np.ndarray  # float64 [N, 3], loss/draw/win


def run_onnx(model_path: Path, inputs: Inputs) -> Outputs:
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    probabilities, wdl = [], []
    for start in range(0, len(inputs.tokens), BATCH_SIZE):
        sl = slice(start, start + BATCH_SIZE)
        logits_move, logits_value, _ = session.run(
            None, {"tokens": inputs.tokens[sl], "self_emb": inputs.self_emb[sl], "oppo_emb": inputs.oppo_emb[sl]}
        )
        masked = np.where(inputs.legal[sl], logits_move.astype(np.float64), -np.inf)
        masked -= masked.max(axis=1, keepdims=True)
        exp = np.exp(masked)
        probabilities.append(exp / exp.sum(axis=1, keepdims=True))
        value = np.exp(logits_value.astype(np.float64) - logits_value.max(axis=1, keepdims=True))
        wdl.append(value / value.sum(axis=1, keepdims=True))
    return Outputs(np.concatenate(probabilities), np.concatenate(wdl))


def fixture_inputs(scorer: Maia3Scorer, path: Path = FIXTURES_PATH) -> Inputs:
    """The parity fixtures' positions, repeated at each of their rating pairs."""
    fixtures = json.loads(path.read_text())

    def unpack(value: str) -> np.ndarray:
        return np.frombuffer(base64.b64decode(value), dtype=np.uint8)

    tokens = np.stack([unpack(p["tokens"]).reshape(64, 97) for p in fixtures["positions"]]).astype(np.float32)
    legal = np.stack([unpack(p["legal_mask"]).astype(bool) for p in fixtures["positions"]])
    pairs = [(e["self_elo"], e["oppo_elo"]) for e in fixtures["positions"][0]["expected"]]
    return _inputs(
        scorer,
        np.tile(tokens, (len(pairs), 1, 1)),
        np.tile(legal, (len(pairs), 1)),
        np.repeat([p[0] for p in pairs], len(tokens)),
        np.repeat([p[1] for p in pairs], len(tokens)),
    )


def heldout_inputs(scorer: Maia3Scorer, count: int) -> Inputs:
    """A seeded sample of held-out moves across all cached accounts, at the ratings recorded in each game."""
    defaults = load_defaults()
    accounts = json.loads((DATA_DIR / "accounts.json").read_text())
    heldout = []
    for a in accounts:
        positions = user_positions(load_games(a["site"], a["username"]), a["username"], defaults["filter"])
        heldout += split_by_time(positions, defaults["fit"]["holdoutFraction"])[1]
    return position_inputs(scorer, random.Random(SEED).sample(heldout, min(count, len(heldout))))


def position_inputs(scorer: Maia3Scorer, positions: list[Position]) -> Inputs:
    """Encode positions at the ratings they carry, keeping the move played as the target."""
    enc = scorer.encode(positions)
    inputs = _inputs(scorer, enc.inputs.numpy().astype(np.float32), enc.legal.numpy(), enc.self_elo.numpy(), enc.oppo_elo.numpy())
    inputs.target = enc.target.numpy()
    return inputs


def _inputs(scorer: Maia3Scorer, tokens, legal, self_elo, oppo_elo) -> Inputs:
    with torch.no_grad():
        self_emb = scorer.model.interpolate_elo(torch.tensor(self_elo, dtype=torch.float32)).numpy()
        oppo_emb = scorer.model.interpolate_elo(torch.tensor(oppo_elo, dtype=torch.float32)).numpy()
    return Inputs(tokens, self_emb, oppo_emb, legal)


def compare(reference: Outputs, variant: Outputs, target: np.ndarray | None = None) -> dict:
    """Metrics for one variant against fp32; per-position values are macro-averaged over positions."""
    p, q = reference.probabilities, variant.probabilities
    with np.errstate(divide="ignore", invalid="ignore"):
        kl = np.where(p > 0, p * (np.log(p) - np.log(np.maximum(q, 1e-300))), 0.0).sum(axis=1)
    wdl_diff = np.abs(reference.wdl - variant.wdl)
    result = {
        "top1_agreement": float((p.argmax(1) == q.argmax(1)).mean()),
        "mean_kl": float(kl.mean()),
        "max_kl": float(kl.max()),
        "max_abs_prob_diff": float(np.abs(p - q).max()),
        "mean_abs_prob_diff": float(np.abs(p - q).sum(axis=1).mean()),
        "max_abs_wdl_diff": float(wdl_diff.max()),
        "mean_abs_wdl_diff": float(wdl_diff.mean()),
    }
    if target is not None:
        rows = np.arange(len(target))
        for name, probs in (("fp32", p), ("variant", q)):
            result[f"{name}_logp"] = float(np.log(np.maximum(probs[rows, target], 1e-300)).mean())
            result[f"{name}_top1"] = float((probs.argmax(1) == target).mean())
    return result


def file_sizes(model_path: Path) -> tuple[int, int]:
    data = model_path.read_bytes()
    return len(data), len(gzip.compress(data, compresslevel=6))


def run(model: str, heldout_count: int) -> str:
    fp32_dir = ARTIFACTS_DIR / model
    scorer = Maia3Scorer(model, real_history=True, device="cpu")
    sets = {"fixtures": fixture_inputs(scorer), "heldout": heldout_inputs(scorer, heldout_count)}
    reference = {name: run_onnx(fp32_dir / "model.onnx", inputs) for name, inputs in sets.items()}

    size, gz = file_sizes(fp32_dir / "model.onnx")
    lines = [
        f"fp32 {size / 1e6:.1f} MB ({gz / 1e6:.1f} MB gzip); "
        f"{len(sets['fixtures'].tokens)} fixture rows (500 positions x 3 rating pairs), "
        f"{len(sets['heldout'].tokens)} held-out positions",
        "",
        "| Variant | Set | Size (MB) | gzip (MB) | Top-1 agree | Mean KL | Max KL | Max abs prob diff | Max abs WDL diff |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    fidelity = []
    for precision in PRECISIONS:
        path = ARTIFACTS_DIR / f"{model}-{precision}" / "model.onnx"
        size, gz = file_sizes(path)
        for name, inputs in sets.items():
            m = compare(reference[name], run_onnx(path, inputs), inputs.target)
            lines.append(
                f"| {precision} | {name} | {size / 1e6:.1f} | {gz / 1e6:.1f} | {m['top1_agreement']:.2%} | {m['mean_kl']:.2e} "
                f"| {m['max_kl']:.2e} | {m['max_abs_prob_diff']:.3f} | {m['max_abs_wdl_diff']:.3f} |"
            )
            if name == "heldout":
                fidelity.append((precision, m))
    ref = fidelity[0][1]
    lines += [
        "",
        "Held-out moves actually played (mean log-likelihood per move / top-1):",
        "",
        "| Variant | Log-likelihood | Top-1 |",
        "|---|---|---|",
        f"| fp32 | {ref['fp32_logp']:.4f} | {ref['fp32_top1']:.2%} |",
        *(f"| {p} | {m['variant_logp']:.4f} | {m['variant_top1']:.2%} |" for p, m in fidelity),
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare fp16 and int8 Maia-3 exports against fp32")
    parser.add_argument("--model", default="maia3-5m")
    parser.add_argument("--heldout", type=int, default=5000, help="held-out positions to sample")
    args = parser.parse_args()
    print(run(args.model, args.heldout))


if __name__ == "__main__":
    main()
