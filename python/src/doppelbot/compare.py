"""Phase 1, step 1: compare Maia-2 and Maia-3 variants on real 800-1800 accounts.

For each account, games are split by time (most recent games held out). Each model variant is
scored on the held-out moves at:
  L0: the ratings recorded in each game (self and opponent)
  L1: the same ratings plus one offset per account, fitted on the training games

Per-position results go to data/results/phase1/; the aggregate tables are printed as markdown.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict

import torch

from doppelbot.accounts import band_of
from doppelbot.games import DATA_DIR, load_games
from doppelbot.positions import load_defaults, split_by_time, user_positions
from doppelbot.scorers import Maia2Scorer, Maia3Scorer

RESULTS_DIR = DATA_DIR / "results" / "phase1"
COARSE = range(-600, 601, 100)
FINE_STEP = 50
TRAIN_SAMPLE = 1500
SEED = 7

VARIANTS = {
    "maia2": lambda: Maia2Scorer(),
    "maia3-5m": lambda: Maia3Scorer("maia3-5m", real_history=False),
    "maia3-5m+hist": lambda: Maia3Scorer("maia3-5m", real_history=True),
    "maia3-23m": lambda: Maia3Scorer("maia3-23m", real_history=False),
    "maia3-23m+hist": lambda: Maia3Scorer("maia3-23m", real_history=True),
}


def fit_offset(scorer, enc) -> int:
    def mean_logp(offset):
        return scorer.score(enc, offset)[0].mean().item()

    best = max(COARSE, key=mean_logp)
    return max((best - FINE_STEP, best, best + FINE_STEP), key=mean_logp)


def run(variants: list[str]) -> None:
    defaults = load_defaults()
    accounts = json.loads((DATA_DIR / "accounts.json").read_text())
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    data = {}
    for a in accounts:
        positions = user_positions(load_games(a["site"], a["username"]), a["username"], defaults["filter"])
        train, heldout = split_by_time(positions, defaults["fit"]["holdoutFraction"])
        train = random.Random(SEED).sample(train, min(TRAIN_SAMPLE, len(train)))
        data[a["username"]] = (a, train, heldout)

    for name in variants:
        scorer = VARIANTS[name]()
        for i, (a, train, heldout) in enumerate(data.values()):
            path = RESULTS_DIR / f"{name}__{a['site']}__{a['username']}.json"
            if path.exists():
                continue
            offset = fit_offset(scorer, scorer.encode(train))
            enc = scorer.encode(heldout)
            l0_logp, l0_top1 = scorer.score(enc, 0)
            l1_logp, l1_top1 = scorer.score(enc, offset)
            path.write_text(
                json.dumps(
                    {
                        "variant": name,
                        "site": a["site"],
                        "rating": a["rating"],
                        "offset": offset,
                        "n_train": len(train),
                        "positions": [
                            {"speed": p.speed, "phase": p.phase, "l0": [a0, b0], "l1": [a1, b1]}
                            for p, a0, b0, a1, b1 in zip(
                                heldout, l0_logp.tolist(), l0_top1.tolist(), l1_logp.tolist(), l1_top1.tolist()
                            )
                        ],
                    }
                )
            )
            print(f"{name} [{i + 1}/{len(data)}] {a['site']} {a['rating']}: offset {offset:+d}", flush=True)
        del scorer
        torch.mps.empty_cache() if torch.backends.mps.is_available() else None


def summarize() -> str:
    """Macro averages: each account counts equally. logp is mean nats per move (higher is better)."""
    rows = defaultdict(lambda: defaultdict(list))  # (variant, level) -> slice -> per-account means
    offsets = defaultdict(list)
    for path in sorted(RESULTS_DIR.glob("*.json")):
        r = json.loads(path.read_text())
        offsets[r["variant"]].append(r["offset"])
        band = band_of(r["rating"])
        for level in ("l0", "l1"):
            groups = defaultdict(list)
            for p in r["positions"]:
                for key in ("all", f"site:{r['site']}", f"band:{band[0]}-{band[1]}", f"phase:{p['phase']}", f"speed:{p['speed']}"):
                    groups[key].append(p[level])
            for key, vals in groups.items():
                logp = sum(v[0] for v in vals) / len(vals)
                top1 = sum(v[1] for v in vals) / len(vals)
                rows[(r["variant"], level)][key].append((logp, top1, len(vals)))

    keys = sorted({k for slices in rows.values() for k in slices}, key=lambda k: (k != "all", k))
    variants = sorted({v for v, _ in rows}, key=list(VARIANTS).index)
    lines = ["| Variant | Level | " + " | ".join(keys) + " |", "|---|---|" + "---|" * len(keys)]
    for v in variants:
        for level in ("l0", "l1"):
            cells = []
            for k in keys:
                vals = rows[(v, level)].get(k, [])
                if not vals:
                    cells.append("–")
                    continue
                logp = sum(x[0] for x in vals) / len(vals)
                top1 = sum(x[1] for x in vals) / len(vals)
                cells.append(f"{logp:.3f} / {top1:.1%}")
            lines.append(f"| {v} | {level.upper()} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("Cells: mean log-likelihood per move (nats, higher is better) / top-1 accuracy; macro-averaged over accounts.")
    lines.append("")
    for v in variants:
        o = offsets[v]
        lines.append(f"- {v} L1 offsets: mean {sum(o) / len(o):+.0f}, range {min(o):+d} to {max(o):+d} (n={len(o)})")
    n_moves = sum(x[2] for x in rows[(variants[0], "l0")]["all"])
    lines.append(f"- held-out moves per variant: {n_moves} across {len(rows[(variants[0], 'l0')]['all'])} accounts")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", nargs="*", default=list(VARIANTS))
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()
    if not args.summary_only:
        run(args.variants)
    print(summarize())


if __name__ == "__main__":
    main()
