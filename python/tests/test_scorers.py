"""Batched scorers must agree with each library's own single-position inference."""

import math

import chess
import pytest
from maia2 import inference
from maia3.uci import Maia3UCIEngine

from doppelbot.games import Game
from doppelbot.positions import user_positions
from doppelbot.scorers import Maia2Scorer, Maia3Scorer

# A short game with castling, captures, a black-to-move position and an en-passant capture.
MOVES = "e2e4 c7c5 g1f3 d7d6 d2d4 c5d4 f3d4 g8f6 b1c3 a7a6 f1e2 e7e5 d4b3 f8e7 e1g1 e8g8 f2f4 b7b5 f4f5 b5b4 c3d5 f6e4 a2a4 b4a3".split()
GAME = Game(
    site="test", id="g1", played_at=0, speed="rapid", rated=True, base=600, inc=0,
    white="w", black="b", white_elo=1250, black_elo=1430, moves=MOVES, clocks=[],
)


@pytest.fixture(scope="module")
def positions():
    return user_positions([GAME], "w") + user_positions([GAME], "b")


def test_maia2_matches_inference_each(positions):
    scorer = Maia2Scorer(device="cpu")
    logp, _ = scorer.score(scorer.encode(positions))
    prepared = inference.prepare()
    for p, lp in zip(positions, logp.tolist()):
        probs, _ = inference.inference_each(scorer.models["rapid"], prepared, p.fens[-1], p.self_elo, p.oppo_elo)
        assert math.exp(lp) == pytest.approx(probs[p.move], abs=1e-3)


@pytest.mark.parametrize("real_history", [False, True])
def test_maia3_matches_engine(positions, real_history):
    scorer = Maia3Scorer("maia3-5m", real_history=real_history, device="cpu")
    logp, _ = scorer.score(scorer.encode(positions))
    engine = Maia3UCIEngine(scorer.cfg)
    engine.model = scorer.model
    engine.multipv = 20  # the engine's cap; positions whose move falls outside it are skipped
    engine.cfg.use_uci_history = real_history
    compared = 0
    for p, lp in zip(positions, logp.tolist()):
        engine.cmd_position("position startpos moves " + " ".join(MOVES[: p.ply]))
        engine.self_elo, engine.oppo_elo = p.self_elo, p.oppo_elo
        _, top = engine.score_moves()
        probs = {m["move"].uci(): m["policy"] for m in top}
        if p.move in probs:
            assert math.exp(lp) == pytest.approx(probs[p.move], abs=1e-5)
            compared += 1
    assert compared >= len(positions) * 0.8
