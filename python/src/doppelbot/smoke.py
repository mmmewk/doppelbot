"""Phase 0 smoke test: load Maia-3 and Maia-2 on CPU and print top moves for one position."""

import chess
from maia2 import inference, model as maia2_model
from maia3.uci import Maia3UCIEngine, parse_args

FEN = "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3"
ELO = 1200


def maia3_top_moves(fen: str, elo: int, model: str = "maia3-5m", k: int = 5):
    cfg = parse_args(["--model", model, "--device", "cpu", "--elo", str(elo), "--multipv", str(k)])
    engine = Maia3UCIEngine(cfg)
    engine.ensure_model_loaded()
    engine.board = chess.Board(fen)
    engine._reset_history()
    _, top = engine.score_moves()
    return [(m["move"].uci(), round(m["policy"], 3)) for m in top]


def maia2_top_moves(fen: str, elo: int, k: int = 5):
    m = maia2_model.from_pretrained(type="rapid", device="cpu")
    prepared = inference.prepare()
    probs, _ = inference.inference_each(m, prepared, fen, elo, elo)
    return [(move, round(p, 3)) for move, p in list(probs.items())[:k]]


def main() -> None:
    print(f"{FEN} @ {ELO}")
    print("maia3-5m:", maia3_top_moves(FEN, ELO))
    print("maia2 rapid:", maia2_top_moves(FEN, ELO))


if __name__ == "__main__":
    main()
