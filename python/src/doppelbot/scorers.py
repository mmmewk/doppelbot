"""Batched scoring of a user's actual moves under Maia-2 and Maia-3.

Each scorer encodes positions once (board inputs, legal-move mask, index of the move played),
then scores them under any rating offset: log P(played move) and whether it was the top move.
Encodings reuse each library's own functions so results match its inference code.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path

import chess
import torch
from maia2 import model as maia2_model
from maia2.utils import board_to_tensor, create_elo_dict, get_all_possible_moves as maia2_moves, map_to_category
from maia3.dataset import get_historical_tokens, get_legal_moves_mask, tokenize_board
from maia3.uci import Maia3UCIEngine, parse_args
from maia3.utils import mirror_move

from doppelbot.positions import Position

MAIA2_DIR = Path(__file__).resolve().parents[2] / "maia2_models"


def default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass
class Encoded:
    inputs: torch.Tensor  # model-specific board inputs (CPU)
    legal: torch.Tensor  # bool [N, n_moves]
    target: torch.Tensor  # long [N]
    self_elo: torch.Tensor  # float [N]
    oppo_elo: torch.Tensor  # float [N]
    group: list[str]  # per-position routing key (Maia-2 model type); unused by Maia-3


def _relative_uci(board: chess.Board, uci: str) -> str:
    """Moves are indexed from the mover's perspective: mirrored when Black is to move."""
    return uci if board.turn == chess.WHITE else mirror_move(uci)


def _score_logits(logits: torch.Tensor, legal: torch.Tensor, target: torch.Tensor):
    logp = logits.float().masked_fill(~legal, float("-inf")).log_softmax(-1)
    return logp.gather(1, target[:, None]).squeeze(1), logp.argmax(-1) == target


class Maia3Scorer:
    def __init__(self, model: str = "maia3-5m", real_history: bool = False, device: str | None = None):
        self.device = device or default_device()
        self.real_history = real_history
        self.cfg = parse_args(["--model", model, "--device", self.device])
        engine = Maia3UCIEngine(self.cfg)
        engine.ensure_model_loaded()
        self.model = engine.model.eval()
        self.moves = {m: i for i, m in enumerate(engine.all_moves)}

    def encode(self, positions: list[Position]) -> Encoded:
        tokens, legal, target = [], [], []
        frames: dict[str, torch.Tensor] = {}
        for p in positions:
            fens = p.fens if self.real_history else p.fens[-1:]
            history = deque(maxlen=self.cfg.history)
            for fen in fens:
                if fen not in frames:
                    frames[fen] = tokenize_board(chess.Board(fen))
                history.append(frames[fen])
            board = p.board
            tokens.append(get_historical_tokens(history, self.cfg, 0.0, 0.0, 0.0, 0.0).to(torch.uint8))
            legal.append(get_legal_moves_mask(board, self.moves))
            target.append(self.moves[_relative_uci(board, p.move)])
        return Encoded(
            inputs=torch.stack(tokens),
            legal=torch.stack(legal),
            target=torch.tensor(target),
            self_elo=torch.tensor([float(p.self_elo) for p in positions]),
            oppo_elo=torch.tensor([float(p.oppo_elo) for p in positions]),
            group=["maia3"] * len(positions),
        )

    @torch.no_grad()
    def score(self, enc: Encoded, offset: float = 0.0, batch_size: int = 512):
        logps, top1s = [], []
        for i in range(0, len(enc.target), batch_size):
            sl = slice(i, i + batch_size)
            logits, _, _ = self.model(
                enc.inputs[sl].to(self.device, torch.float32),
                (enc.self_elo[sl] + offset).to(self.device),
                (enc.oppo_elo[sl] + offset).to(self.device),
            )
            logp, top1 = _score_logits(logits, enc.legal[sl].to(self.device), enc.target[sl].to(self.device))
            logps.append(logp.cpu())
            top1s.append(top1.cpu())
        return torch.cat(logps), torch.cat(top1s)


class Maia2Scorer:
    """Routes blitz games to the blitz model and rapid/classical games to the rapid model."""

    def __init__(self, device: str | None = None):
        self.device = device or default_device()
        self.models = {
            t: maia2_model.from_pretrained(type=t, device=self.device, save_root=str(MAIA2_DIR)).eval()
            for t in ("blitz", "rapid")
        }
        self.moves = {m: i for i, m in enumerate(maia2_moves())}
        self.elo_dict = create_elo_dict()

    def encode(self, positions: list[Position]) -> Encoded:
        boards, legal, target = [], [], []
        for p in positions:
            board = p.board
            relative = board if board.turn == chess.WHITE else board.mirror()
            boards.append(board_to_tensor(relative))
            mask = torch.zeros(len(self.moves), dtype=torch.bool)
            mask[[self.moves[m.uci()] for m in relative.legal_moves]] = True
            legal.append(mask)
            target.append(self.moves[_relative_uci(board, p.move)])
        return Encoded(
            inputs=torch.stack(boards),
            legal=torch.stack(legal),
            target=torch.tensor(target),
            self_elo=torch.tensor([float(p.self_elo) for p in positions]),
            oppo_elo=torch.tensor([float(p.oppo_elo) for p in positions]),
            group=["blitz" if p.speed == "blitz" else "rapid" for p in positions],
        )

    def _categories(self, elos: torch.Tensor) -> torch.Tensor:
        return torch.tensor([map_to_category(int(e), self.elo_dict) for e in elos.tolist()])

    @torch.no_grad()
    def score(self, enc: Encoded, offset: float = 0.0, batch_size: int = 512):
        logp_all = torch.empty(len(enc.target))
        top1_all = torch.empty(len(enc.target), dtype=torch.bool)
        for group, model in self.models.items():
            idx = torch.tensor([i for i, g in enumerate(enc.group) if g == group], dtype=torch.long)
            for i in range(0, len(idx), batch_size):
                b = idx[i : i + batch_size]
                logits, _, _ = model(
                    enc.inputs[b].to(self.device),
                    self._categories(enc.self_elo[b] + offset).to(self.device),
                    self._categories(enc.oppo_elo[b] + offset).to(self.device),
                )
                logp, top1 = _score_logits(logits, enc.legal[b].to(self.device), enc.target[b].to(self.device))
                logp_all[b] = logp.cpu()
                top1_all[b] = top1.cpu()
        return logp_all, top1_all
