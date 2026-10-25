"""Turn a user's games into the positions where they moved, with the context the models need."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import chess

from doppelbot.games import Game

HISTORY = 8  # positions of history Maia-3 sees, current included
DEFAULTS_PATH = Path(__file__).resolve().parents[3] / "packages" / "config" / "defaults.json"


def load_defaults() -> dict:
    return json.loads(DEFAULTS_PATH.read_text())


@dataclass
class Position:
    game_id: str
    played_at: int
    speed: str
    ply: int
    fens: list[str]  # up to HISTORY positions, oldest first, current last
    move: str  # UCI move the user played
    self_elo: int
    oppo_elo: int
    phase: str  # "opening" | "middlegame" | "endgame"

    @property
    def board(self) -> chess.Board:
        return chess.Board(self.fens[-1])


def _phase(board: chess.Board, ply: int) -> str:
    if ply < 20:
        return "opening"
    pieces = sum(
        len(board.pieces(pt, color))
        for pt in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN)
        for color in (chess.WHITE, chess.BLACK)
    )
    return "endgame" if pieces <= 6 else "middlegame"


def _in_time_trouble(game: Game, ply: int, filt: dict) -> bool:
    """Clock before the move: the mover's clock after their previous move, or the base time."""
    if not game.clocks:
        return False
    before = game.clocks[ply - 2] if ply >= 2 else game.base
    return before < filt["minClockSeconds"] or before < filt["minClockFraction"] * game.base


def user_positions(games: list[Game], username: str, filt: dict | None = None) -> list[Position]:
    filt = filt or load_defaults()["filter"]
    username = username.lower()
    out = []
    for game in games:
        if username not in (game.white, game.black):
            continue
        color = chess.WHITE if game.white == username else chess.BLACK
        self_elo, oppo_elo = (
            (game.white_elo, game.black_elo) if color == chess.WHITE else (game.black_elo, game.white_elo)
        )
        board = chess.Board()
        fens = [board.fen()]
        for ply, uci in enumerate(game.moves):
            if board.turn == color and not _in_time_trouble(game, ply, filt):
                out.append(
                    Position(
                        game_id=game.id,
                        played_at=game.played_at,
                        speed=game.speed,
                        ply=ply,
                        fens=fens[-HISTORY:],
                        move=uci,
                        self_elo=self_elo,
                        oppo_elo=oppo_elo,
                        phase=_phase(board, ply),
                    )
                )
            board.push_uci(uci)
            fens.append(board.fen())
    return out


def split_by_time(positions: list[Position], holdout_fraction: float) -> tuple[list[Position], list[Position]]:
    """Hold out the most recent games (not positions), so no game straddles the split."""
    order = sorted({(p.played_at, p.game_id) for p in positions})
    cut = int(len(order) * (1 - holdout_fraction))
    heldout_games = {gid for _, gid in order[cut:]}
    train = [p for p in positions if p.game_id not in heldout_games]
    heldout = [p for p in positions if p.game_id in heldout_games]
    return train, heldout
