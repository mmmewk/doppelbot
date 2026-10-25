"""Generate local Maia-3 parity fixtures from the cached public games."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import chess
import torch

from doppelbot.games import DATA_DIR, Game
from doppelbot.positions import Position
from doppelbot.scorers import Maia3Scorer

DEFAULT_OUTPUT_PATH = DATA_DIR / "fixtures" / "maia3-5m.json"
RATING_PAIRS = ((800, 800), (1200, 1400), (1600, 1800))
COVERAGE = {
    "underpromotion": 20,
    "promotion": 40,
    "en_passant": 80,
    "castling": 60,
    "short_history": 80,
    "black_to_move": 120,
}


@dataclass(frozen=True)
class FixturePosition:
    key: str
    fens: tuple[str, ...]
    tags: frozenset[str]

    @property
    def board(self) -> chess.Board:
        return chess.Board(self.fens[-1])


def _position_key(game: Game, ply: int) -> str:
    return f"{game.site}:{game.id}:{ply}"


def collect_positions(games: list[Game]) -> list[FixturePosition]:
    """Return every game position with the history and special-case tags it needs."""
    positions = []
    for game in games:
        board = chess.Board()
        fens = [board.fen()]
        for ply, uci in enumerate(game.moves):
            move = chess.Move.from_uci(uci)
            tags = set()
            if board.turn == chess.BLACK:
                tags.add("black_to_move")
            if len(fens) < 8:
                tags.add("short_history")
            if board.is_castling(move):
                tags.add("castling")
            if board.is_en_passant(move):
                tags.add("en_passant")
            if move.promotion:
                tags.add("promotion")
                if move.promotion != chess.QUEEN:
                    tags.add("underpromotion")
            positions.append(
                FixturePosition(
                    key=_position_key(game, ply),
                    fens=tuple(fens[-8:]),
                    tags=frozenset(tags),
                )
            )
            board.push(move)
            fens.append(board.fen())
    return positions


def _sample(positions: list[FixturePosition], count: int, salt: str) -> list[FixturePosition]:
    return sorted(
        positions,
        key=lambda position: hashlib.sha256(f"{salt}:{position.key}".encode()).digest(),
    )[:count]


def select_positions(positions: list[FixturePosition], limit: int) -> list[FixturePosition]:
    """Choose a deterministic sample while guaranteeing every edge case is represented."""
    if limit < sum(COVERAGE.values()):
        raise ValueError(f"limit must be at least {sum(COVERAGE.values())}")

    selected: dict[str, FixturePosition] = {}
    for tag, count in COVERAGE.items():
        tagged = [position for position in positions if tag in position.tags]
        if len(tagged) < count:
            raise ValueError(f"need {count} positions tagged {tag}, found {len(tagged)}")
        for position in _sample(tagged, count, tag):
            selected[position.key] = position

    for position in _sample(positions, limit, "general"):
        if len(selected) == limit:
            break
        selected.setdefault(position.key, position)
    if len(selected) != limit:
        raise ValueError(f"need {limit} positions, found {len(selected)}")
    return sorted(selected.values(), key=lambda position: position.key)


def _pack(values: torch.Tensor) -> str:
    return base64.b64encode(values.detach().cpu().to(torch.uint8).numpy().tobytes()).decode()


def _masked_probabilities(logits: torch.Tensor, legal: torch.Tensor) -> torch.Tensor:
    return logits.float().masked_fill(~legal, float("-inf")).softmax(dim=-1)


def generate_fixtures(games: list[Game], output_path: Path, limit: int = 500, top_k: int = 10) -> dict:
    """Write positions and expected Maia outputs for tokenization and ONNX parity tests."""
    positions = select_positions(collect_positions(games), limit)
    scorer = Maia3Scorer("maia3-5m", real_history=True, device="cpu")
    move_names = tuple(scorer.moves)

    tokens, legal = [], []
    for position in positions:
        board = position.board
        encoded = scorer.encode([
            Position(
                game_id=position.key,
                played_at=0,
                speed="rapid",
                ply=0,
                fens=list(position.fens),
                move=next(iter(board.legal_moves)).uci(),
                self_elo=0,
                oppo_elo=0,
                phase="opening",
            )
        ])
        tokens.append(encoded.inputs[0])
        legal.append(encoded.legal[0])
    tokens_tensor = torch.stack(tokens).float()
    legal_tensor = torch.stack(legal)

    expected = [[] for _ in positions]
    with torch.no_grad():
        for self_elo, oppo_elo in RATING_PAIRS:
            batch_size = 128
            for start in range(0, len(positions), batch_size):
                end = min(start + batch_size, len(positions))
                count = end - start
                logits_move, logits_value, _ = scorer.model(
                    tokens_tensor[start:end],
                    torch.full((count,), float(self_elo)),
                    torch.full((count,), float(oppo_elo)),
                )
                probabilities = _masked_probabilities(logits_move, legal_tensor[start:end])
                wdl = logits_value.float().softmax(dim=-1)
                for offset, (probability, value) in enumerate(zip(probabilities, wdl)):
                    values, indices = probability.topk(top_k)
                    expected[start + offset].append(
                        {
                            "self_elo": self_elo,
                            "oppo_elo": oppo_elo,
                            "top_k": [
                                {"move": move_names[index], "probability": float(prob)}
                                for prob, index in zip(values.tolist(), indices.tolist())
                            ],
                            "wdl": {
                                "win": float(value[2]),
                                "draw": float(value[1]),
                                "loss": float(value[0]),
                            },
                        }
                    )

    fixtures = {
        "model": "maia3-5m",
        "tokens": {"shape": [64, 97], "encoding": "base64-uint8"},
        "legal_mask": {"shape": [4352], "encoding": "base64-uint8"},
        "positions": [
            {
                "fens": list(position.fens),
                "tags": sorted(position.tags),
                "tokens": _pack(token),
                "legal_mask": _pack(mask),
                "expected": expected[index],
            }
            for index, (position, token, mask) in enumerate(zip(positions, tokens_tensor, legal_tensor))
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(fixtures, separators=(",", ":")) + "\n")
    return fixtures


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate local Maia-3 parity fixtures")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--limit", type=int, default=500)
    args = parser.parse_args()
    games = [
        Game(**json.loads(line))
        for path in sorted((DATA_DIR / "games").glob("*/*.jsonl"))
        for line in path.read_text().splitlines()
    ]
    fixtures = generate_fixtures(games, args.output, args.limit)
    print(f"wrote {len(fixtures['positions'])} fixtures to {args.output}")


if __name__ == "__main__":
    main()