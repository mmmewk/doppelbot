"""Fetch and normalize a user's games from Lichess and chess.com, with a local JSONL cache."""

from __future__ import annotations

import io
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import chess
import chess.pgn
import requests

USER_AGENT = "doppelbot-research (github.com/mmmewk/doppelbot)"
DATA_DIR = Path(__file__).resolve().parents[2] / "data"
SPEEDS = ("blitz", "rapid", "classical")


@dataclass
class Game:
    site: str  # "lichess" | "chesscom"
    id: str
    played_at: int  # unix seconds, end of game
    speed: str  # "blitz" | "rapid" | "classical"
    rated: bool
    base: int  # seconds
    inc: int  # seconds
    white: str  # lowercase username
    black: str
    white_elo: int
    black_elo: int
    moves: list[str]  # UCI
    clocks: list[float] = field(default_factory=list)  # seconds left after each ply; empty if unknown


def _session() -> requests.Session:
    s = requests.Session()
    s.headers["User-Agent"] = USER_AGENT
    return s


def _get(s: requests.Session, url: str, **kwargs) -> requests.Response:
    while True:
        r = s.get(url, timeout=60, **kwargs)
        if r.status_code == 429:
            time.sleep(60)
            continue
        r.raise_for_status()
        return r


def fetch_lichess(username: str, max_games: int) -> list[Game]:
    s = _session()
    r = _get(
        s,
        f"https://lichess.org/api/games/user/{username}",
        params={
            "max": max_games,
            "rated": "true",
            "perfType": ",".join(SPEEDS),
            "clocks": "true",
            "moves": "true",
            "opening": "false",
            "sort": "dateDesc",
        },
        headers={"Accept": "application/x-ndjson"},
        stream=True,
    )
    games = []
    for line in r.iter_lines():
        if not line:
            continue
        g = json.loads(line)
        if g.get("variant") != "standard" or "clock" not in g or not g.get("moves"):
            continue
        white, black = g["players"]["white"], g["players"]["black"]
        if "user" not in white or "user" not in black or "rating" not in white or "rating" not in black:
            continue  # AI or anonymous opponent
        board = chess.Board()
        uci = []
        for san in g["moves"].split():
            move = board.parse_san(san)
            uci.append(move.uci())
            board.push(move)
        # Lichess sends one clock per ply (time left after it), sometimes plus a trailing entry.
        clocks = [c / 100 for c in g.get("clocks", [])]
        clocks = clocks[: len(uci)] if len(clocks) >= len(uci) else []
        games.append(
            Game(
                site="lichess",
                id=g["id"],
                played_at=g["lastMoveAt"] // 1000,
                speed=g["speed"],
                rated=g["rated"],
                base=g["clock"]["initial"],
                inc=g["clock"]["increment"],
                white=white["user"]["id"],
                black=black["user"]["id"],
                white_elo=white["rating"],
                black_elo=black["rating"],
                moves=uci,
                clocks=clocks,
            )
        )
    return games


def _parse_chesscom_time_control(tc: str) -> tuple[int, int] | None:
    if "/" in tc:  # daily
        return None
    base, _, inc = tc.partition("+")
    return int(base), int(inc or 0)


def fetch_chesscom(username: str, max_games: int) -> list[Game]:
    s = _session()
    archives = _get(s, f"https://api.chess.com/pub/player/{username}/games/archives").json()["archives"]
    games: list[Game] = []
    for archive in reversed(archives):
        month = _get(s, archive).json()["games"]
        for g in sorted(month, key=lambda g: g["end_time"], reverse=True):
            if g.get("rules") != "chess" or g.get("time_class") not in SPEEDS or not g.get("rated"):
                continue
            tc = _parse_chesscom_time_control(g["time_control"])
            if tc is None or "pgn" not in g:
                continue
            pgn = chess.pgn.read_game(io.StringIO(g["pgn"]))
            if pgn is None or "FEN" in pgn.headers:
                continue
            moves, clocks = [], []
            for node in pgn.mainline():
                moves.append(node.move.uci())
                clock = node.clock()
                if clock is not None:
                    clocks.append(clock)
            if not moves:
                continue
            games.append(
                Game(
                    site="chesscom",
                    id=g["url"].rsplit("/", 1)[-1],
                    played_at=g["end_time"],
                    speed=g["time_class"],
                    rated=True,
                    base=tc[0],
                    inc=tc[1],
                    white=g["white"]["username"].lower(),
                    black=g["black"]["username"].lower(),
                    white_elo=g["white"]["rating"],
                    black_elo=g["black"]["rating"],
                    moves=moves,
                    clocks=clocks if len(clocks) == len(moves) else [],
                )
            )
            if len(games) >= max_games:
                return games
    return games


def load_games(site: str, username: str, max_games: int = 300, refresh: bool = False) -> list[Game]:
    """Return the user's most recent rated blitz/rapid/classical games, newest first."""
    path = DATA_DIR / "games" / site / f"{username.lower()}.jsonl"
    if path.exists() and not refresh:
        with path.open() as f:
            return [Game(**json.loads(line)) for line in f]
    fetch = {"lichess": fetch_lichess, "chesscom": fetch_chesscom}[site]
    games = fetch(username, max_games)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for game in games:
            f.write(json.dumps(asdict(game)) + "\n")
    return games
