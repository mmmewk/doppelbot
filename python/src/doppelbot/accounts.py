"""Sample real 800-1800 accounts from Lichess and chess.com for the Phase 1 experiments.

Lichess candidates come from the start of a monthly open-database dump; chess.com candidates
from country player lists. Both are then filtered to active accounts with enough rated games.
Writes data/accounts.json.
"""

from __future__ import annotations

import io
import json
import random
import re
import time

import requests
import zstandard

from doppelbot.games import DATA_DIR, USER_AGENT

BANDS = [(800, 1000), (1000, 1200), (1200, 1400), (1400, 1600), (1600, 1800)]
PER_BAND = 2
MIN_GAMES = 300
ACTIVE_DAYS = 30
SEED = 7

HEADER = re.compile(r'^\[(\w+) "(.*)"\]$')


def band_of(rating: int) -> tuple[int, int] | None:
    return next((b for b in BANDS if b[0] <= rating < b[1]), None)


def lichess_candidates(month: str, max_games: int = 150_000) -> dict[str, tuple[str, int]]:
    """Players seen in rated blitz/rapid games at the start of a monthly dump: id -> (speed, rating)."""
    url = f"https://database.lichess.org/standard/lichess_db_standard_rated_{month}.pgn.zst"
    r = requests.get(url, stream=True, timeout=60, headers={"User-Agent": USER_AGENT})
    r.raise_for_status()
    reader = io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(r.raw), encoding="utf-8")
    seen: dict[str, tuple[str, int]] = {}
    tags: dict[str, str] = {}
    games = 0
    for line in reader:
        m = HEADER.match(line.strip())
        if m:
            tags[m.group(1)] = m.group(2)
            continue
        if not tags or "WhiteElo" not in tags:
            continue
        event = tags.get("Event", "")
        speed = "blitz" if "Blitz" in event else "rapid" if "Rapid" in event else None
        if speed:
            for color in ("White", "Black"):
                elo = tags.get(f"{color}Elo", "?")
                if elo.isdigit():
                    seen[tags[color].lower()] = (speed, int(elo))
        tags = {}
        games += 1
        if games >= max_games:
            break
    r.close()
    return seen


def pick_lichess(rng: random.Random, month: str) -> list[dict]:
    candidates = lichess_candidates(month)
    by_band: dict[tuple[int, int], list[str]] = {b: [] for b in BANDS}
    for user, (_, rating) in candidates.items():
        band = band_of(rating)
        if band:
            by_band[band].append(user)
    picked = []
    now_ms = time.time() * 1000
    for band, users in by_band.items():
        rng.shuffle(users)
        chosen = 0
        for i in range(0, len(users), 300):
            batch = users[i : i + 300]
            r = requests.post(
                "https://lichess.org/api/users",
                data=",".join(batch),
                timeout=60,
                headers={"User-Agent": USER_AGENT, "Content-Type": "text/plain"},
            )
            r.raise_for_status()
            for u in r.json():
                if u.get("disabled") or u.get("tosViolation") or u.get("title") == "BOT":
                    continue
                if now_ms - u.get("seenAt", 0) > ACTIVE_DAYS * 86_400_000:
                    continue
                perfs = u.get("perfs", {})
                speed = max(("blitz", "rapid"), key=lambda s: perfs.get(s, {}).get("games", 0))
                perf = perfs.get(speed, {})
                if perf.get("games", 0) < MIN_GAMES or perf.get("prov") or band_of(perf.get("rating", 0)) != band:
                    continue
                picked.append({"site": "lichess", "username": u["id"], "speed": speed, "rating": perf["rating"]})
                chosen += 1
                if chosen >= PER_BAND:
                    break
            if chosen >= PER_BAND:
                break
    return picked


def pick_chesscom(rng: random.Random, countries=("US", "GB", "IN", "CA", "AU", "DE", "PH", "BR")) -> list[dict]:
    s = requests.Session()
    s.headers["User-Agent"] = USER_AGENT
    pool: list[str] = []
    for country in countries:
        r = s.get(f"https://api.chess.com/pub/country/{country}/players", timeout=60)
        if r.ok:
            pool.extend(r.json()["players"])
    rng.shuffle(pool)
    need = {b: PER_BAND for b in BANDS}
    picked = []
    now = time.time()
    for username in pool[:3000]:
        if not any(need.values()):
            break
        r = s.get(f"https://api.chess.com/pub/player/{username}/stats", timeout=60)
        if not r.ok:
            continue
        stats = r.json()
        best = None
        for speed in ("blitz", "rapid"):
            st = stats.get(f"chess_{speed}")
            if not st:
                continue
            rec = st.get("record", {})
            games = rec.get("win", 0) + rec.get("loss", 0) + rec.get("draw", 0)
            if games >= MIN_GAMES and now - st["last"]["date"] < ACTIVE_DAYS * 86_400:
                if best is None or games > best[2]:
                    best = (speed, st["last"]["rating"], games)
        if best is None:
            continue
        band = band_of(best[1])
        if band and need[band] > 0:
            need[band] -= 1
            picked.append({"site": "chesscom", "username": username.lower(), "speed": best[0], "rating": best[1]})
    return picked


def main() -> None:
    rng = random.Random(SEED)
    accounts = pick_lichess(rng, month="2026-09") + pick_chesscom(rng)
    path = DATA_DIR / "accounts.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(accounts, indent=2) + "\n")
    for a in accounts:
        print(a["site"], a["speed"], a["rating"])
    print(f"wrote {len(accounts)} accounts to {path}")


if __name__ == "__main__":
    main()
