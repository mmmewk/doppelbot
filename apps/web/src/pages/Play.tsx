import { useEffect, useRef, useState, type CSSProperties } from "react";
import { Chess, type Color, type Square } from "chess.js";
import { Chessboard } from "react-chessboard";
import { DEFAULT_CONFIG } from "@doppelbot/config";
import type { BotMove, BotReply, BotRequest } from "../bot.ts";
import {
  copyGame,
  isPromotion,
  outcome,
  recentFens,
  targets,
  withMove,
  type UciMove,
} from "../game.ts";
import { displayName, useProfile, type Profile } from "../profile.tsx";
import {
  SITE_NAMES,
  lichessEquivalent,
  lookupProfile,
  pickRating,
  type SiteRating,
} from "../ratings.ts";

const FALLBACK_RATING = 1500;
const MAX_RATING = DEFAULT_CONFIG.play.maxBotRating;
// Bot moves wait at least this long, so replies don't appear instantly.
const MIN_DELAY_MS = 300;
const MAX_DELAY_MS = 800;

type SidePreference = Color | "random";

export default function Play() {
  const { profile, openModal } = useProfile();
  if (!profile) {
    return (
      <main className="play-empty">
        <h1>Play Maia</h1>
        <p>Add your chess.com or Lichess username so Maia can play at your level.</p>
        <button className="button" onClick={openModal}>
          Let's get started
        </button>
      </main>
    );
  }
  // Remount when the usernames change, so the rating is looked up again.
  return <Game key={`${profile.lichess}|${profile.chesscom}`} profile={profile} />;
}

function Game({ profile }: { profile: Profile }) {
  const { openModal } = useProfile();
  const [lookup, setLookup] = useState<"loading" | SiteRating | "failed">("loading");
  const [rating, setRating] = useState(FALLBACK_RATING);
  const [stretch, setStretch] = useState(DEFAULT_CONFIG.play.stretchRating);
  const [sidePreference, setSidePreference] = useState<SidePreference>("w");
  const [color, setColor] = useState<Color>("w");
  const [game, setGame] = useState(() => new Chess());
  const [resigned, setResigned] = useState<Color>();
  const [selected, setSelected] = useState<string>();
  const [promotion, setPromotion] = useState<UciMove>();
  const [topMove, setTopMove] = useState(false);
  const [backend, setBackend] = useState("loading");
  const [error, setError] = useState<string>();
  const [debug, setDebug] = useState<BotMove>();

  const worker = useRef<Worker>(undefined);
  const request = useRef({ id: 0, sentAt: 0 });

  useEffect(() => {
    let cancelled = false;
    void lookupProfile(profile).then((ratings) => {
      if (cancelled) return;
      const best = pickRating(ratings);
      setLookup(best ?? "failed");
      if (best) setRating(lichessEquivalent(best));
    });
    return () => {
      cancelled = true;
    };
  }, [profile]);

  useEffect(() => {
    const next = new Worker(new URL("../maia.worker.ts", import.meta.url), { type: "module" });
    worker.current = next;
    next.onerror = (event) => setError(event.message || "Maia worker failed to start");
    next.onmessage = ({ data }: MessageEvent<BotReply>) => {
      if ("backend" in data) return setBackend(data.backend);
      if (data.id !== request.current.id) return;
      if ("error" in data) return setError(data.error);
      const elapsed = performance.now() - request.current.sentAt;
      const delay = MIN_DELAY_MS + Math.random() * (MAX_DELAY_MS - MIN_DELAY_MS);
      setTimeout(
        () => {
          if (data.id !== request.current.id) return;
          setError(undefined);
          setDebug(data);
          setGame((current) => withMove(current, data.move));
        },
        Math.max(0, delay - elapsed),
      );
    };
    return () => next.terminate();
  }, []);

  const botRating = Math.min(rating + stretch, MAX_RATING);
  const over = outcome(game, resigned);
  const ready = lookup !== "loading";
  const botTurn = ready && !over && game.turn() !== color;

  // Ask the bot for a move whenever it's its turn. Cleanup invalidates the request, so a reset or
  // undo while it's thinking drops the stale reply.
  useEffect(() => {
    if (!botTurn) return;
    request.current = { id: request.current.id + 1, sentAt: performance.now() };
    worker.current?.postMessage({
      id: request.current.id,
      fens: recentFens(game),
      selfElo: botRating,
      oppoElo: Math.min(rating, MAX_RATING),
      temperature: topMove ? 0 : DEFAULT_CONFIG.bots.speed.temperature,
    } satisfies BotRequest);
    return () => {
      request.current.id += 1;
    };
    // Rating, stretch and the toggle apply from the next bot move; they don't restart this one.
  }, [botTurn, game]);

  const yourTurn = ready && !over && game.turn() === color;

  function tryMove(move: UciMove): boolean {
    if (!yourTurn) return false;
    if (!targets(game, move.from).includes(move.to)) return false;
    setSelected(undefined);
    if (!move.promotion && isPromotion(game, move)) {
      setPromotion(move);
      return false;
    }
    setGame(withMove(game, move));
    return true;
  }

  function clickSquare(square: string) {
    if (promotion) return;
    if (selected && tryMove({ from: selected, to: square })) return;
    const piece = game.get(square as Square);
    setSelected(yourTurn && piece?.color === color && square !== selected ? square : undefined);
  }

  function newGame() {
    const side = sidePreference === "random" ? (Math.random() < 0.5 ? "w" : "b") : sidePreference;
    setColor(side);
    setGame(new Chess());
    setResigned(undefined);
    setSelected(undefined);
    setPromotion(undefined);
    setDebug(undefined);
    setError(undefined);
  }

  function undo() {
    // Take back your last move, and the bot's reply if it has made one.
    const next = copyGame(game);
    if (next.turn() === color) next.undo();
    next.undo();
    setGame(next);
    setResigned(undefined);
    setSelected(undefined);
    setDebug(undefined);
  }

  const history = game.history({ verbose: true });
  const last = history.at(-1);
  const squareStyles: Record<string, CSSProperties> = {};
  if (last) {
    squareStyles[last.from] = { background: "var(--last-move)" };
    squareStyles[last.to] = { background: "var(--last-move)" };
  }
  if (game.inCheck()) {
    const king = game
      .board()
      .flat()
      .find((piece) => piece?.type === "k" && piece.color === game.turn());
    if (king) squareStyles[king.square] = { background: "var(--check)" };
  }
  if (selected) {
    squareStyles[selected] = { background: "var(--selected)" };
    for (const target of targets(game, selected)) {
      squareStyles[target] = {
        ...squareStyles[target],
        backgroundImage: game.get(target as Square)
          ? "radial-gradient(circle, transparent 58%, var(--target) 60%)"
          : "radial-gradient(circle, var(--target) 22%, transparent 24%)",
      };
    }
  }

  const converted = typeof lookup === "object" && lookup.site === "chesscom";
  const ratingLabel =
    lookup === "loading"
      ? "Looking up your rating…"
      : lookup === "failed"
        ? "Couldn't find a blitz or rapid rating. Set one below."
        : lookup.site === "lichess"
          ? `Lichess ${lookup.timeControl} ${lookup.rating}, ${lookup.games} games`
          : `${SITE_NAMES[lookup.site]} ${lookup.timeControl} ${lookup.rating} (${lookup.games} games) ≈ Lichess ${lichessEquivalent(lookup)}`;
  const wdl = debug?.wdl;
  const status = over
    ? `${over.winner ? (over.winner === color ? "You win" : "Maia wins") : "Draw"} by ${over.reason}.`
    : !ready
      ? "Getting ready…"
      : botTurn
        ? "Maia is thinking…"
        : "Your move.";

  return (
    <main className="play">
      <header className="play-header">
        <h1>Ready, {displayName(profile)}?</h1>
        <button className="link-button" onClick={openModal}>
          Edit profile
        </button>
      </header>

      <div className="play-layout">
        <div className="board-wrap">
          <Chessboard
            options={{
              id: "play",
              position: game.fen(),
              boardOrientation: color === "w" ? "white" : "black",
              squareStyles,
              canDragPiece: ({ piece }) => yourTurn && piece.pieceType[0] === color,
              onPieceDrag: ({ square }) => setSelected(square ?? undefined),
              onPieceDrop: ({ sourceSquare, targetSquare }) =>
                targetSquare !== null && tryMove({ from: sourceSquare, to: targetSquare }),
              onSquareClick: ({ square }) => clickSquare(square),
            }}
          />
          {promotion && (
            <div className="promotion" role="dialog" aria-label="Choose a promotion piece">
              {(["q", "r", "b", "n"] as const).map((piece) => (
                <button
                  key={piece}
                  onClick={() => {
                    setGame(withMove(game, { ...promotion, promotion: piece }));
                    setPromotion(undefined);
                  }}
                >
                  {{ q: "Queen", r: "Rook", b: "Bishop", n: "Knight" }[piece]}
                </button>
              ))}
              <button className="secondary" onClick={() => setPromotion(undefined)}>
                Cancel
              </button>
            </div>
          )}
        </div>

        <aside className="panel">
          <p className="status" role="status">
            {status}
          </p>
          {error && <p role="alert">Maia error: {error}</p>}
          <div className="row">
            <button className="button secondary" onClick={undo} disabled={!yourTurn && !over}>
              Undo
            </button>
            <button
              className="button secondary"
              onClick={() => setResigned(color)}
              disabled={Boolean(over) || history.length === 0}
            >
              Resign
            </button>
            <button className="button" onClick={newGame}>
              New game
            </button>
          </div>

          <fieldset>
            <legend>Game setup</legend>
            <p className="muted">{ratingLabel}</p>
            <label>
              Your rating (Lichess scale)
              <input
                type="number"
                min={0}
                max={3000}
                value={rating}
                onChange={(event) => setRating(Number(event.target.value))}
                aria-describedby={converted ? "rating-help" : undefined}
              />
              {converted && (
                <span id="rating-help" className="muted">
                  Your chess.com rating is mapped to its Lichess equivalent, because Maia was
                  trained on Lichess games and uses Lichess ratings.{" "}
                  <a
                    href="https://chessgoals.com/rating-comparison/"
                    target="_blank"
                    rel="noreferrer"
                  >
                    How it's mapped
                  </a>
                </span>
              )}
            </label>
            <label>
              Stretch: {stretch >= 0 ? "+" : ""}
              {stretch} (Maia plays at {botRating}
              {rating + stretch > MAX_RATING ? ", its maximum" : ""})
              <input
                type="range"
                min={-300}
                max={500}
                step={25}
                value={stretch}
                onChange={(event) => setStretch(Number(event.target.value))}
              />
            </label>
            <label>
              Play as
              <select
                value={sidePreference}
                onChange={(event) => setSidePreference(event.target.value as SidePreference)}
              >
                <option value="w">White</option>
                <option value="b">Black</option>
                <option value="random">Random</option>
              </select>
            </label>
            <p className="muted">Side changes apply when you start a new game.</p>
          </fieldset>

          <details className="debug">
            <summary>Debug</summary>
            <dl>
              <dt>Model</dt>
              <dd>
                {DEFAULT_CONFIG.model.id} on {backend}
              </dd>
              <dt>Cross-origin isolated</dt>
              <dd>{window.crossOriginIsolated ? "yes" : "no"}</dd>
              {debug && wdl && (
                <>
                  <dt>Last inference</dt>
                  <dd>{debug.ms.toFixed(0)} ms</dd>
                  <dt>Maia's view before its move (win / draw / loss)</dt>
                  <dd>
                    {(wdl.win * 100).toFixed(0)}% / {(wdl.draw * 100).toFixed(0)}% /{" "}
                    {(wdl.loss * 100).toFixed(0)}%
                  </dd>
                  <dt>Top moves (played {debug.move})</dt>
                  <dd>
                    <ol>
                      {debug.top.map((move) => (
                        <li key={move.move}>
                          {move.move} {(move.probability * 100).toFixed(1)}%
                        </li>
                      ))}
                    </ol>
                  </dd>
                </>
              )}
            </dl>
            <label className="inline">
              <input
                type="checkbox"
                checked={topMove}
                onChange={(event) => setTopMove(event.target.checked)}
              />
              Always play the top move
            </label>
          </details>
        </aside>
      </div>
    </main>
  );
}
