import { useEffect, useRef, useState } from "react";
import { Chess } from "chess.js";
import { Chessboard } from "react-chessboard";
import { DEFAULT_CONFIG } from "@doppelbot/config";
import { decodeWdl } from "@doppelbot/maia";

/** The last 8 positions (oldest first) Maia sees as history, rebuilt from the game's move list. */
function recentFens(game: Chess): string[] {
  const moves = game.history({ verbose: true });
  return [moves[0]?.before ?? game.fen(), ...moves.map((move) => move.after)].slice(-8);
}

function App() {
  const game = useRef(new Chess());
  const worker = useRef<Worker | undefined>(undefined);
  const [fen, setFen] = useState(game.current.fen());
  const [fens, setFens] = useState([fen]);
  const [rating, setRating] = useState(1200);
  const [stretch, setStretch] = useState(DEFAULT_CONFIG.play.stretchRating);
  const [side, setSide] = useState<"w" | "b">("w");
  const [thinking, setThinking] = useState(false);
  const [backend, setBackend] = useState("loading");
  const [error, setError] = useState<string>();
  const [debug, setDebug] = useState<{
    top: { move: string; probability: number }[];
    logits: number[];
    ms: number;
  }>();
  useEffect(() => {
    const next = new Worker(new URL("./maia.worker.ts", import.meta.url), { type: "module" });
    worker.current = next;
    next.onerror = (event) => {
      setThinking(false);
      setError(event.message || "Maia worker failed to start");
    };
    next.onmessage = ({ data }) => {
      if (data.type === "backend") return setBackend(data.backend);
      setThinking(false);
      setError(data.error);
      if (!data.error) {
        game.current.move({
          from: data.move.slice(0, 2),
          to: data.move.slice(2, 4),
          promotion: data.move[4],
        });
        const board = game.current.fen();
        setFen(board);
        setFens((history) => [...history, board].slice(-8));
        setDebug(data);
      }
    };
    return () => next.terminate();
  }, []);
  function bot(history = fens) {
    setThinking(true);
    worker.current?.postMessage({
      fens: history,
      selfElo: rating + stretch,
      oppoElo: rating,
      temperature: DEFAULT_CONFIG.bots.speed.temperature,
    });
  }
  function drop({
    sourceSquare,
    targetSquare,
  }: {
    sourceSquare: string;
    targetSquare: string | null;
  }) {
    if (targetSquare === null || thinking || game.current.turn() !== side) return false;
    try {
      game.current.move({ from: sourceSquare, to: targetSquare, promotion: "q" });
      const board = game.current.fen();
      const history = [...fens, board].slice(-8);
      setFen(board);
      setFens(history);
      bot(history);
      return true;
    } catch {
      return false;
    }
  }
  function reset() {
    game.current = new Chess();
    const board = game.current.fen();
    setFen(board);
    setFens([board]);
    setDebug(undefined);
    if (side === "b") bot([board]);
  }
  const wdl = debug && decodeWdl(debug.logits);
  return (
    <main>
      <header>
        <h1>doppelbot</h1>
        <span>Cross-origin isolated: {window.crossOriginIsolated ? "yes" : "no"}</span>
      </header>
      <section>
        <label>
          Rating{" "}
          <input
            type="number"
            value={rating}
            onChange={(event) => setRating(Number(event.target.value))}
          />
        </label>
        <label>
          Stretch {stretch}
          <input
            type="range"
            min="-300"
            max="500"
            value={stretch}
            onChange={(event) => setStretch(Number(event.target.value))}
          />
        </label>
        <select value={side} onChange={(event) => setSide(event.target.value as "w" | "b")}>
          <option value="w">Play white</option>
          <option value="b">Play black</option>
        </select>
        <button onClick={reset}>New game</button>
      </section>
      <div className="game">
        <Chessboard
          options={{
            position: fen,
            onPieceDrop: drop,
            boardOrientation: side === "w" ? "white" : "black",
          }}
        />
        <aside>
          <strong>
            {thinking ? "Maia is thinking" : game.current.isGameOver() ? "Game over" : "Your move"}
          </strong>
          {error && <p role="alert">Maia error: {error}</p>}
          <p>
            {DEFAULT_CONFIG.model.id} · {backend} · bot {rating + stretch}
          </p>
          <button
            onClick={() => {
              if (thinking) return;
              game.current.undo();
              game.current.undo();
              const board = game.current.fen();
              setFen(board);
              setFens(recentFens(game.current));
            }}
          >
            Undo
          </button>
          <details>
            <summary>Debug</summary>
            <p>
              {debug &&
                `${debug.ms.toFixed(0)}ms · W ${wdl!.win.toFixed(2)} D ${wdl!.draw.toFixed(2)} L ${wdl!.loss.toFixed(2)}`}
            </p>
            {debug?.top.map((move) => (
              <div key={move.move}>
                {move.move} {(move.probability * 100).toFixed(1)}%
              </div>
            ))}
          </details>
        </aside>
      </div>
    </main>
  );
}

export default App;
