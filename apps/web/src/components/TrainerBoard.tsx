import { Chessboard } from "react-chessboard";
import { playLine } from "../trainers.ts";

/** A static board showing where a line ends, with an arrow on its last move. */
export function TrainerBoard({
  id,
  line,
  label,
}: {
  id: string;
  line: readonly string[];
  label: string;
}) {
  const game = playLine(line);
  const last = game.history({ verbose: true }).at(-1);
  return (
    <figure className="trainer-board" aria-label={label}>
      <Chessboard
        options={{
          id,
          position: game.fen(),
          allowDragging: false,
          allowDrawingArrows: false,
          showAnimations: false,
          arrows: last
            ? [
                {
                  startSquare: last.from,
                  endSquare: last.to,
                  color: game.isCheckmate() ? "rgb(214, 69, 65)" : "rgb(74, 144, 226)",
                },
              ]
            : [],
        }}
      />
    </figure>
  );
}
