import { expect, test } from "vite-plus/test";
import { TRAINERS, playLine } from "./trainers.ts";

test("every trainer line is legal", () => {
  for (const trainer of TRAINERS) expect(() => playLine(trainer.line)).not.toThrow();
});

test("Loki's line ends in smothered mate", () => {
  const game = playLine(TRAINERS.find((trainer) => trainer.name === "Loki")!.line);
  expect(game.isCheckmate()).toBe(true);
  expect(game.history({ verbose: true }).at(-1)).toMatchObject({ from: "d4", to: "f3" });
});
