import type { Profile } from "./profile.tsx";

export type Site = "lichess" | "chesscom";
export type TimeControl = "blitz" | "rapid";
export type SiteRating = { site: Site; timeControl: TimeControl; rating: number; games: number };

export const SITE_NAMES: Record<Site, string> = { lichess: "Lichess", chesscom: "chess.com" };

/** The site answered, and the username doesn't exist (or the account is closed). */
export class UserNotFoundError extends Error {
  constructor(site: Site, username: string) {
    super(`No ${SITE_NAMES[site]} account named "${username}"`);
  }
}

type LichessPerf = { rating: number; games: number };
type LichessUser = { disabled?: boolean; perfs?: Partial<Record<TimeControl, LichessPerf>> };

export async function lookupLichess(username: string): Promise<SiteRating[]> {
  const response = await fetch(`https://lichess.org/api/user/${encodeURIComponent(username)}`);
  if (response.status === 404) throw new UserNotFoundError("lichess", username);
  if (!response.ok) throw new Error(`Lichess lookup failed (${response.status})`);
  const user = (await response.json()) as LichessUser;
  if (user.disabled) throw new UserNotFoundError("lichess", username);
  return (["blitz", "rapid"] as const).flatMap((timeControl) => {
    const perf = user.perfs?.[timeControl];
    return perf?.games
      ? [{ site: "lichess", timeControl, rating: perf.rating, games: perf.games }]
      : [];
  });
}

type ChessComStat = {
  last?: { rating: number };
  record?: { win: number; loss: number; draw: number };
};
type ChessComStats = Partial<Record<`chess_${TimeControl}`, ChessComStat>>;

export async function lookupChessCom(username: string): Promise<SiteRating[]> {
  const response = await fetch(
    `https://api.chess.com/pub/player/${encodeURIComponent(username.toLowerCase())}/stats`,
  );
  if (response.status === 404) throw new UserNotFoundError("chesscom", username);
  if (!response.ok) throw new Error(`chess.com lookup failed (${response.status})`);
  const stats = (await response.json()) as ChessComStats;
  return (["blitz", "rapid"] as const).flatMap((timeControl) => {
    const stat = stats[`chess_${timeControl}`];
    const record = stat?.record;
    const games = record ? record.win + record.loss + record.draw : 0;
    return stat?.last && games
      ? [{ site: "chesscom", timeControl, rating: stat.last.rating, games }]
      : [];
  });
}

/** Look up every username in the profile. Sites that fail are skipped. */
export async function lookupProfile(profile: Profile): Promise<SiteRating[]> {
  const lookups = await Promise.allSettled([
    profile.lichess ? lookupLichess(profile.lichess) : Promise.resolve([]),
    profile.chesscom ? lookupChessCom(profile.chesscom) : Promise.resolve([]),
  ]);
  return lookups.flatMap((lookup) => (lookup.status === "fulfilled" ? lookup.value : []));
}

/** The rating from the time control with the most games, across both sites. */
export function pickRating(ratings: readonly SiteRating[]): SiteRating | undefined {
  return ratings.reduce<SiteRating | undefined>(
    (best, rating) => (!best || rating.games > best.games ? rating : best),
    undefined,
  );
}

// Chess.com to Lichess rating equivalents from ChessGoals' rating comparison (updated July 2026;
// players with RD < 150 on both sites, about 50% fall within ±70 for blitz):
// https://chessgoals.com/rating-comparison/. Maia was trained on Lichess games, so its Elo input
// is on the Lichess scale. Each table is [x, y] points, interpolated linearly.
type Table = readonly (readonly [number, number])[];

const CHESSCOM_BLITZ_TO_LICHESS_BLITZ: Table = [
  [500, 1090],
  [600, 1160],
  [700, 1225],
  [800, 1290],
  [900, 1360],
  [1000, 1425],
  [1100, 1490],
  [1200, 1555],
  [1300, 1620],
  [1400, 1685],
  [1500, 1755],
  [1600, 1820],
  [1700, 1885],
  [1800, 1950],
  [1900, 2015],
  [2000, 2080],
  [2100, 2145],
  [2200, 2210],
  [2300, 2275],
  [2400, 2340],
  [2500, 2410],
  [2600, 2475],
  [2700, 2540],
  [2800, 2610],
  [2900, 2675],
  [3000, 2745],
];

// ChessGoals maps chess.com rapid only via chess.com blitz, so rapid goes rapid → blitz → Lichess.
const CHESSCOM_RAPID_TO_CHESSCOM_BLITZ: Table = [
  [815, 500],
  [905, 600],
  [995, 700],
  [1085, 800],
  [1170, 900],
  [1255, 1000],
  [1340, 1100],
  [1420, 1200],
  [1500, 1300],
  [1580, 1400],
  [1655, 1500],
  [1730, 1600],
  [1800, 1700],
  [1870, 1800],
  [1935, 1900],
  [1995, 2000],
  [2055, 2100],
  [2110, 2200],
  [2165, 2300],
  [2215, 2400],
  [2260, 2500],
  [2300, 2600],
  [2340, 2700],
  [2375, 2800],
  [2405, 2900],
  [2430, 3000],
];

/** Piecewise-linear lookup, extending the end segments past the table's range. */
function interpolate(table: Table, x: number): number {
  let i = 1;
  while (i < table.length - 1 && x > table[i]![0]) i += 1;
  const [x0, y0] = table[i - 1]!;
  const [x1, y1] = table[i]!;
  return y0 + ((x - x0) * (y1 - y0)) / (x1 - x0);
}

/** The rating on Lichess's scale, which is what Maia expects. */
export function lichessEquivalent(rating: Pick<SiteRating, "site" | "timeControl" | "rating">) {
  if (rating.site === "lichess") return rating.rating;
  const blitz =
    rating.timeControl === "rapid"
      ? interpolate(CHESSCOM_RAPID_TO_CHESSCOM_BLITZ, rating.rating)
      : rating.rating;
  return Math.round(interpolate(CHESSCOM_BLITZ_TO_LICHESS_BLITZ, blitz));
}
