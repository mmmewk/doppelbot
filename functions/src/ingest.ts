import { gzipSync } from "node:zlib";
import { DEFAULT_CONFIG, type Config } from "@doppelbot/config";
import {
  HttpClient,
  HttpError,
  buildOccurrences,
  encodeOccurrences,
  fetchChesscomArchives,
  fetchChesscomMonth,
  fetchLichessCreatedAt,
  fetchLichessMonth,
  gamesToJsonl,
  needsFetch,
  recentMonths,
  monthOf,
  type Game,
  type MonthId,
  type Site,
  type Speed,
} from "@doppelbot/games";

// Ingestion core: fetch the months a profile is missing, normalize, and store. All I/O goes
// through the interfaces below so the logic is testable without AWS or the network
// (aws.ts has the S3, DynamoDB and SQS implementations).

export interface IngestRequest {
  jobId: string;
  site: Site;
  username: string;
  /** Override ingest.targetGames, e.g. to backfill further. */
  targetGames?: number;
  /** Overrides ingest.maxMonths. */
  maxMonths?: number;
  /** How many times this job has been re-enqueued to continue after running low on time. */
  attempt: number;
}

export type Ratings = Partial<Record<Speed, { rating: number; playedAt: number }>>;

/** DynamoDB `MONTH#<yyyy-mm>` item. */
export interface MonthItem {
  month: MonthId;
  /** S3 keys; null when the month had no games (nothing is written to S3). */
  gamesKey: string | null;
  occurrencesKey: string | null;
  games: number;
  /** Occurrence rows, i.e. moves the user played. */
  userMoves: number;
  /** Unix ms. A month fetched before it ended is refetched on the next refresh. */
  fetchedAt: number;
  /** The user's latest rating per speed within this month. */
  ratings: Ratings;
}

/** DynamoDB `META` item fields derived from the MONTH items. */
export interface MetaSummary {
  games: number;
  userMoves: number;
  months: number;
  firstMonth: MonthId | null;
  lastMonth: MonthId | null;
  ratings: Ratings;
  /** Unix ms of the last completed ingestion. */
  ingestedAt: number;
}

export interface JobPatch {
  status?: "queued" | "running" | "done" | "error";
  error?: string;
  monthsTotal?: number;
  monthsDone?: number;
  /** Games counted towards ingest.targetGames so far, stored and newly fetched. */
  games?: number;
  targetGames?: number;
}

export interface IngestStore {
  listMonths(site: Site, username: string): Promise<MonthItem[]>;
  putMonth(site: Site, username: string, item: MonthItem): Promise<void>;
  putMeta(site: Site, username: string, summary: MetaSummary): Promise<void>;
  updateJob(site: Site, username: string, jobId: string, patch: JobPatch): Promise<void>;
}

export interface BlobStore {
  put(
    key: string,
    body: Uint8Array,
    options: { contentType: string; contentEncoding?: string },
  ): Promise<void>;
}

export interface RequestQueue {
  enqueue(request: IngestRequest): Promise<void>;
}

export interface IngestDeps {
  store: IngestStore;
  blobs: BlobStore;
  queue: RequestQueue;
  /** Unix ms by which the run must have stopped fetching (Lambda timeout minus a margin). */
  deadline: number;
  now?: () => number;
  fetch?: typeof fetch;
  sleep?: (ms: number) => Promise<void>;
}

/** Stop starting new months with less than this left; a month's fetch plus a 429 backoff fits. */
export const MIN_MONTH_BUDGET_MS = 150_000;
/** A job that has had to continue this many times is failing for a reason retries won't fix. */
export const MAX_ATTEMPTS = 6;

export function playerKey(site: Site, username: string): string {
  return `PLAYER#${site}:${username.toLowerCase()}`;
}

export function s3Keys(site: Site, username: string, month: MonthId) {
  const prefix = `${site}/${username.toLowerCase()}/${month}`;
  return { games: `games/${prefix}.jsonl.gz`, occurrences: `occurrences/${prefix}.bin` };
}

class UserNotFound extends Error {}

export type IngestOutcome = "done" | "continued" | "error";

export async function runIngest(
  request: IngestRequest,
  deps: IngestDeps,
  config: Config["ingest"] = DEFAULT_CONFIG.ingest,
): Promise<IngestOutcome> {
  const { site, username, jobId } = request;
  const now = deps.now ?? Date.now;
  const job = (patch: JobPatch) => deps.store.updateJob(site, username, jobId, patch);
  const sleep =
    deps.sleep ?? ((ms: number) => new Promise<void>((resolve) => setTimeout(resolve, ms)));
  const summarizeNow = async () =>
    deps.store.putMeta(
      site,
      username,
      summarize(await deps.store.listMonths(site, username), now()),
    );
  try {
    await job({ status: "running" });
    const http = new HttpClient({
      minIntervalMs:
        site === "lichess" ? config.lichessMinIntervalMs : config.chesscomMinIntervalMs,
      backoffMs: config.rateLimitBackoffSeconds * 1000,
      canWait: (ms) => now() + ms < deps.deadline,
      fetch: deps.fetch,
      sleep,
      now,
    });
    const signal = () => AbortSignal.timeout(Math.max(deps.deadline - now(), 1));

    // Which months can hold games at all: Lichess from account creation, chess.com from its archive list.
    let available: (month: MonthId) => boolean;
    if (site === "lichess") {
      const createdAt = await fetchLichessCreatedAt(http, username, signal());
      if (createdAt === null) throw new UserNotFound();
      const first = monthOf(createdAt);
      available = (month) => month >= first;
    } else {
      const archives = await fetchChesscomArchives(http, username, signal());
      if (archives === null) throw new UserNotFound();
      available = (month) => archives.has(month);
    }

    // Walk back newest first until the target game count is reached. Final months already stored
    // count towards it without a fetch; the rest are fetched. monthsTotal is an upper bound.
    const stored = new Map((await deps.store.listMonths(site, username)).map((m) => [m.month, m]));
    const window = recentMonths(now(), request.maxMonths ?? config.maxMonths).filter(available);
    const targetGames = request.targetGames ?? config.targetGames;
    const todo = window.filter((month) => needsFetch(month, stored.get(month)?.fetchedAt));
    let games = 0;
    let done = 0;
    await job({ monthsTotal: todo.length, monthsDone: 0, games, targetGames });

    for (const month of window) {
      if (games >= targetGames) break;
      const item = stored.get(month);
      if (item && !needsFetch(month, item.fetchedAt)) {
        games += item.games;
        continue;
      }
      if (deps.deadline - now() < MIN_MONTH_BUDGET_MS) {
        return await continueLater(request, deps, summarizeNow);
      }
      try {
        const fetchedAt = now();
        const fetched =
          site === "lichess"
            ? await fetchLichessMonth(http, username, month, {
                includeCasual: config.includeCasual,
                signal: signal(),
              })
            : await fetchChesscomMonth(http, username, month, {
                includeCasual: config.includeCasual,
                signal: signal(),
              });
        const saved = await storeMonth(deps, site, username, month, fetched, fetchedAt);
        await deps.store.putMonth(site, username, saved);
        games += saved.games;
      } catch (error) {
        // Ran out of time mid-month (abort or a 429 we couldn't wait out): pick up from here later.
        if (!isOutOfTime(error)) throw error;
        // A rate limit is only worth retrying after the backoff, so use up the remaining time first.
        if (error instanceof HttpError) await sleep(Math.max(deps.deadline - now(), 0));
        return await continueLater(request, deps, summarizeNow);
      }
      await job({ monthsDone: ++done, games });
    }

    await summarizeNow();
    await job({ status: "done" });
    return "done";
  } catch (error) {
    await job({ status: "error", error: describe(error) }).catch(() => {});
    return "error";
  }
}

async function continueLater(
  request: IngestRequest,
  deps: IngestDeps,
  summarize: () => Promise<void>,
): Promise<IngestOutcome> {
  const job = (patch: JobPatch) =>
    deps.store.updateJob(request.site, request.username, request.jobId, patch);
  await summarize(); // months finished so far are usable immediately
  if (request.attempt + 1 >= MAX_ATTEMPTS) {
    await job({ status: "error", error: "ran out of time repeatedly; try again later" });
    return "error";
  }
  await deps.queue.enqueue({ ...request, attempt: request.attempt + 1 });
  await job({ status: "queued" });
  return "continued";
}

function isOutOfTime(error: unknown): boolean {
  if (error instanceof HttpError) return error.status === 429;
  return error instanceof Error && (error.name === "TimeoutError" || error.name === "AbortError");
}

function describe(error: unknown): string {
  if (error instanceof UserNotFound) return "user not found";
  return error instanceof Error ? error.message : String(error);
}

async function storeMonth(
  deps: IngestDeps,
  site: Site,
  username: string,
  month: MonthId,
  fetched: Game[],
  fetchedAt: number,
): Promise<MonthItem> {
  // Oldest first with a stable order, so gameIndex in the occurrences file is the line number in
  // the games file and a refetch of an open month keeps earlier indices.
  const seen = new Set<string>();
  const games = fetched
    .filter((game) => !seen.has(game.id) && seen.add(game.id))
    .sort((a, b) => a.playedAt - b.playedAt || a.id.localeCompare(b.id));
  const user = username.toLowerCase();
  const occurrences = games.flatMap((game, index) => buildOccurrences(game, index, user));
  const ratings: Ratings = {};
  for (const game of games) {
    const rating = game.white === user ? game.whiteElo : game.black === user ? game.blackElo : null;
    if (rating !== null && game.playedAt >= (ratings[game.speed]?.playedAt ?? 0)) {
      ratings[game.speed] = { rating, playedAt: game.playedAt };
    }
  }
  const keys = s3Keys(site, username, month);
  if (games.length > 0) {
    await deps.blobs.put(keys.games, gzipSync(gamesToJsonl(games)), {
      contentType: "application/x-ndjson",
      contentEncoding: "gzip",
    });
    await deps.blobs.put(keys.occurrences, encodeOccurrences(occurrences), {
      contentType: "application/octet-stream",
    });
  }
  return {
    month,
    gamesKey: games.length > 0 ? keys.games : null,
    occurrencesKey: games.length > 0 ? keys.occurrences : null,
    games: games.length,
    userMoves: occurrences.length,
    fetchedAt,
    ratings,
  };
}

export function summarize(months: MonthItem[], now: number): MetaSummary {
  const sorted = [...months].sort((a, b) => a.month.localeCompare(b.month));
  const ratings: Ratings = {};
  for (const item of sorted) {
    for (const [speed, latest] of Object.entries(item.ratings) as [
      Speed,
      NonNullable<Ratings[Speed]>,
    ][]) {
      if (latest.playedAt >= (ratings[speed]?.playedAt ?? 0)) ratings[speed] = latest;
    }
  }
  return {
    games: sorted.reduce((sum, item) => sum + item.games, 0),
    userMoves: sorted.reduce((sum, item) => sum + item.userMoves, 0),
    months: sorted.length,
    firstMonth: sorted[0]?.month ?? null,
    lastMonth: sorted.at(-1)?.month ?? null,
    ratings,
    ingestedAt: now,
  };
}
