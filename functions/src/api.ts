import { resolveConfig, type Config } from "@doppelbot/config";
import type { Site } from "@doppelbot/games";
import type { IngestRequest } from "./ingest.ts";

// POST /api/ingest   {site, username, maxMonths?}  -> 202 {jobId}   enqueue a refresh
// GET  /api/ingest?site=&username=[&jobId=]        -> 200 {meta, job?}  progress and stored totals
//
// Public and unauthenticated: the data is public games, ingestion only ever fetches what the
// site itself serves, and refreshes of one profile are throttled (ingest.minRefreshIntervalSeconds).

export interface ApiRequest {
  method: string;
  query: Record<string, string | undefined>;
  body: string | null;
}

export interface ApiResponse {
  statusCode: number;
  body: string;
}

export interface MetaRecord {
  lastRefreshRequestedAt?: number;
  lastJobId?: string;
  [field: string]: unknown;
}

export interface ApiDeps {
  getMeta(site: Site, username: string): Promise<MetaRecord | null>;
  getJob(site: Site, username: string, jobId: string): Promise<Record<string, unknown> | null>;
  /** Creates the queued JOB item and records the request on META in one step. */
  createJob(site: Site, username: string, jobId: string, requestedAt: number): Promise<void>;
  enqueue(request: IngestRequest): Promise<void>;
  now: () => number;
  newId: () => string;
  config: Config["ingest"];
}

const USERNAME = /^[A-Za-z0-9_-]{2,30}$/;

function json(statusCode: number, value: unknown): ApiResponse {
  return { statusCode, body: JSON.stringify(value) };
}

function parseSiteAndUsername(
  site: unknown,
  username: unknown,
): { site: Site; username: string } | string {
  if (site !== "lichess" && site !== "chesscom") return "site must be lichess or chesscom";
  if (typeof username !== "string" || !USERNAME.test(username)) return "invalid username";
  return { site, username: username.toLowerCase() };
}

export async function handleApi(request: ApiRequest, deps: ApiDeps): Promise<ApiResponse> {
  if (request.method === "POST") {
    let body: Record<string, unknown>;
    try {
      body = JSON.parse(request.body ?? "") as Record<string, unknown>;
    } catch {
      return json(400, { error: "body must be JSON" });
    }
    const target = parseSiteAndUsername(body.site, body.username);
    if (typeof target === "string") return json(400, { error: target });
    const { maxMonths } = body;
    if (maxMonths !== undefined) {
      try {
        resolveConfig({ ingest: { maxMonths: maxMonths as number } });
      } catch {
        return json(400, { error: "maxMonths out of range" });
      }
    }

    const now = deps.now();
    const meta = await deps.getMeta(target.site, target.username);
    const last = meta?.lastRefreshRequestedAt;
    if (last !== undefined && now - last < deps.config.minRefreshIntervalSeconds * 1000) {
      return json(200, { skipped: "refreshed recently", jobId: meta?.lastJobId });
    }
    const jobId = deps.newId();
    await deps.createJob(target.site, target.username, jobId, now);
    await deps.enqueue({
      jobId,
      ...target,
      maxMonths: maxMonths as number | undefined,
      attempt: 0,
    });
    return json(202, { jobId });
  }

  if (request.method === "GET") {
    const target = parseSiteAndUsername(request.query.site, request.query.username);
    if (typeof target === "string") return json(400, { error: target });
    const [meta, job] = await Promise.all([
      deps.getMeta(target.site, target.username),
      request.query.jobId
        ? deps.getJob(target.site, target.username, request.query.jobId)
        : Promise.resolve(undefined),
    ]);
    if (!meta && !job) return json(404, { error: "not ingested yet" });
    return json(200, { meta, job });
  }

  return json(405, { error: "method not allowed" });
}
