import { DEFAULT_CONFIG } from "@doppelbot/config";
import { expect, test } from "vite-plus/test";
import { handleApi, type ApiDeps, type MetaRecord } from "../src/api.ts";
import type { IngestRequest } from "../src/ingest.ts";

function setup(meta: MetaRecord | null = null) {
  const enqueued: IngestRequest[] = [];
  const created: string[] = [];
  const deps: ApiDeps = {
    getMeta: async () => meta,
    getJob: async (_s, _u, jobId) => (jobId === "j1" ? { status: "running" } : null),
    createJob: async (_s, _u, jobId) => void created.push(jobId),
    enqueue: async (request) => void enqueued.push(request),
    now: () => 1_000_000,
    newId: () => "new-job",
    config: DEFAULT_CONFIG.ingest,
  };
  return { deps, enqueued, created };
}

const post = (body: unknown) => ({ method: "POST", query: {}, body: JSON.stringify(body) });

test("POST enqueues a refresh and returns the job id", async () => {
  const { deps, enqueued, created } = setup();
  const response = await handleApi(
    post({ site: "chesscom", username: "Alice_1", maxMonths: 6 }),
    deps,
  );
  expect(response.statusCode).toBe(202);
  expect(JSON.parse(response.body)).toEqual({ jobId: "new-job" });
  expect(created).toEqual(["new-job"]);
  expect(enqueued).toEqual([
    { jobId: "new-job", site: "chesscom", username: "alice_1", maxMonths: 6, attempt: 0 },
  ]);
});

test("POST rejects bad input", async () => {
  const { deps, enqueued } = setup();
  for (const body of [
    { site: "chess24", username: "alice" },
    { site: "lichess", username: "a" },
    { site: "lichess", username: "alice/../bob" },
    { site: "lichess", username: "alice", maxMonths: 0 },
    { site: "lichess", username: "alice", maxMonths: 9999 },
  ]) {
    expect((await handleApi(post(body), deps)).statusCode).toBe(400);
  }
  expect((await handleApi({ method: "POST", query: {}, body: "nope" }, deps)).statusCode).toBe(400);
  expect(enqueued).toEqual([]);
});

test("POST is throttled per profile", async () => {
  const { deps, enqueued } = setup({
    lastRefreshRequestedAt: 1_000_000 - 60_000,
    lastJobId: "old",
  });
  const response = await handleApi(post({ site: "lichess", username: "alice" }), deps);
  expect(response.statusCode).toBe(200);
  expect(JSON.parse(response.body)).toEqual({ skipped: "refreshed recently", jobId: "old" });
  expect(enqueued).toEqual([]);

  const later = setup({ lastRefreshRequestedAt: 1_000_000 - 301_000 });
  expect(
    (await handleApi(post({ site: "lichess", username: "alice" }), later.deps)).statusCode,
  ).toBe(202);
});

test("GET returns metadata and job status", async () => {
  const { deps } = setup({ games: 12 });
  const response = await handleApi(
    { method: "GET", query: { site: "lichess", username: "alice", jobId: "j1" }, body: null },
    deps,
  );
  expect(JSON.parse(response.body)).toEqual({ meta: { games: 12 }, job: { status: "running" } });
  const missing = await handleApi(
    { method: "GET", query: { site: "lichess", username: "alice" }, body: null },
    setup().deps,
  );
  expect(missing.statusCode).toBe(404);
  expect((await handleApi({ method: "PUT", query: {}, body: null }, deps)).statusCode).toBe(405);
});
