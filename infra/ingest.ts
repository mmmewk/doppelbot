import { dataBucket, table } from "./storage.ts";
import { router } from "./web.ts";

// Ingestion: POST /api/ingest enqueues a refresh; the worker fetches missing months from
// Lichess and chess.com into the data bucket and table. See docs/PLAN.md "Phase 2".

const workerTimeout = 15; // minutes, the Lambda maximum

// FIFO with one message group per site (see functions/src/aws.ts), so only one worker request
// stream is ever open against each site. Lichess asks for strictly sequential API use.
export const queue = new sst.aws.Queue("IngestQueue", {
  fifo: true,
  visibilityTimeout: `${workerTimeout + 1} minutes`,
});

// Batch size 1: one profile per invocation, which may run for most of the timeout. A worker that
// runs low on time saves its progress and enqueues a continuation itself.
queue.subscribe(
  {
    handler: "functions/src/handlers.ingest",
    timeout: `${workerTimeout} minutes`,
    memory: "1024 MB",
    link: [table, dataBucket, queue],
  },
  { batch: { size: 1 } },
);

// Same origin as the site, so the browser needs no CORS.
export const api = new sst.aws.Function("IngestApi", {
  handler: "functions/src/handlers.api",
  timeout: "10 seconds",
  link: [table, queue],
  url: { router: { instance: router, path: "/api/ingest" } },
});
