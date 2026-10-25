import type {
  Context,
  SQSBatchResponse,
  SQSEvent,
  APIGatewayProxyEventV2,
  APIGatewayProxyResultV2,
} from "aws-lambda";
import { DEFAULT_CONFIG } from "@doppelbot/config";
import { handleApi } from "./api.ts";
import { apiDeps, blobs, queue, store } from "./aws.ts";
import { runIngest, type IngestRequest } from "./ingest.ts";

// Stop fetching this long before the Lambda timeout, leaving time to write the last month and
// re-enqueue the continuation.
const DEADLINE_MARGIN_MS = 45_000;

/** SQS consumer (batch size 1). runIngest records failures on the job, so it never throws. */
export async function ingest(event: SQSEvent, context: Context): Promise<SQSBatchResponse> {
  const deadline = Date.now() + context.getRemainingTimeInMillis() - DEADLINE_MARGIN_MS;
  for (const record of event.Records) {
    const request = JSON.parse(record.body) as IngestRequest;
    const outcome = await runIngest(
      request,
      { store, blobs, queue, deadline },
      DEFAULT_CONFIG.ingest,
    );
    console.log(
      JSON.stringify({ job: request.jobId, site: request.site, user: request.username, outcome }),
    );
  }
  return { batchItemFailures: [] };
}

/** Function URL, served at /api/ingest through the router. */
export async function api(event: APIGatewayProxyEventV2): Promise<APIGatewayProxyResultV2> {
  const body =
    event.isBase64Encoded && event.body ? Buffer.from(event.body, "base64").toString() : event.body;
  const response = await handleApi(
    {
      method: event.requestContext.http.method,
      query: event.queryStringParameters ?? {},
      body: body ?? null,
    },
    apiDeps(DEFAULT_CONFIG.ingest),
  );
  return { ...response, headers: { "content-type": "application/json" } };
}
