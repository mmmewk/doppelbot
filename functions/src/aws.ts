import { randomUUID } from "node:crypto";
import { PutObjectCommand, S3Client } from "@aws-sdk/client-s3";
import { SendMessageCommand, SQSClient } from "@aws-sdk/client-sqs";
import { DynamoDBClient } from "@aws-sdk/client-dynamodb";
import {
  DynamoDBDocumentClient,
  GetCommand,
  PutCommand,
  QueryCommand,
  UpdateCommand,
} from "@aws-sdk/lib-dynamodb";
import { Resource } from "sst";
import type { Site } from "@doppelbot/games";
import type { ApiDeps } from "./api.ts";
import {
  playerKey,
  type BlobStore,
  type IngestRequest,
  type IngestStore,
  type RequestQueue,
} from "./ingest.ts";

// AWS implementations of the interfaces in ingest.ts and api.ts. Item layout (docs/PLAN.md
// "Data and storage"): pk PLAYER#<site>:<username>; sk META, MONTH#<yyyy-mm> or JOB#<jobId>.

const JOB_TTL_SECONDS = 30 * 24 * 3600;

const dynamo = DynamoDBDocumentClient.from(new DynamoDBClient({}), {
  marshallOptions: { removeUndefinedValues: true },
});
const s3 = new S3Client({});
const sqs = new SQSClient({});

async function update(site: Site, username: string, sk: string, fields: Record<string, unknown>) {
  const names = Object.keys(fields);
  await dynamo.send(
    new UpdateCommand({
      TableName: Resource.Data.name,
      Key: { pk: playerKey(site, username), sk },
      UpdateExpression: "SET " + names.map((_, i) => `#f${i} = :v${i}`).join(", "),
      ExpressionAttributeNames: Object.fromEntries(names.map((name, i) => [`#f${i}`, name])),
      ExpressionAttributeValues: Object.fromEntries(
        names.map((name, i) => [`:v${i}`, fields[name]]),
      ),
    }),
  );
}

export const store: IngestStore = {
  async listMonths(site, username) {
    const items: Record<string, unknown>[] = [];
    let startKey: Record<string, unknown> | undefined;
    do {
      const page = await dynamo.send(
        new QueryCommand({
          TableName: Resource.Data.name,
          KeyConditionExpression: "pk = :pk AND begins_with(sk, :sk)",
          ExpressionAttributeValues: { ":pk": playerKey(site, username), ":sk": "MONTH#" },
          ExclusiveStartKey: startKey,
        }),
      );
      items.push(...(page.Items ?? []));
      startKey = page.LastEvaluatedKey;
    } while (startKey);
    return items as unknown as Awaited<ReturnType<IngestStore["listMonths"]>>;
  },
  async putMonth(site, username, item) {
    await dynamo.send(
      new PutCommand({
        TableName: Resource.Data.name,
        Item: { pk: playerKey(site, username), sk: `MONTH#${item.month}`, ...item },
      }),
    );
  },
  // Update, not Put: META also holds lastRefreshRequestedAt, written by the API.
  async putMeta(site, username, summary) {
    await update(site, username, "META", { site, username: username.toLowerCase(), ...summary });
  },
  async updateJob(site, username, jobId, patch) {
    await update(site, username, `JOB#${jobId}`, {
      ...patch,
      updatedAt: Date.now(),
    });
  },
};

export const blobs: BlobStore = {
  async put(key, body, options) {
    await s3.send(
      new PutObjectCommand({
        Bucket: Resource.DataBucket.name,
        Key: key,
        Body: body,
        ContentType: options.contentType,
        ContentEncoding: options.contentEncoding,
      }),
    );
  },
};

// FIFO with one group per site, so the Lambda never has two requests in flight to the same site
// (Lichess asks for strictly sequential requests).
export const queue: RequestQueue = {
  async enqueue(request: IngestRequest) {
    await sqs.send(
      new SendMessageCommand({
        QueueUrl: Resource.IngestQueue.url,
        MessageBody: JSON.stringify(request),
        MessageGroupId: request.site,
        MessageDeduplicationId: `${request.jobId}:${request.attempt}`,
      }),
    );
  },
};

export const apiStore = {
  async getMeta(site: Site, username: string) {
    const result = await dynamo.send(
      new GetCommand({
        TableName: Resource.Data.name,
        Key: { pk: playerKey(site, username), sk: "META" },
      }),
    );
    return result.Item ?? null;
  },
  async getJob(site: Site, username: string, jobId: string) {
    const result = await dynamo.send(
      new GetCommand({
        TableName: Resource.Data.name,
        Key: { pk: playerKey(site, username), sk: `JOB#${jobId}` },
      }),
    );
    return result.Item ?? null;
  },
  async createJob(site: Site, username: string, jobId: string, requestedAt: number) {
    await dynamo.send(
      new PutCommand({
        TableName: Resource.Data.name,
        Item: {
          pk: playerKey(site, username),
          sk: `JOB#${jobId}`,
          type: "ingest",
          site,
          username,
          status: "queued",
          createdAt: requestedAt,
          updatedAt: requestedAt,
          expiresAt: Math.floor(requestedAt / 1000) + JOB_TTL_SECONDS,
        },
      }),
    );
    await update(site, username, "META", { lastRefreshRequestedAt: requestedAt, lastJobId: jobId });
  },
};

export const apiDeps = (config: ApiDeps["config"]): ApiDeps => ({
  ...apiStore,
  enqueue: (request) => queue.enqueue(request),
  now: () => Date.now(),
  newId: randomUUID,
  config,
});
