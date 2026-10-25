import { expect, test } from "vite-plus/test";
import { HttpClient, HttpError, USER_AGENT } from "../src/index.ts";

function setup(statuses: number[], options: { canWait?: (ms: number) => boolean } = {}) {
  let clock = 0;
  const sleeps: number[] = [];
  const calls: { url: string; headers: Record<string, string>; at: number }[] = [];
  const queue = [...statuses];
  const client = new HttpClient({
    minIntervalMs: 1000,
    backoffMs: 60_000,
    now: () => clock,
    sleep: async (ms) => {
      sleeps.push(ms);
      clock += ms;
    },
    fetch: (async (url: string, init: RequestInit) => {
      calls.push({ url, headers: init.headers as Record<string, string>, at: clock });
      return new Response("{}", { status: queue.shift() ?? 200 });
    }) as typeof fetch,
    ...options,
  });
  return { client, sleeps, calls };
}

test("sends a User-Agent and spaces requests by the minimum interval", async () => {
  const { client, sleeps, calls } = setup([]);
  await client.get("https://x/a");
  await client.get("https://x/b");
  expect(calls.map((c) => c.headers["User-Agent"])).toEqual([USER_AGENT, USER_AGENT]);
  expect(sleeps).toEqual([1000]);
  expect(calls[1]!.at - calls[0]!.at).toBe(1000);
});

test("waits out a 429 and retries", async () => {
  const { client, sleeps, calls } = setup([429, 200]);
  const response = await client.get("https://x/a");
  expect(response.status).toBe(200);
  expect(calls).toHaveLength(2);
  expect(sleeps).toContain(60_000);
});

test("gives up on a 429 when there is no time to wait", async () => {
  const { client, calls } = setup([429, 200], { canWait: () => false });
  await expect(client.get("https://x/a")).rejects.toThrow(HttpError);
  expect(calls).toHaveLength(1);
});

test("retries 5xx a few times, then fails", async () => {
  const { client, calls } = setup([503, 503, 503, 503, 200]);
  await expect(client.get("https://x/a")).rejects.toMatchObject({ status: 503 });
  expect(calls).toHaveLength(4);
});

test("returns 404 to the caller instead of throwing", async () => {
  const { client } = setup([404]);
  expect((await client.get("https://x/a")).status).toBe(404);
});
