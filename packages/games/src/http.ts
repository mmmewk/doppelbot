export const USER_AGENT = "doppelbot-ingest (github.com/mmmewk/doppelbot)";

export class HttpError extends Error {
  constructor(
    readonly status: number,
    readonly url: string,
  ) {
    super(`HTTP ${status} for ${url}`);
  }
}

export interface HttpClientOptions {
  /** Minimum gap between the starts of consecutive requests, ms. */
  minIntervalMs: number;
  /** Wait after an HTTP 429, ms. */
  backoffMs: number;
  /** Retries after 429 or 5xx before giving up. */
  maxRetries?: number;
  userAgent?: string;
  /**
   * Called before sleeping through a 429 backoff. Return false when there isn't enough time left
   * (e.g. near a Lambda deadline); the request then fails with the 429 instead of waiting.
   */
  canWait?: (ms: number) => boolean;
  fetch?: typeof fetch;
  sleep?: (ms: number) => Promise<void>;
  now?: () => number;
}

/** Sequential, rate-limited GETs with a User-Agent. One instance per site. */
export class HttpClient {
  private lastStart = -Infinity;
  private readonly fetchFn: typeof fetch;
  private readonly sleep: (ms: number) => Promise<void>;
  private readonly now: () => number;

  constructor(private readonly options: HttpClientOptions) {
    this.fetchFn = options.fetch ?? fetch;
    this.sleep = options.sleep ?? ((ms) => new Promise((resolve) => setTimeout(resolve, ms)));
    this.now = options.now ?? Date.now;
  }

  /** Returns the response for any status except 429 and 5xx (callers handle 404 themselves). */
  async get(url: string, init: { headers?: Record<string, string>; signal?: AbortSignal } = {}) {
    const maxRetries = this.options.maxRetries ?? 3;
    for (let attempt = 0; ; attempt++) {
      const wait = this.lastStart + this.options.minIntervalMs - this.now();
      if (wait > 0) await this.sleep(wait);
      this.lastStart = this.now();
      const response = await this.fetchFn(url, {
        headers: { "User-Agent": this.options.userAgent ?? USER_AGENT, ...init.headers },
        signal: init.signal,
      });
      const retryable = response.status === 429 || response.status >= 500;
      if (!retryable) return response;
      await response.body?.cancel();
      const backoff = response.status === 429 ? this.options.backoffMs : 2000 * 2 ** attempt;
      const canWait = response.status !== 429 || (this.options.canWait?.(backoff) ?? true);
      if (attempt >= maxRetries || !canWait) throw new HttpError(response.status, url);
      await this.sleep(backoff);
    }
  }
}
