# doppelbot: Expected AWS costs

Estimates as of 2026-10-25, for `us-west-1` on-demand list prices. They are planning numbers, not a quote:
check the AWS pricing pages before relying on any single figure. The calculation is a small script; the
assumptions below are its inputs, so change an input and redo the arithmetic rather than editing totals by hand.

## Summary

| Monthly active users | Without fitting (Phase 2 as built) | With server fitting (Phase 3) |
| -------------------- | ---------------------------------- | ----------------------------- |
| 100                  | about $0.10                        | about $13                     |
| 1,000                | about $1                           | about $130                    |
| 10,000               | about $25                          | about $1,300                  |
| 100,000              | about $460 (about $750 today)      | about $13,000                 |

Assumes 70% one-time users and 30% power users each month. Bills include the AWS free tier, which
covers almost everything below about 5,000 users.

- **Server-side fitting is 90 to 97% of the cost.** If Phase 1b says Go, where the fit runs is the
  most important cost decision: in the browser (free to us), on Fargate only for returning users, or on Fargate for everyone.
- **Everything else costs about a cent per user per month.** Without fitting, the bill stays under $30 a month up to 10,000 users.
- **The first download is the biggest per-user item today.** A first visit downloads about 50 MB:
  - The 28 MB onnxruntime WASM file goes uncompressed, because CloudFront only compresses files up to 10 MB.
  - The fp32 model is another 21 MB.

  Shipping a pre-compressed WASM file (about 6.5 MB gzipped) and the fp16 model (11 MB) cuts this to about 19 MB.

## Prices used

| Service         | Price                                                                                   | Always-free monthly allowance            |
| --------------- | --------------------------------------------------------------------------------------- | ---------------------------------------- |
| CloudFront      | $0.085/GB out (North America), $0.01 per 10k HTTPS requests, $0.10 per 1M function runs | 1 TB out, 10M requests, 2M function runs |
| Lambda (x86)    | $0.0000166667 per GB-second, $0.20 per 1M requests                                      | 400k GB-seconds, 1M requests             |
| S3 Standard     | $0.026/GB-month, $0.0055 per 1k PUTs, $0.00044 per 1k GETs                              | None after the first 12 months           |
| DynamoDB        | On demand: about $0.70 per 1M writes, $0.14 per 1M reads, $0.28/GB-month                | 25 GB storage                            |
| SQS (FIFO)      | $0.50 per 1M requests                                                                   | 1M requests                              |
| CloudWatch Logs | $0.50/GB ingested                                                                       | 5 GB                                     |
| Fargate         | $0.04656 per vCPU-hour, $0.00511 per GB-hour                                            | None                                     |

## Assumptions per user per month

**One-time user:** visits once or twice in one month and never returns.
**Power user:** plays about 150 games a month and opens the app about 40 times.

| Item                                     | One-time user                                          | Power user                                                                                            |
| ---------------------------------------- | ------------------------------------------------------ | ----------------------------------------------------------------------------------------------------- |
| Download (app + model + their game data) | 52 MB today, 19 MB optimized                           | 65 MB today, 28 MB optimized (one full redownload a month: new device, cache eviction or a new model) |
| CloudFront requests                      | 100                                                    | 1,500                                                                                                 |
| Ingestion                                | 1 backfill of up to 1,000 games (`ingest.targetGames`) | About 30 refreshes, each refetching the open month                                                    |
| Lambda time at 1 GB                      | 66 GB-seconds                                          | 200 GB-seconds                                                                                        |
| S3 writes / reads                        | 50 / 60                                                | 120 / 300                                                                                             |
| S3 data stored                           | 2 MB, kept after they leave                            | 3 MB                                                                                                  |
| DynamoDB writes / reads                  | 100 / 200                                              | 500 / 2,000 (job status polling)                                                                      |
| Fargate fits (Phase 3 only)              | 1                                                      | 4 (refits throttled to about weekly)                                                                  |

- **Lambda time** is mostly waiting on the sites. Lichess streams its game export at about 20 games a second
  without OAuth, so a 1,000-game backfill takes about a minute.
- **Measured on `ingest-dev`, 2026-10-25:** a real chess.com backfill of about 1,000 games over 39 months took 58 s at
  1 GB, using 225 MB at peak, and stored about 1 KB per game in S3 (games plus occurrences). The Lambda could run at
  512 MB, which would halve its GB-seconds if it isn't CPU-bound.
- **A fit** is assumed to take 30 minutes on 2 vCPU and 8 GB, which is $0.067. This is the least certain
  number on the page. Fitting a 128-d vector by backpropagating through the frozen 5M model over a 200-game window
  could plausibly take anywhere from 10 to 60 minutes on CPU. Phase 1b should measure it.

## Cost per user per month

| Piece                    | One-time user    | Power user       |
| ------------------------ | ---------------- | ---------------- |
| CloudFront (today)       | $0.0045          | $0.0073          |
| CloudFront (optimized)   | $0.0017          | $0.0042          |
| Lambda (ingest + API)    | $0.0011          | $0.0034          |
| S3                       | $0.0004          | $0.0009          |
| DynamoDB                 | $0.0001          | $0.0006          |
| CloudWatch Logs          | $0.0001          | $0.0005          |
| SQS                      | about $0         | about $0         |
| **Subtotal, no fitting** | **$0.003–0.006** | **$0.010–0.013** |
| Fargate fits (Phase 3)   | $0.067           | $0.27            |
| **Total with fitting**   | **about $0.07**  | **about $0.28**  |

These are before the free tier. Blended at 70/30, that is about $0.008 per user per month without
fitting and about $0.13 with it.

## Monthly bill by scale

Optimized download, 70/30 mix, after the free tier. S3 storage is shown after 12 months, by which time the stored data
of one-time users who have left has piled up.

| Piece            | 100 users | 1,000 users | 10,000 users | 100,000 users |
| ---------------- | --------- | ----------- | ------------ | ------------- |
| CloudFront       | $0        | $0          | $1           | $152          |
| Lambda           | $0        | $0          | $11          | $174          |
| S3               | $0.09     | $0.91       | $9           | $91           |
| DynamoDB         | $0.03     | $0.26       | $3           | $26           |
| Logs and SQS     | $0        | $0          | $0           | $20           |
| **No fitting**   | **$0.12** | **$1.17**   | **$24**      | **$462**      |
| Fargate fits     | $13       | $127        | $1,273       | $12,730       |
| **With fitting** | **$13**   | **$128**    | **$1,297**   | **$13,192**   |

At 100,000 users the unoptimized download adds about $290 a month in CloudFront.

## Fixed costs (independent of users)

- **Route 53:** $0.50 a month for the `mekoppe.com` hosted zone, which already exists.
- **SQS polling:** the Lambda queue subscription long-polls continuously, about 650k requests a month per deployed stage.
  That is inside the free tier for one stage. Remove stages like `ingest-dev` when they are no longer needed.
- **Model files in S3:** negligible.
- **Phase 3, if built:**
  - **ECR:** about $0.30 a month for a 2 to 3 GB PyTorch image.
  - **Avoid a NAT gateway.** One costs about $35 a month before any traffic. Run the Fargate tasks in public subnets with a
    public IP ($0.005 an hour only while a task runs), or use SST's EC2 NAT option (a few dollars a month).

## Levers, biggest first

1. **Where fits run.** Fitting in the browser on WebGPU takes this line to $0. If fits stay on Fargate:
   - **Don't fit one-time users.** Fit on the second visit, or once the user opts in. This saves about $0.067 for 70% of users.
   - **Throttle refits:** for example, at most once a week, or only after 20 new games. Refitting on every refresh would
     cost about $2 per power user per month instead of $0.27.
   - **Use Fargate Spot** for fits, which can be retried. It is about 70% cheaper.
2. **First download size:**
   - Upload the onnxruntime WASM file pre-compressed (`Content-Encoding: gzip` or `br`), since CloudFront won't compress it.
   - Ship fp16 (the Phase 1 step 6 recommendation).
   - Together these cut about 33 MB per new user.
3. **Stored data of departed users:** game data is a rebuildable cache, so an S3 lifecycle rule can delete the data of
   users not seen for 6 months. This keeps S3 flat instead of growing with every one-time user.
4. **Job status polling:** poll every 3 to 5 seconds instead of every 1 to 2. Or return the status with the `POST` and
   poll only while a job is running. This mostly matters for DynamoDB reads and Lambda requests at 100k users.
