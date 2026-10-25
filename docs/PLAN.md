# doppelbot: Plan

## Status and handoff (2026-10-25)

Read this section first if you are picking up the project.

**Done**

- **Phase 0, scaffold:**
  - Vite+ monorepo with `apps/web` (a React placeholder) and `packages/config` (zod schema plus `defaults.json`).
  - SST v4 infra in `sst.config.ts` and `infra/`.
  - A Python uv project in `python/`.
- **Phase 1, step 1: model comparison.** The code is in `python/src/doppelbot/`:
  - `games.py`: Lichess and chess.com fetchers, cached to `python/data/`, which is gitignored.
  - `accounts.py`: samples 20 accounts, two per 200-point band from 800 to 1800 on each site.
  - `positions.py`: extracts the user's moves, filters out time trouble, and splits by time.
  - `scorers.py`: batched Maia-2 and Maia-3 scoring. `tests/test_scorers.py` checks parity with each library's own inference.
  - `compare.py`: runs the comparison. Run it with `uv run python -m doppelbot.compare`; it resumes and skips finished results.
- **Phase 1, step 2: ONNX export.** `doppelbot.export` exports `maia3-5m` with dynamic-batch `tokens [B,64,97]`, `self_emb [B,128]` and `oppo_emb [B,128]`, alongside `moves.json` and `elo_emb.json`. Its tests compare the explicit-embedding wrapper with native PyTorch and ONNX Runtime.
- **Phase 1, step 3: local parity fixtures.** `doppelbot.fixtures` deterministically samples 500 cached-game positions with coverage for black turns, short histories, castling, en passant, promotions and underpromotions. It writes ignored real-game-derived fixtures to `python/data/fixtures/maia3-5m.json`, with tokens, legal masks, and top-k/WDL results at three rating pairs.
- **Phase 1, steps 4 to 5: TypeScript and parity.** `packages/maia` mirrors Maia tokenization, mover-relative legal masks, masked softmax, WDL decoding, Elo interpolation, and ONNX session execution. Its tests match all 500 Python fixtures in Node/WASM and Chromium WASM/WebGPU within $10^{-4}$.

- **Phase 1.5: done and deployed (2026-10-25).** In `apps/web`:
  - React Router with Home and Play tabs.
  - The landing page with the four trainers (Doppel, Zephyr, Loki, Maia) and their board lines, in `src/trainers.ts`.
  - The profile modal, which stores usernames in `localStorage`.
  - A Play page with rating lookup, click and drag moves, legal-move dots, promotion choice, undo, resign, game-end detection, a top-move toggle and a debug panel.
  - `packages/maia` gained `rankMoves` and `sampleMove`. Its browser test plays bot moves from three positions and checks that each is legal.
  - Verified on production with headless Chromium:
    - Cross-origin isolated, and `/play` loads directly.
    - The model is served from `/models` with immutable caching.
    - The modal rejects unknown usernames, and the profile survives a reload.
    - The rating lookup works, and Maia replies on WASM in about 20 to 90 ms.
    - Undo, resign and playing as Black work.

**Phase 1 step 1 results**

20 accounts, 35,631 held-out moves. Each cell is mean log-likelihood per move (higher is better) / top-1 accuracy.

| Variant                                                | Stated rating (L0) | Fitted offset (L1) |
| ------------------------------------------------------ | ------------------ | ------------------ |
| Maia-2 (blitz/rapid models)                            | −1.604 / 48.9%     | −1.589 / 49.3%     |
| Maia-3 5M, current position repeated (library default) | −1.490 / 51.5%     | −1.461 / 52.1%     |
| **Maia-3 5M, real 8-position history**                 | **−1.410 / 54.1%** | **−1.383 / 54.6%** |
| Maia-3 23M, repeated                                   | −1.467 / 52.1%     | −1.434 / 52.8%     |
| Maia-3 23M, real history                               | −1.382 / 54.8%     | −1.352 / 55.5%     |

- **Decision: use `maia3-5m` and always feed the real 8-position history.** It beats Maia-2 in every band, site, game phase and time control.
- **23M:** about +0.7 to +0.9 points of top-1 over 5M at roughly 4× the size. That isn't worth it for the browser, so 5M stays the default and 23M remains a quality option.
- **Calibration (L1):** a gain of about +0.5 points. The fitted offsets average +200 rating points, with a range of −50 to +650.

**Next steps, in order**

1. **Phase 1 step 6:** fp16/int8 size and accuracy, and browser throughput.
2. **Then Phase 2.**

**Conventions**

- **Checks:** `just ready` must pass. It runs `vp check`, the TypeScript tests, the Chromium browser tests, the Python tests and the web build. Run `just` to list all recipes.
- **Tunable values** belong in `packages/config`. After changing defaults, run `vp run @doppelbot/config#export`.
- **Never commit game data.** Results are reported only in aggregate, by band and site.
- **Deploy:** `AWS_PROFILE=personal vp exec sst deploy --stage production`.

A web app where a player enters a Lichess and/or chess.com username. The app builds a
model of how they play (the "me-bot" or doppelbot) on top of Maia-3. It then trains them against a
slightly stronger version of themselves with sparring bots and drills.

Target players: **800 to 1800**.

Core goals:

1. **Fidelity.** The me-bot plays as close to the user as possible, so it can find the puzzles and
   positions the user is likely to get wrong.
2. **Learning shows up in games.** There is no "I understand this now" checkbox. When the user stops making a mistake in
   their games, their me-bot stops making it too, and the related drills retire on their own.
3. **Everything tunable.** Most parameters are settings, so we can experiment without code changes.

## Decisions

| Area        | Decision                                                                                                                                                                                  |
| ----------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Model       | Maia-3 (CSSLab/maia3, Chessformer architecture). Default `maia3-5m` (about 21 MB fp32), with `maia3-23m` as a quality option. Maia-2 is the fallback if the Phase 1 comparison disagrees. |
| License     | AGPL-3.0 to match Maia-3's code and weights. This also covers porting GPL/AGPL code (maia-platform-frontend, chessground, stockfish.js, lichess-puzzler).                                 |
| Frontend    | Vite+ (`vp`) + React + TypeScript. Mirrors the Maia platform frontend (CSSLab/maia-platform-frontend) but without Next.js.                                                                |
| Inference   | ONNX export, run in the browser with onnxruntime-web: WebGPU first, WASM fallback, inside a web worker.                                                                                   |
| Chess logic | chess.js for rules. Stockfish WASM in a web worker for evaluation.                                                                                                                        |
| Storage     | DynamoDB (single table) + S3. The game archives are the source of truth; everything stored is a rebuildable cache.                                                                        |
| Compute     | Ingestion: SQS → Lambda (TypeScript). Fitting: SQS → Fargate task (Python, CPU). Live inference: browser.                                                                                 |
| Infra       | SST v4 on AWS, region `us-west-1`, domain `doppelbot.mekoppe.com` (Route53 zone `mekoppe.com`, same account).                                                                             |
| Accounts    | None at first. Public data is keyed by username. Private data (settings, drill state, bot games) stays in the browser until accounts are added.                                           |
| Repo        | `git@github.com:mmmewk/doppelbot.git`                                                                                                                                                     |

## What Maia-3 looks like inside (this drives the design)

Checked against `CSSLab/maia3` source (`maia3/models.py`, `dataset.py`, `uci.py`, `model_registry.py`):

- **Input:** `tokens[B, 64, 12 * history]` with `history = 8`.
  - Each square gets 12 one-hot piece channels for each of the last 8 positions.
  - When Black is to move, the board is mirrored so the model always sees the position from the mover's side.
  - With no real history, the current position is repeated to fill the 8 slots.
  - There are no castling or en-passant planes, and the released models use no clock input (`include_time_info = False`).
- **Rating is continuous:**
  - `emb(elo) = (elo/5000) * e_low + (1 - elo/5000) * e_high`, where `e_low` and `e_high` are learned 128-d vectors and `elo` is clamped to [0, 5000].
  - Every rating is a point on one line in 128-d space.
- **Where rating enters:** at the input. The `self` and `oppo` embeddings are concatenated onto every square
  token, then everything goes through `token_projection` and 8 transformer blocks. Those blocks use Geometric
  Attention Bias (GAB), a position-dependent attention bias. Rating therefore influences every layer.
- **Outputs:** `forward(tokens, self_elos, oppo_elos) -> (logits_move[B, 4352], logits_value[B, 3], logits_ponder[B])`.
  - Moves: a 64×64 from-to grid, plus 256 promotion entries (8 from-files × 8 to-files × q/r/b/n).
  - Value: win/draw/loss.
  - Ponder: a predicted thinking time. It could be useful later, for example to flag positions where the user tends to think too long.
- **Sizes:**

  | Model | dim_vit | Heads | Checkpoint |
  | ----- | ------- | ----- | ---------- |
  | 5M    | 256     | 8     | 21 MB      |
  | 23M   | 512     | 16    |            |
  | 79M   | 1024    | 32    |            |

  All have 8 blocks and `dim_emb = 128`.

- **No training code is published.** That doesn't matter for our use: fitting a per-user vector or adapter
  only needs the forward pass, a cross-entropy loss over legal moves, and an optimizer.

Consequences:

1. **The export takes the rating vectors (and later LoRA weights) as inputs.** The ONNX graph takes `tokens`,
   `self_emb[B,128]` and `oppo_emb[B,128]`. `e_low` and `e_high` ship as JSON, and the rating-to-vector
   conversion is done in TypeScript. Continuous ratings, "+100" and per-user vectors all use one model file.
2. **The 800 to 1800 range is fully resolved.** There are no rating bands, and "+100" literally means `elo + 100`.
3. **No caching shortcut.** Rating enters at the input, so every candidate vector needs a full forward pass.
   In-browser throughput must be measured (Phase 1).

## The me-bot

### Layers of personalization

| Layer       | What it captures                                                  | Size per user       | Fitted where          |
| ----------- | ----------------------------------------------------------------- | ------------------- | --------------------- |
| Rating      | Which point on the "typical player" line the user is              | 1 number            | Browser               |
| Vector      | Which _kind_ of player: style and tendencies across all positions | 128 floats, ~0.5 KB | Fargate (gradients)   |
| LoRA        | Small edits to how the model judges positions                     | ~50 to 200 KB       | Fargate               |
| Move counts | Exact habits in positions the user reaches repeatedly             | Rebuilt from games  | Browser, at read time |

**LoRA** (Low-Rank Adaptation) works like this:

- The base model's weights stay frozen.
- A few weight grids `W` get a small trainable patch: `W + A·B`. For a 256×256 grid, `A` is 256×r and `B` is r×256, with r around 4 to 8.
- Only `A` and `B` are trained, about 4k numbers per grid instead of 65k.
- The patches are passed to the ONNX model as inputs, the same way vectors are.

The vector shifts _who_ is playing. LoRA can change _what the model has learned_, for example "this player no longer
falls for knight forks". Concept-level change like that may be too much for 128 numbers. Phase 2b decides whether LoRA is needed.

**Off-line vectors:** the base model has only ever seen vectors on the rating line. A vector that drifts far off
it may fit past games and still predict new ones badly. So fits carry a penalty for drifting from the calibrated
rating point. Held-out games pick how strong that penalty is.

### Move counts blended with the model

For a position the user has reached before, combine their actual history with the model's prediction:

```
P(move | pos) = (α · P_model(move | pos) + c_move) / (α + Σ c)
c_move = Σ over occurrences of  sourceWeight · 0.5^(age / halfLife)
```

- **Unseen positions:** this is just the model.
- **Frequently reached positions (mostly openings):** the user's real habits dominate.
- **Position key:** the board, side to move, castling and en-passant rights, ignoring move counters. Transpositions merge.
- **Counts are computed at read time from raw occurrences.** That's why `halfLife`, `α` and source weights are instant settings, not recomputes.
- **Cross-check:** Lichess's per-player opening explorer gives the same counts for Lichess games.

### Evidence and recency

Every move the user plays is evidence, weighted by where it came from and how old it is:

| Source                           | Default weight | Notes                                                                                               |
| -------------------------------- | -------------- | --------------------------------------------------------------------------------------------------- |
| Rated games (Lichess, chess.com) | 1.0            |                                                                                                     |
| Casual games                     | 0.8            |                                                                                                     |
| Games vs our bots                | 0.3            | They count, but less: beating a weakness against the bot takes more repetitions than in real games. |

- Moves played with under `minClockSeconds` or under `minClockFraction` of the starting clock are excluded.
  Clocks are stored raw, so this threshold is also a setting.
- The source weights apply to move counts at read time and to the loss when fitting vectors or LoRA.

## Objective and evaluation

The primary metric is the **log-likelihood of the user's actual moves on held-out recent games**.
Top-1 accuracy is reported but secondary, because finding struggle positions depends on well-calibrated probabilities.
Recency weighting is built into the test, because the test is "predict my latest games".

Results are always reported in slices:

- **Known positions** (the user has reached them before): tests the move counts and repertoire memory.
- **New positions**: tests the model's generalization, which is what matters for puzzles.
- **By time control** and **by game phase**.

Puzzle-relevant check: in held-out positions where the me-bot puts high probability on an eval-losing move,
how often did the user actually play it?

### Experiment ladder (offline in Python first)

The baseline is the stated rating fed in directly.

| Level  | What gets fitted                                                | How                                                                  |
| ------ | --------------------------------------------------------------- | -------------------------------------------------------------------- |
| L0     | Stated rating, as-is                                            | –                                                                    |
| L1     | Best continuous `self` rating (and optionally an `oppo` offset) | 1-D/2-D search, forward passes only                                  |
| L1+C   | L1 + move counts                                                | Tune `α` and `halfLife`                                              |
| L2(+C) | Free 128-d `self` vector, starting from L1                      | Adam with a penalty for drifting off the L1 point; base model frozen |
| L3(+C) | L2 + LoRA                                                       | Adam                                                                 |

Questions to answer:

1. **Data needs:** how many games before each level beats the one below it? (Maia4All reports plain fine-tuning
   failing at a few hundred games. Many 800 to 1800 accounts have thousands.)
2. **Is LoRA needed?** How much do L2 and then L3 gain over L1+C?
3. **Theme tracking (the "learning shows up in games" test):**
   1. Split a user's history into time windows.
   2. Tag the user's Stockfish-confirmed mistakes in each window by theme (fork, hanging piece, back rank…) using the lichess-puzzler tagger.
   3. Fit the me-bot on games up to each window.
   4. Check whether its predicted mistake rate per theme follows the user's real rate as that rate changes.

   Run this for L2 and L3. This decides whether self-retiring drills are trustworthy.

4. **Noise floor:** fit the same window twice on random halves of its games. The distance between the two
   results is the noise floor. Changes smaller than that are never reported or acted on.
5. **Skill tracks rating:** does the along-line component of each window's vector (see Progression) follow the
   user's real rating history?
6. **Window size:** how many games per window (or what fit-time half-life) balances responsiveness against noise?
7. **Does `oppo` matter?** Does fitting how the user plays against different opponents help?
8. **Browser fitting:** with the 5M model, is any of L2 fast enough to fit in the browser?

Test data: a dozen real accounts across 800 to 1800, on both sites.

## Progression over time

Vectors are fitted per time window (default: rolling ~200 games). Each one is tagged with the model ID and the
hash of the config that produced it. All windows start from the same anchor point with the same penalty, so they are comparable.
(Starting each window from the previous one would bias the results toward "no change".)

Each change between windows is split in two:

- **Along the rating line = skill:** the projection of Δv onto `e_low − e_high`, converted to rating points.
  For example: "your play moved +80 rating points this quarter".
- **Off the line = style:** the remaining component, meaning change that rating doesn't explain.
  It includes noise, so it is only shown when it exceeds the noise floor.
- **Behavioural distance:** the average KL divergence between two vectors' move distributions on a fixed
  reference set of positions. This is the honest "how differently do you play now" number.
  Euclidean distance is only a cheap stand-in for search.
- **"You play like…":** the nearest stored player vectors, found by brute-force search (no vector database needed).

## Bots

A bot is one or more vectors (plus the user's move counts where relevant) and a rule for choosing moves:

| Bot                        | Name   | Uses                               | Move-choice rule                                                               |
| -------------------------- | ------ | ---------------------------------- | ------------------------------------------------------------------------------ |
| Speed                      | Doppel | `me`                               | Sample from `P(move \| me)` at a temperature setting                           |
| Variety                    | Zephyr | `me`, `peer = emb(user rating)`    | `P_peer / P_me`, restricted to moves within `evalMarginCp` of Stockfish's best |
| Accuracy: traps            | Loki   | `me` (to predict the user's reply) | Plausible + sound + likely to draw a bad reply; at most `maxTrapsPerGame`      |
| Accuracy: planted blunders | Loki   | `peer`                             | Common errors at the user's level that the user is unlikely to punish          |
| Reference ("+100")         | –      | `me + (emb(r+100) − emb(r))`       | Diff analysis and drills                                                       |
| Baseline                   | Maia   | `emb(user rating + stretch)`       | Unpersonalized Maia-3 (Phase 1.5)                                              |

Bots of other people (a friend, a club, any set of games) are just other players' vectors.

## The drill loop

1. **Find weaknesses.** Drill sources, highest confidence first:
   1. **Repertoire leaks:** positions the user reaches often where their most common move loses eval per Stockfish.
      These come from real counts, not model guesses, so the sample is large and the user will meet them again.
   2. **Model-predicted struggles:** positions where the me-bot is likely to play an eval-losing move,
      verified by Stockfish and grouped by theme. Includes `me` vs "+100" diff analysis.
2. **Drill.** Each puzzle is a position, a set of accepted answers (moves within `acceptMarginCp` of best), the trap move and a short explanation.
   A varied set is picked by clustering position embeddings. Missed puzzles come back on a spaced-repetition schedule.
3. **Play.** Real games, plus sparring bots that steer into weak positions on purpose. The traps bot does this by design.
4. **Update.** New games update the move counts at once, and the vector or LoRA on the next refit.
5. **Retire automatically.** A drill retires when the refreshed me-bot's probability of the mistake drops below `retireProbability`
   _and_ the change exceeds the noise floor. It comes back if the user slips again.

Every "retired" verdict shows the evidence behind it: which games, from which sources.

## Configuration

Every parameter lives in one typed config (`packages/config`: a zod schema plus defaults). It is shared by the app,
the Lambdas and Python (via an exported `defaults.json`). A settings page exposes it, with an "experimental" section.

**Rule:** apply parameters as late as possible.

- **Read-time parameters** change instantly in the browser.
- **Fit-time parameters** require a refit. Fitted results are keyed by `(modelId, configHash)`, so old and new results can sit side by side.
- **Every experiment run records the full config it used.**

| Parameter                                     | Default                         | Applied                                |
| --------------------------------------------- | ------------------------------- | -------------------------------------- |
| `recency.halfLifeDays`                        | 120                             | Read time (counts)                     |
| `blend.alpha`                                 | 5                               | Read time                              |
| `evidence.weights`                            | rated 1.0, casual 0.8, bot 0.3  | Read time (counts); fit time (vectors) |
| `filter.minClockSeconds` / `minClockFraction` | 10 / 0.10                       | Read and fit time                      |
| `model.id` / `model.precision`                | `maia3-5m` / decided in Phase 1 | Load time                              |
| `fit.windowGames`                             | 200                             | Fit time                               |
| `fit.driftPenalty`                            | Tuned in Phase 2b               | Fit time                               |
| `fit.holdoutFraction`                         | 0.2                             | Fit time                               |
| `play.maxBotRating`                           | 2600                            | Play time                              |
| `bots.speed.temperature`                      | 1.0                             | Play time                              |
| `bots.variety.evalMarginCp`                   | 25                              | Play time                              |
| `bots.traps.maxPerGame`                       | 1                               | Play time                              |
| `drills.acceptMarginCp`                       | 30                              | Read time                              |
| `drills.retireProbability`                    | 0.1                             | Read time                              |

Settings live in the browser (IndexedDB) until accounts exist. Requests for fits with non-default configs are limited to allowed ranges.

## Data and storage

**Principle:** the game archives are the source of truth. DynamoDB and S3 hold rebuildable caches of whatever is slow to recompute.

**Writes are trusted:** the server fetches public games itself, by username, and the server computes all fits.
Browsers never upload derived data for a username.

**DynamoDB (single table):**

| PK                         | SK                                        | Contents                                                                |
| -------------------------- | ----------------------------------------- | ----------------------------------------------------------------------- |
| `PLAYER#<site>:<username>` | `META`                                    | Ratings history, last fetched month, game counts                        |
| `PLAYER#…`                 | `MONTH#<yyyy-mm>`                         | S3 key of that month's games and occurrences, fetched-at                |
| `PLAYER#…`                 | `VEC#<modelId>#<configHash>#<windowEnd>`  | Vector (float32, base64), moves used, held-out metrics, noise radius    |
| `PLAYER#…`                 | `LORA#<modelId>#<configHash>#<windowEnd>` | S3 key of the adapter, metrics                                          |
| `PLAYER#…`                 | `JOB#<jobId>`                             | Ingestion or fit job status                                             |
| `POS#<positionKey>`        | `SF#<depth>`                              | Stockfish eval and top moves. Shared across all users; openings repeat. |

**S3 (data bucket):**

- `games/<site>/<username>/<yyyy-mm>.jsonl.gz`: normalized games (moves, clocks, ratings, time control, result).
- `occurrences/<site>/<username>/<yyyy-mm>.bin`: compact (positionKey, move, timestamp, source, clock) rows.
  The browser downloads these to build move counts.
- `adapters/…`: LoRA weights.

**Browser (IndexedDB):** settings, drill and spaced-repetition state, games vs our bots, downloaded data.

Games vs our bots are only in the browser for now, so they feed the move counts but not server fits.
They feed server fits once accounts exist, because without accounts anyone could poison anyone's me-bot.
"Log in with Lichess" (OAuth PKCE) is the easy first option.

## Repo layout

```
doppelbot/
├─ sst.config.ts
├─ infra/              web.ts, storage.ts (Dynamo + buckets), ingest.ts, fit.ts
├─ apps/web/           Vite+ React TS app
├─ packages/
│  ├─ config/          zod schema + defaults (exported to JSON for Python)
│  ├─ maia/            tokenization, move vocab, ONNX session, worker, vector math
│  ├─ games/           Lichess + chess.com fetch, PGN/clock parsing, position keys, occurrences
│  ├─ counts/          recency-weighted counts + blending
│  └─ engine/          Stockfish worker wrapper
├─ functions/          Lambda handlers (ingestion, API)
├─ python/             uv project: export, test data, experiments, Fargate fit job
└─ docs/PLAN.md
```

Both chess.com and Lichess return permissive CORS headers (checked 2026-10-24). Ingestion still runs on the server, so stored data is trustworthy.

Required browser headers: onnxruntime-web and Stockfish need multithreaded WASM (SharedArrayBuffer).
That needs `Cross-Origin-Opener-Policy: same-origin` and `Cross-Origin-Embedder-Policy: require-corp`
on both the CloudFront distribution (response headers policy) and the Vite dev server.

## Phases

### Model and effort for Claude Code

Each phase (and some steps within phases) lists the recommended Claude model and effort level. Prices as of 2026-10:

| Model      | Input / output per 1M tokens | Use for                                                                                                    |
| ---------- | ---------------------------- | ---------------------------------------------------------------------------------------------------------- |
| Fable 5.1  | $10 / $50                    | Escalation only: stuck on hard debugging, or research decisions that everything downstream depends on      |
| Opus 5.5   | $4 / $20                     | Numerical correctness, ML methodology, algorithm design. Default effort is `medium`, so set it explicitly. |
| Sonnet 5.5 | $2 / $10                     | Scaffolding, infra, UI, porting existing code, plumbing                                                    |
| Haiku 5.5  | $0.10 / $0.50                | Subagents for codebase search and reading, monitoring long runs, mechanical edits                          |

Rules of thumb:

- **Escalate on failure, not up front.** After two failed attempts at the same problem, raise effort one level first. If that still fails, raise the model one tier.
- **One session per phase step.** Use this plan as the handoff document, so each session starts with a clean, small context.
- **Plan on the stronger model, execute on the cheaper one.** Design the approach at the listed level. When most of the remaining work is mechanical, drop to Sonnet 5.5 for it.
- **Delegate reading to cheap subagents.** Codebase sweeps and doc reading go to Haiku 5.5 (Explore) subagents.
- **Review before closing a phase.** Run `/code-review` on Opus 5.5 `high` before a phase is marked done.

### Phase 0: Scaffold

**Model:** Sonnet 5.5, `high`. SST v4 and Vite+ are newer than most training data, so expect doc lookups.

- Vite+ monorepo, React TS app, and SST v4 `StaticSite` at `doppelbot.mekoppe.com`, deployed with `AWS_PROFILE=personal`.
- DynamoDB table, data bucket, and a model bucket behind CloudFront with immutable caching.
- `packages/config` with the schema and defaults.
- A `python/` uv project with maia3 (pinned to a commit) and maia2 (for the comparison).

### Phase 1: Model choice + parity (risk spike; blocks everything else)

**Model, by step:**

| Step                                        | Model and effort  |
| ------------------------------------------- | ----------------- |
| 1. Model comparison                         | Opus 5.5 `high`   |
| 2. Export                                   | Opus 5.5 `xhigh`  |
| 3 to 5. Test data, TypeScript, parity tests | Opus 5.5 `high`   |
| Parity mismatch not found after two rounds  | Fable 5.1 `high`  |
| 6. Size and speed                           | Sonnet 5.5 `high` |

1. **Maia-3 vs Maia-2, in Python.** On held-out moves from real 800 to 1800 accounts, compare:
   - Maia-3 5M and 23M
   - Maia-2 rapid/blitz

   Measure log-likelihood and top-1 accuracy, with the stated rating and with L1. Confirm the default model.

2. **Export.** Wrap `MAIA3Model` so it takes embedding inputs (with a LoRA input path stubbed in), and export
   to ONNX with a dynamic batch size. Also export `moves.json` and `elo_emb.json` (`e_low`, `e_high`).
3. **Python test data.** About 500 positions from real games, deliberately covering:
   - Black to move, promotions and under-promotions, castling, en passant
   - Positions with fewer than 8 previous positions

   For each, at several rating pairs, record: tokens, legal mask, top-k probabilities, WDL.

4. **TypeScript.** Write tokenization with 8-position history (chess.js), mirroring, the legal mask,
   masked softmax and WDL decoding.
5. **Parity tests.** Tokens must match exactly. Probabilities must match within 1e-4:
   - in Node first
   - then in the browser on WebGPU and WASM (Vitest browser mode)
6. **Size and speed.** Compare fp16 and int8 against fp32: top-1 agreement and KL divergence. Measure positions per
   second in the browser on WebGPU and WASM.

### Phase 1.5: Play standard Maia (no personalization)

**Model:** Sonnet 5.5 `medium`. This is UI work on top of the parity-tested `packages/maia`.

**Goal:** a deployed site with a landing page that explains the app and its trainers, and a play page where you
play a game against unpersonalized Maia-3 at your rating. It proves that model loading, tokenization with real
history, and inference all work in a real browser. It also gives a baseline that the later me-bot can be compared
against. It uses no backend: no DynamoDB, SQS or Fargate, and no model training.

- **Routing and top nav:**
  - Two routes: Home (`/`) and Play (`/play`), using React Router in library mode.
  - A top nav on every page with the doppelbot wordmark and Home and Play tabs. The active tab is highlighted.
  - The `StaticSite` needs an SPA fallback so a direct load or refresh of `/play` serves `index.html`.
- **Landing page (Home):**
  - **Hero:** one or two sentences on what doppelbot is: personal chess trainers built from a model of how _you_ play,
    trained on your own Lichess and chess.com games.
  - **Four trainer cards.** Each has a name, a one-line role, a paragraph, a recommended time control and a board graphic.
    In this phase only Maia is playable; the other three carry a "Coming soon" badge.

    | Trainer    | Role               | Copy                                                                                                                                                                                                                                                                                                           | Graphic                                                                                                                                                       |
    | ---------- | ------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
    | **Doppel** | Your doppelgänger  | Plays chess like you and prefers your openings. Excels at helping you practice the lines you already know and at improving your speed. Play at **fast** time controls to sharpen your pattern recognition.                                                                                                     | The most common London System setup: `1.d4 d5 2.Bf4 Nf6 3.e3 e6 4.Nf3 c5 5.c3 Nc6 6.Nbd2 Bd6 7.Bg3 O-O 8.Bd3`.                                                |
    | **Zephyr** | Your companion     | Named for the Greek god of the west wind (and a dog). Follows you around but explores from a different perspective: knows the same tactics you do, and like a gentle wind blows you into new, unexpected lines. Play at **medium** time controls to practice responding when the game deviates from your plan. | An uncommon Queen's Gambit line, the Albin Countergambit: `1.d4 d5 2.c4 e5 3.dxe5 d4 4.Nf3 Nc6`.                                                              |
    | **Loki**   | Your nemesis       | Plays the traps you are likely not to see, but also makes the blunders you are likely not to punish. Play at **longer** time controls to practice spotting the blunders and advantages you normally miss.                                                                                                      | The Blackburne Shilling Gambit smothered mate: `1.e4 e5 2.Nf3 Nc6 3.Bc4 Nd4 4.Nxe5 Qg5 5.Nxf7 Qxg2 6.Rf1 Qxe4+ 7.Be2 Nf3#`, with an arrow on the mating move. |
    | **Maia**   | The average player | A human-like model of a typical player at a given rating, from the [Maia project](https://www.maiachess.com). Maia is the base model that Doppel, Zephyr and Loki are all built on. Playable now.                                                                                                              | The starting position, or a neutral middlegame.                                                                                                               |

  - **Board graphics:** static, non-draggable `react-chessboard` boards. Each is defined as a SAN move list in one
    `positions.ts` module and turned into a FEN with chess.js at load time, so no FEN is written by hand.
    A unit test checks that every line is legal and that the Loki line ends in checkmate.
  - **Footer CTA:** a "Let's get started" button at the bottom of the page that opens the profile modal.
- **Profile modal:**
  - Fields: chess.com username and Lichess username (at least one is required), plus an optional name. The name is used for the greeting and defaults to a username.
  - Light validation on submit: look up each entered username (the rating endpoints below) and show an inline error if it doesn't exist.
  - Save to `localStorage` as `doppelbot.profile` = `{ name?: string, chesscom?: string, lichess?: string, savedAt }`. Wrap every read and write in try/catch; if storage is unavailable the profile lives in memory for the session.
  - On save, navigate to `/play`.
  - Nothing is uploaded and no model is trained yet. The modal says so in one line.
- **Play page:**
  - If no profile is saved, show a short prompt with the same "Let's get started" button and modal.
  - With a profile, greet the user by name ("Ready, {name}?"; with no name set, use the Lichess username if both are set) and start a game against Maia.
  - An "Edit profile" link reopens the modal with the saved values.
- **Model hosting:**
  - Upload `model.onnx`, `moves.json` and `elo_emb.json` to the `ModelBucket`, served at `/models/maia3-5m/...` through the router.
  - `pnpm run upload-models` (`scripts/models.sh`) exports the files if needed and runs `aws s3 sync` with `AWS_PROFILE=personal` and immutable cache headers.
  - In dev, a Vite middleware in `apps/web/vite.config.ts` serves the same paths straight from `python/artifacts/`. (Copying them into `public/` would also ship 21 MB in every site build.)
  - Cache the model in the browser (Cache API) so reloads don't download it again.
- **Rating lookup (browser only, using the saved profile's usernames):**
  - Lichess: `GET https://lichess.org/api/user/{username}`, reading `perfs.blitz` and `perfs.rapid`.
  - chess.com: `GET https://api.chess.com/pub/player/{username}/stats`, reading `chess_blitz.last.rating` and `chess_rapid.last.rating`.
  - Pick the time control with the most games across both sites. Show it, and allow a manual rating if the lookup fails.
  - Maia was trained on Lichess games, so chess.com ratings are converted to the Lichess scale (`lichessEquivalent` in `apps/web/src/ratings.ts`).
    - The conversion uses [ChessGoals' rating comparison](https://chessgoals.com/rating-comparison/), updated July 2026, built from players with RD < 150 on both sites.
    - Chess.com blitz maps straight to Lichess blitz, with about 50% of players within ±70.
    - Chess.com rapid goes rapid → chess.com blitz → Lichess, because ChessGoals has no direct rapid mapping.
    - Other sources disagree by up to about 100 points at 1200 (chess.com's own blog, Chessiro), so treat it as an estimate. Phase 2's L1 calibration replaces it with a fitted rating anyway.
- **Game setup:**
  - Bot rating = your rating + a stretch slider. The default comes from a new `play.stretchRating` setting in `packages/config`, default `+100`, range −300 to +500.
  - The Maia inputs are `self_elo` = bot rating and `oppo_elo` = your rating.
  - Both are capped at `play.maxBotRating` (default 2600), because Maia's training data thins out above that.
  - You choose white, black or random.
- **Board:**
  - Use `react-chessboard` with chess.js, which `packages/maia` already uses.
  - Support legal-move highlighting, promotion choice, undo of your last move, resign, and new game.
  - Detect game end: mate, stalemate, threefold repetition, the 50-move rule, and insufficient material.
- **Bot moves:**
  - Run inference in a Web Worker so the UI stays responsive.
  - Always pass the real 8-position history from the game so far.
  - Sample from the masked softmax, with the temperature from `packages/config` (default 1). Also offer a "top move" toggle for debugging.
  - Add a short artificial delay (for example 300 to 800 ms) so the moves don't appear instantly.
  - Use WebGPU when it's available, otherwise WASM. Show which one is in use.
- **Debug panel (collapsible):**
  - The bot's top 5 moves with their probabilities, for the current position.
  - The WDL from the side to move's perspective.
  - Backend, inference time in milliseconds, and the model ID.
- **Done when:**
  - On production (`doppelbot.mekoppe.com`), the landing page shows the four trainers with their board graphics.
  - "Let's get started" opens the modal; saving a real Lichess and/or chess.com username lands you on Play, greeted by name, in a game against Maia at your rating + 100. A reload keeps the profile.
  - Loading `/play` directly works.
  - The page shows "Cross-origin isolated: yes" and the debug panel shows the backend in use.
  - A Vitest browser test plays a few bot moves from a fixture position and checks that each one is legal.

Nothing here is thrown away later: Phase 2 replaces the stated rating with the calibrated (L1) one and adds the
user's move counts, Phase 3 swaps in fitted vectors, and Phase 4 adds Stockfish and polishes this same page.

### Phase 2: Ingestion, counts, calibration

**Model:** Sonnet 5.5 `medium` for ingestion, parsers and UI; Opus 5.5 `high` for position keys, recency-weighted counts, blending and L1 calibration (subtle bugs there silently corrupt every downstream metric).

- **Ingestion (SQS → Lambda):**
  - Fetch new months from Lichess (NDJSON, `clocks=true`) and chess.com (monthly archives, `%clk`).
  - Normalize, then write games and occurrences to S3 and metadata to DynamoDB.
  - "Refresh" fetches only missing months.
- **Browser:**
  - Download occurrences and build recency-weighted counts.
  - Run L1 calibration on the user's games.
  - Show effective rating per time control, held-out metrics, and the known/new slices.

### Phase 2b: Personalization experiments

**Model:**

- Experiment design and reading the results: Opus 5.5 `xhigh`. Use Fable 5.1 `high` for the final go/no-go decisions on LoRA and the noise floor.
- Training and evaluation code: Opus 5.5 `high`.
- Watching long runs: Haiku 5.5 `low`.

Run the experiment ladder above in Python, against the data that ingestion stored in S3. Decide:

- whether LoRA is needed
- the window size and default penalty strength
- the noise floor
- what can run in the browser

### Phase 3: Fitting service

**Model:** Sonnet 5.5 `high`. This is infra plus packaging the already-validated Phase 2b fit code.

- SQS → Fargate (Python + maia3, CPU) fits vectors (and LoRA, if 2b says so) per window.
- Results are written to DynamoDB and S3, keyed by `(modelId, configHash)`.
- A refit is triggered after ingestion brings in new games.

### Phase 4: Play + Speed bot

**Model:** Sonnet 5.5 `medium` for porting the board UI and game flow; Sonnet 5.5 `high` for the Stockfish and inference workers.

- Board UI ported from maia-platform-frontend.
- Stockfish worker.
- Me-bot sampling (counts + model).
- Bot games are saved locally as weighted evidence.
- Optional opening book from a Lichess study PGN.

### Phase 5: Variety bot, then Accuracy bot

**Model:** Opus 5.5 `high` for move-scoring and trap-search algorithms; Sonnet 5.5 `medium` for bot UI and outcome logging.

- Traps and planted-blunder modes.
- Log each outcome: fell for it, punished it, or missed it.

### Phase 6: Drill loop

**Model:** Opus 5.5 `high` for leak detection, clustering and retirement logic; Sonnet 5.5 `medium` for tagger integration, spaced repetition and drill UI.

- Repertoire leaks first, then model-predicted struggles (themes via the lichess-puzzler tagger).
- Spaced repetition.
- Retirement driven by the refreshed me-bot, gated on the noise floor.

### Phase 7: Progression + accounts

**Model:** Opus 5.5 `high` for skill/style decomposition and behavioural distance; Sonnet 5.5 `medium` for OAuth, sync and progression UI.

- Progression view: skill trajectory in rating points, style drift, "you play like…".
- Lichess OAuth so games vs our bots and settings can sync and feed server fits.

## Risks to verify early

1. Maia-3 actually beats Maia-2 for 800 to 1800 players on their own games.
2. ONNX export parity, including GAB attention and the promotion head.
3. Tokenization, history and move-index parity between TypeScript and Python.
4. Model size and speed versus accuracy under fp16 and int8; in-browser throughput.
5. Calibration and counts actually improve held-out prediction.
6. Personalized vectors (or LoRA) beat L1+C with a realistic game count.
7. The me-bot's per-theme mistake rate follows the user's real improvement (theme tracking). Self-retiring drills depend on this.
8. The noise floor is small enough that real progress is detectable at a usable window size.
