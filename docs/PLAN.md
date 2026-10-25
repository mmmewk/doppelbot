# doppelbot: Plan

A web app where a player enters a Lichess and/or chess.com username. The app builds a
model of how they play (the "doppelbot") on top of Maia-3, then trains them against a
slightly stronger version of themselves with sparring bots and drills.

Target players: **800 to 1800**.

Long-term goal for personalization: the doppelbot plays **as close to the user as
possible**, so it can find the puzzles and positions the user is likely to get wrong.

## Decisions

| Area | Decision |
|---|---|
| Model | Maia-3 (CSSLab/maia3, Chessformer architecture). Default `maia3-5m` (about 21 MB fp32), with `maia3-23m` as a quality option. Maia-2 is the fallback if the Phase 1 comparison disagrees. |
| License | AGPL-3.0 to match Maia-3's code and weights. This also covers porting GPL-3.0 code (maia-platform-frontend, chessground, stockfish.js). |
| Frontend | Vite+ (`vp`) + React + TypeScript. Mirrors the Maia platform frontend (CSSLab/maia-platform-frontend) but without Next.js. |
| Inference | ONNX export, run in the browser with onnxruntime-web: WebGPU first, WASM fallback, inside a web worker. |
| Chess logic | chess.js for rules. Stockfish WASM in a web worker for evaluation. |
| Infra | SST v4 on AWS, region `us-west-1`, domain `doppelbot.mekoppe.com` (Route53 zone `mekoppe.com`, same account). |
| Data | Version 1 is local-first: games, profiles and drill state live in IndexedDB. No accounts. |
| Repo | `git@github.com:mmmewk/doppelbot.git` |

## What Maia-3 looks like inside (this drives the design)

Checked against `CSSLab/maia3` source (`maia3/models.py`, `dataset.py`, `uci.py`, `model_registry.py`):

- **Input:** `tokens[B, 64, 12 * history]` with `history = 8`.
  - Each square gets 12 one-hot piece channels for each of the last 8 positions.
  - When Black is to move, the board is mirrored so the model always sees the position from the mover's side.
  - With no real history, the current position is repeated to fill the 8 slots.
  - There are no castling or en-passant planes, and the released models use no clock input (`include_time_info = False`).
- **Rating is continuous:**
  - `emb(elo) = (elo/5000) * e_low + (1 - elo/5000) * e_high`, where `e_low` and `e_high` are learned 128-d vectors and `elo` is clamped to [0, 5000].
  - So every rating is a point on one line in 128-d space.
- **Where rating enters:** at the input. The `self` and `oppo` embeddings are concatenated onto every square
  token, then everything goes through `token_projection` and 8 transformer blocks. Those blocks use Geometric
  Attention Bias (GAB), a position-dependent attention bias. Rating therefore influences every layer.
- **Outputs:** `forward(tokens, self_elos, oppo_elos) -> (logits_move[B, 4352], logits_value[B, 3], logits_ponder[B])`.
  - Moves: a 64×64 from-to grid, plus 256 promotion entries (8 from-files × 8 to-files × q/r/b/n).
  - Value: win/draw/loss.
  - Ponder: a predicted thinking time. It could be useful later, for example to flag positions where the user tends to think too long.
- **Sizes:**

  | Model | dim_vit | Heads | Checkpoint |
  |---|---|---|---|
  | 5M | 256 | 8 | 21 MB |
  | 23M | 512 | 16 | |
  | 79M | 1024 | 32 | |

  All have 8 blocks and `dim_emb = 128`.
- **No training code is published.** That doesn't matter for our use: fitting a per-user vector or adapter
  only needs the forward pass, a cross-entropy loss over legal moves, and an optimizer.

Consequences:

1. **The export takes the rating vectors as inputs.** The ONNX graph takes `tokens`,
   `self_emb[B,128]` and `oppo_emb[B,128]`. `e_low` and `e_high` ship as JSON, and the rating-to-vector
   conversion is done in TypeScript. Continuous ratings, "+100" and per-user vectors all use one model file.
2. **The 800 to 1800 range is fully resolved.** There are no rating bands, so 800 and 1050 are different players,
   and "+100" literally means `elo + 100`.
3. **No caching shortcut.** Rating enters at the input, so every candidate vector needs a full forward pass.
   The 5M model is small, but in-browser throughput must be measured (Phase 1).

## Personalization: a player is a vector

A player is a 128-float vector used as the `self` embedding. A bot is one or more vectors plus a
rule for choosing moves:

| Bot | Vectors | Move-choice rule |
|---|---|---|
| Speed | `me` | Sample from `P(move \| me)` |
| Variety | `me`, `peer = emb(user rating)` | `P_peer / P_me`, restricted to moves within ~25cp of Stockfish's best |
| Accuracy: traps | `me` (to predict the user's reply) | Plausible + sound + likely to draw a bad reply |
| Accuracy: planted blunders | `peer` | Common errors at the user's level that the user is unlikely to punish |
| Reference ("+100") | `me + (emb(r+100) − emb(r))` | Diff analysis and drills |

A vector is about 0.5 KB. Bots of other people (a friend, a club, any set of games) are just more vectors.

**Off-line vectors:** the base model has only ever seen vectors on the rating line. A vector that drifts far off
it may fit the user's past games and still predict new ones badly. So personalized vectors carry a penalty
for drifting from the calibrated rating point, and held-out games pick how strong that penalty is.
If vectors alone hit a ceiling, the next step is a low-rank adapter (LoRA) on `token_projection` and/or the
attention projections. That is still far from a full fine-tune.

### Objective

Maximize the log-likelihood of the user's actual moves on held-out games. Top-1 accuracy is reported but is
secondary: finding struggle positions depends on well-calibrated probabilities. The puzzle-relevant check is this:
among held-out positions where the doppelbot puts high probability on an eval-losing move, how often did the user
actually play it?

### Experiment ladder (run offline in Python first)

Each level is evaluated on the user's held-out games: the most recent 20%, with moves in severe time trouble filtered out.
The baseline is the stated rating fed in directly.

| Level | What gets fitted | How | Where it can run |
|---|---|---|---|
| L0 | Stated rating, as-is | – | – |
| L1 | Best continuous `self` rating (and optionally an `oppo` offset) | 1-D/2-D search | Browser (forward passes only) |
| L2 | Free 128-d `self` vector, starting from L1 | Gradient descent (Adam) with a penalty for drifting off the L1 point; base model frozen | Python (local, then Fargate) |
| L3 | L2 plus a LoRA | Gradient descent | Python |

Questions the experiments answer:

1. How many games before L2 or L3 beats L1? (Maia4All reports plain fine-tuning failing at a few hundred games.
   Many 800 to 1800 accounts have thousands of games, which is tens of thousands of moves.)
2. How much does L2 gain over L1, and does L3 gain enough on top to justify it?
3. Does fitting `oppo` (how the user plays against different opponents) matter?
4. Is in-browser gradient-free fitting of L2 fast enough with the 5M model? If not, use Fargate.

Test data: a dozen real accounts across 800 to 1800, on both sites.

### Server path (if L2+ can't run in the browser)

1. The browser uploads cleaned positions to S3.
2. SST `Queue` (SQS) → SST `Task` (Fargate, Python 3.12 + maia3, CPU).
3. The job writes the vector or adapter to S3 and status to DynamoDB.
4. The browser polls and downloads the result.

The result is only adopted if it beats L1 on held-out games.

## Repo layout

```
doppelbot/
├─ sst.config.ts
├─ infra/              web.ts, models.ts, (later) personalize.ts
├─ apps/web/           Vite+ React TS app
├─ packages/
│  ├─ maia/            tokenization, move vocab, ONNX session, worker, vector math
│  ├─ games/           Lichess + chess.com fetch, PGN/clock parsing, filters
│  └─ engine/          Stockfish worker wrapper
├─ python/             uv project: export, test data, personalization experiments
└─ docs/PLAN.md
```

Both chess.com and Lichess return permissive CORS headers (checked 2026-10-24), so no proxy is needed.

Required browser headers: onnxruntime-web and Stockfish need multithreaded WASM (SharedArrayBuffer).
That needs `Cross-Origin-Opener-Policy: same-origin` and `Cross-Origin-Embedder-Policy: require-corp`
on both the CloudFront distribution (response headers policy) and the Vite dev server.

## Phases

### Phase 0: Scaffold

- Vite+ monorepo, React TS app, and SST v4 `StaticSite` at `doppelbot.mekoppe.com`, deployed with `AWS_PROFILE=personal`.
- A model bucket behind CloudFront with immutable caching.
- A `python/` uv project with maia3 (pinned to a commit) and maia2 (for the comparison).

### Phase 1: Model choice + parity (risk spike; blocks everything else)

1. **Maia-3 vs Maia-2, in Python.** On held-out moves from real 800 to 1800 accounts, compare:
   - Maia-3 5M and 23M
   - Maia-2 rapid/blitz

   Measure top-1 accuracy and log-likelihood, with the stated rating and with L1 calibration. Confirm the default model.
2. **Export.** Wrap `MAIA3Model` so it takes embedding inputs, and export to ONNX with a dynamic batch size.
   Also export `moves.json` (the vocabulary/index mapping) and `elo_emb.json` (`e_low`, `e_high`).
3. **Python test data.** About 500 positions from real games, deliberately covering:
   - Black to move, promotions and under-promotions, castling, en passant
   - Positions early in the game with fewer than 8 previous positions

   For each, at several rating pairs, record: tokens, legal mask, top-k probabilities, WDL.
4. **TypeScript.** Write tokenization with 8-position history (chess.js), mirroring, the legal mask,
   masked softmax and WDL decoding.
5. **Parity tests.** Tokens must match exactly. Probabilities must match within 1e-4:
   - in Node first
   - then in the browser on WebGPU and WASM (Vitest browser mode)
6. **Size and speed.** Compare fp16 and int8 against fp32: top-1 agreement and KL divergence. Measure positions per
   second in the browser on WebGPU and WASM.

### Phase 2: Game ingestion + calibration (L1)

- **Fetching:**
  - Lichess: NDJSON stream with `clocks=true`.
  - chess.com: monthly archives, `%clk` from the PGN.
  - Normalize both to one `Game` type and cache in IndexedDB.
- **Filter:** drop moves played with under 10s or under 10% of the starting clock.
  Maia-3 is a single model, so time control is just a filter and profile dimension.
- **Calibration:** fit the L1 rating on training games and report the gain over L0 on held-out games.
- **UI:** username → effective rating per time control, likelihood/accuracy, and the games used.

### Phase 2b: Personalization experiments (L2, L3)

Run the experiment ladder above in Python on real accounts. Then decide what runs in the browser and whether the Fargate path is needed.

### Phase 3: Play + Speed bot

- Board UI ported from maia-platform-frontend.
- Stockfish worker.
- Doppelbot sampling.
- Optional opening book from a Lichess study PGN.

### Phase 4: Variety bot, then Accuracy bot

- Traps and planted-blunder modes.
- Log each outcome: fell for it, punished it, or missed it.

### Phase 5: Diff analysis + drills

- Compare `me` with "+100", verify with Stockfish, and group by theme.
- Turn flagged positions into puzzles: accepted answers, the trap move, an explanation.
- Pick a varied set by clustering position embeddings.
- Spaced repetition on misses.

### Phase 6: Server personalization (only if Phase 2b says so)

SQS + Fargate + S3 + DynamoDB, as described above. Accounts arrive here if needed for cross-device sync.

## Risks to verify early

1. Maia-3 actually beats Maia-2 for 800 to 1800 players on their own games.
2. ONNX export parity, including GAB attention and the promotion head.
3. Tokenization, history and move-index parity between TypeScript and Python.
4. Model size and speed versus accuracy under fp16 and int8; in-browser throughput.
5. Calibration actually improves held-out prediction.
6. Personalized vectors beat calibration with a realistic game count (Phase 2b).
