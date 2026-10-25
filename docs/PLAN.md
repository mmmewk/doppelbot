# Chess Trainer: Plan

A web app where a player enters a Lichess and/or chess.com username. The app builds a
model of how they play (the "me bot") on top of Maia-2, then trains them against a
slightly stronger version of themselves with sparring bots and drills.

Target players: **800 to 1800**.

## Decisions

| Area | Decision |
|---|---|
| Model | Maia-2 (MIT, `pip install maia2`), `rapid` and `blitz` checkpoints. Not Maia-3 (AGPL, inference-only). |
| Frontend | Vite+ (`vp`) + React + TypeScript. Mirrors the Maia platform frontend (CSSLab/maia-platform-frontend) but without Next.js. |
| License | GPL-3.0. This lets us port code from maia-platform-frontend, chessground and stockfish.js, which are all GPL-3.0. |
| Inference | ONNX export, run in the browser with onnxruntime-web: WebGPU first, WASM fallback, inside a web worker. |
| Chess logic | chess.js for rules. Stockfish WASM in a web worker for evaluation. |
| Infra | SST v4 on AWS, region `us-west-1`, domain `chess-trainer.mekoppe.com` (Route53 zone `mekoppe.com`, same account). |
| Data | Version 1 is local-first: games, profiles and drill state live in IndexedDB. No accounts. |

## What Maia-2 looks like inside (this drives the design)

Checked against `CSSLab/maia2` source:

- **Input:** an 18×8×8 board tensor: 12 piece planes, side to move, 4 castling planes, en passant. Positions with Black to move are mirrored, so the model always sees things from the mover's side. Moves are mirrored back afterwards.
- **Ratings are categorical:** 11 bands: `<1100`, `1100-1199` … `1900-1999`, `>=2000`. One shared `nn.Embedding(11, 128)` is used for both `elo_self` and `elo_oppo`.
- **Architecture:**
  - A CNN (256 channels, 5 residual blocks) turns the board into 8 tokens.
  - Those go through a transformer (dim 1024, 2 blocks, 16 heads), then the heads: move logits, side-info, value.
- **Where rating enters:** only in the transformer's attention queries:
  `q = q + elo_query([e_self; e_oppo])`, where `elo_query` is `Linear(256, 1024, bias=False)` per layer. The CNN never sees rating.
- **Outputs:** `forward(boards, elos_self, elos_oppo) -> (logits_maia, logits_side_info, logits_value)`.
  Policy = masked softmax over legal moves. Value: `clamp(v/2 + 0.5)`, flipped when Black is to move.

Consequences:

1. **The export takes the rating vectors as inputs, not band indices.** The ONNX graph takes
   `elo_self_emb[B,128]` and `elo_oppo_emb[B,128]` directly, and the 11 band vectors ship as JSON.
   Calibration, blending between bands, "+100" and per-user personalization all work with one model file.
2. **The model can be split in two.** Part A (CNN + token embedding) does not depend on rating. Part B (transformer + heads)
   is small. Caching Part A's output for each of a user's positions makes it cheap to re-score
   thousands of positions under many candidate vectors. That is what makes in-browser
   calibration and personalization realistic.
3. **Bands are coarse at the low end.** Everything under 1100 is one band. An 800 and a 1050
   player look the same to the base model, and "+100" for a 900 player is the same band.
   Personalization (below) is how we tell players apart there.

## Personalization: a player is a vector

A player is represented by a 128-float vector `e_self`. A bot is one or more vectors plus a
rule for choosing moves:

| Bot | Vectors | Move-choice rule |
|---|---|---|
| Speed | `me` | Sample from `P(move \| me)` |
| Variety | `me`, `peer` (the user's band) | `P_peer / P_me`, restricted to moves within ~25cp of Stockfish's best |
| Accuracy: traps | `me` (to predict the user's reply) | Plausible + sound + likely to draw a bad reply |
| Accuracy: planted blunders | `peer` | Common errors at the user's level that the user is unlikely to punish |
| Reference ("+100") | `me + Δ`, where `Δ` points toward the next band up | Diff analysis and drills |

A vector is about 0.5 KB. Bots of other people (a friend, a club, any set of games) are just more vectors.

**Limits:** a vector can only move the model along the directions the rating pathway
already controls: per-layer query biases in 2 attention layers. It may capture
"plays like a slightly different 1400 who loves early queen moves" well. It may not capture
a specific opening repertoire. If vectors alone are not enough, the next step up is a
low-rank adapter (LoRA) on `elo_query` or the attention projections. That means kilobytes to megabytes per user, and still no full fine-tune.

### Experiment ladder (run offline in Python first)

Each step is evaluated on the user's held-out games: the most recent 20%, with moves in severe time trouble filtered out.
Metrics are move-matching top-1 accuracy and mean log-likelihood, compared against the stated rating mapped to its band.

| Level | What gets fitted | How | Where it can run |
|---|---|---|---|
| L0 | Best single band | Score all 11 bands | Browser (forward passes only) |
| L1 | Blend between 2 adjacent bands | 1-D search | Browser |
| L2 | Weighted mix of the 11 band vectors | Derivative-free (CMA-ES), using cached Part A outputs | Browser |
| L3 | Free 128-d vector, starting from L1/L2 | Gradient descent (Adam), with a penalty for drifting far from the start; base model frozen | Python (local, then Fargate) |
| L4 | L3 plus a LoRA on `elo_query` | Gradient descent | Python |

Questions the experiments answer:

1. How many games before L2 or L3 beats L0? (Maia4All suggests plain fine-tuning fails on small
   data. A regularized vector should be much more data-efficient.)
2. Does L2 (in the browser, instant) get most of L3's gain? If so, ship L2 in the browser and use Fargate only for L3+.
3. Does `elo_oppo` matter? Fit the user's vector with the real opponents' bands as `elo_oppo`.

Test data: a dozen real accounts across 800 to 1800, on both sites.

### Server path (if L3+ is worth it)

1. The browser uploads cleaned positions to S3.
2. SST `Queue` (SQS) → SST `Task` (Fargate, Python 3.12 + maia2, CPU).
3. The job writes the vector to S3 and status to DynamoDB.
4. The browser polls and downloads the vector.

The vector is only adopted if it beats L0/L2 on held-out games.

## Repo layout

```
chess-trainer/
├─ sst.config.ts
├─ infra/              web.ts, models.ts, (later) personalize.ts
├─ apps/web/           Vite+ React TS app
├─ packages/
│  ├─ maia/            encoding, move vocab, ONNX session, worker, vector math
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

- Git, a Vite+ monorepo, React TS app, and SST v4 `StaticSite` at `chess-trainer.mekoppe.com`, deployed with `AWS_PROFILE=personal`.
- A model bucket behind CloudFront with immutable caching.
- A `python/` uv project with maia2 pinned. PyPI has 0.11.0; check it against the repo's `main`.

### Phase 1: Model parity (risk spike; blocks everything else)

1. **Export from Python.** Wrap `MAIA2Model` so it takes embedding inputs, and export as:
   - the full model
   - Part A (CNN + tokens)
   - Part B (transformer + heads)

   Use a dynamic batch size. Also export `moves.json` (the move vocabulary) and `elo_bands.json` (11×128).
2. **Python test data.** About 500 positions from real games, deliberately covering: Black to move, castling rights,
   en passant, promotions and under-promotions. For each, at several rating pairs, record: tensor, legal mask, top-k probabilities, value.
3. **TypeScript.** Write `boardToTensor` (chess.js), mirroring, the legal mask, masked softmax and value decoding.
   Use the move vocabulary loaded from JSON; don't regenerate it.
4. **Parity tests.** Tensors must match exactly. Probabilities must match within 1e-4:
   - in Node first
   - then in the browser on WebGPU and WASM (Vitest browser mode)

   Also check that full model ≡ Part B(Part A).
5. **Quantization.** Compare fp16 and int8 against fp32: top-1 agreement and KL divergence. Pick the default size to ship.

### Phase 2: Game ingestion + calibration (L0/L1)

- **Fetching:**
  - Lichess: NDJSON stream with `clocks=true`.
  - chess.com: monthly archives, `%clk` from the PGN.
  - Normalize both to one `Game` type and cache in IndexedDB.
- **Filters:**
  - Drop moves played with under 10s or under 10% of the starting clock.
  - Bullet and blitz use the `blitz` model; rapid and classical use `rapid`.
- **Calibration:** run Part A once per position, then score L0/L1 on held-out games.
  Report the improvement over the stated rating.
- **UI:** username → effective rating per time control, accuracy, and the games used.

### Phase 2b: Personalization experiments (L2 to L4)

Run the experiment ladder above in Python on real accounts. Then decide what runs in the browser and whether the Fargate path is needed.

### Phase 3: Play + Speed bot

- Board UI ported from maia-platform-frontend.
- Stockfish worker.
- Me-bot sampling.
- Optional opening book from a Lichess study PGN.

### Phase 4: Variety bot, then Accuracy bot

- Traps and planted-blunder modes.
- Log each outcome: fell for it, punished it, or missed it.

### Phase 5: Diff analysis + drills

- Compare `me` with `me + Δ`, verify with Stockfish, and group by theme.
- Turn flagged positions into puzzles: accepted answers, the trap move, an explanation.
- Pick a varied set by clustering position embeddings.
- Spaced repetition on misses.

### Phase 6: Server personalization (only if Phase 2b says so)

SQS + Fargate + S3 + DynamoDB, as described above. Accounts arrive here if needed for cross-device sync.

## Risks to verify early

1. ONNX export parity, including the split model.
2. Board encoding and move-index parity between TypeScript and Python.
3. Model size versus accuracy under fp16 and int8.
4. Calibration actually improves held-out prediction.
5. Personalized vectors beat calibration with a realistic game count (Phase 2b).
6. In-browser throughput for scoring a user's full history (WebGPU vs WASM).
