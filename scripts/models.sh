#!/usr/bin/env bash
# Export the Maia ONNX model (if needed) and sync it to the production ModelBucket.
# `vp dev` serves the same files straight from python/artifacts; see apps/web/vite.config.ts.
set -euo pipefail

MODEL_ID=maia3-5m
ROOT=$(cd "$(dirname "$0")/.." && pwd)
ARTIFACTS="$ROOT/python/artifacts/$MODEL_ID"

if [[ ! -f "$ARTIFACTS/model.onnx" ]]; then
  (cd "$ROOT/python" && uv run python -m doppelbot.export --model "$MODEL_ID")
fi

export AWS_PROFILE=${AWS_PROFILE:-personal}
BUCKET=$(aws s3api list-buckets --output text \
  --query "Buckets[?starts_with(Name, 'doppelbot-production-modelbucket')].Name | [0]")
if [[ -z "$BUCKET" || "$BUCKET" == "None" ]]; then
  echo "no production ModelBucket found; deploy first" >&2
  exit 1
fi
# Keys are versioned by model ID, so the files can be cached forever.
aws s3 sync "$ARTIFACTS" "s3://$BUCKET/$MODEL_ID" \
  --exclude "*" --include model.onnx --include moves.json --include elo_emb.json \
  --cache-control "public, max-age=31536000, immutable"
