# Run `just` to list recipes. See docs/PLAN.md "Conventions" for what must pass before a step is done.

# List recipes
default:
    @just --list

# Install JS and Python dependencies
install:
    vp install
    cd python && uv sync

# Web app dev server (no AWS needed); serves /models from python/artifacts
web port="5173":
    cd apps/web && vp dev --port {{port}} --strictPort

# Full SST dev mode (needs AWS credentials)
sst-dev:
    AWS_PROFILE=personal vp exec sst dev

# Format, lint and type check
check:
    vp check

# Format and apply lint fixes
fix:
    vp check --fix

# TypeScript unit tests in every package
test:
    vp run -r test

# Maia parity and bot-play tests in Chromium (needs the exported model and fixtures)
test-browser:
    cd packages/maia && vp run test:browser

# Python tests
test-py:
    cd python && uv run python -m pytest

# Everything that must pass before a phase step is marked done
ready: check test test-browser test-py build

# Production build of the web app
build:
    cd apps/web && pnpm run build

# Export the Maia ONNX model, moves.json and elo_emb.json to python/artifacts/<model>
export-model model="maia3-5m":
    cd python && uv run python -m doppelbot.export --model {{model}}

# Regenerate the parity fixtures from cached games
fixtures limit="500":
    cd python && uv run python -m doppelbot.fixtures --limit {{limit}}

# Rerun the Phase 1 model comparison (resumes; pass --summary-only to just print results)
compare *args:
    cd python && uv run python -m doppelbot.compare {{args}}

# Write packages/config/defaults.json after changing config defaults
config-export:
    vp run @doppelbot/config#export

# Upload the model files to the production ModelBucket
upload-models:
    scripts/models.sh

# Deploy the production stage
deploy:
    AWS_PROFILE=personal vp exec sst deploy --stage production
