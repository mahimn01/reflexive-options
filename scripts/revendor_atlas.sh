#!/usr/bin/env bash
# Re-vendor ATLAS + RAT modules from the trading-algo source repo.
#
# Use when upstream `trading-algo` ships changes to ATLAS internals and we
# want to pull them in. NOT a normal-development operation — only run when
# explicitly re-syncing.
#
# Prerequisites:
#   - trading-algo cloned at $TRADING_ALGO_DIR (default: ../randomThings/trading_algo)
#   - This repo's working tree is clean (`git status` shows nothing).
#
# Usage: bash scripts/revendor_atlas.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRADING_ALGO_DIR="${TRADING_ALGO_DIR:-$HOME/Documents/Dev/randomThings/trading_algo}"

if [[ ! -d "$TRADING_ALGO_DIR/quant_core/models/atlas" ]]; then
    echo "ERROR: ATLAS source not found at $TRADING_ALGO_DIR/quant_core/models/atlas" >&2
    exit 1
fi

if ! git -C "$REPO_ROOT" diff --quiet HEAD; then
    echo "ERROR: working tree dirty. Commit or stash before re-vendoring." >&2
    exit 1
fi

ATLAS_DEST="$REPO_ROOT/src/reflexive_options/third_party/atlas"
RAT_DEST="$REPO_ROOT/src/reflexive_options/third_party/rat"

# Re-derived from the import-surface analysis at
# ~/Documents/reflexivity-research/atlas_import_surface.md
# Only Tier-1 (AS-IS) and Tier-2 (ADAPT) files are copied. Edit the manifest
# to add/remove files.
ATLAS_TIER1_FILES=(
    mamba.py
    attention.py
    backbone_v7.py
    fusion.py
    vsn.py
    config.py
    time_encoding.py
    rewards.py
    memory_bank.py
    train_ppo.py
    train_bc.py
    train_ewc.py
    train_curriculum.py
    inference.py
    validate.py
    model.py
    data_pipeline.py
    features.py
)

RAT_FILES=(
    reflexivity/meter.py
    topology/detector.py
    attention/flow.py
    attention/tracker.py
    config.py
    signals.py
)

echo "Re-vendoring ATLAS files into $ATLAS_DEST"
for f in "${ATLAS_TIER1_FILES[@]}"; do
    src="$TRADING_ALGO_DIR/quant_core/models/atlas/$f"
    if [[ -f "$src" ]]; then
        cp "$src" "$ATLAS_DEST/$f"
    else
        echo "  WARN: missing $src — skipping" >&2
    fi
done

echo "Re-vendoring RAT files into $RAT_DEST"
for f in "${RAT_FILES[@]}"; do
    src="$TRADING_ALGO_DIR/rat/$f"
    dst="$RAT_DEST/$f"
    mkdir -p "$(dirname "$dst")"
    if [[ -f "$src" ]]; then
        cp "$src" "$dst"
    else
        echo "  WARN: missing $src — skipping" >&2
    fi
done

# Patch import paths so `quant_core.models.atlas.X` -> `reflexive_options.third_party.atlas.X`
echo "Patching import paths"
find "$ATLAS_DEST" "$RAT_DEST" -name "*.py" -type f -print0 | xargs -0 sed -i '' \
    -e 's|from quant_core\.models\.atlas|from reflexive_options.third_party.atlas|g' \
    -e 's|from quant_core\.models\.atlas|from reflexive_options.third_party.atlas|g' \
    -e 's|from trading_algo\.rat|from reflexive_options.third_party.rat|g' \
    -e 's|import quant_core\.models\.atlas|import reflexive_options.third_party.atlas|g'

echo "Running ruff on vendored code"
ruff check "$ATLAS_DEST" "$RAT_DEST" || true

echo "Running smoke test"
pytest tests/test_atlas_import.py -v

echo "Re-vendor complete. Review the diff before committing:"
echo "  git diff src/reflexive_options/third_party/"
