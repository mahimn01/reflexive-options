#!/usr/bin/env bash
# Generates tests/repro/baseline_v0.1.0.json — the canonical reproducibility receipt.
# Run after any *intentional* change to experiment outputs to refresh the baseline.
#
# Usage:  bash scripts/generate_repro_baseline.sh
#
# The receipt is the single source of truth for tests/test_reproducibility.py.
# Commit any change to baseline_v0.1.0.json with a message describing *why*
# the numbers moved.
set -euo pipefail

cd "$(dirname "$0")/.."

if command -v uv >/dev/null 2>&1; then
    uv run --no-sync python -m reflexive_options.experiments._generate_repro_baseline
else
    python -m reflexive_options.experiments._generate_repro_baseline
fi
