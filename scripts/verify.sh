#!/usr/bin/env bash
# verify.sh — run the same gauntlet CI runs, in the same order.
# Exit on first failure so the developer sees the upstream-most problem.
#
# Usage: bash scripts/verify.sh
#
# Tools required: ruff, mypy, pytest (and pytest-cov). Install via either:
#   uv sync --group dev
#   pip install -e ".[dev]"
set -euo pipefail

cd "$(dirname "$0")/.."

# Pick the runner: prefer `uv run` if uv is installed (matches CI exactly).
if command -v uv >/dev/null 2>&1; then
    RUN="uv run --no-sync"
else
    RUN=""
fi

run() {
    if [[ -n "${RUN}" ]]; then
        ${RUN} "$@"
    else
        "$@"
    fi
}

require() {
    if ! command -v "$1" >/dev/null 2>&1 && [[ -z "${RUN}" ]]; then
        echo "verify.sh: missing tool '$1' — install dev deps: pip install -e \".[dev]\"" >&2
        exit 2
    fi
}

require ruff
require mypy
require pytest

echo "==> ruff check src tests"
run ruff check src tests

echo "==> ruff format --check src tests"
run ruff format --check src tests

echo "==> mypy src"
run mypy src

echo "==> pytest --cov-fail-under=85"
run pytest --cov=reflexive_options --cov-report=term-missing --cov-fail-under=85

echo
echo "verify.sh: all checks passed."
