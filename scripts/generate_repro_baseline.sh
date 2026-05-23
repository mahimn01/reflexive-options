#!/usr/bin/env bash
# Generates the reproducibility receipts:
#   - tests/repro/baseline_v0.1.0.json — the original four experiments
#     (bifurcation_scan, phase_diagram, synthetic_replication,
#      reflexive_transfer) at the canonical seed with whole-metric hashing.
#   - tests/repro/baseline_v0.3.3.json — the Wave 1–6 experiments
#     (lambda_scaling, limit_cycle_supercritical, hawkes_sv_equivalence,
#      codim2_analysis, mckean_vlasov_validation, kappa_star_robustness,
#      h_bimod_2d_scan, h1_synthetic_validation) with per-metric tolerance
#     (1e-10 abs for deterministic, 5% rel for stochastic).
#
# Run after any *intentional* change to experiment outputs to refresh the
# corresponding baseline. The receipts are the single source of truth for
# tests/test_reproducibility.py.
#
# Pass `--v033-only` or `--v010-only` to regenerate just one receipt.
#
# Usage:
#     bash scripts/generate_repro_baseline.sh                # both
#     bash scripts/generate_repro_baseline.sh --v033-only    # Wave 1–6 only
#     bash scripts/generate_repro_baseline.sh --v010-only    # original only
#
# Each leg is idempotent: rerunning with the same git HEAD + locked seeds
# produces a bit-identical JSON (modulo the generated_at_utc timestamp).
# Commit any change to either receipt with a message describing *why*
# the numbers moved.
set -euo pipefail

cd "$(dirname "$0")/.."

if command -v uv >/dev/null 2>&1; then
    RUN=(uv run --no-sync python -m)
else
    RUN=(python -m)
fi

RUN_V010=1
RUN_V033=1

for arg in "$@"; do
    case "$arg" in
        --v033-only) RUN_V010=0 ;;
        --v010-only) RUN_V033=0 ;;
        -h|--help)
            sed -n '2,26p' "$0"
            exit 0
            ;;
        *)
            echo "generate_repro_baseline.sh: unknown arg '$arg' (try --help)" >&2
            exit 2
            ;;
    esac
done

if [[ "$RUN_V010" -eq 1 ]]; then
    echo "==> Generating v0.1.0 reproducibility receipt"
    "${RUN[@]}" reflexive_options.experiments._generate_repro_baseline
fi

if [[ "$RUN_V033" -eq 1 ]]; then
    echo
    echo "==> Generating v0.3.3 reproducibility receipt (Wave 1–6)"
    "${RUN[@]}" reflexive_options.experiments._generate_repro_baseline_v033
fi
