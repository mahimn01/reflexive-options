"""Data-free validation of the primary H1' GEX regression.

Runs the full H1' pipeline on simulator-generated data with KNOWN ground-truth
coupling kappa, mirroring how the original H4 was validated (known ground-truth
coupling in the SIMULATOR, no real data):

  - POSITIVE control: kappa in the strong-feedback regime (KAPPA_HIGH) -> the
    GEX -> next-day realized-vol-of-vol coefficient should be negative (short
    gamma amplifies) and significant, on a single 121-day event window and on
    the pooled 3 x 121-day panel.
  - NEGATIVE control: kappa = 0 (pure Heston) -> the coefficient should be null
    (false-positive rate at or below the nominal 5%).
  - QUIET-REGIME control: weak coupling (KAPPA_QUIET) -> the effect should be
    markedly weaker than the strong-feedback regime.

Self-contained: the ground-truth coupling is a property of the GEX simulator we
control directly; we do not import any frozen theory module. KAPPA_HIGH is fixed
at the value where the reflexive variance feedback materially exceeds the Heston
vol-of-vol diffusion (see gex_simulator docstring), the data-free analogue of
"near kappa*".

Saves a JSON report under runs/gex_validation/<ts>/.

Run:  python -m reflexive_options.empirical.gex_validation [n_trials]
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

from reflexive_options.empirical.gex_regression import (
    H1PrimeResult,
    estimate_gex_series,
    run_h1prime,
    run_h1prime_pooled,
)
from reflexive_options.empirical.gex_simulator import GEXReflexiveSimulator, GEXSimParams

N_WIN = 121  # trading days per event window
N_EVENTS = 3  # COVID + 2022 selloff + 2023 regional-bank (data-free analogues)

# Ground-truth couplings (data-free). These are simulator parameters whose
# values we control, exactly as the original H4 swept the simulator's kappa.
KAPPA_HIGH = 0.8  # strong-feedback regime: reflexive term dominates diffusion
KAPPA_QUIET = 0.15  # weak-feedback control: effect should be much weaker
KAPPA_NULL = 0.0  # pure Heston: GEX must be a null predictor


def _windows(kappa: float, base_seed: int) -> list[dict[str, np.ndarray]]:
    wins = []
    for e in range(N_EVENTS):
        out = GEXReflexiveSimulator(GEXSimParams(kappa=kappa), seed=base_seed * 1000 + e).simulate(
            N_WIN
        )
        wins.append(
            {
                "gex": estimate_gex_series(out.grids),
                "returns": out.returns,
                "vix": out.vix,
            }
        )
    return wins


def _single(kappa: float, seed: int, n_boot: int = 600) -> H1PrimeResult:
    out = GEXReflexiveSimulator(GEXSimParams(kappa=kappa), seed=seed).simulate(N_WIN)
    gex = estimate_gex_series(out.grids)
    return run_h1prime(gex, out.returns, vix=out.vix, n_boot=n_boot, seed=seed)


def run_validation(n_trials: int = 60, n_boot: int = 600, seed0: int = 0) -> dict[str, object]:
    pooled_hi_rej, pooled_hi_neg, pooled_z_rej, pooled_quiet_rej = [], [], [], []
    pooled_hi_coef, pooled_z_coef = [], []
    single_hi_rej, single_z_rej = [], []

    for s in range(seed0, seed0 + n_trials):
        rp_hi = run_h1prime_pooled(_windows(KAPPA_HIGH, s), n_boot=n_boot, seed=s)
        rp_z = run_h1prime_pooled(_windows(KAPPA_NULL, s), n_boot=n_boot, seed=s)
        rp_q = run_h1prime_pooled(_windows(KAPPA_QUIET, s), n_boot=n_boot, seed=s)
        pooled_hi_rej.append(rp_hi.reject_sign_and_sig)
        pooled_hi_neg.append(rp_hi.coef_gex < 0)
        pooled_hi_coef.append(rp_hi.coef_gex)
        pooled_z_rej.append(rp_z.reject_sign_and_sig)
        pooled_z_coef.append(rp_z.coef_gex)
        pooled_quiet_rej.append(rp_q.reject_sign_and_sig)

        single_hi_rej.append(_single(KAPPA_HIGH, s, n_boot).reject_sign_and_sig)
        single_z_rej.append(_single(KAPPA_NULL, s, n_boot).reject_sign_and_sig)

    return {
        "design": "primary = pooled 3 x 121-day event windows w/ event FE (363 obs)",
        "outcome": "next-day realized vol-of-vol; regressor = standardized signed GEX",
        "kappa_high": KAPPA_HIGH,
        "kappa_quiet_control": KAPPA_QUIET,
        "kappa_null": KAPPA_NULL,
        "n_trials": n_trials,
        "POOLED": {
            "power_high_kappa": float(np.mean(pooled_hi_rej)),
            "sign_correct_high": float(np.mean(pooled_hi_neg)),
            "fpr_zero_kappa": float(np.mean(pooled_z_rej)),
            "reject_rate_quiet_control": float(np.mean(pooled_quiet_rej)),
            "mean_coef_high": float(np.mean(pooled_hi_coef)),
            "mean_coef_zero": float(np.mean(pooled_z_coef)),
        },
        "SINGLE_WINDOW_121d": {
            "power_high_kappa": float(np.mean(single_hi_rej)),
            "fpr_zero_kappa": float(np.mean(single_z_rej)),
        },
    }


def main() -> None:
    n_trials = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    report = run_validation(n_trials=n_trials)
    ts = time.strftime("%Y%m%dT%H%M%S")
    outdir = Path("runs/gex_validation") / ts
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"\nSaved -> {outdir / 'report.json'}")


if __name__ == "__main__":
    main()
