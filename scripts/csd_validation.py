"""CSD H4-redesign validation harness.

Quantifies detection power (fraction of seeds firing at alpha=0.05) for the
critical-slowing-down early-warning detector across record (event-window) lengths
{60, 121, 252} trading days and several kappa-proximity ramps, with stationary
negative controls.

Ground truth.  The positive control ramps the AR(1) recovery coefficient phi
toward 1, which is the linearised image of the reflexive SDE as kappa ->
kappa_star (phi = exp(Re(lambda) dt), Re(lambda) -> 0).  The volatility proxy is
a deterministic folded transform |z| of the slowing latent z, mirroring how
|r_t| in the model inherits its autocorrelation/variance from the slowing
process.  The negative control holds phi fixed far below criticality
(stationary, no critical slowing down).

Writes JSON + CSV + summary artifacts under runs/csd_validation/<ts>/.

Run:  uv run python scripts/csd_validation.py
"""

from __future__ import annotations

import csv
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from reflexive_options.theory.critical_slowing_down import csd_test

ALPHA = 0.05
N_SURR = 500
N_SEEDS = 60
PROXY = "abs_latent"  # |z| folded transform of the slowing latent
# Absolute ~6-week rolling window (Dakos et al. 2012 use fixed-length windows).
# A smaller window leaves MORE Kendall-tau samples across the record, which is
# what gives the trend test its power inside the short event window. (Using 50%
# of the record leaves too few samples and over-smooths -> low power.)
ROLL_WINDOW = 30
PHI_MIN = 0.20
SIGMA = 0.05  # FIXED innovation size; phi (recovery rate) is what slows


def _phi(kappa_frac: float) -> float:
    return 1.0 - (1.0 - PHI_MIN) * (1.0 - kappa_frac)


def gen_stationary(n: int, phi: float, seed: int) -> np.ndarray:
    """Stationary AR(1), fixed innovation SIGMA; |z| proxy. Both EWS flat."""
    rng = np.random.default_rng(seed)
    z = np.empty(n, dtype=np.float64)
    z[0] = rng.standard_normal() * SIGMA / np.sqrt(max(1.0 - phi * phi, 1e-3))
    for t in range(1, n):
        z[t] = phi * z[t - 1] + SIGMA * rng.standard_normal()
    return np.abs(z)


def gen_ramp(n: int, seed: int, kappa_end_frac: float) -> np.ndarray:
    """AR(1) with phi ramping toward 1; fixed innovation SIGMA so BOTH the lag-1
    autocorrelation and the variance (var = SIGMA^2/(1-phi^2)) rise. |z| proxy.
    """
    rng = np.random.default_rng(seed)
    kf = np.linspace(0.0, kappa_end_frac, n)
    phis = np.array([_phi(float(k)) for k in kf])
    z = np.empty(n, dtype=np.float64)
    z[0] = rng.standard_normal() * SIGMA / np.sqrt(max(1.0 - phis[0] * phis[0], 1e-3))
    for t in range(1, n):
        z[t] = phis[t] * z[t - 1] + SIGMA * rng.standard_normal()
    return np.abs(z)


def _roll_window(record_days: int) -> int:
    return max(15, min(ROLL_WINDOW, record_days - 5))


@dataclass
class Cell:
    label: str
    kind: str  # "positive" | "negative"
    record_days: int
    roll_window: int
    statistic: str
    kappa_end_frac: float | None
    power: float
    mean_tau: float
    median_p: float
    n_seeds: int


def run_positive(record_days: int, statistic: str, kappa_end_frac: float) -> Cell:
    rw = _roll_window(record_days)
    fires, taus, ps = 0, [], []
    for seed in range(N_SEEDS):
        proxy = gen_ramp(record_days, seed, kappa_end_frac)
        res = csd_test(
            proxy,
            window=rw,
            statistic=statistic,
            n_surrogates=N_SURR,
            surrogate="ar1",
            alpha=ALPHA,
            seed=100000 + seed,
        )
        taus.append(res.tau)
        ps.append(res.p_value)
        if res.significant:
            fires += 1
    return Cell(
        label=f"pos_{statistic}_d{record_days}_k{kappa_end_frac:.2f}",
        kind="positive",
        record_days=record_days,
        roll_window=rw,
        statistic=statistic,
        kappa_end_frac=kappa_end_frac,
        power=fires / N_SEEDS,
        mean_tau=float(np.nanmean(taus)),
        median_p=float(np.nanmedian(ps)),
        n_seeds=N_SEEDS,
    )


def run_negative(record_days: int, statistic: str) -> Cell:
    rw = _roll_window(record_days)
    fires, taus, ps = 0, [], []
    for seed in range(N_SEEDS):
        proxy = gen_stationary(record_days, phi=0.5, seed=200000 + seed)
        res = csd_test(
            proxy,
            window=rw,
            statistic=statistic,
            n_surrogates=N_SURR,
            surrogate="ar1",
            alpha=ALPHA,
            seed=300000 + seed,
        )
        taus.append(res.tau)
        ps.append(res.p_value)
        if res.significant:
            fires += 1
    return Cell(
        label=f"neg_stationary_{statistic}_d{record_days}",
        kind="negative",
        record_days=record_days,
        roll_window=rw,
        statistic=statistic,
        kappa_end_frac=None,
        power=fires / N_SEEDS,
        mean_tau=float(np.nanmean(taus)),
        median_p=float(np.nanmedian(ps)),
        n_seeds=N_SEEDS,
    )


def main() -> None:
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path("runs/csd_validation") / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    cells: list[Cell] = []
    record_days_grid = [60, 121, 252]
    statistics = ["autocorr", "variance"]
    kappa_end_fracs = [0.85, 0.95, 0.99]

    for d in record_days_grid:
        for stat in statistics:
            for kf in kappa_end_fracs:
                c = run_positive(d, stat, kf)
                cells.append(c)
                print(
                    f"[POS] {c.label:34s} power={c.power:.2f} "
                    f"tau={c.mean_tau:+.3f} p~{c.median_p:.3f}"
                )
    for d in record_days_grid:
        for stat in statistics:
            c = run_negative(d, stat)
            cells.append(c)
            print(
                f"[NEG] {c.label:34s} power={c.power:.2f} tau={c.mean_tau:+.3f} p~{c.median_p:.3f}"
            )

    rows = [asdict(c) for c in cells]
    (out_dir / "results.json").write_text(
        json.dumps(
            {
                "alpha": ALPHA,
                "n_surrogates": N_SURR,
                "n_seeds": N_SEEDS,
                "proxy": PROXY,
                "roll_window": ROLL_WINDOW,
                "phi_min": PHI_MIN,
                "sigma": SIGMA,
                "ground_truth": (
                    "AR(1) recovery-rate ramp, FIXED innovation; phi->1 as "
                    "kappa->kappa* so autocorr AND variance rise; proxy |z|"
                ),
                "cells": rows,
            },
            indent=2,
        )
    )
    with (out_dir / "results.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    pos = [c for c in cells if c.kind == "positive"]
    neg = [c for c in cells if c.kind == "negative"]
    achievers = [c for c in pos if c.power >= 0.80]
    lines = ["min (record_days, kappa_end_frac, stat) with power>=0.80:"]
    if achievers:
        for c in sorted(achievers, key=lambda c: (c.record_days, -(c.kappa_end_frac or 0)))[:10]:
            lines.append(
                f"  d={c.record_days} kf={c.kappa_end_frac} stat={c.statistic} power={c.power:.2f}"
            )
    else:
        lines.append("  NONE reached 0.80 in this grid")
    lines.append(f"max negative-control FPR across cells: {max(c.power for c in neg):.3f}")
    summary = "\n".join(lines)
    (out_dir / "summary.txt").write_text(summary + "\n")
    print("\n" + summary)
    print(f"\nArtifacts: {out_dir}")


if __name__ == "__main__":
    main()
