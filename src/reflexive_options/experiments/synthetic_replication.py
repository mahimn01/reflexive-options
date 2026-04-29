"""Replicate Marketron paper figures from published parameter sets.

Sanity-check our simulator + baseline implementations against the
Halperin-Itkin (2025) Marketron options paper (arXiv:2508.09863v2).
We do NOT need real data — the target is the published moment tables.

See ~/Documents/reflexivity-research/marketron_technical_brief.md for the
exact parameter sets (Tables 2, 5, 6) and the expected moments (Tables 7, 8).

IMPORTANT — what this experiment can and cannot reproduce:

  Marketron is a 3-factor *quasi-particle* SDE on (x, y, θ) with two memory
  channels (y, θ) feeding a non-linear potential V_M(x). Our reflexive sim is
  a different mechanism: 3-factor (S, v, z) with a dealer-gamma feedback
  G(S, z, v) entering the drift. The two SDEs are NOT the same — a 1:1
  reproduction of Marketron's Tables 7/8 from our simulator is impossible.

  What we CAN do:
    1. Run our `HestonSimulator` with parameters derived from the Marketron
       baseline (Table 2 σ + ancillary BMs). This is the "Heston comparator"
       leg the brief discusses around §6. Heston-side moments should be
       insensitive to mechanism — this leg is a sanity check that QuantLib
       integration produces sane moments.
    2. Run our reflexive sim and compare the *qualitative* moments (positive
       skew at long horizon, modest excess kurtosis, mean log-return drift)
       against Marketron's reported numbers. We expect:
         - mean / vol: at best order-of-magnitude agreement (mechanism-specific).
         - skew sign at long horizon: should match (positive in both, see brief §4).
         - kurtosis sign + decay-with-horizon: should match qualitatively.
       We expect to MISS the absolute volatility levels (Marketron itself
       overstates realized vol ~3–5×; brief §6.4) and quantitative skew at
       short horizon.

Run: python -m reflexive_options.experiments.synthetic_replication
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from reflexive_options.baselines.heston import HestonSimulator
from reflexive_options.experiments._common import (
    deterministic_rng,
    make_run_dir,
    save_config,
    save_metrics,
    timed,
)
from reflexive_options.simulator.gamma_aggregator import (
    GammaAggregator,
    GammaAggregatorConfig,
)
from reflexive_options.simulator.reflexive import ReflexiveSimulator
from reflexive_options.types import (
    HestonParams,
    OpenInterestGrid,
    PathArray,
    ReflexiveParams,
    SurfaceGrid,
)

# ---------------------------------------------------------------------------
# Marketron published parameter sets (brief §6, Tables 2/5/6)
# ---------------------------------------------------------------------------

# Halperin-Itkin original Marketron parameters use σ (price vol), σ_y (y-channel
# vol), σ_z (signal vol), plus the potential / signal coefficients. We map them
# to our (Heston, reflexive) parameter dataclasses as the closest mechanistic
# match per leg. The mapping is necessarily loose — see module docstring.
MARKETRON_PARAM_SETS: dict[str, dict[str, float]] = {
    # Brief §6.1 Table 2 — synthetic baseline (S&P500 weekly returns 2000–2024
    # via Halperin-Itkin 2025 Table 1).
    "table_2_synthetic": {
        "sigma": 0.37,
        "sigma_y": 0.3800,
        "sigma_z": 0.8334,
        "gamma": 0.2,
        "k": 1.2869,
        "mu": 1.6671,
        "g": 0.6831,
        "theta_hat": 6.7865,
        "y_bar": 0.4731,
        "y0": 0.1,
        "c": 3.9305,
        "b1": 1.6819,
        "b2": -1.2102,
        "theta0": 0.5,
    },
    # Brief §6.2 Table 5 — calibrated to SPX options at T = 0.425, Jan-Feb 2017.
    "table_5_calibrated_2017": {
        "sigma": 0.3934,
        "sigma_y": 1.008,
        "sigma_z": 0.8912,
        "gamma": 1.1031,
        "k": 2.7069,
        "mu": 4.6154,
        "g": 0.3173,
        "theta_hat": 6.9242,
        "y_bar": 1.6208,
        "y0": -0.0589,
        "c": 0.8897,
        "b1": 0.1220,
        "b2": -0.0549,
        "theta0": 1.1007,
    },
    # Brief §6.3 Table 6 — calibrated to SPX options at T = 0.041, Jan-Feb 2017.
    # (Argparse choice name says "2020" — this is historical; honor it for CLI
    # back-compat but it actually is the *short-T 2017* calibration.)
    "table_6_calibrated_2020": {
        "sigma": 0.8950,
        "sigma_y": 0.1244,
        "sigma_z": 0.2004,
        "gamma": 5.4118,
        "k": 1.8831,
        "mu": 4.5869,
        "g": 0.3108,
        "theta_hat": 7.5284,
        "y_bar": 1.1148,
        "y0": -0.2356,
        "c": 1.1189,
        "b1": 0.2455,
        "b2": 1.1286,
        "theta0": -0.2014,
    },
}

# ---------------------------------------------------------------------------
# Marketron published moment targets (brief §6.4, Tables 7 / 8)
# ---------------------------------------------------------------------------
#
# Each entry is (mean, vol, skew, kurtosis) annualized, at the listed horizon
# in years. Tables transcribed verbatim from the brief.
#
# Naming convention:
#   - table_5_calibrated_2017 → Table 8 moments (T = 0.425 calibration).
#   - table_6_calibrated_2020 → Table 7 moments (T = 0.041 calibration).
#   - table_2_synthetic       → no published moment table; derived defaults
#     (mechanism-specific, treated as soft targets only).
MarketronMoments = dict[float, dict[str, float]]

MARKETRON_MOMENT_TARGETS: dict[str, MarketronMoments] = {
    # Brief Table 7 (p. 23) — calibrated to options at T = 0.041.
    "table_6_calibrated_2020": {
        0.0397: {"mean": -0.1034, "vol": 0.6331, "skew": 0.0003, "excess_kurt": -0.0001},
        0.0833: {"mean": -0.1085, "vol": 0.7734, "skew": 0.0012, "excess_kurt": 0.0004},
        0.25: {"mean": -0.1086, "vol": 0.8340, "skew": 0.0112, "excess_kurt": 0.0006},
        0.50: {"mean": -0.0981, "vol": 0.8022, "skew": 0.0546, "excess_kurt": -0.0082},
        1.00: {"mean": 0.0855, "vol": 0.7423, "skew": 0.1688, "excess_kurt": -0.0074},
        2.00: {"mean": 0.2038, "vol": 0.6400, "skew": 0.4535, "excess_kurt": 0.2740},
        3.00: {"mean": 0.2005, "vol": 0.5895, "skew": 0.5482, "excess_kurt": 1.0588},
    },
    # Brief Table 8 (p. 24) — calibrated to options at T = 0.425.
    "table_5_calibrated_2017": {
        0.0397: {"mean": -0.0136, "vol": 0.2781, "skew": -0.0006, "excess_kurt": -0.0004},
        0.0833: {"mean": -0.0114, "vol": 0.3403, "skew": 0.0006, "excess_kurt": -0.0004},
        0.25: {"mean": 0.0505, "vol": 0.3693, "skew": -0.0011, "excess_kurt": 0.0030},
        0.50: {"mean": 0.2099, "vol": 0.3606, "skew": 0.0058, "excess_kurt": 0.0043},
        1.00: {"mean": 0.3558, "vol": 0.3333, "skew": 0.0533, "excess_kurt": 0.0175},
        2.00: {"mean": 0.3793, "vol": 0.2991, "skew": 0.1807, "excess_kurt": 0.0730},
        3.00: {"mean": 0.3548, "vol": 0.2812, "skew": 0.3147, "excess_kurt": 0.1686},
    },
    # No Marketron-published moment table for the synthetic Table 2 set.
    # Use the closest analogue (Table 8) as a SOFT target — these will likely
    # miss tolerance and that is documented as expected behavior.
    "table_2_synthetic": {
        0.0833: {"mean": 0.05, "vol": 0.37, "skew": 0.0, "excess_kurt": 0.0},
        0.25: {"mean": 0.05, "vol": 0.37, "skew": 0.0, "excess_kurt": 0.0},
        1.00: {"mean": 0.05, "vol": 0.37, "skew": 0.0, "excess_kurt": 0.0},
    },
}

# Per the brief §5.1: "8% relative accuracy when calibration set contains both
# Puts and Calls". Adopt the same tolerance for moment matching.
DEFAULT_REL_TOLERANCE: float = 0.08

# Moments where mechanism difference makes a tolerance match impossible.
# Brief §6.4 + §9.1: Marketron over-estimates vol 3-5× (joint-cal failure)
# and the skew sign question is itself unresolved. Mark as "informational".
INFORMATIONAL_MOMENTS: tuple[str, ...] = ("vol",)


@dataclass(frozen=True)
class ReplicationConfig:
    """Configuration for the Marketron replication experiment."""

    parameter_set: str = "table_5_calibrated_2017"
    n_paths: int = 50_000
    n_steps: int = 252
    dt: float = 1.0 / 252
    seed: int = 42
    initial_spot: float = 100.0
    risk_free_rate: float = 0.01
    horizons_years: tuple[float, ...] = field(
        default_factory=lambda: (0.0397, 0.0833, 0.25, 0.50, 1.00)
    )
    rel_tolerance: float = DEFAULT_REL_TOLERANCE


# ---------------------------------------------------------------------------
# Moment helpers
# ---------------------------------------------------------------------------


def _skew(x: np.ndarray) -> float:
    m = x.mean()
    s = x.std()
    if s == 0:
        return 0.0
    return float(((x - m) ** 3).mean() / s**3)


def _excess_kurt(x: np.ndarray) -> float:
    m = x.mean()
    s = x.std()
    if s == 0:
        return 0.0
    return float(((x - m) ** 4).mean() / s**4 - 3.0)


def annualized_log_return_moments(
    spots: PathArray,
    horizon_years: float,
    dt: float,
    initial_spot: float,
) -> dict[str, float]:
    """Annualized moments of the cumulative log-return at a given horizon.

    Marketron's convention (brief Tables 7/8): mean and vol are annualized
    (mean/T, vol*sqrt(1/T)), skew/kurt are dimensionless on the cumulative
    log-return at horizon T. We follow that convention exactly so cells line
    up with the brief.
    """
    if horizon_years <= 0:
        raise ValueError(f"horizon_years must be > 0, got {horizon_years}")
    n_steps_total = spots.shape[1] - 1
    horizon_steps = round(horizon_years / dt)
    if horizon_steps < 1:
        raise ValueError(
            f"horizon_years={horizon_years} smaller than dt={dt}, no samples available"
        )
    if horizon_steps > n_steps_total:
        raise ValueError(
            f"horizon_years={horizon_years} ({horizon_steps} steps) exceeds the simulated "
            f"{n_steps_total} steps; bump n_steps."
        )
    log_terminal = np.log(np.maximum(spots[:, horizon_steps], 1e-12))
    log_initial = math.log(initial_spot)
    cum_log_return = log_terminal - log_initial
    mean_ann = float(cum_log_return.mean() / horizon_years)
    vol_ann = float(cum_log_return.std(ddof=1) / math.sqrt(horizon_years))
    return {
        "mean": mean_ann,
        "vol": vol_ann,
        "skew": _skew(cum_log_return),
        "excess_kurt": _excess_kurt(cum_log_return),
    }


# ---------------------------------------------------------------------------
# Mappings: Marketron raw params → our HestonParams / ReflexiveParams
# ---------------------------------------------------------------------------


def _heston_params_from_marketron(raw: dict[str, float]) -> HestonParams:
    """Closest-fit Heston interpretation of a Marketron parameter set.

    Marketron's σ is the diffusion coefficient on log-price → maps to √θ in
    Heston (long-run vol). σ_y, the y-channel vol, is the closest analogue
    to the vol-of-vol ξ. Brownian motions in Marketron are uncorrelated
    (brief §1.1) ⇒ ρ = 0. κ_v has no direct Marketron analogue; we use a
    standard 2.0 (mean-reversion rate ≈ half-year half-life). v0 = θ
    (start at the long-run mean — Marketron does not specify v0).
    """
    theta = float(raw["sigma"]) ** 2
    xi = float(raw["sigma_y"])
    return HestonParams(
        kappa=2.0,
        theta=theta,
        xi=xi,
        rho=0.0,
        v0=theta,
    )


def _reflexive_params_from_marketron(raw: dict[str, float]) -> ReflexiveParams:
    """Closest-fit reflexive interpretation of a Marketron parameter set.

    Mapping is mechanistic-by-mechanistic — *not* a calibration. We use the
    Heston embedding above as the (S, v) backbone, then translate the
    Marketron coupling/leverage parameters into our (κ, γ, α, β) channels:

    - Marketron `c` (price-impact strength c(t) in front of V_M(x)) is the
      closest analogue of our `coupling` (κ in dS/S = (μ + κ G(S,z,v)) dt).
      We rescale by 1e-12 to land in the literature-prior O(1e-12) per
      USD-of-dealer-gamma (CLAUDE.md).
    - Marketron `g` (coupling constant in V_M) modulates feedback strength;
      we let it modulate `leverage` (γ in dv = κ_v(θ-v) + γ z + ...).
    - Marketron `mu` is the y-channel mean reversion rate (μ in
      dy = h(θ) + μ(ȳ - y) ...) — closest analogue of our `memory_decay` α.
    - Marketron `b1` (signal coefficient on f(θ) = b1 cos θ) modulates the
      drift contribution from the signal — closest analogue of `memory_intake`
      β scaled into the log-price intake channel.
    """
    base = _heston_params_from_marketron(raw)
    coupling = float(raw["c"]) * 1e-12  # land in literature prior O(1e-12)
    leverage = float(raw["g"]) * 1e-3  # modest leverage on memory→variance
    memory_decay = max(float(raw["mu"]), 1e-3)  # protect __post_init__ guard
    memory_intake = abs(float(raw["b1"]))  # sign absorbed by sign of z below
    drift = 0.0  # Marketron μ_x is state-dependent; risk-neutral here
    return ReflexiveParams(
        base=base,
        coupling=coupling,
        drift=drift,
        memory_decay=memory_decay,
        memory_intake=memory_intake,
        leverage=leverage,
    )


def _default_oi_grid(initial_spot: float) -> OpenInterestGrid:
    """Synthetic ATM-clustered OI grid for the gamma aggregator.

    Marketron does not specify dealer OI (it is not a dealer-flow model). We
    use a standard SPX-like ATM-clustered grid so G is a non-trivial input to
    the reflexive feedback. The exact magnitudes wash out via the κ rescale
    above — we just need a non-zero, sign-correct G.
    """
    log_moneyness = np.linspace(-0.10, 0.10, 11)
    maturities = np.array([7, 30, 60, 90, 180, 365], dtype=np.float64) / 365.25
    grid = SurfaceGrid(log_moneyness=log_moneyness, maturities=maturities)
    # Triangular ATM-peaked OI, scaled up so dealer-gamma is order-of-magnitude
    # similar to SPX (~$10B notional gamma ATM).
    weights = np.exp(-(log_moneyness**2) / 0.005)
    weights /= weights.sum()
    base_oi = 50_000.0  # contracts at peak strike, summed across maturities
    oi = base_oi * weights[:, None] * np.ones_like(maturities)[None, :]
    return OpenInterestGrid(grid=grid, contracts_open=oi.astype(np.float64))


# ---------------------------------------------------------------------------
# Replication legs
# ---------------------------------------------------------------------------


def run_marketron_heston_baseline(cfg: ReplicationConfig) -> dict[str, Any]:
    """Reproduce Marketron's Heston comparator using QuantLib through HestonSimulator.

    Brief §6: Marketron compares its own pricer against a Heston benchmark.
    We can run that exactly through QuantLib's `AnalyticHestonEngine`. The
    Heston parameters are derived from the Marketron set per
    `_heston_params_from_marketron`.
    """
    raw = MARKETRON_PARAM_SETS[cfg.parameter_set]
    heston_params = _heston_params_from_marketron(raw)
    sim = HestonSimulator(
        regimes=[heston_params],
        breakpoints=[],
        spot0=cfg.initial_spot,
        drift=cfg.risk_free_rate,
    )
    spots, variances = sim.simulate(
        n_paths=cfg.n_paths,
        n_steps=cfg.n_steps,
        dt=cfg.dt,
        seed=cfg.seed,
    )
    horizon_metrics: dict[str, dict[str, float]] = {}
    for h in cfg.horizons_years:
        if h <= cfg.n_steps * cfg.dt:
            horizon_metrics[f"{h:.4f}y"] = annualized_log_return_moments(
                spots, h, cfg.dt, cfg.initial_spot
            )
    return {
        "leg": "marketron_heston_baseline",
        "heston_params": {
            "kappa": heston_params.kappa,
            "theta": heston_params.theta,
            "xi": heston_params.xi,
            "rho": heston_params.rho,
            "v0": heston_params.v0,
        },
        "horizon_metrics": horizon_metrics,
        "terminal_spot_mean": float(spots[:, -1].mean()),
        "terminal_spot_std": float(spots[:, -1].std()),
        "terminal_variance_mean": float(variances[:, -1].mean()),
    }


def run_reflexive_with_matched_marketron_calibration(
    cfg: ReplicationConfig,
) -> dict[str, Any]:
    """Run our reflexive sim with parameters mapped from a Marketron set.

    This is NOT a κ optimization — Marketron and reflexive are different SDEs
    so a κ that "matches Marketron moments" is not well-defined a priori. We
    instead use the natural mapping in `_reflexive_params_from_marketron`
    and report the resulting moments alongside the Marketron targets so the
    comparison is transparent. See module docstring for what we expect to
    match vs miss.
    """
    raw = MARKETRON_PARAM_SETS[cfg.parameter_set]
    params = _reflexive_params_from_marketron(raw)
    oi_grid = _default_oi_grid(cfg.initial_spot)
    aggregator = GammaAggregator(
        oi_grid=oi_grid,
        risk_free_rate=cfg.risk_free_rate,
        config=GammaAggregatorConfig(),
    )
    sim = ReflexiveSimulator(
        params=params,
        gamma_aggregator=aggregator,
        initial_spot=cfg.initial_spot,
        antithetic=True,
    )
    spots, variances = sim.simulate(
        n_paths=cfg.n_paths,
        n_steps=cfg.n_steps,
        dt=cfg.dt,
        seed=cfg.seed,
    )
    horizon_metrics: dict[str, dict[str, float]] = {}
    for h in cfg.horizons_years:
        if h <= cfg.n_steps * cfg.dt:
            horizon_metrics[f"{h:.4f}y"] = annualized_log_return_moments(
                spots, h, cfg.dt, cfg.initial_spot
            )
    return {
        "leg": "reflexive_matched_marketron",
        "reflexive_params": {
            "coupling": params.coupling,
            "leverage": params.leverage,
            "memory_decay": params.memory_decay,
            "memory_intake": params.memory_intake,
            "base_kappa": params.base.kappa,
            "base_theta": params.base.theta,
            "base_xi": params.base.xi,
            "base_rho": params.base.rho,
            "base_v0": params.base.v0,
        },
        "horizon_metrics": horizon_metrics,
        "terminal_spot_mean": float(spots[:, -1].mean()),
        "terminal_spot_std": float(spots[:, -1].std()),
        "terminal_variance_mean": float(variances[:, -1].mean()),
    }


# ---------------------------------------------------------------------------
# Comparison + pass/fail logic
# ---------------------------------------------------------------------------


def _relative_error(measured: float, target: float) -> float:
    """Symmetric-safe relative error.

    For target ≈ 0 (Marketron skew/kurtosis at short horizons are ~1e-4) we
    fall back to absolute error against a small floor so we don't divide by
    zero. The floor 1e-3 is one order of magnitude below the smallest
    non-trivial Marketron entry (Table 7 row 1: skew = 0.0003).
    """
    denom = max(abs(target), 1e-3)
    return abs(measured - target) / denom


def compare_to_marketron_targets(
    our_metrics: dict[str, Any],
    target_set: str,
    rel_tolerance: float = DEFAULT_REL_TOLERANCE,
) -> dict[str, Any]:
    """Compute relative error per (horizon, moment) and pass/fail.

    Args:
        our_metrics: must contain `horizon_metrics: dict[str, dict[str, float]]`
            keyed by "{h:.4f}y" → {mean, vol, skew, excess_kurt}.
        target_set: a key into `MARKETRON_MOMENT_TARGETS`.
        rel_tolerance: per-moment relative-error threshold (default 8% per
            brief §5.1).

    Returns a dict with per-cell relative errors, a summary count of
    pass/fail/informational entries, and an overall "passed" bool that
    requires ALL non-informational cells to be within tolerance.
    """
    if target_set not in MARKETRON_MOMENT_TARGETS:
        raise KeyError(f"Unknown target_set {target_set!r}")
    targets = MARKETRON_MOMENT_TARGETS[target_set]
    our_horizon_metrics: dict[str, dict[str, float]] = our_metrics["horizon_metrics"]

    per_cell: dict[str, dict[str, dict[str, float | bool | str]]] = {}
    n_pass = 0
    n_fail = 0
    n_info = 0
    n_skipped = 0

    for horizon, target_moments in targets.items():
        horizon_key = f"{horizon:.4f}y"
        if horizon_key not in our_horizon_metrics:
            n_skipped += len(target_moments)
            continue
        our_moments = our_horizon_metrics[horizon_key]
        per_horizon: dict[str, dict[str, float | bool | str]] = {}
        for moment_name, target_value in target_moments.items():
            if moment_name not in our_moments:
                continue
            measured = our_moments[moment_name]
            rel_err = _relative_error(measured, target_value)
            informational = moment_name in INFORMATIONAL_MOMENTS
            if informational:
                status = "informational"
                n_info += 1
            elif rel_err <= rel_tolerance:
                status = "pass"
                n_pass += 1
            else:
                status = "fail"
                n_fail += 1
            per_horizon[moment_name] = {
                "measured": float(measured),
                "target": float(target_value),
                "relative_error": float(rel_err),
                "status": status,
                "informational": informational,
            }
        per_cell[horizon_key] = per_horizon

    overall_passed = n_fail == 0 and n_pass > 0
    return {
        "target_set": target_set,
        "rel_tolerance": rel_tolerance,
        "per_cell": per_cell,
        "n_pass": n_pass,
        "n_fail": n_fail,
        "n_informational": n_info,
        "n_skipped": n_skipped,
        "overall_passed": overall_passed,
    }


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-paths", type=int, default=50_000)
    parser.add_argument("--n-steps", type=int, default=252)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--parameter-set",
        type=str,
        default="table_5_calibrated_2017",
        choices=sorted(MARKETRON_PARAM_SETS),
    )
    parser.add_argument(
        "--rel-tolerance",
        type=float,
        default=DEFAULT_REL_TOLERANCE,
        help="Per-moment relative error threshold (default per brief §5.1: 0.08)",
    )
    args = parser.parse_args()

    cfg = ReplicationConfig(
        parameter_set=args.parameter_set,
        n_paths=args.n_paths,
        n_steps=args.n_steps,
        seed=args.seed,
        rel_tolerance=args.rel_tolerance,
    )
    run_dir = make_run_dir("synthetic_replication", seed=cfg.seed)
    save_config(run_dir, cfg)

    rng = deterministic_rng(cfg.seed)

    with timed("heston_baseline"):
        heston_metrics = run_marketron_heston_baseline(cfg)
    with timed("reflexive_matched"):
        reflexive_metrics = run_reflexive_with_matched_marketron_calibration(cfg)

    heston_comparison = compare_to_marketron_targets(
        heston_metrics, cfg.parameter_set, rel_tolerance=cfg.rel_tolerance
    )
    reflexive_comparison = compare_to_marketron_targets(
        reflexive_metrics, cfg.parameter_set, rel_tolerance=cfg.rel_tolerance
    )

    metrics: dict[str, Any] = {
        "config": {
            "parameter_set": cfg.parameter_set,
            "n_paths": cfg.n_paths,
            "n_steps": cfg.n_steps,
            "dt": cfg.dt,
            "seed": cfg.seed,
            "rel_tolerance": cfg.rel_tolerance,
        },
        "marketron_param_set_raw": MARKETRON_PARAM_SETS[cfg.parameter_set],
        "marketron_moment_targets": {
            f"{h:.4f}y": v for h, v in MARKETRON_MOMENT_TARGETS[cfg.parameter_set].items()
        },
        "heston_baseline": heston_metrics,
        "reflexive_matched": reflexive_metrics,
        "heston_vs_marketron": heston_comparison,
        "reflexive_vs_marketron": reflexive_comparison,
        "rng_state": str(rng),
    }
    save_metrics(run_dir, metrics)
    print(f"Wrote results to: {run_dir}")
    print(
        f"Heston leg:    pass={heston_comparison['n_pass']} fail={heston_comparison['n_fail']} "
        f"info={heston_comparison['n_informational']} → overall_passed={heston_comparison['overall_passed']}"
    )
    print(
        f"Reflexive leg: pass={reflexive_comparison['n_pass']} fail={reflexive_comparison['n_fail']} "
        f"info={reflexive_comparison['n_informational']} → overall_passed={reflexive_comparison['overall_passed']}"
    )


if __name__ == "__main__":
    main()
