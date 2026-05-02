"""Coarse-grid tuning of reflexive (κ, γ, T_eff, μ_q, σ_q) vs Marketron shape moments.

Maximizes the count of `shape_target` cells matching across all three published
Marketron parameter sets. We tune the *reflexive* knobs only — the simulator
itself is canonical (v0.1.0) and is NOT modified by this script.

Tuning grid (864 configs total):

    κ ∈ {1e-13, 5e-13, 1e-12, 5e-12, 1e-11, 5e-11}     (6, log-spaced)
    γ ∈ {0, 0.5, 1.5, 3.0}                              (4)
    T_eff ∈ {0.041, 0.083, 0.25, 0.5}  → α = 1/T_eff    (4)
    μ_q ∈ {-0.05, 0.0, +0.05}                           (3, OI grid centre)
    σ_q ∈ {0.05, 0.10, 0.20}                            (3, OI grid width)

Per (parameter_set, config) we record:
    - number of cells passing strict 8% gate
    - shape_target cells with matching skew sign
    - shape_target cells with matching excess-kurt sign
    - shape_target cells with matching horizon-direction (slope of mean)
    - mean absolute error on shape moments

Output: ``runs/marketron_tuning/<timestamp>/grid_results.parquet`` plus
``best_overrides.json`` for downstream consumption by the replication script.

Compute envelope (n_paths=2000, n_steps=504 ≈ 2y):
    864 configs × 3 sets × ~0.75s/sim ≈ 32 min on Apple M-series CPU.

The compute envelope deliberately drops the 3-year Marketron horizon (Tables
7/8 row 8). The shape signal we care about is already saturated by the 2-year
row in both Tables 7 and 8, so the trim is methodologically clean. See
`paper/mechanism_decomposition.md` Table 1 for the cell-level coverage.
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from itertools import pairwise, product
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from reflexive_options.experiments._common import (
    REPO_ROOT,
    make_run_dir,
    save_config,
    timed,
)
from reflexive_options.experiments.synthetic_replication import (
    DEFAULT_REL_TOLERANCE,
    MARKETRON_MOMENT_TARGETS,
    ReplicationConfig,
    TunedReflexiveOverrides,
    _sign_of,
    classify_mechanism,
    compare_to_marketron_targets,
    run_reflexive_with_matched_marketron_calibration,
)


@dataclass(frozen=True)
class TuningGridConfig:
    """Coarse-grid axes (and Monte-Carlo budget) for tuning."""

    kappa_grid: tuple[float, ...] = (1e-13, 5e-13, 1e-12, 5e-12, 1e-11, 5e-11)
    gamma_grid: tuple[float, ...] = (0.0, 0.5, 1.5, 3.0)
    t_eff_grid: tuple[float, ...] = (0.041, 0.083, 0.25, 0.5)
    mu_q_grid: tuple[float, ...] = (-0.05, 0.0, 0.05)
    sigma_q_grid: tuple[float, ...] = (0.05, 0.10, 0.20)
    parameter_sets: tuple[str, ...] = (
        "table_5_calibrated_2017",
        "table_6_calibrated_2020",
        "table_2_synthetic",
    )
    n_paths: int = 2000
    n_steps: int = 504  # 2y at 1/252; covers all Marketron horizons except 3y
    dt: float = 1.0 / 252
    seed: int = 42
    initial_spot: float = 100.0
    risk_free_rate: float = 0.01
    horizons_years: tuple[float, ...] = field(
        default_factory=lambda: (0.0397, 0.0833, 0.25, 0.50, 1.00, 2.00)
    )

    @property
    def grid_size(self) -> int:
        return (
            len(self.kappa_grid)
            * len(self.gamma_grid)
            * len(self.t_eff_grid)
            * len(self.mu_q_grid)
            * len(self.sigma_q_grid)
        )


@dataclass(frozen=True)
class TuningOutcome:
    """One row of the tuning grid_results.parquet output."""

    parameter_set: str
    kappa: float
    gamma: float
    t_eff: float
    memory_decay: float  # = 1 / t_eff
    mu_q: float
    sigma_q: float
    n_paths: int
    n_steps: int
    seed: int
    # Strict 8%-gate counters (legacy comparison).
    n_pass_strict_8pct: int
    # Shape-target counters (the ones we tune on).
    shape_total: int
    shape_skew_sign_match: int
    shape_kurt_sign_match: int
    shape_sign_match_total: int
    shape_horizon_direction_match: int
    shape_mean_abs_error: float
    overall_passed: bool
    shape_match_rate: float


# ---------------------------------------------------------------------------
# Tuning evaluation primitives
# ---------------------------------------------------------------------------


def _iterate_configs(
    grid: TuningGridConfig,
) -> Iterator[tuple[float, float, float, float, float]]:
    """Iterate the cartesian product of (κ, γ, T_eff, μ_q, σ_q)."""
    return product(
        grid.kappa_grid,
        grid.gamma_grid,
        grid.t_eff_grid,
        grid.mu_q_grid,
        grid.sigma_q_grid,
    )


def _measured_mean_horizon_direction(
    measured_per_horizon: dict[str, dict[str, float]],
    target_per_horizon: dict[float, dict[str, float]],
) -> tuple[int, int]:
    """Match horizon-by-horizon direction of the *mean* log-return between measured and target.

    Returns ``(matches, total_pairs)``. We compare the *sign of the slope of
    mean(h)* between adjacent horizons — i.e., is the simulated drift trending
    up vs down with horizon, the same way Marketron's mean curve does.
    """
    sorted_target_horizons = sorted(target_per_horizon.keys())
    matches = 0
    total = 0
    for h_prev, h_curr in pairwise(sorted_target_horizons):
        key_prev = f"{h_prev:.4f}y"
        key_curr = f"{h_curr:.4f}y"
        if key_prev not in measured_per_horizon or key_curr not in measured_per_horizon:
            continue
        target_slope = target_per_horizon[h_curr]["mean"] - target_per_horizon[h_prev]["mean"]
        measured_slope = (
            measured_per_horizon[key_curr]["mean"] - measured_per_horizon[key_prev]["mean"]
        )
        if _sign_of(measured_slope) == _sign_of(target_slope):
            matches += 1
        total += 1
    return matches, total


def _evaluate_one_config(
    parameter_set: str,
    kappa: float,
    gamma: float,
    t_eff: float,
    mu_q: float,
    sigma_q: float,
    grid: TuningGridConfig,
) -> TuningOutcome:
    """Run the reflexive sim once at this config × parameter set, score against Marketron."""
    overrides = TunedReflexiveOverrides(
        kappa=kappa,
        leverage=gamma,
        memory_decay=1.0 / max(t_eff, 1e-3),
        oi_mu_q=mu_q,
        oi_sigma_q=sigma_q,
    )
    cfg = ReplicationConfig(
        parameter_set=parameter_set,
        n_paths=grid.n_paths,
        n_steps=grid.n_steps,
        dt=grid.dt,
        seed=grid.seed,
        initial_spot=grid.initial_spot,
        risk_free_rate=grid.risk_free_rate,
        horizons_years=grid.horizons_years,
        rel_tolerance=DEFAULT_REL_TOLERANCE,
        tuned_overrides=overrides,
    )
    metrics = run_reflexive_with_matched_marketron_calibration(cfg, overrides=overrides)
    comparison = compare_to_marketron_targets(metrics, parameter_set, DEFAULT_REL_TOLERANCE)

    target_per_horizon = MARKETRON_MOMENT_TARGETS[parameter_set]
    horizon_dir_match, _horizon_dir_total = _measured_mean_horizon_direction(
        metrics["horizon_metrics"], target_per_horizon
    )

    # Per-cell breakdown for skew vs kurt. NaN cells (sim blew up at this κ)
    # are tallied as failed sign-match AND penalized in `abs_errors` so the
    # downstream selection ranks blow-up configs as worst-case.
    skew_sign_match = 0
    kurt_sign_match = 0
    abs_errors: list[float] = []
    for _horizon_key, per_horizon in comparison["per_cell"].items():
        for moment_name, cell in per_horizon.items():
            if classify_mechanism(moment_name, parameter_set) != "shape_target":
                continue
            if moment_name == "skew" and cell["sign_match"]:
                skew_sign_match += 1
            elif moment_name == "excess_kurt" and cell["sign_match"]:
                kurt_sign_match += 1
            measured = float(cell["measured"])
            target = float(cell["target"])
            err = abs(measured - target)
            if not np.isfinite(err):
                # NaN/inf measured ⇒ simulator blew up; treat as worst-case error.
                err = 1.0e9
            abs_errors.append(err)

    return TuningOutcome(
        parameter_set=parameter_set,
        kappa=kappa,
        gamma=gamma,
        t_eff=t_eff,
        memory_decay=1.0 / max(t_eff, 1e-3),
        mu_q=mu_q,
        sigma_q=sigma_q,
        n_paths=grid.n_paths,
        n_steps=grid.n_steps,
        seed=grid.seed,
        n_pass_strict_8pct=int(comparison["n_pass"]),
        shape_total=int(comparison["shape_target_cells_total"]),
        shape_skew_sign_match=skew_sign_match,
        shape_kurt_sign_match=kurt_sign_match,
        shape_sign_match_total=int(comparison["shape_target_sign_match"]),
        shape_horizon_direction_match=horizon_dir_match,
        shape_mean_abs_error=float(np.mean(abs_errors)) if abs_errors else 0.0,
        overall_passed=bool(comparison["overall_passed"]),
        shape_match_rate=float(comparison["shape_match_rate"]),
    )


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def run_tuning_grid(
    grid: TuningGridConfig,
    *,
    progress_every: int = 50,
) -> list[TuningOutcome]:
    """Iterate the full grid, evaluating every (set, config) pair in series.

    Returns the flat list of outcomes (one row per (set, config)). The driver
    can write/aggregate as needed.
    """
    outcomes: list[TuningOutcome] = []
    n_configs = grid.grid_size
    n_evals = n_configs * len(grid.parameter_sets)
    print(
        f"Tuning grid: {n_configs} configs × {len(grid.parameter_sets)} parameter sets "
        f"= {n_evals} evaluations"
    )
    print(f"  n_paths={grid.n_paths}, n_steps={grid.n_steps} (~{grid.n_steps * grid.dt:.2f}y)")

    t0 = time.perf_counter()
    eval_idx = 0
    for kappa, gamma, t_eff, mu_q, sigma_q in _iterate_configs(grid):
        for parameter_set in grid.parameter_sets:
            outcomes.append(
                _evaluate_one_config(
                    parameter_set=parameter_set,
                    kappa=kappa,
                    gamma=gamma,
                    t_eff=t_eff,
                    mu_q=mu_q,
                    sigma_q=sigma_q,
                    grid=grid,
                )
            )
            eval_idx += 1
            if eval_idx % progress_every == 0:
                elapsed = time.perf_counter() - t0
                rate = eval_idx / max(elapsed, 1e-9)
                eta = (n_evals - eval_idx) / max(rate, 1e-9)
                print(
                    f"  [{eval_idx}/{n_evals}] "
                    f"elapsed={elapsed:.1f}s rate={rate:.2f}/s ETA={eta:.0f}s"
                )

    elapsed = time.perf_counter() - t0
    print(f"Tuning grid complete: {n_evals} evals in {elapsed:.1f}s ({elapsed / 60:.1f} min)")
    return outcomes


# ---------------------------------------------------------------------------
# Best-config selection
# ---------------------------------------------------------------------------


def select_best_overrides_per_set(
    outcomes: list[TuningOutcome],
) -> dict[str, TunedReflexiveOverrides]:
    """Pick the best (κ, γ, T_eff, μ_q, σ_q) config per parameter set.

    Selection order: (1) maximize shape_sign_match_total, (2) maximize
    horizon_direction_match, (3) minimize shape_mean_abs_error.
    """
    by_set: dict[str, list[TuningOutcome]] = {}
    for o in outcomes:
        by_set.setdefault(o.parameter_set, []).append(o)

    best: dict[str, TunedReflexiveOverrides] = {}
    for parameter_set, rows in by_set.items():
        rows_sorted = sorted(
            rows,
            key=lambda r: (
                -r.shape_sign_match_total,
                -r.shape_horizon_direction_match,
                r.shape_mean_abs_error,
            ),
        )
        winner = rows_sorted[0]
        best[parameter_set] = TunedReflexiveOverrides(
            kappa=winner.kappa,
            leverage=winner.gamma,
            memory_decay=winner.memory_decay,
            oi_mu_q=winner.mu_q,
            oi_sigma_q=winner.sigma_q,
        )
    return best


def select_best_overrides_global(
    outcomes: list[TuningOutcome],
) -> TunedReflexiveOverrides:
    """Pick the single (κ, γ, T_eff, μ_q, σ_q) config that wins *across all sets*.

    Aggregates by config-tuple, sums shape_sign_match_total over all sets in
    that group, then applies the same tiebreakers as
    `select_best_overrides_per_set`.
    """
    by_config: dict[tuple[float, float, float, float, float], list[TuningOutcome]] = {}
    for o in outcomes:
        key = (o.kappa, o.gamma, o.t_eff, o.mu_q, o.sigma_q)
        by_config.setdefault(key, []).append(o)

    aggregates: list[tuple[tuple[float, float, float, float, float], int, int, float, float]] = []
    for key, rows in by_config.items():
        total_sign_match = sum(r.shape_sign_match_total for r in rows)
        total_horizon_match = sum(r.shape_horizon_direction_match for r in rows)
        mean_abs_err = float(np.mean([r.shape_mean_abs_error for r in rows]))
        max_match_rate = max(r.shape_match_rate for r in rows)
        aggregates.append(
            (key, total_sign_match, total_horizon_match, mean_abs_err, max_match_rate)
        )

    aggregates.sort(
        key=lambda a: (
            -a[1],  # max shape_sign_match_total
            -a[2],  # max horizon_direction_match
            a[3],  # min shape_mean_abs_error
            -a[4],  # max single-set shape_match_rate (tiebreaker for ties)
        )
    )
    winner_key = aggregates[0][0]
    kappa, gamma, t_eff, mu_q, sigma_q = winner_key
    return TunedReflexiveOverrides(
        kappa=kappa,
        leverage=gamma,
        memory_decay=1.0 / max(t_eff, 1e-3),
        oi_mu_q=mu_q,
        oi_sigma_q=sigma_q,
    )


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------


def write_grid_results(outcomes: list[TuningOutcome], out_path: Path) -> None:
    """Persist outcomes as parquet, mirroring the repo's surface/io.py pattern."""
    if not outcomes:
        raise ValueError("write_grid_results: empty outcomes list")
    columns: dict[str, list[Any]] = {
        "parameter_set": [],
        "kappa": [],
        "gamma": [],
        "t_eff": [],
        "memory_decay": [],
        "mu_q": [],
        "sigma_q": [],
        "n_paths": [],
        "n_steps": [],
        "seed": [],
        "n_pass_strict_8pct": [],
        "shape_total": [],
        "shape_skew_sign_match": [],
        "shape_kurt_sign_match": [],
        "shape_sign_match_total": [],
        "shape_horizon_direction_match": [],
        "shape_mean_abs_error": [],
        "overall_passed": [],
        "shape_match_rate": [],
    }
    for o in outcomes:
        columns["parameter_set"].append(o.parameter_set)
        columns["kappa"].append(o.kappa)
        columns["gamma"].append(o.gamma)
        columns["t_eff"].append(o.t_eff)
        columns["memory_decay"].append(o.memory_decay)
        columns["mu_q"].append(o.mu_q)
        columns["sigma_q"].append(o.sigma_q)
        columns["n_paths"].append(o.n_paths)
        columns["n_steps"].append(o.n_steps)
        columns["seed"].append(o.seed)
        columns["n_pass_strict_8pct"].append(o.n_pass_strict_8pct)
        columns["shape_total"].append(o.shape_total)
        columns["shape_skew_sign_match"].append(o.shape_skew_sign_match)
        columns["shape_kurt_sign_match"].append(o.shape_kurt_sign_match)
        columns["shape_sign_match_total"].append(o.shape_sign_match_total)
        columns["shape_horizon_direction_match"].append(o.shape_horizon_direction_match)
        columns["shape_mean_abs_error"].append(o.shape_mean_abs_error)
        columns["overall_passed"].append(o.overall_passed)
        columns["shape_match_rate"].append(o.shape_match_rate)

    table = pa.table(columns)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # See surface/io.py for the rationale on the type: ignore.
    pq.write_table(table, out_path, compression="snappy")  # type: ignore[no-untyped-call]


def write_best_overrides(
    best_per_set: dict[str, TunedReflexiveOverrides],
    best_global: TunedReflexiveOverrides,
    out_path: Path,
) -> None:
    """Write the best-overrides JSON consumed by `synthetic_replication.load_tuned_overrides`."""
    payload: dict[str, Any] = {"_global": asdict(best_global)}
    for parameter_set, overrides in best_per_set.items():
        payload[parameter_set] = asdict(overrides)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, default=str))


def link_latest(run_dir: Path) -> None:
    """Maintain ``runs/marketron_tuning/latest`` → most recent run dir.

    Uses an actual symlink on POSIX; copies the manifest if symlinks fail
    (e.g. Windows without dev mode). The downstream `load_tuned_overrides`
    in `synthetic_replication.py` only needs ``best_overrides.json`` at this
    path, so we copy that file as a fallback.
    """
    latest = run_dir.parent / "latest"
    try:
        if latest.is_symlink() or latest.exists():
            latest.unlink()
        latest.symlink_to(run_dir.name)
    except OSError:
        # Best-effort fallback for filesystems without symlink support.
        latest.mkdir(parents=True, exist_ok=True)
        src = run_dir / "best_overrides.json"
        if src.exists():
            (latest / "best_overrides.json").write_text(src.read_text())


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-paths", type=int, default=2000)
    parser.add_argument("--n-steps", type=int, default=504)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--smoketest",
        action="store_true",
        help="Tiny grid (single config per axis) used by tests / quick CI",
    )
    args = parser.parse_args()

    if args.smoketest:
        grid = TuningGridConfig(
            kappa_grid=(1e-12,),
            gamma_grid=(0.5,),
            t_eff_grid=(0.083,),
            mu_q_grid=(0.0,),
            sigma_q_grid=(0.10,),
            n_paths=args.n_paths if args.n_paths < 2000 else 1000,
            n_steps=args.n_steps if args.n_steps < 504 else 252,
            seed=args.seed,
        )
    else:
        grid = TuningGridConfig(n_paths=args.n_paths, n_steps=args.n_steps, seed=args.seed)

    run_dir = make_run_dir("marketron_tuning", seed=grid.seed)
    save_config(run_dir, grid)
    print(f"Tuning run dir: {run_dir}")

    with timed("tuning_grid"):
        outcomes = run_tuning_grid(grid)

    parquet_path = run_dir / "grid_results.parquet"
    write_grid_results(outcomes, parquet_path)
    print(f"Wrote grid_results.parquet ({len(outcomes)} rows)")

    best_per_set = select_best_overrides_per_set(outcomes)
    best_global = select_best_overrides_global(outcomes)
    overrides_path = run_dir / "best_overrides.json"
    write_best_overrides(best_per_set, best_global, overrides_path)
    print(f"Wrote best_overrides.json: {overrides_path}")
    link_latest(run_dir)

    print()
    print("Best overrides per parameter set:")
    for parameter_set, overrides in best_per_set.items():
        print(f"  {parameter_set}: {overrides}")
    print(f"  global winner: {best_global}")

    # Summarize headline counts at the per-set winners. Excludes
    # `table_2_synthetic` because it has only placeholder zero-skew/zero-kurt
    # targets (Marketron didn't publish a moments table for that set).
    from reflexive_options.experiments.synthetic_replication import (
        _HEADLINE_PARAMETER_SETS,
    )

    total_shape_match = 0
    total_shape_cells = 0
    for o in outcomes:
        if o.parameter_set not in _HEADLINE_PARAMETER_SETS:
            continue
        winner = best_per_set.get(o.parameter_set)
        if winner is None:
            continue
        if (
            o.kappa == winner.kappa
            and o.gamma == winner.leverage
            and abs(o.memory_decay - winner.memory_decay) < 1e-9
            and o.mu_q == winner.oi_mu_q
            and o.sigma_q == winner.oi_sigma_q
        ):
            total_shape_match += o.shape_sign_match_total
            total_shape_cells += o.shape_total
    rate = (total_shape_match / total_shape_cells) if total_shape_cells else 0.0
    print()
    print(
        f"Headline: {total_shape_match}/{total_shape_cells} shape-feature cells match "
        f"across the {len(_HEADLINE_PARAMETER_SETS)} Marketron parameter sets with "
        f"published moment tables, at the per-set tuned coupling (rate = {rate:.2%})"
    )

    # Repo-relative path so downstream consumers can find the run.
    rel_dir = run_dir.relative_to(REPO_ROOT) if run_dir.is_relative_to(REPO_ROOT) else run_dir
    print(f"Results in: {rel_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
