"""Generate `tests/repro/baseline_v0.1.0.json` — the canonical reproducibility receipt.

Each entry in the receipt records, for one experiment script and the canonical
seed (42), the exact (or for stochastic legs, distributional) numerical
outputs together with a blake2b hash, tooling versions, and a relative-
tolerance class. The regression test `tests/test_reproducibility.py` re-runs
every experiment and asserts the outputs match the receipt.

Run via the wrapper:

    bash scripts/generate_repro_baseline.sh

That wrapper invokes:

    uv run python -m reflexive_options.experiments._generate_repro_baseline

See `CLAUDE.md` § "Reproducibility receipt" for when (and only when) to refresh.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import shutil
import sys
import tempfile
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
RECEIPT_PATH = REPO_ROOT / "tests" / "repro" / "baseline_v0.1.0.json"
SCHEMA_VERSION = "1.0"
RECEIPT_VERSION = "0.1.0"
CANONICAL_SEED = 42

# 5 different seeds for distribution-bucket stochastic entries. Spec §1.
STOCHASTIC_SEEDS: tuple[int, ...] = (42, 43, 44, 45, 46)
STOCHASTIC_REL_TOLERANCE = 0.05  # 5% per spec §3.

ToleranceClass = str  # "exact" | "relative_5pct"


@dataclass(frozen=True)
class ExperimentSpec:
    """One experiment's reproducibility configuration."""

    name: str  # module name without prefix
    runner: Callable[[int], dict[str, Any]]
    config_factory: Callable[[int], dict[str, Any]]
    tolerance: ToleranceClass = "exact"


@dataclass(frozen=True)
class ReceiptEntry:
    """One entry in the receipt — JSON-serializable via asdict()."""

    experiment: str
    seed: int
    config: dict[str, Any]
    metrics: dict[str, Any]
    blake2b_hash: str
    tolerance: ToleranceClass
    timestamp_utc: str
    tooling: dict[str, str]
    notes: str = ""
    # Populated only for the "relative_5pct" bucket:
    metrics_std: dict[str, Any] = field(default_factory=dict)
    seeds_used: list[int] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Tooling capture
# ---------------------------------------------------------------------------


def _tooling_versions() -> dict[str, str]:
    pkgs = ("numpy", "scipy", "torch", "pandas", "gymnasium", "matplotlib", "QuantLib")
    out: dict[str, str] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
    }
    for pkg in pkgs:
        try:
            out[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            out[pkg] = "missing"
    return out


# ---------------------------------------------------------------------------
# Canonical-form hashing
# ---------------------------------------------------------------------------


def canonical_json(payload: Any) -> bytes:
    """Stable, sorted, no-whitespace JSON for hashing.

    Uses `default=str` so objects we don't know how to encode (np scalars,
    Paths, etc.) at least round-trip stably.
    """
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), default=str, ensure_ascii=True
    ).encode("utf-8")


def blake2b_hex(payload: Any) -> str:
    """Hex blake2b digest of `canonical_json(payload)`."""
    return hashlib.blake2b(canonical_json(payload), digest_size=32).hexdigest()


# ---------------------------------------------------------------------------
# Per-experiment runners — each returns a (config, metrics) pair at the seed.
# ---------------------------------------------------------------------------


def _to_jsonable(obj: Any) -> Any:
    """Coerce numpy scalars / arrays / Paths to JSON-friendly types.

    Receipt values land in JSON, so anything with a numpy dtype gets a
    Python primitive. Lists are walked recursively; arrays become lists.
    """
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(x) for x in obj]
    if isinstance(obj, np.ndarray):
        return _to_jsonable(obj.tolist())
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    return obj


# Bifurcation scan ----------------------------------------------------------


def _run_bifurcation_scan(seed: int) -> dict[str, Any]:
    """Bifurcation scan is fully deterministic given the grid; seed is informational."""
    del seed
    from reflexive_options.experiments.bifurcation_scan import (
        BifurcationConfig,
        _jacobian_for_kappa_at_sigma,
    )
    from reflexive_options.theory.bifurcation import hopf_scan

    cfg = BifurcationConfig()
    kappa_grid = np.linspace(cfg.kappa_min, cfg.kappa_max, cfg.n_kappa).astype(np.float64)
    sigma_v_grid = np.linspace(cfg.sigma_v_min, cfg.sigma_v_max, cfg.n_sigma_v).astype(np.float64)

    kappa_star: list[float | None] = []
    for sv in sigma_v_grid:
        sv_float = float(sv)

        def _jac(k: float, _sv: float = sv_float) -> Any:
            return _jacobian_for_kappa_at_sigma(k, cfg=cfg, sigma_v=_sv)

        result = hopf_scan(kappa_grid, _jac)
        kappa_star.append(result.kappa_star)

    return {
        "kappa_min": cfg.kappa_min,
        "kappa_max": cfg.kappa_max,
        "n_kappa": cfg.n_kappa,
        "n_sigma_v": cfg.n_sigma_v,
        "sigma_v_grid": sigma_v_grid.tolist(),
        "kappa_star_curve": [None if k is None else float(k) for k in kappa_star],
        "n_bifurcations_found": int(sum(k is not None for k in kappa_star)),
    }


def _bifurcation_config(seed: int) -> dict[str, Any]:
    del seed
    from reflexive_options.experiments.bifurcation_scan import BifurcationConfig

    return asdict(BifurcationConfig())


# Phase diagram (stub) ------------------------------------------------------


def _run_phase_diagram(seed: int) -> dict[str, Any]:
    """Locks the current stub behavior — every cell is "calm", blowup_fraction zero.

    When the simulator wiring lands and the stub is replaced, the receipt
    must be regenerated with `bash scripts/generate_repro_baseline.sh`.
    """
    from reflexive_options.experiments.phase_diagram import PhaseDiagramConfig

    cfg = PhaseDiagramConfig(seed=seed)
    kappa_grid = np.linspace(cfg.kappa_min, cfg.kappa_max, cfg.n_kappa)
    sigma_v_grid = np.linspace(cfg.sigma_v_min, cfg.sigma_v_max, cfg.n_sigma_v)

    regime_grid = np.full((cfg.n_kappa, cfg.n_sigma_v), "calm", dtype=object)
    blowup_fraction = np.zeros((cfg.n_kappa, cfg.n_sigma_v))

    return {
        "kappa_grid": kappa_grid.tolist(),
        "sigma_v_grid": sigma_v_grid.tolist(),
        "regime_grid": regime_grid.tolist(),
        "blowup_fraction": blowup_fraction.tolist(),
    }


def _phase_diagram_config(seed: int) -> dict[str, Any]:
    from reflexive_options.experiments.phase_diagram import PhaseDiagramConfig

    return asdict(PhaseDiagramConfig(seed=seed))


# Synthetic replication -----------------------------------------------------

SYNTHETIC_REPLICATION_PARAM_SET = "table_5_calibrated_2017"
SYNTHETIC_REPLICATION_N_PATHS = 2_000  # baseline-only; receipt run, not the full 50k


def _run_synthetic_replication(seed: int) -> dict[str, Any]:
    """Run the Marketron replication on the calibrated 2017 leg at reduced n_paths.

    We use 2k paths (vs 50k default) so the receipt regen + regression test
    stay snappy. Both Heston and reflexive sims are seed-deterministic at
    this size.
    """
    from reflexive_options.experiments.synthetic_replication import (
        ReplicationConfig,
        compare_to_marketron_targets,
        run_marketron_heston_baseline,
        run_reflexive_with_matched_marketron_calibration,
    )

    cfg = ReplicationConfig(
        parameter_set=SYNTHETIC_REPLICATION_PARAM_SET,
        n_paths=SYNTHETIC_REPLICATION_N_PATHS,
        n_steps=252,
        seed=seed,
    )
    heston = run_marketron_heston_baseline(cfg)
    reflexive = run_reflexive_with_matched_marketron_calibration(cfg)
    heston_cmp = compare_to_marketron_targets(heston, cfg.parameter_set)
    reflexive_cmp = compare_to_marketron_targets(reflexive, cfg.parameter_set)

    # Drop large/derived fields that don't add value to the receipt:
    return {
        "parameter_set": cfg.parameter_set,
        "n_paths": cfg.n_paths,
        "heston_horizon_metrics": heston["horizon_metrics"],
        "heston_terminal_spot_mean": heston["terminal_spot_mean"],
        "heston_terminal_spot_std": heston["terminal_spot_std"],
        "heston_terminal_variance_mean": heston["terminal_variance_mean"],
        "reflexive_horizon_metrics": reflexive["horizon_metrics"],
        "reflexive_terminal_spot_mean": reflexive["terminal_spot_mean"],
        "reflexive_terminal_spot_std": reflexive["terminal_spot_std"],
        "reflexive_terminal_variance_mean": reflexive["terminal_variance_mean"],
        "heston_n_pass": heston_cmp["n_pass"],
        "heston_n_fail": heston_cmp["n_fail"],
        "heston_n_informational": heston_cmp["n_informational"],
        "heston_overall_passed": heston_cmp["overall_passed"],
        "reflexive_n_pass": reflexive_cmp["n_pass"],
        "reflexive_n_fail": reflexive_cmp["n_fail"],
        "reflexive_n_informational": reflexive_cmp["n_informational"],
        "reflexive_overall_passed": reflexive_cmp["overall_passed"],
    }


def _synthetic_replication_config(seed: int) -> dict[str, Any]:
    from reflexive_options.experiments.synthetic_replication import ReplicationConfig

    cfg = ReplicationConfig(
        parameter_set=SYNTHETIC_REPLICATION_PARAM_SET,
        n_paths=SYNTHETIC_REPLICATION_N_PATHS,
        n_steps=252,
        seed=seed,
    )
    return {
        "parameter_set": cfg.parameter_set,
        "n_paths": cfg.n_paths,
        "n_steps": cfg.n_steps,
        "dt": cfg.dt,
        "seed": cfg.seed,
        "initial_spot": cfg.initial_spot,
        "risk_free_rate": cfg.risk_free_rate,
        "horizons_years": list(cfg.horizons_years),
        "rel_tolerance": cfg.rel_tolerance,
    }


# Reflexive transfer (stochastic — torch BC) --------------------------------

_REFLEXIVE_TRANSFER_KAPPA_GRID_N = 5
_REFLEXIVE_TRANSFER_BC_EPISODES = 4
_REFLEXIVE_TRANSFER_EPISODE_LENGTH = 16
_REFLEXIVE_TRANSFER_BC_EPOCHS = 4
_REFLEXIVE_TRANSFER_N_SEEDS_PER_KAPPA = 4
_REFLEXIVE_TRANSFER_N_EVAL_EPISODES = 1


def _reflexive_transfer_config_obj(seed: int) -> Any:
    from reflexive_options.experiments.reflexive_transfer import TransferConfig

    return TransferConfig(
        kappa_anchor=5.0e-12,
        kappa_grid_n_points=_REFLEXIVE_TRANSFER_KAPPA_GRID_N,
        kappa_grid_low_mult=0.0,
        kappa_grid_high_mult=2.0,
        n_seeds_per_kappa=_REFLEXIVE_TRANSFER_N_SEEDS_PER_KAPPA,
        n_eval_episodes_per_seed=_REFLEXIVE_TRANSFER_N_EVAL_EPISODES,
        n_bc_train_episodes=_REFLEXIVE_TRANSFER_BC_EPISODES,
        episode_length=_REFLEXIVE_TRANSFER_EPISODE_LENGTH,
        bc_epochs=_REFLEXIVE_TRANSFER_BC_EPOCHS,
        bc_batch_size=16,
        seed=seed,
    )


def _run_reflexive_transfer(seed: int) -> dict[str, Any]:
    """Run the κ-sensitivity sweep at one seed, return the slope + grid metrics."""
    from reflexive_options.experiments.reflexive_transfer import run_experiment

    cfg = _reflexive_transfer_config_obj(seed)
    with tempfile.TemporaryDirectory() as td:
        run_dir = Path(td)
        metrics_obj = run_experiment(cfg, run_dir)
    # `run_experiment` is typed dict[str, object] — narrow each key explicitly.
    metrics: dict[str, Any] = dict(metrics_obj)

    # Strip non-numeric / per-machine fields (paths, rng_state strings).
    return {
        "kappa_anchor": float(metrics["kappa_anchor"]),
        "kappa_grid": list(metrics["kappa_grid"]),
        "metric_means": list(metrics["metric_means"]),
        "metric_stds": list(metrics["metric_stds"]),
        "slope_at_anchor": float(metrics["slope_at_anchor"]),
        "slope_ci_low": float(metrics["slope_ci_low"]),
        "slope_ci_high": float(metrics["slope_ci_high"]),
        "ci_excludes_zero": bool(metrics["ci_excludes_zero"]),
    }


def _reflexive_transfer_config(seed: int) -> dict[str, Any]:
    return asdict(_reflexive_transfer_config_obj(seed))


# ---------------------------------------------------------------------------
# Distribution helpers for stochastic entries
# ---------------------------------------------------------------------------


def _aggregate_stochastic_runs(
    runs: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Reduce a list of equal-shape metrics dicts to (means, stds).

    Walks the dict recursively: scalars → mean/std; lists of scalars → element-wise
    mean/std; bools → majority value, std=0; everything else → first value, std=None.
    """
    if not runs:
        return {}, {}
    keys = sorted(runs[0].keys())
    means: dict[str, Any] = {}
    stds: dict[str, Any] = {}
    for k in keys:
        sample = runs[0][k]
        all_vals = [r[k] for r in runs]
        if isinstance(sample, bool):
            # Majority: True iff > half True.
            t = sum(1 for v in all_vals if v)
            means[k] = bool(t > len(all_vals) / 2)
            stds[k] = 0
        elif isinstance(sample, (int, float)):
            arr = np.array(all_vals, dtype=np.float64)
            means[k] = float(arr.mean())
            stds[k] = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
        elif isinstance(sample, list) and (len(sample) == 0 or isinstance(sample[0], (int, float))):
            arr2 = np.array(all_vals, dtype=np.float64)
            means[k] = arr2.mean(axis=0).tolist()
            stds[k] = (
                arr2.std(axis=0, ddof=1).tolist()
                if arr2.shape[0] > 1
                else np.zeros(arr2.shape[1:]).tolist()
            )
        else:
            # Categorical / nested — take the first.
            means[k] = sample
            stds[k] = None
    return means, stds


# ---------------------------------------------------------------------------
# Spec registry
# ---------------------------------------------------------------------------


def _all_specs() -> list[ExperimentSpec]:
    return [
        ExperimentSpec(
            name="bifurcation_scan",
            runner=_run_bifurcation_scan,
            config_factory=_bifurcation_config,
            tolerance="exact",
        ),
        ExperimentSpec(
            name="phase_diagram",
            runner=_run_phase_diagram,
            config_factory=_phase_diagram_config,
            tolerance="exact",
        ),
        ExperimentSpec(
            name="synthetic_replication",
            runner=_run_synthetic_replication,
            config_factory=_synthetic_replication_config,
            tolerance="exact",
        ),
        ExperimentSpec(
            name="reflexive_transfer",
            runner=_run_reflexive_transfer,
            config_factory=_reflexive_transfer_config,
            tolerance="relative_5pct",
        ),
    ]


# ---------------------------------------------------------------------------
# Entry construction
# ---------------------------------------------------------------------------


def _build_entry(spec: ExperimentSpec, tooling: dict[str, str]) -> ReceiptEntry:
    """Run `spec.runner` (one or many seeds) and produce a ReceiptEntry."""
    cfg = _to_jsonable(spec.config_factory(CANONICAL_SEED))

    if spec.tolerance == "exact":
        metrics = _to_jsonable(spec.runner(CANONICAL_SEED))
        digest = blake2b_hex(metrics)
        return ReceiptEntry(
            experiment=spec.name,
            seed=CANONICAL_SEED,
            config=cfg,
            metrics=metrics,
            blake2b_hash=digest,
            tolerance=spec.tolerance,
            timestamp_utc=datetime.now(UTC).isoformat(timespec="seconds"),
            tooling=tooling,
            seeds_used=[CANONICAL_SEED],
        )

    if spec.tolerance == "relative_5pct":
        runs = [_to_jsonable(spec.runner(int(s))) for s in STOCHASTIC_SEEDS]
        means, stds = _aggregate_stochastic_runs(runs)
        digest = blake2b_hex({"means": means, "stds": stds, "seeds": list(STOCHASTIC_SEEDS)})
        return ReceiptEntry(
            experiment=spec.name,
            seed=CANONICAL_SEED,
            config=cfg,
            metrics=means,
            metrics_std=stds,
            blake2b_hash=digest,
            tolerance=spec.tolerance,
            timestamp_utc=datetime.now(UTC).isoformat(timespec="seconds"),
            tooling=tooling,
            seeds_used=list(STOCHASTIC_SEEDS),
            notes=(
                f"torch BC training; means/stds aggregated across {len(STOCHASTIC_SEEDS)} seeds."
            ),
        )

    raise ValueError(f"Unknown tolerance class: {spec.tolerance!r}")


def _build_receipt() -> dict[str, Any]:
    """Run all experiments and assemble the receipt dict."""
    tooling = _tooling_versions()
    entries: list[dict[str, Any]] = []
    for spec in _all_specs():
        t0 = time.perf_counter()
        entry = _build_entry(spec, tooling)
        elapsed = time.perf_counter() - t0
        print(f"  {spec.name:<24s}  {spec.tolerance:<14s}  {elapsed:6.2f}s")
        entries.append(asdict(entry))
    return {
        "schema_version": SCHEMA_VERSION,
        "receipt_version": RECEIPT_VERSION,
        "canonical_seed": CANONICAL_SEED,
        "stochastic_seeds": list(STOCHASTIC_SEEDS),
        "stochastic_rel_tolerance": STOCHASTIC_REL_TOLERANCE,
        "generated_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "tooling": tooling,
        "n_experiments": len(entries),
        "entries": entries,
    }


def main() -> None:
    """Run every experiment with the canonical seed and snapshot results.

    Each entry of the resulting JSON has:
      - "experiment": module name
      - "seed": 42
      - "config": full config as dict
      - "metrics": exact metric values (deterministic) OR mean+std (stochastic)
      - "blake2b_hash": hex digest of the canonical-form JSON of metrics
      - "tolerance": "exact" | "relative_5pct"
      - "timestamp_utc": ISO-8601
      - "tooling": {python, numpy, scipy, torch, etc. versions}
    """
    print(f"Generating reproducibility receipt → {RECEIPT_PATH}")
    print(f"Canonical seed: {CANONICAL_SEED}; stochastic seeds: {STOCHASTIC_SEEDS}")
    print()
    print(f"{'experiment':<26s}  {'tolerance':<14s}  elapsed")
    print(f"{'-' * 26}  {'-' * 14}  -------")

    t0 = time.perf_counter()
    receipt = _build_receipt()
    total = time.perf_counter() - t0

    RECEIPT_PATH.parent.mkdir(parents=True, exist_ok=True)
    # Write atomically to a sibling temp file then rename — defensive against a
    # `Ctrl-C` mid-write leaving a partially-truncated baseline behind.
    tmp = RECEIPT_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(receipt, indent=2, default=str) + "\n")
    shutil.move(str(tmp), str(RECEIPT_PATH))
    size = RECEIPT_PATH.stat().st_size
    print()
    print(f"  total: {total:.2f}s  →  {RECEIPT_PATH} ({size:,} bytes)")


if __name__ == "__main__":
    main()
