"""Tests for the Marketron synthetic-replication experiment.

Targets the four behaviors specified in the experiment task:
  1. End-to-end run on table_2_synthetic produces a metrics.json with the
     expected keys.
  2. Hardcoded Marketron moment targets cover the documented horizons.
  3. Comparison passes when measured == target exactly.
  4. Comparison fails when measured is shifted past tolerance.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from reflexive_options.experiments.synthetic_replication import (
    DEFAULT_REL_TOLERANCE,
    INFORMATIONAL_MOMENTS,
    MARKETRON_MOMENT_TARGETS,
    MARKETRON_PARAM_SETS,
    ReplicationConfig,
    annualized_log_return_moments,
    compare_to_marketron_targets,
    run_marketron_heston_baseline,
    run_reflexive_with_matched_marketron_calibration,
)

# ---------------------------------------------------------------------------
# Hardcoded targets dict
# ---------------------------------------------------------------------------


def test_marketron_target_dict_has_known_horizons() -> None:
    """Every published parameter set has at least one moment-target horizon."""
    for set_name in MARKETRON_PARAM_SETS:
        assert set_name in MARKETRON_MOMENT_TARGETS, (
            f"missing moment targets for param set {set_name!r}"
        )
        targets = MARKETRON_MOMENT_TARGETS[set_name]
        assert len(targets) >= 1
        for horizon, moments in targets.items():
            assert horizon > 0
            assert {"mean", "vol", "skew", "excess_kurt"}.issubset(moments)


def test_marketron_table_5_known_values() -> None:
    """Spot-check Table 8 (table_5_calibrated_2017) entry at horizon 1y."""
    # Brief Table 8 row 5: 1.00 yr → mean 0.3558, vol 0.3333, skew 0.0533, kurt 0.0175.
    targets = MARKETRON_MOMENT_TARGETS["table_5_calibrated_2017"]
    assert 1.00 in targets
    cell = targets[1.00]
    assert cell["mean"] == pytest.approx(0.3558)
    assert cell["vol"] == pytest.approx(0.3333)
    assert cell["skew"] == pytest.approx(0.0533)
    assert cell["excess_kurt"] == pytest.approx(0.0175)


def test_marketron_table_6_known_values() -> None:
    """Spot-check Table 7 (table_6_calibrated_2020) entry at horizon 1y."""
    # Brief Table 7 row 5: 1.00 yr → mean 0.0855, vol 0.7423, skew 0.1688, kurt -0.0074.
    targets = MARKETRON_MOMENT_TARGETS["table_6_calibrated_2020"]
    assert 1.00 in targets
    cell = targets[1.00]
    assert cell["mean"] == pytest.approx(0.0855)
    assert cell["vol"] == pytest.approx(0.7423)
    assert cell["skew"] == pytest.approx(0.1688)
    assert cell["excess_kurt"] == pytest.approx(-0.0074)


# ---------------------------------------------------------------------------
# compare_to_marketron_targets — pass / fail behavior
# ---------------------------------------------------------------------------


def _exact_match_metrics(target_set: str) -> dict[str, dict[str, dict[str, float]]]:
    """Build a `our_metrics`-shaped dict that matches Marketron targets exactly."""
    horizon_metrics: dict[str, dict[str, float]] = {}
    for horizon, moments in MARKETRON_MOMENT_TARGETS[target_set].items():
        horizon_metrics[f"{horizon:.4f}y"] = dict(moments)
    return {"horizon_metrics": horizon_metrics}


def test_compare_to_marketron_targets_passes_at_match() -> None:
    target_set = "table_5_calibrated_2017"
    metrics = _exact_match_metrics(target_set)
    result = compare_to_marketron_targets(metrics, target_set)

    assert result["overall_passed"] is True
    assert result["n_fail"] == 0
    assert result["n_pass"] > 0
    # Every cell either passes or is informational (vol).
    for _horizon, per_horizon in result["per_cell"].items():
        for moment, cell in per_horizon.items():
            assert cell["status"] in {"pass", "informational"}
            assert cell["relative_error"] == pytest.approx(0.0, abs=1e-12)
            if moment in INFORMATIONAL_MOMENTS:
                assert cell["status"] == "informational"


def test_compare_to_marketron_targets_fails_at_disagreement() -> None:
    target_set = "table_5_calibrated_2017"
    metrics = _exact_match_metrics(target_set)
    # Shift every (non-informational) moment by 50% relative — well past the
    # 8% tolerance. Use a multiplicative bump so even small targets get pushed
    # outside the abs(target)=1e-3 floor in `_relative_error`.
    for horizon_metrics in metrics["horizon_metrics"].values():
        for moment_name in list(horizon_metrics):
            if moment_name in INFORMATIONAL_MOMENTS:
                continue
            v = horizon_metrics[moment_name]
            # Push every moment well outside the 8% tolerance band.
            horizon_metrics[moment_name] = (v + 1.0) * 1.5

    result = compare_to_marketron_targets(metrics, target_set)
    assert result["overall_passed"] is False
    assert result["n_fail"] > 0


def test_compare_to_marketron_targets_unknown_set_raises() -> None:
    with pytest.raises(KeyError):
        compare_to_marketron_targets({"horizon_metrics": {}}, "not_a_real_set")


def test_compare_to_marketron_targets_skips_unmeasured_horizons() -> None:
    """Cells in the target dict that we did not measure are counted as skipped."""
    target_set = "table_5_calibrated_2017"
    metrics: dict[str, dict[str, dict[str, float]]] = {"horizon_metrics": {}}
    result = compare_to_marketron_targets(metrics, target_set)
    assert result["n_pass"] == 0
    assert result["n_fail"] == 0
    assert result["n_skipped"] > 0
    # No measured cells → cannot pass overall.
    assert result["overall_passed"] is False


# ---------------------------------------------------------------------------
# annualized_log_return_moments — math sanity
# ---------------------------------------------------------------------------


def test_annualized_log_return_moments_constant_paths_zero_vol() -> None:
    import numpy as np

    n_paths, n_steps = 16, 80
    spots = np.full((n_paths, n_steps + 1), 100.0)
    out = annualized_log_return_moments(spots, horizon_years=0.25, dt=1 / 252, initial_spot=100.0)
    assert out["mean"] == pytest.approx(0.0)
    assert out["vol"] == pytest.approx(0.0)
    assert out["skew"] == pytest.approx(0.0)
    assert out["excess_kurt"] == pytest.approx(0.0)


def test_annualized_log_return_moments_lognormal() -> None:
    """Lognormal terminal → annualized vol approaches σ as n_paths grows."""
    import numpy as np

    rng = np.random.default_rng(0)
    n_paths, n_steps = 50_000, 252
    dt = 1 / 252
    sigma = 0.20
    drift = 0.05
    z = rng.standard_normal((n_paths, n_steps))
    log_spots = np.log(100.0) + np.cumsum(
        (drift - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z, axis=1
    )
    spots = np.empty((n_paths, n_steps + 1))
    spots[:, 0] = 100.0
    spots[:, 1:] = np.exp(log_spots)
    out = annualized_log_return_moments(spots, horizon_years=1.0, dt=dt, initial_spot=100.0)
    # Annualized vol is σ.
    assert out["vol"] == pytest.approx(sigma, abs=2e-3)
    # Annualized mean is drift - 0.5 σ²  (Itô).
    assert out["mean"] == pytest.approx(drift - 0.5 * sigma**2, abs=5e-3)


# ---------------------------------------------------------------------------
# End-to-end run on table_2_synthetic
# ---------------------------------------------------------------------------


def test_marketron_table_2_synthetic_runs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Full end-to-end with the table_2_synthetic param set succeeds and dumps metrics.json."""
    # Tiny config so the test is fast; we only need the wiring to work end-to-end.
    cfg = ReplicationConfig(
        parameter_set="table_2_synthetic",
        n_paths=200,
        n_steps=63,  # ~quarter year at 1/252
        dt=1 / 252,
        seed=7,
        horizons_years=(0.0833, 0.25),
    )
    heston = run_marketron_heston_baseline(cfg)
    reflexive = run_reflexive_with_matched_marketron_calibration(cfg)

    for leg in (heston, reflexive):
        assert "horizon_metrics" in leg
        assert "0.0833y" in leg["horizon_metrics"]
        assert "0.2500y" in leg["horizon_metrics"]
        for h_metrics in leg["horizon_metrics"].values():
            assert {"mean", "vol", "skew", "excess_kurt"}.issubset(h_metrics)
            for v in h_metrics.values():
                assert isinstance(v, float)

    # Comparison harness must accept these and produce a structured result.
    h_cmp = compare_to_marketron_targets(heston, cfg.parameter_set)
    r_cmp = compare_to_marketron_targets(reflexive, cfg.parameter_set)
    for cmp in (h_cmp, r_cmp):
        assert "overall_passed" in cmp
        assert "per_cell" in cmp
        assert cmp["rel_tolerance"] == pytest.approx(DEFAULT_REL_TOLERANCE)


def test_marketron_table_2_synthetic_writes_metrics_json(tmp_path: Path) -> None:
    """Smoke-test the file-writing side without invoking the full CLI: build a
    minimal payload through the same plumbing main() uses, write it via
    save_metrics, and check the JSON has the expected top-level keys."""
    from reflexive_options.experiments._common import save_metrics

    cfg = ReplicationConfig(
        parameter_set="table_2_synthetic",
        n_paths=128,
        n_steps=21,
        dt=1 / 252,
        seed=11,
        horizons_years=(0.0833,),
    )
    heston = run_marketron_heston_baseline(cfg)
    reflexive = run_reflexive_with_matched_marketron_calibration(cfg)
    h_cmp = compare_to_marketron_targets(heston, cfg.parameter_set)
    r_cmp = compare_to_marketron_targets(reflexive, cfg.parameter_set)

    payload = {
        "config": {
            "parameter_set": cfg.parameter_set,
            "n_paths": cfg.n_paths,
            "n_steps": cfg.n_steps,
            "dt": cfg.dt,
            "seed": cfg.seed,
            "rel_tolerance": cfg.rel_tolerance,
        },
        "marketron_param_set_raw": MARKETRON_PARAM_SETS[cfg.parameter_set],
        "heston_baseline": heston,
        "reflexive_matched": reflexive,
        "heston_vs_marketron": h_cmp,
        "reflexive_vs_marketron": r_cmp,
    }
    save_metrics(tmp_path, payload)
    loaded = json.loads((tmp_path / "metrics.json").read_text())
    expected_keys = {
        "config",
        "marketron_param_set_raw",
        "heston_baseline",
        "reflexive_matched",
        "heston_vs_marketron",
        "reflexive_vs_marketron",
    }
    assert expected_keys.issubset(loaded)
    assert loaded["heston_vs_marketron"]["target_set"] == cfg.parameter_set
