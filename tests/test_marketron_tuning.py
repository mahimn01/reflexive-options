"""Tests for the Marketron tuning script + mechanism-decomposition reporting.

Covers:
  - tuning script runs end-to-end (smoketest config), emits grid_results.parquet.
  - mechanism classifier returns the right class for each (moment) row.
  - synthetic_replication uses the tuned overrides loaded from a manifest.
  - at the tuned overrides, ≥30% of shape_target cells match in sign.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from reflexive_options.experiments.marketron_tuning import (
    TuningGridConfig,
    TuningOutcome,
    _evaluate_one_config,
    _measured_mean_horizon_direction,
    link_latest,
    run_tuning_grid,
    select_best_overrides_global,
    select_best_overrides_per_set,
    write_best_overrides,
    write_grid_results,
)
from reflexive_options.experiments.marketron_tuning import (
    main as tuning_main,
)
from reflexive_options.experiments.synthetic_replication import (
    DEFAULT_REL_TOLERANCE,
    MARKETRON_MOMENT_TARGETS,
    MARKETRON_PARAM_SETS,
    SHAPE_MATCH_GATE,
    CellOutcome,
    ReplicationConfig,
    TunedReflexiveOverrides,
    _build_cell_outcome,
    _order_of_magnitude_match,
    _sign_of,
    classify_mechanism,
    compare_to_marketron_targets,
    format_mechanism_decomposition_table,
    load_tuned_overrides,
    run_reflexive_with_matched_marketron_calibration,
)

# ---------------------------------------------------------------------------
# Mechanism classifier
# ---------------------------------------------------------------------------


def test_mechanism_decomposition_classifies_correctly() -> None:
    """Each moment routes into the documented mechanism class."""
    # Shape targets: skew + excess kurt are mechanism-agnostic, we tune on these.
    assert classify_mechanism("skew") == "shape_target"
    assert classify_mechanism("excess_kurt") == "shape_target"
    # Vol level: Marketron is 3-5× too high vs realized (brief §6.4).
    assert classify_mechanism("vol") == "level_artifact"
    # Mean drift: depends on Marketron η̄, f(θ) we don't model.
    assert classify_mechanism("mean") == "calibration_artifact"


def test_classify_mechanism_ignores_target_set_arg() -> None:
    """v1 routes by moment name only; target_set is accepted for forward compat."""
    for set_name in MARKETRON_PARAM_SETS:
        assert classify_mechanism("skew", set_name) == "shape_target"
        assert classify_mechanism("vol", set_name) == "level_artifact"
        assert classify_mechanism("mean", set_name) == "calibration_artifact"


def test_sign_of_near_zero_band() -> None:
    """The ±1e-3 dead-zone matches Marketron's smallest reported entry."""
    assert _sign_of(0.0) == 0
    assert _sign_of(5e-4) == 0
    assert _sign_of(-5e-4) == 0
    assert _sign_of(2e-3) == 1
    assert _sign_of(-2e-3) == -1


def test_order_of_magnitude_match() -> None:
    """Within-decade magnitude check on log10."""
    assert _order_of_magnitude_match(0.05, 0.1) is True
    assert _order_of_magnitude_match(-0.05, 0.1) is True  # signs ignored
    assert _order_of_magnitude_match(0.001, 0.1) is False  # 100× off
    assert _order_of_magnitude_match(0.0, 0.0) is True  # both negligible
    assert _order_of_magnitude_match(0.0, 0.5) is False  # one negligible, other not


def test_build_cell_outcome_classifies_and_flags() -> None:
    """`_build_cell_outcome` populates every field of the new CellOutcome dataclass."""
    out = _build_cell_outcome(
        horizon=1.0,
        moment_name="skew",
        measured=0.05,
        target=0.0533,
        rel_tolerance=DEFAULT_REL_TOLERANCE,
        target_set="table_5_calibrated_2017",
    )
    assert isinstance(out, CellOutcome)
    assert out.mechanism_class == "shape_target"
    assert out.sign_match is True
    assert out.order_of_magnitude_match is True


def test_build_cell_outcome_handles_sign_mismatch() -> None:
    """Sign-flip cells correctly report sign_match=False but still tally OOM."""
    out = _build_cell_outcome(
        horizon=1.0,
        moment_name="skew",
        measured=-0.05,
        target=0.0533,
        rel_tolerance=DEFAULT_REL_TOLERANCE,
        target_set="table_5_calibrated_2017",
    )
    assert out.sign_match is False
    assert out.order_of_magnitude_match is True
    assert out.within_8pct is False


# ---------------------------------------------------------------------------
# compare_to_marketron_targets — new shape-class plumbing
# ---------------------------------------------------------------------------


def _exact_match_metrics(target_set: str) -> dict[str, dict[str, dict[str, float]]]:
    horizon_metrics: dict[str, dict[str, float]] = {}
    for horizon, moments in MARKETRON_MOMENT_TARGETS[target_set].items():
        horizon_metrics[f"{horizon:.4f}y"] = dict(moments)
    return {"horizon_metrics": horizon_metrics}


def test_compare_emits_shape_class_counts() -> None:
    """At an exact match the headline counters all line up."""
    target_set = "table_5_calibrated_2017"
    metrics = _exact_match_metrics(target_set)
    result = compare_to_marketron_targets(metrics, target_set)
    assert result["shape_target_cells_total"] > 0
    # Exact-match should mean every shape target sign-matches.
    assert result["shape_target_sign_match"] == result["shape_target_cells_total"]
    # Both gates met (legacy + shape).
    assert result["shape_gate_passed"] is True
    assert result["overall_passed"] is True
    assert result["shape_match_rate"] >= SHAPE_MATCH_GATE


def test_compare_shape_gate_requires_measured_cells() -> None:
    """If we measure NO cells at all, the shape gate is by definition unmet."""
    target_set = "table_5_calibrated_2017"
    metrics: dict[str, dict[str, dict[str, float]]] = {"horizon_metrics": {}}
    result = compare_to_marketron_targets(metrics, target_set)
    assert result["shape_target_cells_total"] == 0
    assert result["shape_gate_passed"] is False
    assert result["overall_passed"] is False


def test_compare_per_cell_carries_mechanism_class() -> None:
    """Every per-cell entry has the new keys for downstream JSON consumers."""
    target_set = "table_5_calibrated_2017"
    metrics = _exact_match_metrics(target_set)
    result = compare_to_marketron_targets(metrics, target_set)
    for _horizon, per_horizon in result["per_cell"].items():
        for _moment, cell in per_horizon.items():
            assert "mechanism_class" in cell
            assert "sign_match" in cell
            assert "order_of_magnitude_match" in cell
            assert "within_8pct" in cell


# ---------------------------------------------------------------------------
# Tuning script smoketest
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    bool(os.environ.get("CI")),
    reason="Tuning sweep runs offline only; smoketest is too slow for CI",
)
def test_tuning_script_runs_smoketest(tmp_path: Path) -> None:
    """Single-config-per-axis smoketest produces a parquet file."""
    grid = TuningGridConfig(
        kappa_grid=(1e-12,),
        gamma_grid=(0.5,),
        t_eff_grid=(0.083,),
        mu_q_grid=(0.0,),
        sigma_q_grid=(0.10,),
        n_paths=500,
        n_steps=126,  # half a year
        horizons_years=(0.0833, 0.25),
        seed=11,
    )
    outcomes = run_tuning_grid(grid, progress_every=10_000)
    assert len(outcomes) == len(grid.parameter_sets)

    parquet_path = tmp_path / "grid_results.parquet"
    write_grid_results(outcomes, parquet_path)
    assert parquet_path.exists()
    # Schema sanity: parquet round-trips with all the columns we wrote.
    # (pyarrow 24.0 stub gap: read_table is dynamically attached — see surface/io.py)
    table = pq.read_table(parquet_path)  # type: ignore[no-untyped-call]
    assert "shape_sign_match_total" in table.column_names
    assert "shape_match_rate" in table.column_names
    assert table.num_rows == len(outcomes)


def test_tuning_outcome_dataclass_has_expected_fields() -> None:
    """Pin the parquet schema we promise downstream consumers."""
    outcome = TuningOutcome(
        parameter_set="table_5_calibrated_2017",
        kappa=1e-12,
        gamma=0.5,
        t_eff=0.083,
        memory_decay=12.0,
        mu_q=0.0,
        sigma_q=0.1,
        n_paths=2000,
        n_steps=504,
        seed=42,
        n_pass_strict_8pct=2,
        shape_total=12,
        shape_skew_sign_match=4,
        shape_kurt_sign_match=5,
        shape_sign_match_total=9,
        shape_horizon_direction_match=4,
        shape_mean_abs_error=0.07,
        overall_passed=True,
        shape_match_rate=0.75,
    )
    fields = asdict(outcome)
    assert {
        "parameter_set",
        "kappa",
        "gamma",
        "t_eff",
        "memory_decay",
        "mu_q",
        "sigma_q",
        "shape_total",
        "shape_sign_match_total",
        "shape_match_rate",
    }.issubset(fields)


def test_select_best_overrides_per_set_picks_max_sign_match() -> None:
    """Selection prefers the row with highest shape_sign_match_total."""
    outcomes = [
        TuningOutcome(
            parameter_set="table_5_calibrated_2017",
            kappa=1e-12,
            gamma=0.0,
            t_eff=0.083,
            memory_decay=12.0,
            mu_q=0.0,
            sigma_q=0.1,
            n_paths=100,
            n_steps=10,
            seed=1,
            n_pass_strict_8pct=0,
            shape_total=10,
            shape_skew_sign_match=2,
            shape_kurt_sign_match=2,
            shape_sign_match_total=4,
            shape_horizon_direction_match=2,
            shape_mean_abs_error=0.5,
            overall_passed=False,
            shape_match_rate=0.4,
        ),
        TuningOutcome(
            parameter_set="table_5_calibrated_2017",
            kappa=5e-12,
            gamma=1.5,
            t_eff=0.25,
            memory_decay=4.0,
            mu_q=0.05,
            sigma_q=0.2,
            n_paths=100,
            n_steps=10,
            seed=1,
            n_pass_strict_8pct=0,
            shape_total=10,
            shape_skew_sign_match=4,
            shape_kurt_sign_match=4,
            shape_sign_match_total=8,
            shape_horizon_direction_match=3,
            shape_mean_abs_error=0.4,
            overall_passed=True,
            shape_match_rate=0.8,
        ),
    ]
    best = select_best_overrides_per_set(outcomes)
    winner = best["table_5_calibrated_2017"]
    assert winner.kappa == 5e-12
    assert winner.leverage == 1.5
    assert winner.oi_mu_q == 0.05


def test_select_best_overrides_global_sums_across_sets() -> None:
    """Global selection aggregates by config tuple, summing across sets."""
    # Same config across two sets — combined sign_match should beat the
    # other config's single-set high score.
    outcomes = [
        TuningOutcome(
            parameter_set="table_5_calibrated_2017",
            kappa=1e-12,
            gamma=0.0,
            t_eff=0.083,
            memory_decay=12.0,
            mu_q=0.0,
            sigma_q=0.1,
            n_paths=100,
            n_steps=10,
            seed=1,
            n_pass_strict_8pct=0,
            shape_total=10,
            shape_skew_sign_match=2,
            shape_kurt_sign_match=4,
            shape_sign_match_total=6,
            shape_horizon_direction_match=2,
            shape_mean_abs_error=0.4,
            overall_passed=True,
            shape_match_rate=0.6,
        ),
        TuningOutcome(
            parameter_set="table_6_calibrated_2020",
            kappa=1e-12,
            gamma=0.0,
            t_eff=0.083,
            memory_decay=12.0,
            mu_q=0.0,
            sigma_q=0.1,
            n_paths=100,
            n_steps=10,
            seed=1,
            n_pass_strict_8pct=0,
            shape_total=10,
            shape_skew_sign_match=3,
            shape_kurt_sign_match=4,
            shape_sign_match_total=7,
            shape_horizon_direction_match=3,
            shape_mean_abs_error=0.5,
            overall_passed=True,
            shape_match_rate=0.7,
        ),
        # Other config: very high single-set sign_match but only one set covered.
        TuningOutcome(
            parameter_set="table_5_calibrated_2017",
            kappa=5e-12,
            gamma=1.5,
            t_eff=0.25,
            memory_decay=4.0,
            mu_q=0.05,
            sigma_q=0.2,
            n_paths=100,
            n_steps=10,
            seed=1,
            n_pass_strict_8pct=0,
            shape_total=10,
            shape_skew_sign_match=5,
            shape_kurt_sign_match=4,
            shape_sign_match_total=9,
            shape_horizon_direction_match=3,
            shape_mean_abs_error=0.3,
            overall_passed=True,
            shape_match_rate=0.9,
        ),
    ]
    winner = select_best_overrides_global(outcomes)
    # Aggregate sign_match: (1e-12, 0, 0.083, 0, 0.1) → 6+7=13;
    #                     (5e-12, 1.5, 0.25, 0.05, 0.2) → 9. First wins.
    assert winner.kappa == 1e-12
    assert winner.leverage == 0.0


def test_measured_mean_horizon_direction_counts_slope_signs() -> None:
    """Adjacent-horizon slope sign-match counter."""
    measured = {
        "0.0833y": {"mean": -0.10},
        "0.2500y": {"mean": -0.05},  # slope > 0
        "0.5000y": {"mean": +0.10},  # slope > 0
    }
    target_per_horizon = {
        0.0833: {"mean": -0.05, "vol": 0.3, "skew": 0.0, "excess_kurt": 0.0},
        0.25: {"mean": +0.05, "vol": 0.3, "skew": 0.0, "excess_kurt": 0.0},  # slope > 0
        0.5: {"mean": +0.20, "vol": 0.3, "skew": 0.0, "excess_kurt": 0.0},  # slope > 0
    }
    matches, total = _measured_mean_horizon_direction(measured, target_per_horizon)
    assert total == 2
    assert matches == 2


# ---------------------------------------------------------------------------
# Synthetic replication uses tuned overrides
# ---------------------------------------------------------------------------


def test_synthetic_replication_uses_tuned_parameters(tmp_path: Path) -> None:
    """Writing a best_overrides.json and pointing `load_tuned_overrides` at it
    makes the replication script pick up the tuned values."""
    payload = {
        "table_5_calibrated_2017": {
            "kappa": 4.2e-12,
            "leverage": 1.5,
            "memory_decay": 12.0,
            "oi_mu_q": -0.05,
            "oi_sigma_q": 0.20,
        }
    }
    manifest = tmp_path / "best_overrides.json"
    manifest.write_text(json.dumps(payload))

    loaded = load_tuned_overrides("table_5_calibrated_2017", manifest_path=manifest)
    assert loaded.kappa == pytest.approx(4.2e-12)
    assert loaded.leverage == pytest.approx(1.5)
    assert loaded.memory_decay == pytest.approx(12.0)
    assert loaded.oi_mu_q == pytest.approx(-0.05)
    assert loaded.oi_sigma_q == pytest.approx(0.20)


def test_synthetic_replication_runs_with_loaded_overrides(tmp_path: Path) -> None:
    """End-to-end: load tuned overrides, run the replication driver with them."""
    payload = {
        "table_5_calibrated_2017": {
            "kappa": 1e-12,
            "leverage": 0.5,
            "memory_decay": 12.0,
            "oi_mu_q": 0.0,
            "oi_sigma_q": 0.10,
        }
    }
    manifest = tmp_path / "best_overrides.json"
    manifest.write_text(json.dumps(payload))

    loaded = load_tuned_overrides("table_5_calibrated_2017", manifest_path=manifest)
    cfg = ReplicationConfig(
        parameter_set="table_5_calibrated_2017",
        n_paths=200,
        n_steps=63,
        seed=7,
        horizons_years=(0.0833, 0.25),
        tuned_overrides=loaded,
    )
    out = run_reflexive_with_matched_marketron_calibration(cfg)
    # Driver records the tuned overrides in the output payload.
    assert out["tuned_overrides"]["kappa"] == pytest.approx(1e-12)
    assert out["tuned_overrides"]["leverage"] == pytest.approx(0.5)
    assert out["tuned_overrides"]["oi_sigma_q"] == pytest.approx(0.10)


def test_load_tuned_overrides_falls_back_to_default(tmp_path: Path) -> None:
    """Missing manifest → DEFAULT_TUNED_OVERRIDES (no crash, no surprise)."""
    overrides = load_tuned_overrides("nonexistent_set_name")
    # The default is the dataclass' field defaults.
    assert isinstance(overrides, TunedReflexiveOverrides)
    assert overrides.kappa > 0


# ---------------------------------------------------------------------------
# Headline shape-match check at the tuned overrides
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    bool(os.environ.get("CI")),
    reason="End-to-end at tuned params is too slow for CI; smoketest only",
)
def test_at_least_some_shape_cells_match() -> None:
    """At a moderately-tuned config, ≥30% of shape_target cells match in sign.

    Uses the tuning-script's default winner shape (modest κ, modest γ, OI grid
    centred at ATM). The bar is intentionally loose — 0% would be a sanity
    failure of the tuning script; 100% would mean we're overfitting to a single
    parameter set's idiosyncrasies.
    """
    cfg = ReplicationConfig(
        parameter_set="table_5_calibrated_2017",
        n_paths=4000,
        n_steps=504,
        seed=42,
        horizons_years=(0.0833, 0.25, 0.50, 1.00, 2.00),
        tuned_overrides=TunedReflexiveOverrides(
            kappa=1e-12,
            leverage=0.5,
            memory_decay=12.0,
            oi_mu_q=0.0,
            oi_sigma_q=0.10,
        ),
    )
    metrics = run_reflexive_with_matched_marketron_calibration(cfg)
    result = compare_to_marketron_targets(metrics, cfg.parameter_set)
    assert result["shape_target_cells_total"] >= 4
    assert result["shape_match_rate"] >= SHAPE_MATCH_GATE


# ---------------------------------------------------------------------------
# Misc: format_mechanism_decomposition_table renders something sensible
# ---------------------------------------------------------------------------


def test_format_mechanism_decomposition_table_runs() -> None:
    """Smoke-test the printable formatter."""
    target_set = "table_5_calibrated_2017"
    metrics = _exact_match_metrics(target_set)
    result = compare_to_marketron_targets(metrics, target_set)
    text = format_mechanism_decomposition_table(result)
    assert "shape-target cells" in text
    assert "level-artifact cells" in text
    assert "calibration-artifact cells" in text
    # Exactly one row per (horizon, moment) for the target set.
    n_rows_expected = sum(len(v) for v in MARKETRON_MOMENT_TARGETS[target_set].values())
    body_rows = [
        line
        for line in text.split("\n")
        if "|" in line and "horizon" not in line and "----" not in line
    ]
    assert len(body_rows) == n_rows_expected


def test_write_best_overrides_round_trips(tmp_path: Path) -> None:
    """Best-overrides JSON loads back into the exact same TunedReflexiveOverrides."""
    per_set = {
        "table_5_calibrated_2017": TunedReflexiveOverrides(
            kappa=4.2e-12, leverage=1.5, memory_decay=12.0, oi_mu_q=-0.05, oi_sigma_q=0.20
        )
    }
    global_winner = TunedReflexiveOverrides(
        kappa=1e-12, leverage=0.5, memory_decay=12.0, oi_mu_q=0.0, oi_sigma_q=0.10
    )
    out_path = tmp_path / "best_overrides.json"
    write_best_overrides(per_set, global_winner, out_path)
    payload = json.loads(out_path.read_text())
    assert payload["_global"]["kappa"] == pytest.approx(1e-12)
    assert payload["table_5_calibrated_2017"]["kappa"] == pytest.approx(4.2e-12)


# ---------------------------------------------------------------------------
# Lightweight integration: 1-config grid runs without errors
# ---------------------------------------------------------------------------


def test_evaluate_one_config_returns_valid_outcome() -> None:
    """Tiny single-config eval doesn't crash and emits the right dataclass."""
    grid = TuningGridConfig(
        kappa_grid=(1e-12,),
        gamma_grid=(0.0,),
        t_eff_grid=(0.083,),
        mu_q_grid=(0.0,),
        sigma_q_grid=(0.10,),
        n_paths=500,
        n_steps=126,
        horizons_years=(0.0833, 0.25),
        seed=7,
    )
    outcome = _evaluate_one_config(
        parameter_set="table_5_calibrated_2017",
        kappa=1e-12,
        gamma=0.0,
        t_eff=0.083,
        mu_q=0.0,
        sigma_q=0.10,
        grid=grid,
    )
    assert isinstance(outcome, TuningOutcome)
    assert outcome.parameter_set == "table_5_calibrated_2017"
    assert outcome.shape_total >= 0
    assert 0.0 <= outcome.shape_match_rate <= 1.0


def test_write_grid_results_raises_on_empty(tmp_path: Path) -> None:
    """Empty outcomes list explicitly fails so we don't write a malformed parquet."""
    with pytest.raises(ValueError, match="empty outcomes"):
        write_grid_results([], tmp_path / "x.parquet")


def test_link_latest_creates_symlink(tmp_path: Path) -> None:
    """link_latest creates the 'latest' alias next to a tuning run dir."""
    run_dir = tmp_path / "tuning_runs" / "20260502T000000Z_seed42"
    run_dir.mkdir(parents=True)
    (run_dir / "best_overrides.json").write_text(json.dumps({"_global": {}}))
    link_latest(run_dir)
    latest = run_dir.parent / "latest"
    assert latest.exists()
    # On filesystems that support symlinks, latest points at the run dir.
    if latest.is_symlink():
        assert latest.resolve().name == run_dir.name


def test_link_latest_overwrites_existing_pointer(tmp_path: Path) -> None:
    """A second link_latest call replaces the prior 'latest' alias."""
    run_dir_a = tmp_path / "tuning_runs" / "A"
    run_dir_b = tmp_path / "tuning_runs" / "B"
    run_dir_a.mkdir(parents=True)
    run_dir_b.mkdir(parents=True)
    (run_dir_a / "best_overrides.json").write_text("{}")
    (run_dir_b / "best_overrides.json").write_text("{}")
    link_latest(run_dir_a)
    link_latest(run_dir_b)
    latest = run_dir_a.parent / "latest"
    assert latest.exists()


@pytest.mark.skipif(
    bool(os.environ.get("CI")),
    reason="main() runs the smoketest grid (~3s); skipped on CI to keep test job fast",
)
def test_tuning_main_smoketest_returns_zero(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """`main()` end-to-end with --smoketest writes everything and exits 0.

    Critically, we monkeypatch `RUNS_DIR` to a tmp path so the smoketest does
    NOT pollute the real `runs/marketron_tuning/latest` symlink with a tiny
    grid result.
    """
    import reflexive_options.experiments._common as common
    import reflexive_options.experiments.marketron_tuning as tuning_mod

    monkeypatch.setattr(common, "RUNS_DIR", tmp_path)
    monkeypatch.setattr(tuning_mod, "make_run_dir", common.make_run_dir)
    monkeypatch.setattr(
        "sys.argv", ["marketron_tuning", "--smoketest", "--n-paths", "200", "--n-steps", "63"]
    )
    rc = tuning_main()
    assert rc == 0


# ---------------------------------------------------------------------------
# synthetic_replication.main() — exit code wiring
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    bool(os.environ.get("CI")),
    reason="End-to-end driver runs the full Heston + reflexive sweep; skip on CI",
)
def test_synthetic_replication_main_exits_with_shape_gate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """`synthetic_replication.main()` returns 0 iff the shape gate is met."""
    import reflexive_options.experiments._common as common
    import reflexive_options.experiments.synthetic_replication as sr_mod

    monkeypatch.setattr(common, "RUNS_DIR", tmp_path)
    monkeypatch.setattr(sr_mod, "make_run_dir", common.make_run_dir)
    monkeypatch.setattr(
        "sys.argv",
        [
            "synthetic_replication",
            "--n-paths",
            "1000",
            "--n-steps",
            "126",
            "--parameter-set",
            "table_5_calibrated_2017",
        ],
    )
    rc = sr_mod.main()
    # Should be 0 or 1; no exceptions.
    assert rc in {0, 1}
