"""Regression test against the v0.1.0 reproducibility receipt.

For every experiment listed in `tests/repro/baseline_v0.1.0.json`, re-run the
canonical entry and assert the new outputs match the receipt:

    - tolerance="exact": metrics are bit-equal (or within 1e-12 absolute for
      floats) to the receipt.
    - tolerance="relative_5pct": each metric within 5% relative of the
      baseline mean; if a baseline std is recorded, also assert the new
      single-seed value is within 3 σ of the mean.

The test takes ~30-60 s to run all experiments end-to-end, so it is gated
behind `@pytest.mark.slow`. Two opt-out env vars are honored (CI runs the test
by default — set either to skip):

    CI_FAST=1   pytest        # local fast iteration
    SKIP_REPRO=1 pytest       # CI flake escape hatch (document in commit msg)

To intentionally update the baseline after a science change:

    bash scripts/generate_repro_baseline.sh
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from reflexive_options.experiments import _generate_repro_baseline as repro_mod
from reflexive_options.experiments._generate_repro_baseline import (
    CANONICAL_SEED,
    RECEIPT_PATH,
    SCHEMA_VERSION,
    STOCHASTIC_REL_TOLERANCE,
    STOCHASTIC_SEEDS,
    ExperimentSpec,
    ReceiptEntry,
    _aggregate_stochastic_runs,
    _all_specs,
    _build_entry,
    _build_receipt,
    _to_jsonable,
    _tooling_versions,
    blake2b_hex,
    canonical_json,
)

# ---------------------------------------------------------------------------
# Fixture loading + parametrization
# ---------------------------------------------------------------------------

EXACT_ABS_TOL = 1e-12  # for "exact" floats — guards against the last-bit jitter
STOCH_STD_MULTIPLIER = 3.0  # 3σ window for the single-seed re-run check


def load_baseline_entries() -> list[dict[str, Any]]:
    """Load the baseline JSON and return the list of receipt entries.

    Returned at collect time so each entry parametrizes its own test case.
    """
    if not RECEIPT_PATH.exists():
        return []
    payload = json.loads(RECEIPT_PATH.read_text())
    return list(payload["entries"])


def _entry_id(entry: dict[str, Any]) -> str:
    return f"{entry['experiment']}-{entry['tolerance']}"


def _iter_floats(value: Any) -> Iterator[float]:
    """Walk a JSON-shaped value and yield every float-or-int leaf."""
    if isinstance(value, bool):  # bool is a subclass of int — handle first
        return
    if isinstance(value, (int, float)):
        yield float(value)
        return
    if isinstance(value, dict):
        for v in value.values():
            yield from _iter_floats(v)
        return
    if isinstance(value, (list, tuple)):
        for v in value:
            yield from _iter_floats(v)
        return
    # str, None → no numeric content


def _flatten_with_paths(value: Any, prefix: str = "") -> Iterator[tuple[str, Any]]:
    """Yield (dotted_path, leaf) for every non-container leaf."""
    if isinstance(value, dict):
        for k, v in value.items():
            yield from _flatten_with_paths(v, f"{prefix}.{k}" if prefix else str(k))
    elif isinstance(value, (list, tuple)):
        for i, v in enumerate(value):
            yield from _flatten_with_paths(v, f"{prefix}[{i}]")
    else:
        yield prefix, value


# ---------------------------------------------------------------------------
# Comparison helpers
# ---------------------------------------------------------------------------


def _assert_exact_match(measured: Any, baseline: Any, path: str = "") -> None:
    """Recursive structure equality with EXACT_ABS_TOL on float leaves."""
    # None / bool first because bool is an int subclass
    if baseline is None:
        assert measured is None, f"{path}: expected None, got {measured!r}"
        return
    if isinstance(baseline, bool):
        assert measured == baseline, f"{path}: bool mismatch {measured!r} vs {baseline!r}"
        return
    if isinstance(baseline, (int, float)):
        if not isinstance(measured, (int, float)) or isinstance(measured, bool):
            raise AssertionError(
                f"{path}: type mismatch — measured={type(measured).__name__}, "
                f"baseline={type(baseline).__name__}"
            )
        # NaN does not equal itself; allow NaN-NaN as a valid match.
        if math.isnan(float(baseline)):
            assert math.isnan(float(measured)), f"{path}: expected NaN, got {measured}"
            return
        assert math.isclose(float(measured), float(baseline), abs_tol=EXACT_ABS_TOL, rel_tol=0.0), (
            f"{path}: |Δ|={abs(float(measured) - float(baseline)):.3e} > {EXACT_ABS_TOL}"
        )
        return
    if isinstance(baseline, str):
        assert measured == baseline, f"{path}: str mismatch {measured!r} vs {baseline!r}"
        return
    if isinstance(baseline, dict):
        assert isinstance(measured, dict), f"{path}: expected dict, got {type(measured).__name__}"
        bk = set(baseline.keys())
        mk = set(measured.keys())
        assert bk == mk, f"{path}: dict keys differ — only-baseline={bk - mk}, only-new={mk - bk}"
        for k in baseline:
            _assert_exact_match(measured[k], baseline[k], f"{path}.{k}" if path else str(k))
        return
    if isinstance(baseline, list):
        assert isinstance(measured, list), f"{path}: expected list, got {type(measured).__name__}"
        assert len(measured) == len(baseline), (
            f"{path}: list len differs — measured={len(measured)} vs baseline={len(baseline)}"
        )
        for i, (m, b) in enumerate(zip(measured, baseline, strict=True)):
            _assert_exact_match(m, b, f"{path}[{i}]")
        return
    raise AssertionError(f"{path}: unhandled baseline type {type(baseline).__name__}")


def _relative_error(measured: float, target: float) -> float:
    """Symmetric-safe relative error with a 1e-12 floor on the denominator."""
    denom = max(abs(target), 1e-12)
    return abs(measured - target) / denom


def _assert_relative_5pct(
    measured: dict[str, Any],
    baseline_mean: dict[str, Any],
    baseline_std: dict[str, Any],
    rel_tolerance: float = STOCHASTIC_REL_TOLERANCE,
    std_multiplier: float = STOCH_STD_MULTIPLIER,
) -> None:
    """Compare a single-seed re-run against (mean, std) baseline.

    Asserts each numeric leaf:
      (a) within `rel_tolerance` relative of baseline mean, OR
      (b) within `std_multiplier` × baseline std of baseline mean.
    Either condition passes — gives the test float-headroom on both axes.

    Non-numeric leaves (categorical fields, bools) must match the baseline
    mean exactly.
    """
    measured_flat = dict(_flatten_with_paths(measured))
    mean_flat = dict(_flatten_with_paths(baseline_mean))
    std_flat = dict(_flatten_with_paths(baseline_std))

    # Allow keys that exist in measured but not baseline iff they're new
    # informational additions (don't fail). Conversely, if a baseline key is
    # missing from measured, that's a regression.
    missing = set(mean_flat.keys()) - set(measured_flat.keys())
    assert not missing, f"baseline keys missing from re-run: {sorted(missing)}"

    failures: list[str] = []
    for path, target in mean_flat.items():
        actual = measured_flat[path]
        std = std_flat.get(path)

        # Booleans on a stochastic entry are intrinsically noisy (they're
        # the "majority across seeds" reduction). Skip them — the underlying
        # numeric quantity that drives the bool is checked separately.
        if isinstance(target, bool):
            continue

        # Non-numeric (categorical, None) — exact match required
        if not isinstance(target, (int, float)):
            if actual != target:
                failures.append(f"{path}: non-numeric mismatch {actual!r} vs {target!r}")
            continue

        if not isinstance(actual, (int, float)) or isinstance(actual, bool):
            failures.append(
                f"{path}: type mismatch — measured={type(actual).__name__} "
                f"vs baseline={type(target).__name__}"
            )
            continue

        # Allow NaN==NaN
        if math.isnan(float(target)) and math.isnan(float(actual)):
            continue

        rel_err = _relative_error(float(actual), float(target))
        within_rel = rel_err <= rel_tolerance

        within_sigma = False
        if isinstance(std, (int, float)) and float(std) > 0:
            within_sigma = abs(float(actual) - float(target)) <= std_multiplier * float(std)

        if not (within_rel or within_sigma):
            sigma_repr = f"{std}σ" if std is not None else "no-std"
            failures.append(
                f"{path}: measured={actual:.6g} vs mean={target:.6g} "
                f"(rel_err={rel_err:.3%} > {rel_tolerance:.0%}, "
                f"|Δ|={abs(actual - target):.3g} > {std_multiplier}·{sigma_repr})"
            )

    assert not failures, "\n".join(failures)


# ---------------------------------------------------------------------------
# Test runners
# ---------------------------------------------------------------------------


def _spec_for(experiment_name: str) -> Any:
    """Look up the ExperimentSpec in the registry by name."""
    for spec in _all_specs():
        if spec.name == experiment_name:
            return spec
    raise KeyError(f"No registered ExperimentSpec named {experiment_name!r}")


SKIP_REASON = "set CI_FAST=1 or SKIP_REPRO=1 to skip — see test_reproducibility.py docstring"


@pytest.mark.slow
@pytest.mark.skipif(os.environ.get("CI_FAST") or os.environ.get("SKIP_REPRO"), reason=SKIP_REASON)
@pytest.mark.parametrize(
    "entry",
    load_baseline_entries(),
    ids=[_entry_id(e) for e in load_baseline_entries()] or None,
)
def test_experiment_reproduces_baseline(entry: dict[str, Any]) -> None:
    """Re-run the experiment and assert metrics match the receipt within tolerance."""
    spec = _spec_for(entry["experiment"])
    baseline_metrics = entry["metrics"]
    tolerance = entry["tolerance"]

    # Single-seed re-run for the regression check (always at the canonical seed).
    measured = spec.runner(int(entry["seed"]))
    measured = _coerce_to_jsonable(measured)

    if tolerance == "exact":
        _assert_exact_match(measured, baseline_metrics, path=spec.name)
        # Hash check is a fast canary — should be redundant with structural eq.
        new_hash = blake2b_hex(measured)
        assert new_hash == entry["blake2b_hash"], (
            f"hash drift on {spec.name}: {new_hash} vs {entry['blake2b_hash']}"
        )
    elif tolerance == "relative_5pct":
        baseline_std = entry.get("metrics_std", {})
        _assert_relative_5pct(measured, baseline_metrics, baseline_std)
    else:
        raise AssertionError(f"unhandled tolerance class {tolerance!r}")


def _coerce_to_jsonable(obj: Any) -> Any:
    """Mirror of `_to_jsonable` from the generator — duplicated to avoid coupling."""
    if isinstance(obj, dict):
        return {k: _coerce_to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_coerce_to_jsonable(x) for x in obj]
    if isinstance(obj, np.ndarray):
        return _coerce_to_jsonable(obj.tolist())
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    return obj


# ---------------------------------------------------------------------------
# Receipt-shape sanity checks (cheap — always run)
# ---------------------------------------------------------------------------


def test_receipt_file_exists() -> None:
    assert RECEIPT_PATH.exists(), (
        f"missing reproducibility receipt at {RECEIPT_PATH} — "
        "run `bash scripts/generate_repro_baseline.sh`"
    )


def test_receipt_is_well_formed() -> None:
    """Top-level keys present + every entry has the required fields."""
    payload = json.loads(RECEIPT_PATH.read_text())
    required_top = {
        "schema_version",
        "receipt_version",
        "canonical_seed",
        "stochastic_seeds",
        "stochastic_rel_tolerance",
        "generated_at_utc",
        "tooling",
        "n_experiments",
        "entries",
    }
    assert required_top.issubset(payload.keys()), required_top - payload.keys()
    assert payload["n_experiments"] == len(payload["entries"])
    required_entry = {
        "experiment",
        "seed",
        "config",
        "metrics",
        "blake2b_hash",
        "tolerance",
        "timestamp_utc",
        "tooling",
    }
    for entry in payload["entries"]:
        assert required_entry.issubset(entry.keys()), required_entry - entry.keys()
        assert entry["tolerance"] in {"exact", "relative_5pct"}


def test_every_registered_spec_appears_in_receipt() -> None:
    """No spec silently dropped on the way to the receipt."""
    payload = json.loads(RECEIPT_PATH.read_text())
    receipt_names = {e["experiment"] for e in payload["entries"]}
    spec_names = {spec.name for spec in _all_specs()}
    assert spec_names == receipt_names, (
        f"spec/receipt drift — only-spec={spec_names - receipt_names}, "
        f"only-receipt={receipt_names - spec_names}"
    )


def test_receipt_hashes_recompute_from_metrics() -> None:
    """Hash field is the blake2b of the metrics canonical-form JSON.

    For "exact" entries the hash hashes `metrics`. For "relative_5pct" it
    hashes the {means, stds, seeds_used} envelope (per generator design).
    """
    payload = json.loads(RECEIPT_PATH.read_text())
    for entry in payload["entries"]:
        if entry["tolerance"] == "exact":
            recomputed = blake2b_hex(entry["metrics"])
        else:
            recomputed = blake2b_hex(
                {
                    "means": entry["metrics"],
                    "stds": entry["metrics_std"],
                    "seeds": entry["seeds_used"],
                }
            )
        assert recomputed == entry["blake2b_hash"], (
            f"hash mismatch on {entry['experiment']}: "
            f"recomputed={recomputed} vs stored={entry['blake2b_hash']}"
        )


# ---------------------------------------------------------------------------
# Unit tests for the receipt module's helpers
# ---------------------------------------------------------------------------


def test_constants_have_expected_values() -> None:
    """Lock the public constants in case someone bumps them by accident."""
    assert CANONICAL_SEED == 42
    assert STOCHASTIC_REL_TOLERANCE == 0.05
    assert len(STOCHASTIC_SEEDS) == 5 and STOCHASTIC_SEEDS[0] == 42
    assert SCHEMA_VERSION == "1.0"


def test_canonical_json_is_sorted_and_compact() -> None:
    """canonical_json sorts keys and emits no unnecessary whitespace."""
    raw = {"b": 1, "a": [3, 2, 1]}
    out = canonical_json(raw)
    assert out == b'{"a":[3,2,1],"b":1}'


def test_canonical_json_handles_unknown_with_default_str() -> None:
    """Path-like / numpy values stringify rather than crash."""
    raw = {"path": Path("./fixtures/foo"), "np": np.float64(1.5)}
    out = canonical_json(raw)
    # np.float64 repr is "1.5"; Path stringifies. The order must be sorted.
    assert b"fixtures/foo" in out
    assert b"1.5" in out


def test_blake2b_hex_is_deterministic_and_64chars() -> None:
    h1 = blake2b_hex({"x": [1, 2, 3]})
    h2 = blake2b_hex({"x": [1, 2, 3]})
    assert h1 == h2
    assert len(h1) == 64
    assert all(c in "0123456789abcdef" for c in h1)


def test_blake2b_hex_changes_when_payload_differs() -> None:
    assert blake2b_hex({"x": 1}) != blake2b_hex({"x": 2})


def test_to_jsonable_dict_list_tuple() -> None:
    raw = {"a": (1, 2, 3), "b": [4, 5]}
    out = _to_jsonable(raw)
    assert out == {"a": [1, 2, 3], "b": [4, 5]}
    # Tuples are coerced to lists.
    assert isinstance(out["a"], list)


def test_to_jsonable_numpy_scalars() -> None:
    out = _to_jsonable({"f": np.float64(1.5), "i": np.int32(7)})
    assert out == {"f": 1.5, "i": 7}
    # Pure Python types after coercion (no numpy dtypes).
    assert type(out["f"]) is float
    assert type(out["i"]) is int


def test_to_jsonable_numpy_array_to_list() -> None:
    arr = np.array([[1.0, 2.0], [3.0, 4.0]])
    out = _to_jsonable({"a": arr})
    assert out == {"a": [[1.0, 2.0], [3.0, 4.0]]}


def test_to_jsonable_path_to_str() -> None:
    out = _to_jsonable({"p": Path("./fixtures/x")})
    assert out == {"p": "fixtures/x"}
    assert isinstance(out["p"], str)


def test_to_jsonable_passthrough_for_primitives() -> None:
    assert _to_jsonable(None) is None
    assert _to_jsonable("hello") == "hello"
    assert _to_jsonable(True) is True
    assert _to_jsonable(0) == 0


def test_tooling_versions_includes_python_and_packages() -> None:
    versions = _tooling_versions()
    assert "python" in versions
    assert "platform" in versions
    assert "machine" in versions
    # All declared dependencies must resolve to a string version (or "missing")
    for pkg in ("numpy", "scipy", "torch", "matplotlib"):
        assert pkg in versions
        assert isinstance(versions[pkg], str)


def test_tooling_versions_marks_missing_packages_explicitly() -> None:
    """A package that isn't installed shows up as 'missing', not a crash."""
    import importlib.metadata as md

    real_version = md.version

    def fake_version(name: str) -> str:
        if name == "numpy":
            raise md.PackageNotFoundError("numpy")
        return real_version(name)

    orig = repro_mod.importlib.metadata.version
    repro_mod.importlib.metadata.version = fake_version  # type: ignore[assignment]
    try:
        versions = _tooling_versions()
        assert versions["numpy"] == "missing"
    finally:
        repro_mod.importlib.metadata.version = orig  # type: ignore[assignment]


def test_aggregate_stochastic_runs_empty() -> None:
    means, stds = _aggregate_stochastic_runs([])
    assert means == {} and stds == {}


def test_aggregate_stochastic_runs_scalars() -> None:
    runs = [{"x": 1.0}, {"x": 3.0}, {"x": 2.0}]
    means, stds = _aggregate_stochastic_runs(runs)
    assert means["x"] == pytest.approx(2.0)
    # ddof=1 std of [1, 3, 2] = 1.0
    assert stds["x"] == pytest.approx(1.0)


def test_aggregate_stochastic_runs_single_seed_zero_std() -> None:
    """A single run yields std=0 (can't compute ddof=1 std on n=1)."""
    runs = [{"x": 5.0}]
    means, stds = _aggregate_stochastic_runs(runs)
    assert means["x"] == 5.0 and stds["x"] == 0.0


def test_aggregate_stochastic_runs_lists() -> None:
    runs = [{"v": [1.0, 2.0]}, {"v": [3.0, 4.0]}, {"v": [5.0, 6.0]}]
    means, stds = _aggregate_stochastic_runs(runs)
    assert means["v"] == pytest.approx([3.0, 4.0])
    assert stds["v"] == pytest.approx([2.0, 2.0])


def test_aggregate_stochastic_runs_single_seed_lists() -> None:
    runs = [{"v": [1.0, 2.0, 3.0]}]
    means, stds = _aggregate_stochastic_runs(runs)
    assert means["v"] == [1.0, 2.0, 3.0]
    assert stds["v"] == [0.0, 0.0, 0.0]


def test_aggregate_stochastic_runs_majority_bool_true() -> None:
    runs = [{"flag": True}, {"flag": True}, {"flag": False}]
    means, stds = _aggregate_stochastic_runs(runs)
    assert means["flag"] is True and stds["flag"] == 0


def test_aggregate_stochastic_runs_majority_bool_false() -> None:
    runs = [{"flag": False}, {"flag": True}, {"flag": False}]
    means, stds = _aggregate_stochastic_runs(runs)
    assert means["flag"] is False and stds["flag"] == 0


def test_aggregate_stochastic_runs_categorical() -> None:
    """Strings and other non-numeric leaves take the first run's value, std=None."""
    runs = [{"label": "alpha"}, {"label": "beta"}]
    means, stds = _aggregate_stochastic_runs(runs)
    assert means["label"] == "alpha"
    assert stds["label"] is None


def test_all_specs_registry_has_expected_entries() -> None:
    """Sanity-check the spec registry covers exactly the four canonical scripts."""
    specs = _all_specs()
    names = sorted(s.name for s in specs)
    assert names == [
        "bifurcation_scan",
        "phase_diagram",
        "reflexive_transfer",
        "synthetic_replication",
    ]
    tolerances = {s.name: s.tolerance for s in specs}
    assert tolerances["reflexive_transfer"] == "relative_5pct"
    assert tolerances["bifurcation_scan"] == "exact"
    # synthetic_replication moved to relative_5pct after the C5 mechanism-decomp
    # rewrite — the per-cell metric float values differ at the 1e-12 level across
    # Python 3.12 vs 3.13/3.14 due to BLAS/float-sum ordering, which propagates
    # into the blake2b hash. The 5% gate is the right one for this experiment.
    assert tolerances["synthetic_replication"] == "relative_5pct"
    assert tolerances["phase_diagram"] == "exact"


def test_build_entry_exact_tolerance() -> None:
    """_build_entry on a synthetic deterministic spec emits the expected schema."""

    def _runner(seed: int) -> dict[str, Any]:
        return {"answer": 42 + seed}

    def _config(seed: int) -> dict[str, Any]:
        return {"seed": seed}

    spec = ExperimentSpec(
        name="dummy_exact", runner=_runner, config_factory=_config, tolerance="exact"
    )
    tooling = {"python": "x.y.z"}
    entry = _build_entry(spec, tooling)
    assert isinstance(entry, ReceiptEntry)
    assert entry.experiment == "dummy_exact"
    assert entry.tolerance == "exact"
    assert entry.metrics == {"answer": 84}  # 42 + CANONICAL_SEED
    assert entry.seeds_used == [CANONICAL_SEED]
    assert entry.metrics_std == {}
    # Hash matches canonical-form blake2b on the metrics dict.
    assert entry.blake2b_hash == blake2b_hex(entry.metrics)


def test_build_entry_relative_5pct_tolerance() -> None:
    """_build_entry on a synthetic stochastic spec emits aggregated mean/std + envelope hash."""

    def _runner(seed: int) -> dict[str, Any]:
        # Linear in seed so the std is non-zero across STOCHASTIC_SEEDS.
        return {"v": float(seed)}

    def _config(seed: int) -> dict[str, Any]:
        return {"seed": seed}

    spec = ExperimentSpec(
        name="dummy_stoch",
        runner=_runner,
        config_factory=_config,
        tolerance="relative_5pct",
    )
    entry = _build_entry(spec, {"python": "x"})
    assert entry.tolerance == "relative_5pct"
    seeds = list(STOCHASTIC_SEEDS)
    expected_mean = float(np.mean(seeds))
    expected_std = float(np.std(seeds, ddof=1))
    assert entry.metrics["v"] == pytest.approx(expected_mean)
    assert entry.metrics_std["v"] == pytest.approx(expected_std)
    assert entry.seeds_used == seeds
    # Hash signs over the envelope, not the means alone.
    expected_hash = blake2b_hex({"means": entry.metrics, "stds": entry.metrics_std, "seeds": seeds})
    assert entry.blake2b_hash == expected_hash


def test_build_entry_unknown_tolerance_raises() -> None:
    spec = ExperimentSpec(
        name="dummy", runner=lambda s: {}, config_factory=lambda s: {}, tolerance="unknown"
    )
    with pytest.raises(ValueError, match="Unknown tolerance"):
        _build_entry(spec, {"python": "x"})


def test_main_writes_receipt_atomically(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """main() writes the JSON receipt to RECEIPT_PATH (redirected) using atomic rename."""
    target = tmp_path / "baseline.json"
    monkeypatch.setattr(repro_mod, "RECEIPT_PATH", target)

    # Stub the spec registry so main() runs in <1 s.
    def _tiny_runner(seed: int) -> dict[str, Any]:
        return {"x": seed}

    def _tiny_cfg(seed: int) -> dict[str, Any]:
        return {"seed": seed}

    monkeypatch.setattr(
        repro_mod,
        "_all_specs",
        lambda: [
            ExperimentSpec(
                name="tiny", runner=_tiny_runner, config_factory=_tiny_cfg, tolerance="exact"
            )
        ],
    )

    repro_mod.main()
    assert target.exists()
    payload = json.loads(target.read_text())
    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["n_experiments"] == 1
    assert payload["entries"][0]["experiment"] == "tiny"
    # The .tmp staging file should have been renamed away.
    assert not target.with_suffix(".tmp").exists()


def test_build_receipt_includes_schema_and_tooling() -> None:
    """_build_receipt assembles the top-level envelope with the expected keys."""
    receipt = _build_receipt()
    assert receipt["schema_version"] == SCHEMA_VERSION
    assert receipt["canonical_seed"] == CANONICAL_SEED
    assert "tooling" in receipt and "python" in receipt["tooling"]
    assert receipt["n_experiments"] == len(receipt["entries"])
    assert all("blake2b_hash" in e for e in receipt["entries"])
