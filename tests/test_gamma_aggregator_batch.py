"""Numerical equivalence between `GammaAggregator.compute` and `compute_batch`.

The vectorised batch path must agree to ≤1e-12 with the per-path scalar loop
across the plausible (S, v, z) state space; if it diverges, the SDE drift in
`ReflexiveSimulator._g_vectorized` is silently miscomputed.
"""

from __future__ import annotations

import numpy as np

from reflexive_options.experiments.synthetic_replication import _default_oi_grid
from reflexive_options.simulator.gamma_aggregator import (
    GammaAggregator,
    GammaAggregatorConfig,
    GammaSignConvention,
)


def _make_aggregator(
    *,
    fixed_iv: float | None = None,
    initial_spot: float = 100.0,
    sign: GammaSignConvention | None = None,
) -> GammaAggregator:
    return GammaAggregator(
        oi_grid=_default_oi_grid(initial_spot),
        risk_free_rate=0.05,
        dividend_yield=0.0,
        sign=sign,
        config=GammaAggregatorConfig(fixed_iv=fixed_iv),
    )


def _random_state(rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    spots = rng.uniform(60.0, 160.0, size=n)
    variances = rng.uniform(0.005, 0.20, size=n)  # σ ∈ ~7%–45%
    memories = rng.uniform(-0.3, 0.3, size=n)
    return spots, variances, memories


def test_compute_batch_matches_per_path_loop() -> None:
    """For 100 random (S, v, z) triples, batch and loop must agree to 1e-12."""
    agg = _make_aggregator()
    rng = np.random.default_rng(20260422)
    spots, variances, memories = _random_state(rng, 100)

    loop = np.array(
        [
            agg.compute(float(s), float(v), float(z))
            for s, v, z in zip(spots, variances, memories, strict=True)
        ],
        dtype=np.float64,
    )
    batch = agg.compute_batch(spots, variances, memories)

    assert batch.shape == (100,)
    assert np.allclose(loop, batch, atol=1e-12, rtol=0.0), (
        f"max abs diff = {np.max(np.abs(loop - batch))}"
    )


def test_compute_batch_matches_per_path_loop_fixed_iv() -> None:
    agg = _make_aggregator(fixed_iv=0.20)
    rng = np.random.default_rng(7)
    spots, variances, memories = _random_state(rng, 50)

    loop = np.array(
        [
            agg.compute(float(s), float(v), float(z))
            for s, v, z in zip(spots, variances, memories, strict=True)
        ],
        dtype=np.float64,
    )
    batch = agg.compute_batch(spots, variances, memories)
    assert np.allclose(loop, batch, atol=1e-12, rtol=0.0)


def test_compute_batch_matches_with_flipped_sign() -> None:
    agg = _make_aggregator(sign=GammaSignConvention(call_sign=-1.0, put_sign=1.0))
    rng = np.random.default_rng(11)
    spots, variances, memories = _random_state(rng, 25)
    loop = np.array(
        [
            agg.compute(float(s), float(v), float(z))
            for s, v, z in zip(spots, variances, memories, strict=True)
        ],
        dtype=np.float64,
    )
    batch = agg.compute_batch(spots, variances, memories)
    assert np.allclose(loop, batch, atol=1e-12, rtol=0.0)


def test_compute_batch_zero_variance_path_returns_zero() -> None:
    """A path with variance=0 (sigma→0) should return 0, matching the scalar branch."""
    # Force fixed_iv=0 to deterministically trigger the σ ≤ 0 branch.
    agg = _make_aggregator(fixed_iv=0.0)
    spots = np.array([100.0, 110.0], dtype=np.float64)
    variances = np.array([0.04, 0.04], dtype=np.float64)
    memories = np.zeros(2, dtype=np.float64)
    out = agg.compute_batch(spots, variances, memories)
    assert np.all(out == 0.0)


def test_compute_batch_shape_mismatch_raises() -> None:
    agg = _make_aggregator()
    spots = np.array([100.0, 110.0])
    variances = np.array([0.04])  # mismatched shape
    memories = np.zeros(2)
    try:
        agg.compute_batch(spots, variances, memories)
    except ValueError:
        return
    raise AssertionError("expected ValueError on mismatched input shapes")
