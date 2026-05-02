"""Performance benchmark for the vectorised dealer-gamma aggregator.

Design target: a single tensor-broadcast call to `compute_batch` should be ≥10×
faster than the per-path Python loop on a typical MC simulation. We assert ≥5×
in the test (50% safety margin so it remains stable across hardware) and print
the measured speedup so regressions are visible in the test output.

Skipped on CI runners where wall-clock variance is too high to be reliable;
intended to run locally before merging perf-sensitive changes.
"""

from __future__ import annotations

import os
import time

import numpy as np
import pytest

from reflexive_options.experiments.synthetic_replication import _default_oi_grid
from reflexive_options.simulator.gamma_aggregator import (
    GammaAggregator,
    GammaAggregatorConfig,
)
from reflexive_options.simulator.reflexive import ReflexiveSimulator
from reflexive_options.types import HestonParams, ReflexiveParams


def _build_simulator(initial_spot: float = 100.0) -> ReflexiveSimulator:
    oi_grid = _default_oi_grid(initial_spot)
    aggregator = GammaAggregator(
        oi_grid=oi_grid,
        risk_free_rate=0.01,
        config=GammaAggregatorConfig(),
    )
    base = HestonParams(kappa=2.0, theta=0.04, xi=0.3, rho=-0.7, v0=0.04)
    params = ReflexiveParams(
        base=base,
        coupling=1e-12,
        drift=0.0,
        memory_decay=1.0,
        memory_intake=1.0,
        leverage=1e-3,
    )
    return ReflexiveSimulator(
        params=params,
        gamma_aggregator=aggregator,
        initial_spot=initial_spot,
        antithetic=True,
    )


@pytest.mark.benchmark
@pytest.mark.skipif(
    bool(os.environ.get("CI")),
    reason="Skip benchmark on CI runners — wall-clock variance too high",
)
def test_g_vectorized_speedup_geq_10x() -> None:
    """Compare per-path loop vs vectorised batch over a representative state grid.

    Design target is 10×. Assert 5× to absorb hardware variance (~50% margin).
    """
    sim = _build_simulator()
    rng = np.random.default_rng(0)

    # Representative MC-step sized state; 5,000 paths × 50 calls is small enough
    # to keep the benchmark sub-second on the fast path while still exposing the
    # python-loop overhead on the slow path.
    n_paths, n_calls = 5_000, 50
    spots = rng.uniform(80.0, 120.0, size=n_paths)
    variances = rng.uniform(0.01, 0.10, size=n_paths)
    memories = rng.uniform(-0.1, 0.1, size=n_paths)

    # Warm caches so JIT-style first-call overhead doesn't dominate.
    sim._g_vectorized(spots, variances, memories)
    sim._g_per_path(spots[:32], variances[:32], memories[:32])

    t0 = time.perf_counter()
    for _ in range(n_calls):
        sim._g_per_path(spots, variances, memories)
    t_loop = time.perf_counter() - t0

    t0 = time.perf_counter()
    for _ in range(n_calls):
        sim._g_vectorized(spots, variances, memories)
    t_batch = time.perf_counter() - t0

    speedup = t_loop / max(t_batch, 1e-9)
    print(
        f"\n[benchmark] _g_per_path: {t_loop:.4f}s | "
        f"_g_vectorized: {t_batch:.4f}s | speedup: {speedup:.2f}× "
        f"(target ≥10×, gate ≥5×)"
    )
    assert t_batch < t_loop / 5.0, (
        f"Expected ≥5× speedup, got {speedup:.2f}× (loop={t_loop:.4f}s, batch={t_batch:.4f}s)"
    )
