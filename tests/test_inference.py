"""Tests for `theory.inference` — the shared statistical primitives.

Three families:
  - Politis–Romano stationary block bootstrap (V4-B3).
  - BH-FDR (V4-B4).
  - TOST equivalence (V3-B2 / A7).
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from reflexive_options.theory.inference import (
    benjamini_hochberg,
    block_bootstrap_ci,
    stationary_block_bootstrap,
    tost_equivalence,
)

# ---------------------------------------------------------------------------
# Stationary block bootstrap
# ---------------------------------------------------------------------------


def test_stationary_block_bootstrap_returns_correct_shape() -> None:
    """Each row is a resample of the same length as the input."""
    rng = np.random.default_rng(0)
    samples = rng.standard_normal(50).astype(np.float64)
    out = stationary_block_bootstrap(samples, block_length_mean=5.0, n_resamples=100, rng=rng)
    assert out.shape == (100, 50)
    assert np.all(np.isfinite(out))


def test_stationary_block_bootstrap_block_length_one_is_iid() -> None:
    """With block_length_mean=1 (Geometric → iid), the resamples should look like iid bootstrap.

    Operational check: the empirical mean across many resamples concentrates
    on the sample mean with variance ≈ Var(samples) / n.
    """
    rng = np.random.default_rng(123)
    samples = rng.standard_normal(200).astype(np.float64)
    out = stationary_block_bootstrap(samples, block_length_mean=1.0, n_resamples=2000, rng=rng)
    sample_mean = float(samples.mean())
    boot_means = out.mean(axis=1)
    # Empirical SE of the bootstrap means should approximate sample std / sqrt(n).
    empirical_se = float(boot_means.std(ddof=1))
    expected_se = float(samples.std(ddof=1)) / np.sqrt(len(samples))
    assert abs(empirical_se - expected_se) / expected_se < 0.15
    # Mean of bootstrap means is unbiased for the sample mean.
    assert abs(boot_means.mean() - sample_mean) < 0.05 * abs(sample_mean) + 0.02


def test_stationary_block_bootstrap_widens_under_ar1_dependence() -> None:
    """On AR(1) data with rho=0.7, the block bootstrap CI should be wider than iid.

    Rationale: iid bootstrap underestimates the variance of the mean when the
    data has positive autocorrelation; the block bootstrap is designed to
    correct for it. With block_length_mean appropriately set (≈ 1/(1-rho)
    or thereabouts), the resampled means have wider spread.
    """
    rng = np.random.default_rng(42)
    n = 500
    rho = 0.7
    eps = rng.standard_normal(n)
    x = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        x[i] = rho * x[i - 1] + eps[i]

    iid_resamples = stationary_block_bootstrap(
        x, block_length_mean=1.0, n_resamples=1000, rng=np.random.default_rng(1)
    )
    block_resamples = stationary_block_bootstrap(
        x, block_length_mean=20.0, n_resamples=1000, rng=np.random.default_rng(2)
    )
    iid_se = iid_resamples.mean(axis=1).std(ddof=1)
    block_se = block_resamples.mean(axis=1).std(ddof=1)
    # Block SE should be meaningfully larger (target: ≥ 1.5x for rho=0.7 at L=20).
    assert block_se > 1.5 * iid_se, (
        f"block bootstrap SE ({block_se:.4f}) not larger than iid SE ({iid_se:.4f})"
    )


def test_stationary_block_bootstrap_validates_inputs() -> None:
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="block_length_mean"):
        stationary_block_bootstrap(np.array([1.0]), block_length_mean=0.0, n_resamples=10, rng=rng)
    with pytest.raises(ValueError, match="n_resamples"):
        stationary_block_bootstrap(np.array([1.0]), block_length_mean=1.0, n_resamples=0, rng=rng)
    with pytest.raises(ValueError, match="non-empty"):
        stationary_block_bootstrap(
            np.array([], dtype=np.float64), block_length_mean=1.0, n_resamples=5, rng=rng
        )
    with pytest.raises(ValueError, match="1D"):
        stationary_block_bootstrap(np.zeros((3, 3)), block_length_mean=1.0, n_resamples=5, rng=rng)


def test_block_bootstrap_ci_recovers_mean_with_coverage() -> None:
    """Coverage ≈ 95% under correctly-specified iid Gaussian null.

    Uses block_length_mean=1 (iid) + the sample mean as statistic; the
    bootstrap CI should contain the population mean ~95% of the time.
    """
    n_reps = 200
    n = 100
    true_mean = 0.0
    inside = 0
    for r in range(n_reps):
        rng = np.random.default_rng(r)
        x = rng.standard_normal(n).astype(np.float64)
        _, lo, hi = block_bootstrap_ci(
            x,
            statistic=lambda s: float(s.mean()),
            confidence=0.95,
            block_length_mean=1.0,
            n_resamples=400,
            rng=np.random.default_rng(r + 1000),
        )
        if lo <= true_mean <= hi:
            inside += 1
    coverage = inside / n_reps
    # ~95% with reasonable tolerance for finite n_reps.
    assert 0.88 <= coverage <= 1.0, f"coverage {coverage:.3f} not in expected range"


def test_block_bootstrap_ci_validates_confidence() -> None:
    with pytest.raises(ValueError, match="confidence"):
        block_bootstrap_ci(
            np.array([1.0, 2.0, 3.0]),
            statistic=lambda s: float(s.mean()),
            confidence=1.5,
        )


def test_block_bootstrap_ci_handles_degenerate_statistic() -> None:
    """When the statistic returns non-finite for all resamples, raise."""
    samples = np.array([1.0, 2.0, 3.0])
    with pytest.raises(RuntimeError, match="non-finite"):
        block_bootstrap_ci(
            samples,
            statistic=lambda _s: float("nan"),
            n_resamples=10,
            block_length_mean=1.0,
            rng=np.random.default_rng(0),
        )


# ---------------------------------------------------------------------------
# Benjamini–Hochberg FDR
# ---------------------------------------------------------------------------


def test_benjamini_hochberg_no_rejections_when_all_above_alpha() -> None:
    p = np.array([0.5, 0.6, 0.7, 0.8])
    rejected = benjamini_hochberg(p, alpha=0.05)
    assert not rejected.any()
    assert rejected.shape == p.shape


def test_benjamini_hochberg_textbook_example() -> None:
    """Standard textbook example: m=10 sorted p-values check k where p_(k) ≤ k·α/m.

    Use p = [0.001, 0.008, 0.039, 0.041, 0.042, 0.060, 0.074, 0.205, 0.212, 0.216] at α=0.05.
    Threshold sequence: 0.005, 0.010, 0.015, 0.020, 0.025, 0.030, 0.035, 0.040, 0.045, 0.050.
    Largest k satisfying p_(k) ≤ k * 0.05 / 10 = 0.005 k is k=5 (p_(5)=0.042 ≤ 0.025? No).
    Re-check: p_(1)=0.001 ≤ 0.005 ✓, p_(2)=0.008 ≤ 0.010 ✓, p_(3)=0.039 ≤ 0.015? No.
    So the largest k where p_(k) ≤ 0.005k is k=2 → reject hypotheses with p ≤ 0.008.
    """
    p = np.array([0.001, 0.008, 0.039, 0.041, 0.042, 0.060, 0.074, 0.205, 0.212, 0.216])
    rejected = benjamini_hochberg(p, alpha=0.05)
    # Reject hypotheses 0, 1 (p=0.001, 0.008); accept the rest.
    expected = np.array([True, True, False, False, False, False, False, False, False, False])
    np.testing.assert_array_equal(rejected, expected)


def test_benjamini_hochberg_controls_fdr_under_independent_nulls() -> None:
    """Under all-true-null with independent uniform p-values, P(any reject) ≤ alpha."""
    n_sim = 2000
    m = 20
    alpha = 0.05
    rng = np.random.default_rng(0)
    n_any = 0
    for _ in range(n_sim):
        p = rng.uniform(size=m)
        rejected = benjamini_hochberg(p, alpha=alpha)
        if rejected.any():
            n_any += 1
    fwer = n_any / n_sim
    # BH controls FDR ≤ alpha; under all-null, FDR = P(any false reject) so FWER ≤ alpha.
    # Stochastic tolerance: alpha ± 2 SE.
    assert fwer <= alpha + 0.02


def test_benjamini_hochberg_correct_rejection_count_on_known_mixture() -> None:
    """Mixture: 5 nulls (uniform) + 5 alternatives (small p). Expect ~5 rejections."""
    rng = np.random.default_rng(1)
    n_sim = 500
    n_alt_rej_total = 0
    n_null_rej_total = 0
    for _ in range(n_sim):
        p_null = rng.uniform(size=5)
        p_alt = np.minimum(rng.uniform(size=5) * 0.01, 0.005)  # small p for alternatives
        p = np.concatenate([p_null, p_alt])
        rejected = benjamini_hochberg(p, alpha=0.05)
        n_alt_rej_total += int(rejected[5:].sum())
        n_null_rej_total += int(rejected[:5].sum())
    avg_alt_rej = n_alt_rej_total / n_sim
    avg_null_rej = n_null_rej_total / n_sim
    # Most alternatives should be rejected.
    assert avg_alt_rej >= 4.5
    # FDR should be controlled — null rejections small.
    fdr = avg_null_rej / max(avg_null_rej + avg_alt_rej, 1e-9)
    assert fdr <= 0.10


def test_benjamini_hochberg_validates_inputs() -> None:
    with pytest.raises(ValueError, match="alpha"):
        benjamini_hochberg(np.array([0.1, 0.2]), alpha=0.0)
    with pytest.raises(ValueError, match="finite"):
        benjamini_hochberg(np.array([np.nan, 0.2]))
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        benjamini_hochberg(np.array([-0.1, 0.5]))
    with pytest.raises(ValueError, match="1D"):
        benjamini_hochberg(np.zeros((2, 2)))


def test_benjamini_hochberg_empty_input() -> None:
    out = benjamini_hochberg(np.zeros(0, dtype=np.float64))
    assert out.shape == (0,)


# ---------------------------------------------------------------------------
# TOST equivalence
# ---------------------------------------------------------------------------


def test_tost_equivalence_centered_estimate_concludes_equivalence() -> None:
    """At estimate=0, SE=0.05 within margin=0.1, should conclude equivalence with p≈0.025."""
    is_equiv, max_p = tost_equivalence(estimate=0.0, standard_error=0.05, margin=0.1)
    assert is_equiv
    # p = P(Z ≤ -2.0) = 0.02275.
    assert abs(max_p - stats.norm.cdf(-2.0)) < 1e-6


def test_tost_equivalence_at_margin_boundary_does_not_conclude() -> None:
    """At estimate=0.5, SE=0.05 (way above the ±0.1 margin), no equivalence."""
    is_equiv, max_p = tost_equivalence(estimate=0.5, standard_error=0.05, margin=0.1)
    assert not is_equiv
    assert max_p > 0.5  # one of the one-sided p-values is large


def test_tost_equivalence_huge_se_does_not_conclude() -> None:
    """At estimate=0, SE=10 — huge uncertainty — no equivalence."""
    is_equiv, max_p = tost_equivalence(estimate=0.0, standard_error=10.0, margin=0.1)
    assert not is_equiv
    assert max_p > 0.4  # both one-sided tests fail badly


def test_tost_equivalence_uses_t_distribution_when_df_set() -> None:
    """With small df, the t-distribution gives larger p-values than the z-test."""
    is_equiv_z, max_p_z = tost_equivalence(estimate=0.0, standard_error=0.05, margin=0.1)
    _, max_p_t = tost_equivalence(estimate=0.0, standard_error=0.05, margin=0.1, df=5)
    assert max_p_t > max_p_z
    # The z-test at this configuration concludes equivalence; the t-test at
    # df=5 sits right at the alpha=0.05 boundary, so we don't assert it
    # passes — the point of this test is the directional comparison.
    assert is_equiv_z
    # With more df the t-test should also pass cleanly.
    is_equiv_t_large, _ = tost_equivalence(estimate=0.0, standard_error=0.05, margin=0.1, df=50)
    assert is_equiv_t_large


def test_tost_equivalence_validates_inputs() -> None:
    with pytest.raises(ValueError, match="standard_error"):
        tost_equivalence(estimate=0.0, standard_error=0.0, margin=0.1)
    with pytest.raises(ValueError, match="margin"):
        tost_equivalence(estimate=0.0, standard_error=0.05, margin=-0.1)
    with pytest.raises(ValueError, match="alpha"):
        tost_equivalence(estimate=0.0, standard_error=0.05, margin=0.1, alpha=1.0)
    with pytest.raises(ValueError, match="df"):
        tost_equivalence(estimate=0.0, standard_error=0.05, margin=0.1, df=0)
