"""Pre-extraction empirical protocols and legacy reproducibility utilities.

Amendments A13--A15 use sign-agnostic open-interest book summaries. Public OI
does not identify dealer positions. The older signed-GEX H1' code is retained
only to reproduce the superseded A9 registration.
"""

from reflexive_options.empirical.oi_proxy_protocol import (
    A13Design,
    A13RegressionResult,
    A14FamilyDecision,
    A14FamilyResult,
    GammaBookSummary,
    benjamini_hochberg_adjusted,
    build_a13_design,
    classify_a14_family,
    gamma_book_summary,
    interpolate_zero_rate,
    put_call_parity_forward,
    run_a13_regression,
    run_a14_family,
    transform_primary_summaries,
    variance_inflation_factors,
)

__all__ = [
    "A13Design",
    "A13RegressionResult",
    "A14FamilyDecision",
    "A14FamilyResult",
    "GammaBookSummary",
    "benjamini_hochberg_adjusted",
    "build_a13_design",
    "classify_a14_family",
    "gamma_book_summary",
    "interpolate_zero_rate",
    "put_call_parity_forward",
    "run_a13_regression",
    "run_a14_family",
    "transform_primary_summaries",
    "variance_inflation_factors",
]
