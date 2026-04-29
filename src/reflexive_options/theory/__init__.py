"""Analytical results — the novel data-free theoretical contributions of the paper.

- `bifurcation`: Hopf bifurcation analysis of the reflexive SDE.
- `stationary`: Fokker-Planck stationary marginal density.
- `sensitivity`: numerical ∂(metric)/∂κ pipeline.
"""

from reflexive_options.theory.bifurcation import (
    HopfScanResult,
    build_bilinear_trilinear_tensors,
    compute_lambda_correction,
    compute_lyapunov_coefficient,
    hopf_scan,
    jacobian_eigenvalues,
    stochastic_hopf_shift_numeric,
    top_lyapunov_exponent_linearised,
)
from reflexive_options.theory.sensitivity import kappa_sensitivity_curve
from reflexive_options.theory.stationary import (
    BimodalityResult,
    StationaryDensity,
    TailIndexCurve,
    compare_to_heston,
    detect_bimodality,
    heston_log_return_cdf,
    heston_log_return_quantiles,
    heston_stationary_variance_density,
    solve_stationary,
    tail_index_vs_kappa_curve,
)

__all__ = [
    "BimodalityResult",
    "HopfScanResult",
    "StationaryDensity",
    "TailIndexCurve",
    "build_bilinear_trilinear_tensors",
    "compare_to_heston",
    "compute_lambda_correction",
    "compute_lyapunov_coefficient",
    "detect_bimodality",
    "heston_log_return_cdf",
    "heston_log_return_quantiles",
    "heston_stationary_variance_density",
    "hopf_scan",
    "jacobian_eigenvalues",
    "kappa_sensitivity_curve",
    "solve_stationary",
    "stochastic_hopf_shift_numeric",
    "tail_index_vs_kappa_curve",
    "top_lyapunov_exponent_linearised",
]
