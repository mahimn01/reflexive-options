"""Reflexive simulator — the central contribution of the paper.

dS/S = (μ + κ · G(S, t, v)) dt + σ(S, t, v) dW_S
dv   = κ_v (θ_v - v) dt + ξ √v dW_v
d⟨W_S, W_v⟩ = ρ dt

When κ = 0 the reflexive simulator reduces to standard time-dep Heston.
G(S, t) is the aggregate dealer-gamma exposure (gamma_aggregator.py).
"""

from reflexive_options.simulator.coupling import check_coupling_stable
from reflexive_options.simulator.gamma_aggregator import GammaAggregator
from reflexive_options.simulator.integrators import euler_maruyama_step
from reflexive_options.simulator.reflexive import ReflexiveSimulator
from reflexive_options.simulator.stability import detect_blowup

__all__ = [
    "GammaAggregator",
    "ReflexiveSimulator",
    "check_coupling_stable",
    "detect_blowup",
    "euler_maruyama_step",
]
