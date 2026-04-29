"""Non-reflexive baselines — comparators for the reflexive simulator.

- `heston.HestonSimulator`: time-dependent Heston (5–10 piecewise-constant regimes), the primary baseline.
- `lsv.LSVSimulator`: local-stochastic vol (secondary, slow to calibrate).
- `sv32.SV32Simulator`: 3/2 stochastic vol (smile-shape robustness).
- `gamma_aware.GammaAwareSimulator`: state-symmetric to the reflexive simulator (agent observes G_t)
  but G_t does NOT feed back into dynamics. This is the *clean* ablation isolating
  state-richness from the feedback contribution.
"""

from reflexive_options.baselines.gamma_aware import GammaAwareSimulator
from reflexive_options.baselines.heston import HestonSimulator
from reflexive_options.baselines.lsv import LSVSimulator
from reflexive_options.baselines.sv32 import SV32Simulator

__all__ = [
    "GammaAwareSimulator",
    "HestonSimulator",
    "LSVSimulator",
    "SV32Simulator",
]
