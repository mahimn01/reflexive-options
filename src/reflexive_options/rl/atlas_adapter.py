"""Thin adapter re-exporting the vendored ATLAS + RAT public surface.

This is the single import bridge the rest of ``reflexive_options`` should use
when reaching into the vendored ATLAS/RAT modules. Domain-specific glue
(state/action/reward translation, options-env wiring) lives in sibling
modules — not here.
"""

from reflexive_options.third_party.atlas.config import ATLASConfig
from reflexive_options.third_party.atlas.features import (
    ATLASFeatureComputer,
    RollingNormalizer,
)
from reflexive_options.third_party.atlas.inference import ATLASInference
from reflexive_options.third_party.atlas.mamba import (
    CausalTransformerBlock,
    MambaBackbone,
)
from reflexive_options.third_party.atlas.model import ATLASModel
from reflexive_options.third_party.atlas.train_bc import train_behavioral_cloning
from reflexive_options.third_party.atlas.train_ewc import EWCAdapter
from reflexive_options.third_party.rat.attention.flow import AttentionFlow
from reflexive_options.third_party.rat.reflexivity.meter import ReflexivityMeter
from reflexive_options.third_party.rat.signals import Signal, SignalSource, SignalType
from reflexive_options.third_party.rat.topology.detector import TopologyDetector

__all__ = [
    "ATLASConfig",
    "ATLASFeatureComputer",
    "ATLASInference",
    "ATLASModel",
    "AttentionFlow",
    "CausalTransformerBlock",
    "EWCAdapter",
    "MambaBackbone",
    "ReflexivityMeter",
    "RollingNormalizer",
    "Signal",
    "SignalSource",
    "SignalType",
    "TopologyDetector",
    "train_behavioral_cloning",
]
