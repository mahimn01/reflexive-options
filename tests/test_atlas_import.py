"""Smoke tests for the vendored ATLAS + RAT third-party modules.

Verifies (a) imports resolve, (b) no residual ``trading_algo`` references
remain in the vendored tree, and (c) at least one Mamba block runs a
forward pass on synthetic input.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import torch

THIRD_PARTY = Path(__file__).resolve().parent.parent / "src" / "reflexive_options" / "third_party"


def test_can_import_mamba() -> None:
    from reflexive_options.third_party.atlas.mamba import (
        CausalTransformerBlock,
        MambaBackbone,
        MambaBlock,
    )

    assert CausalTransformerBlock is MambaBlock
    assert MambaBackbone is not None


def test_can_import_train_ppo() -> None:
    """train_ppo.py was Tier-3 SKIP — verify the BC trainer imports instead."""
    # Per the import-surface brief, train_ppo.py is *not* vendored (its
    # OptionsEnvironment is reimplemented in the project's gymnasium env,
    # task #17). The PPO replacement surface lives in train_bc + train_ewc.
    from reflexive_options.third_party.atlas.train_bc import train_behavioral_cloning
    from reflexive_options.third_party.atlas.train_ewc import EWCAdapter

    assert callable(train_behavioral_cloning)
    assert EWCAdapter is not None


def test_rat_reflexivity_meter_imports() -> None:
    from reflexive_options.third_party.rat.reflexivity.meter import (
        ReflexivityMeter,
        ReflexivityStage,
        ReflexivityState,
    )

    assert ReflexivityMeter is not None
    assert ReflexivityStage is not None
    assert ReflexivityState is not None


def test_rat_topology_and_attention_import() -> None:
    from reflexive_options.third_party.rat.attention.flow import AttentionFlow
    from reflexive_options.third_party.rat.attention.tracker import AttentionTracker
    from reflexive_options.third_party.rat.config import (
        AttentionConfig,
        RATConfig,
        WeightingMethod,
    )
    from reflexive_options.third_party.rat.signals import Signal, SignalSource, SignalType
    from reflexive_options.third_party.rat.topology.detector import TopologyDetector

    assert AttentionFlow is not None
    assert AttentionTracker is not None
    assert AttentionConfig is not None
    assert RATConfig is not None
    assert WeightingMethod is not None
    assert Signal is not None
    assert SignalSource is not None
    assert SignalType is not None
    assert TopologyDetector is not None


def test_no_trading_algo_imports() -> None:
    """No file under third_party/ may import from trading_algo.

    Comments and string literals referencing the upstream package name are
    fine (they document the vendoring), but any executable
    ``import trading_algo`` or ``from trading_algo`` must be patched out.
    """
    pattern = re.compile(r"^\s*(from\s+trading_algo|import\s+trading_algo)", re.MULTILINE)
    offenders: list[str] = []
    for f in THIRD_PARTY.rglob("*.py"):
        text = f.read_text()
        if pattern.search(text):
            offenders.append(str(f.relative_to(THIRD_PARTY)))
    assert not offenders, f"trading_algo imports remaining in vendored files: {offenders}"


def test_atlas_adapter_reexports() -> None:
    """The atlas_adapter bridge must surface the public ATLAS+RAT API."""
    from reflexive_options.rl import atlas_adapter

    for name in (
        "ATLASConfig",
        "ATLASModel",
        "MambaBackbone",
        "EWCAdapter",
        "ReflexivityMeter",
        "TopologyDetector",
        "AttentionFlow",
        "Signal",
    ):
        assert hasattr(atlas_adapter, name), f"atlas_adapter missing {name}"


def test_tiny_forward_pass() -> None:
    """Instantiate a minimal Mamba backbone and run a forward pass."""
    from reflexive_options.third_party.atlas.mamba import (
        CausalTransformerBlock,
        MambaBackbone,
    )

    torch.manual_seed(0)

    # Smallest viable config: d_model=8, 1 layer, 1 head, seq_len=4, batch=2.
    d_model, seq_len, batch = 8, 4, 2
    block = CausalTransformerBlock(d_model=d_model, n_heads=1, ffn_mult=2)
    x = torch.randn(batch, seq_len, d_model)
    y = block(x)
    assert y.shape == (batch, seq_len, d_model)
    assert torch.isfinite(y).all()

    backbone = MambaBackbone(d_model=d_model, n_layers=2, n_heads=1, ffn_mult=2)
    z = backbone(x)
    assert z.shape == (batch, seq_len, d_model)
    assert torch.isfinite(z).all()


def test_features_iv_externalization() -> None:
    """The Appendix B Change-1 patch: callers can supply iv_series."""
    import numpy as np

    from reflexive_options.third_party.atlas.features import ATLASFeatureComputer

    rng = np.random.default_rng(0)
    T = 300
    closes = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, T)))
    highs = closes * 1.01
    lows = closes * 0.99
    volumes = rng.uniform(1e6, 2e6, T)
    iv = np.full(T, 0.20)

    fc = ATLASFeatureComputer()
    feats_supplied = fc.compute_features(closes, highs, lows, volumes, iv_series=iv)
    feats_default = fc.compute_features(closes, highs, lows, volumes)
    assert feats_supplied.shape == (T, 12)
    assert feats_default.shape == (T, 12)
    # When we override IV with a constant, the IV column should match.
    assert np.allclose(feats_supplied[:, 5], 0.20)
    # And it should differ from the RV-based default.
    assert not np.allclose(feats_supplied[:, 5], feats_default[:, 5], equal_nan=True)


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
