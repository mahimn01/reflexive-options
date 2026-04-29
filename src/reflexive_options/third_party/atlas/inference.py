"""Live inference adapter for ATLAS."""

from __future__ import annotations

from collections import deque
from datetime import datetime

import numpy as np
import torch

from reflexive_options.third_party.atlas.config import ATLASConfig
from reflexive_options.third_party.atlas.model import ATLASModel
from reflexive_options.third_party.atlas.features import ATLASFeatureComputer, RollingNormalizer

# NOTE: ``execution_bridge`` (TradeDecision, action_to_trade) is intentionally
# NOT vendored — it embeds options-wheel domain logic. Callers should map the
# raw 5-D action vector returned by ``predict`` to their own trade
# representation via the adapter at ``reflexive_options.rl.atlas_adapter``.


class ATLASInference:
    """
    Load trained model, maintain rolling 90-day buffer, generate daily predictions.

    Usage:
        atlas = ATLASInference.from_checkpoint('checkpoints/atlas/best.pt')
        action = atlas.predict(
            date=datetime.now(), price=245.73, high=247.10, low=244.50,
            volume=1_200_000, position_state=0, position_pnl=0.0,
            days_in_trade=0, cash_pct=1.0,
        )
        # `action` is an (5,) numpy array; the caller maps it to a domain
        # trade decision (see ``reflexive_options.rl.atlas_adapter``).
    """

    def __init__(self, model: ATLASModel, config: ATLASConfig, device: str = "cpu"):
        self.model = model.to(device).eval()
        self.config = config
        self.device = device
        self.feature_computer = ATLASFeatureComputer()
        self.normalizer = RollingNormalizer(lookback=252)

        self._closes: deque[float] = deque(maxlen=400)
        self._highs: deque[float] = deque(maxlen=400)
        self._lows: deque[float] = deque(maxlen=400)
        self._volumes: deque[float] = deque(maxlen=400)
        self._dates: deque[datetime] = deque(maxlen=400)

    @classmethod
    def from_checkpoint(cls, path: str, device: str = "cpu") -> "ATLASInference":
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        config = checkpoint.get("config", ATLASConfig())
        if isinstance(config, dict):
            config = ATLASConfig(**config)
        model = ATLASModel(config)
        model.load_state_dict(checkpoint["model_state_dict"])
        return cls(model, config, device)

    def update_buffer(self, date: datetime, price: float, high: float, low: float, volume: float) -> None:
        self._dates.append(date)
        self._closes.append(price)
        self._highs.append(high)
        self._lows.append(low)
        self._volumes.append(volume)

    def predict(
        self,
        date: datetime,
        price: float,
        high: float,
        low: float,
        volume: float,
        position_state: float = 0.0,
        position_pnl: float = 0.0,
        days_in_trade: float = 0.0,
        cash_pct: float = 1.0,
        target_sharpe: float = 1.0,
    ) -> np.ndarray:
        """Run a forward pass and return the raw (5,) action vector.

        Returns ``None`` (well, raises or returns a zero vector) when the
        rolling buffer has fewer bars than ``context_len + 252``. Callers map
        the action to a domain decision via their own adapter — this module
        does not embed options-wheel logic.
        """
        self.update_buffer(date, price, high, low, volume)

        n = len(self._closes)
        if n < self.config.context_len + 252:
            # Not enough history for a confident forward pass; return a
            # zero/no-op action so the caller can decide what to do.
            return np.zeros(5, dtype=np.float32)

        closes = np.array(self._closes)
        highs = np.array(self._highs)
        lows = np.array(self._lows)
        volumes = np.array(self._volumes)

        # Compute features
        raw_features = self.feature_computer.compute_features(closes, highs, lows, volumes)

        # Add position state features
        pos_features = np.zeros((len(raw_features), 4))
        pos_features[-1] = [position_state, position_pnl, days_in_trade, cash_pct]
        full_features = np.concatenate([raw_features, pos_features], axis=1)  # (T, 16)

        # Normalize
        normed, mu_arr, sigma_arr = self.normalizer.normalize(full_features)

        # Extract 90-day window (last 90 days)
        L = self.config.context_len
        window = normed[-L:]  # (90, 16)
        mu_window = mu_arr[-L:]
        sigma_window = sigma_arr[-L:]

        # Build tensors
        features_t = torch.tensor(window, dtype=torch.float32).unsqueeze(0)  # (1, 90, 16)
        ts_t = torch.tensor([d.timestamp() for d in list(self._dates)[-L:]], dtype=torch.float32).unsqueeze(0)
        dates_list = list(self._dates)[-L:]
        dow_t = torch.tensor([d.weekday() for d in dates_list], dtype=torch.long).unsqueeze(0)
        mo_t = torch.tensor([d.month - 1 for d in dates_list], dtype=torch.long).unsqueeze(0)
        opex_t = torch.zeros(1, L)
        qtr_t = torch.zeros(1, L)
        mu_t = torch.tensor(mu_window, dtype=torch.float32).unsqueeze(0)
        sigma_t = torch.tensor(sigma_window, dtype=torch.float32).unsqueeze(0)
        rtg_t = torch.tensor([target_sharpe], dtype=torch.float32)

        # Forward
        with torch.no_grad():
            action = self.model(
                features_t.to(self.device), ts_t.to(self.device),
                dow_t.to(self.device), mo_t.to(self.device),
                opex_t.to(self.device), qtr_t.to(self.device),
                mu_t.to(self.device), sigma_t.to(self.device),
                rtg_t.to(self.device),
            )

        return action.squeeze(0).cpu().numpy()
