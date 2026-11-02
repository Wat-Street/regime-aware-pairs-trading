"""Connect any trained ``RegimeGate`` to the signals layer (direction + dynamic exit).

``signals.reversion.ReversionProbabilityEstimator`` is the seam the direction+exit
PR left for "a future transformer". This adapter fills it, so ``should_exit`` can
use a TFT / Autoformer / W-Transformer ``p_t`` with no change to ``exit.py``:

    gate = RegimeGate.load("results/<run>/models/tft.pt")
    est = GateReversionEstimator(gate)
    p_t = est.probability(pair_dataset.features.loc[:today], horizon=cfg.reversion_horizon)
    decision = should_exit(position, spread_window, p_t, cfg)

It can also gate ENTRIES: ``est.allow_entry(features)`` -> bool (p >= threshold).
"""

from __future__ import annotations

import pandas as pd

from pairs_trading.data.labels import FEATURE_NAMES
from pairs_trading.gate.model import RegimeGate


class GateReversionEstimator:
    """Implements ``ReversionProbabilityEstimator`` on top of a fitted ``RegimeGate``.

    ``features`` must contain the gate's canonical feature columns
    (``labels.FEATURE_NAMES``) for at least ``input_length`` days, oldest -> newest.
    """

    def __init__(self, gate: RegimeGate, floor: float = 1e-3) -> None:
        self.gate = gate
        self.floor = floor  # p_t sits in a denominator in the exit band

    def probability(self, features: pd.DataFrame, horizon: int) -> float:
        missing = [c for c in FEATURE_NAMES if c not in features.columns]
        if missing:
            raise ValueError(f"features missing canonical columns {missing}")
        frame = features[list(FEATURE_NAMES)]
        if len(frame) < self.gate.config.input_length:
            raise ValueError(
                f"need >= {self.gate.config.input_length} rows of features, got {len(frame)}"
            )
        return max(self.gate.predict_frame(frame), self.floor)

    def allow_entry(self, features: pd.DataFrame) -> bool:
        return self.probability(features, horizon=0) >= self.gate.config.threshold
