"""Shared configuration: the trade rule (labels == backtest) and training settings."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict


@dataclass(frozen=True)
class TradeRules:
    """The ONE entry/exit rule used for labels and for the backtest.

    Entry : |z_t| >= entry_z (side = -sign(z_t)), only when flat.
    Exit  : first of
              - target : z crosses back through +/- target_z  (label 1)
              - stop   : |z| >= stop_z                        (label 0)
              - time   : held ``horizon`` days                (label 0)

    ``z`` is the rolling z-score of the spread (``zscore_lookback`` days).
    """

    entry_z: float = 1.5
    target_z: float = 0.5
    stop_z: float = 3.0
    horizon: int = 8  # H, trading days
    zscore_lookback: int = 20

    def __post_init__(self) -> None:
        if not 0 <= self.target_z < self.entry_z < self.stop_z:
            raise ValueError("need 0 <= target_z < entry_z < stop_z")
        if self.horizon < 1:
            raise ValueError("horizon must be >= 1")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GateConfig:
    """Training settings shared by every model so comparisons are like-for-like."""

    input_length: int = 60
    epochs: int = 60
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    max_grad_norm: float = 1.0
    early_stopping_patience: int = 10
    balance_classes: bool = True  # BCE pos_weight = n_neg / n_pos on train
    threshold: float = 0.5  # gate opens when p >= threshold
    device: str = "cpu"
    seed: int = 0
    # Per-model architecture overrides, e.g. {"tft": {"d_model": 32}}
    model_kwargs: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
