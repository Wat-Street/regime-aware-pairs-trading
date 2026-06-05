"""Public API for the signal-generation and exit-logic layer."""

from pairs_trading.signals.exit import (
    compute_kappa,
    dynamic_exit_threshold,
    should_exit,
)
from pairs_trading.signals.generate import generate_signal
from pairs_trading.signals.history import FeatureStore
from pairs_trading.signals.params import default_config
from pairs_trading.signals.position import live_spread, open_position
from pairs_trading.signals.reversion import BaselineReversionEstimator
from pairs_trading.signals.schemas import (
    ExitDecision,
    FeatureRow,
    Position,
    Side,
    Signal,
    SignalConfig,
)
from pairs_trading.signals.sizing import size_position

__all__ = [
    "generate_signal",
    "open_position",
    "live_spread",
    "should_exit",
    "size_position",
    "compute_kappa",
    "dynamic_exit_threshold",
    "BaselineReversionEstimator",
    "FeatureStore",
    "Signal",
    "Position",
    "ExitDecision",
    "FeatureRow",
    "SignalConfig",
    "Side",
    "default_config",
]
