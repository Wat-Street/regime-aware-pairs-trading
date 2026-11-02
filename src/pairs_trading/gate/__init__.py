"""Standard regime-gate interface shared by every model (TFT, Autoformer, W-Transformer, ...).

One contract for all models
---------------------------
INPUT   canonical feature frame per pair (``labels.FEATURE_NAMES``), windowed to
        ``x: float32 (N, input_length, n_features)``, scaled with a scaler fit on
        the training slice only.
OUTPUT  ``predict_proba(x) -> float32 (N,)`` = P(trade entered at the window's
        last timestep reverts to target before stop / time-stop), in [0, 1].
LABELS  ``generate_regime_labels(stop_rule="first_touch")`` with one shared
        ``TradeRules`` (entry / target / stop / horizon) — the exact same rule
        the backtest uses to enter and exit trades.

Usage
-----
    from pairs_trading.gate import GateConfig, TradeRules, build_pair_dataset, RegimeGate

    ds = build_pair_dataset(pair, data_a, data_b, rules=TradeRules(), formation_end="2015-12-31")
    gate = RegimeGate.create("tft", input_length=60, n_features=ds.n_features)
    gate.fit(x_train, y_train, x_val, y_val)
    p = gate.predict_proba(x_test)

Add a new model: write a builder in ``registry.py`` returning an ``nn.Module``
that maps ``(B, L, F) -> logits (B,)`` and decorate it with ``@register("name")``.
Nothing else in the pipeline changes.
"""

from pairs_trading.gate.config import GateConfig, TradeRules
from pairs_trading.gate.dataset import (
    PairDataset,
    WindowSet,
    build_pair_dataset,
    make_windows,
)
from pairs_trading.gate.estimator import GateReversionEstimator
from pairs_trading.gate.model import RegimeGate
from pairs_trading.gate.registry import MODEL_REGISTRY, available_models, register

__all__ = [
    "GateConfig",
    "TradeRules",
    "PairDataset",
    "WindowSet",
    "build_pair_dataset",
    "make_windows",
    "RegimeGate",
    "GateReversionEstimator",
    "MODEL_REGISTRY",
    "available_models",
    "register",
]
