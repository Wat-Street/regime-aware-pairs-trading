"""Dynamic exit logic: exit when the live z-score falls inside an adaptive band.

Exit when ``|z_t| < z_exit,t`` where ``z_exit,t = (c_z + lambda) / (p_t * kappa_t)``.
Higher reversion confidence/speed shrinks the band (hold longer); higher cost/risk
widens it (exit sooner).
"""

import math

import pandas as pd

from pairs_trading.data.spread import compute_zscore, fit_arma_garch
from pairs_trading.signals.schemas import ExitDecision, Position, SignalConfig

EPS = 1e-12


def compute_kappa(spread_window: pd.Series) -> float:
    """Live mean-reversion speed from a refit ARMA; ``kappa = -ln(phi)``.

    Refits every call (accepted: simplicity over performance). Returns 0.0 when the
    fitted ``phi`` is outside ``(0, 1)`` (no usable mean reversion).
    """
    phi = fit_arma_garch(spread_window).phi
    return -math.log(phi) if 0.0 < phi < 1.0 else 0.0


def dynamic_exit_threshold(p_t: float, kappa_t: float, config: SignalConfig) -> float:
    """Adaptive exit band ``(c_z + lambda) / (p_t * kappa_t)``."""
    return (config.exit_cost_z + config.risk_buffer) / max(p_t * kappa_t, EPS)


def should_exit(
    position: Position,
    spread_window: pd.Series,
    p_t: float,
    config: SignalConfig,
) -> ExitDecision:
    """Decide whether to exit, recomputing kappa and the rolling z-score live."""
    kappa_t = compute_kappa(spread_window)
    z_t = float(
        compute_zscore(spread_window, lookback=config.rolling_lookback).iloc[-1]
    )
    z_exit_t = dynamic_exit_threshold(p_t, kappa_t, config)
    return ExitDecision(
        should_exit=abs(z_t) < z_exit_t,
        z_t=z_t,
        z_exit_t=z_exit_t,
        p_t=p_t,
        kappa_t=kappa_t,
    )
