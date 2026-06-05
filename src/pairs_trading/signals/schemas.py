"""Schemas for the signal-generation and exit-logic layer."""

from enum import Enum
from typing import Optional

import pandas as pd
from pydantic import BaseModel, ConfigDict

from pairs_trading.data.schemas import CointegrationResult, Pair


class Side(str, Enum):
    """Trade direction for a pair."""

    LONG_A_SHORT_B = "long_a_short_b"
    SHORT_A_LONG_B = "short_a_long_b"
    FLAT = "flat"


class SignalConfig(BaseModel):
    """Hyperparameters for signal generation and exit logic.

    Schema only: default values live in ``signals/params.py`` (``default_config``).
    """

    model_config = ConfigDict(frozen=True)

    entry_threshold: float
    capital_per_trade: float
    min_observations: int
    exit_cost_z: float
    risk_buffer: float
    reversion_horizon: int
    rolling_lookback: int


class Signal(BaseModel):
    """Entry decision plus the frozen model state captured at one time step."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    pair: Pair
    timestamp: pd.Timestamp
    side: Side
    z_score: float
    should_enter: bool
    mu: float
    hedge_ratio: float
    intercept: float
    sigma: float
    price_a: float
    price_b: float
    half_life: Optional[float]
    cointegration: CointegrationResult


class Position(BaseModel):
    """Snapshot taken at entry; ``hedge_ratio`` and ``intercept`` are frozen for the trade."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    pair: Pair
    side: Side
    entry_time: pd.Timestamp
    hedge_ratio: float
    intercept: float
    mu: float
    sigma: float
    entry_z: float
    entry_price_a: float
    entry_price_b: float
    volume_a: float
    volume_b: float


class ExitDecision(BaseModel):
    """Output of the dynamic exit check at one time step."""

    model_config = ConfigDict(frozen=True)

    should_exit: bool
    z_t: float
    z_exit_t: float
    p_t: float
    kappa_t: float


class FeatureRow(BaseModel):
    """One day of stored features, fed to the reversion-probability estimator."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    timestamp: pd.Timestamp
    rolling_z: float
    volatility: float
    kappa: float
