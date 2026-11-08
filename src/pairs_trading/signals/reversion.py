"""Reversion-probability estimation for the dynamic exit (p_t)."""

import math
from typing import Protocol

import pandas as pd


class ReversionProbabilityEstimator(Protocol):
    """Estimates P(spread reverts within ``horizon`` days) from recent features."""

    def probability(self, features: pd.DataFrame, horizon: int) -> float: ...


class BaselineReversionEstimator:
    """Zero-ML baseline: faster mean reversion implies a higher reversion probability."""

    def probability(self, features: pd.DataFrame, horizon: int) -> float:
        kappa = float(features["kappa"].iloc[-1])
        return 1.0 - math.exp(-max(kappa, 0.0) * horizon)


# Phase 3: TransformerReversionEstimator implements the same Protocol.
