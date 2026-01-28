"""Compute spread, hedge ratio, and z-score for pairs."""

import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant

from .schemas import Pair, PriceData, SpreadData


def compute_hedge_ratio(price_a: pd.Series, price_b: pd.Series) -> float:
    """
    Compute hedge ratio using OLS regression.

    price_a = alpha + beta * price_b + error

    Returns beta (the hedge ratio).
    """
    y = price_a.values
    X = add_constant(price_b.values)

    model = OLS(y, X).fit()

    # beta is the second coefficient (first is intercept)
    return model.params[1]


def compute_half_life(spread: pd.Series) -> float:
    """
    Compute mean reversion half-life using AR(1) regression.

    spread_t - spread_{t-1} = rho * spread_{t-1} + error
    half_life = -log(2) / log(1 + rho)

    Returns half-life in number of periods (days).
    """
    spread_lag = spread.shift(1).dropna()
    spread_diff = spread.diff().dropna()

    # Align series
    spread_lag = spread_lag.iloc[1:]
    spread_diff = spread_diff.iloc[1:]

    # Regress diff on lag
    X = add_constant(spread_lag.values)
    y = spread_diff.values

    model = OLS(y, X).fit()
    rho = model.params[1]

    # Avoid log of non-positive number
    if rho >= 0:
        return np.inf  # Not mean reverting

    half_life = -np.log(2) / np.log(1 + rho)
    return half_life


def compute_zscore(spread: pd.Series, lookback: int = 20) -> pd.Series:
    """
    Compute rolling z-score of spread.

    z = (spread - rolling_mean) / rolling_std
    """
    mean = spread.rolling(window=lookback).mean()
    std = spread.rolling(window=lookback).std()

    z_score = (spread - mean) / std
    return z_score


def compute_spread(
    pair: Pair,
    data_a: PriceData,
    data_b: PriceData,
    zscore_lookback: int = 20,
) -> SpreadData:
    """
    Compute spread data for a pair.

    Args:
        pair: The Pair object
        data_a: PriceData for asset A
        data_b: PriceData for asset B
        zscore_lookback: Rolling window for z-score calculation

    Returns:
        SpreadData with spread, z-score, hedge ratio, half-life
    """
    price_a = data_a.close
    price_b = data_b.close

    # Compute hedge ratio: A = alpha + beta * B
    hedge_ratio = compute_hedge_ratio(price_a, price_b)

    # Compute spread: A - beta * B
    spread = price_a - hedge_ratio * price_b

    # Compute z-score
    z_score = compute_zscore(spread, lookback=zscore_lookback)

    # Compute half-life
    half_life = compute_half_life(spread)

    return SpreadData(
        pair=pair,
        spread=spread,
        z_score=z_score,
        hedge_ratio=hedge_ratio,
        half_life=half_life,
    )
