"""Unit tests for spread analytics and ARMA/GARCH modeling."""

import importlib.util

import numpy as np
import pandas as pd
import pytest

from pairs_trading.data.schemas import Asset, Pair, PriceData
from pairs_trading.data.spread import (
    compute_half_life,
    compute_hedge_ratio,
    compute_spread,
    compute_zscore,
    fit_arma_garch,
    test_cointegration,
)


def _make_price_data(symbol: str, close: np.ndarray) -> PriceData:
    index = pd.date_range("2023-01-01", periods=len(close), freq="D")
    df = pd.DataFrame(
        {
            "open": close,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": np.full(len(close), 1_000),
        },
        index=index,
    )
    return PriceData(df=df, symbol=symbol)


def test_hedge_ratio():
    np.random.seed(42)
    n = 100
    true_hedge_ratio = 1.5
    true_alpha = 10

    price_b = np.cumsum(np.random.randn(n)) + 100
    noise = np.random.randn(n) * 2
    price_a = true_alpha + true_hedge_ratio * price_b + noise

    calculated_ratio = compute_hedge_ratio(pd.Series(price_a), pd.Series(price_b))

    assert abs(true_hedge_ratio - calculated_ratio) < 0.1


def test_zscore():
    spread = pd.Series([10, 12, 11, 13, 15, 14, 12, 10, 8, 12, 14, 16, 15, 13, 11])
    z_score = compute_zscore(spread, lookback=5)

    window = spread.iloc[2:7]
    manual_z = (spread.iloc[6] - window.mean()) / window.std()

    assert abs(manual_z - z_score.iloc[6]) < 1e-4


def test_half_life():
    np.random.seed(42)
    n = 500
    theta = 0.1

    spread_mr = [0.0]
    for _ in range(1, n):
        spread_mr.append(spread_mr[-1] - theta * spread_mr[-1] + np.random.randn())

    half_life = compute_half_life(pd.Series(spread_mr))
    theoretical_hl = -np.log(2) / np.log(1 - theta)

    assert abs(theoretical_hl - half_life) < 3


def test_compute_spread_with_prefetched_price_data():
    rng = np.random.default_rng(21)
    n = 90
    price_b = 100 + np.cumsum(rng.normal(0, 1, n))
    price_a = 8 + 1.25 * price_b + rng.normal(0, 0.8, n)

    pair = Pair(asset_a=Asset(symbol="MSFT"), asset_b=Asset(symbol="AAPL"))
    data_a = _make_price_data("MSFT", price_a)
    data_b = _make_price_data("AAPL", price_b)

    spread_data = compute_spread(pair, data_a, data_b, zscore_lookback=20)

    assert spread_data.pair == pair
    assert len(spread_data.spread) == n
    assert spread_data.z_score.isna().sum() >= 19
    assert np.isfinite(spread_data.hedge_ratio)


def test_cointegration_stationary():
    rng = np.random.default_rng(42)
    n = 600
    phi = 0.7
    eps = rng.normal(0, 1, n)
    spread = np.zeros(n)
    for t in range(1, n):
        spread[t] = phi * spread[t - 1] + eps[t]

    result = test_cointegration(pd.Series(spread))

    assert result.is_stationary
    assert result.p_value < 0.05


def test_cointegration_non_stationary():
    rng = np.random.default_rng(7)
    random_walk = np.cumsum(rng.normal(0, 1, 600))

    result = test_cointegration(pd.Series(random_walk))

    assert not result.is_stationary
    assert result.p_value >= 0.05


@pytest.mark.skipif(
    importlib.util.find_spec("arch") is None,
    reason="'arch' package not installed",
)
def test_arma_garch_fit_searches_multiple_parameter_combinations():
    rng = np.random.default_rng(123)
    n = 700
    spread = np.zeros(n)
    for t in range(1, n):
        shock_scale = 0.6 + 0.4 * abs(np.sin(t / 25))
        spread[t] = 0.65 * spread[t - 1] + rng.normal(0, shock_scale)

    result = fit_arma_garch(
        pd.Series(spread),
        arma_candidates=[(1, 0, 0), (1, 0, 1)],
        garch_candidates=[(1, 1), (1, 2)],
    )

    assert result.arma_order in {(1, 0, 0), (1, 0, 1)}
    assert result.garch_order in {(1, 1), (1, 2)}
    assert len(result.arma_residuals) == n
    assert len(result.conditional_volatility) == n
    assert len(result.vol_scaled_z_score) == n
    assert np.isfinite(result.spread_forecast_next)
    assert np.isfinite(result.variance_forecast_next)
    assert result.variance_forecast_next > 0
    assert np.isfinite(result.arma_aic)
    assert np.isfinite(result.garch_aic)
