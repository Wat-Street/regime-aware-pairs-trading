"""Unit tests for spread analytics and ARMA/GARCH modeling."""

import importlib.util

import numpy as np
import pandas as pd
import pytest
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant

from pairs_trading.data.schemas import Asset, Pair, PriceData
from pairs_trading.data.spread import (
    check_cointegration,
    compute_half_life,
    compute_hedge_ratio,
    compute_spread,
    compute_zscore,
    fit_arma_garch,
)


def _make_price_data(
    symbol: str,
    close: np.ndarray,
    *,
    index: pd.DatetimeIndex | None = None,
) -> PriceData:
    if index is None:
        index = pd.date_range("2023-01-01", periods=len(close), freq="D")

    close_array = np.asarray(close, dtype=float)
    df = pd.DataFrame(
        {
            "open": close_array,
            "high": close_array + 0.5,
            "low": close_array - 0.5,
            "close": close_array,
            "volume": np.full(len(close_array), 1_000),
        },
        index=index,
    )
    return PriceData(df=df, symbol=symbol)


def test_hedge_ratio_aligns_on_shared_timestamps():
    rng = np.random.default_rng(42)
    n = 120
    true_hedge_ratio = 1.5
    true_intercept = 10.0

    full_index = pd.date_range("2023-01-01", periods=n, freq="D")
    price_b = pd.Series(np.cumsum(rng.normal(size=n)) + 100, index=full_index)
    overlap_index = full_index[7:]
    price_a = pd.Series(
        true_intercept
        + true_hedge_ratio * price_b.loc[overlap_index].to_numpy()
        + rng.normal(scale=1.5, size=len(overlap_index)),
        index=overlap_index,
    )

    calculated_ratio = compute_hedge_ratio(price_a, price_b)

    assert abs(true_hedge_ratio - calculated_ratio) < 0.1


def test_zscore():
    spread = pd.Series([10, 12, 11, 13, 15, 14, 12, 10, 8, 12, 14, 16, 15, 13, 11])
    z_score = compute_zscore(spread, lookback=5)

    window = spread.iloc[2:7]
    manual_z = (spread.iloc[6] - window.mean()) / window.std()

    assert abs(manual_z - z_score.iloc[6]) < 1e-4


def test_zscore_rejects_invalid_lookback():
    with pytest.raises(ValueError, match="lookback"):
        compute_zscore(pd.Series([1.0, 2.0, 3.0]), lookback=1)


def test_half_life():
    rng = np.random.default_rng(42)
    n = 500
    theta = 0.1

    spread_mr = [0.0]
    for _ in range(1, n):
        spread_mr.append(spread_mr[-1] - theta * spread_mr[-1] + rng.normal())

    half_life = compute_half_life(pd.Series(spread_mr))
    theoretical_hl = -np.log(2) / np.log(1 - theta)

    assert abs(theoretical_hl - half_life) < 3


def test_half_life_returns_inf_for_invalid_domain():
    alternating_spread = pd.Series([1.0, -1.0] * 80)
    assert compute_half_life(alternating_spread) == np.inf


def test_compute_spread_with_prefetched_price_data_uses_residuals_and_cointegration():
    rng = np.random.default_rng(21)
    n = 150
    index = pd.date_range("2023-01-01", periods=n, freq="D")
    price_b = 100 + np.cumsum(rng.normal(0, 1, n))
    price_a = 8 + 1.25 * price_b + rng.normal(0, 0.5, n)

    pair = Pair(asset_a=Asset(symbol="MSFT"), asset_b=Asset(symbol="AAPL"))
    data_a = _make_price_data("MSFT", price_a, index=index)
    data_b = _make_price_data("AAPL", price_b[10:], index=index[10:])

    spread_data = compute_spread(pair, data_a, data_b, zscore_lookback=20)

    expected_a = pd.Series(price_a[10:], index=index[10:], name="price_a")
    expected_b = pd.Series(price_b[10:], index=index[10:], name="price_b")
    regression = OLS(expected_a, add_constant(expected_b)).fit()
    expected_intercept = float(regression.params["const"])
    expected_beta = float(regression.params["price_b"])
    expected_spread = expected_a - (expected_intercept + expected_beta * expected_b)

    assert spread_data.pair == pair
    assert len(spread_data.spread) == n - 10
    assert spread_data.spread.index.equals(index[10:])
    assert np.isclose(spread_data.intercept, expected_intercept)
    assert np.isclose(spread_data.hedge_ratio, expected_beta)
    assert np.allclose(spread_data.spread.to_numpy(), expected_spread.to_numpy())
    assert spread_data.cointegration.is_cointegrated
    assert spread_data.cointegration.method == "engle_granger"
    assert spread_data.half_life is not None
    assert spread_data.z_score.isna().sum() >= 19


def test_compute_spread_non_cointegrated_pair_returns_none_half_life():
    rng = np.random.default_rng(7)
    n = 700
    index = pd.date_range("2023-01-01", periods=n, freq="D")
    price_a = np.cumsum(rng.normal(0.1, 1.0, n))
    price_b = np.cumsum(rng.normal(-0.05, 1.2, n))

    pair = Pair(asset_a=Asset(symbol="MSFT"), asset_b=Asset(symbol="AAPL"))
    data_a = _make_price_data("MSFT", price_a, index=index)
    data_b = _make_price_data("AAPL", price_b, index=index)

    spread_data = compute_spread(pair, data_a, data_b, zscore_lookback=20)

    assert not spread_data.cointegration.is_cointegrated
    assert spread_data.half_life is None


def test_compute_spread_rejects_symbol_mismatch():
    index = pd.date_range("2023-01-01", periods=40, freq="D")
    pair = Pair(asset_a=Asset(symbol="MSFT"), asset_b=Asset(symbol="AAPL"))
    data_a = _make_price_data("GOOG", np.arange(40), index=index)
    data_b = _make_price_data("AAPL", np.arange(40), index=index)

    with pytest.raises(ValueError, match="asset_a"):
        compute_spread(pair, data_a, data_b)


def test_cointegration_detects_pair_level_relationship():
    rng = np.random.default_rng(42)
    n = 600
    base = np.cumsum(rng.normal(0, 1, n))
    price_b = pd.Series(base, index=pd.date_range("2022-01-01", periods=n, freq="D"))
    stationary_noise = np.zeros(n)
    for t in range(1, n):
        stationary_noise[t] = 0.6 * stationary_noise[t - 1] + rng.normal(0, 0.5)
    price_a = pd.Series(
        4.0 + 1.8 * base + stationary_noise,
        index=price_b.index,
    )

    result = check_cointegration(price_a, price_b)

    assert result.is_cointegrated
    assert result.p_value < 0.05


def test_cointegration_rejects_independent_random_walks():
    rng = np.random.default_rng(7)
    n = 700
    index = pd.date_range("2022-01-01", periods=n, freq="D")
    price_a = pd.Series(np.cumsum(rng.normal(0.1, 1.0, n)), index=index)
    price_b = pd.Series(np.cumsum(rng.normal(-0.05, 1.2, n)), index=index)

    result = check_cointegration(price_a, price_b)

    assert not result.is_cointegrated
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
