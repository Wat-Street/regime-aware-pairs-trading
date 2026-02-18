"""Test spread calculations."""

from datetime import datetime
import importlib.util

import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant

from pairs_trading.data.fetcher import fetch_pair_data
from pairs_trading.data.schemas import Asset, Pair
from pairs_trading.data.spread import (
    compute_spread,
    compute_hedge_ratio,
    compute_zscore,
    compute_half_life,
    test_cointegration,
    fit_arma_garch,
)


def test_hedge_ratio():
    """Test hedge ratio with synthetic data."""
    print("\n=== Test 1: Hedge Ratio ===")

    np.random.seed(42)
    n = 100
    true_hedge_ratio = 1.5
    true_alpha = 10

    price_b = np.cumsum(np.random.randn(n)) + 100
    noise = np.random.randn(n) * 2
    price_a = true_alpha + true_hedge_ratio * price_b + noise

    price_a = pd.Series(price_a)
    price_b = pd.Series(price_b)

    calculated_ratio = compute_hedge_ratio(price_a, price_b)

    print(f"True hedge ratio: {true_hedge_ratio}")
    print(f"Calculated hedge ratio: {calculated_ratio:.4f}")
    print(f"Difference: {abs(true_hedge_ratio - calculated_ratio):.4f}")

    assert abs(true_hedge_ratio - calculated_ratio) < 0.1
    print("✓ PASSED")


def test_zscore():
    """Test z-score with synthetic data."""
    print("\n=== Test 2: Z-Score ===")

    spread = pd.Series([10, 12, 11, 13, 15, 14, 12, 10, 8, 12, 14, 16, 15, 13, 11])
    lookback = 5

    z_score = compute_zscore(spread, lookback=lookback)

    window = spread.iloc[2:7]
    manual_mean = window.mean()
    manual_std = window.std()
    manual_z = (spread.iloc[6] - manual_mean) / manual_std

    print(f"Manual z-score: {manual_z:.4f}")
    print(f"Calculated z-score: {z_score.iloc[6]:.4f}")

    assert abs(manual_z - z_score.iloc[6]) < 0.0001
    print("✓ PASSED")


def test_half_life():
    """Test half-life with mean-reverting series."""
    print("\n=== Test 3: Half-Life ===")

    np.random.seed(42)
    n = 500
    theta = 0.1
    mu = 0
    sigma = 1

    spread_mr = [0]
    for i in range(1, n):
        spread_mr.append(
            spread_mr[-1] - theta * (spread_mr[-1] - mu) + sigma * np.random.randn()
        )

    spread_mr = pd.Series(spread_mr)
    half_life = compute_half_life(spread_mr)
    theoretical_hl = -np.log(2) / np.log(1 - theta)

    print(f"Theoretical half-life: {theoretical_hl:.2f}")
    print(f"Calculated half-life: {half_life:.2f}")

    assert abs(theoretical_hl - half_life) < 3
    print("✓ PASSED")


def test_real_data():
    """Test with real MSFT vs AAPL data."""
    print("\n=== Test 4: Real Data (MSFT vs AAPL) ===")

    asset_a = Asset(symbol="MSFT")
    asset_b = Asset(symbol="AAPL")
    pair = Pair(asset_a=asset_a, asset_b=asset_b)

    start = datetime(2022, 1, 1)
    end = datetime(2023, 1, 1)

    data_a, data_b = fetch_pair_data(asset_a, asset_b, start, end)
    spread_data = compute_spread(pair, data_a, data_b, zscore_lookback=20)

    print(f"Pair: {pair.pair_id}")
    print(f"Data points: {len(spread_data.spread)}")
    print(f"Hedge ratio: {spread_data.hedge_ratio:.4f}")
    print(f"Half-life: {spread_data.half_life:.2f} days")
    print(
        f"Z-score range: [{spread_data.z_score.min():.2f}, {spread_data.z_score.max():.2f}]"
    )

    # Verify manually
    y = data_a.close.values
    X = add_constant(data_b.close.values)
    model = OLS(y, X).fit()
    manual_hedge_ratio = model.params[1]

    assert np.isclose(spread_data.hedge_ratio, manual_hedge_ratio)
    print("✓ PASSED")


def test_cointegration_stationary():
    """Test ADF-based stationarity detection on stationary data."""
    print("\n=== Test 5: Cointegration/Stationarity (Stationary Series) ===")

    rng = np.random.default_rng(42)
    n = 600
    phi = 0.7
    eps = rng.normal(0, 1, n)
    spread = np.zeros(n)
    for t in range(1, n):
        spread[t] = phi * spread[t - 1] + eps[t]

    result = test_cointegration(pd.Series(spread))
    print(f"ADF p-value: {result.p_value:.6f}")
    print(f"Is stationary: {result.is_stationary}")

    assert result.is_stationary
    assert result.p_value < 0.05
    print("✓ PASSED")


def test_cointegration_non_stationary():
    """Test ADF-based stationarity detection on non-stationary data."""
    print("\n=== Test 6: Cointegration/Stationarity (Non-Stationary Series) ===")

    rng = np.random.default_rng(7)
    n = 600
    random_walk = np.cumsum(rng.normal(0, 1, n))

    result = test_cointegration(pd.Series(random_walk))
    print(f"ADF p-value: {result.p_value:.6f}")
    print(f"Is stationary: {result.is_stationary}")

    assert not result.is_stationary
    assert result.p_value >= 0.05
    print("✓ PASSED")


def test_arma_garch_fit():
    """Test ARMA(1,1)+GARCH(1,1)-t fitting pipeline."""
    print("\n=== Test 7: ARMA+GARCH ===")

    if importlib.util.find_spec("arch") is None:
        print("Skipping ARMA+GARCH test: 'arch' package not installed")
        return

    rng = np.random.default_rng(123)
    n = 700
    spread = np.zeros(n)
    for t in range(1, n):
        shock_scale = 0.6 + 0.4 * abs(np.sin(t / 25))
        spread[t] = 0.65 * spread[t - 1] + rng.normal(0, shock_scale)

    spread_series = pd.Series(spread)
    result = fit_arma_garch(spread_series)

    print(f"mu={result.mu:.4f}, phi={result.phi:.4f}, theta={result.theta:.4f}")
    print(
        f"next spread forecast={result.spread_forecast_next:.4f}, "
        f"next variance forecast={result.variance_forecast_next:.4f}"
    )

    assert len(result.arma_residuals) == len(spread_series)
    assert len(result.conditional_volatility) == len(spread_series)
    assert len(result.vol_scaled_z_score) == len(spread_series)
    assert np.isfinite(result.spread_forecast_next)
    assert np.isfinite(result.variance_forecast_next)
    assert result.variance_forecast_next > 0
    print("✓ PASSED")


if __name__ == "__main__":
    print("=" * 50)
    print("SPREAD CALCULATION TESTS")
    print("=" * 50)

    test_hedge_ratio()
    test_zscore()
    test_half_life()
    test_real_data()
    test_cointegration_stationary()
    test_cointegration_non_stationary()
    test_arma_garch_fit()

    print("\n" + "=" * 50)
    print("ALL TESTS PASSED ✓")
    print("=" * 50)
