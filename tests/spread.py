"""Compute spread, hedge ratio, and z-score for pairs."""

from datetime import datetime
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

from ..src.pairs_trading.data.fetcher import fetch_pair_data
from ..src.pairs_trading.data.schemas import ArmaGarchResult, CointegrationResult, Pair, PriceData, SpreadData


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


def test_cointegration(spread: pd.Series, alpha: float = 0.05) -> CointegrationResult:
    """
    Test spread stationarity via ADF unit-root test.

    If p-value < alpha, treat spread as stationary (cointegration-compatible).
    """
    # Work on a clean numeric series so statsmodels receives valid input.
    spread_clean = spread.dropna().astype(float)
    if len(spread_clean) < 20:
        raise ValueError("Spread series must have at least 20 non-null observations")

    # ADF null hypothesis: unit root (non-stationary spread).
    test_statistic, p_value, _, _, critical_values, _ = adfuller(
        spread_clean, autolag="AIC"
    )
    return CointegrationResult(
        test_statistic=float(test_statistic),
        p_value=float(p_value),
        critical_values={k: float(v) for k, v in critical_values.items()},
        is_stationary=bool(p_value < alpha),
        alpha=alpha,
    )


def _iter_model_candidates(
    arma_order: tuple[int, int, int],
    garch_order: tuple[int, int],
    arma_candidates: Optional[Iterable[tuple[int, int, int]]] = None,
    garch_candidates: Optional[Iterable[tuple[int, int]]] = None,
) -> list[tuple[tuple[int, int, int], tuple[int, int]]]:
    """Build the ARMA/GARCH candidate grid, preserving caller order."""
    arma_grid = list(arma_candidates) if arma_candidates is not None else [arma_order]
    garch_grid = list(garch_candidates) if garch_candidates is not None else [garch_order]
    return [(arma_cfg, garch_cfg) for arma_cfg in arma_grid for garch_cfg in garch_grid]


def fit_arma_garch(
    spread: pd.Series,
    arma_order: tuple[int, int, int] = (1, 0, 1),
    garch_order: tuple[int, int] = (1, 1),
    *,
    arma_candidates: Optional[Iterable[tuple[int, int, int]]] = None,
    garch_candidates: Optional[Iterable[tuple[int, int]]] = None,
    distribution: str = "t",
) -> ArmaGarchResult:
    """
    Fit ARMA on spread and GARCH on ARMA residuals.

    Optionally evaluates multiple model orders and returns the lowest-AIC fit.
    """
    # Keep one aligned, non-null series for both ARMA and GARCH stages.
    spread_clean = spread.dropna().astype(float)
    if len(spread_clean) < 40:
        raise ValueError("Spread series must have at least 40 non-null observations")

    try:
        from arch import arch_model
    except ImportError as exc:
        raise ImportError(
            "fit_arma_garch requires the 'arch' package. Install dependencies first."
        ) from exc

    best_result = None
    candidate_errors = []

    for arma_cfg, garch_cfg in _iter_model_candidates(
        arma_order, garch_order, arma_candidates, garch_candidates
    ):
        try:
            arma = ARIMA(spread_clean, order=arma_cfg, trend="c")
            arma_fit = arma.fit()
            arma_residuals = pd.Series(arma_fit.resid, index=spread_clean.index)

            garch = arch_model(
                arma_residuals,
                mean="Zero",
                vol="GARCH",
                p=garch_cfg[0],
                q=garch_cfg[1],
                dist=distribution,
                rescale=False,
            )
            garch_fit = garch.fit(disp="off")
        except Exception as exc:  # pragma: no cover - best-effort model search
            candidate_errors.append(f"ARMA{arma_cfg}/GARCH{garch_cfg}: {exc}")
            continue

        score = (float(arma_fit.aic), float(garch_fit.aic))
        if best_result is None or score < best_result["score"]:
            best_result = {
                "score": score,
                "arma_order": arma_cfg,
                "garch_order": garch_cfg,
                "arma_fit": arma_fit,
                "garch_fit": garch_fit,
                "arma_residuals": arma_residuals,
            }

    if best_result is None:
        raise ValueError(
            "Unable to fit any ARMA/GARCH candidate combination. "
            + "; ".join(candidate_errors)
        )

    arma_fit = best_result["arma_fit"]
    garch_fit = best_result["garch_fit"]
    arma_residuals = best_result["arma_residuals"]
    selected_arma_order = best_result["arma_order"]
    selected_garch_order = best_result["garch_order"]

    # Map statsmodels coefficients into RAPTS notation (mu, phi, theta).
    phi = float(arma_fit.arparams[0]) if len(arma_fit.arparams) else 0.0
    theta = float(arma_fit.maparams[0]) if len(arma_fit.maparams) else 0.0
    const = float(arma_fit.params.get("const", 0.0))
    mu = const / (1.0 - phi) if not np.isclose(1.0 - phi, 0.0) else float("nan")
    spread_forecast_next = float(arma_fit.get_forecast(steps=1).predicted_mean.iloc[0])

    params = garch_fit.params
    omega = float(params["omega"])
    alpha = float(params["alpha[1]"])
    beta = float(params[[idx for idx in params.index if idx.startswith("beta[")][0]])
    nu = float(params.get("nu", np.nan))

    conditional_volatility = pd.Series(
        garch_fit.conditional_volatility, index=spread_clean.index
    )
    variance_forecast_next = float(
        garch_fit.forecast(horizon=1, reindex=False).variance.iloc[-1, 0]
    )
    vol_scaled_z_score = (spread_clean - mu) / conditional_volatility

    return ArmaGarchResult(
        arma_order=selected_arma_order,
        garch_order=selected_garch_order,
        arma_aic=float(arma_fit.aic),
        garch_aic=float(garch_fit.aic),
        arma_params={str(k): float(v) for k, v in arma_fit.params.items()},
        garch_params={str(k): float(v) for k, v in garch_fit.params.items()},
        mu=mu,
        phi=phi,
        theta=theta,
        arma_residuals=arma_residuals,
        spread_forecast_next=spread_forecast_next,
        omega=omega,
        alpha=alpha,
        beta=beta,
        nu=nu,
        conditional_volatility=conditional_volatility,
        variance_forecast_next=variance_forecast_next,
        vol_scaled_z_score=vol_scaled_z_score,
    )


def compute_spread(
    pair: Pair,
    data_a: Optional[PriceData] = None,
    data_b: Optional[PriceData] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    source: str = "yfinance",
    zscore_lookback: int = 20,
) -> SpreadData:
    """
    Compute spread data for a pair.

    Can either pass pre-fetched data OR date range to auto-fetch.

    Usage:
        # Option 1: Pass pre-fetched data
        spread_data = compute_spread(pair, data_a, data_b)

        # Option 2: Auto-fetch with dates
        spread_data = compute_spread(pair, start_date=start, end_date=end)

    Args:
        pair: The Pair object
        data_a: PriceData for asset A (optional if dates provided)
        data_b: PriceData for asset B (optional if dates provided)
        start_date: Start date for auto-fetch
        end_date: End date for auto-fetch
        source: Data provider for auto-fetch
        zscore_lookback: Rolling window for z-score calculation

    Returns:
        SpreadData with spread, z-score, hedge ratio, half-life
    """
    # Auto-fetch if data not provided
    if data_a is None or data_b is None:
        if start_date is None or end_date is None:
            raise ValueError(
                "Must provide either (data_a, data_b) or (start_date, end_date)"
            )
        data_a, data_b = fetch_pair_data(
            pair.asset_a, pair.asset_b, start_date, end_date, source
        )

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
