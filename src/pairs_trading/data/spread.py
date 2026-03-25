"""Compute spread, hedge ratio, and z-score for pairs."""

from datetime import datetime
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import coint

from pairs_trading.data.fetcher import fetch_pair_data
from pairs_trading.data.schemas import (
    ArmaGarchResult,
    CointegrationResult,
    Pair,
    PriceData,
    SpreadData,
)


def _align_price_series(
    price_a: pd.Series,
    price_b: pd.Series,
    *,
    min_observations: int,
    context: str,
) -> tuple[pd.Series, pd.Series]:
    """Align two price series on their shared non-null timestamps."""
    aligned = pd.concat(
        [
            price_a.rename("price_a").astype(float),
            price_b.rename("price_b").astype(float),
        ],
        axis=1,
        join="inner",
    ).dropna()
    aligned = aligned.sort_index()

    if len(aligned) < min_observations:
        raise ValueError(
            f"{context} requires at least {min_observations} overlapping observations"
        )

    return aligned["price_a"], aligned["price_b"]


def _fit_spread_regression(
    price_a: pd.Series,
    price_b: pd.Series,
    *,
    min_observations: int = 2,
) -> tuple[pd.Series, pd.Series, float, float, pd.Series]:
    """Fit OLS with intercept and return aligned series plus residual spread."""
    aligned_a, aligned_b = _align_price_series(
        price_a,
        price_b,
        min_observations=min_observations,
        context="Spread regression",
    )
    model = OLS(aligned_a, add_constant(aligned_b)).fit()
    intercept = float(model.params["const"])
    hedge_ratio = float(model.params["price_b"])
    spread = aligned_a - (intercept + hedge_ratio * aligned_b)
    return aligned_a, aligned_b, intercept, hedge_ratio, spread


def _cointegration_from_aligned_prices(
    price_a: pd.Series,
    price_b: pd.Series,
    *,
    alpha: float,
    trend: str,
) -> CointegrationResult:
    """Run Engle-Granger cointegration on already aligned price series."""
    test_statistic, p_value, critical_values = coint(
        price_a,
        price_b,
        trend=trend,
        autolag="aic",
    )
    critical_value_labels = ("1%", "5%", "10%")

    return CointegrationResult(
        test_statistic=float(test_statistic),
        p_value=float(p_value),
        critical_values={
            label: float(value)
            for label, value in zip(critical_value_labels, critical_values, strict=True)
        },
        is_cointegrated=bool(p_value < alpha),
        alpha=alpha,
        trend=trend,
    )


def _validate_pair_inputs(pair: Pair, data_a: PriceData, data_b: PriceData) -> None:
    """Validate that provided data belongs to the requested pair."""
    if data_a.symbol != pair.asset_a.symbol:
        raise ValueError(
            f"PriceData symbol mismatch for asset_a: expected {pair.asset_a.symbol}, "
            f"got {data_a.symbol}"
        )
    if data_b.symbol != pair.asset_b.symbol:
        raise ValueError(
            f"PriceData symbol mismatch for asset_b: expected {pair.asset_b.symbol}, "
            f"got {data_b.symbol}"
        )


def compute_hedge_ratio(price_a: pd.Series, price_b: pd.Series) -> float:
    """
    Compute hedge ratio using OLS regression.

    price_a = alpha + beta * price_b + error

    Returns beta (the hedge ratio).
    """
    _, _, _, hedge_ratio, _ = _fit_spread_regression(price_a, price_b)
    return hedge_ratio


def compute_half_life(spread: pd.Series) -> float:
    """
    Compute mean reversion half-life using AR(1) regression.

    spread_t - spread_{t-1} = rho * spread_{t-1} + error
    half_life = -log(2) / log(1 + rho)

    Returns half-life in number of periods (days).
    """
    spread_clean = spread.dropna().astype(float)
    if len(spread_clean) < 3:
        raise ValueError("Spread series must have at least 3 non-null observations")

    regression_frame = pd.DataFrame(
        {
            "spread_diff": spread_clean.diff(),
            "spread_lag": spread_clean.shift(1),
        }
    ).dropna()

    model = OLS(
        regression_frame["spread_diff"],
        add_constant(regression_frame["spread_lag"]),
    ).fit()
    rho = float(model.params["spread_lag"])

    if not np.isfinite(rho) or rho >= 0 or rho <= -1:
        return np.inf

    return float(-np.log(2) / np.log1p(rho))


def compute_zscore(spread: pd.Series, lookback: int = 20) -> pd.Series:
    """
    Compute rolling z-score of spread.

    z = (spread - rolling_mean) / rolling_std
    """
    if lookback < 2:
        raise ValueError("lookback must be at least 2")

    mean = spread.rolling(window=lookback).mean()
    std = spread.rolling(window=lookback).std()

    z_score = (spread - mean) / std
    return z_score


def check_cointegration(
    price_a: pd.Series,
    price_b: pd.Series,
    *,
    alpha: float = 0.05,
    trend: str = "c",
) -> CointegrationResult:
    """
    Test pair cointegration via the Engle-Granger procedure.

    If p-value < alpha, treat the pair as cointegrated.
    """
    aligned_a, aligned_b = _align_price_series(
        price_a,
        price_b,
        min_observations=20,
        context="Cointegration test",
    )
    return _cointegration_from_aligned_prices(
        aligned_a,
        aligned_b,
        alpha=alpha,
        trend=trend,
    )


def _iter_model_candidates(
    arma_order: tuple[int, int, int],
    garch_order: tuple[int, int],
    arma_candidates: Optional[Iterable[tuple[int, int, int]]] = None,
    garch_candidates: Optional[Iterable[tuple[int, int]]] = None,
) -> list[tuple[tuple[int, int, int], tuple[int, int]]]:
    """Build the ARMA/GARCH candidate grid, preserving caller order."""
    arma_grid = list(arma_candidates) if arma_candidates is not None else [arma_order]
    garch_grid = (
        list(garch_candidates) if garch_candidates is not None else [garch_order]
    )
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
    cointegration_alpha: float = 0.05,
    cointegration_trend: str = "c",
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
        cointegration_alpha: Significance level for Engle-Granger test
        cointegration_trend: Deterministic trend used in the cointegration test

    Returns:
        SpreadData with residual spread, z-score, hedge ratio, intercept,
        cointegration result, and half-life
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
    else:
        _validate_pair_inputs(pair, data_a, data_b)

    aligned_a, aligned_b, intercept, hedge_ratio, spread = _fit_spread_regression(
        data_a.close,
        data_b.close,
        min_observations=20,
    )
    cointegration = _cointegration_from_aligned_prices(
        aligned_a,
        aligned_b,
        alpha=cointegration_alpha,
        trend=cointegration_trend,
    )

    z_score = compute_zscore(spread, lookback=zscore_lookback)
    half_life = compute_half_life(spread) if cointegration.is_cointegrated else None

    return SpreadData(
        pair=pair,
        spread=spread,
        z_score=z_score,
        intercept=intercept,
        hedge_ratio=hedge_ratio,
        cointegration=cointegration,
        half_life=half_life,
    )
