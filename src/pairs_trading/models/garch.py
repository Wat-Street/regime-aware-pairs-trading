"""GARCH modeling utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
import pandas as pd
from arch import arch_model


MeanModel = Literal["Constant", "Zero", "AR"]
DistModel = Literal["normal", "t", "skewt", "ged"]


@dataclass(frozen=True)
class GarchSpec:
    """Specification for a GARCH model."""

    p: int = 1
    q: int = 1
    mean: MeanModel = "Zero"
    dist: DistModel = "t"
    lags: int = 0  # used when mean == "AR"


@dataclass
class GarchFitResult:
    """Results from fitting a GARCH model."""

    params: pd.Series
    volatility: pd.Series
    residuals: pd.Series
    fitted_values: Optional[pd.Series]
    model_summary: str


def validate_returns(returns: pd.Series) -> pd.Series:
    """Validate and clean a returns series for GARCH fitting."""
    if returns is None:
        raise ValueError("returns cannot be None")
    if not isinstance(returns, pd.Series):
        raise TypeError("returns must be a pandas Series")

    cleaned = returns.replace([np.inf, -np.inf], np.nan).dropna()
    if cleaned.empty:
        raise ValueError("returns cannot be empty after cleaning")
    return cleaned.astype(float)


def fit_garch(returns: pd.Series, spec: GarchSpec = GarchSpec()) -> GarchFitResult:
    """Fit a GARCH model to a returns series."""
    r = validate_returns(returns)

    model = arch_model(
        r,
        p=spec.p,
        q=spec.q,
        mean=spec.mean,
        lags=spec.lags if spec.mean == "AR" else 0,
        vol="GARCH",
        dist=spec.dist,
        rescale=True,
    )
    res = model.fit(disp="off")

    vol = res.conditional_volatility
    fitted = res.params.get("mu")
    fitted_series = None
    if fitted is not None:
        fitted_series = pd.Series(fitted, index=r.index)

    return GarchFitResult(
        params=res.params,
        volatility=vol,
        residuals=res.resid,
        fitted_values=fitted_series,
        model_summary=str(res.summary()),
    )


def forecast_volatility(result: GarchFitResult, horizon: int = 5) -> pd.Series:
    """Forecast volatility for a given horizon (in periods)."""
    if horizon <= 0:
        raise ValueError("horizon must be positive")

    # If you want multi-step forecasts, refit is required; keep simple here.
    last_vol = result.volatility.iloc[-1]
    idx = pd.RangeIndex(start=1, stop=horizon + 1, step=1)
    return pd.Series([last_vol] * horizon, index=idx, name="vol_forecast")
