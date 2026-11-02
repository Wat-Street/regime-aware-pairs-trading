"""Leak-free per-pair dataset: formation-fit spread -> canonical features -> labels -> windows.

Look-ahead protection
---------------------
``compute_spread`` fits the hedge ratio on the WHOLE sample and ``fit_arma_garch``
estimates GARCH params on the whole sample. Fine for research notebooks, but it
leaks future information into every backtest. Here both are estimated on the
formation period only (``<= formation_end``) and then applied forward with
frozen parameters. Everything after that (rolling z, features, labels) is causal.
Rows inside the formation period are flagged so the benchmark never scores on them.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Callable, Optional, Sequence

import numpy as np
import pandas as pd
from arch import arch_model
from statsmodels.api import OLS, add_constant
from statsmodels.tsa.arima.model import ARIMA

from pairs_trading.data.labels import (
    DEFAULT_HALF_LIFE_WINDOW,
    DEFAULT_MAX_HALF_LIFE,
    FEATURE_NAMES,
    build_feature_frame,
    generate_regime_labels,
)
from pairs_trading.data.schemas import Pair, PriceData, SpreadData
from pairs_trading.data.spread import (
    check_cointegration,
    compute_half_life,
    compute_zscore,
    fit_arma_garch,
)
from pairs_trading.gate.config import TradeRules

FrameTransform = Callable[[pd.DataFrame], pd.DataFrame]


@dataclass
class PairDataset:
    """Everything the benchmark needs for one pair, on one aligned date index."""

    pair_id: str
    features: pd.DataFrame  # canonical features, index = dates (warm-up rows dropped)
    labels: pd.Series  # int8 {-1 no signal, 0 bad, 1 good}, same index as features
    z: pd.Series  # rolling z used for entry/exit, full index
    spread: pd.Series  # frozen-beta spread, full index
    price_a: pd.Series
    price_b: pd.Series
    hedge_ratio: float
    intercept: float
    formation_end: pd.Timestamp
    cointegration_pvalue: float
    rules: TradeRules

    @property
    def n_features(self) -> int:
        return self.features.shape[1]


@dataclass
class WindowSet:
    """Model-ready windows, possibly pooled across pairs.

    Row i is the decision at ``dates[i]`` for ``pair_ids[i]``; ``x[i]`` holds the
    ``input_length`` feature rows ending at (and including) that date.
    """

    x: np.ndarray  # float32 (N, L, F)
    labels: np.ndarray  # int8 (N,)  -1 / 0 / 1
    dates: np.ndarray  # datetime64 (N,)
    pair_ids: np.ndarray  # object (N,)
    feature_names: tuple

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def subset(self, mask: np.ndarray) -> "WindowSet":
        return WindowSet(
            self.x[mask],
            self.labels[mask],
            self.dates[mask],
            self.pair_ids[mask],
            self.feature_names,
        )

    @staticmethod
    def concat(sets: Sequence["WindowSet"]) -> "WindowSet":
        sets = [s for s in sets if len(s)]
        if not sets:
            raise ValueError("no windows to concatenate")
        names = sets[0].feature_names
        if any(s.feature_names != names for s in sets):
            raise ValueError("feature layouts differ across pairs")
        out = WindowSet(
            np.concatenate([s.x for s in sets]),
            np.concatenate([s.labels for s in sets]),
            np.concatenate([s.dates for s in sets]),
            np.concatenate([s.pair_ids for s in sets]),
            names,
        )
        order = np.lexsort((out.pair_ids.astype(str), out.dates))
        return out.subset(order)


# ---------------------------------------------------------------------------
# Formation-period estimation, applied forward with frozen parameters
# ---------------------------------------------------------------------------


def causal_arma_garch(
    spread: pd.Series, formation_end: pd.Timestamp
) -> SimpleNamespace:
    """Fit ARMA/GARCH on the formation slice, then filter the full series with frozen params.

    Returns an object exposing ``conditional_volatility`` and ``arma_residuals``
    (the only fields ``build_feature_frame`` reads).
    """
    s = spread.dropna()
    form = s.loc[:formation_end]
    fit = fit_arma_garch(form)  # model selection + params on formation only

    arma = ARIMA(form, order=fit.arma_order, trend="c").fit()
    resid_full = pd.Series(arma.apply(s).resid, index=s.index)

    g = arch_model(
        resid_full,
        mean="Zero",
        vol="GARCH",
        p=fit.garch_order[0],
        q=fit.garch_order[1],
        dist="t",
        rescale=False,
    )
    names = g.volatility.parameter_names() + g.distribution.parameter_names()
    params = pd.Series(fit.garch_params)[names]
    garch = g.fix(params.to_numpy())
    cond_vol = pd.Series(np.asarray(garch.conditional_volatility), index=s.index)
    return SimpleNamespace(conditional_volatility=cond_vol, arma_residuals=resid_full)


def build_pair_dataset(
    pair: Pair,
    data_a: PriceData,
    data_b: PriceData,
    *,
    rules: TradeRules,
    formation_end: str | pd.Timestamp,
    features: Sequence[str] = FEATURE_NAMES,
    half_life_window: int = DEFAULT_HALF_LIFE_WINDOW,
    max_half_life: int = DEFAULT_MAX_HALF_LIFE,
) -> PairDataset:
    """Build one pair's leak-free features + labels."""
    formation_end = pd.Timestamp(formation_end)
    prices = pd.concat(
        {"a": data_a.close, "b": data_b.close}, axis=1, join="inner"
    ).dropna()
    prices = prices[~prices.index.duplicated()].sort_index()
    form = prices.loc[:formation_end]
    if len(form) < 120:
        raise ValueError(
            f"{pair.pair_id}: formation period has {len(form)} rows (< 120)"
        )
    if len(prices) - len(form) < 120:
        raise ValueError(f"{pair.pair_id}: fewer than 120 rows after formation_end")

    ols = OLS(form["a"], add_constant(form["b"])).fit()
    intercept, beta = float(ols.params["const"]), float(ols.params["b"])
    spread = (prices["a"] - intercept - beta * prices["b"]).rename("spread")
    z = compute_zscore(spread, lookback=rules.zscore_lookback).rename("z")
    coint = check_cointegration(form["a"], form["b"])

    spread_data = SpreadData(
        pair=pair,
        spread=spread,
        z_score=z,
        intercept=intercept,
        hedge_ratio=beta,
        cointegration=coint,
        half_life=float(compute_half_life(spread.loc[:formation_end])),
    )
    arma_like = causal_arma_garch(spread, formation_end)

    feats = build_feature_frame(
        spread_data,
        arma_like,
        features=features,
        half_life_window=half_life_window,
        max_half_life=max_half_life,
    ).astype(np.float32)

    labels = generate_regime_labels(
        spread_data,
        forward_window=rules.horizon,
        entry_threshold=rules.entry_z,
        reversion_target=rules.target_z,
        stop_loss=rules.stop_z,
        stop_rule="first_touch",
    )
    labels = labels.reindex(feats.index).fillna(-1).astype(np.int8)

    return PairDataset(
        pair_id=pair.pair_id,
        features=feats,
        labels=labels,
        z=z,
        spread=spread,
        price_a=prices["a"],
        price_b=prices["b"],
        hedge_ratio=beta,
        intercept=intercept,
        formation_end=formation_end,
        cointegration_pvalue=float(coint.p_value),
        rules=rules,
    )


def make_windows(
    ds: PairDataset,
    input_length: int,
    transform: Optional[FrameTransform] = None,
    select: str = "labeled",
) -> WindowSet:
    """Slide ``input_length`` windows over the (optionally transformed) feature frame.

    ``transform`` is a model's causal representation hook (e.g. wavelet bands);
    it sees the whole frame but must only use past values at each row.

    ``select``:
      "labeled" decision dates with a known label (0/1) -> training / metrics
      "signal"  every date with |z| >= entry_z, including the last H days whose
                outcome is still unknown (label -1) -> backtest inference
      "all"     every date
    """
    if select not in ("labeled", "signal", "all"):
        raise ValueError("select must be 'labeled', 'signal' or 'all'")
    frame = transform(ds.features) if transform is not None else ds.features
    values = frame.to_numpy(dtype=np.float32)
    n = len(values)
    if n < input_length:
        raise ValueError(
            f"{ds.pair_id}: {n} feature rows < input_length {input_length}"
        )

    ends = np.arange(input_length - 1, n)
    labels = ds.labels.reindex(frame.index).to_numpy(dtype=np.int8)[ends]
    if select == "labeled":
        keep = labels >= 0
    elif select == "signal":
        zabs = ds.z.reindex(frame.index).abs().to_numpy()[ends]
        keep = zabs >= ds.rules.entry_z
    else:
        keep = np.ones(len(ends), dtype=bool)
    ends, labels = ends[keep], labels[keep]

    idx = ends[:, None] + np.arange(-input_length + 1, 1)[None, :]
    x = (
        values[idx]
        if len(ends)
        else np.zeros((0, input_length, values.shape[1]), np.float32)
    )
    dates = frame.index.to_numpy()[ends]
    return WindowSet(
        x=x,
        labels=labels,
        dates=dates,
        pair_ids=np.full(len(ends), ds.pair_id, dtype=object),
        feature_names=tuple(frame.columns),
    )
