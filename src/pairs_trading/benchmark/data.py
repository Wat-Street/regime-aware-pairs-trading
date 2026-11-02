"""Price loading: yfinance (cached to CSV), local CSVs, or a synthetic regime-switching universe."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from pairs_trading.benchmark.config import DataConfig
from pairs_trading.data.schemas import Asset, Pair, PriceData

log = logging.getLogger(__name__)


def _to_price_data(close: pd.Series, symbol: str, source: str) -> PriceData:
    close = close.dropna().astype(float)
    close.index = pd.DatetimeIndex(close.index).tz_localize(None).normalize()
    close = close[~close.index.duplicated(keep="last")].sort_index()
    df = pd.DataFrame(
        {"open": close, "high": close, "low": close, "close": close, "volume": 0.0}
    )
    return PriceData(df=df, symbol=symbol, source=source)


def _load_yfinance(symbol: str, cfg: DataConfig) -> PriceData:
    cache = Path(cfg.cache_dir) / f"{symbol}_{cfg.start}_{cfg.end}.csv"
    if cache.exists():
        s = pd.read_csv(cache, index_col=0, parse_dates=True)["close"]
        return _to_price_data(s, symbol, "yfinance-cache")
    from pairs_trading.data.fetcher import fetch_price_data

    pdata = fetch_price_data(
        Asset(symbol=symbol),
        datetime.fromisoformat(cfg.start),
        datetime.fromisoformat(cfg.end),
        source="yfinance",
    )
    cache.parent.mkdir(parents=True, exist_ok=True)
    pdata.close.rename("close").to_csv(cache)
    return _to_price_data(pdata.close, symbol, "yfinance")


def _load_csv(symbol: str, cfg: DataConfig) -> PriceData:
    path = Path(cfg.csv_dir) / f"{symbol}.csv"
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    col = next(
        (c for c in df.columns if c.lower() in ("close", "adj close", "adj_close")),
        None,
    )
    if col is None:
        raise ValueError(f"{path}: needs a 'close' column")
    s = df[col].loc[cfg.start : cfg.end]
    return _to_price_data(s, symbol, "csv")


# ---------------------------------------------------------------------------
# Synthetic universe — offline smoke tests / CI, and a sanity check that the
# gates can find a regime signal when one provably exists.
# ---------------------------------------------------------------------------


def synthetic_pair(
    n: int, seed: int, start: str = "2015-01-01"
) -> Tuple[pd.Series, pd.Series, np.ndarray]:
    """Cointegrated pair whose spread switches between two hidden regimes.

    regime 0 (stable): fast OU reversion, low vol        -> trades mostly revert
    regime 1 (broken): near-random-walk spread, high vol -> trades mostly fail
    Regimes are persistent (Markov), so recent vol/half-life carry information.
    """
    rng = np.random.default_rng(seed)
    p_stay = np.array([0.985, 0.96])
    kappa = np.array([0.25, 0.01])
    vol = np.array([0.6, 1.4])
    regime = np.zeros(n, dtype=int)
    for t in range(1, n):
        stay = rng.random() < p_stay[regime[t - 1]]
        regime[t] = regime[t - 1] if stay else 1 - regime[t - 1]
    spread = np.zeros(n)
    for t in range(1, n):
        k, v = kappa[regime[t]], vol[regime[t]]
        spread[t] = spread[t - 1] - k * spread[t - 1] + v * rng.standard_normal()
    log_b = np.log(50) + np.cumsum(0.0002 + 0.012 * rng.standard_normal(n))
    b = np.exp(log_b)
    beta = rng.uniform(0.8, 1.6)
    a = 5.0 + beta * b + spread
    a = np.maximum(a, 1.0)
    idx = pd.bdate_range(start, periods=n)
    return pd.Series(a, idx), pd.Series(b, idx), regime


def load_universe(cfg: DataConfig) -> List[Tuple[Pair, PriceData, PriceData]]:
    """Return [(pair, data_a, data_b)] for every pair that loads; failures are logged and skipped."""
    out: List[Tuple[Pair, PriceData, PriceData]] = []

    if cfg.source == "synthetic":
        n = len(pd.bdate_range(cfg.start, cfg.end))
        for i in range(cfg.synthetic_pairs):
            a, b, _ = synthetic_pair(n, cfg.synthetic_seed + i, cfg.start)
            pair = Pair(
                asset_a=Asset(symbol=f"SYNA{i}"), asset_b=Asset(symbol=f"SYNB{i}")
            )
            out.append(
                (
                    pair,
                    _to_price_data(a, f"SYNA{i}", "synthetic"),
                    _to_price_data(b, f"SYNB{i}", "synthetic"),
                )
            )
        return out

    loader = {"yfinance": _load_yfinance, "csv": _load_csv}.get(cfg.source)
    if loader is None:
        raise ValueError(f"unknown data source {cfg.source!r}")
    cache: Dict[str, PriceData] = {}
    for sym_a, sym_b in cfg.resolve_pairs():
        try:
            for s in (sym_a, sym_b):
                if s not in cache:
                    cache[s] = loader(s, cfg)
            pair = Pair(asset_a=Asset(symbol=sym_a), asset_b=Asset(symbol=sym_b))
            out.append((pair, cache[sym_a], cache[sym_b]))
        except Exception as exc:  # network / missing file: skip the pair, keep going
            log.warning("skipping %s/%s: %s", sym_a, sym_b, exc)
    return out
