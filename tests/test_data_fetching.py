"""Tests for data schemas, providers, and pair fetching."""

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from pairs_trading.data import fetcher
from pairs_trading.data.fetcher import fetch_pair_data
from pairs_trading.data.providers.yfinance import YFinanceProvider
from pairs_trading.data.schemas import Asset, PriceData


def _make_ohlcv_frame(index: pd.DatetimeIndex) -> pd.DataFrame:
    values = np.arange(len(index), dtype=float) + 100.0
    return pd.DataFrame(
        {
            "open": values,
            "high": values + 1.0,
            "low": values - 1.0,
            "close": values,
            "volume": np.full(len(index), 1_000),
        },
        index=index,
    )


def test_price_data_requires_sorted_index():
    index = pd.to_datetime(["2023-01-02", "2023-01-01", "2023-01-03"])
    with pytest.raises(ValueError, match="sorted"):
        PriceData(df=_make_ohlcv_frame(index), symbol="MSFT")


def test_price_data_requires_unique_index():
    index = pd.to_datetime(["2023-01-01", "2023-01-01", "2023-01-02"])
    with pytest.raises(ValueError, match="unique"):
        PriceData(df=_make_ohlcv_frame(index), symbol="MSFT")


def test_fetch_pair_data_rejects_empty_overlap(monkeypatch: pytest.MonkeyPatch):
    data_a = PriceData(
        df=_make_ohlcv_frame(pd.date_range("2023-01-01", periods=5, freq="D")),
        symbol="MSFT",
    )
    data_b = PriceData(
        df=_make_ohlcv_frame(pd.date_range("2023-02-01", periods=5, freq="D")),
        symbol="AAPL",
    )

    def fake_fetch_price_data(
        asset: Asset,
        start_date: datetime,
        end_date: datetime,
        source: str = "yfinance",
    ) -> PriceData:
        return data_a if asset.symbol == "MSFT" else data_b

    monkeypatch.setattr(fetcher, "fetch_price_data", fake_fetch_price_data)

    with pytest.raises(ValueError, match="No overlapping timestamps"):
        fetch_pair_data(
            Asset(symbol="MSFT"),
            Asset(symbol="AAPL"),
            datetime(2023, 1, 1),
            datetime(2023, 3, 1),
        )


def test_yfinance_provider_uses_adjusted_sorted_history(
    monkeypatch: pytest.MonkeyPatch,
):
    captured: dict[str, object] = {}
    unsorted_index = pd.to_datetime(["2023-01-03", "2023-01-01", "2023-01-02"])

    class FakeTicker:
        def __init__(self, symbol: str):
            captured["symbol"] = symbol

        def history(self, **kwargs: object) -> pd.DataFrame:
            captured.update(kwargs)
            return pd.DataFrame(
                {
                    "Open": [103.0, 101.0, 102.0],
                    "High": [104.0, 102.0, 103.0],
                    "Low": [102.0, 100.0, 101.0],
                    "Close": [103.0, 101.0, 102.0],
                    "Volume": [1_100, 1_000, 1_050],
                },
                index=unsorted_index,
            )

    monkeypatch.setattr("pairs_trading.data.providers.yfinance.yf.Ticker", FakeTicker)

    provider = YFinanceProvider()
    result = provider.fetch(
        Asset(symbol="MSFT"),
        datetime(2023, 1, 1),
        datetime(2023, 1, 4),
    )

    assert captured["symbol"] == "MSFT"
    assert captured["auto_adjust"] is True
    assert result.df.index.is_monotonic_increasing
    assert result.df.index.equals(pd.date_range("2023-01-01", periods=3, freq="D"))


def test_yfinance_provider_rejects_empty_history(monkeypatch: pytest.MonkeyPatch):
    class FakeTicker:
        def __init__(self, symbol: str):
            self.symbol = symbol

        def history(self, **kwargs: object) -> pd.DataFrame:
            return pd.DataFrame()

    monkeypatch.setattr("pairs_trading.data.providers.yfinance.yf.Ticker", FakeTicker)

    with pytest.raises(ValueError, match="No price history returned"):
        YFinanceProvider().fetch(
            Asset(symbol="MSFT"),
            datetime(2023, 1, 1),
            datetime(2023, 1, 4),
        )


def test_yfinance_provider_rejects_missing_columns(monkeypatch: pytest.MonkeyPatch):
    class FakeTicker:
        def __init__(self, symbol: str):
            self.symbol = symbol

        def history(self, **kwargs: object) -> pd.DataFrame:
            index = pd.date_range("2023-01-01", periods=3, freq="D")
            return pd.DataFrame(
                {
                    "Open": [1.0, 2.0, 3.0],
                    "High": [2.0, 3.0, 4.0],
                    "Low": [0.5, 1.5, 2.5],
                    "Close": [1.5, 2.5, 3.5],
                },
                index=index,
            )

    monkeypatch.setattr("pairs_trading.data.providers.yfinance.yf.Ticker", FakeTicker)

    with pytest.raises(ValueError, match="missing columns"):
        YFinanceProvider().fetch(
            Asset(symbol="MSFT"),
            datetime(2023, 1, 1),
            datetime(2023, 1, 4),
        )
