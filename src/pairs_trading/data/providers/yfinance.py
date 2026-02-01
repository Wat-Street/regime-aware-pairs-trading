"""YFinance data provider."""

from datetime import datetime

import yfinance as yf

from .base import DataProvider
from ..schemas import Asset, PriceData


class YFinanceProvider(DataProvider):
    """Data provider using Yahoo Finance API."""

    @property
    def name(self) -> str:
        return "yfinance"

    def fetch(
        self,
        asset: Asset,
        start_date: datetime,
        end_date: datetime,
    ) -> PriceData:
        """Fetch OHLCV data for a single asset."""
        ticker = yf.Ticker(asset.symbol)
        df = ticker.history(start=start_date, end=end_date, auto_adjust=False)

        # Standardize column names to lowercase
        df.columns = [c.lower().replace(" ", "_") for c in df.columns]

        # Keep only OHLCV columns
        df = df[["open", "high", "low", "close", "volume"]]

        return PriceData(df=df, symbol=asset.symbol, source="yfinance")
