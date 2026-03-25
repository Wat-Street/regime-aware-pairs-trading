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
        df = ticker.history(start=start_date, end=end_date, auto_adjust=True)
        if df.empty:
            raise ValueError(f"No price history returned for {asset.symbol}")

        # Standardize column names to lowercase
        df.columns = [c.lower().replace(" ", "_") for c in df.columns]
        required_columns = {"open", "high", "low", "close", "volume"}
        missing_columns = required_columns.difference(df.columns)
        if missing_columns:
            raise ValueError(
                f"Provider data for {asset.symbol} is missing columns: "
                f"{sorted(missing_columns)}"
            )

        # Keep only OHLCV columns
        df = df[["open", "high", "low", "close", "volume"]].sort_index()

        return PriceData(df=df, symbol=asset.symbol, source="yfinance")
