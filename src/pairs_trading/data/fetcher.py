"""Fetch market data from Yahoo Finance."""
from datetime import datetime

import yfinance as yf

from .schemas import Asset, PriceData


def fetch_price_data(
    asset: Asset,
    start_date: datetime,
    end_date: datetime,
) -> PriceData:
    """
    Fetch OHLCV data for a single asset.
    
    Args:
        asset: The asset to fetch
        start_date: Start of date range
        end_date: End of date range
    
    Returns:
        PriceData with OHLCV DataFrame
    """
    ticker = yf.Ticker(asset.symbol)
    df = ticker.history(start=start_date, end=end_date, auto_adjust=False)
    
    # Standardize column names to lowercase
    df.columns = [c.lower().replace(" ", "_") for c in df.columns]
    
    # Keep only OHLCV columns
    df = df[["open", "high", "low", "close", "volume"]]
    
    return PriceData(df=df, symbol=asset.symbol)


def fetch_pair_data(
    asset_a: Asset,
    asset_b: Asset,
    start_date: datetime,
    end_date: datetime,
) -> tuple[PriceData, PriceData]:
    """
    Fetch OHLCV data for a pair of assets, aligned to common dates.
    
    Args:
        asset_a: First asset
        asset_b: Second asset
        start_date: Start of date range
        end_date: End of date range
    
    Returns:
        Tuple of (PriceData_A, PriceData_B) with aligned timestamps
    """
    data_a = fetch_price_data(asset_a, start_date, end_date)
    data_b = fetch_price_data(asset_b, start_date, end_date)
    
    # Align to common dates (inner join)
    common_idx = data_a.df.index.intersection(data_b.df.index)
    
    aligned_a = PriceData(
        df=data_a.df.loc[common_idx],
        symbol=asset_a.symbol,
    )
    aligned_b = PriceData(
        df=data_b.df.loc[common_idx],
        symbol=asset_b.symbol,
    )
    
    return aligned_a, aligned_b