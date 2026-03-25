"""Fetch market data from providers."""

from datetime import datetime

from .providers.base import DataProvider
from .providers.yfinance import YFinanceProvider
from .schemas import Asset, PriceData


def get_provider(source: str) -> DataProvider:
    """Get the appropriate data provider."""
    providers = {
        "yfinance": YFinanceProvider,
        # Add more here later:
        # "alpha_vantage": AlphaVantageProvider,
        # "binance": BinanceProvider,
    }

    if source not in providers:
        raise ValueError(
            f"Unsupported data source: {source}. Available: {list(providers.keys())}"
        )

    return providers[source]()


def fetch_price_data(
    asset: Asset,
    start_date: datetime,
    end_date: datetime,
    source: str = "yfinance",
) -> PriceData:
    """
    Fetch OHLCV data for a single asset.

    Args:
        asset: The asset to fetch
        start_date: Start of date range
        end_date: End of date range
        source: Data provider ("yfinance", "alpha_vantage", "binance")

    Returns:
        PriceData with OHLCV DataFrame
    """
    provider = get_provider(source)
    return provider.fetch(asset, start_date, end_date)


def fetch_pair_data(
    asset_a: Asset,
    asset_b: Asset,
    start_date: datetime,
    end_date: datetime,
    source: str = "yfinance",
) -> tuple[PriceData, PriceData]:
    """
    Fetch OHLCV data for a pair of assets, aligned to common dates.

    Args:
        asset_a: First asset
        asset_b: Second asset
        start_date: Start of date range
        end_date: End of date range
        source: Data provider ("yfinance", "alpha_vantage", "binance")

    Returns:
        Tuple of (PriceData_A, PriceData_B) with aligned timestamps
    """
    data_a = fetch_price_data(asset_a, start_date, end_date, source)
    data_b = fetch_price_data(asset_b, start_date, end_date, source)

    # Align to common dates (inner join)
    common_idx = data_a.df.index.intersection(data_b.df.index).sort_values()
    if len(common_idx) == 0:
        raise ValueError(
            f"No overlapping timestamps found for {asset_a.symbol} and {asset_b.symbol}"
        )

    aligned_a = PriceData(
        df=data_a.df.loc[common_idx],
        symbol=asset_a.symbol,
        source=data_a.source,
    )
    aligned_b = PriceData(
        df=data_b.df.loc[common_idx],
        symbol=asset_b.symbol,
        source=data_b.source,
    )

    return aligned_a, aligned_b
