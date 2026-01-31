"""Data source interfaces and registry."""

from __future__ import annotations

from abc import ABC, abstractmethod


class DataSource(ABC):
    """Base interface for market data sources."""

    @abstractmethod
    def get_data(self) -> None:
        """Fetch or initialize data from the source."""
        raise NotImplementedError

class YahooFinance(DataSource):
    def get_data(self) -> None:
        # Placeholder implementation
        return None

data_source_mapping = {
    "yahoo_finance": YahooFinance,
}

def get_data_source(source_name: str) -> DataSource:
    source_class = data_source_mapping.get(source_name.lower())
    if not source_class:
        raise ValueError(f"Unknown data source: {source_name}")
    return source_class()

def use_data_source(source: DataSource) -> None:
    source.get_data()
    # Placeholder function to demonstrate usage of data source
    print(f"Using data source: {type(source).__name__}")
