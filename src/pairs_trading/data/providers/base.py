"""Abstract base class for data providers."""

from abc import ABC, abstractmethod
from datetime import datetime

from ..schemas import Asset, PriceData


class DataProvider(ABC):
    """Abstract interface for market data providers."""

    @abstractmethod
    def fetch(
        self,
        asset: Asset,
        start_date: datetime,
        end_date: datetime,
    ) -> PriceData:
        """Fetch OHLCV data for a single asset."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider identifier."""
        pass
