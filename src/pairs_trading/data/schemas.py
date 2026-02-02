"""Core data models for the pairs trading system."""

from typing import Optional

import pandas as pd
from pydantic import BaseModel, ConfigDict, field_validator


class Asset(BaseModel):
    """A single tradeable asset."""

    model_config = ConfigDict(frozen=True)

    symbol: str
    name: Optional[str] = None

    @field_validator("symbol")
    @classmethod
    def uppercase_symbol(cls, v: str) -> str:
        return v.upper().strip()

    def __hash__(self) -> int:
        return hash(self.symbol)


class Pair(BaseModel):
    """A pair of assets for pairs trading."""

    model_config = ConfigDict(frozen=True)

    asset_a: Asset
    asset_b: Asset

    @property
    def pair_id(self) -> str:
        return f"{self.asset_a.symbol}_{self.asset_b.symbol}"

    def __hash__(self) -> int:
        return hash(self.pair_id)


class PriceData(BaseModel):
    """Container for OHLCV price data."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    df: pd.DataFrame  # columns: open, high, low, close, volume
    symbol: str
    source: str = "yfinance"  # data provider name

    @field_validator("df")
    @classmethod
    def validate_df(cls, v: pd.DataFrame) -> pd.DataFrame:
        if v.empty:
            raise ValueError("DataFrame cannot be empty")
        required = {"open", "high", "low", "close", "volume"}
        if not required.issubset(set(v.columns)):
            raise ValueError(f"DataFrame must have columns: {required}")
        if not isinstance(v.index, pd.DatetimeIndex):
            raise ValueError("DataFrame index must be DatetimeIndex")
        return v

    @property
    def close(self) -> pd.Series:
        return self.df["close"]


class SpreadData(BaseModel):
    """Computed spread between a pair."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    pair: Pair
    spread: pd.Series
    z_score: pd.Series
    hedge_ratio: float
    half_life: Optional[float] = None

    @property
    def timestamps(self) -> pd.DatetimeIndex:
        return self.spread.index
