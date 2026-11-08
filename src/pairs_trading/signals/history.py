"""Append-only per-day feature store feeding the reversion-probability estimator."""

import pandas as pd

from pairs_trading.signals.schemas import FeatureRow


class FeatureStore:
    """Stores one ``FeatureRow`` per day in arrival order."""

    def __init__(self) -> None:
        self._rows: list[FeatureRow] = []

    def append(self, row: FeatureRow) -> None:
        self._rows.append(row)

    def recent(self, n: int) -> pd.DataFrame:
        """Return the last ``n`` rows (oldest->newest); fewer if not enough stored."""
        rows = self._rows[-n:] if n > 0 else []
        return pd.DataFrame(
            [
                {
                    "timestamp": row.timestamp,
                    "rolling_z": row.rolling_z,
                    "volatility": row.volatility,
                    "kappa": row.kappa,
                }
                for row in rows
            ],
            columns=["timestamp", "rolling_z", "volatility", "kappa"],
        )
