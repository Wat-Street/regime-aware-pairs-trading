"""Open a position and evaluate its frozen spread while held."""

from pairs_trading.signals.schemas import Position, Signal, SignalConfig
from pairs_trading.signals.sizing import size_position


def open_position(signal: Signal, config: SignalConfig) -> Position:
    """Freeze the spread definition at entry and size the trade.

    Only ``hedge_ratio`` and ``intercept`` are frozen for the life of the trade.
    ``mu``/``sigma`` are kept for reference/PnL but are not the exit trigger.
    """
    volume_a, volume_b = size_position(signal, config)
    return Position(
        pair=signal.pair,
        side=signal.side,
        entry_time=signal.timestamp,
        hedge_ratio=signal.hedge_ratio,
        intercept=signal.intercept,
        mu=signal.mu,
        sigma=signal.sigma,
        entry_z=signal.z_score,
        entry_price_a=signal.price_a,
        entry_price_b=signal.price_b,
        volume_a=volume_a,
        volume_b=volume_b,
    )


def live_spread(position: Position, price_a: float, price_b: float) -> float:
    """Spread under the frozen entry-time hedge ratio and intercept."""
    return price_a - (position.intercept + position.hedge_ratio * price_b)
