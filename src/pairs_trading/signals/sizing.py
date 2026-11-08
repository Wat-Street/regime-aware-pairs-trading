"""Position sizing: bet smaller when volatility is high (inverse-vol)."""

from pairs_trading.signals.schemas import Side, Signal, SignalConfig

EPS = 1e-12


def size_position(signal: Signal, config: SignalConfig) -> tuple[float, float]:
    """Return ``(volume_a, volume_b)`` in shares, sized inversely to volatility.

    Assumes ``signal.price_a > 0``. ``volume_b`` is the opposite leg, scaled by the
    hedge ratio, so a negative hedge ratio yields same-direction legs.
    """
    notional = config.capital_per_trade / max(signal.sigma, EPS)
    # Future: multiply `notional` by a regime score here once the gate exists.
    k = notional / signal.price_a

    direction = 1.0 if signal.side == Side.LONG_A_SHORT_B else -1.0
    volume_a = direction * k
    volume_b = -direction * signal.hedge_ratio * k
    return volume_a, volume_b
