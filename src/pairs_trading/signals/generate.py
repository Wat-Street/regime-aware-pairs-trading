"""Entry-signal generation for a pair at a single time step."""

from pairs_trading.data.schemas import Pair, PriceData
from pairs_trading.data.spread import compute_spread, fit_arma_garch
from pairs_trading.signals.params import default_config
from pairs_trading.signals.schemas import Side, Signal, SignalConfig


def generate_signal(
    pair: Pair,
    data_a: PriceData,
    data_b: PriceData,
    config: SignalConfig = default_config(),
) -> Signal:
    """Decide whether to enter a pair trade and snapshot the model state.

    Uses the volatility-scaled z-score ``Z = (s - mu) / sigma`` from the ARMA/GARCH fit.
    Enters only when the pair is cointegrated and ``|Z|`` exceeds ``entry_threshold``.
    """
    spread_data = compute_spread(pair, data_a, data_b)
    fit = fit_arma_garch(spread_data.spread)

    z = float(fit.vol_scaled_z_score.iloc[-1])
    sigma = float(fit.conditional_volatility.iloc[-1])

    should_enter = spread_data.cointegration.is_cointegrated and (
        abs(z) > config.entry_threshold
    )
    if not should_enter:
        side = Side.FLAT
    elif z > 0:
        # Spread is rich and expected to fall: short A, long B.
        side = Side.SHORT_A_LONG_B
    else:
        side = Side.LONG_A_SHORT_B

    return Signal(
        pair=pair,
        timestamp=spread_data.spread.index[-1],
        side=side,
        z_score=z,
        should_enter=should_enter,
        mu=fit.mu,
        hedge_ratio=spread_data.hedge_ratio,
        intercept=spread_data.intercept,
        sigma=sigma,
        price_a=float(data_a.close.iloc[-1]),
        price_b=float(data_b.close.iloc[-1]),
        half_life=spread_data.half_life,
        cointegration=spread_data.cointegration,
    )
