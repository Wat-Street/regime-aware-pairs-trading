"""Tunable hyperparameters for the signal layer.

Edit the values here to tune behavior. ``schemas.SignalConfig`` defines the shape;
this module holds the values and builds the default config.
"""

from pairs_trading.signals.schemas import SignalConfig

# --- entry / sizing ---
ENTRY_THRESHOLD = 2.0  # m   : enter when |Z| > m
CAPITAL_PER_TRADE = 10_000.0  # base notional per trade
MIN_OBSERVATIONS = 40  # min window for ARMA/GARCH fit

# --- dynamic exit logic ---
EXIT_COST_Z = 0.1  # c_z : transaction cost in z-score units
RISK_BUFFER = 0.1  # lambda : risk buffer
REVERSION_HORIZON = 5  # H   : horizon (days) for p_t / labels
ROLLING_LOOKBACK = 30  # rolling z-score window


def default_config() -> SignalConfig:
    """Build the default ``SignalConfig`` from the constants above."""
    return SignalConfig(
        entry_threshold=ENTRY_THRESHOLD,
        capital_per_trade=CAPITAL_PER_TRADE,
        min_observations=MIN_OBSERVATIONS,
        exit_cost_z=EXIT_COST_Z,
        risk_buffer=RISK_BUFFER,
        reversion_horizon=REVERSION_HORIZON,
        rolling_lookback=ROLLING_LOOKBACK,
    )
