"""Unit tests for the signal-generation layer."""

import numpy as np
import pandas as pd

from pairs_trading.data.schemas import Asset, CointegrationResult, Pair, PriceData
from pairs_trading.signals.generate import generate_signal
from pairs_trading.signals.exit import dynamic_exit_threshold, should_exit
from pairs_trading.signals.history import FeatureStore
from pairs_trading.signals.params import default_config
from pairs_trading.signals.position import live_spread, open_position
from pairs_trading.signals.reversion import BaselineReversionEstimator
from pairs_trading.signals.schemas import FeatureRow, Side, Signal
from pairs_trading.signals.sizing import size_position


def _make_price_data(
    symbol: str,
    close: np.ndarray,
    *,
    index: pd.DatetimeIndex | None = None,
) -> PriceData:
    if index is None:
        index = pd.date_range("2023-01-01", periods=len(close), freq="D")

    close_array = np.asarray(close, dtype=float)
    df = pd.DataFrame(
        {
            "open": close_array,
            "high": close_array + 0.5,
            "low": close_array - 0.5,
            "close": close_array,
            "volume": np.full(len(close_array), 1_000),
        },
        index=index,
    )
    return PriceData(df=df, symbol=symbol)


def _cointegrated_pair(
    *,
    n: int = 250,
    seed: int = 42,
    beta: float = 1.3,
    intercept: float = 5.0,
    noise_scale: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a cointegrated (price_a, price_b) pair with AR(1) stationary residuals."""
    rng = np.random.default_rng(seed)
    base = np.cumsum(rng.normal(0, 1, n)) + 100.0
    noise = np.zeros(n)
    for t in range(1, n):
        noise[t] = 0.6 * noise[t - 1] + rng.normal(0, noise_scale)
    price_b = base
    price_a = intercept + beta * base + noise
    return price_a, price_b


def _pair() -> Pair:
    return Pair(asset_a=Asset(symbol="AAA"), asset_b=Asset(symbol="BBB"))


def _make_signal(
    *,
    sigma: float,
    hedge_ratio: float,
    side: Side,
    price_a: float = 100.0,
    price_b: float = 80.0,
) -> Signal:
    return Signal(
        pair=_pair(),
        timestamp=pd.Timestamp("2023-01-01"),
        side=side,
        z_score=3.0,
        should_enter=True,
        mu=0.0,
        hedge_ratio=hedge_ratio,
        intercept=0.0,
        sigma=sigma,
        price_a=price_a,
        price_b=price_b,
        half_life=10.0,
        cointegration=CointegrationResult(
            test_statistic=-4.0,
            p_value=0.01,
            critical_values={"1%": -3.9, "5%": -3.3, "10%": -3.0},
            is_cointegrated=True,
        ),
    )


def test_generate_signal_enters_on_extreme_spread():
    price_a, price_b = _cointegrated_pair()
    price_a[-1] += 6.0  # spike spread up -> z > 0 -> short A / long B

    signal = generate_signal(
        _pair(),
        _make_price_data("AAA", price_a),
        _make_price_data("BBB", price_b),
    )

    assert signal.cointegration.is_cointegrated
    assert signal.should_enter
    assert signal.z_score > 0
    assert signal.side == Side.SHORT_A_LONG_B


def test_generate_signal_enters_short_b_on_negative_extreme():
    price_a, price_b = _cointegrated_pair(seed=3)
    price_a[-1] -= 6.0  # spike spread down -> z < 0 -> long A / short B

    signal = generate_signal(
        _pair(),
        _make_price_data("AAA", price_a),
        _make_price_data("BBB", price_b),
    )

    assert signal.should_enter
    assert signal.z_score < 0
    assert signal.side == Side.LONG_A_SHORT_B


def test_generate_signal_no_entry_when_threshold_not_exceeded():
    price_a, price_b = _cointegrated_pair(seed=7)
    config = default_config().model_copy(update={"entry_threshold": 100.0})

    signal = generate_signal(
        _pair(),
        _make_price_data("AAA", price_a),
        _make_price_data("BBB", price_b),
        config,
    )

    assert not signal.should_enter
    assert signal.side == Side.FLAT


def test_generate_signal_no_entry_when_not_cointegrated():
    rng = np.random.default_rng(11)
    n = 250
    price_a = np.cumsum(rng.normal(0.1, 1.0, n)) + 100.0
    price_b = np.cumsum(rng.normal(-0.05, 1.2, n)) + 100.0

    signal = generate_signal(
        _pair(),
        _make_price_data("AAA", price_a),
        _make_price_data("BBB", price_b),
    )

    assert not signal.cointegration.is_cointegrated
    assert not signal.should_enter
    assert signal.side == Side.FLAT


def test_size_position_inverse_vol():
    config = default_config()
    low_vol = _make_signal(sigma=0.5, hedge_ratio=1.5, side=Side.LONG_A_SHORT_B)
    high_vol = _make_signal(sigma=2.0, hedge_ratio=1.5, side=Side.LONG_A_SHORT_B)

    vol_a_low, _ = size_position(low_vol, config)
    vol_a_high, _ = size_position(high_vol, config)

    assert abs(vol_a_high) < abs(vol_a_low)


def test_size_position_positive_hedge_ratio_gives_opposite_signs():
    config = default_config()
    for side in (Side.LONG_A_SHORT_B, Side.SHORT_A_LONG_B):
        volume_a, volume_b = size_position(
            _make_signal(sigma=1.0, hedge_ratio=1.5, side=side), config
        )
        assert volume_a != 0.0 and volume_b != 0.0
        assert (volume_a > 0) != (volume_b > 0)  # opposite signs


def test_size_position_negative_hedge_ratio_gives_same_signs():
    config = default_config()
    for side in (Side.LONG_A_SHORT_B, Side.SHORT_A_LONG_B):
        volume_a, volume_b = size_position(
            _make_signal(sigma=1.0, hedge_ratio=-1.5, side=side), config
        )
        assert volume_a != 0.0 and volume_b != 0.0
        assert (volume_a > 0) == (volume_b > 0)  # same sign


def test_open_position_freezes_spread_definition():
    config = default_config()
    signal = _make_signal(
        sigma=1.0,
        hedge_ratio=1.5,
        side=Side.SHORT_A_LONG_B,
        price_a=100.0,
        price_b=80.0,
    )
    position = open_position(signal, config)

    assert position.hedge_ratio == signal.hedge_ratio
    assert position.intercept == signal.intercept

    # Evaluating the live spread at new prices must not mutate the frozen definition.
    new_price_a, new_price_b = 110.0, 75.0
    result = live_spread(position, new_price_a, new_price_b)
    expected = new_price_a - (position.intercept + position.hedge_ratio * new_price_b)

    assert result == expected
    assert position.hedge_ratio == signal.hedge_ratio
    assert position.intercept == signal.intercept


def _feature_row(kappa: float, *, day: int = 1) -> FeatureRow:
    return FeatureRow(
        timestamp=pd.Timestamp("2023-01-01") + pd.Timedelta(days=day),
        rolling_z=0.0,
        volatility=1.0,
        kappa=kappa,
    )


def test_baseline_reversion_probability_bounds_and_monotonicity():
    estimator = BaselineReversionEstimator()
    horizon = 5

    p_slow = estimator.probability(pd.DataFrame({"kappa": [0.05]}), horizon)
    p_fast = estimator.probability(pd.DataFrame({"kappa": [0.5]}), horizon)

    assert 0.0 < p_slow < 1.0
    assert 0.0 < p_fast < 1.0
    assert p_fast > p_slow


def test_feature_store_recent_returns_last_n_oldest_to_newest():
    store = FeatureStore()
    for day in range(1, 6):
        store.append(_feature_row(kappa=float(day), day=day))

    recent = store.recent(3)

    assert list(recent.columns) == ["timestamp", "rolling_z", "volatility", "kappa"]
    assert recent["kappa"].tolist() == [3.0, 4.0, 5.0]


def test_feature_store_recent_handles_fewer_than_n_rows():
    store = FeatureStore()
    store.append(_feature_row(kappa=1.0, day=1))
    store.append(_feature_row(kappa=2.0, day=2))

    recent = store.recent(10)

    assert recent["kappa"].tolist() == [1.0, 2.0]


def _mean_reverting_spread(
    *,
    n: int = 250,
    phi: float = 0.7,
    theta: float = 0.4,
    seed: int = 0,
    noise: float = 1.0,
) -> pd.Series:
    """ARMA(1,1) spread; well-specified so the ARMA refit recovers phi in (0, 1)."""
    rng = np.random.default_rng(seed)
    s = np.zeros(n)
    e = np.zeros(n)
    for t in range(1, n):
        e[t] = rng.normal(0, noise)
        s[t] = phi * s[t - 1] + e[t] + theta * e[t - 1]
    return pd.Series(s, index=pd.date_range("2023-01-01", periods=n, freq="D"))


def _open_dummy_position():
    config = default_config()
    return open_position(
        _make_signal(sigma=1.0, hedge_ratio=1.5, side=Side.SHORT_A_LONG_B), config
    )


def test_dynamic_exit_threshold_monotonicity():
    base = default_config()

    t0 = dynamic_exit_threshold(0.5, 0.5, base)
    t_cost = dynamic_exit_threshold(
        0.5, 0.5, base.model_copy(update={"exit_cost_z": base.exit_cost_z + 0.1})
    )
    t_buffer = dynamic_exit_threshold(
        0.5, 0.5, base.model_copy(update={"risk_buffer": base.risk_buffer + 0.1})
    )

    assert t_cost > t0  # higher cost -> wider band -> exit sooner
    assert t_buffer > t0  # higher risk buffer -> wider band

    # higher p_t * kappa_t -> narrower band -> hold longer
    assert dynamic_exit_threshold(0.9, 0.9, base) < dynamic_exit_threshold(
        0.3, 0.3, base
    )


def test_should_exit_true_when_z_near_zero():
    config = default_config()
    position = _open_dummy_position()

    window = _mean_reverting_spread(seed=1)
    vals = window.to_numpy().copy()
    vals[-1] = vals[-30:-1].mean()  # force latest rolling z-score to ~0
    window = pd.Series(vals, index=window.index)

    decision = should_exit(position, window, p_t=0.9, config=config)

    assert decision.kappa_t > 0  # guards: a usable mean-reversion speed was fit
    assert abs(decision.z_t) < 1e-6
    assert decision.z_exit_t > 0
    assert decision.p_t == 0.9
    assert decision.should_exit


def test_should_exit_false_when_z_large():
    config = default_config()
    position = _open_dummy_position()

    window = _mean_reverting_spread(seed=1)
    vals = window.to_numpy().copy()
    std = vals[-30:-1].std(ddof=1)
    vals[-1] = vals[-30:-1].mean() + 3.5 * std  # latest point well outside the band
    window = pd.Series(vals, index=window.index)

    decision = should_exit(position, window, p_t=0.9, config=config)

    assert decision.kappa_t > 0
    assert abs(decision.z_t) > decision.z_exit_t
    assert not decision.should_exit
