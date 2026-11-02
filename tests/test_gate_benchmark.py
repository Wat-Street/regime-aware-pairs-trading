"""Contract tests for the standard regime-gate interface and the benchmark pipeline."""

import dataclasses

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")

from pairs_trading.benchmark.backtest import backtest_portfolio
from pairs_trading.benchmark.config import (
    BacktestConfig,
    BenchmarkConfig,
    DataConfig,
    WalkForwardConfig,
)
from pairs_trading.benchmark.data import load_universe
from pairs_trading.benchmark.run import run_benchmark
from pairs_trading.benchmark.splits import fold_masks, walk_forward_folds
from pairs_trading.data.labels import FEATURE_NAMES, _first_touch_label
from pairs_trading.gate import (
    GateConfig,
    GateReversionEstimator,
    RegimeGate,
    TradeRules,
    available_models,
    build_pair_dataset,
    make_windows,
)
from pairs_trading.signals.reversion import ReversionProbabilityEstimator

RULES = TradeRules(entry_z=1.5, target_z=0.5, stop_z=3.0, horizon=8)
SMALL = {
    "tft": {"d_model": 8, "n_heads": 2},
    "autoformer": {"d_model": 8, "num_layers": 1},
    "wtransformer": {"d_model": 8, "num_heads": 2},
}


@pytest.fixture(scope="module")
def datasets():
    cfg = DataConfig(
        source="synthetic", synthetic_pairs=2, start="2015-01-01", end="2018-12-31"
    )
    return {
        p.pair_id: build_pair_dataset(p, a, b, rules=RULES, formation_end="2016-06-30")
        for p, a, b in load_universe(cfg)
    }


@pytest.fixture(scope="module")
def ds(datasets):
    return next(iter(datasets.values()))


# -- labels ------------------------------------------------------------------


def test_first_touch_label_order_matters():
    # short-spread trade (z>0): target 0.5 hit on day 2, stop on day 3 -> good trade
    assert _first_touch_label(2.0, np.array([1.8, 0.4, 3.5]), 0.5, 3.0) == 1
    # stop first -> bad trade, even though it later reverts
    assert _first_touch_label(2.0, np.array([3.2, 0.1]), 0.5, 3.0) == 0
    # long-spread trade (z<0) never reaches -0.5 -> bad
    assert _first_touch_label(-2.0, np.array([-1.9, -1.2, -0.8]), 0.5, 3.0) == 0


# -- dataset / windows ----------------------------------------------------------


def test_features_are_canonical_and_finite(ds):
    assert tuple(ds.features.columns) == FEATURE_NAMES
    assert np.isfinite(ds.features.to_numpy()).all()
    assert set(np.unique(ds.labels)) <= {-1, 0, 1}


def test_window_ends_on_decision_date(ds):
    w = make_windows(ds, 20, select="labeled")
    i = len(w) // 2
    row = ds.features.loc[pd.Timestamp(w.dates[i])].to_numpy()
    np.testing.assert_allclose(w.x[i, -1], row)
    assert (w.labels >= 0).all()


def test_signal_windows_include_unlabeled_tail(ds):
    sig = make_windows(ds, 20, select="signal")
    lab = make_windows(ds, 20, select="labeled")
    assert len(sig) >= len(lab)
    zs = ds.z.reindex(pd.DatetimeIndex(sig.dates)).abs()
    assert (zs >= RULES.entry_z).all()


def test_hedge_ratio_uses_formation_only(ds):
    from statsmodels.api import OLS, add_constant

    form = slice(None, ds.formation_end)
    beta = (
        OLS(ds.price_a.loc[form], add_constant(ds.price_b.loc[form]))
        .fit()
        .params.iloc[1]
    )
    assert ds.hedge_ratio == pytest.approx(beta)


# -- model contract ---------------------------------------------------------------


@pytest.mark.parametrize("name", available_models())
def test_every_model_honours_the_contract(name, ds, tmp_path):
    cfg = GateConfig(input_length=24, epochs=2, batch_size=32, model_kwargs=SMALL)
    tf = RegimeGate.frame_transform(name, cfg)
    w = make_windows(ds, cfg.input_length, tf, select="labeled")
    gate = RegimeGate(name, w.x.shape[-1], w.feature_names, cfg)
    n = len(w)
    gate.fit(w.subset(np.arange(n) < n * 0.8), w.subset(np.arange(n) >= n * 0.8))

    p = gate.predict_proba(w)
    assert p.shape == (n,) and p.dtype == np.float32
    assert ((p >= 0) & (p <= 1)).all()

    path = gate.save(tmp_path / f"{name}.pt")
    np.testing.assert_allclose(
        RegimeGate.load(path).predict_proba(w), p, rtol=1e-5, atol=1e-6
    )

    # live path from a raw canonical feature frame matches the windowed path
    live = gate.predict_frame(ds.features.loc[: pd.Timestamp(w.dates[-1])])
    assert live == pytest.approx(float(p[-1]), abs=1e-5)


def test_wrong_input_shape_rejected(ds):
    cfg = GateConfig(input_length=24, epochs=1, model_kwargs=SMALL)
    w = make_windows(ds, 24, select="labeled")
    gate = RegimeGate("logistic", w.x.shape[-1], w.feature_names, cfg)
    gate.fit(w)
    with pytest.raises(ValueError):
        gate.predict_proba(w.x[:, :10])


def test_estimator_plugs_into_signals_layer(ds):
    cfg = GateConfig(input_length=24, epochs=1)
    w = make_windows(ds, 24, select="labeled")
    gate = RegimeGate("logistic", w.x.shape[-1], w.feature_names, cfg)
    gate.fit(w)
    est: ReversionProbabilityEstimator = GateReversionEstimator(
        gate
    )  # satisfies the Protocol
    p_t = est.probability(ds.features, horizon=RULES.horizon)
    assert 0 < p_t <= 1
    assert isinstance(est.allow_entry(ds.features), bool)


# -- splits / backtest -------------------------------------------------------------


def test_walk_forward_is_purged(ds):
    cal = ds.features.index
    folds = walk_forward_folds(
        cal, ds.formation_end, n_folds=3, horizon=RULES.horizon, val_fraction=0.15
    )
    w = make_windows(ds, 20, select="labeled")
    for f in folds:
        tr, va, te = fold_masks(w.dates, f, RULES.horizon, cal)
        dates = pd.DatetimeIndex(w.dates)
        gap = cal.get_loc(f.test_start) - cal.get_loc(dates[tr | va].max())
        assert gap > RULES.horizon
        if tr.any() and va.any():
            assert (
                cal.get_loc(dates[va].min()) - cal.get_loc(dates[tr].max())
                > RULES.horizon
            )
        assert not (tr & te).any() and not (va & te).any()


def test_oracle_backtest_matches_labels(datasets):
    """Labels and backtest share one rule: a perfect gate only ever exits at target."""
    start = max(d.formation_end for d in datasets.values())
    end = min(d.z.index[-1] for d in datasets.values())
    probs = {pid: (d.labels == 1).astype(float) for pid, d in datasets.items()}
    _, trades = backtest_portfolio(
        datasets, probs, start, end, RULES, BacktestConfig(), 0.5
    )
    assert len(trades) > 0
    assert set(trades["reason"]) <= {"target", "end"}


# -- end to end --------------------------------------------------------------------


def test_benchmark_end_to_end(tmp_path):
    cfg = BenchmarkConfig(
        name="test",
        models=("logistic", "tft"),
        seeds=(0,),
        data=DataConfig(
            source="synthetic",
            synthetic_pairs=2,
            start="2015-01-01",
            end="2018-12-31",
            formation_end="2016-06-30",
        ),
        rules=RULES,
        gate=GateConfig(input_length=20, epochs=2, model_kwargs=SMALL),
        walk_forward=WalkForwardConfig(n_folds=2, min_train_signals=30),
    )
    res = run_benchmark(cfg, out_root=tmp_path)
    assert list(res.summary.index) == ["logistic", "tft", "ungated", "oracle"]
    assert res.summary.loc["ungated", "take_rate"] == 1.0
    assert res.summary.loc["oracle", "auc"] == 1.0
    for f in (
        "report.md",
        "summary.csv",
        "predictions.csv.gz",
        "equity.csv",
        "models/tft_seed0.pt",
    ):
        assert (res.out_dir / f).exists()
    # every model is scored on the same decisions as the ungated reference
    p = res.predictions

    def keys(s):
        return set(
            map(tuple, p[p.strategy == s][["pair_id", "date"]].astype(str).to_numpy())
        )

    assert keys("tft") == keys("logistic") == keys("ungated")


def test_config_file_loads():
    from pairs_trading.benchmark.config import load_config

    for path in ("configs/benchmark.toml", "configs/smoke.toml"):
        cfg = load_config(path)
        assert set(cfg.models) <= set(available_models())
        assert dataclasses.is_dataclass(cfg.rules)
