"""Tests for the TFT regime gate classifier."""

import importlib.util

import numpy as np
import pandas as pd
import pytest

from pairs_trading.data.schemas import (
    Asset,
    CointegrationResult,
    Pair,
    SpreadData,
)
from pairs_trading.data.labels import (
    LabelConfig,
    build_feature_frame,
    make_latest_window,
    make_tft_dataset,
)

HAS_TORCH = importlib.util.find_spec("torch") is not None
skip_no_torch = pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")

if HAS_TORCH:
    from pairs_trading.models.tft import (
        N_FEATURES,
        FEATURE_NAMES,
        TFTScaling,
        TFTWindowData,
        classification_metrics,
        evaluate_baselines,
        fit_tft_scaling,
        purged_split_indices,
    )


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_spread_data(n: int = 200) -> SpreadData:
    rng = np.random.default_rng(1)
    index = pd.date_range("2020-01-01", periods=n, freq="B")
    spread = pd.Series(
        np.cumsum(rng.normal(0, 0.1, n)) * 0.5,
        index=index,
        name="spread",
    )
    z = (spread - spread.rolling(20).mean()) / spread.rolling(20).std()
    return SpreadData(
        pair=Pair(asset_a=Asset(symbol="V"), asset_b=Asset(symbol="MA")),
        spread=spread,
        z_score=z,
        intercept=0.0,
        hedge_ratio=1.0,
        cointegration=CointegrationResult(
            test_statistic=-4.0,
            p_value=0.01,
            critical_values={"1%": -3.9, "5%": -3.3, "10%": -3.0},
            is_cointegrated=True,
        ),
        half_life=18.0,
    )


def _make_arma_result(n: int, index: pd.DatetimeIndex):
    from pairs_trading.data.schemas import ArmaGarchResult

    rng = np.random.default_rng(0)
    cond_vol = pd.Series(np.abs(rng.normal(0.1, 0.02, n)), index=index)
    return ArmaGarchResult(
        arma_order=(1, 0, 1),
        garch_order=(1, 1),
        arma_aic=-100.0,
        garch_aic=-80.0,
        arma_params={"const": 0.0, "ar.L1": 0.3, "ma.L1": -0.1, "sigma2": 0.01},
        garch_params={"omega": 0.01, "alpha[1]": 0.1, "beta[1]": 0.85, "nu": 8.0},
        mu=0.0,
        phi=0.3,
        theta=-0.1,
        arma_residuals=pd.Series(rng.normal(0, 0.1, n), index=index),
        spread_forecast_next=0.0,
        omega=0.01,
        alpha=0.1,
        beta=0.85,
        nu=8.0,
        conditional_volatility=cond_vol,
        variance_forecast_next=0.01,
        vol_scaled_z_score=pd.Series(rng.normal(0, 1, n), index=index),
    )


def _make_window_data(n: int = 150, L: int = 10) -> tuple:
    """Return (TFTWindowData, labels, positions) for training tests."""
    rng = np.random.default_rng(42)
    x = rng.normal(size=(n, L, N_FEATURES)).astype(np.float32)
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    data = TFTWindowData(x=x, target_index=idx, input_length=L)
    labels = rng.choice([0.0, 1.0], size=n).astype(np.float32)
    positions = np.arange(n)
    return data, labels, positions


def _label_config(H: int = 5) -> LabelConfig:
    return LabelConfig(H=H, half_life=18.0, label_balance=0.5, pair_id="V_MA")


# ---------------------------------------------------------------------------
# TFTScaling / fit_tft_scaling
# ---------------------------------------------------------------------------


def test_tft_scaling_transform_roundtrip():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(50, 10, 3)).astype(np.float32)
    sc = fit_tft_scaling(x)
    out = sc.transform(x)
    np.testing.assert_allclose(out.reshape(-1, 3).mean(axis=0), np.zeros(3), atol=1e-5)
    np.testing.assert_allclose(out.reshape(-1, 3).std(axis=0), np.ones(3), atol=1e-4)
    np.testing.assert_allclose(sc.inverse_transform(out), x, atol=1e-5)


def test_tft_scaling_constant_column_gets_std_one():
    x = np.ones((10, 5, 3), dtype=np.float32)
    sc = fit_tft_scaling(x)
    assert np.all(sc.std == 1.0)


def test_tft_window_data_n_samples():
    x = np.zeros((17, 10, 3), dtype=np.float32)
    idx = pd.date_range("2021-01-01", periods=17, freq="B")
    data = TFTWindowData(x=x, target_index=idx, input_length=10)
    assert data.n_samples == 17


# ---------------------------------------------------------------------------
# classification_metrics
# ---------------------------------------------------------------------------


def test_classification_metrics_perfect():
    y = np.array([1, 1, 0, 0], dtype=float)
    m = classification_metrics(y, y)
    assert m["precision"] == pytest.approx(1.0, abs=1e-6)
    assert m["recall"] == pytest.approx(1.0, abs=1e-6)
    assert m["specificity"] == pytest.approx(1.0, abs=1e-6)
    assert m["f1"] == pytest.approx(1.0, abs=1e-6)
    assert m["mcc"] == pytest.approx(1.0, abs=1e-6)


def test_classification_metrics_all_ones_specificity_zero():
    y = np.array([1, 0, 1, 0], dtype=float)
    m = classification_metrics(np.ones(4), y)
    assert m["specificity"] == pytest.approx(0.0, abs=1e-6)
    assert m["recall"] == pytest.approx(1.0, abs=1e-6)


def test_classification_metrics_all_zeros_recall_zero():
    y = np.array([1, 0, 1, 0], dtype=float)
    m = classification_metrics(np.zeros(4), y)
    assert m["recall"] == pytest.approx(0.0, abs=1e-6)
    assert m["specificity"] == pytest.approx(1.0, abs=1e-6)


def test_classification_metrics_degenerate_mcc_zero():
    y = np.ones(10, dtype=float)
    m = classification_metrics(np.ones(10), y)
    assert m["mcc"] == pytest.approx(0.0, abs=1e-6)


def test_classification_metrics_threshold_respected():
    scores = np.array([0.4, 0.6], dtype=float)
    y = np.array([0.0, 1.0])
    assert classification_metrics(scores, y, threshold=0.3)["tp"] == 1.0
    assert classification_metrics(scores, y, threshold=0.7)["tp"] == 0.0


def test_classification_metrics_counts():
    y = np.array([1, 1, 0, 0], dtype=float)
    m = classification_metrics(y, y)
    assert m["n"] == 4
    assert m["base_rate_positive"] == pytest.approx(0.5, abs=1e-6)


# ---------------------------------------------------------------------------
# purged_split_indices
# ---------------------------------------------------------------------------


def test_purged_split_indices_basic_partition():
    pos = np.arange(200)
    tr, va, te = purged_split_indices(
        pos, val_fraction=0.15, test_fraction=0.15, embargo=0
    )
    assert len(tr) >= 1 and len(va) >= 1 and len(te) >= 1
    assert len(tr) + len(va) + len(te) <= 200


def test_purged_split_indices_chronological_order():
    pos = np.arange(200)
    tr, va, te = purged_split_indices(
        pos, val_fraction=0.15, test_fraction=0.15, embargo=0
    )
    assert tr.max() < va.min()
    assert va.max() < te.min()


def test_purged_split_indices_embargo_shrinks_train():
    pos = np.arange(300)
    tr0, _, _ = purged_split_indices(
        pos, val_fraction=0.15, test_fraction=0.15, embargo=0
    )
    tr5, _, _ = purged_split_indices(
        pos, val_fraction=0.15, test_fraction=0.15, embargo=5
    )
    assert len(tr5) <= len(tr0)


def test_purged_split_indices_too_small_raises():
    pos = np.arange(2)
    with pytest.raises(ValueError, match="Not enough windows"):
        purged_split_indices(pos, val_fraction=0.4, test_fraction=0.4, embargo=0)


def test_purged_split_indices_embargo_too_large_raises():
    pos = np.arange(20)
    with pytest.raises(ValueError, match="emptied a split"):
        purged_split_indices(pos, val_fraction=0.15, test_fraction=0.15, embargo=1000)


# ---------------------------------------------------------------------------
# build_feature_frame / make_tft_dataset / make_latest_window (labels.py)
# ---------------------------------------------------------------------------


def test_build_feature_frame_columns_match_feature_names():
    sd = _make_spread_data(n=200)
    ar = _make_arma_result(200, sd.spread.index)
    df = build_feature_frame(sd, ar, half_life_window=30)
    assert tuple(df.columns) == FEATURE_NAMES
    assert np.isfinite(df.to_numpy()).all()


def test_make_tft_dataset_output_shape_and_labels():
    sd = _make_spread_data(n=200)
    ar = _make_arma_result(200, sd.spread.index)
    data, labels, positions = make_tft_dataset(
        sd, ar, input_length=20, label_config=_label_config(), half_life_window=30
    )
    assert data.x.shape[1:] == (20, N_FEATURES)
    assert data.n_samples == len(labels) == len(positions)
    # -1 (no-signal) windows must already be filtered out
    assert set(np.unique(labels)).issubset({0.0, 1.0})


def test_make_tft_dataset_positions_strictly_increasing():
    sd = _make_spread_data(n=200)
    ar = _make_arma_result(200, sd.spread.index)
    _, _, positions = make_tft_dataset(
        sd, ar, input_length=20, label_config=_label_config(), half_life_window=30
    )
    assert np.all(np.diff(positions) > 0)


def test_make_tft_dataset_pre_fitted_scaling_applied():
    sd = _make_spread_data(n=200)
    ar = _make_arma_result(200, sd.spread.index)
    raw, _, _ = make_tft_dataset(
        sd, ar, input_length=20, label_config=_label_config(), half_life_window=30
    )
    sc = fit_tft_scaling(raw.x)
    scaled, _, _ = make_tft_dataset(
        sd,
        ar,
        input_length=20,
        label_config=_label_config(),
        half_life_window=30,
        scaling=sc,
    )
    np.testing.assert_allclose(scaled.x, sc.transform(raw.x), atol=1e-5)


def test_make_tft_dataset_too_short_raises():
    sd = _make_spread_data(n=60)
    ar = _make_arma_result(60, sd.spread.index)
    with pytest.raises(ValueError, match="Not enough observations"):
        make_tft_dataset(
            sd, ar, input_length=200, label_config=_label_config(), half_life_window=30
        )


def test_make_tft_dataset_input_length_lt_2_raises():
    sd = _make_spread_data(n=200)
    ar = _make_arma_result(200, sd.spread.index)
    with pytest.raises(ValueError, match="input_length must be at least 2"):
        make_tft_dataset(
            sd, ar, input_length=1, label_config=_label_config(), half_life_window=30
        )


def test_make_tft_dataset_x_is_float32():
    sd = _make_spread_data(n=200)
    ar = _make_arma_result(200, sd.spread.index)
    data, _, _ = make_tft_dataset(
        sd, ar, input_length=20, label_config=_label_config(), half_life_window=30
    )
    assert data.x.dtype == np.float32


def test_make_latest_window_shape():
    sd = _make_spread_data(n=200)
    ar = _make_arma_result(200, sd.spread.index)
    w = make_latest_window(sd, ar, input_length=20, half_life_window=30)
    assert w.shape == (1, 20, N_FEATURES)


def test_make_latest_window_matches_training_features():
    """The inference window must equal the last rows of the training features."""
    sd = _make_spread_data(n=200)
    ar = _make_arma_result(200, sd.spread.index)
    df = build_feature_frame(sd, ar, half_life_window=30)
    w = make_latest_window(sd, ar, input_length=20, half_life_window=30)
    np.testing.assert_allclose(w[0], df.to_numpy(dtype=np.float32)[-20:], atol=1e-6)


# ---------------------------------------------------------------------------
# PyTorch model tests
# ---------------------------------------------------------------------------


@skip_no_torch
def test_tft_classifier_forward_shape():
    import torch
    from pairs_trading.models.tft import TFTClassifier

    model = TFTClassifier(
        input_length=12, d_model=16, n_heads=4, lstm_layers=1, dropout=0.0
    )
    x = torch.randn(4, 12, N_FEATURES)
    assert model(x).shape == (4,)


@skip_no_torch
def test_tft_classifier_predict_regime_score_in_unit_interval():
    import torch
    from pairs_trading.models.tft import TFTClassifier

    model = TFTClassifier(
        input_length=12, d_model=16, n_heads=4, lstm_layers=1, dropout=0.0
    )
    x = torch.randn(4, 12, N_FEATURES)
    scores = model.predict_regime_score(x)
    assert torch.all(scores >= 0) and torch.all(scores <= 1)


@skip_no_torch
def test_tft_classifier_variable_importance_sums_to_one():
    import torch
    from pairs_trading.models.tft import TFTClassifier

    model = TFTClassifier(
        input_length=12, d_model=16, n_heads=4, lstm_layers=1, dropout=0.0
    )
    x = torch.randn(4, 12, N_FEATURES)
    imp = model.get_variable_importance(x)
    assert set(imp.keys()) == set(FEATURE_NAMES)
    assert sum(imp.values()) == pytest.approx(1.0, abs=1e-4)


@skip_no_torch
def test_tft_classifier_d_model_not_divisible_raises():
    from pairs_trading.models.tft import TFTClassifier

    with pytest.raises(
        ValueError, match=r"d_model \(\d+\) must be divisible by n_heads"
    ):
        TFTClassifier(input_length=10, d_model=10, n_heads=3)


@skip_no_torch
def test_grn_with_context():
    import torch
    from pairs_trading.models.tft import GatedResidualNetwork

    grn = GatedResidualNetwork(8, 8, 8, context_dim=4, dropout=0.0)
    out = grn(torch.randn(5, 8), torch.randn(5, 4))
    assert out.shape == (5, 8)


@skip_no_torch
def test_grn_without_context():
    import torch
    from pairs_trading.models.tft import GatedResidualNetwork

    grn = GatedResidualNetwork(8, 8, 8, dropout=0.0)
    assert grn(torch.randn(5, 8)).shape == (5, 8)


@skip_no_torch
def test_grn_dim_mismatch_uses_skip_proj():
    import torch
    from pairs_trading.models.tft import GatedResidualNetwork

    grn = GatedResidualNetwork(4, 8, 16, dropout=0.0)
    assert grn(torch.randn(3, 4)).shape == (3, 16)


@skip_no_torch
def test_attention_no_mask():
    import torch
    from pairs_trading.models.tft import InterpretableMultiHeadAttention

    attn = InterpretableMultiHeadAttention(d_model=16, n_heads=4, dropout=0.0)
    out, weights = attn(torch.randn(2, 10, 16))
    assert out.shape == (2, 10, 16)
    assert weights.shape == (2, 10, 10)


@skip_no_torch
def test_attention_with_causal_mask():
    import torch
    from pairs_trading.models.tft import InterpretableMultiHeadAttention

    L = 8
    attn = InterpretableMultiHeadAttention(d_model=16, n_heads=4, dropout=0.0)
    mask = torch.triu(torch.ones(L, L), diagonal=1).bool()
    out, _ = attn(torch.randn(2, L, 16), mask=mask)
    assert out.shape == (2, L, 16)


@skip_no_torch
def test_vsn_weights_sum_to_one():
    # NOTE: VSN signature changed from (n_vars, input_dim, d_model) to
    # (n_vars, d_model) — input_dim is always 1 (scalar per variable) and
    # is now hardcoded. This is an intentional design fix, not a regression.
    import torch
    from pairs_trading.models.tft import VariableSelectionNetwork

    vsn = VariableSelectionNetwork(n_vars=3, d_model=16, dropout=0.0)
    x = torch.randn(4, 10, 3)
    combined, weights = vsn(x)
    assert combined.shape == (4, 10, 16)
    np.testing.assert_allclose(
        weights.sum(dim=-1).detach().numpy(), np.ones((4, 10)), atol=1e-5
    )


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


@skip_no_torch
def test_train_tft_classifier_runs_and_returns_history():
    from pairs_trading.models.tft import TFTClassifier, train_tft_classifier

    data, labels, pos = _make_window_data()
    model = TFTClassifier(
        input_length=10, d_model=8, n_heads=2, lstm_layers=1, dropout=0.0
    )
    h = train_tft_classifier(
        model,
        data,
        labels,
        window_positions=pos,
        embargo=2,
        epochs=3,
        batch_size=16,
        val_fraction=0.15,
        test_fraction=0.15,
        device="cpu",
    )
    assert len(h.train_loss) >= 1
    assert len(h.val_loss) >= 1
    assert "mcc" in h.test_metrics
    assert isinstance(h.scaling, TFTScaling)
    assert isinstance(h.pos_weight, float)


@skip_no_torch
def test_train_tft_classifier_scaling_fit_on_train_only():
    from pairs_trading.models.tft import TFTClassifier, train_tft_classifier

    data, labels, pos = _make_window_data(n=200)
    model = TFTClassifier(
        input_length=10, d_model=8, n_heads=2, lstm_layers=1, dropout=0.0
    )
    h = train_tft_classifier(
        model,
        data,
        labels,
        window_positions=pos,
        embargo=2,
        epochs=2,
        batch_size=16,
        device="cpu",
    )
    tr, _, _ = purged_split_indices(
        pos, val_fraction=0.15, test_fraction=0.15, embargo=2
    )
    expected = fit_tft_scaling(data.x[tr])
    np.testing.assert_allclose(h.scaling.mean, expected.mean, atol=1e-5)
    np.testing.assert_allclose(h.scaling.std, expected.std, atol=1e-5)


@skip_no_torch
def test_train_tft_classifier_labels_length_mismatch_raises():
    from pairs_trading.models.tft import TFTClassifier, train_tft_classifier

    data, labels, pos = _make_window_data()
    model = TFTClassifier(
        input_length=10, d_model=8, n_heads=2, lstm_layers=1, dropout=0.0
    )
    with pytest.raises(ValueError, match="labels length"):
        train_tft_classifier(
            model,
            data,
            labels[:-5],
            window_positions=pos,
            embargo=2,
            epochs=1,
            device="cpu",
        )


@skip_no_torch
def test_train_tft_classifier_val_plus_test_ge_1_raises():
    from pairs_trading.models.tft import TFTClassifier, train_tft_classifier

    data, labels, pos = _make_window_data()
    model = TFTClassifier(
        input_length=10, d_model=8, n_heads=2, lstm_layers=1, dropout=0.0
    )
    with pytest.raises(ValueError, match="val_fraction"):
        train_tft_classifier(
            model,
            data,
            labels,
            window_positions=pos,
            embargo=0,
            val_fraction=0.5,
            test_fraction=0.6,
            epochs=1,
            device="cpu",
        )


@skip_no_torch
def test_train_tft_classifier_early_stopping_fires():
    from pairs_trading.models.tft import TFTClassifier, train_tft_classifier

    data, labels, pos = _make_window_data(n=200)
    model = TFTClassifier(
        input_length=10, d_model=8, n_heads=2, lstm_layers=1, dropout=0.0
    )
    h = train_tft_classifier(
        model,
        data,
        labels,
        window_positions=pos,
        embargo=2,
        epochs=100,
        batch_size=16,
        early_stopping_patience=1,
        device="cpu",
    )
    assert len(h.train_loss) < 100


@skip_no_torch
def test_train_tft_classifier_pos_weight_capped():
    from pairs_trading.models.tft import (
        MAX_POS_WEIGHT,
        TFTClassifier,
        train_tft_classifier,
    )

    data, _, pos = _make_window_data()
    labels = np.ones(len(data.x), dtype=np.float32)
    model = TFTClassifier(
        input_length=10, d_model=8, n_heads=2, lstm_layers=1, dropout=0.0
    )
    h = train_tft_classifier(
        model,
        data,
        labels,
        window_positions=pos,
        embargo=0,
        epochs=1,
        batch_size=16,
        device="cpu",
    )
    assert h.pos_weight <= MAX_POS_WEIGHT


@skip_no_torch
def test_evaluate_tft_multi_seed_returns_mean_std_keys():
    from pairs_trading.models.tft import TFTClassifier, evaluate_tft_multi_seed

    n, L = 120, 8
    rng = np.random.default_rng(0)
    x = rng.normal(size=(n, L, N_FEATURES)).astype(np.float32)
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    data = TFTWindowData(x=x, target_index=idx, input_length=L)
    labels = rng.choice([0.0, 1.0], size=n).astype(np.float32)
    pos = np.arange(n)

    def make_model():
        return TFTClassifier(
            input_length=L, d_model=8, n_heads=2, lstm_layers=1, dropout=0.0
        )

    _, agg, models = evaluate_tft_multi_seed(
        make_model,
        data,
        labels,
        seeds=(0, 1),
        window_positions=pos,
        embargo=2,
        epochs=2,
        batch_size=16,
        device="cpu",
    )
    assert len(models) == 2
    for k in [
        "f1",
        "mcc",
        "specificity",
        "recall",
        "precision",
        "balanced_accuracy",
        "npv",
    ]:
        assert k in agg
        assert "mean" in agg[k] and "std" in agg[k]


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------


@skip_no_torch
def test_evaluate_baselines_both_present():
    n, L = 120, 8
    rng = np.random.default_rng(0)
    x = rng.normal(size=(n, L, N_FEATURES)).astype(np.float32)
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    data = TFTWindowData(x=x, target_index=idx, input_length=L)
    labels = rng.choice([0.0, 1.0], size=n).astype(np.float32)
    result = evaluate_baselines(data, labels, window_positions=np.arange(n), embargo=2)
    assert "all_ones" in result and "logistic" in result
    for m in result.values():
        assert "mcc" in m and "f1" in m and "specificity" in m


@skip_no_torch
def test_evaluate_baselines_all_ones_specificity_zero():
    n, L = 100, 5
    rng = np.random.default_rng(0)
    x = rng.normal(size=(n, L, N_FEATURES)).astype(np.float32)
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    data = TFTWindowData(x=x, target_index=idx, input_length=L)
    labels = rng.choice([0.0, 1.0], size=n).astype(np.float32)
    result = evaluate_baselines(data, labels, window_positions=np.arange(n), embargo=0)
    assert result["all_ones"]["specificity"] == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Auxiliary forecasting head (dual-head, shared trunk)
# ---------------------------------------------------------------------------


@skip_no_torch
def test_forecast_head_forward_shapes():
    import torch
    from pairs_trading.models.tft import TFTClassifier

    model = TFTClassifier(
        input_length=12, d_model=16, n_heads=2, dropout=0.0, forecast_horizon=5
    )
    x = torch.randn(4, 12, N_FEATURES)
    logit, z_hat = model(x, return_forecast=True)
    assert logit.shape == (4,)
    assert z_hat.shape == (4, 5)
    # Default forward (inference path) still returns logits only.
    assert model(x).shape == (4,)


@skip_no_torch
def test_forecast_head_absent_by_default():
    import torch
    from pairs_trading.models.tft import TFTClassifier

    model = TFTClassifier(input_length=12, d_model=16, n_heads=2, dropout=0.0)
    assert model.forecast_head is None
    with pytest.raises(ValueError, match="forecast_horizon"):
        model(torch.randn(2, 12, N_FEATURES), return_forecast=True)


@skip_no_torch
def test_dataset_builder_fills_future_z():
    sd = _make_spread_data(n=200)
    ar = _make_arma_result(200, sd.spread.index)
    H = 5
    data, labels, _ = make_tft_dataset(
        sd, ar, input_length=20, label_config=_label_config(H), half_life_window=30
    )
    assert data.future_z is not None
    assert data.future_z.shape == (data.n_samples, H)
    # Spot-check: future_z is exactly z[t+1 .. t+H] for the window timestamp.
    z_clean = sd.z_score.dropna().astype(float)
    i = data.n_samples // 2
    pos = z_clean.index.get_loc(data.target_index[i])
    np.testing.assert_allclose(
        data.future_z[i], z_clean.to_numpy()[pos + 1 : pos + 1 + H], rtol=1e-5
    )


@skip_no_torch
def test_train_with_aux_forecast_loss():
    from pairs_trading.models.tft import TFTClassifier, train_tft_classifier

    data, labels, pos = _make_window_data(n=200)
    rng = np.random.default_rng(0)
    data = TFTWindowData(
        x=data.x,
        target_index=data.target_index,
        input_length=data.input_length,
        future_z=rng.normal(size=(200, 5)).astype(np.float32),
    )
    model = TFTClassifier(
        input_length=10, d_model=8, n_heads=2, dropout=0.0, forecast_horizon=5
    )
    h = train_tft_classifier(
        model,
        data,
        labels,
        window_positions=pos,
        embargo=2,
        lambda_forecast=0.1,
        epochs=3,
        batch_size=16,
        device="cpu",
    )
    assert h.lambda_forecast == 0.1
    assert len(h.val_forecast_mse) == len(h.val_loss)
    assert np.isfinite(h.test_forecast_mse)
    # Selection metric still describes the restored checkpoint.
    assert h.best_val_mcc == pytest.approx(h.val_mcc[h.best_epoch])


@skip_no_torch
def test_train_aux_requires_head_and_targets():
    from pairs_trading.models.tft import TFTClassifier, train_tft_classifier

    data, labels, pos = _make_window_data(n=150)
    model_no_head = TFTClassifier(input_length=10, d_model=8, n_heads=2, dropout=0.0)
    with pytest.raises(ValueError, match="forecast_horizon"):
        train_tft_classifier(
            model_no_head,
            data,
            labels,
            window_positions=pos,
            embargo=2,
            lambda_forecast=0.1,
            epochs=1,
            device="cpu",
        )
    model = TFTClassifier(
        input_length=10, d_model=8, n_heads=2, dropout=0.0, forecast_horizon=5
    )
    with pytest.raises(ValueError, match="future_z"):
        train_tft_classifier(
            model,
            data,
            labels,
            window_positions=pos,
            embargo=2,
            lambda_forecast=0.1,
            epochs=1,
            device="cpu",
        )


@skip_no_torch
def test_get_dynamic_time_stop_crossing():
    import torch
    from pairs_trading.models.tft import get_dynamic_time_stop

    # Long-side entry (z>0): first index where predicted path <= 0.
    assert get_dynamic_time_stop(torch.tensor([1.4, 0.9, 0.3, -0.1, -0.5]), 2.0) == 3
    # Short-side entry (z<0): first index where predicted path >= 0.
    assert get_dynamic_time_stop(torch.tensor([-1.1, -0.4, 0.2, 0.8]), -1.8) == 2
    # Exact zero counts as a crossing.
    assert get_dynamic_time_stop(torch.tensor([0.7, 0.0, -0.2]), 1.5) == 1


@skip_no_torch
def test_get_dynamic_time_stop_no_crossing_returns_h():
    import torch
    from pairs_trading.models.tft import get_dynamic_time_stop

    assert get_dynamic_time_stop(torch.tensor([2.0, 1.8, 1.6, 1.5, 1.4]), 2.2) == 5


@skip_no_torch
def test_ramzy_exit_threshold_formula_and_intuition():
    from pairs_trading.models.tft import ramzy_exit_threshold

    # z_exit = (c_z + lambda) / (p * kappa): 0.15 / (0.5 * 0.1) = 3.0
    assert ramzy_exit_threshold(0.5, 0.1, cost_z=0.05, risk_buffer=0.1) == (
        pytest.approx(3.0)
    )
    # Intuition: lower confidence -> HIGHER threshold -> exits sooner.
    assert ramzy_exit_threshold(0.3, 0.1) > ramzy_exit_threshold(0.9, 0.1)
    # Slower reversion -> higher threshold.
    assert ramzy_exit_threshold(0.7, 0.05) > ramzy_exit_threshold(0.7, 0.2)
    # Higher costs / buffer -> higher threshold.
    assert ramzy_exit_threshold(0.7, 0.1, cost_z=0.2) > ramzy_exit_threshold(
        0.7, 0.1, cost_z=0.05
    )
    # Invalid kappa (no measurable reversion) -> inf -> immediate exit.
    assert ramzy_exit_threshold(0.7, float("nan")) == float("inf")
    assert ramzy_exit_threshold(0.7, 0.0) == float("inf")


@skip_no_torch
def test_reversion_speed_from_half_life():
    from pairs_trading.data.spread import reversion_speed_from_half_life

    # kappa = ln(2)/half_life; half_life=1 -> ln 2; round-trip identity.
    assert reversion_speed_from_half_life(1.0) == pytest.approx(np.log(2))
    hl = 10.0
    assert np.log(2) / reversion_speed_from_half_life(hl) == pytest.approx(hl)
    # Equivalence with the AR(1) convention: hl from phi=2^(-1/10) gives
    # kappa == -ln(phi).
    phi = 2 ** (-1 / hl)
    assert reversion_speed_from_half_life(hl) == pytest.approx(-np.log(phi))
    # Invalid half-life (incl. compute_half_life's inf for no reversion) -> nan.
    assert np.isnan(reversion_speed_from_half_life(float("inf")))
    assert np.isnan(reversion_speed_from_half_life(0.0))
    assert np.isnan(reversion_speed_from_half_life(-5.0))
    assert np.isnan(reversion_speed_from_half_life(float("nan")))


@skip_no_torch
def test_predict_regime_and_path_one_pass():
    import torch
    from pairs_trading.models.tft import TFTClassifier

    model = TFTClassifier(
        input_length=12, d_model=16, n_heads=2, dropout=0.0, forecast_horizon=5
    )
    probs, paths = model.predict_regime_and_path(torch.randn(4, 12, N_FEATURES))
    assert probs.shape == (4,) and paths.shape == (4, 5)
    assert torch.all(probs >= 0) and torch.all(probs <= 1)


@skip_no_torch
def test_dataset_builder_fills_entry_z():
    sd = _make_spread_data(n=200)
    ar = _make_arma_result(200, sd.spread.index)
    data, _, _ = make_tft_dataset(
        sd, ar, input_length=20, label_config=_label_config(5), half_life_window=30
    )
    assert data.entry_z is not None and data.entry_z.shape == (data.n_samples,)
    # Every labeled window is an entry, so |entry_z| >= entry_threshold.
    assert np.all(np.abs(data.entry_z) >= _label_config().entry_threshold - 1e-6)
    # Spot-check the signed value against the z series.
    z_clean = sd.z_score.dropna().astype(float)
    i = data.n_samples // 2
    assert data.entry_z[i] == pytest.approx(
        float(z_clean.loc[data.target_index[i]]), rel=1e-5
    )


@skip_no_torch
def test_train_lambda_zero_is_pure_classifier():
    from pairs_trading.models.tft import TFTClassifier, train_tft_classifier

    data, labels, pos = _make_window_data(n=150)
    model = TFTClassifier(input_length=10, d_model=8, n_heads=2, dropout=0.0)
    h = train_tft_classifier(
        model,
        data,
        labels,
        window_positions=pos,
        embargo=2,
        lambda_forecast=0.0,
        epochs=2,
        batch_size=16,
        device="cpu",
    )
    assert h.lambda_forecast == 0.0
    assert h.val_forecast_mse == []
    assert np.isnan(h.test_forecast_mse)
