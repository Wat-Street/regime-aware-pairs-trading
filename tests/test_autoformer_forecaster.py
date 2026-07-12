import numpy as np
import pandas as pd
import pytest
import torch

from types import SimpleNamespace

from pairs_trading.models.autoformer_forecaster import (
    AutoCorrelationBlock,
    AutoformerDecoderLayer,
    AutoformerForecaster,
    make_autoformer_forecast_dataset,
    train_autoformer_forecaster,
)


def test_autocorrelation_block_train_vs_eval_modes_differ_per_sample():
    torch.manual_seed(0)
    length, d_model = 64, 8
    t = torch.arange(length, dtype=torch.float32)
    s0 = torch.sin(2 * torch.pi * t / 8.0)
    s1 = torch.sin(2 * torch.pi * t / 21.0)
    x = torch.stack([s0, s1]).unsqueeze(-1).repeat(1, 1, d_model)  # (2, L, d_model)

    block = AutoCorrelationBlock(d_model=d_model, n_heads=2, top_k=3, dropout=0.0)

    block.eval()
    with torch.no_grad():
        out_eval = block(x)
    assert out_eval.shape == x.shape
    assert not torch.allclose(out_eval[0], out_eval[1], atol=1e-4)

    block.train()
    out_train = block(x)
    assert out_train.shape == x.shape


def test_autocorrelation_block_cross_attention_resizes_and_responds_to_kv():
    torch.manual_seed(0)
    d_model = 8
    block = AutoCorrelationBlock(d_model=d_model, n_heads=2, top_k=2, dropout=0.0)
    block.eval()

    query = torch.randn(3, 20, d_model)
    kv_short = torch.randn(3, 12, d_model)   # shorter than query -> zero-padded
    kv_long = torch.randn(3, 40, d_model)    # longer than query -> truncated

    with torch.no_grad():
        out_short = block(query, kv=kv_short)
        out_long = block(query, kv=kv_long)
        out_self = block(query)

    assert out_short.shape == query.shape
    assert out_long.shape == query.shape
    # Cross-attending to a different kv source must actually change the
    # output, not silently fall back to self-attention.
    assert not torch.allclose(out_short, out_self, atol=1e-4)
    assert not torch.allclose(out_long, out_self, atol=1e-4)


def test_autoformer_decoder_layer_shapes_and_trend_units():
    torch.manual_seed(0)
    d_model, num_features = 16, 5
    decoder_len, encoder_len = 20, 60
    layer = AutoformerDecoderLayer(
        d_model=d_model, n_heads=2, dim_feedforward=32, moving_avg=7,
        top_k=2, dropout=0.0, num_features=num_features,
    )
    x = torch.randn(2, decoder_len, d_model)
    encoder_out = torch.randn(2, encoder_len, d_model)

    seasonal, trend_delta = layer(x, encoder_out)
    assert seasonal.shape == (2, decoder_len, d_model)
    assert trend_delta.shape == (2, decoder_len, num_features)  # already in feature units


def test_autoformer_forecaster_forward_shape_and_sensitivity():
    torch.manual_seed(0)
    model = AutoformerForecaster(
        input_length=60, num_features=7, horizon=8, target_index=0,
        d_model=16, n_heads=2, num_encoder_layers=1, num_decoder_layers=1,
        dim_feedforward=32, moving_avg=9, top_k=2, dropout=0.0,
    )
    x = torch.randn(4, 60, 7)
    forecast = model(x)
    assert forecast.shape == (4, 8, 7)

    target = model.target_series(x)
    assert target.shape == (4, 8)
    torch.testing.assert_close(target, forecast[:, :, 0])

    x2 = x.clone()
    x2[:, :30, :] += 5.0
    assert not torch.allclose(model(x), model(x2), atol=1e-4)


def test_autoformer_forecaster_rejects_bad_target_index():
    with pytest.raises(ValueError):
        AutoformerForecaster(input_length=60, num_features=5, target_index=5)


def test_make_autoformer_forecast_dataset_shapes():
    idx = pd.date_range("2020-01-01", periods=200, freq="B")
    rng = np.random.default_rng(1)
    spread = pd.Series(rng.normal(size=200).cumsum() * 0.1, index=idx)
    z = (spread - spread.rolling(20).mean()) / spread.rolling(20).std()
    data = SimpleNamespace(spread=spread, z_score=z.fillna(0.0))

    dataset = make_autoformer_forecast_dataset(data, None, input_length=60, horizon=8, target="spread")
    assert dataset.x.shape[1:] == (60, dataset.n_features)
    assert dataset.y.shape == (dataset.n_samples, 8, dataset.n_features)
    assert dataset.target_index == dataset.feature_names.index("spread")
    assert np.isfinite(dataset.x).all()
    assert np.isfinite(dataset.y).all()


def test_train_autoformer_forecaster_runs_and_reports_naive_baseline():
    idx = pd.date_range("2020-01-01", periods=300, freq="B")
    rng = np.random.default_rng(2)
    spread_vals = np.zeros(300)
    for t in range(1, 300):
        spread_vals[t] = 0.9 * spread_vals[t - 1] + rng.normal(0, 0.2)
    spread = pd.Series(spread_vals, index=idx)
    z = (spread - spread.rolling(20).mean()) / spread.rolling(20).std()
    data = SimpleNamespace(spread=spread, z_score=z.fillna(0.0))

    dataset = make_autoformer_forecast_dataset(data, None, input_length=40, horizon=5, target="spread")
    model = AutoformerForecaster(
        input_length=40, num_features=dataset.n_features, horizon=5, target_index=dataset.target_index,
        d_model=8, n_heads=2, num_encoder_layers=1, num_decoder_layers=1,
        dim_feedforward=16, moving_avg=5, top_k=2, dropout=0.0,
    )
    history = train_autoformer_forecaster(model, dataset, epochs=5, batch_size=16, early_stopping_patience=3)

    assert len(history.train_loss) > 0
    assert np.isfinite(history.test_mse)
    assert np.isfinite(history.naive_test_mse)
    assert sum(history.split_sizes) == dataset.n_samples
