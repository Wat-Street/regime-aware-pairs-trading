"""Tests for the wavelet Transformer data interface and model."""

import importlib.util

import numpy as np
import pandas as pd
import pytest

from pairs_trading.data.schemas import Asset, CointegrationResult, Pair, SpreadData
from pairs_trading.models.w_transformer import (
    WaveletTransformerForecaster,
    haar_modwt_decompose,
    make_latest_wavelet_window,
    make_wavelet_forecasting_dataset,
    recommended_wavelet_levels,
)


def _make_spread_data(n: int = 80) -> SpreadData:
    index = pd.date_range("2024-01-01", periods=n, freq="D")
    t = np.arange(n, dtype=float)
    spread = pd.Series(
        0.4 * np.sin(t / 4.0) + 0.02 * t + 0.1 * np.cos(t / 13.0),
        index=index,
        name="spread",
    )
    z_score = (spread - spread.rolling(10).mean()) / spread.rolling(10).std()

    return SpreadData(
        pair=Pair(asset_a=Asset(symbol="MSFT"), asset_b=Asset(symbol="AAPL")),
        spread=spread,
        z_score=z_score,
        intercept=1.0,
        hedge_ratio=0.8,
        cointegration=CointegrationResult(
            test_statistic=-3.5,
            p_value=0.02,
            critical_values={"1%": -3.9, "5%": -3.3, "10%": -3.0},
            is_cointegrated=True,
        ),
        half_life=7.0,
    )


def test_recommended_wavelet_levels_matches_paper_rule():
    assert recommended_wavelet_levels(254) == 4
    assert recommended_wavelet_levels(2167) == 6


def test_haar_modwt_decomposition_reconstructs_spread():
    spread_data = _make_spread_data()

    decomposition = haar_modwt_decompose(spread_data, levels=3)

    assert decomposition.component_names == (
        "detail_1",
        "detail_2",
        "detail_3",
        "smooth",
    )
    assert decomposition.components.shape == (80, 4)
    np.testing.assert_allclose(
        decomposition.reconstruct(),
        spread_data.spread.to_numpy(),
        atol=1e-10,
    )


def test_wavelet_dataset_uses_existing_spread_data_pipeline_output():
    spread_data = _make_spread_data(n=70)

    dataset = make_wavelet_forecasting_dataset(
        spread_data,
        input_length=12,
        forecast_horizon=2,
        levels=2,
    )

    assert dataset.n_samples == 57
    assert dataset.x.shape == (57, 12, 3)
    assert dataset.y_components.shape == (57, 2, 3)
    assert dataset.y_spread.shape == (57, 2)
    assert list(dataset.target_index[:2]) == list(spread_data.spread.index[12:14])
    assert dataset.scaling is not None
    assert np.isfinite(dataset.x).all()
    assert np.isfinite(dataset.y_components).all()


def test_latest_wavelet_window_uses_dataset_scaling():
    spread_data = _make_spread_data(n=70)
    dataset = make_wavelet_forecasting_dataset(
        spread_data,
        input_length=12,
        levels=2,
    )

    latest_window = make_latest_wavelet_window(
        spread_data,
        input_length=12,
        levels=2,
        scaling=dataset.scaling,
    )

    assert latest_window.shape == (1, 12, 3)
    assert np.isfinite(latest_window).all()


@pytest.mark.skipif(
    importlib.util.find_spec("torch") is None,
    reason="'torch' package not installed",
)
def test_wavelet_transformer_forward_returns_recombined_forecast():
    import torch

    dataset = make_wavelet_forecasting_dataset(
        _make_spread_data(n=50),
        input_length=12,
        forecast_horizon=3,
        levels=2,
    )
    model = WaveletTransformerForecaster(
        input_length=dataset.input_length,
        forecast_horizon=dataset.forecast_horizon,
        num_components=dataset.n_components,
        d_model=8,
        num_heads=2,
        num_encoder_layers=1,
        dim_feedforward=16,
        dropout=0.0,
    )

    batch = torch.as_tensor(dataset.x[:4], dtype=torch.float32)
    component_forecast = model.predict_components(batch)
    spread_forecast = model(batch)

    assert component_forecast.shape == (4, 3, 3)
    assert spread_forecast.shape == (4, 3)
    torch.testing.assert_close(
        spread_forecast,
        component_forecast.sum(dim=-1),
    )
