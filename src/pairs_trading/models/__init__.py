"""Forecasting models for regime-aware pairs trading."""

from pairs_trading.models.w_transformer import (
    TransformerTrainingHistory,
    WaveletDecomposition,
    WaveletScaling,
    WaveletTransformerForecaster,
    WaveletWindowData,
    fit_wavelet_scaling,
    haar_modwt_decompose,
    make_latest_wavelet_window,
    make_wavelet_forecasting_dataset,
    predict_spread_forecast,
    recommended_wavelet_levels,
    train_wavelet_transformer,
)

__all__ = [
    "TransformerTrainingHistory",
    "WaveletDecomposition",
    "WaveletScaling",
    "WaveletTransformerForecaster",
    "WaveletWindowData",
    "fit_wavelet_scaling",
    "haar_modwt_decompose",
    "make_latest_wavelet_window",
    "make_wavelet_forecasting_dataset",
    "predict_spread_forecast",
    "recommended_wavelet_levels",
    "train_wavelet_transformer",
]
