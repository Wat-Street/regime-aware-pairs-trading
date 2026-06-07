"""Wavelet-based Transformer forecaster for pair-spread time series.

The implementation adapts the W-Transformers paper to this project by
forecasting decomposed spread components and recombining them into a spread
forecast. The wavelet transform is implemented as a Haar, MODWT-style,
same-length multiresolution decomposition so it can be used without adding a
separate wavelet dependency.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from pairs_trading.data.schemas import SpreadData

try:  # Torch is optional until the neural model is instantiated or trained.
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:  # pragma: no cover - exercised only in environments without torch
    torch = None
    nn = None
    DataLoader = None
    TensorDataset = None


WaveletMode = Literal["causal", "circular"]
SpreadTarget = Literal["spread", "z_score"]


@dataclass(frozen=True)
class WaveletScaling:
    """Per-component scaling statistics for Transformer inputs/targets."""

    mean: np.ndarray
    std: np.ndarray

    def transform(self, values: np.ndarray) -> np.ndarray:
        """Standardize values whose last axis is the component axis."""
        return (values - self.mean) / self.std

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        """Undo component-wise standardization."""
        return values * self.std + self.mean


@dataclass(frozen=True)
class WaveletDecomposition:
    """Same-length wavelet components for one spread-like series."""

    components: np.ndarray
    component_names: tuple[str, ...]
    index: pd.Index
    mode: WaveletMode

    @property
    def n_components(self) -> int:
        return self.components.shape[-1]

    def reconstruct(self) -> np.ndarray:
        """Recombine detail and smooth components back into the source series."""
        return self.components.sum(axis=-1)

    def to_frame(self) -> pd.DataFrame:
        """Return the decomposition as a timestamp-indexed DataFrame."""
        return pd.DataFrame(
            self.components,
            index=self.index,
            columns=self.component_names,
        )


@dataclass(frozen=True)
class WaveletWindowData:
    """Supervised windows produced from an existing spread pipeline output."""

    x: np.ndarray
    y_components: np.ndarray
    y_spread: np.ndarray
    component_names: tuple[str, ...]
    input_length: int
    forecast_horizon: int
    target_index: pd.Index
    scaling: WaveletScaling | None
    mode: WaveletMode

    @property
    def n_samples(self) -> int:
        return self.x.shape[0]

    @property
    def n_components(self) -> int:
        return self.x.shape[-1]


@dataclass(frozen=True)
class TransformerTrainingHistory:
    """Loss history from fitting a wavelet Transformer."""

    train_loss: list[float]
    validation_loss: list[float]


def _require_torch() -> None:
    if torch is None or nn is None or DataLoader is None or TensorDataset is None:
        raise ImportError(
            "WaveletTransformerForecaster requires PyTorch. Install the ML extra "
            "for this project, for example: uv sync --extra ml --dev"
        )


def _coerce_series(
    source: pd.Series | SpreadData,
    *,
    target: SpreadTarget,
) -> pd.Series:
    if isinstance(source, SpreadData):
        series = source.spread if target == "spread" else source.z_score
    elif isinstance(source, pd.Series):
        series = source
    else:
        raise TypeError("source must be a pandas Series or SpreadData")

    clean = series.dropna().astype(float).sort_index()
    if clean.empty:
        raise ValueError("source series must contain at least one non-null value")
    return clean


def recommended_wavelet_levels(n_observations: int) -> int:
    """
    Choose the paper-inspired default number of wavelet detail levels.

    The W-Transformers paper uses J + 1 = floor(log(N)) decomposed series. Since
    the final component is the smooth series, this returns J detail levels.
    """
    if n_observations < 4:
        raise ValueError("at least 4 observations are required for decomposition")
    return max(1, int(np.floor(np.log(n_observations))) - 1)


def _lag(values: np.ndarray, periods: int, *, mode: WaveletMode) -> np.ndarray:
    if mode == "circular":
        return np.roll(values, periods)
    if mode != "causal":
        raise ValueError("mode must be either 'causal' or 'circular'")

    lagged = np.empty_like(values)
    lagged[:periods] = values[0]
    lagged[periods:] = values[:-periods]
    return lagged


def haar_modwt_decompose(
    source: pd.Series | SpreadData,
    *,
    levels: int | None = None,
    target: SpreadTarget = "spread",
    mode: WaveletMode = "causal",
) -> WaveletDecomposition:
    """
    Decompose a spread-like series into additive Haar wavelet components.

    ``causal`` mode is the trading-safe default: each timestamp only uses current
    and prior values. ``circular`` mode is closer to common offline MODWT
    boundary handling, but wraps the end of the series into the beginning.
    """
    series = _coerce_series(source, target=target)
    values = series.to_numpy(dtype=float)
    if levels is None:
        levels = recommended_wavelet_levels(len(values))
    if levels < 1:
        raise ValueError("levels must be at least 1")

    current = values.copy()
    components: list[np.ndarray] = []

    for level in range(1, levels + 1):
        lag_periods = 2 ** (level - 1)
        smooth = 0.5 * (current + _lag(current, lag_periods, mode=mode))
        detail = current - smooth
        components.append(detail)
        current = smooth

    components.append(current)
    names = tuple(f"detail_{level}" for level in range(1, levels + 1)) + ("smooth",)

    return WaveletDecomposition(
        components=np.column_stack(components),
        component_names=names,
        index=series.index,
        mode=mode,
    )


def fit_wavelet_scaling(values: np.ndarray) -> WaveletScaling:
    """Fit component-wise scaling statistics on an array with components last."""
    flattened = values.reshape(-1, values.shape[-1])
    mean = flattened.mean(axis=0)
    std = flattened.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    return WaveletScaling(mean=mean, std=std)


def make_wavelet_forecasting_dataset(
    source: pd.Series | SpreadData,
    *,
    input_length: int = 12,
    forecast_horizon: int = 1,
    levels: int | None = None,
    target: SpreadTarget = "spread",
    mode: WaveletMode = "causal",
    standardize: bool = True,
    scaling: WaveletScaling | None = None,
) -> WaveletWindowData:
    """
    Build supervised Transformer windows from a spread or ``SpreadData`` object.

    ``x`` has shape ``(samples, input_length, wavelet_components)``.
    ``y_components`` has shape ``(samples, forecast_horizon, wavelet_components)``.
    ``y_spread`` keeps the unscaled recombined spread target for evaluation.
    """
    if input_length < 2:
        raise ValueError("input_length must be at least 2")
    if forecast_horizon < 1:
        raise ValueError("forecast_horizon must be at least 1")

    decomposition = haar_modwt_decompose(
        source,
        levels=levels,
        target=target,
        mode=mode,
    )
    components = decomposition.components
    n_observations = len(components)
    n_samples = n_observations - input_length - forecast_horizon + 1
    if n_samples <= 0:
        raise ValueError(
            "not enough observations for the requested input_length and "
            "forecast_horizon"
        )

    x = np.stack(
        [components[i : i + input_length] for i in range(n_samples)],
        axis=0,
    )
    y_components_unscaled = np.stack(
        [
            components[i + input_length : i + input_length + forecast_horizon]
            for i in range(n_samples)
        ],
        axis=0,
    )
    y_spread = y_components_unscaled.sum(axis=-1)

    fitted_scaling = scaling
    y_components = y_components_unscaled
    if standardize:
        fitted_scaling = scaling or fit_wavelet_scaling(x)
        x = fitted_scaling.transform(x)
        y_components = fitted_scaling.transform(y_components_unscaled)

    target_start = input_length
    target_stop = input_length + n_samples

    return WaveletWindowData(
        x=x.astype(np.float32),
        y_components=y_components.astype(np.float32),
        y_spread=y_spread.astype(np.float32),
        component_names=decomposition.component_names,
        input_length=input_length,
        forecast_horizon=forecast_horizon,
        target_index=decomposition.index[target_start:target_stop],
        scaling=fitted_scaling,
        mode=mode,
    )


def make_latest_wavelet_window(
    source: pd.Series | SpreadData,
    *,
    input_length: int,
    levels: int | None = None,
    target: SpreadTarget = "spread",
    mode: WaveletMode = "causal",
    scaling: WaveletScaling | None = None,
) -> np.ndarray:
    """Create one model-ready input window from the latest available values."""
    if input_length < 2:
        raise ValueError("input_length must be at least 2")

    decomposition = haar_modwt_decompose(
        source,
        levels=levels,
        target=target,
        mode=mode,
    )
    if len(decomposition.components) < input_length:
        raise ValueError("not enough observations for input_length")

    x = decomposition.components[-input_length:][np.newaxis, :, :]
    if scaling is not None:
        x = scaling.transform(x)
    return x.astype(np.float32)


if torch is not None and nn is not None:

    class SinusoidalPositionalEncoding(nn.Module):
        """Fixed positional encoding for timestamp order inside each window."""

        def __init__(self, d_model: int, max_length: int = 4096) -> None:
            super().__init__()
            position = torch.arange(max_length, dtype=torch.float32).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, d_model, 2, dtype=torch.float32)
                * (-math.log(10000.0) / d_model)
            )
            pe = torch.zeros(max_length, d_model)
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
            self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x + self.pe[:, : x.size(1)]

    class ComponentTransformerForecaster(nn.Module):
        """Local Transformer for one wavelet component."""

        def __init__(
            self,
            *,
            input_length: int,
            forecast_horizon: int,
            d_model: int = 16,
            num_heads: int = 8,
            num_encoder_layers: int = 2,
            dim_feedforward: int = 64,
            dropout: float = 0.1,
            activation: str = "relu",
        ) -> None:
            super().__init__()
            if d_model % num_heads != 0:
                raise ValueError("d_model must be divisible by num_heads")

            self.input_length = input_length
            self.forecast_horizon = forecast_horizon
            self.input_projection = nn.Linear(1, d_model)
            self.position = SinusoidalPositionalEncoding(
                d_model=d_model,
                max_length=input_length,
            )
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=num_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation=activation,
                batch_first=True,
            )
            self.encoder = nn.TransformerEncoder(
                encoder_layer,
                num_layers=num_encoder_layers,
            )
            self.head = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, forecast_horizon),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            if x.ndim != 3 or x.shape[1] != self.input_length or x.shape[2] != 1:
                raise ValueError(
                    "component input must have shape (batch, input_length, 1)"
                )

            encoded = self.input_projection(x)
            encoded = self.position(encoded)
            encoded = self.encoder(encoded)
            return self.head(encoded[:, -1, :])

    class WaveletTransformerForecaster(nn.Module):
        """
        W-Transformer-style model with one local Transformer per component.

        The component forecasts are summed to produce the final spread forecast,
        matching the decomposed-then-recombined flow from the paper.
        """

        def __init__(
            self,
            *,
            input_length: int,
            forecast_horizon: int,
            num_components: int,
            d_model: int = 16,
            num_heads: int = 8,
            num_encoder_layers: int = 2,
            dim_feedforward: int = 64,
            dropout: float = 0.1,
            activation: str = "relu",
        ) -> None:
            super().__init__()
            if num_components < 2:
                raise ValueError("num_components must include detail(s) and smooth")

            self.input_length = input_length
            self.forecast_horizon = forecast_horizon
            self.num_components = num_components
            self.component_models = nn.ModuleList(
                [
                    ComponentTransformerForecaster(
                        input_length=input_length,
                        forecast_horizon=forecast_horizon,
                        d_model=d_model,
                        num_heads=num_heads,
                        num_encoder_layers=num_encoder_layers,
                        dim_feedforward=dim_feedforward,
                        dropout=dropout,
                        activation=activation,
                    )
                    for _ in range(num_components)
                ]
            )

        def predict_components(self, x: torch.Tensor) -> torch.Tensor:
            """Forecast each decomposed component separately."""
            if x.ndim != 3:
                raise ValueError(
                    "wavelet input must have shape "
                    "(batch, input_length, num_components)"
                )
            if x.shape[1] != self.input_length or x.shape[2] != self.num_components:
                raise ValueError(
                    "wavelet input shape does not match model configuration"
                )

            forecasts = [
                component_model(x[:, :, idx : idx + 1])
                for idx, component_model in enumerate(self.component_models)
            ]
            return torch.stack(forecasts, dim=-1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Return the recombined spread forecast."""
            return self.predict_components(x).sum(dim=-1)

else:

    class WaveletTransformerForecaster:  # pragma: no cover
        """Placeholder that raises a helpful error when torch is unavailable."""

        def __init__(self, *_args: object, **_kwargs: object) -> None:
            _require_torch()


def train_wavelet_transformer(
    model: "WaveletTransformerForecaster",
    data: WaveletWindowData,
    *,
    epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    validation_split: float = 0.0,
    shuffle: bool = True,
    device: str | None = None,
) -> TransformerTrainingHistory:
    """Train a W-Transformer on component-level forecasting targets."""
    _require_torch()
    if epochs < 1:
        raise ValueError("epochs must be at least 1")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if not 0 <= validation_split < 1:
        raise ValueError("validation_split must be in [0, 1)")

    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif (
            getattr(torch.backends, "mps", None) is not None
            and torch.backends.mps.is_available()
        ):
            device = "mps"
        else:
            device = "cpu"

    x = torch.as_tensor(data.x, dtype=torch.float32)
    y = torch.as_tensor(data.y_components, dtype=torch.float32)
    n_validation = int(len(x) * validation_split)
    n_train = len(x) - n_validation
    if n_train <= 0:
        raise ValueError("validation_split leaves no training samples")

    train_dataset = TensorDataset(x[:n_train], y[:n_train])
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
    )
    validation_loader = None
    if n_validation:
        validation_dataset = TensorDataset(x[n_train:], y[n_train:])
        validation_loader = DataLoader(validation_dataset, batch_size=batch_size)

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    train_loss: list[float] = []
    validation_loss: list[float] = []

    for _ in range(epochs):
        model.train()
        epoch_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            loss = criterion(model.predict_components(batch_x), batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item()) * len(batch_x)

        train_loss.append(epoch_loss / n_train)

        if validation_loader is not None:
            model.eval()
            total_validation_loss = 0.0
            with torch.no_grad():
                for batch_x, batch_y in validation_loader:
                    batch_x = batch_x.to(device)
                    batch_y = batch_y.to(device)
                    loss = criterion(model.predict_components(batch_x), batch_y)
                    total_validation_loss += float(loss.item()) * len(batch_x)
            validation_loss.append(total_validation_loss / n_validation)

    return TransformerTrainingHistory(
        train_loss=train_loss,
        validation_loss=validation_loss,
    )


def predict_spread_forecast(
    model: "WaveletTransformerForecaster",
    x: np.ndarray,
    *,
    scaling: WaveletScaling | None = None,
    device: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Predict spread values from model-ready wavelet windows.

    Returns ``(spread_forecast, component_forecasts)`` as NumPy arrays. If the
    model was trained on standardized components, pass the dataset scaling to
    return forecasts in the original spread units.
    """
    _require_torch()
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    with torch.no_grad():
        tensor_x = torch.as_tensor(x, dtype=torch.float32, device=device)
        component_forecasts = model.predict_components(tensor_x).cpu().numpy()

    if scaling is not None:
        component_forecasts = scaling.inverse_transform(component_forecasts)

    spread_forecast = component_forecasts.sum(axis=-1)
    return spread_forecast, component_forecasts
