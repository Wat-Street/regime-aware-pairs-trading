"""Wavelet-based Transformer forecaster and regime classifier for pair-spread time series.

The implementation adapts the W-Transformers paper to this project by
forecasting decomposed spread components and recombining them into a spread
forecast. The wavelet transform is implemented as a Haar, MODWT-style,
same-length multiresolution decomposition so it can be used without adding a
separate wavelet dependency.

Two model variants are provided:

- ``WaveletTransformerForecaster``: original per-component regression model,
  outputs ``spread_forecast_next`` aligned with ``ArmaGarchResult``.

- ``WaveletTransformerClassifier``: adds cross-component fusion and a binary
  regime gate head, outputs a regime score in [0, 1]. Built on top of the
  forecaster's component encoders so both tasks can share weights.

Multivariate input is supported via ``make_wavelet_forecasting_dataset`` by
passing an ``ArmaGarchResult`` alongside ``SpreadData``. Each of the five
features (spread, z-score, ARMA residuals, conditional volatility,
vol-scaled z-score) is decomposed independently; the resulting band arrays
are concatenated along the component axis so the model sees every frequency
of every feature.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

from pairs_trading.data.schemas import ArmaGarchResult, SpreadData

try:  # Torch is optional until the neural model is instantiated or trained.
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:  # pragma: no cover
    torch = None
    nn = None
    DataLoader = None
    TensorDataset = None

WaveletMode = Literal["causal", "circular"]
SpreadTarget = Literal["spread", "z_score"]


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


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

    x: np.ndarray                        # (N, input_length, total_components)
    y_components: np.ndarray             # (N, forecast_horizon, total_components)
    y_spread: np.ndarray                 # (N, forecast_horizon) unscaled spread target
    component_names: tuple[str, ...]     # name per column in x/y_components
    input_length: int
    forecast_horizon: int
    target_index: pd.Index
    scaling: WaveletScaling | None
    mode: WaveletMode
    feature_names: tuple[str, ...] = field(default=("spread",))

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


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


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


def _align_features(
    spread_data: SpreadData,
    arma_result: ArmaGarchResult | None,
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    """
    Align all input features onto a shared DatetimeIndex.

    When ``arma_result`` is None only spread and z-score are included,
    giving a univariate-style (T, 2) frame. With ``arma_result`` all five
    canonical features are included: spread, z-score, ARMA residuals,
    conditional volatility, and vol-scaled z-score.
    """
    cols: dict[str, pd.Series] = {
        "spread":  spread_data.spread,
        "z_score": spread_data.z_score,
    }
    if arma_result is not None:
        cols["arma_residuals"]        = arma_result.arma_residuals
        cols["conditional_volatility"] = arma_result.conditional_volatility
        cols["vol_scaled_z_score"]     = arma_result.vol_scaled_z_score

    df = pd.DataFrame(cols).dropna()
    if df.empty:
        raise ValueError(
            "No overlapping non-null observations across all input features. "
            "Check that SpreadData and ArmaGarchResult share a common DatetimeIndex."
        )
    return df, tuple(cols.keys())


# ---------------------------------------------------------------------------
# Wavelet decomposition
# ---------------------------------------------------------------------------


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


def _decompose_feature_matrix(
    df: pd.DataFrame,
    *,
    levels: int | None,
    mode: WaveletMode,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """
    Decompose every column in ``df`` independently and concatenate results.

    Returns an array of shape ``(T, n_features * n_wavelet_components)`` and
    the corresponding column names interleaved as
    ``[feat_detail_1, feat_detail_2, ..., feat_smooth, ...]``.
    """
    all_components: list[np.ndarray] = []
    all_names: list[str] = []

    for col in df.columns:
        decomp = haar_modwt_decompose(
            df[col],
            levels=levels,
            mode=mode,
        )
        all_components.append(decomp.components)
        all_names.extend(f"{col}_{name}" for name in decomp.component_names)

    return np.concatenate(all_components, axis=-1), tuple(all_names)


# ---------------------------------------------------------------------------
# Scaling
# ---------------------------------------------------------------------------


def fit_wavelet_scaling(values: np.ndarray) -> WaveletScaling:
    """Fit component-wise scaling statistics on an array with components last."""
    flattened = values.reshape(-1, values.shape[-1])
    mean = flattened.mean(axis=0)
    std = flattened.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    return WaveletScaling(mean=mean, std=std)


# ---------------------------------------------------------------------------
# Dataset builders
# ---------------------------------------------------------------------------


def make_wavelet_forecasting_dataset(
    source: pd.Series | SpreadData,
    arma_result: ArmaGarchResult | None = None,
    *,
    input_length: int = 60,
    forecast_horizon: int = 1,
    levels: int | None = None,
    target: SpreadTarget = "spread",
    mode: WaveletMode = "causal",
    standardize: bool = True,
    scaling: WaveletScaling | None = None,
) -> WaveletWindowData:
    """
    Build supervised Transformer windows from spread pipeline outputs.

    When ``arma_result`` is provided the dataset uses all five features:
    spread, z-score, ARMA residuals, conditional volatility, and
    vol-scaled z-score. Each feature is decomposed independently; the
    resulting bands are concatenated so ``x`` has shape
    ``(N, input_length, n_features * n_wavelet_components)``.

    When ``arma_result`` is None only spread and z-score are used,
    matching the original univariate behaviour.

    ``y_components`` has shape ``(N, forecast_horizon, total_components)``.
    ``y_spread`` keeps the unscaled recombined spread target for evaluation.
    """
    if input_length < 2:
        raise ValueError("input_length must be at least 2")
    if forecast_horizon < 1:
        raise ValueError("forecast_horizon must be at least 1")

    # Build aligned feature frame
    if isinstance(source, (SpreadData,)) or arma_result is not None:
        spread_data = source if isinstance(source, SpreadData) else None
        if spread_data is None:
            raise TypeError(
                "When arma_result is provided, source must be a SpreadData instance."
            )
        df, feature_names = _align_features(spread_data, arma_result)
    else:
        # Plain pd.Series — univariate path
        series = _coerce_series(source, target=target)
        df = series.to_frame(name="spread")
        feature_names = ("spread",)

    # Decompose all features
    components, component_names = _decompose_feature_matrix(
        df, levels=levels, mode=mode
    )

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
    )  # (N, input_length, total_components)

    y_components_unscaled = np.stack(
        [
            components[i + input_length : i + input_length + forecast_horizon]
            for i in range(n_samples)
        ],
        axis=0,
    )  # (N, forecast_horizon, total_components)

    # y_spread: sum only the spread-feature components for evaluation
    spread_cols = [
        idx for idx, name in enumerate(component_names)
        if name.startswith("spread_")
    ]
    if spread_cols:
        y_spread = y_components_unscaled[:, :, spread_cols].sum(axis=-1)
    else:
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
        component_names=component_names,
        input_length=input_length,
        forecast_horizon=forecast_horizon,
        target_index=df.index[target_start:target_stop],
        scaling=fitted_scaling,
        mode=mode,
        feature_names=feature_names,
    )


def make_latest_wavelet_window(
    source: pd.Series | SpreadData,
    arma_result: ArmaGarchResult | None = None,
    *,
    input_length: int,
    levels: int | None = None,
    target: SpreadTarget = "spread",
    mode: WaveletMode = "causal",
    scaling: WaveletScaling | None = None,
) -> np.ndarray:
    """
    Create one model-ready input window from the latest available values.

    Accepts the same ``arma_result`` argument as
    ``make_wavelet_forecasting_dataset`` so the inference window is built
    from the same feature set the model was trained on.
    """
    if input_length < 2:
        raise ValueError("input_length must be at least 2")

    if isinstance(source, SpreadData) or arma_result is not None:
        spread_data = source if isinstance(source, SpreadData) else None
        if spread_data is None:
            raise TypeError(
                "When arma_result is provided, source must be a SpreadData instance."
            )
        df, _ = _align_features(spread_data, arma_result)
    else:
        series = _coerce_series(source, target=target)
        df = series.to_frame(name="spread")

    components, _ = _decompose_feature_matrix(df, levels=levels, mode=mode)

    if len(components) < input_length:
        raise ValueError("not enough observations for input_length")

    x = components[-input_length:][np.newaxis, :, :]  # (1, input_length, total_components)

    if scaling is not None:
        x = scaling.transform(x)

    return x.astype(np.float32)


# ---------------------------------------------------------------------------
# PyTorch model definitions
# ---------------------------------------------------------------------------

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
        """Local Transformer encoder for one wavelet component."""

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
            self.d_model = d_model

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

        def encode(self, x: torch.Tensor) -> torch.Tensor:
            """
            Return the last-step encoder representation without projecting to output.

            Shape: ``(batch, d_model)``. Used by the fusion layer in the
            classifier so all component representations can be combined before
            the regime gate head.
            """
            if x.ndim != 3 or x.shape[1] != self.input_length or x.shape[2] != 1:
                raise ValueError(
                    "component input must have shape (batch, input_length, 1)"
                )
            encoded = self.input_projection(x)
            encoded = self.position(encoded)
            encoded = self.encoder(encoded)
            return encoded[:, -1, :]  # (batch, d_model)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Return the component forecast of shape ``(batch, forecast_horizon)``."""
            return self.head(self.encode(x))

    class WaveletTransformerForecaster(nn.Module):
        """
        W-Transformer-style model with one local Transformer per component.

        The component forecasts are summed to produce the final spread forecast,
        matching the decomposed-then-recombined flow from the paper.

        Supports multivariate input: when the dataset was built with
        ``arma_result`` the model receives components from all five features
        concatenated along the last axis. Pass the correct ``num_components``
        (``n_features * n_wavelet_levels_per_feature``) at construction time.
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

        def _validate_input(self, x: torch.Tensor) -> None:
            if x.ndim != 3:
                raise ValueError(
                    "wavelet input must have shape "
                    "(batch, input_length, num_components)"
                )
            if x.shape[1] != self.input_length or x.shape[2] != self.num_components:
                raise ValueError(
                    f"expected input shape (batch, {self.input_length}, "
                    f"{self.num_components}), got {tuple(x.shape)}"
                )

        def encode_components(self, x: torch.Tensor) -> torch.Tensor:
            """
            Return per-component encoder representations.

            Shape: ``(batch, num_components, d_model)``. Used by
            ``WaveletTransformerClassifier`` for cross-component fusion.
            """
            self._validate_input(x)
            encodings = [
                model.encode(x[:, :, idx : idx + 1])
                for idx, model in enumerate(self.component_models)
            ]
            return torch.stack(encodings, dim=1)  # (batch, num_components, d_model)

        def predict_components(self, x: torch.Tensor) -> torch.Tensor:
            """Forecast each decomposed component separately."""
            self._validate_input(x)
            forecasts = [
                model(x[:, :, idx : idx + 1])
                for idx, model in enumerate(self.component_models)
            ]
            return torch.stack(forecasts, dim=-1)  # (batch, horizon, num_components)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Return the recombined spread forecast of shape ``(batch, horizon)``."""
            return self.predict_components(x).sum(dim=-1)

    class WaveletTransformerClassifier(nn.Module):
        """
        Regime gate classifier built on top of ``WaveletTransformerForecaster``.

        Architecture
        ------------
        1. Each component encoder (from the forecaster) produces a
           ``(batch, d_model)`` representation of its frequency band.
        2. All component representations are concatenated and passed through a
           linear fusion layer to produce a single ``(batch, fusion_dim)``
           multi-scale embedding.
        3. A two-layer MLP with sigmoid output produces a regime score in [0, 1].
           Scores above ``regime_threshold`` indicate a stable mean-reverting
           regime suitable for trade entry.

        The forecaster's weights are shared — you can either train both tasks
        jointly or pre-train the forecaster and fine-tune only the classifier
        head by freezing ``forecaster.parameters()``.

        Parameters
        ----------
        forecaster:
            A ``WaveletTransformerForecaster`` instance (trained or fresh).
        fusion_dim:
            Hidden size of the cross-component fusion layer.
        dropout:
            Dropout applied before the final classification layer.
        regime_threshold:
            Inference-time threshold used by ``predict_regime_score``.
            Not used during training.
        """

        def __init__(
            self,
            forecaster: WaveletTransformerForecaster,
            *,
            fusion_dim: int = 32,
            dropout: float = 0.1,
            regime_threshold: float = 0.6,
        ) -> None:
            super().__init__()
            self.forecaster = forecaster
            self.regime_threshold = regime_threshold

            d_model = forecaster.component_models[0].d_model
            num_components = forecaster.num_components

            # Cross-component fusion: flatten all component encodings → fusion_dim
            self.fusion = nn.Sequential(
                nn.Linear(num_components * d_model, fusion_dim),
                nn.LayerNorm(fusion_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )

            # Regime gate head: fusion_dim → scalar probability
            self.regime_head = nn.Sequential(
                nn.Linear(fusion_dim, fusion_dim // 2),
                nn.ReLU(),
                nn.Linear(fusion_dim // 2, 1),
                nn.Sigmoid(),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Return regime scores of shape ``(batch,)`` in [0, 1].

            Higher scores indicate a stable mean-reverting regime. The
            forecaster's per-component encoders are run once; the
            forecasting head is not called here.
            """
            # (batch, num_components, d_model)
            component_encodings = self.forecaster.encode_components(x)

            # Flatten component dimension: (batch, num_components * d_model)
            batch_size = component_encodings.shape[0]
            flat = component_encodings.view(batch_size, -1)

            # Cross-component fusion: (batch, fusion_dim)
            fused = self.fusion(flat)

            # Regime probability: (batch,)
            return self.regime_head(fused).squeeze(-1)

        def predict_regime_score(self, x: torch.Tensor) -> torch.Tensor:
            """
            Alias for ``forward``. Returns raw scores in [0, 1].
            Use ``score > self.regime_threshold`` for a binary gate signal.
            """
            return self.forward(x)

else:

    class WaveletTransformerForecaster:  # pragma: no cover
        """Placeholder that raises a helpful error when torch is unavailable."""

        def __init__(self, *_args: object, **_kwargs: object) -> None:
            _require_torch()

    class WaveletTransformerClassifier:  # pragma: no cover
        """Placeholder that raises a helpful error when torch is unavailable."""

        def __init__(self, *_args: object, **_kwargs: object) -> None:
            _require_torch()


# ---------------------------------------------------------------------------
# Training loops
# ---------------------------------------------------------------------------


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
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)

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
            total_val_loss = 0.0
            with torch.no_grad():
                for batch_x, batch_y in validation_loader:
                    batch_x = batch_x.to(device)
                    batch_y = batch_y.to(device)
                    loss = criterion(model.predict_components(batch_x), batch_y)
                    total_val_loss += float(loss.item()) * len(batch_x)
            validation_loss.append(total_val_loss / n_validation)

    return TransformerTrainingHistory(
        train_loss=train_loss,
        validation_loss=validation_loss,
    )


def train_wavelet_classifier(
    classifier: "WaveletTransformerClassifier",
    data: WaveletWindowData,
    labels: np.ndarray,
    *,
    epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    validation_split: float = 0.0,
    shuffle: bool = True,
    device: str | None = None,
    freeze_forecaster: bool = False,
) -> TransformerTrainingHistory:
    """
    Train the regime gate classifier on binary reversion labels.

    Parameters
    ----------
    classifier:
        A ``WaveletTransformerClassifier`` instance.
    data:
        The same ``WaveletWindowData`` used for (or compatible with) the
        underlying forecaster.
    labels:
        Binary array of shape ``(N,)`` where 1 = spread reverted within
        the forecast horizon (favorable regime) and 0 = it did not.
        Produced by ``pairs_trading.data.labels.generate_regime_labels``.
    freeze_forecaster:
        If True, gradients are blocked through the shared forecaster encoders
        and only the fusion + regime head weights are updated. Useful when
        fine-tuning on top of a pre-trained forecaster.
    """
    _require_torch()

    if epochs < 1:
        raise ValueError("epochs must be at least 1")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if not 0 <= validation_split < 1:
        raise ValueError("validation_split must be in [0, 1)")
    if len(labels) != data.n_samples:
        raise ValueError(
            f"labels length ({len(labels)}) must match dataset n_samples "
            f"({data.n_samples})"
        )

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

    if freeze_forecaster:
        for param in classifier.forecaster.parameters():
            param.requires_grad = False

    x = torch.as_tensor(data.x, dtype=torch.float32)
    y = torch.as_tensor(labels, dtype=torch.float32)

    n_validation = int(len(x) * validation_split)
    n_train = len(x) - n_validation

    if n_train <= 0:
        raise ValueError("validation_split leaves no training samples")

    train_dataset = TensorDataset(x[:n_train], y[:n_train])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)

    validation_loader = None
    if n_validation:
        validation_dataset = TensorDataset(x[n_train:], y[n_train:])
        validation_loader = DataLoader(validation_dataset, batch_size=batch_size)

    classifier.to(device)
    trainable_params = [p for p in classifier.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=learning_rate)
    criterion = nn.BCELoss()

    train_loss: list[float] = []
    validation_loss: list[float] = []

    for _ in range(epochs):
        classifier.train()
        epoch_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            scores = classifier(batch_x)
            loss = criterion(scores, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item()) * len(batch_x)
        train_loss.append(epoch_loss / n_train)

        if validation_loader is not None:
            classifier.eval()
            total_val_loss = 0.0
            with torch.no_grad():
                for batch_x, batch_y in validation_loader:
                    batch_x = batch_x.to(device)
                    batch_y = batch_y.to(device)
                    loss = criterion(classifier(batch_x), batch_y)
                    total_val_loss += float(loss.item()) * len(batch_x)
            validation_loss.append(total_val_loss / n_validation)

    return TransformerTrainingHistory(
        train_loss=train_loss,
        validation_loss=validation_loss,
    )


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------


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


def predict_regime_score(
    classifier: "WaveletTransformerClassifier",
    x: np.ndarray,
    *,
    device: str | None = None,
) -> np.ndarray:
    """
    Return regime scores in [0, 1] for a batch of wavelet windows.

    Scores above ``classifier.regime_threshold`` indicate a stable
    mean-reverting regime. Typically called with a single window built
    by ``make_latest_wavelet_window`` for live signal generation.
    """
    _require_torch()

    if device is None:
        device = next(classifier.parameters()).device

    classifier.eval()
    with torch.no_grad():
        tensor_x = torch.as_tensor(x, dtype=torch.float32, device=device)
        scores = classifier(tensor_x).cpu().numpy()

    return scores