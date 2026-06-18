"""Autoformer forecaster -- a separate, secondary capability from the
regime-gate classifier in ``autoformer.py``.

    spread/ARMA/GARCH features -> encoder (seasonal modeling) ->
    decoder (cross-attention to encoder + progressive trend accumulation)
    -> multi-step forecast of the target feature

Output:
    A ``horizon``-day-ahead forecast of one chosen feature (default:
    ``spread``), in real units.

References:
    Autoformer: Decomposition Transformers with Auto-Correlation for
    Long-Term Series Forecasting, Wu et al. (NeurIPS 2021).

Why this is its own file rather than added to ``autoformer.py``
------------------------------------------------------------------
This is intentionally self-contained: it duplicates ``SeriesDecomposition``,
the correlation block, and the encoder layer rather than importing them
from ``autoformer.py``. That means it can be dropped in alongside an
existing classifier-only ``autoformer.py`` with zero coupling, nothing
here can affect that file, and nothing there needs to change for this to
work. The only thing genuinely new relative to the classifier's encoder is
the decoder: cross-attention against the encoder's output, and trend
accumulated (in real feature units, not the embedded ``d_model`` space)
across decoder layers, which is what turns this from a pooled
classification representation into an actual multi-step forecast.

Scope note: the paper predicts 96-720 steps; this project only needs a
short lookahead. ``horizon`` is meant to roughly match the label window H
already being swept for the classifier (typically single digits to ~10),
not the paper's long-horizon scale.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError as exc:  # pragma: no cover
    raise ImportError("PyTorch is required for pairs_trading.models.autoformer_forecaster") from exc


# ---------------------------------------------------------------------------
# Data container + scaling (duplicated from autoformer.py on purpose -- see
# module docstring for why this file doesn't import from it)
# ---------------------------------------------------------------------------


@dataclass
class WindowStandardScaler:
    """Train-only standardizer for 3D window data."""

    mean_: Optional[np.ndarray] = None
    std_: Optional[np.ndarray] = None

    def fit(self, x: np.ndarray) -> "WindowStandardScaler":
        flat = x.reshape(-1, x.shape[-1])
        self.mean_ = flat.mean(axis=0, keepdims=True)
        self.std_ = flat.std(axis=0, keepdims=True)
        self.std_ = np.where(self.std_ < 1e-8, 1.0, self.std_)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise ValueError("Scaler must be fit before transform.")
        return ((x - self.mean_) / self.std_).astype(np.float32)

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        return self.fit(x).transform(x)


def _build_feature_frame(spread_data, arma_result=None):
    """Same feature set as autoformer.py's classifier dataset builder, so a
    forecaster and a classifier built from the same data are directly
    comparable. Duplicated rather than imported -- see module docstring."""
    import pandas as pd

    spread = spread_data.spread.astype(float).rename("spread")
    z = spread_data.z_score.astype(float).rename("z_score")
    ret_1 = spread.diff().rename("spread_return")
    momentum_5 = spread.diff(5).rename("momentum_5")
    vol_20 = ret_1.rolling(20, min_periods=5).std().rename("vol_20")

    features = [spread, z, ret_1, momentum_5, vol_20]

    if arma_result is not None and getattr(arma_result, "conditional_volatility", None) is not None:
        features.append(arma_result.conditional_volatility.astype(float).rename("conditional_volatility"))
    if arma_result is not None and getattr(arma_result, "vol_scaled_z_score", None) is not None:
        features.append(arma_result.vol_scaled_z_score.astype(float).rename("vol_scaled_z_score"))

    df = pd.concat(features, axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    return df, list(df.columns)


@dataclass(frozen=True)
class AutoformerForecastData:
    """Supervised forecasting windows: window ``i`` covers
    ``[i, i+input_length)`` and its target is the next ``horizon`` days,
    for every feature. ``target_index`` says which feature column is the
    one actually trained against and reported."""

    x: np.ndarray  # (N, input_length, num_features)
    y: np.ndarray  # (N, horizon, num_features) -- true future values, unscaled
    feature_names: Sequence[str]
    target_index: int
    input_length: int
    horizon: int
    target_timestamps: object  # timestamp of the FIRST forecasted day per window

    @property
    def n_samples(self) -> int:
        return int(self.x.shape[0])

    @property
    def n_features(self) -> int:
        return int(self.x.shape[-1])


def make_autoformer_forecast_dataset(
    spread_data,
    arma_result=None,
    input_length: int = 60,
    horizon: int = 8,
    target: str = "spread",
) -> AutoformerForecastData:
    """Build supervised forecasting windows. ``horizon`` is meant to be
    short, matching the label window H already swept for the classifier in
    ``autoformer.py``, not the original paper's 96-720 step scale."""
    if input_length < 2:
        raise ValueError("input_length must be at least 2")
    if horizon < 1:
        raise ValueError("horizon must be at least 1")

    df, feature_names = _build_feature_frame(spread_data, arma_result)
    if target not in feature_names:
        raise ValueError(f"target {target!r} not in feature_names {feature_names}")
    target_index = feature_names.index(target)

    values = df.to_numpy(dtype=np.float32)
    n = len(values)

    x_windows, y_windows, target_timestamps = [], [], []
    for end in range(input_length, n - horizon + 1):
        x_windows.append(values[end - input_length : end])
        y_windows.append(values[end : end + horizon])
        target_timestamps.append(df.index[end])

    if not x_windows:
        raise ValueError(
            "not enough observations for the requested input_length and horizon"
        )

    return AutoformerForecastData(
        x=np.stack(x_windows).astype(np.float32),
        y=np.stack(y_windows).astype(np.float32),
        feature_names=feature_names,
        target_index=target_index,
        input_length=input_length,
        horizon=horizon,
        target_timestamps=df.index.__class__(target_timestamps),
    )


def make_latest_autoformer_forecast_window(
    spread_data,
    arma_result=None,
    input_length: int = 60,
    horizon: int = 8,
    target: str = "spread",
) -> np.ndarray:
    """Single model-ready window from the latest data, shape
    (1, input_length, num_features), unscaled."""
    dataset = make_autoformer_forecast_dataset(
        spread_data, arma_result, input_length=input_length, horizon=horizon, target=target,
    )
    return dataset.x[-1:]


# ---------------------------------------------------------------------------
# Model blocks
# ---------------------------------------------------------------------------


class SeriesDecomposition(nn.Module):
    """Autoformer-style moving-average decomposition.

    Paper idea:
        trend = AvgPool(Padding(x))
        seasonal = x - trend
    """

    def __init__(self, kernel_size: int = 25) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd so padding is symmetric.")
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (B, L, D)
        pad = (self.kernel_size - 1) // 2
        x_t = x.transpose(1, 2)  # (B, D, L)
        front = x_t[:, :, 0:1].repeat(1, 1, pad)
        end = x_t[:, :, -1:].repeat(1, 1, pad)
        trend = self.avg(torch.cat([front, x_t, end], dim=-1)).transpose(1, 2)
        seasonal = x - trend
        return seasonal, trend


class AutoCorrelationBlock(nn.Module):
    """Autoformer-inspired autocorrelation block, with both self-attention
    (default) and cross-attention (pass ``kv``) support, the latter is what
    the decoder's encoder-attention step needs and the classifier's encoder
    in ``autoformer.py`` doesn't, which is the one real difference between
    this copy and that file's version.

    Implements the paper's Appendix G.1 dual aggregation (Algorithms 3/4):
    training uses batch-shared delays via ``torch.roll`` (cheap); eval uses
    per-sample delays via ``torch.gather`` (so a single live window isn't
    pulled toward whatever delays were typical of a training batch).
    """

    def __init__(self, d_model: int, n_heads: int = 4, top_k: int = 3, dropout: float = 0.1) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads.")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.top_k = top_k
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def _split(self, x: torch.Tensor) -> torch.Tensor:
        b, seq_len, _ = x.shape
        return x.view(b, seq_len, self.n_heads, self.head_dim).transpose(1, 2)  # B,H,L,C

    def _resize_kv(self, kv: torch.Tensor, target_len: int) -> torch.Tensor:
        """Match kv's length to the query length via truncation (keep the
        most recent steps) or zero-padding (paper's Algorithm 2: "Resize is
        truncation or zero filling")."""
        b, l_kv, d = kv.shape
        if l_kv == target_len:
            return kv
        if l_kv > target_len:
            return kv[:, -target_len:, :]
        pad = torch.zeros(b, target_len - l_kv, d, device=kv.device, dtype=kv.dtype)
        return torch.cat([pad, kv], dim=1)

    def forward(self, x: torch.Tensor, kv: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Self-attention when ``kv`` is omitted; cross-attention when
        given (queries from ``x``, keys/values from ``kv``, resized to
        ``x``'s length first)."""
        b, seq_len, _ = x.shape
        if kv is None:
            kv = x
        else:
            kv = self._resize_kv(kv, seq_len)

        q = self._split(self.q_proj(x))
        k = self._split(self.k_proj(kv))
        v = self._split(self.v_proj(kv))

        q_fft = torch.fft.rfft(q.float(), dim=2)
        k_fft = torch.fft.rfft(k.float(), dim=2)
        corr = torch.fft.irfft(q_fft * torch.conj(k_fft), n=seq_len, dim=2)
        k_top = min(self.top_k, max(1, seq_len - 1))

        if self.training:
            delay_scores = corr.mean(dim=(0, 1, 3)).clone()  # (L,)
            delay_scores[0] = -torch.inf  # ignore zero lag
            delays = torch.topk(delay_scores, k=k_top).indices
            weights = torch.softmax(delay_scores[delays], dim=0)

            agg = torch.zeros_like(v)
            for w, tau in zip(weights, delays):
                agg = agg + w * torch.roll(v, shifts=-int(tau.item()), dims=2)
        else:
            delay_scores = corr.mean(dim=(1, 3)).clone()  # (B, L)
            delay_scores[:, 0] = -torch.inf
            weights, delays = torch.topk(delay_scores, k=k_top, dim=1)  # (B, k_top)
            weights = torch.softmax(weights, dim=1)

            tiled_v = v.repeat(1, 1, 2, 1)  # (B,H,2L,C)
            base_index = torch.arange(seq_len, device=v.device).view(1, 1, seq_len, 1).expand(
                b, self.n_heads, seq_len, self.head_dim
            )
            agg = torch.zeros_like(v)
            for i in range(k_top):
                delay_i = delays[:, i].view(b, 1, 1, 1).expand(b, self.n_heads, seq_len, self.head_dim)
                gather_index = base_index + delay_i
                rolled = torch.gather(tiled_v, dim=2, index=gather_index)
                w_i = weights[:, i].view(b, 1, 1, 1)
                agg = agg + w_i * rolled

        out = agg.transpose(1, 2).contiguous().view(b, seq_len, self.d_model)
        return self.out_proj(self.dropout(out))


class AutoformerEncoderLayer(nn.Module):
    """One compact decomposition + autocorrelation block. Identical to the
    classifier's encoder layer in autoformer.py -- the encoder's job
    (seasonal pattern modeling, discarding trend at every step) is the
    same whether you're feeding a classifier head or a decoder."""

    def __init__(
        self, d_model: int, n_heads: int, dim_feedforward: int, moving_avg: int, top_k: int, dropout: float,
    ) -> None:
        super().__init__()
        self.decomp1 = SeriesDecomposition(moving_avg)
        self.autocorr = AutoCorrelationBlock(d_model, n_heads, top_k, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.decomp2 = SeriesDecomposition(moving_avg)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        y = self.norm1(x + self.dropout(self.autocorr(x)))
        seasonal, trend1 = self.decomp1(y)
        z = self.norm2(seasonal + self.dropout(self.ff(seasonal)))
        seasonal, trend2 = self.decomp2(z)
        return seasonal, trend1 + trend2


class AutoformerDecoderLayer(nn.Module):
    """Decoder layer: inner self-correlation, cross-correlation against the
    encoder's output, then feedforward -- each followed by a decomposition,
    with trend accumulated across all three (Eq. 4). Unlike the encoder,
    trend is NOT thrown away here: each step's trend is projected from
    ``d_model`` down to ``num_features`` (the accumulator lives in the same
    units as the actual forecast target) and added to a running total --
    this is the piece that turns seasonal modeling into a usable forecast.
    """

    def __init__(
        self, d_model: int, n_heads: int, dim_feedforward: int, moving_avg: int, top_k: int,
        dropout: float, num_features: int,
    ) -> None:
        super().__init__()
        self.self_attn = AutoCorrelationBlock(d_model, n_heads, top_k, dropout)
        self.decomp1 = SeriesDecomposition(moving_avg)
        self.norm1 = nn.LayerNorm(d_model)
        self.cross_attn = AutoCorrelationBlock(d_model, n_heads, top_k, dropout)
        self.decomp2 = SeriesDecomposition(moving_avg)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )
        self.decomp3 = SeriesDecomposition(moving_avg)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.trend_proj1 = nn.Linear(d_model, num_features)
        self.trend_proj2 = nn.Linear(d_model, num_features)
        self.trend_proj3 = nn.Linear(d_model, num_features)

    def forward(self, x: torch.Tensor, encoder_out: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (seasonal_output, trend_delta). ``trend_delta`` is in
        ``num_features`` units, ready to add to a running accumulator."""
        y = self.norm1(x + self.dropout(self.self_attn(x)))
        seasonal, trend1 = self.decomp1(y)

        z = self.norm2(seasonal + self.dropout(self.cross_attn(seasonal, encoder_out)))
        seasonal, trend2 = self.decomp2(z)

        w = self.norm3(seasonal + self.dropout(self.ff(seasonal)))
        seasonal, trend3 = self.decomp3(w)

        trend_delta = (
            self.trend_proj1(trend1) + self.trend_proj2(trend2) + self.trend_proj3(trend3)
        )
        return seasonal, trend_delta


class AutoformerForecaster(nn.Module):
    """
    The paper's actual forecasting mechanism: encoder + decoder with
    cross-attention and progressive trend accumulation.

    forward(x) returns the full multivariate forecast, shape
    ``(batch, horizon, num_features)``; ``target_index`` says which
    feature column is the actual quantity being forecast (e.g. "spread").
    Train with MSE against true future values on that column; see
    ``train_autoformer_forecaster``.
    """

    def __init__(
        self,
        input_length: int,
        num_features: int,
        horizon: int = 8,
        target_index: int = 0,
        d_model: int = 32,
        n_heads: int = 4,
        num_encoder_layers: int = 2,
        num_decoder_layers: int = 1,
        dim_feedforward: int = 64,
        moving_avg: int = 25,
        top_k: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        half = input_length // 2
        if half < 1:
            raise ValueError("input_length must be at least 2")
        if not (0 <= target_index < num_features):
            raise ValueError("target_index must index an existing feature column")

        self.input_length = input_length
        self.num_features = num_features
        self.horizon = horizon
        self.target_index = target_index
        decoder_length = half + horizon

        self.input_proj = nn.Linear(num_features, d_model)
        self.enc_pos = nn.Parameter(torch.zeros(1, input_length, d_model))
        self.encoder_layers = nn.ModuleList(
            [
                AutoformerEncoderLayer(d_model, n_heads, dim_feedforward, moving_avg, top_k, dropout)
                for _ in range(num_encoder_layers)
            ]
        )

        self.init_decomp = SeriesDecomposition(moving_avg)
        self.dec_input_proj = nn.Linear(num_features, d_model)
        self.dec_pos = nn.Parameter(torch.zeros(1, decoder_length, d_model))
        self.decoder_layers = nn.ModuleList(
            [
                AutoformerDecoderLayer(
                    d_model, n_heads, dim_feedforward, moving_avg, top_k, dropout, num_features
                )
                for _ in range(num_decoder_layers)
            ]
        )
        self.seasonal_proj = nn.Linear(d_model, num_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, input_length, num_features) raw features ->
        (batch, horizon, num_features) forecast."""
        b = x.shape[0]
        half = self.input_length // 2

        h = self.input_proj(x) + self.enc_pos
        for layer in self.encoder_layers:
            h, _ = layer(h)  # trend discarded, same as the classifier's encoder
        encoder_out = h

        recent_half = x[:, -half:, :]
        seasonal_half, trend_half = self.init_decomp(recent_half)
        zero_pad = torch.zeros(b, self.horizon, self.num_features, device=x.device, dtype=x.dtype)
        mean_pad = x.mean(dim=1, keepdim=True).repeat(1, self.horizon, 1)
        seasonal_init = torch.cat([seasonal_half, zero_pad], dim=1)
        trend_acc = torch.cat([trend_half, mean_pad], dim=1)

        dec_h = self.dec_input_proj(seasonal_init) + self.dec_pos
        for layer in self.decoder_layers:
            dec_h, trend_delta = layer(dec_h, encoder_out)
            trend_acc = trend_acc + trend_delta

        seasonal_out = self.seasonal_proj(dec_h)
        forecast_full = seasonal_out + trend_acc
        return forecast_full[:, -self.horizon:, :]

    def target_series(self, x: torch.Tensor) -> torch.Tensor:
        """Convenience: forecast for just the target feature column,
        shape (batch, horizon)."""
        return self.forward(x)[:, :, self.target_index]


# ---------------------------------------------------------------------------
# Training/inference
# ---------------------------------------------------------------------------


@dataclass
class ForecastTrainHistory:
    train_loss: list
    validation_loss: list
    test_mse: float
    test_mae: float
    naive_test_mse: float
    split_sizes: Tuple[int, int, int]
    scaling: WindowStandardScaler


def train_autoformer_forecaster(
    model: AutoformerForecaster,
    data: AutoformerForecastData,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    early_stopping_patience: int = 10,
    device: str = "cpu",
    seed: int = 0,
) -> ForecastTrainHistory:
    """Train with MSE against the true future values of
    ``data.target_index``'s feature column. Plain chronological
    train/val/test split (no purged embargo): every window here has a
    real forecast target, so there are no label gaps to purge around."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    n = data.n_samples
    n_test = max(1, int(n * test_fraction))
    n_val = max(1, int(n * val_fraction))
    n_train = n - n_val - n_test
    if n_train < 1:
        raise ValueError("Not enough windows for the requested split fractions")

    train_idx = np.arange(0, n_train)
    val_idx = np.arange(n_train, n_train + n_val)
    test_idx = np.arange(n_train + n_val, n)

    scaler = WindowStandardScaler().fit(data.x[train_idx])
    x_scaled = scaler.transform(data.x)

    mean_t = float(scaler.mean_[0, data.target_index])
    std_t = float(scaler.std_[0, data.target_index])
    y_target = data.y[:, :, data.target_index]
    y_scaled = (y_target - mean_t) / std_t

    x_train = torch.as_tensor(x_scaled[train_idx], dtype=torch.float32)
    y_train = torch.as_tensor(y_scaled[train_idx], dtype=torch.float32)
    x_val = torch.as_tensor(x_scaled[val_idx], dtype=torch.float32)
    y_val = torch.as_tensor(y_scaled[val_idx], dtype=torch.float32)
    x_test = torch.as_tensor(x_scaled[test_idx], dtype=torch.float32)
    y_test = torch.as_tensor(y_scaled[test_idx], dtype=torch.float32)

    criterion = nn.MSELoss()
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=learning_rate)
    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)

    train_loss: list = []
    validation_loss: list = []
    best_val = float("inf")
    best_state = None
    patience = 0

    for _epoch in range(epochs):
        model.train()
        total, count = 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            pred = model.target_series(xb)
            loss = criterion(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float(loss.item()) * len(xb)
            count += len(xb)
        train_loss.append(total / max(count, 1))

        model.eval()
        with torch.no_grad():
            v_pred = model.target_series(x_val.to(device))
            v_loss = criterion(v_pred, y_val.to(device)).item()
        validation_loss.append(float(v_loss))

        if v_loss < best_val:
            best_val = v_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= early_stopping_patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        test_pred_scaled = model.target_series(x_test.to(device)).cpu().numpy()
    test_true_scaled = y_test.numpy()

    test_pred = test_pred_scaled * std_t + mean_t
    test_true = test_true_scaled * std_t + mean_t
    test_mse = float(np.mean((test_pred - test_true) ** 2))
    test_mae = float(np.mean(np.abs(test_pred - test_true)))

    last_known = data.x[test_idx][:, -1, data.target_index]
    naive_pred = np.repeat(last_known[:, None], data.horizon, axis=1)
    naive_test_mse = float(np.mean((naive_pred - data.y[test_idx][:, :, data.target_index]) ** 2))

    return ForecastTrainHistory(
        train_loss=train_loss,
        validation_loss=validation_loss,
        test_mse=test_mse,
        test_mae=test_mae,
        naive_test_mse=naive_test_mse,
        split_sizes=(len(train_idx), len(val_idx), len(test_idx)),
        scaling=scaler,
    )


def predict_spread_forecast(
    model: AutoformerForecaster,
    window: np.ndarray,
    scaling: WindowStandardScaler,
    device: str = "cpu",
) -> np.ndarray:
    """Forecast for the target feature only, in real (unscaled) units,
    shape (batch, horizon)."""
    model.eval().to(device)
    x_scaled = scaling.transform(window)
    x = torch.as_tensor(x_scaled, dtype=torch.float32, device=device)
    mean_t = float(scaling.mean_[0, model.target_index])
    std_t = float(scaling.std_[0, model.target_index])
    with torch.no_grad():
        pred_scaled = model.target_series(x).cpu().numpy()
    return pred_scaled * std_t + mean_t
