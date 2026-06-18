"""Lightweight Autoformer-style regime gate for pairs trading.
    spread/ARMA/GARCH features -> decomposition blocks -> auto-correlation blocks
    -> pooled representation -> sigmoid regime score

Output:
    P(favorable mean-reversion regime)

References:
    Autoformer: Decomposition Transformers with Auto-Correlation for
    Long-Term Series Forecasting, Wu et al. (NeurIPS 2021).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Optional, Sequence, Tuple

import numpy as np

try:
    import torch
    from torch import nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
except ImportError as exc:  # pragma: no cover
    raise ImportError("PyTorch is required for pairs_trading.models.autoformer") from exc


# ---------------------------------------------------------------------------
# Data container + scaling
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AutoformerWindowData:
    """Windowed feature matrix for regime classification.

    x shape is (n_windows, input_length, n_features).
    target_index is the timestamp associated with each window target.
    """

    x: np.ndarray
    target_index: object
    feature_names: Sequence[str]
    input_length: int
    scaling: Optional["WindowStandardScaler"] = None

    @property
    def n_samples(self) -> int:
        return int(self.x.shape[0])

    @property
    def n_features(self) -> int:
        return int(self.x.shape[-1])


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


@dataclass
class ClassificationMetrics:
    accuracy: float
    balanced_accuracy: float
    precision: float
    recall: float
    specificity: float
    f1: float
    mcc: float
    npv: float


@dataclass
class TrainHistory:
    train_loss: list[float]
    validation_loss: list[float]
    validation_metrics: Dict[str, float]
    test_metrics: Dict[str, float]
    split_sizes: Tuple[int, int, int]
    pos_weight: float
    scaling: WindowStandardScaler




@dataclass(frozen=True)
class HyperparameterTrial:
    """One validation-only tuning result.

    The tuner chooses a small demo configuration using validation MCC, with
    validation specificity as a tie-breaker. Test metrics are deliberately not
    used for model selection.
    """

    params: Dict[str, Any]
    seed: int
    best_validation_loss: float
    validation_mcc: float
    validation_specificity: float
    validation_epochs: int

@dataclass(frozen=True)
class AutoformerExitDiagnostics:
    """Interpretable decomposition outputs for exit logic.

    These are not extra learned labels. They are structural signals computed
    from the latest spread window using the same Autoformer idea: separate the
    sequence into trend and seasonal/cyclical components.
    """

    seasonal_ratio: float
    trend_ratio: float
    trend_slope: float
    autocorr_strength: float
    trend_against_trade: bool
    exit_suggestion: str


# ---------------------------------------------------------------------------
# Model blocks
# ---------------------------------------------------------------------------


class SeriesDecomposition(nn.Module):
    """Autoformer-style moving-average decomposition.

    Paper idea:
        trend = AvgPool(Padding(x))
        seasonal = x - trend

    Here we use it inside a classifier instead of a decoder forecaster.
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
        # Replicate boundary values to avoid look-ahead and preserve length.
        front = x_t[:, :, 0:1].repeat(1, 1, pad)
        end = x_t[:, :, -1:].repeat(1, 1, pad)
        trend = self.avg(torch.cat([front, x_t, end], dim=-1)).transpose(1, 2)
        seasonal = x - trend
        return seasonal, trend


class AutoCorrelationBlock(nn.Module):
    """Autoformer-inspired autocorrelation block.

    Full Autoformer computes period delays with FFT and performs time-delay
    aggregation, and -- per the paper's Appendix G.1 (Algorithms 3 and 4) --
    does this two different ways depending on whether the model is
    training or being run at inference:
      - training: delays are chosen from the BATCH-AVERAGED correlation,
        one shared set of delays per step, aggregated with ``torch.roll``.
        Cheap, used for speed during training.
      - inference: each sample in the batch picks its OWN top-k delays
        from its own correlation profile, aggregated with ``torch.gather``
        since the shift differs per sample. This matters for a live
        regime score: without it, a single window scored at inference
        would still be implicitly pulled toward whatever delays happened
        to be typical of the training batch, rather than reflecting its
        own periodicity.
    This block implements both modes, switched on ``self.training``,
    rather than always using the cheaper training-mode shortcut.
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
        b, l, _ = x.shape
        return x.view(b, l, self.n_heads, self.head_dim).transpose(1, 2)  # B,H,L,C

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, l, _ = x.shape
        q = self._split(self.q_proj(x))
        k = self._split(self.k_proj(x))
        v = self._split(self.v_proj(x))

        # FFT autocorrelation over time dimension. Shape: B,H,L,C.
        q_fft = torch.fft.rfft(q.float(), dim=2)
        k_fft = torch.fft.rfft(k.float(), dim=2)
        corr = torch.fft.irfft(q_fft * torch.conj(k_fft), n=l, dim=2)
        k_top = min(self.top_k, max(1, l - 1))

        if self.training:
            # Algorithm 3: one delay profile shared by the whole batch.
            delay_scores = corr.mean(dim=(0, 1, 3)).clone()  # (L,)
            delay_scores[0] = -torch.inf  # ignore zero lag
            delays = torch.topk(delay_scores, k=k_top).indices
            weights = torch.softmax(delay_scores[delays], dim=0)

            agg = torch.zeros_like(v)
            for w, tau in zip(weights, delays):
                agg = agg + w * torch.roll(v, shifts=-int(tau.item()), dims=2)
        else:
            # Algorithm 4: each sample in the batch picks its own delays.
            delay_scores = corr.mean(dim=(1, 3)).clone()  # (B, L)
            delay_scores[:, 0] = -torch.inf
            weights, delays = torch.topk(delay_scores, k=k_top, dim=1)  # (B, k_top)
            weights = torch.softmax(weights, dim=1)

            tiled_v = v.repeat(1, 1, 2, 1)  # (B,H,2L,C) -- avoids wraparound in gather
            base_index = torch.arange(l, device=v.device).view(1, 1, l, 1).expand(
                b, self.n_heads, l, self.head_dim
            )
            agg = torch.zeros_like(v)
            for i in range(k_top):
                delay_i = delays[:, i].view(b, 1, 1, 1).expand(b, self.n_heads, l, self.head_dim)
                gather_index = base_index + delay_i
                rolled = torch.gather(tiled_v, dim=2, index=gather_index)
                w_i = weights[:, i].view(b, 1, 1, 1)
                agg = agg + w_i * rolled

        out = agg.transpose(1, 2).contiguous().view(b, l, self.d_model)
        return self.out_proj(self.dropout(out))


class AutoformerEncoderLayer(nn.Module):
    """One compact decomposition + autocorrelation block."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dim_feedforward: int,
        moving_avg: int,
        top_k: int,
        dropout: float,
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


class AutoformerRegimeClassifier(nn.Module):
    """Lightweight Autoformer classifier for regime gating.

    This is intentionally close in size to the repo's TFT/W-Transformer demos.
    It returns logits. Apply sigmoid(logits) for regime probabilities.
    """

    def __init__(
        self,
        input_length: int,
        num_features: int,
        d_model: int = 32,
        n_heads: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 64,
        moving_avg: int = 25,
        top_k: int = 3,
        dropout: float = 0.3,
        regime_threshold: float = 0.5,
    ) -> None:
        super().__init__()
        self.input_length = input_length
        self.num_features = num_features
        self.regime_threshold = regime_threshold
        self.input_proj = nn.Linear(num_features, d_model)
        self.pos = nn.Parameter(torch.zeros(1, input_length, d_model))
        self.layers = nn.ModuleList(
            [
                AutoformerEncoderLayer(
                    d_model=d_model,
                    n_heads=n_heads,
                    dim_feedforward=dim_feedforward,
                    moving_avg=moving_avg,
                    top_k=top_k,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.trend_pool = nn.AdaptiveAvgPool1d(1)
        self.season_pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.LayerNorm(2 * d_model),
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x) + self.pos[:, : x.shape[1], :]
        trend_acc = torch.zeros_like(h)
        for layer in self.layers:
            h, trend = layer(h)
            trend_acc = trend_acc + trend

        seasonal_vec = self.season_pool(h.transpose(1, 2)).squeeze(-1)
        trend_vec = self.trend_pool(trend_acc.transpose(1, 2)).squeeze(-1)
        logits = self.head(torch.cat([seasonal_vec, trend_vec], dim=-1)).squeeze(-1)
        return logits

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(x))


# ---------------------------------------------------------------------------
# Feature/window builders
# ---------------------------------------------------------------------------


def make_autoformer_dataset(
    spread_data,
    arma_result=None,
    input_length: int = 60,
    standardize: bool = False,
) -> AutoformerWindowData:
    """Build Autoformer windows from project SpreadData-like objects.

    Expected fields on spread_data:
        spread: pandas Series
        z_score: pandas Series
    Optional fields on arma_result:
        conditional_volatility
        vol_scaled_z_score
    """

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
    feature_names = list(df.columns)

    x_windows = []
    target_index = []
    values = df.to_numpy(dtype=np.float32)
    for end in range(input_length, len(df)):
        x_windows.append(values[end - input_length : end])
        target_index.append(df.index[end])

    x = np.stack(x_windows).astype(np.float32)
    scaler = None
    if standardize:
        scaler = WindowStandardScaler().fit(x)
        x = scaler.transform(x)

    return AutoformerWindowData(
        x=x,
        target_index=df.index.__class__(target_index),
        feature_names=feature_names,
        input_length=input_length,
        scaling=scaler,
    )


def make_latest_autoformer_window(
    spread_data,
    arma_result=None,
    input_length: int = 60,
    scaling: Optional[WindowStandardScaler] = None,
) -> np.ndarray:
    dataset = make_autoformer_dataset(
        spread_data,
        arma_result,
        input_length=input_length,
        standardize=False,
    )
    latest = dataset.x[-1:]
    if scaling is not None:
        latest = scaling.transform(latest)
    return latest.astype(np.float32)


# ---------------------------------------------------------------------------
# Training/evaluation helpers
# ---------------------------------------------------------------------------


def _compute_metrics(y_true: np.ndarray, prob: np.ndarray, threshold: float = 0.5) -> ClassificationMetrics:
    y_true = y_true.astype(int)
    y_pred = (prob >= threshold).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    eps = 1e-12
    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    npv = tn / max(tn + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, eps)
    balanced_accuracy = 0.5 * (recall + specificity)
    denom = np.sqrt(max((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn), eps))
    mcc = ((tp * tn) - (fp * fn)) / denom
    return ClassificationMetrics(
        accuracy=float(accuracy),
        balanced_accuracy=float(balanced_accuracy),
        precision=float(precision),
        recall=float(recall),
        specificity=float(specificity),
        f1=float(f1),
        mcc=float(mcc),
        npv=float(npv),
    )


def purged_chronological_split(
    n: int,
    positions: Optional[np.ndarray] = None,
    embargo: int = 0,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Chronological split with a real, position-aware embargo.

    ``positions`` should be each window's actual trading-day position in
    the underlying (pre-filter) series -- NOT just its row index in the
    labeled array, which can have gaps (e.g. days with no entry signal).
    The embargo is enforced in those real units: a window is kept in
    train/val only if it is more than ``embargo`` actual trading days
    away from the val/test boundary, not just ``embargo`` rows away in a
    possibly-gappy array. If ``positions`` is omitted, row index is used
    as a fallback -- only correct when the labeled array has no gaps.
    """
    if positions is None:
        positions = np.arange(n)
    positions = np.asarray(positions)
    if len(positions) != n:
        raise ValueError(f"positions length ({len(positions)}) must match n ({n})")

    idx        = np.arange(n)
    test_start = int(n * (1.0 - test_fraction))
    val_start  = int(test_start - n * val_fraction)
    val_start  = max(0, min(val_start, test_start))

    val_start_pos  = positions[val_start] if val_start < n else positions[-1]
    test_start_pos = positions[test_start] if test_start < n else positions[-1]

    train_idx = idx[:val_start][positions[:val_start] <= val_start_pos - embargo]
    val_idx   = idx[val_start:test_start][
        positions[val_start:test_start] <= test_start_pos - embargo
    ]
    test_idx  = idx[test_start:]

    return train_idx, val_idx, test_idx


def _window_summary_features(x: np.ndarray) -> np.ndarray:
    """Per-window summaries: last value + mean of each channel -> (N, 2*F)."""
    last = x[:, -1, :]
    mean = x.mean(axis=1)
    return np.concatenate([last, mean], axis=1)


def _fit_logistic_numpy(
    X: np.ndarray, y: np.ndarray, *, lr: float = 0.1, epochs: int = 2000, l2: float = 1e-3,
) -> np.ndarray:
    """Tiny L2-regularised logistic regression (no sklearn dependency)."""
    Xb = np.concatenate([X, np.ones((len(X), 1))], axis=1)
    w  = np.zeros(Xb.shape[1])
    for _ in range(epochs):
        p    = 1.0 / (1.0 + np.exp(-Xb @ w))
        grad = Xb.T @ (p - y) / len(y) + l2 * np.concatenate([w[:-1], [0.0]])
        w   -= lr * grad
    return w


def evaluate_baselines(
    data: AutoformerWindowData,
    labels: np.ndarray,
    *,
    window_positions: np.ndarray,
    embargo: int,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
) -> Dict[str, Dict[str, float]]:
    """
    Trivial baselines on the IDENTICAL purged split the classifier uses.

      all_ones : predict favorable regime for every window -- the bar
                 positive-class F1 must clear with zero intelligence.
      logistic : logistic regression on per-window summary features
                 (last + mean of each channel) -- the bar any deep model
                 must clear to justify its parameters.

    Self-contained on purpose: an earlier version of the demo imported
    this from ``pairs_trading.models.tft``, which doesn't exist on this
    branch on its own and crashed the demo at import time the moment TFT
    wasn't also present in the same checkout.
    """
    train_idx, _val_idx, test_idx = purged_chronological_split(
        len(labels), window_positions, embargo, val_fraction, test_fraction,
    )
    y_train, y_test = labels[train_idx], labels[test_idx]

    results = {"all_ones": _compute_metrics(y_test, np.ones(len(test_idx))).__dict__}

    feats  = _window_summary_features(data.x)
    scaler = WindowStandardScaler().fit(feats[train_idx][:, None, :])
    f_train = scaler.transform(feats[train_idx][:, None, :])[:, 0, :]
    f_test  = scaler.transform(feats[test_idx][:, None, :])[:, 0, :]

    w = _fit_logistic_numpy(f_train, y_train.astype(float))
    p_test = 1.0 / (1.0 + np.exp(
        -(np.concatenate([f_test, np.ones((len(f_test), 1))], axis=1) @ w)
    ))
    results["logistic"] = _compute_metrics(y_test, p_test).__dict__

    return results


def train_autoformer_classifier(
    model: AutoformerRegimeClassifier,
    data: AutoformerWindowData,
    labels: np.ndarray,
    window_positions: Optional[np.ndarray] = None,
    forward_window: int = 1,
    epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    early_stopping_patience: int = 10,
    device: str = "cpu",
    seed: int = 0,
) -> TrainHistory:
    torch.manual_seed(seed)
    np.random.seed(seed)

    embargo = data.input_length + forward_window
    train_idx, val_idx, test_idx = purged_chronological_split(
        len(labels), window_positions, embargo, val_fraction, test_fraction
    )
    scaler = WindowStandardScaler().fit(data.x[train_idx])
    x_scaled = scaler.transform(data.x)

    x_train = torch.as_tensor(x_scaled[train_idx], dtype=torch.float32)
    y_train = torch.as_tensor(labels[train_idx], dtype=torch.float32)
    x_val = torch.as_tensor(x_scaled[val_idx], dtype=torch.float32)
    y_val = torch.as_tensor(labels[val_idx], dtype=torch.float32)
    x_test = torch.as_tensor(x_scaled[test_idx], dtype=torch.float32)
    y_test_np = labels[test_idx].astype(int)

    n_pos = float(y_train.sum().item())
    n_neg = float(len(y_train) - n_pos)
    pos_weight_value = n_neg / max(n_pos, 1.0)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(pos_weight_value, device=device)
    )
    # Adam is the default optimizer for this demo because it is standard for
    # Transformer-style time-series models and is also used in the Autoformer
    # paper's experimental setup.
    opt = torch.optim.Adam(model.parameters(), lr=learning_rate)

    model.to(device)
    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)

    train_loss: list[float] = []
    val_loss: list[float] = []
    best_val = float("inf")
    best_state = None
    patience = 0

    for _epoch in range(epochs):
        model.train()
        total = 0.0
        count = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float(loss.item()) * len(xb)
            count += len(xb)
        train_loss.append(total / max(count, 1))

        model.eval()
        with torch.no_grad():
            if len(x_val) > 0:
                v_loss = criterion(model(x_val.to(device)), y_val.to(device)).item()
            else:
                v_loss = train_loss[-1]
        val_loss.append(float(v_loss))

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
        if len(x_val) > 0:
            val_probs = torch.sigmoid(model(x_val.to(device))).cpu().numpy()
        else:
            val_probs = np.array([], dtype=float)
        test_probs = torch.sigmoid(model(x_test.to(device))).cpu().numpy()

    if len(x_val) > 0:
        validation_metrics = _compute_metrics(
            labels[val_idx].astype(int),
            val_probs,
            threshold=model.regime_threshold,
        )
    else:
        validation_metrics = ClassificationMetrics(
            accuracy=0.0,
            balanced_accuracy=0.0,
            precision=0.0,
            recall=0.0,
            specificity=0.0,
            f1=0.0,
            mcc=0.0,
            npv=0.0,
        )

    test_metrics = _compute_metrics(y_test_np, test_probs, threshold=model.regime_threshold)

    return TrainHistory(
        train_loss=train_loss,
        validation_loss=val_loss,
        validation_metrics=validation_metrics.__dict__,
        test_metrics=test_metrics.__dict__,
        split_sizes=(len(train_idx), len(val_idx), len(test_idx)),
        pos_weight=float(pos_weight_value),
        scaling=scaler,
    )


def evaluate_autoformer_multi_seed(
    make_model: Callable[[], AutoformerRegimeClassifier],
    data: AutoformerWindowData,
    labels: np.ndarray,
    seeds: Iterable[int] = (0, 1, 2),
    **train_kwargs,
) -> Tuple[list[TrainHistory], Dict[str, Dict[str, float]]]:
    histories = []
    for seed in seeds:
        history = train_autoformer_classifier(
            make_model(),
            data,
            labels,
            seed=seed,
            **train_kwargs,
        )
        histories.append(history)

    keys = histories[0].test_metrics.keys()
    aggregate = {}
    for key in keys:
        vals = np.array([h.test_metrics[key] for h in histories], dtype=float)
        aggregate[key] = {
            "mean": float(vals.mean()),
            "std": float(vals.std(ddof=0)),
        }
    return histories, aggregate


def tune_autoformer_hyperparameters(
    make_model_from_params: Callable[[Dict[str, Any]], AutoformerRegimeClassifier],
    data: AutoformerWindowData,
    labels: np.ndarray,
    param_grid: Sequence[Dict[str, Any]],
    seed: int = 0,
    **train_kwargs,
) -> Tuple[Dict[str, Any], list[HyperparameterTrial]]:
    """Run a small validation-only hyperparameter search.

    This is intentionally lightweight: it answers code-review concerns without
    turning the demo into a long research sweep. The chosen configuration is
    selected by validation MCC, with validation specificity as a tie-breaker. The
    test set is never used for model selection.
    """

    if not param_grid:
        raise ValueError("param_grid must contain at least one configuration.")

    trials: list[HyperparameterTrial] = []
    for params in param_grid:
        history = train_autoformer_classifier(
            make_model_from_params(params),
            data,
            labels,
            seed=seed,
            **train_kwargs,
        )
        best_validation_loss = min(history.validation_loss)
        trials.append(
            HyperparameterTrial(
                params=dict(params),
                seed=seed,
                best_validation_loss=float(best_validation_loss),
                validation_mcc=float(history.validation_metrics["mcc"]),
                validation_specificity=float(history.validation_metrics["specificity"]),
                validation_epochs=len(history.validation_loss),
            )
        )

    best = max(
        trials,
        key=lambda trial: (
            trial.validation_mcc,
            trial.validation_specificity,
            -trial.best_validation_loss,
        ),
    )
    return dict(best.params), trials


def compute_exit_diagnostics_from_spread(
    spread_data,
    input_length: int = 60,
    moving_avg: int = 25,
    regime_score: Optional[float] = None,
    current_z: Optional[float] = None,
    target_z: float = 0.5,
    stop_z: float = 3.0,
    min_seasonal_ratio: float = 0.40,
    max_bad_trend_ratio: float = 0.60,
    min_autocorr_strength: float = 0.10,
    regime_exit_threshold: float = 0.35,
) -> AutoformerExitDiagnostics:
    """Compute Autoformer-style structural outputs for trade exit logic.

    The paper decomposes a time series into seasonal and trend-cyclical parts
    using a moving average block. For a regime gate, these decomposed parts can
    be turned into interpretable exit diagnostics:

        seasonal_ratio    high => spread is mostly cyclical/mean-reverting
        trend_ratio       high => spread is mostly drifting/trending
        trend_slope       direction of the slow spread drift
        autocorr_strength high => repeated cycle still exists
    
        ```text
        Seasonal ratio      : share of decomposed energy in the seasonal/cyclical part
        Trend ratio         : share of decomposed energy in the trend/drift part
        Trend slope         : direction and strength of slow spread drift
        Trend against trade : whether the drift is moving away from mean reversion
        Autocorr strength   : simple cycle-confidence proxy from seasonal autocorrelation
        Exit suggestion     : HOLD or reason to exit
        ```
    The function is intentionally model-independent, so it can be used even
    before/after the classifier head is trained. It uses only past/latest spread
    values, so it is safe for live diagnostics.
    """

    spread = spread_data.spread.dropna().astype(float)
    if len(spread) < max(5, input_length):
        raise ValueError("Not enough spread observations for exit diagnostics.")

    window = spread.iloc[-input_length:].to_numpy(dtype=float)
    l = len(window)
    kernel = max(3, min(int(moving_avg), l))

    # Causal moving-average trend: no future values are used.
    trend = np.empty_like(window)
    for i in range(l):
        start = max(0, i - kernel + 1)
        trend[i] = window[start : i + 1].mean()
    seasonal = window - trend

    # Energies are computed after centering so a nonzero spread level does not
    # dominate the ratio.
    seasonal_centered = seasonal - seasonal.mean()
    trend_centered = trend - trend.mean()
    seasonal_energy = float(np.mean(seasonal_centered ** 2))
    trend_energy = float(np.mean(trend_centered ** 2))
    denom = seasonal_energy + trend_energy + 1e-12
    seasonal_ratio = seasonal_energy / denom
    trend_ratio = trend_energy / denom

    x_axis = np.arange(l, dtype=float)
    trend_slope = float(np.polyfit(x_axis, trend, 1)[0]) if l >= 2 else 0.0

    # Autocorrelation strength of the seasonal component. Use positive max
    # correlation across reasonable lags as a simple cycle confidence measure.
    s = seasonal_centered
    if np.std(s) < 1e-12:
        autocorr_strength = 0.0
    else:
        max_lag = min(l // 2, 30)
        vals = []
        for lag in range(2, max_lag + 1):
            a = s[:-lag]
            b = s[lag:]
            if len(a) > 3 and np.std(a) > 1e-12 and np.std(b) > 1e-12:
                vals.append(float(np.corrcoef(a, b)[0, 1]))
        autocorr_strength = max([v for v in vals if np.isfinite(v)] + [0.0])

    if current_z is None:
        z_series = getattr(spread_data, "z_score", None)
        current_z = float(z_series.dropna().iloc[-1]) if z_series is not None else 0.0

    # If z and slow trend move in the same sign direction, the spread is drifting
    # further away from equilibrium for a standard mean-reversion trade.
    trend_against_trade = bool(current_z * trend_slope > 0)

    if abs(current_z) <= target_z:
        suggestion = "EXIT: target reached"
    elif abs(current_z) >= stop_z:
        suggestion = "EXIT: stop loss reached"
    elif regime_score is not None and regime_score < regime_exit_threshold:
        suggestion = "EXIT: regime deteriorated"
    elif seasonal_ratio < min_seasonal_ratio:
        suggestion = "EXIT: mean-reversion cycle weakened"
    elif trend_ratio > max_bad_trend_ratio and trend_against_trade:
        suggestion = "EXIT: trend drifting against trade"
    elif autocorr_strength < min_autocorr_strength:
        suggestion = "EXIT: autocorrelation/cycle weak"
    else:
        suggestion = "HOLD"

    return AutoformerExitDiagnostics(
        seasonal_ratio=float(seasonal_ratio),
        trend_ratio=float(trend_ratio),
        trend_slope=float(trend_slope),
        autocorr_strength=float(autocorr_strength),
        trend_against_trade=trend_against_trade,
        exit_suggestion=suggestion,
    )


def predict_regime_score(
    model: AutoformerRegimeClassifier,
    window: np.ndarray,
    device: str = "cpu",
) -> np.ndarray:
    model.eval().to(device)
    x = torch.as_tensor(window, dtype=torch.float32, device=device)
    with torch.no_grad():
        return torch.sigmoid(model(x)).cpu().numpy()
