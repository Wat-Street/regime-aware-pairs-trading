"""Temporal Fusion Transformer regime gate classifier for pair-spread time series.

Adapts Lim et al. (2021), arXiv:1912.09363, for binary regime classification.

Methodology fixes in this version (vs. first draft):

  LEAKAGE
  - Purged & embargoed chronological splits: windows within
    ``embargo = input_length + forward_window`` trading days of a split
    boundary are dropped, so no training input overlaps validation data
    and no training label is computed from validation-period prices.
  - Feature scaling is fit on the TRAIN slice only (was: full dataset).
  - Rolling half-life no longer backfills with a full-sample statistic
    (was look-ahead). Invalid fits are forward-filled causally; the
    warm-up period is dropped.

  EVALUATION
  - Full confusion-matrix metrics. For a regime GATE the headline numbers
    are specificity (negative-class recall: of the bad regimes, how many
    did we block?) and MCC -- not positive-class F1, which a trivial
    all-ones classifier maximises when labels are imbalanced positive.
  - Multi-seed evaluation (``evaluate_tft_multi_seed``) reporting
    mean +/- std, because single-run numbers at N~300 have error bars
    wider than typical architecture differences.
  - Trivial baselines (``evaluate_baselines``): all-ones and a logistic
    regression on simple window-summary features. If the TFT does not
    beat these, that is the finding.

  TRAINING
  - BCEWithLogitsLoss with native pos_weight (head outputs logits;
    sigmoid applied only at inference) for numerical stability.
  - shuffle=True on the train loader. The chronological SPLIT is what
    prevents leakage; ordered mini-batches only hurt optimisation.
  - Default dropout 0.3 per the paper's small/noisy-dataset finding
    (Volatility dataset, Table 1).
  - pos_weight capped at MAX_POS_WEIGHT; gradient clipping at 1.0.

Input: rolling windows of ``[z_score, conditional_volatility, half_life]``
(Ramzy's canonical feature vector), shape (N, L, 3).
Output: logits; ``predict_regime_score`` returns sigmoid(logits) in [0, 1].
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from pairs_trading.data.schemas import ArmaGarchResult, SpreadData
from pairs_trading.data.spread import compute_half_life

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:  # pragma: no cover
    torch = None
    nn = None
    DataLoader = None
    TensorDataset = None


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_FEATURES     = 3
FEATURE_NAMES  = ("z_score", "conditional_volatility", "half_life")
MAX_POS_WEIGHT = 10.0


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TFTScaling:
    """Per-feature mean/std. Fit on TRAIN data only to avoid leakage."""

    mean: np.ndarray
    std: np.ndarray

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        return x * self.std + self.mean


def fit_tft_scaling(x: np.ndarray) -> TFTScaling:
    """Fit per-feature scaling on an array with features on the last axis."""
    flat = x.reshape(-1, x.shape[-1])
    mean = flat.mean(axis=0)
    std  = flat.std(axis=0)
    std  = np.where(std < 1e-8, 1.0, std)
    return TFTScaling(mean=mean, std=std)


@dataclass(frozen=True)
class TFTWindowData:
    """Supervised windows. ``x`` is UNSCALED unless a scaling was applied
    explicitly -- scaling is fit downstream on the train slice only."""

    x: np.ndarray              # (N, input_length, N_FEATURES) float32
    target_index: pd.Index     # one timestamp per window
    input_length: int
    scaling: Optional[TFTScaling]   # None when built with standardize=False

    @property
    def n_samples(self) -> int:
        return self.x.shape[0]


@dataclass(frozen=True)
class TFTTrainingHistory:
    """Outputs of one training run."""

    train_loss: list
    val_loss: list
    val_f1: list
    test_metrics: dict            # full confusion-matrix metrics on held-out test
    scaling: TFTScaling           # fit on train slice -- needed for inference
    pos_weight: float
    split_sizes: tuple            # (n_train, n_val, n_test) AFTER purging
    seed: int


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def classification_metrics(
    scores: np.ndarray,
    labels: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    """
    Full confusion-matrix metrics from probabilities and binary labels.

    For a regime gate the operative numbers are:
      - specificity  : of the BAD regimes, fraction correctly blocked
      - npv          : when we block, fraction we were right
      - mcc          : balance-insensitive overall quality (0 = chance)
    Positive-class F1 is included but is NOT the headline -- with
    imbalanced-positive labels an all-ones classifier maximises it.
    """
    pred = (scores >= threshold).astype(float)
    y    = labels.astype(float)

    tp = float(((pred == 1) & (y == 1)).sum())
    fp = float(((pred == 1) & (y == 0)).sum())
    fn = float(((pred == 0) & (y == 1)).sum())
    tn = float(((pred == 0) & (y == 0)).sum())

    eps = 1e-12
    precision   = tp / (tp + fp + eps)
    recall      = tp / (tp + fn + eps)
    specificity = tn / (tn + fp + eps)          # negative-class recall
    npv         = tn / (tn + fn + eps)          # negative-class precision
    f1          = 2 * precision * recall / (precision + recall + eps)
    balanced    = 0.5 * (recall + specificity)

    mcc_den = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc     = (tp * tn - fp * fn) / mcc_den if mcc_den > 0 else 0.0

    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "npv": npv,
        "f1": f1,
        "balanced_accuracy": balanced,
        "mcc": mcc,
        "n": int(tp + fp + fn + tn),
        "base_rate_positive": (tp + fn) / max(tp + fp + fn + tn, 1.0),
    }


# ---------------------------------------------------------------------------
# Purged & embargoed chronological split
# ---------------------------------------------------------------------------

def purged_split_indices(
    positions: np.ndarray,
    *,
    val_fraction: float,
    test_fraction: float,
    embargo: int,
) -> tuple:
    """
    Chronological train/val/test split with purging.

    ``positions`` are the trading-day positions of each (labeled) window
    in the ORIGINAL contiguous series -- e.g. ``np.where(mask)[0]`` after
    label alignment. Labeled windows are generally NOT contiguous in time,
    so the embargo must be measured in trading days, not window counts.

    Any train window within ``embargo`` trading days of the first val
    window is dropped (its 60-day input and/or H-day label horizon would
    overlap validation data). Same between val and test.

    embargo should be input_length + forward_window.
    """
    n = len(positions)
    n_test  = max(1, int(n * test_fraction))
    n_val   = max(1, int(n * val_fraction))
    n_train = n - n_val - n_test
    if n_train < 1:
        raise ValueError("Not enough windows for the requested split fractions")

    val_start_pos  = positions[n_train]
    test_start_pos = positions[n_train + n_val]

    train_idx = [i for i in range(n_train)
                 if positions[i] <= val_start_pos - embargo]
    val_idx   = [i for i in range(n_train, n_train + n_val)
                 if positions[i] <= test_start_pos - embargo]
    test_idx  = list(range(n_train + n_val, n))

    if len(train_idx) == 0 or len(val_idx) == 0 or len(test_idx) == 0:
        raise ValueError(
            f"Purging with embargo={embargo} emptied a split "
            f"(train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}). "
            "Reduce embargo, input_length, or forward_window, or use more data."
        )

    return np.array(train_idx), np.array(val_idx), np.array(test_idx)


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------

def _build_rolling_half_life(
    spread: pd.Series,
    window: int = 60,
) -> pd.Series:
    """
    Causal rolling half-life. NO look-ahead:

      - timesteps t < window get NaN (later dropped), NOT a full-sample value
      - invalid fits (non-finite / failed) are forward-filled from the most
        recent valid CAUSAL estimate
    """
    spread_clean = spread.dropna().astype(float)
    values = spread_clean.to_numpy()
    n = len(values)
    half_lives = np.full(n, np.nan)

    for t in range(window, n):
        segment = pd.Series(values[t - window: t])
        try:
            hl = compute_half_life(segment)
            if np.isfinite(hl):
                half_lives[t] = hl
        except Exception:
            pass  # stays NaN; ffill below is causal

    out = pd.Series(half_lives, index=spread_clean.index, name="half_life")
    return out.ffill()   # leading NaNs remain and are dropped downstream


def make_tft_dataset(
    spread_data: SpreadData,
    arma_result: ArmaGarchResult,
    *,
    input_length: int = 60,
    half_life_window: int = 60,
    standardize: bool = False,
    scaling: Optional[TFTScaling] = None,
) -> TFTWindowData:
    """
    Build supervised windows from spread pipeline outputs.

    Features (Ramzy's canonical vector):
        z_score, conditional_volatility (GARCH sigma_t), rolling half_life.

    NOTE on scaling: the default is now ``standardize=False``. Scaling is
    fit INSIDE ``train_tft_classifier`` on the train slice only and
    returned in the history, to prevent test statistics from leaking into
    training inputs. Pass ``scaling=`` only for inference with a
    previously fit scaler.
    """
    if input_length < 2:
        raise ValueError("input_length must be at least 2")

    rolling_hl = _build_rolling_half_life(spread_data.spread, window=half_life_window)

    feature_df = pd.DataFrame({
        "z_score":                spread_data.z_score,
        "conditional_volatility": arma_result.conditional_volatility,
        "half_life":              rolling_hl,
    }).dropna()

    if len(feature_df) < input_length + 1:
        raise ValueError(
            f"Not enough observations ({len(feature_df)}) for input_length={input_length}"
        )

    values    = feature_df.to_numpy(dtype=np.float32)
    n_samples = len(values) - input_length

    x = np.stack(
        [values[i: i + input_length] for i in range(n_samples)],
        axis=0,
    )
    target_index = feature_df.index[input_length:]

    fitted = scaling
    if scaling is not None:
        x = scaling.transform(x)
    elif standardize:
        fitted = fit_tft_scaling(x)
        x = fitted.transform(x)

    return TFTWindowData(
        x=x.astype(np.float32),
        target_index=target_index,
        input_length=input_length,
        scaling=fitted,
    )


def make_latest_tft_window(
    spread_data: SpreadData,
    arma_result: ArmaGarchResult,
    *,
    input_length: int,
    half_life_window: int = 60,
    scaling: Optional[TFTScaling] = None,
) -> np.ndarray:
    """Single model-ready window from the latest data, shape (1, L, 3)."""
    dataset = make_tft_dataset(
        spread_data,
        arma_result,
        input_length     = input_length,
        half_life_window = half_life_window,
        standardize      = False,
        scaling          = scaling,
    )
    return dataset.x[[-1]]


# ---------------------------------------------------------------------------
# PyTorch model blocks
# ---------------------------------------------------------------------------

def _require_torch() -> None:
    if torch is None:
        raise ImportError(
            "TFTClassifier requires PyTorch. Install with: uv sync"
        )


if torch is not None and nn is not None:

    class GatedResidualNetwork(nn.Module):
        """GRN(a, c) = LayerNorm(a + GLU(W1 * ELU(W2*a + W3*c) + b1)).
        Paper Section 4.1."""

        def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            output_dim: int,
            context_dim: int = 0,
            dropout: float = 0.3,
        ) -> None:
            super().__init__()
            self.w2 = nn.Linear(input_dim, hidden_dim)
            self.w3 = nn.Linear(context_dim, hidden_dim, bias=False) if context_dim > 0 else None
            self.w1 = nn.Linear(hidden_dim, output_dim)
            self.w4 = nn.Linear(output_dim, output_dim)
            self.w5 = nn.Linear(output_dim, output_dim)
            self.layer_norm = nn.LayerNorm(output_dim)
            self.dropout    = nn.Dropout(dropout)
            self.elu        = nn.ELU()
            self.skip_proj  = (
                nn.Linear(input_dim, output_dim, bias=False)
                if input_dim != output_dim else nn.Identity()
            )

        def forward(self, a, c=None):
            eta2 = self.w2(a)
            if self.w3 is not None and c is not None:
                eta2 = eta2 + self.w3(c)
            eta2 = self.elu(eta2)
            eta1 = self.dropout(self.w1(eta2))
            glu  = torch.sigmoid(self.w4(eta1)) * self.w5(eta1)
            return self.layer_norm(self.skip_proj(a) + glu)


    class VariableSelectionNetwork(nn.Module):
        """Softmax selection weights over input variables. Paper Section 4.2."""

        def __init__(self, n_vars, input_dim, d_model, dropout=0.3):
            super().__init__()
            self.n_vars  = n_vars
            self.d_model = d_model
            self.var_projections = nn.ModuleList(
                [nn.Linear(input_dim, d_model) for _ in range(n_vars)]
            )
            self.var_grns = nn.ModuleList(
                [GatedResidualNetwork(d_model, d_model, d_model, dropout=dropout)
                 for _ in range(n_vars)]
            )
            self.selection_grn = GatedResidualNetwork(
                n_vars * d_model, d_model, n_vars, dropout=dropout
            )
            self.softmax = nn.Softmax(dim=-1)

        def forward(self, x):
            batch, time, _ = x.shape
            projected = [proj(x[:, :, i: i + 1])
                         for i, proj in enumerate(self.var_projections)]
            flat = torch.cat(projected, dim=-1)
            var_weights = self.softmax(
                self.selection_grn(flat.view(batch * time, -1))
                    .view(batch, time, self.n_vars)
            )
            processed = [
                grn(p.view(batch * time, self.d_model)).view(batch, time, self.d_model)
                for grn, p in zip(self.var_grns, projected)
            ]
            stacked  = torch.stack(processed, dim=-1)
            combined = (stacked * var_weights.unsqueeze(2)).sum(dim=-1)
            return combined, var_weights


    class InterpretableMultiHeadAttention(nn.Module):
        """Multi-head attention with shared value weights. Paper Section 4.4."""

        def __init__(self, d_model, n_heads, dropout=0.3):
            super().__init__()
            if d_model % n_heads != 0:
                raise ValueError("d_model must be divisible by n_heads")
            self.d_attn = d_model // n_heads
            self.w_q = nn.ModuleList([nn.Linear(d_model, self.d_attn, bias=False)
                                      for _ in range(n_heads)])
            self.w_k = nn.ModuleList([nn.Linear(d_model, self.d_attn, bias=False)
                                      for _ in range(n_heads)])
            self.w_v = nn.Linear(d_model, self.d_attn, bias=False)
            self.w_h = nn.Linear(self.d_attn, d_model, bias=False)
            self.dropout = nn.Dropout(dropout)
            self.scale   = math.sqrt(self.d_attn)

        def forward(self, x, mask=None):
            V = self.w_v(x)
            outs, attns = [], []
            for wq, wk in zip(self.w_q, self.w_k):
                scores = torch.bmm(wq(x), wk(x).transpose(1, 2)) / self.scale
                if mask is not None:
                    scores = scores.masked_fill(mask.unsqueeze(0), float("-inf"))
                attn = self.dropout(torch.softmax(scores, dim=-1))
                outs.append(torch.bmm(attn, V))
                attns.append(attn)
            mean_out  = torch.stack(outs,  dim=0).mean(dim=0)
            mean_attn = torch.stack(attns, dim=0).mean(dim=0)
            return self.w_h(mean_out), mean_attn


    class TFTClassifier(nn.Module):
        """
        TFT adapted for binary regime classification.

        forward() returns LOGITS (unbounded). Use ``predict_regime_score``
        (or sigmoid) for probabilities. Training uses BCEWithLogitsLoss,
        which is numerically stabler than sigmoid + BCELoss and supports
        a native pos_weight.

        Default dropout is 0.3 per the paper's finding that gating +
        higher dropout is most beneficial on small, noisy (financial)
        datasets.
        """

        def __init__(
            self,
            input_length: int,
            d_model: int = 32,
            n_heads: int = 4,
            lstm_layers: int = 1,
            dropout: float = 0.3,
            regime_threshold: float = 0.5,
        ) -> None:
            super().__init__()
            if d_model % n_heads != 0:
                raise ValueError("d_model must be divisible by n_heads")

            self.input_length     = input_length
            self.d_model          = d_model
            self.regime_threshold = regime_threshold

            self.vsn  = VariableSelectionNetwork(N_FEATURES, 1, d_model, dropout)
            self.lstm = nn.LSTM(d_model, d_model, lstm_layers, batch_first=True,
                                dropout=dropout if lstm_layers > 1 else 0.0)

            self.lstm_gate      = nn.Linear(d_model, d_model)
            self.lstm_gate_sig  = nn.Linear(d_model, d_model)
            self.lstm_layernorm = nn.LayerNorm(d_model)

            self.attention      = InterpretableMultiHeadAttention(d_model, n_heads, dropout)
            self.attn_gate      = nn.Linear(d_model, d_model)
            self.attn_gate_sig  = nn.Linear(d_model, d_model)
            self.attn_layernorm = nn.LayerNorm(d_model)

            self.poswise_grn     = GatedResidualNetwork(d_model, d_model, d_model,
                                                        dropout=dropout)
            self.final_gate      = nn.Linear(d_model, d_model)
            self.final_gate_sig  = nn.Linear(d_model, d_model)
            self.final_layernorm = nn.LayerNorm(d_model)

            # Head outputs raw logits -- NO sigmoid here.
            self.classifier = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, 1),
            )

            mask = torch.triu(torch.ones(input_length, input_length), diagonal=1).bool()
            self.register_buffer("causal_mask", mask, persistent=False)

        def _glu_skip(self, x, residual, gate, gate_sig, layernorm):
            return layernorm(residual + torch.sigmoid(gate_sig(x)) * gate(x))

        def forward(self, x):
            """x: (batch, input_length, N_FEATURES) -> logits (batch,)."""
            vsn_out, _  = self.vsn(x)
            lstm_out, _ = self.lstm(vsn_out)
            after_lstm  = self._glu_skip(lstm_out, vsn_out,
                                         self.lstm_gate, self.lstm_gate_sig,
                                         self.lstm_layernorm)
            attn_out, _ = self.attention(after_lstm, mask=self.causal_mask)
            after_attn  = self._glu_skip(attn_out, after_lstm,
                                         self.attn_gate, self.attn_gate_sig,
                                         self.attn_layernorm)
            batch, time, _ = after_attn.shape
            grn_out = self.poswise_grn(
                after_attn.view(batch * time, self.d_model)
            ).view(batch, time, self.d_model)
            final = self._glu_skip(grn_out, after_attn,
                                   self.final_gate, self.final_gate_sig,
                                   self.final_layernorm)
            return self.classifier(final[:, -1, :]).squeeze(-1)

        def predict_regime_score(self, x):
            """Probabilities in [0, 1] = sigmoid(logits)."""
            return torch.sigmoid(self.forward(x))

        def get_variable_importance(self, x) -> dict:
            """
            Mean VSN selection weights. CAVEAT: only interpretable when the
            model performs above baseline -- a failing model's weights
            describe what it grasped at, not what drives regimes.
            """
            self.eval()
            with torch.no_grad():
                _, var_weights = self.vsn(x)
            mean_weights = var_weights.mean(dim=(0, 1)).cpu().numpy()
            return {name: float(w) for name, w in zip(FEATURE_NAMES, mean_weights)}

else:

    class TFTClassifier:  # pragma: no cover
        def __init__(self, *args, **kwargs):
            _require_torch()


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_tft_classifier(
    model: "TFTClassifier",
    data: TFTWindowData,
    labels: np.ndarray,
    *,
    window_positions: Optional[np.ndarray] = None,
    embargo: Optional[int] = None,
    forward_window: int = 0,
    epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    pos_weight: Optional[float] = None,
    early_stopping_patience: int = 10,
    device: Optional[str] = None,
    seed: int = 0,
) -> TFTTrainingHistory:
    """
    Train the TFT regime gate on binary reversion labels.

    Split: purged & embargoed chronological [train][val][test].
      - ``window_positions``: trading-day position of each labeled window
        in the original contiguous series (``np.where(mask)[0]``).
        Required for correct purging when labeled windows are sparse.
      - ``embargo``: trading days to purge at each boundary. Defaults to
        ``data.input_length + forward_window`` -- pass forward_window (H)
        so the default is correct.

    Scaling is fit on the (purged) TRAIN slice only and returned in the
    history; pass it to ``make_latest_tft_window`` at inference.

    Loss: BCEWithLogitsLoss with pos_weight = n_neg/n_pos (train slice),
    capped at MAX_POS_WEIGHT. Train loader shuffles (split, not batch
    order, is what prevents leakage). Early stopping on val loss restores
    the best checkpoint before test evaluation.
    """
    _require_torch()

    if len(labels) != data.n_samples:
        raise ValueError(
            f"labels length ({len(labels)}) must match data.n_samples ({data.n_samples})"
        )
    if val_fraction + test_fraction >= 1.0:
        raise ValueError("val_fraction + test_fraction must be < 1.0")

    torch.manual_seed(seed)
    np.random.seed(seed)

    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    if window_positions is None:
        window_positions = np.arange(len(labels))
    if embargo is None:
        embargo = data.input_length + forward_window

    train_idx, val_idx, test_idx = purged_split_indices(
        window_positions,
        val_fraction  = val_fraction,
        test_fraction = test_fraction,
        embargo       = embargo,
    )

    # Scaling: fit on TRAIN only, then transform everything.
    scaling  = fit_tft_scaling(data.x[train_idx])
    x_scaled = scaling.transform(data.x).astype(np.float32)

    x_all = torch.as_tensor(x_scaled, dtype=torch.float32)
    y_all = torch.as_tensor(labels,    dtype=torch.float32)

    x_train, y_train = x_all[train_idx], y_all[train_idx]
    x_val,   y_val   = x_all[val_idx],   y_all[val_idx]
    x_test,  y_test  = x_all[test_idx],  y_all[test_idx]

    n_train, n_val = len(train_idx), len(val_idx)
    if n_train < batch_size:
        batch_size = max(2, n_train // 2)

    if pos_weight is None:
        n_pos = float(y_train.sum().item())
        n_neg = float((1 - y_train).sum().item())
        pos_weight = min(n_neg / (n_pos + 1e-8), MAX_POS_WEIGHT)

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight], dtype=torch.float32).to(device)
    )

    train_loader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=batch_size,
        shuffle=True,            # batch order != leakage; helps optimisation
        generator=torch.Generator().manual_seed(seed),
    )
    val_loader = DataLoader(TensorDataset(x_val, y_val),
                            batch_size=batch_size, shuffle=False)

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_loss_hist, val_loss_hist, val_f1_hist = [], [], []
    best_val_loss, patience, best_state = float("inf"), 0, None

    for _epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item() * len(bx)
        train_loss_hist.append(epoch_loss / n_train)

        model.eval()
        v_loss, v_scores, v_labels = 0.0, [], []
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                logits = model(bx)
                v_loss += criterion(logits, by).item() * len(bx)
                v_scores.append(torch.sigmoid(logits).cpu())
                v_labels.append(by.cpu())
        val_loss = v_loss / n_val
        val_loss_hist.append(val_loss)
        vm = classification_metrics(
            torch.cat(v_scores).numpy(), torch.cat(v_labels).numpy()
        )
        val_f1_hist.append(vm["f1"])

        if val_loss < best_val_loss:
            best_val_loss, patience = val_loss, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
        if patience >= early_stopping_patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Held-out test -- never touched during training or early stopping.
    model.eval()
    with torch.no_grad():
        test_scores = torch.sigmoid(model(x_test.to(device))).cpu().numpy()
    test_metrics = classification_metrics(test_scores, y_test.numpy())
    test_metrics["loss"] = float(
        nn.BCEWithLogitsLoss()(
            torch.logit(torch.clamp(torch.as_tensor(test_scores), 1e-6, 1 - 1e-6)),
            y_test,
        ).item()
    )

    return TFTTrainingHistory(
        train_loss   = train_loss_hist,
        val_loss     = val_loss_hist,
        val_f1       = val_f1_hist,
        test_metrics = test_metrics,
        scaling      = scaling,
        pos_weight   = pos_weight,
        split_sizes  = (len(train_idx), len(val_idx), len(test_idx)),
        seed         = seed,
    )


def evaluate_tft_multi_seed(
    make_model,                      # callable () -> TFTClassifier (fresh weights)
    data: TFTWindowData,
    labels: np.ndarray,
    *,
    seeds: tuple = (0, 1, 2),
    **train_kwargs,
) -> tuple:
    """
    Train across multiple seeds; report mean +/- std of test metrics.

    Single-run numbers at N~300 have error bars wider than typical
    architecture differences -- never report one seed.

    Returns (histories, aggregate) where aggregate maps each test metric
    to {"mean": float, "std": float}.
    """
    histories = []
    for s in seeds:
        model = make_model()
        h = train_tft_classifier(model, data, labels, seed=s, **train_kwargs)
        histories.append(h)

    keys = ["f1", "precision", "recall", "specificity", "npv",
            "balanced_accuracy", "mcc"]
    aggregate = {}
    for k in keys:
        vals = np.array([h.test_metrics[k] for h in histories])
        aggregate[k] = {"mean": float(vals.mean()), "std": float(vals.std())}
    return histories, aggregate


# ---------------------------------------------------------------------------
# Baselines -- if the TFT does not beat these, that is the finding
# ---------------------------------------------------------------------------

def _window_summary_features(x: np.ndarray) -> np.ndarray:
    """Simple per-window summaries: last value + mean of each feature -> (N, 6)."""
    last = x[:, -1, :]
    mean = x.mean(axis=1)
    return np.concatenate([last, mean], axis=1)


def _fit_logistic_numpy(
    X: np.ndarray, y: np.ndarray,
    *, lr: float = 0.1, epochs: int = 2000, l2: float = 1e-3,
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
    data: TFTWindowData,
    labels: np.ndarray,
    *,
    window_positions: np.ndarray,
    embargo: int,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
) -> dict:
    """
    Trivial baselines on the IDENTICAL purged split the TFT uses.

      all_ones : predict favorable regime for every window. With
                 imbalanced-positive labels this maximises F1 with zero
                 intelligence -- it is the bar positive-class F1 must clear.
      logistic : logistic regression on per-window summary features
                 (last + mean of each of the 3 features). The bar any
                 deep model must clear to justify its parameters.
    """
    train_idx, _val_idx, test_idx = purged_split_indices(
        window_positions,
        val_fraction=val_fraction, test_fraction=test_fraction, embargo=embargo,
    )

    y_train, y_test = labels[train_idx], labels[test_idx]

    results = {
        "all_ones": classification_metrics(np.ones(len(test_idx)), y_test)
    }

    feats   = _window_summary_features(data.x)
    scaling = fit_tft_scaling(feats[train_idx][:, None, :])  # reuse helper
    f_train = scaling.transform(feats[train_idx][:, None, :])[:, 0, :]
    f_test  = scaling.transform(feats[test_idx][:, None, :])[:, 0, :]

    w = _fit_logistic_numpy(f_train, y_train.astype(float))
    p_test = 1.0 / (1.0 + np.exp(
        -(np.concatenate([f_test, np.ones((len(f_test), 1))], axis=1) @ w)
    ))
    results["logistic"] = classification_metrics(p_test, y_test)

    return results


# ---------------------------------------------------------------------------
# Inference helper
# ---------------------------------------------------------------------------

def predict_regime_score(
    model: "TFTClassifier",
    x: np.ndarray,
    *,
    device: Optional[str] = None,
) -> np.ndarray:
    """Regime probabilities in [0, 1] for a batch of (already scaled) windows."""
    _require_torch()
    if device is None:
        device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        scores = torch.sigmoid(
            model(torch.as_tensor(x, dtype=torch.float32, device=device))
        ).cpu().numpy()
    return scores