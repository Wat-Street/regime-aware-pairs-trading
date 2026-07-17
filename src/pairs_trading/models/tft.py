"""Temporal Fusion Transformer regime gate classifier.

Adapts Lim et al. (2021) arXiv:1912.09363 for binary regime classification
in the RAPTS pairs-trading pipeline.

Architecture (faithful to the paper unless noted):
  - Per-variable GRN embeddings + softmax VSN (Section 4.2, Eq. 6-8)
  - LSTM sequence-to-sequence local processing (Section 4.5.1)
  - Gated skip residual over LSTM (Eq. 17)
  - Optional context-free static enrichment GRN (Eq. 18, c=0). Ablatable.
  - Interpretable multi-head self-attention with shared values (Section 4.4)
  - Gated skip residual over attention block (Eq. 20)
  - Position-wise feed-forward GRN with shared weights (Section 4.5.4, Eq. 21-22)
  - Plain MLP classification head (logit out; sigmoid at inference only)

Deliberate deviations from the paper, with rationale:
  - ATTENTION IS BIDIRECTIONAL BY DEFAULT (causal_attention=False). The paper's
    decoder mask exists to stop the decoder seeing future *known* inputs during
    multi-horizon forecasting. Here the task is sequence-to-ONE classification:
    every feature in the lookback window is historical relative to the decision
    point, and temporal safety across windows is already guaranteed by the
    purged+embargoed split. Bidirectional attention within the window is strictly
    more expressive with no leakage. Set causal_attention=True to restore the
    paper's masked behaviour.
  - CLASSIFICATION HEAD IS A PLAIN MLP, not a GRN. A GRN with output_dim=1 turns
    its residual skip into a learned linear read-out that adds to the GLU output
    inside LayerNorm, conflating the skip-connection and prediction roles. A clean
    Linear -> ELU -> Dropout -> Linear head is more transparent and standard.
  - STATIC ENRICHMENT IS OPTIONAL (use_static_enrichment, default True). There
    are no static covariates in this pipeline, so the "enrichment" GRN reduces
    to an extra context-free nonlinear layer. Kept on by default (it won its
    ablation in prior runs); ablatable in the hyperparameter search.

Data contract:
  - Features are defined ONCE in pairs_trading.data.labels (FEATURE_NAMES,
    build_feature_frame). This module never builds features itself — training
    datasets come from labels.make_tft_dataset / labels.build_multi_pair_dataset
    and the inference window from labels.make_latest_window, so training and
    inference preprocessing cannot drift.

Training:
  - Constant-lr AdamW (weight_decay=0 by default == plain Adam) with early
    stopping — matches the paper and the sibling autoformer/w_transformer
    trainers. No warmup/cosine schedule: with early stopping it starved the
    first epochs of learning rate and stopped runs before they trained.
  - BCEWithLogitsLoss with pos_weight = n_neg/n_pos (capped at MAX_POS_WEIGHT)
  - Gradient clipping (configurable max_grad_norm)
  - Purged & embargoed chronological splits (no leakage)
  - Train-only feature scaling returned in ModelBundle for consistent inference
  - Early stopping on val loss, best-checkpoint restore. best_val_mcc is the
    val MCC AT the restored checkpoint epoch — not the max over epochs, which
    would describe a model that was never saved.
  - Hyperparameter search SELECTS ON VALIDATION MCC and NEVER EVALUATES the
    test set (evaluate_test=False during search). Test is touched exactly once,
    by the final multi-seed run, for the reported number.

Integration:
  - ModelBundle: versioned checkpoint (weights + scaler + label config +
    feature names + run_id)
  - RegimeGate: production wrapper that loads the bundle once and serves
    gate probabilities.
  - gate_signal(spread_data, arma_result, bundle) -> float in [0,1]
    One-shot convenience wrapper around the same path.
"""

from __future__ import annotations

import json
import math
import random
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Canonical definitions live in the data layer; import them so ModelBundle and
# the model agree with labels.py about what a feature vector is.
from pairs_trading.data.labels import (
    FEATURE_NAMES,
    LabelConfig,
    make_latest_window,
)

try:
    from skopt import gp_minimize
    from skopt.space import Categorical, Real

    HAS_SKOPT = True
except ImportError:
    HAS_SKOPT = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_FEATURES: int = len(FEATURE_NAMES)
MAX_POS_WEIGHT: float = 10.0

# Bayesian search budget. 40 for the 8-dimensional space —
# raise via --n-trials for a serious re-tune.
N_CALLS: int = 40
SEARCH_SEEDS: int = 2
LOOKBACK_CHOICES: Tuple[int, ...] = (30, 40, 60, 80, 100)

# Architecture as VALID (d_model x n_heads) pairs, encoded "dxh". Searching
# the pair jointly (instead of two independent dimensions) means the optimizer
# can never sample a structurally invalid combination — no wasted trials, no
# fake penalty scores polluting the surrogate.
# Grounding: the paper (Appendix A) searches state size and n_heads {1, 4};
# these are all combinations of d_model {16, 32, 64} x heads {1, 2, 4} whose
# per-head width stays >= 16.
_ARCH_CHOICES: Tuple[str, ...] = ("16x1", "32x1", "32x2", "64x1", "64x2", "64x4")

# Fixed params. lstm_layers is pinned at 1 per the paper (single
# encoder/decoder LSTM layer); capacity is explored through _ARCH_CHOICES and
# use_static_enrichment instead.
_BAYES_FIXED: Dict[str, Any] = {
    "lstm_layers": 1,
}

# Discrete fallback space used when scikit-optimize is not installed.
# Values are the paper's Appendix A grid adapted to this data:
#   learning_rate {1e-4..1e-2}   — paper grid, log-spaced fill-in
#   dropout {0.1..0.5}           — lower half of the paper's grid
#   max_grad_norm {0.01,1,100}   — verbatim paper grid
#   batch {32,64,128}            — paper uses {64,128,256}; shifted one notch
#                                  down because our N is far below the paper's
#   weight_decay                 — our addition (paper uses plain Adam);
#                                  0.0 = paper behaviour, kept searchable
_BAYES_FALLBACK_SPACE: Dict[str, List[Any]] = {
    "input_length": list(LOOKBACK_CHOICES),
    "arch": list(_ARCH_CHOICES),
    "use_static_enrichment": [True, False],
    "dropout": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
    "batch_size": [32, 64, 128, 256],
    "learning_rate": [1e-4, 3e-4, 1e-3, 3e-3, 1e-2],
    "weight_decay": [0.0, 1e-5, 1e-4, 1e-3],
    "max_grad_norm": [0.01, 1.0, 100.0],
    "lambda_forecast": [0.01, 0.03, 0.1, 0.3, 1.0],
}


def _resolve_device(device: Optional[str] = None) -> str:
    """Best available device: explicit choice > cuda > mps > cpu."""
    if device is not None:
        return device
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TFTScaling:
    """Per-feature mean/std fit on TRAIN data only — never on val or test."""

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
    std = flat.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    return TFTScaling(mean=mean, std=std)


@dataclass(frozen=True)
class TFTWindowData:
    """Sliding-window dataset.

    x is UNSCALED unless a scaler was explicitly applied by the builder.
    Scaling is fit downstream on the train slice only and stored in
    ModelBundle so inference always uses the same scaler as the training
    run that produced the weights.
    """

    x: np.ndarray  # (N, input_length, N_FEATURES) float32
    target_index: pd.Index  # one timestamp per window (label timestamp)
    input_length: int
    pair_id: Optional[str] = None
    # Future z-scores z[t+1 .. t+H] per window — the auxiliary forecasting
    # target. Filled by the labels.py builders; None on hand-built data.
    future_z: Optional[np.ndarray] = None  # (N, H) float32
    # SIGNED z at the window's decision point. The model features are
    # deliberately sign-free (z_magnitude), but execution logic (trade
    # direction, dynamic time stop) needs the sign. Filled by the builders.
    entry_z: Optional[np.ndarray] = None  # (N,) float32

    @property
    def n_samples(self) -> int:
        return self.x.shape[0]


@dataclass
class ModelBundle:
    """Versioned production checkpoint: everything needed to reproduce inference."""

    run_id: str
    model_state: Dict[str, torch.Tensor]
    model_config: Dict[str, Any]
    scaling: TFTScaling
    label_config: LabelConfig
    test_metrics: Dict[str, float]
    half_life_window: int = 60
    feature_names: Tuple[str, ...] = FEATURE_NAMES
    created_at: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%S"))

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "run_id": self.run_id,
                "model_state": self.model_state,
                "model_config": self.model_config,
                "scaling_mean": self.scaling.mean,
                "scaling_std": self.scaling.std,
                "label_config": {
                    "H": self.label_config.H,
                    "half_life": self.label_config.half_life,
                    "label_balance": self.label_config.label_balance,
                    "pair_id": self.label_config.pair_id,
                    "max_half_life": self.label_config.max_half_life,
                    "entry_threshold": self.label_config.entry_threshold,
                    "reversion_target": self.label_config.reversion_target,
                    "stop_loss": self.label_config.stop_loss,
                },
                "test_metrics": self.test_metrics,
                "half_life_window": self.half_life_window,
                "feature_names": list(self.feature_names),
                "created_at": self.created_at,
            },
            path,
        )

    @classmethod
    def load(cls, path: Path) -> "ModelBundle":
        ckpt = torch.load(path, map_location="cpu", weights_only=False)

        saved_features = tuple(ckpt.get("feature_names", ()))
        unknown = [f for f in saved_features if f not in FEATURE_NAMES]
        if unknown:
            raise ValueError(
                f"Bundle was trained on features {unknown} that the current "
                f"pipeline no longer provides (registry: {FEATURE_NAMES}). "
                "Retrain before serving."
            )
        if not saved_features:
            warnings.warn(
                "Bundle predates feature-name versioning — it was trained on an "
                "older feature set and should be retrained.",
                UserWarning,
            )

        return cls(
            run_id=ckpt["run_id"],
            model_state=ckpt["model_state"],
            model_config=ckpt["model_config"],
            scaling=TFTScaling(mean=ckpt["scaling_mean"], std=ckpt["scaling_std"]),
            label_config=LabelConfig(**ckpt["label_config"]),
            test_metrics=ckpt["test_metrics"],
            half_life_window=ckpt.get("half_life_window", 60),
            feature_names=saved_features or FEATURE_NAMES,
            created_at=ckpt["created_at"],
        )

    def build_model(self) -> "TFTClassifier":
        model = TFTClassifier(**self.model_config)
        model.load_state_dict(self.model_state)
        return model


@dataclass
class TrainingHistory:
    """Outputs of one training run."""

    train_loss: List[float]  # combined loss (BCE + lambda*MSE when aux is on)
    val_loss: List[float]  # classification (BCE) loss ONLY — drives selection
    val_mcc: List[float]
    best_val_mcc: float  # val MCC at the restored (best val loss) epoch
    best_epoch: int
    test_metrics: Dict[str, float]  # empty when evaluate_test=False
    scaling: TFTScaling
    pos_weight: float
    split_sizes: Tuple[int, int, int]
    seed: int
    hparams: Dict[str, Any]
    epochs_trained: int
    # Auxiliary forecast-head diagnostics (only populated when lambda > 0).
    lambda_forecast: float = 0.0
    val_forecast_mse: List[float] = field(default_factory=list)
    test_forecast_mse: float = float("nan")


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def classification_metrics(
    scores: np.ndarray,
    labels: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Full confusion-matrix metrics from probabilities and binary labels.

    For a regime gate the operative numbers are specificity (of bad regimes,
    fraction blocked), npv, and MCC (balance-insensitive, 0 = chance).
    Positive-class F1 is reported but is NOT the headline — an all-ones
    classifier maximises it on imbalanced-positive labels with zero skill.
    """
    pred = (scores >= threshold).astype(float)
    y = labels.astype(float)

    tp = float(((pred == 1) & (y == 1)).sum())
    fp = float(((pred == 1) & (y == 0)).sum())
    fn = float(((pred == 0) & (y == 1)).sum())
    tn = float(((pred == 0) & (y == 0)).sum())

    eps = 1e-12
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    specificity = tn / (tn + fp + eps)
    npv = tn / (tn + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    balanced = 0.5 * (recall + specificity)
    mcc_den = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / mcc_den if mcc_den > 0 else 0.0

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
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
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Chronological train/val/test split with purging and embargo.

    positions: trading-day positions of each labeled window in the original
    contiguous series (np.where(valid_mask)[0]). Labeled windows are generally
    NOT contiguous, so embargo is measured in trading days, not window counts.
    """
    n = len(positions)
    n_test = max(1, int(n * test_fraction))
    n_val = max(1, int(n * val_fraction))
    n_train = n - n_val - n_test
    if n_train < 1:
        raise ValueError(
            f"Not enough windows ({n}) for val={val_fraction}, test={test_fraction}."
        )

    val_start_pos = positions[n_train]
    test_start_pos = positions[n_train + n_val]

    train_idx = [i for i in range(n_train) if positions[i] <= val_start_pos - embargo]
    val_idx = [
        i
        for i in range(n_train, n_train + n_val)
        if positions[i] <= test_start_pos - embargo
    ]
    test_idx = list(range(n_train + n_val, n))

    if not train_idx or not val_idx or not test_idx:
        raise ValueError(
            f"Purging with embargo={embargo} emptied a split "
            f"(train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}). "
            "Reduce embargo or provide more data."
        )
    return np.array(train_idx), np.array(val_idx), np.array(test_idx)


# ---------------------------------------------------------------------------
# PyTorch model blocks
# ---------------------------------------------------------------------------


class GatedLinearUnit(nn.Module):
    """GLU(gamma) = sigmoid(W4 gamma + b4) ⊙ (W5 gamma + b5). Paper Eq. 5.

    Explicit module so the sigmoid-gate and value paths are unambiguous —
    no comment/name inversion risk.
    """

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.gate = nn.Linear(input_dim, output_dim)  # → sigmoid
        self.value = nn.Linear(input_dim, output_dim)  # → linear value

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.gate(x)) * self.value(x)


class GatedResidualNetwork(nn.Module):
    """GRN(a, c) = LayerNorm(a + GLU(W1·ELU(W2·a + W3·c))). Paper Eq. 2-5.

    Dropout applied to η1 (after W1, before the GLU and LayerNorm) per the paper.
    skip_proj handles input_dim != output_dim for the residual add.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        context_dim: int = 0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.w2 = nn.Linear(input_dim, hidden_dim)
        self.w3 = (
            nn.Linear(context_dim, hidden_dim, bias=False) if context_dim > 0 else None
        )
        self.w1 = nn.Linear(hidden_dim, output_dim)
        self.glu = GatedLinearUnit(output_dim, output_dim)
        self.layer_norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)
        self.elu = nn.ELU()
        self.skip_proj = (
            nn.Linear(input_dim, output_dim, bias=False)
            if input_dim != output_dim
            else nn.Identity()
        )

    def forward(
        self, a: torch.Tensor, c: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        eta2 = self.w2(a)
        if self.w3 is not None and c is not None:
            eta2 = eta2 + self.w3(c)
        eta2 = self.elu(eta2)
        eta1 = self.dropout(self.w1(eta2))  # dropout on η1 before gating
        return self.layer_norm(self.skip_proj(a) + self.glu(eta1))


class GatedSkip(nn.Module):
    """Gated skip connection: LayerNorm(residual + GLU(x)). Paper Eq. 17/20/22."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.glu = GatedLinearUnit(d_model, d_model)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        return self.layer_norm(residual + self.glu(x))


class VariableSelectionNetwork(nn.Module):
    """Per-variable GRN embeddings + softmax selection weights. Paper Eq. 6-8.

    Each scalar variable is projected to d_model, processed by its own GRN
    (weights shared across time), and combined by softmax selection weights
    produced from the flattened concatenation. selection_grn hidden dim is
    n_vars*d_model — no information bottleneck.
    """

    def __init__(self, n_vars: int, d_model: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.n_vars = n_vars
        self.d_model = d_model
        self.var_projections = nn.ModuleList(
            [nn.Linear(1, d_model) for _ in range(n_vars)]
        )
        self.var_grns = nn.ModuleList(
            [
                GatedResidualNetwork(d_model, d_model, d_model, dropout=dropout)
                for _ in range(n_vars)
            ]
        )
        self.selection_grn = GatedResidualNetwork(
            input_dim=n_vars * d_model,
            hidden_dim=n_vars * d_model,
            output_dim=n_vars,
            dropout=dropout,
        )
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch, time, _ = x.shape
        projected = [
            self.var_projections[j](x[:, :, j : j + 1]) for j in range(self.n_vars)
        ]
        flat = torch.cat(projected, dim=-1)
        flat_2d = flat.view(batch * time, self.n_vars * self.d_model)
        weights_2d = self.softmax(self.selection_grn(flat_2d))
        var_weights = weights_2d.view(batch, time, self.n_vars)
        processed = [
            self.var_grns[j](projected[j].view(batch * time, self.d_model)).view(
                batch, time, self.d_model
            )
            for j in range(self.n_vars)
        ]
        stacked = torch.stack(processed, dim=-1)  # (B, T, d_model, n_vars)
        combined = (stacked * var_weights.unsqueeze(2)).sum(dim=-1)
        return combined, var_weights


class InterpretableMultiHeadAttention(nn.Module):
    """Multi-head attention with SHARED value weights. Paper Eq. 13-16.

    Per-head Q/K, one shared V, additive averaging of head outputs — the mean
    attention matrix is directly interpretable as temporal importance.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
            )
        self.n_heads = n_heads
        self.d_attn = d_model // n_heads
        self.w_q = nn.ModuleList(
            [nn.Linear(d_model, self.d_attn, bias=False) for _ in range(n_heads)]
        )
        self.w_k = nn.ModuleList(
            [nn.Linear(d_model, self.d_attn, bias=False) for _ in range(n_heads)]
        )
        self.w_v = nn.Linear(d_model, self.d_attn, bias=False)  # shared across heads
        self.w_h = nn.Linear(self.d_attn, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_attn)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        V = self.w_v(x)
        head_outputs, head_attns = [], []
        for wq, wk in zip(self.w_q, self.w_k):
            scores = torch.bmm(wq(x), wk(x).transpose(1, 2)) / self.scale
            if mask is not None:
                scores = scores.masked_fill(mask.unsqueeze(0), float("-inf"))
            attn = self.dropout(torch.softmax(scores, dim=-1))
            head_outputs.append(torch.bmm(attn, V))
            head_attns.append(attn)
        mean_out = torch.stack(head_outputs, dim=0).mean(dim=0)
        mean_attn = torch.stack(head_attns, dim=0).mean(dim=0)
        return self.w_h(mean_out), mean_attn


class TFTClassifier(nn.Module):
    """TFT adapted for binary regime classification, with an optional
    auxiliary forecasting head.

    ONE model, ONE trunk (VSN -> LSTM -> attention -> GRNs), and at the trunk
    output two branches:
      - classification head -> logit -> sigmoid = P(favorable regime). This
        is THE gate signal, always present.
      - forecasting head (when forecast_horizon > 0): Linear(d_model, H)
        predicting the next H z-scores. TRAINING-ONLY — during training both
        heads' losses are summed (BCE + lambda*MSE) and backprop through the
        shared trunk, forcing it to learn representations that also know
        where the spread is going. At inference the forecast head is simply
        never called; the gate uses only the classification head.

    forward() returns LOGITS. Use predict_regime_score() for probabilities.
    Set return_internals=True on forward() to retrieve attention/VSN weights
    without re-running the pipeline (no code duplication, no drift risk).

    Defaults are the best configuration from the Bayesian search
    (selected on validation MCC).
    """

    def __init__(
        self,
        input_length: int,
        d_model: int = 64,
        n_heads: int = 2,
        lstm_layers: int = 1,
        dropout: float = 0.58,
        use_static_enrichment: bool = True,
        causal_attention: bool = False,
        regime_threshold: float = 0.5,
        feature_names: Sequence[str] = FEATURE_NAMES,
        forecast_horizon: int = 0,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
            )

        self.input_length = input_length
        self.d_model = d_model
        self.use_static_enrichment = use_static_enrichment
        self.causal_attention = causal_attention
        self.regime_threshold = regime_threshold
        # The feature vector this model was built for (labels.FEATURE_NAMES).
        # Defines the input width and the VSN importance labels.
        self.feature_names = tuple(feature_names)

        # 1. Variable selection
        self.vsn = VariableSelectionNetwork(len(self.feature_names), d_model, dropout)

        # 2. LSTM local processing + 3. gated skip (Eq. 17)
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.lstm_skip = GatedSkip(d_model)

        # 4. Optional static enrichment (context-free GRN, Eq. 18)
        self.static_enrichment = (
            GatedResidualNetwork(d_model, d_model, d_model, dropout=dropout)
            if use_static_enrichment
            else None
        )

        # 5. Attention + 6. gated skip (Eq. 19-20)
        self.attention = InterpretableMultiHeadAttention(d_model, n_heads, dropout)
        self.attn_skip = GatedSkip(d_model)

        # 7. Position-wise FF GRN + final gated skip over the transformer block (Eq. 21-22)
        self.poswise_grn = GatedResidualNetwork(
            d_model, d_model * 2, d_model, dropout=dropout
        )
        self.ff_skip = GatedSkip(d_model)

        # 8. Plain MLP classification head (transparent; no GRN-skip-as-readout)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

        # 8b. Auxiliary forecasting head — branches from the SAME trunk output
        # as the classifier. Training-only; inference ignores it entirely.
        self.forecast_horizon = forecast_horizon
        self.forecast_head = (
            nn.Linear(d_model, forecast_horizon) if forecast_horizon > 0 else None
        )

        # Causal mask buffer (only applied when causal_attention=True)
        mask = torch.triu(torch.ones(input_length, input_length), diagonal=1).bool()
        self.register_buffer("causal_mask", mask, persistent=False)

    def forward(
        self,
        x: torch.Tensor,
        return_internals: bool = False,
        return_forecast: bool = False,
    ):
        """x: (batch, input_length, N_FEATURES) -> logits (batch,).

        return_forecast=True (training loop only) additionally returns the
        forecast head output: (logits, z_forecast (batch, H)). Requires
        forecast_horizon > 0.
        If return_internals: returns (logits, {"attn_weights", "var_weights"}).
        Single code path — get_attention_weights / get_variable_importance call
        this with return_internals=True so they can never drift from forward().
        """
        batch, time, _ = x.shape

        vsn_out, var_weights = self.vsn(x)
        lstm_out, _ = self.lstm(vsn_out)
        after_lstm = self.lstm_skip(lstm_out, vsn_out)

        if self.static_enrichment is not None:
            enriched = self.static_enrichment(
                after_lstm.view(batch * time, self.d_model)
            ).view(batch, time, self.d_model)
        else:
            enriched = after_lstm

        mask = self.causal_mask if self.causal_attention else None
        attn_out, attn_weights = self.attention(enriched, mask=mask)
        after_attn = self.attn_skip(attn_out, enriched)

        ff_out = self.poswise_grn(after_attn.view(batch * time, self.d_model)).view(
            batch, time, self.d_model
        )
        final = self.ff_skip(ff_out, after_lstm)  # long skip back to seq2seq output

        # Trunk output — the branch point for both heads.
        trunk = final[:, -1, :]
        logit = self.classifier(trunk).squeeze(-1)

        if return_forecast:
            if self.forecast_head is None:
                raise ValueError("return_forecast=True requires forecast_horizon > 0")
            return logit, self.forecast_head(trunk)
        if return_internals:
            return logit, {"attn_weights": attn_weights, "var_weights": var_weights}
        return logit

    @torch.no_grad()
    def predict_regime_score(self, x: torch.Tensor) -> torch.Tensor:
        """Gate probability in [0,1] — sigmoid(logit). Inference only."""
        self.eval()
        return torch.sigmoid(self.forward(x))

    @torch.no_grad()
    def predict_regime_and_path(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """ONE forward pass -> (gate probability (B,), predicted z path (B, H)).

        The gate decision still comes from the classification head alone; the
        predicted path is exposed for downstream analysis (e.g. the dynamic
        time stop in train.py's exit evaluation). Requires forecast_horizon > 0.
        """
        self.eval()
        logit, z_path = self.forward(x, return_forecast=True)
        return torch.sigmoid(logit), z_path

    @torch.no_grad()
    def get_variable_importance(self, x: torch.Tensor) -> Dict[str, float]:
        """Mean VSN selection weights per feature — flat dict summing to 1.0."""
        self.eval()
        _, internals = self.forward(x, return_internals=True)
        w = internals["var_weights"].cpu().numpy()
        mean_weights = w.reshape(-1, w.shape[-1]).mean(axis=0)
        return {
            name: float(mean_weights[j]) for j, name in enumerate(self.feature_names)
        }

    @torch.no_grad()
    def get_variable_importance_percentiles(
        self, x: torch.Tensor
    ) -> Dict[str, Dict[str, float]]:
        """VSN selection weight percentiles — paper Table 3 format (p10/p50/p90)."""
        self.eval()
        _, internals = self.forward(x, return_internals=True)
        w_flat = (
            internals["var_weights"].cpu().numpy().reshape(-1, len(self.feature_names))
        )
        return {
            name: {
                "p10": float(np.percentile(w_flat[:, j], 10)),
                "p50": float(np.percentile(w_flat[:, j], 50)),
                "p90": float(np.percentile(w_flat[:, j], 90)),
            }
            for j, name in enumerate(self.feature_names)
        }

    @torch.no_grad()
    def get_attention_weights(self, x: torch.Tensor) -> np.ndarray:
        """Mean attention weight matrix (seq_len, seq_len) across the batch."""
        self.eval()
        _, internals = self.forward(x, return_internals=True)
        return internals["attn_weights"].cpu().numpy().mean(axis=0)


# ---------------------------------------------------------------------------
# Exit rules — read the model's outputs to decide WHEN to close a trade.
# Two independent mechanisms:
#   ramzy_exit_threshold  : the CONFIDENCE exit ("edge dropped, get out now")
#   get_dynamic_time_stop : the PATIENCE exit ("reversion is behind the
#                           schedule the model predicted at entry")
# When both are enabled, whichever fires first closes the trade.
# ---------------------------------------------------------------------------


def ramzy_exit_threshold(
    p_t: float,
    kappa_t: float,
    *,
    cost_z: float = 0.05,
    risk_buffer: float = 0.1,
) -> float:
    """Ramzy's dynamic exit z-threshold for day t of an open trade:

        z_exit_t = (c_z + lambda) / (p_t * kappa_t)

    where c_z is the transaction cost normalized to z units, lambda is a
    fixed risk buffer, p_t is the gate probability from the classification
    head recomputed on the window ending at day t, and kappa_t is the mean
    reversion speed refit on data up to day t (kappa = ln(2)/half_life via
    the canonical spread.compute_half_life estimator — see
    spread.reversion_speed_from_half_life).

    Exit rule: close the trade when today's |z_t| dips BELOW z_exit_t.

    Intuition: the threshold RISES (so the trade exits sooner) when
      - confidence p_t is low        (the gate no longer believes the regime),
      - reversion kappa_t is slow    (capture per unit time is poor),
      - costs c_z are high, or
      - the risk buffer lambda is large.
    Holding on under any of those conditions means taking on more risk for
    less expected capture — so the bar for staying in the trade goes up.

    Guards: p_t and kappa_t are floored at 1e-6; an invalid kappa (nan/<=0,
    i.e. no measurable mean reversion in the data so far) returns inf, which
    triggers an immediate exit — with no reversion there is no thesis left.
    """
    if not np.isfinite(kappa_t) or kappa_t <= 0:
        return float("inf")
    p = max(float(p_t), 1e-6)
    k = max(float(kappa_t), 1e-6)
    return (cost_z + risk_buffer) / (p * k)


def get_dynamic_time_stop(predicted_path: torch.Tensor, entry_z: float) -> int:
    """Timestep index at which the model expects mean reversion to complete.

    Defined as the first index t where predicted_path[t] crosses zero from
    the same sign as entry_z (i.e. sign(entry_z) * predicted_path[t] <= 0).
    Returns H (the path length) when no crossing is predicted within the
    horizon — the dynamic time stop then degenerates to the fixed H-day stop.

    Execution usage: compute once at entry from the forecast head's output
    and store as expected_reversion_day; while the trade is open, if
    current_day > expected_reversion_day and the spread has not reverted,
    exit regardless of the current z-score — the model's own thesis
    ("reversion completes by day t") has been falsified, so there is no
    reason to keep paying time risk waiting for the fixed stop.
    """
    path = torch.as_tensor(predicted_path).flatten()
    horizon = len(path)
    sign = 1.0 if entry_z > 0 else -1.0
    crossed = torch.nonzero(sign * path <= 0)
    return int(crossed[0].item()) if len(crossed) > 0 else horizon


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_tft_classifier(
    model: TFTClassifier,
    data: TFTWindowData,
    labels: np.ndarray,
    *,
    window_positions: Optional[np.ndarray] = None,
    embargo: Optional[int] = None,
    forward_window: int = 0,
    epochs: int = 150,
    batch_size: int = 64,
    learning_rate: float = 2e-3,
    weight_decay: float = 1e-5,
    max_grad_norm: float = 1.0,
    lambda_forecast: float = 0.0,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    pos_weight: Optional[float] = None,
    early_stopping_patience: int = 15,
    evaluate_test: bool = True,
    device: Optional[str] = None,
    seed: int = 0,
    hparams: Optional[Dict[str, Any]] = None,
) -> TrainingHistory:
    """Train the TFT regime gate.

    Split: purged & embargoed chronological [train | val | test].
    Optimizer: constant-lr AdamW (== Adam at weight_decay=0, the default).
    Constant lr with early stopping matches the paper and the sibling
    autoformer/w_transformer trainers; an earlier warmup+cosine schedule was
    removed because warmup at start_factor 1e-3 kept the lr near zero for the
    first ~15 epochs, letting early stopping fire on a barely-trained model.
    Loss: BCEWithLogitsLoss with pos_weight = n_neg/n_pos (capped).

    AUXILIARY FORECAST LOSS (lambda_forecast > 0): both heads run on every
    batch and loss = BCE + lambda * MSE(z_hat, z[t+1..t+H]), backpropagated
    through the shared trunk — the multi-task pressure is the point. Requires
    a model built with forecast_horizon > 0 and data with future_z (the
    labels.py builders fill it). Early stopping / checkpointing / selection
    use the VALIDATION BCE ONLY, so the auxiliary objective can never steer
    which model is kept; the forecast head is dead weight at inference.

    Scaling fit on the purged train slice only; returned in TrainingHistory.

    Checkpointing: early stopping and best-checkpoint restore both use val
    (classification) loss. best_val_mcc is the val MCC at that same restored
    epoch, so the selection number always describes the returned model.

    evaluate_test=False skips the held-out test evaluation entirely — use it
    during hyperparameter search so the test set is never touched before the
    final run.

    window_positions MUST be supplied for sparse/non-contiguous labeled windows
    (pass np.where(valid_mask)[0]). If omitted, a contiguous arange is assumed.
    """
    if len(labels) != data.n_samples:
        raise ValueError(
            f"labels length ({len(labels)}) != data.n_samples ({data.n_samples})"
        )
    if val_fraction + test_fraction >= 1.0:
        raise ValueError("val_fraction + test_fraction must be < 1.0")
    if lambda_forecast < 0:
        raise ValueError("lambda_forecast must be non-negative")
    aux = lambda_forecast > 0
    if aux and model.forecast_head is None:
        raise ValueError(
            "lambda_forecast > 0 requires a model built with forecast_horizon > 0"
        )
    if aux and data.future_z is None:
        raise ValueError(
            "lambda_forecast > 0 requires data.future_z (use the labels.py builders)"
        )
    if aux and data.future_z.shape[1] != model.forecast_horizon:
        raise ValueError(
            f"data.future_z horizon ({data.future_z.shape[1]}) != "
            f"model.forecast_horizon ({model.forecast_horizon})"
        )

    torch.manual_seed(seed)
    np.random.seed(seed)

    device = _resolve_device(device)

    if window_positions is None:
        # Safe only if windows are genuinely contiguous. Callers with sparse
        # labels MUST pass real positions.
        window_positions = np.arange(len(labels))
    else:
        window_positions = np.asarray(window_positions)
        if len(window_positions) != len(labels):
            raise ValueError(
                f"window_positions length ({len(window_positions)}) != "
                f"labels length ({len(labels)})"
            )

    if embargo is None:
        embargo = data.input_length + forward_window

    train_idx, val_idx, test_idx = purged_split_indices(
        window_positions,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
        embargo=embargo,
    )

    scaling = fit_tft_scaling(data.x[train_idx])  # train-only
    x_scaled = scaling.transform(data.x).astype(np.float32)

    x_all = torch.as_tensor(x_scaled, dtype=torch.float32)
    y_all = torch.as_tensor(labels.astype(np.float32), dtype=torch.float32)
    # z targets are already unitless (z-scores) — no scaling. A zero dummy
    # keeps the loader layout identical when the aux head is off.
    z_all = (
        torch.as_tensor(data.future_z, dtype=torch.float32)
        if aux
        else torch.zeros(len(y_all), 1)
    )

    x_train, y_train, z_train = x_all[train_idx], y_all[train_idx], z_all[train_idx]
    x_val, y_val, z_val = x_all[val_idx], y_all[val_idx], z_all[val_idx]
    x_test, y_test, z_test = x_all[test_idx], y_all[test_idx], z_all[test_idx]

    n_train, n_val = len(train_idx), len(val_idx)
    actual_batch = min(batch_size, max(2, n_train // 2))

    if pos_weight is None:
        n_pos = float(y_train.sum().item())
        n_neg = float((1 - y_train).sum().item())
        pos_weight = min(n_neg / (n_pos + 1e-8), MAX_POS_WEIGHT)

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight], dtype=torch.float32).to(device)
    )
    criterion_forecast = nn.MSELoss()

    train_loader = DataLoader(
        TensorDataset(x_train, y_train, z_train),
        batch_size=actual_batch,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    val_loader = DataLoader(
        TensorDataset(x_val, y_val, z_val), batch_size=actual_batch, shuffle=False
    )

    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    train_loss_hist, val_loss_hist, val_mcc_hist = [], [], []
    val_mse_hist: List[float] = []
    best_val_loss = float("inf")
    best_state: Optional[Dict] = None
    best_epoch = 0
    patience = 0
    epochs_trained = 0

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for bx, by, bz in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            if aux:
                logit, z_hat = model(bx, return_forecast=True)
                loss = criterion(logit, by) + lambda_forecast * criterion_forecast(
                    z_hat, bz.to(device)
                )
            else:
                loss = criterion(model(bx), by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            optimizer.step()
            epoch_loss += loss.item() * len(bx)
        train_loss_hist.append(epoch_loss / n_train)

        model.eval()
        v_loss = 0.0
        v_mse = 0.0
        v_scores, v_labels = [], []
        with torch.no_grad():
            for bx, by, bz in val_loader:
                bx, by = bx.to(device), by.to(device)
                if aux:
                    logits, z_hat = model(bx, return_forecast=True)
                    v_mse += criterion_forecast(z_hat, bz.to(device)).item() * len(bx)
                else:
                    logits = model(bx)
                # Selection signal is the CLASSIFICATION loss only.
                v_loss += criterion(logits, by).item() * len(bx)
                v_scores.append(torch.sigmoid(logits).cpu())
                v_labels.append(by.cpu())
        val_loss = v_loss / n_val
        val_loss_hist.append(val_loss)
        if aux:
            val_mse_hist.append(v_mse / n_val)
        vm = classification_metrics(
            torch.cat(v_scores).numpy(), torch.cat(v_labels).numpy()
        )
        val_mcc_hist.append(vm["mcc"])
        epochs_trained = epoch + 1

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
        if patience >= early_stopping_patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Val MCC at the restored checkpoint — the honest selection number.
    best_val_mcc = val_mcc_hist[best_epoch] if val_mcc_hist else 0.0

    # Held-out test — only when explicitly requested (never during search)
    test_metrics: Dict[str, float] = {}
    test_forecast_mse = float("nan")
    if evaluate_test:
        model.eval()
        with torch.no_grad():
            if aux:
                logits, z_hat = model(x_test.to(device), return_forecast=True)
                test_forecast_mse = float(
                    criterion_forecast(z_hat, z_test.to(device)).item()
                )
            else:
                logits = model(x_test.to(device))
            test_scores = torch.sigmoid(logits).cpu().numpy()
        test_metrics = classification_metrics(test_scores, y_test.numpy())

    return TrainingHistory(
        train_loss=train_loss_hist,
        val_loss=val_loss_hist,
        val_mcc=val_mcc_hist,
        best_val_mcc=best_val_mcc,
        best_epoch=best_epoch,
        test_metrics=test_metrics,
        scaling=scaling,
        pos_weight=pos_weight,
        split_sizes=(len(train_idx), len(val_idx), len(test_idx)),
        seed=seed,
        hparams=hparams or {},
        epochs_trained=epochs_trained,
        lambda_forecast=lambda_forecast,
        val_forecast_mse=val_mse_hist,
        test_forecast_mse=test_forecast_mse,
    )


def evaluate_tft_multi_seed(
    make_model_fn: Callable[[], TFTClassifier],
    data: TFTWindowData,
    labels: np.ndarray,
    *,
    seeds: Sequence[int] = (0, 1, 2),
    **train_kwargs,
) -> Tuple[List[TrainingHistory], Dict[str, Dict[str, float]], List[TFTClassifier]]:
    """Train across seeds; report mean ± std of val (and test) metrics.

    Single-run numbers at small N have error bars wider than typical
    architecture differences — never report one seed.

    Returns (histories, aggregate, models). models[i] is the trained model for
    seeds[i] with its best checkpoint restored — reuse it for variable
    importance and bundle saving instead of retraining.
    """
    histories: List[TrainingHistory] = []
    models: List[TFTClassifier] = []
    for s in seeds:
        model = make_model_fn()
        h = train_tft_classifier(model, data, labels, seed=s, **train_kwargs)
        histories.append(h)
        models.append(model)

    keys = [
        "f1",
        "precision",
        "recall",
        "specificity",
        "npv",
        "balanced_accuracy",
        "mcc",
    ]
    aggregate: Dict[str, Dict[str, float]] = {}
    if histories[0].test_metrics:  # empty when evaluate_test=False
        for k in keys:
            vals = np.array([h.test_metrics[k] for h in histories])
            aggregate[k] = {"mean": float(vals.mean()), "std": float(vals.std())}
        if histories[0].lambda_forecast > 0:
            mses = np.array([h.test_forecast_mse for h in histories])
            aggregate["test_forecast_mse"] = {
                "mean": float(mses.mean()),
                "std": float(mses.std()),
            }

    # Validation-MCC aggregate — used for model selection (NEVER test metrics)
    val_mccs = np.array([h.best_val_mcc for h in histories])
    aggregate["val_mcc"] = {
        "mean": float(val_mccs.mean()),
        "std": float(val_mccs.std()),
    }

    return histories, aggregate, models


# ---------------------------------------------------------------------------
# Hyperparameter search — Bayesian, selects on VALIDATION MCC, never sees test
# ---------------------------------------------------------------------------


def run_grid_search(
    dataset_builder: Callable[[int], Tuple[TFTWindowData, np.ndarray, np.ndarray]],
    *,
    forward_window: int,
    lookback_choices: Sequence[int] = LOOKBACK_CHOICES,
    n_iter: int = N_CALLS,
    search_seed: int = 0,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    verbose: bool = True,
) -> Tuple[Dict[str, Any], List[Dict]]:
    """Bayesian-optimized hyperparameter search, including the feature lookback.

    dataset_builder(input_length) must return (data, labels, window_positions)
    for that lookback — the dataset has to be rebuilt when the window length
    changes. Results (and failures — see below) are cached per lookback, so
    each of the (few) lookback values is built at most once.

    Searched (9 dimensions, bounds grounded in the paper's Appendix A grid;
    lambda_forecast is ours — auxiliary-head weight, log-uniform [0.01, 1.0]):
      - input_length : lookback_choices (default {30,40,60,80,100})
      - arch         : joint (d_model x n_heads), all valid pairs of
                       d_model {16,32,64} x heads {1,2,4} with per-head
                       width >= 16 (paper searches state size and heads {1,4})
      - use_static_enrichment : {True, False} (paper ablation; False = smaller model)
      - dropout      : [0.1, 0.7] — the paper's grid without its extreme 0.9
      - batch_size   : {32, 64, 128, 256} — paper's {64,128,256} plus 32 for
                       small single-pair runs (the n_train//2 guard clamps)
      - learning_rate: [1e-4, 1e-2] log — the paper's exact range
      - weight_decay : [0 via 1e-6, 1e-2] log — our addition; paper uses plain Adam
      - max_grad_norm: {0.01, 1.0, 100.0} — verbatim paper grid
    Fixed: lstm_layers=1 (paper uses a single LSTM layer).

    A lookback whose dataset_builder raises (usually: not enough labeled
    windows at that window length) is cached as a failure so later trials
    that sample the same lookback fail immediately instead of repeating the
    (expensive) dataset rebuild.

    SELECTION CRITERION IS VALIDATION MCC. The test set is NEVER evaluated
    during the search (evaluate_test=False on every trial). Each trial trains
    for 100 epochs (vs. 150 in the final run) — early stopping usually fires
    well before that, but a config that only pulls ahead after epoch 100
    would be undervalued here; treat search results as a ranking signal, not
    a final score.
    Falls back to random search (seeded, reproducible) if scikit-optimize is
    not installed.

    Returns (best_hparams, all_results) sorted by mean validation MCC.
    best_hparams includes input_length, d_model, n_heads and all fixed params,
    ready for the final retrain.
    """
    all_results: List[Dict] = []
    # Cache maps input_length -> dataset tuple, OR -> the Exception raised
    # trying to build it (see docstring: avoids repeatedly retrying a lookback
    # that will never have enough data).
    dataset_cache: Dict[int, Any] = {}
    trial_count = [0]
    t0 = time.time()
    n_calls = n_iter

    def _get_dataset(input_length: int):
        if input_length not in dataset_cache:
            try:
                dataset_cache[input_length] = dataset_builder(input_length)
            except Exception as exc:
                dataset_cache[input_length] = exc
        cached = dataset_cache[input_length]
        if isinstance(cached, Exception):
            raise cached
        return cached

    def _report(label: str, sel: float) -> float:
        """Single print+bookkeeping path for every trial outcome (success,
        dataset failure, or structurally invalid combo) — nothing bypasses
        the progress bar."""
        done = trial_count[0] + 1
        elapsed = time.time() - t0
        eta = (elapsed / done) * (n_calls - done) if done < n_calls else 0.0
        bar_len = 30
        filled = int(bar_len * done / n_calls)
        bar = "█" * filled + "░" * (bar_len - filled)
        best_so_far = max(
            (r["selection_val_mcc"] for r in all_results), default=float("nan")
        )
        if verbose:
            print(f"  trial {done:2d}/{n_calls}  {label}", flush=True)
        print(
            f"    [{bar}] {done}/{n_calls} | "
            f"elapsed {int(elapsed // 60):02d}:{int(elapsed % 60):02d} | "
            f"ETA {int(eta // 60):02d}:{int(eta % 60):02d} | "
            f"best val MCC so far: {best_so_far:.4f}",
            flush=True,
        )
        trial_count[0] += 1
        return -sel if np.isfinite(sel) else 1.0

    def _eval_config(params_dict: Dict[str, Any]) -> float:
        """Evaluate one config. Returns value to MINIMIZE (negative mean val MCC)."""
        input_length = int(params_dict["input_length"])
        # arch is a joint "d_model x n_heads" token — always a valid pair.
        d_model, n_heads = (int(v) for v in str(params_dict["arch"]).split("x"))

        hparams: Dict[str, Any] = {
            "input_length": input_length,
            "d_model": d_model,
            "n_heads": n_heads,
            "use_static_enrichment": bool(params_dict["use_static_enrichment"]),
            "dropout": float(params_dict["dropout"]),
            "batch_size": int(params_dict["batch_size"]),
            "learning_rate": float(params_dict["learning_rate"]),
            "weight_decay": float(params_dict["weight_decay"]),
            "max_grad_norm": float(params_dict["max_grad_norm"]),
            "lambda_forecast": float(params_dict["lambda_forecast"]),
            **_BAYES_FIXED,
        }

        sel = float("nan")
        label = ""
        try:
            data, labels, window_positions = _get_dataset(input_length)

            def make_model(hp: Dict[str, Any] = hparams) -> "TFTClassifier":
                return TFTClassifier(
                    input_length=input_length,
                    d_model=hp["d_model"],
                    n_heads=hp["n_heads"],
                    lstm_layers=hp["lstm_layers"],
                    dropout=hp["dropout"],
                    use_static_enrichment=hp["use_static_enrichment"],
                    forecast_horizon=forward_window,
                )

            _, aggregate, _ = evaluate_tft_multi_seed(
                make_model,
                data,
                labels,
                seeds=tuple(range(SEARCH_SEEDS)),
                window_positions=window_positions,
                forward_window=forward_window,
                epochs=100,
                batch_size=hparams["batch_size"],
                learning_rate=hparams["learning_rate"],
                weight_decay=hparams["weight_decay"],
                max_grad_norm=hparams["max_grad_norm"],
                lambda_forecast=hparams["lambda_forecast"],
                val_fraction=val_fraction,
                test_fraction=test_fraction,
                evaluate_test=False,  # test is off-limits during search
                hparams=hparams,
            )
            sel = float(aggregate["val_mcc"]["mean"])
            all_results.append(
                {"hparams": hparams, "aggregate": aggregate, "selection_val_mcc": sel}
            )
            label = (
                f"val MCC {sel:+.4f}  "
                f"L={input_length} arch={d_model}x{n_heads} "
                f"enrich={hparams['use_static_enrichment']} "
                f"lambda={hparams['lambda_forecast']:.2f} "
                f"lr={hparams['learning_rate']:.1e} "
                f"wd={hparams['weight_decay']:.1e} "
                f"dropout={hparams['dropout']:.2f} "
                f"bs={hparams['batch_size']}"
            )
        except Exception as e:
            label = f"FAILED (L={input_length}): {e}"

        return _report(label, sel)

    if HAS_SKOPT:
        bayes_space = [
            Categorical(list(lookback_choices), name="input_length"),
            Categorical(list(_ARCH_CHOICES), name="arch"),
            Categorical([True, False], name="use_static_enrichment"),
            Real(0.1, 0.7, name="dropout"),
            Categorical([32, 64, 128, 256], name="batch_size"),
            Real(1e-4, 1e-2, prior="log-uniform", name="learning_rate"),
            # 1e-6 is effectively 0 (paper's plain-Adam behaviour) on a log scale.
            Real(1e-6, 1e-2, prior="log-uniform", name="weight_decay"),
            Categorical([0.01, 1.0, 100.0], name="max_grad_norm"),
            # Auxiliary forecast-head weight (BCE + lambda*MSE).
            Real(0.01, 1.0, prior="log-uniform", name="lambda_forecast"),
        ]

        def objective(x: list) -> float:
            params = {dim.name: v for dim, v in zip(bayes_space, x)}
            return _eval_config(params)

        gp_minimize(
            objective,
            bayes_space,
            n_calls=n_calls,
            # A third of the budget is pure random exploration. With only 10
            # fixed initial points the GP spent the rest exploiting one lucky
            # region of a NOISY (2-seed) objective, so every trial looked the
            # same. noise="gaussian" additionally tells the GP the scores are
            # noisy, damping over-exploitation of a single good draw.
            n_initial_points=min(n_calls, max(10, n_calls // 3)),
            noise="gaussian",
            # Vary --search-seed across runs to explore different regions;
            # the same seed reproduces the same search trajectory.
            random_state=search_seed,
            verbose=False,
        )
    else:
        print(
            "  [WARNING] scikit-optimize not installed — using random search over the fallback space.\n"
            "  Install it for smarter Bayesian search: uv pip install scikit-optimize",
            flush=True,
        )
        fallback = dict(_BAYES_FALLBACK_SPACE)
        fallback["input_length"] = list(lookback_choices)
        rng = random.Random(search_seed)  # seeded so the path is reproducible
        for _ in range(n_calls):
            _eval_config({k: rng.choice(v) for k, v in fallback.items()})

    if not all_results:
        raise RuntimeError(
            "Hyperparameter search produced no successful trials — inspect the "
            "FAILED lines above (most often: not enough labeled windows for the "
            "requested lookbacks/fractions)."
        )

    all_results.sort(key=lambda r: r["selection_val_mcc"], reverse=True)
    best_hparams = all_results[0]["hparams"]
    best_val_mcc = all_results[0]["selection_val_mcc"]

    sep = "─" * 60
    print(f"\n{sep}")
    print("  BAYESIAN SEARCH RESULT")
    print(sep)
    print(
        f"  Trials         : {n_calls} ({SEARCH_SEEDS} seeds each, selection on val MCC)"
    )
    print(f"  Best val MCC   : {best_val_mcc:.4f}")
    print(f"  Best params    : {best_hparams}")

    return best_hparams, all_results


def save_search_results(
    all_results: List[Dict],
    path: Path,
    *,
    top_k: int = 4,
    metadata: Optional[Dict[str, Any]] = None,
) -> Path:
    """Persist search results as JSON: top-k configs plus EVERY trial tried.

    all_results is the sorted list returned by run_grid_search. top_configs
    carry full hparams dicts ready to paste into DEFAULT_HPARAMS or replay as
    a final training run. all_trials records the complete spec of every
    successful trial so a search's coverage of the space can be audited
    afterward (clustered specs = the optimizer was exploiting, not exploring).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def _row(rank: int, r: Dict) -> Dict:
        return {
            "rank": rank,
            "val_mcc_mean": float(r["selection_val_mcc"]),
            "val_mcc_std": float(r["aggregate"]["val_mcc"]["std"]),
            "hparams": r["hparams"],
        }

    payload = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "selection_metric": (
            "validation MCC, mean across search seeds, "
            "measured at the restored best-val-loss checkpoint"
        ),
        "n_trials_completed": len(all_results),
        "metadata": metadata or {},
        "top_configs": [_row(i + 1, r) for i, r in enumerate(all_results[:top_k])],
        "all_trials": [_row(i + 1, r) for i, r in enumerate(all_results)],
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return path


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------


def _window_summary_features(x: np.ndarray) -> np.ndarray:
    """Per-window summary: last value + mean of each feature → (N, 2*F)."""
    return np.concatenate([x[:, -1, :], x.mean(axis=1)], axis=1)


def _fit_logistic_numpy(
    X: np.ndarray,
    y: np.ndarray,
    *,
    lr: float = 0.1,
    epochs: int = 2000,
    l2: float = 1e-3,
) -> np.ndarray:
    """Tiny L2-regularised logistic regression (no sklearn dependency)."""
    Xb = np.concatenate([X, np.ones((len(X), 1))], axis=1)
    w = np.zeros(Xb.shape[1])
    for _ in range(epochs):
        p = 1.0 / (1.0 + np.exp(-np.clip(Xb @ w, -30, 30)))
        grad = Xb.T @ (p - y) / len(y) + l2 * np.append(w[:-1], 0.0)
        w -= lr * grad
    return w


def evaluate_baselines(
    data: TFTWindowData,
    labels: np.ndarray,
    *,
    window_positions: np.ndarray,
    embargo: int,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
) -> Dict[str, Dict[str, float]]:
    """Trivial baselines on the identical purged split the TFT uses.

    all_ones : the F1 floor any model must clear with zero skill.
    logistic : the bar any deep model must clear to justify its parameters.
               If the TFT can't beat this, sample size is the bottleneck.
    """
    train_idx, _val_idx, test_idx = purged_split_indices(
        window_positions,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
        embargo=embargo,
    )
    y_train, y_test = labels[train_idx], labels[test_idx]

    results: Dict[str, Dict[str, float]] = {
        "all_ones": classification_metrics(np.ones(len(test_idx)), y_test)
    }

    feats = _window_summary_features(data.x)
    feat_mean = feats[train_idx].mean(axis=0)
    feat_std = np.where(
        feats[train_idx].std(axis=0) < 1e-8, 1.0, feats[train_idx].std(axis=0)
    )
    f_train = (feats[train_idx] - feat_mean) / feat_std
    f_test = (feats[test_idx] - feat_mean) / feat_std

    w = _fit_logistic_numpy(f_train, y_train.astype(float))
    Xb_test = np.concatenate([f_test, np.ones((len(f_test), 1))], axis=1)
    p_test = 1.0 / (1.0 + np.exp(-np.clip(Xb_test @ w, -30, 30)))
    results["logistic"] = classification_metrics(p_test, y_test)
    return results


# ---------------------------------------------------------------------------
# Integration interface — the RAPTS pipeline entry point
# ---------------------------------------------------------------------------


class RegimeGate:
    """Production inference wrapper: load the bundle once, serve many signals.

    Usage:
        gate = RegimeGate(Path("data/bundles/<bundle>.pt"))
        prob = gate.signal(spread_data, arma_result)
        if prob >= 0.5:
            execute_trade(...)
    """

    def __init__(self, bundle_path: Path, *, device: Optional[str] = None) -> None:
        self.bundle = ModelBundle.load(Path(bundle_path))
        self.device = _resolve_device(device)
        self.model = self.bundle.build_model().to(self.device).eval()

    @property
    def threshold(self) -> float:
        return float(self.model.regime_threshold)

    def signal(self, spread_data: Any, arma_result: Any) -> float:
        """Gate probability in [0,1] for the most recent window.

        Feature preprocessing (lookback, half-life cap, scaler) is read from
        the bundle so it exactly matches the training run.
        """
        x_np = make_latest_window(
            spread_data,
            arma_result,
            input_length=self.bundle.model_config["input_length"],
            features=self.bundle.feature_names,
            half_life_window=self.bundle.half_life_window,
            max_half_life=self.bundle.label_config.max_half_life,
            scaling=self.bundle.scaling,
        )
        x = torch.as_tensor(x_np, dtype=torch.float32).to(self.device)
        return float(self.model.predict_regime_score(x).item())


def gate_signal(
    spread_data: Any,
    arma_result: Any,
    bundle: ModelBundle,
    *,
    device: Optional[str] = None,
) -> float:
    """One-shot gate probability from live spread pipeline outputs.

    Convenience wrapper that rebuilds the model on every call — fine for
    scripts and demos. For a live loop, hold a RegimeGate instance instead.
    """
    device = _resolve_device(device)

    x_np = make_latest_window(
        spread_data,
        arma_result,
        input_length=bundle.model_config["input_length"],
        features=bundle.feature_names,
        half_life_window=bundle.half_life_window,
        max_half_life=bundle.label_config.max_half_life,
        scaling=bundle.scaling,
    )

    model = bundle.build_model().to(device).eval()
    x = torch.as_tensor(x_np, dtype=torch.float32).to(device)
    return float(model.predict_regime_score(x).item())
