"""Model registry: every regime gate reduces to ``(B, L, F) float32 -> logits (B,)``.

Each entry wraps a team member's network without editing their file:

  tft          kiana/regime-gate      models/tft.py           TFTClassifier
  autoformer   shruti/autoformer      models/autoformer.py    AutoformerRegimeClassifier
  wtransformer tft-wavelet-model-dev  models/w_transformer.py WaveletTransformerClassifier
  logistic     baseline: linear model on window summary stats (same trainer)

Optional ``transform``: a CAUSAL per-pair representation hook applied to the
canonical feature frame before windowing (the W-Transformer uses it for Haar
MODWT bands). Same information in, model-specific representation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Sequence

import pandas as pd
import torch
from torch import nn

Builder = Callable[..., nn.Module]
TransformFactory = Callable[..., Callable[[pd.DataFrame], pd.DataFrame]]


@dataclass(frozen=True)
class ModelSpec:
    name: str
    build: (
        Builder  # build(input_length, n_features, feature_names, **kwargs) -> nn.Module
    )
    transform: Optional[TransformFactory] = (
        None  # transform(input_length, **kwargs) -> fn
    )
    description: str = ""


MODEL_REGISTRY: Dict[str, ModelSpec] = {}


def register(
    name: str, *, transform: Optional[TransformFactory] = None, description: str = ""
):
    def deco(fn: Builder) -> Builder:
        MODEL_REGISTRY[name] = ModelSpec(name, fn, transform, description)
        return fn

    return deco


def available_models() -> list[str]:
    return sorted(MODEL_REGISTRY)


def get_spec(name: str) -> ModelSpec:
    try:
        return MODEL_REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"unknown model {name!r}; available: {available_models()}"
        ) from None


# ---------------------------------------------------------------------------
# TFT (kiana/regime-gate)
# ---------------------------------------------------------------------------


@register(
    "tft", description="Temporal Fusion Transformer gate (VSN -> LSTM -> attention)"
)
def _build_tft(
    input_length: int, n_features: int, feature_names: Sequence[str], **kw: Any
) -> nn.Module:
    from pairs_trading.models.tft import TFTClassifier

    kw.setdefault(
        "forecast_horizon", 0
    )  # aux forecast head off: same loss for every model
    return TFTClassifier(
        input_length=input_length, feature_names=tuple(feature_names), **kw
    )


# ---------------------------------------------------------------------------
# Autoformer (shruti/autoformer)
# ---------------------------------------------------------------------------


@register(
    "autoformer",
    description="Autoformer gate (series decomposition + auto-correlation)",
)
def _build_autoformer(
    input_length: int, n_features: int, feature_names: Sequence[str], **kw: Any
) -> nn.Module:
    from pairs_trading.models.autoformer import AutoformerRegimeClassifier

    # moving-average kernel must fit inside the window and be odd
    max_kernel = input_length - 1 if (input_length - 1) % 2 else input_length - 2
    kw["moving_avg"] = min(kw.get("moving_avg", 25), max(3, max_kernel))
    return AutoformerRegimeClassifier(
        input_length=input_length, num_features=n_features, **kw
    )


# ---------------------------------------------------------------------------
# W-Transformer (tft-wavelet-model-dev)
# ---------------------------------------------------------------------------


def _wavelet_levels(input_length: int, levels: Optional[int]) -> int:
    from pairs_trading.models.w_transformer import recommended_wavelet_levels

    return levels if levels is not None else recommended_wavelet_levels(input_length)


def _wavelet_transform(input_length: int, levels: Optional[int] = None, **_: Any):
    from pairs_trading.models.w_transformer import _decompose_feature_matrix

    lv = _wavelet_levels(input_length, levels)

    def transform(frame: pd.DataFrame) -> pd.DataFrame:
        comps, names = _decompose_feature_matrix(frame, levels=lv, mode="causal")
        return pd.DataFrame(comps, index=frame.index, columns=list(names))

    return transform


class _LogitHead(nn.Module):
    """W-Transformer's classifier ends in nn.Sigmoid; strip it so every model returns logits."""

    def __init__(self, classifier: nn.Module) -> None:
        super().__init__()
        if isinstance(classifier.regime_head[-1], nn.Sigmoid):
            classifier.regime_head[-1] = nn.Identity()
        self.classifier = classifier

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


@register(
    "wtransformer",
    transform=_wavelet_transform,
    description="Wavelet (Haar MODWT) Transformer gate, one encoder per band",
)
def _build_wtransformer(
    input_length: int, n_features: int, feature_names: Sequence[str], **kw: Any
) -> nn.Module:
    from pairs_trading.models.w_transformer import (
        WaveletTransformerClassifier,
        WaveletTransformerForecaster,
    )

    kw.pop("levels", None)  # consumed by the transform; n_features already reflects it
    fusion_dim = kw.pop("fusion_dim", 32)
    clf_dropout = kw.pop("classifier_dropout", 0.1)
    forecaster = WaveletTransformerForecaster(
        input_length=input_length,
        forecast_horizon=1,
        num_components=n_features,
        d_model=kw.pop("d_model", 16),
        num_heads=kw.pop("num_heads", 4),
        num_encoder_layers=kw.pop("num_encoder_layers", 1),
        dim_feedforward=kw.pop("dim_feedforward", 32),
        dropout=kw.pop("dropout", 0.1),
        **kw,
    )
    return _LogitHead(
        WaveletTransformerClassifier(
            forecaster, fusion_dim=fusion_dim, dropout=clf_dropout
        )
    )


# ---------------------------------------------------------------------------
# Baseline
# ---------------------------------------------------------------------------


class _SummaryLogistic(nn.Module):
    """Logistic regression on [last value, window mean, window std] of every feature."""

    def __init__(self, n_features: int) -> None:
        super().__init__()
        self.linear = nn.Linear(3 * n_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = torch.cat([x[:, -1, :], x.mean(dim=1), x.std(dim=1)], dim=-1)
        return self.linear(feats).squeeze(-1)


@register(
    "logistic", description="Baseline: logistic regression on window summary stats"
)
def _build_logistic(
    input_length: int, n_features: int, feature_names: Sequence[str], **kw: Any
) -> nn.Module:
    return _SummaryLogistic(n_features)
