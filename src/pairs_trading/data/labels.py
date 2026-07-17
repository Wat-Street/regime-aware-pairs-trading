"""Regime label generation and feature windowing for the RAPTS TFT regime gate.

This module is the SINGLE source of truth for:
  - the feature vector fed to the TFT (build_feature_frame)
  - label generation (generate_regime_labels)
  - dataset construction for training (make_tft_dataset, build_multi_pair_dataset)
  - the latest inference window (make_latest_window)

Training and inference both build features through build_feature_frame(), so
they can never drift apart.

Label logic (entry-signal aware)
---------------------------------
Labels answer: "If we entered a pairs trade here, would it have profitably
reverted within H days?"

For each timestep t:
  1. Did |z_score| exceed entry_threshold? If not → -1 (no signal, excluded).
  2. Did |z_score| hit stop_loss within H days? If yes → 0 (stopped out).
  3. Did z cross back through reversion_target within H days? → 1 or 0.

Windows with label -1 are EXCLUDED from training via align_labels_to_dataset().
Only windows where a real trade signal exists are used to train the gate.

H selection
-----------
H is forced explicitly (train.py --force-h) and must be identical for every
pair in a training run. It is frozen in a LabelConfig and stored in the
ModelBundle — never recomputed at demo or inference time. (An earlier
half-life-based auto-selection mode was removed: it made runs incomparable.)

Public API
----------
build_feature_frame(spread_data, arma_result, ...) -> pd.DataFrame
generate_regime_labels(spread_data, *, forward_window, ...) -> pd.Series
align_labels_to_dataset(labels, target_index) -> (np.ndarray, np.ndarray)
compute_label_config(spread_data, pair_id, *, H, ...) -> LabelConfig
make_tft_dataset(spread_data, arma_result, *, input_length, label_config, ...) -> (TFTWindowData, np.ndarray, np.ndarray)
build_multi_pair_dataset(pairs, *, input_length, ...) -> (TFTWindowData, np.ndarray, np.ndarray)
make_latest_window(spread_data, arma_result, *, input_length, ...) -> np.ndarray
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np
import pandas as pd

from pairs_trading.data.schemas import SpreadData

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Canonical feature vector. Order matters — it defines the model input layout
# and is persisted in every ModelBundle for integrity checks.
#
# STANDARD: every statistic is computed by spread.py (the data layer's
# canonical implementations). This module only aligns, windows, and labels —
# it never does its own math.
#
# Five unitless regime indicators, each answering one distinct causal question
# about "will this stretch revert within H":
#
#   z_magnitude        : how stretched?      spread.compute_z_magnitude (|z|)
#   reversion_velocity : turning back or     spread.compute_reversion_velocity
#                        still diverging?    (5-day, signed toward equilibrium)
#   half_life          : fast enough for H?  spread.compute_half_life on causal
#                                            rolling windows, capped
#   vol_ratio          : vol burst vs own    spread.compute_volatility_ratio
#                        norm?               (sigma_t vs its 60d mean)
#   variance_ratio     : reversion currently spread.compute_variance_ratio
#                        operative?          (rolling VR(5), 60d)
#
# Scale-free -> pools cleanly across pairs.
FEATURE_NAMES: Tuple[str, ...] = (
    "z_magnitude",
    "reversion_velocity",
    "half_life",
    "vol_ratio",
    "variance_ratio",
)

DEFAULT_HALF_LIFE_WINDOW: int = 60
DEFAULT_MAX_HALF_LIFE: int = 63  # 3 calendar months
DEFAULT_ENTRY_THRESHOLD: float = 1.2
DEFAULT_REVERSION_TARGET: float = 0.5
DEFAULT_STOP_LOSS: float = 3.0


# ---------------------------------------------------------------------------
# LabelConfig — frozen per-run label configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LabelConfig:
    """Frozen label configuration.

    Computed once by compute_label_config() with an explicitly forced H and
    stored in ModelBundle. Never recomputed at demo time or inference time.
    """

    H: int  # forward lookahead horizon (trading days) — forced, shared per run
    half_life: float  # mean rolling half-life (diagnostic only)
    label_balance: float  # fraction positive in labeled (non -1) windows
    pair_id: str
    max_half_life: int = DEFAULT_MAX_HALF_LIFE
    entry_threshold: float = DEFAULT_ENTRY_THRESHOLD
    reversion_target: float = DEFAULT_REVERSION_TARGET
    stop_loss: float = DEFAULT_STOP_LOSS


# ---------------------------------------------------------------------------
# Label generation — entry-signal aware
# ---------------------------------------------------------------------------


def generate_regime_labels(
    spread_data: SpreadData,
    *,
    forward_window: int,
    entry_threshold: float = DEFAULT_ENTRY_THRESHOLD,
    reversion_target: float = DEFAULT_REVERSION_TARGET,
    stop_loss: float = DEFAULT_STOP_LOSS,
) -> pd.Series:
    """Binary regime labels aligned to the spread index.

    For each timestep t:
      1. |z_t| >= entry_threshold? If not → -1 (no trade signal; excluded
         from training by align_labels_to_dataset).
      2. Any |z| in forward window >= stop_loss? → 0 (stopped out).
      3. z crosses reversion_target within forward_window days? → 1 else 0.

    Label values:
       1  favorable regime (trade would have reverted)
       0  unfavorable / stopped out
      -1  no entry signal (excluded from training)

    The trailing forward_window timesteps cannot be labeled (their outcome is
    unknown) and are always -1.
    """
    if forward_window < 1:
        raise ValueError("forward_window must be at least 1")
    if entry_threshold <= 0:
        raise ValueError("entry_threshold must be positive")
    if stop_loss <= entry_threshold:
        raise ValueError("stop_loss must be greater than entry_threshold")

    z = spread_data.z_score.dropna().astype(float)
    values = z.to_numpy()
    n = len(values)
    labels = np.full(n, -1, dtype=np.int8)

    for t in range(n - forward_window):
        z_t = values[t]

        # Step 1: entry signal check
        if abs(z_t) < entry_threshold:
            continue

        forward = values[t + 1 : t + 1 + forward_window]

        # Step 2: stop loss check
        if np.any(np.abs(forward) >= stop_loss):
            labels[t] = 0
            continue

        # Step 3: reversion check
        if z_t > 0:
            labels[t] = 1 if np.any(forward <= reversion_target) else 0
        else:
            labels[t] = 1 if np.any(forward >= -reversion_target) else 0

    return pd.Series(labels, index=z.index, name="regime_label", dtype=np.int8)


# ---------------------------------------------------------------------------
# Label alignment
# ---------------------------------------------------------------------------


def align_labels_to_dataset(
    labels: pd.Series,
    target_index: pd.Index,
) -> Tuple[np.ndarray, np.ndarray]:
    """Align a label Series to TFTWindowData.target_index.

    Returns
    -------
    labels_array : np.ndarray (N,) float32 — full label array inc. -1s
    mask         : np.ndarray (N,) bool — True where label is 0 or 1
                   (i.e. a real trade signal existed at that window).
    """
    aligned = labels.reindex(target_index)

    if aligned.isna().any():
        raise ValueError(
            "Some window timestamps in target_index have no corresponding label. "
            "Ensure labels were generated from the same SpreadData used to build "
            "the TFTWindowData."
        )

    labels_array = aligned.to_numpy(dtype=np.int8)
    mask = labels_array >= 0  # True where label is 0 or 1 (not -1)

    return labels_array.astype(np.float32), mask


# ---------------------------------------------------------------------------
# Rolling half-life (causal, no look-ahead, max-capped)
# ---------------------------------------------------------------------------


def _compute_rolling_half_life(
    spread: pd.Series,
    *,
    window: int = DEFAULT_HALF_LIFE_WINDOW,
    max_half_life: int = DEFAULT_MAX_HALF_LIFE,
) -> pd.Series:
    """Causal rolling half-life with max cap and cointegration warning.

    - t < window → NaN (dropped downstream). NOT backfilled with full-sample value.
    - Invalid fits → forward-filled from most recent valid causal estimate.
    - Values > max_half_life → clipped. Warning if >10% of estimates are capped
      (signals temporary cointegration breakdown).

    Delegates the AR(1) fit to spread.compute_half_life — the canonical
    implementation in the data layer.
    """
    from pairs_trading.data.spread import compute_half_life

    spread_clean = spread.dropna().astype(float)
    values = spread_clean.to_numpy()
    n = len(values)
    half_lives = np.full(n, np.nan)
    n_valid = 0
    n_capped = 0

    for t in range(window, n):
        segment = pd.Series(values[t - window : t])
        try:
            hl = compute_half_life(segment)
            if np.isfinite(hl) and hl > 0:
                n_valid += 1
                if hl > max_half_life:
                    n_capped += 1
                half_lives[t] = min(hl, float(max_half_life))
        except Exception:
            pass

    if n_valid > 0 and n_capped / n_valid > 0.10:
        warnings.warn(
            f"{n_capped}/{n_valid} rolling half-life estimates "
            f"({100 * n_capped / n_valid:.1f}%) exceeded max_half_life={max_half_life}. "
            "The pair may have temporarily lost cointegration. Check your date range.",
            UserWarning,
            stacklevel=2,
        )

    out = pd.Series(half_lives, index=spread_clean.index, name="half_life")
    return out.ffill()  # leading NaNs remain; dropped in build_feature_frame


# ---------------------------------------------------------------------------
# Feature construction — the ONE place the feature vector is defined
# ---------------------------------------------------------------------------


def build_feature_frame(
    spread_data: SpreadData,
    arma_result,
    *,
    features: Sequence[str] = FEATURE_NAMES,
    half_life_window: int = DEFAULT_HALF_LIFE_WINDOW,
    max_half_life: int = DEFAULT_MAX_HALF_LIFE,
) -> pd.DataFrame:
    """Build the feature DataFrame for the requested feature names, in order.

    Pure alignment of spread.py outputs — no statistic is computed here.
    Every column is a direct SpreadData/ArmaGarchResult field or a spread.py
    function applied to the spread (see the FEATURE_NAMES comment).
    Every feature is causal: value at t uses only data up to t.
    """
    from pairs_trading.data.spread import (
        compute_reversion_velocity,
        compute_variance_ratio,
        compute_volatility_ratio,
        compute_z_magnitude,
    )

    spread = spread_data.spread
    registry = {
        "z_magnitude": lambda: compute_z_magnitude(spread_data.z_score),
        "reversion_velocity": lambda: compute_reversion_velocity(spread_data.z_score),
        "half_life": lambda: _compute_rolling_half_life(
            spread, window=half_life_window, max_half_life=max_half_life
        ),
        "vol_ratio": lambda: compute_volatility_ratio(
            arma_result.conditional_volatility
        ),
        "variance_ratio": lambda: compute_variance_ratio(spread),
    }

    unknown = [f for f in features if f not in registry]
    if unknown:
        raise ValueError(f"Unknown features {unknown}. Available: {list(registry)}")

    feature_df = pd.DataFrame({name: registry[name]() for name in features})
    feature_df = feature_df.replace([np.inf, -np.inf], np.nan).dropna()
    return feature_df


# ---------------------------------------------------------------------------
# LabelConfig computation — forced H, run once per pair, freeze result
# ---------------------------------------------------------------------------


def compute_label_config(
    spread_data: SpreadData,
    pair_id: str,
    *,
    H: int,
    max_half_life: int = DEFAULT_MAX_HALF_LIFE,
    half_life_window: int = DEFAULT_HALF_LIFE_WINDOW,
    entry_threshold: float = DEFAULT_ENTRY_THRESHOLD,
    reversion_target: float = DEFAULT_REVERSION_TARGET,
    stop_loss: float = DEFAULT_STOP_LOSS,
) -> LabelConfig:
    """Compute and freeze the label configuration for one pair at a forced H.

    H is supplied explicitly and must be the same for every pair in a run
    (build_multi_pair_dataset enforces this). half_life is recorded as a
    diagnostic; it does not influence H.
    """
    if H < 1:
        raise ValueError(f"H must be at least 1, got {H}")

    rolling_hl = _compute_rolling_half_life(
        spread_data.spread,
        window=half_life_window,
        max_half_life=max_half_life,
    )
    valid_hl = rolling_hl.dropna()
    mean_hl = float(valid_hl.mean()) if len(valid_hl) > 0 else float(max_half_life // 2)

    labels = generate_regime_labels(
        spread_data,
        forward_window=H,
        entry_threshold=entry_threshold,
        reversion_target=reversion_target,
        stop_loss=stop_loss,
    )
    n_fav = int((labels == 1).sum())
    n_unfav = int((labels == 0).sum())
    total = n_fav + n_unfav
    balance = n_fav / total if total else float("nan")

    if total == 0:
        warnings.warn(
            f"No labeled windows found for {pair_id} at H={H}. "
            "Check entry_threshold and spread data.",
            UserWarning,
        )

    return LabelConfig(
        H=H,
        half_life=mean_hl,
        label_balance=balance,
        pair_id=pair_id,
        max_half_life=max_half_life,
        entry_threshold=entry_threshold,
        reversion_target=reversion_target,
        stop_loss=stop_loss,
    )


# ---------------------------------------------------------------------------
# Dataset builder (single pair)
# ---------------------------------------------------------------------------


def make_tft_dataset(
    spread_data: SpreadData,
    arma_result,
    *,
    input_length: int,
    label_config: LabelConfig,
    features: Sequence[str] = FEATURE_NAMES,
    half_life_window: int = DEFAULT_HALF_LIFE_WINDOW,
    scaling=None,
) -> Tuple:
    """Build (TFTWindowData, labels_array, window_positions) for one pair.

    Returns
    -------
    data             : TFTWindowData with unscaled windows (unless scaling given)
    labels_array     : float32 array of shape (n_labeled,) — 0s and 1s only,
                       -1 windows already filtered out via mask
    window_positions : int array of shape (n_labeled,) — trading-day positions
                       of labeled windows in the original series.
                       Pass directly to purged_split_indices() and
                       train_tft_classifier().

    Scaling: DO NOT standardize here for training. Fit scaling inside
    train_tft_classifier() on the train slice only. For inference pass
    scaling=bundle.scaling.
    """
    from pairs_trading.models.tft import TFTWindowData

    if input_length < 2:
        raise ValueError(f"input_length must be at least 2, got {input_length}")

    feature_df = build_feature_frame(
        spread_data,
        arma_result,
        features=features,
        half_life_window=half_life_window,
        max_half_life=label_config.max_half_life,
    )

    if len(feature_df) < input_length + 1:
        raise ValueError(
            f"Not enough observations ({len(feature_df)}) for input_length={input_length}."
        )

    # Build sliding windows
    values = feature_df.to_numpy(dtype=np.float32)
    n_windows = len(values) - input_length
    x = np.stack([values[i : i + input_length] for i in range(n_windows)], axis=0)
    target_index = feature_df.index[input_length:]

    # Generate entry-signal aware labels using the frozen config
    label_series = generate_regime_labels(
        spread_data,
        forward_window=label_config.H,
        entry_threshold=label_config.entry_threshold,
        reversion_target=label_config.reversion_target,
        stop_loss=label_config.stop_loss,
    )

    labels_array, mask = align_labels_to_dataset(label_series, target_index)

    # Future z-score targets z[t+1 .. t+H] for the auxiliary forecasting head.
    # Labeled windows always have a complete future (labels are -1 for the
    # final H steps), but guard the slice anyway.
    H = label_config.H
    z_clean = spread_data.z_score.dropna().astype(float)
    z_values = z_clean.to_numpy()
    z_pos = z_clean.index.get_indexer(target_index)
    if (z_pos < 0).any():
        raise ValueError(
            "Window timestamps not found in the z-score index — features and "
            "z-scores were built from different data."
        )
    mask = mask & (z_pos + H < len(z_values))

    # Filter to signal windows only (mask=True means label is 0 or 1)
    x_masked = x[mask]
    labels_masked = labels_array[mask]
    future_z = (
        np.stack([z_values[p + 1 : p + 1 + H] for p in z_pos[mask]]).astype(np.float32)
        if mask.any()
        else np.zeros((0, H), dtype=np.float32)
    )
    entry_z = z_values[z_pos[mask]].astype(np.float32)

    # Map each labeled window to its trading-day position in the original
    # contiguous series (embargo distances are measured in trading days).
    orig_index = spread_data.spread.dropna().index
    indexer = orig_index.get_indexer(target_index)
    if (indexer < 0).any():
        raise ValueError(
            "Window timestamps not found in the original spread index — "
            "features and spread were built from different data."
        )
    window_positions = indexer[mask]

    x_out = x_masked.astype(np.float32)
    if scaling is not None:
        x_out = scaling.transform(x_out).astype(np.float32)

    data = TFTWindowData(
        x=x_out,
        target_index=target_index[mask],
        input_length=input_length,
        pair_id=label_config.pair_id,
        future_z=future_z,
        entry_z=entry_z,
    )

    return data, labels_masked, np.asarray(window_positions)


# ---------------------------------------------------------------------------
# Multi-pair dataset builder
# ---------------------------------------------------------------------------


def build_multi_pair_dataset(
    pairs: Sequence[Tuple],
    *,
    input_length: int,
    features: Sequence[str] = FEATURE_NAMES,
    half_life_window: int = DEFAULT_HALF_LIFE_WINDOW,
) -> Tuple:
    """Concatenate windows from multiple pairs into one training dataset,
    positioned on a SHARED CALENDAR-TIME axis.

    Every pair contributes windows across the whole date range. window_positions
    are derived from each window's actual calendar timestamp, mapped onto a
    single sorted union of all trading days seen across all pairs. This makes
    the downstream purged chronological split CALENDAR-TIME ALIGNED: the early
    date range is train for every pair, the middle is val for every pair, the
    most recent is test for every pair. No pair is held out wholesale.

    All pairs must share an identical label configuration (H, thresholds) —
    enforced here, because mixing horizons would make labels incomparable.

    Args:
        pairs: list of (spread_data, arma_result, label_config) tuples.
               Call compute_label_config() for each pair first.
        input_length: shared lookback window — must match across all pairs.
        half_life_window: rolling window for half-life computation.

    Returns:
        data:             concatenated TFTWindowData
        labels:           concatenated float32 labels array (0s and 1s only)
        window_positions: calendar-day-rank positions on a shared time axis
    """
    from pairs_trading.models.tft import TFTWindowData

    if not pairs:
        raise ValueError("build_multi_pair_dataset needs at least one pair")

    ref = pairs[0][2]
    for _, _, lc in pairs[1:]:
        shared = (
            "H",
            "entry_threshold",
            "reversion_target",
            "stop_loss",
            "max_half_life",
        )
        for attr in shared:
            if getattr(lc, attr) != getattr(ref, attr):
                raise ValueError(
                    f"All pairs must share the same label configuration: "
                    f"{lc.pair_id}.{attr}={getattr(lc, attr)} != "
                    f"{ref.pair_id}.{attr}={getattr(ref, attr)}"
                )

    all_x, all_labels, all_future_z, all_entry_z, all_timestamps = [], [], [], [], []

    for spread_data, arma_result, label_config in pairs:
        data, labels, _positions = make_tft_dataset(
            spread_data,
            arma_result,
            input_length=input_length,
            label_config=label_config,
            features=features,
            half_life_window=half_life_window,
        )
        all_x.append(data.x)
        all_labels.append(labels)
        all_future_z.append(data.future_z)
        all_entry_z.append(data.entry_z)
        # data.target_index holds the real pandas Timestamp of each window
        all_timestamps.append(np.asarray(data.target_index))

    combined_x = np.concatenate(all_x, axis=0)
    combined_labels = np.concatenate(all_labels, axis=0)
    combined_future_z = np.concatenate(all_future_z, axis=0)
    combined_entry_z = np.concatenate(all_entry_z, axis=0)
    combined_ts = np.concatenate(all_timestamps, axis=0)

    # Build a SHARED calendar axis: sorted union of all window timestamps.
    # Each window's position is its rank on this axis, so identical dates
    # from different pairs get the same position and land in the same split.
    unique_days = np.unique(combined_ts)
    day_to_pos = {ts: i for i, ts in enumerate(unique_days)}
    combined_positions = np.array([day_to_pos[ts] for ts in combined_ts], dtype=int)

    # Sort by calendar position so the chronological split cuts by real time.
    sort_idx = np.argsort(combined_positions, kind="stable")
    combined_x = combined_x[sort_idx]
    combined_labels = combined_labels[sort_idx]
    combined_future_z = combined_future_z[sort_idx]
    combined_entry_z = combined_entry_z[sort_idx]
    combined_positions = combined_positions[sort_idx]

    combined_data = TFTWindowData(
        x=combined_x.astype(np.float32),
        target_index=pd.RangeIndex(len(combined_x)),
        input_length=input_length,
        pair_id="multi_pair",
        future_z=combined_future_z.astype(np.float32),
        entry_z=combined_entry_z.astype(np.float32),
    )

    return combined_data, combined_labels, combined_positions


# ---------------------------------------------------------------------------
# Latest inference window
# ---------------------------------------------------------------------------


def make_latest_window(
    spread_data: SpreadData,
    arma_result,
    *,
    input_length: int,
    features: Sequence[str] = FEATURE_NAMES,
    half_life_window: int = DEFAULT_HALF_LIFE_WINDOW,
    max_half_life: int = DEFAULT_MAX_HALF_LIFE,
    scaling=None,
) -> np.ndarray:
    """Single model-ready window ending at the most recent timestep.

    Shape (1, input_length, len(features)). Uses build_feature_frame(), the
    same code path as training, so inference features can never drift. Pass
    features=bundle.feature_names, scaling=bundle.scaling and
    max_half_life=bundle.label_config.max_half_life so preprocessing exactly
    matches the training run.
    """
    feature_df = build_feature_frame(
        spread_data,
        arma_result,
        features=features,
        half_life_window=half_life_window,
        max_half_life=max_half_life,
    )
    if len(feature_df) < input_length:
        raise ValueError(
            f"Not enough observations ({len(feature_df)}) for input_length={input_length}."
        )
    x = feature_df.to_numpy(dtype=np.float32)[-input_length:][None, ...]
    if scaling is not None:
        x = scaling.transform(x).astype(np.float32)
    return x
