"""Regime label generation for supervised classifier training.

Labels answer one question per window: "If we entered a pairs trade here,
would the spread have reverted profitably within H days?"

H SELECTION -- the key insight from this project's failures
-----------------------------------------------------------
H is not a constant to hardcode and not simply "2x half-life". H controls
LABEL INFORMATIVENESS:

  - H too small vs the pair's reversion speed  -> labels ~all 0
    (KO/PEP, H=10 vs half-life ~145d: model collapsed to all-zeros)
  - H too large vs the pair's reversion speed  -> labels ~all 1
    (V/MA, H=36 = 2x half-life of 18.3d: 76.6% positive, model collapsed
    softly toward all-ones; an all-ones classifier beat it on F1)

A label is only informative when the outcome is genuinely uncertain.
The principled rule: choose H so label balance lands near 50%, where each
label carries maximum information. ``sweep_forward_window`` implements
this -- run it before training instead of guessing.

Two labelling strategies:

- ``generate_regime_labels`` (recommended): entry-signal aware. Only labels
  windows where |z| exceeded ``entry_threshold`` (a trade would actually
  have fired). No-signal windows are marked -1 and excluded from training.

- ``generate_simple_reversion_labels``: unconditional, for sanity checks.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pairs_trading.data.schemas import SpreadData


def generate_regime_labels(
    spread_data: SpreadData,
    *,
    forward_window: int,
    entry_threshold: float = 1.5,
    reversion_target: float = 0.5,
    stop_loss: float = 3.0,
    fill_no_entry: bool = False,
) -> pd.Series:
    """
    Generate binary regime labels aligned to the spread index.

    For each timestep t:
      1. Was |z| above ``entry_threshold``? (would a trade have fired?)
      2. Did z cross back through ``reversion_target`` within
         ``forward_window`` days WITHOUT first hitting ``stop_loss``?

    Label values: 1 favorable, 0 unfavorable/stop-hit, -1 no signal.

    NOTE: ``forward_window`` has no default -- it must be chosen per pair
    via ``sweep_forward_window`` (see module docstring for why).
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

        if abs(z_t) < entry_threshold:
            if fill_no_entry:
                labels[t] = 0
            continue

        forward = values[t + 1: t + 1 + forward_window]

        if np.any(np.abs(forward) >= stop_loss):
            labels[t] = 0
            continue

        if z_t > 0:
            reverted = np.any(forward <= reversion_target)
        else:
            reverted = np.any(forward >= -reversion_target)

        labels[t] = 1 if reverted else 0

    for t in range(n - forward_window, n):
        labels[t] = 0 if fill_no_entry else -1

    return pd.Series(labels, index=z.index, name="regime_label", dtype=np.int8)


def generate_simple_reversion_labels(
    spread_data: SpreadData,
    *,
    forward_window: int,
    reversion_target: float = 0.5,
) -> pd.Series:
    """Unconditional reversion labels for every timestep (sanity checks)."""
    if forward_window < 1:
        raise ValueError("forward_window must be at least 1")

    z = spread_data.z_score.dropna().astype(float)
    values = z.to_numpy()
    n = len(values)
    labels = np.zeros(n, dtype=np.int8)

    for t in range(n - forward_window):
        z_t = values[t]
        forward = values[t + 1: t + 1 + forward_window]
        if z_t > 0:
            labels[t] = 1 if np.any(forward <= reversion_target) else 0
        else:
            labels[t] = 1 if np.any(forward >= -reversion_target) else 0

    return pd.Series(labels, index=z.index, name="regime_label", dtype=np.int8)


def sweep_forward_window(
    spread_data: SpreadData,
    *,
    candidates: tuple = (3, 5, 8, 10, 13, 16, 20, 25, 30, 36, 45),
    entry_threshold: float = 1.5,
    reversion_target: float = 0.5,
    stop_loss: float = 3.0,
    target_balance: float = 0.5,
) -> pd.DataFrame:
    """
    Sweep H candidates and report label balance for each.

    Pick the H whose positive-label fraction is closest to
    ``target_balance`` (default 0.5 -- maximum label informativeness).

    Returns a DataFrame sorted by the sweep order with columns:
        H, n_favorable, n_unfavorable, n_labeled, balance, distance_to_target
    The recommended H is ``df.loc[df.distance_to_target.idxmin(), "H"]``.
    """
    rows = []
    for h in candidates:
        labels = generate_regime_labels(
            spread_data,
            forward_window   = h,
            entry_threshold  = entry_threshold,
            reversion_target = reversion_target,
            stop_loss        = stop_loss,
            fill_no_entry    = False,
        )
        n_fav   = int((labels == 1).sum())
        n_unfav = int((labels == 0).sum())
        total   = n_fav + n_unfav
        balance = n_fav / total if total else float("nan")
        rows.append({
            "H": h,
            "n_favorable": n_fav,
            "n_unfavorable": n_unfav,
            "n_labeled": total,
            "balance": balance,
            "distance_to_target": abs(balance - target_balance) if total else float("inf"),
        })
    return pd.DataFrame(rows)


def recommend_forward_window(
    spread_data: SpreadData,
    **sweep_kwargs,
) -> tuple:
    """Convenience wrapper: returns (recommended_H, sweep_dataframe)."""
    df = sweep_forward_window(spread_data, **sweep_kwargs)
    best = df.loc[df["distance_to_target"].idxmin()]
    return int(best["H"]), df


def align_labels_to_dataset(
    labels: pd.Series,
    target_index: pd.Index,
) -> tuple:
    """
    Align a label Series to ``TFTWindowData.target_index``.

    Returns
    -------
    labels_array : np.ndarray (N,) float32
    mask         : np.ndarray (N,) bool -- True where label is 0/1.
                   ``np.where(mask)[0]`` gives the trading-day positions
                   needed by ``purged_split_indices`` for embargo purging.
    """
    aligned = labels.reindex(target_index)

    if aligned.isna().any():
        raise ValueError(
            "Some window timestamps in target_index have no corresponding label. "
            "Ensure labels were generated from the same SpreadData used to build "
            "the TFTWindowData."
        )

    labels_array = aligned.to_numpy(dtype=np.int8)
    mask = labels_array >= 0

    return labels_array.astype(np.float32), mask