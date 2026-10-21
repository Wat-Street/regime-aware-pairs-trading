"""Regime label generation for supervised classifier training.

Labels answer one question per window: "If we entered a pairs trade here,
would the spread have reverted profitably within H days?"

A label of 1 means YES — the spread reverted toward zero within the forward
window, suggesting a stable mean-reverting regime at that point in time.
A label of 0 means NO — the spread did not revert (or blew out further),
suggesting an unfavorable or broken regime.

These binary labels are the supervision signal for
``WaveletTransformerClassifier`` via ``train_wavelet_classifier``.

Two labelling strategies are provided:

- ``generate_regime_labels`` (recommended): entry-signal aware. Only labels
  windows where the z-score was actually extreme enough to trigger a trade
  (above ``entry_threshold``). Windows where no trade would have been entered
  are marked -1 (ignored) by default, or can be filled as 0.

- ``generate_simple_reversion_labels``: unconditional. Labels every window
  based purely on whether the spread moved toward zero in the next H days.
  Simpler to reason about; useful for initial experiments.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pairs_trading.data.schemas import SpreadData


def generate_regime_labels(
    spread_data: SpreadData,
    *,
    forward_window: int = 5,
    entry_threshold: float = 1.5,
    reversion_target: float = 0.5,
    stop_loss: float = 3.0,
    fill_no_entry: bool = False,
) -> pd.Series:
    """
    Generate binary regime labels aligned to the spread index.

    For each timestep t the function asks:
      1. Was the absolute z-score above ``entry_threshold``?
         (i.e. would we have actually considered a trade here?)
      2. Did the z-score cross back through ``reversion_target`` within
         the next ``forward_window`` days WITHOUT first hitting ``stop_loss``?

    Label values
    ------------
    1  : trade would have entered AND reverted successfully (favorable regime)
    0  : trade entered but did NOT revert, or stop-loss was hit (bad regime)
    -1 : z-score was not extreme enough to trigger a trade (no signal)
         Only present when ``fill_no_entry=False``.

    Parameters
    ----------
    spread_data:
        Output of ``compute_spread``. Must contain a valid ``z_score`` Series.
    forward_window:
        Number of trading days to look ahead for reversion. 5 = one week.
    entry_threshold:
        Minimum absolute z-score to consider a trade entry. Typical: 1.5–2.0.
    reversion_target:
        The z-score level the spread must cross back through to count as
        reverted. Typical: 0.0–0.5.
    stop_loss:
        If the absolute z-score exceeds this level before reversion occurs,
        the trade is labelled 0 regardless of later behaviour.
    fill_no_entry:
        If True, windows with no trade signal are labelled 0 instead of -1.
        Use this if you want a dense binary array with no masked values.

    Returns
    -------
    pd.Series of int8 (values 0/1, or 0/1/-1) indexed to spread_data.z_score.
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
        abs_z_t = abs(z_t)

        # No trade signal at this timestep
        if abs_z_t < entry_threshold:
            if fill_no_entry:
                labels[t] = 0
            continue

        # Direction of reversion: if z > 0 we expect it to fall, and vice versa
        forward = values[t + 1 : t + 1 + forward_window]

        # Check stop-loss first — did the spread blow out before reverting?
        stop_hit = np.any(np.abs(forward) >= stop_loss)
        if stop_hit:
            labels[t] = 0
            continue

        # Check reversion: did the spread cross back through reversion_target?
        if z_t > 0:
            reverted = np.any(forward <= reversion_target)
        else:
            reverted = np.any(forward >= -reversion_target)

        labels[t] = 1 if reverted else 0

    # Last forward_window steps can't be labelled — mark as no-entry or 0
    for t in range(n - forward_window, n):
        labels[t] = 0 if fill_no_entry else -1

    return pd.Series(labels, index=z.index, name="regime_label", dtype=np.int8)


def generate_simple_reversion_labels(
    spread_data: SpreadData,
    *,
    forward_window: int = 5,
    reversion_target: float = 0.5,
) -> pd.Series:
    """
    Unconditional reversion labels for every timestep.

    Simpler than ``generate_regime_labels``: ignores whether there was an
    actual trade signal, and labels every window 1 if the z-score crossed
    ``reversion_target`` within ``forward_window`` days.

    Useful for quick sanity checks and baseline experiments. For proper
    training, prefer ``generate_regime_labels``.
    """
    if forward_window < 1:
        raise ValueError("forward_window must be at least 1")

    z = spread_data.z_score.dropna().astype(float)
    values = z.to_numpy()
    n = len(values)

    labels = np.zeros(n, dtype=np.int8)

    for t in range(n - forward_window):
        z_t = values[t]
        forward = values[t + 1 : t + 1 + forward_window]
        if z_t > 0:
            labels[t] = 1 if np.any(forward <= reversion_target) else 0
        else:
            labels[t] = 1 if np.any(forward >= -reversion_target) else 0

    return pd.Series(labels, index=z.index, name="regime_label", dtype=np.int8)


def align_labels_to_dataset(
    labels: pd.Series,
    target_index: pd.Index,
) -> np.ndarray:
    """
    Align a label Series to the target index of a ``WaveletWindowData``.

    ``WaveletWindowData.target_index`` contains one timestamp per window
    (the first timestep of the forecast horizon). This function reindexes
    the label Series to match, dropping any windows whose timestamp has a
    -1 (no-signal) label.

    Returns
    -------
    labels_array : np.ndarray of shape ``(N,)`` with dtype float32, ready
        to pass directly to ``train_wavelet_classifier``.
    mask : np.ndarray of bool shape ``(N,)`` — True where label is valid (0 or 1).
        Use this to filter ``WaveletWindowData.x`` and ``y_components``
        before training if you used ``fill_no_entry=False``.
    """
    aligned = labels.reindex(target_index)

    if aligned.isna().any():
        raise ValueError(
            "Some window timestamps in target_index have no corresponding label. "
            "Ensure labels were generated from the same SpreadData used to build "
            "the WaveletWindowData."
        )

    labels_array = aligned.to_numpy(dtype=np.int8)
    mask = labels_array >= 0  # False where label == -1 (no-signal windows)

    return labels_array.astype(np.float32), mask