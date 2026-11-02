"""Expanding-window walk-forward splits over a common date calendar, purged by the label horizon."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Fold:
    k: int
    train_end: pd.Timestamp  # last decision date allowed in train/val (after purge)
    val_start: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def walk_forward_folds(
    calendar: pd.DatetimeIndex,
    oos_start: pd.Timestamp,
    *,
    n_folds: int,
    horizon: int,
    val_fraction: float,
) -> List[Fold]:
    """Split dates after ``oos_start`` into ``n_folds`` consecutive test blocks.

    Fold k trains on every decision before its test block, minus a purge gap of
    ``horizon`` trading days (a label at t looks at t+1..t+H, so train decisions
    closer than H days to the test block would leak test-period prices into
    training labels). The last ``val_fraction`` of the train dates (again purged
    by ``horizon``) is held out for early stopping.
    """
    cal = pd.DatetimeIndex(sorted(set(calendar)))
    oos = cal[cal > oos_start]
    if len(oos) < n_folds * 20:
        raise ValueError("out-of-sample period too short for the requested folds")
    blocks = np.array_split(np.arange(len(oos)), n_folds)
    folds: List[Fold] = []
    for k, blk in enumerate(blocks):
        test_start, test_end = oos[blk[0]], oos[blk[-1]]
        pos = cal.get_loc(test_start)
        train_end = cal[max(pos - horizon - 1, 0)]
        train_dates = cal[cal <= train_end]
        n_val = max(int(len(train_dates) * val_fraction), 1)
        val_start = train_dates[-n_val]
        folds.append(Fold(k, train_end, val_start, test_start, test_end))
    return folds


def fold_masks(dates: np.ndarray, fold: Fold, horizon: int, calendar: pd.DatetimeIndex):
    """Boolean masks (train, val, test) over decision ``dates`` for one fold."""
    d = pd.DatetimeIndex(dates)
    cal = pd.DatetimeIndex(sorted(set(calendar)))
    # purge train decisions whose label window overlaps the validation block
    vpos = cal.get_loc(fold.val_start)
    train_cut = cal[max(vpos - horizon - 1, 0)]
    train = np.asarray(d <= train_cut)
    val = np.asarray((d >= fold.val_start) & (d <= fold.train_end))
    test = np.asarray((d >= fold.test_start) & (d <= fold.test_end))
    return train, val, test
