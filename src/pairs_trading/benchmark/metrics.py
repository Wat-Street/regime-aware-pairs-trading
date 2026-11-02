"""Classification metrics (is the gate's probability any good?) and trading metrics (does it make money?)."""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
from scipy.stats import rankdata

TRADING_DAYS = 252


def roc_auc(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y).astype(bool)
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(p)
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def classification_metrics(
    y: np.ndarray, p: np.ndarray, threshold: float
) -> Dict[str, float]:
    y = np.asarray(y).astype(float)
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    pred = p >= threshold
    tp = float(((pred) & (y == 1)).sum())
    return {
        "n_signals": float(len(y)),
        "base_rate": float(y.mean()) if len(y) else float("nan"),
        "auc": roc_auc(y, p),
        "brier": float(np.mean((p - y) ** 2)),
        "log_loss": float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))),
        "accuracy": float(np.mean(pred == (y == 1))),
        "take_rate": float(pred.mean()),
        "precision": tp / pred.sum() if pred.sum() else float("nan"),
        "recall": tp / (y == 1).sum() if (y == 1).sum() else float("nan"),
    }


def trading_metrics(
    ret: pd.Series, trades: pd.DataFrame, capital: float
) -> Dict[str, float]:
    ret = ret.fillna(0.0)
    equity = (1 + ret).cumprod()
    dd = equity / equity.cummax() - 1
    vol = ret.std()
    years = max(len(ret) / TRADING_DAYS, 1e-9)
    n = len(trades)
    wins = trades["pnl"] > 0 if n else pd.Series(dtype=bool)
    gross_win = trades.loc[wins, "pnl"].sum() if n else 0.0
    gross_loss = -trades.loc[~wins, "pnl"].sum() if n else 0.0
    return {
        "sharpe": float(ret.mean() / vol * np.sqrt(TRADING_DAYS)) if vol > 0 else 0.0,
        "ann_return": float(equity.iloc[-1] ** (1 / years) - 1) if len(ret) else 0.0,
        "ann_vol": float(vol * np.sqrt(TRADING_DAYS)),
        "max_drawdown": float(dd.min()) if len(dd) else 0.0,
        "n_trades": float(n),
        "win_rate": float(wins.mean()) if n else float("nan"),
        "avg_trade_ret_bps": float(trades["pnl"].mean() / capital * 1e4)
        if n
        else float("nan"),
        "profit_factor": float(gross_win / gross_loss)
        if gross_loss > 0
        else float("nan"),
        "avg_hold_days": float(trades["days"].mean()) if n else float("nan"),
    }
