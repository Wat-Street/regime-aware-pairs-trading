"""Event backtest that executes EXACTLY the rule the labels encode (``TradeRules``).

Per pair, at each close t:
  in a trade : mark to market, then exit at first of target / stop / H days held
  flat       : enter if |z_t| >= entry_z AND gate p_t >= threshold
               side = -sign(z_t); units = capital / (P_a + |beta| P_b)
PnL of one spread unit is dSpread = dP_a - beta dP_b (beta frozen from formation).
Costs: cost_bps per side on gross notional. Fills at the signal close.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from pairs_trading.benchmark.config import BacktestConfig
from pairs_trading.gate.config import TradeRules
from pairs_trading.gate.dataset import PairDataset


@dataclass
class Trade:
    pair_id: str
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    side: int  # +1 long spread (long A, short B), -1 short spread
    entry_z: float
    exit_z: float
    p_entry: float
    reason: str  # target | stop | time | end
    days: int
    pnl: float


def backtest_pair(
    ds: PairDataset,
    probs: pd.Series,
    start: pd.Timestamp,
    end: pd.Timestamp,
    rules: TradeRules,
    cfg: BacktestConfig,
    threshold: float,
) -> Tuple[pd.Series, List[Trade]]:
    """Return (daily PnL in $, trades) for one pair over [start, end]."""
    z = ds.z.loc[start:end]
    spread = ds.spread.reindex(z.index)
    pa = ds.price_a.reindex(z.index)
    pb = ds.price_b.reindex(z.index)
    p = probs.reindex(z.index).fillna(0.0)
    beta = ds.hedge_ratio
    cost = cfg.cost_bps * 1e-4

    pnl = np.zeros(len(z))
    trades: List[Trade] = []
    side = 0
    units = held = 0.0
    entry_i = 0
    zv, sv, pav, pbv, pv = (
        z.to_numpy(),
        spread.to_numpy(),
        pa.to_numpy(),
        pb.to_numpy(),
        p.to_numpy(),
    )

    def close(i: int, reason: str) -> None:
        nonlocal side
        fee = cost * units * (pav[i] + abs(beta) * pbv[i])
        pnl[i] -= fee
        trade_pnl = float(pnl[entry_i : i + 1].sum())
        trades.append(
            Trade(
                ds.pair_id,
                z.index[entry_i],
                z.index[i],
                side,
                float(zv[entry_i]),
                float(zv[i]),
                float(pv[entry_i]),
                reason,
                int(held),
                trade_pnl,
            )
        )
        side = 0

    for i in range(len(z)):
        if side != 0:
            held += 1
            pnl[i] += side * units * (sv[i] - sv[i - 1])
            hit_target = (
                zv[i] >= -rules.target_z if side > 0 else zv[i] <= rules.target_z
            )
            if hit_target:
                close(i, "target")
            elif abs(zv[i]) >= rules.stop_z:
                close(i, "stop")
            elif held >= rules.horizon:
                close(i, "time")
            elif i == len(z) - 1:
                close(i, "end")
            continue  # no same-day re-entry after an exit
        if (
            np.isfinite(zv[i])
            and abs(zv[i]) >= rules.entry_z
            and pv[i] >= threshold
            and i < len(z) - 1
        ):
            side = -1 if zv[i] > 0 else 1
            units = cfg.capital_per_pair / (pav[i] + abs(beta) * pbv[i])
            held, entry_i = 0, i
            pnl[i] -= cost * units * (pav[i] + abs(beta) * pbv[i])

    return pd.Series(pnl, index=z.index, name=ds.pair_id), trades


def backtest_portfolio(
    datasets: Dict[str, PairDataset],
    probs: Dict[str, pd.Series],
    start: pd.Timestamp,
    end: pd.Timestamp,
    rules: TradeRules,
    cfg: BacktestConfig,
    threshold: float,
) -> Tuple[pd.Series, pd.DataFrame]:
    """Equal capital per pair; returns (daily portfolio return, trades table)."""
    pnls, trades = [], []
    for pid, ds in datasets.items():
        s, t = backtest_pair(
            ds,
            probs.get(pid, pd.Series(dtype=float)),
            start,
            end,
            rules,
            cfg,
            threshold,
        )
        pnls.append(s)
        trades.extend(t)
    daily = pd.concat(pnls, axis=1).fillna(0.0).sum(axis=1)
    ret = daily / (cfg.capital_per_pair * len(datasets))
    tdf = pd.DataFrame([t.__dict__ for t in trades])
    return ret.rename("return"), tdf
