"""Benchmark orchestration: data -> datasets -> walk-forward per model/seed -> metrics + backtest -> report."""

from __future__ import annotations

import logging
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from pairs_trading.benchmark.backtest import backtest_portfolio
from pairs_trading.benchmark.config import BenchmarkConfig
from pairs_trading.benchmark.data import load_universe
from pairs_trading.benchmark.metrics import classification_metrics, trading_metrics
from pairs_trading.benchmark.splits import fold_masks, walk_forward_folds
from pairs_trading.gate.config import GateConfig
from pairs_trading.gate.dataset import (
    PairDataset,
    WindowSet,
    build_pair_dataset,
    make_windows,
)
from pairs_trading.gate.model import RegimeGate

log = logging.getLogger("pairs_trading.benchmark")

REFERENCE_STRATEGIES = ("ungated", "oracle")


@dataclass
class BenchmarkResult:
    out_dir: Path
    summary: pd.DataFrame  # one row per strategy (mean over seeds)
    per_seed: pd.DataFrame
    predictions: pd.DataFrame


def build_datasets(cfg: BenchmarkConfig) -> Dict[str, PairDataset]:
    datasets: Dict[str, PairDataset] = {}
    for pair, a, b in load_universe(cfg.data):
        try:
            ds = build_pair_dataset(
                pair, a, b, rules=cfg.rules, formation_end=cfg.data.formation_end
            )
        except Exception as exc:
            log.warning("skipping %s: %s", pair.pair_id, exc)
            continue
        if (
            cfg.data.max_coint_pvalue is not None
            and ds.cointegration_pvalue > cfg.data.max_coint_pvalue
        ):
            log.info(
                "dropping %s (coint p=%.3f)", pair.pair_id, ds.cointegration_pvalue
            )
            continue
        datasets[pair.pair_id] = ds
    if not datasets:
        raise RuntimeError("no usable pairs (check data source / dates / network)")
    return datasets


def _pool(
    datasets: Dict[str, PairDataset], L: int, transform, select: str
) -> WindowSet:
    return WindowSet.concat(
        [make_windows(ds, L, transform, select) for ds in datasets.values()]
    )


def _to_series_by_pair(df: pd.DataFrame, col: str) -> Dict[str, pd.Series]:
    return {
        pid: g.set_index("date")[col].astype(float) for pid, g in df.groupby("pair_id")
    }


def run_benchmark(
    cfg: BenchmarkConfig,
    out_root: Optional[Path] = None,
    config_path: Optional[Path] = None,
) -> BenchmarkResult:
    t_start = time.time()
    out_dir = (
        Path(out_root or cfg.output_dir)
        / f"{cfg.name}_{time.strftime('%Y%m%d_%H%M%S')}"
    )
    (out_dir / "models").mkdir(parents=True, exist_ok=True)
    rules, L = cfg.rules, cfg.gate.input_length

    datasets = build_datasets(cfg)
    log.info("pairs: %s", ", ".join(datasets))
    calendar = pd.DatetimeIndex(
        sorted(set().union(*[set(d.features.index) for d in datasets.values()]))
    )
    folds = walk_forward_folds(
        calendar,
        pd.Timestamp(cfg.data.formation_end),
        n_folds=cfg.walk_forward.n_folds,
        horizon=rules.horizon,
        val_fraction=cfg.walk_forward.val_fraction,
    )
    oos_start, oos_end = folds[0].test_start, folds[-1].test_end

    # Reference windows (no transform) define the decision set shared by everyone.
    ref_signal = _pool(datasets, L, None, "signal")

    pred_rows: List[pd.DataFrame] = []
    fit_info: List[dict] = []

    for name in cfg.models:
        tf = RegimeGate.frame_transform(name, cfg.gate)
        labeled = _pool(datasets, L, tf, "labeled")
        signal = _pool(datasets, L, tf, "signal")
        for seed in cfg.seeds:
            gcfg = GateConfig(**{**cfg.gate.to_dict(), "seed": seed})
            for fold in folds:
                tr, va, _ = fold_masks(labeled.dates, fold, rules.horizon, calendar)
                _, _, te = fold_masks(signal.dates, fold, rules.horizon, calendar)
                if tr.sum() < cfg.walk_forward.min_train_signals:
                    log.warning(
                        "%s fold %d: only %d train signals, skipped",
                        name,
                        fold.k,
                        tr.sum(),
                    )
                    continue
                gate = RegimeGate(
                    name, labeled.x.shape[-1], labeled.feature_names, gcfg
                )
                hist = gate.fit(labeled.subset(tr), labeled.subset(va))
                test = signal.subset(te)
                p = gate.predict_proba(test)
                pred_rows.append(
                    pd.DataFrame(
                        {
                            "strategy": name,
                            "seed": seed,
                            "fold": fold.k,
                            "pair_id": test.pair_ids,
                            "date": pd.DatetimeIndex(test.dates),
                            "p": p,
                            "label": test.labels,
                        }
                    )
                )
                fit_info.append(
                    {
                        "strategy": name,
                        "seed": seed,
                        "fold": fold.k,
                        "n_train": int(tr.sum()),
                        "n_val": int(va.sum()),
                        "n_test": int(te.sum()),
                        "best_epoch": hist.best_epoch,
                        "epochs_run": len(hist.train_loss),
                        "train_seconds": round(hist.seconds, 2),
                        "n_params": gate.n_parameters,
                    }
                )
                log.info(
                    "%-12s seed %d fold %d: train %d val %d test %d | %.1fs",
                    name,
                    seed,
                    fold.k,
                    tr.sum(),
                    va.sum(),
                    te.sum(),
                    hist.seconds,
                )
            gate.save(out_dir / "models" / f"{name}_seed{seed}.pt")

    # Reference strategies on the same decision set
    oos_mask = (pd.DatetimeIndex(ref_signal.dates) >= oos_start) & (
        pd.DatetimeIndex(ref_signal.dates) <= oos_end
    )
    ref = ref_signal.subset(np.asarray(oos_mask))
    base = pd.DataFrame(
        {
            "seed": 0,
            "fold": -1,
            "pair_id": ref.pair_ids,
            "date": pd.DatetimeIndex(ref.dates),
            "label": ref.labels,
        }
    )
    pred_rows.append(base.assign(strategy="ungated", p=1.0))
    pred_rows.append(base.assign(strategy="oracle", p=(ref.labels == 1).astype(float)))
    preds = pd.concat(pred_rows, ignore_index=True)

    # Score every (strategy, seed)
    rows = []
    cap = cfg.backtest.capital_per_pair
    equity_curves = {}
    for (strategy, seed), g in preds.groupby(["strategy", "seed"], sort=False):
        lab = g[g["label"] >= 0]
        cls = classification_metrics(
            lab["label"].to_numpy(), lab["p"].to_numpy(), cfg.gate.threshold
        )
        ret, trades = backtest_portfolio(
            datasets,
            _to_series_by_pair(g, "p"),
            oos_start,
            oos_end,
            rules,
            cfg.backtest,
            cfg.gate.threshold,
        )
        trd = trading_metrics(ret, trades, cap)
        rows.append({"strategy": strategy, "seed": seed, **cls, **trd})
        equity_curves[(strategy, seed)] = (1 + ret).cumprod()
        if len(trades):
            trades.assign(strategy=strategy, seed=seed).to_csv(
                out_dir / "trades.csv.gz",
                mode="a",
                header=not (out_dir / "trades.csv.gz").exists(),
                index=False,
            )

    per_seed = pd.DataFrame(rows)
    fits = pd.DataFrame(fit_info)
    summary = _summarize(per_seed, fits, cfg)

    preds.to_csv(out_dir / "predictions.csv.gz", index=False)
    per_seed.to_csv(out_dir / "metrics_per_seed.csv", index=False)
    summary.to_csv(out_dir / "summary.csv")
    fits.to_csv(out_dir / "fits.csv", index=False)
    pd.DataFrame(equity_curves).to_csv(out_dir / "equity.csv")
    _write_pair_table(datasets, out_dir)
    _plot(equity_curves, out_dir)
    if config_path is not None:
        shutil.copy(config_path, out_dir / "config.toml")
    _write_report(cfg, summary, datasets, folds, out_dir, time.time() - t_start)
    log.info("done in %.0fs -> %s", time.time() - t_start, out_dir)
    return BenchmarkResult(out_dir, summary, per_seed, preds)


def _summarize(
    per_seed: pd.DataFrame, fits: pd.DataFrame, cfg: BenchmarkConfig
) -> pd.DataFrame:
    num = per_seed.drop(columns=["seed"]).groupby("strategy", sort=False)
    summary = num.mean()
    if per_seed["seed"].nunique() > 1:
        for col in ("auc", "sharpe"):
            summary[f"{col}_std"] = num[col].std()
    if len(fits):
        f = fits.groupby("strategy").agg(
            train_seconds=("train_seconds", "sum"), n_params=("n_params", "max")
        )
        f["train_seconds"] /= max(len(cfg.seeds), 1)
        summary = summary.join(f)
    order = [m for m in cfg.models if m in summary.index] + [
        r for r in REFERENCE_STRATEGIES if r in summary.index
    ]
    return summary.loc[order]


def _write_pair_table(datasets: Dict[str, PairDataset], out_dir: Path) -> None:
    pd.DataFrame(
        [
            {
                "pair_id": pid,
                "hedge_ratio": ds.hedge_ratio,
                "coint_pvalue_formation": ds.cointegration_pvalue,
                "n_days": len(ds.features),
                "n_signals": int((ds.labels >= 0).sum()),
                "label_balance": float(
                    (ds.labels == 1).sum() / max((ds.labels >= 0).sum(), 1)
                ),
            }
            for pid, ds in datasets.items()
        ]
    ).to_csv(out_dir / "pairs.csv", index=False)


def _plot(curves: dict, out_dir: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # pragma: no cover
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    by_strategy: Dict[str, list] = {}
    for (s, _), c in curves.items():
        by_strategy.setdefault(s, []).append(c)
    for s, cs in by_strategy.items():
        mean = pd.concat(cs, axis=1).mean(axis=1)
        style = "--" if s in REFERENCE_STRATEGIES else "-"
        ax.plot(mean.index, mean.values, style, label=s, linewidth=1.4)
    ax.set_title("Out-of-sample equity (mean over seeds)")
    ax.set_ylabel("growth of $1")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "equity.png", dpi=130)
    plt.close(fig)


def _fmt(v: float, kind: str) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "–"
    if kind == "pct":
        return f"{v * 100:.1f}%"
    if kind == "int":
        return f"{v:,.0f}"
    return f"{v:.3f}" if kind == "3" else f"{v:.2f}"


def _write_report(cfg, summary, datasets, folds, out_dir: Path, seconds: float) -> None:
    cols = [
        ("auc", "AUC", "3"),
        ("brier", "Brier", "3"),
        ("take_rate", "Take rate", "pct"),
        ("precision", "Precision", "pct"),
        ("sharpe", "Sharpe", "2"),
        ("ann_return", "Ann. ret", "pct"),
        ("max_drawdown", "Max DD", "pct"),
        ("n_trades", "Trades", "int"),
        ("win_rate", "Win rate", "pct"),
        ("avg_trade_ret_bps", "Avg trade (bps)", "2"),
        ("train_seconds", "Train s", "int"),
    ]
    cols = [c for c in cols if c[0] in summary.columns]
    head = "| Strategy | " + " | ".join(c[1] for c in cols) + " |"
    sep = "|---|" + "|".join("---:" for _ in cols) + "|"
    body = [
        f"| {s} | " + " | ".join(_fmt(summary.loc[s, c], k) for c, _, k in cols) + " |"
        for s in summary.index
    ]
    r = cfg.rules
    fold_lines = "\n".join(
        f"| {f.k} | ≤ {f.train_end.date()} | {f.val_start.date()} → {f.train_end.date()} | "
        f"{f.test_start.date()} → {f.test_end.date()} |"
        for f in folds
    )
    text = f"""# Regime-gate benchmark — {cfg.name}

Generated {time.strftime("%Y-%m-%d %H:%M")} in {seconds:.0f}s · seeds {list(cfg.seeds)} · \
{len(datasets)} pairs · source `{cfg.data.source}`

## Results (out-of-sample, mean over seeds)

{head}
{sep}
{chr(10).join(body)}

- **ungated**: take every entry signal (what the gate must beat).
- **oracle**: take only signals whose label is 1 (perfect gate; upper bound, not achievable).
- AUC / Brier / precision are on labeled OOS entry signals; the trading columns come from
  the backtest below. Take rate = share of signals the gate lets through at p ≥ {cfg.gate.threshold}.

![equity](equity.png)

## Setup

**Trade rule (labels and backtest use the same one):** enter when |z| ≥ {r.entry_z}
(z = {r.zscore_lookback}-day rolling z of the spread); exit at the first of z back through
±{r.target_z} (label 1), |z| ≥ {r.stop_z} (label 0), or {r.horizon} days held (label 0).
Costs {cfg.backtest.cost_bps} bps per side on gross notional, ${cfg.backtest.capital_per_pair:,.0f} per pair.

**Inputs (every model):** canonical features `{", ".join(next(iter(datasets.values())).features.columns)}`,
window {cfg.gate.input_length} days, standardised on the training slice. W-Transformer
additionally decomposes each feature into causal Haar MODWT bands.

**Training (every model):** AdamW lr {cfg.gate.learning_rate}, batch {cfg.gate.batch_size},
≤ {cfg.gate.epochs} epochs, early stop on val loss (patience {cfg.gate.early_stopping_patience}),
class-balanced BCE.

**Leakage controls:** hedge ratio, intercept and ARMA/GARCH params fit on data ≤ {cfg.data.formation_end}
only, then frozen. Walk-forward with expanding train window; train/val decisions are purged
{r.horizon} days before the next block.

| Fold | Train | Val | Test |
|---|---|---|---|
{fold_lines}

## Files

`summary.csv` · `metrics_per_seed.csv` · `predictions.csv.gz` (every OOS probability) ·
`trades.csv.gz` · `equity.csv` · `pairs.csv` · `fits.csv` · `models/*.pt` (last fold, reload
with `RegimeGate.load`).
"""
    (out_dir / "report.md").write_text(text)
