"""CLI: ``uv run python -m pairs_trading.benchmark --config configs/benchmark.toml``.

Examples
--------
  # full run on real data (downloads via yfinance once, then cached in data/cache/)
  uv run python -m pairs_trading.benchmark --config configs/benchmark.toml

  # just two models, three seeds
  uv run python -m pairs_trading.benchmark --config configs/benchmark.toml --models tft autoformer --seeds 0 1 2

  # offline smoke test on synthetic regime-switching pairs (~1-2 min on CPU)
  uv run python -m pairs_trading.benchmark --config configs/smoke.toml
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
from pathlib import Path

from pairs_trading.benchmark.config import load_config
from pairs_trading.benchmark.run import run_benchmark
from pairs_trading.gate.registry import available_models


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        prog="python -m pairs_trading.benchmark",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--config", type=Path, default=Path("configs/benchmark.toml"))
    ap.add_argument(
        "--models", nargs="+", choices=available_models(), help="override [models]"
    )
    ap.add_argument("--seeds", nargs="+", type=int, help="override seeds")
    ap.add_argument("--epochs", type=int, help="override gate.epochs")
    ap.add_argument(
        "--source",
        choices=["yfinance", "csv", "synthetic"],
        help="override data.source",
    )
    ap.add_argument("--out", type=Path, help="override output_dir")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    for noisy in ("yfinance", "urllib3", "matplotlib", "peewee"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    cfg = load_config(args.config)
    if args.models:
        cfg = cfg.override(models=tuple(args.models))
    if args.seeds:
        cfg = cfg.override(seeds=tuple(args.seeds))
    if args.epochs:
        cfg = cfg.override(gate=dataclasses.replace(cfg.gate, epochs=args.epochs))
    if args.source:
        cfg = cfg.override(data=dataclasses.replace(cfg.data, source=args.source))

    result = run_benchmark(cfg, out_root=args.out, config_path=args.config)
    cols = [
        c
        for c in (
            "auc",
            "take_rate",
            "precision",
            "sharpe",
            "ann_return",
            "max_drawdown",
            "n_trades",
        )
        if c in result.summary.columns
    ]
    print("\n" + result.summary[cols].round(3).to_string())
    print(f"\nreport: {result.out_dir / 'report.md'}")


if __name__ == "__main__":
    main()
