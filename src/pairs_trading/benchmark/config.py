"""Benchmark configuration, loaded from a TOML file (see configs/benchmark.toml)."""

from __future__ import annotations

import json
import tomllib
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pairs_trading.gate.config import GateConfig, TradeRules


@dataclass(frozen=True)
class DataConfig:
    source: str = "yfinance"  # "yfinance" | "csv" | "synthetic"
    pairs: Tuple[Tuple[str, str], ...] = ()
    pairs_file: Optional[str] = "data/pairs.json"
    start: str = "2015-01-01"
    end: str = "2024-12-31"
    formation_end: str = "2017-12-31"  # hedge ratio + GARCH fit on <= this date
    cache_dir: str = "data/cache"  # yfinance downloads cached here as CSV
    csv_dir: str = "data/prices"  # source="csv": <csv_dir>/<SYMBOL>.csv (date + close)
    max_coint_pvalue: Optional[float] = None  # e.g. 0.10 to drop non-cointegrated pairs
    synthetic_pairs: int = 6
    synthetic_seed: int = 7

    def resolve_pairs(self) -> List[Tuple[str, str]]:
        if self.pairs:
            return [tuple(p) for p in self.pairs]
        if self.pairs_file:
            spec = json.loads(Path(self.pairs_file).read_text())
            return [(str(a), str(b)) for a, b in spec["pairs"]]
        return []


@dataclass(frozen=True)
class WalkForwardConfig:
    n_folds: int = 4  # OOS period split into n_folds expanding-window test blocks
    val_fraction: float = 0.15  # tail of each training block used for early stopping
    min_train_signals: int = 200


@dataclass(frozen=True)
class BacktestConfig:
    capital_per_pair: float = 10_000.0
    cost_bps: float = 5.0  # per side, on gross notional


@dataclass(frozen=True)
class BenchmarkConfig:
    name: str = "benchmark"
    models: Tuple[str, ...] = ("logistic", "tft", "autoformer", "wtransformer")
    seeds: Tuple[int, ...] = (0,)
    output_dir: str = "results"
    data: DataConfig = field(default_factory=DataConfig)
    rules: TradeRules = field(default_factory=TradeRules)
    gate: GateConfig = field(default_factory=GateConfig)
    walk_forward: WalkForwardConfig = field(default_factory=WalkForwardConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def override(self, **kw: Any) -> "BenchmarkConfig":
        return replace(self, **kw)


def _tuplify(v: Any) -> Any:
    if isinstance(v, list):
        return tuple(_tuplify(x) for x in v)
    return v


def load_config(path: str | Path) -> BenchmarkConfig:
    raw = tomllib.loads(Path(path).read_text())
    sections = {
        "data": DataConfig,
        "rules": TradeRules,
        "gate": GateConfig,
        "walk_forward": WalkForwardConfig,
        "backtest": BacktestConfig,
    }
    kwargs: Dict[str, Any] = {}
    for key, cls in sections.items():
        sec = dict(raw.pop(key, {}))
        if cls is GateConfig:
            sec["model_kwargs"] = sec.pop("model_kwargs", {})
            kwargs[key] = cls(**sec)
        else:
            kwargs[key] = cls(**{k: _tuplify(v) for k, v in sec.items()})
    for k, v in raw.items():
        kwargs[k] = _tuplify(v)
    return BenchmarkConfig(**kwargs)
