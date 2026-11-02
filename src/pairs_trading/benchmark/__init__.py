"""Benchmark pipeline: compare every regime gate on identical data, labels, splits and trade rules.

uv run python -m pairs_trading.benchmark --config configs/benchmark.toml
"""

from pairs_trading.benchmark.config import BenchmarkConfig, load_config
from pairs_trading.benchmark.run import BenchmarkResult, run_benchmark

__all__ = ["BenchmarkConfig", "BenchmarkResult", "load_config", "run_benchmark"]
