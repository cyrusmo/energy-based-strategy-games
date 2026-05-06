"""Benchmark adapters and runners."""

from strategy_games.benchmarks.registry import available_benchmarks, make_benchmark_adapter
from strategy_games.benchmarks.runner import run_benchmark_from_config, run_benchmark_suite
from strategy_games.benchmarks.types import BenchmarkDependencyError, BenchmarkResult

__all__ = [
    "BenchmarkDependencyError",
    "BenchmarkResult",
    "available_benchmarks",
    "make_benchmark_adapter",
    "run_benchmark_from_config",
    "run_benchmark_suite",
]
