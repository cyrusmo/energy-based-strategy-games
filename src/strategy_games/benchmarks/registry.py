"""Benchmark adapter registry."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from strategy_games.benchmarks.adapters import (
    BenchmarkAdapter,
    CustomGridworldBenchmarkAdapter,
    PettingZooPursuitBenchmarkAdapter,
)


def available_benchmarks() -> tuple[str, ...]:
    """Return supported benchmark environment ids."""

    return ("custom_gridworld_v0", "pettingzoo_pursuit_v4")


def make_benchmark_adapter(env_id: str, config: Mapping[str, Any] | None = None) -> BenchmarkAdapter:
    """Create a benchmark adapter by environment id."""

    if env_id == "custom_gridworld_v0":
        return CustomGridworldBenchmarkAdapter(config)
    if env_id == "pettingzoo_pursuit_v4":
        return PettingZooPursuitBenchmarkAdapter(config)
    raise KeyError(f"Unknown benchmark env_id {env_id!r}; choose from {available_benchmarks()}")
