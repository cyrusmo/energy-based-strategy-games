"""Shared benchmark result types."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class BenchmarkResult:
    """One benchmark rollout result using a comparable public schema."""

    env_id: str
    baseline: str
    seed: int
    episode_return: float
    win_rate: float
    goal_rate: float
    catch_rate: float
    timeout_rate: float
    survival_or_capture_rate: float
    steps: int
    strategy_label: str
    wall_clock_seconds: float
    average_case_value: float | None = None
    worst_case_value: float | None = None
    exploitability_proxy: float | None = None
    strategy_diversity: float | None = None

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe row dictionary."""

        return asdict(self)


class BenchmarkDependencyError(RuntimeError):
    """Raised when an optional benchmark dependency is not installed."""
