"""Replay buffer for strategy embeddings and evaluation metadata."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch
from torch import Tensor

from strategy_games.strategies.embeddings import pairwise_diversity


@dataclass(frozen=True)
class StrategyRecord:
    """A strategy embedding plus scalar evaluation metadata."""

    embedding: Tensor
    episode_return: float
    robustness_score: float
    exploitability_proxy: float
    iteration: int
    label: str | None = None
    average_case_value: float | None = None
    worst_case_value: float | None = None
    goal_rate: float = 0.0
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, object] = field(default_factory=dict)

    def score(self) -> float:
        """Selection score used for positive sampling."""

        average = self.average_case_value if self.average_case_value is not None else self.episode_return
        return float(average + self.episode_return + self.goal_rate - 0.5 * self.exploitability_proxy + 0.5 * self.robustness_score)


class StrategyBuffer:
    """Fixed-capacity store of candidate strategy embeddings and metrics."""

    def __init__(self, capacity: int = 10_000) -> None:
        if capacity < 1:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self._records: list[StrategyRecord] = []

    def __len__(self) -> int:
        return len(self._records)

    def add(self, record: StrategyRecord) -> None:
        """Add a record, evicting the oldest item if capacity is exceeded."""

        if record.embedding.ndim != 1:
            raise ValueError("record.embedding must be 1D")
        if not torch.isfinite(record.embedding).all():
            raise ValueError("record.embedding contains non-finite values")
        self._records.append(record)
        if len(self._records) > self.capacity:
            self._records.pop(0)

    def extend(self, records: list[StrategyRecord]) -> None:
        """Add multiple records."""

        for record in records:
            self.add(record)

    def records(self) -> list[StrategyRecord]:
        """Return a shallow copy of all records."""

        return list(self._records)

    def topk(self, k: int) -> list[StrategyRecord]:
        """Return the top ``k`` records by buffer score."""

        if k < 1:
            raise ValueError("k must be positive")
        return sorted(self._records, key=lambda r: r.score(), reverse=True)[:k]

    def sample_positive(
        self,
        batch_size: int,
        quantile: float = 0.5,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        """Sample embeddings from the top-scoring quantile of the buffer."""

        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if not 0 < quantile <= 1:
            raise ValueError("quantile must be in (0, 1]")
        if not self._records:
            raise ValueError("Cannot sample from an empty StrategyBuffer")

        sorted_records = sorted(self._records, key=lambda r: r.score(), reverse=True)
        cutoff = max(1, int(len(sorted_records) * quantile))
        candidates = sorted_records[:cutoff]
        indices = torch.randint(0, len(candidates), (batch_size,), generator=generator)
        return torch.stack([candidates[int(i)].embedding.detach().clone() for i in indices], dim=0)

    def diversity(self) -> float:
        """Mean pairwise distance among stored embeddings."""

        if len(self._records) < 2:
            return 0.0
        return pairwise_diversity(torch.stack([record.embedding for record in self._records], dim=0))
