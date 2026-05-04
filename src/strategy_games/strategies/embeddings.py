"""Utilities for strategy embeddings.

The first few dimensions of named heuristic embeddings are deliberately
interpretable for debugging: directness, evasiveness, aggression, patience, and
feinting. Extra dimensions are left as small deterministic offsets.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor


@dataclass(frozen=True)
class StrategyEmbedding:
    """A latent strategy vector with optional metadata for debugging."""

    vector: Tensor
    label: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.vector.ndim != 1:
            raise ValueError(f"StrategyEmbedding.vector must be 1D, got shape {tuple(self.vector.shape)}")
        if not torch.isfinite(self.vector).all():
            raise ValueError("StrategyEmbedding.vector contains non-finite values")

    @property
    def dim(self) -> int:
        """Embedding dimension."""

        return int(self.vector.shape[0])

    def as_batch(self) -> Tensor:
        """Return the embedding as a batch of size one."""

        return self.vector.unsqueeze(0)


_HEURISTIC_PROTOTYPES: dict[str, list[float]] = {
    "aggressive": [0.40, 0.05, 1.00, 0.10, 0.10],
    "evasive": [0.35, 1.00, 0.05, 0.20, 0.30],
    "feint": [0.55, 0.40, 0.20, 0.15, 1.00],
    "patient": [0.20, 0.45, 0.10, 1.00, 0.20],
    "direct_goal": [1.00, 0.10, 0.20, 0.05, 0.05],
}


def available_heuristic_strategies() -> tuple[str, ...]:
    """Return named strategies available for debugging."""

    return tuple(_HEURISTIC_PROTOTYPES)


def named_strategy_embedding(label: str, dim: int, device: torch.device | str | None = None) -> StrategyEmbedding:
    """Create a deterministic named strategy embedding."""

    if dim < 1:
        raise ValueError("dim must be positive")
    if label not in _HEURISTIC_PROTOTYPES:
        raise KeyError(f"Unknown strategy label {label!r}; choose from {available_heuristic_strategies()}")

    values = torch.zeros(dim, dtype=torch.float32, device=device)
    prototype = torch.tensor(_HEURISTIC_PROTOTYPES[label], dtype=torch.float32, device=device)
    n = min(dim, prototype.numel())
    values[:n] = prototype[:n]
    if dim > prototype.numel():
        values[prototype.numel() :] = torch.linspace(-0.15, 0.15, dim - prototype.numel(), device=device)
    return StrategyEmbedding(values, label=label, metadata={"source": "heuristic"})


def random_strategy_embeddings(
    num_strategies: int,
    dim: int,
    seed: int | None = None,
    device: torch.device | str | None = None,
) -> Tensor:
    """Sample random strategy embeddings from a standard Gaussian."""

    if num_strategies < 1:
        raise ValueError("num_strategies must be positive")
    if dim < 1:
        raise ValueError("dim must be positive")
    generator = None
    if seed is not None:
        generator = torch.Generator(device=device or "cpu")
        generator.manual_seed(seed)
    return torch.randn(num_strategies, dim, generator=generator, device=device)


def pairwise_diversity(strategies: Tensor) -> float:
    """Return mean pairwise Euclidean distance among strategy embeddings."""

    if strategies.ndim != 2:
        raise ValueError(f"strategies must have shape [batch, dim], got {tuple(strategies.shape)}")
    if strategies.shape[0] < 2:
        return 0.0
    distances = torch.pdist(strategies.detach().float(), p=2)
    return float(distances.mean().item())
