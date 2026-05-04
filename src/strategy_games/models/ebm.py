"""Energy-based model over latent strategy embeddings."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class EnergyMLP(nn.Module):
    """A small MLP that maps strategy embeddings to scalar energies."""

    def __init__(self, strategy_dim: int, hidden_dim: int = 64, num_layers: int = 2) -> None:
        super().__init__()
        if strategy_dim < 1:
            raise ValueError("strategy_dim must be positive")
        if hidden_dim < 1:
            raise ValueError("hidden_dim must be positive")
        if num_layers < 1:
            raise ValueError("num_layers must be positive")

        layers: list[nn.Module] = []
        input_dim = strategy_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.SiLU())
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, strategy: Tensor) -> Tensor:
        """Return one scalar energy per strategy embedding."""

        if strategy.ndim != 2:
            raise ValueError(f"strategy must have shape [batch, dim], got {tuple(strategy.shape)}")
        return self.net(strategy).squeeze(-1)


def contrastive_divergence_loss(
    model: EnergyMLP,
    positive_strategies: Tensor,
    negative_strategies: Tensor,
    l2_energy: float = 1e-4,
) -> Tensor:
    """Simple contrastive divergence objective for EBM training.

    Positive strategies should receive lower energy than negative samples. This
    helper is intentionally minimal and will likely be replaced by a more
    careful EBM objective once experiments mature.
    """

    positive_energy = model(positive_strategies)
    negative_energy = model(negative_strategies)
    cd = positive_energy.mean() - negative_energy.mean()
    regularizer = l2_energy * (positive_energy.square().mean() + negative_energy.square().mean())
    return cd + regularizer


def score_matching_stub(model: EnergyMLP, strategies: Tensor) -> Tensor:
    """Placeholder for future denoising score matching experiments."""

    energy = model(strategies)
    return F.mse_loss(energy, torch.zeros_like(energy))
