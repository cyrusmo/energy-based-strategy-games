"""Sampling utilities for latent strategies."""

from __future__ import annotations

import math
from collections.abc import Callable

import torch
from torch import Tensor, nn


EnergyFunction = Callable[[Tensor], Tensor] | nn.Module


def langevin_sample(
    energy_model: EnergyFunction,
    num_samples: int,
    strategy_dim: int,
    steps: int = 30,
    step_size: float = 1e-2,
    noise_scale: float = 1.0,
    init: Tensor | None = None,
    clamp: tuple[float, float] | None = (-5.0, 5.0),
    device: torch.device | str | None = None,
) -> Tensor:
    """Sample strategy embeddings with unadjusted Langevin dynamics.

    This is a deliberately simple sampler for debugging. It optimizes toward
    lower-energy regions while injecting Gaussian noise at each step.
    """

    if num_samples < 1:
        raise ValueError("num_samples must be positive")
    if strategy_dim < 1:
        raise ValueError("strategy_dim must be positive")
    if steps < 0:
        raise ValueError("steps must be non-negative")
    if step_size <= 0:
        raise ValueError("step_size must be positive")
    if noise_scale < 0:
        raise ValueError("noise_scale must be non-negative")

    if init is None:
        x = torch.randn(num_samples, strategy_dim, device=device)
    else:
        if init.shape != (num_samples, strategy_dim):
            raise ValueError(f"init must have shape {(num_samples, strategy_dim)}, got {tuple(init.shape)}")
        x = init.detach().clone().to(device=device)

    for _ in range(steps):
        x = x.detach().requires_grad_(True)
        energy = energy_model(x)
        if energy.shape != (num_samples,):
            energy = energy.reshape(num_samples)
        grad = torch.autograd.grad(energy.sum(), x, create_graph=False)[0]
        with torch.no_grad():
            noise = torch.randn_like(x) * (noise_scale * math.sqrt(step_size))
            x = x - 0.5 * step_size * grad + noise
            if clamp is not None:
                x = x.clamp(min=clamp[0], max=clamp[1])

    return x.detach()


class GaussianStrategySampler:
    """Simple baseline sampler that ignores an energy model."""

    def __init__(self, strategy_dim: int, scale: float = 1.0) -> None:
        if strategy_dim < 1:
            raise ValueError("strategy_dim must be positive")
        if scale <= 0:
            raise ValueError("scale must be positive")
        self.strategy_dim = strategy_dim
        self.scale = scale

    def sample(self, num_samples: int, device: torch.device | str | None = None) -> Tensor:
        """Return Gaussian strategy embeddings."""

        return self.scale * torch.randn(num_samples, self.strategy_dim, device=device)
