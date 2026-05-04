"""Loss helpers for early experiments."""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F


def policy_gradient_surrogate(log_probs: Tensor, advantages: Tensor) -> Tensor:
    """Minimal policy-gradient surrogate loss."""

    return -(log_probs * advantages.detach()).mean()


def world_model_loss(predicted_delta: Tensor, target_delta: Tensor, predicted_reward: Tensor, target_reward: Tensor) -> Tensor:
    """Basic supervised loss for transition and reward prediction."""

    return F.mse_loss(predicted_delta, target_delta) + F.mse_loss(predicted_reward, target_reward)


def no_op_loss(device: torch.device | str | None = None) -> Tensor:
    """Return a scalar zero loss for TODO hooks."""

    return torch.zeros((), device=device)
