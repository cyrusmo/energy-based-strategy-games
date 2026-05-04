"""World model interface and lightweight placeholder implementation."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class LearnedWorldModel(nn.Module):
    """Stub transition/reward model for future model-based evaluation.

    The first scaffold uses true environment rollouts for evaluation. This
    module provides a clean interface for later learned rollouts without
    entangling the evaluator with a specific architecture.
    """

    def __init__(self, state_dim: int, action_dim: int, strategy_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        if state_dim < 1 or action_dim < 1 or strategy_dim < 1:
            raise ValueError("state_dim, action_dim, and strategy_dim must be positive")
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.strategy_dim = strategy_dim
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim + strategy_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.delta_head = nn.Linear(hidden_dim, state_dim)
        self.reward_head = nn.Linear(hidden_dim, 1)

    def forward(self, state: Tensor, action_one_hot: Tensor, strategy: Tensor) -> tuple[Tensor, Tensor]:
        """Predict next-state delta and reward."""

        x = torch.cat([state.float(), action_one_hot.float(), strategy.float()], dim=-1)
        hidden = self.net(x)
        delta = self.delta_head(hidden)
        reward = self.reward_head(hidden).squeeze(-1)
        return delta, reward

    def rollout(self, *args: object, **kwargs: object) -> None:
        """Reserved for future model-based imaginary rollouts."""

        del args, kwargs
        raise NotImplementedError("LearnedWorldModel.rollout is a research TODO; use true environment rollouts for now.")
