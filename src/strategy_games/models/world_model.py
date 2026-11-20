"""World model interface and lightweight placeholder implementation."""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor, nn


class LearnedWorldModel(nn.Module):
    """One-step transition/reward model with open-loop imagined rollouts.

    Evaluation in this repository still uses true environment rollouts.
    ``rollout`` composes one-step predictions; its accuracy against the
    environment has not been measured, and it is not used for best response.
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

    @torch.no_grad()
    def rollout(
        self,
        state: Tensor,
        strategy: Tensor,
        policy: Callable[[Tensor, Tensor], Tensor],
        horizon: int,
    ) -> tuple[Tensor, Tensor]:
        """Imagine ``horizon`` steps from ``state`` under ``policy``.

        ``state`` is ``[batch, state_dim]`` and ``strategy`` is
        ``[batch, strategy_dim]`` or a single vector. ``policy(state, strategy)``
        returns integer actions of shape ``[batch]``. Returns the imagined states
        ``[batch, horizon + 1, state_dim]`` and rewards ``[batch, horizon]``.

        The model has no termination head, so rollouts always run the full
        horizon; callers that need episode returns must truncate themselves.
        """

        if horizon < 1:
            raise ValueError("horizon must be positive")
        if state.ndim != 2 or state.shape[-1] != self.state_dim:
            raise ValueError(f"state must have shape [batch, {self.state_dim}], got {tuple(state.shape)}")
        strategy = strategy.float()
        if strategy.ndim == 1:
            strategy = strategy.unsqueeze(0).expand(state.shape[0], -1)
        current = state.float()
        states, rewards = [current], []
        for _ in range(horizon):
            action = policy(current, strategy).long()
            one_hot = torch.nn.functional.one_hot(action, num_classes=self.action_dim).float()
            delta, reward = self.forward(current, one_hot, strategy)
            current = current + delta
            states.append(current)
            rewards.append(reward)
        return torch.stack(states, dim=1), torch.stack(rewards, dim=1)
