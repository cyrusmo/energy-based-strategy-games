"""Strategy-conditioned and baseline policies."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.distributions import Categorical


class StrategyConditionedPolicy(nn.Module):
    """MLP policy for actions conditioned on state and latent strategy."""

    def __init__(self, state_dim: int, strategy_dim: int, action_dim: int = 5, hidden_dim: int = 64) -> None:
        super().__init__()
        if state_dim < 1:
            raise ValueError("state_dim must be positive")
        if strategy_dim < 1:
            raise ValueError("strategy_dim must be positive")
        if action_dim < 1:
            raise ValueError("action_dim must be positive")
        self.state_dim = state_dim
        self.strategy_dim = strategy_dim
        self.action_dim = action_dim
        self.net = nn.Sequential(
            nn.Linear(state_dim + strategy_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, state: Tensor, strategy: Tensor) -> Tensor:
        """Return action logits for ``state`` and ``strategy`` batches."""

        if state.ndim != 2:
            raise ValueError(f"state must have shape [batch, state_dim], got {tuple(state.shape)}")
        if strategy.ndim != 2:
            raise ValueError(f"strategy must have shape [batch, strategy_dim], got {tuple(strategy.shape)}")
        if state.shape[0] != strategy.shape[0]:
            raise ValueError("state and strategy batch sizes must match")
        x = torch.cat([state.float(), strategy.float()], dim=-1)
        return self.net(x)

    @torch.no_grad()
    def act(self, state: Tensor, strategy: Tensor, deterministic: bool = False) -> int:
        """Return a single action for a single state/strategy pair."""

        if state.ndim == 1:
            state = state.unsqueeze(0)
        if strategy.ndim == 1:
            strategy = strategy.unsqueeze(0)
        logits = self.forward(state, strategy)
        if deterministic:
            return int(torch.argmax(logits, dim=-1).item())
        return int(Categorical(logits=logits).sample().item())


class RandomPolicy:
    """Uniform random policy with the same interface as learned policies."""

    def __init__(self, action_dim: int = 5, seed: int | None = None) -> None:
        if action_dim < 1:
            raise ValueError("action_dim must be positive")
        self.action_dim = action_dim
        self.generator = torch.Generator()
        if seed is not None:
            self.generator.manual_seed(seed)

    def act(self, state: Tensor | None = None, strategy: Tensor | None = None, deterministic: bool = False) -> int:
        """Return a uniformly sampled action."""

        del state, strategy, deterministic
        return int(torch.randint(0, self.action_dim, (1,), generator=self.generator).item())
