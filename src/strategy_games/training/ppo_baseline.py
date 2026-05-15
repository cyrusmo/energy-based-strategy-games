"""PPO-lite and heuristic baselines for the custom gridworld.

This module keeps the first learning baseline intentionally narrow: a single
attacker actor-critic trained against the gridworld's built-in greedy defender.
It is a research baseline for comparison, not the strategy-loop implementation.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.distributions import Categorical
from torch.nn import functional as F

from strategy_games.envs.gridworld import AttackerDefenderGridworld, GridworldConfig, greedy_action_towards
from strategy_games.models.policy import RandomPolicy
from strategy_games.utils.config import load_config
from strategy_games.utils.seeding import set_global_seed


@dataclass(frozen=True)
class PPOConfig:
    """Configuration for the custom-gridworld PPO-lite baseline."""

    seed: int = 11
    total_steps: int = 512
    rollout_steps: int = 128
    update_epochs: int = 4
    minibatch_size: int = 64
    hidden_dim: int = 64
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    grad_clip_norm: float = 0.5
    eval_episodes: int = 5
    env: GridworldConfig = field(default_factory=GridworldConfig)


@dataclass
class PPORolloutBatch:
    """On-policy rollout batch collected from the custom gridworld."""

    states: Tensor
    actions: Tensor
    old_log_probs: Tensor
    values: Tensor
    rewards: Tensor
    dones: Tensor
    last_observation: np.ndarray
    last_done: bool
    episode_returns: list[float]
    episode_outcomes: list[str]
    running_episode_return: float


class ActorCriticPolicy(nn.Module):
    """Small MLP actor-critic for vector gridworld observations."""

    def __init__(self, state_dim: int, action_dim: int = 5, hidden_dim: int = 64) -> None:
        super().__init__()
        if state_dim < 1:
            raise ValueError("state_dim must be positive")
        if action_dim < 1:
            raise ValueError("action_dim must be positive")
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.actor = nn.Linear(hidden_dim, action_dim)
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, state: Tensor) -> tuple[Tensor, Tensor]:
        """Return action logits and scalar state values for a batch."""

        if state.ndim != 2:
            raise ValueError(f"state must have shape [batch, state_dim], got {tuple(state.shape)}")
        features = self.backbone(state.float())
        logits = self.actor(features)
        values = self.critic(features).squeeze(-1)
        return logits, values

    @torch.no_grad()
    def act(self, state: Tensor, deterministic: bool = False) -> int:
        """Return a single action for one state."""

        if state.ndim == 1:
            state = state.unsqueeze(0)
        logits, _ = self.forward(state)
        if deterministic:
            return int(torch.argmax(logits, dim=-1).item())
        return int(Categorical(logits=logits).sample().item())


def run_random_policy_baseline(
    episodes: int = 5,
    seed: int | None = 0,
    env_config: GridworldConfig | None = None,
) -> dict[str, float]:
    """Run a uniform random attacker against a greedy defender."""

    if episodes < 1:
        raise ValueError("episodes must be positive")
    env_config = env_config or GridworldConfig()
    policy = RandomPolicy(action_dim=AttackerDefenderGridworld.action_dim, seed=seed)
    returns: list[float] = []
    outcomes: list[str] = []
    for _ in range(episodes):
        env = AttackerDefenderGridworld(env_config)
        env.reset()
        done = False
        total = 0.0
        info = {"outcome": "running"}
        while not done:
            result = env.step(policy.act())
            total += result.reward
            done = result.done
            info = result.info
        returns.append(total)
        outcomes.append(str(info["outcome"]))
    return _summarize(returns, outcomes)


def run_direct_goal_baseline(episodes: int = 5, env_config: GridworldConfig | None = None) -> dict[str, float]:
    """Run a direct-to-goal attacker against a greedy defender."""

    if episodes < 1:
        raise ValueError("episodes must be positive")
    env_config = env_config or GridworldConfig()
    returns: list[float] = []
    outcomes: list[str] = []
    for _ in range(episodes):
        env = AttackerDefenderGridworld(env_config)
        env.reset()
        done = False
        total = 0.0
        info = {"outcome": "running"}
        while not done:
            action = greedy_action_towards(env.attacker_pos, env.goal_pos)
            result = env.step(action)
            total += result.reward
            done = result.done
            info = result.info
        returns.append(total)
        outcomes.append(str(info["outcome"]))
    return _summarize(returns, outcomes)


def train_ppo_baseline(config: PPOConfig | None = None) -> dict[str, float | int]:
    """Train a PPO-lite actor-critic and return public scalar metrics."""

    config = config or PPOConfig()
    _validate_ppo_config(config)
    set_global_seed(config.seed)

    env = AttackerDefenderGridworld(config.env)
    policy = ActorCriticPolicy(env.state_dim, env.action_dim, hidden_dim=config.hidden_dim)
    optimizer = torch.optim.Adam(policy.parameters(), lr=config.learning_rate)

    observation = env.reset()
    running_episode_return = 0.0
    train_returns: list[float] = []
    train_outcomes: list[str] = []
    last_update = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
    steps_done = 0
    updates = 0

    while steps_done < config.total_steps:
        steps_this_batch = min(config.rollout_steps, config.total_steps - steps_done)
        batch = collect_ppo_rollout(
            env=env,
            policy=policy,
            observation=observation,
            steps=steps_this_batch,
            running_episode_return=running_episode_return,
        )
        observation = batch.last_observation
        running_episode_return = batch.running_episode_return
        train_returns.extend(batch.episode_returns)
        train_outcomes.extend(batch.episode_outcomes)

        last_value = bootstrap_value(policy, batch.last_observation, batch.last_done)
        returns, advantages = generalized_advantage_estimate(
            rewards=batch.rewards,
            dones=batch.dones,
            values=batch.values,
            last_value=last_value,
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
        )
        last_update = update_ppo_policy(policy, optimizer, batch, returns, advantages, config)
        steps_done += steps_this_batch
        updates += 1

    eval_metrics = evaluate_ppo_policy(policy, config.env, episodes=config.eval_episodes)
    return {
        **eval_metrics,
        "policy_loss": float(last_update["policy_loss"]),
        "value_loss": float(last_update["value_loss"]),
        "entropy": float(last_update["entropy"]),
        "episodes": int(config.eval_episodes),
        "train_episodes": int(len(train_returns)),
        "updates": int(updates),
        "total_steps": int(steps_done),
    }


def train_ppo_from_config(path: str | Path = "configs/gridworld_ppo_baseline.yaml") -> dict[str, Any]:
    """Load a PPO baseline YAML config, train, and optionally write metrics."""

    raw = load_config(path)
    config = ppo_config_from_mapping(raw)
    result: dict[str, Any] = dict(train_ppo_baseline(config))

    logging_raw = raw.get("logging", {})
    if not isinstance(logging_raw, Mapping):
        logging_raw = {}
    if bool(logging_raw.get("enabled", True)):
        output_dir = Path(str(logging_raw.get("output_dir", "outputs/public/ppo_baseline")))
        metrics_path = save_ppo_metrics(result, output_dir / "metrics.json")
        result["artifacts"] = {"metrics_json": str(metrics_path)}
    return result


def collect_ppo_rollout(
    env: AttackerDefenderGridworld,
    policy: ActorCriticPolicy,
    observation: np.ndarray,
    steps: int,
    running_episode_return: float = 0.0,
) -> PPORolloutBatch:
    """Collect one on-policy batch from the custom gridworld."""

    states: list[Tensor] = []
    actions: list[Tensor] = []
    old_log_probs: list[Tensor] = []
    values: list[Tensor] = []
    rewards: list[float] = []
    dones: list[bool] = []
    episode_returns: list[float] = []
    episode_outcomes: list[str] = []
    last_done = False

    for _ in range(steps):
        state = torch.as_tensor(observation, dtype=torch.float32)
        with torch.no_grad():
            logits, value = policy(state.unsqueeze(0))
            distribution = Categorical(logits=logits)
            action = distribution.sample()
            log_prob = distribution.log_prob(action).squeeze(0)

        result = env.step(int(action.item()))
        running_episode_return += float(result.reward)

        states.append(state)
        actions.append(action.squeeze(0))
        old_log_probs.append(log_prob)
        values.append(value.squeeze(0))
        rewards.append(float(result.reward))
        dones.append(bool(result.done))

        observation = result.observation
        last_done = bool(result.done)
        if result.done:
            episode_returns.append(float(running_episode_return))
            episode_outcomes.append(str(result.info["outcome"]))
            observation = env.reset()
            running_episode_return = 0.0

    return PPORolloutBatch(
        states=torch.stack(states),
        actions=torch.stack(actions).long(),
        old_log_probs=torch.stack(old_log_probs).detach(),
        values=torch.stack(values).detach(),
        rewards=torch.tensor(rewards, dtype=torch.float32),
        dones=torch.tensor(dones, dtype=torch.float32),
        last_observation=observation,
        last_done=last_done,
        episode_returns=episode_returns,
        episode_outcomes=episode_outcomes,
        running_episode_return=float(running_episode_return),
    )


def bootstrap_value(policy: ActorCriticPolicy, observation: np.ndarray, done: bool) -> Tensor:
    """Return the bootstrap value for an unfinished rollout."""

    if done:
        return torch.zeros((), dtype=torch.float32)
    state = torch.as_tensor(observation, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        _, value = policy(state)
    return value.squeeze(0).detach()


def generalized_advantage_estimate(
    rewards: Tensor,
    dones: Tensor,
    values: Tensor,
    last_value: Tensor,
    gamma: float,
    gae_lambda: float,
) -> tuple[Tensor, Tensor]:
    """Compute GAE returns and normalized advantages."""

    if rewards.shape != dones.shape or rewards.shape != values.shape:
        raise ValueError("rewards, dones, and values must have matching shapes")
    advantages = torch.zeros_like(rewards)
    gae = torch.zeros((), dtype=torch.float32)
    for step in reversed(range(rewards.shape[0])):
        next_value = last_value if step == rewards.shape[0] - 1 else values[step + 1]
        next_non_terminal = 1.0 - dones[step]
        delta = rewards[step] + gamma * next_value * next_non_terminal - values[step]
        gae = delta + gamma * gae_lambda * next_non_terminal * gae
        advantages[step] = gae
    returns = advantages + values
    return returns.detach(), normalize_advantages(advantages.detach())


def normalize_advantages(advantages: Tensor) -> Tensor:
    """Normalize advantages while preserving one-transition batches."""

    if advantages.numel() < 2:
        return advantages
    std = advantages.std(unbiased=False)
    if float(std.item()) < 1e-8:
        return advantages - advantages.mean()
    return (advantages - advantages.mean()) / (std + 1e-8)


def ppo_loss(
    policy: ActorCriticPolicy,
    states: Tensor,
    actions: Tensor,
    old_log_probs: Tensor,
    returns: Tensor,
    advantages: Tensor,
    clip_range: float,
    value_coef: float,
    entropy_coef: float,
) -> tuple[Tensor, dict[str, float]]:
    """Compute clipped PPO loss and public loss stats."""

    logits, values = policy(states)
    distribution = Categorical(logits=logits)
    new_log_probs = distribution.log_prob(actions)
    entropy = distribution.entropy().mean()
    ratio = torch.exp(new_log_probs - old_log_probs)

    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range) * advantages
    policy_loss = -torch.min(unclipped, clipped).mean()
    value_loss = F.mse_loss(values, returns)
    loss = policy_loss + value_coef * value_loss - entropy_coef * entropy
    return loss, {
        "policy_loss": float(policy_loss.detach().item()),
        "value_loss": float(value_loss.detach().item()),
        "entropy": float(entropy.detach().item()),
    }


def update_ppo_policy(
    policy: ActorCriticPolicy,
    optimizer: torch.optim.Optimizer,
    batch: PPORolloutBatch,
    returns: Tensor,
    advantages: Tensor,
    config: PPOConfig,
) -> dict[str, float]:
    """Run PPO minibatch updates over one rollout batch."""

    batch_size = batch.states.shape[0]
    minibatch_size = min(config.minibatch_size, batch_size)
    stats: list[dict[str, float]] = []

    for _ in range(config.update_epochs):
        for indices in torch.randperm(batch_size).split(minibatch_size):
            loss, loss_stats = ppo_loss(
                policy=policy,
                states=batch.states[indices],
                actions=batch.actions[indices],
                old_log_probs=batch.old_log_probs[indices],
                returns=returns[indices],
                advantages=advantages[indices],
                clip_range=config.clip_range,
                value_coef=config.value_coef,
                entropy_coef=config.entropy_coef,
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), config.grad_clip_norm)
            optimizer.step()
            stats.append(loss_stats)

    return {
        "policy_loss": _mean_dict(stats, "policy_loss"),
        "value_loss": _mean_dict(stats, "value_loss"),
        "entropy": _mean_dict(stats, "entropy"),
    }


def evaluate_ppo_policy(
    policy: ActorCriticPolicy,
    env_config: GridworldConfig | None = None,
    episodes: int = 5,
) -> dict[str, float]:
    """Evaluate the learned PPO policy deterministically against the greedy defender."""

    if episodes < 1:
        raise ValueError("episodes must be positive")
    env_config = env_config or GridworldConfig()
    returns: list[float] = []
    outcomes: list[str] = []
    for _ in range(episodes):
        env = AttackerDefenderGridworld(env_config)
        observation = env.reset()
        done = False
        total = 0.0
        info = {"outcome": "running"}
        while not done:
            state = torch.as_tensor(observation, dtype=torch.float32)
            action = policy.act(state, deterministic=True)
            result = env.step(action)
            observation = result.observation
            total += result.reward
            done = result.done
            info = result.info
        returns.append(total)
        outcomes.append(str(info["outcome"]))
    return _summarize(returns, outcomes)


def ppo_config_from_mapping(raw: Mapping[str, Any]) -> PPOConfig:
    """Build a typed PPO config from a YAML-style mapping."""

    env_raw = raw.get("env", {})
    ppo_raw = raw.get("ppo", {})
    if not isinstance(env_raw, Mapping):
        env_raw = {}
    if not isinstance(ppo_raw, Mapping):
        ppo_raw = {}
    return PPOConfig(
        seed=int(raw.get("seed", 11)),
        total_steps=int(ppo_raw.get("total_steps", 512)),
        rollout_steps=int(ppo_raw.get("rollout_steps", 128)),
        update_epochs=int(ppo_raw.get("update_epochs", 4)),
        minibatch_size=int(ppo_raw.get("minibatch_size", 64)),
        hidden_dim=int(ppo_raw.get("hidden_dim", 64)),
        learning_rate=float(ppo_raw.get("learning_rate", 3e-4)),
        gamma=float(ppo_raw.get("gamma", 0.99)),
        gae_lambda=float(ppo_raw.get("gae_lambda", 0.95)),
        clip_range=float(ppo_raw.get("clip_range", 0.2)),
        value_coef=float(ppo_raw.get("value_coef", 0.5)),
        entropy_coef=float(ppo_raw.get("entropy_coef", 0.01)),
        grad_clip_norm=float(ppo_raw.get("grad_clip_norm", 0.5)),
        eval_episodes=int(ppo_raw.get("eval_episodes", 5)),
        env=_gridworld_config_from_mapping(env_raw),
    )


def save_ppo_metrics(metrics: Mapping[str, Any], path: str | Path) -> Path:
    """Write PPO baseline metrics to JSON."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return output_path


def _validate_ppo_config(config: PPOConfig) -> None:
    if config.total_steps < 1:
        raise ValueError("total_steps must be positive")
    if config.rollout_steps < 1:
        raise ValueError("rollout_steps must be positive")
    if config.update_epochs < 1:
        raise ValueError("update_epochs must be positive")
    if config.minibatch_size < 1:
        raise ValueError("minibatch_size must be positive")
    if not 0.0 <= config.gamma <= 1.0:
        raise ValueError("gamma must be in [0, 1]")
    if not 0.0 <= config.gae_lambda <= 1.0:
        raise ValueError("gae_lambda must be in [0, 1]")
    if config.eval_episodes < 1:
        raise ValueError("eval_episodes must be positive")


def _gridworld_config_from_mapping(raw: Mapping[str, Any]) -> GridworldConfig:
    return GridworldConfig(
        grid_size=int(raw.get("grid_size", 10)),
        max_steps=int(raw.get("max_steps", 50)),
        attacker_start=tuple(raw.get("attacker_start", [0, 0])),  # type: ignore[arg-type]
        defender_start=tuple(raw.get("defender_start", [9, 9])),  # type: ignore[arg-type]
        goal_pos=tuple(raw.get("goal_pos", [9, 0])),  # type: ignore[arg-type]
        catch_radius=int(raw.get("catch_radius", 0)),
        step_penalty=float(raw.get("step_penalty", -0.01)),
        goal_reward=float(raw.get("goal_reward", 1.0)),
        catch_reward=float(raw.get("catch_reward", -1.0)),
        timeout_reward=float(raw.get("timeout_reward", -0.2)),
    )


def _summarize(returns: list[float], outcomes: list[str]) -> dict[str, float]:
    return {
        "episode_return": float(np.mean(returns)),
        "win_rate": float(outcomes.count("goal") / len(outcomes)),
        "goal_rate": float(outcomes.count("goal") / len(outcomes)),
        "catch_rate": float(outcomes.count("caught") / len(outcomes)),
        "timeout_rate": float(outcomes.count("timeout") / len(outcomes)),
    }


def _mean_dict(items: list[dict[str, float]], key: str) -> float:
    values = [float(item[key]) for item in items if key in item]
    if not values:
        return 0.0
    return float(np.mean(values))
