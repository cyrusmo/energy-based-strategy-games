"""Tiny PPO-lite pursuer baseline for the custom multi-evader pursuit game."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.distributions import Categorical
from torch.nn import functional as F

from strategy_games.envs.multi_evader_pursuit import MultiEvaderPursuitConfig, MultiEvaderPursuitEnv
from strategy_games.envs.pursuit_actions import ACTIONS
from strategy_games.models.pursuit_observation import (
    OBSERVATION_SCHEMA,
    PursuitObservationSpec,
    encode_pursuer_observation,
    validate_pursuit_ppo_env,
)
from strategy_games.policies.pursuit_targets import PursuitActorCritic
from strategy_games.policies.scripted_pursuit import scripted_pursuit_actions
from strategy_games.rollouts import multi_evader_config_from_mapping
from strategy_games.utils.config import load_config
from strategy_games.utils.device import resolve_device
from strategy_games.utils.seeding import set_global_seed


@dataclass(frozen=True)
class PursuitPPOConfig:
    """Config for the narrow PPO pursuer smoke baseline."""

    seed: int = 23
    training_run_id: str = "ppo_pursuer_smoke"
    policy_id: str = "ppo_pursuer_v1"
    max_pursuers: int = 1
    max_evaders: int = 2
    num_updates: int = 3
    rollout_steps: int = 48
    update_epochs: int = 2
    minibatch_size: int = 32
    hidden_dim: int = 64
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    grad_clip_norm: float = 0.5
    evader_policy: str = "evader_feint"
    feint_steps: int = 3
    eval_seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    output_dir: Path = Path("outputs/public/pursuit_models/ppo_pursuer")
    checkpoint_path: Path = Path("outputs/private/checkpoints/ppo_pursuer.pt")
    save_checkpoint: bool = True
    device: str = "auto"
    env: MultiEvaderPursuitConfig = field(default_factory=MultiEvaderPursuitConfig)


@dataclass
class PursuitRolloutBatch:
    """On-policy pursuit rollout data."""

    observations: Tensor
    actions: Tensor
    old_log_probs: Tensor
    values: Tensor
    rewards: Tensor
    dones: Tensor
    episode_returns: list[float]
    episode_summaries: list[dict[str, float]]
    running_episode_return: float


def train_ppo_pursuer(config: PursuitPPOConfig | None = None) -> dict[str, Any]:
    """Train a tiny PPO pursuer and write public/private artifacts."""

    config = config or PursuitPPOConfig()
    spec = PursuitObservationSpec(role="pursuer", max_pursuers=config.max_pursuers, max_evaders=config.max_evaders)
    _validate_pursuit_ppo_config(config, spec)
    set_global_seed(config.seed)
    device = resolve_device(config.device, job="ppo_update")

    env = MultiEvaderPursuitEnv(config.env)
    validate_pursuit_ppo_env(env, spec)
    model = PursuitActorCritic(obs_dim=spec.obs_dim, action_dim=len(ACTIONS), hidden_dim=config.hidden_dim).to(
        device=device, dtype=torch.float32
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    rng = np.random.default_rng(config.seed)
    env.reset()
    running_episode_return = 0.0
    train_episode_summaries: list[dict[str, float]] = []
    loss_stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
    update_history: list[dict[str, float | int]] = []

    for update_idx in range(config.num_updates):
        batch = collect_pursuit_ppo_rollout(
            env=env,
            model=model,
            spec=spec,
            rng=rng,
            steps=config.rollout_steps,
            evader_policy=config.evader_policy,
            feint_steps=config.feint_steps,
            running_episode_return=running_episode_return,
        )
        running_episode_return = batch.running_episode_return
        train_episode_summaries.extend(batch.episode_summaries)
        last_value = bootstrap_pursuit_value(model, env, spec)
        returns, advantages = generalized_advantage_estimate(
            rewards=batch.rewards,
            dones=batch.dones,
            values=batch.values,
            last_value=last_value,
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
        )
        loss_stats = update_pursuit_ppo_policy(model, optimizer, batch, returns, advantages, config)
        update_history.append(
            {
                "update": int(update_idx + 1),
                "policy_loss": float(loss_stats["policy_loss"]),
                "value_loss": float(loss_stats["value_loss"]),
                "entropy": float(loss_stats["entropy"]),
                "train_mean_pursuer_return": float(np.mean(batch.episode_returns)) if batch.episode_returns else 0.0,
            }
        )

    train_metrics = summarize_pursuit_episodes(train_episode_summaries)
    train_metrics.update(loss_stats)
    train_metrics["updates"] = int(config.num_updates)
    train_metrics["rollout_steps"] = int(config.rollout_steps)
    eval_metrics = evaluate_ppo_pursuer(model, config, spec)
    checkpoint_written = False
    if config.save_checkpoint:
        save_pursuit_ppo_checkpoint(model, config, spec, config.checkpoint_path)
        checkpoint_written = True

    artifact = {
        "training_run_id": config.training_run_id,
        "policy_id": config.policy_id,
        "training_scope": training_scope(config),
        "observation_schema": spec.to_dict(),
        "action_space": list(ACTIONS),
        "eval_seeds": list(config.eval_seeds),
        "checkpoint_written": checkpoint_written,
        "device": str(device),
        "train_metrics": train_metrics,
        "eval_metrics": eval_metrics,
        "update_history": update_history,
    }
    write_pursuit_ppo_artifacts(config, artifact)
    return artifact


def collect_pursuit_ppo_rollout(
    env: MultiEvaderPursuitEnv,
    model: PursuitActorCritic,
    spec: PursuitObservationSpec,
    rng: np.random.Generator,
    steps: int,
    evader_policy: str,
    feint_steps: int,
    running_episode_return: float = 0.0,
) -> PursuitRolloutBatch:
    """Collect a pursuit PPO rollout with learned pursuer and scripted evaders."""

    observations: list[Tensor] = []
    actions: list[Tensor] = []
    old_log_probs: list[Tensor] = []
    values: list[Tensor] = []
    rewards: list[float] = []
    dones: list[bool] = []
    episode_returns: list[float] = []
    episode_summaries: list[dict[str, float]] = []
    device = next(model.parameters()).device

    for _ in range(steps):
        obs = torch.as_tensor(encode_pursuer_observation(env, "pursuer_0", spec), dtype=torch.float32, device=device)
        with torch.no_grad():
            logits, value = model(obs.unsqueeze(0))
            distribution = Categorical(logits=logits)
            action = distribution.sample()
            log_prob = distribution.log_prob(action).squeeze(0)

        evader_actions = scripted_pursuit_actions(
            env=env,
            pursuer_policy="pursuer_greedy_nearest",
            evader_policy=evader_policy,
            rng=rng,
            step_index=env.steps,
            feint_steps=feint_steps,
        )
        env_actions = dict(evader_actions)
        env_actions["pursuer_0"] = ACTIONS[int(action.item())]
        result = env.step(env_actions)
        reward = float(result.rewards["pursuer_0"])
        running_episode_return += reward

        observations.append(obs)
        actions.append(action.squeeze(0))
        old_log_probs.append(log_prob)
        values.append(value.squeeze(0))
        rewards.append(reward)
        dones.append(bool(result.done))

        if result.done:
            episode_returns.append(float(running_episode_return))
            episode_summaries.append(_episode_summary(env))
            env.reset()
            running_episode_return = 0.0

    return PursuitRolloutBatch(
        observations=torch.stack(observations),
        actions=torch.stack(actions).long(),
        old_log_probs=torch.stack(old_log_probs).detach(),
        values=torch.stack(values).detach(),
        rewards=torch.tensor(rewards, dtype=torch.float32, device=device),
        dones=torch.tensor(dones, dtype=torch.float32, device=device),
        episode_returns=episode_returns,
        episode_summaries=episode_summaries,
        running_episode_return=float(running_episode_return),
    )


def bootstrap_pursuit_value(model: PursuitActorCritic, env: MultiEvaderPursuitEnv, spec: PursuitObservationSpec) -> Tensor:
    """Bootstrap current state value unless the environment is terminated."""

    if env.done:
        return torch.zeros((), dtype=torch.float32, device=next(model.parameters()).device)
    obs = torch.as_tensor(
        encode_pursuer_observation(env, "pursuer_0", spec),
        dtype=torch.float32,
        device=next(model.parameters()).device,
    ).unsqueeze(0)
    with torch.no_grad():
        _, value = model(obs)
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
    gae = torch.zeros((), dtype=torch.float32, device=rewards.device)
    for step in reversed(range(rewards.shape[0])):
        next_value = last_value if step == rewards.shape[0] - 1 else values[step + 1]
        next_non_terminal = 1.0 - dones[step]
        delta = rewards[step] + gamma * next_value * next_non_terminal - values[step]
        gae = delta + gamma * gae_lambda * next_non_terminal * gae
        advantages[step] = gae
    returns = advantages + values
    return returns.detach(), normalize_advantages(advantages.detach())


def normalize_advantages(advantages: Tensor) -> Tensor:
    """Normalize advantages while preserving tiny batches."""

    if advantages.numel() < 2:
        return advantages
    std = advantages.std(unbiased=False)
    if float(std.item()) < 1e-8:
        return advantages - advantages.mean()
    return (advantages - advantages.mean()) / (std + 1e-8)


def pursuit_ppo_loss(
    model: PursuitActorCritic,
    observations: Tensor,
    actions: Tensor,
    old_log_probs: Tensor,
    returns: Tensor,
    advantages: Tensor,
    config: PursuitPPOConfig,
) -> tuple[Tensor, dict[str, float]]:
    """Compute clipped PPO loss for pursuit observations."""

    logits, values = model(observations)
    distribution = Categorical(logits=logits)
    new_log_probs = distribution.log_prob(actions)
    entropy = distribution.entropy().mean()
    ratio = torch.exp(new_log_probs - old_log_probs)
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1.0 - config.clip_range, 1.0 + config.clip_range) * advantages
    policy_loss = -torch.min(unclipped, clipped).mean()
    value_loss = F.mse_loss(values, returns)
    loss = policy_loss + config.value_coef * value_loss - config.entropy_coef * entropy
    return loss, {
        "policy_loss": float(policy_loss.detach().item()),
        "value_loss": float(value_loss.detach().item()),
        "entropy": float(entropy.detach().item()),
    }


def update_pursuit_ppo_policy(
    model: PursuitActorCritic,
    optimizer: torch.optim.Optimizer,
    batch: PursuitRolloutBatch,
    returns: Tensor,
    advantages: Tensor,
    config: PursuitPPOConfig,
) -> dict[str, float]:
    """Run PPO minibatch updates for the pursuit actor-critic."""

    batch_size = batch.observations.shape[0]
    minibatch_size = min(config.minibatch_size, batch_size)
    stats: list[dict[str, float]] = []
    for _ in range(config.update_epochs):
        for indices in torch.randperm(batch_size, device=batch.observations.device).split(minibatch_size):
            loss, loss_stats = pursuit_ppo_loss(
                model=model,
                observations=batch.observations[indices],
                actions=batch.actions[indices],
                old_log_probs=batch.old_log_probs[indices],
                returns=returns[indices],
                advantages=advantages[indices],
                config=config,
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
            optimizer.step()
            stats.append(loss_stats)
    return {
        "policy_loss": _mean_dict(stats, "policy_loss"),
        "value_loss": _mean_dict(stats, "value_loss"),
        "entropy": _mean_dict(stats, "entropy"),
    }


@torch.no_grad()
def evaluate_ppo_pursuer(
    model: PursuitActorCritic,
    config: PursuitPPOConfig,
    spec: PursuitObservationSpec,
) -> dict[str, float]:
    """Evaluate PPO pursuer deterministically over fixed seeds."""

    summaries: list[dict[str, float]] = []
    device = next(model.parameters()).device
    for seed in config.eval_seeds:
        env = MultiEvaderPursuitEnv(config.env)
        validate_pursuit_ppo_env(env, spec)
        env.reset()
        rng = np.random.default_rng(seed)
        while not env.done:
            obs = torch.as_tensor(
                encode_pursuer_observation(env, "pursuer_0", spec), dtype=torch.float32, device=device
            )
            action = ACTIONS[model.act(obs, deterministic=True)]
            scripted = scripted_pursuit_actions(
                env=env,
                pursuer_policy="pursuer_greedy_nearest",
                evader_policy=config.evader_policy,
                rng=rng,
                step_index=env.steps,
                feint_steps=config.feint_steps,
            )
            env_actions = dict(scripted)
            env_actions["pursuer_0"] = action
            env.step(env_actions)
        summaries.append(_episode_summary(env))
    return summarize_pursuit_episodes(summaries)


def summarize_pursuit_episodes(summaries: Sequence[Mapping[str, float]]) -> dict[str, float]:
    """Aggregate pursuit episode summaries."""

    if not summaries:
        return {
            "capture_rate": 0.0,
            "survival_rate": 0.0,
            "mean_pursuer_return": 0.0,
            "mean_evader_return": 0.0,
            "average_steps": 0.0,
            "episodes": 0.0,
        }
    return {
        "capture_rate": _mean(summaries, "capture_rate"),
        "survival_rate": _mean(summaries, "survival_rate"),
        "mean_pursuer_return": _mean(summaries, "mean_pursuer_return"),
        "mean_evader_return": _mean(summaries, "mean_evader_return"),
        "average_steps": _mean(summaries, "average_steps"),
        "episodes": float(len(summaries)),
    }


def pursuit_ppo_config_from_mapping(raw: Mapping[str, Any]) -> PursuitPPOConfig:
    """Build a PPO pursuit config from a YAML-style mapping."""

    ppo_raw = raw.get("ppo", {})
    env_raw = raw.get("env", {})
    output_raw = raw.get("output", {})
    if not isinstance(ppo_raw, Mapping):
        ppo_raw = {}
    if not isinstance(env_raw, Mapping):
        env_raw = {}
    if not isinstance(output_raw, Mapping):
        output_raw = {}
    return PursuitPPOConfig(
        seed=int(raw.get("seed", 23)),
        training_run_id=str(raw.get("training_run_id", "ppo_pursuer_smoke")),
        policy_id=str(raw.get("policy_id", "ppo_pursuer_v1")),
        max_pursuers=int(raw.get("max_pursuers", 1)),
        max_evaders=int(raw.get("max_evaders", 2)),
        num_updates=int(ppo_raw.get("num_updates", 3)),
        rollout_steps=int(ppo_raw.get("rollout_steps", 48)),
        update_epochs=int(ppo_raw.get("update_epochs", 2)),
        minibatch_size=int(ppo_raw.get("minibatch_size", 32)),
        hidden_dim=int(ppo_raw.get("hidden_dim", 64)),
        learning_rate=float(ppo_raw.get("learning_rate", 3e-4)),
        gamma=float(ppo_raw.get("gamma", 0.99)),
        gae_lambda=float(ppo_raw.get("gae_lambda", 0.95)),
        clip_range=float(ppo_raw.get("clip_range", 0.2)),
        value_coef=float(ppo_raw.get("value_coef", 0.5)),
        entropy_coef=float(ppo_raw.get("entropy_coef", 0.01)),
        grad_clip_norm=float(ppo_raw.get("grad_clip_norm", 0.5)),
        evader_policy=str(raw.get("evader_policy", "evader_feint")),
        feint_steps=int(raw.get("feint_steps", 3)),
        eval_seeds=tuple(int(seed) for seed in raw.get("eval_seeds", [0, 1, 2, 3, 4])),
        output_dir=Path(str(output_raw.get("public_dir", "outputs/public/pursuit_models/ppo_pursuer"))),
        checkpoint_path=Path(str(output_raw.get("checkpoint_path", "outputs/private/checkpoints/ppo_pursuer.pt"))),
        save_checkpoint=bool(output_raw.get("save_checkpoint", True)),
        device=str(raw.get("device", ppo_raw.get("device", "auto"))),
        env=multi_evader_config_from_mapping(env_raw),
    )


def train_ppo_pursuer_from_config(path: str | Path) -> dict[str, Any]:
    """Load config, train PPO pursuer, and write artifacts."""

    return train_ppo_pursuer(pursuit_ppo_config_from_mapping(load_config(path)))


def save_pursuit_ppo_checkpoint(
    model: PursuitActorCritic,
    config: PursuitPPOConfig,
    spec: PursuitObservationSpec,
    path: str | Path,
) -> Path:
    """Save a private checkpoint with strict metadata."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "observation_schema": OBSERVATION_SCHEMA,
        "action_space": list(ACTIONS),
        "max_pursuers": spec.max_pursuers,
        "max_evaders": spec.max_evaders,
        "obs_dim": spec.obs_dim,
        "hidden_dim": config.hidden_dim,
        "policy_id": config.policy_id,
        "training_run_id": config.training_run_id,
        "training_scope": training_scope(config),
        "created_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
    }
    torch.save(checkpoint, output_path)
    return output_path


def write_pursuit_ppo_artifacts(config: PursuitPPOConfig, metrics: Mapping[str, Any]) -> None:
    """Write public config and metrics artifacts."""

    config.output_dir.mkdir(parents=True, exist_ok=True)
    public_config = {
        "training_run_id": config.training_run_id,
        "policy_id": config.policy_id,
        "training_scope": training_scope(config),
        "observation_schema": metrics["observation_schema"],
        "action_space": list(ACTIONS),
        "eval_seeds": list(config.eval_seeds),
        "ppo": {
            "num_updates": config.num_updates,
            "rollout_steps": config.rollout_steps,
            "update_epochs": config.update_epochs,
            "minibatch_size": config.minibatch_size,
            "hidden_dim": config.hidden_dim,
            "learning_rate": config.learning_rate,
            "gamma": config.gamma,
            "gae_lambda": config.gae_lambda,
            "clip_range": config.clip_range,
            "value_coef": config.value_coef,
            "entropy_coef": config.entropy_coef,
            "grad_clip_norm": config.grad_clip_norm,
        },
        "env": {
            "grid_size": list(config.env.grid_size),
            "num_evaders": config.env.num_evaders,
            "num_pursuers": config.env.num_pursuers,
            "max_steps": config.env.max_steps,
            "catch_radius": config.env.catch_radius,
        },
    }
    _write_json(public_config, config.output_dir / "config.json")
    _write_json(dict(metrics), config.output_dir / "metrics.json")


def training_scope(config: PursuitPPOConfig) -> dict[str, object]:
    """Return training scope metadata for public artifacts."""

    return {
        "controlled_agent_id": "pursuer_0",
        "trained_role": "pursuer",
        "opponent_policy": "scripted",
        "scripted_evader_policy": config.evader_policy,
        "self_play": False,
    }


def _validate_pursuit_ppo_config(config: PursuitPPOConfig, spec: PursuitObservationSpec) -> None:
    if config.env.num_pursuers != 1:
        raise ValueError("PPO pursuer path requires num_pursuers == 1")
    if config.env.num_pursuers > config.max_pursuers:
        raise ValueError("env num_pursuers exceeds max_pursuers")
    if config.env.num_evaders > config.max_evaders:
        raise ValueError("env num_evaders exceeds max_evaders")
    if spec.obs_dim < 1:
        raise ValueError("observation dimension must be positive")
    if config.num_updates < 1 or config.rollout_steps < 1:
        raise ValueError("num_updates and rollout_steps must be positive")
    if config.update_epochs < 1 or config.minibatch_size < 1:
        raise ValueError("update_epochs and minibatch_size must be positive")
    if not 0.0 <= config.gamma <= 1.0:
        raise ValueError("gamma must be in [0, 1]")
    if not 0.0 <= config.gae_lambda <= 1.0:
        raise ValueError("gae_lambda must be in [0, 1]")
    if not config.eval_seeds:
        raise ValueError("eval_seeds must be non-empty")


def _episode_summary(env: MultiEvaderPursuitEnv) -> dict[str, float]:
    captured = sum(1 for status in env.per_evader_status.values() if status == "captured")
    survived = sum(1 for status in env.per_evader_status.values() if status == "survived")
    pursuer_returns = [env.per_agent_returns[pursuer_id] for pursuer_id in env.pursuer_ids]
    evader_returns = [env.per_agent_returns[evader_id] for evader_id in env.evader_ids]
    return {
        "capture_rate": float(captured / env.config.num_evaders),
        "survival_rate": float(survived / env.config.num_evaders),
        "mean_pursuer_return": float(np.mean(pursuer_returns)),
        "mean_evader_return": float(np.mean(evader_returns)),
        "average_steps": float(env.steps),
    }


def _mean(items: Sequence[Mapping[str, float]], key: str) -> float:
    values = [float(item[key]) for item in items if key in item]
    return float(np.mean(values)) if values else 0.0


def _mean_dict(items: Sequence[Mapping[str, float]], key: str) -> float:
    values = [float(item[key]) for item in items if key in item]
    return float(np.mean(values)) if values else 0.0


def _write_json(payload: Mapping[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return path
