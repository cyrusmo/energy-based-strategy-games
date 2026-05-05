"""Generate-evaluate-execute-update training loop scaffold."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor
from torch.distributions import Categorical
from torch.nn import functional as F

from strategy_games.envs.gridworld import AttackerDefenderGridworld, GridworldConfig
from strategy_games.evaluation.best_response import GameTheoreticEvaluator
from strategy_games.models.ebm import EnergyMLP, contrastive_divergence_loss
from strategy_games.models.policy import StrategyConditionedPolicy
from strategy_games.models.world_model import LearnedWorldModel
from strategy_games.strategies.buffer import StrategyBuffer, StrategyRecord
from strategy_games.strategies.embeddings import available_heuristic_strategies, named_strategy_embedding, pairwise_diversity
from strategy_games.strategies.sampler import langevin_sample
from strategy_games.training.losses import policy_gradient_surrogate, world_model_loss
from strategy_games.utils.seeding import set_global_seed


@dataclass(frozen=True)
class TrainingConfig:
    """Configuration for the debug training loop."""

    seed: int = 0
    iterations: int = 3
    candidate_strategies: int = 6
    strategy_dim: int = 8
    policy_hidden_dim: int = 64
    ebm_hidden_dim: int = 64
    world_model_hidden_dim: int = 64
    langevin_steps: int = 10
    langevin_step_size: float = 0.02
    episodes_per_opponent: int = 1
    policy_lr: float = 3e-3
    ebm_lr: float = 1e-3
    world_model_lr: float = 1e-3
    gamma: float = 0.99
    entropy_coef: float = 0.01
    grad_clip_norm: float = 1.0
    ebm_batch_size: int = 8
    positive_quantile: float = 0.5
    train_policy: bool = True
    train_ebm: bool = True
    train_world_model: bool = True
    env: GridworldConfig = field(default_factory=GridworldConfig)


@dataclass
class Trajectory:
    """Trajectory data collected from one strategy-conditioned rollout."""

    states: list[Tensor] = field(default_factory=list)
    actions: list[int] = field(default_factory=list)
    log_probs: list[Tensor] = field(default_factory=list)
    entropies: list[Tensor] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    next_states: list[Tensor] = field(default_factory=list)
    dones: list[bool] = field(default_factory=list)
    outcome: str = "running"

    @property
    def total_reward(self) -> float:
        """Total reward for the trajectory."""

        return float(sum(self.rewards))

    @property
    def steps(self) -> int:
        """Number of environment transitions."""

        return len(self.rewards)

    def summary(self) -> dict[str, float | str | int]:
        """Return public scalar rollout metrics."""

        return {
            "episode_return": self.total_reward,
            "outcome": self.outcome,
            "steps": self.steps,
            "goal_rate": float(self.outcome == "goal"),
            "catch_rate": float(self.outcome == "caught"),
            "timeout_rate": float(self.outcome == "timeout"),
            "win_rate": float(self.outcome == "goal"),
        }


def run_training_loop(config: TrainingConfig | None = None) -> dict[str, object]:
    """Run a small end-to-end loop without claiming full RL training.

    Each iteration samples strategies, evaluates them against heuristic defender
    responses, executes the selected strategy, records summaries in a strategy
    buffer, and performs lightweight policy/EBM/world-model updates.
    """

    config = config or TrainingConfig()
    set_global_seed(config.seed)

    env = AttackerDefenderGridworld(config.env)
    ebm = EnergyMLP(config.strategy_dim, hidden_dim=config.ebm_hidden_dim)
    policy = StrategyConditionedPolicy(
        env.state_dim,
        config.strategy_dim,
        env.action_dim,
        hidden_dim=config.policy_hidden_dim,
    )
    world_model = LearnedWorldModel(
        env.state_dim,
        env.action_dim,
        config.strategy_dim,
        hidden_dim=config.world_model_hidden_dim,
    )
    policy_optimizer = torch.optim.Adam(policy.parameters(), lr=config.policy_lr)
    ebm_optimizer = torch.optim.Adam(ebm.parameters(), lr=config.ebm_lr)
    world_model_optimizer = torch.optim.Adam(world_model.parameters(), lr=config.world_model_lr)
    evaluator = GameTheoreticEvaluator(
        env_config=config.env,
        episodes_per_opponent=config.episodes_per_opponent,
        strategy_dim=config.strategy_dim,
        seed=config.seed,
    )
    buffer = StrategyBuffer(capacity=1_000)
    history: list[dict[str, object]] = []

    for iteration in range(config.iterations):
        candidates = generate_candidate_strategies(ebm, config)
        labels = candidate_labels(config.candidate_strategies)
        evaluated = []
        for idx, strategy in enumerate(candidates):
            label = labels[idx] if idx < len(labels) else None
            metrics = evaluator.evaluate_strategy(strategy.detach().cpu(), label=label)
            selection_score = (
                float(metrics["average_case_value"])
                + float(metrics["robustness_score"])
                - float(metrics["exploitability_proxy"])
            )
            evaluated.append((selection_score, strategy.detach().cpu(), label, metrics))

        evaluated.sort(key=lambda item: item[0], reverse=True)
        _, selected_strategy, selected_label, selected_metrics = evaluated[0]
        trajectory = collect_policy_rollout(env, policy, selected_strategy)
        rollout_summary = trajectory.summary()

        record = StrategyRecord(
            embedding=selected_strategy,
            episode_return=float(rollout_summary["episode_return"]),
            robustness_score=float(selected_metrics["robustness_score"]),
            exploitability_proxy=float(selected_metrics["exploitability_proxy"]),
            average_case_value=float(selected_metrics["average_case_value"]),
            worst_case_value=float(selected_metrics["worst_case_value"]),
            iteration=iteration,
            label=selected_label,
            metadata={"rollout": rollout_summary, "best_response_label": selected_metrics["best_response_label"]},
        )
        buffer.add(record)

        update_metrics = update_models(
            config=config,
            trajectory=trajectory,
            selected_strategy=selected_strategy,
            policy=policy,
            policy_optimizer=policy_optimizer,
            ebm=ebm,
            ebm_optimizer=ebm_optimizer,
            world_model=world_model,
            world_model_optimizer=world_model_optimizer,
            buffer=buffer,
        )

        history.append(
            {
                "iteration": iteration,
                "selected_label": selected_label or "latent",
                "selection_metrics": selected_metrics,
                "rollout": rollout_summary,
                "buffer_size": len(buffer),
                "candidate_diversity": pairwise_diversity(candidates.detach().cpu()),
                "updates": update_metrics,
            }
        )

    return {
        "history": history,
        "buffer_size": len(buffer),
        "buffer_diversity": buffer.diversity(),
        "final_selected_label": history[-1]["selected_label"] if history else None,
    }


def generate_candidate_strategies(ebm: EnergyMLP, config: TrainingConfig) -> torch.Tensor:
    """Generate a mix of named heuristics and Langevin EBM samples."""

    heuristic_labels = list(available_heuristic_strategies())
    strategies: list[torch.Tensor] = []
    for label in heuristic_labels[: config.candidate_strategies]:
        strategies.append(named_strategy_embedding(label, config.strategy_dim).vector)

    remaining = config.candidate_strategies - len(strategies)
    if remaining > 0:
        samples = langevin_sample(
            ebm,
            num_samples=remaining,
            strategy_dim=config.strategy_dim,
            steps=config.langevin_steps,
            step_size=config.langevin_step_size,
        )
        strategies.extend([sample for sample in samples])
    return torch.stack(strategies, dim=0)


def candidate_labels(num_candidates: int) -> list[str | None]:
    """Labels aligned with the heuristic-first candidate generator."""

    labels: list[str | None] = list(available_heuristic_strategies())[:num_candidates]
    while len(labels) < num_candidates:
        labels.append(None)
    return labels


def execute_policy_rollout(
    env: AttackerDefenderGridworld,
    policy: StrategyConditionedPolicy,
    strategy: torch.Tensor,
) -> dict[str, float | str | int]:
    """Execute the current policy and return scalar rollout metrics."""

    return collect_policy_rollout(env, policy, strategy).summary()


def collect_policy_rollout(
    env: AttackerDefenderGridworld,
    policy: StrategyConditionedPolicy,
    strategy: Tensor,
) -> Trajectory:
    """Collect one strategy-conditioned rollout with differentiable log-probs."""

    obs = env.reset()
    done = False
    trajectory = Trajectory()
    while not done:
        state = torch.as_tensor(obs, dtype=torch.float32)
        strategy_batch = strategy.detach().float().unsqueeze(0)
        logits = policy(state.unsqueeze(0), strategy_batch)
        distribution = Categorical(logits=logits)
        action_tensor = distribution.sample()
        action = int(action_tensor.item())

        result = env.step(action)
        next_state = torch.as_tensor(result.observation, dtype=torch.float32)

        trajectory.states.append(state.detach())
        trajectory.actions.append(action)
        trajectory.log_probs.append(distribution.log_prob(action_tensor).squeeze(0))
        trajectory.entropies.append(distribution.entropy().squeeze(0))
        trajectory.rewards.append(float(result.reward))
        trajectory.next_states.append(next_state.detach())
        trajectory.dones.append(bool(result.done))

        obs = result.observation
        done = result.done
        trajectory.outcome = str(result.info["outcome"])

    return trajectory


def update_models(
    config: TrainingConfig,
    trajectory: Trajectory,
    selected_strategy: Tensor,
    policy: StrategyConditionedPolicy,
    policy_optimizer: torch.optim.Optimizer,
    ebm: EnergyMLP,
    ebm_optimizer: torch.optim.Optimizer,
    world_model: LearnedWorldModel,
    world_model_optimizer: torch.optim.Optimizer,
    buffer: StrategyBuffer,
) -> dict[str, float]:
    """Apply one lightweight update pass for policy, EBM, and world model."""

    updates: dict[str, float] = {}
    if config.train_policy and trajectory.log_probs:
        updates.update(update_policy(policy, policy_optimizer, trajectory, config))
    if config.train_world_model and trajectory.states:
        updates.update(update_world_model(world_model, world_model_optimizer, trajectory, selected_strategy, config))
    if config.train_ebm and len(buffer) > 0:
        updates.update(update_ebm(ebm, ebm_optimizer, buffer, config))
    return updates


def update_policy(
    policy: StrategyConditionedPolicy,
    optimizer: torch.optim.Optimizer,
    trajectory: Trajectory,
    config: TrainingConfig,
) -> dict[str, float]:
    """Run a minimal REINFORCE-style update for the conditioned policy."""

    log_probs = torch.stack(trajectory.log_probs)
    entropies = torch.stack(trajectory.entropies)
    returns = discounted_returns(trajectory.rewards, gamma=config.gamma)
    advantages = normalize_advantages(returns)
    loss = policy_gradient_surrogate(log_probs, advantages) - config.entropy_coef * entropies.mean()

    optimizer.zero_grad()
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), config.grad_clip_norm)
    optimizer.step()
    return {
        "policy_loss": float(loss.detach().item()),
        "policy_grad_norm": float(grad_norm),
        "policy_entropy": float(entropies.detach().mean().item()),
    }


def update_world_model(
    world_model: LearnedWorldModel,
    optimizer: torch.optim.Optimizer,
    trajectory: Trajectory,
    selected_strategy: Tensor,
    config: TrainingConfig,
) -> dict[str, float]:
    """Fit the placeholder world model on observed one-step transitions."""

    states = torch.stack(trajectory.states)
    next_states = torch.stack(trajectory.next_states)
    actions = torch.tensor(trajectory.actions, dtype=torch.long)
    action_one_hot = F.one_hot(actions, num_classes=world_model.action_dim).float()
    strategy_batch = selected_strategy.detach().float().unsqueeze(0).expand(states.shape[0], -1)
    rewards = torch.tensor(trajectory.rewards, dtype=torch.float32)

    predicted_delta, predicted_reward = world_model(states, action_one_hot, strategy_batch)
    loss = world_model_loss(predicted_delta, next_states - states, predicted_reward, rewards)

    optimizer.zero_grad()
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(world_model.parameters(), config.grad_clip_norm)
    optimizer.step()
    return {
        "world_model_loss": float(loss.detach().item()),
        "world_model_grad_norm": float(grad_norm),
    }


def update_ebm(
    ebm: EnergyMLP,
    optimizer: torch.optim.Optimizer,
    buffer: StrategyBuffer,
    config: TrainingConfig,
) -> dict[str, float]:
    """Train the EBM to assign lower energy to high-scoring buffer strategies."""

    positive = buffer.sample_positive(config.ebm_batch_size, quantile=config.positive_quantile)
    negative = langevin_sample(
        ebm,
        num_samples=config.ebm_batch_size,
        strategy_dim=config.strategy_dim,
        steps=config.langevin_steps,
        step_size=config.langevin_step_size,
    )
    positive_energy = ebm(positive).detach().mean()
    negative_energy = ebm(negative).detach().mean()
    loss = contrastive_divergence_loss(ebm, positive, negative)

    optimizer.zero_grad()
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(ebm.parameters(), config.grad_clip_norm)
    optimizer.step()
    return {
        "ebm_loss": float(loss.detach().item()),
        "ebm_grad_norm": float(grad_norm),
        "positive_energy": float(positive_energy.item()),
        "negative_energy": float(negative_energy.item()),
    }


def discounted_returns(rewards: list[float], gamma: float) -> Tensor:
    """Compute reward-to-go returns."""

    if not 0 <= gamma <= 1:
        raise ValueError("gamma must be in [0, 1]")
    returns: list[float] = []
    running = 0.0
    for reward in reversed(rewards):
        running = float(reward) + gamma * running
        returns.append(running)
    returns.reverse()
    return torch.tensor(returns, dtype=torch.float32)


def normalize_advantages(advantages: Tensor) -> Tensor:
    """Normalize advantages when possible while preserving one-step episodes."""

    if advantages.numel() < 2:
        return advantages
    std = advantages.std(unbiased=False)
    if float(std.item()) < 1e-8:
        return advantages - advantages.mean()
    return (advantages - advantages.mean()) / (std + 1e-8)
