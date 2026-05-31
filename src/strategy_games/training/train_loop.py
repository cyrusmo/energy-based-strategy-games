"""Generate-evaluate-execute-update training loop scaffold."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor
from torch.distributions import Categorical
from torch.nn import functional as F

from strategy_games.envs.gridworld import AttackerDefenderGridworld, GridworldConfig, manhattan
from strategy_games.evaluation.best_response import GameTheoreticEvaluator, attacker_heuristic_action
from strategy_games.experiments.convergence import detect_convergence
from strategy_games.models.ebm import EnergyMLP, contrastive_divergence_loss
from strategy_games.models.policy import StrategyConditionedPolicy
from strategy_games.models.world_model import LearnedWorldModel
from strategy_games.strategies.buffer import StrategyBuffer, StrategyRecord
from strategy_games.strategies.embeddings import available_heuristic_strategies, named_strategy_embedding, pairwise_diversity
from strategy_games.strategies.sampler import GaussianStrategySampler, langevin_sample
from strategy_games.training.losses import policy_gradient_surrogate, world_model_loss
from strategy_games.utils.device import resolve_device
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
    episodes_per_update: int = 1
    policy_lr: float = 3e-3
    ebm_lr: float = 1e-3
    world_model_lr: float = 1e-3
    gamma: float = 0.99
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    grad_clip_norm: float = 1.0
    ebm_batch_size: int = 8
    positive_quantile: float = 0.5
    sampler_type: str = "langevin"
    gaussian_scale: float = 1.0
    langevin_noise_scale: float = 0.25
    max_heuristic_candidates: int = 5
    robustness_aware_selection: bool = True
    average_value_weight: float = 1.0
    robustness_weight: float = 0.5
    exploitability_weight: float = 0.5
    goal_rate_weight: float = 1.0
    use_buffer_positives: bool = True
    train_policy: bool = True
    train_ebm: bool = True
    train_world_model: bool = True
    evaluator_action_source: str = "heuristic"
    behavior_clone_iterations: int = 0
    behavior_clone_coef: float = 0.0
    shaping_coef: float = 0.0
    convergence_metric: str = "goal_rate"
    convergence_target: float = 1.0
    convergence_patience: int = 3
    convergence_window: int = 3
    convergence_min_iter: int = 1
    early_stop_on_convergence: bool = False
    device: str = "auto"
    env: GridworldConfig = field(default_factory=GridworldConfig)


@dataclass
class Trajectory:
    """Trajectory data collected from one strategy-conditioned rollout."""

    states: list[Tensor] = field(default_factory=list)
    actions: list[int] = field(default_factory=list)
    log_probs: list[Tensor] = field(default_factory=list)
    entropies: list[Tensor] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    training_rewards: list[float] = field(default_factory=list)
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

    @property
    def training_return(self) -> float:
        """Total shaped reward used for policy optimization."""

        return float(sum(self.training_rewards or self.rewards))

    def summary(self) -> dict[str, float | str | int]:
        """Return public scalar rollout metrics."""

        return {
            "episode_return": self.total_reward,
            "training_return": self.training_return,
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
    device = resolve_device(config.device, job="langevin_sampling")

    env = AttackerDefenderGridworld(config.env)
    ebm = EnergyMLP(config.strategy_dim, hidden_dim=config.ebm_hidden_dim).to(device=device, dtype=torch.float32)
    policy = StrategyConditionedPolicy(
        env.state_dim,
        config.strategy_dim,
        env.action_dim,
        hidden_dim=config.policy_hidden_dim,
    ).to(device=device, dtype=torch.float32)
    world_model = LearnedWorldModel(
        env.state_dim,
        env.action_dim,
        config.strategy_dim,
        hidden_dim=config.world_model_hidden_dim,
    ).to(device=device, dtype=torch.float32)
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
        labels = candidate_labels(config.candidate_strategies, max_heuristic_candidates=config.max_heuristic_candidates)
        evaluated = []
        for idx, strategy in enumerate(candidates):
            label = labels[idx] if idx < len(labels) else None
            metrics = evaluator.evaluate_strategy(
                strategy.detach().cpu(),
                label=label,
                policy=policy if config.evaluator_action_source == "policy" else None,
            )
            selection_score = selection_score_from_metrics(metrics, config)
            evaluated.append((selection_score, strategy.detach().cpu(), label, metrics))

        evaluated.sort(key=lambda item: item[0], reverse=True)
        _, selected_strategy, selected_label, selected_metrics = evaluated[0]
        trajectories = [
            collect_policy_rollout(env, policy, selected_strategy, config=config)
            for _ in range(config.episodes_per_update)
        ]
        rollout_summary = summarize_trajectories(trajectories)

        record = StrategyRecord(
            embedding=selected_strategy,
            episode_return=float(rollout_summary["episode_return"]),
            robustness_score=float(selected_metrics["robustness_score"]),
            exploitability_proxy=float(selected_metrics["exploitability_proxy"]),
            average_case_value=float(selected_metrics["average_case_value"]),
            worst_case_value=float(selected_metrics["worst_case_value"]),
            goal_rate=float(selected_metrics["goal_rate"]),
            iteration=iteration,
            label=selected_label,
            metadata={"rollout": rollout_summary, "best_response_label": selected_metrics["best_response_label"]},
        )
        buffer.add(record)

        update_metrics = update_models(
            config=config,
            trajectories=trajectories,
            selected_strategy=selected_strategy,
            selected_label=selected_label,
            iteration=iteration,
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
        if config.early_stop_on_convergence:
            convergence = detect_convergence(
                history,
                metric=config.convergence_metric,
                target=config.convergence_target,
                window=config.convergence_window,
                patience=config.convergence_patience,
                min_iter=config.convergence_min_iter,
            )
            history[-1]["convergence"] = convergence
            if bool(convergence["converged"]):
                break

    return {
        "history": history,
        "buffer_size": len(buffer),
        "buffer_diversity": buffer.diversity(),
        "final_selected_label": history[-1]["selected_label"] if history else None,
        "device": str(device),
    }


def generate_candidate_strategies(ebm: EnergyMLP, config: TrainingConfig) -> torch.Tensor:
    """Generate a mix of named heuristics and sampled latent strategies."""

    device = next(ebm.parameters()).device
    heuristic_labels = list(available_heuristic_strategies())[: config.max_heuristic_candidates]
    strategies: list[torch.Tensor] = []
    for label in heuristic_labels[: config.candidate_strategies]:
        strategies.append(named_strategy_embedding(label, config.strategy_dim, device=device).vector)

    remaining = config.candidate_strategies - len(strategies)
    if remaining > 0:
        if config.sampler_type == "langevin":
            samples = langevin_sample(
                ebm,
                num_samples=remaining,
                strategy_dim=config.strategy_dim,
                steps=config.langevin_steps,
                step_size=config.langevin_step_size,
                noise_scale=config.langevin_noise_scale,
                device=device,
            )
        elif config.sampler_type == "gaussian":
            samples = GaussianStrategySampler(config.strategy_dim, scale=config.gaussian_scale).sample(
                remaining, device=device
            )
        else:
            raise ValueError(f"Unknown sampler_type: {config.sampler_type}")
        strategies.extend([sample for sample in samples])
    return torch.stack(strategies, dim=0)


def candidate_labels(num_candidates: int, max_heuristic_candidates: int = 5) -> list[str | None]:
    """Labels aligned with the heuristic-first candidate generator."""

    labels: list[str | None] = [
        label for label in list(available_heuristic_strategies())[: min(num_candidates, max_heuristic_candidates)]
    ]
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


def selection_score_from_metrics(metrics: dict[str, float | str], config: TrainingConfig) -> float:
    """Score a candidate while rewarding goal-reaching over low-variance losing."""

    if not config.robustness_aware_selection:
        return float(metrics["average_case_value"])
    return (
        config.average_value_weight * float(metrics["average_case_value"])
        + config.goal_rate_weight * float(metrics.get("goal_rate", 0.0))
        + config.robustness_weight * float(metrics["robustness_score"])
        - config.exploitability_weight * float(metrics["exploitability_proxy"])
    )


def summarize_trajectories(trajectories: list[Trajectory]) -> dict[str, float | str | int]:
    """Aggregate rollout summaries over an update batch."""

    if not trajectories:
        return {
            "episode_return": 0.0,
            "training_return": 0.0,
            "outcome": "empty",
            "steps": 0,
            "goal_rate": 0.0,
            "catch_rate": 0.0,
            "timeout_rate": 0.0,
            "win_rate": 0.0,
            "episodes": 0,
        }
    summaries = [trajectory.summary() for trajectory in trajectories]
    return {
        "episode_return": float(sum(float(item["episode_return"]) for item in summaries) / len(summaries)),
        "training_return": float(sum(float(item["training_return"]) for item in summaries) / len(summaries)),
        "outcome": str(summaries[-1]["outcome"]),
        "steps": float(sum(float(item["steps"]) for item in summaries) / len(summaries)),
        "goal_rate": float(sum(float(item["goal_rate"]) for item in summaries) / len(summaries)),
        "catch_rate": float(sum(float(item["catch_rate"]) for item in summaries) / len(summaries)),
        "timeout_rate": float(sum(float(item["timeout_rate"]) for item in summaries) / len(summaries)),
        "win_rate": float(sum(float(item["win_rate"]) for item in summaries) / len(summaries)),
        "episodes": len(summaries),
    }


def collect_policy_rollout(
    env: AttackerDefenderGridworld,
    policy: StrategyConditionedPolicy,
    strategy: Tensor,
    config: TrainingConfig | None = None,
) -> Trajectory:
    """Collect one strategy-conditioned rollout with differentiable log-probs."""

    config = config or TrainingConfig(env=env.config)
    obs = env.reset()
    done = False
    trajectory = Trajectory()
    while not done:
        device = next(policy.parameters()).device
        state = torch.as_tensor(obs, dtype=torch.float32, device=device)
        strategy_batch = strategy.detach().float().to(device).unsqueeze(0)
        logits = policy(state.unsqueeze(0), strategy_batch)
        distribution = Categorical(logits=logits)
        action_tensor = distribution.sample()
        action = int(action_tensor.item())

        phi_before = gridworld_potential(env)
        result = env.step(action)
        phi_after = gridworld_potential(env)
        training_reward = float(result.reward) + config.shaping_coef * (config.gamma * phi_after - phi_before)
        next_state = torch.as_tensor(result.observation, dtype=torch.float32, device=device)

        trajectory.states.append(state.detach())
        trajectory.actions.append(action)
        trajectory.log_probs.append(distribution.log_prob(action_tensor).squeeze(0))
        trajectory.entropies.append(distribution.entropy().squeeze(0))
        trajectory.rewards.append(float(result.reward))
        trajectory.training_rewards.append(float(training_reward))
        trajectory.next_states.append(next_state.detach())
        trajectory.dones.append(bool(result.done))

        obs = result.observation
        done = result.done
        trajectory.outcome = str(result.info["outcome"])

    return trajectory


def gridworld_potential(env: AttackerDefenderGridworld) -> float:
    """Potential for dense progress shaping; higher is closer to the goal."""

    max_manhattan = max(1, 2 * (env.config.grid_size - 1))
    return -float(manhattan(env.attacker_pos, env.goal_pos) / max_manhattan)


def behavior_cloning_loss(
    policy: StrategyConditionedPolicy,
    states: Tensor,
    strategy_batch: Tensor,
    selected_strategy: Tensor,
    selected_label: str | None,
    config: TrainingConfig,
) -> Tensor:
    """Warm-start policy behavior toward the heuristic implied by a strategy."""

    device = states.device
    if config.behavior_clone_coef <= 0.0 or config.behavior_clone_iterations <= 0:
        return torch.zeros((), dtype=torch.float32, device=device)
    targets = torch.tensor(
        [
            heuristic_action_from_state(state.detach().cpu(), selected_strategy.detach().cpu(), selected_label, config.env)
            for state in states
        ],
        dtype=torch.long,
        device=device,
    )
    logits = policy(states, strategy_batch)
    return F.cross_entropy(logits, targets)


def heuristic_action_from_state(
    state: Tensor,
    strategy: Tensor,
    label: str | None,
    env_config: GridworldConfig,
) -> int:
    """Reconstruct a minimal env from an observation and query the heuristic policy."""

    scale = max(1, env_config.grid_size - 1)
    env = AttackerDefenderGridworld(env_config)
    env.attacker_pos = (round(float(state[0]) * scale), round(float(state[1]) * scale))
    env.defender_pos = (round(float(state[2]) * scale), round(float(state[3]) * scale))
    env.goal_pos = (round(float(state[4]) * scale), round(float(state[5]) * scale))
    env.steps = round(float(state[6]) * max(1, env_config.max_steps))
    return attacker_heuristic_action(env, strategy, label=label)


def update_models(
    config: TrainingConfig,
    trajectories: list[Trajectory],
    selected_strategy: Tensor,
    selected_label: str | None,
    iteration: int,
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
    if config.train_policy and any(trajectory.log_probs for trajectory in trajectories):
        updates.update(update_policy(policy, policy_optimizer, trajectories, selected_strategy, selected_label, iteration, config))
    if config.train_world_model and any(trajectory.states for trajectory in trajectories):
        updates.update(update_world_model(world_model, world_model_optimizer, trajectories, selected_strategy, config))
    if config.train_ebm and len(buffer) > 0:
        updates.update(update_ebm(ebm, ebm_optimizer, buffer, config))
    return updates


def update_policy(
    policy: StrategyConditionedPolicy,
    optimizer: torch.optim.Optimizer,
    trajectories: list[Trajectory],
    selected_strategy: Tensor,
    selected_label: str | None,
    iteration: int,
    config: TrainingConfig,
) -> dict[str, float]:
    """Run an actor-critic update for the strategy-conditioned policy."""

    device = next(policy.parameters()).device
    log_probs = torch.cat([torch.stack(trajectory.log_probs) for trajectory in trajectories if trajectory.log_probs])
    entropies = torch.cat([torch.stack(trajectory.entropies) for trajectory in trajectories if trajectory.entropies])
    states = torch.cat([torch.stack(trajectory.states) for trajectory in trajectories if trajectory.states]).to(device)
    returns = torch.cat(
        [
            discounted_returns(trajectory.training_rewards or trajectory.rewards, gamma=config.gamma)
            for trajectory in trajectories
            if trajectory.rewards
        ]
    ).to(device)
    strategy_batch = selected_strategy.detach().float().to(device).unsqueeze(0).expand(states.shape[0], -1)
    values = policy.value(states, strategy_batch)
    advantages = normalize_advantages(returns - values.detach())
    policy_loss = policy_gradient_surrogate(log_probs, advantages)
    value_loss = F.mse_loss(values, returns)
    entropy_loss = -config.entropy_coef * entropies.mean()
    behavior_clone_loss = behavior_cloning_loss(policy, states, strategy_batch, selected_strategy, selected_label, config)
    if iteration >= config.behavior_clone_iterations:
        behavior_clone_loss = torch.zeros((), dtype=torch.float32, device=device)
    loss = policy_loss + config.value_coef * value_loss + entropy_loss + config.behavior_clone_coef * behavior_clone_loss

    optimizer.zero_grad()
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), config.grad_clip_norm)
    optimizer.step()
    return {
        "policy_loss": float(loss.detach().item()),
        "policy_actor_loss": float(policy_loss.detach().item()),
        "policy_value_loss": float(value_loss.detach().item()),
        "behavior_clone_loss": float(behavior_clone_loss.detach().item()),
        "policy_grad_norm": float(grad_norm),
        "policy_entropy": float(entropies.detach().mean().item()),
    }


def update_world_model(
    world_model: LearnedWorldModel,
    optimizer: torch.optim.Optimizer,
    trajectories: list[Trajectory],
    selected_strategy: Tensor,
    config: TrainingConfig,
) -> dict[str, float]:
    """Fit the placeholder world model on observed one-step transitions."""

    device = next(world_model.parameters()).device
    states = torch.cat([torch.stack(trajectory.states) for trajectory in trajectories if trajectory.states]).to(device)
    next_states = torch.cat([torch.stack(trajectory.next_states) for trajectory in trajectories if trajectory.next_states]).to(
        device
    )
    actions = torch.tensor(
        [action for trajectory in trajectories for action in trajectory.actions],
        dtype=torch.long,
        device=device,
    )
    action_one_hot = F.one_hot(actions, num_classes=world_model.action_dim).float()
    strategy_batch = selected_strategy.detach().float().unsqueeze(0).expand(states.shape[0], -1)
    strategy_batch = strategy_batch.to(device)
    rewards = torch.tensor(
        [reward for trajectory in trajectories for reward in trajectory.rewards],
        dtype=torch.float32,
        device=device,
    )

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

    if config.use_buffer_positives:
        positive = buffer.sample_positive(config.ebm_batch_size, quantile=config.positive_quantile).to(
            next(ebm.parameters()).device
        )
    else:
        positive = torch.randn(config.ebm_batch_size, config.strategy_dim, device=next(ebm.parameters()).device)
    negative = langevin_sample(
        ebm,
        num_samples=config.ebm_batch_size,
        strategy_dim=config.strategy_dim,
        steps=config.langevin_steps,
        step_size=config.langevin_step_size,
        noise_scale=config.langevin_noise_scale,
        device=next(ebm.parameters()).device,
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
