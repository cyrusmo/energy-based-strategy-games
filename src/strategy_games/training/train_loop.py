"""Generate-evaluate-execute-update training loop scaffold."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from strategy_games.envs.gridworld import AttackerDefenderGridworld, GridworldConfig
from strategy_games.evaluation.best_response import GameTheoreticEvaluator
from strategy_games.models.ebm import EnergyMLP
from strategy_games.models.policy import StrategyConditionedPolicy
from strategy_games.strategies.buffer import StrategyBuffer, StrategyRecord
from strategy_games.strategies.embeddings import available_heuristic_strategies, named_strategy_embedding, pairwise_diversity
from strategy_games.strategies.sampler import langevin_sample
from strategy_games.utils.seeding import set_global_seed


@dataclass(frozen=True)
class TrainingConfig:
    """Configuration for the first debug training loop."""

    seed: int = 0
    iterations: int = 3
    candidate_strategies: int = 6
    strategy_dim: int = 8
    ebm_hidden_dim: int = 64
    langevin_steps: int = 10
    langevin_step_size: float = 0.02
    episodes_per_opponent: int = 1
    env: GridworldConfig = field(default_factory=GridworldConfig)


def run_training_loop(config: TrainingConfig | None = None) -> dict[str, object]:
    """Run a tiny end-to-end loop without claiming full RL training.

    Each iteration samples strategies, evaluates them against heuristic defender
    responses, executes the selected strategy with an untrained
    strategy-conditioned policy, and records summaries in a strategy buffer.
    """

    config = config or TrainingConfig()
    set_global_seed(config.seed)

    env = AttackerDefenderGridworld(config.env)
    ebm = EnergyMLP(config.strategy_dim, hidden_dim=config.ebm_hidden_dim)
    policy = StrategyConditionedPolicy(env.state_dim, config.strategy_dim, env.action_dim)
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
        rollout_summary = execute_policy_rollout(env, policy, selected_strategy)

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

        # TODO: update policy with rollout trajectories.
        # TODO: update EBM with positive buffer samples and negative Langevin samples.
        # TODO: update world model from observed transitions.

        history.append(
            {
                "iteration": iteration,
                "selected_label": selected_label or "latent",
                "selection_metrics": selected_metrics,
                "rollout": rollout_summary,
                "buffer_size": len(buffer),
                "candidate_diversity": pairwise_diversity(candidates.detach().cpu()),
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
    """Execute the current policy conditioned on a selected strategy."""

    obs = env.reset()
    done = False
    total_reward = 0.0
    info: dict[str, object] = {"outcome": "running", "steps": 0}
    while not done:
        state = torch.as_tensor(obs, dtype=torch.float32)
        action = policy.act(state, strategy, deterministic=False)
        result = env.step(action)
        obs = result.observation
        total_reward += result.reward
        done = result.done
        info = result.info

    outcome = str(info["outcome"])
    return {
        "episode_return": float(total_reward),
        "outcome": outcome,
        "steps": int(info["steps"]),
        "goal_rate": float(outcome == "goal"),
        "catch_rate": float(outcome == "caught"),
        "win_rate": float(outcome == "goal"),
    }
