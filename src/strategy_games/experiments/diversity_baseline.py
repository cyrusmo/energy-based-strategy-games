"""Diversity-only baseline: Gaussian strategy samples without adversarial scoring."""

from __future__ import annotations

from dataclasses import replace

from strategy_games.envs.gridworld_v2 import GridworldV2Config
from strategy_games.training.train_loop import TrainingConfig, run_training_loop


def run_diversity_baseline(
    env: GridworldV2Config | None = None,
    seed: int = 0,
    iterations: int = 4,
    candidate_strategies: int = 16,
) -> dict[str, object]:
    """Sample latent strategies uniformly and select by average-case value only."""

    config = TrainingConfig(
        seed=seed,
        iterations=iterations,
        candidate_strategies=candidate_strategies,
        max_heuristic_candidates=0,
        sampler_type="gaussian",
        robustness_aware_selection=False,
        evaluator_action_source="heuristic",
        executor_action_source="parametric",
        train_policy=False,
        train_ebm=False,
        train_world_model=False,
    )
    if env is not None:
        config = replace(config, env=env)
    result = run_training_loop(config)
    result["method"] = "diversity_only"
    return result
