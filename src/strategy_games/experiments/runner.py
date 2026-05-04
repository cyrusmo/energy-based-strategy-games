"""Experiment runner for YAML configs."""

from __future__ import annotations

from pathlib import Path

from strategy_games.envs.gridworld import GridworldConfig
from strategy_games.training.train_loop import TrainingConfig, run_training_loop
from strategy_games.utils.config import load_config


def run_from_config(path: str | Path) -> dict[str, object]:
    """Load a YAML config and run the scaffold training loop."""

    raw = load_config(path)
    env_raw = raw.get("env", {})
    env = GridworldConfig(
        grid_size=int(env_raw.get("grid_size", 10)),
        max_steps=int(env_raw.get("max_steps", 50)),
        attacker_start=tuple(env_raw.get("attacker_start", [0, 0])),  # type: ignore[arg-type]
        defender_start=tuple(env_raw.get("defender_start", [9, 9])),  # type: ignore[arg-type]
        goal_pos=tuple(env_raw.get("goal_pos", [9, 0])),  # type: ignore[arg-type]
        catch_radius=int(env_raw.get("catch_radius", 0)),
    )
    training_raw = raw.get("training", {})
    ebm_raw = raw.get("ebm", {})
    evaluator_raw = raw.get("evaluator", {})
    config = TrainingConfig(
        seed=int(raw.get("seed", 0)),
        iterations=int(training_raw.get("iterations", 3)),
        candidate_strategies=int(training_raw.get("candidate_strategies", 6)),
        strategy_dim=int(training_raw.get("strategy_dim", 8)),
        ebm_hidden_dim=int(ebm_raw.get("hidden_dim", 64)),
        langevin_steps=int(ebm_raw.get("langevin_steps", 10)),
        langevin_step_size=float(ebm_raw.get("langevin_step_size", 0.02)),
        episodes_per_opponent=int(evaluator_raw.get("episodes_per_opponent", 1)),
        env=env,
    )
    return run_training_loop(config)
