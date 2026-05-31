"""Experiment runner for YAML configs."""

from __future__ import annotations

from pathlib import Path

from strategy_games.experiments.logging import ExperimentLogger
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
    policy_raw = raw.get("policy", {})
    world_model_raw = raw.get("world_model", {})
    sampler_raw = raw.get("sampler", {})
    selection_raw = raw.get("selection", {})
    updates_raw = raw.get("updates", {})
    convergence_raw = raw.get("convergence", {})
    config = TrainingConfig(
        seed=int(raw.get("seed", 0)),
        iterations=int(training_raw.get("iterations", 3)),
        candidate_strategies=int(training_raw.get("candidate_strategies", 6)),
        strategy_dim=int(training_raw.get("strategy_dim", 8)),
        policy_hidden_dim=int(policy_raw.get("hidden_dim", 64)),
        ebm_hidden_dim=int(ebm_raw.get("hidden_dim", 64)),
        world_model_hidden_dim=int(world_model_raw.get("hidden_dim", 64)),
        langevin_steps=int(ebm_raw.get("langevin_steps", 10)),
        langevin_step_size=float(ebm_raw.get("langevin_step_size", 0.02)),
        episodes_per_opponent=int(evaluator_raw.get("episodes_per_opponent", 1)),
        episodes_per_update=int(training_raw.get("episodes_per_update", 1)),
        policy_lr=float(updates_raw.get("policy_lr", 3e-3)),
        ebm_lr=float(updates_raw.get("ebm_lr", 1e-3)),
        world_model_lr=float(updates_raw.get("world_model_lr", 1e-3)),
        gamma=float(updates_raw.get("gamma", 0.99)),
        entropy_coef=float(updates_raw.get("entropy_coef", 0.01)),
        value_coef=float(updates_raw.get("value_coef", 0.5)),
        grad_clip_norm=float(updates_raw.get("grad_clip_norm", 1.0)),
        ebm_batch_size=int(updates_raw.get("ebm_batch_size", 8)),
        positive_quantile=float(updates_raw.get("positive_quantile", 0.5)),
        sampler_type=str(sampler_raw.get("type", "langevin")),
        gaussian_scale=float(sampler_raw.get("scale", 1.0)),
        langevin_noise_scale=float(sampler_raw.get("noise_scale", ebm_raw.get("langevin_noise_scale", 0.25))),
        max_heuristic_candidates=int(sampler_raw.get("max_heuristic_candidates", 5)),
        robustness_aware_selection=bool(selection_raw.get("robustness_aware", True)),
        average_value_weight=float(selection_raw.get("average_value_weight", 1.0)),
        robustness_weight=float(selection_raw.get("robustness_weight", 0.5)),
        exploitability_weight=float(selection_raw.get("exploitability_weight", 0.5)),
        goal_rate_weight=float(selection_raw.get("goal_rate_weight", 1.0)),
        use_buffer_positives=bool(updates_raw.get("use_buffer_positives", True)),
        train_policy=bool(updates_raw.get("train_policy", True)),
        train_ebm=bool(updates_raw.get("train_ebm", True)),
        train_world_model=bool(updates_raw.get("train_world_model", True)),
        evaluator_action_source=str(evaluator_raw.get("action_source", "heuristic")),
        behavior_clone_iterations=int(updates_raw.get("behavior_clone_iterations", 0)),
        behavior_clone_coef=float(updates_raw.get("behavior_clone_coef", 0.0)),
        shaping_coef=float(updates_raw.get("shaping_coef", 0.0)),
        convergence_metric=str(convergence_raw.get("metric", "goal_rate")),
        convergence_target=float(convergence_raw.get("target", 1.0)),
        convergence_patience=int(convergence_raw.get("patience", 3)),
        convergence_window=int(convergence_raw.get("window", 3)),
        convergence_min_iter=int(convergence_raw.get("min_iter", 1)),
        early_stop_on_convergence=bool(convergence_raw.get("early_stop", False)),
        device=str(raw.get("device", "auto")),
        env=env,
    )
    result = run_training_loop(config)
    logging_raw = raw.get("logging", {})
    if bool(logging_raw.get("enabled", False)):
        output_dir = logging_raw.get("output_dir", "outputs/public/debug")
        run_name = logging_raw.get("run_name", Path(path).stem)
        logger = ExperimentLogger(output_dir=output_dir, run_name=str(run_name))
        result["artifacts"] = logger.log_run(raw, result)
    return result
