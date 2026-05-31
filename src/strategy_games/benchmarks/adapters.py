"""Benchmark adapters for custom and optional external environments."""

from __future__ import annotations

import os
import time
from collections.abc import Mapping
from typing import Any, Protocol

import numpy as np

from strategy_games.benchmarks.types import BenchmarkDependencyError, BenchmarkResult
from strategy_games.envs.gridworld import AttackerDefenderGridworld, GridworldConfig, greedy_action_towards
from strategy_games.models.policy import RandomPolicy
from strategy_games.training.train_loop import TrainingConfig, run_training_loop


class BenchmarkAdapter(Protocol):
    """Minimal benchmark adapter interface."""

    env_id: str

    def rollout(self, baseline: str, seed: int) -> BenchmarkResult:
        """Run one seeded benchmark rollout."""


class CustomGridworldBenchmarkAdapter:
    """Adapter for the repository's custom attacker-defender gridworld."""

    env_id = "custom_gridworld_v0"

    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        config = config or {}
        self.env_config = _gridworld_config(config.get("env", {}))
        self.training_config = _training_config(config, self.env_config)

    def rollout(self, baseline: str, seed: int) -> BenchmarkResult:
        """Run one seeded custom-gridworld benchmark rollout."""

        if baseline == "random_policy":
            return self._random_policy(seed)
        if baseline == "direct_goal_heuristic":
            return self._direct_goal(seed)
        if baseline == "strategy_loop":
            return self._strategy_loop(seed)
        raise KeyError(f"Unsupported custom gridworld baseline: {baseline}")

    def _random_policy(self, seed: int) -> BenchmarkResult:
        start = time.perf_counter()
        env = AttackerDefenderGridworld(self.env_config)
        policy = RandomPolicy(action_dim=env.action_dim, seed=seed)
        env.reset()
        total_reward = 0.0
        done = False
        info: dict[str, object] = {"outcome": "running", "steps": 0}
        while not done:
            result = env.step(policy.act())
            total_reward += result.reward
            done = result.done
            info = result.info
        return _custom_result(
            baseline="random_policy",
            seed=seed,
            total_reward=total_reward,
            outcome=str(info["outcome"]),
            steps=_safe_int(info.get("steps", 0)),
            elapsed=time.perf_counter() - start,
            strategy_label="uniform_random",
        )

    def _direct_goal(self, seed: int) -> BenchmarkResult:
        start = time.perf_counter()
        env = AttackerDefenderGridworld(self.env_config)
        env.reset()
        total_reward = 0.0
        done = False
        info: dict[str, object] = {"outcome": "running", "steps": 0}
        while not done:
            result = env.step(greedy_action_towards(env.attacker_pos, env.goal_pos))
            total_reward += result.reward
            done = result.done
            info = result.info
        return _custom_result(
            baseline="direct_goal_heuristic",
            seed=seed,
            total_reward=total_reward,
            outcome=str(info["outcome"]),
            steps=_safe_int(info.get("steps", 0)),
            elapsed=time.perf_counter() - start,
            strategy_label="direct_goal",
        )

    def _strategy_loop(self, seed: int) -> BenchmarkResult:
        start = time.perf_counter()
        config = _replace_training_seed(self.training_config, seed)
        result = run_training_loop(config)
        history = result["history"]
        if not isinstance(history, list) or not history:
            raise ValueError("strategy loop produced empty history")
        rollouts = [item["rollout"] for item in history if isinstance(item, dict)]
        final = history[-1]
        selection_metrics = final["selection_metrics"]
        mean_return = float(np.mean([float(item["episode_return"]) for item in rollouts]))
        mean_win = float(np.mean([float(item["win_rate"]) for item in rollouts]))
        mean_goal = float(np.mean([float(item["goal_rate"]) for item in rollouts]))
        mean_catch = float(np.mean([float(item["catch_rate"]) for item in rollouts]))
        mean_timeout = float(np.mean([float(item.get("timeout_rate", 0.0)) for item in rollouts]))
        mean_steps = int(round(float(np.mean([float(item["steps"]) for item in rollouts]))))
        return BenchmarkResult(
            env_id=self.env_id,
            baseline="strategy_loop",
            seed=seed,
            episode_return=mean_return,
            win_rate=mean_win,
            goal_rate=mean_goal,
            catch_rate=mean_catch,
            timeout_rate=mean_timeout,
            survival_or_capture_rate=mean_catch,
            steps=mean_steps,
            strategy_label=str(final["selected_label"]),
            average_case_value=float(selection_metrics["average_case_value"]),
            worst_case_value=float(selection_metrics["worst_case_value"]),
            exploitability_proxy=float(selection_metrics["exploitability_proxy"]),
            strategy_diversity=_safe_float(result.get("buffer_diversity", 0.0)),
            wall_clock_seconds=time.perf_counter() - start,
        )


class PettingZooPursuitBenchmarkAdapter:
    """Optional adapter for PettingZoo Pursuit."""

    env_id = "pettingzoo_pursuit_v4"

    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        config = config or {}
        self.env_kwargs = dict(config.get("env", {}))
        self.env_kwargs.setdefault("max_cycles", 25)
        self.env_kwargs.setdefault("x_size", 8)
        self.env_kwargs.setdefault("y_size", 8)
        self.env_kwargs.setdefault("n_evaders", 4)
        self.env_kwargs.setdefault("n_pursuers", 2)
        self.env_kwargs.setdefault("obs_range", 5)

    def rollout(self, baseline: str, seed: int) -> BenchmarkResult:
        """Run one seeded PettingZoo Pursuit rollout."""

        if baseline not in {"random_policy", "direct_goal_heuristic", "strategy_loop"}:
            raise KeyError(f"Unsupported PettingZoo pursuit baseline: {baseline}")
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
        try:
            from pettingzoo.sisl import pursuit_v4
        except ModuleNotFoundError as exc:
            raise BenchmarkDependencyError(
                "PettingZoo Pursuit is optional. Install with `pip install -e '.[bench]'`."
            ) from exc

        start = time.perf_counter()
        env = pursuit_v4.parallel_env(**self.env_kwargs)
        observations, _ = env.reset(seed=seed)
        total_reward = 0.0
        steps = 0
        rng = np.random.default_rng(seed)
        del observations
        while env.agents:
            actions = {
                agent: _pettingzoo_action(env, agent, baseline, steps, rng)
                for agent in env.agents
            }
            _, rewards, terminations, truncations, _ = env.step(actions)
            total_reward += float(sum(rewards.values())) / max(1, len(rewards))
            steps += 1
            if all(terminations.values()) or all(truncations.values()):
                break
        env.close()

        capture_proxy = float(total_reward > 0.0)
        return BenchmarkResult(
            env_id=self.env_id,
            baseline=baseline,
            seed=seed,
            episode_return=float(total_reward),
            win_rate=capture_proxy,
            goal_rate=0.0,
            catch_rate=capture_proxy,
            timeout_rate=float(steps >= int(self.env_kwargs["max_cycles"])),
            survival_or_capture_rate=capture_proxy,
            steps=steps,
            strategy_label=_pettingzoo_strategy_label(baseline),
            wall_clock_seconds=time.perf_counter() - start,
        )


def _custom_result(
    baseline: str,
    seed: int,
    total_reward: float,
    outcome: str,
    steps: int,
    elapsed: float,
    strategy_label: str,
) -> BenchmarkResult:
    goal_rate = float(outcome == "goal")
    catch_rate = float(outcome == "caught")
    timeout_rate = float(outcome == "timeout")
    return BenchmarkResult(
        env_id=CustomGridworldBenchmarkAdapter.env_id,
        baseline=baseline,
        seed=seed,
        episode_return=float(total_reward),
        win_rate=goal_rate,
        goal_rate=goal_rate,
        catch_rate=catch_rate,
        timeout_rate=timeout_rate,
        survival_or_capture_rate=catch_rate,
        steps=steps,
        strategy_label=strategy_label,
        wall_clock_seconds=elapsed,
    )


def _gridworld_config(raw: object) -> GridworldConfig:
    if not isinstance(raw, Mapping):
        return GridworldConfig()
    return GridworldConfig(
        grid_size=int(raw.get("grid_size", 10)),
        max_steps=int(raw.get("max_steps", 50)),
        attacker_start=tuple(raw.get("attacker_start", [0, 0])),  # type: ignore[arg-type]
        defender_start=tuple(raw.get("defender_start", [9, 9])),  # type: ignore[arg-type]
        goal_pos=tuple(raw.get("goal_pos", [9, 0])),  # type: ignore[arg-type]
        catch_radius=int(raw.get("catch_radius", 0)),
    )


def _training_config(raw: Mapping[str, Any], env_config: GridworldConfig) -> TrainingConfig:
    training = raw.get("training", {})
    ebm = raw.get("ebm", {})
    updates = raw.get("updates", {})
    return TrainingConfig(
        seed=int(raw.get("seed", 0)),
        iterations=int(training.get("iterations", 2)),
        candidate_strategies=int(training.get("candidate_strategies", 4)),
        strategy_dim=int(training.get("strategy_dim", 8)),
        ebm_hidden_dim=int(ebm.get("hidden_dim", 32)),
        langevin_steps=int(ebm.get("langevin_steps", 3)),
        langevin_step_size=float(ebm.get("langevin_step_size", 0.02)),
        ebm_batch_size=int(updates.get("ebm_batch_size", 4)),
        episodes_per_opponent=int(raw.get("evaluator", {}).get("episodes_per_opponent", 1)),
        device=str(raw.get("device", "auto")),
        env=env_config,
    )


def _replace_training_seed(config: TrainingConfig, seed: int) -> TrainingConfig:
    return TrainingConfig(
        seed=seed,
        iterations=config.iterations,
        candidate_strategies=config.candidate_strategies,
        strategy_dim=config.strategy_dim,
        policy_hidden_dim=config.policy_hidden_dim,
        ebm_hidden_dim=config.ebm_hidden_dim,
        world_model_hidden_dim=config.world_model_hidden_dim,
        langevin_steps=config.langevin_steps,
        langevin_step_size=config.langevin_step_size,
        episodes_per_opponent=config.episodes_per_opponent,
        episodes_per_update=config.episodes_per_update,
        policy_lr=config.policy_lr,
        ebm_lr=config.ebm_lr,
        world_model_lr=config.world_model_lr,
        gamma=config.gamma,
        entropy_coef=config.entropy_coef,
        value_coef=config.value_coef,
        grad_clip_norm=config.grad_clip_norm,
        ebm_batch_size=config.ebm_batch_size,
        positive_quantile=config.positive_quantile,
        sampler_type=config.sampler_type,
        gaussian_scale=config.gaussian_scale,
        langevin_noise_scale=config.langevin_noise_scale,
        max_heuristic_candidates=config.max_heuristic_candidates,
        robustness_aware_selection=config.robustness_aware_selection,
        average_value_weight=config.average_value_weight,
        robustness_weight=config.robustness_weight,
        exploitability_weight=config.exploitability_weight,
        goal_rate_weight=config.goal_rate_weight,
        use_buffer_positives=config.use_buffer_positives,
        train_policy=config.train_policy,
        train_ebm=config.train_ebm,
        train_world_model=config.train_world_model,
        evaluator_action_source=config.evaluator_action_source,
        behavior_clone_iterations=config.behavior_clone_iterations,
        behavior_clone_coef=config.behavior_clone_coef,
        shaping_coef=config.shaping_coef,
        convergence_metric=config.convergence_metric,
        convergence_target=config.convergence_target,
        convergence_patience=config.convergence_patience,
        convergence_window=config.convergence_window,
        convergence_min_iter=config.convergence_min_iter,
        early_stop_on_convergence=config.early_stop_on_convergence,
        device=config.device,
        env=config.env,
    )


def _safe_int(value: object, default: int = 0) -> int:
    return int(value) if isinstance(value, int | float | str) else default


def _safe_float(value: object, default: float = 0.0) -> float:
    return float(value) if isinstance(value, int | float | str) else default


def _pettingzoo_action(env: Any, agent: str, baseline: str, step: int, rng: np.random.Generator) -> int:
    action_space = env.action_space(agent)
    if baseline == "random_policy":
        return int(action_space.sample())
    if baseline == "direct_goal_heuristic":
        return int((step + _agent_index(agent)) % action_space.n)
    return int((step // 2 + _agent_index(agent) + rng.integers(0, action_space.n)) % action_space.n)


def _agent_index(agent: str) -> int:
    try:
        return int(agent.rsplit("_", 1)[-1])
    except ValueError:
        return 0


def _pettingzoo_strategy_label(baseline: str) -> str:
    if baseline == "random_policy":
        return "uniform_random"
    if baseline == "direct_goal_heuristic":
        return "coordinated_sweep_proxy"
    return "latent_strategy_proxy"
