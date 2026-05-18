"""Rollout runner that emits validated pursuit/evasion traces."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import numpy as np

from strategy_games.envs.multi_evader_pursuit import MultiEvaderPursuitConfig, MultiEvaderPursuitEnv
from strategy_games.policies.scripted_pursuit import scripted_pursuit_actions
from strategy_games.traces.pursuit_trace import (
    SCHEMA_VERSION,
    TRACE_TYPE,
    PursuitStep,
    PursuitSummary,
    PursuitTrace,
    validate_pursuit_trace,
)


@dataclass(frozen=True)
class PursuitRolloutConfig:
    """Configuration for scripted pursuit/evasion trace generation."""

    seed: int = 0
    episode_id: str | None = None
    pursuer_policy: str = "pursuer_greedy_nearest"
    evader_policy: str = "evader_feint"
    feint_steps: int = 3
    config_path: str | None = None
    created_at: str | None = None
    env: MultiEvaderPursuitConfig = field(default_factory=MultiEvaderPursuitConfig)


def run_scripted_pursuit_rollout(config: PursuitRolloutConfig | None = None) -> PursuitTrace:
    """Run scripted policies in the custom env and return a validated trace."""

    config = config or PursuitRolloutConfig()
    rng = np.random.default_rng(config.seed)
    env = MultiEvaderPursuitEnv(config.env)
    initial_positions = _positions_to_json(env.reset())
    steps: list[PursuitStep] = []

    done = False
    while not done:
        actions = scripted_pursuit_actions(
            env=env,
            pursuer_policy=config.pursuer_policy,
            evader_policy=config.evader_policy,
            rng=rng,
            step_index=len(steps),
            feint_steps=config.feint_steps,
        )
        result = env.step(actions)
        step = PursuitStep(
            t=len(steps),
            agent_positions=_positions_to_json(result.agent_positions),
            agent_roles=env.agent_roles(),
            actions=dict(actions),
            step_rewards={agent_id: float(reward) for agent_id, reward in result.rewards.items()},
            captures=result.captures,
            active_evaders=list(result.active_evaders),
            done=result.done,
        )
        steps.append(step)
        done = result.done

    summary = _build_summary(env, initial_positions, steps)
    trace = PursuitTrace(
        schema_version=SCHEMA_VERSION,
        trace_type=TRACE_TYPE,
        env_id=env.env_id,
        episode_id=config.episode_id or f"{env.env_id}-seed-{config.seed}",
        seed=config.seed,
        grid_size=[env.grid_size[0], env.grid_size[1]],
        num_evaders=env.config.num_evaders,
        num_pursuers=env.config.num_pursuers,
        metadata={
            "policy_mode": "scripted",
            "pursuer_policy": config.pursuer_policy,
            "evader_policy": config.evader_policy,
            "feint_steps": config.feint_steps,
            "catch_radius": env.config.catch_radius,
            "max_steps": env.config.max_steps,
            "created_at": config.created_at or datetime.now(UTC).replace(microsecond=0).isoformat(),
            "config_path": config.config_path,
            "env_params": {
                "grid_size": [env.grid_size[0], env.grid_size[1]],
                "num_evaders": env.config.num_evaders,
                "num_pursuers": env.config.num_pursuers,
                "max_steps": env.config.max_steps,
                "catch_radius": env.config.catch_radius,
            },
        },
        steps=steps,
        summary=summary,
    )
    validate_pursuit_trace(trace)
    return trace


def pursuit_rollout_config_from_mapping(raw: Mapping[str, Any], config_path: str | None = None) -> PursuitRolloutConfig:
    """Build a rollout config from a YAML-style mapping."""

    policies = raw.get("policies", {})
    if not isinstance(policies, Mapping):
        policies = {}
    return PursuitRolloutConfig(
        seed=int(raw.get("seed", 0)),
        episode_id=str(raw["episode_id"]) if raw.get("episode_id") is not None else None,
        pursuer_policy=str(policies.get("pursuer", "pursuer_greedy_nearest")),
        evader_policy=str(policies.get("evader", "evader_feint")),
        feint_steps=int(policies.get("feint_steps", 3)),
        config_path=config_path,
        created_at=str(raw["created_at"]) if raw.get("created_at") is not None else None,
        env=multi_evader_config_from_mapping(raw.get("env", {})),
    )


def multi_evader_config_from_mapping(raw: Any) -> MultiEvaderPursuitConfig:
    """Build a multi-evader environment config from a YAML-style mapping."""

    if not isinstance(raw, Mapping):
        raw = {}
    return MultiEvaderPursuitConfig(
        grid_size=_grid_size(raw.get("grid_size", [9, 9])),
        num_evaders=int(raw.get("num_evaders", 2)),
        num_pursuers=int(raw.get("num_pursuers", 1)),
        max_steps=int(raw.get("max_steps", 30)),
        catch_radius=int(raw.get("catch_radius", 0)),
        pursuer_starts=_positions_tuple(raw.get("pursuer_starts")),
        evader_starts=_positions_tuple(raw.get("evader_starts")),
        evader_goals=_positions_tuple(raw.get("evader_goals")),
        evader_survival_reward=float(raw.get("evader_survival_reward", 1.0)),
        evader_capture_reward=float(raw.get("evader_capture_reward", -10.0)),
        evader_timeout_bonus=float(raw.get("evader_timeout_bonus", 10.0)),
        pursuer_step_reward=float(raw.get("pursuer_step_reward", -0.1)),
        pursuer_capture_reward=float(raw.get("pursuer_capture_reward", 10.0)),
        pursuer_all_captured_bonus=float(raw.get("pursuer_all_captured_bonus", 5.0)),
    )


def _build_summary(
    env: MultiEvaderPursuitEnv,
    initial_positions: dict[str, list[int]],
    steps: list[PursuitStep],
) -> PursuitSummary:
    captured = [evader_id for evader_id, status in env.per_evader_status.items() if status == "captured"]
    survived = [evader_id for evader_id, status in env.per_evader_status.items() if status == "survived"]
    final_positions = _positions_to_json(env.positions())
    pursuer_returns = [env.per_agent_returns[pursuer_id] for pursuer_id in env.pursuer_ids]
    evader_returns = [env.per_agent_returns[evader_id] for evader_id in env.evader_ids]
    terminated_reason = env.terminated_reason if env.terminated_reason in {"all_evaders_captured", "timeout"} else "unknown"
    return PursuitSummary(
        outcome=terminated_reason,
        terminated_reason=terminated_reason,  # type: ignore[arg-type]
        capture_rate=float(len(captured) / env.config.num_evaders),
        survival_rate=float(len(survived) / env.config.num_evaders),
        all_evaders_captured=len(captured) == env.config.num_evaders,
        mean_evader_return=float(np.mean(evader_returns)),
        mean_pursuer_return=float(np.mean(pursuer_returns)),
        per_agent_returns={agent_id: float(value) for agent_id, value in env.per_agent_returns.items()},
        initial_positions=initial_positions,
        final_positions=final_positions if steps else initial_positions,
        total_steps=len(steps),
        per_evader_status=dict(env.per_evader_status),
    )


def _positions_to_json(positions: Mapping[str, tuple[int, int]]) -> dict[str, list[int]]:
    return {agent_id: [int(position[0]), int(position[1])] for agent_id, position in positions.items()}


def _grid_size(raw: Any) -> tuple[int, int]:
    if isinstance(raw, int):
        return (raw, raw)
    if not isinstance(raw, (list, tuple)) or len(raw) != 2:
        raise ValueError("grid_size must be an int or [height, width]")
    return (int(raw[0]), int(raw[1]))


def _positions_tuple(raw: Any) -> tuple[tuple[int, int], ...] | None:
    if raw is None:
        return None
    if not isinstance(raw, list):
        raise ValueError("positions must be a list of [row, col] pairs")
    return tuple((int(position[0]), int(position[1])) for position in raw)
