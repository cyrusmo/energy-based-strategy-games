"""Draft rollout config helpers shared by the pursuit trace viewer example."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from strategy_games.envs.multi_evader_pursuit import MultiEvaderPursuitConfig
from strategy_games.rollouts import PursuitRolloutConfig
from strategy_games.viewers.pursuit_render_data import trace_metadata


def default_draft_config() -> dict[str, object]:
    return {
        "seed": 7,
        "grid_size": [9, 9],
        "num_evaders": 2,
        "num_pursuers": 1,
        "max_steps": 30,
        "catch_radius": 0,
        "pursuer_policy": "pursuer_greedy_nearest",
        "evader_policy": "evader_feint",
        "feint_steps": 3,
    }


def rollout_config_from_draft(draft: Mapping[str, object]) -> PursuitRolloutConfig:
    grid_size = draft["grid_size"]
    if not isinstance(grid_size, list):
        raise ValueError("draft grid_size must be a list")
    return PursuitRolloutConfig(
        seed=int(draft["seed"]),
        pursuer_policy=str(draft["pursuer_policy"]),
        evader_policy=str(draft["evader_policy"]),
        feint_steps=int(draft["feint_steps"]),
        env=MultiEvaderPursuitConfig(
            grid_size=(int(grid_size[0]), int(grid_size[1])),
            num_evaders=int(draft["num_evaders"]),
            num_pursuers=int(draft["num_pursuers"]),
            max_steps=int(draft["max_steps"]),
            catch_radius=int(draft["catch_radius"]),
        ),
    )


def active_config_from_trace(trace: Any) -> dict[str, object]:
    metadata = trace_metadata(trace)
    return {
        "source": "loaded_trace",
        "env_id": metadata["env_id"],
        "episode_id": metadata["episode_id"],
        "seed": metadata["seed"],
        "grid_size": metadata["grid_size"],
        "num_evaders": metadata["num_evaders"],
        "num_pursuers": metadata["num_pursuers"],
        "max_steps": metadata["max_steps"],
        "catch_radius": metadata["catch_radius"],
    }
