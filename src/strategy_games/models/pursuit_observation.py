"""Versioned observation encoders for trainable pursuit policies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from strategy_games.envs.multi_evader_pursuit import MultiEvaderPursuitEnv
from strategy_games.envs.pursuit_actions import manhattan

OBSERVATION_SCHEMA = "pursuit_obs/v1"
PADDING_VALUE = 0.0


@dataclass(frozen=True)
class PursuitObservationSpec:
    """Fixed observation contract for pursuit policies."""

    role: Literal["pursuer", "evader"] = "pursuer"
    max_pursuers: int = 1
    max_evaders: int = 2
    padding_value: float = PADDING_VALUE

    @property
    def schema(self) -> str:
        """Return the observation schema name."""

        return OBSERVATION_SCHEMA

    @property
    def feature_order(self) -> list[str]:
        """Return feature names in encoder order."""

        if self.role == "pursuer":
            features = [
                "own_row_norm",
                "own_col_norm",
                "time_frac",
                "grid_height_norm",
                "grid_width_norm",
                "catch_radius_norm",
            ]
            for idx in range(self.max_evaders):
                prefix = f"evader_{idx}"
                features.extend(
                    [
                        f"{prefix}_row_norm",
                        f"{prefix}_col_norm",
                        f"{prefix}_active_mask",
                        f"{prefix}_captured_mask",
                        f"{prefix}_distance_norm",
                        f"{prefix}_goal_row_norm",
                        f"{prefix}_goal_col_norm",
                    ]
                )
            return features

        features = [
            "own_row_norm",
            "own_col_norm",
            "goal_row_norm",
            "goal_col_norm",
            "active_mask",
            "captured_mask",
            "time_frac",
            "grid_height_norm",
            "grid_width_norm",
            "catch_radius_norm",
        ]
        for idx in range(self.max_pursuers):
            prefix = f"pursuer_{idx}"
            features.extend(
                [
                    f"{prefix}_row_norm",
                    f"{prefix}_col_norm",
                    f"{prefix}_present_mask",
                    f"{prefix}_distance_norm",
                ]
            )
        return features

    @property
    def obs_dim(self) -> int:
        """Return encoded observation dimension."""

        return len(self.feature_order)

    def to_dict(self) -> dict[str, object]:
        """Return JSON-serializable metadata for artifacts and checkpoints."""

        return {
            "observation_schema": self.schema,
            "role": self.role,
            "max_pursuers": self.max_pursuers,
            "max_evaders": self.max_evaders,
            "obs_dim": self.obs_dim,
            "padding_value": self.padding_value,
            "feature_order": self.feature_order,
            "normalization": "positions divide by grid bounds; distances and catch radius divide by max Manhattan distance",
            "mask_semantics": {
                "active_mask": "1 for active controlled/opponent evader, else 0",
                "captured_mask": "1 for captured evader, else 0",
                "present_mask": "1 for present pursuer slot, else 0",
            },
        }


def encode_pursuer_observation(
    env: MultiEvaderPursuitEnv,
    pursuer_id: str = "pursuer_0",
    spec: PursuitObservationSpec | None = None,
) -> np.ndarray:
    """Encode one pursuer-centric observation."""

    spec = spec or PursuitObservationSpec(role="pursuer")
    if spec.role != "pursuer":
        raise ValueError("spec role must be pursuer")
    if pursuer_id not in env.pursuer_positions:
        raise ValueError(f"unknown pursuer_id: {pursuer_id}")
    if env.config.num_evaders > spec.max_evaders:
        raise ValueError("environment has more evaders than observation spec supports")

    height, width = env.grid_size
    scale_y, scale_x, max_dist = _scales(env)
    own = env.pursuer_positions[pursuer_id]
    values = [
        own[0] / scale_y,
        own[1] / scale_x,
        env.steps / max(1, env.config.max_steps),
        height / max(1, max(height, width)),
        width / max(1, max(height, width)),
        env.config.catch_radius / max_dist,
    ]
    for idx in range(spec.max_evaders):
        if idx >= len(env.evader_ids):
            values.extend([spec.padding_value] * 7)
            continue
        evader_id = env.evader_ids[idx]
        pos = env.evader_positions[evader_id]
        active = evader_id in env.active_evaders
        captured = env.per_evader_status.get(evader_id) == "captured"
        goal = env.evader_goals[evader_id]
        values.extend(
            [
                pos[0] / scale_y,
                pos[1] / scale_x,
                float(active),
                float(captured),
                manhattan(own, pos) / max_dist,
                goal[0] / scale_y,
                goal[1] / scale_x,
            ]
        )
    return np.asarray(values, dtype=np.float32)


def encode_evader_observation(
    env: MultiEvaderPursuitEnv,
    evader_id: str = "evader_0",
    spec: PursuitObservationSpec | None = None,
) -> np.ndarray:
    """Encode one evader-centric observation."""

    spec = spec or PursuitObservationSpec(role="evader")
    if spec.role != "evader":
        raise ValueError("spec role must be evader")
    if evader_id not in env.evader_positions:
        raise ValueError(f"unknown evader_id: {evader_id}")
    if env.config.num_pursuers > spec.max_pursuers:
        raise ValueError("environment has more pursuers than observation spec supports")

    height, width = env.grid_size
    scale_y, scale_x, max_dist = _scales(env)
    own = env.evader_positions[evader_id]
    goal = env.evader_goals[evader_id]
    active = evader_id in env.active_evaders
    captured = env.per_evader_status.get(evader_id) == "captured"
    values = [
        own[0] / scale_y,
        own[1] / scale_x,
        goal[0] / scale_y,
        goal[1] / scale_x,
        float(active),
        float(captured),
        env.steps / max(1, env.config.max_steps),
        height / max(1, max(height, width)),
        width / max(1, max(height, width)),
        env.config.catch_radius / max_dist,
    ]
    for idx in range(spec.max_pursuers):
        if idx >= len(env.pursuer_ids):
            values.extend([spec.padding_value] * 4)
            continue
        pursuer_id = env.pursuer_ids[idx]
        pos = env.pursuer_positions[pursuer_id]
        values.extend([pos[0] / scale_y, pos[1] / scale_x, 1.0, manhattan(own, pos) / max_dist])
    return np.asarray(values, dtype=np.float32)


def validate_pursuit_ppo_env(env: MultiEvaderPursuitEnv, spec: PursuitObservationSpec) -> None:
    """Validate the narrow env support for today's PPO pursuer path."""

    if spec.role != "pursuer":
        raise ValueError("PPO pursuer training requires a pursuer observation spec")
    if env.config.num_pursuers != 1:
        raise ValueError("PPO pursuer training currently supports exactly one pursuer")
    if env.config.num_pursuers > spec.max_pursuers:
        raise ValueError("environment has more pursuers than observation spec supports")
    if env.config.num_evaders > spec.max_evaders:
        raise ValueError("environment has more evaders than observation spec supports")


def _scales(env: MultiEvaderPursuitEnv) -> tuple[float, float, float]:
    height, width = env.grid_size
    scale_y = float(max(1, height - 1))
    scale_x = float(max(1, width - 1))
    max_dist = float(max(1, (height - 1) + (width - 1)))
    return scale_y, scale_x, max_dist
