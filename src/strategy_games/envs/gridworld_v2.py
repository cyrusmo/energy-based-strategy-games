"""Attacker-defender gridworld v2 with walls, two goals, and optional slip.

v1 stays untouched for snapshot reproduction. This environment is the paper
testbed: distinct routes, decoy-capable goals, and a shared latent strategy
space for both players.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from strategy_games.envs.gridworld import (
    ACTION_NAMES,
    ACTION_TO_DELTA,
    Action,
    Position,
    StepResult,
    manhattan,
)
from strategy_games.envs.pathfinding import apply_move, bfs_distance, is_free, nearest_goal


@dataclass(frozen=True)
class GridworldV2Config:
    """Configuration for the v2 pursuit/evasion game."""

    grid_size: int = 7
    max_steps: int = 20
    attacker_start: Position = (0, 0)
    defender_start: Position = (3, 5)
    goal_positions: tuple[Position, ...] = ((0, 6), (6, 6))
    obstacles: frozenset[Position] = field(default_factory=frozenset)
    catch_radius: int = 0
    step_penalty: float = -0.01
    goal_reward: float = 1.0
    catch_reward: float = -1.0
    timeout_reward: float = -0.2
    slip_prob: float = 0.0
    seed: int | None = None

    @property
    def goal_pos(self) -> Position:
        """Primary (first) goal, for v1-compatible call sites."""

        if not self.goal_positions:
            raise ValueError("goal_positions must be non-empty")
        return self.goal_positions[0]

    @property
    def decoy_goal(self) -> Position:
        """Decoy or secondary goal; equals the primary goal if only one exists."""

        if len(self.goal_positions) > 1:
            return self.goal_positions[-1]
        return self.goal_positions[0]


def gridworld_v2_config_from_mapping(raw: dict[str, Any]) -> GridworldV2Config:
    """Build a v2 config from a YAML-style mapping."""

    goals_raw = raw.get("goal_positions", raw.get("goals"))
    if goals_raw is None and "goal_pos" in raw:
        goals = (tuple(raw["goal_pos"]),)
    elif goals_raw is None:
        goals = ((0, 6), (6, 6))
    else:
        goals = tuple(tuple(item) for item in goals_raw)
    obstacles_raw = raw.get("obstacles", ())
    return GridworldV2Config(
        grid_size=int(raw.get("grid_size", 7)),
        max_steps=int(raw.get("max_steps", 20)),
        attacker_start=tuple(raw.get("attacker_start", (0, 0))),  # type: ignore[arg-type]
        defender_start=tuple(raw.get("defender_start", (3, 5))),  # type: ignore[arg-type]
        goal_positions=goals,  # type: ignore[arg-type]
        obstacles=frozenset(tuple(item) for item in obstacles_raw),  # type: ignore[misc]
        catch_radius=int(raw.get("catch_radius", 0)),
        step_penalty=float(raw.get("step_penalty", -0.01)),
        goal_reward=float(raw.get("goal_reward", 1.0)),
        catch_reward=float(raw.get("catch_reward", -1.0)),
        timeout_reward=float(raw.get("timeout_reward", -0.2)),
        slip_prob=float(raw.get("slip_prob", 0.0)),
        seed=None if raw.get("seed") is None else int(raw["seed"]),
    )


class AttackerDefenderGridworldV2:
    """Two-goal gridworld with obstacles and optional action slip."""

    action_dim = len(ACTION_TO_DELTA)

    def __init__(self, config: GridworldV2Config | None = None) -> None:
        self.config = config or GridworldV2Config()
        self._validate_config()
        self.attacker_pos: Position = self.config.attacker_start
        self.defender_pos: Position = self.config.defender_start
        self.goal_positions: tuple[Position, ...] = self.config.goal_positions
        self.obstacles: frozenset[Position] = self.config.obstacles
        self.steps = 0
        self.rng = np.random.default_rng(self.config.seed)

    @property
    def state_dim(self) -> int:
        """Dimension of the vector observation."""

        return 11

    @property
    def goal_pos(self) -> Position:
        """Primary goal position."""

        return self.goal_positions[0]

    @goal_pos.setter
    def goal_pos(self, value: Position) -> None:
        goals = list(self.goal_positions) or [value]
        goals[0] = value
        self.goal_positions = tuple(goals)

    def reset(
        self,
        attacker_pos: Position | None = None,
        defender_pos: Position | None = None,
        goal_pos: Position | None = None,
        seed: int | None = None,
    ) -> np.ndarray:
        """Reset positions and return the initial observation."""

        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.attacker_pos = attacker_pos or self.config.attacker_start
        self.defender_pos = defender_pos or self.config.defender_start
        if goal_pos is not None:
            self.goal_pos = goal_pos
        else:
            self.goal_positions = self.config.goal_positions
        self.steps = 0
        self._validate_position(self.attacker_pos, "attacker_pos")
        self._validate_position(self.defender_pos, "defender_pos")
        for idx, goal in enumerate(self.goal_positions):
            self._validate_position(goal, f"goal_positions[{idx}]")
        return self.observe()

    def observe(self) -> np.ndarray:
        """Return a normalized vector observation with both goals."""

        scale = max(1, self.config.grid_size - 1)
        max_manhattan = max(1, 2 * (self.config.grid_size - 1))
        goal_a = self.goal_positions[0]
        goal_b = self.goal_positions[-1]
        nearest = nearest_goal(self.attacker_pos, self.goal_positions, self.config.grid_size, self.obstacles)
        goal_dist = bfs_distance(self.attacker_pos, nearest, self.config.grid_size, self.obstacles)
        if goal_dist is None:
            goal_norm = 1.0
        else:
            goal_norm = min(1.0, goal_dist / max_manhattan)
        defender_dist = manhattan(self.defender_pos, self.attacker_pos) / max_manhattan
        return np.array(
            [
                self.attacker_pos[0] / scale,
                self.attacker_pos[1] / scale,
                self.defender_pos[0] / scale,
                self.defender_pos[1] / scale,
                goal_a[0] / scale,
                goal_a[1] / scale,
                goal_b[0] / scale,
                goal_b[1] / scale,
                self.steps / max(1, self.config.max_steps),
                goal_norm,
                defender_dist,
            ],
            dtype=np.float32,
        )

    def set_state_from_observation(self, observation: np.ndarray) -> None:
        """Reconstruct discrete positions from a normalized observation."""

        scale = max(1, self.config.grid_size - 1)
        self.attacker_pos = (round(float(observation[0]) * scale), round(float(observation[1]) * scale))
        self.defender_pos = (round(float(observation[2]) * scale), round(float(observation[3]) * scale))
        primary = (round(float(observation[4]) * scale), round(float(observation[5]) * scale))
        decoy = (round(float(observation[6]) * scale), round(float(observation[7]) * scale))
        if len(self.goal_positions) > 1:
            self.goal_positions = (primary, *self.goal_positions[1:-1], decoy)
        else:
            self.goal_positions = (primary,)
        self.steps = round(float(observation[8]) * max(1, self.config.max_steps))

    def step(self, attacker_action: Action, defender_action: Action | None = None) -> StepResult:
        """Advance one simultaneous-move step."""

        if attacker_action not in ACTION_TO_DELTA:
            raise ValueError(f"attacker_action must be in {sorted(ACTION_TO_DELTA)}, got {attacker_action}")
        if defender_action is None:
            defender_action = self._default_defender_action()
        if defender_action not in ACTION_TO_DELTA:
            raise ValueError(f"defender_action must be in {sorted(ACTION_TO_DELTA)}, got {defender_action}")

        self.attacker_pos = self._stochastic_move(self.attacker_pos, attacker_action)
        self.defender_pos = self._stochastic_move(self.defender_pos, defender_action)
        self.steps += 1

        reward, done, outcome, info = resolve_v2_outcome(
            self.attacker_pos,
            self.defender_pos,
            self.steps,
            self.config,
            attacker_action=attacker_action,
            defender_action=defender_action,
        )
        return StepResult(self.observe(), float(reward), done, info)

    def render(self, mode: Literal["text", "array"] = "text") -> str | np.ndarray:
        """Render the grid as text or as an integer array.

        Legend: empty=0, primary goal=1, attacker=2, defender=3, overlap=4,
        obstacle=5, decoy goal=6.
        """

        grid = np.zeros((self.config.grid_size, self.config.grid_size), dtype=np.int8)
        for obstacle in self.obstacles:
            grid[obstacle] = 5
        for idx, goal in enumerate(self.goal_positions):
            grid[goal] = 1 if idx == 0 else 6
        grid[self.attacker_pos] = 2
        grid[self.defender_pos] = 3
        if self.attacker_pos == self.defender_pos:
            grid[self.attacker_pos] = 4

        if mode == "array":
            return grid
        if mode != "text":
            raise ValueError(f"Unsupported render mode: {mode}")

        symbols = {0: ".", 1: "G", 2: "A", 3: "D", 4: "X", 5: "#", 6: "g"}
        rows = [" ".join(symbols[int(cell)] for cell in row) for row in grid]
        return "\n".join(rows)

    def clone(self) -> "AttackerDefenderGridworldV2":
        """Return a copy with the same configuration and current state."""

        env = AttackerDefenderGridworldV2(self.config)
        env.attacker_pos = self.attacker_pos
        env.defender_pos = self.defender_pos
        env.goal_positions = self.goal_positions
        env.obstacles = self.obstacles
        env.steps = self.steps
        env.rng = np.random.default_rng(self.config.seed)
        return env

    def _stochastic_move(self, pos: Position, action: Action) -> Position:
        if self.config.slip_prob > 0.0 and float(self.rng.random()) < self.config.slip_prob:
            action = int(self.rng.integers(0, self.action_dim))
        return apply_move(pos, action, self.config.grid_size, self.obstacles)

    def _default_defender_action(self) -> Action:
        from strategy_games.envs.pathfinding import bfs_first_action

        if manhattan(self.defender_pos, self.attacker_pos) <= 1:
            return 0
        return bfs_first_action(
            self.defender_pos, self.attacker_pos, self.config.grid_size, self.obstacles
        )

    def _validate_config(self) -> None:
        if self.config.grid_size < 3:
            raise ValueError("grid_size must be at least 3")
        if self.config.max_steps < 1:
            raise ValueError("max_steps must be positive")
        if not self.config.goal_positions:
            raise ValueError("goal_positions must be non-empty")
        if not 0.0 <= self.config.slip_prob < 1.0:
            raise ValueError("slip_prob must be in [0, 1)")
        for name, pos in (
            ("attacker_start", self.config.attacker_start),
            ("defender_start", self.config.defender_start),
        ):
            self._validate_position(pos, name)
        for idx, goal in enumerate(self.config.goal_positions):
            self._validate_position(goal, f"goal_positions[{idx}]")
        for obstacle in self.config.obstacles:
            y, x = obstacle
            if not (0 <= y < self.config.grid_size and 0 <= x < self.config.grid_size):
                raise ValueError(f"obstacle {obstacle} is outside the grid")

    def _validate_position(self, pos: Position, name: str) -> None:
        if not is_free(pos, self.config.grid_size, self.config.obstacles):
            raise ValueError(f"{name}={pos} is blocked or outside the grid")


def resolve_v2_outcome(
    attacker_pos: Position,
    defender_pos: Position,
    steps: int,
    config: GridworldV2Config,
    attacker_action: int | None = None,
    defender_action: int | None = None,
) -> tuple[float, bool, str, dict[str, object]]:
    """Return reward, termination, outcome, and info for a resolved state."""

    caught = manhattan(attacker_pos, defender_pos) <= config.catch_radius
    reached_goal = attacker_pos in config.goal_positions
    timed_out = steps >= config.max_steps
    reward = config.step_penalty
    done = False
    outcome = "running"
    if caught:
        reward += config.catch_reward
        done = True
        outcome = "caught"
    elif reached_goal:
        reward += config.goal_reward
        done = True
        outcome = "goal"
    elif timed_out:
        reward += config.timeout_reward
        done = True
        outcome = "timeout"
    info: dict[str, object] = {
        "outcome": outcome,
        "attacker_pos": attacker_pos,
        "defender_pos": defender_pos,
        "goal_pos": config.goal_pos,
        "goal_positions": config.goal_positions,
        "steps": steps,
        "caught": caught,
        "goal_reached": reached_goal,
        "timed_out": timed_out,
        "attacker_action": ACTION_NAMES.get(attacker_action, "unknown") if attacker_action is not None else None,
        "defender_action": ACTION_NAMES.get(defender_action, "unknown") if defender_action is not None else None,
    }
    return float(reward), done, outcome, info


def gridworld_v2_potential(env: AttackerDefenderGridworldV2) -> float:
    """Potential for dense progress shaping; higher is closer to a goal."""

    nearest = nearest_goal(env.attacker_pos, env.goal_positions, env.config.grid_size, env.obstacles)
    dist = bfs_distance(env.attacker_pos, nearest, env.config.grid_size, env.obstacles)
    max_dist = max(1, env.config.grid_size * env.config.grid_size)
    if dist is None:
        return -1.0
    return -float(dist / max_dist)


def env_config_from_mapping(raw: dict[str, Any]):
    """Parse either a v1 or v2 env mapping."""

    from strategy_games.envs.gridworld import GridworldConfig

    version = str(raw.get("version", "v1")).lower()
    v2_keys = {"goal_positions", "obstacles", "slip_prob", "goals"}
    if version in {"v2", "gridworld_v2"} or v2_keys.intersection(raw):
        return gridworld_v2_config_from_mapping(raw)
    return GridworldConfig(
        grid_size=int(raw.get("grid_size", 10)),
        max_steps=int(raw.get("max_steps", 50)),
        attacker_start=tuple(raw.get("attacker_start", (0, 0))),  # type: ignore[arg-type]
        defender_start=tuple(raw.get("defender_start", (9, 9))),  # type: ignore[arg-type]
        goal_pos=tuple(raw.get("goal_pos", (9, 0))),  # type: ignore[arg-type]
        catch_radius=int(raw.get("catch_radius", 0)),
        step_penalty=float(raw.get("step_penalty", -0.01)),
        goal_reward=float(raw.get("goal_reward", 1.0)),
        catch_reward=float(raw.get("catch_reward", -1.0)),
        timeout_reward=float(raw.get("timeout_reward", -0.2)),
    )


def make_gridworld(config: GridworldV2Config | Any):
    """Construct a v1 or v2 env from its config dataclass."""

    from strategy_games.envs.gridworld import AttackerDefenderGridworld, GridworldConfig

    if isinstance(config, GridworldV2Config):
        return AttackerDefenderGridworldV2(config)
    if isinstance(config, GridworldConfig):
        return AttackerDefenderGridworld(config)
    raise TypeError(f"Unsupported env config type: {type(config)!r}")


def is_v2_env(env: Any) -> bool:
    """Return whether ``env`` is a v2 gridworld."""

    return isinstance(env, AttackerDefenderGridworldV2)
