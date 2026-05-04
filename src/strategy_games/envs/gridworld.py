"""Simple attacker-defender gridworld.

The environment is intentionally small and deterministic so that strategy
generation and evaluation code can be debugged before moving to richer domains.
Rewards are reported from the attacker's perspective.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

import numpy as np

Position: TypeAlias = tuple[int, int]
Action: TypeAlias = int

ACTION_TO_DELTA: dict[Action, Position] = {
    0: (0, 0),   # stay
    1: (-1, 0),  # up
    2: (1, 0),   # down
    3: (0, -1),  # left
    4: (0, 1),   # right
}

ACTION_NAMES = {
    0: "stay",
    1: "up",
    2: "down",
    3: "left",
    4: "right",
}


@dataclass(frozen=True)
class GridworldConfig:
    """Configuration for the attacker-defender pursuit/evasion game."""

    grid_size: int = 10
    max_steps: int = 50
    attacker_start: Position = (0, 0)
    defender_start: Position = (9, 9)
    goal_pos: Position = (9, 0)
    catch_radius: int = 0
    step_penalty: float = -0.01
    goal_reward: float = 1.0
    catch_reward: float = -1.0
    timeout_reward: float = -0.2


@dataclass(frozen=True)
class StepResult:
    """Container returned by :meth:`AttackerDefenderGridworld.step`."""

    observation: np.ndarray
    reward: float
    done: bool
    info: dict[str, object]


class AttackerDefenderGridworld:
    """A small two-agent gridworld with attacker and defender positions.

    The attacker tries to reach a fixed goal. The defender tries to occupy the
    attacker's cell, or come within ``catch_radius`` Manhattan distance.
    """

    action_dim = len(ACTION_TO_DELTA)

    def __init__(self, config: GridworldConfig | None = None) -> None:
        self.config = config or GridworldConfig()
        self._validate_config()
        self.attacker_pos: Position = self.config.attacker_start
        self.defender_pos: Position = self.config.defender_start
        self.goal_pos: Position = self.config.goal_pos
        self.steps = 0

    @property
    def state_dim(self) -> int:
        """Dimension of the vector observation returned by the environment."""

        return 9

    def reset(
        self,
        attacker_pos: Position | None = None,
        defender_pos: Position | None = None,
        goal_pos: Position | None = None,
    ) -> np.ndarray:
        """Reset positions and return the initial observation."""

        self.attacker_pos = attacker_pos or self.config.attacker_start
        self.defender_pos = defender_pos or self.config.defender_start
        self.goal_pos = goal_pos or self.config.goal_pos
        self.steps = 0
        self._validate_position(self.attacker_pos, "attacker_pos")
        self._validate_position(self.defender_pos, "defender_pos")
        self._validate_position(self.goal_pos, "goal_pos")
        return self.observe()

    def observe(self) -> np.ndarray:
        """Return a normalized vector observation."""

        scale = max(1, self.config.grid_size - 1)
        max_manhattan = max(1, 2 * (self.config.grid_size - 1))
        attacker_goal = manhattan(self.attacker_pos, self.goal_pos) / max_manhattan
        defender_attacker = manhattan(self.defender_pos, self.attacker_pos) / max_manhattan
        return np.array(
            [
                self.attacker_pos[0] / scale,
                self.attacker_pos[1] / scale,
                self.defender_pos[0] / scale,
                self.defender_pos[1] / scale,
                self.goal_pos[0] / scale,
                self.goal_pos[1] / scale,
                self.steps / max(1, self.config.max_steps),
                attacker_goal,
                defender_attacker,
            ],
            dtype=np.float32,
        )

    def step(self, attacker_action: Action, defender_action: Action | None = None) -> StepResult:
        """Advance the game by one simultaneous-move step.

        If ``defender_action`` is omitted, the defender greedily moves toward the
        attacker. This keeps examples short while still allowing explicit
        adversarial policies in evaluators.
        """

        self._validate_action(attacker_action, "attacker_action")
        if defender_action is None:
            defender_action = greedy_action_towards(self.defender_pos, self.attacker_pos)
        self._validate_action(defender_action, "defender_action")

        self.attacker_pos = self._move(self.attacker_pos, attacker_action)
        self.defender_pos = self._move(self.defender_pos, defender_action)
        self.steps += 1

        caught = manhattan(self.attacker_pos, self.defender_pos) <= self.config.catch_radius
        reached_goal = self.attacker_pos == self.goal_pos
        timed_out = self.steps >= self.config.max_steps

        reward = self.config.step_penalty
        done = False
        outcome = "running"

        if caught:
            reward += self.config.catch_reward
            done = True
            outcome = "caught"
        elif reached_goal:
            reward += self.config.goal_reward
            done = True
            outcome = "goal"
        elif timed_out:
            reward += self.config.timeout_reward
            done = True
            outcome = "timeout"

        info: dict[str, object] = {
            "outcome": outcome,
            "attacker_pos": self.attacker_pos,
            "defender_pos": self.defender_pos,
            "goal_pos": self.goal_pos,
            "steps": self.steps,
            "caught": caught,
            "goal_reached": reached_goal,
            "timed_out": timed_out,
            "attacker_action": ACTION_NAMES[attacker_action],
            "defender_action": ACTION_NAMES[defender_action],
        }
        return StepResult(self.observe(), float(reward), done, info)

    def render(self, mode: Literal["text", "array"] = "text") -> str | np.ndarray:
        """Render the grid as text or as an integer array.

        Integer array legend: empty=0, goal=1, attacker=2, defender=3, overlap=4.
        """

        grid = np.zeros((self.config.grid_size, self.config.grid_size), dtype=np.int8)
        grid[self.goal_pos] = 1
        grid[self.attacker_pos] = 2
        grid[self.defender_pos] = 3
        if self.attacker_pos == self.defender_pos:
            grid[self.attacker_pos] = 4

        if mode == "array":
            return grid
        if mode != "text":
            raise ValueError(f"Unsupported render mode: {mode}")

        symbols = {0: ".", 1: "G", 2: "A", 3: "D", 4: "X"}
        rows = [" ".join(symbols[int(cell)] for cell in row) for row in grid]
        return "\n".join(rows)

    def clone(self) -> "AttackerDefenderGridworld":
        """Return a copy with the same configuration and current state."""

        env = AttackerDefenderGridworld(self.config)
        env.attacker_pos = self.attacker_pos
        env.defender_pos = self.defender_pos
        env.goal_pos = self.goal_pos
        env.steps = self.steps
        return env

    def _move(self, pos: Position, action: Action) -> Position:
        dy, dx = ACTION_TO_DELTA[action]
        next_pos = (pos[0] + dy, pos[1] + dx)
        return clip_position(next_pos, self.config.grid_size)

    def _validate_config(self) -> None:
        if self.config.grid_size < 2:
            raise ValueError("grid_size must be at least 2")
        if self.config.max_steps < 1:
            raise ValueError("max_steps must be positive")
        for name, pos in (
            ("attacker_start", self.config.attacker_start),
            ("defender_start", self.config.defender_start),
            ("goal_pos", self.config.goal_pos),
        ):
            self._validate_position(pos, name)

    def _validate_position(self, pos: Position, name: str) -> None:
        y, x = pos
        if not (0 <= y < self.config.grid_size and 0 <= x < self.config.grid_size):
            raise ValueError(f"{name}={pos} is outside a {self.config.grid_size}x{self.config.grid_size} grid")

    @staticmethod
    def _validate_action(action: Action, name: str) -> None:
        if action not in ACTION_TO_DELTA:
            raise ValueError(f"{name} must be in {sorted(ACTION_TO_DELTA)}, got {action}")


def clip_position(pos: Position, grid_size: int) -> Position:
    """Clip a grid position to valid bounds."""

    return (
        int(np.clip(pos[0], 0, grid_size - 1)),
        int(np.clip(pos[1], 0, grid_size - 1)),
    )


def manhattan(a: Position, b: Position) -> int:
    """Return Manhattan distance between two grid positions."""

    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def greedy_action_towards(source: Position, target: Position) -> Action:
    """Return an action that greedily reduces Manhattan distance to ``target``."""

    sy, sx = source
    ty, tx = target
    if sy < ty:
        return 2
    if sy > ty:
        return 1
    if sx < tx:
        return 4
    if sx > tx:
        return 3
    return 0


def greedy_action_away_from(source: Position, threat: Position, grid_size: int) -> Action:
    """Return an action that maximizes distance from ``threat`` after one step."""

    best_action = 0
    best_distance = manhattan(source, threat)
    for action in ACTION_TO_DELTA:
        candidate = clip_position((source[0] + ACTION_TO_DELTA[action][0], source[1] + ACTION_TO_DELTA[action][1]), grid_size)
        distance = manhattan(candidate, threat)
        if distance > best_distance:
            best_action = action
            best_distance = distance
    return best_action
