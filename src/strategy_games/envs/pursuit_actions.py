"""Shared action and agent-id conventions for pursuit/evasion demos."""

from __future__ import annotations

from typing import TypeAlias

Position: TypeAlias = tuple[int, int]

ACTIONS = ["stay", "up", "down", "left", "right"]
ACTION_DELTAS: dict[str, Position] = {
    "stay": (0, 0),
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}


def pursuer_id(index: int) -> str:
    """Return the stable pursuer id for ``index``."""

    if index < 0:
        raise ValueError("index must be non-negative")
    return f"pursuer_{index}"


def evader_id(index: int) -> str:
    """Return the stable evader id for ``index``."""

    if index < 0:
        raise ValueError("index must be non-negative")
    return f"evader_{index}"


def pursuer_ids(num_pursuers: int) -> list[str]:
    """Return stable pursuer ids."""

    if num_pursuers < 1:
        raise ValueError("num_pursuers must be positive")
    return [pursuer_id(index) for index in range(num_pursuers)]


def evader_ids(num_evaders: int) -> list[str]:
    """Return stable evader ids."""

    if num_evaders < 1:
        raise ValueError("num_evaders must be positive")
    return [evader_id(index) for index in range(num_evaders)]


def all_agent_ids(num_pursuers: int, num_evaders: int) -> list[str]:
    """Return pursuer ids followed by evader ids."""

    return [*pursuer_ids(num_pursuers), *evader_ids(num_evaders)]


def is_pursuer(agent_id: str) -> bool:
    """Return whether an agent id follows the pursuer convention."""

    return agent_id.startswith("pursuer_")


def is_evader(agent_id: str) -> bool:
    """Return whether an agent id follows the evader convention."""

    return agent_id.startswith("evader_")


def clip_position(position: Position, grid_size: tuple[int, int]) -> Position:
    """Clip a position to ``grid_size`` bounds."""

    height, width = grid_size
    y, x = position
    return (min(max(int(y), 0), height - 1), min(max(int(x), 0), width - 1))


def move_position(position: Position, action: str, grid_size: tuple[int, int]) -> Position:
    """Move a position by an action label and clip it to bounds."""

    if action not in ACTION_DELTAS:
        raise ValueError(f"invalid action: {action}")
    dy, dx = ACTION_DELTAS[action]
    return clip_position((position[0] + dy, position[1] + dx), grid_size)


def manhattan(a: Position, b: Position) -> int:
    """Return Manhattan distance between two grid positions."""

    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def greedy_action_towards(source: Position, target: Position, grid_size: tuple[int, int]) -> str:
    """Return a deterministic action that greedily reduces Manhattan distance."""

    best_action = "stay"
    best_distance = manhattan(source, target)
    for action in ACTIONS:
        candidate = move_position(source, action, grid_size)
        distance = manhattan(candidate, target)
        if distance < best_distance:
            best_action = action
            best_distance = distance
    return best_action


def greedy_action_away_from(source: Position, threat: Position, grid_size: tuple[int, int]) -> str:
    """Return a deterministic action that maximizes distance from ``threat``."""

    best_action = "stay"
    best_distance = manhattan(source, threat)
    for action in ACTIONS:
        candidate = move_position(source, action, grid_size)
        distance = manhattan(candidate, threat)
        if distance > best_distance:
            best_action = action
            best_distance = distance
    return best_action
