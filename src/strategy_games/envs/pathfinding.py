"""Grid pathfinding helpers shared by v1-compatible and v2 environments."""

from __future__ import annotations

from collections import deque

from strategy_games.envs.gridworld import ACTION_TO_DELTA, Position, clip_position

Obstacles = frozenset[Position] | set[Position]


def is_free(pos: Position, grid_size: int, obstacles: Obstacles = frozenset()) -> bool:
    """Return whether ``pos`` is inside the grid and not an obstacle."""

    y, x = pos
    return 0 <= y < grid_size and 0 <= x < grid_size and pos not in obstacles


def apply_move(pos: Position, action: int, grid_size: int, obstacles: Obstacles = frozenset()) -> Position:
    """Move ``pos`` by ``action``, staying put on walls or out-of-bounds cells."""

    if action not in ACTION_TO_DELTA:
        raise ValueError(f"action must be in {sorted(ACTION_TO_DELTA)}, got {action}")
    dy, dx = ACTION_TO_DELTA[action]
    nxt = (pos[0] + dy, pos[1] + dx)
    if is_free(nxt, grid_size, obstacles):
        return nxt
    if is_free(pos, grid_size, obstacles):
        return pos
    return clip_position(pos, grid_size)


def neighbors(
    pos: Position,
    grid_size: int,
    obstacles: Obstacles = frozenset(),
    include_stay: bool = False,
) -> list[tuple[int, Position]]:
    """Return legal ``(action, next_pos)`` pairs from ``pos``."""

    actions = range(5) if include_stay else range(1, 5)
    out: list[tuple[int, Position]] = []
    for action in actions:
        nxt = apply_move(pos, action, grid_size, obstacles)
        if action == 0 or nxt != pos:
            out.append((action, nxt))
    return out


def bfs_distance(
    source: Position,
    target: Position,
    grid_size: int,
    obstacles: Obstacles = frozenset(),
) -> int | None:
    """Shortest-path length avoiding obstacles, or ``None`` if unreachable."""

    if source == target:
        return 0
    if not is_free(source, grid_size, obstacles) or not is_free(target, grid_size, obstacles):
        return None
    queue: deque[tuple[Position, int]] = deque([(source, 0)])
    seen = {source}
    while queue:
        pos, dist = queue.popleft()
        for _, nxt in neighbors(pos, grid_size, obstacles):
            if nxt in seen:
                continue
            if nxt == target:
                return dist + 1
            seen.add(nxt)
            queue.append((nxt, dist + 1))
    return None


def bfs_first_action(
    source: Position,
    target: Position,
    grid_size: int,
    obstacles: Obstacles = frozenset(),
) -> int:
    """Return the first action on a deterministic shortest path to ``target``."""

    if source == target:
        return 0
    dist = bfs_distance(source, target, grid_size, obstacles)
    if dist is None:
        return 0
    for action, nxt in neighbors(source, grid_size, obstacles):
        nxt_dist = bfs_distance(nxt, target, grid_size, obstacles)
        if nxt_dist is not None and nxt_dist + 1 == dist:
            return action
    return 0


def shortest_path_first_steps(
    source: Position,
    target: Position,
    grid_size: int,
    obstacles: Obstacles = frozenset(),
) -> list[tuple[int, Position]]:
    """Return every first step that lies on some shortest path."""

    dist = bfs_distance(source, target, grid_size, obstacles)
    if dist is None or dist == 0:
        return [(0, source)]
    steps: list[tuple[int, Position]] = []
    for action, nxt in neighbors(source, grid_size, obstacles):
        nxt_dist = bfs_distance(nxt, target, grid_size, obstacles)
        if nxt_dist is not None and nxt_dist + 1 == dist:
            steps.append((action, nxt))
    return steps or [(0, source)]


def nearest_goal(
    source: Position,
    goals: tuple[Position, ...] | list[Position],
    grid_size: int,
    obstacles: Obstacles = frozenset(),
) -> Position:
    """Return the closest reachable goal, breaking ties by list order."""

    if not goals:
        raise ValueError("goals must be non-empty")
    best = goals[0]
    best_dist = bfs_distance(source, best, grid_size, obstacles)
    for goal in goals[1:]:
        dist = bfs_distance(source, goal, grid_size, obstacles)
        if dist is None:
            continue
        if best_dist is None or dist < best_dist:
            best = goal
            best_dist = dist
    return best
