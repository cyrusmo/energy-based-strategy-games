"""Named gridworld layouts, including held-out maps that expose heuristic overfitting.

The default attacker-defender geometry lets a direct-to-goal path miss the greedy
defender. Held-out layouts change starts, goals, or defender placement so that
the same heuristic is no longer a free win.
"""

from __future__ import annotations

from strategy_games.envs.gridworld import GridworldConfig

DEFAULT_LAYOUT = GridworldConfig(
    grid_size=10,
    max_steps=50,
    attacker_start=(0, 0),
    defender_start=(9, 9),
    goal_pos=(9, 0),
    catch_radius=0,
)

HELD_OUT_LAYOUTS: dict[str, GridworldConfig] = {
    "default": DEFAULT_LAYOUT,
    "open_right": GridworldConfig(
        grid_size=10,
        max_steps=50,
        attacker_start=(0, 0),
        defender_start=(9, 0),
        goal_pos=(0, 9),
        catch_radius=0,
    ),
    "intercept_path": GridworldConfig(
        grid_size=10,
        max_steps=50,
        attacker_start=(0, 0),
        defender_start=(4, 0),
        goal_pos=(9, 0),
        catch_radius=0,
    ),
    "goal_near_defender": GridworldConfig(
        grid_size=10,
        max_steps=50,
        attacker_start=(0, 0),
        defender_start=(9, 0),
        goal_pos=(9, 1),
        catch_radius=0,
    ),
    "cramped": GridworldConfig(
        grid_size=6,
        max_steps=30,
        attacker_start=(0, 0),
        defender_start=(2, 2),
        goal_pos=(5, 0),
        catch_radius=0,
    ),
}

TRAIN_LAYOUT_NAME = "default"
HELD_OUT_LAYOUT_NAMES = tuple(name for name in HELD_OUT_LAYOUTS if name != TRAIN_LAYOUT_NAME)


def get_layout(name: str) -> GridworldConfig:
    """Return a named layout config."""

    if name not in HELD_OUT_LAYOUTS:
        known = ", ".join(HELD_OUT_LAYOUTS)
        raise KeyError(f"Unknown layout {name!r}; choose from {known}")
    return HELD_OUT_LAYOUTS[name]


def layout_names() -> tuple[str, ...]:
    """Return layout names in canonical report order."""

    return tuple(HELD_OUT_LAYOUTS)
