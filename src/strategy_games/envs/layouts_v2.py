"""Named v2 layouts designed so scripted strategies trade off.

The default map is a split corridor with two scoring goals. Held-out maps
are constructed so that ``evasive``, ``feint``, and ``patient`` are each the
unique best scripted attacker on at least one layout.
"""

from __future__ import annotations

from strategy_games.envs.gridworld_v2 import GridworldV2Config

# Vertical wall with gaps on rows 2 and 4, creating a short top route and a
# longer bottom route between the left start and the two right-side goals.
SPLIT_WALL: frozenset[tuple[int, int]] = frozenset(
    {
        (0, 3),
        (1, 3),
        (3, 3),
        (5, 3),
        (6, 3),
    }
)

# Single corridor: every route to either goal passes through (2, 3).
CHOKE_WALL: frozenset[tuple[int, int]] = frozenset(
    {
        (0, 3),
        (1, 3),
        (3, 3),
        (4, 3),
        (5, 3),
        (6, 3),
    }
)


def _v2(
    attacker_start: tuple[int, int],
    defender_start: tuple[int, int],
    goal_positions: tuple[tuple[int, int], ...],
    max_steps: int = 20,
    obstacles: frozenset[tuple[int, int]] = SPLIT_WALL,
    slip_prob: float = 0.0,
    catch_radius: int = 0,
    defender_period: int = 2,
) -> GridworldV2Config:
    return GridworldV2Config(
        grid_size=7,
        max_steps=max_steps,
        attacker_start=attacker_start,
        defender_start=defender_start,
        goal_positions=goal_positions,
        obstacles=obstacles,
        catch_radius=catch_radius,
        slip_prob=slip_prob,
        defender_period=defender_period,
    )


DEFAULT_V2_LAYOUT = _v2(
    attacker_start=(0, 0),
    defender_start=(3, 5),
    # Primary is the farther goal so aggressive and direct_goal diverge.
    goal_positions=((6, 6), (0, 6)),
    max_steps=20,
)

HELD_OUT_LAYOUTS_V2: dict[str, GridworldV2Config] = {
    "corridor_split": DEFAULT_V2_LAYOUT,
    "evasive_detour": _v2(
        attacker_start=(0, 0),
        defender_start=(2, 4),
        # Both goals sit at the end of the short (blocked) corridor.
        goal_positions=((0, 6), (1, 6)),
        max_steps=16,
    ),
    "feint_switch": _v2(
        attacker_start=(2, 0),
        defender_start=(0, 6),
        goal_positions=((0, 6), (6, 6)),
        max_steps=18,
    ),
    "patient_hold": _v2(
        attacker_start=(0, 0),
        defender_start=(0, 2),
        goal_positions=((0, 4), (6, 6)),
        max_steps=16,
        obstacles=frozenset(),
    ),
    "open_bottom": _v2(
        attacker_start=(6, 0),
        defender_start=(0, 5),
        goal_positions=((6, 6), (0, 6)),
        max_steps=20,
        # Equal-speed held-out map: Nash equals timeout, so exploitability has
        # spread across layouts (train is winnable; this one is not).
        defender_period=1,
    ),
}

TRAIN_LAYOUT_V2_NAME = "corridor_split"
HELD_OUT_LAYOUT_V2_NAMES = tuple(name for name in HELD_OUT_LAYOUTS_V2 if name != TRAIN_LAYOUT_V2_NAME)

# Layouts used to certify Gate 1(c): each named attacker is uniquely best.
GATE1_UNIQUE_BEST_LAYOUTS: dict[str, str] = {
    "evasive": "evasive_detour",
    "feint": "feint_switch",
    "patient": "patient_hold",
}


def get_layout_v2(name: str) -> GridworldV2Config:
    """Return a named v2 layout config."""

    if name not in HELD_OUT_LAYOUTS_V2:
        known = ", ".join(HELD_OUT_LAYOUTS_V2)
        raise KeyError(f"Unknown v2 layout {name!r}; choose from {known}")
    return HELD_OUT_LAYOUTS_V2[name]


def layout_v2_names() -> tuple[str, ...]:
    """Return v2 layout names in canonical report order."""

    return tuple(HELD_OUT_LAYOUTS_V2)
