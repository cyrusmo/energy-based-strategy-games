from strategy_games.envs.gridworld_v2 import AttackerDefenderGridworldV2, GridworldV2Config
from strategy_games.envs.layouts_v2 import SPLIT_WALL, get_layout_v2
from strategy_games.envs.pathfinding import bfs_distance, bfs_first_action


def test_v2_reset_observation_and_walls() -> None:
    env = AttackerDefenderGridworldV2(get_layout_v2("corridor_split"))
    obs = env.reset()
    assert obs.shape == (env.state_dim,)
    rendered = env.render()
    assert "A" in rendered
    assert "D" in rendered
    assert "G" in rendered
    assert "#" in rendered
    assert env.step(attacker_action=4, defender_action=0).observation.shape == (env.state_dim,)


def test_v2_blocks_moves_into_walls() -> None:
    config = GridworldV2Config(
        grid_size=7,
        attacker_start=(0, 2),
        defender_start=(6, 6),
        goal_positions=((0, 6), (6, 0)),
        obstacles=SPLIT_WALL,
        max_steps=5,
    )
    env = AttackerDefenderGridworldV2(config)
    env.reset()
    result = env.step(attacker_action=4, defender_action=0)
    assert env.attacker_pos == (0, 2)
    assert result.info["outcome"] == "running"


def test_v2_scores_on_either_goal() -> None:
    config = GridworldV2Config(
        grid_size=5,
        max_steps=5,
        attacker_start=(0, 0),
        defender_start=(4, 4),
        goal_positions=((0, 1), (4, 0)),
        obstacles=frozenset(),
    )
    env = AttackerDefenderGridworldV2(config)
    env.reset()
    result = env.step(attacker_action=4, defender_action=0)
    assert result.done
    assert result.info["outcome"] == "goal"


def test_bfs_paths_around_split_wall() -> None:
    start = (0, 0)
    goal = (0, 6)
    dist = bfs_distance(start, goal, 7, SPLIT_WALL)
    assert dist is not None
    assert dist < 20
    action = bfs_first_action(start, goal, 7, SPLIT_WALL)
    assert action in {1, 2, 3, 4}
