from strategy_games.envs.gridworld_v2 import GridworldV2Config
from strategy_games.training.psro import PSROConfig, run_psro


def test_psro_lite_returns_meta_nash() -> None:
    config = GridworldV2Config(
        grid_size=3,
        max_steps=4,
        attacker_start=(0, 0),
        defender_start=(2, 2),
        goal_positions=((0, 2), (2, 0)),
        obstacles=frozenset(),
    )
    result = run_psro(PSROConfig(env=config, iterations=1, episodes_per_opponent=1))
    assert result["method"] == "psro_lite"
    assert result["env_steps"] > 0
    assert abs(sum(result["attacker_mix"].values()) - 1.0) < 1e-6
    assert result["support"]
