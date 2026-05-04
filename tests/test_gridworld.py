import numpy as np
import pytest

from strategy_games.envs.gridworld import AttackerDefenderGridworld, GridworldConfig


def test_reset_observation_shape_and_render() -> None:
    env = AttackerDefenderGridworld()
    obs = env.reset()
    assert obs.shape == (env.state_dim,)
    assert obs.dtype == np.float32
    rendered = env.render()
    assert "A" in rendered
    assert "D" in rendered
    assert "G" in rendered


def test_attacker_can_reach_goal() -> None:
    config = GridworldConfig(
        grid_size=5,
        max_steps=5,
        attacker_start=(0, 0),
        defender_start=(4, 4),
        goal_pos=(0, 2),
    )
    env = AttackerDefenderGridworld(config)
    env.reset()
    first = env.step(attacker_action=4, defender_action=0)
    assert not first.done
    second = env.step(attacker_action=4, defender_action=0)
    assert second.done
    assert second.info["outcome"] == "goal"
    assert second.reward > 0


def test_defender_can_catch_attacker() -> None:
    config = GridworldConfig(
        grid_size=5,
        max_steps=5,
        attacker_start=(0, 0),
        defender_start=(0, 1),
        goal_pos=(4, 4),
    )
    env = AttackerDefenderGridworld(config)
    env.reset()
    result = env.step(attacker_action=0, defender_action=3)
    assert result.done
    assert result.info["outcome"] == "caught"
    assert result.reward < 0


def test_invalid_action_raises() -> None:
    env = AttackerDefenderGridworld()
    env.reset()
    with pytest.raises(ValueError):
        env.step(attacker_action=99)
