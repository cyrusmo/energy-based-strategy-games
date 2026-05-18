import math

from strategy_games.envs.multi_evader_pursuit import MultiEvaderPursuitConfig, MultiEvaderPursuitEnv


def test_multi_evader_reset_uses_stable_agent_ids() -> None:
    env = MultiEvaderPursuitEnv(MultiEvaderPursuitConfig(num_evaders=3, num_pursuers=2))
    positions = env.reset()
    assert list(env.pursuer_ids) == ["pursuer_0", "pursuer_1"]
    assert list(env.evader_ids) == ["evader_0", "evader_1", "evader_2"]
    assert set(positions) == {"pursuer_0", "pursuer_1", "evader_0", "evader_1", "evader_2"}


def test_boundary_clipping() -> None:
    env = MultiEvaderPursuitEnv(
        MultiEvaderPursuitConfig(
            grid_size=(3, 3),
            num_evaders=1,
            max_steps=2,
            pursuer_starts=((0, 0),),
            evader_starts=((2, 2),),
        )
    )
    result = env.step({"pursuer_0": "up", "evader_0": "right"})
    assert result.agent_positions["pursuer_0"] == (0, 0)
    assert result.agent_positions["evader_0"] == (2, 2)


def test_capture_is_after_movement_only() -> None:
    env = MultiEvaderPursuitEnv(
        MultiEvaderPursuitConfig(
            grid_size=(9, 9),
            num_evaders=1,
            max_steps=5,
            pursuer_starts=((4, 4),),
            evader_starts=((4, 6),),
            evader_goals=((4, 6),),
        )
    )
    first = env.step({"pursuer_0": "right", "evader_0": "stay"})
    assert first.captures == []
    assert first.active_evaders == ["evader_0"]

    second = env.step({"pursuer_0": "right", "evader_0": "stay"})
    assert second.done
    assert second.terminated_reason == "all_evaders_captured"
    assert second.captures[0].evader_id == "evader_0"
    assert second.captures[0].t == 1
    assert math.isclose(second.rewards["evader_0"], -10.0)
    assert math.isclose(second.rewards["pursuer_0"], 14.9)


def test_crossing_paths_do_not_count_as_capture() -> None:
    env = MultiEvaderPursuitEnv(
        MultiEvaderPursuitConfig(
            grid_size=(3, 3),
            num_evaders=1,
            max_steps=3,
            pursuer_starts=((1, 1),),
            evader_starts=((1, 2),),
        )
    )
    result = env.step({"pursuer_0": "right", "evader_0": "left"})
    assert result.agent_positions["pursuer_0"] == (1, 2)
    assert result.agent_positions["evader_0"] == (1, 1)
    assert result.captures == []
    assert not result.done


def test_timeout_reward_convention() -> None:
    env = MultiEvaderPursuitEnv(
        MultiEvaderPursuitConfig(
            grid_size=(5, 5),
            num_evaders=1,
            max_steps=1,
            pursuer_starts=((0, 0),),
            evader_starts=((4, 4),),
        )
    )
    result = env.step({"pursuer_0": "stay", "evader_0": "stay"})
    assert result.done
    assert result.terminated_reason == "timeout"
    assert math.isclose(result.rewards["evader_0"], 11.0)
    assert math.isclose(result.rewards["pursuer_0"], -0.1)
