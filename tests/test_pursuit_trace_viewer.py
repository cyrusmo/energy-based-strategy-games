from strategy_games.rollouts import run_scripted_pursuit_rollout
from strategy_games.viewers.pursuit_trace import (
    active_config_from_trace,
    default_draft_config,
    rollout_config_from_draft,
)


def test_draft_config_conversion_does_not_mutate_draft() -> None:
    draft = default_draft_config()
    draft["seed"] = 13
    draft["grid_size"] = [7, 8]

    config = rollout_config_from_draft(draft)

    assert config.seed == 13
    assert config.env.grid_size == (7, 8)
    assert draft["grid_size"] == [7, 8]


def test_active_config_from_trace_records_trace_metadata() -> None:
    trace = run_scripted_pursuit_rollout(rollout_config_from_draft(default_draft_config()))
    active = active_config_from_trace(trace)

    assert active["source"] == "loaded_trace"
    assert active["env_id"] == trace.env_id
    assert active["episode_id"] == trace.episode_id
    assert active["seed"] == trace.seed
    assert active["grid_size"] == trace.grid_size
    assert active["num_evaders"] == trace.num_evaders
    assert active["num_pursuers"] == trace.num_pursuers
