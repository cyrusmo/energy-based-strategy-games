import copy
import math
from pathlib import Path

import pytest

from strategy_games.traces import (
    SCHEMA_VERSION,
    TRACE_TYPE,
    load_pursuit_trace,
    pursuit_trace_from_dict,
    pursuit_trace_to_dict,
    validate_pursuit_trace,
)
from strategy_games.viewers import agent_status_rows, grid_frame, summary_metrics, trace_table_rows


FIXTURE_PATH = Path("examples/fixtures/pursuit_trace_2_evader_9x9.json")


def test_pursuit_trace_fixture_loads_and_round_trips() -> None:
    trace = load_pursuit_trace(FIXTURE_PATH)
    validate_pursuit_trace(trace)
    assert trace.schema_version == SCHEMA_VERSION
    assert trace.trace_type == TRACE_TYPE
    assert trace.summary.capture_rate == 0.5

    round_tripped = pursuit_trace_from_dict(pursuit_trace_to_dict(trace))
    assert round_tripped.summary.final_positions == trace.summary.final_positions
    assert round_tripped.steps[1].captures[0].evader_id == "evader_0"


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("schema_version",), "bad/v1"),
        (("trace_type",), "other_trace"),
        (("steps", 0, "actions", "evader_0"), "jump"),
        (("steps", 0, "agent_positions", "evader_0"), [99, 99]),
        (("steps", 1, "captures", 0, "evader_id"), "evader_99"),
        (("summary", "initial_positions", "evader_0"), [99, 99]),
        (("summary", "final_positions", "evader_0"), [0, 0]),
    ],
)
def test_validate_pursuit_trace_rejects_invalid_fields(path: tuple[object, ...], value: object) -> None:
    raw = pursuit_trace_to_dict(load_pursuit_trace(FIXTURE_PATH))
    target = raw
    for key in path[:-1]:
        target = target[key]  # type: ignore[index]
    target[path[-1]] = value  # type: ignore[index]

    with pytest.raises(ValueError):
        pursuit_trace_from_dict(raw)


def test_validate_pursuit_trace_rejects_invalid_step_reward_keys() -> None:
    raw = pursuit_trace_to_dict(load_pursuit_trace(FIXTURE_PATH))
    del raw["steps"][0]["step_rewards"]["evader_0"]
    with pytest.raises(ValueError):
        pursuit_trace_from_dict(raw)


def test_viewer_render_helpers_consume_trace() -> None:
    trace = load_pursuit_trace(FIXTURE_PATH)
    frame = grid_frame(trace, 2)
    metrics = summary_metrics(trace)
    table = trace_table_rows(trace)
    status = agent_status_rows(trace)

    assert frame[4][6] == "P0 E0"
    assert math.isclose(float(metrics["capture_rate"]), 0.5)
    assert table[1]["captures"] == ["pursuer_0->evader_0"]
    assert {row["agent_id"] for row in status} == {"pursuer_0", "evader_0", "evader_1"}


def test_mutating_round_trip_data_does_not_change_fixture_object() -> None:
    trace = load_pursuit_trace(FIXTURE_PATH)
    raw = pursuit_trace_to_dict(trace)
    mutated = copy.deepcopy(raw)
    mutated["summary"]["per_agent_returns"]["evader_0"] = -999.0
    assert raw["summary"]["per_agent_returns"]["evader_0"] == -9.0
