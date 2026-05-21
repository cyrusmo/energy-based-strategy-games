import copy
import json
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
from strategy_games.viewers import (
    COMPARISON_GENERATION_COMMAND,
    agent_status_rows,
    clamp_frame_index,
    comparison_diagnostics_summary,
    evader_effectiveness_summary,
    grid_frame,
    load_policy_comparison_state,
    pursuer_effectiveness_summary,
    step_frame_index,
    styled_grid_frame,
    summary_metrics,
    trace_metadata,
    trace_table_rows,
    transition_context,
)


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


def test_frame_navigation_helpers_clamp_to_trace_bounds() -> None:
    trace = load_pursuit_trace(FIXTURE_PATH)

    assert clamp_frame_index(trace, -5) == 0
    assert clamp_frame_index(trace, 99) == len(trace.steps)
    assert step_frame_index(trace, 0, -1) == 0
    assert step_frame_index(trace, 1, 1) == 2
    assert step_frame_index(trace, 2, 1) == 2


def test_trace_metadata_and_role_summaries_are_explicit() -> None:
    trace = load_pursuit_trace(FIXTURE_PATH)
    metadata = trace_metadata(trace)
    pursuer = pursuer_effectiveness_summary(trace)
    evader = evader_effectiveness_summary(trace)

    assert metadata["env_id"] == "custom_multi_evader_pursuit_v0"
    assert metadata["episode_id"] == "fixture_2_evader_capture_timeout"
    assert metadata["seed"] == 101
    assert metadata["grid_size"] == [9, 9]
    assert metadata["num_evaders"] == 2
    assert metadata["num_pursuers"] == 1
    assert metadata["max_steps"] == 2
    assert metadata["catch_radius"] == 0
    assert metadata["outcome"] == "timeout"
    assert metadata["terminated_reason"] == "timeout"
    assert pursuer["capture_rate"] == pytest.approx(0.5)
    assert pursuer["captured_evaders"] == 1
    assert "pursuer" in str(pursuer["interpretation"])
    assert evader["survival_rate"] == pytest.approx(0.5)
    assert evader["surviving_evaders"] == 1
    assert "evader" in str(evader["interpretation"])


def test_transition_context_and_styled_grid_capture_state() -> None:
    trace = load_pursuit_trace(FIXTURE_PATH)

    initial = transition_context(trace, 0)
    assert initial["t"] == "initial"
    assert initial["actions"] == {}
    assert initial["active_evaders"] == ["evader_0", "evader_1"]

    capture_context = transition_context(trace, 2)
    assert capture_context["captures"] == ["pursuer_0->evader_0"]
    assert capture_context["done"] is True
    assert capture_context["active_evaders"] == ["evader_1"]

    cells = styled_grid_frame(trace, 2)
    capture_cell = cells[4][6]
    assert capture_cell["capture_highlight"] is True
    assert capture_cell["role"] == "mixed"
    statuses = {agent["agent_id"]: agent["status"] for agent in capture_cell["agents"]}
    assert statuses["evader_0"] == "captured"
    assert statuses["pursuer_0"] == "active"


def test_comparison_artifact_load_states(tmp_path: Path) -> None:
    missing = load_policy_comparison_state(tmp_path / "missing.json")
    assert missing.comparison_artifact is None
    assert missing.comparison_path == str(tmp_path / "missing.json")
    assert COMPARISON_GENERATION_COMMAND in str(missing.comparison_error)

    malformed_path = tmp_path / "malformed.json"
    malformed_path.write_text("{not json", encoding="utf-8")
    malformed = load_policy_comparison_state(malformed_path)
    assert malformed.comparison_artifact is None
    assert "Could not load comparison artifact" in str(malformed.comparison_error)

    valid_path = tmp_path / "policy_comparison.json"
    valid_path.write_text(json.dumps(_minimal_comparison_artifact()), encoding="utf-8")
    loaded = load_policy_comparison_state(valid_path)
    summary = comparison_diagnostics_summary(loaded)
    assert loaded.comparison_path == str(valid_path)
    assert loaded.comparison_error is None
    assert summary["loaded"] is True
    assert summary["source_path"] == str(valid_path)
    assert summary["payoff_orientation"] == "rows=pursuer_policies, columns=evader_policies"
    assert summary["payoff_metric"] == "mean_pursuer_return"


def test_mutating_round_trip_data_does_not_change_fixture_object() -> None:
    trace = load_pursuit_trace(FIXTURE_PATH)
    raw = pursuit_trace_to_dict(trace)
    mutated = copy.deepcopy(raw)
    mutated["summary"]["per_agent_returns"]["evader_0"] = -999.0
    assert raw["summary"]["per_agent_returns"]["evader_0"] == -9.0


def _minimal_comparison_artifact() -> dict[str, object]:
    return {
        "schema_version": "pursuit_policy_comparison/v1",
        "payoff_orientation": "rows=pursuer_policies, columns=evader_policies",
        "payoff_metric": "mean_pursuer_return",
        "pursuer_policies": ["pursuer_greedy_nearest"],
        "evader_policies": ["evader_feint"],
        "methodology": {"row_player": "pursuer", "column_player": "evader"},
        "metrics": {"mean_pursuer_return": [[1.0]]},
        "empirical_game": {
            "row_payoff_vs_uniform_column_mixture": [1.0],
            "empirical_regret_vs_uniform_column_mixture": [0.0],
            "maximin_policy": "pursuer_greedy_nearest",
            "payoff_weighted_row_policy_ranking_distribution": {
                "probabilities": {"pursuer_greedy_nearest": 1.0}
            },
            "notes": "diagnostic only",
        },
    }
