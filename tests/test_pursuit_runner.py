import json
import subprocess
import sys
from pathlib import Path

import yaml

from strategy_games.envs.multi_evader_pursuit import MultiEvaderPursuitConfig, MultiEvaderPursuitEnv
from strategy_games.policies.scripted_pursuit import scripted_pursuit_actions
from strategy_games.rollouts import PursuitRolloutConfig, run_scripted_pursuit_rollout
from strategy_games.traces import load_pursuit_trace, validate_pursuit_trace


def test_two_evader_rollout_produces_valid_trace() -> None:
    trace = run_scripted_pursuit_rollout(
        PursuitRolloutConfig(
            seed=2,
            created_at="2026-05-18T00:00:00+00:00",
            env=MultiEvaderPursuitConfig(max_steps=4),
        )
    )
    validate_pursuit_trace(trace)
    assert trace.num_evaders == 2
    assert trace.summary.total_steps <= 4
    assert set(trace.summary.per_evader_status) == {"evader_0", "evader_1"}


def test_n_evader_rollout_supports_more_than_two_evaders() -> None:
    trace = run_scripted_pursuit_rollout(
        PursuitRolloutConfig(
            seed=3,
            created_at="2026-05-18T00:00:00+00:00",
            env=MultiEvaderPursuitConfig(num_evaders=4, num_pursuers=2, max_steps=3),
        )
    )
    validate_pursuit_trace(trace)
    assert trace.num_evaders == 4
    assert trace.num_pursuers == 2


def test_scripted_policies_differ_on_controlled_state() -> None:
    env = MultiEvaderPursuitEnv(
        MultiEvaderPursuitConfig(
            grid_size=(9, 9),
            num_evaders=1,
            pursuer_starts=((4, 4),),
            evader_starts=((4, 6),),
            evader_goals=((4, 4),),
        )
    )
    flee_actions = scripted_pursuit_actions(env, evader_policy="evader_flee_nearest")
    goal_actions = scripted_pursuit_actions(env, evader_policy="evader_goal_directed")
    assert flee_actions["evader_0"] in {"right", "up", "down"}
    assert goal_actions["evader_0"] == "left"
    assert flee_actions["evader_0"] != goal_actions["evader_0"]


def test_cli_exporter_writes_trace_and_summary(tmp_path) -> None:
    config_path = tmp_path / "trace_config.yaml"
    output_dir = tmp_path / "pursuit_demo"
    config = {
        "seed": 5,
        "episode_id": "test_cli_trace",
        "created_at": "2026-05-18T00:00:00+00:00",
        "env": {
            "grid_size": [5, 5],
            "num_evaders": 2,
            "num_pursuers": 1,
            "max_steps": 3,
            "pursuer_starts": [[2, 2]],
            "evader_starts": [[2, 4], [0, 4]],
            "evader_goals": [[4, 4], [4, 0]],
        },
        "policies": {"pursuer": "pursuer_greedy_nearest", "evader": "evader_feint", "feint_steps": 1},
        "output": {"dir": str(output_dir), "trace_filename": "trace.json", "summary_filename": "summary.json"},
    }
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    subprocess.run(
        [
            sys.executable,
            "examples/export_pursuit_trace.py",
            "--config",
            str(config_path),
            "--no-plot",
        ],
        check=True,
        cwd=Path.cwd(),
        capture_output=True,
        text=True,
    )

    trace_path = output_dir / "trace.json"
    summary_path = output_dir / "summary.json"
    assert trace_path.exists()
    assert summary_path.exists()
    validate_pursuit_trace(load_pursuit_trace(trace_path))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert "capture_rate" in summary
