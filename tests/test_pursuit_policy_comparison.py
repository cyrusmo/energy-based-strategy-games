import csv
import json
import math
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import yaml

from strategy_games.envs.multi_evader_pursuit import MultiEvaderPursuitConfig
from strategy_games.envs.pursuit_actions import ACTIONS
from strategy_games.experiments.pursuit_comparison import (
    CSV_FIELDS,
    PursuitPolicyComparisonConfig,
    compute_pursuit_policy_comparison,
    empirical_game_diagnostics,
    format_policy_comparison_summary,
    payoff_weighted_row_policy_ranking_distribution,
    save_policy_comparison_csv,
    save_policy_comparison_json,
)
from strategy_games.models.pursuit_observation import OBSERVATION_SCHEMA, PursuitObservationSpec
from strategy_games.policies.pursuit_targets import PursuitActorCritic


def tiny_comparison_config(tmp_path: Path, eta: float = 1.0) -> PursuitPolicyComparisonConfig:
    return PursuitPolicyComparisonConfig(
        seeds=(0, 1),
        pursuer_policies=("pursuer_greedy_nearest", "pursuer_random"),
        evader_policies=("evader_flee_nearest", "evader_goal_directed", "evader_random"),
        eta=eta,
        feint_steps=1,
        env=MultiEvaderPursuitConfig(
            grid_size=(5, 5),
            num_evaders=2,
            num_pursuers=1,
            max_steps=4,
            pursuer_starts=((2, 2),),
            evader_starts=((2, 4), (0, 4)),
            evader_goals=((4, 4), (4, 0)),
        ),
        output_json=tmp_path / "policy_comparison.json",
        output_csv=tmp_path / "policy_comparison.csv",
        created_at="2026-05-19T00:00:00+00:00",
        git_commit=None,
    )


def test_pursuit_policy_comparison_artifact_schema(tmp_path: Path) -> None:
    config = tiny_comparison_config(tmp_path)
    result = compute_pursuit_policy_comparison(config)

    assert result["schema_version"] == "pursuit_policy_comparison/v1"
    assert result["env_id"] == "multi_evader_pursuit_v1"
    assert result["metadata"]["created_at"] == "2026-05-19T00:00:00+00:00"
    assert result["metadata"]["git_commit"] is None or isinstance(result["metadata"]["git_commit"], str)
    assert result["config"]["grid_size"] == [5, 5]
    assert result["config"]["num_evaders"] == 2
    assert result["methodology"]["ranking_distribution_params"]["eta"] == 1.0
    assert result["methodology"]["is_zero_sum"] is False
    assert result["methodology"]["is_equilibrium_solver"] is False

    payoff = result["payoff_matrix"]
    pursuers = result["pursuer_policies"]
    evaders = result["evader_policies"]
    assert len(payoff) == len(pursuers)
    assert all(len(row) == len(evaders) for row in payoff)
    assert payoff == result["metrics"]["mean_pursuer_return"]
    assert result["cell_sample_count"] == [[2, 2, 2], [2, 2, 2]]
    assert result["num_episodes_per_cell"] == 2


def test_empirical_game_regret_maximin_and_distribution_are_valid(tmp_path: Path) -> None:
    result = compute_pursuit_policy_comparison(tiny_comparison_config(tmp_path))
    empirical = result["empirical_game"]

    regrets = empirical["empirical_regret_vs_uniform_column_mixture"]
    assert all(math.isfinite(float(value)) and float(value) >= 0.0 for value in regrets)
    assert min(float(value) for value in regrets) == pytest.approx(0.0)
    assert empirical["maximin_policy"] in result["pursuer_policies"]

    distribution = empirical["payoff_weighted_row_policy_ranking_distribution"]
    probabilities = distribution["probabilities"]
    assert distribution["eta"] == 1.0
    assert set(probabilities) == set(result["pursuer_policies"])
    assert all(math.isfinite(float(value)) and float(value) >= 0.0 for value in probabilities.values())
    assert sum(float(value) for value in probabilities.values()) == pytest.approx(1.0)


def test_eta_changes_ranking_distribution_sharpness() -> None:
    labels = ("pursuer_low", "pursuer_high")
    low_eta = payoff_weighted_row_policy_ranking_distribution([0.0, 2.0], labels, eta=0.1)
    high_eta = payoff_weighted_row_policy_ranking_distribution([0.0, 2.0], labels, eta=3.0)

    assert high_eta["probabilities"]["pursuer_high"] > low_eta["probabilities"]["pursuer_high"]
    assert sum(high_eta["probabilities"].values()) == pytest.approx(1.0)


def test_empirical_game_rejects_bad_shapes() -> None:
    with pytest.raises(ValueError):
        empirical_game_diagnostics([[1.0, 2.0]], ("row_a", "row_b"))


def test_policy_comparison_json_csv_and_summary_outputs(tmp_path: Path) -> None:
    result = compute_pursuit_policy_comparison(tiny_comparison_config(tmp_path))
    json_path = save_policy_comparison_json(result, tmp_path / "policy_comparison.json")
    csv_path = save_policy_comparison_csv(result, tmp_path / "policy_comparison.csv")
    summary = format_policy_comparison_summary(result)

    loaded = json.loads(json_path.read_text(encoding="utf-8"))
    assert loaded["schema_version"] == result["schema_version"]
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == len(result["pursuer_policies"]) * len(result["evader_policies"])
    assert tuple(rows[0]) == CSV_FIELDS
    assert "maximin_policy=" in summary


def test_policy_comparison_is_deterministic_with_fixed_metadata(tmp_path: Path) -> None:
    config = tiny_comparison_config(tmp_path)
    first = compute_pursuit_policy_comparison(config)
    second = compute_pursuit_policy_comparison(config)
    assert first == second


def test_policy_comparison_can_include_explicit_learned_pursuer_row(tmp_path: Path) -> None:
    checkpoint_path = _write_dummy_pursuer_checkpoint(tmp_path / "ppo_pursuer.pt")
    config = PursuitPolicyComparisonConfig(
        seeds=(0,),
        pursuer_policies=("pursuer_greedy_nearest",),
        evader_policies=("evader_flee_nearest",),
        env=MultiEvaderPursuitConfig(
            grid_size=(5, 5),
            num_evaders=2,
            num_pursuers=1,
            max_steps=3,
            pursuer_starts=((2, 2),),
            evader_starts=((2, 4), (0, 4)),
            evader_goals=((4, 4), (4, 0)),
        ),
        created_at="2026-05-19T00:00:00+00:00",
        git_commit=None,
        learned_pursuer_checkpoint=checkpoint_path,
    )

    result = compute_pursuit_policy_comparison(config)

    assert result["pursuer_policies"] == ["pursuer_greedy_nearest", "dummy_ppo_pursuer"]
    assert len(result["payoff_matrix"]) == 2
    assert all(len(row) == 1 for row in result["payoff_matrix"])
    learned_targets = [
        target
        for target in result["pursuer_policy_targets"]
        if target["policy_type"] == "learned"
    ]
    assert len(learned_targets) == 1
    assert learned_targets[0]["policy_id"] == "dummy_ppo_pursuer"
    assert "checkpoint_path" not in learned_targets[0]
    assert learned_targets[0]["metadata"]["observation_schema"] == OBSERVATION_SCHEMA


def test_compare_pursuit_policies_cli_writes_artifacts(tmp_path: Path) -> None:
    config_path = tmp_path / "comparison.yaml"
    output_dir = tmp_path / "pursuit_demo"
    config = {
        "seeds": [0],
        "metadata": {"created_at": "2026-05-19T00:00:00+00:00"},
        "env": {
            "grid_size": [5, 5],
            "num_evaders": 2,
            "num_pursuers": 1,
            "max_steps": 3,
            "pursuer_starts": [[2, 2]],
            "evader_starts": [[2, 4], [0, 4]],
            "evader_goals": [[4, 4], [4, 0]],
        },
        "policies": {
            "pursuers": ["pursuer_greedy_nearest", "pursuer_random"],
            "evaders": ["evader_flee_nearest", "evader_random"],
        },
        "empirical_game": {"eta": 0.5},
        "output": {"dir": str(output_dir)},
    }
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            "examples/compare_pursuit_policies.py",
            "--config",
            str(config_path),
        ],
        check=True,
        cwd=Path.cwd(),
        capture_output=True,
        text=True,
    )

    assert "maximin_policy=" in completed.stdout
    assert (output_dir / "policy_comparison.json").exists()
    assert (output_dir / "policy_comparison.csv").exists()


def _write_dummy_pursuer_checkpoint(path: Path) -> Path:
    spec = PursuitObservationSpec(role="pursuer", max_pursuers=1, max_evaders=2)
    model = PursuitActorCritic(obs_dim=spec.obs_dim, action_dim=len(ACTIONS), hidden_dim=16)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "observation_schema": OBSERVATION_SCHEMA,
            "action_space": list(ACTIONS),
            "max_pursuers": spec.max_pursuers,
            "max_evaders": spec.max_evaders,
            "obs_dim": spec.obs_dim,
            "hidden_dim": 16,
            "policy_id": "dummy_ppo_pursuer",
            "training_run_id": "dummy_run",
            "training_scope": {
                "controlled_agent_id": "pursuer_0",
                "trained_role": "pursuer",
                "opponent_policy": "scripted",
                "self_play": False,
            },
        },
        path,
    )
    return path
