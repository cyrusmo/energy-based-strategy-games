import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

from examples.run_ablation_suite import ABLATIONS, build_config, summarize_rows as summarize_ablation_rows
from examples.run_multiseed_protocol import (
    configure_ppo,
    configure_strategy_loop,
    summarize_by_baseline,
)


def test_smoke_script_has_required_commands_and_is_executable() -> None:
    path = Path("scripts/smoke_reproduce.sh")
    content = path.read_text(encoding="utf-8")

    assert os.access(path, os.X_OK)
    assert content.startswith("#!/usr/bin/env bash\nset -euo pipefail")
    assert '${PYTHON}" -m pytest' in content
    assert '${PYTHON}" -m ruff check .' in content
    assert '${PYTHON}" scripts/check_public_leaks.py' in content
    assert "examples/compare_baselines.py --episodes 2 --no-ppo" in content
    assert "examples/compute_payoff_matrix.py --episodes-per-opponent 1" in content
    assert "examples/run_ablation_suite.py --seeds 0 --only gaussian_sampler no_world_model" in content
    assert "examples/run_multiseed_protocol.py --seeds 0 --episodes 2 --ppo-total-steps 128 --no-ppo" in content
    assert "scripts/make_paper_figures.py --no-compute" in content
    assert "Smoke reproduction completed successfully." in content


def test_ci_runs_pytest_and_ruff() -> None:
    workflow = yaml.safe_load(Path(".github/workflows/ci.yml").read_text(encoding="utf-8"))
    steps = workflow["jobs"]["test"]["steps"]
    commands = "\n".join(str(step.get("run", "")) for step in steps)

    assert "pytest" in commands
    assert "ruff check ." in commands


def test_ablation_config_override_and_summary_schema(tmp_path: Path) -> None:
    base = {
        "training": {"candidate_strategies": 8},
        "updates": {"train_world_model": True},
        "logging": {"enabled": False},
    }
    config = build_config(base, ABLATIONS["gaussian_sampler"], seed=3, output_dir=tmp_path, ablation_name="gaussian")
    summary = summarize_ablation_rows(
        [
            {
                "ablation": "gaussian_sampler",
                "seed": 3,
                "mean_episode_return": 1.0,
                "mean_win_rate": 0.5,
                "buffer_diversity": 0.25,
            }
        ]
    )

    assert config["seed"] == 3
    assert config["sampler"]["type"] == "gaussian"
    assert config["logging"]["enabled"] is True
    assert config["logging"]["run_name"] == "gaussian_seed3"
    assert summary["runs"][0]["ablation"] == "gaussian_sampler"
    assert summary["runs"][0]["mean_episode_return"] == 1.0
    assert "exploitability_proxy is approximate" in summary["notes"][1]


def test_multiseed_config_generation_and_summary_schema(tmp_path: Path) -> None:
    strategy_config = configure_strategy_loop({"logging": {"enabled": False}}, seed=4, output_dir=tmp_path)
    ppo_config = configure_ppo({"ppo": {"total_steps": 1}}, seed=4, total_steps=128, output_dir=tmp_path)
    summary = summarize_by_baseline(
        [
            {
                "seed": 4,
                "baseline": "random_policy",
                "episode_return": -1.0,
                "win_rate": 0.0,
                "goal_rate": 0.0,
                "catch_rate": 1.0,
                "timeout_rate": 0.0,
            }
        ]
    )

    assert strategy_config["seed"] == 4
    assert strategy_config["logging"]["run_name"] == "strategy_seed4"
    assert ppo_config["ppo"]["total_steps"] == 128
    assert ppo_config["logging"]["output_dir"].endswith("ppo_runs/seed4")
    assert summary["baselines"][0]["baseline"] == "random_policy"
    assert summary["baselines"][0]["episode_return_mean"] == -1.0
    assert "episode_return_ci_low" in summary["baselines"][0]


def test_paper_figure_manifest_records_provenance(tmp_path: Path) -> None:
    baseline_path = tmp_path / "baselines.json"
    payoff_path = tmp_path / "payoff.json"
    ablation_path = tmp_path / "ablation.json"
    output_dir = tmp_path / "figures"

    baseline_path.write_text(
        json.dumps(
            [
                {
                    "baseline": "random_policy",
                    "episode_return": -1.0,
                    "win_rate": 0.0,
                    "goal_rate": 0.0,
                    "catch_rate": 1.0,
                    "timeout_rate": 0.0,
                }
            ]
        ),
        encoding="utf-8",
    )
    payoff_path.write_text(
        json.dumps(
            {
                "strategy_labels": ["direct_goal"],
                "opponent_labels": ["aggressive"],
                "average_reward_matrix": [[1.0]],
            }
        ),
        encoding="utf-8",
    )
    ablation_path.write_text(
        json.dumps({"runs": [{"ablation": "gaussian_sampler", "mean_episode_return": -0.5}]}),
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            "scripts/make_paper_figures.py",
            "--output-dir",
            str(output_dir),
            "--baseline-json",
            str(baseline_path),
            "--payoff-json",
            str(payoff_path),
            "--ablation-json",
            str(ablation_path),
            "--no-compute",
        ],
        check=True,
        cwd=Path.cwd(),
        capture_output=True,
        text=True,
    )

    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["generated_at"]
    assert "git_commit" in manifest
    assert manifest["is_smoke"] is True
    assert str(baseline_path) in manifest["source_artifacts"]
    assert "python scripts/make_paper_figures.py" in manifest["commands"][0]


def test_public_leak_scanner_allows_generic_checkpoint_mentions_and_rejects_private_paths(tmp_path: Path) -> None:
    clean = tmp_path / "clean"
    clean.mkdir()
    (clean / "README.md").write_text("Generic .pt checkpoints are ignored by default.\n", encoding="utf-8")
    subprocess.run(
        [sys.executable, "scripts/check_public_leaks.py", str(clean)],
        check=True,
        cwd=Path.cwd(),
        capture_output=True,
        text=True,
    )

    dirty = tmp_path / "dirty"
    dirty.mkdir()
    (dirty / "README.md").write_text("Do not publish outputs/private/checkpoints/model.pt\n", encoding="utf-8")
    completed = subprocess.run(
        [sys.executable, "scripts/check_public_leaks.py", str(dirty)],
        check=False,
        cwd=Path.cwd(),
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 1
    assert "private output path" in completed.stderr
