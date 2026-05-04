"""Run simple gridworld baselines."""

from __future__ import annotations

import json

from strategy_games.training.ppo_baseline import run_direct_goal_baseline, run_random_policy_baseline


def main() -> None:
    results = {
        "random_policy": run_random_policy_baseline(episodes=5, seed=0),
        "direct_goal_heuristic": run_direct_goal_baseline(episodes=5),
    }
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
