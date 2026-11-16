"""Run the Gate 2 comparison used to decide paper framing."""

from __future__ import annotations

import argparse
from pathlib import Path

from strategy_games.experiments.gate2 import Gate2Config, run_gate2


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare the v2 method against PPO, diversity-only, and PSRO.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--strategy-iterations", type=int, default=8)
    parser.add_argument("--ppo-total-steps", type=int, default=4096)
    parser.add_argument("--output-dir", type=Path, default=Gate2Config.output_dir)
    args = parser.parse_args()
    result = run_gate2(
        Gate2Config(
            seeds=tuple(args.seeds),
            strategy_iterations=args.strategy_iterations,
            ppo_total_steps=args.ppo_total_steps,
            output_dir=args.output_dir,
        )
    )
    summary = result["summary"]
    print(f"Gate 2 framing: {summary['framing']}")
    print(f"passed={summary['passed']} beats_ppo={summary['beats_ppo']} beats_diversity={summary['beats_diversity_only']}")
    for method in summary["methods"]:
        exploit = method["exploitability"]
        table_goal = method["heldout_table_goal_rate"]
        draw_once_goal = method.get("heldout_draw_once_goal_rate")
        diagnostic = f" heldout_draw_once_goal={draw_once_goal['mean']:.3f}" if draw_once_goal else ""
        print(
            f"  {method['method']}: exploitability={exploit['mean']:.3f} "
            f"[{exploit['ci_low']:.3f}, {exploit['ci_high']:.3f}] "
            f"heldout_table_goal={table_goal['mean']:.3f}{diagnostic}"
        )


if __name__ == "__main__":
    main()
