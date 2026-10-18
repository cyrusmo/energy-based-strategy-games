"""Run PSRO-lite on a named v2 layout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from strategy_games.envs.layouts_v2 import get_layout_v2
from strategy_games.training.psro import PSROConfig, run_psro


def main() -> None:
    parser = argparse.ArgumentParser(description="Run restricted PSRO on a v2 layout.")
    parser.add_argument("--layout", default="corridor_split")
    parser.add_argument("--iterations", type=int, default=4)
    parser.add_argument("--output", type=Path, default=Path("outputs/public/psro/summary.json"))
    args = parser.parse_args()
    result = run_psro(PSROConfig(env=get_layout_v2(args.layout), iterations=args.iterations))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"meta_nash_value={result['meta_nash_value']:.3f} support={result['support']}")


if __name__ == "__main__":
    main()
