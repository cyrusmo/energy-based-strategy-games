"""Run the energy-guided weight-space repair pilot on Gridworld v2.

    python examples/run_energy_repair.py --seeds 20 --workers 4

Writes ``summary.json``, ``summary.md`` and ``seeds.json`` to the output dir.
The exact per-position return table is cached there as ``positions.npz``.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from pathlib import Path

import torch

from strategy_games.experiments.energy_repair import (
    RepairConfig,
    aggregate,
    build_position_table,
    load_position_table,
    run_seed,
    save_position_table,
    summary_markdown,
)

_TABLE = None


def _init(table_path: str) -> None:
    global _TABLE
    torch.set_num_threads(1)
    _TABLE = load_position_table(table_path)


def _run(seed: int) -> dict:
    return run_seed(_TABLE, seed, RepairConfig())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--first-seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output-dir", default="outputs/public/energy_repair")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    table_path = out / "positions.npz"
    if not table_path.exists():
        print("Rolling out every strategy from every position (one-off, a few minutes)...")
        save_position_table(build_position_table(workers=args.workers), str(table_path))

    seeds = list(range(args.first_seed, args.first_seed + args.seeds))
    with mp.get_context("spawn").Pool(args.workers, initializer=_init, initargs=(str(table_path),)) as pool:
        results = pool.map(_run, seeds, chunksize=1)

    summary = aggregate(results, RepairConfig())
    (out / "seeds.json").write_text(json.dumps(results, indent=2))
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    (out / "summary.md").write_text(summary_markdown(summary))
    print(summary_markdown(summary))


if __name__ == "__main__":
    main()
