#!/usr/bin/env bash
set -euo pipefail

python examples/compare_baselines.py --output outputs/public/baselines/metrics.json
python examples/compute_payoff_matrix.py --output outputs/public/payoff_matrix/matrix.json
python examples/run_ablation_suite.py --seeds 0 1 2 3 4
python examples/run_multiseed_protocol.py --seeds 0 1 2 3 4 --ppo-total-steps 2048
python scripts/make_paper_figures.py

echo "Paper artifacts written under outputs/public/."
