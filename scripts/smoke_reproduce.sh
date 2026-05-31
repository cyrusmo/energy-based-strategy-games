#!/usr/bin/env bash
set -euo pipefail

if [[ -x ".venv/bin/python" ]]; then
  PYTHON=".venv/bin/python"
else
  PYTHON="python"
fi

echo "Using Python interpreter: ${PYTHON}"

"${PYTHON}" -m pytest
"${PYTHON}" -m ruff check .
"${PYTHON}" scripts/check_public_leaks.py
"${PYTHON}" examples/compare_baselines.py --episodes 2 --no-ppo --output outputs/public/baselines/metrics.json
"${PYTHON}" examples/compute_payoff_matrix.py --episodes-per-opponent 1 --output outputs/public/payoff_matrix/matrix.json
"${PYTHON}" examples/run_ablation_suite.py --seeds 0 --only gaussian_sampler no_world_model
"${PYTHON}" examples/run_multiseed_protocol.py --seeds 0 --episodes 2 --ppo-total-steps 128 --no-ppo
"${PYTHON}" scripts/make_paper_figures.py --no-compute

echo "Smoke reproduction completed successfully."
