# Energy-Based Strategy Generation with Game-Theoretic Evaluation

This repository is an early-stage research scaffold for separating high-level strategy generation from low-level action execution in multi-agent environments. The initial domain is a small attacker-defender gridworld where a latent strategy embedding proposes behavior, approximate game-theoretic evaluation scores candidate strategies against sampled opponent responses, and a strategy-conditioned policy executes the selected strategy.

```mermaid
flowchart LR
    B["Strategy Buffer"] --> EBM["Energy-Based Strategy Generator"]
    EBM --> S["Langevin Candidate Strategies"]
    S --> Eval["Game-Theoretic Evaluator"]
    Eval --> Select["Select Strongest Strategy"]
    Select --> Policy["Strategy-Conditioned Policy"]
    Policy --> Env["Attacker-Defender Gridworld"]
    Env --> Rollouts["Rollout Data"]
    Rollouts --> B
    Rollouts --> Updates["Policy / EBM / World Model Updates"]
    Updates --> EBM
```

## Generate -> Evaluate -> Execute -> Update

1. **Generate:** sample latent strategy embeddings from an Energy-Based Model (EBM) using Langevin dynamics, with named heuristic strategies included for debugging.
2. **Evaluate:** score each candidate against sampled defender responses using average-case value, worst-case value, robustness, and an approximate exploitability proxy.
3. **Execute:** condition a policy on the selected strategy and roll it out in the gridworld.
4. **Update:** store rollout summaries in a strategy buffer and leave explicit hooks for policy, EBM, and world-model updates.

## Installation

```bash
pip install -e ".[dev]"
```

Python 3.11+ is required. The default scaffold uses PyTorch, NumPy, PyYAML, pytest, and matplotlib only.

Optional external pursuit benchmarks use a narrow PettingZoo Pursuit dependency set:

```bash
pip install -e ".[bench]"
```

The optional pursuit trace viewer uses Streamlit:

```bash
pip install -e ".[dev,demo]"
```

## Quickstart

```bash
pytest
python examples/run_gridworld_baseline.py
python examples/sample_strategies.py
python examples/evaluate_strategy.py
python examples/run_training_loop.py --config configs/gridworld_day3.yaml
python examples/train_ppo_baseline.py --config configs/gridworld_ppo_baseline.yaml
python examples/visualize_rollout.py
python examples/compare_baselines.py
python examples/compute_payoff_matrix.py
python examples/run_benchmarks.py --config configs/benchmarks/debug_suite.yaml
python examples/export_pursuit_trace.py --config configs/demo/custom_2_evader_9x9.yaml
```

Expected outputs are small JSON-like metric dictionaries, sampled strategy shapes and energies, evaluator summaries for named strategies, a short training-loop history, public artifact files under `outputs/public/`, a rollout trace PNG, a PPO-lite baseline metrics file, a baseline metric table, a named-strategy payoff matrix, and a validated pursuit/evasion `trace.json`. The commands are designed to finish quickly on CPU.

To inspect a pursuit trace interactively:

```bash
streamlit run examples/pursuit_trace_viewer.py
```

This viewer is intended for inspecting environment dynamics, scripted policy behavior, and trace-level metrics. It does not demonstrate learned robustness, optimality, or exact game-theoretic guarantees.

## Key Metrics

- **episode_return:** total attacker reward over an episode.
- **win_rate / goal_rate:** fraction of episodes where the attacker reaches the goal.
- **catch_rate:** fraction of episodes where the defender catches the attacker.
- **average_case_value:** mean attacker value against sampled opponent responses.
- **worst_case_value:** lowest attacker value against the sampled opponent set.
- **robustness_score:** conservative proxy combining worst-case value and value variance.
- **exploitability_proxy:** approximate vulnerability to sampled best responses. This is not exact Nash exploitability.
- **strategy_diversity:** mean pairwise distance between latent strategy embeddings.
- **zero-shot transfer:** planned metric for evaluating strategies against held-out maps, goals, opponent policies, or environment perturbations.
- **benchmark summary:** mean/std return, rates, sampled-response metrics where applicable, strategy diversity, and wall-clock time per environment/baseline.
- **PursuitTrace:** versioned JSON artifact for multi-agent pursuit/evasion rollouts, including per-step actions, rewards, captures, active evaders, and summary metrics.

## Current Status

This is a research scaffold and early-stage experimental framework. It does not claim state-of-the-art performance, convergence guarantees, or exact equilibrium computation. The current gridworld and evaluator are intentionally simple so the Generate -> Evaluate -> Execute -> Update loop can be inspected and tested end to end. The Day 2 loop includes lightweight REINFORCE-style policy updates, contrastive EBM updates, and one-step world-model fitting. The Day 3-7 harness adds public logging, rollout visualization, baseline comparison, payoff-matrix evaluation, and a benchmark runner. A first PPO-lite attacker baseline is available for the custom gridworld, but it is not yet a strong policy. PettingZoo Pursuit is available as an optional transfer benchmark, not a replacement for the custom research gridworld.

## Research Roadmap

- Harden the PPO-lite baseline and compare it against the strategy loop across seed sweeps.
- Improve EBM training with better negative sampling and replay schedules.
- Use the learned world model for imaginary strategy evaluation.
- Expand game-theoretic evaluation beyond sampled heuristic responses.
- Add held-out grid layouts for zero-shot transfer and robustness tests.
- Compare against PPO, heuristic policies, Gaussian latent strategies, and PSRO-style strategy populations.
- Add richer visualizations for strategy embeddings, payoff matrices, and rollout traces.
- Use validated pursuit traces as the shared artifact for rollout viewers, notebooks, and reports.
- Promote selected public artifacts into versioned reports once experiments are stable.
- Expand benchmark adapters once the custom gridworld metrics are stable across seed sweeps.

## Citation / Related Work

This scaffold is motivated by ideas from Policy-Space Response Oracles (PSRO), Proximal Policy Optimization (PPO), Diversity Is All You Need (DIAYN), Counterfactual Regret Minimization (CFR), Energy-Based Models (EBMs), Dreamer-style world models, MuZero-style planning with learned dynamics, and multi-agent environment standards such as PettingZoo. The code currently uses these as research context rather than claiming a full implementation of each method.
