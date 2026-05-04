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

Python 3.11+ is required. The first scaffold uses PyTorch, NumPy, PyYAML, pytest, and matplotlib only.

## Quickstart

```bash
pytest
python examples/run_gridworld_baseline.py
python examples/sample_strategies.py
python examples/evaluate_strategy.py
```

Expected outputs are small JSON-like metric dictionaries, sampled strategy shapes and energies, and evaluator summaries for named strategies. The commands are designed to finish quickly on CPU.

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

## What Is Public vs Private?

Public surfaces should include clean architecture, reproducible toy experiments, core algorithmic scaffolding, public experiment results, demo notebooks, and roadmap-level research claims.

Private surfaces should include messy scratch notebooks, interview-specific positioning, unvalidated claims, large raw rollout dumps, proprietary datasets, compute logs, failed experiment archaeology, and any future non-public partner data.

The `.gitignore` protects `notebooks/private/`, `experiments/private/`, `data/private/`, `reports/private/`, `.env`, `private_config/`, private outputs, model checkpoints, and common experiment tracking directories.

## Current Status

This is a research scaffold and early-stage experimental framework. It does not claim state-of-the-art performance, convergence guarantees, or exact equilibrium computation. The current gridworld and evaluator are intentionally simple so the Generate -> Evaluate -> Execute -> Update loop can be inspected and tested end to end.

## Research Roadmap

- Implement real policy optimization for the strategy-conditioned policy.
- Train the EBM from high-performing strategy buffer samples.
- Add a learned world model for imaginary strategy evaluation.
- Expand game-theoretic evaluation beyond sampled heuristic responses.
- Add held-out grid layouts for zero-shot transfer and robustness tests.
- Compare against PPO, heuristic policies, Gaussian latent strategies, and PSRO-style strategy populations.
- Add richer visualizations for strategy embeddings, payoff matrices, and rollout traces.

## Citation / Related Work

This scaffold is motivated by ideas from Policy-Space Response Oracles (PSRO), Proximal Policy Optimization (PPO), Diversity Is All You Need (DIAYN), Counterfactual Regret Minimization (CFR), Energy-Based Models (EBMs), Dreamer-style world models, and MuZero-style planning with learned dynamics. The code currently uses these as research context rather than claiming a full implementation of each method.
