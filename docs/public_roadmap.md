# Public Roadmap

This repository is currently a research scaffold for early experiments in energy-based strategy generation and game-theoretic evaluation.

## Current

- Configurable attacker-defender gridworld
- Named heuristic strategy embeddings
- Energy MLP over strategy embeddings
- Langevin sampler
- Approximate sampled-response evaluator
- Strategy buffer
- Debug training loop with lightweight policy, EBM, and world-model updates
- Heuristic-aligned strategy execution (evaluator and executor share an action source)
- Public experiment logging
- Rollout trace and path visualization
- Baseline comparison harness
- Named-strategy payoff matrix
- Benchmark registry and runner for custom gridworld seed sweeps
- Validated pursuit/evasion trace schema and custom multi-evader rollout viewer
- Optional PettingZoo Pursuit adapter for transfer benchmarking
- PPO baseline with potential-based shaping, longer default budget, and training diagnosis
- Gridworld learning-study protocol with multi-seed curves and triage
- Held-out grid layouts that expose direct-goal overfitting
- Unit and smoke tests

## Near Term

- Interpret learning-study and held-out artifacts before expanding the method stack
- Improve EBM training from buffer positives and sampler negatives if selected-strategy diversity collapses
- Add richer rollout logging and visualization summaries
- Add more pursuit trace fixtures and curated public demos
- Add benchmark reports for seed sweeps without overclaiming performance
- Add simple payoff-matrix experiments

## Medium Term

- Improve approximate best-response search
- Add learned world-model rollouts
- Add zero-shot transfer benchmarks beyond layout swaps
- Add PSRO-style population comparisons
- Add more interpretable strategy embedding visualizations
- Compare strategy-conditioned PPO against unconditioned PPO

## Long Term

- Move beyond toy gridworlds into richer multi-agent domains
- Study how energy-based latent strategy spaces interact with equilibrium-seeking evaluation
- Compare model-free, model-based, and hybrid strategy evaluation pipelines

This roadmap is intentionally conservative. Results should be reported only after reproducible experiments are implemented and reviewed. Medium-term items stay deferred until a learning method beats or clearly challenges direct-goal on held-out layouts.
